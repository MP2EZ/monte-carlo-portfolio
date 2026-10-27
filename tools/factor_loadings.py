#!/usr/bin/env python3
"""
Factor loadings for the ticker universe — compute, inspect, compare.
====================================================================

    python3 tools/factor_loadings.py              # the whole universe
    python3 tools/factor_loadings.py VOO AVLV VT  # just these
    python3 tools/factor_loadings.py --compare    # + what mu WOULD become

Standalone on purpose. `data_collection_regime_model.py` writes these into the
JSON on its next run, but that regenerates every parameter from current market
data — so this exists to look at the loadings without moving a single number
the model currently uses.

`--compare` shows the factor build-up beside today's historical mean. The engine reads the
artifact under MC_MU_METHOD=factor, the default.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings('ignore')

import pandas as pd                                             # noqa: E402

from montecarlo import factors                                                  # noqa: E402
from montecarlo.loaders import MODEL_TICKERS, SINGLE_STOCKS                # noqa: E402


def factor_frames(kf):
    """region -> DataFrame, indexed by month-start, factor columns only."""
    us = kf.parse_ff5_factors().join(kf.parse_momentum_factor(), how='inner')
    out = {'us': us, 'dev': kf.parse_developed_ex_us_factors(),
           'emg': kf.parse_emerging_factors()}
    for r, df in out.items():
        df = df.dropna()
        df.index = df.index.to_period('M').to_timestamp()
        out[r] = df
    return out


def fund_costs(ticker):
    """(expense ratio, annual turnover) from yfinance. (None, None) for a stock.

    Turnover drives the implementation estimate — see factors.ROUND_TRIP. It is
    what showed COST_ADJUSTMENTS' `drag` to be 3-40x too large for the Avantis
    funds, which turn over 4-7% a year.
    """
    import yfinance as yf
    try:
        ops = yf.Ticker(ticker).funds_data.fund_operations
        return (float(ops.loc['Annual Report Expense Ratio', ticker]),
                float(ops.loc['Annual Holdings Turnover', ticker]))
    except Exception:
        try:
            v = (yf.Ticker(ticker).info or {}).get('netExpenseRatio')
            return (float(v) / 100.0 if v is not None else None), None
        except Exception:
            return None, None


def fit_ticker(ticker, frames, dc):
    """Loadings from the ACTUAL fund's history — never the spliced series ."""
    try:
        px = dc.download_with_retry(ticker, start='1990-01-01')
    except Exception as e:
        return None, f'download failed: {e.__class__.__name__}'
    r = px.resample('ME').last().pct_change().dropna()
    r.index = r.index.to_period('M').to_timestamp()

    # A single stock gets its US loadings on its own 20 years (ledger entry N10, unpublished).
    if ticker in SINGLE_STOCKS:
        fit = factors.fit_single_stock(r, frames['us'])
        return fit, (None if fit else 'too little overlap with the US factors')

    # Align on the months every chosen region can cover. The regional files
    # start at different dates (US 1963-07, developed 1990-11, emerging
    # 1992-07), so an inner join across all three would throw away three
    # decades of US history. Fit each region on its own overlap instead.
    per_region = {}
    for reg, df in frames.items():
        j = pd.concat([r.rename('y'), df], axis=1).dropna()
        if len(j) >= factors.MIN_MONTHS_ANY:
            per_region[reg] = j
    if not per_region:
        return None, 'no overlap with any factor set'

    best, fit = None, None
    for reg, j in per_region.items():
        f = factors.fit_loadings(j['y'] - j['rf'], {reg: j}, regions=[reg])
        if f and (best is None or f['adj_r2'] > best):
            best, fit, chosen = f['adj_r2'], f, reg

    # Multi-region only where every candidate region overlaps the same months.
    common = None
    for reg, j in per_region.items():
        common = j.index if common is None else common.intersection(j.index)
    if common is not None and len(common) >= factors.MIN_MONTHS_MULTI:
        aligned = {reg: frames[reg].loc[common] for reg in per_region}
        y = r.loc[common] - frames[chosen].loc[common]['rf']
        multi = factors.fit_loadings(y, aligned, regions=list(per_region))
        if multi and multi['multi_region'] and multi['adj_r2'] > fit['adj_r2']:
            fit = multi
    return fit, None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('tickers', nargs='*', help='default: the whole universe')
    ap.add_argument('--compare', action='store_true',
                    help='show what the build-up produces vs the current mean')
    ap.add_argument('--json', metavar='PATH', help='write the fits to a file')
    args = ap.parse_args(argv)

    import data_collection_regime_model as dc
    kf = dc.KenFrenchParser(str(ROOT / 'reference'))
    frames = factor_frames(kf)
    tickers = args.tickers or list(MODEL_TICKERS)

    cur = {}
    jf = ROOT / 'output' / 'ticker_parameters_regimes_v8.json'
    if jf.exists():
        cur = json.loads(jf.read_text())

    head = (f"{'tkr':<7}{'method':<14}{'regions':<14}{'adjR2':>7}{'n':>5}"
            f"{'alpha':>8}{'SE':>7}")
    if args.compare:
        head += f"{'factor mu':>11}{'current':>9}{'delta':>8}"
    print(head)
    print('-' * (len(head) + 2))

    from montecarlo.loaders import _BOND_FUNDS
    rf_series = frames['us']['rf']

    fits = {}
    for t in tickers:
        fit, err = fit_ticker(t, frames, dc)
        if fit is None:
            print(f"{t:<7}{'—':<14}{err}")
            continue
        # Below the gate a BOND fund gets a spread over contemporaneous cash
        # rather than a historical mean; anything else keeps its mean.
        # COST is the case that makes the distinction necessary — below the gate
        # for a completely different reason, and not a bond.
        if fit['method'] == 'below_gate' and t in _BOND_FUNDS:
            try:
                px = dc.download_with_retry(t, start='1990-01-01')
                r = px.resample('ME').last().pct_change().dropna()
                r.index = r.index.to_period('M').to_timestamp()
                j = pd.concat([r.rename('y'), rf_series.rename('rf')],
                              axis=1).dropna()
                fit = {'method': 'spread',
                       'spread': factors.bond_spread(j['y'], j['rf']),
                       'n_months': int(len(j)), 'adj_r2': fit['adj_r2'],
                       'regions': [], 'alpha': None, 'alpha_se': None}
            except Exception:
                fit['method'] = 'historical'
        elif fit['method'] == 'below_gate':
            fit['method'] = 'historical'
        fit['expense'], fit['turnover'] = fund_costs(t)
        fits[t] = fit
        a = (f"{fit['alpha'] * 100:>7.2f}%" if fit.get('alpha') is not None
             else f"{'—':>8}")
        ase = (f"{fit['alpha_se'] * 100:>6.2f}%" if fit.get('alpha_se') is not None
               else f"{'—':>7}")
        row = (f"{t:<7}{fit['method']:<14}{'+'.join(fit['regions']):<14}"
               f"{fit['adj_r2']:>7.3f}{fit['n_months']:>5}{a}{ase}")
        if args.compare:
            expense, impl = factors.costs_for(t, fit)
            if fit['method'] == 'factor':
                mu = factors.expected_return(
                    fit, costs=(expense or 0.0, impl))
            elif fit['method'] == 'spread':
                mu = factors.RF_FORWARD + fit['spread']
            elif fit['method'] == 'single_stock':
                mu = factors.single_stock_return(fit)
            else:
                mu = None
            now = cur.get(t, {}).get('expected_return')
            row += f"{mu * 100:>10.2f}%" if mu is not None else f"{'—':>11}"
            if now is not None:
                row += f"{now * 100:>8.2f}%"
                if mu is not None:
                    row += f"{(mu - now) * 100:>+7.2f}"
            if fit['method'] == 'factor' and expense is None:
                row += '  ⚠ no expense'
        print(row)

    by = {}
    for f in fits.values():
        by[f['method']] = by.get(f['method'], 0) + 1
    print(f"\n  " + ' · '.join(f'{v} {k}' for k, v in sorted(by.items())))
    no_exp = [t for t, f in fits.items()
              if f['method'] == 'factor' and factors.costs_for(t, f)[0] is None]
    if no_exp:
        print(f"  ⚠ no expense ratio for {len(no_exp)}: {', '.join(sorted(no_exp))}")
        print(f"    treated as free, which OVERSTATES them. Set factors.EXPENSE.")
    if args.json:
        # MERGED into an existing file: refitting one ticker must not drop the
        # other 25 from the artifact the engine reads.
        out = Path(args.json)
        old = json.loads(out.read_text()) if out.exists() else {}
        out.write_text(json.dumps({**old, **fits}, indent=2) + '\n')
        print(f"  wrote {', '.join(fits)} into {args.json}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
