#!/usr/bin/env python3
"""
Factor premia — measure each one on its OWN window, not on the intersection.
============================================================================

    python3 tools/premia.py              # the table to paste into factors.py
    python3 tools/premia.py --long       # ...with mkt_rf/hml extended to 1926

Premia used to be measured on the months where EVERY factor in a region was
available, which is a silent and expensive choice. US momentum has 1195 months
on disk and only 757 were used; emerging mkt_rf/smb/hml have 445 and used 409,
because emerging CMA does not start until 1992. Nothing said so.

WHY 1926 IS NOT THE DEFAULT. Ken French publishes mkt_rf, smb and hml back to
1926-07, and the roadmap expected that to cut the market premium's standard
error from 1.95pp to ~1.55pp. It does not — measured, it goes to 1.84pp, a 6%
gain rather than 20%, because pre-1963 volatility is 22.5%/yr against 15.5%
after and the extra history carries more variance per observation. What it DOES
do is move the level: mkt_rf 7.19% -> 8.33%, hml 3.59% -> 4.26%.

That is a bad trade here, for a reason specific to this model: every second
moment it uses comes from a modern window — base volatility from 2010+,
correlations from the last 15 years, regimes from a 12-month rolling estimate.
Pairing a premium earned in a 22.5%-volatility era, under a risk-free rate that
was administratively pegged at 1.35%, with modern-volatility machinery is
internally inconsistent. `--long` / MC_PREMIA_LONG_WINDOW=1 exists so the
position is one flag away rather than an argument.

SMB CANNOT BE EXTENDED AT ALL, which the roadmap assumed it could. The 3-factor
SMB sorts on B/M only; the 5-factor SMB averages across B/M, OP and INV sorts.
They correlate 0.98 but differ by 4.5pp/yr in mean absolute terms and by 0.48pp
in the mean. Loadings here are fit against the 5-factor definition, so taking
the premium from the 3-factor series would pair a beta with a different factor.
mkt_rf and hml are literally the same series on the overlap (max |diff| 0.0000
and 0.0008), which is why those two are extendable and size is not.
"""

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings('ignore')

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402

FACTORS = ('mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom')
# Only these two are the same series in the 3-factor and 5-factor files.
EXTENDABLE = ('mkt_rf', 'hml')


def series_by_region(kf, long_window=False):
    """region -> {factor: series}, each on its own longest usable window."""
    f5 = kf.parse_ff5_factors()
    mom = kf.parse_momentum_factor()
    us = {f: f5[f] for f in ('mkt_rf', 'smb', 'hml', 'rmw', 'cma')}
    us['mom'] = mom.iloc[:, 0]
    if long_window:
        f3 = kf.parse_ff3_factors()
        for f in EXTENDABLE:
            us[f] = f3[f]
    out = {'us': us}
    for region, frame in (('dev', kf.parse_developed_ex_us_factors()),
                          ('emg', kf.parse_emerging_factors())):
        out[region] = {f: frame[f] for f in FACTORS if f in frame.columns}
    return out


def measure(series):
    v = series.dropna()
    if len(v) < 24:
        return None
    return {'premium': float(v.mean() * 12),
            'se': float(v.std(ddof=1) / np.sqrt(len(v)) * 12),
            'months': int(len(v)),
            'start': f"{v.index[0]:%Y-%m}", 'end': f"{v.index[-1]:%Y-%m}"}


def recent_lag(regions, start, end):
    """How far each region's MARKET premium trailed the US's over [start, end].

    A gap, measured on the months all three share, with the standard error of
    the mean monthly difference — the regions co-move ~0.75, so the gap is
    better pinned than either level, but over two decades still not well.
    """
    us = regions['us']['mkt_rf'].dropna()
    out = {}
    for region in ('dev', 'emg'):
        s = regions[region]['mkt_rf'].dropna()
        o = us.index.intersection(s.index)
        o = o[(o >= start) & (o <= pd.Period(end, 'M').to_timestamp(how='end'))]
        diff = us.loc[o] - s.loc[o]
        out[region] = {'premium': float(s.loc[o].mean() * 12),
                       'us': float(us.loc[o].mean() * 12),
                       'lag': float(diff.mean() * 12),
                       'se': float(diff.std(ddof=1) / np.sqrt(len(o)) * 12),
                       'months': int(len(o)),
                       'start': f"{o[0]:%Y-%m}", 'end': f"{o[-1]:%Y-%m}"}
    return out


def error_correlation(regions, keys):
    """Correlation between the ESTIMATION ERRORS of the premia in `keys`.

    Two premia measured as sample means over windows A and B err together only
    through the months they share:

        corr = n_O * cov_O(a, b) / (sqrt(n_A * n_B) * sd_A(a) * sd_B(b))

    — the month-to-month correlation on the overlap, scaled by how much of each
    window the overlap is, with each series keeping the full-window spread its
    standard error was measured on. The scaling matters: the US and developed
    market premia move together ~0.75 month to month, but only 433 of the US
    premium's 757 months are shared, so their ERRORS correlate ~0.57.
    """
    s = {k: regions[k[0]][k[1]].dropna() for k in keys}
    n = len(keys)
    out = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = s[keys[i]], s[keys[j]]
            o = a.index.intersection(b.index)
            if len(o) < 2:
                continue
            cov = float(np.cov(a.loc[o], b.loc[o], ddof=1)[0, 1])
            out[i, j] = out[j, i] = len(o) * cov / (
                np.sqrt(len(a) * len(b)) * a.std(ddof=1) * b.std(ddof=1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--long', action='store_true',
                    help='extend mkt_rf and hml to 1926 (see the docstring)')
    ap.add_argument('--since', metavar='YYYY-MM',
                    help="print only each region's market-premium gap to the US "
                         "from this month: the RECENT_LAG table")
    ap.add_argument('--until', metavar='YYYY-MM', default='2100-12',
                    help='last month for --since (default: the latest on disk)')
    args = ap.parse_args(argv)

    import data_collection_regime_model as dc
    from montecarlo import factors
    kf = dc.KenFrenchParser(str(ROOT / 'reference'))
    regions = series_by_region(kf, long_window=args.long)

    if args.since:
        lag = recent_lag(regions, args.since, args.until)
        print(f"\n  {'region':<7}{'window':>18}{'months':>8}{'region':>9}"
              f"{'US':>8}{'gap':>8}{'SE':>7}   full-window premium")
        print('  ' + '-' * 80)
        for region, m in lag.items():
            print(f"  {region:<7}{m['start'] + '..' + m['end']:>18}{m['months']:>8}"
                  f"{m['premium']:>9.2%}{m['us']:>8.2%}{m['lag']:>8.2%}{m['se']:>7.2%}"
                  f"   {factors.PREMIA[region]['mkt_rf']:.2%}")
        first = next(iter(lag.values()))
        print(f"\n  RECENT_LAG_WINDOW = ('{first['start']}', '{first['end']}')")
        print('  RECENT_LAG = {' + ', '.join(f"'{r}': {m['lag']:.4f}"
                                         for r, m in lag.items()) + '}')
        return 0

    print(f"\n  {'region':<7}{'factor':<8}{'window':>20}{'months':>8}"
          f"{'premium':>10}{'SE':>8}   in factors.py")
    print('  ' + '-' * 76)
    out = {}
    for region, by_f in regions.items():
        out[region] = {}
        for f in FACTORS:
            if f not in by_f:
                continue
            m = measure(by_f[f])
            if m is None:
                continue
            out[region][f] = m
            cur = factors.PREMIA.get(region, {}).get(f)
            delta = (f"{cur:.2%} ({m['premium'] - cur:+.2%})" if cur is not None else '—')
            print(f"  {region:<7}{f:<8}{m['start'] + '..' + m['end']:>20}"
                  f"{m['months']:>8}{m['premium']:>10.2%}{m['se']:>8.2%}   {delta}")

    for table, cell in (
            ('PREMIA', lambda m: f"{m['premium']:.4f}"),
            ('PREMIUM_SE', lambda m: f"{m['se']:.4f}"),
            ('PREMIUM_WINDOW', lambda m: f"({m['months']}, '{m['start']}')")):
        print(f"\n  {table} = {{")
        for region, by_f in out.items():
            cells = ', '.join(f"'{f}': {cell(by_f[f])}" for f in FACTORS if f in by_f)
            print(f"      '{region}': {{{cells}}},")
        print("  }")
    corr = error_correlation(regions, factors.PREMIUM_KEYS)
    name = 'PREMIUM_CORR_LONG' if args.long else 'PREMIUM_CORR'
    print(f"\n  {name} = (   # rows and columns in PREMIUM_KEYS order")
    for i, (region, f) in enumerate(factors.PREMIUM_KEYS):
        cells = ', '.join(f"{v:+.2f}" for v in corr[i])
        print(f"      ({cells}),   # {region} {f}")
    print("  )")
    print(f"  min eigenvalue {np.linalg.eigvalsh(corr).min():.3f} (must be > 0)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
