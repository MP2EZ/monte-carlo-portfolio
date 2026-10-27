#!/usr/bin/env python3
"""
Parameter drift — did this regen change more than a data refresh should?
========================================================================

    python3 tools/check_parameter_drift.py            # compare to the baseline
    python3 tools/check_parameter_drift.py --update   # re-bless after a change
    python3 tools/check_parameter_drift.py --verbose  # show every move

THE GAP THIS FILLS. `regression_check.py` guards the ENGINE on a synthetic
universe, deliberately, so that updating market data does not break it. The
consequence is that until 2026-09-11 the real parameters had NO regression guard
at all: `validate_parameters.py` checks each number is individually plausible
(mu in [-5%, 25%], a valid Markov row, a PSD matrix), which a badly wrong number
passes easily.

Both of the defects found on 2026-09-10/11 lived in that hole for as long as the
code had existed, and both were caught by diffing two regens BY HAND:

  - the international proxy table was rotated by one, so VEA carried EMERGING
    market returns for 41% of its history — its high-regime volatility was
    6.6pp wrong, and every individual number was perfectly plausible
  - `corr_df.dropna()` let the newest ticker truncate the correlation window
    from 180 months to 136, and the printed window length was computed after
    the truncation, so it reported 136 and looked correct

This makes that diff routine. It is the GOLDEN MASTER FOR DATA: a committed
snapshot, a tolerance per field, and `--update` to re-bless deliberately.

WHY THE TOLERANCES ARE WIDE. Parameters are regenerated from live market data,
so everything moves a little every time. These are sized to pass a quarterly
refresh and fail a bug — roughly an order of magnitude above observed
quarter-to-quarter drift. A move inside tolerance is REPORTED with --verbose and
never fails; the point is to catch a 6pp regime shift, not a 6bp one.

HISTORY_MONTHS IS THE SHARPEST CHECK and costs nothing: a ticker's history can
only GROW. A decrease means a splice silently dropped data, which is precisely
the third bug fixed on 2026-09-09 (actual data fetched from a hardcoded 2010,
discarding 72 months of VTV's own returns). Any decrease fails, at any size.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                               # noqa: E402

from montecarlo.loaders import MODEL_TICKERS, SINGLE_STOCKS                 # noqa: E402

# Rolling-window tickers churn more per refresh; see ROLLING_TOL_MULT.
ROLLING = {'AVEE'} | set(SINGLE_STOCKS)

JSON = ROOT / 'output' / 'ticker_parameters_regimes_v8.json'
BASELINE = Path(__file__).resolve().parent / 'parameter_baseline.json'

# Absolute tolerances on annualized rates, sized from MEASURED refresh drift
# rather than guessed. The 2026-09-09 double-regen isolated exactly this: eleven
# days of new market data moved every parameter by <= 0.11pp, except COST at
# 0.54pp. So a 1pp band on mu clears a refresh with room and still catches a
# real defect.
#
# The first version of this file used 2pp, and that was too loose to do its job:
# the VOO price-index bug fixed on 2026-09-09 moved mu 9.52% -> 11.23%, a 1.70pp
# error that would have passed silently. A guard calibrated above the bug it is
# meant to catch is theatre.
TOL = {
    'expected_return': 0.010,
    'base_volatility': 0.015,
    'parameter_std': 0.008,
    'regime_low': 0.025,
    'regime_medium': 0.025,
    'regime_high': 0.025,
}
# ...except for tickers on a rolling window, which genuinely churn: each regen
# drops months off the back as well as adding them to the front, so COST moves
# several times more than a fixed-window ticker on the same refresh.
ROLLING_TOL_MULT = 3.0
CORR_MEAN_TOL = 0.05        # mean |off-diagonal| per matrix
CORR_PAIR_TOL = 0.25        # any single named pair


def snapshot():
    """The fields worth guarding, flattened."""
    data = json.loads(JSON.read_text())
    out = {'tickers': {}, 'correlations': {}, 'market_regime': {}}
    for t in MODEL_TICKERS:
        d = data[t]
        r = d['volatility_regimes']
        out['tickers'][t] = {
            'expected_return': float(d['expected_return']),
            'base_volatility': float(d['base_volatility']),
            'parameter_std': float(d['parameter_std']),
            'regime_low': float(r['low']),
            'regime_medium': float(r['medium']),
            'regime_high': float(r['high']),
            'history_months': int(d['history_months']),
        }
    n = len(MODEL_TICKERS)
    iu = np.triu_indices(n, 1)
    for kind in ('baseline', 'low', 'medium', 'high'):
        m = np.load(ROOT / 'output' / f'correlation_{kind}_{n}.npy')
        out['correlations'][kind] = {'mean_abs': float(np.abs(m[iu]).mean())}
    mr = (data.get('metadata') or {}).get('market_regime') or {}
    if mr.get('realised_vol'):
        out['market_regime'] = {k: float(v) for k, v in mr['realised_vol'].items()}
    out['last_updated'] = (data.get('metadata') or {}).get('last_updated')
    return out


def compare(old, new, verbose=False):
    problems, moved = [], []
    for t, fields in new['tickers'].items():
        prev = old['tickers'].get(t)
        if prev is None:
            moved.append(f"{t}: NEW ticker")
            continue
        for f, v in fields.items():
            if f == 'history_months':
                # Can only grow. A decrease means a splice dropped data.
                if v < prev[f]:
                    problems.append(f"{t}.history_months FELL {prev[f]} -> {v} "
                                    f"— a splice dropped data")
                elif v != prev[f] and verbose:
                    moved.append(f"{t}.history_months {prev[f]} -> {v}")
                continue
            d = v - prev[f]
            tol = TOL[f] * (ROLLING_TOL_MULT if t in ROLLING else 1.0)
            if abs(d) > tol:
                problems.append(f"{t}.{f} {prev[f]:.2%} -> {v:.2%} ({d:+.2%}, "
                                f"tolerance {tol:.1%})")
            elif abs(d) > 1e-9 and verbose:
                moved.append(f"{t}.{f} {prev[f]:.2%} -> {v:.2%} ({d:+.2%})")
    for kind, cur in new['correlations'].items():
        prev = (old['correlations'].get(kind) or {}).get('mean_abs')
        if prev is None:
            continue
        d = cur['mean_abs'] - prev
        if abs(d) > CORR_MEAN_TOL:
            problems.append(f"correlation[{kind}] mean |rho| {prev:.3f} -> "
                            f"{cur['mean_abs']:.3f} ({d:+.3f})")
        elif abs(d) > 1e-9 and verbose:
            moved.append(f"correlation[{kind}] mean |rho| {d:+.3f}")
    for r, v in (new.get('market_regime') or {}).items():
        prev = (old.get('market_regime') or {}).get(r)
        if prev is None:
            continue
        if abs(v - prev) > 0.03:
            problems.append(f"market_regime[{r}] realised vol {prev:.1%} -> "
                            f"{v:.1%} ({v - prev:+.1%})")
    for t in old['tickers']:
        if t not in new['tickers']:
            problems.append(f"{t}: DISAPPEARED from the parameters")
    return problems, moved


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--update', action='store_true',
                    help='re-bless the baseline from the current parameters')
    ap.add_argument('--verbose', action='store_true',
                    help='list every move, including those inside tolerance')
    args = ap.parse_args(argv)

    if not JSON.exists():
        print(f"  no parameters at {JSON.relative_to(ROOT)} — run "
              f"data_collection_regime_model.py")
        return 1
    new = snapshot()

    if args.update or not BASELINE.exists():
        why = 're-blessed' if BASELINE.exists() else 'created'
        BASELINE.write_text(json.dumps(new, indent=2, sort_keys=True) + '\n')
        print(f"  [ -- ] parameter baseline {why} -> {BASELINE.name} "
              f"(parameters dated {new['last_updated']})")
        return 0

    old = json.loads(BASELINE.read_text())
    problems, moved = compare(old, new, args.verbose)
    print(f"Parameter drift vs baseline ({old.get('last_updated')} -> "
          f"{new.get('last_updated')})")
    print("=" * 60)
    if args.verbose and moved:
        for m in moved:
            print(f"  ·      {m}")
    if problems:
        print(f"  [FAIL] {len(problems)} parameter(s) moved more than a refresh "
              f"should")
        for p in problems:
            print(f"         {p}")
        print("=" * 60)
        print("✗ Investigate the splice/proxy for these, or --update if intended.")
        return 1
    print(f"  [PASS] every parameter is within tolerance of the baseline")
    print("=" * 60)
    print("✓ No unexplained parameter drift.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
