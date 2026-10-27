#!/usr/bin/env python3
"""
Is the single stock still earning its place? — an annual check.
================================================================

    python3 tools/single_stock_benchmark.py            # every ticker in SINGLE_STOCKS
    python3 tools/single_stock_benchmark.py COST --threshold -0.07

The rule it checks: hold a single stock (any ticker in `loaders.SINGLE_STOCKS`;
COST ships) while its ROLLING 10-YEAR excess return over its FACTOR BUILD-UP
stays above an exit trigger (default −5%/yr), checked once a year. The
build-up is the recipe the model forecasts from — six US factor loadings
fitted on the stock's last 240 months (`factors.fit_single_stock`) — so "the
stock is earning its place" and "the model's forecast is being met" are one
test. A market-only benchmark (β × VOO + cash) omits the profitability premium
a stock like COST carries, which the model pays every fund.

Why ten years and an annual check: COST's stock-specific volatility is ~15%/yr,
so one year's excess is ±15 points of noise. In simulation, any 3- or
4-quarter rule checked quarterly says SELL within ten years with certainty,
whatever the stock's true edge.

The 5/3/1-year figures are printed for context and decide nothing. The
loadings are fitted on the same window the excess is measured over, so the
reading is slightly flattering.
"""

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
warnings.filterwarnings('ignore')

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402

from montecarlo import factors                                                   # noqa: E402
from montecarlo.loaders import SINGLE_STOCKS                                # noqa: E402

# The default exit trigger. The model's break-even — where holding the stock instead of
# VOO stops paying — is about −1%/yr (2026-10-12); the trigger sits below it on
# purpose, because selling a stock that keeps its edge costs ~10% of lifetime
# spending while holding one that lags by 2 points costs ~0-3%.
EXIT_THRESHOLD = -0.05
WINDOW_YEARS = 10
MODEL_BREAK_EVEN = -0.01

fit = factors.fit_single_stock


def excess(returns, frame, fit_):
    """Monthly return over the factor build-up: r − rf − Σ β·f."""
    loads = fit_['beta']['us']
    j = pd.concat([returns.rename('y'), frame], axis=1).dropna()
    bench = j['rf'] + sum(b * j[f] for f, b in loads.items())
    return j['y'] - bench


def verdict(monthly_excess, threshold=EXIT_THRESHOLD, years=WINDOW_YEARS):
    """HOLD / EXIT on the trailing `years`, or INSUFFICIENT without them."""
    n = 12 * years
    if len(monthly_excess) < n:
        return {'status': 'INSUFFICIENT', 'months': len(monthly_excess)}
    e = monthly_excess.iloc[-n:]
    mean = float(e.mean() * 12)
    se = float(e.std() * np.sqrt(12) / np.sqrt(years))
    return {'status': 'EXIT' if mean < threshold else 'HOLD',
            'mean': mean, 'se': se, 'threshold': threshold}


def _monthly(ticker, dc):
    px = dc.download_with_retry(ticker, start='1990-01-01')
    r = px.resample('ME').last().pct_change().dropna()
    r.index = r.index.to_period('M').to_timestamp()
    return r


def report(ticker, threshold):
    import data_collection_regime_model as dc
    from factor_loadings import factor_frames
    us = factor_frames(dc.KenFrenchParser(str(ROOT / 'reference')))['us']
    r = _monthly(ticker, dc)
    f = fit(r, us)
    if f is None:
        print(f"  {ticker}: too little overlap with the US factors")
        return 1
    ex = excess(r, us, f)
    v = verdict(ex, threshold)
    loads = f['beta']['us']
    print(f"\n  {ticker}  ·  data to {ex.index[-1]:%Y-%m}  ·  loadings on the last "
          f"{f['n_months']} months")
    print("  loadings: " + '  '.join(f"{k} {b:+.2f}" for k, b in loads.items()))
    stored = ((factors.load_fits() or {}).get(ticker) or {}).get('beta', {}).get('us')
    if stored:
        drift = max(abs(loads[k] - stored.get(k, 0.0)) for k in loads)
        print(f"  largest move vs the model's stored loadings: {drift:.2f}"
              + ("  — refresh config/factor_loadings.json" if drift > 0.10 else ""))
    print(f"\n  excess over the factor build-up, %/yr (± one standard error):")
    for yrs in (10, 5, 3, 1):
        e = ex.iloc[-12 * yrs:]
        tag = '  ← the rule' if yrs == WINDOW_YEARS else '  (context only)'
        print(f"    last {yrs:>2} yrs  {e.mean() * 1200:+6.1f}  "
              f"±{e.std() * np.sqrt(12) / np.sqrt(yrs) * 100:4.1f}{tag}")
    if v['status'] == 'INSUFFICIENT':
        print(f"\n  VERDICT  INSUFFICIENT — {v['months']} months, the rule needs "
              f"{12 * WINDOW_YEARS}")
        return 0
    print(f"\n  VERDICT  {v['status']} — 10-yr excess {v['mean']:+.1%}/yr against an "
          f"exit trigger of {threshold:+.0%}/yr")
    print(f"           the model's break-even (hold vs VOO) is about "
          f"{MODEL_BREAK_EVEN:+.0%}/yr")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('tickers', nargs='*', help='default: loaders.SINGLE_STOCKS')
    ap.add_argument('--threshold', type=float, default=EXIT_THRESHOLD,
                    help=f'exit trigger, annual decimal (default {EXIT_THRESHOLD})')
    args = ap.parse_args(argv)
    rc = 0
    for t in args.tickers or sorted(SINGLE_STOCKS):
        rc |= report(t, args.threshold)
    return rc


if __name__ == '__main__':
    sys.exit(main())
