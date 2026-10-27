#!/usr/bin/env python3
"""
Why do the two mu methods disagree, ticker by ticker?
=====================================================

    python3 tools/mu_decompose.py

Splits `factor mu - historical mu` into its sources, because the answer turned
out not to be "factors":

  rf      the historical mean embeds the risk-free rate of the years the fund
          LIVED THROUGH; the build-up uses RF_FORWARD. AVDE earned 8.41% in
          years when cash paid 1.69%, and this term is +2.66 of its +2.97 gap.
  premia  the build-up uses premia over 1963+/1990+, not over the fund's own
          years.
  alpha   what the fund earned beyond its exposures, which the build-up drops.
  other   residual — the historical series is SPLICED, so its window and the
          actual fund's do not coincide.

VOO and AVUV are the control: their windows span 1970 and 1963, so their rf
terms are +0.01 and -0.00 and their gaps are small. The disagreement is
concentrated in funds whose history is mostly post-2000.

WHAT THIS DOES NOT LICENCE. It is tempting to conclude the historical path
should be re-based on contemporaneous rf, as the bond path already is.
Measured across decades, that does not hold up: sd(TOTAL return) is 6.11pp
against sd(EXCESS) 6.10pp, and corr(rf, total) is +0.22 against corr(rf, excess)
-0.21. Seven decades cannot tell the conventions apart, so "the equity premium
is stable while rf varies" is an assertion, not a finding. `historical` stays
the unmodified record, which is the only thing that makes it useful as a check.
"""
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings('ignore')

import pandas as pd                                             # noqa: E402
import data_collection_regime_model as dc                       # noqa: E402
from montecarlo import factors                                                  # noqa: E402


def main():
    kf = dc.KenFrenchParser(str(ROOT / 'reference'))
    f5 = kf.parse_ff5_factors()
    f5.index = f5.index.to_period('M').to_timestamp()
    rf = f5['rf']
    sets = {'us': f5.join(kf.parse_momentum_factor(), how='inner').dropna(),
            'dev': kf.parse_developed_ex_us_factors().dropna(),
            'emg': kf.parse_emerging_factors().dropna()}
    for k in sets:
        sets[k].index = sets[k].index.to_period('M').to_timestamp()

    fits = factors.load_fits()
    if not fits:
        raise SystemExit('no config/factor_loadings.json')
    cur = json.loads((ROOT / 'output' /
                      'ticker_parameters_regimes_v8.json').read_text())
    end = pd.Timestamp(cur['metadata']['last_updated']).to_period('M').to_timestamp()

    print(f"\n  {'tkr':<7}{'hist':>7}{'factor':>8}{'gap':>7}   = "
          f"{'rf':>7}{'premia':>8}{'alpha':>7}{'other':>7}")
    print('  ' + '-' * 60)
    rows = []
    for t in sorted(fits):
        fit = fits[t]
        if fit.get('method') != 'factor' or t not in cur:
            continue
        n = cur[t]['history_months']
        start = end - pd.DateOffset(months=n)
        hist = cur[t]['expected_return'] * 100
        mu = factors.expected_return(fit, costs=factors.costs_for(t, fit)) * 100

        d_rf = factors.RF_FORWARD * 100 - rf[rf.index >= start].mean() * 12 * 100
        full = same = 0.0
        for reg, loads in fit['beta'].items():
            win = sets[reg][sets[reg].index >= start]
            for k, b in loads.items():
                full += b * factors.PREMIA[reg][k] * 100
                same += b * win[k].mean() * 12 * 100
        d_prem = full - same
        d_alpha = -(fit.get('alpha') or 0.0) * 100
        other = (mu - hist) - d_rf - d_prem - d_alpha
        rows.append((abs(mu - hist), t, hist, mu, d_rf, d_prem, d_alpha, other))

    for _, t, hist, mu, a, b, c, d in sorted(rows, reverse=True):
        print(f"  {t:<7}{hist:>6.2f}%{mu:>7.2f}%{mu - hist:>+7.2f}   = "
              f"{a:>+7.2f}{b:>+8.2f}{c:>+7.2f}{d:>+7.2f}")

    big_rf = [r for r in rows if abs(r[4]) > 1.0]
    print(f"\n  rf is worth more than 1pp for {len(big_rf)} of {len(rows)} "
          f"tickers: {', '.join(sorted(r[1] for r in big_rf))}")
    print("  Those are the funds whose history is mostly post-2000, i.e. mostly "
          "ZIRP.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
