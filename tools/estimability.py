#!/usr/bin/env python3
"""
Why the LEVEL cannot be estimated, demonstrated rather than asserted.
====================================================================

    python3 tools/estimability.py

The load-bearing claim of this whole model: a fund's
expected return is not recoverable from its return series, no matter how much
data you have or how carefully you clean it. Everything downstream — the factor
build-up, the named scenarios, the refusal to quote a single number — exists
because of it. It deserves a demonstration on the actual data rather than a
citation.

THE ASYMMETRY (Merton 1980). For a series with volatility sigma observed over T
years at any frequency:

    SE(mean)     = sigma / sqrt(T)        depends ONLY on calendar SPAN
    SE(variance) ~ sigma^2 * sqrt(2/n)    depends on the NUMBER of observations

Sampling more often raises n and leaves T alone. So volatility can be pinned
arbitrarily well by sampling faster, and the mean cannot be improved at all.
That is not a limitation of the estimator; it is what the estimator is.

This script shows both halves on the repo's own spliced series, then shows what
pooling across assets buys — which is the entire reason the factor build-up
exists — and where that hits its own floor.
"""

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings('ignore')

import numpy as np                                               # noqa: E402


def main(argv=None):
    import data_collection_regime_model as dc
    from montecarlo import factors
    from montecarlo.loaders import MARKET_PROXY

    print(__doc__.split('THE ASYMMETRY')[0].rstrip())
    kf = dc.KenFrenchParser(str(ROOT / 'reference'))
    mkt = kf.parse_ff3_factors()['mkt_rf'].dropna()
    T = len(mkt) / 12.0
    sig = float(mkt.std() * np.sqrt(12))
    mu = float(mkt.mean() * 12)

    print(f"\n  THE US MARKET PREMIUM — the best-measured number in finance")
    print(f"  {len(mkt)} months, {T:.0f} years, "
          f"{mkt.index[0]:%Y-%m}..{mkt.index[-1]:%Y-%m}")
    print(f"    mean {mu:.2%}/yr      volatility {sig:.1%}/yr")
    print(f"    SE(mean) = sigma/sqrt(T) = {sig:.3f}/sqrt({T:.0f}) = "
          f"{sig/np.sqrt(T):.2%}/yr")
    print(f"    95% interval: {mu - 1.96*sig/np.sqrt(T):+.2%} .. "
          f"{mu + 1.96*sig/np.sqrt(T):+.2%}")
    print(f"\n  A CENTURY of the cleanest data available still leaves a "
          f"{2*1.96*sig/np.sqrt(T):.1%} wide interval.")

    print(f"\n  HOW LONG WOULD IT TAKE to pin it to a given precision?")
    print(f"    {'target SE':>12}{'years needed':>16}")
    for target in (0.02, 0.01, 0.005, 0.0025):
        yrs = (sig / target) ** 2
        print(f"    {target:>11.2%}{yrs:>15,.0f}")
    print(f"  Nobody is getting {(sig / 0.005) ** 2:,.0f} years of stationary data "
          f"(the 0.5% row). This is the whole")
    print(f"  argument, and no amount of care about proxies or splices touches it.")

    print(f"\n  THE ASYMMETRY — sampling FASTER fixes vol and does nothing for mu.")
    px = dc.download_with_retry(MARKET_PROXY, start='2010-01-01')
    d = px.pct_change().dropna()
    print(f"    {'sampling':<12}{'n obs':>8}{'years':>8}{'SE(mean)/yr':>14}{'SE(vol)/yr':>13}")
    for label, rule in (('daily', None), ('weekly', 'W'), ('monthly', 'ME'),
                        ('quarterly', 'QE')):
        r = d if rule is None else px.resample(rule).last().pct_change().dropna()
        per_year = {'daily': 252, 'weekly': 52, 'monthly': 12, 'quarterly': 4}[label]
        n = len(r)
        years = n / per_year
        se_mu = float(r.std() * np.sqrt(per_year)) / np.sqrt(years)
        vol = float(r.std() * np.sqrt(per_year))
        se_vol = vol / np.sqrt(2 * n)
        print(f"    {label:<12}{n:>8}{years:>8.1f}{se_mu:>13.2%}{se_vol:>13.2%}")
    print(f"  SE(mean) is FLAT down the column — every row covers the same calendar")
    print(f"  span. SE(vol) shrinks with the square root of the observation count.")

    print(f"\n  WHAT POOLING BUYS — and why the factor build-up exists.")
    print(f"  A single fund's own mean, against the premium it is priced off:")
    print(f"    {'source':<34}{'months':>8}{'SE of mu':>11}")
    fits = factors.load_fits()
    for t in ('VOO', 'AVUV', 'AVLV'):
        f = fits.get(t)
        if not f or f.get('method') != 'factor':
            continue
        n = f['n_months']
        s = 0.16 / np.sqrt(n / 12.0)
        print(f"    {t + ' own mean':<34}{n:>8}{s:>10.2%}")
    # The months behind each premium, not len(mkt): PREMIUM_SE is measured on
    # whatever window factors.PREMIUM_WINDOW records, which is the 1963 5-factor
    # window unless MC_PREMIA_LONG_WINDOW is set.
    for f, label in (('mkt_rf', 'US market premium (pooled)'),
                     ('hml', 'value premium (pooled)')):
        n_f = factors.PREMIUM_WINDOW['us'][f][0]
        print(f"    {label:<34}{n_f:>8}{factors.PREMIUM_SE['us'][f]:>10.2%}")
    print(f"\n  Pooling roughly HALVES the error, because every fund shares the same")
    print(f"  few premia and those are estimated on a century rather than on one")
    print(f"  fund's lifetime. That is the entire gain, and it stops there: the")
    print(f"  floor is the premium's own SE, which is the first table again.")
    print(f"\n  So the model estimates what CONVERGES (betas, volatility, regimes,")
    print(f"  correlations) and refuses to pin what does not. tools/sensitivity.py")
    print(f"  (its `scenario` and `premium` sweeps) replaces the point estimate.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
