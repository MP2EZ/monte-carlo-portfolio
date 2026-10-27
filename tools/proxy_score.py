#!/usr/bin/env python3
"""
Proxy fidelity — score every backfill proxy against the fund it stands in for.
=============================================================================

    python3 tools/proxy_score.py                # the whole universe
    python3 tools/proxy_score.py VEA VWO VSS    # just these
    python3 tools/proxy_score.py --json output/proxy_scores.json

The first fidelity check (2026-09-09) scored the seven Ken French-blended
tickers and concluded "the proxies are fine". That covered seven of twenty-two. The other
fifteen are plain yfinance backfills and had never been scored at all — and all
three splice bugs fixed on 2026-09-09 lived in exactly that machinery.

METHOD, identical to that check: a proxy covers the months BEFORE the real fund
existed, so it can only be checked on the months AFTER, which the pipeline
discards. Extend the proxy forward over that overlap and compare.

`contrib` IS THE COLUMN THAT DECIDES WHETHER A BAD SCORE MATTERS. It counts the
months the proxy actually reaches the spliced series. For a Ken French-blended
ticker whose academic window runs to its own inception the proxy layer is 0-1
months — vestigial, and a bad score there is a loaded gun rather than a live
defect. At `contrib` 158 it is 41% of the fund's history.

WHAT THE COLUMNS MEAN. `corr` and `β` say whether the proxy is the same ASSET —
that is what a backfill has to get right, because the spliced series sets
volatility, regimes and correlations. `gap` is the annualized difference in mean
return and is the column NOT to over-read: at a 50-month overlap it carries a
standard error of several pp, which is why `t` is printed beside it. A proxy
with corr 0.99 and a 2pp gap is doing its job; one with corr 0.6 is a different
asset whatever its mean does.

Under MU_METHOD='factor' the proxy no longer touches expected returns at all —
it builds the series that sets the second moments. Score it accordingly.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings('ignore')

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402

# A backfill this far from its fund is a different asset, not a noisy estimate.
CORR_FLOOR = 0.90
BETA_BAND = (0.80, 1.20)

# ...except for near-cash funds, where correlation is the wrong statistic. JPST
# moves 1.03%/yr; almost all of that is microstructure, so two genuinely similar
# ultra-short funds correlate around 0.9 no matter how well one stands in for the
# other. What matters there is whether the proxy moves by the right AMOUNT, so a
# fund under CASHLIKE_VOL is gated on tracking error instead.
CASHLIKE_VOL = 0.02
CASHLIKE_TE = 0.01

# A proxy layer this short cannot be wrong about anything yet. Named, not hidden.
VESTIGIAL_MONTHS = 3


def monthly(ticker, dc, start='1970-01-01'):
    px = dc.download_with_retry(ticker, start=start)
    r = px.resample('ME').last().pct_change().dropna()
    r.index = r.index.to_period('M').to_timestamp('M')
    return r


def proxy_series(ticker, dc):
    """(series, label) for whatever stands in for `ticker` before inception."""
    blend = dc.BLENDED_PROXIES.get(ticker)
    if blend:
        comps = blend['components']
        df = pd.DataFrame({c: monthly(c, dc) for c in comps}).dropna()
        s = sum(w * df[c] for c, w in comps.items())
        return s, ' + '.join(f'{w:.0%} {c}' for c, w in comps.items())
    p = dc.PROXIES.get(ticker)
    if not p:
        return None, None
    return monthly(p, dc), p


def contributed(ticker, prox, act, dc):
    """Months of the proxy that actually survive into the spliced series.

    Mirrors the splice branches in data_collection_regime_model. A Ken French
    ticker's proxy is downloaded from the academic cutoff, so it covers
    cutoff..inception and nothing earlier; a ROLLING_WINDOW ticker is then
    truncated to its last 240 months, which can discard proxy months wholesale.

    The pipeline computes this against the final spliced series and cannot drift.
    This standalone copy has to reconstruct it, so the rolling window is applied
    explicitly — without it AVEE reported 308 contributed months against a
    240-month history, which is impossible.
    """
    # PREFER THE STORED COUNT. data_collection_regime_model computes this
    # against the series it actually built, so it cannot be wrong; anything here
    # is a reconstruction and the two drifted by a month on every Ken French
    # ticker (a boundary month that the splice resolves and a rebuild guesses
    # at). One authoritative computation, reused.
    stored = _stored_contribution(ticker)
    if stored is not None:
        return stored

    first = act.index.min()
    if ticker in dc.KF_TICKERS:
        cut = pd.Timestamp(dc.KF_TICKERS[ticker]['end'])
        n = int(((prox.index >= cut) & (prox.index < first)).sum())
    else:
        n = int((prox.index < first).sum())
    if ticker in dc.ROLLING_WINDOW_TICKERS:
        # Only the tail survives; the proxy keeps whatever the actual fund does
        # not already fill of those 240 months.
        n = max(0, min(n, 240 - len(act)))
    return n


def _stored_contribution(ticker):
    """The pipeline's own count, if parameters have been generated."""
    try:
        meta = json.loads((ROOT / 'output' /
                           'ticker_parameters_regimes_v8.json').read_text())['metadata']
        return int(meta['proxy_scores'][ticker]['contributed_months'])
    except (OSError, KeyError, ValueError, TypeError):
        return None


def score(ticker, dc):
    """Compare proxy and fund on the overlap the pipeline throws away."""
    prox, label = proxy_series(ticker, dc)
    if prox is None:
        return None
    act = monthly(ticker, dc)
    if len(act) < 2:
        return None
    j = pd.concat([act.rename('act'), prox.rename('prox')], axis=1).dropna()
    j = j[j.index >= act.index.min()]
    contrib = contributed(ticker, prox, act, dc)
    if len(j) < 12:
        return {'ticker': ticker, 'proxy': label, 'n': len(j),
                'contrib': contrib, 'thin': True}

    a, p = j['act'].values, j['prox'].values
    d = a - p
    fund_vol = float(a.std(ddof=1) * np.sqrt(12))
    beta = float(np.cov(a, p, ddof=1)[0, 1] / np.var(p, ddof=1))
    gap = float(d.mean() * 12)
    # SE of the mean gap, annualized — the reason `gap` gets a t and not a verdict.
    se = float(d.std(ddof=1) / np.sqrt(len(d)) * 12)
    return {
        'ticker': ticker, 'proxy': label, 'n': int(len(j)),
        'contrib': contrib, 'share': contrib / (contrib + len(act)),
        'window': f"{j.index[0]:%Y-%m}..{j.index[-1]:%Y-%m}",
        'corr': float(np.corrcoef(a, p)[0, 1]), 'beta': beta,
        'fund_vol': fund_vol, 'proxy_vol': float(p.std(ddof=1) * np.sqrt(12)),
        'te': float(d.std(ddof=1) * np.sqrt(12)),
        'proxy_mu': float(p.mean() * 12), 'actual_mu': float(a.mean() * 12),
        'gap': gap, 'gap_se': se, 't': gap / se if se else float('nan'),
        'kf': ticker in dc.KF_TICKERS,
    }


def verdict(s):
    if s.get('thin'):
        return 'THIN'
    if s['fund_vol'] < CASHLIKE_VOL:
        bad, why = s['te'] > CASHLIKE_TE, 'TE too wide for a cash-like fund'
    else:
        bad = s['corr'] < CORR_FLOOR
        why = 'WRONG ASSET'
        if not bad and not BETA_BAND[0] <= s['beta'] <= BETA_BAND[1]:
            bad, why = True, 'beta off'
    if not bad:
        return 'ok'
    if s['contrib'] <= VESTIGIAL_MONTHS:
        return 'vestigial'
    return why


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('tickers', nargs='*', help='default: every proxied ticker')
    ap.add_argument('--json', metavar='PATH')
    args = ap.parse_args(argv)

    import data_collection_regime_model as dc
    from montecarlo.loaders import MODEL_TICKERS

    want = args.tickers or [t for t in MODEL_TICKERS
                            if dc.PROXIES.get(t) or t in dc.BLENDED_PROXIES]

    head = (f"{'tkr':<7}{'proxy':<22}{'n':>5}{'used':>6}{'share':>7}"
            f"{'corr':>7}{'beta':>7}{'TE/yr':>8}{'proxy mu':>10}{'fund mu':>9}"
            f"{'gap':>8}{'t':>6}  verdict")
    print(head)
    print('-' * (len(head) + 4))

    out, bad, vestigial = {}, [], []
    for t in want:
        try:
            s = score(t, dc)
        except Exception as e:
            print(f"{t:<7}{'—':<22}{e.__class__.__name__}: {e}")
            continue
        if s is None:
            continue
        out[t] = s
        v = verdict(s)
        if v not in ('ok', 'vestigial'):
            bad.append((t, v))
        elif v == 'vestigial':
            vestigial.append(t)
        if s.get('thin'):
            print(f"{t:<7}{s['proxy']:<22}{s['n']:>5}{s['contrib']:>6}"
                  f"   only {s['n']} overlap months")
            continue
        mark = '' if s['kf'] else ' *'
        print(f"{t:<7}{s['proxy']:<22}{s['n']:>5}{s['contrib']:>6}"
              f"{s['share']:>7.0%}{s['corr']:>7.3f}{s['beta']:>7.2f}{s['te']:>7.1%}"
              f"{s['proxy_mu']:>10.2%}{s['actual_mu']:>9.2%}"
              f"{s['gap'] * 100:>+8.2f}{s['t']:>6.1f}  {v}{mark}")

    print(f"\n  * never scored before — the first check covered only the Ken French blends")
    if bad:
        print(f"\n  ⚠ {len(bad)} proxy(ies) fail the asset check "
              f"(corr < {CORR_FLOOR}, or beta outside {BETA_BAND}):")
        for t, v in bad:
            s = out[t]
            print(f"      {t:<6} proxied by {s['proxy']:<20} {v}"
                  f"   corr {s.get('corr', float('nan')):.3f}"
                  f"   {s['contrib']} months ({s['share']:.0%} of its history)")
    else:
        print(f"\n  every proxy clears corr ≥ {CORR_FLOOR} and beta in {BETA_BAND}"
              f"\n  (cash-like funds under {CASHLIKE_VOL:.0%} vol gated on "
              f"TE ≤ {CASHLIKE_TE:.0%}/yr instead)")

    if vestigial:
        print(f"  {len(vestigial)} vestigial (bad fit, reaches ≤{VESTIGIAL_MONTHS} months): "
              f"{', '.join(vestigial)}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2) + '\n')
        print(f"  wrote {args.json}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
