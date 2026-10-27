"""
Generated-parameter sanity validator.
======================================

`check_consistency.py` proves the ticker *plumbing* lines up; this proves the
generated *numbers* are physically plausible. It loads the JSON parameters + the
four correlation matrices that `data_collection_regime_model.py` writes and FAILS
loudly on anything a silent splicing/proxy bug would produce (a 110% vol, a
non-stochastic Markov row, a non-PSD correlation matrix, a NaN).

`loaders.load_parameters_from_json` already *warns* on some of these; this turns the
important ones into hard failures and adds physical-range + ordering smell tests.

Run after every `data_collection_regime_model.py` regen:
    python3 tools/validate_parameters.py     # exit 0 = sane, exit 1 = problems
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

# This script lives in tools/; the project root is one level up.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from montecarlo.loaders import MODEL_TICKERS, _BOND_FUNDS

OUTPUT = ROOT / 'output'
JSON = OUTPUT / 'ticker_parameters_regimes_v8.json'

# Physical plausibility bands (annualized). Wide enough to never flag a legitimate
# fund, tight enough to catch a splicing blowup. Observed today: mu 0.016-0.18,
# sigma 0.009-0.21.
MU_MIN, MU_MAX = -0.05, 0.25
SIGMA_MIN, SIGMA_MAX = 0.003, 0.60
PSTD_MIN, PSTD_MAX = 0.0, 0.10
MIN_HISTORY_MONTHS = 60          # 5y — below this, estimates are too noisy
REGIME_VOL_MIN_RATIO = 0.15      # a regime vol this far below base vol is a broken splice
LOW_VOL_ASSET = 0.01             # below 1% base vol, a flat 36m window is genuinely possible
ABSOLUTE_VOL_FLOOR = 0.0002      # 0.02%/yr — only a true zero is impossible
PROB_TOL = 1e-6                  # Markov row sum tolerance

# Proxy fidelity. The score is computed at collection time, where both
# series are in memory and the network is available; this gates the STORED
# result so validate_all.py still runs offline.
#
# A backfill's job is to be the same ASSET — it sets volatility, regimes and
# correlations, and under MU_METHOD='factor' it no longer touches expected
# returns at all. So the gate is correlation and beta, never the mean gap: at a
# 47-month overlap a mean is +-7.6pp and ranking on it would be ranking noise.
# The thresholds are proxy_score's own, imported rather than copied, so the
# tool that scores a proxy and the gate that judges the stored score cannot
# disagree. Near-cash funds are gated on tracking error, not correlation; a
# proxy reaching <= VESTIGIAL_MONTHS is reported, not failed.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from proxy_score import (CORR_FLOOR as PROXY_CORR_FLOOR,           # noqa: E402
                         BETA_BAND as PROXY_BETA_BAND,
                         CASHLIKE_VOL as PROXY_CASHLIKE_VOL,
                         CASHLIKE_TE as PROXY_CASHLIKE_TE,
                         VESTIGIAL_MONTHS as PROXY_VESTIGIAL_MONTHS)

# The market regime exists to make correlations depend on market stress. If the
# labelled states do not actually differ in realised volatility, they are not
# states. The majority-vote rule this replaced scored 1.00x here (medium 14.1%,
# high 14.1%); the 12-month market-proxy rule scores 2.3x.
MARKET_REGIME_MIN_SEPARATION = 1.5
MIN_REGIME_MONTHS = 36
PSD_EIG_TOL = -1e-8             # min-eigenvalue floor for "PSD enough"
TICKER_KEYS = {
    'expected_return', 'base_volatility', 'parameter_std',
    'volatility_regimes', 'regime_transitions',
    'history_months', 'history_start', 'history_end',
}
# The market-conditioned chains (item 5b). Gated below, and optional in the key
# check so parameters written before 2026-09-10 report one clear failure rather
# than one per ticker.
COND_KEY = 'regime_transitions_by_market'
REGIMES = ('low', 'medium', 'high')


def correlation_order_problem(stored, model_tickers):
    """Describe how a stored correlation ticker order disagrees with the model's.

    Returns '' when they agree. Pure so check_logic can test it without a JSON;
    the on-disk case is exercised by this validator itself.

    `check_consistency` already catches a plain reorder, via the order-sensitive
    `json_tickers == MODEL_TICKERS` comparison. This binds the ORDER TO THE
    MATRICES rather than to the JSON's key order, which is an incidental
    property of how the file was written: anything that regenerates the
    matrices WITHOUT rewriting the JSON — a correlation-only experiment, say —
    is invisible to that check and visible to this one.
    """
    if stored is None:
        return ("metadata.correlation_tickers absent — regenerate "
                "(python3 data_collection_regime_model.py)")
    stored, model_tickers = list(stored), list(model_tickers)
    if stored == model_tickers:
        return ""
    extra = [t for t in stored if t not in model_tickers]
    gone = [t for t in model_tickers if t not in stored]
    if extra or gone:
        detail = f"universe changed: +{gone or '-'} / -{extra or '-'}"
    else:
        first = next(i for i, (a, b) in enumerate(zip(stored, model_tickers)) if a != b)
        detail = (f"REORDERED at position {first}: matrices were built with "
                  f"'{stored[first]}' there, MODEL_TICKERS now has "
                  f"'{model_tickers[first]}'")
    return detail + " — regenerate; the matrices index the OLD order"


def main():
    problems = []

    def check(ok, label, detail=''):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if not ok:
            if detail:
                print(f"         {detail}")
            problems.append(label)

    print("Parameter sanity check")
    print("=" * 60)

    if not JSON.exists():
        check(False, "parameters JSON present", f"missing {JSON} — run data_collection")
        return _finish(problems)

    data = json.loads(JSON.read_text())
    tickers = [k for k in data if k != 'metadata']

    # --- 1. Structure: every model ticker present with the expected keys ----------
    missing = [t for t in MODEL_TICKERS if t not in data]
    check(not missing, "all model tickers present in JSON", f"missing: {missing}")

    bad_keys = {t: sorted(TICKER_KEYS - set(data[t])) for t in tickers
                if t in data and not TICKER_KEYS.issubset(data[t])}
    check(not bad_keys, "every ticker has the full key set", f"missing keys: {bad_keys}")

    # --- 2. Finite values + physical bounds ---------------------------------------
    mu_bad, sig_bad, pstd_bad, nan_bad = [], [], [], []
    for t in tickers:
        e = data[t]
        mu, sig, pstd = e.get('expected_return'), e.get('base_volatility'), e.get('parameter_std')
        for name, v in (('expected_return', mu), ('base_volatility', sig), ('parameter_std', pstd)):
            if v is None or not np.isfinite(v):
                nan_bad.append(f"{t}.{name}={v}")
        if mu is not None and np.isfinite(mu) and not (MU_MIN <= mu <= MU_MAX):
            mu_bad.append(f"{t}={mu:.3f}")
        if sig is not None and np.isfinite(sig) and not (SIGMA_MIN <= sig <= SIGMA_MAX):
            sig_bad.append(f"{t}={sig:.3f}")
        if pstd is not None and np.isfinite(pstd) and not (PSTD_MIN <= pstd <= PSTD_MAX):
            pstd_bad.append(f"{t}={pstd:.3f}")

    check(not nan_bad, "no NaN/Inf/missing scalar parameters", f"{nan_bad}")
    check(not mu_bad, f"expected_return in [{MU_MIN}, {MU_MAX}]", f"out of band: {mu_bad}")
    check(not sig_bad, f"base_volatility in [{SIGMA_MIN}, {SIGMA_MAX}]", f"out of band: {sig_bad}")
    check(not pstd_bad, f"parameter_std in [{PSTD_MIN}, {PSTD_MAX}]", f"out of band: {pstd_bad}")

    # --- 3. Regime vols: present, finite, ordered low <= medium <= high -----------
    regime_bad, order_bad = [], []
    for t in tickers:
        vr = data[t].get('volatility_regimes', {})
        vals = [vr.get(r) for r in REGIMES]
        if any(v is None or not np.isfinite(v) or v < 0 for v in vals):
            regime_bad.append(f"{t}={vals}")
            continue
        if not (vals[0] <= vals[1] <= vals[2]):
            order_bad.append(f"{t}: low={vals[0]:.3f} med={vals[1]:.3f} high={vals[2]:.3f}")
    check(not regime_bad, "regime vols present, finite, non-negative", f"{regime_bad}")

    # A broken splice can pass the finite/ordered checks with a near-zero regime
    # vol (JQUA shipped low=0.09% against a 12.62% base: the AQR workbook was
    # divided by 100 when already decimal). So test each regime vol against the
    # asset's OWN base volatility. Healthy low/base ratios run 62-66% (VOO, AVUV,
    # BND, BSV); the relative floor applies only above LOW_VOL_ASSET, because a
    # genuinely ultra-short fund (JPST proxies SHV, ZIRP-era vol ~0.07%) is
    # legitimately near-flat. Below that threshold only a true zero is flagged.
    tiny_regime = []
    for t in tickers:
        vr = data[t].get('volatility_regimes', {})
        base = data[t].get('base_volatility')
        if not base:
            continue
        floor = REGIME_VOL_MIN_RATIO * base if base >= LOW_VOL_ASSET else ABSOLUTE_VOL_FLOOR
        for r in REGIMES:
            v = vr.get(r)
            if v is not None and v < floor:
                tiny_regime.append(f"{t}.{r}={v:.4%} (base {base:.2%})")
    check(not tiny_regime,
          "regime vols plausible given each asset's own volatility",
          f"{tiny_regime}")
    check(not order_bad, "regime vols ordered low <= medium <= high", f"{order_bad}")

    # --- 4. Markov transition matrices: rows sum to 1, entries in [0,1] ------------
    trans_bad = []
    for t in tickers:
        rt = data[t].get('regime_transitions', {})
        for src in REGIMES:
            row = rt.get(src, {})
            probs = [row.get(dst) for dst in REGIMES]
            if any(p is None or not np.isfinite(p) or not (0.0 <= p <= 1.0) for p in probs):
                trans_bad.append(f"{t}.{src} entries={probs}")
            elif abs(sum(probs) - 1.0) > PROB_TOL:
                trans_bad.append(f"{t}.{src} sum={sum(probs):.6f}")
    check(not trans_bad, "regime_transitions are valid stochastic rows (sum=1, in [0,1])",
          f"{trans_bad[:8]}{' ...' if len(trans_bad) > 8 else ''}")

    # --- 5. History adequacy + date sanity ----------------------------------------
    hist_bad = []
    for t in tickers:
        e = data[t]
        hm = e.get('history_months')
        hs, he = str(e.get('history_start', '')), str(e.get('history_end', ''))
        if not isinstance(hm, int) or hm < MIN_HISTORY_MONTHS:
            hist_bad.append(f"{t}: {hm} months")
        elif hs and he and hs > he:
            hist_bad.append(f"{t}: start {hs} > end {he}")
    check(not hist_bad, f"history >= {MIN_HISTORY_MONTHS} months and start <= end", f"{hist_bad}")

    # --- 6. Smell test: bonds calmer than every equity ----------------------------
    bond_sigs = {t: data[t]['base_volatility'] for t in _BOND_FUNDS if t in data}
    equity = [t for t in tickers if t not in _BOND_FUNDS]
    equity_sigs = {t: data[t]['base_volatility'] for t in equity if t in data}
    if bond_sigs and equity_sigs:
        max_bond = max(bond_sigs.values())
        min_equity = min(equity_sigs.values())
        ok = max_bond < min_equity
        check(ok, "all bond vols < every equity vol (duration ordering)",
              f"max bond sigma {max_bond:.3f} >= min equity sigma {min_equity:.3f}")

    # --- 7. Metadata --------------------------------------------------------------
    meta = data.get('metadata', {})
    check(meta.get('ticker_count') == len(MODEL_TICKERS),
          f"metadata.ticker_count == {len(MODEL_TICKERS)}", f"got {meta.get('ticker_count')}")
    last = meta.get('last_updated', '')
    try:
        date.fromisoformat(str(last))
        last_ok = True
    except ValueError:
        last_ok = False
    check(last_ok and bool(meta.get('version')), "metadata version + ISO last_updated present",
          f"version={meta.get('version')} last_updated={last}")

    # --- 8. Correlation matrices --------------------------------------------------
    n = len(MODEL_TICKERS)

    # The matrices are POSITIONAL — every consumer reads a pair as
    # C[TICKERS.index(a), TICKERS.index(b)] and the .npy carries no labels. A
    # reorder of MODEL_TICKERS without a regen leaves a matrix that passes every
    # check below (still 25x25, symmetric, unit-diagonal, PSD) while attaching
    # each correlation to the wrong pair. Measured 2026-09-12: prepending one
    # ticker moves 141's crisis-regime volatility -11.4%, biased DOWNWARD.
    # This is the only check that can see it.
    detail = correlation_order_problem(meta.get('correlation_tickers'), MODEL_TICKERS)
    check(not detail, "correlation matrices were built in the current ticker order", detail)

    cor_files = meta.get('correlation_files', {})
    for kind in ('baseline', 'low', 'medium', 'high'):
        fname = cor_files.get(kind, f'correlation_{kind}_{n}.npy')
        path = OUTPUT / fname
        if not path.exists():
            check(False, f"correlation '{kind}' loads", f"missing {path}")
            continue
        m = np.load(path)
        issues = []
        if m.shape != (n, n):
            issues.append(f"shape {m.shape} != ({n},{n})")
        if np.isnan(m).any() or np.isinf(m).any():
            issues.append("contains NaN/Inf")
        if not np.allclose(m, m.T, atol=1e-8):
            issues.append("not symmetric")
        if not np.allclose(np.diag(m), 1.0, atol=1e-6):
            issues.append("diagonal != 1")
        if m.size and (m.min() < -1.0 - 1e-9 or m.max() > 1.0 + 1e-9):
            issues.append(f"entries outside [-1,1] (min {m.min():.3f}, max {m.max():.3f})")
        if not issues:  # eigen check only meaningful on a clean matrix
            min_eig = float(np.linalg.eigvalsh(m).min())
            if min_eig < PSD_EIG_TOL:
                issues.append(f"not PSD (min eig {min_eig:.2e})")
        check(not issues, f"correlation '{kind}' is a valid {n}x{n} correlation matrix",
              "; ".join(issues))

    # ---- market-conditioned per-ticker chains (item 5b) -----------------
    missing, bad_rows, no_signal = [], [], []
    for t in MODEL_TICKERS:
        cond = (data.get(t) or {}).get(COND_KEY)
        if not cond:
            missing.append(t)
            continue
        for market in REGIMES:
            rows = cond.get(market)
            if not rows:
                bad_rows.append(f"{t}/{market} missing")
                continue
            for frm in REGIMES:
                row = rows.get(frm)
                if not row or abs(sum(row.values()) - 1.0) > PROB_TOL:
                    bad_rows.append(f"{t}/{market}/{frm} does not sum to 1")
        # Coupling that is identical across market states is not coupling. It is
        # legitimate for a ticker with too little overlap to fall back to its
        # unconditional chain, so this is REPORTED rather than failed.
        try:
            spread = max(cond[m]['medium']['high'] for m in REGIMES) - \
                     min(cond[m]['medium']['high'] for m in REGIMES)
            if spread < 1e-9:
                no_signal.append(t)
        except (KeyError, TypeError):
            pass
    check(not missing, f"every ticker carries {COND_KEY}",
          f"{len(missing)} without it: {', '.join(missing[:6])}"
          + ('...' if len(missing) > 6 else ''))
    check(not bad_rows, 'market-conditioned transitions are valid stochastic rows',
          '; '.join(bad_rows[:6]) + ('...' if len(bad_rows) > 6 else ''))
    if no_signal:
        print(f"         note: {len(no_signal)} ticker(s) fell back to their "
              f"unconditional chain (too little market overlap): "
              f"{', '.join(no_signal)}")

    # ---- market regime -------------------------------------------------
    mr = meta.get('market_regime')
    if not mr:
        check(False, 'market_regime present in metadata',
              'no metadata.market_regime — regenerate with data_collection_regime_model.py')
    else:
        issues = []
        rows = mr.get('transitions', {})
        for r in REGIMES:
            row = rows.get(r)
            if not row or abs(sum(row.values()) - 1.0) > PROB_TOL:
                issues.append(f"{r} row does not sum to 1")
            elif any(not 0.0 <= v <= 1.0 for v in row.values()):
                issues.append(f"{r} row has a probability outside [0,1]")
        check(not issues, 'market_regime transitions are valid stochastic rows',
              '; '.join(issues))

        # The whole point of conditioning correlations on a market state is that
        # the states differ. If realised volatility does not increase across
        # low -> medium -> high the labelling has stopped meaning anything — which
        # is exactly what a 36-month estimation window did to it (it put October
        # 2008 in 'medium').
        rv = mr.get('realised_vol', {})
        ordered = all(rv.get(a, 0) <= rv.get(b, 0)
                      for a, b in (('low', 'medium'), ('medium', 'high')))
        sep = (rv.get('high', 0) / rv['low']) if rv.get('low') else 0.0
        check(ordered and sep >= MARKET_REGIME_MIN_SEPARATION,
              f"market regimes separate realised volatility "
              f"(high/low >= {MARKET_REGIME_MIN_SEPARATION}x)",
              f"low {rv.get('low', 0):.1%} medium {rv.get('medium', 0):.1%} "
              f"high {rv.get('high', 0):.1%} — {sep:.2f}x")

        thin = {r: n for r, n in (mr.get('correlation_window_months') or {}).items()
                if n < MIN_REGIME_MONTHS}
        if thin:
            print(f"         note: thin regime bucket(s) in the correlation window: "
                  + ', '.join(f'{r} {n}mo' for r, n in thin.items())
                  + f" — a 25x25 matrix is 300 free parameters")

    # ---- proxy fidelity ------------------------------------------------
    scores = meta.get('proxy_scores')
    if not scores:
        check(False, 'proxy fidelity scores present in metadata',
              'no metadata.proxy_scores — regenerate with data_collection_regime_model.py')
    else:
        bad, vestigial, thin = [], [], []
        for t, v in sorted(scores.items()):
            if v.get('thin'):
                thin.append(t)
                continue
            if v['fund_vol'] < PROXY_CASHLIKE_VOL:
                why = (f"TE {v['tracking_error']:.2%}/yr on a cash-like fund"
                       if v['tracking_error'] > PROXY_CASHLIKE_TE else None)
            elif v['corr'] < PROXY_CORR_FLOOR:
                why = f"corr {v['corr']:.3f} — different asset"
            elif not PROXY_BETA_BAND[0] <= v['beta'] <= PROXY_BETA_BAND[1]:
                why = f"beta {v['beta']:.2f}"
            else:
                why = None
            if why is None:
                continue
            if v['contributed_months'] <= PROXY_VESTIGIAL_MONTHS:
                vestigial.append(f"{t} ({v['proxy']}, {v['contributed_months']}mo)")
            else:
                bad.append(f"{t} proxied by {v['proxy']}: {why}, "
                           f"{v['contributed_months']} months in the series")
        # A proxy cannot supply more months than the ticker's whole history.
        # AVEE did (308 against 240) because the count predated the rolling-window
        # truncation, and nothing would have noticed.
        impossible = [f"{t}: {v['contributed_months']}mo contributed into a "
                      f"{(data.get(t) or {}).get('history_months', 0)}mo history"
                      for t, v in sorted(scores.items())
                      if v.get('contributed_months', 0) >
                      (data.get(t) or {}).get('history_months', 0)]
        check(not impossible, 'no proxy contributes more months than its history',
              '; '.join(impossible))
        check(not bad, f"every contributing proxy tracks its fund "
                       f"(corr >= {PROXY_CORR_FLOOR}, beta in {PROXY_BETA_BAND})",
              '; '.join(bad))
        if vestigial:
            print(f"         note: {len(vestigial)} vestigial proxy(ies) score badly but "
                  f"reach <= {PROXY_VESTIGIAL_MONTHS} months: {', '.join(vestigial)}")
        if thin:
            print(f"         note: too little overlap to score: {', '.join(thin)}")

    # ---- single stocks (ledger entry N10, unpublished) -----------------------------------
    # A single stock's mean IS its alpha, and it is in the universe because it
    # won. Built from its loadings it cannot lead the proxy by more than they and the
    # stated edge; carrying its historical mean, COST led VOO by ~7pp real.
    from montecarlo import factors
    from montecarlo.loaders import MARKET_PROXY
    fits = factors.load_fits()
    nominal, _ = factors.build_nominal(
        {t: v['expected_return'] for t, v in data.items() if t in MODEL_TICKERS}, fits)
    issues = factors.single_stock_problems(nominal, fits, MARKET_PROXY)
    check(not issues, f"single stocks earn their factor loadings plus the stated "
                      f"edge ({factors.SINGLE_STOCK_EDGE:.1%}), not their history",
          '; '.join(issues))

    return _finish(problems)


def _finish(problems):
    print("=" * 60)
    if problems:
        print(f"✗ {len(problems)} problem(s). Regenerate with data_collection or "
              f"investigate the splice/proxy for the flagged tickers.")
        return 1
    print("✓ All checks passed — generated parameters are sane.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
