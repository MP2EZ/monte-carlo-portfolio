#!/usr/bin/env python3
"""
Monte Carlo Regime-Switching Model: Unified Data Collection Pipeline
=====================================================================

Handles complete data collection for all 26 tickers.

FEATURES:
- Ken French academic data (AVUV, VTV, AVLV, MTUM, VFMO, VBR)
- AQR Quality data (JQUA)
- yfinance proxies (all tickers)
- 2-stage and 3-stage splicing
- Rolling 20-year window (240 months) for select tickers:
  * COST: actual stock data only (no proxies, no splicing)
  * AVEE: proxy-spliced data (DEMSX + actual) - early EM small-cap data
    too volatile/unrepresentative (1998-2002 post-Asian crisis: 111% vol)
- Blended proxy support (DFWIX: 78% DFIEX + 22% DFCEX; VT: 50% VTSMX + 50% VGTSX)
- Regime parameters (volatility percentiles, Markov transitions)
- Correlation matrices with 15-year rolling window (baseline + 3 regimes)
  * Window automatically updates: always uses most recent 15 years (180 months)
  * Example: Nov 2025 run → Nov 2010 to Nov 2025
  * Example: Mar 2026 run → Mar 2011 to Mar 2026
- Auto-runs tools/validate_parameters.py at the end to sanity-gate the output

OUTPUTS:
- ticker_parameters_regimes_v8.json
- correlation_baseline_N.npy  (N = number of tickers)
- correlation_low_N.npy
- correlation_medium_N.npy
- correlation_high_N.npy

USAGE:
    python3 data_collection_regime_model.py
    # then `python3 tools/validate_all.py` for the full structural+data+behavior gate

Author: Claude Code
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import pickle
from pathlib import Path
from datetime import datetime
from scipy.linalg import eigh
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent
REFERENCE_DIR = BASE_DIR / 'reference'
OUTPUT_DIR = BASE_DIR / 'output'
TIMESTAMP = datetime.now().strftime('%Y%m%d')

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# Ticker universe — single source of truth lives in loaders.MODEL_TICKERS.
# Imported here (not redefined) so the data pipeline, simulator, and tools can
# never drift apart. To change the universe, edit MODEL_TICKERS in loaders.py
# (see docs/RUNBOOK_tickers.md), then update PROXIES etc. below to match.
from montecarlo.loaders import (MODEL_TICKERS as TICKERS, SINGLE_STOCKS,
                     MARKET_PROXY, MARKET_REGIME_WINDOW, REGIME_WINDOW,
                     REGIME_CUTS, REGIME_MARKET_SHRINKAGE,
                     BASE_VOL_WINDOW_YEARS, CORR_WINDOW_YEARS_DEFAULT,
                     correlation_filename)

# yfinance proxies
PROXIES = {
    # NOT ^GSPC: that is the S&P 500 PRICE index and excludes dividends —
    # 8.25%/yr against the Ken French US market total return of 11.28%/yr over
    # 1970-2026, a 3.03pp gap that covered 72% of VOO's history. VOO is also
    # engine.MARKET_PROXY, so the error propagated to every MC_PREMIUM_SCALE
    # result and inflated every "premium over VOO" figure by ~2.5pp.
    # VFINX is the same index as a real fund: total return, 1980-02 onward, and
    # -0.80pp against the Ken French market where ^GSPC was -3.03pp.
    'VOO': 'VFINX', 'COST': None, 'JQUA': 'DFEOX', 'AVUV': 'DFSVX', 'VTI': 'VTSMX',
    'AVDE': 'DFIEX', 'AVEM': 'DFCEX', 'AVDV': 'DISVX', 'AVEE': 'DEMSX',
    # The four international proxies were ROTATED BY ONE — each fund carried the
    # next one's stand-in. VEA (developed) was backfilled with VEIEX (EMERGING)
    # at corr 0.856 and 10.5%/yr tracking error for 158 months, 41% of its
    # history; VWO (emerging) with DFISX (developed small) at corr 0.840. Found
    # 2026-09-10 by tools/proxy_score.py, which is the permanent gate. Every
    # replacement below is the diagonal of the cross-correlation matrix, and all
    # four improve — three of them also gain history.
    'VXUS': 'VGTSX',   # Total International Stock Index — corr 0.999, was VTMGX
    'VEA': 'DFALX',    # DFA Large Cap International — corr 0.995 from 1991-08.
                       # VTMGX is tighter (0.999) but starts 1999-09; 192 months
                       # of a 0.995 fit beats 95 months of a 0.999 one when the
                       # series' job is to set regime percentiles over all of it.
    'VWO': 'VEIEX',    # Emerging Markets Stock Index — corr 0.992, was DFISX
    'VSS': 'DFISX',    # DFA International Small Company — corr 0.979, was VGTSX
    'BND': 'VBMFX',
    'JPLD': None,      # Uses BLENDED_PROXIES (VGSH alone is treasury-only)
    'JPST': None,      # Uses BLENDED_PROXIES (SHV alone scored corr 0.397)
    # These three are Ken French-blended, and their KF windows run to their own
    # inceptions, so the middle layer is 0-1 months. VIVAX rather than ^GSPC
    # anyway: a vestigial entry is still a loaded gun if a window ever moves.
    'MTUM': 'VFINX', 'VFMO': 'VFINX', 'VTV': 'VIVAX',
    # AVLV's KF window ends 2019-09 and the fund starts 2021-10, so this layer
    # is 25 real months — not vestigial. DFSVX is DFA US SMALL Cap Value
    # standing in for a LARGE value fund: corr 0.922 at beta 0.76. DFLVX is the
    # same house's large-value fund, same 1993-03 start: corr 0.971, beta 1.02.
    'AVLV': 'DFLVX',
    'DFWIX': None,  # Uses BLENDED_PROXIES
    'VT': None,     # Uses BLENDED_PROXIES (VTSMX + VGTSX global blend)
    'VBR': 'VISVX', # Vanguard Small-Cap Value Index (1998+), middle splice layer
    'VUSB': 'VUBFX',# Vanguard Ultra-Short-Term Bond Fund (2015+), JPST analog
    'BSV': 'VBISX',  # Vanguard Short-Term Bond Index Investor (1994+), JPLD analog
    # T. Rowe Price Limited-Duration Inflation Focused Bond (2006-09+): 1-5y
    # TIPS, duration 2.7 against STIP's 2.3, and a real fund through 2008's
    # deflation scare, which a yield-based simulation would soften.
    'STIP': 'TRBFX',
    # Vanguard Intermediate-Term Treasury Investor (1991-10+). Added 2026-10-24
    # as the universe's first pure-Treasury fund: every other bond here moved
    # WITH stocks in the crisis regime (ledger entry 2026-10-24, unpublished).
    'VGIT': 'VFITX',
}

# What a proxy charged above the fund it stands in for, added back to the
# proxy's months only (annual, so /12 per month). TRBFX costs 0.48% against
# STIP's 0.03%; left in, 51 months of proxy would understate STIP's mean by
# ~0.1pp/yr.
PROXY_EXPENSE_GAP = {'STIP': 0.0045}


def splice_with_proxy(proxy_monthly, actual_monthly, expense_gap=0.0):
    """Proxy returns before the fund's first month, the fund's own from then on.

    The proxy is charged back `expense_gap` (annual) on its months only — the
    fund's own returns already net its own costs. Returns (spliced, n_proxy).
    """
    inception = actual_monthly.index[0]
    before = proxy_monthly[proxy_monthly.index < inception] + expense_gap / 12.0
    spliced = pd.concat([before, actual_monthly], axis=0)
    spliced = spliced[~spliced.index.duplicated(keep='last')].sort_index()
    return spliced, len(before)


def splice_layers(academic, proxy, actual):
    """Academic blend, then a fund proxy up to the real fund's first month,
    then the real fund. Returns (spliced, proxy months used).

    Each layer is truncated to where the next begins and the most
    authoritative goes LAST, so keep='last' lets the real fund win any overlap
    with a proxy and the proxy win over the academic blend. The old concat +
    duplicated(keep='first') was wrong twice over once actual data is fetched
    from inception: it preferred the PROXY over the real fund wherever they
    overlapped (161 months of real MTUM lost to a price index), and it left the
    index non-monotonic, which a later .loc[] slice cannot handle at all.
    """
    layers, n_proxy = [academic], 0
    if len(actual) > 0:
        if len(proxy) > 0:
            before = proxy[proxy.index < actual.index.min()]
            layers.append(before)
            n_proxy = len(before)
        layers.append(actual)
    elif len(proxy) > 0:
        layers.append(proxy)
    spliced = pd.concat(layers, axis=0).sort_index()
    return spliced[~spliced.index.duplicated(keep='last')], n_proxy


# Blended proxies: weighted combination of multiple funds as a single proxy
BLENDED_PROXIES = {
    'DFWIX': {
        'components': {'DFIEX': 0.78, 'DFCEX': 0.22},
        'description': 'DM + EM core equity blend matching DFWIX regional weights'
    },
    'VT': {
        'components': {'VTSMX': 0.50, 'VGTSX': 0.50},
        'description': 'Global US + ex-US blend (pre-2008 VT proxy); VGTSX 1996 inception bounds the history'
    },
    # Both of these are SINGLE-fund proxies replaced by blends, because neither
    # ultra-short fund has a clean analog with enough history. The pairing is
    # duration + credit: a treasury leg sets the (very small) rate sensitivity
    # and a credit leg supplies the spread exposure the treasury leg has none of.
    'JPST': {
        # SHV is 0-1y TREASURIES against JPST's ultra-short CREDIT: corr 0.397
        # over 112 months, and its 0.58%/yr vol is 44% below JPST's own 1.03%.
        # Halving it into MINT (PIMCO Enhanced Short Maturity, the direct
        # analog) gives corr 0.899, TE 0.46%/yr and vol 0.85%. Costs 34 months
        # of backfill — MINT starts 2009-12 — which is worth it at corr 0.40.
        'components': {'SHV': 0.50, 'MINT': 0.50},
        'description': 'Ultra-short treasury + ultra-short credit; MINT 2009-12 bounds the history'
    },
    'JPLD': {
        # VGSH alone matches the VOL almost exactly (1.55% against 1.59%) but is
        # treasury-only, so it under-earns JPLD by 1.31pp/yr at t=3.2 — the one
        # statistically resolved mean gap in the whole scoring table. 30% VFSTX
        # buys the credit exposure: corr 0.890 -> 0.926, TE 0.74% -> 0.67%, at
        # the cost of overstating vol by 11%. VGSH still bounds the start.
        'components': {'VGSH': 0.70, 'VFSTX': 0.30},
        'description': 'Short treasury + short investment-grade credit; VGSH 2009-12 bounds the history'
    }
}

# Ken French enhanced tickers
KF_TICKERS = {
    # VOO: the KF market covers 1970-02..1980-01, then VFINX takes over until
    # VOO's own inception. Without this the series starts in 1980 and inherits a
    # +1.29pp favourable-window bias — see _build_market_proxy.
    'VOO': {'start': '1970-02-01', 'end': '1980-01-31', 'source': 'market'},
    'AVUV': {'start': '1963-07-01', 'end': '1993-02-28', 'source': '32_portfolios'},
    'VTV': {'start': '1973-01-01', 'end': '2004-01-31', 'source': '25_size_value'},
    'AVLV': {'start': '1973-01-01', 'end': '2019-09-30', 'source': '25_size_value'},
    'MTUM': {'start': '1973-01-01', 'end': '2013-03-31', 'source': '25_size_momentum'},
    'VFMO': {'start': '1973-01-01', 'end': '2018-01-31', 'source': '25_size_momentum'},
    'JQUA': {'start': '1973-01-01', 'end': '2017-10-31', 'source': 'aqr_quality'},
    'VBR': {'start': '1973-01-01', 'end': '1998-05-31', 'source': '25_size_value'}
}

# Ken French blend weights (optimized from October scripts)
KF_BLENDS = {
    'AVUV': {'ME1 BM4 OP2': 0.46, 'ME1 BM2 OP4': 0.26, 'ME1 BM2 OP3': 0.28},
    'VTV': {'ME5 BM3': 0.40, 'ME5 BM4': 0.30, 'ME5 BM2': 0.30},
    'AVLV': {'ME4 BM4': 0.20, 'ME5 BM4': 0.40, 'ME4 BM2': 0.40},
    'MTUM': {'Big_High': 0.80, 'Size4_High': 0.20},
    'VFMO': {'Small_High': 0.15, 'Size2_High': 0.2125, 'Size3_High': 0.2125,
             'Size4_High': 0.2125, 'Big_High': 0.2125},
    'JQUA': {'P10 (high quality)': 0.60, 'P9': 0.15, 'P8': 0.25},
    # VBR ≈ CRSP US Small Cap Value: small size (ME1/ME2), moderate-high value
    # (BM3/BM4), centered like VTV's large-value blend but at small cap.
    'VBR': {'ME2 BM4': 0.40, 'ME1 BM4': 0.30, 'ME2 BM3': 0.30}
}

# Cost adjustments (annual rates)
COST_ADJUSTMENTS = {
    # The Ken French market is an academic index: gross of everything. VOO's
    # published expense ratio is 0.03%; no implementation drag for a fund that
    # tracks a cap-weighted index this closely.
    'VOO': {'expense': -0.0003, 'drag': 0.0},
    'AVUV': {'expense': -0.0025, 'drag': -0.0040},
    'VTV': {'expense': -0.0005, 'drag': 0.0},
    'AVLV': {'expense': -0.0015, 'drag': -0.0020},
    'MTUM': {'expense': -0.0015, 'drag': -0.0070},
    'VFMO': {'expense': -0.0013, 'drag': -0.0110},
    'JQUA': {'expense': -0.0024, 'drag': -0.0040},
    'VBR': {'expense': -0.0007, 'drag': 0.0}  # ER ≈ 0.07%, broad index → negligible drag
}


def net_of_costs(proxy, ticker):
    """An academic (gross) proxy charged its fund's costs.

    COST_ADJUSTMENTS holds them as NEGATIVE annual rates, so they are ADDED.
    Every builder subtracted them until 2026-10-26, which credited each proxy
    its costs instead: VFMO's history ran 1.23%/yr high where it should have
    run 1.23%/yr low.
    """
    costs = COST_ADJUSTMENTS[ticker]
    return proxy + (costs['expense'] + costs['drag']) / 12

# Tickers using rolling 20-year window (240 months) for expected return calculation
# because their early-period data is unrepresentative of current markets:
#   SINGLE_STOCKS (e.g. COST): a single company was a different business 20+ years ago
#   AVEE: post-Asian crisis EM small-cap data (1998-2002) has 111% annualized vol
# Single stocks are pulled in from loaders.SINGLE_STOCKS so the policy stays in one place.
ROLLING_WINDOW_TICKERS = {'AVEE'} | SINGLE_STOCKS

# Parameter uncertainty overrides (explicit values win over the defaults below).
#
# These are a hand-calibrated FACTOR-TILT band, not a sampling-error estimate:
# the ordering AVDE 0.014 < AVDV 0.018 < AVUV 0.020 tracks how factor-tilted each
# fund is, not how much history it has. Do NOT "fix" them by flooring at
# sigma/sqrt(years) — that measures a different quantity (how precisely the
# historical mean was measured), and base_volatility is computed over 2010+ while
# expected_return is the mean over 1973+, so the two cannot even be combined into
# a valid standard error.
#
# Known limitation: these bands are small relative to genuine uncertainty about
# whether factor premia persist. At AVUV 0.020 the model implies a 0.09%
# probability that AVUV has no premium at all. Widening them is a judgment call in
# the same spirit as the values below; see the A3 section of the v3.1 review notes (unpublished).
PARAM_STD_OVERRIDE = {
    'AVUV': 0.020, 'AVDE': 0.014, 'AVEM': 0.022,
    'AVDV': 0.018, 'AVEE': 0.025, 'VTV': 0.015, 'AVLV': 0.0135,
    'DFWIX': 0.016,  # Broad intl, moderate factor tilts (between AVDE 0.014 and AVDV 0.018)
    'VBR': 0.018     # US small-cap value factor band (≈ AVUV 0.020 / AVDV 0.018)
}

# Default parameter-uncertainty multipliers (× annualized vol) for tickers NOT in
# PARAM_STD_OVERRIDE. Single stocks get an inflated band for idiosyncratic /
# survivorship risk that empirical vol (from a survivor) understates.
DEFAULT_PARAM_STD_MULT = 0.15
SINGLE_STOCK_PARAM_STD_MULT = 0.25

# ============================================================================
# HELPERS
# ============================================================================

def fix_correlation_matrix(corr):
    """Fix correlation matrix to be positive semi-definite."""
    eigenvalues, eigenvectors = eigh(corr)
    eigenvalues[eigenvalues < 0] = 1e-8
    corrected = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    d = np.sqrt(np.diag(corrected))
    return corrected / np.outer(d, d)

# TIPS real yields, by tenor, from FRED. The only input in this model that is
# QUOTED rather than estimated — see roadmap item 9 and factors.RF_REAL.
TIPS_SERIES = {5: 'DFII5', 10: 'DFII10', 20: 'DFII20', 30: 'DFII30'}
TIPS_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id={}'


def fetch_tips_real_yields(timeout=30):
    """{tenor: {'latest', 'as_of', 'avg_1y', 'avg_full', 'min', 'max'}}.

    Both the spot quote AND its own history, because the spot alone is
    misleading here: the 30-year real yield has ranged -0.59% to 3.06% since
    2010 and today's 2.98% is at the top of that. A 58-year model that anchors
    on one day's print is making a bet, not reading a fact — so the averages
    ship alongside it and factors.RF_REAL documents the choice.

    Network failure is NOT fatal: this is an assumption the model already has a
    default for, not data it cannot run without.
    """
    import io
    import urllib.request
    out = {}
    for tenor, series in TIPS_SERIES.items():
        try:
            with urllib.request.urlopen(TIPS_URL.format(series), timeout=timeout) as r:
                df = pd.read_csv(io.BytesIO(r.read()))
            df.columns = ['date', 'v']
            df['v'] = pd.to_numeric(df['v'], errors='coerce')
            df['date'] = pd.to_datetime(df['date'])
            df = df.dropna().set_index('date').sort_index()
            if df.empty:
                continue
            recent = df.loc[df.index >= df.index[-1] - pd.DateOffset(years=1), 'v']
            out[str(tenor)] = {
                'series': series,
                'latest': float(df['v'].iloc[-1]) / 100.0,
                'as_of': f"{df.index[-1]:%Y-%m-%d}",
                'avg_1y': float(recent.mean()) / 100.0,
                'avg_full': float(df['v'].mean()) / 100.0,
                'min': float(df['v'].min()) / 100.0,
                'max': float(df['v'].max()) / 100.0,
                'history_from': f"{df.index[0]:%Y-%m-%d}",
            }
        except Exception as e:
            print(f"    ⚠ {series}: {e.__class__.__name__} — skipped")
    return out


REGIMES = ('low', 'medium', 'high')


def _transition_matrix(labels):
    """Month-to-month Markov transitions from a sequence of regime labels."""
    c = {a: {b: 0.0 for b in REGIMES} for a in REGIMES}
    for a, b in zip(labels[:-1], labels[1:]):
        c[a][b] += 1.0
    out = {}
    for a in REGIMES:
        tot = sum(c[a].values())
        out[a] = ({b: n / tot for b, n in c[a].items()} if tot
                  else {'low': 0.33, 'medium': 0.34, 'high': 0.33})
    return out


def vol_regime_labels(returns, window):
    """Each month labelled low / medium / high by its trailing `window`-month
    volatility, cut at REGIME_CUTS over the series' own history. Returns
    (labels, rolling vol, (low cut, high cut))."""
    rv = (returns.rolling(window=window).std() * np.sqrt(12)).dropna()
    lo, hi = rv.quantile(REGIME_CUTS[0]), rv.quantile(REGIME_CUTS[1])
    labels = pd.Series(np.where(rv <= lo, 'low', np.where(rv <= hi, 'medium', 'high')),
                       index=rv.index)
    return labels, rv, (lo, hi)


def classify_regimes(returns):
    """(labels, per-regime realised volatility) for one return series.

    Volatility is the REALISED volatility of the months in each bucket, not a
    percentile of the rolling estimator — see loaders.REGIME_CUTS for why, and
    for why that had to change at the same time as the window.
    """
    labels, rv, _ = vol_regime_labels(returns, REGIME_WINDOW)
    vols = {}
    for r in REGIMES:
        idx = labels[labels == r].index
        # A bucket can be empty for a near-constant series; fall back to the
        # rolling estimator's own median there rather than emitting a NaN.
        v = returns.loc[idx].std() * np.sqrt(12) if len(idx) > 2 else float(rv.median())
        vols[r] = float(v) if np.isfinite(v) else float(rv.median())
    # Regimes must stay ORDERED even if a thin bucket makes the raw estimate
    # cross over; validate_parameters enforces this and a crossover there means
    # the labels have stopped meaning anything.
    lo_v, md_v, hi_v = (vols['low'], vols['medium'], vols['high'])
    vols['medium'] = max(md_v, lo_v)
    vols['high'] = max(hi_v, vols['medium'])
    return labels, vols


def conditional_transitions(labels, market_by_month, k=None):
    """P(next ticker regime | current ticker regime, MARKET regime next month).

    Grouped by the market regime being moved INTO, which is the order the engine
    draws in: market first, then each ticker conditioned on it.

    Shrunk toward the ticker's own unconditional row by `k` pseudo-counts. That
    is not a refinement — the raw conditional estimate scores -2.36 out of sample
    against the unconditional -0.53, because cells are nearly empty (VOO shows
    P(medium->high) = 1.00 off ONE observation). At k=10 it scores -0.48.
    """
    k = REGIME_MARKET_SHRINKAGE if k is None else k
    idx = labels.index.to_period('M')
    lab = pd.Series(labels.values, index=idx)
    lab = lab[~lab.index.duplicated()]
    shared = lab.index.intersection(market_by_month.index)
    uncond = _transition_matrix(labels.values)
    if len(shared) < 24:
        return {r: uncond for r in REGIMES}, 0
    lab, mkt = lab.loc[shared], market_by_month.loc[shared]
    counts = {r: {a: {b: 0.0 for b in REGIMES} for a in REGIMES} for r in REGIMES}
    for i in range(len(shared) - 1):
        counts[mkt.iloc[i + 1]][lab.iloc[i]][lab.iloc[i + 1]] += 1.0
    out = {}
    for r in REGIMES:
        out[r] = {}
        for a in REGIMES:
            row = counts[r][a]
            blended = {b: row[b] + k * uncond[a][b] for b in REGIMES}
            tot = sum(blended.values())
            # A CELL CAN BE COMPLETELY EMPTY. State `a` may never occur moving
            # into market regime `r` — and at k=0 the shrinkage term is gone
            # too, so `tot` is zero and this divided by it. Unreachable at the
            # shipped k=10, because the unconditional row sums to 1 and floors
            # `tot` at k; reachable the moment anyone scores k=0, which is the
            # raw-conditional comparison this shrinkage was chosen by.
            out[r][a] = ({b: v / tot for b, v in blended.items()} if tot
                         else dict(uncond[a]))
    return out, int(len(shared))


def score_proxy(actual, proxy, contrib, label):
    """How well the backfill tracks the fund it stands in for.

    A proxy covers the months BEFORE the real fund existed, so it can only be
    checked on the months AFTER, which the splice discards. `contrib` is how
    many months of proxy actually survive into the spliced series — the column
    that separates a live defect from a vestigial table entry.

    Scored here rather than in the validator because this is where both series
    are already in memory; `tools/validate_parameters.py` gates the stored
    result offline so `validate_all.py` still runs without a network.

    This exists because the four international proxies were rotated by one and
    nobody noticed for as long as the table had existed. See tools/proxy_score.py.
    """
    j = pd.concat([actual.rename('a'), proxy.rename('p')], axis=1).dropna()
    j = j[j.index >= actual.index.min()]
    if len(j) < 12:
        return {'proxy': label, 'overlap_months': int(len(j)),
                'contributed_months': int(contrib), 'thin': True}
    a, q = j['a'].values, j['p'].values
    d = a - q
    return {
        'proxy': label,
        'overlap_months': int(len(j)),
        'contributed_months': int(contrib),
        'corr': float(np.corrcoef(a, q)[0, 1]),
        'beta': float(np.cov(a, q, ddof=1)[0, 1] / np.var(q, ddof=1)),
        'tracking_error': float(d.std(ddof=1) * np.sqrt(12)),
        'fund_vol': float(a.std(ddof=1) * np.sqrt(12)),
        'proxy_vol': float(q.std(ddof=1) * np.sqrt(12)),
        'mean_gap': float(d.mean() * 12),
    }


# One fetch per (ticker, start, end) per process: proxies are shared (VFINX
# stands in for VOO, MTUM and VFMO; the DFWIX and VT blend components double as
# single-fund proxies). Callers get a copy, so none can alter another's series.
_DOWNLOADS = {}


def download_with_retry(ticker, start=None, end=None, max_retries=3):
    """Download ticker data with retries."""
    key = (ticker, start, end)
    if key in _DOWNLOADS:
        return _DOWNLOADS[key].copy()
    for attempt in range(max_retries):
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            if not data.empty:
                # Extract Close column, ensuring we get a Series
                close_data = data['Close']
                if isinstance(close_data, pd.DataFrame):
                    close_data = close_data.iloc[:, 0]  # Take first column if DataFrame
                _DOWNLOADS[key] = close_data
                return close_data.copy()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2)
    raise Exception(f"Failed to download {ticker}")

# ============================================================================
# KEN FRENCH PARSER
# ============================================================================

def _month_end(index):
    """Date a monthly table at month-END, as every yfinance series is.

    Ken French files say `199002` for February; parsed with '%Y%m' that is
    1990-02-01. The correlation step snaps each series onto month-end dates by
    NEAREST within 5 days, which files 1990-02-01 under 1990-01-31 — so until
    2026-10-25 AVLV, VFMO, MTUM, VTV, VBR and VOO's 1970s were correlated with
    the PREVIOUS month's market (AVLV-VOO -0.03 in calm months, +0.88 aligned).
    parse_32_portfolios always did this; the other parsers did not.
    """
    return index.to_period('M').to_timestamp('M')


class KenFrenchParser:
    """Parse Ken French CSV files."""

    def __init__(self, reference_dir):
        self.ref_dir = Path(reference_dir)

    def parse_32_portfolios(self):
        """Parse 32 portfolios (Size x Value x Profitability)."""
        file = self.ref_dir / '32_Portfolios_ME_BEME_OP_2x4x4.csv'
        df = pd.read_csv(file, skiprows=23, nrows=1500)
        df = df[df['Unnamed: 0'].astype(str).str.match(r'^\d{6}$', na=False)]
        df['DATE'] = pd.to_datetime(df['Unnamed: 0'], format='%Y%m')
        df = df.set_index('DATE').drop(columns=['Unnamed: 0'])
        for col in df.columns:
            df[col] = df[col].astype(float) / 100
        df.index = df.index.to_period('M').to_timestamp('M')
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        return df

    def parse_25_size_value(self):
        """Parse 25 portfolios (Size x Book-to-Market)."""
        file = self.ref_dir / '25_Portfolios_5x5.csv'
        df = pd.read_csv(file, skiprows=15, nrows=1500)
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'].astype(str).str.strip(), format='%Y%m', errors='coerce')
        df = df[df['Date'].notna()].set_index('Date')
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # Sentinels masked BEFORE scaling. -99.99/100 is -0.9998999999999999,
        # which never equals the literal -0.9999, so the old post-scale
        # `.replace([-0.9999, ...])` silently never fired. It left 4 months of
        # -99.99% in 25_size_momentum (Big_Low, 1927-1938 — outside every bucket
        # and window in use, so nothing shipped moved) and 60 in the emerging
        # factor file, where RMW read -46.50%/yr. `<= -99` catches -99.99, -999
        # and -9999 without depending on float equality at all.
        df = df.mask(df <= -99.0)
        df = df / 100.0
        # Rename columns to ME/BM format
        size_labels = ['ME1', 'ME2', 'ME3', 'ME4', 'ME5']
        bm_labels = ['BM1', 'BM2', 'BM3', 'BM4', 'BM5']
        new_cols = [f'{s} {b}' for s in size_labels for b in bm_labels]
        df.columns = new_cols[:len(df.columns)]
        df.index = _month_end(df.index)
        # The Ken French files concatenate several tables — value-weighted
        # monthly, then equal-weighted monthly, then annual — and `nrows` cannot
        # separate them. 308 EQUAL-weighted months (1926-07..1952-02) were being
        # appended to the value-weighted series and interleaved by the sort,
        # giving two different returns for the same month. Same guard
        # parse_32_portfolios has always had.
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        return df

    def parse_25_size_momentum(self):
        """Parse 25 portfolios (Size x Momentum)."""
        file = self.ref_dir / '25_Portfolios_ME_Prior_12_2.csv'
        df = pd.read_csv(file, skiprows=11, nrows=1185)
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'].astype(str).str.strip(), format='%Y%m', errors='coerce')
        df = df[df['Date'].notna()].set_index('Date')
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # Sentinels masked BEFORE scaling: see parse_25_size_value.
        df = df.mask(df <= -99.0)
        df = df / 100.0
        size_labels = ['Small', 'Size2', 'Size3', 'Size4', 'Big']
        mom_labels = ['Low', 'Mom2', 'Mom3', 'Mom4', 'High']
        new_cols = [f'{s}_{m}' for s in size_labels for m in mom_labels]
        df.columns = new_cols[:len(df.columns)]
        df.index = _month_end(df.index)
        # Duplicate-month guard: see parse_25_size_value.
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        return df

    def _parse_factor_file(self, filename, skiprows, columns):
        """Shared reader for the Fama/French factor CSVs.

        Each file concatenates a monthly table and an annual one. The annual
        rows carry 4-digit years, which fail `format='%Y%m'` and drop out — but
        the duplicate guard stays anyway, for the same reason the 25-portfolio
        parsers now carry one: `nrows`-style assumptions about these files have
        already been wrong once.
        """
        df = pd.read_csv(self.ref_dir / filename, skiprows=skiprows)
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'].astype(str).str.strip(),
                                    format='%Y%m', errors='coerce')
        df = df[df['Date'].notna()].set_index('Date')
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # Sentinels masked BEFORE scaling: see parse_25_size_value.
        df = df.mask(df <= -99.0)
        df = df / 100.0
        df.columns = columns[:len(df.columns)]
        df.index = _month_end(df.index)
        df = df[~df.index.duplicated(keep='first')]
        return df.sort_index()

    def parse_ff5_factors(self):
        """Mkt-RF, SMB, HML, RMW, CMA and the risk-free rate. Monthly, 1963-07+.

        RF is why this file rather than corner-construction from the
        25-portfolio sets: `expected_return = rf + sum(beta * premium)` has no
        other source for it. RMW matters because the Avantis funds screen on
        profitability, which the 3-factor set cannot express.
        """
        return self._parse_factor_file(
            'F-F_Research_Data_5_Factors_2x3.csv', 3,
            ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'rf'])

    def parse_ff3_factors(self):
        """Mkt-RF, SMB, HML and the risk-free rate. Monthly, 1926-07+.

        The 5-factor file starts 1963-07 because RMW and CMA need accounting
        data that does not exist earlier. These three do not, and Ken French
        publishes them back to 1926 — 100 years against 63, which is worth about
        20% off the market premium's standard error for one download.

        Deliberately a SEPARATE window, not a replacement: mixing it into the
        5-factor frame would silently give rmw and cma 437 months of NaN. See
        factors.PREMIUM_WINDOW, which records what each premium was measured on.
        """
        return self._parse_factor_file(
            'F-F_Research_Data_Factors.csv', 3, ['mkt_rf', 'smb', 'hml', 'rf'])

    def parse_developed_ex_us_factors(self):
        """Developed ex-US 5 factors + momentum, USD. Monthly, 1990-07+.

        EX-US deliberately: the plain "Developed" set includes the US and would
        double-count against parse_ff5_factors. Momentum is a separate file and
        starts four months later, so it is joined rather than assumed aligned.
        """
        f = self._parse_factor_file(
            'Developed_ex_US_5_Factors.csv', 6,
            ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'rf'])
        m = self._parse_factor_file('Developed_ex_US_MOM_Factor.csv', 6, ['mom'])
        return f.join(m, how='left')

    def parse_emerging_factors(self):
        """Emerging 5 factors + momentum, USD. Monthly, 1989-07+.

        RMW and CMA are -99.99 (missing) for the earliest months; the shared
        reader maps those to NaN, so callers must dropna rather than assume the
        frame is rectangular from its first row.
        """
        f = self._parse_factor_file(
            'Emerging_5_Factors.csv', 6,
            ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'rf'])
        m = self._parse_factor_file('Emerging_MOM_Factor.csv', 6, ['mom'])
        return f.join(m, how='left')

    def parse_momentum_factor(self):
        """UMD. Monthly, 1927-01+. For MTUM and VFMO loadings."""
        return self._parse_factor_file(
            'F-F_Momentum_Factor.csv', 12, ['mom'])

# ============================================================================
# AQR PARSER
# ============================================================================

class AQRParser:
    """Parse AQR Quality data."""

    def __init__(self, reference_dir):
        self.ref_dir = Path(reference_dir)

    def parse_quality_10(self):
        """Parse Quality 10 portfolios."""
        file = self.ref_dir / 'Quality Minus Junk 10 QualitySorted Portfolios Monthly.xlsx'
        df = pd.read_excel(file, sheet_name='10 Portfolios Formed on Quality', skiprows=18)
        # First column should be 'DATE'
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        df = df[df['DATE'].notna()].set_index('DATE')
        # AQR ships this workbook ALREADY IN DECIMAL (raw P10 values look like
        # 0.017519, -0.056645), unlike the Ken French CSVs which are in percent
        # (5.8276, -1.7006). Dividing by 100 here scaled JQUA's whole 1973-2017
        # segment down 100x, which is 84% of its history:
        #   regime vols       0.09% / 0.16%   (should be ~9% / 15%)
        #   expected return  -0.08% real      (should be ~5% real)
        # base_volatility looked plausible at 12.62% only because it is computed
        # from 2010+, which is mostly actual fund data — so the defect hid behind
        # one healthy-looking number.
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.sort_index()
        # Keep original column names (e.g., 'P10 (high quality)', 'P9', 'P8', etc.)
        return df

# ============================================================================
# PROXY BUILDER
# ============================================================================

class ProxyBuilder:
    """Build Ken French and AQR proxies for enhanced tickers."""

    def __init__(self, kf_parser, aqr_parser):
        self.kf = kf_parser
        self.aqr = aqr_parser
        self._kf_data = {}

    def build_proxy(self, ticker):
        """Build appropriate proxy for ticker."""
        if ticker not in KF_TICKERS:
            return None

        config = KF_TICKERS[ticker]
        source = config['source']

        if source == '32_portfolios':
            return self._build_avuv_proxy()
        elif source == '25_size_value':
            return self._build_size_value_proxy(ticker)
        elif source == '25_size_momentum':
            return self._build_momentum_proxy(ticker)
        elif source == 'aqr_quality':
            return self._build_quality_proxy()
        elif source == 'market':
            return self._build_market_proxy(ticker)
        else:
            raise ValueError(f"Unknown source: {source}")

    def _build_market_proxy(self, ticker):
        """US market TOTAL return, for the years before a fund proxy exists.

        VOO's oldest fund proxy, VFINX, starts 1980-02 — which quietly began the
        estimate at the start of a historic bull market. Measured: swapping
        ^GSPC for VFINX moved VOO +2.46pp, but only +1.25pp of that was the
        missing dividend; +1.29pp was dropping the 1970s. VOO is
        engine.MARKET_PROXY, so a favourable start date there propagates to
        every premium in the model.

        The Ken French market factor is total-return and reaches back to
        1963-07, so it covers the gap. It is the whole CRSP market rather than
        the S&P 500, which overstates VOO slightly in a decade when small caps
        did well — a much smaller bias than the one it removes.
        """
        f = self.kf.parse_ff5_factors()
        proxy = (f['mkt_rf'] + f['rf']).loc[
            KF_TICKERS[ticker]['start']:KF_TICKERS[ticker]['end']]
        return net_of_costs(proxy, ticker)

    def _build_avuv_proxy(self):
        """Build AVUV 3-way blend from 32 portfolios."""
        df = self.kf.parse_32_portfolios()
        weights = KF_BLENDS['AVUV']

        proxy = sum(weights[col] * df[col] for col in weights.keys())
        proxy = proxy.loc[KF_TICKERS['AVUV']['start']:KF_TICKERS['AVUV']['end']]

        return net_of_costs(proxy, 'AVUV')

    def _build_size_value_proxy(self, ticker):
        """Build VTV/AVLV from 25 size/value portfolios."""
        if '25_size_value' not in self._kf_data:
            self._kf_data['25_size_value'] = self.kf.parse_25_size_value()
        df = self._kf_data['25_size_value']

        weights = KF_BLENDS[ticker]
        proxy = sum(weights[col] * df[col] for col in weights.keys())
        proxy = proxy.loc[KF_TICKERS[ticker]['start']:KF_TICKERS[ticker]['end']]

        return net_of_costs(proxy, ticker)

    def _build_momentum_proxy(self, ticker):
        """Build MTUM/VFMO from 25 size/momentum portfolios."""
        if '25_size_momentum' not in self._kf_data:
            self._kf_data['25_size_momentum'] = self.kf.parse_25_size_momentum()
        df = self._kf_data['25_size_momentum']

        weights = KF_BLENDS[ticker]
        proxy = sum(weights[col] * df[col] for col in weights.keys())
        proxy = proxy.loc[KF_TICKERS[ticker]['start']:KF_TICKERS[ticker]['end']]

        return net_of_costs(proxy, ticker)

    def _build_quality_proxy(self):
        """Build JQUA from AQR quality data."""
        df = self.aqr.parse_quality_10()
        weights = KF_BLENDS['JQUA']

        proxy = sum(weights[col] * df[col] for col in weights.keys())
        proxy = proxy.loc[KF_TICKERS['JQUA']['start']:KF_TICKERS['JQUA']['end']]

        return net_of_costs(proxy, 'JQUA')

# Below this many shared months a pair takes its baseline entry; see
# regime_correlations.
MIN_PAIR_MONTHS = 24


def regime_correlations(corr_df, market_labels):
    """The baseline correlation matrix and one per market regime.

    PAIRWISE, so a ticker that starts late shortens only its own pairs; a pair
    with fewer than MIN_PAIR_MONTHS shared months takes its baseline entry, and
    a regime with 10 or fewer months takes the whole baseline matrix. Each
    comes out PSD with a unit diagonal (fix_correlation_matrix repairs the mild
    indefiniteness pairwise estimation can introduce). Months are labelled by
    the nearest market-regime label within 5 days.
    Returns (baseline, {regime: matrix}, {regime: months}).
    """
    def regime_corr(frame, fallback=None):
        C = frame.corr(min_periods=MIN_PAIR_MONTHS).values
        thin = int(np.isnan(C)[np.triu_indices(len(C), 1)].sum())
        if thin:
            src = fallback if fallback is not None else np.eye(len(C))
            C = np.where(np.isnan(C), src, C)
            print(f"      ({thin} pair(s) under {MIN_PAIR_MONTHS} months — "
                  f"took the {'baseline' if fallback is not None else 'identity'} entry)")
        np.fill_diagonal(C, 1.0)
        return fix_correlation_matrix(C)

    window_labels = market_labels.reindex(
        corr_df.index, method='nearest', tolerance=pd.Timedelta('5 days'))
    baseline = regime_corr(corr_df)
    by_regime, months = {}, {}
    for r in REGIMES:
        idx = window_labels[window_labels == r].index
        months[r] = len(idx)
        by_regime[r] = (regime_corr(corr_df.loc[idx], baseline) if len(idx) > 10
                        else baseline)
    return baseline, by_regime, months


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(corr_window_years=CORR_WINDOW_YEARS_DEFAULT, corr_only=False):
    """The full pipeline, or with `corr_only` just the correlation matrices at
    `corr_window_years`, written as their own file set (loaders.correlation_filename)
    so the JSON and the default matrices are left exactly as they were."""
    print("=" * 80)
    print("UNIFIED DATA COLLECTION PIPELINE")
    print("=" * 80)
    print(f"Timestamp: {TIMESTAMP}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Initialize parsers
    print("[Step 1/7] Initializing parsers...")
    kf_parser = KenFrenchParser(REFERENCE_DIR)
    aqr_parser = AQRParser(REFERENCE_DIR)
    proxy_builder = ProxyBuilder(kf_parser, aqr_parser)
    print("  ✓ Parsers ready\n")

    # Build Ken French/AQR proxies
    print("[Step 2/7] Building Ken French & AQR proxies...")
    kf_proxies = {}
    for ticker in KF_TICKERS.keys():
        print(f"  Building {ticker} proxy...")
        kf_proxies[ticker] = proxy_builder.build_proxy(ticker)
        print(f"    ✓ {len(kf_proxies[ticker])} months")
    print()

    # Splice data together
    print("[Step 3/7] Splicing data...")
    spliced_returns = {}
    proxy_pairs = {}   # ticker -> (actual, proxy, contributed_months, label)

    # Ken French enhanced tickers (3-stage or 2-stage splice)
    for ticker in KF_TICKERS.keys():
        print(f"  Splicing {ticker}...")
        kf_proxy = kf_proxies[ticker]

        # Download yfinance proxy (middle layer)
        proxy_ticker = PROXIES[ticker]
        if proxy_ticker:
            proxy_data = download_with_retry(proxy_ticker, start=KF_TICKERS[ticker]['end'])
            proxy_monthly = proxy_data.resample('ME').last().pct_change().dropna()

            kf_series = pd.Series(kf_proxy.values, index=kf_proxy.index, name=ticker)
            proxy_series = pd.Series(proxy_monthly.values, index=proxy_monthly.index, name=ticker)
        else:
            kf_series = pd.Series(kf_proxy.values, index=kf_proxy.index, name=ticker)
            proxy_series = pd.Series([], name=ticker)

        # Download actual ETF data.
        #
        # From 1970, NOT a hardcoded 2010: the old start date silently discarded
        # real fund history in favour of a proxy. VTV lost 72 months of its own
        # returns (inception 2004-01) to the ^GSPC PRICE index, and VBR lost 72
        # to VISVX. Fetching from the fund's inception also collapses VTV's
        # ^GSPC exposure to zero, since its Ken French window ends exactly where
        # the real fund begins.
        actual_data = download_with_retry(ticker, start='1970-01-01')
        actual_monthly = actual_data.resample('ME').last().pct_change().dropna()
        actual_series = pd.Series(actual_monthly.values, index=actual_monthly.index, name=ticker)

        spliced, n_proxy = splice_layers(kf_series, proxy_series, actual_series)

        spliced_returns[ticker] = spliced
        if len(proxy_series) > 0 and len(actual_series) > 0:
            proxy_pairs[ticker] = (actual_series, proxy_series, n_proxy,
                                   proxy_ticker)
        print(f"    ✓ {len(spliced)} months total")

    # Blended proxy tickers (e.g., DFWIX = 78% DFIEX + 22% DFCEX)
    for ticker, blend_config in BLENDED_PROXIES.items():
        print(f"  Splicing {ticker} (blended proxy)...")
        components = blend_config['components']

        # Download all component funds
        component_returns = {}
        for comp_ticker in components:
            comp_data = download_with_retry(comp_ticker, start='1970-01-01')
            comp_monthly = comp_data.resample('ME').last().pct_change().dropna()
            component_returns[comp_ticker] = comp_monthly
            print(f"    Component {comp_ticker}: {len(comp_monthly)} months")

        # Align to common dates and blend
        comp_df = pd.DataFrame(component_returns).dropna()
        blended = sum(weight * comp_df[comp_ticker] for comp_ticker, weight in components.items())
        blended.name = ticker
        print(f"    Blended proxy: {len(blended)} months ({blended.index[0].strftime('%Y-%m')} to {blended.index[-1].strftime('%Y-%m')})")

        # Download actual fund data
        actual_data = download_with_retry(ticker, start='1970-01-01')
        actual_monthly = actual_data.resample('ME').last().pct_change().dropna()
        # Splice: blended proxy (before inception) + actual (after inception)
        spliced, n_proxy = splice_with_proxy(blended, actual_monthly)

        spliced_returns[ticker] = spliced
        proxy_pairs[ticker] = (
            actual_monthly, blended, n_proxy,
            ' + '.join(f'{w:.0%} {c}' for c, w in components.items()))
        print(f"    ✓ Proxy: {n_proxy} months, Actual: {len(actual_monthly)} months, Total: {len(spliced)} months")

    # Regular tickers (2-stage splice: yfinance proxy + actual)
    for ticker in TICKERS:
        if ticker in spliced_returns:
            continue  # Already handled by Ken French

        print(f"  Splicing {ticker}...")

        # Check if this ticker has a proxy
        proxy_ticker = PROXIES.get(ticker)

        if proxy_ticker:
            # 2-stage splice: proxy + actual
            try:
                # Download proxy data
                proxy_data = download_with_retry(proxy_ticker, start='1970-01-01')
                proxy_monthly = proxy_data.resample('ME').last().pct_change().dropna()

                # Download actual ETF data
                actual_data = download_with_retry(ticker, start='1970-01-01')
                actual_monthly = actual_data.resample('ME').last().pct_change().dropna()

                # Splice: proxy (before inception, charged back any fee gap)
                # + actual (from inception).
                spliced, n_proxy = splice_with_proxy(
                    proxy_monthly, actual_monthly, PROXY_EXPENSE_GAP.get(ticker, 0.0))

                proxy_pairs[ticker] = (actual_monthly, proxy_monthly,
                                       n_proxy, proxy_ticker)
                print(f"    ✓ Proxy: {n_proxy} months, Actual: {len(actual_monthly)} months, Total: {len(spliced)} months")
            except Exception as e:
                print(f"    ⚠ Proxy splice failed: {e}, using actual only")
                data = download_with_retry(ticker, start='1970-01-01')
                spliced = data.resample('ME').last().pct_change().dropna()
        else:
            # No proxy - use actual data only
            data = download_with_retry(ticker, start='1970-01-01')
            spliced = data.resample('ME').last().pct_change().dropna()
            print(f"    ✓ {len(spliced)} months (no proxy)")

        # Rolling 20-year window for select tickers
        if ticker in ROLLING_WINDOW_TICKERS and len(spliced) > 240:
            pre_window = len(spliced)
            spliced = spliced.iloc[-240:]  # Keep most recent 240 months
            print(f"    → Rolling 20-year window applied: {pre_window} → {len(spliced)} months")

        spliced_returns[ticker] = spliced

    # The spliced series are the input to EVERY parameter below — expected
    # returns, volatility, regimes, transitions and all four correlation
    # matrices. Auditing any of them offline means having them, and rebuilding
    # them outside this function means reimplementing the splice. So: dump on
    # request, and never as a side effect of a normal run.
    if os.environ.get('MC_DUMP_SPLICED'):
        dump = Path(os.environ['MC_DUMP_SPLICED'])
        with open(dump, 'wb') as fh:
            pickle.dump(spliced_returns, fh)
        print(f"  (dumped spliced series to {dump})")

    print()

    # ---- THE MARKET REGIME ---------------------------------------------------
    # One definition, used by every consumer: the per-ticker chains below, the
    # correlation buckets in Step 6, and engine.py at simulation time. Until
    # 2026-09-10 there were three and they disagreed — the pipeline labelled a
    # month by `corr_df.std(axis=1)`, which is CROSS-SECTIONAL dispersion ACROSS
    # the 25 funds and not volatility over time (a crash where everything falls
    # together has near-zero dispersion), while engine.dominant_regime took a
    # MAJORITY VOTE of the per-ticker regimes, which did not separate medium from
    # high at all.
    market_labels, _, (mkt_p30, mkt_p70) = vol_regime_labels(
        spliced_returns[MARKET_PROXY], MARKET_REGIME_WINDOW)
    market_transitions = _transition_matrix(market_labels.values)

    print(f"[Step 4/7] Market regime ({MARKET_PROXY}, {MARKET_REGIME_WINDOW}m vol): "
          f"cuts {mkt_p30:.1%} / {mkt_p70:.1%}")
    for r in REGIMES:
        idx = market_labels[market_labels == r].index
        stay = market_transitions[r][r]
        print(f"    {r:<7}{len(idx):>5} months  stay {stay:.3f} "
              f"(~{1 / max(1 - stay, 1e-9):.0f} month regimes)  "
              f"realised vol {spliced_returns[MARKET_PROXY].loc[idx].std() * np.sqrt(12):.1%}")
    # Month-indexed, for joining against each ticker's own labels below.
    market_by_month = pd.Series(market_labels.values,
                                index=market_labels.index.to_period('M'))
    market_by_month = market_by_month[~market_by_month.index.duplicated()]
    print()

    # Calculate regime parameters
    print("[Step 5/7] Calculating regime parameters...")
    ticker_params = {}

    for ticker in TICKERS:
        print(f"  Processing {ticker}...")
        returns = spliced_returns[ticker]

        # Expected return (arithmetic mean from 1973-01-01 onwards to match baseline)
        expected_return = returns.loc['1973-01-01':].mean() * 12  # Annualized

        # Base volatility (2010-2025 period)
        # A ROLLING window, not a frozen date. This was '2010-01-01' — 15 years
        # before the tool was built in 2025 — so the intent was a 15-year
        # lookback. Left as a literal it had already become 16 years and would
        # widen by one every year, so a statistic meant to describe the RECENT
        # regime was quietly absorbing more history each regen.
        #
        # base_volatility is currently DEAD (nothing reads it; the simulation uses
        # VOLATILITY_REGIMES), so this changes no result today. See loaders.BASE_VOL_WINDOW_YEARS.
        cutoff = returns.index[-1] - pd.DateOffset(years=BASE_VOL_WINDOW_YEARS)
        recent_returns = returns.loc[cutoff:]
        base_vol = recent_returns.std() * np.sqrt(12)  # Annualized

        # Regime labels and per-regime volatility, on a 12-month window with
        # within-bucket volatilities. Both changed together on 2026-09-10 and
        # had to: see loaders.REGIME_WINDOW.
        regime_labels, regime_vols = classify_regimes(returns)
        regime_transitions = _transition_matrix(regime_labels.values)
        # ...and the same chain conditioned on the MARKET regime, so the engine
        # can produce the joint tail (correlations spiking WHILE assets are
        # volatile) that independent draws could not.
        cond_transitions, cond_months = conditional_transitions(
            regime_labels, market_by_month)

        # Parameter uncertainty: explicit override wins; otherwise a multiple of
        # volatility — inflated for single stocks (idiosyncratic/survivorship risk).
        if ticker in PARAM_STD_OVERRIDE:
            param_std = PARAM_STD_OVERRIDE[ticker]
        else:
            mult = SINGLE_STOCK_PARAM_STD_MULT if ticker in SINGLE_STOCKS else DEFAULT_PARAM_STD_MULT
            param_std = returns.std() * np.sqrt(12) * mult

        # Store parameters
        ticker_params[ticker] = {
            'expected_return': float(expected_return),
            'base_volatility': float(base_vol),
            'parameter_std': float(param_std),
            'volatility_regimes': {r: float(regime_vols[r]) for r in REGIMES},
            'regime_transitions': regime_transitions,
            'regime_transitions_by_market': cond_transitions,
            'regime_months': {r: int((regime_labels == r).sum()) for r in REGIMES},
            'history_months': len(returns),
            'history_start': returns.index[0].strftime('%Y-%m'),
            'history_end': returns.index[-1].strftime('%Y-%m')
        }

        print(f"    ✓ μ={expected_return*100:.2f}%, σ={base_vol*100:.2f}%, "
              f"regimes {regime_vols['low']*100:.1f}/{regime_vols['medium']*100:.1f}/"
              f"{regime_vols['high']*100:.1f}%, {len(returns)} months")

    print()

    # Calculate correlation matrices
    print("[Step 6/7] Calculating correlation matrices...")

    # Use spliced returns for a rolling window (15 years = 180 months by default)
    # Calculate window dynamically based on current date
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=corr_window_years)
    standard_dates = pd.date_range(start=start_date, end=end_date, freq='ME')

    print(f"  Using {corr_window_years}-year rolling window: {len(standard_dates)} months")
    print(f"    {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")

    corr_returns = {}
    for ticker in TICKERS:
        # Use the spliced returns we already calculated
        returns = spliced_returns[ticker]
        # Get the window ending at current month
        returns_window = returns.loc[start_date:end_date]

        # Reindex to standard month-end dates
        # This aligns all tickers to the same date index
        returns_aligned = returns_window.reindex(standard_dates, method='nearest', tolerance=pd.Timedelta('5 days'))

        if len(returns_aligned.dropna()) >= 60:  # At least 5 years of actual data
            corr_returns[ticker] = returns_aligned
        else:
            print(f"  ⚠ Warning: {ticker} has insufficient data for correlation ({len(returns_aligned.dropna())} months)")

    # Align all series to common dates (all now use standard_dates index)
    corr_df = pd.DataFrame(corr_returns)

    # Check for any remaining NaNs from reindex mismatches
    nan_counts = corr_df.isna().sum()
    if nan_counts.sum() > 0:
        print(f"\n  Note: {int(nan_counts.sum())} NaNs from reindex tolerance or late inception:")
        for ticker in nan_counts[nan_counts > 0].index:
            print(f"    {ticker}: {nan_counts[ticker]} NaNs")

    # PAIRWISE, NOT dropna(). The old `corr_df.dropna()` deleted a whole ROW when
    # any ONE ticker was missing, so the newest ticker in the universe silently
    # set the window for all 25. Measured 2026-09-10: VUSB's proxy starts 2015-03,
    # so the "15-year rolling window" was 136 months, not 180 — a quarter of the
    # sample thrown away by one convenience ticker added for a Vanguard
    # comparison, and nothing said so. A pair is estimated on its own overlap and
    # falls back to the baseline entry below MIN_PAIR_MONTHS; fix_correlation_matrix
    # repairs the mild indefiniteness that pairwise estimation can introduce.
    full_span = corr_df.columns[corr_df.notna().all()]
    years = len(corr_df) / 12
    print(f"\n  ✓ Correlation window: {len(corr_df)} months ({years:.1f} years)")
    print(f"    Date range: {corr_df.index[0].strftime('%Y-%m')} to {corr_df.index[-1].strftime('%Y-%m')}")
    print(f"    {len(full_span)}/{len(corr_df.columns)} tickers span it in full; "
          f"the rest are estimated pairwise")

    corr_baseline, by_regime, months = regime_correlations(corr_df, market_labels)
    corr_low, corr_medium, corr_high = (by_regime[r] for r in REGIMES)
    print(f"    regime months: low {months['low']}, medium {months['medium']}, "
          f"high {months['high']}")

    # Save correlation matrices (dynamic filenames based on ticker count)
    n = len(TICKERS)
    if corr_only:
        # Positional like the default set, so the columns must be the universe
        # in order — a ticker dropped for thin history would misalign every pair.
        assert [str(c) for c in corr_df.columns] == list(TICKERS), corr_df.columns
        for r, C in (('baseline', corr_baseline), ('low', corr_low),
                     ('medium', corr_medium), ('high', corr_high)):
            np.save(OUTPUT_DIR / correlation_filename(r, n, corr_window_years), C)
        print(f"  ✓ Saved 4 correlation matrices ({n}×{n}), "
              f"{correlation_filename('<regime>', n, corr_window_years)}")
        print("  --corr-only: JSON and default matrices left untouched.")
        return
    np.save(OUTPUT_DIR / f'correlation_baseline_{n}.npy', corr_baseline)
    np.save(OUTPUT_DIR / f'correlation_low_{n}.npy', corr_low)
    np.save(OUTPUT_DIR / f'correlation_medium_{n}.npy', corr_medium)
    np.save(OUTPUT_DIR / f'correlation_high_{n}.npy', corr_high)
    print(f"  ✓ Saved 4 correlation matrices ({n}×{n})")

    print()

    # Output JSON
    print("[Step 7/7] Generating JSON output...")

    # The one QUOTED input in the model. Stored so a run can use it offline and
    # so its staleness is visible; see factors.RF_REAL for why it is not the
    # default.
    print("  Fetching TIPS real yields (FRED)...")
    tips = fetch_tips_real_yields()
    for tenor in sorted(tips, key=int):
        t = tips[tenor]
        print(f"    {tenor:>2}y {t['latest']:>6.2%} as of {t['as_of']}   "
              f"(1y avg {t['avg_1y']:.2%}, full avg {t['avg_full']:.2%}, "
              f"range {t['min']:.2%}..{t['max']:.2%})")

    # Score every backfill against the fund it stands in for. Stored, not
    # asserted: tools/validate_parameters.py gates these offline.
    proxy_scores = {}
    for ticker, (actual, proxy, contrib, label) in sorted(proxy_pairs.items()):
        try:
            # Count contribution against the FINAL series, not the pre-splice
            # one. ROLLING_WINDOW_TICKERS are truncated to 240 months after
            # splicing, so a proxy that filled 308 months can end up supplying
            # far fewer — AVEE claimed 308 against a 240-month history, which is
            # impossible and was the tell.
            #
            # Only the PROXY layer counts. For a Ken French ticker everything
            # before the academic cutoff is the KF blend, not the yfinance
            # proxy; counting all pre-inception months made MTUM's vestigial
            # 2-month proxy look like 483 and tripped the fidelity gate.
            final = spliced_returns[ticker]
            in_proxy = final.index < actual.index.min()
            if ticker in KF_TICKERS:
                in_proxy &= final.index >= pd.Timestamp(KF_TICKERS[ticker]['end'])
            contrib = int(in_proxy.sum())
            proxy_scores[ticker] = score_proxy(actual, proxy, contrib, label)
        except Exception as e:
            print(f"  ⚠ could not score {ticker}'s proxy: {e}")
    if proxy_scores:
        worst = min((v for v in proxy_scores.values()
                     if not v.get('thin') and v['contributed_months'] > 3),
                    key=lambda v: v['corr'], default=None)
        if worst:
            print(f"  proxy fidelity: {len(proxy_scores)} scored, "
                  f"lowest corr {worst['corr']:.3f} ({worst['proxy']})")

    # Add metadata section
    ticker_params['metadata'] = {
        'version': 'v8',
        'last_updated': datetime.now().strftime('%Y-%m-%d'),
        'ticker_count': len(TICKERS),
        'proxy_scores': proxy_scores,
        'tips_real_yields': tips,
        'market_regime': {
            'proxy': MARKET_PROXY,
            'window_months': MARKET_REGIME_WINDOW,
            'ticker_window_months': REGIME_WINDOW,
            'shrinkage': REGIME_MARKET_SHRINKAGE,
            'cuts': {'p30': float(mkt_p30), 'p70': float(mkt_p70)},
            'transitions': market_transitions,
            'months': {r: int((market_labels == r).sum())
                       for r in ('low', 'medium', 'high')},
            'realised_vol': {
                r: float(spliced_returns[MARKET_PROXY]
                         .loc[market_labels[market_labels == r].index].std() * np.sqrt(12))
                for r in ('low', 'medium', 'high')},
            'correlation_window_months': {r: int(months[r]) for r in REGIMES},
        },
        # THE ORDER THE MATRICES ARE INDEXED BY. Every consumer reads a pair as
        # C[TICKERS.index(a), TICKERS.index(b)], so the matrices are POSITIONAL
        # and carry no keys of their own. Reorder MODEL_TICKERS without
        # regenerating and every correlation silently attaches to the wrong pair
        # while staying a valid 25x25 PSD unit-diagonal matrix — so the shape
        # check passes, validate_parameters passes, and the run is confidently
        # wrong. Measured 2026-09-12: prepending one ticker moves 141's
        # crisis-regime volatility -11.4%, and the bias is DOWNWARD, which
        # flatters exactly the diversified books this tool recommends.
        # Recorded from corr_df.columns rather than TICKERS, so it also catches
        # a ticker dropped for insufficient history above.
        'correlation_tickers': [str(c) for c in corr_df.columns],
        'correlation_files': {
            'baseline': f'correlation_baseline_{n}.npy',
            'low': f'correlation_low_{n}.npy',
            'medium': f'correlation_medium_{n}.npy',
            'high': f'correlation_high_{n}.npy'
        }
    }

    output_file = OUTPUT_DIR / 'ticker_parameters_regimes_v8.json'
    with open(output_file, 'w') as f:
        json.dump(ticker_params, f, indent=2)
    print(f"  ✓ Saved {output_file}")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Outputs:")
    print(f"  - ticker_parameters_regimes_v8.json ({len(TICKERS)} tickers)")
    print(f"  - correlation_baseline_{n}.npy ({n}×{n})")
    print(f"  - correlation_low_{n}.npy ({n}×{n})")
    print(f"  - correlation_medium_{n}.npy ({n}×{n})")
    print(f"  - correlation_high_{n}.npy ({n}×{n})")
    print("=" * 80)

    # Sanity-gate the data we just wrote (non-fatal: outputs already saved, but a
    # bad splice/proxy is flagged immediately rather than discovered in a sim).
    print()
    try:
        from tools.validate_parameters import main as _validate
    except ModuleNotFoundError:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent / 'tools'))
        from validate_parameters import main as _validate
    rc = _validate()
    if rc != 0:
        print("⚠ Parameter validation FAILED — inspect the flagged tickers before "
              "running simulations (see tools/validate_parameters.py).")

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument('--corr-window-years', type=int, default=None,
                    help='write ONLY the correlation matrices, on this many years, '
                         'as the _w<years> file set (selected by MC_CORR_WINDOW)')
    args = ap.parse_args()
    if args.corr_window_years == CORR_WINDOW_YEARS_DEFAULT:
        # Would overwrite the default matrices from fresh data while the JSON
        # stays as it was. The default set is only ever written by a full run.
        ap.error(f'{CORR_WINDOW_YEARS_DEFAULT} is the default window; run without '
                 f'the flag to regenerate everything')
    if args.corr_window_years is None:
        main()
    else:
        main(corr_window_years=args.corr_window_years, corr_only=True)
