"""
Loaders for Monte Carlo Portfolio Analysis
===========================================

Two helpers, merged into one module:

  load_parameters_from_json()  - simulation parameters + correlation matrices
                                 from output/ (JSON + NPY), with staleness warnings.
  load_portfolios_from_csv()   - portfolio templates from portfolio_templates.csv.

Usage:
    from montecarlo.loaders import load_parameters_from_json, load_portfolios_from_csv

    params = load_parameters_from_json()
    EXPECTED_RETURNS = params['EXPECTED_RETURNS']

    portfolios = load_portfolios_from_csv()
    # {portfolio_id: {'name': ..., 'allocation': {ticker: weight, ...}}}

CSV comment/blank rows:
    Any row whose first column (portfolio_id) is blank or starts with '#' is
    skipped, so you can annotate portfolio_templates.csv with section headers, e.g.

        # === 100s: 100% equity / 0% bonds ===
        100,...,0.41,...

Author: Claude Code
"""

import csv
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np


# =============================================================================
# SCHEMA: model tickers (the real allocation columns) + derived nav columns
# =============================================================================
# ★ SINGLE SOURCE OF TRUTH for the ticker universe. ★
# data_collection_regime_model.py and run.py both do
# `from montecarlo.loaders import MODEL_TICKERS as TICKERS` — edit the universe HERE only.
# Order is canonical: it defines correlation-matrix row order and JSON order.
# These are also the ONLY columns the loaders treat as allocations — any other
# column (e.g. the derived nav columns below) is ignored when parsing weights.
# After changing this list, follow docs/RUNBOOK_tickers.md.
MODEL_TICKERS = [
    'VOO', 'COST', 'JQUA', 'AVUV', 'VTI', 'AVDE', 'AVEM', 'AVDV', 'AVEE',
    'VXUS', 'VEA', 'VWO', 'VSS', 'BND', 'JPLD', 'JPST', 'MTUM', 'VFMO',
    'VTV', 'AVLV', 'DFWIX', 'VT', 'VBR', 'VUSB', 'BSV', 'STIP', 'VGIT',
]

# Single companies (not funds): they carry idiosyncratic / survivorship risk a
# diversified fund doesn't. Each automatically gets a 20-year rolling window AND an
# inflated parameter-uncertainty band (0.25x vol vs the 0.15x default) — applied in
# data_collection_regime_model.py. Must be a subset of MODEL_TICKERS. To add one,
# see docs/RUNBOOK_tickers.md.
SINGLE_STOCKS = {'COST'}

# Derived navigation columns appended to portfolio_templates.csv. Computed from
# the allocations (see compute_derived), so they self-verify and auto-refresh.
DERIVED_COLUMNS = [
    'bond_pct', 'us_pct', 'dm_share', 'tilt_family',
    'mom_fund', 'val_fund', 'n_funds', 'max_single_fund',
]

# Role membership for the roll-ups. Two tickers are global/regional blends handled
# explicitly in compute_derived (not in a single bucket): DFWIX (65% DM / 35% EM)
# and VT (Total World ≈ 60% US / 30% DM / 10% EM).
_US_FUNDS = ['VOO', 'VTI', 'COST', 'MTUM', 'VFMO', 'AVUV', 'VTV', 'AVLV', 'JQUA', 'VBR']
_BOND_FUNDS = ['BND', 'JPLD', 'JPST', 'VUSB', 'BSV', 'STIP', 'VGIT']  # VUSB≈JPST, BSV≈JPLD (Vanguard analogs)
# Bond funds whose principal is indexed to CPI. Still bonds for every role —
# bond tier, glide sleeves, the premium scale — but not for inflation: a
# nominal bond loses real return to an inflation surprise, a TIPS fund does
# not, and that difference is the only reason to hold one. See
# factors.INFL_BETA_TIPS.
INFLATION_LINKED = ['STIP']
# Effective duration (years) of every NOMINAL bond fund, as each issuer reports
# it — a property of the fund's mandate, so it does not go stale the way a
# yield would. Sets how much of the bond inflation pass-through a fund carries
# (factors.bond_infl_beta). Sources, checked 2026-10-24: BSV 2.6 and VUSB 0.9-
# 1.0 (Vanguard, Aug 2026), JPLD 1.8-2.05 (several profiles, target <= 3),
# JPST ~0.5 (J.P. Morgan: target < 1, historically run near half a year), BND
# ~6 and VGIT ~5 (anything at or above BSV's 2.6 gets the full figure, so their
# exact values do not matter). A new nominal bond fund needs an entry here.
BOND_DURATION = {'BND': 6.0, 'BSV': 2.6, 'JPLD': 1.9, 'VUSB': 0.95, 'JPST': 0.5,
                 'VGIT': 5.0}

# ---------------------------------------------------------------------------
# Bond sleeves — the third axis of a portfolio
# ---------------------------------------------------------------------------
# A portfolio is (equity sleeve, bond sleeve, bond %). Those are INDEPENDENT:
# the same equity mix can be paired with any bond sleeve at any weight, and you
# may want to compare sleeve A against sleeve B holding the equity side fixed.
#
# The CSV once materialised nine rows that were exact derivatives of three
# bases (240/340/540 from 140, 245/345/545 from 145, 202/302/502 from 102 —
# verified identical to 5.6e-17); it now holds only 1xx rows, and every bond
# tier is synthesised by with_bonds(). Synthesising them means a ladder is
# like-for-like BY CONSTRUCTION. That matters: 141's ladder previously pointed at
# the x40 rows, which use AVDE where 141 holds VEA, so "derisking" silently
# bundled a fund swap with the bond decision.
BOND_SLEEVES = {
    'core': {'BND': 0.60, 'JPLD': 0.25, 'JPST': 0.15},
    'vanguard': {'BND': 0.60, 'BSV': 0.25, 'VUSB': 0.15},   # Vanguard analogs
    'core_tips': {'BND': 0.60, 'JPLD': 0.25, 'STIP': 0.15},  # core, STIP for JPST
    'thirds': {'BND': 1/3, 'JPLD': 1/3, 'STIP': 1/3},        # equal BND / JPLD / STIP
    'quarters': {'BND': 0.25, 'JPLD': 0.25, 'JPST': 0.25, 'STIP': 0.25},  # equal four
    # Treasuries for duration, credit only at the short end (2026-10-25)
    'core_vgit': {'VGIT': 0.60, 'JPLD': 0.25, 'JPST': 0.15},       # core, BND -> VGIT
    'core_vgit_tips': {'VGIT': 0.60, 'JPLD': 0.25, 'STIP': 0.15},  # + JPST -> STIP
    'tsy_tips': {'VGIT': 0.70, 'STIP': 0.30},                      # Treasuries + TIPS
    'quarters_vgit': {'VGIT': 0.25, 'JPLD': 0.25, 'JPST': 0.25, 'STIP': 0.25},  # quarters, BND -> VGIT
    'bnd_only': {'BND': 1.00},          # aggregate only — the simple baseline
    'jpst_only': {'JPST': 1.00},        # ultra-short only — isolates duration
    'jpld_only': {'JPLD': 1.00},        # limited duration only (~1.9y)
    'vusb_only': {'VUSB': 1.00},        # ultra-short only, Vanguard (~0.95y)
    'bsv_only': {'BSV': 1.00},          # short-term only, Vanguard
    'vgit_only': {'VGIT': 1.00},        # intermediate Treasury only — against
                                        # bnd_only (similar duration) it isolates credit
    'stip_only': {'STIP': 1.00},        # short TIPS only — against bsv_only (similar
                                        # duration) it isolates inflation protection
}

# Which sleeve a base portfolio uses when none is named. Anything not listed
# falls back to 'core'.
# Only BASE portfolios need an entry: with_bonds() synthesises every rung from
# the base, so the sleeve is looked up once per base rather than per rung.
NATIVE_SLEEVE = {'145': 'vanguard', '102': 'bnd_only'}


def native_sleeve(portfolio_id):
    """The bond sleeve a portfolio uses unless told otherwise."""
    return NATIVE_SLEEVE.get(str(portfolio_id), 'core')


def equity_sleeve(allocation):
    """Strip bonds and renormalise the equity side to sum to 1.0."""
    equity = {t: w for t, w in allocation.items()
              if t not in _BOND_FUNDS and w > 0}
    total = sum(equity.values())
    if total <= 0:
        raise ValueError("allocation has no equity to renormalise")
    return {t: w / total for t, w in equity.items()}


def with_bonds(allocation, bond_pct, sleeve='core'):
    """Scale an allocation's equity to (1 - bond_pct) and add the bond sleeve.

    Reproduces the stored tier rows exactly (to floating-point noise), so
    synthesising a ladder cannot change any existing result.
    """
    if not 0.0 <= bond_pct < 1.0:
        raise ValueError(f"bond_pct must be in [0, 1), got {bond_pct}")
    if sleeve not in BOND_SLEEVES:
        raise ValueError(
            f"unknown bond sleeve {sleeve!r}; have {', '.join(sorted(BOND_SLEEVES))}")

    # Start from the FULL ticker universe with explicit zeros. The CSV rows carry
    # every ticker, and the engine iterates TICKERS when rebalancing
    # (execute_full_rebalance does target_weights[t] for t in TICKERS), so a dict
    # holding only the non-zero positions raises KeyError the moment a glide
    # switches onto it — after age 65, not at startup, which is why a control run
    # looked fine.
    out = {t: 0.0 for t in MODEL_TICKERS}
    for ticker, weight in equity_sleeve(allocation).items():
        out[ticker] = weight * (1.0 - bond_pct)
    for ticker, share in BOND_SLEEVES[sleeve].items():
        out[ticker] = out.get(ticker, 0.0) + share * bond_pct
    return out

# The one ticker that stands for "the market". It sets the premium baseline
# (engine.scale_equity_premia collapses onto it as k -> 0) and, since 2026-09-10,
# the MARKET REGIME that selects the correlation matrix. Lives here rather than
# in engine.py because data_collection_regime_model.py needs it too and must not
# import the engine.
MARKET_PROXY = 'VOO'

# Regimes are a persistent market STATE, and the window used to estimate that
# state decides what "high volatility" even means. Measured 2026-09-10 on VOO's
# 680-month spliced series:
#
#     window   GFC 2008-10   COVID 2020-03   realised 1m vol lo/med/hi   duration
#       36m      medium         medium          11.2 / 16.4 / 18.0%      12-24mo
#       12m      HIGH           HIGH             9.0 / 14.8 / 20.5%       6-13mo
#        6m      HIGH           HIGH             8.8 / 13.2 / 22.0%       4-5mo
#
# A 36-month window does not identify crises AT ALL — it labels the worst month
# of the GFC as "medium", because the two calm years either side are inside the
# same window. 12 months identifies both crises, separates realised volatility
# 2.3x, and still produces 6-13 month regimes rather than month-to-month noise.
MARKET_REGIME_WINDOW = 12

# The same window for the PER-TICKER regimes, and for the same reason. It used
# to be 36 and the two changes below had to be made together: shortening the
# window alone makes the model WORSE, because the old regime volatilities were
# percentiles (p10/p50/p90) of the rolling estimator, and a 12-month estimator
# is noisy enough that its p10 and p90 spread far past any real regime.
#
# Scored by out-of-sample Gaussian log-likelihood of next month's return, fit on
# the first 60% of each ticker's history and scored on the last 40%:
#
#     window    p10/p50/p90 vols    within-bucket vols
#        6            1.5150               1.7209
#       12            1.6943             **1.8867**
#       18            1.6801               1.8805
#       24            1.7059               1.8722
#       36          **1.8220**             1.8407      <- the old combination
#
# Within-bucket volatility wins at EVERY window, and once it is in place 12
# months wins — improving on the old rule for 18 of 24 tickers.
REGIME_WINDOW = 12

# Regime volatility is the realised volatility of the months in that bucket, not
# a percentile of the rolling estimator. Besides the score above, this is the
# only rule that reproduces each ticker's own volatility: mixing p10/p50/p90 by
# the chain's stationary distribution OVERSTATED it by +2.0% on average and
# +9.0% for AVEM, because the labels are cut at p30/p70 and the vols were read
# at p10/p90. Within-bucket vols cut the mean error to +0.5%.
#
# Labels are cut here; the vols follow from the buckets.
REGIME_CUTS = (0.30, 0.70)

# Per-ticker transitions are conditioned on the MARKET regime, shrunk toward the
# ticker's own unconditional row by this many pseudo-counts. Measured across the
# universe, P(medium -> high) runs 0.019 / 0.048 / 0.271 as the market goes
# low / medium / high — a 14x span, and the joint tail the engine could not
# produce while the two states were independent.
#
# Many conditional cells are nearly empty (VOO shows P = 1.00 off ONE
# observation), so shrinkage is not a refinement, it is the thing that makes
# this estimable at all. Chosen by out-of-sample likelihood of the realised next
# regime: k=0 scores -2.3623, k=10 scores -0.4804, unconditional -0.5275. The
# curve is flat from 5 to 20, so 10 is not a knife edge.
REGIME_MARKET_SHRINKAGE = 10.0

# base_volatility's lookback, in YEARS. It was written as a hardcoded
# '2010-01-01' — which was exactly 15 years before the tool was built in 2025,
# so the INTENT was a 15-year rolling window. Frozen as a date it had already
# become 16 years by 2026 and would widen by a year every year, silently mixing
# more of the past into a statistic meant to describe the recent regime.
#
# Relative now, so it means what it was supposed to mean. (base_volatility is
# currently DEAD — the engine loads it and never reads it, the simulation draws
# from VOLATILITY_REGIMES — so this changes no result today. Fixed anyway: a
# constant that drifts is a bug waiting for a consumer.)
BASE_VOL_WINDOW_YEARS = 15

# The correlation matrices' lookback, in YEARS. 15 leaves out 2000-02 and 2008,
# so every crisis it sees is 2020 or 2022 — and BND's crisis correlation with
# VOO reads +0.49 against +0.16 since 1991 (ledger entry 2026-10-24, unpublished). Another window
# is a separate set of files, `correlation_<regime>_<N>_w<years>.npy`, written by
# `data_collection_regime_model.py --corr-window-years Y` and selected at run
# time by MC_CORR_WINDOW (engine.CORR_WINDOW). The default files keep their name.
CORR_WINDOW_YEARS_DEFAULT = 15


def correlation_filename(corr_type, n_tickers, window_years=CORR_WINDOW_YEARS_DEFAULT):
    """The .npy name for one regime's matrix at a given correlation window."""
    suffix = '' if window_years == CORR_WINDOW_YEARS_DEFAULT else f'_w{window_years}'
    return f'correlation_{corr_type}_{n_tickers}{suffix}.npy'

_DM_FUNDS = ['AVDE', 'VEA', 'VXUS', 'AVDV', 'VSS']   # intl developed
_EM_FUNDS = ['AVEM', 'VWO', 'AVEE']                  # intl emerging


def compute_derived(allocation):
    """Compute the DERIVED_COLUMNS for one portfolio from its allocation dict.

    `allocation` maps ticker -> weight (weights, not percentages). Returns a dict
    keyed by DERIVED_COLUMNS. Pure function — used by both the annotator and the
    holdings-update tools so the columns never go stale.
    """
    def g(t):
        return float(allocation.get(t, 0) or 0)

    bond = sum(g(t) for t in _BOND_FUNDS)
    # VT (Total World) and DFWIX are blends split across regions, not single-bucket.
    # VT ≈ FTSE Global All Cap: 60% US / 30% DM-exUS / 10% EM.
    us = sum(g(t) for t in _US_FUNDS) + 0.60 * g('VT')
    # DFWIX 65/35 as Dimensional reports it (2026-09-30, its own country
    # assignments: Korea 6.3% and Taiwan 10.8% emerging; 64.82 / 35.09 / 0.09
    # US). Morningstar's regions read ~19% emerging because they file Korea
    # and Taiwan as developed; the fitted loadings' ~30% understates it too.
    # Display only — the simulation uses the fund's own returns. The 78/22 it
    # replaced is the pre-2013 PROXY blend in data_collection, which stays.
    dm = sum(g(t) for t in _DM_FUNDS) + 0.65 * g('DFWIX') + 0.30 * g('VT')
    em = sum(g(t) for t in _EM_FUNDS) + 0.35 * g('DFWIX') + 0.10 * g('VT')
    mom, val, qual = g('MTUM') + g('VFMO'), g('VTV') + g('AVLV'), g('JQUA')

    if qual > 0:
        tilt = 'quality'
    elif mom > 0 and val > 0:
        tilt = 'mom+val'
    elif mom > 0:
        tilt = 'momentum'
    elif val > 0:
        tilt = 'value'
    else:
        tilt = 'core'

    nonzero = [g(t) for t in MODEL_TICKERS if g(t) > 0]
    return {
        'bond_pct': round(bond, 4),
        'us_pct': round(us, 4),
        'dm_share': round(dm / (dm + em), 4) if (dm + em) else 0.0,
        'tilt_family': tilt,
        'mom_fund': 'MTUM' if g('MTUM') > 0 else ('VFMO' if g('VFMO') > 0 else ''),
        'val_fund': 'VTV' if g('VTV') > 0 else ('AVLV' if g('AVLV') > 0 else ''),
        'n_funds': len(nonzero),
        'max_single_fund': round(max(nonzero), 4) if nonzero else 0.0,
    }


# =============================================================================
# PARAMETER LOADER (JSON + NPY from output/)
# =============================================================================
def load_parameters_from_json(
    params_file='ticker_parameters_regimes_v8.json',
    correlation_dir=None,
    warn_age_days=90,
    base_dir=None,
    verbose=True,
    corr_window_years=CORR_WINDOW_YEARS_DEFAULT
):
    """
    Load all simulation parameters from JSON and NPY files.

    Parameters:
    -----------
    params_file : str
        Filename of the parameters JSON file
    correlation_dir : Path or None
        Directory containing correlation .npy files (defaults to base_dir)
    warn_age_days : int
        Warn if parameters are older than this many days
    base_dir : Path or None
        Base directory for all files (defaults to <cwd>/output)
    verbose : bool
        If True, print loading progress messages (default: True)
    corr_window_years : int
        Correlation lookback; anything but the default loads the `_w<years>`
        file set (see correlation_filename) and fails if it was never generated

    Returns:
    --------
    dict with keys:
        - EXPECTED_RETURNS : dict[str, float]
        - BASE_VOLATILITY : dict[str, float]
        - PARAM_UNCERTAINTY_STD : dict[str, float]
        - VOLATILITY_REGIMES : dict[str, dict[str, float]]
        - REGIME_TRANSITIONS : dict[str, dict[str, dict[str, float]]]
        - CORRELATION_BASELINE / LOW / MEDIUM / HIGH : np.ndarray (N x N)
        - metadata : dict
        - ticker_list : list[str]

    Raises:
    -------
    FileNotFoundError : If required files are missing
    ValueError : If data validation fails
    """
    # Setup paths
    if base_dir is None:
        # Default to 'output' subdirectory where data_collection_regime_model.py writes files
        base_dir = Path.cwd() / 'output'
    else:
        base_dir = Path(base_dir)

    if correlation_dir is None:
        correlation_dir = base_dir
    else:
        correlation_dir = Path(correlation_dir)

    params_path = base_dir / params_file

    # ------------------------------------------------------------------
    # STEP 1: Load JSON parameters
    # ------------------------------------------------------------------
    if not params_path.exists():
        raise FileNotFoundError(
            f"\n{'='*70}\n"
            f"ERROR: Parameter file not found\n"
            f"{'='*70}\n"
            f"File: {params_path}\n\n"
            f"Please generate parameters first:\n"
            f"  cd {base_dir.parent}\n"
            f"  python3 data_collection_regime_model.py\n\n"
            f"This will download data from yfinance and generate:\n"
            f"  - {params_file}\n"
            f"  - correlation_*.npy files\n"
            f"{'='*70}\n"
        )

    try:
        with open(params_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {params_path}: {e}")

    # ------------------------------------------------------------------
    # STEP 2: Check metadata and staleness
    # ------------------------------------------------------------------
    metadata = data.get('metadata', {})

    version = metadata.get('version', 'unknown')
    if verbose:
        print(f"Loading parameters version: {version}")

    last_updated_str = metadata.get('last_updated')
    if last_updated_str:
        try:
            last_updated = datetime.strptime(last_updated_str, '%Y-%m-%d')
            days_old = (datetime.now() - last_updated).days

            if days_old > warn_age_days:
                warnings.warn(
                    f"\n{'='*70}\n"
                    f"WARNING: Parameters are {days_old} days old\n"
                    f"{'='*70}\n"
                    f"Last updated: {last_updated_str}\n"
                    f"Consider refreshing:\n"
                    f"  python3 data_collection_regime_model.py\n"
                    f"{'='*70}\n",
                    UserWarning
                )
            elif verbose:
                print(f"Parameters last updated: {last_updated_str} ({days_old} days ago)")
        except ValueError:
            warnings.warn(f"Could not parse last_updated date: {last_updated_str}")

    # ------------------------------------------------------------------
    # STEP 3: Extract ticker parameters
    # ------------------------------------------------------------------
    tickers_data = {k: v for k, v in data.items() if k != 'metadata'}

    if not tickers_data:
        raise ValueError(f"No ticker data found in {params_path}")

    ticker_list = list(tickers_data.keys())
    n_tickers = len(ticker_list)
    if verbose:
        print(f"Found {n_tickers} tickers: {', '.join(ticker_list)}")

    expected_returns = {}
    base_volatility = {}
    param_uncertainty_std = {}
    volatility_regimes = {}
    regime_transitions = {}

    for ticker, ticker_data in tickers_data.items():
        try:
            expected_returns[ticker] = ticker_data['expected_return']
        except KeyError:
            raise ValueError(f"{ticker}: Missing expected_return")

        try:
            base_volatility[ticker] = ticker_data['base_volatility']
        except KeyError:
            raise ValueError(f"{ticker}: Missing base_volatility")

        param_uncertainty_std[ticker] = ticker_data.get('parameter_std')
        if param_uncertainty_std[ticker] is None:
            raise ValueError(f"{ticker}: Missing parameter_std")


        try:
            volatility_regimes[ticker] = ticker_data['volatility_regimes']
        except KeyError as e:
            raise ValueError(f"{ticker}: Missing volatility_regimes: {e}")

        try:
            regime_transitions[ticker] = ticker_data['regime_transitions']
            for regime in ['low', 'medium', 'high']:
                if regime not in regime_transitions[ticker]:
                    raise ValueError(f"{ticker}: Missing {regime} regime transitions")
                trans = regime_transitions[ticker][regime]
                total = sum(trans.values())
                if not (0.99 < total < 1.01):
                    warnings.warn(f"{ticker} {regime} transitions sum to {total:.4f}, not 1.0")
        except KeyError as e:
            raise ValueError(f"{ticker}: Missing regime_transitions: {e}")

        # Transitions conditioned on the MARKET regime, keyed (ticker, market).
        # Optional: parameters written before 2026-09-10 have none, and the
        # engine falls back to the unconditional chain for those.
        by_market = ticker_data.get('regime_transitions_by_market')
        if by_market:
            for market, rows in by_market.items():
                regime_transitions[(ticker, market)] = rows

    # ------------------------------------------------------------------
    # STEP 4: Load correlation matrices
    # ------------------------------------------------------------------
    corr_files_meta = metadata.get('correlation_files', {})
    corr_types = ['baseline', 'low', 'medium', 'high']
    correlation_matrices = {}

    for corr_type in corr_types:
        if corr_window_years != CORR_WINDOW_YEARS_DEFAULT:
            corr_filename = correlation_filename(corr_type, n_tickers, corr_window_years)
            if not (correlation_dir / corr_filename).exists():
                raise FileNotFoundError(
                    f"MC_CORR_WINDOW={corr_window_years}: {corr_filename} not found "
                    f"in {correlation_dir}. Generate it with:\n"
                    f"  python3 data_collection_regime_model.py "
                    f"--corr-window-years {corr_window_years}")
        elif corr_type in corr_files_meta:
            corr_filename = corr_files_meta[corr_type]
        else:
            pattern = f'correlation_{corr_type}_{n_tickers}_*.npy'
            matches = list(correlation_dir.glob(pattern))
            if not matches:
                pattern = f'correlation_{corr_type}_{n_tickers}.npy'
                matches = list(correlation_dir.glob(pattern))

            if not matches:
                raise FileNotFoundError(
                    f"\n{'='*70}\n"
                    f"ERROR: Correlation matrix not found\n"
                    f"{'='*70}\n"
                    f"Looking for: {pattern}\n"
                    f"In directory: {correlation_dir}\n\n"
                    f"Please generate correlation matrices:\n"
                    f"  python3 data_collection_regime_model.py\n"
                    f"{'='*70}\n"
                )

            corr_filename = sorted(matches)[-1].name

        corr_path = correlation_dir / corr_filename

        if not corr_path.exists():
            raise FileNotFoundError(
                f"\n{'='*70}\n"
                f"ERROR: Correlation file specified in metadata not found\n"
                f"{'='*70}\n"
                f"File: {corr_path}\n"
                f"From metadata: {corr_files_meta.get(corr_type, 'N/A')}\n"
                f"{'='*70}\n"
            )

        try:
            matrix = np.load(corr_path)
        except Exception as e:
            raise ValueError(f"Failed to load {corr_path}: {e}")

        if matrix.shape != (n_tickers, n_tickers):
            raise ValueError(
                f"Correlation matrix {corr_type} has shape {matrix.shape}, "
                f"expected ({n_tickers}, {n_tickers})"
            )

        if not np.allclose(matrix, matrix.T):
            warnings.warn(f"Correlation matrix {corr_type} is not symmetric")

        eigenvalues = np.linalg.eigvalsh(matrix)
        if np.any(eigenvalues < -1e-10):
            warnings.warn(
                f"Correlation matrix {corr_type} has negative eigenvalues: "
                f"min = {eigenvalues.min():.6f}"
            )

        correlation_matrices[f'CORRELATION_{corr_type.upper()}'] = matrix
        if verbose:
            print(f"Loaded {corr_type} correlation matrix: {corr_path.name}")

    # ------------------------------------------------------------------
    # STEP 5: Final validation
    # ------------------------------------------------------------------
    for name, d in [
        ('expected_returns', expected_returns),
        ('base_volatility', base_volatility),
        ('param_uncertainty_std', param_uncertainty_std),
        ('volatility_regimes', volatility_regimes),
        ('regime_transitions', regime_transitions)
    ]:
        # Tuple keys are the (ticker, market regime) conditional chains, which
        # are deliberately not one-per-ticker. Only the plain ticker keys have to
        # match the universe.
        keys = {k for k in d if isinstance(k, str)}
        if keys != set(ticker_list):
            missing = set(ticker_list) - keys
            extra = keys - set(ticker_list)
            raise ValueError(
                f"{name} ticker mismatch:\n"
                f"  Missing: {missing}\n"
                f"  Extra: {extra}"
            )

    if verbose:
        print(f"\n✓ Successfully loaded all parameters for {n_tickers} tickers")

    # ------------------------------------------------------------------
    # STEP 6: Return standardized format
    # ------------------------------------------------------------------
    return {
        'EXPECTED_RETURNS': expected_returns,
        'BASE_VOLATILITY': base_volatility,
        'PARAM_UNCERTAINTY_STD': param_uncertainty_std,
        'VOLATILITY_REGIMES': volatility_regimes,
        'REGIME_TRANSITIONS': regime_transitions,
        **correlation_matrices,  # CORRELATION_BASELINE / LOW / MEDIUM / HIGH
        'metadata': metadata,
        'ticker_list': ticker_list,
    }


# =============================================================================
# PORTFOLIO LOADER (portfolio_templates.csv)
# =============================================================================
TEMPLATES_CSV = 'portfolios/portfolio_templates.csv'
ARCHIVE_CSV = 'portfolios/portfolio_archive.csv'


def archived_portfolio_ids(base_dir=None):
    """Ids moved to portfolio_archive.csv — settled questions kept restorable.

    Read only to say WHY an id is unknown: an archived row is not loadable until
    it is moved back into portfolio_templates.csv (same columns, cut and paste).
    """
    f = Path(base_dir or Path(__file__).resolve().parent.parent) / ARCHIVE_CSV
    if not f.exists():
        return set()
    return {line.split(',', 1)[0] for line in f.read_text().splitlines()[1:]
            if line and not line.startswith(('#', ','))}


def load_portfolios_from_csv(csv_file=TEMPLATES_CSV, base_dir=None, verbose=True):
    """
    Load portfolio library from CSV file.

    Rows whose portfolio_id is blank or starts with '#' are treated as
    comments/separators and skipped, so the CSV can carry section headers.

    Returns:
    --------
    dict
        {portfolio_id: {'name': str, 'allocation': {ticker: weight, ...}}}

    Raises:
    -------
    FileNotFoundError : If CSV file not found
    ValueError : If data validation fails (e.g. allocations don't sum to 1.0)
    """
    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)

    csv_path = base_dir / csv_file

    if not csv_path.exists():
        raise FileNotFoundError(
            f"\n{'='*70}\n"
            f"ERROR: Portfolio template file not found\n"
            f"{'='*70}\n"
            f"File: {csv_path}\n\n"
            f"Please ensure {csv_file} exists in:\n"
            f"  {base_dir}\n"
            f"{'='*70}\n"
        )

    portfolio_library = {}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)

        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"CSV file {csv_path} is empty or malformed")

        # Allowlist: only real model tickers are allocations. Any other column
        # (portfolio_id, name, or a derived nav column) is ignored here.
        ticker_columns = [col for col in fieldnames if col in MODEL_TICKERS]

        if verbose:
            print(f"Loading portfolios from: {csv_path.name}")
            print(f"Expected tickers: {len(ticker_columns)} ({', '.join(ticker_columns)})")

        portfolio_count = 0
        for row_num, row in enumerate(reader, start=2):  # row 1 is the header
            portfolio_id = (row.get('portfolio_id') or '').strip()

            # Skip blank separator rows and '#' comment rows
            if not portfolio_id or portfolio_id.startswith('#'):
                continue
            if portfolio_id in portfolio_library:
                raise ValueError(f"Row {row_num}: duplicate portfolio_id {portfolio_id} "
                                 f"would replace the earlier row")

            name = (row.get('name') or '').strip()
            if not name:
                raise ValueError(f"Row {row_num}: Missing name")

            allocation = {}
            for ticker in ticker_columns:
                value_str = (row.get(ticker) or '').strip()

                if not value_str or value_str == '00':
                    allocation[ticker] = 0.0
                else:
                    try:
                        allocation[ticker] = float(value_str)
                    except ValueError:
                        raise ValueError(
                            f"Row {row_num} (Portfolio {portfolio_id}): "
                            f"Invalid allocation for {ticker}: '{value_str}'"
                        )

            short = [t for t, w in allocation.items() if w < 0]
            if short:
                raise ValueError(f"Row {row_num} (Portfolio {portfolio_id}): "
                                 f"negative allocation for {', '.join(short)}")

            total = sum(allocation.values())
            if not (0.99 < total < 1.01):
                raise ValueError(
                    f"Row {row_num} (Portfolio {portfolio_id}): "
                    f"Allocations sum to {total:.6f}, expected 1.0 (±0.01)\n"
                    f"  Name: {name}\n"
                    f"  Non-zero allocations: "
                    f"{', '.join(f'{k}={v}' for k, v in allocation.items() if v > 0)}"
                )

            portfolio_library[portfolio_id] = {
                'name': name,
                'allocation': allocation
            }
            portfolio_count += 1

    if not portfolio_library:
        raise ValueError(f"No portfolios loaded from {csv_path}")

    if verbose:
        print(f"✓ Successfully loaded {portfolio_count} portfolios")
        print(f"  Portfolio IDs: {', '.join(sorted(portfolio_library.keys(), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0, x)))}")

    return portfolio_library


if __name__ == '__main__':
    print("Testing loaders...")
    print("=" * 70)
    portfolios = load_portfolios_from_csv()
    print(f"\nLoaded {len(portfolios)} portfolios. Sample:")
    for pid, pdata in list(portfolios.items())[:3]:
        non_zero = {k: v for k, v in pdata['allocation'].items() if v > 0}
        print(f"  {pid}: sum={sum(pdata['allocation'].values()):.4f} | {non_zero}")
    print("\n✓ montecarlo/loaders.py OK")
