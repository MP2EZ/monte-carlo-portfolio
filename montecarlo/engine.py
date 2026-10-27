# ============================================================================
# Monte Carlo Simulation Engine
# ============================================================================
# This module owns everything numeric: parameter loading, regime machinery,
# correlated return generation, rebalancing band logic, the monthly simulation
# loop, and the multiprocessing driver. It prints nothing and imports no
# plotting library.
#
# Workers unpickle `engine.run_single_simulation` by qualified name, so the
# module-level constants below (TICKERS, REBALANCING_BANDS, the loaded
# parameters) are re-created in every worker process by re-importing this
# module. Assigning to them at runtime from the parent will NOT propagate to
# workers under the `spawn` start method — treat them as source-level
# constants.
# ============================================================================

import os

import numpy as np
from scipy.linalg import eigh
from multiprocessing import Pool, cpu_count

# TICKERS is the single source of truth in loaders.MODEL_TICKERS — imported here
# (not redefined) so this module can never drift out of sync with the universe.
from montecarlo.loaders import (load_parameters_from_json, MODEL_TICKERS as TICKERS,
                     _BOND_FUNDS, INFLATION_LINKED as _INFLATION_LINKED,
                     BOND_DURATION as _BOND_DURATION,
                     MARKET_PROXY as _MARKET_PROXY,
                     CORR_WINDOW_YEARS_DEFAULT as _CORR_WINDOW_DEFAULT)
from montecarlo import policies as _policies

# ─────────────────────────────────────────────────────────────────────────────
# REBALANCING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# Rebalancing bands per ticker (relative OR absolute, whichever is larger)
# Band width = max(target × relative, absolute)
REBALANCING_BANDS = {
    # Core holdings
    'VOO': {'relative': 0.15, 'absolute': 0.05},     # ±max(15% of target, 5pp)
    'VTI': {'relative': 0.15, 'absolute': 0.05},
    # US Factor tilts
    'AVLV': {'relative': 0.30, 'absolute': 0.04},
    'VTV': {'relative': 0.25, 'absolute': 0.04},
    'AVUV': {'relative': 0.35, 'absolute': 0.04},
    'VFMO': {'relative': 0.30, 'absolute': 0.035},
    'MTUM': {'relative': 0.25, 'absolute': 0.035},
    'JQUA': {'relative': 0.20, 'absolute': 0.04},
    'COST': {'relative': 0.40, 'absolute': 0.02},
    # International
    'VXUS': {'relative': 0.20, 'absolute': 0.04},
    'VEA': {'relative': 0.20, 'absolute': 0.04},
    'AVDE': {'relative': 0.20, 'absolute': 0.04},
    'VWO': {'relative': 0.25, 'absolute': 0.035},
    'AVEM': {'relative': 0.25, 'absolute': 0.03},
    'VSS': {'relative': 0.30, 'absolute': 0.035},
    'AVDV': {'relative': 0.35, 'absolute': 0.03},
    'AVEE': {'relative': 0.35, 'absolute': 0.02},
    'DFWIX': {'relative': 0.20, 'absolute': 0.04},   # Broad intl, similar to AVDE
    # Benchmark / Vanguard-only additions
    'VT': {'relative': 0.15, 'absolute': 0.05},      # Total World core, like VOO/VTI
    'VBR': {'relative': 0.35, 'absolute': 0.04},     # US small-cap value, like AVUV
    # Bonds
    'BND': {'relative': 0.10, 'absolute': 0.03},
    'JPLD': {'relative': 0.08, 'absolute': 0.02},
    'JPST': {'relative': 0.05, 'absolute': 0.012},
    'BSV': {'relative': 0.08, 'absolute': 0.02},     # Short-term bond, like JPLD
    'STIP': {'relative': 0.08, 'absolute': 0.02},    # 0-5y TIPS: short duration, like BSV
    'VUSB': {'relative': 0.05, 'absolute': 0.012},   # Ultra-short bond, like JPST
    'VGIT': {'relative': 0.10, 'absolute': 0.03},    # Intermediate Treasury, like BND
}

# The rule a written investment policy states: the greater of 25% relative or
# 5pp absolute, for every position, with a single stock kept at the band above
# (COST: 5% target, 7% ceiling). MC_REBALANCING_BANDS=uniform simulates it
# ('ips', its name until 2026-10-26, still reads); the default is the per-fund
# pairs above, so no shipped number moves. Read from the environment because
# workers re-import under `spawn`.
UNIFORM_BAND = {'relative': 0.25, 'absolute': 0.05}
REBALANCING_POLICY = os.environ.get('MC_REBALANCING_BANDS', 'model').strip() or 'model'
REBALANCING_POLICY = {'ips': 'uniform'}.get(REBALANCING_POLICY, REBALANCING_POLICY)
if REBALANCING_POLICY == 'uniform':
    from montecarlo.loaders import SINGLE_STOCKS as _SINGLE_STOCKS
    REBALANCING_BANDS = {t: (dict(b) if t in _SINGLE_STOCKS else dict(UNIFORM_BAND))
                         for t, b in REBALANCING_BANDS.items()}
elif REBALANCING_POLICY != 'model':
    raise ValueError(f"MC_REBALANCING_BANDS={REBALANCING_POLICY!r}; "
                     f"expected 'model' or 'uniform'")

# Safety limit for iterative rebalancing (Q1-Q3 only)
MAX_REBALANCE_ITERATIONS = 10

# ============================================================================
# INFLATION — the model reports in real (today's) dollars
# ============================================================================
# A 50-year nominal horizon is unreadable: a median of $12M at age 87 is roughly
# $3.2M in today's money at 3%. Rather than deflate at the end, every rate is
# converted to real terms at load, so the whole simulation runs in today's
# dollars and initial_value, contributions and (later) withdrawals are all
# directly comparable to what you would spend now.
#
# Inflation is deterministic here, which makes the algebra exact rather than
# approximate: with a fixed i, real = (1 + nominal) / (1 + i) - 1. That is affine
# in the nominal return, so means transform by that formula and dispersions
# (volatility, parameter-uncertainty bands) simply divide by (1 + i).
#
# Note the common shortcut `real = nominal - i` is wrong by ~0.19pp on VOO's
# 9.47% at 3% — small per year, material over 50.
#
# Stochastic inflation, and therefore inflation risk on a withdrawal stream, is
# deliberately deferred; see the v3.1 review notes (unpublished).
# Overridable via MC_INFLATION for sensitivity runs. Env rather than assignment
# because workers re-import under `spawn` — same constraint as PREMIUM_SCALE.
INFLATION = float(os.environ.get('MC_INFLATION', '0.03'))


# ============================================================================
# CRISIS SEVERITY — a STRESS TEST, not a correction
# ============================================================================
# The A2/A4 audit items claimed this model's crisis regime was far too mild,
# citing "2008 volatility north of 40%". THAT WAS A UNITS ERROR: 40% is DAILY
# volatility. This model steps monthly, so monthly-sampled volatility is the
# right comparison, and by that measure the model is well calibrated:
#
#     realised S&P vol, annualised from MONTHLY returns
#       2008 calendar year   21.0%      2020 COVID       25.9%
#       2008-09 crisis       24.7%      1973-74          19.5%
#       full history 1970+   15.2%
#     36-month rolling vol:  p90 20.1%, max 22.2%
#     model's VOO high regime: 20.1% nominal
#
# A4 does not survive either. The worst single month on record is -21.8%
# (October 1987), about -3.8 sigma against a 20%-vol Gaussian — roughly a 4%
# chance of occurring at least once in 600 months, and history has one in 679.
#
# So these knobs are NOT corrections and must not be defaulted on. They answer
# "how bad would markets have to get before bonds earn their place?" — measured
# answer: about 2x the worst three-year stretch on record, at which point a 10%
# bond allocation becomes roughly free (paired CI straddles zero) though still
# not beneficial.
#
#   MC_CRISIS_VOL_SCALE  multiplies high-regime volatility. 2.0 is DOUBLE any
#                        historical monthly-sampled crisis, not a calibration.
#   MC_TAIL_DF           Student-t df for high-regime shocks, rescaled to unit
#                        variance (shape, not size). Measured 2026-10-14 at df 4:
#                        no decision moves; plan levels RISE ~5-6% because the
#                        90th-percentile path sits in the shoulder the t thins.
#                        An earlier "within noise once vol was scaled" predates
#                        the -100% return floor and is unverified.
#
# Read from the environment for the same reason as PREMIUM_SCALE: workers
# re-import under `spawn`, so a value assigned in the parent would not reach them.
CRISIS_VOL_SCALE = float(os.environ.get('MC_CRISIS_VOL_SCALE', '1.0'))
_tail_df = os.environ.get('MC_TAIL_DF', '').strip()
TAIL_DF = float(_tail_df) if _tail_df else None
# Zero means Gaussian — crisis_shock already treats df <= 2 that way — and is
# stored as None so a sweep's 'off' point is not announced as a moved knob.
if TAIL_DF is not None and TAIL_DF <= 0:
    TAIL_DF = None

#   MC_CORR_WINDOW       Years of history the correlation matrices are built on
#                        (loaders.CORR_WINDOW_YEARS_DEFAULT = 15). Another value
#                        loads the `_w<years>` file set, which must have been
#                        generated first. Only the matrices change: regime
#                        labels, vols and the random stream do not, so a run
#                        under another window shares its seeds with a default one.
CORR_WINDOW = int(os.environ.get('MC_CORR_WINDOW', str(_CORR_WINDOW_DEFAULT)))


def crisis_shock(n_assets, tail_df=None):
    """Standardised shocks for one month, Gaussian or Student-t.

    The t is drawn as a scale mixture: one chi-square draw SHARED across assets,
    so a fat-tailed month hits everything at once. That is the point — a crisis
    is a common shock, not n independent surprises — and it produces tail
    dependence the Gaussian copula cannot.

    Rescaled to unit variance so `tail_df` changes the SHAPE of the distribution
    without changing its volatility; crisis magnitude is CRISIS_VOL_SCALE's job.
    """
    z = np.random.randn(n_assets)
    if not tail_df or tail_df <= 2:
        return z
    w = np.random.chisquare(tail_df)
    return z * np.sqrt(tail_df / w) / np.sqrt(tail_df / (tail_df - 2.0))


# Rescales each EQUITY ticker's premium over the market proxy:
#     mu[t] = mu[VOO] + PREMIUM_SCALE * (mu[t] - mu[VOO])
# 1.0 is the model's premium, 0.0 the null hypothesis (every equity earns the
# market return). BONDS ARE EXEMPT: scaling their large negative premium would
# make a low-premium world nonsensically kind to bonds. Env, not assignment.
PREMIUM_SCALE = float(os.environ.get('MC_PREMIUM_SCALE', '1.0'))

# How expected returns are built. 'factor' is the DEFAULT as of 2026-09-10:
# mu = rf + sum(beta*premium) - costs for anything clearing the fit gate,
# rf + spread for bonds, and a historical mean for whatever neither covers.
# 'historical' restores the old behaviour — every ticker's own spliced mean —
# and is what every result recorded before that date used.
#
# Why the switch: a ticker's own mean has standard error sigma/sqrt(T), so
# AVLV's carried a genuine 95% interval of 8.2%-18.0% on 54 years of data. The
# level is not recoverable from a return series, and nothing forced agreement
# between tickers — VTI implied +1.94pp/yr over VOO for two funds correlated at
# ~0.99. See docs/METHODOLOGY.md, Expected Returns.
#
# Read from the environment, like PREMIUM_SCALE and for the same reason: workers
# re-import this module under `spawn`, so a value assigned to a module global in
# the parent process never reaches them.
MU_METHOD = os.environ.get('MC_MU_METHOD', 'factor')
PREMIA_SCENARIO = os.environ.get('MC_PREMIA_SCENARIO', 'full history')
PREMIUM_HAIRCUT = float(os.environ.get('MC_PREMIUM_HAIRCUT', '1.0'))

# Widens (or narrows) the EPISTEMIC band — how unsure the model is about each
# ticker's mean return, sampled once per path. Read from the environment for the
# same reason as PREMIUM_SCALE: workers re-import under `spawn`.
#
# It exists because the shipped bands are a hand-calibrated judgement, and a
# tight one: at AVUV's 0.020 the model implies a ~0.09% probability that AVUV has
# no premium at all, which no reading of the historical record supports. Rather
# than pick a new number by argument, this knob makes the CONSEQUENCE of the
# choice measurable — sweep it and see whether the plan level and the ranking
# move. Do NOT reach for sigma/sqrt(years) as a "fix": `base_volatility` is
# estimated on a 2010+ window while `expected_return` runs from 1973, so that
# standard error mixes two different histories.
PARAM_STD_SCALE = float(os.environ.get('MC_PARAM_STD_SCALE', '1.0'))
# Re-exported from loaders, which owns it because the data pipeline needs it too
# and must not import the engine.
MARKET_PROXY = _MARKET_PROXY

# Re-exported from factors so provenance and reports can read the run's cash-rate
# assumption off the engine, without every caller learning where it lives.
# Unconditional: under MU_METHOD='historical' these are not load-bearing, but the
# run record still says what they were, and it records mu_method beside them so a
# reader can tell.
from montecarlo import factors as _factors                                       # noqa: E402
RF_REAL = _factors.RF_REAL
RF_REAL_SOURCE = _factors.RF_REAL_SOURCE
RF_REAL_AS_OF = _factors.RF_REAL_AS_OF
RF_FORWARD = _factors.RF_FORWARD
RF_REAL_STD = _factors.RF_REAL_STD
INFLATION_STD = _factors.INFLATION_STD
# Whether a fund's estimation error is drawn through the premia it shares with
# every other fund, or on its own. See factors.PARAM_ERRORS.
PARAM_ERRORS = _factors.PARAM_ERRORS
# The edge a single stock earns over its factor build-up (ledger entry N10, unpublished).
# Zero unless MC_SINGLE_STOCK_EDGE states one; re-exported for provenance.
SINGLE_STOCK_EDGE = _factors.SINGLE_STOCK_EDGE
PREMIUM_KEYS = _factors.PREMIUM_KEYS
# How the premia's estimation errors move together (factors.PREMIUM_CORR), and
# its factor — fixed for the whole run, so it is computed once here.
PREMIUM_CORR = np.asarray(_factors.PREMIUM_CORR, dtype=float)
PREMIUM_CHOL = np.linalg.cholesky(PREMIUM_CORR)
# A fixed root for the premium-error stream, kept apart from the main one so
# the market draws are identical under both PARAM_ERRORS settings.
PREMIUM_SEED = 4242
# Per-ticker inflation pass-through, resolved once at import: bonds are
# measurably more exposed than equities and that DIFFERENTIAL is the only part
# of stochastic inflation that can change a decision rather than a level. A
# nominal bond's share of it scales with duration (factors.bond_infl_beta).
INFL_BETA_DEFAULT = _factors.INFL_BETA_EQUITY
INFL_BETA_TIPS = _factors.INFL_BETA_TIPS
INFL_BETA = {t: (_factors.INFL_BETA_TIPS if t in _INFLATION_LINKED
                 else _factors.bond_infl_beta(_BOND_DURATION[t]) if t in _BOND_FUNDS
                 else _factors.INFL_BETA_EQUITY) for t in TICKERS}

# Reserved key under which the MARKET regime's Markov chain is stored inside
# REGIME_TRANSITIONS, so select_next_regime and its CDF cache work on it
# unchanged. Every universe the engine simulates must carry it.
MARKET_REGIME_KEY = '__market__'


def scale_equity_premia(expected_returns, scale, market=MARKET_PROXY):
    """Rescale equity premia over the market proxy. Bonds pass through untouched."""
    if scale == 1.0:
        return dict(expected_returns)
    mkt = expected_returns[market]
    return {t: (mu if t in _BOND_FUNDS else mkt + scale * (mu - mkt))
            for t, mu in expected_returns.items()}


def real_rate(nominal_rate, inflation):
    """Convert a nominal rate of return to real terms. Exact for deterministic
    inflation, since real = (1 + nominal) / (1 + inflation) - 1."""
    return (1.0 + nominal_rate) / (1.0 + inflation) - 1.0


def real_dispersion(nominal_dispersion, inflation):
    """Convert a nominal standard deviation to real terms.

    real is affine in nominal with slope 1/(1+i), so Var scales by 1/(1+i)**2
    and the standard deviation simply divides by (1 + i).
    """
    return nominal_dispersion / (1.0 + inflation)


# ============================================================================
# PARAMETERS - AUTO-LOADED FROM JSON
# ============================================================================
# Parameters are loaded from ticker_parameters_regimes_v8.json at startup
# To update with latest market data: python3 data_collection_regime_model.py
#
# The loader will:
#   - Load all ticker parameters
#   - Use parameter uncertainty overrides where specified
#   - Load regime-dependent correlation matrices
#   - Warn if data is >90 days old
#   - Validate all data structures
#
# See loaders.py for implementation details

# Load parameters silently at module level (runs in both main and worker processes)
# Verbose output only shown once in main process (see bottom of file)
try:
    _params = load_parameters_from_json(
        params_file='ticker_parameters_regimes_v8.json',
        warn_age_days=90,
        verbose=False,  # Silent loading to avoid spam in multiprocessing workers
        corr_window_years=CORR_WINDOW,
    )

    # Extract parameters from loader, converting to real terms. The JSON holds
    # nominal figures (that is what the historical data measures); everything
    # downstream of here is in today's dollars.
    # Nominal first, then deflate. The substitution happens BEFORE real_rate and
    # before scale_equity_premia, so `EXPECTED_RETURNS` keeps its contract —
    # dict[ticker -> real mu] — and nothing downstream, including the regression
    # golden master, can tell the difference.
    _nominal = dict(_params['EXPECTED_RETURNS'])
    _nominal_std = dict(_params['PARAM_UNCERTAINTY_STD'])
    MU_NOTES = []
    _fits = _factors.load_fits()
    if MU_METHOD == 'factor':
        _nominal, MU_NOTES = _factors.build_nominal(
            _nominal, _fits,
            scenario=PREMIA_SCENARIO, haircut=PREMIUM_HAIRCUT)
        # Bands are DERIVED too, not carried over: the hand-set table they
        # replace had gone inverted, giving AVLV 0.07 x vol against VOO's 0.16.
        _nominal_std, _band_notes = _factors.build_param_std(
            _nominal_std, _fits,
            scenario=PREMIA_SCENARIO, haircut=PREMIUM_HAIRCUT)
        MU_NOTES = MU_NOTES + _band_notes
    elif MU_METHOD != 'historical':
        raise ValueError(
            f"MC_MU_METHOD={MU_METHOD!r}; expected 'historical' or 'factor'")

    EXPECTED_RETURNS = scale_equity_premia(
        {t: real_rate(mu, INFLATION) for t, mu in _nominal.items()},
        PREMIUM_SCALE)
    PARAM_UNCERTAINTY_STD = {t: real_dispersion(sd, INFLATION) * PARAM_STD_SCALE
                             for t, sd in _nominal_std.items()}
    # Each fund's split of that band between the premia it shares and itself
    # (factors.premium_shares). Taken from the factor fits under EITHER mu
    # method: a fund's mean is uncertain through the same premia however the
    # point estimate was built. Fractions, so PARAM_STD_SCALE and any band a
    # test zeroes carry straight through.
    _share_fits = _fits or {}
    PREMIUM_SHARES = {}
    for _t in TICKERS:
        _s = _factors.premium_shares(_share_fits.get(_t), scenario=PREMIA_SCENARIO,
                                        haircut=PREMIUM_HAIRCUT)
        if _s is not None:
            PREMIUM_SHARES[_t] = np.asarray(_s, dtype=float)
    VOLATILITY_REGIMES = {
        t: {r: real_dispersion(sd, INFLATION) * (CRISIS_VOL_SCALE if r == 'high' else 1.0)
            for r, sd in by_r.items()}
        for t, by_r in _params['VOLATILITY_REGIMES'].items()}
    # The nominal returns this run ACTUALLY used, not the JSON's. Under
    # MU_METHOD='factor' those differ, and pointing this at the JSON broke the
    # deflation identity that test_production_parameters_are_real checks.
    NOMINAL_EXPECTED_RETURNS = dict(_nominal)
    REGIME_TRANSITIONS = dict(_params['REGIME_TRANSITIONS'])
    # The MARKET regime chain, if the parameters carry one. Stored beside the
    # per-ticker chains so select_next_regime works on it unchanged. Per-ticker
    # regimes still drive each asset's VOLATILITY; this drives the CORRELATION
    # matrix and the tail-shock flag, which are market-state properties.
    MARKET_REGIME = (_params.get('metadata') or {}).get('market_regime')
    if MARKET_REGIME and MARKET_REGIME.get('transitions'):
        REGIME_TRANSITIONS[MARKET_REGIME_KEY] = MARKET_REGIME['transitions']
    CORRELATION_BASELINE = _params['CORRELATION_BASELINE']
    CORRELATION_LOW = _params['CORRELATION_LOW']
    CORRELATION_MEDIUM = _params['CORRELATION_MEDIUM']
    CORRELATION_HIGH = _params['CORRELATION_HIGH']

    # Validate TICKERS list matches loaded data
    LOADED_TICKERS = _params['ticker_list']
    if TICKERS != LOADED_TICKERS:
        print(f"\n⚠️  WARNING: TICKERS list mismatch!")
        print(f"  Script expects: {TICKERS}")
        print(f"  JSON contains:  {LOADED_TICKERS}")
        raise ValueError("Ticker list mismatch - check TICKERS constant vs JSON file")

except FileNotFoundError as e:
    print(f"\n{'='*70}")
    print("ERROR: Cannot load parameters")
    print("="*70)
    print(str(e))
    print("\nPlease run: python3 data_collection_regime_model.py")
    print("="*70)
    raise


# ============================================================================
# MONTE CARLO HELPER FUNCTIONS
# ============================================================================

# Cache of per-ticker, per-regime cumulative transition probabilities.
# Rebuilt automatically if REGIME_TRANSITIONS is reassigned (the validation tools
# swap in synthetic universes), detected by object identity.
_regime_cdf_cache = {'src': None, 'table': None}


def _regime_cdf_table():
    """Build (or reuse) the cumulative transition table keyed by ticker and regime."""
    global _regime_cdf_cache
    if _regime_cdf_cache['src'] is not REGIME_TRANSITIONS:
        table = {}
        for ticker, by_regime in REGIME_TRANSITIONS.items():
            table[ticker] = {}
            for current, probs in by_regime.items():
                # Outcome order must follow dict order, matching what
                # np.random.choice(list(probs.keys()), p=...) used to see.
                outcomes = tuple(probs.keys())
                cdf = np.cumsum(np.asarray(list(probs.values()), dtype=float))
                cdf /= cdf[-1]
                table[ticker][current] = (cdf, outcomes)
        _regime_cdf_cache = {'src': REGIME_TRANSITIONS, 'table': table}
    return _regime_cdf_cache['table']


def select_next_regime(ticker, current_regime, u=None):
    """Select next regime based on Markov transition.

    Bit-identical to the previous `np.random.choice(regimes, p=probabilities)`:
    for a scalar draw, choice consumes exactly one random_sample() and resolves it
    with cdf.searchsorted(u, side='right'). Replicating that inline avoids
    rebuilding the outcome list and the CDF on every one of the ~5,400 calls per
    path, which was 37% of total runtime.

    `u` lets the caller supply the uniform from a draw it took for the whole
    universe, so a fund's transition reads a FIXED position in the stream
    whatever else the plan holds. Omitted, it draws one itself.
    """
    cdf, outcomes = _regime_cdf_table()[ticker][current_regime]
    if u is None:
        u = np.random.random()
    return outcomes[int(cdf.searchsorted(u, side='right'))]

def premium_draw(sim_id):
    """This path's premium errors, one per PREMIUM_KEYS entry, correlated as
    PREMIUM_CORR says. From a generator of its own seeded by the path, so it is
    the same draw whatever a plan holds and it never moves the main stream."""
    z = np.random.default_rng((PREMIUM_SEED, sim_id)).standard_normal(len(PREMIUM_KEYS))
    return PREMIUM_CHOL @ z


def estimation_error(ticker, band, own_z, prem_z=None):
    """One fund's draw from its epistemic band, for one path.

    Independent (`prem_z` None, or a fund the factor model does not cover):
    band * own_z. Shared: the band splits into the premia the fund loads on,
    share @ prem_z, whose variance is q = share' C share with C the premia's
    error correlation, and the fund's own remainder, sqrt(1 - q) * own_z. The
    two are independent, so the fund's band is exactly `band` either way —
    what changes is that funds loading on the same or related premia now move
    TOGETHER, as their errors really do. A fund whose loadings would put more
    than its whole band in the shared part (q > 1) keeps the band: the shared
    part is scaled to fit and nothing is left over.
    """
    share = PREMIUM_SHARES.get(ticker) if prem_z is not None else None
    if share is None:
        return band * own_z
    q = float(share @ PREMIUM_CORR @ share)
    if q > 1.0:
        return band * float(share @ prem_z) / q ** 0.5
    return band * (float(share @ prem_z) + (1.0 - q) ** 0.5 * own_z)


_REGIMES = ('low', 'medium', 'high')
_REGIME_INDEX = {r: i for i, r in enumerate(_REGIMES)}


def _regime_arrays(tickers, coupled):
    """Per-fund transition rows as arrays, for stepping every held fund at once.

    cdf[m, h, c] is fund h's cumulative row from regime c when the market moves
    to m, padded with +inf; nxt[m, h, c, k] is the regime that outcome k means.
    select_next_regime resolves u with cdf.searchsorted(u, 'right') — the count
    of entries <= u — so counting (cdf <= u) gives the same outcome, exactly.
    """
    table = _regime_cdf_table()
    n = len(tickers)
    cdf = np.full((3, n, 3, 3), np.inf)
    nxt = np.zeros((3, n, 3, 3), dtype=np.int64)
    for mi, m in enumerate(_REGIMES):
        for h, t in enumerate(tickers):
            key = (t, m) if coupled.get(t) else t
            for ci, c in enumerate(_REGIMES):
                row, outcomes = table[key][c]
                cdf[mi, h, ci, :len(row)] = row
                idx = [_REGIME_INDEX[o] for o in outcomes]
                nxt[mi, h, ci] = idx + [idx[-1]] * (3 - len(idx))
    return cdf, nxt


def get_volatility_for_regime(ticker, regime):
    """Get volatility for a specific regime"""
    return VOLATILITY_REGIMES[ticker][regime]

def get_correlation_for_regime(regime):
    """Get correlation matrix for a specific regime"""
    if regime == 'low':
        return CORRELATION_LOW
    elif regime == 'medium':
        return CORRELATION_MEDIUM
    elif regime == 'high':
        return CORRELATION_HIGH
    else:
        return CORRELATION_BASELINE

# ============================================================================
# REBALANCING HELPER FUNCTIONS
# ============================================================================

def calculate_band_limits(target_weight, ticker):
    """
    Calculate rebalancing band limits using max(relative, absolute).

    Args:
        target_weight: Target allocation (e.g., 0.38 for 38%)
        ticker: Ticker symbol

    Returns:
        (lower_limit, upper_limit) tuple

    Example:
        VOO target 0.38, relative 0.15, absolute 0.05
        - Relative: 0.38 × 0.15 = 0.057 (5.7pp)
        - Absolute: 0.05 (5pp)
        - Use larger: 0.057
        - Bands: [0.323, 0.437]
    """
    bands = REBALANCING_BANDS[ticker]
    relative_band = target_weight * bands['relative']
    absolute_band = bands['absolute']
    band_width = max(relative_band, absolute_band)

    return (target_weight - band_width, target_weight + band_width)

def check_band_breach(current_weight, target_weight, ticker):
    """
    Check if current weight is outside rebalancing bands.

    Returns: True if breached, False otherwise
    """
    lower, upper = calculate_band_limits(target_weight, ticker)
    return current_weight < lower or current_weight > upper

def calculate_band_consumption(current_weight, target_weight, ticker):
    """
    Calculate percentage of band consumed (0.0-1.0).

    Used for 70% threshold in annual drift check.

    Returns:
        0.0-1.0 representing % of band consumed
        0.0 = at target
        0.70 = 70% toward band edge (annual trigger)
        1.0+ = breached band

    Example:
        VOO target 0.38, bands [0.323, 0.437]
        Current 0.42, upper band width 0.057
        Consumption: (0.42 - 0.38) / 0.057 = 0.70
    """
    lower, upper = calculate_band_limits(target_weight, ticker)

    if current_weight > target_weight:
        # Drifting upward
        band_width = upper - target_weight
        drift = current_weight - target_weight
    else:
        # Drifting downward
        band_width = target_weight - lower
        drift = target_weight - current_weight

    return drift / band_width if band_width > 0 else 0.0

def calculate_portfolio_volatility(weights, regime_vols, corr_matrix):
    """Annual portfolio volatility sqrt(w' Σ w), from the CURRENT regime's
    volatilities and correlation matrix."""
    # Convert to numpy arrays in TICKERS order
    w = np.array([weights[ticker] for ticker in TICKERS])
    vols = np.array([regime_vols[ticker] for ticker in TICKERS])

    # Covariance matrix: Σ = diag(σ) × ρ × diag(σ)
    vol_matrix = np.outer(vols, vols)
    cov_matrix = vol_matrix * corr_matrix

    # Portfolio variance: w^T × Σ × w
    portfolio_variance = np.dot(w, np.dot(cov_matrix, w))

    return np.sqrt(portfolio_variance)

def get_breached_positions(current_weights, target_weights, tickers=None):
    """
    Get list of tickers that breached their bands.

    `tickers` narrows the scan to the funds a plan can hold. Every other fund
    sits at exactly zero weight and zero target, which no band can breach.

    Returns: List of ticker names (e.g., ['VXUS', 'COST'])
    """
    breached = []
    for ticker in (tickers or TICKERS):
        if check_band_breach(current_weights[ticker], target_weights[ticker], ticker):
            breached.append(ticker)
    return breached

def get_positions_near_bands(current_weights, target_weights, threshold=0.6,
                             tickers=None):
    """
    Get tickers with >60% band consumption (approaching breach).

    Used to decide between PARTIAL and FULL rebalancing.

    Returns: List of ticker names
    """
    near_bands = []
    for ticker in (tickers or TICKERS):
        consumption = calculate_band_consumption(
            current_weights[ticker],
            target_weights[ticker],
            ticker
        )
        if consumption > threshold:
            near_bands.append(ticker)
    return near_bands

def find_most_overweight(current_weights, target_weights):
    """
    Find ticker with largest positive drift from target.

    Used to source funds for partial rebalancing.

    Returns: Ticker name (e.g., 'VOO')
    """
    max_drift = 0
    most_overweight = None

    for ticker in TICKERS:
        drift = current_weights[ticker] - target_weights[ticker]
        if drift > max_drift:
            max_drift = drift
            most_overweight = ticker

    return most_overweight

def find_most_underweight(current_weights, target_weights):
    """Ticker with the largest shortfall from target — where partial
    rebalancing puts the sale of an overweight breach."""
    max_shortfall = 0
    most_underweight = None

    for ticker in TICKERS:
        shortfall = target_weights[ticker] - current_weights[ticker]
        if shortfall > max_shortfall:
            max_shortfall = shortfall
            most_underweight = ticker

    return most_underweight

def annual_drift_check_triggered(current_weights, target_weights,
                                  regime_vols, corr_matrix, tickers=None):
    """Annual (Q4) drift check: any position >70% toward its band edge, or
    portfolio volatility >10% above the target's, at the current regime's
    volatilities. Returns (triggered, reason)."""
    # Check Trigger 1: 70% band consumption
    for ticker in (tickers or TICKERS):
        consumption = calculate_band_consumption(
            current_weights[ticker],
            target_weights[ticker],
            ticker
        )
        if consumption > 0.70:
            return (
                True,
                f"Band consumption: {ticker} {consumption*100:.0f}% "
                f"({current_weights[ticker]:.3f} vs {target_weights[ticker]:.3f} target)"
            )

    # Check Trigger 2: 10% volatility increase
    current_vol = calculate_portfolio_volatility(current_weights, regime_vols, corr_matrix)
    target_vol = calculate_portfolio_volatility(target_weights, regime_vols, corr_matrix)

    vol_increase = (current_vol / target_vol) - 1.0
    if vol_increase > 0.10:
        return (
            True,
            f"Volatility shift: {current_vol*100:.1f}% vs {target_vol*100:.1f}% target "
            f"(+{vol_increase*100:.1f}%)"
        )

    return (False, None)

def rebalance_decision(current_weights, target_weights, month,
                       regime_vols, corr_matrix, tickers=None):
    """
    Determine rebalancing strategy based on conditions.

    Decision tree:
    - Q4: Check bands + annual drift → FULL if either triggered, else NONE
    - Q1-Q3: Check bands only → PARTIAL/FULL based on conditions

    Args:
        current_weights: Current portfolio weights
        target_weights: Target portfolio weights
        month: Month within year (3, 6, 9, or 12)
        regime_vols: Current regime volatilities
        corr_matrix: Current regime correlation matrix

    Returns:
        (decision: str, reason: str)
        - decision: 'FULL', 'PARTIAL', or 'NONE'
        - reason: Explanation for logging

    Q4 Logic (December - Month 12):
        Check threshold bands AND annual drift
        If EITHER triggered → FULL rebalance
        No iterative rebalancing in Q4

    Q1-Q3 Logic (Months 3, 6, 9):
        Multiple breaches → FULL
        Single breach + many near bands → FULL
        Single breach + few near bands → PARTIAL
        No breaches → NONE
    """
    breached = get_breached_positions(current_weights, target_weights, tickers)

    # Q4 (December): Combined check
    if month == 12:
        # Check threshold bands
        if len(breached) > 0:
            return ('FULL', f"Q4 threshold breach: {len(breached)} position(s)")

        # Check annual drift conditions
        annual_triggered, annual_reason = annual_drift_check_triggered(
            current_weights, target_weights, regime_vols, corr_matrix, tickers
        )
        if annual_triggered:
            return ('FULL', f"Q4 annual drift: {annual_reason}")

        # Neither triggered
        return ('NONE', None)

    # Q1-Q3: Iterative breach resolution
    if len(breached) == 0:
        return ('NONE', None)

    # Multiple breaches → FULL
    if len(breached) >= 2:
        return ('FULL', f"Multiple breaches: {', '.join(breached)}")

    # Check positions near bands (excluding already-breached positions)
    near_bands = [t for t in get_positions_near_bands(current_weights, target_weights, threshold=0.7,
                                                      tickers=tickers)
                  if t not in breached]

    # Single breach + many OTHER positions near bands → FULL
    if len(near_bands) > 1:
        return ('FULL', f"Single breach ({breached[0]}) + {len(near_bands)} other near bands")

    # Single breach + few near bands → PARTIAL
    return ('PARTIAL', f"Single breach: {breached[0]}")

def execute_partial_rebalance(current_values, target_weights, breached_tickers):
    """
    Execute partial rebalancing: one bilateral trade that returns the breached
    ticker to target.

    Process:
        1. For first breached ticker (others wait for next iteration)
        2. Calculate amount needed to reach target
        3. Underweight breach: sell the most overweight ticker, buy the breach.
           Overweight breach: sell the breach, buy the most underweight ticker.
        4. Leave all other positions unchanged

    Args:
        current_values: Dict of current $ values per ticker
        target_weights: Dict of target % allocations
        breached_tickers: List of breached ticker names

    Returns:
        (new_values: dict, trades: list)
        - new_values: Updated portfolio values
        - trades: List of trade descriptions for logging
    """
    total_value = sum(current_values.values())
    current_weights = {ticker: current_values[ticker] / total_value
                       for ticker in TICKERS}

    # Process first breached position only
    breached = breached_tickers[0]
    target_value = total_value * target_weights[breached]
    needed = target_value - current_values[breached]

    # The counterparty sits on the other side of target: an underweight breach
    # is funded by the most overweight fund, an overweight breach is sold into
    # the most underweight one. Using the most overweight fund for both made an
    # overweight breach either a no-op (it WAS that fund) or a purchase of it.
    if needed >= 0:
        counterparty = find_most_overweight(current_weights, target_weights)
        sold, bought = counterparty, breached
    else:
        counterparty = find_most_underweight(current_weights, target_weights)
        sold, bought = breached, counterparty

    # Execute bilateral trade
    new_values = current_values.copy()
    new_values[breached] = target_value
    new_values[counterparty] -= needed

    # Create trade description
    trade_desc = (
        f"Partial: Sold {abs(needed)/total_value*100:.1f}% {sold} "
        f"(${abs(needed)/1e6:.1f}M), Bought {abs(needed)/total_value*100:.1f}% {bought} "
        f"(${abs(needed)/1e6:.1f}M)"
    )

    return (new_values, [trade_desc])

def traded_dollars(before, after):
    """Dollars that changed hands in one rebalance.

    Halved because every dollar moved appears twice — once as a sell and once
    as a buy — and turnover is conventionally the one-way figure.

    MEASURED, NOT CHARGED. `factors.ROUND_TRIP` already prices each FUND's
    internal turnover inside its expected return; nothing prices YOUR trades
    between funds, and until now nothing counted them either, so the cost could
    only be estimated. `REBALANCING_BANDS` is fourteen hand-tuned pairs audited
    on the SHAPE of the band-versus-volatility relationship (corr +0.860)
    because the test that would falsify them — what they cost in turnover — did
    not exist.
    """
    # A FIXED order: summing over a set made the last digit depend on string
    # hashing, which Python randomises per process, so the same path reported
    # different turnover in different runs (49 of 72 paths, 2026-10-03).
    keys = list(before) + [t for t in after if t not in before]
    return 0.5 * sum(abs(after.get(t, 0.0) - before.get(t, 0.0)) for t in keys)


def execute_full_rebalance(current_values, target_weights):
    """
    Execute full rebalancing: reset ALL positions to exact targets.

    Args:
        current_values: Dict of current $ values per ticker
        target_weights: Dict of target % allocations

    Returns:
        (new_values: dict, trade_description: str)
    """
    total_value = sum(current_values.values())

    # Reset all positions to target
    new_values = {ticker: total_value * target_weights[ticker]
                  for ticker in TICKERS}

    # Count changes
    sells = sum(1 for ticker in TICKERS
                if new_values[ticker] < current_values[ticker])
    buys = sum(1 for ticker in TICKERS
               if new_values[ticker] > current_values[ticker])
    unchanged = len(TICKERS) - sells - buys

    trade_desc = f"Full rebalance: {sells} sells, {buys} buys, {unchanged} unchanged"

    return (new_values, trade_desc)

PSD_EIG_FLOOR = 1e-8   # matches data_collection_regime_model.fix_correlation_matrix


def near_psd_higham(A, max_iter=100):
    """
    Higham's algorithm for nearest positive semi-definite matrix
    """
    n = A.shape[0]
    
    # Symmetrize
    A = (A + A.T) / 2
    
    # Initial values
    Y = A.copy()
    dS = np.zeros_like(A)
    
    for _ in range(max_iter):
        # Project to S (PSD matrices)
        R = Y - dS
        eigvals, eigvecs = eigh(R)
        eigvals[eigvals < 0] = 0
        X = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        # Update dS
        dS = X - R
        
        # Project to U (unit diagonal)
        Y = X.copy()
        np.fill_diagonal(Y, 1)
        
        # Check convergence
        if np.allclose(Y, X, atol=1e-6):
            break
    
    # Final symmetrization
    Y = (Y + Y.T) / 2
    np.fill_diagonal(Y, 1)

    # FLOOR THE SPECTRUM, then renormalise. Higham's last step is the projection
    # onto unit diagonal, which is not a projection onto PSD -- so the result is
    # PSD only to tolerance. np.linalg.cholesky needs strict positive
    # definiteness, so without this the repair returns a matrix the caller
    # cannot factor: measured on [[1,.9,-.9],[.9,1,.9],[-.9,.9,1]] the loop
    # lands at min eig -4.5e-06 and cholesky_factor RAISES, having already spent
    # its one retry on this same function. Unreachable for the matrices actually
    # shipped (all four are PD, so the repair never runs and no simulated number
    # moves), but every regen clips at least one negative eigenvalue per regime
    # matrix upstream, so an indefinite input here is a question of when.
    # Same floor and same renormalisation as the pipeline's
    # fix_correlation_matrix -- one repair, two implementations, now agreeing.
    eigvals, eigvecs = eigh(Y)
    if eigvals.min() < PSD_EIG_FLOOR:
        eigvals[eigvals < PSD_EIG_FLOOR] = PSD_EIG_FLOOR
        Y = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d = np.sqrt(np.diag(Y))
        Y = Y / np.outer(d, d)
        Y = (Y + Y.T) / 2
        np.fill_diagonal(Y, 1)

    return Y

def cholesky_factor(corr_matrix):
    """Validate/repair a correlation matrix and return its Cholesky factor.

    Kept out of the monthly loop: there are only three reachable regime matrices
    per path, and repeating the symmetry check, eigenvalue PSD check and
    decomposition 600 times per path was 30% of total runtime.
    """
    if not np.allclose(corr_matrix, corr_matrix.T, atol=1e-6):
        corr_matrix = (corr_matrix + corr_matrix.T) / 2

    eigvals = np.linalg.eigvalsh(corr_matrix)
    if np.min(eigvals) < 0:
        corr_matrix = near_psd_higham(corr_matrix)

    try:
        return np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(near_psd_higham(corr_matrix))


def calculate_annual_contribution(age, contribution_schedule, inflation=0.0):
    """Calculate total contribution for given age, in real terms.

    Account `growth_rate`s are nominal (they were fitted to nominal contribution
    histories), so they are deflated here. A contribution growing at exactly the
    inflation rate is therefore flat in real terms — which is the point: the
    taxable account's 3% 'inflation proxy' correctly becomes 0% real growth.

    `inflation` defaults to 0.0 so the function is still a pure nominal
    calculator when called directly.
    """
    total = 0
    accounts = contribution_schedule['accounts']
    # The age the amounts were last trued up at — NOT the investor's age. It was
    # a bare literal 36 here and in no config anywhere, so updating your age and
    # your contribution amounts in the same sitting gave the new amounts a year
    # of compounding they already contained. Defaults to 36 so every record
    # written before the field existed reproduces exactly.
    measured_at_age = contribution_schedule.get('measured_at_age', 36)

    for account_name, account_config in accounts.items():
        base_amount = account_config['annual_amount']
        growth_rate = real_rate(account_config['growth_rate'], inflation)
        stop_age = account_config.get('stop_age')
        
        # Check if contributions have stopped
        if stop_age is not None and age > stop_age:
            continue
        
        years_elapsed = age - measured_at_age
        
        # Apply growth
        amount = base_amount * ((1 + growth_rate) ** years_elapsed)
        total += amount
    
    return total

def normalize_allocation_plan(allocation):
    """Accept either a static allocation dict or a glide plan; return a plan.

    A plan is an ordered tuple of (age, allocation_dict). A bare dict becomes a
    one-waypoint plan starting at -inf, so static portfolios take exactly the same
    code path — and produce bit-identical results, since a single waypoint can
    never fire a switch.
    """
    if isinstance(allocation, dict):
        return ((float('-inf'), allocation),)
    return tuple((float(age), alloc) for age, alloc in allocation)


def _resolve_initial_withdrawal(wd_cfg):
    """First-year withdrawal in real dollars: `target_real_annual`, a fixed
    dollar bar every strategy faces, which is what makes a comparison valid."""
    return float(wd_cfg['target_real_annual'])


def run_single_simulation(args):
    """Run a single Monte Carlo simulation with optional rebalancing.

    `args` is (allocation, config, sim_id). `allocation` is either a static
    {ticker: weight} dict or a glide plan: an ordered sequence of
    (age, {ticker: weight}) waypoints (see glide.py). Waypoints are steps — at the
    switch age the target allocation simply becomes the next one.
    """
    allocation_plan, config, sim_id = args
    allocation_plan = normalize_allocation_plan(allocation_plan)
    allocation = allocation_plan[0][1]

    np.random.seed(42 + sim_id)

    # Extract parameters
    initial_value = config['initial_value']
    time_horizon_years = config['time_horizon_years']
    n_months = time_horizon_years * 12
    current_age = config['age']
    contribution_schedule = config['contribution_schedule']
    rebalancing_strategy = config.get('rebalancing_strategy', 'none')

    # Initialize
    portfolio_value = np.zeros(n_months + 1)
    portfolio_value[0] = initial_value

    monthly_returns = []

    # Initialize regimes for all tickers
    current_regimes = {ticker: 'medium' for ticker in TICKERS}

    # Active tickers are the UNION across every waypoint, so the fund set — and
    # therefore the size of each month's random draw and the Cholesky submatrix —
    # is constant for the whole path even when the allocation switches. A ticker
    # not yet held simply sits at zero. For a static portfolio the union is the
    # same single set, so results are unchanged.
    active_tickers = [ticker for ticker in TICKERS
                      if any(alloc.get(ticker, 0) > 0 for _, alloc in allocation_plan)]
    active_weights = np.array([allocation[ticker] for ticker in active_tickers])
    plan_index = 0

    # COMMON RANDOM NUMBERS ACROSS FUND SETS. Every draw below is taken for the
    # WHOLE universe, in TICKERS order, and a plan reads only the positions of
    # the funds it holds — so a fund's market on a given path is the same
    # whatever else is held. Until 2026-10-02 the draws were sized to the plan's
    # own funds: `166` and `166>166+5@94`, the same strategy until 94, saw
    # different markets and read 587k and 603k, and every head-to-head between
    # strategies holding different funds was silently unpaired.
    n_universe = len(TICKERS)
    ticker_indices = [TICKERS.index(ticker) for ticker in active_tickers]

    # Initialize individual ticker values for rebalancing
    ticker_values = {ticker: initial_value * allocation[ticker] for ticker in TICKERS}

    # Initialize rebalancing statistics
    rebalancing_stats = {
        'count': 0,
        'partial_count': 0,
        'full_count': 0,
        'quarters_checked': 0,
        'q4_sessions': 0,
        'iterations_total': 0,
        'triggers': {},
        'trades': [],
        'traded': 0.0,
        'value_months': 0.0,
        'quarterly_counts': {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0},
        'q1q3_decision_reasons': {
            'multiple_breaches': 0,
            'near_bands': 0,
            'partial': 0
        }
    }

    # STAGE 2: Apply parameter uncertainty to expected returns (once per path)
    # This captures estimation risk - uncertainty about the TRUE expected return
    # Sample adjusted returns from N(base_return, param_uncertainty)
    # ONE draw for the whole path, shared by every ticker. The real cash rate is
    # common to all of them — a world where cash yields 2% real is one where
    # every asset's expected return is higher — so it cannot be represented by
    # the independent per-ticker draws below, which largely cancel across 25
    # funds. Before 2026-09-11 the model was certain about the one input nobody
    # can know and unsure about the 25 it can estimate. See factors.RF_REAL_STD.
    #
    # Drawn BEFORE the per-ticker loop so the stream position is deterministic
    # regardless of how many tickers a portfolio holds.
    rf_shock = np.random.normal(0, RF_REAL_STD) if RF_REAL_STD else 0.0

    # STOCHASTIC INFLATION. One draw per path, like the cash rate — but applied
    # through a per-ticker pass-through, because that is the whole point. A
    # shock hitting every asset equally would be another level-only term; what
    # can actually change a decision is that NOMINAL BONDS lose more real return
    # to inflation than equities do (-1.15 against -0.80 per 1pp, measured
    # 1947-2026 net of the cash channel). Until 2026-09-11 the model deflated by
    # a constant, so a nominal bond carried no inflation risk whatsoever.
    infl_shock = np.random.normal(0, INFLATION_STD) if INFLATION_STD else 0.0

    # SHARED PREMIUM ERRORS (factors.PARAM_ERRORS): one draw per premium per
    # path, passed through each fund's loadings. From a generator of its OWN,
    # seeded by the path, so the main stream — and every market draw on it — is
    # identical under 'shared' and 'independent': the two settings compare
    # paired, and 'independent' reproduces the engine before 2026-10-02 exactly.
    prem_z = premium_draw(sim_id) if PARAM_ERRORS == 'shared' else None
    # One estimation-error draw per fund in the UNIVERSE, read by position.
    param_z = np.random.standard_normal(n_universe)
    adjusted_returns = {
        ticker: EXPECTED_RETURNS[ticker] + rf_shock
                + INFL_BETA.get(ticker, INFL_BETA_DEFAULT) * infl_shock
                + estimation_error(ticker, PARAM_UNCERTAINTY_STD[ticker],
                                   param_z[pos], prem_z)
        for ticker, pos in zip(active_tickers, ticker_indices)
    }

    # ---- per-path invariants, hoisted out of the monthly loop --------------------
    # None of this depends on the month, yet all of it used to be recomputed 600
    # times per path. The Cholesky factors in particular were the second-largest
    # cost in the engine (30% of runtime) despite there being only three reachable
    # regime matrices. Draw order is untouched, so results are bit-identical.
    # The FULL universe's factor, keeping only the held funds' rows: row i of
    # L @ z is fund i's shock in the all-fund simulation, so it cannot depend on
    # which other rows were kept. Factoring the held SUBMATRIX instead (as until
    # 2026-10-02) gives the right covariance but a different shock for the same z.
    chol_by_regime = {
        regime: cholesky_factor(get_correlation_for_regime(regime))[ticker_indices, :]
        for regime in ('low', 'medium', 'high')
    }
    # Hoisted: dict lookups per month for values that cannot change mid-path.
    _coupled = {t: (t, 'low') in REGIME_TRANSITIONS for t in active_tickers}
    market_regime = 'medium'
    # ITEM 3 — WHAT THE PATH ACTUALLY VISITED. The regime machinery is the
    # model's headline claim and has been reworked five times; nothing ever
    # checked that a RUN reproduces the chain it was built from. Three ints per
    # path, incremented in a loop that already branches on the regime, so the
    # cost is not measurable — and no np.random call is added, so the draw
    # order and the golden master are untouched.
    _regime_months = {'low': 0, 'medium': 0, 'high': 0}
    # The per-fund regime step as arrays (see _regime_arrays) — the same draws
    # and the same outcomes as stepping each fund through select_next_regime.
    _rcdf, _rnext = _regime_arrays(active_tickers, _coupled)
    _held = np.arange(len(active_tickers))
    _pos = np.asarray(ticker_indices, dtype=np.int64)
    _reg = np.full(len(active_tickers), _REGIME_INDEX['medium'], dtype=np.int64)
    _rvol = np.array([[get_volatility_for_regime(t, r) / np.sqrt(12) for r in _REGIMES]
                      for t in active_tickers])
    # STAGE 2 draws are fixed for the whole path, so the drift vector is too.
    monthly_means = np.array([adjusted_returns[ticker] / 12
                              for ticker in active_tickers])

    # ---- withdrawal setup --------------------------------------------------------
    # Absent a 'withdrawal' block this is accumulation-only and every branch below
    # is skipped, so existing results are untouched.
    wd_cfg = config.get('withdrawal')
    wd_state = None                 # policies.PolicyState, created at start_age
    monthly_withdrawal = 0.0
    spending = []                   # realised TOTAL annual spend (portfolio + SS)
    depletion_age = None
    ss_cfg = config.get('social_security')
    if wd_cfg is not None:
        wd_policy_name = wd_cfg.get('policy', 'guyton_klinger')
        wd_policy = _policies.POLICIES[wd_policy_name]
        wd_self_directed = wd_policy_name in _policies.SELF_DIRECTED
        wd_source = _policies.SOURCING[wd_cfg.get('sourcing', 'excess')]
        wd_start_age = wd_cfg['start_age']
        wd_end_age = current_age + time_horizon_years

    # Simulation loop
    for month in range(n_months):
        # Calculate age
        age = current_age + month / 12

        # Glide waypoint due? Switch the target and move the book to it straight
        # away. Waiting for the next quarterly band check would misrepresent the
        # decision — derisking is an intentional act, not a drift the bands
        # happen to notice up to three months later.
        while (plan_index + 1 < len(allocation_plan)
               and age >= allocation_plan[plan_index + 1][0]):
            plan_index += 1
            allocation = allocation_plan[plan_index][1]
            if rebalancing_strategy != 'none':
                _before = dict(ticker_values)
                ticker_values, trade_desc = execute_full_rebalance(
                    ticker_values, allocation)
                rebalancing_stats['traded'] += traded_dollars(
                    _before, ticker_values)
                rebalancing_stats['count'] += 1
                rebalancing_stats['full_count'] += 1
                rebalancing_stats['triggers']['glide_switch'] = \
                    rebalancing_stats['triggers'].get('glide_switch', 0) + 1
                rebalancing_stats['trades'].append(f"Glide -> {allocation_plan[plan_index]}")
            else:
                total = sum(ticker_values.values())
                ticker_values = {t: total * allocation.get(t, 0.0) for t in TICKERS}

        # Annual contribution (at month 0 of each year)
        if month % 12 == 0:
            annual_contrib = calculate_annual_contribution(
                int(age), contribution_schedule, INFLATION)
            monthly_contrib = annual_contrib / 12

            # Annual withdrawal, recomputed on the same cadence. The policy sees
            # the balance at the start of the year, so a bad year shows up in the
            # NEXT year's spending — which is how a real guardrail rule works.
            if wd_cfg is not None and age >= wd_start_age:
                balance = sum(ticker_values.values())
                if wd_state is None:
                    w_initial = _resolve_initial_withdrawal(wd_cfg)
                    wd_state = _policies.PolicyState(
                        w_initial=w_initial, v_at_retirement=balance,
                        initial_rate=(w_initial / balance) if balance > 0 else 0.0,
                        w_prior_year=w_initial,
                        retirement_age=wd_start_age, end_age=wd_end_age,
                        assumed_real_return=wd_cfg.get('assumed_real_return', 0.035),
                        vpw_planning_age=wd_cfg.get(
                            'vpw_planning_age',
                            _policies.vpw_planning_age(wd_end_age)),
                        vpw_bequest=float(wd_cfg.get('vpw_bequest', 0.0)),
                        # Resolved HERE, inside the worker, from the config that
                        # was pickled to it — never from a module global the
                        # parent set, which `spawn` would not carry across.
                        **{f'gk_{k}': v for k, v in _policies.gk_guardrails(
                            wd_cfg.get('gk_response')).items()})
                    if wd_self_directed:
                        # This policy sets its own opening draw; the target is a
                        # bar it is judged against, not a control. Safe to call
                        # here because a self-directed policy reads only the
                        # fields already populated above (planning age, assumed
                        # return, reserve) — never w_initial or initial_rate.
                        own = wd_policy(age, balance, wd_state)
                        wd_state = wd_state._replace(
                            w_initial=own, w_prior_year=own,
                            initial_rate=(own / balance) if balance > 0 else 0.0)
                        w_initial = own
                    annual_withdrawal = w_initial
                else:
                    annual_withdrawal = wd_policy(age, balance, wd_state)

                # Social Security funds part of the need, so the portfolio only
                # has to cover the remainder. The benefit arrives regardless of
                # what the portfolio does, which is why it also puts a FLOOR
                # under realised spending.
                ss = _policies.ss_income(age, ss_cfg)
                requested = max(0.0, annual_withdrawal)
                from_portfolio = max(0.0, requested - ss)

                # CARE COSTS COME OUT OF THE PORTFOLIO BUT ARE NOT SPENDING.
                # A nursing home is a cost, not consumption you enjoy, so it
                # drains the balance and leaves `total_spend` alone — which is
                # what lets the adequacy statistic show the knock-on effect in
                # LATER years rather than recording the care year as a bumper
                # one. Adding it to spending would have made a $120k shock look
                # like a $120k raise.
                care = _policies.ltc_cost(age, wd_cfg.get('ltc'))
                from_portfolio += care

                # Depletion means the portfolio can no longer fund ITS share —
                # not that it reached exactly zero, which never happens: a
                # shrinking balance still earns a return, so it decays
                # asymptotically (spending trails off to 1e-21 while the naive
                # `balance <= 0` test never fires).
                drawn = min(from_portfolio, max(balance, 0.0))
                if depletion_age is None and drawn < from_portfolio - 1e-9:
                    depletion_age = age

                # Total spending is what actually reaches you: the portfolio draw
                # plus the benefit. It can exceed the policy's request when the
                # benefit alone is larger than the remaining need.
                # `drawn` may have been capped by the balance, so take the
                # care cost off the top: an underfunded year shorts the
                # LIFESTYLE, not the nursing home.
                drawn_care = min(care, drawn)
                total_spend = (drawn - drawn_care) + ss
                wd_state = wd_state._replace(w_prior_year=requested)
                monthly_withdrawal = drawn / 12
                spending.append(total_spend)

        # Add monthly contribution proportionally to all tickers.
        # net_flow is the external cash moved this month. It must be excluded from
        # the reported return below (Modified Dietz) or the deposit gets booked as
        # investment performance. Accumulated rather than assumed equal to
        # monthly_contrib so a negative flow (withdrawals, later phases) and any
        # allocation not summing to exactly 1.0 are both handled correctly.
        net_flow = 0.0
        for ticker in TICKERS:
            if allocation[ticker] > 0:
                flow = monthly_contrib * allocation[ticker]
                ticker_values[ticker] += flow
                net_flow += flow

        # Withdrawal: a negative external flow, so it belongs in net_flow too.
        # Which holdings it comes from is the sourcing rule's job.
        if monthly_withdrawal > 0:
            available = sum(ticker_values[t] for t in active_tickers)
            take = min(monthly_withdrawal, available)
            if take > 0:
                sold = wd_source(ticker_values, allocation, take, active_tickers)
                for ticker, amount in sold.items():
                    # Clamp at zero: a fully drawn-down holding can land a hair
                    # below it on floating-point residue (observed -9.2e-13 on a
                    # depleted path). Harmless numerically, but it violates the
                    # no-negative-value constraint and would corrupt anything
                    # downstream that takes a ratio or a log of the balance.
                    ticker_values[ticker] = max(0.0, ticker_values[ticker] - amount)
                net_flow -= take
            if available - take <= 0:
                monthly_withdrawal = 0.0

        # THE MARKET MOVES FIRST, and every ticker's transition is conditioned on
        # where it moved to. Drawing them independently — which is what happened
        # until 2026-09-10 — cannot produce the joint tail this model most needs
        # to get right: correlations spiking WHILE most assets are volatile.
        # Historically, given a high market regime 53% of tickers sit in their
        # own high-vol state against 10% in a low one, and P(medium -> high) for
        # the average ticker runs 0.019 / 0.048 / 0.271 across the three market
        # states. Independent draws made simulated crises milder than the record.
        #
        # The market regime also selects the correlation matrix. It used to be a
        # MAJORITY VOTE across the per-ticker regimes, which sounds equivalent and
        # is not: each ticker's cut-points come from ITS OWN history, running from
        # 1963 (AVUV) to 2015 (VUSB), so a month at AVUV's p70 and one at VUSB's
        # p70 are not the same market state. That vote separated realised market
        # volatility 8.6% / 14.1% / 14.1% — medium and high were the same thing.
        # One uniform per fund in the universe, read by position (see above).
        market_regime = select_next_regime(MARKET_REGIME_KEY, market_regime)
        u = np.random.random(n_universe)
        _mi = _REGIME_INDEX[market_regime]
        _k = (_rcdf[_mi, _held, _reg] <= u[_pos][:, None]).sum(axis=1)
        _reg = _rnext[_mi, _held, _reg, _k]

        # Get volatilities for current regimes (STAGE 3: regime-switching)
        monthly_vols = _rvol[_held, _reg]

        # Generate correlated returns from the pre-factored regime matrix: one
        # standardised shock per fund in the universe, through the held rows.
        _regime_months[market_regime] += 1
        shock = crisis_shock(n_universe,
                             TAIL_DF if market_regime == 'high' else None)
        asset_returns = monthly_vols * (chol_by_regime[market_regime] @ shock)

        # Add drift (expected return)
        asset_returns += monthly_means
        # Limited liability: a fund cannot lose more than everything in a month.
        # A Gaussian month never comes near this; a Student-t crisis month
        # (MC_TAIL_DF) can sit 17 sigma out and, unfloored, drove COST and AVEE
        # NEGATIVE — every MC_TAIL_DF=4 run reported CAGR 'nan' (2026-10-14).
        np.maximum(asset_returns, -1.0, out=asset_returns)

        # Apply returns to individual ticker values
        for i, ticker in enumerate(active_tickers):
            ticker_values[ticker] *= (1 + asset_returns[i])

        # Calculate aggregate portfolio value and return
        total_portfolio = sum(ticker_values.values())
        portfolio_value[month + 1] = total_portfolio
        # Denominator for turnover. A ratio over MEAN portfolio value is the
        # only form comparable across paths and across a 58-year growth curve —
        # raw dollars traded would just say "the book got bigger".
        rebalancing_stats['value_months'] += total_portfolio
        # Modified Dietz: the denominator is the capital actually at risk this
        # month — last month's close plus the external flow — so contributions
        # and withdrawals are not mistaken for returns.
        denom = portfolio_value[month] + net_flow
        portfolio_return = (total_portfolio - denom) / denom if denom > 0 else 0
        monthly_returns.append(portfolio_return)

        # Quarterly rebalancing check (end of quarters: months 3, 6, 9, 12 of each year)
        current_month_of_year = (month + 1) % 12 or 12
        if current_month_of_year % 3 == 0 and rebalancing_strategy != 'none':
            rebalancing_stats['quarters_checked'] += 1

            # Get current regime's volatilities and correlation matrix. The
            # array step keeps the regimes in _reg; bring the dict up to date.
            for _h, _t in enumerate(active_tickers):
                current_regimes[_t] = _REGIMES[_reg[_h]]
            current_regime_vols = {
                ticker: get_volatility_for_regime(ticker, current_regimes[ticker])
                for ticker in TICKERS
            }
            current_corr = get_correlation_for_regime(market_regime)

            # Calculate current weights
            total_value = sum(ticker_values.values())
            # A depleted path has nothing to rebalance. Without this guard the
            # division yields nan weights, every band comparison against nan is
            # False, and the book silently coasts on nan — which happened to
            # produce the right answer but flooded any solve with RuntimeWarnings,
            # because the binary search deliberately probes spending levels that
            # exhaust the portfolio.
            if total_value <= 0:
                continue
            current_weights = {ticker: ticker_values[ticker] / total_value for ticker in TICKERS}

            # Q4 (December): Combined check, single FULL rebalance max
            if current_month_of_year == 12:
                rebalancing_stats['q4_sessions'] += 1

                # Combined decision (threshold + annual drift)
                decision, reason = rebalance_decision(
                    current_weights, allocation, 12,
                    current_regime_vols, current_corr, active_tickers
                )

                if decision == 'FULL':
                    _before = dict(ticker_values)
                    ticker_values, trade_desc = execute_full_rebalance(ticker_values, allocation)
                    rebalancing_stats['traded'] += traded_dollars(
                        _before, ticker_values)
                    rebalancing_stats['count'] += 1
                    rebalancing_stats['full_count'] += 1
                    rebalancing_stats['iterations_total'] += 1
                    rebalancing_stats['quarterly_counts']['Q4'] += 1
                    rebalancing_stats['triggers'][reason] = \
                        rebalancing_stats['triggers'].get(reason, 0) + 1
                    rebalancing_stats['trades'].append(trade_desc)

            # Q1-Q3: Iterative breach resolution
            else:
                # Determine quarter for tracking
                quarter_name = {3: 'Q1', 6: 'Q2', 9: 'Q3'}[current_month_of_year]

                iteration = 0
                while iteration < MAX_REBALANCE_ITERATIONS:
                    iteration += 1

                    # Recalculate current weights
                    total_value = sum(ticker_values.values())
                    if total_value <= 0:        # depleted mid-iteration
                        break
                    current_weights = {ticker: ticker_values[ticker] / total_value for ticker in TICKERS}

                    # Check for rebalancing need
                    decision, reason = rebalance_decision(
                        current_weights, allocation, current_month_of_year,
                        current_regime_vols, current_corr, active_tickers
                    )

                    if decision == 'NONE':
                        break  # No breaches, exit iteration loop

                    # Execute rebalancing
                    if decision == 'PARTIAL':
                        breached = get_breached_positions(current_weights, allocation,
                                                          active_tickers)
                        _before = dict(ticker_values)
                        ticker_values, trades = execute_partial_rebalance(ticker_values, allocation, breached)
                        rebalancing_stats['traded'] += traded_dollars(
                            _before, ticker_values)
                        rebalancing_stats['count'] += 1
                        rebalancing_stats['partial_count'] += 1
                        rebalancing_stats['iterations_total'] += 1
                        rebalancing_stats['quarterly_counts'][quarter_name] += 1
                        rebalancing_stats['q1q3_decision_reasons']['partial'] += 1
                        rebalancing_stats['triggers'][reason] = \
                            rebalancing_stats['triggers'].get(reason, 0) + 1
                        rebalancing_stats['trades'].extend(trades)
                        # Continue loop to check for more breaches

                    elif decision == 'FULL':
                        # Categorize the FULL decision reason for Q1-Q3 diagnostics
                        if 'Multiple breaches' in reason:
                            rebalancing_stats['q1q3_decision_reasons']['multiple_breaches'] += 1
                        elif 'other near bands' in reason:
                            rebalancing_stats['q1q3_decision_reasons']['near_bands'] += 1

                        _before = dict(ticker_values)
                        ticker_values, trade_desc = execute_full_rebalance(ticker_values, allocation)
                        rebalancing_stats['traded'] += traded_dollars(
                            _before, ticker_values)
                        rebalancing_stats['count'] += 1
                        rebalancing_stats['full_count'] += 1
                        rebalancing_stats['iterations_total'] += 1
                        rebalancing_stats['quarterly_counts'][quarter_name] += 1
                        rebalancing_stats['triggers'][reason] = \
                            rebalancing_stats['triggers'].get(reason, 0) + 1
                        rebalancing_stats['trades'].append(trade_desc)
                        break  # Full rebalance fixes everything, exit loop

    withdrawal_stats = {
        'spending': spending,
        'depletion_age': depletion_age,
        'w_initial': wd_state.w_initial if wd_state is not None else None,
        # THE BALANCE THE PLAN LEVEL IS A PERCENTAGE OF. Already computed —
        # PolicyState pins it at the withdrawal start age — and one float per
        # path to marshal. Without it the report can print a $190k plan level
        # and never say it is 2.6% of $7.3M, which is the ratio every piece of
        # retirement advice is stated in.
        'v_at_retirement': (wd_state.v_at_retirement
                            if wd_state is not None else None),
        # ITEM 4 — WHAT THIS PATH BELIEVED, so the report can separate not
        # knowing the future from not knowing the inputs. The three-stage
        # architecture exists to keep those apart and nothing reported the
        # split. All three are already drawn above; this only stops discarding
        # them. `path_mu` is weighted by the STARTING allocation — the weights
        # move under a glide path, and a single number has to pick one.
        'path_mu': float(np.dot(active_weights,
                                [adjusted_returns[t] for t in active_tickers])),
        'rf_shock': float(rf_shock),
        'infl_shock': float(infl_shock),
        'regime_months': _regime_months,
    }
    return (portfolio_value[-1], portfolio_value, np.array(monthly_returns),
            rebalancing_stats, withdrawal_stats)


def run_single_simulation_light(args):
    """Same simulation, returning only what a solver needs.

    The full result marshals a 601-float path, 600 monthly returns and a
    rebalancing 'trades' list holding one STRING per rebalancing event back from
    each worker. The solver reads none of it — only the final value and the
    withdrawal stats — and the pickling dominated: 500 paths cost ~0.9s of compute
    across 8 cores but 5.7s wall-clock. Dropping the unused payload is the fix.

    Shape is kept 5-wide, with None in the unused slots, so callers index
    identically to run_single_simulation.
    """
    final, _paths, _returns, _rebal, wd = run_single_simulation(args)
    return (final, None, None, None, wd)

def aggregate_rebalancing_stats(rebalancing_stats_list, time_horizon_years):
    """
    Aggregate rebalancing statistics across all simulations.

    Parameters:
        rebalancing_stats_list: List of rebalancing_stats dicts from each simulation
        time_horizon_years: Simulation time horizon in years

    Returns:
        Dictionary with aggregated statistics
    """
    n_sims = len(rebalancing_stats_list)

    # Aggregate counts
    total_events = sum(s['count'] for s in rebalancing_stats_list)
    total_partial = sum(s['partial_count'] for s in rebalancing_stats_list)
    total_full = sum(s['full_count'] for s in rebalancing_stats_list)
    total_iterations = sum(s['iterations_total'] for s in rebalancing_stats_list)
    total_quarters_checked = sum(s['quarters_checked'] for s in rebalancing_stats_list)
    total_traded = sum(s.get('traded', 0.0) for s in rebalancing_stats_list)
    total_value_months = sum(s.get('value_months', 0.0) for s in rebalancing_stats_list)
    total_q4_sessions = sum(s['q4_sessions'] for s in rebalancing_stats_list)

    # Per-simulation averages
    events_per_sim = [s['count'] for s in rebalancing_stats_list]
    events_per_year = [s['count'] / time_horizon_years for s in rebalancing_stats_list]

    # Aggregate trigger reasons
    all_triggers = {}
    for stats in rebalancing_stats_list:
        for trigger, count in stats['triggers'].items():
            all_triggers[trigger] = all_triggers.get(trigger, 0) + count

    # Sort triggers by frequency
    sorted_triggers = sorted(all_triggers.items(), key=lambda x: x[1], reverse=True)

    # Aggregate quarterly counts
    quarterly_totals = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}
    for stats in rebalancing_stats_list:
        for quarter, count in stats['quarterly_counts'].items():
            quarterly_totals[quarter] += count

    # Aggregate Q1-Q3 decision reasons
    decision_reasons = {
        'multiple_breaches': 0,
        'near_bands': 0,
        'partial': 0
    }
    for stats in rebalancing_stats_list:
        for reason, count in stats['q1q3_decision_reasons'].items():
            decision_reasons[reason] += count

    # Calculate percentages
    partial_pct = (total_partial / total_events * 100) if total_events > 0 else 0
    full_pct = (total_full / total_events * 100) if total_events > 0 else 0

    aggregated = {
        'n_sims': n_sims,
        'total_events': total_events,
        'total_partial': total_partial,
        'total_full': total_full,
        'total_iterations': total_iterations,
        'total_quarters_checked': total_quarters_checked,
        'total_q4_sessions': total_q4_sessions,
        'avg_events_per_sim': np.mean(events_per_sim),
        'median_events_per_sim': np.median(events_per_sim),
        'avg_events_per_year': np.mean(events_per_year),
        'median_events_per_year': np.median(events_per_year),
        'partial_pct': partial_pct,
        'full_pct': full_pct,
        'triggers': sorted_triggers,
        'quarterly_totals': quarterly_totals,
        'q1q3_decision_reasons': decision_reasons,
        'time_horizon_years': time_horizon_years,
        # TURNOVER, at last. Dollars traded over mean portfolio value, per year.
        # `value_months` sums the balance every month, so dividing by it and
        # multiplying by 12 gives an annual rate against the AVERAGE balance
        # rather than against the starting or ending one — the only version
        # that means anything over a 58-year horizon.
        'traded_total': total_traded,
        'turnover_per_year': ((total_traded / (total_value_months / 12.0))
                              if total_value_months > 0 else 0.0),
    }

    return aggregated

def time_weighted_cagr(monthly_returns, time_horizon_years):
    """Annualised time-weighted return per path, from the monthly return series.

    This is the honest "how did the portfolio perform" number. The obvious
    alternative -- (final_value / initial_value) ** (1/years) - 1 -- is NOT a
    return: over a long horizon the contributions dwarf the starting balance, so
    it credits deposited principal to investment performance (measured at 11.64%
    vs a true 9.53% on a 50-year path). Time-weighted also strips out the
    contribution schedule, which is identical across portfolios, making it the
    right basis for comparing allocations.

    Args:
        monthly_returns: (n_paths, n_months) array of periodic returns
        time_horizon_years: horizon used to annualise

    Returns:
        (n_paths,) array of annualised returns as decimals
    """
    growth = np.prod(1.0 + np.asarray(monthly_returns), axis=1)
    return growth ** (1.0 / time_horizon_years) - 1.0


# ============================================================================
# UNCERTAINTY IN THE ESTIMATES THEMSELVES
# ============================================================================
# Every figure this model reports is a statistic of a finite sample of paths, so
# it carries Monte Carlo error. Without a stated interval there is no way to tell
# a real difference between portfolios from sampling noise — and the differences
# here are small: five portfolios span a ~3% range in median Sortino, which
# hundreds of paths cannot resolve.

def bootstrap_ci(values, statistic=np.median, confidence=0.95,
                 n_resamples=2000, seed=12345):
    """Percentile bootstrap confidence interval for `statistic` of `values`.

    Uses a local Generator rather than the global numpy RNG: the simulation's
    determinism depends on the global stream, and resampling must not disturb it.
    """
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    stats = np.array([statistic(values[row]) for row in idx])
    tail = (1.0 - confidence) / 2.0 * 100.0
    return float(np.percentile(stats, tail)), float(np.percentile(stats, 100.0 - tail))


def paired_bootstrap_diff(a, b, statistic=np.median, confidence=0.95,
                          n_resamples=2000, seed=12345):
    """CI for statistic(a) - statistic(b) where a and b are PAIRED path-for-path.

    Portfolios are simulated under common random numbers (seed 42 + sim_id), so
    path i faces the same market in every portfolio. That makes the difference
    far better determined than either level, and the two must be resampled with
    the SAME indices to exploit it.

    Consequence worth stating plainly: overlapping marginal CIs do NOT mean two
    portfolios are indistinguishable. Judge a head-to-head on this interval —
    if it excludes zero, the difference is real even when the marginals overlap.
    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b):
        raise ValueError(f"paired bootstrap needs equal lengths, got {len(a)} and {len(b)}")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_resamples, len(a)))
    diffs = np.array([statistic(a[row]) - statistic(b[row]) for row in idx])
    tail = (1.0 - confidence) / 2.0 * 100.0
    return float(np.percentile(diffs, tail)), float(np.percentile(diffs, 100.0 - tail))


def aggregate_sortino_stats(monthly_returns, mar_annual=0.0):
    """Per-path Sortino ratios and aggregate statistics.

    Downside deviation is the *target semideviation* of Sortino & Price (1994):
    the squared shortfall below MAR averaged over EVERY period, with upside
    contributing zero — not averaged over the down periods alone. Averaging over
    the down subset inflates the denominator by sqrt(N / n_down) and understates
    the ratio; on this model's data (~43% down months) that was a factor of ~1.52,
    which left the numbers incomparable to any published Sortino or to the
    pct_sortino_above_* thresholds below.

    The bias was very nearly a constant multiplier across portfolios, so rankings
    computed under the old denominator were still valid — only the absolute
    levels were wrong.
    """
    mar_monthly = mar_annual / 12
    returns = np.asarray(monthly_returns)

    excess = returns - mar_monthly
    shortfall = np.minimum(excess, 0.0)
    downside_std = np.sqrt(np.mean(shortfall ** 2, axis=1))
    # A path that never fell below MAR has no downside; keep the guard so the
    # ratio stays finite rather than dividing by zero.
    downside_std = np.where(downside_std > 0, downside_std, 1e-10)

    sortino_ratios = excess.mean(axis=1) * 12 / (downside_std * np.sqrt(12))

    return {
        'sortino_median': np.median(sortino_ratios),
        'sortino_mean': np.mean(sortino_ratios),
        'sortino_10th': np.percentile(sortino_ratios, 10),
        'sortino_90th': np.percentile(sortino_ratios, 90),
        'pct_sortino_above_1': np.mean(sortino_ratios > 1.0) * 100,
        'pct_sortino_above_0_7': np.mean(sortino_ratios > 0.7) * 100,
        'all_sortino_ratios': sortino_ratios
    }
