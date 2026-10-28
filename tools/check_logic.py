"""
Pure-logic checks for the deterministic pieces of the engine.
=============================================================

Plain-assert unit checks (no pytest) for the functions whose behavior is exact and
worth pinning: rebalancing-band math, the derived-column calculator, the PSD repair,
correlated-return generation, Markov regime selection, and the contribution schedule.

    python3 tools/check_logic.py        # exit 0 = all pass, exit 1 = failures

Each check is a function that raises AssertionError on failure; the harness reports
every failure rather than stopping at the first.
"""

import contextlib
import inspect
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from montecarlo.loaders import (BOND_SLEEVES, compute_derived, equity_sleeve,
                     load_portfolios_from_csv, native_sleeve, with_bonds)
from montecarlo import loaders
from montecarlo import glide
from montecarlo import policies
from montecarlo import solver
from montecarlo import persistence
import run as runner
from montecarlo import factors
from montecarlo import diagnostics
from montecarlo import instrument
from montecarlo import investor
from montecarlo import report_retirement
from montecarlo import engine as pa

# Decouple band tests from the real REBALANCING_BANDS config.
pa.REBALANCING_BANDS = dict(pa.REBALANCING_BANDS)
pa.REBALANCING_BANDS['TST'] = {'relative': 0.15, 'absolute': 0.05}


# --- rebalancing band math ----------------------------------------------------------
def test_band_limits_relative_wins():
    lo, hi = pa.calculate_band_limits(0.38, 'TST')          # rel 0.057 > abs 0.05
    assert abs(lo - 0.323) < 1e-9 and abs(hi - 0.437) < 1e-9, (lo, hi)


def test_band_limits_absolute_wins():
    lo, hi = pa.calculate_band_limits(0.10, 'TST')          # rel 0.015 < abs 0.05
    assert abs(lo - 0.05) < 1e-9 and abs(hi - 0.15) < 1e-9, (lo, hi)


def test_band_breach():
    assert pa.check_band_breach(0.50, 0.40, 'TST') is True   # 0.50 > upper 0.46
    assert pa.check_band_breach(0.43, 0.40, 'TST') is False  # within band


def test_band_consumption_halfway():
    # target 0.40, width max(0.06, 0.05)=0.06; current 0.43 -> 0.03/0.06 = 0.5
    c = pa.calculate_band_consumption(0.43, 0.40, 'TST')
    assert abs(c - 0.5) < 1e-9, c


# --- compute_derived (loaders) ------------------------------------------------------
def test_derived_vt_global_split():
    d = compute_derived({'VT': 1.0})
    assert d['us_pct'] == 0.6 and d['dm_share'] == 0.75 and d['bond_pct'] == 0.0, d
    assert d['tilt_family'] == 'core' and d['n_funds'] == 1 and d['max_single_fund'] == 1.0, d


def test_derived_bond_bucket_includes_new_funds():
    d = compute_derived({'BND': 0.5, 'VUSB': 0.3, 'BSV': 0.2})
    assert abs(d['bond_pct'] - 1.0) < 1e-9, d        # VUSB + BSV must count as bonds


def test_derived_dfwix_split():
    """DFWIX's split as Dimensional reports it (dimensional.com, as of
    2026-09-30, its own country assignments, Korea and Taiwan emerging):
    64.82% international / 35.09% emerging / 0.09% US. The column read 78/22,
    the pre-2013 proxy blend, then 70/30 from fitted loadings — both wrong."""
    d = compute_derived({'DFWIX': 1.0})
    assert d['dm_share'] == 0.65 and d['us_pct'] == 0.0, d


def test_derived_tilt_family():
    assert compute_derived({'VOO': 0.8, 'JQUA': 0.2})['tilt_family'] == 'quality'
    assert compute_derived({'VOO': 0.8, 'VTV': 0.2})['tilt_family'] == 'value'
    assert compute_derived({'VOO': 0.8, 'VFMO': 0.2})['tilt_family'] == 'momentum'
    assert compute_derived({'VOO': 0.5, 'VFMO': 0.3, 'VTV': 0.2})['tilt_family'] == 'mom+val'
    assert compute_derived({'VOO': 1.0})['tilt_family'] == 'core'


# --- near_psd_higham ----------------------------------------------------------------
# Contract under test: the *near-singular* matrices the engine actually produces
# (a tiny-negative min eigenvalue, e.g. the real bond-heavy correlation matrices)
# repair to a symmetric, unit-diagonal, Cholesky-able matrix. (Higham is only an
# approximate repair, so deeply-indefinite inputs are out of scope by design.)
def _near_singular_corr():
    """A near-singular, ~unit-diagonal matrix with a tiny negative eigenvalue
    (mimics the real highly-correlated-bond case). High base correlation makes the
    smallest eigenvalue tiny, so nudging it negative barely perturbs the diagonal."""
    base = np.full((3, 3), 0.999)
    np.fill_diagonal(base, 1.0)
    vals, vecs = np.linalg.eigh(base)     # ascending; smallest ≈ 0.001
    vals[0] = -1e-9                       # push it just negative (don't reset diagonal)
    return vecs @ np.diag(vals) @ vecs.T


def test_near_psd_repairs_near_singular():
    m = _near_singular_corr()
    assert np.linalg.eigvalsh(m).min() < 0, "test precondition: input should be non-PSD"
    fixed = pa.near_psd_higham(m)
    assert np.allclose(fixed, fixed.T, atol=1e-8), "not symmetric"
    assert np.allclose(np.diag(fixed), 1.0, atol=1e-6), "diagonal != 1"
    np.linalg.cholesky(fixed)             # raises if repair didn't yield a usable matrix


# --- cholesky_factor ----------------------------------------------------------------
def test_cholesky_factor_identity_and_robustness():
    L = pa.cholesky_factor(np.eye(2))
    assert np.allclose(L, np.eye(2)), L
    # Must not raise on a near-singular input (repairs internally) — the real case
    # that arises from the five highly-correlated bond funds.
    m = _near_singular_corr()
    L2 = pa.cholesky_factor(m)
    assert L2.shape == (3, 3) and np.all(np.isfinite(L2)), L2
    assert np.allclose(np.diag(L2 @ L2.T), 1.0, atol=1e-6), L2 @ L2.T


# --- regime Markov selection --------------------------------------------------------
def test_select_next_regime_degenerate():
    # Rebinds the engine table and seeds the global stream: both put back, or
    # every test sorted after this one runs against a fake ticker 'X'.
    rng_state = np.random.get_state()
    try:
        with _engine_state():
            pa.REGIME_TRANSITIONS = dict(pa.REGIME_TRANSITIONS)
            pa.REGIME_TRANSITIONS['X'] = {
                'low': {'low': 1.0, 'medium': 0.0, 'high': 0.0},
                'medium': {'low': 0.0, 'medium': 0.0, 'high': 1.0},   # medium -> always high
                'high': {'low': 0.0, 'medium': 0.0, 'high': 1.0},
            }
            np.random.seed(1)
            assert all(pa.select_next_regime('X', 'medium') == 'high' for _ in range(20))
    finally:
        np.random.set_state(rng_state)


def test_market_regime_drives_the_correlation_matrix():
    """A pinned market chain must pin the correlation matrix — both ways.

    The point of the 2026-09-10 change is that the matrix follows a MARKET state
    rather than a majority vote of per-ticker regimes. That is only true if the
    engine actually reads the market chain, so drive it to a degenerate 'high'
    and to a degenerate 'low' and check the realised dispersion of an equally
    weighted two-asset book moves with it. A wiring bug here would silently
    restore the old behaviour and every other check would still pass.
    """
    with _engine_state():
        # Both shared shocks add COMMON variance to both books, which compresses
        # the ratio this test measures (3.9x -> 2.7x with the cash rate alone).
        # Isolate the correlation matrix, which is what is under test.
        pa.RF_REAL_STD = pa.INFLATION_STD = 0.0
        pa.TICKERS = ['A', 'B']
        pa.EXPECTED_RETURNS = {'A': 0.0, 'B': 0.0}
        pa.PARAM_UNCERTAINTY_STD = {'A': 0.0, 'B': 0.0}
        # Per-ticker regimes pinned and identical, so ONLY the matrix can differ.
        pa.VOLATILITY_REGIMES = {t: {'low': 0.16, 'medium': 0.16, 'high': 0.16}
                                 for t in ('A', 'B')}
        flat = {r: {'low': 0.0, 'medium': 1.0, 'high': 0.0}
                for r in ('low', 'medium', 'high')}
        pa.CORRELATION_LOW = np.array([[1.0, -0.9], [-0.9, 1.0]])
        pa.CORRELATION_MEDIUM = np.eye(2)
        pa.CORRELATION_HIGH = np.array([[1.0, 0.9], [0.9, 1.0]])
        pa.CORRELATION_BASELINE = pa.CORRELATION_MEDIUM
        # Five years, not thirty: with no rebalancing a rho -0.9 book drifts into
        # one asset, which pushes its realised dispersion back up toward sigma and
        # compresses the ratio below (3.9x at 5 years, 2.0x at 30).
        cfg = {'age': 40, 'retirement_age': 65, 'time_horizon_years': 5,
               'initial_value': 100_000.0, 'rebalancing_strategy': 'none',
               'contribution_schedule': {'frequency': 'annual', 'accounts': {}}}

        sd = {}
        for pin in ('low', 'high'):
            pinned = {r: {k: (1.0 if k == pin else 0.0)
                          for k in ('low', 'medium', 'high')}
                      for r in ('low', 'medium', 'high')}
            pa.REGIME_TRANSITIONS = {'A': flat, 'B': flat,
                                     pa.MARKET_REGIME_KEY: pinned}
            out = pa.run_single_simulation(({'A': 0.5, 'B': 0.5}, cfg, 0))
            sd[pin] = float(np.std(out[2]))

        # Equal weights, equal vols: at rho -0.9 the book's variance is
        # 0.05*sigma^2 and at +0.9 it is 0.95*sigma^2 — sqrt(19) = 4.4x in sd,
        # measured at 3.9x once weight drift is allowed for.
        assert sd['high'] > 3.0 * sd['low'], sd


def test_the_vectorised_regime_step_is_select_next_regime():
    """The monthly loop steps every held fund at once through _regime_arrays,
    on a docstring's word that it resolves a uniform exactly as
    select_next_regime does. The golden master pins whatever the vectorised
    path produces, not that it agrees with the reference, so a permuted
    outcome index would move regime frequencies with nothing failing. This
    checks every (market, fund, current) cell, coupled and not, on a grid of
    uniforms that includes each cdf edge, and a row with only two outcomes."""
    saved = pa.REGIME_TRANSITIONS
    table = dict(saved)
    table['SHORT'] = {'low': {'low': 0.7, 'medium': 0.3},
                      'medium': {'high': 0.4, 'low': 0.6},
                      'high': {'medium': 0.2, 'high': 0.5, 'low': 0.3}}
    tickers = ['VOO', 'COST', 'SHORT']
    coupled = {t: (t, 'low') in table for t in tickers}
    assert coupled == {'VOO': True, 'COST': True, 'SHORT': False}, coupled
    try:
        pa.REGIME_TRANSITIONS = table
        cdf, nxt = pa._regime_arrays(tickers, coupled)
        # Uniforms are in [0, 1), so every edge but the row's final 1.0.
        edges = {float(e) for e in cdf.ravel() if e < 1.0}
        grid = sorted(set(np.linspace(0, 0.999999, 97)) | edges)
        held = np.arange(len(tickers))
        for mi, m in enumerate(pa._REGIMES):
            for ci, c in enumerate(pa._REGIMES):
                reg = np.full(len(tickers), ci)
                for u in grid:
                    k = (cdf[mi, held, reg] <= u).sum(axis=1)
                    got = [pa._REGIMES[i] for i in nxt[mi, held, reg, k]]
                    want = [pa.select_next_regime((t, m) if coupled[t] else t,
                                                  c, u) for t in tickers]
                    assert got == want, (m, c, u, got, want)
    finally:
        pa.REGIME_TRANSITIONS = saved


def test_market_regime_couples_the_per_ticker_draws():
    """A ticker's transition must depend on the MARKET state, not just its own.

    Item 5b: the two were drawn independently until 2026-09-10, so the engine
    could not produce the joint tail — correlations spiking WHILE most assets are
    volatile. Here the market is pinned to 'high' and the ticker is given a
    conditional chain that goes high only under a high market. If the engine
    ignored the (ticker, market) key it would use the unconditional row, which
    never leaves 'low', and the realised dispersion would not move.
    """
    with _engine_state():
        pa.TICKERS = ['A']
        pa.EXPECTED_RETURNS = {'A': 0.0}
        pa.PARAM_UNCERTAINTY_STD = {'A': 0.0}
        pa.VOLATILITY_REGIMES = {'A': {'low': 0.02, 'medium': 0.10, 'high': 0.60}}
        for m in ('LOW', 'MEDIUM', 'HIGH', 'BASELINE'):
            setattr(pa, f'CORRELATION_{m}', np.array([[1.0]]))
        stay_low = {r: {'low': 1.0, 'medium': 0.0, 'high': 0.0}
                    for r in ('low', 'medium', 'high')}
        go_high = {r: {'low': 0.0, 'medium': 0.0, 'high': 1.0}
                   for r in ('low', 'medium', 'high')}
        pin_high = {r: {'low': 0.0, 'medium': 0.0, 'high': 1.0}
                    for r in ('low', 'medium', 'high')}
        cfg = {'age': 40, 'retirement_age': 65, 'time_horizon_years': 5,
               'initial_value': 100_000.0, 'rebalancing_strategy': 'none',
               'contribution_schedule': {'frequency': 'annual', 'accounts': {}}}

        # Unconditional row says stay low; the conditional row under a HIGH
        # market says go high. The market chain is pinned to high.
        pa.REGIME_TRANSITIONS = {
            'A': stay_low, pa.MARKET_REGIME_KEY: pin_high,
            ('A', 'low'): stay_low, ('A', 'medium'): stay_low,
            ('A', 'high'): go_high,
        }
        coupled = float(np.std(pa.run_single_simulation(({'A': 1.0}, cfg, 0))[2]))

        # Same everything, minus the conditional keys: the engine must fall back
        # to the unconditional row and stay in the 2% regime.
        pa.REGIME_TRANSITIONS = {'A': stay_low, pa.MARKET_REGIME_KEY: pin_high}
        uncoupled = float(np.std(pa.run_single_simulation(({'A': 1.0}, cfg, 0))[2]))

        assert coupled > 5.0 * uncoupled, (uncoupled, coupled)


def test_volatility_for_regime():
    with _engine_state():
        pa.VOLATILITY_REGIMES = dict(pa.VOLATILITY_REGIMES)
        pa.VOLATILITY_REGIMES['X'] = {'low': 0.07, 'medium': 0.14, 'high': 0.22}
        assert pa.get_volatility_for_regime('X', 'high') == 0.22


# --- contribution schedule ----------------------------------------------------------
def test_contribution_basic_and_growth():
    sched = {'accounts': {'a': {'annual_amount': 10000.0, 'growth_rate': 0.0, 'stop_age': None}}}
    assert pa.calculate_annual_contribution(36, sched) == 10000.0     # years_elapsed 0
    assert pa.calculate_annual_contribution(40, sched) == 10000.0     # growth 0
    sched_g = {'accounts': {'a': {'annual_amount': 10000.0, 'growth_rate': 0.05, 'stop_age': None}}}
    assert abs(pa.calculate_annual_contribution(38, sched_g) - 10000.0 * 1.05 ** 2) < 1e-6


def test_contribution_stop_age():
    sched = {'accounts': {'a': {'annual_amount': 10000.0, 'growth_rate': 0.0, 'stop_age': 39}}}
    assert pa.calculate_annual_contribution(40, sched) == 0.0          # past stop_age


# --- flow accounting (C1) -----------------------------------------------------------
# A zero-volatility, single-asset universe is an exact oracle: every month's true
# return is mu/12 regardless of how much money is paid in. Any dependence of the
# reported return on the contribution size is the C1 bug.
@contextlib.contextmanager
def _engine_state():
    """Every engine global put back on exit, whatever the test reassigned.

    Snapshots the whole module namespace rather than a list of names: the
    hand-written lists drifted from what the fixtures set (one saved 10
    globals and set 12, and BASE_VOLATILITY was set and never restored), and
    a leaked global makes every later test order-dependent. Rebinding only —
    a dict mutated in place is the test's own to undo."""
    saved = dict(vars(pa))
    try:
        yield
    finally:
        for name in [n for n in vars(pa) if n not in saved]:
            delattr(pa, name)
        for name, value in saved.items():
            setattr(pa, name, value)


def _zero_vol_engine(monthly_contrib_annual):
    """Install a deterministic 1-asset universe; return (config, allocation).

    DETERMINISTIC means the shared cash-rate shock is off too. It randomises mu
    itself — which is the whole point of it — so leaving it on turns every
    exact-arithmetic assertion built on this fixture into a one-sample noise
    test. Four of them started failing by ~0.0004/month the moment it landed.
    """
    pa.RF_REAL_STD = 0.0
    pa.INFLATION_STD = 0.0
    pa.TICKERS = ['Z']
    pa.EXPECTED_RETURNS = {'Z': 0.12}
    pa.PARAM_UNCERTAINTY_STD = {'Z': 0.0}
    pa.VOLATILITY_REGIMES = {'Z': {'low': 0.0, 'medium': 0.0, 'high': 0.0}}
    stay = {r: {'low': 0.0, 'medium': 1.0, 'high': 0.0}
            for r in ('low', 'medium', 'high')}
    pa.REGIME_TRANSITIONS = {'Z': stay, pa.MARKET_REGIME_KEY: stay}
    one = np.array([[1.0]])
    pa.CORRELATION_BASELINE = pa.CORRELATION_LOW = one
    pa.CORRELATION_MEDIUM = pa.CORRELATION_HIGH = one
    cfg = {
        'age': 40, 'retirement_age': 65, 'time_horizon_years': 3,
        'initial_value': 100_000.0, 'rebalancing_strategy': 'none',
        'contribution_schedule': {'frequency': 'annual', 'accounts': {
            'main': {'annual_amount': monthly_contrib_annual,
                     'growth_rate': 0.0, 'stop_age': None}}},
    }
    return cfg, {'Z': 1.0}


def test_reported_return_excludes_contributions():
    with _engine_state():
        expected = 0.12 / 12                      # true monthly return, vol = 0
        for annual_contrib in (0.0, 60_000.0):    # 60k/yr on 100k is a huge flow
            cfg, alloc = _zero_vol_engine(annual_contrib)
            returns = pa.run_single_simulation((alloc, cfg, 0))[2]
            worst = max(abs(r - expected) for r in returns)
            assert worst < 1e-12, (
                f"contrib={annual_contrib:,.0f}: reported monthly return deviates "
                f"from the true {expected:.6f} by up to {worst:.6f}")


def test_no_fund_can_lose_more_than_everything_in_a_month():
    """A review of the model found it: under MC_TAIL_DF=4 every run reported CAGR 'nan'.
    Returns are applied arithmetically, and a Student-t crisis month — one
    chi-square draw shared by every fund, rescaled to unit variance — can sit
    17 sigma out, past -100% for a high-volatility fund (COST, AVEE). The fund's
    value went NEGATIVE and the run carried on. Limited liability: a fund's
    monthly return is floored at -100%. A Gaussian month never reaches it."""
    with _engine_state():
        cfg, alloc = _zero_vol_engine(0.0)
        pa.VOLATILITY_REGIMES = {'Z': {'low': 0.2, 'medium': 0.2, 'high': 0.2}}
        pa.BASE_VOLATILITY = {'Z': 0.2}
        pa.crisis_shock = lambda n, tail_df=None: np.full(n, -50.0)
        out = pa.run_single_simulation((alloc, cfg, 0))
        returns = np.asarray(out[2], dtype=float)
        assert np.isfinite(returns).all(), returns[:5]
        # exactly -100%: a floor set higher clips real crash months, and a
        # floor below it lets the fund go negative
        assert abs(returns.min() + 1.0) < 1e-12, returns.min()
        cagr = pa.time_weighted_cagr(returns[None, :], cfg['time_horizon_years'])
        assert np.isfinite(cagr).all(), cagr


# --- time-weighted return (C2) ------------------------------------------------------
def test_time_weighted_cagr_matches_closed_form():
    """A constant 1%/month series must annualise to exactly 1.01**12 - 1."""
    r = np.full((3, 24), 0.01)                       # 3 paths, 2 years
    cagr = pa.time_weighted_cagr(r, 2)
    expected = 1.01 ** 12 - 1
    assert cagr.shape == (3,)
    assert np.allclose(cagr, expected, rtol=0, atol=1e-12), (cagr, expected)


def test_time_weighted_cagr_ignores_contributions():
    """TWR from the corrected return series must not depend on contribution size."""
    with _engine_state():
        out = []
        for annual_contrib in (0.0, 60_000.0):
            cfg, alloc = _zero_vol_engine(annual_contrib)
            r = np.array([pa.run_single_simulation((alloc, cfg, 0))[2]])
            out.append(pa.time_weighted_cagr(r, cfg['time_horizon_years'])[0])
        assert abs(out[0] - out[1]) < 1e-12, out
        assert abs(out[0] - (1.01 ** 12 - 1)) < 1e-12, out


# --- real-terms conversion (A1) -----------------------------------------------------
def test_real_rate_is_exact_not_subtraction():
    """With deterministic inflation, real = (1+nominal)/(1+i) - 1 exactly.

    Naive subtraction (nominal - i) is close but wrong by ~0.19pp on VOO's 9.47%
    at 3% inflation — material when compounded over 50 years.
    """
    r = pa.real_rate(0.09465522294902093, 0.03)
    assert abs(r - (1.09465522294902093 / 1.03 - 1)) < 1e-15
    naive = 0.09465522294902093 - 0.03
    assert abs(r - naive) > 1.5e-3, "exact and naive should differ materially"
    assert abs(pa.real_rate(0.05, 0.0) - 0.05) < 1e-15   # no inflation is identity
                                                        # (not ==: 1.05-1.0 != 0.05 in binary FP)
    assert abs(pa.real_rate(0.03, 0.03)) < 1e-15     # return == inflation -> 0 real


def test_dispersions_scale_by_one_plus_inflation():
    """real = (1+nom)/(1+i) - 1 is affine in nom, so sigma_real = sigma_nom/(1+i)."""
    assert abs(pa.real_dispersion(0.1448, 0.03) - 0.1448 / 1.03) < 1e-15
    assert pa.real_dispersion(0.1448, 0.0) == 0.1448


def test_contribution_growth_deflates():
    """A contribution growing at exactly the inflation rate is flat in real terms."""
    sched = {'accounts': {'a': {'annual_amount': 10000.0, 'growth_rate': 0.03,
                                'stop_age': None}}}
    assert abs(pa.calculate_annual_contribution(46, sched, inflation=0.03) - 10000.0) < 1e-9
    # and unchanged when no inflation is applied (default preserves old behaviour)
    assert abs(pa.calculate_annual_contribution(46, sched)
               - 10000.0 * 1.03 ** 10) < 1e-9


def test_production_parameters_are_real():
    """Loaded parameters must be deflated by exactly engine.INFLATION.

    Pinned here rather than in the golden master so that retuning the inflation
    assumption fails a named check instead of an opaque wealth-statistics drift.
    """
    assert pa.INFLATION == 0.03, f"inflation assumption changed to {pa.INFLATION}"
    for t in pa.TICKERS:
        nominal = pa.NOMINAL_EXPECTED_RETURNS[t]
        expected = pa.real_rate(nominal, pa.INFLATION)
        # Premium scaling (if any) is applied after deflation, so compare the
        # market proxy exactly and every other ticker only for the direction.
        if t == pa.MARKET_PROXY or pa.PREMIUM_SCALE == 1.0:
            assert abs(pa.EXPECTED_RETURNS[t] - expected) < 1e-15, t
        assert pa.EXPECTED_RETURNS[t] < nominal, t

    # Deliberately NOT pinned to a snapshot value. An earlier version asserted
    # VOO == 0.06277206111555422, which broke on the first quarterly regen (VOO
    # moved 9.47% -> 9.52% nominal) even though nothing was wrong. A test that
    # cries wolf on a routine data refresh trains you to ignore the harness.
    voo_real = pa.EXPECTED_RETURNS[pa.MARKET_PROXY]
    assert 0.02 < voo_real < 0.10, f"market real return implausible: {voo_real:.2%}"


# --- glide path parsing / resolution -----------------------------------------------
def test_parse_bare_id_is_single_waypoint():
    sch = glide.parse_schedule('141', 37)
    assert sch == (glide.Waypoint('141', 37),), sch


def test_parse_waypoints():
    sch = glide.parse_schedule('141>340@65>540@75', 37)
    assert sch == (glide.Waypoint('141', 37), glide.Waypoint('340', 65),
                   glide.Waypoint('540', 75)), sch


def test_parse_tolerates_whitespace_and_explicit_start_age():
    assert glide.parse_schedule('  141 > 340 @ 65 ', 37) == glide.parse_schedule('141>340@65', 37)
    assert glide.parse_schedule('141@40>340@65', 37)[0] == glide.Waypoint('141', 40)


def test_parse_rejects_bad_specs():
    for bad, why in [
        ('141>340', 'non-first waypoint without an age'),
        ('141>340@65>540@60', 'ages must increase'),
        ('141>340@65>540@65', 'ages must strictly increase'),
        ('', 'empty spec'),
        ('141>@65', 'missing portfolio id'),
        ('141>340@notanage', 'non-numeric age'),
    ]:
        try:
            glide.parse_schedule(bad, 37)
        except ValueError:
            continue
        raise AssertionError(f"should have rejected {bad!r} ({why})")


def test_portfolio_at_steps_on_the_switch_age():
    sch = glide.parse_schedule('141>340@65>540@75', 37)
    assert glide.portfolio_at(sch, 37) == '141'
    assert glide.portfolio_at(sch, 64.99) == '141'
    assert glide.portfolio_at(sch, 65.0) == '340'      # inclusive at the switch
    assert glide.portfolio_at(sch, 74.99) == '340'
    assert glide.portfolio_at(sch, 75.0) == '540'
    assert glide.portfolio_at(sch, 120) == '540'


def test_portfolio_at_before_start_holds_first():
    sch = glide.parse_schedule('141@40>340@65', 37)
    assert glide.portfolio_at(sch, 20) == '141'


def test_format_spec_round_trips():
    for spec in ['141', '141>340@65', '141>340@65>540@75']:
        sch = glide.parse_schedule(spec, 37)
        assert glide.format_spec(sch) == spec, (glide.format_spec(sch), spec)


def test_derisk_variants_builds_an_increasing_bond_ladder():
    """Control first, then one rung per tier in increasing bond weight, all
    synthesised from the base's own equity sleeve."""
    P = _P()
    specs = glide.derisk_variants('141', 65)
    assert specs[0] == '141', specs
    assert len(specs) == 1 + len(glide.DERISK_TIERS), specs
    for sp in specs[1:]:
        assert sp.startswith('141>141+') and sp.endswith('@65'), sp
    bonds = [runner.target_bond_pct(sp, P) for sp in specs]
    assert bonds == sorted(bonds) and len(set(bonds)) == len(bonds), bonds
    assert bonds[0] == 0 and all(b > 0 for b in bonds[1:]), bonds


def test_single_waypoint_plan_matches_static_dict():
    """A one-waypoint glide plan must be bit-identical to passing a plain dict."""
    P = _P()
    cfg = _glide_cfg()
    alloc = P['141']['allocation']
    plan = glide.build_plan(glide.parse_schedule('141', 37), P)
    a = pa.run_single_simulation((alloc, cfg, 3))
    b = pa.run_single_simulation((plan, cfg, 3))
    assert a[0] == b[0] and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])


def _glide_cfg(years=50):
    return {'age': 37, 'retirement_age': 65, 'time_horizon_years': years,
            'initial_value': 300000, 'rebalancing_strategy': 'quarterly_annual',
            'contribution_schedule': {'frequency': 'annual', 'accounts': {
                'main': {'annual_amount': 12000, 'growth_rate': 0.0, 'stop_age': 60}}}}


def test_glide_switch_changes_the_book():
    """After the switch age the portfolio must hold the new target's bond weight."""
    P = _P()
    plan = glide.build_plan(glide.parse_schedule('141>141+20@65', 37), P)
    # 141 holds no bonds; the synthesised 141+20 rung holds 20%.
    assert compute_derived(P['141']['allocation'])['bond_pct'] == 0
    assert compute_derived(plan[-1][1])['bond_pct'] > 0.15

    # Horizon ending just before the switch: still no bonds.
    before = pa.run_single_simulation((plan, _glide_cfg(27), 3))     # to age 64
    # Horizon running past it: bonds present.
    after = pa.run_single_simulation((plan, _glide_cfg(30), 3))      # to age 67

    assert before[0] > 0 and after[0] > 0
    # The switch must have registered as a rebalancing event exactly once.
    assert before[3]['triggers'].get('glide_switch', 0) == 0, before[3]['triggers']
    assert after[3]['triggers'].get('glide_switch', 0) == 1, after[3]['triggers']
    # and the book actually changed: the same paths without the switch end elsewhere
    control = pa.run_single_simulation((glide.build_plan(
        glide.parse_schedule('141', 37), P), _glide_cfg(30), 3))
    assert after[0] != control[0], "post-switch outcome must differ from never switching"


# --- Sortino denominator ------------------------------------------------------------
def test_sortino_uses_target_semideviation():
    """Downside deviation averages the squared shortfall over ALL periods.

    Sortino & Price (1994) define the denominator as the target semideviation:
    sqrt(mean(min(r - MAR, 0)**2)) over every period, upside contributing zero.
    Averaging over only the down months instead inflates the denominator by
    sqrt(N / n_down) and understates the ratio (~1.5x on this model's data).

    Worked by hand: r = [.02, -.01, .03, -.02], MAR = 0
      mean            = 0.005
      shortfall       = [0, -.01, 0, -.02]
      mean(sq)        = (0 + 1e-4 + 0 + 4e-4)/4 = 1.25e-4
      dd              = 0.0111803398875
      sortino         = 0.005*12 / (dd*sqrt(12)) = 1.5491933384830
    """
    r = np.array([[0.02, -0.01, 0.03, -0.02]])
    got = pa.aggregate_sortino_stats(r, mar_annual=0.0)['sortino_median']
    assert abs(got - 1.5491933384829668) < 1e-12, got
    # the buggy denominator (down months only) would give sqrt(2)x less
    assert abs(got / 1.0954451150103321 - 2 ** 0.5) < 1e-12


def test_sortino_handles_no_downside():
    """An all-positive series has zero downside deviation; must not divide by zero."""
    r = np.array([[0.01, 0.02, 0.03]])
    got = pa.aggregate_sortino_stats(r, mar_annual=0.0)['sortino_median']
    assert np.isfinite(got) and got > 0, got


# --- bootstrap confidence intervals -------------------------------------------------
def test_bootstrap_ci_brackets_the_statistic():
    rng = np.random.default_rng(0)
    x = rng.normal(100.0, 15.0, 4000)
    lo, hi = pa.bootstrap_ci(x, np.median, seed=1)
    assert lo < np.median(x) < hi, (lo, np.median(x), hi)
    assert lo < 100.0 < hi, (lo, hi)          # contains the true median


def test_bootstrap_ci_is_deterministic_and_leaves_global_rng_alone():
    x = np.arange(500.0)
    assert pa.bootstrap_ci(x, np.median, seed=7) == pa.bootstrap_ci(x, np.median, seed=7)
    np.random.seed(123)
    before = np.random.random()
    np.random.seed(123)
    pa.bootstrap_ci(x, np.median, seed=7)
    assert np.random.random() == before, "bootstrap must not disturb the global RNG"


def test_bootstrap_ci_narrows_with_sample_size():
    rng = np.random.default_rng(2)
    wide = pa.bootstrap_ci(rng.normal(0, 1, 250), np.median, seed=3)
    tight = pa.bootstrap_ci(rng.normal(0, 1, 4000), np.median, seed=3)
    assert (tight[1] - tight[0]) < (wide[1] - wide[0]) / 2, (wide, tight)


def test_paired_bootstrap_is_tighter_than_independent_cis():
    """Common random numbers make the DIFFERENCE far better determined than
    either level, so overlapping marginal CIs do not mean 'indistinguishable'."""
    rng = np.random.default_rng(5)
    shared = rng.normal(0, 1.0, 3000)         # the shared market history
    a = shared + rng.normal(0, 0.02, 3000)
    b = shared + rng.normal(0, 0.02, 3000) + 0.05    # a real 0.05 edge
    ci_a = pa.bootstrap_ci(a, np.mean, seed=1)
    ci_b = pa.bootstrap_ci(b, np.mean, seed=1)
    overlap = min(ci_a[1], ci_b[1]) - max(ci_a[0], ci_b[0])
    assert overlap > 0, "marginal CIs should overlap here"
    lo, hi = pa.paired_bootstrap_diff(b, a, np.mean, seed=1)
    assert lo > 0, (lo, hi)                   # paired test resolves it anyway
    assert (hi - lo) < (ci_a[1] - ci_a[0]) / 5


def test_paired_bootstrap_on_identical_samples_contains_zero():
    x = np.random.default_rng(9).normal(0, 1, 800)
    lo, hi = pa.paired_bootstrap_diff(x, x, np.median, seed=4)
    assert lo <= 0 <= hi and abs(hi - lo) < 1e-9, (lo, hi)


# --- withdrawal sourcing ------------------------------------------------------------
# Worked example from the v3.1 review notes (unpublished): $3M book, 90/10 target, drifted to
# 86.3/13.7, withdrawing $10k. Bonds are the only holding above their
# post-withdrawal target, so `excess` takes the whole $10k from bonds while
# `pro_rata` splits it and leaves the drift untouched.
_SRC_TARGET = {'EQ': 0.90, 'BD': 0.10}
_SRC_VALUES = {'EQ': 2_589_000.0, 'BD': 411_000.0}
_SRC_ACTIVE = ['EQ', 'BD']


def _drift(values, target):
    total = sum(values.values())
    return sum(abs(values[t] / total - target[t]) for t in values)


def test_sourcing_rules_sum_to_the_withdrawal():
    for fn in (policies.source_pro_rata, policies.source_excess):
        for w in (1.0, 10_000.0, 250_000.0):
            sold = fn(_SRC_VALUES, _SRC_TARGET, w, _SRC_ACTIVE)
            assert abs(sum(sold.values()) - w) < 1e-6, (fn.__name__, w, sold)


def test_pro_rata_leaves_weights_unchanged():
    sold = policies.source_pro_rata(_SRC_VALUES, _SRC_TARGET, 10_000.0, _SRC_ACTIVE)
    after = {t: _SRC_VALUES[t] - sold[t] for t in _SRC_ACTIVE}
    assert abs(_drift(after, _SRC_TARGET) - _drift(_SRC_VALUES, _SRC_TARGET)) < 1e-12


def test_excess_takes_only_from_the_overweight_holding():
    sold = policies.source_excess(_SRC_VALUES, _SRC_TARGET, 10_000.0, _SRC_ACTIVE)
    assert abs(sold['EQ']) < 1e-9, sold      # equity is below its post-withdrawal target
    assert abs(sold['BD'] - 10_000.0) < 1e-6, sold


def test_excess_never_increases_drift():
    for w in (1_000.0, 10_000.0, 200_000.0, 900_000.0):
        sold = policies.source_excess(_SRC_VALUES, _SRC_TARGET, w, _SRC_ACTIVE)
        after = {t: _SRC_VALUES[t] - sold[t] for t in _SRC_ACTIVE}
        assert _drift(after, _SRC_TARGET) <= _drift(_SRC_VALUES, _SRC_TARGET) + 1e-12, w


def test_sourcing_rules_agree_when_at_target():
    at_target = {'EQ': 2_700_000.0, 'BD': 300_000.0}
    a = policies.source_pro_rata(at_target, _SRC_TARGET, 10_000.0, _SRC_ACTIVE)
    b = policies.source_excess(at_target, _SRC_TARGET, 10_000.0, _SRC_ACTIVE)
    for t in _SRC_ACTIVE:
        assert abs(a[t] - b[t]) < 1e-6, (t, a, b)


def test_sourcing_never_sells_more_than_a_holding_has():
    for fn in (policies.source_pro_rata, policies.source_excess):
        sold = fn(_SRC_VALUES, _SRC_TARGET, 2_900_000.0, _SRC_ACTIVE)   # near-total
        for t in _SRC_ACTIVE:
            assert -1e-9 <= sold[t] <= _SRC_VALUES[t] + 1e-6, (fn.__name__, t, sold)


# --- withdrawal policies ------------------------------------------------------------
def _state(w_initial=100_000.0, v_at_retirement=2_500_000.0, retirement_age=65,
           end_age=87, w_prior=None):
    return policies.PolicyState(
        w_initial=w_initial, v_at_retirement=v_at_retirement,
        initial_rate=w_initial / v_at_retirement, w_prior_year=w_prior or w_initial,
        retirement_age=retirement_age, end_age=end_age, assumed_real_return=0.035,
        # the guardrail arithmetic below is written against the original rule
        **{f'gk_{k}': v for k, v in policies.gk_guardrails('slow').items()})


def test_constant_real_ignores_the_balance():
    st = _state()
    assert policies.constant_real(70, 5_000_000.0, st) == 100_000.0
    assert policies.constant_real(70, 500_000.0, st) == 100_000.0


def test_fixed_pct_scales_with_the_balance():
    st = _state()                       # initial rate 4%
    assert abs(policies.fixed_pct(70, 2_500_000.0, st) - 100_000.0) < 1e-9
    assert abs(policies.fixed_pct(70, 1_250_000.0, st) - 50_000.0) < 1e-9


def test_guyton_klinger_cuts_when_the_upper_guardrail_breaks():
    st = _state()                       # initial rate 4%, guardrail at 4.8%
    # balance down to 1.5M -> implied rate 6.7% -> cut 10%
    assert abs(policies.guyton_klinger(70, 1_500_000.0, st) - 90_000.0) < 1e-9


def test_guyton_klinger_raises_when_the_lower_guardrail_breaks():
    st = _state()                       # lower guardrail at 3.2%
    # balance up to 5M -> implied rate 2% -> raise 10%
    assert abs(policies.guyton_klinger(70, 5_000_000.0, st) - 110_000.0) < 1e-9


def test_guyton_klinger_holds_inside_the_guardrails():
    st = _state()
    assert abs(policies.guyton_klinger(70, 2_400_000.0, st) - 100_000.0) < 1e-9


def test_vpw_rate_rises_with_age():
    st = _state()
    rates = [policies.vpw(a, 2_500_000.0, st) / 2_500_000.0 for a in (65, 75, 85)]
    assert rates[0] < rates[1] < rates[2], rates
    # Must track the published VPW table, which amortises over life expectancy
    # plus a buffer — not to the simulation endpoint, which would send the rate
    # toward 100% as the horizon closes.
    assert abs(rates[0] - 0.050) < 0.008, rates   # ~5.0% at 65
    assert abs(rates[2] - 0.087) < 0.015, rates   # ~8.7% at 85
    assert rates[2] < 0.15, "rate must stay sane near the endpoint"


def test_every_policy_is_registered_and_returns_a_number():
    assert set(policies.POLICIES) == {
        'constant_real', 'fixed_pct', 'guyton_klinger', 'vpw'}, set(policies.POLICIES)
    assert set(policies.SOURCING) == {'pro_rata', 'excess'}, set(policies.SOURCING)
    st = _state()
    for name, fn in policies.POLICIES.items():
        v = fn(70, 2_000_000.0, st)
        assert isinstance(v, float) and v >= 0, (name, v)


# --- withdrawals in the engine ------------------------------------------------------
def _wd_cfg(years=30, withdrawal=None, initial=1_000_000.0, contrib=0.0):
    cfg = {'age': 60, 'retirement_age': 65, 'time_horizon_years': years,
           'initial_value': initial, 'rebalancing_strategy': 'none',
           'contribution_schedule': {'frequency': 'annual', 'accounts': {
               'main': {'annual_amount': contrib, 'growth_rate': 0.0, 'stop_age': None}}}}
    if withdrawal is not None:
        cfg['withdrawal'] = withdrawal
    return cfg


def test_no_withdrawal_config_is_unchanged():
    """Omitting the withdrawal block must reproduce accumulation exactly."""
    with _engine_state():
        cfg, alloc = _zero_vol_engine(0.0)
        a = pa.run_single_simulation((alloc, cfg, 5))
        b = pa.run_single_simulation((alloc, dict(cfg), 5))
        assert a[0] == b[0] and np.array_equal(a[1], b[1])
        assert a[4]['depletion_age'] is None and a[4]['spending'] == []


def test_constant_real_spends_exactly_the_target():
    """Zero vol, ample balance: every retirement year spends exactly the target."""
    with _engine_state():
        _zero_vol_engine(0.0)          # installs the deterministic universe
        wd = {'start_age': 65, 'policy': 'constant_real', 'sourcing': 'pro_rata',
              'target_real_annual': 40_000.0}
        r = pa.run_single_simulation(({'Z': 1.0}, _wd_cfg(30, wd), 0))
        spend = r[4]['spending']
        assert len(spend) == 25, len(spend)                 # ages 65..89
        assert all(abs(v - 40_000.0) < 1e-6 for v in spend), spend[:3]
        assert r[4]['depletion_age'] is None


def test_withdrawals_reduce_the_balance_versus_none():
    with _engine_state():
        _zero_vol_engine(0.0)
        wd = {'start_age': 65, 'policy': 'constant_real', 'sourcing': 'pro_rata',
              'target_real_annual': 40_000.0}
        with_wd = pa.run_single_simulation(({'Z': 1.0}, _wd_cfg(30, wd), 0))[0]
        without = pa.run_single_simulation(({'Z': 1.0}, _wd_cfg(30, None), 0))[0]
        assert with_wd < without, (with_wd, without)


def test_depletion_is_detected_and_recorded():
    """An unsustainable draw must deplete, record the age, and stop withdrawing."""
    with _engine_state():
        _zero_vol_engine(0.0)
        wd = {'start_age': 65, 'policy': 'constant_real', 'sourcing': 'pro_rata',
              'target_real_annual': 300_000.0}
        r = pa.run_single_simulation(({'Z': 1.0}, _wd_cfg(30, wd, initial=500_000.0), 0))
        assert r[4]['depletion_age'] is not None, r[4]
        assert 65 < r[4]['depletion_age'] < 90, r[4]['depletion_age']
        assert r[0] >= 0.0, "value must never go negative"
        assert min(r[1]) >= 0.0


def test_reported_return_excludes_withdrawal_flows():
    """The Modified Dietz denominator must net out withdrawals too."""
    with _engine_state():
        _zero_vol_engine(0.0)
        wd = {'start_age': 65, 'policy': 'constant_real', 'sourcing': 'pro_rata',
              'target_real_annual': 40_000.0}
        r = pa.run_single_simulation(({'Z': 1.0}, _wd_cfg(30, wd), 0))
        expected = 0.12 / 12
        worst = max(abs(x - expected) for x in r[2])
        assert worst < 1e-12, worst


# --- inverse solver -----------------------------------------------------------------
def _solver_cfg(spend=40_000.0, years=30):
    """Deterministic universe, no contributions, withdrawals from day one."""
    return {'age': 65, 'retirement_age': 65, 'time_horizon_years': years,
            'initial_value': 1_000_000.0, 'rebalancing_strategy': 'none',
            'contribution_schedule': {'frequency': 'annual', 'accounts': {
                'main': {'annual_amount': 0.0, 'growth_rate': 0.0, 'stop_age': None}}},
            'withdrawal': {'start_age': 65, 'policy': 'constant_real',
                           'sourcing': 'pro_rata', 'target_real_annual': spend,
                           'floor': 0.85}}


def test_success_rate_falls_as_spending_rises():
    with _engine_state():
        _zero_vol_engine(0.0)
        rates = [solver.evaluate({'Z': 1.0}, _solver_cfg(), s, n_sims=8,
                                 parallel=False)['success_rate']
                 for s in (40_000.0, 120_000.0, 400_000.0)]
        assert rates[0] >= rates[1] >= rates[2], rates
        assert rates[0] == 1.0 and rates[2] == 0.0, rates


def test_solver_matches_the_annuity_on_a_deterministic_universe():
    """Zero vol makes the answer a closed-form annuity, so the solver is checkable.

    $1M at 12% real for 30 years, drawn monthly in advance. The monthly annuity-due
    payment is V * i / (1 - (1+i)^-n) / (1+i) with i = 0.01, n = 360.
    """
    with _engine_state():
        _zero_vol_engine(0.0)
        i, n, V = 0.12 / 12, 360, 1_000_000.0
        annuity_due = V * i / (1 - (1 + i) ** -n) / (1 + i) * 12
        got = solver.solve_sustainable_spend(
            {'Z': 1.0}, _solver_cfg(), confidence=0.90, search_sims=4,
            verify_sims=4, n_points=4, parallel=False)['spend']
        assert abs(got - annuity_due) / annuity_due < 0.05, (got, annuity_due)


def test_solver_is_deterministic():
    with _engine_state():
        _zero_vol_engine(0.0)
        kw = dict(confidence=0.90, search_sims=4, verify_sims=4, n_points=4,
                  parallel=False)
        a = solver.solve_sustainable_spend({'Z': 1.0}, _solver_cfg(), **kw)['spend']
        b = solver.solve_sustainable_spend({'Z': 1.0}, _solver_cfg(), **kw)['spend']
        assert a == b, (a, b)


def test_evaluate_overrides_the_configured_target():
    """The solver must drive spend itself, not inherit whatever config carried."""
    with _engine_state():
        _zero_vol_engine(0.0)
        cfg = _solver_cfg(spend=999_999.0)
        r = solver.evaluate({'Z': 1.0}, cfg, 30_000.0, n_sims=4, parallel=False)
        assert r['spend'] == 30_000.0
        assert cfg['withdrawal']['target_real_annual'] == 999_999.0, "must not mutate caller's config"


def test_solver_gives_every_policy_a_workable_answer():
    """Regression: the original every-year floor drove fixed_pct to $0.

    fixed_pct spends a fixed fraction of the balance, so "every year above
    0.85 x target" was really "the portfolio never drops 15% below its age-65
    value" — a drawdown test almost no path survives. Each policy must now solve
    to a positive, finite spend.
    """
    with _engine_state():
        _zero_vol_engine(0.0)
        for policy in policies.POLICIES:
            cfg = _solver_cfg()
            cfg['withdrawal']['policy'] = policy
            r = solver.solve_sustainable_spend(
                {'Z': 1.0}, cfg, confidence=0.90, search_sims=4, verify_sims=4,
                n_points=4, parallel=False)
            assert r['spend'] > 0, f"{policy} solved to {r['spend']}"
            assert np.isfinite(r['worst_cut']), f"{policy} worst_cut is {r['worst_cut']}"


# --- equity premium scaling (A3) ----------------------------------------------------
def test_premium_scale_one_is_identity():
    mus = {'VOO': 0.06, 'AVUV': 0.12, 'BND': 0.02}
    assert pa.scale_equity_premia(mus, 1.0) == mus


def test_premium_scale_zero_flattens_equities_to_the_market():
    mus = {'VOO': 0.06, 'AVUV': 0.12, 'VFMO': 0.14, 'BND': 0.02}
    out = pa.scale_equity_premia(mus, 0.0)
    assert abs(out['AVUV'] - 0.06) < 1e-15 and abs(out['VFMO'] - 0.06) < 1e-15, out
    assert abs(out['VOO'] - 0.06) < 1e-15


def test_premium_scale_leaves_bonds_alone():
    """Bond premia over VOO are large and NEGATIVE, so scaling them would drag
    bonds up toward the equity return and invert the comparison being tested."""
    mus = {'VOO': 0.06, 'AVUV': 0.12, 'BND': 0.02, 'JPST': -0.01}
    for k in (0.0, 0.5, 1.0):
        out = pa.scale_equity_premia(mus, k)
        assert out['BND'] == 0.02, (k, out)
        assert out['JPST'] == -0.01, (k, out)


def test_premium_scale_half_halves_the_premium():
    mus = {'VOO': 0.06, 'AVUV': 0.12}
    out = pa.scale_equity_premia(mus, 0.5)
    assert abs(out['AVUV'] - 0.09) < 1e-15, out       # 6% + 0.5 * 6pp


# --- interactive runner: run-spec resolution ----------------------------------------
def _P():
    return load_portfolios_from_csv(verbose=False)


def test_resolve_no_derisk_is_the_bare_portfolios():
    specs, grid = runner.resolve_strategies(
        {'mode': 'retirement', 'portfolios': ['141', '140']}, _P())
    assert specs == ['141', '140'] and grid is False, (specs, grid)


def test_resolve_single_base_with_derisk_is_flat_not_grid():
    """One portfolio x N bond tiers varies on ONE axis, so it stays a flat table."""
    specs, grid = runner.resolve_strategies(
        {'mode': 'retirement', 'portfolios': ['141'], 'derisk_at': 65}, _P())
    assert specs[0] == '141' and len(specs) > 1, specs
    assert grid is False, "single base is one axis"
    assert all(s.startswith('141') for s in specs), specs


def test_resolve_multi_base_with_derisk_is_a_grid():
    specs, grid = runner.resolve_strategies(
        {'mode': 'retirement', 'portfolios': ['141', '140'], 'derisk_at': 65}, _P())
    assert grid is True, "two axes should select the matrix layout"
    assert any(s.startswith('141>') for s in specs) and any(s.startswith('140>') for s in specs)


def test_resolve_custom_bypasses_generation():
    specs, grid = runner.resolve_strategies(
        {'mode': 'retirement', 'portfolios': ['141'], 'derisk_at': 65,
         'custom': ['141>540@65>240@80']}, _P())
    assert specs == ['141>540@65>240@80'] and grid is False, (specs, grid)


def test_build_config_omits_withdrawal_for_accumulation():
    acc = runner.build_config({'mode': 'accumulation', 'portfolios': ['141']})
    assert 'withdrawal' not in acc
    ret = runner.build_config({'mode': 'retirement', 'portfolios': ['141'],
                               'spend': 'solve'})
    assert ret['withdrawal']['start_age'] == acc['retirement_age']
    assert ret['withdrawal']['policy'] == 'guyton_klinger'


def test_build_config_uses_a_fixed_spend_when_given():
    cfg = runner.build_config({'mode': 'retirement', 'portfolios': ['141'],
                               'spend': 123_000.0})
    assert cfg['withdrawal']['target_real_annual'] == 123_000.0


def test_every_generated_spec_parses_and_validates():
    """Every schedule the runner GENERATES must survive the validator.

    Had no assertion of its own: it relied on validate() raising, so it passed
    against a no-op. Found by tools/vacuity_sweep.py --static. The behaviour was
    covered elsewhere — gutting glide.validate fails two other checks — but a
    test that cannot fail should not be the one carrying this name.
    """
    P = _P()
    specs, _ = runner.resolve_strategies(
        {'mode': 'retirement', 'portfolios': ['141', '140', '145'],
         'derisk_at': 65}, P)
    assert len(specs) >= 6, specs        # 3 bases x at least 2 tiers
    for s in specs:
        parsed = glide.parse_schedule(s, 37)
        assert parsed, s
        glide.validate(parsed, P)

    # ...and the validator must still REJECT something, or the loop above is
    # asserting that a no-op does not raise.
    try:
        glide.validate(glide.parse_schedule('141>NOSUCH@65', 37), P)
    except (ValueError, KeyError):
        pass
    else:
        raise AssertionError('validate accepted an unknown portfolio id')


def test_preset_round_trips():
    """A TEMP presets file, never the user's config/runs.json. This used to
    write the real file twice — add `_selftest`, then remove it — which left the
    content identical (so test_zzz passed) while a cloud-synced project folder
    turned the two rapid writes into an edit-conflict copy
    ("runs (# Edit conflict ... #).json", observed 2026-09-29). Interrupted
    between the writes, it would also have left `_selftest` in your presets."""
    real = runner.PRESETS
    runner.PRESETS = Path(tempfile.mkdtemp()) / 'runs.json'
    try:
        spec = {'mode': 'retirement', 'portfolios': ['141'], 'spend': 'solve'}
        path = runner.save_preset(spec, '_selftest')
        assert path == runner.PRESETS and path != real, path
        assert runner.load_preset('_selftest') == spec
        # a second preset must not clobber the first
        runner.save_preset({'mode': 'accumulation', 'portfolios': ['140']}, 'b')
        assert runner.load_preset('_selftest') == spec
    finally:
        runner.PRESETS = real


def test_library_render_collapses_bond_tiers():
    """The collapse is driven by the derived bond_pct, not a hardcoded list.

    The live library has held no bonded rows since 2026-09-13, so this builds
    one — the branch must still work if anyone materialises a rung again.
    """
    P = dict(_P())
    P['940'] = {'name': 'synthetic tier', 'allocation': with_bonds(
        P['140']['allocation'], 0.10, 'core')}
    text = runner.render_library(P)
    assert runner.BOND_TIER_NOTE in text, "bond-tier siblings should collapse to one line"
    assert 'momentum' in text and '940' in text, text


def test_matrix_columns_span_every_base_not_just_the_first():
    """Bases have ragged variant counts — 141 (momentum) has four bond tiers,
    120 (value) has none. Columns keyed off one base misalign the rest."""
    P = _P()
    spec = {'mode': 'retirement', 'portfolios': ['144', '141'], 'derisk_at': 65}
    cols = runner.bond_tier_columns(spec, P)
    assert len(cols) >= 4, cols          # union, though 120 alone would give 1
    assert cols[0][1] == 'no derisk'
    assert [t for t, _ in cols] == sorted(t for t, _ in cols), cols


def test_target_bond_pct_reads_the_final_waypoint():
    P = _P()
    assert runner.target_bond_pct('141', P) == 0
    assert runner.target_bond_pct('141>141+20@65', P) > 0.15
    # a re-risking path is placed by where it ENDS, not its deepest point
    assert (runner.target_bond_pct('141>141+20@65>141+5@80', P)
            == runner.target_bond_pct('141+5', P))


def test_policy_prompt_covers_every_implemented_policy():
    """A policy that exists but is unreachable from the runner is a dead feature —
    which is exactly what guyton_klinger being hardcoded made the other three."""
    assert set(runner.POLICY_ORDER) == set(policies.POLICIES), (
        set(policies.POLICIES) - set(runner.POLICY_ORDER))


def test_estimate_scales_with_the_policy_sweep():
    """Grows with the sweep, but NOT linearly: vpw solves in closed form and
    costs a fraction of a searched policy. This asserted 4x for a 4-policy
    sweep, true only while every policy was priced alike."""
    base = {'mode': 'retirement', 'portfolios': ['141'], 'spend': 'solve'}
    searched = runner.estimate_minutes(4, base)                  # guyton_klinger
    cheap = runner.estimate_minutes(4, dict(base, compare_policies=['vpw']))
    swept = runner.estimate_minutes(
        4, dict(base, compare_policies=list(runner.POLICY_ORDER)))
    n_cheap = sum(1 for p in runner.POLICY_ORDER if p in policies.SELF_DIRECTED)
    n_searched = len(runner.POLICY_ORDER) - n_cheap
    assert abs(swept - (n_searched * searched + n_cheap * cheap)) < 1e-9
    assert swept > searched, 'must still grow with the sweep'
    assert cheap < searched / 3, 'closed form must be much cheaper'


def test_floor_grid_brackets_the_default():
    assert solver.FLOOR_GRID[0] < 0.85 <= solver.FLOOR_GRID[-1], solver.FLOOR_GRID
    assert solver.FLOOR_GRID[-1] == 1.00, "must test 'no cut tolerated' at the top"


def test_reference_data_units_are_as_the_parsers_assume():
    """Ken French CSVs are in PERCENT; the AQR workbook is in DECIMAL.

    Applying one convention to both scaled JQUA's 1973-2017 segment down 100x.
    A units check is cheap and the failure was invisible for a long time.
    """
    ref = ROOT / 'reference'
    kf = pd.read_csv(ref / '25_Portfolios_5x5.csv', skiprows=15, nrows=40)
    vals = pd.to_numeric(kf.iloc[:, 1], errors='coerce').dropna().abs()
    assert vals.median() > 0.5, f"Ken French should be percent, median {vals.median()}"

    aqr = pd.read_excel(ref / 'Quality Minus Junk 10 QualitySorted Portfolios Monthly.xlsx',
                        sheet_name='10 Portfolios Formed on Quality', skiprows=18)
    q = pd.to_numeric(aqr['P10 (high quality)'], errors='coerce').dropna().abs()
    assert q.median() < 0.5, f"AQR should be decimal, median {q.median()}"
    # sanity: as-read, AQR annualises to an equity-like volatility
    assert 0.08 < q.std() * 12 ** 0.5 < 0.30, q.std() * 12 ** 0.5


# --- run persistence ----------------------------------------------------------------
def _fake_result(spend=150_000.0):
    return {'spend': spend, 'success_rate': 0.92, 'depletion_rate': 0.01,
            'median_realised': spend * 1.1, 'p10_realised': spend * 0.9,
            'worst_cut': 0.05, 'median_bequest': 2.0e7,
            'success_by_floor': {0.7: 0.98, 0.85: 0.94, 1.0: 0.88},
            'per_path_mean': np.linspace(spend * 0.8, spend * 1.4, 200),
            'spending': np.full((200, 22), spend),
            'depletion_ages': [None] * 200, 'start_age': 65}


def test_saved_run_stores_scalars_not_arrays():
    """Path arrays are ~48MB per portfolio; percentiles answer every question a
    stored run is asked."""
    spec = {'mode': 'retirement', 'portfolios': ['141'], 'spend': 'solve'}
    by_policy = {'guyton_klinger': {'141': _fake_result()}}
    with tempfile.TemporaryDirectory() as td:
        path = persistence.save_run(spec, by_policy, ['141'],
                                    runner.build_config(spec), outdir=td)
        rec = json.loads(path.read_text())
    saved = rec['results']['guyton_klinger']['141']
    assert 'spending' not in saved and 'per_path_mean' not in saved, saved.keys()
    assert saved['spend'] == 150_000.0
    assert saved['spend_pctiles']['50'] > 0
    assert set(saved['success_by_floor']) == {'0.7', '0.85', '1.0'}


def test_saved_run_captures_provenance():
    """A quarterly regen moves every parameter, so a stored result is meaningless
    without knowing which parameter file produced it."""
    prov = persistence.provenance()
    for field in ('parameters_last_updated', 'inflation', 'premium_scale',
                  'ticker_count'):
        assert field in prov, field
    assert prov['inflation'] == pa.INFLATION
    assert prov['premium_scale'] == pa.PREMIUM_SCALE


def test_diff_reports_both_values_and_provenance_changes():
    spec = {'mode': 'retirement', 'spend': 'solve'}      # solving -> spend IS an output
    older = {'timestamp': 'a', 'spec': spec, 'strategies': ['141'],
             'provenance': {'premium_scale': 1.0, 'parameters_last_updated': '2026-06-23'},
             'results': {'gk': {'141': {'spend': 100_000.0}}}}
    newer = {'timestamp': 'b', 'spec': spec, 'strategies': ['141'],
             'provenance': {'premium_scale': 0.5, 'parameters_last_updated': '2026-06-23'},
             'results': {'gk': {'141': {'spend': 60_000.0}}}}
    rows, prov = persistence.diff_runs(older, newer)
    assert rows == [(('gk', '141'), 100_000.0, 60_000.0, -0.4)], rows
    assert prov == [('premium_scale', 1.0, 0.5)], prov


def test_headline_metric_never_reports_an_input_as_a_result():
    """`spend` is an OUTPUT when solving and an INPUT when a fixed amount is given.

    Conflating the two produced three separate bugs: a matrix of the user's own
    input, a policy column headed 'sustainable' that was the target, and a diff
    reporting +0.0%. One rule, pinned here.
    """
    hm = persistence.headline_metric
    assert hm({'mode': 'retirement', 'spend': 'solve'}) == 'spend'
    assert hm({'mode': 'retirement', 'spend': 130_000.0}) == 'success_rate'
    assert hm({'mode': 'accumulation'}) == 'median'
    # an under-specified spec must not fall back to reporting the input
    assert hm({}) != 'spend'


# --- bond sleeves as an independent axis ---------------------------------------------
# From the single source of truth, not a copy: a hand-kept list here went
# stale the day STIP was added (2026-10-02).
_BONDS = set(loaders._BOND_FUNDS)


def test_synthesis_reproduces_the_stored_tier_rows():
    """The safety property the whole refactor rests on.

    The CSV used to materialise nine rows that were exact derivatives of three
    bases. They were deleted on 2026-09-13 once synthesis replaced them — but
    the proof that synthesis reproduces them has to survive the deletion, so it
    reads a FROZEN fixture carrying each base AND its expected rung.

    Self-contained on purpose: the old version read the live library, so editing
    a base portfolio broke it. That happened — a COST change to 145 left
    245/345/545 stale and failed this check along with check_templates. The
    fixture cannot go stale because nothing else can edit it.
    """
    fx = json.loads((Path(__file__).resolve().parent /
                      'bond_tier_fixture.json').read_text())
    assert len(fx['cases']) == 9, fx['cases']
    for c in fx['cases']:
        got = with_bonds(c['base'], c['bond_pct'], c['sleeve'])
        want = c['expected']
        worst = max(abs(got.get(t, 0) - want.get(t, 0)) for t in set(got) | set(want))
        assert worst < 1e-12, (f"{c['base_id']}+{c['bond_pct']:.0%}({c['sleeve']}) "
                               f"!= {c['tier_id']}, off by {worst:.1e}")
        assert abs(sum(got.values()) - 1.0) < 1e-12, c['tier_id']

def test_with_bonds_preserves_the_equity_mix_exactly():
    """A ladder rung must differ from its control ONLY in bond weight.

    The old sibling-lookup did not guarantee this: 141's ladder resolved to the
    x40 rows, which hold AVDE where 141 holds VEA, so 'derisking' bundled a fund
    swap with the bond decision.
    """
    P = _P()
    base = P['141']['allocation']
    ref = equity_sleeve(base)
    for pct in (0.0, 0.05, 0.10, 0.20, 0.5):
        got = with_bonds(base, pct, 'core')
        assert abs(sum(got.values()) - 1.0) < 1e-12, pct
        assert abs(sum(w for t, w in got.items() if t in _BONDS) - pct) < 1e-12, pct
        mix = equity_sleeve(got)
        for t in set(ref) | set(mix):
            assert abs(ref.get(t, 0) - mix.get(t, 0)) < 1e-12, (t, pct)


def test_141_ladder_no_longer_swaps_funds():
    """Regression for the confound: 141's rungs must still hold VEA, not AVDE."""
    P = _P()
    specs = glide.derisk_variants('141', 65)
    base_eq = equity_sleeve(P['141']['allocation'])
    assert base_eq.get('VEA', 0) > 0 and 'VEA' in base_eq
    for spec in specs[1:]:
        alloc = glide.resolve_ref(spec.split('>')[-1].split('@')[0], P)
        mix = equity_sleeve(alloc)
        assert abs(mix.get('VEA', 0) - base_eq['VEA']) < 1e-12, spec
        assert abs(mix.get('AVDE', 0) - base_eq['AVDE']) < 1e-12, spec


def test_reference_syntax_round_trips_and_rejects_nonsense():
    P = _P()
    assert glide.resolve_ref('141', P) == P['141']['allocation']
    a = glide.resolve_ref('141+10', P)
    assert abs(sum(w for t, w in a.items() if t in _BONDS) - 0.10) < 1e-12
    b = glide.resolve_ref('141+10:bnd_only', P)
    assert abs(b['BND'] - 0.10) < 1e-12 and b.get('JPLD', 0) == 0
    for bad in ('999+10', '141+abc', '141+10:nosuch', 'nope'):
        try:
            glide.resolve_ref(bad, P)
        except ValueError:
            continue
        raise AssertionError(f"should have rejected {bad!r}")


def test_sleeve_choice_is_independent_of_equity_sleeve():
    """The point of the third axis: any equity side with any bond sleeve."""
    P = _P()
    for base in ('141', '140', '145', '144'):
        for sleeve in BOND_SLEEVES:
            a = with_bonds(P[base]['allocation'], 0.10, sleeve)
            assert abs(sum(a.values()) - 1.0) < 1e-12, (base, sleeve)
            assert abs(sum(w for t, w in a.items() if t in _BONDS) - 0.10) < 1e-12


def test_derisk_variants_labels_a_non_native_sleeve():
    P = _P()
    native = glide.derisk_variants('141', 65)
    assert all(':' not in s for s in native), native
    swapped = glide.derisk_variants('141', 65, sleeve='bnd_only')
    assert all(':bnd_only' in s for s in swapped[1:]), swapped
    assert native_sleeve('145') == 'vanguard'


def test_synthesised_allocations_carry_the_full_ticker_universe():
    """The engine iterates TICKERS when rebalancing, so a synthesised allocation
    must hold every ticker with explicit zeros, exactly as the CSV rows do.

    Returning only the non-zero positions raised KeyError in
    execute_full_rebalance — and only AFTER a glide switched onto the synthesised
    target at 65, so the control leg of every comparison ran clean first.
    """
    P = _P()
    stored_keys = set(P['141']['allocation'])
    for ref in ('141+5', '141+10', '141+20', '141+10:bnd_only', '145+10'):
        alloc = glide.resolve_ref(ref, P)
        assert set(alloc) == stored_keys, (
            ref, stored_keys - set(alloc), set(alloc) - stored_keys)
        assert set(alloc) == set(pa.TICKERS), ref
        assert abs(sum(alloc.values()) - 1.0) < 1e-12, ref


# --- gaps found by auditing what the session's bugs had in common -------------------
def test_light_worker_matches_the_full_simulation():
    """The 2.9x speedup added a parallel code path. If it diverges, every solver
    result is quietly wrong and nothing fails."""
    P = _P()
    plan = glide.build_plan(glide.parse_schedule('141>141+10@65', 37), P)
    cfg = _wd_cfg_real()
    for i in (0, 7, 42):
        full = pa.run_single_simulation((plan, cfg, i))
        light = pa.run_single_simulation_light((plan, cfg, i))
        assert full[0] == light[0], (i, full[0], light[0])
        assert full[4]['depletion_age'] == light[4]['depletion_age'], i
        assert full[4]['spending'] == light[4]['spending'], i
        assert len(light) == len(full), "shape must match so callers index alike"


def _wd_cfg_real(spend=150_000.0):
    return {'age': 37, 'retirement_age': 65, 'time_horizon_years': 50,
            'initial_value': 300000, 'rebalancing_strategy': 'quarterly_annual',
            'contribution_schedule': {'frequency': 'annual', 'accounts': {
                'm': {'annual_amount': 12000, 'growth_rate': 0.0, 'stop_age': 60}}},
            'withdrawal': {'start_age': 65, 'policy': 'guyton_klinger',
                           'sourcing': 'excess', 'target_real_annual': spend,
                           'floor': 0.85}}


def test_a_fund_sees_the_same_market_whatever_else_the_plan_holds():
    """Common random numbers must hold ACROSS fund sets, not just within one.

    The engine drew its randomness for the funds a plan held — one parameter
    draw each, a regime draw each, and a correlated shock through the Cholesky
    factor of THEIR submatrix. So two strategies holding different funds saw
    different markets, and every head-to-head between them was unpaired while
    the report said 'every strategy runs on the SAME random markets'. Measured
    2026-10-01: `166` and `166>166+5@94` — the same strategy until 94 — read
    587k and 603k, and the sleeve-vs-no-bonds interval was 49k wide against 9k
    once the control held the same funds.

    AVUV alone, against a plan that lists VOO at an age the horizon never
    reaches: VOO sorts BEFORE AVUV, so it changes the submatrix, the parameter
    draws and the regime draws all at once. AVUV's returns must not move."""
    def full(**w):
        out = {t: 0.0 for t in pa.TICKERS}
        out.update(w)
        return out
    cfg = {'age': 37, 'retirement_age': 65, 'time_horizon_years': 10,
           'initial_value': 100_000.0, 'rebalancing_strategy': 'none',
           'contribution_schedule': {'frequency': 'annual', 'accounts': {}}}
    alone = ((float('-inf'), full(AVUV=1.0)),)
    listed = ((float('-inf'), full(AVUV=1.0)), (90.0, full(VOO=1.0)))
    assert pa.TICKERS.index('VOO') < pa.TICKERS.index('AVUV')
    for sim_id in range(4):
        a = pa.run_single_simulation((alone, cfg, sim_id))[2]
        b = pa.run_single_simulation((listed, cfg, sim_id))[2]
        worst = max(abs(x - y) for x, y in zip(a, b))
        assert worst < 1e-12, (
            f"sim {sim_id}: AVUV's monthly return moves by up to {worst:.2e} "
            f"because the plan also lists VOO — the market depends on the fund set")


def test_a_never_held_sleeve_does_not_change_what_you_spend():
    """The user-facing form of the fix above: a strategy that only differs in a
    waypoint the horizon never reaches must spend IDENTICALLY, year by year,
    through guyton_klinger's discrete guardrail decisions."""
    P = _P()
    cfg = _wd_cfg_real()                         # GK, to age 87
    bare = glide.build_plan(glide.parse_schedule('166', cfg['age']), P)
    late = glide.build_plan(glide.parse_schedule('166>166+5@94', cfg['age']), P)
    for sim_id in range(3):
        a = pa.run_single_simulation((bare, cfg, sim_id))[4]['spending']
        b = pa.run_single_simulation((late, cfg, sim_id))[4]['spending']
        assert len(a) == len(b) > 0
        worst = max(abs(x - y) / max(abs(x), 1.0) for x, y in zip(a, b))
        assert worst < 1e-9, (
            f"sim {sim_id}: '166' and '166>166+5@94' spend differently (up to "
            f"{worst:.1%} in a year) though they hold the same thing to age 87")


def _install_universe(tickers, corr, vols, transitions, mu=0.0, param_std=0.0):
    """A synthetic universe with both shared shocks off and the MARKET chain
    present, so the production branch of the regime loop is the one exercised.
    `vols` is per regime, shared by every fund. Install it inside
    _engine_state(), which puts the real universe back."""
    stay = {r: {'low': 0.0, 'medium': 1.0, 'high': 0.0}
            for r in ('low', 'medium', 'high')}
    pa.RF_REAL_STD = pa.INFLATION_STD = 0.0
    pa.TICKERS = list(tickers)
    pa.EXPECTED_RETURNS = {t: mu for t in tickers}
    pa.PARAM_UNCERTAINTY_STD = {t: param_std for t in tickers}
    pa.VOLATILITY_REGIMES = {t: dict(vols) for t in tickers}
    pa.REGIME_TRANSITIONS = {**{t: transitions for t in tickers},
                             pa.MARKET_REGIME_KEY: stay}
    for m in ('BASELINE', 'LOW', 'MEDIUM', 'HIGH'):
        setattr(pa, f'CORRELATION_{m}', np.array(corr, dtype=float))


def _alone(ticker):
    """A 100% book in one fund of the installed universe."""
    return {t: float(t == ticker) for t in pa.TICKERS}


_SYNTH_CFG = {'age': 40, 'retirement_age': 65, 'time_horizon_years': 10,
              'initial_value': 100_000.0, 'rebalancing_strategy': 'none',
              'contribution_schedule': {'frequency': 'annual', 'accounts': {}}}


def test_each_fund_draws_its_own_estimation_error():
    """Universe-sized draws must still be ONE PER FUND. Reading one shared
    position would pass both common-random-number tests above — every fund
    would get the same error on every path whatever the plan holds — while
    making estimation error perfectly correlated across funds, the opposite of
    what an independent per-fund band means. Zero volatility, so a 100% book's
    monthly return IS its drawn mu / 12."""
    stay = {r: {'low': 0.0, 'medium': 1.0, 'high': 0.0}
            for r in ('low', 'medium', 'high')}
    with _engine_state():
        _install_universe(['A', 'B'], np.eye(2),
                          {'low': 0.0, 'medium': 0.0, 'high': 0.0}, stay,
                          mu=0.05, param_std=0.02)
        cfg = dict(_SYNTH_CFG, time_horizon_years=1)
        mu = {t: [12 * pa.run_single_simulation((_alone(t), cfg, i))[2][0]
                  for i in range(150)] for t in ('A', 'B')}
        sd = float(np.std(mu['A']))
        assert 0.015 < sd < 0.025, f"the band is not live: sd {sd:.4f}, set 0.02"
        corr = float(np.corrcoef(mu['A'], mu['B'])[0, 1])
        assert abs(corr) < 0.3, (
            f"A's and B's estimation errors correlate at {corr:.2f} — one shared "
            f"draw, not one per fund (independent ~0, shared exactly 1)")


def test_each_fund_draws_its_own_regime_transition():
    """One uniform per fund per month, not one per month. A shared draw moves
    every fund with the same chain into the same regime in the same month — and
    still passes the common-random-number tests above. Two funds, the same
    mixing chain, the same per-regime vols (5% / 15% / 40%), shocks correlated
    at 0.999: the ratio of their monthly moves is then the ratio of their
    CURRENT vols — ~1 in lockstep, 2.7-8x whenever their regimes differ, which
    independent chains do two months in three."""
    mix = {r: {'low': 1 / 3, 'medium': 1 / 3, 'high': 1 / 3}
           for r in ('low', 'medium', 'high')}
    with _engine_state():
        _install_universe(['A', 'B'], [[1.0, 0.999], [0.999, 1.0]],
                          {'low': 0.05, 'medium': 0.15, 'high': 0.40}, mix)
        apart = []
        for sim_id in range(2):
            ra = np.array(pa.run_single_simulation((_alone('A'), _SYNTH_CFG, sim_id))[2])
            rb = np.array(pa.run_single_simulation((_alone('B'), _SYNTH_CFG, sim_id))[2])
            apart.extend(np.abs(np.log(np.abs(ra) / np.abs(rb))) > 0.5)
        share = float(np.mean(apart))
        assert share > 0.25, (
            f"A and B sat in different vol regimes in only {share:.0%} of months "
            f"— their chains move in lockstep (independent ~65%, shared ~4%)")


def test_funds_held_apart_keep_their_correlation():
    """A fund's shock is its ROW of the full factor. Three funds, B between A
    and C in universe order, corr(A, C) = 0.8; A and C are each held ALONE on
    the same paths, so across months their returns must correlate at 0.8 and C
    must keep its own volatility. Reading the first rows instead of the held
    ones gives C the row of A — correlation 1.0 — and so did the pre-2026-10-02
    engine, which handed every single-fund book the same first draw."""
    stay = {r: {'low': 0.0, 'medium': 1.0, 'high': 0.0}
            for r in ('low', 'medium', 'high')}
    corr = [[1.0, 0.3, 0.8], [0.3, 1.0, 0.5], [0.8, 0.5, 1.0]]
    with _engine_state():
        _install_universe(['A', 'B', 'C'], corr,
                          {'low': 0.16, 'medium': 0.16, 'high': 0.16}, stay)
        ra, rc = [], []
        for sim_id in range(10):
            ra.extend(pa.run_single_simulation((_alone('A'), _SYNTH_CFG, sim_id))[2])
            rc.extend(pa.run_single_simulation((_alone('C'), _SYNTH_CFG, sim_id))[2])
        r = float(np.corrcoef(ra, rc)[0, 1])
        assert abs(r - 0.8) < 0.06, f"A and C correlate at {r:.3f}, target 0.8"
        ratio = float(np.std(rc)) / (0.16 / np.sqrt(12))
        assert abs(ratio - 1.0) < 0.08, f"C's volatility is {ratio:.2f}x its own"


def _premium_draws(n):
    """The engine's own per-path premium draws, correlated as PREMIUM_CORR says."""
    return np.array([pa.premium_draw(i) for i in range(n)])


def _expected_error_corr(a, b):
    """Error correlation two funds' loadings imply, with the same cap the engine
    applies when a fund's shared part would exceed its band."""
    C = np.asarray(pa.PREMIUM_CORR)
    ua, ub = pa.PREMIUM_SHARES[a], pa.PREMIUM_SHARES[b]
    qa, qb = float(ua @ C @ ua), float(ub @ C @ ub)
    return float(ua @ C @ ub) / (max(qa, 1.0) * max(qb, 1.0)) ** 0.5


def _sample_errors(ticker, own_z, prem_z):
    """A fund's estimation errors for a batch of draws (prem_z None = independent)."""
    band = pa.PARAM_UNCERTAINTY_STD[ticker]
    return np.array([pa.estimation_error(ticker, band, own_z[i],
                                         None if prem_z is None else prem_z[i])
                     for i in range(len(own_z))])


def test_a_fund_keeps_its_own_band_whether_errors_are_shared_or_not():
    """Sharing the premia re-routes a fund's estimation error; it must not
    resize it. Each band is built and checked elsewhere (factors.parameter_std);
    splitting it into a shared part and an own part has to give it back
    exactly, or switching MC_PARAM_ERRORS would quietly change every fund's
    uncertainty as well as how the funds move together."""
    rng = np.random.default_rng(7)
    n = 8000
    own = rng.standard_normal(n)
    prem = _premium_draws(n)
    funds = ['VOO', 'AVLV', 'AVUV', 'DFWIX', 'AVEM']
    missing = [t for t in funds if t not in pa.PREMIUM_SHARES]
    assert not missing, f"no premium shares for factor funds {missing}"
    for t in funds + ['BND']:
        for mode, z in (('shared', prem), ('independent', None)):
            ratio = float(np.std(_sample_errors(t, own, z))) / pa.PARAM_UNCERTAINTY_STD[t]
            assert abs(ratio - 1.0) < 0.03, f"{t} ({mode}): band drawn at {ratio:.3f}x its size"


def test_funds_that_load_on_one_premium_share_its_error():
    """The point of 'shared': VOO's band is essentially the US market premium's
    standard error, and so is most of AVLV's — if that premium is overstated it
    is overstated for both. Their errors must correlate as their loadings say
    (share_VOO . share_AVLV), well above zero, while BND, which the factor model
    does not cover, stays independent. Drawn independently, as until
    2026-10-02, every pair reads ~0 — which let a nine-fund portfolio average
    away a premium it holds nine times."""
    rng = np.random.default_rng(11)
    n = 8000
    own = rng.standard_normal((n, 3))
    prem = _premium_draws(n)
    voo, avlv = _sample_errors('VOO', own[:, 0], prem), _sample_errors('AVLV', own[:, 1], prem)
    expect = _expected_error_corr('VOO', 'AVLV')
    assert expect > 0.5, f"VOO and AVLV share only {expect:.2f} of their error"
    got = float(np.corrcoef(voo, avlv)[0, 1])
    assert abs(got - expect) < 0.04, f"VOO-AVLV errors correlate at {got:.3f}, loadings say {expect:.3f}"
    bnd = _sample_errors('BND', own[:, 2], prem)
    assert abs(float(np.corrcoef(voo, bnd)[0, 1])) < 0.05, 'BND picked up a premium it has no loading on'
    ind = float(np.corrcoef(_sample_errors('VOO', own[:, 0], None),
                            _sample_errors('AVLV', own[:, 1], None))[0, 1])
    assert abs(ind) < 0.05, f"'independent' errors correlate at {ind:.3f}"


def test_the_premium_draws_carry_the_measured_correlation():
    """The per-path premium draw must reproduce PREMIUM_CORR — US and developed
    market premia +0.57, developed and emerging +0.76 — with unit variance per
    premium. Drawn as 18 independent normals (as until the correlation landed)
    every off-diagonal reads ~0 and a US miss never coincides with a developed
    one."""
    z = _premium_draws(6000)
    C = np.asarray(pa.PREMIUM_CORR)
    got = np.corrcoef(z.T)
    worst = float(np.max(np.abs(got - C)))
    assert worst < 0.06, f"premium draws miss PREMIUM_CORR by up to {worst:.3f}"
    sd = z.std(axis=0)
    assert np.all(np.abs(sd - 1.0) < 0.05), sd
    k = {key: i for i, key in enumerate(pa.PREMIUM_KEYS)}
    us, dev = k[('us', 'mkt_rf')], k[('dev', 'mkt_rf')]
    assert C[us, dev] > 0.4, 'the regional market link is missing from the table'


def test_a_us_fund_and_a_developed_fund_now_err_together():
    """VOO's band is the US market premium and AVDE's the developed one. With
    the premia drawn independently their errors were unrelated (0.00), so
    holding both diversified away a miss that, measured, tends to land on both.
    They must now correlate as the loadings and PREMIUM_CORR imply — well
    above zero."""
    rng = np.random.default_rng(13)
    n = 8000
    own = rng.standard_normal((n, 2))
    prem = _premium_draws(n)
    voo, avde = _sample_errors('VOO', own[:, 0], prem), _sample_errors('AVDE', own[:, 1], prem)
    expect = _expected_error_corr('VOO', 'AVDE')
    assert expect > 0.4, f"loadings imply only {expect:.2f}"
    got = float(np.corrcoef(voo, avde)[0, 1])
    assert abs(got - expect) < 0.04, f"VOO-AVDE errors correlate at {got:.3f}, expected {expect:.3f}"


def test_the_premium_correlation_table_is_what_the_factor_data_says():
    """PREMIUM_CORR is a pasted table, like PREMIUM_SE — so pin it to the data
    it was measured from. Re-measured from the Ken French series by
    tools/premia.py, every cell must match to its printed precision; a table
    left behind by a data refresh, or pasted in the wrong key order, fails."""
    import premia as _premia
    import data_collection_regime_model as dc
    kf = dc.KenFrenchParser(str(ROOT / 'reference'))
    # The measurement and the table follow the SAME switch. Measuring on the
    # default window while MC_PREMIA_LONG_WINDOW picked the long table compared
    # the wrong measurement to the wrong table; the long table is pinned too,
    # whatever the env says.
    for name, table, long_window in (
            ('PREMIUM_CORR', factors.PREMIUM_CORR, factors.PREMIA_LONG_WINDOW),
            ('PREMIUM_CORR_LONG', factors.PREMIUM_CORR_LONG, True)):
        measured = _premia.error_correlation(
            _premia.series_by_region(kf, long_window=long_window), factors.PREMIUM_KEYS)
        table = np.asarray(table)
        worst = float(np.max(np.abs(measured - table)))
        assert worst <= 0.0051, f"{name} is stale: up to {worst:.3f} off the data"
        assert np.allclose(table, table.T) and np.allclose(np.diag(table), 1.0)
        assert np.linalg.eigvalsh(table).min() > 0.05, f'{name} is not positive definite'


def test_the_shared_premium_draw_is_one_per_path_for_every_strategy():
    """Wiring, end to end: the engine must take the premium errors from ONE draw
    per path — the same draw whatever the plan holds. Two synthetic funds whose
    bands are entirely the same premium must then get IDENTICAL errors on every
    path — A held alone, B held in a plan that also LISTS A, so the two plans
    differ in fund count too — and unrelated ones under 'independent'. Zero
    volatility, so a 100% book's monthly return is its drawn mu / 12."""
    stay = {r: {'low': 0.0, 'medium': 1.0, 'high': 0.0}
            for r in ('low', 'medium', 'high')}
    with _engine_state():
        _install_universe(['A', 'B'], np.eye(2),
                          {'low': 0.0, 'medium': 0.0, 'high': 0.0}, stay,
                          mu=0.05, param_std=0.02)
        whole = np.zeros(len(pa.PREMIUM_KEYS))
        whole[0] = 1.0
        pa.PREMIUM_SHARES = {'A': whole, 'B': whole}
        cfg = dict(_SYNTH_CFG, time_horizon_years=1)

        b_with_a_listed = ((float('-inf'), _alone('B')), (90.0, _alone('A')))

        def mus(plan):
            return np.array([12 * pa.run_single_simulation((plan, cfg, i))[2][0]
                             for i in range(60)])
        pa.PARAM_ERRORS = 'shared'
        a, b = mus(_alone('A')), mus(b_with_a_listed)
        assert float(np.std(a)) > 0.01, 'the shared draw never reached the engine'
        assert float(np.max(np.abs(a - b))) < 1e-12, \
            'A and B drew different premium errors on the same path'
        pa.PARAM_ERRORS = 'independent'
        a, b = mus(_alone('A')), mus(b_with_a_listed)
        assert abs(float(np.corrcoef(a, b)[0, 1])) < 0.4, "'independent' still shares the draw"


def test_error_sharing_is_a_knob_workers_and_reports_carry():
    """MC_PARAM_ERRORS is read at import (workers re-import under `spawn`), is
    announced and replayed when moved, is recorded in provenance so `--diff`
    can attribute a change to it, and is sweepable. A typo must fail loudly
    rather than silently fall back to either setting."""
    import sensitivity
    assert diagnostics.DEFAULT_ASSUMPTIONS['PARAM_ERRORS'] == ('shared', 'MC_PARAM_ERRORS')
    assert sensitivity.DIMENSIONS['param_errors'].env == 'MC_PARAM_ERRORS'
    assert persistence.provenance()['param_errors'] == pa.PARAM_ERRORS == 'shared'
    code = ("import sys; sys.path.insert(0, %r); from montecarlo import engine; print(engine.PARAM_ERRORS)"
            % str(ROOT))
    out, bad = _children([(code, dict(os.environ, MC_PARAM_ERRORS=v))
                          for v in ('independent', 'sharde')])
    assert out.stdout.split()[-1:] == ['independent'], out.stdout + out.stderr
    assert bad.returncode != 0 and 'MC_PARAM_ERRORS' in bad.stderr, bad.stderr[-300:]


def test_premium_scale_reaches_spawned_workers():
    """MC_PREMIUM_SCALE is read from the environment at import. Workers re-import
    under `spawn`, so an env var propagates where an assigned module global would
    not — the same constraint that put REBALANCING_BANDS in engine.py.

    If this broke, the parent would report k=0.5 while workers simulated at 1.0:
    confident, wrong, and silent. Checked by running a worker in a subprocess.
    """
    code = ("import os, sys; sys.path.insert(0, %r); from montecarlo import engine; "
            "print(engine.PREMIUM_SCALE, engine.EXPECTED_RETURNS['AVUV'])" % str(ROOT))
    env = dict(os.environ, MC_PREMIUM_SCALE='0.5')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, env=env).stdout.split()
    assert float(out[0]) == 0.5, f"worker saw scale {out[0]}"
    scaled_avuv = float(out[1])
    mkt = pa.EXPECTED_RETURNS[pa.MARKET_PROXY]
    full_prem = pa.NOMINAL_EXPECTED_RETURNS['AVUV']
    # at k=0.5 AVUV's premium over the market must be halved, not full or zero
    assert mkt < scaled_avuv < pa.EXPECTED_RETURNS['AVUV'] + 1e-12, scaled_avuv
    assert abs((scaled_avuv - mkt) - 0.5 * (pa.EXPECTED_RETURNS['AVUV'] - mkt)) < 1e-9


def test_the_premium_haircut_reaches_the_means_the_bands_and_the_shared_draw():
    """MC_PREMIUM_HAIRCUT is the knob that moves the EQUITY premium itself, and
    engine.py hands it to three factors calls (means, bands, premium shares).
    The factors tests pass `haircut=` directly, so a call site that dropped it
    left mu moving while the bands and shares stayed at full history — silent.
    A subprocess, because it is read at import and workers re-import."""
    probe = ("{'nominal': engine.NOMINAL_EXPECTED_RETURNS, "
             "'std': engine.PARAM_UNCERTAINTY_STD, "
             "'shares': {t: [float(x) for x in v] "
             "for t, v in engine.PREMIUM_SHARES.items()}, "
             "'haircut': engine.PREMIUM_HAIRCUT}")
    (full_p, full), (half_p, half), (bad, _) = _engine_attrs(
        (probe, {'MC_PREMIUM_HAIRCUT': '1.0'}),
        (probe, {'MC_PREMIUM_HAIRCUT': '0.5'}),
        ('1', {'MC_PREMIUM_HAIRCUT': 'half'}))
    assert full is not None and half is not None, full_p.stderr[-300:] + half_p.stderr[-300:]
    assert half['haircut'] == 0.5 and full['haircut'] == 1.0

    # mu: exactly what factors builds at that haircut from the same inputs
    want, _ = factors.build_nominal(dict(pa._params['EXPECTED_RETURNS']),
                                    factors.load_fits(), scenario=pa.PREMIA_SCENARIO,
                                    haircut=0.5)
    for t in ('VOO', 'AVUV', 'AVLV'):
        assert abs(half['nominal'][t] - want[t]) < 1e-12, (t, half['nominal'][t], want[t])
        assert half['nominal'][t] < full['nominal'][t] - 1e-4, t
    assert half['nominal']['BND'] == full['nominal']['BND'], 'bonds carry no factor premium'

    # the bands and the shares move with it, not just the means
    assert half['std']['AVUV'] != full['std']['AVUV'], 'bands ignored the haircut'
    assert half['std']['AVUV'] < full['std']['AVUV']
    moved = max(abs(a - b) for a, b in zip(half['shares']['VOO'], full['shares']['VOO']))
    assert moved > 1e-5, 'premium shares ignored the haircut'

    assert bad.returncode != 0 and 'MC_PREMIUM_HAIRCUT' in bad.stderr, bad.stderr[-300:]


def test_depleted_paths_emit_no_warnings():
    """A depleted path used to divide by a zero total_value, yielding nan weights.
    The answer happened to come out right; the warnings did not."""
    import warnings
    P = _P()
    plan = glide.build_plan(glide.parse_schedule('141', 37), P)
    # Depleting this model takes deliberate effort. constant_real is required
    # (guyton_klinger CUTS rather than running out), and the draw must outpace
    # growth from the START — with contributions and 28 years of compounding
    # first, even a $2M/yr draw left $59M standing. So: retire immediately, no
    # contributions, and a 30% withdrawal rate.
    cfg = {'age': 65, 'retirement_age': 65, 'time_horizon_years': 25,
           'initial_value': 1_000_000.0, 'rebalancing_strategy': 'quarterly_annual',
           'contribution_schedule': {'frequency': 'annual', 'accounts': {
               'm': {'annual_amount': 0.0, 'growth_rate': 0.0, 'stop_age': None}}},
           'withdrawal': {'start_age': 65, 'policy': 'constant_real',
                          'sourcing': 'excess', 'target_real_annual': 300_000.0,
                          'floor': 0.85}}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        r = pa.run_single_simulation((plan, cfg, 1))
    bad = [w for w in caught if 'divide' in str(w.message) or 'invalid' in str(w.message)]
    assert not bad, [str(w.message) for w in bad]
    assert r[4]['depletion_age'] is not None, "this spend must deplete"
    assert r[0] >= 0.0 and not np.isnan(r[0])


def test_solver_result_carries_every_key_the_report_reads():
    """Structural mismatch between producer and consumer is a recurring shape
    of bug (the matrix showing an input; synthesised allocations missing keys).
    Pin the contract."""
    P = _P()
    plan = glide.build_plan(glide.parse_schedule('141', 37), P)
    res = solver.evaluate(plan, _wd_cfg_real(), 150_000.0, n_sims=12, parallel=False)
    for key in ('spend', 'success_rate', 'success_by_floor', 'depletion_rate',
                'median_realised', 'p10_realised', 'worst_cut', 'median_bequest',
                'per_path_mean', 'spending', 'depletion_ages', 'start_age'):
        assert key in res, f"solver.evaluate dropped {key!r}"
    assert set(res['success_by_floor']) == set(solver.FLOOR_GRID)
    assert len(res['depletion_ages']) == 12 and res['spending'].shape[0] == 12


def test_parse_tiers_accepts_percentages_and_rejects_nonsense():
    assert runner.parse_tiers('5,10,20') == (0.05, 0.10, 0.20)
    assert runner.parse_tiers('5 10 20') == (0.05, 0.10, 0.20)
    assert runner.parse_tiers('12.5') == (0.125,)
    assert runner.parse_tiers('20,5,10') == (0.05, 0.10, 0.20), 'must sort'
    assert runner.parse_tiers('10,10') == (0.10,), 'must dedupe'
    for bad in ('', '0', '100', '-5', 'abc', '0.5,abc'):
        try:
            runner.parse_tiers(bad)
        except ValueError:
            continue
        raise AssertionError(f'should have rejected {bad!r}')


def test_custom_tiers_and_sleeve_reach_the_generated_specs():
    """The synthesis machinery existed before the prompts did; a feature the
    runner cannot reach is a dead feature, which is what hardcoding
    guyton_klinger did to the other three policies."""
    P = _P()
    spec = {'mode': 'retirement', 'portfolios': ['141'], 'derisk_at': 65,
            'bond_tiers': [0.07, 0.15], 'bond_sleeve': 'bnd_only'}
    specs, _ = runner.resolve_strategies(spec, P)
    assert specs == ['141', '141>141+7:bnd_only@65', '141>141+15:bnd_only@65'], specs
    for sp in specs[1:]:
        alloc = glide.resolve_ref(sp.split('>')[-1].split('@')[0], P)
        assert alloc['BND'] > 0 and alloc.get('JPLD', 0) == 0, 'bnd_only ignored'
    cols = [label for _, label in runner.bond_tier_columns(spec, P)]
    assert cols == ['no derisk', '->7%', '->15%'], cols


def test_every_bond_sleeve_is_offerable_and_described():
    """Same guard as POLICY_ORDER: a sleeve that exists but cannot be chosen,
    or has no description to show, is unreachable."""
    for name in BOND_SLEEVES:
        text = runner.describe_sleeve(name)
        assert text and all(t in text for t in BOND_SLEEVES[name]), (name, text)
    assert abs(sum(BOND_SLEEVES['bnd_only'].values()) - 1.0) < 1e-12
    for name, mix in BOND_SLEEVES.items():
        assert abs(sum(mix.values()) - 1.0) < 1e-12, name


def test_every_reference_helper_handles_synthesised_refs():
    """Indexing portfolios[ref] directly breaks on synthesised refs, and that
    mistake has been made three times (target_bond_pct, confirm(), describe()).
    Every helper must resolve."""
    P = _P()
    for ref in ('141', '141+10', '141+10:bnd_only', '145+20'):
        assert 0 <= runner.ref_bond_pct(ref, P) < 1, ref
        assert '/' in runner.ref_split(ref, P), ref
        assert runner.describe(P, ref), ref
    assert runner.ref_split('141', P) == '100/0'
    assert runner.ref_split('141+20', P) == '80/20'
    assert runner.target_bond_pct('141>141+10@65', P) == runner.ref_bond_pct('141+10', P)


def test_compare_sleeves_holds_equity_and_bond_weight_fixed():
    """Isolating COMPOSITION means every rung shares the equity mix and the bond
    percentage, differing only in which bond funds are used."""
    P = _P()
    spec = {'mode': 'retirement', 'portfolios': ['141'], 'derisk_at': 65,
            'bond_tiers': [0.10], 'compare_sleeves': sorted(BOND_SLEEVES)}
    specs, _ = runner.resolve_strategies(spec, P)
    assert specs[0] == '141'
    assert len(specs) == 1 + len(BOND_SLEEVES), specs

    base_eq = equity_sleeve(P['141']['allocation'])
    seen = set()
    for sp in specs[1:]:
        alloc = glide.resolve_ref(sp.split('>')[-1].split('@')[0], P)
        assert abs(runner.ref_bond_pct(sp.split('>')[-1].split('@')[0], P) - 0.10) < 1e-12
        mix = equity_sleeve(alloc)
        for t in set(base_eq) | set(mix):
            assert abs(base_eq.get(t, 0) - mix.get(t, 0)) < 1e-12, (sp, t)
        bonds = tuple(sorted((t, round(alloc[t], 6)) for t in alloc
                             if t in _BONDS and alloc[t] > 0))
        seen.add(bonds)
    assert len(seen) == len(BOND_SLEEVES), f"sleeves not distinct: {seen}"


def test_single_fund_sleeves_isolate_duration():
    """jpst_only vs bnd_only differ only in the bond fund's duration."""
    P = _P()
    a = glide.resolve_ref('141+10:jpst_only', P)
    b = glide.resolve_ref('141+10:bnd_only', P)
    assert abs(a['JPST'] - 0.10) < 1e-12 and a.get('BND', 0) == 0
    assert abs(b['BND'] - 0.10) < 1e-12 and b.get('JPST', 0) == 0
    assert equity_sleeve(a) == equity_sleeve(b)


def test_custom_schedule_accepts_synthesised_refs():
    """The custom builder is the only route to a multi-step glide, and it
    rejected '141+20' outright — so bond levels were unreachable there."""
    P = _P()
    spec = '141+20>141+10@72>141@80'
    sch = glide.validate(glide.parse_schedule(spec, 37), P)
    assert len(sch) == 3
    pcts = [runner.ref_bond_pct(w.portfolio_id, P) for w in sch]
    assert pcts == sorted(pcts, reverse=True), f"rising-equity path: {pcts}"
    assert abs(pcts[0] - 0.20) < 1e-12 and pcts[-1] == 0


def test_multiple_custom_schedules_resolve_together():
    """Comparing two glide SHAPES in one run — e.g. a single switch at 65 against
    a staged 45/55/65 path — which needed two runs before."""
    P = _P()
    spec = {'mode': 'retirement', 'portfolios': ['140'], 'derisk_at': 65,
            'custom': ['140', '140>140+10@65',
                       '140>140+5@45>140+10@55>140+20@65']}
    specs, grid = runner.resolve_strategies(spec, P)
    assert specs == spec['custom'] and grid is False
    for sp in specs:
        sch = glide.validate(glide.parse_schedule(sp, 37), P)
        glide.build_plan(sch, P)
    staged = glide.parse_schedule(specs[2], 37)
    assert [w.age for w in staged] == [37, 45, 55, 65]
    pcts = [runner.ref_bond_pct(w.portfolio_id, P) for w in staged]
    assert pcts == [0, 0.05, 0.10, 0.20], pcts


def test_custom_start_prompt_rejects_a_whole_schedule():
    """The start prompt takes ONE id. Pasting a full schedule into it used to be
    accepted silently and failed much later with an opaque error."""
    P = _P()
    for whole in ('140+5 at 45, 140+10 at 65', '140, 145'):
        real, runner.ask = runner.ask, _scripted([whole, '140', ''])
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                got = runner.ask_custom_schedule(P, 45, default_base='140')
        finally:
            runner.ask = real
        assert got == '140', (whole, got)
        assert 'ONLY the starting id' in buf.getvalue(), (whole, buf.getvalue())
    # and a bare id, synthesised or not, must still resolve
    for good in ('140', '140+10', '141+10:bnd_only'):
        glide.resolve_ref(good, P)


def test_labels_appear_only_when_specs_are_wide():
    """A label is indirection. Worth it for
    '140>140+5@45>140+10@55>140+20@65' repeated in every column; not for '141'."""
    assert runner.strategy_labels(['141', '140']) == {}, 'short specs need no labels'
    assert runner.strategy_labels(['141']) == {}, 'a single strategy needs none'
    wide = ['140', '140>140+10@65', '140>140+5@45>140+10@55>140+20@65']
    labels = runner.strategy_labels(wide)
    assert labels == {wide[0]: 'A', wide[1]: 'B', wide[2]: 'C'}, labels
    assert len(set(labels.values())) == len(wide), 'labels must be unique'


def test_sortino_note_is_conditional_not_blanket():
    """Sortino DOES separate genuinely different equity mixes — 145 vs 141
    resolved at [-0.133, -0.070]. What it cannot separate is portfolios differing
    only in bond weight, since it is risk-adjusted and bonds move return and
    downside deviation together. The note must fire per-comparison, not always."""
    assert report_retirement._straddles_zero((-0.02, 0.04)) is True
    assert report_retirement._straddles_zero((-0.133, -0.070)) is False
    assert report_retirement._straddles_zero((0.01, 0.05)) is False
    assert report_retirement._straddles_zero((0.0, 0.05)) is True     # touching zero counts


def test_prompt_text_fits_a_narrow_terminal():
    """Long literals wrap with a hanging continuation and read as broken output.
    Hold every printed prompt line to 72 columns."""
    src = (ROOT / 'run.py').read_text()
    over = []
    for lit in re.findall(r'print\(f?"([^"]*)"', src):
        for piece in lit.split('\\n'):
            text = piece.replace('\\t', '    ')
            # f-string placeholders vary at runtime; measure the literal skeleton
            skeleton = re.sub(r'\{[^}]*\}', '', text)
            if len(skeleton) > 72:
                over.append(skeleton)
    assert not over, f"prompt lines over 72 cols: {over}"


def test_planning_horizon_reaches_a_high_age_percentile():
    """Planning to life EXPECTANCY means a coin-flip chance of outliving the
    plan. The errors are asymmetric — dying early with money left is fine,
    running out at 92 is not — so the horizon targets a high percentile."""
    cfg = runner.default_config()
    end = cfg['age'] + cfg['time_horizon_years']
    assert end >= 95, f"horizon ends at {end}, too short for planning"
    assert end - cfg['retirement_age'] >= 30, "retirement phase too short to test"


def test_vpw_planning_age_is_beyond_the_simulation_horizon():
    """VPW amortises to its planning age, so that must sit past the horizon or
    the rate runs away as the endpoint closes — the 53%-at-85 bug."""
    cfg = runner.default_config()
    end_age = cfg['age'] + cfg['time_horizon_years']
    st = policies.PolicyState(
        w_initial=1.0, v_at_retirement=1.0, initial_rate=0.04, w_prior_year=1.0,
        retirement_age=65, end_age=end_age, assumed_real_return=0.035,
        vpw_planning_age=policies.vpw_planning_age(end_age))
    assert st.vpw_planning_age > st.end_age, (st.vpw_planning_age, st.end_age)
    # and the rate stays sane at the far end of the horizon
    assert policies.vpw(st.end_age, 1.0, st) < 0.25


def test_custom_schedule_sleeve_is_applied_to_bare_refs():
    """Choosing a sleeve on the custom path must reach refs typed without one."""
    P = _P()
    a = glide.resolve_ref('140+10:bnd_only', P)
    b = glide.resolve_ref('140+10', P)
    assert a['BND'] > b['BND'], 'bnd_only should concentrate into BND'
    assert a.get('JPLD', 0) == 0 and b.get('JPLD', 0) > 0


def test_policy_multiselect_supports_a_subset():
    """One-or-all made the common case — comparing exactly two policies —
    inexpressible. estimate_minutes must scale with however many are picked."""
    base = {'mode': 'retirement', 'portfolios': ['141'], 'spend': 'solve'}
    two = dict(base, compare_policies=['guyton_klinger', 'vpw'])
    four = dict(base, compare_policies=list(runner.POLICY_ORDER))
    t1 = runner.estimate_minutes(3, base)                     # guyton_klinger
    vpw_only = runner.estimate_minutes(3, dict(base, compare_policies=['vpw']))
    # priced per policy, because vpw solves in closed form and the others search
    assert abs(runner.estimate_minutes(3, two) - (t1 + vpw_only)) < 1e-9
    assert abs(runner.estimate_minutes(3, four) - (3 * t1 + vpw_only)) < 1e-9
    assert runner.estimate_minutes(3, four) > runner.estimate_minutes(3, two)
    for p in two['compare_policies']:
        assert p in policies.POLICIES, p


# --- crisis severity (A2 / A4) ------------------------------------------------------
def test_crisis_shock_defaults_to_gaussian():
    """Off by default, so the shipped model and the golden master are unchanged."""
    np.random.seed(0)
    got = [pa.crisis_shock(3) for _ in range(5)]
    np.random.seed(0)
    want = [np.random.randn(3) for _ in range(5)]
    assert all(np.array_equal(a, b) for a, b in zip(got, want))
    assert pa.CRISIS_VOL_SCALE == 1.0 and pa.TAIL_DF is None


def test_student_t_shocks_keep_unit_variance_but_fatten_the_tail():
    """tail_df changes the SHAPE, not the magnitude — crisis size is
    CRISIS_VOL_SCALE's job. Conflating them would make the two knobs
    uninterpretable together."""
    np.random.seed(7)
    t = np.array([pa.crisis_shock(1, 5.0)[0] for _ in range(40000)])
    assert abs(t.std() - 1.0) < 0.05, t.std()
    kurt = (t ** 4).mean() / t.var() ** 2
    assert kurt > 5.0, f"t(5) should be leptokurtic, got {kurt:.1f}"
    assert np.mean(np.abs(t) > 4) > 10 * 0.0000633, "4-sigma months must be commoner"


def test_tail_shock_is_a_common_factor_not_independent_surprises():
    """One chi-square draw is SHARED across assets, so a fat month hits
    everything at once — tail dependence a Gaussian copula cannot produce."""
    np.random.seed(3)
    extreme_together = 0
    for _ in range(4000):
        v = pa.crisis_shock(6, 4.0)
        if np.abs(v).min() > 1.5:        # every asset extreme in the same month
            extreme_together += 1
    np.random.seed(3)
    indep = sum(1 for _ in range(4000) if np.abs(np.random.randn(6)).min() > 1.5)
    assert extreme_together > indep, (extreme_together, indep)


def test_crisis_vol_scale_lifts_only_the_high_regime():
    """A2 is about the crisis regime being too mild; calm regimes are fine."""
    code = ("import sys; sys.path.insert(0, %r); from montecarlo import engine; "
            "v = engine.VOLATILITY_REGIMES['VOO']; "
            "print(v['low'], v['medium'], v['high'])" % str(ROOT))
    base = pa.VOLATILITY_REGIMES['VOO']
    env = dict(os.environ, MC_CRISIS_VOL_SCALE='2.0')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, env=env).stdout.split()
    lo, med, hi = (float(x) for x in out)
    assert abs(lo - base['low']) < 1e-12 and abs(med - base['medium']) < 1e-12
    assert abs(hi - 2.0 * base['high']) < 1e-12, (hi, base['high'])
    # NOT "2008 levels" — that was a daily-vs-monthly units error. Monthly-sampled
    # S&P vol peaked at 25.9% (2020) and 24.7% (2008-09); the model's 20.1% high
    # regime is well calibrated, so x2 is a stress test at roughly double anything
    # in the record.
    assert hi > 0.35, f"x2 should roughly double the high regime, got {hi:.1%}"
    assert base['high'] < 0.30, (
        "the shipped high regime should sit near historical monthly crisis vol "
        f"(2008 21.0%, 2020 25.9%), got {base['high']:.1%}")


def test_sensitivity_dimensions_change_what_they_claim():
    """apply_dim must touch the field it names and nothing else, or a sweep
    silently measures the wrong thing."""
    import sensitivity
    import copy
    base = runner.build_config({'mode': 'retirement', 'portfolios': ['141'],
                                'spend': 'solve'})

    c = sensitivity.apply_dim(copy.deepcopy(base), 'retirement_age', 70)
    assert c['retirement_age'] == 70 and c['withdrawal']['start_age'] == 70, (
        "retirement age must move the withdrawal start too, or the sweep is a no-op")
    assert c['time_horizon_years'] == base['time_horizon_years']

    c = sensitivity.apply_dim(copy.deepcopy(base), 'horizon', 50)
    assert c['time_horizon_years'] == 50
    assert c['retirement_age'] == base['retirement_age']

    c = sensitivity.apply_dim(copy.deepcopy(base), 'contributions', 2.0)
    for k, a in c['contribution_schedule']['accounts'].items():
        want = base['contribution_schedule']['accounts'][k]['annual_amount'] * 2
        assert a['annual_amount'] == want, k

    # env-backed dims are applied before import, so apply_dim leaves config alone
    c = sensitivity.apply_dim(copy.deepcopy(base), 'inflation', 0.04)
    assert c == base
    try:
        sensitivity.apply_dim(copy.deepcopy(base), 'nonsense', 1)
    except SystemExit:
        pass
    else:
        raise AssertionError('unknown dimension should be rejected')


def test_inflation_is_env_overridable_in_a_worker():
    """Swept dims read at import must reach a fresh interpreter, as premium does."""
    code = ("import sys; sys.path.insert(0, %r); from montecarlo import engine; "
            "print(engine.INFLATION)" % str(ROOT))
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                         env=dict(os.environ, MC_INFLATION='0.045')).stdout.strip()
    assert abs(float(out) - 0.045) < 1e-12, out


# --- Social Security ----------------------------------------------------------------
def test_ss_claiming_schedule_matches_the_real_rules():
    """70% of PIA at 62, 100% at FRA, 124% at 70, no further gain after."""
    pia = 40_000.0
    assert abs(policies.ss_benefit(pia, 67) - pia) < 1e-9
    assert abs(policies.ss_benefit(pia, 62) / pia - 0.70) < 0.005
    assert abs(policies.ss_benefit(pia, 70) / pia - 1.24) < 0.005
    assert policies.ss_benefit(pia, 72) == policies.ss_benefit(pia, 70), 'capped at 70'
    ages = [62, 64, 66, 67, 68, 70]
    vals = [policies.ss_benefit(pia, a) for a in ages]
    assert vals == sorted(vals), 'benefit must rise monotonically with claiming age'


def test_ss_income_is_zero_before_claiming():
    cfg = {'annual_at_fra': 40_000.0, 'claim_age': 70}
    assert policies.ss_income(65, cfg) == 0.0
    assert policies.ss_income(69.9, cfg) == 0.0
    assert policies.ss_income(70, cfg) > 0
    assert policies.ss_income(80, None) == 0.0


def test_ss_puts_a_floor_under_spending_after_depletion():
    """The level effect is obvious; THIS is the one that could change decisions.
    The benefit arrives whether or not the portfolio survives, so depletion stops
    being catastrophic — and downside shape is where allocation choices are made."""
    P = _P()
    plan = glide.build_plan(glide.parse_schedule('141', 37), P)
    base = {'age': 65, 'retirement_age': 65, 'time_horizon_years': 30,
            'initial_value': 800_000.0, 'rebalancing_strategy': 'none',
            'contribution_schedule': {'frequency': 'annual', 'accounts': {
                'm': {'annual_amount': 0.0, 'growth_rate': 0.0, 'stop_age': None}}},
            'withdrawal': {'start_age': 65, 'policy': 'constant_real',
                           'sourcing': 'pro_rata', 'target_real_annual': 200_000.0,
                           'floor': 0.85}}
    without = pa.run_single_simulation((plan, base, 1))[4]['spending']
    with_ss = pa.run_single_simulation(
        (plan, dict(base, social_security={'annual_at_fra': 40_000.0,
                                           'claim_age': 67}), 1))[4]['spending']
    assert min(without) == 0.0, 'control must actually deplete to zero'
    assert min(with_ss[3:]) >= 39_999, f'benefit must floor spending: {min(with_ss[3:])}'


def test_ss_reduces_the_portfolio_draw_not_the_spending():
    """Spending is the exogenous need; SS changes who pays for it."""
    P = _P()
    plan = glide.build_plan(glide.parse_schedule('141', 37), P)
    cfg = {'age': 65, 'retirement_age': 65, 'time_horizon_years': 20,
           'initial_value': 5_000_000.0, 'rebalancing_strategy': 'none',
           'contribution_schedule': {'frequency': 'annual', 'accounts': {
               'm': {'annual_amount': 0.0, 'growth_rate': 0.0, 'stop_age': None}}},
           'withdrawal': {'start_age': 65, 'policy': 'constant_real',
                          'sourcing': 'pro_rata', 'target_real_annual': 150_000.0,
                          'floor': 0.85}}
    plain = pa.run_single_simulation((plan, cfg, 2))
    withss = pa.run_single_simulation(
        (plan, dict(cfg, social_security={'annual_at_fra': 40_000.0,
                                          'claim_age': 65}), 2))
    # same spending, bigger portfolio left over
    assert abs(plain[4]['spending'][5] - withss[4]['spending'][5]) < 1e-6
    assert withss[0] > plain[0], 'SS should leave more in the portfolio'


def test_no_ss_config_is_unchanged():
    P = _P()
    plan = glide.build_plan(glide.parse_schedule('141', 37), P)
    cfg = {'age': 65, 'retirement_age': 65, 'time_horizon_years': 20,
           'initial_value': 2_000_000.0, 'rebalancing_strategy': 'none',
           'contribution_schedule': {'frequency': 'annual', 'accounts': {
               'm': {'annual_amount': 0.0, 'growth_rate': 0.0, 'stop_age': None}}},
           'withdrawal': {'start_age': 65, 'policy': 'guyton_klinger',
                          'sourcing': 'excess', 'target_real_annual': 100_000.0,
                          'floor': 0.85}}
    a = pa.run_single_simulation((plan, cfg, 4))
    b = pa.run_single_simulation((plan, dict(cfg, social_security=None), 4))
    assert a[0] == b[0] and a[4]['spending'] == b[4]['spending']


def test_ss_benefit_scale_models_scheduled_vs_payable():
    """Scheduled benefits are a promise the system does not currently fund. The
    trustees project ~75-80% payable from the mid-2030s absent legislation, well
    before a 37-year-old would claim."""
    cfg = {'annual_at_fra': 40_000.0, 'claim_age': 67}
    assert policies.ss_income(70, cfg) == 40_000.0, 'default is scheduled in full'
    assert policies.ss_income(70, dict(cfg, benefit_scale=0.8)) == 32_000.0
    assert policies.ss_income(70, dict(cfg, benefit_scale=0.0)) == 0.0
    # a haircut arriving mid-retirement
    mid = dict(cfg, benefit_scale=0.8, scale_from_age=75)
    assert policies.ss_income(74, mid) == 40_000.0
    assert policies.ss_income(75, mid) == 32_000.0
    # and it still respects the claiming age
    assert policies.ss_income(60, dict(cfg, benefit_scale=0.8)) == 0.0


def test_ss_scale_is_a_sweep_not_a_random_draw():
    """Drawing the shortfall per path would average it away and make the plan
    look safer than a pessimistic-but-plausible policy outcome warrants — the
    same reason mortality is not drawn per path."""
    import sensitivity, copy
    base = runner.build_config({'mode': 'retirement', 'portfolios': ['141'],
                                'spend': 'solve',
                                'social_security': {'annual_at_fra': 40_000.0,
                                                    'claim_age': 67}})
    c = sensitivity.apply_dim(copy.deepcopy(base), 'ss_scale', 0.8)
    assert c['social_security']['benefit_scale'] == 0.8
    assert c['social_security']['annual_at_fra'] == 40_000.0
    # deterministic: the same config gives the same benefit every call
    vals = {policies.ss_income(70, c['social_security']) for _ in range(5)}
    assert len(vals) == 1, vals


def test_vpw_bequest_dials_between_spend_down_and_preservation():
    """VPW out-spends guyton_klinger because it amortises the WHOLE balance.
    Reserving an amount moves it continuously toward a preservation rule, which
    is the middle ground between the two policies."""
    def st(b):
        return policies.PolicyState(200_000., 5e6, 0.04, 200_000., 65, 95,
                                    0.035, 100.0, b)
    draws = [policies.vpw(70, 6e6, st(b)) for b in (0, 2e6, 5e6, 10e6)]
    assert draws == sorted(draws, reverse=True), draws
    assert draws[0] > 300_000, 'plain VPW should spend hard'
    assert policies.vpw(70, 6e6, st(20e6)) == 0.0, 'reserve above the balance -> nothing'
    # default is unchanged
    assert policies.vpw(70, 6e6, st(0)) == policies.vpw(
        70, 6e6, policies.PolicyState(200_000., 5e6, 0.04, 200_000., 65, 95, 0.035))


def test_vpw_at_a_zero_assumed_return():
    """The r = 0 branch (a clean 1/n) had never run. The payout must be 1/n
    of the balance and continuous with a tiny r either side."""
    def st(r, b=0.0):
        return policies.PolicyState(200_000., 5e6, 0.04, 200_000., 65, 95,
                                    r, 100.0, b)
    flat = policies.vpw(70, 3e6, st(0.0))
    assert flat == 3e6 / 30, flat
    for r in (1e-7, -1e-7):
        assert abs(policies.vpw(70, 3e6, st(r)) - flat) < 1.0, r


def test_vpw_reserve_is_discounted_not_held_back_whole():
    """The reserve is a target for the END of the horizon. Holding back its full
    size from day one would over-reserve badly in the early years."""
    def st(b):
        return policies.PolicyState(200_000., 5e6, 0.04, 200_000., 65, 95,
                                    0.035, 100.0, b)
    early = policies.vpw(66, 6e6, st(3e6))     # 34 years of discounting
    late = policies.vpw(94, 6e6, st(3e6))      # 6 years
    assert early > 0, 'a distant target must not zero out early spending'
    naive_rate = policies.vpw(66, 6e6 - 3e6, st(0)) / (6e6 - 3e6)
    assert early / 6e6 > naive_rate * 0.5, 'discounting must soften the reserve'
    assert late < policies.vpw(94, 6e6, st(0)), 'near the end it bites properly'


def test_money_input_accepts_how_people_write_it():
    """Demanding raw digits in a prompt that is otherwise all menus is a small
    cruelty, and '5000000' is hard to check at a glance."""
    assert runner.parse_money('40k') == 40_000
    assert runner.parse_money('1.5m') == 1_500_000
    assert runner.parse_money('$40,000') == 40_000
    assert runner.parse_money('  5M ') == 5_000_000
    assert runner.parse_money('40000') == 40_000
    for bad in ('', '   ', 'abc', '-5k', '5x'):
        try:
            runner.parse_money(bad)
        except ValueError:
            continue
        raise AssertionError(f'should have rejected {bad!r}')
    assert runner.fmt_money(5e6) == '$5M' and runner.fmt_money(40e3) == '$40k'
    assert runner.fmt_money(7.5e6) == '$7.5M'


def test_bequest_choices_span_spend_all_to_preservation():
    """A menu because this is a preference, not a number you look up."""
    ch = runner.VPW_RESERVE_CHOICES
    assert ch[0] == 0.0, 'spend-it-all must be the first option'
    assert list(ch) == sorted(ch), ch
    assert ch[-1] >= 20e6, 'top option should approach a preservation rule'

    def st(b):
        return policies.PolicyState(200_000., 5e6, 0.04, 200_000., 65, 95,
                                    0.035, 100.0, b)
    draws = [policies.vpw(70, 30e6, st(b)) for b in ch]
    assert draws == sorted(draws, reverse=True), draws
    assert len(set(draws)) == len(draws), 'each choice must be distinguishable'


def test_ss_monthly_vs_annual_is_a_12x_trap():
    """ssa.gov statements quote a MONTHLY benefit. Asking for an annual figure
    invites an error that looks entirely plausible — $5,000 is a realistic
    monthly benefit and an unrealistic annual one, and nothing downstream flags
    it. A 12x error in an input that puts a floor under retirement spending."""
    monthly = 5_000.0
    as_annual_by_mistake = monthly                      # what the old prompt took
    correct = monthly * 12
    cfg_wrong = {'annual_at_fra': as_annual_by_mistake, 'claim_age': 67}
    cfg_right = {'annual_at_fra': correct, 'claim_age': 67}
    assert policies.ss_income(70, cfg_right) - policies.ss_income(70, cfg_wrong) > 50_000


def test_vpw_reserve_menu_is_multiselect_like_the_policy_prompt():
    """Seeing the frontier is the point of the reserve, so picking several must
    work the same way it does for policies."""
    assert callable(runner.ask_multi)
    ch = runner.VPW_RESERVE_CHOICES
    assert len(ch) >= 3, 'need enough points for a frontier'


def test_the_vpw_reserve_prompt_turns_picks_into_the_keys_policy_runs_reads():
    """The keys decide how many solves run: 2+ reserves fan out through
    `compare_vpw_bequests`, exactly one positive one is `vpw_bequest`, and
    'no reserve' alone sets neither. The prompt's own post-logic was never run."""
    def drive(picks, typed=()):
        spec = {}
        real = runner.ask_multi, runner.ask
        runner.ask_multi = lambda *a, **k: list(picks)
        runner.ask = _scripted(typed)
        try:
            with _temp_investor(None), contextlib.redirect_stdout(io.StringIO()):
                runner.ask_vpw_bequest(spec)
        finally:
            runner.ask_multi, runner.ask = real
        return spec

    five, ten = str(5e6), str(10e6)
    spec = drive(['0', five, '__other__'], ['7.5m'])
    assert spec == {'compare_vpw_bequests': [0.0, 5e6, 7.5e6]}, spec
    assert [t for t, _, _ in runner.policy_runs(dict(spec, mode='retirement', policy='vpw'))] \
        == ['vpw no reserve', 'vpw reserve 5M', 'vpw reserve 7.5M']

    assert drive(['0', ten]) == {'compare_vpw_bequests': [0.0, 10e6]}, 'two is a comparison'
    assert drive([ten]) == {'vpw_bequest': 10e6}
    assert drive(['0']) == {}, 'no reserve alone sets neither key'
    # one positive reserve reached two ways is still ONE reserve
    assert drive([five, '__other__'], ['5m']) == {'vpw_bequest': 5e6}
    # blank amount: nothing added; a bad one (text, negative) is re-asked
    assert drive(['__other__'], ['']) == {}
    assert drive(['__other__'], ['abc', '-5k', '250k']) == {'vpw_bequest': 250e3}

    for picks, typed in ((['0', five, '__other__'], ['7.5m']), ([ten], ())):
        spec = dict(drive(picks, typed), portfolios=['141'], policy='vpw')
        assert runner.parse_spec(json.dumps(spec), {'141': {}})['policy'] == 'vpw'


def test_vpw_never_liquidates_in_the_last_year_of_any_plan():
    """VPW amortised to a fixed 100. When plan_to_age moved to 100
    (2026-10-09) the schedule ended on the horizon, so the last simulated year
    withdrew the whole balance — the 53%-at-85 bug again, through the config.
    The planning age must sit past the horizon whatever the horizon is, and
    must stay 100 for a plan to 95 so the figures quoted before are unmoved."""
    for end in (90.0, 95.0, 100.0, 105.0):
        assert policies.vpw_planning_age(end) > end, end
    assert policies.vpw_planning_age(95.0) == policies.VPW_PLANNING_AGE
    with _engine_state():
        _zero_vol_engine(0.0)
        for years in (30, 35):                       # plan to 95, then to 100
            cfg = _solver_cfg(years=years)
            cfg['withdrawal'] = dict(cfg['withdrawal'], policy='vpw')
            final, _, _, _, wd = pa.run_single_simulation(({'Z': 1.0}, cfg, 0))
            last = wd['spending'][-1]
            share = last / (last + final)
            assert share < 0.35, (
                f'plan to {65 + years}: the last year drew {share:.0%} of what was left')


def test_vpw_planning_age_is_one_constant_everywhere():
    """The runner's explanatory copy and the policy's arithmetic must read the
    same planning age. They drifted apart once: the prompt promised "runs to
    zero by 95" and "end with $5M at 95" while `vpw` amortised to 100, so every
    run left roughly the unamortised tail (~$6M) more than the label claimed and
    the bequest column looked broken. The model was right; the copy was wrong.

    Pinning the constant is the fix — a literal in either place can drift again.
    """
    assert policies.VPW_PLANNING_AGE == policies.PolicyState._field_defaults[
        'vpw_planning_age']
    # engine must default to the shared constant, not its own literal
    src = inspect.getsource(pa)
    assert "vpw_planning_age', 100.0)" not in src, \
        'engine still carries its own planning-age literal'
    # and the runner's copy must not name a different age
    copy = inspect.getsource(runner.ask_vpw_bequest)
    assert 'policies_mod.vpw_planning_age(' in copy, \
        'runner copy must read the planning age, not hardcode one'


def test_vpw_reserve_is_a_dial_not_an_end_balance_promise():
    """Even with NO reserve, VPW leaves a balance at the horizon, because it
    amortises to 100 and the simulation stops earlier. So a menu labelled
    'end with $X at <horizon>' is false for every option, not just some.
    """
    end_age, plan_age = 95.0, policies.VPW_PLANNING_AGE
    assert plan_age > end_age, 'the whole effect depends on this'
    st = policies.PolicyState(
        w_initial=100_000.0, v_at_retirement=2e6, initial_rate=0.05,
        w_prior_year=100_000.0, retirement_age=65.0, end_age=end_age,
        assumed_real_return=0.035, vpw_bequest=0.0)
    bal = 5e6
    draw = policies.vpw(end_age, bal, st)
    # at the horizon the schedule still has (plan_age - end_age) years to run,
    # so most of the balance is deliberately left standing
    assert draw < bal * 0.30, f'expected a partial draw at {end_age}, got {draw/bal:.0%}'
    assert bal - draw > 0.60 * bal, 'a "spend it all" label would be wrong here'


def test_time_estimate_counts_vpw_reserves_not_just_policies():
    """The estimate priced `compare_policies` only. Comparing 2 policies across
    3 VPW reserves is 6 runs by that count and 12 in fact, so an 11-minute run
    was announced as 3. Both now read run.policy_runs."""
    spec = {'mode': 'retirement', 'spend': 'solve',
            'compare_policies': ['guyton_klinger', 'vpw'],
            'compare_vpw_bequests': [0.0, 5e6, 10e6]}
    assert len(runner.policy_runs(spec)) == 4, 'GK + 3 VPW reserves'
    one = dict(spec, compare_vpw_bequests=None)
    with_reserves = runner.estimate_minutes(3, spec)
    without = runner.estimate_minutes(3, one)
    vpw_solve = runner.estimate_minutes(3, {'mode': 'retirement',
                                            'spend': 'solve',
                                            'compare_policies': ['vpw']})
    # two EXTRA vpw runs, each priced as the closed form it actually uses
    assert abs(with_reserves - (without + 2 * vpw_solve)) < 1e-9
    assert with_reserves > without, 'must scale with reserves'


def test_policy_run_fanout_has_one_definition():
    """The estimate, the confirm preview, the analysis header and the table all
    have to agree about what is being run. They didn't: the header named a
    single policy above a four-policy table."""
    src = inspect.getsource(report_retirement)
    assert '_run.policy_runs(spec)' in src, 'report must reuse the fan-out'
    assert 'vpw keep ' not in src, 'stale end-balance label'
    spec = {'mode': 'retirement', 'compare_policies': ['vpw'],
            'compare_vpw_bequests': [0.0, 5e6]}
    tags = [t for t, _, _ in runner.policy_runs(spec)]
    assert tags == ['vpw no reserve', 'vpw reserve 5M'], tags


def test_robustness_never_hides_a_warning_silently():
    """It listed `sorted(fragile.items())[:6]` — alphabetical, cut at six, no
    indication anything was dropped. On the 4-policy run that printed 6 of 10
    and swallowed every `vpw no reserve` row: the highest-spending option, and
    the one whose fragility mattered most."""
    grid = solver.FLOOR_GRID
    specs = ['140', '140>140+5@65', '140>140+10@65']
    # four policies, ten fragile — the shape of the run that exposed this
    rates = {('guyton_klinger', 0): 0.845, ('guyton_klinger', 1): 0.853,
             ('guyton_klinger', 2): 0.853, ('vpw no reserve', 0): 0.857,
             ('vpw no reserve', 1): 0.879, ('vpw no reserve', 2): 0.867,
             ('vpw reserve 5M', 1): 0.893, ('vpw reserve 10M', 0): 0.885}
    by_policy = {}
    for pol in ('guyton_klinger', 'vpw no reserve', 'vpw reserve 5M',
                'vpw reserve 10M'):
        by_policy[pol] = {}
        for i, g in enumerate(specs):
            hi = rates.get((pol, i), 0.95)
            # `hi` is the rate at the SOLVE floor: these must be genuine misses
            # of the certified confidence, not the arithmetic consequence of a
            # stricter floor, which robustness() now separates out.
            by_policy[pol][g] = {'floor': grid[1],
                                 'success_by_floor': {grid[0]: 0.95, grid[1]: hi,
                                                      grid[2]: hi - 0.02}}
    # one result fragile at the MIDDLE floor — strictly worse, must sort first
    by_policy['vpw reserve 5M'][specs[0]] = {
        'floor': grid[1],
        'success_by_floor': {grid[0]: 0.80, grid[1]: 0.80, grid[2]: 0.78}}

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_retirement.robustness(by_policy, specs)
    out = buf.getvalue()
    listed = [l for l in out.splitlines() if l.lstrip().startswith('⚠')]
    fragile_n = sum(1 for pol in by_policy for g in specs
                    if min(by_policy[pol][g]['success_by_floor'][f]
                           for f in (grid[0], grid[1])) < 0.90)
    assert f"of {len(by_policy) * len(specs)} came in below" in out, 'must state the total'
    assert str(fragile_n) in out.split('came in below')[0], \
        'must state how many missed their target'
    assert len(listed) == fragile_n or '… and' in out, \
        'a cap must announce what it hid'
    # worst first: the middle-floor failure leads
    assert 'vpw reserve 5M/140:' in listed[0], listed[0]
    # and every no-reserve row is present, not swallowed by an alphabetical cut
    assert sum('vpw no reserve' in l for l in listed) == 3


def test_policy_tradeoff_notes_survive_tagged_labels():
    """Keyed on the full tag, the block vanished the moment reserves fanned rows
    out to "vpw reserve 5M". Silent suppression, not a visible error."""
    src = inspect.getsource(report_retirement._policy_table)
    assert "startswith(k + ' ')" in src, 'must match on the base policy'
    tags = ['guyton_klinger', 'vpw no reserve', 'vpw reserve 5M']
    tradeoff = {'guyton_klinger': 'a', 'vpw': 'b', 'constant_real': 'c'}
    seen, notes = set(), []
    for tag in tags:
        base = next((k for k in tradeoff if tag == k or tag.startswith(k + ' ')), None)
        if base and base not in seen:
            seen.add(base); notes.append(tradeoff[base])
    assert notes == ['a', 'b'], notes   # deduped, and vpw is found


def test_life_stage_bands_partition_the_retirement_span():
    """Bands are thirds of the span, not hardcoded decades: retirement_age is a
    config field the sensitivity sweep varies (60/65/70), and a fixed
    65-74/75-84/85-94 grid would mislabel every other case. The final spending
    year is start+n-1, because withdrawals run start_age..end_age-1."""
    for start, end, n, want in [(65, 95, 30, ['65-74', '75-84', '85-94']),
                                (60, 95, 35, ['60-71', '72-82', '83-94']),
                                (70, 95, 25, ['70-77', '78-86', '87-94'])]:
        b = solver.stage_bounds(start, end, n)
        assert [lab for _, _, lab in b] == want, (start, b)
        assert sum(hi - lo for lo, hi, _ in b) == n, 'must partition exactly'
        assert b[0][0] == 0 and b[-1][1] == n, 'must cover the whole span'


def test_life_stage_bands_clip_to_the_array():
    """A horizon shorter than the nominal span must not index past the end."""
    b = solver.stage_bounds(65, 95, 12)
    assert b[-1][1] <= 12, b


def test_life_stage_stats_separate_flat_from_rising():
    """The point of the table: one scalar cannot compare a flat trajectory with a
    rising one. Measured on portfolio 140, guyton_klinger ran 197k/314k/503k at
    the median against vpw's 423k/632k/960k, with vpw ahead in the p10 of every
    band too — the opposite of the order the headline column implied."""
    bounds = solver.stage_bounds(65, 95, 30)
    flat = np.full((400, 30), 200_000.0)
    st = solver.stage_stats(flat, bounds)
    assert all(abs(m - p) < 1e-6 for _, m, p in st), 'flat: p10 == median'
    assert len({round(m) for _, m, _ in st}) == 1, 'flat: no ramp'
    rising = np.cumprod(np.full((400, 30), 1.05), axis=1) * 100_000
    st2 = solver.stage_stats(rising, bounds)
    assert st2[0][1] < st2[1][1] < st2[2][1], 'rising: must show the ramp'


def test_headline_column_is_not_called_sustainable():
    """It is the level the solver certified, not a forecast: realised spending
    runs above it in the median and below in bad cases, by margins that differ
    per policy (p10 of the first band over the bar: 100% constant_real, 86% gk,
    67% vpw, 47% vpw+5M). Calling it "sustainable" invited exactly the ranking
    error the life-stage table exists to prevent."""
    for fn in (report_retirement._policy_table, report_retirement._flat_table):
        src = inspect.getsource(fn)
        assert "'sustainable' if solving" not in src, fn.__name__
    assert persistence.metric_label('spend') == 'plan level'


def test_life_stage_table_is_ruled_between_bands():
    """Unruled, the eye reads "202k 449k" as a pair when they belong to
    different decades. Every row must carry the same number of dividers as the
    header, or a column has silently drifted."""
    bounds = solver.stage_bounds(65, 95, 30)
    sp = np.tile(np.linspace(200_000, 600_000, 30), (200, 1))
    rows = [('gk', '140', {'spending': sp, 'median_bequest': 3e7,
                           'start_age': 65})]
    cfg = {'age': 37, 'time_horizon_years': 58, 'retirement_age': 65}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_retirement._spending_shape(rows, cfg, {}, policy_width=8)
    lines = [l for l in buf.getvalue().splitlines() if '\u2502' in l]
    assert lines, 'no dividers rendered'
    counts = {l.count('\u2502') for l in lines}
    # one after the lifetime anchor, one per band boundary, one before estate
    assert counts == {len(bounds) + 1}, counts
    # header, sub-header and the data row must all be ruled alike
    assert len(lines) >= 3, lines


def test_self_directed_policy_sets_its_own_opening_draw():
    """vpw's withdrawal is rate(years to planning age) x spendable; it reads
    neither w_initial nor initial_rate. Seeding year 1 from the plan's target
    therefore gave it one year it did not choose, and made the reported first
    year neither vpw's draw nor a level it sustains.

    The three controlled policies must be unaffected: constant_real spends the
    target, fixed_pct turns it into a rate, guyton_klinger sets guardrails off
    that rate."""
    assert policies.SELF_DIRECTED == frozenset({'vpw'})
    for name in policies.POLICIES:
        assert (name in policies.SELF_DIRECTED) == (name == 'vpw'), name

    # a self-directed policy must read only fields available before the target
    # is known — otherwise the engine cannot call it to seed year 1
    src = inspect.getsource(policies.vpw)
    for forbidden in ('w_initial', 'initial_rate', 'w_prior_year'):
        assert f'state.{forbidden}' not in src, forbidden

    st = policies.PolicyState(
        w_initial=1.0, v_at_retirement=7.3e6, initial_rate=0.0,
        w_prior_year=1.0, retirement_age=65.0, end_age=95.0,
        assumed_real_return=0.035, vpw_bequest=0.0)
    own = policies.vpw(65.0, 7.3e6, st)
    assert own / 7.3e6 > 0.045, 'vpw opens near 5% of balance, not the target'
    # and the draw must not depend on the placeholder w_initial it was given
    st2 = st._replace(w_initial=999_999.0, initial_rate=0.9, w_prior_year=1e6)
    assert abs(policies.vpw(65.0, 7.3e6, st2) - own) < 1e-9


def test_engine_asks_whether_a_policy_is_self_directed():
    """Declared as data in policies.SELF_DIRECTED, not by naming 'vpw' in the
    engine, so adding a policy that sets its own draw needs no engine edit."""
    src = inspect.getsource(pa)
    assert 'SELF_DIRECTED' in src, 'engine must consult the registry'
    assert "== 'vpw'" not in src, 'engine must not special-case a policy by name'


def _adequacy_cfg():
    """Minimal config for the adequacy helpers (distinct from _wd_cfg above,
    which builds an ENGINE config for simulation)."""
    return {'age': 37, 'time_horizon_years': 58,
            'withdrawal': {'start_age': 65, 'policy': 'vpw', 'floor': 0.85}}


def test_adequacy_per_band_is_the_worst_stretch_not_the_lifetime_mean():
    """A lifetime mean is shape-blind: a bad first decade can be repaid by a rich
    ninth, and measurably was — the share of PASSING paths short in the first
    band ran 0.0% constant_real, 3.8% guyton_klinger, 8.5% vpw, 12.5% vpw+5M,
    monotone in back-loading."""
    cfg = _adequacy_cfg()
    # a path that averages well but collapses in the middle third
    sp = np.concatenate([np.full((1, 10), 300e3), np.full((1, 10), 50e3),
                         np.full((1, 10), 400e3)], axis=1)
    life = solver.adequacy_statistic(sp, cfg, 'lifetime')[0]
    band = solver.adequacy_statistic(sp, cfg, 'per_band')[0]
    assert abs(life - 250e3) < 1, life
    assert abs(band - 50e3) < 1, band
    assert band < life, 'per_band must see the collapse the mean averages away'


def test_adequacy_per_band_leaves_a_flat_policy_untouched():
    """constant_real spends the target every year, so every band equals the
    lifetime mean and the criterion cannot move it. Measured end to end: 212k
    under both, while the variable policies dropped 28-58%."""
    cfg = _adequacy_cfg()
    flat = np.full((50, 30), 200e3)
    a = solver.adequacy_statistic(flat, cfg, 'lifetime')
    b = solver.adequacy_statistic(flat, cfg, 'per_band')
    assert np.allclose(a, b), 'a flat trajectory must score identically'


def test_adequacy_rejects_an_unknown_mode():
    try:
        solver.adequacy_statistic(np.ones((2, 30)), _adequacy_cfg(), 'whatever')
    except ValueError:
        return
    raise AssertionError('an unknown adequacy mode must not pass silently')


# --- window adequacy (N11, 2026-10-21) --------------------------------------
# Thirds of the retirement made the rule's strictness a function of the span:
# 165 certified $123.0k planning to 100 and $120.05k to 95 on paths whose first
# 30 years were bit-identical, and retiring at 60 read ~4pp better than it was.

def _thirds_reference(spending, start_age, end_age):
    """The per-band statistic exactly as it stood before 'window' existed."""
    bounds = solver.stage_bounds(start_age, end_age, spending.shape[1])
    per_band = np.stack([spending[:, lo:hi].mean(axis=1) for lo, hi, _ in bounds])
    return per_band.min(axis=0)


def _cfg_to(plan_to, retire=65):
    return {'age': 37, 'time_horizon_years': plan_to - 37,
            'withdrawal': {'start_age': retire, 'policy': 'guyton_klinger',
                           'floor': 0.85}}


def test_window_adequacy_is_the_default_and_its_length_is_named():
    assert solver.DEFAULT_ADEQUACY == 'window', solver.DEFAULT_ADEQUACY
    assert solver.ADEQUACY_WINDOW_YEARS == 10, solver.ADEQUACY_WINDOW_YEARS
    assert runner.default_config()['adequacy'] == 'window'
    assert set(solver.ADEQUACY) == {'lifetime', 'thirds', 'window'}, solver.ADEQUACY


def test_window_adequacy_is_the_worst_rolling_ten_year_average():
    """A ten-year dip that straddles two thirds is split by them and averaged
    with good years on both sides; the window sees all of it."""
    cfg = _cfg_to(95)
    sp = np.full((1, 30), 300e3)
    sp[0, 5:15] = 50e3                       # 70-79: across the 75 boundary
    win = solver.adequacy_statistic(sp, cfg, 'window')[0]
    thirds = solver.adequacy_statistic(sp, cfg, 'thirds')[0]
    assert abs(win - 50e3) < 1e-6, win
    assert abs(thirds - 175e3) < 1e-6, thirds
    # and against brute force on arbitrary paths
    rng = np.random.default_rng(7)
    sp = rng.uniform(50e3, 400e3, size=(40, 30))
    brute = np.array([min(sp[p, i:i + 10].mean() for i in range(21))
                      for p in range(40)])
    assert np.allclose(solver.adequacy_statistic(sp, cfg, 'window'), brute)


def test_window_adequacy_never_certifies_more_for_a_longer_plan():
    """The horizon does not move a guyton_klinger path, only how much of it is
    scored, so a longer plan scored on the same paths must certify AT MOST as
    much. Every window of the 30-year plan is a window of the 35-year one."""
    rng = np.random.default_rng(11)
    sp100 = rng.uniform(50e3, 400e3, size=(500, 35))
    sp95 = sp100[:, :30]                     # the prefix: what planning to 95 sees
    w95 = solver.adequacy_statistic(sp95, _cfg_to(95), 'window')
    w100 = solver.adequacy_statistic(sp100, _cfg_to(100), 'window')
    assert (w100 <= w95 + 1e-9).all(), 'a longer plan certified more on some path'
    q95, q100 = np.percentile(w95, 10), np.percentile(w100, 10)
    assert q100 <= q95, (q100, q95)
    # the fixture can see the defect it exists for: thirds break it
    t95 = solver.adequacy_statistic(sp95, _cfg_to(95), 'thirds')
    t100 = solver.adequacy_statistic(sp100, _cfg_to(100), 'thirds')
    assert (t100 > t95 + 1e-9).any(), 'fixture too weak to tell the rules apart'


def test_window_adequacy_is_equally_strict_at_every_retirement_age():
    """Retire at 60 and the span is 35 years, at 70 it is 25; thirds gave bands
    of 11-12 and 8-9 years. The same ten bad years must score the same."""
    def path(retire):
        n = 95 - retire
        sp = np.full((1, n), 300e3)
        sp[0, :10] = 80e3                    # the first ten years are bad
        return sp
    w60 = solver.adequacy_statistic(path(60), _cfg_to(95, 60), 'window')[0]
    w70 = solver.adequacy_statistic(path(70), _cfg_to(95, 70), 'window')[0]
    assert abs(w60 - 80e3) < 1e-6 and abs(w70 - 80e3) < 1e-6, (w60, w70)
    t60 = solver.adequacy_statistic(path(60), _cfg_to(95, 60), 'thirds')[0]
    t70 = solver.adequacy_statistic(path(70), _cfg_to(95, 70), 'thirds')[0]
    assert t60 > t70 + 1e3, ('thirds must differ here, or the fixture is idle',
                             t60, t70)


def test_window_adequacy_scores_a_short_span_as_one_window():
    """Under ten years there is no ten-year window: the whole span is one."""
    rng = np.random.default_rng(3)
    for retire, n in ((88, 7), (86, 9), (85, 10)):
        sp = rng.uniform(50e3, 400e3, size=(20, n))
        got = solver.adequacy_statistic(sp, _cfg_to(95, retire), 'window')
        assert np.allclose(got, sp.mean(axis=1)), (retire, n)
    # clipped to the array, as the thirds are, when fewer years come back
    sp = rng.uniform(50e3, 400e3, size=(20, 6))
    got = solver.adequacy_statistic(sp, _cfg_to(95), 'window')
    assert np.allclose(got, sp.mean(axis=1))


def test_thirds_and_lifetime_score_exactly_as_before():
    """Old runs replay: selecting 'thirds' (or its old name 'per_band') or
    'lifetime' must give the statistic bit for bit."""
    rng = np.random.default_rng(5)
    for plan_to, retire in ((95, 65), (100, 65), (95, 60), (95, 70), (95, 88)):
        n = plan_to - retire
        sp = rng.uniform(50e3, 400e3, size=(200, n))
        cfg = _cfg_to(plan_to, retire)
        want = _thirds_reference(sp, retire, plan_to)
        assert np.array_equal(solver.adequacy_statistic(sp, cfg, 'thirds'), want)
        assert np.array_equal(solver.adequacy_statistic(sp, cfg, 'per_band'), want)
        assert np.array_equal(solver.adequacy_statistic(sp, cfg, 'lifetime'),
                              sp.mean(axis=1))
    assert solver.adequacy_mode('per_band') == 'thirds'


def test_portfolio_share_reads_the_binding_window():
    """The share must come from the stretch that set the level. Social Security
    alone funds years 0-9 (the portfolio funded nothing there); the window
    covering exactly those years binds, so the share is zero."""
    cfg = _cfg_to(95)
    cfg['social_security'] = {'annual_at_fra': 60_000.0, 'claim_age': 65.0,
                              'benefit_scale': 1.0}
    ss = solver.guaranteed_income_row(cfg, 30)
    sp = np.full((1, 30), 300e3)
    sp[0, :10] = ss[:10]                     # SS only, for the first ten years
    share = solver.portfolio_share_of_binding_band(sp, cfg, 'window')
    assert share[0] == 0.0, share
    sp2 = np.full((1, 30), 300e3)
    sp2[0, 3:13] = ss[3:13]                  # a window thirds would split
    assert solver.portfolio_share_of_binding_band(sp2, cfg, 'window')[0] == 0.0
    assert solver.portfolio_share_of_binding_band(sp2, cfg, 'thirds')[0] > 0.0


def test_binding_band_reports_the_binding_windows_start_age():
    cfg = dict(_diag_cfg(), adequacy='window')
    n = 50
    sp = np.full(30, 300e3)
    sp[7:17] = 100e3                         # the worst stretch starts at 72
    raw = {'spending': np.tile(sp, (n, 1)), 'depleted': np.zeros(n, dtype=bool),
           'depletion_ages': [None] * n}
    b = diagnostics.binding_band(raw, cfg, 0.85, 200e3)
    assert b['mode'] == 'window', b
    assert b['start_age']['all'] == {10: 72.0, 50: 72.0, 90: 72.0}, b['start_age']
    assert b['start_age']['failing'][50] == 72.0, b['start_age']
    assert b['n_failing'] == n
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        diagnostics.report({'165': dict(raw, spend=200e3)}, cfg, 0.85)
    out = buf.getvalue()
    assert 'WHICH 10-YEAR STRETCH SET THE LEVEL' in out, out
    row = next(l for l in out.splitlines() if l.strip().startswith('165'))
    assert '72 / 72 / 72' in row, row


def test_provenance_records_the_window_rule_under_a_new_scoring_version():
    assert persistence.SCORING_VERSION == 4, persistence.SCORING_VERSION
    def prov(**kw):
        return persistence.provenance({**kw, 'withdrawal': {'floor': 0.85}})
    assert prov(adequacy='window')['adequacy'] == 'window'
    assert prov()['adequacy'] == 'window'
    # an old record's name and the new one are the same rule, not a change
    assert prov(adequacy='per_band')['adequacy'] == 'thirds'


def test_score_is_pure_and_evaluate_is_simulate_plus_score():
    """Split so a policy whose paths do not move with the target can be simulated
    ONCE and scored at the level the closed form dictates."""
    src = inspect.getsource(solver.evaluate)
    assert 'simulate(' in src and 'score(' in src, src
    # score must not simulate: same raw arrays, two spends, no engine call
    cfg = _adequacy_cfg()
    raw = {'spending': np.full((100, 30), 200e3),
           'depleted': np.zeros(100, dtype=bool),
           'finals': np.full(100, 1e6), 'depletion_ages': [None] * 100,
           'returns': None}
    lo = solver.score(raw, cfg, 100e3, 0.85, 'per_band')
    hi = solver.score(raw, cfg, 400e3, 0.85, 'per_band')
    assert lo['success_rate'] == 1.0 and hi['success_rate'] == 0.0
    assert lo['median_bequest'] == hi['median_bequest'], 'scoring must not resimulate'


def test_closed_form_applies_exactly_to_self_directed_policies():
    """vpw's paths do not move with the target, so success is
    P(stat >= floor x spend) and the answer is a quantile, not a search:
    spend* = quantile(stat, 1 - confidence) / floor.

    Measured against bisection on portfolio 140: $253.9k at exactly 90.0%
    confidence in 6.9s, versus $256.2k at 89.5% in 99.4s — 14.4x faster AND
    exact, because the quantile is taken on the verification sample rather than
    a smaller search sample."""
    cfg = dict(_adequacy_cfg())
    assert solver._self_directed(cfg) is True
    assert solver._self_directed(
        dict(cfg, withdrawal=dict(cfg['withdrawal'], policy='guyton_klinger'))) is False


def test_crossing_is_estimated_not_bisected():
    """Bisection compared a noisy sample against a bare threshold and moved a
    BOUND on the result, so every unlucky read was absorbed permanently and the
    answer came out low — measured -17% for guyton_klinger, -8% for fixed_pct
    against the true 90% level. Sampling the bracket and fitting averages the
    noise, and costs fewer evaluations than the search it replaces.

    Measured on the search sample's own crossing (the first spend verified),
    over 400 fixed noise draws of 1.5pp — a 400-path read — on a curve crossing
    at $187k: exact without noise, unbiased with it (mean -$54), mean miss
    $6.5k. Interpolating the straddling pair reads +$1.6k / $7.9k, taking the
    nearest probe -$1.9k / $9.7k, and skipping the refinement pass $7.4k."""
    truth = lambda x: 0.90 - (x - 187e3) * 0.9e-6
    _, checked, _ = _stub_solve(truth, truth)
    assert abs(checked[0] - 187e3) < 500, checked
    miss = []
    for seed in range(400):
        noisy = lambda x, seed=seed: truth(x) + 0.015 * np.random.default_rng(
            (seed, int(round(x)))).standard_normal()
        _, checked, _ = _stub_solve(noisy, truth)
        miss.append(checked[0] - 187e3)
    miss = np.array(miss)
    assert abs(miss.mean()) < 800, f'biased: {miss.mean():+,.0f}'
    assert np.abs(miss).mean() < 7_000, f'noisy: {np.abs(miss).mean():,.0f}'


def test_isotonic_imposes_the_shape_we_know_holds():
    """Success is non-increasing in spend; a noisy estimate need not be. Pooling
    adjacent violators averages the noise rather than letting one low read stand,
    which is exactly what bisection could not do."""
    fit = solver._isotonic_decreasing([0.95, 0.88, 0.91, 0.80])
    assert all(fit[i] >= fit[i + 1] - 1e-12 for i in range(len(fit) - 1)), fit
    assert abs(fit[1] - fit[2]) < 1e-12, 'the violating pair must pool'
    assert abs(fit[1] - 0.895) < 1e-9, fit          # mean of 0.88 and 0.91
    assert abs(sum(fit) - (0.95 + 0.88 + 0.91 + 0.80)) < 1e-9, 'mass preserved'
    already = [0.95, 0.90, 0.85]
    assert np.allclose(solver._isotonic_decreasing(already), already)


def test_seed_offset_is_inert_at_zero_and_independent_otherwise():
    """The default must change nothing; a non-zero offset must change everything.

    seed_offset exists so the solver's answer can be checked on paths it was not
    fitted to (tools/solver_calibration.py). Both halves matter: if offset 0 ever
    stopped reproducing, every stored run and every paired bootstrap in this repo
    would be silently invalidated, and if a non-zero offset did NOT draw fresh
    paths the calibration check would quietly compare a sample with itself.
    """
    with _engine_state():
        cfg = _solver_cfg()
        _zero_vol_engine(0.0)
        # The zero-vol fixture is deterministic by construction, so give it real
        # dispersion — otherwise every seed produces the same path and the
        # independence half of this test passes for the wrong reason.
        pa.VOLATILITY_REGIMES = {'Z': {'low': 0.16, 'medium': 0.16, 'high': 0.16}}
        base = solver._paths({'Z': 1.0}, cfg, 6, parallel=False)
        same = solver._paths({'Z': 1.0}, cfg, 6, parallel=False, seed_offset=0)
        assert [r[0] for r in base] == [r[0] for r in same], 'offset 0 must reproduce'

        other = solver._paths({'Z': 1.0}, cfg, 6, parallel=False, seed_offset=500)
        a, b = [r[0] for r in base], [r[0] for r in other]
        assert not any(x == y for x, y in zip(a, b)), 'offset must draw fresh paths'

        again = solver._paths({'Z': 1.0}, cfg, 6, parallel=False, seed_offset=500)
        assert [r[0] for r in other] == [r[0] for r in again], 'must stay reproducible'


def test_solver_reports_whether_it_converged():
    """A level that did not land must say so rather than look like an answer.

    The Newton loop exits on four conditions and only one is success. Portfolio
    102's curve is steep enough that its solved level still moves with sample
    size, so a caller needs to be able to see that the number is not settled.
    """

    def stuck(plan, config, spend, n_sims=500, floor=0.85, parallel=True,
              full=False, adequacy=solver.DEFAULT_ADEQUACY, seed_offset=0):
        # A curve that never reaches 90% anywhere: the best it does is 80%.
        rate = max(0.0, min(0.80, 0.80 - max(0.0, spend - 100_000.0) * 1e-6))
        return {'spend': float(spend), 'success_rate': rate,
                'success_by_floor': {}, 'depletion_rate': 0.0}

    real = solver.evaluate
    try:
        solver.evaluate = stuck
        got = solver.solve_sustainable_spend(
            {'Z': 1.0}, _solver_cfg(), confidence=0.90, floor=0.85,
            search_sims=100, verify_sims=100, n_points=6, parallel=False)
    finally:
        solver.evaluate = real
    assert got['converged'] is False, got
    assert abs(got['gap']) > 0.01, got


def test_rf_source_resolves_and_rejects_nonsense():
    """The cash rate is the biggest level lever; where it came from must be exact.

    Three routes with a fixed precedence — an explicit MC_RF_REAL, a TIPS quote,
    or the historical default — and a typo must fail loudly rather than fall
    through to a default that silently changes every expected return by more
    than a percentage point.

    SUBPROCESSES, not importlib.reload. engine binds `factors` at import, so
    reloading it mid-suite leaves engine holding stale scalars and makes later
    tests order-dependent — which made this suite intermittently fail under
    validate_all while passing standalone. A fresh interpreter is also what
    production does: every MC_ var is read at import, and workers re-import
    under `spawn`.
    """

    code = ("import sys; sys.path.insert(0, %r); from montecarlo import factors; import json; "
            "print(json.dumps([factors.RF_REAL, factors.RF_REAL_AS_OF, "
            "factors.RF_FORWARD, factors._RF_REAL_DEFAULT]))" % str(ROOT))
    base = {k: v for k, v in os.environ.items()
            if k not in ('MC_RF_REAL', 'MC_RF_SOURCE')}
    cases = {'default': {}, 'rate': {'MC_RF_REAL': '0.02'},
             'rate+source': {'MC_RF_REAL': '0.02', 'MC_RF_SOURCE': 'tips30'},
             'tips30': {'MC_RF_SOURCE': 'tips30'},
             'tips30_avg': {'MC_RF_SOURCE': 'tips30_avg'},
             'tips99': {'MC_RF_SOURCE': 'tips99'}, 'bogus': {'MC_RF_SOURCE': 'bogus'}}
    runs = dict(zip(cases, _children([(code, dict(base, **env))
                                      for env in cases.values()], cwd=str(ROOT))))

    def resolve(case):
        out = runs[case]
        assert out.returncode == 0, (case, out.stderr[-400:])
        return json.loads(out.stdout.strip().splitlines()[-1])

    rf, as_of, fwd, default = resolve('default')
    assert abs(rf - default) < 1e-12, rf
    assert as_of is None
    assert abs(resolve('rate')[0] - 0.02) < 1e-12
    # An explicit rate WINS over a source, so a sweep cannot be overridden by a
    # stale environment variable left set somewhere else.
    assert abs(resolve('rate+source')[0] - 0.02) < 1e-12

    spot, as_of, fwd, _ = resolve('tips30')
    assert 0.0 < spot < 0.10 and as_of, (spot, as_of)
    assert resolve('tips30_avg')[0] != spot, '_avg must differ'
    # The nominal must stay DERIVED from the real rate and inflation.
    assert abs(fwd - ((1 + spot) * (1 + 0.03) - 1)) < 1e-9, (spot, fwd)

    for bad in ('tips99', 'bogus'):
        assert runs[bad].returncode != 0, f'{bad} must raise'


def test_premia_tables_agree_on_their_keys():
    """PREMIA, PREMIUM_SE and PREMIUM_WINDOW must cover the same cells.

    They are three hand-maintained tables that are read together — a factor
    present in one and missing from another would either raise deep inside a
    build-up or, worse, silently contribute nothing. PREMIUM_WINDOW was added
    2026-09-11 precisely because the three windows are NOT equal, which makes
    drift between them easy and invisible.
    """
    for region in factors.PREMIA:
        a = set(factors.PREMIA[region])
        b = set(factors.PREMIUM_SE.get(region, {}))
        c = set(factors.PREMIUM_WINDOW.get(region, {}))
        assert a == b == c, (region, a ^ b, a ^ c)
        for f, (months, start) in factors.PREMIUM_WINDOW[region].items():
            assert months > 24 and len(start) == 7, (region, f, months, start)


def test_regime_vols_are_ordered_and_the_clamp_is_only_a_guard():
    """classify_regimes must return low <= medium <= high, from real buckets.

    The ordering clamp exists for a thin bucket in a near-constant series. It
    fires on 0 of 25 production tickers, so it is a guard — but it CAN silently
    flatten two regimes into one, which would make the regime machinery a no-op
    for that ticker, so both halves are pinned: ordered output, and untouched
    values when the raw buckets are already ordered.
    """
    import data_collection_regime_model as dc

    idx = pd.date_range('1990-01-31', periods=400, freq='ME')
    rng = np.random.default_rng(3)
    # Genuinely regime-switching: a calm half and a wild half.
    r = pd.Series(np.concatenate([rng.normal(0, 0.02, 200),
                                  rng.normal(0, 0.07, 200)]), index=idx)
    labels, vols = dc.classify_regimes(r)
    assert vols['low'] <= vols['medium'] <= vols['high'], vols
    assert vols['high'] > 1.5 * vols['low'], vols      # the buckets must separate
    # ...and the values must be the buckets' OWN realised vol, not a clamp.
    for g in ('low', 'medium', 'high'):
        own = r.loc[labels[labels == g].index].std() * np.sqrt(12)
        assert abs(vols[g] - own) < 1e-9, (g, vols[g], own)

    # A constant series cannot separate; it must still come back ordered and
    # finite rather than NaN.
    flat = pd.Series(np.full(400, 0.01), index=idx)
    _, v2 = dc.classify_regimes(flat)
    assert all(np.isfinite(x) for x in v2.values()), v2
    assert v2['low'] <= v2['medium'] <= v2['high'], v2


def test_market_regime_labels_and_regime_correlations_on_synthetic_data():
    """The market regime decides which months count as a crisis and every
    regime correlation matrix, and both were inline in main(), reachable only by
    a network regen. CLAUDE.md quotes results from them (VOO-VEA 0.654 ->
    0.914), and their history includes a dropna that let the newest ticker cut
    the window from 180 months to 136."""
    import data_collection_regime_model as dc
    idx = pd.date_range('1990-01-31', periods=240, freq='ME')
    rng = np.random.default_rng(5)
    mkt = pd.Series(rng.normal(0, 0.04, 240) * np.repeat([0.5, 1.0, 2.0, 1.0], 60),
                    index=idx)
    labels, _, (lo, hi) = dc.vol_regime_labels(mkt, 12)
    assert lo < hi and len(labels) == 240 - 11
    share = labels.value_counts(normalize=True)
    p30, p70 = dc.REGIME_CUTS
    for r, expect in zip(dc.REGIMES, (p30, p70 - p30, 1 - p70)):
        assert abs(share[r] - expect) < 0.02, share
    assert (labels.loc['2000-06':'2004-12'] == 'high').mean() > 0.8, 'the wild stretch'

    # Four funds: B tracks A, C starts late, D has 30 months — enough for the
    # baseline, too few once split by regime — and E has 20, too few for both.
    a = rng.normal(0, 0.04, 240)
    df = pd.DataFrame({'A': a, 'B': 0.8 * a + rng.normal(0, 0.02, 240),
                       'C': rng.normal(0, 0.04, 240), 'D': rng.normal(0, 0.04, 240),
                       'E': rng.normal(0, 0.04, 240)}, index=idx)
    df.loc[df.index[:120], 'C'] = np.nan
    df.loc[df.index[:210], 'D'] = np.nan
    df.loc[df.index[:220], 'E'] = np.nan
    with contextlib.redirect_stdout(io.StringIO()):
        base, by, months = dc.regime_correlations(df, labels)
    # A late starter shortens only its own pairs.
    assert abs(base[0, 1] - df['A'].corr(df['B'])) < 1e-9, 'the A-B window was cut'
    assert abs(base[0, 2] - df['A'].corr(df['C'])) < 1e-9
    assert abs(base[0, 3] - df['A'].corr(df['D'])) < 1e-9 and base[0, 3] != 0.0
    assert base[0, 4] == 0.0 and base[4, 4] == 1.0, 'a thin pair takes identity'
    assert sum(months.values()) == len(labels)
    thin = 0
    for r, C in by.items():
        assert np.allclose(np.diag(C), 1.0) and np.linalg.eigvalsh(C).min() > -1e-10
        own = df.loc[labels[labels == r].index]
        if own['D'].notna().sum() < dc.MIN_PAIR_MONTHS:
            thin += 1
            assert abs(C[0, 3] - base[0, 3]) < 1e-9, f'{r}: a thin pair takes the baseline entry'
        assert abs(C[0, 1] - own['A'].corr(own['B'])) < 1e-9, r
    assert thin, 'the fixture must leave D thin in some regime'
    # A regime of 10 months or fewer takes the whole baseline matrix.
    few = labels.copy()
    few[:] = 'medium'
    few.iloc[:10] = 'low'
    with contextlib.redirect_stdout(io.StringIO()):
        base2, by, _ = dc.regime_correlations(df, few)
    assert by['low'] is base2 and by['high'] is base2 and by['medium'] is not base2


def test_a_proxy_cannot_contribute_more_than_the_history_it_feeds():
    """The invariant that caught the rolling-window bug.

    AVEE reported 308 contributed months into a 240-month history, because the
    count was taken before ROLLING_WINDOW_TICKERS truncate the spliced series.
    Impossible by construction, and nothing noticed until the numbers were put
    beside each other.
    """
    # `from montecarlo import engine` at the top of this module already needs this file.
    f = Path(__file__).resolve().parent.parent / 'output' / 'ticker_parameters_regimes_v8.json'
    data = json.loads(f.read_text())
    scores = (data.get('metadata') or {}).get('proxy_scores') or {}
    for t, v in scores.items():
        hist = (data.get(t) or {}).get('history_months')
        if hist is None:
            continue
        assert v.get('contributed_months', 0) <= hist, (t, v['contributed_months'], hist)


def _smoke(mode, _capture=None, _figures=None, **extra):
    """Drive run.py's real entry points end to end, tiny, and capture the output.

    Everything here is the path a real run takes: build_config -> resolve the
    strategies -> report_retirement.run -> the tables. The pieces are unit-tested
    individually; this is the only thing that checks they are still WIRED
    TOGETHER. It patches nothing except sizes and the figure/persistence
    side-effects. Pass a list as `_figures` to draw the figures for real, into
    a temp dir, and get their paths back in it.
    """

    P = _P()
    spec = {'mode': mode, 'portfolios': ['141', '145'], **extra}
    cfg = runner.build_config(spec)
    # Small enough to run inside the gate; large enough that every branch fires.
    cfg = dict(cfg, time_horizon_years=12, age=60, retirement_age=64)
    if 'withdrawal' in cfg:
        cfg['withdrawal'] = dict(cfg['withdrawal'], start_age=64)
    specs, is_grid = runner.resolve_strategies(spec, P)

    real_paths, real_save = solver._paths, persistence.save_run

    def _no_save(*a, **k):
        raise RuntimeError('smoke run — deliberately not persisted')

    figs, persisted, real_eval = report_retirement.figures, None, report_retirement.evaluate_strategies
    buf = io.StringIO()
    try:
        if _capture is not None:
            def _keep(*a, **k):
                out = real_eval(*a, **k)
                _capture.append(out)
                return out
            report_retirement.evaluate_strategies = _keep
        # Shrink the work at the ONE choke point that generates paths, rather
        # than stubbing the maths. Everything above it — bracketing, the crossing
        # estimator, the Newton pass, the policies, the renderers — runs for
        # real on real parameters; there is just very little of it. Patching
        # solver.evaluate instead does not work: it is called with six
        # positional arguments in places, so a keyword-shaped wrapper breaks.
        def _tiny(plan, config, n_sims, parallel, light=True, seed_offset=0):
            return real_paths(plan, config, min(int(n_sims), 24), False,
                              light=light, seed_offset=seed_offset)
        solver._paths = _tiny
        tmp = tempfile.TemporaryDirectory()

        def _drawn(real):
            def draw(*a, **k):
                paths = list(real(*a, **dict(k, outdir=tmp.name)))
                _figures.extend(paths)
                return paths
            return draw

        report_retirement.figures = _drawn(figs) if _figures is not None else (lambda *a, **k: [])
        # A smoke run is NOT a run. report_retirement.run persists every run it
        # completes, so without this the gate wrote two 24-path records into
        # output/runs on every invocation and `run.py --history` filled up with
        # garbage that looked exactly like real results.
        persistence.save_run = _no_save
        persisted = report_retirement.accumulation_figures
        report_retirement.accumulation_figures = (_drawn(persisted) if _figures is not None
                                   else (lambda *a, **k: []))
        with contextlib.redirect_stdout(buf):
            report_retirement.run(spec, specs, P, cfg, is_grid=is_grid)
        if _figures is not None:
            _figures[:] = [(q, q.exists()) for q in _figures]
        tmp.cleanup()
    finally:
        solver._paths = real_paths
        persistence.save_run = real_save
        report_retirement.evaluate_strategies = real_eval
        report_retirement.figures = figs
        if persisted is not None:
            report_retirement.accumulation_figures = persisted
        solver.shutdown_pool()
    return buf.getvalue()


def test_the_figures_draw_for_real_and_name_the_real_horizon():
    """Every smoke run stubbed the figures out, so nothing ever drew one: a
    matplotlib change or a bad key in the reduced stats would pass the gate and
    fail in a user's run. Figure 3's title also said '50-Year' on every plan,
    whatever its horizon — the bug the drawdown title was fixed for."""
    from montecarlo import report
    titles, real_save = [], report.save_figures

    def keep(figures, *a, **k):
        titles.extend(f._suptitle.get_text() if f._suptitle else '' for f in figures)
        return real_save(figures, *a, **k)

    for mode, extra in (('accumulation', {}),
                        ('retirement', {'spend': 'solve', 'policy': 'guyton_klinger'})):
        drawn = []
        report.save_figures = keep
        try:
            _smoke(mode, _figures=drawn, **extra)
        finally:
            report.save_figures = real_save
        assert drawn and all(ok for _, ok in drawn), (mode, drawn)
    assert len([t for t in titles if t]) >= 5, titles
    fig3 = [t for t in titles if t.startswith('Figure 3')]
    assert fig3 and all('12-Year' in t for t in fig3), fig3


def test_the_decumulation_figure_reads_the_one_plan_runs_the_tables_read():
    """The figure plotted each strategy at its OWN plan level while every table
    beside it read all of them at the first strategy's level: on 2026-10-26 the
    chart showed 103 spending ~$70k/yr where RESULTS said $149k. Across
    policies the figure keeps each run's own level, as the tables do."""
    import numpy as np
    own = np.full((4, 3), 50_000.0)
    one = np.full((4, 3), 100_000.0)
    rec = lambda: {'spend': 40_000.0, 'spending': own, 'depletion_ages': [None] * 4,
                   'start_age': 65,
                   'common': {'spend': 60_000.0, 'spending': one,
                              'depletion_ages': [None, None, None, 66], 'start_age': 65}}
    by_policy = {'guyton_klinger': {'A': rec(), 'B': rec()}}
    cfg = {'age': 37, 'time_horizon_years': 30}
    series, note = report_retirement._figure_data(by_policy, ['A', 'B'], cfg, one_plan=True)
    assert all(np.allclose(series[g]['median'], 100.0) for g in 'AB'), series
    assert series['A']['depleted'][-1] == 25.0, series['A']['depleted']
    assert "A's $60k/yr plan" in note, note
    series, note = report_retirement._figure_data(by_policy, ['A', 'B'], cfg, one_plan=False)
    assert all(np.allclose(series[g]['median'], 50.0) for g in 'AB'), series
    assert note == '', note

    # The wiring: a one-policy solved run draws at one plan; a policy
    # comparison draws each run at its own level.
    seen, real = [], report_retirement.figures
    report_retirement.figures = lambda *a, **k: seen.append(k.get('one_plan')) or []
    try:
        _smoke('retirement', _figures=[], spend='solve', policy='guyton_klinger')
        _smoke('retirement', _figures=[], spend='solve', policy='guyton_klinger',
               compare_policies=['guyton_klinger', 'vpw'])
    finally:
        report_retirement.figures = real
    assert seen == [True, False], seen


def test_a_retirement_run_survives_end_to_end():
    """The wiring from a config to a rendered decumulation report.

    check_logic covers evaluate_strategies by INSPECTING ITS SOURCE and never
    calling it, and nothing called report_retirement.run at all — so a refactor
    that broke the hand-off between run.py, solver, policies and the renderers
    passed every check. This is that gap.
    """
    kept = []
    out = _smoke('retirement', _capture=kept, spend='solve', policy='guyton_klinger')
    # The PRODUCER half of the one-plan hand-off: evaluate_strategies must hand
    # back the whole score, not four fields of it. The two tables agreeing with
    # each other cannot show this — with a thin `common` both fall back to
    # own-level runs and still agree (verified with a mutant).
    (res,), = [kept]
    for g in ('141', '145'):
        for k in ('median_realised', 'p10_realised', 'per_path_mean', 'spend'):
            assert k in res[g]['common'], (g, k, sorted(res[g]['common']))
    # STRUCTURE, not a digit count. A digit heuristic was tried first and is too
    # weak: emptying _flat_table's body still left enough digits elsewhere to
    # pass, which is exactly the silent-missing-table failure this exists to
    # catch. Name the sections instead.
    for section in ('RETIREMENT ANALYSIS', 'SOLVING', 'RESULTS',
                    'HEAD-TO-HEAD', 'SPENDING BY LIFE STAGE'):
        assert section in out, (section, out[:600])
    for token in ('141', '145', 'guyton_klinger'):
        assert token in out, (token, out[:400])
    # Every strategy must appear as a ROW of the results table, not merely
    # somewhere in the prose.
    body = out.split('RESULTS', 1)[1].split('HEAD-TO-HEAD', 1)[0]
    for g in ('141', '145'):
        assert any(l.strip().startswith(g) for l in body.splitlines()), (g, body)
    # solving two strategies: the head-to-head must be read at ONE plan
    assert 'COMPARING' in out and 'at one plan' in out, out.split('HEAD-TO-HEAD', 1)[-1][:500]
    # ...and so must RESULTS and the life-stage table, or the report describes
    # two sets of runs under one banner. Same runs, so the same lifetime figure.
    shape = out.split('SPENDING BY LIFE STAGE', 1)[1]
    assert "every strategy at 141's" in '\n'.join(shape.splitlines()[:4]), shape[:400]
    for g in ('141', '145'):
        r_row = next(l for l in body.splitlines() if l.strip().startswith(g))
        s_row = next(l for l in shape.splitlines() if l.strip().startswith(g))
        assert r_row.split()[1] == s_row.split()[1], (g, r_row, s_row)
    # ...and both must be the ONE-PLAN figure, not 145's own-level one
    want = f"{res['145']['common']['median_realised'] / 1e3:,.0f}k"
    r145 = next(l for l in body.splitlines() if l.strip().startswith('145'))
    assert want in r145, (want, r145)
    # window adequacy fails a path on ANY ten-year stretch, not on every one
    assert "any 10-year stretch's average" in out, out[:800]
    assert "EVERY 10-year stretch stays" in out, out[:800]
    assert 'Traceback' not in out
    assert 'IF THE RETURN MODEL IS WRONG' not in out, 'the check must be asked for'


def test_an_accumulation_run_survives_end_to_end():
    """Same, for the other mode — which takes a different path through the
    renderer (no withdrawal block, a rebalancing summary, different tables)."""
    out = _smoke('accumulation')
    for section in ('ALLOCATION ANALYSIS', 'RUNNING', 'REBALANCING'):
        assert section in out, (section, out[:600])
    for token in ('141', '145'):
        assert token in out, (token, out[:400])
    assert 'Traceback' not in out


def test_the_cash_rate_shock_is_shared_across_tickers():
    """One rf draw per path, common to every holding — not per-ticker noise.

    The whole point of item 9b: independent per-ticker bands largely cancel
    across 25 funds, so the input every asset SHARES has to move them together
    or it contributes nothing. Two tickers with zero idiosyncratic band must
    therefore move by exactly the same amount on a given path, and by a
    DIFFERENT amount on the next one.
    """
    with _engine_state():
        # Isolate the cash rate. The inflation shock is shared too and moves
        # real returns, so left on it makes the "disabled" paths below differ.
        # This passed for months only because three earlier tests leaked
        # INFLATION_STD = 0.0 into everything after them.
        pa.INFLATION_STD = 0.0
        pa.TICKERS = ['A', 'B']
        pa.EXPECTED_RETURNS = {'A': 0.05, 'B': 0.05}
        pa.PARAM_UNCERTAINTY_STD = {'A': 0.0, 'B': 0.0}   # no idiosyncratic band
        pa.VOLATILITY_REGIMES = {t: {r: 0.0 for r in ('low', 'medium', 'high')}
                                 for t in ('A', 'B')}
        stay = {r: {'low': 0.0, 'medium': 1.0, 'high': 0.0}
                for r in ('low', 'medium', 'high')}
        pa.REGIME_TRANSITIONS = {t: stay for t in ('A', 'B', pa.MARKET_REGIME_KEY)}
        one = np.array([[1.0, 0.0], [0.0, 1.0]])
        for m in ('BASELINE', 'LOW', 'MEDIUM', 'HIGH'):
            setattr(pa, f'CORRELATION_{m}', one)
        cfg = {'age': 40, 'retirement_age': 65, 'time_horizon_years': 1,
               'initial_value': 100_000.0, 'rebalancing_strategy': 'none',
               'contribution_schedule': {'frequency': 'annual', 'accounts': {}}}

        def growth(alloc, sim_id):
            out = pa.run_single_simulation((alloc, cfg, sim_id))
            return out[0] / cfg['initial_value']

        pa.RF_REAL_STD = 0.02
        # DISCRIMINATOR: variance across paths must not fall when the book holds
        # two assets instead of one. A shared shock hits both identically, so a
        # 50/50 book is exactly as uncertain as a single holding. Independent
        # per-ticker shocks would AVERAGE, cutting the variance roughly in half
        # — which is the very diversification that makes a per-ticker band the
        # wrong way to express an assumption every asset shares.
        #
        # Comparing two SINGLE-asset books cannot tell the two apart: each draws
        # one number at the same position in the stream either way. That was the
        # first version of this test and it passed against a deliberately
        # per-ticker implementation.
        one = [growth({'A': 1.0, 'B': 0.0}, i) for i in range(120)]
        both = [growth({'A': 0.5, 'B': 0.5}, i) for i in range(120)]
        sd1, sd2 = float(np.std(one)), float(np.std(both))
        assert sd1 > 1e-6, sd1
        # Shared -> ratio ~1.0. Independent -> two draws average, ratio ~1/sqrt(2)
        # = 0.707. The threshold has to sit between those, not ON one of them:
        # 0.70 was tried and let the per-ticker implementation through by a
        # thousandth.
        assert sd2 > 0.85 * sd1, (sd1, sd2, sd2 / sd1)

        # Disabled, every path is the deterministic 5%.
        pa.RF_REAL_STD = 0.0
        g = [growth({'A': 1.0, 'B': 0.0}, i) for i in range(3)]
        assert max(g) - min(g) < 1e-12, g
        assert abs(g[0] - 1.05) < 2e-3, g


def test_inflation_hits_bonds_harder_than_equities():
    """The inflation shock must be SHARED but not UNIFORM.

    A shock applied equally to every asset is just another level term,
    structurally identical to the cash-rate band and incapable of changing a
    decision. The content is the DIFFERENTIAL: nominal bonds lose more real
    return per point of inflation than equities do (-1.15 vs -0.80, measured
    1947-2026 net of the cash channel). So this pins three things — the shock
    exists, it is shared within an asset class, and a bond book is hurt MORE
    than an equity book by the same draw.
    """
    with _engine_state():
        pa.TICKERS = ['EQ', 'BD']
        pa.EXPECTED_RETURNS = {'EQ': 0.05, 'BD': 0.05}
        pa.PARAM_UNCERTAINTY_STD = {'EQ': 0.0, 'BD': 0.0}
        pa.VOLATILITY_REGIMES = {t: {r: 0.0 for r in ('low', 'medium', 'high')}
                                 for t in ('EQ', 'BD')}
        stay = {r: {'low': 0.0, 'medium': 1.0, 'high': 0.0}
                for r in ('low', 'medium', 'high')}
        pa.REGIME_TRANSITIONS = {t: stay for t in ('EQ', 'BD', pa.MARKET_REGIME_KEY)}
        one = np.array([[1.0, 0.0], [0.0, 1.0]])
        for m in ('BASELINE', 'LOW', 'MEDIUM', 'HIGH'):
            setattr(pa, f'CORRELATION_{m}', one)
        pa.RF_REAL_STD = 0.0                      # isolate inflation
        pa.INFLATION_STD = 0.02
        pa.INFL_BETA = {'EQ': -0.80, 'BD': -1.15}
        cfg = {'age': 40, 'retirement_age': 65, 'time_horizon_years': 1,
               'initial_value': 100_000.0, 'rebalancing_strategy': 'none',
               'contribution_schedule': {'frequency': 'annual', 'accounts': {}}}

        def g(alloc, i):
            return pa.run_single_simulation((alloc, cfg, i))[0] / 100_000.0

        eq = np.array([g({'EQ': 1.0, 'BD': 0.0}, i) for i in range(150)])
        bd = np.array([g({'EQ': 0.0, 'BD': 1.0}, i) for i in range(150)])
        # Shared: both move, on the same draws.
        assert np.std(eq) > 1e-6 and np.std(bd) > 1e-6, (np.std(eq), np.std(bd))
        # NOT uniform: the bond book swings ~1.15/0.80 = 1.44x as much.
        ratio = float(np.std(bd) / np.std(eq))
        assert 1.25 < ratio < 1.65, ratio
        # Perfectly correlated — it is ONE draw, not two.
        assert abs(np.corrcoef(eq, bd)[0, 1]) > 0.99, np.corrcoef(eq, bd)[0, 1]
        # ...and turning it off restores determinism.
        pa.INFLATION_STD = 0.0
        off = [g({'EQ': 1.0, 'BD': 0.0}, i) for i in range(3)]
        assert max(off) - min(off) < 1e-12, off


_TURNOVER_PROBE = (
    "import sys; sys.path.insert(0, %r); from montecarlo import engine as pa, glide; import run as runner; "
    "from montecarlo.loaders import load_portfolios_from_csv as L; P = L(verbose=False); "
    "cfg = runner.build_config({'mode': 'retirement', 'portfolios': ['166'], 'spend': 210000, "
    "'policy': 'guyton_klinger', 'gk_response': 'fast'}); "
    "plan = glide.build_plan(glide.parse_schedule('166', cfg['age']), P); "
    "print([repr(pa.run_single_simulation((plan, cfg, i))[3]['traded']) for i in range(12)])")


def _turnover_across_hash_seeds(root):
    return {r.stdout.strip() for r in _children(
        [(_TURNOVER_PROBE % str(root), dict(os.environ, PYTHONHASHSEED=str(seed)))
         for seed in (1, 2, 3)], cwd=str(root))}


def test_turnover_is_the_same_in_every_process():
    """traded_dollars summed over a SET, and Python randomises string hashing
    per process — so the same path reported a different turnover in a
    different run: 23 of 72 paths differed in the last digit between two runs
    of the same engine (2026-10-03). Twelve real paths, three hash seeds: the
    turnover must agree to the bit. (A small hand-built probe cannot show it —
    since Python 3.12 sum() over floats is compensated, which absorbs simple
    order effects; real paths still differ.)"""
    got = _turnover_across_hash_seeds(ROOT)
    assert len(got) == 1 and '' not in got, f"turnover differs across hash seeds: {got}"


def test_the_head_to_head_compares_every_strategy_at_one_plan():
    """Each strategy is SOLVED at its own level, and a level is only placed to
    its level_se — so comparing realised spending across levels mixes the
    portfolio's effect with the solver's aim. On 2026-10-03 that made 166 look
    ~1.5% ahead of 165; at one plan the gap was ~0.4% and did not resolve. When
    every strategy carries a common-plan run, the head-to-head must read IT,
    not the own-level paths."""
    rng = np.random.default_rng(3)
    shared = rng.lognormal(12.5, 0.5, 2000)
    res = {'A': {'spend': 160_000.0, 'per_path_mean': shared,
                 'common': {'spend': 160_000.0, 'per_path_mean': shared}},
           'B': {'spend': 168_000.0, 'per_path_mean': shared * 1.05,     # own level: +5%
                 'common': {'spend': 160_000.0, 'per_path_mean': shared.copy()}}}
    (g, lo, hi, verdict, unit), = report_retirement._h2h_rows(['A', 'B'], res, 'A', solving=True)
    assert unit == 'spend' and abs(lo) < 1e-6 and abs(hi) < 1e-6, (lo, hi)
    assert verdict == 'not resolved', verdict


def test_the_leader_lines_read_the_one_plan_runs_too():
    """'✓ most spending 166 $521k' printed two sections above a head-to-head
    that said 165 and 166 do not separate — the leader line still ranked on
    each strategy's OWN solved level, the same defect the head-to-head had. With
    common-plan runs it must rank on them, say so, and flag a leader that is
    not resolved against the base."""
    rng = np.random.default_rng(5)
    x = rng.lognormal(12.5, 0.5, 3000)
    res = {'A': {'spend': 160_000.0, 'median_realised': 500_000.0, 'p10_realised': 1.0,
                 'per_path_mean': x, 'common': {'spend': 160_000.0, 'per_path_mean': x * 1.002}},
           'B': {'spend': 168_000.0, 'median_realised': 530_000.0, 'p10_realised': 2.0,   # own level flatters B
                 'per_path_mean': x * 1.06, 'common': {'spend': 160_000.0, 'per_path_mean': x}}}
    lead = report_retirement._leaders(['A', 'B'], res, 'A', solving=True)
    assert lead['best'] == 'A' and lead['at_one_plan'], lead
    assert lead['best_value'] == float(np.median(x * 1.002))
    # B a hair ahead at one plan but NOISILY (path by path it wanders +-5%), so
    # the paired interval straddles zero: it leads and must carry the flag
    res['B']['common']['per_path_mean'] = x * (1.004 + rng.normal(0, 0.05, x.size))
    lead = report_retirement._leaders(['A', 'B'], res, 'A', solving=True)
    assert lead['best'] == 'B' and lead['unresolved'], lead
    # and a lead that is the same on every path resolves, so no flag
    res['B']['common']['per_path_mean'] = x * 1.004
    lead = report_retirement._leaders(['A', 'B'], res, 'A', solving=True)
    assert lead['best'] == 'B' and not lead['unresolved'], lead


def _scored(spending, spend, start_age=64):
    """A real score() of synthetic paths, so a renderer gets every field it reads."""
    cfg = {'age': 60, 'time_horizon_years': 12, 'retirement_age': 64,
           'withdrawal': {'start_age': start_age}}
    n = spending.shape[0]
    raw = {'spending': spending, 'depleted': np.zeros(n, bool),
           'finals': np.full(n, 1e6), 'depletion_ages': [None] * n,
           'returns': None}
    return solver.score(raw, cfg, spend, 0.85, 'per_band')


def test_the_results_table_reads_the_one_plan_runs():
    """RESULTS printed each strategy's OWN-level paths under a banner saying
    every strategy was being compared at A's level — so the table and the
    head-to-head below it described different runs. With one-plan runs on every
    strategy, RESULTS reads THEM and says which plan it is."""
    rng = np.random.default_rng(11)
    base = rng.lognormal(12.5, 0.2, (400, 8))
    a = _scored(base, 160_000.0)
    own_b = _scored(base * 1.30, 190_000.0)       # B's own, higher level
    one_b = _scored(base * 1.01, 160_000.0)       # B at A's plan
    a.update(converged=True, gap=0.0)
    own_b.update(converged=True, gap=0.0)
    res = {'A': dict(a, common=dict(a, depleted=np.zeros(400, bool))),
           'B': dict(own_b, common=dict(one_b, depleted=np.zeros(400, bool)))}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_retirement._flat_table(['A', 'B'], res, {'mode': 'retirement', 'spend': 'solve'}, 'A')
    out = buf.getvalue()
    body = out.split('RESULTS', 1)[1].split('WHY THE FAILING', 1)[0].split('✓', 1)[0]
    row = next(l for l in body.splitlines() if l.strip().startswith('B'))
    want = f"{one_b['median_realised'] / 1e3:,.0f}k"
    wrong = f"{own_b['median_realised'] / 1e3:,.0f}k"
    assert want in row and wrong not in row, (want, wrong, row)
    head = out.split('RESULTS', 1)[1].splitlines()[0]
    assert '160k' in head and 'A' in head, head


def test_the_life_stage_table_reads_the_one_plan_runs():
    """The same defect as RESULTS, one table down: the life-stage rows were
    each strategy's own level under the 'at A's level' banner. One policy reads
    the one-plan runs; several policies, or a plan choice, keep each row's own
    level, because those comparisons are about the level."""
    rng = np.random.default_rng(13)
    base = rng.lognormal(12.5, 0.2, (300, 8))
    a, own_b, one_b = (_scored(base, 160_000.0), _scored(base * 1.30, 190_000.0),
                       _scored(base * 1.01, 160_000.0))
    res = {'A': dict(a, common=dict(a)), 'B': dict(own_b, common=dict(one_b))}
    rows, note = report_retirement._shape_rows([('gk', 'guyton_klinger', {})], ['A', 'B'],
                                {'gk': res}, {})
    got = {g: r['median_realised'] for _, g, r in rows}
    assert got['B'] == one_b['median_realised'], (got, one_b['median_realised'])
    assert "A's $160k/yr plan" in note, note
    two = [('gk', 'guyton_klinger', {}), ('vpw', 'vpw', {})]
    rows, note = report_retirement._shape_rows(two, ['A', 'B'], {'gk': res, 'vpw': res}, {})
    assert all(r['median_realised'] == res[g]['median_realised'] for _, g, r in rows)
    assert note == '', note


def test_a_grid_run_keeps_each_row_at_its_own_level():
    """A solved grid prints IN DETAIL from each strategy's OWN-level run, so the
    life-stage table beside it must too — overlaying the one-plan runs there
    gave one report two 'lifetime average' figures per strategy, from different
    runs, the defect the flat table had."""
    base = np.random.default_rng(13).lognormal(12.5, 0.2, (300, 8))
    a, own_b, one_b = (_scored(base, 160_000.0), _scored(base * 1.30, 190_000.0),
                       _scored(base * 1.01, 160_000.0))
    res = {'A': dict(a, common=dict(a)), 'B': dict(own_b, common=dict(one_b))}
    runs = [('gk', 'guyton_klinger', {})]
    rows, note = report_retirement._shape_rows(runs, ['A', 'B'], {'gk': res}, {}, is_grid=True)
    got = {g: r['median_realised'] for _, g, r in rows}
    assert got['B'] == own_b['median_realised'] != one_b['median_realised'], got
    assert note == '', note
    rows, note = report_retirement._shape_rows(runs, ['A', 'B'], {'gk': res}, {})        # not a grid: unchanged
    assert {g: r['median_realised'] for _, g, r in rows}['B'] == one_b['median_realised']


def test_the_best_low_case_is_flagged_when_it_does_not_resolve():
    """'✓ best low case 165' was printed with no check at all, while 'most
    spending' beside it was tested against the base in the head-to-head. A low
    case that does not separate from the base must say so."""
    x = np.random.default_rng(7).lognormal(12.5, 0.5, 3000)

    def lead(b):
        res = {'A': {'spend': 160_000.0, 'per_path_mean': x * 1.02,
                     'common': {'spend': 160_000.0, 'per_path_mean': x * 1.02}},
               'B': {'spend': 160_000.0, 'per_path_mean': b,
                     'common': {'spend': 160_000.0, 'per_path_mean': b}}}
        return report_retirement._leaders(['A', 'B'], res, 'A', solving=True)

    # B's p10 a hair above A's, by noise: it leads the low case, unresolved
    noisy = x * 1.02 * (1.005 + np.random.default_rng(2).normal(0, 0.10, x.size))
    got = lead(noisy)
    assert got['best'] == 'A' and got['safest'] == 'B', got
    assert got['safest_unresolved'], got
    # a low case that is better on every path resolves
    got = lead(x * 1.05)
    assert got['safest'] == 'B' and not got['safest_unresolved'], got


def test_the_low_case_caveat_is_printed_only_when_it_applies():
    """The caveat is the user-visible half of the low-case fix, and the dict
    flag alone does not show it reaches the page: `if False:` in its place
    passed every other check. Render the table both ways. B is the safer
    strategy (a thinner lower tail) with a LOWER median, so A leads on
    spending and B is the only one with a '✓ best low case' line to qualify."""
    base = np.random.default_rng(7).lognormal(12.5, 0.2, (3000, 8))
    a_paths = _scored(base, 160_000.0)['per_path_mean']
    ma, pa = float(np.median(a_paths)), float(np.percentile(a_paths, 10))
    mp = base.mean(1, keepdims=True)
    thinner = (ma + 0.5 * (mp - ma)) * (base / mp)         # same median, tighter spread

    def render(b):
        a = _scored(base, 160_000.0)
        bb = _scored(b, 160_000.0)
        for r in (a, bb):
            r.update(converged=True, gap=0.0)
        res = {'A': dict(a, common=dict(a)), 'B': dict(bb, common=dict(bb))}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report_retirement._flat_table(['A', 'B'], res, {'mode': 'retirement', 'spend': 'solve'}, 'A')
        return buf.getvalue(), report_retirement._leaders(['A', 'B'], res, 'A', True)

    # path by path B wanders, and its p10 clears A's by 0.1% — a lead of noise
    noisy = thinner * np.random.default_rng(2).normal(1.0, 0.03, (3000, 1))
    noisy = noisy * (1.001 * pa / np.percentile(_scored(noisy, 160_000.0)['per_path_mean'], 10))
    out, lead = render(noisy)
    assert lead['best'] == 'A' and lead['safest'] == 'B', lead     # the fixture is what it claims
    assert 'not resolved against A: the paired low cases overlap' in out, out
    # a tail that is better on every path resolves: the line is there, the caveat is not
    out, lead = render(thinner * 0.98)
    assert lead['best'] == 'A' and lead['safest'] == 'B' and not lead['safest_unresolved'], lead
    assert '✓ best low case   B' in out and 'low cases overlap' not in out, out


def test_span_collapses_equal_ends_and_keeps_unequal_ones():
    assert report_retirement._span(0.83, 0.83) == '83.0%'
    assert report_retirement._span(0.801, 0.83) == '80.1%–83.0%'
    assert report_retirement._span(0.8301, 0.8299) == '83.0%'          # equal AT THE PRINTED PRECISION


def test_robustness_does_not_round_a_miss_up_to_the_target():
    """89.6% printed as '90%' on the row that says 'below 90%' — a flagged row
    contradicting itself. And a range whose ends are equal printed '83%–83%'."""
    near = {'spend': 1e5, 'floor': 0.85, 'worst_cut': 0.0,
            'success_by_floor': {0.7: 0.95, 0.85: 0.896, 1.0: 0.70}}
    far = {'spend': 1e5, 'floor': 0.85, 'worst_cut': 0.0,
           'success_by_floor': {0.7: 0.80, 0.85: 0.80, 1.0: 0.60}}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_retirement.robustness({'gk': {'a': near, 'b': far}}, ['a', 'b'],
                                     confidence=0.90)
    out = buf.getvalue()
    row = next(l for l in out.splitlines() if '⚠' in l and 'a:' in l)
    assert '89.6%' in row, row
    same = {'spend': 1e5, 'floor': 0.85, 'worst_cut': 0.0,
            'success_by_floor': {0.7: 0.95, 0.85: 0.83, 1.0: 0.60}}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_retirement.robustness({'gk': {'a': same, 'b': dict(same)}},
                                     ['a', 'b'], confidence=0.90)
    out = buf.getvalue()
    assert '83.0% success there' in out, out
    assert '–83' not in out, out


def test_the_strict_floor_is_not_described_as_never_cutting():
    """The 100% column is the share of paths whose every LIFE STAGE averages at
    least the plan. A year below plan inside a stage still passes it, so 'paths
    where you never have to cut at all' claimed more than it measures."""
    res = {'a': {'spend': 1e5, 'floor': 0.85, 'worst_cut': 0.0,
                 'success_by_floor': {0.7: 0.94, 0.85: 0.92, 1.0: 0.80}}}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_retirement.robustness({'gk': res}, ['a'], confidence=0.90)
    out = buf.getvalue()
    assert 'never have to cut' not in out and 'never take a cut' not in out, out
    # it names the stretch the floor actually scores: a window by default...
    assert '10-year stretch' in out.split('HOW TO USE THIS', 1)[1], out
    # ...and a life stage when the run was scored on thirds
    res['a']['adequacy'] = 'thirds'
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_retirement.robustness({'gk': res}, ['a'], confidence=0.90)
    out = buf.getvalue()
    assert 'life stage' in out.split('HOW TO USE THIS', 1)[1], out
    assert '10-year' not in out, out


def test_social_security_is_capitalised_at_the_retirement_age():
    """The block is headed 'at 65, before the first withdrawal' and sets the
    benefit beside the balance at 65 — but discounted it only to the CLAIM age,
    so a benefit claimed at 70 was overstated by five years of discounting."""
    res = {'160': {'spend': 200_000.0,
                   'balance_at_retirement': {10: 3.1e6, 50: 7.3e6, 90: 19.1e6}}}
    cfg = {'age': 37, 'time_horizon_years': 58, 'retirement_age': 65,
           'withdrawal': {'start_age': 65},
           'social_security': {'annual_at_fra': 60000.0, 'claim_age': 70.0,
                               'benefit_scale': 1.0}}
    annual = policies.ss_income(70.5, cfg['social_security'])
    pv = sum(annual / 1.03 ** (a - 65) for a in range(70, 95))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_retirement._plan_in_context(['160'], res, cfg)
    out = buf.getvalue()
    assert f"${pv / 1e6:.2f}M" in out, (pv, out)


def test_at_one_fixed_plan_the_head_to_head_compares_ruin():
    """Under fixed spending every surviving path spends exactly the plan, so at
    one plan the median lifetime spending is identical for every strategy and
    says nothing — the 2026-10-03 sleeve run printed +0.0k for all three. What
    differs is WHO RUNS OUT, so the comparison becomes the paired difference in
    P(run out), and fewer ruins is BETTER."""
    n = 2000
    flat = np.full(n, 160_000.0)
    dep_a = np.zeros(n, bool); dep_a[:400] = True          # 20% run out
    dep_b = np.zeros(n, bool); dep_b[:100] = True          # 5%, a subset
    res = {'A': {'spend': 160_000.0, 'per_path_mean': flat,
                 'common': {'spend': 160_000.0, 'per_path_mean': flat, 'depleted': dep_a}},
           'B': {'spend': 160_000.0, 'per_path_mean': flat,
                 'common': {'spend': 160_000.0, 'per_path_mean': flat, 'depleted': dep_b}}}
    (g, lo, hi, verdict, unit), = report_retirement._h2h_rows(['A', 'B'], res, 'A', solving=True)
    assert unit == 'ruin', unit
    assert hi < 0 and verdict == 'BETTER', (lo, hi, verdict)
    assert abs((lo + hi) / 2 - (-0.15)) < 0.02, (lo, hi)


def test_fixed_spending_verdicts_respect_the_solvers_own_precision():
    """Under constant_real every surviving path spends exactly its plan level,
    so the paired interval on realised spending collapses to the gap between
    two SOLVED levels — zero width — and printed BETTER / WORSE on +-0.3%
    (2026-10-03: four sleeves at $226-227k). Each level is only placed to its
    level_se, so a gap inside the two strategies' combined noise is not a
    verdict. A gap well outside it still is, and a flexible policy's verdict is
    left alone."""
    def r(level, se, flexible=False):
        per_path = (np.linspace(0.5, 2.0, 400) * level if flexible
                    else np.full(400, float(level)))
        return {'spend': float(level), 'level_se': float(se), 'per_path_mean': per_path}
    base = r(227_000, 5_000)
    assert report_retirement._h2h_verdict(r(226_000, 5_000), base, -700.0, -700.0, True) == 'within solver noise'
    assert report_retirement._h2h_verdict(r(190_000, 5_000), base, -37_000.0, -37_000.0, True) == 'WORSE'
    # flexible spending: the interval is real, so it decides
    assert report_retirement._h2h_verdict(r(226_000, 5_000, True), r(227_000, 5_000, True),
                           -900.0, -100.0, True) == 'WORSE'
    # a fixed TARGET (not solving) has no solver noise to respect
    assert report_retirement._h2h_verdict(r(226_000, 5_000), base, -700.0, -700.0, False) == 'WORSE'


def test_tips_keep_their_real_value_through_inflation():
    """A TIPS fund is a bond for every role — tier, sleeves, the premium scale
    — EXCEPT inflation, where a nominal bond loses real return to a surprise
    and a TIPS fund does not. That difference is the only reason to hold one,
    so it is pinned twice: STIP must be classified as inflation-linked and get
    the TIPS pass-through, not the bond default it would inherit from
    _BOND_FUNDS — and a TIPS book must ride out the inflation draw that moves a
    nominal bond book."""
    assert 'STIP' in loaders.INFLATION_LINKED
    assert set(loaders.INFLATION_LINKED) <= set(loaders._BOND_FUNDS) <= set(loaders.MODEL_TICKERS)
    assert pa.INFL_BETA['STIP'] == factors.INFL_BETA_TIPS != factors.INFL_BETA_BOND, (
        pa.INFL_BETA['STIP'], factors.INFL_BETA_TIPS, factors.INFL_BETA_BOND)
    assert pa.INFL_BETA['BND'] == factors.INFL_BETA_BOND

    with _engine_state():
        pa.TICKERS = ['BD', 'TP']
        pa.EXPECTED_RETURNS = {'BD': 0.02, 'TP': 0.02}
        pa.PARAM_UNCERTAINTY_STD = {'BD': 0.0, 'TP': 0.0}
        pa.VOLATILITY_REGIMES = {t: {r: 0.0 for r in ('low', 'medium', 'high')}
                                 for t in ('BD', 'TP')}
        stay = {r: {'low': 0.0, 'medium': 1.0, 'high': 0.0}
                for r in ('low', 'medium', 'high')}
        pa.REGIME_TRANSITIONS = {t: stay for t in ('BD', 'TP', pa.MARKET_REGIME_KEY)}
        for m in ('BASELINE', 'LOW', 'MEDIUM', 'HIGH'):
            setattr(pa, f'CORRELATION_{m}', np.eye(2))
        pa.RF_REAL_STD, pa.INFLATION_STD = 0.0, 0.02      # isolate inflation
        pa.INFL_BETA = {'BD': factors.INFL_BETA_BOND, 'TP': factors.INFL_BETA_TIPS}
        cfg = {'age': 40, 'retirement_age': 65, 'time_horizon_years': 1,
               'initial_value': 100_000.0, 'rebalancing_strategy': 'none',
               'contribution_schedule': {'frequency': 'annual', 'accounts': {}}}

        def g(alloc, i):
            return pa.run_single_simulation((alloc, cfg, i))[0] / 100_000.0
        bd = np.array([g({'BD': 1.0, 'TP': 0.0}, i) for i in range(80)])
        tp = np.array([g({'BD': 0.0, 'TP': 1.0}, i) for i in range(80)])
        assert np.std(bd) > 1e-3, 'the inflation draw never reached the nominal bond'
        assert np.std(tp) < 1e-12, f"TIPS moved with inflation: sd {np.std(tp):.2e}"


def test_a_cash_like_bond_fund_is_not_hit_by_inflation_like_BND():
    """The bond pass-through is excess over real CASH, so a fund that reprices
    like cash carries almost none of it. Until 2026-10-24 JPST and VUSB took
    BND's full -1.15, more than equities' -0.80 — inflation risk on the asset
    held to avoid it. Pinned: every nominal bond has a duration, the
    pass-through shrinks with it (ultra-short below equities, BSV and BND at
    the full figure), and the scaling is linear to zero below BSV."""
    nominal = set(loaders._BOND_FUNDS) - set(loaders.INFLATION_LINKED)
    assert set(loaders.BOND_DURATION) == nominal, (
        sorted(loaders.BOND_DURATION), sorted(nominal))

    b = pa.INFL_BETA
    full = factors.INFL_BETA_BOND
    assert b['BND'] == b['BSV'] == full, (b['BND'], b['BSV'], full)
    assert full < b['JPLD'] < b['VUSB'] < b['JPST'] < 0, {t: b[t] for t in nominal}
    for t in ('JPST', 'VUSB'):
        assert b[t] > pa.INFL_BETA_DEFAULT, f'{t} {b[t]} as exposed as equities'
    # the ratio, not just the order: half BSV's duration, half its exposure
    ref = factors.INFL_REF_DURATION
    assert factors.bond_infl_beta(0.0) == 0.0
    assert abs(factors.bond_infl_beta(ref / 2) - full / 2) < 1e-12
    assert factors.bond_infl_beta(3 * ref) == full
    assert abs(b['JPST'] / full - loaders.BOND_DURATION['JPST'] / ref) < 1e-12


def test_a_proxy_is_charged_back_its_fee_gap_on_its_own_months():
    """TRBFX stands in for STIP before December 2010 and costs 0.45%/yr more.
    The gap is added back to the PROXY's months only — the fund's own returns
    already net its own fee — and the fund's first month is its own, not the
    proxy's."""
    import data_collection_regime_model as dc
    idx = pd.date_range('2009-01-31', periods=36, freq='ME')
    proxy = pd.Series(0.004, index=idx)
    actual = pd.Series(0.006, index=idx[24:])
    spliced, n_proxy = dc.splice_with_proxy(proxy, actual, 0.0045)
    assert n_proxy == 24 and len(spliced) == 36, (n_proxy, len(spliced))
    assert np.allclose(spliced.iloc[:24], 0.004 + 0.0045 / 12), spliced.iloc[:3]
    assert np.allclose(spliced.iloc[24:], 0.006), 'the fund\'s own months were charged too'
    assert dc.PROXY_EXPENSE_GAP.get('STIP') == 0.0045 and dc.PROXIES.get('STIP') == 'TRBFX'


def test_regional_scenarios_move_markets_not_factors():
    """`regions` must touch mkt_rf per region and leave style premia alone.

    Multipliers cannot express this — they scale all three regions together —
    which is why every style scenario left US at 7.19% against developed 4.87%
    and a US-heavy portfolio won every column partly for a reason nothing
    tested. So both halves are pinned: the market premia converge, and the
    style premia do not move.
    """
    base = factors.premia_for('full history')
    assert base['us']['mkt_rf'] != base['dev']['mkt_rf'], base

    catch = factors.premia_for('intl catches up')
    assert catch['dev']['mkt_rf'] == base['us']['mkt_rf'], catch['dev']
    assert catch['emg']['mkt_rf'] == base['emg']['mkt_rf'], 'emg must be untouched'

    conv = factors.premia_for('regions converge')
    mkts = {conv[r]['mkt_rf'] for r in ('us', 'dev', 'emg')}
    assert len(mkts) == 1, conv

    # Style premia untouched by a regional scenario, and vice versa.
    for r in ('us', 'dev', 'emg'):
        for f in ('smb', 'hml', 'rmw', 'cma', 'mom'):
            assert conv[r][f] == base[r][f], (r, f)
    vd = factors.premia_for('value dead')
    assert vd['us']['hml'] == 0.0
    assert vd['dev']['mkt_rf'] == base['dev']['mkt_rf'], 'style kill must not move markets'

    # 'us stays ahead' is the default under another name — an alias, so the
    # status quo appears in the menu as a choice rather than as a given.
    assert factors.premia_for('us stays ahead') == base


def test_every_sensitivity_dimension_is_wired():
    """The registry is the single source for the CLI and run.py's menu.

    Each entry must name a real group, carry default values of its own kind,
    and — if env-backed — name a variable something actually reads. A dimension
    that looks present in the menu and silently sweeps nothing is the failure
    this prevents.
    """
    import sensitivity as sens
    groups = {g for g, _, _ in sens.GROUPS}
    seen_env = set()
    for name, d in sens.DIMENSIONS.items():
        assert d.group in groups, (name, d.group)
        assert d.kind in ('numeric', 'named'), (name, d.kind)
        assert d.note and d.default, name
        vals = [v.strip() for v in d.default.split(',') if v.strip()]
        assert len(vals) >= 2, (name, vals)
        if d.kind == 'numeric':
            for v in vals:
                float(v)                       # must parse, or the sweep dies
        if d.env:
            assert d.env.startswith('MC_'), (name, d.env)
            seen_env.add(d.env)
    # Named scenario dimensions must offer values factors actually knows.
    for name in ('scenario', 'regions'):
        for v in sens.DIMENSIONS[name].default.split(','):
            assert v.strip() in factors.SCENARIOS, (name, v)
    # The knobs added 2026-09-11 are the whole point of the registry; if any
    # stops being swept, the conclusion that depends on it is unfalsifiable.
    for must in ('MC_RF_REAL', 'MC_INFL_BETA_EQUITY', 'MC_INFLATION_STD'):
        assert must in seen_env, must


def test_a_report_warns_when_a_level_did_not_converge():
    """A level that missed its confidence target must SAY so, and say which way.

    solve_sustainable_spend has returned `converged`/`gap` since 2026-09-11 and
    the report ignored them, so portfolio 102 — whose level still moves with
    sample size — printed in the same typeface as a settled one. The SIGN
    matters too: landing below the target means the figure is too high (a risk),
    landing above means the search stopped early (conservative).
    """

    def render(gap, converged):
        res = {'141': {'spend': 200_000.0, 'p10_realised': 180_000.0,
                       'depletion_rate': 0.0, 'median_bequest': 1e6,
                       'worst_cut': 0.0, 'success_rate': 0.90 + gap,
                       'twr_cagr': np.array([0.05]), 'sortino_median': 1.0,
                       'per_path_mean': np.array([1.0, 2.0]),
                       'converged': converged, 'gap': gap}}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report_retirement._flat_table(['141'], res, {'mode': 'retirement'}, '141')
        return buf.getvalue()

    ok = render(0.002, True)
    assert 'OPTIMISTIC' not in ok and 'conservative' not in ok, ok

    high = render(-0.025, False)
    assert 'OPTIMISTIC' in high, high
    assert 'too high' in high, high

    low = render(+0.025, False)
    assert 'conservative' in low, low
    assert 'OPTIMISTIC' not in low, 'an overshoot is not a risk warning'


def test_the_accumulation_table_shows_the_spread_not_just_the_median():
    """p10/p90 and the drawdown must reach the page.

    The shared cash-rate and inflation bands added 2026-09-11 exist to widen the
    distribution; reporting only a median understates what the model knows about
    its own uncertainty. The drawdown was already computed and discarded — and
    _max_drawdowns returns PERCENT, so a %-format would multiply by 100 twice
    (it printed -2361% before this was caught).
    """

    finals = np.linspace(0.5, 3.0, 200)
    res = {'141': {'finals': finals, 'median': float(np.median(finals)),
                   'sortino': np.linspace(0.5, 1.5, 200), 'sortino_median': 1.0,
                   'twr_cagr': np.array([0.07]),
                   'max_drawdown': np.array([-30.0] * 200),
                   'median_path': np.linspace(1e5, 3e6, 400)}}
    report_retirement._CONFIG_FOR_SPLIT['cfg'] = {
        'age': 40, 'retirement_age': 65, 'time_horizon_years': 30,
        'initial_value': 100_000.0,
        'contribution_schedule': {'frequency': 'annual', 'accounts': {}}}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_retirement._flat_table(['141'], res, {'mode': 'accumulation'}, '141')
    out = buf.getvalue()
    for token in ('p10', 'p90', 'drawdown', 'WHAT GOT YOU THERE', 'from growth'):
        assert token in out, (token, out)
    assert '-30%' in out, f'drawdown must print as percent, not percent-of-percent: {out}'
    assert '-3000%' not in out and '-2361%' not in out, out


def test_the_consistency_gate_fails_on_each_drift_it_names():
    """Stage 1 of validate_all had only ever run against the live repo, so
    nothing showed it could fail. Each check is fed the drift it exists for on
    a copy of the real files: deleting the import (which check 1 used to pass,
    since 'imported' and 'undefined' both read as None), a hardcoded list, a
    swapped JSON order, a wrong-shape matrix, a missing CSV column and a
    ticker in two role buckets."""
    import shutil
    import check_consistency as cc
    real = {k: getattr(cc, k) for k in ('ROOT', 'JSON', 'CSV', '_US_FUNDS')}
    MODEL_TICKERS = loaders.MODEL_TICKERS
    n = len(MODEL_TICKERS)

    def gate(break_it):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'output').mkdir()
            shutil.copy(real['ROOT'] / 'data_collection_regime_model.py', root)
            shutil.copy(real['JSON'], root / 'output')
            shutil.copy(real['CSV'], root)
            for kind in ('baseline', 'low', 'medium', 'high'):
                shutil.copy(real['ROOT'] / 'output' / f'correlation_{kind}_{n}.npy',
                            root / 'output')
            try:
                cc.ROOT, cc.JSON = root, root / 'output' / real['JSON'].name
                cc.CSV = root / real['CSV'].name
                break_it(root)
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    try:
                        cc.main()
                    except SystemExit as e:
                        code = e.code
            finally:
                for k, v in real.items():
                    setattr(cc, k, v)
        return code, [l.strip() for l in out.getvalue().splitlines()
                      if '[FAIL]' in l]

    assert gate(lambda r: None) == (0, [])

    def edit(name, old, new):
        def go(root):
            f = root / name
            text = f.read_text()
            assert old in text, (name, old)
            f.write_text(text.replace(old, new, 1))
        return go

    dc = 'data_collection_regime_model.py'
    no_import = edit(dc, 'MODEL_TICKERS as TICKERS, ', '')
    literal = edit(dc, '\nfrom montecarlo.loaders import (',
                   f'\nTICKERS = {list(MODEL_TICKERS)!r}\nfrom montecarlo.loaders import (')

    def swapped(root):
        f = root / 'output' / real['JSON'].name
        data = json.loads(f.read_text())
        keys = list(data)
        i = keys.index(MODEL_TICKERS[0])
        keys[i], keys[i + 1] = keys[i + 1], keys[i]
        f.write_text(json.dumps({k: data[k] for k in keys}))

    def misshaped(root):
        np.save(root / 'output' / f'correlation_high_{n}.npy', np.eye(n - 1))

    def no_column(root):
        f = root / real['CSV'].name
        lines = f.read_text().splitlines(True)
        lines[0] = lines[0].replace(f',{MODEL_TICKERS[1]},', ',XXXX,', 1)
        f.write_text(''.join(lines))

    def two_roles(root):
        cc._US_FUNDS = list(real['_US_FUNDS']) + [loaders._BOND_FUNDS[0]]

    for name, brk, label in (
            ('no import', no_import, 'data_collection imports MODEL_TICKERS'),
            ('literal', literal, 'data_collection imports MODEL_TICKERS'),
            ('json order', swapped, 'output JSON ticker_list'),
            ('npy shape', misshaped, f'correlation matrices are {n}x{n}'),
            ('csv column', no_column, 'has all ticker columns'),
            ('two roles', two_roles, 'every ticker classified into one role')):
        code, fails = gate(brk)
        assert code == 1 and len(fails) == 1 and label in fails[0], (name, fails)


def test_parameter_drift_separates_a_bug_from_a_refresh():
    """The data golden master's comparison logic, which nothing else pins.

    check_parameter_drift is a gate stage, so a fault in `compare` shows up as a
    gate that passes everything — the worst failure mode a guard has. The
    tolerances were sized from MEASURED refresh drift (<=0.11pp over eleven
    days, COST 0.54pp); the first version used 2pp, too loose to catch the
    1.70pp VOO bug it was written for.
    """
    import copy
    import check_parameter_drift as cpd

    base = {'tickers': {'VOO': {'expected_return': 0.11, 'base_volatility': 0.15,
                                'parameter_std': 0.02, 'regime_low': 0.09,
                                'regime_medium': 0.15, 'regime_high': 0.20,
                                'history_months': 680},
                        'COST': {'expected_return': 0.18, 'base_volatility': 0.19,
                                 'parameter_std': 0.05, 'regime_low': 0.13,
                                 'regime_medium': 0.19, 'regime_high': 0.25,
                                 'history_months': 240}},
            'correlations': {'baseline': {'mean_abs': 0.50}},
            'market_regime': {'low': 0.09, 'medium': 0.15, 'high': 0.20},
            'last_updated': '2026-01-01'}
    assert cpd.compare(base, copy.deepcopy(base))[0] == []

    def moved(tkr, field, delta):
        n = copy.deepcopy(base)
        n['tickers'][tkr][field] += delta
        return cpd.compare(base, n)[0]

    assert moved('VOO', 'expected_return', 0.0170), 'the VOO-class bug must FAIL'
    assert not moved('VOO', 'expected_return', 0.0011), 'a refresh must pass'
    # A rolling-window ticker genuinely churns more, hence the multiplier.
    assert not moved('COST', 'expected_return', 0.0054), 'COST churn must pass'
    assert cpd.ROLLING_TOL_MULT > 1.0
    # history can only GROW — a decrease is a splice dropping data, at any size.
    assert moved('VOO', 'history_months', -1), 'any history loss must FAIL'
    assert not moved('VOO', 'history_months', +3)

    # Every banded field fails just over its band and passes just under it.
    for f, tol in cpd.TOL.items():
        assert moved('VOO', f, tol * 1.05), f'{f} over its band must FAIL'
        assert not moved('VOO', f, tol * 0.95), f'{f} under its band must pass'
    # The defect this gate exists for: VEA carrying emerging-market regime vol.
    assert moved('VOO', 'regime_high', 0.0567)

    gone = copy.deepcopy(base); gone['tickers'].pop('COST')
    assert cpd.compare(base, gone)[0]
    new = copy.deepcopy(base)
    new['tickers']['AVLV'] = dict(base['tickers']['VOO'])
    assert cpd.compare(base, new)[0] == [], 'a NEW ticker is reported, not failed'
    corr = copy.deepcopy(base); corr['correlations']['baseline']['mean_abs'] = 0.72
    assert cpd.compare(base, corr)[0]
    mr = copy.deepcopy(base); mr['market_regime']['high'] += 0.04
    assert cpd.compare(base, mr)[0], 'a 4pp market-regime vol move must FAIL'
    mr['market_regime']['high'] = base['market_regime']['high'] + 0.02
    assert cpd.compare(base, mr)[0] == []

    # ...and a failure reaches the gate's exit code.
    saved = cpd.BASELINE, cpd.snapshot
    with tempfile.TemporaryDirectory() as d:
        try:
            cpd.BASELINE = Path(d) / 'baseline.json'
            cpd.BASELINE.write_text(json.dumps(base))
            bad = copy.deepcopy(base)
            bad['tickers']['VOO']['regime_high'] += 0.0567
            cpd.snapshot = lambda: bad
            with contextlib.redirect_stdout(io.StringIO()):
                assert cpd.main([]) == 1
            cpd.snapshot = lambda: copy.deepcopy(base)
            with contextlib.redirect_stdout(io.StringIO()):
                assert cpd.main([]) == 0
        finally:
            cpd.BASELINE, cpd.snapshot = saved


def test_round_trip_reaches_expected_returns():
    """MC_ROUND_TRIP must actually move mu, or the sweep dimension is a no-op.

    It is exposed rather than enhanced — the asset-aware version measured
    +0.02pp and was discarded — so the whole value of the knob is that it
    reaches the number. MTUM is where it shows: 116%/yr turnover, so 0.15% vs
    0.40% round-trip is ~0.29pp of expected return.

    The arithmetic is exercised by setting the attribute and the ENV plumbing in
    a subprocess — no importlib.reload, for the reason in
    test_rf_source_resolves_and_rejects_nonsense.
    """

    fit = factors.load_fits().get('MTUM')
    if fit and fit.get('method') == 'factor':
        saved = factors.ROUND_TRIP
        try:
            mus = {}
            for v in (0.0015, 0.0040):
                factors.ROUND_TRIP = v
                exp, impl = factors.costs_for('MTUM', fit)
                mus[v] = factors.expected_return(fit, costs=(exp or 0.0, impl))
        finally:
            factors.ROUND_TRIP = saved
        assert mus[0.0015] > mus[0.0040], mus
        gap = mus[0.0015] - mus[0.0040]
        assert 0.001 < gap < 0.006, gap

    code = ("import sys; sys.path.insert(0, %r); from montecarlo import factors; import json; "
            "print(json.dumps(factors.ROUND_TRIP))" % str(ROOT))
    out = subprocess.run([sys.executable, '-c', code],
                         env=dict(os.environ, MC_ROUND_TRIP='0.0040'),
                         capture_output=True, text=True, cwd=str(ROOT))
    assert abs(json.loads(out.stdout.strip().splitlines()[-1]) - 0.0040) < 1e-12


def test_a_sweep_accepts_negative_values_without_eating_its_flags():
    """`infl_beta_equity -1.15,-0.80,0.0` must parse, and so must the flags.

    argparse reads a leading '-' as a flag, so the dimension that REVERSED the
    derisk answer could not be swept from the command line at all — it died with
    "unrecognized arguments". Every inflation pass-through beta is negative, so
    this is not an edge case.

    `--` is NOT the fix and this pins that too: it makes argparse treat
    --strategies and --sims as positionals, which fails differently and later.
    """
    import sensitivity as sens

    # No values -> the menu, exit 2. Flags must survive alongside a negative list.
    ns = []
    real = sens.sweep_point
    try:
        # Stub the measurement — this pins ARGUMENT PARSING, and a real sweep
        # would be three solves. Output is swallowed so the suite stays legible.
        sens.sweep_point = lambda d, v, st, si, *_: ns.append((d, v, st, si)) or {
            g: 1.0 + 0.1 * i for i, g in enumerate(st)}
        with contextlib.redirect_stdout(io.StringIO()):
            sens.main(['infl_beta_equity', '-1.15,-0.80,0.0',
                       '--strategies', '141,145', '--sims', '7'])
    finally:
        sens.sweep_point = real

    assert [v for _, v, _, _ in ns] == ['-1.15', '-0.80', '0.0'], ns
    assert all(st == ['141', '145'] for _, _, st, _ in ns), ns
    assert all(si == 7 for _, _, _, si in ns), 'the --sims flag must not be eaten'
    assert all('\x00' not in v for _, v, _, _ in ns), 'the sentinel must be stripped'


def test_the_report_quotes_the_premium_span_the_sweep_actually_measured():
    """The footer's premium warning must agree with sensitivity's own note.

    Every retirement run ends with a line quoting how far the plan level moves
    across the premium range. It was hardcoded at "~5x ($124k to $25k)" — a
    figure RETRACTED on 2026-09-10 when the VOO price-index proxy was fixed —
    and it survived every documentation sweep because it lives inside a print()
    rather than in a markdown table. It was still being printed at the end of
    every run two days later.

    So the two are now pinned to each other: if a re-measurement updates the
    sweep's registry note and not the report, this fails.
    """
    import sensitivity as sens

    src = inspect.getsource(report_retirement.run)
    quoted = set(re.findall(r'(\d+\.\d+)x', src))
    note = sens.DIMENSIONS['premium'].note
    measured = set(re.findall(r'(\d+\.\d+)x', note))
    assert measured, f'the premium dimension should record its span: {note!r}'
    assert quoted, 'the report should quote a premium span'
    assert measured <= quoted, (
        f'report quotes {quoted}, sweep measured {measured} — re-measure one '
        f'and not the other and this is how it shows')

    # And the retracted figures must not come back.
    for dead in ('~5x', '$124k', '$25k', '66.9M', '20.2M', '4.3x'):
        assert dead not in src, f'{dead} was retracted on 2026-09-10'


def test_head_to_head_says_what_it_is_differencing_and_which_way():
    """A confidence interval is unreadable without its subject and its sign.

    The block printed `145  [-463.86, -322.67]k  WORSE` under a header that
    explained why the intervals in a DIFFERENT table should be ignored — true,
    and not what someone reading THIS table needs. Three things have to be on
    the page: what is being differenced, which direction the sign runs, and
    that 'not resolved' is not 'equal'.
    """

    rng = np.random.default_rng(1)
    res = {g: {'per_path_mean': rng.normal(m, 20e3, 300), 'spend': m}
           for g, m in [('140', 123e3), ('145', 68e3), ('161', 131e3),
                        ('161>161+5@65', 124e3)]}
    per_base = {'140': ['140'], '145': ['145'],
                '161': ['161', '161>161+5@65']}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_retirement._matrix_pairs(['140', '145', '161'], per_base, res,
                         {'mode': 'retirement'}, [(131e3, '161')])
    out = buf.getvalue()

    assert 'how much MORE lifetime average spending than 140' in out, out
    assert 'not resolved' in out or 'straddles zero' in out, out
    assert 'low' in out and 'high' in out, 'the interval needs column headers'
    # The old header explained a DIFFERENT table's intervals; it must not return.
    assert 'intervals below do NOT mean indistinguishable' not in out, out
    # Accumulation differences a different quantity and must say so.
    buf2 = io.StringIO()
    res2 = {g: {'finals': rng.normal(m, 5.0, 300)} for g, m in
            [('140', 90.0), ('145', 45.0)]}
    with contextlib.redirect_stdout(buf2):
        report_retirement._matrix_pairs(['140', '145'], {'140': ['140'], '145': ['145']},
                         res2, {'mode': 'accumulation'}, [])
    assert 'final wealth' in buf2.getvalue(), buf2.getvalue()


def test_worst_year_is_a_percent_of_plan_not_a_negative_cut():
    """A 'cut' cannot exceed 100%, and a rising policy makes it negative.

    solver.score reports worst_cut = 1 - min(spending)/plan, which is sensible
    for a flat policy and nonsense for a rising one: vpw spends far above its
    certified floor, so the "cut" goes NEGATIVE and a real run printed
    "worst cut -155.2%". Against plan it is always placeable.
    """

    assert report_retirement.worst_year_pct(-1.122) == _approx(212.2), report_retirement.worst_year_pct(-1.122)
    assert report_retirement.worst_year_pct(0.15) == _approx(85.0)
    assert report_retirement.worst_year_pct(0.0) == _approx(100.0)
    # Never negative for any realised spending, which is the whole point.
    for wc in (-2.0, -1.0, -0.5, 0.0, 0.5, 0.99):
        assert report_retirement.worst_year_pct(wc) > 0.0, wc
    # And the column must not still be called a cut where it can exceed 100.
    src = inspect.getsource(report_retirement._matrix)
    assert "'worst cut'" not in src, 'renamed to worst yr — see worst_year_pct'


def _approx(v, tol=1e-6):
    class _A:
        def __eq__(self, other):
            return abs(other - v) < tol
    return _A()


def test_the_timing_verdict_says_which_way_it_was_wrong():
    """'0.5x high' means the estimate was 2x too high, which is not what it says.

    And the calibration floor was 20,000 paths — ABOVE what a typical run
    produces. Twelve strategies under a closed-form policy generate 1,500 each
    = 18,000, so the runs actually being done could never recalibrate the
    constant they were estimated with. It sat at 9.52ms against a real 4.77ms
    and every such run was announced at roughly twice its true length.
    """

    assert instrument.MIN_CALIBRATION_PATHS <= 18_000, (
        'a 12-strategy closed-form run makes 18k paths and must be able to '
        'recalibrate')
    assert instrument.MIN_CALIBRATION_PATHS >= 1_000, 'but not from noise'

    saved, saved_file = instrument.ENABLED, instrument.TIMING_FILE
    try:
        # REDIRECT THE TIMING FILE. summary() ends by calling calibrate(), which
        # WRITES. Without this the test recorded a fake 9,000-path run of
        # near-zero duration straight into the real config/timing.json — and
        # since it runs on every validate_all, it drove the stored constant
        # toward zero and made the run-time estimate progressively too low.
        with tempfile.TemporaryDirectory() as d:
            instrument.TIMING_FILE = Path(d) / 'timing.json'
            instrument.enable(True)
            instrument.reset()
            with instrument.timed('x'):
                instrument.record_paths(9_000, parallel=True)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                instrument.summary(estimated_minutes=60.0)   # over-estimated
            out = buf.getvalue()
            assert 'too HIGH' in out, out
            assert 'x high' not in out, 'the old ambiguous phrasing must be gone'
    finally:
        instrument.enable(saved)
        instrument.TIMING_FILE = saved_file


def test_a_saturated_row_does_not_assert_an_unverified_cause():
    """NOT BINDING must report what is observed, not a mechanism it guessed.

    The first version said "guaranteed income already covers the plan level" —
    inferred from one run, and only true from the CLAIM AGE. That run's Social
    Security started at 67 against withdrawals from 65, so the portfolio funded
    two years alone and the bare VT row did NOT saturate while its derisked
    siblings did. The asserted mechanism was wrong in the very output that
    prompted it.
    """

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_retirement.confidence_note({'vpw': {'A': {'success_rate': 1.0},
                                    'B': {'success_rate': 0.913}}}, ['A', 'B'])
    out = buf.getvalue()
    assert 'NOT BINDING' in out and 'every path succeeds' in out, out
    assert 'already covers' not in out, 'that cause was inferred, not measured'
    assert 'claim age' in out, 'the reader needs the thing to go check'
    # A tie-free claim about the closed form was also false.
    assert 'land exactly' not in out or 'no TIES' in out, out


def test_the_confidence_note_sizes_a_miss_against_the_sampling_error():
    """The 2026-09-13 run landed 87.7% against 90% at 1,500 paths — about 3 SE —
    under a note calling every gap "roughly one standard error". The note now
    sizes each miss against sqrt(p(1-p)/n) and flags what is not noise; that
    sizing had no test, and neither did the verification-path count it uses.
    (A row within 1pp of the target is not listed at all, so the "small" miss
    here is 1.2pp, not 0.3pp.)"""
    n = 1500
    se = (0.9 * 0.1 / n) ** 0.5

    def results(rates, with_n=True):
        return {'guyton_klinger': {
            g: dict({'success_rate': r}, **({'adequacy_stat': np.zeros(n)} if with_n else {}))
            for g, r in rates.items()}}

    def note(by_policy, specs):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report_retirement.confidence_note(by_policy, specs)
        return buf.getvalue()

    out = note(results({'far': 0.877, 'near': 0.888, 'inside': 0.897}), ['far', 'near', 'inside'])
    rows = {l.split(':')[0].strip(): l for l in out.splitlines() if '% ' in l or 'SE)' in l}
    assert f"({0.023 / se:.1f} SE)" in rows['far'], rows
    assert f"({0.012 / se:.1f} SE)" in rows['near'], rows
    assert 'NOT sampling error' in rows['far'], rows['far']
    assert 'NOT sampling error' not in rows['near'], rows['near']
    assert 'inside' not in rows, 'a miss under 1pp is silent'
    assert f"One standard error at {n:,} verification paths is {se:.2%}" in out, out
    assert 'did not converge' in out, 'a >2 SE row must say it is unresolved'

    # only a small miss: sized, but never called non-convergence
    out = note(results({'near': 0.888}), ['near'])
    assert '(1.5 SE)' in out and 'NOT sampling error' not in out, out
    assert 'did not converge' not in out, out

    # at most six rows are listed; the rest are counted
    seven = {f'S{i}': 0.85 + 0.002 * i for i in range(7)}
    out = note(results(seven), list(seven))
    assert sum(1 for l in out.splitlines() if l.strip().startswith('S')) == 6, out
    assert '… and 1 more' in out, out

    # no verification-path count to size against: say so, never a made-up SE
    out = note(results({'far': 0.877}, with_n=False), ['far'])
    assert 'SE)' not in out and 'finite sample' in out, out


def test_the_verification_path_count_falls_back_from_the_stat_to_the_spending_matrix():
    """confidence_note sizes its standard error from this count: adequacy_stat's
    length first, then the spending matrix's rows, then nothing."""
    verify = report_retirement._verify_sims
    both = {'p': {'A': {'adequacy_stat': np.zeros(1500), 'spending': np.zeros((800, 30))}}}
    assert verify(both, ['A']) == 1500
    only_spending = {'p': {'A': {'adequacy_stat': np.zeros(0), 'spending': np.zeros((800, 30))}}}
    assert verify(only_spending, ['A']) == 800, 'an empty stat must fall through'
    assert verify({'p': {'A': {'spending': np.zeros((800, 30))}}}, ['A']) == 800
    assert verify({'p': {'A': {'success_rate': 0.9}}}, ['A']) == 0
    # the first strategy that carries a count decides
    assert verify({'p': {'A': {}, 'B': {'adequacy_stat': np.zeros(64)}}}, ['A', 'B']) == 64


def test_calibration_moves_slower_than_its_own_noise():
    """One run must not be able to halve the stored constant.

    At a 50/50 blend it did: the stored seconds-per-path went 9.52ms -> ~2.3ms
    on a single measurement, and the next run was then announced at 0.7 minutes
    when it took 1.5. Three IDENTICAL 12-strategy runs measured 3.63, 4.77 and
    5.14 ms/path, so the measurement carries a 1.4x spread from machine load
    alone — the constant has to move slower than that, or it chases noise.
    """
    import time

    saved_file, saved_enabled = instrument.TIMING_FILE, instrument.ENABLED
    try:
        with tempfile.TemporaryDirectory() as d:
            instrument.TIMING_FILE = Path(d) / 't.json'
            instrument.TIMING_FILE.write_text(json.dumps(
                {'seconds_per_path': 0.010}) + '\n')
            instrument.enable(True)
            instrument.reset()
            with instrument.timed('x'):
                instrument.record_paths(instrument.MIN_CALIBRATION_PATHS + 1,
                                        parallel=True)
            # Fake the elapsed time. calibrate() now refuses anything under
            # MIN_PLAUSIBLE_SECONDS_PER_PATH, because a near-zero measurement is
            # a test recording fake paths — which is exactly what this is. Wind
            # the clock back instead of sleeping for real.
            instrument._t0 = time.time() - 0.002 * (
                instrument.MIN_CALIBRATION_PATHS + 1)
            got = instrument.calibrate()
            assert got, 'a large-enough parallel run must calibrate'
            measured, stored = got
            # However fast the fake run looked, the stored value may not fall
            # below half the prior.
            assert stored >= 0.010 * 0.5 - 1e-12, (measured, stored)
            assert stored <= 0.010 * 1.5 + 1e-12, (measured, stored)
    finally:
        instrument.TIMING_FILE, _ = saved_file, None
        instrument.enable(saved_enabled)


def test_the_best_caveat_names_its_direction():
    """The same pair is shown twice with opposite signs; say which is which.

    `best:` reports top-minus-second while HEAD-TO-HEAD, fifteen lines below,
    reports challenger-minus-base. A real run printed [-55.11, +93.71]k in one
    and [-93.7k, +55.1k] in the other for 161 against 161>161+5@65 — both
    correct, reading as a contradiction.
    """

    rng = np.random.default_rng(3)
    res = {'A': {'per_path_mean': rng.normal(131e3, 40e3, 400)},
           'B': {'per_path_mean': rng.normal(124e3, 40e3, 400)}}
    out = report_retirement._best_caveat([(131e3, 'A'), (124e3, 'B')], res, {'mode': 'retirement'})
    assert out, 'a two-row comparison should produce a caveat or a margin'
    if 'NOT resolved' in out:
        # Only this branch needed the direction spelled out — it straddles zero,
        # so the sign alone does not say which way. "beats B by [+x, +y]" cannot
        # be read backwards.
        assert 'A minus B' in out, out
    else:
        assert 'beats B' in out, out
    # One decimal, matching the head-to-head table it sits beside.
    assert not re.search(r'[+-]\d+\.\d\dk', out), f'match the table: {out}'


def test_crossing_recovers_a_known_threshold():
    """On a clean line the crossing must be exact; with noise it must stay close,
    because it fits the neighbourhood rather than interpolating two points."""
    xs = np.linspace(50e3, 250e3, 9)
    ys = 0.98 - (xs - 50e3) * (0.20 / 200e3)        # crosses 0.90 at 130k
    x0, slope, clip = solver._crossing(xs, ys, 0.90)
    assert abs(x0 - 130e3) < 1e-6, x0
    assert clip == 0, 'an interior crossing must not report a clip'
    assert slope < 0, 'success must fall with spend'
    rng = np.random.default_rng(3)
    errs = []
    for _ in range(40):
        noisy = solver._isotonic_decreasing(ys + rng.normal(0, 0.015, len(ys)))
        errs.append(abs(solver._crossing(xs, noisy, 0.90)[0] - 130e3) / 130e3)
    assert np.median(errs) < 0.12, np.median(errs)
    # and it must never wander outside the sampled bracket
    assert xs[0] <= solver._crossing(xs, ys * 0 + 0.99, 0.90)[0] <= xs[-1]


def test_bracket_break_allows_for_sampling_error():
    """Bisection can never leave [lo, hi], so the level that closes the bracket
    is a permanent ceiling. Breaking the instant a SEARCH sample dips below the
    target let one unlucky evaluation cap the answer: measured on portfolio 140,
    guyton_klinger at $100k scored 89.2% on 400 paths and 91.2% on 1500, closing
    the bracket at $100k when the true 90% level was $109k — a 17% undershoot
    (fixed_pct, 8%, identically). A 0.8pp miss is well inside the 1.5pp standard
    error of a 400-path estimate, so this was likely, not unlucky.

    The break now sits two standard errors below the target."""
    import math
    src = inspect.getsource(solver.solve_sustainable_spend)
    assert 'bracket_floor' in src, 'bracket must not break at the bare target'
    for n, conf in ((400, 0.90), (200, 0.90), (1500, 0.90)):
        se = math.sqrt(conf * (1 - conf) / n)
        floor_ = conf - 2 * se
        assert floor_ < conf, 'must be below the target'
        # the observed miscall (89.2% at n=400) must no longer break the bracket
        if n == 400:
            assert 0.892 > floor_, f'89.2% should survive, floor is {floor_:.3f}'
            assert 0.85 < floor_, 'but a genuine failure must still stop it'
        # a smaller sample is noisier, so it must tolerate more
        assert floor_ < conf - 0.005


def test_provenance_records_what_changes_every_number():
    """adequacy and the scoring version change every result without appearing
    anywhere in the run spec, so a diff across that boundary would blame a
    scoring change on whatever moved in the spec. Runs saved before per-band
    adequacy carry neither key, which is how a reader can tell."""
    cfg = {'adequacy': 'thirds', 'withdrawal': {'floor': 0.85}}
    pr = persistence.provenance(cfg)
    assert pr['adequacy'] == 'thirds'
    assert pr['floor'] == 0.85
    assert pr['scoring_version'] == persistence.SCORING_VERSION >= 2
    # still callable without a config (history listing), minus the run-specific keys
    bare = persistence.provenance()
    assert 'scoring_version' in bare and 'adequacy' not in bare


def test_diff_flags_an_incomparable_scoring_change_loudly():
    """Not one difference among several: different criteria make the rows
    meaningless, so it must be said before the numbers."""
    def rec(stamp, ver, adq, spend):
        return {'timestamp': stamp, 'spec': {'mode': 'retirement', 'spend': 'solve'},
                'provenance': {'scoring_version': ver, 'adequacy': adq,
                               'inflation': 0.03},
                'config': {}, 'strategies': ['140'],
                'results': {'gk': {'140': {'spend': spend}}}}
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / 'a.json', Path(d) / 'b.json'
        a.write_text(json.dumps(rec('old', 1, 'lifetime', 181e3)))
        b.write_text(json.dumps(rec('new', 2, 'per_band', 115e3)))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            persistence.print_diff(a, b)
    out = buf.getvalue()
    assert 'NOT COMPARABLE' in out, out
    assert out.index('NOT COMPARABLE') < out.index('before'), 'must precede the rows'


def test_sensitivity_verdict_treats_resolution_as_a_tie():
    """Compared as point estimates the verdict fired on noise: the inflation
    sweep announced "RANKING-RELEVANT. This assumption changes what you should
    do." on a $1k gap over $205k. Near the confidence crossing the success curve
    is flat, so a few percent of spend is resolution, not signal.

    Drives the verdict the tool prints, through main with the measurement
    stubbed — the verdict is the conclusion the tool exists for."""
    import sensitivity as sens
    assert 0 < sens.TIE_REL < 0.25, sens.TIE_REL

    # A $1k swap over $205k is a tie, not a reversal.
    noise = {'1.0': {'141': 205_000, '145': 204_000},
             '0.0': {'141': 204_000, '145': 205_000}}
    assert sens.reversals(noise, ['141', '145'], ['1.0', '0.0']) == []
    assert sens.tied_order(noise, ['141', '145'], '1.0') == '141 = 145'
    # A material swap is one, signed per point.
    # 169 trails 141 materially at both points: ordered, but no reversal.
    real = {'1.0': {'141': 285_000, '145': 245_000, '140': 244_000, '169': 200_000},
            '0.0': {'141': 170_000, '145': 185_000, '140': 184_000, '169': 120_000}}
    flips = sens.reversals(real, ['141', '145', '140', '169'], ['1.0', '0.0'])
    assert [(a, b) for a, b, _ in flips] == [('141', '145'), ('141', '140')], flips
    assert flips[0][2] == {'1.0': 1, '0.0': -1}, flips
    assert sens.tied_order(real, list(real['1.0']), '1.0') == '141 > 145 = 140 > 169'
    # Tied only when inside the band of the LARGER value, either way round.
    assert sens.separation(100.0, 100.0 * (1 - sens.TIE_REL) + 0.01) == 0
    assert sens.separation(100.0, 100.0 * (1 - sens.TIE_REL) - 0.01) == 1
    assert sens.separation(100.0 * (1 - sens.TIE_REL) - 0.01, 100.0) == -1

    def verdict(results):
        saved = sens.sweep_point
        try:
            sens.sweep_point = lambda d, v, st, *_: results[v]
            with contextlib.redirect_stdout(io.StringIO()) as out:
                sens.main(['premium', ','.join(results),
                           '--strategies', ','.join(next(iter(results.values())))])
        finally:
            sens.sweep_point = saved
        return out.getvalue()
    assert '-> LEVEL-ONLY' in verdict(noise), verdict(noise)
    shown = verdict(real)
    assert '-> RANKING-RELEVANT' in shown and '141 vs 145' in shown, shown
    assert 'premium=0.0: 145 > 141' in shown, shown


def test_param_std_scale_widens_the_epistemic_band():
    """The shipped bands are a hand-calibrated judgement, and a tight one: at
    AVUV's 0.020 the model implies a ~0.09% probability that AVUV has no premium
    at all. The knob does not pick a better number — it makes the CONSEQUENCE of
    the choice measurable, so the judgement can be informed by a sweep.

    Read from the environment because workers re-import under `spawn`, the same
    constraint as PREMIUM_SCALE; an assigned module global would not reach them.
    """
    src = inspect.getsource(pa)
    assert "os.environ.get('MC_PARAM_STD_SCALE'" in src, 'must be env-read'
    assert 'PARAM_UNCERTAINTY_STD' in src
    # scaling is applied where the band is built, so every ticker moves together
    assert '* PARAM_STD_SCALE' in src
    # and the sweep tool must route it through a subprocess, like the other
    # env-backed dimensions
    import importlib
    sens = importlib.import_module('tools.sensitivity')
    assert sens.ENV_DIMS.get('param_std') == 'MC_PARAM_STD_SCALE'


def test_instrumentation_is_free_when_disabled():
    """Off by default, and every entry point a flag check — a debug facility that
    costs anything when off will get switched off and rot."""
    instrument.enable(False)
    instrument.reset()
    with instrument.timed('ignored'):
        instrument.record_paths(1_000_000)
    assert instrument.totals() == (0, 0), 'must record nothing when disabled'
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        instrument.summary(estimated_minutes=5.0)
    assert buf.getvalue() == '', 'must print nothing when disabled'


def test_instrumentation_counts_paths_and_nests_phases():
    try:
        instrument.enable(True)
        instrument.reset()
        with instrument.timed('solve'):
            with instrument.timed('bracketing'):
                instrument.record_paths(400)
            with instrument.timed('verification'):
                instrument.record_paths(1500)
        paths, evals = instrument.totals()
        assert (paths, evals) == (1900, 2), (paths, evals)
        assert instrument._phases['solve / bracketing']['depth'] == 1
    finally:
        instrument.enable(False)


def test_serial_batches_never_calibrate_the_estimate():
    """Batches under solver.PARALLEL_THRESHOLD skip the pool, so their per-path
    cost measures a different execution mode — a smoke run showed 32.9ms/path
    against a 4ms constant purely because nothing was parallel. Calibrating from
    that would make every production forecast wrong the other way."""
    try:
        instrument.enable(True)
        instrument.reset()
        with instrument.timed('serial work'):
            instrument.record_paths(50_000, parallel=False)   # over the size gate
        assert instrument.calibrate() is None, 'serial runs must not calibrate'
        instrument.reset()
        with instrument.timed('tiny parallel'):
            instrument.record_paths(100, parallel=True)       # under the size gate
        assert instrument.calibrate() is None, 'small samples must not calibrate'
    finally:
        instrument.enable(False)


def test_time_estimate_matches_the_algorithm_it_prices():
    """_SOLVE_EVALS described the bisection that no longer exists and counted one
    verification where there are two, together under-pricing a run ~2.6x: a
    9-solve run announced 4 minutes and took 11."""
    assert runner._VERIFY_RUNS == 2, 'fit + Newton confirmation'
    assert 12 <= runner._SOLVE_EVALS <= 18, runner._SOLVE_EVALS
    spec = {'mode': 'retirement', 'spend': 'solve', 'search_sims': 400,
            'verify_sims': 1500}
    paths = 400 * runner._SOLVE_EVALS + 1500 * runner._VERIFY_RUNS
    assert paths > 400 * 14 + 1500, 'must price more than the old bisection did'
    # and a measured value, once recorded, must win over the hand-set constant
    assert runner._seconds_per_path() > 0


def test_estimate_prices_the_closed_form_route_separately():
    """vpw solves in closed form — one batch of verify_sims — where a searched
    policy runs ~15 evaluations plus two verifications. Measured at production
    sims: 9.0k paths per guyton_klinger solve against 1.5k per vpw solve.

    Pricing them alike over-counted a 6-solve run at 54k paths against an actual
    24k, which happened to cancel against a seconds-per-path constant that was
    itself half the true value. The estimate looked accurate while both terms
    were wrong, and would have broken the moment either was fixed."""
    base = {'mode': 'retirement', 'spend': 'solve', 'search_sims': 400,
            'verify_sims': 1500}
    gk = runner.estimate_minutes(1, dict(base, compare_policies=['guyton_klinger']))
    vpw = runner.estimate_minutes(1, dict(base, compare_policies=['vpw']))
    assert vpw < gk / 3, f'closed form must be much cheaper: {vpw:.3f} vs {gk:.3f}'
    # and a mixed run must cost the sum, not twice the expensive one
    both = runner.estimate_minutes(1, dict(base,
                                           compare_policies=['guyton_klinger', 'vpw']))
    assert abs(both - (gk + vpw)) < 1e-9, (both, gk, vpw)


def test_fractional_bond_tiers_render_honestly():
    """A 2.5% tier rendered as "98/2" under :.0f — neither the allocation nor a
    pair that sums to 100. Fractional tiers are selectable from the prompt, so
    the label has to carry the decimal when there is one."""
    assert runner._pct(2.5) == '2.5'
    assert runner._pct(97.5) == '97.5'
    assert runner._pct(10.0) == '10', 'whole numbers must stay clean'
    assert runner._pct(0.0) == '0'
    for whole, bond in ((100, 0.0), (97.5, 0.025), (90, 0.10)):
        lhs, rhs = runner._pct((1 - bond) * 100), runner._pct(bond * 100)
        assert abs(float(lhs) + float(rhs) - 100) < 1e-9, (lhs, rhs)


def test_calibration_reports_the_value_it_stores():
    """The message printed the run's MEASUREMENT while the file stored a blend of
    it with the previous run — it read "recalibrated: 8.57ms" when 8.65 was
    written. Blending is right; reporting the wrong number of the two is not."""
    try:
        instrument.enable(True)
        instrument.reset()
        with tempfile.TemporaryDirectory() as d:
            orig = instrument.TIMING_FILE
            instrument.TIMING_FILE = Path(d) / 'timing.json'
            try:
                instrument.TIMING_FILE.write_text(
                    json.dumps({'seconds_per_path': 0.010}))
                with instrument.timed('work'):
                    instrument.record_paths(50_000, parallel=True)
                # See MIN_PLAUSIBLE_SECONDS_PER_PATH — a zero-duration fake run
                # is refused, so give it a believable one.
                import time as _t
                instrument._t0 = _t.time() - 0.004 * 50_000
                got = instrument.calibrate()
                assert got is not None
                measured, stored = got
                prior = 0.010
                # The POINT of this test is that the reported number is the
                # stored one. Do not re-state the blend weights here: they moved
                # from 0.5 to 0.25 when the constant turned out to chase its own
                # noise, and a duplicated formula just fails the next time they
                # move. Assert what must be true of ANY blend instead.
                assert min(measured, prior) - 1e-12 <= stored <= max(measured, prior) + 1e-12, \
                    (measured, prior, stored)
                assert stored != measured or abs(measured - prior) < 1e-12, \
                    'a blend with a different prior must not equal the raw measurement'
                on_disk = json.loads(instrument.TIMING_FILE.read_text())
                assert abs(on_disk['seconds_per_path'] - stored) < 1e-6, \
                    'the reported stored value must be what is on disk'
            finally:
                instrument.TIMING_FILE = orig
    finally:
        instrument.enable(False)


def _stub_solve(search, verify, verbose=False):
    """Solve against two KNOWN curves: `search` answers the 100-path reads and
    `verify` the 300-path ones, so a test can place the verification curve
    wherever the solver's correction has to go. Returns (result, the spends
    verified, what it printed)."""
    checked = []

    def fake(plan, config, spend, n_sims=500, floor=0.85, parallel=True,
             full=False, adequacy=solver.DEFAULT_ADEQUACY, seed_offset=0):
        if n_sims == 300:
            checked.append(float(spend))
        curve = verify if n_sims == 300 else search
        return {'spend': float(spend),
                'success_rate': float(np.clip(curve(spend), 0.0, 1.0)),
                'success_by_floor': {}, 'depletion_rate': 0.0}

    real, buf = solver.evaluate, io.StringIO()
    try:
        solver.evaluate = fake
        with contextlib.redirect_stdout(buf):
            got = solver.solve_sustainable_spend(
                {}, {'age': 65, 'time_horizon_years': 30,
                     'withdrawal': {'start_age': 65}},
                confidence=0.90, search_sims=100, verify_sims=300,
                parallel=False, verbose=verbose)
    finally:
        solver.evaluate = real
    return got, checked, buf.getvalue()


def test_verification_step_iterates_on_a_steep_curve():
    """One step is not enough where the curve is steep. Portfolio 102 took a
    single +7.9% step and landed at 82.1% against a 90% target, then stopped —
    reporting a plan level under a heading claiming 90%. The fitted slope came
    from a flatter neighbourhood than the answer actually sits in.

    Iterating with a SECANT slope — measured between the two verified points
    rather than reused from the search-sample fit — walked it 82.1% -> 84.9% ->
    87.2% -> 88.3%, and the plan level down 7% from $73k to $68k. It stops there
    because 1.7pp is about two standard errors of a 1500-path verification;
    further steps would chase noise."""
    # The search curve crosses at 205k (off the 200k probe, so the fit is not
    # read as clipped); the verification curve crosses at 125k and is four
    # times flatter, so every step on the search slope covers a quarter of the
    # gap.
    got, checked, _ = _stub_solve(lambda x: 0.90 - (x - 205_000.0) * 2e-6,
                                  lambda x: 0.90 - (x - 125_000.0) * 0.5e-6)
    assert got['converged'] is True and got['spend'] < 145_000.0, got
    # a secant lands on a line in two steps; the search slope alone needs five
    assert got['newton_steps'] <= 3, (got['newton_steps'], checked)
    # a landed level carries its dollar precision
    assert np.isfinite(got['level_se']) and got['level_se'] > 0, got['level_se']
    assert solver.MAX_NEWTON >= 4, 'three was measured to cut off mid-convergence'
    assert solver.MAX_NEWTON <= 10, 'each step costs a full verification'


def test_verification_step_stops_when_it_is_close_enough():
    """The loop must not burn verifications once inside a point of the target,
    or every well-behaved solve pays for a pathological one."""
    # half a point off: one verification, no step
    got, checked, _ = _stub_solve(lambda x: 0.90 - (x - 205_000.0) * 2e-6,
                                  lambda x: 0.905 - (x - 205_000.0) * 2e-6)
    assert got['newton_steps'] == 0 and len(checked) == 1, \
        (got['newton_steps'], checked)
    # two points off, but on a curve so steep the step is 0.5% of the level
    got, checked, _ = _stub_solve(lambda x: 0.90 - (x - 205_000.0) * 2e-5,
                                  lambda x: 0.90 - (x - 204_000.0) * 2e-5)
    assert len(checked) == 1, ('a tiny step was verified anyway', checked)


def test_library_labels_separate_rows_that_share_a_prefix():
    """The picker cut every name where they still agreed.

    Nine momentum templates rendered as the identical string
    "70D:30I // 100:0 // 5C // 12AV", and 150-155 differ ONLY in a trailing
    parenthetical, i.e. exactly the part a fixed-width truncation removes.
    """
    P = {'140': {'name': '70D:30I // 100:0 // 5C // 12AVUV // 12VFMO -- Status Quo'},
         '142': {'name': '60D:40I // 100:0 // 5C // 12AVUV // 12VFMO'},
         '144': {'name': '70D:30I // 100:0 // 5C // 12AVUV // 12MTUM'},
         '150': {'name': '70D:30I // 100:0 // 5C // 12AVUV // 12VFMO (Simple Intl)'},
         '151': {'name': '70D:30I // 100:0 // 5C // 12AVUV // 12VFMO (Simple DE)'}}
    got = runner.distinguish(P, list(P))
    assert got['140'] == 'Status Quo', got['140']          # -- comment wins
    assert got['142'] == '60D:40I', got['142']
    assert got['144'] == '12MTUM', got['144']
    assert got['150'] == '12VFMO (Simple Intl)', got['150']
    assert len(set(got.values())) == len(P), f'labels collide: {got}'
    for label in got.values():
        assert len(label) <= 30, f'{label!r} would still be truncated'


def test_library_labels_are_unique_even_when_the_difference_is_shared():
    """Describing each row's own difference does NOT make the labels unique.

    120 ('... 12AVUV // 10VTV') and 160 ('... 12AVUV // 12VFMO // 10VTV') both
    reduced to '10VTV', so a 6-portfolio matrix printed two rows under the same
    name with $26M between them (44.53M and 70.66M). A colliding group has to be
    widened until it separates.
    """
    P = {'140': {'name': '70D:30I // 100:0 // 5C // 12AVUV // 12VFMO -- Status Quo'},
         '120': {'name': '70D:30I // 100:0 // 5C // 12AVUV // 10VTV'},
         '121': {'name': '70D:30I // 100:0 // 5C // 12AVUV // 10AVLV'},
         '160': {'name': '70D:30I // 100:0 // 5C // 12AVUV // 12VFMO // 10VTV'},
         '161': {'name': '70D:30I // 100:0 // 5C // 12AVUV // 12VFMO // 10AVLV'}}
    got = runner.distinguish(P, list(P))
    assert len(set(got.values())) == len(P), f'labels collide: {got}'
    assert got['120'] == '10VTV', got['120']
    assert got['160'] == '12VFMO // 10VTV', got['160']
    assert got['161'] == '12VFMO // 10AVLV', got['161']
    # identical templates cannot be separated, and must not hang trying
    same = {'a': {'name': 'X // Y'}, 'b': {'name': 'X // Y'}}
    assert len(runner.distinguish(same, list(same))) == 2


def test_details_editor_saves_only_on_s_and_only_a_valid_record():
    """edit_investor is the only UI path that writes config/investor.json, and
    a bad write there silently changes every plan level. Driven end to end on a
    temp file: q leaves the file alone, an invalid record is refused and the
    loop goes on, the last account cannot be removed, a duplicate name is
    refused, and '2,3' — every other prompt in run.py takes a list, and this
    was the first thing typed at it — walks both rows before one save."""
    real_ask, real_menu = runner.ask, runner._ask_menu

    def drive(menu, fields=(), rec=None):
        with _temp_investor(None) as f:
            if rec is not None:
                f.write_text(json.dumps(rec))
            before = f.read_bytes()
            try:
                asked = []
                script = _scripted(menu)
                runner._ask_menu = lambda *a: asked.append(1) or script(*a)
                runner.ask = _scripted(list(fields))
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    runner.edit_investor()
            finally:
                runner.ask, runner._ask_menu = real_ask, real_menu
            assert len(asked) == len(menu), f'the loop ended early: {asked}'
            return before, f.read_bytes(), out.getvalue()

    before, after, _ = drive(['2', 'q'], ['99'])
    assert after == before, 'q must leave the file untouched'

    age = investor.DEFAULTS['age']
    before, after, out = drive(['2', 's', 'q'], [str(age - 1)])
    assert after == before and 'not saved' in out, out

    import copy
    one = copy.deepcopy(investor.DEFAULTS)
    one['social_security'] = None
    keep = next(iter(one['accounts']))
    one['accounts'] = {keep: one['accounts'][keep]}
    _, after, out = drive(['x5', 's'], rec=one)
    assert 'at least one account' in out, out
    assert list(json.loads(after)['accounts']) == [keep]

    before, after, out = drive(['a', 'q'], [keep])
    assert 'already exists' in out and after == before, out

    _, after, out = drive(['2,3', 's'], ['96', '66'])
    saved = json.loads(after)
    assert (saved['plan_to_age'], saved['retirement_age']) == (96, 66), saved
    assert 'saved' in out, out


def test_the_details_field_parsers_read_what_was_typed():
    """Every field on the details screen goes through one of these, and what
    they return is written to config/investor.json. '0.5%' was stored as 0.5
    — 50%/yr growth, inside validation's +/-50% band, so nothing said so."""
    R = runner
    # A '%' always means percent; a bare number from 1 up is read as one too.
    for typed, want in (('0.5%', 0.005), ('1%', 0.01), ('1', 0.01),
                        ('2.9', 0.029), ('0.029', 0.029), ('-1%', -0.01)):
        assert R._rate(typed, 0.03) == want, (typed, R._rate(typed, 0.03))
    for typed, want in (('300000', 300_000.0), ('300,000', 300_000.0),
                        ('$300k', 300_000.0), ('0.5m', 500_000.0), ('', 7.0)):
        assert R._money(typed, 7.0) == want, typed
    assert R._whole('', 64) == 64 and R._whole('70', 64) == 70
    assert all(R._whole(t, 64) is None for t in ('none', 'Never', '-'))
    assert R._claim_age('', 67.0) == 67.0 and R._claim_age('62', 67.0) == 62.0
    for typed, want in (('75', 75.0), ('75%', 75.0), ('0.75', 75.0), ('', 90.0)):
        assert R._scheduled_pct(typed, 90.0) == want, typed
    assert R._monthly_benefit('5,000', 0.0) == 5_000.0
    with contextlib.redirect_stdout(io.StringIO()):
        for parse, typed in ((R._claim_age, '61'), (R._claim_age, '71'),
                             (R._scheduled_pct, '101'), (R._whole, 'soon'),
                             (R._monthly_benefit, '60,000'), (R._rate, 'x')):
            try:
                parse(typed, None)
                raise AssertionError(f'{parse.__name__}({typed!r}) must refuse')
            except ValueError:
                pass


def test_rate_parser_does_not_write_float_noise():
    """2.9/100 is 0.028999999999999998, which landed in the saved JSON."""
    assert runner._rate('2.9', 0.0) == 0.029
    assert runner._rate('4.1', 0.0) == 0.041
    assert runner._rate('3', 0.0) == 0.03
    assert runner._rate('0.029', 0.0) == 0.029        # already a decimal
    assert runner._rate('', 0.055) == 0.055           # blank keeps


def test_library_labels_never_claim_a_holding_is_absent():
    """145 names no sleeves and holds them; "no 12VFMO" would be a lie.

    A row with nothing distinguishing in its name falls back to its own last
    segment rather than to an inference from what the modal name has and it
    does not.
    """
    P = {'140': {'name': '70D:30I // 100:0 // 5C // 12AVUV // 12VFMO'},
         '145': {'name': '70D:30I // 100:0 // VG-only (mirrors 140)'},
         '100': {'name': '70D:30I // 100:0 // 5C // 12AVUV'}}
    got = runner.distinguish(P, list(P))
    assert got['145'] == 'VG-only (mirrors 140)', got['145']
    assert got['100'] == '12AVUV', got['100']
    assert not any('no ' in v for v in got.values()), got


def test_paths_are_counted_at_the_real_choke_point():
    """Accumulation calls solver._paths directly, bypassing solver.simulate.

    Counting in `simulate` therefore reported ZERO paths for an allocation run:
    no per-path cost, one undifferentiated phase, and no recalibration of the
    estimate the run had just tested.
    """
    src = inspect.getsource(solver._paths)
    assert 'record_paths' in src, 'the choke point must count its own paths'
    assert 'record_paths' not in inspect.getsource(solver.simulate), \
        'simulate must not double-count what _paths already recorded'
    instrument.enable(True)
    instrument.reset()
    try:
        with instrument.timed('accumulation'):
            instrument.record_paths(400, True)
        paths, evals = instrument.totals()
        assert (paths, evals) == (400, 1), (paths, evals)
    finally:
        instrument.enable(False)


def test_grid_view_offers_a_paired_comparison():
    """The matrix printed point estimates and TOP 3 printed UNPAIRED intervals.

    141 read 66.85M [63.80, 69.86] against 140's 65.31M [62.94, 68.10] — 88%
    overlap, which reads as "indistinguishable" and is not what paired data
    says. The flat table has always shown the paired interval; the grid, which
    is what a multi-portfolio run actually prints, dropped it.
    """
    assert '_matrix_pairs' in inspect.getsource(report_retirement._matrix)
    rng = np.random.default_rng(7)
    common = rng.lognormal(3.0, 0.9, 800)
    res = {'A': {'finals': common * 1.06, 'median': 1.0},
           'B': {'finals': common, 'median': 1.0}}
    lo, hi = report_retirement.engine.paired_bootstrap_diff(res['A']['finals'],
                                             res['B']['finals'], np.median)
    assert lo > 0, 'a 6% paired edge under common random numbers must resolve'
    m_lo, m_hi = report_retirement.engine.bootstrap_ci(res['A']['finals'], np.median)
    n_lo, n_hi = report_retirement.engine.bootstrap_ci(res['B']['finals'], np.median)
    assert m_lo < n_hi and n_lo < m_hi, \
        'and the unpaired intervals must overlap, which is the whole point'


def test_best_is_qualified_when_the_margin_is_noise():
    """`best: 141` was printed for a 50.7%-of-paths coin flip.

    On the 5,000-path run that prompted this, 141 led 140 66.85M to 65.31M with
    a paired interval of [-2.22, +5.02]M. The headline named a winner the data
    does not support, while the real result in the same run — 141 over the
    Vanguard-only mirror by +22 to +29M — was never stated.
    """
    rng = np.random.default_rng(11)
    common = rng.lognormal(3.0, 0.9, 900)
    spec = {'mode': 'accumulation'}
    # A shared market with INDEPENDENT idiosyncratic legs, which is what 141 and
    # 140 are. A uniform rescaling would not do: under common random numbers it
    # is a deterministic difference and resolves at any sample size.
    noise = {'A': {'finals': common * rng.lognormal(0, 0.45, 900) * 1.02},
             'B': {'finals': common * rng.lognormal(0, 0.45, 900)}}
    tie = report_retirement._best_caveat([(1.02, 'A'), (1.0, 'B')], noise, spec)
    assert 'NOT resolved' in tie, tie

    real = {'A': {'finals': common * 1.30}, 'B': {'finals': common}}
    won = report_retirement._best_caveat([(1.3, 'A'), (1.0, 'B')], real, spec)
    assert 'beats B' in won and 'NOT resolved' not in won, won

    assert report_retirement._best_caveat([], {}, spec) == ''
    assert report_retirement._best_caveat([(1.0, 'A')], noise, spec) == ''


def test_grid_detail_shows_the_risk_columns_it_already_paid_for():
    """The grid listed 12 strategies, detailed 3, and printed no risk column.

    Accumulation asks the workers for the full monthly return series
    (light=False) SOLELY so Sortino and CAGR can be reported — the grid bought
    that and threw it away, while the flat table has always shown both.
    """
    src = inspect.getsource(report_retirement._matrix)
    assert 'twr_cagr' in src and 'sortino_median' in src, \
        'the grid must report the risk columns it computes'
    assert 'TOP 3' not in src, 'detail is no longer an arbitrary top 3'
    assert '_sortino_note' in src
    assert '_sortino_note' in inspect.getsource(report_retirement._flat_table), \
        'one implementation of the caveat, used by both tables'


def test_sortino_caveat_names_a_few_and_counts_many():
    """Naming every unresolved row is useful at 3 strategies, unreadable at 12."""
    rng = np.random.default_rng(3)
    base = rng.normal(0.9, 0.25, 600)
    res = {f'S{i}': {'sortino': base + rng.normal(0, 0.002, 600)} for i in range(9)}
    res['S0']['sortino'] = base

    def note(keys):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report_retirement._sortino_note(keys, res, keys[0])
        return buf.getvalue()

    few = note(['S0', 'S1', 'S2'])
    assert 'S1, S2' in few, few
    many = note([f'S{i}' for i in range(9)])
    assert 'of the 8 other rows' in many, many
    assert 'S1, S2' not in many, many
    assert note(['S0']) == '', 'a single row has nothing to separate'


def test_investor_split_reproduces_the_old_schedule_exactly():
    """Splitting the accounts must not move a single number.

    The old schedule was one 401k line at 24,000/2.9%/stop 60; it is now two
    halves with the same rate and stop age, plus an HSA at zero. That is
    arithmetically identical, and this asserts it at every age in the horizon
    rather than trusting the argument.
    """
    old = {'frequency': 'annual', 'accounts': {
        '401k': {'annual_amount': 24000, 'growth_rate': 0.029, 'stop_age': 60},
        # 80 until 2026-10-08, when the default moved before retirement
        'IRA': {'annual_amount': 7500, 'growth_rate': 0.041, 'stop_age': 64},
        'taxable': {'annual_amount': 6000, 'growth_rate': 0.03, 'stop_age': 60}}}
    new = investor.contribution_schedule(investor.DEFAULTS)
    for age in range(37, 96):
        a = pa.calculate_annual_contribution(age, old, 0.03)
        b = pa.calculate_annual_contribution(age, new, 0.03)
        assert abs(a - b) < 1e-9, f"age {age}: {a} != {b}"
    assert investor.total_balance(investor.DEFAULTS) == 300000.0


def test_contribution_growth_starts_from_a_named_age():
    """`years_elapsed = age - 36` was a bare literal in no config anywhere.

    It encodes when the amounts were last trued up, not the investor's age, so
    entering current amounts alongside a bumped age gave them a year of growth
    they already contained. The default must stay 36 or every record written
    before the field existed changes meaning.
    """
    sched = {'frequency': 'annual', 'accounts': {
        'a': {'annual_amount': 1000, 'growth_rate': 0.10, 'stop_age': None}}}
    assert abs(pa.calculate_annual_contribution(38, sched) - 1000 * 1.10 ** 2) < 1e-9
    named = dict(sched, measured_at_age=38)
    assert abs(pa.calculate_annual_contribution(38, named) - 1000.0) < 1e-9
    assert abs(pa.calculate_annual_contribution(40, named) - 1000 * 1.10 ** 2) < 1e-9


def test_horizon_and_initial_value_are_derived_never_stored():
    """Both were stored, and both had already drifted.

    `time_horizon_years` was a literal 58, correct only while age was 37 —
    `end_age = age + time_horizon_years`, so bumping age moved the plan to 96.
    Portfolio Analysis.py's copy said 50, i.e. age 87, and nothing noticed.
    """
    import copy
    assert 'time_horizon_years' not in investor.DEFAULTS
    assert 'initial_value' not in investor.DEFAULTS
    inv = copy.deepcopy(investor.DEFAULTS)
    inv['age'] = 45
    cfg = {'age': inv['age'], 'time_horizon_years': inv['plan_to_age'] - inv['age']}
    assert cfg['age'] + cfg['time_horizon_years'] == 95, 'the plan must still end at 95'
    live = runner.default_config()
    assert live['age'] + live['time_horizon_years'] == investor.load()['plan_to_age']
    assert live['initial_value'] == investor.total_balance(investor.load())


def test_investor_validation_refuses_an_incoherent_record():
    import copy, pathlib
    ok = copy.deepcopy(investor.DEFAULTS)
    assert investor.validate(ok) == []

    def why(**over):
        return ' '.join(investor.validate(dict(copy.deepcopy(ok), **over)))

    assert 'retirement_age' in why(retirement_age=30)          # below age
    assert 'plan_to_age' in why(plan_to_age=50)                # below retirement
    assert 'future' in why(contributions_measured_at_age=99)   # measured ahead
    assert 'at least one account' in why(accounts={})
    assert 'tax must be one of' in why(
        accounts={'x': dict(ok['accounts']['taxable'], tax='sheltered')})
    assert 'growth_rate' in why(
        accounts={'x': dict(ok['accounts']['taxable'], growth_rate=12)})

    # And an invalid record must never reach disk.
    with tempfile.TemporaryDirectory() as d:
        target = Path(d) / 'investor.json'
        try:
            investor.save(dict(ok, plan_to_age=50), target)
            raise AssertionError('save accepted an incoherent record')
        except ValueError:
            pass
        assert not target.exists(), 'a refused save must not create the file'


def test_investor_load_is_side_effect_free_and_falls_back():
    """Reading must not create the file: otherwise a validation failure can be
    laundered into a written config."""
    with tempfile.TemporaryDirectory() as d:
        missing = Path(d) / 'nope.json'
        assert investor.load(missing) == investor.DEFAULTS
        assert not missing.exists()
        broken = Path(d) / 'broken.json'
        broken.write_text('{ not json')
        assert investor.load(broken) == investor.DEFAULTS


def test_investor_fallback_is_loud():
    """DEFAULTS are seeded guesses, not anyone's real numbers.

    A missing, unreadable, or invalid file would otherwise produce a complete
    and plausible-looking run on the wrong balances with nothing on screen to
    say so — the same silent-wrong-answer class as everything else here.
    """
    with tempfile.TemporaryDirectory() as d:
        dd = Path(d)

        def said(path):
            investor._WARNED.clear()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                investor.load(path)
            return buf.getvalue()

        assert 'not found' in said(dd / 'nope.json')
        (dd / 'bad.json').write_text('{ not json')
        assert 'could not be read' in said(dd / 'bad.json')
        (dd / 'inv.json').write_text(
            json.dumps(dict(investor.DEFAULTS, age=200)))
        out = said(dd / 'inv.json')
        assert 'does NOT validate' in out and 'age' in out
        # valid JSON of the wrong SHAPE crashed with AttributeError instead
        (dd / 'list.json').write_text('[]')
        assert 'could not be read' in said(dd / 'list.json')
        (dd / 'acct.json').write_text(
            json.dumps(dict(investor.DEFAULTS, accounts={'x': 5})))
        out = said(dd / 'acct.json')
        assert 'does NOT validate' in out and 'x' in out

        # and it says it ONCE — default_config() loads several times a run
        investor._WARNED.clear()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            for _ in range(4):
                investor.load(dd / 'nope.json')
        assert buf.getvalue().count('not found') == 1

        # a good file stays silent
        good = dd / 'good.json'
        investor.save(investor.DEFAULTS, good)
        assert said(good) == ''


def test_no_default_account_contributes_into_retirement():
    """stop_age is the LAST age an account contributes (engine: age <= stop_age),
    and withdrawals start AT the retirement age. The Roth IRA defaulted to 80,
    fifteen years of contributions alongside withdrawals, from the accumulation-
    only version. Roth contributions need earned income."""
    ret = investor.DEFAULTS['retirement_age']
    for name, a in investor.DEFAULTS['accounts'].items():
        assert a['stop_age'] is not None and a['stop_age'] < ret, (name, a['stop_age'])


def test_contributions_into_retirement_are_flagged_not_refused():
    """An account still contributing at the retirement age overlaps the
    withdrawals. That is usually a mistake, so loading says so — but it can be
    real (part-time work), so it is a warning and not a validation failure."""
    import copy, pathlib
    inv = copy.deepcopy(investor.DEFAULTS)
    assert investor.overlaps(inv) == []
    inv['accounts']['roth_ira']['stop_age'] = inv['retirement_age']   # one year over
    inv['accounts']['taxable']['stop_age'] = None                      # never stops
    inv['accounts']['hsa'].update(stop_age=90, annual_amount=0.0)      # nothing to add
    assert investor.overlaps(inv) == ['roth_ira', 'taxable'], investor.overlaps(inv)
    assert investor.validate(inv) == [], investor.validate(inv)
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 'inv.json'
        f.write_text(json.dumps(inv))
        investor._WARNED.clear()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            investor.load(f)
            investor.load(f)
        out = buf.getvalue()
    assert 'roth_ira' in out and 'taxable' in out and 'retirement' in out, out
    assert out.count('roth_ira') == 1, 'say it once a run'


def test_your_details_says_the_stop_age_is_the_last_year():
    """'contribute until age 65' reads as stopping AT 65; the engine contributes
    through 65. The prompt names the inclusive meaning, a new account defaults
    to the year before retirement, and the screen marks an overlap."""
    import copy
    inv = copy.deepcopy(investor.DEFAULTS)
    assert runner._new_account(inv)['stop_age'] == inv['retirement_age'] - 1
    inv['accounts']['x'] = runner._new_account(inv)
    _, out = _drive(lambda: runner._edit_account(inv, 'x'), ['', '', '', '', ''])
    assert 'last age you contribute' in out, out
    inv['accounts']['roth_ira']['stop_age'] = inv['retirement_age']
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runner._investor_screen(inv)
    row = next(l for l in buf.getvalue().splitlines() if 'roth_ira' in l)
    assert '↻' in row, row
    assert 'into retirement' in buf.getvalue(), buf.getvalue()


def test_investor_facts_reach_run_provenance():
    """Editing your balance between runs used to leave --diff claiming
    "provenance identical", which was then false."""
    import copy
    cfg = runner.default_config()
    prov = persistence.provenance(cfg)
    for field in ('investor_age', 'investor_plan_to_age', 'investor_initial_value',
                  'investor_first_year_contribution', 'investor_accounts',
                  'investor_contributions_measured_at_age'):
        assert field in prov, field
        assert not isinstance(prov[field], (dict, list)), \
            f"{field} must be a flat scalar — diff_runs compares field by field"

    # A stop_age change that leaves year one untouched must still be visible.
    moved = copy.deepcopy(cfg)
    moved['contribution_schedule'] = copy.deepcopy(cfg['contribution_schedule'])
    moved['contribution_schedule']['accounts']['taxable']['stop_age'] = 70
    after = persistence.provenance(moved)
    assert after['investor_first_year_contribution'] == \
        prov['investor_first_year_contribution'], 'year one is unchanged by design'
    assert after['investor_accounts'] != prov['investor_accounts'], \
        'the fingerprint is what catches it'


def test_scoring_a_plan_with_no_retirement_years_says_why():
    """score() returned {'success_rate', 'spend'} for zero retirement years,
    and every renderer then died on the first other key it read, far from the
    cause. Validated specs cannot reach it; anything that does must be told."""
    raw = {'spending': np.zeros((5, 0)), 'depleted': np.zeros(5, dtype=bool)}
    try:
        solver.score(raw, {}, 100e3)
    except ValueError as e:
        assert 'no retirement years' in str(e), e
    else:
        raise AssertionError('an empty retirement must raise, not half-score')


def test_max_drawdown_is_the_worst_peak_to_trough():
    # rises to 100, falls to 60 (-40%), recovers, then a shallower dip
    wealth = np.array([[50.0, 100.0, 60.0, 90.0, 81.0]])
    assert abs(report_retirement._max_drawdowns(wealth)[0] - (-40.0)) < 1e-9
    flat = np.array([[10.0, 11.0, 12.0]])          # never falls
    assert abs(report_retirement._max_drawdowns(flat)[0]) < 1e-9


def test_figures_take_reduced_stats_not_the_raw_matrices():
    """`returns` and the wealth paths are n_sims x ~700 floats EACH.

    At 5,000 paths that is 56MB per strategy, and they were retained for every
    strategy purely so figures could be drawn at the end — near 700MB on a
    12-strategy run, for two scalars and a median.
    """
    from montecarlo import report
    src = inspect.getsource(report.build_figures)
    assert "results[label]['returns']" not in src
    assert "results[label]['paths']" not in src
    assert "results[label]['twr_cagr']" in src
    assert "results[label]['median_path']" in src
    # and the accumulation result must not carry them either
    acc = inspect.getsource(report_retirement.evaluate_strategies)
    body = acc[acc.index("if spec['mode'] == 'accumulation':"):acc.index('continue')]
    assert "'returns':" not in body and "'paths':" not in body


def test_drawdown_chart_title_uses_the_real_horizon():
    """It said "50 Years" on a 58-year run — a hardcoded number of exactly the
    kind investor.py exists to eliminate."""
    from montecarlo import report
    src = inspect.getsource(report.build_figures)
    assert '50 Years' not in src
    assert "config['time_horizon_years']" in src


def test_factor_files_parse_with_the_risk_free_rate():
    """The factor files are what the factor build-up needs.

    RF is why these files rather than corner-construction from the 25-portfolio
    sets: `expected_return = rf + sum(beta * premium)` has no other source for
    it. Each file concatenates a monthly and an annual table, so the same
    duplicate guard applies as everywhere else here.
    """
    import data_collection_regime_model as dcm
    ref = Path(__file__).resolve().parent.parent / 'reference'
    # git-tracked: a moved file must fail this test, not turn it into a no-op
    assert (ref / 'F-F_Research_Data_5_Factors_2x3.csv').exists(), ref
    kf = dcm.KenFrenchParser(str(ref))
    f5 = kf.parse_ff5_factors()
    assert list(f5.columns) == ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'rf']
    assert f5.index.duplicated().sum() == 0
    assert f5.index.is_monotonic_increasing
    assert str(f5.index[0])[:7] == '1963-07', str(f5.index[0])
    # the annual block carries 4-digit years and must not survive as months
    assert len(f5) > 700 and len(f5) < 900, len(f5)
    # sane magnitudes: monthly decimals, not percents
    assert 0.0 < f5['mkt_rf'].mean() * 12 < 0.20
    assert 0.0 < f5['rf'].mean() * 12 < 0.15

    mom = kf.parse_momentum_factor()
    assert list(mom.columns) == ['mom']
    assert mom.index.duplicated().sum() == 0
    assert str(mom.index[0])[:7] == '1927-01', str(mom.index[0])


def test_every_academic_proxy_is_monthly_decimals_on_its_window():
    """The Ken French and AQR proxies are most of the history of eight funds,
    and only KenFrenchParser had a test. AQRParser carried a 100x unit bug that
    scaled 84% of JQUA's history and hid behind one plausible number. Built
    offline from the committed reference/ files: each proxy covers exactly its
    window, month by month, in monthly decimals with plausible risk and return,
    and each blend's weights sum to 1."""
    import data_collection_regime_model as dcm
    ref = Path(__file__).resolve().parent.parent / 'reference'
    for t, w in dcm.KF_BLENDS.items():
        assert t in dcm.KF_TICKERS, t
        assert abs(sum(w.values()) - 1.0) < 1e-9, (t, sum(w.values()))
    qmj = ref / 'Quality Minus Junk 10 QualitySorted Portfolios Monthly.xlsx'
    assert qmj.exists(), qmj                    # git-tracked, as above
    aqr = dcm.AQRParser(ref)
    q = aqr.parse_quality_10()['P10 (high quality)'].dropna()
    assert 0.03 < q.mean() * 12 < 0.25, f'AQR quality scaled wrong? {q.mean()}'
    pb = dcm.ProxyBuilder(dcm.KenFrenchParser(str(ref)), aqr)
    for t, c in dcm.KF_TICKERS.items():
        p = pb.build_proxy(t)
        months = pd.period_range(c['start'], c['end'], freq='M')
        assert list(p.index.to_period('M')) == list(months), (t, p.index[[0, -1]])
        assert p.notna().all(), t
        mean, vol = p.mean() * 12, p.std() * 12 ** 0.5
        assert 0.03 < mean < 0.25 and 0.08 < vol < 0.30, (t, mean, vol)


def test_every_academic_proxy_is_net_of_its_costs():
    """COST_ADJUSTMENTS stores expense and drag as NEGATIVE annual rates, and
    every builder SUBTRACTED them, so each academic proxy ran ABOVE the gross
    blend by its costs instead of below: VFMO +1.23%/yr where -1.23% was meant,
    worth +2.06pp on its historical mean. Live from the first commit until
    2026-10-26; the plausibility bounds above pass either sign. Each proxy must
    sit exactly its stated costs BELOW the same blend built with no costs."""
    import data_collection_regime_model as dcm
    from unittest import mock
    ref = Path(__file__).resolve().parent.parent / 'reference'

    def build_all():
        pb = dcm.ProxyBuilder(dcm.KenFrenchParser(str(ref)), dcm.AQRParser(ref))
        return {t: pb.build_proxy(t) for t in dcm.KF_TICKERS}

    net = build_all()
    free = {t: {'expense': 0.0, 'drag': 0.0} for t in dcm.COST_ADJUSTMENTS}
    with mock.patch.dict(dcm.COST_ADJUSTMENTS, free):
        gross = build_all()
    assert set(dcm.COST_ADJUSTMENTS) == set(dcm.KF_TICKERS)
    for t, c in dcm.COST_ADJUSTMENTS.items():
        assert c['expense'] < 0 and c['drag'] <= 0, (t, c)  # a cost, not a credit
        gap = (net[t] - gross[t]) * 12
        assert np.allclose(gap, c['expense'] + c['drag'], atol=1e-12), (
            t, float(gap.mean()), c)


def test_the_proxy_layer_never_outranks_the_real_fund():
    """The middle proxy layer had NO end date and won under keep='first'.

    Measured on MTUM: the ^GSPC layer ran 2013-05..2026-09, exactly overlapping
    all 161 months of real MTUM data, and the proxy won every one — 161 months
    of the actual fund replaced by a price index returning 12.37%/yr against
    MTUM's own 15.57%. Every Ken French-blended ticker was affected. The splice
    must truncate each layer and let the most authoritative source win.
    """
    import data_collection_regime_model as dcm
    idx = pd.date_range('2020-01-31', periods=6, freq='ME')
    kf = pd.Series([0.01] * 4, index=idx[:4])
    proxy = pd.Series([0.02] * 5, index=idx[1:])           # no end date
    actual = pd.Series([0.03] * 3, index=idx[3:])
    out, n = dcm.splice_layers(kf, proxy, actual)
    assert list(out.values) == [0.01, 0.02, 0.02, 0.03, 0.03, 0.03], list(out)
    assert out.index.is_monotonic_increasing and out.index.is_unique
    assert n == 2, 'proxy months are counted only where the proxy is used'
    # Layers that do not arrive in date order still come out in order.
    out, _ = dcm.splice_layers(proxy[-1:], kf[:2], pd.Series([], dtype=float))
    assert out.index.is_monotonic_increasing, list(out.index)

    # No real fund yet: the proxy beats the academic blend on overlap.
    out, n = dcm.splice_layers(kf, proxy, pd.Series([], dtype=float))
    assert list(out.values) == [0.01, 0.02, 0.02, 0.02, 0.02, 0.02], list(out)
    # No middle layer: academic then real.
    out, n = dcm.splice_layers(kf, pd.Series([], dtype=float), actual)
    assert list(out.values) == [0.01] * 3 + [0.03] * 3 and n == 0, list(out)

    # The blended-proxy splice goes through the same two-stage helper.
    out, n = dcm.splice_with_proxy(proxy, actual)
    assert list(out.values) == [0.02, 0.02, 0.03, 0.03, 0.03] and n == 2


def test_no_proxy_is_a_price_only_index():
    """PROXIES['VOO'] was '^GSPC', the S&P 500 PRICE index — no dividends.

    Measured 1970-02..2026-07: 8.25%/yr against the Ken French US market TOTAL
    return of 11.28%, a 3.03pp gap covering ~72% of VOO's history. VOO is
    engine.MARKET_PROXY, so it dragged every MC_PREMIUM_SCALE result down and
    inflated every "premium over VOO" figure by ~2.5pp.

    A '^' prefix on Yahoo means an index, and indices there are price-only.
    Proxies must be FUNDS, whose adjusted closes include distributions.
    """
    import data_collection_regime_model as dcm
    bad = {t_: p for t_, p in dcm.PROXIES.items() if p and str(p).startswith('^')}
    assert not bad, f'price-only index proxies: {bad}'
    for blend in dcm.BLENDED_PROXIES.values():
        for name_ in (blend.get('components') or blend):
            assert not str(name_).startswith('^'), name_


def test_actual_history_is_not_discarded_for_a_proxy():
    """The Ken French splice fetched actual data from a hardcoded 2010-01-01.

    Real fund returns before that were thrown away in favour of a proxy: VTV
    lost 72 months of its own history (inception 2004-01) to the ^GSPC price
    index, and VBR lost 72 to VISVX. Fetching from inception also collapses
    VTV's price-index exposure to zero, since its KF window ends exactly where
    the real fund begins.
    """
    import data_collection_regime_model as dcm
    src = inspect.getsource(dcm)
    assert "start='2010-01-01'" not in src, \
        'actual history must be fetched from inception, not a hardcoded year'


def test_implementation_cost_comes_from_turnover_not_the_old_drag():
    """§6e first proposed COST_ADJUSTMENTS' `drag`. Turnover overturned it.

    The Avantis funds turn over 4-7% a year and cannot be paying the
    0.20-0.40%/yr `drag` assigns them — 3 to 40 times what their trading
    implies. `drag` was calibrated to close a Ken French BLEND vs fund gap,
    which bundles implementation cost with exposure mismatch, and beta now
    handles exposure mismatch.
    """
    low = factors.costs_for('X', {'expense': 0.001, 'turnover': 0.06})[1]
    high = factors.costs_for('Y', {'expense': 0.001, 'turnover': 1.16})[1]
    assert low < 0.0005, low                 # a 6%-turnover fund is ~free
    assert 0.001 < high < 0.003, high        # a 116% one pays ~0.17%
    assert high > low * 10

    # a reported zero is a data gap, not a fund that never trades
    floored = factors.costs_for('Z', {'expense': 0.003, 'turnover': 0.0})[1]
    assert floored == factors.MIN_TURNOVER * factors.ROUND_TRIP
    assert floored > 0

    # no turnover at all -> no implementation charge, not a guess
    assert factors.costs_for('W', {'expense': 0.003})[1] == 0.0


def test_the_real_cash_rate_survives_an_inflation_change():
    """RF_FORWARD and INFLATION used to be set INDEPENDENTLY.

    A nominal 4.35% against 3% inflation is 1.31% real; raising MC_INFLATION to
    4% left the nominal alone and dropped the real rate to 0.34%. So an
    inflation sensitivity sweep was invisibly a real-rate sweep too, and the
    assumption that actually matters moved without anyone touching it.

    The real rate is now the input and the nominal is derived, so the two
    cannot drift apart.
    """
    root = Path(__file__).resolve().parent.parent
    code = ("import sys; sys.path.insert(0, %r); from montecarlo import factors, engine; import json; "
            "print(json.dumps([factors.RF_FORWARD, engine.INFLATION]))" % str(root))
    base = {k: v for k, v in os.environ.items() if k != 'MC_RF_FORWARD'}
    levels = ('0.02', '0.03', '0.04')
    reals = {}
    for infl, out in zip(levels, _children(
            [(code, dict(base, MC_INFLATION=i)) for i in levels], cwd=str(root))):
        nom, i = json.loads(out.stdout.strip().splitlines()[-1])
        assert i == float(infl), f'MC_INFLATION={infl} did not reach the child: {i}'
        reals[infl] = (1 + nom) / (1 + i) - 1
    assert max(reals.values()) - min(reals.values()) < 1e-9, reals
    assert abs(reals['0.03'] - factors._RF_REAL_DEFAULT) < 1e-6, reals['0.03']


def test_rf_forward_is_late_bound_not_frozen_at_import():
    """`rf=RF_FORWARD` as a DEFAULT ARGUMENT froze the value at import.

    MC_RF_FORWARD could never reach it, and a test that patched the module
    global measured nothing — it returned byte-identical medians at 2%, 4.35%
    and 6% and looked like a clean "level-only" result. Resolve it inside the
    function instead.
    """
    for fn in (factors.expected_return, factors.build_nominal):
        sig = inspect.signature(fn)
        assert sig.parameters['rf'].default is None, \
            f'{fn.__name__} freezes rf at import'
    fit = {'method': 'factor', 'regions': ['us'], 'adj_r2': 0.95,
           'beta': {'us': {f: (1.0 if f == 'mkt_rf' else 0.0)
                           for f in factors.FACTORS}}}
    a = factors.expected_return(fit, rf=0.02)
    b = factors.expected_return(fit, rf=0.06)
    assert abs((b - a) - 0.04) < 1e-12, (a, b)


def test_the_haircut_is_level_only():
    """Measured 2026-09-10, not inferred: identical ranking at haircut 1.0, 0.7
    and 0.4 while 141's median wealth falls 103.4M -> 30.7M -> 9.5M.

    mu = rf + h*SUM(beta*p) - costs compresses every portfolio toward rf as h
    falls, which is mechanically what MC_PREMIUM_SCALE does. So the haircut is a
    LEVEL knob and no allocation decision depends on it — which is why it does
    not gate adopting the factor method.
    """
    full = factors.premia_for('full history', haircut=1.0)
    half = factors.premia_for('full history', haircut=0.5)
    for r in factors.PREMIA:
        for f in factors.FACTORS:
            assert abs(half[r][f] - full[r][f] * 0.5) < 1e-12, (r, f)
    # it scales the MARKET premium too — unlike a scenario, which never does
    assert half['us']['mkt_rf'] < full['us']['mkt_rf']
    dead = factors.premia_for('value dead', haircut=1.0)
    assert dead['us']['mkt_rf'] == full['us']['mkt_rf']


def test_derived_bands_are_no_longer_inverted():
    """PARAM_STD_OVERRIDE had gone backwards: the speculative premia carried the
    TIGHTEST bands.

    Measured before: AVLV 0.07 x vol, AVDE 0.09, AVUV 0.10 against VOO 0.16 and
    VEA 0.19 — the model was more confident about a 2021-inception fund built
    from a hand-weighted proxy than about the S&P 500. Nobody chose that; it is
    what an override table does over time. Derived from
    Var(mu) = SUM[beta^2 Var(p) + p^2 Var(beta)] + Var(alpha) the order comes
    out right, because a large factor loading multiplies an uncertain premium.
    """
    us = {f: 0.0 for f in factors.FACTORS}
    def fit(**loads):
        b = dict(us, **loads)
        return {'method': 'factor', 'regions': ['us'], 'beta': {'us': b},
                'beta_se': {'us': {f: 0.01 for f in factors.FACTORS}},
                'alpha_se': 0.005, 'adj_r2': 0.95, 'n_months': 200}
    plain = factors.parameter_std(fit(mkt_rf=1.0))
    tilted = factors.parameter_std(fit(mkt_rf=1.0, hml=0.5, smb=0.5))
    assert tilted > plain, (tilted, plain)
    # and a bigger tilt is wider still — monotone in exposure
    more = factors.parameter_std(fit(mkt_rf=1.0, hml=0.9, smb=0.9))
    assert more > tilted, (more, tilted)
    # not covered by the factor model -> no derived band, caller keeps its own
    assert factors.parameter_std({'method': 'spread', 'spread': 0.01}) is None
    assert factors.parameter_std(None) is None


def test_bands_are_left_alone_where_the_factor_model_does_not_reach():
    """A spread or historical ticker is not a factor forecast, so the
    derivation does not apply and its existing band must survive."""
    hist = {'AAA': 0.02, 'BBB': 0.03, 'CCC': 0.04}
    fits = {
        'AAA': {'method': 'factor', 'regions': ['us'],
                'beta': {'us': {f: (1.0 if f == 'mkt_rf' else 0.0)
                                for f in factors.FACTORS}},
                'beta_se': {'us': {f: 0.01 for f in factors.FACTORS}},
                'alpha_se': 0.004, 'adj_r2': 0.95, 'n_months': 200},
        'BBB': {'method': 'spread', 'spread': 0.0158},
    }
    out, _ = factors.build_param_std(hist, fits)
    assert out['AAA'] != 0.02, 'a factor ticker must get a derived band'
    assert out['BBB'] == 0.03, 'a bond spread keeps its own band'
    assert out['CCC'] == 0.04, 'no recipe keeps its own band'


def test_mu_method_defaults_to_factor_and_keeps_its_contract():
    """Phase E flipped the default on 2026-09-10. The CONTRACT is what matters.

    The substitution happens on the NOMINAL dict, before real_rate and before
    scale_equity_premia, so engine.EXPECTED_RETURNS stays dict[ticker -> real
    mu] and nothing downstream can tell the difference. That is what lets the
    regression golden master, which injects its own synthetic universe over
    these globals, survive the change untouched — and it is why flipping the
    default did not require re-blessing it.
    """
    assert pa.MU_METHOD == 'factor', 'Phase E default'
    assert isinstance(pa.EXPECTED_RETURNS, dict)
    assert all(isinstance(v, float) for v in pa.EXPECTED_RETURNS.values())
    assert len(pa.EXPECTED_RETURNS) == len(pa.TICKERS)
    # a ticker the factor model does not cover still gets a number
    assert pa.EXPECTED_RETURNS['COST'] > 0
    # read from the environment, not assigned — workers re-import under spawn
    src = inspect.getsource(pa)
    assert "os.environ.get('MC_MU_METHOD'" in src
    assert "os.environ.get('MC_PREMIA_SCENARIO'" in src


def test_build_nominal_falls_back_rather_than_crashing():
    """A missing or partial artifact must degrade, not take a run down."""
    hist = {'AAA': 0.09, 'BBB': 0.05, 'CCC': 0.04}

    out, notes = factors.build_nominal(hist, None)
    assert out == hist
    assert any('no factor_loadings' in n for n in notes)

    fits = {
        'AAA': {'method': 'factor', 'regions': ['us'], 'adj_r2': 0.95,
                'beta': {'us': {f: 0.0 for f in factors.FACTORS}}},
        'BBB': {'method': 'spread', 'spread': 0.0158},
        # CCC absent entirely
    }
    fits['AAA']['beta']['us']['mkt_rf'] = 1.0
    out, notes = factors.build_nominal(hist, fits)
    assert abs(out['AAA'] - (factors.RF_FORWARD
                             + factors.PREMIA['us']['mkt_rf'])) < 1e-9
    assert abs(out['BBB'] - (factors.RF_FORWARD + 0.0158)) < 1e-12
    assert out['CCC'] == 0.04, 'no recipe -> historical'
    assert any('CCC' in n for n in notes)
    # AAA has no expense ratio set, so it must say so rather than pretend free
    assert any('NO EXPENSE RATIO' in n for n in notes)


def _single_stock_fit(beta=0.7, rmw=0.0, alpha=0.09):
    return {'method': 'single_stock', 'regions': ['us'],
            'beta': {'us': {'mkt_rf': beta, 'rmw': rmw}},
            'beta_se': {'us': {'mkt_rf': 0.07, 'rmw': 0.14}},
            'alpha': alpha, 'alpha_se': 0.036, 'adj_r2': 0.38, 'n_months': 240}


def test_fit_single_stock_reads_us_factors_on_the_recent_window():
    """N10: a single stock's recipe is its US factor loadings over the same 20
    years its volatility is measured on — a business 30 years ago was a
    different business, which is why SINGLE_STOCKS gets the rolling window at
    all. Market-only was the first version and dropped COST's profitability
    loading (0.56, t 4.1), a premium the model pays every fund that carries it."""
    rng = np.random.default_rng(0)
    idx = pd.date_range('1990-01-01', periods=400, freq='MS')
    frame = pd.DataFrame({f: rng.normal(0.003, 0.03, 400) for f in factors.FACTORS},
                         index=idx)
    frame['mkt_rf'] = rng.normal(0.006, 0.045, 400)
    frame['rf'] = 0.002
    # beta 1.5 for the old business, 0.7 for the last 240 months
    beta = np.where(np.arange(400) < 160, 1.5, 0.7)
    y = (0.004 + beta * frame['mkt_rf'].values + 0.6 * frame['rmw'].values
         + rng.normal(0, 0.01, 400))
    fit = factors.fit_single_stock(pd.Series(y, index=idx), frame)
    assert fit['method'] == 'single_stock', fit
    assert fit['n_months'] == factors.SINGLE_STOCK_MONTHS == 240, fit['n_months']
    assert fit['regions'] == ['us'] and set(fit['beta']) == {'us'}, fit['beta']
    assert set(fit['beta']['us']) == set(factors.FACTORS), fit['beta']
    # the whole 400 months would read ~1.0; the last 240 read 0.7
    assert abs(fit['beta']['us']['mkt_rf'] - 0.7) < 0.1, fit['beta']
    assert abs(fit['beta']['us']['rmw'] - 0.6) < 0.15, fit['beta']


def test_a_single_stock_earns_its_loadings_plus_a_stated_edge_never_its_alpha():
    """N10: COST kept its 20-year historical mean (14.65% real against VOO's
    7.49%), so the alpha that put it in the universe flowed straight into its
    forecast. Built now as rf + sum(beta x premium) + an edge you state."""
    fits = {'S': _single_stock_fit(beta=0.7, rmw=0.5, alpha=0.09)}
    p = factors.PREMIA['us']
    loaded = 0.7 * p['mkt_rf'] + 0.5 * p['rmw']
    out, _ = factors.build_nominal({'S': 0.18}, fits, edge=0.0)
    assert abs(out['S'] - (factors.RF_FORWARD + loaded)) < 1e-12, out
    out, _ = factors.build_nominal({'S': 0.18}, fits, edge=0.03)
    assert abs(out['S'] - (factors.RF_FORWARD + loaded + 0.03)) < 1e-12, out
    # the haircut reaches its premia as it reaches every fund's
    out, _ = factors.build_nominal({'S': 0.18}, fits, haircut=0.5, edge=0.0)
    assert abs(out['S'] - (factors.RF_FORWARD + 0.5 * loaded)) < 1e-12, out
    # and a scenario that kills a factor kills it here too
    out, _ = factors.build_nominal({'S': 0.18}, fits, scenario='quality dead', edge=0.0)
    assert abs(out['S'] - (factors.RF_FORWARD + 0.7 * p['mkt_rf'])) < 1e-12, out
    # the default edge is the module's, which is zero unless the env says
    assert factors.SINGLE_STOCK_EDGE == 0.0
    assert factors.expected_return(fits['S']) is None, \
        'expected_return is the FACTOR path; a single stock must not slip into it'


def test_a_single_stock_keeps_its_band_and_shares_its_premium_errors():
    """The 0.25 x vol single-stock band is kept (it is about the size of alpha's
    standard error); what changes is that its premium errors are now the ones
    every fund loading on those premia draws, not an independent one."""
    fits = {'S': _single_stock_fit(rmw=0.5)}
    out, _ = factors.build_param_std({'S': 0.047}, fits)
    assert out['S'] == 0.047, 'the single-stock band must survive'
    shares = factors.premium_shares(fits['S'])
    assert shares is not None, 'a single stock must share its premium errors'
    nonzero = {k for k, s in zip(factors.PREMIUM_KEYS, shares) if s}
    assert nonzero == {('us', 'mkt_rf'), ('us', 'rmw')}, nonzero
    assert 0.0 < sum(s * s for s in shares) < 1.0, shares


def test_the_single_stock_gate_flags_a_lead_the_edge_does_not_allow():
    """N10 step 3: a single stock whose mu leads the market proxy by more than
    its beta and the stated edge allow is carrying alpha — fail it."""
    fits = {'S': _single_stock_fit(beta=0.7), 'M': {'method': 'factor'}}
    p = factors.PREMIA['us']['mkt_rf']
    built, voo = factors.RF_FORWARD + 0.7 * p, 0.106
    assert factors.single_stock_problems({'S': built, 'M': voo}, fits, 'M',
                                         stocks={'S'}, edge=0.0) == []
    # the historical mean is exactly what this exists to catch
    assert factors.single_stock_problems({'S': 0.18, 'M': voo}, fits, 'M',
                                         stocks={'S'}, edge=0.0)
    # a stated edge allows its own size over the market and no more
    assert factors.single_stock_problems({'S': voo + 0.03, 'M': voo}, fits, 'M',
                                         stocks={'S'}, edge=0.03) == []
    assert factors.single_stock_problems({'S': voo + 0.05, 'M': voo}, fits, 'M',
                                         stocks={'S'}, edge=0.03)
    # a lead its LOADINGS earn is allowed: profitability is a priced premium
    rich = {'S': _single_stock_fit(beta=1.0, rmw=0.6), 'M': {'method': 'factor'}}
    mkt = factors.RF_FORWARD + p
    lead = mkt + 0.6 * factors.PREMIA['us']['rmw']
    assert lead - mkt > factors.SINGLE_STOCK_GATE_TOL
    assert factors.single_stock_problems({'S': lead, 'M': mkt}, rich, 'M',
                                         stocks={'S'}, edge=0.0) == []
    # no recipe means it fell back to its historical mean: also a failure
    got = factors.single_stock_problems({'S': built, 'M': voo},
                                        {'M': {'method': 'factor'}}, 'M',
                                        stocks={'S'}, edge=0.0)
    assert got and 'S' in got[0], got


def test_production_single_stocks_are_built_not_historical():
    """Every ticker in SINGLE_STOCKS carries a single-stock recipe, and the
    engine's numbers pass the gate."""
    fits = factors.load_fits()
    for t in loaders.SINGLE_STOCKS:
        assert fits[t]['method'] == 'single_stock', (t, fits[t]['method'])
        assert fits[t]['n_months'] <= factors.SINGLE_STOCK_MONTHS, fits[t]['n_months']
    assert factors.single_stock_problems(pa.NOMINAL_EXPECTED_RETURNS, fits,
                                         pa.MARKET_PROXY) == []
    for t in loaders.SINGLE_STOCKS:
        want = factors.single_stock_return(fits[t])
        assert abs(pa.NOMINAL_EXPECTED_RETURNS[t] - want) < 1e-12, (t, want)
    for t in loaders.SINGLE_STOCKS:
        assert t in pa.PREMIUM_SHARES, t


def test_single_stock_edge_is_a_knob_workers_and_reports_carry():
    """MC_SINGLE_STOCK_EDGE is read at import (workers re-import under `spawn`),
    moves only the single stocks, is recorded in provenance and announced when
    moved, and is sweepable. Under MC_MU_METHOD=historical a single stock keeps
    its realised mean: that view is every ticker's own mean by definition."""
    import sensitivity
    assert diagnostics.DEFAULT_ASSUMPTIONS['SINGLE_STOCK_EDGE'] == (0.0, 'MC_SINGLE_STOCK_EDGE')
    assert sensitivity.DIMENSIONS['single_stock_edge'].env == 'MC_SINGLE_STOCK_EDGE'
    assert persistence.provenance()['single_stock_edge'] == pa.SINGLE_STOCK_EDGE == 0.0
    edge = {'MC_SINGLE_STOCK_EDGE': '0.03'}
    runs = _engine_attrs(
        ('engine.NOMINAL_EXPECTED_RETURNS', {}),
        ('engine.NOMINAL_EXPECTED_RETURNS', edge),
        ('engine.NOMINAL_EXPECTED_RETURNS', dict(edge, MC_MU_METHOD='historical')),
        # recorded and announced when MOVED — at the default, 0.0 == 0.0
        # cannot tell a recorded edge from a hard-coded one
        ("[__import__('montecarlo.persistence', fromlist=['_']).provenance()['single_stock_edge'], "
         "[m[0] for m in __import__('montecarlo.diagnostics', fromlist=['_']).moved_assumptions()]]", edge))
    for out, got in runs:
        assert got is not None, out.stderr[-300:]
    (_, base), (_, edged), (_, hist), (out, got) = runs
    assert abs(edged['COST'] - base['COST'] - 0.03) < 1e-12, (edged['COST'], base['COST'])
    assert all(edged[t] == base[t] for t in base if t != 'COST')
    assert hist['COST'] == pa._params['EXPECTED_RETURNS']['COST'], hist['COST']
    assert got is not None, out.stderr[-300:]
    assert got == [0.03, ['SINGLE_STOCK_EDGE']], got


def _children(jobs, cwd=None):
    """[(code, env)] -> [CompletedProcess], every child started at once.

    Each MC_ setting needs a fresh interpreter, because it is read at import,
    but nothing makes one probe wait for another: run in turn they were most of
    the suite's subprocess time, nearly all of it interpreter start-up."""
    procs = [subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, env=env, cwd=cwd)
             for code, env in jobs]
    done = []
    for p in procs:
        out, err = p.communicate()
        done.append(subprocess.CompletedProcess(p.args, p.returncode, out, err))
    return done


def _engine_attrs(*jobs):
    """[(expression, env overrides)] -> [(process, its JSON value or None)],
    each evaluated in its own fresh interpreter, all at once."""
    runs = _children([(("import sys, json; sys.path.insert(0, %r); from montecarlo import engine; "
                        "print(json.dumps(%s))" % (str(ROOT), code)),
                       dict(os.environ, **env)) for code, env in jobs])
    return [(r, json.loads(r.stdout.strip().splitlines()[-1])
             if r.returncode == 0 else None) for r in runs]


def test_uniform_rebalancing_bands_spare_only_the_single_stock():
    """A written policy rebalances on the greater of 25% relative or 5pp
    absolute for every position, with COST capped at 7% (target 5%). The model
    simulated fourteen hand-tuned per-fund pairs, so its results described a
    different policy. MC_REBALANCING_BANDS=uniform simulates the written rule
    ('ips', its old name, still reads); the default stays the model's, so no
    shipped number moves."""
    import sensitivity
    assert pa.REBALANCING_POLICY == 'model'
    assert pa.REBALANCING_BANDS['VOO'] == {'relative': 0.15, 'absolute': 0.05}
    uni = {'MC_REBALANCING_BANDS': 'uniform'}
    (out, got), (bad, _), (moved_out, moved), (_, legacy) = _engine_attrs(
        ("{t: engine.REBALANCING_BANDS[t] for t in ('VOO', 'AVUV', 'BND', 'COST')}", uni),
        ("1", {'MC_REBALANCING_BANDS': 'uniformm'}),
        ("[__import__('montecarlo.persistence', fromlist=['_']).provenance()['rebalancing_bands'], "
         "[m[0] for m in __import__('montecarlo.diagnostics', fromlist=['_']).moved_assumptions()]]", uni),
        ("[engine.REBALANCING_POLICY, engine.REBALANCING_BANDS['VOO']]",
         {'MC_REBALANCING_BANDS': 'ips'}))
    assert got is not None, out.stderr[-300:]
    for t in ('VOO', 'AVUV', 'BND'):
        assert got[t] == {'relative': 0.25, 'absolute': 0.05}, (t, got[t])
    for t in loaders.SINGLE_STOCKS:
        assert got[t] == pa.REBALANCING_BANDS[t], 'a single stock keeps its 7% ceiling'
    assert abs(pa.calculate_band_limits(0.05, 'COST')[1] - 0.07) < 1e-12
    assert bad.returncode != 0 and 'MC_REBALANCING_BANDS' in bad.stderr, bad.stderr[-300:]
    # carried like every knob: banner, provenance, sweep
    assert diagnostics.DEFAULT_ASSUMPTIONS['REBALANCING_POLICY'] == ('model', 'MC_REBALANCING_BANDS')
    assert persistence.provenance()['rebalancing_bands'] == 'model'
    assert sensitivity.DIMENSIONS['rebalancing'].env == 'MC_REBALANCING_BANDS'
    # and when MOVED: a hard-coded 'model' passes every default-run check
    assert moved is not None, moved_out.stderr[-300:]
    assert moved == ['uniform', ['REBALANCING_POLICY']], moved
    assert legacy == ['uniform', {'relative': 0.25, 'absolute': 0.05}], legacy


def test_tail_df_of_zero_means_gaussian_and_is_not_announced():
    """A fat-tail sweep needs a 'none' point. crisis_shock already
    treats df <= 2 as Gaussian, but TAIL_DF = 0.0 would be reported as a moved
    knob on a run that is the shipped model. Zero is read as None."""
    import sensitivity
    (out, got), (_, five) = _engine_attrs(("engine.TAIL_DF", {'MC_TAIL_DF': '0'}),
                                          ("engine.TAIL_DF", {'MC_TAIL_DF': '5'}))
    assert got is None and out.returncode == 0, out.stderr[-300:]
    assert five == 5.0, five
    assert sensitivity.DIMENSIONS['tail_df'].env == 'MC_TAIL_DF'


class _FakeBlock:
    def __init__(self, text):
        self.type, self.text = 'text', text


class _FakeMessages:
    def __init__(self, reply, stop='end_turn'):
        self.reply, self.stop, self.calls = reply, stop, []

    def create(self, **kw):
        self.calls.append(kw)
        msg = type('M', (), {})()
        msg.content, msg.stop_reason, msg.stop_details = (
            [_FakeBlock(self.reply)], self.stop, None)
        return msg


class _FakeClient:
    def __init__(self, reply, stop='end_turn'):
        self.messages = _FakeMessages(reply, stop)


def _proposal(**over):
    from montecarlo import ask
    base = {k: None for k in ask.SPEC_SCHEMA['properties']}
    base.update({'mode': 'retirement', 'portfolios': ['165', '170']})
    base.update(over)
    return base


def test_ask_schema_is_strict_and_built_from_the_code_it_must_agree_with():
    """N8: Haiku only PROPOSES a spec. The schema's enums come from the same
    registries the runner validates against, so a new policy or plan choice is
    proposable the day it exists and an invented one never is."""
    from montecarlo import ask
    s = ask.SPEC_SCHEMA
    assert s['additionalProperties'] is False
    assert set(s['required']) == set(s['properties']), 'strict schemas need every key required'
    enum = lambda k: [x for x in s['properties'][k]['enum'] if x is not None]
    assert set(enum('policy')) == set(policies.POLICIES)
    assert set(enum('gk_response')) == set(policies.GK_RESPONSE)
    dims = s['properties']['compare_choice']['anyOf'][0]['properties']['dimension']['enum']
    assert set(dims) == set(runner.PLAN_CHOICES)


def test_ask_turns_a_proposal_into_a_spec_only_through_the_runners_validation():
    """The model can propose; only parse_spec and glide can accept. A typo'd
    portfolio, a malformed schedule, or an unknown sleeve is refused before
    anything runs."""
    from montecarlo import ask
    P = _P()
    spec = ask.to_spec(_proposal(return_views=True, gk_response='fast'), P)
    assert spec['portfolios'] == ['165', '170'] and spec['return_views'] is True
    assert 'custom' not in spec and 'compare_choice' not in spec, 'nulls are dropped'
    assert spec == runner.parse_spec(json.dumps(spec), P), 'same validation as --spec'
    ok = ask.to_spec(_proposal(portfolios=['166'],
                               custom=['166', '166>166+5:stip_only@60>166@65']), P)
    assert ok['custom'][1] == '166>166+5:stip_only@60>166@65'
    for bad in (_proposal(portfolios=['999']),
                _proposal(custom=['165>165+5:gold_only@60']),
                _proposal(custom=['165>167@60>165@55']),
                _proposal(compare_choice={'dimension': 'retirement_age', 'values': []}),
                _proposal(unsupported='sweep the cash rate')):
        try:
            ask.to_spec(bad, P)
        except ValueError:
            continue
        raise AssertionError(f'accepted a bad proposal: {bad}')


def test_ask_calls_haiku_with_structured_output_and_refuses_on_a_refusal():
    from montecarlo import ask
    P = _P()
    fake = _FakeClient(json.dumps(_proposal()))
    got = ask.propose('165 vs 170', P, client=fake)
    assert got['portfolios'] == ['165', '170']
    kw = fake.messages.calls[0]
    assert kw['model'] == ask.ASK_MODEL
    assert kw['output_config']['format'] == {'type': 'json_schema', 'schema': ask.SPEC_SCHEMA}
    # the catalogue it chooses from is the real one, not a remembered one
    sys_text = kw['system'] if isinstance(kw['system'], str) else kw['system'][0]['text']
    assert '170' in sys_text and 'stip_only' in sys_text
    assert kw['messages'][-1]['content'].endswith('165 vs 170')
    for stop in ('refusal', 'max_tokens'):
        try:
            ask.propose('x', P, client=_FakeClient('{}', stop=stop))
        except ValueError:
            continue
        raise AssertionError(f'{stop} must not be read as a spec')


def test_run_ask_shows_the_spec_and_runs_nothing_without_a_yes():
    """--ask is propose -> show -> confirm. A 'no' (or any answer but yes)
    must leave nothing run; the prompt must be the only way through."""
    from montecarlo import ask
    real = (ask.propose, runner.ask)
    real_run = report_retirement.run

    def _never(*a, **k):
        raise AssertionError('a run started without confirmation')

    script = iter(['n'])

    def _scripted(_p, _d=''):
        try:
            return next(script)
        except StopIteration:
            raise AssertionError('the prompt asked for more input than scripted')

    ask.propose = lambda sentence, portfolios, client=None: _proposal()
    runner.ask, report_retirement.run = _scripted, _never
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = runner.main(['--ask', '165 against 170'])
    finally:
        ask.propose, runner.ask = real
        report_retirement.run = real_run
    out = buf.getvalue()
    assert rc == 1, rc
    assert '"portfolios": ["165", "170"]' in out, out[-500:]
    assert '--spec' in out, 'the replayable form must be shown'


def _ssb():
    sys.path.insert(0, str(ROOT / 'tools'))
    import single_stock_benchmark
    return single_stock_benchmark


def _ssb_frame(n=240, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2006-08-01', periods=n, freq='MS')
    f = pd.DataFrame({k: rng.normal(0.004, 0.03, n) for k in factors.FACTORS}, index=idx)
    f['rf'] = 0.002
    return f


def test_single_stock_benchmark_measures_excess_over_the_models_own_recipe():
    """The single-stock rule: COST is held while its rolling 10-year excess
    over its FACTOR BUILD-UP stays above the exit trigger. The benchmark must be
    the recipe the model forecasts from (factors.fit_single_stock), or 'COST is
    earning its place' and 'the forecast is being met' stop being one test."""
    ssb = _ssb()
    assert ssb.fit is factors.fit_single_stock
    f = _ssb_frame()
    loads = {'mkt_rf': 0.8, 'rmw': 0.6}
    alpha_m = 0.03 / 12
    y = f['rf'] + alpha_m + sum(b * f[k] for k, b in loads.items())
    fit = {'beta': {'us': {k: loads.get(k, 0.0) for k in factors.FACTORS}}}
    ex = ssb.excess(y, f, fit)
    assert abs(ex.mean() * 12 - 0.03) < 1e-9, ex.mean() * 12


def test_single_stock_rule_triggers_only_on_a_full_window_below_the_threshold():
    ssb = _ssb()
    assert ssb.EXIT_THRESHOLD == -0.05 and ssb.WINDOW_YEARS == 10
    idx = pd.date_range('2010-01-01', periods=120, freq='MS')
    v = ssb.verdict(pd.Series(-0.06 / 12, index=idx))
    assert v['status'] == 'EXIT' and abs(v['mean'] + 0.06) < 1e-9, v
    v = ssb.verdict(pd.Series(-0.04 / 12, index=idx))
    assert v['status'] == 'HOLD', v
    # a short history decides nothing — one year's noise is +-15 points
    v = ssb.verdict(pd.Series(-0.30 / 12, index=idx[:60]))
    assert v['status'] == 'INSUFFICIENT', v
    # the default threshold is a policy choice, and arguable
    v = ssb.verdict(pd.Series(-0.06 / 12, index=idx), threshold=-0.07)
    assert v['status'] == 'HOLD', v
    # the TRAILING ten years decide, not the whole history: COST has ~35
    # years, and a bad first decade must not exit a stock that is earning now
    long_idx = pd.date_range('2000-01-01', periods=240, freq='MS')
    s = pd.Series(np.r_[np.full(120, -0.10 / 12), np.full(120, -0.04 / 12)],
                  index=long_idx)
    v = ssb.verdict(s)
    assert v['status'] == 'HOLD' and abs(v['mean'] + 0.04) < 1e-9, v
    # and its band is the standard error of a 120-month mean, annualised
    e = pd.Series(np.random.default_rng(3).normal(0, 0.02, 240), index=long_idx)
    v = ssb.verdict(e)
    want = 12 * e.iloc[-120:].std() / np.sqrt(120)
    assert abs(v['se'] - want) < 1e-12, (v['se'], want)


def test_parse_spec_validates_custom_schedules_before_anything_runs():
    """parse_spec's own docstring: VALIDATED HERE, not at the first use. It
    checked portfolio ids but passed every `custom` schedule through, so
    --dry-run accepted '168>168+5:gold_only@60' and the run failed minutes in."""
    P = _P()
    ok = runner.parse_spec(json.dumps({'portfolios': ['168'],
                                       'custom': ['168', '168>168+5:stip_only@60>168@65']}), P)
    assert ok['custom'][1] == '168>168+5:stip_only@60>168@65'
    for bad in ('168>168+5:gold_only@60', '168>999@60', '168>167@60>168@55', ''):
        try:
            runner.parse_spec(json.dumps({'portfolios': ['168'], 'custom': ['168', bad]}), P)
        except ValueError as exc:
            assert 'custom' in str(exc), exc
            continue
        raise AssertionError(f'accepted a bad custom schedule: {bad!r}')


def test_the_design_runs_skill_only_hands_over_commands_that_validate():
    """The design-runs skill is how a session (often a smaller model) builds
    run.py commands. Every --spec it shows must pass the runner's own
    validation, every sweep must name a real dimension, and every env knob it
    lists must be one the model reads — or the skill teaches a command that
    fails, which is the drift this exists to catch."""
    import sensitivity
    P = _P()
    text = (ROOT / '.claude' / 'skills' / 'design-runs' / 'SKILL.md').read_text()
    # a shell loop splices its variable into the JSON as '"$p"'
    specs = re.findall(r"run\.py --spec '(.+?)'(?:;|\s*$)",
                       text.replace("'\"$p\"'", 'constant_real'), re.M)
    assert len(specs) >= 6, f'expected the recipes, found {len(specs)}'
    for raw in specs:
        spec = runner.parse_spec(raw, P)
        strategies, _ = runner.resolve_strategies(spec, P)
        assert strategies, raw
    sweeps = re.findall(r'tools/sensitivity\.py (\w+) ([\d.,]+)(?:.*--strategies ([\w,>+@:]+))?', text)
    assert sweeps
    for dim, values, strategies in sweeps:
        assert dim in sensitivity.DIMENSIONS, dim
        for s_ in (strategies.split(',') if strategies else []):
            glide.validate(glide.parse_schedule(s_, runner.default_config()['age']), P)
    known = {env for _, env in diagnostics.DEFAULT_ASSUMPTIONS.values()}
    listed = set(re.findall(r'`(MC_[A-Z_]+)`', text))
    assert listed and listed <= known, sorted(listed - known)


def test_every_academic_series_is_dated_at_month_end():
    """The correlation step snaps each series onto month-end dates by NEAREST
    within 5 days. A Ken French month dated the 1st (1990-02-01 = February's
    return) lands on 1990-01-31, so the fund was correlated with the PREVIOUS
    month's market. Live until 2026-10-25 for AVLV, VFMO, MTUM, VTV, VBR and
    VOO's 1970s: AVLV-VOO read -0.03 in the calm regime against +0.88 aligned.
    Every parsed table and every academic proxy must carry month-end dates."""
    import data_collection_regime_model as dc
    kf = dc.KenFrenchParser(dc.REFERENCE_DIR)
    tables = {name: getattr(kf, name)() for name in (
        'parse_32_portfolios', 'parse_25_size_value', 'parse_25_size_momentum',
        'parse_ff5_factors', 'parse_ff3_factors', 'parse_developed_ex_us_factors',
        'parse_emerging_factors', 'parse_momentum_factor')}
    pb = dc.ProxyBuilder(kf, dc.AQRParser(dc.REFERENCE_DIR))
    with contextlib.redirect_stdout(io.StringIO()):
        tables.update({f'proxy {t}': pb.build_proxy(t) for t in dc.KF_TICKERS})
    bad = {name: str(df.index[~df.index.is_month_end][0].date())
           for name, df in tables.items() if not df.index.is_month_end.all()}
    assert not bad, f'not month-end dated: {bad}'


def test_another_correlation_window_loads_its_own_file_set():
    """MC_CORR_WINDOW picks the `_w<years>` matrices. A missing set must FAIL —
    silently falling back to the 15-year files would report a long-window run
    that never happened — and the env var must reach an importing engine, since
    workers re-import under `spawn`."""
    n = len(loaders.MODEL_TICKERS)
    assert loaders.correlation_filename('high', n) == f'correlation_high_{n}.npy'
    assert loaders.correlation_filename('high', n, 35) == f'correlation_high_{n}_w35.npy'
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = ROOT / 'output'
        (tmp / 'ticker_parameters_regimes_v8.json').write_text(
            (src / 'ticker_parameters_regimes_v8.json').read_text())
        for r in ('baseline', 'low', 'medium', 'high'):
            np.save(tmp / loaders.correlation_filename(r, n),
                    np.load(src / loaders.correlation_filename(r, n)))
            np.save(tmp / loaders.correlation_filename(r, n, 35), np.eye(n))
        with warnings_suppressed():
            dflt = loaders.load_parameters_from_json(base_dir=tmp, verbose=False)
            w35 = loaders.load_parameters_from_json(base_dir=tmp, verbose=False,
                                                    corr_window_years=35)
        assert not np.allclose(dflt['CORRELATION_HIGH'], np.eye(n))
        assert np.array_equal(w35['CORRELATION_HIGH'], np.eye(n))
        try:
            with warnings_suppressed():
                loaders.load_parameters_from_json(base_dir=tmp, verbose=False,
                                                  corr_window_years=40)
        except FileNotFoundError as e:
            assert '--corr-window-years 40' in str(e), e
        else:
            raise AssertionError('a missing window loaded something')
    code = "import sys; sys.path.insert(0, %r); from montecarlo import engine" % str(ROOT)
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                         cwd=ROOT, env=dict(os.environ, MC_CORR_WINDOW='99'))
    assert out.returncode != 0 and 'MC_CORR_WINDOW=99' in out.stderr, out.stderr[-500:]


@contextlib.contextmanager
def warnings_suppressed():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


def test_every_model_knob_is_announced_when_moved():
    """The moved-assumption banner (diagnostics.DEFAULT_ASSUMPTIONS) exists so
    a run with a knob moved does not look like a default run. Found 2026-10-19:
    MC_PARAM_STD_SCALE, MC_RF_SOURCE, MC_INFL_BETA_BOND and
    MC_PREMIA_LONG_WINDOW were read by the model and announced by nothing.
    MC_RF_FORWARD is excluded on purpose: its default is derived from two other
    knobs, so it has no constant to compare against."""
    src = ''.join((ROOT / f).read_text() for f in ('montecarlo/engine.py', 'montecarlo/factors.py'))
    read = set(re.findall(r"environ(?:\.get\(|\[)\s*'(MC_[A-Z_]+)'", src)) - {'MC_RF_FORWARD'}
    known = {env for _, env in diagnostics.DEFAULT_ASSUMPTIONS.values()}
    assert read - known == set(), sorted(read - known)
    # and each one resolves to a real attribute, so the banner can read it
    for attr in diagnostics.DEFAULT_ASSUMPTIONS:
        assert hasattr(pa, attr) or hasattr(factors, attr), attr
    assert diagnostics.moved_assumptions() == [], diagnostics.moved_assumptions()


def test_bond_spread_is_measured_over_contemporaneous_cash():
    """§6c: the raw mean inherits the fund's rate era; the spread does not."""
    rf = np.concatenate([np.full(120, 0.05 / 12), np.full(120, 0.01 / 12)])
    r = rf + 0.015 / 12
    assert abs(factors.bond_spread(r, rf) - 0.015) < 1e-9
    # and it is genuinely different from the raw mean, which straddles two eras
    assert abs(r.mean() * 12 - 0.015) > 0.02


def test_a_spurious_region_is_rejected_under_collinearity():
    """factors.REGION_T alone was the wrong rule, and this pins why.

    The regional market factors correlate at 0.777, so on a long sample a
    trivial spurious loading still clears t=2. VTI — Vanguard TOTAL US MARKET,
    holding no international — took a developed market beta of 0.022 over 301
    months, and VTV took 0.135, moving its expected excess return -0.68pp. A
    t-stat threshold is sample-size dependent in the wrong direction: longer
    histories admit SMALLER spurious loadings, so magnitude and explained
    variance have to gate it too.
    """
    rng = np.random.default_rng(11)
    n = 400
    # two correlated regional market factors, plus their style columns
    a = rng.normal(0.006, 0.042, n)
    b = 0.777 * a + np.sqrt(1 - 0.777 ** 2) * rng.normal(0.006, 0.042, n)
    frames = {}
    for tag, mkt in (('us', a), ('dev', b)):
        frames[tag] = pd.DataFrame(
            {'mkt_rf': mkt,
             **{f: rng.normal(0.002, 0.02, n) for f in factors.FACTORS[1:]}})
    # a purely "us" fund: no dev exposure whatsoever
    y = 0.98 * a + rng.normal(0, 0.004, n)
    fit = factors.fit_loadings(y, frames, regions=['us', 'dev'])
    assert fit is not None
    assert fit['regions'] == ['us'], fit['regions']
    assert not fit['multi_region']
    assert abs(fit['beta']['us']['mkt_rf'] - 0.98) < 0.05

    # and the three criteria are all actually consulted
    assert factors.REGION_MIN_DELTA_ADJ_R2 > 0
    assert factors.REGION_MIN_MKT_BETA > 0


def test_scenarios_move_factor_premia_and_never_the_market():
    """§5's rule: mixing the market premium into a factor scenario produces a
    beta conclusion dressed as a factor conclusion. Measured: raw last-15y
    premia read 141 vs 102 at -0.56pp/yr, of which market-beta is -0.57pp and
    factors are +0.02pp.
    """
    base = factors.premia_for('full history')
    # STYLE scenarios only. The regional ones added 2026-09-11 move mkt_rf on
    # purpose — that IS their content — and they carry no style multipliers, so
    # the two kinds cannot be confused. The rule this test exists for is intact:
    # no scenario mixes the two, so a factor conclusion can never be a beta
    # conclusion in disguise.
    regional = {n for n, m in factors.SCENARIOS.items() if factors._REGIONAL in m}
    for name_ in factors.SCENARIOS:
        if name_ in regional:
            continue
        p = factors.premia_for(name_)
        for r in factors.PREMIA:
            assert p[r]['mkt_rf'] == base[r]['mkt_rf'], (name_, r)
    for name_ in regional:
        mult = factors.SCENARIOS[name_]
        assert set(mult) == {factors._REGIONAL}, (name_, 'must not mix kinds')
        p = factors.premia_for(name_)
        for r in factors.PREMIA:
            for f in factors.FACTORS:
                if f != 'mkt_rf':
                    assert p[r][f] == base[r][f], (name_, r, f)
    dead = factors.premia_for('value dead')
    for r in factors.PREMIA:
        assert dead[r]['hml'] == 0.0
        assert dead[r]['smb'] == base[r]['smb'], 'only the named factor moves'
    half = factors.premia_for('all half')
    assert abs(half['us']['hml'] - base['us']['hml'] * 0.5) < 1e-12
    none = factors.premia_for('no factors')
    assert all(none['us'][f] == 0.0 for f in factors.FACTORS if f != 'mkt_rf')
    try:
        factors.premia_for('nonsense')
        raise AssertionError('unknown scenario must raise')
    except KeyError:
        pass


def test_intl_keeps_lagging_holds_the_recent_gap_not_the_recent_level():
    """'intl keeps lagging' is the belief that international trails the US by
    the margin it did over the window its funds' own means cover. That margin
    was the US running AHEAD — both regions earned more than their full-window
    premia over it — so the scenario holds the GAP below the US's long-run
    premium. Each region at its recent level would have RAISED both, testing
    the opposite belief."""
    base = factors.premia_for('full history')
    lag = factors.premia_for('intl keeps lagging')
    for r in ('dev', 'emg'):
        assert factors.RECENT_LAG[r] > 0, r
        assert abs(lag[r]['mkt_rf'] - (base['us']['mkt_rf'] - factors.RECENT_LAG[r])) < 1e-12, r
        assert lag[r]['mkt_rf'] < base[r]['mkt_rf'], f'{r} must lag MORE than by default'
    assert lag['us'] == base['us'], 'the US keeps its long-run premia'
    # A haircut is a claim about every premium, so the gap shrinks with it.
    half = factors.premia_for('intl keeps lagging', haircut=0.5)
    for r in ('dev', 'emg'):
        assert abs(half[r]['mkt_rf'] - 0.5 * lag[r]['mkt_rf']) < 1e-12, r
    # The plain {target: source} form still copies with no gap.
    caught = factors.premia_for('intl catches up')
    assert caught['dev']['mkt_rf'] == base['us']['mkt_rf']


def test_the_recent_lag_is_what_the_factor_data_says():
    """RECENT_LAG is a pasted figure like PREMIA, so pin it to the Ken French
    series over RECENT_LAG_WINDOW. And pin WHY it is a gap: each region's
    shortfall against its own full-window premium is under a quarter of the
    gap (developed has none), so 'international lagged' means the US ran
    ahead — a region set to its recent level would barely lag at all."""
    import premia as _premia
    import data_collection_regime_model as dc
    kf = dc.KenFrenchParser(str(ROOT / 'reference'))
    start, end = factors.RECENT_LAG_WINDOW
    measured = _premia.recent_lag(_premia.series_by_region(kf), start, end)
    assert set(measured) == set(factors.RECENT_LAG)
    for r, lag in factors.RECENT_LAG.items():
        m = measured[r]
        assert (m['start'], m['end']) == (start, end), (r, m)
        assert abs(m['lag'] - lag) <= 0.00005, f"RECENT_LAG[{r!r}] is stale: data says {m['lag']:.4f}"
        shortfall = factors.PREMIA[r]['mkt_rf'] - m['premium']
        assert shortfall < m['lag'] / 4, (r, shortfall, m['lag'])


def test_expected_return_subtracts_both_costs_and_respects_the_gate():
    fit = {'method': 'factor', 'regions': ['us'],
           'beta': {'us': {f: 0.0 for f in factors.FACTORS}},
           'alpha': 0.0, 'alpha_se': 0.0, 'adj_r2': 0.95,
           'n_months': 200, 'multi_region': False}
    fit['beta']['us']['mkt_rf'] = 1.0
    bare = factors.expected_return(fit, costs=(0.0, 0.0))
    assert abs(bare - (factors.RF_FORWARD + factors.PREMIA['us']['mkt_rf'])) < 1e-12
    # BOTH costs come out — dropping alpha added both back (§6e)
    netted = factors.expected_return(fit, costs=(0.0025, 0.0040))
    assert abs((bare - netted) - 0.0065) < 1e-12, bare - netted
    # below the gate there is no factor answer at all
    fit['method'] = 'below_gate'
    assert factors.expected_return(fit) is None
    assert factors.expected_return(None) is None


def test_the_factor_gate_reads_a_bimodal_split_not_a_tuned_threshold():
    """Documents the measured partition §5d depends on, so a future edit that
    moves the 0.80 gate has to argue with a number.

    Adjusted R² against the best regional factor set, all 26 tickers: the six
    bond funds and the single stock land at 0.217-0.577, every equity fund at
    0.888-0.998 (STIP, added 2026-10-02, measured 0.386; VGIT, added 2026-10-24,
    0.329). Nothing occupies 0.58-0.88, so any cutoff in that range gives
    the same answer and the threshold is not a tuned parameter.
    """
    BELOW = {'BSV': 0.217, 'COST': 0.253, 'BND': 0.258, 'STIP': 0.386,
             'JPST': 0.396, 'VUSB': 0.466, 'JPLD': 0.577, 'VGIT': 0.329}
    ABOVE = {'MTUM': 0.888, 'AVEE': 0.910, 'VT': 0.929, 'VOO': 0.996,
             'VTI': 0.998}
    gate = factors.GATE_ADJ_R2
    assert max(BELOW.values()) < gate < min(ABOVE.values()), gate
    # and the excluded set is exactly bonds + the single stock, by construction
    assert set(loaders.SINGLE_STOCKS) <= set(BELOW), loaders.SINGLE_STOCKS
    assert set(loaders._BOND_FUNDS) <= set(BELOW), loaders._BOND_FUNDS


def test_missing_data_sentinels_are_masked_before_scaling():
    """-99.99/100 is -0.9998999999999999, which never equals the literal -0.9999.

    The old `.replace([-0.9999, ...])` ran AFTER the divide, so it silently
    never fired. It left 4 months of -99.99% in 25_size_momentum (Big_Low,
    1927-1938 — outside every bucket and window in use, so nothing shipped
    moved) and 60 in the emerging factor file, where RMW read -46.50%/yr. The
    guard is now `<= -99` before scaling, which does not depend on float
    equality at all.
    """
    import data_collection_regime_model as dcm
    assert (-99.99 / 100.0) != -0.9999, 'the premise of the bug'
    ref = Path(__file__).resolve().parent.parent / 'reference'
    assert ref.is_dir(), ref                    # git-tracked: fail, never skip
    kf = dcm.KenFrenchParser(str(ref))
    names = ['parse_25_size_value', 'parse_25_size_momentum', 'parse_32_portfolios',
             'parse_ff5_factors', 'parse_momentum_factor',
             'parse_developed_ex_us_factors', 'parse_emerging_factors']
    for name in names:
        df = getattr(kf, name)()
        survivors = int((df <= -0.99).sum().sum())
        assert survivors == 0, f'{name} kept {survivors} sentinel value(s)'


def test_international_factors_span_the_international_sleeve():
    """US factors leave 20-45% of the international funds unexplained.

    VEA regressed against US factors gives R^2 0.78 and -4.41%/yr of "alpha"
    that is not alpha — it is international market exposure the US factors
    cannot see. Since the design drops residual alpha as un-estimable, that
    would move VEA by 4pp/yr on no evidence, and ~30% of the live portfolios is
    international. Against developed-ex-US factors the same fund reads R^2 0.98
    and -0.76%.
    """
    import data_collection_regime_model as dcm
    ref = Path(__file__).resolve().parent.parent / 'reference'
    assert (ref / 'Developed_ex_US_5_Factors.csv').exists(), ref   # git-tracked
    kf = dcm.KenFrenchParser(str(ref))
    dev = kf.parse_developed_ex_us_factors()
    emg = kf.parse_emerging_factors()
    for name, df, first in (('developed ex US', dev, '1990-07'),
                            ('emerging', emg, '1989-07')):
        assert list(df.columns) == ['mkt_rf', 'smb', 'hml', 'rmw', 'cma',
                                    'rf', 'mom'], (name, list(df.columns))
        assert str(df.index[0])[:7] == first, (name, str(df.index[0]))
        assert df.index.duplicated().sum() == 0, name
        # exactly one row per month between the endpoints: these files carry an
        # "Annual Factors" block too, and a leak would otherwise only be caught
        # indirectly by the magnitude assertions below
        span = ((df.index[-1].year - df.index[0].year) * 12
                + df.index[-1].month - df.index[0].month + 1)
        assert len(df) == span, (name, len(df), span)
        c = df.dropna()
        # sane magnitudes once the sentinels are gone
        assert 0.0 < c['mkt_rf'].mean() * 12 < 0.20, (name, c['mkt_rf'].mean())
        assert -0.10 < c['rmw'].mean() * 12 < 0.20, (name, c['rmw'].mean())


def test_ken_french_parsers_return_one_row_per_month():
    """The CSVs concatenate several tables and `nrows` cannot separate them.

    parse_25_size_value read 1,498 rows where only 1,190 are the value-weighted
    monthly table; the remaining 308 were EQUAL-weighted months (1926-07 ..
    1952-02) which the sort then interleaved, giving two different returns for
    the same month (ME1 BM1 1926-07 held both 0.058276 and 0.066093).

    No shipped parameter moved — every proxy window starts 1963-07 or later and
    the duplicates ended 1952-02 — but any window widened past 1952 would have
    silently averaged value- and equal-weighted returns. parse_32_portfolios
    always had this guard; the two 25-portfolio parsers did not.
    """
    import data_collection_regime_model as dcm
    ref = Path(__file__).resolve().parent.parent / 'reference'
    assert ref.is_dir(), ref                    # git-tracked: fail, never skip
    kf = dcm.KenFrenchParser(str(ref))
    for name in ('parse_25_size_value', 'parse_25_size_momentum',
                 'parse_32_portfolios'):
        df = getattr(kf, name)()
        dups = int(df.index.duplicated().sum())
        assert dups == 0, f'{name} returned {dups} duplicated months'
        assert df.index.is_monotonic_increasing, f'{name} is not sorted'
        assert len(df) > 600, f'{name} returned only {len(df)} rows'


def test_zzz_the_suite_left_no_user_config_modified():
    """No test may write to the user's config/. Runs last, by name.

    This is not hypothetical. A timing test called instrument.summary() without
    redirecting TIMING_FILE, so every validate_all recorded a fake 9,000-path
    run of near-zero duration into config/timing.json. The stored
    seconds-per-path was dragged toward zero, and the run-time estimate got
    progressively more wrong — two commits were spent chasing the symptom before
    the leak itself was found.

    A unit suite that mutates the state it is validating will always be caught
    late and diagnosed wrongly, so this checks the whole directory rather than
    one file.
    """

    cfg = Path(__file__).resolve().parent.parent / 'config'
    if not cfg.is_dir():
        return
    # CONTENT AND MTIME. Content alone let a test that wrote the real
    # config/runs.json and then restored it pass — identical bytes — while the
    # cloud-synced project folder turned the two rapid writes into an
    # edit-conflict copy. Any write is the defect, restored or not. New files
    # count too: a test that CREATES state in config/ is writing to it.
    now = _config_fingerprint()
    before = _CONFIG_FINGERPRINT
    changed = [n for n, v in now.items() if before.get(n) not in (None, v)]
    created = [n for n in now if n not in before]
    assert not changed and not created, (
        f"config/ was written during the suite — changed: {changed or '-'}, "
        f"created: {created or '-'}. A test is touching real user state; point "
        f"it at a tempfile. (Or a run.py you started meanwhile saved a preset — "
        f"re-run the suite when it finishes.)")


def _config_fingerprint():
    import hashlib
    cfg = Path(__file__).resolve().parent.parent / 'config'
    if not cfg.is_dir():
        return {}
    return {f.name: (hashlib.sha256(f.read_bytes()).hexdigest(),
                     f.stat().st_mtime_ns)
            for f in sorted(cfg.glob('*.json'))}


# --- correlation matrices are POSITIONAL --------------------------------------------
# The .npy files carry no labels: every consumer reads a pair as
# C[TICKERS.index(a), TICKERS.index(b)]. Measured 2026-09-12 on the real
# matrices, prepending one ticker to MODEL_TICKERS without regenerating moves
# 141's crisis-regime volatility -11.4% while the matrix stays a valid 25x25
# PSD unit-diagonal correlation matrix. The bias is DOWNWARD, which flatters
# exactly the multi-fund books this tool is used to compare.

def test_correlation_order_agreeing_is_silent():
    from tools.validate_parameters import correlation_order_problem as cop
    assert cop(['A', 'B', 'C'], ['A', 'B', 'C']) == ''
    assert cop(('A', 'B'), ['A', 'B']) == '', 'a tuple from JSON must compare equal'


def test_correlation_order_catches_a_reorder():
    from tools.validate_parameters import correlation_order_problem as cop
    msg = cop(['C', 'A', 'B'], ['A', 'B', 'C'])
    assert 'REORDER' in msg, msg
    assert 'position 0' in msg, msg
    # The set is identical, so any set-based comparison would pass here. That is
    # the whole point: this is the only failure mode a sorted()/set() check
    # cannot see.
    assert set(['C', 'A', 'B']) == set(['A', 'B', 'C'])


def test_correlation_order_catches_a_universe_change_and_an_absent_record():
    from tools.validate_parameters import correlation_order_problem as cop
    assert 'universe changed' in cop(['A', 'B'], ['A', 'B', 'C'])
    assert 'absent' in cop(None, ['A', 'B'])


def test_the_shipped_parameters_record_the_matrix_order():
    """The on-disk record must be present and current — not just the function.

    Without this the pure checks above pass forever on a file that never gained
    the key. `validate_parameters` gates the same thing; this fails in the unit
    suite too, where it is cheap to run.
    """
    from tools.validate_parameters import correlation_order_problem as cop
    from montecarlo.loaders import MODEL_TICKERS
    j = Path(__file__).resolve().parent.parent / 'output' / 'ticker_parameters_regimes_v8.json'
    meta = json.loads(j.read_text())['metadata']
    assert cop(meta.get('correlation_tickers'), MODEL_TICKERS) == '', \
        cop(meta.get('correlation_tickers'), MODEL_TICKERS)


def test_near_psd_repair_returns_something_cholesky_can_factor():
    """The repair must yield a POSITIVE DEFINITE matrix, not merely PSD.

    Higham's final step projects onto unit diagonal, which is not a projection
    onto PSD, so the loop lands PSD only to tolerance. np.linalg.cholesky needs
    strict definiteness. Before the eigenvalue floor, this input left the repair
    at min eig -4.5e-06 and cholesky_factor RAISED -- having already spent its
    one retry calling the same repair again.
    """
    m = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    assert np.linalg.eigvalsh(m).min() < 0, 'the fixture must actually be indefinite'
    fixed = pa.near_psd_higham(m)
    assert np.linalg.eigvalsh(fixed).min() > 0, 'repair left a non-PD matrix'
    assert np.allclose(np.diag(fixed), 1.0), 'repair broke the unit diagonal'
    np.linalg.cholesky(fixed)          # must not raise
    pa.cholesky_factor(m)              # nor must the caller that retries it


# --- a plan level can be GUARANTEED INCOME's, not the portfolio's -------------------
# Found 2026-09-12 in a live run: under `vpw reserve 10M` all twelve strategies
# solved to 0.80-1.00x the annual Social Security benefit, and two DIFFERENT
# portfolios (140, 180) returned the same figure bit-identically -- both are
# exactly the benefit / 0.85. A large reserve leaves vpw asking for less than the
# benefit in the bottom decile, `drawn = max(0, requested - ss)` is zero for a
# whole band, and the band mean is the benefit exactly.

_SS_CFG = {'annual_at_fra': 60_000.0, 'claim_age': 67.0, 'benefit_scale': 1.0}


def _ss_config():
    return {'age': 37, 'time_horizon_years': 58,
            'withdrawal': {'start_age': 65, 'floor': 0.85},
            'social_security': dict(_SS_CFG)}


def test_guaranteed_income_row_starts_at_the_claim_age():
    cfg = _ss_config()
    row = solver.guaranteed_income_row(cfg, 30)
    assert row[0] == 0.0 and row[1] == 0.0, 'ages 65-66 precede a claim at 67'
    assert row[2] == 60_000.0, row[:4]
    assert solver.guaranteed_income_row({**cfg, 'social_security': None}, 30).sum() == 0


def test_portfolio_share_is_zero_when_only_social_security_is_spent():
    cfg = _ss_config()
    ss = solver.guaranteed_income_row(cfg, 30)
    only_ss = np.tile(ss, (8, 1))
    funded = only_ss + 100_000.0
    share = solver.portfolio_share_of_binding_band(
        np.vstack([only_ss[:4], funded[:4]]), cfg, 'per_band')
    assert np.allclose(share[:4], 0.0), share[:4]
    assert (share[4:] > 0.5).all(), share[4:]


def test_a_level_set_by_social_security_is_flagged_not_ranked():
    """score() must mark it, so the report can refuse to rank it.

    The failure this pins is silent: the number is arithmetically correct and
    looks like every other level in the column.
    """
    cfg = _ss_config()
    ss = solver.guaranteed_income_row(cfg, 30)
    raw = {'spending': np.tile(ss, (200, 1)),
           'depleted': np.zeros(200, dtype=bool),
           'finals': np.full(200, 1e7), 'depletion_ages': [None] * 200}
    # the level the solver would certify on these paths: worst band 46,080
    r = solver.score(raw, cfg, 46_080.0 / 0.85, floor=0.85, adequacy='per_band')
    assert r['portfolio_share'] <= report_retirement.GUARANTEED_SHARE_TOL, \
        r['portfolio_share']
    # and the same paths WITH a real portfolio contribution must not be flagged
    raw2 = dict(raw, spending=np.tile(ss + 80_000.0, (200, 1)))
    r2 = solver.score(raw2, cfg, 100_000.0, floor=0.85, adequacy='per_band')
    assert r2['portfolio_share'] > report_retirement.GUARANTEED_SHARE_TOL, \
        r2['portfolio_share']


def test_robustness_separates_a_stricter_floor_from_a_missed_target():
    """A level certified at an 85% floor MUST score lower at 100%.

    Reporting that as fragility said "36 of 36 results fall below 90% at some
    floor" on the 2026-09-12 run, where 32 failed only at the stricter floor and
    the 4 that actually missed their own target were buried inside the headline.
    """
    definitional = {'spend': 1e5, 'floor': 0.85, 'worst_cut': 0.0,
                    'success_by_floor': {0.7: 0.95, 0.85: 0.92, 1.0: 0.70}}
    missed = {'spend': 1e5, 'floor': 0.85, 'worst_cut': 0.0,
              'success_by_floor': {0.7: 0.95, 0.85: 0.80, 1.0: 0.60}}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_retirement.robustness({'gk': {'a': definitional, 'b': missed}},
                                     ['a', 'b'], confidence=0.90)
    out = buf.getvalue()
    assert '1 of 2 hold at their own 85% floor' in out, out
    assert '1 of 2 came in below the 90%' in out, out
    assert 'more than the solver' in out, out          # 80% is 10pp out, not noise


def test_robustness_reads_floors_that_survived_a_json_round_trip():
    """JSON keys are strings; the grid is floats. The lookup defaulted to 1.0."""
    res = {'spend': 1e5, 'floor': 0.85, 'worst_cut': 0.0,
           'success_by_floor': {0.7: 0.95, 0.85: 0.80, 1.0: 0.60}}
    round_tripped = json.loads(json.dumps(res))
    assert set(round_tripped['success_by_floor']) == {'0.7', '0.85', '1.0'}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_retirement.robustness({'gk': {'a': round_tripped}}, ['a'],
                                     confidence=0.90)
    out = buf.getvalue()
    # Which branch prints depends on whether ALL results fail alike; assert the
    # thing common to both — that the shortfall was SEEN. Before the float/str
    # fix this printed the all-clear.
    assert '⚠' in out and '85%' in out, out
    assert 'your spending flexibility does not change the answer' not in out, out


def test_timing_is_calibrated_per_workload_not_pooled():
    """A decumulation path costs ~3x an accumulation one; one constant for both
    is wrong for both.

    Measured 2026-09-12 on a 12-strategy retirement run: estimate_minutes
    predicted the PATH COUNT to 1.7% (151,967 against 149,400) and the estimate
    was still 2.8x low — 11.2 min against an actual 31.3 — entirely because the
    stored 4.42ms/path had been set by cheaper work.
    """
    saved_file, saved_mode = instrument.TIMING_FILE, instrument._mode
    try:
        with tempfile.TemporaryDirectory() as d:
            instrument.TIMING_FILE = Path(d) / 'timing.json'
            instrument.TIMING_FILE.write_text(json.dumps({
                'seconds_per_path': 0.004,
                'modes': {'retirement': {'seconds_per_path': 0.012},
                          'accumulation': {'seconds_per_path': 0.003}}}))
            assert instrument.load_seconds_per_path('retirement') == 0.012
            assert instrument.load_seconds_per_path('accumulation') == 0.003
            # an unmeasured mode falls back to the legacy pooled value
            assert instrument.load_seconds_per_path('nonesuch') == 0.004
            # and so does a file that predates the split
            instrument.TIMING_FILE.write_text(json.dumps({'seconds_per_path': 0.009}))
            assert instrument.load_seconds_per_path('retirement') == 0.009
    finally:
        instrument.TIMING_FILE, instrument._mode = saved_file, saved_mode


def test_the_estimate_prices_a_retirement_path_above_an_accumulation_one():
    """The split only helps if the ESTIMATE actually reads it."""
    saved_file, saved_mode = instrument.TIMING_FILE, instrument._mode
    try:
        with tempfile.TemporaryDirectory() as d:
            instrument.TIMING_FILE = Path(d) / 'timing.json'
            instrument.TIMING_FILE.write_text(json.dumps({
                'seconds_per_path': 0.004,
                'modes': {'retirement': {'seconds_per_path': 0.012},
                          'accumulation': {'seconds_per_path': 0.003}}}))
            instrument.set_mode(None)
            ret = runner._seconds_per_path('retirement')
            acc = runner._seconds_per_path('accumulation')
            assert ret > 3 * acc, (ret, acc)
            spec = {'mode': 'retirement', 'spend': 'solve',
                    'policy': 'guyton_klinger', 'portfolios': ['141']}
            mins = runner.estimate_minutes(1, spec)
            assert mins > 0
            # the same run priced at the pooled constant is materially cheaper
            instrument.TIMING_FILE.write_text(json.dumps({'seconds_per_path': 0.004}))
            assert runner.estimate_minutes(1, spec) < mins * 0.5
    finally:
        instrument.TIMING_FILE, instrument._mode = saved_file, saved_mode


def _shape_row(pol, tag, per_year, n_paths=200, bequest=1e7):
    """One WHAT YOU ACTUALLY SPEND row with a chosen spending trajectory."""
    return (pol, tag, {'spending': np.tile(np.asarray(per_year, float), (n_paths, 1)),
                       'median_bequest': bequest, 'start_age': 65})


def test_life_stage_table_ranks_on_the_average_ACROSS_bands():
    """A row can lose every early decade and still spend the most overall.

    The multi-policy table was printed in policy-major spec order — i.e. not
    ranked at all — so a rising policy sat below a flat one that it beat on
    lifetime spending. Ranking within a single band would keep it there, which
    is why the key is the average across them.
    """
    cfg = {'age': 37, 'time_horizon_years': 58, 'retirement_age': 65}
    flat = [300_000.0] * 30                       # 300/300/300, avg 300k
    rising = [100_000.0] * 10 + [300_000.0] * 10 + [800_000.0] * 10   # avg 400k
    rows = [_shape_row('gk', 'FLAT', flat), _shape_row('vpw', 'RISING', rising)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_retirement._spending_shape(rows, cfg, policy_width=6)
    out = buf.getvalue()
    body = [l for l in out.splitlines() if 'FLAT' in l or 'RISING' in l]
    assert len(body) == 2, out
    assert 'RISING' in body[0], f'the higher lifetime spender must lead:\n{out}'
    # ...while LOSING the first band, which a within-band ranking would rank on.
    # [1], not [0]: the lifetime average is now the anchor column before the
    # bands, so [0] is that cell and the assertion below would read it instead.
    first_band = lambda line: line.split('\u2502')[1]
    assert '100k' in first_band(body[0]), first_band(body[0])
    assert '300k' in first_band(body[1]), first_band(body[1])
    assert 'across bands' in out.lower(), out


def test_life_stage_ranking_is_insensitive_to_the_order_rows_arrive_in():
    cfg = {'age': 37, 'time_horizon_years': 58, 'retirement_age': 65}
    a = _shape_row('p1', 'LOW', [100_000.0] * 30)
    b = _shape_row('p2', 'MID', [200_000.0] * 30)
    c = _shape_row('p3', 'HIGH', [300_000.0] * 30)
    for order in ([a, b, c], [c, a, b], [b, c, a]):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report_retirement._spending_shape(list(order), cfg, policy_width=5)
        tags = [t for l in buf.getvalue().splitlines()
                for t in ('HIGH', 'MID', 'LOW') if l.strip().startswith(('p1', 'p2', 'p3'))
                and t in l]
        assert tags == ['HIGH', 'MID', 'LOW'], (order, tags)


def test_life_stage_table_bands_are_decades_counted_back_from_the_plan_age():
    """N4. The table's bands are calendar ages, not thirds of each retirement:
    thirds put a 60 retiree's decades at different ages from a 65 retiree's, so
    a retirement-age choice could not be shown at all. Decades end on the plan
    age, so 'the last decade' is a real column; below the first one the cuts
    are the options' own retirement ages, so no cell is half working. At plan
    to 95 this is exactly 65-74/75-84/85-94, the bands every earlier figure
    was quoted in."""
    bands = report_retirement.age_bands
    for starts, end, want in [
            ([65], 95, ['65-74', '75-84', '85-94']),
            ([65], 100, ['65-69', '70-79', '80-89', '90-99']),
            ([60, 65, 70], 100, ['60-64', '65-69', '70-79', '80-89', '90-99']),
            ([62, 65], 100, ['62-64', '65-69', '70-79', '80-89', '90-99']),
            ([69, 70], 100, ['69', '70-79', '80-89', '90-99'])]:
        b = bands(starts, end)
        assert [lab for _, _, lab in b] == want, (starts, end, b)
        assert b[0][0] == min(starts) and b[-1][1] == end, 'must cover the span'
        assert all(b[i][1] == b[i + 1][0] for i in range(len(b) - 1)), 'no gaps'


def test_life_stage_table_lines_up_options_that_retire_at_different_ages():
    """It used to print 'not shown' for a retirement-age choice — the one
    question where seeing where the extra years go matters most. Each row's
    band reads the same calendar ages, and a band before a row retires says
    so instead of averaging an empty slice."""
    cfg = {'age': 40, 'time_horizon_years': 60, 'retirement_age': 65}  # to 100
    early = _shape_row('', 'EARLY', [111e3] * 5 + [222e3] * 35)
    early[2]['start_age'] = 60
    late = _shape_row('', 'LATE', [333e3] * 5 + [444e3] * 30)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_retirement._spending_shape([early, late], cfg)
    out = buf.getvalue()
    assert 'not shown' not in out, out
    head = next(l for l in out.splitlines() if '60-64' in l)
    assert [c.strip() for c in head.split('│')[1:6]] == \
        ['60-64', '65-69', '70-79', '80-89', '90-99'], head
    cells = {t: [c.strip() for c in l.split('│')[1:6]]
             for l in out.splitlines() for t in ('EARLY', 'LATE')
             if l.strip().startswith(t)}
    assert cells['EARLY'][0].startswith('111k'), cells
    assert cells['EARLY'][1].startswith('222k'), cells
    assert cells['LATE'][0] == 'working', cells
    assert cells['LATE'][1].startswith('333k'), cells
    assert cells['LATE'][2].startswith('444k'), cells


def test_robustness_explains_what_a_floor_is_before_printing_floor_numbers():
    """The section opened straight into "[70%:95% 85%:90% 100%:75%]".

    It never said what a floor was, what the three numbers meant, or which of
    them the run had solved for — so the only readers who could use it were the
    ones who already knew.
    """
    res = {'spend': 1e5, 'floor': 0.85, 'worst_cut': 0.0,
           'success_by_floor': {0.7: 0.95, 0.85: 0.92, 1.0: 0.75}}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_retirement.robustness({'gk': {'a': res}}, ['a'], confidence=0.90)
    out = buf.getvalue()
    head, _, _ = out.partition('[70%')
    for needed in ('90%', '85%', 'cut', 'what was solved for'):
        assert needed in head, f'{needed!r} must appear BEFORE the first bracket:\n{head}'
    # every floor on the grid is spelled out as a plain-English cut
    for f in solver.FLOOR_GRID:
        assert f"{f:.0%}" in head, (f, head)
    assert 'HOW TO USE THIS' in out, out


def test_crossing_reports_no_slope_when_its_local_fit_is_not_decreasing():
    """_crossing's second return value is 0.0 whenever the local fit is flat.

    That zero is what the verification guard used to test, so a noisy
    neighbourhood silently disabled the whole correction.
    """
    xs = [100.0, 200.0, 300.0, 400.0]
    flat = [0.90, 0.90, 0.90, 0.90]
    _, sl, _clip = solver._crossing(np.array(xs), np.array(flat), 0.90)
    assert sl == 0.0, sl
    falling = [0.98, 0.94, 0.90, 0.86]
    x0, sl2, _c2 = solver._crossing(np.array(xs), np.array(falling), 0.90)
    assert sl2 < 0, sl2


def test_a_flat_local_fit_still_gets_the_verification_correction():
    """8 of 15 strategies took NO verification step on the run of 2026-09-13.

    `142>142+5@65` reported $200,000 — a round bracket probe — and verified at
    87.7% against a 90% target, ranked second in both tables. The guard read
    `slope < -1e-12`, and a flat local fit returns exactly 0.0, so the Newton
    pass that moves the level onto the verification curve never ran. The bracket
    clip is only harmless WHEN NEWTON RUNS, which is the part nothing checked.
    """
    xs = np.array([100.0, 200.0, 300.0, 400.0])
    # RISING right at the crossing, falling across the bracket — noise does this
    # regularly, and it is the case that produced slope 0.0 and no correction.
    ys = np.array([0.88, 0.90, 0.92, 0.80])
    assert solver._crossing(xs, ys, 0.90)[1] == 0.0, 'fixture must produce a 0 slope'
    got = solver.usable_slope(0.0, xs, ys)
    assert got < -1e-12, got
    assert abs(got - (0.80 - 0.88) / 300.0) < 1e-12, got
    # a usable local slope is never overridden
    assert solver.usable_slope(-0.0004, xs, ys) == -0.0004
    # and a genuinely flat curve has no slope to invent
    assert solver.usable_slope(0.0, xs, np.full(4, 0.90)) == 0.0
    # The solver must ASK for it — a guard that reads the raw slope is the bug.
    # This search curve rises through the target, so the local fit is flat;
    # the verification curve crosses at $205k. Reading the raw slope it stays
    # at $248k, 4pp short, with no step taken.
    search = lambda x: (0.97 if x < 140e3 else
                        0.88 + (x - 140e3) * 0.2e-6 if x < 260e3 else 0.80)
    got, _, _ = _stub_solve(search, lambda x: 0.90 - (x - 205e3) * 1e-6)
    assert got['newton_steps'] >= 1 and got['converged'], got
    assert abs(got['spend'] - 205e3) < 5e3, got['spend']


def test_the_guaranteed_income_flag_fires_on_a_SINGLE_policy_run():
    """It used to hang off the end of _policy_table, which only runs when more
    than one policy is compared — so the common case never saw it.

    Missed live on 2026-09-13: a VT benchmark solved to $61,390 against a
    $49,916 benefit. Re-solved, its portfolio_share is 0.048 — the portfolio
    funded 4.8% of the binding decade on the paths that set the level — and it
    printed beside 161's $196,500 (share 0.703) with nothing to separate them.
    """
    one = {'gk': {'102': {'spend': 61_390.0, 'portfolio_share': 0.048,
                          'floor': 0.85},
                  '161': {'spend': 196_500.0, 'portfolio_share': 0.703,
                          'floor': 0.85}}}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_retirement._guaranteed_floor_note(one, ['102', '161'])
    out = buf.getvalue()
    assert '102' in out, out
    assert '161' not in out.split('Rank these')[0], 'a real result must not be flagged'
    assert '5%' in out or '4%' in out, f'must report the measured share:\n{out}'

    # ...and run() must call it OUTSIDE the multi-policy branch
    src = inspect.getsource(report_retirement.run)
    call = src.find('_guaranteed_floor_note(')
    assert call != -1, 'run() never calls it'
    branch = src.find('_flat_table(specs, res, spec')
    assert branch != -1 and call > branch, \
        'the call must come after the table branch, so every mode reaches it'
    assert '_guaranteed_floor_note' not in inspect.getsource(
        report_retirement._policy_table), 'it must not live inside one table'


def test_a_saved_run_keeps_whether_its_level_was_a_portfolio_result():
    """A record that omits portfolio_share cannot tell a portfolio level from a
    Social Security one after the fact, and --diff would call them comparable."""
    for field in ('portfolio_share', 'converged', 'gap', 'newton_steps'):
        assert field in persistence.SCALARS, field


def test_head_to_head_states_the_SIZE_of_a_resolved_difference():
    """Pairing resolves effects the level table cannot see, so BETTER routinely
    means one percent — and a dollar range beside a six-figure base reads as a
    finding.

    On the run of 2026-09-13, 142 over 140 printed "+0.6k +12.8k BETTER". That
    is +0.15% to +3.10% of 140's $412,320, and the model's own arithmetic puts
    the edge at 1.01x over 58 years. 161 over 140 was +5.19% to +13.97% — an
    order of magnitude larger, and indistinguishable in the old format.
    """
    src = inspect.getsource(report_retirement)
    i = src.find('how much MORE lifetime average spending than')
    assert i != -1
    block_src = src[i:i + 2000]
    assert '% of base' in block_src, 'no size column'
    assert 'base_level' in block_src, 'the percentage needs the base to divide by'


def test_a_solved_level_reports_how_precise_it_is():
    """A plan level printed to three significant figures claims precision it
    does not have.

    Measured on the run of 2026-09-13: the same four portfolios said derisking
    at 65 was worth +4% / -6% / +8% / +4% under guyton_klinger and
    -15% / -9% / -5% / -11% under vpw, on identical paths. Re-solved, the gk
    level carries 1 SE of $8,037 on $168,667 — 4.8% — which is the entire
    guyton_klinger scatter. Without the band the reader has no way to see that.
    """
    # the quantile route: a dispersed statistic must give a non-trivial band
    stat = np.random.default_rng(0).normal(200_000, 60_000, 1500)
    se = solver._quantile_level_se(stat, 0.90, 0.85)
    assert se == se and se > 0, se
    # a degenerate statistic has no sampling error to speak of
    tight = solver._quantile_level_se(np.full(1500, 200_000.0), 0.90, 0.85)
    assert tight == tight and tight < 1.0, tight
    # too few observations to bootstrap -> nan, not a fake zero
    assert solver._quantile_level_se(np.zeros(3), 0.90, 0.85) != \
        solver._quantile_level_se(np.zeros(3), 0.90, 0.85)


def test_the_comparison_table_is_outcomes_only():
    """Every column left is something the paths PRODUCED, not something the
    solver chose."""
    r = {'spend': 200e3, 'level_se': 9e3, 'median_realised': 523e3,
         'p10_realised': 214e3, 'annual_vol': 14.2, 'depletion_rate': 0.001,
         'median_bequest': 44.9e6, 'worst_cut': 0.0,
         'twr_cagr': np.full(5, 0.0841), 'sortino_median': 1.042,
         'clipped': 0, 'floor': 0.85}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_retirement._policy_table(
            ['160'], {'gk': {'160': r}}, {'spend': 'solve', 'mode': 'retirement'})
    out = buf.getvalue()
    assert 'you spend' in out and '523k' in out, out
    assert 'vol' in out and '14.2%' in out, out
    assert '1 SE' not in out and '±9k' not in out, out
    assert 'year 1' not in out and 'plan level' not in out.split('rank on')[0], out

def test_every_gk_response_keeps_the_SAME_downside_rule():
    """The whole point: you choose participation, not protection.

    If a preset ever loosened `upper` or `cut`, the option would be selling a
    bigger pay cut as 'more upside' — which is exactly what fixed_pct does (a
    41.8% single-year cut in the 35%/40% crash test, against 10% here).
    """
    downside = {(v['upper'], v['cut']) for v in policies.GK_RESPONSE.values()}
    assert len(downside) == 1, policies.GK_RESPONSE
    assert downside == {(policies.GK_UPPER, policies.GK_CUT)}, downside
    # and the raise side must actually differ, or the knob does nothing
    assert len({(v['lower'], v['raise']) for v in policies.GK_RESPONSE.values()}) == 3


def test_the_default_response_is_fast_and_slow_is_the_original_rule():
    """`fast` is the default since 2026-10-09 — every measured run in the ledger (unpublished)
    used it, and sweeps built at the old `slow` default reported levels ~15%
    off the runs they were compared against. `slow` stays the original rule
    bit for bit, so a run that names it reproduces exactly."""
    original = {'upper': policies.GK_UPPER, 'lower': policies.GK_LOWER,
                'cut': policies.GK_CUT, 'raise': policies.GK_RAISE}
    fast = {'upper': 1.20, 'lower': 0.95, 'cut': 0.90, 'raise': 1.20}
    assert policies.GK_RESPONSE_DEFAULT == 'fast'
    assert policies.gk_guardrails(None) == fast
    assert policies.gk_guardrails('nonsense') == fast, 'must not raise'
    assert policies.gk_guardrails('slow') == original
    # a PolicyState built without guardrails must not quietly run another rule
    st = policies.PolicyState(1.0, 1.0, 0.03, 1.0, 65, 95, 0.035)
    assert {'upper': st.gk_upper, 'lower': st.gk_lower,
            'cut': st.gk_cut, 'raise': st.gk_raise} == fast


def test_guardrails_are_read_from_STATE_not_module_globals():
    """Workers re-import policies under `spawn`, so a value assigned in the
    parent never reaches them — the trap that put REBALANCING_BANDS in
    engine.py. The policy must read the state it is handed."""
    src = inspect.getsource(policies.guyton_klinger)
    for g in ('GK_UPPER', 'GK_LOWER', 'GK_CUT', 'GK_RAISE'):
        assert g not in src, f'{g} read as a module global in guyton_klinger'
    for f in ('state.gk_upper', 'state.gk_lower', 'state.gk_cut', 'state.gk_raise'):
        assert f in src, f


def test_a_faster_response_raises_spending_sooner():
    """Behavioural, not structural: same balance, same prior spend, three rules."""
    def step(resp, balance, prior, initial_rate=0.030):
        g = policies.gk_guardrails(resp)
        st = policies.PolicyState(prior, 1.0, initial_rate, prior, 65, 95, 0.035,
                                  **{f'gk_{k}': v for k, v in g.items()})
        return policies.guyton_klinger(70, balance, st)
    # portfolio has outgrown spending: rate is 12% below plan
    bal, prior = 240_000 / (0.030 * 0.88), 240_000.0
    slow, medium, fast = (step(r, bal, prior) for r in ('slow', 'medium', 'fast'))
    assert slow == prior, 'a 12% drift must not trigger the shipped rule'
    assert medium > prior and fast > medium, (slow, medium, fast)
    # ...and the CUT side is untouched at every setting
    over = 240_000 / (0.030 * 1.30)           # rate 30% above plan
    cuts = [step(r, over, prior) for r in ('slow', 'medium', 'fast')]
    assert len(set(cuts)) == 1, cuts
    assert abs(cuts[0] / prior - policies.GK_CUT) < 1e-12, cuts


def test_the_response_reaches_the_engine_and_the_run_record():
    src = inspect.getsource(pa.run_single_simulation)
    assert 'gk_guardrails' in src, 'engine must resolve it inside the worker'
    assert "wd_cfg.get('gk_response')" in src, 'must come from the pickled config'
    pr = persistence.provenance({'withdrawal': {'floor': 0.85, 'gk_response': 'fast'},
                                 'age': 37, 'time_horizon_years': 58})
    assert pr['gk_response'] == 'fast', pr
    bare = persistence.provenance({'withdrawal': {'floor': 0.85},
                                   'age': 37, 'time_horizon_years': 58})
    assert bare['gk_response'] == policies.GK_RESPONSE_DEFAULT, bare


def test_the_response_is_sweepable():
    import tools.sensitivity as sens
    assert 'gk_response' in sens.DIMENSIONS, sorted(sens.DIMENSIONS)
    d = sens.DIMENSIONS['gk_response']
    assert d.kind == 'named', d.kind
    assert set(d.default.split(',')) == set(policies.GK_RESPONSE), d.default


def test_comparing_gk_responses_actually_fans_out():
    """A multi-select that is stored and never read is a silent drop.

    Caught while rendering the prompt: `compare_gk_responses` reached the spec
    and nothing consumed it, so picking 'medium,fast' would have solved medium
    and thrown fast away with no message. Same class as the policy prompt that
    offered four policies while the runner hardcoded one.
    """
    spec = {'mode': 'retirement', 'policy': 'guyton_klinger',
            'compare_gk_responses': ['slow', 'medium', 'fast']}
    runs = runner.policy_runs(spec)
    assert len(runs) == 3, runs
    assert [s['gk_response'] for _, _, s in runs] == ['slow', 'medium', 'fast']
    assert len({t for t, _, _ in runs}) == 3, 'each run needs its own tag'
    # ...and it composes with the vpw reserve fan-out rather than replacing it
    both = runner.policy_runs({
        'mode': 'retirement', 'compare_policies': ['guyton_klinger', 'vpw'],
        'compare_gk_responses': ['slow', 'fast'], 'compare_vpw_bequests': [0, 5e6]})
    assert len(both) == 4, both
    assert sum(p == 'guyton_klinger' for _, p, _ in both) == 2, both
    assert sum(p == 'vpw' for _, p, _ in both) == 2, both


def test_a_single_response_does_not_fan_out():
    """One choice must stay one run, tagged plainly."""
    runs = runner.policy_runs({'mode': 'retirement', 'policy': 'guyton_klinger',
                               'gk_response': 'fast'})
    assert len(runs) == 1 and runs[0][0] == 'guyton_klinger', runs
    assert runs[0][2]['gk_response'] == 'fast', runs


def test_per_run_settings_travel_as_a_dict_not_a_positional_float():
    """The third element was a bare reserve while vpw's was the only per-run
    setting. A second positional float is a mis-assignment waiting to happen."""
    _, _, settings = runner.policy_runs(
        {'mode': 'retirement', 'policy': 'vpw', 'vpw_bequest': 5e6})[0]
    assert isinstance(settings, dict), settings
    assert settings['vpw_bequest'] == 5e6, settings
    src = inspect.getsource(report_retirement.run)
    assert 'for tag, policy, settings in runs' in src, src[:0] or 'consumer not updated'
    assert 'if v is not None' in src, 'a None setting must not clobber build_config'


def test_sortino_note_does_not_say_N_of_the_N():
    """"8 of the 8 other rows" is a fraction whose numerator equals its
    denominator — the one case where a count is the wrong phrasing."""
    src = inspect.getsource(report_retirement._sortino_note)
    assert 'ANY of the' in src, 'the all-unresolved case needs its own wording'
    assert 'len(unresolved) == others' in src, src


def test_the_derisk_section_says_when_its_baseline_is_unresolved():
    """The derisk verdict hangs off ONE portfolio, picked by the plan-level
    ranking — the noisiest column in the report.

    Observed 2026-09-13 on 140/160/152: the plan level called 160 best, the
    life-stage table ranked 152 first, and the head-to-head resolved neither
    against the other, while the header stated "the best portfolio, 160" as
    plain fact.
    """
    src = inspect.getsource(report_retirement._head_to_head) \
        if hasattr(report_retirement, '_head_to_head') else None
    if src is None:                      # the block lives in the flat table
        src = inspect.getsource(report_retirement)
    assert 'conditional' in src, 'no conditionality stated'
    assert 'rivals = [first[b] for b in bases' in src, 'rivals never computed'
    assert '_straddles_zero(engine.paired_bootstrap_diff(' in src
    # and the caveat must be a separate argument, not glued into the title:
    # the combined string ran 110 characters.
    assert 'def block(title, base, rows, caveat=None):' in src, src[:0] or 'block signature'


def test_crossing_reports_which_edge_it_clipped_to():
    """A clipped crossing is not an estimate — the window is saying the answer
    lies past it. The caller has to be able to tell the two apart."""
    xs = np.array([100.0, 200.0, 300.0, 400.0])
    assert solver._crossing(xs, np.array([.98, .94, .90, .86]), .90)[2] == 0
    assert solver._crossing(xs, np.array([.80, .78, .76, .74]), .90)[2] == -1
    assert solver._crossing(xs, np.array([.99, .98, .97, .96]), .90)[2] == +1
    # the clipped value itself must still land ON the edge, not past it
    lo_x, _, lo_c = solver._crossing(xs, np.array([.80, .78, .76, .74]), .90)
    assert lo_c == -1 and lo_x == xs[0], (lo_x, lo_c)
    # ...including when the local fit is FLAT. Pooled reads that all sit
    # under the target fell to interpolation, which returned whichever tied
    # point sorted last — an interior spend, unflagged — so the solver never
    # reopened the bracket downward.
    assert solver._crossing(xs, np.array([.85, .85, .85, .81]), .90)[::2] == (100.0, -1)
    assert solver._crossing(xs, np.array([.95, .95, .95, .95]), .90)[::2] == (400.0, +1)


def test_a_clipped_level_is_never_accepted_on_tolerance_alone():
    """The 1pp tolerance means 'close enough to the target', which is a claim
    about a crossing that was ESTIMATED. An edge is not one.

    Measured 2026-09-13 on a 15-strategy run: A/B/C/L/M/N/O all reported
    exactly $200,000, none took a verification step, and every flagged row came
    in at 89% against a 90% target — ten rows, NONE above 90%, p = 0.001 under
    a fair coin. That is a systematic overstatement, not sampling noise.
    """
    # Every search read is below 90%, and the lowest — the window's edge at
    # 170k — is within a standard error of it, so the window is not reopened
    # and the crossing clips to 170k. Verified there it reads 89.9%: inside the
    # tolerance, and a step of only $1k.
    search = lambda x: 0.88 - (x - 170_000.0) * 1e-6
    got, checked, _ = _stub_solve(search, lambda x: search(x) + 0.019)
    # the outcome is recorded, so a reader can see the level began on an edge
    assert got['clipped'] == 1, got
    # neither the tolerance nor the small-step break may veto the forced step
    assert got['newton_steps'] >= 1 and len(checked) >= 2, \
        (got['newton_steps'], checked)
    assert got['spend'] < 169_500.0, got['spend']


def test_a_held_bracket_edge_says_so_instead_of_printing_nothing():
    """Forcing the step made the solver correct; it did not make it legible.

    Observed 2026-09-13 on 160: the clipped level took its step, the step did
    not improve, the loop broke BEFORE the print, and $200,000 appeared with no
    line under it — indistinguishable from the seven unverified rows the forced
    step exists to prevent.
    """
    # the edge of the clipped fixture above, where the forced step lands WORSE
    got, checked, out = _stub_solve(
        lambda x: 0.88 - (x - 170_000.0) * 1e-6,
        lambda x: 0.899 if abs(x - 170_000.0) < 1.0 else 0.80, verbose=True)
    assert abs(got['spend'] - 170_000.0) < 1.0 and len(checked) == 2, \
        (got['spend'], checked)
    held = [l for l in out.splitlines() if 'held' in l]
    assert held and '$170,000' in held[0] and 'bracket edge' in held[0], out[-800:]


def test_the_bar_and_its_caveats_live_in_ONE_place():
    """`year 1` and its standard error were removed from the comparison tables:
    a table for CHOOSING between strategies should hold outcomes, not the knob
    that produced them — and the bar was the noisiest column on the page while
    the report told the reader four times not to rank on it.

    It is an instruction for the strategy you pick, so it lives in YOUR PLAN IN
    CONTEXT with its precision and its clip warning attached. The warning
    followed it: a `~` marking a column that no longer exists would mark
    nothing.
    """
    res = {'160': {'spend': 200_000.0, 'level_se': 9_000.0, 'clipped': 1,
                   'balance_at_retirement': {10: 3.1e6, 50: 7.3e6, 90: 19.1e6}}}
    cfg = {'age': 37, 'time_horizon_years': 58, 'retirement_age': 65,
           'withdrawal': {'start_age': 65}, 'social_security': None}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_retirement._plan_in_context(
            ['160'], res, cfg, spec={'policy': 'guyton_klinger'})
    out = buf.getvalue()
    assert 'year 1' in out, out
    assert '±$9k' in out, out
    assert 'BRACKET EDGE' in out, out
    # ...and the comparison tables must NOT carry them any more
    for fn in (report_retirement._policy_table, report_retirement._flat_table,
               report_retirement._matrix):
        src = inspect.getsource(fn)
        cols = re.search(r"p?cols = \(?\[(.*?)\]", src, re.S)
        assert cols and "'se'" not in cols.group(1), fn.__name__
        assert "'level'" not in cols.group(1) and "'year1'" not in cols.group(1), \
            fn.__name__

def test_a_saved_run_records_whether_the_level_was_an_edge():
    for f in ('clipped', 'level_se'):
        assert f in persistence.SCALARS, f


def test_in_detail_leads_with_what_you_spend():
    """The two columns a reader conflated — the certified bar and realised
    spending — are no longer adjacent, because the bar is no longer here."""
    P = _P()

    def row(sp, med, p10):
        return {'spend': sp, 'median_realised': med, 'p10_realised': p10,
                'annual_vol': 14.0, 'median_bequest': 45e6, 'worst_cut': 0.0,
                'twr_cagr': np.full(10, 0.084), 'sortino_median': 1.04,
                'clipped': 0, 'level_se': 9e3, 'depletion_rate': 0.0,
                'success_rate': 0.90, 'sortino': np.full(200, 1.04),
                'per_path_mean': np.full(200, med)}
    res = {'140': row(200_000.0, 519e3, 213e3),
           '140>140+5@65': row(191_000.0, 505e3, 197e3)}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_retirement._matrix(['140', '140>140+5@65'], res, {
            'spend': 'solve', 'mode': 'retirement', 'portfolios': ['140'],
            'derisk_at': 65, 'bond_tiers': [0.05]}, P)
    detail = buf.getvalue()[buf.getvalue().index('IN DETAIL'):]
    hdr = next(l for l in detail.splitlines() if 'you spend' in l)
    assert hdr.strip().startswith('strategy'), hdr
    assert 'you spend' in hdr and 'low case' in hdr, hdr
    assert 'year 1' not in hdr and 'plan level' not in hdr, hdr
    assert ' spend ' not in hdr.replace('you spend', ''), hdr

def test_plan_in_context_states_the_withdrawal_rate():
    """The report never printed the one ratio retirement advice is stated in.

    Measured on the live plan: a $190k level is 2.6% of a $7.3M median balance,
    against the 4% rule's $290k — and Social Security is ~$1.1M of
    inflation-linked bond at 65 (see the test below for how it is valued).
    """
    res = {'160': {'spend': 200_000.0,
                   'balance_at_retirement': {10: 3.1e6, 50: 7.3e6, 90: 19.1e6}}}
    cfg = {'age': 37, 'time_horizon_years': 58, 'retirement_age': 65,
           'withdrawal': {'start_age': 65},
           'social_security': {'annual_at_fra': 60000.0, 'claim_age': 70.0,
                               'benefit_scale': 1.0}}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_retirement._plan_in_context(['160'], res, cfg)
    out = buf.getvalue()
    assert '2.7%' in out or '2.6%' in out, out       # level / median balance
    assert '4% rule' in out and '292k' in out or '290k' in out, out
    assert 'inflation-linked BOND' in out, out
    assert '%/' in out, 'must restate the true equity/bond split'
    # no balance -> no block, rather than a divide by zero
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        report_retirement._plan_in_context(['160'], {'160': {'spend': 1.0}}, cfg)
    assert buf2.getvalue() == '', buf2.getvalue()


def test_the_retirement_balance_travels_out_of_the_worker():
    """PolicyState already pins it; it just was not returned. One float a path."""
    src = inspect.getsource(pa.run_single_simulation)
    assert "'v_at_retirement'" in src, 'engine must return it'
    assert 'wd_state.v_at_retirement' in src, src[:0] or 'not taken from the state'
    ssrc = inspect.getsource(solver)
    assert "'balance_at_retirement'" in ssrc, 'solver must summarise it'
    assert 'nanpercentile' in ssrc, 'a depleted/absent path must not poison it'


def test_all_three_level_tables_share_one_column_spec():
    """There were three tables printing the same quantities under three sets of
    names — `spend` here and `plan level` there, `worst yr` and `worst cut`,
    the standard error a COLUMN in one and a footnote in the next. Adding a
    column meant editing three places and the names drifted between them.
    """
    for fn in (report_retirement._policy_table, report_retirement._flat_table,
               report_retirement._matrix):
        src = inspect.getsource(fn)
        assert '_level_row(' in src, f'{fn.__name__} formats its own rows'
        assert '_level_header(' in src, f'{fn.__name__} writes its own header'
    # every column a table asks for must exist in the spec
    for fn in (report_retirement._policy_table, report_retirement._flat_table,
               report_retirement._matrix):
        for key in re.findall(r"'(level|se|p10|vol|ruin|bequest|worst|cagr|sortino)'",
                              inspect.getsource(fn)):
            assert key in report_retirement._LEVEL_COLUMNS, key


def test_one_width_governs_the_header_and_the_row():
    """The first version let each formatter choose its own padding while the
    header used the declared width. They disagreed immediately: the plan level
    ran into the standard error as `200k~±   9k`."""
    r = {'spend': 200e3, 'clipped': 1, 'level_se': 9e3, 'p10_realised': 214e3,
         'annual_vol': 14.2, 'depletion_rate': 0.001, 'median_bequest': 44.9e6,
         'worst_cut': 0.0, 'twr_cagr': np.full(5, 0.0841), 'sortino_median': 1.042}
    cols = ['level', 'se', 'p10', 'vol', 'ruin', 'bequest', 'cagr', 'sortino']
    top, sub, wide = report_retirement._level_header(cols, 'strategy', 22)
    line = report_retirement._level_row(r, cols, '160', 22)
    assert len(line) == wide, (len(line), wide, line)
    # and the numbers must not touch — every column keeps a gap
    assert '200k~ ' in line + ' ', line
    assert '~±' not in line, line


def test_the_retirement_table_reports_annualised_volatility():
    """It says what the RIDE was like rather than where it ended. The
    accumulation path has always reported it; the retirement path never did."""
    cfg = {'age': 37, 'time_horizon_years': 58, 'withdrawal': {'start_age': 65}}
    raw = {'spending': np.full((50, 30), 2e5), 'depleted': np.zeros(50, bool),
           'finals': np.full(50, 4e7), 'depletion_ages': [None] * 50,
           'returns': np.random.default_rng(0).normal(0.007, 0.04, (50, 600))}
    out = solver.score(raw, cfg, 2e5, 0.85, 'per_band')
    assert 'annual_vol' in out, sorted(out)
    assert 13.0 < out['annual_vol'] < 15.0, out['annual_vol']
    # a run without the return series must not invent one
    raw2 = dict(raw); raw2.pop('returns')
    assert 'annual_vol' not in solver.score(raw2, cfg, 2e5, 0.85, 'per_band')


def test_the_table_shows_the_median_spend_not_only_the_bad_case():
    """`median_realised` was computed by the solver from the start and displayed
    NOWHERE. The table showed how bad it gets and never what it usually is."""
    cols = report_retirement._LEVEL_COLUMNS
    assert 'median' in cols, sorted(cols)
    r = {'median_realised': 523_000.0, 'p10_realised': 214_000.0}
    assert cols['median'][3](r) == '523k', cols['median'][3](r)
    assert cols['p10'][3](r) == '214k', cols['p10'][3](r)
    # a result without it must render a dash, not crash or invent one
    assert cols['median'][3]({'p10_realised': 1.0}) == '-'


def test_a_self_directed_policy_forbids_the_year_1_label():
    """vpw computes its own opening draw — the target is a bar it is judged
    against, not a control. In a run containing one the column cannot be headed
    `year 1` for any row without being wrong for that one.

    Matched on the BASE policy: a row is tagged `vpw reserve 5M`, not `vpw`.
    Keying on the full tag is what silently suppressed the policy-tradeoff block
    once the reserve frontier fanned rows out, and it failed here first time too.
    """
    k = report_retirement._level_key
    assert k({'policy': 'guyton_klinger'}) == 'year1'
    assert k({'policy': 'vpw'}) == 'level'
    assert k({'compare_policies': ['guyton_klinger', 'vpw']}) == 'level'
    assert k({}, {'guyton_klinger': 1}) == 'year1'
    assert k({}, {'guyton_klinger': 1, 'constant_real': 1}) == 'year1'
    # the tagged forms, which is where the naive version broke
    assert k({}, {'guyton_klinger': 1, 'vpw reserve 5M': 1}) == 'level'
    assert k({}, {'vpw no reserve': 1}) == 'level'
    # a policy whose NAME merely starts with a self-directed one must not match
    assert k({}, {'vpwish': 1}) == 'year1'


def test_the_bracket_starts_from_the_balance_not_from_50k():
    """Doubling from $50,000 put the probes at 50/100/200/400k, so the window
    handed to the sampler was a FACTOR OF TWO wide and landed wherever the
    doubling stopped. For a $7.4M book the first three probes carried no
    information, and 7 of 15 levels came back sitting on an edge — every one of
    them at exactly $200,000.

    The balance at retirement does not depend on the spend level, so one probe
    reveals it and 4% of it is a far better start.
    """
    src = inspect.getsource(solver.solve_sustainable_spend)
    assert 'hi = 50_000.0' not in src, 'the uninformed seed is back'
    assert "balance_at_retirement" in src, 'the bracket must read the balance'
    assert '0.04 * bal' in src, 'the 4% rule is the documented seed'
    # through evaluate, or every test that stubs it is bypassed
    i = src.index('balance_at_retirement')
    assert 'evaluate(' in src[max(0, i - 400):i], src[max(0, i - 400):i]


def test_the_sampled_window_is_padded_downward_only():
    """The search curve sits ~2pp BELOW the verification curve, so a straddle
    built on search reads clips at its LOW edge. Measured: 4 of 8 strategies
    clipped after the informed bracket landed, every one at the low end;
    padding only downward took it to 1 of 8."""
    src = inspect.getsource(solver.solve_sustainable_spend)
    assert 'lo = max(lo * 0.85, 0.0)' in src, 'no downward pad'
    # ...and nothing widens the HIGH side, which would only re-introduce the
    # coarse window the informed bracket exists to avoid
    assert 'hi = hi * 1.15' not in src and 'hi *= 1.15' not in src


def test_the_bracket_finds_a_known_crossing_without_walking_far():
    """Behavioural: a stubbed curve with a known 90% point."""
    calls = []
    TRUE = 250_000.0

    def fake(plan, config, spend, n_sims=500, floor=0.85, parallel=True,
             full=False, adequacy=solver.DEFAULT_ADEQUACY, seed_offset=0):
        calls.append(float(spend))
        rate = float(np.clip(0.90 - (spend - TRUE) * 1.2e-6, 0.0, 1.0))
        return {'spend': float(spend), 'success_rate': rate,
                'success_by_floor': {}, 'depletion_rate': 0.0,
                'balance_at_retirement': {10: 3.1e6, 50: 7.4e6, 90: 19e6},
                'adequacy_stat': np.zeros(10), 'spending': np.zeros((10, 30))}

    real = solver.evaluate
    try:
        solver.evaluate = fake
        r = solver.solve_sustainable_spend({}, {'age': 37, 'time_horizon_years': 58,
                                            'withdrawal': {'start_age': 65}},
                                       search_sims=100, verify_sims=100,
                                       verbose=False)
    finally:
        solver.evaluate = real
    assert abs(r['spend'] - TRUE) / TRUE < 0.05, (r['spend'], TRUE)
    # the seed is 4% of 7.4M = 296k, one 1.5x step from the answer
    assert any(abs(c - 296_000.0) < 1_000.0 for c in calls), calls[:6]
    assert len(calls) < 30, f'{len(calls)} evaluations to bracket and fit'


def test_the_bracket_walks_down_to_a_crossing_below_its_informed_guess():
    """Every other stubbed-evaluate test puts the crossing at or above the
    informed guess, so the downward walk's success exit (the first lo that
    clears the bracket floor) never ran: only the guess-already-failing entry
    was reached. Here the balance implies a 296k guess against a 120k crossing.
    Without the exit the walk runs all ten steps, hands the sampler a window
    of a few thousand dollars, and the answer comes back near 5k."""
    calls = []
    TRUE = 120_000.0

    def fake(plan, config, spend, n_sims=500, floor=0.85, parallel=True,
             full=False, adequacy=solver.DEFAULT_ADEQUACY, seed_offset=0):
        calls.append(float(spend))
        rate = float(np.clip(0.90 - (spend - TRUE) * 1.2e-6, 0.0, 1.0))
        return {'spend': float(spend), 'success_rate': rate,
                'success_by_floor': {}, 'depletion_rate': 0.0,
                'balance_at_retirement': {10: 3.1e6, 50: 7.4e6, 90: 19e6},
                'adequacy_stat': np.zeros(10), 'spending': np.zeros((10, 30))}

    real = solver.evaluate
    try:
        solver.evaluate = fake
        r = solver.solve_sustainable_spend({}, {'age': 37, 'time_horizon_years': 58,
                                            'withdrawal': {'start_age': 65}},
                                       search_sims=100, verify_sims=100,
                                       verbose=False)
    finally:
        solver.evaluate = real
    assert abs(r['spend'] - TRUE) / TRUE < 0.05, (r['spend'], TRUE)
    guess = 0.04 * 7.4e6
    assert any(abs(c - guess) < 1.0 for c in calls), 'the informed guess was never probed'
    # 296k -> 197k fails the bracket floor, 131.6k clears it: the walk stops there
    assert any(abs(c - guess / 1.5 ** 2) < 1.0 for c in calls), calls[:6]
    assert all(abs(c - guess / 1.5 ** 3) > 1.0 for c in calls), \
        'the walk went past the first passing level'
    assert min(calls) > TRUE / 2, min(calls)
    assert len(calls) < 30, f'{len(calls)} evaluations to bracket and fit'


def test_each_spend_level_enters_the_crossing_fit_once():
    """A probe and a sample at the same spend run the same seeds, so they are
    ONE read, not two. The search appended the probes to the fit twice, and the
    sampler's endpoints repeat the bracket's probes, so some reads counted two
    or three times in the local least squares — weight no measurement earned."""
    seen = []

    def fake(plan, config, spend, n_sims=500, floor=0.85, parallel=True,
             full=False, adequacy=solver.DEFAULT_ADEQUACY, seed_offset=0):
        rate = float(np.clip(0.90 - (spend - 250_000.0) * 1.2e-6, 0.0, 1.0))
        return {'spend': float(spend), 'success_rate': rate,
                'success_by_floor': {}, 'depletion_rate': 0.0,
                'balance_at_retirement': {10: 3.1e6, 50: 7.4e6, 90: 19e6},
                'adequacy_stat': np.zeros(10), 'spending': np.zeros((10, 30))}

    real_eval, real_cross = solver.evaluate, solver._crossing

    def recording(xs, ys, target):
        seen.append(list(xs))
        return real_cross(xs, ys, target)

    try:
        solver.evaluate, solver._crossing = fake, recording
        solver.solve_sustainable_spend({}, {'age': 37, 'time_horizon_years': 58,
                                        'withdrawal': {'start_age': 65}},
                                   search_sims=100, verify_sims=100, verbose=False)
    finally:
        solver.evaluate, solver._crossing = real_eval, real_cross
    assert seen, 'the search never fitted a crossing'
    for xs in seen:
        assert len(xs) == len(set(xs)), sorted(xs)


# --- long-term care -----------------------------------------------------------------

def test_ltc_is_a_window_not_a_lump_and_is_off_by_default():
    c = {'annual_amount': 120_000.0, 'start_age': 82.0, 'years': 3.0}
    assert [policies.ltc_cost(a, c) for a in (80, 81, 82, 83, 84, 85)] == \
        [0.0, 0.0, 120_000.0, 120_000.0, 120_000.0, 0.0]
    for off in (None, {}, {'annual_amount': 0.0}):
        assert policies.ltc_cost(85, off) == 0.0, off
    assert policies.LTC_DEFAULTS['annual_amount'] == 0.0, 'must ship OFF'


def test_care_costs_drain_the_portfolio_but_are_not_counted_as_spending():
    """A nursing home is a cost, not consumption. Adding it to `total_spend`
    would make a $120k shock read as a $120k RAISE in the adequacy statistic,
    and the per-band score would reward the very thing being tested."""
    src = inspect.getsource(pa.run_single_simulation)
    assert 'ltc_cost' in src, 'engine never asks for it'
    assert 'from_portfolio += care' in src, 'care must come out of the portfolio'
    assert 'drawn_care = min(care, drawn)' in src, \
        'a balance-capped year must short the lifestyle, not the nursing home'
    i = src.index('total_spend = ')
    assert '(drawn - drawn_care) + ss' in src[i:i + 80], src[i:i + 80]


def test_ltc_is_deterministic_and_swept_not_drawn_per_path():
    """Same reasoning as `benefit_scale` and mortality: a 20%-chance-of-$500k
    shock averages away across paths and leaves the 10th percentile almost
    untouched, which makes the plan look safer than a plausible outcome
    warrants."""
    # the BODY, not the docstring — which explains why there is no random
    # draw and so contains the word.
    src = inspect.getsource(policies.ltc_cost)
    body = src.split('"""')[-1]
    for forbidden in ('random', 'rng(', 'uniform', 'choice('):
        assert forbidden not in body, f'{forbidden} in a scenario knob'
    import tools.sensitivity as sens
    assert 'ltc_shock' in sens.DIMENSIONS, sorted(sens.DIMENSIONS)
    assert sens.DIMENSIONS['ltc_shock'].kind == 'numeric', 'a cost has an ordering'
    cfg = {'withdrawal': {'start_age': 65}}
    sens.apply_dim(cfg, 'ltc_shock', 120_000.0)
    assert cfg['withdrawal']['ltc']['annual_amount'] == 120_000.0
    assert cfg['withdrawal']['ltc']['years'] == policies.LTC_DEFAULTS['years']


def test_the_headline_never_announces_a_loser_as_a_winner():
    """`cells` ranks on the PLAN LEVEL; the margin is measured on REALISED
    spending. The two can disagree, and the code assumed they could not.

    Observed 2026-09-14: 152>152+5@65 led on the level by $1k — one fourteenth
    of the ±$14k the same run printed as its own standard error — while
    spending $53k LESS. The line read
        best: 152>152+5@65   (beats 152 by [-53.4, -5.5]k)
    a wholly negative margin announced as a win.
    """
    rng = np.random.default_rng(0)
    base = rng.normal(496e3, 60e3, 600)
    res = {'A': {'per_path_mean': base, 'level_se': 14e3},
           'B': {'per_path_mean': base - 30e3 + rng.normal(0, 3e3, 600),
                 'level_se': 14e3}}
    # B leads on the level and LOSES on spending
    out = report_retirement._best_caveat([(187e3, 'B'), (186e3, 'A')], res,
                                         {'mode': 'retirement'})
    assert 'SPENDS MORE' in out, out
    assert 'beats' not in out, out
    assert '±$14k' in out, 'the level band belongs in the warning'
    # the ordinary case still reads as a win
    res2 = {'A': {'per_path_mean': base},
            'B': {'per_path_mean': base + 40e3 + rng.normal(0, 3e3, 600)}}
    win = report_retirement._best_caveat([(1, 'B'), (0, 'A')], res2,
                                         {'mode': 'retirement'})
    assert 'beats A by [+' in win, win


def test_the_floor_guidance_quotes_THIS_run(): 
    """It read "At 77%, roughly one path in four" under a run whose strictest
    column was 80-82% — the same class of hardcoded figure as the premium span
    that kept printing a retracted number for two days."""
    res = {'a': {'spend': 1e5, 'floor': 0.85, 'worst_cut': 0.0,
                 'success_by_floor': {0.7: 0.94, 0.85: 0.92, 1.0: 0.80}}}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_retirement.robustness({'gk': res}, ['a'], confidence=0.90)
    out = buf.getvalue()
    assert 'Yours reads 80%' in out, out
    assert 'one path in 5' in out, out
    assert 'At 77%' not in out, 'the hardcoded example is back'
    # and a run with no floor data must not fabricate one
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        report_retirement._floor_guidance(solver.FLOOR_GRID, None)
    assert 'Yours reads' not in buf2.getvalue(), buf2.getvalue()


# --- rebalancing turnover -----------------------------------------------------------

def test_traded_dollars_is_the_one_way_figure():
    """Every dollar moved appears twice — once as a sell, once as a buy."""
    before = {'A': 100.0, 'B': 100.0}
    after = {'A': 120.0, 'B': 80.0}          # $20 moved from B to A
    assert pa.traded_dollars(before, after) == 20.0
    assert pa.traded_dollars(before, before) == 0.0, 'no trade, no turnover'
    # a ticker appearing on only one side still counts
    assert pa.traded_dollars({'A': 100.0}, {'A': 50.0, 'B': 50.0}) == 50.0


def _book(weights, total=1_000.0):
    from montecarlo.loaders import MODEL_TICKERS
    return {t: total * weights.get(t, 0.0) for t in MODEL_TICKERS}


def test_a_partial_rebalance_sells_an_OVERWEIGHT_breach_into_the_most_underweight_fund():
    """The partial trade assumed the breached fund was UNDERWEIGHT: it set the
    breach to target and took the difference out of the most overweight fund.
    When the breach was itself the most overweight fund the two cancelled — the
    book came back unchanged, logged as 'Sold 15.0% VOO, Bought -15.0% VOO' and
    counted as a rebalance, ten times a quarter. When another fund was more
    overweight, the sale of the breach was BOUGHT INTO that fund."""
    target = _book({'VOO': 0.40, 'AVUV': 0.60}, total=1.0)

    # the breach is the most overweight fund
    vals = _book({'VOO': 0.55, 'AVUV': 0.45})
    after, _ = pa.execute_partial_rebalance(vals, target, ['VOO'])
    assert abs(sum(after.values()) - 1_000.0) < 1e-9, 'a rebalance conserves value'
    assert abs(after['VOO'] - 400.0) < 1e-9, after['VOO']
    assert abs(after['AVUV'] - 600.0) < 1e-9, after['AVUV']

    # another fund is MORE overweight: the sale goes to the underweight fund
    target = _book({'VOO': 0.40, 'AVUV': 0.30, 'AVLV': 0.30}, total=1.0)
    vals = _book({'VOO': 0.50, 'AVUV': 0.45, 'AVLV': 0.05})
    after, _ = pa.execute_partial_rebalance(vals, target, ['VOO'])
    assert abs(sum(after.values()) - 1_000.0) < 1e-9, 'a rebalance conserves value'
    assert abs(after['VOO'] - 400.0) < 1e-9, after['VOO']
    assert abs(after['AVUV'] - 450.0) < 1e-9, 'the overweight bystander is untouched'
    assert abs(after['AVLV'] - 150.0) < 1e-9, after['AVLV']


def test_a_partial_rebalance_funds_an_UNDERWEIGHT_breach_from_the_most_overweight_fund():
    target = _book({'VOO': 0.40, 'AVUV': 0.30, 'AVLV': 0.30}, total=1.0)
    vals = _book({'VOO': 0.30, 'AVUV': 0.45, 'AVLV': 0.25})
    after, _ = pa.execute_partial_rebalance(vals, target, ['VOO'])
    assert abs(sum(after.values()) - 1_000.0) < 1e-9, 'a rebalance conserves value'
    assert abs(after['VOO'] - 400.0) < 1e-9, after['VOO']
    assert abs(after['AVUV'] - 350.0) < 1e-9, after['AVUV']
    assert abs(after['AVLV'] - 250.0) < 1e-9, 'the underweight bystander is untouched'


def test_rebalance_decision_follows_its_decision_tree():
    """Bands (real config): VOO 40% +/-6pp, AVUV 30% +/-10.5pp, AVLV 30% +/-9pp;
    in the four-fund book AVUV/AVLV/VFMO are 20% +/-7/6/6pp."""
    vols = {t: 0.15 for t in pa.TICKERS}
    corr = np.eye(len(pa.TICKERS))
    three = _book({'VOO': 0.40, 'AVUV': 0.30, 'AVLV': 0.30}, total=1.0)
    four = _book({'VOO': 0.40, 'AVUV': 0.20, 'AVLV': 0.20, 'VFMO': 0.20}, total=1.0)

    def decide(current, target, month):
        return pa.rebalance_decision(current, target, month, vols, corr)[0]

    on_target = dict(three)
    one_breach = _book({'VOO': 0.47, 'AVUV': 0.27, 'AVLV': 0.26}, total=1.0)
    two_breaches = _book({'VOO': 0.47, 'AVUV': 0.19, 'AVLV': 0.34}, total=1.0)
    near_not_breached = _book({'VOO': 0.45, 'AVUV': 0.25, 'AVLV': 0.30}, total=1.0)
    breach_plus_two_near = _book(
        {'VOO': 0.47, 'AVUV': 0.15, 'AVLV': 0.155, 'VFMO': 0.225}, total=1.0)

    # Q1-Q3: bands only
    assert decide(on_target, three, 6) == 'NONE'
    assert decide(near_not_breached, three, 6) == 'NONE', 'no drift check before Q4'
    assert decide(one_breach, three, 6) == 'PARTIAL'
    assert decide(two_breaches, three, 6) == 'FULL'
    assert decide(breach_plus_two_near, four, 6) == 'FULL'
    # Q4: a breach or >70% of a band is a FULL, never a PARTIAL
    assert decide(one_breach, three, 12) == 'FULL'
    assert decide(near_not_breached, three, 12) == 'FULL'
    # A breach is past 70% of its band too; the trigger tally tells them apart
    reason = pa.rebalance_decision(one_breach, three, 12, vols, corr)[1]
    assert reason.startswith('Q4 threshold breach'), reason
    reason = pa.rebalance_decision(near_not_breached, three, 12, vols, corr)[1]
    assert reason.startswith('Q4 annual drift'), reason
    assert decide(on_target, three, 12) == 'NONE'


def test_the_Q4_drift_check_fires_on_a_volatility_shift_with_every_band_inside_70_percent():
    """Trigger 2 of the annual drift check. The decision-tree test uses uniform
    vols, so only the band-consumption trigger was ever reached: a sign flip, a
    dropped `- 1.0` or a moved threshold in the vol branch passed everything.
    Same book, same bands; only AVUV's volatility moves, so the portfolio is
    +11.9% riskier than its target in one case and +9.1% in the other."""
    corr = np.eye(len(pa.TICKERS))
    target = _book({'VOO': 0.40, 'AVUV': 0.30, 'AVLV': 0.30}, total=1.0)
    drifted = _book({'VOO': 0.36, 'AVUV': 0.36, 'AVLV': 0.28}, total=1.0)
    for t in ('VOO', 'AVUV', 'AVLV'):
        used = pa.calculate_band_consumption(drifted[t], target[t], t)
        assert used < 0.70, f'{t} at {used:.2f} of its band would fire Trigger 1 instead'

    def vols(avuv):
        v = {t: 0.0 for t in pa.TICKERS}
        v.update(VOO=0.10, AVUV=avuv, AVLV=0.10)
        return v

    def shift(avuv):
        v = vols(avuv)
        return (pa.calculate_portfolio_volatility(drifted, v, corr)
                / pa.calculate_portfolio_volatility(target, v, corr) - 1.0)

    assert 0.10 < shift(0.25) < 0.13, shift(0.25)
    assert 0.08 < shift(0.20) < 0.10, shift(0.20)

    hit, reason = pa.annual_drift_check_triggered(drifted, target, vols(0.25), corr)
    assert hit and reason.startswith('Volatility shift'), (hit, reason)
    hit, reason = pa.annual_drift_check_triggered(drifted, target, vols(0.20), corr)
    assert not hit and reason is None, (hit, reason)

    decision, why = pa.rebalance_decision(drifted, target, 12, vols(0.25), corr)
    assert decision == 'FULL' and why.startswith('Q4 annual drift: Volatility shift'), why
    assert pa.rebalance_decision(drifted, target, 12, vols(0.20), corr)[0] == 'NONE'
    # the drift check is Q4 only
    assert pa.rebalance_decision(drifted, target, 6, vols(0.25), corr)[0] == 'NONE'


def test_portfolio_volatility_is_the_quadratic_form():
    vols = {t: 0.0 for t in pa.TICKERS}
    vols.update(VOO=0.20, AVUV=0.20)
    corr = np.eye(len(pa.TICKERS))
    one = _book({'VOO': 1.0}, total=1.0)
    half = _book({'VOO': 0.5, 'AVUV': 0.5}, total=1.0)
    assert abs(pa.calculate_portfolio_volatility(one, vols, corr) - 0.20) < 1e-12
    # two uncorrelated halves: sqrt(2 * 0.5^2 * 0.2^2) = 0.2 / sqrt(2)
    v = pa.calculate_portfolio_volatility(half, vols, corr)
    assert abs(v - 0.20 / np.sqrt(2)) < 1e-12, v
    i, j = pa.TICKERS.index('VOO'), pa.TICKERS.index('AVUV')
    corr[i, j] = corr[j, i] = 1.0
    v = pa.calculate_portfolio_volatility(half, vols, corr)
    assert abs(v - 0.20) < 1e-12, 'perfectly correlated halves do not diversify'


def test_every_Q1_to_Q3_partial_rebalance_in_the_loop_resolves_its_breach():
    """The monthly loop re-checks after each partial trade, up to
    MAX_REBALANCE_ITERATIONS. A trade that leaves the breach in place is
    retried every iteration and counted every time — the overweight-breach
    no-op logged ten partial rebalances a quarter and moved nothing.

    Zero volatility, Z earning 60% and Y nothing: Z drifts up through its band
    while Y is still inside its own, which is the single-breach PARTIAL case."""
    with _engine_state():
        cfg, _ = _zero_vol_engine(0.0)
        pa.TICKERS = ['Z', 'Y']
        pa.EXPECTED_RETURNS = {'Z': 0.60, 'Y': 0.0}
        pa.BASE_VOLATILITY = {'Z': 0.0, 'Y': 0.0}
        pa.PARAM_UNCERTAINTY_STD = {'Z': 0.0, 'Y': 0.0}
        pa.VOLATILITY_REGIMES = {t: {'low': 0.0, 'medium': 0.0, 'high': 0.0}
                                 for t in ('Z', 'Y')}
        pa.REGIME_TRANSITIONS = {t: {r: {'low': 0.0, 'medium': 1.0, 'high': 0.0}
                                     for r in ('low', 'medium', 'high')}
                                 for t in ('Z', 'Y', pa.MARKET_REGIME_KEY)}
        eye = np.eye(2)
        pa.CORRELATION_BASELINE = pa.CORRELATION_LOW = eye
        pa.CORRELATION_MEDIUM = pa.CORRELATION_HIGH = eye
        pa.REBALANCING_BANDS['Z'] = {'relative': 0.15, 'absolute': 0.05}
        pa.REBALANCING_BANDS['Y'] = {'relative': 0.15, 'absolute': 0.05}
        cfg = dict(cfg, rebalancing_strategy='quarterly_annual', time_horizon_years=3)
        stats = pa.run_single_simulation(({'Z': 0.40, 'Y': 0.60}, cfg, 0))[3]
        assert stats['partial_count'] > 0, 'the fixture must reach a partial rebalance'
        q1q3 = sum(stats['quarterly_counts'][q] for q in ('Q1', 'Q2', 'Q3'))
        assert q1q3 <= 3 * 3, f"{q1q3} Q1-Q3 rebalances in 9 sessions: {stats['trades']}"
        for trade in stats['trades']:
            m = re.match(r'Partial: Sold \S+ (\S+) .*Bought \S+ (\S+) ', trade)
            if m:
                assert m.group(1) != m.group(2), f"sold and bought the same fund: {trade}"


def test_a_single_fund_portfolio_cannot_generate_turnover():
    """The sanity check that proves the measurement is real: VT is ONE fund, so
    its weights cannot drift and it must trade EXACTLY zero. Measured over 400
    paths and 58 years it does — 0.00% — while a 9-fund book runs ~3%/yr."""
    from montecarlo.loaders import MODEL_TICKERS
    vals = {t: (1_000.0 if t == 'VT' else 0.0) for t in MODEL_TICKERS}
    target = {t: (1.0 if t == 'VT' else 0.0) for t in MODEL_TICKERS}
    assert pa.traded_dollars(vals, vals) == 0.0
    # and a full rebalance of an already-on-target book is a no-op
    after, _ = pa.execute_full_rebalance(vals, target)
    assert pa.traded_dollars(vals, after) < 1e-9, pa.traded_dollars(vals, after)


def test_turnover_is_a_rate_against_the_AVERAGE_balance():
    """Raw dollars traded over 58 years would just say 'the book got bigger'."""
    # THE FIXTURE MUST MATCH THE REAL DICT. The first version guessed the shape,
    # raised KeyError, and was swallowed by a bare `except: return` — so it
    # passed against a denominator deliberately broken to the FINAL balance
    # instead of the mean. A test that disables itself on error is worse than no
    # test; it reports safety it never checked.
    stats = [{'count': 1, 'partial_count': 1, 'full_count': 0,
              'quarters_checked': 4, 'q4_sessions': 1, 'iterations_total': 1,
              'triggers': {}, 'trades': [], 'traded': 120.0,
              'value_months': 12_000.0,
              'quarterly_counts': {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 1},
              'q1q3_decision_reasons': {'multiple_breaches': 0,
                                        'near_bands': 0, 'partial': 0}}]
    agg = pa.aggregate_rebalancing_stats(stats, 1)
    # $120 traded against a MEAN balance of $1,000 (12,000 value-months / 12)
    assert abs(agg['turnover_per_year'] - 0.12) < 1e-9, agg['turnover_per_year']
    assert agg['traded_total'] == 120.0, agg['traded_total']
    # dividing by the raw value-months instead would give 0.01 — a twelfth
    assert abs(agg['turnover_per_year'] - 0.01) > 1e-6


def test_turnover_is_measured_and_NOT_charged():
    """Measured at 0.15% round-trip it is half a basis point a year — 0.3% over
    58 years — so the decision gate said stop at measurement. Charging it would
    move the golden master for a rounding error.

    The number that DOES matter is the tax one, and it is now available for it:
    3%/yr of realised gains in a taxable sleeve is roughly 60x the commission.
    """
    # STRIP THE DOCSTRINGS FIRST. The engine's own comment explains that
    # factors.ROUND_TRIP prices a fund's INTERNAL turnover, so grepping the
    # raw source for the name matches the prose that says it is not used --
    # the same way the LTC test matched the word 'random' in a docstring
    # explaining why there is no random draw. Assert on code, not on words.
    src = inspect.getsource(pa)
    code = ''.join(src.split('"""')[::2])
    assert 'ROUND_TRIP' not in code, 'the engine must not charge it yet'
    assert "rebalancing_stats['traded'] +=" in code, 'nothing is counted'
    # ...and the retirement path must actually CARRY it — asserting that the
    # string 'rebalancing_raw' appears in the source still matched a version
    # hard-wired to the empty list. Behaviour, not grep.
    raw = {'spending': np.full((4, 30), 2e5), 'depleted': np.zeros(4, bool),
           'finals': np.full(4, 4e7), 'depletion_ages': [None] * 4,
           'rebalancing_raw': [
               {'count': 1, 'partial_count': 1, 'full_count': 0,
                'quarters_checked': 4, 'q4_sessions': 1, 'iterations_total': 1,
                'triggers': {}, 'trades': [], 'traded': 120.0,
                'value_months': 12_000.0,
                'quarterly_counts': {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 1},
                'q1q3_decision_reasons': {'multiple_breaches': 0,
                                          'near_bands': 0, 'partial': 0}}]}
    cfg = {'age': 37, 'time_horizon_years': 58, 'withdrawal': {'start_age': 65}}
    scored = solver.score(raw, cfg, 2e5, 0.85, 'per_band')
    assert 'rebalancing' in scored, 'score() dropped the counters'
    assert scored['rebalancing']['turnover_per_year'] > 0, scored['rebalancing']

    # ...and SIMULATE must put them there in the first place. Asserting only on
    # score() passed against a `simulate` hard-wired to an empty list, because
    # this fixture hands score() a raw dict it built itself. Stub the worker so
    # the real collection line runs.
    rb = raw['rebalancing_raw'][0]
    fake_runs = [(1e6, None, None, rb, {'spending': np.full(30, 2e5),
                                        'depletion_age': None,
                                        'w_initial': 2e5,
                                        'v_at_retirement': 7e6})] * 3
    real_paths = solver._paths
    try:
        solver._paths = lambda *a, **k: fake_runs
        out = solver.simulate({}, cfg, 2e5, 3, False, full=True)
    finally:
        solver._paths = real_paths
    assert out['rebalancing_raw'], 'simulate() dropped the counters'
    assert len(out['rebalancing_raw']) == 3, out['rebalancing_raw']
    rr = inspect.getsource(report_retirement.run)
    j = rr.index('_rebalancing_summary(')
    assert "if spec['mode'] == 'accumulation'" not in rr[max(0, j - 200):j], \
        'the summary is gated to accumulation again'


def test_rebalancing_summary_reports_turnover_PER_strategy():
    """The header claimed "(representative — the machinery barely varies by
    allocation)" and measuring turnover falsified it: VT trades 0.00%/yr
    because a single fund cannot drift, while a nine-fund book runs ~3.4%.
    That is the difference between zero and something, and it was hidden behind
    a claim about the machinery."""

    def rb(t, ev):
        return {'turnover_per_year': t, 'avg_events_per_year': ev,
                'total_partial': 1, 'partial_pct': 50.0, 'total_full': 1,
                'full_pct': 50.0, 'total_events': 2,
                'median_events_per_year': ev, 'traded_total': 1.0,
                'quarterly_totals': {'Q1': 1, 'Q2': 0, 'Q3': 0, 'Q4': 1},
                'q1q3_decision_reasons': {'multiple_breaches': 0,
                                          'near_bands': 0, 'partial': 1},
                'triggers': [], 'time_horizon_years': 58}
    res = {'140': {'rebalancing': rb(0.0266, 1.5)},
           '102': {'rebalancing': rb(0.0, 0.0)}}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_retirement._rebalancing_summary(res, ['140', '102'])
    out = buf.getvalue()
    assert '2.66%/yr' in out and '0.00%/yr' in out, out
    assert 'barely varies' not in out, 'the falsified claim is back'
    assert 'does NOT charge' in out, 'must say the cost is unpriced'
    # a run with no counters must print nothing rather than crash
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        report_retirement._rebalancing_summary({'x': {}}, ['x'])
    assert buf2.getvalue() == '', buf2.getvalue()


def test_the_derisk_baseline_is_the_SPENDING_leader_not_the_level_leader():
    """Every number in that section is realised spending; the baseline was
    picked on the plan level. Observed 2026-09-14: the level made 160 the
    baseline while 160 was the ONE portfolio that did not resolve against 140,
    and 152 — which did, and topped the life-stage table — sat unexamined."""
    P = _P()
    rng = np.random.default_rng(1)

    def row(sp, med):
        return {'spend': sp, 'median_realised': med, 'p10_realised': med * 0.42,
                'per_path_mean': med + rng.normal(0, 45e3, 600),
                'annual_vol': 14.5, 'median_bequest': 47e6, 'worst_cut': 0.0,
                'twr_cagr': np.full(5, 0.0844), 'sortino_median': 1.03,
                'clipped': 0, 'level_se': 12e3, 'depletion_rate': 0.0,
                'success_rate': 0.90, 'sortino': np.full(600, 1.03)}
    # 161 leads the LEVEL (186k); 152 leads realised SPENDING (487k)
    specs = ['140', '140>140+5@65', '152', '152>152+5@65', '161', '161>161+5@65']
    res = {'140': row(176e3, 462e3), '140>140+5@65': row(167e3, 425e3),
           '152': row(182e3, 487e3), '152>152+5@65': row(184e3, 459e3),
           '161': row(186e3, 474e3), '161>161+5@65': row(171e3, 447e3)}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_retirement._matrix(specs, res, {
            'spend': 'solve', 'mode': 'retirement',
            'portfolios': ['140', '152', '161'], 'derisk_at': 65,
            'bond_tiers': [0.05]}, P)
    out = buf.getvalue()
    head = out[out.index('WHETHER TO DERISK'):]
    assert 'the best portfolio, 152' in head, head[:120]
    assert 'the best portfolio, 161' not in head, 'still ranking on the level'


def test_a_portfolio_id_is_accepted_at_the_shortlist_prompt():
    """The shortlist reads "1  141 / 2  140 / 3  145", so typing 140 — the
    thing you actually want — is the obvious move, and it was rejected with
    "enter numbers from the list". Observed on a real run's FIRST keystroke.
    Drives the real prompt with scripted keystrokes; a MIX must still be
    refused, because `1` would be ambiguous between a menu slot and an id."""
    P = _P()

    def answer(keys):
        it = iter(keys)
        real = runner.ask
        runner.ask = lambda _prompt, _default='': next(it)
        try:
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                got = runner.ask_portfolios(P)
        finally:
            runner.ask = real
        return got, buf.getvalue()

    ids, _ = answer(['140'])
    assert ids == ['140'], ids
    ids, _ = answer(['140,145'])
    assert ids == ['140', '145'], ids
    # menu slots still work, and still mean the SLOT
    slots = runner.suggested_bases(P)
    ids, _ = answer(['2'])
    assert ids == [slots[1][0]], ids
    # a mix is ambiguous -> re-ask rather than guess
    ids, out = answer(['1,145', '140'])
    assert ids == ['140'], ids
    assert 'mixed' in out or '1,145' in out or '1 ' in out.split('?')[-1], out
    # a genuine typo is still refused
    ids, out = answer(['999', '140'])
    assert ids == ['140'], ids
    assert '999' in out, out


def test_the_run_record_says_how_many_paths_it_bought():
    """All twelve real retirement records on disk carried `sims: None` — the
    mode never sets it — while the run actually spent 400 search + 1,500
    verification paths per strategy. So `--diff` could not tell a re-draw from
    a resolution change, on a metric this repo has ALREADY seen flip: a
    600-path `regions` sweep reported a reversal that vanished at 1,600."""
    solve = persistence.provenance(None, {'mode': 'retirement', 'spend': 'solve'})
    assert solve['search_sims'] == solver.SEARCH_SIMS, solve
    assert solve['verify_sims'] == solver.VERIFY_SIMS, solve
    assert 'sims' not in solve, 'a solve run does not have one path count'

    fixed = persistence.provenance(None, {'mode': 'retirement', 'spend': 200_000.0})
    assert fixed['sims'] == solver.FIXED_SIMS, fixed
    assert 'verify_sims' not in fixed, fixed

    # an explicit count must win over the default
    custom = persistence.provenance(None, {'spend': 'solve', 'verify_sims': 3200})
    assert custom['verify_sims'] == 3200, custom

    # ...and the DIFF must flag it as a resolution change, not a real move
    d = Path(tempfile.mkdtemp())

    def rec(name, verify, spend):
        r = {'timestamp': name,
             'spec': {'mode': 'retirement', 'spend': 'solve', 'policy': 'gk'},
             'provenance': {'scoring_version': persistence.SCORING_VERSION,
                            'search_sims': 400, 'verify_sims': verify},
             'config': {}, 'strategies': ['140'],
             'results': {'gk': {'140': {'spend': spend}}}}
        p = d / f'run_{name}.json'
        p.write_text(json.dumps(r))
        return p

    with contextlib.redirect_stdout(io.StringIO()) as buf:
        persistence.print_diff(rec('A', 400, 178e3), rec('B', 1600, 191e3),
                               key='spend')
    out = buf.getvalue()
    assert 'PATH COUNT changed' in out, out
    assert 'provenance identical' not in out, out

    # two runs at the SAME count must NOT get the warning
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        persistence.print_diff(rec('C', 1600, 178e3), rec('D', 1600, 191e3),
                               key='spend')
    assert 'PATH COUNT changed' not in buf.getvalue(), buf.getvalue()


def test_the_path_counts_are_defined_exactly_once():
    """They were literals in three files and two disagreed: solver's signature
    said 500/2000 while every real caller passed 400/1500, so the dead defaults
    were the ones a reader found first."""
    sig = inspect.signature(solver.solve_sustainable_spend)
    assert sig.parameters['search_sims'].default is None, 'a live literal is back'
    assert sig.parameters['verify_sims'].default is None, 'a live literal is back'
    for mod in (runner, report_retirement):
        src = re.sub(r'#[^\n]*', '', inspect.getsource(mod))
        for lit in ("'search_sims', 400", "'verify_sims', 1500", "'sims', 2000"):
            assert lit not in src, f'{mod.__name__} still hard-codes {lit}'
    # and the resolved defaults must be what the caller actually spends
    assert (solver.SEARCH_SIMS, solver.VERIFY_SIMS) == (400, 1500)


def test_a_run_can_be_driven_from_a_spec_without_prompts():
    """The interactive prompts were the ONLY way in, so every measurement
    quoted in CLAUDE.md was unreproducible without a human re-answering nine
    questions identically. `--spec` is the same spec `--again` replays, so a
    run from it records and diffs like any other."""
    P = _P()
    spec = runner.parse_spec('{"portfolios": ["140", "152"]}', P)
    assert spec['portfolios'] == ['140', '152'], spec
    assert spec['mode'] == 'retirement' and spec['spend'] == 'solve', spec
    # it must reach the real strategy resolver, not a parallel implementation
    specs, _ = runner.resolve_strategies(spec, P)
    assert specs == ['140', '152'], specs

    # a comma string is the shape a human types
    assert runner.parse_spec('{"portfolios": "140,152"}', P)['portfolios'] == \
        ['140', '152']

    # --sims is ONE knob, split the way the solver spends it
    solve = runner.parse_spec('{"portfolios": ["140"]}', P, sims=400)
    assert (solve['verify_sims'], solve['search_sims']) == (400, 200), solve
    assert 'sims' not in solve, 'a solve does not have a single count'
    fixed = runner.parse_spec('{"portfolios": ["140"], "spend": 200000}', P,
                              sims=400)
    assert fixed['sims'] == 400 and 'verify_sims' not in fixed, fixed
    acc = runner.parse_spec('{"mode": "accumulation", "portfolios": ["140"]}',
                            P, sims=300)
    assert acc['sims'] == 300, acc

    # and it must REFUSE rather than resolve to something smaller than asked
    for bad, why in [('{"portfolios": ["999"]}', '999'),
                     ('{"portfolios": []}', 'at least one'),
                     ('{"portfolios": ["140"], "policy": "magic"}', 'magic'),
                     ('{"portfolios": ["140"], "mode": "sideways"}', 'sideways'),
                     ('[1,2,3]', 'object'),
                     ('{oops}', 'JSON')]:
        try:
            runner.parse_spec(bad, P)
        except ValueError as exc:
            assert why in str(exc), f'{bad} -> {exc}'
        else:
            raise AssertionError(f'{bad} was accepted')

    # a file path is accepted too
    f = Path(tempfile.mkdtemp()) / 'spec.json'
    f.write_text('{"portfolios": ["140"], "policy": "vpw"}')
    assert runner.parse_spec(str(f), P)['policy'] == 'vpw'


def test_a_headless_run_does_not_clobber_the_interactive_preset():
    """A test run must not cost you the configuration you spent the prompts
    building — `--again` has to keep meaning the last run YOU set up."""
    src = inspect.getsource(runner.main)
    body = re.sub(r'#[^\n]*', '', src)
    i = body.index("save_preset(spec, 'last')")
    guard = body[max(0, i - 200):i]
    assert 'if not headless' in guard, 'the preset save is not gated on headless'


def _main_smoke(argv, paths=4):
    """Drive run.main() from ARGV, tiny. Returns (exit code, output, saved record).

    The two end-to-end smoke tests start at build_config and so never touch
    main() — argument parsing, the spec source, the confirm bypass and the
    preset guard are all downstream of where they begin. This is about to be
    the entry point for a 15-portfolio batch, so the wiring from argv to a
    persisted record needs to be the thing under test, not the thing assumed.
    """

    real_paths, real_save, real_figs = solver._paths, persistence.save_run, report_retirement.figures
    real_acc = report_retirement.accumulation_figures
    real_confirm = runner.confirm
    saved = {}

    def _no_prompt(*a, **k):
        # NOT a stdin trick. `ask` returns its DEFAULT on EOF, so with stdin at
        # /dev/null a reinstated confirm() silently auto-accepts and the test
        # sees nothing — while a real terminal would block. Assert the prompt is
        # never REACHED.
        raise AssertionError('a headless run reached the confirm prompt')

    def _tiny(plan, config, n_sims, parallel, light=True, seed_offset=0):
        return real_paths(plan, config, min(int(n_sims), paths), False,
                          light=light, seed_offset=seed_offset)

    def _capture(spec, by_policy, specs, config, outdir=None, **_):
        saved['spec'] = spec
        saved['provenance'] = persistence.provenance(config, spec)
        return None

    buf = io.StringIO()
    try:
        solver._paths = _tiny
        persistence.save_run = _capture
        runner.confirm = _no_prompt
        report_retirement.figures = lambda *a, **k: []
        report_retirement.accumulation_figures = lambda *a, **k: []
        # STDIN IS CLOSED DELIBERATELY. A headless run must not reach a prompt,
        # and if it ever does the failure has to be an EOFError here rather
        # than a block: deleting the confirm() bypass made this gate HANG for
        # ten minutes instead of failing, which reads as a slow test.
        real_stdin, sys.stdin = sys.stdin, open(os.devnull)
        try:
            with contextlib.redirect_stdout(buf):
                code = runner.main(argv)
        finally:
            sys.stdin.close()
            sys.stdin = real_stdin
    finally:
        solver._paths, persistence.save_run, report_retirement.figures = real_paths, real_save, real_figs
        runner.confirm = real_confirm
        report_retirement.accumulation_figures = real_acc
        solver.shutdown_pool()
    return code, buf.getvalue(), saved


def test_a_return_view_child_that_fails_or_runs_the_wrong_model_is_refused():
    """run_view's two guards — the child exited badly, or it ran a different
    return model than asked — had only ever met a monkeypatched run_view. A
    view that silently fell back to the default would print a column equal to
    the factor one and read as 'the decision survives'. Real children, no
    plans, so each costs one interpreter."""
    from montecarlo import return_views as rv
    cfg = runner.default_config()
    got = rv.run_view('historical', cfg, {}, 100e3, 4)
    assert got['engine']['MU_METHOD'] == 'historical', got['engine']

    real = rv.view_env
    try:
        rv.view_env = lambda key, base=None: real('factor', base)
        try:
            rv.run_view('historical', cfg, {}, 100e3, 4)
            raise AssertionError('a child on the wrong model must be refused')
        except RuntimeError as e:
            assert "asked for 'historical'" in str(e), e
        rv.view_env = lambda key, base=None: dict(real(key, base),
                                                  MC_RF_SOURCE='bogus')
        try:
            rv.run_view('historical', cfg, {}, 100e3, 4)
            raise AssertionError('a failed child must be refused')
        except RuntimeError as e:
            assert 'child exited' in str(e), e
    finally:
        rv.view_env = real


def test_main_replays_lists_and_diffs_saved_runs():
    """--again, --history and --diff had never been driven through main().
    Both stores point at a temp directory: --again with nothing saved refuses,
    with a preset it runs it without asking and leaves the preset as it was;
    --history and --diff read the run records, oldest on the left."""
    d = Path(tempfile.mkdtemp())
    real = runner.PRESETS, persistence.RUNS_DIR
    try:
        runner.PRESETS, persistence.RUNS_DIR = d / 'runs.json', d / 'runs'
        code, out, _ = _main_smoke(['--again'])
        assert code == 1 and 'No previous run' in out, out[-300:]

        spec = runner.parse_spec(json.dumps({'portfolios': ['141'],
                                             'policy': 'guyton_klinger'}),
                                 _P(), sims=4)
        runner.save_preset(spec, 'last')
        before = runner.PRESETS.read_text()
        code, out, saved = _main_smoke(['--again'])
        assert code == 0 and 'Traceback' not in out, out[-800:]
        assert saved['spec'] == spec, saved.get('spec')
        assert json.loads(runner.PRESETS.read_text()) == json.loads(before)

        with contextlib.redirect_stdout(io.StringIO()) as buf:
            assert runner.main(['--history']) == 0
            assert runner.main(['--diff']) == 1
        assert 'No saved runs yet' in buf.getvalue(), buf.getvalue()
        assert 'Need two saved runs' in buf.getvalue(), buf.getvalue()

        persistence.RUNS_DIR.mkdir()
        for name, spend in (('2026-10-01', 178e3), ('2026-10-02', 191e3)):
            (persistence.RUNS_DIR / f'run_{name}.json').write_text(json.dumps({
                'timestamp': name, 'spec': {'mode': 'retirement', 'spend': 'solve'},
                'provenance': {'scoring_version': persistence.SCORING_VERSION},
                'config': {}, 'strategies': ['141'],
                'results': {'gk': {'141': {'spend': spend}}}}))
            if name == '2026-10-01':
                with contextlib.redirect_stdout(io.StringIO()):
                    assert runner.main(['--diff']) == 1, 'one run is not two'
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            assert runner.main(['--history']) == 0
            assert runner.main(['--diff']) == 0
        out = buf.getvalue()
        assert '2026-10-01' in out and '2026-10-02' in out, out
        assert '2026-10-01  →  2026-10-02' in out, out
    finally:
        runner.PRESETS, persistence.RUNS_DIR = real


def test_main_runs_a_spec_end_to_end_from_argv():
    """argv -> parse_spec -> resolve -> report -> a persisted record."""
    code, out, saved = _main_smoke(
        ['--spec', '{"portfolios": ["141", "145"], "policy": "guyton_klinger"}',
         '--sims', '4'])
    assert code == 0, (code, out[-800:])
    assert 'Traceback' not in out, out[-800:]
    for section in ('RETIREMENT ANALYSIS', 'RESULTS', 'SPENDING BY LIFE STAGE'):
        assert section in out, (section, out[:600])
    # every requested portfolio must reach a ROW, not merely the prose
    body = out.split('RESULTS', 1)[1].split('HEAD-TO-HEAD', 1)[0]
    for g in ('141', '145'):
        assert any(l.strip().startswith(g) for l in body.splitlines()), (g, body)
    # ...and the record must know what it cost
    assert saved.get('spec', {}).get('portfolios') == ['141', '145'], saved
    assert saved['provenance']['verify_sims'] == 4, saved['provenance']
    assert saved['provenance']['search_sims'] == 2, saved['provenance']


def test_main_accumulation_runs_from_argv_too():
    """The other mode takes a different path through report_retirement."""
    code, out, saved = _main_smoke(
        ['--spec', '{"mode": "accumulation", "portfolios": ["141", "102"]}',
         '--sims', '4'])
    assert code == 0, (code, out[-800:])
    assert 'Traceback' not in out, out[-800:]
    assert saved['provenance']['sims'] == 4, saved['provenance']
    assert 'verify_sims' not in saved['provenance'], saved['provenance']


def test_dry_run_spends_nothing():
    """--dry-run must resolve and cost NOTHING: the point is to check a spec
    before committing minutes to it."""
    spent = []
    real = solver._paths

    def _count(*a, **k):
        spent.append(1)
        return real(*a, **k)

    buf = io.StringIO()
    try:
        solver._paths = _count
        with contextlib.redirect_stdout(buf):
            code = runner.main(['--spec',
                                '{"portfolios": ["141"], "derisk_at": 65}',
                                '--dry-run'])
    finally:
        solver._paths = real
    out = buf.getvalue()
    assert code == 0, (code, out)
    assert not spent, f'--dry-run simulated {len(spent)} batches'
    assert '141' in out and 'min' in out, out
    # it must show what it RESOLVED to, which is the whole point
    assert any('141>' in l for l in out.splitlines()), out


def test_a_bad_spec_exits_nonzero_without_simulating():
    """A batch driver needs a usable exit code, and a typo must not cost paths."""
    spent = []
    real = solver._paths
    buf = io.StringIO()
    try:
        solver._paths = lambda *a, **k: (spent.append(1), real(*a, **k))[1]
        with contextlib.redirect_stdout(buf):
            code = runner.main(['--spec', '{"portfolios": ["999"]}'])
    finally:
        solver._paths = real
    assert code == 1, (code, buf.getvalue())
    assert not spent, 'a rejected spec still simulated'
    assert '999' in buf.getvalue(), buf.getvalue()


def test_the_two_spending_tables_are_one_number_not_two():
    """IN DETAIL printed 'you spend / lifetime avg' and the band table was
    titled 'WHAT YOU ACTUALLY SPEND', so one quantity wore three framings and
    read as three quantities — the note between them even said the second
    'shows higher numbers again'. The band table now repeats the lifetime
    average as its anchor column, so the arithmetic is on screen."""
    cfg = {'age': 37, 'time_horizon_years': 58, 'retirement_age': 65}
    rng = np.random.default_rng(11)
    sp = np.tile(np.concatenate([np.full(10, 240e3), np.full(10, 440e3),
                                 np.full(10, 800e3)]), (300, 1))
    sp = sp * rng.lognormal(0, 0.05, sp.shape)
    life = float(np.median(sp.mean(axis=1)))
    rows = [('', '152', {'spending': sp, 'median_bequest': 51.6e6,
                         'median_realised': life, 'start_age': 65})]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_retirement._spending_shape(rows, cfg)
    out = buf.getvalue()
    assert 'lifetime' in out, out
    # the anchor cell must be the SAME figure _LEVEL_COLUMNS renders
    detail = report_retirement._LEVEL_COLUMNS['median'][3]({'median_realised': life})
    row = [l for l in out.splitlines() if l.strip().startswith('152')][0]
    assert detail in row.split('\u2502')[0], (detail, row)
    # ...and it must reconcile with the decades it is split into
    bands = [float(c.strip().rstrip('k')) for c in row.split('\u2502')[1:4]
             for c in [c.split()[0]]]
    assert abs(sum(bands) / 3 - life / 1e3) / (life / 1e3) < 0.05, (bands, life)


def test_the_life_stage_table_ranks_on_the_column_it_prints():
    """The sort was the length-weighted mean of band MEDIANS while the printed
    anchor is the median of per-path MEANS. They agree to ~1% — enough to order
    two rows against the number printed beside them, which is the exact
    confusion this rework exists to remove."""
    cfg = {'age': 37, 'time_horizon_years': 58, 'retirement_age': 65}
    flat = np.tile(np.full(30, 300e3), (200, 1))

    def row(tag, life):
        return ('', tag, {'spending': flat, 'median_bequest': 1e7,
                          'median_realised': life, 'start_age': 65})

    # identical band stats; only the lifetime column separates them
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_retirement._spending_shape([row('LOW', 400e3), row('HIGH', 500e3)],
                                          cfg)
    body = [l for l in buf.getvalue().splitlines()
            if l.strip().startswith(('LOW', 'HIGH'))]
    assert body[0].strip().startswith('HIGH'), body

    # a row predating median_realised must fall back, not crash or sort as 0
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_retirement._spending_shape(
            [('', 'OLD', {'spending': flat, 'median_bequest': 1e7,
                          'start_age': 65}), row('NEW', 500e3)], cfg)
    out = buf.getvalue()
    assert 'OLD' in out and 'NEW' in out, out


def test_most_spending_quotes_spending_not_the_certified_bar():
    """`✓ most spending 161 $201k/yr` printed directly under a table whose
    spending column read 525k: it was keyed on 'spend', the solver's LEVEL.
    Three lines below, the same block says every column above is what the paths
    PRODUCED and that the bar lives in YOUR PLAN IN CONTEXT — so the line
    disclaimed the bar and printed it. `safest` beside it was already on
    realised spending, which is what hid the mismatch."""
    rng = np.random.default_rng(2)

    def row(level, realised, p10):
        return {'spend': level, 'median_realised': realised,
                'p10_realised': p10, 'annual_vol': 14.5,
                'per_path_mean': realised + rng.normal(0, 50e3, 400),
                'median_bequest': 47e6, 'worst_cut': 0.0, 'clipped': 0,
                'twr_cagr': np.full(5, 0.084), 'sortino_median': 1.03,
                'sortino': np.full(400, 1.03), 'depletion_rate': 0.001,
                'success_rate': 0.90, 'level_se': 9_000.0}

    def line(res):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report_retirement._flat_table(list(res), res, {
                'spend': 'solve', 'mode': 'retirement'}, list(res)[0])
        return [l for l in buf.getvalue().splitlines() if 'most spending' in l][0]

    # the observed case: the level ranks 161 top and so does spending
    got = line({'140': row(177_154, 465_305, 197_365),
                '161': row(201_314, 525_177, 214_956)})
    assert '525k' in got, got
    assert '201k' not in got, f'still quoting the certified bar: {got}'

    # ...and where the two DISAGREE, spending must win the naming
    got = line({'A': row(201_000, 400_000, 190_000),   # higher bar
                'B': row(177_000, 500_000, 195_000)})  # more spending
    assert 'most spending   B' in got, got
    assert '500k' in got, got

    # a run without realised spending falls back and SAYS it is the bar
    bare = {g: {k: v for k, v in r.items() if k != 'median_realised'}
            for g, r in {'A': row(201_000, 400_000, 190_000),
                         'B': row(177_000, 500_000, 195_000)}.items()}
    got = line(bare)
    assert 'plan level' in got, got


def _diag_cfg():
    return {'age': 37, 'time_horizon_years': 58, 'retirement_age': 65,
            'adequacy': 'per_band',
            'withdrawal': {'start_age': 65, 'policy': 'guyton_klinger',
                           'floor': 0.85, 'target_real_annual': 200_000.0}}


def test_binding_band_names_the_stage_that_set_the_level():
    """Per-band adequacy certifies the WORST band, so exactly one decides each
    plan level — and the report never said which. Band 1 binding is sequence
    risk in your sixties; band 3 is longevity. Same number, opposite remedy."""
    cfg = _diag_cfg()
    n = 200
    # every path starves in the LAST decade and is rich in the first two
    late = np.concatenate([np.full(20, 300e3), np.full(10, 100e3)])
    raw = {'spending': np.tile(late, (n, 1)),
           'depleted': np.zeros(n, dtype=bool),
           'depletion_ages': [None] * n}
    b = diagnostics.binding_band(raw, cfg, 0.85, 200e3)
    assert b['labels'] == ['65-74', '75-84', '85-94'], b['labels']
    assert b['all'][2] == 1.0, b['all']     # the last band binds, every path
    assert b['all'][0] == 0.0, b['all']

    # flip it: starve EARLY and band 1 must bind instead
    early = np.concatenate([np.full(10, 100e3), np.full(20, 300e3)])
    raw['spending'] = np.tile(early, (n, 1))
    b = diagnostics.binding_band(raw, cfg, 0.85, 200e3)
    assert b['all'][0] == 1.0, b['all']


def test_failure_timing_separates_sequence_risk_from_a_high_level():
    """"Mostly UNDERSPENDING, not ruin" was the whole story. Concentrated early
    is sequence risk and argues for a glide path; spread evenly is a level set
    too high. The report could not tell them apart."""
    cfg = _diag_cfg()
    n = 100
    bar = 0.85 * 200e3
    early = np.tile(np.concatenate([np.full(10, bar * 0.5),
                                    np.full(20, bar * 2)]), (n, 1))
    raw = {'spending': early, 'depleted': np.zeros(n, dtype=bool),
           'depletion_ages': [None] * n}
    t = diagnostics.failure_timing(raw, cfg, 0.85, 200e3)
    assert t['first_dip_age'][50] == 65, t          # the very first year
    thirds = t['dip_share_by_third']
    assert thirds[0] == 1.0, thirds                 # ALL of it in decade one

    late = np.tile(np.concatenate([np.full(20, bar * 2),
                                   np.full(10, bar * 0.5)]), (n, 1))
    raw['spending'] = late
    t = diagnostics.failure_timing(raw, cfg, 0.85, 200e3)
    assert t['first_dip_age'][50] == 85, t
    assert t['dip_share_by_third'][2] == 1.0, t['dip_share_by_third']

    # depletion ages are reported when paths actually run out
    raw['depletion_ages'] = [80] * 40 + [None] * 60
    t = diagnostics.failure_timing(raw, cfg, 0.85, 200e3)
    assert t['n_depleted'] == 40 and t['depletion_age'][50] == 80, t


def test_regime_occupancy_reproduces_the_chain_and_catches_a_broken_one():
    """The regime machinery is the model's headline claim, reworked five times,
    and nothing checked that a RUN visits the states at the rate its own chain
    implies. A degenerate chain should be visible here, not as an unexplained
    level shift."""
    st = diagnostics.market_stationary()
    assert st is not None and abs(sum(st.values()) - 1.0) < 1e-9, st
    # the shipped chain's own long-run distribution, computed not quoted
    assert 0.25 < st['low'] < 0.36 and 0.34 < st['medium'] < 0.46, st

    # a run matching the chain reports a small gap...
    months = [{k: int(1000 * v) for k, v in st.items()} for _ in range(20)]
    occ = diagnostics.regime_occupancy({'regime_months': months})
    assert max(abs(occ['realised'][k] - st[k]) for k in st) < 0.01, occ

    # ...and one stuck in a single state reports a large one
    stuck = [{'low': 0, 'medium': 1000, 'high': 0} for _ in range(20)]
    occ = diagnostics.regime_occupancy({'regime_months': stuck})
    assert occ['realised']['medium'] == 1.0, occ
    assert max(abs(occ['realised'][k] - st[k]) for k in st) > 0.25, occ


def test_uncertainty_split_separates_the_draw_from_the_market():
    """The three-stage architecture exists to keep estimation error apart from
    market noise and never reported the split. A share near zero would mean the
    parameter bands are decorative; near one, that the model mostly restates its
    own priors."""
    rng = np.random.default_rng(5)
    n = 500
    mu = rng.normal(0.08, 0.02, n)

    # outcome driven ENTIRELY by the drawn mu -> share near 1
    raw = {'path_mu': mu, 'finals': np.exp(mu * 58) * 1e6,
           'rf_shock': rng.normal(0, 0.01, n),
           'infl_shock': rng.normal(0, 0.015, n)}
    u = diagnostics.uncertainty_split(raw)
    assert u['epistemic_share'] > 0.98, u

    # outcome independent of it -> share near 0
    raw['finals'] = np.exp(rng.normal(0, 1.0, n)) * 1e6
    u = diagnostics.uncertainty_split(raw)
    assert u['epistemic_share'] < 0.10, u

    # the shared draws are reported at the width they were configured with
    assert abs(u['rf_spread'] - 0.01) < 0.002, u
    assert abs(u['infl_spread'] - 0.015) < 0.003, u


def test_diagnostics_are_silent_without_debug_and_never_break_a_run():
    """--debug is already the switch for "tell me about the machinery"; a
    second flag would be a second thing to remember. And a diagnostic must
    never cost a real result."""
    cfg = _diag_cfg()
    res = {'140': {'spend': 200e3, 'raw': {'spending': np.full((8, 30), 300e3),
                                           'depleted': np.zeros(8, dtype=bool),
                                           'depletion_ages': [None] * 8}}}
    was = getattr(instrument, 'ENABLED', False)
    try:
        instrument.enable(False)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report_retirement._diagnostics(res, ['140'], {}, cfg,
                                           {'mode': 'retirement'})
        assert buf.getvalue() == '', buf.getvalue()

        instrument.enable(True)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report_retirement._diagnostics(res, ['140'], {}, cfg,
                                           {'mode': 'retirement'})
        assert 'WHICH LIFE STAGE' in buf.getvalue(), buf.getvalue()

        # a malformed raw must be caught, not raised into the run
        bad = {'140': {'spend': 200e3, 'raw': {'spending': 'not an array'}}}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report_retirement._diagnostics(bad, ['140'], {}, cfg,
                                           {'mode': 'retirement'})
        assert 'unavailable' in buf.getvalue(), buf.getvalue()
    finally:
        instrument.enable(was)


def test_the_engine_carries_the_diagnostics_without_moving_the_draws():
    """Regime counts and the drawn mu are REPORTED, not generated: no np.random
    call was added, so the stream position and the golden master are untouched.
    That is the one property that makes this safe to put in the monthly loop."""
    src = inspect.getsource(pa.run_single_simulation)
    body = re.sub(r'#[^\n]*', '', src)
    # the counter must sit in the loop but add no draw beside it
    assert '_regime_months[market_regime] += 1' in body, 'counter not in the loop'
    assert "'path_mu'" in body and "'rf_shock'" in body, body[-400:]
    # and the values must be the ones already drawn, not fresh ones
    assert body.count('np.random.normal(0, RF_REAL_STD)') == 1, 'rf drawn twice'
    assert body.count('np.random.normal(0, INFLATION_STD)') == 1, 'infl drawn twice'


def test_in_detail_ranks_on_the_column_it_prints():
    """`(all 9, ranked by spend)` named the PLAN LEVEL while the first column is
    realised spending — observed 2026-09-14 on nine rows whose printed spending
    ran 464/496/438/457/465/434/455/429/409, monotone in nothing on screen.

    Worse than a label: the list is cut at MAX_DETAIL_ROWS, so a wrong key can
    drop the best-spending strategy off the table."""
    P = _P()
    rng = np.random.default_rng(1)

    def row(level, realised):
        return {'spend': level, 'median_realised': realised,
                'p10_realised': realised * 0.42, 'annual_vol': 14.5,
                'per_path_mean': realised + rng.normal(0, 50e3, 300),
                'median_bequest': 47e6, 'worst_cut': 0.0, 'clipped': 0,
                'twr_cagr': np.full(5, 0.084), 'sortino_median': 1.03,
                'sortino': np.full(300, 1.03), 'depletion_rate': 0.0,
                'success_rate': 0.90, 'level_se': 9_000.0}

    # the level and the spending ORDER DIFFERENTLY: A leads on level, B spends more
    specs = ['140', '140>140+5@65', '152', '152>152+5@65']
    res = {'140': row(177_154, 465e3), '140>140+5@65': row(169_392, 429e3),
           '152': row(186_269, 496e3), '152>152+5@65': row(186_573, 464e3)}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_retirement._matrix(specs, res, {
            'spend': 'solve', 'mode': 'retirement',
            'portfolios': ['140', '152'], 'derisk_at': 65,
            'bond_tiers': [0.05]}, P)
    out = buf.getvalue()
    # Bound the section: HEAD-TO-HEAD now prints AFTER IN DETAIL, and its rows
    # are the same strategy names, so an unbounded scan counts them twice and
    # reads as unsorted.
    detail = out[out.index('IN DETAIL'):]
    detail = detail.split('HEAD-TO-HEAD')[0]
    assert 'ranked by lifetime average spending' in detail, detail[:300]
    assert 'ranked by spend)' not in detail, detail[:300]

    # the printed column must be non-increasing down the table
    # Match on the FIRST TOKEN, not a prefix: '152>152+5@65' starts with '152',
    # so a startswith scan counts one row twice and reads as unsorted.
    seen = []
    for line in detail.splitlines():
        tok = line.strip().split(' ')[0]
        if tok in res and 'k' in line:
            seen.append(res[tok]['median_realised'])
    assert len(seen) == len(specs), (seen, detail[:400])
    assert seen == sorted(seen, reverse=True), seen
    assert seen[0] == 496e3, 'the biggest spender is not first'


def test_epistemic_share_excludes_depleted_paths_rather_than_clamping_them():
    """log(max(y, 1e-9)) turned a depleted path into a -20.7 outlier and
    crushed the correlation: measured on real runs, 140 read 23% with ten
    depleted paths out of 1500 and 47% once they were dropped, while 102 — the
    only portfolio with ZERO depletions — did not move at all. A statistic that
    swings 2x on 0.7% of paths is reporting the clamp, not the model."""
    rng = np.random.default_rng(0)
    n = 1500
    mu = rng.normal(0.08, 0.02, n)
    finals = np.exp(mu * 30) * 1e6 * np.exp(rng.normal(0, 0.8, n))
    raw = {'path_mu': mu, 'finals': finals.copy(),
           'rf_shock': rng.normal(0, 0.01, n),
           'infl_shock': rng.normal(0, 0.015, n)}
    clean = diagnostics.uncertainty_split(raw)['epistemic_wealth']

    # deplete 1% of paths; the estimate must barely move
    raw['finals'][:15] = 0.0
    with_zeros = diagnostics.uncertainty_split(raw)['epistemic_wealth']
    assert abs(with_zeros - clean) < 0.05, (clean, with_zeros)

    # and a run where EVERY path depleted reports nothing rather than nonsense
    raw['finals'] = np.zeros(n)
    got = diagnostics.uncertainty_split(raw)
    assert got is None or got['epistemic_wealth'] is None, got


def test_epistemic_share_is_reported_for_spending_and_bequest_separately():
    """The report ranks on spending and this defaulted to `finals`, the
    BEQUEST. They give different answers: measured at 1500 paths, VT reads 29%
    on spending and 56% on bequest, because the guardrails absorb the parameter
    draw into the estate instead of into the lifestyle."""
    rng = np.random.default_rng(4)
    n = 400
    mu = rng.normal(0.08, 0.02, n)
    # bequest tracks mu; spending is deliberately decoupled from it
    raw = {'path_mu': mu, 'finals': np.exp(mu * 30) * 1e6,
           'spending': np.tile(rng.normal(3e5, 5e4, n)[:, None], (1, 30)),
           'rf_shock': rng.normal(0, 0.01, n),
           'infl_shock': rng.normal(0, 0.015, n)}
    u = diagnostics.uncertainty_split(raw)
    assert u['epistemic_wealth'] > 0.95, u
    assert u['epistemic_spending'] < 0.10, u
    # the headline share follows SPENDING when spending is available
    assert u['epistemic_share'] == u['epistemic_spending'], u


def test_worst_year_reports_the_bad_case_not_the_median_path():
    """The column read exactly 100% for EVERY strategy in every run, because
    `worst_cut` is a median across paths and under guyton_klinger or
    constant_real the median path never cuts. Meanwhile ROBUSTNESS in the same
    report said one path in five takes a cut — two true statements that read as
    a contradiction, because nothing said the first was the median."""
    cfg = {'age': 37, 'time_horizon_years': 58, 'retirement_age': 65,
           'adequacy': 'per_band',
           'withdrawal': {'start_age': 65, 'policy': 'guyton_klinger',
                          'floor': 0.85, 'target_real_annual': 100_000.0}}
    n = 1000
    # 80% of paths never cut; 20% take three compounding 10% cuts
    spending = np.full((n, 30), 100_000.0)
    spending[800:, 10] = 100_000.0 * 0.9 ** 3
    raw = {'spending': spending, 'depleted': np.zeros(n, dtype=bool),
           'depletion_ages': [None] * n, 'finals': np.full(n, 1e7),
           'v_at_retirement': np.full(n, 5e6), 'returns': None,
           'rebalancing_raw': [], 'path_mu': np.full(n, 0.08),
           'rf_shock': np.zeros(n), 'infl_shock': np.zeros(n),
           'regime_months': [None] * n}
    got = solver.score(raw, cfg, 100_000.0, floor=0.85)
    # the median path is untouched, which is exactly why it says nothing
    assert got['worst_cut'] == 0.0, got['worst_cut']
    # ...and the bad case sees the cuts
    assert abs(got['worst_cut_bad'] - (1 - 0.9 ** 3)) < 1e-9, got['worst_cut_bad']
    assert abs(report_retirement.worst_year_pct(got['worst_cut_bad']) - 72.9) < 0.1

    # the column must RENDER the bad case, falling back only when it is absent
    col = report_retirement._LEVEL_COLUMNS['worst']
    assert col[1] == '(p10)', 'the header must say which case this is'
    assert col[3]({'worst_cut': 0.0, 'worst_cut_bad': 0.271}) == '73%'
    assert col[3]({'worst_cut': 0.0}) == '100%', 'old records must still render'


def test_the_record_keeps_every_column_the_report_shows():
    """502 of 502 saved results carried no annual_vol: it is rendered in
    IN DETAIL and was never in the SCALARS whitelist, so `--diff` could not
    compare the one column that describes RISK and no saved run could be
    re-read for it. `twr_cagr` is a percentile ARRAY, which a record should not
    carry — but dropping it lost the CAGR column entirely, so its median is
    stored instead."""
    rendered = set()
    for _key, spec in report_retirement._LEVEL_COLUMNS.items():
        src = inspect.getsource(spec[3])
        rendered |= set(re.findall(r"r(?:\.get\(|\[)'([a-z_0-9]+)'", src))
    stored = set(persistence.SCALARS) | {'sortino_median', 'cagr_median'}
    # twr_cagr is stored as its median under a different name
    missing = {c for c in rendered if c not in stored and c != 'twr_cagr'}
    assert not missing, f'shown in the report, absent from the record: {missing}'

    got = persistence._summarise({'spend': 1.0, 'annual_vol': 14.6,
                                  'twr_cagr': np.array([.07, .08, .084, .09, .1]),
                                  'sortino_median': 1.03})
    assert got['annual_vol'] == 14.6, got
    assert abs(got['cagr_median'] - 0.084) < 1e-9, got
    # an array must never reach the record
    assert not any(isinstance(v, np.ndarray) for v in got.values()), got


def test_the_failing_path_count_is_not_mistakable_for_a_portfolio_id():
    """It was printed INSIDE the strategy column as "↳ 145 failing", truncated
    to "↳ 145 failin" — and 140, 144 and 145 are all real portfolio ids, so a
    path count read as a portfolio. The count now sits after the percentages,
    where nothing else is numbered."""
    cfg = {'age': 37, 'time_horizon_years': 58, 'retirement_age': 65,
           'adequacy': 'per_band',
           'withdrawal': {'start_age': 65, 'policy': 'guyton_klinger',
                          'floor': 0.85, 'target_real_annual': 200_000.0}}
    n = 200
    spending = np.full((n, 30), 300_000.0)
    spending[:145, 25:] = 10_000.0          # 145 paths starve late
    raw = {'spending': spending, 'depleted': np.zeros(n, dtype=bool),
           'depletion_ages': [None] * n, 'finals': np.full(n, 1e7),
           'path_mu': np.full(n, 0.08), 'rf_shock': np.zeros(n),
           'infl_shock': np.zeros(n), 'regime_months': [None] * n,
           'spend': 200_000.0}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        diagnostics.report({'140': raw}, cfg, 0.85)
    out = buf.getvalue()
    row = [l for l in out.splitlines() if 'failing' in l][0]
    # The label column only — `split('%')[0]` would swallow the first
    # percentage's digits and flag a row that is perfectly clear.
    label = row[:4 + 14]
    assert not any(ch.isdigit() for ch in label), repr(label)
    assert 'failing' in label and not label.rstrip().endswith('failin'), repr(label)
    # ...and the count must still be reported, just somewhere unambiguous
    assert '145' in row[4 + 14:].split('%')[-1], row


def test_the_uncertainty_note_states_what_was_measured():
    """The note asserted "a one-fund book should read HIGHER" as a prediction.
    Measured at 1500 paths it holds on BEQUEST (VT 56%) and fails on SPENDING
    (VT 29%, the lowest of three) — the policy sits between the draw and the
    lifestyle. A note that contradicts this repo's own measurement is worse
    than no note."""
    src = inspect.getsource(diagnostics.report)
    assert 'should read HIGHER' not in src, 'the refuted prediction is back'
    cfg = {'age': 37, 'time_horizon_years': 58, 'retirement_age': 65,
           'adequacy': 'per_band',
           'withdrawal': {'start_age': 65, 'policy': 'guyton_klinger',
                          'floor': 0.85, 'target_real_annual': 200_000.0}}
    n = 100
    raw = {'spending': np.full((n, 30), 300_000.0),
           'depleted': np.zeros(n, dtype=bool), 'depletion_ages': [None] * n,
           'finals': np.full(n, 1e7), 'path_mu': np.linspace(.06, .10, n),
           'rf_shock': np.zeros(n), 'infl_shock': np.zeros(n),
           'regime_months': [None] * n, 'spend': 200_000.0}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        diagnostics.report({'140': raw}, cfg, 0.85)
    out = buf.getvalue()
    assert 'on spending' in out and 'on bequest' in out, out
    assert '52%' in out and '40-42%' in out, 'the measured figures are not cited'


def test_the_search_curve_is_printed_and_says_when_it_did_not_bracket():
    """Every solver defect this repo has had was a property of these points and
    none was visible: a flat local fit skipping verification (513e4e4), a
    crossing clipped to a probe and accepted on tolerance (e775e47), a bracket
    seeded blind at $50k (7f552e2). Diagnosing any needed a re-solve."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        # a curve that CROSSES 90% — no warning
        solver._print_curve([(100e3, 0.93), (150e3, 0.91), (200e3, 0.87)],
                            0.90, n_sims=400)
    out = buf.getvalue()
    assert 'probes:' in out, out
    assert '100k' in out and '200k' in out, out
    assert 'not bracketed' not in out, out

    # every probe BELOW the target — the crossing is extrapolated
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        solver._print_curve([(100e3, 0.88), (150e3, 0.85), (200e3, 0.80)],
                            0.90, n_sims=400)
    assert 'extrapolated' in buf.getvalue(), buf.getvalue()
    assert 'sits below' in buf.getvalue(), buf.getvalue()
    # it must show HOW FAR off, and point at what settles it. Four strategies
    # tripped this in one run and all four verified at 90.0-90.4%: the search
    # window's low end sits under the target because `bracket_floor` is slack
    # by design, so the line describes rather than alarms.
    assert 'verification step' in buf.getvalue(), buf.getvalue()

    # every probe ABOVE it, the other failure
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        solver._print_curve([(100e3, 0.99), (150e3, 0.96), (200e3, 0.94)],
                            0.90, n_sims=400)
    assert 'sits above' in buf.getvalue(), buf.getvalue()

    # a probe at 91% with a 1.5pp band is ABOVE the target, not "within noise":
    # keying the bracket test on the band warned that a good solve was
    # extrapolated. The band marks per-probe resolution and nothing else.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        solver._print_curve([(109e3, 0.91), (193e3, 0.84)], 0.90,
                            n_sims=400)
    assert 'extrapolated' not in buf.getvalue(), buf.getvalue()

    # the band must follow the sample size, not be pinned
    small, large = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(small):
        solver._print_curve([(100e3, 0.88)], 0.90, n_sims=150)
    with contextlib.redirect_stdout(large):
        solver._print_curve([(100e3, 0.88)], 0.90, n_sims=4000)
    assert '=' in small.getvalue(), small.getvalue()   # 2.4pp -> within noise
    assert '<' in large.getvalue(), large.getvalue()   # 0.5pp -> resolved


def test_sequence_risk_controls_for_compounding():
    """An early return compounds for thirty years and a late one for ten, so the
    first decade explains more even when ORDER is irrelevant. Comparing the two
    raw R-squareds measures compounding, not sequence risk — the test has to
    hold the WHOLE-retirement return fixed."""
    cfg = {'age': 37, 'time_horizon_years': 58, 'retirement_age': 65,
           'withdrawal': {'start_age': 65}}
    rng = np.random.default_rng(3)
    n, months = 400, 696
    rets = rng.normal(0.006, 0.04, (n, months))

    # outcome driven ONLY by the total retirement return -> no sequence effect
    start = (65 - 37) * 12
    total = np.log1p(rets[:, start:]).sum(axis=1)
    spending = np.exp(total)[:, None] * np.full((n, 30), 2e5)
    got = diagnostics.sequence_risk({'returns': rets, 'spending': spending}, cfg)
    assert got is not None
    assert got['first_decade'] > 0.05, got     # still explains a lot...
    assert got['sequence_extra'] < 0.02, got   # ...but adds nothing over total

    # now make the FIRST decade matter beyond the total
    early = np.log1p(rets[:, start:start + 120]).sum(axis=1)
    spending = np.exp(total + 3 * early)[:, None] * np.full((n, 30), 2e5)
    got = diagnostics.sequence_risk({'returns': rets, 'spending': spending}, cfg)
    assert got['sequence_extra'] > 0.10, got

    # a search evaluation has no monthly series and must report nothing
    assert diagnostics.sequence_risk({'returns': None, 'spending': spending},
                                     cfg) is None


def test_the_debug_report_prints_its_sequence_risk_and_regime_occupancy_blocks():
    """Every other report() test passes `regime_months: [None]*n` and no
    `returns`, so both blocks were skipped and a key or format bug in them
    would reach a user only as the swallowed "(diagnostics unavailable)"."""
    cfg = {'age': 37, 'time_horizon_years': 58, 'retirement_age': 65,
           'adequacy': 'per_band',
           'withdrawal': {'start_age': 65, 'policy': 'guyton_klinger',
                          'floor': 0.85, 'target_real_annual': 200_000.0}}
    rng = np.random.default_rng(3)
    n = 400
    rets = rng.normal(0.006, 0.04, (n, 696))
    start = (65 - 37) * 12
    total = np.log1p(rets[:, start:]).sum(axis=1)
    early = np.log1p(rets[:, start:start + 120]).sum(axis=1)
    st = diagnostics.market_stationary()
    on_chain = [{k: int(1000 * v) for k, v in st.items()} for _ in range(n)]
    stuck = [{'low': 0, 'medium': 1000, 'high': 0} for _ in range(n)]

    def render(months):
        raw = {'spending': np.exp(total + 3 * early)[:, None] * np.full((n, 30), 2e5),
               'returns': rets, 'regime_months': months,
               'depleted': np.zeros(n, dtype=bool), 'depletion_ages': [None] * n,
               'finals': np.full(n, 1e7), 'path_mu': np.linspace(.06, .10, n),
               'rf_shock': np.zeros(n), 'infl_shock': np.zeros(n),
               'spend': 200_000.0}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            diagnostics.report({'140': raw}, cfg, 0.85)
        return buf.getvalue()

    out = render(on_chain)
    assert 'SEQUENCE RISK' in out and 'ORDER alone' in out, out
    row = out.split('first 10y')[1].splitlines()[1]
    assert row.split()[0] == '140' and row.count('%') == 4, row
    assert 'REGIME OCCUPANCY' in out and 'stationary' in out, out
    assert 'LARGER THAN EXPECTED' not in out, out
    assert 'is expected' in out, out

    out = render(stuck)
    assert 'REGIME OCCUPANCY' in out and 'LARGER THAN EXPECTED' in out, out


def test_every_moved_assumption_is_announced():
    """Twelve env vars move every number and only PREMIUM_SCALE was announced —
    as "equity premium scaled", the mislabel already corrected in the footer.
    They reached the run RECORD through provenance, so --diff could see them
    while the run on screen looked identical to a default one."""

    # a default run must say NOTHING, or the banner cries wolf every time
    assert diagnostics.moved_assumptions() == [], diagnostics.moved_assumptions()

    class Fake:
        pass
    f = Fake()
    for attr in diagnostics.DEFAULT_ASSUMPTIONS:
        default = diagnostics.DEFAULT_ASSUMPTIONS[attr][0]
        setattr(f, attr, default)
    f.PREMIUM_SCALE, f.TAIL_DF, f.MU_METHOD = 0.5, 5, 'historical'
    moved = {m[0] for m in diagnostics.moved_assumptions(f)}
    assert moved == {'PREMIUM_SCALE', 'TAIL_DF', 'MU_METHOD'}, moved

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        diagnostics.assumption_banner(f)
    out = buf.getvalue()
    for env in ('MC_PREMIUM_SCALE', 'MC_TAIL_DF', 'MC_MU_METHOD'):
        assert env in out, (env, out)
    assert '0.5' in out and 'historical' in out, out

    # the old single-knob line, with its mislabel, must be gone
    src = inspect.getsource(report_retirement)
    assert 'equity premium scaled to' not in src, 'the mislabelled line is back'
    assert 'assumption_banner' in src, 'the banner is not wired in'


def test_a_scenario_alias_for_the_default_is_not_announced_as_moved():
    """`us stays ahead` is the default under another name, there so the status
    quo reads as a choice. The banner compared NAMES and called it a moved knob
    on the 2026-09-29 regions run, whose numbers were the shipped model's. Only
    a scenario that changes a premium is moved — and one that does must still
    be caught, or the fix has blinded the banner."""

    class Fake:
        pass
    f = Fake()
    for attr, (default, _) in diagnostics.DEFAULT_ASSUMPTIONS.items():
        setattr(f, attr, default)

    f.PREMIA_SCENARIO = 'us stays ahead'
    assert diagnostics.moved_assumptions(f) == [], diagnostics.moved_assumptions(f)

    f.PREMIA_SCENARIO = 'intl catches up'
    moved = {m[0] for m in diagnostics.moved_assumptions(f)}
    assert moved == {'PREMIA_SCENARIO'}, moved


def test_transition_matrix_rows_are_a_distribution_even_when_a_state_is_unseen():
    """The Markov estimator behind every regime in the model, and untested. A
    row with no observations must still be a distribution — the engine draws
    from it directly, and a row of zeros would make select_next_regime pick
    nothing."""
    import data_collection_regime_model as dcm
    labels = ['low'] * 10 + ['medium'] * 10 + ['low'] * 5
    out = dcm._transition_matrix(labels)
    for state, row in out.items():
        assert abs(sum(row.values()) - 1.0) < 1e-9, (state, row)
        assert all(v >= 0 for v in row.values()), (state, row)
    # 'high' never occurs: it must fall back, not divide by zero
    assert abs(sum(out['high'].values()) - 1.0) < 1e-9, out['high']
    # a perfectly persistent series must read as persistent
    out = dcm._transition_matrix(['medium'] * 50)
    assert out['medium']['medium'] == 1.0, out['medium']


def test_conditional_transitions_shrink_toward_the_unconditional_row():
    """Shrinkage is not a refinement here, it is what makes the estimate usable:
    the RAW conditional scores -2.36 out of sample against the unconditional
    -0.53, because cells are nearly empty (VOO shows P(medium->high) = 1.00 off
    ONE observation). k=10 scores -0.48. The constant was load-bearing and had
    no unit test."""
    import data_collection_regime_model as dcm
    idx = pd.period_range('2000-01', periods=120, freq='M')
    rng = np.random.default_rng(0)
    lab = pd.Series(rng.choice(['low', 'medium', 'high'], 120),
                    index=idx.to_timestamp())
    mkt = pd.Series(['high'] * 120, index=idx)

    # k=0 is the raw conditional estimate; large k must pull toward uncond
    raw, n = dcm.conditional_transitions(lab, mkt, k=0)
    heavy, _ = dcm.conditional_transitions(lab, mkt, k=10_000)
    uncond = dcm._transition_matrix(lab.values)
    assert n > 0, n
    for a in ('low', 'medium', 'high'):
        near = max(abs(heavy['high'][a][b] - uncond[a][b])
                   for b in ('low', 'medium', 'high'))
        assert near < 0.01, (a, heavy['high'][a], uncond[a])
        # every row stays a distribution at any k
        for r in ('low', 'medium', 'high'):
            assert abs(sum(raw[r][a].values()) - 1.0) < 1e-9, raw[r][a]

    # a single observation must NOT produce a probability of 1.0 once shrunk —
    # that is the exact pathology the shrinkage exists to prevent
    tiny_idx = pd.period_range('2000-01', periods=30, freq='M')
    one = pd.Series(['medium'] * 29 + ['high'], index=tiny_idx.to_timestamp())
    got, _ = dcm.conditional_transitions(one, pd.Series(['high'] * 30,
                                                        index=tiny_idx))
    assert got['high']['medium']['high'] < 1.0, got['high']['medium']

    # too little overlap must fall back to the unconditional row for every state
    short = lab.iloc[:10]
    got, n = dcm.conditional_transitions(short, mkt.iloc[:10])
    assert n == 0, n
    assert got['low'] == got['high'], 'a fallback must not vary by market regime'


def test_the_pipeline_repair_produces_what_the_engine_can_factor():
    """Two implementations of one idea: the pipeline writes matrices with
    `fix_correlation_matrix` and the engine repairs with `near_psd_higham`. The
    engine's is tested; the pipeline's — which produces the files actually
    shipped — was not. If they disagree the matrices on disk can be ones the
    simulation cannot Cholesky-factor."""
    import data_collection_regime_model as dcm
    bad = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    assert np.linalg.eigvalsh(bad).min() < 0, 'the fixture must be indefinite'
    fixed = dcm.fix_correlation_matrix(bad)
    assert np.allclose(np.diag(fixed), 1.0), np.diag(fixed)
    assert np.allclose(fixed, fixed.T), 'a correlation matrix must be symmetric'
    assert np.linalg.eigvalsh(fixed).min() >= 0, np.linalg.eigvalsh(fixed).min()
    # the engine must be able to factor what the pipeline writes
    pa.cholesky_factor(fixed)
    # and an already-valid matrix must come back essentially unchanged
    good = np.array([[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]])
    assert np.allclose(dcm.fix_correlation_matrix(good), good, atol=1e-8)


def test_score_proxy_separates_a_good_backfill_from_a_rotated_one():
    """The function that FOUND the worst data defect in this repo's history —
    four international proxies rotated by one, VEA carrying EMERGING market
    returns for 41% of its history at corr 0.856 — and it had no unit test of
    its own. Its numbers gate every regeneration through validate_parameters.
    """
    import data_collection_regime_model as dcm
    idx = pd.date_range('2015-01-31', periods=120, freq='ME')
    rng = np.random.default_rng(7)
    common = rng.normal(0.008, 0.045, 120)

    # a GOOD proxy: the fund plus small idiosyncratic noise
    fund = pd.Series(common + rng.normal(0, 0.004, 120), index=idx)
    good = pd.Series(common + rng.normal(0, 0.004, 120), index=idx)
    g = dcm.score_proxy(fund, good, contrib=158, label='DFALX')
    assert g['corr'] > 0.95, g
    assert 0.85 < g['beta'] < 1.15, g
    assert g['tracking_error'] < 0.05, g
    assert g['overlap_months'] == 120 and g['contributed_months'] == 158, g

    # a ROTATED proxy: a different asset entirely. This is the shape that went
    # unnoticed for as long as the table existed.
    wrong = pd.Series(rng.normal(0.008, 0.075, 120), index=idx)
    b = dcm.score_proxy(fund, wrong, contrib=158, label='VEIEX')
    assert b['corr'] < 0.5, b
    assert b['tracking_error'] > g['tracking_error'] * 3, (b, g)

    # JPST's defect was VOLUME, not direction: a treasury fund standing in for
    # a credit fund, with 44% too little volatility. corr alone would miss it.
    quiet = pd.Series(common * 0.5, index=idx)
    q = dcm.score_proxy(fund, quiet, contrib=44, label='SHV')
    assert q['corr'] > 0.9, 'this defect is invisible to correlation'
    assert q['proxy_vol'] < 0.6 * q['fund_vol'], q

    # too little overlap must be FLAGGED, not scored on noise
    thin = dcm.score_proxy(fund.iloc[:6], good.iloc[:6], 0, 'P')
    assert thin.get('thin') is True, thin
    assert 'corr' not in thin, 'a 6-month correlation must not be reported'

    # months BEFORE the fund existed cannot be scored — that is the whole point
    early = pd.Series(rng.normal(0, 0.05, 240),
                      index=pd.date_range('2005-01-31', periods=240, freq='ME'))
    scoped = dcm.score_proxy(fund, early, 0, 'P')
    assert scoped['overlap_months'] <= 120, scoped


def test_tips_fetch_survives_a_dead_network_and_parses_a_live_one():
    """`RF_REAL` is the biggest level assumption in the model and the only
    QUOTED input. A network failure must not be fatal — the model has a default
    — and a successful fetch must ship the spot AND its own history, because
    today's 30-year print sits at the top of its entire range."""
    import urllib.request
    import data_collection_regime_model as dcm

    csv = b'observation_date,DFII30\n' + b''.join(
        f"20{y:02d}-01-01,{1.0 + 0.1 * y}\n".encode() for y in range(10, 26))

    class Resp:
        def __init__(self, body):
            self.body = body

        def read(self):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    real = urllib.request.urlopen
    try:
        urllib.request.urlopen = lambda *a, **k: Resp(csv)
        got = dcm.fetch_tips_real_yields()
        assert '30' in got, got
        row = got['30']
        # percent in the file, FRACTION in the output — a units slip here is
        # the same class as the VOO price-index bug
        assert 0.02 < row['latest'] < 0.04, row
        assert row['min'] < row['latest'], row
        assert row['avg_full'] < row['latest'], 'the spot is at the top here'
        assert row['as_of'].startswith('2025'), row

        # a DEAD network must return empty, not raise
        def boom(*a, **k):
            raise OSError('no route to host')
        urllib.request.urlopen = boom
        assert dcm.fetch_tips_real_yields() == {}, 'network failure must not be fatal'

        # a file that parses but holds no numbers must be skipped, not stored
        urllib.request.urlopen = lambda *a, **k: Resp(
            b'observation_date,DFII30\n2020-01-01,.\n2020-01-02,.\n')
        assert dcm.fetch_tips_real_yields() == {}, 'an all-NaN series was stored'
    finally:
        urllib.request.urlopen = real


def test_a_depleted_path_can_never_be_adequate():
    """`stat = np.where(depleted, -np.inf, stat)`. Flip that sign and every
    ruined path scores as the BEST possible outcome, so the solver certifies
    levels that bankrupt you — and nothing caught it. Found by
    tools/vacuity_sweep.py --subtle.

    A path that ran out cannot be adequate at any spend, however rich its
    spending looked before it did.
    """
    cfg = {'age': 37, 'time_horizon_years': 58, 'retirement_age': 65,
           'adequacy': 'per_band',
           'withdrawal': {'start_age': 65, 'policy': 'vpw', 'floor': 0.85,
                          'target_real_annual': 100_000.0}}
    n = 100
    # every path spends lavishly; HALF of them run out doing it
    spending = np.full((n, 30), 900_000.0)
    depleted = np.zeros(n, dtype=bool)
    depleted[:50] = True
    raw = {'spending': spending, 'depleted': depleted,
           'depletion_ages': [80 if d else None for d in depleted],
           'finals': np.where(depleted, 0.0, 1e7),
           'v_at_retirement': np.full(n, 5e6), 'returns': None,
           'rebalancing_raw': [], 'path_mu': np.full(n, 0.08),
           'rf_shock': np.zeros(n), 'infl_shock': np.zeros(n),
           'regime_months': [None] * n}
    got = solver.score(raw, cfg, 100_000.0, floor=0.85)
    # exactly the non-depleted half can succeed
    assert abs(got['success_rate'] - 0.5) < 1e-9, got['success_rate']
    assert abs(got['depletion_rate'] - 0.5) < 1e-9, got['depletion_rate']

    # THE CLOSED-FORM ROUTE, which is where the sign actually lives. `score`
    # has its own depletion handling, so asserting there passed happily while
    # solve_sustainable_spend was mutated — testing the wrong path is how a
    # hole survives a test written to close it.
    real = solver.simulate
    cfg_vpw = dict(cfg)
    cfg_vpw['withdrawal'] = dict(cfg['withdrawal'], policy='vpw')
    try:
        solver.simulate = lambda *a, **k: raw
        out = solver.solve_sustainable_spend(
            {'Z': 1.0}, cfg_vpw, confidence=0.90, floor=0.85,
            verify_sims=n, parallel=False)
    finally:
        solver.simulate = real
    assert out['solved_by'] == 'closed_form', out.get('solved_by')
    # Half the paths ruin. A level certified at 90% confidence cannot be set
    # from their spending: if depleted paths sorted as the BEST outcome the
    # 10th percentile would land among the solvent ones and the level would be
    # the lavish 900k, not a figure the ruined half drags down.
    assert out['spend'] < 900_000.0, out['spend']
    assert out['spend'] <= 1.0, (
        'a level was certified from paths that ran out: %r' % out['spend'])


def test_the_closed_form_takes_the_LOWER_tail():
    """`np.percentile(stat, 100 * (1 - confidence))`. Use `100 * confidence`
    instead and a 90%-confidence solve reads the 90th percentile rather than
    the 10th — certifying a level only one path in ten can sustain. Nothing
    caught it; vpw and every self-directed policy goes down this route.
    """
    src = re.sub(r'#[^\n]*', '', inspect.getsource(solver.solve_sustainable_spend))
    assert '100.0 * (1.0 - confidence)' in src, \
        'the closed form is no longer reading the lower tail'

    # ...and the behaviour, not just the expression: a spread of outcomes must
    # certify near the BOTTOM of it, never the top.
    cfg = {'age': 37, 'time_horizon_years': 58, 'retirement_age': 65,
           'adequacy': 'per_band',
           'withdrawal': {'start_age': 65, 'policy': 'vpw', 'floor': 1.0,
                          'target_real_annual': 100_000.0}}
    n = 1000
    rng = np.random.default_rng(2)
    per_path = rng.uniform(50_000, 250_000, n)
    spending = np.tile(per_path[:, None], (1, 30))
    stat = solver.adequacy_statistic(spending, cfg, 'per_band')
    lower = float(np.percentile(stat, 10))
    upper = float(np.percentile(stat, 90))
    assert lower < upper, (lower, upper)
    # the level a 90%-confidence solve should reach is the LOWER one
    assert abs(lower - 70_000) < 15_000, lower
    assert upper > 200_000, 'the upper tail is what the bug would have taken'


def test_inflation_moves_real_returns_DOWN_not_just_differentially():
    """`test_inflation_hits_bonds_harder_than_equities` pins the DIFFERENTIAL
    and passed happily when the sign was flipped with abs(): -1.15 and -0.80
    become +1.15 and +0.80, which preserves the ordering while making inflation
    a windfall for every asset. Found by tools/vacuity_sweep.py --subtle.

    The direction is the whole point: a nominal bond LOSES real return to
    inflation.
    """
    assert pa.INFL_BETA_DEFAULT < 0, pa.INFL_BETA_DEFAULT
    for t, b in pa.INFL_BETA.items():
        assert b <= 0, f'{t} gains real return from inflation: {b}'

    # and the engine must apply it with its sign intact, not its magnitude
    src = re.sub(r'#[^\n]*', '', inspect.getsource(pa.run_single_simulation))
    assert 'INFL_BETA.get(ticker, INFL_BETA_DEFAULT) * infl_shock' in src, src[-400:]
    assert 'abs(INFL_BETA' not in src, 'the sign is being discarded'

    # behaviour: a POSITIVE inflation draw must lower a bond's expected return
    beta = pa.INFL_BETA.get('BND', pa.INFL_BETA_DEFAULT)
    assert beta * 0.01 < 0, beta
    eq = pa.INFL_BETA.get('VOO', pa.INFL_BETA_DEFAULT)
    assert eq * 0.01 < 0, eq
    assert beta * 0.01 < eq * 0.01, 'bonds must be hurt MORE, and both hurt'


def test_a_near_depleted_path_is_censored_not_regressed():
    """Excluding `bequest == 0` was not enough. A path that NEARLY depletes ends
    on a few hundred dollars, and log() puts it ~14 units from the mass: three
    to nine such paths out of 1,500 swung this statistic from 7% to 38% across
    six independent samples of ONE strategy, while the spending column beside it
    held 39-45%.

    Those paths are censored by the withdrawal policy, not drawn from the wealth
    distribution, so they belong with `depleted`. The threshold is one year of
    spending — a figure with model meaning, not a percentile chosen to look
    stable — and a Spearman rank correlation, which needs no threshold at all,
    agrees within 1pp on every sample.
    """
    rng = np.random.default_rng(11)
    n = 1500
    mu = rng.normal(0.08, 0.02, n)
    spending = np.full((n, 30), 200_000.0)
    # a clean lognormal bequest that mu genuinely drives
    # sd 0.4 against a mu spread of 0.02*30 = 0.6, so the belief DOMINATES:
    # the test is about censoring, not about the fixture's signal-to-noise.
    finals = np.exp(mu * 30 + rng.normal(0, 0.4, n)) * 1e6
    raw = {'path_mu': mu, 'spending': spending, 'finals': finals.copy(),
           'rf_shock': rng.normal(0, 0.01, n),
           'infl_shock': rng.normal(0, 0.015, n)}
    clean = diagnostics.uncertainty_split(raw)['epistemic_wealth']
    assert clean > 0.5, clean

    # five paths that nearly ran out. NOT zero — zero was already excluded.
    raw['finals'][:5] = np.array([12.0, 480.0, 95.0, 1_200.0, 37.0])
    got = diagnostics.uncertainty_split(raw)
    assert got['wealth_censored'] == 5, got['wealth_censored']
    assert abs(got['epistemic_wealth'] - clean) < 0.03, (clean, got)

    # without the censor those five alone would wreck it — that is the defect
    mu_ok = np.isfinite(mu)
    wrecked = float(np.corrcoef(mu[mu_ok],
                                np.log(raw['finals'][mu_ok]))[0, 1]) ** 2
    assert wrecked < clean - 0.15, (wrecked, clean)

    # and the threshold must track SPENDING, not a hard-coded dollar figure
    raw2 = dict(raw, spending=np.full((n, 30), 20_000.0))
    raw2['finals'] = finals.copy()
    raw2['finals'][:5] = 25_000.0          # above 1yr of 20k -> kept
    assert diagnostics.uncertainty_split(raw2)['wealth_censored'] == 0

    # censoring MOST of the run describes a selected subsample, not the run
    raw3 = dict(raw)
    raw3['finals'] = finals.copy()
    raw3['finals'][:400] = 50.0
    assert diagnostics.uncertainty_split(raw3)['epistemic_wealth'] is None


def test_solve_mode_does_not_print_the_plan_level_matrix():
    """The plan level is the noisiest column in the report — a poorly-estimated
    percentile with a ~+/-$13k standard error on a ~$200k figure — and printing
    it first made it the headline while the text told the reader three times to
    rank on realised spending. It produced the shape behind four separate fixes
    (the derisk baseline, the "highest bar" tie, "most spending", IN DETAIL's
    ranking): a number read to a precision it does not have.

    The SAME matrix is MEDIAN WEALTH in accumulation and SUCCESS RATE at a fixed
    spend, and both are the main table of their view — so this is about solve
    mode only, and HEAD-TO-HEAD must still rank on the cells it builds.
    """
    P = _P()
    rng = np.random.default_rng(9)

    def row(level, realised):
        return {'spend': level, 'median_realised': realised,
                'p10_realised': realised * 0.42, 'annual_vol': 14.4,
                'per_path_mean': realised + rng.normal(0, 55e3, 400),
                'median_bequest': 47e6, 'worst_cut': 0.0,
                'worst_cut_bad': 0.271, 'clipped': 0, 'level_se': 13_000.0,
                'twr_cagr': np.full(5, 0.084), 'sortino_median': 1.04,
                'sortino': np.full(400, 1.04), 'depletion_rate': 0.0,
                'success_rate': 0.90, 'median': level / 1e6}

    specs = ['140', '140>140+10@65', '165', '165>165+10@65']
    res = {'140': row(188e3, 506e3), '140>140+10@65': row(174e3, 443e3),
           '165': row(205e3, 577e3), '165>165+10@65': row(205e3, 539e3)}

    def render(spend):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report_retirement._matrix(specs, res, {
                'spend': spend, 'mode': 'retirement',
                'portfolios': ['140', '165'], 'derisk_at': 65,
                'bond_tiers': [0.10]}, P)
        return buf.getvalue()

    solve = render('solve')
    assert 'PLAN LEVEL' not in solve, solve[:400]
    assert 'highest bar' not in solve, solve[:400]
    # ...but the sections that matter are all still there
    for section in ('IN DETAIL', 'HEAD-TO-HEAD'):
        assert section in solve, (section, solve[:400])
    # and HEAD-TO-HEAD comes AFTER the detail, not before it
    assert solve.index('IN DETAIL') < solve.index('HEAD-TO-HEAD'), \
        'HEAD-TO-HEAD must read as the verdict on the table above it'

    # a FIXED-SPEND run keeps its matrix: success rate is what differs there
    fixed = render(200_000.0)
    assert 'SUCCESS RATE' in fixed, fixed[:400]


def test_the_interval_chart_puts_zero_where_the_verdict_changes():
    """`-31.3k  +13.9k` reads as "roughly break-even" and MEANS "we cannot
    tell". The verdict column already says so; the number line stops the digits
    arguing with it, because crossing zero is a question about POSITION."""

    def draw(rows, label_width=8):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report_retirement._interval_chart(rows, label_width)
        return buf.getvalue()

    out = draw([('A', 33.5e3, 85.7e3), ('B', -31.3e3, 13.9e3),
                ('C', -70e3, -20e3)])
    body = {l.split()[0]: l for l in out.splitlines() if l.strip()
            and l.split()[0] in ('A', 'B', 'C')}
    zero = body['A'].index('|') if '|' in body['A'] else None
    assert zero is not None, out
    # every row must mark zero in the SAME column, or the chart lies
    for tag, line in body.items():
        assert line.index('|') == zero, (tag, out)
    # an interval that STRADDLES zero must have bracket ends on both sides
    assert body['B'].index('[') < zero < body['B'].index(']'), out
    # one entirely positive sits wholly right of it; entirely negative, left
    assert body['A'].index('[') > zero, out
    assert body['C'].index(']') < zero, out
    # the axis must not run its labels together — it read "worse0better"
    assert 'worse0better' not in out, out
    assert '<- worse' in out and 'better ->' in out, out

    # a single row is a worse table, not a chart
    assert draw([('A', 1e3, 2e3)]) == ''
    assert draw([]) == ''
    # NaN rows (a strategy with no paired sample) must not crash it
    assert draw([('A', float('nan'), 1.0), ('B', 1e3, 2e3)]) == ''


def test_the_stacked_bar_keeps_one_width_whatever_the_rounding():
    """Three shares summing to 1 are what a stacked bar is for, and the SHAPE is
    the finding: across all paths the first decade binds ~85% of the time, and
    among the FAILING paths it inverts to the last. Rounding must not change the
    bar's length or the rows stop lining up under their own header."""
    W = diagnostics._STACK_WIDTH
    # [0.1, .1, .8] rounds to 2+2+19 = 23 and [.07, .07, .86] to 2+2+21 = 25:
    # these are the cases that catch a missing drift correction. Shares that
    # happen to round exactly would pass against no correction at all.
    for shares in ([0.85, 0.07, 0.08], [1 / 3, 1 / 3, 1 / 3],
                   [0.1, 0.1, 0.8], [0.07, 0.07, 0.86],
                   [0.15, 0.33, 0.52], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                   [0.334, 0.333, 0.333]):
        bar = diagnostics._stack(shares)
        assert len(bar) == W, (shares, len(bar), bar)
    # the glyph ORDER must follow the life stages, not the sizes
    bar = diagnostics._stack([0.5, 0.25, 0.25])
    assert bar.startswith('#') and bar.endswith('.'), bar
    assert bar.index('+') < bar.index('.'), bar
    # a dominant last stage must read as one
    assert diagnostics._stack([0.0, 0.0, 1.0]) == '.' * W
    assert diagnostics._stack([]) == ''


def test_retirement_mode_can_set_the_path_count():
    """Retirement mode never asked for a path count — it was only ever in the
    ACCUMULATION branch. So a reader who saw `not resolved` and wanted to know
    whether more paths would settle it had to edit source or learn the --spec
    syntax. A plan level is a poorly-estimated percentile: a 600-path sweep in
    this repo's record reported a ranking reversal that vanished at 1,600."""

    def answer(typed):
        spec = {}
        real = runner.ask
        runner.ask = lambda _p, _d='': typed if typed else _d
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                runner.ask_paths(spec)
        finally:
            runner.ask = real
        return spec

    assert answer('') == {}, 'the default must leave the spec untouched'
    got = answer('4000')
    assert got['verify_sims'] == 4000, got
    # the SHIPPED ratio, not half: search runs ~15 evaluations against
    # verification's 1-3, so scaling it to n/2 made 2x the paths cost 3.2x the
    # wall clock and silently retuned the solver.
    ratio = solver.SEARCH_SIMS / solver.VERIFY_SIMS
    assert got['search_sims'] == round(4000 * ratio), (got, ratio)

    # time must then scale PROPORTIONALLY to the number typed
    base = {'mode': 'retirement', 'spend': 'solve', 'policy': 'guyton_klinger'}
    t1 = runner.estimate_minutes(6, base)
    t2 = runner.estimate_minutes(6, {**base, **answer('3000')})
    assert abs(t2 / t1 - 2.0) < 0.15, (t1, t2)

    # shorthand, garbage and absurd values must not produce a broken run
    assert answer('4k')['verify_sims'] == 4000
    assert answer('4,000')['verify_sims'] == 4000
    assert answer('abc') == {}, 'garbage must fall back to the default'
    assert answer('99999999')['verify_sims'] <= 50_000, 'not clamped'
    assert answer('1')['verify_sims'] >= 100, 'not floored'

    # and it must be reached on a RETIREMENT run, which is the whole point
    src = re.sub(r'#[^\n]*', '', inspect.getsource(runner.prompt_run))
    # `prompt_run` has several `else:` before this one, so splitting on the
    # first is meaningless. The invariant is ORDER: ask_paths must come before
    # the accumulation-only sims prompt, which is the branch it is NOT on.
    assert 'ask_paths(spec)' in src, 'ask_paths is never called'
    acc = src.index("spec['sims'] = int(ask(")
    assert src.index('ask_paths(spec)') < acc, \
        'ask_paths sits on the accumulation branch, not the retirement one'


def test_the_replay_line_actually_reproduces_the_run():
    """Nine prompts produce a spec that is already saved, already in provenance
    and already replayable — and the only way back to it was to answer all nine
    identically. Printing the line is only worth doing if it ROUND-TRIPS, so
    that is what this asserts: parse the emitted command back and require the
    same spec and the same resolved strategies."""
    import shlex
    P = _P()
    spec = {'mode': 'retirement', 'portfolios': ['166'],
            'custom': ['166', '166>166+5@65', '166>166+5@65>166@70'],
            'spend': 'solve', 'policy': 'guyton_klinger',
            'gk_response': 'fast', 'verify_sims': 3000, 'search_sims': 800,
            'social_security': {'annual_at_fra': 60000.0, 'claim_age': 70.0,
                                'benefit_scale': 0.75}}
    cmd = runner.replay_command(spec)
    assert cmd.startswith('python3 run.py --spec '), cmd

    # split it the way a SHELL would, not by guessing where the quotes are
    payload = shlex.split(cmd)[-1]
    assert json.loads(payload) == spec, 'the emitted JSON is not the spec'
    back = runner.parse_spec(payload, P)
    assert back == spec, {k: (spec.get(k), back.get(k))
                          for k in set(spec) | set(back)
                          if spec.get(k) != back.get(k)}
    # ...and it must resolve to the SAME strategies, which is what a reader
    # actually wants reproduced
    assert (runner.resolve_strategies(spec, P)[0]
            == runner.resolve_strategies(back, P)[0])

    # a spec containing a shell metacharacter must survive quoting. Glide
    # syntax already carries '>' , which an unquoted line would REDIRECT.
    assert '>' in payload, 'the fixture no longer exercises redirection'
    assert payload.startswith("'") or '"' in cmd, 'the payload is unquoted'

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runner.print_replay(spec)
    out = buf.getvalue()
    assert 'to run this again' in out and '--spec' in out, out
    # the --sims hint belongs only where the count was NOT already chosen
    assert '--sims' not in out, 'hint shown despite an explicit path count'
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runner.print_replay({k: v for k, v in spec.items()
                             if k not in ('verify_sims', 'search_sims')})
    assert '--sims' in buf.getvalue(), buf.getvalue()
    # `--sims N` lands in the spec as 'sims', not verify_sims — and the hint
    # told a 3,000-path run to append --sims 3000 (2026-10-01)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runner.print_replay({**{k: v for k, v in spec.items()
                                if k not in ('verify_sims', 'search_sims')},
                             'sims': 3000})
    assert '--sims' not in buf.getvalue().split('--spec', 1)[1].split("'", 2)[2], \
        buf.getvalue()


def test_the_replay_line_carries_the_assumptions_the_run_was_made_under():
    """The replay line printed the spec and nothing else. A run made under
    MC_PREMIA_SCENARIO='intl catches up' told you to re-run it with a command
    that reproduces the DEFAULT scenario — a different run, printed under the
    words 'to run this again'. The knobs travel as env vars, so the line must
    set them, and it must set the value the run USED, read off the engine."""
    import shlex

    class Fake:
        pass
    f = Fake()
    for attr, (default, _) in diagnostics.DEFAULT_ASSUMPTIONS.items():
        setattr(f, attr, default)
    spec = {'mode': 'retirement', 'portfolios': ['161', '166'], 'spend': 'solve'}

    # the shipped model: no prefix at all
    assert runner.replay_command(spec, f).startswith('python3 run.py --spec '), \
        runner.replay_command(spec, f)

    f.PREMIA_SCENARIO, f.PREMIUM_SCALE = 'intl catches up', 0.5
    words = shlex.split(runner.replay_command(spec, f))
    cut = words.index('python3')
    env = dict(w.split('=', 1) for w in words[:cut])
    assert env == {'MC_PREMIA_SCENARIO': 'intl catches up',
                   'MC_PREMIUM_SCALE': '0.5'}, env
    assert words[cut:cut + 3] == ['python3', 'run.py', '--spec'], words
    assert json.loads(words[-1]) == spec, 'the prefix corrupted the payload'

    # an alias for the default is not a moved knob, so it earns no prefix
    f.PREMIA_SCENARIO, f.PREMIUM_SCALE = 'us stays ahead', 1.0
    assert runner.replay_command(spec, f).startswith('python3 run.py'), \
        runner.replay_command(spec, f)


def test_a_whole_schedule_can_be_pasted_as_the_starting_portfolio():
    """The start-portfolio line REJECTED '>' outright — while every strategy
    name the report prints, and the --spec line it emits to replay a run, are
    in exactly that form. Output you cannot feed back as input is a trap: the
    obvious move after reading `166>166+5@65>166@70` in a results table is to
    paste it, and it produced '? this line takes ONLY the starting id'."""
    P = _P()

    def build(*keys):
        # EXHAUSTED INPUT MUST RAISE, not return ''. ask_custom_schedule
        # re-asks forever on a bad line, so a harness that keeps handing it a
        # blank makes the whole gate HANG instead of fail — which reads as a
        # slow test. Deleting the paste branch is exactly the mutation that
        # triggers it.
        it = iter(keys)

        def _next(_p, _d=''):
            try:
                return next(it)
            except StopIteration:
                raise AssertionError(
                    'the prompt asked for more input than the test supplied — '
                    'it is looping') from None

        real = runner.ask
        runner.ask = _next
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                got = runner.ask_custom_schedule(P, 37, default_base='166')
        finally:
            runner.ask = real
        return got, buf.getvalue()

    # a pasted schedule is accepted VERBATIM and needs no further input
    for pasted in ('166>166+5@65>166@70', '166>166+5@65', '140>140+10@65'):
        got, out = build(pasted)
        assert got == pasted, (pasted, got)
        assert 'ONLY the starting id' not in out, out
        # and it must preview, like the built path does
        assert 'equity %' in out, out

    # it must still RESOLVE to the same strategies as building it by hand
    typed, _ = build('166', '166+5 at 65', '')
    pasted, _ = build('166>166+5@65')
    assert glide.parse_schedule(typed, 37) == glide.parse_schedule(pasted, 37)

    # a BAD paste must re-ask rather than be accepted or crash
    got, out = build('166>999@65', '166', '')
    assert got == '166', (got, out)
    assert '999' in out, out

    # the plain-id path is untouched
    got, _ = build('166', '')
    assert got == '166', got


def _drive(fn, keys):
    """Call an interactive prompt with scripted answers; return (result, stdout).

    The prompts are echoed into stdout, as a terminal shows them. Raises once
    the script runs out — a prompt that re-asks on bad input would otherwise
    loop forever on a default."""
    it = iter(keys)

    def _next(_p, _d=''):
        print(_p)
        try:
            return next(it)
        except StopIteration:
            raise AssertionError('the prompt asked for more input than the '
                                 'test supplied') from None

    real = runner.ask
    runner.ask = _next
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            got = fn()
    finally:
        runner.ask = real
    return got, buf.getvalue()


def test_custom_schedule_controls_keep_the_order_you_typed():
    """The controls were added with `insert(0, base)` one at a time, which
    REVERSED them: '140, 152, 165' came out as 165, 152, 140, so 165 became A
    — the head-to-head base and the plan every strategy is compared at — when
    the status quo had been typed first. A mixed schedule is used so no
    same-shape offer applies."""
    P = _P()
    got, _ = _drive(lambda: runner.ask_custom_schedules(
        P, 37, ['140', '152', '165']), ['140>166+5@65', 'n'])
    assert got == ['140', '152', '165', '140>166+5@65'], got


def test_same_shape_swaps_the_portfolio_and_keeps_everything_else():
    """One schedule's shape on another portfolio: ages, bond levels and an
    explicit sleeve carry over; only the portfolio id changes."""
    cases = {
        '140>140+5@60>140@65': '152>152+5@60>152@65',
        '140>140+5:bnd_only@60>140@65': '152>152+5:bnd_only@60>152@65',
        '140@40>140+10@65': '152@40>152+10@65',     # explicit start age kept
        '140': '152',
    }
    for spec, want in cases.items():
        got = glide.same_shape(spec, '152')
        assert got == want, (spec, got)
        # and it still means the same thing it says
        assert glide.parse_schedule(got, 37) == glide.parse_schedule(want, 37)

    # a schedule that MOVES between portfolios has no single base to swap
    try:
        glide.same_shape('140>166+5@65', '152')
    except ValueError:
        pass
    else:
        raise AssertionError('a mixed schedule was given a shape')


def test_one_custom_schedule_can_be_applied_to_every_portfolio():
    """The same 5% 60->65 sleeve on three portfolios took three passes through
    the custom-schedule prompts. Build it once and the others are offered."""
    P = _P()
    ask3 = lambda: runner.ask_custom_schedules(P, 37, ['140', '152', '165'])
    built = ['140', '140+5 at 60', '140 at 65', '']

    got, out = _drive(ask3, built + ['y', 'n'])
    assert got == ['140', '152', '165', '140>140+5@60>140@65',
                   '152>152+5@60>152@65', '165>165+5@60>165@65'], got
    assert 'Same shape for 152, 165?' in out, out

    # declining keeps just the one
    got, _ = _drive(ask3, built + ['n', 'n'])
    assert got == ['140', '152', '165', '140>140+5@60>140@65'], got

    # offered to the OTHER portfolios, whichever one the shape was built on
    got, out = _drive(ask3, ['152>152+5@60>152@65', 'y', 'n'])
    assert 'Same shape for 140, 165?' in out, out
    assert got == ['140', '152', '165', '152>152+5@60>152@65',
                   '140>140+5@60>140@65', '165>165+5@60>165@65'], got

    # a portfolio that already has this shape is not added twice
    got, _ = _drive(ask3, ['140>140+5@65', 'y', 'y', '152>152+5@65', 'n'])
    assert got.count('152>152+5@65') == 1, got

    # one portfolio: nothing to offer, so no extra question (the script would
    # run out and raise if it were asked)
    got, _ = _drive(lambda: runner.ask_custom_schedules(P, 37, ['140']),
                    built + ['n'])
    assert got == ['140', '140>140+5@60>140@65'], got


def test_the_tracker_workbook_reads_into_percent_holdings():
    """update_holdings_from_xlsx writes the holdings row (141), and its
    reader had never run outside the real workbook. Labels match case- and
    space-insensitively, a '(proxy)' column is skipped, weights become
    percents, a fund in two accounts is added, and a missing label or sheet
    is refused by name."""
    import openpyxl
    try:
        import update_holdings_from_xlsx as ux
    except ImportError:
        return          # personal to the author; the public snapshot leaves it out
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'tracker.xlsx'
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = ux.SHEET
        ws['A2'], ws['A4'] = '  ticker ', 'PLACED (%)'
        for col, (t, w) in enumerate((('VOO', 0.31), ('(proxy)', 0.5), ('AVUV', 0.4),
                                      ('VOO', 0.09), (None, 0.7), ('BND', None)),
                                     start=ux.FIRST_DATA_COL):
            ws.cell(2, col, t)
            ws.cell(4, col, w)
        wb.save(path)
        got = ux.read_holdings_from_xlsx(path)
        assert got.keys() == {'VOO', 'AVUV', 'BND'}, got
        assert abs(got['VOO'] - 40.0) < 1e-9 and abs(got['AVUV'] - 40.0) < 1e-9, got
        assert got['BND'] == 0.0, got
        for kw, want in (({'sheet': 'Nope'}, 'Nope'),):
            try:
                ux.read_holdings_from_xlsx(path, **kw)
                raise AssertionError(kw)
            except ValueError as e:
                assert want in str(e), e
        ws['A4'] = 'Target (%)'
        wb.save(path)
        try:
            ux.read_holdings_from_xlsx(path)
            raise AssertionError('a missing Placed row must be refused')
        except ValueError as e:
            assert ux.VALUE_ROW_LABEL in str(e), e


def test_a_pasted_fund_held_twice_is_added_not_overwritten():
    """The same fund in two accounts is two entries in one paste. The parser
    kept only the last, which the 100% check catches unless the dropped
    entry is under its 1pp tolerance; the xlsx reader already adds them."""
    import update_holdings as uh
    got = uh.parse_paste('31 0.8 68.2 // VOO VOO AVUV')
    assert got == {'VOO': 31.8, 'AVUV': 68.2}, got
    for bad in ('31 x // VOO AVUV', '31 // VOO AVUV', '31 69 VOO AVUV'):
        try:
            uh.parse_paste(bad)
            raise AssertionError(f'{bad!r} must be refused')
        except ValueError:
            pass


def test_the_template_tools_keep_the_csv_byte_for_byte_except_what_they_change():
    """csv.writer ends lines with CRLF by default, and the committed CSV is LF,
    so annotate_templates, update_holdings and new_template each rewrote every
    line of portfolio_templates.csv — the file's history flips line endings
    three times. annotate on an already-annotated file must change nothing;
    update_row must change exactly its row; a scaffold is LF too."""
    import shutil
    import annotate_templates as at
    import update_holdings as uh
    import new_template
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'portfolio_templates.csv'
        shutil.copy(ROOT / 'portfolios' / 'portfolio_templates.csv', path)
        original = path.read_bytes()
        assert b'\r' not in original, 'the committed CSV is LF'
        with contextlib.redirect_stdout(io.StringIO()):
            at.annotate(path)
        assert path.read_bytes() == original, 'annotate is not a no-op'
        with contextlib.redirect_stdout(io.StringIO()):
            uh.update_row(path, '141', {'VOO': 60.0, 'AVUV': 40.0})
        before, after = original.split(b'\n'), path.read_bytes().split(b'\n')
        assert len(before) == len(after), (len(before), len(after))
        assert sum(a != b for a, b in zip(before, after)) == 1
        fresh = Path(d) / 'fresh.csv'
        with contextlib.redirect_stdout(io.StringIO()):
            new_template.scaffold(fresh)
        assert b'\r' not in fresh.read_bytes()


def test_the_glide_chart_draws_each_step_at_its_equity_and_its_ages():
    """The custom-schedule prompts show this chart and nothing checked it. A
    10% bond sleeve from 65 on a 40-to-90 plan: 100% then 90% equity, each bar
    starting where its step starts and as long as its span, ages on the axis."""
    P = _P()
    chart = glide.format_chart(glide.parse_schedule('141>141+10@65', 40), P, 90,
                               width=50)
    lines = chart.splitlines()
    first = next(l for l in lines if l.rstrip().endswith(' 141'))
    second = next(l for l in lines if l.rstrip().endswith(' 141+10'))
    assert first.split()[0] == '100' and second.split()[0] == '90', chart
    # 25 of 50 years each: half the width, the second starting where the first ends
    assert first.count('█') == second.count('█') == 25, chart
    assert second.index('█') == first.index('█') + 25, chart
    assert lines[-1].split() == ['40', '50', '60', '70', '80', '90'], chart


def test_update_holdings_writes_a_row_the_template_gate_accepts():
    """Thirds normalise to 0.333333 each at six decimals, a row summing to
    0.999999 — which check_templates fails at 1e-9. The helper that exists to
    keep the CSV valid could leave it invalid. Negative holdings were accepted
    as long as the total came to 100."""
    import shutil
    import update_holdings as uh
    import check_templates
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'portfolio_templates.csv'
        shutil.copy(ROOT / 'portfolios' / 'portfolio_templates.csv', path)
        before = path.read_text().splitlines()
        with contextlib.redirect_stdout(io.StringIO()):
            uh.update_row(path, '141', {'VOO': 33.33, 'AVUV': 33.33, 'VEA': 33.33})
        after = path.read_text().splitlines()
        changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
        assert len(before) == len(after) and len(changed) == 1, changed
        saved = check_templates.CSV_FILE
        try:
            check_templates.CSV_FILE = path
            with contextlib.redirect_stdout(io.StringIO()) as out:
                rc = check_templates.main()
        finally:
            check_templates.CSV_FILE = saved
        assert rc == 0, out.getvalue()[-600:]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                uh.update_row(path, '141', {'VOO': 110.0, 'AVUV': -10.0})
        except ValueError:
            pass
        else:
            raise AssertionError('a negative holding was written')


def test_regression_update_refuses_to_bless_a_failing_engine():
    """--update wrote the new baseline even when determinism or the analytic
    anchor had just failed, so a broken engine could become the golden master.
    In a child process: regression_check installs its synthetic universe into
    `engine` and does not restore it."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / 'baseline.json'
        code = f"""
import sys; sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / 'tools')!r}]
from pathlib import Path
from montecarlo import engine; import regression_check as rc
rc.BASELINE = Path({str(out)!r})
real, calls = engine.run_single_simulation, []
def flaky(args):
    o = real(args); calls.append(1)
    return (o[0], o[1] * 1.01) + tuple(o[2:]) if len(calls) == 2 else o
engine.run_single_simulation = flaky
print('RC', rc.main(['--update']))
"""
        r = subprocess.run([sys.executable, '-c', code], capture_output=True,
                           text=True, cwd=ROOT)
        assert 'RC 1' in r.stdout, r.stdout[-400:] + r.stderr[-400:]
        assert not out.exists(), 'a failing engine was blessed'


def test_regression_check_fails_on_drift_and_on_anything_the_baseline_has_that_the_run_lacks():
    """The comparison walked only the CURRENT run's keys, so a statistic, case or
    metric dropped from the run while still in the baseline passed: the gate
    held less coverage than it had blessed. The run's three sections are stubbed
    to the baseline's own contents (so the control must pass), then one thing is
    changed at a time. Child process, as above; the baseline is a temp copy."""
    import shutil
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / 'baseline.json'
        shutil.copy(ROOT / 'tools' / 'regression_baseline.json', tmp)
        code = f"""
import sys; sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / 'tools')!r}]
import copy, io, json, contextlib
from pathlib import Path
import regression_check as rc
rc.BASELINE = Path({str(tmp)!r})
saved = json.loads(rc.BASELINE.read_text())
dcase = next(iter(saved['decumulation']))
fcase = next(iter(saved['features']))
stat = next(iter(saved['stats']))
def perturb(c):  c['stats'][stat] *= 1.5
def lose_stat(c):  del c['stats'][stat]
def lose_dcase(c):  del c['decumulation'][dcase]
def lose_dmetric(c):  del c['decumulation'][dcase][next(iter(c['decumulation'][dcase]))]
def lose_fcase(c):  del c['features'][fcase]
def lose_fmetric(c):  del c['features'][fcase][next(iter(c['features'][fcase]))]
out = {{}}
for name, change in [('control', lambda c: None), ('perturb', perturb),
                     ('lose_stat', lose_stat), ('lose_dcase', lose_dcase),
                     ('lose_dmetric', lose_dmetric), ('lose_fcase', lose_fcase),
                     ('lose_fmetric', lose_fmetric)]:
    cur = copy.deepcopy(saved); change(cur)
    rc._run_synth = lambda cur=cur: cur['stats']
    rc._run_decum = lambda cur=cur: cur['decumulation']
    rc._run_features = lambda cur=cur: cur['features']
    with contextlib.redirect_stdout(io.StringIO()):
        out[name] = rc.main([])
rc.BASELINE = Path({str(Path(d) / 'absent.json')!r})
with contextlib.redirect_stdout(io.StringIO()):
    out['no_baseline'] = rc.main([])
print('RC', json.dumps(out))
"""
        r = subprocess.run([sys.executable, '-c', code], capture_output=True,
                           text=True, cwd=ROOT)
        line = [l for l in r.stdout.splitlines() if l.startswith('RC ')]
        assert line, r.stdout[-400:] + r.stderr[-600:]
        got = json.loads(line[0][3:])
        assert got['control'] == 0, got
        assert all(v == 1 for k, v in got.items() if k != 'control'), got


def test_mutation_sweep_fails_on_a_stale_target_and_counts_a_crashed_suite_as_noticed():
    """`mutate()` printed NOT FOUND for a renamed target and went on to say every
    target was caught, so a function stopped being mutation-tested without any
    sign; a mutant that broke the suite at import printed no [FAIL] lines and
    read as a hole. ROOT and SUITE are pointed at a tempdir: no real source is
    ever rewritten."""
    import contextlib
    import io
    import vacuity_sweep as vs
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        src = 'def f():\n    return 1\n'
        (root / 'a.py').write_text(src)
        suite = root / 'suite.py'
        old = (vs.ROOT, vs.SUITE)
        vs.ROOT, vs.SUITE = root, suite
        try:
            def sweep(suite_code, target=('a.py', 'f')):
                suite.write_text(suite_code)
                with contextlib.redirect_stdout(io.StringIO()):
                    holes = vs.mutate([target])
                assert (root / 'a.py').read_text() == src, 'source not restored'
                return holes

            assert sweep('print("[FAIL] test_x")\nraise SystemExit(1)\n') == []
            assert sweep('raise SystemExit(1)\n') == [], 'a crashed suite is noticed'
            assert sweep('pass\n') == [('a.py', 'f')], 'a silent pass is a hole'
            assert sweep('raise SystemExit(1)\n', ('a.py', 'renamed_away')), \
                'a stale target must fail the sweep'
        finally:
            vs.ROOT, vs.SUITE = old


def test_the_portfolio_loader_rejects_a_duplicate_id_and_a_short_position():
    """A copy-pasted row with an existing id silently REPLACED the earlier
    portfolio (last wins), and VOO 1.5 / BND -0.5 loaded as a valid portfolio
    because only the sum was checked. check_templates catches the negative, but
    only when someone runs it; the loader is on every run's path."""
    from montecarlo.loaders import MODEL_TICKERS
    header = ['portfolio_id', 'name'] + list(MODEL_TICKERS)

    def row(pid, name, **w):
        return [pid, name] + [str(w.get(t, 0)) for t in MODEL_TICKERS]

    def load(*rows):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 't.csv'
            path.write_text('\n'.join(','.join(r) for r in (header,) + rows) + '\n')
            return load_portfolios_from_csv(str(path), verbose=False)

    assert set(load(row('1', 'a', VOO=1.0), row('2', 'b', VOO=1.0))) == {'1', '2'}
    for rows in ((row('1', 'a', VOO=1.0), row('1', 'b', BND=1.0)),
                 (row('1', 'a', VOO=1.5, BND=-0.5),)):
        try:
            load(*rows)
        except ValueError:
            continue
        raise AssertionError(f'loaded: {rows}')

    # One row per remaining branch. Each rejection must say WHY, not just raise.
    def rejected(says, *rows, header_row=header):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 't.csv'
            path.write_text('\n'.join(','.join(r) for r in (header_row,) + rows) + '\n')
            try:
                load_portfolios_from_csv(str(path), verbose=False)
            except ValueError as exc:
                assert says in str(exc), (says, str(exc))
                return
        raise AssertionError(f'loaded, expected {says!r}: {rows}')

    rejected('Allocations sum to', row('1', 'a', VOO=0.9))
    rejected('Allocations sum to', row('1', 'a', VOO=1.02))
    bad_weight = row('1', 'a', VOO=1.0)
    bad_weight[2 + MODEL_TICKERS.index('BND')] = 'abc'
    rejected("Invalid allocation for BND: 'abc'", bad_weight)
    rejected('Missing name', row('1', '', VOO=1.0))
    rejected('No portfolios loaded', row('#', 'c', VOO=1.0))      # header-only after skips

    # Inside the +-0.01 tolerance, with a '00' cell, blank cells, and '#' and
    # blank-id rows skipped, it loads and the '00' reads as zero.
    zero = row('3', 'c', VOO=1.005)
    zero[2 + MODEL_TICKERS.index('BND')] = '00'
    zero[2 + MODEL_TICKERS.index('AVUV')] = ''
    got = load(row('#', 'section', VOO=7.0), row('', 'blank id', VOO=7.0), zero)
    assert list(got) == ['3'] and got['3']['allocation']['BND'] == 0.0, got
    assert abs(got['3']['allocation']['VOO'] - 1.005) < 1e-12

    with tempfile.TemporaryDirectory() as d:
        try:
            load_portfolios_from_csv('absent.csv', base_dir=d, verbose=False)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('a missing file loaded')
        (Path(d) / 'empty.csv').write_text('')
        try:
            load_portfolios_from_csv('empty.csv', base_dir=d, verbose=False)
        except ValueError as exc:
            assert 'empty or malformed' in str(exc), exc
        else:
            raise AssertionError('an empty file loaded')

    # The default is verbose, and its summary line sorts ids numerically: a
    # non-numeric id loaded silently and then crashed in the print.
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 't.csv'
        path.write_text('\n'.join(','.join(r) for r in (
            header, row('10', 'a', VOO=1.0), row('9', 'b', VOO=1.0),
            row('A1', 'c', VOO=1.0))) + '\n')
        with contextlib.redirect_stdout(io.StringIO()) as out:
            load_portfolios_from_csv(str(path), verbose=True)
    assert 'Portfolio IDs: 9, 10, A1' in out.getvalue(), out.getvalue()


def test_section_headers_are_idempotent_and_keep_your_notes():
    """add_headers strips its own headers before re-inserting them, and its
    docstring promises to leave other '#' notes alone — but it stripped ANY line
    starting '# ---', so a note like '# --- watch this one' vanished."""
    import shutil
    import add_section_headers as ash
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'portfolio_templates.csv'
        shutil.copy(ROOT / 'portfolios' / 'portfolio_templates.csv', path)
        lines = path.read_text().splitlines(keepends=True)
        lines.insert(4, '# --- watch this one\n')
        path.write_text(''.join(lines))
        with contextlib.redirect_stdout(io.StringIO()):
            ash.add_headers(path)
            once = path.read_text()
            ash.add_headers(path)
        assert path.read_text() == once, 'a second run changed the file'
        assert '# --- watch this one' in once, 'your note was deleted'


def test_section_headers_are_written_where_the_tiers_and_tilts_change():
    """The idempotence test passes if the tool does nothing. This one pins what
    it writes on a known fixture: a tier header per hundreds digit, a tilt
    sub-header only in a tier that holds more than one tilt, pid 100 labelled as
    the status quo, the file's own newline, and a spreadsheet's trailing commas
    on old headers."""
    import add_section_headers as ash
    rows = ['portfolio_id,name,VOO,tilt_family', '100,a,1.0,momentum',
            '102,b,1.0,value', '200,c,1.0,core', '201,d,1.0,core',
            '300,e,1.0,core']
    tier = lambda n: f'# ===== {ash.TIER_LABELS[n]} ====='
    want = [rows[0], tier(1), '# --- momentum (base) ---', rows[1],
            '# --- value tilt ---', rows[2], tier(2), rows[3], rows[4],
            tier(3), rows[5]]

    def run(text, newline='\n'):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 't.csv'
            with open(path, 'w', newline='') as f:
                f.write(text)
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                ash.add_headers(path)
            return open(path, newline='').read(), buf.getvalue()

    got, said = run('\n'.join(rows) + '\n')
    assert got == '\n'.join(want) + '\n', got
    assert 'Inserted 3 tier headers + 2 tilt sub-headers' in said, said

    crlf, _ = run('\r\n'.join(rows) + '\r\n')
    assert crlf == '\r\n'.join(want) + '\r\n', repr(crlf)

    # headers a spreadsheet re-saved with trailing commas are ours: replaced, not kept
    stale = [rows[0], tier(1) + ',,', '# --- stale ---,,', rows[1], rows[2],
             tier(2) + ',', rows[3], rows[4], rows[5]]
    got, _ = run('\n'.join(stale) + '\n')
    assert got == '\n'.join(want) + '\n', got

    for text, why in (('', 'is empty'), (tier(1) + '\n', 'is empty'),
                      ('portfolio_id,name\n100,a\n', 'run annotate_templates.py')):
        try:
            run(text)
        except ValueError as exc:
            assert why in str(exc), (text, str(exc))
        else:
            raise AssertionError(f'{text!r} was accepted')


def test_the_drift_baseline_says_created_the_first_time():
    """It asked BASELINE.exists() after writing it, so a first run always
    claimed to have 're-blessed' a baseline that did not exist."""
    import check_parameter_drift as cpd
    saved = cpd.BASELINE
    with tempfile.TemporaryDirectory() as d:
        try:
            cpd.BASELINE = Path(d) / 'baseline.json'
            with contextlib.redirect_stdout(io.StringIO()) as first:
                cpd.main([])
            with contextlib.redirect_stdout(io.StringIO()) as again:
                cpd.main(['--update'])
        finally:
            cpd.BASELINE = saved
    assert 'baseline created' in first.getvalue(), first.getvalue()
    assert 'baseline re-blessed' in again.getvalue(), again.getvalue()


def test_a_second_scaffold_does_not_lose_the_current_portfolios():
    """It backed up to .bak only when no .bak existed, then ALWAYS overwrote
    the live CSV — so a second run destroyed whatever had been entered since
    the first, with no copy anywhere."""
    import new_template
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'portfolio_templates.csv'
        path.write_text('original\n')
        with contextlib.redirect_stdout(io.StringIO()):
            new_template.scaffold(path)
            path.write_text('entered since\n')
            new_template.scaffold(path)
        kept = {q.read_text() for q in Path(d).glob('*.bak')}
        assert kept == {'original\n', 'entered since\n'}, kept


def test_pooled_paths_are_the_serial_paths():
    """Every production solve runs through the Pool, and no check ever did:
    every test passed parallel=False or stayed under PARALLEL_THRESHOLD. Under
    `spawn` a worker re-imports everything, so a setting that lives only in the
    parent, or a plan that does not survive pickling, would make the pooled
    paths differ from the serial ones — silently, in every real run."""
    import pickle
    P = _P()
    cfg = runner.default_config()
    cfg['withdrawal'] = {'start_age': cfg['retirement_age'], 'policy': 'guyton_klinger',
                         'sourcing': 'excess', 'target_real_annual': 110_000.0,
                         'floor': 0.85}
    # Short enough for the gate: retire two years in, glide in and out of bonds.
    cfg = dict(cfg, age=58, retirement_age=60, time_horizon_years=10)
    cfg['withdrawal']['start_age'] = 60
    plan = glide.build_plan(glide.parse_schedule('168>168+10@60>168@65', 58), P)
    n = solver.PARALLEL_THRESHOLD
    try:
        pooled = solver._paths(plan, cfg, n, True)
    finally:
        solver.shutdown_pool()
    serial = solver._paths(plan, cfg, n, False)
    assert len(pooled) == len(serial) == n
    same = sum(pickle.dumps(a) == pickle.dumps(b) for a, b in zip(pooled, serial))
    assert same == n, f'{n - same} of {n} pooled paths differ from serial'


def test_care_costs_drain_the_book_but_are_not_spending():
    """The engine's care-cost rules were pinned only by grepping its source.
    Run them: zero return, zero volatility, so every dollar is accounted for.
    Care must leave realised spending untouched and take exactly its total off
    the balance; a year the balance cannot cover shorts the LIFESTYLE first."""
    with _engine_state():
        _zero_vol_engine(0.0)
        pa.EXPECTED_RETURNS = {'Z': 0.0}
        wd = {'start_age': 65, 'policy': 'constant_real', 'sourcing': 'pro_rata',
              'target_real_annual': 40_000.0}
        care = dict(wd, ltc={'annual_amount': 100_000.0, 'start_age': 80, 'years': 3})

        def run(w, initial):
            return pa.run_single_simulation(({'Z': 1.0}, _wd_cfg(30, w, initial=initial), 0))

        plain, cared = run(wd, 2_000_000.0), run(care, 2_000_000.0)
        assert cared[4]['spending'] == plain[4]['spending'], 'care counted as spending'
        assert abs((plain[0] - cared[0]) - 300_000.0) < 1e-6, plain[0] - cared[0]

        # 15 years x $40k leaves $50k at 80, against $140,000 of need: the care
        # bill takes it all and the lifestyle gets nothing.
        short = run(care, 650_000.0)[4]
        at80 = short['spending'][80 - 65]
        assert short['depletion_age'] == 80, short['depletion_age']
        assert abs(at80) < 1e-6, f'lifestyle got {at80:,.0f} in an underfunded care year'


def _gate_output(mod, attrs):
    """Run a gate's main() with its input constants pointed at `path`."""
    saved = {a: getattr(mod, a) for a in attrs}
    try:
        for a, v in attrs.items():
            setattr(mod, a, v)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = mod.main()
    finally:
        for a, v in saved.items():
            setattr(mod, a, v)
    return code, out.getvalue()


def test_the_parameter_gate_fails_on_each_kind_of_bad_regen():
    """validate_parameters had only ever run against the live output/, so
    nothing showed its checks CAN fail. One corrupted copy per check: each
    must fail, naming its own check, and the clean copy must pass."""
    import shutil
    import validate_parameters as vp
    src = ROOT / 'output'
    cases = {
        'base_volatility in': lambda d, o: d['VOO'].update(base_volatility=1.10),
        'regime vols ordered': lambda d, o: d['VOO']['volatility_regimes'].update(
            low=0.40, high=0.05),
        'regime_transitions are valid stochastic rows':
            lambda d, o: d['VOO']['regime_transitions']['low'].update(low=0.9, medium=0.3),
        'history >= 60 months': lambda d, o: d['VOO'].update(history_months=10),
        "correlation 'high' is a valid": lambda d, o: np.save(
            o / f'correlation_high_{len(loaders.MODEL_TICKERS)}.npy',
            np.full((len(loaders.MODEL_TICKERS),) * 2, 1.5)),
        'expected_return in': lambda d, o: d['VOO'].update(expected_return=0.40),
        'parameter_std in': lambda d, o: d['VOO'].update(parameter_std=0.20),
        # The 2026-09-10 defect: a proxy table rotated by one.
        'every contributing proxy tracks its fund': lambda d, o: d['metadata'][
            'proxy_scores']['AVDE'].update(corr=0.70),
        'no proxy contributes more months than its history': lambda d, o: d[
            'metadata']['proxy_scores']['AVDE'].update(contributed_months=9999),
    }
    def gate(corrupt):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            for f in src.glob('correlation_*.npy'):
                shutil.copy(f, out / f.name)
            data = json.loads((src / vp.JSON.name).read_text())
            corrupt(data, out)
            (out / vp.JSON.name).write_text(json.dumps(data))
            return _gate_output(vp, {'OUTPUT': out, 'JSON': out / vp.JSON.name})

    for label, corrupt in [(None, lambda d, o: None)] + list(cases.items()):
        code, text = gate(corrupt)
        fails = [ln for ln in text.splitlines() if '[FAIL]' in ln]
        if label is None:
            assert code == 0 and not fails, fails
        else:
            assert code and any(label in ln for ln in fails), (label, fails)

    # Whole check groups, one corrupted copy each. These share a label with
    # nothing else, but the proxy ones share one among themselves, so each also
    # names the reason its own branch gives (the line printed under the FAIL).
    def drop(d, o):
        d['VOO'].pop('regime_transitions_by_market')

    def unbalance(d, o):
        d['VOO']['regime_transitions_by_market']['high']['medium'].update(low=0.9)

    own = [
        ('all model tickers present', 'missing: [', lambda d, o: d.pop('VTV')),
        ('every ticker has the full key set', 'history_end',
         lambda d, o: d['VOO'].pop('history_end')),
        ('no NaN/Inf/missing scalar parameters', 'VOO.expected_return=nan',
         lambda d, o: d['VOO'].update(expected_return=float('nan'))),
        ("regime vols plausible given each asset's own", 'VOO.low=',
         lambda d, o: d['VOO']['volatility_regimes'].update(low=0.001)),
        ('all bond vols < every equity vol', 'max bond sigma',
         lambda d, o: d['BND'].update(base_volatility=0.50)),
        ('metadata.ticker_count ==', 'got 5',
         lambda d, o: d['metadata'].update(ticker_count=5)),
        ('ISO last_updated present', 'last_updated=31/12/2026',
         lambda d, o: d['metadata'].update(last_updated='31/12/2026')),
        ('every ticker carries regime_transitions_by_market', 'VOO', drop),
        ('market-conditioned transitions are valid stochastic rows',
         'VOO/high/medium does not sum to 1', unbalance),
        ('market_regime transitions are valid stochastic rows',
         'medium row does not sum to 1',
         lambda d, o: d['metadata']['market_regime']['transitions']['medium'].update(
             low=0.5)),
        ('market regimes separate realised volatility', 'low 9.0% medium 10.0% high 11.0%',
         lambda d, o: d['metadata']['market_regime']['realised_vol'].update(
             low=0.09, medium=0.10, high=0.11)),
        # The stored score's own key names (tracking_error, contributed_months),
        # which proxy_score.verdict() does not read (te, contrib).
        ('every contributing proxy tracks its fund', 'JPST proxied by',
         lambda d, o: d['metadata']['proxy_scores']['JPST'].update(tracking_error=0.05)),
        ('every contributing proxy tracks its fund', 'AVDE proxied by',
         lambda d, o: d['metadata']['proxy_scores']['AVDE'].update(beta=1.5)),
    ]
    for label, detail, corrupt in own:
        code, text = gate(corrupt)
        lines = text.splitlines()
        at = [i for i, ln in enumerate(lines) if '[FAIL]' in ln and label in ln]
        assert code and at, (label, [ln for ln in lines if '[FAIL]' in ln])
        assert any(detail in ln for ln in lines[at[0]:at[0] + 2]), (label, detail, lines[at[0]:at[0] + 2])
    # The cash-like branch judges tracking error, and says so.
    code, text = gate(own[-2][2])
    assert 'cash-like' in text and 'TE 5.00%' in text, text[-500:]
    code, text = gate(own[-1][2])
    assert 'beta 1.50' in text, text[-500:]

    # A score marked thin is skipped whatever it says: it never failed the gate.
    code, text = gate(lambda d, o: d['metadata']['proxy_scores']['AVDE'].update(
        thin=True, corr=0.10, beta=3.0))
    assert code == 0 and '[FAIL]' not in text and 'too little overlap to score: AVDE' in text, \
        text[-500:]

    # A badly scoring proxy that reaches <= 3 months is a note, not a failure.
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        for f in src.glob('correlation_*.npy'):
            shutil.copy(f, out / f.name)
        data = json.loads((src / vp.JSON.name).read_text())
        data['metadata']['proxy_scores']['AVDE'].update(corr=0.70, contributed_months=2)
        (out / vp.JSON.name).write_text(json.dumps(data))
        code, text = _gate_output(vp, {'OUTPUT': out, 'JSON': out / vp.JSON.name})
    assert code == 0 and 'vestigial' in text and 'AVDE' in text, text[-600:]


def test_proxy_score_verdicts_and_the_gate_share_one_threshold_set():
    """proxy_score judges a proxy when it is scored and validate_parameters
    judges the stored score; the thresholds were copied between them. Now
    imported, and verdict() is pinned at each edge it decides on."""
    import proxy_score as ps
    import validate_parameters as vp
    assert (vp.PROXY_CORR_FLOOR, vp.PROXY_BETA_BAND, vp.PROXY_CASHLIKE_VOL,
            vp.PROXY_CASHLIKE_TE, vp.PROXY_VESTIGIAL_MONTHS) == (
        ps.CORR_FLOOR, ps.BETA_BAND, ps.CASHLIKE_VOL, ps.CASHLIKE_TE,
        ps.VESTIGIAL_MONTHS)
    good = {'fund_vol': 0.15, 'te': 0.03, 'corr': 0.97, 'beta': 1.0, 'contrib': 120}
    v = lambda **k: ps.verdict(dict(good, **k))
    assert v() == 'ok'
    assert v(corr=ps.CORR_FLOOR - 0.01) == 'WRONG ASSET'
    assert v(beta=ps.BETA_BAND[1] + 0.01) == 'beta off'
    assert v(corr=0.5, contrib=ps.VESTIGIAL_MONTHS) == 'vestigial'
    # Near cash, correlation is not the test; tracking error is.
    assert v(fund_vol=ps.CASHLIKE_VOL / 2, corr=0.5, te=ps.CASHLIKE_TE / 2) == 'ok'
    assert v(fund_vol=ps.CASHLIKE_VOL / 2, te=ps.CASHLIKE_TE * 2) == \
        'TE too wide for a cash-like fund'
    assert ps.verdict({'thin': True}) == 'THIN'


def test_the_template_gate_fails_on_each_kind_of_bad_row():
    """check_templates had never been fed a bad CSV. One defect per row; each
    must be named, and the comment and section rows must be skipped."""
    import csv as _csv, shutil
    import check_templates as ct
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'portfolio_templates.csv'
        shutil.copy(ROOT / 'portfolios' / 'portfolio_templates.csv', path)
        code, text = _gate_output(ct, {'CSV_FILE': path})
        assert code == 0, text[-400:]

        rows = list(_csv.reader(open(path, newline='')))
        col = {n: i for i, n in enumerate(rows[0])}
        data = [r for r in rows[1:] if r and r[0].strip() and not r[0].startswith('#')]
        bad = {data[0][0]: ('VOO', '0.9'),            # sum off
               data[1][0]: ('AVUV', '-0.1'),          # negative (set, not added)
               data[2][0]: ('BND', 'abc'),            # non-numeric
               data[3][0]: ('bond_pct', '0.77'),      # stale derived column
               data[4][0]: ('tilt_family', 'zzz')}    # stale string column
        for r in rows[1:]:
            if r and r[0] in bad:
                c, v = bad[r[0]]
                if c == 'VOO':
                    r[col[c]] = str(float(r[col[c]] or 0) + float(v))
                else:
                    r[col[c]] = v
        with open(path, 'w', newline='') as f:
            _csv.writer(f).writerows(rows)
        code, text = _gate_output(ct, {'CSV_FILE': path})
    assert code, text[-400:]
    for pid, want in ((data[0][0], 'weights sum to 1.0'),
                      (data[1][0], 'no negative weights'),
                      (data[2][0], 'non-numeric weight in BND'),
                      (data[3][0], "derived 'bond_pct' stale"),
                      (data[4][0], "derived 'tilt_family' stale")):
        assert f'[{pid}] {want}' in text, (pid, want, text[-600:])

    # A header missing a ticker column stops the lint before any row.
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'portfolio_templates.csv'
        lines = (ROOT / 'portfolios' / 'portfolio_templates.csv').read_text().splitlines(True)
        lines[0] = lines[0].replace(',AVUV,', ',AVUX,', 1)
        path.write_text(''.join(lines))
        code, text = _gate_output(ct, {'CSV_FILE': path})
    assert code and 'AVUV' in text and 'weights sum' not in text, text[-600:]


def test_archived_portfolios_are_listed_named_by_parse_spec_and_linted():
    """portfolio_archive.csv had no reader under test and no lint: archived_portfolio_ids
    is only how parse_spec says WHY an id is unknown, and a bad archived row
    was first read the day someone moved it back."""
    import shutil
    import check_templates as ct
    from montecarlo.loaders import archived_portfolio_ids
    arch = (ROOT / 'portfolios' / 'portfolio_archive.csv').read_text().splitlines(True)

    with tempfile.TemporaryDirectory() as d:
        assert archived_portfolio_ids(d) == set(), 'a missing archive is an empty set'
        from montecarlo.loaders import ARCHIVE_CSV
        (Path(d) / ARCHIVE_CSV).parent.mkdir(parents=True)
        (Path(d) / ARCHIVE_CSV).write_text(
            'portfolio_id,name\n9,nine\n# --- a section note\n,blank id\n8,eight\n')
        assert archived_portfolio_ids(d) == {'9', '8'}, archived_portfolio_ids(d)

    # parse_spec names the file for an archived id, and only for one.
    P = _P()
    archived = sorted(archived_portfolio_ids() - set(P))
    assert archived, 'the real archive holds no id that is absent from the templates'
    try:
        runner.parse_spec(json.dumps({'portfolios': [archived[0], '99999']}), P)
    except ValueError as exc:
        msg = str(exc)
    else:
        raise AssertionError('an archived id was accepted')
    assert f'{archived[0]} archived: move the row back from portfolios/portfolio_archive.csv' in msg, msg
    assert '99999 archived' not in msg and '99999' in msg, msg

    # The lint: a clean copy passes; a bad sum, a negative weight and an id that
    # is also live each fail under their own name.
    def lint(edit):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'portfolio_archive.csv'
            path.write_text(''.join(edit(list(arch))))
            return _gate_output(ct, {'ARCHIVE_FILE': path})

    code, text = lint(lambda lines: lines)
    assert code == 0 and 'archived portfolios: weights sum' in text, text
    first = arch[1].split(',')[0]
    other = arch[2].split(',')[0]

    def edit_cell(lines, i, ticker, value):
        head = lines[0].rstrip('\n').split(',')
        cells = lines[i].rstrip('\n').split(',')
        cells[head.index(ticker)] = value
        lines[i] = ','.join(cells) + '\n'
        return lines

    code, text = lint(lambda l: edit_cell(l, 1, 'VOO', '0.5'))
    assert code and f'[{first}] archive: weights sum to 1.0' in text, text
    code, text = lint(lambda l: edit_cell(l, 2, 'VOO', '-0.5'))
    assert code and f'[{other}] archive: no negative weights' in text, text
    live = next(r.split(',')[0] for r in (ROOT / 'portfolios' / 'portfolio_templates.csv').read_text()
                .splitlines()[1:] if r and not r.startswith('#'))
    code, text = lint(lambda l: edit_cell(l, 1, 'portfolio_id', live))
    assert code and f'[{live}] id is in both' in text, text


def test_the_parameter_loader_refuses_or_warns_on_a_bad_file():
    """load_parameters_from_json is the guard between a bad quarterly regen and
    a plausible-looking run, and had only ever loaded the real file. A tiny
    two-fund fixture: the happy path (with the (ticker, market) keys), then one
    defect per raise or warning."""
    import copy, datetime, warnings
    from montecarlo.loaders import load_parameters_from_json as load
    row = {'low': 0.8, 'medium': 0.15, 'high': 0.05}
    fund = {'expected_return': 0.07, 'base_volatility': 0.15, 'parameter_std': 0.01,
            'volatility_regimes': {'low': 0.1, 'medium': 0.15, 'high': 0.25},
            'regime_transitions': {r: dict(row) for r in ('low', 'medium', 'high')},
            'regime_transitions_by_market': {'high': {r: dict(row) for r in
                                                      ('low', 'medium', 'high')}}}
    good = {'metadata': {'version': 'test',
                         'last_updated': datetime.date.today().isoformat()},
            'T1': copy.deepcopy(fund), 'T2': copy.deepcopy(fund)}

    def attempt(data, matrix=np.eye(2), raw=None):
        with tempfile.TemporaryDirectory() as d, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            out = Path(d)
            (out / 'p.json').write_text(raw if raw is not None else json.dumps(data))
            for kind in ('baseline', 'low', 'medium', 'high'):
                np.save(out / f'correlation_{kind}_2.npy', matrix)
            try:
                got = load('p.json', base_dir=out, verbose=False)
            except (ValueError, FileNotFoundError) as e:
                return type(e).__name__, [str(x.message) for x in w]
            return got, [str(x.message) for x in w]

    got, warned = attempt(good)
    assert got['ticker_list'] == ['T1', 'T2'] and not warned, warned
    assert ('T1', 'high') in got['REGIME_TRANSITIONS']

    def broken(edit):
        d = copy.deepcopy(good)
        edit(d)
        return d

    assert attempt(good, raw='{ not json')[0] == 'ValueError'
    assert attempt(broken(lambda d: d['T1'].pop('expected_return')))[0] == 'ValueError'
    assert attempt(broken(lambda d: d['T2']['regime_transitions'].pop('high')))[0] == 'ValueError'
    assert attempt(good, matrix=np.eye(3))[0] == 'ValueError', 'shape mismatch'
    with tempfile.TemporaryDirectory() as d:
        try:
            load('nope.json', base_dir=d, verbose=False)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('a missing file loaded')
    for data, matrix, says in (
            (broken(lambda d: d['T1']['regime_transitions']['low'].update(low=0.9)),
             np.eye(2), 'sum to'),
            (good, np.array([[1.0, 0.3], [0.1, 1.0]]), 'not symmetric'),
            (good, np.array([[1.0, 2.0], [2.0, 1.0]]), 'negative eigenvalues'),
            (broken(lambda d: d['metadata'].update(last_updated='2001-01-01')),
             np.eye(2), 'days old')):
        got, warned = attempt(data, matrix)
        assert isinstance(got, dict) and any(says in m for m in warned), (says, warned)


def test_the_parameter_loader_finds_the_correlation_files_a_regen_names():
    """The loader test above writes legacy correlation_<kind>_2.npy files and no
    metadata, so it never reaches the branch production uses: the regen names
    its files in metadata.correlation_files. A stale or renamed file after a
    regen, or a required key dropped from a ticker, must refuse by name."""
    import copy, datetime, warnings
    from montecarlo.loaders import load_parameters_from_json as load
    row = {'low': 0.8, 'medium': 0.15, 'high': 0.05}
    fund = {'expected_return': 0.07, 'base_volatility': 0.15, 'parameter_std': 0.01,
            'volatility_regimes': {'low': 0.1, 'medium': 0.15, 'high': 0.25},
            'regime_transitions': {r: dict(row) for r in ('low', 'medium', 'high')}}
    kinds = ('baseline', 'low', 'medium', 'high')
    meta = {'version': 'test', 'last_updated': datetime.date.today().isoformat()}
    good = {'metadata': meta, 'T1': copy.deepcopy(fund), 'T2': copy.deepcopy(fund)}

    def corr(r):
        return np.array([[1.0, r], [r, 1.0]])

    def attempt(data, files):
        """files: {filename: off-diagonal} written beside the parameters."""
        with tempfile.TemporaryDirectory() as d, warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = Path(d)
            (out / 'p.json').write_text(json.dumps(data))
            for name, r in files.items():
                np.save(out / name, corr(r))
            try:
                return load('p.json', base_dir=out, verbose=False)
            except (ValueError, FileNotFoundError) as exc:
                return exc

    # The regen's names win over a legacy file of the same kind.
    named = dict(good, metadata=dict(meta, correlation_files={
        k: f'regen_{k}.npy' for k in kinds}))
    files = {f'regen_{k}.npy': 0.1 * (i + 1) for i, k in enumerate(kinds)}
    files.update({f'correlation_{k}_2.npy': -0.9 for k in kinds})
    got = attempt(named, files)
    for i, k in enumerate(kinds):
        assert abs(got[f'CORRELATION_{k.upper()}'][0, 1] - 0.1 * (i + 1)) < 1e-12, (k, got)

    # A name the metadata promises but the directory lacks is refused, even
    # with a legacy file standing by that would have loaded.
    missing = dict(good, metadata=dict(meta, correlation_files={
        k: f'regen_{k}.npy' for k in kinds}))
    del files['regen_high.npy']
    err = attempt(missing, files)
    assert isinstance(err, FileNotFoundError) and 'specified in metadata' in str(err), err

    # With no metadata names, the newest dated file wins over an older one.
    dated = {f'correlation_{k}_2_{day}.npy': r
             for k in kinds for day, r in (('20260301', 0.7), ('20260101', 0.2))}
    got = attempt(good, dated)
    assert all(abs(got[f'CORRELATION_{k.upper()}'][0, 1] - 0.7) < 1e-12
               for k in kinds), got
    err = attempt(good, {})
    assert isinstance(err, FileNotFoundError) and 'not found' in str(err), err

    # One required key at a time: each refuses, naming the ticker and the key.
    legacy = {f'correlation_{k}_2.npy': 0.3 for k in kinds}
    for key in ('base_volatility', 'parameter_std', 'volatility_regimes'):
        data = copy.deepcopy(good)
        data['T2'].pop(key)
        err = attempt(data, legacy)
        assert isinstance(err, ValueError) and f'T2: Missing {key}' in str(err), (key, err)


def test_a_regen_downloads_each_series_once():
    """VFINX was fetched three times a regen (VOO, MTUM, VFMO), the DFWIX and VT
    components twice, and a whole step downloaded every fund into a dict
    nothing read. Each is a full-history request with retries."""
    import data_collection_regime_model as dc
    calls = []

    def fake(ticker, start=None, end=None, progress=False):
        calls.append((ticker, start))
        idx = pd.date_range('2000-01-31', periods=3, freq='ME')
        return pd.DataFrame({'Close': [1.0, 2.0, 3.0]}, index=idx)

    real = dc.yf.download
    dc._DOWNLOADS.clear()
    try:
        dc.yf.download = fake
        a = dc.download_with_retry('VFINX', start='1970-01-01')
        a.iloc[0] = 99.0                                  # a caller mutating its copy
        b = dc.download_with_retry('VFINX', start='1970-01-01')
        dc.download_with_retry('VFINX', start='1993-01-01')
    finally:
        dc.yf.download = real
        dc._DOWNLOADS.clear()
    assert calls == [('VFINX', '1970-01-01'), ('VFINX', '1993-01-01')], calls
    assert b.iloc[0] == 1.0, 'the cache handed out a shared, mutable series'
    src = inspect.getsource(dc.main)
    assert 'yf_data' not in src, 'the unread Step 3 download loop is back'


def test_an_in_process_sweep_keeps_its_worker_pool():
    """measure() shut the Pool down after every sweep point, so an in-process
    sweep re-spawned cpu_count() workers per value, each re-importing the
    engine under `spawn`. Only the subprocess path needs a fresh interpreter."""
    import sensitivity as sens
    dim = next(n for n in sens.DIMENSIONS if n not in sens.ENV_DIMS
               and sens.DIMENSIONS[n].kind == 'numeric')
    value = sens.DIMENSIONS[dim].default.split(',')[0]
    shut = []
    real_solve, real_shut = solver.solve_sustainable_spend, solver.shutdown_pool
    try:
        solver.solve_sustainable_spend = lambda *a, **k: {'median_realised': 1.0}
        solver.shutdown_pool = lambda: shut.append(1)
        for _ in range(3):
            sens.sweep_point(dim, value, ['141'], 24)
    finally:
        solver.solve_sustainable_spend, solver.shutdown_pool = real_solve, real_shut
    assert not shut, f'the pool was torn down {len(shut)} times in one sweep'


def _tax_mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'tax_estimate', Path(__file__).resolve().parent / 'tax_estimate.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tax_shares_contribute_when_the_engine_does():
    """The engine contributes THROUGH stop_age (it stops when age > stop_age)
    and treats a None stop_age as never stopping. project_shares kept its own
    copy of the rule — `y < stop_age` — so it dropped the last contribution
    year, and raised TypeError on None."""
    m = _tax_mod()
    rec = {'age': 40, 'retirement_age': 65, 'contributions_measured_at_age': 40,
           'accounts': {
               'A': {'balance': 1000.0, 'annual_amount': 0.0, 'growth_rate': 0.0,
                     'stop_age': 40},
               'B': {'balance': 1000.0, 'annual_amount': 100.0, 'growth_rate': 0.0,
                     'stop_age': 40}}}
    shares = m.project_shares(rec, 42)
    assert shares[41]['B'] > shares[41]['A'], 'B contributes in its stop_age year'
    assert abs(shares[42]['B'] - shares[41]['B']) < 1e-12, 'and not after it'
    rec['accounts']['B']['stop_age'] = None
    shares = m.project_shares(rec, 42)
    assert shares[42]['B'] > shares[41]['B'], 'None never stops'


def test_tax_brackets_stack_gains_on_top_of_ordinary_income():
    """Capital gains are taxed in the bands ABOVE ordinary income, not from
    zero. Getting this wrong is what made the repo's only existing tax figure
    (R6's '20% on half the traded amount') wrong in both directions: at this
    investor's income most gains fall in the 0% LTCG band, and the rate is
    15%+3.8% only once RMDs push ordinary income up."""
    m = _tax_mod()
    # ordinary: 10% to 23,850 then 12% -> hand-checked
    assert abs(m._stacked_tax(41_159, m.ORDINARY_BANDS, 0.0) - 4_462) < 2
    # gains stacked under the 0% ceiling are FREE
    assert m._stacked_tax(24_102, m.LTCG_BANDS, 41_159) == 0.0
    # ...and crossing it is taxed only on the excess
    got = m._stacked_tax(60_000, m.LTCG_BANDS, 80_000)
    assert abs(got - 0.15 * (80_000 + 60_000 - m.LTCG_BANDS[0][0])) < 1, got
    # the same gains are NOT free once ordinary income already fills the band
    assert m._stacked_tax(24_102, m.LTCG_BANDS, 96_700) > 3_000

    # ...and the STACKING must survive into tax_a_path, not just exist as a
    # helper. Same gains, different ordinary income: if the call site passed 0
    # instead of the ordinary base, the gains tax would be identical in both.
    tax_of = {'401k_trad': 'pretax', 'roth_ira': 'roth', 'taxable': 'taxable'}
    n = 5                       # ages 65-69, before any RMD can interfere
    bal, draw = [9_000_000.0] * n, np.full(n, 400_000.0)

    def gains_tax(pretax_share):
        with_t = {'401k_trad': pretax_share, 'roth_ira': 0.9 - pretax_share,
                  'taxable': 0.10}
        without = {'401k_trad': pretax_share,
                   'roth_ira': 1.0 - pretax_share, 'taxable': 0.0}
        sh_a = {a: with_t for a in range(65, 96)}
        sh_b = {a: without for a in range(65, 96)}
        a = m.tax_a_path(bal, draw, sh_a, 0.0, 65, tax_of, 70, 75)[0]
        b = m.tax_a_path(bal, draw, sh_b, 0.0, 65, tax_of, 70, 75)[0]
        return a - b

    low, high = gains_tax(0.02), gains_tax(0.60)
    # $28,800 of gains a year for five years: taxed at 15% once ordinary income
    # fills the 0% band, so ~$21,600 plus NIIT, against ~$0 when ordinary income
    # is low. A LOOSE threshold here is vacuous — passing 0.0 instead of the
    # ordinary base still leaves ~$3,572 of NIIT, which cleared a $1,000 bar.
    assert low < 1_000, low
    assert high > 15_000, (low, high, 'gains are not being stacked')


def test_rmds_force_withdrawals_beyond_what_was_wanted():
    """The tax that matters here is not on the draw you chose — it is on the
    one the RMD compels. A pro-rata draw takes ~$25.6k from the pretax sleeve
    while the RMD on the age-73 balance is $61-71k."""
    m = _tax_mod()
    shares = {a: {'401k_trad': 0.175, 'roth_ira': 0.725, 'taxable': 0.10}
              for a in range(65, 96)}
    tax_of = {'401k_trad': 'pretax', 'roth_ira': 'roth', 'taxable': 'taxable'}
    n = 31
    balances = [8_000_000.0] * n
    draws = np.full(n, 150_000.0)

    _, _, forced = m.tax_a_path(balances, draws, shares, 55_800.0, 65, tax_of,
                                   70, 75)
    assert forced > 0, 'RMDs never bound on an $8M book'

    # a draw large enough to satisfy the RMD on its own forces nothing
    _, _, none_forced = m.tax_a_path(balances, np.full(n, 900_000.0), shares,
                                     55_800.0, 65, tax_of, 70, 75)
    assert none_forced < forced, (none_forced, forced)

    # and before the RMD age there is no RMD at all
    _, _, early = m.tax_a_path(balances[:5], draws[:5], shares, 55_800.0, 65,
                               tax_of, 70, 75)
    assert early == 0.0, early

    # THE RMD AGE FOLLOWS THE BIRTH YEAR. This tool first hard-coded 73, which
    # is SECURE 2.0's rule for someone born 1951-59 and wrong for this investor
    # (born ~1989 -> 75). The law moved it three times between 2019 and 2022.
    assert m.rmd_start_age(1989) == 75
    assert m.rmd_start_age(1960) == 75
    assert m.rmd_start_age(1955) == 73
    assert m.rmd_start_age(1950) == 72
    # The age GATES the start: over 65-74 an RMD age of 73 forces withdrawals
    # and 75 forces none. NOT "a later start forces less in total" — that was
    # the first version of this assertion and it is false: the pretax balance
    # compounds two more years untouched and the later divisors are smaller,
    # so lifetime forced withdrawals come out slightly LARGER at 75.
    _, _, gate73 = m.tax_a_path(balances[:10], draws[:10], shares, 55_800.0,
                                65, tax_of, 70, rmd_age=73)
    _, _, gate75 = m.tax_a_path(balances[:10], draws[:10], shares, 55_800.0,
                                65, tax_of, 70, rmd_age=75)
    assert gate73 > 0 and gate75 == 0.0, (gate73, gate75)


def test_the_tax_tool_puts_social_security_in_the_SIMULATION():
    """Building the config without it solved to $169,730 against the $206,182
    the same portfolio reaches WITH it — and the tax maths then credited a
    benefit the paths never received. That flatters the result twice.

    The invariant is ONE SOURCE: the benefit in the tax maths must come from
    the very `cfg` the paths were simulated with, through the engine's own
    policies.ss_income — not a hard-coded fallback copy of the benefit, and not
    the tool's own delayed-credit arithmetic, both of which it once carried."""
    m = _tax_mod()
    src = re.sub(r'#[^\n]*', '', inspect.getsource(m.main))
    build = src.index('runner.build_config(spec)')
    read = src.index("cfg.get('social_security')")
    assert build < read, 'SS is read before the simulation config exists'
    assert 'policies.ss_income' in src, "the benefit is not the engine's own"
    assert '60_000' not in src and '60000' not in src, \
        'a hard-coded copy of the benefit is back'
    assert '0.08' not in src, 'the delayed-credit arithmetic is duplicated again'


_SS = {'annual_at_fra': 60_000.0, 'claim_age': 70.0, 'benefit_scale': 0.75}


def _temp_investor(ss):
    """investor.FILE pointed at a temp record for the duration — NEVER the
    user's config/investor.json, which test_zzz checks is left untouched."""
    import copy

    @contextlib.contextmanager
    def ctx():
        rec = copy.deepcopy(investor.DEFAULTS)
        rec['social_security'] = copy.deepcopy(ss)
        f = Path(tempfile.mkdtemp()) / 'investor.json'
        f.write_text(json.dumps(rec))
        real = investor.FILE
        investor.FILE = f
        try:
            yield f
        finally:
            investor.FILE = real
    return ctx()


def _scripted(keys):
    """runner.ask replaced by a script that RAISES when exhausted — a re-asking
    prompt must fail the test, not hang the gate."""
    it = iter(keys)

    def _ask(_p, _d=''):
        try:
            return next(it)
        except StopIteration:
            raise AssertionError('the prompt asked for more input than scripted')
    return _ask


def test_social_security_lives_in_your_details_and_validates():
    """It was asked on EVERY run and carried on the spec — three numbers typed
    identically each time. It now lives with the accounts, and a bad block must
    be refused by the same validator, not discovered as a wrong plan level."""
    import copy
    assert investor.validate_social_security(_SS) == []
    assert investor.validate_social_security(None) == [], 'no benefit is valid'
    for bad in (dict(_SS, claim_age=61.0), dict(_SS, claim_age=71.0),
                dict(_SS, annual_at_fra=0.0), dict(_SS, annual_at_fra=600_000.0),
                dict(_SS, benefit_scale=1.5), dict(_SS, benefit_scale=-0.1),
                'yes'):
        assert investor.validate_social_security(bad), bad
    rec = copy.deepcopy(investor.DEFAULTS)
    rec['social_security'] = dict(_SS, claim_age=75.0)
    assert any('claim_age' in b for b in investor.validate(rec)), \
        'the full validator does not reach the block'
    rec['social_security'] = dict(_SS)
    assert 'SS $5.0k/mo @70' in investor.summary(rec), investor.summary(rec)


def test_a_run_reads_social_security_from_your_details_unless_the_spec_overrides():
    """Presence, not truthiness: a spec that carries the key overrides your
    details for that run (None = exclude), which is also what makes older saved
    specs replay with the benefit typed that day. Absent, the stored one applies."""
    base = {'mode': 'retirement', 'portfolios': ['140'], 'spend': 'solve',
            'policy': 'guyton_klinger'}
    with _temp_investor(_SS):
        assert runner.build_config(base)['social_security'] == _SS
        over = dict(_SS, claim_age=67.0)
        assert runner.build_config({**base, 'social_security': over})[
            'social_security'] == over
        assert runner.build_config({**base, 'social_security': None})[
            'social_security'] is None, 'None must EXCLUDE it, not fall through'
        cfg = runner.build_config(base)
        cfg['social_security']['claim_age'] = 62.0
        assert runner.build_config(base)['social_security']['claim_age'] == 70.0, \
            'mutating one run config reached the stored details'
    with _temp_investor(None):
        assert runner.build_config(base)['social_security'] is None


def test_a_stored_benefit_is_not_asked_for_again():
    real = runner.ask
    try:
        with _temp_investor(_SS):
            runner.ask = _scripted([])          # ANY prompt fails the test
            spec, buf = {}, io.StringIO()
            with contextlib.redirect_stdout(buf):
                runner.ask_social_security(spec)
            assert 'social_security' not in spec, \
                'the spec must not become a second copy of your details'
            assert 'from your details' in buf.getvalue(), buf.getvalue()

        # not stored: ask once, then offer to store it
        with _temp_investor(None):
            runner.ask = _scripted(['5000', '70', '75', 'y'])
            spec = {}
            with contextlib.redirect_stdout(io.StringIO()):
                runner.ask_social_security(spec)
            assert investor.load()['social_security'] == _SS, \
                investor.load()['social_security']
            assert 'social_security' not in spec

        # declining the save keeps it to this run only
        with _temp_investor(None):
            runner.ask = _scripted(['5000', '70', '75', 'n'])
            spec = {}
            with contextlib.redirect_stdout(io.StringIO()):
                runner.ask_social_security(spec)
            assert spec['social_security'] == _SS, spec
            assert investor.load()['social_security'] is None
    finally:
        runner.ask = real


def test_social_security_prompt_unhappy_paths():
    """Each of these sets the benefit every plan level rests on: blank and
    non-numeric exclude it, a monthly figure over $20k is offered as annual (the
    12x guard), the claim age and haircut are bounded, and a refused save falls
    back to a per-run override rather than losing the figure."""
    def drive(typed, save=None):
        spec, buf = {}, io.StringIO()
        real = runner.ask, investor.save
        runner.ask = _scripted(typed)
        if save:
            investor.save = save
        try:
            with _temp_investor(None), contextlib.redirect_stdout(buf):
                runner.ask_social_security(spec)
                stored = investor.load()['social_security']
        finally:
            runner.ask, investor.save = real
        return spec, buf.getvalue(), stored

    spec, out, stored = drive([''])
    assert spec == {} and stored is None and 'not a number' not in out, (spec, out)
    spec, out, stored = drive(['abc'])
    assert spec == {} and stored is None and 'excluding Social Security' in out, out

    # $60,000 typed as a MONTHLY figure: 'y' divides it by 12, 'n' keeps it
    spec, out, _ = drive(['60000', 'y', '67', '100', 'n'])
    assert 'looks like an ANNUAL figure' in out, out
    assert spec['social_security']['annual_at_fra'] == 60_000.0, spec
    spec, _, _ = drive(['60000', 'n', '67', '100', 'n'])
    assert spec['social_security']['annual_at_fra'] == 60_000.0 * 12, spec

    # claim age: out of range and text are re-asked until 62-70
    spec, out, _ = drive(['5000', '61', 'abc', '71', '66', '100', 'n'])
    assert spec['social_security']['claim_age'] == 66.0, spec
    assert out.count('between 62 and 70') == 3, out

    # haircut: text means 100%, over 100% clamps to 100%
    for typed, want in (('abc', 1.0), ('150', 1.0), ('80', 0.8)):
        spec, _, _ = drive(['5000', '67', typed, 'n'])
        assert spec['social_security']['benefit_scale'] == want, (typed, spec)

    # save refused: say so, keep the figure for this run, store nothing
    def refuse(_inv):
        raise ValueError('boom')
    spec, out, stored = drive(['5000', '70', '75', 'y'], save=refuse)
    assert spec['social_security'] == _SS, spec
    assert stored is None and 'not saved (boom)' in out, out


def test_the_details_screen_edits_and_removes_social_security():
    import copy
    inv = copy.deepcopy(investor.DEFAULTS)
    inv['social_security'] = None
    names = list(inv['accounts'])
    row = 5 + len(names)
    real = runner.ask
    try:
        # an ANNUAL figure typed in the monthly box is refused, not stored x12
        runner.ask = _scripted(['60000', '5,000', '70', '75'])
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            runner._edit_row(inv, row, names)
        assert inv['social_security'] == _SS, inv['social_security']
        assert 'ANNUAL' in buf.getvalue(), buf.getvalue()
        assert investor.validate(inv) == []

        with contextlib.redirect_stdout(io.StringIO()) as buf:
            runner._investor_screen(inv)
        screen = buf.getvalue()
        assert f'{row:>3}  social security' in screen, screen
        assert '$5,000/mo' in screen and '75% of scheduled' in screen, screen

        runner.ask = _scripted(['0'])        # 0 removes it
        with contextlib.redirect_stdout(io.StringIO()):
            runner._edit_row(inv, row, names)
        assert inv['social_security'] is None
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            runner._investor_screen(inv)
        assert 'not set' in buf.getvalue(), buf.getvalue()
    finally:
        runner.ask = real


def test_editing_your_age_asks_whether_the_contributions_moved_with_it():
    """Contributions compound from `contributions_measured_at_age`; a new age
    with amounts that are already current, left unmoved, gives them a year of
    growth they contain. Only 'n' leaves the field alone; rows 2-4 are plain."""
    import copy

    def edit(row, typed):
        inv = copy.deepcopy(investor.DEFAULTS)
        real = runner.ask
        runner.ask = _scripted(typed)
        try:
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                runner._edit_row(inv, row, list(inv['accounts']))
        finally:
            runner.ask = real
        return inv, buf.getvalue()

    was = investor.DEFAULTS['contributions_measured_at_age']
    assert investor.DEFAULTS['age'] != 41 and was != 41
    inv, out = edit(1, ['41', 'y'])
    assert inv['age'] == 41 and inv['contributions_measured_at_age'] == 41, inv
    assert 'contributions now measured at age 41' in out, out
    inv, out = edit(1, ['41', 'n'])
    assert inv['age'] == 41 and inv['contributions_measured_at_age'] == was, inv
    inv, _ = edit(1, ['', ])                 # unchanged age: no question to answer
    assert inv['age'] == investor.DEFAULTS['age']
    inv, out = edit(1, ['4x', '41', ''])     # unreadable is re-asked; blank = yes
    assert inv['contributions_measured_at_age'] == 41 and 'could not read' in out, out

    inv, _ = edit(2, ['90'])
    assert inv['plan_to_age'] == 90 and inv['age'] == investor.DEFAULTS['age'], inv
    inv, _ = edit(3, ['62'])
    assert inv['retirement_age'] == 62 and inv['plan_to_age'] == 95, inv
    inv, _ = edit(4, ['38'])
    assert inv['contributions_measured_at_age'] == 38 and inv['age'] == 37, inv


def test_the_run_record_says_which_benefit_it_used():
    """Moving the claim age from 67 to 70 changes every plan level; --diff has
    to be able to attribute it rather than report 'provenance identical'."""
    with_ss = persistence.provenance({'social_security': _SS})
    assert with_ss['ss_annual_at_fra'] == 60_000.0
    assert with_ss['ss_claim_age'] == 70.0 and with_ss['ss_benefit_scale'] == 0.75
    without = persistence.provenance({'social_security': None})
    assert without['ss_annual_at_fra'] == 0.0 and without['ss_claim_age'] is None


def test_the_social_security_sweeps_run_from_your_details():
    """`ss_scale` existed and could not run: it demanded you 'set one in
    run.default_config() first' — edit source — because the benefit lived only
    on a per-run spec. `ss_claim_age` is new: claiming age is a real decision."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'sensitivity', Path(__file__).resolve().parent / 'sensitivity.py')
    sens = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sens)
    assert 'ss_claim_age' in sens.DIMENSIONS
    base = {'mode': 'retirement', 'portfolios': ['140'], 'spend': 'solve',
            'policy': 'guyton_klinger'}
    with _temp_investor(_SS):
        cfg = runner.build_config(base)
        sens.apply_dim(cfg, 'ss_claim_age', '67')
        assert cfg['social_security']['claim_age'] == 67.0
        assert cfg['social_security']['benefit_scale'] == 0.75, 'clobbered'
        cfg = runner.build_config(base)
        sens.apply_dim(cfg, 'ss_scale', '0.8')
        assert cfg['social_security']['benefit_scale'] == 0.8
        assert cfg['social_security']['claim_age'] == 70.0, 'clobbered'
    with _temp_investor(None):
        cfg = runner.build_config(base)
        try:
            sens.apply_dim(cfg, 'ss_claim_age', '67')
        except SystemExit as e:
            assert 'Your details' in str(e), e
        else:
            raise AssertionError('swept a benefit that does not exist')


def test_a_sweep_runs_the_guardrail_response_you_actually_use():
    """Every sweep point was built at the SHIPPED `slow` response, while the
    user's own runs use `fast` — and slow / medium / fast moved the same
    portfolio's plan level by ~15%. A sweep that silently runs a different
    policy reports levels for a plan nobody is running."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'sensitivity', Path(__file__).resolve().parent / 'sensitivity.py')
    sens = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sens)
    assert sens._base_spec()['policy'] == 'guyton_klinger'
    assert 'gk_response' not in sens._base_spec(), 'the default must stay shipped'
    cfg = runner.build_config(sens._base_spec('fast'))
    assert cfg['withdrawal']['gk_response'] == 'fast', cfg['withdrawal']

    # and main must hand the flag to every point
    seen = []
    real = sens.sweep_point
    try:
        sens.sweep_point = lambda d, v, st, si, gk=None: seen.append(gk) or {
            s_: {'spend': 1.0} for s_ in st}
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                sens.main(['ss_claim_age', '67,70', '--strategies', '140',
                           '--gk-response', 'fast'])
            except Exception:
                pass            # the report may want more than the stub gives
    finally:
        sens.sweep_point = real
    assert seen and all(g == 'fast' for g in seen), seen


def test_an_rmd_is_the_balance_over_THIS_ages_divisor():
    """The existing RMD tests check that RMDs bind and when they start — not
    the AMOUNT. A divisor read one row off (age 74's or 76's instead of 75's)
    moves every forced withdrawal by ~3.5% and passed all of them. $2.46M of
    pretax at 75, divisor 24.6, is exactly $100,000."""
    m = _tax_mod()
    one = {a: {'401k_trad': 1.0} for a in range(60, 100)}
    tax_of = {'401k_trad': 'pretax'}
    forced = m.tax_a_path([2_460_000.0], np.zeros(1), one, 0.0, 75, tax_of, 70, 75)[2]
    assert abs(forced - 100_000.0) < 0.01, forced
    # and the next year uses the NEXT row, on the balance left after the first
    two = m.tax_a_path([2_460_000.0] * 2, np.zeros(2), one, 0.0, 75, tax_of, 70, 75)[2]
    left = (2_460_000.0 - 100_000.0) * 1.05
    assert abs(two - (100_000.0 + left / m.RMD_DIVISOR[76])) < 0.01, two


def test_the_confirm_screen_shows_the_benefit_the_run_will_actually_use():
    """`effective_ss` feeds the confirm screen — the last thing you read before
    a run starts — and could be gutted with nothing noticing, because every
    headless test stubs `confirm` out. Found by tools/vacuity_sweep.py --mutate.
    A screen that shows your stored benefit while a per-run override is about
    to replace it is exactly the misreport it exists to prevent."""
    P = _P()
    base = {'mode': 'retirement', 'portfolios': ['140'], 'spend': 'solve',
            'policy': 'guyton_klinger'}
    specs, _ = runner.resolve_strategies(base, P)

    def screen(spec):
        real = runner.ask
        runner.ask = _scripted(['y'])
        try:
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                runner.confirm(spec, specs, P)
        finally:
            runner.ask = real
        return buf.getvalue()

    with _temp_investor(_SS):
        assert runner.effective_ss(base) == _SS
        assert 'Social Security $56k/yr from 70' in screen(base), screen(base)
        # an override REPLACES the stored benefit on screen, not beside it
        over = {**base, 'social_security': dict(_SS, claim_age=67.0)}
        assert runner.effective_ss(over)['claim_age'] == 67.0
        shown = screen(over)
        assert 'from 67' in shown and 'from 70' not in shown, shown
        # and None means the run has NONE, so the screen must say nothing
        off = {**base, 'social_security': None}
        assert runner.effective_ss(off) is None
        assert 'Social Security' not in screen(off), screen(off)


# ---------------------------------------------------------------------------
# IF THE RETURN MODEL IS WRONG — return_views.py
# ---------------------------------------------------------------------------

def test_a_return_view_pins_every_return_knob_and_inherits_the_rest():
    """A view must PIN the three knobs that define the return model, not just
    the one it moves: a run made under MC_MU_METHOD=historical would otherwise
    carry it into 'intl keeps lagging', where the scenario does nothing because
    no factor build-up happens. Everything else — inflation, the cash rate —
    is inherited, so a view varies the return model and only that."""
    from montecarlo import return_views as rv
    base = {'MC_MU_METHOD': 'historical', 'MC_PREMIA_SCENARIO': 'value dead',
            'MC_INFLATION': '0.04', 'PATH': '/usr/bin'}
    env = rv.view_env('intl keeps lagging', base)
    assert env['MC_MU_METHOD'] == 'factor', env
    assert env['MC_PREMIA_SCENARIO'] == 'intl keeps lagging', env
    assert env['MC_PREMIUM_SCALE'] == '1.0', env
    assert env['MC_INFLATION'] == '0.04' and env['PATH'] == '/usr/bin', env
    env = rv.view_env('historical', base)
    assert env['MC_MU_METHOD'] == 'historical', env
    assert env['MC_PREMIA_SCENARIO'] == 'full history', env
    assert base['MC_PREMIA_SCENARIO'] == 'value dead', 'view_env mutated its input'
    # every view names settings the engine accepts, so none can die at import
    for v in rv.VIEWS:
        e = rv.view_env(v.key, {})
        assert e['MC_PREMIA_SCENARIO'] in factors.SCENARIOS, v.key
        assert e['MC_MU_METHOD'] in ('factor', 'historical'), v.key
        float(e['MC_PREMIUM_SCALE'])
    assert rv.VIEWS[0].key == 'factor', 'the default view must lead the table'


def test_the_run_knows_which_view_it_already_is():
    """The main run IS one of the views, and re-running it in a child would
    cost a full evaluation to reproduce numbers already in hand. 'us stays
    ahead' is the default under another name, so it must match too — and a
    view the list does not hold must match nothing, not the nearest one."""
    from montecarlo import return_views as rv
    assert rv.current_view('factor', 'full history', 1.0) == 'factor'
    assert rv.current_view('factor', 'us stays ahead', 1.0) == 'factor'
    assert rv.current_view('factor', 'intl keeps lagging', 1.0) == 'intl keeps lagging'
    assert rv.current_view('historical', 'full history', 1.0) == 'historical'
    assert rv.current_view('factor', 'value dead', 1.0) is None
    assert rv.current_view('factor', 'full history', 0.5) is None


def _views_fixture(medians_by_view, n=400, noise=1.0, seed=7):
    """Paired books: one shared market term per path, a little noise each."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0, 20.0, n)
    return {view: {g: {'per_path_mean': m + market + rng.normal(0.0, noise, n),
                       'depleted': np.zeros(n, bool), 'spend': 50.0}
                   for g, m in meds.items()}
            for view, meds in medians_by_view.items()}


def test_regret_is_the_gap_to_each_views_leader_and_minimax_takes_the_smallest_worst():
    """b never leads and is still the answer if you cannot say which view is
    true: it gives up at most ~3% anywhere, where a and c each risk 5-6%."""
    from montecarlo import return_views as rv
    data = _views_fixture({'factor': {'a': 100, 'b': 97, 'c': 95},
                           'historical': {'a': 90, 'b': 93, 'c': 96}})
    t = rv.regrets(data, ['a', 'b', 'c'])
    assert t['metric'] == 'spend', t['metric']
    assert t['views']['factor']['leader'] == 'a'
    assert t['views']['historical']['leader'] == 'c'
    r = t['views']['factor']['regret']
    assert r['a'] == 0.0 and abs(r['b'] - 0.03) < 0.005 and abs(r['c'] - 0.05) < 0.005, r
    assert abs(t['worst']['a'] - 6 / 96) < 0.006, t['worst']
    assert abs(t['worst']['b'] - 3 / 96) < 0.006, t['worst']
    assert t['worst_view']['a'] == 'historical', t['worst_view']
    assert t['pick'] == 'b', t['pick']
    # every gap is 3+ points on 1-point noise: nothing holds up everywhere
    assert t['robust'] == [], t['robust']


def test_a_strategy_tied_with_the_leader_has_not_lost_that_view():
    """A view's leader is a point estimate. A strategy the paired interval
    cannot separate from it has not LOST that view, and one that has lost no
    view is the robust answer, whatever its point regrets say."""
    from montecarlo import return_views as rv
    data = _views_fixture({'factor': {'a': 100, 'b': 99.95, 'c': 95},
                           'intl keeps lagging': {'a': 99.95, 'b': 100, 'c': 97}})
    t = rv.regrets(data, ['a', 'b', 'c'])
    assert t['views']['factor']['holds']['b'], t['views']['factor']
    assert t['views']['factor']['holds']['a']
    assert t['views']['intl keeps lagging']['holds']['a']
    assert not t['views']['factor']['holds']['c']
    assert sorted(t['robust']) == ['a', 'b'], t['robust']


def test_under_fixed_spending_the_regret_is_extra_ruin():
    """At one fixed plan every surviving path spends exactly the plan, so
    median spending is one number for every strategy and a regret on it is
    zero everywhere — a check that cannot fail. What differs is who runs out."""
    from montecarlo import return_views as rv
    n = 2000
    u = np.random.default_rng(3).permutation(n) / n     # exact rates

    def book(p):              # paired: the same bad paths run out first
        return {'per_path_mean': np.full(n, 50.0), 'depleted': u < p, 'spend': 50.0}

    data = {'factor': {'a': book(0.10), 'b': book(0.06)},
            'historical': {'a': book(0.08), 'b': book(0.13)}}
    t = rv.regrets(data, ['a', 'b'])
    assert t['metric'] == 'ruin', t['metric']
    assert t['views']['factor']['leader'] == 'b'
    assert t['views']['historical']['leader'] == 'a'
    assert abs(t['views']['factor']['regret']['a'] - 0.04) < 1e-9
    assert abs(t['worst']['b'] - 0.05) < 1e-9, t['worst']
    assert t['pick'] == 'a', t['pick']


def test_the_return_model_table_names_every_view_and_who_holds_up():
    from montecarlo import return_views as rv
    data = _views_fixture({v.key: {'a': 100, 'b': 97} for v in rv.VIEWS})
    t = rv.regrets(data, ['a', 'b'])
    plan = {v.key: {'success_rate': 0.9, 'median': 100_000.0, 'p10': 80_000.0}
            for v in rv.VIEWS}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rv.render(t, ['a', 'b'], plan, level=160_000, plan_name='a', n=400)
    out = buf.getvalue()
    assert 'IF THE RETURN MODEL IS WRONG' in out, out
    for v in rv.VIEWS:
        assert v.title in out, (v.title, out)
    rows = [l.strip() for l in out.splitlines()]
    for g in ('a', 'b'):
        assert any(l.startswith(g + ' ') for l in rows), (g, out)
    assert 'every view' in out, out          # a holds up everywhere — say so
    assert '2005' in out, 'the historical view must carry its stress-test caveat'
    assert '90%' in out and '$160k' in out, 'the plan under each view is missing'


def test_a_view_runs_in_its_own_process_on_the_same_markets():
    """Workers re-import under spawn, so a return view cannot be set in this
    process — it has to be a child with its env var set before import. Two
    things are worth proving: the child really ran under the view it was asked
    for, and it saw the SAME markets as this process. The second is what lets
    a view's numbers sit beside the main run's in one table: a child on other
    paths would be comparing two samples, not two models."""
    from montecarlo import return_views as rv
    P = _P()
    cfg = runner.build_config({'mode': 'retirement', 'portfolios': ['141'],
                               'spend': 'solve', 'policy': 'guyton_klinger'})
    cfg = dict(cfg, time_horizon_years=12, age=60, retirement_age=64)
    cfg['withdrawal'] = dict(cfg['withdrawal'], start_age=64)
    specs = ['141', '145']
    plans = {g: glide.build_plan(glide.parse_schedule(g, 60), P) for g in specs}

    got = rv.run_view('factor', cfg, plans, level=150_000, n=6)
    assert got['engine'] == {'MU_METHOD': 'factor', 'PREMIA_SCENARIO': 'full history',
                             'PREMIUM_SCALE': 1.0}, got['engine']
    for g in specs:
        assert len(got['books'][g]['per_path_mean']) == 6, got['books'][g]
        assert 0.0 <= got['books'][g]['success_rate'] <= 1.0
    if rv.current_view(pa.MU_METHOD, pa.PREMIA_SCENARIO, pa.PREMIUM_SCALE) == 'factor':
        for g in specs:
            here = solver.evaluate(plans[g], cfg, 150_000, n_sims=6, parallel=False,
                                   floor=0.85, full=True)
            assert np.allclose(here['per_path_mean'], got['books'][g]['per_path_mean']), \
                (g, here['per_path_mean'], got['books'][g]['per_path_mean'])
    solver.shutdown_pool()

    lag = rv.run_view('intl keeps lagging', cfg, plans, level=150_000, n=6)
    assert lag['engine']['PREMIA_SCENARIO'] == 'intl keeps lagging', lag['engine']


def test_a_portfolio_run_can_check_itself_against_other_return_models():
    """End to end: the spec asks, every view runs, the section prints."""
    from montecarlo import return_views as rv
    out = _smoke('retirement', spend='solve', policy='guyton_klinger',
                 return_views=True, verify_sims=24, search_sims=12)
    assert 'Traceback' not in out, out[-2000:]
    assert 'IF THE RETURN MODEL IS WRONG' in out, out[-2000:]
    sec = out.split('IF THE RETURN MODEL IS WRONG', 1)[1]
    for v in rv.VIEWS:
        assert v.title in sec, (v.title, sec[:1500])
    rows = [l.strip() for l in sec.splitlines()]
    for g in ('141', '145'):
        assert any(l.startswith(g + ' ') for l in rows), (g, sec[:1500])


def test_a_fixed_spend_retirement_run_survives_end_to_end():
    """Every other retirement smoke SOLVES. A fixed dollar amount takes
    evaluate_strategies' other branch (solver.evaluate, no 'common' one-plan
    re-run), and its results reach the tables and the return-model check in a
    different shape: raw.depleted and a top-level per_path_mean."""
    kept = []
    out = _smoke('retirement', _capture=kept, spend=150_000.0, policy='guyton_klinger',
                 return_views=True)
    assert 'Traceback' not in out, out[-2000:]
    (res,), = [kept]
    for g in ('141', '145'):
        assert 'common' not in res[g], 'a fixed-spend run has no one-plan re-run'
        assert res[g]['spend'] == 150_000.0, res[g]['spend']
        assert 'depleted' in res[g]['raw'] and len(res[g]['per_path_mean']), g
    for section in ('RETIREMENT ANALYSIS', 'RUNNING', 'RESULTS', 'HEAD-TO-HEAD',
                    'SPENDING BY LIFE STAGE', 'IF THE RETURN MODEL IS WRONG'):
        assert section in out, (section, out[:600])
    assert 'SOLVING' not in out and 'COMPARING' not in out, out[:600]
    # Nothing was aimed at 90%, so a success rate far from it is not a miss.
    assert 'CONFIDENCE   aimed at' not in out, out.split('CONFIDENCE', 1)[-1][:600]
    assert out.count('$150k/yr plan level') == 2, out[:900]
    body = out.split('RESULTS', 1)[1].split('HEAD-TO-HEAD', 1)[0]
    for g in ('141', '145'):
        assert any(l.strip().startswith(g) for l in body.splitlines()), (g, body)
    sec = out.split('IF THE RETURN MODEL IS WRONG', 1)[1]
    assert 'YOUR PLAN under each view: 141 at $150k/yr' in sec, sec[-1200:]
    assert 'certified at' not in sec, 'nothing was solved'
    assert 'lifetime spending given up' in sec, 'spending varies by path, so it is scored'

    # Spending that never varies (constant_real, comfortably funded) is scored on
    # the chance of running out instead — the metric that reads raw.depleted.
    from montecarlo import return_views as rv
    summaries, real_check, kept = [], rv.check, []

    def spy(*a, **k):
        summaries.append(real_check(*a, **k))
        return summaries[-1]
    rv.check = spy
    try:
        out = _smoke('retirement', _capture=kept, spend=70_000.0,
                     policy='constant_real', return_views=True)
    finally:
        rv.check = real_check
    assert 'Traceback' not in out, out[-2000:]
    sec = out.split('IF THE RETURN MODEL IS WRONG', 1)[1]
    assert 'extra chance of running out' in sec, sec[:900]
    assert 'lifetime spending given up' not in sec, sec[:900]
    ruin = {g: float(np.mean(kept[0][g]['raw']['depleted'])) for g in ('141', '145')}
    assert ruin['141'] != ruin['145'], f'the fixture no longer separates the funds: {ruin}'
    lead = min(ruin, key=ruin.get)
    other = '145' if lead == '141' else '141'
    got = summaries[0]['views']['factor']['regret'][other]
    assert summaries[0]['metric'] == 'ruin' and abs(got - (ruin[other] - ruin[lead])) < 1e-12, \
        (got, ruin)


def test_the_estimate_prices_the_return_model_check():
    """Four more evaluations of every strategy is not free, and the confirm
    screen is where you decide whether it is worth it."""
    from montecarlo import return_views as rv
    spec = {'mode': 'retirement', 'spend': 'solve', 'policy': 'guyton_klinger'}
    base = runner.estimate_minutes(3, spec)
    more = runner.estimate_minutes(3, dict(spec, return_views=True))
    paths = (len(rv.VIEWS) - 1) * 3 * solver.VERIFY_SIMS
    assert more - base >= paths * runner._seconds_per_path('retirement') / 60 - 1e-9, \
        (base, more)


def test_a_portfolio_comparison_asks_whether_to_check_the_return_model():

    def answer(typed, n=3, **spec):
        spec = {'mode': 'retirement', 'policy': 'guyton_klinger', **spec}
        real = runner.ask
        runner.ask = lambda _p, _d='': typed if typed else _d
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                runner.ask_return_views(spec, n)
        finally:
            runner.ask = real
        return spec

    assert answer('')['return_views'] is True, 'the default must be to check'
    assert 'return_views' not in answer('n'), 'no must leave the spec as it was'
    # nothing to rank with one strategy, and a policy comparison is a plan question
    assert 'return_views' not in answer('', n=1)
    assert 'return_views' not in answer('', compare_policies=['vpw', 'guyton_klinger'])
    src = re.sub(r'#[^\n]*', '', inspect.getsource(runner.prompt_run))
    assert 'ask_return_views(' in src, 'prompt_run never asks'


# ---------------------------------------------------------------------------
# RETIREMENT CHOICES — one portfolio, several plans, the same paths
# ---------------------------------------------------------------------------

def test_a_plan_choice_fans_out_one_run_per_value():
    """The policy fan-out already gave each policy its own solve on shared
    paths. A plan choice is the same shape: one run per value, tagged so the
    table reads as the choice, with the value carried to the config."""
    spec = {'mode': 'retirement', 'portfolios': ['141'], 'spend': 'solve',
            'policy': 'guyton_klinger', 'gk_response': 'fast',
            'compare_choice': {'dimension': 'retirement_age', 'values': [60, 65, 70]}}
    runs = runner.policy_runs(spec)
    assert [t for t, _, _ in runs] == ['retire at 60', 'retire at 65', 'retire at 70'], runs
    assert all(p == 'guyton_klinger' for _, p, _ in runs)
    assert runs[0][2]['choice'] == ('retirement_age', 60.0), runs[0][2]
    assert runs[0][2]['gk_response'] == 'fast', 'the run settings must ride along'
    assert runner.choice_tag('contributions', 1.5) == 'save ×1.5'
    assert runner.choice_tag('ss_claim_age', 70) == 'claim at 70'


def test_a_plan_choice_changes_one_thing_in_a_copy():
    """Options share one base config, and scaling contributions works on the
    account dicts in place — so an option that edited the base would hand its
    savings rate to every option after it."""
    cfg = runner.build_config({'mode': 'retirement', 'portfolios': ['141'],
                               'spend': 'solve', 'policy': 'guyton_klinger'})
    before = {k: a['annual_amount']
              for k, a in cfg['contribution_schedule']['accounts'].items()}
    more = runner.apply_choice(cfg, 'contributions', 1.5)
    for k, a in more['contribution_schedule']['accounts'].items():
        assert abs(a['annual_amount'] - 1.5 * before[k]) < 1e-9, (k, a)
    assert {k: a['annual_amount'] for k, a in
            cfg['contribution_schedule']['accounts'].items()} == before, \
        'apply_choice edited the base config'

    early = runner.apply_choice(cfg, 'retirement_age', 60)
    assert early['retirement_age'] == 60 and early['withdrawal']['start_age'] == 60
    assert cfg['withdrawal']['start_age'] == cfg['retirement_age'] != 60 or \
        cfg['retirement_age'] == 60

    if cfg.get('social_security'):
        late = runner.apply_choice(cfg, 'ss_claim_age', 70)
        assert late['social_security']['claim_age'] == 70
        assert late['social_security']['annual_at_fra'] == \
            cfg['social_security']['annual_at_fra']
    try:
        runner.apply_choice(dict(cfg, social_security=None), 'ss_claim_age', 70)
        raise AssertionError('a claiming age without a benefit must refuse')
    except ValueError:
        pass


def test_a_retirement_age_choice_stops_contributions_before_it():
    """'Retire at 60' kept each account's own stop age, so a Roth IRA whose last
    contribution is 63 went on contributing three years into that retirement —
    flattering the early date with savings it would not have. A retirement-age
    choice now caps every account's last contribution at the year before it. It
    never EXTENDS one: retiring at 70 does not invent savings."""
    cfg = {'retirement_age': 65, 'withdrawal': {'start_age': 65},
           'contribution_schedule': {'accounts': {
               'ira': {'annual_amount': 7500.0, 'growth_rate': 0.0, 'stop_age': 63},
               'k401': {'annual_amount': 24500.0, 'growth_rate': 0.0, 'stop_age': 50},
               'open': {'annual_amount': 1000.0, 'growth_rate': 0.0, 'stop_age': None}}}}
    stops = lambda c: {k: a['stop_age'] for k, a in
                       c['contribution_schedule']['accounts'].items()}
    assert stops(runner.apply_choice(cfg, 'retirement_age', 60)) == \
        {'ira': 59, 'k401': 50, 'open': 59}
    assert stops(runner.apply_choice(cfg, 'retirement_age', 70)) == \
        {'ira': 63, 'k401': 50, 'open': 69}
    assert stops(cfg) == {'ira': 63, 'k401': 50, 'open': None}, 'edited the base'
    # the engine sees it: nothing contributed at 60 under 'retire at 60'
    early = runner.apply_choice(cfg, 'retirement_age', 60)['contribution_schedule']
    assert pa.calculate_annual_contribution(60, early) == 0.0
    assert pa.calculate_annual_contribution(59, early) > 0.0


def test_your_current_value_is_the_baseline():
    """A choice is read against what you would do anyway: the table's paired
    column is 'compared with retiring at 65', not with whichever option was
    typed first."""
    cfg = runner.build_config({'mode': 'retirement', 'portfolios': ['141'],
                               'spend': 'solve', 'policy': 'guyton_klinger'})
    ra = float(cfg['retirement_age'])
    assert runner.choice_baseline(cfg, 'retirement_age', [ra - 5, ra, ra + 5]) == ra
    assert runner.choice_baseline(cfg, 'contributions', [0.5, 1.0, 1.5]) == 1.0
    # yours not among them: the first value typed
    assert runner.choice_baseline(cfg, 'retirement_age', [ra + 1, ra + 2]) == ra + 1


def test_a_choice_spec_is_validated_before_anything_runs():
    P = {'141': {}, '145': {}}

    def refusal(**extra):
        spec = {'portfolios': ['141'], 'spend': 'solve',
                'compare_choice': {'dimension': 'retirement_age', 'values': [60, 65]}}
        spec.update(extra)
        try:
            return None, runner.parse_spec(json.dumps(spec), P)
        except ValueError as exc:
            return str(exc), None

    def bad(**extra):
        return refusal(**extra)[0] is not None

    assert not bad(), 'a valid choice spec was refused'
    assert bad(compare_choice={'dimension': 'shoe_size', 'values': [9]})
    assert bad(compare_choice={'dimension': 'retirement_age', 'values': []})
    assert bad(compare_policies=['vpw', 'guyton_klinger'])
    assert bad(spend=150000), 'the level IS the answer: a choice must solve'
    assert bad(portfolios=['141', '145']), 'one portfolio: a choice compares plans'

    # the branches whose message names the cause, so a typo fails before a solve
    def why(want, **extra):
        msg = refusal(**extra)[0]
        assert msg and want in msg, (want, msg)

    why('needs mode retirement', mode='accumulation')
    why('must be numbers', compare_choice={'dimension': 'retirement_age',
                                           'values': ['a']})
    why('must be numbers', compare_choice={'dimension': 'retirement_age',
                                           'values': [None]})
    # a comma-separated string is accepted and normalised to floats
    msg, spec = refusal(compare_choice={'dimension': 'retirement_age',
                                        'values': '60, 65'})
    assert msg is None and spec['compare_choice']['values'] == [60.0, 65.0], (msg, spec)
    assert isinstance(spec['compare_choice']['values'][0], float)
    why('at least one value', compare_choice={'dimension': 'retirement_age',
                                              'values': ' , '})
    # claiming age needs a benefit: stored in your details, or on the spec
    claim = {'dimension': 'ss_claim_age', 'values': [67, 70]}
    with _temp_investor(None):
        why('needs Social Security', compare_choice=claim)
        assert not bad(compare_choice=claim, social_security=_SS), \
            "a spec's own benefit should satisfy it"
        why('needs Social Security', compare_choice=claim, social_security=None)
    with _temp_investor(_SS):
        assert not bad(compare_choice=claim)
        why('62 to 70', compare_choice=dict(claim, values=[61, 70]))


def test_the_sweep_and_the_choice_apply_a_plan_choice_the_same_way():
    """Two code paths applying 'retire at 60' is how they come to disagree."""
    import sensitivity as sens
    for dim, v in (('retirement_age', 60), ('contributions', 1.5)):
        cfg = runner.build_config(sens._base_spec())
        via_choice = runner.apply_choice(cfg, dim, v)
        sens.apply_dim(cfg, dim, v)
        assert cfg == via_choice, dim


def test_a_retirement_age_choice_runs_end_to_end():
    out = _smoke('retirement', spend='solve', policy='guyton_klinger',
                 portfolios=['141'],
                 compare_choice={'dimension': 'retirement_age', 'values': [62, 64]})
    assert 'Traceback' not in out, out[-2000:]
    assert 'RETIREMENT CHOICES' in out, out[-2000:]
    sec = out.split('RETIREMENT CHOICES', 1)[1]
    rows = [l.strip() for l in sec.splitlines()]
    assert any(l.startswith('retire at 62') for l in rows), sec[:1500]
    assert any(l.startswith('retire at 64 (yours)') for l in rows), sec[:1500]
    assert 'compared with retire at 64' in sec, sec[:1500]
    assert 'POLICY COMPARISON' not in out, 'a plan choice is not a policy table'
    # the context block describes YOUR option, at YOUR age
    assert 'YOUR PLAN IN CONTEXT   at 64' in out, out[-3000:]
    # different retirement ages share one set of calendar-age bands (N4)
    shape = out.split('SPENDING BY LIFE STAGE', 1)[-1]
    assert 'SPENDING BY LIFE STAGE' in out and '62-63' in shape, out[-2500:]
    assert 'working' in shape, 'retire at 64 is still working at 62-63'


def test_a_claiming_age_choice_keeps_the_life_stage_table():
    with _temp_investor(_SS):          # never the user's own Social Security
        out = _smoke('retirement', spend='solve', policy='guyton_klinger',
                     portfolios=['141'],
                     compare_choice={'dimension': 'ss_claim_age', 'values': [67, 70]})
    assert 'Traceback' not in out, out[-2000:]
    assert 'RETIREMENT CHOICES' in out
    assert 'SPENDING BY LIFE STAGE' in out, 'same start age: the bands line up'


def test_a_choice_value_given_twice_is_refused():
    """Runs are keyed by their tag, so 'retire at 65' twice is ONE row: the
    second solve silently overwrites the first and the table shows two
    options where three were asked for."""
    spec = {'portfolios': ['141'],
            'compare_choice': {'dimension': 'retirement_age', 'values': [65, 65, 70]}}
    try:
        runner.parse_spec(json.dumps(spec), {'141': {}})
    except ValueError:
        return
    raise AssertionError('a repeated value was accepted')


def test_a_choice_outside_what_is_possible_is_refused():
    cfg = runner.default_config()

    def refused(dim, values):
        spec = {'portfolios': ['141'],
                'compare_choice': {'dimension': dim, 'values': values}}
        try:
            runner.parse_spec(json.dumps(spec), {'141': {}})
        except ValueError:
            return True
        return False

    assert refused('retirement_age', [cfg['age']]), 'retiring at your current age'
    assert refused('retirement_age', [cfg['age'] + cfg['time_horizon_years']])
    assert refused('contributions', [-0.5])
    if cfg.get('social_security'):
        assert refused('ss_claim_age', [61]) and refused('ss_claim_age', [71])
        assert not refused('ss_claim_age', [62, 70])


def test_paths_that_all_spend_one_amount_are_fixed_spending_even_off_the_level():
    """constant_real below your Social Security benefit spends the BENEFIT on
    every path once it starts, so no path spends exactly the plan — and the
    return-model check scored median spending, printed 0.0% for every cell,
    and could not fail. What makes spending uninformative is that every path
    spends the same, not that it spends the level."""
    from montecarlo import return_views as rv
    n = 400
    u = np.random.default_rng(5).permutation(n) / n
    book = lambda p: {'per_path_mean': np.full(n, 51_307.0), 'depleted': u < p,
                      'spend': 40_000.0}
    assert report_retirement._spending_is_fixed(book(0.0))
    t = rv.regrets({'factor': {'a': book(0.02), 'b': book(0.05)}}, ['a', 'b'])
    assert t['metric'] == 'ruin', t['metric']


def _views_check(monkey_run_view, mine='factor', spend='solve', res=None):
    """return_views.check on synthetic one-plan results, children replaced.
    `res` replaces the default solved-shape results (a fixed-spend run has no
    'common' key)."""
    from montecarlo import return_views as rv
    P = _P()
    cfg = runner.build_config({'mode': 'retirement', 'portfolios': ['141'],
                               'spend': spend, 'policy': 'guyton_klinger'})
    rng = np.random.default_rng(9)
    market = rng.normal(0, 20_000, 300)
    if res is None:
        res = {g: {'common': {'spend': 160_000.0,
                              'per_path_mean': 400_000.0 + m + market + rng.normal(0, 500, 300),
                              'depleted': np.zeros(300, bool), 'success_rate': 0.9}}
               for g, m in (('141', 0.0), ('145', -8_000.0))}
    real = (rv.run_view, rv.current_view)
    rv.run_view = monkey_run_view
    rv.current_view = lambda *a: mine
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            summary = rv.check(['141', '145'], res, cfg, P)
    finally:
        rv.run_view, rv.current_view = real
    return summary, buf.getvalue(), res


def _fake_books(n, plans, level, offset=0.0):
    return {'books': {g: {'per_path_mean': np.full(n, 400_000.0 + offset) + np.arange(n),
                          'depleted': np.zeros(n, bool), 'success_rate': 0.9,
                          'spend': level} for g in plans}}


def test_a_failed_view_is_reported_and_the_rest_still_compare():
    """One child dying must not cost a run that has already spent its solve —
    and must not vanish quietly either, or a four-view table reads as five."""

    def fake(key, cfg, plans, level, n):
        if key == 'historical':
            raise RuntimeError('child exited 1: boom')
        return _fake_books(n, plans, level)

    summary, out, _ = _views_check(fake)
    assert '⚠ historical: failed' in out and 'boom' in out, out[:800]
    assert 'IF THE RETURN MODEL IS WRONG' in out, out[:800]
    assert set(summary['views']) == {'factor', 'intl keeps lagging',
                                     'intl catches up', 'no factors'}, summary['views']
    # the record keeps real booleans, not the strings default=str makes of numpy's
    back = json.loads(json.dumps(summary, default=str))
    assert all(type(v) is bool for v in back['views']['factor']['holds'].values()), back


def test_a_run_under_an_unlisted_model_keeps_its_own_column():
    """A run made under, say, 'value dead' is none of the five views. It must
    still be a column, and all five views must run as children."""
    ran = []

    def fake(key, cfg, plans, level, n):
        ran.append(key)
        return _fake_books(n, plans, level)

    from montecarlo import return_views as rv
    summary, out, _ = _views_check(fake, mine=None)
    assert sorted(ran) == sorted(v.key for v in rv.VIEWS), ran
    assert rv.THIS_RUN in summary['views'], summary['views']
    assert rv.THIS_RUN in out, out[:800]


def test_the_return_model_check_reads_a_fixed_spend_run_not_only_a_solved_one():
    """`_views_check` only ever built `{'common': ...}` results, so check()'s
    other branch — a run at a fixed dollar amount, whose plan is the config's
    target and whose books come from `raw.depleted` and the top-level
    `per_path_mean` — was never executed."""
    n = 300
    rng = np.random.default_rng(4)
    u = rng.permutation(n) / n

    def fixed_res(per_path, ruin):
        return {g: {'spend': 160_000.0, 'success_rate': 1.0 - p,
                    'per_path_mean': per_path(g),
                    'raw': {'depleted': u < p}}
                for g, p in zip(('141', '145'), ruin)}

    def books(per_path, ruin):
        def fake(key, cfg, plans, level, n_):
            assert level == 160_000.0, 'the plan is the config target when nothing was solved'
            return {'books': {g: {'per_path_mean': per_path(g), 'depleted': u < p,
                                  'success_rate': 1.0 - p, 'spend': level}
                              for g, p in zip(('141', '145'), ruin)}}
        return fake

    # spending that varies by path: scored on spending, and no 'certified' claim
    varied = lambda g: 100_000.0 + 20_000.0 * rng.normal(size=n) - (5_000.0 if g == '145' else 0.0)
    ruin = (0.05, 0.10)
    res = fixed_res(varied, ruin)
    summary, out, _ = _views_check(books(varied, ruin), spend=160_000.0, res=res)
    assert summary is not None and summary['level'] == 160_000.0, summary
    assert summary['paths'] == n and summary['metric'] == 'spend', summary
    assert 'IF THE RETURN MODEL IS WRONG' in out, out[:800]
    assert 'certified at' not in out, 'nothing was solved, so nothing was certified'
    assert 'YOUR PLAN under each view: 141 at $160k/yr' in out, out[-900:]

    # every path spends the same amount: the metric becomes the chance of ruin
    same = lambda g: np.full(n, 51_307.0)
    res = fixed_res(same, ruin)
    summary, out, _ = _views_check(books(same, ruin), spend=160_000.0, res=res)
    assert summary['metric'] == 'ruin', summary['metric']
    assert summary['pick'] == '141' and summary['worst']['145'] > 0.04, summary
    # the loaded book is the run's own raw.depleted: 141 ruins 5%, 145 10%
    assert abs(summary['views']['factor']['regret']['145'] - 0.05) < 0.01, summary['views']
    assert 'extra chance of running out' in out, out[:1200]


# ---------------------------------------------------------------------------
# THE MENU — three questions, named by what they compare
# ---------------------------------------------------------------------------

def _by_default(fn, *args, picks=None, preset=None):
    """Call a prompt with every question answered by its default, except the
    ask_choice questions named in `picks`. Returns (result, menus seen)."""
    seen = {}

    def fake_choice(q, opts, default=1):
        seen.setdefault(q, opts)
        return (picks or {}).get(q, opts[default - 1][0])

    real = (runner.ask, runner.ask_choice, runner.load_preset)
    runner.ask = lambda _p, d='': d
    runner.ask_choice = fake_choice
    if preset is not None:
        runner.load_preset = lambda name='last': preset
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            out = fn(*args)
    finally:
        runner.ask, runner.ask_choice, runner.load_preset = real
    return out, seen


def _typed(keys):
    """builtins.input replaced by a script that RAISES when exhausted; an
    EOFError entry is end-of-input. Lets the REAL prompt primitives run."""
    it = iter(keys)

    def _input(_prompt=''):
        try:
            got = next(it)
        except StopIteration:
            raise AssertionError('the prompt asked for more input than scripted')
        if got is EOFError:
            raise EOFError
        return got
    return _input


def _run_typed(fn, keys, *args):
    import builtins
    real = builtins.input
    builtins.input = _typed(keys)
    try:
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            out = fn(*args)
    finally:
        builtins.input = real
    return out, buf.getvalue()


def test_the_real_prompt_primitives_parse_what_is_typed():
    """Every other prompt test swaps ask/ask_choice for a fake, so the primitive
    every menu depends on, and portfolios' 'show all' path, never ran."""
    opts = [('a', 'alpha', 'first'), ('b', 'beta', ''), ('c', 'gamma', 'third')]
    got, out = _run_typed(runner.ask_choice, ['3'], 'Which?', opts)
    assert got == 'c' and '> 1  alpha   first' in out and '  3  gamma   third' in out, out
    got, out = _run_typed(runner.ask_choice, ['0', '99', 'abc', '2'], 'Which?', opts)
    assert got == 'b' and out.count('? enter 1-3') == 3, (got, out)
    # blank and end-of-input both take the default
    assert _run_typed(runner.ask_choice, [''], 'Which?', opts, 3)[0] == 'c'
    assert _run_typed(runner.ask_choice, [EOFError], 'Which?', opts, 2)[0] == 'b'

    got, out = _run_typed(runner.ask_multi, ['x', '4', '1,3'], 'Which?', opts)
    assert got == ['a', 'c'], got
    assert out.count('? enter 1-3, or several separated by commas') == 2, out
    assert _run_typed(runner.ask_multi, ['2,2,1'], 'Which?', opts)[0] == ['b', 'a'], \
        'a repeat is dropped, the order given is kept'

    P = _P()
    ids = sorted(P)[:2]
    got, out = _run_typed(runner.ask_portfolios,
                          ['a', 'zzz', 'a', '', 'a', ','.join(ids)], P)
    assert got == ids, got
    assert '? unknown: zzz' in out and '? unknown: (none given)' in out, out
    assert out.count('show all') == 3, 'each bad entry returns to the menu'
    assert _run_typed(runner.ask_portfolios, ['1'], P)[0] == [
        runner.suggested_bases(P)[0][0]]


def test_the_menu_names_the_three_questions_by_what_they_compare():
    """'Which portfolio? / Which plan?' told you nothing you could act on. Each
    entry names what changes, and its hint carries YOUR numbers: the portfolios
    you last compared and your own retirement age."""
    P = _P()
    last = {'mode': 'retirement', 'portfolios': ['161', '165']}
    spec, seen = _by_default(runner.prompt_run, P, preset=last,
                           picks={'What do you want to compare?': 'again'})
    assert spec == last
    opts = seen['What do you want to compare?']
    assert [k for k, _, _ in opts] == ['portfolios', 'choices', 'sensitivity',
                                       'again', 'config'], opts
    by = {k: (label, hint) for k, label, hint in opts}
    assert by['portfolios'][0] == 'Portfolios' and '161' in by['portfolios'][1], by
    assert by['choices'][0] == 'Retirement choices', by
    ra = runner.default_config()['retirement_age']
    assert f"{ra:g}" in by['choices'][1], by['choices']
    assert 'Market' in by['sensitivity'][0], by['sensitivity']
    assert by['config'][0] == 'Your details', by['config']


def test_portfolios_asks_retirement_or_accumulation_first():
    """'Allocations (accumulation only)' was a top-level entry; it compares
    portfolios too, so it is a follow-up under Portfolios."""
    P = _P()
    real = runner.ask_portfolios
    runner.ask_portfolios = lambda _P: ['141', '145']
    try:
        spec, seen = _by_default(runner.prompt_run, P, preset={},
                               picks={'What do you want to compare?': 'portfolios',
                                      'Compare them over': 'accumulation'})
    finally:
        runner.ask_portfolios = real
    assert spec['mode'] == 'accumulation', spec
    assert spec['portfolios'] == ['141', '145']


def test_the_portfolios_route_builds_specs_the_runner_accepts():
    """The main interactive route had never been driven past 'Compare them
    over'. Each branch — derisk yes / no / custom, solve or a fixed amount,
    one policy or several — must build a spec that --spec accepts and that
    resolves to strategies, or a key-name drift dies only when a person
    reaches that prompt."""
    P = _P()
    ra = None
    real = runner.ask_portfolios, runner.ask_multi
    runner.ask_portfolios = lambda _P: ['141', '165']
    try:
        with _temp_investor(_SS):
            ra = runner.default_config()['retirement_age']
            derisk = f'Derisk at retirement ({ra})?'

            def build(**picks):
                spec, _ = _by_default(runner.prompt_run, P, preset={}, picks={
                    'What do you want to compare?': 'portfolios',
                    'Compare them over': 'retirement', **picks})
                runner.parse_spec(json.dumps(spec), P)
                return spec, runner.resolve_strategies(spec, P)[0]

            spec, strats = build(**{derisk: 'yes'})
            assert spec['derisk_at'] == ra and spec['spend'] == 'solve', spec
            assert f'141>141+10@{ra:g}' in strats, strats

            spec, strats = build(**{derisk: 'no', 'Spending?': 'fixed'})
            assert spec['spend'] == 200_000.0 and 'derisk_at' not in spec, spec
            assert list(strats) == ['141', '165'], strats

            spec, strats = build(**{derisk: 'custom'})
            assert spec['custom'] == ['141', '165'], spec

            # Several rules: the policy question answered 1,3, the rest default.
            runner.ask_multi = lambda q, o, *a, **k: (
                ['guyton_klinger', 'vpw'] if o is runner.POLICY_OPTIONS
                else real[1](q, o, *a, **k))
            spec, _ = build(**{derisk: 'no'})
            assert spec['compare_policies'] == ['guyton_klinger', 'vpw'], spec
            assert spec['gk_response'] == 'fast', spec
    finally:
        runner.ask_portfolios, runner.ask_multi = real


def test_retirement_choices_builds_a_spec_the_runner_accepts():
    """What the prompt builds, --spec must accept: the prompt is only a way of
    writing a spec, and a spec the validator refuses would die mid-run."""
    P = _P()
    real = runner.ask_portfolios
    runner.ask_portfolios = lambda _P: ['161']
    try:
        spec, seen = _by_default(runner.prompt_choices, P,
                               picks={'Which choice?': 'retirement_age'})
        pol, _ = _by_default(runner.prompt_choices, P,
                           picks={'Which choice?': 'policy'})
    finally:
        runner.ask_portfolios = real
    ra = runner.default_config()['retirement_age']
    assert spec['compare_choice'] == {'dimension': 'retirement_age',
                                      'values': [ra - 5, ra, ra + 5]}, spec
    assert spec['portfolios'] == ['161'] and spec['spend'] == 'solve', spec
    runner.parse_spec(json.dumps(spec), P)
    # the withdrawal rule is a plan choice too, through the policy fan-out
    assert 'compare_choice' not in pol, pol
    assert len(pol['compare_policies']) == 4, pol
    runner.parse_spec(json.dumps(pol), P)


def test_market_assumptions_offers_only_what_you_do_not_choose():
    """Your choices moved to their own route. What is left is the world —
    markets, and the parts of your life you do not decide."""
    import sensitivity as sens
    _, seen = _by_default(runner.prompt_sensitivity,
                        picks={'Which kind of assumption?': 'back'})
    groups = [k for k, _, _ in seen['Which kind of assumption?']]
    assert 'plan' not in groups, groups
    assert 'life' in groups, groups
    life = {n for n, d in sens.DIMENSIONS.items() if d.group == 'life'}
    assert life == {'horizon', 'ltc_shock', 'ss_scale'}, life


def test_zzz_the_suite_left_the_engine_as_it_found_it():
    """No test may leave an engine knob moved. Runs last, by name.

    Three flow-accounting tests saved TEN engine globals around
    `_zero_vol_engine`, which sets TWELVE — so RF_REAL_STD and INFLATION_STD
    stayed at 0.0 for every test after them. Every later end-to-end check ran
    a model with the cash-rate and inflation bands switched off, and passed.
    It surfaced only because the replay line started printing the knobs a run
    moved, and printed `MC_INFLATION_STD=0.0 MC_RF_REAL_STD=0.0` for a default
    run. Compared against the values at IMPORT, so a suite run deliberately
    under an env var is not flagged."""
    moved = {a: (v, getattr(pa, a)) for a, v in _ENGINE_KNOBS.items()
             if getattr(pa, a) != v}
    assert not moved, f"a test left engine knobs moved (was, now): {moved}"
    # The tables too: a rebound dict carrying a fake ticker is invisible to the
    # scalar knobs above.
    tables = {n: set(getattr(pa, n)) for n in _ENGINE_TABLE_KEYS
              if set(getattr(pa, n)) != _ENGINE_TABLE_KEYS[n]}
    assert not tables, f"a test left engine tables changed: {sorted(tables)}"


_CONFIG_FINGERPRINT = _config_fingerprint()
_ENGINE_KNOBS = {a: getattr(pa, a) for a in diagnostics.DEFAULT_ASSUMPTIONS
                 if hasattr(pa, a)}

_ENGINE_TABLE_KEYS = {n: set(getattr(pa, n))
                      for n in ('REGIME_TRANSITIONS', 'VOLATILITY_REGIMES')}

def test_the_public_snapshot_refuses_a_personal_figure():
    """tools/publish_public.py is the only thing between this repo's private
    history and a public one. Its guard must catch a real balance in every way
    Python and prose write it (prose abbreviates, and a balance total and a
    monthly benefit written that way slipped past the first version in README's
    sample menu), a private block or private
    import that survived, a ledger link and a broken link — and must not flag a
    clean tree."""
    import json, tempfile
    try:
        import publish_public as pp
    except ImportError:
        return          # the public snapshot ships without its own publisher
    # spelled in pieces, so this file does not trip the guard it tests
    OPEN, CLOSE, LEDGER = '<!-- ' + 'private -->', '<!-- /' + 'private -->', 'FIND' + 'INGS.md'
    IMPORT = '@private/' + 'SETTLED.md'
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        inv = d / 'investor.json'
        inv.write_text(json.dumps({'accounts': {'a': {'balance': 123456.0},
                                                'b': {'balance': 20000.0}},
                                   'social_security': {'annual_at_fra': 60000.0}}))
        tokens = pp.personal_tokens(inv)
        want = {'123456', '123,456', '123_456', '$123.5k', '$123k',     # a balance
                '143,456', '$143k',                                     # the total
                '60000', '5,000', '$5.0k'}                              # SS, yearly and monthly
        assert want <= tokens, want - tokens
        text = f'keep\n{OPEN}\nsecret\n{CLOSE}\nsee [the ledger]({LEDGER})\n\n{IMPORT}\n'
        out = pp.transform(text)
        assert 'secret' not in out and 'private' not in out, out
        assert '(RESULTS.md)' in out and LEDGER not in out, out
        assert IMPORT not in out and '(../docs/RESULTS.md)' in out, out
        TAG = 'FIND' + 'INGS'       # a ledger entry cited by id in a code comment
        out = pp.transform(f'({TAG} N10) and {TAG} 2026-10-24; {TAG} still quotes it')
        assert out == ('(ledger entry N10, unpublished) and ledger entry 2026-10-24, '
                       'unpublished; the ledger (unpublished) still quotes it'), out
        tree = d / 'tree'
        (tree / 'docs').mkdir(parents=True)
        (tree / 'docs' / 'RESULTS.md').write_text('ok\n')
        (tree / 'a.md').write_text('[r](docs/RESULTS.md) 1,234,567 and $1234.5k\n')
        assert pp.find_leaks(tree, tokens) == [], 'a clean tree was refused'
        # 123456.5 is flagged on purpose: a balance stored with cents prints that way
        for planted in ('x = 123_456', 'had $123,456', 'SS 60000/yr', 'b 123456.5',
                        'about $123k', 'SS $5.0k/mo', 'in all $143k', 'had 123,456, then', IMPORT,
                        OPEN, 'see ' + LEDGER, '[gone](nope.md)'):
            (tree / 'b.md').write_text(planted + '\n')
            assert pp.find_leaks(tree, tokens), f'missed: {planted!r}'
        (tree / 'b.md').write_text('1123456, 0.123456 and 123,456,789 are other numbers\n')
        assert pp.find_leaks(tree, tokens) == [], 'flagged a number that only contains the token'
        # every file that reads as text is scanned, whatever its suffix: ERD.html
        # once carried a ledger path past an allowlist of suffixes
        for name in ('b.html', 'NOTES'):
            (tree / name).write_text('about $123k\n')
            assert pp.find_leaks(tree, tokens), f'missed a figure in {name}'
            (tree / name).unlink()
        (tree / '.claude' / 'private').mkdir(parents=True)
        assert pp.find_leaks(tree, tokens), 'the private CLAUDE.md import shipped'
    # History: a doc once carried DEFAULTS' OLD balances, which today's file
    # no longer holds. Real figures are kept; round placeholders are not, or a
    # token like 20,000 would refuse half the repo.
    old = "'balance': 123456.5, 'x': 1, 'balance': 20_000.0, 'annual_at_fra': 61_200.0"
    got = sorted(round(x, 2) for x in pp.amounts_in(old))
    assert got == [5100.0, 61200.0, 123456.5, 143456.5], got
    assert pp.amounts_in("'balance': 150000.0, 'balance': 25_000.0") == [], 'a placeholder became a token'
    assert '143,456' in pp.personal_tokens(d / 'missing.json', pp.amounts_in(old))


CHECKS =[v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]


def main():
    print("Pure-logic checks")
    print("=" * 60)
    # CHECKS is built from globals(), so a second `def` with an existing test
    # name silently replaces the first and drops that test. Compare against the
    # definitions in the source, which a rebinding cannot hide.
    names = re.findall(r'^def (test_\w+)', Path(__file__).read_text(), re.M)
    if len(names) != len(CHECKS):
        dupes = sorted({n for n in names if names.count(n) > 1})
        print(f"✗ {len(names)} test definitions in the source but {len(CHECKS)} "
              f"collected — duplicated names: {dupes or 'none found'}")
        return 1
    failures = []
    for fn in CHECKS:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
        except Exception as e:
            print(f"  [FAIL] {fn.__name__}")
            print(f"         {type(e).__name__}: {e}")
            failures.append(fn.__name__)
    print("=" * 60)
    if failures:
        print(f"✗ {len(failures)}/{len(CHECKS)} checks failed: {failures}")
        return 1
    print(f"✓ All {len(CHECKS)} pure-logic checks passed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
