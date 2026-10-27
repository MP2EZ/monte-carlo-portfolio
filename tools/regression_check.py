"""
Engine regression check (synthetic golden master).
===================================================

Guards `engine.py` against accidental behavior changes when it's edited.
Three checks, all on a FIXED SYNTHETIC universe (hardcoded params below) so the
baseline tracks ENGINE LOGIC, not the quarterly-regenerated data:

  1. Golden master — run N deterministic paths (single-process; each path seeds
     42+sim_id), summarize final values, compare to tools/regression_baseline.json.
  2. Determinism   — the same path (sim_id=0) run twice is byte-identical.
  3. Analytic anchor — a single asset with zero vol and no contributions must
     compound to initial*(1+mu/12)^months exactly.

Usage:
    python3 tools/regression_check.py            # exit 0 = matches baseline
    python3 tools/regression_check.py --update   # re-bless the baseline (after an
                                                 # INTENTIONAL engine change)

If you change the SYNTH_* constants below, you must --update the baseline.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from montecarlo import engine as pa

BASELINE = Path(__file__).resolve().parent / 'regression_baseline.json'
RTOL = 1e-9

# --- Synthetic universe (the engine's "unit under test") ----------------------------
SYNTH_TICKERS = ['A', 'B']
SYNTH = {
    'TICKERS': SYNTH_TICKERS,
    'EXPECTED_RETURNS': {'A': 0.08, 'B': 0.04},
    'PARAM_UNCERTAINTY_STD': {'A': 0.012, 'B': 0.004},
    'VOLATILITY_REGIMES': {
        'A': {'low': 0.10, 'medium': 0.16, 'high': 0.26},
        'B': {'low': 0.03, 'medium': 0.05, 'high': 0.09},
    },
    'REGIME_TRANSITIONS': {
        'A': {'low': {'low': 0.70, 'medium': 0.25, 'high': 0.05},
              'medium': {'low': 0.20, 'medium': 0.50, 'high': 0.30},
              'high': {'low': 0.10, 'medium': 0.40, 'high': 0.50}},
        'B': {'low': {'low': 0.75, 'medium': 0.20, 'high': 0.05},
              'medium': {'low': 0.25, 'medium': 0.55, 'high': 0.20},
              'high': {'low': 0.15, 'medium': 0.45, 'high': 0.40}},
    },
}
SYNTH_CORR = {
    'CORRELATION_BASELINE': np.array([[1.0, 0.30], [0.30, 1.0]]),
    'CORRELATION_LOW': np.array([[1.0, 0.20], [0.20, 1.0]]),
    'CORRELATION_MEDIUM': np.array([[1.0, 0.35], [0.35, 1.0]]),
    'CORRELATION_HIGH': np.array([[1.0, 0.60], [0.60, 1.0]]),
}

# Deliberately NOT the same as either synthetic ticker's chain: if the engine
# ever went back to deriving the market state from the tickers, this would move.
SYNTH_MARKET_TRANSITIONS = {
    'low':    {'low': 0.90, 'medium': 0.08, 'high': 0.02},
    'medium': {'low': 0.10, 'medium': 0.80, 'high': 0.10},
    'high':   {'low': 0.02, 'medium': 0.18, 'high': 0.80},
}

# How much a ticker's row leans toward 'high' in each market state. Stands in
# for the real conditional estimate, whose measured signature is the same shape:
# P(medium -> high) runs 0.019 / 0.048 / 0.271 as the market goes low to high.
SYNTH_COUPLING = {'low': -0.5, 'medium': 0.0, 'high': 1.0}


def _tilt_row(row, tilt):
    """Shift probability mass toward (tilt > 0) or away from (tilt < 0) 'high'."""
    if not tilt:
        return dict(row)
    out = dict(row)
    if tilt > 0:
        moved = min(tilt, 1.0) * (out['low'] + out['medium'])
        scale = 1.0 - min(tilt, 1.0)
        out = {'low': out['low'] * scale, 'medium': out['medium'] * scale,
               'high': out['high'] + moved}
    else:
        moved = min(-tilt, 1.0) * out['high']
        out = {'low': out['low'] + moved, 'medium': out['medium'],
               'high': out['high'] - moved}
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items()}
SYNTH_CONFIG = {
    'age': 40,
    'retirement_age': 65,
    'time_horizon_years': 10,
    'initial_value': 100_000.0,
    'rebalancing_strategy': 'none',
    'contribution_schedule': {
        'frequency': 'annual',
        'accounts': {'main': {'annual_amount': 6000.0, 'growth_rate': 0.0, 'stop_age': None}},
    },
}
SYNTH_ALLOC = {'A': 0.60, 'B': 0.40}
N_PATHS = 300

# --- Decumulation universe ----------------------------------------------------------
# The accumulation master above retires at 65 but stops at 50, so withdrawals
# never fire: the entire retirement half of the engine — every policy, both
# sourcing rules, the depletion test, the spending series — was unpinned. This
# config retires mid-horizon so all of it runs.
SYNTH_WD_CONFIG = {
    'age': 60,
    'retirement_age': 65,
    'time_horizon_years': 30,          # ages 60-89; withdrawals from 65
    'initial_value': 1_000_000.0,
    'rebalancing_strategy': 'none',
    'contribution_schedule': {
        'frequency': 'annual',
        'accounts': {'main': {'annual_amount': 0.0, 'growth_rate': 0.0,
                              'stop_age': None}},
    },
    'withdrawal': {
        'start_age': 65,
        'policy': 'guyton_klinger',    # overridden per case
        'sourcing': 'excess',          # overridden per case
        'target_real_annual': 50_000.0,
        'floor': 0.85,
        # Pinned: the golden master tests the engine, not which preset is the
        # default, so moving the default must not move the baseline.
        'gk_response': 'slow',
    },
}
# Every policy, and both sourcing rules on one of them — `excess` rebalances as
# it withdraws while `pro_rata` leaves weights untouched, which is a different
# code path and measurably a different return series.
WD_CASES = [('constant_real', 'excess'), ('fixed_pct', 'excess'),
            ('guyton_klinger', 'excess'), ('vpw', 'excess'),
            ('guyton_klinger', 'pro_rata')]
N_WD_PATHS = 80

# --- Feature cases ------------------------------------------------------------------
# Production rebalances quarterly, can glide, claims Social Security, can carry a
# care-cost window and shares estimation errors through the premia. Neither
# master above had any of it (both run rebalancing 'none'), and a partial-
# rebalance bug that no-op'd every overweight breach lived in that gap. Every
# case here rebalances, as production does; each adds one feature on top.
FEATURE_CASES = {
    'rebalance': {},
    'glide': {'plan': ((60, {'A': 0.60, 'B': 0.40}), (70, {'A': 0.30, 'B': 0.70}))},
    'social_security': {'social_security': {'annual_at_fra': 30_000.0,
                                            'claim_age': 67, 'benefit_scale': 1.0}},
    'ltc': {'ltc': {'annual_amount': 80_000.0, 'start_age': 80, 'years': 3}},
    'shared_errors': {'shared': True},
}
SYNTH_BANDS = {'A': {'relative': 0.15, 'absolute': 0.05},
               'B': {'relative': 0.15, 'absolute': 0.05}}

# --- Analytic anchor universe (single asset, zero vol) ------------------------------
ANCHOR_MU = 0.08
ANCHOR_YEARS = 10
ANCHOR_INIT = 100_000.0


def _apply(tickers, exp, pstd, vol_regimes, transitions, corr):
    # The synthetic universe injects rates directly, so they are already whatever
    # terms the test intends. Pin INFLATION to 0 so the golden master measures
    # engine mechanics only and does not move when the production inflation
    # assumption is retuned — that constant has its own targeted checks in
    # check_logic.py.
    pa.INFLATION = 0.0
    pa.TICKERS = tickers
    pa.EXPECTED_RETURNS = exp
    pa.PARAM_UNCERTAINTY_STD = pstd
    pa.VOLATILITY_REGIMES = vol_regimes
    pa.REGIME_TRANSITIONS = dict(transitions)
    # The engine needs a MARKET chain; give the synthetic universe one.
    pa.REGIME_TRANSITIONS[pa.MARKET_REGIME_KEY] = SYNTH_MARKET_TRANSITIONS
    # ...and per-ticker chains CONDITIONED on it, so the golden master covers the
    # coupled branch production takes rather than the unconditional fallback.
    for tkr, rows in transitions.items():
        for market, tilt in SYNTH_COUPLING.items():
            pa.REGIME_TRANSITIONS[(tkr, market)] = {
                frm: _tilt_row(row, tilt) for frm, row in rows.items()}
    pa.CORRELATION_BASELINE = corr['CORRELATION_BASELINE']
    pa.CORRELATION_LOW = corr['CORRELATION_LOW']
    pa.CORRELATION_MEDIUM = corr['CORRELATION_MEDIUM']
    pa.CORRELATION_HIGH = corr['CORRELATION_HIGH']
    pa.REBALANCING_BANDS = dict(pa.REBALANCING_BANDS, **SYNTH_BANDS)


def _apply_synth():
    _apply(SYNTH['TICKERS'], SYNTH['EXPECTED_RETURNS'],
           SYNTH['PARAM_UNCERTAINTY_STD'], SYNTH['VOLATILITY_REGIMES'],
           SYNTH['REGIME_TRANSITIONS'], SYNTH_CORR)


def _run_synth():
    _apply_synth()
    runs = [pa.run_single_simulation((SYNTH_ALLOC, SYNTH_CONFIG, i))
            for i in range(N_PATHS)]
    finals = np.array([r[0] for r in runs])
    returns = np.array([r[2] for r in runs])
    # Wealth stats alone cannot see a flow-accounting bug: contributions booked as
    # returns leave every dollar path untouched (that is how C1 survived). Pin the
    # reported return series too.
    # Explicit hurdle: the baseline must pin what it measures, not inherit a
    # default that can be retuned elsewhere (same reasoning as INFLATION above).
    sortino = pa.aggregate_sortino_stats(returns, mar_annual=0.0)
    cagr = pa.time_weighted_cagr(returns, SYNTH_CONFIG['time_horizon_years'])
    return {
        'mean': float(np.mean(finals)),
        'median': float(np.median(finals)),
        'p10': float(np.percentile(finals, 10)),
        'p90': float(np.percentile(finals, 90)),
        'sortino_median': float(sortino['sortino_median']),
        'twr_cagr_median': float(np.median(cagr)),
        'monthly_return_mean': float(np.mean(returns)),
    }


def _run_decum():
    """Golden master for the retirement engine: one case per policy."""
    _apply_synth()
    out = {}
    for policy, sourcing in WD_CASES:
        cfg = dict(SYNTH_WD_CONFIG)
        cfg['withdrawal'] = dict(SYNTH_WD_CONFIG['withdrawal'],
                                 policy=policy, sourcing=sourcing)
        runs = [pa.run_single_simulation((SYNTH_ALLOC, cfg, i))
                for i in range(N_WD_PATHS)]
        finals = np.array([r[0] for r in runs])
        spending = np.array([r[4]['spending'] for r in runs])
        depleted = np.array([r[4]['depletion_age'] is not None for r in runs])
        per_path = spending.mean(axis=1)
        out[f'{policy}/{sourcing}'] = {
            # first-year draw pins the seeding rule: a self-directed policy sets
            # its own, a controlled one takes the plan's target
            'first_year_median': float(np.median(spending[:, 0])),
            'mean_spend_median': float(np.median(per_path)),
            'mean_spend_p10': float(np.percentile(per_path, 10)),
            'last_year_median': float(np.median(spending[:, -1])),
            'final_median': float(np.median(finals)),
            'depletion_rate': float(depleted.mean()),
        }
    return out


def _run_features():
    """Golden master for what production turns on: one case per feature."""
    _apply_synth()
    out = {}
    for case, f in FEATURE_CASES.items():
        cfg = dict(SYNTH_WD_CONFIG, rebalancing_strategy='quarterly_annual')
        cfg['withdrawal'] = dict(SYNTH_WD_CONFIG['withdrawal'])
        if 'social_security' in f:
            cfg['social_security'] = f['social_security']
        if 'ltc' in f:
            cfg['withdrawal']['ltc'] = f['ltc']
        saved = pa.PARAM_ERRORS, pa.PREMIUM_SHARES
        if f.get('shared'):
            # A loads on the first two premia, B on the third; the rest of each
            # band stays the fund's own.
            k = len(pa.PREMIUM_KEYS)
            pa.PARAM_ERRORS = 'shared'
            pa.PREMIUM_SHARES = {'A': np.eye(k)[0] * 0.6 + np.eye(k)[1] * 0.3,
                                 'B': np.eye(k)[2] * 0.5}
        else:
            pa.PARAM_ERRORS, pa.PREMIUM_SHARES = 'independent', {}
        try:
            runs = [pa.run_single_simulation((f.get('plan', SYNTH_ALLOC), cfg, i))
                    for i in range(N_WD_PATHS)]
        finally:
            pa.PARAM_ERRORS, pa.PREMIUM_SHARES = saved
        per_path = np.array([np.mean(r[4]['spending']) for r in runs])
        stats = [r[3] for r in runs]
        out[case] = {
            'final_median': float(np.median([r[0] for r in runs])),
            'mean_spend_median': float(np.median(per_path)),
            'mean_spend_p10': float(np.percentile(per_path, 10)),
            'depletion_rate': float(np.mean([r[4]['depletion_age'] is not None
                                             for r in runs])),
            'partial_mean': float(np.mean([st['partial_count'] for st in stats])),
            'full_mean': float(np.mean([st['full_count'] for st in stats])),
            'traded_mean': float(np.mean([st['traded'] for st in stats])),
        }
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    update = '--update' in argv
    problems = []

    def check(ok, label, detail=''):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if not ok:
            if detail:
                print(f"         {detail}")
            problems.append(label)

    print("Engine regression check (synthetic golden master)")
    print("=" * 60)

    # 2. Determinism: same path twice -> identical.
    _apply_synth()
    p1 = pa.run_single_simulation((SYNTH_ALLOC, SYNTH_CONFIG, 0))[1]
    p2 = pa.run_single_simulation((SYNTH_ALLOC, SYNTH_CONFIG, 0))[1]
    check(np.array_equal(p1, p2), "determinism: identical path for fixed seed")

    # 3. Analytic anchor: zero-vol single asset compounds to closed form.
    _apply(['Z'], {'Z': ANCHOR_MU}, {'Z': 0.0},
           {'Z': {'low': 0.0, 'medium': 0.0, 'high': 0.0}},
           {'Z': {r: {'low': 0.0, 'medium': 1.0, 'high': 0.0} for r in ('low', 'medium', 'high')}},
           {k: np.array([[1.0]]) for k in SYNTH_CORR})
    anchor_cfg = dict(SYNTH_CONFIG, time_horizon_years=ANCHOR_YEARS, initial_value=ANCHOR_INIT,
                      contribution_schedule={'frequency': 'annual',
                                             'accounts': {'main': {'annual_amount': 0.0,
                                                                   'growth_rate': 0.0, 'stop_age': None}}})
    # The shared cash-rate shock randomises mu itself, which is exactly what it
    # is for — and it makes a DETERMINISTIC compounding anchor impossible. This
    # check pins the arithmetic, not the uncertainty, so the shock is switched
    # off here and nowhere else. (Leaving it on silently turned the anchor into
    # a one-sample noise test: it read 200,105 against an expected 221,964.)
    # BOTH shared shocks: the cash rate and inflation each randomise mu, so a
    # deterministic anchor needs both off. Adding one and forgetting the other
    # is how this check silently became a noise test twice in one day.
    _saved = (pa.RF_REAL_STD, pa.INFLATION_STD)
    pa.RF_REAL_STD = pa.INFLATION_STD = 0.0
    try:
        final = pa.run_single_simulation(({'Z': 1.0}, anchor_cfg, 7))[0]
    finally:
        pa.RF_REAL_STD, pa.INFLATION_STD = _saved
    expected = ANCHOR_INIT * (1 + ANCHOR_MU / 12) ** (ANCHOR_YEARS * 12)
    check(np.isclose(final, expected, rtol=RTOL),
          "analytic anchor: zero-vol compounding matches closed form",
          f"engine={final:.6f} expected={expected:.6f}")

    # 1. Golden master vs committed baseline — accumulation and decumulation.
    stats = _run_synth()
    decum = _run_decum()
    features = _run_features()
    if update and problems:
        print("  [ -- ] NOT re-blessed: determinism or the analytic anchor failed")
    elif update:
        BASELINE.write_text(json.dumps(
            {'n_paths': N_PATHS, 'time_horizon_years': SYNTH_CONFIG['time_horizon_years'],
             'stats': stats, 'n_wd_paths': N_WD_PATHS, 'decumulation': decum,
             'features': features},
            indent=2) + '\n')
        print(f"  [ -- ] baseline re-blessed -> {BASELINE.name}")
    elif not BASELINE.exists():
        check(False, "regression baseline present",
              f"missing {BASELINE.name} — run with --update to create it")
    else:
        saved = json.loads(BASELINE.read_text())
        base = saved['stats']
        diffs = {k: (base.get(k), stats[k]) for k in stats
                 if not np.isclose(base.get(k, np.nan), stats[k], rtol=RTOL)}
        # Both directions: a statistic dropped from the run is also a change,
        # and the gate would otherwise pass with less coverage than it blessed.
        diffs.update({k: (base[k], 'missing from run') for k in base if k not in stats})
        check(not diffs, "golden master matches baseline (accumulation)",
              f"drifted: {diffs} (intentional? re-bless with --update)")

        base_d = saved.get('decumulation')
        if base_d is None:
            check(False, "decumulation baseline present",
                  "baseline predates withdrawal coverage — re-bless with --update")
        else:
            wd_diffs = {}
            for case, got in decum.items():
                was = base_d.get(case)
                if was is None:
                    wd_diffs[case] = 'new case'
                    continue
                for k, v in got.items():
                    if not np.isclose(was.get(k, np.nan), v, rtol=RTOL):
                        wd_diffs[f'{case}.{k}'] = (was.get(k), v)
                wd_diffs.update({f'{case}.{k}': (was[k], 'missing from run')
                                 for k in was if k not in got})
            wd_diffs.update({case: 'missing from run' for case in base_d
                             if case not in decum})
            check(not wd_diffs, "golden master matches baseline (decumulation)",
                  f"drifted: {wd_diffs} (intentional? re-bless with --update)")

        base_f = saved.get('features')
        if base_f is None:
            check(False, "feature baseline present",
                  "baseline predates feature coverage — re-bless with --update")
        else:
            f_diffs = {}
            for case, got in features.items():
                was = base_f.get(case)
                if was is None:
                    f_diffs[case] = 'new case'
                    continue
                for k, v in got.items():
                    if not np.isclose(was.get(k, np.nan), v, rtol=RTOL):
                        f_diffs[f'{case}.{k}'] = (was.get(k), v)
                f_diffs.update({f'{case}.{k}': (was[k], 'missing from run')
                                for k in was if k not in got})
            f_diffs.update({case: 'missing from run' for case in base_f
                            if case not in features})
            check(not f_diffs, "golden master matches baseline (features)",
                  f"drifted: {f_diffs} (intentional? re-bless with --update)")

    print("=" * 60)
    if problems:
        print(f"✗ {len(problems)} problem(s) — the engine's behavior changed.")
        return 1
    print("✓ All checks passed — engine behavior unchanged.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
