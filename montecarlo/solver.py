# ============================================================================
# Inverse solver — what can this strategy actually sustain?
# ============================================================================
# The forward question ("I need $X; will it last?") requires knowing X. The
# inverse question ("at 90% confidence, what is the largest X?") does not, and is
# the better basis for comparing strategies: it holds CONFIDENCE constant rather
# than a dollar bar, so each strategy reports its own maximum and the resulting
# figures are directly comparable.
#
# It also sidesteps the flaw in percentage-of-balance comparisons, where a
# high-growth portfolio sets itself an easier dollar target and looks safer than
# it is.
#
# Success:
#     never depleted AND average realised spending >= `floor` x target
#
# The design doc specified "never cut below floor in ANY year". That was tried
# and is wrong — see the comment in evaluate(). Averaging is what makes the
# variable policies scoreable at all; it reduces to a pure depletion test for
# constant_real, which spends exactly the target until the money runs out. The
# floor clause is still what stops the search running to infinity for
# fixed_pct, guyton_klinger and vpw, none of which can deplete by construction.
# ============================================================================

import atexit
import copy
import math
from multiprocessing import Pool, cpu_count

import numpy as np

from montecarlo import engine
from montecarlo import instrument
from montecarlo import policies

VERBOSE = False            # set by run.py --debug; prints the crossing probes
# Verification steps onto the true curve. Most solves need 0 or 1 — the loop
# stops as soon as it is within 1pp, or when a step would move under 1%. A steep
# curve needs more: portfolio 102 went 82.1% -> 84.9% -> 87.2% and was still
# gaining ~2.5pp a step when a cap of 3 cut it off, leaving a row reporting a
# plan level under a heading that claims 90%. Each extra step costs one
# verification and is only paid where it is needed.
MAX_NEWTON = 6
PARALLEL_THRESHOLD = 400   # below this the Pool setup costs more than it saves
FLOOR_GRID = (0.70, 0.85, 1.00)   # adequacy thresholds re-scored for free


_POOL = None


def _pool():
    """One long-lived Pool for the whole process.

    Creating a Pool per evaluation is very expensive here: under the `spawn`
    start method every worker re-imports engine.py, which reloads the parameter
    JSON and four 25x25 correlation matrices. A single solve is ~15 evaluations
    (a fit-based crossing search plus verification), so an 8-strategy run was
    creating over a hundred pools and spawning workers into the hundreds — dominating the runtime and
    making the time estimate meaningless.
    """
    global _POOL
    if _POOL is None:
        _POOL = Pool(cpu_count())
    return _POOL


def shutdown_pool():
    """Release the worker pool. Safe to call more than once."""
    global _POOL
    if _POOL is not None:
        _POOL.terminate()
        _POOL.join()
        _POOL = None


atexit.register(shutdown_pool)


def _paths(plan, config, n_sims, parallel, light=True, seed_offset=0):
    """Run n_sims paths. `light` drops the per-path arrays the solver never reads,
    which is most of the wall-clock cost when running through a Pool.

    `seed_offset` shifts the sim ids, and therefore the seeds. Everything in a
    normal run uses 0, so paths are reproducible and two strategies compared at
    the same offset share their random numbers — which is what makes the paired
    bootstrap valid. A NON-ZERO offset draws a genuinely independent sample, and
    exists for one reason: the solver sets the plan level on its verification
    sample, so asking that sample whether the level is right is asking a question
    it cannot fail. See `tools/solver_calibration.py`.
    """
    worker = engine.run_single_simulation_light if light else engine.run_single_simulation
    args = [(plan, config, seed_offset + i) for i in range(n_sims)]
    # Counted HERE rather than in `simulate`, which is not actually the choke
    # point: accumulation calls _paths directly, so an allocation run reported a
    # single undifferentiated phase and no per-path cost at all, and could never
    # recalibrate the estimate it had just tested.
    pooled = parallel and n_sims >= PARALLEL_THRESHOLD
    instrument.record_paths(n_sims, pooled)
    if pooled:
        return _pool().map(worker, args)
    return [worker(a) for a in args]


def stage_bounds(start_age, end_age, n_years, n_stages=3):
    """Split the retirement span into `n_stages` bands, labelled by real ages.

    Thirds rather than hardcoded decades: retirement age is a config field the
    sensitivity sweep varies (60/65/70), and a fixed 65-74/75-84/85-94 grid would
    silently mislabel every other case. At the default 65-95 the thirds ARE
    decades, which is how most people think about it.

    Lives here rather than in the report because bands are an ADEQUACY concept
    first — `adequacy='thirds'` scores on them — and the report is a consumer.
    No longer the default rule: thirds of a longer span are longer bands, so
    their strictness moved with the horizon and the retirement age. See
    `adequacy_windows`.
    Clipped to the array actually returned, so a shorter horizon cannot index
    past the end.
    """
    span = max(int(round(end_age - start_age)), 1)
    span = min(span, int(n_years))
    if span < n_stages:
        return [(0, span, f"{int(start_age)}-{int(start_age) + span - 1}")]
    edges = [round(i * span / n_stages) for i in range(n_stages + 1)]
    out = []
    for i in range(n_stages):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        out.append((lo, hi, f"{int(start_age) + lo}-{int(start_age) + hi - 1}"))
    return out


def stage_stats(spending, bounds):
    """Median and 10th-percentile of each band's AVERAGE annual spending.

    The percentile is taken per band, not along one path: the question a retiree
    asks is "how bad could my seventies be", not "trace me the single path that
    was tenth-worst over thirty years". So a row is not one scenario, and the
    report says so.
    """
    out = []
    for lo, hi, label in bounds:
        seg = spending[:, lo:hi]
        if seg.size == 0:
            out.append((label, float('nan'), float('nan')))
            continue
        per_path = seg.mean(axis=1)
        out.append((label, float(np.median(per_path)),
                    float(np.percentile(per_path, 10))))
    return out


ADEQUACY = ('lifetime', 'thirds', 'window')

# Records written before 2026-10-21 call the thirds rule 'per_band'. Same rule,
# same numbers: the name changed when a second banded rule arrived.
ADEQUACY_ALIASES = {'per_band': 'thirds'}

# How long a stretch the 'window' rule averages over. Ten years absorbs a
# single crash, as the bands did, and at the default 65-95 it is the length the
# thirds had — so the change is WHERE the stretch may start, not how long it is.
ADEQUACY_WINDOW_YEARS = 10

# Default: 'window', since 2026-10-21 (ledger entry N11, unpublished). The worst ROLLING
# ten-year average must clear the floor.
#
# 'thirds' was the default before it, and it is still the reason a lifetime
# mean is not: a level you cannot count on in your sixties is not a level you
# plan around. Measured on portfolio 140, thirds cost 28-30% of the lifetime
# headline for guyton_klinger, fixed_pct and vpw, 58% for vpw+5M, and left
# constant_real exactly unchanged.
#
# But a third is a fraction of the span, so its LENGTH — and with it the
# rule's strictness — moved with the horizon and the retirement age. 165
# certified $123.0k planning to 100 and $120.05k to 95 on paths whose first 30
# years were bit-identical, and retiring at 60 read ~4pp better against 65 than
# a fixed-length stretch says. A window has one length everywhere, and a
# longer plan holds every window a shorter one does, so it can never certify
# more. Solved on 165 at 65-95 it costs 7.3% ($120,050 → $111,334): ~1pp of
# success at every spend, on a success curve that guyton_klinger keeps flat.
#
# The binding stretch still differs by policy — vpw's is early (its rate rises
# with age), guyton_klinger's mostly the first decade (portfolio alone to 70) —
# and `--debug` reports where it starts.
DEFAULT_ADEQUACY = 'window'


def adequacy_mode(mode):
    """The canonical name of an adequacy rule, or ValueError for an unknown one."""
    mode = ADEQUACY_ALIASES.get(mode, mode)
    if mode not in ADEQUACY:
        raise ValueError(f"adequacy must be one of {ADEQUACY}, got {mode!r}")
    return mode


def adequacy_windows(start_age, end_age, n_years, years=ADEQUACY_WINDOW_YEARS):
    """Every rolling `years`-long window of the retirement, as (lo, hi, label).

    A span shorter than the window is ONE window over all of it: there is no
    ten-year stretch to find, and scoring the years that exist is the nearest
    honest reading. Clipped to the array actually returned, like
    `stage_bounds`.
    """
    span = max(int(round(end_age - start_age)), 1)
    span = min(span, int(n_years))
    width = min(int(years), span)
    return [(lo, lo + width, f"{int(start_age) + lo}-{int(start_age) + lo + width - 1}")
            for lo in range(span - width + 1)]


def adequacy_bounds(config, n_years, mode):
    """The stretches a banded rule takes its minimum over: thirds or windows."""
    start_age = config['withdrawal']['start_age']
    end_age = config['age'] + config['time_horizon_years']
    if adequacy_mode(mode) == 'window':
        return adequacy_windows(start_age, end_age, n_years)
    return stage_bounds(start_age, end_age, n_years)


def adequacy_unit(mode):
    """What the reports call the stretch that must clear the floor."""
    if adequacy_mode(mode) == 'window':
        return f"{ADEQUACY_WINDOW_YEARS}-year stretch"
    return 'life stage'


def adequacy_statistic(spending, config, mode=DEFAULT_ADEQUACY):
    """Per-path spending level that must clear `floor x spend`.

    'lifetime'  the mean over the whole retirement. Tolerant of temporary dips,
                and reduces to a pure depletion test for constant_real. But it is
                SHAPE-BLIND: a bad first decade can be repaid by a rich ninth, and
                measurably is — the share of PASSING paths that run short in the
                first band was 0.0% for constant_real, 3.8% guyton_klinger, 8.5%
                vpw, 12.5% vpw+5M, monotone in how back-loaded the policy is.

    'thirds'    the WORST third's mean ('per_band' in records before
                2026-10-21). You live through the first decade whether or not
                the ninth repays it, so the level you can count on is the one
                every stretch clears. Bands rather than years on purpose: the
                every-YEAR form was tried and is a drawdown test in disguise —
                for fixed_pct it reads "the portfolio never drops 15% below its
                age-65 value", which almost every path fails, and it drove the
                solver to $0. A ten-year mean absorbs a single crash; a sustained
                shortfall still fails it.

    'window'    the WORST rolling ADEQUACY_WINDOW_YEARS-year mean — the default.
                The thirds' shape-awareness without their dependence on the
                span: a third of 35 years is a more lenient test than a third
                of 25. A span under the window is scored as one window.
    """
    mode = adequacy_mode(mode)
    if mode == 'lifetime':
        return spending.mean(axis=1)
    bounds = adequacy_bounds(config, spending.shape[1], mode)
    per_band = np.stack([spending[:, lo:hi].mean(axis=1) for lo, hi, _ in bounds])
    return per_band.min(axis=0)


def guaranteed_income_row(config, n_years):
    """The guaranteed-income schedule, by year of retirement. Deterministic.

    Social Security is paid regardless of what the portfolio does, so it is a
    known function of age, not a simulated quantity — which is exactly why it
    can be subtracted back out of realised spending to see what the PORTFOLIO
    actually contributed.
    """
    ss_cfg = config.get('social_security')
    if not ss_cfg:
        return np.zeros(n_years)
    start = config['withdrawal']['start_age']
    return np.array([policies.ss_income(start + i, ss_cfg) for i in range(n_years)],
                    dtype=float)


def portfolio_share_of_binding_band(spending, config, mode=DEFAULT_ADEQUACY):
    """Per path: what fraction of the BINDING band's spending the portfolio funded.

    THE LEVEL CAN BE SET BY SOCIAL SECURITY RATHER THAN BY THE PORTFOLIO, and
    nothing said so. Measured on the run of 2026-09-12: under `vpw reserve 10M`
    every one of twelve strategies solved to between 0.80x and 1.00x the annual
    benefit, and two of them — 140 and 180, different portfolios — returned
    the same figure BIT-IDENTICALLY, because both are exactly the benefit / 0.85.
    In the bottom decile a large reserve leaves VPW asking for less than the
    benefit, `drawn = max(0, requested - ss)` is then zero for a whole decade,
    and the band mean is the benefit exactly. The solver was reporting a
    property of Social Security as a property of the portfolio, ranked in the
    same table as levels that genuinely measure one.

    Returns a share in [0, 1]; 0.0 means the portfolio funded none of it.
    """
    n_paths, n_years = spending.shape
    ss = guaranteed_income_row(config, n_years)
    if not ss.any():
        return np.ones(n_paths)
    portfolio = np.maximum(spending - ss[None, :], 0.0)
    mode = adequacy_mode(mode)
    if mode == 'lifetime':
        total = spending.mean(axis=1)
        return np.divide(portfolio.mean(axis=1), total,
                         out=np.ones(n_paths), where=total > 0)
    bounds = adequacy_bounds(config, n_years, mode)
    per_band = np.stack([spending[:, lo:hi].mean(axis=1) for lo, hi, _ in bounds])
    pf_band = np.stack([portfolio[:, lo:hi].mean(axis=1) for lo, hi, _ in bounds])
    binding = per_band.argmin(axis=0)
    idx = np.arange(n_paths)
    tot, pf = per_band[binding, idx], pf_band[binding, idx]
    return np.divide(pf, tot, out=np.ones(n_paths), where=tot > 0)


def simulate(plan, config, spend, n_sims=500, parallel=True, full=False,
             seed_offset=0):
    """Run `n_sims` paths at a given annual real spend. No scoring.

    Split from `evaluate` so a policy whose paths do not depend on `spend` can be
    simulated ONCE and scored at whatever level the closed form dictates, instead
    of being bisected toward a threshold that is analytically known.
    """
    cfg = copy.deepcopy(config)
    cfg['withdrawal'] = dict(cfg['withdrawal'])
    cfg['withdrawal']['target_real_annual'] = float(spend)

    # `full` keeps the monthly return series so Sortino and CAGR can be
    # reported. Used for the single final verification, never for the ~13
    # search evaluations, where the marshalling cost is what made solves slow.
    runs = _paths(plan, cfg, n_sims, parallel, light=not full,
                  seed_offset=seed_offset)
    return {
        'spending': np.array([r[4]['spending'] for r in runs], dtype=float),
        'depleted': np.array([r[4]['depletion_age'] is not None for r in runs]),
        'finals': np.array([r[0] for r in runs]),
        'v_at_retirement': np.array(
            [r[4].get('v_at_retirement') or np.nan for r in runs], dtype=float),
        # Rebalancing counters, present only on FULL runs — the light worker
        # returns None here. Carried so the retirement report can show turnover,
        # which until now stopped at the solver and was silently discarded.
        'rebalancing_raw': [r[3] for r in runs if r[3] is not None],
        'depletion_ages': [r[4]['depletion_age'] for r in runs],
        'returns': (np.array([r[2] for r in runs]) if full else None),
        # Per-path diagnostics. Three floats and three ints each — cheaper to
        # marshal than the depletion ages beside them — so they ride on every
        # run and `--debug` decides whether to SHOW them, rather than a flag
        # deciding whether to compute them. A module global could not do that
        # job anyway: workers re-import under spawn.
        'path_mu': np.array([r[4].get('path_mu', np.nan) for r in runs],
                            dtype=float),
        'rf_shock': np.array([r[4].get('rf_shock', np.nan) for r in runs],
                             dtype=float),
        'infl_shock': np.array([r[4].get('infl_shock', np.nan) for r in runs],
                               dtype=float),
        'regime_months': [r[4].get('regime_months') for r in runs],
    }


def _marginal_portfolio_share(spending, config, adequacy, stat, threshold):
    """Portfolio share for the paths that DECIDE the level, not the average path.

    The level is set by the paths sitting on the adequacy threshold, so the
    median path's funding mix is the wrong question — a book can be
    portfolio-funded in the median and entirely benefit-funded at the boundary,
    which is the case this exists to catch.
    """
    share = portfolio_share_of_binding_band(spending, config, adequacy)
    if share.size == 0:
        return 1.0
    k = max(1, min(share.size, int(round(0.05 * share.size))))
    near = np.argsort(np.abs(stat - threshold))[:k]
    return float(np.median(share[near]))


def score(raw, config, spend, floor=0.85, adequacy=DEFAULT_ADEQUACY):
    """Score already-simulated paths at a given spend. Pure, no simulation."""
    spending, depleted = raw['spending'], raw['depleted']
    if spending.size == 0:
        # No retirement years to score. A partial result here crashed every
        # renderer later on a missing key, far from the cause.
        raise ValueError('no retirement years to score: withdrawals start at '
                         'or after the end of the plan')

    stat = adequacy_statistic(spending, config, adequacy)
    success = (~depleted) & (stat >= floor * spend - 1e-9)

    # Success at other floors is FREE — same paths, different threshold — so the
    # run can report whether the answer survives a stricter adequacy standard
    # without simulating anything again.
    by_floor = {f: float(((~depleted) & (stat >= f * spend - 1e-9)).mean())
                for f in FLOOR_GRID}

    per_path_mean = spending.mean(axis=1)
    out = {
        'spend': float(spend),
        'success_rate': float(success.mean()),
        'success_by_floor': by_floor,
        'adequacy': adequacy_mode(adequacy),
        'depletion_rate': float(depleted.mean()),
        'median_realised': float(np.median(per_path_mean)),
        'p10_realised': float(np.percentile(per_path_mean, 10)),
        'worst_cut': (float(np.median(1.0 - spending.min(axis=1) / spend))
                      if spend > 0 else 0.0),
        # THE BAD CASE, not the median path. `worst_cut` is a median across
        # paths, and under guyton_klinger or constant_real the median path never
        # cuts at all — so the column read exactly 100% for every strategy in
        # every run, while ROBUSTNESS in the same report said one path in five
        # takes a cut. Two true statements that read as a contradiction, because
        # nothing said the first was the median. p90 of the cut is the 10th
        # percentile of the year, which is how every other 'low case' column
        # here is defined.
        'worst_cut_bad': (float(np.percentile(1.0 - spending.min(axis=1) / spend,
                                              90))
                          if spend > 0 else 0.0),
        'median_bequest': float(np.median(raw['finals'])),
        'balance_at_retirement': (
            {q: float(np.nanpercentile(raw['v_at_retirement'], q))
             for q in (10, 50, 90)}
            if raw.get('v_at_retirement') is not None
            and np.isfinite(raw['v_at_retirement']).any() else None),
        'per_path_mean': per_path_mean,
        'adequacy_stat': stat,
        # What the PORTFOLIO funded in the binding band, for the path sitting on
        # the decision boundary. Near zero means this level is Social Security's,
        # not the portfolio's — see portfolio_share_of_binding_band.
        'portfolio_share': _marginal_portfolio_share(
            spending, config, adequacy, stat, floor * spend),
        'spending': spending,
        'depletion_ages': raw['depletion_ages'],
        'start_age': config['withdrawal']['start_age'],
    }
    if raw.get('returns') is not None:
        rets = raw['returns']
        out['sortino'] = engine.aggregate_sortino_stats(rets, mar_annual=0.0)[
            'all_sortino_ratios']
        out['sortino_median'] = float(np.median(out['sortino']))
        out['twr_cagr'] = engine.time_weighted_cagr(
            rets, config['time_horizon_years'])
        # ANNUALISED VOLATILITY. Free — the monthly return series is already in
        # hand for sortino and cagr, and the accumulation path has reported this
        # since it existed while the retirement path never did. It is the one
        # column that says what the RIDE was like rather than where it ended,
        # which is what a reader comparing a 100/0 book to a derisked one
        # actually wants to see.
        out['annual_vol'] = float(np.std(rets) * np.sqrt(12) * 100)
    raw_rb = raw.get('rebalancing_raw')
    if raw_rb:
        try:
            out['rebalancing'] = engine.aggregate_rebalancing_stats(
                raw_rb, config['time_horizon_years'])
        except Exception:          # diagnostics must never take a run down
            pass
    return out


def evaluate(plan, config, spend, n_sims=500, floor=0.85, parallel=True,
             full=False, adequacy=DEFAULT_ADEQUACY, seed_offset=0):
    """Run `n_sims` paths at a given annual real spend and score them."""
    raw = simulate(plan, config, spend, n_sims, parallel, full, seed_offset)
    out = score(raw, config, spend, floor, adequacy)
    # KEEP THE PATHS A --debug RUN NEEDS, and only on a FULL evaluation. The
    # ~13 search evaluations pass full=False, so this attaches once per
    # strategy — to the verification run whose numbers the report actually
    # printed. Scoring an earlier probe would describe a level nobody saw.
    if full:
        out['raw'] = raw
    return out


def _isotonic_decreasing(y):
    """Pool adjacent violators into a non-increasing fit.

    True success is non-increasing in spend; a noisy estimate need not be. PAVA
    imposes the shape we know holds and, in doing so, AVERAGES the noise instead
    of letting a single low read stand — which is what bisection did, ratcheting
    its ceiling down on one unlucky comparison and never recovering.
    """
    vals, wts = [], []
    for v in map(float, y):
        vals.append(v)
        wts.append(1.0)
        while len(vals) > 1 and vals[-2] < vals[-1]:
            v2, w2 = vals.pop(), wts.pop()
            v1, w1 = vals.pop(), wts.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2)
    out = []
    for v, w in zip(vals, wts):
        out.extend([v] * int(round(w)))
    return np.array(out)


def _crossing(xs, ys, target):
    """Where the success curve crosses `target`, by local least squares.

    Interpolating between the two bracketing points would use two noisy reads
    and throw the rest away. Near the crossing the curve is close to linear, so
    a line through the points in that neighbourhood uses all of them and its
    error falls with their count. This matters because the curve is GENTLE
    there: on portfolio 140 guyton_klinger moves about 0.09pp of success per
    $1k, so a 1.5pp read error is worth ~$17k of spend. Averaging is the only
    thing that makes the estimate usable.
    """
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    near = [i for i in range(len(xs)) if abs(ys[i] - target) <= 0.06]
    if len(near) < 3:
        near = sorted(range(len(xs)), key=lambda i: abs(ys[i] - target))[:4]
    near = sorted(set(near))
    def _clip(x, slope):
        # THIRD RETURN VALUE: did the answer hit an edge of the sampled window?
        # A clipped crossing is not an estimate of anything — it is the window
        # saying "the answer is somewhere past here". Accepting one because its
        # gap happens to fall inside the convergence tolerance is how seven of
        # fifteen strategies reported exactly $200,000 on 2026-09-13, all of
        # them at 89% against a 90% target. Ten flagged rows, none above 90%:
        # p = 0.001 under a fair coin, i.e. a systematic overstatement rather
        # than sampling noise.
        lo_e, hi_e = float(xs[0]), float(xs[-1])
        return (float(np.clip(x, lo_e, hi_e)), float(slope),
                -1 if x < lo_e else (1 if x > hi_e else 0))

    X, Y = xs[near], ys[near]
    if len(X) >= 2 and np.ptp(Y) > 1e-9:
        slope, intercept = np.polyfit(X, Y, 1)
        if slope < -1e-12:
            return _clip((target - intercept) / slope, slope)
    # Outside the reads' range the answer is past an edge, and must say so:
    # interpolating a flat run of reads returned whichever tied point sorted
    # last, an interior spend that never reopened the bracket.
    if target > ys.max():
        return _clip(-np.inf, 0.0)
    if target < ys.min():
        return _clip(np.inf, 0.0)
    order = np.argsort(ys)
    return _clip(np.interp(target, ys[order], xs[order]), 0.0)


def _self_directed(config):
    """Does this policy ignore the plan's target, making paths independent of it?"""
    return config.get('withdrawal', {}).get(
        'policy', 'guyton_klinger') in policies.SELF_DIRECTED


def _quantile_level_se(stat, confidence, floor, n_boot=400, seed=12345):
    """Sampling error of a closed-form level, by resampling the statistic."""
    stat = np.asarray(stat, dtype=float)
    finite = stat[np.isfinite(stat)]
    if finite.size < 20 or floor <= 0:
        return float('nan')
    rng = np.random.default_rng(seed)
    q = 100.0 * (1.0 - confidence)
    draws = [np.percentile(rng.choice(stat, stat.size, replace=True), q)
             for _ in range(n_boot)]
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    return float(draws.std() / floor) if draws.size > 1 else float('nan')


def usable_slope(slope, xs, ys):
    """A negative slope for the Newton correction, or the input unchanged.

    `_crossing` returns 0.0 whenever its LOCAL linear fit is not decreasing — a
    flat or upward-sloping neighbourhood, which noise produces regularly right
    at the crossing. The verification guard tested that value, so a zero
    silently disabled the whole correction and left the level at the
    search-sample crossing, on the curve the solver knows sits ~2pp off. It also
    left the bracket clip uncorrected, since that clip is only harmless WHEN
    NEWTON RUNS — the part that had never been checked.

    Falls back to the secant across the WHOLE search curve. Success is
    monotonically decreasing in spend, so that secant is negative whenever the
    curve moved at all, and the loop replaces it with a verification-sample
    secant after the first step.
    """
    if slope < -1e-12:
        return float(slope)
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if xs.size < 2 or xs[-1] <= xs[0]:
        return float(slope)
    secant = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
    return float(secant) if secant < -1e-12 else float(slope)


# THE PATH COUNTS, IN ONE PLACE. They lived as literals in three files and
# two of them disagreed: this signature said 500/2000 while every real caller
# passed 400/1500, so the dead defaults were the ones a reader found first.
# They belong here because `solve_sustainable_spend` is what spends them.
SEARCH_SIMS = 400
VERIFY_SIMS = 1500
FIXED_SIMS = 2000


def _print_curve(pts, confidence, n_sims=None):
    """The (spend, success) probes, marked against the target.

    `<` and `>` say which side of the target each probe fell, so a curve that
    never crosses — the shape that produces a clipped answer — reads as all one
    symbol. `=` is within one binomial standard error, which is the resolution
    the probes actually have: 2.4pp at 150 paths, 1.5pp at 400, 0.8pp at 1500.
    Pinning it at one number would call a 150-path probe decisive.
    """
    rows = sorted(set((round(float(x), 2), float(y)) for x, y in pts))
    if not rows:
        return
    se = (math.sqrt(confidence * (1 - confidence) / n_sims)
          if n_sims else 0.015)
    print(f"        probes: " + '  '.join(
        f"{x / 1e3:,.0f}k{'=' if abs(y - confidence) <= se else ('>' if y > confidence else '<')}{y:.0%}"
        for x, y in rows))
    # BRACKETED means the curve CROSSES the target, which is a question about
    # the raw values — not about clearing a standard-error band. Keying the
    # warning on the band called a 91% probe "within noise of 90%", found no
    # probe strictly above, and warned that a properly bracketed solve was
    # extrapolated. The band belongs on the per-probe marker, where it says how
    # much resolution a single probe has, and nowhere else.
    ys = [y for _, y in rows]
    if max(ys) < confidence or min(ys) > confidence:
        # DESCRIBE, do not alarm. `bracket_floor` is deliberately slack so the
        # walk does not stop early on a noisy read, which means the window's low
        # end can sit a hair under the target while the search is working
        # exactly as designed — observed on four strategies that then verified
        # at 90.0-90.4%. The crossing really is extrapolated and that is worth
        # seeing; what settles it is the verification line underneath, so point
        # there rather than implying the answer is wrong.
        side = 'below' if max(ys) < confidence else 'above'
        edge = max(ys) if max(ys) < confidence else min(ys)
        print(f"        · every probe sits {side} {confidence:.0%} "
              f"(nearest {edge:.1%}) — the crossing is extrapolated off the "
              f"window's edge;\n          the verification step below is what "
              f"settles it")


def solve_sustainable_spend(plan, config, confidence=0.90, floor=0.85,
                            search_sims=None, verify_sims=None, n_points=6,
                            hi=None, parallel=True, verbose=False,
                            adequacy=DEFAULT_ADEQUACY):
    """The largest real annual spend meeting `confidence`.

    Two routes, chosen by whether the policy consumes the target.

    CLOSED FORM, for a self-directed policy (vpw). Its withdrawals are a function
    of the balance alone, so the paths do not move when the target does — verified
    bit-identical at a $200k and a $450k bar. Success is then simply
    P(statistic >= floor x spend), and the largest spend meeting `confidence` is

        spend* = quantile(statistic, 1 - confidence) / floor

    exactly. Bisecting for it was not merely ~14x wasteful, it was LESS accurate:
    the search ran at `search_sims` while the reported confidence came from the
    `verify_sims` run, so a solve aimed at 90% reported 91.5%. Taking the quantile
    on the verification sample removes that mismatch by construction.

    BISECTION, otherwise. constant_real spends the target, fixed_pct turns it into
    a rate and guyton_klinger sets its guardrails off that rate, so for those the
    paths genuinely move with the target and success is monotonically decreasing
    in it. Searches on a reduced path count and verifies at full count — roughly
    5x cheaper than solving at full precision throughout.
    """
    search_sims = SEARCH_SIMS if search_sims is None else search_sims
    verify_sims = VERIFY_SIMS if verify_sims is None else verify_sims
    if _self_directed(config):
        with instrument.timed('closed form'):
            raw = simulate(plan, config, 1.0, verify_sims, parallel, full=True)
        stat = adequacy_statistic(raw['spending'], config, adequacy)
        # A depleted path can never be adequate at any spend, so it must sit
        # below every quantile rather than contribute its spending level.
        stat = np.where(raw['depleted'], -np.inf, stat)
        q = float(np.percentile(stat, 100.0 * (1.0 - confidence)))
        spend = max(q / floor, 0.0) if np.isfinite(q) else 0.0
        result = score(raw, config, spend, floor, adequacy)
        result['confidence_target'] = confidence
        result['floor'] = floor
        result['solved_by'] = 'closed_form'
        # Exact by construction: the quantile IS the crossing, taken on the same
        # sample the level is reported against.
        result['newton_steps'] = 0
        result['gap'] = float(result['success_rate'] - confidence)
        result['converged'] = True
        # HOW PRECISE THE LEVEL IS. The closed form is exact ON ITS SAMPLE, which
        # is not the same as precise: it is a quantile of a dispersed statistic,
        # so a different 1,500 paths give a different answer. Bootstrapped rather
        # than derived, because a quantile's SE depends on the density at the
        # quantile and there is no closed form for it here.
        result['level_se'] = _quantile_level_se(stat, confidence, floor)
        return result

    # Bracket upward from a rough guess, so the caller need not know the scale.
    #
    # The break must allow for SAMPLING ERROR, and used to not. One 400-path
    # evaluation has SE ~1.5pp at 90%, so a level that truly passes fails the
    # search roughly half the time it sits near the threshold — and the old
    # bisection could never leave [lo, hi], so that single unlucky test capped
    # the answer permanently. Measured on portfolio 140: guyton_klinger at $100k
    # scored 89.2% on 400 paths and 91.2% on 1500, closing the bracket at $100k
    # when the true 90% level was $109k. Breaking two standard errors below the
    # target keeps the ceiling above the crossing for one extra doubling.
    se = math.sqrt(max(confidence * (1.0 - confidence), 1e-12)
                   / max(int(search_sims), 1))
    bracket_floor = confidence - 2.0 * se

    lo = 0.0
    probes = []
    if hi is None:
        # BRACKET FROM AN INFORMED GUESS, NOT FROM $50,000.
        #
        # Doubling from 50k puts the probes at 50/100/200/400k, so the window
        # handed to the sampler is a FACTOR OF TWO wide and lands wherever the
        # doubling happens to stop. For a $7.4M book the first three probes
        # carry no information at all, and the window near a ~$190k answer was
        # [100k, 200k] — 100k wide, with the answer near its edge. Measured on
        # a 15-strategy run AFTER both clipping fixes: 7 of 15 levels still came
        # back sitting on an edge.
        #
        # The balance at retirement is INDEPENDENT of the spend level — it is
        # the balance before the first withdrawal — so one probe at any level
        # reveals it, and 4% of it is a far better starting guess than 50k. That
        # is the same 4% rule the report already prints as context.
        with instrument.timed('bracketing'):
            first = float(config.get('withdrawal', {}).get(
                'target_real_annual') or 200_000.0)
            # Through `evaluate`, not `simulate`: the balance rides along in the
            # scored result, and going around evaluate would bypass every test
            # that stubs it.
            r0 = evaluate(plan, config, first, search_sims, floor, parallel,
                          adequacy=adequacy)
            probes.append((first, r0['success_rate']))
            bal = (r0.get('balance_at_retirement') or {}).get(50)
            guess = max(0.04 * bal, 1_000.0) if bal and bal > 0 else first

            # Walk geometrically from the guess until two probes straddle the
            # target. 1.5x rather than 2x: the guess should be within a factor
            # of two, so a coarser step only widens the window it hands on.
            def _probe(x):
                r = evaluate(plan, config, x, search_sims, floor, parallel,
                             adequacy=adequacy)['success_rate']
                probes.append((float(x), r))
                return r

            r = _probe(guess) if abs(guess - first) > 1e-9 else probes[0][1]
            if r >= bracket_floor:
                lo, hi = guess, guess * 1.5
                for _ in range(10):
                    if _probe(hi) < bracket_floor:
                        break
                    lo, hi = hi, hi * 1.5
            else:
                lo, hi = guess / 1.5, guess
                for _ in range(10):
                    if _probe(lo) >= bracket_floor:
                        break
                    lo, hi = lo / 1.5, lo

    # ESTIMATE the crossing rather than bisect toward it. Bisection compared a
    # noisy sample against a bare threshold and moved a BOUND on the result, so
    # every unlucky read was absorbed permanently and the answer came out low —
    # measured -17% for guyton_klinger and -8% for fixed_pct against the true
    # 90% level.
    #
    # Two passes, because the curve has two very different shapes. Under real
    # volatility it is gentle near the crossing (guyton_klinger moves ~0.09pp of
    # success per $1k), so the estimate lives or dies on averaging noise. On a
    # deterministic universe it is a STEP, and a line fitted through a step is
    # meaningless — there the second pass, which resamples inside the adjacent
    # pair that brackets the target, is what converges. Bisection was ideal for
    # the step and bad for the slope; this is adequate for both, at ~10
    # evaluations against bisection's ~11.
    def _sample(a, b):
        pts = []
        for x in np.linspace(a, b, n_points):
            pts.append((float(x), evaluate(plan, config, x, search_sims, floor,
                                           parallel, adequacy=adequacy)['success_rate']))
        return pts

    # PAD THE LOW SIDE BEFORE SAMPLING. The search curve sits ~2pp BELOW the
    # verification curve (documented above: 89.2% on 400 paths against 91.2% on
    # 1500 at the same spend), so the crossing FITTED on search points lands
    # below the one the straddle was built around — and a straddle built on
    # search reads therefore tends to clip at its lower edge. Measured after the
    # informed bracket landed: 4 of 8 strategies still reported a clip, every
    # one of them at the low end. Widening only downward costs one sample point
    # of resolution and removes the asymmetry that caused it.
    lo = max(lo * 0.85, 0.0)
    with instrument.timed('sampling'):
        pts = _sample(lo, hi)
    pts += [(x, y) for x, y in probes if lo <= x <= hi]

    def _fit(points):
        # One point per spend: a probe and a sample at the same spend run the
        # same seeds, so they are one read and must carry one read's weight.
        pts_sorted = sorted(dict(points).items())
        xs = np.array([q[0] for q in pts_sorted])
        ys = _isotonic_decreasing([q[1] for q in pts_sorted])
        x0, sl, clip = _crossing(xs, ys, confidence)
        return xs, ys, x0, sl, clip

    xs, ys, spend, slope, clipped = _fit(pts)

    # THE BRACKET'S LOWER EDGE IS NOT A FLOOR ON THE ANSWER, and it used to act
    # like one. `lo` is the last probe scoring at least `bracket_floor`, which is
    # two standard errors BELOW the target — so a probe reading 88% when the
    # target is 90% still becomes `lo`, the sampled window starts above the true
    # crossing, and `_crossing` clips to `xs[0]` and reports it.
    #
    # Measured on portfolio 141: the bracket came back [200k .. 400k] with 200k
    # scoring exactly 0.900, every sample at or below target, and the crossing
    # clipped to 200,000. There it was harmless because the edge WAS the answer.
    # It is not harmless in general, and it is silent when it is not.
    #
    # So when the fit lands on the lower edge AND that edge is clearly below the
    # target, open the bracket downward and refit.
    #
    # Both conditions, not just the clip. A clip whose edge reads AT the target
    # means the edge is the crossing — which is the 141 case above — and
    # resampling a wider window there would only add curvature and noise to a
    # fit that was already right. Only an edge sitting a standard error BELOW
    # the target says the crossing is genuinely out of range.
    #
    # Bounded, because a policy that never reaches the target at any spend (a
    # depleting universe) would otherwise halve forever.
    for _ in range(4):
        clipped = spend <= xs[0] * (1.0 + 1e-6)
        if not clipped or ys[0] >= confidence - se or lo <= 0.0:
            break
        lo = max(lo * 0.5, 0.0)
        with instrument.timed('sampling'):
            pts += _sample(lo, xs[0])
        xs, ys, spend, slope, clipped = _fit(pts)

    # Two different problems, so two tools, chosen by what the samples show.
    #
    # If every observed success rate is exactly 0 or 1, the curve is a STEP: the
    # universe is deterministic (the zero-volatility anchor) or the sample is too
    # small to resolve anything else. There is nothing to average and a fitted
    # line through a step is not merely imprecise but BIASED — it read $103k
    # against a $122k annuity. Root-finding is the right tool, so bisect the
    # straddling pair.
    #
    # Otherwise the rates are genuine proportions and the crossing is a threshold
    # to be ESTIMATED, where averaging is everything: one refinement pass inside
    # the straddling pair, where a line is a better approximation.
    def _straddle(xs_, ys_):
        above = np.where(ys_ >= confidence)[0]
        below = np.where(ys_ < confidence)[0]
        if not len(above) or not len(below):
            return None
        a, b = float(xs_[above[-1]]), float(xs_[below[0]])
        return (a, b) if b > a else None

    stepwise = bool(np.all((ys < 1e-9) | (ys > 1 - 1e-9)))
    span = _straddle(xs, ys)
    if stepwise:
        if span is not None:
            a, b = span
            with instrument.timed('bisecting step'):
                for _ in range(12):
                    if b - a <= 0.01 * max(abs(b), 1.0):
                        break
                    mid = 0.5 * (a + b)
                    ok = evaluate(plan, config, mid, search_sims, floor, parallel,
                                  adequacy=adequacy)['success_rate']
                    a, b = (mid, b) if ok >= confidence else (a, mid)
            spend = a
    elif span is not None:
        a, b = span
        if (b - a) > 0.02 * max(abs(b), 1.0):
            with instrument.timed('refining'):
                _, _, spend, slope, clipped = _fit(_sample(a, b)
                                          + [(x, y) for x, y in pts
                                             if a <= x <= b])
    if verbose or VERBOSE:
        print(f"      crossing ${spend:,.0f} "
              f"({'bisected step' if stepwise else 'fitted'}, on the search sample)")
        # THE CURVE THE CROSSING WAS FITTED TO. Solver defects (a flat local fit
        # returning slope 0, a crossing clipped to a probe, a blind-seeded
        # bracket) were properties of these points and invisible without them.
        _print_curve(pts, confidence, search_sims)

    with instrument.timed('verification'):
        result = evaluate(plan, config, spend, verify_sims, floor, parallel,
                          full=True, adequacy=adequacy)

    # Set the LEVEL from the verification sample, not the search sample.
    #
    # The two disagree systematically, not randomly: the search paths are a
    # subset of the verification paths (both seed 42+sim_id), so whichever way
    # that subset leans, it leans the same way at every spend. The search-sample
    # curve sits ~2pp low (guyton_klinger at $100k read 89.2% on 400 paths and
    # 91.2% on 1500), and every method built on it lands low however carefully
    # it estimates.
    #
    # A Newton step on the fitted slope moves the answer onto the verification
    # curve, then a second verification confirms it; vpw's closed form takes its
    # quantile on the verification sample for the same reason. ITERATED, because
    # one step is not enough on a steep curve: the fitted slope can come from a
    # flatter neighbourhood than the one the answer sits in, so a single step
    # overshoots and stops short of the target.
    #
    # Each pass re-estimates the slope from the two verification points it now
    # has, which is a secant step rather than a fixed-slope one, and is kept only
    # if it lands closer. Bounded at MAX_NEWTON so a pathological curve cannot
    # spend the run.
    newton_steps = 0
    # A zero local slope (a flat or upward neighbourhood, which noise produces
    # near the crossing) must not skip the correction: it would leave the level
    # at the search-sample crossing and the bracket clip uncorrected, and the
    # clip is only harmless WHEN NEWTON RUNS. `usable_slope` falls back to the
    # secant across the whole search curve; the loop's own secant update then
    # replaces it with a verification-sample slope after the first step.
    if not stepwise:
        slope = usable_slope(slope, xs, ys)

    if not stepwise and slope < -1e-12:
        local_slope = slope
        # A CLIPPED CROSSING MUST TAKE AT LEAST ONE STEP, whatever its gap.
        # The 1pp tolerance means "close enough to the target", which is a
        # statement about a crossing that was actually ESTIMATED. An edge is not
        # an estimate — the window is saying the answer lies past it — so
        # accepting one because its gap happens to land inside tolerance stops
        # the search at the wrong number and reports it to three figures.
        forced = bool(clipped)
        for _ in range(MAX_NEWTON):
            gap = result['success_rate'] - confidence
            if abs(gap) <= 0.01 and not forced:
                break
            forced = False
            newton_steps += 1
            step = float(np.clip(gap / abs(local_slope),
                                 -0.40 * spend, 0.40 * spend))
            candidate = max(spend + step, 0.0)
            if abs(step) <= 0.01 * max(spend, 1.0) and not clipped:
                break
            with instrument.timed('verification'):
                confirmed = evaluate(plan, config, candidate, verify_sims, floor,
                                     parallel, full=True, adequacy=adequacy)
            # keep it only if it genuinely lands closer to the target
            if (abs(confirmed['success_rate'] - confidence)
                    >= abs(result['success_rate'] - confidence)):
                # SAY SO WHEN A CLIPPED LEVEL WAS CHECKED AND HELD: a held edge
                # must not print like an edge nobody looked at.
                if clipped and (verbose or VERBOSE):
                    print(f"      held      ${spend:,.0f} (bracket edge — the "
                          f"step to ${candidate:,.0f} scored "
                          f"{confirmed['success_rate']:.1%}, no closer)")
                break
            if verbose or VERBOSE:
                print(f"      verified  ${candidate:,.0f} "
                      f"({(candidate / spend - 1):+.1%} onto the verification "
                      f"curve, {confirmed['success_rate']:.1%})")
            # secant: the slope BETWEEN the two verified points beats the one
            # fitted on the search sample, which is what overshot in the first
            # place. Tested on the RATIO, not on d_rate: a step DOWN raises the
            # rate, and a `d_rate < 0` guard kept every downward correction on
            # the search slope.
            moved = candidate - spend
            d_rate = confirmed['success_rate'] - result['success_rate']
            if abs(moved) > 1e-9 and d_rate / moved < -1e-12:
                local_slope = d_rate / moved
            spend, result = candidate, confirmed

    result['confidence_target'] = confidence
    result['floor'] = floor
    result['solved_by'] = 'curve_fit'
    # WHETHER THE ANSWER ACTUALLY LANDED. The Newton loop stops on three
    # conditions and only one of them means success: inside 1pp of the target
    # (converged), a step too small to matter, a step that did not improve, or
    # MAX_NEWTON exhausted. The last three can leave a level sitting well off the
    # confidence the row is headed with, and nothing said so.
    #
    # Portfolio 102 is why this is recorded rather than assumed: its success
    # curve is steep enough that the solved level still moves with sample size
    # (30.5k / 28.4k / 26.9k / 17.0k at 400 / 800 / 1600 / 3200 verification
    # paths). A caller quoting that number should be able to see it is not
    # settled. `tools/solver_calibration.py --convergence` is the measurement.
    result['newton_steps'] = newton_steps
    result['clipped'] = int(clipped)
    result['gap'] = float(result['success_rate'] - confidence)
    result['converged'] = bool(abs(result['gap']) <= 0.01)
    # HOW PRECISE THE LEVEL IS, in dollars. The success rate is a binomial
    # proportion over `verify_sims` paths, so it carries SE sqrt(p(1-p)/n);
    # dividing by the local slope converts that into the dollar band the level
    # could have landed anywhere inside.
    #
    # This is not a footnote: vpw solves in closed form and carries no search
    # noise, while guyton_klinger's band (about +/-$7k at 1 SE on a $170k level)
    # is the entire scatter between portfolios on identical paths.
    sl = abs(local_slope) if (not stepwise and slope < -1e-12) else 0.0
    p_hat = float(result['success_rate'])
    se_rate = (p_hat * (1 - p_hat) / max(verify_sims, 1)) ** 0.5
    result['level_se'] = float(se_rate / sl) if sl > 1e-15 else float('nan')
    return result
