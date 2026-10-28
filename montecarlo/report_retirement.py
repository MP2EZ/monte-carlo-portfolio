"""
Decumulation reporting
======================

Runs the strategies a run spec resolves to and prints the comparison.

Two layouts, chosen by the shape of the question:

  ONE AXIS   N portfolios, or one portfolio x N bond tiers -> flat table
  TWO AXES   N portfolios x N bond tiers                   -> matrix

The matrix exists because a flat 12-row list destroys the structure being
compared. It also makes INTERACTION visible: if every row improves by the same
amount when derisking, the two decisions are independent and can be made
separately; if one improves while another worsens, they interact.

Every figure carries a confidence interval, and head-to-heads are judged on the
PAIRED interval. Strategies run under common random numbers, so path i faces the
same market in every one of them — which means overlapping marginal intervals do
NOT imply two strategies are indistinguishable.
"""

from datetime import datetime
from pathlib import Path

import numpy as np

from montecarlo import engine
from montecarlo import glide
from montecarlo import diagnostics
from montecarlo import instrument
from montecarlo import persistence
from montecarlo import policies
from montecarlo import solver

CONFIDENCE = 0.90

# One rule for every major break, so the eye can find where a phase ends.
_RULE = '  ' + '─' * 80

# Minimum Acceptable Return for Sortino, in REAL terms — 0% means "downside is
# losing purchasing power", the natural zero once returns are real.
#
# Reported WITH a confidence interval deliberately. Measured on this universe,
# Sortino does not separate these portfolios: all five span 0.903-0.935 and no
# pairwise difference resolves at 2,000 paths, because adding bonds lowers return
# and lowers downside deviation in near-equal measure. A bare number would imply
# a precision it does not have; the interval shows it is not discriminating.
MAR_ANNUAL = 0.00


def _straddles_zero(interval):
    lo, hi = interval
    return lo <= 0 <= hi


def _verdict(lo, hi):
    return 'BETTER' if lo > 0 else ('WORSE' if hi < 0 else 'not resolved')


def _spending_is_fixed(r, share=0.5):
    """True when most paths spend the SAME amount — fixed spending, where the
    median of realised spending is one number for every strategy and carries
    no information.

    Usually that amount is the plan level. Under constant_real below your
    Social Security benefit it is the benefit, once it starts: every path
    spends it and none spends exactly the plan, so a test against the level
    missed it and the return-model check scored a median that could not move.
    If more than half the paths share one amount, the median IS that amount.
    """
    per_path = r.get('per_path_mean')
    if per_path is None or len(per_path) == 0:
        return False
    per_path = np.asarray(per_path, float)
    return float(np.mean(np.isclose(per_path, np.median(per_path),
                                    rtol=1e-6))) > share


def _h2h_verdict(r, r_base, lo, hi, solving):
    """BETTER / WORSE / not resolved — or 'within solver noise'.

    Under fixed spending in solve mode every surviving path spends exactly its
    solved level, so the paired interval collapses to the gap between two
    levels and cannot see that each level is only placed to its level_se. A gap
    inside 2 x the combined standard error is not a verdict (2026-10-03: four
    sleeves solved to $226-227k printed BETTER / WORSE on +-0.3%).
    """
    verdict = _verdict(lo, hi)
    if not (solving and _spending_is_fixed(r) and _spending_is_fixed(r_base)):
        return verdict
    se_a, se_b = r.get('level_se'), r_base.get('level_se')
    if se_a is None or se_b is None or se_a != se_a or se_b != se_b:
        return verdict
    noise = 2.0 * (se_a ** 2 + se_b ** 2) ** 0.5
    return 'within solver noise' if abs(r['spend'] - r_base['spend']) <= noise else verdict


def _h2h_rows(specs, res, base, solving):
    """(strategy, lo, hi, verdict, unit) for every strategy but the base.

    With a common-plan run on every strategy (see evaluate_strategies) the
    comparison reads THAT: one plan, the same paths, so the gap is the
    portfolio's alone. At one FIXED plan every surviving path spends exactly
    the plan and the median says nothing, so the comparison becomes the paired
    difference in P(run out) (unit 'ruin', fewer is BETTER). Without common
    runs — one strategy solved, or a fixed-target run — it falls back to the
    own-level paths, guarded by _h2h_verdict.
    """
    common = all('common' in res[g] for g in specs)
    fixed = common and all(
        _spending_is_fixed(res[g]['common']) and res[g]['common'].get('depleted') is not None
        for g in specs)
    rows = []
    for g in specs:
        if g == base:
            continue
        if fixed:
            lo, hi = engine.paired_bootstrap_diff(
                res[g]['common']['depleted'].astype(float),
                res[base]['common']['depleted'].astype(float), np.mean)
            verdict = 'BETTER' if hi < 0 else ('WORSE' if lo > 0 else 'not resolved')
            rows.append((g, lo, hi, verdict, 'ruin'))
        elif common:
            lo, hi = engine.paired_bootstrap_diff(
                res[g]['common']['per_path_mean'], res[base]['common']['per_path_mean'],
                np.median)
            rows.append((g, lo, hi, _verdict(lo, hi), 'spend'))
        else:
            lo, hi = engine.paired_bootstrap_diff(
                res[g]['per_path_mean'], res[base]['per_path_mean'], np.median)
            rows.append((g, lo, hi, _h2h_verdict(res[g], res[base], lo, hi, solving), 'spend'))
    return rows


def _leaders(specs, res, base, solving):
    """Who spends most and who has the best low case — on the one-plan runs
    when every strategy has them (see evaluate_strategies), on each strategy's
    own level otherwise. `unresolved` flags a leader that is not the base and
    does not resolve BETTER against it in the head-to-head."""
    at_one = all('common' in res[g] for g in specs) and not (
        all(_spending_is_fixed(res[g]['common']) for g in specs))
    if at_one:
        med = {g: float(np.median(res[g]['common']['per_path_mean'])) for g in specs}
        p10 = {g: float(np.percentile(res[g]['common']['per_path_mean'], 10)) for g in specs}
    else:
        spendable = all(res[g].get('median_realised') is not None for g in specs)
        k = 'median_realised' if spendable else 'spend'
        med = {g: res[g][k] for g in specs}
        p10 = {g: res[g]['p10_realised'] for g in specs}
    best = max(specs, key=med.get)
    safest = max(specs, key=p10.get)
    unresolved = safest_unresolved = False
    if at_one and best != base:
        row = next(r for r in _h2h_rows(specs, res, base, solving) if r[0] == best)
        unresolved = row[3] != 'BETTER'
    # The low case gets the same test, on its own statistic. It was printed with
    # none, so a p10 a few hundred dollars ahead by noise read as a finding.
    if at_one and safest != base:
        lo, _ = engine.paired_bootstrap_diff(
            res[safest]['common']['per_path_mean'], res[base]['common']['per_path_mean'],
            lambda v: np.percentile(v, 10))
        safest_unresolved = lo <= 0
    return {'best': best, 'best_value': med[best], 'safest': safest,
            'safest_value': p10[safest], 'at_one_plan': at_one,
            'plan': res[base]['common']['spend'] if at_one else None,
            'unresolved': unresolved, 'safest_unresolved': safest_unresolved,
            'level_key': at_one or all(res[g].get('median_realised') is not None
                                       for g in specs)}


def worst_year_pct(worst_cut):
    """The worst year's spending AS A PERCENT OF PLAN.

    `solver.score` reports `worst_cut` = 1 - min(spending)/plan, which is a
    sensible number for a flat policy and a nonsense one for a rising policy:
    vpw spends far above its certified floor, so the "cut" goes NEGATIVE and a
    real run printed "worst cut -155.2%". That is not a cut, and -155% is not a
    percentage of anything a reader can place.

    Expressed against plan it is always meaningful and always positive:

        100%   never spent below plan
         85%   worst year ran 15% under plan
        212%   worst year ran at twice plan

    One definition, so the column header and the arithmetic cannot drift.
    """
    return (1.0 - float(worst_cut)) * 100.0


def _max_drawdowns(wealth):
    """Worst peak-to-trough drop on each path, in percent (negative)."""
    running_max = np.maximum.accumulate(wealth, axis=1)
    return ((wealth - running_max) / running_max * 100.0).min(axis=1)


def _rebalancing(runs, config):
    """Aggregate the per-path rebalancing counters, or None if not rebalancing."""
    if config.get('rebalancing_strategy', 'none') == 'none':
        return None
    try:
        return engine.aggregate_rebalancing_stats(
            [r[3] for r in runs], config.get('time_horizon_years', 25))
    except Exception:          # diagnostics must never take a run down
        return None


def evaluate_strategies(specs, portfolios, config, spec, labels=None):
    """Simulate every strategy and collect what the tables need."""
    start_age = config['age']
    solving = spec.get('spend') == 'solve'
    labels = labels or {}
    out = {}
    print(f"\n  {'SOLVING' if solving else 'RUNNING'}   "
          f"{'finding each strategy\'s maximum' if solving else 'simulating each strategy'}"
          f" — this is the slow part")

    for i, gspec in enumerate(specs, 1):
        plan = glide.build_plan(glide.parse_schedule(gspec, start_age), portfolios)

        if spec['mode'] == 'accumulation':
            n = spec.get('sims', solver.FIXED_SIMS)
            # light=False here: accumulation reports Sortino, which needs the
            # monthly return series the light worker drops. Costs one run's worth
            # of marshalling, not the ~14 a solve would.
            with instrument.timed(labels.get(gspec, gspec)):
                runs = solver._paths(plan, config, n, parallel=True, light=False)
            finals = np.array([r[0] for r in runs]) / 1e6
            returns = np.array([r[2] for r in runs])
            wealth = np.array([r[1] for r in runs], dtype=float)
            sortino = engine.aggregate_sortino_stats(returns, mar_annual=MAR_ANNUAL)
            out[gspec] = {
                'finals': finals,
                'median': float(np.median(finals)),
                'sortino': sortino['all_sortino_ratios'],
                'sortino_median': float(sortino['sortino_median']),
                'sortino_stats': sortino,
                'twr_cagr': engine.time_weighted_cagr(
                    returns, config['time_horizon_years']),
                # Reduced HERE, and the two big matrices dropped with `runs`.
                # `returns` and the wealth paths are n_sims x ~700 floats each —
                # 56MB per strategy at 5,000 paths — and the only things anything
                # downstream wants from them are a volatility, a median path and
                # a drawdown per path. Retaining them for 12 strategies so that
                # figures could be drawn at the end cost ~700MB for three arrays.
                'annual_vol': float(np.std(returns) * np.sqrt(12) * 100),
                'median_path': np.median(wealth, axis=0),
                'max_drawdown': _max_drawdowns(wealth),
                'rebalancing': _rebalancing(runs, config),
            }
            print(f"    {labels.get(gspec, gspec):<{6 if labels else 26}} "
                  f"median ${np.median(finals):.2f}M   ({i}/{len(specs)})")
            continue

        adequacy = config.get('adequacy', solver.DEFAULT_ADEQUACY)
        if solving:
          with instrument.timed(f'solve {labels.get(gspec, gspec)}'):
            res = solver.solve_sustainable_spend(
                plan, config, confidence=CONFIDENCE,
                floor=config['withdrawal'].get('floor', 0.85),
                search_sims=spec.get('search_sims', solver.SEARCH_SIMS),
                verify_sims=spec.get('verify_sims', solver.VERIFY_SIMS),
                adequacy=adequacy)
        else:
          with instrument.timed(f'evaluate {labels.get(gspec, gspec)}'):
            res = solver.evaluate(plan, config, config['withdrawal']['target_real_annual'],
                                  n_sims=spec.get('sims', solver.FIXED_SIMS),
                                  floor=config['withdrawal'].get('floor', 0.85),
                                  full=True, adequacy=adequacy)
        out[gspec] = res
        tag = labels.get(gspec, gspec)
        print(f"    {tag:<{6 if labels else 26}} ${res['spend'] / 1e3:.0f}k/yr plan level"
              f"   ({i}/{len(specs)})")

    # ONE PLAN FOR THE COMPARISON. Each strategy was solved at its OWN level,
    # and a level is only placed to its level_se, so realised spending compared
    # across levels mixes the portfolio's effect with the solver's aim: on
    # 2026-10-03 that made 166 look ~1.5% ahead of 165, and at one plan the gap
    # was ~0.4% and did not resolve. So every strategy runs again at the FIRST
    # strategy's level, on the same verification paths, and the head-to-head
    # reads these runs. The base re-runs too, so all of them come from one
    # identical evaluation.
    if solving and spec['mode'] == 'retirement' and len(specs) > 1:
        level = out[specs[0]]['spend']
        n = spec.get('verify_sims', solver.VERIFY_SIMS)
        adequacy = config.get('adequacy', solver.DEFAULT_ADEQUACY)
        name0 = labels.get(specs[0], specs[0])
        print(f"\n  COMPARING  every strategy at {name0}'s ${level / 1e3:.0f}k/yr, "
              f"on the same {n:,} paths")
        for gspec in specs:
            plan = glide.build_plan(glide.parse_schedule(gspec, start_age), portfolios)
            with instrument.timed(f'compare {labels.get(gspec, gspec)}'):
                r = solver.evaluate(plan, config, level, n_sims=n,
                                    floor=config['withdrawal'].get('floor', 0.85),
                                    full=True, adequacy=adequacy)
            # The whole score, not four fields of it: RESULTS and the
            # life-stage table read these runs too (see _at_one_plan).
            out[gspec]['common'] = {**{k: v for k, v in r.items() if k != 'raw'},
                                    'spend': float(level),
                                    'depleted': np.asarray(r['raw']['depleted'], bool)}

    return out


def _at_one_plan(specs, res):
    """Each strategy as its one-plan run saw it, or `res` unchanged.

    The banner says every strategy is compared at A's level, and the head-to-
    head and the leader lines read those runs — but RESULTS, the failing-paths
    table and the life-stage table read each strategy's OWN level, so one
    report described two different sets of runs under one heading. Laying the
    one-plan statistics over each record keeps the solve's own facts (`spend`,
    `converged`, `gap`, `level_se`) and replaces everything the paths produced.
    """
    if len(specs) < 2 or not all('common' in res.get(g, {}) for g in specs):
        return res
    return {g: {**res[g], **{k: v for k, v in res[g]['common'].items() if k != 'spend'}}
            for g in specs}


def _one_plan_note(specs, res, name):
    """'every strategy at A's $160k/yr plan', or '' without one-plan runs."""
    if len(specs) < 2 or not all('common' in res.get(g, {}) for g in specs):
        return ''
    return (f"every strategy at {name(specs[0])}'s "
            f"${res[specs[0]]['common']['spend'] / 1e3:,.0f}k/yr plan")


def _reads_one_plan(runs, spec, is_grid):
    """True when the report compares every strategy at one plan: one policy, no
    plan choice, no grid. The life-stage table and the figure both ask."""
    return len(runs) == 1 and not spec.get('compare_choice') and not is_grid


def _shape_rows(runs, specs, by_policy, spec, labels=None, is_grid=False):
    """The life-stage table's rows, and the plan they were read at.

    ONE POLICY: the same one-plan runs RESULTS and the head-to-head read.
    Across policies or plan choices each row keeps its own level — those
    comparisons are ABOUT the level, and no single plan spans them. So does a
    grid: its IN DETAIL table prints each strategy's own-level run, and the two
    must not give one strategy two lifetime averages.
    """
    one = _reads_one_plan(runs, spec, is_grid)
    rows = []
    for tag, _, _ in runs:
        res = _at_one_plan(specs, by_policy[tag]) if one else by_policy[tag]
        rows += [(tag if len(runs) > 1 else '', g, res[g]) for g in specs]
    note = (_one_plan_note(specs, by_policy[runs[0][0]],
                           lambda g: (labels or {}).get(g, g)) if one else '')
    return rows, note


def _legend(specs, labels, portfolios, config):
    """Print the label key once, so tables can stay narrow."""
    if not labels:
        return
    print("\n  STRATEGIES")
    for g in specs:
        sch = glide.parse_schedule(g, config['age'])
        import run as _run
        chain = ' → '.join(_run.ref_split(w.portfolio_id, portfolios) for w in sch)
        note = 'no derisking — the control' if len(sch) == 1 else ''
        print(f"    {labels[g]}   {g:<34} {chain:<28} {note}")


MAX_DETAIL_ROWS = 16
# Naming every unresolved row is useful at 3 strategies and unreadable at 12.
MAX_SORTINO_NAMED = 4


def _sortino_note(specs, res, base, name=None):
    """State plainly when Sortino cannot tell these rows apart.

    Measured with paired bootstrap CIs at 2,000 paths, the current portfolios
    all span 0.903-0.935 and no pairwise difference resolves: adding bonds
    lowers return and lowers downside deviation, and being risk-ADJUSTED the
    ratio barely moves. Printing the column without this invites ranking on
    three decimal places of noise.
    """
    name = name or (lambda g: g)
    if len(specs) < 2 or not all('sortino' in res[g] for g in specs):
        return
    unresolved = [g for g in specs if g != base and
                  _straddles_zero(engine.paired_bootstrap_diff(
                      res[g]['sortino'], res[base]['sortino'], np.median))]
    if not unresolved:
        return
    # "8 of the 8 other rows" is how this read on a 9-strategy run — a
    # fraction whose numerator equals its denominator, which is the one case
    # where a count is the wrong phrasing entirely.
    others = len(specs) - 1
    if len(unresolved) <= MAX_SORTINO_NAMED:
        who = ', '.join(name(g) for g in unresolved)
    elif len(unresolved) == others:
        who = f"ANY of the {others} other rows"
    else:
        who = f"{len(unresolved)} of the {others} other rows" 
    print(f"\n    Sortino (MAR {MAR_ANNUAL:.0%} real) does not separate "
          f"{who} from {name(base)}.")
    print( "    It is risk-ADJUSTED, so a change that lowers return and "
           "downside deviation\n    together — adding bonds is the clear "
           "case — leaves it near-unchanged.\n    Rank those on wealth or "
           "sustainable spending instead.")


def _plan_in_context(specs, res, config, indent='  ', name=None,
                     spec=None, by_policy=None):
    """The plan level as a WITHDRAWAL RATE, and the capital behind it.

    The report printed plan levels, realised spending, estates, CAGR and
    Sortino, and never the one ratio that says whether any of it is aggressive.
    A reader asked why 100% equity wins when every textbook says 60/40, and the
    answer — a 2.6% withdrawal rate against a $71k real annuity — could not be
    got from the output; it took a separate simulation.

    Guaranteed income is shown as CAPITAL because that is what it is. Social
    Security at $71k/yr from 70 capitalises to ~$1.1M of inflation-linked
    bond at 65 — a sizeable share of total retirement capital, more at p10. A
    derisking question asked about the portfolio alone is asked about the wrong
    denominator.
    """
    name = name or (lambda g: str(g))
    bals = [res[g]['balance_at_retirement'] for g in specs
            if res.get(g, {}).get('balance_at_retirement')]
    if not bals:
        return
    med = float(np.median([b[50] for b in bals]))
    lo = float(np.median([b[10] for b in bals]))
    hi = float(np.median([b[90] for b in bals]))
    if med <= 0:
        return
    ret_age = (config.get('withdrawal') or {}).get('start_age',
                                                   config.get('retirement_age'))
    print(f"\n{indent}YOUR PLAN IN CONTEXT   at {int(ret_age)}, before the first "
          f"withdrawal")
    print(f"{indent}  portfolio    p10 ${lo / 1e6:.1f}M    median "
          f"${med / 1e6:.1f}M    p90 ${hi / 1e6:.1f}M")
    # THE CERTIFIED BAR LIVES HERE, NOT IN THE COMPARISON TABLES. It is what
    # you withdraw in year one — an instruction for the strategy you pick, not a
    # metric for choosing between them. In a comparison column it was the
    # noisiest thing on the page (+/-$9k on $200k) and the report told the
    # reader four times not to rank on it; here it is the answer to "find the
    # maximum I can sustain", stated once, with the precision attached.
    lv = [res[g]['spend'] for g in specs if res.get(g, {}).get('spend')]
    ses = [res[g]['level_se'] for g in specs
           if res.get(g, {}).get('level_se')
           and res[g]['level_se'] == res[g]['level_se']]
    if lv:
        a, b = min(lv), max(lv)
        rng = (f"${a / 1e3:,.0f}k" if a == b
               else f"${a / 1e3:,.0f}k-${b / 1e3:,.0f}k")
        band = f"  (±${max(ses) / 1e3:,.0f}k)" if ses else ''
        edge = [name(g) for g in specs if res.get(g, {}).get('clipped')]
        word = 'year 1' if _level_key(spec or {}, by_policy) == 'year1' \
            else 'plan level'
        print(f"{indent}  {word:<12} {rng}{band}  =  {_span(a / med, b / med)} "
              f"of the median, {_span(a / lo, b / lo)} at p10")
        print(f"{indent}  the 4% rule  would support ${0.04 * med / 1e3:,.0f}k "
              f"at the median — the bar most retirement advice is set against")
        # The clip warning follows the bar. It used to mark the level COLUMN;
        # with that column gone it would otherwise have been orphaned — a
        # marker with nothing to mark.
        if edge:
            print(f"{indent}  ~ {', '.join(edge[:6])}"
                  f"{' and others' if len(edge) > 6 else ''} solved to a "
                  f"BRACKET EDGE: the search")
            print(f"{indent}    window ended there, so read those as 'at least "
                  f"this', not as an estimate.")
    ss = config.get('social_security')
    if ss and ss.get('annual_at_fra'):
        claim = float(ss.get('claim_age', 67))
        annual = policies.ss_income(claim + 0.5, ss)
        end = config['age'] + config['time_horizon_years']
        # VALUED AT RETIREMENT, beside the balance it is compared with. It was
        # discounted only to the claim age, so a benefit claimed at 70 against
        # a balance at 65 skipped five years of discounting (~12% too high).
        # One payment a year from the claim age, none before retirement.
        r = 0.03
        ages = range(max(int(round(claim)), int(np.ceil(ret_age))), int(round(end)))
        yrs = max(len(ages), 1)
        pv = sum(annual / (1 + r) ** (a - ret_age) for a in ages)
        print(f"{indent}  guaranteed   ${annual / 1e3:,.0f}k/yr from "
              f"{int(claim)} is ${pv / 1e6:.2f}M at {int(ret_age)} of "
              f"inflation-linked BOND")
        print(f"{indent}               at {r:.0%} real over {yrs} years — "
              f"{pv / (pv + med):.0%} of your total capital, "
              f"{pv / (pv + lo):.0%} at p10.")
        print(f"{indent}               A 100/0 portfolio with that behind it is "
              f"nearer {1 - pv / (pv + med):.0%}/{pv / (pv + med):.0%} overall.")


# ONE DEFINITION OF EVERY LEVEL COLUMN — name, sub-head, width and format.
#
# There were THREE tables printing the same quantities: RESULTS (_flat_table),
# IN DETAIL (_matrix) and POLICY COMPARISON (_policy_table). They had drifted
# into three different column sets with three different headers — `spend` in
# one and `plan level` in another for the identical number, `worst yr` here and
# `worst cut` there, the level's standard error a COLUMN in one table and a
# footnote in the next. A reader moving between them was reading the same thing
# under different names, which is how "why is spend different in the two
# tables?" became a real question about a report that was answering it twice.
#
# Each table now picks a SUBSET BY NAME. Adding a column, or renaming one,
# happens once.
_LEVEL_COLUMNS = {
    # THE MOST ACTIONABLE NUMBER IN THE REPORT, and it was labelled as an
    # abstraction. `engine._resolve_initial_withdrawal` is documented as
    # "First-year withdrawal in real dollars", and the engine seeds
    # `w_prior_year=w_initial` from it — so for guyton_klinger, constant_real
    # and fixed_pct the certified bar is LITERALLY what you withdraw in your
    # first year of retirement. Everything else in the table is an average over
    # decades; this is the instruction.
    #
    # NOT for a self-directed policy. vpw computes its own opening draw from the
    # balance and the target is only a bar it is judged against — see
    # policies.SELF_DIRECTED and the engine's `if wd_self_directed` branch — so
    # the label has to say `certified bar` whenever one is in the run.
    'year1':    ('year 1', 'you withdraw', 12,
                 lambda r: f"{r['spend'] / 1e3:,.0f}k"
                           + ('~' if r.get('clipped') else '')),
    'level':    ('certified', 'bar', 12,
                 lambda r: f"{r['spend'] / 1e3:,.0f}k"
                           + ('~' if r.get('clipped') else '')),
    'se':       ('1 SE', '', 7,
                 lambda r: (f"±{r['level_se'] / 1e3:,.0f}k"
                            if r.get('level_se') and r['level_se'] == r['level_se']
                            else '-')),
    # SPENDING BY LIFE STAGE, both halves. `median_realised` was computed by
    # the solver from the beginning and displayed NOWHERE — the table showed the
    # bad case and not the ordinary one, so a reader could see how bad it gets
    # and not what it usually is.
    'median':   ('you spend', 'lifetime avg', 13,
                 lambda r: (f"{r['median_realised'] / 1e3:,.0f}k"
                            if r.get('median_realised') is not None else '-')),
    'p10':      ('low case', '(p10)', 11,
                 lambda r: f"{r['p10_realised'] / 1e3:,.0f}k"),
    'vol':      ('vol', 'annual', 8,
                 lambda r: (f"{r['annual_vol']:.1f}%"
                            if r.get('annual_vol') is not None else '-')),
    'ruin':     ('P(run', 'out)', 8,
                 lambda r: f"{r['depletion_rate']:.1%}"),
    'bequest':  ('bequest', '', 11,
                 lambda r: f"{r['median_bequest'] / 1e6:,.1f}M"),
    'worst':    ('worst yr', '(p10)', 10,
                 lambda r: (f"{worst_year_pct(r['worst_cut_bad']):.0f}%"
                            if r.get('worst_cut_bad') is not None
                            else f"{worst_year_pct(r['worst_cut']):.0f}%")),
    'cagr':     ('CAGR', '', 8,
                 lambda r: (f"{np.median(r['twr_cagr']) * 100:.2f}%"
                            if 'twr_cagr' in r else '-')),
    'sortino':  ('Sortino', '', 9,
                 lambda r: f"{r.get('sortino_median', float('nan')):.3f}"),
}


def _level_key(spec, by_policy=None):
    """'year1' when the bar IS the first year's withdrawal, else 'level'.

    True for every policy that takes the target as a control. A self-directed
    one (vpw) sets its own opening draw, so in a run containing one the column
    cannot be called `year 1` for any row without being wrong for that one.
    """
    tags = list(by_policy or ())
    if not tags:
        tags = list(spec.get('compare_policies')
                    or [spec.get('policy', 'guyton_klinger')])
    # MATCH ON THE BASE POLICY. A row is tagged `vpw reserve 5M`, not `vpw` —
    # keying on the full tag silently suppressed the policy-tradeoff block once
    # the reserve frontier started fanning rows out, and it is the same trap
    # here: the column would have been headed `year 1` in a run containing vpw,
    # for which it is not.
    return ('level'
            if any(t == sd or t.startswith(sd + ' ')
                   for t in tags for sd in policies.SELF_DIRECTED)
            else 'year1')


def _level_header(cols, lead, lead_w):
    """Two header lines for a chosen set of level columns."""
    top = f"{lead:<{lead_w}}" + ''.join(
        f"{_LEVEL_COLUMNS[c][0]:>{_LEVEL_COLUMNS[c][2]}}" for c in cols)
    sub = f"{'':<{lead_w}}" + ''.join(
        f"{_LEVEL_COLUMNS[c][1]:>{_LEVEL_COLUMNS[c][2]}}" for c in cols)
    return top.rstrip(), sub.rstrip(), len(top)


def _level_row(r, cols, lead, lead_w):
    """ONE width per column, governing the header AND the row.

    The first version let the formatter choose its own padding while the header
    used the declared width, so the two could disagree — and immediately did:
    the plan level ran into the standard error as `200k~±   9k`. The formatters
    now return bare content and this pads it, so a column cannot be wide in one
    line and narrow in the next.
    """
    return f"{lead:<{lead_w}}" + ''.join(
        f"{_LEVEL_COLUMNS[c][3](r):>{_LEVEL_COLUMNS[c][2]}}" for c in cols)


def _flat_table(specs, res, spec, base, labels=None):
    labels = labels or {}
    def name(g):
        return labels.get(g, g)
    width = 6 if labels else 26

    if spec['mode'] == 'accumulation':
        # TWO tables rather than one 96-column row. The old single line put
        # 'median wealth' over a number four columns to its right and ran past
        # any sane terminal; nothing here is worth a horizontal scroll.
        #
        # p10 and the worst drawdown are NEW, and they are the point: the model
        # gained a shared cash-rate band and a shared inflation band on
        # 2026-09-11, both of which exist to widen the distribution. Reporting
        # only the median understates what the model now knows about its own
        # uncertainty. The drawdown was already being computed and thrown away.
        print(f"\n{_RULE}\n  WHERE YOU END UP   real 2026 dollars\n{_RULE}")
        print(f"  {'':<{width}}{'p10':>10}{'median':>10}{'p90':>10}"
              f"{'worst':>10}{'CAGR':>9}")
        print(f"  {'':<{width}}{'bad case':>10}{'':>10}{'good case':>10}"
              f"{'drawdown':>10}{'':>9}")
        print("  " + "-" * (width + 49))
        for g in specs:
            r = res[g]
            f_ = r['finals']
            dd = (np.median(r['max_drawdown']) if 'max_drawdown' in r
                  else float('nan'))
            print(f"  {name(g):<{width}}{np.percentile(f_, 10):>9.1f}M"
                  f"{r['median']:>9.1f}M{np.percentile(f_, 90):>9.1f}M"
                  # _max_drawdowns already returns PERCENT, so :.0% would
                  # multiply by 100 twice — it printed -2361% before this.
                  f"{dd:>9.0f}%{np.median(r['twr_cagr']) * 100:>8.2f}%")

        print(f"\n  CONFIDENCE IN THAT MEDIAN   95% CI, and risk-adjusted return")
        print(f"  {'':<{width}}{'median wealth 95% CI':>26}{'Sortino':>10}"
              f"{'95% CI':>18}")
        print("  " + "-" * (width + 54))
        for g in specs:
            r = res[g]
            w_lo, w_hi = engine.bootstrap_ci(r['finals'], np.median)
            s_lo, s_hi = engine.bootstrap_ci(r['sortino'], np.median)
            print(f"  {name(g):<{width}}[{w_lo:>10.1f}M, {w_hi:>9.1f}M]"
                  f"{r['sortino_median']:>10.3f}"
                  f"   [{s_lo:>6.3f}, {s_hi:>6.3f}]")
        _contribution_split(specs, res, spec, name, width)

        print(f"\n{_RULE}\n  HEAD-TO-HEAD   is the difference real, or "
              f"sampling noise?\n{_RULE}")
        print(f"    Same random markets for every strategy, so the DIFFERENCE is "
              f"pinned far")
        print(f"    better than either level — the intervals ABOVE overlap and "
              f"that means")
        print(f"    nothing. Excluding zero is real; straddling zero is not "
              f"resolved.")
        print(f"\n    how much MORE than {name(base)}")
        print(f"    {'':<{width}} {'final wealth ($M)':>24}   {'Sortino':>24}")
        for g in specs:
            if g == base:
                continue
            w = engine.paired_bootstrap_diff(res[g]['finals'], res[base]['finals'],
                                             np.median)
            v = engine.paired_bootstrap_diff(res[g]['sortino'], res[base]['sortino'],
                                             np.median)
            print(f"    {name(g):<{width}} [{w[0]:+7.2f}, {w[1]:+7.2f}] "
                  f"{_verdict(*w):<13} [{v[0]:+7.3f}, {v[1]:+7.3f}] {_verdict(*v)}")
        _sortino_note(specs, res, base, name)
        return

    # The solve's own facts (converged, gap) still come from `res`; every
    # column the paths produced comes from the one-plan runs when there are any.
    plan_note = _one_plan_note(specs, res, name)
    paths = _at_one_plan(specs, res)
    show_cut = any(paths[g]['worst_cut'] > 0.005 for g in specs)
    print(f"\n{_RULE}\n  RESULTS   per year, real 2026 dollars"
          f"{'  ·  ' + plan_note if plan_note else ''}\n{_RULE}")
    # NOT "if it goes badly": this is the 10th percentile of each path's AVERAGE
    # spending, and a ratcheting policy like guyton_klinger pushes even that above
    # the starting target — the column read $126k against a $120k "sustainable"
    # figure, which looks like nonsense until you know what it measures.
    cols = ['median', 'p10', 'vol', 'ruin', 'bequest', 'cagr', 'sortino']
    top, sub, wide = _level_header(cols, '', width + 1)
    print('  ' + top)
    print('  ' + sub)
    print("  " + "-" * wide)
    for g in specs:
        print('  ' + _level_row(paths[g], cols, name(g), width + 1))
    if show_cut:
        print(f"\n  median worst single-year cut: "
              + ', '.join(f"{name(g)} {paths[g]['worst_cut']:.0%}" for g in specs))

    # WHY THE FAILING PATHS FAILED. 'P(run out)' is already in the table, but
    # alone it is misleading: success is (never depleted) AND (average spending
    # cleared the floor), so a strategy can miss the confidence target without
    # ever running out of money. Measured on VT, every failure is inadequacy and
    # NONE is ruin — which changes what the number means and what you would do
    # about it. Both terms are already in `score`; nothing contrasted them.
    fails = [(g, 1.0 - paths[g]['success_rate'], paths[g]['depletion_rate'])
             for g in specs]
    if any(f > 0.001 for _, f, _ in fails):
        print(f"\n  WHY THE FAILING PATHS FAILED")
        print(f"  {'':<{width}}{'failed':>9}{'ran out':>10}{'underspent':>12}")
        print("  " + "-" * (width + 31))
        for g, fail, ruin in fails:
            # success = (not depleted) AND adequate, so the residual is exactly
            # 'stayed solvent but could not sustain the floor'.
            print(f"  {name(g):<{width}}{fail:>9.1%}{ruin:>10.1%}"
                  f"{max(fail - ruin, 0.0):>12.1%}")
        worst = max(fails, key=lambda t: t[1] - t[2])
        if worst[1] - worst[2] > 0.7 * max(worst[1], 1e-9):
            print(f"\n  Mostly UNDERSPENDING, not ruin: the money lasts, the "
                  f"lifestyle does not.")
            print(f"  A lower plan level fixes this; a safer portfolio may not.")

    # DID THE SOLVER ACTUALLY LAND? Three of the four ways its Newton pass exits
    # are not success, and a level that stopped short is reported in exactly the
    # same typeface as one that converged. Portfolio 102's plan level still
    # moves with sample size, which is precisely the case this flags.
    off = [(g, res[g].get('gap', 0.0)) for g in specs
           if res[g].get('converged') is False]
    if off:
        # The SIGN decides whether this is a risk or a rounding note, and they
        # deserve different words. Landing BELOW the target means the level is
        # too high — the row is headed 90% and does not deliver it. Landing
        # above means the search stopped early and left spending on the table,
        # which is a smaller problem and the opposite kind.
        high = [(g, gap) for g, gap in off if gap < 0]
        low = [(g, gap) for g, gap in off if gap > 0]
        if high:
            print(f"\n  ⚠ LEVEL IS OPTIMISTIC — did not reach "
                  f"{CONFIDENCE:.0%} confidence:")
            for g, gap in high:
                print(f"      {name(g):<{width}} delivered {CONFIDENCE + gap:.1%}, "
                      f"{gap:+.1%} short. The figure above is too high.")
        if low:
            print(f"\n  · conservative — the search stopped above "
                  f"{CONFIDENCE:.0%}, so these could likely support more:")
            for g, gap in low:
                print(f"      {name(g):<{width}} delivered {CONFIDENCE + gap:.1%} "
                      f"({gap:+.1%})")
        print(f"    tools/solver_calibration.py --convergence measures whether a "
              f"level is settled.")

    # RANK AND QUOTE THE SPENDING, not the bar. This read `most spending 161
    # $201k/yr` directly under a table whose spending column said 525k — it was
    # keyed on 'spend', the solver's certified LEVEL. Three lines below, the
    # same block says "Every column above is something the paths PRODUCED. The
    # bar the solver certified is in YOUR PLAN IN CONTEXT", so the line
    # disclaimed the bar and then printed it. `safest` next to it was already
    # on realised spending, which is what made the mismatch invisible.
    #
    # The two can pick different strategies — that is the point, and the whole
    # reason the derisk baseline moved off the level in 2da04e0.
    lead = _leaders(specs, res, base, spec.get('spend') == 'solve')
    where = (f" at one plan (${lead['plan'] / 1e3:,.0f}k/yr)" if lead['at_one_plan'] else '')
    tail = ' lifetime average' if lead['level_key'] else '/yr plan level'
    print(f"\n  ✓ most spending   {name(lead['best'])}  "
          f"${lead['best_value'] / 1e3:,.0f}k/yr{tail}{where}")
    if lead['unresolved']:
        print(f"                    — not resolved against {name(base)}: see HEAD-TO-HEAD")
    if lead['safest'] != lead['best']:
        print(f"  ✓ best low case   {name(lead['safest'])}  "
              f"${lead['safest_value'] / 1e3:,.0f}k/yr lifetime average, "
              f"10th percentile{where}")
        if lead['safest_unresolved']:
            print(f"                    — not resolved against {name(base)}: the paired "
                  f"low cases overlap")
    # The old note here explained the `plan level` column. That column moved to
    # YOUR PLAN IN CONTEXT, and a footnote describing an absent column is worse
    # than none — it sends the reader looking for something that is not there.
    print(f"\n  Every column above is something the paths PRODUCED. The bar the")
    print(f"  solver certified is in YOUR PLAN IN CONTEXT, with its precision.")
    print(f"  CAGR and Sortino describe the portfolio, not the plan. Rank on the")
    print(f"  life-stage table below.")

    if len(specs) == 1:
        return
    print(f"\n{_RULE}\n  HEAD-TO-HEAD   is the difference real, or sampling "
          f"noise?\n{_RULE}")
    print(f"    Every strategy runs on the SAME random markets, so the "
          f"DIFFERENCE between")
    print(f"    two is pinned far better than either level. A range excluding "
          f"zero is a")
    print(f"    real difference; one straddling zero is not resolved — which is "
          f"not 'equal'.")
    # AS A PERCENTAGE TOO. Pairing under common random numbers resolves effects
    # far smaller than the level table can see, so "BETTER" here routinely means
    # ONE PERCENT — and read as a dollar range next to a six-figure base it
    # looks like a finding. Measured on the run of 2026-09-13: 142 over 140 read
    # "+0.6k +12.8k BETTER", which is +0.15% to +3.10% of 140's $412,320, and
    # the model's own arithmetic puts the edge at 1.01x over 58 years. 161 over
    # 140 read +5.19% to +13.97% — an order of magnitude larger, and the only
    # one of the two worth acting on.
    solving = spec.get('spend') == 'solve'
    rows = _h2h_rows(specs, res, base, solving)
    at_one = 'common' in res[base]
    ppm = (res[base]['common'] if at_one else res[base])['per_path_mean']
    base_level = float(np.median(ppm)) or 0.0
    verdicts, chart = [], []
    if rows and rows[0][4] == 'ruin':
        dep0 = float(np.mean(res[base]['common']['depleted']))
        plan = res[base]['common']['spend']
        print(f"\n    every strategy at one plan, ${plan / 1e3:,.0f}k/yr fixed: each surviving")
        print(f"    path spends exactly that, so what differs is WHO RUNS OUT.")
        print(f"    {name(base)} runs out on {dep0:.1%} of paths; the change in that, at one plan:")
        print(f"    {'strategy':<{width}}{'P(run out)':>12}{'change':>22}   verdict")
        print("    " + "-" * (width + 46))
        for g, lo, hi, verdict, _ in rows:
            dep = float(np.mean(res[g]['common']['depleted']))
            print(f"    {name(g):<{width}}{dep:>11.1%}{lo * 100:>+12.1f}..{hi * 100:+.1f}pp"
                  f"   {verdict}")
            verdicts.append(verdict)
    else:
        if at_one:
            print(f"\n    how much MORE lifetime average spending than {name(base)}, per year,")
            print(f"    every strategy at one plan (${res[base]['common']['spend'] / 1e3:,.0f}k/yr) — "
                  f"so the gap is the portfolio's, not the solver's")
        else:
            print(f"\n    how much MORE lifetime average spending than {name(base)}, "
                  f"per year")
        print(f"    (${base_level / 1e3:,.0f}k/yr is {name(base)}'s median there, so "
              f"the % column is the size")
        print(f"     of the effect — a resolved difference can still be a small one)")
        print(f"    {'strategy':<{width}}{'low':>10}{'high':>9}{'% of base':>18}"
              f"   verdict")
        print("    " + "-" * (width + 52))
        for g, lo, hi, verdict, _ in rows:
            pct = (f"{lo / base_level:+.1%} .. {hi / base_level:+.1%}"
                   if base_level else '')
            print(f"    {name(g):<{width}}{lo / 1e3:>+9.1f}k{hi / 1e3:>+8.1f}k"
                  f"{pct:>18}   {verdict}")
            verdicts.append(verdict)
            chart.append((name(g), lo, hi))
        _interval_chart(chart, width)
    if 'within solver noise' in verdicts:
        print("\n    'within solver noise': under fixed spending every surviving path spends")
        print("    exactly its plan level, so these intervals are just the gap between two")
        print("    SOLVED levels, each placed only to its +-SE. Compare at one fixed spend.")
    if verdicts and all(v == 'WORSE' for v in verdicts):
        print(f"\n    → every alternative tested is worse than {name(base)}.")
    elif 'BETTER' not in verdicts:
        print(f"\n    → nothing tested beats {name(base)}.")


def _interval_chart(rows, label_width, half=24):
    """Paired intervals on a number line, with zero marked.

    The report's central claim is that a range excluding zero is a real
    difference and one straddling zero is not resolved. That is a question about
    POSITION, which numbers make you compute and a line makes you see: read as
    text, `-31.3k  +13.9k` looks like "roughly break-even" when it means "we
    cannot tell". The verdict column already says so; this stops the numbers
    arguing with it.

    Deliberately NOT a distribution chart. These are PAIRED differences under
    common random numbers — plotting each strategy's own p10/median/p90 instead
    would show overlapping ranges, which this report explicitly says means
    nothing, and it would look more authoritative than the paired result while
    being weaker.

    `rows` is [(label, lo, hi)] in dollars. Silent on a single row: a number
    line with one interval is a worse table.
    """
    rows = [r for r in rows if r[1] == r[1] and r[2] == r[2]]
    if len(rows) < 2:
        return
    span = max(max(abs(lo), abs(hi)) for _, lo, hi in rows)
    if not span:
        return
    span *= 1.08                      # breathing room so an endpoint is not the edge

    def col(v):
        return half + int(round(v / span * half))

    # Build the header into the SAME buffer the rows use, so the labels cannot
    # drift from the axis they describe. Right-aligning 'worse' onto the zero
    # column ran it into 'better' — "worse0better".
    axis = [' '] * (2 * half + 1)
    axis[half] = '0'
    for text, at in (('<- worse', half - 10), ('better ->', half + 2)):
        for k, ch in enumerate(text):
            if 0 <= at + k < len(axis):
                axis[at + k] = ch
    print(f"\n    {' ' * label_width}{''.join(axis)}")
    for label, lo, hi in rows:
        line = [' '] * (2 * half + 1)
        line[half] = '|'
        a, b = sorted((col(lo), col(hi)))
        for x in range(max(a, 0), min(b, len(line) - 1) + 1):
            if line[x] == ' ':
                line[x] = '='
        for x, ch in ((a, '['), (b, ']')):
            if 0 <= x < len(line):
                line[x] = ch
        print(f"    {label:<{label_width}}{''.join(line)}")
    print(f"    {'':<{label_width}}{' ' * half}^ touching this = not resolved")


def _contribution_split(specs, res, spec, name, width):
    """What you PUT IN against what compounding did, and the balance at 65.

    Both are nearly free and neither was reported. The split is closed-form from
    the config — contributions are deterministic — and it reframes the exercise:
    on the live plan it comes out around 2% you, 98% compounding, which says the
    savings rate is a much smaller lever than time or return. The balance at
    retirement comes off `median_path`, which is already retained; the table
    above it ends at the plan-to age, which is not the decision point.
    """
    cfg = _CONFIG_FOR_SPLIT.get('cfg')
    if cfg is None:
        return
    age, horizon = cfg['age'], cfg['time_horizon_years']
    put_in = sum(engine.calculate_annual_contribution(
        age + y, cfg['contribution_schedule'], engine.INFLATION)
        for y in range(horizon))
    start = cfg['initial_value']
    ret_age = cfg.get('retirement_age')
    ret_m = int((ret_age - age) * 12) if ret_age else None

    print(f"\n  WHAT GOT YOU THERE        you put in ${(start + put_in) / 1e6:.2f}M "
          f"(${start / 1e6:.2f}M now + ${put_in / 1e6:.2f}M contributed to "
          f"{age + horizon})")
    show_ret = ret_m is not None and any(
        'median_path' in res[g] and len(res[g]['median_path']) > ret_m for g in specs)
    hdr = f"  {'':<{width}}{'at ' + str(int(ret_age)) if show_ret else '':>12}"
    print(hdr + f"{'at the end':>13}{'from growth':>14}")
    print("  " + "-" * (width + 39))
    for g in specs:
        r = res[g]
        final = r['median'] * 1e6
        growth = max(final - start - put_in, 0.0)
        at_ret = ''
        if show_ret and 'median_path' in r and len(r['median_path']) > ret_m:
            at_ret = f"{r['median_path'][ret_m] / 1e6:.1f}M"
        print(f"  {name(g):<{width}}{at_ret:>12}{final / 1e6:>12.1f}M"
              f"{growth / final if final else 0:>13.0%}")
    # Derived, not asserted. On the live 58-year plan this is ~98% growth and
    # the line below is the honest headline; on a short horizon with a large
    # opening balance it is not, and claiming it anyway would be wrong.
    shares = [max(res[g]['median'] * 1e6 - start - put_in, 0.0)
              / max(res[g]['median'] * 1e6, 1e-9) for g in specs]
    top = max(shares) if shares else 0.0
    if top >= 0.80:
        print(f"\n  Compounding does {top:.0%} of it. That is the argument for")
        print(f"  starting early, and for the horizon mattering more than the rate.")
    else:
        print(f"\n  Contributions and the opening balance are still "
              f"{1 - top:.0%} of the best outcome here —")
        print(f"  a short horizon has not yet let compounding take over.")


def _best_caveat(cells, res, spec):
    """Qualify `best:` when the margin over the runner-up is not a real one.

    Measured on a 5,000-path allocation run: 141 led 140 by 66.85M to 65.31M and
    the line read "best: 141", but the paired interval is [-2.22, +5.02]M and
    141 is ahead on 50.7% of paths — a coin flip. The genuine result in that
    same run went unstated: 141 beats the Vanguard-only mirror 145 by +22 to
    +29M, ahead on 63% of paths. Announcing a winner the data cannot support is
    worse than announcing none, because this is the line a reader takes away.
    """
    field = 'finals' if spec['mode'] == 'accumulation' else 'per_path_mean'
    if len(cells) < 2 or not all(field in res[g] for _, g in cells[:2]):
        return ''
    top, second = cells[0][1], cells[1][1]
    unit, scale = ('M', 1.0) if spec['mode'] == 'accumulation' else ('k', 1e3)
    lo, hi = engine.paired_bootstrap_diff(res[top][field], res[second][field],
                                          np.median)
    # NAME THE DIRECTION. This is top-minus-second, while HEAD-TO-HEAD fifteen
    # lines below is challenger-minus-base — so the SAME pair appears twice with
    # opposite signs and nothing said which was which. Both were right and it
    # read as a contradiction. Decimals match the table below too.
    if hi < 0:
        # THE TWO METRICS CAN DISAGREE, and this branch assumed they could not.
        # `cells` is ranked by the PLAN LEVEL; the interval is measured on
        # REALISED spending. Observed 2026-09-14: 152>152+5@65 led on level by
        # $1k — one fourteenth of the ±$14k the same run printed as its own
        # standard error — while spending $53k LESS, and the line read
        # "best: 152>152+5@65 (beats 152 by [-53.4, -5.5]k)". A negative margin
        # announced as a win.
        se = [res[g].get('level_se') for _, g in cells[:2]]
        se = max([x for x in se if x and x == x], default=None)
        band = (f"; the level column is ±${se / 1e3:,.0f}k here" if se else '')
        return (f"   ⚠ but {second} SPENDS MORE — {top} minus {second} = "
                f"[{lo / scale:+.1f}, {hi / scale:+.1f}]{unit}. This line ranks "
                f"on the certified bar{band}, and the bar and the spending "
                f"disagree. Rank on the life-stage table.")
    if not _straddles_zero((lo, hi)):
        # "beats B by [+5.8, +6.9]M" states the direction — a POSITIVE margin
        # over a named runner-up cannot be read backwards.
        return (f"   (beats {second} by [{lo / scale:+.1f}, "
                f"{hi / scale:+.1f}]{unit})")
    return (f"   — but its margin over {second} is NOT resolved "
            f"({top} minus {second} = [{lo / scale:+.1f}, "
            f"{hi / scale:+.1f}]{unit}, paired)")


def _matrix_pairs(bases, per_base, res, spec, cells):
    """Paired head-to-head for the grid view.

    The matrix prints bare point estimates and TOP 3 prints UNPAIRED intervals,
    which together are the worst available basis for the one question a grid is
    asked. Measured: 141 read 66.85M [63.80, 69.86] against 140's 65.31M
    [62.94, 68.10] — 88% overlap, which reads as "indistinguishable" and is not
    what the data says. Strategies share seeds (42 + sim_id), so path i faces
    the same market in every one of them and the DIFFERENCE is far better
    determined than either level. `_flat_table` has always said so and shown the
    paired interval; the grid view silently dropped it, so the closest pair got
    the weakest test.
    """
    field = 'finals' if spec['mode'] == 'accumulation' else 'per_path_mean'
    have = [g for row in per_base.values() for g in row if g]
    if len(have) < 2 or not all(field in res[g] for g in have):
        return
    unit, scale = ('M', 1.0) if spec['mode'] == 'accumulation' else ('k', 1e3)

    what = ('final wealth' if spec['mode'] == 'accumulation'
            else 'lifetime average spending')

    def block(title, base, rows, caveat=None):
        # The reader has to know THREE things to read a row — what is being
        # differenced, which way the sign runs, and what 'not resolved' means —
        # and the old header said none of them. It explained why the intervals
        # in a DIFFERENT table above should be ignored, which is true and is not
        # what someone looking at this table needs.
        print(f"\n  {title}")
        print(f"    how much MORE {what} than {base}, per year, "
              f"as a 95% range")
        if spec['mode'] != 'accumulation':
            # These numbers are much LARGER than the plan-level gaps in the
            # table above, and adjacent to them: 145 vs 140 read -463.9k here
            # against a 55k gap in plan level. Both are right — the plan level
            # is a certified floor, this is what the paths actually spent, and
            # a rising policy spends far above its floor. Unsaid, it looks like
            # one of the two tables is broken.
            print(f"    (REALISED spending, not the plan level above — a "
                  f"rising policy spends")
            print(f"     well above its certified floor, so these gaps are "
                  f"much larger)")
        if caveat:
            print(f"    ⚠ {caveat}")
        print(f"    {'strategy':<24}{'low':>10}{'high':>9}   verdict")
        print("    " + "-" * 56)
        drawn = []
        for g in rows:
            lo, hi = engine.paired_bootstrap_diff(res[g][field], res[base][field],
                                                  np.median)
            print(f"    {g:<24}{lo / scale:>+9.1f}{unit} {hi / scale:>+8.1f}{unit}"
                  f"   {_verdict(lo, hi)}")
            drawn.append((g, lo, hi))
        _interval_chart(drawn, 24)

    print(f"\n{_RULE}")
    print(f"  HEAD-TO-HEAD   is the difference real, or is it sampling noise?")
    print(f"{_RULE}")
    print(f"    Every strategy is run on the SAME random markets (seed 42 + path"
          f" number),")
    print(f"    so the DIFFERENCE between two of them is pinned far better than"
          f" either")
    print(f"    level is. A range that excludes zero is a real difference; one"
          f" that")
    print(f"    straddles zero is not resolved at this many paths — not"
          f" 'equal'.")

    first = {b: next((g for g in per_base[b] if g), None) for b in bases}
    ref = first[bases[0]]
    if ref and sum(1 for b in bases if first[b]) > 1:
        block('WHICH PORTFOLIO  (before any derisking)', ref,
              [first[b] for b in bases[1:] if first[b]])

    if cells:
        # PICK THE BASELINE ON THE METRIC THIS SECTION REPORTS. `cells` is
        # ranked by the PLAN LEVEL; every number below is realised spending.
        # Mixing the two is what let a wholly negative margin be announced as a
        # win (fixed 7df4395), and it showed again on 2026-09-14: the plan level
        # made 160 the baseline while 160 was the one portfolio that did NOT
        # resolve against 140 — and 152, which did, sat unexamined.
        ranked = sorted(
            (b for b in bases if first[b] and first[b] in res),
            key=lambda b: -float(np.median(res[first[b]][field])))
        home = ranked[0] if ranked else next(
            b for b in bases if cells[0][1] in per_base[b])
        tiers = [g for g in per_base[home] if g and g != first[home]]
        if first[home] and tiers:
            # WHOSE derisking is this? `home` comes from the PLAN LEVEL ranking,
            # which this report now prints a standard error beside because it is
            # the noisiest column here. Observed 2026-09-13 on 140/160/152: the
            # plan level called 160 best, the life-stage table ranked 152 first,
            # and the head-to-head above resolved neither against the other —
            # yet the header said "the best portfolio, 160" as a plain fact and
            # the whole derisk verdict hung off it.
            rivals = [first[b] for b in bases
                      if b != home and first[b] and first[b] in res]
            tied = [g for g in rivals
                    if _straddles_zero(engine.paired_bootstrap_diff(
                        res[g][field], res[first[home]][field], np.median))]
            block(f'WHETHER TO DERISK  (the best portfolio, {first[home]})',
                  first[home], tiers,
                  caveat=(f"{first[home]} is NOT resolved against "
                          f"{', '.join(tied)} — this section is conditional\n"
                          f"     on that pick, and the plan-level ranking that "
                          f"made it is the noisiest\n     column here."
                          if tied else None))


def _matrix(specs, res, spec, portfolios):
    import run as _run
    bases = spec['portfolios']
    columns = _run.bond_tier_columns(spec, portfolios)
    cols = [label for _, label in columns]
    # Key each cell by the bond tier it lands in, so ragged bases (141 has four
    # tiers, 120 has none) line up under the right heading instead of shifting left.
    per_base = {}
    for b in bases:
        mine = [g for g in specs if g == b or g.startswith(f"{b}>")]
        by_tier = {round(_run.target_bond_pct(g, portfolios), 6): g for g in mine}
        per_base[b] = [by_tier.get(t) for t, _ in columns]

    # The matrix must display a metric that VARIES across cells. In fixed-spend
    # mode `spend` is the INPUT, identical everywhere, so a matrix of it says
    # nothing — success rate is what differs when every strategy is handed the
    # same bar. Only when solving does `spend` become an output worth tabulating.
    key = persistence.headline_metric(spec)
    if key == 'median':
        unit, scale, fmt = 'M', 1.0, '{:>11.2f}'
        title = 'MEDIAN WEALTH (real $M)'
    elif key == 'spend':
        unit, scale, fmt = 'k', 1e3, '{:>11.0f}'
        title = f'PLAN LEVEL ({CONFIDENCE:.0%} confidence, real $)'
    else:
        unit, scale, fmt = '%', 0.01, '{:>11.1f}'
        target = res[specs[0]]['spend']
        title = f'SUCCESS RATE at ${target:,.0f}/yr (never depleted, spending held up)'

    # THE PLAN-LEVEL MATRIX IS NOT PRINTED. It is the noisiest column in the
    # report — a poorly-estimated percentile carrying a +/-$13k standard error
    # on a ~$200k figure — and the report already tells the reader three times
    # to rank on realised spending instead. Printing it first made it the
    # headline anyway, and it produced the shape that has now cost four
    # separate fixes (the derisk baseline, the "highest bar" tie, "most
    # spending", IN DETAIL's ranking): a number read to a precision it does not
    # have. The levels still appear in YOUR PLAN IN CONTEXT as a range, which
    # is the honest form.
    #
    # ONLY in solve mode. The same matrix is MEDIAN WEALTH in accumulation and
    # SUCCESS RATE at a fixed spend, and both of those are the main table of
    # their view. `cells` is still built either way — _matrix_pairs ranks on it.
    show_matrix = key != 'spend'
    if show_matrix:
        print(f"\n{_RULE}\n  {title}\n{_RULE}\n")
        print("    " + " " * 26 + "".join(f"{c:>12}" for c in cols))
        print("    " + "-" * (26 + 12 * len(cols)))
    short = _run.distinguish(portfolios, bases)
    cells = []
    for b in bases:
        # Truncate on a WORD boundary and mark it. A hard [:20] cut
        # '145 VG-only (mirrors 140' mid-parenthesis, which reads as a broken
        # label rather than a shortened one.
        lbl = short[b]
        if len(lbl) > 20:
            lbl = lbl[:19].rsplit(' ', 1)[0] if ' ' in lbl[:19] else lbl[:19]
            lbl += '…'
        row = f"    {b:<5} {lbl:<20}"
        for g in per_base[b]:
            if g is None:
                row += f"{'—':>12}"
                continue
            v = res[g][key] / scale
            cells.append((v, g))
            row += fmt.format(v) + unit
        if show_matrix:
            print(row)
    cells.sort(reverse=True)
    best = cells[0][1] if cells else None
    if show_matrix:
        # No tie test here: the views this matrix prints (median wealth,
        # success rate at a fixed spend) carry no error bar, so a tie rule
        # would have nothing to read. `level_se` is set only by the solver,
        # and solve mode does not print this matrix.
        print(f"\n    highest bar: {best}{_best_caveat(cells, res, spec)}")

    # Every row, and with the CAGR/Sortino columns the flat table has always
    # shown. The grid dropped both: it listed 12 strategies and then detailed 3,
    # which is arbitrary, and it printed neither risk column — while paying for
    # them. Accumulation asks the workers for the full monthly return series
    # (light=False) SOLELY to compute Sortino and CAGR, so the grid was buying
    # the expensive result and discarding it.
    # RANK ON THE COLUMN THIS TABLE PRINTS. `headline_metric` returns 'spend'
    # for a solve — the certified LEVEL — while the first column is realised
    # spending, so the label "ranked by spend" named the wrong number and the
    # order was not monotone in anything on screen. Observed 2026-09-14: nine
    # rows whose printed spending ran 464/496/438/457/465/434/455/429/409.
    #
    # Worse than mislabelling: the list is truncated to MAX_DETAIL_ROWS, so a
    # wrong key can DROP the best-spending strategy off the table entirely.
    # `headline_metric` is left alone — persistence uses it to pick what
    # `--diff` compares, which is a different question.
    rank_key = key
    if key == 'spend' and all(res[g].get('median_realised') is not None
                              for g in specs):
        rank_key = 'median_realised'
    ranked = sorted(specs, key=lambda g: -res[g][rank_key])
    shown = ranked[:MAX_DETAIL_ROWS]
    scope = (f"all {len(ranked)}" if len(shown) == len(ranked)
             else f"top {len(shown)} of {len(ranked)}")
    print(f"\n{_RULE}\n  IN DETAIL\n{_RULE}")
    label = {'median_realised': 'lifetime average spending',
             'spend': 'plan level',
             'median': 'median final wealth'}.get(rank_key,
                                                  rank_key.replace('_', ' '))
    print(f"    ({scope}, ranked by {label})")
    if spec['mode'] == 'accumulation':
        print(f"    {'strategy':<26} {'median':>10} {'95% CI':>20} {'CAGR':>7} "
              f"{'vol':>6} {'max DD':>8} {'Sortino':>9}")
        for g in shown:
            r = res[g]
            lo, hi = engine.bootstrap_ci(r['finals'], np.median)
            dd = np.median(r['max_drawdown']) if 'max_drawdown' in r else float('nan')
            print(f"    {g:<26} {r['median']:9.2f}M  [{lo:6.2f}, {hi:6.2f}] "
                  f"{np.median(r['twr_cagr']) * 100:6.2f}% "
                  f"{r.get('annual_vol', float('nan')):5.1f}% {dd:7.1f}% "
                  f"{r['sortino_median']:9.3f}")
    else:
        # NAME THE COLUMNS THE WAY THE POLICY TABLE DOES. These two are
        # different quantities with the same units, and calling them 'spend'
        # and 'p10' invited exactly the confusion it got: `spend` is the PLAN
        # LEVEL — the bar the solver certified, an INPUT to the policy — while
        # `p10` is the tenth percentile of each path's LIFETIME MEAN realised
        # spending, an OUTCOME. On 160 they read 200k and 214k, and the
        # life-stage table below shows a third thing again (realised spending
        # per BAND: 255k/445k/792k). Three numbers, three meanings, and only
        # the multi-policy table said so.
        cols = ['median', 'p10', 'vol', 'bequest', 'worst', 'cagr', 'sortino']
        top, sub, wide = _level_header(cols, 'strategy', 28)
        print('    ' + top)
        print('    ' + sub)
        print("    " + "-" * wide)
        for g in shown:
            print('    ' + _level_row(res[g], cols, str(g), 28))
        print(f"\n    The plan level is the certified BAR; realised spending "
              f"runs above it in")
        print(f"    the median — which is why the lifetime-average column can "
              f"exceed it, and")
        print(f"    why SPENDING BY LIFE STAGE below splits that same average, "
              f"by decade.")
    _sortino_note(shown, res, shown[0] if shown else None)
    # AFTER the detail, not before it. The paired comparison answers "is this
    # difference real", which is a question the reader only has once they have
    # seen the rows — and it reads as the verdict on the table above rather
    # than a preamble to one they have not reached yet.
    _matrix_pairs(bases, per_base, res, spec, cells)


def age_bands(start_ages, end_age):
    """The life-stage table's bands, as (first age, end age, label) in AGES.

    Calendar ages rather than thirds of each retirement (N4): thirds put a 60
    retiree's decades at different ages from a 65 retiree's, so a retirement-
    age choice — the one question where you want to see where the extra years
    go — could not be tabled at all. Decades are counted back from the plan
    age, so the last one is the last decade of the plan; below the first full
    decade the cuts are the options' own retirement ages, so every cell is
    wholly retired or wholly working. At plan to 95 from 65 this is
    65-74/75-84/85-94 — the bands every figure before 2026-10-09 was quoted in.

    Display only. The solver's bands stay thirds; moving them would move every
    certified level.
    """
    lo, end = int(min(start_ages)), int(end_age)
    cuts = {lo, end} | {int(a) for a in start_ages if lo <= a < end}
    cuts |= set(range(end - 10, lo, -10))
    cuts = sorted(cuts)
    return [(a, b, f"{a}-{b - 1}" if b - a > 1 else f"{a}")
            for a, b in zip(cuts, cuts[1:])]


def _spending_shape(rows, config, labels=None, policy_width=0, plan_note=''):
    """What each plan ACTUALLY pays out, by life stage.

    The single headline number cannot compare a flat trajectory to a rising one.
    Measured on portfolio 140: guyton_klinger runs 197k/314k/503k at the median
    with a p10 of 153k/137k/130k, while vpw runs 423k/632k/960k with a p10 of
    168k/181k/228k — vpw ahead in every band in BOTH the median and the bad case,
    at an estate of $5.0M against $31.2M. Ranked on the old headline column the
    order came out the other way round, because that column is the solver's bar
    and the two policies stand in completely different relations to it (p10 of
    the first band over the bar: 100% constant_real, 86% gk, 67% vpw, 47% vpw+5M).

    Free: `spending` is already returned by the verification run.
    """
    labels = labels or {}
    usable = [(pol, g, r) for pol, g, r in rows
              if isinstance(r.get('spending'), np.ndarray) and r['spending'].size]
    if not usable:
        return
    # CALENDAR AGES, shared by every row (N4), so options that retire at
    # different ages line up: a band before a row retires reads 'working'.
    end_age = config['age'] + config['time_horizon_years']
    starts = [int(r.get('start_age', config['retirement_age'])) for _, _, r in usable]
    bands = age_bands(starts, end_age)

    def _row_stats(r, start):
        """One entry per band: None while still working, else (label, median, p10)."""
        n = r['spending'].shape[1]
        out = []
        for a, b, lab in bands:
            if b <= start:
                out.append(None)
                continue
            # cuts include every start age, so a retired band starts at or after it
            out += solver.stage_stats(r['spending'], [(a - start, min(b - start, n), lab)])
        return out

    # RANKED ACROSS BANDS, not within one. Every cell in a row is a single
    # band's average, so the row's own ranking figure is the average ACROSS
    # them — a policy that spends 200k/350k/650k and one that spends
    # 440k/840k/1640k are not comparable on any single decade, and the multi-
    # policy table was printed in policy-major spec order, i.e. unranked.
    # Length-weighted because the bands are uneven: a 5-year stub before the
    # first decade must not count as much as a decade.
    widths = [b - a for a, b, _ in bands]

    def _across_bands(band_stats):
        good = [(st[1], w) for st, w in zip(band_stats, widths)
                if st is not None and st[1] == st[1]]
        if not good:
            return float('-inf')
        return sum(m * w for m, w in good) / sum(w for _, w in good)

    # Carry each row's band stats with it: computed once, used to rank and then
    # to print, so the ordering and the printed cells cannot disagree.
    usable = [(pol, g, r, _row_stats(r, st))
              for (pol, g, r), st in zip(usable, starts)]
    # RANK ON THE COLUMN THE READER CAN SEE. `_across_bands` is the length-
    # weighted mean of the band MEDIANS; the lifetime column is the median of
    # the per-path MEANS. They agree to ~1%, which is enough to order two rows
    # differently from the number printed beside them — the exact confusion
    # this rework is fixing. Fall back when a row predates median_realised.
    lifetimes = [r.get('median_realised') for _, _, r, _ in usable]
    if all(x is not None and x == x for x in lifetimes):
        usable.sort(key=lambda row: -row[2]['median_realised'])
    else:
        usable.sort(key=lambda row: -_across_bands(row[3]))

    gw = max([len(labels.get(g, g)) for _, g, _, _ in usable] + [8])
    # Ruled between bands: without a divider the eye reads "202k 449k" as a pair
    # when they belong to different decades. CELL is one band's (median, low),
    # 15 wide so a three-age retirement choice (five bands) stays near 130.
    CELL, SEP = 15, ' \u2502 '
    # ONE NUMBER, DECOMPOSED — not a second opinion. IN DETAIL prints the
    # lifetime average and this printed the decades, both headed "spend", and
    # the note between them said this table "shows higher numbers again" — so
    # three framings of one quantity read as three quantities. They reconcile:
    # 152's decades of 241k/439k/796k average to 492k against a 496k lifetime
    # figure, the 1% being median-of-means against mean-of-medians. The
    # lifetime column is repeated here as the anchor so the arithmetic is on
    # screen rather than asserted.
    print(f"\n{_RULE}\n  SPENDING BY LIFE STAGE   real $/yr — the lifetime "
          f"average, split by decade\n  ranked by that lifetime average, i.e. "
          f"ACROSS bands and not within one"
          f"{chr(10) + '  ' + plan_note if plan_note else ''}\n{_RULE}")
    stem = f"    {'':<{policy_width}}{'':<{gw}} " if policy_width \
        else f"    {'':<{gw}} "
    head = stem + SEP.join(f"{lab:^{CELL}}" for _, _, lab in bands)
    head += SEP + f"{'estate':>9}"
    sub = f"    {'policy':<{policy_width}}{'strategy':<{gw}} " if policy_width \
        else f"    {'strategy':<{gw}} "
    sub += SEP.join(f"{'median':>7}{'low':>8}" for _ in bands)
    sub += SEP + f"{'median':>9}"
    head = stem + f"{'lifetime':>9}" + SEP + head[len(stem):]
    sub = sub[:len(stem)] + f"{'avg':>9}" + SEP + sub[len(stem):]
    print(head)
    print(sub)
    print("    " + "-" * (len(sub) - 4))
    for pol, g, r, band_stats in usable:
        line = f"    {pol:<{policy_width}}" if policy_width else "    "
        line += f"{labels.get(g, g):<{gw}} "
        life = r.get('median_realised')
        line += (f"{life / 1e3:8,.0f}k" if life is not None else f"{'—':>9}") + SEP
        line += SEP.join(f"{'working':^{CELL}}" if st is None
                         else f"{'—':^{CELL}}" if st[1] != st[1]
                         else f"{st[1] / 1e3:6.0f}k{st[2] / 1e3:7.0f}k"
                         for st in band_stats)
        line += SEP + f"{r['median_bequest'] / 1e6:8.1f}M"
        print(line)
    print(f"\n    'lifetime avg' is the same figure IN DETAIL prints — the decades")
    print(f"    are that money spread over time, not a different measure. They rise")
    print(f"    because the policy ratchets spending up as the portfolio grows; a")
    print(f"    flat policy would show similar decades.")
    if len(set(starts)) > 1:
        print(f"\n    These options retire at different ages, so each 'lifetime avg'")
        print(f"    covers a different span. Compare them band by band; 'working'")
        print(f"    is a band before that option retires.")
    print(f"\n    'low' is the 10th percentile WITHIN each band, so a row is not")
    print(f"    one scenario — it answers 'how bad could this stretch be', band by")
    print(f"    band. This is the comparison the single headline figure cannot make.")


# Below this share of the binding band, the level is mostly your guaranteed
# income's and only slightly the portfolio's. Not a knife edge — the flagged
# rows print their ACTUAL share, so the reader judges; the threshold only
# decides when to draw attention. Measured live at 600 paths, `vpw` with a $10M
# reserve and a ~$60k benefit funds 13% of the binding band from the portfolio
# (level $65,757); the same solve with no benefit at all supports $18,455. At
# 1,500 paths two different portfolios hit share 0.000 and returned exactly
# the benefit / 0.85 to the cent.
GUARANTEED_SHARE_TOL = 0.25


def _guaranteed_floor_note(by_policy, specs, labels=None):
    """Say so when a plan level is Social Security's, not the portfolio's.

    Found 2026-09-12 in a live run. Under `vpw reserve 10M` all twelve
    strategies solved to between 0.80x and 1.00x the annual benefit, and two
    DIFFERENT portfolios (140 and 180) returned the same figure bit-
    identically — both are exactly the benefit / 0.85. A large reserve leaves VPW
    asking for less than the benefit in the bottom decile; `drawn` is then zero
    for a whole band and the band mean is the benefit exactly. The column was a
    property of Social Security, ranked beside columns that measure a portfolio,
    with nothing to tell them apart.
    """
    labels = labels or {}
    hits = [(p, g) for p, res in by_policy.items() for g in specs
            if res[g].get('portfolio_share', 1.0) <= GUARANTEED_SHARE_TOL]
    if not hits:
        return
    by_pol = {}
    for p, g in hits:
        by_pol.setdefault(p, []).append(labels.get(g, g))
    shares = [res[g].get('portfolio_share', 1.0) for p, res in by_policy.items()
              for g in specs if res[g].get('portfolio_share', 1.0)
              <= GUARANTEED_SHARE_TOL]
    print(f"    ⊘ BARELY A PORTFOLIO RESULT — on the paths that set these levels")
    print(f"      the portfolio funded only {min(shares):.0%}–{max(shares):.0%} of "
          f"the binding decade; the rest is")
    floors = {res[g].get('floor', 0.85) for p, res in by_policy.items() for g in specs}
    fl = f"{min(floors):.0%}" if len(floors) == 1 else "adequacy"
    print(f"      your guaranteed income, and the level is close to that divided")
    print(f"      by the {fl} floor. Two different portfolios can land on the")
    print(f"      identical figure to the cent, and have.")
    for pol, gs in by_pol.items():
        shown = ', '.join(gs[:6]) + (f" (+{len(gs) - 6} more)" if len(gs) > 6 else '')
        print(f"        {pol}: {shown}")
    print(f"      Rank these on SPENDING BY LIFE STAGE, never on the level.")
    print()


def _policy_table(specs, by_policy, spec, labels=None):
    """One row per (policy, strategy). Policy moves the answer several times more
    than the derisking choice, so it deserves its own comparison."""
    # In fixed-spend mode `spend` is the INPUT, identical down the column, so
    # naming it "sustainable" would imply it was solved for.
    solving = spec.get('spend') == 'solve'
    # NOT "sustainable": this is the level the solver certified, i.e. the bar
    # against which 90% of paths clear the adequacy floor. Realised spending runs
    # ABOVE it in the median and below it in bad cases, and by wildly different
    # margins per policy — see SPENDING BY LIFE STAGE below, which is the column
    # to rank on.
    col = 'plan level' if solving else 'target'
    head = (f"POLICY COMPARISON ({CONFIDENCE:.0%} confidence, real $)" if solving
            else f"POLICY COMPARISON at ${float(spec['spend']):,.0f}/yr (real $)")
    print(f"\n  {head}\n")
    labels = labels or {}
    sw = 8 if labels else 22
    show_cut = any(res[g]['worst_cut'] > 0.005
                   for res in by_policy.values() for g in specs)
    cut_h = f"{'worst cut':>10}" if show_cut else ''
    pcols = (['median', 'p10', 'vol', 'ruin', 'bequest']
             + (['worst'] if show_cut else []) + ['cagr', 'sortino'])
    ptop, psub, pwide = _level_header(pcols, 'policy', 16 + sw)
    print('    ' + ptop.replace('policy' + ' ' * (16 + sw - 6),
                                f"{'policy':<16}{'strategy':<{sw}}", 1))
    print('    ' + psub)
    print("    " + "-" * pwide)
    for policy, res in by_policy.items():
        for g in specs:
            r = res[g]
            cagr = np.median(r['twr_cagr']) * 100 if 'twr_cagr' in r else float('nan')
            cut = f"{r['worst_cut']:9.1%} " if show_cut else ''
            # ⊘ marks a level SET BY GUARANTEED INCOME, not by the portfolio.
            # See solver.portfolio_share_of_binding_band: on the paths that
            # decide the level, the portfolio funded none of the binding band.
            # ⊘ guaranteed income, ~ a level that started on a bracket edge.
            # An edge that survives its forced step is a real answer, but it is
            # the ONE place the bracket rather than the data chose the number,
            # and a round figure like $200,000 deserves to say so.
            lead = f"{policy:<16}{labels.get(g, g):<{sw}}"
            line = '    ' + _level_row(r, pcols, lead, 16 + sw)
            if r.get('portfolio_share', 1.0) <= GUARANTEED_SHARE_TOL:
                line = line.rstrip() + '  ⊘'      # guaranteed-income level
            print(line)
        print()

    # Only describe the policies actually run — naming ones absent from the table
    # sends the reader looking for rows that are not there. Matched on the BASE
    # policy, because a row is tagged "vpw reserve 5M", not "vpw": keying on the
    # full tag silently suppressed this whole block once the reserve frontier
    # started fanning rows out.
    tradeoff = {
        'constant_real': 'constant_real buys spending with depletion risk',
        'vpw': 'vpw converts bequest into spending — it amortises the portfolio',
        'fixed_pct': 'fixed_pct takes the hit as deeper cuts in bad years',
        'guyton_klinger': 'guyton_klinger preserves the portfolio by cutting',
    }
    seen, notes = set(), []
    for tag in by_policy:
        base = next((k for k in tradeoff if tag == k or tag.startswith(k + ' ')), None)
        if base and base not in seen:
            seen.add(base)
            notes.append(tradeoff[base])
    if len(notes) > 1:
        print("    Higher spending is never free:")
        for n in notes:
            print(f"      · {n}")

    print(f"\n    Every column above is something the paths PRODUCED. The bar")
    print(f"    the solver certified is in YOUR PLAN IN CONTEXT, with its")
    print(f"    precision. Rank on the life-stage table below.")

    if any(p.startswith('vpw') for p in by_policy):
        cfg = _CONFIG_FOR_SPLIT.get('cfg')
        plan_age = int(policies.vpw_planning_age(
            cfg['age'] + cfg['time_horizon_years']) if cfg
            else policies.VPW_PLANNING_AGE)
        print(f"\n    VPW amortises to age {plan_age}, not to the horizon, so it leaves a")
        print(f"    balance at the end even with no reserve. A reserve is held back")
        print(f"    on top of that: it is a dial, not a target. The 'bequest' column")
        print(f"    is what actually lands.")

    # Measured, not assumed: with sourcing=pro_rata these agree to 3e-8 across
    # policies (weights come out unchanged, so the return path is identical under
    # common random numbers). With sourcing=excess they do NOT — selling from
    # what is overweight rebalances as it withdraws, so a policy that draws more
    # rebalances more. Observed spread on a 200-path check: 0.04pp of CAGR,
    # 0.003 of Sortino. The old note claimed these were "identical ... they
    # ignore withdrawals", which is wrong in mechanism and invited the reader to
    # distrust a table that plainly showed 7.80 / 7.77 / 7.79.
    sourcing = spec.get('sourcing', 'excess')
    print(f"\n    CAGR and Sortino describe the PORTFOLIO, not the plan. Under")
    if sourcing == 'pro_rata':
        print(f"    sourcing pro_rata they are identical across policies: selling")
        print(f"    every holding pro-rata leaves the weights, and so the return")
        print(f"    path, untouched.")
    else:
        print(f"    sourcing {sourcing} they vary slightly across policies (~0.04pp)")
        print(f"    because selling from what is overweight rebalances as it")
        print(f"    withdraws — a policy that draws more rebalances more. The")
        print(f"    difference is that, not a difference in the portfolios.")


def _choice_table(specs, by_policy, spec, config, labels=None):
    """RETIREMENT CHOICES: one row per option, each solved at its own level.

    Here the LEVEL is the answer, not noise around one. Retiring later or
    claiming later is worth exactly what it does to the plan, so each option
    is solved, and the paired column reads realised spending at each option's
    own level against your current choice. The gap is the choice's, solver aim
    included, which is what the choice costs or buys.
    """
    import run as _run
    ch = spec['compare_choice']
    dim, values = ch['dimension'], [float(v) for v in ch['values']]
    base_tag = _run.choice_tag(dim, _run.choice_baseline(config, dim, values))
    mine = _run.current_choice(config, dim)
    yours = _run.choice_tag(dim, mine) if mine is not None else None
    g = specs[0]
    name = (labels or {}).get(g, g)
    tw = max(len(t) for t in by_policy) + len(' (yours)') + 2

    def row(t):
        return t + (' (yours)' if t == yours else '')

    print(f"\n{_RULE}")
    print(f"  RETIREMENT CHOICES   {_run.PLAN_CHOICES[dim][0].lower()} · {name} · "
          f"{spec.get('policy', 'guyton_klinger')} · {CONFIDENCE:.0%} confidence")
    print(_RULE)
    cols = [_level_key(spec, by_policy), 'se', 'median', 'p10', 'ruin', 'bequest']
    top, sub, wide = _level_header(cols, 'option', tw)
    print('    ' + top)
    print('    ' + sub)
    print('    ' + '-' * wide)
    for t, res in by_policy.items():
        print('    ' + _level_row(res[g], cols, row(t), tw))

    base = by_policy[base_tag][g]
    others = [t for t in by_policy if t != base_tag]
    if others:
        b_med = float(np.median(base['per_path_mean']))
        print(f"\n    compared with {row(base_tag)}, on the same paths:")
        print(f"    lifetime average spending, each option at its own plan level")
        print(f"    {'option':<{tw}}{'low':>9}{'high':>9}{'% of yours':>22}   verdict")
        print("    " + "-" * (tw + 52))
        for t in others:
            lo, hi = engine.paired_bootstrap_diff(
                by_policy[t][g]['per_path_mean'], base['per_path_mean'], np.median)
            verdict = 'MORE' if lo > 0 else ('LESS' if hi < 0 else 'not resolved')
            pct = f"{lo / b_med:+.1%} .. {hi / b_med:+.1%}"
            print(f"    {row(t):<{tw}}{lo / 1e3:>+8.1f}k{hi / 1e3:>+8.1f}k"
                  f"{pct:>22}   {verdict}")

    print()
    if dim == 'retirement_age':
        end = config['age'] + config['time_horizon_years']
        spans = '; '.join(f"from {v:g}, {end - v:g} years" for v in values)
        print(f"    Per-year figures cover different spans of retirement: {spans}.")
        print(f"    Each option stops every contribution the year before it retires;")
        print(f"    a later age does not extend an account that stops sooner.")
    elif dim == 'ss_claim_age':
        print(f"    The plan level sees delaying least: a later claim pays off in long")
        print(f"    lives and bad markets, so read the low case and the last decade.")
    elif dim == 'contributions':
        print(f"    ×1 is today's contributions, in every account.")


MAX_FRAGILE_LISTED = 10


def _verify_sims(by_policy, specs):
    """Paths behind the reported confidence, for sizing its standard error."""
    for res in by_policy.values():
        for g in specs:
            stat = (res.get(g) or {}).get('adequacy_stat')
            if stat is not None and len(stat):
                return int(len(stat))
            sp = (res.get(g) or {}).get('spending')
            if sp is not None and getattr(sp, 'shape', None):
                return int(sp.shape[0])
    return 0


def confidence_note(by_policy, specs, confidence=CONFIDENCE, labels=None):
    """Say when a row did not actually land on the confidence it was aimed at.

    The header claims "90% confidence" for every row, but a solved level is
    estimated from a finite sample and lands near the target rather than on it.
    At production sims the four policies came in at 89.7-90.2%; at the reduced
    counts used for smoke runs the gap widens to roughly one standard error of
    the verification sample. A closed-form row lands exactly. Printing the gap
    beats implying there isn't one.
    """
    labels = labels or {}
    off = []
    for policy, res in by_policy.items():
        for g in specs:
            got = res[g].get('success_rate')
            if got is None or abs(got - confidence) <= 0.01:
                continue
            pol = f"{policy}/" if len(by_policy) > 1 else ""
            off.append((abs(got - confidence), f"{pol}{labels.get(g, g)}", got))
    if not off:
        return
    off.sort(reverse=True)
    # A row at 100% has NOT missed its target — nothing can go wrong on any
    # path, which happens when Social Security alone covers the plan level. The
    # run that surfaced this had SS above VT's whole plan level, so
    # the portfolio funded $0 of it and every path succeeded. Listing that
    # beside a row that genuinely landed at 91.3% invites the reader to treat
    # them as the same kind of miss, and the note then contradicted itself by
    # adding "closed-form rows land exactly" on a closed-form run.
    sat = [(t, g) for _, t, g in off if g >= 0.9999]
    miss = [(d, t, g) for d, t, g in off if g < 0.9999]
    print(f"\n{_RULE}\n  CONFIDENCE   aimed at {confidence:.0%}, achieved:\n{_RULE}")
    if sat:
        # STATE THE OBSERVATION, not an inferred cause. The first version said
        # "guaranteed income already covers the plan level", which was inferred
        # from one run and is only true from the CLAIM AGE — that run's SS
        # started at 67 against withdrawals from 65, so the portfolio still
        # funded two full years alone, and the bare VT row did NOT saturate
        # while its derisked siblings did. Asserting the mechanism got it wrong
        # in the very output that prompted it.
        print(f"    NOT BINDING — every path succeeds, so the portfolio is not")
        print(f"    what limits these. Usually guaranteed income covering the")
        print(f"    plan level; check the claim age against when withdrawals")
        print(f"    start, since years before it are funded by the portfolio alone:")
        for tag, _ in sat:
            print(f"      {tag}: 100%")
        if miss:
            print("")
    # SIZE THE GAP AGAINST THE SAMPLING ERROR, rather than asserting it IS the
    # sampling error. On the run of 2026-09-13, 142>142+5@65 came in at 87.7%
    # against a 90% target — 2.3pp, or about 3 standard errors at 1,500 paths —
    # under a note saying the gap is "roughly one standard error". It was not
    # noise: the level was a bracket probe the verification pass had skipped.
    n = _verify_sims(by_policy, specs)
    se = ((confidence * (1 - confidence) / n) ** 0.5) if n else 0.0
    for _, tag, got in miss[:6]:
        sigma = abs(got - confidence) / se if se else 0.0
        flag = '' if sigma <= 2.0 else '   ← NOT sampling error'
        suffix = f"   ({sigma:.1f} SE)" if se else ''
        print(f"    {tag}: {got:.1%}{suffix}{flag}")
    if len(miss) > 6:
        print(f"    … and {len(miss) - 6} more")
    if not miss:
        return
    if se:
        print(f"    One standard error at {n:,} verification paths is "
              f"{se:.2%}, so a gap")
        print(f"    inside ~2 SE is the finite sample and nothing more. Raise")
        print(f"    verify_sims to narrow it.")
        if any(abs(g - confidence) > 2.0 * se for _, _, g in miss):
            print(f"    A row marked NOT sampling error did not converge — its "
                  f"level sits")
            print(f"    where the solver left it, not where {confidence:.0%} "
                  f"actually is. Treat it")
            print(f"    as unresolved rather than as a result.")
    else:
        print(f"    A solved level is estimated from a finite sample, so it "
              f"lands near")
        print(f"    the target rather than on it. Raise verify_sims to narrow "
              f"the gap.")
    print(f"    A closed-form row (vpw) lands exactly only when the adequacy")
    print(f"    statistic has no TIES — guaranteed income creates them, because")
    print(f"    many paths then clear the floor by the same amount, and the")
    print(f"    quantile lands on a step rather than between two points.")


def _worst_at(floor, by_policy, specs):
    """Lowest success rate at a given floor across everything in this run."""
    vals = []
    for res in by_policy.values():
        for g in specs:
            bf = {float(k): v for k, v in
                  (res.get(g, {}).get('success_by_floor') or {}).items()}
            if floor in bf:
                vals.append(bf[floor])
    return min(vals) if vals else None


def _span(lo, hi, fmt='.1%'):
    """'83.0%' when both ends agree, '80.1%–83.0%' when they do not."""
    return f"{lo:{fmt}}" if f"{lo:{fmt}}" == f"{hi:{fmt}}" else f"{lo:{fmt}}–{hi:{fmt}}"


def _floor_guidance(grid, worst=None, unit='life stage'):
    """How to read the strictest column. Printed on EVERY exit from robustness().

    It used to hang off the end of the fragile-listing branch, so the one case
    that shows nothing but the stricter-floor line — which is the common case,
    and the one a reader is least equipped to interpret — got the numbers
    without the guidance.
    """
    print()
    print(f"    HOW TO USE THIS: the {grid[-1]:.0%} number is the one that "
          f"matters if your")
    # NOT "never have to cut". The strict floor scores each life stage's
    # AVERAGE, so a path with a year below plan inside a stage still passes.
    print(f"    spending is mostly fixed — housing, insurance, care. It is the "
          f"share of")
    # FROM THIS RUN, not a remembered example. It read "At 77%, roughly one
    # path in four" under a run whose own strictest column was 80-82% — the
    # same class of hardcoded figure as the premium span that kept printing a
    # retracted number for two days.
    if worst is not None and 0 < worst < 1:
        share = 1.0 / max(1.0 - worst, 1e-9)
        print(f"    paths where no {unit} averages below plan. Yours reads "
              f"{worst:.0%}, so about")
        print(f"    one path in {share:.0f} has a stretch of years that does, and "
              f"the level above")
        print(f"    is only safe if you can take that cut.")
    else:
        print(f"    paths where no {unit} averages below plan. Whatever it "
              f"reads, the")
        print(f"    remainder is the share of paths with a stretch of years that "
              f"does — and")
        print(f"    the level above is only safe if you can take that cut.")


def robustness(by_policy, specs, confidence=CONFIDENCE, labels=None):
    """Does the answer survive a stricter adequacy standard?

    Free: each strategy's spending trajectories are already in hand, so success
    at other floors is a different threshold on the same data.

    Summarised rather than listed when every result fails alike — that is ONE
    fact about the plan, not N facts about the strategies.

    When they DON'T fail alike, the list is ordered worst-first and its length is
    always stated. It used to be `sorted(fragile.items())[:6]`: alphabetical, cut
    at six, with no indication anything had been dropped. On a 4-policy run that
    printed 6 of 10 warnings and silently swallowed every `vpw no reserve` row —
    the highest-spending option in the table, and precisely the one whose
    fragility the reader needed. Ordering by severity and stating the count is
    the fix; the cap is a readability limit, never a silent one.
    """
    labels = labels or {}
    grid = solver.FLOOR_GRID
    print(f"\n{_RULE}\n  ROBUSTNESS   what if you CAN'T cut spending when the plan "
          f"says to?\n{_RULE}")

    # A LEGEND, because the bracketed triples were unreadable without one. The
    # section used to open straight into "guyton_klinger/145: below 90% at a 85%
    # floor  [70%:95% 85%:90% 100%:75%]" and never said what a floor was, what
    # the three numbers were, or which of them the run had actually solved for.
    solve_floor = max((res[g].get('floor', 0.85)
                       for res in by_policy.values() for g in specs), default=0.85)
    # The rule the results were scored under, in words: "life stage" was the
    # thirds' word and reads wrong for a rolling window.
    unit = solver.adequacy_unit(next(
        (res[g]['adequacy'] for res in by_policy.values() for g in specs
         if res.get(g, {}).get('adequacy')), solver.DEFAULT_ADEQUACY))
    print(f"    Every level above is certified as: in {confidence:.0%} of paths, "
          f"average spending in")
    print(f"    EVERY {unit} stays at or above {solve_floor:.0%} of it. That "
          f"{solve_floor:.0%} is an assumption")
    print(f"    about YOU — the pay cut you could absorb. Re-scoring the same "
          f"paths against")
    print(f"    other assumptions is free, and [a:b] below reads "
          f"'floor:{'%'} of paths clearing it':")
    print()
    for f in grid:
        if f >= 0.999:
            desc = "no life stage may average below plan — the strict reading"
        elif abs(f - solve_floor) < 1e-9:
            desc = (f"you could take a {1 - f:.0%} cut"
                    f"{' ' * max(0, 10 - len(f'{1 - f:.0%}'))}← what was solved for")
        else:
            desc = f"you could take a {1 - f:.0%} cut"
        print(f"      {f:>6.0%}   {desc}")
    print()

    # A JSON round-trip turns these keys into STRINGS, and `grid` holds floats,
    # so `by_floor.get(0.7)` silently returned the 1.0 default and flagged
    # nothing. Live runs pass dicts with float keys and were unaffected, which
    # is why it survived — any path that re-renders a loaded run record did not.
    def _floors(res_g):
        return {float(k): v for k, v in (res_g.get('success_by_floor') or {}).items()}

    # SPLIT BY WHETHER THE FAILURE MEANS ANYTHING. The solver certifies
    # `confidence` AT the solve floor, so scoring below `confidence` at a
    # STRICTER floor is arithmetic, not a finding — it is true of every result
    # by construction. Reporting the two together produced "36 of 36 results
    # fall below 90% at some floor" on the run of 2026-09-12, where 32 of the 36
    # failed only at the 100% floor and the 4 that mattered — results missing
    # the confidence they were certified at — were buried inside that headline.
    fragile, stricter = {}, {}
    for policy, res in by_policy.items():
        for g in specs:
            by_floor = _floors(res[g])
            failing = [f for f in grid if by_floor.get(f, 1.0) < confidence]
            if not failing:
                continue
            if min(failing) <= solve_floor + 1e-9:
                fragile[(policy, g)] = (min(failing), by_floor)
            else:
                stricter[(policy, g)] = (min(failing), by_floor)

    total = len(by_policy) * len(specs)
    if stricter:
        rates = [bf[f] for f, bf in stricter.values()]
        print(f"    {len(stricter)} of {total} hold at their own {solve_floor:.0%} "
              f"floor but not at a stricter one")
        print(f"      ({_span(min(rates), max(rates))} success at a "
              f"{min(f for f, _ in stricter.values()):.0%} floor). That is "
              f"arithmetic, not")
        print(f"      fragility: a level certified at {solve_floor:.0%} must score "
              f"lower against a higher bar.")
        if not fragile:
            print(f"      Nothing here missed the confidence it was certified at.")
        print()
    if not fragile:
        if not stricter:
            print(f"    ✓ all {total} hold from a {grid[0]:.0%} floor to "
                  f"{grid[-1]:.0%} —")
            print(f"      your spending flexibility does not change the answer")
        _floor_guidance(grid, _worst_at(grid[-1], by_policy, specs), unit)
        return

    worst_floor = min(f for f, _ in fragile.values())
    if len(fragile) == total and len({f for f, _ in fragile.values()}) == 1:
        # One fact about the plan, not N facts about the strategies.
        rates = [bf[worst_floor] for _, bf in fragile.values()]
        print(f"    ⚠ all {len(fragile)} fall below {confidence:.0%} at a "
              f"{worst_floor:.0%} floor ({_span(min(rates), max(rates))} "
              f"success there).")
        print(f"      Every result here assumes you can absorb some spending cut.")
    else:
        # Worst first: the lowest floor at which it fails, then the lowest
        # success rate there. Alphabetical order carried no information and put
        # the decision-relevant rows last, where the cap removed them.
        ranked = sorted(fragile.items(), key=lambda kv: (kv[1][0], kv[1][1][kv[1][0]]))
        # The solver declares convergence at |gap| <= 1pp, so a 89.x% read
        # against a 90% target is INSIDE its own tolerance and is not a defect.
        # Saying "missed" without that reads as a failure when it is the
        # sampling error the solver already accounts for.
        gaps = [confidence - bf[f] for f, bf in fragile.values()]
        inside = sum(g <= 0.01 for g in gaps)
        note = (f" — all within the solver's 1pp tolerance" if inside == len(gaps)
                else f" — {len(gaps) - inside} by more than the solver's 1pp tolerance")
        print(f"    {len(fragile)} of {total} came in below the {confidence:.0%} "
              f"they were certified at,")
        print(f"    {note.lstrip(' —')}. Worst first:")
        for (policy, g), (floor, bf) in ranked[:MAX_FRAGILE_LISTED]:
            # One decimal: 89.6% rounded to '90%' on a row flagged 'below 90%'.
            rates = "  ".join(f"{f:.0%}:{bf[f]:.1%}" for f in grid)
            tag = labels.get(g, g)
            pol = f"{policy}/" if len(by_policy) > 1 else ""
            print(f"    ⚠ {pol}{tag}: below {confidence:.0%} at a {floor:.0%} floor"
                  f"   [{rates}]")
        hidden = ranked[MAX_FRAGILE_LISTED:]
        if hidden:
            floors = {f"{f:.0%}" for (_, _), (f, _) in hidden}
            print(f"    … and {len(hidden)} more, all failing only at a "
                  f"{'/'.join(sorted(floors))} floor")
    _floor_guidance(grid, _worst_at(grid[-1], by_policy, specs), unit)


def _diagnostics(res, specs, labels, config, spec):
    """The four --debug blocks. Silent on a normal run, and on any run whose
    paths were not kept.

    Gated on `instrument.ENABLED` rather than a flag of its own: --debug is
    already the switch for "tell me about the machinery", and a second one
    would be a second thing to remember. The per-path arrays ride on every run
    regardless (they are six numbers each), so this only decides whether to
    SHOW them — a module global could not decide whether to COMPUTE them
    anyway, because workers re-import under spawn.
    """
    if not getattr(instrument, 'ENABLED', False):
        return
    if spec.get('mode') != 'retirement':
        return                      # the four questions are all decumulation
    keep = {}
    for g in specs:
        raw = (res.get(g) or {}).get('raw')
        if raw:
            keep[labels.get(g, g)] = dict(raw, spend=res[g].get('spend', 0.0))
    if not keep:
        return
    floor = config.get('withdrawal', {}).get('floor', 0.85)
    print(f"\n{_RULE}\n  DIAGNOSTICS   --debug: why these numbers, and did the "
          f"machinery behave\n{_RULE}")
    try:
        diagnostics.report(keep, config, floor)
    except Exception as exc:        # diagnostics must never cost a real result
        print(f"    (diagnostics unavailable: {exc})")


def _rebalancing_summary(res, specs, labels=None):
    """Turnover for EVERY strategy, then the machinery for one.

    The header used to read "(representative — the machinery barely varies by
    allocation)", and measuring turnover on 2026-09-14 falsified it: VT trades
    0.00%/yr because a single fund cannot drift, while a nine-fund book runs
    ~3.4%. That is not "barely varies" — it is the difference between zero and
    something, and it is a portfolio CHARACTERISTIC the report was hiding
    behind a claim about the machinery.

    So turnover is now per strategy and the twenty-line band breakdown stays on
    one, which is what the original note was really defending.
    """
    labels = labels or {}
    have = [g for g in specs if res[g].get('rebalancing')]
    if not have:
        return
    from montecarlo import report as _acc
    print(f"\n  REBALANCING   turnover is a property of the ALLOCATION: a book "
          f"that cannot")
    print(f"                drift cannot trade.")
    width = max(len(str(labels.get(g, g))) for g in have)
    for g in have:
        rb = res[g]['rebalancing']
        t = rb.get('turnover_per_year')
        n = rb.get('avg_events_per_year')
        print(f"    {labels.get(g, g):<{width}}  {t:6.2%}/yr"
              f"   {n:5.1f} events/yr" if t is not None else
              f"    {labels.get(g, g):<{width}}       -")
    print(f"\n    at a 15bp round trip that is "
          f"{min(res[g]['rebalancing']['turnover_per_year'] for g in have) * 0.0015:.3%}"
          f"-{max(res[g]['rebalancing']['turnover_per_year'] for g in have) * 0.0015:.3%}"
          f"/yr, which the model does NOT charge.")
    print(f"    The tax on realised gains in a taxable sleeve is the version "
          f"that matters.")
    lead = have[0]
    print(f"\n    band machinery, {labels.get(lead, lead)}:")
    _acc.print_rebalancing_summary(res[lead]['rebalancing'])


def accumulation_figures(res, specs, portfolios, config, outdir=None):
    """The five accumulation figures, saved rather than shown.

    The data is already there: accumulation runs light=False for Sortino, which
    is the same worker output the figures need.
    """
    import matplotlib
    matplotlib.use('Agg')          # before report imports pyplot
    from montecarlo import report as _acc

    # report.build_figures indexes `portfolios[label]` for a name and the COST
    # weight. A label here is a glide SPEC ('141>141+5@65'), which is not a
    # template id and has no single allocation — so it gets the STARTING one,
    # which is what the annotation describes.
    views = {}
    for g in specs:
        _, alloc = glide.build_plan(glide.parse_schedule(g, config['age']),
                                    portfolios)[0]
        views[g] = {'name': g, 'allocation': alloc}

    figs = {g: dict(res[g], final_values=np.asarray(res[g]['finals']) * 1e6)
            for g in specs}
    drawdowns = {g: res[g]['max_drawdown'] for g in specs}
    built = _acc.build_figures(figs, views, list(specs), config, drawdowns)
    outdir = outdir or (Path(__file__).resolve().parent.parent / 'output' / 'figures')
    return _acc.save_figures(built, outdir, prefix='accumulation')


def _figure_data(by_policy, specs, config, one_plan=False):
    """What the decumulation figure plots, per strategy, and the plan it is at.

    With `one_plan`, every strategy as its one-plan run saw it — the runs
    RESULTS, the head-to-head and the life-stage table read. It used to plot
    each strategy's OWN level beside tables that compared them at one, so the
    chart and the numbers under it disagreed (103: ~$70k/yr drawn, $149k in
    the table).
    """
    res = by_policy[next(iter(by_policy))]
    if one_plan:
        res = _at_one_plan(specs, res)
    end_age = config['age'] + config['time_horizon_years']
    series = {}
    for g in specs:
        r = res[g]
        sp = np.asarray(r['spending'], dtype=float)
        if sp.size == 0:
            continue
        ages_d = [a for a in r.get('depletion_ages', []) if a is not None]
        n = max(len(r.get('depletion_ages', [])), 1)
        grid = np.arange(r['start_age'], end_age + 1)
        series[g] = {
            'ages': np.arange(r['start_age'], r['start_age'] + sp.shape[1]),
            'median': np.median(sp, axis=0) / 1e3,
            'p10': np.percentile(sp, 10, axis=0) / 1e3,
            'p90': np.percentile(sp, 90, axis=0) / 1e3,
            'depletion_ages': grid,
            'depleted': [100.0 * sum(1 for a in ages_d if a <= x) / n for x in grid],
        }
    note = _one_plan_note(specs, by_policy[next(iter(by_policy))],
                          lambda g: g) if one_plan else ''
    return series, note


def figures(by_policy, specs, config, outdir=None, one_plan=False):
    """Spending fan chart + depletion curve. Saved, not shown, so a run is
    non-blocking; the tables are the primary output."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    outdir = Path(outdir or (Path(__file__).resolve().parent.parent / 'output' / 'figures'))
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y-%m-%d_%H%M')
    written = []

    policy = next(iter(by_policy))
    series, note = _figure_data(by_policy, specs, config, one_plan=one_plan)
    bands = len(series) <= 2      # bands only stay legible for one or two strategies

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    for g, s in series.items():
        line, = ax1.plot(s['ages'], s['median'], lw=2, label=g)
        if bands:
            ax1.fill_between(s['ages'], s['p10'], s['p90'],
                             alpha=0.18, color=line.get_color())
    ax1.set_xlabel('Age'); ax1.set_ylabel('Annual spending ($k, real)')
    ax1.set_title(f'Realised spending — {policy}' +
                  ('  (band = 10th–90th pct)' if bands else ''))
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    for g, s in series.items():
        ax2.plot(s['depletion_ages'], s['depleted'], lw=2, label=g)
    ax2.set_xlabel('Age'); ax2.set_ylabel('Cumulative P(depleted) %')
    ax2.set_title('Depletion by age')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    fig.suptitle('Decumulation — real 2026 dollars' + (f' · {note}' if note else ''),
                 fontweight='bold')
    fig.tight_layout()
    path = outdir / f'decumulation_{stamp}.png'
    fig.savefig(path, dpi=110); plt.close(fig)
    written.append(path)
    return written


_CONFIG_FOR_SPLIT = {}


def run(spec, specs, portfolios, config, is_grid):
    # The accumulation table needs the contribution schedule and retirement age,
    # and _flat_table's signature is already five wide and reached from three
    # call sites. Stashed rather than threaded.
    _CONFIG_FOR_SPLIT['cfg'] = config
    print(f"\n{'=' * 84}")
    print(f"  {'RETIREMENT' if spec['mode'] == 'retirement' else 'ALLOCATION'} ANALYSIS"
          f"   ·  ages {config['age']}–{config['age'] + config['time_horizon_years']}"
          f"  ·  real 2026 dollars ({engine.INFLATION:.0%} inflation removed)")
    if spec['mode'] == 'retirement':
        import run as _r
        wd = config['withdrawal']
        # Naming config['withdrawal']['policy'] here described ONE policy above a
        # table holding four, so a saved transcript read as if every row were
        # guyton_klinger. Name what actually runs.
        tags = [t for t, _, _ in _r.policy_runs(spec)]
        shown = tags[0] if len(tags) == 1 else ', '.join(tags)
        print(f"  {shown} · sourcing {wd['sourcing']} · withdrawals from "
              f"{wd['start_age']}")
        adequacy = solver.adequacy_mode(
            config.get('adequacy', solver.DEFAULT_ADEQUACY))
        # ANY, not every: a banded rule scores the WORST stretch's mean, so one
        # stretch below the floor fails the path.
        scope = (f"any {solver.adequacy_unit(adequacy)}'s average"
                 if adequacy != 'lifetime' else "lifetime average")
        print(f"  counts as failure: ran out, or {scope} spending fell below "
              f"{wd['floor']:.0%} of plan")
    # EVERY knob, not just this one. Twelve environment variables move every
    # number in the report and only PREMIUM_SCALE was announced — and it was
    # announced as "equity premium scaled", which is the mislabel this repo
    # already corrected in the footer: MC_PREMIUM_SCALE moves the FACTOR premia
    # over the market proxy and leaves the equity premium over bonds alone.
    # They reach the run record through provenance, so --diff could see them
    # while the run on screen looked identical to a default one.
    #
    # NOT gated on --debug: a silently moved assumption is a correctness
    # problem, not a diagnostic nicety.
    diagnostics.assumption_banner()
    print("=" * 84 + "\n")

    import run as _run
    labels = _run.strategy_labels(specs)
    _legend(specs, labels, portfolios, config)

    # Comparing VPW estate reserves fans out along a second axis: one run per
    # (policy, reserve), labelled so the frontier reads as a spread rather than
    # a list of near-identical policy names. Defined in run.policy_runs so the
    # time estimate and the header count the same runs this loop performs.
    runs = _run.policy_runs(spec)

    by_policy, cfgs = {}, {}
    for tag, policy, settings in runs:
        cfg = dict(config)
        settings = dict(settings or {})
        choice = settings.pop('choice', None)
        if 'withdrawal' in cfg:
            wd = dict(cfg['withdrawal'], policy=policy)
            # Only what this run actually sets: a None would otherwise overwrite
            # a value build_config had already put there.
            for k, v in (settings or {}).items():
                if v is not None:
                    wd[k] = v
            wd.setdefault('vpw_bequest', 0.0)
            cfg['withdrawal'] = wd
        if choice:
            cfg = _run.apply_choice(cfg, *choice)
        if len(runs) > 1:
            print(f"\n  ── {tag} " + "─" * max(4, 60 - len(tag)))
        cfgs[tag] = cfg
        with instrument.timed(tag):
            by_policy[tag] = evaluate_strategies(specs, portfolios, cfg, spec, labels)

    res = by_policy[runs[0][0]]

    if spec.get('compare_choice'):
        _choice_table(specs, by_policy, spec, config, labels)
    elif len(runs) > 1:
        _policy_table(specs, by_policy, spec, labels)
    elif is_grid:
        _matrix(specs, res, spec, portfolios)
    else:
        _flat_table(specs, res, spec, specs[0], labels)

    # Right after the head-to-head it qualifies: the table says which portfolio
    # wins under the model, this says whether it still wins if the model is
    # wrong. One policy only — comparing policies is a question about the plan.
    views = None
    if (spec.get('return_views') and spec['mode'] == 'retirement'
            and len(runs) == 1 and len(specs) > 1):
        from montecarlo import return_views
        with instrument.timed('return views'):
            views = return_views.check(specs, res, cfgs[runs[0][0]], portfolios,
                                       labels)

    # AFTER WHICHEVER TABLE RAN, not inside one of them. It used to hang off the
    # end of _policy_table, so it only appeared on MULTI-POLICY runs — and a
    # single-policy run is the common case. Missed live on 2026-09-13: a VT
    # benchmark solved to $61,390 against a $49,916 benefit, i.e. almost
    # entirely Social Security, printed beside 161's $196,500 with nothing to
    # separate them.
    # For a plan choice, the context is YOUR option's: the first row may retire
    # at another age, and this block names the age it is describing.
    ctx = runs[0][0]
    if spec.get('compare_choice'):
        ch = spec['compare_choice']
        ctx = _run.choice_tag(ch['dimension'], _run.choice_baseline(
            config, ch['dimension'], ch['values']))
    _plan_in_context(specs, by_policy[ctx], cfgs[ctx],
                     indent='  ', name=lambda g: (labels or {}).get(g, g),
                     spec=spec, by_policy=by_policy)
    _guaranteed_floor_note(by_policy, specs, labels)

    # The decision view. Costs nothing — the spending trajectories are already
    # in hand — and it is the only table here that compares a flat plan with a
    # rising one on the same terms.
    if spec['mode'] == 'retirement':
        pw = max((len(t) for t, _, _ in runs), default=0) + 2 if len(runs) > 1 else 0
        shape_rows, note = _shape_rows(runs, specs, by_policy, spec, labels, is_grid)
        _spending_shape(shape_rows, config, labels, policy_width=pw, plan_note=note)

    try:
        with instrument.timed('persistence'):
            saved = persistence.save_run(spec, by_policy, specs, config,
                                         return_views=views)
        print(f"\n  saved → {saved.relative_to(Path(__file__).resolve().parent.parent)}")
    except Exception as e:                          # a run is worth keeping even
        print(f"\n  (run not saved: {e})")         # if the record cannot be written

    # BOTH MODES. The summary survived the deletion of Portfolio Analysis.py but
    # only on the path that runner used, so no retirement run has ever shown it
    # — and until 2026-09-14 the retirement results did not even carry the data.
    _rebalancing_summary(res, specs, labels)
    _diagnostics(res, specs, labels, config, spec)
    if spec['mode'] == 'accumulation':
        try:
            with instrument.timed('figures'):
                paths_out = accumulation_figures(res, specs, portfolios, config)
            for path in paths_out:
                print(f"  figures → {path}")
        except Exception as e:                      # never let plotting kill a run
            print(f"\n  (figures skipped: {e})")

    if spec['mode'] == 'retirement':
        # A fixed-dollar run aims at no confidence, so its success rate is a
        # result, not a miss against the solver's target.
        if spec.get('spend') == 'solve':
            confidence_note(by_policy, specs, labels=labels)
        robustness(by_policy, specs, labels=labels)
        try:
            with instrument.timed('figures'):
                paths_out = list(figures(by_policy, specs, config,
                                         one_plan=_reads_one_plan(runs, spec, is_grid)))
            for path in paths_out:
                print(f"\n  figures → {path}")
        except Exception as e:                      # never let plotting kill a run
            print(f"\n  (figures skipped: {e})")

    # NOT "the equity premium". `engine.scale_equity_premia` rescales each
    # equity's premium OVER THE MARKET PROXY and leaves bonds alone, so at k=0
    # VOO still earns 7.49% real against BND's 2.10% — a full equity premium —
    # and only the TILTS have collapsed. Calling it the equity premium sent one
    # reader (and the author of this comment, mid-analysis) looking for a
    # stock-versus-bond result the knob cannot produce. MC_PREMIUM_HAIRCUT is
    # the one that moves the equity premium itself.
    print(f"\n  Reminder: results assume the full historical FACTOR premia —")
    print(f"  value, size, profitability, momentum — unless MC_PREMIUM_SCALE says")
    print(f"  otherwise. At k=0 every equity earns what VOO earns; the equity")
    print(f"  premium over bonds is untouched (that is MC_PREMIUM_HAIRCUT).")
    # Measured on a retirement horizon: `mu[t] = mu[VOO] + k*(mu[t] - mu[VOO])`,
    # so at k=0 every equity portfolio collapses onto VOO and the tilt has
    # nothing left to earn. Named as an EXAMPLE: this run may not include 141,
    # and a reader who compared 160 against 161 should not be left hunting for a
    # row that is not in their table.
    print(f"  Re-measured 2026-10-22, 141's lifetime spending spans 1.61x")
    print(f"  across the range (k=1.0 to k=0.0) — larger than any choice")
    print(f"  shown above, and the default is the most optimistic.")
    print(f"  Mostly LEVEL: at k=0 the plain-index 145 catches 141 (+0.2..+3.2%).")
    print(f"  `tools/sensitivity.py`.")
    print("=" * 84)
    return res
