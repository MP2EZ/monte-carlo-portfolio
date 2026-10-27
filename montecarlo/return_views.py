"""
If the return model is wrong — does the portfolio decision survive?
===================================================================

A portfolio comparison is answered under ONE model of what markets pay: the
factor build-up, every premium as measured, the US ahead of international.
The decision between two portfolios can hinge on that model while the level
barely notices: 161/165/166 ranked one way under the factor model and the
other way under historical means (docs/RESULTS.md). The run's own estimation
errors cannot see this. They are drawn SYMMETRICALLY around the factor centre,
so a directional belief ("international keeps lagging") cancels out of them,
and the method itself — factor or historical — is never drawn at all.

So the comparison is repeated, at the same plan and on the same markets, under
each view below, and scored by REGRET: the lifetime spending a strategy gives
up against the best strategy in that view. The strategy with the smallest
worst case is the minimax pick, which is the answer if you cannot say which
view is true.

Each view runs in a CHILD PROCESS. engine.py reads its return model from the
environment at import, and workers re-import under `spawn`, so a view cannot
be switched inside one process. Children evaluate the same seeds, so a view's
numbers are paired with the main run's path for path.
"""

import os
import pickle
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# The three knobs that together ARE the return model. A view pins all three,
# not just the one it moves: a run made under MC_MU_METHOD=historical would
# otherwise carry it into 'intl keeps lagging', where the scenario does nothing
# because no factor build-up happens. Everything else is inherited, so a view
# varies the return model and only that.
RETURN_KNOBS = {'MC_MU_METHOD': 'factor',
                'MC_PREMIA_SCENARIO': 'full history',
                'MC_PREMIUM_SCALE': '1.0'}


class View(NamedTuple):
    key: str
    title: str          # column header, at most 10 characters
    blurb: str          # the legend line
    knobs: dict         # env overrides on top of RETURN_KNOBS


VIEWS = (
    View('factor', 'factor',
         'the default: premia as measured, the US ahead of international', {}),
    View('intl keeps lagging', 'intl lags',
         'international keeps trailing the US by its 2005-2026 gap',
         {'MC_PREMIA_SCENARIO': 'intl keeps lagging'}),
    View('intl catches up', 'intl gains',
         'developed international earns the US market premium',
         {'MC_PREMIA_SCENARIO': 'intl catches up'}),
    View('no factors', 'no factors',
         'value, size, profitability, investment and momentum pay nothing',
         {'MC_PREMIA_SCENARIO': 'no factors'}),
    View('historical', 'historical',
         "each fund's own sample mean. A stress test: the international "
         "funds' history starts in 2005", {'MC_MU_METHOD': 'historical'}),
)
VIEW_BY_KEY = {v.key: v for v in VIEWS}
# A run made under a return model the list does not hold is still a column.
THIS_RUN = 'this run'

# Below this many paths a child runs serially: a pool costs more to start than
# a handful of paths cost to simulate.
SERIAL_BELOW = 200
# Extra wall clock per child, on top of its paths: interpreter, engine import
# and pool start-up.
CHILD_OVERHEAD_SECONDS = 5.0


def view_env(key, base=None):
    """The environment a view's child runs under."""
    env = dict(os.environ if base is None else base)
    env.update(RETURN_KNOBS)
    env.update(VIEW_BY_KEY[key].knobs)
    return env


def _same_scenario(a, b):
    from montecarlo import factors
    if a == b:
        return True
    # 'us stays ahead' is the default under another name: compare what a
    # scenario DOES, not what it is called.
    return a in factors.SCENARIOS and b in factors.SCENARIOS and \
        factors.SCENARIOS[a] == factors.SCENARIOS[b]


def current_view(mu_method, scenario, scale):
    """The view this process's engine is already running, or None."""
    for v in VIEWS:
        k = dict(RETURN_KNOBS, **v.knobs)
        if (mu_method == k['MC_MU_METHOD']
                and abs(float(scale) - float(k['MC_PREMIUM_SCALE'])) < 1e-12
                and _same_scenario(scenario, k['MC_PREMIA_SCENARIO'])):
            return v.key
    return None


# ---------------------------------------------------------------------------
# Regret
# ---------------------------------------------------------------------------

def regrets(books, specs):
    """Score every strategy in every view against that view's best.

    `books` is {view: {strategy: {'per_path_mean', 'depleted', 'spend'}}},
    every book in a view on the same paths. The metric is median lifetime
    spending, given up as a share of the leader's. Under FIXED spending every
    surviving path spends exactly the plan, so that median is one number for
    everyone and a regret on it could never fail; the metric becomes the
    extra chance of running out, in absolute terms.

    `holds` is whether the paired interval leaves the strategy unresolved
    against the leader: a leader is a point estimate, and a strategy it
    cannot be separated from has not lost that view. `robust` is the
    strategies that have lost no view.
    """
    from montecarlo import engine
    from montecarlo.report_retirement import _spending_is_fixed

    fixed = all(_spending_is_fixed(vb[g]) for vb in books.values() for g in specs)
    views = {}
    for view, vb in books.items():
        if fixed:
            val = {g: float(np.mean(vb[g]['depleted'])) for g in specs}
            leader = min(specs, key=val.get)
            regret = {g: val[g] - val[leader] for g in specs}
        else:
            val = {g: float(np.median(vb[g]['per_path_mean'])) for g in specs}
            leader = max(specs, key=val.get)
            top = max(val[leader], 1e-9)
            regret = {g: (val[leader] - val[g]) / top for g in specs}
        holds = {}
        for g in specs:
            if g == leader:
                holds[g] = True
            elif fixed:
                lo, _ = engine.paired_bootstrap_diff(
                    np.asarray(vb[g]['depleted'], float),
                    np.asarray(vb[leader]['depleted'], float), np.mean)
                holds[g] = bool(lo <= 0)
            else:
                _, hi = engine.paired_bootstrap_diff(
                    vb[g]['per_path_mean'], vb[leader]['per_path_mean'], np.median)
                holds[g] = bool(hi >= 0)
        views[view] = {'value': val, 'leader': leader, 'regret': regret,
                       'holds': holds}

    order = list(books)
    worst = {g: max(views[v]['regret'][g] for v in order) for g in specs}
    worst_view = {g: max(order, key=lambda v: views[v]['regret'][g]) for g in specs}
    pick = min(specs, key=lambda g: (worst[g], specs.index(g)))
    robust = [g for g in specs if all(views[v]['holds'][g] for v in order)]
    return {'metric': 'ruin' if fixed else 'spend', 'views': views,
            'order': order, 'worst': worst, 'worst_view': worst_view,
            'pick': pick, 'robust': robust}


# ---------------------------------------------------------------------------
# Running a view
# ---------------------------------------------------------------------------

def run_view(key, cfg, plans, level, n):
    """Evaluate every plan at `level` under view `key`, in a child process.

    Returns {'engine': the return model the child actually ran,
             'books': {strategy: per_path_mean, depleted, success_rate, spend}}.
    Raises if the child failed or ran under a different model than asked —
    a view that silently fell back to the default would print a column
    identical to the factor one, and read as 'the decision survives'.
    """
    with tempfile.TemporaryDirectory() as tmp:
        inp, outp = Path(tmp) / 'in.pkl', Path(tmp) / 'out.pkl'
        with open(inp, 'wb') as f:
            pickle.dump({'cfg': cfg, 'plans': plans, 'level': float(level),
                         'n': int(n)}, f)
        code = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
                f"from montecarlo import return_views; return_views._child(sys.argv[1], sys.argv[2])")
        done = subprocess.run([sys.executable, '-c', code, str(inp), str(outp)],
                              capture_output=True, text=True, cwd=str(ROOT),
                              env=view_env(key))
        if done.returncode != 0 or not outp.exists():
            raise RuntimeError(f"child exited {done.returncode}: "
                               f"{done.stderr.strip()[-600:]}")
        with open(outp, 'rb') as f:
            got = pickle.load(f)

    want = dict(RETURN_KNOBS, **VIEW_BY_KEY[key].knobs)
    ran = got['engine']
    if not (ran['MU_METHOD'] == want['MC_MU_METHOD']
            and abs(ran['PREMIUM_SCALE'] - float(want['MC_PREMIUM_SCALE'])) < 1e-12
            and _same_scenario(ran['PREMIA_SCENARIO'], want['MC_PREMIA_SCENARIO'])):
        raise RuntimeError(f"asked for {key!r}, the child ran {ran}")
    return got


def _child(inp, outp):
    """One view, in a fresh interpreter whose environment already holds it."""
    from montecarlo import engine
    from montecarlo import solver

    with open(inp, 'rb') as f:
        p = pickle.load(f)
    cfg, n = p['cfg'], p['n']
    wd = cfg.get('withdrawal') or {}
    books = {}
    for g, plan in p['plans'].items():
        r = solver.evaluate(plan, cfg, p['level'], n_sims=n,
                            floor=wd.get('floor', 0.85),
                            parallel=n >= SERIAL_BELOW, full=True,
                            adequacy=cfg.get('adequacy', solver.DEFAULT_ADEQUACY))
        books[g] = {'per_path_mean': np.asarray(r['per_path_mean'], float),
                    'depleted': np.asarray(r['raw']['depleted'], bool),
                    'success_rate': float(r['success_rate']),
                    'spend': float(r['spend'])}
    solver.shutdown_pool()
    with open(outp, 'wb') as f:
        pickle.dump({'engine': {'MU_METHOD': engine.MU_METHOD,
                                'PREMIA_SCENARIO': engine.PREMIA_SCENARIO,
                                'PREMIUM_SCALE': float(engine.PREMIUM_SCALE)},
                     'books': books}, f)


def extra_paths(n_strategies, n):
    """Paths the check adds to a run: every strategy, once per other view."""
    return (len(VIEWS) - 1) * n_strategies * n


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

_RULE = '  ' + '─' * 80


def _title(key):
    return VIEW_BY_KEY[key].title if key in VIEW_BY_KEY else THIS_RUN


def _cell(t, view, g):
    v = t['views'][view]
    if g == v['leader']:
        return 'best'
    r = v['regret'][g]
    txt = f"{r * 100:.1f}pp" if t['metric'] == 'ruin' else f"{r * 100:.1f}%"
    return ('~' if v['holds'][g] else '') + txt


def _amount(t, r):
    return f"{r * 100:.1f}pp" if t['metric'] == 'ruin' else f"{r * 100:.1f}%"


def _names(gs):
    return gs[0] if len(gs) == 1 else ', '.join(gs[:-1]) + ' and ' + gs[-1]


def _wrap(text, first='    ', rest='    ', width=82):
    for line in textwrap.wrap(text, width=width, initial_indent=first,
                              subsequent_indent=rest):
        print(line)


def render(t, specs, plan, level, plan_name, n, labels=None, certified=None):
    """Print the regret table, the verdict and the plan under each view."""
    labels = labels or {}
    name = lambda g: labels.get(g, g)
    order = t['order']
    ruin = t['metric'] == 'ruin'

    print("\n" + _RULE)
    print("  IF THE RETURN MODEL IS WRONG   does the winner change?")
    print(_RULE)
    _wrap(f"Every strategy at {plan_name}'s ${level / 1e3:.0f}k/yr plan, on the same "
          f"{n:,} paths, under {len(order)} views of what markets pay. A cell is what "
          f"that strategy gives up against the best one in that view. '~' marks a "
          f"gap the paired paths cannot separate from zero, so that view is not lost.")
    print()
    print("    " + ("extra chance of running out vs the best" if ruin
                    else "lifetime spending given up vs the best"))
    lw = max([10] + [len(name(g)) + 2 for g in specs])
    cw = 11
    print(f"    {'strategy':<{lw}}" + ''.join(f"{_title(v):>{cw}}" for v in order)
          + f"{'worst':>{cw}}")
    print("    " + "-" * (lw + cw * (len(order) + 1)))
    for g in specs:
        row = ''.join(f"{_cell(t, v, g):>{cw}}" for v in order)
        mark = '  ←' if g == t['pick'] else ''
        print(f"    {name(g):<{lw}}{row}{_amount(t, t['worst'][g]):>{cw}}{mark}")
    print()
    for v in order:
        blurb = VIEW_BY_KEY[v].blurb if v in VIEW_BY_KEY else \
            "the return model this run was made under (see the banner above)"
        _wrap(blurb, first=f"    {_title(v):<12}", rest=' ' * 16)
    print()

    robust = t['robust']
    if len(robust) == len(specs):
        # Everyone holds up because nothing separates — that is a path count
        # too small to see the gaps, not a choice that survives every model.
        _wrap(f"~ No gap is resolved in any view at {n:,} paths. The strategies are "
              f"indistinguishable here, which is not the same as equal; more "
              f"paths may separate them.")
    elif robust:
        who = _names([name(g) for g in robust])
        if len(robust) == 1:
            print(f"    ✓ {who} is never resolved behind the best, so it holds up in "
                  f"every view.")
        else:
            print(f"    ✓ {who} are never resolved behind the best, so they hold up "
                  f"in every view.")
        print("      The choice does not hinge on which return model is right.")
    else:
        print("    ⚠ No strategy holds up in every view: which one wins depends on")
        print("      which return model is right.")
    p = t['pick']
    _wrap(f"← Smallest worst case: {name(p)}, at most {_amount(t, t['worst'][p])} "
          f"behind ({_title(t['worst_view'][p])}). That is the minimax-regret pick, "
          f"the answer if you cannot say which view is true"
          + (", though here it rests on gaps that are not resolved."
             if len(robust) == len(specs) else "."), rest='      ')

    print()
    head = f"    YOUR PLAN under each view: {plan_name} at ${level / 1e3:.0f}k/yr"
    if certified is not None:
        head += f", certified at {certified:.0%} in this run"
    print(head)
    print(f"    {'view':<12}{'success':>9}{'median spend':>15}{'low case (p10)':>17}")
    for v in order:
        r = plan[v]
        print(f"    {_title(v):<12}{r['success_rate']:>9.0%}"
              f"{r['median'] / 1e3:>14.0f}k{r['p10'] / 1e3:>16.0f}k")
    print("    Where success falls well short, that view would ask this plan for")
    print("    more cuts than it promised. That is a question about the PLAN, not")
    print("    the portfolio; 'Market & life assumptions' sizes it.")


def check(specs, res, cfg, portfolios, labels=None):
    """Repeat the comparison under every other view and print it.

    Returns the summary the run record keeps, or None when nothing ran.
    """
    from montecarlo import engine
    from montecarlo import glide
    from montecarlo import instrument

    labels = labels or {}
    if all('common' in res[g] for g in specs):
        level = res[specs[0]]['common']['spend']
        here = {g: res[g]['common'] for g in specs}
    else:
        level = cfg['withdrawal']['target_real_annual']
        here = {g: {'per_path_mean': res[g]['per_path_mean'],
                    'depleted': np.asarray(res[g]['raw']['depleted'], bool),
                    'success_rate': res[g]['success_rate'],
                    'spend': res[g]['spend']} for g in specs}
    n = len(here[specs[0]]['per_path_mean'])
    plans = {g: glide.build_plan(glide.parse_schedule(g, cfg['age']), portfolios)
             for g in specs}

    mine = current_view(engine.MU_METHOD, engine.PREMIA_SCENARIO, engine.PREMIUM_SCALE)
    todo = [v for v in VIEWS if v.key != mine]
    print(f"\n  CHECKING   every strategy under {len(todo)} other return models, "
          f"same plan, same {n:,} paths")
    books = {} if mine else {THIS_RUN: here}
    done = 0
    for v in VIEWS:
        if v.key == mine:
            books[v.key] = here
            continue
        try:
            with instrument.timed(f'view {v.key}'):
                books[v.key] = run_view(v.key, cfg, plans, level, n)['books']
            done += 1
            print(f"    {v.title:<12} done   ({done}/{len(todo)})")
        except Exception as e:            # one failed view must not cost the run
            print(f"    ⚠ {v.title}: failed, left out of the table — {e}")
    if len(books) < 2:
        print("    ⚠ no other view ran, so there is nothing to compare.")
        return None

    t = regrets(books, specs)
    plan = {v: {'success_rate': float(books[v][specs[0]]['success_rate']),
                'median': float(np.median(books[v][specs[0]]['per_path_mean'])),
                'p10': float(np.percentile(books[v][specs[0]]['per_path_mean'], 10))}
            for v in books}
    certified = (here[specs[0]]['success_rate']
                 if 'common' in res[specs[0]] else None)
    render(t, specs, plan, level, labels.get(specs[0], specs[0]), n, labels,
           certified=certified)
    return {'level': float(level), 'paths': n, 'metric': t['metric'],
            'pick': t['pick'], 'robust': t['robust'], 'worst': t['worst'],
            'views': {v: {'leader': t['views'][v]['leader'],
                          'regret': t['views'][v]['regret'],
                          'holds': t['views'][v]['holds']} for v in t['order']},
            'plan': plan}
