#!/usr/bin/env python3
"""
Sensitivity sweeps — does an assumption change the LEVEL or the DECISION?
========================================================================

    python3 tools/sensitivity.py retirement_age 60,65,70
    python3 tools/sensitivity.py contributions  0.5,1.0,1.5     (multiplier)
    python3 tools/sensitivity.py horizon        50,58,63        (years)
    python3 tools/sensitivity.py inflation      0.02,0.03,0.04
    python3 tools/sensitivity.py ss_scale       1.0,0.8,0.0   (benefit haircut)
    python3 tools/sensitivity.py premium        1.0,0.5,0.0
    python3 tools/sensitivity.py param_std      1.0,2.0,3.0   (epistemic band)

Assumptions split into two kinds and they need different attention:

  LEVEL    moves every strategy together — changes how much you can spend, not
           what you should do. Social Security is the extreme case: it lifts
           everyone by the benefit amount and reorders nothing.
  RANKING  moves strategies differently — changes the decision. Re-measured
           2026-10-08, only the FACTOR premium does, and only at k=0: the
           plain-index 145 catches the tilted 141 (a paired tie) and passes
           the bonded 141. Every other sweep is level-only.

The report says which, so effort goes where it changes an answer. A sweep that
only moves the level is worth knowing about and not worth agonising over.

`inflation` and `premium` are read from the environment by engine.py at import,
so those two re-exec a subprocess per value; the rest are plain config fields
passed to workers through the args tuple.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_STRATEGIES = ['141', '141>141+10@65', '145']
# Two strategies closer than this are reported as tied. The solver locates a
# threshold on a flat curve, so a few percent of spend is resolution, not signal.
TIE_REL = 0.05
# argparse reads a leading '-' as a flag, so `infl_beta_equity -1.15,-0.80,0.0`
# died with "unrecognized arguments". Negative values are legitimate — every
# inflation pass-through beta is negative — and `--` is not the fix: it makes
# argparse treat --strategies and --sims as positionals too. A sentinel hides
# the sign from the parser and is stripped immediately after.
_NEG = '\x00neg'
# ---------------------------------------------------------------------------
# THE DIMENSION REGISTRY
# ---------------------------------------------------------------------------
# One entry per assumption. Adding a knob is one line, which is the point: every
# env var engine.py reads is sweepable by construction, and before this they
# were not — the real cash rate was the biggest lever in the model and had no
# dimension at all.
#
# `kind` picks the REPORT, not the machinery:
#   numeric  values are ordered, so a SPAN is meaningful ("1.90x across the sweep")
#   named    values are labels, so only the RANKING is meaningful. A span over
#            {value dead, size dead} would be arithmetic on nothing.
#
# `group` is the menu grouping, and it splits on WHO CONTROLS THE ASSUMPTION —
# which is this model's central finding in the shape of a menu. Everything in
# world/cash/inflation/confidence is unknowable and level-only; everything in
# `plan` is a decision you actually make, and retirement age alone moves the
# answer roughly ten times more than portfolio choice does.
#
# Every figure in a note was re-measured 2026-10-12 (plan to 100, COST on its
# six US factor loadings — ledger entry N10, unpublished) as MEDIAN LIFETIME SPENDING of 141 — what the
# sweep ranks on since 2026-10-03. Notes before that date quoted the plan level
# and are not comparable. Full table: docs/RESULTS.md.

class Dim:
    def __init__(self, group, note, kind='numeric', env=None, default=''):
        self.group, self.note, self.kind = group, note, kind
        self.env, self.default = env, default


GROUPS = [
    ('returns',    'Returns',    'what the market pays'),
    ('cash',       'Cash',       'the biggest lever, and unknowable'),
    ('inflation',  'Inflation',  'level, uncertainty, and who it hurts'),
    ('confidence', 'Confidence', 'how sure the model claims to be'),
    ('life',       'Your life',  'lifespan, care costs, benefit cuts — not yours to choose'),
    ('plan',       'Your plan',  'the things you choose — run.py compares these '
                                 'as Retirement choices'),
]

DIMENSIONS = {
    # --- returns -----------------------------------------------------------
    'premium': Dim('returns', 'mostly level, 1.61x on 141 (2026-10-22); at k=0 the plain-index 145 edges 141 (+0.2..+3.2%) and passes the bonded 141 — FACTOR premia over the market, not the equity premium', env='MC_PREMIUM_SCALE',
                   default='1.0,0.5,0.0'),
    'scenario': Dim('returns', 'kill one factor at a time, or collapse regions',
                    kind='named', env='MC_PREMIA_SCENARIO',
                    default='full history,value dead,size dead,momentum dead'),
    'regions': Dim('returns', 'level-only — 141 at 1.00 / 1.23 / 1.15 / 0.72x; '
                              'intl catching up or lagging moves every book and none overtakes',
                   kind='named',
                   env='MC_PREMIA_SCENARIO',
                   default='us stays ahead,intl catches up,regions converge,'
                           'intl keeps lagging'),
    'mu_method': Dim('returns', 'factor build-up vs each ticker\'s own mean',
                     kind='named', env='MC_MU_METHOD', default='factor,historical'),
    'single_stock_edge': Dim('returns', 'what a single stock (COST) earns over '
                             'its factor build-up — 0.07 is roughly its 20-year '
                             'lead. RANKING: 165 passes 167 (no COST) at '
                             'about 0 to -1pp/yr; 1.26x on 165',
                             env='MC_SINGLE_STOCK_EDGE',
                             default='-0.03,0.0,0.03,0.07'),
    # --- cash --------------------------------------------------------------
    'rf_real': Dim('cash', 'THE biggest lever — 4.31x on 141',
                   env='MC_RF_REAL', default='0.0,0.0057,0.0298'),
    'rf_source': Dim('cash', 'historical average vs the market\'s TIPS quote',
                     kind='named', env='MC_RF_SOURCE',
                     default='historical,tips30_avg,tips30'),
    'rf_std': Dim('cash', 'how unsure the shared cash-rate band is',
                  env='MC_RF_REAL_STD', default='0.0,0.010,0.020'),
    # --- inflation ---------------------------------------------------------
    'inflation': Dim('inflation', 'level-only, 1.09x on 141',
                     env='MC_INFLATION', default='0.02,0.03,0.04'),
    'inflation_std': Dim('inflation', 'inflation as a RISK rather than a number',
                         env='MC_INFLATION_STD', default='0.0,0.015,0.025'),
    'infl_beta_equity': Dim(
        'inflation',
        'level-only, 1.06x on 141 — the derisk answer does not hinge on it '
        'any more: bonds cost spending across the range',
        env='MC_INFL_BETA_EQUITY', default='-1.15,-0.80,0.0'),
    'infl_beta_tips': Dim('inflation', 'level-only, 1.00x on STIP sleeves — how much '
                                    'an inflation surprise costs TIPS (0 by construction)',
                          env='MC_INFL_BETA_TIPS', default='0.0,-0.30,-0.60'),
    # --- confidence --------------------------------------------------------
    'param_std': Dim('confidence', 'level-only, 1.31x on 141',
                     env='MC_PARAM_STD_SCALE', default='1.0,2.0,3.0'),
    'param_errors': Dim('confidence', 'one error per premium, shared by every '
                                      'fund — or each fund its own (pre-2026-10-02)',
                        kind='named', env='MC_PARAM_ERRORS',
                        default='shared,independent'),
    'tail_df': Dim('confidence', 'level-only, 1.06x on 141 — crisis '
                   'SHAPE at fixed variance; levels rise because the 90% path '
                   'sits in the shoulder the t thins', env='MC_TAIL_DF',
                   default='0,5,3'),
    'round_trip': Dim('confidence', 'implementation cost per unit turnover',
                      env='MC_ROUND_TRIP', default='0.0005,0.0015,0.0040'),
    # --- your plan ---------------------------------------------------------
    'rebalancing': Dim('plan', 'level-only — uniform 25%/5pp bands spend the '
                       'same as the per-fund pairs (141 identical either way) '
                       'and trade less', kind='named',
                       env='MC_REBALANCING_BANDS', default='model,uniform'),
    'retirement_age': Dim('plan', 'level-only, 1.34x — and yours to choose',
                          default='60,65,70'),
    'ltc_shock': Dim('life', 'annual cost of care for 3 years from 82 — the one '
                     'un-modelled risk that could argue FOR bonds',
                     default='0,60000,120000,240000'),
    'gk_response': Dim('plan', 'how fast guyton_klinger follows a growing '
                       'portfolio — participation only, the 10% cut cap never '
                       'moves', kind='named', default='slow,medium,fast'),
    'contributions': Dim('plan', 'multiplier on every account', default='0.5,1.0,1.5'),
    'horizon': Dim('life', 'years, from your current age', default='50,58,63'),
    'ss_scale': Dim('life', 'scheduled vs payable benefits; needs Social '
                            'Security in your details', default='1.0,0.8,0.0'),
    'ss_claim_age': Dim('plan', 'claim at 62, 67 or 70 — a real decision, '
                                'not an assumption', default='62,67,70'),
}

ENV_DIMS = {k: d.env for k, d in DIMENSIONS.items() if d.env}


def apply_dim(cfg, dim, value):
    """Mutate a config for one sweep point. Env-backed dims are handled upstream."""
    if dim == 'ltc_shock':
        from montecarlo import policies as _pol
        cfg['withdrawal']['ltc'] = {**_pol.LTC_DEFAULTS,
                                    'annual_amount': float(value)}
    elif dim == 'gk_response':
        cfg['withdrawal']['gk_response'] = str(value)
    elif dim in ('retirement_age', 'contributions', 'ss_claim_age'):
        # A plan choice, applied the one way run.py's retirement-choices route
        # applies it — two code paths for 'retire at 60' is how they drift.
        import run as runner
        if dim == 'ss_claim_age' and not cfg.get('social_security'):
            raise SystemExit(
                f"{dim} needs Social Security in your details — set it from "
                f"run.py, 'Your details'")
        cfg.update(runner.apply_choice(cfg, dim, value))
    elif dim == 'horizon':
        cfg['time_horizon_years'] = int(value)
    elif dim == 'ss_scale':
        if not cfg.get('social_security'):
            raise SystemExit(
                f"{dim} needs Social Security in your details — set it from "
                f"run.py, 'Your details'")
        cfg['social_security'] = dict(cfg['social_security'],
                                      benefit_scale=float(value))
    elif dim in ENV_DIMS:
        pass
    else:
        raise SystemExit(f"unknown dimension {dim!r}")
    return cfg


def measure(strategies, cfg, sims):
    """Median lifetime spending per strategy at this sweep point — what the
    paths REALISED at the solved plan level, not the level itself.

    It returned the plan level until 2026-10-02, against the rule this repo
    keeps relearning: the level is the noisiest column in the model. An
    800-path premium sweep read 141 / 145 at 78k / 84k at k=0 — a reversal —
    which paired at 3,000 paths on realised spending is 141 ahead by 2.6-4.9%.
    """
    from montecarlo import glide
    from montecarlo import solver
    from montecarlo.loaders import load_portfolios_from_csv

    P = load_portfolios_from_csv(verbose=False)
    out = {}
    for spec in strategies:
        plan = glide.build_plan(glide.parse_schedule(spec, cfg['age']), P)
        r = solver.solve_sustainable_spend(
            plan, cfg, confidence=0.90, floor=0.85, search_sims=sims,
            verify_sims=sims * 2)
        out[spec] = r['median_realised']
    return out


def _child(args):
    """One sweep point, run in-process (called directly or via subprocess)."""
    import run as runner

    cfg = runner.build_config(_base_spec(args.get('gk_response')))
    apply_dim(cfg, args['dim'], args['value'])
    print(json.dumps(measure(args['strategies'], cfg, args['sims'])))
    # A child exists for one point; in-process sweeps keep the pool across
    # points, and solver's atexit hook releases it.
    from montecarlo import solver
    solver.shutdown_pool()


def _base_spec(gk_response=None):
    """The run every sweep point varies ONE thing away from.

    The guardrail response rides along because it is not a detail: slow /
    medium / fast moved the same portfolio's plan level by ~15%, so a sweep run
    at the shipped `slow` while your own runs use `fast` reports levels for a
    plan you are not running.
    """
    spec = {'mode': 'retirement', 'portfolios': ['141'], 'spend': 'solve',
            'policy': 'guyton_klinger'}
    if gk_response:
        spec['gk_response'] = gk_response
    return spec


def sweep_point(dim, value, strategies, sims, gk_response=None):
    if dim in ENV_DIMS:
        # engine reads these at import, so the value must be set before the
        # module loads — hence a fresh interpreter per point.
        payload = json.dumps({'dim': dim, 'value': value,
                              'strategies': strategies, 'sims': sims,
                              'gk_response': gk_response})
        env = dict(os.environ, **{ENV_DIMS[dim]: str(value)})
        code = (f"import sys, json; sys.path.insert(0, {str(ROOT)!r}); "
                f"from tools.sensitivity import _child; _child(json.loads({payload!r}))")
        out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                             text=True, env=env, cwd=str(ROOT))
        lines = [l for l in out.stdout.splitlines() if l.startswith('{')]
        if not lines:
            raise SystemExit(f"sweep point {dim}={value} failed:\n{out.stderr[-2000:]}")
        return json.loads(lines[-1])

    import run as runner
    cfg = runner.build_config(_base_spec(gk_response))
    apply_dim(cfg, dim, value)
    return measure(strategies, cfg, sims)


def print_menu(group=None):
    """The dimension list, grouped by WHO CONTROLS THE ASSUMPTION.

    A bare usage line makes you go read -h. This is the menu run.py drives too,
    so the CLI and the interactive path offer exactly the same choices — there
    is one registry and one place to add to it.
    """
    print("\n  SENSITIVITY — does an assumption change the LEVEL or the DECISION?\n")
    print("    LEVEL    moves every strategy together. Worth knowing, not worth")
    print("             agonising over: it changes how much, not what to hold.")
    print("    RANKING  moves them differently. That is a decision.\n")
    for g, title, blurb in GROUPS:
        if group and g != group:
            continue
        print(f"  {title.upper()}  — {blurb}")
        for name, d in DIMENSIONS.items():
            if d.group != g:
                continue
            tag = '' if d.kind == 'numeric' else '  [named]'
            if len(d.default) <= 32:
                print(f"    {name:<18}{d.default:<34}{d.note}{tag}")
            else:
                print(f"    {name:<18}{d.note}{tag}")
                print(f"    {'':<18}{d.default}")
        print("")
    print("    python3 tools/sensitivity.py <dimension> <comma,separated,values>")
    print("    --sims 800 or more: below that the dollar answer is poorly")
    print("    conditioned at low premium, where the success curve is flattest.\n")
    print("  All eight re-measured 2026-09-12 on the corrected cash rate:")
    print("     24 points, NO material reversal anywhere.")
    print("  ⚠ Sweep points are CORRELATED, not independent — they all solve on")
    print("     seed_offset=0. A level gap needs an out-of-sample check before")
    print("     it is quoted; see tools/solver_calibration.py.")


def separation(v, w):
    """+1 if v beats w materially, -1 if w beats v, 0 if they are tied."""
    big = max(abs(v), abs(w), 1e-9)
    if abs(v - w) < TIE_REL * big:
        return 0
    return 1 if v > w else -1


def reversals(results, strategies, values):
    """The pairs that separate materially one way at one sweep point and
    materially the other way at another, as (a, b, {value: sign}). A pair that
    merely swaps inside the tie band is not one."""
    flips = []
    for i, a in enumerate(strategies):
        for b in strategies[i + 1:]:
            signs = {v: separation(results[v][a], results[v][b]) for v in values}
            nonzero = {s for s in signs.values() if s}
            if len(nonzero) > 1:
                flips.append((a, b, signs))
    return flips


def tied_order(results, strategies, v):
    """The ranking at one sweep point, with tied neighbours joined by '='."""
    ranked = sorted(strategies, key=lambda s: -results[v][s])
    out, i = [], 0
    while i < len(ranked):
        grp = [ranked[i]]
        while (i + 1 < len(ranked)
               and separation(results[v][ranked[i]],
                              results[v][ranked[i + 1]]) == 0):
            i += 1
            grp.append(ranked[i])
        out.append(' = '.join(grp) if len(grp) > 1 else grp[0])
        i += 1
    return ' > '.join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dimension', nargs='?', help='what to sweep')
    ap.add_argument('values', nargs='?', help='comma-separated')
    ap.add_argument('--strategies', default=','.join(DEFAULT_STRATEGIES))
    ap.add_argument('--sims', type=int, default=400)
    ap.add_argument('--gk-response', choices=('slow', 'medium', 'fast'),
                    default=None, help="guardrail response for every point "
                    "(default: policies.GK_RESPONSE_DEFAULT, 'fast')")
    # A value list can start with a minus sign: recognised by shape and hidden
    # from argparse with the _NEG sentinel (see above).
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = [_NEG + a if (a.startswith('-') and ',' in a
                         and re.fullmatch(r'-?[\d.,\-]+', a)) else a
            for a in argv]
    args = ap.parse_args(argv)
    if args.values:
        args.values = args.values.replace(_NEG, '')

    if not args.dimension or not args.values:
        print_menu()
        return 2

    if args.dimension not in DIMENSIONS:
        print(f"\n  unknown dimension {args.dimension!r}\n")
        print_menu()
        return 2
    dim = DIMENSIONS[args.dimension]
    strategies = [s.strip() for s in args.strategies.split(',') if s.strip()]
    values = [v.strip() for v in args.values.split(',') if v.strip()]

    print(f"\n  SENSITIVITY: {args.dimension}   ({len(values)} points x "
          f"{len(strategies)} strategies)")
    print("  median lifetime spending, real $/yr, at each strategy's solved plan "
          "level\n  (ranked on this, never on the plan level)\n")
    results = {}
    for v in values:
        results[v] = sweep_point(args.dimension, v, strategies, args.sims,
                                 args.gk_response)
        shown = '  '.join(f"{s}={results[v][s]/1e3:.0f}k" for s in strategies)
        print(f"    {args.dimension}={v:<18} {shown}")

    # Width from the actual labels: scenario names ('us stays ahead') are far
    # wider than the numeric values this table was built for, and a fixed 10
    # silently knocked every column out of line.
    vw = max([len('value')] + [len(str(v)) for v in values]) + 2
    cw = max([12] + [len(s) + 2 for s in strategies])
    print(f"\n  {'value':<{vw}}" + ''.join(f"{s:>{cw}}" for s in strategies))
    print("  " + "-" * (vw + cw * len(strategies)))
    for v in values:
        row = ''.join(f"{results[v][s]/1e3:>{cw-1}.0f}k" for s in strategies)
        print(f"  {v:<{vw}}{row}")

    # LEVEL vs RANKING
    #
    # Compared as point estimates this fired on noise: the inflation sweep
    # declared "RANKING-RELEVANT. This assumption changes what you should do."
    # on a $1k gap over $205k. Near the confidence crossing the success curve is
    # flat for the ratcheting policies — guyton_klinger moves ~0.09pp per $1k —
    # so a few percent of spend is not a difference, it is resolution.
    #
    # A ranking only counts as CHANGED when some pair separates materially one
    # way at one sweep point and materially the other way at another. Pairs that
    # merely swap inside the tie band are reported as tied, not as a decision.
    flips = reversals(results, strategies, values)

    spans = [max(results[v].values()) - min(results[v].values()) for v in values]
    level_move = (max(max(r.values()) for r in results.values())
                  - min(max(r.values()) for r in results.values()))

    print(f"\n  VERDICT   (differences under {TIE_REL:.0%} are treated as ties —")
    print(f"            that is resolution, not a decision)")
    if dim.kind == 'numeric':
        # A SPAN needs an ordering. Over {value dead, size dead} it would be
        # arithmetic on labels, so named dimensions report the level move
        # without calling it a span.
        lo = min(max(r.values()) for r in results.values())
        print(f"    level effect: best strategy moves ${level_move/1e3:,.0f}k/yr "
              f"across the sweep"
              + (f"  ({(lo + level_move)/max(lo, 1e-9):.2f}x)" if lo > 0 else ""))
    else:
        print(f"    level effect: best strategy varies ${level_move/1e3:,.0f}k/yr "
              f"between scenarios (no span — these are labels, not a scale)")
    for v in values:
        print(f"      {args.dimension}={v:<18} {tied_order(results, strategies, v)}")
    if not flips:
        print(f"    ranking:      no pair reverses materially")
        print(f"    -> LEVEL-ONLY. Worth knowing, does not change the decision.")
    else:
        print(f"    ranking:      REVERSES materially —")
        for a, b, signs in flips:
            where = ", ".join(
                f"{args.dimension}={v}: " +
                (f"{a} > {b}" if signs[v] > 0 else
                 f"{b} > {a}" if signs[v] < 0 else "tied")
                for v in values)
            print(f"      {a} vs {b}   {where}")
        print(f"    -> RANKING-RELEVANT. This assumption changes what you should do.")

    print(f"\n    (within-point spread between strategies: "
          f"${min(spans)/1e3:,.0f}k-${max(spans)/1e3:,.0f}k)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
