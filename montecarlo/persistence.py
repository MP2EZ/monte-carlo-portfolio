"""
Run persistence
===============

Every run writes a JSON record to `output/runs/`. Without this a run's numbers
exist only in terminal scrollback: you cannot tell whether a conclusion changed
because you altered a strategy or because a quarterly data regen moved the
parameters underneath you.

That distinction matters here more than usual — `data_collection_regime_model.py`
refreshes every expected return to current market data, so the same run can give
materially different answers a quarter apart for reasons having nothing to do
with any decision you made.

Records store SCALARS ONLY. Path arrays are ~48MB per portfolio and are not worth
persisting; percentiles answer every question a stored run is asked.

The provenance block is the point: parameter file date, inflation assumption and
premium scale are captured alongside the results, so a diff can attribute a
change to its cause.
"""

import hashlib
import json

from montecarlo import policies
from datetime import datetime
from pathlib import Path

import numpy as np

from montecarlo import engine

RUNS_DIR = Path(__file__).resolve().parent.parent / 'output' / 'runs'

def headline_metric(spec):
    """The field that is an OUTPUT for this run's mode.

    `spend` is an OUTPUT when solving and an INPUT when a fixed amount was given.
    Displaying it without checking has now produced three separate bugs — a matrix
    of the user's own input, a policy column headed "sustainable" that was the
    target, and a diff reporting +0.0%. Centralised so there is one place to be
    right about it.
    """
    if spec.get('mode') == 'accumulation':
        return 'median'
    return 'spend' if spec.get('spend') == 'solve' else 'success_rate'


def metric_label(key):
    return {'median': 'median wealth', 'spend': 'plan level',
            'success_rate': 'success rate'}.get(key, key)


# Fields worth keeping. Everything else in a solver result is an array.
SCALARS = ('spend', 'success_rate', 'depletion_rate', 'median_realised',
           'p10_realised', 'worst_cut', 'worst_cut_bad', 'median_bequest',
           # Shown in IN DETAIL since the volatility column was added and never
           # stored: 502 of 502 saved results carried no annual_vol, so `--diff`
           # could not compare the one column that describes RISK, and no saved
           # run could be re-read for it afterwards.
           'annual_vol', 'cagr_median',
           # What the PORTFOLIO funded in the binding band, on the paths that
           # set the level. Near zero means the level is guaranteed income's.
           # Kept because a saved run that omits it cannot tell a portfolio
           # result from a Social Security one after the fact, and `--diff`
           # would report two levels as comparable when one of them is a
           # benefit divided by the adequacy floor.
           'portfolio_share',
           # Whether the Newton pass actually landed. Without these a stored
           # level carries no record of how well it converged.
           'converged', 'gap', 'newton_steps',
           # Whether the level began on a bracket edge. A saved run that omits
           # it cannot tell a measured crossing from one the window chose.
           'clipped', 'level_se')


# Bumped when a change makes new records incomparable to old ones. A diff across
# such a boundary would otherwise blame a scoring change on whatever moved in the
# spec, silently.
#   1  original: lifetime-mean adequacy, bisected solve
#   2  per-band adequacy; crossing estimated on the verification sample; vpw
#      solved in closed form and seeded from its own opening draw
#   3  expected returns from the factor build-up (mu_method 'factor')
#   4  window adequacy: the worst rolling 10-year average, not the worst third
#      (2026-10-21, ledger entry N11, unpublished). Records before it say 'per_band' = 'thirds'
SCORING_VERSION = 4


def provenance(config=None, spec=None):
    """What the results depended on, beyond the run spec itself.

    `adequacy` and `scoring_version` belong here for the same reason inflation
    does: they change every number without appearing anywhere in the spec. Runs
    saved before per-band adequacy carry neither, which is exactly how a reader
    (or `--diff`) can tell they are not comparable.
    """
    meta = getattr(engine, '_params', {}).get('metadata', {})
    out = {
        # How expected returns were built. Flipped to 'factor' on 2026-09-10;
        # every run before that used 'historical' and is not comparable, which
        # is what SCORING_VERSION 3 marks.
        'mu_method': engine.MU_METHOD,
        'premia_scenario': engine.PREMIA_SCENARIO,
        'premium_haircut': engine.PREMIUM_HAIRCUT,
        # Whether estimation errors were drawn through the shared premia.
        # Absent from records before 2026-10-02, which were 'independent'.
        'param_errors': engine.PARAM_ERRORS,
        # The edge a single stock earns over its factor build-up. Absent from
        # records before 2026-10-10, when COST kept its historical mean
        # instead — those runs are not comparable on any COST portfolio.
        'single_stock_edge': engine.SINGLE_STOCK_EDGE,
        'parameters_version': meta.get('version'),
        'parameters_last_updated': meta.get('last_updated'),
        'ticker_count': len(engine.TICKERS),
        'inflation': engine.INFLATION,
        # The real cash rate and WHERE IT CAME FROM. It is the biggest level
        # lever in the model (7x on final wealth), and a market quote carries an
        # as-of date that the rest of the parameters do not — so a run compared
        # against another must be able to say "the cash rate moved", not just
        # report a different number.
        'rf_real': engine.RF_REAL,
        'rf_source': engine.RF_REAL_SOURCE,
        'rf_as_of': engine.RF_REAL_AS_OF,
        'premium_scale': engine.PREMIUM_SCALE,
        'crisis_vol_scale': engine.CRISIS_VOL_SCALE,
        'tail_df': engine.TAIL_DF,
        # Which rebalancing bands were simulated: the model's per-fund pairs or
        # the uniform 25%/5pp ('ips' before 2026-10-26). Absent before
        # 2026-10-14 = 'model'.
        'rebalancing_bands': engine.REBALANCING_POLICY,
        # Years behind the correlation matrices. Absent before 2026-10-25 = 15.
        'corr_window_years': engine.CORR_WINDOW,
        'scoring_version': SCORING_VERSION,
    }
    if config is not None:
        from montecarlo import solver
        out['adequacy'] = solver.adequacy_mode(
            config.get('adequacy', solver.DEFAULT_ADEQUACY))
        out['floor'] = config.get('withdrawal', {}).get('floor')
        # Two runs that differ ONLY in the guardrail response produce different
        # levels from identical parameters; without this `--diff` would report
        # "provenance identical" and attribute the move to nothing.
        out['gk_response'] = (config.get('withdrawal', {}).get('gk_response')
                              or policies.GK_RESPONSE_DEFAULT)
        out.update(_investor_provenance(config))
    if spec is not None:
        out.update(_path_counts(spec))
    return out


def _path_counts(spec):
    """HOW MANY PATHS THE RUN ACTUALLY BOUGHT.

    Nothing recorded this, and it is the single most likely cause of a numerical
    difference between two retirement runs: a plan level is a poorly-estimated
    percentile (roadmap item 7), and this repo's own record has a 600-path
    `regions` sweep reporting a ranking reversal that vanished at 1,600. Without
    it `--diff` says "provenance identical — any difference below is the
    strategies or Monte Carlo noise" and cannot tell a re-draw from a
    resolution change.

    Retirement never sets `sims` at all, so all twelve real records on disk
    carried `sims: None` while the run spent 400 search + 1,500 verification
    paths per strategy.
    """
    from montecarlo import solver
    if spec.get('spend') == 'solve':
        return {'search_sims': spec.get('search_sims', solver.SEARCH_SIMS),
                'verify_sims': spec.get('verify_sims', solver.VERIFY_SIMS)}
    return {'sims': spec.get('sims', solver.FIXED_SIMS)}


def _investor_provenance(config):
    """The investor facts, as FLAT scalars.

    None of this was recorded, so editing your balance or contributions between
    runs left `--diff` reporting "provenance identical — any difference below is
    the strategies or Monte Carlo noise", which was then false. It is exactly the
    misattribution `scoring_version` already guards against.

    Flat because `diff_runs` compares values field by field: a nested dict
    differs as one unreadable blob. The fingerprint covers what the scalars
    cannot — a stop_age moved, or money shifted between accounts, either of
    which can leave the year-one total untouched.
    """
    age = config.get('age')
    sched = config.get('contribution_schedule')
    out = {
        'investor_age': age,
        'investor_retirement_age': config.get('retirement_age'),
        'investor_initial_value': config.get('initial_value'),
    }
    horizon = config.get('time_horizon_years')
    if age is not None and horizon is not None:
        out['investor_plan_to_age'] = age + horizon
    if sched:
        out['investor_contributions_measured_at_age'] = sched.get('measured_at_age')
        if age is not None:
            out['investor_first_year_contribution'] = round(
                engine.calculate_annual_contribution(int(age), sched,
                                                     engine.INFLATION), 2)
        payload = json.dumps(sched.get('accounts', {}), sort_keys=True)
        out['investor_accounts'] = hashlib.sha256(
            payload.encode()).hexdigest()[:8]
    # The benefit the run ACTUALLY used — stored details or a per-run override.
    # Flat, like the rest: moving the claim age from 67 to 70 changes every
    # plan level, and --diff has to be able to say so.
    ss = config.get('social_security') or {}
    out['ss_annual_at_fra'] = float(ss.get('annual_at_fra', 0.0)) if ss else 0.0
    out['ss_claim_age'] = ss.get('claim_age') if ss else None
    out['ss_benefit_scale'] = ss.get('benefit_scale', 1.0) if ss else None
    return out


def _summarise(result):
    out = {k: result[k] for k in SCALARS if k in result}
    if 'success_by_floor' in result:
        out['success_by_floor'] = {str(k): v for k, v in result['success_by_floor'].items()}
    if 'median' in result:                       # accumulation runs
        out['median'] = result['median']
    if 'sortino_median' in result:
        out['sortino_median'] = result['sortino_median']
    if out.get('cagr_median') is None and result.get('twr_cagr') is not None:
        # `twr_cagr` is a percentile ARRAY, which a record should not carry, but
        # dropping it entirely lost the CAGR column from every saved run.
        vals = np.asarray(result['twr_cagr'], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out['cagr_median'] = float(np.median(vals))
    sortino = result.get('sortino')
    if sortino is not None and len(sortino):
        out['sortino_pctiles'] = {p: float(np.percentile(sortino, int(p)))
                                  for p in ('10', '50', '90')}
    finals = result.get('finals')
    if finals is not None and len(finals):
        out['wealth_pctiles'] = {p: float(np.percentile(finals, int(p)))
                                 for p in ('10', '25', '50', '75', '90')}
    per_path = result.get('per_path_mean')
    if per_path is not None and len(per_path):
        out['spend_pctiles'] = {p: float(np.percentile(per_path, int(p)))
                                for p in ('10', '25', '50', '75', '90')}
    return out


def save_run(spec, by_policy, specs, config, outdir=None, return_views=None):
    outdir = Path(outdir or RUNS_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')

    record = {
        'timestamp': stamp,
        'spec': spec,
        'provenance': provenance(config, spec),
        'config': {k: v for k, v in config.items() if k != 'contribution_schedule'},
        'strategies': list(specs),
        'results': {policy: {g: _summarise(res[g]) for g in specs}
                    for policy, res in by_policy.items()},
    }
    if return_views:
        record['return_views'] = return_views
    path = outdir / f'run_{stamp}.json'
    path.write_text(json.dumps(record, indent=2, default=str) + '\n')
    return path


def list_runs(outdir=None, limit=20):
    outdir = Path(outdir or RUNS_DIR)
    if not outdir.exists():
        return []
    return sorted(outdir.glob('run_*.json'), reverse=True)[:limit]


def load_run(path):
    return json.loads(Path(path).read_text())


def _flatten(record):
    """(policy, strategy) -> summary, for diffing."""
    return {(p, g): v for p, byg in record['results'].items() for g, v in byg.items()}


def diff_runs(older, newer, key=None):
    """Compare two run records on one metric. Returns rows and provenance deltas."""
    key = key or headline_metric(newer.get('spec', {}))
    a, b = _flatten(older), _flatten(newer)
    rows = []
    for k in sorted(set(a) & set(b)):
        va, vb = a[k].get(key), b[k].get(key)
        if va is None or vb is None:
            continue
        rows.append((k, va, vb, (vb - va) / va if va else float('nan')))

    prov = []
    for field in sorted(set(older['provenance']) | set(newer['provenance'])):
        pa = older['provenance'].get(field)
        pb = newer['provenance'].get(field)
        if field == 'adequacy':
            # 'per_band' is the thirds rule's name before 2026-10-21
            from montecarlo import solver
            pa, pb = (solver.ADEQUACY_ALIASES.get(v, v) for v in (pa, pb))
        if pa != pb:
            prov.append((field, pa, pb))
    return rows, prov


def print_history(outdir=None, limit=12):
    paths = list_runs(outdir, limit)
    if not paths:
        print("  No saved runs yet.")
        return paths
    print(f"\n  {'when':<18} {'mode':<13} {'strategies':<11} {'params':<12} "
          f"{'k':>5}  headline")
    print("  " + "-" * 82)
    for p in paths:
        r = load_run(p)
        pr = r['provenance']
        first_policy = next(iter(r['results']), None)
        head = ''
        if first_policy:
            hk = headline_metric(r.get('spec', {}))
            vals = [v.get(hk) for v in r['results'][first_policy].values()]
            vals = [v for v in vals if v is not None]
            if vals:
                head = {'spend': f"${max(vals) / 1e3:,.0f}k",
                        'median': f"${max(vals):,.1f}M",
                        'success_rate': f"{max(vals):.1%} ok"}[hk]
        print(f"  {r['timestamp']:<18} {r['spec'].get('mode', '?'):<13} "
              f"{len(r['strategies']):<11} {str(pr.get('parameters_last_updated')):<12} "
              f"{pr.get('premium_scale', 1.0):>5g}  {head}")
    return paths


def print_diff(older_path, newer_path, key=None):
    older, newer = load_run(older_path), load_run(newer_path)
    key = key or headline_metric(newer.get('spec', {}))
    rows, prov = diff_runs(older, newer, key)

    print(f"\n  {older['timestamp']}  →  {newer['timestamp']}")

    # A scoring change is not one difference among several: it makes the rows
    # below incomparable, because the two runs were graded against different
    # criteria. Say so before the numbers, not in a list after them.
    sa = older['provenance'].get('scoring_version', 1)
    sb = newer['provenance'].get('scoring_version', 1)
    if sa != sb:
        print(f"\n  ⚠ NOT COMPARABLE — scoring changed (v{sa} → v{sb}).")
        print(f"    The two runs were graded against different criteria, so the")
        print(f"    change below is mostly that, not your strategies. Re-run the")
        print(f"    older configuration to compare like with like.")
        oa = older['provenance'].get('adequacy', 'lifetime (pre-v2)')
        ob = newer['provenance'].get('adequacy', 'lifetime (pre-v2)')
        if oa != ob:
            print(f"    adequacy: {oa} → {ob}")

    if prov:
        print("\n  what changed underneath the run:")
        for field, a, b in prov:
            print(f"    {field}: {a} → {b}")
        # A PATH-COUNT CHANGE IS NOT LIKE THE OTHERS. Every other field moves
        # the answer; this one moves how well the answer is MEASURED, and a
        # plan level is a poorly-estimated percentile. Listing it as one line
        # among several invites reading a resolution change as a real move.
        paths = {f for f, _, _ in prov} & {'search_sims', 'verify_sims', 'sims'}
        if paths:
            print(f"\n  ⚠ the PATH COUNT changed ({', '.join(sorted(paths))}).")
            print("    That does not move the answer, it moves how precisely the")
            print("    answer is measured — and a plan level is a poorly-estimated")
            print("    percentile. Treat a difference below as unresolved until it")
            print("    survives at the HIGHER count.")
    else:
        print("\n  provenance identical — any difference below is the strategies "
              "or Monte Carlo noise,\n  not the parameters moving underneath you.")

    if not rows:
        print("\n  no comparable (policy, strategy) pairs.")
        return
    print(f"\n  comparing {metric_label(key)}")
    print(f"\n  {'policy / strategy':<40} {'before':>10} {'after':>10} {'change':>9}")
    print("  " + "-" * 73)
    for (policy, g), va, vb, rel in rows:
        label = f"{policy} / {g}"
        if key == 'spend':
            scale, unit, prec = 1e3, 'k', 0
        elif key == 'success_rate':
            scale, unit, prec = 0.01, '%', 1
        else:
            scale, unit, prec = 1.0, 'M', 2
        print(f"  {label:<40} {va / scale:9.{prec}f}{unit} {vb / scale:9.{prec}f}{unit} "
              f"{rel:+8.1%}")
