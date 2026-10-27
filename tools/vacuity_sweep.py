#!/usr/bin/env python3
"""Prove the test suite can actually fail. Two passes, static then empirical.

A test that cannot fail is worse than no test: it reports safety that is not
there. `verify-test` is the skill for one test at a time; this is the sweep over
all of them, and it is how you check the suite rather than the change.

    python3 tools/vacuity_sweep.py --static      # seconds
    python3 tools/vacuity_sweep.py --mutate      # minutes: one full run per target
    python3 tools/vacuity_sweep.py --subtle      # minutes: WRONG answers, not absent ones

NOT part of tools/validate_all.py. --mutate runs the whole suite once per
target, so it is a deliberate audit rather than a gate. Run it after adding a
group of tests, or when a refactor moves a lot of assertions.

⚠ --mutate REWRITES SOURCE FILES IN PLACE while it runs, restoring each in a
`finally`. Do not run anything else against this checkout until it finishes:
running check_logic.py concurrently during this sweep reported 8 spurious
failures, because report_retirement.py was gutted at that moment. If the sweep
is killed hard, `git status` will show the gutted file — `git checkout` it.
"""

import argparse
import ast
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SUITE = ROOT / 'tools' / 'check_logic.py'

# One entry per function worth proving the suite notices. Not exhaustive — these
# are the ones whose silent failure would corrupt a REPORTED number.
TARGETS = [
    ('montecarlo/solver.py', 'adequacy_statistic'), ('montecarlo/solver.py', 'usable_slope'),
    ('montecarlo/solver.py', 'stage_stats'), ('montecarlo/solver.py', '_marginal_portfolio_share'),
    ('montecarlo/policies.py', 'guyton_klinger'), ('montecarlo/policies.py', 'vpw'),
    ('montecarlo/policies.py', 'ltc_cost'), ('montecarlo/policies.py', 'gk_guardrails'),
    ('montecarlo/glide.py', 'parse_schedule'), ('montecarlo/glide.py', 'build_plan'),
    ('montecarlo/glide.py', 'validate'),
    ('montecarlo/engine.py', 'select_next_regime'), ('montecarlo/engine.py', 'near_psd_higham'),
    ('montecarlo/engine.py', 'traded_dollars'),
    ('montecarlo/engine.py', 'calculate_annual_contribution'),
    ('montecarlo/loaders.py', 'compute_derived'), ('montecarlo/loaders.py', 'equity_sleeve'),
    ('montecarlo/persistence.py', 'headline_metric'),
    ('montecarlo/persistence.py', '_investor_provenance'),
    ('montecarlo/diagnostics.py', 'binding_band'), ('montecarlo/diagnostics.py', 'failure_timing'),
    ('montecarlo/diagnostics.py', 'uncertainty_split'),
    ('montecarlo/diagnostics.py', 'sequence_risk'),
    ('montecarlo/diagnostics.py', 'regime_occupancy'),
    ('montecarlo/diagnostics.py', 'moved_assumptions'),
    ('montecarlo/diagnostics.py', 'market_stationary'),
    ('montecarlo/solver.py', '_print_curve'), ('montecarlo/persistence.py', '_path_counts'),
    ('run.py', 'parse_spec'),
    ('run.py', 'effective_ss'), ('run.py', '_edit_social_security'),
    ('montecarlo/investor.py', 'validate_social_security'),
    ('tools/tax_estimate.py', 'tax_a_path'),
    ('tools/tax_estimate.py', 'rmd_start_age'),
    ('tools/sensitivity.py', '_base_spec'),
    ('montecarlo/report_retirement.py', 'worst_year_pct'),
    ('montecarlo/report_retirement.py', '_best_caveat'),
    ('data_collection_regime_model.py', '_transition_matrix'),
    ('data_collection_regime_model.py', 'conditional_transitions'),
    ('data_collection_regime_model.py', 'fix_correlation_matrix'),
    ('data_collection_regime_model.py', 'score_proxy'),
]

# ---- SUBTLE mutations -------------------------------------------------------
# Gutting a function asks whether its ABSENCE is noticed, which is the weaker
# question: a function returning None usually explodes somewhere downstream.
# The real one is whether a WRONG ANSWER is noticed — a percentile read from
# the wrong tail, a flipped comparison, a band ten times too wide. EVERY defect
# this repo has actually shipped was of that kind, not a missing function: a
# price index for a total-return one, a proxy table rotated by one position, a
# median printed where a bad case was meant.
#
# (file, find, replace, what it breaks). `find` must occur EXACTLY ONCE or the
# case is reported STALE rather than applied blindly to the wrong line.
SUBTLE = [
    ('montecarlo/solver.py', '                                              90))',
     '                                              10))',
     'worst-year read from the wrong tail'),
    ('montecarlo/solver.py', '-np.inf, stat', 'np.inf, stat',
     'a depleted path scores as the BEST outcome'),
    ('montecarlo/solver.py', 'np.percentile(stat, 100.0 * (1.0 - confidence))',
     'np.percentile(stat, 100.0 * confidence)',
     'closed-form solve takes the wrong quantile'),
    ('montecarlo/diagnostics.py', 'argmin(axis=0)', 'argmax(axis=0)',
     'binding band is the BEST life stage, not the worst'),
    ('montecarlo/engine.py', 'np.random.normal(0, RF_REAL_STD)',
     'np.random.normal(0, RF_REAL_STD * 10)',
     'shared cash-rate band 10x too wide'),
    ('montecarlo/engine.py', 'INFL_BETA.get(ticker, INFL_BETA_DEFAULT) * infl_shock',
     'abs(INFL_BETA.get(ticker, INFL_BETA_DEFAULT)) * infl_shock',
     'inflation HELPS every asset instead of hurting'),
    # --- Social Security and taxes, added 2026-09-29 ---
    ('montecarlo/policies.py', '(1.0 + DELAY_CREDIT_PER_YEAR * years_late)',
     '(1.0 + 2 * DELAY_CREDIT_PER_YEAR * years_late)',
     'delayed claiming credit applied twice'),
    ('montecarlo/policies.py', '    return float(benefit * scale)',
     '    return float(benefit * scale * scale)',
     'benefit haircut applied twice'),
    ('montecarlo/investor.py', 'not 62 <= c <= 70', 'not 62 <= c <= 75',
     'a claim age past 70 is accepted'),
    ('run.py', "        cfg['social_security'] = spec['social_security']",
     "        cfg['social_security'] = (spec['social_security']\n"
     "                                  or cfg['social_security'])",
     'a None override no longer excludes Social Security'),
    ('tools/tax_estimate.py', 'pretax / RMD_DIVISOR[age]',
     'pretax / RMD_DIVISOR.get(age + 1, RMD_DIVISOR[age])',
     'RMD divisor read one row off'),
    ('tools/tax_estimate.py', 'if birth_year >= 1960:',
     'if birth_year >= 1990:', 'RMD age reverted to 73 for this investor'),
    ('tools/sensitivity.py',
     "                                      benefit_scale=float(value))",
     "                                      claim_age=float(value))",
     'the SS sweeps move the wrong field'),
    # --- universe-sized draws, added 2026-10-02 ---------------------------
    # Each keeps common random numbers ACROSS fund sets intact — so the two
    # CRN tests pass — while giving a wrong answer within one.
    ('montecarlo/engine.py', '                                   param_z[pos], prem_z)',
     '                                   param_z[0], prem_z)',
     'every fund shares one estimation-error draw'),
    ('montecarlo/engine.py', '<= u[_pos][:, None]).sum(axis=1)',
     '<= np.full(len(_pos), u[0])[:, None]).sum(axis=1)',
     "every fund's regime moves in lockstep"),
    ('montecarlo/engine.py', 'regime))[ticker_indices, :]',
     'regime))[:len(ticker_indices), :]',
     'the first rows of the factor, not the held ones'),
    # --- shared premium errors, added 2026-10-02 ----------------------------
    ('montecarlo/engine.py', '+ (1.0 - q) ** 0.5 * own_z)', '+ own_z)',
     "a fund's band inflated by its shared part"),
    ('montecarlo/engine.py', 'return band * (float(share @ prem_z) +',
     'return band * (float(share @ np.random.standard_normal(len(prem_z))) +',
     'each fund draws its own premium errors'),
    ('montecarlo/engine.py', '    return PREMIUM_CHOL @ z', '    return z',
     'premium errors drawn uncorrelated'),
    ('montecarlo/engine.py', 'q = float(share @ PREMIUM_CORR @ share)', 'q = float(share @ share)',
     'own part sized as if the premia were uncorrelated'),
    ('montecarlo/factors.py', '(+1.00, +0.27, -0.21,', '(+1.00, +0.07, -0.21,',
     'one cell of PREMIUM_CORR left stale'),
    ('montecarlo/engine.py', 'prem_z = premium_draw(sim_id) if',
     'prem_z = premium_draw(sim_id + len(active_tickers)) if',
     'the premium draw depends on what the plan holds'),
    ('montecarlo/factors.py', "_os.environ.get('MC_PARAM_ERRORS', 'shared')",
     "_os.environ.get('MC_PARAM_ERRORS', 'independent')",
     'estimation errors silently back to independent'),
    ('montecarlo/factors.py', "mkt_rf=out[source]['mkt_rf'] - gap * haircut",
     "mkt_rf=out[source]['mkt_rf'] - gap",
     'a regional gap that ignores the haircut'),
    ('montecarlo/factors.py', "mkt_rf=out[source]['mkt_rf'] - gap * haircut",
     "mkt_rf=out[source]['mkt_rf'] + gap * haircut",
     "'intl keeps lagging' that has international LEADING"),
    ('montecarlo/factors.py', "RECENT_LAG = {'dev': 0.0395, 'emg': 0.0224}",
     "RECENT_LAG = {'dev': 0.0395, 'emg': 0.0124}",
     'RECENT_LAG left stale'),
    ('montecarlo/return_views.py', "    env.update(RETURN_KNOBS)\n", "",
     'a return view that inherits the run\'s model'),
    ('montecarlo/return_views.py', "regret = {g: (val[leader] - val[g]) / top for g in specs}",
     "regret = {g: (val[g] - val[leader]) / top for g in specs}",
     'regret measured the wrong way round'),
    ('montecarlo/return_views.py', "holds[g] = bool(hi >= 0)", "holds[g] = bool(lo >= 0)",
     'a tie with the leader counted as a loss'),
    ('montecarlo/return_views.py', "and _same_scenario(scenario, k['MC_PREMIA_SCENARIO'])):",
     "):", 'every scenario read as the default view'),
    ('run.py', "    out = copy.deepcopy(cfg)\n    value = float(value)",
     "    out = cfg\n    value = float(value)", 'a plan choice that edits the shared base'),
    ('run.py', "    return mine if mine in values else values[0]",
     "    return values[0]", 'the first value typed read as yours'),
    ('run.py', "    if len(set(vals)) != len(vals):", "    if False:",
     'a repeated choice overwriting a solve'),
    ('montecarlo/report_retirement.py', "np.isclose(per_path, np.median(per_path),",
     "np.isclose(per_path, r.get('spend') or 0.0,",
     'fixed spending recognised only at the plan level'),
    ('montecarlo/return_views.py', "            print(f\"    ⚠ {v.title}: failed, left out of the table — {e}\")",
     "            pass", 'a failed view dropped silently'),
    # --- single stock (N10), added 2026-10-20 ----------------------------------
    ('montecarlo/factors.py',
     'axis=1).dropna().iloc[-months:]',
     'axis=1).dropna()',
     'single stock fitted on its whole history'),
    ('montecarlo/factors.py',
     "    return dict(fit, method='single_stock')",
     '    return fit',
     'a single-stock fit labelled as a fund'),
    ('montecarlo/factors.py',
     '_loaded_premium(fit, premia_for(scenario, haircut)) + edge',
     "_loaded_premium(fit, premia_for(scenario, haircut)) + edge + fit.get('alpha', 0.0)",
     "a single stock's alpha flows into its forecast"),
    ('montecarlo/factors.py',
     '    return rf + _loaded_premium(fit, premia_for(scenario, haircut)) + edge',
     '    return rf + _loaded_premium(fit, premia_for(scenario)) + edge',
     'the haircut skips the single stock'),
    ('montecarlo/factors.py',
     '    edge = SINGLE_STOCK_EDGE if edge is None else edge\n    return rf',
     '    edge = SINGLE_STOCK_EDGE\n    return rf',
     'a stated edge ignored'),
    ('montecarlo/factors.py',
     "_os.environ.get('MC_SINGLE_STOCK_EDGE', '0.0')",
     "_os.environ.get('MC_SINGLE_STOCK_EDGE', '0.01')",
     'the default edge is not zero'),
    ('montecarlo/factors.py',
     '                out[t] = mu\n',
     '                out[t] = mu + SINGLE_STOCK_EDGE\n',
     'the edge reaches every fund'),
    ('montecarlo/factors.py',
     "        elif method == 'single_stock':\n            out[t] = single",
     "        elif method == 'single-stock':\n            out[t] = single",
     'a single stock falls back to its historical mean'),
    ('montecarlo/factors.py',
     '            continue           # keeps the 0.25 x vol single-stock band',
     '            pass',
     'the single-stock band replaced by a derived one'),
    ('montecarlo/factors.py',
     "fit.get('method') not in ('factor', 'single_stock'):",
     "fit.get('method') not in ('factor',):",
     'a single stock draws its premium errors alone'),
    ('montecarlo/factors.py',
     '        allowed = edge + max(loaded, 0.0) + SINGLE_STOCK_GATE_TOL',
     '        allowed = edge + SINGLE_STOCK_GATE_TOL',
     'the gate fails a lead the loadings earn'),
    ('montecarlo/factors.py',
     '        allowed = edge + max(loaded, 0.0) + SINGLE_STOCK_GATE_TOL',
     '        allowed = max(loaded, 0.0) + SINGLE_STOCK_GATE_TOL',
     'the gate ignores the stated edge'),
    ('montecarlo/factors.py',
     "        loaded = _loaded_premium(fit, p) - p['us']['mkt_rf']",
     '        loaded = _loaded_premium(fit, p)',
     'the gate allows the market premium twice'),
    ('montecarlo/factors.py',
     '        if lead > allowed:',
     '        if lead > allowed + 0.05:',
     'the gate 5pp too loose'),
    ('montecarlo/factors.py',
     '            out.append(f"{t}: no single-stock recipe',
     '            (f"{t}: no single-stock recipe',
     'a stock with no recipe passes the gate'),
    ('montecarlo/persistence.py',
     "'single_stock_edge': engine.SINGLE_STOCK_EDGE,",
     "'single_stock_edge': 0.0,",
     'provenance records no edge whatever was set'),
    ('montecarlo/persistence.py',
     "'rebalancing_bands': engine.REBALANCING_POLICY,",
     "'rebalancing_bands': 'model',",
     'provenance records model bands always'),
    ('montecarlo/diagnostics.py',
     "    'SINGLE_STOCK_EDGE': (0.0, 'MC_SINGLE_STOCK_EDGE'),\n",
     '',
     'a moved edge is not announced'),
    # --- return floor, uniform bands, tail df --------------------------------------
    ('montecarlo/engine.py',
     'np.maximum(asset_returns, -1.0, out=asset_returns)',
     'np.maximum(asset_returns, -2.0, out=asset_returns)',
     'a fund can lose 200% in a month'),
    ('montecarlo/engine.py',
     'np.maximum(asset_returns, -1.0, out=asset_returns)',
     'np.maximum(asset_returns, -0.5, out=asset_returns)',
     'a real crash month clipped at -50%'),
    ('montecarlo/engine.py',
     "UNIFORM_BAND = {'relative': 0.25, 'absolute': 0.05}",
     "UNIFORM_BAND = {'relative': 0.20, 'absolute': 0.05}",
     'uniform band 20% not 25%'),
    ('montecarlo/engine.py',
     '(dict(b) if t in _SINGLE_STOCKS else dict(UNIFORM_BAND))',
     'dict(UNIFORM_BAND)',
     'COST loses its 7% ceiling under uniform'),
    ('montecarlo/engine.py',
     "elif REBALANCING_POLICY != 'model':",
     'elif False:',
     'a misspelt band policy silently runs model'),
    ('montecarlo/engine.py',
     'if TAIL_DF is not None and TAIL_DF <= 0:',
     'if TAIL_DF is not None and TAIL_DF < 0:',
     'MC_TAIL_DF=0 kept as a moved knob'),
    # --- ask.py / run.py --ask -------------------------------------------------
    ('montecarlo/ask.py',
     "        'additionalProperties': False,\n    }",
     "        'additionalProperties': True,\n    }",
     'the ask schema is not strict'),
    ('montecarlo/ask.py',
     "'policy': _nullable_enum(_policies.POLICIES)",
     "'policy': _nullable_enum(list(_policies.POLICIES)[:-1])",
     'a policy the model cannot propose'),
    ('montecarlo/ask.py',
     "    if proposal.get('unsupported'):",
     '    if False:',
     'an unsupported sentence becomes a nearby run'),
    ('montecarlo/ask.py',
     '    spec = runner.parse_spec(json.dumps(spec), portfolios)\n',
     '',
     "a proposal skips the runner's validation"),
    ('montecarlo/ask.py',
     "if response.stop_reason == 'refusal':",
     "if response.stop_reason == 'refused':",
     'a refusal read as a spec'),
    ('montecarlo/ask.py',
     '        system=_system(portfolios),',
     '        system=_system({}),',
     'the model chooses from no catalogue'),
    ('run.py',
     "'n').lower() not in ('y', 'yes'):",
     "'n').lower() in ('q',):",
     '--ask runs on any answer but q'),
    ('run.py',
     'return f"{env}python3 run.py --spec {shlex.quote(payload)}"',
     'return f"{env}python3 run.py {shlex.quote(payload)}"',
     '--ask shows no replayable --spec'),
    # --- tools/single_stock_benchmark.py ---------------------------------------
    ('tools/single_stock_benchmark.py',
     "    bench = j['rf'] + sum(b * j[f] for f, b in loads.items())",
     "    bench = j['rf'] + loads['mkt_rf'] * j['mkt_rf']",
     'benchmark is market-only'),
    ('tools/single_stock_benchmark.py',
     '    e = monthly_excess.iloc[-n:]',
     '    e = monthly_excess',
     'verdict reads the whole history, not 10 years'),
    ('tools/single_stock_benchmark.py',
     '    se = float(e.std() * np.sqrt(12) / np.sqrt(years))',
     '    se = float(e.std() * np.sqrt(12))',
     'verdict SE ignores the window'),
    ('tools/single_stock_benchmark.py',
     '    if len(monthly_excess) < n:',
     '    if len(monthly_excess) < 12:',
     'one year of history decides'),
    ('tools/single_stock_benchmark.py',
     "'EXIT' if mean < threshold else 'HOLD'",
     "'EXIT' if mean < EXIT_THRESHOLD else 'HOLD'",
     'the --threshold is ignored'),
    # --- parse_spec custom schedules -------------------------------------------
    ('run.py',
     '                glide.validate(glide.parse_schedule(gspec, age), portfolios)\n            except ValueError as exc:\n                raise ValueError(f"custom schedule',
     '                glide.parse_schedule(gspec, age)\n            except ValueError as exc:\n                raise ValueError(f"custom schedule',
     'custom schedules parsed but not validated'),
    # --- drift tests: the design-runs skill and the knob banner ----------------
    ('.claude/skills/design-runs/SKILL.md',
     '"portfolios":["168","169","140"]',
     '"portfolios":["168","169","999"]',
     'the skill recommends a missing portfolio'),
    ('.claude/skills/design-runs/SKILL.md',
     'sensitivity.py premium 1.0,0.5,0.0',
     'sensitivity.py premia 1.0,0.5,0.0',
     'the skill names a missing sweep'),
    ('.claude/skills/design-runs/SKILL.md',
     '--strategies 168,169',
     '--strategies 168,168>168+5:gold_only@60',
     'the skill sweeps a bad schedule'),
    ('.claude/skills/design-runs/SKILL.md',
     '`MC_ROUND_TRIP`',
     '`MC_ROUND_TRIPS`',
     'the skill lists a knob nothing reads'),
    ('montecarlo/diagnostics.py',
     "    'PARAM_STD_SCALE': (1.0, 'MC_PARAM_STD_SCALE'),\n",
     '',
     'a knob the model reads, announced by nothing'),
    ('montecarlo/diagnostics.py',
     "    'TAIL_DF': (None, 'MC_TAIL_DF'),",
     "    'TAIL_DF': (None, 'MC_TAIL_DFX'),",
     'the banner names the wrong env var'),
]


def subtle(cases=None):
    """Apply each wrong-answer mutation; count the checks that notice."""
    print(f"  {'what it breaks':<50}{'noticed by':>11}")
    missed = []
    for path, find, repl, what in (cases or SUBTLE):
        original = (ROOT / path).read_text()
        hits = original.count(find)
        if hits != 1:
            # A STALE case, not a pass: the code it targets has moved, so the
            # mutation is no longer being tested. Three sat skipped unnoticed
            # until 2026-10-20 because a skip did not count as a miss.
            print(f"  {what:<50}{'STALE x' + str(hits):>11}   <-- RE-POINT IT")
            missed.append(f'{what} (stale: find matched {hits}x)')
            continue
        (ROOT / path).write_text(original.replace(find, repl))
        try:
            run = subprocess.run([sys.executable, str(SUITE)],
                                 capture_output=True, text=True, timeout=1800)
            n = len(re.findall(r'\[FAIL\] (\S+)', run.stdout + run.stderr))
        finally:
            (ROOT / path).write_text(original)   # ALWAYS
        flag = '   <-- NOT NOTICED' if n == 0 else ''
        print(f"  {what:<50}{n:>11}{flag}")
        if n == 0:
            missed.append(what)
    return missed


def _strip_prose(code):
    """Comments and docstrings removed — rule 5 of verify-test.

    Twice in one session a test grepped module source for a term that appears
    only in a comment EXPLAINING why the term is not used, so the assertion
    passed against prose rather than code.
    """
    out = re.sub(r'#[^\n]*', '', code)
    out = re.sub(r'"""[\s\S]*?"""', '', out)
    return re.sub(r"'''[\s\S]*?'''", '', out)


def _source_of(dotted):
    """Source of `mod.attr` / `attr`, or None when it cannot be resolved."""
    import importlib
    import inspect
    parts = dotted.split('.')
    for i in range(len(parts) - 1, 0, -1):
        try:
            mod = importlib.import_module('.'.join(parts[:i]))
        except Exception:
            try:    # the suite names library modules bare: engine.X
                mod = importlib.import_module('montecarlo.' + '.'.join(parts[:i]))
            except Exception:
                continue
        obj = mod
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
            return inspect.getsource(obj)
        except Exception:
            return None
    return None


def static():
    """Flag tests that cannot fail by construction, without running anything."""
    src = SUITE.read_text()
    tree = ast.parse(src)
    tests = [n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')]
    findings = []
    for node in tests:
        body = ast.get_source_segment(src, node) or ''
        code = _strip_prose(body)
        why = []
        asserts = [n for n in ast.walk(node) if isinstance(n, ast.Assert)]
        raises = [n for n in ast.walk(node) if isinstance(n, ast.Raise)]
        if not asserts and not raises:
            why.append('no assert and no raise — nothing can fail')
        for a in asserts:
            if isinstance(a.test, ast.Constant) and a.test.value:
                why.append('assert on a literal truth')
        # A swallowing handler only matters when NOTHING else in the test can
        # fail. `try: ... except ValueError: return` followed by `raise
        # AssertionError` is the CORRECT way to assert a rejection, and a test
        # with real assertions elsewhere is not made vacuous by one handler.
        if not asserts and not raises:
            for h in [n for n in ast.walk(node)
                      if isinstance(n, ast.ExceptHandler)]:
                if all(isinstance(b, (ast.Return, ast.Pass)) for b in h.body):
                    why.append('except swallows the failure')
        # An assertion matching ONLY prose. Scoped to the symbol the test
        # actually passed to getsource — checking the whole module flags
        # `assert 'TOP 3' not in getsource(_matrix)` because the phrase lives
        # in a DIFFERENT function's comment, which says nothing about this
        # assertion. ADVISORY: resolving the symbol statically is best-effort.
        grepped = re.findall(r'getsource\(([\w.]+)\)', code)
        for lit in re.findall(
                r"assert\s+'([^']{4,})'\s+(?:not\s+)?in\s+(?:src|body|source)",
                code):
            for sym in grepped:
                text = _source_of(sym)
                if text and lit in text and lit not in _strip_prose(text):
                    why.append(f"{lit!r} appears only in {sym}'s prose")
        if why:
            findings.append((node.name, sorted(set(why))))
    print(f"  {len(tests)} tests scanned, {len(findings)} flagged\n")
    for name, why in findings:
        print(f"    {name}")
        for w in why:
            print(f"        {w}")
    if not findings:
        print("    (nothing flagged)")
    return findings


def _gut(path, fn):
    """Replace a function's body with `return None`, keeping its signature."""
    src = (ROOT / path).read_text()
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.FunctionDef) and node.name == fn):
            continue
        lines = src.splitlines(keepends=True)
        start = node.lineno - 1
        end = max(c.end_lineno for c in ast.walk(node) if hasattr(c, 'end_lineno'))
        head, i = [lines[start]], start
        while not lines[i].rstrip().endswith(':'):
            i += 1
            head.append(lines[i])
        pad = ' ' * (len(lines[start]) - len(lines[start].lstrip()) + 4)
        return ''.join(lines[:start]) + ''.join(head) + f"{pad}return None\n" \
            + ''.join(lines[end:])
    return None


def mutate(targets=TARGETS):
    """Gut each target and count the checks that notice. Zero is a hole."""
    print(f"  {'file':<22}{'function':<32}{'noticed by':>11}")
    holes = []
    for path, fn in targets:
        original = (ROOT / path).read_text()
        mutant = _gut(path, fn)
        if mutant is None:
            # A STALE target, not a pass: the function was renamed or removed,
            # so it is no longer being mutation-tested (same class as subtle()).
            print(f"  {path:<22}{fn:<32}{'NOT FOUND':>11}   <-- RE-POINT IT")
            holes.append((path, f'{fn} (not found)'))
            continue
        (ROOT / path).write_text(mutant)
        try:
            run = subprocess.run([sys.executable, str(SUITE)],
                                 capture_output=True, text=True, timeout=1800)
            n = len(re.findall(r'\[FAIL\] (\S+)', run.stdout + run.stderr))
        finally:
            (ROOT / path).write_text(original)   # ALWAYS, even on Ctrl-C
        if n == 0 and run.returncode != 0:
            # The mutant broke the suite before any check could report (an
            # import error): that is noticed, not a hole.
            print(f"  {path:<22}{fn:<32}{'crashed':>11}")
            continue
        flag = '   <-- NOTHING NOTICED' if n == 0 else ''
        print(f"  {path:<22}{fn:<32}{n:>11}{flag}")
        if n == 0:
            holes.append((path, fn))
    return holes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--static', action='store_true', help='fast structural scan')
    ap.add_argument('--mutate', action='store_true',
                    help='gut each target and count the checks that notice')
    ap.add_argument('--subtle', action='store_true',
                    help='introduce WRONG ANSWERS rather than absent functions')
    args = ap.parse_args(argv)
    if not (args.static or args.mutate or args.subtle):
        args.static = args.mutate = args.subtle = True

    bad = False
    if args.static:
        print("\nSTATIC — tests that cannot fail by construction")
        static()
    if args.mutate:
        print("\nMUTATION — can the suite tell when a function stops working?")
        holes = mutate()
        if holes:
            bad = True
            print(f"\n  {len(holes)} function(s) not caught, or not found:")
            for path, fn in holes:
                print(f"    {path}::{fn}")
        else:
            print("\n  Every target was caught by at least one check.")
    if args.subtle:
        print("\nSUBTLE — is a WRONG ANSWER noticed, not just a missing function?")
        missed = subtle()
        if missed:
            bad = True
            print(f"\n  {len(missed)} wrong answer(s) nothing caught:")
            for what in missed:
                print(f"    {what}")
        else:
            print("\n  Every wrong answer was caught.")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
