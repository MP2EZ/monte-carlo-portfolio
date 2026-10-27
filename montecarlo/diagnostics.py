"""What a --debug run can say about WHY the numbers came out as they did.

Four questions the model could always answer and never did. Each is computed
from data the solver already returns; nothing here simulates anything, and
nothing here is on the path of a normal run.

The ordering is deliberate — 1 and 2 explain a particular result, 3 and 4 check
that the machinery producing it behaved as designed.
"""

import numpy as np

from montecarlo import engine
from montecarlo import factors
from montecarlo import solver

STATES = ('low', 'medium', 'high')


def market_stationary():
    """The market chain's own long-run distribution, from the shipped chain.

    Read from `engine.REGIME_TRANSITIONS` rather than quoted from the docs, so
    the comparison cannot go stale the next time the chain is re-estimated —
    which has happened five times.
    """
    rows = engine.REGIME_TRANSITIONS.get(engine.MARKET_REGIME_KEY)
    if not rows:
        return None
    P = np.array([[rows[a].get(b, 0.0) for b in STATES] for a in STATES])
    v = np.full(3, 1 / 3)
    for _ in range(2000):          # power iteration; the chain is small and ergodic
        nxt = v @ P
        if np.abs(nxt - v).max() < 1e-12:
            break
        v = nxt
    return dict(zip(STATES, (v / v.sum()).tolist()))


# One glyph per life stage. Three shares that sum to 1 are exactly what a
# stacked bar is for, and the SHAPE is the finding: across all paths the first
# decade binds ~85% of the time, while among the FAILING paths it inverts to the
# last. That reversal is the most interesting thing in this block and it is easy
# to skim past as six percentages.
_STACK_GLYPHS = ('#', '+', '.')
_STACK_WIDTH = 24


def _stack(shares):
    """A stacked bar for shares that sum to 1. Rounds to a fixed total width."""
    if not shares:
        return ''
    cells = [int(round(x * _STACK_WIDTH)) for x in shares]
    # Rounding must not change the bar's length, or rows stop lining up.
    drift = _STACK_WIDTH - sum(cells)
    if cells:
        cells[max(range(len(cells)), key=lambda i: cells[i])] += drift
    return ''.join(g * max(c, 0) for g, c in
                   zip(_STACK_GLYPHS, cells))[:_STACK_WIDTH]


def binding_band(raw, config, floor, spend):
    """ITEM 1 — which life stage set the level.

    `adequacy='per_band'` certifies the WORST band's average, so one band
    decides every plan level and the report never said which. Same $201k means
    opposite things: band 1 binding is sequence risk in your sixties, band 3 is
    longevity. The argmin is already computed inside
    `_marginal_portfolio_share` and thrown away.

    Reported for ALL paths and for the FAILING ones separately, because those
    are different questions — the second is what a higher level would have to
    survive.

    Under 'window' adequacy there are twenty-odd overlapping windows, not three
    bands, so the answer is the binding window's START AGE (10th / median /
    90th percentile across paths) rather than a share per band. 'lifetime' has
    no binding stretch and reports the thirds, as it always did.
    """
    spending = raw['spending']
    if spending.size == 0:
        return None
    mode = solver.adequacy_mode(config.get('adequacy', solver.DEFAULT_ADEQUACY))
    start_age = config['withdrawal']['start_age']
    end_age = config['age'] + config['time_horizon_years']
    bounds = (solver.adequacy_bounds(config, spending.shape[1], mode)
              if mode == 'window'
              else solver.stage_bounds(start_age, end_age, spending.shape[1]))
    per_band = np.stack([spending[:, lo:hi].mean(axis=1) for lo, hi, _ in bounds])
    binding = per_band.argmin(axis=0)

    stat = solver.adequacy_statistic(spending, config, mode)
    failed = (stat < floor * spend) | raw['depleted']
    labels = [lab for _, _, lab in bounds]

    if mode == 'window':
        starts = start_age + np.array([lo for lo, _, _ in bounds], dtype=float)

        def ages(mask):
            if not mask.any():
                return None
            sel = starts[binding[mask]]
            return {q: float(np.percentile(sel, q)) for q in (10, 50, 90)}

        return {'mode': 'window',
                'start_age': {'all': ages(np.ones(len(binding), dtype=bool)),
                              'failing': ages(failed)},
                'n_failing': int(failed.sum())}

    def share(mask):
        if not mask.any():
            return None
        sel = binding[mask]
        return [float((sel == i).mean()) for i in range(len(bounds))]

    return {'mode': mode, 'labels': labels,
            'all': share(np.ones(len(binding), dtype=bool)),
            'failing': share(failed),
            'n_failing': int(failed.sum())}


def failure_timing(raw, config, floor, spend):
    """ITEM 2 — WHEN the failing paths failed.

    The report says "Mostly UNDERSPENDING, not ruin" and stops there. Underspend
    concentrated at 66-72 is sequence-of-returns risk and argues for a glide
    path; spread evenly it is simply a level set too high, and the fix is the
    level. Same headline, different remedy.
    """
    spending = raw['spending']
    if spending.size == 0:
        return None
    start_age = config['withdrawal']['start_age']
    bar = floor * spend
    stat = solver.adequacy_statistic(spending, config,
                                     config.get('adequacy',
                                                solver.DEFAULT_ADEQUACY))
    under = stat < bar
    out = {'n_paths': int(spending.shape[0]), 'n_underspent': int(under.sum())}

    if under.any():
        # First year the path's own spending dropped under the bar. argmax on a
        # boolean gives the first True; rows with no dip are excluded already.
        below = spending[under] < bar
        has = below.any(axis=1)
        first = below[has].argmax(axis=1) + start_age
        out['first_dip_age'] = {q: float(np.percentile(first, q))
                                for q in (10, 50, 90)} if first.size else None
        # Where in the horizon the shortfall sits, as a share of under-bar years
        thirds = np.array_split(np.arange(spending.shape[1]), 3)
        out['dip_share_by_third'] = [
            float(below[:, t].sum() / max(below.sum(), 1)) for t in thirds]

    ages = [a for a in raw.get('depletion_ages') or [] if a is not None]
    out['n_depleted'] = len(ages)
    out['depletion_age'] = ({q: float(np.percentile(ages, q))
                             for q in (10, 50, 90)} if ages else None)
    return out


def regime_occupancy(raw):
    """ITEM 3 — did the run reproduce the chain it was built from?

    The regime machinery has been reworked five times and nothing ever checked
    that a RUN visits the states at the rate its own Markov chain implies. The
    shipped chain's stationary distribution is 0.304/0.403/0.293 against a
    historical 0.300/0.399/0.300; paths start pinned at 'medium', which is
    recorded as mixing within 33 months and running ~3.3% hotter in year one.
    This turns that argument into a measurement, and a degenerate chain becomes
    visible immediately rather than as an unexplained level shift.
    """
    months = [m for m in raw.get('regime_months') or [] if m]
    if not months:
        return None
    total = {k: sum(m.get(k, 0) for m in months) for k in ('low', 'medium', 'high')}
    n = sum(total.values())
    if not n:
        return None
    return {'realised': {k: v / n for k, v in total.items()},
            'stationary': market_stationary(),
            'months': n}


def uncertainty_split(raw):
    """ITEM 4 — not knowing the FUTURE versus not knowing the INPUTS.

    The three-stage architecture exists to keep estimation error (epistemic,
    drawn once per path) apart from market noise (aleatoric, drawn monthly), and
    nothing reported how much of the spread each one owns. `path_mu` is what the
    path believed; regressing the outcome on it gives the share of variance the
    belief explains, and the rest is the market.

    A share near zero would mean the parameter bands are decorative; near one,
    that the model is mostly restating its own priors. Neither is knowable
    without measuring it.

    The first version of this reported 29% against 56% for VT. That was the
    UNCENSORED statistic, which three to nine near-depleted paths out of 1,500
    could swing from 7% to 38% — so the gap was real but its size was noise.
    Re-measured: 40-42% against 52%.

    R-squared of a single regressor, not a variance decomposition of the full
    model — the relationship is monotone but not exactly linear over 58 years of
    compounding, so read it as a lower bound on the epistemic share.
    """
    mu = np.asarray(raw.get('path_mu'), dtype=float)
    if np.isfinite(mu).sum() < 8 or np.nanstd(mu) == 0:
        return None

    # A BEQUEST BELOW ONE YEAR OF SPENDING IS RUIN, NOT AN OUTCOME. Excluding
    # only `y == 0` was not enough: a path that NEARLY depletes ends on a few
    # hundred dollars, and log() puts it ~14 units from the mass, where three to
    # nine such paths out of 1,500 swung this statistic from 7% to 38% across
    # six independent samples of ONE strategy. Those paths are censored by the
    # withdrawal policy rather than drawn from the wealth distribution, so they
    # belong with `depleted`, not in the regression.
    #
    # The threshold has model meaning — it is not a percentile picked to look
    # stable. Confirmed by a Spearman rank correlation, which needs no
    # threshold at all and is immune to leverage points by construction: the
    # two agree within 1pp on all six samples (44/41/41/46/43/44 against
    # 45/41/40/46/43/44).
    spending_arr = np.asarray(raw.get('spending'), dtype=float) \
        if raw.get('spending') is not None and np.size(raw.get('spending')) \
        else None
    floor_y = (float(np.median(spending_arr)) if spending_arr is not None
               else float(raw.get('spend') or 0.0))
    dropped = [0]

    def share(y, censor=0.0):
        # log, because wealth is a product of monthly returns: the belief enters
        # multiplicatively and a linear fit on levels would understate it purely
        # through skew.
        y = np.asarray(y, dtype=float)
        pos = np.isfinite(mu) & np.isfinite(y) & (y > 0)
        ok = pos & (y >= censor)
        dropped[0] = int(pos.sum() - ok.sum())
        if ok.sum() < 8:
            return None
        # More than a tenth censored and this describes a SELECTED subsample,
        # not the run. Say nothing rather than something about a different
        # population.
        if pos.sum() and dropped[0] > 0.10 * pos.sum():
            return None
        r = float(np.corrcoef(mu[ok], np.log(y[ok]))[0, 1])
        return r * r

    # TWO OUTCOMES, because the report ranks on one and this defaulted to the
    # other. `finals` in a retirement run is the BEQUEST; the spending the whole
    # report is about is a different question and can give a different answer.
    # Reporting only the bequest under a heading about "outcome spread" is the
    # same error as the ✓ line that quoted the plan level under "most spending".
    spending = raw.get('spending')
    by_spend = (share(np.asarray(spending, dtype=float).mean(axis=1))
                if spending is not None and np.size(spending) else None)
    by_wealth = share(raw['finals'], censor=floor_y)
    ok = np.isfinite(mu)
    r2 = by_spend if by_spend is not None else by_wealth
    if r2 is None:
        return None
    rf, infl = raw.get('rf_shock'), raw.get('infl_shock')

    def spread(a):
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        return float(a.std()) if a.size else None

    return {'epistemic_share': r2,
            'epistemic_spending': by_spend,
            'epistemic_wealth': by_wealth,
            'wealth_censored': dropped[0],
            'mu_spread': float(np.std(mu[ok])),
            'mu_p10_p90': (float(np.percentile(mu[ok], 10)),
                           float(np.percentile(mu[ok], 90))),
            'rf_spread': spread(rf),
            'infl_spread': spread(infl),
            'n': int(ok.sum())}


def report(by_strategy, config, floor, width=14):
    """Print all four blocks. Called only from a --debug run.

    Takes {label: raw} where raw is what `solver.simulate` returned for the
    strategy's FINAL verification run — the one whose numbers the report
    printed. Scoring an earlier search evaluation would describe a level that
    was never reported.
    """
    rows = [(g, r) for g, r in by_strategy.items() if r]
    if not rows:
        return

    first = binding_band(rows[0][1], config, floor, rows[0][1].get('spend', 0.0))
    if first and first['mode'] == 'window':
        print(f"\n  WHICH {solver.ADEQUACY_WINDOW_YEARS}-YEAR STRETCH SET THE LEVEL")
        print(f"    window adequacy certifies the WORST {solver.ADEQUACY_WINDOW_YEARS}"
              f"-year average; where it starts says\n    whether the risk is "
              f"sequence (early) or longevity (late).")
        print(f"    {'strategy':<{width}}{'starts at  p10 / median / p90':>32}")
        for g, raw in rows:
            b = binding_band(raw, config, floor, raw.get('spend', 0.0))
            if not b or not b['start_age']['all']:
                continue
            for tag, key, n in ((g, 'all', ''), ('  ↳ failing', 'failing',
                                                 f"  {b['n_failing']:,}")):
                a = b['start_age'][key]
                if a:
                    print(f"    {tag:<{width}}{a[10]:>20.0f} / {a[50]:.0f} / "
                          f"{a[90]:.0f}{n}")
    elif first:
        print(f"\n  WHICH LIFE STAGE SET THE LEVEL")
        print(f"    per-band adequacy certifies the WORST band, so one decides "
              f"each level.")
        head = ''.join(f"{lab:>12}" for lab in first['labels'])
        legend = ' '.join(f"{g}={lab}" for g, lab in
                          zip(_STACK_GLYPHS, first['labels']))
        print(f"    {'strategy':<{width}}{head}   {legend}")
        for g, raw in rows:
            b = binding_band(raw, config, floor, raw.get('spend', 0.0))
            if not b or not b['all']:
                continue
            print(f"    {g:<{width}}" + ''.join(f"{x:>12.0%}" for x in b['all'])
                  + '   ' + _stack(b['all']))
            if b['failing']:
                # THE COUNT CANNOT SIT IN THE STRATEGY COLUMN. "↳ 145 failing"
                # truncated to "↳ 145 failin" and 145, 144 and 140 are all real
                # portfolio ids — so a path count read as a portfolio. It goes
                # after the percentages, where nothing else is numbered.
                print(f"    {'  ↳ failing':<{width}}"
                      + ''.join(f"{x:>12.0%}" for x in b['failing'])
                      + '   ' + _stack(b['failing'])
                      + f"  {b['n_failing']:,}")

    print(f"\n  WHEN THE FAILING PATHS FAILED")
    print(f"    concentrated early is sequence risk — a glide path is the "
          f"answer;\n    spread evenly is a level set too high, and the level "
          f"is the answer.")
    print(f"    {'strategy':<{width}}{'under bar':>10}{'1st dip':>9}"
          f"{'early/mid/late':>18}{'depleted':>10}")
    for g, raw in rows:
        t = failure_timing(raw, config, floor, raw.get('spend', 0.0))
        if not t:
            continue
        dip = t.get('first_dip_age')
        thirds = t.get('dip_share_by_third')
        dep = t.get('depletion_age')
        rate = t['n_underspent'] / max(t['n_paths'], 1)
        when = f"age {dip[50]:.0f}" if dip else '—'
        split = ' '.join(f"{x:.0%}" for x in thirds) if thirds else '—'
        print(f"    {g:<{width}}{rate:>10.1%}{when:>9}{split:>18}"
              f"{t['n_depleted']:>10}")

    seq = [(g, sequence_risk(raw, config)) for g, raw in rows]
    seq = [(g, v) for g, v in seq if v]
    if seq:
        print(f"\n  SEQUENCE RISK   does the ORDER of returns matter, or just "
              f"the level?")
        print(f"    {'strategy':<{width}}{'first 10y':>11}{'last 10y':>10}"
              f"{'whole':>8}{'ORDER alone':>14}")
        for g, v in seq:
            whole = f"{v['whole']:.0%}" if v['whole'] is not None else '—'
            extra = (f"{v['sequence_extra']:.0%}"
                     if v['sequence_extra'] is not None else '—')
            print(f"    {g:<{width}}{v['first_decade']:>11.0%}"
                  f"{v['last_decade']:>10.0%}{whole:>8}{extra:>14}")
        print(f"    The first three are share of the spending outcome "
              f"explained. They are NOT\n    a sequence result on their own: an "
              f"early return compounds for thirty years\n    and a late one for "
              f"ten, so the first decade leads even if order is\n    irrelevant. "
              f"'ORDER alone' holds the WHOLE retirement return fixed and asks "
              f"what\n    the first decade still adds — that is sequence risk, "
              f"and only that column\n    argues for a glide path.")

    occ = regime_occupancy(rows[0][1])
    if occ and occ['stationary']:
        print(f"\n  REGIME OCCUPANCY   did the run reproduce its own chain?")
        r, st = occ['realised'], occ['stationary']
        print(f"    {'':<{width}}{'low':>10}{'medium':>10}{'high':>10}")
        print(f"    {'realised':<{width}}" + ''.join(f"{r[k]:>10.1%}" for k in STATES))
        print(f"    {'stationary':<{width}}" + ''.join(f"{st[k]:>10.1%}" for k in STATES))
        gap = max(abs(r[k] - st[k]) for k in STATES)
        note = ("paths start pinned at 'medium', so a small excess there is "
                "expected" if gap < 0.05 else
                "LARGER THAN EXPECTED — check the chain, not the portfolio")
        print(f"    worst gap {gap:.1%} over {occ['months']:,} months — {note}")

    print(f"\n  WHAT IS UNCERTAIN   not knowing the future vs the inputs")
    print(f"    {'strategy':<{width}}{'on spending':>12}{'on bequest':>11}"
          f"{'drawn mu p10-p90':>20}{'rf sd':>8}{'infl sd':>9}")
    for g, raw in rows:
        u = uncertainty_split(raw)
        if not u:
            continue
        lo, hi = u['mu_p10_p90']
        band = f"{lo:.2%} .. {hi:.2%}"
        rf = f"{u['rf_spread']:.2%}" if u['rf_spread'] else '—'
        infl = f"{u['infl_spread']:.2%}" if u['infl_spread'] else '—'
        sp = f"{u['epistemic_spending']:.0%}" if u['epistemic_spending'] is not None else '—'
        wl = f"{u['epistemic_wealth']:.0%}" if u['epistemic_wealth'] is not None else '—'
        print(f"    {g:<{width}}{sp:>12}{wl:>11}{band:>20}{rf:>8}{infl:>9}")
    print(f"    share of spread explained by the path's own drawn mu — a LOWER "
          f"BOUND, from\n    one regressor; the rest is the market.")
    print(f"    The two columns differ because the POLICY sits between them: "
          f"guardrails absorb\n    a wrong return assumption into the estate "
          f"rather than the lifestyle. Measured\n    at 1,500 paths, VT reads "
          f"40-42% on spending against 52% on bequest, and 140\n    reads "
          f"35-38% against 43-45% — the single-fund book carries the wider gap.")


def sequence_risk(raw, config):
    """ITEM 2 — is it the ORDER of returns, or just the level?

    `failure_timing` reports WHEN the shortfall fell and the conclusion was
    drawn from that: spread evenly, so it is the level rather than sequence.
    That is an inference from timing and it is not airtight — a path can dip
    late precisely because bad early returns compounded.

    The direct test compares how well the FIRST decade of retirement predicts
    the outcome against how well the LAST one does. Sequence risk is the claim
    that early returns matter more than late ones for the same average; if the
    two decades explain the outcome about equally, the order does not matter and
    the level is the lever.

    Returns None on a search evaluation — the monthly series only comes back on
    a full run.
    """
    rets = raw.get('returns')
    spending = raw.get('spending')
    if rets is None or spending is None or not np.size(rets):
        return None
    rets = np.asarray(rets, dtype=float)
    start = int((config['withdrawal']['start_age'] - config['age']) * 12)
    if rets.shape[1] < start + 240:
        return None

    def cum(a, b):
        # log-sum, not a product: the outcome compounds, and a 120-month product
        # of near-1 numbers loses precision for no benefit.
        return np.log1p(rets[:, a:b]).sum(axis=1)

    outcome = np.asarray(spending, dtype=float).mean(axis=1)
    ok = np.isfinite(outcome) & (outcome > 0)
    if ok.sum() < 30:
        return None
    y = np.log(outcome[ok])
    n_ret = rets.shape[1] - start

    def r2(a, b):
        x = cum(start + a, start + b)[ok]
        if np.std(x) == 0:
            return None
        return float(np.corrcoef(x, y)[0, 1]) ** 2

    first, last = r2(0, 120), r2(max(n_ret - 120, 120), n_ret)
    if first is None or last is None:
        return None

    # THE CONFOUND, AND THE ACTUAL TEST. An early return compounds for thirty
    # years and a late one for ten, so the first decade explains more even when
    # the ORDER is irrelevant — comparing the two raw R-squareds measures
    # compounding, not sequence risk. Sequence risk is the claim that for the
    # SAME total return the order still matters, so the test is whether the
    # first decade adds anything ON TOP of the whole-retirement return.
    total = cum(start, start + n_ret)[ok]
    early = cum(start, start + 120)[ok]
    extra = None
    if np.std(total) > 0 and np.std(early) > 0:
        X = np.column_stack([np.ones(ok.sum()), total, early])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        both = 1.0 - float(np.var(resid) / np.var(y))
        base = float(np.corrcoef(total, y)[0, 1]) ** 2
        extra = max(both - base, 0.0)

    return {'first_decade': first, 'last_decade': last,
            'whole': r2(0, n_ret), 'sequence_extra': extra, 'n': int(ok.sum())}


# What each knob reads with nothing set. A run that matches these is the
# shipped model; anything else was DELIBERATELY moved, and the report should
# say so rather than looking identical to a default run.
DEFAULT_ASSUMPTIONS = {
    'MU_METHOD': ('factor', 'MC_MU_METHOD'),
    'PREMIA_SCENARIO': ('full history', 'MC_PREMIA_SCENARIO'),
    'PARAM_ERRORS': ('shared', 'MC_PARAM_ERRORS'),
    'PREMIUM_HAIRCUT': (1.0, 'MC_PREMIUM_HAIRCUT'),
    'PREMIUM_SCALE': (1.0, 'MC_PREMIUM_SCALE'),
    'RF_REAL': (0.0057, 'MC_RF_REAL'),
    'RF_REAL_STD': (0.01, 'MC_RF_REAL_STD'),
    'INFLATION': (0.03, 'MC_INFLATION'),
    'INFLATION_STD': (0.015, 'MC_INFLATION_STD'),
    'INFL_BETA_DEFAULT': (-0.80, 'MC_INFL_BETA_EQUITY'),
    'INFL_BETA_TIPS': (0.0, 'MC_INFL_BETA_TIPS'),
    'CRISIS_VOL_SCALE': (1.0, 'MC_CRISIS_VOL_SCALE'),
    'TAIL_DF': (None, 'MC_TAIL_DF'),
    'REBALANCING_POLICY': ('model', 'MC_REBALANCING_BANDS'),
    'ROUND_TRIP': (0.0015, 'MC_ROUND_TRIP'),
    'SINGLE_STOCK_EDGE': (0.0, 'MC_SINGLE_STOCK_EDGE'),
    # Added 2026-10-19: read by the model, announced by nothing until then.
    'PARAM_STD_SCALE': (1.0, 'MC_PARAM_STD_SCALE'),
    'RF_REAL_SOURCE': ('historical', 'MC_RF_SOURCE'),
    'INFL_BETA_BOND': (-1.15, 'MC_INFL_BETA_BOND'),
    'PREMIA_LONG_WINDOW': (False, 'MC_PREMIA_LONG_WINDOW'),
    'CORR_WINDOW': (15, 'MC_CORR_WINDOW'),
}


def moved_assumptions(engine_mod=None):
    """Every knob that is NOT at its shipped default, with the env var that did it.

    Twelve environment variables can move every number in a report and the
    footer named exactly one. They reach the run RECORD through provenance, so
    `--diff` can see them — but the run on screen looked identical to a default
    one, which is the same gap the path count had: the record knew, the output
    did not.
    """
    eng = engine_mod or engine
    out = []
    _MISSING = object()
    for attr, (default, env) in sorted(DEFAULT_ASSUMPTIONS.items()):
        # ROUND_TRIP lives in factors, not engine. Reading it off engine alone
        # returned None and reported a DEFAULT run as having a knob moved — a
        # banner that cries wolf on every run is worse than no banner.
        got = getattr(eng, attr, _MISSING)
        if got is _MISSING:
            got = getattr(factors, attr, _MISSING)
        if got is _MISSING:
            continue          # the constant has moved or gone; say nothing
        if isinstance(default, float) and isinstance(got, (int, float)):
            same = abs(float(got) - default) < 1e-12
        elif attr == 'PREMIA_SCENARIO' and got in factors.SCENARIOS:
            # 'us stays ahead' is the default under another name. Compare what
            # the scenario DOES, not what it is called.
            same = factors.SCENARIOS[got] == factors.SCENARIOS[default]
        else:
            same = got == default
        if not same:
            out.append((attr, default, got, env))
    return out


def assumption_banner(engine_mod=None):
    """One line per moved assumption. Silent when the run is the shipped model."""
    moved = moved_assumptions(engine_mod)
    if not moved:
        return
    print(f"\n  ⚠ NON-DEFAULT ASSUMPTIONS — {len(moved)} knob"
          f"{'' if len(moved) == 1 else 's'} moved for this run")
    for attr, default, got, env in moved:
        def fmt(v):
            if v is None:
                return 'off'
            if isinstance(v, float):
                return f"{v:.4g}"
            return str(v)
        print(f"      {env:<24}{fmt(default)} → {fmt(got)}")
    print(f"      every number below reflects these, not the shipped model")
