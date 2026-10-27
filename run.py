#!/usr/bin/env python3
"""
Interactive runner
==================

    python3 run.py            # ask what to compare, then run it
    python3 run.py --again    # repeat the last run

Question-driven rather than list-driven: you pick the QUESTION and the tool
derives the portfolios. With 29 templates a flat list is the failure mode, so
the common case shows three relevant options plus an escape hatch to the full
grouped library.

The glide syntax ('141>340@65') is the stored form, not something you type — the
derisk sets are generated from the tilt family and bond tiers already in the CSV.

Design note: everything above `main()` is pure. Prompting is confined to the
`ask_*` helpers so the run-spec logic can be unit-tested without stdin.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

from montecarlo import engine
from montecarlo import glide
from montecarlo import investor
from montecarlo import persistence
from montecarlo import policies as policies_mod
from montecarlo import instrument
from montecarlo import solver as solver_mod
from montecarlo import loaders
from montecarlo.loaders import (BOND_SLEEVES, compute_derived, load_portfolios_from_csv,
                     native_sleeve)

ROOT = Path(__file__).resolve().parent
PRESETS = ROOT / 'config' / 'runs.json'

BOND_TIER_NOTE = "same tilt, more bonds"

# Order for the 'compare all four' sweep — most conservative first, so the
# table reads as a spectrum of who absorbs a bad market.
POLICY_ORDER = ('guyton_klinger', 'constant_real', 'vpw', 'fixed_pct')


# The RETIREMENT CHOICES: decisions about the plan, compared on one portfolio.
# Each value is solved in full and every option sees the same paths, so the gap
# between two of them is paired. Policies are a plan choice too, but they had
# their own fan-out first (compare_policies) and keep it.
#   key: (menu title, row tag)
PLAN_CHOICES = {
    'retirement_age': ('Retirement age', 'retire at {v:g}'),
    'ss_claim_age': ('Social Security claiming age', 'claim at {v:g}'),
    'contributions': ('Savings', 'save ×{v:g}'),
}


# ---------------------------------------------------------------------------
# Pure logic — no I/O
# ---------------------------------------------------------------------------

def default_config():
    """Stable facts about the investor, from config/investor.json.

    Two fields are DERIVED and never stored, because storing them means keeping
    them in step by hand and both had already failed at that:
    `time_horizon_years` is `plan_to_age - age` (it was a literal 58, correct
    only while age was 37, and the legacy runner's copy said 50 = age 87), and
    `initial_value` is the sum of the account balances. See investor.py.

    `adequacy` and `rebalancing_strategy` stay here: they are modelling choices,
    not facts about you.
    """
    inv = investor.load()
    return {
        'age': inv['age'],
        'retirement_age': inv['retirement_age'],
        # How a path is judged adequate. 'window' (the default) requires every
        # rolling 10-year average to clear the floor; 'thirds' (the default
        # before 2026-10-21, 'per_band' in old records) every third of the
        # retirement; 'lifetime' only the whole-retirement average, which is
        # shape-blind and lets a bad first decade be repaid by a rich ninth.
        # See solver.DEFAULT_ADEQUACY.
        'adequacy': solver_mod.DEFAULT_ADEQUACY,
        'time_horizon_years': inv['plan_to_age'] - inv['age'],
        # From your details (config/investor.json), like the accounts. A spec
        # may override it for one run — see build_config.
        'social_security': (dict(inv['social_security'])
                            if inv.get('social_security') else None),
        'initial_value': investor.total_balance(inv),
        'rebalancing_strategy': 'quarterly_annual',
        'contribution_schedule': investor.contribution_schedule(inv),
    }


def choice_tag(dim, value):
    """The row label for one option, e.g. 'retire at 60'."""
    return PLAN_CHOICES[dim][1].format(v=float(value))


def current_choice(cfg, dim):
    """What your details already say for this choice, or None."""
    if dim == 'retirement_age':
        return float(cfg['retirement_age'])
    if dim == 'ss_claim_age':
        ss = cfg.get('social_security')
        return float(ss['claim_age']) if ss else None
    if dim == 'contributions':
        return 1.0                       # x1 is today's contributions
    raise ValueError(f"unknown choice {dim!r}")


def choice_baseline(cfg, dim, values):
    """The option the others are read against: yours if it is among them,
    else the first value given."""
    values = [float(v) for v in values]
    mine = current_choice(cfg, dim)
    return mine if mine in values else values[0]


def apply_choice(cfg, dim, value):
    """A COPY of `cfg` with one plan choice changed.

    Never the original: the options share one base config, and contributions
    scale the account dicts in place, so an option that edited the base would
    hand its savings rate to every option after it. The sensitivity sweep
    applies these through here too, so the two cannot disagree about what
    'retire at 60' means.

    Retirement age moves when withdrawals start AND caps every account's last
    contribution at the year before it (stop_age is inclusive). It used to
    leave the stop ages alone, so 'retire at 60' kept a Roth IRA contributing
    to 63 — savings that retirement would not have. It only caps: retiring
    later does not extend an account you stop sooner.
    """
    out = copy.deepcopy(cfg)
    value = float(value)
    if dim == 'retirement_age':
        out['retirement_age'] = value
        if out.get('withdrawal'):
            out['withdrawal']['start_age'] = value
        last = int(value) - 1
        for acct in out['contribution_schedule']['accounts'].values():
            stop = acct.get('stop_age')
            acct['stop_age'] = last if stop is None else min(stop, last)
    elif dim == 'ss_claim_age':
        if not out.get('social_security'):
            raise ValueError("a claiming age needs Social Security in your details "
                             "— set it from 'Your details'")
        out['social_security'] = dict(out['social_security'], claim_age=value)
    elif dim == 'contributions':
        for acct in out['contribution_schedule']['accounts'].values():
            acct['annual_amount'] *= value
    else:
        raise ValueError(f"unknown choice {dim!r}")
    return out


def resolve_strategies(spec, portfolios):
    """Turn a run spec into the ordered list of glide specs to simulate.

    Returns (specs, is_grid). `is_grid` is True when the comparison varies on two
    axes (several portfolios x several bond tiers), which selects a matrix layout
    instead of a flat table — a flat list destroys the structure being compared.
    """
    if spec.get('custom'):
        return list(spec['custom']), False

    bases = list(spec['portfolios'])
    at = spec.get('derisk_at')
    if at is None:
        return bases, False

    tiers = tuple(spec.get('bond_tiers') or glide.DERISK_TIERS)
    sleeves = spec.get('compare_sleeves')
    if sleeves:
        # One control plus a rung per (sleeve, tier): identical equity and bond
        # weight, different bond funds — which is what isolates composition.
        grid = []
        for b in bases:
            row = [b]
            for sl in sleeves:
                row += glide.derisk_variants(b, at, tiers=tiers, sleeve=sl)[1:]
            grid.append(row)
    else:
        grid = [glide.derisk_variants(b, at, tiers=tiers,
                                      sleeve=spec.get('bond_sleeve'))
                for b in bases]
    widths = {len(row) for row in grid}
    is_grid = len(bases) > 1 and max(widths) > 1
    return [s for row in grid for s in row], is_grid


def ref_bond_pct(ref, portfolios):
    """Bond weight of any portfolio reference, stored OR synthesised.

    Every caller must go through this rather than indexing `portfolios[ref]`
    directly — synthesised refs like '141+10:bnd_only' are not CSV rows, and
    indexing them raises KeyError. That mistake has been made three times now, so
    the resolution lives in exactly one place.
    """
    return compute_derived(glide.resolve_ref(ref, portfolios))['bond_pct']


def _pct(x):
    """Percent with a decimal only when one is needed: 10, 2.5, 97.5."""
    return f"{x:.1f}".rstrip('0').rstrip('.')


def ref_split(ref, portfolios):
    """'90/10'-style equity/bond label for any reference.

    Fractional tiers are real — a 2.5% bond level rendered as "98/2" under
    `:.0f`, which is neither the allocation nor a pair that sums to 100.
    """
    bond = ref_bond_pct(ref, portfolios)
    return f"{_pct((1 - bond) * 100)}/{_pct(bond * 100)}"


def target_bond_pct(gspec, portfolios):
    """Bond weight of a glide spec's FINAL target — the matrix column it belongs in."""
    return ref_bond_pct(gspec.split('>')[-1].split('@')[0], portfolios)


def bond_tier_columns(spec, portfolios):
    """Ordered (bond_pct, label) columns for the matrix layout.

    Built from the UNION across every selected base, not just the first. Bases can
    have ragged variant counts — 141 (momentum) has four bond tiers while 120
    (value) has none — so keying columns off one base silently misaligns the rest.
    """
    tiers = set()
    for base in spec['portfolios']:
        for gspec in glide.derisk_variants(
            base, spec['derisk_at'],
            tiers=tuple(spec.get('bond_tiers') or glide.DERISK_TIERS),
            sleeve=spec.get('bond_sleeve')):
            tiers.add(round(target_bond_pct(gspec, portfolios), 6))
    ordered = sorted(tiers)
    return [(t, 'no derisk' if i == 0 else f"->{t:.0%}")
            for i, t in enumerate(ordered)]


LABEL_WIDTH_TRIGGER = 18      # beyond this a spec crowds every table column


def strategy_labels(specs):
    """Map specs to short labels (A, B, C…) when the specs are too wide to read.

    Returns {} when they are short enough to print directly — a label is
    indirection, worth paying for only when '140>140+5@45>140+10@55>140+20@65'
    would otherwise be repeated in every column of every table.
    """
    if len(specs) < 2 or max(len(s) for s in specs) <= LABEL_WIDTH_TRIGGER:
        return {}
    return {spec: chr(ord('A') + i) for i, spec in enumerate(specs)}


def build_config(spec):
    """Full simulation config from the run spec."""
    cfg = default_config()
    # PRESENCE, not truthiness: a spec that carries the key overrides your
    # details for this run, and None means "exclude it this run". Absent, the
    # stored benefit applies — which is what makes older saved specs (whose
    # key held the benefit typed that day) replay exactly.
    if 'social_security' in spec:
        cfg['social_security'] = spec['social_security']
    if spec['mode'] == 'retirement':
        wd = {
            'start_age': cfg['retirement_age'],
            'policy': spec.get('policy', 'guyton_klinger'),
            'sourcing': spec.get('sourcing', 'excess'),
            'floor': spec.get('floor', 0.85),
        }
        if spec.get('vpw_bequest'):
            wd['vpw_bequest'] = float(spec['vpw_bequest'])
        if spec.get('gk_response'):
            wd['gk_response'] = spec['gk_response']
        target = spec.get('spend')
        wd['target_real_annual'] = 200_000.0 if target == 'solve' else float(target)
        cfg['withdrawal'] = wd
    return cfg


# Measured wall-clock per path THROUGH THE POOL, not raw compute. The two differ
# by more than they should: ~14ms of simulation but ~4ms/path of pickling and
# dispatch even after dropping the unused arrays. Estimating from raw compute
# made this 5x optimistic, so a solve that took 11 minutes was announced as 2.
_SECONDS_PER_PATH = 0.004
# Bracketing (~2-3) + two sampling passes of n_points (~12) + two verification
# runs. The old value of 14 described the bisection that no longer exists, and
# the verification count was 1 rather than 2 — together that under-priced a run
# by ~2.6x, which is why a 9-solve run announced 4 minutes and took 11.
#
# Still a hand count, so `--debug` measures the real seconds-per-path and writes
# it to config/timing.json; `_seconds_per_path()` prefers that over the constant.
_SOLVE_EVALS = 15
_VERIFY_RUNS = 2


def _seconds_per_path(mode=None):
    """Measured from the last instrumented run of THIS MODE, else the constant.

    Per mode because a decumulation path costs ~3x an accumulation one; see the
    note on instrument.set_mode.
    """
    return instrument.load_seconds_per_path(mode) or _SECONDS_PER_PATH


def policy_runs(spec):
    """The (tag, policy, reserve) fan-out a retirement run will actually perform.

    Defined HERE, and imported by the report, so the estimate, the confirm
    preview, the analysis header and the table cannot disagree about what is
    being run. They did: the estimate counted `compare_policies` only, so a run
    comparing 2 policies across 3 VPW reserves was priced at 6 solves and ran
    12, and the header named a single policy for a table with four.
    """
    if spec.get('mode') != 'retirement':
        return [('accumulation', None, {})]
    choice = spec.get('compare_choice')
    if choice:
        policy = spec.get('policy', 'guyton_klinger')
        return [(choice_tag(choice['dimension'], v), policy,
                 {'vpw_bequest': float(spec.get('vpw_bequest') or 0.0),
                  'gk_response': spec.get('gk_response'),
                  'choice': (choice['dimension'], float(v))})
                for v in choice['values']]
    policies = spec.get('compare_policies') or [spec.get('policy', 'guyton_klinger')]
    reserves = spec.get('compare_vpw_bequests')
    # The THIRD element is a settings dict, not a bare reserve. It was a float
    # while vpw's reserve was the only per-run setting; `gk_response` made that
    # a second one, and a second positional float would have been a silent
    # mis-assignment waiting to happen. Every other unpack site reads the tag or
    # the policy and is unaffected.
    responses = spec.get('compare_gk_responses')
    runs = []
    for policy in policies:
        if policy == 'guyton_klinger' and responses and len(responses) > 1:
            for resp in responses:
                runs.append((f"guyton_klinger ({resp})", policy,
                             {'gk_response': resp}))
            continue
        if policy == 'vpw' and reserves:
            # "keep $5M" read as a promise about the end balance; it is not one.
            # VPW amortises to age 100, so a balance remains at the horizon
            # regardless — the reserve is held back ON TOP of that.
            for reserve in reserves:
                tag = ('vpw no reserve' if not reserve
                       else f"vpw reserve {reserve / 1e6:g}M")
                runs.append((tag, policy, {'vpw_bequest': float(reserve)}))
        else:
            runs.append((policy, policy,
                         {'vpw_bequest': float(spec.get('vpw_bequest') or 0.0),
                          'gk_response': spec.get('gk_response')}))
    return runs


def estimate_minutes(n_strategies, spec):
    """Rough wall-clock estimate, so the confirm step can warn before a long run.

    Priced per POLICY, because a self-directed one costs a fraction of a searched
    one: vpw solves in closed form, so it simulates `verify_sims` once where
    guyton_klinger runs ~15 search evaluations plus two verifications. Measured
    at production sims: 9.0k paths per gk solve against 1.5k per vpw solve.

    Pricing them alike over-counted a 6-solve run 54k paths against an actual
    24k. That happened to cancel against a seconds-per-path constant which was
    itself half the true value, so the estimate looked accurate while both of its
    terms were wrong — and would have broken the moment either was corrected.
    """
    search = spec.get('search_sims', solver_mod.SEARCH_SIMS)
    verify = spec.get('verify_sims', solver_mod.VERIFY_SIMS)
    solving = spec.get('spend') == 'solve'
    per_path = _seconds_per_path(spec.get('mode'))

    total_paths = 0
    for _, policy, _ in policy_runs(spec):
        if not solving:
            paths = spec.get('sims', solver_mod.FIXED_SIMS)
        elif policy in policies_mod.SELF_DIRECTED:
            paths = verify                      # one batch, no search
        else:
            paths = search * _SOLVE_EVALS + verify * _VERIFY_RUNS
        total_paths += n_strategies * paths
    return total_paths * per_path / 60.0 + return_view_minutes(n_strategies, spec)


def checks_return_views(n_strategies, spec):
    """Whether this run repeats its comparison under other return models.

    One policy only: comparing policies is a question about the plan, and the
    check ranks portfolios. One strategy has nothing to rank."""
    return (bool(spec.get('return_views')) and spec.get('mode') == 'retirement'
            and n_strategies > 1 and len(policy_runs(spec)) == 1)


def return_view_minutes(n_strategies, spec):
    """What the return-model check adds: one evaluation of every strategy
    per other view, at the run's path count, plus a child process each."""
    if not checks_return_views(n_strategies, spec):
        return 0.0
    from montecarlo import return_views as rv
    n = (spec.get('verify_sims', solver_mod.VERIFY_SIMS)
         if spec.get('spend') == 'solve'
         else spec.get('sims', solver_mod.FIXED_SIMS))
    seconds = (rv.extra_paths(n_strategies, n) * _seconds_per_path('retirement')
               + (len(rv.VIEWS) - 1) * rv.CHILD_OVERHEAD_SECONDS)
    return seconds / 60.0


def save_preset(spec, name='last'):
    PRESETS.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(PRESETS.read_text()) if PRESETS.exists() else {}
    data[name] = spec
    PRESETS.write_text(json.dumps(data, indent=2) + '\n')
    return PRESETS


def load_preset(name='last'):
    if not PRESETS.exists():
        return None
    return json.loads(PRESETS.read_text()).get(name)


# ---------------------------------------------------------------------------
# Library rendering
# ---------------------------------------------------------------------------

def suggested_bases(portfolios):
    """The three most likely starting points, so 29 templates are never dumped."""
    picks = []
    for pid, note in (('141', 'your holdings'), ('140', 'base'),
                      ('145', 'Vanguard-only mirror')):
        if pid in portfolios:
            picks.append((pid, note))
    if not picks:
        picks = [(pid, '') for pid in sorted(portfolios)[:3]]
    return picks


def describe(portfolios, pid):
    d = compute_derived(glide.resolve_ref(pid, portfolios))
    return (f"{(1 - d['bond_pct']) * 100:.0f}/{d['bond_pct'] * 100:.0f} · "
            f"{d['tilt_family']} · {d['n_funds']} funds")


def distinguish(portfolios, pids):
    """Short labels showing what makes each portfolio DIFFER from its peers.

    Template names share a long common prefix — "70D:30I // 100:0 // 5C //
    12AVUV // 12VFMO" — so truncating one to fit a column cuts exactly where
    they all still agree. The library picker rendered NINE momentum rows as the
    identical string "70D:30I // 100:0 // 5C // 12AV", which makes choosing
    between them impossible.

    A `--` comment is the author's own label and wins. Otherwise show only the
    `//` segments that differ from the peer group's modal value, widening any
    group that still collides until it separates.

    Deliberately NOT inferred: a segment the modal row has and this one lacks
    does not mean the holding is absent. 145 is "VG-only (mirrors 140)" and
    names no sleeves at all while holding them, so "no 12VFMO" would be a lie.
    """
    from collections import Counter
    parsed = {}
    for pid in pids:
        head, sep, note = portfolios[pid]['name'].partition('--')
        parsed[pid] = ([s.strip() for s in head.split('//') if s.strip()],
                       note.strip() if sep else '')
    n = max((len(segs) for segs, _ in parsed.values()), default=0)
    modes = []
    for i in range(n):
        vals = [segs[i] for segs, _ in parsed.values() if i < len(segs)]
        modes.append(Counter(vals).most_common(1)[0][0] if vals else None)

    # Which segments each row shows: those where it departs from the modal
    # value, or — when it departs nowhere — its own last segment.
    keep = {}
    for pid, (segs, note) in parsed.items():
        if note:
            keep[pid] = None                      # the author's label wins
            continue
        odd = {i for i, seg in enumerate(segs) if seg != modes[i]}
        keep[pid] = odd or ({len(segs) - 1} if segs else set())

    def render(pid):
        segs, note = parsed[pid]
        if note:
            return note
        idx = sorted(keep[pid])
        return ' // '.join(segs[i] for i in idx) if idx else pid

    # Describing each row's own difference does NOT make the labels unique.
    # 120 ("... 12AVUV // 10VTV") and 160 ("... 12AVUV // 12VFMO // 10VTV") both
    # reduced to "10VTV", and the results matrix printed two rows under the same
    # name with $26M between them. Widen a colliding group until it separates.
    for _ in range(n + 1):
        groups = {}
        for pid in pids:
            groups.setdefault(render(pid), []).append(pid)
        clashing = [g for g in groups.values()
                    if len(g) > 1 and all(keep[p] is not None for p in g)]
        if not clashing:
            break
        widened = False
        for group in clashing:
            for i in range(n):
                vals = {tuple(parsed[p][0][i:i + 1]) for p in group}
                if len(vals) > 1 and any(i not in keep[p] for p in group):
                    for p in group:
                        if i < len(parsed[p][0]):
                            keep[p].add(i)
                    widened = True
                    break
        if not widened:
            break                    # genuinely identical templates; leave them
    return {pid: render(pid) for pid in pids}


def render_library(portfolios):
    """Grouped by tilt family, with bond-tier siblings collapsed onto one line —
    they are the same portfolio at different bond weights, i.e. the glide axis,
    not separate choices."""
    families = {}
    for pid, p in portfolios.items():
        d = compute_derived(p['allocation'])
        families.setdefault(d['tilt_family'], []).append((pid, d))

    out = []
    for fam in sorted(families):
        out.append(f"\n  ── {fam} " + "─" * max(2, 44 - len(fam)) + "  bonds  funds")
        zero, tiers = [], []
        for pid, d in sorted(families[fam]):
            (zero if d['bond_pct'] == 0 else tiers).append((pid, d))
        short = distinguish(portfolios, [pid for pid, _ in zero])
        for pid, d in zero:
            name = short[pid][:30]
            out.append(f"    {pid:<5} {name:<32} {d['bond_pct']:>5.0%} {d['n_funds']:>6d}")
        if tiers:
            ids = ' · '.join(pid for pid, _ in tiers)
            pcts = '/'.join(f"{d['bond_pct']:.0%}" for _, d in tiers)
            out.append(f"    {ids:<38} {pcts:>11}   {BOND_TIER_NOTE}")
    return '\n'.join(out)


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

def ask(prompt, default=''):
    try:
        got = input(f"{prompt} ").strip()
    except EOFError:
        return default
    return got or default


def ask_choice(question, options, default=1):
    """options: list of (key, label, detail). Returns the chosen key.

    The detail column is aligned to the widest LABEL rather than set three
    spaces after each one, so the explanations form a column instead of a ragged
    edge. Every menu in the tool goes through here, and the bond-sleeve one was
    the worst case: `bnd_only   BND 100%` next to `core   BND 60% · JPLD 25%`
    put two unrelated things at the same indent.

    A very wide label (the native-sleeve readout names every portfolio) would
    otherwise push the whole column off the terminal, so alignment is capped and
    anything longer simply keeps its single gap.
    """
    print(f"\n  {question}")
    widths = [len(l) for _, l, d in options if d]
    pad = min(max(widths, default=0), 34)
    for i, (_, label, detail) in enumerate(options, 1):
        mark = '>' if i == default else ' '
        tail = f"   {detail}" if detail else ''
        if detail and len(label) <= pad:
            tail = f"{' ' * (pad - len(label))}   {detail}"
        print(f"  {mark} {i}  {label}{tail}")
    while True:
        got = ask(f"\n  [{default}]", str(default))
        if got.isdigit() and 1 <= int(got) <= len(options):
            return options[int(got) - 1][0]
        print(f"  ? enter 1-{len(options)}")


def parse_money(text):
    """Accept money the way people write it: 40k, 1.5m, $40,000, 40000.

    Demanding raw digits is a small cruelty in a prompt that is otherwise all
    menus, and '5000000' is genuinely hard to check at a glance.
    """
    t = text.strip().lower().replace(',', '').replace('$', '').replace(' ', '')
    if not t:
        raise ValueError('no amount given')
    mult = 1.0
    if t.endswith('k'):
        mult, t = 1e3, t[:-1]
    elif t.endswith('m'):
        mult, t = 1e6, t[:-1]
    value = float(t) * mult
    if value < 0:
        raise ValueError('amount cannot be negative')
    return value


def fmt_money(value):
    if value >= 1e6:
        return f"${value / 1e6:,.1f}M".replace('.0M', 'M')
    return f"${value / 1e3:,.0f}k"


def ask_multi(question, options, default=1, note=None):
    """Like ask_choice but accepts several, comma-separated.

    The portfolio prompt has always taken '1,3'; the policy prompt only offered
    one-or-all, so comparing exactly two — the common case, e.g. guardrails
    against VPW — was not expressible.
    """
    print(f"\n  {question}")
    if note:
        print(f"    {note}")
    for i, (_, label, detail) in enumerate(options, 1):
        mark = '>' if i == default else ' '
        tail = f"   {detail}" if detail else ''
        print(f"  {mark} {i}  {label}{tail}")
    while True:
        got = ask(f"\n  [{default}]", str(default))
        picks = [x.strip() for x in got.split(',') if x.strip()]
        if picks and all(x.isdigit() and 1 <= int(x) <= len(options) for x in picks):
            seen, out = set(), []
            for x in picks:                      # dedupe, keep the order given
                key = options[int(x) - 1][0]
                if key not in seen:
                    seen.add(key)
                    out.append(key)
            return out
        print(f"  ? enter 1-{len(options)}, or several separated by commas")


def ask_portfolios(portfolios):
    picks = suggested_bases(portfolios)
    while True:
        print("\n  Starting portfolio?   [numbers, comma-separated]")
        for i, (pid, note) in enumerate(picks, 1):
            tag = f"● {note}" if note else ''
            print(f"    {i}  {pid:<5} {tag:<24} {describe(portfolios, pid)}")
        print(f"    a  show all {len(portfolios)}")
        got = ask("\n  [1]", "1")
        if got.lower() == 'a':
            print(render_library(portfolios))
            print("\n  Enter portfolio id(s), comma-separated:")
            ids = [x.strip() for x in ask("  >", "").split(',') if x.strip()]
            unknown = [x for x in ids if x not in portfolios]
            if ids and not unknown:
                return ids
            print(f"  ? unknown: {', '.join(unknown) if unknown else '(none given)'}")
            continue
        idx = [x.strip() for x in got.split(',') if x.strip()]
        if all(x.isdigit() and 1 <= int(x) <= len(picks) for x in idx) and idx:
            return [picks[int(x) - 1][0] for x in idx]
        # A PORTFOLIO ID IS A NUMBER TOO. The shortlist reads "1  141 / 2  140 /
        # 3  145", so typing 140 — the thing you actually want — is the obvious
        # move and was rejected with "enter numbers from the list". Observed on
        # a real run's FIRST keystroke. Accepted now, provided every entry is a
        # known id: mixing the two (`1,145`) stays an error, because `1` would
        # be ambiguous and guessing is worse than asking again.
        if idx and all(x in portfolios for x in idx):
            return idx
        bad = [x for x in idx if not (x in portfolios
                                      or (x.isdigit()
                                          and 1 <= int(x) <= len(picks)))]
        print(f"  ? {', '.join(bad) if bad else 'mixed'} — give either menu "
              f"numbers (1-{len(picks)}), or portfolio ids, or 'a'")


def parse_tiers(text):
    """'5,10,20' or '5 10 20' -> (0.05, 0.10, 0.20). Percentages, not fractions."""
    parts = [p for p in text.replace(',', ' ').split() if p]
    if not parts:
        raise ValueError("no bond levels given")
    out = []
    for part in parts:
        pct = float(part)
        if not 0 < pct < 100:
            raise ValueError(f"bond level must be between 0 and 100, got {pct:g}")
        out.append(round(pct / 100.0, 6))
    return tuple(sorted(set(out)))


def describe_sleeve(name):
    mix = BOND_SLEEVES[name]
    return ' · '.join(f"{t} {w:.0%}" for t, w in sorted(mix.items(), key=lambda kv: -kv[1]))


VPW_RESERVE_CHOICES = (0.0, 5e6, 10e6, 20e6)


def ask_gk_response(spec):
    """How fast Guyton-Klinger's spending follows a GROWING portfolio.

    The shipped rule is conservative by LAG rather than by design: a portfolio
    compounding ~6% real drifts 20% below the initial rate only every 3-4 years,
    so a +10% raise on that cadence grows spending ~2.5-3%/yr while the
    portfolio grows 6%. Over 30 years that is the whole reason guyton_klinger
    spends 0.61x of what VPW does.

    Only the RAISE side moves. Every option keeps upper=1.20 and cut=0.90, so
    the worst single-year cut stays capped at 10% — which is the reason to reach
    for this rather than for fixed_pct, whose 3% rule takes a 41.8% pay cut in
    the same crash test.
    """
    print("\n  How fast should spending follow a GROWING portfolio?")
    print("    Guyton-Klinger raises spending when the withdrawal RATE drifts")
    print("    below plan — i.e. when the portfolio has outgrown your spending.")
    print("    The original trigger is slow, so spending falls behind a fast-")
    print("    compounding portfolio and the estate absorbs the difference.")
    print()
    print("    The DOWNSIDE rule is identical in all three: cut 10% when the")
    print("    rate runs 20% above plan. You are choosing participation, not")
    print("    protection.")

    # ask_multi does not pad — the call site does, the way the sleeve prompt
    # does. With labels of unequal length an unpadded menu reads as ragged.
    order = ('slow', 'medium', 'fast')
    blurb = {
        'slow':   'raise 10% when the rate drifts 20% low   — the original rule',
        'medium': 'raise 15% when it drifts 10% low',
        'fast':   'raise 20% when it drifts  5% low   — tracks the portfolio (default)',
    }
    w = max(len(k) for k in order)
    options = [(k, f"{k:<{w}}", blurb[k]) for k in order]
    chosen = ask_multi("Spending response to growth?", options,
                       default=order.index(policies_mod.GK_RESPONSE_DEFAULT) + 1,
                       note='pick one, or several to compare')
    spec['gk_response'] = chosen[0]
    if len(chosen) > 1:
        spec['compare_gk_responses'] = chosen
    print(f"    → {', '.join(chosen)}")
    # Only where it applies: `slow` IS the original rule, so warning about a
    # faster ratchet there is noise the reader has to decide to ignore.
    if any(c != 'slow' for c in chosen):
        print("      A faster raise also means more to give back after a bad")
        print("      year: the per-year cut is still capped at 10%, but")
        print("      consecutive cuts are possible. Read 'worst yr' and the")
        print("      life-stage 'low' column.")


def ask_vpw_bequest(spec):
    """VPW's aggression is a dial, not a fixed setting.

    Plain VPW amortises the whole balance, which is why it out-spends
    guyton_klinger heavily while leaving far less behind. Reserving an amount
    moves it continuously toward a preservation rule.

    The reserve is a DIAL POSITION, not a promise about the end balance. VPW
    amortises to `policies.vpw_planning_age` (100, or five years past the
    plan), not to the simulation's horizon, so a balance normally remains at the horizon whatever the reserve
    is — the schedule still has years to run. An earlier version of this prompt
    described the reserve as "end with $5M at 95", which was wrong by the whole
    unamortised tail (~$6M on a typical run) and made the bequest column look
    broken. What actually lands is a RESULT, and the table already reports it.

    Offered as a menu because this is a preference rather than a number you look
    up. Multi-select is the genuinely useful case: it shows the whole
    spending-versus-estate frontier in one run.
    """
    cfg = default_config()
    end_age = int(cfg['age'] + cfg['time_horizon_years'])
    plan_age = int(policies_mod.vpw_planning_age(end_age))
    # Computed, not typed: the rates move with the planning age, which moves
    # with the plan once it runs past 95.
    rate = lambda age: policies_mod.vpw(age, 1.0, policies_mod.PolicyState(
        w_initial=0.0, v_at_retirement=1.0, initial_rate=0.0, w_prior_year=0.0,
        retirement_age=65.0, end_age=end_age, assumed_real_return=0.035,
        vpw_planning_age=plan_age))
    print("\n  How VPW works")
    print(f"    Each year it withdraws roughly (balance / years to {plan_age}), so")
    print(f"    the rate RISES with age: ~{rate(65):.0%} at 65, ~{rate(75):.0%} at 75, "
          f"~{rate(85):.0%} at 85.")
    print()
    print(f"    It plans to {plan_age}, not to {end_age}, deliberately — amortising to")
    print(f"    the horizon would demand a ~100% withdrawal at {end_age - 1}, which is an")
    print("    artefact of a fixed endpoint, not a policy anyone would follow.")
    print(f"    So there is normally a balance left at {end_age}: the schedule still")
    print("    has years to run.")
    print()
    print("    A reserve is held back from the amortisation ON TOP of that, so")
    print("    more reserve = less spending, larger estate. It is a dial, not a")
    print(f"    target — read the 'bequest' column for what actually lands at {end_age}.")

    options = [('0', 'No reserve', 'amortise the whole balance — most spending')]
    options += [(str(v), f"Reserve {fmt_money(v)}",
                 'held back from the amortisation, today\'s dollars')
                for v in VPW_RESERVE_CHOICES[1:]]
    options.append(('__other__', 'Other amount…', ''))

    # MULTI-select, matching the policy prompt. Seeing the frontier is the point
    # of the reserve, and a separate "compare these" option could only offer the
    # fixed set — picking 1,2,3 is both more flexible and one less concept.
    chosen = ask_multi("VPW: hold back a reserve?", options,
                       note='pick one, or several to see the frontier (e.g. 1,2,3)')

    reserves = []
    for c in chosen:
        if c == '__other__':
            while True:
                got = ask("    amount (e.g. 5m, 7.5m, 250k) >", "")
                if not got.strip():
                    break
                try:
                    reserves.append(parse_money(got))
                    break
                except ValueError as e:
                    print(f"      ? {e}")
        else:
            reserves.append(float(c))

    reserves = sorted(set(reserves))
    if len(reserves) > 1:
        spec['compare_vpw_bequests'] = reserves
        print("    \u2192 comparing reserves: "
              + ', '.join('none' if not v else fmt_money(v) for v in reserves))
    elif reserves and reserves[0] > 0:
        spec['vpw_bequest'] = reserves[0]
        print(f"    \u2192 holding back {fmt_money(reserves[0])}")


def effective_ss(spec):
    """The Social Security block a run will use: the spec's if it carries one
    (None = excluded for this run), otherwise your stored details."""
    if 'social_security' in spec:
        return spec['social_security']
    return investor.load().get('social_security')


def ss_line(ss):
    """'$55,800/yr from 70 (75% of scheduled)' — the benefit a run will pay."""
    benefit = policies_mod.ss_income(99, ss)
    scale = ss.get('benefit_scale', 1.0)
    return (f"{fmt_money(benefit)}/yr from {ss['claim_age']:g}"
            + (f" ({scale:.0%} of scheduled)" if scale < 1 else ""))


def ask_social_security(spec):
    """Only when nothing is stored — otherwise say what will be used and move on.

    Worth including for a spending FIGURE: it funds part of the need, so the
    portfolio covers less, and it puts a floor under spending that survives
    portfolio depletion.

    It used to be asked on EVERY run, with the same three numbers retyped each
    time. Stored in your details it is asked once; to vary it, edit it ('Your details')
    or sweep it (`ss_scale`, `ss_claim_age` in the sensitivity menu).
    """
    stored = investor.load().get('social_security')
    if stored:
        print(f"\n  Social Security  {ss_line(stored)}"
              f"   — from your details (change it there)")
        return
    # MONTHLY, because that is what an ssa.gov statement shows. Asking for an
    # annual figure invites a 12x error that looks entirely plausible: $5,000 is
    # a realistic monthly benefit and an unrealistic annual one, and nothing
    # downstream would flag it.
    print("\n  Social Security — your ssa.gov statement shows a MONTHLY benefit")
    print("  at full retirement age (67). Enter that figure in today's dollars.")
    got = ask("    monthly benefit at 67, e.g. 3200  (blank to exclude) >", "")
    if not got.strip():
        return
    try:
        monthly = parse_money(got)
    except ValueError:
        print("  ? not a number — excluding Social Security")
        return
    if monthly > 20_000:
        print(f"  ? {fmt_money(monthly)}/month looks like an ANNUAL figure.")
        if ask("    treat it as annual instead? [Y/n]", "y").lower().startswith('y'):
            monthly /= 12.0
    annual = monthly * 12.0
    print(f"    = {fmt_money(annual)}/yr")

    while True:
        claim = ask("    claim at age (62-70) [67] >", "67")
        try:
            claim_age = float(claim)
        except ValueError:
            print("      ? enter an age between 62 and 70")
            continue
        if 62 <= claim_age <= 70:
            break
        print("      ? claiming is only possible between 62 and 70")
    # SCHEDULED benefits are not PAYABLE benefits. The trustees project the
    # OASI trust fund depleting in the early-to-mid 2030s, after which continuing
    # payroll-tax income covers roughly 75-80% of scheduled benefits absent
    # legislation — well before this user would claim.
    haircut = ask("    assume what % of scheduled benefits? [100] >", "100")
    try:
        scale = max(0.0, min(float(haircut) / 100.0, 1.0))
    except ValueError:
        scale = 1.0
    ss = {'annual_at_fra': annual, 'claim_age': claim_age,
          'benefit_scale': scale}
    if ask("    save to your details, so runs stop asking? [Y/n]",
           "y").lower().startswith('y'):
        inv = investor.load()
        inv['social_security'] = ss
        try:
            investor.save(inv)
            print("    ✓ saved to config/investor.json — change it from 'Your details'")
            return
        except ValueError as e:
            print(f"    ✗ not saved ({e}) — using it for this run only")
    spec['social_security'] = ss      # a per-run override, not your details
    actual = policies_mod.ss_benefit(annual, claim_age) * scale
    note = '' if scale == 1.0 else f"  ({scale:.0%} of scheduled)"
    print(f"    → ${actual:,.0f}/yr from age {claim_age:g}{note}")
    if scale == 1.0:
        print("      note: 100% assumes current law is funded in full; the")
        print("      trustees project ~75-80% payable from the mid-2030s.")


def ask_sleeve(spec, allow_compare=False):
    """Which bond funds to use. Asked on every path that can add bonds — the
    custom-schedule path used to skip it, so a sleeve could only be chosen by
    knowing to type '140+10:bnd_only' by hand."""
    # Name WHICH portfolio maps to WHICH sleeve, not just the set of sleeves.
    # This read `native (bnd_only, core, vanguard)` — a bare alphabetical list
    # sitting directly above selectable options with those same names, so it
    # looked like a fourth set of choices rather than a readout of what `native`
    # resolves to. And the sort put `bnd_only` first when only one portfolio
    # used it. You could not tell which was which without reading NATIVE_SLEEVE.
    by_sleeve = {}
    for b in spec['portfolios']:
        by_sleeve.setdefault(native_sleeve(b), []).append(str(b))
    detail = ' · '.join(f"{', '.join(ids)} → {sl}"
                        for sl, ids in sorted(by_sleeve.items()))
    options = [('__native__', 'native — each portfolio keeps its own', detail)]
    options += [(n, n, describe_sleeve(n)) for n in sorted(BOND_SLEEVES)]
    if allow_compare:
        options.append(('__all__', 'Compare all sleeves',
                        'same equity and bond %, different funds'))
    sleeve = ask_choice('Which bond sleeve?', options)
    if sleeve == '__all__':
        spec['compare_sleeves'] = sorted(BOND_SLEEVES)
    elif sleeve != '__native__':
        spec['bond_sleeve'] = sleeve


def ask_bond_ladder(spec):
    """Bond levels and sleeve. Both default, so this is one keypress to skip."""
    default = ' / '.join(f"{t * 100:g}" for t in glide.DERISK_TIERS)
    print(f"\n  Bond levels to test?   [percentages, comma-separated]")
    print(f"    default: {default}")
    while True:
        got = ask(f"\n  [{default.replace(' / ', ',')}]", '')
        if not got:
            break
        try:
            spec['bond_tiers'] = list(parse_tiers(got))
            break
        except ValueError as e:
            print(f"  ? {e}")

    ask_sleeve(spec, allow_compare=True)


def ask_custom_schedule(portfolios, start_age, default_base='141', sleeve=None):
    print("\n  Custom schedule — you'll be asked for:")
    print("    1. the portfolio you start with    e.g. 140")
    print("    2. each change, as 'id at age'     e.g. 140+10 at 65")
    print("       …as many as you like; blank line to finish")
    print()
    print("  Ids can carry a bond level:  140+20 · 141+10:bnd_only")
    print("  …or paste a whole schedule:  166>166+5@65>166@70")
    print("  Tip: bonds high at 65, falling after, has the best support.")

    while True:
        base = ask(f"    start portfolio [{default_base}] >", default_base).strip()
        # PASTE A WHOLE SCHEDULE. This line used to REJECT '>' outright — while
        # every strategy name the report prints, and the --spec line it now
        # emits to replay a run, are in exactly that form. Output you cannot
        # feed back as input is a trap: the obvious move after reading
        # `166>166+5@65>166@70` in a results table is to paste it.
        if '>' in base or '@' in base:
            try:
                sch = glide.validate(glide.parse_schedule(base, start_age),
                                     portfolios)
                print(f"\n{glide.format_chart(sch, portfolios, start_age + 50)}")
                return base
            except ValueError as e:
                print(f"      ? {e}")
                continue
        if ' at ' in base or ',' in base:
            print("      ? this line takes ONLY the starting id (e.g. 140),")
            print("        or a whole schedule like 166>166+5@65.")
            continue
        try:
            glide.resolve_ref(base, portfolios)
            break
        except ValueError as e:
            print(f"      ? {e}")
    parts = [base]
    while True:
        got = ask("    add change, or Enter to finish >", "")
        if not got:
            break
        bits = got.replace(' at ', ' ').split()
        if len(bits) != 2:
            print("      ? format: <id> at <age>   (id may be '141' or '141+10')")
            continue
        try:
            glide.resolve_ref(bits[0], portfolios)   # accepts synthesised refs
        except ValueError as e:
            print(f"      ? {e}")
            continue
        ref = bits[0]
        if sleeve and '+' in ref and ':' not in ref:
            ref = f"{ref}:{sleeve}"          # apply the chosen sleeve by default
        parts.append(f"{ref}@{bits[1]}")
    spec = '>'.join(parts)
    try:
        sch = glide.validate(glide.parse_schedule(spec, start_age), portfolios)
    except ValueError as e:
        print(f"    ✗ {e}")
        return None
    print(f"\n{glide.format_chart(sch, portfolios, start_age + 50)}")
    return spec


def _offer_same_shape(spec, bases, built, portfolios, start_age):
    """Offer `spec`'s shape on the other chosen portfolios, appending to `built`.

    The same sleeve on three portfolios used to take three passes through
    ask_custom_schedule. A schedule that moves between portfolios has no single
    shape, so it gets no offer.
    """
    try:
        copies = {b: glide.same_shape(spec, b) for b in bases}
    except ValueError:
        return
    copies = {b: c for b, c in copies.items() if c not in built}
    if not copies:
        return
    if not ask(f"\n    Same shape for {', '.join(copies)}? [Y/n]",
               "y").lower().startswith('y'):
        return
    added = []
    for b, c in copies.items():
        try:
            glide.validate(glide.parse_schedule(c, start_age), portfolios)
        except ValueError as e:
            print(f"    ✗ {b}: {e}")
            continue
        built.append(c)
        added.append(c)
    if added:
        print(f"    ✓ added: {', '.join(added)}")


def ask_custom_schedules(portfolios, start_age, bases, sleeve=None):
    """Build SEVERAL custom schedules, so alternative shapes compare in one run.

    Comparing 'derisk at 65' against a staged 45/55/65 path is the whole point of
    custom schedules; building one at a time meant two runs and a manual diff.
    The bare base portfolios are added as controls unless already present — a
    comparison of two glide shapes with nothing to measure them against is hard
    to read.
    """
    built = []
    while True:
        spec = ask_custom_schedule(portfolios, start_age, bases[0], sleeve)
        if spec in built:
            print(f"\n    · already comparing {spec}")
        elif spec:
            built.append(spec)
            print(f"\n    ✓ added: {spec}")
            _offer_same_shape(spec, bases, built, portfolios, start_age)
        if not ask("\n    Add another schedule? [y/N]", "n").lower().startswith('y'):
            break

    # Controls, so the shapes have a baseline — in the order typed. Inserting
    # each at 0 reversed them, and the first strategy is the head-to-head base.
    built = [b for b in bases if b not in built] + built
    print(f"\n    comparing {len(built)}: {', '.join(built)}")
    return built


# ---------------------------------------------------------------------------
# Your details
# ---------------------------------------------------------------------------

def _money(text, current):
    """Accepts 300000, 300,000, $300k, 0.5m. Blank keeps the current value."""
    return parse_money(text) if text.strip() else current


def _rate(text, current):
    """Accepts 2.9, 2.9%, or 0.029 — all meaning 2.9%/yr.

    A '%' always means percent, so '0.5%' is 0.5%/yr — it was read as the
    decimal 0.5, 50%/yr, which validation's +/-50% band let through. A bare
    number from 1 up is a percentage too: a growth rate never approaches
    100%/yr, so "1" meaning 100% is not a case worth preserving.
    """
    t = text.strip()
    if not t:
        return current
    v = float(t.replace('%', ''))
    # Rounded: 2.9/100 is 0.028999999999999998, which is what landed in the
    # saved JSON and reads as noise next to a hand-entered 0.03.
    return round(v / 100.0, 6) if '%' in t or abs(v) >= 1 else round(v, 6)


def _whole(text, current):
    t = text.strip()
    if not t:
        return current
    if t.lower() in ('none', 'never', '-'):
        return None
    return int(t)


def _ask_field(label, current, parse, show=None):
    """Prompt until it parses. Blank keeps what is there."""
    shown = show(current) if show else current
    while True:
        got = ask(f"      {label} [{shown}]:")
        try:
            return parse(got, current)
        except (ValueError, TypeError):
            print(f"      ? could not read {got!r}")


def _new_account(inv):
    """A blank account that stops contributing the year before retirement."""
    return {'tax': 'taxable', 'balance': 0.0, 'annual_amount': 0.0,
            'growth_rate': 0.03, 'stop_age': inv['retirement_age'] - 1}


def _edit_account(inv, name):
    a = inv['accounts'][name]
    print(f"\n    {name}")
    a['balance'] = _ask_field('balance', a['balance'], _money,
                              lambda v: f"${v:,.0f}")
    a['annual_amount'] = _ask_field('contribution per year', a['annual_amount'],
                                    _money, lambda v: f"${v:,.0f}")
    a['growth_rate'] = _ask_field('contribution growth %/yr', a['growth_rate'],
                                  _rate, lambda v: f"{v * 100:.1f}%")
    # INCLUSIVE: the engine contributes while age <= stop_age. "Until 65" read
    # as stopping AT 65 and contributed through the first year of retirement.
    a['stop_age'] = _ask_field('last age you contribute (none = never stops)',
                               a['stop_age'], _whole)
    while True:
        got = ask(f"      tax treatment "
                  f"({'/'.join(investor.TAX_TREATMENTS)}) [{a['tax']}]:")
        got = got.strip() or a['tax']
        if got in investor.TAX_TREATMENTS:
            a['tax'] = got
            break
        print(f"      ? must be one of {', '.join(investor.TAX_TREATMENTS)}")
    # Confirming the numbers is exactly what the mark was waiting for.
    a.pop('unconfirmed', None)


def _monthly_benefit(text, current):
    """A MONTHLY benefit, as the ssa.gov statement shows it. 0 removes it."""
    v = _money(text, current)
    if v > 20_000:
        print("      ? that looks like an ANNUAL figure — enter the monthly one")
        raise ValueError(text)
    return v


def _claim_age(text, current):
    t = text.strip()
    if not t:
        return current
    v = float(t)
    if not 62 <= v <= 70:
        print("      ? claiming is only possible between 62 and 70")
        raise ValueError(text)
    return v


def _scheduled_pct(text, current):
    """Accepts 75 or 75% or 0.75 — all meaning 75% of scheduled benefits."""
    t = text.strip().replace('%', '')
    if not t:
        return current
    v = float(t)
    v = v if v > 1 else v * 100.0
    if not 0 <= v <= 100:
        raise ValueError(text)
    return v


def _edit_social_security(inv):
    ss = inv.get('social_security') or {}
    print("\n    social security — MONTHLY benefit at 67 from your ssa.gov "
          "statement, today's dollars")
    monthly = _ask_field('monthly benefit at 67 (0 = none)',
                         ss.get('annual_at_fra', 0.0) / 12.0, _monthly_benefit,
                         lambda v: f"${v:,.0f}")
    if monthly <= 0:
        inv['social_security'] = None
        print("      removed — runs will ask again")
        return
    claim = _ask_field('claim at age (62-70)', ss.get('claim_age', 67.0),
                       _claim_age, lambda v: f"{v:g}")
    # SCHEDULED is not PAYABLE: the trustees project the OASI fund depleting in
    # the 2030s, after which payroll tax covers roughly 75-80% of scheduled.
    pct = _ask_field('% of scheduled benefits you assume is paid',
                     ss.get('benefit_scale', 1.0) * 100.0, _scheduled_pct,
                     lambda v: f"{v:g}%")
    inv['social_security'] = {'annual_at_fra': round(monthly * 12.0, 2),
                              'claim_age': float(claim),
                              'benefit_scale': round(pct / 100.0, 4)}


def _investor_screen(inv):
    """Print the record and return the account names in display order."""
    names = list(inv['accounts'])
    horizon = inv['plan_to_age'] - inv['age']
    print(f"\n  YOUR DETAILS{' ' * 36}config/investor.json\n")
    print(f"    1  age                             {inv['age']}")
    print(f"    2  plan to age                     {inv['plan_to_age']}"
          f"    horizon {horizon} years")
    print(f"    3  retirement age                  {inv['retirement_age']}")
    print(f"    4  contributions current as of age "
          f"{inv['contributions_measured_at_age']}")
    print(f"\n       {'account':<12} {'balance':>12} {'contribution':>14} "
          f"{'growth':>8} {'until':>7}   tax")
    bal = con = 0.0
    over = investor.overlaps(inv)
    for i, name in enumerate(names, start=5):
        a = inv['accounts'][name]
        bal += a['balance']
        con += a['annual_amount']
        mark = '⚠' if a.get('unconfirmed') else ' '
        until = 'never' if a.get('stop_age') is None else f"{a['stop_age']}"
        until = ('↻ ' if name in over else '') + until
        print(f"  {i:>3}  {name:<12} {mark} {a['balance']:>10,.0f} "
              f"{a['annual_amount']:>11,.0f}/yr {a['growth_rate'] * 100:>7.1f}% "
              f"{until:>7}   {a['tax']}")
    print(f"       {'':<12}   {'-' * 42}")
    print(f"       {'total':<12}   {bal:>10,.0f} {con:>11,.0f}/yr")
    print(f"       'until' is the LAST age each account contributes.")
    if over:
        print(f"       ↻ still contributing into retirement (withdrawals start at "
              f"{inv['retirement_age']}).")
    ss = inv.get('social_security')
    row = 5 + len(names)
    if ss:
        print(f"\n  {row:>3}  social security   ${ss['annual_at_fra'] / 12:,.0f}/mo "
              f"at 67 · claim at {ss['claim_age']:g} · "
              f"{ss.get('benefit_scale', 1.0):.0%} of scheduled"
              f"   → {ss_line(ss)}")
    else:
        print(f"\n  {row:>3}  social security   not set — every run will ask")
    flagged = investor.unconfirmed(inv)
    if flagged:
        print(f"\n    ⚠ {len(flagged)} seeded with a guess, not a fact you gave: "
              f"{', '.join(flagged)}.")
        print(f"      Harmless today — the engine sums the accounts — but they")
        print(f"      become load-bearing when taxes and RMDs land. Editing one")
        print(f"      clears its mark.")
    return names


def _ask_menu(prompt, default):
    """Like ask(), but end-of-input means QUIT rather than "take the default".

    ask() folds EOF into the default, and in a loop that re-prompts after a
    refused save that is an infinite loop: stdin ends, the default 's' comes
    back, save rejects the invalid record, round it goes forever. Found by
    piping a plan_to_age of 30 at it. Ctrl-D has to end the editor.
    """
    try:
        return input(f"{prompt} ").strip() or default
    except EOFError:
        return 'q'


def _edit_row(inv, n, names):
    """Edit one numbered row of the details screen."""
    if n == 1:
        was = inv['age']
        inv['age'] = _ask_field('age', was, _whole)
        # The trap this whole file exists to close: contribution amounts
        # compound from `contributions_measured_at_age`, so entering current
        # amounts alongside a new age without moving that field gives them a
        # year of growth they already contain. Ask at the moment it bites.
        if inv['age'] != was:
            cur = ask(f"      are the contribution amounts below current as "
                      f"of age {inv['age']}? [Y/n]", 'y')
            if not cur.lower().startswith('n'):
                inv['contributions_measured_at_age'] = inv['age']
                print(f"      ✓ contributions now measured at age {inv['age']}")
    elif n == 2:
        inv['plan_to_age'] = _ask_field('plan to age', inv['plan_to_age'], _whole)
    elif n == 3:
        inv['retirement_age'] = _ask_field('retirement age',
                                           inv['retirement_age'], _whole)
    elif n == 4:
        inv['contributions_measured_at_age'] = _ask_field(
            'contributions current as of age',
            inv['contributions_measured_at_age'], _whole)
    elif 5 <= n < 5 + len(names):
        _edit_account(inv, names[n - 5])
    elif n == 5 + len(names):
        _edit_social_security(inv)
    else:
        print(f"      ? no row {n}")


def edit_investor():
    """Menu option: edit config/investor.json in place.

    Nothing is written until `s`, and nothing invalid is written at all —
    `investor.save` refuses a record that fails validation rather than let a
    silently wrong config produce confidently wrong runs.
    """
    inv = investor.load()
    while True:
        names = _investor_screen(inv)
        got = _ask_menu("\n  Edit which?  [number · a add · x N remove · s save · "
                        "q cancel]", 's').lower()

        if got in ('q', 'quit', 'cancel'):
            print("    (unchanged)")
            return
        if got in ('s', 'save', ''):
            try:
                investor.save(inv)
            except ValueError as e:
                print(f"\n    ✗ not saved — {e}")
                continue
            shown = investor.FILE
            try:
                shown = shown.relative_to(Path(__file__).resolve().parent)
            except ValueError:
                pass
            print(f"    ✓ saved → {shown}")
            return
        if got == 'a':
            name = ask("      account name (e.g. 457b):").strip()
            if not name:
                continue
            if name in inv['accounts']:
                print(f"      ? {name} already exists — edit it by number")
                continue
            inv['accounts'][name] = _new_account(inv)
            _edit_account(inv, name)
            continue
        if got.startswith('x'):
            idx = got[1:].strip()
            if idx.isdigit() and 5 <= int(idx) < 5 + len(names):
                gone = names[int(idx) - 5]
                if len(names) == 1:
                    print("      ? at least one account is required")
                    continue
                del inv['accounts'][gone]
                print(f"      removed {gone}")
            else:
                print("      ? use x followed by an account's number, e.g. x6")
            continue

        # Every other prompt in run.py takes a comma-separated list, so this one
        # gets tried that way too — "4,5,6,7,8,9", to walk the whole record in
        # one pass, was the first thing typed at it and it was rejected.
        rows = [t.strip() for t in got.split(',') if t.strip()]
        if not rows or not all(t.isdigit() for t in rows):
            print(f"      ? did not understand {got!r}")
            continue
        for t in rows:
            _edit_row(inv, int(t), names)
        problems = investor.validate(inv)
        if problems:
            print("\n    ⚠ not valid yet: " + "; ".join(problems))


def prompt_sensitivity():
    """Pick an assumption you do not control to sweep. Returns a spec, or None
    to go back.

    Two menus rather than one flat list: the grouping is by WHO CONTROLS THE
    ASSUMPTION. Returns, cash, inflation, confidence and your life are not
    yours to decide. `Your plan` is left out: those are choices, and the
    Retirement choices route compares them on shared paths. The CLI still
    sweeps them.

    The registry lives in tools/sensitivity.py so the CLI and this path cannot
    drift — adding a dimension there adds it here.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'tools'))
    import sensitivity as sens

    groups = [(g, title, blurb) for g, title, blurb in sens.GROUPS
              if g != 'plan'
              and any(d.group == g for d in sens.DIMENSIONS.values())]
    gsel = ask_choice('Which kind of assumption?',
                      groups + [('back', 'back', '')])
    if gsel == 'back':
        return None

    dims = [(n, n, d.note) for n, d in sens.DIMENSIONS.items() if d.group == gsel]
    dsel = ask_choice('Sweep what?', dims + [('back', 'back', '')])
    if dsel == 'back':
        return None

    d = sens.DIMENSIONS[dsel]
    vals = ask(f"Values (comma-separated) [{d.default}]", d.default) or d.default
    return {'mode': 'sensitivity', 'dimension': dsel, 'values': vals}


def _menu_options(last, cfg):
    """The top menu: three questions, named by what CHANGES in each.

    Portfolios: the portfolios change, your plan is held fixed. Retirement
    choices: your plan changes, one portfolio is held fixed. Market & life
    assumptions: what you do not control. The hints carry YOUR numbers (the
    portfolios you last compared, your retirement age) so each entry says what
    it would do before you pick it. `config` sits last, after the run options.
    """
    ra = cfg['retirement_age']
    ports = (last or {}).get('portfolios') or []
    eg = ' vs '.join(ports[:3]) if len(ports) > 1 else '161 vs 165'
    claim = ', claim 62/67/70' if cfg.get('social_security') else ''
    opts = [('portfolios', 'Portfolios',
             f"{eg}, or a bond sleeve, on your plan"),
            ('choices', 'Retirement choices',
             f"retire {ra - 5:g}/{ra:g}/{ra + 5:g}{claim}, savings, withdrawal rule"),
            ('sensitivity', 'Market & life assumptions',
             'returns, cash, inflation, lifespan, care costs')]
    if last:
        opts.append(('again', 'Re-run last', describe_preset(last)))
    opts.append(('config', 'Your details', investor.summary(investor.load())))
    return opts


def describe_preset(spec):
    """One line naming what a saved spec compares."""
    ports = ', '.join(spec.get('portfolios', []))
    ch = spec.get('compare_choice')
    if ch:
        return (', '.join(choice_tag(ch['dimension'], v) for v in ch['values'])
                + f" on {ports}")
    if spec.get('compare_policies'):
        return f"{', '.join(spec['compare_policies'])} on {ports}"
    return ports + (' (accumulation)' if spec.get('mode') == 'accumulation' else '')


def prompt_run(portfolios):
    last = load_preset('last')

    # Editing and backing out of a sub-menu loop back rather than exiting,
    # because you open them to check a number as often as to change one.
    while True:
        mode = ask_choice('What do you want to compare?',
                          _menu_options(last, default_config()))
        if mode in ('sensitivity', 'choices'):
            # None when the user backs out, which loops rather than exiting.
            picked = (prompt_sensitivity() if mode == 'sensitivity'
                      else prompt_choices(portfolios))
            if picked is not None:
                return picked
            continue
        if mode != 'config':
            break
        edit_investor()

    cfg = default_config()
    if mode == 'again':
        return last

    # Accumulation-only was a top-level entry; it compares portfolios too, so
    # it is a follow-up here.
    mode = ask_choice('Compare them over',
                      [('retirement', 'your retirement',
                        'what each lets you spend, the usual question'),
                       ('accumulation', 'accumulation only',
                        'wealth at retirement, no withdrawals')])
    spec = {'mode': mode, 'portfolios': ask_portfolios(portfolios)}

    derisk = ask_choice(
        f"Derisk at retirement ({cfg['retirement_age']})?",
        [('yes', 'yes', 'compare each available bond tier'),
         ('no', 'no', 'hold each portfolio throughout'),
         ('custom', 'custom schedule…', '')])
    if derisk == 'yes':
        spec['derisk_at'] = cfg['retirement_age']
        ask_bond_ladder(spec)
    elif derisk == 'custom':
        ask_sleeve(spec)
        spec['custom'] = ask_custom_schedules(portfolios, cfg['age'],
                                              spec['portfolios'],
                                              spec.get('bond_sleeve'))

    if mode == 'retirement':
        how = ask_choice('Spending?',
                         [('solve', 'find the maximum I can sustain', '90% confidence'),
                          ('fixed', 'test a specific amount…', '')])
        if how == 'solve':
            spec['spend'] = 'solve'
        else:
            while True:
                try:
                    spec['spend'] = parse_money(
                        ask("    real $/year (e.g. 150k) [200k]", "200k"))
                    break
                except ValueError as e:
                    print(f"      ? {e}")
        ask_social_security(spec)
        ask_policies(spec)
        ask_paths(spec)
        ask_return_views(spec, len(resolve_strategies(spec, portfolios)[0]))
    else:
        spec['sims'] = int(ask("\n  Simulations per portfolio [2000]", "2000"))
    return spec


POLICY_OPTIONS = [
    ('guyton_klinger', 'Guyton-Klinger guardrails',
     'steady, cut 10% if the rate drifts 20% high'),
    ('constant_real', 'Constant real',
     'same amount every year; the only one that can deplete'),
    ('vpw', 'VPW',
     'rate rises with age; spends down deliberately, smaller bequest'),
    ('fixed_pct', 'Fixed percentage',
     'fixed % of balance; a 35% crash is a 35% pay cut'),
]


def ask_policies(spec, default=1, several=True):
    """The withdrawal policy, or several to compare, and their settings."""
    question = 'Spending policy?  (who absorbs a bad market — you, or the portfolio)'
    if several:
        chosen = ask_multi(question, POLICY_OPTIONS, default=default,
                           note='pick one, or several to compare (e.g. 1,3)')
    else:
        chosen = [ask_choice(question, POLICY_OPTIONS)]
    spec['policy'] = chosen[0]
    if len(chosen) > 1:
        spec['compare_policies'] = chosen
    if 'vpw' in chosen:
        ask_vpw_bequest(spec)
    if 'guyton_klinger' in chosen:
        ask_gk_response(spec)
    return spec


def choice_defaults(cfg, dim):
    """The values offered for a plan choice, built around yours."""
    if dim == 'retirement_age':
        ra, end = cfg['retirement_age'], cfg['age'] + cfg['time_horizon_years']
        return [float(v) for v in (ra - 5, ra, ra + 5) if cfg['age'] < v < end]
    if dim == 'ss_claim_age':
        mine = (cfg.get('social_security') or {}).get('claim_age')
        return sorted({62.0, 67.0, 70.0} | ({float(mine)} if mine else set()))
    return [0.5, 1.0, 1.5]


def prompt_choices(portfolios):
    """RETIREMENT CHOICES: one portfolio, several plans, the same paths.

    The question is what a choice costs or buys, so every option is solved:
    here the plan level IS the answer, not a number to look past. Returns a
    spec, or None to go back.
    """
    cfg = default_config()
    ss = cfg.get('social_security')
    opts = [('retirement_age', 'Retirement age',
             f"yours is {cfg['retirement_age']:g}")]
    if ss:
        opts.append(('ss_claim_age', 'Social Security claiming age',
                     f"yours is {ss['claim_age']:g}"))
    opts += [('contributions', 'Savings', "today's contributions x0.5, x1, x1.5"),
             ('policy', 'Withdrawal rule', 'guardrails, constant real, VPW, fixed %'),
             ('back', 'back', '')]
    dim = ask_choice('Which choice?', opts)
    if dim == 'back':
        return None

    ids = ask_portfolios(portfolios)
    if len(ids) > 1:
        print(f"  a plan choice is compared on ONE portfolio — using {ids[0]}")
    spec = {'mode': 'retirement', 'portfolios': ids[:1], 'spend': 'solve'}
    if dim == 'policy':
        ask_policies(spec, default='1,2,3,4')
    else:
        default = ','.join(f"{v:g}" for v in choice_defaults(cfg, dim))
        while True:
            raw = ask(f"\n  {PLAN_CHOICES[dim][0]}: values to compare "
                      f"[{default}]", default) or default
            try:
                vals = [float(x) for x in raw.split(',') if x.strip()]
                _validate_choice(dict(spec, compare_choice={
                    'dimension': dim, 'values': vals}))
                break
            except ValueError as e:
                print(f"  ? {e}")
        spec['compare_choice'] = {'dimension': dim, 'values': vals}
        ask_policies(spec, several=False)
    ask_social_security(spec)
    ask_paths(spec)
    return spec


def ask_return_views(spec, n_strategies):
    """Whether to check the winner under other return models.

    Asked rather than always on: it evaluates every strategy once more per
    view, a sizeable share of the run. The default is yes, because whether the
    winner survives a wrong model is the question a portfolio comparison is
    for. Not offered when there is nothing to rank, or for a policy comparison.
    """
    probe = dict(spec, return_views=True)
    if not checks_return_views(n_strategies, probe):
        return spec
    from montecarlo import return_views as rv
    mins = return_view_minutes(n_strategies, probe)
    print(f"\n  Check the winner under {len(rv.VIEWS) - 1} other return models?")
    for v in rv.VIEWS[1:]:
        print(f"    {v.title:<12}{v.blurb}")
    print(f"    adds ~{max(mins, 1):.0f} min")
    if ask("\n  [Y/n]", "y").lower().startswith('y'):
        spec['return_views'] = True
    return spec


def ask_paths(spec):
    """How many paths. Retirement mode never asked, and it is the one knob that
    decides whether a result is believable.

    A plan level is a poorly-estimated percentile: this repo's own record has a
    600-path sweep reporting a ranking reversal that VANISHED at 1,600, and a
    verification count that moved portfolio 102's level 30.5k -> 17.0k between
    400 and 3,200 paths. Anyone who reads `not resolved` and wants to know
    whether more paths would settle it had to edit source or learn the --spec
    syntax.

    Only the VERIFICATION count is asked for; search paths ride at the SHIPPED
    RATIO (400/1500), so the solver keeps the balance it was tuned at and the
    time estimate stays proportional to the number typed. Scaling search to
    HALF the verification count instead — which is what this did first — made
    2x the paths cost 3.2x the wall clock, because search runs ~15 evaluations
    against verification's 1-3 and so dominates the total.
    """
    default = solver_mod.VERIFY_SIMS
    print(f"\n  How many paths?   more paths = tighter answer, linear cost")
    print(f"    {default:>5}   the default — fine for 'which is better', "
          f"~1 min per strategy")
    print(f"    {default * 2:>5}   when a head-to-head came back NOT RESOLVED "
          f"and you want it settled")
    print(f"    {default * 4:>5}   before quoting a number, or when two "
          f"strategies are within a few percent")
    raw = ask(f"\n  [{default}]", str(default))
    try:
        n = int(float(str(raw).replace(',', '').replace('k', '000')))
    except ValueError:
        n = default
    n = max(100, min(n, 50_000))
    if n != default:
        spec['verify_sims'] = n
        ratio = solver_mod.SEARCH_SIMS / solver_mod.VERIFY_SIMS
        spec['search_sims'] = max(100, int(round(n * ratio)))
    return spec


def confirm(spec, specs, portfolios):
    cfg = default_config()
    mins = estimate_minutes(len(specs), spec)
    print("\n  " + "─" * 70)
    # A RANGE, because the underlying measurement cannot support a point.
    # Three identical 12-strategy runs measured 3.63, 4.77 and 5.14 ms/path —
    # a 1.4x spread from machine load alone — so "about 3 min" claims a
    # precision that does not exist, and reads as broken when the run takes 1.4.
    est = ("under a minute" if mins < 0.8
           else f"roughly {max(mins * 0.7, 0.5):.0f}-{mins * 1.4:.0f} min")
    print(f"   {len(specs)} strateg{'y' if len(specs) == 1 else 'ies'} · "
          f"est. {est} · real 2026 dollars")
    print()
    labels = strategy_labels(specs)
    for gspec in specs:
        sch = glide.parse_schedule(gspec, cfg['age'])
        chain = ' → '.join(ref_split(w.portfolio_id, portfolios) for w in sch)
        tag = f"{labels[gspec]}  " if labels else ''
        print(f"     {tag}{gspec:<34} {chain}")
    if spec['mode'] == 'retirement':
        runs = policy_runs(spec)
        pol = (f"{len(runs)} policy runs" if len(runs) > 1
               else spec.get('policy', 'guyton_klinger'))
        if spec.get('compare_choice'):
            ch = spec['compare_choice']
            base = choice_baseline(cfg, ch['dimension'], ch['values'])
            print(f"   {PLAN_CHOICES[ch['dimension']][0]}: "
                  + ', '.join(f"{v:g}" + (' (yours)' if v == base else '')
                              for v in ch['values']))
        if spec.get('compare_vpw_bequests'):
            print(f"   VPW reserves: "
                  + ', '.join('none' if not v else fmt_money(v)
                              for v in spec['compare_vpw_bequests']))
        elif spec.get('vpw_bequest'):
            print(f"   VPW holds back {fmt_money(spec['vpw_bequest'])} "
                  f"from the amortisation")
        ss = effective_ss(spec)
        if ss:
            print(f"   Social Security {ss_line(ss)}")
        if spec.get('derisk_at') is not None:
            tiers = spec.get('bond_tiers') or glide.DERISK_TIERS
            sl = (f"{len(spec['compare_sleeves'])} sleeves"
                  if spec.get('compare_sleeves')
                  else (spec.get('bond_sleeve') or 'native'))
            print(f"   bond levels {', '.join(f'{t*100:g}%' for t in tiers)}"
                  f" · sleeve {sl}")
        if checks_return_views(len(specs), spec):
            from montecarlo import return_views as rv
            print(f"   then checked under {len(rv.VIEWS) - 1} other return models")
        print(f"\n   {pol} · withdrawals from "
              f"{cfg['retirement_age']} · "
              + ("solving for the maximum" if spec.get('spend') == 'solve'
                 else f"testing ${float(spec['spend']):,.0f}/yr"))
    if len(specs) > 8:
        print(f"\n   ⚠ {len(specs)} strategies is a lot to read; consider fewer.")
    print("  " + "─" * 70)
    return ask("\n  Run? [Y/n]", "y").lower().startswith('y')


def parse_spec(text, portfolios, sims=None):
    """A run spec from JSON on the command line, or a path to a .json file.

    The interactive prompts are the only way in, which makes every measurement
    anyone takes here unreproducible: a finding quoted in CLAUDE.md cannot be
    re-run without a human re-answering nine questions the same way. This is the
    same spec `--again` replays and `persistence` records, so a run from here is
    a first-class run, not a side door with its own semantics.

    VALIDATED HERE, not at the first use. A typo'd portfolio id would otherwise
    surface minutes into a solve, or — worse — silently resolve to fewer
    strategies than asked for.
    """
    text = text.strip()
    # Anything opening a JSON value is JSON; only otherwise is it a path. Keying
    # on '{' alone sent a JSON ARRAY down the file branch, so the error a typo'd
    # spec produced was "no such file: [1,2,3]".
    if not text[:1] in ('{', '['):
        path = Path(text)
        if not path.exists():
            raise ValueError(f"no such file: {text} (and it is not JSON)")
        text = path.read_text()
    try:
        spec = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not valid JSON — {exc}") from exc
    if not isinstance(spec, dict):
        raise ValueError(f"expected an object, got {type(spec).__name__}")

    spec.setdefault('mode', 'retirement')
    if spec['mode'] not in ('retirement', 'accumulation'):
        raise ValueError(f"mode must be retirement or accumulation, "
                         f"got {spec['mode']!r}")

    ids = spec.get('portfolios')
    if not ids:
        raise ValueError("give at least one portfolio, e.g. "
                         '{"portfolios": ["140", "152"]}')
    if isinstance(ids, str):
        ids = [x.strip() for x in ids.split(',') if x.strip()]
    ids = [str(x) for x in ids]
    unknown = [x for x in ids if x not in portfolios]
    if unknown:
        archived = sorted(set(unknown) & loaders.archived_portfolio_ids())
        note = (f" ({', '.join(archived)} archived: move the row back from "
                f"{loaders.ARCHIVE_CSV} into portfolio_templates.csv)" if archived else '')
        raise ValueError(f"unknown portfolio(s): {', '.join(unknown)}{note}")
    spec['portfolios'] = ids

    # Every custom schedule too: a typo'd sleeve or a backwards age used to pass
    # --dry-run and fail minutes into the solve.
    custom = spec.get('custom')
    if custom is not None:
        if isinstance(custom, str) or not all(isinstance(g, str) for g in custom):
            raise ValueError("custom must be a list of glide specs, e.g. "
                             '["168", "168>168+5@60>168@65"]')
        age = default_config()['age']
        for gspec in custom:
            try:
                glide.validate(glide.parse_schedule(gspec, age), portfolios)
            except ValueError as exc:
                raise ValueError(f"custom schedule {gspec!r}: {exc}") from None

    if spec['mode'] == 'retirement':
        spec.setdefault('spend', 'solve')
        spec.setdefault('policy', 'guyton_klinger')
        pols = spec.get('compare_policies') or [spec['policy']]
        bad = [p for p in pols if p not in policies_mod.POLICIES]
        if bad:
            raise ValueError(f"unknown polic(ies): {', '.join(bad)}  "
                             f"(have: {', '.join(sorted(policies_mod.POLICIES))})")
        spec['policy'] = pols[0]

    if spec.get('compare_choice') is not None:
        _validate_choice(spec)

    if sims is not None:
        if sims < 2:
            raise ValueError(f"--sims must be at least 2, got {sims}")
        # ONE knob, split the way the solver spends it. Asking for "400 paths"
        # and getting 400 search + 1500 verification would misreport the cost
        # by 4x, which is the whole reason the count is now in provenance.
        if spec.get('spend') == 'solve':
            spec['verify_sims'] = sims
            spec['search_sims'] = max(2, sims // 2)
        else:
            spec['sims'] = sims
    return spec


def _validate_choice(spec):
    """A retirement-choice spec, checked before anything runs."""
    ch = spec['compare_choice']
    if spec['mode'] != 'retirement':
        raise ValueError("compare_choice needs mode retirement")
    if not isinstance(ch, dict) or ch.get('dimension') not in PLAN_CHOICES:
        raise ValueError(f"compare_choice.dimension must be one of "
                         f"{', '.join(PLAN_CHOICES)}")
    vals = ch.get('values')
    if isinstance(vals, str):
        vals = [v for v in vals.split(',') if v.strip()]
    try:
        vals = [float(v) for v in (vals or [])]
    except (TypeError, ValueError):
        raise ValueError(f"compare_choice.values must be numbers, got {vals!r}")
    if not vals:
        raise ValueError("compare_choice needs at least one value")
    # Runs are keyed by their tag, so a repeat would overwrite a solve.
    if len(set(vals)) != len(vals):
        raise ValueError(f"compare_choice repeats a value: {vals}")
    ch['values'] = vals
    if (len(spec.get('compare_policies') or []) > 1
            or spec.get('compare_vpw_bequests')
            or len(spec.get('compare_gk_responses') or []) > 1):
        raise ValueError("one comparison at a time: compare_choice cannot be "
                         "combined with a policy, reserve or response comparison")
    # The LEVEL is the answer to a plan question, so a choice is always solved.
    if spec.get('spend') != 'solve':
        raise ValueError("compare_choice solves each option's plan level; "
                         "drop 'spend'")
    if len(spec['portfolios']) != 1 or spec.get('derisk_at') is not None \
            or spec.get('custom') and len(spec['custom']) != 1:
        raise ValueError("compare_choice compares plans on ONE portfolio")
    dim = ch['dimension']
    if dim == 'ss_claim_age':
        if not effective_ss(spec):
            raise ValueError("a claiming age needs Social Security in your details")
        if any(not 62 <= v <= 70 for v in vals):
            raise ValueError("claiming ages run from 62 to 70")
    if dim == 'retirement_age':
        cfg = default_config()
        end = cfg['age'] + cfg['time_horizon_years']
        if any(not cfg['age'] < v < end for v in vals):
            raise ValueError(f"retirement ages must fall between {cfg['age']} "
                             f"and {end}")
    if dim == 'contributions' and any(v < 0 for v in vals):
        raise ValueError("a savings multiplier cannot be negative")


def replay_command(spec, engine_mod=None):
    """The `--spec` line that reproduces this run.

    Nine prompts produce a spec that is already saved, already recorded in
    provenance and already replayable — and the only way to get back to it was
    to answer all nine identically or to know the --spec syntax. A run whose
    numbers are worth discussing should print the command that regenerates it.

    Emits the WHOLE spec rather than a minimal one: a shorter line that quietly
    fell back to a default would reproduce a different run, which is worse than
    a long line. For the same reason it sets every env-var knob the run moved —
    the spec alone replayed an `intl catches up` run as the default scenario.
    The value is the one the engine USED, not whatever the shell holds now.
    """
    import shlex
    from montecarlo import diagnostics
    payload = json.dumps(spec, sort_keys=True, separators=(',', ':'))
    env = ''.join(f"{name}={shlex.quote(str(got))} " for _, _, got, name
                  in sorted(diagnostics.moved_assumptions(engine_mod),
                            key=lambda m: m[3]))
    return f"{env}python3 run.py --spec {shlex.quote(payload)}"


def print_replay(spec):
    """Print the replay line, wrapped so a long spec stays copy-pasteable."""
    cmd = replay_command(spec)
    print(f"\n  to run this again:")
    print(f"    {cmd}")
    if (spec.get('mode') == 'retirement'
            and 'verify_sims' not in spec and 'sims' not in spec):
        # The knob most likely to be wanted on a second run, and the one that
        # was invisible until it got a prompt.
        print(f"    ...append  --sims 3000  to settle anything that came back "
              f"NOT RESOLVED")


def _spec_from_sentence(sentence, portfolios, dry_run=False):
    """--ask: propose a spec, show it, and return it only on an explicit yes.

    The model proposes; `ask.to_spec` validates exactly as --spec does. Nothing
    runs on a proposal alone, and the replayable --spec line is printed so the
    run can be repeated without the model.
    """
    from montecarlo import ask as ask_mod
    try:
        spec = ask_mod.spec_from_sentence(sentence, portfolios)
    except ValueError as exc:
        print(f"  --ask: {exc}")
        return None
    specs, is_grid = resolve_strategies(spec, portfolios)
    print(f"\n  Proposed from: {sentence!r}")
    print(f"  {json.dumps(spec, sort_keys=True)}")
    print(f"\n  {len(specs)} strateg{'y' if len(specs) == 1 else 'ies'}, "
          f"est. {estimate_minutes(len(specs), spec):.1f} min"
          f"{'  (grid)' if is_grid else ''}")
    for g in specs:
        print(f"    {g}")
    # replay_command, not a hand-built line: it carries the env-var knobs too.
    print(f"\n  to run it without the model:\n    {replay_command(spec)}")
    if dry_run:
        return None
    if ask("\n  Run it? [y/N]", 'n').lower() not in ('y', 'yes'):
        print("  cancelled.")
        return None
    return spec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--again', action='store_true', help='repeat the last run')
    ap.add_argument('--history', action='store_true', help='list saved runs')
    ap.add_argument('--debug', action='store_true',
                    help='report where the wall clock went, and recalibrate the estimate')
    ap.add_argument('--diff', nargs='?', const='last2', metavar='A,B',
                    help='compare two saved runs (default: the two most recent)')
    ap.add_argument('--spec', metavar='JSON_OR_FILE',
                    help='run headlessly from a spec, no prompts '
                         '(e.g. \'{"mode":"retirement","portfolios":["140","152"]}\')')
    ap.add_argument('--ask', metavar='SENTENCE',
                    help='describe the run in one sentence; it is shown for a yes '
                         'before anything runs (needs the anthropic package)')
    ap.add_argument('--sims', type=int, metavar='N',
                    help='with --spec: paths per strategy (search gets N/2)')
    ap.add_argument('--dry-run', action='store_true',
                    help='with --spec: print the resolved strategies and stop')
    args = ap.parse_args(argv)

    if args.history:
        persistence.print_history()
        return 0
    if args.diff:
        paths = (persistence.list_runs(limit=2) if args.diff == 'last2'
                 else [Path(x.strip()) for x in args.diff.split(',')])
        if len(paths) < 2:
            print("  Need two saved runs to compare. Run `python3 run.py --history`.")
            return 1
        persistence.print_diff(paths[1], paths[0])   # list_runs is newest-first
        return 0

    portfolios = load_portfolios_from_csv(verbose=False)

    headless = bool(args.spec or args.ask)
    if args.ask:
        spec = _spec_from_sentence(args.ask, portfolios, dry_run=args.dry_run)
        if spec is None:
            return 1
    elif args.spec:
        try:
            spec = parse_spec(args.spec, portfolios, sims=args.sims)
        except ValueError as exc:
            print(f"  bad --spec: {exc}")
            return 1
    elif args.again:
        spec = load_preset('last')
        if spec is None:
            print("No previous run saved. Run `python3 run.py` first.")
            return 1
    else:
        spec = prompt_run(portfolios)

    if spec.get('mode') == 'sensitivity':
        # Hand straight to the sweep tool rather than reimplementing it here.
        # It re-execs a subprocess per point (engine reads these at import), so
        # it has to own the loop.
        sys.path.insert(0, str(Path(__file__).resolve().parent / 'tools'))
        import sensitivity as sens
        return sens.main([spec['dimension'], spec['values'], '--sims', '800'])

    specs, is_grid = resolve_strategies(spec, portfolios)
    if args.dry_run:
        print(f"\n  {len(specs)} strateg{'y' if len(specs) == 1 else 'ies'}, "
              f"est. {estimate_minutes(len(specs), spec):.1f} min"
              f"{'  (grid)' if is_grid else ''}")
        for g in specs:
            print(f"    {g}")
        print(f"\n  spec: {json.dumps(spec, sort_keys=True)}")
        return 0
    if not headless and not args.again and not confirm(spec, specs, portfolios):
        print("  cancelled.")
        return 1

    # --spec does NOT overwrite the interactive preset: a test run must not
    # cost you the configuration you spent the prompts building.
    if not headless:
        save_preset(spec, 'last')

    if args.debug:
        instrument.enable()
        instrument.reset()
        instrument.set_mode(spec.get('mode'))
        solver_mod.VERBOSE = True

    from montecarlo import report_retirement
    estimated = estimate_minutes(len(specs), spec)
    report_retirement.run(spec, specs, portfolios, build_config(spec), is_grid)
    print_replay(spec)
    # After the tables, where it cannot disturb a column.
    instrument.summary(estimated_minutes=estimated)
    return 0


if __name__ == '__main__':
    sys.exit(main())
