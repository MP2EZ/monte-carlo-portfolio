"""
The facts about YOU, in one place.
==================================

One record (`config/investor.json`), edited from `run.py`'s Your details, so
the same fact is never typed in two places.

Two design rules, both learned from bugs that were live in this repo:

STORE WHAT YOU KNOW, DERIVE WHAT FOLLOWS.
    `time_horizon_years` was stored, and is only correct while age is 37 —
    `end_age = age + time_horizon_years`, so bumping age to 38 silently moved
    the plan to 96. Store `plan_to_age` and derive the horizon. Likewise
    `initial_value` is the sum of the account balances, never typed twice.

NAME THE AGE THE CONTRIBUTIONS WERE MEASURED AT.
    `engine.calculate_annual_contribution` grew every contribution from a bare
    literal 36 — `years_elapsed = age - 36` — which encodes WHEN the amounts
    were last trued up, not the investor's age, and appeared in no config
    anywhere. Update your age and your amounts in the same sitting and the new
    amounts get a year of compounding they already contain.
"""

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / 'config' / 'investor.json'

TAX_TREATMENTS = ('pretax', 'roth', 'hsa', 'taxable')

# Committed fallback, so a fresh clone runs before anyone edits anything.
# The two 401k lines share a growth rate and stop age.
DEFAULTS = {
    'age': 37,
    # Plan to 95, not to life expectancy. A 65-year-old's median life
    # expectancy is early-to-mid 80s, but the errors are not symmetric: dying at
    # 80 with money left over is fine, running out at 92 is not. Planning to the
    # median means a coin-flip chance of outliving the plan, so the convention
    # is a high percentile — roughly 1 in 4 of a 65-year-old couple sees 95.
    #
    # Mortality is deliberately NOT modelled per path. Drawing a death age would
    # end many paths early, before the withdrawal stream is tested, and make
    # every result look better for a reason that is no comfort if you are the
    # one who lives. A fixed long horizon is the conservative choice.
    'plan_to_age': 95,
    'retirement_age': 65,
    'contributions_measured_at_age': 36,
    # SOCIAL SECURITY, from your ssa.gov statement: the benefit at full
    # retirement age (stored ANNUAL, entered monthly), the age you claim, and
    # the share of scheduled benefits you assume is actually paid. None = no
    # benefit.
    #
    # Runs read it from this file, the per-run prompt appears only when it is
    # unset, and a spec carries it only as an explicit per-run override (older
    # saved runs, --spec, sensitivity sweeps).
    'social_security': None,
    # Illustrative round balances, not anyone's accounts: this file is public.
    'accounts': {
        '401k_trad': {'tax': 'pretax', 'balance': 150000.0,
                      'annual_amount': 18000.0, 'growth_rate': 0.029,
                      'stop_age': 60, 'unconfirmed': True},
        '401k_roth': {'tax': 'roth', 'balance': 25000.0,
                      'annual_amount': 6000.0, 'growth_rate': 0.029,
                      'stop_age': 60, 'unconfirmed': True},
        # stop_age is the LAST age an account contributes, and withdrawals
        # start AT retirement_age, so 64 stops one year before them. Roth money
        # needs earned income.
        'roth_ira': {'tax': 'roth', 'balance': 75000.0,
                     'annual_amount': 7500.0, 'growth_rate': 0.041,
                     'stop_age': 64},
        'hsa': {'tax': 'hsa', 'balance': 25000.0,
                'annual_amount': 0.0, 'growth_rate': 0.029,
                'stop_age': 60, 'unconfirmed': True},
        'taxable': {'tax': 'taxable', 'balance': 25000.0,
                    'annual_amount': 6000.0, 'growth_rate': 0.030,
                    'stop_age': 60},
    },
}

_WARNED = set()


def _warn_once(key, *lines):
    """Say it once per process. default_config() is called several times a run."""
    if key in _WARNED:
        return
    _WARNED.add(key)
    for line in lines:
        print(line)


def overlaps(inv):
    """Accounts still contributing at the retirement age, in record order.

    The engine contributes while age <= stop_age and withdraws from the
    retirement age, so stop_age >= retirement_age means paying in and drawing
    out in the same years; None (never stops) does it for the whole of
    retirement. Usually a mistake, occasionally real (part-time work), so this
    WARNS and validate() does not refuse it. An account adding nothing is
    never an overlap.
    """
    ret = inv.get('retirement_age')
    return [name for name, a in (inv.get('accounts') or {}).items()
            if (a.get('annual_amount') or 0) > 0
            and (a.get('stop_age') is None or a['stop_age'] >= ret)]


def load(path=None):
    """The investor record, falling back to DEFAULTS when no file exists.

    Deliberately side-effect free: reading must not create the file, so a
    validation failure cannot be laundered into a written config.

    The fallback is LOUD. DEFAULTS are the seeded guesses this module shipped
    with, not anyone's real numbers, so a missing or corrupt file would
    otherwise produce a complete, plausible-looking run on the wrong balances
    and contributions with nothing on screen to say so. Same for a file that
    parses but does not validate — hand-edit an age to 200 and every table
    downstream is quietly wrong.
    """
    path = Path(path) if path else FILE
    try:
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError('not a JSON object')
    except FileNotFoundError:
        _warn_once('missing',
                   f"  ⚠ {path.name} not found — falling back to the committed",
                   f"    defaults, which are SEEDED GUESSES, not your numbers.",
                   f"    Set them with 'Your details' in run.py.")
        return copy.deepcopy(DEFAULTS)
    except (OSError, ValueError) as e:
        _warn_once('unreadable',
                   f"  ⚠ {path.name} could not be read ({e.__class__.__name__}) —",
                   f"    falling back to the committed defaults, which are SEEDED",
                   f"    GUESSES, not your numbers.")
        return copy.deepcopy(DEFAULTS)
    inv = copy.deepcopy(DEFAULTS)
    inv.update({k: v for k, v in raw.items() if k != 'accounts'})
    if isinstance(raw.get('accounts'), dict):
        inv['accounts'] = copy.deepcopy(raw['accounts'])
    problems = validate(inv)
    if problems:
        _warn_once('invalid',
                   f"  ⚠ {path.name} loaded but does NOT validate:",
                   *[f"      {p}" for p in problems],
                   f"    Results below are computed from it anyway. Fix it with",
                   f"    'Your details' in run.py.")
    elif overlaps(inv):
        _warn_once('overlap',
                   f"  ⚠ still contributing into retirement (from "
                   f"{inv['retirement_age']}): {', '.join(overlaps(inv))}.",
                   f"    Stop age is the LAST age you contribute; set it below "
                   f"the retirement",
                   f"    age in 'Your details' unless you will have earned income then.")
    return inv


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate(inv):
    """Every problem, not just the first. Returns a list of strings."""
    bad = []
    age = inv.get('age')
    ret = inv.get('retirement_age')
    end = inv.get('plan_to_age')
    measured = inv.get('contributions_measured_at_age')
    for name, v in (('age', age), ('retirement_age', ret), ('plan_to_age', end),
                    ('contributions_measured_at_age', measured)):
        if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= 120:
            bad.append(f"{name} must be a whole number of years, got {v!r}")
    if not bad:
        if not age < ret:
            bad.append(f"retirement_age ({ret}) must be above age ({age})")
        if not ret < end:
            bad.append(f"plan_to_age ({end}) must be above retirement_age ({ret})")
        if measured > age:
            bad.append(f"contributions_measured_at_age ({measured}) is in the "
                       f"future — it cannot exceed age ({age})")

    accounts = inv.get('accounts') or {}
    if not accounts:
        bad.append("at least one account is required")
    for name, a in accounts.items():
        if not isinstance(a, dict):
            bad.append(f"{name}: an account must be an object, got {a!r}")
            continue
        if a.get('tax') not in TAX_TREATMENTS:
            bad.append(f"{name}: tax must be one of {', '.join(TAX_TREATMENTS)}, "
                       f"got {a.get('tax')!r}")
        for field in ('balance', 'annual_amount'):
            v = a.get(field)
            if not _is_number(v) or v < 0:
                bad.append(f"{name}: {field} must be a non-negative number, got {v!r}")
        g = a.get('growth_rate')
        if not _is_number(g) or abs(g) > 0.5:
            bad.append(f"{name}: growth_rate must be a decimal within +/-0.5 "
                       f"(0.029 = 2.9%/yr), got {g!r}")
        s = a.get('stop_age')
        if s is not None and (not isinstance(s, int) or isinstance(s, bool)
                              or not 0 <= s <= 120):
            bad.append(f"{name}: stop_age must be a whole number or null, got {s!r}")
    bad.extend(validate_social_security(inv.get('social_security')))
    return bad


def validate_social_security(ss):
    """Problems with a social_security block; None (no benefit) is valid."""
    if ss is None:
        return []
    if not isinstance(ss, dict):
        return [f"social_security must be a block or null, got {ss!r}"]
    bad = []
    a = ss.get('annual_at_fra')
    # $240k/yr is $20k/month — well above any real benefit. The prompt already
    # asks about a MONTHLY figure over $20k because it is almost certainly an
    # annual one typed in the wrong box; this is the same guard for a hand edit.
    if not _is_number(a) or not 0 < a <= 240_000:
        bad.append(f"social_security.annual_at_fra must be a yearly amount above "
                   f"0 and at most 240,000, got {a!r}")
    c = ss.get('claim_age')
    if not _is_number(c) or not 62 <= c <= 70:
        bad.append(f"social_security.claim_age must be between 62 and 70, got {c!r}")
    sc = ss.get('benefit_scale', 1.0)
    if not _is_number(sc) or not 0 <= sc <= 1:
        bad.append(f"social_security.benefit_scale must be between 0 and 1 "
                   f"(0.75 = 75% of scheduled), got {sc!r}")
    return bad


def save(inv, path=None):
    """Write, but only a record that validates. Raises ValueError otherwise."""
    problems = validate(inv)
    if problems:
        raise ValueError("; ".join(problems))
    path = Path(path) if path else FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(inv, indent=2, sort_keys=False) + '\n')


def total_balance(inv):
    return float(sum(a.get('balance', 0.0) for a in inv['accounts'].values()))


def contribution_schedule(inv):
    """The shape `engine.calculate_annual_contribution` consumes.

    Only the three fields the engine reads are passed through. `tax`, `balance`
    and `unconfirmed` are investor bookkeeping and would just be dead weight
    marshalled to every worker process.
    """
    return {
        'frequency': 'annual',
        'measured_at_age': inv['contributions_measured_at_age'],
        'accounts': {
            name: {'annual_amount': float(a['annual_amount']),
                   'growth_rate': float(a['growth_rate']),
                   'stop_age': a.get('stop_age')}
            for name, a in inv['accounts'].items()},
    }


def unconfirmed(inv):
    """Accounts seeded with a guess that only you can settle."""
    return [n for n, a in inv['accounts'].items() if a.get('unconfirmed')]


def summary(inv):
    """One line for the menu, so a stale config is visible before a run.

    Carries the unconfirmed count rather than printing a separate warning line:
    it is seen every time the menu is, and costs no vertical space. One of the
    seeded guesses — an HSA contribution of zero — may be materially wrong
    rather than merely unverified, so it should not be easy to stop noticing.
    """
    annual = sum(a.get('annual_amount', 0.0) for a in inv['accounts'].values())
    line = (f"age {inv['age']} · ${total_balance(inv) / 1e3:,.0f}k · "
            f"${annual / 1e3:,.1f}k/yr")
    ss = inv.get('social_security')
    if ss:
        # Monthly, because that is the unit on the statement you check it against.
        line += (f" · SS ${ss['annual_at_fra'] / 12 / 1e3:,.1f}k/mo "
                 f"@{ss['claim_age']:g}")
    n = len(unconfirmed(inv))
    return f"{line} · ⚠ {n} unconfirmed" if n else line
