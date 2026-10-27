# ============================================================================
# Withdrawal policies and sourcing rules
# ============================================================================
# Everything here is in REAL (today's) dollars — see engine.INFLATION. That is a
# large simplification: a "constant real" withdrawal is literally constant, with
# no inflation indexing to carry around, and guardrails compare real rates.
#
# Two independent choices:
#
#   POLICY   how much to withdraw this year. The four differ in who absorbs a
#            bad market — you, or the portfolio.
#   SOURCING which holdings to sell to fund it.
#
# See the v3.1 review notes (unpublished) for the design rationale, including why literal
# bonds-first sourcing is subsumed by `excess` and deliberately not built.
# ============================================================================

from typing import NamedTuple

# VPW amortises to this age, NOT to the simulation's end_age. Exported so the
# runner's explanatory copy reads the same number the policy uses — the two
# drifted apart once already, and the UI claimed a zero balance at a horizon the
# policy was never amortising to.
VPW_PLANNING_AGE = 100.0
# ...and never closer to the plan's end than this. A plan to 100 put the
# schedule's end ON the horizon, so the last simulated year withdrew the whole
# balance (2026-10-09). Five years leaves the last year drawing ~19%.
VPW_HORIZON_MARGIN = 5.0


def vpw_planning_age(end_age):
    """The age VPW amortises to: 100, or five years past the plan if later.

    100 for any plan ending by 95, so every VPW figure measured to 95 stands;
    past that it moves with the plan, so the schedule never ends on the
    horizon and liquidates in its final year.
    """
    return max(VPW_PLANNING_AGE, float(end_age) + VPW_HORIZON_MARGIN)


# Guyton-Klinger guardrails, as fractions of the initial withdrawal rate. These
# are the ORIGINAL rule — the `slow` preset below — not the default.
GK_UPPER = 1.20      # rate drifted 20% above plan -> spending is unsustainable
GK_LOWER = 0.80      # rate drifted 20% below plan -> spending can rise
GK_CUT = 0.90
GK_RAISE = 1.10

# HOW FAST SPENDING FOLLOWS A GROWING PORTFOLIO.
#
# The original rule is conservative by LAG, not by design. A portfolio
# compounding ~6% real drifts 20% below the initial rate only every 3-4 years,
# so a +10% raise on that cadence lets spending grow ~2.5-3%/yr against a
# portfolio growing 6% — and it falls further behind every year. Measured on a
# deterministic 6% path from $8M, guyton_klinger averages 0.61x of what VPW
# spends over 30 years; loosening only the RAISE side reaches 0.68x.
#
# The DOWNSIDE PAIR IS IDENTICAL IN EVERY PRESET, and that is the whole point:
# `upper`/`cut` are the protection, `lower`/`raise` are the participation, and
# only the second pair moves. Under the same 35%-at-72 and 40%-at-80 crash test,
# every preset below caps the worst single-year cut at -10%, against -41.8% for
# fixed_pct at 3% — which is the reason not to reach for a percentage rule to
# get the same upside.
#
# ⚠ A faster ratchet raises spending after a good year and must then cut after a
# bad one. The per-year cut stays capped, but consecutive cuts are possible, so
# read `worst yr` and the life-stage `low` column before adopting one.
GK_RESPONSE = {
    'slow':   {'upper': 1.20, 'lower': 0.80, 'cut': 0.90, 'raise': 1.10},
    'medium': {'upper': 1.20, 'lower': 0.90, 'cut': 0.90, 'raise': 1.15},
    'fast':   {'upper': 1.20, 'lower': 0.95, 'cut': 0.90, 'raise': 1.20},
}
# `fast` since 2026-10-09: every measured run in docs/RESULTS.md used it, and
# sweeps built at the old `slow` default reported levels ~15% off the runs they
# were compared against. A record without the field ran `slow` before then; its
# provenance says so, which is how `run.py --diff` attributes the change.
GK_RESPONSE_DEFAULT = 'fast'


def gk_guardrails(response=None):
    """Resolve a named response to its four guardrails.

    Returns the default for None or an unknown name rather than raising.
    """
    return dict(GK_RESPONSE.get(response or GK_RESPONSE_DEFAULT,
                                GK_RESPONSE[GK_RESPONSE_DEFAULT]))



class PolicyState(NamedTuple):
    """Per-path withdrawal state, fixed at retirement except `w_prior_year`."""
    w_initial: float            # first year's withdrawal, real dollars
    v_at_retirement: float      # portfolio value when withdrawals began
    initial_rate: float         # w_initial / v_at_retirement
    w_prior_year: float         # last year's actual withdrawal
    retirement_age: float
    end_age: float
    assumed_real_return: float  # used by vpw's amortisation
    vpw_planning_age: float = VPW_PLANNING_AGE  # NOT end_age; see above
    vpw_bequest: float = 0.0    # real dollars vpw preserves rather than spending
    # Guardrails travel WITH THE STATE, not as module globals: workers re-import
    # policies under `spawn`, so a value assigned in the parent would never
    # reach them — the same trap that put REBALANCING_BANDS in engine.py.
    gk_upper: float = GK_RESPONSE[GK_RESPONSE_DEFAULT]['upper']
    gk_lower: float = GK_RESPONSE[GK_RESPONSE_DEFAULT]['lower']
    gk_cut: float = GK_RESPONSE[GK_RESPONSE_DEFAULT]['cut']
    gk_raise: float = GK_RESPONSE[GK_RESPONSE_DEFAULT]['raise']


# ---------------------------------------------------------------------------
# Social Security
# ---------------------------------------------------------------------------
# A COLA'd lifetime income stream, so it is naturally constant in REAL terms —
# which is exactly the units this model already works in.
#
# It does two different things, and the second is easy to miss:
#   1. LEVEL — the portfolio only has to fund (need - benefit), so every
#      strategy can support more spending.
#   2. FLOOR — the benefit arrives whether or not the portfolio survives, so
#      depletion stops being catastrophic. That changes the SHAPE of the
#      downside, and downside shape is where allocation and policy decisions get
#      made. So this may be ranking-relevant, not merely level-only.
#
# Claiming age is a real decision with a known schedule, so it is modelled
# rather than assumed: benefits are reduced for claiming before Full Retirement
# Age and increased for delaying past it.

FRA_DEFAULT = 67.0          # Full Retirement Age for anyone born 1960 or later
DELAY_CREDIT_PER_YEAR = 0.08        # +8%/yr for delaying past FRA, capped at 70
EARLY_REDUCTION_FIRST_36M = 0.0667  # -6.67%/yr for the first three early years
EARLY_REDUCTION_BEYOND = 0.05       # -5%/yr for anything earlier still
MAX_DELAY_AGE = 70.0


def ss_benefit(annual_at_fra, claim_age, fra=FRA_DEFAULT):
    """Annual benefit in real dollars, adjusted for when it is claimed.

    Claiming early reduces it permanently; delaying past FRA earns credits up to
    age 70, after which there is no further gain. Roughly: 70% of PIA at 62,
    100% at 67, 124% at 70.
    """
    claim_age = float(claim_age)
    if claim_age >= fra:
        years_late = min(claim_age, MAX_DELAY_AGE) - fra
        return float(annual_at_fra * (1.0 + DELAY_CREDIT_PER_YEAR * years_late))

    years_early = fra - claim_age
    first = min(years_early, 3.0)
    beyond = max(years_early - 3.0, 0.0)
    reduction = first * EARLY_REDUCTION_FIRST_36M + beyond * EARLY_REDUCTION_BEYOND
    return float(annual_at_fra * max(1.0 - reduction, 0.0))


def ss_income(age, ss_config):
    """Benefit receivable at this age — zero before the claiming age.

    `benefit_scale` haircuts the benefit, because SCHEDULED benefits are not the
    same thing as PAYABLE benefits. The Social Security trustees project the OASI
    trust fund depleting in the early-to-mid 2030s, after which continuing
    payroll-tax income covers roughly 75-80% of scheduled benefits unless
    Congress acts. Anyone retiring after that is planning against a promise the
    system does not currently fund.

    A scale rather than a probability, deliberately: modelling it as a random
    draw would average the shortfall away across paths and make the plan look
    safer than a pessimistic-but-plausible policy outcome warrants — the same
    reason mortality is not drawn per path. Sweep it instead:

        1.00   scheduled benefits, i.e. current law honoured in full
        0.80   near the trustees' payable-benefit projection
        0.75   a harsher legislative outcome
        0.00   no benefit at all — a bound, not a forecast

    `scale_from_age` applies the haircut only from that age onward, for modelling
    a reduction that arrives mid-retirement.
    """
    if not ss_config:
        return 0.0
    claim_age = ss_config.get('claim_age', FRA_DEFAULT)
    if age < claim_age:
        return 0.0
    benefit = ss_benefit(ss_config['annual_at_fra'], claim_age,
                         ss_config.get('fra', FRA_DEFAULT))
    scale = float(ss_config.get('benefit_scale', 1.0))
    from_age = ss_config.get('scale_from_age')
    if from_age is not None and age < float(from_age):
        return float(benefit)
    return float(benefit * scale)


# ---------------------------------------------------------------------------
# Sourcing — which holdings fund the withdrawal
# ---------------------------------------------------------------------------

def source_pro_rata(values, target, withdrawal, active):
    """Sell the same fraction of every holding. Weights come out unchanged.

    The neutral control: it does nothing clever, so it is what any smarter rule
    has to beat.
    """
    total = sum(values[t] for t in active)
    if total <= 0:
        return {t: 0.0 for t in active}
    frac = min(withdrawal / total, 1.0)
    return {t: values[t] * frac for t in active}


def source_excess(values, target, withdrawal, active):
    """Sell from holdings above their post-withdrawal target, in proportion to
    the excess.

    Never increases drift and never forces a full rebalance. When the book is at
    target this degenerates to pro-rata (the excesses are exactly proportional to
    the target weights), so the two rules differ only when the book has drifted —
    which is what makes a comparison between them a clean measurement of whether
    sourcing matters at all.

    Handles a crash with no special-casing: after equities fall, bonds are the
    overweight holding, so the money comes from bonds. That is the bucket
    strategy, derived rather than bolted on — and unlike literal bonds-first it
    stops draining bonds once they reach target instead of leaving the investor
    all-equity at the worst possible moment.
    """
    total = sum(values[t] for t in active)
    if total <= 0:
        return {t: 0.0 for t in active}
    withdrawal = min(withdrawal, total)
    total_after = total - withdrawal

    excess = {t: max(0.0, values[t] - target.get(t, 0.0) * total_after) for t in active}
    pool = sum(excess.values())

    if pool >= withdrawal:
        # Partial move toward target: take it all from what is above target.
        return {t: withdrawal * excess[t] / pool for t in active}

    # Withdrawal exceeds the total overweight: take every dollar of excess, then
    # split the remainder pro-rata across what is left.
    rest = withdrawal - pool
    base = {t: values[t] - excess[t] for t in active}
    base_total = sum(base.values())
    if base_total <= 0:
        return dict(excess)
    return {t: excess[t] + rest * base[t] / base_total for t in active}


SOURCING = {
    'pro_rata': source_pro_rata,
    'excess': source_excess,
}


# ---------------------------------------------------------------------------
# Long-term care
# ---------------------------------------------------------------------------

# A spending shock UNCORRELATED WITH MARKETS, late in life. It is the one
# un-modelled risk that could argue for a bond sleeve, and every derisking
# result in this repo was measured in a world where it cannot happen.
#
# Defaults sized from the US cost of care: a private nursing-home room runs
# roughly $110-120k/yr, and the average stay is about 2-3 years. Off unless
# asked for.
LTC_DEFAULTS = {'annual_amount': 0.0, 'start_age': 82.0, 'years': 3.0}


def ltc_cost(age, ltc_config):
    """Care cost payable at this age, in real dollars. Zero outside the window.

    DETERMINISTIC AND SWEPT, not drawn per path — the same reasoning as
    `benefit_scale` and as mortality. A 20%-chance-of-$500k shock averages away
    across paths and leaves the 10th percentile almost untouched, which makes
    the plan look safer than a plausible outcome warrants. A scenario you vary
    answers "could I absorb this"; a random draw answers "what is the mean",
    which is not the question anyone asks about care costs.
    """
    if not ltc_config:
        return 0.0
    amount = float(ltc_config.get('annual_amount', 0.0) or 0.0)
    if amount <= 0:
        return 0.0
    start = float(ltc_config.get('start_age', LTC_DEFAULTS['start_age']))
    years = float(ltc_config.get('years', LTC_DEFAULTS['years']))
    return amount if start <= age < start + years else 0.0


# ---------------------------------------------------------------------------
# Policies — how much to withdraw
# ---------------------------------------------------------------------------

def constant_real(age, portfolio_value, state):
    """Take the same real amount every year, whatever the market does.

    The portfolio absorbs all the risk, so this is the only policy that can
    actually deplete — and the only one for which "probability of ruin" is a
    meaningful measure.
    """
    return float(state.w_initial)


def fixed_pct(age, portfolio_value, state):
    """Take a fixed percentage of the CURRENT balance.

    Mathematically cannot deplete, because you always take a fraction of what is
    left. The spender absorbs all the risk instead: a 35% crash is an immediate
    35% pay cut.
    """
    return float(state.initial_rate * portfolio_value)


def guyton_klinger(age, portfolio_value, state):
    """Hold spending steady in real terms, with guardrails at +/-20% of the
    initial withdrawal RATE: breach the top and cut 10%, breach the bottom and
    raise 10%.

    The middle ground — the portfolio takes the small shocks, the spender takes
    the large ones. Guyton-Klinger's separate "skip the inflation raise after a
    down year" rule has no effect here: in real terms there is no inflation raise
    to skip.
    """
    spend = state.w_prior_year
    if portfolio_value <= 0:
        return 0.0
    rate = spend / portfolio_value
    if rate > state.gk_upper * state.initial_rate:
        spend *= state.gk_cut
    elif rate < state.gk_lower * state.initial_rate:
        spend *= state.gk_raise
    return float(spend)


def vpw(age, portfolio_value, state):
    """Variable Percentage Withdrawal: amortise the portfolio over the years
    remaining, so the rate rises with age (~5.0% at 65, ~8.7% at 85).

    Cannot deplete, and unlike fixed_pct it deliberately spends the portfolio
    down rather than preserving a perpetual fraction. Suited to someone who does
    not care about leaving a bequest.

    Amortises to `vpw_planning_age` (100, or five past the plan), NOT to the simulation's
    end_age. Amortising to the horizon would drive the rate to 100% as the
    horizon closes — 53% at 85 against an age-87 endpoint, i.e. near-total
    liquidation — which is an artefact of the fixed endpoint, not a policy anyone
    would follow. Real VPW amortises over life expectancy plus a buffer; with a
    planning age of 100 this reproduces the published table closely (~5.0% at 65,
    ~8.7% at 85).

    Rate is the annuity payout factor r / (1 - (1+r)^-n) for n years remaining at
    an assumed real return r; with r = 0 it degenerates to a clean 1/n.
    """
    years_left = max(state.vpw_planning_age - age, 1.0)
    r = state.assumed_real_return
    if abs(r) < 1e-12:
        rate = 1.0 / years_left
    else:
        rate = r / (1.0 - (1.0 + r) ** -years_left)

    # `vpw_bequest` is the dial between the two extremes. Plain VPW amortises the
    # WHOLE balance, which is why it out-spends guyton_klinger so heavily while
    # leaving far less behind. Reserving an amount and amortising only the excess
    # moves continuously between them:
    #
    #   bequest 0        maximum spending, minimum estate — VPW as usually stated
    #   bequest large    approaches "live off the growth", like a preservation rule
    #
    # The reserve is discounted to today because it is a target for the END of the
    # horizon; holding back its full nominal size from day one would over-reserve
    # badly in the early years.
    spendable = portfolio_value
    if state.vpw_bequest > 0:
        reserve = state.vpw_bequest / ((1.0 + r) ** years_left)
        spendable = max(0.0, portfolio_value - reserve)
    return float(min(rate, 1.0) * spendable)


POLICIES = {
    'constant_real': constant_real,
    'fixed_pct': fixed_pct,
    'guyton_klinger': guyton_klinger,
    'vpw': vpw,
}

# Policies that derive their OWN opening draw from the balance, rather than
# taking the plan's target as a control.
#
# `constant_real` spends the target, `fixed_pct` turns it into a rate
# (initial_rate = target / balance) and `guyton_klinger` sets its guardrails off
# that same rate — for those three the target genuinely steers the path. `vpw`
# reads none of it: its withdrawal is rate(years to planning age) x spendable.
#
# Seeding year 1 from the target therefore handed vpw one year it did not
# choose, and made the reported first-year figure neither vpw's own draw nor a
# level it sustains. Measured: on a $7.3M balance vpw's own opening draw is ~5%,
# so a $250k bar understated its first year by ~12%, and moving the bar 2.25x
# shifted lifetime spending 2.3% through that single seeded year.
#
# Declared as data so the engine asks a question rather than naming a policy.
SELF_DIRECTED = frozenset({'vpw'})
