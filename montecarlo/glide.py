# ============================================================================
# Glide paths — allocation schedules over age
# ============================================================================
# A glide path is a schedule that points at rows already in
# portfolio_templates.csv. Authoring happens in portfolio-ID space, never in
# ticker space: asking anyone to specify 25 weights at each of N ages is
# unusable, but "hold 141 until 65, then 340" is not.
#
# Waypoints are STEPS, not interpolated. That means zero new allocation maths —
# the target simply becomes a different CSV row, and the existing quarterly
# rebalancer moves the book to it. Interpolating would make the bands chase a
# drifting target, generating trades that are neither rebalancing nor glide.
# Fine-grained waypoints approximate a smooth glide if wanted:
#     141@37 > 240@60 > 340@67 > 540@75
#
# The `>` syntax is the internal and stored form. The interactive runner (phase
# 7) builds these; users are not expected to type them.
# ============================================================================

import re
from typing import NamedTuple

from montecarlo.loaders import BOND_SLEEVES, compute_derived, native_sleeve, with_bonds


class Waypoint(NamedTuple):
    """A portfolio that takes effect at `age` and holds until the next waypoint."""
    portfolio_id: str
    age: float


def parse_schedule(spec, start_age):
    """Parse a glide spec into an ordered tuple of Waypoints.

    Accepted forms (whitespace anywhere is ignored):
        '141'                     hold 141 for the whole horizon
        '141>340@65'              hold 141, switch to 340 at age 65
        '141>340@65>540@75'       two switches
        '141@40>340@65'           explicit start age on the first waypoint

    Only the first waypoint may omit `@age` (it defaults to `start_age`).
    Ages must strictly increase.

    Raises:
        ValueError: on an empty spec, a missing portfolio id, a non-numeric age,
            a non-first waypoint without an age, or non-increasing ages.
    """
    if not spec or not spec.strip():
        raise ValueError("empty glide spec")

    waypoints = []
    for i, chunk in enumerate(spec.split('>')):
        chunk = chunk.strip()
        if not chunk:
            raise ValueError(f"empty segment in glide spec {spec!r}")

        if '@' in chunk:
            pid, _, age_text = chunk.partition('@')
            pid, age_text = pid.strip(), age_text.strip()
            try:
                age = float(age_text)
            except ValueError:
                raise ValueError(
                    f"age must be a number in {chunk!r} (glide spec {spec!r})") from None
        else:
            if i > 0:
                raise ValueError(
                    f"{chunk!r} needs an age, e.g. {chunk}@65 (glide spec {spec!r}) — "
                    "only the first portfolio may omit it")
            pid, age = chunk, float(start_age)

        if not pid:
            raise ValueError(f"missing portfolio id in {chunk!r} (glide spec {spec!r})")
        if waypoints and age <= waypoints[-1].age:
            raise ValueError(
                f"glide ages must strictly increase: {pid}@{age:g} does not follow "
                f"{waypoints[-1].portfolio_id}@{waypoints[-1].age:g} (spec {spec!r})")
        waypoints.append(Waypoint(pid, age))

    return tuple(waypoints)


def resolve_ref(ref, portfolios):
    """Resolve a portfolio reference to an allocation dict.

    A reference is either a plain id from the CSV, or a synthesised one:

        141                 as stored
        141+10              141's equity sleeve at 10% bonds, native sleeve
        141+10:bnd_only     same, but 100% BND — for comparing bond sleeves

    Synthesis is what makes a bond ladder like-for-like. Looking up sibling rows
    instead (the old behaviour) meant 141's ladder pointed at the x40 family,
    which holds AVDE where 141 holds VEA — so a "derisking" comparison silently
    bundled a fund swap with the bond decision.
    """
    ref = ref.strip()
    if ref in portfolios:
        return portfolios[ref]['allocation']

    if '+' not in ref:
        raise ValueError(
            f"unknown portfolio {ref!r}. Available: {', '.join(sorted(portfolios))}")

    base, _, rest = ref.partition('+')
    base = base.strip()
    if base not in portfolios:
        raise ValueError(f"unknown base portfolio {base!r} in reference {ref!r}")

    pct_text, _, sleeve = rest.partition(':')
    sleeve = sleeve.strip() or native_sleeve(base)
    try:
        bond_pct = float(pct_text) / 100.0
    except ValueError:
        raise ValueError(
            f"bond percentage must be a number in {ref!r} (e.g. 141+10)") from None
    if sleeve not in BOND_SLEEVES:
        raise ValueError(
            f"unknown bond sleeve {sleeve!r} in {ref!r}; "
            f"have {', '.join(sorted(BOND_SLEEVES))}")
    return with_bonds(portfolios[base]['allocation'], bond_pct, sleeve)


def portfolio_at(schedule, age):
    """The portfolio id in effect at `age`.

    Inclusive at the switch age, so a waypoint at 65 applies from the moment the
    investor turns 65. Ages before the first waypoint hold the first portfolio.
    """
    current = schedule[0].portfolio_id
    for wp in schedule:
        if age >= wp.age:
            current = wp.portfolio_id
        else:
            break
    return current


def format_spec(schedule):
    """Render a schedule back to its canonical spec string (round-trips)."""
    if len(schedule) == 1:
        return schedule[0].portfolio_id
    parts = [schedule[0].portfolio_id]
    parts += [f"{wp.portfolio_id}@{wp.age:g}" for wp in schedule[1:]]
    return '>'.join(parts)


def same_shape(spec, base):
    """`spec` with its portfolio swapped for `base`, everything else verbatim.

        same_shape('140>140+5@60>140@65', '152')  ->  '152>152+5@60>152@65'

    Ages, bond levels and an explicit `:sleeve` carry over; a native sleeve
    stays native, so it resolves to `base`'s own. Works on the text rather than
    a parsed schedule because `format_spec` drops an explicit first age.

    Raises:
        ValueError: if the schedule moves between portfolios — there is then no
            single portfolio to swap.
    """
    chunks = [c.strip() for c in spec.split('>')]
    ids = [re.match(r'[^+@]*', c).group() for c in chunks]   # '140' of '140+5@60'
    own = {i.strip() for i in ids}
    if len(own) != 1:
        raise ValueError(f"{spec!r} moves between portfolios "
                         f"({', '.join(sorted(own))}); it has no single shape")
    return '>'.join(base + c[len(i):] for c, i in zip(chunks, ids))


def validate(schedule, portfolios):
    """Raise ValueError if any waypoint names something unresolvable."""
    for wp in schedule:
        resolve_ref(wp.portfolio_id, portfolios)      # raises with a useful message
    return schedule


DERISK_TIERS = (0.05, 0.10, 0.20)   # the bond ladder the CSV's x2/x3/x5 encode


def derisk_variants(base_id, switch_age, tiers=DERISK_TIERS, sleeve=None):
    """Generate a like-for-like derisk ladder for any base portfolio.

    Synthesised from the base's OWN equity sleeve, so every rung differs from the
    control only in bond weight. Previously this looked up sibling rows sharing a
    tilt family and bond tier, which is not the same thing: 141's ladder resolved
    to the x40 rows, which hold AVDE where 141 holds VEA, so the comparison
    bundled a fund swap with the bond decision. Synthesis makes that class of
    confound impossible.

    `sleeve` selects the bond mix, defaulting to the base's native one — pass it
    explicitly to compare sleeve A against sleeve B on a fixed equity side.

    Returns glide specs, control (no bonds added) first.
    """
    sleeve = sleeve or native_sleeve(base_id)
    if sleeve not in BOND_SLEEVES:
        raise ValueError(f"unknown bond sleeve {sleeve!r}")

    suffix = '' if sleeve == native_sleeve(base_id) else f":{sleeve}"
    specs = [base_id]
    specs += [f"{base_id}>{base_id}+{pct * 100:g}{suffix}@{switch_age:g}"
              for pct in sorted(tiers)]
    return specs


def build_plan(schedule, portfolios):
    """Turn a schedule into the (age, allocation) plan the engine consumes."""
    return tuple((wp.age, resolve_ref(wp.portfolio_id, portfolios)) for wp in schedule)


def _equity_pct(portfolios, pid):
    return (1.0 - compute_derived(resolve_ref(pid, portfolios))['bond_pct']) * 100.0


def format_chart(schedule, portfolios, end_age, width=56):
    """Compact ASCII equity-percentage chart, so a schedule can be eyeballed."""
    start = schedule[0].age
    span = max(end_age - start, 1e-9)

    rows = []
    for i, wp in enumerate(schedule):
        until = schedule[i + 1].age if i + 1 < len(schedule) else end_age
        lead = int(round((wp.age - start) / span * width))
        length = max(int(round((until - wp.age) / span * width)), 1)
        rows.append((_equity_pct(portfolios, wp.portfolio_id), lead, length,
                     wp.portfolio_id))

    out = ['  equity %']
    for pct, lead, length, pid in rows:
        out.append(f"   {pct:5.0f} {' ' * lead}{'█' * length} {pid}")

    axis_ticks = 6
    ages = [start + span * k / (axis_ticks - 1) for k in range(axis_ticks)]
    positions = [int(round((a - start) / span * width)) for a in ages]
    bar, labels = [' '] * (width + 1), [' '] * (width + 6)
    for pos, age in zip(positions, ages):
        if pos < len(bar):
            bar[pos] = '┬'
        text = f"{age:.0f}"
        at = max(pos - len(text) // 2, 0)
        for j, ch in enumerate(text):
            if at + j < len(labels):
                labels[at + j] = ch
    out.append('         └' + ''.join(bar).rstrip())
    out.append('          ' + ''.join(labels).rstrip())
    return '\n'.join(out)
