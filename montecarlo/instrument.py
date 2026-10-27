"""
Where the wall clock actually goes.
===================================

Enabled by `--debug` on run.py, or MC_DEBUG=1. Off by default and close to free
when off: every entry point is a flag check.

Per-line timestamps were the obvious design and are the wrong one. The only
print inside a solve fires AFTER it finishes, so stamping lines tells you
"strategy 2 took 74s" — which is already visible from the lines appearing — and
says nothing about where inside the 74 seconds it went. It would also prefix the
tables, breaking the column alignment they depend on.

So: phases are timed explicitly, paths are counted at the one choke point that
generates them (`solver._paths`), and the summary prints once at the end where
it cannot disturb a table. `_paths` is the right hook and `simulate` was not:
accumulation runs bypass `simulate` entirely, so an allocation run counted zero
paths, printed one undifferentiated phase, and never recalibrated.

The measured seconds-per-path is written back to config/timing.json, because
`run._SECONDS_PER_PATH` is a constant measured once by hand against an algorithm
that has since been replaced. An estimate that recalibrates from its own last
run beats one that drifts silently — the estimate was 2x low for months because
nothing ever checked it against reality.
"""

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIMING_FILE = ROOT / 'config' / 'timing.json'

ENABLED = bool(os.environ.get('MC_DEBUG'))

# The floor for trusting a run's measured seconds-per-path. It was 20,000, and
# that is ABOVE what a typical run produces: twelve strategies under a
# closed-form policy generate 1,500 paths each = 18,000. So the runs actually
# being done could never recalibrate the constant they were being estimated
# with, and it stayed at 9.52ms against a real 4.77ms — every vpw run was
# announced at roughly twice its true length.
#
# 5,000 is still far above the noise floor (it is several strategies' worth) and
# below anything a real comparison produces.
MIN_CALIBRATION_PATHS = 5_000

# Below this a "measurement" is not one. Observed range on real work is 3.6-5.1
# ms/path; 0.2ms is two orders of magnitude under that.
MIN_PLAUSIBLE_SECONDS_PER_PATH = 0.0002

# label -> {'seconds', 'calls', 'evals', 'paths', 'depth'}
_phases = {}
_order = []
_stack = []
_t0 = None


def enable(on=True):
    """Turn instrumentation on for this process."""
    global ENABLED
    ENABLED = bool(on)


def reset():
    global _phases, _order, _stack, _t0
    _phases, _order, _stack = {}, [], []
    _t0 = time.time()


def _slot(label, depth):
    if label not in _phases:
        _phases[label] = {'seconds': 0.0, 'calls': 0, 'evals': 0,
                          'paths': 0, 'depth': depth}
        _order.append(label)
    return _phases[label]


@contextmanager
def timed(label):
    """Time a phase. Nests: the label is joined with whatever is already open."""
    if not ENABLED:
        yield
        return
    if _t0 is None:
        reset()
    _stack.append(label)
    slot = _slot(' / '.join(_stack), len(_stack) - 1)
    start = time.time()
    try:
        yield
    finally:
        slot['seconds'] += time.time() - start
        slot['calls'] += 1
        _stack.pop()


def record_paths(n, parallel=True):
    """One batch of simulated paths, attributed to the innermost open phase.

    Called from solver.simulate, which is the single place paths are generated —
    the closed-form route and the fitted route both go through it, so the count
    is complete by construction rather than by remembering to add call sites.
    """
    if not ENABLED or not _stack:
        return
    slot = _phases.get(' / '.join(_stack))
    if slot is not None:
        slot['evals'] += 1
        slot['paths'] += int(n)
        if not parallel:
            slot['serial'] = slot.get('serial', 0) + int(n)


def _fmt(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {seconds % 60:04.1f}s"


def totals():
    paths = sum(p['paths'] for p in _phases.values())
    evals = sum(p['evals'] for p in _phases.values())
    return paths, evals


# WHICH WORKLOAD THE CONSTANT DESCRIBES. A decumulation path runs the withdrawal
# policy, Social Security and the per-band adequacy scoring every month; an
# accumulation path does none of that, so keeping one constant for both drags
# the estimate toward whichever mode ran last.
#
# ⚠ BUT MODE IS NOT THE DOMINANT AXIS — MACHINE LOAD IS, and this was seeded
# from a measurement that did not know that. Two retirement runs of nearly
# identical size measured 12.59ms/path (2026-09-12, 149.4k paths) and
# 5.36ms/path (2026-09-13, 149.0k paths), 2.3x apart. The second run shows the
# same spread INSIDE itself: its first two solves ran at 8.07 and 10.77ms/path
# and everything after them at 3.7-4.7, consistent with other work on the
# machine finishing partway through. Seeding `retirement` with the 12.59ms
# figure then made the next estimate 2.1x too HIGH — the opposite error, same
# size. The seed has been removed; the damped, clamped blend converges from
# whatever clean runs it sees. Keep the per-mode split, but do not read a
# single run's measurement as a property of the workload.
_mode = None


def set_mode(mode):
    """Name the workload this run's calibration belongs to."""
    global _mode
    _mode = mode if mode in ('retirement', 'accumulation') else None


def _read_timing():
    try:
        return json.loads(TIMING_FILE.read_text())
    except (OSError, ValueError):
        return {}


def calibrate():
    """Persist measured seconds-per-path so the next estimate is grounded.

    Blended with the stored value rather than replacing it, and only written
    from a sample big enough to mean something — a two-strategy smoke run should
    not reset the constant a production run established.

    Written under the current MODE (see `set_mode`), keeping the legacy
    top-level key in step so an older reader — or a run whose mode was never
    set — still finds a value.
    """
    paths, _ = totals()
    serial = sum(p.get('serial', 0) for p in _phases.values())
    # A serial run measures a different execution mode; calibrating the estimate
    # from it would make every production forecast wrong in the other direction.
    if not ENABLED or paths < MIN_CALIBRATION_PATHS or serial or _t0 is None:
        return None
    measured = (time.time() - _t0) / paths
    # A REAL path costs milliseconds. Anything below this is not a measurement —
    # it is a test recording fake paths, or a run whose work never happened —
    # and letting it through poisons the constant for every future estimate.
    # That is exactly what happened: a unit test recorded 9,000 paths of
    # near-zero duration into the real timing file on every validate_all.
    if measured < MIN_PLAUSIBLE_SECONDS_PER_PATH:
        return None
    prior = load_seconds_per_path()
    # SLOW blend, and clamped. At 50/50 one noisy run moved the stored constant
    # from 9.52ms to ~2.3ms, which then announced a 1.5-minute run as 0.7 — an
    # over-correction in the opposite direction. The measurement varies 1.4x
    # run-to-run on identical work, so the constant must move slower than the
    # noise, and no single run may move it more than 50% either way.
    if prior:
        blended = 0.25 * measured + 0.75 * prior
        blended = min(max(blended, prior * 0.5), prior * 1.5)
    else:
        blended = measured
    try:
        TIMING_FILE.parent.mkdir(parents=True, exist_ok=True)
        rec = _read_timing()
        entry = {'seconds_per_path': round(blended, 6),
                 'last_measured': round(measured, 6),
                 'paths': paths}
        rec.update(entry)                       # legacy top level, still read
        if _mode:
            rec.setdefault('modes', {})[_mode] = entry
        TIMING_FILE.write_text(json.dumps(rec, indent=2) + '\n')
    except OSError:
        return measured, blended
    return measured, blended


def load_seconds_per_path(mode=None):
    """Measured value if one has been recorded, else None (caller falls back).

    Prefers the entry for THIS mode; falls back to the legacy top-level value
    when that mode has never been measured, so a fresh file still estimates.
    """
    rec = _read_timing()
    for value in ((rec.get('modes') or {}).get(mode or _mode or '', {})
                  .get('seconds_per_path'), rec.get('seconds_per_path')):
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if v > 0:
            return v
    return None


def summary(estimated_minutes=None):
    """Print the breakdown. Called once, at the end, where no table can be hurt."""
    if not ENABLED or not _order:
        return
    total = time.time() - _t0
    paths, evals = totals()
    print(f"\n  TIMING   total {_fmt(total)}")
    for label in _order:
        p = _phases[label]
        if p['seconds'] < 0.05 and not p['paths']:
            continue
        name = label.split(' / ')[-1]
        pad = '  ' * p['depth']
        share = f"({p['seconds'] / total:4.0%})" if total > 0 else ''
        detail = ''
        if p['paths']:
            detail = f"   {p['evals']:>4} evals, {p['paths'] / 1e3:>7.1f}k paths"
        elif p['calls'] > 1:
            detail = f"   {p['calls']:>4} calls"
        print(f"    {pad}{name:<22} {_fmt(p['seconds']):>10} {share}{detail}")

    if paths:
        per = total / paths
        serial = sum(p.get('serial', 0) for p in _phases.values())
        print(f"\n    {paths / 1e3:,.1f}k paths at {per * 1e3:.2f}ms each")
        if serial:
            # Batches under solver.PARALLEL_THRESHOLD skip the pool entirely, so
            # a small run's per-path cost says nothing about a production one.
            # Without this line the figure reads as an 8x regression against the
            # 4ms constant when it is simply a different execution mode.
            print(f"    ⚠ {serial / paths:.0%} of paths ran SERIALLY (batches under"
                  f" solver.PARALLEL_THRESHOLD)")
            print(f"      per-path cost here is NOT comparable to a production run")
        else:
            # Marshalling is a quarter of the cost and the first thing to move if
            # something starts being shipped to workers that should not be.
            print(f"    (simulation + pool marshalling; the split was ~14ms/~4ms "
                  f"when last profiled)")
    if estimated_minutes:
        actual = total / 60.0
        ratio = actual / estimated_minutes
        # "0.5x high" is not readable — it means the ESTIMATE was 2x too high,
        # and the number shown was the actual/estimate ratio. Say which way, in
        # the direction a reader cares about.
        verdict = ('accurate' if 0.75 <= ratio <= 1.33
                   else (f"estimate was {ratio:.1f}x too LOW" if ratio > 1
                         else f"estimate was {1 / ratio:.1f}x too HIGH"))
        print(f"    estimate was {estimated_minutes:.1f} min, actual "
              f"{actual:.1f} min — {verdict}")
    got = calibrate()
    if got:
        # Report BOTH: the run measured one thing, the file stores a blend of it
        # with the previous run. Printing only the measurement implied the stored
        # value was 8.57 when it was 8.65.
        measured, stored = got
        note = (f"measured {measured * 1e3:.2f}ms/path" if measured == stored
                else f"measured {measured * 1e3:.2f}ms/path, blended to "
                     f"{stored * 1e3:.2f} with the previous run")
        print(f"    recalibrated: {note} → {TIMING_FILE.relative_to(ROOT)}")
