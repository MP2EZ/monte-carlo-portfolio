"""
Portfolio-template linter.
==========================

`check_consistency.py` confirms the CSV *has a column* for every ticker; this checks
the *rows themselves* are valid and internally consistent:

  - every portfolio's weights sum to 1.0 (the sim assumes this)
  - no negative weights
  - portfolio_archive.csv rows (settled questions kept restorable) also sum to 1.0
    with no negative weights, and no id appears in both files — an archived row is
    first read the day someone moves it back, so it is checked now
  - the derived nav columns match `compute_derived()` recomputed from the weights
    (catches a hand-edited row that wasn't re-run through annotate_templates.py)

Reads the CSV raw (not via `load_portfolios_from_csv`, which raises on the first bad
row) so it can report EVERY problem in one pass.

    python3 tools/check_templates.py        # exit 0 = clean, exit 1 = problems
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from montecarlo.loaders import MODEL_TICKERS, DERIVED_COLUMNS, compute_derived

CSV_FILE = ROOT / 'portfolios' / 'portfolio_templates.csv'
ARCHIVE_FILE = ROOT / 'portfolios' / 'portfolio_archive.csv'
SUM_TOL = 1e-9
NUM_TOL = 1e-4                      # derived numeric columns are rounded to 4dp
NUMERIC_DERIVED = {'bond_pct', 'us_pct', 'dm_share', 'n_funds', 'max_single_fund'}


def lint_archive(archive, template_ids, fail):
    """Row-sum and id-collision lint of the archive. Optional file: absent is clean.
    Returns the number of archived rows checked."""
    if not archive.exists():
        return 0
    rows = list(csv.reader(open(archive, newline='')))
    if not rows:
        return 0
    col = {name: i for i, name in enumerate(rows[0])}
    miss = [t for t in MODEL_TICKERS if t not in col]
    if 'portfolio_id' not in col or miss:
        fail("archive header has all ticker columns", f"missing: {miss}")
        return 0
    n = 0
    for row in rows[1:]:
        pid = row[col['portfolio_id']].strip() if row else ''
        if not pid or pid.startswith('#'):
            continue
        n += 1
        if pid in template_ids:
            fail(f"[{pid}] id is in both portfolio_templates.csv and the archive",
                 "an id lives in one file: move the row, don't copy it")
        alloc = {}
        for t in MODEL_TICKERS:
            v = row[col[t]] if col[t] < len(row) else ''
            try:
                alloc[t] = float(v) if v != '' else 0.0
            except ValueError:
                fail(f"[{pid}] archive: non-numeric weight in {t}", f"got {v!r}")
                break
        else:
            total = sum(alloc.values())
            if abs(total - 1.0) > SUM_TOL:
                fail(f"[{pid}] archive: weights sum to 1.0", f"sum = {total:.10f}")
            neg = {t: w for t, w in alloc.items() if w < 0}
            if neg:
                fail(f"[{pid}] archive: no negative weights", f"{neg}")
    return n


def main():
    problems = []

    def fail(label, detail=''):
        problems.append(label)
        print(f"  [FAIL] {label}")
        if detail:
            print(f"         {detail}")

    print("Portfolio template lint")
    print("=" * 60)

    if not CSV_FILE.exists():
        fail("portfolio_templates.csv present", f"missing {CSV_FILE}")
        return _finish(problems, 0)

    rows = list(csv.reader(open(CSV_FILE, newline='')))
    header = rows[0]
    col = {name: i for i, name in enumerate(header)}

    # Header must carry every ticker + every derived column.
    miss_tick = [t for t in MODEL_TICKERS if t not in col]
    miss_deriv = [d for d in DERIVED_COLUMNS if d not in col]
    if miss_tick:
        fail("header has all ticker columns", f"missing: {miss_tick}")
    if miss_deriv:
        fail("header has all derived columns", f"missing: {miss_deriv}")
    if miss_tick or miss_deriv:
        return _finish(problems, 0)  # can't lint rows without the columns

    id_i = col['portfolio_id']
    n_checked = 0
    template_ids = {r[id_i].strip() for r in rows[1:]
                    if r and r[id_i].strip() and not r[id_i].lstrip().startswith('#')}
    for row in rows[1:]:
        if not row or not row[id_i].strip() or row[id_i].lstrip().startswith('#'):
            continue  # blank / comment / section-header row
        pid = row[id_i].strip()
        n_checked += 1

        def cell(name):
            i = col[name]
            return row[i] if i < len(row) else ''

        # Parse weights for the model tickers.
        alloc, parse_bad = {}, False
        for t in MODEL_TICKERS:
            v = cell(t)
            try:
                alloc[t] = float(v) if v != '' else 0.0
            except ValueError:
                fail(f"[{pid}] non-numeric weight in {t}", f"got {v!r}")
                parse_bad = True
        if parse_bad:
            continue

        # Sum to 1.0, non-negative.
        total = sum(alloc.values())
        if abs(total - 1.0) > SUM_TOL:
            fail(f"[{pid}] weights sum to 1.0", f"sum = {total:.10f}")
        neg = {t: w for t, w in alloc.items() if w < 0}
        if neg:
            fail(f"[{pid}] no negative weights", f"{neg}")

        # Derived columns match a fresh compute_derived().
        want = compute_derived(alloc)
        for d in DERIVED_COLUMNS:
            got_raw = cell(d)
            exp = want[d]
            if d in NUMERIC_DERIVED:
                try:
                    got = float(got_raw) if got_raw != '' else None
                except ValueError:
                    got = None
                if got is None or abs(got - float(exp)) > NUM_TOL:
                    fail(f"[{pid}] derived '{d}' stale", f"csv={got_raw!r} expected={exp}")
            else:  # string columns: tilt_family, mom_fund, val_fund
                if got_raw.strip() != str(exp):
                    fail(f"[{pid}] derived '{d}' stale", f"csv={got_raw!r} expected={exp!r}")

    n_archived = lint_archive(ARCHIVE_FILE, template_ids, fail)

    if not problems:
        print(f"  [PASS] {n_checked} portfolios: weights sum to 1.0, non-negative, "
              f"derived columns current")
        print(f"  [PASS] {n_archived} archived portfolios: weights sum to 1.0, "
              f"non-negative, no id also live")
    return _finish(problems, n_checked)


def _finish(problems, n_checked):
    print("=" * 60)
    if problems:
        print(f"✗ {len(problems)} problem(s). Fix the row(s); if only derived columns are "
              f"stale, run `python3 tools/annotate_templates.py`.")
        return 1
    print(f"✓ All checks passed — {n_checked} portfolios valid.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
