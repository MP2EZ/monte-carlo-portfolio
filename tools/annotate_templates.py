"""
Append (or refresh) derived navigation columns on portfolio_templates.csv.
===========================================================================

Adds columns computed from each portfolio's allocations — see DERIVED_COLUMNS
in loaders.py for the list and compute_derived() for the math:

    bond_pct  us_pct  dm_share  tilt_family  mom_fund  val_fund  n_funds  max_single_fund

They're a navigation layer only: the raw ticker columns are untouched, and the
loader/simulator ignore these columns (they parse weights by ticker allowlist).

Idempotent — re-run any time the templates change to recompute. Comment ('#')
and blank rows are preserved as-is.

Usage:
    python3 annotate_templates.py
"""

import csv
import sys
from pathlib import Path

# This script lives in tools/; make the project root importable for `loaders`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from montecarlo.loaders import MODEL_TICKERS, DERIVED_COLUMNS, compute_derived

# CSV lives at the project root; anchor to it so the tool works from any cwd.
CSV_FILE = str(Path(__file__).resolve().parent.parent / 'portfolios' / 'portfolio_templates.csv')


def annotate(path=CSV_FILE):
    path = Path(path)
    rows = list(csv.reader(open(path, newline='')))
    if not rows:
        raise ValueError(f"{path} is empty")

    header = rows[0]
    # Base = everything that isn't a derived column (so re-running just refreshes).
    base_cols = [c for c in header if c not in DERIVED_COLUMNS]
    base_pos = [header.index(c) for c in base_cols]
    tick_pos = {t: header.index(t) for t in MODEL_TICKERS if t in header}
    id_i = header.index('portfolio_id')

    def cell(row, i):
        return row[i] if i < len(row) else ''

    out = [base_cols + DERIVED_COLUMNS]
    annotated = 0
    for row in rows[1:]:
        if not row:
            out.append(row)                       # blank separator line
            continue
        base = [cell(row, i) for i in base_pos]
        pid = (cell(row, id_i)).strip()
        if not pid or pid.startswith('#'):
            out.append(base)                      # comment row: no derived values
            continue
        alloc = {}
        for t, i in tick_pos.items():
            v = cell(row, i)
            alloc[t] = float(v) if v not in ('', None) else 0.0
        d = compute_derived(alloc)
        out.append(base + [d[c] for c in DERIVED_COLUMNS])
        annotated += 1

    # LF, as the CSV is committed: csv.writer's CRLF default rewrote every line.
    with open(path, 'w', newline='') as f:
        csv.writer(f, lineterminator='\n').writerows(out)
    print(f"✓ Annotated {annotated} portfolios with {len(DERIVED_COLUMNS)} derived columns:")
    print(f"  {', '.join(DERIVED_COLUMNS)}")


if __name__ == '__main__':
    annotate()
