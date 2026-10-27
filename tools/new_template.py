"""
Scaffold a fresh portfolio_templates.csv for a NEW ticker universe.
==================================================================

When you change the ticker set (loaders.MODEL_TICKERS), the existing
portfolio_templates.csv still has columns for the OLD tickers. This writes a
clean skeleton whose columns match the CURRENT MODEL_TICKERS, with one valid
equal-weight example portfolio to build from.

Safety: backs up any existing portfolio_templates.csv first — to .bak, or to a
timestamped .bak when one already exists — so no run loses the current file.

Usage:
    python3 tools/new_template.py

Then edit/add your portfolios and run:
    python3 tools/annotate_templates.py && python3 tools/add_section_headers.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from montecarlo.loaders import MODEL_TICKERS  # noqa: E402

DEFAULT_CSV = Path(__file__).resolve().parent.parent / 'portfolios' / 'portfolio_templates.csv'


def scaffold(path=DEFAULT_CSV):
    path = Path(path)

    # Back up the file being replaced, every time, never clobbering a backup.
    if path.exists():
        bak = path.with_suffix(path.suffix + '.bak')
        if bak.exists():
            from datetime import datetime
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            bak = path.with_suffix(f"{path.suffix}.{stamp}.bak")
        bak.write_text(path.read_text())
        print(f"backed up existing {path.name} -> {bak.name}")

    header = ['portfolio_id', 'name'] + list(MODEL_TICKERS)
    n = len(MODEL_TICKERS)

    # One valid example: equal weight across all tickers, corrected to sum to 1.0.
    w = round(1.0 / n, 6)
    eq = [w] * n
    eq[0] = round(eq[0] + (1.0 - sum(eq)), 6)

    rows = [
        header,
        ['# === EXAMPLE — replace with your own portfolios ==='],
        ['# columns after "name" are the tickers from loaders.MODEL_TICKERS'],
        ['# each portfolio row must sum to 1.0 across the ticker columns'],
        ['100', 'Example // equal weight'] + eq,
    ]

    with open(path, 'w', newline='') as f:
        csv.writer(f, lineterminator='\n').writerows(rows)

    print(f"✓ wrote skeleton {path.name}: {n} ticker columns + 1 example (equal weight, sums to {sum(eq):.4f})")
    print("  next: add your portfolios, then")
    print("        python3 tools/annotate_templates.py && python3 tools/add_section_headers.py")


if __name__ == '__main__':
    scaffold()
