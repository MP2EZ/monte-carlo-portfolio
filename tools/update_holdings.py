"""
Update a portfolio row in portfolio_templates.csv from your actual holdings.
============================================================================

Enter your real allocation as PERCENTAGES per fund (they should add up to ~100).
The script normalizes them to weights summing to exactly 1.0 and rewrites ONLY
the target row in portfolio_templates.csv — every other template is untouched.

Why this exists: because of account placement (e.g. EM split across AVEM in the
Roth and DFWIX-EM in the 401k), your live allocation is a blend that can't be
typed cleanly as "all Avantis" or "all Vanguard." This lets you keep row 100 in
sync without hand-editing the big CSV or breaking the sum-to-1.0 rule the loader
enforces.

Usage:
    python3 update_holdings.py
    -> it PROMPTS you to paste your holdings in the terminal, then rewrites the
       TARGET_ID row in portfolio_templates.csv (normalized to sum 1.0) and prints
       a before/after diff. Re-run anytime your holdings drift.

    Paste format: a row of PERCENTAGES, a line with '//', then a row of TICKERS,
    mapped left-to-right (1st value -> 1st ticker). You can also pipe a file:
       python3 update_holdings.py < my_holdings.txt
"""

import csv
import sys
from pathlib import Path

# This script lives in tools/; make the project root importable for `loaders`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from montecarlo.loaders import MODEL_TICKERS, DERIVED_COLUMNS, compute_derived

# =============================================================================
# EDIT THIS BLOCK
# =============================================================================
CSV_FILE = str(Path(__file__).resolve().parent.parent / 'portfolios' / 'portfolio_templates.csv')

TARGET_ID = '141'           # which portfolio_id row to overwrite (141 = your actual holdings)

# Leave NAME = None to keep the existing row name, or set a string to replace it.
NAME = None

# --- INPUT --------------------------------------------------------------------
# By default this script PROMPTS you in the terminal — just paste your holdings
# there when asked. No need to edit anything below.
#
# Optional: to skip the prompt (e.g. for automation), hard-code the paste here.
# Leave it as "" to keep the interactive prompt. Example of a filled-in value:
#     PASTE = '''
#     5.0%  3.0%  4.5%  12.0%  12.0%  9.0%  4.1%  41.0%  9.4%  0.0%
#     //
#     COST  AVEE  AVDV  AVUV   VFMO   AVEM  AVDE  VOO    VEA   VWO
#     '''
PASTE = ""

# How close to 100 the entered percentages must be before the script complains.
SUM_TOLERANCE = 1.0   # percentage points
# =============================================================================


def parse_paste(paste):
    """Parse a 'values // tickers' paste block into {ticker: percent}.

    Values come first, then a '//' separator, then tickers. Items are split on
    any whitespace and mapped positionally. Trailing '%' and commas are stripped.
    A ticker listed twice (one fund in two accounts) is ADDED, as the xlsx
    reader does.
    """
    if '//' not in paste:
        raise ValueError("PASTE must contain '//' between the values and the tickers.")
    values_str, tickers_str = paste.split('//', 1)
    values = [tok.rstrip('%').replace(',', '') for tok in values_str.split()]
    tickers = tickers_str.split()
    if len(values) != len(tickers):
        raise ValueError(
            f"Got {len(values)} values but {len(tickers)} tickers — they must line up 1:1.\n"
            f"  values ({len(values)}):  {values}\n"
            f"  tickers ({len(tickers)}): {tickers}"
        )
    holdings = {}
    for ticker, value in zip(tickers, values):
        try:
            holdings[ticker] = holdings.get(ticker, 0.0) + float(value)
        except ValueError:
            raise ValueError(f"Couldn't read '{value}' as a percentage for {ticker}.")
    return holdings


def prompt_for_holdings():
    """Read holdings interactively from the terminal (or piped stdin).

    Paste a row of PERCENTAGES, a line with '//', then a row of TICKERS, then
    press Enter on an empty line. As a shortcut you can also skip the '//' and
    just paste two lines: percentages first, tickers second.
    """
    print("Paste your holdings, then press Enter on an empty line to finish.")
    print("  • a row of percentages")
    print("  • a line with  //")
    print("  • a row of tickers   (values map to tickers left-to-right)")
    print("(Shortcut: skip the '//' and just paste two lines — values, then tickers.)\n")

    lines = []
    while True:
        try:
            line = input()
        except EOFError:          # Ctrl-D or end of a piped file
            break
        if line.strip() == '':
            if lines:
                break             # blank line after content -> done
            continue              # ignore leading blank lines from a paste
        lines.append(line)

    content = [ln for ln in lines if ln.strip()]
    if not content:
        raise SystemExit("No input received — nothing to do.")

    block = "\n".join(content)
    if '//' in block:
        return parse_paste(block)
    if len(content) == 2:                       # two-line shortcut, no '//'
        return parse_paste(content[0] + " // " + content[1])
    raise SystemExit(
        "Couldn't parse that. Put a '//' between the values and the tickers, "
        "or paste exactly two lines (values, then tickers)."
    )


def update_row(csv_path, target_id, holdings, name=None):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    with open(csv_path, newline='') as f:
        rows = list(csv.reader(f))

    header = rows[0]
    # Only real model tickers are allocations; derived/meta columns are left alone.
    ticker_cols = [c for c in header if c in MODEL_TICKERS]

    unknown = [t for t in holdings if t not in ticker_cols]
    if unknown:
        raise ValueError(
            f"These funds aren't columns in the CSV: {unknown}\n"
            f"Valid tickers: {ticker_cols}"
        )

    negative = [t for t, v in holdings.items() if v < 0]
    if negative:
        raise ValueError(f"Negative holdings: {negative}")
    entered_sum = sum(holdings.values())
    if abs(entered_sum - 100.0) > SUM_TOLERANCE:
        raise ValueError(
            f"Entered percentages sum to {entered_sum:.2f}%, expected ~100% "
            f"(±{SUM_TOLERANCE}). Check for a typo before continuing."
        )

    # Rounded to the six decimals the CSV stores, with the rounding residual on
    # the largest holding: thirds would otherwise be written as 0.333333 x 3.
    weights = {t: round(holdings.get(t, 0.0) / entered_sum, 6) for t in ticker_cols}
    largest = max(weights, key=weights.get)
    weights[largest] = round(weights[largest] + 1.0 - sum(weights.values()), 6)

    id_idx = header.index('portfolio_id')
    name_idx = header.index('name')
    target_row_num = None
    for i, row in enumerate(rows):
        if i == 0:
            continue
        if row and row[id_idx].strip() == str(target_id):
            target_row_num = i
            break
    if target_row_num is None:
        raise ValueError(f"portfolio_id {target_id} not found in {csv_path.name}")

    old_row = list(rows[target_row_num])

    derived = compute_derived(weights)

    new_row = list(old_row)
    while len(new_row) < len(header):   # pad if the stored row was ragged
        new_row.append('')
    if name is not None:
        new_row[name_idx] = name
    for col_i, col in enumerate(header):
        if col in weights:
            # trim trailing zeros: 0.41 not 0.410000
            new_row[col_i] = f"{weights[col]:.6f}".rstrip('0').rstrip('.')
        elif col in DERIVED_COLUMNS:
            new_row[col_i] = derived[col]
    rows[target_row_num] = new_row

    def nonzero(row):
        out = {}
        for col_i, col in enumerate(header):
            if col in ticker_cols:
                try:
                    v = float(row[col_i] or 0)
                except ValueError:
                    v = 0.0
                if v > 0:
                    out[col] = v
        return out

    before, after = nonzero(old_row), nonzero(new_row)
    print(f"Portfolio {target_id}: {new_row[name_idx]}")
    print(f"  weights sum -> {sum(after.values()):.6f}\n")
    print(f"  {'fund':<7} {'before':>9} {'after':>9}")
    print(f"  {'-'*7} {'-'*9} {'-'*9}")
    for fund in sorted(set(before) | set(after)):
        b = before.get(fund, 0.0)
        a = after.get(fund, 0.0)
        mark = '' if abs(a - b) < 1e-6 else '  <- changed'
        print(f"  {fund:<7} {b:>9.4f} {a:>9.4f}{mark}")

    with open(csv_path, 'w', newline='') as f:
        # LF, as the CSV is committed: csv.writer's CRLF default rewrote every line.
        csv.writer(f, lineterminator='\n').writerows(rows)
    print(f"\n✓ Wrote row {target_id} to {csv_path.name}")


if __name__ == '__main__':
    holdings = parse_paste(PASTE) if PASTE.strip() else prompt_for_holdings()
    update_row(CSV_FILE, TARGET_ID, holdings, NAME)
