"""
Insert (or refresh) human-readable section header rows in portfolio_templates.csv.
=================================================================================

Adds two levels of '#' comment rows so the library is easy to scan in a
spreadsheet, placed on the seams already in the ID scheme:

    # ===== 100% EQUITY / 0% BONDS (1xx) =====     <- per bond tier (hundreds digit)
    # --- core ---                                 <- per tilt family, inside multi-tilt tiers

These are comment rows: the loader and simulator skip them (see loaders.py), and
the update tools preserve them. Requires the derived 'tilt_family' column, so run
annotate_templates.py first.

Idempotent — re-run any time portfolios are added/reordered. It strips its own
previously-inserted headers first, but leaves any other '#' notes you added alone.

Usage:
    python3 add_section_headers.py
"""

import csv
from pathlib import Path

# CSV lives at the project root (this script is in tools/).
CSV_FILE = str(Path(__file__).resolve().parent.parent / 'portfolios' / 'portfolio_templates.csv')

TIER_LABELS = {
    1: '100% EQUITY / 0% BONDS (1xx)',
    2: '95% EQUITY / 5% BONDS (2xx)',
    3: '90% EQUITY / 10% BONDS (3xx)',
    4: '85% EQUITY / 15% BONDS (4xx)',
    5: '80% EQUITY / 20% BONDS (5xx)',
    6: '75% EQUITY / 25% BONDS (6xx)',
}
TILT_LABELS = {
    'core': 'core',
    'value': 'value tilt',
    'momentum': 'momentum tilt',
    'mom+val': 'momentum + value',
    'quality': 'quality (JQUA)',
}


def _is_generated_header(line):
    # The exact forms written below, with any trailing commas a spreadsheet
    # adds; a note that merely starts '# ---' is yours and stays.
    s = line.strip().rstrip(',').rstrip()
    return ((s.startswith('# ===== ') and s.endswith(' ====='))
            or (s.startswith('# --- ') and s.endswith(' ---')))


def add_headers(path=CSV_FILE):
    path = Path(path)
    raw = open(path, newline='').read().splitlines(keepends=True)
    # Strip our own previous headers (idempotent); keep data + any other comments.
    raw = [ln for ln in raw if not _is_generated_header(ln)]
    if not raw:
        raise ValueError(f"{path} is empty")

    header_line = raw[0]
    nl = '\r\n' if header_line.endswith('\r\n') else '\n'
    H = next(csv.reader([header_line]))
    if 'tilt_family' not in H:
        raise ValueError("No 'tilt_family' column — run annotate_templates.py first.")
    i_id, i_tilt = H.index('portfolio_id'), H.index('tilt_family')

    def parse(ln):
        if not ln.strip():
            return None
        return next(csv.reader([ln]))

    # First pass: which tiers have more than one tilt family (only those get sub-headers).
    tier_tilts = {}
    for ln in raw[1:]:
        f = parse(ln)
        if not f or not f[i_id].strip() or f[i_id].startswith('#'):
            continue
        tier = int(f[i_id]) // 100
        tier_tilts.setdefault(tier, set()).add(f[i_tilt])
    multi = {t: len(v) > 1 for t, v in tier_tilts.items()}

    # Second pass: emit, inserting headers on tier/tilt boundaries.
    out, prev_tier, prev_tilt = [header_line], None, None
    n_tier = n_tilt = 0
    for ln in raw[1:]:
        f = parse(ln)
        if not f or not f[i_id].strip() or f[i_id].startswith('#'):
            out.append(ln)                                   # blank / other comment
            continue
        pid = int(f[i_id])
        tier, tilt = pid // 100, f[i_tilt]
        if tier != prev_tier:
            out.append(f'# ===== {TIER_LABELS.get(tier, f"{tier}xx")} ====={nl}')
            n_tier += 1
            if multi.get(tier):
                label = 'momentum (base)' if pid == 100 else TILT_LABELS.get(tilt, tilt)
                out.append(f'# --- {label} ---{nl}')
                n_tilt += 1
            prev_tilt = tilt
        elif tilt != prev_tilt and multi.get(tier):
            out.append(f'# --- {TILT_LABELS.get(tilt, tilt)} ---{nl}')
            n_tilt += 1
            prev_tilt = tilt
        out.append(ln)
        prev_tier = tier

    open(path, 'w', newline='').write(''.join(out))
    print(f"✓ Inserted {n_tier} tier headers + {n_tilt} tilt sub-headers into {path.name}")


if __name__ == '__main__':
    add_headers()
