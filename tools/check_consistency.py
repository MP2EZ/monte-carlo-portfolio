"""
Cross-file ticker consistency checker.
======================================

The ticker universe is defined in several places that MUST agree. Adding or
removing a ticker touches all of them (see RUNBOOK_tickers.md). This script
verifies they're in sync and fails loudly if not — run it after any ticker change
and before a production simulation.

Checks:
  1. data_collection_regime_model.py imports MODEL_TICKERS (no divergent literal)
  2. output JSON ticker_list                    == MODEL_TICKERS                  (order-sensitive)
  3. correlation_*_<N>.npy matrices are <N>x<N> with N == len(MODEL_TICKERS)
  4. portfolio_templates.csv has a column for every model ticker
  5. every model ticker is classified into exactly one role bucket in compute_derived
  6. single stocks are valid tickers and not shadowed by a PARAM_STD_OVERRIDE entry

Usage:
    python3 check_consistency.py        # exit 0 = all good, exit 1 = problems
"""

import ast
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

# This script lives in tools/; the project root is one level up.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from montecarlo.loaders import MODEL_TICKERS, SINGLE_STOCKS, _US_FUNDS, _BOND_FUNDS, _DM_FUNDS, _EM_FUNDS
JSON = ROOT / 'output' / 'ticker_parameters_regimes_v8.json'
CSV = ROOT / 'portfolios' / 'portfolio_templates.csv'
SPECIAL_ROLE = {'DFWIX', 'VT'}  # handled explicitly in compute_derived (blended regions)


def _extract_list(pyfile, name):
    """Return a top-level `name = [literal list]` from a .py file, or None.

    None means the name isn't assigned a hardcoded list literal — e.g. it's
    imported (`from montecarlo.loaders import MODEL_TICKERS as TICKERS`) or aliased to
    another name. That's the desired post-consolidation state.
    """
    tree = ast.parse((ROOT / pyfile).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    try:
                        return ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        return None
    return None


def _imports(pyfile, name):
    """True if `pyfile` has a `from ... import name` (aliased or not).

    _extract_list returns None for an imported name and for one that is not
    defined at all, so on its own it cannot tell 'imports MODEL_TICKERS' from
    'TICKERS was deleted'."""
    tree = ast.parse((ROOT / pyfile).read_text())
    return any(isinstance(node, ast.ImportFrom)
               and any(a.name == name for a in node.names)
               for node in ast.walk(tree))


def main():
    problems = []

    def check(ok, label, detail=''):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if not ok:
            if detail:
                print(f"         {detail}")
            problems.append(label)

    dc = _extract_list('data_collection_regime_model.py', 'TICKERS')

    print("Ticker consistency check")
    print("=" * 60)

    # MODEL_TICKERS (loaders.py) is the single source of truth.
    # 1: the consumers must IMPORT it, not redefine a divergent literal.
    check(dc is None and _imports('data_collection_regime_model.py', 'MODEL_TICKERS'),
          "data_collection imports MODEL_TICKERS (no hardcoded TICKERS list)",
          "a hardcoded list, or no import — wire it as `from montecarlo.loaders import MODEL_TICKERS as TICKERS`")

    # 2. JSON ticker_list matches MODEL_TICKERS (order-sensitive — defines matrix row order).
    if JSON.exists():
        data = json.loads(JSON.read_text())
        json_tickers = [k for k in data if k != 'metadata']
        check(json_tickers == MODEL_TICKERS, "output JSON ticker_list == MODEL_TICKERS",
              "regenerate with data_collection_regime_model.py")
    else:
        check(False, "output JSON present", f"missing {JSON} — run data_collection")

    # 3. Correlation matrices are N x N for N == len(MODEL_TICKERS).
    exp_n = len(MODEL_TICKERS)
    cor_ok = True
    cor_detail = ''
    for kind in ('baseline', 'low', 'medium', 'high'):
        matches = glob.glob(str(ROOT / 'output' / f'correlation_{kind}_{exp_n}*.npy'))
        if not matches:
            cor_ok, cor_detail = False, f"no correlation_{kind}_{exp_n}*.npy (stale N?)"
            break
        shape = np.load(matches[0]).shape
        if shape != (exp_n, exp_n):
            cor_ok, cor_detail = False, f"{kind} is {shape}, expected ({exp_n},{exp_n})"
            break
    check(cor_ok, f"correlation matrices are {exp_n}x{exp_n}", cor_detail)

    # 4. CSV has a column for every model ticker.
    if CSV.exists():
        header = next(csv.reader(open(CSV)))
        missing = [t for t in MODEL_TICKERS if t not in header]
        check(not missing, "portfolio_templates.csv has all ticker columns",
              f"missing columns: {missing}")
    else:
        check(False, "portfolio_templates.csv present", f"missing {CSV}")

    # 5. Every model ticker is classified into exactly one role bucket.
    role_union = _US_FUNDS + _BOND_FUNDS + _DM_FUNDS + _EM_FUNDS
    dupes = {t for t in role_union if role_union.count(t) > 1}
    unclassified = [t for t in MODEL_TICKERS if t not in role_union and t not in SPECIAL_ROLE]
    check(not unclassified and not dupes, "every ticker classified into one role (compute_derived)",
          f"unclassified: {unclassified}; in >1 role: {dupes}")

    # 6. Single stocks: valid tickers AND not shadowed by an explicit PARAM_STD_OVERRIDE
    #    (which would silently bypass the inflated single-stock parameter-uncertainty band).
    pso = _extract_list('data_collection_regime_model.py', 'PARAM_STD_OVERRIDE') or {}
    not_tickers = [t for t in SINGLE_STOCKS if t not in MODEL_TICKERS]
    shadowed = [t for t in SINGLE_STOCKS if t in pso]
    check(not not_tickers and not shadowed,
          f"single stocks get the inflation [{', '.join(sorted(SINGLE_STOCKS)) or 'none'}]",
          f"not in MODEL_TICKERS: {not_tickers}; shadowed by PARAM_STD_OVERRIDE "
          f"(bypasses the bump): {shadowed}")

    print("=" * 60)
    if problems:
        print(f"✗ {len(problems)} problem(s). Fix per RUNBOOK_tickers.md, then re-run.")
        sys.exit(1)
    print(f"✓ All checks passed — {exp_n} tickers consistent across the system.")
    sys.exit(0)


if __name__ == '__main__':
    main()
