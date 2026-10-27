"""
Run the full validation harness.
================================

One command that chains every check, in cheapest-first order, and reports a summary.
Use it as the pre-commit / pre-production-run gate.

  1. check_consistency   — ticker universe agrees across files (structure)
  2. validate_parameters — generated JSON + correlation matrices are sane (data)
  3. check_templates     — portfolio rows valid, derived columns current (data)
  4. check_parameter_drift — parameters moved no more than a refresh should (data)
  5. check_logic         — pure-logic unit checks (behavior)
  6. regression_check    — engine golden master unchanged (behavior)

Stages 2 and 4 are different questions and both are needed. `validate_parameters`
asks whether each number is individually plausible, which a badly wrong one
passes easily — VEA carried EMERGING market volatility for 41% of its history
and every figure sat comfortably inside its bounds. `check_parameter_drift` asks
whether it MOVED, against a committed baseline, which is what actually catches a
proxy or splice defect. It is the golden master for data, and like the engine's
it must be re-blessed deliberately:

    python3 tools/check_parameter_drift.py --update

Each sub-tool is its own process (isolates their sys.exit), run via the same
interpreter. Exit 0 iff every stage passes.

    python3 tools/validate_all.py
"""

import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
STAGES = [
    'check_consistency.py',
    'validate_parameters.py',
    'check_templates.py',
    'check_parameter_drift.py',
    'check_logic.py',
    'regression_check.py',
]


def main():
    results = {}
    for script in STAGES:
        print(f"\n{'#' * 60}\n# {script}\n{'#' * 60}")
        rc = subprocess.run([sys.executable, str(TOOLS / script)]).returncode
        results[script] = rc

    print(f"\n{'=' * 60}\nVALIDATION SUMMARY\n{'=' * 60}")
    for script, rc in results.items():
        print(f"  {'PASS' if rc == 0 else 'FAIL'}  {script}")
    failed = [s for s, rc in results.items() if rc != 0]
    print("=" * 60)
    if failed:
        print(f"✗ {len(failed)} stage(s) failed: {failed}")
        return 1
    print("✓ All validation stages passed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
