---
name: refresh-parameters
description: This skill should be used whenever the model's parameters are regenerated from market data — e.g. "refresh the data", "update the parameters", "regenerate", "pull the latest market data", "run the quarterly update", or any time data_collection_regime_model.py is about to be run. Use it ALSO whenever tools/check_parameter_drift.py reports a FAIL, because interpreting that failure correctly is the whole point of this skill.
---

# Refresh the Parameters

Regenerate every parameter from live market data and decide, deliberately,
whether what moved was a refresh or a defect.

## The failure this exists to prevent

`check_parameter_drift.py` is the golden master FOR DATA. When it fails, the
green-gate reflex is to run `--update` and move on. **That erases the only thing
standing between the model and a silent data defect.** Both defects found on
2026-09-10/11 — the rotated international proxies and the truncated correlation
window — produced individually plausible numbers and were caught ONLY by this
diff. A blessed-away failure is a defect shipped.

**A drift failure is a question, not a chore.**

## Procedure

1. **Regenerate** — `python3 data_collection_regime_model.py` (~10 min, needs
   network). It fetches 27 tickers plus proxies, Ken French / AQR files and TIPS
   quotes, and runs `validate_parameters.py` at the end.
2. **Gate** — `python3 tools/validate_all.py`.
3. **If `check_parameter_drift` FAILS, investigate before blessing.** Run
   `python3 tools/check_parameter_drift.py --verbose` to see every move, not just
   the failures, and ask what mechanism explains it:
   - a **`history_months` DECREASE** is never routine — a splice dropped data.
     Find out which layer before going further.
   - a **regime volatility** move of multiple pp means the underlying series
     changed. Check `tools/proxy_score.py` for that ticker.
   - a **correlation mean** move means the window or the regime labelling moved.
   - a genuine refresh moves μ by ≲0.11pp (0.54pp for rolling-window tickers).
4. **Bless only once the cause is known** —
   `python3 tools/check_parameter_drift.py --update`. Say in the commit WHY it
   moved. If it cannot be explained, it is a defect: stop and fix it.
5. **Re-bless the ENGINE baseline only if the engine changed** —
   `python3 tools/regression_check.py --update`. A data refresh alone must never
   move it; the golden master runs on a synthetic universe precisely so it does
   not. If it moved without an engine edit, something is wrong.
6. **Re-measure anything the docs quote** — see the `remeasure` skill. Parameter
   changes invalidate plan levels and sweep spans.

## Hard rules

- Never `--update` a drift failure you cannot explain.
- Never hand-edit `output/` — always regenerate.
- `tools/validate_all.py` must be all-PASS before declaring done.
- `output/` is gitignored; `tools/parameter_baseline.json` is committed and IS
  the record. Commit it with the explanation.

## Reference

- Tolerances and why they are sized as they are: `tools/check_parameter_drift.py`
- Proxy fidelity: `tools/proxy_score.py`, `docs/METHODOLOGY.md` (Standard ETFs)
