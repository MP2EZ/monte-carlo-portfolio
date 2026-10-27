---
name: remeasure
description: This skill should be used after ANY change that moves the model's numbers — a parameter regen, an engine edit, a corrected constant, a new uncertainty band — and whenever the user asks to "re-measure", "update the figures", "re-run the sweeps", or says a documented number looks stale. Use it proactively before quoting any dollar figure from the docs, because measured numbers in this repo live in five places and markdown is only three of them.
---

# Re-measure After a Change

Find every place a measured number lives and bring them all forward together.
Partial updates are worse than none: a stale figure beside a fresh one reads as
authoritative.

## The failure this exists to prevent

The 2026-09-10 premium retraction updated CLAUDE.md, README and the tool
summaries. It missed a hardcoded `~5x` span, with its two dollar levels, inside a `print()` in
`montecarlo/report_retirement.py` — so a **retracted finding kept printing at the end of
every run for two days**, and was found only from a user's terminal transcript.

**Numbers hide in code, not just in markdown.** Grep the prints.

## Where measured numbers live

1. `README.md` "What it has found" and `docs/RESULTS.md` — the public summary,
   in relative terms only (no dollar levels: the publish guard refuses yours)
2. `tools/sensitivity.py` — each `Dim(note=...)` in `DIMENSIONS`
3. `montecarlo/report_retirement.py` — the premium reminder in `run()`, printed every run
4. `docs/METHODOLOGY.md` — the Expected Returns chapter (premia table, measured gaps)

`check_logic` pins 2 against 3 (`test_the_report_quotes_the_premium_span...`),
so those two cannot drift apart. Nothing pins the rest — check them by hand.

## Procedure

1. **Scope it** — which measurements does this change invalidate? A parameter or
   μ change invalidates every dollar figure. A correlation or regime change
   invalidates levels but not spans. A pure display change invalidates nothing.
2. **Sweep** — `python3 tools/sensitivity.py <dim> <values> --sims 800`. Run bare
   for the menu. ~10 min per dimension; eight dimensions is ~80 min.
3. **Accumulation** — the median/p10/p90 table, 2,000 paths.
4. **Update all five locations.** Grep for the old figures by value
   (`grep -rn "2.33x\|121k" *.py *.md .claude/`) — that catches the prints.
5. **Mark, do not delete, what you did not re-measure.** A flagged stale figure
   is honest; an unflagged one is a lie with a date on it.
6. **Gate** — `python3 tools/validate_all.py`.

## Hard rules

- SPANS are ratios and survive a uniform level shift; DOLLAR figures do not.
- Sweep points are CORRELATED — they all solve on `seed_offset=0`. A level gap
  needs an out-of-sample check (`tools/solver_calibration.py`) before quoting.
  Twelve sweep points agreeing is ONE sample, not twelve confirmations.
- Never quote a plan level without saying when it was measured.

## Reference

- The sweep tool and its registry: `tools/sensitivity.py`
- Out-of-sample checks: `tools/solver_calibration.py`
