---
name: engine-change
description: This skill should be used for ANY edit to montecarlo/engine.py that touches the simulation loop, the random draws, or the per-path setup — adding an uncertainty band, changing a regime rule, altering how correlations are selected, adding or removing a np.random call. Trigger it whenever the user says "add a shock", "make X uncertain", "change the regime logic", or when a change would shift the random stream. The golden master and several fixtures assume determinism, and forgetting to pin them turns them into noise tests that still pass.
---

# Change the Simulation Engine

Edit the monthly loop or the per-path draws without silently disabling the
checks that guard them.

## The failure this exists to prevent

Adding the shared cash-rate band and then the inflation band each randomised
`mu`, which is what they are FOR. Each time, the analytic anchor —
`regression_check`'s zero-volatility compounding test — kept running against a
now-random `mu` and became a one-sample noise test. **It happened twice in one
day**, and the second time only because the first fix pinned one shock and not
the other. It read 200,105 against an expected 221,964; a looser tolerance would
have let it pass forever.

**Every deterministic fixture must pin EVERY shared shock, not the new one.**

## Procedure

1. **Constant first** — put the parameter in `montecarlo/factors.py` with an `MC_` env var
   (workers re-import under `spawn`, so a module global set in the parent never
   reaches them). Re-export it in `montecarlo/engine.py` beside `RF_REAL`.
2. **Size it from measurement, not judgment.** Record the measurement in the
   comment. Every band in this model has one.
3. **Wire it** — a per-path draw goes BEFORE the per-ticker loop, so the stream
   position does not depend on how many tickers a portfolio holds.
4. **Pin the deterministic fixtures OFF** — search for every place that assumes
   a fixed `mu`:
   - `tools/regression_check.py` — the analytic anchor saves and restores
     `RF_REAL_STD` and `INFLATION_STD` together. Add yours to that tuple.
   - `tools/check_logic.py` — `_zero_vol_engine`, `_install_universe`, and any
     test isolating one mechanism. Install fixtures inside `with
     _engine_state():`, which puts every engine global back on exit; never a
     hand-written save list (two drifted from what their fixtures set).
5. **Test the DIFFERENTIAL, not the existence.** A shock applied uniformly to
   every asset is another level term and cannot change a decision. If the point
   is that it hits assets differently, assert the ratio — and verify the test
   FAILS against a deliberately uniform version.
6. **Re-bless** — `python3 tools/regression_check.py --update`. Expected: the
   draw order moved. Determinism and the analytic anchor must still PASS on
   their own; if they do not, step 4 is incomplete.
7. **Measure and document** — see the `remeasure` skill.

## Hard rules

- A new `np.random` call moves the stream: the golden master WILL change, and
  that is fine. Determinism and the analytic anchor changing is NOT fine.
- **Size every draw to the UNIVERSE, never to the plan's holdings**, and read
  it by position (`ticker_indices`). Until 2026-10-02 draws were sized to the
  held funds, so strategies holding different funds saw different markets and
  every head-to-head between them was silently unpaired — `166` and
  `166>166+5@94`, identical until 94, read 587k vs 603k. The golden master
  CANNOT catch this (its synthetic plan holds every fund); two `check_logic`
  tests do: `test_a_fund_sees_the_same_market_whatever_else_the_plan_holds`
  and `test_a_never_held_sleeve_does_not_change_what_you_spend`.
- Add a dimension to `sensitivity.DIMENSIONS` for any knob that can change a
  conclusion — an unsweepable assumption is an unfalsifiable one.
- Nothing is done until `tools/validate_all.py` is all-PASS.

## Reference

- Worked examples: `factors.RF_REAL_STD`, `factors.INFLATION_STD`
- Why bands are shared vs per-ticker: `.claude/CLAUDE.md`, the cash-rate section
