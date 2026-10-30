# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Monte Carlo portfolio simulation system with regime-switching volatility and parameter uncertainty. Models 25-50 year investment horizons across **27 tickers** (18 equity ETFs + 7 bond ETFs, one of them TIPS and one pure Treasury + 1 mutual fund + 1 stock) with sophisticated statistical techniques including Markov regime transitions, regime-dependent correlation matrices, and epistemic/aleatoric uncertainty separation.

**Current Status:** v3.5 (October 2026) — real-dollar reporting, glide paths,
retirement-era withdrawals, inverse solver, bootstrap CIs, sensitivity sweeps,
Social Security, LTC shocks, turnover measurement, `--debug` diagnostics and a
headless `--spec` runner. 18 active portfolios (18 settled ones in
`portfolios/portfolio_archive.csv`). Engine split out of the monolith.

**Read [docs/RESULTS.md](../docs/RESULTS.md) for the evidence** behind every
claim in this file — and before re-deriving anything, because most of it has
already been measured, several things twice.

## ⭐ If the user wants to run this with DIFFERENT tickers

Two invokable skills are the entry points (in `.claude/skills/`):
- **`/setup-portfolio`** — full ticker-universe swap + build sample portfolios.
- **`/add-ticker`** — incremental add/remove of a single ticker.

These are thin launchers that orchestrate the canonical docs and **gate on
`python3 tools/check_consistency.py` (must be all-PASS)**. The procedures live in
**[docs/CUSTOMIZING.md](../docs/CUSTOMIZING.md)** (full swap + backtest-window
reasoning) and **[docs/RUNBOOK_tickers.md](../docs/RUNBOOK_tickers.md)** (exhaustive
coupling map). Do NOT improvise a ticker change — the skills + docs exist so every
session does it the same way.

## Architecture

### Three-Stage Simulation Model

The codebase implements a progressive uncertainty model:

1. **Stage 1 (Baseline):** Updated parameters from Phase 1 data collection, fixed volatility
2. **Stage 2 (Parameter Uncertainty):** + Epistemic uncertainty (estimation error) sampled once per path
3. **Stage 3 (Regime-Switching):** + Aleatoric uncertainty (volatility regimes) transitioning monthly via Markov chains

Each stage builds on the previous, allowing controlled comparison of uncertainty impacts.

### Data Flow

```
Ken French/AQR Data + yfinance
  → data_collection_regime_model.py
  → output/ticker_parameters_regimes_v8.json (27 tickers with metadata)
  → output/correlation_*.npy (4 matrices: baseline, low, medium, high)
  → montecarlo/loaders.py  → montecarlo/engine.py  (deflated to REAL dollars at import)
  → run.py (interactive: retirement strategies or allocations)
  → montecarlo/report_retirement.py / montecarlo/report.py
```

**Everything downstream of `montecarlo/engine.py` is in REAL (today's) dollars.** Nominal rates are
converted at load via `engine.real_rate` / `engine.real_dispersion`; `engine.INFLATION`
is the assumption. A "constant real" withdrawal is therefore literally constant.

### Key Scripts

- **`data_collection_regime_model.py`**: Unified pipeline - downloads 27 actual tickers + proxies, splices with Ken French/AQR academic data (1963-2026), calculates regime parameters and 4 correlation matrices (27×27). Outputs v8 JSON with embedded metadata. Supports blended proxies (DFWIX = 78% DFIEX + 22% DFCEX; VT = 50% VTSMX + 50% VGTSX).

- **`run.py`**: ⭐ **The way to run this.** Interactive, question-driven: pick the question
  and it derives the portfolios. `python3 run.py` to configure, `python3 run.py --again` to
  repeat. Never type glide syntax — derisk sets are generated from tilt family + bond tiers.
  The menu is the three routes below (Portfolios / Retirement choices / Market & life
  assumptions), then Re-run last and Your details, which edits `config/investor.json`
  in place.

- **`montecarlo/investor.py`**: Age, plan-to age, retirement age, the five accounts —
  401k trad/Roth, Roth IRA, HSA, taxable — each with a balance, contribution,
  growth rate, stop age and `tax` treatment — and **Social Security** (benefit
  at 67, claim age, % of scheduled assumed paid). All edited from `run.py`'s
  Your details. SS moved here from the per-run spec on 2026-09-23: runs read it from
  this file and stop asking; a spec carrying `social_security` overrides it for
  that run only (None = exclude), which is how older saved runs still replay. `tools/tax_estimate.py` reads
  `tax` and the balances (R5 Stage 0); the ENGINE does not — it simulates one
  pooled book, and per-account state was deliberately not built because taxes
  proved level-only for the portfolio decision (see docs/RESULTS.md, R5).

  Two rules, both from bugs that were live here:
  - **Store what you know, derive what follows.** `time_horizon_years` is
    `plan_to_age - age` and `initial_value` is the sum of the balances. Both used
    to be stored, and both had drifted — `Portfolio Analysis.py` planned to age
    87 while `run.py` planned to 95, and the since-deleted `tools/account_allocation.py`'s docstring
    disagreed with its own dict about the 401k Roth / Roth IRA split.
  - **Name the age the contributions were measured at.**
    `contributions_measured_at_age` replaces a bare literal 36 in
    `engine.calculate_annual_contribution`. It defaults to 36, so every record
    written before the field existed reproduces exactly. Bumping your age in the
    editor prompts to move it too — that is the whole point of the prompt.

  `investor.save` refuses a record that fails validation rather than let a
  silently wrong config produce confidently wrong runs. The investor facts now
  reach run provenance as flat scalars plus an accounts fingerprint, so
  `run.py --diff` attributes a change to a config edit instead of reporting
  "provenance identical".

- **`montecarlo/engine.py`**: The simulation. Parameter loading (deflated to real), regime machinery,
  correlated returns, rebalancing bands, monthly loop, withdrawals, multiprocessing driver.
  Prints nothing, imports no plotting library. `REBALANCING_BANDS` lives here, not in the
  runner — under `spawn` a worker re-imports the module that defines them, so a value set in
  the parent would not reach workers.

- **`montecarlo/policies.py`**: 4 withdrawal policies (constant_real, fixed_pct, guyton_klinger, vpw)
  and 2 sourcing rules (pro_rata, excess). Pure functions, all in real dollars.

- **`montecarlo/solver.py`**: Inverse solver — "at 90% confidence, what PLAN LEVEL does this
  support?" Holds CONFIDENCE constant rather than a dollar bar, so strategies are
  comparable. Three things to know:
  - The answer is a **plan level**, not a forecast of spending. Realised spending runs
    ABOVE it in the median and below in bad cases, by margins that differ per policy.
    Rank on `report_retirement`'s life-stage table, not on this number.
  - Adequacy is the **worst rolling 10-year window** (`solver.DEFAULT_ADEQUACY = 'window'`,
    since 2026-10-21): every 10-year average of spending must clear the floor, not just the
    30-year mean. The lifetime mean was shape-blind — a bad first decade could be repaid
    by a rich ninth, and 12.5% of passing `vpw+5M` paths did exactly that. The rule before
    it, THIRDS of the retirement (`'thirds'`, `'per_band'` in old records), made the
    rule's strictness move with the span: 165 certified 2.5% more to 100 than to 95 on
    bit-identical paths (ledger entry N11, unpublished). A window is equally strict at every horizon and
    retirement age, and a longer plan never certifies more. A span under 10 years is one
    window. `config['adequacy'] = 'thirds'` or `'lifetime'` replays older runs; levels
    dated before 2026-10-21 were certified on thirds.
  - The crossing is **estimated, not bisected**, and the level is set on the VERIFICATION
    sample. Bisection read the search sample, which is a subset of the verification paths
    and therefore leans the same way at every spend — worth -17% on guyton_klinger.
    `vpw` is solved in closed form because its paths do not move with the target.

- **`montecarlo/glide.py`**: Allocation schedules over age. Waypoints are STEPS, not interpolated, so
  there is no new allocation maths. Bond ladders are SYNTHESISED from a base portfolio's
  own equity sleeve (`141+10`, `141+10:bnd_only`) rather than looked up as sibling rows —
  the old lookup silently swapped funds as well as adding bonds.

- **`montecarlo/persistence.py`**: A JSON record per run in `output/runs/`, with provenance
  (parameter date, inflation, premium scale, crisis scale). `run.py --history` and
  `--diff` compare runs and attribute a change to its cause.

- **`tools/sensitivity.py`**: ⭐ **The single sweep tool.** One registry
  (`DIMENSIONS`), 24 assumptions, grouped in the menu by WHO CONTROLS THEM —
  returns / cash / inflation / confidence / your life are unknowable, `your
  plan` is what you decide. Reports whether an assumption moves the LEVEL or the
  RANKING. Run it bare for the menu; `run.py`'s Market & life assumptions drives
  the same registry, minus `your plan`, which the Retirement choices route
  compares on shared paths instead. Those choices are applied through
  `run.apply_choice` in both places, so a sweep and a choice cannot disagree
  about what 'retire at 60' means.

  Two report styles, because the distinction is real: a **numeric** sweep has an
  ordering so a SPAN means something; a **named** one (scenarios, regions,
  rf_source, mu_method) does not, and reporting a span over labels would be
  arithmetic on nothing.

  Factor scenarios are the `scenario` dimension (the old
  `tools/factor_scenarios.py` was folded in and deleted). The merge also added
  **`regions`**, which that file's own docstring had flagged as the gap: every style scenario left US at 7.19%
  against developed 4.87%, so a US-heavy portfolio won every column partly for a
  reason nothing tested. `us stays ahead` / `intl catches up` / `regions
  converge` / `intl keeps lagging` make that a choice. The last holds the
  US's 2005-10..2026-07 lead over each region (`factors.RECENT_LAG`) — a GAP,
  because over that window developed earned MORE than its long-run premium;
  the US ran ahead.

- **`montecarlo/return_views.py`**: "If the return model is wrong". A portfolio comparison
  can repeat itself, at the same plan and on the same paths, under four other
  return models (intl keeps lagging, intl catches up, no factor premia,
  historical means), and scores each strategy by REGRET against the best in
  each view. It prints the minimax pick, which strategies are never resolved
  behind the best, and your plan's success under each view. Each view is a
  child process with its env vars set, because workers re-import under
  `spawn`. Children run the same seeds, so the views are paired with the main
  run. Opt-in per run (`"return_views": true`; the prompt defaults to yes),
  because it adds one evaluation of every strategy per view.

- **`montecarlo/report_retirement.py`** / **`montecarlo/report.py`**: Decumulation tables + figures / the legacy
  five accumulation figures.

- **`montecarlo/report.py`**: The five accumulation figures + the rebalancing summary. Called
  by `report_retirement.accumulation_figures`; `Portfolio Analysis.py`, which used to be
  its only caller, has been deleted.

- **`montecarlo/loaders.py`**: Helpers to load JSON parameters + correlation matrices (from `output/`, with staleness warnings) and portfolio templates (from `portfolios/portfolio_templates.csv`). Comment/blank rows in the CSV are skipped.

- **`tools/update_holdings.py`**: Rewrite a single portfolio row in `portfolio_templates.csv` from percent-per-fund input (normalizes to sum 1.0; a fund listed twice — one fund in two accounts — is added). Useful when account placement makes your live allocation a blend. It and the other CSV tools write LF, as the file is committed; csv.writer's CRLF default had turned every one-row edit into a whole-file diff.

**See [docs/GUIDE.md](../docs/GUIDE.md) for running it and [docs/METHODOLOGY.md](../docs/METHODOLOGY.md) for how it is built.**

### Correlation System

Four matrices, one per market regime plus the full history:
- `correlation_baseline_N.npy` - full-history N×N matrix (N = ticker count, currently 27)
- `correlation_low_N.npy` - calm market correlations (30% of history)
- `correlation_medium_N.npy` - normal correlations (40% of history)
- `correlation_high_N.npy` - crisis correlations (30% of history)

**Which months count as a crisis is decided by `metadata.market_regime`** — the
market proxy's own 12-month volatility, cut at p30/p70 over its full history.
One definition, written by the pipeline and simulated by the engine, because
until 2026-09-10 there were three and they disagreed.

Re-measured on that labelling: **VOO-VEA 0.654 → 0.914, +40%**, and **292 of
300 pairs rise in the high regime** (mean +0.252; on the 27-ticker 2026-10-25
matrices, 326 of 351, mean +0.198, VOO-VEA 0.653 → 0.914). The long-quoted 0.64 → 0.91
is essentially right and had drifted out of reach of the code that produced it.
Under the labelling in force the day before, the same data read 0.737 → 0.829.

### Multiprocessing Strategy

Paths run on one `multiprocessing.Pool(cpu_count())`, created once and reused
(`solver._pool`). A solve searches on `solver.SEARCH_SIMS` (400) paths and
sets the level on `solver.VERIFY_SIMS` (1,500); the runner offers 2x and 4x,
and `--dry-run` prints the time estimate before anything runs.

## Common Commands

### ⭐ The three steps ⭐
```bash
# Step 1: Update parameters with latest market data (quarterly recommended)
python3 data_collection_regime_model.py
# Downloads 27 actual tickers + proxies from yfinance + Ken French/AQR data
# Outputs to output/: ticker_parameters_regimes_v8.json (~70KB) + 4 correlation matrices
# Auto-runs the parameter sanity validator at the end
# Time: about 10 minutes, mostly downloads

# Step 2: Validate (structural + data + behavior gate; also run after any engine edit)
python3 tools/validate_all.py
# All-PASS required before a production run. After an intentional engine change,
# re-bless the regression baseline: python3 tools/regression_check.py --update

# Step 3: Run it
python3 run.py            # interactive — the normal way
python3 run.py --again    # repeat the last run


# Premium sensitivity — the single most consequential knob:
MC_PREMIUM_SCALE=0.5 python3 run.py     # half the historical factor premium
MC_PREMIUM_SCALE=0.0 python3 run.py     # null hypothesis: no premium at all
```

## Where a question goes — three routes, by what CHANGES

**To design a run or hand the user a command, load the `design-runs` skill
first** (`.claude/skills/design-runs/SKILL.md`): the full `--spec` reference,
the schedule syntax, a working recipe per question type, how to add a
portfolio, and which table is the verdict. `check_logic` validates every
command in it, so its recipes cannot drift from `run.py`.

Every question this model answers is one of three, and a new check belongs to
the route whose changing thing it is about:

- **Portfolios**: portfolios, bond sleeves or glides change; your plan stays
  fixed. Answer: the paired head-to-head at ONE plan, then whether the winner
  survives a wrong return model (`montecarlo/return_views.py`).
- **Retirement choices**: retirement age, claiming age, contributions or the
  withdrawal policy change; one portfolio stays fixed. Here the LEVEL is the
  answer. "Level-only, doesn't change the decision" is the sensitivity tool's
  verdict about assumptions; applied to a choice you make, it is backwards.
  Each option is solved and read against YOUR current value, paired on the
  same paths (`compare_choice` in a spec; `run.PLAN_CHOICES`). The withdrawal
  rule goes through the older `compare_policies` fan-out.
- **Market assumptions**: things you do not control (returns, cash rate,
  inflation, confidence). Answer: how far the figures could move, and whether
  any of it reorders the portfolios (`tools/sensitivity.py`).

A return model is an assumption, so sweeping it is a market-assumptions
question. Asking whether the PORTFOLIO choice survives it is a portfolios
question, which is why the return-model check lives in that route and not in
sensitivity.

## ⭐ What is already settled — do not re-derive it

The settled conclusions, in relative terms, are in [docs/RESULTS.md](../docs/RESULTS.md). Read the section covering what you are about to touch before re-deriving it.

## Current File Structure

### Core (root entry points + the `montecarlo/` library)
- **`run.py`** - ⭐ Interactive runner (RUN). `--spec '{"portfolios":["140"]}'`
  runs headlessly from the SAME spec `--again` replays and `persistence` records;
  `--dry-run` resolves and costs nothing. Use it for any measurement you intend
  to quote, so the next person can reproduce it without re-answering prompts.
- **`montecarlo/ask.py`** - `run.py --ask "sentence"`: Claude Haiku proposes a run spec,
  validated exactly as `--spec` is, shown with its cost and the replayable
  `--spec` line, and run only on a yes. Optional — needs the `anthropic`
  package; nothing else imports it.
- **`montecarlo/investor.py`** - The facts about YOU; loads/validates `config/investor.json`
- **`data_collection_regime_model.py`** - Data collection pipeline (RUN, quarterly)
- **`montecarlo/engine.py`** - Simulation: parameters, regimes, returns, bands, monthly loop, withdrawals
- **`montecarlo/policies.py`** - Withdrawal policies + sourcing rules
- **`montecarlo/solver.py`** - Inverse solver for the plan level (worst-10-year-window adequacy)
- **`montecarlo/glide.py`** - Allocation schedules over age
- **`montecarlo/report_retirement.py`** - Decumulation tables + figures
- **`montecarlo/return_views.py`** - The portfolio comparison re-run under other return models
- **`montecarlo/report.py`** - The five legacy accumulation figures
- **`montecarlo/diagnostics.py`** - The four `--debug` blocks: which life stage set the level,
  when the failing paths failed, regime occupancy, and epistemic vs aleatoric.
  Silent on a normal run; the per-path data rides on every run regardless
  (six numbers), so `--debug` decides whether to SHOW it — a module global could
  not decide whether to COMPUTE it, because workers re-import under `spawn`.
- **`montecarlo/loaders.py`** - Load JSON/NPY parameters + CSV portfolios; `MODEL_TICKERS` lives here
- **`portfolios/portfolio_templates.csv`** - The portfolio library (data); `portfolios/portfolio_archive.csv` holds settled rows
- **`config/runs.json`** - Saved run presets (written by `run.py`)
- **`config/investor.json`** - Age, plan-to age, retirement age, the five
  accounts (balance, contribution, growth, stop age, tax treatment) and Social
  Security. Edited from `run.py`'s Your details, never by hand-patching source.

### tools/ — occasional maintenance scripts (run from the project root)
- **`tools/single_stock_benchmark.py`** - An annual single-stock check: COST's rolling 10-year excess over its six-factor build-up (the model's own recipe, `factors.fit_single_stock`) against the exit trigger (−5%/yr; `--threshold`).
- **`tools/update_holdings.py`** - Update one portfolio row from percent-per-fund input (interactive/paste)
- **`tools/annotate_templates.py`** - Append/refresh derived navigation columns on portfolio_templates.csv (idempotent)
- **`tools/add_section_headers.py`** - Insert/refresh '#' section header rows (bond tier + tilt family) for human scanning (idempotent; needs tilt_family column)
- **`tools/check_consistency.py`** - Verify the ticker universe agrees across all files (run after any ticker change; see docs/RUNBOOK_tickers.md)
- **`tools/new_template.py`** - Scaffold a fresh portfolio_templates.csv for a new ticker universe (backs up the old one)
- **`tools/estimability.py`** - ⭐ Demonstrates §1 — why a fund's expected return cannot be recovered from its returns — on this repo's own data. A century of market data leaves a 7.2pp interval; ±1pp would take 338 years; and sampling daily vs quarterly leaves SE(mean) FLAT while SE(vol) shrinks 7x. Read it before proposing a better estimate of the level.
- **`tools/solver_calibration.py`** - Re-score a solved plan level on an INDEPENDENT sample (the solver sets it on its own verification sample, so that sample cannot fail it). `--convergence 400,800,1600,3200` reports how much the level itself moves with path count — which is how portfolio 102 was found to be unconverged.
- **`tools/premia.py`** - Measure each factor premium on its OWN window (not the intersection of all six) and print the `PREMIA` / `PREMIUM_SE` / `PREMIUM_WINDOW` tables for `montecarlo/factors.py`. `--long` extends US market + value to 1926.
- **`tools/proxy_score.py`** - Score every backfill proxy against the fund it stands in for (corr, β, tracking error, and `used` — the months it actually reaches the spliced series). Found the international-proxy rotation; the pipeline now stores these scores and `validate_parameters.py` gates them.

**Validation harness** (run `python3 tools/validate_all.py` after any data/engine change — it chains all of these):
- **`tools/check_parameter_drift.py`** - ⭐ **The golden master FOR DATA.** Compares the regenerated parameters against a committed baseline (`tools/parameter_baseline.json`) and fails on a move larger than a data refresh should produce. `--update` to re-bless, `--verbose` to see every move. This is a different question from `validate_parameters`, and the gap between them is where both 2026-09-10/11 defects lived: every one of VEA's numbers was individually plausible while it carried EMERGING market volatility for 41% of its history. Verified against the real thing — replaying the day's fixes as one regen, it raises 5 failures including VEA's regime vol moving 5.67pp. `history_months` can only GROW; any decrease fails at any size, because that is a splice dropping data (the third bug of 2026-09-09). **Tolerances are sized from MEASURED refresh drift, not guessed** — the 2026-09-09 double-regen isolated a refresh at ≤0.11pp (COST 0.54pp), so μ gets a 1pp band, ×3 for rolling-window tickers which genuinely churn. The first version used 2pp and was too loose to do its job: the VOO price-index bug moved μ 1.70pp and would have passed silently. A guard calibrated above the bug it is meant to catch is theatre.
- **`tools/validate_parameters.py`** - Data sanity gate on the generated JSON + correlation matrices (finite values, plausible μ/σ bounds, regime ordering, valid Markov rows, PSD correlations, **proxy fidelity**). Auto-runs at the end of `data_collection_regime_model.py`.
- **`tools/check_templates.py`** - Portfolio lint: every row sums to 1.0, non-negative, and derived nav columns match `compute_derived()`.
- **`tools/check_logic.py`** - Plain-assert unit checks for pure logic (band math, `compute_derived`, `near_psd_higham`, regime transitions, contributions) **plus two END-TO-END SMOKE TESTS** that drive `run.build_config` → `resolve_strategies` → `report_retirement.run` for both modes at 24 paths. Everything else here tests a function; these test that the functions are still WIRED TOGETHER — `evaluate_strategies` was covered only by inspecting its source, and `report_retirement.run` was never called at all, so a refactor breaking the hand-off passed all 224 checks. They assert SECTION STRUCTURE, not a digit count: a digit heuristic was tried and was too weak — emptying `_flat_table` still passed it, which is the exact silent-missing-table failure they exist to catch.
- **`tools/regression_check.py`** - Engine golden master on a synthetic universe (accumulation, decumulation, and one case per feature production turns on: banded rebalancing, a glide switch, Social Security, care costs, shared parameter errors) + determinism + analytic-compounding anchor; `--update` to re-bless the baseline (`tools/regression_baseline.json`), and it refuses to bless a run whose own checks fail.
- (`tools/_engine.py` was deleted — `montecarlo/engine.py` imports normally, so the importlib shim and
  the module-global monkey-patching it required are gone.)
- **`tools/validate_all.py`** - Runs consistency + the five validators above in order; the single pre-commit / pre-run gate.
- **`tools/vacuity_sweep.py`** - ⭐ Proves the SUITE can fail. `--static` flags tests
  that cannot fail by construction; `--mutate` guts a function and counts the checks
  that notice; `--subtle` introduces WRONG ANSWERS, which is the class this repo has
  actually shipped — it found three uncovered on its first run. NOT part of
  `validate_all` (minutes, not seconds), and `--mutate` rewrites source in place, so
  do not run anything else against the checkout while it works.

Tools add the project root to `sys.path` and anchor data paths to it, so run them as `python3 tools/<name>.py` from the project root.

**Derived columns:** `portfolio_templates.csv` carries 8 navigation columns after the 27 ticker columns (`bond_pct, us_pct, dm_share, tilt_family, mom_fund, val_fund, n_funds, max_single_fund`), computed from allocations by `compute_derived()` in `montecarlo/loaders.py`. The loader/simulator ignore them (weights are parsed by `MODEL_TICKERS` allowlist); the holdings-update tools auto-refresh them. Re-run `tools/annotate_templates.py` after bulk template edits.

### Generated Data Files (output/ subdirectory)
- **`ticker_parameters_regimes_v8.json`** (~70KB) - All 27 ticker parameters with metadata
- **`correlation_baseline_N.npy`** (~6KB) - Baseline N×N correlation matrix (N = ticker count)
- **`correlation_low_N.npy`** - Low volatility regime correlations
- **`correlation_medium_N.npy`** - Medium volatility regime correlations
- **`correlation_high_N.npy`** - High volatility regime correlations

### Reference Data (reference/ subdirectory - static)
- Ken French CSV files (10): the US, developed ex-US and emerging factor sets, and
  the 25/32-portfolio sorts the factor funds' backfills are blended from
- AQR Quality Minus Junk Excel file (JQUA's backfill)

**Everything in `output/` is regenerated by running `data_collection_regime_model.py`** (quarterly); `reference/` is refreshed by hand.

## Technical Concepts

### Regime-Switching Volatility

Assets transition between LOW/MED/HIGH volatility regimes **monthly** using Markov chains,
estimated from a rolling **12-month** volatility (36 until 2026-09-10 — which labelled
October 2008 "medium"; see roadmap item 5). Regime volatility is the REALISED volatility
of the months in each bucket, not a percentile of the rolling estimator.

Diagonals now run ~0.86–0.94, giving regime durations of roughly **6–13 months**.

**Each ticker's transition is conditioned on the MARKET regime it is moving into**
(item 5b). Across the universe, P(medium → high) is 0.019 / 0.048 / 0.271 as the market
goes low / medium / high. The conditional estimate is shrunk toward each ticker's
unconditional row by 10 pseudo-counts — without that it scores WORSE than no coupling at
all, because most conditional cells are nearly empty.

Implementation: `select_next_regime(ticker, current_regime)` in `montecarlo/engine.py`.

**The CORRELATION matrix is not selected by these.** It follows a separate
MARKET regime (`metadata.market_regime`) — the market proxy's own 12-month
volatility, its own Markov chain, simulated alongside. Per-ticker regimes set
each asset's volatility; the market regime sets the correlation structure and
the tail-shock flag. Before 2026-09-10 the engine majority-voted the per-ticker
regimes instead, which did not separate medium from high at all. See roadmap
items 5 and 6 — including why the 36-month window used for the per-ticker
regimes calls October 2008 "medium".

### Data Splicing - Key Details

**Multi-stage splicing** for maximum history:
- **AVUV**: Ken French (1963-1993) + DFSVX (1993-2019) + actual (2019+) = ~759 months
- **Factor funds** (VTV, AVLV, MTUM, VFMO, JQUA): Ken French (1973+) + actual = ~646 months
- **Standard ETFs**: yfinance proxy + actual = varies by inception
- **DFWIX**: Blended proxy (78% DFIEX + 22% DFCEX, 2005-2013) + actual (2013+) = ~253 months
- **COST**: Rolling 20-year window (240 months) - NO splicing, actual stock data only
- **AVEE**: Rolling 20-year window (240 months) - proxy-spliced (DEMSX + actual), early data unrepresentative

**Single-stock policy:** tickers in `loaders.SINGLE_STOCKS` (COST today) automatically get the 20-yr rolling window **and** an inflated parameter-uncertainty band (`0.25× vol` vs the `0.15×` default) for idiosyncratic/survivorship risk. Their expected return is their six US factor loadings over that window plus `MC_SINGLE_STOCK_EDGE` (default 0), never their historical mean (`factors.fit_single_stock`, ledger entry N10, unpublished); `validate_parameters` fails a single stock that leads VOO by more. Don't also list them in `PARAM_STD_OVERRIDE` (bypasses the bump; `check_consistency.py` flags it).

**See [docs/METHODOLOGY.md](../docs/METHODOLOGY.md) → "Data Splicing Methodology".**

### Portfolio Templates

Portfolio configurations live in `portfolios/portfolio_templates.csv` (one row per portfolio):
- **100s-600s**: Organized by equity/bond allocation (100%/0% to 75%/25%) — the hundreds digit
- **Within each hundred**: tens digit = tilt (x0 core, x2 value, x4 momentum, x5 momentum/intl-experiments, x6 mom+val, x8 quality); 140=base, 141=your holdings (the row `tools/update_holdings*` overwrite; it ships holding the author's allocation as an example); 102/103/122 are starters with no single stock
- Section header rows + derived nav columns make it scannable (see tools/ above)
- **Archived rows** live in `portfolios/portfolio_archive.csv` (same columns): settled
  questions kept restorable — VTV variants, the COST twins of 141/165, the
  international-structure experiments on the old base, 161's regional variants,
  and the pre-168 VFMO/AVUV twins (2026-10-25). Running an archived id fails
  with a pointer to the file; move the row back to use it. the ledger (unpublished) still quotes
  them by id.
- **168's twins** — each moves one fund's weight into VOO: 169 (AVLV), 170
  (COST), 171 (VFMO), 172 (AVUV); 173 swaps VFMO for MTUM.

**See [docs/GUIDE.md](../docs/GUIDE.md) → "Portfolio Templates" for the full naming legend.**

## Important Constraints

1. **Allocation Validation:** All portfolio allocations must sum to exactly 1.0 (enforced in code)
2. **Ticker Consistency:** The ticker universe is defined ONCE in `loaders.MODEL_TICKERS`; `data_collection_regime_model.py` imports it (`from montecarlo.loaders import MODEL_TICKERS as TICKERS`). To add/remove a ticker, edit `MODEL_TICKERS`, follow **[docs/RUNBOOK_tickers.md](../docs/RUNBOOK_tickers.md)**, and verify with `python3 tools/check_consistency.py` (confirms the consumers don't redefine the list, and that JSON, correlation dims, CSV columns, and role classification all agree).
3. **Correlation Matrix Dimensions:** Must be N×N matching TICKERS list order (N = ticker count)
4. **Regime Classification:** Only 'low', 'medium', 'high' are valid (case-sensitive)
5. **Multiprocessing:** Cannot nest Pool() calls - use sequential scenarios with parallel simulations within each
6. **Random Seeds:** Use `np.random.seed(42)` for reproducibility but beware multiprocessing may break determinism
7. **No Negative Final Values:** Model does not handle bankruptcy/margin calls; assumes continuous positive wealth
8. **File Locations:** Parameter/correlation files in `output/` subdirectory; reference data in `reference/`

## Documentation References

- **The measurement ledger:** [docs/RESULTS.md](../docs/RESULTS.md) - what has been measured, what was wrong, what is ruled out. Read the section covering what you are about to touch.
- **The docs:** [README.md](../README.md) is the front page; [docs/GUIDE.md](../docs/GUIDE.md) - workflow, your details, reading the output, troubleshooting, the portfolio naming legend; [docs/METHODOLOGY.md](../docs/METHODOLOGY.md) - universe, regimes, correlations, splicing, parameter uncertainty; [CHANGELOG.md](../CHANGELOG.md)
- **Add/remove a ticker:** [docs/RUNBOOK_tickers.md](../docs/RUNBOOK_tickers.md) - every coupling point + the consistency checker
- **Source data:** `reference/` - Ken French / AQR academic datasets (read by data_collection_regime_model.py)
- **How returns are forecast:** [docs/METHODOLOGY.md](../docs/METHODOLOGY.md) → Expected Returns - why per-ticker historical means were incoherent, and the factor build-up that replaced them (the default since 2026-09-10). Only a valuation anchor remains unbuilt, by choice.
- **The data map:** [docs/ERD.md](../docs/ERD.md) - every file, record and registry the engine and workflow pass between them, as Mermaid ERDs. **Keep it current:** a change that adds, removes or renames an entity, a stored field or a link between them — a new file in `output/`, a key in a run record or its provenance, a field in `config/investor.json`, a CSV column, an entry in a registry (`DIMENSIONS`, `PLAN_CHOICES`, the withdrawal policies), a new route — updates `ERD.md` in the same commit. Edit only the `.md`; the `erd` skill rebuilds `docs/ERD.html` from it and republishes the artifact to the URL in its front matter. Logic and parameter changes that leave the shapes alone don't touch it.

