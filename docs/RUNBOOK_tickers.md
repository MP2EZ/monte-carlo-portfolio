# Runbook: Adding / Removing a Ticker

Changing the ticker universe is the highest-risk edit in this system, but the
list itself now lives in **one place**: `MODEL_TICKERS` in `montecarlo/loaders.py`.
`data_collection_regime_model.py` does
`from loaders import MODEL_TICKERS as TICKERS`, so it can't drift from it. What
still needs your attention per ticker is the *supporting* config — proxies, role
classification, rebalance bands — plus the generated data and the template CSV.
The order in `MODEL_TICKERS` is canonical (it defines JSON & correlation-matrix
row order), and a run hard-fails if the JSON doesn't match it.

> **Always finish with `python3 tools/validate_all.py`.** It runs the structural
> check (`check_consistency.py`) plus the data + behavior validators
> (`validate_parameters.py`, `check_templates.py`, `check_logic.py`,
> `regression_check.py`) and exits non-zero if anything is out of sync OR if a
> generated number is implausible. It's the safety net that makes this runbook
> reliable — a missed step or a bad splice shows up there, not in a silently-wrong
> simulation.

---

## Where tickers live (coupling map)

**Hand-edited (sources of truth):**

| File | What | Required? | Order matters? |
|---|---|---|---|
| `montecarlo/loaders.py` | **`MODEL_TICKERS`** (≈L45) — **THE single source of truth** | **yes** | **yes** — defines JSON & matrix order |
| `montecarlo/loaders.py` | role bucket `_US_/_BOND_/_DM_/_EM_FUNDS` (≈L59–62) | **yes** — classify the ticker | no |
| `montecarlo/loaders.py` | `SINGLE_STOCKS` | only if the ticker is a single company → auto 20-yr window + 0.25× param_std | no |
| `data_collection_regime_model.py` | `PROXIES` (≈L71) | **yes** — yfinance history proxy, or `None` | no |
| `data_collection_regime_model.py` | `BLENDED_PROXIES`, `KF_TICKERS`, `KF_BLENDS`, `COST_ADJUSTMENTS`, `ROLLING_WINDOW_TICKERS`, `PARAM_STD_OVERRIDE` | only if applicable to the fund type | no |
| `montecarlo/engine.py` | `REBALANCING_BANDS` drift-band dict (≈L38) | **yes** — `{'relative':_, 'absolute':_}` | no |
| `montecarlo/loaders.py` | `_BOND_FUNDS` list (≈L73) | only for bond/cash funds | no |
| `montecarlo/loaders.py` | `INFLATION_LINKED` (TIPS) or `BOND_DURATION` (nominal: the issuer's effective duration, years) | **yes, for every bond fund** — engine import fails on a nominal bond without a duration | no |
| `portfolios/portfolio_templates.csv` | the ticker **column** | **yes** | no (loader maps by name) |
| `InvestmentTracker.xlsx` (outside the repo) | the Allocation tab's `Ticker` row | only to pull the ticker via the xlsx updater | no (symbols are read as written) |
| `CLAUDE.md`, `README.md`, `docs/METHODOLOGY.md` | ticker counts/list | cosmetic | no |

**Generated — never hand-edit, always regenerate:**

| File | Produced by |
|---|---|
| `output/ticker_parameters_regimes_v8.json` | `data_collection_regime_model.py` |
| `output/correlation_{baseline,low,medium,high}_<N>.npy` | `data_collection_regime_model.py` |

---

## ADD a ticker (call it `XYZW`)

1. **`montecarlo/loaders.py` — the universe + its classification (the only list to touch)**
   - Add `XYZW` to **`MODEL_TICKERS`**. **Append at the end** — appending avoids
     reordering existing correlation-matrix rows. This one edit propagates to
     `data_collection` and `Portfolio Analysis` automatically (they import it).
   - **Classify it into exactly one role bucket** (`_US_FUNDS`, `_BOND_FUNDS`,
     `_DM_FUNDS`, or `_EM_FUNDS`). Easiest step to forget, and it fails *silently*:
     an unclassified ticker still loads, but `us_pct`/`bond_pct`/`dm_share` come
     out wrong and the row stops reconciling to 1.0. (DM/EM blend? Handle it in
     `compute_derived` like DFWIX.)
   - **Single company (not a fund)?** Add it to **`SINGLE_STOCKS`**. It then
     auto-gets the 20-year rolling window and an inflated parameter-uncertainty band
     (`0.25× vol` vs `0.15×`) for idiosyncratic/survivorship risk. Don't also put it
     in `PARAM_STD_OVERRIDE` (that bypasses the bump; `check_consistency.py` flags it).

2. **`data_collection_regime_model.py` — where its history comes from**
   - Add a **`PROXIES`** entry: a yfinance symbol with long history to splice
     against (an index or older fund), or `None` if `XYZW` has enough history of
     its own. **Required** — collection will fail without it.
   - *If it's a factor fund* wanting academic history: add `KF_TICKERS` +
     `KF_BLENDS` + `COST_ADJUSTMENTS`. *If it's a broad/simple ETF:* skip these.
   - *Optional:* `ROLLING_WINDOW_TICKERS`, `PARAM_STD_OVERRIDE`, `BLENDED_PROXIES`.

3. **Regenerate data:** `python3 data_collection_regime_model.py`
   - Produces the JSON with `N+1` tickers and `correlation_*_<N+1>.npy`.
   - **Delete the old `correlation_*_<N>.npy`** (stale — the loader globs by N so
     it ignores them, but remove to avoid confusion).

4. **`montecarlo/engine.py` / `montecarlo/loaders.py` — per-ticker simulation config** (no list edit — it
   imports `MODEL_TICKERS`)
   - Add a **rebalance drift-band** entry in the per-ticker dict (≈L140–161):
     `'XYZW': {'relative': 0.20, 'absolute': 0.04}` (copy a similar fund's bands).
   - *If it's a bond/cash fund*, add it to the bond list at ≈L1671.

5. **`portfolio_templates.csv`**
   - Add a column header `XYZW`; set `0` for existing portfolios (they don't hold
     it yet), and real weights where you want it.
   - Refresh derived columns + section headers:
     `python3 tools/annotate_templates.py && python3 tools/add_section_headers.py`

6. **Optional downstream:** your `InvestmentTracker.xlsx` Allocation tab is
   **outside this repo** — to pull `XYZW` via the xlsx updater, its `Ticker` row
   must spell it exactly as `MODEL_TICKERS` does (there is no alias table) and its
   "Placed (%)" row must include it.

7. **Validate:** `python3 tools/validate_all.py` → must pass (structural + data +
   behavior). Then smoke-test: `python3 run.py` (it should load
   without the ticker-mismatch error).

8. **Docs (cosmetic):** bump the ticker counts in `CLAUDE.md` / `README.md` / `docs/METHODOLOGY.md`.

---

## REMOVE a ticker

Same map in reverse. Note removal **shrinks N**, so the correlation matrices
change dimension.

1. **Clear its use first.** In `portfolio_templates.csv`, confirm the `XYZW`
   column is `0` everywhere (or reassign those weights to other funds). Removing a
   ticker that a portfolio still holds drops that weight and breaks sum-to-1.0.
2. **`montecarlo/loaders.py`:** remove from `MODEL_TICKERS` and from its role bucket.
   (This drops it from `data_collection` and `Portfolio Analysis` automatically.)
3. **`data_collection_regime_model.py`:** remove `XYZW` from `PROXIES` and any of
   `KF_TICKERS`/`KF_BLENDS`/`COST_ADJUSTMENTS`/`ROLLING_WINDOW_TICKERS`/
   `PARAM_STD_OVERRIDE`/`BLENDED_PROXIES` that reference it.
4. **Regenerate:** `python3 data_collection_regime_model.py`. **Delete the stale
   `correlation_*_<oldN>.npy`.**
5. **`montecarlo/engine.py` / `montecarlo/loaders.py`:** remove its `REBALANCING_BANDS` entry and the bond-list
   entry (if applicable). No list edit — it imports `MODEL_TICKERS`.
6. **`portfolio_templates.csv`:** delete the `XYZW` column, then
   `python3 tools/annotate_templates.py && python3 tools/add_section_headers.py`.
7. **Optional downstream:** drop the column from your tracker's Allocation tab.
8. **Validate:** `python3 tools/check_consistency.py`; then a smoke run.

---

## Backtest window mechanics

How much history a ticker gets, and over what window its stats are estimated:

- **Global, fixed windows** (not per-ticker): expected return μ = mean from **1973
  → today** (or inception → today if later); base volatility σ = **2010 → today**;
  correlations = trailing **15 years**; regime vol = **36-month rolling** percentiles.
- **The one per-ticker lever is the proxy** (`PROXIES`) — it sets how far back the
  spliced history reaches, i.e. the effective backtest window. Longer proxy →
  less-noisy μ/σ; `None` → the ticker's own history only.
- **`ROLLING_WINDOW_TICKERS`** caps a ticker's estimate at the most recent **20
  years** — use when early history is unrepresentative (e.g. `AVEE`; single stocks
  are added automatically).
- **`PARAM_STD_OVERRIDE`** widens a ticker's estimation-uncertainty band (explicit
  value; wins over the defaults).
- **`SINGLE_STOCKS`** (in `montecarlo/loaders.py`) is the single-stock policy: membership
  auto-applies the 20-yr window **and** an inflated parameter-uncertainty band
  (`0.25× vol` vs the `0.15×` default) for idiosyncratic/survivorship risk.

For the *reasoning* (how to judge window adequacy, worked example), see
[CUSTOMIZING.md](CUSTOMIZING.md) → "Backtest window & history".

## Gotchas (the things that bite)

- **Order-sensitivity.** `MODEL_TICKERS` order defines JSON & correlation-matrix
  row order. **Append at the end**; never reorder existing tickers (it would
  scramble which asset each matrix row represents). Regenerate after any change.
- **Silent role failure.** A new ticker missing from a `_*_FUNDS` bucket loads
  fine but corrupts the derived nav columns. The checker flags unclassified
  tickers (check #6).
- **Stale matrices.** Changing N leaves `correlation_*_<oldN>.npy` behind. Harmless
  to the loader (it globs by current N) but delete them.
- **The tracker is external.** `InvestmentTracker.xlsx` lives outside the repo; the
  runbook can't change it. Update its Allocation tab yourself if you want the xlsx
  importer to see the new ticker.

## Rollback

Everything here is in git. To review: `git diff`. To abandon an in-progress change:
`git checkout -- <file>` (one file) or `git reset --hard HEAD` (everything), then
re-run `python3 data_collection_regime_model.py` to rebuild `output/`.
