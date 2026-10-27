---
name: add-ticker
description: This skill should be used when the user wants to add or remove a SINGLE ticker/fund in the existing model — e.g. "add a ticker", "remove a fund", "add VTI to the model", "drop COST". For a full universe swap, use the setup-portfolio skill instead.
argument-hint: [ticker]
---

# Add or Remove a Ticker

Drive an incremental, single-ticker change to the model. This skill orchestrates
and enforces the checkpoints; the exhaustive per-file coupling map lives in
`docs/RUNBOOK_tickers.md` (the single source of truth).

## Procedure (adding ticker `XYZW`)

1. **Universe + role** — in `montecarlo/loaders.py`: add `XYZW` to `MODEL_TICKERS`
   (**append, never reorder**), and classify it into exactly one role bucket
   (`_US_FUNDS` / `_BOND_FUNDS` / `_DM_FUNDS` / `_EM_FUNDS`). The two consumers
   (`data_collection_regime_model`, `engine`) import this list automatically.
   **DECISION — single company (not a fund)?** If so, also add it to `SINGLE_STOCKS`
   in `montecarlo/loaders.py` → it auto-gets a 20-yr window + inflated parameter uncertainty
   (`0.25× vol`) for idiosyncratic/survivorship risk. (Don't also put it in
   `PARAM_STD_OVERRIDE` — that bypasses the bump; the checker flags it.)
2. **History / backtest window** — in `data_collection_regime_model.py`: add a
   `PROXIES` entry. The proxy IS the window decision — ask the user whether to use a
   longer-history proxy or `None`. If early history is unrepresentative, add the
   ticker to `ROLLING_WINDOW_TICKERS`. See `docs/CUSTOMIZING.md` → "Backtest window
   & history" for the reasoning.
3. **Sim config** — in `montecarlo/engine.py`: add a `REBALANCING_BANDS` entry (scale it
   with the ticker's volatility — the existing table correlates +0.86 with vol).
   If it's a bond/cash fund, add it to `_BOND_FUNDS` in `montecarlo/loaders.py`, which also
   sets its inflation pass-through.
4. **Regenerate** — `python3 data_collection_regime_model.py`; delete the stale
   `output/correlation_*_<oldN>.npy`.
5. **Template column** — add an `XYZW` column to `portfolios/portfolio_templates.csv` (0 for
   existing portfolios), then `python3 tools/annotate_templates.py` and
   `python3 tools/add_section_headers.py`.
6. **Hard gate** — `python3 tools/validate_all.py` must be all-PASS. It chains the
   structural check (`check_consistency.py`) with the data + behavior validators
   (`validate_parameters.py`, `check_templates.py`, `check_logic.py`,
   `regression_check.py`). `data_collection_regime_model.py` also runs the parameter
   validator automatically at the end of step 4.

To **remove** a ticker: clear its weights from all portfolios first, then reverse
steps 1–5 (remove from `MODEL_TICKERS` + role, `PROXIES`, bands, CSV column),
regenerate, and gate on the checker.

## Hard rules

- `tools/validate_all.py` MUST be all-PASS before declaring done (it includes
  `check_consistency.py` plus the data/behavior validators).
- **Append, never reorder** existing tickers in `MODEL_TICKERS`.
- Never hand-edit `output/` — always regenerate.

## Reference

- Exhaustive coupling map: `docs/RUNBOOK_tickers.md`
- Backtest-window reasoning: `docs/CUSTOMIZING.md`
