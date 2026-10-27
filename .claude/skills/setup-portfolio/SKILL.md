---
name: setup-portfolio
description: This skill should be used when the user wants to set up this Monte Carlo model for their OWN portfolio or ticker set — e.g. "set this up for my portfolio", "use my own tickers", "customize the tickers", "I want to run this with different funds". It drives the full ticker-universe swap and builds fresh sample portfolios, gated by tools/check_consistency.py.
---

# Set Up the Model for Your Own Portfolio

Drive the user through replacing the default ticker universe with their own and
building sample portfolios. This skill orchestrates and enforces the checkpoints;
the detailed procedure lives in `docs/CUSTOMIZING.md` (the single source of truth).

## Procedure

1. **Run it as-is first** — prove the toolchain works before changing anything:
   ```bash
   pip install -r requirements.txt
   python3 data_collection_regime_model.py
   python3 run.py
   ```

2. **Gather the user's decisions** before editing (do not guess these):
   - The ticker list they want.
   - Per ticker: the **proxy / backtest window** (a longer-history proxy vs the
     ticker's own history) — see `docs/CUSTOMIZING.md` → "Backtest window & history".
   - Any **new asset class** (gold, REIT, crypto) that needs a new role bucket, or
     a decision to drop the derived nav columns.
   - Any **single stocks** (individual companies, not funds) — add them to
     `SINGLE_STOCKS` in `montecarlo/loaders.py` so they get the 20-yr window + inflated
     parameter uncertainty (`0.25× vol`) for idiosyncratic/survivorship risk.
   - Their **portfolio strategy / risk level** for the sample portfolios.

3. **Follow `docs/CUSTOMIZING.md` step by step.** In short: edit
   `loaders.MODEL_TICKERS` (the one ticker list) + role buckets; set `PROXIES` and
   any window treatment in `data_collection_regime_model.py`; add rebalance bands in
   `montecarlo/engine.py`; regenerate data; scaffold portfolios with
   `python3 tools/new_template.py`, then `python3 tools/annotate_templates.py` and
   `python3 tools/add_section_headers.py`.

4. **Hard gate** — run `python3 tools/check_consistency.py`. It must print all PASS.
   Iterate edits until green; do not proceed otherwise.

5. **Smoke run** — `python3 run.py` (option 2, Allocations, is quickest).

## Hard rules

- `tools/check_consistency.py` MUST be all-PASS before declaring done — it is the
  oracle that catches a missed coupling point.
- **Append, never reorder** existing tickers in `loaders.MODEL_TICKERS` — order
  defines correlation-matrix rows.
- Never hand-edit `output/` — always regenerate via `data_collection`.

## Reference

- Detailed procedure: `docs/CUSTOMIZING.md`
- Exhaustive per-file coupling map: `docs/RUNBOOK_tickers.md`
- For changing a single ticker later, use the `add-ticker` skill.
