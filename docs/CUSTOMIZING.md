# Customizing for Your Own Tickers (Claude-facilitated)

This is the playbook for running the model on a **different set of tickers** than
the original. It's written to be driven by **Claude Code**: clone the repo, open
the folder in Claude Code, and paste the starter prompt below — Claude follows
this file step by step and uses `tools/check_consistency.py` as the pass/fail gate.

## Starter prompt

> "I want to run this with my own tickers: **\<list them\>**. Follow
> `docs/CUSTOMIZING.md`: set up the universe, build me a few sample portfolios,
> and get `tools/check_consistency.py` green."

## Step 0 — Run it as-is first (don't skip)

Prove the toolchain works before changing anything:

```bash
pip install -r requirements.txt
python3 data_collection_regime_model.py
python3 run.py
```

## The flow Claude follows

1. **Universe** — edit `MODEL_TICKERS` in `montecarlo/loaders.py` (the single source of truth;
   `data_collection` and `Portfolio Analysis` import it). **Append, don't reorder**
   existing tickers — order defines correlation-matrix rows.
2. **Roles** — classify each ticker into one bucket in `montecarlo/loaders.py`
   (`_US_FUNDS` / `_BOND_FUNDS` / `_DM_FUNDS` / `_EM_FUNDS`).
   - **DECISION — ask the user:** a new asset class (gold, REIT, crypto) that fits
     none of the four → either add a new bucket and extend `compute_derived`, or
     drop the derived nav columns entirely (they're an optional convenience).
3. **History & backtest window** — in `data_collection_regime_model.py`. Read the
   **"Backtest window & history"** section below first — it's the most consequential
   modeling choice. Per ticker:
   - Set a `PROXIES` entry. **The proxy IS the backtest-window decision** (it sets how
     far back history splices). **DECISION — ask the user:** a longer-history proxy
     (older sibling fund/index) or `None` for the ticker's own history. Claude should
     *propose* proxies (it knows inception dates) and confirm.
   - **DECISION — judge window adequacy:** is there enough *representative* history?
     If the early part is unrepresentative of the fund today, add the ticker to
     `ROLLING_WINDOW_TICKERS` (caps the estimate at the most recent 20 years).
   - *Optional:* `PARAM_STD_OVERRIDE` (widen estimation-uncertainty for shaky funds),
     `BLENDED_PROXIES` (weighted multi-fund proxy).
   - **Delete** the `KF_TICKERS` / `KF_BLENDS` / `COST_ADJUSTMENTS` entries unless the
     user explicitly wants pre-inception academic factor history (most non-factor
     universes don't — deleting these *simplifies* setup).
4. **Per-ticker sim config**:
   - Add a rebalance drift-band per ticker to `REBALANCING_BANDS` in
     `montecarlo/engine.py` (≈L38; copy a similar one).
   - Update `_BOND_FUNDS` in `montecarlo/loaders.py` (≈L73) to the user's actual
     bond/cash tickers.
5. **Regenerate** — `python3 data_collection_regime_model.py`; delete the stale
   `output/correlation_*_<oldN>.npy`.
6. **Portfolios**:
   - `python3 tools/new_template.py` — backs up the old CSV and scaffolds a fresh
     one with the new ticker columns + an equal-weight example.
   - **DECISION — ask the user:** their risk level / strategy. Claude writes
     portfolio rows that sum to 1.0 using the new tickers.
   - `python3 tools/annotate_templates.py && python3 tools/add_section_headers.py`
7. **THE GATE** — `python3 tools/check_consistency.py`. Must print **all PASS**.
   Iterate edits until green; do not proceed otherwise.
8. **Smoke run** — `python3 tools/validate_all.py`, then `python3 run.py`.

## Backtest window & history

This is the part that most affects results, so do it deliberately. The model
estimates each ticker's return/risk from history. **Most windows are global and
fixed** — you do NOT set them per ticker, but you must know them to judge whether a
ticker has *enough* history:

| Estimate | Window (fixed, global) | Source |
|---|---|---|
| Expected return μ | mean of **1973 → today**, or **inception → today** if the ticker/proxy starts later | data_collection ≈L505 |
| Base volatility σ | **2010 → today** | data_collection ≈L513 |
| Correlation matrices | trailing **15 years** (180 months), recomputed each run | data_collection ≈L588 |
| Regime classification | **36-month rolling** volatility, percentile-binned | data_collection ≈L517 |

**The one window you control per ticker is how far back its history reaches — set
via the proxy:**

- `PROXIES['XYZW'] = 'OLDER'` — splice an older sibling fund/index before inception.
  Longer history → less noisy μ/σ, and the ticker actually participates in the
  pre-2010 / 15-year windows above.
- `PROXIES['XYZW'] = None` — use only the ticker's own history. Fine at ~15+ years;
  risky if young (its μ is a mean over just a few years).
- `BLENDED_PROXIES` — weighted mix as one proxy (e.g. DFWIX = 78% DFIEX + 22% DFCEX).
- `KF_TICKERS` + `KF_BLENDS` — extend a factor fund to 1963/1973 via Ken French
  portfolios. Bespoke; **skip unless you specifically want pre-inception factor data.**

**Judging window adequacy — the key call Claude must raise with the user:**

1. **Enough history?** A μ estimated over ~5 years of one market regime is misleading.
   Prefer a proxy that reaches back through at least one full cycle. If a ticker can
   only offer a few years and no good proxy exists, say so — the estimate will be noisy.
2. **Is the *early* history representative of the fund today?** If not, add the ticker
   to `ROLLING_WINDOW_TICKERS` to cap the estimate at the most recent **20 years**.
   Repo examples: `COST` (a different company 20 yrs ago) and `AVEE` (post-Asian-crisis
   EM small-cap, ~111% annualized vol early on).
3. **Shaky parameters?** Add a `PARAM_STD_OVERRIDE` entry to widen its
   estimation-error (epistemic) band so the simulation reflects that uncertainty.
4. **Is it a single company (not a fund)?** Add it to `SINGLE_STOCKS` in
   `montecarlo/loaders.py`. That *automatically* gives it the 20-year rolling window **and** an
   inflated parameter-uncertainty band (`0.25× vol` vs the `0.15×` default) — single
   stocks carry idiosyncratic / survivorship risk that empirical vol understates.
   Do **not** also add it to `PARAM_STD_OVERRIDE` (that would bypass the bump;
   `check_consistency.py` flags this).

**Worked example — SCHD (inception 2011):**
- `None` → μ/σ from 2011→today (~14 yrs); workable but short.
- `'VYM'` → splices Vanguard High-Dividend back to ~2006; better.
- a Ken French high-dividend/value blend → back to 1973 (only if you set KF entries).
- SCHD's history is clean, so **no** `ROLLING_WINDOW_TICKERS` cap needed.

## Hard rules (for Claude)

- `tools/check_consistency.py` **must be green** before declaring done — it is the
  oracle that catches a missed coupling point (the whole reason this is safe to
  hand-drive).
- **Never reorder** existing tickers in `MODEL_TICKERS` — it silently scrambles
  which asset each correlation-matrix row represents. Append only.
- **Never hand-edit `output/`** — always regenerate via `data_collection`.

## Reverting

Everything is in git. `git reset --hard HEAD` undoes uncommitted edits; or restore
`portfolio_templates.csv.bak`. Then re-run `python3 data_collection_regime_model.py`
to rebuild `output/`.

For the exhaustive coupling map (every file a ticker touches), see
[RUNBOOK_tickers.md](RUNBOOK_tickers.md).
