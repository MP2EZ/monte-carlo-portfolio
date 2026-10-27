---
eyebrow: Monte Carlo Analysis · v3.4
artifact: https://claude.ai/artifact/SMkPjJS5SbzD2s2RwBHe64
footer: Built from the parameters JSON, `config/investor.json`, `portfolio_templates.csv` and `output/runs/` on 2026-10-25.
---

# Monte Carlo Data Map

How the simulation's data fits together. There is no database: every entity is
a file, a CSV row, a key in a JSON record, or a registry in code, and each
diagram's comments say which.

## Overview

All 27 entities without their attributes. Each layer below repeats its part
with attributes and the file or registry it lives in.

```mermaid
erDiagram
    PARAMETER_SET ||--|{ TICKER : "describes 27"
    PARAMETER_SET ||--|| MARKET_REGIME : "carries"
    PARAMETER_SET ||--|{ CORRELATION_MATRIX : "writes"
    TICKER ||--|{ SOURCE_SERIES : "spliced from"
    TICKER ||--|{ REGIME_PARAMS : "has 3"
    TICKER ||--o{ LOADING : "loads on"
    FACTOR_PREMIUM ||--o{ LOADING : "paid through"
    MARKET_REGIME ||--|{ CORRELATION_MATRIX : "selects"
    INVESTOR ||--|{ ACCOUNT : "holds 5"
    INVESTOR ||--|| SOCIAL_SECURITY : "claims"
    PORTFOLIO ||--|{ WEIGHT : "sums to 1.0"
    TICKER ||--o{ WEIGHT : "held at"
    STRATEGY ||--|{ WAYPOINT : "steps through"
    WAYPOINT }o--|| PORTFOLIO : "holds"
    RUN_SPEC ||--|{ STRATEGY : "resolves to"
    RUN_SPEC }o--|| WITHDRAWAL_POLICY : "uses"
    RUN_RECORD ||--|| RUN_SPEC : "replays"
    RUN_RECORD ||--|| PROVENANCE : "snapshots"
    PROVENANCE }o--|| PARAMETER_SET : "copies version of"
    PROVENANCE }o--|| INVESTOR : "copies facts of"
    RUN_RECORD ||--|{ RESULT : "results[policy][strategy]"
    RESULT }o--|| STRATEGY : "for"
    RESULT }o--|| WITHDRAWAL_POLICY : "under"
    RESULT ||--|{ PATH : "computed over"
    RUN_RECORD ||--o{ RETURN_VIEW_SCORE : "keeps"
    RETURN_VIEW ||--o{ RETURN_VIEW_SCORE : "scored in"
    SENSITIVITY_DIM ||--|{ SWEEP_POINT : "one per value"
    PLAN_CHOICE ||--o{ RUN_SPEC : "compare_choice"
    FINDING }o--o{ RUN_RECORD : "cites"
    FINDING }o--o{ SWEEP_POINT : "cites"
```

## Market layer

Regenerated each quarter by `data_collection_regime_model.py` into `output/`.
The ticker list sets the order of every matrix row and every random draw.

```mermaid
erDiagram
    PARAMETER_SET {
        string version "output/ticker_parameters_regimes_v8.json metadata"
        date last_updated
        int ticker_count "27, = len(loaders.MODEL_TICKERS)"
        map proxy_scores "corr, beta, TE, months used per proxy"
        map tips_real_yields
    }
    TICKER {
        string ticker PK "order of loaders.MODEL_TICKERS = draw position"
        float expected_return "rf + sum(loading x premium), nominal"
        float base_volatility
        float parameter_std "epistemic band; 0.25x vol for single stocks"
        date history_start
        date history_end
        int history_months "may only grow (check_parameter_drift)"
    }
    SOURCE_SERIES {
        string ticker FK
        string source "Ken French, AQR, yfinance proxy, actual"
        string blend "e.g. DFWIX = 78% DFIEX + 22% DFCEX"
        date from
        date to
    }
    REGIME_PARAMS {
        string ticker FK
        string regime PK "low, medium, high"
        float volatility "realised vol of months in the bucket"
        matrix transitions "3x3, unconditional"
        matrix transitions_by_market "3x3 per market regime, shrunk 10 pseudo-counts"
        int regime_months
    }
    FACTOR_PREMIUM {
        string region PK "us, dev, emg"
        string factor PK "mkt_rf, smb, hml, rmw, cma, mom"
        float premium "factors.PREMIA"
        float se "factors.PREMIUM_SE"
        string window "factors.PREMIUM_WINDOW, each its own"
        matrix premium_corr "factors.PREMIUM_CORR, premium x premium"
    }
    LOADING {
        string ticker FK
        string factor FK
        float beta "fitted on the actual fund, not the splice"
    }
    MARKET_REGIME {
        string proxy "market proxy, 12-month vol"
        float p30_cut
        float p70_cut
        matrix transitions "its own Markov chain"
    }
    CORRELATION_MATRIX {
        string regime PK "baseline, low, medium, high"
        string window PK "full, w35 (MC_CORR_WINDOW)"
        file path "output/correlation_<regime>_27[_w35].npy"
        matrix values "27x27, PSD via near_psd_higham"
    }

    PARAMETER_SET ||--|{ TICKER : "describes"
    PARAMETER_SET ||--|| MARKET_REGIME : "carries"
    PARAMETER_SET ||--|{ CORRELATION_MATRIX : "writes"
    TICKER ||--|{ SOURCE_SERIES : "spliced from"
    TICKER ||--|{ REGIME_PARAMS : "has 3"
    TICKER ||--o{ LOADING : "loads on"
    FACTOR_PREMIUM ||--o{ LOADING : "paid through"
    MARKET_REGIME ||--|{ CORRELATION_MATRIX : "selects by regime"
    MARKET_REGIME ||--|{ REGIME_PARAMS : "conditions transitions"
    CORRELATION_MATRIX }|--|{ TICKER : "rows and columns"
```

## Plan, portfolio and run

Your facts come from `config/investor.json` and portfolios from
`portfolio_templates.csv`. A run resolves them into strategies and writes one
record to `output/runs/`.

```mermaid
erDiagram
    INVESTOR {
        int age "config/investor.json"
        int plan_to_age "95"
        int retirement_age
        int contributions_measured_at_age
        int time_horizon_years "derived: plan_to_age - age"
        float initial_value "derived: sum of balances"
    }
    ACCOUNT {
        string kind PK "401k_trad, 401k_roth, roth_ira, hsa, taxable"
        float balance
        float annual_amount
        float growth_rate
        int stop_age
        string tax "read by tools/tax_estimate.py, not the engine"
    }
    SOCIAL_SECURITY {
        float annual_at_fra
        int claim_age
        float benefit_scale "share of scheduled benefit assumed paid"
    }
    PORTFOLIO {
        string id PK "portfolio_templates.csv row; archive.csv not runnable"
        float bond_pct "derived columns via compute_derived()"
        float us_pct
        float dm_share
        string tilt_family
        int n_funds
        float max_single_fund
    }
    WEIGHT {
        string portfolio_id FK
        string ticker FK
        float weight "27 columns per row"
    }
    STRATEGY {
        string key PK "e.g. 168>168+30:bnd_only@60>168@70 (montecarlo/glide.py)"
    }
    WAYPOINT {
        string strategy_key FK
        int seq
        string portfolio_id FK "base portfolio"
        float sleeve_pct "synthesised bond sleeve, optional"
        string sleeve_fund "bnd_only, stip_only, vgit_only..."
        int from_age "steps, never interpolated"
    }
    WITHDRAWAL_POLICY {
        string policy PK "constant_real, fixed_pct, guyton_klinger, vpw"
        string sourcing "pro_rata, excess"
        float floor
        string gk_response
        int start_age
        float target_real_annual
    }
    RUN_SPEC {
        string mode "config/runs.json last, or --spec"
        list portfolios
        list custom "glide schedules"
        string policy FK
        float spend
        int search_sims
        int verify_sims
        string compare_choice "optional"
        bool return_views "optional"
        map social_security "optional override"
    }
    RUN_RECORD {
        datetime timestamp PK "output/runs/run_<timestamp>.json"
        map config "resolved engine config"
    }
    PROVENANCE {
        string parameters_version
        string mu_method
        float premium_scale
        float premium_haircut
        string param_errors "shared, independent"
        float single_stock_edge
        int corr_window_years
        float inflation
        float rf_real
        string adequacy "window, thirds, lifetime"
        string scoring_version
        string investor_accounts "fingerprint"
        int ss_claim_age
    }
    RESULT {
        string policy FK
        string strategy_key FK
        float spend "solved PLAN LEVEL, not a forecast"
        float level_se
        bool converged
        float success_rate
        float depletion_rate
        float median_realised "rank on this"
        float p10_realised
        list spend_pctiles
        float worst_cut
        float median_bequest
    }
    PATH {
        int seed "not persisted; same seeds across strategies"
        map epistemic_draw "once per path, shared through premia"
        list monthly_regimes "per ticker plus market"
        matrix returns "27 wide, read by position"
    }

    INVESTOR ||--|{ ACCOUNT : "holds 5"
    INVESTOR ||--|| SOCIAL_SECURITY : "claims"
    PORTFOLIO ||--|{ WEIGHT : "sums to 1.0"
    WEIGHT }o--|| TICKER : "of"
    STRATEGY ||--|{ WAYPOINT : "steps through"
    WAYPOINT }o--|| PORTFOLIO : "holds"
    RUN_SPEC ||--|{ STRATEGY : "resolves to"
    RUN_SPEC }o--|| WITHDRAWAL_POLICY : "uses"
    RUN_RECORD ||--|| RUN_SPEC : "replays"
    RUN_RECORD ||--|| PROVENANCE : "snapshots"
    PROVENANCE }o--|| INVESTOR : "copies facts of"
    RUN_RECORD ||--|{ RESULT : "results[policy][strategy]"
    RESULT }o--|| STRATEGY : "for"
    RESULT }o--|| WITHDRAWAL_POLICY : "under"
    RESULT ||--|{ PATH : "computed over"
```

## Analysis layer

The tools that fan out to many evaluations. Sweep points are printed but not
saved. Return-view scores are saved inside the run record.

```mermaid
erDiagram
    SENSITIVITY_DIM {
        string name PK "tools/sensitivity.DIMENSIONS, 16"
        string group "returns, cash, inflation, confidence, life, plan"
        string kind "numeric (has a span), named (no span)"
        string env "e.g. MC_PREMIUM_SCALE"
        string default_values
    }
    SWEEP_POINT {
        string dimension FK
        string value
        string verdict "moves LEVEL or RANKING; printed, not persisted"
    }
    RETURN_VIEW {
        string name PK "return_views.VIEWS, 5"
        map env "set in a child process; workers re-import"
    }
    RETURN_VIEW_SCORE {
        string view FK
        string strategy_key FK
        float regret "vs best strategy in that view"
        bool minimax_pick
    }
    PLAN_CHOICE {
        string key PK "run.PLAN_CHOICES: retirement_age, ss_claim_age, contributions"
        string label
    }
    FINDING {
        string id "ledger entry id, e.g. N10, N11"
        date measured
        string claim
    }

    SENSITIVITY_DIM ||--|{ SWEEP_POINT : "one per value"
    RETURN_VIEW ||--o{ RETURN_VIEW_SCORE : "scored in"
    RUN_RECORD ||--o{ RETURN_VIEW_SCORE : "keeps summary"
    RETURN_VIEW_SCORE }o--|| STRATEGY : "for"
    PLAN_CHOICE ||--o{ RUN_SPEC : "compare_choice"
    FINDING }o--o{ RUN_RECORD : "cites"
    FINDING }o--o{ SWEEP_POINT : "cites"
```

## What holds it together

- **MODEL_TICKERS is the root key.** Correlation order, CSV columns, JSON keys
  and the position of each random draw all follow it.
  `tools/check_consistency.py` checks that they agree.
- **The strategy key is the glide.** A key such as
  `168>168+30:bnd_only@60>168@70` spells out the waypoints and also indexes
  `results`, so there is no separate glide table.
- **Provenance copies, it does not reference.** The investor file and the
  parameters can change after a run. The record keeps the values that run used,
  which is how `run.py --diff` explains a change.
- **Paths are never stored.** Pairing still rests on them. Every result in a
  run, and every return view, uses the same seeds and draws across all 27
  tickers.
