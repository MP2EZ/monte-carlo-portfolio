# Changelog

What the model has concluded is summarised in [docs/RESULTS.md](docs/RESULTS.md).

## v3.5 (October 2026) — paired markets, one-plan comparisons, return views

**Comparisons**
- A menu of three questions, named by what changes: **Portfolios**, **Retirement
  choices**, **Market & life assumptions**
- Retirement choices (retire 60/65/70, claim 62/67/70, savings, withdrawal rule)
  are solved on one portfolio and read against your current value on shared paths
- Head-to-heads compare every strategy at ONE plan, so differences are in
  spending rather than in certified levels
- `return_views.py`: a portfolio comparison repeated under four other return
  models, scored by regret; prints the minimax pick
- An "intl keeps lagging" regional scenario
- STIP (short-term TIPS) added as the 26th ticker, with a `stip_only` bond sleeve;
  sweeps rank on realised spending
- Adequacy is the worst rolling 10-year window of spending, not each third of
  the retirement, so the rule is equally strict at every horizon
- A single stock's expected return is its six US factor loadings plus a stated
  edge (`MC_SINGLE_STOCK_EDGE`, default 0), never its historical mean
- VGIT, the first pure-Treasury fund (27 tickers), and more bond sleeves:
  `core_tips`, `thirds`, `quarters` and VGIT variants
- `MC_CORR_WINDOW` selects correlation matrices built on another lookback;
  `MC_REBALANCING_BANDS=uniform` simulates one 25%/5pp band for every fund
- `run.py --ask "sentence"` proposes a run spec from plain English (optional;
  needs the `anthropic` package)
- Claude Code skills: `design-runs`, `setup-portfolio`, `add-ticker` and others
- Layout: the library moved into `montecarlo/` and the portfolio CSVs into
  `portfolios/`; `run.py` and the data pipeline stay at the root

**Fixes**
- Every strategy now sees the same markets whatever funds it holds. Draws were
  sized to the funds a plan held, so comparisons across fund sets were unpaired
- Estimation errors are drawn through the factor premia the funds share, with
  the premia's measured correlation. Independent per-fund draws let a portfolio
  average away common uncertainty (understated 1.4-1.6x)
- Custom schedules keep their typed order; repeated choice values are refused
- Ken French tables were dated a month early in the correlation step, which
  misaligned the factor funds' correlations (AVLV-VOO read −0.03 in calm markets)
- A nominal bond's inflation pass-through scales with its duration
- A fund can no longer lose more than everything in a month (fat-tail runs
  reported NaN)
- DFWIX is 65/35 developed/emerging, as Dimensional reports it
- A partial rebalance sells an overweight fund into the most underweight one

## v3.4 (September 2026) — scoring v2, sensitivity, your details

Results from before this version are NOT comparable to results after it;
`run.py --diff` says so.

**Scoring**
- Rank on spending by life stage, not one number. Measured on portfolio 140, the
  old headline put guyton_klinger ahead of vpw; by life stage vpw is ahead in the
  median AND the 10th percentile of every band
- "Sustainable spend" renamed **plan level**: the level the solver certified,
  not a forecast of spending
- Adequacy is per-band: a path passes only if EVERY life stage clears the floor
- The solver estimates the crossing instead of bisecting, and sets the level on
  the verification sample (bisection read 17% low for guyton_klinger); `vpw` is
  solved in closed form

**Capability**
- `tools/sensitivity.py`: one registry of assumptions, reporting LEVEL vs RANKING
- Stochastic inflation; a shared uncertainty band on the real cash rate
- Social Security lives in your details (`config/investor.json`), not on every run
- Long-term care shock, rebalancing turnover, `tools/tax_estimate.py`
- `run.py --spec` for headless, reproducible runs; four `--debug` diagnostics
- `tools/check_parameter_drift.py`, a golden master for DATA

**Fixes**
- The real cash rate was overstated by 0.74pp (assumed, not realised, inflation)
- International proxies were rotated by one (VEA was backfilled with an emerging
  markets fund); every proxy is now scored against the fund it stands in for
- The correlation window was 136 months, not 180, and three different rules
  labelled the crisis regime; there is now one (`metadata.market_regime`)
- VOO was understated 1.70pp/yr by a price-index proxy

## v3.3 (August 2026) — sensitivity, Social Security, bond sleeves

- `tools/sensitivity.py`: sweeps an assumption and reports LEVEL vs RANKING
- Social Security with a real claiming schedule (70% at 62, 124% at 70) and a
  `benefit_scale` haircut, because scheduled benefits are not payable benefits
- Bond sleeves as an independent axis; ladders synthesised from a base portfolio's own
  equity sleeve, fixing a confound where "derisking" also swapped VEA for AVDE
- Planning horizon 87 → 95 (a high age percentile, not life expectancy)
- `MC_CRISIS_VOL_SCALE` / `MC_TAIL_DF` stress tests
- Run persistence with provenance; `--history` and `--diff`
- **A2/A4 retracted** — the "2008 vol was 40%" claim was daily volatility against a
  monthly model; the crisis regime is well calibrated

## v3.2 (August 2026) — real dollars + retirement era

**Reporting**
- All output in REAL (today's) dollars; `engine.INFLATION` is the assumption
- Bootstrap confidence intervals on every figure; paired comparisons for
  head-to-heads, since strategies share common random numbers
- Run records in `output/runs/` with provenance, plus `--history` and `--diff`

**Capability**
- Retirement-era withdrawals: `constant_real`, `fixed_pct`, `guyton_klinger`, `vpw`
- Sourcing rules: `pro_rata` (control) and `excess` (default)
- Glide paths composed from existing CSV rows; waypoints are steps, not interpolated
- Inverse solver: the largest plan level a strategy supports at a target
  confidence, judged per life stage and estimated on the verification sample
- `MC_PREMIUM_SCALE` for equity-premium sensitivity, `MC_PARAM_STD_SCALE` for the
  epistemic band
- `run.py` interactive runner; `--debug` for phase timing and estimate calibration

**Fixes** (several were producing materially wrong numbers)
- Contributions were counted as investment returns — Sortino overstated 27%
- Reported "CAGR" credited contributed principal (11.64% vs a true 9.53%)
- Sortino denominator averaged over down months only, not all periods (~1.5x low)
- Depletion was never detected: a shrinking portfolio decays asymptotically and
  never hits the `balance <= 0` test
- VPW amortised to the simulation endpoint, giving a 53% withdrawal rate at 85
- AQR quality data divided by 100 when already in decimal, scaling 84% of JQUA's
  history down 100x
- `MAR_ANNUAL` silently became a ~7.1%-nominal hurdle after the real-dollar change

**Performance**
- 4.32x faster and bit-identical: regime CDFs precomputed, Cholesky factors
  hoisted out of the monthly loop
- 547 lines of stale 20-ticker parameters removed from the legacy runner

**Known limitations**
- Sortino does not discriminate between the current portfolios (all five span
  0.903-0.935, no pairwise difference resolves) — rank on wealth or spending
- `parameter_std` factor bands imply only a 0.09% chance AVUV has no premium,
  more confident than the record supports
- JQUA parameters are stale pending a regen

## Earlier versions

- **v3.1** (Jun 2026): Durable-comparison expansion → **25 tickers**
  - Added VT (Total World, one-fund benchmark) and VBR (small-cap value)
  - Added VUSB + BSV (Vanguard analogs of JPST/JPLD) for an all-Vanguard bond sleeve
  - New portfolios: VT benchmarks (102/202/302/502), Vanguard-only mirrors (145/245/345/545)
  - First-class single-stock risk inflation (0.25× vol band via `SINGLE_STOCKS`)

- **v3.0** (Nov 2025): Unified data collection pipeline with Ken French integration (1963-2025 history)
  - Auto-loading parameter system with JSON
  - Initial 21-ticker support with regime-dependent correlations
  - Quarterly rebalancing with hybrid partial/full logic
  - Validation report generation
  - Parameter uncertainty overrides for multi-factor funds

- **v2.0** (Oct 2025): 20-ticker system, quarterly rebalancing, regime correlations
  - Added MTUM, VFMO, VTV, AVLV (4 new tickers)
  - Regime-dependent correlation matrices (4 files)
  - Quarterly rebalancing system

- **v1.0** (Oct 2025): Initial 18-ticker implementation
  - Regime-switching volatility model
  - 3-stage simulation architecture
  - 16 ETF/fund tickers + COST + BTC

- **Legacy** (Oct 2025): Jupyter notebook jump-diffusion model (deprecated)

## Parameter file versions

- **v8** (current schema; regenerated quarterly): now spans all **27 tickers**
  - Standardized JSON schema (flat structure); extended in place as tickers were added
    (v8 originally introduced MTUM/VFMO/VTV/AVLV at 20 tickers; now includes DFWIX, VT,
    VBR, VUSB, BSV, STIP, VGIT)
  - Parameter uncertainty overrides for 9 funds (see docs/METHODOLOGY.md, "Current Overrides")
  - Auto-loading via `loaders.py`
  - Metadata includes version, last_updated, ticker_count, correlation file references

- **v7** (Oct 21, 2025): 18 tickers with mixed schema
  - Mixed nested + flat JSON structure
  - Added BTC (later removed in v8)

- **v2** (Oct 15, 2025): Original 16 tickers
  - Initial regime-switching implementation
  - Manual parameter copying required
