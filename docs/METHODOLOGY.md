# Methodology

How the model is built: the fund universe, regimes, correlations, history splicing, expected returns and parameter uncertainty.

## What the Model Does

### What This Does

Monte Carlo simulation of portfolio performance over 25-50 year horizons with:

1. **Regime-Switching Volatility**: Assets transition between LOW/MED/HIGH volatility states monthly
2. **Regime-Dependent Correlations**: Correlations rise during crises (VOO-VEA: 0.65 → 0.91). Which months count as a crisis is set by `metadata.market_regime` — the market proxy's own 12-month volatility — and `montecarlo/engine.py` simulates that same state, so producer and consumer share one definition
3. **Parameter Uncertainty**: Captures estimation error in expected returns
4. **Quarterly Rebalancing**: Hybrid partial/full rebalancing with configurable bands

---

## Technical Details

### Available Tickers (27 total)

Universe defined once in `loaders.MODEL_TICKERS` (order is canonical — it defines
JSON & correlation-matrix row order). Composition: 18 equity ETFs + 7 bond ETFs +
1 mutual fund (DFWIX) + 1 stock (COST).

**US Equities:**
- VOO (S&P 500), COST (Costco — the one single stock), JQUA (Quality), VTI (Total Market)
- AVUV (Small-Cap Value), VBR (Small-Cap Value, Vanguard analog of AVUV)
- VFMO (Momentum), MTUM (Momentum Alt)
- VTV (Large-Cap Value), AVLV (Large-Cap Value Alt)

**International Equities:**
- AVDE (Developed Large), AVDV (Developed Small Value)
- AVEM (Emerging Large), AVEE (Emerging Small)
- VEA (Developed), VXUS (Intl Total), VWO (Emerging), VSS (Intl Small)
- DFWIX (DFA World ex-US Core Equity — the one mutual fund)

**Global:**
- VT (Total World — one-fund benchmark; ~60% US / 30% DM / 10% EM)

**Bonds:**
- BND (Total Bond), JPLD (Limited Duration), JPST (Ultra Short)
- BSV (Short-Term Bond, Vanguard analog of JPLD), VUSB (Ultra-Short Bond, Vanguard analog of JPST)
- STIP (0-5 Year TIPS — the one inflation-linked fund), VGIT (Intermediate Treasury — the one pure-Treasury fund)

### Regime-Switching Mechanics

Each ticker has 3 volatility regimes, and moves between them **monthly**. A month
is labelled by the ticker's own trailing 12-month volatility, cut at the 30th and
70th percentiles of that ticker's history:
- **LOW**: the calmest 30% of months
- **MEDIUM**: the middle 40%
- **HIGH**: the most volatile 30%

Each regime's volatility is the **realised** volatility of the months in that
bucket, not a percentile of the rolling estimate.

Transitions follow Markov chains estimated from those labels. Each ticker's
transition is **conditioned on the market regime the month moves into**
(`regime_transitions_by_market` in the JSON), shrunk toward the ticker's
unconditional row by 10 pseudo-counts so that nearly empty cells do not dominate.

**Example (VOO, unconditional rows, 2026-10):**
- Low regime (9.0% vol): 92% stay, 8% → medium, 0% → high
- Medium regime (14.8% vol): 6% → low, 84% stay, 10% → high
- High regime (20.5% vol): 0% → low, 14% → medium, 86% stay

### Correlation Dynamics

Correlation matrices use a **15-year rolling window** that automatically updates:
- **Window size**: 180 months (exactly 15 years)
- **Updates automatically**: When you run the pipeline, it uses the most recent 15 years
- **Aligned pairwise**: each pair is estimated on its own overlap, so a newer fund does not shorten the window for the rest

Four correlation matrices are generated:
- **Baseline**: Full-window correlations (180 months)
- **Low Regime**: Calm market correlations (30% of history)
- **Medium Regime**: Normal correlations (40% of history)
- **High Regime**: Crisis correlations (30% of history)

**Key finding**: VOO-VEA correlation rises 40% in crises (0.65 → 0.91), and **326 of 351 pairs rise** from the low to the high regime (mean +0.198; 2026-10-25 matrices).

Two bugs were fixed on 2026-09-10 to get there. The window was **136 months, not the intended 180** — `dropna()` let the newest ticker in the universe truncate every other ticker's sample. Pairs are now estimated on their own overlap. And *three different rules* were labelling a month's regime: the pipeline used cross-sectional dispersion across funds (which is near zero in a crash where everything falls together), while `montecarlo/engine.py` majority-voted the per-ticker Markov regimes (which did not separate medium from high at all — 14.1% against 14.1%). There is now one definition, `metadata.market_regime`, written by the pipeline and simulated by the engine.

### Rebalancing System

**Quarterly hybrid system:**
- **Partial rebalance**: Single position breaches band → rebalance only that position (~2 trades)
- **Full rebalance**: Multiple breaches, or a December check that finds a breach or annual drift → rebalance everything (~10 trades)
- **Near-band logic**: a single breach is escalated to a full rebalance when more than one other position has used over 70% of its band. Always on; there is no toggle.

**Bands:**
- Each position's band is the larger of a relative band (% of its target) and an absolute band (percentage points), set per fund in `engine.REBALANCING_BANDS`
- `MC_REBALANCING_BANDS=uniform` replaces them with one 25% relative / 5pp absolute band for every position (single stocks keep theirs)

---

## Validation

Run **`python3 tools/validate_all.py`** after any data regen or engine edit (and before
a production run). It chains six stages, cheapest-first, and exits non-zero if any fail:

| Tool | Tier | Catches |
|------|------|---------|
| `check_consistency.py` | structure | ticker universe disagreeing across files |
| `validate_parameters.py` | data | implausible μ/σ, unordered regimes, non-stochastic Markov rows, non-PSD/NaN correlations, **a proxy that isn't tracking the fund it backfills**, **market regimes that don't separate realised volatility** — *auto-runs at the end of `data_collection_regime_model.py`* |
| `check_templates.py` | data | a portfolio row that doesn't sum to 1.0, negative weights, or stale derived columns |
| `check_parameter_drift.py` | data | **a regen that moved a parameter more than a data refresh should** — the golden master for data, against a committed baseline. `validate_parameters` asks whether each number is plausible; this asks whether it MOVED, which is what catches a proxy or splice defect. `--update` to re-bless |
| `check_logic.py` | behavior | pure-logic regressions (band math, `compute_derived`, PSD repair, regime transitions, contributions) |
| `regression_check.py` | behavior | the simulation engine producing different numbers than before (synthetic golden master, accumulation AND decumulation AND one case per feature — rebalancing, glide, Social Security, care costs, shared errors — + determinism + a zero-vol analytic anchor) |

The golden master covers **both phases**. It used to retire at 65 and stop at 50, so
withdrawals never fired: every policy, both sourcing rules, the depletion test and the
whole spending series were unpinned, and a change to any of them passed the gate
untouched. The decumulation master retires mid-horizon and pins one case per policy plus
`guyton_klinger` under `pro_rata`.

It is pinned to a **synthetic** universe, so it tracks engine logic and
does *not* need re-blessing every quarterly data refresh. After an intentional engine
change, re-bless it with `python3 tools/regression_check.py --update`. No third-party test
framework is required — every check is a standalone script in the project's house style.

---

## Data Splicing Methodology

To extend historical data beyond ETF inception dates, the pipeline uses 2-3 stage splicing:

### Factor Funds with Ken French Integration (7 tickers)
**AVUV** (3-stage splice):
1. **Ken French academic data** (1963-1993): Weighted blend of Size/Value/Profitability portfolios
2. **DFSVX proxy fund** (1993-2019): yfinance data with expense ratio adjustment
3. **AVUV actual** (2019-present): Live ETF data
**Result**: about 759 months back to 1963 (2026-10)

**VTV, AVLV, MTUM, VFMO, JQUA, VBR** (2-stage splice):
1. **Ken French / AQR academic data** (from 1973): Weighted blends from academic portfolios, used until the fund's own or its proxy's history begins — between 1998 and 2019 depending on the fund
2. **Actual ETF data** (from the fund's own inception, via a proxy fund where there is a gap, e.g. DFLVX for AVLV)
**Result**: about 645 months back to 1973 (2026-10)
- **VBR** (US small-cap value): KF 25 Size×B/M small-value blend (ME1/ME2 × BM3/BM4, 1973-1998) → VISVX proxy (1998+) → VBR actual; about 645 months, μ≈14.4% / σ≈18.0% (2026-10; the Vanguard analog of AVUV)

VOO also leans on Ken French for its first decade: the market series covers 1970-02 to 1980-01, then VFINX takes over until VOO's own inception, giving about 681 months.

### Standard ETFs (16 tickers)
1. **yfinance proxy** (e.g., VFINX for VOO pre-2010) with expense ratio adjustment
2. **Actual ETF data** (from the fund's own inception)
**Result**: 140 (VUSB) to 478 (BND) months depending on inception date (2026-10); COST and AVEE are fixed at 240 (rolling window, below)

**Every proxy is now SCORED against the fund it stands in for.**
`data_collection_regime_model.py` computes correlation, β, tracking error and the
mean gap over the months after the real fund exists — which the splice discards —
and writes them to `metadata.proxy_scores`; `tools/validate_parameters.py` fails
the run offline if a *contributing* proxy drops below corr 0.90 or leaves β
outside 0.80–1.20. Inspect with `python3 tools/proxy_score.py`, which owns the
thresholds; the gate imports them, so the two cannot disagree.

This was built because the four international proxies turned out to be **rotated
by one** — VEA (developed markets) was backfilled with VEIEX (*emerging*
markets) at corr 0.856 for 41% of its history, VWO (emerging) with DFISX
(developed small cap) at corr 0.840. Corrected 2026-09-10 to VXUS←VGTSX,
VEA←DFALX, VWO←VEIEX, VSS←DFISX; three of the four gained history as well as
fidelity. AVLV's DFSVX (small value for a *large* value fund) became DFLVX at the
same time.

Includes the two Vanguard bond analogs added for the Vanguard-only comparison, and the two later bond funds, **STIP** (TRBFX proxy, 2006+; about 241 months) and **VGIT** (VFITX proxy, 1991+; about 420 months):
- **VUSB** (Ultra-Short Bond, ≈ JPST analog): VUBFX proxy (2015+) → VUSB actual; about 140 months, μ≈2.4% / σ≈0.9% (2026-10)
- **BSV** (Short-Term Bond, ≈ JPLD analog): VBISX proxy (1994+) → BSV actual; about 392 months, μ≈3.6% / σ≈2.0% (2026-10). Note its longer window spans higher-rate decades, so VG-vs-JPM bond μ isn't a pure like-for-like

### Blended Proxy Funds (4 tickers)
**DFWIX** (DFA World ex-US Core Equity):
1. **Blended proxy** (2005-2013): 78% DFIEX (DFA Intl Core Equity) + 22% DFCEX (DFA Emerging Markets Core Equity), matching DFWIX's actual DM/EM regional split
2. **DFWIX actual** (2013-present): Live mutual fund data
**Result**: about 253 months back to 2005, μ≈8.3% / σ≈15.3% (2026-10)

**VT** (Vanguard Total World — the global one-fund benchmark):
1. **Blended proxy** (1996-2008): 50% VTSMX (Total US) + 50% VGTSX (Total Intl), a global US+ex-US stand-in before VT's 2008 inception
2. **VT actual** (2008-present): Live ETF data
**Result**: about 366 months back to 1996, μ≈9.1% / σ≈14.0% (2026-10). Region roll-ups (us/dm/em nav columns) split VT as ≈60% US / 30% DM / 10% EM, like DFWIX's DM/EM split

**JPST** (ultra-short income) — 50% SHV + 50% MINT, and **JPLD** (limited
duration) — 70% VGSH + 30% VFSTX. Both were single-fund proxies until 2026-09-10
and both were *treasury-only* stand-ins for *credit* funds: SHV scored corr 0.397
against JPST over 112 months at 44% too little volatility, and VGSH under-earned
JPLD by 1.31pp/yr at t=3.2 — the only statistically resolved mean gap in the
scoring table. Each blend pairs a duration leg with a credit leg. The visible
consequence: JPLD–JPST correlation in the HIGH-volatility regime was **−0.173**
and is now **+0.392**.

### Cost Adjustments
- **Expense ratio**: -0.03% to -0.25% annually (actual fund expenses)
- **Implementation drag**: per fund, applied to the Ken French / AQR segment: VFMO -1.10%, MTUM -0.70%, AVUV and JQUA -0.40%, AVLV -0.20%, and none for VOO, VTV or VBR (`COST_ADJUSTMENTS` in `data_collection_regime_model.py`)
- Applied monthly: `(expense + drag) / 12` deducted from returns

### Rolling 20-Year Window Tickers
Select tickers use a rolling 20-year window (240 months) because their early-period data is unrepresentative of current markets. The window automatically updates: always uses most recent 20 years.

**COST** (individual stock):
- Uses most recent 240 months of actual stock data
- No proxies, no splicing — Costco was a fundamentally different business 20+ years ago
- Example: an Oct 2026 run → Nov 2006 to Oct 2026 (240 months)
- Pulled in automatically via the **single-stock policy** (see below), not a one-off entry
- The window sets its volatility and its factor loadings; its expected return is NOT its 20-year mean (see below)

**AVEE** (EM small-cap value):
- Uses most recent 240 months of proxy-spliced data (DEMSX + actual AVEE)
- Early proxy data (1998-2002) reflects post-Asian crisis EM small-cap chaos: 111% annualized vol, 55% arithmetic return — not representative of modern EM markets
- Without the window, those early extremes inflate its historical mean
- With window: ~9.2% historical mean (2026-10); under the default factor method its return comes from its loadings regardless

**Validation**: Splice points validated for continuity (differences < 3% typical)

---

## Expected Returns

Each fund's expected return is **built from factor exposures**, not read off its
own history (`MC_MU_METHOD=factor`, the default). The code is `montecarlo/factors.py`; the
fitted loadings are `config/factor_loadings.json`, written by
`tools/factor_loadings.py`; the premia are measured by `tools/premia.py`.
`MC_MU_METHOD=historical` keeps each fund's own spliced mean, as a stress view.

### Why a fund's history cannot set its expected return

The standard error of a mean return is `σ/√T`. It depends on the calendar span
alone, and sampling more often does not shrink it. `python3 tools/estimability.py`
shows this on the repo's own data:

- A century of US market data (1926–2026, 18.4% volatility) leaves the market
  premium's 95% interval **7.2pp wide**. Pinning it to ±1pp would take about 340
  years; to ±0.5pp, about 1,350.
- Sampling daily instead of quarterly over the same 16 years leaves the standard
  error of the mean flat, while the standard error of volatility shrinks
  sevenfold.
- A single fund is far worse: VOO's own 16 years leave a standard error of
  about 4pp on its mean, AVLV's 5 years about 7pp.

What does converge with more frequent sampling is the second moment: betas,
volatility, correlation, regime structure. So the model splits the job:

```
    fund and proxy history  ->  SHAPE   (loadings, volatility, correlation, regimes)
    stated factor premia    ->  LEVEL   (expected return)
```

No better proxy changes this. A perfect 54-year proxy still yields a level
that is ±5pp.

### Why per-fund means were abandoned

Before the factor method, each fund's expected return was the mean of its own
spliced series, over its own window, with nothing forcing funds to agree. On
the parameter file of 2026-08-29 that produced results that cannot all be true:

- **VXUS sat below both VEA and VWO**, the two funds it is made of.
- **VFMO led MTUM by 2.7pp** a year, though both are the same momentum bet with
  different Ken French blends behind them.
- **The uncertainty bands were inverted.** The speculative premia carried the
  tightest bands (AVLV 0.07× its volatility) and plain market beta the widest
  (VOO 0.16×), because hand-set overrides had accumulated over time.
- **AVLV's history was mostly its proxy, and the proxy carried a size tilt** as
  well as a value one. Rebuilt with large caps only, the proxy lost about 1pp a
  year.

A third apparent incoherence, VTI beating VOO by 2pp a year, turned out to be a
data bug: VOO was backfilled with the S&P 500 *price* index, without
dividends. Fixed, the two sit 0.2pp apart. An argument built on one pivot fund
is only as good as that fund's data.

### The build-up

```
    μ = rf_forward + Σ_region Σ_factor  β × premium  −  expense  −  implementation
```

**Risk-free rate.** `rf_forward` is the real cash rate compounded with the
inflation assumption. The real rate defaults to **0.57%**, the realised real
return on cash over 1963-07..2026-07, the same window as the US premia.
`MC_RF_REAL` states it directly, and `MC_RF_SOURCE=tips30` takes a live TIPS
quote. The quote is not the default because it anchors a multi-decade horizon
on a single day's yield; in September 2026 the 30-year TIPS sat at the top of
its entire history.

**Premia.** Six Fama-French factors per region (market, size, value,
profitability, investment, momentum), each measured on its own full window
rather than on the months every factor shares. Annual, over cash (re-measured
2026-09-11):

| region | market | size | value | profitability | investment | momentum | window |
|---|---|---|---|---|---|---|---|
| US | 7.19% | 2.25% | 3.59% | 3.08% | 2.96% | 7.41% | 1963+ (momentum 1927+) |
| Developed ex-US | 4.87% | 0.41% | 5.09% | 3.24% | 2.02% | 7.93% | 1990+ |
| Emerging | 8.13% | 1.35% | 7.93% | 2.82% | 2.81% | 9.98% | 1989–1992+ |

The international market premia are much less certain than the US one
(standard error 2.7pp and 3.4pp against 1.95pp), because their windows are
about 36 years against 63.

**Loadings come from the actual fund, not the spliced series.** Betas converge
quickly; a fund's own 5 years pin its market beta to about ±0.07. Spliced
loadings would buy precision the model does not need at the cost of bias: they
describe the proxy, not the fund you hold. Measured on six factor funds, four
had loadings that differed significantly between the two series. The two that
agreed were the two with the shortest actual histories, too short to tell.

**Regions are fitted, not mapped.** Each fund is regressed on the US,
developed-ex-US and emerging factor sets, and a region joins the fit only when
it is significant *and* adds real explanatory power. Against US factors alone
the international funds leave 20–45% of their variance unexplained and show up
to 4.4pp a year of spurious "alpha". On their own region's factors they reach
R² 0.97–0.99. One known miss: VT's ~10% emerging sleeve falls below the cutoff,
so VT is fitted US + developed and understated by about 0.3pp.

**The gate.** A fund whose best fit has adjusted R² below 0.80 gets no factor
treatment. This is not a tuned threshold: the universe splits at 0.58 → 0.89
with nothing between. Below the gap sit every bond fund and the single stock.

**Bonds** earn `rf_forward` plus their mean excess return over the cash rate *of
the same months*. A raw mean tracks the era a fund launched in. A fund born
after 2008 inherited near-zero rates as a forecast, which had the model believing
cash-like funds lose 1.2% a year in real terms forever. The spread is robust to
that, and it orders the funds the way duration says it should. A yield-anchored
forecast would be better still, and is not built.

**Single stocks** (`loaders.SINGLE_STOCKS`) earn their six US factor loadings,
fitted on the same 240 months as their volatility, plus `MC_SINGLE_STOCK_EDGE`
(default 0). A stock is in the universe because it outperformed, so its
historical mean *is* its alpha; carrying that forward would make the selection
the forecast. Any edge you believe in is stated, not inherited.

**Costs.** The regression runs on net returns, so `E[r_net] = rf + Σβ·premium
+ α`, where α carries the fee and the trading cost. The factors themselves are
academic portfolios with no fees and no trading. Dropping α (it cannot be
estimated, and its expected value is about zero before costs) therefore adds
back *both* costs, and both are subtracted:

- `expense`: the published expense ratio;
- `implementation`: reported annual turnover (floored at 5%) × `MC_ROUND_TRIP`
  (0.15%).

A check that the framework works: VOO's α comes out at −0.10% ± 0.23%, which
is its fee.

### The uncertainty band

Each fund's band on μ is derived, not hand-set:

```
    Var(μ) = Σ_f [ β_f² · Var(premium_f) + premium_f² · Var(β_f) ] + Var(α)
```

That is: not knowing the premium, not knowing the loading, and not knowing what
the factors miss. A fund with a large value loading gets a wide band
automatically. With `MC_PARAM_ERRORS=shared` (the default), each path draws one
error per premium and every fund that loads on it moves together. The premium
errors are correlated as measured (`factors.PREMIUM_CORR`), so a portfolio
cannot diversify away uncertainty its funds share.

### Varying the premia

The premia are the least knowable input, so they are varied deliberately
rather than pinned:

- **`MC_PREMIUM_SCALE`** scales the style premia (size, value, profitability,
  investment, momentum). At 0 every equity fund earns what its market beta
  earns. It does not touch the equity premium over bonds.
- **`MC_PREMIUM_HAIRCUT`** scales the market premium itself.
- **Named scenarios** (`factors.SCENARIOS`): `all half`, `no factors`, one
  factor dead at a time, and a regional axis. The regional scenarios are `us
  stays ahead` (the default), `intl catches up`, `regions converge` and `intl
  keeps lagging`, which holds the US's 2005–2026 lead over each region.
  `tools/sensitivity.py` sweeps them, and `montecarlo/return_views.py` re-runs a
  portfolio comparison under several of them.

### Why the two methods disagree

For most funds the gap between the factor and historical methods is **mostly
the risk-free rate**, not the factors. A historical mean embeds the cash rate of
the years the fund lived through; the build-up uses `rf_forward`. Funds whose
history is mostly post-2000 differ most, and funds with long histories (VOO,
AVUV) barely differ. `python3 tools/mu_decompose.py` splits each fund's gap into
rate, premia, alpha and a residual.

Re-basing the historical method on the contemporaneous rate was considered and
rejected. Over seven decades, total and excess market returns have the same
dispersion and weak correlations with rf of opposite sign. The data cannot tell
the two conventions apart, and `historical` is useful only as the unmodified
record.

### What this does not fix

- **Forecasts are not accurate.** The premia remain uncertain. What the
  build-up adds is coherence: one premium per factor instead of 27 independent
  guesses, each premium estimated on far more data than any one fund has.
- **Costs are not fully identifiable.** For every factor fund, the measured α
  sits within about 1.1 standard errors of both candidate cost adjustments.
- **There is no valuation anchor.** The model starts from today's prices and
  applies long-run premia, assuming valuations neither help nor hurt.
  Holdings-based build-ups (earnings yield, growth and repricing) have some
  predictive power over a decade. One is not built.

---

## Parameter Uncertainty Overrides

Under the default `MC_MU_METHOD=factor`, `montecarlo/engine.py` builds each fund's expected return **and** its uncertainty band from factor loadings × premia (see [Expected Returns](#expected-returns)). The historical means in the JSON and the bands described below are what `MC_MU_METHOD=historical` uses, and the fallback for any fund without a factor fit.

### What Are Parameter Uncertainty Overrides?

The JSON parameter file (`ticker_parameters_regimes_v8.json`) stores one uncertainty value per ticker, **`parameter_std`**: the standard deviation of the estimation error in its expected return. The pipeline sets it at generation time. For most tickers it is a multiple of annualized volatility; for the tickers in `PARAM_STD_OVERRIDE` (in `data_collection_regime_model.py`) an explicit value replaces that multiple. `montecarlo/loaders.py` reads the stored `parameter_std`; `montecarlo/engine.py` has no override logic of its own.

### Why Override?

Multi-factor funds (small-cap value, momentum, quality) benefit from diversification across multiple systematic factors, reducing expected return uncertainty below what single-asset historical data suggests. Overrides reflect this theoretical reduction.

⚠ **These bands are a hand-calibrated judgement, and a tight one.** At AVUV's
0.020 the model implies a ~0.09% probability that AVUV has no premium at all,
which no reading of the historical record supports.

Do **not** "fix" this by flooring at `sigma/sqrt(years)`: `base_volatility` is
estimated on a 2010+ window while `expected_return` runs from 1973, so that
standard error mixes two different histories.

The consequence is measured rather than argued. `MC_PARAM_STD_SCALE` widens every
band together, and `tools/sensitivity.py param_std 1.0,2.0,3.0` sweeps it:

| band | 141 | 141>141+10@65 | 145 |
|---|---|---|---|
| 1x | 124k | 119k | 73k |   <!-- STALE: pre-2026-09-10 VOO -->
| 2x | 91k | 91k | 46k |
| 3x | 47k | 52k | 18k |

**LEVEL-ONLY.** Tripling the band costs 2.6x of plan level and reverses no pair,
so the width matters enormously for the FIGURE and not at all for the CHOICE —
nobody has to win the argument about what it should be before the allocation can
be settled. ⚠ The 145 figures here are STALE as of 2026-09-12 — the row gained
COST 0.05 (mirroring 140's single-stock sleeve) and is worth +24.6% of median
wealth; see `.claude/CLAUDE.md`. 141's edge over 145 actually *grows* as the
bands widen (1.70x →
1.98x → 2.61x): 141 holds 9 funds against 145's 6, so independent estimation
errors partially cancel. Diversification across parameter uncertainty, not just
across returns. ⚠ **SUPERSEDED 2026-10-02:** that diversification was an
artifact of drawing each fund's estimation error independently. The errors come
from premia the funds share, and since they are drawn that way it largely
disappears — see `docs/RESULTS.md`, "Estimation errors are shared".

### Current Overrides

| Ticker | Fund | Calculated (2026-10) | Override | Change | Rationale |
|--------|------|-----------|----------|-----------|-----------|
| AVUV | US Small-Cap Value | 0.0320 | 0.0200 | -38% | Size + Value + Profitability factors |
| AVDE | Intl Developed All Cap | 0.0228 | 0.0140 | -39% | International diversification |
| AVEM | Emerging Markets | 0.0260 | 0.0220 | -15% | Emerging market value factors |
| AVDV | Intl Developed Small Value | 0.0259 | 0.0180 | -31% | Size + Value factors (intl) |
| AVEE | Emerging Markets Small | 0.0256 | 0.0250 | -2% | Size factor (emerging) |
| VTV | Large-Cap Value | 0.0203 | 0.0150 | -26% | Value factor (large cap) |
| AVLV | Large-Cap Value Alt | 0.0238 | 0.0135 | -43% | Value factor (alternative) |
| DFWIX | DFA World ex-US Core | 0.0229 | 0.0160 | -30% | Broad intl, moderate factor tilts |
| VBR | US Small-Cap Value | 0.0270 | 0.0180 | -33% | Size + Value (Vanguard AVUV analog) |

(Calculated = 0.15 × annualized volatility, the default every other ticker gets. Every override is TIGHTER than that default, which is the inversion the factor method removes: under `MC_MU_METHOD=factor` these bands are rebuilt from loadings and premia, and this table applies only to `historical`.)

**All other tickers** (VOO, VTI, VEA, VXUS, VWO, VSS, BND, JPLD, JPST, JQUA, MTUM, VFMO, VT, VUSB, BSV, STIP, VGIT): calculated value = **0.15 × annualized vol**, no override. Single stocks (COST) instead use **0.25 × vol** — see below.

### Single-Stock Inflation (the opposite direction)

Overrides above *reduce* uncertainty for diversified multi-factor funds. **Single
companies go the other way.** Any ticker in `SINGLE_STOCKS` (in `montecarlo/loaders.py`; `COST`
today) instead uses **0.25 × vol** for `parameter_std` — a *wider* epistemic band —
because a single stock carries idiosyncratic / survivorship risk that its empirical
volatility (measured from a survivor) understates. Single stocks also auto-get the
20-year rolling window. Do **not** add a single stock to `PARAM_STD_OVERRIDE` — an
explicit override wins and would silently cancel the inflation (`check_consistency.py`
flags this). To make another holding a single stock, just add it to `SINGLE_STOCKS`.

**Its expected return is its factor loadings, not its history** (since
2026-10-12, `docs/RESULTS.md` N10). A single stock's mean IS its alpha, and it
is in the universe because it outperformed — COST's 20-year mean put it at
14.65% real against VOO's 7.49%. It is now built like every fund: `rf +
sum(loading x premium) + MC_SINGLE_STOCK_EDGE`, with its six US loadings fitted
on the same 240 months (COST: market 0.81, profitability 0.58) and the edge zero
unless you state one. `tools/sensitivity.py single_stock_edge` sweeps it;
`validate_parameters.py` fails a single stock that leads VOO by more than its
loadings and edge allow. Under `MC_MU_METHOD=historical` it keeps its mean.

### How to Modify Overrides

To change override values:

1. Open `data_collection_regime_model.py`
2. Find the `PARAM_STD_OVERRIDE` dictionary
3. Edit values:
   ```python
   PARAM_STD_OVERRIDE = {
       'AVUV': 0.020,  # Your custom value (e.g., 2.0% annual std dev)
       'AVDE': 0.014,
       # ... etc
   }
   ```
4. Regenerate parameters:
   ```bash
   python3 data_collection_regime_model.py
   ```

The new overrides are written to the JSON file as `parameter_std`, and `montecarlo/loaders.py` reads them from there on the next run.

---

## References

### Academic Papers

- **Regime-Switching**: Hamilton (1989), "A New Approach to Economic Analysis of Nonstationary Time Series"
- **Factor Premiums**: Fama-French (1992, 2015), "The Cross-Section of Expected Stock Returns"
- **Correlation Dynamics**: Longin-Solnik (2001), "Extreme Correlation of International Equity Markets"

### Data Sources

- **Historical prices**: Yahoo Finance (via yfinance)
- **Proxy indices**: Ken French Data Library, S&P/Russell indices
- **Inception dates**: Vanguard, Avantis, JP Morgan fund fact sheets
