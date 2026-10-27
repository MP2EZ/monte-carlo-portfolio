# Results

What this model has concluded, how sure it is, and how to re-run each check.
Measured 2026-10-22 to 2026-10-26 on the current engine: planning to 95,
certified on the worst rolling 10-year window, `guyton_klinger` with the
`fast` response, 3,000 paths unless stated. Every strategy sees the same
simulated markets, and estimation errors are shared through the factor premia.

**Read these as shapes, not figures.** They were measured on one investor's
plan (accumulating until 65, withdrawing to 95) and this repo's portfolio
library. Dollar levels depend on your balances and savings, so this page
reports percentages, ratios and rankings. **BETTER** or **WORSE** means the
paired 95% interval on median lifetime spending excludes zero; **tie** means it
does not, which is not the same as equal.

Portfolio IDs are rows of [`portfolios/portfolio_templates.csv`](../portfolios/portfolio_templates.csv):

All are 100% equity and hold a 5% single-stock position (COST).

| ID | Shape |
|---|---|
| 140 | 70% US / 30% international: VOO core, small value (AVUV), momentum (VFMO); Avantis international, 60/40 developed/emerging |
| 141 | 140 with most of its developed-markets sleeve in VEA instead of AVDE |
| 145 | 140's shape in Vanguard funds — VBR for small value, VEA / VWO / VSS for international — with the same VFMO: weaker tilts, lower fees |
| 161 | 140 with 10% moved from VOO into large value (AVLV) |
| 165 | 161 with an international sleeve built around DFWIX (about 53/47 developed/emerging) |
| 166 | 165 at 65% US / 35% international |
| 168 | 161 with an international sleeve built around DFWIX, rebalanced to 60/40 developed/emerging |

---

## 1. Most assumptions move the level, not the decision

Each assumption swept one at a time (800 paths), on 141 unless noted, ranked
on median lifetime spending at each strategy's own plan. **22 of the 24 move
only the level.** The two that can reorder portfolios are the factor premia (§2) and a
single stock's edge (§6).

| Assumption | Range swept | Spending span |
|---|---|---|
| Real cash rate | 0% / 0.57% / 2.98% | **4.3x**, the largest lever in the model |
| Social Security paid | 100% / 80% / 0% of scheduled | 1.7x |
| Factor premia over the market | 100% / 50% / 0% | 1.6x (see §2) |
| Planning horizon | 50 / 58 / 63 years | 1.5x |
| Contributions | 0.5x / 1x / 1.5x | 1.5x |
| Retirement age | 60 / 65 / 70 | 1.3x |
| Width of the estimation-error band | 1x / 2x / 3x | 1.3x |
| Long-term care shock | none → $240k/yr for 3 years | 1.25x |
| Uncertainty in the cash rate | 0 / 1 / 2pp | 1.24x |
| Inflation uncertainty | 0 / 1.5 / 2.5pp | 1.11x |
| Inflation | 2% / 3% / 4% | 1.09x |
| Social Security claiming age | 62 / 67 / 70 | 1.08x |
| Inflation pass-through to equities | −1.15 / −0.80 / 0 | 1.06x |
| Fat tails (Student-t crisis months) | none / df 5 / df 3 | 1.06x |
| Trading costs | 5 / 15 / 40bp round trip | 1.04x |

The named sweeps are level-only too: regional premia (international catching
up raises spending 23%, international lagging lowers it 28%), factor scenarios
(any one factor dead costs 16–20%), the historical-means return method (+35%, measured before a 2026-10-26 data
fix lowered several factor funds' historical means; not re-run),
independent rather than shared estimation errors (+12%), the guardrail
response (`slow` spends 40% less than `fast`), the cash rate's source, uniform
rebalancing bands (no change) and the TIPS inflation beta (no change).

**What it means:** you can choose a portfolio without first winning the
argument about the cash rate, inflation or how uncertain the parameters are.
Those arguments decide how much you can spend.

```bash
python3 tools/sensitivity.py rf_real 0,0.0057,0.0298 --sims 800 --gk-response fast
```

## 2. The factor tilt is an asymmetric bet

| Factor premia | 145 (plain index) vs 141 (tilted, same shape) |
|---|---|
| 100% of historical | 145 WORSE by 12.7–16.1% |
| 0% | 145 BETTER by 0.2–3.2% |

On 168 at zero premia, 145, 140 and 161 all tie it; at full premia 140 trails
it by 6.0–9.4%. With no premium the tilts earn nothing and their higher fees
remain, so the cheaper Vanguard version edges ahead of 141 by a little, and
ties 168. The tilt gains a lot if
the premia are real and loses a little if they are not. It is the wrong bet
only if you are confident they are gone.

`MC_PREMIUM_SCALE` scales FACTOR premia over the market, not the equity
premium: at zero, VOO still earns 7.5% real against BND's 2.1%.
`MC_PREMIUM_HAIRCUT` moves the equity premium itself.

```bash
MC_PREMIUM_SCALE=0.0 python3 run.py --spec '{"mode":"retirement","portfolios":["141","145"],"policy":"guyton_klinger","gk_response":"fast","spend":"solve","verify_sims":3000,"search_sims":1500}'
```

## 3. Bonds cost spending under flexible withdrawals

Against holding no bonds, every strategy at one plan:

| Sleeve | Lifetime spending |
|---|---|
| 10% permanently from 65 (141) | WORSE by 7.5–9.8% |
| 10% from 65 to 70, then back to equity (141) | WORSE by 1.5–3.5%, and no decade's low case improves |
| 5% from 60 to 65 (168) | tie (−1.0..+0.2%) |
| 10% from 60 to 65 (168) | tie (−1.5..+0.0%) |
| 10% held 60 to 70 (168) | WORSE by 1.5–3.9% |
| 30% held 60 to 70 (168) | WORSE by 6.8–11.1% |

- **The bond fund does not change it.** Held at 30% from 60 to 70, every bond
  fund costs 8–14% and none lifts a decade's low case. Head to head against a
  BND sleeve, no fund resolves BETTER: most tie, and a few resolve slightly
  worse in some runs (ultra-short VUSB; short TIPS STIP in one; ultra-short
  JPST when inflation risk is doubled). Short TIPS buy a slightly better low
  case only when inflation risk is doubled (`MC_INFLATION_STD=0.03`).
- **Nor does the correlation history.** On a 35-year correlation window that
  holds 2000–02 and 2008, where Treasuries hedge in a crisis, the verdicts are
  the same.
- **Nor does fixed spending.** Under `constant_real`, the chance of running out
  is the same with or without any sleeve (about 10%). Under `vpw` even the
  cheapest sleeve (5% from 60 to 65) is 0.1–2.2% worse.
- **If you hold a sleeve anyway:** exiting at 70 beats holding forever
  (+5.5–7.7%). Five years beat ten or fifteen, and before retirement beat
  after, in a 2026-10-01 run on an earlier adequacy rule; later sleeves agree
  in direction.

The likely reason, not measured: guardrail cuts already absorb a bad start, so
the bonds buy insurance the policy duplicates.

```bash
python3 run.py --spec '{"mode":"retirement","portfolios":["141"],"custom":["141","141>141+10@65","141>141+10@65>141@70"],"policy":"guyton_klinger","gk_response":"fast","spend":"solve","verify_sims":3000,"search_sims":1500}'
```

## 4. Every tilt in a diversified portfolio earns its place

Removing one tilt at a time from 168 (its weight moved to VOO), scored under
five return models (2026-10-26):

| Change | Lifetime spending |
|---|---|
| without small value (AVUV) | WORSE by 11.1–14.3% |
| without momentum (VFMO) | WORSE by 8.8–11.8% |
| without large value (AVLV) | WORSE by 4.3–7.2% |
| MTUM in place of VFMO | WORSE by 1.2–4.3% |

168 is never resolved behind the best of these in any return model (its worst
case is 0.8%), and with no factor premia they all tie.

**Large value pays through AVLV, not VTV.** The model credits AVLV with 1.73pp
a year over VOO and VTV with 0.85pp; at a 10% weight the second is 8.5 basis
points a year, below what the simulation can see. A VTV sleeve has never
resolved against no sleeve. AVLV's edge also holds in the historical-means
view.

## 5. Which international sleeve: minimax regret

Which international mix wins depends on the return model, so each portfolio is
scored by REGRET: how much median lifetime spending it gives up against the
best strategy in each of five return models (factor model, international
keeps lagging, international catches up, no factor premia, historical means),
at one plan on the same paths (2026-10-26). 165 (worst case 1.8%) and 168
(2.2%) tie for the smallest worst case; 161 is WORSE than 168 by 1.1–4.0%, and
166 ties.

- **The return METHOD decides the order among these, the regional scenario
  does not.** Under historical means 161's all-Avantis sleeve leads and the
  DFWIX-based sleeves trail it by up to 4%, but those means cover only 2005
  onward for the international funds against 1970 for VOO, so read that view
  as a stress test.
- **"International keeps lagging" lowers the level by 20–30%** and resolves
  no difference between the mixes.

```bash
python3 run.py --spec '{"mode":"retirement","portfolios":["168","161","165","166"],"policy":"guyton_klinger","gk_response":"fast","spend":"solve","verify_sims":3000,"search_sims":1500,"return_views":true}'
```

## 6. A single stock earns its factor loadings, and then pays little

A single stock's expected return is built like every fund's, from its factor
loadings, plus a stated edge (`MC_SINGLE_STOCK_EDGE`, default 0) — never its
own historical mean, which is the alpha that put it in the universe. With no
edge, a 5% COST position ties the same portfolio with that 5% in VOO. The edge
is the one assumption besides the premia that reorders portfolios:

| Edge over its factor build-up | Holding COST vs not |
|---|---|
| −2pp/yr | WORSE by 0.1–2.7% |
| 0 | tie |
| +2pp/yr | BETTER by 1.7–4.8% |
| +7pp/yr (roughly its 20-year history) | BETTER by 8.2–10.9% (measured on 165) |

Twenty years of one stock at 19% volatility pins its mean to about ±4pp, so the
data cannot tell these apart. `python3 tools/sensitivity.py single_stock_edge`
runs the sweep.

## 7. Retirement choices

- **Retirement age is a level, and it is yours to choose:** 1.3x of lifetime
  spending from 60 to 70. The **Retirement choices** route in `run.py` solves
  each option against your current plan on shared paths.
- **Social Security claiming age is level-only, and the horizon decides it.**
  Planning to 95, claiming at 67 spends 1.1–2.6% more over the lifetime than
  70, with the better low case to 84; 70 has the better low case from 85.
  Planning to 105 (2026-10-09, an earlier engine), the edge disappeared and
  70 had the better low case from 91. Claiming at 62 costs about 8%. The model is one person, so it cannot
  see the survivor benefit that makes delaying worth more in a couple.

## 8. How to read a result

These are what the model learned about itself, often by being wrong first.

- **Rank on realised spending by life stage, never on the plan level.** The
  plan level is a 90%-confidence bar, and four separate defects were one shape:
  ranking on it while reporting spending. Plan-level sweeps produced five false
  reversals before the reports were changed.
- **A solved plan level is noisy.** At 3,000 paths, five independent path sets
  put the same portfolio's level about 15% apart. Compare levels only paired.
- **Use at least 800 paths for sweeps.** At low premia the success curve is
  flat; a 300-path sweep once inverted a ranking that 800 paths did not.
- **The headline median is imprecise at any feasible path count:** about ±7%
  at 2,000 paths, and ±1% would need about 87,000. Paired differences are far
  tighter than either level.
- **Sequence risk is real and modest.** Holding the whole-retirement return
  fixed, the order of the first decade's returns moves the outcome 4–5% for
  every portfolio here, 15% for a single world fund (measured 2026-09).
- **Estimation error is 42–48% of the spread in spending** (measured 2026-09),
  and it is shared: one error per factor premium moves every fund that loads
  on it, so a portfolio cannot average it away.

## 9. What the model cannot tell you

- **A fund's expected return is not recoverable from its history.** A century
  of market data leaves a 7pp interval, and ±1pp would take about 340 years.
  `python3 tools/estimability.py` shows this on the repo's own data. That is why
  this page reports rankings and spans, and why every level is conditional on
  the return model.
- **Survivorship bias is smaller than it looks.** Most backfill proxies are
  index or systematic funds, and the factor method discards each fund's alpha,
  the channel selection bias would travel through. Under
  `MC_MU_METHOD=historical` neither defence applies.
- **Taxes are not simulated.** They were measured to move the level, not the
  portfolio decision.

---

The model's design is in [METHODOLOGY.md](METHODOLOGY.md). How to run it is in
[GUIDE.md](GUIDE.md).
