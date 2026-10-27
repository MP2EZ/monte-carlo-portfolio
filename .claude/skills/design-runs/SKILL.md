---
name: design-runs
description: This skill should be used whenever the user asks to run, compare, test or design a simulation, or asks for a command — e.g. "compare 165 and 168", "test adding AVLV", "what if I add bonds 60-65", "try it under constant_real", "what if I retire at 60", "give me the command for…", "add this allocation as a portfolio", "design a series of runs". It is the reference for run.py's --spec format, the glide/schedule syntax, the recipes for each kind of question, and how to read the result. Use it BEFORE writing any run.py command.
---

# Design and hand over a run

Every measurement in this repo is a `python3 run.py --spec '<json>'` command,
run from the project root. The user runs long ones in their own terminal; you
design them, check them, hand them over, and read the output back.

**`tools/check_logic.py` parses every `--spec` command in this file.** If you
change a recipe, run `python3 tools/validate_all.py`; a recipe that no longer
validates fails the suite.

## 0. Before you design anything

1. **Which route is the question?** (CLAUDE.md, "Where a question goes")
   - Portfolios / funds / bond sleeves change, plan fixed → **§3 recipes A–C**.
   - Retirement age, claiming age, savings, withdrawal rule change → **§3 D–E**.
   - Market assumptions you don't control → **§3 F** (sensitivity tool).
2. **Check what is settled.** CLAUDE.md "What is already settled" and
   `docs/RESULTS.md`. Do not re-run a settled comparison unless an input
   changed (engine, data, the investor record, a new portfolio).
3. **The baseline is the user's own allocation** — the portfolio id CLAUDE.md
   names as theirs. If none is named, ask which row it is, or add it with
   `tools/update_holdings.py` (§4). Compare proposals against it.
4. **The horizon, ages and Social Security come from `config/investor.json`**
   (`plan_to_age`, `retirement_age`, `social_security`), NOT from the spec.
   The run header prints them (`ages 37–95`) — confirm they are what the user
   expects before quoting a level.

## 1. The spec — every key `parse_spec` / `build_config` reads

| key | type | default | meaning |
|---|---|---|---|
| `mode` | `"retirement"` \| `"accumulation"` | `"retirement"` | spending in retirement, or wealth to the horizon (no withdrawals) |
| `portfolios` | list of ids (strings) | **required** | CSV `portfolio_id`s. With `custom`, still required (validated) but `custom` decides what runs |
| `custom` | list of glide specs | — | the exact strategies to run, in order; see §2. Include the plain portfolios too — they are the controls |
| `spend` | `"solve"` \| number | `"solve"` | solve the 90%-confidence plan level, or simulate a fixed real $/yr |
| `policy` | `guyton_klinger` \| `constant_real` \| `fixed_pct` \| `vpw` | `guyton_klinger` | withdrawal rule |
| `gk_response` | `slow` \| `medium` \| `fast` | `fast` (policies default) | how fast guyton_klinger follows a rising portfolio |
| `compare_policies` | list of policies | — | one solve per policy (a fan-out). Prefer one run per policy (recipe C) when you need each rule's head-to-head |
| `compare_gk_responses` | list | — | fan-out over guardrail responses |
| `vpw_bequest` / `compare_vpw_bequests` | number / list | 0 | real $ VPW holds back |
| `floor` | 0–1 | 0.85 | per-life-stage adequacy floor the solver certifies |
| `sourcing` | `excess` \| `pro_rata` | `excess` | which funds withdrawals come from |
| `verify_sims` / `search_sims` | int | 1500 / 400 | paths. **Use 3000 / 1500** for anything quoted; 6000 / 1600 to tighten a close pair |
| `sims` | int | 2000 | paths for `spend` ≠ solve and for accumulation |
| `return_views` | bool | false | re-run every strategy under 4 other return models (≈ ×5 cost) |
| `compare_choice` | `{"dimension": d, "values": [..]}` | — | a retirement choice; `d` ∈ `retirement_age`, `ss_claim_age`, `contributions` (multiplier) |
| `derisk_at` + `bond_tiers` / `bond_sleeve` / `compare_sleeves` | age, list, sleeve, list | — | generated bond ladders (older route; `custom` is clearer — prefer it) |
| `social_security` | object \| null | from investor.json | per-run override; `null` excludes it |

CLI flags: `--dry-run` (resolve, validate, price — costs nothing),
`--sims N` (sets verify N / search N/2), `--ask "sentence"` (needs the
`anthropic` package).

## 2. Glide / schedule syntax (`custom` entries)

```
168                          hold 168 for the whole horizon
168>168+5@60>168@65          from 60 hold 168 with 5% bonds; back to 168 at 65
168>168+10@65                from 65 hold 168 with 10% bonds, permanently
168>168+5:stip_only@60>168@65   same 5% sleeve, all short TIPS
```

- `+N` synthesises N% bonds from the base portfolio's OWN equity sleeve, so
  only the bond weight differs from the control.
- `:sleeve` after the percent picks the bond mix: `core` (default: BND 60 /
  JPLD 25 / JPST 15), `core_tips` (STIP for JPST), `thirds` (BND/JPLD/STIP), `quarters` (BND/JPLD/JPST/STIP), `core_vgit`, `core_vgit_tips`, `tsy_tips` (VGIT 70 / STIP 30), `quarters_vgit`, `vanguard`, `bnd_only`, `jpld_only`, `jpst_only`,
  `vusb_only`, `bsv_only`, `stip_only`, `vgit_only` (`loaders.BOND_SLEEVES`).
- Ages must strictly increase; only the first waypoint may omit `@age`.
- `parse_spec` validates every schedule, so `--dry-run` catches a typo.

## 3. Recipes — one per question

Defaults in every recipe: guardrails (`guyton_klinger`, fast), solve, 3,000 /
1,500 paths. Swap ids for the user's. **Always `--dry-run` first** and quote
the estimate (`est. N min`); then hand over the real command.

### A. Does fund X earn its place? (a with/without pair)

Build the twin first (§4): the same portfolio with X's weight folded into
VOO. Run both with the return-model check, ideally with a second pair on a
different base to show the effect is not one portfolio's quirk:

```bash
python3 run.py --spec '{"mode":"retirement","portfolios":["168","169","165","152"],"policy":"guyton_klinger","gk_response":"fast","spend":"solve","verify_sims":3000,"search_sims":1500,"return_views":true}'
```

### B. Do bond sleeves help this portfolio?

```bash
python3 run.py --spec '{"mode":"retirement","portfolios":["168"],"custom":["168","168>168+5@60>168@65","168>168+5@65>168@70","168>168+10@60>168@70"],"policy":"guyton_klinger","gk_response":"fast","spend":"solve","verify_sims":3000,"search_sims":1500}'
```

Several portfolios × sleeves in one run is fine (18 strategies ≈ 1 hour);
leave `return_views` off — sleeves do not touch the return model.

### C. The same comparison under every withdrawal rule

One run per rule, so each gets its own head-to-head. Guardrails is usually
already done (recipe B):

```bash
for p in constant_real fixed_pct vpw; do python3 run.py --spec '{"mode":"retirement","portfolios":["168"],"custom":["168","168>168+5@60>168@65","168>168+5@65>168@70"],"policy":"'"$p"'","spend":"solve","verify_sims":3000,"search_sims":1500}'; done
```

Under `constant_real` spending is fixed, so there is NO head-to-head table —
read **P(run out)** and the low case by decade instead.

### D. A retirement choice (retire at 60? claim at 70? save more?)

The LEVEL is the answer here — each option is solved and read against the
current value, on shared paths:

```bash
python3 run.py --spec '{"mode":"retirement","portfolios":["168"],"policy":"guyton_klinger","gk_response":"fast","spend":"solve","verify_sims":3000,"search_sims":1500,"compare_choice":{"dimension":"retirement_age","values":[60,62,65]}}'
```

### E. Accumulation: wealth, CAGR and drawdown to the horizon

```bash
python3 run.py --spec '{"mode":"accumulation","portfolios":["168","169","140"],"sims":2000}'
```

### F. A market assumption (premium, cash rate, inflation, regions, COST's edge…)

```bash
python3 tools/sensitivity.py premium 1.0,0.5,0.0 --sims 800 --gk-response fast --strategies 168,169
```

Run `python3 tools/sensitivity.py` bare for the dimension menu. It reports
whether the assumption moves the LEVEL or the RANKING.

### Env knobs — prefix a command to change the model for one run

`MC_PREMIUM_SCALE` (factor premia ×k), `MC_PREMIUM_HAIRCUT` (all premia incl.
equity), `MC_PREMIA_SCENARIO` (`value dead`, `intl keeps lagging`, …),
`MC_MU_METHOD` (`factor` \| `historical`), `MC_SINGLE_STOCK_EDGE` (COST's
edge, decimal), `MC_REBALANCING_BANDS` (`model` \| `uniform`), `MC_TAIL_DF`
(fat tails, 0 = off), `MC_CRISIS_VOL_SCALE`, `MC_RF_REAL`, `MC_INFLATION`,
`MC_PARAM_STD_SCALE`, `MC_PARAM_ERRORS`, `MC_ROUND_TRIP`, `MC_CORR_WINDOW`
(years behind the correlation matrices; generate a set first with
`python3 data_collection_regime_model.py --corr-window-years 35`). The full list with
defaults is `diagnostics.DEFAULT_ASSUMPTIONS`; a moved knob is announced in
the report banner. Example:

```bash
MC_PREMIUM_SCALE=0.0 python3 run.py --spec '{"mode":"retirement","portfolios":["168","145"],"policy":"guyton_klinger","gk_response":"fast","spend":"solve","verify_sims":3000,"search_sims":1500}'
```

## 4. Adding a portfolio

1. Pick an unused id in the right family (tens digit = tilt; see docs/GUIDE.md
   "Portfolio Templates"); check `grep '^1[0-9][0-9],' portfolios/portfolio_templates.csv`.
2. Weights must sum to exactly 1.0 and use only `loaders.MODEL_TICKERS`.
3. **The `name` field must contain NO commas** — the CSV is unquoted, and a
   comma shifts every column (this broke 156 once).
4. Insert the row next to its family, then:
   ```bash
   python3 tools/annotate_templates.py && python3 tools/add_section_headers.py && python3 tools/validate_all.py
   ```
5. A **twin** (recipe A) is the same row with the tested fund's weight moved
   into VOO; an **international-weight variant** scales every international
   fund by the same factor (how 166 is built from 165, and 156 from 152).
6. Commit only after `validate_all` prints all-PASS — chain with `&&`, never `;`.

## 5. Reading the output

- **The verdict is the HEAD-TO-HEAD table**: lifetime spending vs the first
  strategy, all at ONE plan on the SAME paths. A range excluding zero is a
  real difference; "not resolved" means the model cannot tell — not "equal".
- **Never rank on the plan level** (`$NNNk/yr plan level` in SOLVING). Five
  false reversals came from it. It is the starting point, not the verdict.
- **SPENDING BY LIFE STAGE**: median and low (10th percentile) by decade —
  use it to say whether something protects the bad case.
- **IF THE RETURN MODEL IS WRONG** (with `return_views`): per-view regret;
  the `←` row is the minimax pick (smallest worst case). Say which view each
  strategy's worst case comes from.
- **constant_real**: no head-to-head; read P(run out) and the low case.
- When you report: lead with the answer, quote intervals as printed
  (`−8.5% .. −5.4%`), give the run's saved file (`output/runs/run_….json`),
  and record a new result in `docs/RESULTS.md` (dated, with its spec).
- A run's output is long; read it with the terminal tool from the end, or
  search the saved text, rather than asking the user to paste it.

## 6. Handing a command to the user

- One command per ```bash block, runnable from the project root.
- Give the time estimate from `--dry-run`.
- Say what to look at when it finishes, and offer to read it from the
  terminal.
- Long batches you run yourself go in the background with a resumable
  `output/studies/remeasure_<date>/batch.py` (see earlier ones) — never a foreground
  command over ~2 minutes.
