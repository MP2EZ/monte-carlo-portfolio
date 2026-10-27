# Monte Carlo Portfolio Analysis with Regime-Switching

A retirement and portfolio simulator for long horizons (25–60 years). It asks
how much you can spend at a given confidence, and whether a portfolio choice
still holds once you allow for the model being wrong.

**v3.5 (October 2026)** · Python 3.10+ · MIT · all figures in REAL (today's) dollars

## What it answers

The interactive runner (`python3 run.py`) offers three questions. Each one is
named by what it varies:

| Question | What varies | What stays fixed | Answer |
|---|---|---|---|
| **Portfolios** | portfolios, bond sleeves, glide paths | your plan | paired head-to-head at one plan, then a check under four other return models |
| **Retirement choices** | retirement age, claiming age, savings, withdrawal rule | one portfolio | each option solved and compared with your current plan on the same paths |
| **Market & life assumptions** | returns, cash rate, inflation, lifespan, care costs | everything you choose | how far the figures move, and whether anything changes the portfolio ranking |

## What the model includes

- **Regime-switching volatility.** Each fund moves monthly between low, medium
  and high volatility on a Markov chain, and those moves are tied to the market's
  own regime.
- **Regime-dependent correlations.** Correlations rise in crises: US–developed
  markets goes from 0.65 to 0.91, and nearly every fund pair rises with it.
- **Parameter uncertainty separated from market noise.** Each path draws its own
  estimation error. That error flows through the factor premia the funds share,
  so a portfolio cannot diversify it away.
- **Long, spliced histories.** Ken French and AQR academic data plus proxy funds
  extend each fund back to 1963–1973. Every proxy is scored against the fund it
  stands in for.
- **Retirement spending.** Four withdrawal policies (`constant_real`,
  `guyton_klinger`, `vpw`, `fixed_pct`), Social Security, long-term care shocks
  and glide paths.
- **An inverse solver.** It finds the largest spending plan the portfolio can
  support at 90% confidence, judged on the worst rolling 10-year stretch of
  spending rather than a lifetime average.
- **Paired comparisons.** Every strategy runs on the same simulated markets, so
  differences between strategies come with confidence intervals that mean
  something.
- **Reproducibility.** Every run is saved with its provenance. `run.py --diff`
  explains why a result changed, and `run.py --spec` replays a run headlessly.

## Quickstart

```bash
pip install -r requirements.txt
```

```bash
python3 data_collection_regime_model.py
```

```bash
python3 tools/validate_all.py
```

```bash
python3 run.py
```

1. The install pulls numpy, scipy, pandas, yfinance and matplotlib.
2. `data_collection_regime_model.py` downloads market data and builds `output/`.
   It takes about 10 minutes and needs a network connection.
3. `validate_all.py` runs the structural, data and behaviour checks. Every check
   must pass.
4. `run.py` starts the interactive runner. Open **Your details** (option 5)
   first to enter your age, accounts and Social Security. They are saved to
   `config/investor.json`, which is gitignored.

Until you enter your details, runs use a placeholder investor (age 37, $300k
across five accounts) and print a warning saying so. Every prompt has a
default, so a typical run is mostly pressing Enter. `python3 run.py --again`
repeats the last run.

## Using it with Claude Code

The repo is set up for [Claude Code](https://claude.com/claude-code): open the
folder and ask in plain English. `.claude/CLAUDE.md` gives Claude the
architecture, and these skills give it the procedures:

| Skill | What it does |
|---|---|
| `/setup-portfolio` | Swap the fund universe for your own funds and build sample portfolios |
| `/add-ticker` | Add or remove a single fund |
| `design-runs` | Turn a question ("what if I retire at 60?") into a reproducible `run.py --spec` command, and read the result |
| `refresh-parameters` | The quarterly data refresh, and what a drift failure means |
| `engine-change` | Change the simulation without breaking determinism or the golden masters |
| `verify-test` | Check that a new test can actually fail |
| `remeasure` | Find every place a measured number lives after something moves it |

The manual procedure for changing funds is in
[docs/CUSTOMIZING.md](docs/CUSTOMIZING.md).

## Sample output

From the placeholder investor (age 37, $300k, retiring at 65, planning to 95),
`guyton_klinger`, portfolios 140 and 145 at 140's plan level:

```
  SPENDING BY LIFE STAGE   real $/yr — the lifetime average, split by decade
  every strategy at 140's $71k/yr plan
              lifetime │      65-74      │      75-84      │      85-94      │    estate
    strategy       avg │  median     low │  median     low │  median     low │    median
    ------------------------------------------------------------------------------------
    140           213k │     99k     69k │    191k     72k │    348k     81k │     40.7M
    145           178k │     95k     66k │    165k     61k │    269k     60k │     25.9M
```

`low` is the 10th percentile within each decade. Compare strategies on this
table. The "plan level" is the level the solver certified, not a forecast of
what you will spend: under a flexible policy, spending ratchets up as the
portfolio grows.

## What it has found

These results were measured on one investor's plan and this repo's portfolio
library. Your levels will differ; the shape of each result is the useful part.
Each is set out with its evidence in [docs/RESULTS.md](docs/RESULTS.md).

- **Most uncertainty changes how much you can spend, not which portfolio is
  best.** 22 of 24 assumption sweeps move only the spending level. The real cash
  rate swings it 4.3x and the factor premia 1.6x.
- **The factor tilt is an asymmetric bet.** At full factor premia a tilted
  portfolio leads the same shape in cheaper Vanguard funds by 13–16% of
  lifetime spending; at zero premia the Vanguard one edges ahead by up to 3%.
- **Bonds cost spending under flexible withdrawals.** A permanent 10% bond
  sleeve from 65 costs 7–10% of lifetime spending under Guyton-Klinger, and
  neither TIPS nor Treasuries change that.
- **Rank on spending by life stage, never on the plan level.** Ranking on the
  plan level produced five false reversals before the reports were changed.

## Documentation

| Doc | What's in it |
|---|---|
| [docs/GUIDE.md](docs/GUIDE.md) | Running the model: workflow, your details, how many paths, reading the output, troubleshooting |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | The fund universe, regimes, correlations, history splicing, parameter uncertainty, validation |
| [docs/RESULTS.md](docs/RESULTS.md) | What the model has concluded, and how firmly |
| [docs/ERD.md](docs/ERD.md) | Every file, record and registry the engine passes between its parts |
| [docs/CUSTOMIZING.md](docs/CUSTOMIZING.md) · [docs/RUNBOOK_tickers.md](docs/RUNBOOK_tickers.md) | Changing the fund universe |
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [.claude/CLAUDE.md](.claude/CLAUDE.md) | Architecture notes, written for AI coding agents |

## Limitations

- **Expected returns are the weakest input.** A century of market data still
  leaves a 7pp interval on a fund's mean return (`tools/estimability.py` shows
  this on the repo's own data). Treat the spending levels as conditional on the
  return model, and use the sensitivity sweeps and return views to see how far
  they move.
- **Taxes are not simulated.** The engine models one pooled portfolio. Taxes
  were measured to change the level but not the portfolio decision.
- **One person, not a couple.** The model cannot see a survivor benefit, which
  makes delaying Social Security worth more in a couple.
- **The fund universe reflects one investor's accounts.** The comparisons are
  only as relevant as the funds you can actually hold.

## License & disclaimer

[MIT](LICENSE). This is a personal research project, not investment advice.
Simulations rest on historical data and assumptions, and actual returns may
differ substantially. Consult a qualified advisor before making financial
decisions.
