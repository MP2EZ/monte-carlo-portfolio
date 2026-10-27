"""
Factor loadings and premia — the LEVEL half of the forecast.
============================================================

The one-paragraph version of the design. A mean return has standard error
sigma/sqrt(T), which depends on calendar span alone, so AVLV's 13.09% carries a
genuine 95% interval of 8.2%-18.0% on 54 years of data and pinning it to +/-1pp
would take 1,301 years. The LEVEL is therefore not recoverable from a return
series and no better proxy fixes that. What IS recoverable is SHAPE — betas,
volatility, correlation — so the level gets built from stated premia instead,
and varied across named scenarios rather than pinned to one number.

Three rules here that were each arrived at by measuring, and each of which is
easy to get backwards:

REGIONS ARE FITTED, NOT MAPPED (docs/METHODOLOGY.md, Expected Returns). Every fund lands on its
own region by adjusted R2 with a 0.09-0.36 margin, except VT at 0.013 — and VT
is the one that matters, because its best single region carries alpha -2.93%,
the worst of any fund clearing the gate, and VT is the benchmark every tilt is
measured against. Regressing on the union recovers 53/34/13% US/dev/emerging
against an actual ~60/30/10, so a blend does not need composing from config.

LOADINGS COME FROM THE ACTUAL FUND, NOT THE SPLICED SERIES. Beta converges
fast enough that 58 months is plenty, so there is no reason to buy window at the
cost of bias — and spliced loadings are dominated by the proxy (AVLV: 561 proxy
months against 83 actual). Measured, 4 of 6 diverge, and the two that "agree"
are the two with the shortest actual histories, where the standard errors are
wide enough to swallow the difference.

THE BUILD-UP IS GROSS OF COSTS. The regression runs on NET returns, so
E[r_net] = rf + sum(beta*E[F]) + alpha with alpha carrying -expense
-implementation, while the Ken French factors are academic portfolios with no
fees and no trading costs. Dropping alpha ADDS BOTH COSTS BACK. The natural
intuition — "the shortfall is already in the fund's returns, so subtracting it
again double-counts" — is backwards: it is in the returns because it is in
alpha, which is exactly what gets discarded.
"""

import json as _json
import os as _os
from pathlib import Path as _Path

import numpy as np

FACTORS = ('mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom')
REGIONS = ('us', 'dev', 'emg')

# THE REAL CASH RATE — the single biggest LEVEL lever in this model (7x on final
# wealth, larger than the equity premium's 2x) and the only input that can be
# QUOTED rather than estimated. TIPS price it directly. The REAL rate is the
# input and the nominal is derived from it and the inflation assumption, so the
# two cannot drift apart. MC_INFLATION is read here as well as in engine, both
# from the environment with the same default, because workers re-import under
# `spawn` and an assigned module global would not reach them.
#
# Resolution, highest precedence first:
#   MC_RF_REAL=0.02        state it yourself
#   MC_RF_SOURCE=tips30    take the live quote (tips5/10/20/30, or tips30_avg
#                          for that tenor's own full-history average)
#   default                0.0057, below
#
# WHY THE QUOTE IS NOT THE DEFAULT, which is the opposite of what roadmap item 9
# expected. "Replace a guess with a quote" trades estimation error for TIMING
# risk, and measured on 2026-09-09 that is a bad trade here:
#
#     tenor    spot     full-history avg     range since 2010
#       5y    2.20%          0.58%
#      10y    2.46%          1.00%           -1.19% .. 3.15%
#      20y    2.79%          1.27%
#      30y    2.98%          1.16%           -0.59% .. 3.06%
#
# Today's 30-year print is at the TOP of its entire history; adopting the spot
# would anchor a 58-year horizon on a single day at a 16-year extreme.
#
# The quote is still the right thing for stating a forward-looking position, and
# `MC_RF_SOURCE=tips30` is one command away. `tools/sensitivity.py` sweeps this
# dimension, which is the interface this model actually prefers for a lever
# nobody can know.
#
# MEASURED, not derived. 4.35% nominal minus an ASSUMED 3% inflation gives
# 1.31%, which is what this default was until 2026-09-11 — but 1963-2026
# actually ran 3.79% CPI inflation, so subtracting the assumption from a nominal
# average earned under different realised inflation overstated the real rate by
# ~0.74pp. The plumbing was fixed on 2026-09-10 to make the REAL rate the input;
# the constant itself was carried over from the nominal-first world and never
# re-derived.
#
# Now the realised real cash return over the same window the premia use
# (Ken French rf vs CPI-U, 1963-07..2026-07, compounded not subtracted):
#
#     window            nominal rf   realised CPI   REAL
#     1963-07..2026-07     4.35%         3.79%     +0.57%
#     1947-02..2026-07     3.83%         3.46%     +0.38%
#
# Note this is the SHORT rate, which is what the premia are excess over. Long
# TIPS average 1.16-1.27% because they carry a term premium of roughly 0.6pp;
# see MC_RF_SOURCE, which offers those and is deliberately not the default.
_RF_REAL_DEFAULT = 0.0057

# ...AND IT IS NOT KNOWN. The real cash rate is the biggest level lever in the
# model (3.58x on 141's lifetime spending, 2026-10-03) and the ONLY input that every asset shares, so
# leaving it a certainty while every ticker carries an epistemic band had the
# uncertainty backwards: independent per-ticker draws largely cancel across 25
# funds, while the common term — the one that cannot diversify away — had no
# band at all.
#
# Drawn ONCE PER PATH and added to every ticker's real mu, because a different
# cash rate moves all of them together. Not double-counted: parameter_std is
# built from PREMIUM_SE, beta_se and alpha_se, all of which are excess over the
# contemporaneous rf and carry no rf term.
#
# Sized from how much a LONG-RUN AVERAGE real cash rate actually varies —
# non-overlapping windows of the 1947-2026 realised series:
#
#     10y windows   sd 1.42pp   range -1.17 .. +2.36   (7 windows)
#     20y windows   sd 0.70pp   range +0.07 .. +1.42   (3 windows)
#     30y windows   sd 1.26pp   range -0.05 .. +1.73   (2 windows)
#
# 1.0pp sits between the 20y and 30y estimates. At +-2sd it spans -1.4% to
# +2.6%, which covers the realised 10-year range. MC_RF_REAL_STD=0 restores the
# old behaviour exactly.
RF_REAL_STD = float(_os.environ.get('MC_RF_REAL_STD', '0.010'))

# ---------------------------------------------------------------------------
# STOCHASTIC INFLATION — as a shared shock with ASSET-SPECIFIC pass-through
# ---------------------------------------------------------------------------
# Everything downstream of engine.py is in real dollars deflated by a CONSTANT
# INFLATION, so until 2026-09-11 a nominal bond carried no inflation risk at
# all. That is the last thing that changes the SHAPE of the answer rather than
# its level — but only because of the DIFFERENTIAL below. A shock that hit every
# asset equally would be level-only and structurally identical to RF_REAL_STD.
#
# Measured 2026-09-11: annual real return regressed on realised CPI, 1947-2026,
# then repeated on real return in EXCESS of real cash so the part the rf band
# already carries is removed. The cash channel itself is small — real cash vs
# inflation is -0.29 (t = -3.1) — so most of the exposure is incremental:
#
#     ticker   yrs   excess slope     t     resolved?
#     BSV       33       -1.04      -3.3      yes
#     JQUA      54       -2.04      -2.8      yes
#     BND       40       -1.26      -2.6      yes
#     VOO       57       -1.34      -1.8      no
#     VTV       54       -0.87      -1.2      no
#     AVUV      64       -0.74      -0.8      no
#
# BOTH bond funds resolve and are near -1.15. Equities cluster around -0.8 but
# only 1 of 14 resolves, and the international tickers have 21-36 years with
# standard errors of 2-4 — those readings are noise, not evidence of hedging.
#
# So: bonds are measurably hurt by inflation beyond the cash channel, equities
# are probably hurt somewhat less, and the DIFFERENCE (~0.35pp per 1pp) is not
# statistically resolved. The defaults encode the measurement; the equity figure
# is the weaker of the two and MC_INFL_BETA_EQUITY exists to argue with it.
#
# Setting both to the same value collapses this to another level-only shock.
# Setting equity to 0 is the "equities are a real asset" position, which the
# data does NOT support at this horizon (Fama-Schwert 1977 found the same).
INFL_BETA_EQUITY = float(_os.environ.get('MC_INFL_BETA_EQUITY', '-0.80'))
INFL_BETA_BOND = float(_os.environ.get('MC_INFL_BETA_BOND', '-1.15'))
# TIPS (loaders.INFLATION_LINKED): principal indexed to CPI, so an inflation
# surprise leaves the REAL return alone — 0 by construction. Measured the same
# way as the two above, on STIP's spliced history (TRBFX to 2010-11), 2007-2025:
#
#     ticker   yrs   excess slope    SE      t
#     STIP      19       +0.25      0.40   +0.6    consistent with 0
#     BND       19       -1.67      0.53   -3.1    same years, resolved
#
# 19 years cannot pin a slope, but they separate the two cleanly; the
# construction sets the value and the measurement does not contradict it.
INFL_BETA_TIPS = float(_os.environ.get('MC_INFL_BETA_TIPS', '0.0'))

# DURATION. INFL_BETA_BOND was measured on BSV (2.6 years) and BND (~6), and
# until 2026-10-24 every nominal bond fund got it — JPST and VUSB too, which
# made a cash-like fund MORE exposed to inflation than equities (-1.15 against
# -0.80). The slope is excess return over real CASH, so a fund that reprices
# like cash carries none of it by construction: deflating it and cash by the
# same CPI cancels. What a short fund does lose is the price fall on what it
# holds when yields rise, which scales with duration.
#
# So below the shortest fund the slope was measured on, it scales linearly to
# zero at zero duration; at or above it a fund gets the full figure:
#
#     fund   duration   pass-through
#     JPST      0.5        -0.22
#     VUSB      0.95       -0.42
#     JPLD      1.9        -0.84
#     BSV       2.6        -1.15
#     BND       ~6         -1.15
#
# Linear is the stated assumption. The two measured points bend the other way
# (BSV -1.04, BND -1.26: concave), and a curve through them and zero puts JPST
# nearer -0.35 — but two points with t ~3 cannot pick a curve, and at an annual
# horizon an ultra-short fund also rolls into the higher yields within the
# year, which pulls the other way. Either reading leaves ultra-short far below
# BND and below equities, which is the part that can change a decision.
# MC_INFL_BETA_BOND still scales every nominal fund together.
INFL_REF_DURATION = 2.6


def bond_infl_beta(duration, beta=None):
    """Inflation pass-through for a nominal bond fund of this duration."""
    beta = INFL_BETA_BOND if beta is None else beta
    return beta * min(1.0, duration / INFL_REF_DURATION)

# Uncertainty in the LONG-RUN AVERAGE inflation rate, from non-overlapping
# windows of realised CPI 1947-2026: 10y sd 1.91pp, 20y sd 2.09pp, 30y sd
# 0.56pp. The 30-year figure rests on two windows and is not credible on its
# own; 1.5pp sits below the 10/20-year estimates and above the 30-year one.
#
# Note the realised 1947-2026 mean is 3.46% against the model's 3.00%
# assumption, so this band is centred slightly low by the same logic that made
# RF_REAL wrong. Left alone: INFLATION is also the deflator, and moving it
# changes what "real dollars" means for the whole model. MC_INFLATION sweeps it.
INFLATION_STD = float(_os.environ.get('MC_INFLATION_STD', '0.015'))
RF_REAL_SOURCE = _os.environ.get('MC_RF_SOURCE', 'historical')

# HOW ESTIMATION ERRORS ARE DRAWN: 'shared' (the default) or 'independent'.
#
# parameter_std builds each fund's band mostly from the standard errors of a
# few premia every fund holds — VOO's 1.89% real is essentially the US market
# premium's 1.95%. Drawing each fund's error on its own, as the engine did until
# 2026-10-02, let a portfolio average away uncertainty that is really COMMON to
# its funds: the mistake RF_REAL_STD fixed for the cash rate, left standing for
# the premia. Measured the day it changed, it understated the spread of a
# portfolio's expected return 1.4x (140/141/145) to 1.6x (161/165/166) — more
# for the value-tilted books, because AVLV stacks the value premium that AVUV
# and AVDV already carry.
#
# 'shared' draws one error per premium per path and passes it through each
# fund's loadings; the loading and alpha uncertainty stay each fund's own.
# Every fund's band is exactly the same under both settings — only the
# correlation between funds moves. Errors in different premia are correlated
# as measured (PREMIUM_CORR): a US market miss and a developed one err together.
#
# Read from the environment: workers re-import under `spawn`.
PARAM_ERRORS = _os.environ.get('MC_PARAM_ERRORS', 'shared')
if PARAM_ERRORS not in ('shared', 'independent'):
    raise ValueError(f"MC_PARAM_ERRORS={PARAM_ERRORS!r}; "
                     f"expected 'shared' or 'independent'")


def _quoted_real_rate(source):
    """Real yield from the TIPS quotes stored by data_collection_regime_model.

    Read from the parameters JSON rather than fetched, so a worker re-importing
    under `spawn` does not make a network call, and so a run is reproducible
    against the parameters it was generated with.
    """
    name = source[len('tips'):]
    want_avg = name.endswith('_avg')
    tenor = name[:-4] if want_avg else name
    f = _Path(__file__).resolve().parent.parent / 'output' / 'ticker_parameters_regimes_v8.json'
    quotes = (_json.loads(f.read_text()).get('metadata') or {}).get('tips_real_yields') or {}
    q = quotes.get(tenor)
    if not q:
        raise ValueError(
            f"MC_RF_SOURCE={source!r}: no stored {tenor}y TIPS quote. "
            f"Run data_collection_regime_model.py (needs network). "
            f"Have: {', '.join(sorted(quotes)) or 'none'}")
    return float(q['avg_full'] if want_avg else q['latest']), q['as_of']


RF_REAL_AS_OF = None
if 'MC_RF_REAL' in _os.environ:
    RF_REAL = float(_os.environ['MC_RF_REAL'])
elif RF_REAL_SOURCE.startswith('tips'):
    RF_REAL, RF_REAL_AS_OF = _quoted_real_rate(RF_REAL_SOURCE)
elif RF_REAL_SOURCE == 'historical':
    RF_REAL = _RF_REAL_DEFAULT
else:
    raise ValueError(
        f"MC_RF_SOURCE={RF_REAL_SOURCE!r}; expected 'historical' or "
        f"tips5/tips10/tips20/tips30 (optionally suffixed _avg)")
_INFLATION = float(_os.environ.get('MC_INFLATION', '0.03'))

# MC_RF_FORWARD still overrides the nominal directly, for anyone who wants to
# state it that way; leaving it unset is the coherent path.
RF_FORWARD = float(_os.environ.get(
    'MC_RF_FORWARD', (1.0 + RF_REAL) * (1.0 + _INFLATION) - 1.0))

# Historical premia per region, annual, over the contemporaneous risk-free rate.
#
# EACH FACTOR ON ITS OWN WINDOW, not on the months where every factor in the
# region happens to be available. The intersection was a silent and expensive
# choice: US momentum has 1195 months on disk and 757 were used, and emerging
# mkt_rf/smb/hml have 445 and used 409 because emerging CMA does not start until
# 1992. `PREMIUM_WINDOW` below records what each figure was actually measured
# on, so a mixed window is stated rather than hidden.
#
# Re-measured 2026-09-11 by `python3 tools/premia.py`, which prints these tables.
#
# Note `smb` is ~0 outside the US, so a size scenario bites the US sleeve far
# harder, while value runs the other way and is stronger abroad.
PREMIA = {
    'us':  {'mkt_rf': 0.0719, 'smb': 0.0225, 'hml': 0.0359,
            'rmw': 0.0308, 'cma': 0.0296, 'mom': 0.0741},
    'dev': {'mkt_rf': 0.0487, 'smb': 0.0041, 'hml': 0.0509,
            'rmw': 0.0324, 'cma': 0.0202, 'mom': 0.0793},
    'emg': {'mkt_rf': 0.0813, 'smb': 0.0135, 'hml': 0.0793,
            'rmw': 0.0282, 'cma': 0.0281, 'mom': 0.0998},
}

# Standard error of each premium, annual, on the same windows. sigma_f/sqrt(T) —
# the same un-estimability that makes a ticker's own mean useless
# (docs/METHODOLOGY.md, Expected Returns), just pooled over far more data than any single fund.
#
# The international market premia are much less pinned than the US one (2.72 and
# 3.36 against 1.95) because their windows are ~36 years against 63. Any report
# showing all three regions should say so rather than presenting them as equally
# known.
PREMIUM_SE = {
    'us':  {'mkt_rf': 0.0195, 'smb': 0.0132, 'hml': 0.0129,
            'rmw': 0.0100, 'cma': 0.0090, 'mom': 0.0163},
    'dev': {'mkt_rf': 0.0272, 'smb': 0.0113, 'hml': 0.0136,
            'rmw': 0.0079, 'cma': 0.0100, 'mom': 0.0198},
    'emg': {'mkt_rf': 0.0336, 'smb': 0.0133, 'hml': 0.0159,
            'rmw': 0.0103, 'cma': 0.0116, 'mom': 0.0181},
}

# (months, first month) behind each figure above. First-class because the
# windows are NOT equal and a reader comparing 'us' mom with 'emg' cma is
# comparing 1195 months against 409.
PREMIUM_WINDOW = {
    'us':  {'mkt_rf': (757, '1963-07'), 'smb': (757, '1963-07'),
            'hml': (757, '1963-07'), 'rmw': (757, '1963-07'),
            'cma': (757, '1963-07'), 'mom': (1195, '1927-01')},
    'dev': {'mkt_rf': (433, '1990-07'), 'smb': (433, '1990-07'),
            'hml': (433, '1990-07'), 'rmw': (433, '1990-07'),
            'cma': (433, '1990-07'), 'mom': (429, '1990-11')},
    'emg': {'mkt_rf': (445, '1989-07'), 'smb': (445, '1989-07'),
            'hml': (445, '1989-07'), 'rmw': (421, '1991-07'),
            'cma': (409, '1992-07'), 'mom': (439, '1990-01')},
}

# The premia a SHARED estimation error is drawn for (see PARAM_ERRORS), in a
# fixed order, so a draw's position never depends on what a plan holds.
PREMIUM_KEYS = tuple((region, f) for region in PREMIUM_SE for f in PREMIUM_SE[region])

# How the premia's ESTIMATION ERRORS move together, in PREMIUM_KEYS order —
# measured by `tools/premia.py` from the same series as PREMIUM_SE, scaled by
# how much of each pair's windows overlap. A US market miss and a developed one
# err together (+0.57), developed and emerging more so (+0.76), US value with
# investment (+0.68). Drawn independently (until 2026-10-02), a US miss and an
# international one were unrelated, which flattered spreading across regions.
PREMIUM_CORR = (   # rows and columns in PREMIUM_KEYS order
    (+1.00, +0.27, -0.21, -0.19, -0.36, -0.12, +0.57, -0.16, -0.14, -0.09, -0.31, -0.17, +0.53, -0.20, +0.02, -0.02, -0.23, -0.09),   # us mkt_rf
    (+0.27, +1.00, +0.02, -0.33, -0.08, -0.06, +0.16, +0.20, -0.02, -0.05, -0.11, -0.00, +0.20, -0.01, -0.04, -0.05, -0.03, -0.03),   # us smb
    (-0.21, +0.02, +1.00, +0.09, +0.68, -0.14, -0.02, +0.01, +0.53, -0.11, +0.35, -0.20, -0.07, +0.08, +0.25, -0.12, +0.13, -0.13),   # us hml
    (-0.19, -0.33, +0.09, +1.00, +0.02, +0.04, -0.22, -0.06, +0.31, +0.22, +0.34, -0.04, -0.26, +0.14, +0.19, -0.02, +0.18, -0.05),   # us rmw
    (-0.36, -0.08, +0.68, +0.02, +1.00, -0.02, -0.13, +0.07, +0.37, -0.08, +0.36, -0.03, -0.16, +0.10, +0.17, -0.08, +0.22, -0.03),   # us cma
    (-0.12, -0.06, -0.14, +0.04, -0.02, +1.00, -0.16, +0.03, -0.11, +0.14, +0.03, +0.45, -0.15, +0.01, -0.08, +0.08, +0.05, +0.31),   # us mom
    (+0.57, +0.16, -0.02, -0.22, -0.13, -0.16, +1.00, -0.21, -0.07, -0.25, -0.29, -0.27, +0.76, -0.21, +0.01, -0.14, -0.23, -0.19),   # dev mkt_rf
    (-0.16, +0.20, +0.01, -0.06, +0.07, +0.03, -0.21, +1.00, +0.07, -0.06, +0.01, +0.11, -0.01, +0.26, +0.08, -0.16, +0.13, +0.03),   # dev smb
    (-0.14, -0.02, +0.53, +0.31, +0.37, -0.11, -0.07, +0.07, +1.00, -0.42, +0.63, -0.27, -0.07, +0.06, +0.38, -0.15, +0.23, -0.19),   # dev hml
    (-0.09, -0.05, -0.11, +0.22, -0.08, +0.14, -0.25, -0.06, -0.42, +1.00, -0.33, +0.33, -0.23, +0.10, -0.10, +0.15, +0.01, +0.20),   # dev rmw
    (-0.31, -0.11, +0.35, +0.34, +0.36, +0.03, -0.29, +0.01, +0.63, -0.33, +1.00, -0.05, -0.40, +0.09, +0.18, -0.02, +0.31, -0.03),   # dev cma
    (-0.17, -0.00, -0.20, -0.04, -0.03, +0.45, -0.27, +0.11, -0.27, +0.33, -0.05, +1.00, -0.20, +0.02, -0.11, +0.15, +0.11, +0.59),   # dev mom
    (+0.53, +0.20, -0.07, -0.26, -0.16, -0.15, +0.76, -0.01, -0.07, -0.23, -0.40, -0.20, +1.00, -0.23, +0.06, -0.22, -0.31, -0.13),   # emg mkt_rf
    (-0.20, -0.01, +0.08, +0.14, +0.10, +0.01, -0.21, +0.26, +0.06, +0.10, +0.09, +0.02, -0.23, +1.00, -0.06, -0.07, +0.07, +0.09),   # emg smb
    (+0.02, -0.04, +0.25, +0.19, +0.17, -0.08, +0.01, +0.08, +0.38, -0.10, +0.18, -0.11, +0.06, -0.06, +1.00, -0.36, +0.31, -0.08),   # emg hml
    (-0.02, -0.05, -0.12, -0.02, -0.08, +0.08, -0.14, -0.16, -0.15, +0.15, -0.02, +0.15, -0.22, -0.07, -0.36, +1.00, -0.17, +0.17),   # emg rmw
    (-0.23, -0.03, +0.13, +0.18, +0.22, +0.05, -0.23, +0.13, +0.23, +0.01, +0.31, +0.11, -0.31, +0.07, +0.31, -0.17, +1.00, +0.03),   # emg cma
    (-0.09, -0.03, -0.13, -0.05, -0.03, +0.31, -0.19, +0.03, -0.19, +0.20, -0.03, +0.59, -0.13, +0.09, -0.08, +0.17, +0.03, +1.00),   # emg mom
)

# The same, with MC_PREMIA_LONG_WINDOW's US mkt_rf and hml reaching 1926.
PREMIUM_CORR_LONG = (   # rows and columns in PREMIUM_KEYS order
    (+1.00, +0.18, +0.22, -0.13, -0.24, -0.34, +0.38, -0.11, -0.09, -0.06, -0.21, -0.11, +0.35, -0.14, +0.01, -0.01, -0.15, -0.06),   # us mkt_rf
    (+0.18, +1.00, +0.01, -0.33, -0.08, -0.06, +0.16, +0.20, -0.02, -0.05, -0.11, -0.00, +0.20, -0.01, -0.04, -0.05, -0.03, -0.03),   # us smb
    (+0.22, +0.01, +1.00, +0.06, +0.45, -0.41, -0.01, +0.01, +0.35, -0.07, +0.23, -0.13, -0.05, +0.05, +0.17, -0.08, +0.08, -0.08),   # us hml
    (-0.13, -0.33, +0.06, +1.00, +0.02, +0.04, -0.22, -0.06, +0.31, +0.22, +0.34, -0.04, -0.26, +0.14, +0.19, -0.02, +0.18, -0.05),   # us rmw
    (-0.24, -0.08, +0.45, +0.02, +1.00, -0.02, -0.13, +0.07, +0.37, -0.08, +0.36, -0.03, -0.16, +0.10, +0.17, -0.08, +0.22, -0.03),   # us cma
    (-0.34, -0.06, -0.41, +0.04, -0.02, +1.00, -0.16, +0.03, -0.11, +0.14, +0.03, +0.45, -0.15, +0.01, -0.08, +0.08, +0.05, +0.31),   # us mom
    (+0.38, +0.16, -0.01, -0.22, -0.13, -0.16, +1.00, -0.21, -0.07, -0.25, -0.29, -0.27, +0.76, -0.21, +0.01, -0.14, -0.23, -0.19),   # dev mkt_rf
    (-0.11, +0.20, +0.01, -0.06, +0.07, +0.03, -0.21, +1.00, +0.07, -0.06, +0.01, +0.11, -0.01, +0.26, +0.08, -0.16, +0.13, +0.03),   # dev smb
    (-0.09, -0.02, +0.35, +0.31, +0.37, -0.11, -0.07, +0.07, +1.00, -0.42, +0.63, -0.27, -0.07, +0.06, +0.38, -0.15, +0.23, -0.19),   # dev hml
    (-0.06, -0.05, -0.07, +0.22, -0.08, +0.14, -0.25, -0.06, -0.42, +1.00, -0.33, +0.33, -0.23, +0.10, -0.10, +0.15, +0.01, +0.20),   # dev rmw
    (-0.21, -0.11, +0.23, +0.34, +0.36, +0.03, -0.29, +0.01, +0.63, -0.33, +1.00, -0.05, -0.40, +0.09, +0.18, -0.02, +0.31, -0.03),   # dev cma
    (-0.11, -0.00, -0.13, -0.04, -0.03, +0.45, -0.27, +0.11, -0.27, +0.33, -0.05, +1.00, -0.20, +0.02, -0.11, +0.15, +0.11, +0.59),   # dev mom
    (+0.35, +0.20, -0.05, -0.26, -0.16, -0.15, +0.76, -0.01, -0.07, -0.23, -0.40, -0.20, +1.00, -0.23, +0.06, -0.22, -0.31, -0.13),   # emg mkt_rf
    (-0.14, -0.01, +0.05, +0.14, +0.10, +0.01, -0.21, +0.26, +0.06, +0.10, +0.09, +0.02, -0.23, +1.00, -0.06, -0.07, +0.07, +0.09),   # emg smb
    (+0.01, -0.04, +0.17, +0.19, +0.17, -0.08, +0.01, +0.08, +0.38, -0.10, +0.18, -0.11, +0.06, -0.06, +1.00, -0.36, +0.31, -0.08),   # emg hml
    (-0.01, -0.05, -0.08, -0.02, -0.08, +0.08, -0.14, -0.16, -0.15, +0.15, -0.02, +0.15, -0.22, -0.07, -0.36, +1.00, -0.17, +0.17),   # emg rmw
    (-0.15, -0.03, +0.08, +0.18, +0.22, +0.05, -0.23, +0.13, +0.23, +0.01, +0.31, +0.11, -0.31, +0.07, +0.31, -0.17, +1.00, +0.03),   # emg cma
    (-0.06, -0.03, -0.08, -0.05, -0.03, +0.31, -0.19, +0.03, -0.19, +0.20, -0.03, +0.59, -0.13, +0.09, -0.08, +0.17, +0.03, +1.00),   # emg mom
)

# MC_PREMIA_LONG_WINDOW=1 takes the US market and value premia from Ken French's
# 3-factor file, which reaches 1926-07 instead of 1963-07.
#
# OFF BY DEFAULT, and the measurement is why. The roadmap expected the extra 37
# years to cut the market premium's standard error from 1.95pp to ~1.55pp. It
# does not — measured, 1.84pp, a 6% gain rather than 20% — because pre-1963
# volatility is 22.5%/yr against 15.5% after, so the added history carries more
# variance per observation and eats most of the 1/sqrt(T) gain. What it DOES do
# is move the level: mkt_rf 7.19% -> 8.33%, hml 3.59% -> 4.26%.
#
# That is a bad trade for a reason specific to this model. Every second moment
# it uses comes from a modern window — base volatility from 2010+, correlations
# from the last 15 years, regimes from a 12-month rolling estimate. Pairing a
# premium earned in a 22.5%-volatility era, under a risk-free rate
# administratively pegged at 1.35% against 4.35% after, with modern-volatility
# machinery is internally inconsistent.
#
# SIZE CANNOT BE EXTENDED AT ALL, which the roadmap assumed it could. 3-factor
# SMB sorts on B/M only; 5-factor SMB averages across B/M, OP and INV sorts.
# They correlate 0.98 but differ 4.5pp/yr in mean absolute terms. Loadings here
# are fit against the 5-factor definition, so taking the premium from the
# 3-factor series would pair a beta with a different factor. mkt_rf and hml are
# literally the same series on the overlap (max |diff| 0.0008 and 0.0000), which
# is exactly why those two can move and size cannot.
PREMIA_LONG_WINDOW = _os.environ.get('MC_PREMIA_LONG_WINDOW', '') not in ('', '0')
if PREMIA_LONG_WINDOW:
    PREMIA['us'].update({'mkt_rf': 0.0833, 'hml': 0.0426})
    PREMIUM_SE['us'].update({'mkt_rf': 0.0184, 'hml': 0.0123})
    PREMIUM_WINDOW['us'].update({'mkt_rf': (1201, '1926-07'), 'hml': (1201, '1926-07')})
    PREMIUM_CORR = PREMIUM_CORR_LONG

# Named scenarios. Fixed and named so they are arguable
# in public; the point of the interface is that nobody has to hold an opinion
# about the profitability premium before comparing two portfolios.
#
# Every scenario varies FACTOR premia only. The market premium is a SEPARATE
# axis, and conflating them produces beta conclusions dressed as factor
# conclusions: run raw last-15y premia including the market and 141 vs 102 reads
# -0.56pp/yr, of which market-beta is -0.57pp and factors are +0.02pp.
_STYLE = tuple(f for f in FACTORS if f != 'mkt_rf')

# `_REGIONAL` is a reserved key: {region: source_region} takes that region's
# MARKET premium from another one, and {region: (source_region, gap)} takes it
# less a fixed gap. Multipliers cannot express this — they scale
# all three regions together — and the regional market premium is the biggest
# untested bet in the model. Every style scenario below leaves US at 7.19%
# against developed 4.87%, a 2.32pp head start, so a US-heavy portfolio wins
# every column partly for a reason nothing was testing.
_REGIONAL = '_regional'

# How far each region's MARKET premium trailed the US's over the window the
# international funds' own means cover — DFWIX and AVDE start 2005-10 —
# measured by `python3 tools/premia.py --since 2005-10 --until 2026-07`.
#
# A GAP, because that is what the lag was: the US earned 10.19% over it against
# its full-window 7.19%, while developed earned 6.24% (ABOVE its 4.87%) and
# emerging 7.96% (against 8.13%). Setting each region to its recent level would
# have RAISED developed and left emerging almost where it was. So 'intl keeps
# lagging' keeps the US's long-run premium and holds this gap below it. One
# 21-year draw each, SE 1.90pp and 2.92pp: a belief to test, not a forecast.
RECENT_LAG_WINDOW = ('2005-10', '2026-07')
RECENT_LAG = {'dev': 0.0395, 'emg': 0.0224}

SCENARIOS = {
    'full history': {},                                     # premia as-is
    'all half':     {f: 0.5 for f in _STYLE},
    'no factors':   {f: 0.0 for f in _STYLE},
    'size dead':    {'smb': 0.0},
    'value dead':   {'hml': 0.0},
    'investment dead': {'cma': 0.0},
    'quality dead': {'rmw': 0.0},
    'momentum dead': {'mom': 0.0},
    # The regional axis. Not style bets — bets about whether US exceptionalism
    # persists, which is the single largest one this model makes by default.
    'intl catches up': {_REGIONAL: {'dev': 'us'}},
    'regions converge': {_REGIONAL: {'dev': 'us', 'emg': 'us'}},
    'intl keeps lagging': {_REGIONAL: {r: ('us', gap) for r, gap in RECENT_LAG.items()}},
    'us stays ahead': {},        # an ALIAS for the default, named so the
}                                # status quo reads as a choice, not a given

# Adjusted R2 below this and a ticker gets no factor treatment: it keeps a
# historical mean and a wide band. NOT a tuned threshold — the 25 tickers split
# at 0.577 -> 0.888 with nothing in between, so anything from 0.60 to 0.88
# partitions identically. Below the gap sit all five bond funds and the single
# stock, which is exactly the set that belongs outside the frame.
GATE_ADJ_R2 = 0.80

# A single stock sits below the gate for a different reason from a bond, and
# keeping its historical mean was the wrong answer for it (ledger entry N10, unpublished). COST's
# mean IS its alpha — 14.65% real against VOO's 7.49% — and the ticker is in
# the universe because it outperformed, so the selection the factor method
# discards for every fund flowed straight into its forecast. 20 years at 18.8%
# vol pins that mean to about +-4pp: the data cannot tell a 9pp edge from none.
#
# So a single stock earns its US factor loadings and nothing else — the
# build-up every fund gets, alpha dropped — fitted on the same 240 months its
# volatility is measured on (a company 30 years ago was a different business —
# the reason loaders.SINGLE_STOCKS gets the rolling window at all). US only:
# the two-region fit gave COST -0.35 developed market and -0.59 developed
# investment, collinearity artefacts on a US retailer.
#
# Six factors, not the market alone. The first version was market-only and
# dropped COST's profitability loading (0.56, t 4.1) — a premium the model
# pays every fund that carries it, worth +2.1pp of nominal mu. That put COST
# 2pp below VOO for a reason that applied to it and to nothing else.
#
# Any edge you believe in is STATED, not inherited from a lucky sample: zero by
# default, nominal and annual, added after the premium. It is read from the
# environment because workers re-import under `spawn`.
SINGLE_STOCK_MONTHS = 240
SINGLE_STOCK_EDGE = float(_os.environ.get('MC_SINGLE_STOCK_EDGE', '0.0'))
# How far a single stock's mu may lead the market proxy's beyond what its beta
# and the stated edge allow — the proxy's own style loadings and fee, which the
# market-only build-up does not carry. ~0.3pp today; 1pp is slack, not a target.
SINGLE_STOCK_GATE_TOL = 0.01

# Below this many months, take the single best region rather than a
# multi-region fit — there is not enough data to identify the split. AVEE (32
# months) is the only ticker in the current universe under it.
MIN_MONTHS_MULTI = 120
MIN_MONTHS_ANY = 30

# A region joins a multi-region fit only when ALL THREE of these hold.
#
# Significance ALONE is the wrong test, and this was got wrong first time. The
# regional market factors correlate at 0.777, so on a long sample a trivial
# spurious loading still clears t=2: VTI — Vanguard TOTAL US MARKET, holding no
# international whatsoever — picked up a developed market beta of 0.022 across
# 301 months, and VTV (US large value) took 0.135, moving its expected excess
# return by -0.68pp. A t-stat threshold is sample-size dependent in exactly the
# wrong direction: longer histories admit SMALLER spurious loadings.
#
# The delta cutoff is not finely tuned. Measured deltas in adjusted R2 from
# adding a second region: VTI 0.000, DFWIX+us 0.003, VTV 0.006, VT+emg 0.006 |
# VXUS+emg 0.028, DFWIX+emg 0.029, VT+dev 0.055. Anything in 0.007-0.027
# partitions identically.
#
# KNOWN FALSE NEGATIVE: VT genuinely holds ~10% emerging and its emerging term
# lands at delta 0.006, inside the reject band, so VT is fitted us+dev and its
# expected return is understated by ~0.34pp. Accepted deliberately — the same
# cutoff prevents VTV's -0.68pp error, and both errors point the same way, so
# the net is favourable. Revisit if a genuinely three-region fund matters more.
REGION_T = 2.0
REGION_MIN_DELTA_ADJ_R2 = 0.01
REGION_MIN_MKT_BETA = 0.05


# Published expense ratios, annual. A ticker with no entry here and none stored
# in its fit is NOT SUPPLIED and is treated as zero with a loud warning, which
# OVERSTATES that fund by its fee.
#
# Deliberately not filled in from memory: these are real published numbers and
# guessing them would be inventing financial data. Look them up and set them.
#
# Needed because the build-up is GROSS — dropping alpha adds the fee back,
# so it has to be subtracted explicitly. The historical-mean method never needed
# this, since realised returns are already net.
EXPENSE_OVERRIDE = {
    # A single stock has no expense ratio; yfinance correctly returns none.
    'COST': 0.0,
}

# Implementation shortfall beyond the fee: TURNOVER x ROUND_TRIP, from the
# fund's reported annual holdings turnover.
#
# Not COST_ADJUSTMENTS' `drag`, which is what 6e originally proposed. Turnover
# settles it: the Avantis funds turn over 4-7% a year and cannot be paying the
# 0.20-0.40%/yr `drag` assigns them — that number is 3 to 40 times what their
# trading implies. `drag` was calibrated to close the gap between the Ken French
# BLEND and the real fund, which bundles implementation cost with EXPOSURE
# MISMATCH, and beta now handles exposure mismatch. It stays in
# data_collection_regime_model for proxy construction, where that job is still
# needed; it just is not this.
#
#   tkr    turnover   drag   turnover x round-trip
#   AVUV        6%   0.40%                   0.01%
#   AVLV        7%   0.20%                   0.01%
#   MTUM      116%   0.70%                   0.17%
#   VFMO      100%   1.10%                   0.15%
#
# ROUND_TRIP is the one asserted number left, and a small one: the whole choice
# is worth ~0.1pp/yr on a live portfolio. 0.15% covers spread plus impact for
# the liquid US and developed equities these funds hold; a small-cap sleeve
# would be dearer and a mega-cap one cheaper.
# EXPOSED, not enhanced — and the asset-aware version was built, measured and
# discarded. The obvious refinement is to scale by region and size tilt, since
# emerging small-cap costs far more to trade than US mega-cap. Measured
# 2026-09-11 it is worth +0.020pp/yr on 141, 140 AND 161 — the same figure to
# three decimals, so it cannot move a ranking — and +0.004pp on VT.
#
# The reason is structural and worth keeping: in THIS universe turnover and
# illiquidity are ANTI-correlated at -0.388. What trades most is US large-cap
# momentum (MTUM 116%, VFMO 100%) and short bonds (55-65%), all cheap to trade;
# what is expensive to trade barely trades at all (VSS intl small 27%, AVEE
# emerging small 5%). The two effects cancel by construction.
#
# So a single number is right here, and MC_ROUND_TRIP makes it arguable rather
# than asserted. 0.15% covers spread plus impact for the liquid US and developed
# equities that do the trading; 0.40% is a pessimistic bound and costs MTUM
# +0.29pp and VFMO +0.25pp, the only two funds where the choice is visible.
ROUND_TRIP = float(_os.environ.get('MC_ROUND_TRIP', '0.0015'))

# A fund REPORTING zero turnover is almost certainly a data gap rather than a
# fund that never trades — AVEM reports 0%. Floor it rather than book the fund
# as free.
MIN_TURNOVER = 0.05


def costs_for(ticker, fit=None):
    """(expense, implementation). Expense comes from the ARTIFACT, which
    tools/factor_loadings.py sources from yfinance's fund_operations, falling
    back to EXPENSE_OVERRIDE and then to None so the caller can warn rather than
    silently pretending a fund is free.

    Sourced rather than typed in: these are real published numbers and the
    repo's hand-maintained ones had already gone stale — JQUA 0.24% against an
    actual 0.12%, VTV 0.05% against 0.03%, VBR 0.07% against 0.05%.
    """
    expense = (fit or {}).get('expense')
    if expense is None:
        expense = EXPENSE_OVERRIDE.get(ticker)
    turnover = (fit or {}).get('turnover')
    if turnover is None:
        implementation = 0.0
    else:
        implementation = max(float(turnover), MIN_TURNOVER) * ROUND_TRIP
    return expense, implementation


def bond_spread(returns, rf):
    """Mean annual excess over the CONTEMPORANEOUS risk-free rate.

    Window-robust where a raw mean is not: living through ZIRP drags the mean
    down and leaves the spread alone. The current model takes raw means, which
    is why it believes cash-like funds earn -1.2% real forever — JPST sits
    2.5pp below the historical risk-free rate purely because its history starts
    in 2007.
    """
    d = np.asarray(returns, dtype=float) - np.asarray(rf, dtype=float)
    return float(d.mean() * 12)


def _ols(y, X):
    """Coefficients, their standard errors, and adjusted R2."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    dof = max(n - k, 1)
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * resid.var(ddof=min(k, n - 1)))
    r2 = 1.0 - resid.var() / y.var() if y.var() > 0 else 0.0
    adj = 1.0 - (1.0 - r2) * (n - 1) / dof
    return beta, se, adj


def _design(frames, regions):
    """Intercept plus the six factors from each region, aligned on month."""
    cols, names = [], []
    for r in regions:
        for f in FACTORS:
            cols.append(frames[r][f].values)
            names.append((r, f))
    return np.column_stack([np.ones(len(cols[0]))] + cols), names


def fit_loadings(excess_returns, frames, regions=REGIONS):
    """Fit one ticker. `excess_returns` is monthly, already net of the rf used.

    `frames` maps region -> DataFrame of aligned factor columns, same index as
    `excess_returns`. Returns None when there is not enough overlap to fit
    anything at all.
    """
    y = np.asarray(excess_returns, dtype=float)
    n = len(y)
    if n < MIN_MONTHS_ANY:
        return None

    single = {}
    for r in regions:
        X, _ = _design(frames, [r])
        b, se, adj = _ols(y, X)
        single[r] = {'adj_r2': adj, 'beta': b, 'se': se}
    best = max(single, key=lambda r: single[r]['adj_r2'])
    chosen = [best]

    # Extend a region at a time, on all three criteria — see REGION_T.
    if n >= MIN_MONTHS_MULTI:
        base_adj = single[best]['adj_r2']
        for r in regions:
            if r == best:
                continue
            trial = chosen + [r]
            X, names = _design(frames, trial)
            b, se, adj = _ols(y, X)
            i = names.index((r, 'mkt_rf')) + 1        # +1 for the intercept
            t_ok = se[i] > 0 and abs(b[i] / se[i]) >= REGION_T
            big_enough = abs(b[i]) >= REGION_MIN_MKT_BETA
            explains = (adj - base_adj) >= REGION_MIN_DELTA_ADJ_R2
            if t_ok and big_enough and explains:
                chosen, base_adj = trial, adj

    X, names = _design(frames, chosen)
    b, se, adj = _ols(y, X)
    beta, beta_se = {}, {}
    for (r, f), v, s in zip(names, b[1:], se[1:]):
        beta.setdefault(r, {})[f] = float(v)
        beta_se.setdefault(r, {})[f] = float(s)
    return {
        'method': 'factor' if adj >= GATE_ADJ_R2 else 'below_gate',
        'regions': chosen,
        'beta': beta,
        'beta_se': beta_se,
        'alpha': float(b[0] * 12),
        'alpha_se': float(se[0] * 12),
        'adj_r2': float(adj),
        'n_months': int(n),
        'multi_region': len(chosen) > 1,
    }


def fit_single_stock(returns, frame, months=SINGLE_STOCK_MONTHS):
    """A single stock's recipe: its six US factor loadings over its last `months`.

    `returns` is monthly and RAW (not excess); `frame` is the US factor frame
    with the six FACTORS and `rf`. Alpha is fitted only so its size can be
    reported beside the forecast that drops it.
    """
    import pandas as pd
    j = pd.concat([returns.rename('y'), frame[list(FACTORS) + ['rf']]],
                  axis=1).dropna().iloc[-months:]
    if len(j) < MIN_MONTHS_ANY:
        return None
    fit = fit_loadings(j['y'] - j['rf'], {'us': j}, regions=['us'])
    return dict(fit, method='single_stock')


def premia_for(scenario='full history', haircut=1.0):
    """The premium vector under a named scenario, haircut applied after."""
    if scenario not in SCENARIOS:
        raise KeyError(f"unknown scenario {scenario!r}; "
                       f"have {', '.join(sorted(SCENARIOS))}")
    mult = SCENARIOS[scenario]
    out = {r: {f: PREMIA[r][f] * mult.get(f, 1.0) * haircut for f in FACTORS}
           for r in PREMIA}
    # Regional overrides last, and on mkt_rf only: a style multiplier is a claim
    # about a factor, this is a claim about a MARKET. Applied after the haircut
    # so the source region's already-haircut premium is what gets copied — and
    # a gap is haircut with it, being a difference of two premia.
    for target, source in (mult.get(_REGIONAL) or {}).items():
        source, gap = (source, 0.0) if isinstance(source, str) else source
        if target in out and source in out:
            out[target] = dict(out[target], mkt_rf=out[source]['mkt_rf'] - gap * haircut)
    return out


def expected_return(fit, costs=(0.0, 0.0), scenario='full history',
                    haircut=1.0, rf=None):
    """Nominal expected return from a fit. GROSS of costs until they are
    subtracted here — see the module docstring, and 6e.

    `costs` is (expense, implementation). Both are subtracted: dropping alpha
    added both back, so both have to come out.
    """
    if fit is None or fit['method'] != 'factor':
        return None
    # Resolved HERE, not as a default argument: `rf=RF_FORWARD` in the signature
    # freezes the value at import, so MC_RF_FORWARD never reached it and a test
    # that patched the module global silently measured nothing.
    rf = RF_FORWARD if rf is None else rf
    p = premia_for(scenario, haircut)
    mu = rf
    for r, loads in fit['beta'].items():
        for f, b in loads.items():
            mu += b * p[r][f]
    expense, implementation = costs
    return mu - abs(expense) - abs(implementation)


# ---------------------------------------------------------------------------
# The artifact, and assembling mu from it
# ---------------------------------------------------------------------------

# config/, not output/: output/ is gitignored as regenerated data, and these
# are a slowly-changing INPUT — betas are stable while prices are refreshed
# quarterly. With the factor method now the default, an untracked artifact would
# mean a fresh clone silently falling back to historical means.
FITS_FILE = _Path(__file__).resolve().parent.parent / 'config' / 'factor_loadings.json'


def load_fits(path=None):
    """The per-ticker recipes written by tools/factor_loadings.py.

    Kept OUT of the parameter JSON on purpose: betas are stable while prices are
    refreshed quarterly, so coupling them would force a full parameter regen —
    moving every number — just to adopt a factor build-up.
    """
    path = _Path(path) if path else FITS_FILE
    try:
        return _json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def single_stock_return(fit, scenario='full history', haircut=1.0, rf=None,
                        edge=None):
    """rf + sum(loading x premium) + the stated edge. Never the alpha."""
    rf = RF_FORWARD if rf is None else rf
    edge = SINGLE_STOCK_EDGE if edge is None else edge
    return rf + _loaded_premium(fit, premia_for(scenario, haircut)) + edge


def _loaded_premium(fit, p):
    return sum(b * p[r][f] for r, loads in fit['beta'].items()
               for f, b in loads.items())


def build_nominal(historical, fits, scenario='full history', haircut=1.0,
                  rf=None, edge=None):
    """Nominal expected returns per ticker, from the recipes.

    Falls back to the historical mean for anything without a usable recipe, so a
    missing or partial artifact degrades rather than crashing. Returns
    (returns, notes) where notes lists what fell back and why — the caller
    decides how loudly to say so.
    """
    rf = RF_FORWARD if rf is None else rf
    out, notes = dict(historical), []
    if not fits:
        return out, ['no factor_loadings.json — every ticker kept its historical mean']
    missing_expense = []
    for t, hist in historical.items():
        fit = fits.get(t)
        if not fit:
            notes.append(f'{t}: no recipe, kept historical')
            continue
        method = fit.get('method')
        if method == 'factor':
            expense, impl = costs_for(t, fit)
            if expense is None:
                missing_expense.append(t)
                expense = 0.0
            mu = expected_return(fit, costs=(expense, impl), scenario=scenario,
                                 haircut=haircut, rf=rf)
            if mu is not None:
                out[t] = mu
        elif method == 'spread':
            out[t] = rf + float(fit['spread'])
        elif method == 'single_stock':
            out[t] = single_stock_return(fit, scenario=scenario, haircut=haircut,
                                         rf=rf, edge=edge)
        else:
            notes.append(f'{t}: {method}, kept historical')
    if missing_expense:
        notes.append('NO EXPENSE RATIO for ' + ', '.join(sorted(missing_expense))
                     + ' — treated as free, which OVERSTATES them')
    return out, notes


def parameter_std(fit, scenario='full history', haircut=1.0):
    """Epistemic band on mu, DERIVED rather than hand-set.

        Var(mu) = SUM_f [ beta_f^2 Var(p_f) + p_f^2 Var(beta_f) ] + Var(alpha)

    Three sources, each of which the old PARAM_STD_OVERRIDE table could only
    guess at: not knowing the premium, not knowing the loading, and not knowing
    what the factors miss. A fund with a large value loading gets a wide band
    automatically, because the value premium is the uncertain input.

    This replaces a table that had gone INVERTED — AVLV carried 0.07 x vol and
    VOO 0.16, so the model was more confident about a 2021-inception fund built
    from a hand-weighted proxy than about the S&P 500. Nobody chose that; it is
    what an override table does over time.

    Returns None for a ticker the factor model does not cover, so the caller
    keeps whatever the historical path gave it.
    """
    # A single stock's band stays the single-stock policy's (build_param_std),
    # but it is derived here too: premium_shares needs it to split that band
    # between the market premium it shares with VOO and its own error.
    if fit is None or fit.get('method') not in ('factor', 'single_stock'):
        return None
    p = premia_for(scenario, haircut)
    var = 0.0
    for region, loads in fit['beta'].items():
        se_f = PREMIUM_SE.get(region, {})
        beta_se = (fit.get('beta_se') or {}).get(region, {})
        for f, b in loads.items():
            var += (b ** 2) * (se_f.get(f, 0.0) ** 2)
            var += (p[region][f] ** 2) * (beta_se.get(f, 0.0) ** 2)
    var += float(fit.get('alpha_se') or 0.0) ** 2
    return var ** 0.5


def premium_shares(fit, scenario='full history', haircut=1.0):
    """Each premium's share of this fund's band, as weights on PREMIUM_KEYS.

        share_k = beta_k * SE_k / parameter_std

    so sum(share**2) is the fraction of the band's VARIANCE that comes from not
    knowing the premia — the term parameter_std sums first — and the rest
    (loading and alpha uncertainty) is the fund's own. FRACTIONS, applied to
    the fund's final band at draw time, so a band that is scaled, overridden or
    zeroed later keeps exactly its size. None where the factor model does not
    cover the fund: its whole band is its own.
    """
    sd = parameter_std(fit, scenario=scenario, haircut=haircut)
    if not sd:
        return None
    return [fit['beta'].get(region, {}).get(f, 0.0) * PREMIUM_SE[region][f] / sd
            for region, f in PREMIUM_KEYS]


def build_param_std(historical, fits, scenario='full history', haircut=1.0):
    """Epistemic bands per ticker, derived where the factor model covers a
    ticker and left alone where it does not.

    Returns (bands, notes). A ticker on the historical or spread path keeps its
    existing band: those are not factor forecasts and the derivation does not
    apply to them.
    """
    out, notes = dict(historical), []
    if not fits:
        return out, ['no factor_loadings.json — every band left as-is']
    for t in list(historical):
        if (fits.get(t) or {}).get('method') == 'single_stock':
            continue           # keeps the 0.25 x vol single-stock band
        sd = parameter_std(fits.get(t), scenario=scenario, haircut=haircut)
        if sd is not None:
            out[t] = sd
    return out, notes


def single_stock_problems(nominal, fits, market, stocks=None, edge=None,
                          scenario='full history', haircut=1.0):
    """Single stocks whose mu leads `market` by more than loadings and edge allow.

    The gate for N10: a single stock carrying its historical mean leads VOO by
    ~7pp real, which only a stated edge may do. What its loadings earn beyond
    the market premium itself is allowed — profitability is priced. A stock
    with no single-stock recipe has fallen back to that mean, so it fails too.
    """
    if stocks is None:
        from montecarlo.loaders import SINGLE_STOCKS as stocks
    edge = SINGLE_STOCK_EDGE if edge is None else edge
    p = premia_for(scenario, haircut)
    out = []
    for t in sorted(stocks):
        fit = (fits or {}).get(t) or {}
        if fit.get('method') != 'single_stock':
            out.append(f"{t}: no single-stock recipe ({fit.get('method')}) — "
                       f"it keeps its historical mean")
            continue
        loaded = _loaded_premium(fit, p) - p['us']['mkt_rf']
        allowed = edge + max(loaded, 0.0) + SINGLE_STOCK_GATE_TOL
        lead = nominal[t] - nominal[market]
        if lead > allowed:
            out.append(f"{t} leads {market} by {lead:+.2%}; its loadings and "
                       f"edge {edge:.2%} allow {allowed:+.2%}")
    return out

