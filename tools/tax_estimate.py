#!/usr/bin/env python3
"""STAGE 0 — bound the tax drag before building a tax model.

Taxes were deferred on the grounds that the taxable sleeve is small. This
measures the drag rather than assuming it, over the SIMULATED balance paths
rather than one projection, because the portfolio spans p10 $3.2M to p90 $22.6M
at 65 and the tax code is progressive — a point estimate at the median cannot
bound a progressive tax.

WHAT IT DOES NOT DO. It does not track per-account balances through the engine
(that is Stage 1). It projects the account SHARES deterministically from
contributions and growth, then applies them to each path's simulated total. The
approximation is that every account earns the same return. They hold different
funds (placement put AVUV in the HSA and AVDV in taxable), so
this is wrong in the third decimal and right in the first — which is what a
bound needs.

    python3 tools/tax_estimate.py --portfolio 166 --sims 1500

REAL DOLLARS, with one correction that matters. The model is in real terms and
tax brackets are CPI-indexed, so brackets and the standard deduction are
constant in real terms. Two thresholds are NOT indexed and erode:
  - Social Security taxability ($32k/$44k MFJ, fixed since 1983/1993)
  - the NIIT threshold ($250k MFJ, fixed since 2013)
Over a 30-year retirement at 3% inflation those lose ~59% of their real value,
which is why the 85%-taxable share of SS is treated as reached, not computed.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from montecarlo import engine                                                   # noqa: E402
from montecarlo import glide                                                    # noqa: E402
from montecarlo import policies                                                 # noqa: E402
from montecarlo import investor                                                 # noqa: E402
import run as runner                                            # noqa: E402
from montecarlo import solver                                                   # noqa: E402
from montecarlo.loaders import load_portfolios_from_csv                    # noqa: E402

# 2026 MFJ, real terms. Brackets and the standard deduction are CPI-indexed, so
# these hold; see the module docstring for the two thresholds that do not.
STD_DEDUCTION = 30_000.0
ORDINARY_BANDS = ((23_850.0, 0.10), (72_900.0, 0.12), (109_750.0, 0.22),
                  (85_500.0, 0.24), (117_300.0, 0.32), (274_800.0, 0.35),
                  (float('inf'), 0.37))
LTCG_BANDS = ((96_700.0, 0.00), (503_000.0, 0.15), (float('inf'), 0.20))
NIIT_RATE, NIIT_THRESHOLD = 0.038, 250_000.0

def rmd_start_age(birth_year):
    """The age RMDs begin, which the law has moved three times since 2019.

    SECURE (2019) moved it from 70 1/2 to 72; SECURE 2.0 (2022) to 73, and to 75
    for anyone born 1960 or later. A 30-year model should expect this number to move again.
    """
    if birth_year >= 1960:
        return 75
    if birth_year >= 1951:
        return 73
    if birth_year == 1950:
        return 72
    return 70


# Uniform Lifetime Table, the divisor applied to the PRIOR year-end balance.
RMD_DIVISOR = {
    73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0, 79: 21.1,
    80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0, 86: 15.2,
    87: 14.4, 88: 13.7, 89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1,
    94: 9.5, 95: 8.9,
}

# What fraction of a taxable-account withdrawal is GAIN rather than basis. A
# 28-year hold at ~7% real is ~85% gain.
TAXABLE_GAIN_FRACTION = 0.72


def _stacked_tax(amount, bands, already):
    """Tax on `amount` stacked ON TOP of `already` of taxable income."""
    tax, floor_, lo = 0.0, already, 0.0
    for width, rate in bands:
        hi = lo + width
        band = max(0.0, min(floor_ + amount, hi) - max(floor_, lo))
        tax += band * rate
        lo = hi
        if lo >= floor_ + amount:
            break
    return tax


def project_shares(record, to_age):
    """Account shares by age, from contributions and growth. Deterministic."""
    # Contributions come from the engine's own rule, one account at a time, so
    # the stop age, growth and measured-at age cannot drift from the simulation.
    sched = investor.contribution_schedule(record)
    bal = {k: float(a['balance']) for k, a in record['accounts'].items()}
    shares = {}
    for y in range(record['age'], to_age + 1):
        total = sum(bal.values())
        shares[y] = ({k: v / total for k, v in bal.items()} if total else {})
        for k, a in sched['accounts'].items():
            bal[k] *= 1.07                       # real; the LEVEL comes from
            bal[k] += engine.calculate_annual_contribution(   # the simulation
                y, dict(sched, accounts={k: a}), engine.INFLATION)
    return shares


def tax_a_path(balances, withdrawals, shares, ss, start_age, tax_of,
               ss_start, rmd_age):
    """One path: returns (total tax, total withdrawn, forced-by-RMD total)."""
    total_tax = forced = 0.0
    pretax = None
    for i, age in enumerate(range(start_age, start_age + len(withdrawals))):
        sh = shares.get(min(age, max(shares)), {})
        bal = balances[i]
        if pretax is None:
            pretax = bal * sum(v for k, v in sh.items() if tax_of[k] == 'pretax')
        draw = float(withdrawals[i])

        # pro-rata draw, then RMD forces more out of the pretax sleeve
        want_pretax = draw * sum(v for k, v in sh.items()
                                 if tax_of[k] == 'pretax')
        rmd = (pretax / RMD_DIVISOR[age]
               if age >= rmd_age and age in RMD_DIVISOR else 0.0)
        from_pretax = max(want_pretax, min(rmd, pretax))
        forced += max(0.0, from_pretax - want_pretax)
        pretax = max(0.0, (pretax - from_pretax) * 1.05)

        taxable_draw = draw * sum(v for k, v in sh.items()
                                  if tax_of[k] == 'taxable')
        gains = taxable_draw * TAXABLE_GAIN_FRACTION

        # SS taxability thresholds are NOT indexed, so in real terms they are
        # long gone: treat the 85% maximum as reached.
        ordinary = from_pretax + 0.85 * (ss if age >= ss_start else 0.0)
        ord_taxable = max(0.0, ordinary - STD_DEDUCTION)
        t = _stacked_tax(ord_taxable, ORDINARY_BANDS, 0.0)
        t += _stacked_tax(gains, LTCG_BANDS, ord_taxable)
        magi = ordinary + gains
        if magi > NIIT_THRESHOLD:
            t += NIIT_RATE * min(gains, magi - NIIT_THRESHOLD)
        total_tax += t
    return total_tax, float(np.sum(withdrawals)), forced


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--portfolio', default='166')
    ap.add_argument('--spend', type=float, default=None,
                    help='plan level; default solves for it')
    ap.add_argument('--sims', type=int, default=800)
    ap.add_argument('--gk-response', default='fast')
    args = ap.parse_args(argv)

    rec = investor.load()
    P = load_portfolios_from_csv(verbose=False)
    tax_of = {k: a['tax'] for k, a in rec['accounts'].items()}
    # SOCIAL SECURITY MUST BE IN THE SIMULATION, not only in the tax maths.
    # Without it the portfolio funds the whole plan, which solved to $169,730
    # against the $206,182 the same portfolio reaches WITH it — and the tax
    # calc then credited a benefit the paths never received.
    #
    # ONE SOURCE: the benefit comes from the same `cfg` the paths were
    # simulated with, through the engine's own policies.ss_income.
    spec = {'mode': 'retirement', 'portfolios': [args.portfolio],
            'spend': args.spend or 200_000.0, 'policy': 'guyton_klinger'}
    cfg = runner.build_config(spec)
    cfg['withdrawal']['gk_response'] = args.gk_response
    ss_cfg = cfg.get('social_security') or {}
    ss = policies.ss_income(999, ss_cfg) if ss_cfg else 0.0
    claim = float(ss_cfg.get('claim_age', 70.0)) if ss_cfg else 999.0
    ss_start = int(claim)
    import datetime
    birth_year = datetime.date.today().year - int(rec['age'])
    rmd_age = rmd_start_age(birth_year)
    plan = glide.build_plan(glide.parse_schedule(args.portfolio, cfg['age']), P)

    print(f"  portfolio {args.portfolio} · {args.sims} paths · "
          f"gk_response={args.gk_response}")
    ss_text = (f"Social Security ${ss:,.0f}/yr from {ss_start} "
               f"({ss_cfg.get('benefit_scale', 1.0):.0%} of scheduled)"
               if ss_cfg else "NO Social Security — none in your details")
    print(f"  {ss_text} · RMDs from {rmd_age} (born ~{birth_year})")
    if args.spend is None:
        r = solver.solve_sustainable_spend(plan, cfg, confidence=0.90,
                                           floor=0.85,
                                           verify_sims=args.sims)
        spend = r['spend']
        print(f"  solved plan level ${spend:,.0f}/yr "
              f"(+/-${r.get('level_se') or 0:,.0f})")
        raw = r['raw']
    else:
        spend = args.spend
        raw = solver.simulate(plan, cfg, spend, n_sims=args.sims, full=True)

    shares = project_shares(rec, cfg['age'] + cfg['time_horizon_years'])
    start = cfg['withdrawal']['start_age']
    spending = np.asarray(raw['spending'], dtype=float)
    v65 = np.asarray(raw['v_at_retirement'], dtype=float)
    rets = np.asarray(raw['returns'], dtype=float)
    m0 = (start - cfg['age']) * 12

    rows = []
    for p in range(spending.shape[0]):
        if not np.isfinite(v65[p]):
            continue
        bal, path = v65[p], []
        for y in range(spending.shape[1]):
            path.append(bal)
            growth = np.prod(1.0 + rets[p, m0 + y * 12:m0 + (y + 1) * 12])
            bal = max(0.0, bal * growth - spending[p, y])
        rows.append(tax_a_path(path, spending[p], shares, ss, start,
                               tax_of, ss_start, rmd_age))

    tax = np.array([r[0] for r in rows])
    spent = np.array([r[1] for r in rows])
    forced = np.array([r[2] for r in rows])
    drag = tax / np.maximum(spent, 1.0)

    print(f"\n  {'':16}{'p10':>12}{'median':>12}{'p90':>12}")
    for label, v in (('lifetime tax', tax), ('lifetime spend', spent),
                     ('forced by RMD', forced)):
        q = np.percentile(v, [10, 50, 90])
        print(f"  {label:<16}" + ''.join(f"{x:>12,.0f}" for x in q))
    q = np.percentile(drag, [10, 50, 90])
    print(f"  {'tax / spend':<16}" + ''.join(f"{x:>11.1%} " for x in q))

    print(f"\n  GATE: the roadmap deferred taxes on an unmeasured ~5% sleeve.")
    print(f"    drag at the median {np.median(drag):.1%}, at p90 "
          f"{np.percentile(drag, 90):.1%}")
    print(f"    RMDs force ${np.median(forced):,.0f} of extra withdrawals at "
          f"the median, ${np.percentile(forced, 90):,.0f} at p90")
    return 0


if __name__ == '__main__':
    sys.exit(main())
