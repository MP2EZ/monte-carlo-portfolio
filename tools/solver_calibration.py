#!/usr/bin/env python3
"""
Is the solved plan level actually a 90% answer?
===============================================

    python3 tools/solver_calibration.py 141 145
    python3 tools/solver_calibration.py 141 --policies guyton_klinger,vpw --sims 1200

`solve_sustainable_spend` searches on one sample and then sets the level on a
VERIFICATION sample. Both are drawn from sim ids 0..n-1, so the search sample is
a strict SUBSET of the verification sample and the reported success rate is
measured on exactly the paths the level was tuned to. Asking that sample whether
the level is right is asking a question it cannot fail.

This re-scores the solved level on a genuinely independent sample (`seed_offset`,
which shifts the seeds) and reports the gap. A solver that is honest about its
own confidence lands near the target; one that fits its verification sample
lands below it, and by how much is the size of the optimism in every plan level
this repo quotes.

Not part of validate_all: it costs a full solve per row plus a verification, and
it measures a statistical property rather than guarding a regression.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                               # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('portfolios', nargs='+')
    ap.add_argument('--policies', default='guyton_klinger,constant_real,vpw')
    ap.add_argument('--sims', type=int, default=1200)
    ap.add_argument('--confidence', type=float, default=0.90)
    ap.add_argument('--floor', type=float, default=0.85)
    ap.add_argument('--convergence', metavar='N,N,N',
                    help='instead of calibrating, solve at each path count and '
                         'report how much the LEVEL itself moves with sample size')
    args = ap.parse_args(argv)

    from montecarlo import glide
    import run as runner
    from montecarlo import solver
    from montecarlo.loaders import load_portfolios_from_csv

    P = load_portfolios_from_csv(verbose=False)
    policies = [p.strip() for p in args.policies.split(',') if p.strip()]

    if args.convergence:
        counts = [int(x) for x in args.convergence.split(',')]
        print(f"\n  LEVEL CONVERGENCE   how much the answer moves with sample size")
        print(f"  {'portfolio':<16}{'policy':<16}" +
              ''.join(f'{n:>12}' for n in counts) + f"{'spread':>10}")
        print('  ' + '-' * (32 + 12 * len(counts) + 10))
        for spec in args.portfolios:
            for pol in policies:
                cfg = runner.build_config({'mode': 'retirement', 'portfolios': [spec],
                                           'spend': 'solve', 'policy': pol})
                plan = glide.build_plan(glide.parse_schedule(spec, cfg['age']), P)
                lv = []
                for n in counts:
                    r = solver.solve_sustainable_spend(
                        plan, cfg, confidence=args.confidence, floor=args.floor,
                        search_sims=max(200, n // 2), verify_sims=n)
                    lv.append(r['spend'])
                spread = (max(lv) - min(lv)) / max(min(lv), 1e-9)
                print(f"  {spec:<16}{pol:<16}" +
                      ''.join(f'{v:>12,.0f}' for v in lv) + f"{spread:>9.1%}")
        solver.shutdown_pool()
        print(f"\n  A level quoted to three significant figures is claiming more")
        print(f"  precision than the spread above supports.")
        return 0

    print(f"\n  SOLVER CALIBRATION   target {args.confidence:.0%}, "
          f"floor {args.floor:.0%}, {args.sims} verification paths")
    print(f"  {'portfolio':<16}{'policy':<16}{'level':>11}{'in-sample':>11}"
          f"{'OUT-of-sample':>15}{'gap':>8}")
    print('  ' + '-' * 78)

    gaps = []
    for spec in args.portfolios:
        for pol in policies:
            cfg = runner.build_config({'mode': 'retirement', 'portfolios': [spec],
                                       'spend': 'solve', 'policy': pol})
            plan = glide.build_plan(glide.parse_schedule(spec, cfg['age']), P)
            got = solver.solve_sustainable_spend(
                plan, cfg, confidence=args.confidence, floor=args.floor,
                search_sims=max(200, args.sims // 2), verify_sims=args.sims)
            # A genuinely independent sample: ids args.sims.. rather than 0..
            oos = solver.evaluate(plan, cfg, got['spend'], n_sims=args.sims,
                                  floor=args.floor, seed_offset=10 * args.sims)
            g = oos['success_rate'] - got['success_rate']
            gaps.append(g)
            print(f"  {spec:<16}{pol:<16}${got['spend']:>10,.0f}"
                  f"{got['success_rate']:>11.1%}{oos['success_rate']:>15.1%}"
                  f"{g:>+8.1%}")
    solver.shutdown_pool()

    g = np.array(gaps)
    # Binomial noise on one success rate at this n; two independent samples, so
    # the gap's own standard error is sqrt(2) times it.
    se = np.sqrt(2 * args.confidence * (1 - args.confidence) / args.sims)
    print(f"\n  mean gap {g.mean():+.2%}   worst {g.min():+.2%}   "
          f"(binomial SE on a single gap ~{se:.2%})")
    if g.mean() < -2 * se:
        print(f"  -> the solver is OPTIMISTIC: levels do not hold out of sample.")
    elif abs(g.mean()) <= 2 * se:
        print(f"  -> calibrated: the out-of-sample gap is inside sampling noise.")
    else:
        print(f"  -> the solver is CONSERVATIVE out of sample.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
