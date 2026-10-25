"""Exact reference for the Tier S QUBO and constrained clustering.

Run through: python main.py classical"""

import argparse
from itertools import product
from math import isclose

import numpy as np

from duquantum import config
from duquantum.tier_s import config as tier_config
from duquantum.tier_s.model import CAPACITY, N_BITS, build_q_matrix, build_qubo, evaluate_assignment


def solve_classically(penalty_weight=tier_config.LAMBDA):
    """Return every assignment, ordered by QUBO energy (including c).

    Each record contains bits, energy, original cost, vehicle loads, and
    original capacity feasibility. All 2**N_BITS assignments are evaluated.
    """
    c, h, J = build_qubo(penalty_weight)
    results = []
    for bits in product((0, 1), repeat=N_BITS):
        z = np.asarray(bits, dtype=int)
        cost, loads, feasible = evaluate_assignment(z)
        results.append(
            {
                "bits": bits,
                "energy": float(c + h @ z + z @ J @ z),
                "cost": cost,
                "loads": loads,
                "feasible": feasible,
            }
        )
    return sorted(results, key=lambda result: result["energy"])


def tied(a, b):
    """Numerical tie rule for reported optima."""
    return isclose(a, b, rel_tol=config.COST_REL_TOL, abs_tol=config.COST_ABS_TOL)


def show_assignment(result):
    bits = result["bits"]
    print(f"Bits: {list(bits)}; F = {result['energy']:.6f}; C = {result['cost']:.6f}")
    print(f"Capacity feasible: {result['feasible']}")
    for vehicle, bit in ((1, 1), (2, 0)):
        locations = [i for i, value in enumerate(bits) if value == bit]
        load = result["loads"][vehicle - 1]
        print(f"Vehicle {vehicle}: locations {locations}, load {load:g}/{CAPACITY:g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lambda",
        dest="penalty_weight",
        type=float,
        default=tier_config.LAMBDA,
        help="positive load-balancing weight (default: %(default)s)",
    )
    parser.add_argument(
        "--show-all",
        action=argparse.BooleanOptionalAction,
        default=config.SHOW_ALL,
        help="print every assignment",
    )
    args = parser.parse_args()
    try:
        c, h, J = build_qubo(args.penalty_weight)
    except ValueError as error:
        parser.error(str(error))

    np.set_printoptions(
        precision=config.PRINT_PRECISION, suppress=config.PRINT_SUPPRESS
    )
    print(f"Variables: {N_BITS}; lambda: {args.penalty_weight:g}")
    print(f"Constant offset c: {c:.6f}")
    print("Linear coefficients h:")
    print(h)
    print("Pair coefficients J (strictly upper triangular):")
    print(J)
    print("Symmetric QUBO matrix M:")
    print(build_q_matrix(args.penalty_weight))

    results = solve_classically(args.penalty_weight)
    ground = [r for r in results if tied(r["energy"], results[0]["energy"])]
    feasible = [r for r in results if r["feasible"]]
    print(
        f"\nFeasible assignments: {len(feasible)}/{len(results)} (exhaustive enumeration)"
    )
    print(f"QUBO minimum F: {results[0]['energy']:.6f}; ground states: {len(ground)}")
    print(f"Feasible ground states: {sum(r['feasible'] for r in ground)}/{len(ground)}")
    for result in ground:
        show_assignment(result)

    if feasible:
        optimum = min(r["cost"] for r in feasible)
        original_optima = [r for r in feasible if tied(r["cost"], optimum)]
        preserved = sum(r["feasible"] and tied(r["cost"], optimum) for r in ground)
        print(f"\nOriginal constrained minimum C: {optimum:.6f}")
        print(f"Originally optimal QUBO ground states: {preserved}/{len(ground)}")
        print("Best original solution (representative; vehicle labels may be swapped):")
        show_assignment(original_optima[0])
        print("Lowest-QUBO-energy feasible solution (representative):")
        show_assignment(feasible[0])
    else:
        print("\nNo assignment satisfies the original capacity constraints.")

    if args.show_all:
        print("\nAll assignments, sorted by F:")
        print("Bits | F | C | Loads (V1, V2) | Feasible")
        for r in results:
            print(
                f"{list(r['bits'])} | {r['energy']:.6f} | {r['cost']:.6f} | "
                f"{r['loads']} | {r['feasible']}"
            )


if __name__ == "__main__":
    main()
