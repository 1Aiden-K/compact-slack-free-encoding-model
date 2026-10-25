"""One-hot comparison model for Tier S, scored like Aiden's 1-bit baseline.

One atom per (customer, vehicle): bit a = i*K + k is 1 when customer i rides
in vehicle k. Same objective, lambda, instances, pulse settings, embedding and
emulator as the baseline benchmark, so the only change is the encoding.

    F(x) = sum_i A_i (1 - sum_k x_ik)^2            one-vehicle penalty
         + lambda * sum_k (sum_i d_i x_ik)^2         load balancing
         + sum_k sum_{i<j} D_ij x_ik x_jk            clustering cost

On valid one-hot strings this equals Aiden's F(z) up to a constant, because
2*lambda*(d@z - T/2)^2 = lambda*(L1^2 + L2^2) - lambda*T^2/2.
Choosing A_i = A + lambda*d_i^2 makes every linear coefficient exactly -A, so
one global detuning reproduces the target with no averaging.

Run from src/:
    uv run python onehot_tier_s.py                  # all 20 instances
    uv run python onehot_tier_s.py --limit 1        # first instance only
"""

import argparse
import csv
from itertools import product
from math import isclose
from pathlib import Path

import numpy as np

K = 2
BASELINE = {  # Matches results/tier_s/baseline (unoptimized pulse).
    "penalty_weight": 0.1,
    "duration_ns": 4000,
    "dt_ns": 1,
    "shots": 1000,
    "embedding_seed": 0,
    "sample_seed": 0,
    "restarts": 8,
    "maxiter": 1500,
}


def build_onehot_q_matrix(coordinates, demands, lam, margin=1.1):
    """Return (M, A) with F(x) = c + x @ M @ x for the one-hot encoding."""
    d = np.asarray(demands, dtype=float)
    r = np.asarray(coordinates, dtype=float)
    D = np.linalg.norm(r[:, None] - r[None, :], axis=2)
    n = len(d)
    # Assigning customer i must always lower F, even if every other customer
    # already shares its vehicle: A > max_i sum_j (2*lam*d_i*d_j + D_ij).
    worst = max(
        sum(2 * lam * d[i] * d[j] + D[i, j] for j in range(n) if j != i)
        for i in range(n)
    )
    A = margin * worst
    A_i = A + lam * d**2

    size = n * K
    h = np.full(size, -A)  # -A_i + lam*d_i^2 = -A for every atom
    J = np.zeros((size, size))  # full pair coefficient, upper triangle
    for i in range(n):
        for k in range(K):
            a = i * K + k
            for l in range(k + 1, K):  # same customer, two vehicles
                J[a, i * K + l] += 2 * A_i[i]
            for j in range(i + 1, n):  # same vehicle, two customers
                J[a, j * K + k] += 2 * lam * d[i] * d[j] + D[i, j]
    return np.diag(h) + (J + J.T) / 2, A


def decode(bits, n):
    """Return Aiden's 1-bit label ('1' = vehicle 0) or None if not one-hot."""
    x = np.array([int(b) for b in bits]).reshape(n, K)
    if not np.all(x.sum(axis=1) == 1):
        return None
    return "".join("1" if row[0] else "0" for row in x)


def ground_state_ok(matrix, reference, n):
    """True when every minimizer of matrix decodes to an optimal clustering."""
    strings = np.array(list(product((0, 1), repeat=n * K)))
    energies = np.einsum("bi,ij,bj->b", strings, matrix, strings)
    tol = 1e-10 * max(np.abs(matrix).sum(), 1e-300)
    minima = strings[np.abs(energies - energies.min()) <= tol]
    optimal = set(reference["optimal_bitstrings"])
    return all(decode("".join(map(str, s)), n) in optimal for s in minima)


def score(counts, reference, cvrp):
    """Same definitions as evaluation._rates and benchmark._score_cvrp_samples."""
    records = reference["assignments"]
    optimum, cvrp_opt = reference["optimal_cost"], cvrp["optimal_cost"]
    optimal = set(reference["optimal_bitstrings"])
    n = reference["n_bits"]
    total = sum(counts.values())
    valid = feasible = hits = 0
    best_cost = best_route = None
    for bits, count in counts.items():
        label = decode(bits, n)
        if label is None:
            continue
        valid += count
        if not records[label]["feasible"]:
            continue
        feasible += count
        hits += count * (label in optimal)
        cost, route = records[label]["cost"], cvrp["routed_costs"][label]
        best_cost = cost if best_cost is None else min(best_cost, cost)
        best_route = route if best_route is None else min(best_route, route)

    def gap(value, ref):
        if value is None:
            return None
        return 0.0 if isclose(value, ref, rel_tol=1e-10, abs_tol=1e-9) else (
            100 * (value - ref) / ref
        )

    return {
        "one_hot_valid_rate": valid / total,
        "feasibility_rate": feasible / total,
        "optimal_probability": hits / total,
        "best_model_gap_pct": gap(best_cost, optimum),
        "best_cvrp_gap_pct": gap(best_route, cvrp_opt),
    }


def assess(path):
    from duquantum.quantum.emulator import emulate
    from duquantum.quantum.pulses import build_sequence
    from duquantum.tier_s.benchmark import _bind_instance, _build_cvrp_reference
    from duquantum.tier_s.evaluation import build_reference
    from duquantum.tier_s.instances import load_instance

    s = BASELINE
    instance = load_instance(path)
    _bind_instance(instance)
    reference = build_reference(s["penalty_weight"])  # identical to Aiden's
    cvrp = _build_cvrp_reference(instance, reference)
    n = reference["n_bits"]

    M, A = build_onehot_q_matrix(
        instance["coordinates"], instance["demands"], s["penalty_weight"]
    )
    assert np.allclose(np.diag(M), -A), "linear terms must be uniform"
    assert (M >= 0)[~np.eye(len(M), dtype=bool)].all(), "pairs must be >= 0"

    program = build_sequence(
        M,
        duration_ns=s["duration_ns"],
        seed=s["embedding_seed"],
        restarts=s["restarts"],
        maxiter=s["maxiter"],
    )
    output = emulate(
        program["sequence"], shots=s["shots"], seed=s["sample_seed"], dt=s["dt_ns"]
    )
    realized = np.asarray(program["diagnostics"]["realized_q_matrix"])
    params = program["parameters"]
    return {
        "instance_id": Path(path).stem,
        "instance_seed": instance["seed"],
        "method": "emu-sv-onehot",
        "atoms": n * K,
        **score(output["counts"], reference, cvrp),
        "target_all_ground_states_original_optimal": ground_state_ok(
            M, reference, n
        ),
        "embedded_all_ground_states_original_optimal": ground_state_ok(
            realized, reference, n
        ),
        "detuning_exact": bool(params["exact_global_diagonal"]),
        "relative_interaction_error": program["diagnostics"][
            "relative_interaction_error"
        ],
        "relative_matrix_error": program["diagnostics"]["relative_matrix_error"],
        "penalty_A": A,
        **s,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-dir", default="instances/tier_s")
    parser.add_argument("--output", default="results/tier_s/onehot_baseline/results.csv")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    paths = sorted(Path(args.instances_dir).glob("*.json"))[: args.limit]
    rows = []
    for path in paths:
        row = assess(path)
        rows.append(row)
        print(
            f"{row['instance_id']}: optimal {row['optimal_probability']:.3f}, "
            f"feasible {row['feasibility_rate']:.3f}, "
            f"one-hot valid {row['one_hot_valid_rate']:.3f}, "
            f"ground state kept {row['embedded_all_ground_states_original_optimal']}, "
            f"embedding error {row['relative_interaction_error']:.3f}",
            flush=True,
        )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
