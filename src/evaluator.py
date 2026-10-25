"""Evaluate two-vehicle clustering samples against the original problem.

Requires NumPy and the existing classical.py / qubo.py / config.py, but no
emulator. Exact references use config.MAX_EXACT_BITS as their limit. They are validation
baselines; enumeration does not select or repair the reported quantum shots.

    uv run evaluator.py run.json
    uv run evaluator.py counts.json --lambda 0.1 --output evaluation.json

Input is either a JSON bitstring-to-count mapping or an emulate.py --output
file. epsilon_pct / --epsilon is a relative model-gap threshold in percent.
Set config.DEPOT or pass --depot X Y to include exact classical routing and
CVRP gaps. The selected depot is fixed, independent of clustering.
Only feasible shots can reach the reference. Vehicle label swaps are the same
clustering, but every shot still contributes to the probability denominator.
"""

import argparse
import json
from collections.abc import Mapping
from itertools import product
from math import isclose
from numbers import Integral
from pathlib import Path

import numpy as np

import config


def _tied(a, b):
    return isclose(
        float(a), float(b), rel_tol=config.COST_REL_TOL, abs_tol=config.COST_ABS_TOL
    )


def _canonical(bits):
    """Canonical representative under swapping the two vehicle labels."""
    return min(bits, bits.translate(str.maketrans("01", "10")))


def _gap(cost, optimum):
    if optimum is None:
        return None
    if _tied(cost, optimum):
        return 0.0
    # A relative gap against zero is undefined unless the costs are tied.
    return None if optimum == 0 else 100.0 * (cost - optimum) / optimum


def configured_depot(depot=None):
    """Resolve an explicit depot or config.DEPOT; never invent a depot location."""
    from qubo import COORDINATES
    from routing import validate_depot

    selected = config.DEPOT if depot is None else depot
    return None if selected is None else validate_depot(COORDINATES, selected).tolist()


def build_reference(
    penalty_weight=config.LAMBDA, *, max_bits=config.MAX_EXACT_BITS, depot=None
):
    """Enumerate the constrained problem and target QUBO once per instance.

    All settings come from the existing config.py through qubo.py. References
    can be reused across pulse/embedding runs with the same instance and lambda.
    An infeasible instance is recorded explicitly; gaps then remain undefined.
    A depot supplied explicitly or as config.DEPOT also enables exact TSP/CVRP
    evaluation. The private router cache is not JSON-serializable; reports are.
    """
    from classical import solve_classically
    from qubo import CAPACITY, COORDINATES, DEMANDS, N_BITS, build_q_matrix, build_qubo

    if isinstance(max_bits, bool) or not isinstance(max_bits, Integral) or max_bits < 1:
        raise ValueError("max_bits must be a positive integer.")
    if N_BITS > max_bits:
        raise ValueError(
            f"Exact evaluation requires 2**{N_BITS} assignments; limit is {max_bits} bits. "
            "Use --skip-evaluation to run the emulator without this reference."
        )
    records = {}
    for result in solve_classically(penalty_weight):
        bits = "".join(map(str, result["bits"]))
        records[bits] = {
            "cost": float(result["cost"]),
            "loads": [float(load) for load in result["loads"]],
            "feasible": bool(result["feasible"]),
            "qubo_energy": float(result["energy"]),
        }
    feasible = [r for r in records.values() if r["feasible"]]
    optimum = min((r["cost"] for r in feasible), default=None)
    optimal_bits = sorted(
        bits
        for bits, r in records.items()
        if r["feasible"] and _tied(r["cost"], optimum)
    )
    reference = {
        "n_bits": int(N_BITS),
        "penalty_weight": float(penalty_weight),
        "coordinates": np.asarray(COORDINATES).tolist(),
        "demands": np.asarray(DEMANDS).tolist(),
        "capacity": float(CAPACITY),
        "assignments": records,
        "optimal_cost": optimum,
        "optimal_bitstrings": optimal_bits,
        "optimal_clusterings": sorted({_canonical(bits) for bits in optimal_bits}),
        "status": "proven_optimum" if feasible else "infeasible_instance",
        "q_matrix": build_q_matrix(penalty_weight).tolist(),
        "qubo_offset": float(build_qubo(penalty_weight)[0]),
        "depot": configured_depot(depot),
        "routing": None,
    }
    if reference["depot"] is not None:
        from routing import ExactRouter

        router = ExactRouter(
            COORDINATES,
            reference["depot"],
            max_customers=min(max_bits, config.MAX_ROUTING_CUSTOMERS),
        )
        reference["_router"] = router
        reference["routing"] = router.solve_cvrp(DEMANDS, CAPACITY)
        for bits, record in records.items():
            record["route_cost"] = (
                router.assignment_cost(bits) if record["feasible"] else None
            )
    return reference


def encoding_from_program(program):
    """Extract JSON-safe encoding diagnostics, without importing pulse code."""
    return {
        "n_atoms": int(program["parameters"]["n_qubits"]),
        "energy_scale": float(program["parameters"]["energy_scale"]),
        "global_q_matrix": np.asarray(
            program["parameters"]["global_q_matrix"]
        ).tolist(),
        "realized_q_matrix": np.asarray(
            program["diagnostics"]["realized_q_matrix"]
        ).tolist(),
        "relative_interaction_error": float(
            program["diagnostics"]["relative_interaction_error"]
        ),
        "relative_matrix_error": float(program["diagnostics"]["relative_matrix_error"]),
    }


def _validate_weights(weights, n_bits, *, counts):
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("Counts/probabilities must be a nonempty bitstring mapping.")
    validated = {}
    for bits, weight in weights.items():
        if not isinstance(bits, str) or len(bits) != n_bits or set(bits) - {"0", "1"}:
            raise ValueError(f"Expected {n_bits}-bit strings; received {bits!r}.")
        if isinstance(weight, (bool, np.bool_)):
            raise ValueError("Boolean counts/probabilities are not valid weights.")
        if counts:
            if not isinstance(weight, Integral) or weight < 0:
                raise ValueError("Counts must be nonnegative integers.")
            validated[bits] = int(weight)
        else:
            if not isinstance(weight, (int, float, np.integer, np.floating)):
                raise ValueError("Probabilities must be finite nonnegative numbers.")
            value = float(weight)
            if not np.isfinite(value) or value < 0:
                raise ValueError("Probabilities must be finite nonnegative numbers.")
            validated[bits] = value
    total = sum(validated.values())
    if total <= 0:
        raise ValueError("Counts/probabilities must have positive total weight.")
    if not counts:
        if not np.isfinite(total) or not isclose(
            total, 1.0, rel_tol=0, abs_tol=config.PROBABILITY_ABS_TOL
        ):
            raise ValueError(
                "Probabilities must sum to one; incomplete distributions are not accepted."
            )
        validated = {bits: value / total for bits, value in validated.items()}
    return validated


def _rates(weights, reference, epsilon_pct, *, routing=False):
    records = reference["assignments"]
    objective = reference["routing"] if routing else reference
    cost_key = "route_cost" if routing else "cost"
    optimal_bits = set(objective["optimal_bitstrings"])
    optimum = objective["optimal_cost"]
    total = sum(weights.values())
    feasible_weight = sum(
        weight for bits, weight in weights.items() if records[bits]["feasible"]
    )
    optimal_weight = sum(
        weight for bits, weight in weights.items() if bits in optimal_bits
    )
    reached_weight = 0
    for bits, weight in weights.items():
        record = records[bits]
        if record["feasible"]:
            gap = _gap(record[cost_key], optimum)
            if gap is not None and gap <= epsilon_pct:
                reached_weight += weight
    return {
        "feasibility_rate": feasible_weight / total,
        "optimal_probability": optimal_weight / total if optimum is not None else None,
        "reference_probability": reached_weight / total
        if optimum is not None
        else None,
        "mean_feasible_cost": (
            sum(
                records[bits][cost_key] * weight
                for bits, weight in weights.items()
                if records[bits]["feasible"]
            )
            / feasible_weight
            if feasible_weight
            else None
        ),
    }


def _routing_report(counts, predicted, reference, epsilon_pct, best_model_solutions):
    """Keep selection by clustering cost separate from selection by tour length."""
    records = reference["assignments"]
    routing = reference["routing"]
    optimum = routing["optimal_cost"]

    def solution(bits):
        record = records[bits]
        return {
            "bits": bits,
            "clustering_cost": record["cost"],
            "route_cost": record["route_cost"],
            "loads": record["loads"],
            "cvrp_gap_pct": _gap(record["route_cost"], optimum),
            "routes": reference["_router"].route_assignment(bits),
        }

    feasible = [
        bits
        for bits, count in counts.items()
        if count > 0 and records[bits]["feasible"]
    ]
    best_cost = min((records[bits]["route_cost"] for bits in feasible), default=None)
    representatives = {}
    for bits in sorted(feasible):
        if _tied(records[bits]["route_cost"], best_cost):
            representatives.setdefault(_canonical(bits), bits)
    sampled = _rates(counts, reference, epsilon_pct, routing=True)
    optimal_bits = set(routing["optimal_bitstrings"])
    sampled.update(
        {
            "best_feasible_cost": best_cost,
            "best_cvrp_gap_pct": _gap(best_cost, optimum)
            if best_cost is not None
            else None,
            "optimal_shots": sum(
                count for bits, count in counts.items() if bits in optimal_bits
            ),
            "best_solutions": [
                solution(representatives[key]) for key in sorted(representatives)
            ],
            # Select the best clustering cost first, then the lexicographically
            # smallest canonical partition. Route length does not break this tie.
            "best_model_shot": (
                solution(
                    min(best_model_solutions, key=lambda r: _canonical(r["bits"]))[
                        "bits"
                    ]
                )
                if best_model_solutions
                else None
            ),
        }
    )
    model_optimum_route_costs = [
        records[bits]["route_cost"] for bits in reference["optimal_clusterings"]
    ]
    route_range = (
        [min(model_optimum_route_costs), max(model_optimum_route_costs)]
        if model_optimum_route_costs
        else None
    )
    return {
        "reference": routing,
        "sampled": sampled,
        "predicted": _rates(predicted, reference, epsilon_pct, routing=True)
        if predicted is not None
        else None,
        "uniform_baseline": _rates(
            {bits: 1 for bits in records}, reference, epsilon_pct, routing=True
        ),
        "model_optimum_route_cost_range": route_range,
        "model_optimum_cvrp_gap_range_pct": (
            [_gap(cost, optimum) for cost in route_range]
            if route_range is not None
            else None
        ),
        "selection": "Best CVRP shot minimizes exact route length among observed feasible shots; "
        "best model shot minimizes clustering cost, breaking ties by canonical partition.",
    }


def _ground_states(matrix, reference, sampled, predicted):
    n = reference["n_bits"]
    matrix = np.asarray(matrix, dtype=float)
    if (
        matrix.shape != (n, n)
        or not np.isfinite(matrix).all()
        or not np.allclose(
            matrix, matrix.T, rtol=config.MATRIX_REL_TOL, atol=config.MATRIX_ABS_TOL
        )
    ):
        raise ValueError(
            "Hamiltonian matrices must be finite, symmetric, and match the instance."
        )
    bits = np.asarray(list(product((0, 1), repeat=n)), dtype=int)
    energies = np.einsum("bi,ij,bj->b", bits, matrix, bits)
    tolerance = config.GROUND_STATE_REL_TOL * max(
        float(np.abs(matrix).sum()), np.finfo(float).tiny
    )
    ground = [
        "".join(map(str, row))
        for row in bits[np.abs(energies - energies.min()) <= tolerance]
    ]
    records = reference["assignments"]
    feasible = [label for label in ground if records[label]["feasible"]]
    optimal_bits = set(reference["optimal_bitstrings"])
    optimal = [label for label in ground if label in optimal_bits]
    best_cost = min((records[label]["cost"] for label in feasible), default=None)
    return {
        "ground_bitstrings": ground,
        "distinct_ground_clusterings": len({_canonical(label) for label in ground}),
        "feasible_ground_bitstrings": len(feasible),
        "original_optimal_ground_bitstrings": len(optimal),
        "has_original_optimum": bool(optimal),
        "all_ground_states_original_optimal": len(optimal) == len(ground),
        "best_feasible_ground_gap_pct": (
            _gap(best_cost, reference["optimal_cost"])
            if best_cost is not None
            else None
        ),
        "sampled_ground_state_probability": sum(
            sampled.get(label, 0) for label in ground
        ),
        "predicted_ground_state_probability": (
            sum(predicted.get(label, 0) for label in ground)
            if predicted is not None
            else None
        ),
    }


def evaluate_samples(
    counts,
    *,
    probabilities=None,
    penalty_weight=config.LAMBDA,
    reference=None,
    encoding=None,
    epsilon_pct=config.EPSILON_PCT,
    depot=None,
):
    """Evaluate raw samples without repairing or discarding infeasible shots.

    Rates use ALL shots as the denominator. Best cost and gaps use only feasible
    shots; no feasible shot means None, not zero. epsilon_pct is in percent.
    Optional probabilities give pre-sampling metrics, not extra observed shots.
    Optional encoding is encoding_from_program(program), enabling separate
    checks of global-diagonal projection and geometric embedding.
    """
    if not np.isfinite(epsilon_pct) or epsilon_pct < 0:
        raise ValueError("epsilon_pct must be finite and nonnegative.")
    if reference is None:
        reference = build_reference(penalty_weight, depot=depot)
    elif depot is not None and configured_depot(depot) != reference["depot"]:
        raise ValueError("Reference depot does not match the requested depot.")
    if float(penalty_weight) != reference["penalty_weight"]:
        raise ValueError("Reference lambda does not match penalty_weight.")
    n = reference["n_bits"]
    counts = _validate_weights(counts, n, counts=True)
    predicted = (
        None
        if probabilities is None
        else _validate_weights(probabilities, n, counts=False)
    )
    shots = sum(counts.values())
    sampled = {bits: count / shots for bits, count in counts.items()}
    records = reference["assignments"]
    feasible = [
        bits
        for bits, count in counts.items()
        if count > 0 and records[bits]["feasible"]
    ]
    best_cost = min((records[bits]["cost"] for bits in feasible), default=None)
    best_bits = sorted(
        bits for bits in feasible if _tied(records[bits]["cost"], best_cost)
    )
    best_by_clustering = {}
    for bits in best_bits:
        # Keep an actually observed labeling as the representative.
        best_by_clustering.setdefault(_canonical(bits), bits)
    best_solutions = [
        {
            "bits": bits,
            **records[bits],
            "clusters": [
                [i for i, bit in enumerate(bits) if bit == value]
                for value in ("1", "0")
            ],
        }
        for bits in best_by_clustering.values()
    ]
    metrics = _rates(counts, reference, epsilon_pct)
    optimal_bits = set(reference["optimal_bitstrings"])
    metrics.update(
        {
            "shots": shots,
            "feasible_shots": sum(
                count for bits, count in counts.items() if records[bits]["feasible"]
            ),
            "optimal_shots": sum(
                count for bits, count in counts.items() if bits in optimal_bits
            ),
            "best_feasible_cost": best_cost,
            "best_model_gap_pct": _gap(best_cost, reference["optimal_cost"])
            if best_cost is not None
            else None,
            "best_solutions": best_solutions,
        }
    )
    stages = {
        "target_qubo": _ground_states(
            reference["q_matrix"], reference, sampled, predicted
        )
    }
    if encoding is not None:
        for name, key in (
            ("global_projection", "global_q_matrix"),
            ("embedded", "realized_q_matrix"),
        ):
            if key in encoding:
                stages[name] = _ground_states(
                    encoding[key], reference, sampled, predicted
                )
    rows = []
    for bits in sorted(
        set(counts) | set(predicted or {}),
        key=lambda label: (-counts.get(label, 0), label),
    ):
        record = records[bits]
        rows.append(
            {
                "bits": bits,
                "count": counts.get(bits, 0),
                **record,
                "gap_pct": _gap(record["cost"], reference["optimal_cost"])
                if record["feasible"]
                else None,
                "probability": predicted.get(bits, 0)
                if predicted is not None
                else None,
            }
        )
        if reference["routing"] is not None:
            rows[-1]["cvrp_gap_pct"] = (
                _gap(record["route_cost"], reference["routing"]["optimal_cost"])
                if record["feasible"]
                else None
            )
    return {
        "reference": {
            key: reference[key]
            for key in (
                "status",
                "n_bits",
                "penalty_weight",
                "coordinates",
                "demands",
                "capacity",
                "optimal_cost",
                "optimal_bitstrings",
                "optimal_clusterings",
                "depot",
            )
        },
        "epsilon_pct": float(epsilon_pct),
        "sampled": metrics,
        "predicted": _rates(predicted, reference, epsilon_pct)
        if predicted is not None
        else None,
        "uniform_baseline": _rates(
            {bits: 1 for bits in records}, reference, epsilon_pct
        ),
        "stages": stages,
        "encoding": encoding,
        "bitstrings": rows,
        "routing": (
            _routing_report(counts, predicted, reference, epsilon_pct, best_solutions)
            if reference["routing"] is not None
            else None
        ),
    }


def print_evaluation(report):
    """Print original-problem metrics and stage-by-stage model diagnostics."""

    def number(value):
        return "undefined" if value is None else f"{value:.6f}"

    def percent(value):
        return "undefined" if value is None else f"{100 * value:.2f}%"

    ref, sampled = report["reference"], report["sampled"]
    print("\nOriginal-problem evaluation (no repair):")
    print(
        f"Reference: {ref['status']}; optimal clustering cost: {number(ref['optimal_cost'])}"
    )
    print(
        f"Distinct optimal clusterings: {len(ref['optimal_clusterings'])} (vehicle labels ignored)"
    )
    print(
        f"Feasible shots: {sampled['feasible_shots']}/{sampled['shots']} "
        f"({percent(sampled['feasibility_rate'])})"
    )
    print(
        f"Optimal shots: {sampled['optimal_shots']}/{sampled['shots']} "
        f"({percent(sampled['optimal_probability'])})"
    )
    print(
        f"Reference probability (gap <= {report['epsilon_pct']:g}%): "
        f"{percent(sampled['reference_probability'])}"
    )
    print(
        f"Best feasible cost: {number(sampled['best_feasible_cost'])}; "
        f"best model gap: {number(sampled['best_model_gap_pct'])}%"
    )
    for solution in sampled["best_solutions"]:
        print(
            f"Best observed bits: {solution['bits']}; loads: {solution['loads']}; "
            f"clusters: {solution['clusters']}"
        )
    for name, values in (
        ("Final-state probabilities", report["predicted"]),
        ("Uniform random baseline", report["uniform_baseline"]),
    ):
        if values is not None:
            print(
                f"{name}: feasible {percent(values['feasibility_rate'])}; "
                f"optimal {percent(values['optimal_probability'])}; "
                f"within epsilon {percent(values['reference_probability'])}"
            )
    print("Ground-state checks against the original constrained clustering optimum:")
    for name, stage in report["stages"].items():
        print(
            f"  {name}: {stage['original_optimal_ground_bitstrings']}/"
            f"{len(stage['ground_bitstrings'])} ground bitstrings originally optimal; "
            f"sampled ground-state probability {percent(stage['sampled_ground_state_probability'])}"
        )
    routing = report["routing"]
    if routing is None:
        print("Exact routing/CVRP gap: disabled; set config.DEPOT or pass --depot X Y.")
        return
    print(f"\nExact classical routing; depot: {routing['reference']['depot']}")
    print(
        f"CVRP reference: {routing['reference']['status']}; "
        f"cost: {number(routing['reference']['optimal_cost'])}; "
        f"distinct optimal partitions: {len(routing['reference']['optimal_clusterings'])}"
    )
    routed = routing["sampled"]
    print(
        f"Best routed feasible shot: {number(routed['best_feasible_cost'])}; "
        f"best CVRP gap: {number(routed['best_cvrp_gap_pct'])}%"
    )
    model_shot = routed["best_model_shot"]
    if model_shot is not None:
        print(
            f"Best clustering shot ({model_shot['bits']}), routed exactly: "
            f"{number(model_shot['route_cost'])}; CVRP gap: {number(model_shot['cvrp_gap_pct'])}%"
        )
    for name, values in (
        ("Sampled", routed),
        ("Final-state probabilities", routing["predicted"]),
        ("Uniform random baseline", routing["uniform_baseline"]),
    ):
        if values is not None:
            print(
                f"{name} CVRP: optimal {percent(values['optimal_probability'])}; "
                f"within epsilon {percent(values['reference_probability'])}"
            )
    for result in routed["best_solutions"]:
        print(f"Best routed bits: {result['bits']}; loads: {result['loads']}")
        for vehicle, route in enumerate(result["routes"], 1):
            print(
                f"  Vehicle {vehicle}: {route['stops']}; distance: {number(route['length'])}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="raw counts JSON or emulate.py --output JSON"
    )
    parser.add_argument(
        "--lambda",
        dest="penalty_weight",
        type=float,
        default=None,
        help="defaults to saved run lambda, or config.LAMBDA for raw counts",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=config.EPSILON_PCT,
        help="reference gap tolerance in percent",
    )
    parser.add_argument(
        "--depot",
        type=float,
        nargs=2,
        default=None,
        help="fixed depot X Y; otherwise use the saved run depot or config.DEPOT",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.EVALUATION_OUTPUT,
        help="write the evaluation as JSON",
    )
    args = parser.parse_args()
    try:
        payload = json.loads(args.input.read_text())
        if not isinstance(payload, dict):
            raise ValueError("Input JSON must be an object.")
        is_run = "counts" in payload
        settings = payload.get("settings", {}) if is_run else {}
        saved_lambda = settings.get("penalty_weight")
        lam = args.penalty_weight if args.penalty_weight is not None else saved_lambda
        lam = config.LAMBDA if lam is None else lam
        if saved_lambda is not None and float(lam) != float(saved_lambda):
            raise ValueError(
                "--lambda does not match the saved run; evaluate using its original lambda."
            )
        saved_instance = payload.get("instance") if is_run else None
        saved_depot = (
            saved_instance.get("depot") if saved_instance is not None else None
        )
        if (
            args.depot is not None
            and saved_depot is not None
            and args.depot != saved_depot
        ):
            raise ValueError("--depot does not match the saved run's fixed depot.")
        depot = args.depot if args.depot is not None else saved_depot
        reference = build_reference(lam, depot=depot)
        # Saved run instance metadata guards against evaluating with a different config.py.
        if saved_instance is not None:
            for key in ("coordinates", "demands", "capacity"):
                if saved_instance[key] != reference[key]:
                    raise ValueError(f"Saved instance {key} differs from config.py.")
        report = evaluate_samples(
            payload["counts"] if is_run else payload,
            probabilities=payload.get("probabilities") if is_run else None,
            penalty_weight=lam,
            reference=reference,
            encoding=payload.get("encoding") if is_run else None,
            epsilon_pct=args.epsilon,
        )
        print_evaluation(report)
        if args.output is not None:
            args.output.write_text(
                json.dumps(report, indent=config.JSON_INDENT, allow_nan=False) + "\n"
            )
    except (ValueError, OSError, ImportError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
