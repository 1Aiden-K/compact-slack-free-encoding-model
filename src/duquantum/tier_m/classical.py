"""Exact constrained clustering and QUBO references for the three-vehicle model."""
import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

from duquantum import config as numerical
from duquantum.tier_m import config
from duquantum.tier_m.model import canonical_partition, build_model


def _label_costs(model, labels):
    loads = np.column_stack([(labels == k) @ model.demands for k in (1, 2, 3)])
    costs = np.zeros(len(labels))
    for i in range(model.n):
        for j in range(i + 1, model.n):
            costs += model.distances[i, j] * (labels[:, i] == labels[:, j])
    return costs, loads, np.all(loads <= model.capacity, axis=1)


def _ground_summary(model, bits, feasible, costs, optimum, energy, method, states_checked):
    valid = ~np.any((bits[:, :model.n] + bits[:, model.n:]) == 2, axis=1)
    optimal = (np.zeros(len(bits), dtype=bool) if optimum is None
               else feasible & np.isclose(costs, optimum, rtol=numerical.COST_REL_TOL,
                                           atol=numerical.COST_ABS_TOL))
    return {
        'minimum_energy': float(energy), 'ground_assignment_count': len(bits),
        'ground_valid_assignments': int(valid.sum()),
        'ground_invalid_assignments': int((~valid).sum()),
        'ground_feasible_assignments': int(feasible.sum()),
        'all_ground_states_original_optimal': bool(np.all(optimal)),
        'representative': model.evaluate_assignment(bits[0]),
        'certification': method, 'binary_states_checked': states_checked,
    }


def _all_binary_ground_states(model, optimum):
    if model.n_bits > config.MAX_EXACT_BINARY_BITS:
        raise ValueError(f"Full binary checking is limited to {config.MAX_EXACT_BINARY_BITS} bits.")
    matrix = model.build_q_matrix()
    tolerance = numerical.GROUND_STATE_REL_TOL * max(float(np.abs(matrix).sum()), 1.)
    shifts = np.arange(model.n_bits - 1, -1, -1, dtype=np.uint64)
    best = float('inf')
    ground_ids, ground_energies = np.array([], dtype=np.uint64), np.array([], dtype=float)
    for start in range(0, 2**model.n_bits, 16384):
        ids = np.arange(start, min(start + 16384, 2**model.n_bits), dtype=np.uint64)
        bits = ((ids[:, None] >> shifts) & 1).astype(np.int8)
        energies = model.offset + np.einsum('bi,ij,bj->b', bits, matrix, bits, optimize=False)
        best = min(best, float(energies.min()))
        old = ground_energies <= best + tolerance
        selected = energies <= best + tolerance
        ground_ids = np.r_[ground_ids[old], ids[selected]]
        ground_energies = np.r_[ground_energies[old], energies[selected]]
    bits = ((ground_ids[:, None] >> shifts) & 1).astype(np.int8)
    valid = ~np.any(bits[:, :model.n] + bits[:, model.n:] == 2, axis=1)
    costs, feasible = np.full(len(bits), np.nan), np.zeros(len(bits), dtype=bool)
    if valid.any():
        a, b = bits[valid, :model.n], bits[valid, model.n:]
        labels = np.where(a, 1, np.where(b, 2, 3))
        costs[valid], _, feasible[valid] = _label_costs(model, labels)
    return _ground_summary(model, bits, feasible, costs, optimum, best,
                           'exhaustive_binary_enumeration', 2**model.n_bits)


def build_reference(model, *, check_all_bitstrings=False,
                    max_assignments=config.MAX_EXACT_ASSIGNMENTS):
    """Enumerate 3^N valid assignments for the true constrained reference.

    For the target QUBO, a sufficient eta proves no invalid ground states exist.
    If eta lacks that guarantee, enumerate all 4^N binary states instead.
    --check-all-bitstrings forces that independent binary check either way.
    No claim of exact capacity encoding is inferred from assignment validity.
    """
    if (isinstance(max_assignments, bool) or not isinstance(max_assignments, (int, np.integer))
            or max_assignments < 1 or 3**model.n > max_assignments):
        raise ValueError(f"Exact reference requires {3**model.n} valid assignments; limit is {max_assignments}.")
    labels = np.asarray(list(product((1, 2, 3), repeat=model.n)), dtype=np.int8)
    costs, loads, feasible = _label_costs(model, labels)
    optimum = float(costs[feasible].min()) if feasible.any() else None
    optimal = (np.zeros(len(labels), dtype=bool) if optimum is None else
               feasible & np.isclose(costs, optimum, rtol=numerical.COST_REL_TOL,
                                     atol=numerical.COST_ABS_TOL))
    representative = model.evaluate_assignment(model.encode(labels[np.flatnonzero(optimal)[0]])) if optimal.any() else None
    if check_all_bitstrings or not model.assignment_validity_guaranteed:
        qubo = _all_binary_ground_states(model, optimum)
    else:
        energies = costs + model.penalty_weight * np.sum((loads - model.total_demand / 3)**2, axis=1)
        minimum = float(energies.min())
        tolerance = numerical.GROUND_STATE_REL_TOL * max(float(np.abs(model.build_q_matrix()).sum()), 1.)
        ground = np.abs(energies - minimum) <= tolerance
        bits = np.column_stack((labels[ground] == 1, labels[ground] == 2)).astype(np.int8)
        qubo = _ground_summary(model, bits, feasible[ground], costs[ground], optimum, minimum,
                               'valid_assignment_enumeration_and_sufficient_eta_proof', 0)
    return {
        'status': 'proven_optimum' if optimum is not None else 'infeasible_instance',
        'optimal_cost': optimum, 'valid_assignments_enumerated': len(labels),
        'feasible_assignments': int(feasible.sum()), 'optimal_assignments': int(optimal.sum()),
        'optimal_partitions': len({canonical_partition(row) for row in labels[optimal]}),
        'representative': representative, 'qubo': qubo,
        'settings': model.settings(),
        'instance': {'coordinates': model.coordinates.tolist(), 'demands': model.demands.tolist(),
                     'capacity': model.capacity},
    }


def add_model_arguments(parser):
    parser.add_argument('--instance', type=Path, help='certified Tier M JSON; default: generated seed 0')
    parser.add_argument('--lambda', dest='penalty_weight', type=float, default=None)
    parser.add_argument('--eta', dest='assignment_penalty', type=float, default=None,
                        help='assignment penalty; default: sufficient validity bound plus 1')


def model_from_args(args, *, saved_instance=None, saved_settings=None):
    from duquantum.tier_m.instances import load_instance
    instance = load_instance(args.instance) if args.instance is not None else saved_instance
    if args.instance is not None and saved_instance is not None:
        for key in ('coordinates', 'demands', 'capacity'):
            if instance[key] != saved_instance[key]:
                raise ValueError(f"--instance differs from the saved run's {key}.")
    saved_settings = saved_settings or {}
    weights = {}
    for key in ('penalty_weight', 'assignment_penalty'):
        explicit = getattr(args, key)
        saved = saved_settings.get(key)
        if explicit is not None and saved is not None and float(explicit) != float(saved):
            raise ValueError(f"{key} differs from the saved run.")
        weights[key] = saved if explicit is None else explicit
    return build_model(instance, **weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument('--check-all-bitstrings', action='store_true',
                        help='independently enumerate all 2^20 binary states, including invalid states')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--qubo-output', type=Path, help='export offset, symmetric matrix, and bit order')
    args = parser.parse_args()
    try:
        model = model_from_args(args)
        reference = build_reference(model, check_all_bitstrings=args.check_all_bitstrings)
        text = json.dumps(reference, indent=2, allow_nan=False) + '\n'
        print(text, end='')
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        if args.qubo_output is not None:
            payload = {'offset': model.offset, 'q_matrix': model.build_q_matrix().tolist(),
                       'settings': model.settings(),
                       'instance': {'coordinates': model.coordinates.tolist(),
                                    'demands': model.demands.tolist(), 'capacity': model.capacity}}
            args.qubo_output.parent.mkdir(parents=True, exist_ok=True)
            args.qubo_output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    except (ValueError, OSError, TypeError) as error:
        parser.error(str(error))
