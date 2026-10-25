"""Classically benchmark fixed vs. adaptive lambda on Tier M instances.

Only the reference checker enumerates assignments. The adaptive rule itself
reads demands and pairwise distances, not reference costs or assignments.
The results describe ideal QUBO ground states, not emulator measurements.
"""

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import numpy as np

from duquantum import config as numerical
from duquantum.tier_m.classical import _label_costs
from duquantum.tier_m.instances import generate_instance
from duquantum.tier_m.model import build_model
from duquantum.tier_m.penalty import STRATEGY


FIELDS = ('seed', 'split', 'method', 'lambda', 'total_demand', 'unused_capacity',
          'ground_state_count', 'all_ground_states_feasible',
          'all_ground_states_clustering_optimal', 'minimum_energy',
          'constrained_optimum_cost', 'best_feasible_ground_cost',
          'best_feasible_ground_gap_pct')


def parse_seeds(spec):
    """Parse an exclusive range '0:20', or a comma-separated seed list."""
    if ':' in spec:
        left, right = spec.split(':', 1)
        start, stop = int(left), int(right)
        if start < 0 or stop <= start:
            raise ValueError('Expected --seeds START:STOP with 0 <= START < STOP.')
        return list(range(start, stop))
    result = [int(part) for part in spec.split(',')]
    if not result or any(seed < 0 for seed in result) or len(set(result)) != len(result):
        raise ValueError('Seeds must be distinct nonnegative integers.')
    return result


def benchmark(seeds, *, fixed=(0.1, 0.3, 1.0, 5.0, 50.0), train_stop=20):
    """Evaluate the ideal target QUBO using its sufficient eta proof.

    The exhaustive valid-assignment enumeration is for *benchmarking* only;
    no information derived from it is used to choose adaptive lambda.
    """
    fixed = tuple(float(v) for v in fixed)
    if any(not np.isfinite(v) or v <= 0 for v in fixed):
        raise ValueError('Fixed lambdas must be positive and finite.')
    rows = []
    for seed in seeds:
        instance = generate_instance(seed)
        model = build_model(instance, penalty_weight='auto')
        labels = np.asarray(list(product((1, 2, 3), repeat=model.n)), dtype=np.int8)
        costs, loads, feasible = _label_costs(model, labels)
        optimum = float(costs[feasible].min()) if feasible.any() else None
        balance = np.sum((loads - model.total_demand / 3)**2, axis=1)
        methods = [('adaptive', model)] + [
            (f'fixed_{weight:g}', build_model(instance, penalty_weight=weight))
            for weight in fixed
        ]
        for method, candidate in methods:
            if not candidate.assignment_validity_guaranteed:
                raise ValueError('Benchmark requires an eta that proves valid assignment ground states.')
            energies = costs + candidate.penalty_weight * balance
            ground_energy = float(energies.min())
            tolerance = (numerical.GROUND_STATE_REL_TOL *
                         max(float(np.abs(candidate.build_q_matrix()).sum()), 1.))
            ground = np.abs(energies - ground_energy) <= tolerance
            all_feasible = bool(np.all(feasible[ground]))
            all_optimal = bool(optimum is not None and np.all(
                feasible[ground] & np.isclose(costs[ground], optimum,
                    rtol=numerical.COST_REL_TOL, atol=numerical.COST_ABS_TOL)))
            feasible_ground_costs = costs[ground & feasible]
            best_cost = float(feasible_ground_costs.min()) if feasible_ground_costs.size else None
            gap = (None if best_cost is None or optimum is None else
                   0. if np.isclose(best_cost, optimum, rtol=numerical.COST_REL_TOL,
                                    atol=numerical.COST_ABS_TOL) else
                   (100 * (best_cost - optimum) / optimum if optimum else None))
            rows.append({
                'seed': seed, 'split': 'training' if seed < train_stop else 'holdout',
                'method': method, 'lambda': candidate.penalty_weight,
                'total_demand': model.total_demand,
                'unused_capacity': 3 * model.capacity - model.total_demand,
                'ground_state_count': int(ground.sum()),
                'all_ground_states_feasible': all_feasible,
                'all_ground_states_clustering_optimal': all_optimal,
                'minimum_energy': ground_energy, 'constrained_optimum_cost': optimum,
                'best_feasible_ground_cost': best_cost,
                'best_feasible_ground_gap_pct': gap,
            })
    by_method = {}
    for row in rows:
        key = f"{row['split']}/{row['method']}"
        summary = by_method.setdefault(key, {
            'instances': 0, 'all_ground_states_feasible': 0,
            'all_ground_states_clustering_optimal': 0})
        summary['instances'] += 1
        summary['all_ground_states_feasible'] += int(row['all_ground_states_feasible'])
        summary['all_ground_states_clustering_optimal'] += int(row['all_ground_states_clustering_optimal'])
    return {
        'description': 'Exact ground states of ideal Tier M QUBO (not an emulator)',
        'lambda_strategy': STRATEGY,
        'training_seeds': '0..19 used to choose heuristic coefficients; not used for per-instance selection',
        'heldout_seeds': '20 and above',
        'reference_use': 'exhaustive 3^10 enumeration for evaluation only',
        'summary': by_method,
        'rows': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='0:40', help='START:STOP (end exclusive), or comma-separated list')
    parser.add_argument('--train-stop', type=int, default=20)
    parser.add_argument('--fixed', nargs='+', type=float, default=[0.1, 0.3, 1., 5., 50.])
    parser.add_argument('--csv', type=Path, help='optional CSV output')
    parser.add_argument('--summary', type=Path, help='optional JSON output (aggregate plus rows)')
    args = parser.parse_args()
    try:
        report = benchmark(parse_seeds(args.seeds), fixed=args.fixed, train_stop=args.train_stop)
        print(json.dumps(report['summary'], indent=2))
        if args.csv is not None:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with args.csv.open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(report['rows'])
        if args.summary is not None:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    except (ValueError, OSError) as error:
        parser.error(str(error))
