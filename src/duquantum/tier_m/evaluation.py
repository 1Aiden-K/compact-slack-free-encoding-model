"""Evaluate twenty-bit Tier M samples without repairing invalid assignments."""
import argparse
import json
import math
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from duquantum import config
from duquantum.tier_m.classical import add_model_arguments, model_from_args, build_reference
from duquantum.tier_m.model import canonical_partition


def _gap(cost, optimum):
    if cost is None or optimum is None:
        return None
    if math.isclose(cost, optimum, rel_tol=config.COST_REL_TOL, abs_tol=config.COST_ABS_TOL):
        return 0.
    return None if optimum == 0 else 100 * (cost - optimum) / optimum


def _weights(values, model, *, counts):
    if not isinstance(values, Mapping) or not values:
        raise ValueError('Counts/probabilities must be a nonempty mapping.')
    normalized = {}
    for bits, value in values.items():
        if not isinstance(bits, str):
            raise ValueError('Measurement keys must be bitstrings.')
        model._bits(bits)
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral if counts else Real)
                or not np.isfinite(value) or value < 0):
            raise ValueError('Counts must be nonnegative integers; probabilities must be finite and nonnegative.')
        normalized[bits] = int(value) if counts else float(value)
    total = sum(normalized.values())
    if not np.isfinite(total) or total <= 0:
        raise ValueError('Total measurement weight must be finite and positive.')
    if not counts:
        if not math.isclose(total, 1., rel_tol=0, abs_tol=config.PROBABILITY_ABS_TOL):
            raise ValueError('Probabilities must sum to one, including invalid and infeasible states.')
        normalized = {bits: value / total for bits, value in normalized.items()}
    return normalized


def evaluate_samples(counts, *, model, probabilities=None, epsilon_pct=0., reference=None):
    if not np.isfinite(epsilon_pct) or epsilon_pct < 0:
        raise ValueError('epsilon_pct must be finite and nonnegative.')
    counts = _weights(counts, model, counts=True)
    probabilities = None if probabilities is None else _weights(probabilities, model, counts=False)
    reference = build_reference(model) if reference is None else reference
    if reference['settings'] != model.settings():
        raise ValueError('Reference settings differ from the measured model.')
    # Settings alone cannot identify an instance: retain an explicit identity.
    expected = {'coordinates': model.coordinates.tolist(), 'demands': model.demands.tolist(),
                'capacity': model.capacity}
    if reference.get('instance') != expected:
        raise ValueError('Reference instance differs from the measured model.')
    optimum = reference['optimal_cost']
    records = {bits: model.evaluate_assignment(bits) for bits in set(counts) | set(probabilities or {})}

    def rates(weights):
        total = sum(weights.values())
        valid = sum(value for bits, value in weights.items() if records[bits]['assignment_valid'])
        feasible = sum(value for bits, value in weights.items() if records[bits]['feasible'])
        optimal, reached = 0., 0.
        for bits, value in weights.items():
            if not records[bits]['feasible']:
                continue
            gap = _gap(records[bits]['cost'], optimum)
            if gap == 0:
                optimal += value
            if gap is not None and gap <= epsilon_pct:
                reached += value
        return {
            'assignment_validity_rate': valid / total,
            'feasibility_rate': feasible / total,
            'optimal_probability': optimal / total if optimum is not None else None,
            'reference_probability': reached / total if optimum is not None else None,
        }

    feasible_bits = [bits for bits, count in counts.items() if count > 0 and records[bits]['feasible']]
    best_cost = min((records[bits]['cost'] for bits in feasible_bits), default=None)
    best = {}
    if best_cost is not None:
        for bits in sorted(feasible_bits):
            record = records[bits]
            if math.isclose(record['cost'], best_cost, rel_tol=config.COST_REL_TOL, abs_tol=config.COST_ABS_TOL):
                best.setdefault(canonical_partition(record['assignments']), record)
    sampled = rates(counts)
    sampled.update({
        'shots': sum(counts.values()),
        'assignment_valid_shots': sum(count for bits, count in counts.items() if records[bits]['assignment_valid']),
        'feasible_shots': sum(count for bits, count in counts.items() if records[bits]['feasible']),
        'best_feasible_cost': best_cost, 'best_model_gap_pct': _gap(best_cost, optimum),
        'best_solutions': list(best.values()),
    })
    return {
        'track': 'A', 'tier': 'M', 'objective': 'capacity_constrained_clustering',
        'reference': reference, 'epsilon_pct': float(epsilon_pct),
        'sampled': sampled, 'predicted': None if probabilities is None else rates(probabilities),
        'bitstrings': [dict(records[bits], count=counts.get(bits, 0),
                            probability=None if probabilities is None else probabilities.get(bits, 0.))
                       for bits in sorted(records, key=lambda key: (-counts.get(key, 0), key))],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='raw twenty-bit counts or a saved counts/instance payload')
    add_model_arguments(parser)
    parser.add_argument('--epsilon', type=float, default=0.)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        payload = json.loads(args.input.read_text())
        if not isinstance(payload, dict):
            raise ValueError('Input must be a JSON object.')
        is_run = 'counts' in payload
        if is_run and payload.get('tier', 'M') != 'M':
            raise ValueError('Saved run is not Tier M.')
        model = model_from_args(args, saved_instance=payload.get('instance') if is_run else None,
                                saved_settings=payload.get('settings') if is_run else None)
        report = evaluate_samples(payload['counts'] if is_run else payload, model=model,
                                  probabilities=payload.get('probabilities') if is_run else None,
                                  epsilon_pct=args.epsilon)
        text = json.dumps(report, indent=2, allow_nan=False) + '\n'
        print(text, end='')
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
    except (ValueError, OSError, TypeError) as error:
        parser.error(str(error))
