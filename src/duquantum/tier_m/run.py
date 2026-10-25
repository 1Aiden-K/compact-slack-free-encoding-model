"""Tier M global-drive simulation: 20-bit model, device embedding, and shots.

The quantum circuit is built exactly as in Tier S, but expensive 2^20 exact
bitstring comparisons and the dense final probability JSON are disabled.
Run --prepare-only to test the physical register/pulse without emulation.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from duquantum import config
from duquantum.tier_m.classical import (
    add_model_arguments, build_reference, model_from_args,
)
from duquantum.tier_m.evaluation import evaluate_samples


def add_run_arguments(parser):
    add_model_arguments(parser)
    # In contrast with Tier S, the default penalty is instance dependent.
    parser.set_defaults(penalty_weight='auto')
    parser.add_argument('--scale', type=float, default=config.ENERGY_SCALE)
    parser.add_argument('--duration', type=int, default=config.PULSE_DURATION, help='pulse length in ns')
    parser.add_argument('--omega-peak', type=float, default=config.OMEGA_PEAK, help='rad/us')
    parser.add_argument('--delta-initial', type=float, default=config.DELTA_INITIAL, help='rad/us')
    parser.add_argument('--seed', type=int, default=config.EMBEDDING_SEED, help='embedding seed')
    parser.add_argument('--restarts', type=int, default=2, help='geometry optimizer restarts')
    parser.add_argument('--maxiter', type=int, default=250, help='optimizer iterations per restart')
    parser.add_argument('--backend', choices=('emu-sv', 'emu-mps'), default='emu-mps')
    parser.add_argument('--dt', type=int, default=config.EMULATOR_DT, help='emulation integration timestep in ns')
    parser.add_argument('--gpu', action='store_true', help='CUDA acceleration for emu-sv only')
    parser.add_argument('--mps-bond-dim', type=int, default=256, help='emu-mps maximum bond dimension')
    parser.add_argument('--mps-precision', type=float, default=1e-5, help='emu-mps truncation tolerance')
    parser.add_argument('--shots', type=int, default=config.SHOTS)
    parser.add_argument('--sample-seed', type=int, default=config.MEASUREMENT_SEED)
    parser.add_argument('--epsilon', type=float, default=config.EPSILON_PCT, help='reference gap threshold (percent)')
    parser.add_argument('--with-modulation', action='store_true')
    parser.add_argument('--prepare-only', action='store_true', help='compile pulse/register; do not simulate')
    parser.add_argument('--skip-evaluation', action='store_true', help='skip constrained reference enumeration')
    parser.add_argument('--ground-check', action='store_true', help='enumerate final Hamiltonian ground states (2^20)')
    parser.add_argument('--output', type=Path, help='save compact run record to JSON')


def _ground_report(matrix, model, reference):
    """Chunk the 2^20 diagonal energies; no million-entry Python dictionaries."""
    from duquantum.tier_m.classical import _all_binary_ground_states
    # Reuse the established exact binary enumerator with a matrix substituted
    # into a lightweight model proxy rather than duplicating its tolerance rules.
    class MatrixProxy:
        n = model.n
        n_bits = model.n_bits
        offset = 0.0
        def build_q_matrix(self):
            return matrix
        def evaluate_assignment(self, bits):
            result = model.evaluate_assignment(bits)
            vector = model._bits(bits)
            result["hamiltonian_energy_without_offset"] = float(vector @ matrix @ vector)
            return result
        def __getattr__(self, key):
            return getattr(model, key)

    return _all_binary_ground_states(MatrixProxy(), reference['optimal_cost'])


def make_run(model, args, *, sequence_builder=None, emulation_fn=None):
    """Compute run output; injectable interfaces support backend-free unit tests."""
    if args.shots <= 0 or args.dt <= 0:
        raise ValueError('--shots and --dt must be positive.')
    if args.gpu and args.backend != 'emu-sv':
        raise ValueError('--gpu requires --backend emu-sv.')
    if not np.isfinite(args.epsilon) or args.epsilon < 0:
        raise ValueError('--epsilon must be finite and nonnegative.')
    if args.mps_bond_dim < 1 or not 0 < args.mps_precision < 1:
        raise ValueError('Invalid MPS truncation settings.')

    if sequence_builder is None:
        from duquantum.quantum.pulses import build_sequence
        sequence_builder = build_sequence
    program = sequence_builder(
        model.build_q_matrix(), energy_scale=args.scale, duration_ns=args.duration,
        omega_peak=args.omega_peak, delta_initial=args.delta_initial,
        seed=args.seed, restarts=args.restarts, maxiter=args.maxiter,
        check_ground_states=False,
    )
    reference = None if args.skip_evaluation else build_reference(model)
    parameters = program['parameters']
    embedding = program['embedding']
    diagnostics = program['diagnostics']
    encoding = {
        'atoms': model.n_bits,
        'energy_scale': float(parameters['energy_scale']),
        'relative_interaction_error': float(diagnostics['relative_interaction_error']),
        'relative_matrix_error': float(diagnostics['relative_matrix_error']),
        'min_separation_um': float(embedding['min_separation_um']),
        'max_radius_um': float(embedding['max_radius_um']),
        'optimizer_converged': bool(embedding['optimizer_converged']),
        'optimizer_message': str(embedding['optimizer_message']),
        'positions_um': np.asarray(embedding['positions_um']).tolist(),
        'delta_initial': float(parameters['delta_initial']),
        'delta_final': float(parameters['delta_final']),
        'omega_peak': float(parameters['omega_peak']),
        'duration_ns': int(parameters['duration_ns']),
    }
    if args.ground_check:
        if reference is None:
            raise ValueError('--ground-check requires reference evaluation.')
        encoding['ground_state_checks'] = {
            'target': reference['qubo'],
            'global_detuning': _ground_report(parameters['global_q_matrix'], model, reference),
            'embedded': _ground_report(diagnostics['realized_q_matrix'], model, reference),
        }
    payload = {
        'tier': 'M', 'track': 'A', 'objective': 'capacity_constrained_clustering',
        'instance': {'coordinates': model.coordinates.tolist(),
                     'demands': model.demands.tolist(), 'capacity': model.capacity},
        'settings': model.settings(),
        'emulation': {
            'backend': args.backend, 'dt_ns': args.dt, 'shots': args.shots,
            'sample_seed': args.sample_seed, 'gpu': args.gpu,
            'with_modulation': args.with_modulation,
            'mps_max_bond_dim': args.mps_bond_dim if args.backend == 'emu-mps' else None,
            'mps_precision': args.mps_precision if args.backend == 'emu-mps' else None,
            'prepare_only': args.prepare_only,
        },
        'encoding': encoding,
    }
    if args.prepare_only:
        return payload
    if emulation_fn is None:
        from duquantum.quantum.emulator import emulate
        emulation_fn = emulate
    output = emulation_fn(
        program['sequence'], shots=args.shots, seed=args.sample_seed, backend=args.backend,
        with_modulation=args.with_modulation, dt=args.dt, gpu=args.gpu,
        compact=True, mps_max_bond_dim=args.mps_bond_dim,
        mps_precision=args.mps_precision,
    )
    payload['counts'] = dict(output['counts'])
    # Exact state probabilities are intentionally not serialized for 20 atoms;
    # the measured distribution is enough for the challenge's shot-based metrics.
    payload['probabilities'] = None
    payload['evaluation'] = (
        None if reference is None else evaluate_samples(output['counts'], model=model,
                                                        epsilon_pct=args.epsilon, reference=reference)
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    args = parser.parse_args()
    try:
        model = model_from_args(args)
        print(f'Tier M: {model.n_bits} atoms, lambda={model.penalty_weight:.6g} '
              f'({model.penalty_strategy}), eta={model.assignment_penalty:.6g}', flush=True)
        print('Embedding and compiling physical pulse...', flush=True)
        result = make_run(model, args)
        enc = result['encoding']
        print(f"Embedding interaction error: {100 * enc['relative_interaction_error']:.2f}%")
        print(f"Full matrix error: {100 * enc['relative_matrix_error']:.2f}%")
        print(f"Geometry: min separation {enc['min_separation_um']:.3f} um; "
              f"radius {enc['max_radius_um']:.3f} um")
        if args.ground_check:
            checks = enc['ground_state_checks']
            for stage in ('target', 'global_detuning', 'embedded'):
                item = checks[stage]
                print(f"{stage}: {item['ground_feasible_assignments']}/"
                      f"{item['ground_assignment_count']} feasible ground states; "
                      f"all clustering-optimal={item['all_ground_states_original_optimal']}")
        if not args.prepare_only:
            evaluation = result['evaluation']
            print(f"Backend: {args.backend}; samples: {sum(result['counts'].values())}")
            if evaluation is not None:
                print(json.dumps(evaluation['sampled'], indent=2, allow_nan=False))
            else:
                print('Evaluation skipped.')
            print('Most common bitstrings:')
            for bits, count in sorted(result['counts'].items(), key=lambda item: -item[1])[:10]:
                print(f'{bits}: {count}')
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
            print(f"Saved: {args.output}")
    except (ValueError, ImportError, OSError, TypeError) as error:
        parser.error(str(error))
