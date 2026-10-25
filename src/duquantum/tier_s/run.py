"""Run the Tier S model through embedding, pulse search, and final emulation.

Run through: python main.py run --instance instances/tier_s/tier_s_seed_0000.json
"""
import argparse
import json
from pathlib import Path
import numpy as np
from duquantum import config
from duquantum.tier_s import config as tier_config
from duquantum.quantum.emulator import emulate
from duquantum.quantum.pulses import print_program

def add_sequence_arguments(parser):
    """Add shared model, embedding, and pulse options to a CLI parser."""
    parser.add_argument(
        "--lambda", dest="penalty_weight", type=float, default=tier_config.LAMBDA
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=config.ENERGY_SCALE,
        help="common scaling; config default may select automatic scaling",
    )
    parser.add_argument(
        "--duration", type=int, default=config.PULSE_DURATION, help="ns"
    )
    parser.add_argument(
        "--omega-peak", type=float, default=config.OMEGA_PEAK, help="rad/us"
    )
    parser.add_argument("--seed", type=int, default=config.EMBEDDING_SEED)
    parser.add_argument("--restarts", type=int, default=config.EMBEDDING_RESTARTS)
    parser.add_argument("--maxiter", type=int, default=config.MAX_ITERATIONS)
    parser.add_argument(
        "--delta-initial", type=float, default=config.DELTA_INITIAL, help="rad/us"
    )


def build_from_args(args):
    """Build a program from the selected instance and shared CLI options."""
    from duquantum.tier_s.model import build_q_matrix
    from duquantum.quantum.pulses import build_sequence

    return build_sequence(
        build_q_matrix(args.penalty_weight),
        energy_scale=args.scale,
        duration_ns=args.duration,
        omega_peak=args.omega_peak,
        seed=args.seed,
        restarts=args.restarts,
        maxiter=args.maxiter,
        delta_initial=args.delta_initial,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_sequence_arguments(parser)
    parser.add_argument("--instance", type=Path, help="certified Tier S instance JSON")
    from duquantum.tier_s.optimization import add_optimization_arguments, validate_optimization_settings
    add_optimization_arguments(parser)
    parser.add_argument(
        "--backend", choices=("emu-sv", "qutip"), default=config.EMULATOR
    )
    parser.add_argument("--shots", type=int, default=config.SHOTS)
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=config.MEASUREMENT_SEED,
        help="measurement seed (default: config.MEASUREMENT_SEED)",
    )
    parser.add_argument(
        "--with-modulation",
        action=argparse.BooleanOptionalAction,
        default=config.MODULATION,
    )
    parser.add_argument(
        "--dt",
        type=int,
        default=config.EMULATOR_DT,
        help="emu-sv timestep in ns (default: %(default)s)",
    )
    parser.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        default=config.GPU,
        help="request CUDA for emu-sv",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=config.EPSILON_PCT,
        help="Track A clustering reference-gap tolerance in percent (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-evaluation",
        action=argparse.BooleanOptionalAction,
        default=config.SKIP_EVALUATION,
        help="omit the exhaustive reference and evaluator when the configured exact-bit limit is exceeded",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.RUN_OUTPUT,
        help="save counts, probabilities, instance, settings, encoding, and evaluation as JSON",
    )
    args = parser.parse_args()
    if args.shots <= 0:
        parser.error("--shots must be positive.")
    if args.backend == "emu-sv" and args.dt <= 0:
        parser.error("--dt must be positive.")
    if args.gpu and args.backend != "emu-sv":
        parser.error("--gpu is only supported by --backend emu-sv.")
    if not np.isfinite(args.epsilon) or args.epsilon < 0:
        parser.error("--epsilon must be finite and nonnegative.")
    sample_seed = args.sample_seed

    try:
        optimization_settings = {
            **vars(args), "epsilon_pct": args.epsilon, "dt_ns": args.dt,
        }
        validate_optimization_settings(optimization_settings)
        if args.optimize_pulses and args.skip_evaluation:
            raise ValueError("Pulse optimization requires the exact clustering reference; use --no-skip-evaluation.")
        from duquantum.tier_s.optimization import _load_cli_instance
        _load_cli_instance(args.instance)
        from duquantum.tier_s.evaluation import (
            build_reference,
            encoding_from_program,
            evaluate_samples,
            print_evaluation,
        )

        # The clustering reference cost is also used for good-shot pulse scoring
        # when optimization is enabled. Geometry and endpoint remain fixed.
        reference = (
            None
            if args.skip_evaluation
            else build_reference(args.penalty_weight)
        )
        program = build_from_args(args)
        optimization = None
        if args.optimize_pulses:
            from duquantum.tier_s.optimization import optimize_program
            from duquantum.tier_s.model import CAPACITY, COORDINATES, DEMANDS

            program, optimization = optimize_program(
                program, coordinates=COORDINATES, demands=DEMANDS,
                capacity=CAPACITY, settings=optimization_settings,
            )
        print_program(program, penalty_weight=args.penalty_weight)
        print(f"Running {args.backend} emulation...")
        output = emulate(
            program["sequence"],
            shots=args.shots,
            seed=sample_seed,
            backend=args.backend,
            with_modulation=args.with_modulation,
            dt=args.dt,
            gpu=args.gpu,
        )
        encoding = encoding_from_program(program)
        evaluation = (
            None
            if args.skip_evaluation
            else evaluate_samples(
                output["counts"],
                probabilities=output["probabilities"],
                penalty_weight=args.penalty_weight,
                reference=reference,
                encoding=encoding,
                epsilon_pct=args.epsilon,
            )
        )
    except (ValueError, ImportError) as error:
        parser.error(str(error))

    print(
        f"Backend: {output['backend']}; modulation: {args.with_modulation}; "
        "ideal coherent emulation, no stochastic noise"
    )
    if args.backend == "emu-sv":
        print(f"Timestep: {args.dt} ns; compute: {'CUDA' if args.gpu else 'CPU'}")
    print(f"Shots: {args.shots}; measurement seed: {sample_seed}")
    print("Bitstring counts:")
    for bits, count in output["counts"].most_common():
        print(bits, count)
    if evaluation is not None:
        print_evaluation(evaluation)
    if args.output is not None:
        from duquantum.tier_s.model import CAPACITY, COORDINATES, DEMANDS

        payload = {
            "track": "A", "objective": "capacity_constrained_clustering",
            "counts": dict(output["counts"]),
            "probabilities": output["probabilities"],
            "instance": {
                "coordinates": np.asarray(COORDINATES).tolist(),
                "demands": np.asarray(DEMANDS).tolist(),
                "capacity": float(CAPACITY),
            },
            "settings": {
                "penalty_weight": args.penalty_weight,
                "energy_scale": program["parameters"]["energy_scale"],
                "duration_ns": program["parameters"]["duration_ns"],
                "omega_peak": program["parameters"]["omega_peak"],
                "delta_initial": program["parameters"]["delta_initial"],
                "delta_final": program["parameters"]["delta_final"],
                "embedding_seed": args.seed,
                "embedding_restarts": args.restarts,
                "embedding_maxiter": args.maxiter,
                "sample_seed": sample_seed,
                "shots": args.shots,
                "backend": output["backend"],
                "dt_ns": output["dt_ns"],
                "gpu": output["gpu"],
                "with_modulation": args.with_modulation,
                "optimize_pulses": args.optimize_pulses,
                "search_seed": args.search_seed,
                "training_seed": args.training_seed,
                "training_shots": args.training_shots,
                "evaluations": args.evaluations,
                "score_mode": args.score_mode,
                "metric": args.metric,
                "duration_bounds": args.duration_bounds,
            },
            "optimization": optimization,
            "positions_um": program["embedding"]["positions_um"].tolist(),
            "encoding": encoding,
            "evaluation": evaluation,
        }
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(payload, indent=config.JSON_INDENT, allow_nan=False) + "\n"
            )
        except (OSError, ValueError) as error:
            parser.error(f"Could not save run: {error}")
        print(f"Saved run to {args.output}")
