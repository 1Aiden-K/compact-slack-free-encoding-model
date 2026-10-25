"""Emulate a sequence built by pulses.py, using emu-sv or QuTiP.

Run with your existing config.py and model modules:
    uv run emulate.py --lambda 0.1 --scale 0.2 --seed 0 --shots 1000 --output run.json
    uv run emulate.py --backend qutip --lambda 0.1 --scale 0.2 --shots 1000

Install emu-sv with `uv add emu-sv`. QuTiP is an optional alternative,
installed with `uv add pulser-simulation`. Only the selected backend is imported.
emu-sv runs on CPU by default; --gpu requests a supported CUDA GPU.
By default the pulse is optimized on this instance before final emulation.
Use --evaluations to set the search budget or --no-optimize-pulses for baseline.

Sources:
https://pasqal-io.github.io/emulators/latest/emu_sv/api/
https://pulser.readthedocs.io/en/stable/tutorials/backends.html
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import config
from pulses import add_sequence_arguments, build_from_args, print_program


def _sv_probabilities(state, n_qubits):
    """Read emu-sv's physical computational indices: g=0 and r=1.

    emu-sv's StateVector.make() constructs the all-ground state with data[0]=1.
    Its tensor indices already encode physical excitation bitstrings. Do not
    reinterpret those indices through eigenstates metadata: versions exposing
    ('r', 'g') there would complement every bit, despite the physical tensor
    convention being g=0. QuTiP's separate conversion remains unchanged.

    Source: https://pasqal-io.github.io/emulators/latest/emu_sv/api/#statevectormake
    """
    amplitudes = state.data.detach().cpu().numpy().reshape(-1)
    if amplitudes.size != 2**n_qubits:
        raise ValueError("emu-sv returned a state vector with an unexpected size.")
    probabilities = np.abs(amplitudes) ** 2
    return {
        format(index, f"0{n_qubits}b"): float(probability)
        for index, probability in enumerate(probabilities)
    }


def emulate(
    sequence,
    *,
    shots=config.SHOTS,
    seed=config.MEASUREMENT_SEED,
    backend=config.EMULATOR,
    with_modulation=config.MODULATION,
    dt=config.EMULATOR_DT,
    gpu=config.GPU,
):
    """Evolve a Pulser sequence and draw reproducible final measurements.

    Returns counts, final probabilities, the native results object, and settings.
    Bitstrings follow register order, with r=1 and g=0, for both backends.
    The seed controls measurement sampling only, not deterministic evolution.
    No stochastic device noise is added. with_modulation includes the device's
    deterministic waveform response. Probabilities come from the numerical
    final state, before sampling, and retain both optimal vehicle labelings.

    dt is emu-sv's integration timestep in ns; use a smaller value to check
    numerical convergence. gpu=True requests CUDA and is only for emu-sv.
    Explicit backend selection never silently switches emulator on failure.
    """
    if (
        isinstance(shots, (bool, np.bool_))
        or not isinstance(shots, (int, np.integer))
        or shots <= 0
    ):
        raise ValueError("shots must be a positive integer.")
    if backend not in ("emu-sv", "qutip"):
        raise ValueError("backend must be 'emu-sv' or 'qutip'.")
    if backend == "qutip" and gpu:
        raise ValueError("--gpu is only supported by the emu-sv backend.")
    if backend == "emu-sv":
        if (
            isinstance(dt, (bool, np.bool_))
            or not isinstance(dt, (int, np.integer))
            or dt <= 0
        ):
            raise ValueError("dt must be a positive integer in ns.")

    # Import these only when emulation is requested, keeping pulses.py usable
    # without either emulator installed.
    from pulser import NoiseModel
    from pulser.backend import StateResult

    config_options = {
        "observables": (StateResult(evaluation_times=(1.0,)),),
        "default_evaluation_times": (1.0,),
        "with_modulation": with_modulation,
        "prefer_device_noise_model": False,
        "noise_model": NoiseModel(),
    }
    if backend == "emu-sv":
        try:
            from emu_sv import SVBackend, SVConfig
        except ImportError as error:
            raise ImportError(
                "Could not import emu-sv. Install it with 'uv add emu-sv', "
                "or use '--backend qutip' with pulser-simulation installed. "
                f"Original error: {error}"
            ) from error
        backend_config = SVConfig(
            dt=int(dt), gpu=gpu, log_level=config.EMULATOR_LOG_LEVEL, **config_options
        )
        results = SVBackend(sequence, config=backend_config).run()
    else:
        try:
            from pulser_simulation import QutipBackendV2, QutipConfig
        except ImportError as error:
            raise ImportError(
                "Could not import pulser-simulation. Install it with "
                f"'uv add pulser-simulation'. Original error: {error}"
            ) from error
        backend_config = QutipConfig(
            sampling_rate=config.QUTIP_SAMPLING_RATE, **config_options
        )
        results = QutipBackendV2(sequence, config=backend_config).run()

    if tuple(results.atom_order) != tuple(sequence.register.qubits):
        raise ValueError("Unexpected atom ordering in emulator results.")
    if backend == "emu-sv":
        probability_map = _sv_probabilities(
            results.final_state, len(sequence.register.qubits)
        )
    else:
        probability_map = results.final_state.bitstring_probabilities(
            one_state="r", cutoff=0.0
        )
    labels = sorted(probability_map)
    probabilities = np.array([probability_map[label] for label in labels], dtype=float)
    total = probabilities.sum()
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or not np.isfinite(total)
        or total <= 0
    ):
        raise ValueError("Emulator returned an invalid probability distribution.")
    probabilities /= total
    draws = np.random.default_rng(seed).multinomial(int(shots), probabilities)
    counts = Counter(
        {label: int(count) for label, count in zip(labels, draws) if count}
    )
    return {
        "counts": counts,
        "probabilities": dict(zip(labels, probabilities.tolist())),
        "results": results,
        "seed": seed,
        "backend": backend,
        "with_modulation": with_modulation,
        "dt_ns": int(dt) if backend == "emu-sv" else None,
        "gpu": gpu if backend == "emu-sv" else False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_sequence_arguments(parser)
    from optimize_pulses import (
        add_optimization_arguments,
        validate_optimization_settings,
    )

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
        help="model and CVRP reference-gap tolerance in percent (default: %(default)s)",
    )
    parser.add_argument(
        "--depot",
        type=float,
        nargs=2,
        default=config.DEPOT,
        help="fixed depot X Y for exact routing; otherwise use config.DEPOT",
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
            **vars(args),
            "epsilon_pct": args.epsilon,
            "dt_ns": args.dt,
        }
        validate_optimization_settings(optimization_settings)
        if args.optimize_pulses and args.skip_evaluation:
            raise ValueError(
                "Pulse optimization requires the exact routing reference; use --no-skip-evaluation."
            )
        from evaluator import (
            build_reference,
            configured_depot,
            encoding_from_program,
            evaluate_samples,
            print_evaluation,
        )

        # The routing reference cost is also used for good-shot pulse scoring
        # when optimization is enabled. Geometry and endpoint remain fixed.
        depot = configured_depot(args.depot)
        reference = (
            None
            if args.skip_evaluation
            else build_reference(args.penalty_weight, depot=depot)
        )
        program = build_from_args(args)
        optimization = None
        if args.optimize_pulses:
            from optimize_pulses import optimize_program
            from qubo import CAPACITY, COORDINATES, DEMANDS

            program, optimization = optimize_program(
                program,
                coordinates=COORDINATES,
                demands=DEMANDS,
                capacity=CAPACITY,
                depot=depot,
                settings=optimization_settings,
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
        from qubo import CAPACITY, COORDINATES, DEMANDS

        payload = {
            "counts": dict(output["counts"]),
            "probabilities": output["probabilities"],
            "instance": {
                "coordinates": np.asarray(COORDINATES).tolist(),
                "demands": np.asarray(DEMANDS).tolist(),
                "capacity": float(CAPACITY),
                "depot": depot,
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
            args.output.write_text(
                json.dumps(payload, indent=config.JSON_INDENT, allow_nan=False) + "\n"
            )
        except (OSError, ValueError) as error:
            parser.error(f"Could not save run: {error}")
        print(f"Saved run to {args.output}")


if __name__ == "__main__":
    main()
