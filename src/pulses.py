"""Build a global-detuning AnalogDevice sequence from a symmetric Q matrix.

Requires numpy, scipy, pulser-core, and the local ising.py / embedding.py.
Optional emulation also requires pulser-simulation. No DMM or local driving.

Run with your existing config.py:
    python pulses.py --lambda 0.1 --seed 0
    python pulses.py --lambda 0.1 --seed 0 --shots 1000

Sources:
https://pulser.readthedocs.io/en/stable/tutorials/qubo.html
https://pulser.readthedocs.io/en/stable/pulses.html
https://pulser.readthedocs.io/en/stable/apidoc/_autosummary/pulser_simulation.QutipBackendV2.html
"""

import argparse
from collections import Counter

import numpy as np

from embedding import embed_register
from ising import _compare_ground_states, build_ising, make_pulse


def build_sequence(
    q_matrix,
    *,
    device=None,
    energy_scale=None,
    duration_ns=4000,
    omega_peak=None,
    delta_initial=None,
    seed=0,
    restarts=8,
    maxiter=1500,
):
    """Return sequence, pulse, parameters, embedding, and final-energy diagnostics.

    q_matrix follows c + z.T @ M @ z. energy_scale=None chooses a conservative
    common scale <= 1 using the device's amplitude, detuning, and strongest
    realizable pair interaction. It does not optimize lambda or pulse quality.
    omega_peak/delta_initial overrides, if supplied, are physical rad/us values.
    No coefficient is independently clipped. Invalid overrides raise an error.

    Default pulse: sin-squared Rabi envelope and linear global-detuning sweep.
    diagnostics uses the commanded pulse endpoint and actual embedded couplings;
    it does not predict dynamical sampling probabilities or modulation effects.
    """
    from pulser import AnalogDevice, Sequence

    if device is None:
        device = AnalogDevice
    channel_id = "rydberg_global"
    if channel_id not in device.channels:
        raise ValueError("The selected device must expose a rydberg_global channel.")
    channel = device.channels[channel_id]
    if isinstance(duration_ns, (bool, np.bool_)) or not isinstance(
        duration_ns, (int, np.integer)
    ):
        raise ValueError("duration_ns must be an integer in ns.")
    if duration_ns % channel.clock_period:
        raise ValueError(
            f"duration_ns must be a multiple of {channel.clock_period} ns."
        )
    if duration_ns < channel.min_duration:
        raise ValueError("Pulse duration is below the channel minimum.")
    for limit in (channel.max_duration, device.max_sequence_duration):
        if limit is not None and duration_ns > limit:
            raise ValueError(f"Pulse duration exceeds the allowed {limit} ns.")

    unit = build_ising(
        q_matrix,
        omega_peak=omega_peak,
        delta_initial=delta_initial,
        duration_ns=duration_ns,
    )
    if energy_scale is None:
        limits = [1.0]
        if omega_peak is None and channel.max_amp is not None:
            limits.append(0.9 * channel.max_amp / unit["omega_peak"])
        if channel.max_abs_detuning is not None:
            # The automatic starting detuning and final mean both scale with M.
            required = float(np.max(np.abs(unit["target_detunings"])))
            if required > 0:
                limits.append(0.9 * channel.max_abs_detuning / required)
        strongest = float(unit["interaction_matrix"].max())
        if strongest > 0:
            max_pair = (
                device.interaction_coeff / (device.min_atom_distance * 1.0001) ** 6
            )
            limits.append(0.9 * max_pair / strongest)
        energy_scale = min(limits)

    parameters = build_ising(
        q_matrix,
        energy_scale=energy_scale,
        duration_ns=duration_ns,
        omega_peak=omega_peak,
        delta_initial=delta_initial,
        interaction_coeff=device.interaction_coeff,
    )
    if channel.max_amp is not None and parameters["omega_peak"] > channel.max_amp:
        raise ValueError(
            "Rabi peak exceeds the device maximum; reduce omega_peak or energy_scale."
        )
    max_detuning = float(np.max(np.abs(parameters["detuning_samples"])))
    if channel.max_abs_detuning is not None and max_detuning > channel.max_abs_detuning:
        raise ValueError(
            "Detuning exceeds the device maximum; reduce energy_scale or delta_initial."
        )

    embedding = embed_register(
        parameters["interaction_matrix"],
        device=device,
        seed=seed,
        restarts=restarts,
        maxiter=maxiter,
    )
    pulse = make_pulse(parameters)
    sequence = Sequence(embedding["register"], device)
    sequence.declare_channel("global", channel_id)
    sequence.add(pulse, "global")
    sequence.measure("ground-rydberg")
    total_duration = sequence.get_duration(include_fall_time=True)
    if (
        device.max_sequence_duration is not None
        and total_duration > device.max_sequence_duration
    ):
        raise ValueError(
            "Sequence including the modulation fall time exceeds the device duration limit."
        )

    realized_matrix = embedding["realized_interactions"] / 2
    np.fill_diagonal(realized_matrix, -parameters["delta_final"])
    target_matrix = parameters["energy_scale"] * parameters["q_matrix"]
    norm = np.linalg.norm(target_matrix)
    diagnostics = {
        "relative_matrix_error": float(
            np.linalg.norm(realized_matrix - target_matrix) / norm
        ),
        "relative_interaction_error": embedding["relative_interaction_error"],
        "target_vs_realized": _compare_ground_states(target_matrix, realized_matrix),
        "global_ideal_vs_realized": _compare_ground_states(
            parameters["global_q_matrix"], realized_matrix
        ),
        "realized_q_matrix": realized_matrix,
        "duration_with_fall_ns": total_duration,
    }
    return {
        "sequence": sequence,
        "pulse": pulse,
        "parameters": parameters,
        "embedding": embedding,
        "diagnostics": diagnostics,
    }


def emulate(sequence, *, shots=1000, seed=0, with_modulation=False):
    """Run ideal coherent QuTiP emulation and draw seeded measurements.

    with_modulation=True includes the device's deterministic waveform response;
    neither setting adds stochastic device noise. Samples use register order.
    Pulser converts its internal basis into bitstrings with r=1 and g=0.
    Returns counts, exact final probabilities, and the Pulser simulation result.
    Requires Pulser's QutipBackendV2 / StateResult API (1.9 or later).
    """
    from pulser import NoiseModel
    from pulser.backend import StateResult
    from pulser_simulation import QutipBackendV2, QutipConfig

    if (
        isinstance(shots, (bool, np.bool_))
        or not isinstance(shots, (int, np.integer))
        or shots <= 0
    ):
        raise ValueError("shots must be a positive integer.")
    config = QutipConfig(
        observables=(StateResult(),),
        default_evaluation_times=(1.0,),
        sampling_rate=1.0,
        with_modulation=with_modulation,
        prefer_device_noise_model=False,
        noise_model=NoiseModel(),
    )
    results = QutipBackendV2(sequence, config=config).run()
    probability_map = results.final_state.bitstring_probabilities(
        one_state="r", cutoff=0.0
    )
    if tuple(results.atom_order) != tuple(sequence.register.qubits):
        raise ValueError("Unexpected atom ordering in emulator results.")
    labels = sorted(probability_map)
    probabilities = np.array([probability_map[label] for label in labels], dtype=float)
    probabilities /= probabilities.sum()
    draws = np.random.default_rng(seed).multinomial(int(shots), probabilities)
    counts = Counter(
        {label: int(count) for label, count in zip(labels, draws) if count}
    )
    return {
        "counts": counts,
        "probabilities": dict(zip(labels, probabilities.tolist())),
        "results": results,
        "seed": seed,
        "with_modulation": with_modulation,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda", dest="penalty_weight", type=float, default=0.1)
    parser.add_argument(
        "--scale", type=float, default=None, help="default: automatic common scaling"
    )
    parser.add_argument("--duration", type=int, default=4000, help="ns")
    parser.add_argument("--omega-peak", type=float, default=None, help="rad/us")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument(
        "--shots", type=int, default=None, help="omit to build without emulation"
    )
    parser.add_argument("--with-modulation", action="store_true")
    args = parser.parse_args()
    from qubo import build_q_matrix

    try:
        program = build_sequence(
            build_q_matrix(args.penalty_weight),
            energy_scale=args.scale,
            duration_ns=args.duration,
            omega_peak=args.omega_peak,
            seed=args.seed,
            restarts=args.restarts,
        )
    except (ValueError, ImportError) as error:
        parser.error(str(error))
    np.set_printoptions(precision=6, suppress=True)
    p, e, d = program["parameters"], program["embedding"], program["diagnostics"]
    print(
        f"Atoms: {p['n_qubits']}; global detuning only; lambda: {args.penalty_weight:g}"
    )
    print(f"Energy scale: {p['energy_scale']:.6g}; duration: {p['duration_ns']} ns")
    print(
        f"Rabi peak: {p['omega_peak']:.6g}; detuning: {p['delta_initial']:.6g} -> {p['delta_final']:.6g} rad/us"
    )
    print("Atom positions (um), in Q-matrix row order:")
    print(e["positions_um"])
    print(
        f"Min spacing: {e['min_separation_um']:.6g} um; max radius: {e['max_radius_um']:.6g} um"
    )
    print(f"Optimizer converged: {e['optimizer_converged']}; {e['optimizer_message']}")
    print(f"Relative interaction error: {100 * d['relative_interaction_error']:.3f}%")
    print(f"Relative full-matrix error: {100 * d['relative_matrix_error']:.3f}%")
    print("Actual final-Hamiltonian ground-state checks:")
    print(d["target_vs_realized"])
    print("Pulser register and sequence validation passed.")
    if args.shots is not None:
        try:
            output = emulate(
                program["sequence"],
                shots=args.shots,
                seed=args.seed,
                with_modulation=args.with_modulation,
            )
        except (ValueError, ImportError) as error:
            parser.error(str(error))
        print(
            f"Modulation: {args.with_modulation}; ideal coherent emulation, no stochastic noise"
        )
        print("Bitstring counts:")
        for bits, count in output["counts"].most_common():
            print(bits, count)


if __name__ == "__main__":
    main()
