"""Build and validate a global-detuning Pulser sequence from a symmetric Q matrix.

No model selection or emulation is performed in this module."""


import numpy as np

from duquantum import config
from duquantum.quantum.embedding import embed_register
from duquantum.quantum.ising import _compare_ground_states, build_ising, make_pulse


def build_sequence(
    q_matrix,
    *,
    device=None,
    energy_scale=config.ENERGY_SCALE,
    duration_ns=config.PULSE_DURATION,
    omega_peak=config.OMEGA_PEAK,
    delta_initial=config.DELTA_INITIAL,
    seed=config.EMBEDDING_SEED,
    restarts=config.EMBEDDING_RESTARTS,
    maxiter=config.MAX_ITERATIONS,
    check_ground_states=config.CHECK_GROUND_STATES,
):
    """Return sequence, pulse, parameters, embedding, and final-energy diagnostics.

    q_matrix follows c + z.T @ M @ z. energy_scale=None chooses a conservative
    common scale <= config.AUTO_SCALE_MAX using the device's amplitude, detuning, and strongest
    realizable pair interaction. It does not optimize lambda or pulse quality.
    omega_peak/delta_initial overrides, if supplied, are physical rad/us values.
    No coefficient is independently clipped. Invalid overrides raise an error.

    Default pulse: sin-squared Rabi envelope and linear global-detuning sweep.
    diagnostics uses the commanded pulse endpoint and actual embedded couplings;
    it does not predict dynamical sampling probabilities or modulation effects.
    """
    import pulser
    from pulser import Sequence

    if device is None:
        device = getattr(pulser, config.DEVICE)
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
        energy_scale=1.0,
        interaction_coeff=None,
        omega_peak=omega_peak,
        delta_initial=delta_initial,
        duration_ns=duration_ns,
        check_ground_states=False,  # Only needed to determine hardware scale.
    )
    if energy_scale is None:
        limits = [config.AUTO_SCALE_MAX]
        if omega_peak is None and channel.max_amp is not None:
            limits.append(
                config.HARDWARE_SAFETY_FACTOR * channel.max_amp / unit["omega_peak"]
            )
        if channel.max_abs_detuning is not None:
            # The automatic starting detuning and final mean both scale with M.
            required = float(np.max(np.abs(unit["target_detunings"])))
            if required > 0:
                limits.append(
                    config.HARDWARE_SAFETY_FACTOR * channel.max_abs_detuning / required
                )
        strongest = float(unit["interaction_matrix"].max())
        if strongest > 0:
            max_pair = (
                device.interaction_coeff
                / (device.min_atom_distance * (1 + config.GEOMETRIC_MARGIN)) ** 6
            )
            limits.append(config.HARDWARE_SAFETY_FACTOR * max_pair / strongest)
        energy_scale = min(limits)

    parameters = build_ising(
        q_matrix,
        energy_scale=energy_scale,
        duration_ns=duration_ns,
        omega_peak=omega_peak,
        delta_initial=delta_initial,
        interaction_coeff=device.interaction_coeff,
        check_ground_states=check_ground_states,
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
        "target_vs_realized": _compare_ground_states(target_matrix, realized_matrix)
        if check_ground_states
        else None,
        "global_ideal_vs_realized": _compare_ground_states(
            parameters["global_q_matrix"], realized_matrix
        )
        if check_ground_states
        else None,
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






def print_program(program, *, penalty_weight):
    """Print the model, pulse, register, and final-energy diagnostics."""
    np.set_printoptions(
        precision=config.PRINT_PRECISION, suppress=config.PRINT_SUPPRESS
    )
    p, e, d = program["parameters"], program["embedding"], program["diagnostics"]
    print(f"Atoms: {p['n_qubits']}; global detuning only; lambda: {penalty_weight:g}")
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
    print(d["target_vs_realized"] if d["target_vs_realized"] is not None
          else "Skipped exhaustive 2^N checks (Tier M).")
    print("Pulser register and sequence validation passed.")
