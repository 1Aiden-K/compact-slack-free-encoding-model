"""Convert qubo.py's symmetric Q matrix into global-drive Pulser parameters.

Convention: F(z) = c + z.T @ M @ z, z_i = n_i = (1 - Z_i) / 2.
With phase zero, the physical Hamiltonian is
    H(t)/hbar = Omega(t)/2 * sum(X_i) - delta(t) * sum(n_i)
                + sum_{i<j} V[i,j] * n_i * n_j.

Frequencies and H/hbar coefficients are in rad/us; durations are in ns;
register distances are in um. energy_scale converts QUBO cost units to rad/us.
Only the mean diagonal is implemented by the global detuning. Pair couplings
still need geometric embedding; ideal distances need not fit a planar register.

Sources:
https://pulser.readthedocs.io/en/stable/tutorials/qubo.html
https://pulser.readthedocs.io/en/stable/pulses.html
"""

import argparse
from itertools import product

import numpy as np


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return value


def _compare_ground_states(target, encoded):
    """Compare diagonal energies in bit order z[0], ..., z[N-1]."""
    bits = np.asarray(list(product((0, 1), repeat=len(target))), dtype=int)
    target_energy = np.einsum("bi,ij,bj->b", bits, target, bits)
    encoded_energy = np.einsum("bi,ij,bj->b", bits, encoded, bits)

    # Scale tolerance to each Hamiltonian, including when energy_scale is tiny.
    def minima(energies, matrix):
        tolerance = 1e-10 * max(float(np.abs(matrix).sum()), np.finfo(float).tiny)
        return np.abs(energies - energies.min()) <= tolerance

    original = minima(target_energy, target)
    projected = minima(encoded_energy, encoded)
    return {
        "target_ground_states": bits[original].tolist(),
        "global_ground_states": bits[projected].tolist(),
        "shared_ground_states": bits[original & projected].tolist(),
        "all_global_ground_states_target_optimal": bool(np.all(original[projected])),
    }


def build_ising(
    q_matrix,
    *,
    energy_scale=1.0,
    duration_ns=4000,
    omega_peak=None,
    delta_initial=None,
    interaction_coeff=None,
    qubo_offset=0.0,
    check_ground_states=True,
):
    """Return a dictionary of target Hamiltonian and pulse parameters.

    q_matrix: symmetric M from qubo.build_q_matrix(); positive pair entries only
        (zero entries are allowed but imply infinite ideal pair separation).
    energy_scale: positive gamma applied to ALL target coefficients together.
    omega_peak: rad/us; defaults to the median positive target pair interaction.
        This is a starting heuristic, not an optimized pulse amplitude.
    delta_initial: negative rad/us; defaults to -max(abs(target_detunings)).
    interaction_coeff: optional device.interaction_coeff, in rad/us * um**6;
        required only to calculate ideal pair separations.
    qubo_offset: optional c, used only for the exact Pauli-Ising constant.
    check_ground_states: exhaustive comparison before geometric embedding;
        limited to 16 bits. It checks QUBO optima, not original capacities.

    Important returned keys:
      interaction_matrix: symmetric physical V, zero diagonal; each unordered
          pair appears ONCE in H, so V[i,j] = 2*gamma*M[i,j].
      target_detunings: exact -gamma*diag(M), generally unequal.
      delta_final: their global mean; this is an approximation.
      global_q_matrix: physical symmetric matrix with mean diagonal, such that
          its bitstring energies equal the ideal final global-drive H/hbar.
      ising_fields, ising_couplings, ising_offset: EXACT target Pauli-Z form,
          offset + fields @ Z + sum_{i<j} couplings[i,j]*Z_i*Z_j.
      rabi_samples, detuning_samples: duration_ns samples at 1 ns intervals,
          using sin-squared amplitude and linear detuning. Both endpoints are
          included, following Pulser's discrete waveform convention.

    No positions are fabricated and no values are independently clipped to
    hardware limits: choose a valid scale/register/pulse and validate a Sequence.
    """
    if np.iscomplexobj(q_matrix):
        raise ValueError("q_matrix must be real.")
    M = np.array(q_matrix, dtype=float, copy=True)
    if M.ndim != 2 or M.shape[0] == 0 or M.shape[0] != M.shape[1]:
        raise ValueError("q_matrix must be a nonempty square matrix.")
    if not np.isfinite(M).all() or not np.allclose(M, M.T, rtol=1e-10, atol=1e-12):
        raise ValueError("q_matrix must be finite and symmetric.")
    M = (M + M.T) / 2
    n = len(M)
    if check_ground_states and n > 16:
        raise ValueError(
            "Exhaustive checks support at most 16 bits; set check_ground_states=False."
        )
    scale = _positive(energy_scale, "energy_scale")
    if (
        isinstance(duration_ns, (bool, np.bool_))
        or not isinstance(duration_ns, (int, np.integer))
        or duration_ns < 2
    ):
        raise ValueError("duration_ns must be an integer of at least 2 ns.")
    duration_ns = int(duration_ns)
    offset = float(qubo_offset)
    if not np.isfinite(offset):
        raise ValueError("qubo_offset must be finite.")

    linear = scale * np.diag(M)
    interactions = 2 * scale * M.copy()
    np.fill_diagonal(interactions, 0.0)
    if np.any(interactions < 0):
        raise ValueError(
            "Negative pair interactions cannot be directly encoded with repulsive Rydberg interactions."
        )
    target_detunings = -linear
    delta_final = float(target_detunings.mean())
    if delta_initial is None:
        delta_initial = -float(np.max(np.abs(target_detunings)))
    delta_initial = float(delta_initial)
    if not np.isfinite(delta_initial) or delta_initial >= 0:
        raise ValueError(
            "Provide a finite negative delta_initial to initialize |00...0>."
        )
    if omega_peak is None:
        positive_pairs = interactions[np.triu_indices(n, 1)]
        positive_pairs = positive_pairs[positive_pairs > 0]
        if positive_pairs.size == 0:
            raise ValueError(
                "Provide omega_peak when there are no positive pair interactions."
            )
        omega_peak = float(np.median(positive_pairs))
    omega_peak = _positive(omega_peak, "omega_peak")

    global_matrix = interactions / 2
    np.fill_diagonal(global_matrix, -delta_final)
    distances = None
    if interaction_coeff is not None:
        coeff = _positive(interaction_coeff, "interaction_coeff")
        distances = np.full((n, n), np.inf)
        positive = interactions > 0
        distances[positive] = (coeff / interactions[positive]) ** (1 / 6)
        np.fill_diagonal(distances, 0.0)

    u = np.linspace(0.0, 1.0, duration_ns)
    rabi_samples = omega_peak * np.sin(np.pi * u) ** 2
    rabi_samples[[0, -1]] = 0.0
    return {
        "n_qubits": n,
        "q_matrix": M,
        "energy_scale": scale,
        "interaction_matrix": interactions,
        "target_detunings": target_detunings,
        "delta_initial": delta_initial,
        "delta_final": delta_final,
        "omega_peak": omega_peak,
        "phase": 0.0,
        "duration_ns": duration_ns,
        "rabi_samples": rabi_samples,
        "detuning_samples": delta_initial + (delta_final - delta_initial) * u,
        "target_distances_um": distances,
        "ising_fields": -linear / 2 - interactions.sum(axis=1) / 4,
        "ising_couplings": np.triu(interactions / 4, 1),
        "ising_offset": float(
            scale * offset + linear.sum() / 2 + np.triu(interactions, 1).sum() / 4
        ),
        "global_q_matrix": global_matrix,
        "diagonal_error": np.diag(global_matrix) - linear,
        "exact_global_diagonal": bool(
            np.allclose(target_detunings, delta_final, rtol=1e-10, atol=0.0)
        ),
        "ground_state_check": _compare_ground_states(scale * M, global_matrix)
        if check_ground_states
        else None,
    }


def make_pulse(parameters):
    """Create a Pulser Pulse; Pulser is required only when calling this helper.

    Add it to a declared Rydberg.Global channel on a valid embedded register.
    Sequence validation is still required; this function does not embed atoms.
    """
    from pulser import CustomWaveform, Pulse

    return Pulse(
        CustomWaveform(parameters["rabi_samples"]),
        CustomWaveform(parameters["detuning_samples"]),
        parameters["phase"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda", dest="penalty_weight", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--duration", type=int, default=4000, help="pulse duration in ns"
    )
    parser.add_argument("--omega-peak", type=float, default=None, help="rad/us")
    parser.add_argument(
        "--interaction-coeff", type=float, default=None, help="device.interaction_coeff"
    )
    args = parser.parse_args()
    from qubo import build_q_matrix, build_qubo

    try:
        parameters = build_ising(
            build_q_matrix(args.penalty_weight),
            energy_scale=args.scale,
            duration_ns=args.duration,
            omega_peak=args.omega_peak,
            interaction_coeff=args.interaction_coeff,
            qubo_offset=build_qubo(args.penalty_weight)[0],
        )
    except ValueError as error:
        parser.error(str(error))
    np.set_printoptions(precision=6, suppress=True)
    for key in (
        "n_qubits",
        "energy_scale",
        "target_detunings",
        "delta_initial",
        "delta_final",
        "omega_peak",
        "duration_ns",
        "interaction_matrix",
        "target_distances_um",
        "ising_fields",
        "ising_couplings",
        "ising_offset",
        "exact_global_diagonal",
        "diagonal_error",
        "ground_state_check",
    ):
        print(f"{key}:\n{parameters[key]}")
    print(
        "Ground-state comparison assumes exact pair interactions; recheck after embedding."
    )


if __name__ == "__main__":
    main()
