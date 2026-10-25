"""Ideal coherent emulation; backends are imported only when requested."""
from collections import Counter
import numpy as np
from duquantum import config

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
    compact=False,
    mps_max_bond_dim=256,
    mps_precision=1e-5,
):
    """Evolve a Pulser sequence and draw reproducible final measurements.

    Returns counts, final probabilities, the native results object, and settings.
    Bitstrings follow register order, with r=1 and g=0, for both backends.
    The seed controls measurement sampling only, not deterministic evolution.
    No stochastic device noise is added. with_modulation includes the device's
    deterministic waveform response. Probabilities come from the numerical
    final state, before sampling, and retain both optimal vehicle labelings.

    dt is the emulator integration timestep in ns; use a smaller value to check
    numerical convergence. gpu=True requests CUDA and is only for emu-sv.
    compact=True samples without constructing 2^N Python probability entries.
    emu-mps uses tensor-network truncation and samples via MPS.sample(); its
    shot-seed uses the local PyTorch CPU RNG, isolated from the caller's RNG.
    Explicit backend selection never silently switches emulator on failure.
    """
    if (
        isinstance(shots, (bool, np.bool_))
        or not isinstance(shots, (int, np.integer))
        or shots <= 0
    ):
        raise ValueError("shots must be a positive integer.")
    if backend not in ("emu-sv", "qutip", "emu-mps"):
        raise ValueError("backend must be 'emu-sv', 'emu-mps' or 'qutip'.")
    if backend != "emu-sv" and gpu:
        raise ValueError("--gpu is only supported by the emu-sv backend.")
    if backend in ("emu-sv", "emu-mps"):
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
    elif backend == "emu-mps":
        if not compact:
            raise ValueError("emu-mps requires compact=True (full probability distribution unavailable).")
        if isinstance(mps_max_bond_dim, bool) or mps_max_bond_dim < 1:
            raise ValueError("mps_max_bond_dim must be a positive integer.")
        if not np.isfinite(mps_precision) or not 0 < mps_precision < 1:
            raise ValueError("mps_precision must lie strictly between 0 and 1.")
        try:
            from emu_mps import MPSBackend, MPSConfig
        except ImportError as error:
            raise ImportError("Could not import emu-mps. Install it with 'uv add emu-mps'.") from error
        backend_config = MPSConfig(
            dt=int(dt), precision=float(mps_precision),
            max_bond_dim=int(mps_max_bond_dim), num_gpus_to_use=0,
            log_level=config.EMULATOR_LOG_LEVEL, **config_options,
        )
        results = MPSBackend(sequence, config=backend_config).run()
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
    n_qubits = len(sequence.register.qubits)
    if backend == "emu-mps":
        # MPS.sample() returns physical g=0,r=1 strings in atom order.
        # Preserve the seeded sampling convention, without materializing 2^N entries.
        import torch
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            counts = Counter(results.final_state.sample(num_shots=int(shots), one_state="r"))
        probability_map = None
    elif compact and backend == "emu-sv":
        amplitudes = results.final_state.data.detach().cpu().numpy().reshape(-1)
        if amplitudes.size != 2**n_qubits:
            raise ValueError("emu-sv returned a state vector with an unexpected size.")
        probabilities = np.abs(amplitudes) ** 2
        total = float(probabilities.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Emulator returned an invalid probability distribution.")
        probabilities = probabilities / total
        if not np.isfinite(probabilities).all():
            raise ValueError("Emulator returned an invalid probability distribution.")
        draw_ids = np.random.default_rng(seed).choice(len(probabilities), size=int(shots), p=probabilities)
        ids, freqs = np.unique(draw_ids, return_counts=True)
        counts = Counter({format(int(i), f"0{n_qubits}b"): int(c) for i,c in zip(ids, freqs)})
        probability_map = None
    else:
        if backend == "emu-sv":
            probability_map = _sv_probabilities(results.final_state, n_qubits)
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
        probability_map = dict(zip(labels, probabilities.tolist()))

    return {
        "counts": counts,
        "probabilities": probability_map,
        "results": results,
        "seed": seed,
        "backend": backend,
        "with_modulation": with_modulation,
        "dt_ns": int(dt) if backend in ("emu-sv", "emu-mps") else None,
        "gpu": gpu if backend == "emu-sv" else False,
        "mps_max_bond_dim": int(mps_max_bond_dim) if backend == "emu-mps" else None,
        "mps_precision": float(mps_precision) if backend == "emu-mps" else None,
    }
