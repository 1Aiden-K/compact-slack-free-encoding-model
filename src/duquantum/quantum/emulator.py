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
