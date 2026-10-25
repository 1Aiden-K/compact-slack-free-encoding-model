"""Bounded pulse-family search, with an injected instance-specific scorer.

The global endpoint stays fixed; amplitude is a single ramp up and down,
and the detuning sweep is monotone. This does not certify adiabatic following.
Every valid search candidate is emulated; final reporting is a separate run.
"""
import math
import time
import numpy as np
from scipy.optimize import differential_evolution
from scipy.interpolate import PchipInterpolator
from duquantum import config

class PulseFamily:
    """Bounded global controls, with an exact copy of the baseline available."""

    def __init__(self, baseline, duration_bounds=None):
        self.baseline = baseline
        p = baseline["parameters"]
        self.device = baseline["sequence"].device
        channel = self.device.channels["rydberg_global"]
        self.clock = channel.clock_period
        self.base_duration = int(p["duration_ns"])
        self.base_peak = float(p["omega_peak"])
        self.delta_initial = float(p["delta_initial"])
        self.delta_final = float(p["delta_final"])
        if not self.delta_initial < min(0, self.delta_final):
            raise ValueError("Baseline must start at negative detuning below its final detuning.")
        amplitude_limit = channel.max_amp
        self.amplitude_limit = (
            3 * self.base_peak
            if amplitude_limit is None
            else min(3 * self.base_peak, float(amplitude_limit))
        )
        self.detuning_limit = (
            2 * max(abs(self.delta_initial), abs(self.delta_final))
            if channel.max_abs_detuning is None
            else float(channel.max_abs_detuning)
        )
        lo, hi = duration_bounds or (self.base_duration / 2, 2 * self.base_duration)
        lo = max(lo, channel.min_duration, 2)
        for limit in (channel.max_duration, self.device.max_sequence_duration):
            if limit is not None:
                hi = min(hi, limit)
        lo = math.ceil(lo / self.clock) * self.clock
        hi = math.floor(hi / self.clock) * self.clock
        if not (lo <= self.base_duration <= hi):
            raise ValueError("Duration bounds must include the baseline duration and obey device limits.")
        self.initial = np.array([1, 1, 0, 0, 1], dtype=float)
        self.bounds = [
            (0.1, self.amplitude_limit / self.base_peak),
            (1.0, min(2.0, self.detuning_limit / abs(self.delta_initial))),
            (-math.log(4), math.log(4)), (-math.log(4), math.log(4)),
            (lo / self.base_duration, hi / self.base_duration),
        ]
        for value, (lower, upper) in zip(self.initial, self.bounds):
            if not lower <= value <= upper:
                raise ValueError("The original pulse lies outside the search bounds.")

    def canonical(self, values):
        x = np.array(values, dtype=float, copy=True)
        if x.shape != (5,) or not np.isfinite(x).all():
            raise ValueError("Expected five finite pulse parameters.")
        bounds = np.asarray(self.bounds)
        if np.any(x < bounds[:, 0] - 1e-12) or np.any(x > bounds[:, 1] + 1e-12):
            raise ValueError("Pulse parameters are outside the search bounds.")
        duration = int(round(x[4] * self.base_duration / self.clock)) * self.clock
        x[4] = duration / self.base_duration
        return x

    def describe(self, values):
        x = self.canonical(values)
        return {
            "omega_peak_rad_us": self.base_peak * x[0],
            "initial_detuning_rad_us": self.delta_initial * x[1],
            "final_detuning_rad_us": self.delta_final,
            "sweep_segment_fractions": self.segment_fractions(x).tolist(),
            "duration_ns": int(round(x[4] * self.base_duration)),
        }

    @staticmethod
    def segment_fractions(x):
        weights = np.exp([x[2], x[3], 0.0])
        return weights / weights.sum()

    def waveforms(self, values):
        x = self.canonical(values)
        duration = int(round(x[4] * self.base_duration))
        u = np.linspace(0.0, 1.0, duration)
        times = np.r_[0.0, np.cumsum(self.segment_fractions(x))]
        times[-1] = 1.0
        progress = PchipInterpolator(times, [0.0, 1/3, 2/3, 1.0])(u)
        amplitude = self.base_peak * x[0] * np.sin(np.pi * progress) ** 2
        amplitude[[0, -1]] = 0.0
        initial = self.delta_initial * x[1]
        detuning = initial + (self.delta_final - initial) * progress
        detuning[[0, -1]] = (initial, self.delta_final)
        if amplitude.max() > self.amplitude_limit * (1 + 1e-12):
            raise ValueError("Rabi amplitude exceeds the channel maximum.")
        if np.abs(detuning).max() > self.detuning_limit * (1 + 1e-12):
            raise ValueError("Detuning exceeds the channel maximum.")
        return amplitude, detuning

    def build(self, values):
        from pulser import CustomWaveform, Pulse, Sequence

        amplitude, detuning = self.waveforms(values)
        pulse = Pulse(CustomWaveform(amplitude), CustomWaveform(detuning),
                      self.baseline["parameters"]["phase"])
        sequence = Sequence(self.baseline["embedding"]["register"], self.device)
        sequence.declare_channel("global", "rydberg_global")
        sequence.add(pulse, "global")
        sequence.measure("ground-rydberg")
        total_duration = sequence.get_duration(include_fall_time=True)
        limit = self.device.max_sequence_duration
        if limit is not None and total_duration > limit:
            raise ValueError("Sequence with modulation fall time exceeds the device duration limit.")
        parameters = dict(self.baseline["parameters"])
        ideal = np.array(parameters["interaction_matrix"], copy=True) / 2
        np.fill_diagonal(ideal, -detuning[-1])
        parameters.update(
            duration_ns=len(amplitude), omega_peak=float(amplitude.max()),
            delta_initial=float(detuning[0]), delta_final=float(detuning[-1]),
            rabi_samples=amplitude, detuning_samples=detuning,
            global_q_matrix=ideal,
            diagonal_error=np.diag(ideal) - parameters["energy_scale"] * np.diag(parameters["q_matrix"]),
            exact_global_diagonal=bool(np.allclose(
                parameters["target_detunings"], detuning[-1], rtol=config.MATRIX_REL_TOL, atol=0
            )),
            ground_state_check=None,
        )
        realized = np.array(self.baseline["embedding"]["realized_interactions"], copy=True) / 2
        np.fill_diagonal(realized, -detuning[-1])
        target = parameters["energy_scale"] * parameters["q_matrix"]
        diagnostics = dict(self.baseline["diagnostics"])
        diagnostics.update(
            realized_q_matrix=realized,
            relative_matrix_error=float(np.linalg.norm(realized - target) / np.linalg.norm(target)),
            duration_with_fall_ns=total_duration,
            target_vs_realized=None, global_ideal_vs_realized=None,
        )
        return {"sequence": sequence, "pulse": pulse, "parameters": parameters,
                "embedding": self.baseline["embedding"], "diagnostics": diagnostics}


class _BudgetExhausted(Exception):
    pass


def bounded_search(initial, bounds, evaluate, *, evaluations, seed=0, progress=None):
    """Differential evolution with a strict call budget and baseline incumbent.

    evaluate(x) returns (loss, payload). Validation failures may return a loss
    worse than every valid candidate. Emulator errors should propagate.
    """
    if isinstance(evaluations, bool) or not isinstance(evaluations, (int, np.integer)) or evaluations < 1:
        raise ValueError("evaluations must be a positive integer.")
    rng = np.random.default_rng(seed)
    initial = np.asarray(initial, dtype=float)
    limits = np.asarray(bounds, dtype=float)
    if initial.ndim != 1 or not np.isfinite(initial).all():
        raise ValueError("Initial search parameters must be a finite vector.")
    if limits.shape != (len(initial), 2) or not np.isfinite(limits).all():
        raise ValueError("Search bounds must be finite pairs matching initial parameters.")
    if np.any(limits[:, 0] > limits[:, 1]) or np.any(initial < limits[:, 0]) or np.any(initial > limits[:, 1]):
        raise ValueError("Invalid bounds or baseline outside bounds.")
    cache = {}
    history = []
    best = None

    def objective(x):
        nonlocal best
        key = tuple(float(v) for v in x)
        if key in cache:
            return cache[key]["loss"]
        if len(history) >= evaluations:
            raise _BudgetExhausted
        loss, payload = evaluate(np.asarray(x, dtype=float))
        loss = float(loss)
        if not np.isfinite(loss):
            raise ValueError("Objective returned a nonfinite loss.")
        entry = {"evaluation": len(history) + 1, "loss": loss, "parameters": list(key),
                 "payload": payload}
        cache[key] = entry
        history.append(entry)
        def rank(item):
            extra = item["payload"].get("tie_break", ()) if isinstance(item["payload"], dict) else ()
            return (item["loss"], *extra)

        if best is None or rank(entry) < rank(best):
            best = entry
        if progress is not None:
            progress(entry, best)
        return loss

    objective(initial)
    status = "baseline_only"
    if evaluations > 1:
        population_size = min(12, max(5, evaluations - 1))
        population = np.empty((population_size, len(initial)))
        population[0] = initial
        width = limits[:, 1] - limits[:, 0]
        # Mix local perturbations with broad exploration; all randomness is seeded.
        for i in range(1, population_size):
            proposal = (
                initial + rng.normal(0, 0.18, len(initial)) * width
                if i < (population_size + 1) // 2
                else rng.uniform(limits[:, 0], limits[:, 1])
            )
            population[i] = np.clip(proposal, limits[:, 0], limits[:, 1])
        try:
            result = differential_evolution(
                objective, bounds, init=population, seed=seed,
                maxiter=evaluations, polish=False, updating="immediate",
                tol=0.0, atol=0.0,
            )
            status = "converged" if result.success else "iteration_limit"
        except _BudgetExhausted:
            status = "evaluation_budget"
    return {"best": best, "baseline": history[0], "history": history, "status": status}


def optimize_pulses(
    baseline, scorer, *, evaluations=40, search_seed=0, sample_seed=0,
    training_shots=1000, score_mode="sampled", metric="reference-probability", backend=config.EMULATOR,
    dt=1, gpu=False, with_modulation=config.MODULATION, duration_bounds=None,
    progress=None, emulator=None,
):
    """Optimize one instance using its declared reference-based shot metric.

    emulator can inject a compatible test runner; otherwise use emulate.emulate.
    Each successful objective call makes exactly one state-evolution call.
    """
    if score_mode not in ("probabilities", "sampled"):
        raise ValueError("score_mode must be 'probabilities' or 'sampled'.")
    if metric not in ("reference-probability", "best-gap", "feasibility"):
        raise ValueError("Unknown optimization metric.")
    if metric == "best-gap" and score_mode != "sampled":
        raise ValueError("best-gap requires sampled mode and a finite training-shot budget.")
    if isinstance(training_shots, bool) or not isinstance(training_shots, (int, np.integer)) or training_shots < 1:
        raise ValueError("training_shots must be a positive integer.")
    if len(baseline["embedding"]["register"].qubits) != scorer.n:
        raise ValueError("The register and clustering scorer must describe the same customer count.")
    if emulator is None:
        from duquantum.quantum.emulator import emulate
        emulator = emulate
    family = PulseFamily(baseline, duration_bounds)
    state_cache = {}

    def evaluate(x):
        controls = family.describe(x)
        key = tuple(family.canonical(x))
        if key in state_cache:
            return state_cache[key]
        is_original = np.array_equal(family.canonical(x), family.initial)
        try:
            program = baseline if is_original else family.build(x)
        except ValueError as error:
            # Invalid candidates rank worse than every valid metric loss (<=1).
            return 2.0, {"controls": controls, "error": str(error), "program": None}
        output = emulator(
            program["sequence"], shots=training_shots, seed=sample_seed,
            backend=backend, dt=dt, gpu=gpu, with_modulation=with_modulation,
        )
        metrics = scorer.metrics(output["counts"])
        predicted_metrics = scorer.metrics(output["probabilities"], include_best=False)
        if score_mode == "probabilities":
            metrics.update(predicted_metrics)
        loss, tie_break = scorer.score(metrics, metric)
        result = (loss, {"controls": controls, "program": program,
                         "probabilities": dict(output["probabilities"]),
                         "metrics": metrics, "predicted_metrics": predicted_metrics,
                         "tie_break": tie_break, "error": None})
        state_cache[key] = result
        return result

    start = time.perf_counter()
    search = bounded_search(family.initial, family.bounds, evaluate,
                            evaluations=evaluations, seed=search_seed, progress=progress)
    search["seconds"] = time.perf_counter() - start
    search["bounds"] = [list(pair) for pair in family.bounds]
    search["simulations"] = len(state_cache)
    search["score_mode"] = score_mode
    search["metric"] = metric
    return search


def resample(probabilities, shots, seed):
    """Fresh final measurements; training shots are never reused or pooled."""
    labels = sorted(probabilities)
    p = np.asarray([probabilities[label] for label in labels], dtype=float)
    if not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1.0, rtol=0, atol=1e-8):
        raise ValueError("Invalid final-state probabilities.")
    if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)) or shots < 1:
        raise ValueError("shots must be a positive integer.")
    draws = np.random.default_rng(seed).multinomial(int(shots), p / p.sum())
    return {label: int(count) for label, count in zip(labels, draws) if count}
