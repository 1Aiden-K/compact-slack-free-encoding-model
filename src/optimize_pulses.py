"""Optimize adiabatic-style global pulses using challenge good-shot metrics.

Place beside config.py, qubo.py, ising.py, embedding.py, pulses.py, emulate.py,
evaluator.py, and (for --instance) instance_generator.py.
Uses the existing NumPy/SciPy/Pulser/emulator dependencies; no new optimizer
package is needed. Importing this module does not run a solver or emulator.

Examples:
    uv run optimize_pulses.py --evaluations 40 --output optimized_run.json
    uv run optimize_pulses.py --instance instances/tier_s_seed_0001.json \
        --evaluations 80 --dt 1 --output optimized_seed_0001.json
    uv run optimize_pulses.py --metric best-gap --training-shots 1000 \
        --search-seed 0 --sample-seed 0 --final-seed 10000
    uv run evaluator.py optimized_run.json

Five search variables: peak amplitude, initial negative-detuning magnitude,
two parameters allocating time across three equal portions of the detuning
sweep, and duration. A monotone C1 cubic interpolator warps sweep progress.
Amplitude is a single smooth sin^2 ramp up and down along that progress.
The final detuning is FIXED to the baseline embedding's endpoint. Initial
detuning is at least as negative as the baseline. Geometry, lambda, and
interaction scale stay fixed; duration is rounded to the device clock.
This preserves the adiabatic recipe but does not certify adiabatic following.

Scoring uses Track A capacity-constrained clustering, not routing.
--metric reference-probability (default) maximizes the fraction of ALL shots
reaching the constrained clustering reference
within --epsilon percent; ties use best feasible gap, then feasibility.
--metric best-gap minimizes the best feasible shot's gap, breaking ties by
reference probability then feasibility. --metric feasibility maximizes
feasibility, then uses best gap and reference probability. Every candidate
records all three metrics. No feasible shot gives an undefined best gap and
ranks worse than any finite gap. The original pulse remains the incumbent
unless another pulse wins this explicit ordering.

Default --score-mode sampled uses a fixed training-shot budget. Optional
--score-mode probabilities uses final-state reference/feasibility probabilities
for emulator development; best-gap is still measured on the finite training
batch. The best-gap primary metric requires sampled mode. Exact clustering reference
COST is used explicitly for challenge scoring, never to set pulse parameters
or supply a bitstring to the search. No depot or route solver is required.
Final runs re-emulate each reported sequence with --final-seed; training
distributions and shots are never substituted for the final run. This is
per-instance optimization; include its full search budget in comparisons.

--evaluations includes the original pulse and rejected candidate attempts.
Results contain both pulses, final distributions, objective values, controls,
search history, current endpoint diagnostics, and a replayable Pulser sequence.
The output is also compatible with evaluator.py; keep the matching config.py
when re-evaluating a saved --instance run. Existing output files are protected.
"""

import argparse
import json
import math
import time
from pathlib import Path
from itertools import product

import numpy as np
from scipy.optimize import differential_evolution
from scipy.interpolate import PchipInterpolator

import config


class ClusteringScorer:
    """Track A: within-cluster pair distance, subject to vehicle capacities."""

    def __init__(self, coordinates, demands, capacity, *, epsilon_pct=0.0):
        coordinates = np.asarray(coordinates, dtype=float)
        if (coordinates.ndim != 2 or coordinates.shape[1] == 0
                or not np.isfinite(coordinates).all()):
            raise ValueError("Coordinates must be a finite matrix with one row per customer.")
        self.demands = np.asarray(demands, dtype=float)
        self.capacity = float(capacity)
        self.n = len(coordinates)
        if (
            self.n < 2
            or self.demands.shape != (self.n,)
            or not np.isfinite(self.demands).all()
            or np.any(self.demands < 0)
            or not np.isfinite(self.capacity)
            or self.capacity < 0
        ):
            raise ValueError("Need at least two customers and finite nonnegative demands/capacity.")
        if self.n > config.MAX_EXACT_BITS:
            raise ValueError(f"Exact clustering reference exceeds {config.MAX_EXACT_BITS} bits.")
        if not np.isfinite(epsilon_pct) or epsilon_pct < 0:
            raise ValueError("epsilon_pct must be finite and nonnegative.")
        self.epsilon_pct = float(epsilon_pct)
        self.pair_distances = np.triu(np.linalg.norm(
            coordinates[:, None, :] - coordinates[None, :, :], axis=2), 1)
        # Retain the exact constrained reference cost, not its bitstrings.
        optimum = None
        for bits in product((0, 1), repeat=self.n):
            z = np.asarray(bits)
            if self._feasible(z):
                cost = self._cost(z)
                optimum = cost if optimum is None else min(optimum, cost)
        if optimum is None:
            raise ValueError("No assignment satisfies the two vehicle capacities.")
        self.reference_cost = float(optimum)
        self.cache = {}

    def _feasible(self, z):
        return max(float(self.demands @ z), float(self.demands @ (1 - z))) <= self.capacity

    def _cost(self, z):
        return float(np.sum(self.pair_distances * (z[:, None] == z[None, :])))

    def assignment(self, bits):
        if not isinstance(bits, str) or len(bits) != self.n or set(bits) - {"0", "1"}:
            raise ValueError(f"Expected a {self.n}-bit assignment string.")
        if bits not in self.cache:
            z = np.fromiter((int(bit) for bit in bits), dtype=int)
            feasible = self._feasible(z)
            cost = self._cost(z) if feasible else None
            if not feasible:
                gap = None
            elif math.isclose(cost, self.reference_cost, rel_tol=config.COST_REL_TOL,
                              abs_tol=config.COST_ABS_TOL):
                gap = 0.0
            elif self.reference_cost == 0:
                gap = None
            else:
                gap = 100 * (cost - self.reference_cost) / self.reference_cost
            self.cache[bits] = (feasible, cost, gap)
        return self.cache[bits]

    def metrics(self, weights, *, include_best=True):
        """Rates use all weights; best gap is only meaningful for shot counts."""
        if not weights:
            raise ValueError("Cannot score an empty distribution.")
        total = 0.0
        feasible_weight = 0.0
        reached_weight = 0.0
        best_gap = None
        for bits, weight in weights.items():
            weight = float(weight)
            if not np.isfinite(weight) or weight < 0:
                raise ValueError("Weights must be finite and nonnegative.")
            feasible, _, gap = self.assignment(bits)
            if feasible:
                feasible_weight += weight
                if gap is not None and gap <= self.epsilon_pct:
                    reached_weight += weight
                if include_best and weight > 0 and gap is not None:
                    best_gap = gap if best_gap is None else min(best_gap, gap)
            total += weight
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Distribution must have positive finite weight.")
        result = {"reference_probability": reached_weight / total,
                  "feasibility_rate": feasible_weight / total}
        if include_best:
            result["best_gap_pct"] = best_gap
        return result

    def score(self, metrics, metric="reference-probability"):
        """Return scalar primary loss and a lexicographic tie-break key."""
        gap = metrics["best_gap_pct"]
        gap_rank = float("inf") if gap is None else gap
        p_ref, p_feasible = metrics["reference_probability"], metrics["feasibility_rate"]
        if metric == "reference-probability":
            return 1 - p_ref, (gap_rank, -p_feasible)
        if metric == "best-gap":
            # Strictly increasing bounded transform; no feasible shot scores 1.
            loss = 1.0 if gap is None else gap / (100 + gap)
            return loss, (-p_ref, -p_feasible)
        if metric == "feasibility":
            return 1 - p_feasible, (gap_rank, -p_ref)
        raise ValueError("Unknown optimization metric.")


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
        from emulate import emulate
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


def add_optimization_arguments(parser):
    """Shared controls for the integrated benchmark and emulator paths."""
    parser.add_argument("--optimize-pulses", action=argparse.BooleanOptionalAction,
                        default=True, help="optimize before final emulation (default: enabled)")
    parser.add_argument("--evaluations", type=int, default=40,
                        help="search attempts including baseline, per instance")
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--training-seed", type=int, default=10000,
                        help="training measurement seed; must differ from --sample-seed")
    parser.add_argument("--training-shots", type=int, default=1000)
    parser.add_argument("--score-mode", choices=("sampled", "probabilities"), default="sampled")
    parser.add_argument("--metric", choices=("reference-probability", "best-gap", "feasibility"),
                        default="reference-probability")
    parser.add_argument("--duration-bounds", type=int, nargs=2, metavar=("MIN_NS", "MAX_NS"))


def validate_optimization_settings(settings):
    if not settings["optimize_pulses"]:
        return
    for name in ("evaluations", "training_shots"):
        if settings[name] < 1:
            raise ValueError(f"{name} must be positive.")
    if settings["search_seed"] < 0 or settings["training_seed"] < 0:
        raise ValueError("Optimization seeds must be nonnegative.")
    if settings["training_seed"] == settings["sample_seed"]:
        raise ValueError("--training-seed must differ from --sample-seed.")
    if settings["metric"] == "best-gap" and settings["score_mode"] != "sampled":
        raise ValueError("--metric best-gap requires --score-mode sampled.")
    bounds = settings["duration_bounds"]
    if bounds is not None and not (0 < bounds[0] <= bounds[1]):
        raise ValueError("--duration-bounds must be positive and ordered.")


def optimize_program(baseline, *, coordinates, demands, capacity, settings,
                     emulator=None):
    """Select an emulated pulse; the caller must emulate it for final reporting.

    Returns the actual selected Pulser program and JSON-safe search accounting.
    Training distributions and counts never substitute for the final run.
    """
    validate_optimization_settings(settings)
    scorer = ClusteringScorer(coordinates, demands, capacity,
                         epsilon_pct=settings["epsilon_pct"])

    def progress(entry, best):
        print(f"Pulse search {entry['evaluation']}/{settings['evaluations']}: "
              f"loss={entry['loss']:.6f}; best={best['loss']:.6f}", flush=True)

    search = optimize_pulses(
        baseline, scorer, evaluations=settings["evaluations"],
        search_seed=settings["search_seed"], sample_seed=settings["training_seed"],
        training_shots=settings["training_shots"], score_mode=settings["score_mode"],
        metric=settings["metric"], backend=settings["backend"], dt=settings["dt_ns"],
        gpu=settings["gpu"], with_modulation=settings["with_modulation"],
        duration_bounds=settings["duration_bounds"], progress=progress, emulator=emulator,
    )
    best = search["best"]
    program = best["payload"]["program"]
    history = [{"evaluation": entry["evaluation"], "loss": entry["loss"],
                "parameters": entry["parameters"], "controls": entry["payload"]["controls"],
                "metrics": entry["payload"].get("metrics"),
                "predicted_metrics": entry["payload"].get("predicted_metrics"),
                "error": entry["payload"].get("error")}
               for entry in search["history"]]
    metadata = {
        "track": "A", "objective": "capacity_constrained_clustering",
        "status": search["status"], "budget": settings["evaluations"],
        "evaluations": len(history), "simulations": search["simulations"],
        "training_shots": search["simulations"] * settings["training_shots"],
        "seconds": search["seconds"], "metric": settings["metric"],
        "score_mode": settings["score_mode"], "reference_cost": scorer.reference_cost,
        "uses_reference_cost": True, "selected_evaluation": best["evaluation"],
        "baseline_retained": best is search["baseline"],
        "baseline_metrics": search["baseline"]["payload"]["metrics"],
        "selected_metrics": best["payload"]["metrics"],
        "fixed_final_detuning_rad_us": baseline["parameters"]["delta_final"],
        "bounds": search["bounds"], "history": history,
        "selected_sequence": json.loads(program["sequence"].to_abstract_repr()),
    }
    return program, metadata


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


def _load_cli_instance(path, depot):
    """Bind before importing modules which snapshot config's instance arrays."""
    if path is not None:
        from instance_generator import load_instance
        instance = load_instance(path)
        config.COORDINATES = np.asarray(instance["coordinates"], dtype=float)
        config.DEMANDS = np.asarray(instance["demands"], dtype=float)
        config.CAPACITY = float(instance["capacity"])
        config.DEPOT = instance["depot"]
    if depot is not None:
        config.DEPOT = depot


def main(argv=None):
    # Kept local so --help and scorer tests do not require Pulser or ising.py.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instance", type=Path, help="optional certified Tier S JSON; config.py is not rewritten")
    parser.add_argument("--lambda", dest="penalty_weight", type=float, default=config.LAMBDA)
    parser.add_argument("--scale", type=float, default=config.ENERGY_SCALE)
    parser.add_argument("--duration", type=int, default=config.PULSE_DURATION, help="baseline duration, ns")
    parser.add_argument("--duration-bounds", type=int, nargs=2, metavar=("MIN_NS", "MAX_NS"),
                        help="must include baseline; default: half to twice baseline")
    parser.add_argument("--omega-peak", type=float, default=config.OMEGA_PEAK)
    parser.add_argument("--delta-initial", type=float, default=config.DELTA_INITIAL)
    parser.add_argument("--seed", type=int, default=config.EMBEDDING_SEED, help="embedding seed")
    parser.add_argument("--restarts", type=int, default=config.EMBEDDING_RESTARTS)
    parser.add_argument("--maxiter", type=int, default=config.MAX_ITERATIONS, help="embedding iterations per restart")
    parser.add_argument("--evaluations", type=int, default=40, help="total search evaluations INCLUDING baseline")
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=config.MEASUREMENT_SEED, help="training measurements")
    parser.add_argument("--final-seed", type=int, default=10000, help="fresh final measurements")
    parser.add_argument("--score-mode", choices=("probabilities", "sampled"), default="sampled")
    parser.add_argument("--metric", choices=("reference-probability", "best-gap", "feasibility"),
                        default="reference-probability", help="primary metric; others break ties")
    parser.add_argument("--training-shots", type=int, default=1000)
    parser.add_argument("--shots", type=int, default=config.SHOTS, help="final shots PER pulse")
    parser.add_argument("--backend", choices=("emu-sv", "qutip"), default=config.EMULATOR)
    parser.add_argument("--dt", type=int, default=1, help="emu-sv integration step, ns")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--with-modulation", action="store_true", default=config.MODULATION)
    parser.add_argument("--depot", type=float, nargs=2)
    parser.add_argument("--epsilon", type=float, default=config.EPSILON_PCT)
    parser.add_argument("--output", type=Path, default=Path("optimized_run.json"))
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError(f"Output already exists: {args.output}; choose a new path.")
        for name in ("evaluations", "training_shots", "shots", "dt", "duration", "restarts", "maxiter"):
            if getattr(args, name) < 1:
                raise ValueError(f"--{name.replace('_', '-')} must be positive.")
        for name in ("seed", "search_seed", "sample_seed", "final_seed"):
            if getattr(args, name) < 0:
                raise ValueError("All seeds must be nonnegative.")
        if args.final_seed == args.sample_seed:
            raise ValueError("--final-seed must differ from --sample-seed.")
        if not np.isfinite(args.epsilon) or args.epsilon < 0:
            raise ValueError("--epsilon must be finite and nonnegative.")
        if args.gpu and args.backend != "emu-sv":
            raise ValueError("--gpu requires --backend emu-sv.")
        if args.metric == "best-gap" and args.score_mode != "sampled":
            raise ValueError("--metric best-gap requires --score-mode sampled.")
        _load_cli_instance(args.instance, args.depot)
        from pulses import build_sequence
        from qubo import build_q_matrix

        scorer = ClusteringScorer(config.COORDINATES, config.DEMANDS, config.CAPACITY,
                             epsilon_pct=args.epsilon)
        print("Building the fixed register and original pulse...", flush=True)
        baseline = build_sequence(
            build_q_matrix(args.penalty_weight), energy_scale=args.scale,
            duration_ns=args.duration, omega_peak=args.omega_peak,
            delta_initial=args.delta_initial, seed=args.seed,
            restarts=args.restarts, maxiter=args.maxiter,
        )

        def progress(entry, best):
            error = entry["payload"].get("error")
            suffix = f"; rejected: {error}" if error else ""
            print(f"[{entry['evaluation']}/{args.evaluations}] loss={entry['loss']:.6f}; "
                  f"best={best['loss']:.6f}; metrics={entry['payload'].get('metrics')}{suffix}", flush=True)

        search = optimize_pulses(
            baseline, scorer, evaluations=args.evaluations, search_seed=args.search_seed,
            sample_seed=args.sample_seed, training_shots=args.training_shots,
            score_mode=args.score_mode, metric=args.metric, backend=args.backend, dt=args.dt, gpu=args.gpu,
            with_modulation=args.with_modulation, duration_bounds=args.duration_bounds,
            progress=progress,
        )
        # Full reference bitstrings/diagnostics are for reporting. During search
        # ClusteringScorer uses only its reference cost and sampled clustering costs.
        from evaluator import build_reference, encoding_from_program, evaluate_samples

        reference = build_reference(args.penalty_weight, depot=config.DEPOT)

        def run_record(entry):
            data = entry["payload"]
            program = data["program"]
            encoding = encoding_from_program(program)
            from emulate import emulate
            final = emulate(
                program["sequence"], shots=args.shots, seed=args.final_seed,
                backend=args.backend, dt=args.dt, gpu=args.gpu,
                with_modulation=args.with_modulation,
            )
            counts = dict(final["counts"])
            evaluation = evaluate_samples(
                counts, probabilities=final["probabilities"], penalty_weight=args.penalty_weight,
                reference=reference, encoding=encoding, epsilon_pct=args.epsilon,
            )
            return {
                "controls": data["controls"], "training_loss": entry["loss"],
                "training_metrics": data["metrics"],
                "predicted_metrics": data["predicted_metrics"],
                "final_metrics": scorer.metrics(counts),
                "counts": counts, "probabilities": final["probabilities"],
                "encoding": encoding, "evaluation": evaluation,
            }

        original = run_record(search["baseline"])
        optimized = run_record(search["best"])
        selected_program = search["best"]["payload"]["program"]
        history = [{"evaluation": entry["evaluation"], "loss": entry["loss"],
                    "parameters": entry["parameters"], "controls": entry["payload"]["controls"],
                    "metrics": entry["payload"].get("metrics"),
                    "error": entry["payload"].get("error")}
                   for entry in search["history"]]
        payload = {
            "schema_version": 3, "method": "adiabatic_good_shot_pulse_optimization",
            "track": "A", "objective": "capacity_constrained_clustering",
            "counts": optimized["counts"], "probabilities": optimized["probabilities"],
            "encoding": optimized["encoding"], "evaluation": optimized["evaluation"],
            "instance": {"coordinates": np.asarray(config.COORDINATES).tolist(),
                         "demands": np.asarray(config.DEMANDS).tolist(),
                         "capacity": float(config.CAPACITY), "depot": np.asarray(config.DEPOT).tolist()},
            "settings": {"penalty_weight": args.penalty_weight,
                         "energy_scale": baseline["parameters"]["energy_scale"],
                         "backend": args.backend, "dt_ns": args.dt if args.backend == "emu-sv" else None,
                         "with_modulation": args.with_modulation, "gpu": args.gpu,
                         "embedding_seed": args.seed, "embedding_restarts": args.restarts,
                         "embedding_maxiter": args.maxiter, "search_seed": args.search_seed,
                         "sample_seed": args.sample_seed, "final_seed": args.final_seed,
                         "shots": args.shots, "training_shots": args.training_shots},
            "optimization": {"algorithm": "scipy.optimize.differential_evolution",
                             "track": "A", "objective": "capacity_constrained_clustering",
                             "status": search["status"], "budget": args.evaluations,
                             "evaluations": len(history), "simulations": search["simulations"],
                             "training_shot_budget": search["simulations"] * args.training_shots,
                             "final_simulations": 2,
                             "total_emulator_runs": search["simulations"] + 2,
                             "seconds": search["seconds"], "score_mode": args.score_mode,
                             "metric": args.metric, "reference_cost": scorer.reference_cost,
                             "epsilon_pct": args.epsilon,
                             "fixed_final_detuning_rad_us": baseline["parameters"]["delta_final"],
                             "adiabatic_following_certified": False,
                             "bounds": search["bounds"], "history": history,
                             "parameter_order": ["amplitude_multiplier", "initial_detuning_multiplier",
                                                 "log_segment_1_weight", "log_segment_2_weight",
                                                 "duration_ratio"]},
            "baseline": original, "optimized": optimized,
            "positions_um": np.asarray(baseline["embedding"]["positions_um"]).tolist(),
            "optimized_sequence": json.loads(selected_program["sequence"].to_abstract_repr()),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive create also protects against a file appearing during the search.
        text = json.dumps(payload, indent=config.JSON_INDENT, allow_nan=False) + "\n"
        with args.output.open("x") as stream:
            stream.write(text)
        print("\nBaseline -> selected pulse:")
        print(f"Primary metric: {args.metric}; training loss "
              f"{original['training_loss']:.6f} -> {optimized['training_loss']:.6f}")
        for label, run in (("Baseline", original), ("Selected", optimized)):
            predicted = run["evaluation"]["predicted"]
            print(f"{label}: feasible={predicted['feasibility_rate']:.2%}; "
                  f"clustering-optimal={predicted['optimal_probability']:.2%}; "
                  f"best sampled clustering gap={run['evaluation']['sampled']['best_model_gap_pct']}%")
        print(f"Search used {len(history)} evaluations, {search['simulations']} simulations; saved {args.output}")
    except (ValueError, ImportError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
