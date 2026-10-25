"""Run with: python -m unittest test_optimize_pulses -v

Requires NumPy/SciPy and the existing config.py. Tests use a small
Pulser interface double, and independently integrate the Schrodinger equation
for the optimization integration test. Real Pulser/emulator integration must
also be checked in the project's installed environment.
"""

import json
import math
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.integrate import solve_ivp

from optimize_pulses import (
    PulseFamily, ClusteringScorer, bounded_search, main, optimize_pulses, resample,
)
import config


class Waveform:
    def __init__(self, samples):
        self.samples = np.asarray(samples)


class Pulse:
    def __init__(self, amplitude, detuning, phase):
        self.amplitude = amplitude
        self.detuning = detuning
        self.phase = phase


class Sequence:
    def __init__(self, register, device):
        self.register = register
        self.device = device

    def declare_channel(self, name, channel):
        pass

    def add(self, pulse, name):
        self.pulse = pulse

    def measure(self, basis):
        pass

    def get_duration(self, include_fall_time=False):
        return len(self.pulse.amplitude.samples) + (self.device.fall_time if include_fall_time else 0)

    def to_abstract_repr(self):
        return json.dumps({"test_double": True})


PULSER = SimpleNamespace(CustomWaveform=Waveform, Pulse=Pulse, Sequence=Sequence)


def baseline_program():
    channel = SimpleNamespace(clock_period=4, min_duration=16, max_duration=8000,
                              max_amp=8.0, max_abs_detuning=10.0)
    device = SimpleNamespace(channels={"rydberg_global": channel},
                             max_sequence_duration=9000, fall_time=0)
    register = SimpleNamespace(qubits={"q0": np.array([-2., 0.]), "q1": np.array([2., 0.])})
    duration = 1000
    u = np.linspace(0, 1, duration)
    amplitude = np.sin(np.pi * u) ** 2
    amplitude[[0, -1]] = 0
    detuning = -4 + 6 * u
    interactions = np.array([[0., 8.], [8., 0.]])
    target = np.array([[-2., 4.], [4., -2.]])
    sequence = Sequence(register, device)
    pulse = Pulse(Waveform(amplitude), Waveform(detuning), 0)
    sequence.add(pulse, "global")
    return {
        "sequence": sequence, "pulse": pulse,
        "parameters": {"duration_ns": duration, "omega_peak": 1.,
                       "delta_initial": -4., "delta_final": 2., "phase": 0.,
                       "rabi_samples": amplitude, "detuning_samples": detuning,
                       "interaction_matrix": interactions, "q_matrix": target,
                       "global_q_matrix": target.copy(), "energy_scale": 1.,
                       "target_detunings": np.array([2., 2.]), "n_qubits": 2},
        "embedding": {"register": register, "realized_interactions": interactions,
                      "positions_um": np.array([[-2., 0.], [2., 0.]]),
                      "relative_interaction_error": 0.},
        "diagnostics": {"relative_interaction_error": 0., "relative_matrix_error": 0.,
                        "realized_q_matrix": target.copy()},
    }


def schrodinger_emulator(sequence, *, shots, seed, **kwargs):
    """Independent physical oracle for two atoms; g=0, r=1 in register order."""
    amplitude = sequence.pulse.amplitude.samples
    detuning = sequence.pulse.detuning.samples
    t = np.arange(len(amplitude)) / 1000  # ns -> us
    x = np.array([[0, 1], [1, 0]])
    drive = (np.kron(x, np.eye(2)) + np.kron(np.eye(2), x)) / 2
    occupation = np.array([0, 1, 1, 2])
    pair_energy = np.array([0, 0, 0, 8.])

    def rhs(time, state):
        omega = np.interp(time, t, amplitude)
        delta = np.interp(time, t, detuning)
        return -1j * (omega * (drive @ state) + (pair_energy - delta * occupation) * state)

    solution = solve_ivp(rhs, (t[0], t[-1]), np.array([1, 0, 0, 0], dtype=complex),
                         rtol=2e-7, atol=2e-9)
    if not solution.success:
        raise RuntimeError(solution.message)
    probabilities = np.abs(solution.y[:, -1]) ** 2
    probabilities /= probabilities.sum()
    probabilities = {format(i, "02b"): float(p) for i, p in enumerate(probabilities)}
    return {"probabilities": probabilities, "counts": resample(probabilities, shots, seed)}


class ScoringTests(unittest.TestCase):
    def test_scores_clustering_instead_of_total_route_length(self):
        scorer = ClusteringScorer([[1, 0], [2, 0], [8, 0], [9, 0]], [1, 1, 1, 1], 4)
        self.assertEqual(scorer.reference_cost, 2.)
        self.assertEqual(scorer.assignment("0011"), (True, 2., 0.))
        # Serving all customers in one tour is shorter than two tours here,
        # but its within-cluster pair distance is much worse for Track A.
        self.assertEqual(scorer.assignment("0000"), (True, 30., 1400.))
        self.assertEqual(scorer.metrics({"0011": 7, "0000": 3}),
                         {"reference_probability": .7, "feasibility_rate": 1., "best_gap_pct": 0.})

    def test_exact_reference_limit_and_invalid_coordinates(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            ClusteringScorer(np.zeros((config.MAX_EXACT_BITS + 1, 2)),
                             np.ones(config.MAX_EXACT_BITS + 1), 100)
        for coordinates in ([1, 2], [[float("nan"), 0], [0, 1]]):
            with self.assertRaises(ValueError):
                ClusteringScorer(coordinates, [1, 1], 1)

    def test_reference_probability_best_gap_and_feasibility(self):
        scorer = ClusteringScorer([[1, 0], [0, 1]], [1, 1], 1)
        self.assertEqual(scorer.assignment("01"), (True, 0., 0.))
        self.assertEqual(scorer.assignment("00"), (False, None, None))
        self.assertEqual(scorer.metrics({"01": 3, "00": 1}),
                         {"reference_probability": .75, "feasibility_rate": .75, "best_gap_pct": 0.})

    def test_coincident_customers_and_zero_clustering_cost(self):
        scorer = ClusteringScorer([[0, 0], [0, 0]], [1, 1], 1)
        good = scorer.metrics({"01": 1})
        bad = scorer.metrics({"00": 1})
        self.assertEqual(scorer.score(good)[0], 0.)
        self.assertEqual(scorer.score(bad)[0], 1.)
        self.assertEqual(good["best_gap_pct"], 0.)
        self.assertIsNone(bad["best_gap_pct"])

    def test_rare_reference_shot_beats_consistently_suboptimal_shots(self):
        from itertools import product
        scorer = ClusteringScorer([[2, 1], [3, 8], [7, 4], [1, 2]], [1, 2, 3, 4], 7)
        optimal = None
        suboptimal = None
        for bits in product("01", repeat=4):
            bits = "".join(bits)
            feasible, _, gap = scorer.assignment(bits)
            if feasible and gap == 0:
                optimal = bits
            if feasible and gap > 0:
                suboptimal = bits
        self.assertIsNotNone(optimal)
        self.assertIsNotNone(suboptimal)
        rare_good = scorer.metrics({optimal: 1, suboptimal: 99})
        no_good = scorer.metrics({suboptimal: 100})
        for metric in ("reference-probability", "best-gap"):
            self.assertLess(scorer.score(rare_good, metric)[0], scorer.score(no_good, metric)[0])
        self.assertEqual(rare_good["best_gap_pct"], 0.)

    def test_no_feasible_shots_rank_worse_than_any_finite_gap(self):
        scorer = ClusteringScorer([[1, 0], [0, 1]], [1, 1], 1)
        self.assertEqual(scorer.score(scorer.metrics({"00": 100}), "best-gap")[0], 1.)
        self.assertLess(scorer.score({"reference_probability": 0, "feasibility_rate": .01,
                                     "best_gap_pct": 1000000}, "best-gap")[0], 1.)

    def test_bad_input_rejected(self):
        scorer = ClusteringScorer([[1, 0], [0, 1]], [1, 1], 1)
        for weights in ({}, {"01": -1}, {"01": float("nan")}, {"01": 0}, {"bad": 1}):
            with self.assertRaises(ValueError):
                scorer.metrics(weights)

    def test_infeasible_instance_rejected(self):
        with self.assertRaisesRegex(ValueError, "No assignment"):
            ClusteringScorer([[1, 0], [0, 1], [1, 1]], [3, 3, 3], 5)


class SearchTests(unittest.TestCase):
    def test_budget_reproducibility_and_incumbent(self):
        def run(budget):
            return bounded_search([.9, .9], [(0, 1)] * 2,
                                  lambda x: (float(np.sum((x - .2) ** 2)), None),
                                  evaluations=budget, seed=7)
        for budget in (1, 2, 5, 20):
            a, b = run(budget), run(budget)
            self.assertLessEqual(len(a["history"]), budget)
            self.assertEqual(a["history"], b["history"])
            self.assertLessEqual(a["best"]["loss"], a["baseline"]["loss"])
        self.assertLess(run(20)["best"]["loss"], run(20)["baseline"]["loss"])

    def test_fixed_parameters_supported(self):
        result = bounded_search([1., 0.], [(1, 1), (-1, 1)],
                                lambda x: (float(x[1] ** 2), None), evaluations=10)
        self.assertEqual(result["best"]["loss"], 0.)

    def test_emulator_failure_propagates(self):
        def fail(x):
            raise RuntimeError("backend unavailable")
        with self.assertRaisesRegex(RuntimeError, "backend unavailable"):
            bounded_search([0.], [(-1, 1)], fail, evaluations=5)

    def test_secondary_metrics_break_primary_ties(self):
        def evaluate(x):
            return 0.5, {"tie_break": (0., -1. if x[0] != 0 else -.5)}
        result = bounded_search([0.], [(-1, 1)], evaluate, evaluations=2)
        self.assertIsNot(result["best"], result["baseline"])


class PulseTests(unittest.TestCase):
    def test_baseline_waveform_and_clock(self):
        baseline = baseline_program()
        family = PulseFamily(baseline)
        amplitude, detuning = family.waveforms(family.initial)
        np.testing.assert_allclose(amplitude, baseline["parameters"]["rabi_samples"], atol=1e-12)
        np.testing.assert_allclose(detuning, baseline["parameters"]["detuning_samples"], atol=1e-12)
        x = family.initial.copy()
        x[-1] = 1.003
        self.assertEqual(family.describe(x)["duration_ns"], 1004)

    def test_endpoints_and_hardware_limits(self):
        family = PulseFamily(baseline_program())
        rng = np.random.default_rng(3)
        limits = np.asarray(family.bounds)
        for _ in range(20):
            x = rng.uniform(limits[:, 0], limits[:, 1])
            amplitude, detuning = family.waveforms(x)
            self.assertEqual(amplitude[0], 0.)
            self.assertEqual(amplitude[-1], 0.)
            self.assertGreaterEqual(amplitude.min(), 0.)
            self.assertLessEqual(amplitude.max(), family.amplitude_limit)
            self.assertLessEqual(np.abs(detuning).max(), family.detuning_limit)
            self.assertEqual(len(amplitude) % family.clock, 0)
            self.assertEqual(detuning[-1], family.delta_final)
            self.assertLessEqual(detuning[0], family.delta_initial)
            self.assertTrue(np.all(np.diff(detuning) >= -1e-12))
            peak = int(np.argmax(amplitude))
            self.assertTrue(np.all(np.diff(amplitude[:peak + 1]) >= -1e-12))
            self.assertTrue(np.all(np.diff(amplitude[peak:]) <= 1e-12))

    def test_endpoint_and_diagnostics_stay_bound_to_embedding(self):
        baseline = baseline_program()
        family = PulseFamily(baseline)
        x = family.initial.copy()
        x[1] = 1.5
        x[2] = .7
        x[3] = -.2
        with patch.dict("sys.modules", {"pulser": PULSER}):
            candidate = family.build(x)
        np.testing.assert_allclose(np.diag(candidate["diagnostics"]["realized_q_matrix"]), -2.)
        np.testing.assert_allclose(np.diag(candidate["parameters"]["global_q_matrix"]), -2.)
        self.assertEqual(baseline["parameters"]["delta_final"], 2.)
        self.assertEqual(candidate["parameters"]["delta_initial"], -6.)
        self.assertEqual(candidate["parameters"]["delta_final"], 2.)
        self.assertEqual(candidate["diagnostics"]["relative_matrix_error"], 0.)

    def test_time_allocation_changes_where_sweep_slows(self):
        family = PulseFamily(baseline_program())
        x = family.initial.copy()
        x[3] = math.log(4)
        fractions = family.describe(x)["sweep_segment_fractions"]
        self.assertGreater(fractions[1], fractions[0])
        self.assertGreater(fractions[1], fractions[2])

    def test_modulation_fall_time_checked(self):
        baseline = baseline_program()
        family = PulseFamily(baseline)
        baseline["sequence"].device.fall_time = 9000
        with patch.dict("sys.modules", {"pulser": PULSER}), self.assertRaisesRegex(ValueError, "fall time"):
            family.build(family.initial)

    def test_real_schrodinger_evolution_optimization(self):
        scorer = ClusteringScorer([[1, 0], [0, 1]], [1, 1], 1)
        with patch.dict("sys.modules", {"pulser": PULSER}):
            result = optimize_pulses(baseline_program(), scorer, evaluations=25,
                                     search_seed=2, emulator=schrodinger_emulator)
        self.assertLessEqual(result["simulations"], 25)
        self.assertLess(result["best"]["loss"], result["baseline"]["loss"] - .005)
        selected = result["best"]["payload"]
        self.assertAlmostEqual(sum(selected["probabilities"].values()), 1.)
        counts = resample(selected["probabilities"], 1000, 10000)
        self.assertEqual(sum(counts.values()), 1000)

    def test_sampled_mode_scores_counts(self):
        scorer = ClusteringScorer([[1, 0], [0, 1]], [1, 1], 1)

        def runner(*args, **kwargs):
            return {"probabilities": {"00": .9, "01": .1}, "counts": {"01": 10}}

        result = optimize_pulses(baseline_program(), scorer, evaluations=1,
                                 score_mode="sampled", emulator=runner)
        self.assertEqual(result["best"]["loss"], 0.)
        self.assertEqual(result["best"]["payload"]["metrics"]["reference_probability"], 1.)
        self.assertAlmostEqual(result["best"]["payload"]["predicted_metrics"]["reference_probability"], .1)

    def test_probability_support_does_not_invent_a_good_observed_shot(self):
        scorer = ClusteringScorer([[1, 0], [0, 1]], [1, 1], 1)

        def runner(*args, **kwargs):
            return {"probabilities": {"00": .999999, "01": .000001}, "counts": {"00": 1000}}

        result = optimize_pulses(baseline_program(), scorer, evaluations=1,
                                 score_mode="probabilities", emulator=runner)
        self.assertIsNone(result["best"]["payload"]["metrics"]["best_gap_pct"])
        with self.assertRaisesRegex(ValueError, "finite training-shot"):
            optimize_pulses(baseline_program(), scorer, metric="best-gap",
                             score_mode="probabilities", emulator=runner)


class MeasurementTests(unittest.TestCase):
    def test_final_seed_and_sample_count(self):
        p = {"00": .5, "11": .5}
        self.assertEqual(resample(p, 1000, 10), resample(p, 1000, 10))
        self.assertNotEqual(resample(p, 1000, 10), resample(p, 1000, 11))
        self.assertEqual(sum(resample(p, 1000, 10).values()), 1000)

    def test_output_protected_before_building(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.json"
            path.write_text("original")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(["--output", str(path)])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(path.read_text(), "original")


class CLIIntegrationTests(unittest.TestCase):
    def test_saved_instance_report_and_evaluator_compatibility(self):
        from instance_generator import generate_instance

        saved_config = {name: getattr(config, name) for name in
                        ("COORDINATES", "DEMANDS", "CAPACITY", "DEPOT")}
        call_count = [0]

        def builder(matrix, **kwargs):
            program = baseline_program()
            n = len(matrix)
            register = SimpleNamespace(qubits={f"q{i}": np.array([i * 6., 0.]) for i in range(n)})
            program["sequence"].register = register
            program["embedding"]["register"] = register
            program["embedding"]["positions_um"] = np.array(list(register.qubits.values()))
            interactions = 2 * matrix.copy()
            np.fill_diagonal(interactions, 0.)
            realized = interactions / 2
            np.fill_diagonal(realized, -2.)
            program["embedding"]["realized_interactions"] = interactions
            program["parameters"].update(
                q_matrix=matrix, interaction_matrix=interactions,
                target_detunings=-np.diag(matrix), n_qubits=n, global_q_matrix=realized,
            )
            program["diagnostics"]["realized_q_matrix"] = realized
            return program

        def runner(*args, **kwargs):
            call_count[0] += 1
            p = .1 if call_count[0] == 1 else .9
            probabilities = {"0000": 1 - p, "0011": p}
            return {"probabilities": probabilities,
                    "counts": resample(probabilities, kwargs["shots"], kwargs["seed"])}

        try:
            with tempfile.TemporaryDirectory() as directory:
                instance_path = Path(directory) / "instance.json"
                instance = generate_instance(1)
                instance_path.write_text(json.dumps(instance))
                output = Path(directory) / "optimized.json"
                with patch.dict("sys.modules", {
                    "pulser": PULSER, "pulses": SimpleNamespace(build_sequence=builder),
                    "emulate": SimpleNamespace(emulate=runner),
                }), contextlib.redirect_stdout(io.StringIO()):
                    code = main(["--instance", str(instance_path), "--evaluations", "2",
                                 "--duration", "1000", "--output", str(output)])
                self.assertEqual(code, 0)
                payload = json.loads(output.read_text())
                self.assertEqual(payload["instance"]["coordinates"], instance["coordinates"])
                self.assertEqual(payload["optimization"]["simulations"], 2)
                self.assertEqual(call_count[0], 4)
                self.assertEqual(payload["optimization"]["total_emulator_runs"], 4)
                self.assertEqual(sum(payload["counts"].values()), 1000)
                self.assertLessEqual(payload["optimized"]["training_loss"], payload["baseline"]["training_loss"])
                self.assertGreater(payload["optimized"]["training_metrics"]["feasibility_rate"],
                                   payload["baseline"]["training_metrics"]["feasibility_rate"])
                self.assertEqual(payload["optimization"]["metric"], "reference-probability")
                self.assertEqual(payload["optimization"]["objective"], "capacity_constrained_clustering")
                self.assertIsNone(payload["evaluation"]["routing"])
                self.assertEqual(payload["optimization"]["score_mode"], "sampled")
                self.assertEqual(payload["optimized"]["controls"]["final_detuning_rad_us"], 2.)
                self.assertNotIn("expected_penalized_route_cost", payload["optimized"])
                # Re-evaluate the saved top-level fields with the real evaluator.
                from evaluator import evaluate_samples
                report = evaluate_samples(payload["counts"], probabilities=payload["probabilities"],
                                          encoding=payload["encoding"], penalty_weight=.1)
                self.assertEqual(report, payload["evaluation"])
                for entry in payload["optimization"]["history"]:
                    self.assertNotIn("optimal_probability", entry)
        finally:
            for name, value in saved_config.items():
                setattr(config, name, value)
            for name in ("qubo", "classical", "evaluator"):
                sys.modules.pop(name, None)


if __name__ == "__main__":
    unittest.main()
