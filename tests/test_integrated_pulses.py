"""Integrated search -> selected sequence -> fresh evolution regression tests.

Uses an independent Schrodinger solver and the real optimizer/evaluator.
Pulser interface doubles stand in for the unavailable installed backend.
Run: python -m unittest test_optimize_pulses test_integrated_pulses -v
"""

import contextlib
import ast
import io
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from collections import Counter
import tempfile

import numpy as np

import benchmark
from optimize_pulses import ClusteringScorer, validate_optimization_settings, _load_cli_instance
from test_optimize_pulses import PULSER, baseline_program, schrodinger_emulator


class IntegratedBenchmarkTests(unittest.TestCase):
    def run_case(self, *, optimize=True, classical=False, fail_final=False):
        baseline = baseline_program()
        calls = []
        instance = {
            "coordinates": [[1., 0.], [0., 1.]], "demands": [1, 1],
            "capacity": 1, "depot": [0., 0.], "seed": 0,
            "n_customers": 2, "load_ratio": 1., "largest_demand_ratio": 1.,
        }
        settings = dict(benchmark.DEFAULTS, optimize_pulses=optimize,
                        classical_only=classical, evaluations=25, search_seed=2,
                        training_shots=503, shots=501)

        def runner(sequence, **kwargs):
            calls.append((sequence, kwargs))
            if fail_final and kwargs["shots"] == settings["shots"]:
                raise RuntimeError("final evolution failed")
            return schrodinger_emulator(sequence, **kwargs)

        def global_projection(matrix):
            matrix = np.array(matrix, copy=True)
            np.fill_diagonal(matrix, np.diag(matrix).mean())
            return {"global_q_matrix": matrix}

        # The evaluator and QUBO modules snapshot config. Restore their module
        # identities after each test as well as the bound instance settings.
        with patch.dict(sys.modules, {
            "pulser": PULSER,
            "pulses": SimpleNamespace(build_sequence=lambda *a, **k: baseline),
            "ising": SimpleNamespace(build_ising=global_projection),
            "emulate": SimpleNamespace(emulate=runner),
            "routing": SimpleNamespace(),  # Track A must not import a router.
        }), patch.object(benchmark, "load_instance", return_value=instance), \
                contextlib.redirect_stdout(io.StringIO()):
            for name in ("evaluator", "qubo", "classical"):
                sys.modules.pop(name, None)
            payload = benchmark.assess_instance(Path("test_instance.json"), settings)
        return payload, calls, baseline

    def test_selected_pulse_is_evolved_again_and_final_metrics_use_new_run(self):
        payload, calls, baseline = self.run_case()
        optimization = payload["optimization"]
        result = payload["methods"]["emu-sv"]
        self.assertEqual(result["status"], "ok")
        self.assertEqual(payload["track"], "A")
        self.assertNotIn("exact_cvrp", payload["methods"])
        self.assertFalse(any("cvrp" in key for key in benchmark.COLUMNS))
        self.assertIsNone(result["report"]["routing"])
        self.assertFalse(optimization["baseline_retained"])
        self.assertLessEqual(optimization["simulations"], 25)
        self.assertEqual(len(calls), optimization["simulations"] + 1)
        final_sequence, final_settings = calls[-1]
        self.assertIsNot(final_sequence, baseline["sequence"])
        self.assertTrue(any(seq is final_sequence for seq, _ in calls[:-1]))
        self.assertEqual(final_settings["shots"], 501)
        self.assertEqual(final_settings["seed"], 0)
        self.assertTrue(all(opts["seed"] == 10000 for _, opts in calls[:-1]))
        self.assertEqual(sum(result["counts"].values()), 501)
        rerun = schrodinger_emulator(final_sequence, **final_settings)
        self.assertEqual(result["counts"], rerun["counts"])
        self.assertEqual(result["probabilities"], rerun["probabilities"])
        self.assertEqual(final_sequence.pulse.detuning.samples[-1], 2.)
        self.assertEqual(payload["program"]["duration_ns"],
                         len(final_sequence.pulse.amplitude.samples))
        row = benchmark._row(payload, "emu-sv", "test.json")
        self.assertEqual(row["duration_ns"], payload["program"]["duration_ns"])
        self.assertEqual(row["total_emulator_runs"], len(calls))
        self.assertEqual(row["optimization_training_shots"], (len(calls) - 1) * 503)
        self.assertEqual(set(row), set(benchmark.COLUMNS))
        json.dumps(payload, allow_nan=False)

    def test_baseline_mode_skips_search(self):
        payload, calls, baseline = self.run_case(optimize=False)
        self.assertIsNone(payload["optimization"])
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], baseline["sequence"])

    def test_classical_only_never_searches_or_emulates(self):
        payload, calls, _ = self.run_case(classical=True)
        self.assertEqual(calls, [])
        self.assertIsNone(payload["optimization"])
        self.assertEqual(payload["methods"]["emu-sv"]["status"], "not_run")

    def test_final_evolution_failure_is_reported_without_cached_substitution(self):
        payload, calls, _ = self.run_case(fail_final=True)
        result = payload["methods"]["emu-sv"]
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["stage"], "emulation")
        self.assertIsNone(result["report"])
        self.assertEqual(len(calls), payload["optimization"]["simulations"] + 1)

    def test_training_seed_must_differ_and_best_gap_needs_shots(self):
        with self.assertRaisesRegex(ValueError, "must differ"):
            validate_optimization_settings(dict(benchmark.DEFAULTS, training_seed=0))
        with self.assertRaisesRegex(ValueError, "sampled"):
            validate_optimization_settings(dict(benchmark.DEFAULTS, metric="best-gap",
                                                 score_mode="probabilities"))

    def test_scorer_matches_existing_clustering_evaluator_on_all_20_instances(self):
        from instance_generator import generate_instance
        with patch.dict(sys.modules, {"routing": SimpleNamespace()}):
            for seed in range(20):
                instance = generate_instance(seed)
                benchmark._bind_instance(instance)
                from evaluator import build_reference
                reference = build_reference(.1)
                scorer = ClusteringScorer(instance["coordinates"], instance["demands"],
                                           instance["capacity"])
                self.assertAlmostEqual(scorer.reference_cost, reference["optimal_cost"])
                for bits, record in reference["assignments"].items():
                    feasible, cost, _ = scorer.assignment(bits)
                    self.assertEqual(feasible, record["feasible"])
                    if feasible:
                        self.assertAlmostEqual(cost, record["cost"])

    def test_track_a_does_not_require_a_depot(self):
        import config
        with patch.object(config, "DEPOT", None):
            _load_cli_instance(None, None)

    def test_emulate_cli_emulates_the_selected_sequence(self):
        import config
        # Load the real CLI definitions without importing embedding dependencies.
        tree = ast.parse(Path(__file__).with_name("pulses.py").read_text())
        definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                          and node.name == "add_sequence_arguments")
        namespace = {"config": config}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), "pulses.py", "exec"), namespace)
        add_sequence_arguments = namespace["add_sequence_arguments"]
        baseline = baseline_program()
        calls = []

        def runner(sequence, **kwargs):
            calls.append((sequence, kwargs))
            output = schrodinger_emulator(sequence, **kwargs)
            output.update(counts=Counter(output["counts"]), backend="emu-sv",
                          dt_ns=kwargs["dt"], gpu=False)
            return output

        pulses = SimpleNamespace(add_sequence_arguments=add_sequence_arguments,
                                 build_from_args=lambda args: baseline,
                                 print_program=lambda *a, **k: None)
        with patch.dict(sys.modules, {"pulses": pulses}), \
                patch.object(config, "COORDINATES", np.array([[1., 0.], [0., 1.]])), \
                patch.object(config, "DEMANDS", np.array([1, 1])), \
                patch.object(config, "CAPACITY", 1), \
                tempfile.TemporaryDirectory() as directory:
            for name in ("evaluator", "qubo", "classical"):
                sys.modules.pop(name, None)
            spec = importlib.util.spec_from_file_location("emulate", Path(__file__).with_name("emulate.py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            output_path = Path(directory) / "run.json"
            with patch.dict(sys.modules, {"emulate": module, "pulser": PULSER}), \
                    patch.object(module, "emulate", side_effect=runner), \
                    patch.object(sys, "argv", ["emulate.py", "--evaluations", "25",
                                              "--search-seed", "2", "--shots", "501",
                                              "--output", str(output_path)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                module.main()
            payload = json.loads(output_path.read_text())
            self.assertFalse(payload["optimization"]["baseline_retained"])
            self.assertEqual(len(calls), payload["optimization"]["simulations"] + 1)
            self.assertEqual(sum(payload["counts"].values()), 501)
            self.assertEqual(calls[-1][1]["seed"], 0)
            self.assertIsNot(calls[-1][0], baseline["sequence"])


if __name__ == "__main__":
    unittest.main()
