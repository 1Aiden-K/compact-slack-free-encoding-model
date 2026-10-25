"""Pulse optimization regression checks.

Run: python -m unittest discover -s tests -v"""

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from duquantum.tier_s import benchmark
from duquantum.tier_s.optimization import ClusteringScorer, validate_optimization_settings
from duquantum.tier_s import config as tier_config
from tests.test_pulse_optimization import PULSER, baseline_program, schrodinger_emulator


class IntegratedBenchmarkTests(unittest.TestCase):
    def setUp(self):
        snapshot = {name: getattr(tier_config, name) for name in ('COORDINATES', 'DEMANDS', 'CAPACITY')}
        self.addCleanup(tier_config.bind_instance, {
            'coordinates': snapshot['COORDINATES'], 'demands': snapshot['DEMANDS'],
            'capacity': snapshot['CAPACITY'],
        })

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
            "duquantum.quantum.pulses": SimpleNamespace(build_sequence=lambda *a, **k: baseline),
            "duquantum.quantum.ising": SimpleNamespace(build_ising=global_projection),
            "duquantum.quantum.emulator": SimpleNamespace(emulate=runner),
        }), patch.object(benchmark, "load_instance", return_value=instance), \
                contextlib.redirect_stdout(io.StringIO()):
            for name in ("duquantum.tier_s.evaluation", "duquantum.tier_s.model", "duquantum.tier_s.classical"):
                sys.modules.pop(name, None)
            payload = benchmark.assess_instance(Path("test_instance.json"), settings)
        return payload, calls, baseline

    def test_selected_pulse_is_evolved_again_and_final_metrics_use_new_run(self):
        payload, calls, baseline = self.run_case()
        optimization = payload["optimization"]
        result = payload["methods"]["emu-sv"]
        self.assertEqual(result["status"], "ok")
        self.assertEqual(payload["track"], "A")
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
        from duquantum.tier_s.instances import generate_instance
        with patch.dict(sys.modules):
            for seed in range(20):
                instance = generate_instance(seed)
                benchmark._bind_instance(instance)
                from duquantum.tier_s.evaluation import build_reference
                reference = build_reference(.1)
                scorer = ClusteringScorer(instance["coordinates"], instance["demands"],
                                           instance["capacity"])
                self.assertAlmostEqual(scorer.reference_cost, reference["optimal_cost"])
                for bits, record in reference["assignments"].items():
                    feasible, cost, _ = scorer.assignment(bits)
                    self.assertEqual(feasible, record["feasible"])
                    if feasible:
                        self.assertAlmostEqual(cost, record["cost"])




if __name__ == "__main__":
    unittest.main()
