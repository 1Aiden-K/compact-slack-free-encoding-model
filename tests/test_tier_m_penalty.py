"""Adaptive Tier M penalty selection and reference benchmark checks."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from duquantum.tier_m.classical import build_reference
from duquantum.tier_m.instances import generate_instance
from duquantum.tier_m.lambda_benchmark import benchmark, parse_seeds
from duquantum.tier_m.model import build_model
from duquantum.tier_m.penalty import STRATEGY, select_lambda

ROOT = Path(__file__).resolve().parents[1]


class PenaltyTests(unittest.TestCase):
    def test_auto_has_no_access_to_exact_solver(self):
        instance = generate_instance(0)
        with patch('duquantum.tier_m.classical.build_reference', side_effect=AssertionError('leak')):
            value = select_lambda(instance)
            model = build_model(instance, penalty_weight='auto')
        self.assertGreater(value, 0)
        self.assertAlmostEqual(model.penalty_weight, value)
        self.assertEqual(model.penalty_strategy, STRATEGY)
        self.assertEqual(model.n_bits, 20)
        np.testing.assert_allclose(model.build_q_matrix(), model.build_q_matrix().T)
        self.assertTrue(model.assignment_validity_guaranteed)
        self.assertEqual(build_model(instance).penalty_weight, 0.1)
        self.assertEqual(build_model(instance, penalty_weight=2).penalty_weight, 2.)

    def test_distance_scale_and_pressure_monotonicity(self):
        instance = generate_instance(8)  # avoids floor/ceiling
        value = select_lambda(instance)
        scaled = dict(instance, coordinates=(2 * np.asarray(instance['coordinates'])).tolist())
        self.assertAlmostEqual(select_lambda(scaled), value * 2)
        increased = dict(instance, demands=list(instance['demands']))
        increased['demands'][0] += 1
        self.assertAlmostEqual(select_lambda(increased) / value, np.exp(.4))
        for invalid in (dict(instance, demands=[1]), dict(instance, capacity=0)):
            with self.assertRaises(ValueError):
                select_lambda(invalid)

    def test_benchmark_agrees_with_independent_reference(self):
        results = benchmark([0, 5], fixed=(0.1, .3))
        for row in results['rows']:
            inst = generate_instance(row['seed'])
            model = build_model(inst, penalty_weight='auto' if row['method'] == 'adaptive' else row['lambda'])
            reference = build_reference(model)
            self.assertAlmostEqual(reference['qubo']['minimum_energy'], row['minimum_energy'], places=6)
            self.assertEqual(reference['qubo']['ground_assignment_count'], row['ground_state_count'])
            self.assertEqual(reference['qubo']['ground_feasible_assignments'] == row['ground_state_count'],
                             row['all_ground_states_feasible'])
            self.assertEqual(reference['qubo']['all_ground_states_original_optimal'],
                             row['all_ground_states_clustering_optimal'])

    def test_seed_parser(self):
        self.assertEqual(parse_seeds('0:3'), [0, 1, 2])
        self.assertEqual(parse_seeds('10,12'), [10, 12])
        for bad in ('5:5', '3:1', '-1:2', '1,1', ''):
            with self.assertRaises(ValueError):
                parse_seeds(bad)

    def test_cli_auto_and_saved_payload(self):
        inst = generate_instance(0)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            instance_path = directory / 'instance.json'
            instance_path.write_text(json.dumps(inst))
            def run(*args):
                return subprocess.run([sys.executable, str(ROOT / 'main.py'), '--tier', 'm', *args],
                                      cwd=ROOT, capture_output=True, text=True)
            classical = run('classical', '--instance', str(instance_path), '--lambda', 'auto')
            self.assertEqual(classical.returncode, 0, classical.stderr)
            ref = json.loads(classical.stdout)
            self.assertEqual(ref['settings']['penalty_strategy'], STRATEGY)
            self.assertTrue(ref['qubo']['all_ground_states_original_optimal'])
            saved = {'tier':'M', 'instance':ref['instance'], 'settings':ref['settings'],
                     'counts': {ref['representative']['bits']: 2}}
            saved_path = directory / 'counts.json'
            saved_path.write_text(json.dumps(saved))
            eval_result = run('evaluate', str(saved_path))
            self.assertEqual(eval_result.returncode, 0, eval_result.stderr)
            self.assertEqual(json.loads(eval_result.stdout)['reference']['settings']['penalty_strategy'], STRATEGY)
            override = run('evaluate', str(saved_path), '--lambda', '0.3')
            self.assertEqual(override.returncode, 2)
            self.assertIn('strategy differs', override.stderr)
            invalid = run('classical', '--lambda', 'not-a-number')
            self.assertEqual(invalid.returncode, 2)
            result = run('lambda-benchmark', '--seeds', '0:2', '--fixed', '.1')
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout)
            self.assertEqual(summary['training/adaptive']['instances'], 2)


if __name__ == '__main__':
    unittest.main()
