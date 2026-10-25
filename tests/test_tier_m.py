"""Verify the three-vehicle encoding against independent small-instance oracles."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from itertools import product, permutations
from pathlib import Path

import numpy as np

from duquantum.tier_m.model import ThreeVehicleModel, canonical_partition
from duquantum.tier_m.instances import generate_instance, validate_instance, load_instance, find_feasible_assignment
from duquantum.tier_m.classical import build_reference
from duquantum.tier_m.evaluation import evaluate_samples

ROOT = Path(__file__).resolve().parents[1]


class ModelTests(unittest.TestCase):
    def test_polynomial_matches_math_on_every_binary_state(self):
        rng = np.random.default_rng(7)
        for eta in (None, 0., 100.):
            model = ThreeVehicleModel(rng.uniform(0, 100, (4, 2)), [1, 2, 3, 4], 5,
                                      penalty_weight=.6, assignment_penalty=eta)
            matrix = model.build_q_matrix()
            c, h, j = model.build_qubo()
            np.testing.assert_allclose(matrix, matrix.T)
            self.assertTrue(np.all(matrix[np.triu_indices(8, 1)] > 0))
            for bits in product((0, 1), repeat=8):
                x = np.array(bits)
                a, b = x[:4], x[4:]
                indicators = [(int(a[i]), int(b[i]), 1 - int(a[i]) - int(b[i])) for i in range(4)]
                cost = sum(float(np.linalg.norm(model.coordinates[i] - model.coordinates[k]))
                           * sum(indicators[i][v] * indicators[k][v] for v in range(3))
                           for i in range(4) for k in range(i + 1, 4))
                loads = [sum(model.demands[i] * indicators[i][v] for i in range(4)) for v in range(3)]
                expected = (cost + .6 * sum((load - 10 / 3)**2 for load in loads)
                            + model.assignment_penalty * sum(int(a[i]) * int(b[i]) for i in range(4)))
                self.assertAlmostEqual(c + h @ x + x @ j @ x, expected, places=8)
                self.assertAlmostEqual(model.energy(x), expected, places=8)
                self.assertAlmostEqual(model.direct_energy(x), expected, places=8)

    def test_encode_decode_all_valid_labels_and_six_vehicle_symmetries(self):
        model = ThreeVehicleModel([[0, 0], [1, 0], [4, 0], [8, 0]], [1, 2, 3, 4], 7)
        for labels in product((1, 2, 3), repeat=4):
            np.testing.assert_array_equal(model.decode(model.encode(labels)), labels)
        original = [1, 2, 3, 1]
        partition = canonical_partition(original)
        energy = model.energy(model.encode(original))
        for mapping in permutations((1, 2, 3)):
            labels = [mapping[label - 1] for label in original]
            self.assertEqual(canonical_partition(labels), partition)
            self.assertAlmostEqual(model.energy(model.encode(labels)), energy)

    def test_invalid_assignments_are_not_repaired_or_given_physical_loads(self):
        model = ThreeVehicleModel([[0, 0], [1, 0]], [1, 1], 2)
        record = model.evaluate_assignment('1010')
        self.assertEqual(record['invalid_customers'], [0])
        self.assertEqual(record['assignments'], [0, 3])
        self.assertFalse(record['assignment_valid'])
        self.assertFalse(record['feasible'])
        self.assertIsNone(record['loads'])
        self.assertIsNone(record['cost'])

    def test_validity_and_capacity_are_separate(self):
        model = ThreeVehicleModel([[0, 0], [1, 0]], [2, 2], 2)
        valid = model.evaluate_assignment(model.encode([1, 1]))
        self.assertTrue(valid['assignment_valid'])
        self.assertFalse(valid['feasible'])
        self.assertEqual(valid['loads'], [4., 0., 0.])

    def test_bad_inputs_and_matrix_copy(self):
        model = ThreeVehicleModel([[0, 0], [1, 0]], [1, 1], 2)
        for bits in ('0123', '000', [True] * 4, [0, 0, 1, float('nan')]):
            with self.assertRaises(ValueError):
                model.decode(bits)
        for labels in ([0, 1], [1., 2.], [True, False]):
            with self.assertRaises(ValueError):
                model.encode(labels)
        for kwargs in ({'penalty_weight': 0}, {'penalty_weight': float('nan')}, {'assignment_penalty': -1}):
            with self.assertRaises(ValueError):
                ThreeVehicleModel([[0, 0], [1, 0]], [1, 1], 2, **kwargs)
        matrix = model.build_q_matrix()
        matrix[:] = 0
        self.assertNotEqual(model.build_q_matrix().sum(), 0)


class ReferenceTests(unittest.TestCase):
    def test_reference_matches_independent_feasible_partition_search(self):
        points, demands, capacity = [[0, 0], [2, 1], [5, 3], [10, 0]], [1, 2, 3, 4], 5
        model = ThreeVehicleModel(points, demands, capacity)
        reference = build_reference(model, check_all_bitstrings=True)
        feasible_costs = []
        for labels in product((1, 2, 3), repeat=4):
            loads = [sum(d for d, label in zip(demands, labels) if label == k) for k in (1, 2, 3)]
            if max(loads) <= capacity:
                feasible_costs.append(sum(np.linalg.norm(np.array(points[i]) - points[j])
                                          for i in range(4) for j in range(i + 1, 4) if labels[i] == labels[j]))
        self.assertAlmostEqual(reference['optimal_cost'], min(feasible_costs))
        self.assertEqual(reference['feasible_assignments'], len(feasible_costs))
        self.assertEqual(reference['qubo']['binary_states_checked'], 256)
        self.assertEqual(reference['qubo']['ground_invalid_assignments'], 0)
        proof = build_reference(model)
        for key in ('minimum_energy', 'ground_assignment_count', 'ground_feasible_assignments',
                    'all_ground_states_original_optimal'):
            if key == 'minimum_energy':
                self.assertAlmostEqual(proof['qubo'][key], reference['qubo'][key])
            else:
                self.assertEqual(proof['qubo'][key], reference['qubo'][key])

    def test_weak_penalty_automatically_checks_invalid_binary_ground_states(self):
        model = ThreeVehicleModel([[0, 0], [10, 0]], [1, 1], 2, assignment_penalty=0)
        reference = build_reference(model)
        self.assertFalse(model.assignment_validity_guaranteed)
        self.assertEqual(reference['qubo']['binary_states_checked'], 16)
        self.assertGreater(reference['qubo']['ground_invalid_assignments'], 0)
        self.assertFalse(reference['qubo']['all_ground_states_original_optimal'])
        self.assertEqual(reference['optimal_cost'], 0.)

    def test_ten_customer_reference_counts_and_label_symmetry(self):
        model = ThreeVehicleModel(np.zeros((10, 2)), [1] * 10, 20)
        reference = build_reference(model)
        self.assertEqual(model.n_bits, 20)
        self.assertEqual(reference['valid_assignments_enumerated'], 59049)
        self.assertEqual(reference['optimal_assignments'], 59049)
        self.assertEqual(reference['optimal_partitions'], 9842)
        self.assertEqual(reference['qubo']['ground_assignment_count'], 12600)
        self.assertEqual(reference['qubo']['ground_invalid_assignments'], 0)

    def test_infeasible_reference_and_limit(self):
        model = ThreeVehicleModel([[0, 0], [1, 0]], [2, 2], 1)
        reference = build_reference(model)
        self.assertEqual(reference['status'], 'infeasible_instance')
        self.assertIsNone(reference['optimal_cost'])
        with self.assertRaises(ValueError):
            build_reference(model, max_assignments=8)


class InstanceTests(unittest.TestCase):
    def test_seed_reproduction_and_certificate(self):
        for seed in (0, 1, 19):
            instance = generate_instance(seed)
            self.assertEqual(instance, generate_instance(seed))
            validate_instance(instance, check_reproducibility=True)
            self.assertEqual(instance['vehicles'], 3)
            self.assertEqual(instance['capacity'], 20)
            self.assertTrue(all(1 <= demand <= 8 for demand in instance['demands']))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'instance.json'
            path.write_text(json.dumps(instance))
            self.assertEqual(load_instance(path), instance)

    def test_total_demand_is_not_used_as_a_certificate(self):
        demands = [8] * 7 + [1] * 3
        self.assertLessEqual(sum(demands), 60)
        self.assertIsNone(find_feasible_assignment(demands))

    def test_invalid_or_stale_instance_rejected(self):
        original = generate_instance(0)
        for mutate in (
            lambda x: x['certificate']['assignment'].__setitem__(0, 0),
            lambda x: x['certificate']['loads'].__setitem__(0, 0),
            lambda x: x['demands'].__setitem__(0, 9),
            lambda x: x['coordinates'][0].__setitem__(0, -1),
            lambda x: x.__setitem__('load_ratio', 0),
            lambda x: x.__setitem__('vehicles', 2),
        ):
            instance = copy.deepcopy(original)
            mutate(instance)
            with self.assertRaises(ValueError):
                validate_instance(instance, check_reproducibility=True)


class SampleTests(unittest.TestCase):
    def test_invalid_and_capacity_violations_stay_in_denominator(self):
        model = ThreeVehicleModel([[0, 0], [10, 0]], [1, 1], 1)
        good = ''.join(map(str, model.encode([1, 2])))
        bad_capacity = ''.join(map(str, model.encode([1, 1])))
        report = evaluate_samples({good: 6, '1111': 2, bad_capacity: 2}, model=model,
                                  probabilities={good: .2, '1111': .5, bad_capacity: .3})
        self.assertEqual(report['sampled']['assignment_validity_rate'], .8)
        self.assertEqual(report['sampled']['feasibility_rate'], .6)
        self.assertEqual(report['sampled']['optimal_probability'], .6)
        self.assertEqual(report['sampled']['best_model_gap_pct'], 0.)
        self.assertEqual(report['predicted']['feasibility_rate'], .2)

    def test_bad_measurements_references_and_no_feasible_shots(self):
        model = ThreeVehicleModel([[0, 0], [10, 0]], [1, 1], 1)
        report = evaluate_samples({'1111': 10}, model=model)
        self.assertEqual(report['sampled']['assignment_validity_rate'], 0.)
        self.assertIsNone(report['sampled']['best_model_gap_pct'])
        self.assertEqual(report['sampled']['optimal_probability'], 0.)
        for counts in ({}, {'1111': True}, {'1111': 1.1}, {'1111': -1}, {'bad': 1}):
            with self.assertRaises(ValueError):
                evaluate_samples(counts, model=model)
        with self.assertRaises(ValueError):
            evaluate_samples({'1111': 1}, model=model, probabilities={'1111': .5})
        other = ThreeVehicleModel([[0, 0], [11, 0]], [1, 1], 1)
        with self.assertRaises(ValueError):
            evaluate_samples({'1111': 1}, model=model, reference=build_reference(other))


class TierMCLITests(unittest.TestCase):
    def command(self, *args):
        return subprocess.run([sys.executable, str(ROOT / 'main.py'), '--tier', 'm', *args],
                              cwd=ROOT, capture_output=True, text=True)

    def test_generate_validate_reference_export_and_evaluate(self):
        with tempfile.TemporaryDirectory() as directory:
            instances = Path(directory) / 'instances'
            generated = self.command('generate', '--seed', '0', '--output-dir', str(instances))
            self.assertEqual(generated.returncode, 0, generated.stderr)
            instance_path = instances / 'tier_m_seed_0000.json'
            checked = self.command('validate', str(instance_path))
            self.assertEqual(checked.returncode, 0, checked.stderr)
            reference_path, qubo_path = Path(directory) / 'reference.json', Path(directory) / 'qubo.json'
            classical = self.command('classical', '--instance', str(instance_path),
                                     '--output', str(reference_path), '--qubo-output', str(qubo_path))
            self.assertEqual(classical.returncode, 0, classical.stderr)
            reference = json.loads(reference_path.read_text())
            self.assertEqual(reference['valid_assignments_enumerated'], 59049)
            self.assertEqual(np.asarray(json.loads(qubo_path.read_text())['q_matrix']).shape, (20, 20))
            counts_path = Path(directory) / 'counts.json'
            counts_path.write_text(json.dumps({reference['representative']['bits']: 100, '1' * 20: 1}))
            evaluated = self.command('evaluate', str(counts_path), '--instance', str(instance_path))
            self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
            report = json.loads(evaluated.stdout)
            self.assertEqual(report['sampled']['optimal_probability'], 100 / 101)
            self.assertEqual(report['sampled']['best_model_gap_pct'], 0.)

    def test_unimplemented_quantum_command_is_explicit(self):
        result = self.command('run')
        self.assertEqual(result.returncode, 2)
        self.assertIn('not integrated yet', result.stderr)


if __name__ == '__main__':
    unittest.main()
