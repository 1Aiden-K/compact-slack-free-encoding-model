"""Run with: uv run --locked python -m unittest test_instance_generator -v."""

import contextlib
import io
import sys
import tempfile
import unittest
from copy import deepcopy
from itertools import product
from pathlib import Path
from unittest.mock import patch

import numpy as np

from instance_generator import (
    find_feasible_assignment,
    generate_instance,
    load_instance,
    main,
    validate_instance,
)


class InstanceGeneratorTests(unittest.TestCase):
    def test_seed_reproduces_complete_instance(self):
        self.assertEqual(generate_instance(0), generate_instance(0))
        self.assertNotEqual(
            generate_instance(0)["coordinates"], generate_instance(1)["coordinates"]
        )

    def test_exhaustive_certification_for_every_demand_vector(self):
        for demands in product(range(1, 5), repeat=4):
            feasible = any(
                sum(d for d, bit in zip(demands, bits) if bit) <= 7
                and sum(d for d, bit in zip(demands, bits) if not bit) <= 7
                for bits in product((0, 1), repeat=4)
            )
            certificate = find_feasible_assignment(demands)
            self.assertEqual(certificate is not None, feasible)
            if certificate is not None:
                bits = certificate["assignment"]
                self.assertEqual(
                    certificate["loads"],
                    [
                        sum(d for d, bit in zip(demands, bits) if bit == 1),
                        sum(d for d, bit in zip(demands, bits) if bit == 0),
                    ],
                )
                self.assertLessEqual(max(certificate["loads"]), 7)

    def test_total_capacity_alone_is_insufficient(self):
        self.assertLessEqual(sum([4, 4, 4, 1]), 14)
        self.assertIsNone(find_feasible_assignment([4, 4, 4, 1]))

    def test_thousand_seeds_are_valid_and_include_rejections(self):
        rejected = 0
        for seed in range(1000):
            instance = generate_instance(seed)
            validate_instance(instance, check_reproducibility=True)
            points = np.asarray(instance["coordinates"])
            self.assertTrue(np.all((points >= 0) & (points <= 100)))
            self.assertEqual(instance["load_ratio"], sum(instance["demands"]) / 14)
            self.assertEqual(
                instance["largest_demand_ratio"], max(instance["demands"]) / 7
            )
            rejected += instance["demand_draw_attempts"] - 1
        self.assertGreater(rejected, 0)

    def test_unused_vehicle_is_allowed(self):
        self.assertEqual(find_feasible_assignment([1, 1, 1, 1])["loads"], [0, 4])

    def test_fixed_depot_does_not_change_random_instance_data(self):
        a, b = generate_instance(5), generate_instance(5, depot=[50, 50])
        self.assertEqual(a["coordinates"], b["coordinates"])
        self.assertEqual(a["demands"], b["demands"])
        self.assertEqual(b["depot"], [50.0, 50.0])

    def test_invalid_inputs(self):
        for seed in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                generate_instance(seed)
        for depot in ([0], [-1, 0], [101, 0], [np.nan, 0]):
            with self.assertRaises(ValueError):
                generate_instance(0, depot=depot)

    def test_invalid_certificates_and_metadata_are_rejected(self):
        original = generate_instance(0)
        changes = [
            ("capacity", 8),
            ("vehicles", 3),
            ("tier", "M"),
            ("coordinates", [[-1.0, 0.0]] * 4),
            ("coordinates", [[np.nan, 0.0]] * 4),
            ("demands", [True, 1, 1, 1]),
            ("load_ratio", 0.123),
            ("largest_demand_ratio", 0.123),
            ("certificate", {"assignment": [0, 1], "loads": [0, 0]}),
            ("certificate", {"assignment": [0, 0, 0, 0], "loads": [0, 999]}),
        ]
        for key, value in changes:
            item = deepcopy(original)
            item[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_instance(item)

    def test_overloaded_certificate_is_rejected(self):
        item = generate_instance(0)
        item["demands"] = [4, 4, 4, 1]
        item["certificate"] = {"assignment": [1, 0, 0, 1], "loads": [5, 8]}
        with self.assertRaisesRegex(ValueError, "capacity"):
            validate_instance(item)

    def test_reproduction_detects_edited_coordinates(self):
        item = generate_instance(0)
        item["coordinates"][0][0] = 0.0
        validate_instance(item)
        with self.assertRaisesRegex(ValueError, "reproduction failed"):
            validate_instance(item, check_reproducibility=True)

    def test_bounded_rejection(self):
        with patch("instance_generator.find_feasible_assignment", return_value=None):
            with self.assertRaisesRegex(ValueError, "after 2 attempts"):
                generate_instance(0, max_attempts=2)

    def test_cli_batch_roundtrip_and_conflict_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "instances"
            command = [
                "instance_generator.py",
                "generate",
                "--seed",
                "0",
                "--count",
                "20",
                "--output-dir",
                str(output),
            ]
            for _ in range(2):
                with (
                    patch.object(sys, "argv", command),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    main()
            paths = sorted(output.glob("*.json"))
            self.assertEqual(len(paths), 20)
            self.assertEqual(
                [load_instance(path)["seed"] for path in paths], list(range(20))
            )
            with (
                patch.object(
                    sys, "argv", ["instance_generator.py", "validate", *map(str, paths)]
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                main()
            paths[0].unlink()
            paths[-1].write_text("{}")
            with (
                patch.object(sys, "argv", command),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                main()
            self.assertFalse(paths[0].exists())


if __name__ == "__main__":
    unittest.main()
