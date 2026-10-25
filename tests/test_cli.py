"""Exercise the entry point, process isolation, and saved-run evaluation."""
import contextlib
import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from duquantum.cli import main
from duquantum.tier_s import config as tier_config
from duquantum.tier_s import run
from tests.test_pulse_optimization import PULSER, baseline_program, schrodinger_emulator

ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(tier_config.bind_instance, {
            'coordinates': tier_config.COORDINATES, 'demands': tier_config.DEMANDS,
            'capacity': tier_config.CAPACITY,
        })

    def command(self, *args, cwd=ROOT):
        return subprocess.run([sys.executable, str(ROOT / 'main.py'), *args],
                              cwd=cwd, text=True, capture_output=True)

    def test_command_help_and_explicit_unsupported_tier(self):
        for command in ('generate', 'validate', 'benchmark', 'run', 'optimize', 'evaluate', 'classical'):
            with self.subTest(command=command):
                result = self.command(command, '--help')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f'main.py {command}', result.stdout)
        result = self.command('--tier', 'l', 'benchmark')
        self.assertEqual(result.returncode, 2)

    def test_generate_validate_and_worker_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            instances = Path(directory) / 'instances'
            results = Path(directory) / 'results'
            generated = self.command('generate', '--seed', '0', '--count', '2',
                                     '--output-dir', str(instances), cwd=directory)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            files = sorted(instances.glob('*.json'))
            self.assertEqual(len(files), 2)
            validated = self.command('validate', *(str(path) for path in files), cwd=directory)
            self.assertEqual(validated.returncode, 0, validated.stderr)
            benchmark = self.command('benchmark', '--instances-dir', str(instances),
                                     '--classical-only', '--output-dir', str(results), cwd=directory)
            self.assertEqual(benchmark.returncode, 0, benchmark.stderr)
            with (results / 'results.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 8)
            self.assertEqual({row['status'] for row in rows}, {'ok', 'not_run'})
            references = [row['clustering_reference_cost'] for row in rows
                          if row['method'] == 'exact_clustering']
            self.assertEqual(len(set(references)), 2)
            manifest = json.loads((results / 'manifest.json').read_text())
            self.assertIn('duquantum.quantum.optimization', manifest['sources'])
            self.assertIn('duquantum.tier_s.config', manifest['sources'])

    def test_single_run_dispatch_uses_selected_pulse_and_saved_instance(self):
        baseline = baseline_program()
        calls = []

        def runner(sequence, **kwargs):
            calls.append((sequence, kwargs))
            output = schrodinger_emulator(sequence, **kwargs)
            output.update(counts=Counter(output['counts']), backend='emu-sv',
                          dt_ns=kwargs['dt'], gpu=False)
            return output

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(tier_config, 'COORDINATES', np.array([[1., 0.], [0., 1.]])), \
                patch.object(tier_config, 'DEMANDS', np.array([1, 1])), \
                patch.object(tier_config, 'CAPACITY', 1), \
                patch.dict(sys.modules, {'pulser': PULSER}):
            # Clear the two model snapshots after changing the test instance.
            for name in ('duquantum.tier_s.model', 'duquantum.tier_s.classical'):
                sys.modules.pop(name, None)
            output_path = Path(directory) / 'nested' / 'run.json'
            with patch.object(run, 'build_from_args', return_value=baseline), \
                    patch.object(run, 'print_program'), patch.object(run, 'emulate', side_effect=runner), \
                    patch('duquantum.quantum.emulator.emulate', side_effect=runner), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = main(['run', '--evaluations', '25', '--search-seed', '2',
                             '--shots', '501', '--output', str(output_path)])
            self.assertEqual(code, 0)
            payload = json.loads(output_path.read_text())
            self.assertFalse(payload['optimization']['baseline_retained'])
            self.assertEqual(len(calls), payload['optimization']['simulations'] + 1)
            self.assertEqual(sum(payload['counts'].values()), 501)
            self.assertIsNot(calls[-1][0], baseline['sequence'])
            # A new process starts with the default four-customer config, so
            # equality proves evaluation binds the saved two-customer instance.
            evaluated_path = Path(directory) / 'evaluation.json'
            evaluated = self.command('evaluate', str(output_path), '--output', str(evaluated_path))
            self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
            self.assertEqual(json.loads(evaluated_path.read_text()), payload['evaluation'])


if __name__ == '__main__':
    unittest.main()
