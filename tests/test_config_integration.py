"""Exercise config defaults through real builders, emulation, and CLI reports."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ConfigIntegrationTests(unittest.TestCase):
    def test_nondefault_settings_and_cli_overrides(self):
        # A fresh interpreter models editing config before a run and avoids
        # leaking import-time instance arrays/defaults between tests.
        script = textwrap.dedent("""
            import json
            import runpy
            import sys
            from pathlib import Path
            import numpy as np
            import config

            directory = Path(sys.argv[1])
            config.LAMBDA = 0.23
            config.PULSE_DURATION = 400
            config.ENERGY_SCALE = 0.2
            config.ISING_ENERGY_SCALE = 0.4
            config.OMEGA_PEAK = 2.0
            config.DELTA_INITIAL = -3.0
            config.PULSE_PHASE = 0.25
            config.EMBEDDING_SEED = 11
            config.MEASUREMENT_SEED = 22
            config.EMBEDDING_RESTARTS = 1
            config.MAX_ITERATIONS = 25
            config.SHOTS = 73
            config.EMULATOR_DT = 20
            config.GPU = True
            config.MODULATION = True
            config.EPSILON_PCT = 5.0
            config.RUN_OUTPUT = str(directory / 'run.json')
            config.EVALUATION_OUTPUT = str(directory / 'evaluation.json')
            config.COORDINATES = np.array([[0., 0.], [3., 1.], [5., 2.]])
            config.DEMANDS = np.array([1, 2, 3])
            config.CAPACITY = 4
            config.DEPOT = [1., -1.]

            from qubo import build_q_matrix
            from classical import solve_classically
            from ising import build_ising
            from pulses import build_sequence
            from emulate import emulate
            from evaluator import build_reference, evaluate_samples

            matrix = build_q_matrix()
            np.testing.assert_allclose(matrix, build_q_matrix(0.23))
            assert not np.allclose(matrix, build_q_matrix(0.1))
            assert len(solve_classically()) == 8
            parameters = build_ising(matrix)
            assert parameters['energy_scale'] == 0.4
            assert parameters['duration_ns'] == 400
            assert parameters['phase'] == 0.25
            program = build_sequence(matrix)
            assert program['embedding']['seed'] == 11
            assert program['embedding']['restarts'] == 1
            assert program['parameters']['energy_scale'] == 0.2
            assert program['parameters']['omega_peak'] == 2.0
            assert program['parameters']['delta_initial'] == -3.0
            direct = emulate(program['sequence'], gpu=False)
            assert sum(direct['counts'].values()) == 73
            assert direct['seed'] == 22
            assert direct['dt_ns'] == 20
            assert direct['with_modulation'] is True
            report = evaluate_samples(direct['counts'])
            assert report['reference']['penalty_weight'] == 0.23
            assert report['epsilon_pct'] == 5.0
            assert report['reference']['depot'] == [1., -1.]

            # The CLI takes its remaining defaults from config and overrides
            # boolean settings as well as numeric settings explicitly supplied.
            sys.argv = ['emulate', '--no-gpu', '--no-with-modulation', '--shots', '37']
            runpy.run_module('emulate', run_name='__main__')
            saved = json.loads((directory / 'run.json').read_text())
            settings = saved['settings']
            assert settings['shots'] == 37
            assert settings['sample_seed'] == 22
            assert settings['embedding_seed'] == 11
            assert settings['embedding_maxiter'] == 25
            assert settings['duration_ns'] == 400
            assert settings['with_modulation'] is False
            assert settings['gpu'] is False
            assert sum(saved['counts'].values()) == 37
            assert saved['instance']['capacity'] == 4

            # Raw counts use config lambda; saved runs keep their recorded lambda.
            (directory / 'counts.json').write_text(json.dumps(saved['counts']))
            sys.argv = ['evaluator', str(directory / 'counts.json')]
            runpy.run_module('evaluator', run_name='__main__')
            evaluated = json.loads((directory / 'evaluation.json').read_text())
            assert evaluated['reference']['penalty_weight'] == 0.23
            config.LAMBDA = 0.5
            sys.argv = ['evaluator', str(directory / 'run.json')]
            runpy.run_module('evaluator', run_name='__main__')
            evaluated = json.loads((directory / 'evaluation.json').read_text())
            assert evaluated['reference']['penalty_weight'] == 0.23

            config.MAX_EXACT_BITS = 2
            try:
                build_reference(max_bits=config.MAX_EXACT_BITS)
            except ValueError:
                pass
            else:
                raise AssertionError('Exact reference ignored configured bit limit')
            config.MAX_ROUTING_CUSTOMERS = 2
            try:
                build_reference(max_bits=3)
            except ValueError:
                pass
            else:
                raise AssertionError('Reference ignored configured routing limit')
        """)
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-c", script, directory],
                cwd=ROOT,
                env={
                    **os.environ,
                    "PYTHONPATH": str(ROOT / "src"),
                    "MPLCONFIGDIR": directory,
                },
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            saved = json.loads((Path(directory) / "run.json").read_text())
            self.assertEqual(saved["evaluation"]["sampled"]["shots"], 37)


if __name__ == "__main__":
    unittest.main()
