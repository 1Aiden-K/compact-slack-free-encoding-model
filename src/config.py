"""User settings shared by the model, builders, emulators, and evaluation CLIs.

CLI options and explicit function arguments override these defaults. Edit settings
before starting Python; defaults and derived instance arrays load at import time.
None selects an automatic value where documented. Physical hardware limits come
from DEVICE, rather than overriding device specifications with local settings.
"""

import logging

import numpy as np

# Problem instance. The binary model represents exactly two vehicles.
COORDINATES = np.array([[0.0, 0.0], [2.0, 1.0], [6.0, 0.0], [7.0, 2.0]])
DEMANDS = np.array([1, 2, 3, 4], dtype=int)
CAPACITY = 7
DEPOT = [0, 0]  # None disables exact routing.
LAMBDA = 0.1

# Pulse construction. DEVICE names a device exported by pulser.
DEVICE = "AnalogDevice"
PULSE_DURATION = 4000  # ns
ENERGY_SCALE = None  # Automatic device-aware scaling in build_sequence.
ISING_ENERGY_SCALE = 1.0  # Standalone build_ising; also its CLI default.
OMEGA_PEAK = None  # Automatic median positive interaction, rad/us.
DELTA_INITIAL = None  # Automatic negative maximum target detuning, rad/us.
INTERACTION_COEFF = None  # Standalone Ising ideal-distance calculation only.
PULSE_PHASE = 0.0
AUTO_SCALE_MAX = 1.0
HARDWARE_SAFETY_FACTOR = 0.9
CHECK_GROUND_STATES = True

# Embedding. Physical limits below apply to standalone optimize_positions only;
# embed_register always reads the selected device's actual limits.
EMBEDDING_SEED = 0
EMBEDDING_RESTARTS = 8
MAX_ITERATIONS = 1500
MIN_ATOM_DISTANCE = 5.0  # um
MAX_RADIAL_DISTANCE = 38.0  # um
EMBEDDING_METHOD = "SLSQP"  # Must support equality and inequality constraints.
EMBEDDING_FTOL = 1e-12
GEOMETRIC_MARGIN = 1e-4  # Fraction of minimum atom spacing.
DISTANCE_FLOOR_FRACTION = 0.1
INITIAL_DISK_RADIUS_FRACTION = 0.45
INITIAL_PLACEMENT_ATTEMPTS = 200
MDS_INTERACTION_FLOOR_FRACTION = 1e-3
MDS_SPACING_FACTOR = 1.001

# Ideal coherent emulation and measurement sampling.
EMULATOR = "emu-sv"
MEASUREMENT_SEED = 0
SHOTS = 1000
MODULATION = False
EMULATOR_DT = 10  # emu-sv integration step, ns
GPU = False
EMULATOR_LOG_LEVEL = logging.WARNING
QUTIP_SAMPLING_RATE = 1.0

# Exact references and numerical comparisons.
MAX_EXACT_BITS = 16
MAX_ROUTING_CUSTOMERS = 16
EPSILON_PCT = 0.0
COST_REL_TOL = 1e-10
COST_ABS_TOL = 1e-9
MATRIX_REL_TOL = 1e-10
MATRIX_ABS_TOL = 1e-12
GROUND_STATE_REL_TOL = 1e-10
PROBABILITY_ABS_TOL = 1e-8

# CLI/report defaults. Output paths may be strings or pathlib.Path objects.
SHOW_ALL = False
SKIP_EVALUATION = False
RUN_OUTPUT = None
EVALUATION_OUTPUT = None
PRINT_PRECISION = 6
PRINT_SUPPRESS = True
JSON_INDENT = 2
