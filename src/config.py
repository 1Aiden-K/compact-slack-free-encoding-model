import numpy as np

COORDINATES = np.array(
    [
        [0.0, 0.0],
        [2.0, 1.0],
        [6.0, 0.0],
        [7.0, 2.0],
    ]
)

DEMANDS = np.array([1, 2, 3, 4], dtype=int)

CAPACITY = 7

DEPOT = [0, 0]

LAMBDA = 0.1

PULSE_DURATION = 4000  # ns

EMULATOR = "emu-sv"

EMBEDDING_SEED = 0

MEASUREMENT_SEED = 0

EMBEDDING_RESTARTS = 8

MAX_ITERATIONS = 1500

SHOTS = 1000

MODULATION = False
