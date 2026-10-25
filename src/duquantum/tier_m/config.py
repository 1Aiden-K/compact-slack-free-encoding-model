"""Tier M model and exact-reference defaults."""

N_CUSTOMERS = 10
N_VEHICLES = 3
CAPACITY = 20
DEMAND_RANGE = (1, 8)
SQUARE_SIDE = 100.0
DEFAULT_SEED = 0
LAMBDA = 0.1
ASSIGNMENT_PENALTY = None  # Automatic: sufficient validity bound plus 1.
MAX_EXACT_ASSIGNMENTS = 3**10
MAX_EXACT_BINARY_BITS = 20

# Heuristic calibrated on seeds 0..19; evaluation should use held-out seeds.
# These constants never depend on a test instance's optimal assignment.
AUTO_LAMBDA_BASE = 0.3
AUTO_LAMBDA_DEMAND_REFERENCE = 48.0
AUTO_LAMBDA_DEMAND_SLOPE = 0.4
AUTO_LAMBDA_DISTANCE_REFERENCE = 50.0
AUTO_LAMBDA_MIN = 0.001
AUTO_LAMBDA_MAX = 100.0
