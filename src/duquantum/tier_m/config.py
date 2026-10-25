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
