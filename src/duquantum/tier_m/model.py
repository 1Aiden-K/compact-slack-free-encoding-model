"""Three-vehicle clustering QUBO with an implicit third vehicle.

Bit order: [a_0, ..., a_(N-1), b_0, ..., b_(N-1)]. Per customer,
(a,b)=(1,0) selects vehicle 1, (0,1) vehicle 2, and (0,0) vehicle 3.
(1,1) is invalid. Invalid measurements are reported without repair.
"""

import numpy as np

from duquantum.tier_m import config


class ThreeVehicleModel:
    """Explicit instance data; no module-level instance snapshots.

    Small instances are accepted for verification. The Tier M generator and
    validator separately enforce the challenge's ten-customer parameters.
    """

    def __init__(self, coordinates, demands, capacity, *, penalty_weight=config.LAMBDA,
                 assignment_penalty=config.ASSIGNMENT_PENALTY, penalty_strategy="fixed"):
        self.coordinates = np.array(coordinates, dtype=float, copy=True)
        self.demands = np.array(demands, dtype=float, copy=True)
        self.capacity = float(capacity)
        self.penalty_weight = float(penalty_weight)
        self.penalty_strategy = penalty_strategy
        if penalty_strategy not in ("fixed", "adaptive_demand_pressure_v1"):
            raise ValueError("Unknown lambda selection strategy.")
        if (self.demands.ndim != 1 or self.demands.size == 0
                or not np.isfinite(self.demands).all() or np.any(self.demands < 0)):
            raise ValueError("Demands must be a nonempty, finite, nonnegative vector.")
        self.n = len(self.demands)
        self.n_bits = 2 * self.n
        if (self.coordinates.ndim != 2 or self.coordinates.shape[0] != self.n
                or self.coordinates.shape[1] == 0 or not np.isfinite(self.coordinates).all()):
            raise ValueError("Coordinates must have one finite, nonempty row per customer.")
        if not np.isfinite(self.capacity) or self.capacity < 0:
            raise ValueError("Capacity must be finite and nonnegative.")
        if not np.isfinite(self.penalty_weight) or self.penalty_weight <= 0:
            raise ValueError("penalty_weight must be finite and positive.")
        self.total_demand = float(self.demands.sum())
        self.distances = np.linalg.norm(
            self.coordinates[:, None, :] - self.coordinates[None, :, :], axis=2)
        d, t, lam = self.demands, self.total_demand, self.penalty_weight
        h = -self.distances.sum(axis=1) + 2 * lam * (d**2 - t * d)
        self.assignment_penalty_bound = float(np.max(-h))
        self.assignment_penalty = float(
            self.assignment_penalty_bound + 1.0
            if assignment_penalty is None else assignment_penalty)
        if not np.isfinite(self.assignment_penalty) or self.assignment_penalty < 0:
            raise ValueError("assignment_penalty must be finite and nonnegative.")
        self.assignment_validity_guaranteed = (
            self.assignment_penalty > self.assignment_penalty_bound)
        w = self.distances + 2 * lam * np.outer(d, d)
        np.fill_diagonal(w, 0.)
        same = w + np.diag(h)
        cross = w / 2 + np.diag(lam * d**2 + self.assignment_penalty / 2)
        self._matrix = np.block([[same, cross], [cross, same]])
        self.offset = float(np.triu(self.distances, 1).sum() + 2 * lam * t**2 / 3)
        if not np.isfinite(self._matrix).all() or not np.isfinite(self.offset):
            raise ValueError("Instance or weights overflow the QUBO coefficients.")
        for array in (self.coordinates, self.demands, self.distances, self._matrix):
            array.setflags(write=False)

    def build_q_matrix(self):
        """Return symmetric M for F(x)=c+x.T@M@x, as an independent array."""
        return self._matrix.copy()

    def build_qubo(self):
        """Return c, h, upper-triangular J with full pair coefficients."""
        return self.offset, np.diag(self._matrix).copy(), 2 * np.triu(self._matrix, 1)

    def _bits(self, bits):
        if isinstance(bits, str):
            if len(bits) != self.n_bits or set(bits) - {'0', '1'}:
                raise ValueError(f"Expected a {self.n_bits}-bit string.")
            return np.fromiter(map(int, bits), dtype=np.int8)
        array = np.asarray(bits)
        if (array.shape != (self.n_bits,) or array.dtype.kind not in 'iuf'
                or not np.isin(array, (0, 1)).all()):
            raise ValueError(f"Expected exactly {self.n_bits} binary values.")
        return array.astype(np.int8)

    def encode(self, assignments):
        """Encode vehicle labels 1, 2, 3, without adding slack variables."""
        labels = np.asarray(assignments)
        if (labels.shape != (self.n,) or labels.dtype.kind not in 'iu'
                or not np.isin(labels, (1, 2, 3)).all()):
            raise ValueError(f"Expected {self.n} integer vehicle labels in 1, 2, 3.")
        return np.r_[labels == 1, labels == 2].astype(np.int8)

    def decode(self, bits):
        """Return vehicle labels; label 0 marks an invalid (1,1) customer."""
        x = self._bits(bits)
        a, b = x[:self.n], x[self.n:]
        return np.where(a + b == 2, 0, np.where(a, 1, np.where(b, 2, 3)))

    def energy(self, bits):
        x = self._bits(bits)
        return float(self.offset + x @ self._matrix @ x)

    def direct_energy(self, bits):
        """Evaluate the mathematical polynomial, including invalid bitstrings.

        The third indicator may be -1 for an invalid customer. This is an
        algebraic diagnostic, not an interpretation as a physical assignment.
        """
        x = self._bits(bits)
        a, b = x[:self.n], x[self.n:]
        indicators = np.column_stack((a, b, 1 - a - b))
        cost = np.triu(self.distances * (indicators @ indicators.T), 1).sum()
        loads = self.demands @ indicators
        return float(cost + self.penalty_weight * np.sum((loads - self.total_demand / 3)**2)
                     + self.assignment_penalty * np.dot(a.astype(float), b))

    def evaluate_assignment(self, bits):
        """Check assignment validity and capacities separately; never repair."""
        x = self._bits(bits)
        labels = self.decode(x)
        invalid = np.flatnonzero(labels == 0).tolist()
        valid = not invalid
        loads = [float(self.demands[labels == k].sum()) for k in (1, 2, 3)] if valid else None
        capacity_feasible = valid and all(load <= self.capacity for load in loads)
        cost = (float(np.triu(self.distances * (labels[:, None] == labels[None, :]), 1).sum())
                if valid else None)
        return {
            'bits': ''.join(map(str, x)), 'assignments': labels.tolist(),
            'assignment_valid': valid, 'invalid_customers': invalid,
            'capacity_feasible': bool(capacity_feasible), 'feasible': bool(capacity_feasible),
            'loads': loads, 'cost': cost, 'energy': self.energy(x),
        }

    def settings(self):
        return {
            'tier': 'M', 'n_customers': self.n, 'vehicles': 3, 'n_bits': self.n_bits,
            'penalty_weight': self.penalty_weight, 'penalty_strategy': self.penalty_strategy,
            'assignment_penalty': self.assignment_penalty,
            'assignment_penalty_bound': self.assignment_penalty_bound,
            'assignment_validity_guaranteed': self.assignment_validity_guaranteed,
            'capacity_is_exactly_encoded': False,
            'bit_order': [f'a_{i}' for i in range(self.n)] + [f'b_{i}' for i in range(self.n)],
        }


def canonical_partition(assignments):
    """Identify partitions under all six permutations of vehicle labels."""
    labels = tuple(assignments)
    if any(label not in (1, 2, 3) for label in labels):
        raise ValueError("Canonical partitions require valid vehicle labels.")
    seen = {}
    return tuple(seen.setdefault(int(label), len(seen)) for label in labels)


def build_model(instance=None, *, penalty_weight=None, assignment_penalty=None):
    """Construct from explicit instance data, or the default certified seed."""
    if instance is None:
        from duquantum.tier_m.instances import generate_instance
        instance = generate_instance(config.DEFAULT_SEED)
    if isinstance(penalty_weight, str) and penalty_weight == 'auto':
        from duquantum.tier_m.penalty import STRATEGY, select_lambda
        weight, strategy = select_lambda(instance), STRATEGY
    else:
        weight, strategy = config.LAMBDA if penalty_weight is None else penalty_weight, 'fixed'
    return ThreeVehicleModel(
        instance['coordinates'], instance['demands'], instance['capacity'],
        penalty_weight=weight, penalty_strategy=strategy,
        assignment_penalty=config.ASSIGNMENT_PENALTY if assignment_penalty is None else assignment_penalty,
    )


def build_qubo(penalty_weight=None, assignment_penalty=None, *, instance=None):
    return build_model(instance, penalty_weight=penalty_weight,
                       assignment_penalty=assignment_penalty).build_qubo()


def build_q_matrix(penalty_weight=None, assignment_penalty=None, *, instance=None):
    return build_model(instance, penalty_weight=penalty_weight,
                       assignment_penalty=assignment_penalty).build_q_matrix()
