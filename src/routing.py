"""Exact depot-to-depot routing and two-vehicle CVRP for small instances.

Held-Karp dynamic programming finds an exact TSP cost for every customer
subset. Enumerating capacity-feasible complementary subsets then gives the
exact CVRP optimum for two identical vehicles, including unused vehicles.

This is classical evaluation of quantum-selected clusters, not quantum routing.
Customer indices match config.COORDINATES; the depot is a separate location,
even when it coincides with a customer. No delivery splitting or reloading.
Requires only NumPy. Time O(N**2 * 2**N), memory O(N * 2**N), limit set by config.MAX_ROUTING_CUSTOMERS.
"""

from numbers import Integral

import numpy as np

import config


def validate_depot(coordinates, depot):
    """Return a finite depot vector with the same dimension as customer points."""
    points = np.asarray(coordinates, dtype=float)
    if points.ndim != 2 or points.shape[1] == 0 or not np.isfinite(points).all():
        raise ValueError(
            "Customer coordinates must be a finite, nonempty-dimensional matrix."
        )
    depot = np.asarray(depot, dtype=float)
    if depot.shape != (points.shape[1],) or not np.isfinite(depot).all():
        raise ValueError(
            "DEPOT must be a finite coordinate vector matching the customers' dimension."
        )
    return depot


class ExactRouter:
    """Cache exact closed tours for every subset of up to the configured customer limit.

    D[mask, j] is the cheapest path starting at the depot, visiting exactly
    mask, and ending at customer j. The return-to-depot edge is added only
    after the subset is complete. Parent pointers reconstruct a valid tour.
    Symmetric tours may have ties; one minimizing order is returned.
    """

    def __init__(
        self, coordinates, depot, *, max_customers=config.MAX_ROUTING_CUSTOMERS
    ):
        self.coordinates = np.array(coordinates, dtype=float, copy=True)
        self.depot = validate_depot(self.coordinates, depot).copy()
        self.n = len(self.coordinates)
        if (
            isinstance(max_customers, bool)
            or not isinstance(max_customers, Integral)
            or max_customers < 0
        ):
            raise ValueError("max_customers must be a nonnegative integer.")
        if self.n > max_customers:
            raise ValueError(
                f"Exact routing supports at most {max_customers} customers."
            )
        self.distances = np.linalg.norm(
            self.coordinates[:, None, :] - self.coordinates[None, :, :], axis=2
        )
        self.depot_distances = np.linalg.norm(self.coordinates - self.depot, axis=1)
        if (
            not np.isfinite(self.distances).all()
            or not np.isfinite(self.depot_distances).all()
        ):
            raise ValueError("Customer/depot distances must be finite.")
        count = 1 << self.n
        paths = np.full((count, self.n), np.inf)
        self.parents = np.full((count, self.n), -1, dtype=np.int32)
        self.costs = np.zeros(count)
        self.ends = np.full(count, -1, dtype=np.int32)
        for mask in range(1, count):
            remaining = mask
            while remaining:
                bit = remaining & -remaining
                j = bit.bit_length() - 1
                previous = mask ^ bit
                if previous == 0:
                    paths[mask, j] = self.depot_distances[j]
                else:
                    candidates = paths[previous] + self.distances[:, j]
                    k = int(np.argmin(candidates))
                    paths[mask, j] = candidates[k]
                    self.parents[mask, j] = k
                remaining ^= bit
            closed = paths[mask] + self.depot_distances
            end = int(np.argmin(closed))
            self.costs[mask] = closed[end]
            self.ends[mask] = end

    def _assignment_mask(self, bits):
        if not isinstance(bits, str) or len(bits) != self.n or set(bits) - {"0", "1"}:
            raise ValueError(f"Expected a {self.n}-bit assignment string.")
        return sum(1 << i for i, bit in enumerate(bits) if bit == "1")

    def _bits(self, mask):
        return "".join("1" if mask & (1 << i) else "0" for i in range(self.n))

    def tour(self, customers):
        """Return one shortest depot-to-depot tour for a specified customer set.

        An unused vehicle has no stops and length zero. A nonempty tour's
        stops are ['depot', customer index, ..., 'depot'].
        """
        customers = list(customers)
        if any(
            isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < self.n
            for i in customers
        ):
            raise ValueError("Customer indices must be integers within the instance.")
        if len(set(customers)) != len(customers):
            raise ValueError("A route's customer set must not contain duplicates.")
        mask = sum(1 << int(i) for i in customers)
        original_mask = mask
        order = []
        end = int(self.ends[mask])
        while mask:
            order.append(end)
            previous_end = int(self.parents[mask, end])
            mask ^= 1 << end
            end = previous_end
        order.reverse()
        return {
            "customer_order": order,
            "stops": ["depot", *order, "depot"] if order else [],
            "length": float(self.costs[original_mask]),
            "used": bool(order),
        }

    def assignment_cost(self, bits):
        """Route both customer groups exactly; capacity is checked by the caller."""
        mask = self._assignment_mask(bits)
        return float(self.costs[mask] + self.costs[((1 << self.n) - 1) ^ mask])

    def route_assignment(self, bits):
        """Return tours in vehicle order: bit 1 first, then bit 0."""
        self._assignment_mask(bits)
        return [
            self.tour(i for i, bit in enumerate(bits) if bit == value)
            for value in ("1", "0")
        ]

    def solve_cvrp(self, demands, capacity):
        """Prove the two-vehicle optimum by enumerating all feasible partitions.

        Distinct solutions count customer partitions, ignoring vehicle labels
        and alternative minimizing tour orders. Both bit labelings are retained
        for evaluating shot success probabilities.
        """
        demands = np.asarray(demands, dtype=float)
        capacity = float(capacity)
        if (
            demands.shape != (self.n,)
            or not np.isfinite(demands).all()
            or np.any(demands < 0)
            or not np.isfinite(capacity)
            or capacity < 0
        ):
            raise ValueError(
                "Demands and capacity must be finite and nonnegative, one demand per customer."
            )
        count = 1 << self.n
        full = count - 1
        loads = np.zeros(count)
        for mask in range(1, count):
            bit = mask & -mask
            loads[mask] = loads[mask ^ bit] + demands[bit.bit_length() - 1]
        optimum = np.inf
        candidates = []
        feasible_partitions = 0
        for mask in range(count):
            complement = full ^ mask
            if (
                mask > complement
                or loads[mask] > capacity
                or loads[complement] > capacity
            ):
                continue
            feasible_partitions += 1
            cost = float(self.costs[mask] + self.costs[complement])
            candidates.append((mask, cost))
            optimum = min(optimum, cost)
        optimal_bits = set()
        for mask, cost in candidates:
            if np.isclose(
                cost, optimum, rtol=config.COST_REL_TOL, atol=config.COST_ABS_TOL
            ):
                optimal_bits.update((self._bits(mask), self._bits(full ^ mask)))

        def canonical(bits):
            return min(bits, bits.translate(str.maketrans("01", "10")))

        optimal_clusterings = sorted({canonical(bits) for bits in optimal_bits})
        representative = optimal_clusterings[0] if optimal_clusterings else None
        return {
            "status": "proven_optimum"
            if feasible_partitions
            else "infeasible_instance",
            "method": "Held-Karp subset TSP and exhaustive two-vehicle partitions",
            "vehicles": 2,
            "depot": self.depot.tolist(),
            "optimal_cost": float(optimum) if feasible_partitions else None,
            "optimal_bitstrings": sorted(optimal_bits),
            "optimal_clusterings": optimal_clusterings,
            "feasible_clusterings": feasible_partitions,
            "best_solution": (
                {
                    "bits": representative,
                    "routes": self.route_assignment(representative),
                    "cost": self.assignment_cost(representative),
                }
                if representative is not None
                else None
            ),
        }
