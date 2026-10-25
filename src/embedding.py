"""Fit positive target Rydberg interactions to a constrained 2D register.

All interactions are in rad/us and all positions/distances in um.
Pass ising.build_ising(M)['interaction_matrix'], NOT the symmetric Q matrix:
the physical pair coefficients already include its factor of two.

The minimized squared relative error is
    sum_{i<j}(V_real[i,j] - V_target[i,j])**2 / sum_{i<j}V_target[i,j]**2.
Every pair is included, even if its target interaction is zero. The reported
relative_interaction_error is the square root of this expression.

Source: https://pulser.readthedocs.io/en/stable/tutorials/qubo.html
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform


def realized_interactions(positions_um, interaction_coeff):
    """Return symmetric V_ij = interaction_coeff / R_ij**6, zero diagonal."""
    positions = np.asarray(positions_um, dtype=float)
    coeff = float(interaction_coeff)
    if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) < 2:
        raise ValueError("positions_um must have shape (N, 2), N >= 2.")
    if not np.isfinite(positions).all() or not np.isfinite(coeff) or coeff <= 0:
        raise ValueError(
            "Positions must be finite and interaction_coeff positive and finite."
        )
    distances = pdist(positions)
    if np.any(distances <= 0):
        raise ValueError("Distinct atoms cannot occupy the same position.")
    return squareform(coeff / distances**6)


def optimize_positions(
    target_interactions,
    *,
    interaction_coeff,
    min_atom_distance=5.0,
    max_radial_distance=38.0,
    seed=0,
    restarts=8,
    maxiter=1500,
):
    """Numerically fit a legal planar layout; requires NumPy and SciPy only.

    Uses reproducible multi-start SLSQP with explicit pair-distance, radial,
    and zero-centroid constraints. It minimizes interaction error, without
    consulting the original optimal assignment. Intended for small registers.
    A small geometric margin avoids rounding a boundary onto an illegal value.
    This is a local optimization: the result need not be the best possible fit.
    """
    if np.iscomplexobj(target_interactions):
        raise ValueError("Target interactions must be real.")
    target = np.array(target_interactions, dtype=float, copy=True)
    if target.ndim != 2 or target.shape[0] != target.shape[1] or len(target) < 2:
        raise ValueError("Expected a square target interaction matrix with N >= 2.")
    if not np.isfinite(target).all() or not np.allclose(
        target, target.T, rtol=1e-10, atol=1e-12
    ):
        raise ValueError("Target interactions must be finite and symmetric.")
    if np.any(target < 0) or np.any(np.diag(target) != 0):
        raise ValueError(
            "Target interactions must be nonnegative with a zero diagonal."
        )
    target = (target + target.T) / 2
    for name, value in (("restarts", restarts), ("maxiter", maxiter)):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer.")
    coeff, min_sep, radius = map(
        float, (interaction_coeff, min_atom_distance, max_radial_distance)
    )
    if not all(np.isfinite(v) and v > 0 for v in (coeff, min_sep, radius)):
        raise ValueError(
            "Interaction coefficient, minimum spacing, and radius must be positive and finite."
        )

    n = len(target)
    pairs = np.triu_indices(n, 1)
    desired = target[pairs]
    denominator = float(desired @ desired)
    if denominator == 0:
        raise ValueError("An all-zero target has no relative-error scale.")
    gap = min_sep * 1e-4
    lower, upper = min_sep + gap, radius - gap
    if upper <= 0 or lower > 2 * upper:
        raise ValueError(
            "No two atoms fit inside the requested radius at the required spacing."
        )
    rng = np.random.default_rng(seed)

    def objective(flat):
        distances = np.maximum(pdist(flat.reshape(n, 2)), min_sep * 0.1)
        residual = coeff / distances**6 - desired
        return float(residual @ residual / denominator)

    def legal(positions):
        return (
            pdist(positions).min() >= min_sep + gap / 2
            and np.linalg.norm(positions, axis=1).max() <= radius - gap / 2
        )

    constraints = [
        {"type": "ineq", "fun": lambda x: pdist(x.reshape(n, 2)) - lower},
        {
            "type": "ineq",
            "fun": lambda x: upper - np.linalg.norm(x.reshape(n, 2), axis=1),
        },
        {"type": "eq", "fun": lambda x: x.reshape(n, 2).sum(axis=0)},
    ]
    positive = desired[desired > 0]
    characteristic_distance = float(np.median((coeff / positive) ** (1 / 6)))

    def random_start():
        # Random placements at a target-informed scale; reject close pairs.
        disk_radius = min(upper * 0.45, max(characteristic_distance, lower))
        for _ in range(200):
            angles = rng.uniform(0, 2 * np.pi, n)
            lengths = disk_radius * np.sqrt(rng.random(n))
            positions = np.column_stack(
                (lengths * np.cos(angles), lengths * np.sin(angles))
            )
            positions -= positions.mean(axis=0)
            if legal(positions):
                return positions
        # A polygon is a deterministic feasible fallback for small N.
        polygon_radius = max(
            lower / (2 * np.sin(np.pi / n)), characteristic_distance / np.sqrt(2)
        )
        polygon_radius = min(polygon_radius, upper)
        angles = 2 * np.pi * np.arange(n) / n + rng.uniform(0, 2 * np.pi)
        positions = polygon_radius * np.column_stack((np.cos(angles), np.sin(angles)))
        positions = positions[rng.permutation(n)]
        positions -= positions.mean(axis=0)
        if not legal(positions):
            raise ValueError(
                "Could not construct a legal starting layout; reduce register size or adjust scale."
            )
        return positions

    # Classical multidimensional scaling gives an informed first placement.
    distances = np.zeros_like(target)
    distances[pairs] = (coeff / np.maximum(desired, positive.min() * 1e-3)) ** (1 / 6)
    distances += distances.T
    centering = np.eye(n) - np.ones((n, n)) / n
    eigenvalues, eigenvectors = np.linalg.eigh(
        -0.5 * centering @ distances**2 @ centering
    )
    initial = eigenvectors[:, -2:] * np.sqrt(np.maximum(eigenvalues[-2:], 0))
    if pdist(initial).min() > 0:
        initial *= max(1.0, lower * 1.001 / pdist(initial).min())

    best = None
    for restart in range(int(restarts)):
        start = initial.copy() if restart == 0 and legal(initial) else random_start()
        result = minimize(
            objective,
            start.ravel(),
            method="SLSQP",
            bounds=[(-upper, upper)] * (2 * n),
            constraints=constraints,
            options={"maxiter": int(maxiter), "ftol": 1e-12},
        )
        fitted = result.x.reshape(n, 2)
        fitted -= fitted.mean(axis=0)
        # Preserve a legal start if an optimizer exits at an invalid point.
        candidates = [(start, False, "Legal initialization", 0)]
        if legal(fitted):
            candidates.append(
                (fitted, bool(result.success), str(result.message), int(result.nit))
            )
        for positions, converged, message, iterations in candidates:
            score = objective(positions.ravel())
            if best is None or score < best[0]:
                best = (score, positions.copy(), converged, message, iterations)

    score, positions, converged, message, iterations = best
    realized = realized_interactions(positions, coeff)
    return {
        "positions_um": positions,
        "target_interactions": target,
        "realized_interactions": realized,
        "relative_interaction_error": float(np.sqrt(score)),
        "min_separation_um": float(pdist(positions).min()),
        "max_radius_um": float(np.linalg.norm(positions, axis=1).max()),
        "seed": seed,
        "restarts": int(restarts),
        "optimizer_converged": converged,
        "optimizer_message": message,
        "optimizer_iterations": iterations,
    }


def embed_register(
    target_interactions, *, device=None, seed=0, restarts=8, maxiter=1500
):
    """Fit positions using device limits, construct and validate a Pulser Register.

    Register IDs q0, q1, ... follow the Q-matrix row order. No atom relabeling
    is performed after fitting. The physical coefficient comes from the device.
    """
    from pulser import AnalogDevice, Register

    if device is None:
        device = AnalogDevice
    n = len(target_interactions)
    if device.max_atom_num is not None and n > device.max_atom_num:
        raise ValueError(
            "The target uses more atoms than the selected device supports."
        )
    result = optimize_positions(
        target_interactions,
        interaction_coeff=device.interaction_coeff,
        min_atom_distance=device.min_atom_distance,
        max_radial_distance=device.max_radial_distance,
        seed=seed,
        restarts=restarts,
        maxiter=maxiter,
    )
    result["register"] = Register(
        {f"q{i}": position for i, position in enumerate(result["positions_um"])}
    )
    device.validate_register(result["register"])
    # Use the register's stored coordinates as the authoritative physical layout.
    positions = np.array([result["register"].qubits[f"q{i}"] for i in range(n)])
    result["positions_um"] = positions
    result["realized_interactions"] = realized_interactions(
        positions, device.interaction_coeff
    )
    result["min_separation_um"] = float(pdist(positions).min())
    result["max_radius_um"] = float(np.linalg.norm(positions, axis=1).max())
    pair_indices = np.triu_indices(n, 1)
    residual = (result["realized_interactions"] - result["target_interactions"])[
        pair_indices
    ]
    result["relative_interaction_error"] = float(
        np.linalg.norm(residual)
        / np.linalg.norm(result["target_interactions"][pair_indices])
    )
    return result
