"""Load-balancing QUBO for two vehicles, with one bit per location.

Requires numpy and config.py defining COORDINATES, DEMANDS, and CAPACITY.
No optimization runs when this module is imported.

Standard form (J is strictly upper triangular):
    F(z) = c + h @ z + z @ J @ z
Symmetric matrix form:
    F(z) = c + z @ M @ z
Direct model:
    F(z) = C(z) + 2 * penalty_weight * (DEMANDS @ z - T / 2)**2

Capacity is checked separately; it does not enter these QUBO coefficients.
"""

import numpy as np

from config import CAPACITY, COORDINATES, DEMANDS

COORDINATES = np.asarray(COORDINATES, dtype=float)
DEMANDS = np.asarray(DEMANDS, dtype=float)
CAPACITY = float(CAPACITY)

if DEMANDS.ndim != 1 or DEMANDS.size == 0:
    raise ValueError("DEMANDS must be a nonempty one-dimensional array.")
N = N_BITS = len(DEMANDS)
if COORDINATES.ndim != 2 or COORDINATES.shape[0] != N or COORDINATES.shape[1] == 0:
    raise ValueError("COORDINATES must have one nonempty coordinate row per demand.")
if not np.isfinite(COORDINATES).all():
    raise ValueError("COORDINATES must be finite.")
if not np.isfinite(DEMANDS).all() or np.any(DEMANDS < 0):
    raise ValueError("DEMANDS must be finite and nonnegative.")
if not np.isfinite(CAPACITY) or CAPACITY < 0:
    raise ValueError("CAPACITY must be finite and nonnegative.")

T = float(DEMANDS.sum())
DISTANCES = np.linalg.norm(COORDINATES[:, None, :] - COORDINATES[None, :, :], axis=2)


def build_qubo(penalty_weight=0.1):
    """Return (c, h, J) for F(z) = c + h @ z + z @ J @ z.

    J[i, j] contains the full pair coefficient for i < j; all other
    entries are zero. The default weight is a starting value, not a guarantee
    of capacity feasibility or preservation of the original optimum.
    """
    lam = float(penalty_weight)
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("penalty_weight must be finite and strictly positive.")
    c = float(np.triu(DISTANCES, 1).sum() + lam * T**2 / 2)
    h = -DISTANCES.sum(axis=1) + 2 * lam * (DEMANDS**2 - T * DEMANDS)
    J = np.triu(2 * DISTANCES + 4 * lam * np.outer(DEMANDS, DEMANDS), 1)
    return c, h, J


def build_q_matrix(penalty_weight=0.1):
    """Return symmetric M for F(z) = c + z @ M @ z.

    Use build_qubo(penalty_weight)[0] for the constant offset c.
    Off-diagonal entries are half the corresponding pair coefficients.
    """
    _, h, J = build_qubo(penalty_weight)
    return np.diag(h) + (J + J.T) / 2


def evaluate_assignment(z):
    """Return original clustering cost, vehicle loads, and capacity feasibility.

    Bit 1 means Vehicle 1; bit 0 means Vehicle 2. No slack bits are used.
    """
    z = np.asarray(z)
    if z.shape != (N,) or not np.isin(z, [0, 1]).all():
        raise ValueError(f"Expected exactly {N} binary assignment values.")
    same_vehicle = z[:, None] == z[None, :]
    cost = float(np.triu(DISTANCES * same_vehicle, 1).sum())
    load1 = float(DEMANDS @ z)
    load2 = float(DEMANDS @ (1 - z))
    feasible = load1 <= CAPACITY and load2 <= CAPACITY
    return cost, (load1, load2), feasible
