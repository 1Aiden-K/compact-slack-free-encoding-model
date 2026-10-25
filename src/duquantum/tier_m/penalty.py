"""Lightweight, instance-dependent Tier M balance-penalty heuristic.

The rule is calculated from the input data alone. It never queries an exact
clustering optimum, searches assignments, or uses a feasibility certificate.
It changes only the weight of the existing quadratic balance surrogate.
"""

import math

import numpy as np

from duquantum.tier_m import config

STRATEGY = 'adaptive_demand_pressure_v1'


def select_lambda(instance):
    """Return a positive lambda using demand pressure and geometric scale.

    lambda = clip(base * (mean_pair_distance / distance_reference)
                  * exp(slope * (sum(demands) - demand_reference)),
                  floor, ceiling).

    The distance factor gives the penalty the same distance units as the
    clustering cost. The exponential applies greater balance pressure when
    demand approaches the total capacity of the fleet. This is a heuristic,
    not a capacity constraint or a guarantee of feasible ground states.
    """
    demands = np.asarray(instance['demands'], dtype=float)
    points = np.asarray(instance['coordinates'], dtype=float)
    capacity = float(instance['capacity'])
    if (demands.ndim != 1 or len(demands) < 2 or
            points.ndim != 2 or points.shape[0] != len(demands) or
            points.shape[1] == 0 or not np.isfinite(points).all() or
            not np.isfinite(demands).all() or (demands < 0).any() or
            not math.isfinite(capacity) or capacity <= 0):
        raise ValueError('Adaptive lambda requires finite coordinates, nonnegative demands, and positive capacity.')
    if len(demands) != config.N_CUSTOMERS or capacity != config.CAPACITY:
        raise ValueError('Adaptive lambda v1 is calibrated only for Tier M (10 customers, capacity 20).')
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    mean_distance = float(distances[np.triu_indices(len(demands), 1)].mean())
    log_lambda = (math.log(config.AUTO_LAMBDA_BASE)
                  + config.AUTO_LAMBDA_DEMAND_SLOPE *
                  (float(demands.sum()) - config.AUTO_LAMBDA_DEMAND_REFERENCE))
    if mean_distance > 0:
        log_lambda += math.log(mean_distance / config.AUTO_LAMBDA_DISTANCE_REFERENCE)
    else:
        return config.AUTO_LAMBDA_MIN
    return float(np.clip(math.exp(np.clip(log_lambda, -700, 700)),
                         config.AUTO_LAMBDA_MIN, config.AUTO_LAMBDA_MAX))
