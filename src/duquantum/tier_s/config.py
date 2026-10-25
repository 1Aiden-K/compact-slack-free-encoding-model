"""Tier S default instance and two-vehicle model settings."""

import sys

import numpy as np

COORDINATES = np.array([[0.0, 0.0], [2.0, 1.0], [6.0, 0.0], [7.0, 2.0]])
DEMANDS = np.array([1, 2, 3, 4], dtype=int)
CAPACITY = 7
LAMBDA = 0.1


def bind_instance(instance):
    """Bind before importing modules that derive arrays from this instance.

    Benchmark workers are separate processes. CLI callers use this before
    constructing a model, so another instance cannot retain derived arrays.
    """
    global COORDINATES, DEMANDS, CAPACITY
    COORDINATES = np.asarray(instance["coordinates"], dtype=float)
    DEMANDS = np.asarray(instance["demands"], dtype=float)
    CAPACITY = float(instance["capacity"])
    for name in ("duquantum.tier_s.model", "duquantum.tier_s.classical"):
        sys.modules.pop(name, None)
