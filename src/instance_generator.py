"""Reproducible, certified-feasible Tier S CVRP instances.

    uv run --locked instance_generator.py generate --seed 0 --count 20
    uv run --locked instance_generator.py validate instances/tier_s_seed_*.json

Tier S: 4 customers, 2 vehicles, capacity 7, integer demands 1..4.
Locations are uniform in [0, 100)^2. The fixed depot defaults to (0, 0).
Independent PCG64 streams generate coordinates and demands. Demand vectors
are sampled uniformly and rejected only when exhaustive assignment search
finds none feasible. No clustering objective or emulator is consulted.

The certificate is a feasible assignment, not an optimization answer. Bit 1
means Vehicle 1; bit 0 means Vehicle 2. Unused vehicles are allowed.
Requires NumPy only. Importing this module does not generate or write files.
"""

import argparse
import json
from itertools import product
from numbers import Integral
from pathlib import Path

import numpy as np

GENERATOR_VERSION = 1
N_CUSTOMERS = 4
N_VEHICLES = 2
CAPACITY = 7
SQUARE_SIDE = 100.0
DEMAND_RANGE = (1, 4)


def _integer(value, label, minimum=0):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < minimum
    ):
        raise ValueError(f"{label} must be an integer >= {minimum}.")
    return int(value)


def _depot(value):
    point = np.asarray(value, dtype=float)
    if (
        point.shape != (2,)
        or not np.isfinite(point).all()
        or np.any((point < 0) | (point > SQUARE_SIDE))
    ):
        raise ValueError("Depot must contain two finite coordinates in [0, 100].")
    return point.tolist()


def find_feasible_assignment(demands):
    """Return one valid Tier S certificate, or None if no assignment exists.

    All 16 assignments are considered; total demand alone is not a certificate.
    The lexicographically first feasible assignment is used deterministically.
    """
    demands = list(demands)
    if len(demands) != N_CUSTOMERS:
        raise ValueError("Tier S requires exactly four demands.")
    demands = [_integer(d, "Demand", 1) for d in demands]
    if any(d > DEMAND_RANGE[1] for d in demands):
        raise ValueError("Tier S demands must be in 1..4.")
    total = sum(demands)
    for bits in product((0, 1), repeat=N_CUSTOMERS):
        load1 = sum(d * bit for d, bit in zip(demands, bits))
        loads = [load1, total - load1]
        if max(loads) <= CAPACITY:
            return {"assignment": list(bits), "loads": loads}
    return None


def generate_instance(seed, *, depot=(0.0, 0.0), max_attempts=10000):
    """Return a JSON-safe feasible instance without modifying config.py.

    Coordinate stream = SeedSequence([seed, 0]); demand stream = [seed, 1].
    Rejections affect only the demand stream. Across accepted instances,
    demands follow the uniform distribution conditioned on capacity feasibility.
    """
    seed = _integer(seed, "Seed")
    max_attempts = _integer(max_attempts, "max_attempts", 1)
    depot = _depot(depot)
    geometry_rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([seed, 0]))
    )
    demand_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 1])))
    coordinates = geometry_rng.uniform(0, SQUARE_SIDE, (N_CUSTOMERS, 2)).tolist()
    for attempt in range(1, max_attempts + 1):
        demands = demand_rng.integers(
            DEMAND_RANGE[0], DEMAND_RANGE[1] + 1, N_CUSTOMERS
        ).tolist()
        certificate = find_feasible_assignment(demands)
        if certificate is not None:
            break
    else:
        raise ValueError(
            f"Seed {seed}: no feasible demand vector after {max_attempts} attempts."
        )
    instance = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "rng": "PCG64",
        "numpy_version": np.__version__,
        "tier": "S",
        "n_customers": N_CUSTOMERS,
        "vehicles": N_VEHICLES,
        "capacity": CAPACITY,
        "square_side": SQUARE_SIDE,
        "demand_range": list(DEMAND_RANGE),
        "coordinates": coordinates,
        "demands": demands,
        "depot": depot,
        "certificate": certificate,
        "load_ratio": sum(demands) / (N_VEHICLES * CAPACITY),
        "largest_demand_ratio": max(demands) / CAPACITY,
        "demand_draw_attempts": attempt,
        "feasibility_method": "uniform demand rejection with exhaustive assignment certificate",
    }
    validate_instance(instance)
    return instance


def validate_instance(instance, *, check_reproducibility=False):
    """Check the tier, geometry, actual certificate loads, and reported ratios.

    Optional seed reproduction also detects edited data with stale seed
    metadata. NumPy version is retained as provenance, not compared as data.
    Returns None on success; malformed or inconsistent instances raise ValueError.
    """
    if not isinstance(instance, dict):
        raise ValueError("Instance must be a JSON object.")
    required = {
        "schema_version",
        "generator_version",
        "seed",
        "rng",
        "numpy_version",
        "tier",
        "n_customers",
        "vehicles",
        "capacity",
        "square_side",
        "demand_range",
        "coordinates",
        "demands",
        "depot",
        "certificate",
        "load_ratio",
        "largest_demand_ratio",
        "demand_draw_attempts",
        "feasibility_method",
    }
    missing = required - instance.keys()
    if missing:
        raise ValueError(f"Missing instance fields: {sorted(missing)}.")
    fixed = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "tier": "S",
        "n_customers": N_CUSTOMERS,
        "vehicles": N_VEHICLES,
        "capacity": CAPACITY,
        "square_side": SQUARE_SIDE,
        "demand_range": list(DEMAND_RANGE),
        "rng": "PCG64",
        "feasibility_method": "uniform demand rejection with exhaustive assignment certificate",
    }
    for key, expected in fixed.items():
        if instance[key] != expected:
            raise ValueError(f"Invalid {key}: expected {expected!r}.")
    for key in (
        "schema_version",
        "generator_version",
        "n_customers",
        "vehicles",
        "capacity",
    ):
        _integer(instance[key], key, 1)
    seed = _integer(instance["seed"], "Seed")
    _integer(instance["demand_draw_attempts"], "demand_draw_attempts", 1)
    if not isinstance(instance["numpy_version"], str) or not instance["numpy_version"]:
        raise ValueError("numpy_version must record the generation environment.")
    points = np.asarray(instance["coordinates"], dtype=float)
    if (
        points.shape != (N_CUSTOMERS, 2)
        or not np.isfinite(points).all()
        or np.any((points < 0) | (points > SQUARE_SIDE))
    ):
        raise ValueError(
            "Customer coordinates must have shape (4, 2) and lie in [0, 100]^2."
        )
    _depot(instance["depot"])
    demands = instance["demands"]
    if not isinstance(demands, list) or len(demands) != N_CUSTOMERS:
        raise ValueError("Tier S requires four demands.")
    demands = [_integer(d, "Demand", 1) for d in demands]
    if max(demands) > DEMAND_RANGE[1]:
        raise ValueError("Tier S demands must be in 1..4.")
    certificate = instance["certificate"]
    if not isinstance(certificate, dict):
        raise ValueError("Certificate must be an assignment-and-loads object.")
    bits = certificate.get("assignment")
    if not isinstance(bits, list) or len(bits) != N_CUSTOMERS:
        raise ValueError("Certificate must assign all four customers.")
    bits = [_integer(bit, "Assignment bit") for bit in bits]
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("Certificate assignments must be binary.")
    loads = [
        sum(d * bit for d, bit in zip(demands, bits)),
        sum(d * (1 - bit) for d, bit in zip(demands, bits)),
    ]
    if max(loads) > CAPACITY:
        raise ValueError("Certificate violates vehicle capacity.")
    reported_loads = certificate.get("loads")
    if not isinstance(reported_loads, list) or len(reported_loads) != N_VEHICLES:
        raise ValueError("Certificate must report both vehicle loads.")
    reported_loads = [_integer(load, "Certificate load") for load in reported_loads]
    if loads != reported_loads:
        raise ValueError(
            "Reported certificate loads differ from the actual demand sums."
        )
    for key, expected in (
        ("load_ratio", sum(demands) / (N_VEHICLES * CAPACITY)),
        ("largest_demand_ratio", max(demands) / CAPACITY),
    ):
        value = instance[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or not np.isclose(value, expected, rtol=0, atol=1e-12)
        ):
            raise ValueError(f"Incorrect {key}.")
    if check_reproducibility:
        regenerated = generate_instance(seed, depot=instance["depot"])
        for key in required - {"numpy_version"}:
            if instance[key] != regenerated[key]:
                raise ValueError(
                    f"Seed reproduction failed for {key}; preserve the generator version/environment."
                )


def load_instance(path, *, check_reproducibility=True):
    """Load and validate an instance for use in config.py or a benchmark runner."""
    instance = json.loads(Path(path).read_text())
    validate_instance(instance, check_reproducibility=check_reproducibility)
    return instance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="generate consecutive seeds")
    generate.add_argument("--seed", type=int, required=True, help="first instance seed")
    generate.add_argument("--count", type=int, default=1)
    generate.add_argument("--depot", type=float, nargs=2, default=(0.0, 0.0))
    generate.add_argument("--output-dir", type=Path, default=Path("instances"))
    validate = commands.add_parser(
        "validate", help="check certificates and reproduce from seeds"
    )
    validate.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "generate":
            _integer(args.count, "Count", 1)
            _integer(args.seed, "Seed")
            items = []
            for seed in range(args.seed, args.seed + args.count):
                instance = generate_instance(seed, depot=args.depot)
                path = args.output_dir / f"tier_s_seed_{seed:04d}.json"
                text = json.dumps(instance, indent=2, allow_nan=False) + "\n"
                # Preflight the entire batch before writing any output.
                if path.exists() and path.read_text() != text:
                    raise ValueError(
                        f"{path} already exists with different data; choose a new output directory."
                    )
                items.append((path, text, instance))
            args.output_dir.mkdir(parents=True, exist_ok=True)
            for path, text, instance in items:
                if not path.exists():
                    path.write_text(text)
                print(
                    f"{path}: demands={instance['demands']}; certificate loads={instance['certificate']['loads']}; "
                    f"rho={instance['load_ratio']:.6f}; dmax/Q={instance['largest_demand_ratio']:.6f}"
                )
        else:
            for path in args.files:
                instance = load_instance(path)
                print(
                    f"VALID {path}: seed={instance['seed']}; certificate and reproduction checked"
                )
    except (ValueError, OSError, TypeError, OverflowError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
