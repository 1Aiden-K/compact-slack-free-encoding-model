"""Seeded, capacity-certified Tier M instances; no optimization selects inputs."""
import argparse
import json
from pathlib import Path
from numbers import Integral, Real

import numpy as np

from duquantum.tier_m import config

GENERATOR_VERSION = 1


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _point(value, name):
    point = np.asarray(value, dtype=float)
    if (point.shape != (2,) or not np.isfinite(point).all()
            or np.any((point < 0) | (point > config.SQUARE_SIDE))):
        raise ValueError(f"{name} must lie in [0, {config.SQUARE_SIDE}]^2.")
    return point.tolist()


def find_feasible_assignment(demands):
    """Return a packing certificate, or None; total demand alone is not proof.

    Backtrack in descending demand order, pruning equivalent vehicle loads.
    No distance or clustering objective enters this feasibility search.
    """
    if len(demands) != config.N_CUSTOMERS:
        raise ValueError("Tier M requires ten demands.")
    demands = [_integer(d, 'Demand', config.DEMAND_RANGE[0]) for d in demands]
    if max(demands) > config.DEMAND_RANGE[1]:
        raise ValueError("Tier M demands must lie in 1..8.")
    if sum(demands) > config.N_VEHICLES * config.CAPACITY:
        return None
    order = sorted(range(len(demands)), key=lambda i: (-demands[i], i))
    loads, labels = [0] * config.N_VEHICLES, [0] * len(demands)

    def assign(position):
        if position == len(order):
            return True
        customer = order[position]
        seen_loads = set()
        for vehicle in range(config.N_VEHICLES):
            if loads[vehicle] in seen_loads:
                continue
            seen_loads.add(loads[vehicle])
            if loads[vehicle] + demands[customer] <= config.CAPACITY:
                loads[vehicle] += demands[customer]
                labels[customer] = vehicle + 1
                if assign(position + 1):
                    return True
                loads[vehicle] -= demands[customer]
                labels[customer] = 0
        return False

    return {'assignment': labels, 'loads': loads} if assign(0) else None


def generate_instance(seed, *, depot=(0., 0.), max_attempts=10000):
    seed = _integer(seed, 'Seed')
    max_attempts = _integer(max_attempts, 'max_attempts', 1)
    depot = _point(depot, 'Depot metadata')
    geometry = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 0])))
    demand_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 1])))
    coordinates = geometry.uniform(0, config.SQUARE_SIDE, (config.N_CUSTOMERS, 2)).tolist()
    for attempt in range(1, max_attempts + 1):
        demands = demand_rng.integers(config.DEMAND_RANGE[0], config.DEMAND_RANGE[1] + 1,
                                     config.N_CUSTOMERS).tolist()
        certificate = find_feasible_assignment(demands)
        if certificate is not None:
            break
    else:
        raise ValueError(f"Seed {seed}: no feasible instance after {max_attempts} attempts.")
    return {
        'schema_version': 1, 'generator_version': GENERATOR_VERSION,
        'tier': 'M', 'seed': seed, 'rng': 'PCG64', 'numpy_version': np.__version__,
        'n_customers': config.N_CUSTOMERS, 'vehicles': config.N_VEHICLES,
        'capacity': config.CAPACITY, 'demand_range': list(config.DEMAND_RANGE),
        'square_side': config.SQUARE_SIDE, 'coordinates': coordinates,
        'demands': demands, 'depot': depot, 'certificate': certificate,
        'load_ratio': sum(demands) / (config.N_VEHICLES * config.CAPACITY),
        'largest_demand_ratio': max(demands) / config.CAPACITY,
        'demand_draw_attempts': attempt,
        'feasibility_method': 'uniform demand rejection with exact packing certificate',
    }


def validate_instance(instance, *, check_reproducibility=False):
    if not isinstance(instance, dict):
        raise ValueError("Instance must be a JSON object.")
    fixed = {
        'schema_version': 1, 'generator_version': GENERATOR_VERSION, 'tier': 'M',
        'n_customers': config.N_CUSTOMERS, 'vehicles': config.N_VEHICLES,
        'capacity': config.CAPACITY, 'demand_range': list(config.DEMAND_RANGE),
        'square_side': config.SQUARE_SIDE, 'rng': 'PCG64',
        'feasibility_method': 'uniform demand rejection with exact packing certificate',
    }
    for name, expected in fixed.items():
        if name not in instance or instance[name] != expected or isinstance(instance[name], bool):
            raise ValueError(f"Invalid {name}; expected {expected!r}.")
    for name in ('schema_version', 'generator_version', 'n_customers', 'vehicles', 'capacity'):
        _integer(instance[name], name, 1)
    required = {'seed', 'numpy_version', 'coordinates', 'demands', 'depot', 'certificate',
                'load_ratio', 'largest_demand_ratio', 'demand_draw_attempts'}
    if required - instance.keys():
        raise ValueError(f"Missing fields: {sorted(required - instance.keys())}.")
    seed = _integer(instance['seed'], 'Seed')
    _integer(instance['demand_draw_attempts'], 'demand_draw_attempts', 1)
    if not isinstance(instance['numpy_version'], str) or not instance['numpy_version']:
        raise ValueError("numpy_version must record the generation environment.")
    points = np.asarray(instance['coordinates'], dtype=float)
    if (points.shape != (config.N_CUSTOMERS, 2) or not np.isfinite(points).all()
            or np.any((points < 0) | (points > config.SQUARE_SIDE))):
        raise ValueError("Tier M requires ten finite points in [0, 100]^2.")
    _point(instance['depot'], 'Depot metadata')
    demands = instance['demands']
    if not isinstance(demands, list) or len(demands) != config.N_CUSTOMERS:
        raise ValueError("Tier M requires ten demands.")
    demands = [_integer(d, 'Demand', 1) for d in demands]
    if max(demands) > config.DEMAND_RANGE[1]:
        raise ValueError("Tier M demands must lie in 1..8.")
    certificate = instance['certificate']
    if not isinstance(certificate, dict):
        raise ValueError("Certificate must be an assignment-and-loads object.")
    labels, reported = certificate.get('assignment'), certificate.get('loads')
    if not isinstance(labels, list) or len(labels) != config.N_CUSTOMERS:
        raise ValueError("Certificate must assign all ten customers.")
    labels = [_integer(label, 'Vehicle label', 1) for label in labels]
    if max(labels) > config.N_VEHICLES:
        raise ValueError("Certificate vehicle labels must be in 1, 2, 3.")
    if not isinstance(reported, list) or len(reported) != config.N_VEHICLES:
        raise ValueError("Certificate must report all three loads.")
    reported = [_integer(load, 'Load') for load in reported]
    loads = [sum(d for d, label in zip(demands, labels) if label == k) for k in (1, 2, 3)]
    if loads != reported or max(loads) > config.CAPACITY:
        raise ValueError("Certificate loads are incorrect or violate capacity.")
    for name, expected in (
        ('load_ratio', sum(demands) / (config.N_VEHICLES * config.CAPACITY)),
        ('largest_demand_ratio', max(demands) / config.CAPACITY),
    ):
        value = instance[name]
        if (isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value)
                or not np.isclose(value, expected, rtol=0, atol=1e-12)):
            raise ValueError(f"Incorrect {name}.")
    if check_reproducibility:
        regenerated = generate_instance(seed, depot=instance['depot'])
        for name in regenerated.keys() - {'numpy_version'}:
            if instance[name] != regenerated[name]:
                raise ValueError(f"Seed reproduction failed for {name}.")


def load_instance(path, *, check_reproducibility=True):
    instance = json.loads(Path(path).read_text())
    validate_instance(instance, check_reproducibility=check_reproducibility)
    return instance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    generate = commands.add_parser('generate')
    generate.add_argument('--seed', type=int, required=True)
    generate.add_argument('--count', type=int, default=1)
    generate.add_argument('--output-dir', type=Path, default=Path('instances/tier_m'))
    generate.add_argument('--depot', type=float, nargs=2, default=(0., 0.))
    validate = commands.add_parser('validate')
    validate.add_argument('files', type=Path, nargs='+')
    args = parser.parse_args()
    try:
        if args.command == 'generate':
            _integer(args.seed, 'Seed')
            _integer(args.count, 'Count', 1)
            pending = []
            for seed in range(args.seed, args.seed + args.count):
                instance = generate_instance(seed, depot=args.depot)
                path = args.output_dir / f'tier_m_seed_{seed:04d}.json'
                text = json.dumps(instance, indent=2, allow_nan=False) + '\n'
                if path.exists() and path.read_text() != text:
                    raise ValueError(f"{path} already exists with different data.")
                pending.append((path, text))
            args.output_dir.mkdir(parents=True, exist_ok=True)
            for path, text in pending:
                if not path.exists():
                    path.write_text(text)
                print(path)
        else:
            for path in args.files:
                load_instance(path)
                print(f'VALID {path}: certificate and seed reproduction checked')
    except (ValueError, OSError, TypeError) as error:
        parser.error(str(error))
