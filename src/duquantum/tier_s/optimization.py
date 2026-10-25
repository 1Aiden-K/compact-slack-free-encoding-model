"""Tier S capacity-constrained clustering scorer and pulse-optimization CLI.

Score reference-reaching probability, best feasible gap, and feasibility.
The exact reference cost is used explicitly; training and final shots are separate.
Run through: python main.py optimize --instance instances/tier_s/tier_s_seed_0000.json
"""
import argparse
import json
import math
from itertools import product
from pathlib import Path
import numpy as np
from duquantum import config
from duquantum.tier_s import config as tier_config
from duquantum.quantum.optimization import optimize_pulses

class ClusteringScorer:
    """Track A: within-cluster pair distance, subject to vehicle capacities."""

    def __init__(self, coordinates, demands, capacity, *, epsilon_pct=0.0):
        coordinates = np.asarray(coordinates, dtype=float)
        if (coordinates.ndim != 2 or coordinates.shape[1] == 0
                or not np.isfinite(coordinates).all()):
            raise ValueError("Coordinates must be a finite matrix with one row per customer.")
        self.demands = np.asarray(demands, dtype=float)
        self.capacity = float(capacity)
        self.n = len(coordinates)
        if (
            self.n < 2
            or self.demands.shape != (self.n,)
            or not np.isfinite(self.demands).all()
            or np.any(self.demands < 0)
            or not np.isfinite(self.capacity)
            or self.capacity < 0
        ):
            raise ValueError("Need at least two customers and finite nonnegative demands/capacity.")
        if self.n > config.MAX_EXACT_BITS:
            raise ValueError(f"Exact clustering reference exceeds {config.MAX_EXACT_BITS} bits.")
        if not np.isfinite(epsilon_pct) or epsilon_pct < 0:
            raise ValueError("epsilon_pct must be finite and nonnegative.")
        self.epsilon_pct = float(epsilon_pct)
        self.pair_distances = np.triu(np.linalg.norm(
            coordinates[:, None, :] - coordinates[None, :, :], axis=2), 1)
        # Retain the exact constrained reference cost, not its bitstrings.
        optimum = None
        for bits in product((0, 1), repeat=self.n):
            z = np.asarray(bits)
            if self._feasible(z):
                cost = self._cost(z)
                optimum = cost if optimum is None else min(optimum, cost)
        if optimum is None:
            raise ValueError("No assignment satisfies the two vehicle capacities.")
        self.reference_cost = float(optimum)
        self.cache = {}

    def _feasible(self, z):
        return max(float(self.demands @ z), float(self.demands @ (1 - z))) <= self.capacity

    def _cost(self, z):
        return float(np.sum(self.pair_distances * (z[:, None] == z[None, :])))

    def assignment(self, bits):
        if not isinstance(bits, str) or len(bits) != self.n or set(bits) - {"0", "1"}:
            raise ValueError(f"Expected a {self.n}-bit assignment string.")
        if bits not in self.cache:
            z = np.fromiter((int(bit) for bit in bits), dtype=int)
            feasible = self._feasible(z)
            cost = self._cost(z) if feasible else None
            if not feasible:
                gap = None
            elif math.isclose(cost, self.reference_cost, rel_tol=config.COST_REL_TOL,
                              abs_tol=config.COST_ABS_TOL):
                gap = 0.0
            elif self.reference_cost == 0:
                gap = None
            else:
                gap = 100 * (cost - self.reference_cost) / self.reference_cost
            self.cache[bits] = (feasible, cost, gap)
        return self.cache[bits]

    def metrics(self, weights, *, include_best=True):
        """Rates use all weights; best gap is only meaningful for shot counts."""
        if not weights:
            raise ValueError("Cannot score an empty distribution.")
        total = 0.0
        feasible_weight = 0.0
        reached_weight = 0.0
        best_gap = None
        for bits, weight in weights.items():
            weight = float(weight)
            if not np.isfinite(weight) or weight < 0:
                raise ValueError("Weights must be finite and nonnegative.")
            feasible, _, gap = self.assignment(bits)
            if feasible:
                feasible_weight += weight
                if gap is not None and gap <= self.epsilon_pct:
                    reached_weight += weight
                if include_best and weight > 0 and gap is not None:
                    best_gap = gap if best_gap is None else min(best_gap, gap)
            total += weight
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Distribution must have positive finite weight.")
        result = {"reference_probability": reached_weight / total,
                  "feasibility_rate": feasible_weight / total}
        if include_best:
            result["best_gap_pct"] = best_gap
        return result

    def score(self, metrics, metric="reference-probability"):
        """Return scalar primary loss and a lexicographic tie-break key."""
        gap = metrics["best_gap_pct"]
        gap_rank = float("inf") if gap is None else gap
        p_ref, p_feasible = metrics["reference_probability"], metrics["feasibility_rate"]
        if metric == "reference-probability":
            return 1 - p_ref, (gap_rank, -p_feasible)
        if metric == "best-gap":
            # Strictly increasing bounded transform; no feasible shot scores 1.
            loss = 1.0 if gap is None else gap / (100 + gap)
            return loss, (-p_ref, -p_feasible)
        if metric == "feasibility":
            return 1 - p_feasible, (gap_rank, -p_ref)
        raise ValueError("Unknown optimization metric.")


def add_optimization_arguments(parser):
    """Shared controls for the integrated benchmark and emulator paths."""
    parser.add_argument("--optimize-pulses", action=argparse.BooleanOptionalAction,
                        default=True, help="optimize before final emulation (default: enabled)")
    parser.add_argument("--evaluations", type=int, default=40,
                        help="search attempts including baseline, per instance")
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--training-seed", type=int, default=10000,
                        help="training measurement seed; must differ from --sample-seed")
    parser.add_argument("--training-shots", type=int, default=1000)
    parser.add_argument("--score-mode", choices=("sampled", "probabilities"), default="sampled")
    parser.add_argument("--metric", choices=("reference-probability", "best-gap", "feasibility"),
                        default="reference-probability")
    parser.add_argument("--duration-bounds", type=int, nargs=2, metavar=("MIN_NS", "MAX_NS"))


def validate_optimization_settings(settings):
    if not settings["optimize_pulses"]:
        return
    for name in ("evaluations", "training_shots"):
        if settings[name] < 1:
            raise ValueError(f"{name} must be positive.")
    if settings["search_seed"] < 0 or settings["training_seed"] < 0:
        raise ValueError("Optimization seeds must be nonnegative.")
    if settings["training_seed"] == settings["sample_seed"]:
        raise ValueError("--training-seed must differ from --sample-seed.")
    if settings["metric"] == "best-gap" and settings["score_mode"] != "sampled":
        raise ValueError("--metric best-gap requires --score-mode sampled.")
    bounds = settings["duration_bounds"]
    if bounds is not None and not (0 < bounds[0] <= bounds[1]):
        raise ValueError("--duration-bounds must be positive and ordered.")


def optimize_program(baseline, *, coordinates, demands, capacity, settings,
                     emulator=None):
    """Select an emulated pulse; the caller must emulate it for final reporting.

    Returns the actual selected Pulser program and JSON-safe search accounting.
    Training distributions and counts never substitute for the final run.
    """
    validate_optimization_settings(settings)
    scorer = ClusteringScorer(coordinates, demands, capacity,
                         epsilon_pct=settings["epsilon_pct"])

    def progress(entry, best):
        print(f"Pulse search {entry['evaluation']}/{settings['evaluations']}: "
              f"loss={entry['loss']:.6f}; best={best['loss']:.6f}", flush=True)

    search = optimize_pulses(
        baseline, scorer, evaluations=settings["evaluations"],
        search_seed=settings["search_seed"], sample_seed=settings["training_seed"],
        training_shots=settings["training_shots"], score_mode=settings["score_mode"],
        metric=settings["metric"], backend=settings["backend"], dt=settings["dt_ns"],
        gpu=settings["gpu"], with_modulation=settings["with_modulation"],
        duration_bounds=settings["duration_bounds"], progress=progress, emulator=emulator,
    )
    best = search["best"]
    program = best["payload"]["program"]
    history = [{"evaluation": entry["evaluation"], "loss": entry["loss"],
                "parameters": entry["parameters"], "controls": entry["payload"]["controls"],
                "metrics": entry["payload"].get("metrics"),
                "predicted_metrics": entry["payload"].get("predicted_metrics"),
                "error": entry["payload"].get("error")}
               for entry in search["history"]]
    metadata = {
        "track": "A", "objective": "capacity_constrained_clustering",
        "status": search["status"], "budget": settings["evaluations"],
        "evaluations": len(history), "simulations": search["simulations"],
        "training_shots": search["simulations"] * settings["training_shots"],
        "seconds": search["seconds"], "metric": settings["metric"],
        "score_mode": settings["score_mode"], "reference_cost": scorer.reference_cost,
        "uses_reference_cost": True, "selected_evaluation": best["evaluation"],
        "baseline_retained": best is search["baseline"],
        "baseline_metrics": search["baseline"]["payload"]["metrics"],
        "selected_metrics": best["payload"]["metrics"],
        "fixed_final_detuning_rad_us": baseline["parameters"]["delta_final"],
        "bounds": search["bounds"], "history": history,
        "selected_sequence": json.loads(program["sequence"].to_abstract_repr()),
    }
    return program, metadata


def _load_cli_instance(path):
    """Bind a certified instance before importing the concrete model."""
    if path is not None:
        from duquantum.tier_s.instances import load_instance
        from duquantum.tier_s.config import bind_instance
        bind_instance(load_instance(path))


def main(argv=None):
    # Kept local so --help and scorer tests do not require Pulser or ising.py.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instance", type=Path, help="optional certified Tier S JSON; config.py is not rewritten")
    parser.add_argument("--lambda", dest="penalty_weight", type=float, default=tier_config.LAMBDA)
    parser.add_argument("--scale", type=float, default=config.ENERGY_SCALE)
    parser.add_argument("--duration", type=int, default=config.PULSE_DURATION, help="baseline duration, ns")
    parser.add_argument("--duration-bounds", type=int, nargs=2, metavar=("MIN_NS", "MAX_NS"),
                        help="must include baseline; default: half to twice baseline")
    parser.add_argument("--omega-peak", type=float, default=config.OMEGA_PEAK)
    parser.add_argument("--delta-initial", type=float, default=config.DELTA_INITIAL)
    parser.add_argument("--seed", type=int, default=config.EMBEDDING_SEED, help="embedding seed")
    parser.add_argument("--restarts", type=int, default=config.EMBEDDING_RESTARTS)
    parser.add_argument("--maxiter", type=int, default=config.MAX_ITERATIONS, help="embedding iterations per restart")
    parser.add_argument("--evaluations", type=int, default=40, help="total search evaluations INCLUDING baseline")
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=config.MEASUREMENT_SEED, help="training measurements")
    parser.add_argument("--final-seed", type=int, default=10000, help="fresh final measurements")
    parser.add_argument("--score-mode", choices=("probabilities", "sampled"), default="sampled")
    parser.add_argument("--metric", choices=("reference-probability", "best-gap", "feasibility"),
                        default="reference-probability", help="primary metric; others break ties")
    parser.add_argument("--training-shots", type=int, default=1000)
    parser.add_argument("--shots", type=int, default=config.SHOTS, help="final shots PER pulse")
    parser.add_argument("--backend", choices=("emu-sv", "qutip"), default=config.EMULATOR)
    parser.add_argument("--dt", type=int, default=1, help="emu-sv integration step, ns")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--with-modulation", action="store_true", default=config.MODULATION)
    parser.add_argument("--epsilon", type=float, default=config.EPSILON_PCT)
    parser.add_argument("--output", type=Path, default=Path("results/tier_s/optimized_run.json"))
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError(f"Output already exists: {args.output}; choose a new path.")
        for name in ("evaluations", "training_shots", "shots", "dt", "duration", "restarts", "maxiter"):
            if getattr(args, name) < 1:
                raise ValueError(f"--{name.replace('_', '-')} must be positive.")
        for name in ("seed", "search_seed", "sample_seed", "final_seed"):
            if getattr(args, name) < 0:
                raise ValueError("All seeds must be nonnegative.")
        if args.final_seed == args.sample_seed:
            raise ValueError("--final-seed must differ from --sample-seed.")
        if not np.isfinite(args.epsilon) or args.epsilon < 0:
            raise ValueError("--epsilon must be finite and nonnegative.")
        if args.gpu and args.backend != "emu-sv":
            raise ValueError("--gpu requires --backend emu-sv.")
        if args.metric == "best-gap" and args.score_mode != "sampled":
            raise ValueError("--metric best-gap requires --score-mode sampled.")
        _load_cli_instance(args.instance)
        from duquantum.quantum.pulses import build_sequence
        from duquantum.tier_s.model import build_q_matrix

        scorer = ClusteringScorer(tier_config.COORDINATES, tier_config.DEMANDS, tier_config.CAPACITY,
                             epsilon_pct=args.epsilon)
        print("Building the fixed register and original pulse...", flush=True)
        baseline = build_sequence(
            build_q_matrix(args.penalty_weight), energy_scale=args.scale,
            duration_ns=args.duration, omega_peak=args.omega_peak,
            delta_initial=args.delta_initial, seed=args.seed,
            restarts=args.restarts, maxiter=args.maxiter,
        )

        def progress(entry, best):
            error = entry["payload"].get("error")
            suffix = f"; rejected: {error}" if error else ""
            print(f"[{entry['evaluation']}/{args.evaluations}] loss={entry['loss']:.6f}; "
                  f"best={best['loss']:.6f}; metrics={entry['payload'].get('metrics')}{suffix}", flush=True)

        search = optimize_pulses(
            baseline, scorer, evaluations=args.evaluations, search_seed=args.search_seed,
            sample_seed=args.sample_seed, training_shots=args.training_shots,
            score_mode=args.score_mode, metric=args.metric, backend=args.backend, dt=args.dt, gpu=args.gpu,
            with_modulation=args.with_modulation, duration_bounds=args.duration_bounds,
            progress=progress,
        )
        # Full reference bitstrings/diagnostics are for reporting. During search
        # ClusteringScorer uses only its reference cost and sampled clustering costs.
        from duquantum.tier_s.evaluation import build_reference, encoding_from_program, evaluate_samples

        reference = build_reference(args.penalty_weight)

        def run_record(entry):
            data = entry["payload"]
            program = data["program"]
            encoding = encoding_from_program(program)
            from duquantum.quantum.emulator import emulate
            final = emulate(
                program["sequence"], shots=args.shots, seed=args.final_seed,
                backend=args.backend, dt=args.dt, gpu=args.gpu,
                with_modulation=args.with_modulation,
            )
            counts = dict(final["counts"])
            evaluation = evaluate_samples(
                counts, probabilities=final["probabilities"], penalty_weight=args.penalty_weight,
                reference=reference, encoding=encoding, epsilon_pct=args.epsilon,
            )
            return {
                "controls": data["controls"], "training_loss": entry["loss"],
                "training_metrics": data["metrics"],
                "predicted_metrics": data["predicted_metrics"],
                "final_metrics": scorer.metrics(counts),
                "counts": counts, "probabilities": final["probabilities"],
                "encoding": encoding, "evaluation": evaluation,
            }

        original = run_record(search["baseline"])
        optimized = run_record(search["best"])
        selected_program = search["best"]["payload"]["program"]
        history = [{"evaluation": entry["evaluation"], "loss": entry["loss"],
                    "parameters": entry["parameters"], "controls": entry["payload"]["controls"],
                    "metrics": entry["payload"].get("metrics"),
                    "error": entry["payload"].get("error")}
                   for entry in search["history"]]
        payload = {
            "schema_version": 3, "method": "adiabatic_good_shot_pulse_optimization",
            "track": "A", "objective": "capacity_constrained_clustering",
            "counts": optimized["counts"], "probabilities": optimized["probabilities"],
            "encoding": optimized["encoding"], "evaluation": optimized["evaluation"],
            "instance": {"coordinates": np.asarray(tier_config.COORDINATES).tolist(),
                         "demands": np.asarray(tier_config.DEMANDS).tolist(),
                         "capacity": float(tier_config.CAPACITY)},
            "settings": {"penalty_weight": args.penalty_weight,
                         "energy_scale": baseline["parameters"]["energy_scale"],
                         "backend": args.backend, "dt_ns": args.dt if args.backend == "emu-sv" else None,
                         "with_modulation": args.with_modulation, "gpu": args.gpu,
                         "embedding_seed": args.seed, "embedding_restarts": args.restarts,
                         "embedding_maxiter": args.maxiter, "search_seed": args.search_seed,
                         "sample_seed": args.sample_seed, "final_seed": args.final_seed,
                         "shots": args.shots, "training_shots": args.training_shots},
            "optimization": {"algorithm": "scipy.optimize.differential_evolution",
                             "track": "A", "objective": "capacity_constrained_clustering",
                             "status": search["status"], "budget": args.evaluations,
                             "evaluations": len(history), "simulations": search["simulations"],
                             "training_shot_budget": search["simulations"] * args.training_shots,
                             "final_simulations": 2,
                             "total_emulator_runs": search["simulations"] + 2,
                             "seconds": search["seconds"], "score_mode": args.score_mode,
                             "metric": args.metric, "reference_cost": scorer.reference_cost,
                             "epsilon_pct": args.epsilon,
                             "fixed_final_detuning_rad_us": baseline["parameters"]["delta_final"],
                             "adiabatic_following_certified": False,
                             "bounds": search["bounds"], "history": history,
                             "parameter_order": ["amplitude_multiplier", "initial_detuning_multiplier",
                                                 "log_segment_1_weight", "log_segment_2_weight",
                                                 "duration_ratio"]},
            "baseline": original, "optimized": optimized,
            "positions_um": np.asarray(baseline["embedding"]["positions_um"]).tolist(),
            "optimized_sequence": json.loads(selected_program["sequence"].to_abstract_repr()),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive create also protects against a file appearing during the search.
        text = json.dumps(payload, indent=config.JSON_INDENT, allow_nan=False) + "\n"
        with args.output.open("x") as stream:
            stream.write(text)
        print("\nBaseline -> selected pulse:")
        print(f"Primary metric: {args.metric}; training loss "
              f"{original['training_loss']:.6f} -> {optimized['training_loss']:.6f}")
        for label, run in (("Baseline", original), ("Selected", optimized)):
            predicted = run["evaluation"]["predicted"]
            print(f"{label}: feasible={predicted['feasibility_rate']:.2%}; "
                  f"clustering-optimal={predicted['optimal_probability']:.2%}; "
                  f"best sampled clustering gap={run['evaluation']['sampled']['best_model_gap_pct']}%")
        print(f"Search used {len(history)} evaluations, {search['simulations']} simulations; saved {args.output}")
    except (ValueError, ImportError, OSError) as error:
        parser.error(str(error))
    return 0
