"""Benchmark Tier S clustering across certified instances.

Run through: python main.py benchmark --evaluations 10
Each instance runs in a fresh process. Reports retain failures and count search costs."""

import argparse
import contextlib
import csv
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from collections import Counter
from importlib import metadata
from pathlib import Path

import numpy as np

from duquantum.tier_s.instances import load_instance
from duquantum.tier_s.config import bind_instance as _bind_instance
from duquantum.tier_s.optimization import add_optimization_arguments, validate_optimization_settings

DEFAULTS = {
    "penalty_weight": 0.1,
    "duration_ns": 4000,
    "energy_scale": None,
    "backend": "emu-sv",
    "dt_ns": 1,
    "embedding_seed": 0,
    "sample_seed": 0,
    "restarts": 8,
    "maxiter": 1500,
    "shots": 1000,
    "epsilon_pct": 0.0,
    "with_modulation": False,
    "gpu": False,
    "classical_only": False,
    "optimize_pulses": True,
    "evaluations": 40,
    "search_seed": 0,
    "training_seed": 10000,
    "training_shots": 1000,
    "score_mode": "sampled",
    "metric": "reference-probability",
    "duration_bounds": None,
}

BASELINES = ("exact_clustering", "exact_qubo", "uniform_random")

COLUMNS = [
    "instance_id",
    "instance_seed",
    "method",
    "status",
    "error_stage",
    "error",
    "load_ratio",
    "largest_demand_ratio",
    "reference_status",
    "clustering_reference_cost",
    "feasibility_rate",
    "optimal_probability",
    "reference_probability",
    "predicted_feasibility_rate",
    "predicted_optimal_probability",
    "predicted_reference_probability",
    "best_model_gap_pct",
    "best_clustering_cost",
    "target_all_ground_states_original_optimal",
    "global_all_ground_states_original_optimal",
    "embedded_all_ground_states_original_optimal",
    "sampled_embedded_ground_probability",
    "predicted_embedded_ground_probability",
    "relative_interaction_error",
    "relative_matrix_error",
    "atoms",
    "shots",
    "penalty_weight",
    "scaling_policy",
    "energy_scale",
    "duration_ns",
    "dt_ns",
    "embedding_seed",
    "sample_seed",
    "restarts",
    "maxiter",
    "with_modulation",
    "gpu",
    "reference_seconds",
    "sequence_seconds",
    "emulation_seconds",
    "pulse_optimized",
    "optimization_seconds",
    "optimization_evaluations",
    "optimization_simulations",
    "optimization_training_shots",
    "total_emulator_runs",
    "detail_file",
]


def _json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")




def assess_instance(path, settings):
    """Assess one instance in a dedicated worker, retaining classical results on emulator failure."""
    settings = {**DEFAULTS, **settings}
    if not settings["classical_only"]:
        validate_optimization_settings(settings)
    instance = load_instance(path)
    _bind_instance(instance)
    from duquantum.tier_s.evaluation import build_reference, encoding_from_program, evaluate_samples
    from duquantum.quantum.ising import build_ising

    start = time.perf_counter()
    reference = build_reference(settings["penalty_weight"])
    reference_seconds = time.perf_counter() - start
    if reference["status"] != "proven_optimum":
        raise ValueError(
            "A certified instance unexpectedly has no feasible clustering reference."
        )

    # Unit scaling is enough to check this ideal projection: multiplying the
    # entire final Hamiltonian by a positive scale cannot change its minimizers.
    encoding = {
        "global_q_matrix": build_ising(reference["q_matrix"])[
            "global_q_matrix"
        ].tolist()
    }
    program_data = None
    sequence_seconds = None
    emulation_seconds = None
    quantum = None
    quantum_error = None
    quantum_stage = None
    optimization = None
    if not settings["classical_only"]:
        try:
            quantum_stage = "sequence_construction"
            from duquantum.quantum.pulses import build_sequence

            start = time.perf_counter()
            program = build_sequence(
                reference["q_matrix"],
                energy_scale=settings["energy_scale"],
                duration_ns=settings["duration_ns"],
                seed=settings["embedding_seed"],
                restarts=settings["restarts"],
                maxiter=settings["maxiter"],
            )
            sequence_seconds = time.perf_counter() - start
            if settings["optimize_pulses"]:
                quantum_stage = "pulse_optimization"
                from duquantum.tier_s.optimization import optimize_program

                program, optimization = optimize_program(
                    program, coordinates=instance["coordinates"],
                    demands=instance["demands"], capacity=instance["capacity"],
                    settings=settings,
                )
            encoding = encoding_from_program(program)
            parameters = program["parameters"]
            program_data = {
                "positions_um": np.asarray(
                    program["embedding"]["positions_um"]
                ).tolist(),
                "energy_scale": float(parameters["energy_scale"]),
                "duration_ns": int(parameters["duration_ns"]),
                "omega_peak": float(parameters["omega_peak"]),
                "delta_initial": float(parameters["delta_initial"]),
                "delta_final": float(parameters["delta_final"]),
                "encoding": encoding,
            }
            quantum_stage = "emulation"
            from duquantum.quantum.emulator import emulate

            start = time.perf_counter()
            quantum = emulate(
                program["sequence"],
                shots=settings["shots"],
                seed=settings["sample_seed"],
                backend=settings["backend"],
                dt=settings["dt_ns"],
                with_modulation=settings["with_modulation"],
                gpu=settings["gpu"],
            )
            emulation_seconds = time.perf_counter() - start
        except Exception as error:
            quantum_error = {
                "type": type(error).__name__,
                "message": str(error),
                "stage": quantum_stage,
                "traceback": traceback.format_exc(),
            }

    def evaluation(counts, probabilities=None):
        return evaluate_samples(
            counts,
            probabilities=probabilities,
            reference=reference,
            penalty_weight=settings["penalty_weight"],
            encoding=encoding,
            epsilon_pct=settings["epsilon_pct"],
        )

    model_bits = reference["optimal_clusterings"][0]
    labels = sorted(reference["assignments"])
    uniform = np.full(len(labels), 1 / len(labels))
    draws = np.random.default_rng(settings["sample_seed"]).multinomial(
        settings["shots"], uniform
    )
    random_counts = {label: int(count) for label, count in zip(labels, draws) if count}
    model_report = evaluation({model_bits: 1})
    qubo_bits = model_report["stages"]["target_qubo"]["ground_bitstrings"][0]
    inputs = {
        "exact_clustering": ({model_bits: 1}, None),
        "exact_qubo": ({qubo_bits: 1}, None),
        "uniform_random": (random_counts, dict(zip(labels, uniform.tolist()))),
    }
    methods = {}
    for name, (counts, probabilities) in inputs.items():
        report = (
            model_report
            if name == "exact_clustering"
            else evaluation(counts, probabilities)
        )
        methods[name] = {
            "status": "ok"
            if report["sampled"]["feasible_shots"]
            else "no_feasible_samples",
            "counts": counts,
            "probabilities": probabilities,
            "report": report,
            "error": None,
        }
    if settings["classical_only"]:
        methods[settings["backend"]] = {
            "status": "not_run",
            "report": None,
            "error": {"stage": "requested_skip", "message": "--classical-only"},
        }
    elif quantum_error is not None:
        methods[settings["backend"]] = {
            "status": "failed",
            "report": None,
            "error": quantum_error,
        }
    else:
        try:
            report = evaluation(quantum["counts"], quantum["probabilities"])
            methods[settings["backend"]] = {
                "status": "ok"
                if report["sampled"]["feasible_shots"]
                else "no_feasible_samples",
                "counts": dict(quantum["counts"]),
                "probabilities": quantum["probabilities"],
                "report": report,
                "error": None,
            }
        except Exception as error:
            methods[settings["backend"]] = {
                "status": "failed",
                "report": None,
                "error": {
                    "stage": "evaluation",
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            }
    return {
        "track": "A", "objective": "capacity_constrained_clustering",
        "instance": instance,
        "settings": settings,
        "program": program_data,
        "optimization": optimization,
        "timings": {
            "reference_seconds": reference_seconds,
            "sequence_seconds": sequence_seconds,
            "emulation_seconds": emulation_seconds,
        },
        "methods": methods,
    }


def _row(payload, method, detail_file):
    instance, settings = payload["instance"], payload["settings"]
    result = payload["methods"][method]
    report = result.get("report")
    baseline = payload["methods"]["exact_clustering"].get("report")
    diagnostics = report or baseline or {}
    stages = diagnostics.get("stages", {})
    sampled = report["sampled"] if report else {}
    predicted = (report.get("predicted") or {}) if report else {}
    reference = diagnostics.get("reference", {})
    encoding = (payload.get("program") or {}).get("encoding", {})
    error = result.get("error") or {}
    is_quantum = method == settings["backend"]
    row = {
        "instance_id": f"tier_s_seed_{instance['seed']:04d}",
        "instance_seed": instance["seed"],
        "method": method,
        "status": result["status"],
        "error_stage": error.get("stage"),
        "error": error.get("message"),
        "load_ratio": instance["load_ratio"],
        "largest_demand_ratio": instance["largest_demand_ratio"],
        "reference_status": reference.get("status"),
        "clustering_reference_cost": reference.get("optimal_cost"),
        "feasibility_rate": sampled.get("feasibility_rate"),
        "optimal_probability": sampled.get("optimal_probability"),
        "reference_probability": sampled.get("reference_probability"),
        "predicted_feasibility_rate": predicted.get("feasibility_rate"),
        "predicted_optimal_probability": predicted.get("optimal_probability"),
        "predicted_reference_probability": predicted.get("reference_probability"),
        "best_model_gap_pct": sampled.get("best_model_gap_pct"),
        "best_clustering_cost": sampled.get("best_feasible_cost"),
        "target_all_ground_states_original_optimal": stages.get("target_qubo", {}).get(
            "all_ground_states_original_optimal"
        ),
        "global_all_ground_states_original_optimal": stages.get(
            "global_projection", {}
        ).get("all_ground_states_original_optimal"),
        "embedded_all_ground_states_original_optimal": stages.get("embedded", {}).get(
            "all_ground_states_original_optimal"
        ),
        "sampled_embedded_ground_probability": stages.get("embedded", {}).get(
            "sampled_ground_state_probability"
        ),
        "predicted_embedded_ground_probability": stages.get("embedded", {}).get(
            "predicted_ground_state_probability"
        ),
        "relative_interaction_error": encoding.get("relative_interaction_error"),
        "relative_matrix_error": encoding.get("relative_matrix_error"),
        "atoms": instance["n_customers"] if is_quantum else 0,
        "shots": sampled.get("shots"),
        "penalty_weight": settings["penalty_weight"],
        "scaling_policy": "automatic" if settings["energy_scale"] is None else "fixed",
        "energy_scale": (payload.get("program") or {}).get("energy_scale"),
        "duration_ns": (payload.get("program") or {}).get("duration_ns", settings["duration_ns"])
        if is_quantum else None,
        "dt_ns": settings["dt_ns"] if is_quantum and method == "emu-sv" else None,
        "embedding_seed": settings["embedding_seed"] if is_quantum else None,
        "sample_seed": settings["sample_seed"]
        if method in ("uniform_random", settings["backend"])
        else None,
        "restarts": settings["restarts"] if is_quantum else None,
        "maxiter": settings["maxiter"] if is_quantum else None,
        "with_modulation": settings["with_modulation"] if is_quantum else None,
        "gpu": settings["gpu"] if is_quantum else None,
        "reference_seconds": payload["timings"]["reference_seconds"],
        "sequence_seconds": payload["timings"]["sequence_seconds"]
        if is_quantum
        else None,
        "emulation_seconds": payload["timings"]["emulation_seconds"]
        if is_quantum
        else None,
        "pulse_optimized": settings.get("optimize_pulses", False) if is_quantum else None,
        "optimization_seconds": (payload.get("optimization") or {}).get("seconds") if is_quantum else None,
        "optimization_evaluations": (payload.get("optimization") or {}).get("evaluations") if is_quantum else None,
        "optimization_simulations": (payload.get("optimization") or {}).get("simulations") if is_quantum else None,
        "optimization_training_shots": (payload.get("optimization") or {}).get("training_shots") if is_quantum else None,
        "total_emulator_runs": ((payload.get("optimization") or {}).get("simulations", 0) + 1)
        if is_quantum and payload["timings"].get("emulation_seconds") is not None else None,
        "detail_file": detail_file,
    }
    return row


def summarize(rows):
    """Include failures and undefined-gap counts alongside available metrics."""
    output = {}
    metrics = (
        "feasibility_rate",
        "optimal_probability",
        "predicted_optimal_probability",
        "best_model_gap_pct",
    )
    for method in sorted({row["method"] for row in rows}):
        group = [row for row in rows if row["method"] == method]
        status = Counter(row["status"] for row in group)
        stats = {"total_runs": len(group), "status_counts": dict(status), "metrics": {}}
        for key in metrics:
            values = [row[key] for row in group if row[key] is not None]
            stats["metrics"][key] = {
                "defined_runs": len(values),
                "undefined_runs": len(group) - len(values),
                "mean": float(np.mean(values)) if values else None,
                "minimum": float(min(values)) if values else None,
                "maximum": float(max(values)) if values else None,
            }
        output[method] = stats
    # One diagnostic per instance, rather than counting the same reference once per method.
    cases = [row for row in rows if row["method"] == "exact_clustering"]
    output["encoding_checks"] = {
        key: {
            "checked": sum(row[key] is not None for row in cases),
            "preserved": sum(row[key] is True for row in cases),
        }
        for key in (
            "target_all_ground_states_original_optimal",
            "global_all_ground_states_original_optimal",
            "embedded_all_ground_states_original_optimal",
        )
    }
    return output


def _environment():
    versions = {}
    for name in (
        "numpy",
        "scipy",
        "pulser-core",
        "pulser-simulation",
        "emu-sv",
        "qutip",
    ):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
    }


def _worker(path):
    settings = json.loads(sys.stdin.read())
    with contextlib.redirect_stdout(sys.stderr):
        payload = assess_instance(path, settings)
    print(json.dumps(payload, allow_nan=False))


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        _worker(Path(sys.argv[2]))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    add_optimization_arguments(parser)
    parser.add_argument(
        "instances", type=Path, nargs="*", help="optional explicit instance paths"
    )
    parser.add_argument("--instances-dir", type=Path, default=Path("instances/tier_s"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/tier_s/benchmark"))
    parser.add_argument(
        "--lambda",
        dest="penalty_weight",
        type=float,
        default=DEFAULTS["penalty_weight"],
    )
    parser.add_argument(
        "--duration", dest="duration_ns", type=int, default=DEFAULTS["duration_ns"]
    )
    parser.add_argument(
        "--scale",
        dest="energy_scale",
        type=float,
        default=None,
        help="omit for frozen automatic scaling",
    )
    parser.add_argument(
        "--backend", choices=("emu-sv", "qutip"), default=DEFAULTS["backend"]
    )
    parser.add_argument("--dt", dest="dt_ns", type=int, default=DEFAULTS["dt_ns"])
    parser.add_argument(
        "--seed", dest="embedding_seed", type=int, default=DEFAULTS["embedding_seed"]
    )
    parser.add_argument("--sample-seed", type=int, default=DEFAULTS["sample_seed"])
    parser.add_argument("--restarts", type=int, default=DEFAULTS["restarts"])
    parser.add_argument("--maxiter", type=int, default=DEFAULTS["maxiter"])
    parser.add_argument("--shots", type=int, default=DEFAULTS["shots"])
    parser.add_argument(
        "--epsilon", dest="epsilon_pct", type=float, default=DEFAULTS["epsilon_pct"]
    )
    parser.add_argument("--with-modulation", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--classical-only", action="store_true", help="record emulator rows as not_run"
    )
    args = parser.parse_args()
    settings = {key: getattr(args, key) for key in DEFAULTS}
    try:
        if not settings["classical_only"]:
            validate_optimization_settings(settings)
        if (
            not np.isfinite(settings["penalty_weight"])
            or settings["penalty_weight"] <= 0
        ):
            raise ValueError("--lambda must be finite and positive.")
        for key in ("duration_ns", "dt_ns", "restarts", "maxiter", "shots"):
            if settings[key] <= 0:
                raise ValueError(f"{key} must be positive.")
        if settings["embedding_seed"] < 0 or settings["sample_seed"] < 0:
            raise ValueError("Embedding and measurement seeds must be nonnegative.")
        if not np.isfinite(settings["epsilon_pct"]) or settings["epsilon_pct"] < 0:
            raise ValueError("--epsilon must be finite and nonnegative.")
        if settings["energy_scale"] is not None and (
            not np.isfinite(settings["energy_scale"]) or settings["energy_scale"] <= 0
        ):
            raise ValueError("--scale must be finite and positive.")
        if settings["gpu"] and settings["backend"] != "emu-sv":
            raise ValueError("--gpu requires --backend emu-sv.")
        paths = args.instances or sorted(args.instances_dir.glob("tier_s_seed_*.json"))
        if not paths:
            raise ValueError("No instance files found.")
        items = [(path.resolve(), load_instance(path)) for path in paths]
        items.sort(key=lambda item: item[1]["seed"])
        if len({item[1]["seed"] for item in items}) != len(items):
            raise ValueError("Instance seeds must be unique within a benchmark suite.")
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError(
                "Output directory is nonempty; choose a new directory to preserve earlier results."
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except (ValueError, OSError, TypeError) as error:
        parser.error(str(error))

    sources = {}
    for name in (
        "duquantum.tier_s.benchmark",
        "duquantum.tier_s.instances",
        "duquantum.tier_s.model",
        "duquantum.tier_s.classical",
        "duquantum.quantum.ising",
        "duquantum.quantum.embedding",
        "duquantum.quantum.pulses",
        "duquantum.quantum.emulator",
        "duquantum.tier_s.optimization",
        "duquantum.tier_s.evaluation",
        "duquantum.config",
        "duquantum.tier_s.config",
        "duquantum.quantum.optimization",
    ):
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        if spec is not None and spec.origin is not None and Path(spec.origin).is_file():
            sources[name] = {
                "path": spec.origin,
                "sha256": hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest(),
            }
    lock = Path(__file__).resolve().parents[2] / "uv.lock"
    manifest = {
        "track": "A", "objective": "capacity_constrained_clustering",
        "settings": settings,
        "environment": _environment(),
        "sources": sources,
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest()
        if lock.exists()
        else None,
        "instances": [
            {
                "path": str(path),
                "seed": item["seed"],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path, item in items
        ],
    }
    _json(args.output_dir / "manifest.json", manifest)
    rows = []
    with (args.output_dir / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        stream.flush()
        for index, (path, item) in enumerate(items, 1):
            environment = dict(os.environ)
            result = subprocess.run(
                [sys.executable, "-m", "duquantum.tier_s.benchmark", "--worker", str(path)],
                input=json.dumps(settings),
                text=True,
                capture_output=True,
                env=environment,
                cwd=Path(__file__).resolve().parents[2],
            )
            detail = f"tier_s_seed_{item['seed']:04d}.json"
            log = args.output_dir / f"tier_s_seed_{item['seed']:04d}.log"
            log.write_text(result.stderr)
            try:
                if result.returncode:
                    raise RuntimeError(
                        f"Worker exited with code {result.returncode}; see {log.name}."
                    )
                payload = json.loads(result.stdout)
                if payload["instance"] != item:
                    raise ValueError("Instance changed during execution.")
            except (ValueError, RuntimeError, KeyError) as error:
                # A worker failure must still leave all expected method rows.
                payload = {
                    "instance": item,
                    "settings": settings,
                    "program": None,
                    "timings": dict.fromkeys(
                        ("reference_seconds", "sequence_seconds", "emulation_seconds")
                    ),
                    "methods": {
                        name: {
                            "status": "failed",
                            "report": None,
                            "error": {"stage": "worker", "message": str(error)},
                        }
                        for name in (*BASELINES, settings["backend"])
                    },
                }
            _json(args.output_dir / detail, payload)
            new_rows = [_row(payload, method, detail) for method in payload["methods"]]
            writer.writerows(new_rows)
            stream.flush()
            rows.extend(new_rows)
            _json(args.output_dir / "summary.json", summarize(rows))
            print(
                f"[{index}/{len(items)}] seed {item['seed']}: "
                + ", ".join(
                    f"{name}={value['status']}"
                    for name, value in payload["methods"].items()
                ),
                flush=True,
            )
    failures = sum(row["status"] == "failed" for row in rows)
    print(
        f"Saved {len(rows)} method rows to {args.output_dir / 'results.csv'}; failed runs: {failures}."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
