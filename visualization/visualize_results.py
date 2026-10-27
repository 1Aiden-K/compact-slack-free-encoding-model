#!/usr/bin/env python3
"""Turn a DuQuantum Tier S benchmark results.csv into live-demo charts.

Usage (from repository root):
    uv run --with matplotlib python visualize_results.py \
        src/results/tier_s/submission_01/results.csv \
        --output demo_charts

Open demo_charts/index.html and use left/right arrow keys to change charts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")  # Works without a display, including CI and headless shells.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BG = "#101827"
PANEL = "#162336"
FG = "#F3F7FD"
MUTED = "#AAB9CC"
GRID = "#334257"
QUANTUM = "#55D6C2"
RANDOM = "#A7AEC2"
FAIL = "#F3AD6B"


def numeric(value):
    try:
        return float(value) if str(value).strip() else None
    except (ValueError, TypeError):
        return None


def pct(value):
    value = numeric(value)
    return None if value is None else 100 * value


def percentage(values):
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def display_pct(value, digits=1):
    return "N/A" if value is None else f"{value:.{digits}f}%"


def load_data(path, method, random_method):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        records = list(csv.DictReader(stream))
    if not records:
        raise ValueError("The input CSV has no data rows.")
    required = {
        "method",
        "instance_seed",
        "reference_probability",
        "feasibility_rate",
        "cvrp_reference_probability",
        "best_cvrp_gap_pct",
        "relative_interaction_error",
        "embedded_all_ground_states_original_optimal",
    }
    missing = sorted(required - records[0].keys())
    if missing:
        raise ValueError("Missing required CSV columns: " + ", ".join(missing))

    def select(name):
        items = [
            row
            for row in records
            if row["method"] == name and row["status"] in ("ok", "no_feasible_samples")
        ]
        return {int(row["instance_seed"]): row for row in items}

    quantum = select(method)
    baseline = select(random_method)
    if not quantum:
        raise ValueError(
            f"No successful '{method}' rows. Available methods: "
            + ", ".join(sorted({r["method"] for r in records}))
        )
    if not baseline:
        raise ValueError(f"No successful '{random_method}' rows.")
    seeds = sorted(set(quantum) & set(baseline))
    if not seeds:
        raise ValueError(
            "The quantum and random methods have no common instance seeds."
        )
    if len(seeds) != len(quantum) or len(seeds) != len(baseline):
        print(f"Note: comparing only the {len(seeds)} seeds shared by both methods.")
    return seeds, [quantum[s] for s in seeds], [baseline[s] for s in seeds]


def setup_plot():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "text.color": FG,
            "axes.labelcolor": FG,
            "axes.edgecolor": GRID,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "savefig.facecolor": BG,
            "axes.titleweight": "bold",
            "grid.color": GRID,
            "grid.alpha": 0.6,
        }
    )


def canvas(title, subtitle, footnote, dpi):
    fig = plt.figure(figsize=(16, 9), dpi=dpi)
    fig.text(0.06, 0.935, title, fontsize=29, color=FG, weight="bold", va="top")
    fig.text(0.06, 0.877, subtitle, fontsize=14, color=MUTED, va="top")
    fig.text(0.06, 0.045, footnote, fontsize=12, color=MUTED, va="bottom")
    return fig


def save(fig, output, dpi):
    fig.savefig(output, dpi=dpi, facecolor=BG)
    plt.close(fig)
    print(f"  {output.name}")


def comparison(seeds, q, rnd, output, dpi, method):
    labels = ["Capacity feasible", "Clustering optimum", "CVRP optimum"]
    keys = ["feasibility_rate", "reference_probability", "cvrp_reference_probability"]
    qvalues = [percentage([pct(x[k]) for x in q]) for k in keys]
    rvalues = [percentage([pct(x[k]) for x in rnd]) for k in keys]
    fig = canvas(
        "Quantum vs. random sampling",
        f"Tier S | {len(seeds)} instances | mean per-shot probabilities | {method} vs uniform random",
        "Each method samples the same certified instances. All percentages are across individual shots.",
        dpi,
    )
    ax = fig.add_axes([0.23, 0.19, 0.69, 0.58])
    y = [2, 1, 0]
    for i, position in enumerate(y):
        a, b = qvalues[i], rvalues[i]
        if a is not None:
            ax.barh(
                position + 0.19,
                a,
                height=0.34,
                color=QUANTUM,
                label=method if i == 0 else None,
            )
            ax.text(
                min(a + 1.0, 102),
                position + 0.19,
                display_pct(a),
                va="center",
                fontsize=16,
                color=FG,
            )
        if b is not None:
            ax.barh(
                position - 0.19,
                b,
                height=0.34,
                color=RANDOM,
                label="Uniform random" if i == 0 else None,
            )
            ax.text(
                min(b + 1.0, 102),
                position - 0.19,
                display_pct(b),
                va="center",
                fontsize=16,
                color=FG,
            )
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Probability (%)", labelpad=14)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=14)
    ax.legend(frameon=False, loc="lower right", labelcolor=FG, fontsize=15)
    save(fig, output, dpi)


def seed_probabilities(seeds, q, rnd, output, dpi):
    fig = canvas(
        "Optimal clustering probability by instance",
        "High and low quantum-success regimes appear even when feasibility is consistently high",
        "Teal = every embedded ground state is clustering-optimal; amber = at least one is not.",
        dpi,
    )
    ax = fig.add_axes([0.09, 0.21, 0.85, 0.57])
    qval = [pct(x["reference_probability"]) for x in q]
    rval = [pct(x["reference_probability"]) for x in rnd]
    xoffset = 0.18
    preservation = [
        x["embedded_all_ground_states_original_optimal"] == "True" for x in q
    ]
    ax.plot(
        [s - xoffset for s in seeds],
        rval,
        linestyle="--",
        color=RANDOM,
        alpha=0.65,
        linewidth=1.8,
    )
    ax.scatter([s - xoffset for s in seeds], rval, s=68, color=RANDOM, zorder=3)
    for seed, val, preserved in zip(seeds, qval, preservation):
        if val is not None:
            ax.scatter(
                seed + xoffset,
                val,
                marker="o",
                s=130,
                facecolor=QUANTUM if preserved else FAIL,
                edgecolor=BG,
                linewidth=1,
                zorder=4,
            )
    ax.set_xlim(min(seeds) - 0.7, max(seeds) + 0.7)
    ax.set_ylim(-5, 105)
    ax.set_xticks(seeds)
    ax.set_xlabel("Instance seed", labelpad=12)
    ax.set_ylabel("Clustering-reference probability (%)", labelpad=11)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=QUANTUM,
                markersize=12,
                label="Quantum: ground states preserved",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=FAIL,
                markersize=12,
                label="Quantum: not preserved",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color=RANDOM,
                markerfacecolor=RANDOM,
                linestyle="--",
                markersize=8,
                label="Uniform random",
            ),
        ],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.12),
        frameon=False,
        labelcolor=FG,
        fontsize=11,
    )
    save(fig, output, dpi)


def ground_states(seeds, q, output, dpi):
    stages = [
        ("Target QUBO", "target_all_ground_states_original_optimal"),
        ("Global detuning", "global_all_ground_states_original_optimal"),
        ("Physical embedding", "embedded_all_ground_states_original_optimal"),
    ]
    stage_counts = [sum(r.get(key) == "True" for r in q) for _, key in stages]
    groups = {True: [], False: []}
    for row in q:
        value = pct(row["reference_probability"])
        if value is not None:
            groups[row["embedded_all_ground_states_original_optimal"] == "True"].append(
                value
            )
    fig = canvas(
        "Does the embedding preserve the optimum?",
        "Ground-state diagnostics versus measured probability of reaching the clustering reference",
        "Counts show the cases where ALL ground states at each stage are clustering-optimal; sampling success is not guaranteed.",
        dpi,
    )
    ax1 = fig.add_axes([0.18, 0.26, 0.36, 0.50])
    y = [2, 1, 0]
    ax1.barh(y, stage_counts, color=[QUANTUM, RANDOM, FAIL], height=0.57)
    ax1.set_yticks(y, [t[0] for t in stages])
    ax1.set_xlim(0, len(seeds) + 3)
    ax1.set_xlabel("Instances preserving optimal ground states", labelpad=12)
    ax1.set_xticks([0, 5, 10, 15, len(seeds)])
    ax1.grid(axis="x")
    ax1.set_axisbelow(True)
    ax1.tick_params(axis="y", length=0, pad=10)
    for yy, val in zip(y, stage_counts):
        ax1.text(
            val + 0.2,
            yy,
            f"{val}/{len(seeds)}",
            fontsize=17,
            color=FG,
            va="center",
            weight="bold",
        )
    ax2 = fig.add_axes([0.68, 0.26, 0.26, 0.50])
    vals = [
        median(groups[True]) if groups[True] else 0,
        median(groups[False]) if groups[False] else 0,
    ]
    ax2.bar([0, 1], vals, width=0.58, color=[QUANTUM, FAIL])
    ax2.set_xticks(
        [0, 1],
        [
            f"Preserved\n(n={len(groups[True])})",
            f"Not preserved\n(n={len(groups[False])})",
        ],
    )
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("Median optimal-shot probability (%)", labelpad=12)
    ax2.grid(axis="y")
    ax2.set_axisbelow(True)
    for x, value in enumerate(vals):
        ax2.text(
            x,
            value + 3,
            display_pct(value),
            ha="center",
            color=FG,
            fontsize=22,
            weight="bold",
        )
    save(fig, output, dpi)


def embedding_vs_success(seeds, q, output, dpi):
    fig = canvas(
        "Interaction error is not the whole story",
        "Embedding error versus observed clustering-optimal probability, one dot per instance",
        "Colors indicate final-Hamiltonian ground-state preservation. Examples: seed 9 and seed 12 break a simple success rule.",
        dpi,
    )
    ax = fig.add_axes([0.11, 0.20, 0.82, 0.59])
    xvalues = [pct(row["relative_interaction_error"]) for row in q]
    yvalues = [pct(row["reference_probability"]) for row in q]
    for seed, x, y, row in zip(seeds, xvalues, yvalues, q):
        if x is None or y is None:
            continue
        preserved = row["embedded_all_ground_states_original_optimal"] == "True"
        ax.scatter(
            x,
            y,
            color=QUANTUM if preserved else FAIL,
            s=155,
            edgecolors=BG,
            linewidths=1.0,
            zorder=4,
        )
        if seed in (9, 12):
            ax.annotate(
                str(seed),
                (x, y),
                xytext=(9, 5),
                textcoords="offset points",
                color=FG,
                fontsize=16,
                weight="bold",
            )
    ax.set_xlim(
        max(0, min(v for v in xvalues if v is not None) - 3),
        max(v for v in xvalues if v is not None) + 4,
    )
    ax.set_ylim(-5, 105)
    ax.set_xlabel("Relative interaction embedding error (%)", labelpad=14)
    ax.set_ylabel("Clustering-reference probability (%)", labelpad=14)
    ax.grid()
    ax.set_axisbelow(True)
    fig.legend(
        handles=[
            Patch(facecolor=QUANTUM, label="All embedded ground states optimal"),
            Patch(facecolor=FAIL, label="Not all embedded ground states optimal"),
        ],
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.94, 0.84),
        ncol=2,
        labelcolor=FG,
        fontsize=12,
    )
    save(fig, output, dpi)


def routing_gap(seeds, q, rnd, output, dpi):
    counts = {int(float(r["shots"])) for r in q if numeric(r.get("shots")) is not None}
    label = (
        f"Best-of-{next(iter(counts)):,} shots"
        if len(counts) == 1
        else "Best measured shots"
    )
    random_gaps = [numeric(r.get("best_cvrp_gap_pct")) for r in rnd]
    footnote = "Only feasible shots contribute to gaps."
    if all(gap is not None and abs(gap) < 1e-9 for gap in random_gaps):
        footnote += (
            " Uniform random attains a 0% best CVRP gap for every included seed."
        )
    fig = canvas(
        f"{label}: routed CVRP gap",
        "The clustering objective and the final routing objective need not agree",
        footnote,
        dpi,
    )
    ax = fig.add_axes([0.09, 0.22, 0.85, 0.56])
    gaps = [numeric(row["best_cvrp_gap_pct"]) for row in q]
    colors = [FAIL if val is not None and val > 1e-9 else QUANTUM for val in gaps]
    known_seeds = [seed for seed, gap in zip(seeds, gaps) if gap is not None]
    known_gaps = [gap for gap in gaps if gap is not None]
    known_colors = [color for color, gap in zip(colors, gaps) if gap is not None]
    ax.bar(known_seeds, known_gaps, color=known_colors, width=0.72)
    for seed, gap in zip(seeds, gaps):
        if gap is None:
            ax.text(seed, 2.5, "N/A", color=FAIL, ha="center", fontsize=11, rotation=90)
        if gap is not None and gap > 1e-9:
            ax.text(
                seed,
                gap + 1.5,
                display_pct(gap),
                ha="center",
                color=FG,
                fontsize=13,
                weight="bold",
            )
    success = sum(gap is not None and abs(gap) <= 1e-9 for gap in gaps)
    ax.text(
        0.02,
        0.91,
        f"Quantum CVRP optimum: {success}/{len(seeds)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=19,
        color=FG,
        bbox={"facecolor": BG, "edgecolor": GRID, "boxstyle": "round,pad=0.5"},
    )
    ax.set_xticks(seeds)
    ax.set_xlim(min(seeds) - 0.6, max(seeds) + 0.6)
    ax.set_ylim(0, max(10, max((v or 0) for v in gaps) + 12))
    ax.set_xlabel("Instance seed", labelpad=12)
    ax.set_ylabel("Best feasible CVRP gap (%)", labelpad=14)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    save(fig, output, dpi)


def gallery(output, files):
    images = [
        {"filename": f.name, "title": f.stem[3:].replace("_", " ").title()}
        for f in files
    ]
    slides_json = json.dumps(images)
    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DuQuantum Tier S · Live Demo</title>
<style>
html,body{margin:0;width:100%;height:100%;background:#101827;color:#f3f7fd;font-family:system-ui,sans-serif;overflow:hidden}
header{height:7vh;display:flex;align-items:center;justify-content:space-between;padding:0 3vw;box-sizing:border-box}
h1{font-size:clamp(15px,2vw,25px);margin:0}nav{display:flex;gap:12px;align-items:center}button{background:#21334a;color:white;border:1px solid #465d74;border-radius:8px;padding:8px 16px;cursor:pointer;font-size:20px}
main{height:89vh;display:flex;align-items:center;justify-content:center}img{display:block;max-width:100vw;max-height:89vh;object-fit:contain}
footer{height:4vh;text-align:center;color:#aab9cc;font-size:13px}
</style></head><body><header><h1>DuQuantum · Tier S benchmark</h1><nav><span id="counter"></span><button id="prev">&#8592;</button><button id="next">&#8594;</button></nav></header>
<main><img id="slide" alt="Benchmark chart"></main>
<footer>Use &larr; and &rarr; arrow keys to navigate. Charts are generated from results.csv.</footer>
<script>
const slides = __SLIDES__; let index=0;
function update(){document.querySelector('#slide').src=slides[index].filename;document.querySelector('#slide').alt=slides[index].title;document.querySelector('#counter').textContent=(index+1)+' / '+slides.length;}
function move(n){index=(index+n+slides.length)%slides.length;update();}
document.querySelector('#prev').onclick=()=>move(-1);document.querySelector('#next').onclick=()=>move(1);
document.addEventListener('keydown',e=>{if(e.key==='ArrowRight'||e.key===' '){e.preventDefault();move(1);}if(e.key==='ArrowLeft'){e.preventDefault();move(-1);}});update();
</script></body></html>"""
    (output / "index.html").write_text(
        page.replace("__SLIDES__", slides_json), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Tier S benchmark results.csv")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("demo_charts"),
        help="output directory",
    )
    parser.add_argument(
        "--method",
        default="emu-sv",
        help="quantum method name in CSV (default: emu-sv)",
    )
    parser.add_argument("--random-method", default="uniform_random")
    parser.add_argument("--dpi", type=int, default=144)
    args = parser.parse_args()
    if args.dpi < 50:
        parser.error("--dpi must be at least 50")
    seeds, q, rnd = load_data(args.csv, args.method, args.random_method)
    args.output.mkdir(parents=True, exist_ok=True)
    setup_plot()
    names = [
        "01_quantum_vs_random.png",
        "02_optimal_probability_by_seed.png",
        "03_ground_state_preservation.png",
        "04_embedding_error_vs_success.png",
        "05_cvrp_gap_by_seed.png",
    ]
    files = [args.output / name for name in names]
    print(f"Generating five demo plots for {len(seeds)} shared seeds:")
    comparison(seeds, q, rnd, files[0], args.dpi, args.method)
    seed_probabilities(seeds, q, rnd, files[1], args.dpi)
    ground_states(seeds, q, files[2], args.dpi)
    embedding_vs_success(seeds, q, files[3], args.dpi)
    routing_gap(seeds, q, rnd, files[4], args.dpi)
    gallery(args.output, files)
    print(f"Open {args.output / 'index.html'} in your browser (left/right arrow keys).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
