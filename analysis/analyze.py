"""Statistics and figures for the Tier S results.

Inputs (copied into analysis/data so this branch is self-contained):
    onebit_baseline.csv   1-bit model, baseline pulse (also holds uniform_random)
    onebit_optimized.csv  1-bit model, optimized pulse (10 evaluations)
    onehot_baseline.csv   one-hot model, baseline pulse (from the one-hot-model branch)
Pulse-optimizer histories are read from src/results/tier_s/optimized_10/*.json.

Run from the repository root:
    uv run python analysis/analyze.py
Writes analysis/stats.md and analysis/figures/*.png.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA, FIGS = ROOT / "data", ROOT / "figures"
HISTORY_DIR = ROOT.parent / "src/results/tier_s/optimized_10"
RNG = np.random.default_rng(0)
BOOTSTRAP = 10_000

# Reference categorical palette, fixed order; random guessing is neutral gray.
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = {
    "Random guessing": "#8a8984",
    "1-bit, baseline pulse": "#2a78d6",
    "1-bit, optimized pulse": "#1baf7a",
    "One-hot, baseline pulse": "#eb6834",
}
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 10,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    }
)


def load(name, method):
    rows = [r for r in csv.DictReader(open(DATA / name)) if r["method"] == method]
    return {int(r["instance_seed"]): r for r in rows}


def col(rows, key):
    return np.array([float(rows[s][key]) for s in sorted(rows)])


def flag(rows, key):
    return np.array([rows[s][key].lower() == "true" for s in sorted(rows)])


def ci(values):
    """Mean with a 95% percentile-bootstrap interval over instances."""
    means = RNG.choice(values, (BOOTSTRAP, len(values))).mean(axis=1)
    return values.mean(), *np.percentile(means, [2.5, 97.5])


def pct(x):
    return f"{100 * x:.1f}%"


METHODS = {
    "Random guessing": load("onebit_baseline.csv", "uniform_random"),
    "1-bit, baseline pulse": load("onebit_baseline.csv", "emu-sv"),
    "1-bit, optimized pulse": load("onebit_optimized.csv", "emu-sv"),
    "One-hot, baseline pulse": load("onehot_baseline.csv", "emu-sv-onehot"),
}
SEEDS = sorted(METHODS["1-bit, baseline pulse"])
for rows in METHODS.values():
    assert sorted(rows) == SEEDS, "every method must cover the same instances"

lines = ["# Tier S statistical analysis", ""]
lines += [
    f"{len(SEEDS)} instances (seeds {SEEDS[0]}–{SEEDS[-1]}), 1000 shots each. "
    f"Intervals are 95% percentile-bootstrap intervals over instances "
    f"({BOOTSTRAP:,} resamples, seed 0). Paired tests are two-sided Wilcoxon "
    "signed-rank tests on the same instances.",
    "",
    "## Summary by method",
    "",
    "| Method | Atoms | Optimal shots, mean [95% CI] | Median | Min | "
    "Feasible shots, mean [95% CI] |",
    "|---|---|---|---|---|---|",
]
summary = {}
for name, rows in METHODS.items():
    opt, feas = col(rows, "optimal_probability"), col(rows, "feasibility_rate")
    summary[name] = (ci(opt), ci(feas))
    atoms = {"Random guessing": "–"}.get(name, rows[SEEDS[0]]["atoms"])
    (m, lo, hi), (fm, flo, fhi) = summary[name]
    lines.append(
        f"| {name} | {atoms} | {pct(m)} [{pct(lo)}, {pct(hi)}] | "
        f"{pct(np.median(opt))} | {pct(opt.min())} | "
        f"{pct(fm)} [{pct(flo)}, {pct(fhi)}] |"
    )

base = METHODS["1-bit, baseline pulse"]
opt_b = col(base, "optimal_probability")
lines += ["", "## Paired comparisons (optimal-shot rate)", ""]
lines += ["| Comparison | Mean difference | Better on | Wilcoxon p |", "|---|---|---|---|"]
for a, b in [
    ("1-bit, baseline pulse", "Random guessing"),
    ("1-bit, optimized pulse", "1-bit, baseline pulse"),
    ("One-hot, baseline pulse", "1-bit, baseline pulse"),
]:
    x = col(METHODS[a], "optimal_probability")
    y = col(METHODS[b], "optimal_probability")
    p = stats.wilcoxon(x, y).pvalue
    lines.append(
        f"| {a} vs {b} | {100 * (x - y).mean():+.1f} pts | "
        f"{(x > y).sum()} of {len(x)} | {p:.3g} |"
    )

kept = flag(base, "embedded_all_ground_states_original_optimal")
global_ok = flag(base, "global_all_ground_states_original_optimal")
lines += [
    "",
    "## Why the 1-bit model is bimodal",
    "",
    f"Averaged global detuning keeps the true optimum as the lowest-energy state on "
    f"{global_ok.sum()} of {len(SEEDS)} instances. After embedding, it is kept on "
    f"{kept.sum()}.",
    "",
    "| 1-bit baseline instances | Count | Mean optimal shots | Median |",
    "|---|---|---|---|",
    f"| Lowest-energy state is the optimum | {kept.sum()} | "
    f"{pct(opt_b[kept].mean())} | {pct(np.median(opt_b[kept]))} |",
    f"| Lowest-energy state is not the optimum | {(~kept).sum()} | "
    f"{pct(opt_b[~kept].mean())} | {pct(np.median(opt_b[~kept]))} |",
    "",
    f"Mann–Whitney U, kept vs broken: p = "
    f"{stats.mannwhitneyu(opt_b[kept], opt_b[~kept]).pvalue:.3g}. "
    f"Broken instances (seeds): {[s for s, k in zip(SEEDS, kept) if not k]}.",
]

onehot = METHODS["One-hot, baseline pulse"]
opt_h = col(onehot, "optimal_probability")
kept_h = flag(onehot, "embedded_all_ground_states_original_optimal")
lines += [
    "",
    "## One-hot on the instances where the 1-bit model broke",
    "",
    f"On the {(~kept).sum()} broken 1-bit instances, one-hot averages "
    f"{pct(opt_h[~kept].mean())} optimal shots versus {pct(opt_b[~kept].mean())} "
    f"for 1-bit. On the other {kept.sum()}, one-hot averages "
    f"{pct(opt_h[kept].mean())} versus {pct(opt_b[kept].mean())}. One-hot keeps "
    f"the correct lowest-energy state on {kept_h.sum()} of {len(SEEDS)}.",
]

lines += ["", "## What predicts success? (Spearman rank correlation with optimal shots)", ""]
lines += ["| Predictor | 1-bit baseline ρ (p) | One-hot ρ (p) |", "|---|---|---|"]
for label, key in [
    ("Embedding error", "relative_interaction_error"),
    ("Load ratio ρ", "load_ratio"),
    ("Largest-demand ratio", "largest_demand_ratio"),
]:
    cells = []
    for rows, opt in ((base, opt_b), (onehot, opt_h)):
        values = col(rows if key in rows[SEEDS[0]] else base, key)
        r, p = stats.spearmanr(values, opt)
        cells.append(f"{r:+.2f} ({p:.2g})")
    lines.append(f"| {label} | {cells[0]} | {cells[1]} |")
(ROOT / "stats.md").write_text("\n".join(lines) + "\n")


def bar_labels(ax, bars, values):
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v + 0.03, pct(v),
            ha="center", va="bottom", color=INK, fontsize=9,
        )


# Figure 1: every method, two small multiples (one measure each, one axis each).
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
names = list(METHODS)
for ax, idx, title in zip(axes, (0, 1), ("Optimal shots", "Feasible shots")):
    means = [summary[n][idx][0] for n in names]
    err = np.array([[m - summary[n][idx][1], summary[n][idx][2] - m]
                    for n, m in zip(names, means)]).T
    bars = ax.bar(range(len(names)), means, width=0.6,
                  color=[COLORS[n] for n in names], edgecolor=SURFACE, linewidth=2)
    ax.errorbar(range(len(names)), means, yerr=err, fmt="none",
                ecolor=MUTED, elinewidth=1.2, capsize=3)
    bar_labels(ax, bars, means)
    ax.set_xticks(range(len(names)), [n.replace(", ", "\n") for n in names], fontsize=8)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="x", visible=False)
fig.suptitle("Tier S: mean over 20 instances, 95% bootstrap intervals",
             x=0.01, ha="left", color=MUTED, fontsize=9)
fig.savefig(FIGS / "fig1_methods.png")
plt.close(fig)

# Figure 2: 1-bit per instance, split by whether the lowest-energy state survived.
order = np.argsort(opt_b)
fig, ax = plt.subplots(figsize=(10, 3.8))
kept_color, broken_color = COLORS["1-bit, baseline pulse"], COLORS["One-hot, baseline pulse"]
ax.bar(range(len(SEEDS)), opt_b[order], width=0.7, edgecolor=SURFACE, linewidth=2,
       color=[kept_color if kept[i] else broken_color for i in order])
ax.set_xticks(range(len(SEEDS)), [str(SEEDS[i]) for i in order])
ax.set_xlabel("Instance seed (sorted)")
ax.set_ylabel("Optimal shots")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.grid(axis="x", visible=False)
ax.legend(
    handles=[plt.Rectangle((0, 0), 1, 1, color=kept_color),
             plt.Rectangle((0, 0), 1, 1, color=broken_color)],
    labels=["Lowest-energy state is the optimum",
            "Averaging/embedding moved the lowest-energy state"],
    frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2,
)
ax.set_title("1-bit model, baseline pulse: success per instance",
             loc="left", fontweight="bold", pad=30)
fig.savefig(FIGS / "fig2_ground_state_split.png")
plt.close(fig)

# Figure 3: one-hot vs 1-bit, paired per instance.
fig, ax = plt.subplots(figsize=(10, 3.8))
x = np.arange(len(SEEDS))
ax.vlines(x, np.minimum(opt_b, opt_h), np.maximum(opt_b, opt_h), color=GRID, linewidth=2)
for values, name in ((opt_b, "1-bit, baseline pulse"), (opt_h, "One-hot, baseline pulse")):
    ax.scatter(x, values, s=42, color=COLORS[name], edgecolor=SURFACE,
               linewidth=1.5, zorder=3, label=f"{name.split(',')[0]} "
               f"({'4' if name.startswith('1') else '8'} atoms)")
for i in np.flatnonzero(~kept):  # shade instances where the 1-bit ground state moved
    ax.axvspan(x[i] - 0.45, x[i] + 0.45, color=GRID, alpha=0.6, zorder=0, linewidth=0)
ax.set_xticks(x, [str(s) for s in SEEDS])
ax.set_xlabel("Instance seed")
ax.set_ylabel("Optimal shots")
ax.set_ylim(-0.08, 1.05)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.grid(axis="x", visible=False)
handles, labels = ax.get_legend_handles_labels()
handles.append(plt.Rectangle((0, 0), 1, 1, color=GRID, alpha=0.6))
labels.append("1-bit lowest-energy state moved")
ax.legend(handles, labels, frameon=False, loc="lower left",
          bbox_to_anchor=(0, 1.02), ncol=3)
ax.set_title("One-hot vs 1-bit, same instances and pulse settings",
             loc="left", fontweight="bold", pad=30)
fig.savefig(FIGS / "fig3_onehot_vs_onebit.png")
plt.close(fig)

# Figure 4 (extra): pulse optimization, paired per instance.
opt_o = col(METHODS["1-bit, optimized pulse"], "optimal_probability")
fig, ax = plt.subplots(figsize=(5, 4))
for b, o in zip(opt_b, opt_o):
    ax.plot([0, 1], [b, o], color=GRID, linewidth=1.5, zorder=1)
ax.scatter(np.zeros(len(opt_b)), opt_b, s=36, color=COLORS["1-bit, baseline pulse"], zorder=2)
ax.scatter(np.ones(len(opt_o)), opt_o, s=36, color=COLORS["1-bit, optimized pulse"], zorder=2)
ax.set_xticks([0, 1], ["Baseline pulse", "Optimized pulse"])
ax.set_xlim(-0.4, 1.4)
ax.set_ylabel("Optimal shots")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.grid(axis="x", visible=False)
ax.set_title("1-bit: effect of pulse optimization", loc="left", fontweight="bold")
fig.savefig(FIGS / "fig4_pulse_optimization.png")
plt.close(fig)

# Figure 5 (extra): embedding error vs success, both models.
fig, ax = plt.subplots(figsize=(6, 4))
for rows, opt, name in ((base, opt_b, "1-bit, baseline pulse"),
                        (onehot, opt_h, "One-hot, baseline pulse")):
    ax.scatter(col(rows, "relative_interaction_error"), opt, s=40,
               color=COLORS[name], edgecolor=SURFACE, linewidth=1.5,
               label=name.split(",")[0])
ax.set_xlabel("Relative interaction (embedding) error")
ax.set_ylabel("Optimal shots")
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.legend(frameon=False)
ax.set_title("Embedding error vs success", loc="left", fontweight="bold")
fig.savefig(FIGS / "fig5_embedding_error.png")
plt.close(fig)

# Figure 6 (extra): optimizer best-so-far loss per instance.
fig, ax = plt.subplots(figsize=(6, 4))
for path in sorted(HISTORY_DIR.glob("tier_s_seed_*.json")):
    history = json.loads(path.read_text())["optimization"]["history"]
    loss = np.minimum.accumulate([h["loss"] for h in history])
    ax.plot(range(1, len(loss) + 1), loss, color=COLORS["1-bit, optimized pulse"],
            linewidth=1.2, alpha=0.6)
ax.set_xlabel("Optimizer evaluation (1 = baseline pulse)")
ax.set_ylabel("Best loss so far (1 − reference probability)")
ax.set_title("Pulse optimizer progress, 20 instances", loc="left", fontweight="bold")
fig.savefig(FIGS / "fig6_optimizer_history.png")
plt.close(fig)

print((ROOT / "stats.md").read_text())
print("Figures:", ", ".join(p.name for p in sorted(FIGS.glob("*.png"))))
