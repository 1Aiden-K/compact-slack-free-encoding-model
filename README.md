# Capacitated Vehicle Routing Quantum Clustering Optimization Using Unbalanced Penalization

### Team: Psychics
#### Challenge: Pasqal Neutral Atom Optimization — Track A (capacity-feasible clustering)
*DuQuantum 2026*

We assign the customers of a Capacitated Vehicle Routing Problem (CVRP) to vehicles on an emulated neutral-atom register. Each Tier S customer is one atom: `z_i = 1` puts customer `i` in vehicle 1 and `z_i = 0` in vehicle 2. Capacity is handled without slack variables, using unbalanced penalization ([Montañez-Barrera et al., arXiv:2211.13914](https://arxiv.org/abs/2211.13914)). With two vehicles, that penalty reduces to a load-balancing term, so the full Tier S model needs only **4 atoms**. The slack-variable version needs 14.

## Quick start: reproduce the headline result

From the repository root:

```bash
uv sync --locked
uv run python src/main.py --tier s benchmark \
  --instances-dir src/instances/tier_s \
  --output-dir src/results/tier_s/reproduce \
  --no-optimize-pulses --backend emu-sv --shots 1000 --dt 10 --epsilon 0
```

This evaluates all 20 certified Tier S instances with four methods: exact clustering, exact QUBO, uniform random sampling and emu-sv emulation. It takes about 30 seconds on a laptop CPU. The output folder must be absent or empty; pick a new name for each run. Results go to `results.csv` (one row per instance and method), `summary.json`, `manifest.json` (settings and hashes), and one detail JSON per instance.

To include pulse optimization (10 search evaluations per instance, slower):

```bash
uv run python src/main.py --tier s benchmark \
  --instances-dir src/instances/tier_s \
  --output-dir src/results/tier_s/reproduce_optimized \
  --optimize-pulses --evaluations 10 --dt 1
```

## Headline results (Tier S, 20 instances, 1000 shots each)

| Method | Atoms | Feasible shots | Optimal shots | Best model gap |
|---|---|---|---|---|
| Uniform random bitstrings | – | 58.7% | 13.1% | 0% |
| emu-sv, baseline pulse | 4 | 96.5% | 64.0% | 0% on all 20 |
| emu-sv, optimized pulse | 4 | 93.4% | 69.0% | 0% on all 20 |

Values are means over instances. "Optimal shots" is the fraction of all shots that equal the exact constrained clustering optimum. The baseline row comes from the quick-start command (also committed in `src/results/tier_s/baseline/`). The optimized row is `src/results/tier_s/optimized_10/`.

**Known limitation.** The register uses one global detuning, but this QUBO's linear coefficients differ per customer. We set the detuning to their mean. On 8 of 20 instances, this averaging alone moves the register's lowest-energy state away from the true optimum. After embedding, 6 instances still have the wrong lowest-energy state, and on 5 of those the optimal-shot rate falls below 4%. This is recorded per instance in the `global_all_ground_states_original_optimal` and `embedded_all_ground_states_original_optimal` columns of `results.csv`.

**CVRP gap.** The quick-start run also routes each sampled feasible clustering exactly (classically) and compares it with the exact CVRP optimum (`best_cvrp_gap_pct`). Because cluster-first / route-second is a heuristic, the clustering optimum is not always the best clustering for routing. The mean best CVRP gap is 5.5% for emu-sv. Uniform random reaches 0% only because 1000 shots cover all 16 possible 4-bit assignments.

## Statistical analysis

Intervals are 95% percentile-bootstrap intervals over the 20 instances. Paired tests are two-sided Wilcoxon signed-rank tests on the same instances.

![Optimal and feasible shots by method](docs/figures/fig1_methods.png)

| Method | Atoms | Optimal shots, mean [95% CI] | Median | Feasible shots, mean [95% CI] |
|---|---|---|---|---|
| Random guessing | – | 13.1% [12.0, 14.2] | 13.8% | 58.7% [48.3, 69.0] |
| 1-bit, baseline pulse | 4 | 64.0% [46.1, 81.5] | 89.8% | 96.5% [94.5, 98.2] |
| 1-bit, optimized pulse | 4 | 69.0% [52.0, 85.1] | 93.5% | 93.4% [86.5, 97.7] |
| One-hot, baseline pulse | 8 | 37.1% [31.5, 42.6] | 38.0% | 74.6% [64.4, 83.9] |

- **The emulator does useful work:** the baseline pulse beats random guessing by 50.9 percentage points on average (p < 0.001).
- **Pulse optimization helps consistently:** it improves the optimal-shot rate on all 20 instances, by 5.0 points on average (p < 0.001).
- **The results are bimodal, and the averaged detuning explains why.** On the 14 instances where the embedded register's lowest-energy state is still the optimum, the mean is 85.1%. On the other 6 (seeds 5, 6, 9, 15, 16, 17) it is 14.7%, with a median of 2.5% (Mann–Whitney p < 0.001). Embedding error, load ratio and largest-demand ratio show no significant correlation with success (all Spearman |ρ| ≤ 0.23, p ≥ 0.3).

![1-bit success per instance, split by ground state](docs/figures/fig2_ground_state_split.png)

### Trade-off: one-hot comparison (branch `one-hot-model`)

We also built a one-hot version: one atom per (customer, vehicle), 8 atoms. It has the same objective, λ, instances and pulse settings, and its linear terms are uniform, so one global detuning reproduces it **exactly**. It is lower on average (37.1% vs 64.0%, p = 0.02) but never collapses (minimum 10.1%). On the 6 instances where the 1-bit model broke, one-hot averages 39.4% against 14.7%.

We submit the 4-atom model and accept its averaged detuning: it halves the atom count and wins on 14 of 20 instances. The one-hot branch documents the alternative and is not merged.

![One-hot vs 1-bit per instance](docs/figures/fig3_onehot_vs_onebit.png)

The full analysis (script, `stats.md`, three more figures) is on branch `graphs-and-statistical-analysis`. It is also not merged.

## Model

Full derivations, with equations:

- [docs/tier_s_model.md](docs/tier_s_model.md): Tier S model (N=4, K=2, Q=7), QUBO, Ising mapping, global detuning, pulse functions
- [docs/tier_s_metrics.md](docs/tier_s_metrics.md): every `results.csv` column and how it maps to the challenge metrics
- [docs/tier_m_model.md](docs/tier_m_model.md): Tier M model (N=10, K=3, 2 bits per customer, 20 atoms). This is a derivation only; it is not implemented in code yet.

In short, the cost is

```math
F(z)=\sum_{i<j}D_{ij}(1-z_i-z_j+2z_iz_j)+2\lambda\left(\sum_i d_i z_i-\frac{T}{2}\right)^2,\qquad \lambda=0.1
```

The first term is the total distance between customers who share a vehicle. The second favors balanced loads; the capacity `Q` cancels out of it. Every measured bitstring is therefore checked against the original capacity constraints, and only feasible shots count toward gaps.

| Encoding | Tier S atoms |
|---|---|
| One-hot assignment | 8 |
| One-hot assignment + binary slack | 14 |
| **This model** | **4** |

## Pipeline

```
certified instance → QUBO (model.py) → Ising + global detuning (ising.py)
  → atom positions (embedding.py) → pulse (pulses.py) → emu-sv (emulator.py)
  → feasibility, gaps, exact routing (evaluation.py, benchmark.py)
```

- **Embedding:** atom positions are fitted numerically (multi-start SLSQP) to the target couplings. The fit respects AnalogDevice's 5 µm minimum spacing and 38 µm radius. The error is reported as `relative_interaction_error`.
- **Pulse:** a sin² Rabi envelope with a linear detuning sweep over 4 µs, using one global channel and no DMM. The optional optimizer tunes amplitude, detuning endpoints, sweep shape and duration.
- **Emulation:** ideal, noiseless emu-sv. The `--dt` flag sets the integration step in ns.

## Commands

Run from the repository root as `uv run python src/main.py --tier s <command>`. Every command supports `--help`.

| Command | Purpose |
|---|---|
| `generate --seed 0 --count 20 --output-dir src/instances/tier_s` | Create reproducible, certified instances |
| `validate src/instances/tier_s/*.json` | Check each instance's feasibility certificate |
| `classical` | Exact enumeration of the QUBO and the constrained problem |
| `run --instance src/instances/tier_s/tier_s_seed_0000.json` | Embed, optionally optimize, and emulate one instance |
| `optimize --instance ...` | Compare the baseline and optimized pulses on one instance |
| `evaluate run.json` | Re-score saved counts or a saved run |
| `benchmark ...` | Every method on every instance → `results.csv` (see Quick start) |

## Instances

`src/instances/tier_s/` holds 20 instances, seeds 0–19 (Tier S: N=4, K=2, Q=7, demands in {1,…,4}). Each file records:

- the seed and generator settings
- coordinates in a 100×100 square
- the depot, fixed at (0, 0)
- a feasibility certificate (one valid assignment and its loads)
- the load ratio ρ and the largest-demand ratio d_max/Q

Feasibility is guaranteed by drawing demands uniformly and rejecting any draw without a valid assignment. Every assignment is enumerated, and one valid assignment is stored as the certificate. Reference values are proven optima by brute force, for both the clustering and the exactly routed CVRP.

## Repository layout

```
docs/                      model derivations and metric definitions
src/main.py                command-line entry point
src/duquantum/quantum/     Ising mapping, embedding, pulses, emulator, pulse optimization
src/duquantum/tier_s/      Tier S model, instances, classical references, evaluation, benchmark
src/instances/tier_s/      the 20 certified instances
src/results/tier_s/        committed benchmark outputs (baseline, optimized_10)
```

## Requirements

- **Python:** 3.13 or later (pinned in `.python-version`)
- **Package manager:** [uv](https://docs.astral.sh/uv/)
- **IDE:** [Visual Studio Code](https://code.visualstudio.com/) (optional)

Exact resolved versions of every package are pinned in `uv.lock`.

| Package | Version |
|---|---|
| [Pulser](https://pypi.org/project/pulser/) | ≥ 1.9.1 |
| [QoolQit](https://pypi.org/project/qoolqit/) | ≥ 1.4.0 |
| [QUBO Solver](https://pypi.org/project/qubo-solver/) | ≥ 0.9.1 |
| [Emu-SV](https://pypi.org/project/emu-sv/) | latest compatible |
| [Pasqal Cloud](https://pypi.org/project/pasqal-cloud/) | ≥ 0.24.0 |
| [NumPy](https://pypi.org/project/numpy/) | ≥ 2.5.4 |
| [SciPy](https://pypi.org/project/scipy/) | installed with Pulser; used by the embedding |
| [Matplotlib](https://pypi.org/project/matplotlib/) | ≥ 3.11.2 |
| [Dotenv](https://pypi.org/project/python-dotenv/) | ≥ 0.9.9 |
| [QuTiP](https://pypi.org/project/qutip/) | ≥ 5.3.1 |

**Development dependency:** [Ruff](https://docs.astral.sh/ruff/), for linting and formatting.

## Setup

### 1. Install uv

**macOS / Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

### 2. Install dependencies

```bash
uv sync --locked
```

This installs the required Python version if necessary, creates `.venv`, and installs the locked dependencies, including the development dependencies.

### 3. VS Code (optional)

Open the repository and accept the recommended extensions (Python, Ruff, Jupyter) from `.vscode/extensions.json`. Then run `Python: Select Interpreter` from the Command Palette and choose `.venv/bin/python` (macOS/Linux) or `.venv\Scripts\python.exe` (Windows). `.vscode/settings.json` already sets Ruff as the formatter and formats on save.

## Managing dependencies

```bash
uv add <package>          # add a dependency
uv remove <package>       # remove a dependency
uv add --dev <package>    # add a development dependency
git pull && uv sync --locked   # sync with the shared environment
```

When changing dependencies, commit both `pyproject.toml` and `uv.lock`.

## Formatting and linting

```bash
uv run ruff check .        # check
uv run ruff check --fix .  # fix what can be fixed automatically
uv run ruff format .       # format
```
