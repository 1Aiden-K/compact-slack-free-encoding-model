> **Branch `one-hot-model` (comparison experiment, not meant to be merged).**
> `src/onehot_tier_s.py` runs a one-hot version of the Tier S model: one atom per (customer, vehicle), so 8 atoms instead of 4.
> It uses the same objective, λ = 0.1, instances, embedding, pulse and emulator settings as the main model's baseline run, so only the encoding differs.
> On every valid assignment, its cost equals the main model's cost plus a constant.
> Choosing the one-vehicle penalty as `A_i = A + λ d_i²` gives every atom the same linear term, so a single global detuning is **exact**; the main model has to average it.
>
> Run from `src/`: `uv run python onehot_tier_s.py` (all 20 instances, about 3 minutes; add `--limit 1` for one instance). Results: `src/results/tier_s/onehot_baseline/results.csv`.
>
> **Trade-off (baseline pulse, 20 instances):** 8 atoms with exact detuning reaches 37.1% optimal shots on average, never below 10%. The main model reaches 64.0% with 4 atoms, but collapses on the instances where averaging moves its lowest-energy state. On those 6 instances, one-hot averages 39.4% against 14.7%. We kept the 4-atom model as the main submission; this branch documents the alternative.

# Capacitated Vehicle Routing Quantum Clustering Optimization Using Unbalanced Penalization
https://arxiv.org/abs/2211.13914

### Team: Psychics
#### Challenge: Pasqal Neutral Atom Optimization
*DuQuantum 2026*


## Requirements

- **Python:** 3.13 or later
- **Package manager:** [uv](https://docs.astral.sh/uv/)
- **IDE:** [Visual Studio Code](https://code.visualstudio.com/) (recommended)

## Dependencies

| Package | Version |
|---|---|
| [Pulser](https://pypi.org/project/pulser/) | ≥ 1.9.1 |
| [QoolQit](https://pypi.org/project/qoolqit/) | ≥ 1.4.0 |
| [QUBO Solver](https://pypi.org/project/qubo-solver/) | ≥ 0.9.1 |
| [Emu-SV](https://pypi.org/project/emu-sv/) | Latest compatible |
| [Pasqal Cloud](https://pypi.org/project/pasqal-cloud/) | ≥ 0.24.0 |
| [NumPy](https://pypi.org/project/numpy/) | ≥ 2.5.4 |
| [Matplotlib](https://pypi.org/project/matplotlib/) | ≥ 3.11.2 |
|[Dotenv](https://pypi.org/project/python-dotenv/) |  ≥ 0.9.9 |
|[QuTip](https://pypi.org/project/qutip/) |  ≥ 5.3.1 |

**Development dependency:** [Ruff](https://docs.astral.sh/ruff/) — Python linting and formatting.

Exact resolved versions are recorded in `uv.lock`.

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

### 2. Install Dependencies

Once the repository has been cloned, run:

```bash
uv sync --locked
```

This automatically installs the required Python version if necessary, creates a virtual environment (`.venv`), and installs the locked dependencies, including development dependencies.

### 3. Configure VS Code

**Install Recommended Extensions**

The repository includes `.vscode/extensions.json`, which recommends the following extensions:

| Extension | Purpose |
|---|---|
| [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) | Python execution, debugging, and environment integration |
| [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) | Linting, formatting, and import organization |
| [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) | Jupyter Notebook support |

VS Code should prompt you to install the recommended extensions when opening the repository. Alternatively, open the Extensions panel and select `Extensions: Show Recommended Extensions` from the Command Palette.

The Python extension also installs Pylance and Python Debugger by default.

**Select Python Interpreter**

1. Open the Command Palette:
   - **macOS:** `Cmd+Shift+P`
   - **Windows:** `Ctrl+Shift+P`
2. Select `Python: Select Interpreter`.
3. Select the interpreter inside the project's `.venv` directory:
   - **macOS / Linux:** `.venv/bin/python`
   - **Windows:** `.venv\Scripts\python.exe`

**Editor Configuration**

The repository includes `.vscode/settings.json`, which configures VS Code to:

- Prefer the project's `.venv` interpreter.
- Use Ruff as the default Python formatter.
- Automatically format Python files on save.
- Organize imports and apply available Ruff fixes on explicit saves.
- Display linting diagnostics in the editor.

No additional editor configuration should be necessary.

### 4. Run Python

**VS Code (Recommended)**

Open a Python file and click `Run Python File` (the play button in the upper-right corner).

Ensure that the selected interpreter is the project's `.venv` environment.

**Terminal**

Alternatively, run Python scripts directly through uv:

```bash
uv run python <script.py>
```

## Managing Dependencies

**Add a dependency:**
```bash
uv add <package>
```

**Remove a dependency:**
```bash
uv remove <package>
```

**Add a development dependency:**
```bash
uv add --dev <package>
```

**Synchronize with the shared environment:**
```bash
git pull
uv sync --locked
```

When changing dependencies, commit both `pyproject.toml` and `uv.lock` to keep the team's environments consistent.

## Code Formatting and Linting

We use [Ruff](https://docs.astral.sh/ruff/) to maintain consistent Python code formatting and catch common errors.

Ruff is configured through `pyproject.toml`, ensuring that everyone uses the same linting and formatting rules.

**In VS Code:**

- Ruff highlights linting issues automatically.
- Saving a Python file automatically formats it.
- Explicit saves also organize imports and apply available automatic fixes.

**From the terminal:**

Check for linting issues:
```bash
uv run ruff check .
```

Automatically fix supported linting issues:
```bash
uv run ruff check --fix .
```

Format all Python files:
```bash
uv run ruff format .
```

Ruff is installed as a development dependency through `uv sync`, so no separate installation is necessary.