# Track A: one-command evaluation CSV

Run from the **GitHub repository root** after `uv sync --locked`:

```bash
uv run python src/main.py --tier s benchmark \
  --instances-dir src/instances/tier_s \
  --output-dir src/results/tier_s/submission_01 \
  --no-optimize-pulses --backend emu-sv --shots 1000 --dt 10 --epsilon 0
```

The command evaluates every certified Tier S instance in the specified directory. Output includes:

- `results.csv`: one row for each `(instance, method, seed)`; methods `exact_clustering`, `exact_qubo`, `uniform_random`, and `emu-sv`.
- `summary.json`: aggregate metrics and undefined-gap counts.
- `manifest.json`: settings, instance hashes and source hashes.
- `tier_s_seed_XXXX.json`: sample counts, detailed results and per-instance diagnostic results.
- `tier_s_seed_XXXX.log`: worker diagnostics.

The output folder must be absent or empty. Change `submission_01` for another run; existing results are never overwritten.

## Challenge metrics

| Challenge metric | CSV field(s) | Definition |
| --- | --- | --- |
| Best model gap | `best_model_gap_pct` | Best *feasible* observed clustering cost vs exact constrained clustering optimum. |
| Best CVRP gap | `best_cvrp_gap_pct` | Minimum exact depot-to-depot routing cost among *feasible measured clusterings* vs exact CVRP optimum. |
| Feasibility rate | `feasibility_rate` | Feasible shots / all measured shots. |
| Probability reaching reference | `reference_probability`, `cvrp_reference_probability` | Fraction of all shots with model/CVRP gap no more than `epsilon_pct`; `--epsilon` is in **percentage points**. |
| Atom efficiency | `atoms` | Physical atoms in the register; 0 means a classical baseline, not a 0-atom implementation. |
| Embedding fidelity | `relative_interaction_error` | The dimensionless error `sqrt(sum_{i<j}(V_ij^real-V_ij^target)^2 / sum_{i<j}(V_ij^target)^2)`; lower is better. Multiply by 100 for percent. |
| Reference status | `reference_status`, `cvrp_reference_status` | Both are `proven_optimum` because all feasible assignments are enumerated, with all route orders exhaustively checked for CVRP. |

`clustering_reference_cost` and `cvrp_reference_cost` give the two reference values. Both gap fields are empty if no feasible shot was sampled, rather than being incorrectly recorded as zero. Reference-reaching probability is then zero if a reference exists.

**The route-second calculation is classical evaluation, not quantum TSP.** It is permitted for Track A CVRP gap evaluation; it does not count as a Track B quantum routing result. All routes start/end at the instance's fixed depot, and empty vehicles cost zero.

If you need an optimized-pulse run instead, remove `--no-optimize-pulses` and supply an explicit optimizer budget (`--evaluations`, see `--help`); expect substantially greater runtime.