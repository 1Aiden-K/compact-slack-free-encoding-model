# Tier S statistical analysis

20 instances (seeds 0–19), 1000 shots each. Intervals are 95% percentile-bootstrap intervals over instances (10,000 resamples, seed 0). Paired tests are two-sided Wilcoxon signed-rank tests on the same instances.

## Summary by method

| Method | Atoms | Optimal shots, mean [95% CI] | Median | Min | Feasible shots, mean [95% CI] |
|---|---|---|---|---|---|
| Random guessing | – | 13.1% [12.0%, 14.2%] | 13.8% | 9.8% | 58.7% [48.3%, 69.0%] |
| 1-bit, baseline pulse | 4 | 64.0% [46.1%, 81.5%] | 89.8% | 1.2% | 96.5% [94.5%, 98.2%] |
| 1-bit, optimized pulse | 4 | 69.0% [52.0%, 85.1%] | 93.5% | 5.7% | 93.4% [86.5%, 97.7%] |
| One-hot, baseline pulse | 8 | 37.1% [31.5%, 42.6%] | 38.0% | 10.1% | 74.6% [64.4%, 83.9%] |

## Paired comparisons (optimal-shot rate)

| Comparison | Mean difference | Better on | Wilcoxon p |
|---|---|---|---|
| 1-bit, baseline pulse vs Random guessing | +50.9 pts | 14 of 20 | 0.000851 |
| 1-bit, optimized pulse vs 1-bit, baseline pulse | +5.0 pts | 20 of 20 | 8.83e-05 |
| One-hot, baseline pulse vs 1-bit, baseline pulse | -26.8 pts | 6 of 20 | 0.0192 |

## Why the 1-bit model is bimodal

Averaged global detuning keeps the true optimum as the lowest-energy state on 12 of 20 instances. After embedding, it is kept on 14.

| 1-bit baseline instances | Count | Mean optimal shots | Median |
|---|---|---|---|
| Lowest-energy state is the optimum | 14 | 85.1% | 90.9% |
| Lowest-energy state is not the optimum | 6 | 14.7% | 2.5% |

Mann–Whitney U, kept vs broken: p = 0.000103. Broken instances (seeds): [5, 6, 9, 15, 16, 17].

## One-hot on the instances where the 1-bit model broke

On the 6 broken 1-bit instances, one-hot averages 39.4% optimal shots versus 14.7% for 1-bit. On the other 14, one-hot averages 36.1% versus 85.1%. One-hot keeps the correct lowest-energy state on 15 of 20.

## What predicts success? (Spearman rank correlation with optimal shots)

| Predictor | 1-bit baseline ρ (p) | One-hot ρ (p) |
|---|---|---|
| Embedding error | +0.06 (0.81) | -0.22 (0.34) |
| Load ratio ρ | +0.06 (0.8) | -0.06 (0.81) |
| Largest-demand ratio | -0.21 (0.36) | -0.23 (0.34) |
