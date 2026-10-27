"""Summary of the 1-bit baseline emu-sv results. Generated with Grok, reviewed by the team."""

import csv
import statistics

opt_probs = []
feas_rates = []
n_true = 0

with open("src/results/tier_s/baseline/results.csv", newline="") as f:
    for row in csv.DictReader(f):
        if row["method"] != "emu-sv":
            continue
        opt_probs.append(float(row["optimal_probability"]))
        feas_rates.append(float(row["feasibility_rate"]))
        if row["embedded_all_ground_states_original_optimal"] == "True":
            n_true += 1

print("optimal_probability:")
print(f"  mean   {statistics.mean(opt_probs):.4f}")
print(f"  median {statistics.median(opt_probs):.4f}")
print(f"  min    {min(opt_probs):.4f}")
print(f"  max    {max(opt_probs):.4f}")
print("feasibility_rate:")
print(f"  mean   {statistics.mean(feas_rates):.4f}")
print(f"  median {statistics.median(feas_rates):.4f}")
print(f"  min    {min(feas_rates):.4f}")
print(f"  max    {max(feas_rates):.4f}")
print(f"embedded_all_ground_states_original_optimal == True: {n_true}")
