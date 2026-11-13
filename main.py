from agents import Agent
from parameters import population
from genetics import evolve_population_batch
from plotting import plot_results
from simulation import run_simulation

import copy
import numpy as np

POPULATION_BATCH_SIZE = 16
GA_EPOCHS = 20
BATCHES = 5

best_overall = None
best_score = -np.inf

for batch in range(BATCHES):
    print(f"\n=== Batch {batch + 1}/{BATCHES} ===")

    populations = []

    for _ in range(POPULATION_BATCH_SIZE):
        agents = [Agent(i) for i in range(population)]
        populations.append(agents)

    batch_results = evolve_population_batch(
        populations,
        ga_epochs=GA_EPOCHS
    )

    best_batch_result = max(batch_results, key=lambda r: r["avg_savings"])

    print(
        "Best batch avg savings:",
        best_batch_result["avg_savings"]
    )

    if best_batch_result["avg_savings"] > best_score:
        best_score = best_batch_result["avg_savings"]
        best_overall = copy.deepcopy(best_batch_result)

# -----------------------------
# FINAL RUN + PLOTTING
# -----------------------------
print("\n=== Running final best population ===")

final_results = run_simulation(best_overall["agents"])

plot_results(final_results)
