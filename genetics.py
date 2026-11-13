import random
import numpy as np
from simulation import run_simulation

def evolve_voting_weights(
    agents,
    elite_fraction=0.1,
    mutation_rate=0.2,
    mutation_scale=0.05,
    tournament_size=3
):
    """
    Genetic algorithm over voting weights.
    Agent 0 is excluded from evolution.
    """

    population_size = len(agents)

    # --- EXCLUDE AGENT 0 ---
    evolving_agents = agents[1:]

    # Fitness = final savings
    fitnesses = np.array([agent.savings for agent in evolving_agents])

    # --- ELITISM ---
    elite_count = max(1, int(elite_fraction * len(evolving_agents)))
    elite_indices = np.argsort(fitnesses)[-elite_count:]

    elites = [evolving_agents[i] for i in elite_indices]

    def tournament_select():
        contenders = random.sample(evolving_agents, tournament_size)
        return max(contenders, key=lambda a: a.savings)

    new_weights = []

    # --- PRESERVE ELITES ---
    for elite in elites:
        new_weights.append(elite.vote_weights.copy())

    # --- GENERATE OFFSPRING ---
    while len(new_weights) < len(evolving_agents):
        parent = tournament_select()

        child_weights = parent.vote_weights.copy()

        for key in child_weights:
            if random.random() < mutation_rate:
                child_weights[key] += random.gauss(0.0, mutation_scale)

                # Hard bounds for stability
                child_weights[key] = min(
                    max(child_weights[key], -1.0),
                    1.0
                )

        new_weights.append(child_weights)

    # --- ASSIGN BACK ---
    for agent, weights in zip(evolving_agents, new_weights):
        agent.vote_weights = weights


def evolve_population_batch(
    populations,
    ga_epochs=10
):
    """
    Evolves each population independently for several GA epochs.
    Returns performance metrics and final populations.
    """

    results = []

    for pop_index, agents in enumerate(populations):
        for epoch in range(ga_epochs):
            run_simulation(agents)
            evolve_voting_weights(agents)

            # reset transient state
            for agent in agents:
                agent.savings = agent.monthly_income * 1.5
                agent.history_income.clear()
                agent.history_savings.clear()

        final_avg_savings = np.mean([a.savings for a in agents])

        results.append({
            "population_index": pop_index,
            "agents": agents,
            "avg_savings": final_avg_savings
        })

    return results

