import random
from parameters import *
import numpy as np


def check_immediate_removal(avg_income_now, avg_income_at_start):
    if avg_income_now < income_drop_threshold * avg_income_at_start:
        return True
    return False

def score_candidate(voter, candidate, recent_income_growth):
    return (
        voter.vote_weights["leader_greed"] * candidate.greedy +
        voter.vote_weights["leader_competence"] * candidate.competence +
        voter.vote_weights["recent_income_growth"] * recent_income_growth
    )

def elect_leader(agents, avg_income_record):
    if len(avg_income_record) >= 4:
        recent_income_growth = (
            avg_income_record[-1] - avg_income_record[-4]
        ) / max(avg_income_record[-4], 1.0)
    else:
        recent_income_growth = 0.0

    scores = []
    for candidate in agents:
        total_score = 0.0
        for voter in agents:
            total_score += score_candidate(voter, candidate, recent_income_growth)
        scores.append(total_score)

    return int(np.argmax(scores)), recent_income_growth

def end_of_term_update(agents, leader_index, leader_greed_frozen):
    """
    Handles:
    - voter learning
    - leader greed adaptation
    - greed freezing for next term
    """

    leader = agents[leader_index]

    # --- VOTER LEARNING ---
    for agent in agents:
        realised_gain = agent.savings - agent.expected_term_savings

        agent.vote_weights["leader_greed"] += (
            0.01 * realised_gain * leader_greed_frozen
        )
        agent.vote_weights["leader_competence"] += (
            0.01 * realised_gain * leader.competence
        )

        agent.expected_term_savings = agent.savings

    # --- LEADER GREED ADAPTATION ---
    greed_adjustment = (
        leader.greed_learning_rate * (leader.savings - leader.expected_term_savings)
    )

    leader.greedy = min(max(leader.greedy + greed_adjustment, 0.0), 1.0)

    # --- FREEZE GREED FOR NEXT TERM ---
    return leader.greedy
