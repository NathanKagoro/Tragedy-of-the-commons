import numpy as np
from parameters import *
import random

class Agent:
    def __init__(self, index):
        self.index = index
        self.greedy = np.random.uniform(greed_low, greed_high)
        self.competence = np.random.uniform(competence_low, competence_high)
        self.monthly_income = base_monthly_income * (1.0 + (self.competence - 0.5) * 0.1)
        self.savings = self.monthly_income * 1.5
        self.history_income = []
        self.history_savings = []

        # Voting weights (private, learned)
        self.vote_weights = {
            "leader_greed": random.uniform(-0.1, 0.1),
            "leader_competence": random.uniform(0.1, 0.3),
            "recent_income_growth": random.uniform(0.1, 0.3),
        }

        # Expected savings baseline for learning
        self.expected_term_savings = self.savings

        # Greed adaptability
        self.greed_learning_rate = random.uniform(0.01, 0.05)

