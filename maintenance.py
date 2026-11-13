import numpy as np
from parameters import *

def compute_maintenance_multiplier(average_income, leader_competence):
    income_scale = average_income / base_monthly_income
    return 1.0 + maintenance_scale_alpha * np.log(income_scale + 1.0) * leader_competence
