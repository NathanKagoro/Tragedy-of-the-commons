import numpy as np
import random

import agents
from parameters import *
from economy import total_monthly_taxable_income
from politics import check_immediate_removal, elect_leader, end_of_term_update, score_candidate

def run_simulation(agents):
    # -----------------------------
    # INITIAL STATE
    # -----------------------------
    leader_index = random.randrange(population)
    leader_greed_frozen = agents[leader_index].greedy

    leader_start_month = 0

    avg_income_initial = np.mean([agent.monthly_income for agent in agents])
    income_scale = avg_income_initial / base_monthly_income
    maintenance_multiplier = 1.0 + maintenance_scale_alpha * np.log(income_scale + 1.0)

    avg_income_at_leader_start = avg_income_initial
    maintenance_fraction = maintenance_target_fraction
    required_maintenance = (
        maintenance_fraction
        * maintenance_multiplier
        * total_monthly_taxable_income(agents)
        * np.random.uniform(0.9, 1.1)
    )

    monthly_cost_base_local = monthly_cost_base

    maintenance_collected_this_quarter = 0.0

    # -----------------------------
    # RECORDING ARRAYS
    # -----------------------------
    months_arr = []
    avg_income_record = []
    avg_maintenance_paid_arr = []
    required_maintenance_arr = []
    avg_savings_arr = []
    leader_savings_arr = []

    leader_indices_timeline = [(0, leader_index)]
    leader_history = []

    public_greed_levels = [agent.greedy for agent in agents]
    public_competence_levels = [agent.competence for agent in agents]


    # -----------------------------
    # MAIN LOOP
    # -----------------------------
    for month in range(months):
        leader = agents[leader_index]

        # Monthly salary and taxation
        for agent in agents:
            agent.savings += agent.monthly_income
            agent.history_income.append(agent.monthly_income)

            monthly_tax = agent.monthly_income * monthly_tax_fraction
            stolen_amount = monthly_tax * leader_greed_frozen
            maintenance_contribution = monthly_tax - stolen_amount

            agent.savings -= monthly_tax
            leader.savings += stolen_amount
            maintenance_collected_this_quarter += maintenance_contribution

            agent.history_savings.append(agent.savings)

        avg_income = np.mean([agent.monthly_income for agent in agents])

        # -----------------------------
        # QUARTERLY MAINTENANCE LOGIC
        # -----------------------------
        if (month + 1) % quarter_months == 0:
            portion_of_maintenance_paid = maintenance_collected_this_quarter / max(required_maintenance, 1.0)

            lookback = min(len(avg_income_record), 12)
            if lookback > 1:
                recent_avg = np.mean(avg_income_record[-lookback:])
                slope = (avg_income - recent_avg) / recent_avg
            else:
                slope = 0.0

            stochastic_factor = np.random.normal(0, 0.02) * (1.0 - leader.competence)
            diminishing_factor = 1.0 - np.exp(-income_scale)

            growth_base = max(0.0, portion_of_maintenance_paid - 1.0)
            decay_base = max(0.0, 1.0 - portion_of_maintenance_paid)

            growth_rate = min(
                quarter_cap,
                growth_base * (1.0 + leader.competence)
                * (1.0 + stochastic_factor)
                * diminishing_factor
                * (1.0 + slope)
            )

            decay_rate = min(
                quarter_cap,
                decay_base * (2.0 - leader.competence)
                * (1.0 + stochastic_factor)
                * diminishing_factor
                * (1.0 - slope)
            )

            if len(avg_income_record) >= lookback_months_for_decay:
                income_drop_ratio = avg_income / avg_income_record[-lookback_months_for_decay]
                if income_drop_ratio < (1.0 - sharp_decline_threshold):
                    damping = leader.competence * decay_damping_strength
                    decay_rate *= (1.0 - damping)

            critical_income_threshold = 0.5 * base_monthly_income
            if avg_income < critical_income_threshold:
                recovery_boost = (
                    (critical_income_threshold - avg_income)
                    / critical_income_threshold
                    * leader.competence
                    * 0.5
                    * np.random.uniform(0.8, 1.2)
                )
            else:
                recovery_boost = 0.0

            for agent in agents:
                if portion_of_maintenance_paid >= 1.0:
                    agent.monthly_income *= (1.0 + growth_rate)
                else:
                    agent.monthly_income *= max(0.0, 1.0 - decay_rate)
                    if recovery_boost > 0.0:
                        agent.monthly_income *= (1.0 + recovery_boost)

            maintenance_multiplier = (
                1.0
                + maintenance_scale_alpha
                * np.log(avg_income / base_monthly_income + 1.0)
                * leader.competence
            )

            if avg_income < avg_income_at_leader_start * 0.98:
                maintenance_fraction = max(
                    maintenance_min_fraction,
                    maintenance_fraction * (1.0 - maintenance_adjustment_rate)
                )
            elif avg_income > avg_income_at_leader_start * 1.02:
                maintenance_fraction = min(
                    maintenance_max_fraction,
                    maintenance_fraction * (1.0 + maintenance_adjustment_rate)
                )
            else:
                maintenance_fraction = min(
                    maintenance_max_fraction,
                    maintenance_fraction * (1.0 + maintenance_adjustment_rate * 0.1)
                )

            required_maintenance = (
                maintenance_fraction
                * maintenance_multiplier
                * total_monthly_taxable_income(agents)
            )

            maintenance_collected_this_quarter = 0.0
            #Annual inflation and cost adjustment
        if (month + 1) % 12 == 0 and len(avg_income_record) >= 12:
            avg_income_last_year = avg_income_record[-12]
            if avg_income > avg_income_last_year * 1.02:
                monthly_cost_base_local *= (1.0 + inflation_rate_annual)
            elif avg_income < avg_income_last_year * 0.98:
                monthly_cost_base_local *= (1.0 - inflation_rate_annual)

        # 
        # -----------------------------
        # LEADER REMOVAL
        # -----------------------------
        immediate_remove = check_immediate_removal(avg_income, avg_income_at_leader_start)

        if immediate_remove:
            end_of_term_update(agents, leader_index, leader_greed_frozen)
            leader_index, _ = elect_leader(agents, avg_income_record)
                        
            leader_start_month = month + 1
            avg_income_at_leader_start = avg_income
            leader_indices_timeline.append((month + 1, leader_index))

            leader_history.append({
                "start_month": leader_start_month,
                "index": leader_index,
                "greed": agents[leader_index].greedy,
                "competence": agents[leader_index].competence,
                "starting_savings": agents[leader_index].savings,
                "starting_avg_income": avg_income
            })

    #Yearly trend vote
        months_in_term = month + 1 - leader_start_month
        if months_in_term >= 12 and len(avg_income_record) >= 12:
            income_trend_slope = np.polyfit(range(12), avg_income_record[-12:], 1)[0]
            if income_trend_slope < 0:
                end_of_term_update(agents, leader_index, leader_greed_frozen)
                leader_index, _ = elect_leader(agents, avg_income_record)
                leader_start_month = month + 1
                avg_income_at_leader_start = avg_income
                leader_indices_timeline.append((month + 1, leader_index))

                leader_history.append({
                    "start_month": leader_start_month,
                    "index": leader_index,
                    "greed": agents[leader_index].greedy,
                    "competence": agents[leader_index].competence,
                    "starting_savings": agents[leader_index].savings,
                    "starting_avg_income": avg_income
                })

        # -----------------------------
        # RECORD METRICS
        # -----------------------------
        months_arr.append(month + 1)
        avg_income_record.append(avg_income)
        avg_maintenance_paid_arr.append(maintenance_collected_this_quarter / (population * quarter_months))
        required_maintenance_arr.append(required_maintenance / population)
        avg_savings_arr.append(np.mean([agent.savings for agent in agents]))
        leader_savings_arr.append(agents[leader_index].savings)

    return {
        "months": months_arr,
        "avg_income": avg_income_record,
        "avg_maintenance_paid": avg_maintenance_paid_arr,
        "required_maintenance": required_maintenance_arr,
        "avg_savings": avg_savings_arr,
        "leader_savings": leader_savings_arr,
        "leader_changes": leader_indices_timeline,
        "leader_history": leader_history
    }
