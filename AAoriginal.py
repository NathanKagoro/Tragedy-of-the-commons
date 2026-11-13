# abm_economy.py
# Agent-based economy simulation - run with Python 3.8+ (numpy, matplotlib required)

import numpy as np
import matplotlib.pyplot as plt
import random
import time

seed = int(time.time())
random.seed(seed)
np.random.seed(seed)


# -----------------------------
# PARAMETERS (tweakable)
# -----------------------------
N = 200
months = 1200

base_monthly_income = 3000.0
monthly_tax_fraction = 0.20

monthly_cost_base = 2000.0
inflation_rate_annual = 0.03

greed_low, greed_high = 0.0, 0.6
competence_low, competence_high = 0.2, 0.9

initial_leader_index = random.randrange(N)
quarter_months = 3
quarter_cap = 0.20

maintenance_target_fraction = 0.25
maintenance_adjustment_rate = 0.02
maintenance_min_fraction, maintenance_max_fraction = 0.05, 0.9
maintenance_scale_alpha = 0.4   # How volatile is maintenance increase at higher economic thresholds

income_drop_threshold = 0.25
trend_window_months = 12

lookback_months_for_decay = 12         # period to evaluate sharp downturns
sharp_decline_threshold = 0.5          # 50% drop triggers decay damping
decay_damping_strength = 0.5           # max fraction to reduce deterioration based on competence

# -----------------------------
# AGENT CLASS
# -----------------------------
class Agent:
    def __init__(self, index):
        self.index = index
        self.greedy = np.random.uniform(greed_low, greed_high)
        self.competence = np.random.uniform(competence_low, competence_high)
        self.monthly_income = base_monthly_income * (1.0 + (self.competence - 0.5) * 0.1)
        self.savings = self.monthly_income * 1.5
        self.history_income = []
        self.history_savings = []

agents = [Agent(index) for index in range(N)]

# -----------------------------
# INITIAL LEADER & MAINTENANCE
# -----------------------------
leader_index = initial_leader_index
leader_start_month = 0
avg_income_at_leader_start = np.mean([agent.monthly_income for agent in agents])

def total_monthly_taxable_income(agent_list):
    return sum(agent.monthly_income for agent in agent_list)

maintenance_fraction = maintenance_target_fraction
avg_income_initial = np.mean([agent.monthly_income for agent in agents])
income_scale = avg_income_initial / base_monthly_income
maintenance_multiplier = 1.0 + maintenance_scale_alpha * np.log(income_scale + 1.0)

required_maintenance = (
    maintenance_fraction
    * maintenance_multiplier
    * total_monthly_taxable_income(agents)
    * np.random.uniform(0.9, 1.1)   # ±10% random buffer
)


months_arr = []
avg_income_record = []
avg_maintenance_paid_arr = []
required_maintenance_arr = []
avg_savings_arr = []
leader_savings_arr = []

leader_indices_timeline = [(0, leader_index)]
leader_history = []

# -----------------------------
# IMMEDIATE REMOVAL CHECKS
# -----------------------------
def check_immediate_removals(avg_income_now, agents, avg_income_at_leader_start):
    if avg_income_now < income_drop_threshold * avg_income_at_leader_start:
        return True, "avg_income_below_25pct_of_start"

    avg_monthly_tax = np.mean([agent.monthly_income * monthly_tax_fraction for agent in agents])
    avg_monthly_cost = monthly_cost_base

    if avg_income_at_leader_start > (avg_monthly_tax + avg_monthly_cost) and avg_income_now <= (avg_monthly_tax + avg_monthly_cost):
        return True, "avg_income_dropped_below_tax_plus_cost"

    return False, None

# -----------------------------
# SIMULATION LOOP (MONTHLY)
# -----------------------------
maintenance_collected_this_quarter = 0.0

for month in range(months):
    leader = agents[leader_index]

    # Monthly salary and taxation
    for agent in agents:
        agent.savings += agent.monthly_income
        agent.history_income.append(agent.monthly_income)

        monthly_tax = agent.monthly_income * monthly_tax_fraction
        stolen_amount = monthly_tax * leader.greedy
        maintenance_contribution = monthly_tax - stolen_amount

        agent.savings -= monthly_tax
        agents[leader_index].savings += stolen_amount
        maintenance_collected_this_quarter += maintenance_contribution

        agent.history_savings.append(agent.savings)

    avg_income = np.mean([agent.monthly_income for agent in agents])

    # Quarterly maintenance and growth logic
    if (month + 1) % quarter_months == 0:
        portion_of_maintenance_paid = maintenance_collected_this_quarter / max(required_maintenance, 1.0)

        # Lookback trend factor
        lookback_months = min(len(avg_income_record), 12)  # 12 months max
        if lookback_months > 1:
            recent_avg = np.mean(avg_income_record[-lookback_months:])
            slope = (avg_income - recent_avg) / recent_avg
        else:
            slope = 0.0

        # Stochastic factor inversely proportional to competence
        stochastic_factor = np.random.normal(0, 0.02) * (1.0 - leader.competence)

        # Diminishing returns based on overall income scale
        diminishing_factor = 1.0 - np.exp(-income_scale)

        # Competence-weighted growth and decay base factors
        growth_base = max(0.0, portion_of_maintenance_paid - 1.0)
        decay_base = max(0.0, 1.0 - portion_of_maintenance_paid)

        # Smooth growth and decay rates
        growth_rate = min(
            quarter_cap,
            growth_base * (1.0 + leader.competence) * (1.0 + stochastic_factor) * diminishing_factor * (1 + slope)
        )
        decay_rate = min(
            quarter_cap,
            decay_base * (2.0 - leader.competence) * (1.0 + stochastic_factor) * diminishing_factor * (1 - slope)
        )

        # Decay damping if income has dropped sharply recently
        if len(avg_income_record) >= lookback_months_for_decay:
            income_drop_ratio = avg_income / avg_income_record[-lookback_months_for_decay]
            if income_drop_ratio < (1.0 - sharp_decline_threshold):
                damping = leader.competence * decay_damping_strength
                decay_rate *= (1.0 - damping)

        # Critical floor recovery to prevent permanent crash
        critical_income_threshold = 0.5 * base_monthly_income  # 50% of baseline
        if avg_income < critical_income_threshold:
            recovery_boost = (critical_income_threshold - avg_income) / critical_income_threshold
            recovery_boost *= leader.competence * 0.5  # scale max 50% of gap
            recovery_boost *= np.random.uniform(0.8, 1.2)    # optional stochastic tweak
        else:
            recovery_boost = 0.0

        # Apply growth, decay, and recovery to all agents
        for agent in agents:
            if portion_of_maintenance_paid >= 1.0:
                agent.monthly_income *= (1.0 + growth_rate)
            else:
                agent.monthly_income *= max(0.0, (1.0 - decay_rate))
                if recovery_boost > 0:
                    agent.monthly_income *= (1.0 + recovery_boost)

        # Maintenance target adaptation (competence-adjusted)
        maintenance_multiplier = 1.0 + maintenance_scale_alpha * np.log(avg_income / base_monthly_income + 1.0) * leader.competence

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

        required_maintenance = maintenance_fraction * maintenance_multiplier * total_monthly_taxable_income(agents)
        maintenance_collected_this_quarter = 0.0

        # Maintenance target adaptation (competence-adjusted)
        maintenance_multiplier = 1.0 + maintenance_scale_alpha * np.log(avg_income / base_monthly_income + 1.0) * leader.competence

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

        required_maintenance = maintenance_fraction * maintenance_multiplier * total_monthly_taxable_income(agents)
        maintenance_collected_this_quarter = 0.0

    # Annual inflation and cost adjustment
    if (month + 1) % 12 == 0 and len(avg_income_record) >= 12:
        avg_income_last_year = avg_income_record[-12]
        if avg_income > avg_income_last_year * 1.02:
            monthly_cost_base *= (1.0 + inflation_rate_annual)
        elif avg_income < avg_income_last_year * 0.98:
            monthly_cost_base *= (1.0 - inflation_rate_annual)

    # Voting and removal
    immediate_remove, _ = check_immediate_removals(avg_income, agents, avg_income_at_leader_start)
    if immediate_remove:
        candidates = [index for index in range(N) if index != leader_index]
        leader_index = random.choice(candidates)
        leader_start_month = month + 1
        avg_income_at_leader_start = avg_income
        leader_indices_timeline.append((month + 1, leader_index))

        leader_history.append({
            "start_month": month + 1,
            "index": leader_index,
            "greed": agents[leader_index].greedy,
            "competence": agents[leader_index].competence,
            "starting_savings": agents[leader_index].savings,
            "starting_avg_income": avg_income
        })

    # Yearly trend vote
    months_in_term = month + 1 - leader_start_month
    if months_in_term >= 12 and len(avg_income_record) >= 12:
        income_trend_slope = np.polyfit(range(12), avg_income_record[-12:], 1)[0]
        if income_trend_slope < 0:
            candidates = [index for index in range(N) if index != leader_index]
            leader_index = random.choice(candidates)
            leader_start_month = month + 1
            avg_income_at_leader_start = avg_income
            leader_indices_timeline.append((month + 1, leader_index))

            leader_history.append({
                "start_month": month + 1,
                "index": leader_index,
                "greed": agents[leader_index].greedy,
                "competence": agents[leader_index].competence,
                "starting_savings": agents[leader_index].savings,
                "starting_avg_income": avg_income
            })

    # Record metrics
    months_arr.append(month + 1)
    avg_income_record.append(avg_income)
    avg_maintenance_paid_arr.append(maintenance_collected_this_quarter / (N * quarter_months))
    required_maintenance_arr.append(required_maintenance / N)
    avg_savings_arr.append(np.mean([agent.savings for agent in agents]))
    leader_savings_arr.append(agents[leader_index].savings)

# -----------------------------
# PLOTTING
# -----------------------------
plt.rcParams.update({'figure.max_open_warning': 0})

fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

leader_change_months = [month for month, _ in leader_indices_timeline]

axes[0].plot(months_arr, avg_income_record, label="Average Income")
for index, month in enumerate(leader_change_months):
    axes[0].axvline(month, linestyle=":", alpha=0.7, label="Leader Change" if index == 0 else None)
axes[0].set_title("Average Monthly Income Over Time")
axes[0].set_ylabel("Income")
axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
axes[0].grid(True)

axes[1].plot(months_arr, avg_maintenance_paid_arr, label="Maintenance Paid")
axes[1].plot(months_arr, required_maintenance_arr, linestyle="--", label="Maintenance Required")
for index, month in enumerate(leader_change_months):
    axes[1].axvline(month, linestyle=":", alpha=0.7, label="Leader Change" if index == 0 else None)
axes[1].set_title("Maintenance Paid vs Required")
axes[1].set_ylabel("Per-Agent Maintenance")
axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
axes[1].grid(True)

axes[2].plot(months_arr, avg_savings_arr, label="Average Savings")
axes[2].plot(months_arr, leader_savings_arr, linestyle="--", label="Leader Savings")
for index, month in enumerate(leader_change_months):
    axes[2].axvline(month, linestyle=":", alpha=0.7, label="Leader Change" if index == 0 else None)
axes[2].set_title("Savings Over Time")
axes[2].set_xlabel("Month")
axes[2].set_ylabel("Savings")
axes[2].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
axes[2].grid(True)

plt.subplots_adjust(hspace=0.35)
plt.tight_layout()
plt.show()

# -----------------------------
# LEADER HISTORY
# -----------------------------
print("\nLeader history and attributes:")
for index, leader_data in enumerate(leader_history):
    print(f"\nLeader {index + 1}")
    print(f"  Start month: {leader_data['start_month']}")
    print(f"  Agent index: {leader_data['index']}")
    print(f"  Greed: {leader_data['greed']:.3f}")
    print(f"  Competence: {leader_data['competence']:.3f}")
    print(f"  Savings at start: {leader_data['starting_savings']:.2f}")
    print(f"  Avg income at start: {leader_data['starting_avg_income']:.2f}")
