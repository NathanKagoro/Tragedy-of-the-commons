def apply_monthly_income_and_tax(agents, leader_index, tax_rate):
    maintenance_collected = 0.0

    for agent in agents:
        agent.savings += agent.monthly_income
        tax = agent.monthly_income * tax_rate
        stolen = tax * agents[leader_index].greedy
        maintenance = tax - stolen

        agent.savings -= tax
        agents[leader_index].savings += stolen
        maintenance_collected += maintenance

    return maintenance_collected

# economy.py

def total_monthly_taxable_income(agent_list):
    """Return the sum of all agents' monthly incomes."""
    return sum(agent.monthly_income for agent in agent_list)
