# plotting.py

import matplotlib.pyplot as plt

plt.rcParams.update({'figure.max_open_warning': 0})


def plot_results(results):
    months = results["months"]
    avg_income = results["avg_income"]
    avg_maintenance_paid = results["avg_maintenance_paid"]
    required_maintenance = results["required_maintenance"]
    avg_savings = results["avg_savings"]
    leader_savings = results["leader_savings"]
    leader_changes = results["leader_changes"]
    leader_history = results["leader_history"]

    leader_change_months = [m for m, _ in leader_changes]

    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    axes[0].plot(months, avg_income, label="Average Income")
    for i, m in enumerate(leader_change_months):
        axes[0].axvline(m, linestyle=":", alpha=0.7,
                        label="Leader Change" if i == 0 else None)
    axes[0].set_title("Average Monthly Income Over Time")
    axes[0].set_ylabel("Income")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(months, avg_maintenance_paid, label="Maintenance Paid")
    axes[1].plot(months, required_maintenance, linestyle="--", label="Maintenance Required")
    for i, m in enumerate(leader_change_months):
        axes[1].axvline(m, linestyle=":", alpha=0.7,
                        label="Leader Change" if i == 0 else None)
    axes[1].set_ylabel("Per-Agent Maintenance")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(months, avg_savings, label="Average Savings")
    axes[2].plot(months, leader_savings, linestyle="--", label="Leader Savings")
    for i, m in enumerate(leader_change_months):
        axes[2].axvline(m, linestyle=":", alpha=0.7,
                        label="Leader Change" if i == 0 else None)
    axes[2].set_xlabel("Month")
    axes[2].set_ylabel("Savings")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

    # -----------------------------
    # LEADER HISTORY PRINT (RESTORED)
    # -----------------------------
    # print("\nLeader history and attributes:")
    # for i, leader_data in enumerate(leader_history):
    #     print(f"\nLeader {i + 1}")
    #     print(f"  Start month: {leader_data['start_month']}")
    #     print(f"  Agent index: {leader_data['index']}")
    #     print(f"  Greed: {leader_data['greed']:.3f}")
    #     print(f"  Competence: {leader_data['competence']:.3f}")
    #     print(f"  Savings at start: {leader_data['starting_savings']:.2f}")
    #     print(f"  Avg income at start: {leader_data['starting_avg_income']:.2f}")
