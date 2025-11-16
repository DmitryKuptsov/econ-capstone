# visualize.py
import matplotlib.pyplot as plt
import numpy as np


def plot_policy_results(results_dict):
    """
    results_dict = {
        "greedy": {0: count, 1: count, 2: count},
        "epsilon_greedy": {...},
        "thompson": {...}
    }
    """

    policies = list(results_dict.keys())
    markets = sorted(list(next(iter(results_dict.values())).keys()))

    # Prepare data matrix (#policies x #markets)
    data = np.array([
        [results_dict[policy][m] for m in markets]
        for policy in policies
    ])

    # Convert counts to percentages
    data = data / data.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(markets))
    width = 0.25  # bar width

    for i, policy in enumerate(policies):
        ax.bar(
            x + (i - 1) * width,
            data[i],
            width,
            label=policy.replace("_", " ").title(),
        )

    ax.set_xlabel("Market ID")
    ax.set_ylabel("% of Visits")
    ax.set_title("Scout Market-Visit Distribution by Policy")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Market {m}" for m in markets])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
