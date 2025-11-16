# visualize_extended.py

import matplotlib.pyplot as plt
import numpy as np


def plot_policy_results(results_dict):
    """
    Bar plot summary:
    results_dict = {
        "greedy": {0: count, 1: count, 2: count},
        "epsilon_greedy": {...},
        "thompson": {...}
    }
    """
    policies = list(results_dict.keys())
    markets = sorted(list(next(iter(results_dict.values())).keys()))

    data = np.array([
        [results_dict[p][m] for m in markets]
        for p in policies
    ])
    data = data / data.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(markets))
    width = 0.25

    for i, policy in enumerate(policies):
        ax.bar(x + (i - 1) * width, data[i], width, label=policy.replace("_", " ").title())

    ax.set_title("Final Choice Distribution (Percentage of Visits)")
    ax.set_xlabel("Market ID")
    ax.set_ylabel("% Visits")
    ax.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()



def plot_time_distribution(time_series_dict, T, markets):
    """
    time_series_dict = {
        "greedy": [market_id_t at step t],
        "epsilon_greedy": [...],
        "thompson": [...]
    }
    """
    markets = sorted(markets)
    n_markets = len(markets)

    fig, ax = plt.subplots(figsize=(12, 6))

    for m in markets:
        # For each policy, compute fraction choosing market m at each t
        # But here we combine policies separately → three line plots per policy
        pass

    # Instead, separate per-policy plots ↓
    fig, axs = plt.subplots(len(time_series_dict), 1, figsize=(10, 3 * len(time_series_dict)), sharex=True)

    for idx, (policy, series) in enumerate(time_series_dict.items()):
        counts = np.zeros((len(markets), T))

        # series[t] = market chosen at t
        for t in range(T):
            chosen = series[t]
            market_index = markets.index(chosen)
            counts[market_index, t] += 1

        # Convert to probabilities (0/1 here because one simulation)
        # If you run multiple simulations, average here.
        ax = axs[idx]
        for m_i, m in enumerate(markets):
            ax.plot(range(1, T + 1), counts[m_i], label=f"Market {m}")

        ax.set_title(f"Market Choice Over Time – {policy.replace('_',' ').title()}")
        ax.set_ylabel("Chosen (0/1)")
        ax.legend(loc="upper right")

    axs[-1].set_xlabel("Time step")
    plt.tight_layout()
    plt.show()



def plot_market_choice_probabilities(time_series_dict, T, markets, n_simulations):
    """
    If multiple simulations → convert to probabilities.
    time_series_dict[policy] is a list of lists:
        time_series_dict[policy][sim][t] = chosen_market
    """
    markets = sorted(markets)
    fig, axs = plt.subplots(len(time_series_dict), 1, figsize=(12, 3 * len(time_series_dict)), sharex=True)

    for idx, (policy, all_runs) in enumerate(time_series_dict.items()):
        ax = axs[idx]

        counts = np.zeros((len(markets), T))

        for run in all_runs:
            for t in range(T):
                chosen = run[t]
                m_idx = markets.index(chosen)
                counts[m_idx, t] += 1

        probs = counts / n_simulations

        for m_i, m in enumerate(markets):
            ax.plot(range(1, T + 1), probs[m_i], label=f"Market {m}")

        ax.set_ylabel("Probability")
        ax.set_title(f"Choice Probability Over Time – {policy.replace('_',' ').title()}")
        ax.legend(loc="upper right")

    axs[-1].set_xlabel("Time step")
    plt.tight_layout()
    plt.show()
