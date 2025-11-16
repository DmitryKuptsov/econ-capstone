"""Main script: runs all three policies, collects statistics, and visualizes results."""

from __future__ import annotations

from typing import Dict, List

from config import default_config
from bayes import NormalNormal
from market import Market
from scout import Scout
from simulation import Simulation
import policy

from visualize_extended import (
    plot_policy_results,
    plot_time_distribution,
    plot_market_choice_probabilities,
)


# --------------------------------------------------------------
#   BUILDERS
# --------------------------------------------------------------

def build_markets_from_config(cfg) -> Dict[int, Market]:
    """Create Market objects from config."""
    return {
        m_cfg.market_id: Market(
            market_id=m_cfg.market_id,
            mu_true=m_cfg.mu_true,
            sigma_true=m_cfg.sigma_true,
            signal_noise_sd=m_cfg.signal_noise_sd,
        )
        for m_cfg in cfg.markets
    }


def create_scout(cfg, markets, policy_fn, policy_kwargs) -> Scout:
    """Create a Scout with priors and a policy."""
    market_posteriors = {
        m_id: NormalNormal.from_prior(
            m0=cfg.prior_m0,
            tau0_sq=cfg.prior_tau0_sq,
            sigma_sq=cfg.assumed_sigma_sq,
        )
        for m_id in markets.keys()
    }

    return Scout(
        market_posteriors=market_posteriors,
        policy_fn=policy_fn,
        policy_kwargs=policy_kwargs,
    )


# --------------------------------------------------------------
#   RUNNERS
# --------------------------------------------------------------

def run_policy(cfg, policy_fn, policy_kwargs):
    """Run simulation once with a given policy."""
    markets = build_markets_from_config(cfg)
    scout = create_scout(cfg, markets, policy_fn, policy_kwargs)
    sim = Simulation(markets=markets, scout=scout, T=cfg.T, X=cfg.X)
    sim.run()
    return sim.results(), sim.logs


def run_policy_series(cfg, policy_fn, policy_kwargs):
    """Return the sequence of market choices (one simulation)."""
    _, logs = run_policy(cfg, policy_fn, policy_kwargs)
    return [log["market_id"] for log in logs]


def run_multiple(cfg, policy_fn, policy_kwargs, n_runs=50):
    """Run many simulations → list of time series."""
    return [
        run_policy_series(cfg, policy_fn, policy_kwargs)
        for _ in range(n_runs)
    ]


# --------------------------------------------------------------
#   MAIN
# --------------------------------------------------------------

def main() -> None:
    cfg = default_config()
    markets = [m.market_id for m in cfg.markets]

    print("Running all 3 policies...\n")

    # ----------------------------------------------------------
    # 1) FINAL DISTRIBUTION (one run per policy)
    # ----------------------------------------------------------

    final_results = {}

    # GREEDY
    out, _ = run_policy(
        cfg,
        policy_fn=lambda means, **_: policy.greedy(means),
        policy_kwargs={}
    )
    final_results["greedy"] = out["choices_per_market"]

    # EPSILON GREEDY
    out, _ = run_policy(
        cfg,
        policy_fn=policy.epsilon_greedy,
        policy_kwargs={"epsilon": cfg.epsilon}
    )
    final_results["epsilon_greedy"] = out["choices_per_market"]

    # THOMPSON
    out, _ = run_policy(
        cfg,
        policy_fn=policy.thompson_sampling,
        policy_kwargs={}
    )
    final_results["thompson"] = out["choices_per_market"]

    # --- VISUALIZE 1 ---
    print("Displaying: Final Visit Distribution...")
    plot_policy_results(final_results)



    # ----------------------------------------------------------
    # 2) TIME-SERIES (single simulation)
    # ----------------------------------------------------------

    # print("Displaying: Time Series (Single Run)...")

    # time_series_single = {
    #     "greedy": run_policy_series(cfg, lambda means, **_: policy.greedy(means), {}),
    #     "epsilon_greedy": run_policy_series(cfg, policy.epsilon_greedy, {"epsilon": cfg.epsilon}),
    #     "thompson": run_policy_series(cfg, policy.thompson_sampling, {}),
    # }

    # # --- VISUALIZE 2 ---
    # plot_time_distribution(time_series_single, cfg.T, markets)



    # ----------------------------------------------------------
    # 3) TIME-SERIES PROBABILITIES (multi-run)
    # ----------------------------------------------------------

    N_SIM = 500
    print(f"Running {N_SIM} simulations per policy for probability curves...")

    time_series_multi = {
        "greedy": run_multiple(cfg, lambda means, **_: policy.greedy(means), {}, N_SIM),
        "epsilon_greedy": run_multiple(cfg, policy.epsilon_greedy, {"epsilon": cfg.epsilon}, N_SIM),
        "thompson": run_multiple(cfg, policy.thompson_sampling, {}, N_SIM),
    }

    # --- VISUALIZE 3 ---
    print("Displaying: Choice Probability Over Time (Smoothed over runs)...")
    plot_market_choice_probabilities(
        time_series_multi, cfg.T, markets, n_simulations=N_SIM
    )


    print("\nDone.")


if __name__ == "__main__":
    main()
