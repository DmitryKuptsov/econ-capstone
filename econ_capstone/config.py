"""Global configuration for the Bayesian scout simulation (1 scout, M markets)."""

from dataclasses import dataclass
from typing import List


@dataclass
class MarketConfig:
    market_id: int
    mu_true: float
    sigma_true: float
    signal_noise_sd: float


@dataclass
class SimulationConfig:
    T: int                   # number of periods
    X: int                   # number of players observed per period in chosen market
    markets: List[MarketConfig]
    # Prior for market mean mu_m: mu_m ~ Normal(m0, tau0_sq)
    prior_m0: float
    prior_tau0_sq: float
    # Assumed known variance of player quality within each market (for beliefs)
    assumed_sigma_sq: float
    # Policy parameters
    epsilon: float           # for epsilon-greedy policy


def default_config() -> SimulationConfig:
    """Return a simple default configuration with 3 markets."""
    markets = [
        MarketConfig(market_id=0, mu_true=1.0, sigma_true=1.0, signal_noise_sd=0.5),
        MarketConfig(market_id=1, mu_true=2.0, sigma_true=1.0, signal_noise_sd=0.5),
        MarketConfig(market_id=2, mu_true=3.0, sigma_true=1.0, signal_noise_sd=0.5),
    ]
    return SimulationConfig(
        T=2000,
        X=10,
        markets=markets,
        prior_m0=0.0,
        prior_tau0_sq=4.0,    # fairly diffuse prior
        assumed_sigma_sq=1.0,
        epsilon=0.1,
    )
