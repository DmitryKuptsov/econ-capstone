"""Market environment: generates players and noisy signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import random

from player import Player


@dataclass
class Market:
    market_id: int
    mu_true: float
    sigma_true: float          # interpreted as standard deviation
    signal_noise_sd: float

    def sample_players(self, X: int) -> List[Player]:
        """Generate X players with latent qualities from N(mu_true, sigma_true^2)."""
        players: List[Player] = []
        for _ in range(X):
            q = random.gauss(self.mu_true, self.sigma_true)
            players.append(Player(quality_true=q))
        return players

    def get_signals(self, players: List[Player]) -> List[float]:
        """Return noisy signals for given players and update their signal attribute."""
        signals = []
        for p in players:
            eps = random.gauss(0.0, self.signal_noise_sd)
            s = p.quality_true + eps
            p.set_signal(s)
            signals.append(s)
        return signals
