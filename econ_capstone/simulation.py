"""Simulation engine for the Bayesian scout and markets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from scout import Scout
from market import Market


@dataclass
class Simulation:
    markets: Dict[int, Market]
    scout: Scout
    T: int
    X: int
    logs: List[dict] = field(default_factory=list)

    def run(self) -> None:
        """Run the simulation for T periods."""
        for t in range(1, self.T + 1):
            market_id = self.scout.choose_market(t)
            market = self.markets[market_id]
            players = market.sample_players(self.X)
            signals = market.get_signals(players)
            self.scout.observe(t, market_id, signals)
            self.logs.append(
                {
                    "t": t,
                    "market_id": market_id,
                    "signals": signals,
                    "best_signal_seen": self.scout.best_signal_seen,
                }
            )

    def results(self) -> dict:
        """Return a summary of the simulation results."""
        counts = {m_id: 0 for m_id in self.markets}
        for log in self.logs:
            counts[log["market_id"]] += 1
        return {
            "choices_per_market": counts,
            "best_signal_seen": self.scout.best_signal_seen,
        }
