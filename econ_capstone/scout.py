"""Scout agent: Bayesian learner over markets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Callable

from bayes import NormalNormal


@dataclass
class Scout:
    market_posteriors: Dict[int, NormalNormal]
    policy_fn: Callable[..., int]
    policy_kwargs: Dict
    best_signal_seen: float = float('-inf')
    history: List[dict] = field(default_factory=list)

    def choose_market(self, t: int) -> int:
        """Choose a market according to the policy.

        For epsilon-greedy: policy_fn expects (posterior_means, **policy_kwargs).
        For Thompson: policy_fn expects (posteriors).
        """
        if self.policy_fn.__name__ == "thompson_sampling":
            return self.policy_fn(self.market_posteriors)

        # epsilon-greedy or greedy
        posterior_means = {
            m_id: post.posterior_mean()
            for m_id, post in self.market_posteriors.items()
        }
        return self.policy_fn(posterior_means, **self.policy_kwargs)


    def observe(self, t: int, market_id: int, signals: List[float]) -> None:
        """Observe signals from chosen market, update beliefs and record payoff."""
        # Bayesian update
        self.update_beliefs(market_id, signals)
        # Payoff update
        self.record_payoff(signals)
        # Logging
        self.history.append(
            {
                "t": t,
                "market_id": market_id,
                "signals": signals,
                "best_signal_seen": self.best_signal_seen,
                "posterior_means": {
                    m_id: post.posterior_mean()
                    for m_id, post in self.market_posteriors.items()
                },
            }
        )

    def update_beliefs(self, market_id: int, signals: List[float]) -> None:
        self.market_posteriors[market_id].update(signals)

    def record_payoff(self, signals: List[float]) -> None:
        if not signals:
            return
        local_best = max(signals)
        if local_best > self.best_signal_seen:
            self.best_signal_seen = local_best

    def get_posterior_summary(self, market_id: int) -> dict:
        post = self.market_posteriors[market_id]
        return {
            "mean": post.posterior_mean(),
            "var": post.posterior_var(),
        }
