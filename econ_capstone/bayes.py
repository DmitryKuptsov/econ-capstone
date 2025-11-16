"""Bayesian belief representation for market means mu_m (Normal-Normal model)."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import math
import random


@dataclass
class NormalNormal:
    """Conjugate prior/posterior for mu with known variance sigma_sq.

    Model:
        mu ~ Normal(m0, tau0_sq)
        y | mu ~ Normal(mu, sigma_sq)

    After observing data y_1, ..., y_n:
        posterior mu | y ~ Normal(m_n, tau_n_sq)
    """

    m: float          # current mean of mu
    tau_sq: float     # current variance of mu
    sigma_sq: float   # known variance of observations

    @classmethod
    def from_prior(cls, m0: float, tau0_sq: float, sigma_sq: float) -> "NormalNormal":
        return cls(m=m0, tau_sq=tau0_sq, sigma_sq=sigma_sq)

    def update(self, samples: Iterable[float]) -> None:
        """Update posterior with new observed samples."""
        samples = list(samples)
        if not samples:
            return
        n = len(samples)
        y_bar = sum(samples) / n
        # Precision form
        prior_prec = 1.0 / self.tau_sq
        like_prec = n / self.sigma_sq
        post_prec = prior_prec + like_prec
        self.tau_sq = 1.0 / post_prec
        self.m = (prior_prec * self.m + like_prec * y_bar) * self.tau_sq

    def posterior_mean(self) -> float:
        return self.m

    def posterior_var(self) -> float:
        return self.tau_sq

    def sample_mu(self) -> float:
        """Draw a sample of mu from the current posterior (for Thompson sampling)."""
        return random.gauss(self.m, math.sqrt(self.tau_sq))
