"""Policies for market selection based on scout's beliefs."""

from __future__ import annotations
from typing import Dict
import random

from bayes import NormalNormal


def greedy(posterior_means: Dict[int, float]) -> int:
    """Pick the market with highest posterior mean."""
    return max(posterior_means, key=posterior_means.get)


def epsilon_greedy(posterior_means: Dict[int, float], epsilon: float) -> int:
    """With prob epsilon, explore randomly; otherwise exploit the best market."""
    if random.random() < epsilon:
        return random.choice(list(posterior_means.keys()))
    return greedy(posterior_means)


def thompson_sampling(posteriors: Dict[int, NormalNormal]) -> int:
    """Sample a mu from each posterior and choose the argmax."""
    samples = {m_id: post.sample_mu() for m_id, post in posteriors.items()}
    return max(samples, key=samples.get)
