"""Player entity: latent quality and (optional) observed signal."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Player:
    quality_true: float
    signal: Optional[float] = None

    def set_signal(self, value: float) -> None:
        self.signal = value
