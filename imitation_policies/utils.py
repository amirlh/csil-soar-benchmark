"""Shared building blocks for imitation algorithms (CSIL, CSIL+SOAR).

Three pieces:
- MLP: feedforward network for both actor logits and Q-values.
       The paper § L (continuous MuJoCo) uses ELU + LayerNorm-in-critic.
       For small discrete envs (CartPole/Acrobot), ReLU without LayerNorm
       trains more reliably — we keep the paper architecture optional via
       `activation` and `use_layernorm` flags but default to ReLU/no-LN.
- ReplayBuffer: FIFO buffer storing Transition objects, random minibatch sampling
- polyak_update: soft target network update used by SAC-family critics
"""

from __future__ import annotations

import random
from collections import deque

import torch
import torch.nn as nn

from imitation_policies.imitation_policy import Transition


class MLP(nn.Module):
    """Two-hidden-layer MLP.

    Default: ReLU activations, no LayerNorm (works well on small discrete envs).
    Pass `activation='elu', use_layernorm=True` to match the paper's MuJoCo setup.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden=(256, 256),
        use_layernorm: bool = False,
        activation: str = "relu",
    ):
        super().__init__()
        act_cls = {"relu": nn.ReLU, "elu": nn.ELU}[activation.lower()]
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(act_cls())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, transitions) -> None:
        if isinstance(transitions, Transition):
            self.buffer.append(transitions)
        else:
            self.buffer.extend(transitions)

    def sample(self, batch_size: int) -> list[Transition]:
        n = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, n)

    def __len__(self) -> int:
        return len(self.buffer)


def polyak_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)
