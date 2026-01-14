from typing import TYPE_CHECKING

import torch
from torch import nn

from .mlp import MLP


class Critic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int = 0,
        hidden_dims=(256, 256),
        activation: type[nn.Module] = nn.ReLU,
        layer_norm: bool = False,
        state_action_input: bool = False,
    ):
        super().__init__()
        if state_action_input:
            observation_dim += action_dim

        self.critic = MLP(
            dim_in=observation_dim,
            dim_out=1,
            hidden_dims=hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    if TYPE_CHECKING:
        __call__ = forward
