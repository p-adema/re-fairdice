from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import torch.distributions as dist
from torch import nn

from .mlp import MLP, orthogonal_init

# Values to clamp the log std deviations to for continous action spaces
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class MuNetwork(nn.Module):
    def __init__(self, reward_dim: int):
        super().__init__()
        self.mu = nn.Parameter(torch.full((reward_dim,), 1.0, requires_grad=True))

    def forward(self):
        return self.mu

    if TYPE_CHECKING:
        __call__ = forward


class DiscretePolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super().__init__()
        self.mlp = MLP(input_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.mlp(obs)
        logits: torch.Tensor = self.action_head(x)
        return logits, logits.softmax(dim=-1)

    if TYPE_CHECKING:
        __call__ = forward


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        action_dim: int,
        activation: Callable[[], nn.Module] = nn.ReLU,
        temperature: float = 1.0,
        log_std_scale: float = 1e-3,
        tanh_squash_distribution: bool = True,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        self.tanh_squash_distribution = tanh_squash_distribution

        self.mlp = MLP(
            input_dim,
            hidden_dims[-1],
            hidden_dims[:-1],
            activation=activation,
            activate_final=True,
            layer_norm=layer_norm,
        )
        self.mean_head = orthogonal_init(nn.Linear(hidden_dims[-1], action_dim))
        self.std_head = orthogonal_init(
            nn.Linear(hidden_dims[-1], action_dim), log_std_scale
        )

    def forward(self, obs: torch.Tensor) -> dist.Distribution:
        x = self.mlp(obs)

        means = self.mean_head(x)
        if not self.tanh_squash_distribution:
            means = means.tanh()

        log_stds = self.std_head(x).clip(LOG_STD_MIN, LOG_STD_MAX)
        normal = dist.Normal(means, log_stds.exp() * self.temperature)

        if not self.tanh_squash_distribution:
            return normal

        return dist.TransformedDistribution(normal, dist.TanhTransform())

    if TYPE_CHECKING:
        __call__ = forward


class MNDPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        activation: Callable[[], nn.Module] = nn.ReLU,
        n_mixtures: int = 5,
        tanh_squash_distribution: bool = True,
        temperature: float = 1.0,
        log_std_scale: float = 1e-3,
        layer_norm: bool = False,
    ):
        super().__init__()

        self.n_mixtures = n_mixtures
        self.action_dim = action_dim
        self.temperature = temperature
        self.tanh_squash_distribution = tanh_squash_distribution

        self.mlp = MLP(
            input_dim,
            hidden_dims[-1],
            hidden_dims[:-1],
            activation=activation,
            activate_final=True,
            layer_norm=layer_norm,
        )
        self.mean_head = orthogonal_init(
            nn.Linear(hidden_dims[-1], action_dim * n_mixtures)
        )
        self.std_head = orthogonal_init(
            nn.Linear(hidden_dims[-1], action_dim * n_mixtures), log_std_scale
        )
        self.logit_head = orthogonal_init(nn.Linear(hidden_dims[-1], n_mixtures))

    def forward(self, obs: torch.Tensor) -> dist.Distribution:
        x = self.mlp(obs)
        means = self.mean_head(x).view(obs.size(0), self.n_mixtures, self.action_dim)
        if not self.tanh_squash_distribution:
            means = means.tanh()

        log_stds = (
            self.std_head(x)
            .clip(LOG_STD_MIN, LOG_STD_MAX)
            .view(obs.size(0), self.n_mixtures, self.action_dim)
        )

        logits = self.logit_head(x)

        multinorm = dist.MixtureSameFamily(
            mixture_distribution=dist.Categorical(logits=logits),
            component_distribution=dist.Normal(
                means, log_stds.exp() * self.temperature
            ),
        )

        if not self.tanh_squash_distribution:
            return multinorm

        return dist.TransformedDistribution(multinorm, dist.TanhTransform())

    if TYPE_CHECKING:
        __call__ = forward
