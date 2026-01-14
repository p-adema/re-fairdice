import itertools
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import nn


def orthogonal_init(layer: nn.Module, gain: float = math.sqrt(2)):
    for name, param in layer.named_parameters():
        if "bias" in name:
            nn.init.zeros_(param)
        elif "weight" in name:
            nn.init.orthogonal_(param, gain=gain)
    return layer


class MLP(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        activation: Callable[[], nn.Module] = nn.ReLU,
        activate_final: bool = False,
        dropout: float = 0.0,
        layer_norm: bool = False,
    ):
        super().__init__()
        dims = [dim_in, *hidden_dims, dim_out]

        layers = []
        for i, (h_in, h_out) in enumerate(itertools.pairwise(dims)):
            layers.append(orthogonal_init(nn.Linear(h_in, h_out)))
            if i < len(dims) - 2:
                if dropout:
                    layers.append(nn.Dropout(dropout))
                layers.append(activation())
                if layer_norm:
                    layers.append(nn.LayerNorm(h_out))

        if activate_final:
            layers.append(activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    if TYPE_CHECKING:
        __call__ = forward
