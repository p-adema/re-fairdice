from enum import Enum

import torch


class FDivergence(str, Enum):
    KL = "KL"
    CHI = "Chi"
    SOFT_CHI = "SoftChi"
    DUAL_DICE = "DualDICE"


def f(x, f_divergence: FDivergence, eps: float = 1e-10):
    x = torch.asarray(x, device="cuda" if torch.cuda.is_available() else "cpu")

    if f_divergence == FDivergence.KL:
        return x * torch.log(x + eps)
    if f_divergence == FDivergence.CHI:
        return 0.5 * (x - 1) ** 2
    if f_divergence == FDivergence.SOFT_CHI:
        return torch.where(x < 1, x * torch.log(x + eps) - x + 1, 0.5 * (x - 1) ** 2)
    if f_divergence == FDivergence.DUAL_DICE:
        return 2 / 3 * x.abs() ** (3 / 2)

    raise ValueError(f"Invalid {f_divergence=}")


def f_derivative_inverse(y, f_divergence: FDivergence):
    y = torch.asarray(y, device="cuda" if torch.cuda.is_available() else "cpu")

    if f_divergence == FDivergence.KL:
        return torch.exp(y - 1)
    if f_divergence == FDivergence.CHI:
        return y + 1
    if f_divergence == FDivergence.SOFT_CHI:
        return torch.where(y < 0, y.exp(), y + 1)
    if f_divergence == FDivergence.DUAL_DICE:
        raise NotImplementedError(f"No F'-1 for {f_divergence=}")

    raise ValueError(f"Invalid {f_divergence=}")


def state_action_ratio(
    nu,
    next_nu,
    rewards,
    beta: float,
    discount: float,
    f_divergence: FDivergence,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nu = torch.asarray(nu, device=device)
    next_nu = torch.asarray(next_nu, device=device)
    rewards = torch.asarray(rewards, device=device)

    e = rewards + discount * next_nu - nu
    return torch.nn.functional.relu(f_derivative_inverse(e / beta, f_divergence))
