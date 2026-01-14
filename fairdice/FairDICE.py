from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
import torchinfo
from torch import Tensor, nn, optim
from torch.nn.functional import relu

from .critic import Critic
from .divergence import FDivergence, f, f_derivative_inverse
from .policy import GaussianPolicy, MuNetwork

if TYPE_CHECKING:
    from ..dataset import Batch  # ty:ignore[unresolved-import]

COMPILE_OPTIONS = {
    "epilogue_fusion": True,
    "max_autotune": True,
    "shape_padding": True,
    "triton.cudagraphs": True,
    "coordinate_descent_tuning": True,
}
torch.set_float32_matmul_precision("high")


class FairDICE(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        torch.manual_seed(config.seed)
        self.config = config
        self.policy = GaussianPolicy(
            input_dim=config.STATE_DIM,
            hidden_dims=config.HIDDEN_DIMS,
            action_dim=config.ACTION_DIM,
            activation=nn.ReLU,
            temperature=config.temperature,
            tanh_squash_distribution=config.tanh_squash_distribution,
            layer_norm=config.layer_norm,
        )
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=config.policy_lr)
        self.policy_sched = optim.lr_scheduler.CosineAnnealingLR(
            self.policy_optim, T_max=config.total_train_steps
        )

        self.nu = Critic(
            observation_dim=config.STATE_DIM,
            hidden_dims=config.HIDDEN_DIMS,
            layer_norm=config.layer_norm,
        )
        self.nu_optim = optim.Adam(self.nu.parameters(), lr=config.nu_lr)

        self.mu = MuNetwork(config.REWARD_DIM)
        self.mu_optim = optim.Adam(self.mu.parameters(), lr=config.mu_lr)

        self.policy.compile(options=COMPILE_OPTIONS)
        self.nu.compile(options=COMPILE_OPTIONS)
        self.apply_grad_penalty = torch.jit.script(
            _GradPenalty(self.nu, self.config.gradient_penalty_coeff)
        )

    def save(self, path: str | Path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> FairDICE:
        state_dict = torch.load(path)
        self = cls(state_dict["config"])
        self.load_state_dict(state_dict)
        return self

    def step(self, batch: Batch):
        self.nu_optim.zero_grad()
        self.mu_optim.zero_grad()
        self.policy_optim.zero_grad()

        if self.config.gradient_penalty_coeff:
            # Undocumented gradient penalty in FairDICE:103-112 of original code
            self.apply_grad_penalty(
                batch.is_valids,
                batch.init_states,
                batch.next_states,
            )

        mu_nu_loss, policy_loss = self._loss(batch)

        (mu_nu_loss + policy_loss).backward()
        self.nu_optim.step()
        self.mu_optim.step()
        self.policy_optim.step()
        self.policy_sched.step()

    # @torch.compile(options=COMPILE_OPTIONS)

    @torch.compile(options=COMPILE_OPTIONS)
    def _loss(self, batch: Batch) -> tuple[Tensor, Tensor]:
        f_divergence = FDivergence[self.config.divergence]

        # Update nu (critic) and mu (preference weights)
        curr_nu = self.nu(batch.states)
        next_nu = self.nu(batch.next_states)
        init_nu = self.nu(batch.init_states)
        mu = self.mu()
        k = 1.0 / (mu + 1e-8)  # We rely on optimiser to keep it positive
        weighted_rewards = (batch.rewards @ mu).view(-1, 1)
        e = weighted_rewards + self.config.gamma * next_nu - curr_nu
        w = relu(f_derivative_inverse(e / self.config.beta, f_divergence))

        loss_1 = (1 - self.config.gamma) * init_nu.mean()
        unmasked_2 = w * e - self.config.beta * f(w, f_divergence)
        loss_2 = (unmasked_2 * batch.is_valids).sum() / (batch.is_valids.sum() + 1e-8)
        loss_3 = torch.sum(torch.log(k) - mu * k)

        mu_nu_loss = loss_1 + loss_2 + loss_3

        # Update policy
        action_dist = self.policy(batch.states)
        log_probs = action_dist.log_prob(batch.actions).sum(1)
        e_renorm = e.detach().sub(e.detach().max())
        stable_w = relu(f_derivative_inverse(e_renorm / self.config.beta, f_divergence))
        stable_w /= stable_w.mean() + 1e-8

        policy_loss = -(
            batch.is_valids * stable_w * log_probs
        ).sum() / batch.is_valids.sum().add(1e-8)
        # print(
            # f"{loss_1=} {loss_2=} {loss_3=} {policy_loss=}\n "
            # f"{batch.actions[:5]=} {log_probs[:5]=} {stable_w[:5]=}\n"
            # f"{action_dist.loc[:5]=}\n{action_dist.scale[:5]=}"
        # )
        return mu_nu_loss, policy_loss

    def print_summary(self):
        print("=== Policy ===")
        torchinfo.summary(self.policy)
        print("=== Nu (critic) ===")
        torchinfo.summary(self.nu)
        print("=== Mu (preference weights) ===")
        torchinfo.summary(self.mu)

    def forward(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        return self.policy(obs)

    if TYPE_CHECKING:
        __call__ = forward


class _GradPenalty(nn.Module):
    def __init__(self, critic: Critic, coeff: float):
        super().__init__()
        self.critic = critic
        self.coeff = coeff

    def forward(
        self,
        is_valids: torch.Tensor,
        init_states: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        # The original code had a single scalar eps shared across the batch, but it
        # seems like that would only increase variance without any real advantage
        eps = torch.rand_like(is_valids, dtype=torch.float32)
        interp_states = init_states * eps + next_states * (1 - eps)
        interp_states.requires_grad_(True)
        interp_nu = self.critic(interp_states)
        (interp_grad,) = torch.autograd.grad(
            [interp_nu.sum()],
            [interp_states],
            create_graph=True,
        )
        if interp_grad is None:
            # Should never happen
            return torch.zeros((), device=eps.device)
        # We require the gradient norm of the critic to be small along the obs. dim.
        grad_penalty = (
            self.coeff * relu(torch.linalg.norm(interp_grad, dim=1) - 5).square().mean()
        )
        grad_penalty.backward()
        return grad_penalty
