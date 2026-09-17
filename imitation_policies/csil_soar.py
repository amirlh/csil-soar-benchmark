"""CSIL+SOAR — Coherent Soft Imitation Learning with IL-SOAR ensemble exploration.

Paper-faithful implementation of Algorithms 5, 6, 7 of Viel et al. 2025
(arXiv 2502.19859, "IL-SOAR : Imitation Learning with Soft Optimistic Actor cRitic").

Inherits all the Algorithm 2 fixes from CSILAgent (early-stopped BC, reward
refinement, weight-decay-free SAC actor, corrected reward formula, LayerNorm
critic) and adds:
  - single critic Q_phi replaced by L=4 ensemble members
  - each ensemble member is a TWIN pair (Q_l^(1), Q_l^(2)) per Algorithm 7,
    with min-of-twins target (reward form; paper uses max in cost form,
    Footnote 8 sign flip)
  - each Q_l TWIN is trained INDEPENDENTLY vs its own pair of targets on
    its own minibatch (bootstrap variant required by IL-SOAR Corollary 4.11)
  - actor uses OPTIMISTIC Q = mean_l Q_l + clip(std_l Q_l, 0, sigma_clip)
    where Q_l(s,a) = min(Q_l^(1)(s,a), Q_l^(2)(s,a))
    (reward formulation: upper-confidence bound, paper Footnote 8 sign flip)

Stages A (BC) and B (SARSA warm-up) reuse the CSIL machinery — SOAR augments
only the SAC fine-tune (Stage C) critic and actor, plus extends Stage B to
initialize the full 2*L twin-ensemble.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from imitation_policies.csil import CSILAgent
from imitation_policies.imitation_policy import Transition
from imitation_policies.utils import MLP, polyak_update


DEFAULT_CONFIGS_SOAR: dict[str, dict] = {
    # Inherits CSIL's paper-faithful defaults plus IL-SOAR ensemble parameters
    # (L=4 twin pairs per Algorithm 7, σ_clip per § 2.4).
    "CartPole-v1": dict(
        alpha=1.0, beta=0.05,
        n_bc_steps=5_000, n_sarsa_steps=2_000, n_grad_steps_per_iter=16,
        L=4, sigma_clip=0.5, lr=1e-3,
        bc_early_stop=True,
        reward_refinement=True, reward_lr=1e-4,
    ),
    "Acrobot-v1": dict(
        alpha=1.0, beta=0.10,
        n_bc_steps=5_000, n_sarsa_steps=2_000, n_grad_steps_per_iter=16,
        L=4, sigma_clip=1.0, lr=1e-3,
        negative_reward=True,
        bc_early_stop=True,
        reward_refinement=True, reward_lr=1e-4,
    ),
}


def default_config_soar(env_name: str) -> dict:
    return dict(DEFAULT_CONFIGS_SOAR.get(env_name, {}))


class CSILSOARAgent(CSILAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        L: int = 4,
        sigma_clip: float = 1.0,
        hidden=(256, 256),
        lr: float = 1e-3,
        save_path: str = "csil_soar_agent.pth",
        **csil_kwargs,
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden=hidden,
            lr=lr,
            save_path=save_path,
            **csil_kwargs,
        )

        # Drop parent's single critic; SOAR uses a twin ensemble of size L.
        del self.q_net, self.q_target, self.opt_q

        self.L = L
        self.sigma_clip = sigma_clip

        # Twin ensemble: each member l has two critics (Q_l^(1), Q_l^(2))
        # per Algorithm 7. Total = 2 * L critics (and 2 * L targets).
        self.q_nets_a = nn.ModuleList([
            MLP(state_dim, action_dim, hidden=hidden).to(self.device)
            for _ in range(L)
        ])
        self.q_nets_b = nn.ModuleList([
            MLP(state_dim, action_dim, hidden=hidden).to(self.device)
            for _ in range(L)
        ])
        self.q_targets_a = nn.ModuleList([copy.deepcopy(q) for q in self.q_nets_a])
        self.q_targets_b = nn.ModuleList([copy.deepcopy(q) for q in self.q_nets_b])
        for q in list(self.q_targets_a) + list(self.q_targets_b):
            for p in q.parameters():
                p.requires_grad_(False)

        self.opt_qs_a = [torch.optim.Adam(q.parameters(), lr=lr) for q in self.q_nets_a]
        self.opt_qs_b = [torch.optim.Adam(q.parameters(), lr=lr) for q in self.q_nets_b]

    # ----- Optimistic Q (UCB across the L-ensemble, twin min within) -------

    def optimistic_q(self, s: torch.Tensor, a: torch.Tensor | None = None) -> torch.Tensor:
        """UCB optimistic Q-value across the L-ensemble (IL-SOAR Algo 5, reward-flipped).

        Within each ensemble member l, the aggregated Q is the min of the two
        twin critics (Algo 7 line 11 in cost form -> min in reward form,
        anti-overestimation, paper Footnote 8):

            Q_l(s, a) = min(Q_l^(1)(s, a), Q_l^(2)(s, a))

        Across the L members, the optimistic estimate is:

            Q_opt(s, a) = mean_l Q_l(s, a) + clip(std_l Q_l, 0, sigma_clip)

        Uses online critics (self.q_nets_a/b), not targets. unbiased=False
        matches paper's 1/L denominator (biased estimator, Algo 5 line 4).
        """
        if a is None:
            per_l = [
                torch.minimum(qa(s), qb(s))
                for qa, qb in zip(self.q_nets_a, self.q_nets_b)
            ]
        else:
            a_idx = a.unsqueeze(1)
            per_l = [
                torch.minimum(qa(s).gather(1, a_idx), qb(s).gather(1, a_idx))
                for qa, qb in zip(self.q_nets_a, self.q_nets_b)
            ]
        qs = torch.stack(per_l, dim=0)
        q_mean = qs.mean(dim=0)
        q_std = (qs.var(dim=0, unbiased=False) + 1e-8).sqrt().clamp(0.0, self.sigma_clip)
        return q_mean + q_std

    # ----- Helpers ---------------------------------------------------------

    def _twin_backup_next_value(self, ell: int, s_next: torch.Tensor,
                                 a_next: torch.Tensor | None = None) -> torch.Tensor:
        """Compute min-of-twins target value at next state (Algo 7 line 3 + 4 backup).

        If a_next is given (SARSA, Stage B): returns min(Q_targ^(1)(s', a'),
        Q_targ^(2)(s', a')) of shape [B, 1].

        If a_next is None (Stage C SAC, expected over actor): returns the
        full [B, A] min-of-twin tensor for caller to combine with pi(s_next).
        """
        if a_next is None:
            qa = self.q_targets_a[ell](s_next)
            qb = self.q_targets_b[ell](s_next)
            return torch.minimum(qa, qb)
        an_idx = a_next.unsqueeze(1)
        qa = self.q_targets_a[ell](s_next).gather(1, an_idx)
        qb = self.q_targets_b[ell](s_next).gather(1, an_idx)
        return torch.minimum(qa, qb)

    # ----- Stage B: SARSA warm-up on ALL L twin pairs ---------------------

    def _run_stage_b(self, expert_batch):
        """Initialize each twin pair (Q_l^(1), Q_l^(2)) independently via TD-SARSA.

        Bootstrap variant: each ensemble member l samples its own minibatch
        (paper Corollary 4.11 — joint independence required for the UCB std
        to be meaningful). Within each l, both twins train on the SAME
        minibatch with the SAME backup (paper Algo 7 line 1: shared B for the
        twin pair).
        """
        s_exp, a_exp, sn_exp, d_exp, _, _ = Transition.to_tensor(expert_batch, self.device)

        s = s_exp[:-1]
        a = a_exp[:-1]
        s_next = sn_exp[:-1]
        a_next = a_exp[1:]
        d = d_exp[:-1]

        valid = d == 0
        s, a, s_next, a_next, d = s[valid], a[valid], s_next[valid], a_next[valid], d[valid]
        n_valid = s.size(0)
        if n_valid == 0:
            return
        if n_valid < self.batch_size:
            print(f"[CSIL+SOAR Stage B] only {n_valid} valid pairs (< batch_size={self.batch_size}); sampling with replacement")

        for step in range(self.n_sarsa_steps):
            for ell in range(self.L):
                idx = torch.randint(0, n_valid, (self.batch_size,), device=self.device)
                s_b = s[idx]
                a_b = a[idx]
                sn_b = s_next[idx]
                an_b = a_next[idx]
                d_b = d[idx].unsqueeze(1)

                with torch.no_grad():
                    log_bc = F.log_softmax(self.bc_policy(s_b), dim=-1)
                    r = self.alpha * (log_bc.gather(1, a_b.unsqueeze(1)) - self.log_p)
                    if self.negative_reward:
                        r = r - self.alpha * (log_bc.max(dim=-1, keepdim=True).values - self.log_p)
                    # Min-of-twins target (Algo 7 line 3, reward form)
                    q_next = self._twin_backup_next_value(ell, sn_b, an_b)
                    y = r + self.gamma * (1.0 - d_b) * q_next

                # Both twins regress on the SAME backup y (Algo 7 lines 5-6).
                a_idx = a_b.unsqueeze(1)
                q_pred_a = self.q_nets_a[ell](s_b).gather(1, a_idx)
                q_pred_b = self.q_nets_b[ell](s_b).gather(1, a_idx)
                loss_a = (q_pred_a - y).pow(2).mean()
                loss_b = (q_pred_b - y).pow(2).mean()

                self.opt_qs_a[ell].zero_grad()
                loss_a.backward()
                self.opt_qs_a[ell].step()

                self.opt_qs_b[ell].zero_grad()
                loss_b.backward()
                self.opt_qs_b[ell].step()

                polyak_update(self.q_targets_a[ell], self.q_nets_a[ell], self.tau)
                polyak_update(self.q_targets_b[ell], self.q_nets_b[ell], self.tau)

    # ----- Stage C critic: L independent twin SAC critic updates ----------

    def _stage_c_critic_step(self, batch):
        """L independent SAC critic steps, one per ensemble member (bootstrap),
        each updating BOTH twin critics on the same minibatch with min-of-twins
        target (Algorithm 7 + reward-form sign flip).
        """
        for ell in range(self.L):
            batch_ell = self.replay_buffer.sample(self.batch_size)
            s, a, s_next, d, _, _ = Transition.to_tensor(batch_ell, self.device)
            d = d.unsqueeze(1)
            a_idx = a.unsqueeze(1)

            with torch.no_grad():
                # r_θ_1 uses FROZEN bc_policy (Algo 1) — stable target for Q.
                log_bc_s = F.log_softmax(self.bc_policy(s), dim=-1)
                r = self.alpha * (log_bc_s.gather(1, a_idx) - self.log_p)
                if self.negative_reward:
                    r = r - self.alpha * (log_bc_s.max(dim=-1, keepdim=True).values - self.log_p)

                logits_next = self.actor(s_next)
                pi_next = F.softmax(logits_next, dim=-1)
                log_pi_next = F.log_softmax(logits_next, dim=-1)
                log_bc_next = F.log_softmax(self.bc_policy(s_next), dim=-1)

                # Min-of-twins target Q (Algo 7 line 3, reward form)
                q_tgt_next = self._twin_backup_next_value(ell, s_next)
                v_next = (pi_next * (q_tgt_next - self.beta * (log_pi_next - log_bc_next))).sum(
                    dim=-1, keepdim=True
                )
                y = r + self.gamma * (1.0 - d) * v_next

            # Both twins regress on the SAME backup y (Algo 7 lines 5-6).
            q_pred_a = self.q_nets_a[ell](s).gather(1, a_idx)
            q_pred_b = self.q_nets_b[ell](s).gather(1, a_idx)
            loss_a = (q_pred_a - y).pow(2).mean()
            loss_b = (q_pred_b - y).pow(2).mean()

            self.opt_qs_a[ell].zero_grad()
            loss_a.backward()
            self.opt_qs_a[ell].step()

            self.opt_qs_b[ell].zero_grad()
            loss_b.backward()
            self.opt_qs_b[ell].step()

            polyak_update(self.q_targets_a[ell], self.q_nets_a[ell], self.tau)
            polyak_update(self.q_targets_b[ell], self.q_nets_b[ell], self.tau)

    # ----- Stage C actor: KL-vs-BC with OPTIMISTIC Q ----------------------

    def _stage_c_actor_step(self, batch):
        """J_pi using the UCB optimistic Q. Like the parent, mixes batch (replay)
        with a same-size sample from the expert cache (paper § I "combined
        equally"). Restores parent's grad clipping (max_norm=1.0) — optimistic
        Q is larger than plain Q, so the actor gradient can blow up logits
        without clipping.
        """
        s, _, _, _, _, _ = Transition.to_tensor(batch, self.device)
        if self._expert_states is not None and self._expert_states.size(0) > 0:
            idx_e = torch.randint(0, self._expert_states.size(0), (s.size(0),), device=self.device)
            s = torch.cat([s, self._expert_states[idx_e]], dim=0)

        logits = self.actor(s)
        pi = F.softmax(logits, dim=-1)
        log_pi = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            log_bc = F.log_softmax(self.bc_policy(s), dim=-1)
            q_val = self.optimistic_q(s)  # [B, action_dim]

        loss_pi = (pi * (self.beta * (log_pi - log_bc) - q_val)).sum(dim=-1).mean()

        self.opt_actor.zero_grad()
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.opt_actor.step()

    # ----- save / load (twin-ensemble-aware) ------------------------------

    def save(self):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "q_nets_a": [q.state_dict() for q in self.q_nets_a],
                "q_nets_b": [q.state_dict() for q in self.q_nets_b],
                "q_targets_a": [q.state_dict() for q in self.q_targets_a],
                "q_targets_b": [q.state_dict() for q in self.q_targets_b],
                "bc_policy": self.bc_policy.state_dict() if self.bc_policy is not None else None,
                "stage": self._stage,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "L": self.L,
                "sigma_clip": self.sigma_clip,
            },
            self.save_path,
        )

    def load(self, path: str | None = None):
        path = path or self.save_path
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        for ell, sd in enumerate(ck["q_nets_a"]):
            self.q_nets_a[ell].load_state_dict(sd)
        for ell, sd in enumerate(ck["q_nets_b"]):
            self.q_nets_b[ell].load_state_dict(sd)
        for ell, sd in enumerate(ck["q_targets_a"]):
            self.q_targets_a[ell].load_state_dict(sd)
        for ell, sd in enumerate(ck["q_targets_b"]):
            self.q_targets_b[ell].load_state_dict(sd)
        if ck["bc_policy"] is not None:
            self.bc_policy = copy.deepcopy(self.actor)
            self.bc_policy.load_state_dict(ck["bc_policy"])
            for p in self.bc_policy.parameters():
                p.requires_grad_(False)
            self.bc_policy.eval()
        self._stage = ck.get("stage", "bc")
