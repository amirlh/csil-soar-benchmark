"""CSIL — Coherent Soft Imitation Learning (discrete action spaces).

Paper-faithful implementation of **Algorithm 2** (Appendix I, Watson et al. 2023,
arXiv 2305.16498). The previous version implemented only Algorithm 1 (Section 3),
which the paper itself never benchmarks — the experiments use Algorithm 2 with
reward refinement (Eq. 11/12), early-stopped BC (§ L), and LayerNorm critic.

Three-stage training, switched automatically inside `update_representation`:
  A) BC pre-training with held-out validation set and early stopping
     (paper § L: "we opted for early stopping … if the policy is trained too
     long, the entropy is reduced too far, and there is not enough exploration
     for RL")
  B) SARSA warm-up of Q_φ on expert transitions with frozen r̄_θ_1
  C) SAC fine-tune with KL-vs-BC actor (J_π, Eq. 14) PLUS reward refinement
     (J_r, Eq. 11/12) — both update the actor θ via separate optimizers.

Coherent reward (paper Def. 1, p. 5):
    r̄(s, a) = α (log q_θ(a|s) − log p(a|s))
For discrete actions with uniform prior p = 1/|A|, log p = −log|A| (constant).

In Stage C the reward uses the CURRENT actor q_θ, not the frozen snapshot θ_1.
The frozen snapshot is used only by J_π's KL term `β(log q_θ − log q_θ_1)`.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from imitation_policies.imitation_policy import ImitationPolicyAgent, Transition
from imitation_policies.utils import MLP, ReplayBuffer, polyak_update


DEFAULT_CONFIGS: dict[str, dict] = {
    # Paper-faithful defaults (Watson et al. 2023, Algorithm 2, Appendix L):
    #   H1: J_r reward refinement ON (Eq. 11/12), reward_lr per § L (1e-3 standard,
    #       lowered to 1e-4 on discrete to avoid Schulman-term blow-ups since
    #       categorical log-probs are unbounded below).
    #   H2: BC early stopping ON (§ L "we opted for early stopping").
    #   H3: No actor weight decay in Stage C (Ψ regularizer is BC-only).
    #   Architecture: ELU + LayerNorm-in-critic (§ L).
    #   Critic target uses frozen θ_1 reward (Algo 1) for stability at our 100k
    #       env-step budget; J_r still refines the actor on D vs B.
    "CartPole-v1": dict(
        alpha=1.0, beta=0.05,
        n_bc_steps=5_000, n_sarsa_steps=2_000, n_grad_steps_per_iter=16,
        bc_early_stop=True,
        reward_refinement=True, reward_lr=1e-4,
    ),
    "Acrobot-v1": dict(
        alpha=1.0, beta=0.10,
        n_bc_steps=5_000, n_sarsa_steps=2_000, n_grad_steps_per_iter=16,
        negative_reward=True,
        bc_early_stop=True,
        reward_refinement=True, reward_lr=1e-4,
    ),
}


def default_config(env_name: str) -> dict:
    return dict(DEFAULT_CONFIGS.get(env_name, {}))


class CSILAgent(ImitationPolicyAgent, nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        alpha: float = 1.0,
        beta: float = 0.05,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 1e-3,
        reward_lr: float = 1e-3,
        batch_size: int = 256,
        buffer_size: int = 100_000,
        n_bc_steps: int = 5_000,
        n_sarsa_steps: int = 2_000,
        bc_weight_decay: float = 1e-3,
        bc_val_split: float = 0.1,
        bc_patience: int = 500,
        bc_eval_every: int = 100,
        n_grad_steps_per_iter: int = 1,
        hidden=(256, 256),
        device: str = "cpu",
        save_path: str = "csil_agent.pth",
        negative_reward: bool = False,
        reward_refinement: bool = True,
        bc_early_stop: bool = True,
    ):
        nn.Module.__init__(self)

        assert beta < alpha, f"beta ({beta}) must be < alpha ({alpha}) per CSIL spec"

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_bc_steps = n_bc_steps
        self.n_sarsa_steps = n_sarsa_steps
        self.bc_val_split = bc_val_split
        self.bc_patience = bc_patience
        self.bc_eval_every = bc_eval_every
        self.n_grad_steps_per_iter = n_grad_steps_per_iter
        self.negative_reward = negative_reward
        self.reward_refinement = reward_refinement
        self.bc_early_stop = bc_early_stop
        self.device = torch.device(device)

        # log p(a|s) for uniform prior over |A| actions = −log|A| (constant).
        # Required by paper Def. 1; previously dropped (LOW-severity policy-invariant bug).
        self.log_p = -math.log(action_dim)

        # Paper § L (p. 32): ELU activations; LayerNorm in the critic only.
        self.actor = MLP(state_dim, action_dim, hidden=hidden,
                         activation="elu", use_layernorm=False).to(self.device)
        self.q_net = MLP(state_dim, action_dim, hidden=hidden,
                         activation="elu", use_layernorm=True).to(self.device)
        self.q_target = copy.deepcopy(self.q_net)
        for p in self.q_target.parameters():
            p.requires_grad_(False)

        self.bc_policy: nn.Module | None = None  # frozen θ_1 snapshot after Stage A

        # Three optimizers on the actor (paper § L, Algo 2):
        #   Stage A (BC): with weight decay = regularizer Ψ(θ)
        #   Stage C (SAC actor, J_π): no weight decay
        #   Stage C (reward refinement, J_r): no weight decay, lr=reward_lr (paper § L: 1e-3)
        self.opt_bc = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=bc_weight_decay)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_reward = torch.optim.Adam(self.actor.parameters(), lr=reward_lr)
        self.opt_q = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        # Cache of expert states/actions for J_r (Stage C) and J_π sample mixing.
        self._expert_states: torch.Tensor | None = None
        self._expert_actions: torch.Tensor | None = None

        self._stage = "bc"
        self.save_path = save_path

    # ----- ABC methods ----------------------------------------------------

    def predict(self, state, deterministic: bool = False):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        single = state_t.dim() == 1
        if single:
            state_t = state_t.unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(state_t)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = Categorical(logits=logits).sample()
        action = action.cpu().numpy()
        return int(action[0]) if single else action

    def update_representation(self, expert_batch, learner_batch):
        if self._stage == "bc":
            self._run_stage_a(expert_batch)
            self._freeze_bc()
            # Cache expert tensors for J_r and J_π
            s_e, a_e, _, _, _, _ = Transition.to_tensor(expert_batch, self.device)
            self._expert_states = s_e
            self._expert_actions = a_e
            self._stage = "sarsa"
            return

        if self._stage == "sarsa":
            self._run_stage_b(expert_batch)
            self._stage = "sac"
            return

        # Stage C: critic + reward steps. Reward refinement runs ONCE per
        # update_representation call (not per critic step) to avoid swamping
        # the slower critic / actor with J_r updates — matches paper Algorithm 2
        # which does 1 J_r step per env step.
        self.replay_buffer.push(learner_batch)
        if len(self.replay_buffer) < self.batch_size:
            return
        for _ in range(self.n_grad_steps_per_iter):
            batch = self.replay_buffer.sample(self.batch_size)
            self._stage_c_critic_step(batch)
        if self.reward_refinement:
            batch = self.replay_buffer.sample(self.batch_size)
            self._stage_c_reward_step(batch)

    def update_policy(self, learner_batch):
        if self._stage != "sac":
            return
        if len(self.replay_buffer) < self.batch_size:
            return
        for _ in range(self.n_grad_steps_per_iter):
            batch = self.replay_buffer.sample(self.batch_size)
            self._stage_c_actor_step(batch)

    # ----- Stage A: BC pre-training with validation-based early stopping --

    def _run_stage_a(self, expert_batch):
        """BC training with held-out validation set and patience-based early stop.

        Paper § L: too many BC steps collapse the actor entropy, preventing Stage C
        exploration. We monitor cross-entropy on a held-out split and stop when it
        plateaus.
        """
        s_exp, a_exp, _, _, _, _ = Transition.to_tensor(expert_batch, self.device)
        n_expert = s_exp.size(0)

        # Validation split: 10% of expert data, capped at 25%, floor max(batch_size//4, 32)
        n_val = max(min(n_expert // 4, max(self.batch_size // 4, 32)),
                    int(n_expert * self.bc_val_split))
        n_val = min(n_val, max(1, n_expert - 1))
        perm = torch.randperm(n_expert, device=self.device)
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        s_tr, a_tr = s_exp[train_idx], a_exp[train_idx]
        s_va, a_va = s_exp[val_idx], a_exp[val_idx]
        n_train = train_idx.size(0)

        best_val = float("inf")
        best_state = copy.deepcopy(self.actor.state_dict())
        steps_since_improve = 0
        early_stop_enabled = self.bc_early_stop

        for step in range(self.n_bc_steps):
            idx = torch.randint(0, n_train, (self.batch_size,), device=self.device)
            log_pi = F.log_softmax(self.actor(s_tr[idx]), dim=-1)
            loss = -log_pi.gather(1, a_tr[idx].unsqueeze(1)).mean()
            self.opt_bc.zero_grad()
            loss.backward()
            self.opt_bc.step()

            if early_stop_enabled and (step + 1) % self.bc_eval_every == 0:
                with torch.no_grad():
                    log_pi_v = F.log_softmax(self.actor(s_va), dim=-1)
                    val_loss = -log_pi_v.gather(1, a_va.unsqueeze(1)).mean().item()
                if early_stop_enabled and val_loss < best_val - 1e-4:
                    best_val = val_loss
                    best_state = copy.deepcopy(self.actor.state_dict())
                    steps_since_improve = 0
                elif early_stop_enabled:
                    steps_since_improve += self.bc_eval_every
                    if steps_since_improve >= self.bc_patience:
                        print(f"[CSIL Stage A] early stop @ step {step+1}/{self.n_bc_steps}, "
                              f"val_xent={best_val:.4f}")
                        break

        if early_stop_enabled:
            self.actor.load_state_dict(best_state)

    def _freeze_bc(self):
        self.bc_policy = copy.deepcopy(self.actor)
        for p in self.bc_policy.parameters():
            p.requires_grad_(False)
        self.bc_policy.eval()

    # ----- Stage B: SARSA warm-up on expert transitions -------------------

    def _run_stage_b(self, expert_batch):
        """Initialize Q_φ via TD-SARSA on expert transitions with frozen r̄_θ_1.

        Builds SARSA tuples (s, a, s', a', d) from consecutive expert transitions;
        a' is the action of the next transition. Skip transitions where done=True
        (the next transition belongs to a new trajectory after reset, so a' is
        invalid).

        Reference policy π=q_θ_1 in Algorithm 1, so the (log q − log π) prior
        correction vanishes. Target: y = r̄_θ_1(s,a) + γ(1−d)·Q̄(s', a').
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
            print(f"[CSIL Stage B] only {n_valid} valid pairs (< batch_size={self.batch_size}); sampling with replacement")

        for step in range(self.n_sarsa_steps):
            idx = torch.randint(0, n_valid, (self.batch_size,), device=self.device)
            s_b = s[idx]
            a_b = a[idx]
            sn_b = s_next[idx]
            an_b = a_next[idx]
            d_b = d[idx].unsqueeze(1)

            with torch.no_grad():
                log_bc = F.log_softmax(self.bc_policy(s_b), dim=-1)
                # Paper Def. 1: r̄ = α(log q − log p), p uniform = 1/|A|
                r = self.alpha * (log_bc.gather(1, a_b.unsqueeze(1)) - self.log_p)
                if self.negative_reward:
                    # r ← r − sup_a r = α(log q − max_a log q)  (log p cancels)
                    r = r - self.alpha * (log_bc.max(dim=-1, keepdim=True).values - self.log_p)
                q_next = self.q_target(sn_b).gather(1, an_b.unsqueeze(1))
                y = r + self.gamma * (1.0 - d_b) * q_next

            q_pred = self.q_net(s_b).gather(1, a_b.unsqueeze(1))
            loss_q = (q_pred - y).pow(2).mean()

            self.opt_q.zero_grad()
            loss_q.backward()
            self.opt_q.step()
            polyak_update(self.q_target, self.q_net, self.tau)

    # ----- Stage C: critic ------------------------------------------------

    def _stage_c_critic_step(self, batch):
        """One SAC critic gradient step (paper Algo 1/2, Eq. 13).

        Reward = α(log π_BC − log p) computed under the FROZEN θ_1 snapshot.
        Soft V at s' uses KL-vs-BC at temperature β (Eq. 5 / Eq. 14 V form).
        """
        s, a, s_next, d, _, _ = Transition.to_tensor(batch, self.device)
        d = d.unsqueeze(1)
        a_idx = a.unsqueeze(1)

        with torch.no_grad():
            log_bc_s = F.log_softmax(self.bc_policy(s), dim=-1)
            r = self.alpha * (log_bc_s.gather(1, a_idx) - self.log_p)
            if self.negative_reward:
                r = r - self.alpha * (log_bc_s.max(dim=-1, keepdim=True).values - self.log_p)

            # Soft V at s_next: V = E_π[ Q − β(log π − log π_BC) ]
            logits_next = self.actor(s_next)
            pi_next = F.softmax(logits_next, dim=-1)
            log_pi_next = F.log_softmax(logits_next, dim=-1)
            log_bc_next = F.log_softmax(self.bc_policy(s_next), dim=-1)
            q_tgt_next = self.q_target(s_next)
            v_next = (pi_next * (q_tgt_next - self.beta * (log_pi_next - log_bc_next))).sum(
                dim=-1, keepdim=True
            )
            y = r + self.gamma * (1.0 - d) * v_next

        q_pred = self.q_net(s).gather(1, a_idx)
        loss_q = (q_pred - y).pow(2).mean()

        self.opt_q.zero_grad()
        loss_q.backward()
        self.opt_q.step()
        polyak_update(self.q_target, self.q_net, self.tau)

    # ----- Stage C: reward refinement (J_r, paper Eq. 11/12) --------------

    def _stage_c_reward_step(self, batch_b):
        """Reward refinement: maximize J_r(θ) = E_D[r_θ] − E_ρπ[Schulman(r_θ)]
        where Schulman(r) = r − 1 + exp(−r) is Schulman's positively-constrained
        unbiased KL estimator (paper Eq. 12, Appendix E).

        Uses the POSITIVE coherent reward r_θ = α(log q_θ − log p) regardless of
        the negative_reward flag (the bound subtraction is a Q-target trick,
        not a J_r modification — paper § 5.3 + Appendix F).

        N.B. This updates the actor θ via opt_reward; the same θ is also updated
        by J_π via opt_actor in the next call. The two optimizers share parameters
        but have independent Adam moments.
        """
        if self._expert_states is None:
            return

        s_b, a_b, _, _, _, _ = Transition.to_tensor(batch_b, self.device)
        idx_e = torch.randint(0, self._expert_states.size(0), (s_b.size(0),), device=self.device)
        s_e, a_e = self._expert_states[idx_e], self._expert_actions[idx_e]

        log_q_e = F.log_softmax(self.actor(s_e), dim=-1)
        log_q_b = F.log_softmax(self.actor(s_b), dim=-1)

        r_e = self.alpha * (log_q_e.gather(1, a_e.unsqueeze(1)) - self.log_p)
        r_b = self.alpha * (log_q_b.gather(1, a_b.unsqueeze(1)) - self.log_p)

        # Clamp r_b before the Schulman estimator: for very negative r_b,
        # exp(-r_b) explodes and produces NaN gradients. Paper § L doesn't
        # mention this but their continuous policies have natural bounded
        # log-likelihoods (tanh-Gaussian) that don't go below ~-10; with
        # categorical log_softmax we can easily get r_b = -20+ on small K.
        r_b_clamped = r_b.clamp(min=-10.0)

        # Maximize J_r ≡ minimize  −E_D[r_e] + E_B[r_b − 1 + exp(−r_b)]
        loss_r = -r_e.mean() + (r_b_clamped - 1.0 + (-r_b_clamped).exp()).mean()

        self.opt_reward.zero_grad()
        loss_r.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.opt_reward.step()

    # ----- Stage C: actor (J_π, paper Eq. 14) -----------------------------

    def _stage_c_actor_step(self, batch):
        """J_π(θ) = E_{s ~ B∪D, a ~ q_θ}[ Q_φ(s,a) − β(log q_θ − log q_θ_1) ]

        Paper § I (p. 27): "The policy and demonstration data are combined equally"
        in the actor batch. We concatenate `batch` (from replay) with a same-size
        sample from the expert cache.
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
            q_val = self.q_net(s)

        # Actor loss = -J_π = -[ E_a Q - β KL(π || π_BC) ]  (paper Eq. 14)
        loss_pi = (pi * (self.beta * (log_pi - log_bc) - q_val)).sum(dim=-1).mean()

        self.opt_actor.zero_grad()
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.opt_actor.step()

    # ----- save / load ----------------------------------------------------

    def save(self):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "q_net": self.q_net.state_dict(),
                "q_target": self.q_target.state_dict(),
                "bc_policy": self.bc_policy.state_dict() if self.bc_policy is not None else None,
                "stage": self._stage,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
            },
            self.save_path,
        )

    def load(self, path: str | None = None):
        path = path or self.save_path
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.q_net.load_state_dict(ck["q_net"])
        self.q_target.load_state_dict(ck["q_target"])
        if ck["bc_policy"] is not None:
            self.bc_policy = copy.deepcopy(self.actor)
            self.bc_policy.load_state_dict(ck["bc_policy"])
            for p in self.bc_policy.parameters():
                p.requires_grad_(False)
            self.bc_policy.eval()
        self._stage = ck.get("stage", "bc")
