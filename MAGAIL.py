# MAGAIL.py
import os, random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy.spatial.distance import jensenshannon

# --- envs ---
from envs.entropy_coordination import (
    CoordinationGame,
    generate_expert_data,
    generate_asymmetric_bimodal_expert_data,
    generate_noisy_bimodal_expert_data,
)
from envs.zero_sum import ZeroSumGame, generate_expert_s0_s2_s3


# =========================
# Reproducible seeding
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# =========================
# Policy (state-conditioned)
# =========================
class StateTabularPolicy(nn.Module):
    """
    Categorical policy π(a|s) with logits table [num_states, num_actions].
    For 1-step envs, set num_states=1 (behaves like a stateless tabular policy).
    """
    def __init__(self, num_states=1, num_actions=2, init_uniform=True):
        super().__init__()
        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        self.logits = nn.Parameter(
            torch.zeros(self.num_states, self.num_actions)
            if init_uniform else 0.1 * torch.randn(self.num_states, self.num_actions)
        )

    def get_probs(self, s_idx=None):
        """
        If s_idx is None -> Tensor [num_states, num_actions]
        Else -> Tensor [num_actions] for that state index
        """
        if s_idx is None:
            return F.softmax(self.logits, dim=-1)
        return F.softmax(self.logits[s_idx], dim=-1)

    def sample_action(self, s_idx=None):
        probs = self.get_probs(0 if s_idx is None else s_idx)
        return torch.multinomial(probs, 1).item()

    def log_prob(self, action, s_idx=None):
        if isinstance(action, torch.Tensor):
            action = int(action.item())
        log_probs = F.log_softmax(self.logits[0 if s_idx is None else s_idx], dim=-1)
        return log_probs[action]

    def entropy(self):
        """Mean entropy across states (scalar)."""
        probs = F.softmax(self.logits, dim=-1)
        logp = F.log_softmax(self.logits, dim=-1)
        ent = -(probs * logp).sum(dim=-1)  # [num_states]
        return ent.mean()


# =========================
# Discriminator (state-conditioned)
# =========================
class TabularStateJointDiscriminator(nn.Module):
    """
    D(s, a0, a1) with logits table [num_states, nA, nA].
    - 1-step env: num_states=1, nA=2 → classic 4-logit joint
    - 2-step env: num_states=3 (s0,s1,s2), nA=3
    """
    def __init__(self, num_states=1, n_actions=2, init_uniform=True):
        super().__init__()
        self.num_states = int(num_states)
        self.n_actions = int(n_actions)
        self.logits = nn.Parameter(
            torch.zeros(self.num_states, self.n_actions, self.n_actions)
            if init_uniform else 0.1 * torch.randn(self.num_states, self.n_actions, self.n_actions)
        )

    @staticmethod
    def _to_tensor(x, device, dtype=torch.long):
        return x.to(device) if torch.is_tensor(x) else torch.as_tensor(x, dtype=dtype, device=device)

    def logit_sa(self, s_idx, a0, a1):
        device = self.logits.device
        s = self._to_tensor(s_idx, device)
        a0 = self._to_tensor(a0, device)
        a1 = self._to_tensor(a1, device)
        return self.logits[s, a0, a1]

    def logit(self, a0, a1, s_idx=None):
        if s_idx is None:
            s_idx = 0 if not torch.is_tensor(a0) else torch.zeros_like(a0, dtype=torch.long, device=self.logits.device)
        return self.logit_sa(s_idx, a0, a1)

    def forward(self, a0, a1, s_idx=None):
        return torch.sigmoid(self.logit(a0, a1, s_idx))


# =========================
# Rollout (generic)
# =========================
def collect_policy_trajectories(policies, num_episodes=100, env_ctor=CoordinationGame):
    """
    Collect per-step samples for either env.
    Returns dict of T samples:
      - s: Long[T]          pre-step state index (0 for 1-step; 0/1/2 for zero-sum)
      - a0,a1: Long[T]
      - joint_idx: Long[T]  a0 * n_actions + a1
      - logp0,logp1: Tensor[T]
      - rewards: Float[T]   env reward (terminal at last step in 2-step env)
    """
    env = env_ctor()
    n_actions = getattr(env, "num_actions", 2)
    state_to_idx = {"s0": 0, "s1": 1, "s2": 2}

    s_list, a0_list, a1_list, lp0_list, lp1_list, rew_list = [], [], [], [], [], []

    for _ in range(num_episodes):
        obs = env.reset()
        if "agent_0" in obs:
            k0, k1 = "agent_0", "agent_1"
        elif "agent_A" in obs:
            k0, k1 = "agent_A", "agent_B"
        else:
            raise KeyError(f"Unknown agent keys in obs: {list(obs.keys())}")

        done = False
        while not done:
            s_val = obs[k0]
            s_idx = state_to_idx.get(s_val, 0) if isinstance(s_val, str) else 0

            act0 = policies["agent_0"].sample_action(s_idx)
            act1 = policies["agent_1"].sample_action(s_idx)
            lp0 = policies["agent_0"].log_prob(act0, s_idx)
            lp1 = policies["agent_1"].log_prob(act1, s_idx)

            next_obs, r, d, _ = env.step({k0: act0, k1: act1})

            s_list.append(s_idx)
            a0_list.append(act0); a1_list.append(act1)
            lp0_list.append(lp0);  lp1_list.append(lp1)
            rew_list.append(float(r[k0]))

            obs = next_obs
            done = bool(d["__all__"])

    s = torch.tensor(s_list, dtype=torch.long)
    a0 = torch.tensor(a0_list, dtype=torch.long)
    a1 = torch.tensor(a1_list, dtype=torch.long)
    logp0 = torch.stack(lp0_list)
    logp1 = torch.stack(lp1_list)
    rewards = torch.tensor(rew_list, dtype=torch.float32)
    joint_idx = a0 * n_actions + a1

    return {"s": s, "a0": a0, "a1": a1, "joint_idx": joint_idx,
            "logp0": logp0, "logp1": logp1, "rewards": rewards}


# =========================
# MAGAIL Trainer
# =========================
class MAGAILTrainer:
    """
    MAGAIL that adapts to the chosen environment via env_ctor.
    - 1-step env: n_actions=2, num_states_for_D=1
    - 2-step env: n_actions=3, num_states_for_D=3
    """
    def __init__(self, env_ctor,
                 beta=0.0, lr_policy=0.01, lr_disc=0.01,
                 policy_init_uniform=True, reward_style="non_saturating",
                 num_states_for_D=None, n_actions=None,
                 initial_policies=None,            # NEW (optional): dict {"agent_0": policy, "agent_1": policy}
                 frozen_agents=None):

        self.env_ctor = env_ctor
        probe_env = env_ctor()
        self.n_actions = int(n_actions if n_actions is not None else getattr(probe_env, "num_actions", 2))
        self.num_states_for_D = int(num_states_for_D if num_states_for_D is not None
                                    else (1 if self.n_actions == 2 else 3))

        self.beta = float(beta)
        self.lr_policy = lr_policy
        self.lr_disc = lr_disc
        self.reward_style = reward_style
        self.frozen_agents = set(frozen_agents or [])

        # Policies (state-conditioned)
        self.policies = {
            "agent_0": StateTabularPolicy(self.num_states_for_D, self.n_actions, policy_init_uniform),
            "agent_1": StateTabularPolicy(self.num_states_for_D, self.n_actions, policy_init_uniform),
        }

        # If provided, copy initial policy weights
        if initial_policies:
            for k, pol in initial_policies.items():
                if k in self.policies:
                    with torch.no_grad():
                        self.policies[k].logits.copy_(pol.logits)

        # Discriminator D(s,a0,a1)
        self.discriminator = TabularStateJointDiscriminator(self.num_states_for_D, self.n_actions, init_uniform=True)

        # Optims: build only for unfrozen agents
        self.policy_optimizers = {}
        for ag in ["agent_0", "agent_1"]:
            if ag in self.frozen_agents:
                # Ensure frozen params truly don't get grads
                for p in self.policies[ag].parameters():
                    p.requires_grad_(False)
                self.policy_optimizers[ag] = None
            else:
                self.policy_optimizers[ag] = torch.optim.Adam(self.policies[ag].parameters(), lr=lr_policy)

        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr_disc)

        self.history = {
            "policy_probs": {"agent_0": [], "agent_1": []},  # stores [num_states, nA]
            "disc_loss": [],
            "policy_loss": {"agent_0": [], "agent_1": []},
            "entropy": {"agent_0": [], "agent_1": []},
            "joint_action_dist": [],  # average over states of product-of-marginals
        }

    # --- Discriminator update ---
    def update_discriminator(self, expert_data, policy_data, batch_size=64):
        def _sample_triplets(data, n, nA):
            # returns s, a0, a1 (Long)
            if isinstance(data, dict):
                if all(k in data for k in ("s", "a0", "a1")):
                    N = len(data["a0"])
                    sel = torch.randperm(N)[:min(n, N)]
                    s  = torch.as_tensor(data["s"])[sel].long()
                    a0 = torch.as_tensor(data["a0"])[sel].long()
                    a1 = torch.as_tensor(data["a1"])[sel].long()
                    return s, a0, a1
                elif "joint_idx" in data:
                    idx = torch.as_tensor(data["joint_idx"], dtype=torch.long)
                    sel = torch.randperm(idx.shape[0])[:min(n, idx.shape[0])]
                    idx = idx[sel]
                    a0 = (idx // nA).long()
                    a1 = (idx %  nA).long()
                    s  = torch.zeros_like(a0, dtype=torch.long)  # unknown/1-step -> s=0
                    return s, a0, a1
                else:
                    raise ValueError("Unsupported expert/policy data dict format.")
            # list-of-dicts fallback (not used in your current setup)
            import numpy as _np
            batch = _np.random.choice(data, size=min(n, len(data)), replace=False)
            a0 = torch.tensor([t["joint_action"][0] for t in batch], dtype=torch.long)
            a1 = torch.tensor([t["joint_action"][1] for t in batch], dtype=torch.long)
            s  = torch.tensor([t.get("state_idx", 0) for t in batch], dtype=torch.long)
            return s, a0, a1

        exp_s, exp_a0, exp_a1 = _sample_triplets(expert_data, batch_size, self.n_actions)
        pol_s, pol_a0, pol_a1 = _sample_triplets(policy_data, batch_size, self.n_actions)

        crit = torch.nn.BCEWithLogitsLoss(reduction="mean")
        exp_logits = self.discriminator.logit(exp_a0, exp_a1, s_idx=exp_s)
        pol_logits = self.discriminator.logit(pol_a0, pol_a1, s_idx=pol_s)

        ones = torch.ones_like(exp_logits)
        zeros = torch.zeros_like(pol_logits)
        loss = crit(exp_logits, ones) + crit(pol_logits, zeros)

        self.disc_optimizer.zero_grad()
        loss.backward()
        self.disc_optimizer.step()

        return float(loss.detach())

    # --- Policy update ---
    def update_policies(self, policy_data, batch_size=64):
        if isinstance(policy_data, dict) and all(k in policy_data for k in ["s", "a0", "a1", "logp0", "logp1"]):
            N = policy_data["a0"].shape[0]
            sel = torch.randperm(N)[:min(batch_size, N)]
            s   = policy_data["s"][sel]
            a0  = policy_data["a0"][sel]
            a1  = policy_data["a1"][sel]
            lp0 = policy_data["logp0"][sel]
            lp1 = policy_data["logp1"][sel]
        else:
            import numpy as _np
            batch = _np.random.choice(policy_data, size=min(batch_size, len(policy_data)), replace=False)
            s   = torch.tensor([t.get("state_idx", 0) for t in batch], dtype=torch.long)
            a0  = torch.tensor([t["joint_action"][0] for t in batch], dtype=torch.long)
            a1  = torch.tensor([t["joint_action"][1] for t in batch], dtype=torch.long)
            lp0 = torch.stack([t["log_probs"]["agent_0"] for t in batch])
            lp1 = torch.stack([t["log_probs"]["agent_1"] for t in batch])

        with torch.no_grad():
            D = torch.sigmoid(self.discriminator.logit(a0, a1, s_idx=s))
            reward = torch.log(D + 1e-8) if self.reward_style == "non_saturating" else -torch.log(1 - D + 1e-8)

        if self.beta > 0.0:
            ent0 = self.policies["agent_0"].entropy()
            ent1 = self.policies["agent_1"].entropy()
        else:
            ent0 = ent1 = torch.tensor(0.0, dtype=torch.float32, device=reward.device)

        loss0 = -(lp0 * reward).mean() - self.beta * ent0
        loss1 = -(lp1 * reward).mean() - self.beta * ent1

        # --- optimize only unfrozen agents ---
        if self.policy_optimizers["agent_0"] is not None:
            self.policy_optimizers["agent_0"].zero_grad()
            loss0.backward(retain_graph=("agent_1" not in self.frozen_agents))
            self.policy_optimizers["agent_0"].step()

        if self.policy_optimizers["agent_1"] is not None:
            self.policy_optimizers["agent_1"].zero_grad()
            loss1.backward()
            self.policy_optimizers["agent_1"].step()

        return {"agent_0": float(loss0.detach()), "agent_1": float(loss1.detach())}
    

    # --- Training loop ---
    def train(self, expert_data, num_epochs=500, batch_size=64, collect_every=10, rollout_episodes=100):
        for epoch in range(num_epochs):
            policy_data = collect_policy_trajectories(self.policies, num_episodes=rollout_episodes, env_ctor=self.env_ctor)
            disc_loss = self.update_discriminator(expert_data, policy_data, batch_size)
            policy_losses = self.update_policies(policy_data, batch_size)
            if epoch % collect_every == 0:
                self.record_statistics(disc_loss, policy_losses)

    # --- Logging ---
    def record_statistics(self, disc_loss, policy_losses):
        for agent in ["agent_0", "agent_1"]:
            probs = self.policies[agent].get_probs().detach().cpu().numpy()  # [S, A] or [A]
            if probs.ndim == 1:
                probs = probs[None, :]
            self.history["policy_probs"][agent].append(probs.copy())

        self.history["disc_loss"].append(disc_loss)
        for agent in ["agent_0", "agent_1"]:
            self.history["policy_loss"][agent].append(policy_losses[agent])

        for agent in ["agent_0", "agent_1"]:
            self.history["entropy"][agent].append(float(self.policies[agent].entropy().item()))

        # Average over states of product-of-marginals (for quick glance; not used in eval)
        p0 = self.policies["agent_0"].get_probs().detach().cpu().numpy()
        p1 = self.policies["agent_1"].get_probs().detach().cpu().numpy()
        if p0.ndim == 1:
            p0 = p0[None, :]; p1 = p1[None, :]
        joints = [np.outer(p0[s], p1[s]).reshape(-1) for s in range(p0.shape[0])]
        self.history["joint_action_dist"].append(np.mean(np.stack(joints, axis=0), axis=0))


# =========================
# Runner + metrics (generic)
# =========================
def _expert_num_samples(expert_data):
    """Number of per-step samples in expert data."""
    if isinstance(expert_data, dict) and "joint_idx" in expert_data:
        return int(np.asarray(expert_data["joint_idx"]).shape[0])
    return len(expert_data)

def _expert_joint_from_data(expert_data, n_actions):
    if isinstance(expert_data, dict) and "joint_idx" in expert_data:
        idx = np.asarray(expert_data["joint_idx"], dtype=int)
    else:
        idx = np.array([t["joint_action"][0] * n_actions + t["joint_action"][1] for t in expert_data], dtype=int)
    counts = np.bincount(idx, minlength=n_actions * n_actions)
    return counts / counts.sum()

def _joint_hist_from_rollout(rollout, n_actions):
    idx = (rollout["a0"] * n_actions + rollout["a1"]).cpu().numpy()
    counts = np.bincount(idx, minlength=n_actions * n_actions)
    return counts / counts.sum()

def _matching_rate(rollout):
    """P(a0 == a1) — works for any discrete action count."""
    return float((rollout["a0"] == rollout["a1"]).float().mean().item())

def run_experiment(
    env_name="coordination",          # "coordination" | "zero_sum"
    seeds = [42, 123, 456, 789, 999],
    beta_values = None,               
    num_epochs = 4000,
    expert_type = "bimodal",          # for coordination: mixed|bimodal|asymmetric|noisy|all_AA
    policy_init_uniform = False,
    reward_style = "non_saturating",
    batch_size = 32,
    rollout_episodes = 200,
    eval_episodes = 5000,
    expert_seed = 0,
    lr_policy = 0.01,
    lr_disc = 0.01,
    collect_every = 10,
    force_zero_entropy=False,         # optional hard override (sets β=[0.0])
):
    # -------- choose env + expert --------
    if env_name == "coordination":
        env_ctor = CoordinationGame
        n_actions = 2
        # expert data (1-step)
        if expert_type == "mixed":
            expert_data = generate_expert_data(num_episodes=1000, seed=expert_seed)
        elif expert_type == "bimodal":
            expert_data = generate_asymmetric_bimodal_expert_data(num_episodes=1000, AA_ratio=0.5, seed=expert_seed)
        elif expert_type == "asymmetric":
            expert_data = generate_asymmetric_bimodal_expert_data(num_episodes=1000, AA_ratio=0.7, seed=expert_seed)
        elif expert_type == "noisy":
            expert_data = generate_noisy_bimodal_expert_data(num_episodes=1000, noise_level=0.1, seed=expert_seed)
        elif expert_type == "all_AA":
            expert_data = generate_asymmetric_bimodal_expert_data(num_episodes=1000, AA_ratio=1.0, seed=expert_seed)
        else:
            raise ValueError(f"Unknown expert_type for coordination: {expert_type}")
        # default β sweep for coordination
        default_betas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]

    elif env_name == "zero_sum":
        env_ctor = ZeroSumGame
        n_actions = 3
        # expert: force s0->s2->s3
        expert_data = generate_expert_s0_s2_s3(num_episodes=1000, seed=expert_seed)
        # default: NO entropy for zero-sum
        default_betas = [0.0]

    else:
        raise ValueError(f"Unknown env_name: {env_name}")

    # -------- decide β values (entropy) --------
    if force_zero_entropy:
        beta_values = [0.0]
    elif beta_values is None:
        beta_values = default_betas  # coord gets sweep, zero_sum gets [0.0]

    n_exp = _expert_num_samples(expert_data)
    expert_joint = _expert_joint_from_data(expert_data, n_actions=n_actions)
    print(f"Generated {n_exp} expert per-step samples for env '{env_name}'")
    print(f"Expert joint probs ({n_actions}x{n_actions} bins) = {np.round(expert_joint, 3)}")
    print(f"β schedule: {beta_values}")

    results = {}
    for beta in beta_values:
        print(f"\nRunning experiments with β = {beta}")
        results[beta] = {}
        for seed in seeds:
            print(f"  Seed {seed}...", end="")
            set_seed(seed)

            trainer = MAGAILTrainer(
                env_ctor=env_ctor,
                beta=beta,                              
                policy_init_uniform=policy_init_uniform,
                reward_style=reward_style,
                lr_policy=lr_policy,
                lr_disc=lr_disc,
            )

            trainer.train(
                expert_data,
                num_epochs=num_epochs,
                batch_size=batch_size,
                collect_every=collect_every,
                rollout_episodes=rollout_episodes,
            )

            # Final policies (per-state)
            p0 = trainer.policies["agent_0"].get_probs().detach().cpu().numpy()
            p1 = trainer.policies["agent_1"].get_probs().detach().cpu().numpy()
            final_probs = {"agent_0": p0, "agent_1": p1}
            final_entropy = {
                "agent_0": float(trainer.policies["agent_0"].entropy().item()),
                "agent_1": float(trainer.policies["agent_1"].entropy().item()),
            }

            # Eval
            eval_roll = collect_policy_trajectories(trainer.policies, num_episodes=eval_episodes, env_ctor=env_ctor)
            learner_joint = _joint_hist_from_rollout(eval_roll, n_actions=n_actions)
            coord_rate = _matching_rate(eval_roll)
            js_dist = float(jensenshannon(expert_joint, learner_joint, base=2))  # distance

            results[beta][seed] = {
                "final_probs": final_probs,
                "final_entropy": final_entropy,
                "coordination_rate": coord_rate,
                "learner_joint": learner_joint,
                "expert_joint": expert_joint,
                "js_distance": js_dist,
                "history": trainer.history,
                "n_actions": n_actions,
                "env_name": env_name,
            }
            print(" Done!")
    return results, expert_data, collect_every



# =========================
# Utilities
# =========================
def print_policy_by_state(policy, name="policy", state_names=None):
    probs = policy.get_probs().detach().cpu().numpy()
    if probs.ndim == 1:
        probs = probs[None, :]
    if state_names is None:
        state_names = [f"s{i}" for i in range(probs.shape[0])]
    for i, s in enumerate(state_names):
        print(f"{name} @ {s}: {np.round(probs[i], 3)}")

def set_policy_row_from_probs(policy: "StateTabularPolicy", s_idx: int, probs, clamp_logit_min=-20.0):
    """
    Set π(a|s=s_idx) to 'probs' approximately by writing logits = log(probs).
    Any zeros are clamped to avoid -inf. Works best when you pass a tiny epsilon
    for zero entries, e.g., [0.5, 0.5, 1e-6].
    """
    p = torch.as_tensor(np.asarray(probs, dtype=np.float32))
    p = torch.clamp(p, min=1e-8)
    logits = torch.log(p)
    logits = torch.clamp(logits, min=clamp_logit_min)
    with torch.no_grad():
        policy.logits[s_idx].copy_(logits)

