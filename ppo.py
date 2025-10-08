# ppo.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Iterable
from envs.zero_sum import ZeroSumGame

# ---------- Fixed opponent ----------
class FixedStatePolicy:
    """Frozen opponent with probs[S,A]."""
    def __init__(self, probs: np.ndarray):
        assert probs.ndim == 2, "probs must be [num_states, num_actions]"
        self.probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)

    def get_probs(self, s_idx: int):
        return self.probs[s_idx]

    def sample_action(self, s_idx: int) -> int:
        p = self.probs[s_idx]
        return int(np.random.choice(len(p), p=p))

# ---------- Learner policy with action masks ----------
class MaskedStatePolicy(nn.Module):
    """
    π(a|s) with logits [S,A] and a binary mask forbidding some actions.
    mask[s,a]=1 → action forbidden (logit -> -inf).
    """
    def __init__(self, num_states=3, num_actions=3, init_uniform=True, mask=None):
        super().__init__()
        self.S, self.A = num_states, num_actions
        self.logits = nn.Parameter(torch.zeros(self.S, self.A) if init_uniform
                                   else 0.1 * torch.randn(self.S, self.A))
        if mask is None:
            mask = np.zeros((self.S, self.A), dtype=np.float32)
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))  # 0 or 1

    def masked_logits(self):
        return self.logits + (-1e9) * self.mask  # -inf where forbidden

    def dist(self, s_idx: Union[torch.Tensor, int]):
        ml = self.masked_logits()
        if isinstance(s_idx, torch.Tensor):
            logits_s = ml[s_idx]              # [B, A] or [A]
        else:
            logits_s = ml[int(s_idx)]         # [A]
        return torch.distributions.Categorical(logits=logits_s)

    def sample_action(self, s_idx: int) -> Tuple[int, torch.Tensor]:
        d = self.dist(s_idx)
        a = d.sample()
        return int(a.item()), d.log_prob(a)

    def log_prob(self, a: torch.Tensor, s_idx: torch.Tensor):
        return self.dist(s_idx).log_prob(a)

    def entropy(self):
        d = torch.distributions.Categorical(logits=self.masked_logits())
        return d.entropy().mean()  # mean across states

# ---------- Tiny PPO (generic: choose which agent learns) ----------
class PPOBR:
    """
    PPO best response for ZeroSumGame with terminal-only rewards.
    learner_key ∈ {"agent_0","agent_1"}; opponent is FixedStatePolicy.
    """
    def __init__(self, policy: MaskedStatePolicy, learner_key: str,
                 lr=3e-3, value_coef=0.5, entropy_coef=0.0, clip_eps=0.2, gamma=1.0):
        assert learner_key in {"agent_0", "agent_1"}
        self.learner_key = learner_key
        self.opponent_key = "agent_1" if learner_key == "agent_0" else "agent_0"
        self.policy = policy
        self.value = nn.Parameter(torch.zeros(policy.S))  # V(s) table
        self.opt = torch.optim.Adam(list(self.policy.parameters()) + [self.value], lr=lr)
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma

    @staticmethod
    def _state_idx_from_obs(obs: dict) -> int:
        # both agents observe same state label; use agent_0's entry
        S_map = {"s0": 0, "s1": 1, "s2": 2}
        return S_map.get(obs["agent_0"], 0)

    def rollout(self, env_ctor=ZeroSumGame, opponent: FixedStatePolicy = None, batch_episodes=512):
        env = env_ctor()
        traj = {k: [] for k in ["s", "a", "logp", "r", "v"]}

        for _ in range(batch_episodes):
            obs = env.reset()
            done = False
            while not done:
                s_idx = self._state_idx_from_obs(obs)

                # opponent acts from fixed distribution
                opp_a = opponent.sample_action(s_idx)
                # learner samples from masked policy
                a, logp = self.policy.sample_action(s_idx)

                if self.learner_key == "agent_0":
                    next_obs, rewards, dones, _ = env.step({"agent_0": a, "agent_1": opp_a})
                else:
                    next_obs, rewards, dones, _ = env.step({"agent_0": opp_a, "agent_1": a})

                traj["s"].append(s_idx)
                traj["a"].append(a)
                traj["logp"].append(logp.detach())
                traj["v"].append(self.value.data[s_idx].detach())
                traj["r"].append(rewards[self.learner_key])  # learner's payoff

                obs = next_obs
                done = bool(dones["__all__"])

        # vectorize
        traj["s"] = torch.tensor(traj["s"], dtype=torch.long)
        traj["a"] = torch.tensor(traj["a"], dtype=torch.long)
        traj["logp"] = torch.stack(traj["logp"]).to(torch.float32)
        traj["v"] = torch.tensor(traj["v"], dtype=torch.float32)
        traj["r"] = torch.tensor(traj["r"], dtype=torch.float32)
        return traj

    def update(self, traj, epochs=4, minibatch_size=1024):
        s, a, old_logp, returns = traj["s"], traj["a"], traj["logp"], traj["r"]
        adv = returns - self.value.data[s]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        N = s.size(0)
        idx = torch.randperm(N)
        last = {}
        for _ in range(epochs):
            for start in range(0, N, minibatch_size):
                mb = idx[start:start+minibatch_size]
                s_mb, a_mb, old_logp_mb = s[mb], a[mb], old_logp[mb]
                ret_mb, adv_mb = returns[mb], adv[mb]

                new_logp = self.policy.log_prob(a_mb, s_mb)
                ratio = torch.exp(new_logp - old_logp_mb)
                obj1 = ratio * adv_mb
                obj2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_mb
                policy_loss = -torch.mean(torch.min(obj1, obj2))

                v_pred = self.value[s_mb]
                value_loss = F.mse_loss(v_pred, ret_mb)

                ent = self.policy.entropy()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * ent

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                last = {"policy_loss": float(policy_loss.detach()),
                        "value_loss": float(value_loss.detach()),
                        "entropy": float(ent.detach())}
        return last

    def evaluate(self, env_ctor=ZeroSumGame, opponent: FixedStatePolicy = None, episodes=5000):
        env = env_ctor()
        rets = []
        count_s0 = 0
        hist_s0 = np.zeros(self.policy.A, dtype=int)

        for _ in range(episodes):
            obs = env.reset()
            done = False
            G = 0.0
            while not done:
                s_idx = self._state_idx_from_obs(obs)
                opp_a = opponent.sample_action(s_idx)
                a, _ = self.policy.sample_action(s_idx)
                if s_idx == 0:
                    count_s0 += 1
                    hist_s0[a] += 1
                if self.learner_key == "agent_0":
                    obs, r, d, _ = env.step({"agent_0": a, "agent_1": opp_a})
                else:
                    obs, r, d, _ = env.step({"agent_0": opp_a, "agent_1": a})
                G = r[self.learner_key]
                done = bool(d["__all__"])
            rets.append(G)

        return {
            "avg_return": float(np.mean(rets)),
            "s0_action_freq": (hist_s0 / max(1, count_s0)).tolist(),
        }

# ---------- convenience wrapper ----------
def train_best_response(opponent_probs: np.ndarray,
                        learner_key="agent_0",
                        forbid_on_states: Iterable[int] = (0,),
                        forbid_action_idx=2,
                        seed=0,
                        iters=20,
                        batch_episodes=1024,
                        lr=3e-3,
                        clip_eps=0.2,
                        value_coef=0.5,
                        entropy_coef=0.0):
    """
    Train PPO best-response for `learner_key` vs a frozen opponent.
    By default forbids action index 2 (a3) at s0.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    S, A = opponent_probs.shape
    mask = np.zeros((S, A), dtype=np.float32)
    for s in forbid_on_states:
        mask[s, forbid_action_idx] = 1.0

    learner = MaskedStatePolicy(num_states=S, num_actions=A, init_uniform=True, mask=mask)
    ppo = PPOBR(learner, learner_key=learner_key, lr=lr, value_coef=value_coef,
                entropy_coef=entropy_coef, clip_eps=clip_eps)
    opponent = FixedStatePolicy(opponent_probs)

    pre = ppo.evaluate(ZeroSumGame, opponent, episodes=2000)
    for i in range(1, iters + 1):
        traj = ppo.rollout(ZeroSumGame, opponent, batch_episodes=batch_episodes)
        stats = ppo.update(traj, epochs=4, minibatch_size=min(batch_episodes * 2, 2048))
        if i % max(1, iters // 5) == 0:
            ev = ppo.evaluate(ZeroSumGame, opponent, episodes=5000)
            print(f"[Iter {i:02d} {learner_key}] avg_ret={ev['avg_return']:.3f} | "
                  f"pol_loss={stats['policy_loss']:.3f} val_loss={stats['value_loss']:.3f} | "
                  f"s0 freq ~ {np.round(ev['s0_action_freq'], 3)}")

    post = ppo.evaluate(ZeroSumGame, opponent, episodes=10000)
    with torch.no_grad():
        probs = F.softmax(learner.masked_logits(), dim=-1).cpu().numpy()
    return {"learner_policy_probs": probs, "pre_eval": pre, "post_eval": post, "mask": mask}
