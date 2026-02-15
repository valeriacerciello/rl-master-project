# envs/entropy_coordination.py

import numpy as np

# Environment
class CoordinationGame:
    """
    Single-state, one-step, two-agent coordination game:
    actions in {A=0, B=1}; reward +1 iff a0==a1, else 0.
    """
    def __init__(self):
        self.num_agents = 2
        self.num_actions = 2
        self.state = 0

    def reset(self):
        return {"agent_0": self.state, "agent_1": self.state}

    def step(self, actions):
        for k, a in actions.items():
            assert a in {0, 1}, f"Invalid action {a} for {k}. Must be 0 or 1."
        a0, a1 = actions["agent_0"], actions["agent_1"]
        reward = 1.0 if a0 == a1 else 0.0
        rewards = {"agent_0": reward, "agent_1": reward}
        dones = {"agent_0": True, "agent_1": True, "__all__": True}
        next_obs = {"agent_0": self.state, "agent_1": self.state}
        return next_obs, rewards, dones, {}

    def get_joint_action_rewards(self):
        return {
            (a0, a1): (1.0, 1.0) if a0 == a1 else (0.0, 0.0)
            for a0 in range(self.num_actions)
            for a1 in range(self.num_actions)
        }

    def print_payoff_table(self):
        print("Payoff Table (Agent 1 rows, Agent 2 columns)")
        print("            Agent 2: A     Agent 2: B")
        for a0 in range(self.num_actions):
            row = [f"Agent 1: {'A' if a0 == 0 else 'B'}"]
            for a1 in range(self.num_actions):
                r0, r1 = self.get_joint_action_rewards()[(a0, a1)]
                row.append(f"({int(r0)}, {int(r1)})")
            print("   ".join(row))


# Expert data generators (vectorized)
def joint_to_index(a0, a1):
    return a0 * 2 + a1

def summarize_joint_counts(joint_actions):
    """
    joint_actions: array (N,2) with actions in {0,1}.
    Returns counts and frequencies over [AA, AB, BA, BB].
    """
    idx = joint_actions[:, 0] * 2 + joint_actions[:, 1]
    counts = np.bincount(idx, minlength=4)
    freqs = counts / counts.sum()
    return counts, freqs

def generate_expert_data(num_episodes=1000, seed=None):
    """
    Mixed independent 50/50 per agent -> joint ≈ uniform (0.25 each).
    Returns dict with vectorized arrays.
    """
    assert num_episodes > 0
    rng = np.random.RandomState(seed)
    a0 = rng.randint(0, 2, size=num_episodes)
    a1 = rng.randint(0, 2, size=num_episodes)
    actions = np.stack([a0, a1], axis=1)
    counts, freqs = summarize_joint_counts(actions)
    print(f"Mixed 50/50 expert | counts [AA,AB,BA,BB]={counts.tolist()} | freqs={np.round(freqs,3)}")
    return {
        "state": np.zeros(num_episodes, dtype=int),
        "actions": actions,
        "joint_idx": actions[:, 0] * 2 + actions[:, 1]
    }

def generate_asymmetric_bimodal_expert_data(num_episodes=1000, AA_ratio=0.5, seed=None):
    """
    Correlated expert: only AA and BB with proportions AA_ratio and 1-AA_ratio.
    """
    assert 0.0 <= AA_ratio <= 1.0
    assert num_episodes > 0
    rng = np.random.RandomState(seed)
    num_AA = int(round(num_episodes * AA_ratio))
    num_BB = num_episodes - num_AA
    actions = np.empty((num_episodes, 2), dtype=int)
    actions[:num_AA] = (0, 0)  # AA
    actions[num_AA:] = (1, 1)  # BB
    rng.shuffle(actions)
    counts, freqs = summarize_joint_counts(actions)
    print(f"Asymmetric bimodal expert | AA_ratio={AA_ratio:.2f} | counts={counts.tolist()} | freqs={np.round(freqs,3)}")
    return {
        "state": np.zeros(num_episodes, dtype=int),
        "actions": actions,
        "joint_idx": actions[:, 0] * 2 + actions[:, 1]
    }

def generate_noisy_bimodal_expert_data(num_episodes=1000, noise_level=0.1, seed=None):
    """
    Mostly AA/BB, with noise_level fraction of AB/BA.
    """
    assert 0.0 <= noise_level < 1.0
    assert num_episodes > 0
    rng = np.random.RandomState(seed)
    num_noise = int(round(num_episodes * noise_level))
    num_coord = num_episodes - num_noise
    num_AA = num_coord // 2
    num_BB = num_coord - num_AA
    actions = np.empty((num_episodes, 2), dtype=int)
    actions[:num_AA] = (0, 0)
    actions[num_AA:num_AA + num_BB] = (1, 1)
    noise_pairs = rng.randint(0, 2, size=(num_noise,))
    actions[num_AA + num_BB:] = np.stack([noise_pairs, 1 - noise_pairs], axis=1)
    rng.shuffle(actions)
    counts, freqs = summarize_joint_counts(actions)
    print(f"Noisy bimodal expert | noise={noise_level:.2f} | counts={counts.tolist()} | freqs={np.round(freqs,3)}")
    return {
        "state": np.zeros(num_episodes, dtype=int),
        "actions": actions,
        "joint_idx": actions[:, 0] * 2 + actions[:, 1]
    }