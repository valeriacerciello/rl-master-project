# envs/zero_sum.py

import numpy as np

# =========================
# Environment
# =========================
class ZeroSumGame:
    """
    Two-agent, three-action, two-step environment that follows the
    transition structure in the provided figure.

    Actions: each agent chooses in {0,1,2} == {a1, a2, a3}

    States:
      s0 (start) -> if (a3,b3) go to s2, else go to s1
      s1 -> if (a1,b1) or (a2,b2) go to Sxplt1
            if (a2,b1) or (a1,b2) go to Sxplt2
            else go to Scopy
      s2 -> always to s3 (regardless of actions)
      s3, Sxplt1, Sxplt2, Scopy are absorbing for further steps ("all" loops)

    Episode length: 2 steps. Rewards are issued ONLY at the end of step 2.

    Terminal payoffs (zero-sum):
        Sxplt1: (+1, -1) agent A exploits agent B
        Sxplt2: (-1, +1) agent B exploits agent A
        Scopy : ( 0,  0)
        s3    : ( 0,  0)
    """

    A1, A2, A3 = 0, 1, 2  # alias

    def __init__(self):
        self.num_agents = 2
        self.num_actions = 3
        self.states = ["s0", "s1", "s2", "s3", "Sxplt1", "Sxplt2", "Scopy"]
        self.start_state = "s0"
        self.absorbing = {"s3", "Sxplt1", "Sxplt2", "Scopy"}
        self.reset()

    def reset(self):
        self.t = 0
        self.state = self.start_state
        return {"agent_0": self.state, "agent_1": self.state}

    def step(self, actions):
        a0, a1 = actions["agent_0"], actions["agent_1"]
        assert a0 in {0,1,2} and a1 in {0,1,2}
        self.state = self._next_state(self.state, a0, a1)
        self.t += 1
        done = self.t >= 2
        r0, r1 = (self._terminal_reward(self.state) if done else (0.0, 0.0))
        obs = {"agent_0": self.state, "agent_1": self.state}
        rewards = {"agent_0": float(r0), "agent_1": float(r1)}
        dones = {"agent_0": done, "agent_1": done, "__all__": done}
        return obs, rewards, dones, {}

    # ---------- Dynamics ----------
    def _next_state(self, s, aA, aB):
        a1, a2, a3 = self.A1, self.A2, self.A3

        if s == "s0":
            # From s0: a3,b3 -> s2 ; else -> s1
            return "s2" if (aA == a3 and aB == a3) else "s1"

        if s == "s1":
            # To Sxplt1: (a1,b1) or (a2,b2)
            if (aA == a1 and aB == a1) or (aA == a2 and aB == a2):
                return "Sxplt1"
            # To Sxplt2: (a2,b1) or (a1,b2)
            if (aA == a2 and aB == a1) or (aA == a1 and aB == a2):
                return "Sxplt2"
            # Else to Scopy
            return "Scopy"

        if s == "s2":
            # Always to s3 regardless of actions
            return "s3"

        # Absorbing states loop
        return s

    def _terminal_reward(self, s):
        if s == "Sxplt1":
            return +1.0, -1.0
        if s == "Sxplt2":
            return -1.0, +1.0
        # Scopy or s3 (or anything else): zero payoff
        return 0.0, 0.0

    # ---------- Utilities ----------
    def current_state(self):
        return self.state

    def is_done(self):
        return self.t >= 2

    def describe(self):
        return {
            "horizon": 2,
            "actions": {0: "a1", 1: "a2", 2: "a3"},
            "states": self.states,
            "absorbing": list(self.absorbing),
            "start": self.start_state,
            "terminal_payoffs": {
                "Sxplt1": (+1, -1),
                "Sxplt2": (-1, +1),
                "Scopy": (0, 0),
                "s3": (0, 0),
            },
        }


# =========================
# Expert data generator: s0 -> s2 -> s3 path
# Returns per-step samples with keys: s, a0, a1, joint_idx
# =========================

_STATE_TO_IDX_ZS = {"s0": 0, "s1": 1, "s2": 2}

def _joint_to_index_3(a0, a1):
    return a0 * 3 + a1

def _summarize_joint_counts_3(actions):
    idx = actions[:, 0] * 3 + actions[:, 1]
    counts = np.bincount(idx, minlength=9)
    freqs = counts / max(1, counts.sum())
    return counts, freqs

def generate_expert_s0_s2_s3(num_episodes=1000, seed=None):
    """
    Expert that always takes (2,2) at s0 -> s2, then plays uniformly at s2.
    Collects BOTH steps as separate samples.
    Returns dict with keys: s, a0, a1, joint_idx.
    """
    rng = np.random.RandomState(seed)
    env = ZeroSumGame()

    s_list, a0_list, a1_list = [], [], []

    for _ in range(num_episodes):
        _ = env.reset()  # at s0

        # ---- step 0 at s0: force (2,2) so we go to s2
        a0, a1 = 2, 2
        next_obs, _, _, _ = env.step({"agent_0": a0, "agent_1": a1})
        s_list.append(_STATE_TO_IDX_ZS["s0"]) 
        a0_list.append(a0)
        a1_list.append(a1)
        assert next_obs["agent_0"] == "s2", f"Expected s2 after (2,2), got {next_obs['agent_0']}"

        # ---- step 1 at s2: actions irrelevant for transition (s2 -> s3), sample uniformly
        b0, b1 = rng.randint(0, 3), rng.randint(0, 3)
        obs_after, _, dones, _ = env.step({"agent_0": b0, "agent_1": b1})
        s_list.append(_STATE_TO_IDX_ZS["s2"])
        a0_list.append(b0)
        a1_list.append(b1)
        # use obs_after to sanity-check terminal
        assert obs_after["agent_0"] == "s3", f"Expected s3, got {obs_after['agent_0']}"
        assert dones["__all__"] is True

    a0 = np.array(a0_list, dtype=int)
    a1 = np.array(a1_list, dtype=int)
    actions = np.stack([a0, a1], axis=1)
    counts, freqs = _summarize_joint_counts_3(actions)
    print(f"[zero-sum expert: s0->s2->s3] counts(9)={counts.tolist()} | freqs={np.round(freqs,3)}")

    return {
        "s": np.array(s_list, dtype=int),
        "a0": a0,
        "a1": a1,
        "joint_idx": _joint_to_index_3(a0, a1),
    }