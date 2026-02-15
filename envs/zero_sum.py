# envs/zero_sum.py

import numpy as np

# Environment
class ZeroSumGame:
    """
    Discounted, looping 7-state game used in the 'ZeroSum' experiments.

    States: [s0, s1, s2, s3, Sxplt1, Sxplt2, Scopy]
    Actions per agent: 3

    Rewards (per step):
      Sxplt1: +1
      Sxplt2: -1
      others: 0

    Transitions:
      s0: (2,2)->s2 else->s1
      s1: (0,0)|(1,1)->Sxplt1 ; (1,0)|(0,1)->Sxplt2 ; else->Scopy
      s2: -> s3
      s3,Sxplt1,Sxplt2,Scopy: self-loop
    """
    def __init__(self):
        self.num_agents = 2
        self.num_actions = 3
        self.states = ["s0","s1","s2","s3","Sxplt1","Sxplt2","Scopy"]
        self.num_states = len(self.states)
        self.start_state = "s0"
        self._name2idx = {s:i for i,s in enumerate(self.states)}
        self.reset()

    def reset(self):
        self.state = self.start_state
        return {"agent_0": self.state, "agent_1": self.state}

    def step(self, actions):
        a0, a1 = actions["agent_0"], actions["agent_1"]
        s = self._name2idx[self.state]
        ns = self._next_state_idx(s, a0, a1)
        self.state = self.states[ns]

        # per-step reward fountain
        r = 0.0
        if self.state == "Sxplt1":
            r = +1.0
        elif self.state == "Sxplt2":
            r = -1.0

        obs = {"agent_0": self.state, "agent_1": self.state}
        # no episodic termination
        dones = {"agent_0": False, "agent_1": False, "__all__": False}
        rewards = {"agent_0": r, "agent_1": -r}  # zero-sum 
        return obs, rewards, dones, {}

    def _next_state_idx(self, s, a, b):
        s0,s1,s2,s3,Sxplt1,Sxplt2,Scopy = range(self.num_states)
        if s == s0:
            return s2 if (a==2 and b==2) else s1
        if s == s1:
            if (a,b) in [(0,0),(1,1)]: return Sxplt1
            if (a,b) in [(1,0),(0,1)]: return Sxplt2
            return Scopy
        if s == s2: return s3
        # self-loops
        return s

    # tensors for exploitability / DP
    def to_markov_game_tensors(self):
        S, A = self.num_states, self.num_actions
        s0,s1,s2,s3,Sxplt1,Sxplt2,Scopy = range(S)

        R = np.zeros((S,A,A), dtype=np.float32)
        P = np.zeros((S,A,A,S), dtype=np.float32)

        R[Sxplt1,:,:] = +1.0
        R[Sxplt2,:,:] = -1.0

        for a in range(A):
            for b in range(A):
                P[s0,a,b, s2 if (a==2 and b==2) else s1] = 1.0

        P[s1,0,0,Sxplt1] = 1.0
        P[s1,1,1,Sxplt1] = 1.0
        P[s1,1,0,Sxplt2] = 1.0
        P[s1,0,1,Sxplt2] = 1.0
        for a in range(A):
            for b in range(A):
                if (a,b) not in [(0,0),(1,1),(1,0),(0,1)]:
                    P[s1,a,b,Scopy] = 1.0

        for a in range(A):
            for b in range(A):
                P[s2,a,b,s3] = 1.0

        for s in [s3,Sxplt1,Sxplt2,Scopy]:
            for a in range(A):
                for b in range(A):
                    P[s,a,b,s] = 1.0

        rho0 = np.zeros(S); rho0[s0] = 1.0
        gamma = 0.9
        return R, P, rho0, gamma


# =========================
# Expert data generator
# =========================
def generate_expert_data(total_samples=1000, gamma=0.9, expert_action=2, seed=None):
    """
    Rollouts under a degenerate expert that always plays `expert_action`.
    """
    rng = np.random.RandomState(seed)
    env = ZeroSumGame()
    S = env.num_states; A = env.num_actions
    name2idx = {s:i for i,s in enumerate(env.states)}

    s_list, a0_list, a1_list = [], [], []

    def geo_len():  # parameter 1-gamma
        return rng.geometric(1.0 - gamma)

    while len(s_list) < total_samples:
        obs = env.reset()
        s = name2idx[obs["agent_0"]]
        T = geo_len()
        for _ in range(T):
            a0 = expert_action
            a1 = expert_action
            s_list.append(s); a0_list.append(a0); a1_list.append(a1)
            # step environment
            next_obs, _, _, _ = env.step({"agent_0": a0, "agent_1": a1})
            s = name2idx[next_obs["agent_0"]]
            if len(s_list) >= total_samples:
                break

    s = np.asarray(s_list, dtype=int)
    a0 = np.asarray(a0_list, dtype=int)
    a1 = np.asarray(a1_list, dtype=int)
    joint_idx = a0 * A + a1
    return {"s": s, "a0": a0, "a1": a1, "joint_idx": joint_idx}