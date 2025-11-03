# BC.py

import numpy as np
import numpy.typing as npt
from __future__ import annotations
from typing import Tuple, List, Callable


__all__ = ["MultiAgentBehaviorCloning"]


def _as_array(x: npt.ArrayLike, dtype=float) -> np.ndarray:
    return np.asarray(x, dtype=dtype)


def _check_gamma(gamma: float) -> None:
    if not (0.0 <= gamma < 1.0):
        raise ValueError(f"gamma must be in [0,1); got {gamma}.")


def _row_stochastic(mat: np.ndarray, axis: int = 1, atol: float = 1e-6) -> bool:
    return np.allclose(mat.sum(axis=axis), 1.0, atol=atol)

# ============================================================
# Multi-Agent Behavior Cloning with Geometric Rollouts
# ============================================================
class MultiAgentBehaviorCloning:
    """
    Multi-agent behavior cloning with geometric rollouts.
    A geometric rollout has length T ~ Geom(p=1-γ), i.e., E[T] = 1/(1-γ).

    Args:
        expert_policies      : tuple of (S, A) arrays, one per agent.
        total_samples        : number of (state, action) samples to draw per agent.
        transition           : (S, A, A, S) transition tensor.
        initial_state_dist   : (S,) start-state distribution.
        payoff_matrix        : (S, A, A) payoff to agent 1 (not used here, kept for API parity).
        gamma                : discount factor in [0,1).
    """

    def __init__(
        self,
        expert_policies: Tuple[np.ndarray, np.ndarray],   # (S, A) each
        total_samples: int,
        transition: np.ndarray,                           # (S, A, A, S)
        initial_state_dist: np.ndarray,                   # (S,)
        payoff_matrix: np.ndarray,                        # (S, A, A)  (kept for API compatibility)
        gamma: float,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.expert1, self.expert2 = ( _as_array(expert_policies[0]),
                                       _as_array(expert_policies[1]) )
        self.total_samples = int(total_samples)
        self.P = _as_array(transition)
        self.rho0 = _as_array(initial_state_dist)
        self.R = _as_array(payoff_matrix)  # not used in BC updates; retained for signature symmetry
        self.gamma = float(gamma)
        self.rng = rng if rng is not None else np.random.default_rng()

        _check_gamma(self.gamma)

        if self.expert1.ndim != 2:
            raise ValueError("expert1 must have shape (S, A).")
        if self.expert2.shape != self.expert1.shape:
            raise ValueError("expert2 must match expert1 shape (S, A).")

        self.S, self.A = self.expert1.shape

        if self.P.shape != (self.S, self.A, self.A, self.S):
            raise ValueError(f"transition must have shape (S, A, A, S); got {self.P.shape}.")
        if self.rho0.shape != (self.S,):
            raise ValueError(f"initial_state_dist must have shape (S,); got {self.rho0.shape}.")
        if self.R.shape != (self.S, self.A, self.A):
            raise ValueError(f"payoff_matrix must have shape (S, A, A); got {self.R.shape}.")

        # Basic stochasticity checks
        if not _row_stochastic(self.expert1, axis=1):
            raise ValueError("Rows of expert1 must sum to 1.")
        if not _row_stochastic(self.expert2, axis=1):
            raise ValueError("Rows of expert2 must sum to 1.")
        if not np.allclose(self.rho0.sum(), 1.0, atol=1e-6):
            raise ValueError("initial_state_dist must sum to 1.")
        P_rows = self.P.reshape(self.S * self.A * self.A, self.S).sum(axis=1)
        if not np.allclose(P_rows, 1.0, atol=1e-6):
            raise ValueError("Each (s,a,b)-row of transition must sum to 1.")

    # =========================
    # Sampling
    # =========================
    def _rollout_geometric(self) -> List[tuple[int, int, int]]:
        """Return a single trajectory: list of (s, a1, a2) with T ~ Geom(1-γ)."""
        traj: List[tuple[int, int, int]] = []
        s = int(self.rng.choice(self.S, p=self.rho0))
        # Geometric with support {1,2,...} and success prob p = 1-γ
        T = int(self.rng.geometric(1.0 - self.gamma))
        for _ in range(T):
            a1 = int(self.rng.choice(self.A, p=self.expert1[s]))
            a2 = int(self.rng.choice(self.A, p=self.expert2[s]))
            traj.append((s, a1, a2))
            s = int(self.rng.choice(self.S, p=self.P[s, a1, a2]))
        return traj

    # =========================
    # Estimation
    # =========================
    @staticmethod
    def _counts_to_policy(counts: np.ndarray) -> np.ndarray:
        """
        Convert per-state action counts to a stochastic policy.
        States with zero total count become uniform over actions.
        """
        totals = counts.sum(axis=1, keepdims=True)
        # Use out/where to avoid division warnings; default to uniform.
        return np.divide(
            counts,
            np.maximum(totals, 1.0),
            out=np.full_like(counts, 1.0 / counts.shape[1], dtype=float),
            where=(totals > 0.0),
        )

    # =========================
    # Training
    # =========================
    def train(
        self,
        eval_interval: int,
        calc_exploitability_true: Callable[
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float], float
        ],
    ) -> tuple[np.ndarray, np.ndarray, List[int], List[float]]:
        """
        Draws total_samples joint (state, action1, action2) from geometric rollouts,
        forms empirical policies by counting, and periodically evaluates exploitability.

        Returns:
            pol1, pol2 : learned (S, A) policies
            iters      : list of iteration indices at which exploitability was computed
            exploits   : corresponding exploitability values
        """
        if eval_interval <= 0:
            raise ValueError("eval_interval must be a positive integer.")

        data1: List[tuple[int, int]] = []
        data2: List[tuple[int, int]] = []

        # Collect samples
        while len(data1) < self.total_samples:
            for s, a1, a2 in self._rollout_geometric():
                data1.append((s, a1))
                data2.append((s, a2))
                if len(data1) >= self.total_samples:
                    break

        counts1 = np.zeros((self.S, self.A), dtype=float)
        counts2 = np.zeros((self.S, self.A), dtype=float)

        iters: List[int] = []
        exploits: List[float] = []

        # t = 0 evaluation (uniform policies)
        pol1 = self._counts_to_policy(counts1)
        pol2 = self._counts_to_policy(counts2)
        e0 = float(calc_exploitability_true(pol1, pol2, self.R, self.P, self.rho0, self.gamma))
        iters.append(0)
        exploits.append(e0)

        # Online counting + periodic evals
        for t, ((s1, a1), (s2, a2)) in enumerate(zip(data1, data2), start=1):
            counts1[s1, a1] += 1.0
            counts2[s2, a2] += 1.0

            if t % eval_interval == 0:
                pol1 = self._counts_to_policy(counts1)
                pol2 = self._counts_to_policy(counts2)
                e = float(calc_exploitability_true(pol1, pol2, self.R, self.P, self.rho0, self.gamma))
                iters.append(t)
                exploits.append(e)

        # Final eval if last step wasn't aligned with eval_interval
        if (len(data1) % eval_interval) != 0:
            pol1 = self._counts_to_policy(counts1)
            pol2 = self._counts_to_policy(counts2)
            e = float(calc_exploitability_true(pol1, pol2, self.R, self.P, self.rho0, self.gamma))
            iters.append(len(data1))
            exploits.append(e)

        return pol1, pol2, iters, exploits