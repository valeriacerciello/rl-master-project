# runners/zero_sum_runner.py
"""
Zero-sum runner (discounted 7-state game).

- Trains MAGAIL on the discounted, looping ZeroSumGame.
- Trains a BC baseline using geometric rollouts from expert policies.
- Evaluates both via (R, P, rho0, gamma) and plots exploitability.

Expected modules:
  - envs.zero_sum: ZeroSumGame, generate_expert_data
  - MAGAIL: MAGAILTrainer, set_seed
  - BC: MultiAgentBehaviorCloning
  - calc_exploitability: calc_exploitability_true, plot_results
"""

from __future__ import annotations

import os
import sys
import numpy as np
from typing import Dict, Tuple

# Ensure local imports work when running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.zero_sum import ZeroSumGame, generate_expert_data
from MAGAIL import MAGAILTrainer, set_seed
from BC import MultiAgentBehaviorCloning as BC
from calc_exploitability import calc_exploitability_true, plot_results


def _extract_policy_snapshots(history: Dict) -> list[Tuple[np.ndarray, np.ndarray]]:
    """Convert MAGAIL history into a list of (mu_pi, nu_pi) arrays shaped (S, A)."""
    p0_list = history["policy_probs"]["agent_0"]
    p1_list = history["policy_probs"]["agent_1"]
    assert len(p0_list) == len(p1_list)
    pairs: list[Tuple[np.ndarray, np.ndarray]] = []
    for p0, p1 in zip(p0_list, p1_list):
        mu = p0 if np.asarray(p0).ndim == 2 else np.asarray(p0)[None, :]
        nu = p1 if np.asarray(p1).ndim == 2 else np.asarray(p1)[None, :]
        pairs.append((np.asarray(mu, dtype=float), np.asarray(nu, dtype=float)))
    return pairs


def _compute_exploitability_curve(history: Dict, R: np.ndarray, P: np.ndarray, rho0: np.ndarray, gamma: float) -> np.ndarray:
    """Compute exploitability at each logged snapshot in history."""
    vals = []
    for mu_pi, nu_pi in _extract_policy_snapshots(history):
        vals.append(
            calc_exploitability_true(
                mu_pi, nu_pi,
                reward=R, transition=P, initial_dist=rho0, gamma=gamma
            )
        )
    return np.asarray(vals, dtype=np.float32)


def main() -> None:
    # ----- True tensors from discounted 7-state ZeroSumGame -----
    R, P, rho0, gamma = ZeroSumGame().to_markov_game_tensors()
    S, A = R.shape[0], R.shape[1]

    # ===== 1) Train MAGAIL =====
    expert_total = 1000
    expert_seed = 0
    expert_data = generate_expert_data(total_samples=expert_total, gamma=gamma, expert_action=2, seed=expert_seed)

    magail_seeds = [42, 123, 456]
    num_epochs = 1000
    collect_every = 10
    rollout_episodes = 200
    lr_policy = 0.01
    lr_disc = 0.01
    beta = 0.0  # zero entropy for zero-sum

    all_queries_magail: list[np.ndarray] = []
    all_exploits_magail: list[np.ndarray] = []

    print("=== MAGAIL on discounted ZeroSumGame ===")
    for sd in magail_seeds:
        print(f"  Seed {sd}...", end="")
        set_seed(sd)
        trainer = MAGAILTrainer(
            env_ctor=ZeroSumGame,
            beta=beta,
            lr_policy=lr_policy,
            lr_disc=lr_disc,
            policy_init_uniform=False,
            reward_style="non_saturating",
            num_states_for_D=7,
            n_actions=3,
        )

        trainer.train(
            expert_data=expert_data,
            num_epochs=num_epochs,
            batch_size=128,
            collect_every=collect_every,
            rollout_episodes=rollout_episodes
        )

        curve = _compute_exploitability_curve(trainer.history, R, P, rho0, gamma)
        magail_q = np.asarray(trainer.history["expert_queries"], dtype=int)  # starts at 0
        all_queries_magail.append(magail_q)
        all_exploits_magail.append(curve)
        print(f" done. final exploitability = {curve[-1]:.3f} (snapshots={len(curve)})")

    # ===== 2) Train BC baseline =====
    expert1 = np.zeros((S, A), dtype=np.float32); expert1[:, 2] = 1.0
    expert2 = np.zeros((S, A), dtype=np.float32); expert2[:, 2] = 1.0

    bc_total_samples = 128_000
    bc_eval_interval = 1_280
    bc_seeds = [42, 123, 456]

    all_queries_bc: list[np.ndarray] = []
    all_exploits_bc: list[np.ndarray] = []

    print("=== BC on discounted ZeroSumGame ===")
    for sd in bc_seeds:
        rng = np.random.default_rng(sd)
        # Keep the BC API; it internally seeds its own Generator when given rng
        bc = BC(
            expert_policies=(expert1, expert2),
            total_samples=bc_total_samples,
            transition=P,
            initial_state_dist=rho0,
            payoff_matrix=R,
            gamma=gamma,
        )
        _, _, iters, exploits = bc.train(eval_interval=bc_eval_interval, calc_exploitability_true=calc_exploitability_true)
        all_queries_bc.append(np.asarray(iters, dtype=int))
        all_exploits_bc.append(np.asarray(exploits, dtype=float))
        print(f"  Seed {sd}: final exploitability = {exploits[-1]:.3f} (len={len(exploits)})")

    # ===== 3) Plot =====
    plot_results(
        all_queries_magail=all_queries_magail,
        all_exploits_magail=all_exploits_magail,
        all_queries_bc=all_queries_bc,
        all_exploits_bc=all_exploits_bc,
        name="zero_sum_discounted",
    )

    # Optional: concise summary at the last logged point
    if len(all_exploits_magail) > 0:
        Lm = min(len(c) for c in all_exploits_magail)
        mag_mean = np.mean([c[:Lm] for c in all_exploits_magail], axis=0)
        mag_std = np.std([c[:Lm] for c in all_exploits_magail], axis=0)
        print(f"MAGAIL last snapshot: {mag_mean[-1]:.3f} ± {mag_std[-1]:.3f}")
    if len(all_exploits_bc) > 0:
        Lb = min(len(c) for c in all_exploits_bc)
        bc_mean = np.mean([c[:Lb] for c in all_exploits_bc], axis=0)
        bc_std = np.std([c[:Lb] for c in all_exploits_bc], axis=0)
        print(f"BC     last snapshot: {bc_mean[-1]:.3f} ± {bc_std[-1]:.3f}")


if __name__ == "__main__":
    # Allow running via: python runners/zero_sum_runner.py
    main()
