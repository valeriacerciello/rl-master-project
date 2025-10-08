# scripts/test_coordination.py
import numpy as np
import torch

# --- imports from your codebase ---
from envs.entropy_coordination import (
    CoordinationGame,
    generate_asymmetric_bimodal_expert_data,  # or generate_expert_data, etc.
)
from MAGAIL import TabularPolicy, TabularStateJointDiscriminator
from MAGAIL import run_experiment
from MAGAIL import MAGAILTrainer
from MAGAIL import collect_policy_trajectories
from scipy.spatial.distance import jensenshannon

# quick seeding
def set_seed(seed=0):
    import random, os
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def smoke_test_env():
    print("== Smoke test: CoordinationGame ==")
    env = CoordinationGame()
    obs = env.reset()
    print("reset ->", obs)
    obs, r, d, _ = env.step({"agent_0": 0, "agent_1": 1})
    print("step (0,1) ->", obs, r, d)
    obs, r, d, _ = env.step({"agent_0": 1, "agent_1": 1})  # new episode ends immediately anyway
    print("step (1,1) ->", obs, r, d)

def short_train_run():
    print("\n== Short MAGAIL run on coordination env ==")
    set_seed(0)

    # 1) Expert data (bimodal: AA & BB 50/50)
    expert = generate_asymmetric_bimodal_expert_data(num_episodes=1000, AA_ratio=0.5, seed=0)

    # 2) Trainer (env_ctor=CoordinationGame makes it 1-step / 2-action automatically)
    trainer = MAGAILTrainer(
        env_ctor=CoordinationGame,
        beta=0.5,                # try a bit of entropy regularization
        reward_style="non_saturating",
        lr_policy=0.01,
        lr_disc=0.01,
    )

    # 3) Train briefly
    trainer.train(
        expert_data=expert,
        num_epochs=500,          # small for a quick check
        batch_size=64,
        collect_every=50,
        rollout_episodes=200,
    )

    # 4) Inspect policies
    p0 = trainer.policies["agent_0"].get_probs().detach().numpy()
    p1 = trainer.policies["agent_1"].get_probs().detach().numpy()
    print("Final policy probs:")
    print("  Agent 0:", np.round(p0, 3))
    print("  Agent 1:", np.round(p1, 3))

    # 5) Quick evaluation rollouts
    eval_roll = collect_policy_trajectories(trainer.policies, num_episodes=5000, env_ctor=CoordinationGame)
    match_rate = float((eval_roll["a0"] == eval_roll["a1"]).float().mean().item())

    # 6) JS distance to expert joint (4 bins)
    expert_joint = np.bincount(expert["joint_idx"], minlength=4)
    expert_joint = expert_joint / expert_joint.sum()
    learner_idx = (eval_roll["a0"] * 2 + eval_roll["a1"]).cpu().numpy()
    learner_joint = np.bincount(learner_idx, minlength=4)
    learner_joint = learner_joint / learner_joint.sum()
    js = float(jensenshannon(expert_joint, learner_joint, base=2.0))

    print(f"Match rate (P[a0==a1]): {match_rate:.3f}")
    print(f"JS distance (learner vs expert joint): {js:.3f}")

if __name__ == "__main__":
    smoke_test_env()
    short_train_run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MAGAIL experiment (coordination | zero_sum)")
    parser.add_argument("--env", type=str, default="coordination",
                        choices=["coordination", "zero_sum"],
                        help="Which environment to use.")
    parser.add_argument("--expert_type", type=str, default="bimodal",
                        help=("Expert type. For 'coordination': one of "
                              "mixed|bimodal|asymmetric|noisy|all_AA. "
                              "For 'zero_sum': ignored (uses s0->s2->s3)."))
    parser.add_argument("--betas", type=float, nargs="+",
                        default=[0.0, 0.1, 0.5, 1.0, 2.0, 5.0],
                        help="Entropy regularization coefficients.")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 123, 456, 789, 999],
                        help="Random seeds.")
    parser.add_argument("--epochs", type=int, default=2000,
                        help="Training epochs.")
    parser.add_argument("--rollout_episodes", type=int, default=200,
                        help="Episodes collected per epoch (for policy data).")
    parser.add_argument("--eval_episodes", type=int, default=5000,
                        help="Episodes for evaluation rollouts.")
    parser.add_argument("--lr_policy", type=float, default=0.01)
    parser.add_argument("--lr_disc", type=float, default=0.01)
    parser.add_argument("--reward_style", type=str, default="non_saturating",
                        choices=["non_saturating", "gail"])
    parser.add_argument("--policy_init_uniform", action="store_true",
                        help="Start policies uniform instead of random.")
    parser.add_argument("--expert_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--collect_every", type=int, default=10)
    parser.add_argument("--no_plots", action="store_true", help="Skip plotting (if you wired plots).")

    args = parser.parse_args()

    print(f"Starting MAGAIL on env='{args.env}' with expert_type='{args.expert_type}' ...")
    results, expert_data, collect_every = run_experiment(
        env_name=args.env,
        seeds=args.seeds,
        beta_values=args.betas,
        num_epochs=args.epochs,
        expert_type=args.expert_type,
        policy_init_uniform=args.policy_init_uniform,
        reward_style=args.reward_style,
        rollout_episodes=args.rollout_episodes,
        eval_episodes=args.eval_episodes,
        lr_policy=args.lr_policy,
        lr_disc=args.lr_disc,
        batch_size=args.batch_size,
        collect_every=args.collect_every,
        expert_seed=args.expert_seed,
    )

    # ---- quick textual summary ----
    print("\n=== Summary ===")
    for beta, by_seed in results.items():
        coord_rates = [by_seed[s]["coordination_rate"] for s in by_seed]
        js_vals = [by_seed[s]["js_distance"] for s in by_seed]
        mean_coord = float(np.mean(coord_rates)) if coord_rates else float("nan")
        mean_js = float(np.mean(js_vals)) if js_vals else float("nan")
        print(f"β={beta}: match-rate={mean_coord:.3f} | JS={mean_js:.3f}")

    # (Optional) If you wired a generic analyze/plot, you can call it here:
    # analysis = analyze_results(results, expert_data, spec_or_envinfo?, use_sample_var=True)
    # if not args.no_plots:
    #     plot_results(results, analysis, spec?, collect_every=collect_every)

    print("\nDone.")
