# runners/zero_sum_runner.py
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
import argparse, numpy as np
from MAGAIL import run_experiment
from ppo import train_best_response

def parse_states(csv: str):
    if not csv:
        return tuple()
    return tuple(int(x) for x in csv.split(","))

def main():
    ap = argparse.ArgumentParser(description="Zero-sum: MAGAIL → PPO best response")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1500, help="MAGAIL epochs")
    ap.add_argument("--rollout_episodes", type=int, default=200)
    ap.add_argument("--eval_episodes", type=int, default=3000)
    ap.add_argument("--lr_policy", type=float, default=0.01)
    ap.add_argument("--lr_disc", type=float, default=0.01)
    ap.add_argument("--learner", type=str, choices=["agent_0","agent_1"], default="agent_0",
                    help="Which agent to train with PPO (the other stays fixed).")
    ap.add_argument("--forbid_states", type=str, default="0",
                    help="Comma-separated list of state indices where a3 is forbidden for the learner (e.g. '0' or '0,1,2').")
    ap.add_argument("--ppo_iters", type=int, default=30)
    ap.add_argument("--ppo_batch_episodes", type=int, default=1024)
    ap.add_argument("--ppo_lr", type=float, default=3e-3)
    args = ap.parse_args()

    # 1) Train MAGAIL on zero-sum (entropy-free by default inside run_experiment)
    print("== Training MAGAIL (zero_sum, β=0) ==")
    results, _, _ = run_experiment(
        env_name="zero_sum",
        seeds=[args.seed],
        beta_values=[0.0],
        num_epochs=args.epochs,
        rollout_episodes=args.rollout_episodes,
        eval_episodes=args.eval_episodes,
        lr_policy=args.lr_policy,
        lr_disc=args.lr_disc,
        batch_size=64,
        collect_every=25,
    )

    beta0 = list(results.keys())[0]
    seed0 = list(results[beta0].keys())[0]
    final = results[beta0][seed0]["final_probs"]  # dict: {"agent_0": [S,A], "agent_1": [S,A]}

    # 2) Decide who is fixed and who learns
    if args.learner == "agent_0":
        opponent_probs = final["agent_1"]
    else:
        opponent_probs = final["agent_0"]

    print("\nFixed opponent per-state probs:")
    for i, s in enumerate(["s0","s1","s2"]):
        print(f"  {s}: {np.round(opponent_probs[i], 3)}")

    forbid_states = parse_states(args.forbid_states)
    print(f"\n== PPO best response: learner={args.learner}, forbid a3 at states {forbid_states} ==")
    out = train_best_response(
        opponent_probs=opponent_probs,
        learner_key=args.learner,
        forbid_on_states=forbid_states,
        seed=args.seed,
        iters=args.ppo_iters,
        batch_episodes=args.ppo_batch_episodes,
        lr=args.ppo_lr,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
    )

    print("\nLearned learner policy (probs by state):")
    for i, s in enumerate(["s0","s1","s2"]):
        print(f"  {s}: {np.round(out['learner_policy_probs'][i], 3)}")
    print(f"\nPre PPO avg_return={out['pre_eval']['avg_return']:.3f}  | "
          f"Post PPO avg_return={out['post_eval']['avg_return']:.3f}")

if __name__ == "__main__":
    main()
