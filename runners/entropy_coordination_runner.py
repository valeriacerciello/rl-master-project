# runners/entropy_coordination_runner.py

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial.distance import jensenshannon
from typing import Dict

# Ensure local imports work when running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from MAGAIL import run_experiment


# =========================
# Helpers (coordination)
# =========================
def _get_prob_A(final_probs) -> float:
    """
    final_probs can be shape (2,) or (1,2) depending on how StateTabularPolicy is stored.
    Return P(A) (action index 0).
    """
    p = np.asarray(final_probs, dtype=float)
    return float(p[0, 0] if p.ndim == 2 else p[0])


def _best_independent_js(expert_joint: np.ndarray, grid: int = 301) -> float:
    """
    Brute-force (p0,p1) ∈ [0,1]^2 to approximate the minimal JS distance
    between expert_joint and any independent product p0 x p1 over {A,B}.
    """
    ps = np.linspace(0.0, 1.0, grid)
    best = np.inf
    for p0 in ps:
        for p1 in ps:
            prod = np.array([p0*p1, p0*(1-p1), (1-p0)*p1, (1-p0)*(1-p1)], dtype=float)
            d = float(jensenshannon(expert_joint, prod, base=2.0))
            if d < best:
                best = d
    return best


# =========================
# Analysis (joint-based)
# =========================
def analyze_results(results: Dict, expert_data=None, use_sample_var: bool = True, report_js_divergence: bool = False) -> Dict:
    """
    Computes across-seed stats per β:
      - mean/variance of P(A) per agent
      - JS metric between learner JOINT and expert JOINT (distance by default)
    """
    analysis: Dict = {}
    ddof = 1 if use_sample_var else 0

    for beta in results.keys():
        probA0, probA1, js_list = [], [], []
        for seed, res in results[beta].items():
            pA0 = _get_prob_A(res["final_probs"]["agent_0"])
            pA1 = _get_prob_A(res["final_probs"]["agent_1"])
            probA0.append(pA0)
            probA1.append(pA1)

            expert_joint = np.asarray(res["expert_joint"], dtype=float)
            learner_joint = np.asarray(res["learner_joint"], dtype=float)
            jsd = jensenshannon(expert_joint, learner_joint, base=2.0)
            if report_js_divergence:
                jsd = jsd**2
            js_list.append(float(jsd))

        analysis[beta] = {
            "prob_A_mean": {
                "agent_0": float(np.mean(probA0)),
                "agent_1": float(np.mean(probA1)),
            },
            "prob_A_variance": {
                "agent_0": float(np.var(probA0, ddof=ddof)),
                "agent_1": float(np.var(probA1, ddof=ddof)),
            },
            ("js_divergence" if report_js_divergence else "js_distance"): {
                "mean": float(np.mean(js_list)),
                "variance": float(np.var(js_list, ddof=ddof)),
            },
            "final_probs_all_seeds": {"agent_0": probA0, "agent_1": probA1},
        }
    return analysis


# =========================
# Plotting
# =========================
def plot_results(results: Dict, analysis: Dict, collect_every: int = 10) -> None:
    """
    results: dict[beta] -> dict[seed] -> {...}
    analysis: dict with per-beta summaries
    collect_every: stride used when sampling training history (epochs)
    """
    def sort_betas(betas):
        return sorted(betas, key=lambda b: float(b))

    def normalize_extremes(analysis_obj, betas_sorted):
        """Return beta values to highlight (indices or explicit values supported)."""
        if isinstance(analysis_obj, dict) and "extremes" in analysis_obj:
            raw = analysis_obj["extremes"]
            if not isinstance(raw, (list, tuple)):
                raw = [raw]
            norm = []
            for e in raw:
                if isinstance(e, (int, np.integer)):
                    if 0 <= int(e) < len(betas_sorted):
                        norm.append(betas_sorted[int(e)])
                else:
                    try:
                        e_float = float(e)
                        idx = int(np.argmin(np.abs(np.array([float(b) for b in betas_sorted]) - e_float)))
                        norm.append(betas_sorted[idx])
                    except Exception:
                        pass
            if norm:
                return list(dict.fromkeys(norm))
        return betas_sorted[:1] if len(betas_sorted) == 1 else [betas_sorted[0], betas_sorted[-1]]

    betas_sorted = sort_betas(list(results.keys()))
    beta_values = betas_sorted
    seeds = sorted(list(results[beta_values[0]].keys()))
    extremes = normalize_extremes(analysis, betas_sorted)

    beta_cmap = mpl.colormaps.get_cmap('tab10').resampled(max(len(beta_values), 1))

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('MAGAIL Entropy Experiment Results', fontsize=16)

    # ---------- Plot 1: P(A) vs β (violins) ----------
    ax1 = axes[0, 0]
    data0 = [analysis[b]["final_probs_all_seeds"]["agent_0"] for b in beta_values]
    data1 = [analysis[b]["final_probs_all_seeds"]["agent_1"] for b in beta_values]

    x = np.arange(len(beta_values)).astype(float)
    offset = 0.18
    width = 0.30

    v0 = ax1.violinplot(data0, positions=x - offset, widths=width, showmeans=True, showextrema=False, showmedians=False)
    v1 = ax1.violinplot(data1, positions=x + offset, widths=width, showmeans=True, showextrema=False, showmedians=False)

    for pc in v0['bodies']:
        pc.set_alpha(0.4)
    for pc in v1['bodies']:
        pc.set_alpha(0.4)
    for coll in [v0.get('cmeans'), v1.get('cmeans')]:
        if coll is not None:
            try:
                coll.set_linewidths(2.0)
            except Exception:
                try:
                    coll.set_linewidth(2.0)
                except Exception:
                    pass

    ax1.axhline(0.5, linestyle='--', alpha=0.7, color='red', label='Max entropy policy')
    ax1.set_xlim(-0.6, len(beta_values) - 0.4)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{b}' for b in beta_values])
    ax1.set_xlabel('β')
    ax1.set_ylabel('P(A) across seeds')
    ax1.set_title('Learning Stability')

    handles, labels = ax1.get_legend_handles_labels()
    vio1_patch = mpl.patches.Patch(alpha=0.4, label="Agent 0")
    vio2_patch = mpl.patches.Patch(color="orange", alpha=0.4, label="Agent 1")
    handles.extend([vio1_patch, vio2_patch])
    labels.extend(["Agent 0", "Agent 1"])
    ax1.legend(handles, labels, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # ---------- Plot 2: JS (joint) vs β ----------
    ax2 = axes[0, 1]
    js_per_beta = []
    for b in beta_values:
        js_vals = []
        for s in seeds:
            js_vals.append(float(jensenshannon(
                np.asarray(results[b][s]["expert_joint"], dtype=float),
                np.asarray(results[b][s]["learner_joint"], dtype=float),
                base=2.0
            )))
        js_per_beta.append(js_vals)

    x = np.arange(len(beta_values)).astype(float)
    v = ax2.violinplot(js_per_beta, positions=x, widths=0.6, showmeans=True, showextrema=False, showmedians=False)
    for pc in v['bodies']:
        pc.set_alpha(0.4)
        pc.set_facecolor("green")
    if v.get('cmeans') is not None:
        try:
            v['cmeans'].set_linewidths(2.0)
            v['cmeans'].set_color("green")
        except Exception:
            try:
                v['cmeans'].set_linewidth(2.0)
                v['cmeans'].set_color("green")
            except Exception:
                pass

    exemplar = results[beta_values[0]][seeds[0]]
    expert_joint = np.asarray(exemplar["expert_joint"], dtype=float)
    indep_limit = _best_independent_js(expert_joint, grid=301)
    ax2.axhline(indep_limit, color="red", linestyle='--', alpha=0.7, label='Independent-policy limit')

    ax2.set_xlabel('β')
    ax2.set_ylabel('JS distance (joint)')
    ax2.set_title('JS Distance vs β')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{b}' for b in beta_values])
    ax2.grid(True, alpha=0.3)

    handles, labels = ax2.get_legend_handles_labels()
    vio_patch = mpl.patches.Patch(color="green", alpha=0.4, label="Across-seed distribution")
    handles.append(vio_patch)
    labels.append("Across-seed distribution")
    ax2.legend(handles, labels, loc='upper right')

    # ---------- Plot 3: Final P(A) per seed ----------
    ax3 = axes[0, 2]
    markers = ['o', 's']
    for i, b in enumerate(beta_values):
        pa0 = analysis[b]["final_probs_all_seeds"]["agent_0"]
        pa1 = analysis[b]["final_probs_all_seeds"]["agent_1"]
        jitter = 0.02
        xs = [i + np.random.uniform(-jitter, jitter) for _ in range(len(seeds))]
        ax3.scatter(xs, pa0, alpha=0.85, color=beta_cmap(i), marker=markers[0], s=50,
                    label=f'β={b} A0' if i < 2 else None)
        ax3.scatter(xs, pa1, alpha=0.85, color=beta_cmap(i), marker=markers[1], s=50,
                    label=f'β={b} A1' if i < 2 else None)
    ax3.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='0.5 ref')
    ax3.set_xlabel('β index (jittered)')
    ax3.set_ylabel('P(A)')
    ax3.set_title('Final Policy Probabilities Across Seeds')
    ax3.set_xticks(np.arange(len(beta_values)))
    ax3.set_xticklabels([f'{b}' for b in beta_values])
    ax3.legend(fontsize=9, ncol=2)
    ax3.grid(True, alpha=0.3)

    # ---------- Plot 4: Equilibrium selection vs β ----------
    ax4 = axes[1, 0]
    eps = 0.05

    prop_AA, prop_BB, prop_sym = [], [], []
    for b in beta_values:
        aa = bb = sym = 0
        for s in seeds:
            p0 = _get_prob_A(results[b][s]["final_probs"]["agent_0"])
            p1 = _get_prob_A(results[b][s]["final_probs"]["agent_1"])
            if p0 > 0.5 + eps and p1 > 0.5 + eps:
                aa += 1
            elif p0 < 0.5 - eps and p1 < 0.5 - eps:
                bb += 1
            else:
                sym += 1
        total = aa + bb + sym if (aa + bb + sym) > 0 else 1
        prop_AA.append(aa / total)
        prop_BB.append(bb / total)
        prop_sym.append(sym / total)

    x = np.arange(len(beta_values))
    ax4.bar(x, prop_AA, alpha=0.85, label='Collapse to AA')
    ax4.bar(x, prop_BB, bottom=prop_AA, alpha=0.85, label='Collapse to BB')
    bottom = (np.array(prop_AA) + np.array(prop_BB)).tolist()
    ax4.bar(x, prop_sym, bottom=bottom, alpha=0.85, label='Symmetric (no collapse)')

    ax4.set_xlabel('β')
    ax4.set_ylabel('Proportion of seeds')
    ax4.set_title('Equilibrium Selection Across Seeds')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{b}' for b in beta_values])
    ax4.set_ylim(0.0, 1.0)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right')

    # ---------- Plot 5: Final joint vs Expert joint (extreme β) ----------
    ax5 = axes[1, 1]
    action_names = ['(A,A)', '(A,B)', '(B,A)', '(B,B)']
    x = np.arange(len(action_names))
    width = 0.35

    exemplar = results[beta_values[0]][seeds[0]]
    expert_joint = np.asarray(exemplar["expert_joint"], dtype=float)

    for j, b in enumerate(extremes):
        final_joint = [np.asarray(results[b][s]["history"]["joint_action_dist"][-1], dtype=float) for s in seeds]
        mean_dist = np.mean(final_joint, axis=0)
        std_dist = np.std(final_joint, axis=0)
        offset = -width/2 if j == 0 else width/2
        ax5.bar(x + offset, mean_dist, width, alpha=0.8, yerr=std_dist, capsize=5, label=f'β={b}')
    ax5.plot(x, expert_joint, 'r--o', linewidth=2, markersize=5, label='Expert joint')
    ax5.set_xlabel('Joint Actions')
    ax5.set_ylabel('Probability')
    ax5.set_title('Final Joint Action Distribution')
    ax5.set_xticks(x)
    ax5.set_xticklabels(action_names)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # ---------- Plot 6: Policy P(A) evolution (repr. seed) ----------
    ax6 = axes[1, 2]
    for b in extremes:
        seed = seeds[0]
        prob_hist = results[b][seed]["history"]["policy_probs"]["agent_0"]
        epochs = np.arange(len(prob_hist)) * collect_every

        def _pA_of_step(arr):
            arr = np.asarray(arr, dtype=float)
            return arr[0, 0] if arr.ndim == 2 else arr[0]

        ax6.plot(epochs, [_pA_of_step(p) for p in prob_hist], label=f'β={b}, P(A)', linewidth=2)
    ax6.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='0.5 ref')
    ax6.set_xlabel('Training Epoch')
    ax6.set_ylabel('P(A) for Agent 0')
    ax6.set_title('Policy Evolution During Training')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def coordination_consistency(results: Dict) -> None:
    """Print mean/std of (AA+BB) mass for each β."""
    for beta in sorted(results.keys(), key=lambda b: float(b)):
        coord_probs = []
        for seed in results[beta].keys():
            final_dist = np.asarray(results[beta][seed]["history"]["joint_action_dist"][-1], dtype=float)
            coord_prob = float(final_dist[0] + final_dist[3])  # AA + BB
            coord_probs.append(coord_prob)
        print(f"β={beta}: Mean coordination = {np.mean(coord_probs):.3f}, Std = {np.std(coord_probs):.3f}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run MAGAIL coordination experiment")
    parser.add_argument("--expert_type", type=str, default="bimodal",
                        choices=["mixed", "bimodal", "asymmetric", "noisy", "all_AA"])
    parser.add_argument("--betas", type=float, nargs="+",
                        default=[0.0, 0.1, 0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 123, 456, 789, 999])
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--rollout_episodes", type=int, default=200)
    parser.add_argument("--eval_episodes", type=int, default=5000)
    parser.add_argument("--lr_policy", type=float, default=0.01)
    parser.add_argument("--lr_disc", type=float, default=0.01)
    parser.add_argument("--reward_style", type=str, default="non_saturating",
                        choices=["non_saturating", "gail"])
    parser.add_argument("--policy_init_uniform", action="store_true",
                        help="Start policies at 0.5/0.5 instead of random")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--collect_every", type=int, default=10)
    args = parser.parse_args()

    print("Starting MAGAIL entropy experiment...")
    results, expert_data, collect_every = run_experiment(
        env_name="coordination",
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
        expert_seed=0,
    )

    print("\nAnalyzing results...")
    analysis = analyze_results(results, expert_data, use_sample_var=True, report_js_divergence=False)

    print("\nGenerating plots...")
    plot_results(results, analysis, collect_every=collect_every)

    print("\nCoordination consistency:")
    coordination_consistency(results)
    print("\nDone.")


if __name__ == "__main__":
    main()
