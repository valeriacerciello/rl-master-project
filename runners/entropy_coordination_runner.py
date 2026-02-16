# runners/entropy_coordination_runner.py

from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jensenshannon

# Ensure local imports work when running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from MAGAIL import run_experiment


# Helpers (coordination)
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
            prod = np.array(
                [p0 * p1, p0 * (1 - p1), (1 - p0) * p1, (1 - p0) * (1 - p1)],
                dtype=float,
            )
            d = float(jensenshannon(expert_joint, prod, base=2.0))
            if d < best:
                best = d
    return best


# Analysis (joint-based)
def analyze_results(
    results: Dict,
    expert_data=None,
    use_sample_var: bool = True,
    report_js_divergence: bool = False,
) -> Dict:
    """
    Computes across-seed stats per β:
      - mean/variance of P(A) per agent
      - JS metric between learner JOINT and expert JOINT (distance by default)
    """
    analysis: Dict = {}
    ddof = 1 if use_sample_var else 0

    for beta in results.keys():
        probA0, probA1, js_list = [], [], []
        for _, res in results[beta].items():
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
            "prob_A_mean": {"agent_0": float(np.mean(probA0)), "agent_1": float(np.mean(probA1))},
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


# Plotting (separate figures)
def plot_results_separate(
    results: Dict,
    analysis: Dict,
    collect_every: int = 10,
    out_prefix: Optional[str] = None,
    outdir: Optional[str] = None,
    show: bool = True,
    save_png: bool = False,
    save_pdf: bool = False,
) -> None:
    """
    Generates SIX separate figures (Plot 1..6). If saving is enabled, each plot is saved
    as its own file. If show=False, figures are closed.

    - out_prefix: base filename prefix, e.g. "new_bimodal"
    - outdir: output directory (created if missing)
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
                        idx = int(
                            np.argmin(
                                np.abs(np.array([float(b) for b in betas_sorted]) - e_float)
                            )
                        )
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

    beta_cmap = mpl.colormaps.get_cmap("tab10").resampled(max(len(beta_values), 1))

    if outdir is None:
        outdir = os.path.join("plots", "entropy_exp")
    if out_prefix is None:
        out_prefix = "new"

    if save_png or save_pdf:
        os.makedirs(outdir, exist_ok=True)

    def _save(fig, stem: str):
        """Save figure to disk according to save_png/save_pdf flags."""
        if not (save_png or save_pdf):
            return
        if save_png:
            p = os.path.join(outdir, f"{out_prefix}_{stem}.png")
            fig.savefig(p, dpi=300, bbox_inches="tight")
            print(f"[saved] {p}")
        if save_pdf:
            p = os.path.join(outdir, f"{out_prefix}_{stem}.pdf")
            fig.savefig(p, dpi=300, bbox_inches="tight")
            print(f"[saved] {p}")

    # Plot 1: P(A) vs β (violins)
    fig1, ax1 = plt.subplots(figsize=(7.5, 5.5))
    fig1.suptitle("Learning Stability", fontsize=14)

    data0 = [analysis[b]["final_probs_all_seeds"]["agent_0"] for b in beta_values]
    data1 = [analysis[b]["final_probs_all_seeds"]["agent_1"] for b in beta_values]

    x = np.arange(len(beta_values)).astype(float)
    offset = 0.18
    width = 0.30

    v0 = ax1.violinplot(
        data0, positions=x - offset, widths=width, showmeans=True, showextrema=False, showmedians=False
    )
    v1 = ax1.violinplot(
        data1, positions=x + offset, widths=width, showmeans=True, showextrema=False, showmedians=False
    )

    for pc in v0["bodies"]:
        pc.set_alpha(0.4)
    for pc in v1["bodies"]:
        pc.set_alpha(0.4)
    for coll in [v0.get("cmeans"), v1.get("cmeans")]:
        if coll is not None:
            try:
                coll.set_linewidths(2.0)
            except Exception:
                try:
                    coll.set_linewidth(2.0)
                except Exception:
                    pass

    ax1.axhline(0.5, linestyle="--", alpha=0.7, color="red", label="Max entropy policy")
    ax1.set_xlim(-0.6, len(beta_values) - 0.4)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{b}" for b in beta_values])
    ax1.set_xlabel("β")
    ax1.set_ylabel("P(A) across seeds")

    handles, labels = ax1.get_legend_handles_labels()
    vio1_patch = mpl.patches.Patch(alpha=0.4, label="Agent 0")
    vio2_patch = mpl.patches.Patch(color="orange", alpha=0.4, label="Agent 1")
    handles.extend([vio1_patch, vio2_patch])
    labels.extend(["Agent 0", "Agent 1"])
    ax1.legend(handles, labels, loc="upper right")
    ax1.grid(True, alpha=0.3)

    fig1.tight_layout()
    _save(fig1, "plot1_learning_stability")

    # Plot 2: JS (joint) vs β 
    fig2, ax2 = plt.subplots(figsize=(7.5, 5.5))
    fig2.suptitle("JS Distance vs β", fontsize=14)

    js_per_beta = []
    for b in beta_values:
        js_vals = []
        for s in seeds:
            js_vals.append(
                float(
                    jensenshannon(
                        np.asarray(results[b][s]["expert_joint"], dtype=float),
                        np.asarray(results[b][s]["learner_joint"], dtype=float),
                        base=2.0,
                    )
                )
            )
        js_per_beta.append(js_vals)

    x = np.arange(len(beta_values)).astype(float)
    v = ax2.violinplot(js_per_beta, positions=x, widths=0.6, showmeans=True, showextrema=False, showmedians=False)
    for pc in v["bodies"]:
        pc.set_alpha(0.4)
        pc.set_facecolor("green")
    if v.get("cmeans") is not None:
        try:
            v["cmeans"].set_linewidths(2.0)
            v["cmeans"].set_color("green")
        except Exception:
            try:
                v["cmeans"].set_linewidth(2.0)
                v["cmeans"].set_color("green")
            except Exception:
                pass

    exemplar = results[beta_values[0]][seeds[0]]
    expert_joint = np.asarray(exemplar["expert_joint"], dtype=float)
    indep_limit = _best_independent_js(expert_joint, grid=301)
    ax2.axhline(indep_limit, color="red", linestyle="--", alpha=0.7, label="Independent-policy limit")

    ax2.set_xlabel("β")
    ax2.set_ylabel("JS distance (joint)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{b}" for b in beta_values])
    ax2.grid(True, alpha=0.3)

    handles, labels = ax2.get_legend_handles_labels()
    vio_patch = mpl.patches.Patch(color="green", alpha=0.4, label="Across-seed distribution")
    handles.append(vio_patch)
    labels.append("Across-seed distribution")
    ax2.legend(handles, labels, loc="upper right")

    fig2.tight_layout()
    _save(fig2, "plot2_js_distance")

    # Plot 3: Final P(A) per seed (deterministic jitter)
    fig3, ax3 = plt.subplots(figsize=(7.5, 5.5))
    fig3.suptitle("Final Policy Probabilities Across Seeds", fontsize=14)

    markers = ["o", "s"]
    jitter = 0.02

    for i, b in enumerate(beta_values):
        pa0 = analysis[b]["final_probs_all_seeds"]["agent_0"]
        pa1 = analysis[b]["final_probs_all_seeds"]["agent_1"]

        rng = np.random.default_rng(int(10_000 * float(b)) + 12345)
        xs = i + rng.uniform(-jitter, jitter, size=len(seeds))

        ax3.scatter(
            xs,
            pa0,
            alpha=0.85,
            color=beta_cmap(i),
            marker=markers[0],
            s=50,
            label=f"β={b} A0" if i < 2 else None,
        )
        ax3.scatter(
            xs,
            pa1,
            alpha=0.85,
            color=beta_cmap(i),
            marker=markers[1],
            s=50,
            label=f"β={b} A1" if i < 2 else None,
        )

    ax3.axhline(0.5, color="red", linestyle="--", alpha=0.7, label="0.5 ref")
    ax3.set_xlabel("β")
    ax3.set_ylabel("P(A)")
    ax3.set_xticks(np.arange(len(beta_values)))
    ax3.set_xticklabels([f"{b}" for b in beta_values])
    ax3.legend(fontsize=9, ncol=2)
    ax3.grid(True, alpha=0.3)

    fig3.tight_layout()
    _save(fig3, "plot3_final_probs_scatter")

    # Plot 4: Equilibrium selection vs β 
    fig4, ax4 = plt.subplots(figsize=(7.5, 5.5))
    fig4.suptitle("Equilibrium Selection Across Seeds", fontsize=14)

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
    ax4.bar(x, prop_AA, alpha=0.85, label="Collapse to AA")
    ax4.bar(x, prop_BB, bottom=prop_AA, alpha=0.85, label="Collapse to BB")
    bottom = (np.array(prop_AA) + np.array(prop_BB)).tolist()
    ax4.bar(x, prop_sym, bottom=bottom, alpha=0.85, label="Symmetric (no collapse)")

    ax4.set_xlabel("β")
    ax4.set_ylabel("Proportion of seeds")
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"{b}" for b in beta_values])
    ax4.set_ylim(0.0, 1.0)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper right")

    fig4.tight_layout()
    _save(fig4, "plot4_equilibrium_selection")

    # Plot 5: Final joint vs Expert joint (extreme β)
    fig5, ax5 = plt.subplots(figsize=(7.5, 5.5))
    fig5.suptitle("Final Joint Action Distribution", fontsize=14)

    action_names = ["(A,A)", "(A,B)", "(B,A)", "(B,B)"]
    x = np.arange(len(action_names))
    width = 0.35

    exemplar = results[beta_values[0]][seeds[0]]
    expert_joint = np.asarray(exemplar["expert_joint"], dtype=float)

    for j, b in enumerate(extremes):
        final_joint = [np.asarray(results[b][s]["history"]["joint_action_dist"][-1], dtype=float) for s in seeds]
        mean_dist = np.mean(final_joint, axis=0)
        std_dist = np.std(final_joint, axis=0)
        off = -width / 2 if j == 0 else width / 2
        ax5.bar(x + off, mean_dist, width, alpha=0.8, yerr=std_dist, capsize=5, label=f"β={b}")

    ax5.plot(x, expert_joint, "r--o", linewidth=2, markersize=5, label="Expert joint")
    ax5.set_xlabel("Joint Actions")
    ax5.set_ylabel("Probability")
    ax5.set_xticks(x)
    ax5.set_xticklabels(action_names)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    fig5.tight_layout()
    _save(fig5, "plot5_final_joint_vs_expert")

    # Plot 6: Policy P(A) evolution (repr. seed) 
    fig6, ax6 = plt.subplots(figsize=(7.5, 5.5))
    fig6.suptitle("Policy Evolution During Training", fontsize=14)

    for b in extremes:
        seed = seeds[0]
        prob_hist = results[b][seed]["history"]["policy_probs"]["agent_0"]
        epochs = np.arange(len(prob_hist)) * collect_every

        def _pA_of_step(arr):
            arr = np.asarray(arr, dtype=float)
            return arr[0, 0] if arr.ndim == 2 else arr[0]

        ax6.plot(epochs, [_pA_of_step(p) for p in prob_hist], label=f"β={b}, P(A)", linewidth=2)

    ax6.axhline(0.5, color="red", linestyle="--", alpha=0.7, label="0.5 ref")
    ax6.set_xlabel("Training Epoch")
    ax6.set_ylabel("P(A) for Agent 0")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    fig6.tight_layout()
    _save(fig6, "plot6_policy_evolution")

    if show:
        plt.show()
    else:
        # close all six figs to avoid GUI / memory use in headless mode
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        plt.close(fig4)
        plt.close(fig5)
        plt.close(fig6)


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
    parser.add_argument(
        "--expert_type",
        type=str,
        default="bimodal",
        choices=["mixed", "bimodal", "asymmetric", "noisy", "all_AA"],
    )
    parser.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.1, 0.5, 1.0, 5.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 999])
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--rollout_episodes", type=int, default=200)
    parser.add_argument("--eval_episodes", type=int, default=5000)
    parser.add_argument("--lr_policy", type=float, default=0.01)
    parser.add_argument("--lr_disc", type=float, default=0.01)
    parser.add_argument("--reward_style", type=str, default="non_saturating", choices=["non_saturating", "gail"])
    parser.add_argument(
        "--policy_init_uniform",
        action="store_true",
        help="Start policies at 0.5/0.5 instead of random",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--collect_every", type=int, default=10)

    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join("plots", "entropy_exp"),
        help="Directory to save figures (paper uses plots/entropy_exp)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="new",
        help="Filename prefix for saved figure, e.g. new -> new_mixed_*.png",
    )
    parser.add_argument("--save", action="store_true", help="Save the figure(s) to disk (PNG + PDF)")
    parser.add_argument("--no_show", action="store_true", help="Do not open a window (useful for headless runs)")
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
    # Build an output prefix like "new_bimodal"
    out_prefix = f"{args.tag}_{args.expert_type}"

    plot_results_separate(
        results=results,
        analysis=analysis,
        collect_every=collect_every,
        out_prefix=out_prefix,
        outdir=args.outdir,
        show=(not args.no_show),
        save_png=args.save,
        save_pdf=args.save,
    )

    print("\nCoordination consistency:")
    coordination_consistency(results)
    print("\nDone.")


if __name__ == "__main__":
    main()