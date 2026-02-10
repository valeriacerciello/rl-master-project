# On the Role of Entropy and Best-Response Uniqueness in MAGAIL

## Overview

This repository contains a set of controlled experiments designed to reveal and analyze **structural weaknesses in the Multi-Agent Generative Adversarial Imitation Learning (MAGAIL)** framework. Although MAGAIL extends GAIL to multi-agent settings, its theoretical guarantees rely on assumptions that seldom hold in realistic environments. Our experiments demonstrate two critical weaknesses:

1. **MAGAIL implicitly relies on the absence of entropy regularization in multi-agent learning**
2. **MAGAIL assumes the existence of *unique* best responses**

This repository provides reproducible environments, training loops, and evaluation tools illustrating these failures in minimal settings.

---

## Weakness #1 — MAGAIL Assumes No Entropy Regularization

### Theoretical Background

MAGAIL's theoretical convergence guarantees assume $\beta=0$ (no entropy regularization). However, standard practice uses $\beta>0$ to stabilize training, which fundamentally changes the optimization landscape: entropy bonuses introduce coordination failures between independently trained agents even when they perfectly match individual occupancy measures.

### Empirical Demonstration: Coordination Environment

To illustrate this weakness, we construct a tiny, reproducible experiment that shows two ways MAGAIL can “fail” in a simple 2×2 coordination game—and how an entropy bonus changes the behavior.

* **Fail 1 — Symmetry/selection failure (β = 0):** with two equally-good pure NE (AA and BB), vanilla MAGAIL arbitrarily collapses to one convention depending on randomness.
* **Fail 2 — Correlated-demo mismatch:** when expert demonstrations are **correlated** (only AA/BB), MAGAIL with **independent** agent policies can’t represent that correlation. Even with entropy, it can only produce an **independent** 50/50 mix (which necessarily places mass on AB/BA), so the joint distribution never matches the demos.

With a sufficiently large entropy bonus β, both agents converge to the **max-entropy mix** (0.5/0.5) and training stabilizes across seeds.

---

## Weakness #2 — MAGAIL Assumes Unique Best Responses

### The™

Below Corollary 5, the MAGAIL authors assume that for each agent $i$, the expert policy $\pi_{i_E}$ is the **unique** optimal response to the other experts' policies. However, this assumption fails in many realistic settings where multiple equally optimal responses exist due to symmetry or payoff indifference.

When best responses are non-unique, occupancy measure matching becomes insufficient: a learner can perfectly match the expert's occupancy distribution while adopting a completely different (and highly exploitable) strategy. This reveals a fundamental limitation: **occupancy matching does not guarantee strategic equivalence in games with multiple best responses.**

### Empirical Demonstration: Multi-Response Exploitability

To illustrate this weakness, we construct a tiny, reproducible experiment that shows how MAGAIL can produce highly exploitable policies in a simple two-agent zero-sum game with multiple best responses.

In this environment, the expert policies form a Nash equilibrium, but each agent has multiple equally optimal responses to the other's strategy. When trained with MAGAIL, the learned policies match the expert occupancy measures but deviate significantly in strategic behavior, leading to high exploitability.

Our experiments demonstrate that even with perfect recovery of the expert's state visitation distribution and complete knowledge of the transition model, MAGAIL produces policies with large Nash gaps. This reveals a fundamental limitation: **occupancy measure matching is insufficient for recovering Nash equilibria when multiple best responses exist.**

---

## Code Structure

```
rl-master-project/
│
├── envs/                      # Minimal multi-agent environments
│   ├── __init__.py
│   ├── entropy_coordination.py
│   └── zero_sum.py
│
├── runners/                   # Training and experiment scripts
│   ├── entropy_coordination_runner.py
│   └── zero_sum_runner.py
│
├── MAGAIL.py                  # Multi-Agent GAIL implementation
├── BC.py                      # Behavior Cloning baseline
├── calc_exploitability.py     # Exploitability analysis utilities
└── README.md                  # Project documentation
```

---

## Requirements

* Python 3.9+
* PyTorch
* NumPy
* Matplotlib
* SciPy

Install dependencies:

```bash
pip install torch numpy matplotlib scipy
```

## Running Experiments

### Entropy Coordination Experiment:

Run: 

```bash
python runners/entropy_coordination_runner.py
```

Reproduce paper figures (save + headless):

```bash
python runners/entropy_coordination_runner.py --expert_type mixed   --save --no_show
python runners/entropy_coordination_runner.py --expert_type noisy   --save --no_show
python runners/entropy_coordination_runner.py --expert_type bimodal --save --no_show
python runners/entropy_coordination_runner.py --expert_type all_AA  --save --no_show
```

Figures are saved to plots/entropy_exp/ with names like new_mixed.png and new_mixed.pdf.

#### Expert types:

Used in the report:

* `mixed`   : uniformly mixed expert (report wording: “uniformly mixed”)
* `noisy`   : mixed but non-uniform expert (report wording: “mixed but non-uniform / noisy”)
* `bimodal` : bimodal expert
* `all_AA`  : single-mode expert (AA only) (report wording: “single-mode only AA”)

Extra (not used in the report):

* `asymmetric` : additional expert distribution included for exploratory analysis / ablations (not referenced in the report figures)

Defaults:

* `--expert_type bimodal` (50% AA, 50% BB, zero AB/BA)
* `--betas 0.0 0.1 0.5 1.0 5.0`
* `--reward_style non_saturating` (`log D`)
* `--lr_policy 0.01` `--lr_disc 0.01`
* `--epochs 400` `--seeds 42 123 456 789 999`

You’ll see training logs, plots, and a “coordination consistency” summary.

CLI options:

```text
--expert_type {mixed,bimodal,asymmetric,noisy,all_AA}   default: bimodal
--betas <floats...>                                    default: 0.0 0.1 0.5 1.0 5.0
--seeds <ints...>                                      default: 42 123 456 789 999
--epochs <int>                                         default: 400
--rollout_episodes <int>                               default: 200
--eval_episodes <int>                                  default: 5000
--lr_policy <float>                                    default: 0.01
--lr_disc <float>                                      default: 0.01
--reward_style {non_saturating,gail}                    default: non_saturating
--policy_init_uniform                                  start both policies at 0.5/0.5
--batch_size <int>                                     default: 64
--collect_every <int>                                  default: 10
--outdir <path>                                        default: plots/entropy_exp
--tag <str>                                            default: new
--save                                                 save figure(s) to disk
--no_show                                              do not open a window (headless)
```

---

### Zero-Sum Exploitability Experiment:

```bash
python runners/zero_sum_runner.py
```

Defaults:

* `expert_action 2` (middle action in 3-action game)
* `expert_total 1000` (expert samples)
* `magail_seeds 42 123 456`
* `bc_seeds 42 123 456`
* `num_epochs 1000`
* `rollout_episodes 200`
* `batch_size 128`
* `lr_policy 0.01` `lr_disc 0.01`
* `beta 0.0` (zero entropy for zero-sum games)
* `gamma 0.9` (discount factor)
* `reward_style non_saturating`

The script trains both MAGAIL and BC baselines, computes exploitability curves, and generates comparison plots.

**Note:** This experiment has no CLI arguments.

