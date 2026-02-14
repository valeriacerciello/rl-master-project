# On the Role of Entropy and Best-Response Uniqueness in MAGAIL

## Overview

This repository contains controlled experiments studying **when occupancy matching identifies an expert equilibrium in MAGAIL**. Existing recovery results rely on strong assumptions that often fail in multi-agent games. We isolate two key issues:

1. **Entropy regularization (β > 0) changes the objective** and can bias learning away from correlated expert behavior under factorized policies, even if it stabilizes training.
2. **Best responses may be non-unique even for a fixed opponent profile**, so occupancy matching can be non-identifying: policies can match expert demonstrations on-support yet differ strategically (e.g., be exploitable off-support).

All experiments are tabular and fully reproducible.

---

## Weakness #1 — Entropy regularization changes the objective

### Theoretical background

The strongest recovery/identification statement in MAGAIL is proved for the **unregularized** setting (β = 0). In practice, however, entropy bonuses (β > 0) are commonly used to stabilize policy optimization. In multi-agent settings this is not an innocuous modification: the entropy term is evaluated under the **learner–learner interaction distribution**, and under **factorized per-agent policies** it can bias learning toward high-entropy independent solutions. As a result, the optimization problem solved in practice can differ qualitatively from the unregularized occupancy-matching objective.

### Empirical demonstration: coordination environment

We illustrate this effect in a minimal 2×2 coordination game. The experiment highlights two distinct phenomena and how entropy changes the outcome:

- **Fail 1 — unstable selection among equally optimal responses (β = 0):**  
  when multiple equally good conventions exist, removing entropy leads to seed-dependent training dynamics: different runs converge to different conventions despite identical expert data.

- **Fail 2 — correlated-demo mismatch under factorized policies:**  
  when expert demonstrations are **correlated** (e.g., only AA/BB), independent per-agent policies cannot represent the correlation exactly. Adding entropy further pushes toward the maximally mixed *independent* solution (marginals near 0.5/0.5), which necessarily assigns probability mass to AB/BA, so the learned joint distribution cannot match the demonstrations.

Overall, increasing β improves reproducibility by reducing across-seed variance, but it does so by changing the effective objective and (in correlated-demo regimes) by favoring high-entropy independent behavior rather than reproducing correlated expert play.

---

## Weakness #2 — Non-unique best responses make occupancy matching non-identifying

### Theoretical background

MAGAIL’s recovery/identification argument relies on a **uniqueness condition**: for each agent \(i\), the expert policy \(\pi_i^E\) is the *unique* optimal response to the other experts’ profile \(\pi_{-i}^E\) (equivalently, the multi-agent RL solution mapping is single-valued on a relevant reward set). This is a strong requirement in multi-agent games: even for a **fixed** opponent strategy profile, the best-response set \(\mathrm{BR}_i(\pi_{-i})\) can contain multiple policies due to symmetry, indifference, or off-support freedom.

When best responses are non-unique, **occupancy matching can become non-identifying**: different policies may agree with the expert on the demonstrated (on-support) behavior while differing elsewhere. Since strategic optimality is defined by **counterfactual unilateral deviations** (best responses), matching the expert occupancy on its support does not in general guarantee strategic equivalence (e.g., low exploitability / small Nash-Gap).

### Empirical demonstration: exploitability under off-support deviations

We demonstrate this in a minimal two-player zero-sum Markov game designed so that expert demonstrations concentrate on a “safe” region of the state space. The expert profile \(\pi^E\) is a Nash equilibrium, but unilateral deviations can reach an unobserved region where correct defensive play matters. Because this region is never visited in the demonstrations, an offline occupancy-matching objective provides no signal there.

Empirically, MAGAIL (and a BC baseline) can closely match the expert behavior on demonstrated trajectories while remaining **highly exploitable**, as measured by a positive Nash-Gap. This illustrates the core limitation: without coverage of states reached by unilateral best-response deviations, occupancy matching alone is insufficient to recover strategically robust equilibria when best responses are not uniquely pinned down by the data.

---

## Code Structure

```
rl-master-project/
│
├── envs/                      # Minimal tabular Markov games (coordination, zero-sum)
│   ├── __init__.py
│   ├── entropy_coordination.py
│   └── zero_sum.py
│
├── runners/                   # Reproducible experiment entrypoints (generate paper figures)
│   ├── entropy_coordination_runner.py
│   └── zero_sum_runner.py
│
├── MAGAIL.py                  # MAGAIL implementation (training loop + logging)
├── BC.py                      # Behavior Cloning baseline
├── calc_exploitability.py     # Exploitability / Nash-gap computation utilities
└── README.md                  # Documentation + reproduction commands
```

Notes:
- `plots/` contains generated figures and can be regenerated by running the scripts in `runners/`.

---

## Requirements

- Python 3.9+ (tested with Python 3.13)
- PyTorch
- NumPy
- Matplotlib
- SciPy

Install dependencies:

```bash
pip install torch numpy matplotlib scipy
```

## Running Experiments

All scripts are deterministic w.r.t. the provided `--seeds` (tabular setting; no GPU required).

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

Outputs are written to `plots/entropy_exp/` with filenames of the form:
- `{tag}_{expert_type}.png`
- `{tag}_{expert_type}.pdf`

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

### Zero-sum exploitability experiment

Run:

```bash
python runners/zero_sum_runner.py
```

This script:
1.	generates an expert dataset from a fixed expert profile,
2.	trains MAGAIL (β = 0) and a Behavior Cloning (BC) baseline on the same offline data,
3.	computes exploitability (Nash-gap style) via best responses in the induced single-agent MDPs, and
4.	writes a comparison plot to `plots/exploitability_exp/`.

Defaults:

* `expert_action = 2` (middle action in 3-action game)
* `expert_total = 1000` (expert samples)
* `magail_seeds = [42, 123, 456]`
* `bc_seeds = [42, 123, 456]`
* `num_epochs = 1000`
* `rollout_episodes = 200`
* `batch_size = 128`
* `lr_policy = 0.01` `lr_disc = 0.01`
* `beta = 0.0` (zero entropy for zero-sum games)
* `gamma = 0.9` (discount factor)
* `reward_style = non_saturating` (uses log D)

Note: this experiment currently has no CLI arguments; to change hyperparameters, edit `runners/zero_sum_runner.py`.

