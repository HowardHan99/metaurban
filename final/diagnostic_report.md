# MBPO on MetaUrban PointNav — Diagnostic Report

**Date:** 2026-04-17 to 2026-04-18
**Goal:** Train a Model-Based Policy Optimization agent (MBPO, Janner et al. 2019) on
`SidewalkStaticMetaUrbanEnv` to match or beat the midterm PPO/SAC baseline. Three
successive training runs were needed to get the agent past "does nothing." All
three ran on the same env config ([env_config.py](env_config.py)) with the
`MBPO_ENV_OVERRIDES` reward rebalancing that already worked for midterm SAC.

## TL;DR

| Run | Intervention | Final / peak behavior | Success rate | Verdict |
|---|---|---|---|---|
| #1 | Default MBPO (α auto-tune, `real_ratio=0.05`) | α ran to **103**, policy became pure noise | 0% at 500k | Algorithm bug |
| #2 | Clamp log α ∈ [log 1e-3, log 1]; `real_ratio=0.5` | α converged to 0.05, policy slid into "creep slowly" local optimum | 0% at 200k | Fix insufficient |
| #3 | Clamp log α ∈ [**log 0.15**, log 1] | Peak `mean_return=+210`, `q75=+326` at step 220k, no success | 0% at 380k+ | Best yet; still no success |

## Setup

- Env: same config as midterm (horizon 1000, `LidarStateObservation` = 31 state + 240 lidar → 271-D), 1000 scenarios, static.
- Reward overrides: `no_negative_reward=False`, `driving_reward=3.0`, `success_reward=15.0`, `speed_reward=1.0`, `lateral_penalty=0.5`, crash penalties 1.0, `out_of_road_penalty=2.0`, `steering_range_penalty=0.5`. Identical to the overrides that unblocked midterm SAC.
- MBPO: ensemble of 7 probabilistic MLPs (5 elites), 4 layers × 200 hidden, `rollout_length=1`, `model_train_freq=500`, `sac_updates_per_step=10`, `sac_batch_size=256`.
- SAC: our own minimal implementation ([sac_agent.py](sac_agent.py)) — squashed-Gaussian actor, twin-Q critic, auto-tuned temperature with target entropy = −act_dim = −2.0.

## Run #1 — α runaway (archived at `mbpo_seed0_run1_alpha_runaway/`)

**Config:** default MBPO — `real_ratio=0.05`, `init_temperature=0.1`, no α clamp.

**Result after 500k steps:** mean eval return −11 to −21 across last 20 evals. Episodes always hit horizon=1000. Vehicle doesn't move in the live viewer.

**Post-mortem probe of `ckpt_step_500000.pt`:** `log_alpha = 4.64 ⇒ α = 103.4`.

**Root cause:** positive-feedback blowup in the temperature dynamics.
With `real_ratio=0.05`, 95% of SAC's updates use next-state + reward from a
learned dynamics model that was trained on only a few thousand to a few
hundred thousand real transitions. Early predictions are noisy. SAC's Q
targets become garbage, so the only stable gradient into the actor is the
entropy bonus `α · log π(a|s)`. The actor starts producing broader action
distributions → `log π` exceeds `target_entropy` systematically → the
α update `α_loss = -log_α · (log π - H_target)` drives log α upward →
bigger α amplifies the entropy pressure → loop.

Terminal state: α = 103 means the entropy term of the actor loss dominates
the Q term by ~1000×. SAC is optimizing for action-distribution entropy and
ignoring reward. In the live viewer this manifests as twitchy near-zero-mean
motion — the deterministic actor output `tanh(μ)` has non-trivial magnitude
(≈0.9), but the trained Q values tell the actor nothing about what's good.

**Conclusion:** a latent failure mode of auto-tuned SAC in the presence of
heavily-model-based off-policy data, not present in standard SAC. Janner's
paper doesn't clamp α because their MuJoCo dynamics are learned well enough
that Q stays informative; our lidar+state task is noisier.

## Run #2 — α clamped, still idle (archived at `mbpo_seed0_run2_safe_idle/`)

**Interventions:**
1. Added `LOG_ALPHA_MIN = log(1e-3)`, `LOG_ALPHA_MAX = log(1.0)` with a
   `clamp_()` after each α optimizer step.
2. `real_ratio` default 0.05 → **0.5**. Gives SAC a 10× stronger real-data
   signal so Q estimates are less dominated by model noise.

**Result after 200k steps:** α remained bounded (0.0096 at step 50k, climbing
to 0.0564 at step 200k — never approached the 1.0 ceiling, clamp dormant).

But the evals show a *different* failure mode:

| step | mean_return | q25 / q75 | mean_length |
|---|---|---|---|
| 10k | +25.7 | (−4, +45) | 63 |
| 30k | +27.3 | (+15, +43) | 50 |
| 60k | +38.3 | (+9, +52) | 72 |
| 100k | −3.8 | (−4, −0.2) | **900** |
| 150k | −62.9 | (−9, −1.4) | **1000** |
| 200k | −14.5 | (−4, −1.8) | 900 |

Early in training (through step ~60k) the agent drove aggressively and
crashed; after 80k it slid into the "safe-idle" attractor. Returns hover
near zero, episodes last the full horizon, 0% success.

**Root cause:** the α clamp prevents the run-#1 runaway, but with `α ≈ 0.05`
the entropy pressure is essentially off. Once the critic punishes a few
crashes, the actor's gradient points toward "do nothing." Sitting still
yields reward ≈ 0 (tiny lane-line and steering-range penalties only) which
is locally optimal w.r.t. the crash-heavy neighborhood. SAC cannot escape
without enough exploration pressure.

**Conclusion:** α-auto-tune finds a floor below the level needed to keep
exploring a sparse-success task. The fix for the runaway was correct but
not sufficient.

## Run #3 — α floor raised to 0.15 (in progress at step 380k+)

**Intervention:** `LOG_ALPHA_MIN` bumped from `log(1e-3)` to `log(0.15)`.
`init_temperature` raised 0.1 → 0.15 to match the new floor.

This forces sustained exploration pressure even after the critic turns
pessimistic. Cost: the policy can never become purely deterministic at
inference — eval uses `deterministic=True` (takes `tanh(μ)`) but the Q
that shaped μ was trained against stochastic actions with fixed σ.

**Training curve (first 380k steps):**

| phase | step range | peak `mean_return` | peak `q75` | avg episode length |
|---|---|---|---|---|
| A: first emergence | 10k–60k | +89 (step 10k) | +113 | 67–135 |
| B: first "idle drift" | 70k–100k | +57 | +86 | 115–821 |
| C: big breakthrough | 110k–130k | **+124** (step 120k) | **+188** | 100–251 |
| D: idle relapse | 140k–210k | +6 | +24 | 582–1000 |
| E: second breakthrough | 220k | **+210.7** | **+326** | 156 |
| F: post-peak regression | 230k–380k | +75 | +118 | 412–917 |

**Key observations:**

1. **α pinned at 0.15** through all 8 checkpoints (50k → 350k). Adam's
   gradient wants α *lower* (policy more deterministic than target-entropy
   would prefer); the clamp is the only thing keeping exploration alive.
   This confirms: without the floor, α would collapse to ~0.05 and the run
   would match run #2.

2. **Peak returns oscillate with ~40–50k-step period.** The agent
   repeatedly discovers aggressive driving, gets punished by crashes, the
   critic depresses Q values, actor reverts to caution, α-forced noise
   eventually stumbles back into aggressive driving. This *is* learning,
   but it's unstable because the "aggressive" peak is a narrow ridge.

3. **Ceiling grew in first half, dropped in second.** Phase-C peak
   `+124 → q75=+188`; phase-E peak `+210 → q75=+326`. But phase-F
   (steps 250k–380k) has mean peaks of only +75 and q75 not exceeding
   +118. The oscillation is degrading, not converging.

4. **Success rate is 0% at every eval through step 380k**, despite
   `q75=+326` at step 220k. This is the most important diagnostic.
   Return +326 with episode length 156 means the agent:
   - Drove fast for 156 steps (`speed_reward=1` × ~0.5 per step ≈ +78)
   - Accumulated massive `driving_reward=3 × Δprogress` (≈ +240)
   - Ended without `arrive_dest=True` — i.e., went out-of-road or
     otherwise terminated.

   **The agent can score 326 without reaching the destination.** It
   hasn't learned the precise "final approach" needed to trigger
   `arrive_dest`. There's no reward signal intermediate between
   "drive forward fast" and the single +15 success bonus to scaffold
   that final skill.

## Root-cause analysis across all three runs

Three *orthogonal* failure modes manifested in sequence:

1. **Temperature runaway** (run #1) — solved by α clamp. Will not
   re-emerge with current code.
2. **Safe-idle local optimum** (run #2) — solved by raising α floor to 0.15.
   Will not re-emerge if floor is kept.
3. **Sparse-success credit-assignment gap** (run #3) — **unsolved**. The
   reward landscape provides dense feedback for "drive fast anywhere" and
   a single +15 bonus for being at the exact goal location, with nothing
   in between. MBPO's model-based rollouts (k=1) can't bridge this gap
   either — they're too short to propagate the success signal backward
   meaningfully, and the model has never seen a successful trajectory to
   learn from.

The algorithm is now correct. The *task shaping* is what's missing.

## Next steps

### Immediate: extend run #3 to 1.6M steps

Match the midterm SAC compute budget (1.63M steps) so the curves are
directly comparable. Will resume from `ckpt_step_500000.pt` when the
current 500k run completes. If return ceiling doesn't climb or
`success_rate` stays 0% for another ~500k, it confirms the credit-
assignment bottleneck — not a question of more compute.

### Then: reward shaping experiment

Add a dense goal-proximity reward that fills the gap between
`driving_reward` (goes to zero as Δprogress → 0 near goal) and
`success_reward` (one-shot +15 on `arrive_dest`). Concretely something
like `+k / (1 + d_to_goal)` per step. This turns the final approach into
a dense signal SAC can gradient-descend into, without needing the agent
to stochastically luck into `arrive_dest` in order to start learning it.

Implementation options:
- **Modify [env_config.py](env_config.py)** to add a custom reward hook
  (cleanest; keeps the change isolated to the reward function).
- **Wrap the env** with a `GoalProximityRewardWrapper` around the
  existing `IdlePenaltyWrapper` (even more isolated; no env-config
  surgery).

Comparison plan: same α floor, `real_ratio=0.5`, same 1.6M budget, with
the new reward term. If `success_rate` crosses 0% we've validated the
hypothesis that the gap was shaping, not algorithm.

## Appendix — probe commands used during diagnosis

```bash
# Check alpha across checkpoints
python -c "
import torch
for s in [50000, 100000, 200000, 500000]:
    c = torch.load(f'final_logs/MBPO/mbpo_seed0/checkpoints/ckpt_step_{s}.pt',
                   map_location='cpu', weights_only=False)
    la = c['sac']['log_alpha']
    print(f'step={s:>7}  alpha={la.exp().item():.4f}')
"

# Live 3D viewer on latest checkpoint
python run_agent_live.py --checkpoint ./final_logs/MBPO/mbpo_seed0/checkpoints/ckpt_latest.pt --episodes 3

# Stochastic eval (samples from π instead of deterministic mean)
python run_agent_live.py --checkpoint ./final_logs/MBPO/mbpo_seed0/checkpoints/ckpt_latest.pt --episodes 3 --stochastic
```
