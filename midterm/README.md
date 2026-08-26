# Midterm Project Scripts — MetaUrban PointNav

## Overview

These scripts train and evaluate RL agents on the MetaUrban `SidewalkStaticMetaUrbanEnv` (PointNav task) for the midterm report.

| Script | Purpose |
|---|---|
| `env_config.py` | Shared environment + reward config (imported by all scripts) |
| `train_ppo.py` | Train PPO (on-policy policy gradient) |
| `train_sac.py` | Train SAC (off-policy Q-function-based RL) |
| `run_random.py` | Run random agent baseline (3 runs, report mean) |
| `evaluate.py` | Evaluate any trained model and report metrics |
| `plot_results.py` | Plot all learning curves on one figure |
| `run_ppo_live.py` | Run PPO in simulator with live 3D window (legacy, prefer `run_agent_live.py`) |
| `run_agent_live.py` | Run PPO or SAC in simulator with live 3D window |
| `record_video.py` | Record top-down bird's-eye videos for the project website |

## Step-by-Step Instructions

All commands should be run from the `midterm/` directory.

### Step 1: Random Agent Baseline

```bash
python run_random.py --num_runs 3 --episodes_per_run 50
```

This outputs `midterm_logs/Random/random_agent_results.json` with the mean return.

### Step 2: Train PPO (On-Policy Policy Gradient)

```bash
python train_ppo.py --total_timesteps 3000000 --n_envs 10 --seed 0
```

3M steps with 10 envs takes ~25 min on 16 CPUs and is enough for a meaningful learning curve.
The original `RL/PointNav/train_ppo.py` targets 1e8 steps; we use 3M for the midterm.
Logs and checkpoints go to `midterm_logs/PPO/`.

### Step 3: Train SAC (Off-Policy Q-Function RL)

```bash
python train_sac.py --total_timesteps 3000000 --n_envs 10 --seed 0
```

Logs and checkpoints go to `midterm_logs/SAC/`.

### Step 4: Plot Learning Curves

After training completes:

```bash
python plot_results.py
```

This produces `midterm_logs/learning_curves.png` with PPO, SAC, and the random baseline on the same figure (mean return vs. training environment steps).

### Step 5: Evaluate Trained Models

```bash
python evaluate.py --algo ppo --model_path ./midterm_logs/PPO/ppo_seed0/best_model/best_model.zip --episodes 50
python evaluate.py --algo sac --model_path ./midterm_logs/SAC/sac_seed0/best_model/best_model.zip --episodes 50
```

### Step 6: Watch Live or Record Videos

**Live viewing** (3D simulator window — requires a display):

```bash
python run_agent_live.py --algo ppo --model_path ./midterm_logs/PPO/ppo_seed1/best_model/best_model.zip --episodes 3
python run_agent_live.py --algo sac --model_path ./midterm_logs/SAC/sac_seed0/best_model/best_model.zip --episodes 3
```

**Record videos** (top-down bird's-eye view, works headless):

```bash
python record_video.py --algo random --episodes 3
python record_video.py --algo ppo --model_path ./midterm_logs/PPO/ppo_seed1/best_model/best_model.zip --episodes 3
python record_video.py --algo sac --model_path ./midterm_logs/SAC/sac_seed0/best_model/best_model.zip --episodes 4
```

Videos are saved to `midterm_logs/videos/`.

## Monitoring Training with TensorBoard

```bash
tensorboard --logdir ./midterm_logs
```

---

## Results & Analysis

### Learning Curves

![Learning Curves](midterm_logs/learning_curves.png)

Both methods are trained on `SidewalkStaticMetaUrbanEnv` (intersection map, static obstacles only) for up to 1.6M environment steps with 10 parallel environments. The random baseline (dashed gray) achieves a mean return of **6.39**.

#### PPO (On-Policy, seed 1)

PPO improves rapidly from the start, peaking at a mean return of **137.4** around **400k steps**. After this peak, performance collapses sharply — by 800k steps the mean return drops to near 0, then partially recovers to ~18 by the end of training. The 25th–75th percentile band is wide during the peak (8–218), reflecting high variance across evaluation episodes. At peak, 7 out of 10 evaluation episodes achieve returns above 50 (a rough success proxy).

The collapse pattern is consistent across seeds and mirrors the reference `pretrained_policy_576k`, which was also an early checkpoint from a 100M-step run — the reference solution experiences the same collapse after its early peak. The root cause is `no_negative_reward=True`: stopping yields a clamped reward of 0, while moving risks crash penalties (−2.0) that bypass the clamp. Once the policy gradient overshoots into a region that favors braking, the on-policy nature of PPO means it only collects "stopped" data from that point, making recovery difficult.

#### SAC (Off-Policy, seed 0)

SAC starts near zero (random exploration phase with `learning_starts=10,000`) and steadily climbs, reaching **490.5** at **1.59M steps**. By 820k steps all 10 evaluation episodes exceed a return of 50 (100% success), and performance remains stable through the end at **416.7**. The IQR band is much tighter than PPO's, indicating consistent performance across episodes.

SAC's advantage comes from three factors: (1) off-policy replay avoids the collapse loop — even if the current policy stops, the replay buffer retains earlier exploratory transitions; (2) the SAC training script uses `no_negative_reward=False` and higher reward scales (`success_reward=15`, `driving_reward=3`), making stopping costly and goal-reaching highly rewarding; (3) the `IdlePenaltyWrapper` adds a −0.1/step penalty when speed drops below 0.5 km/h, directly discouraging braking.

**Note:** The PPO and SAC return scales are not directly comparable because they use different reward configurations (PPO uses the default `env_config.py` with `no_negative_reward=True`; SAC overrides several reward weights).

#### Random Agent

The random agent achieves a mean return of **6.39 ± 1.69** over 150 episodes. Its success rate is 2–4%, from occasionally stumbling toward the goal. This serves as the floor for both methods.

### Task-Relevant Metrics

| Metric | Random | PPO (best, 400k) | PPO (final, 1.6M) | SAC (final, 1.6M) |
|---|---|---|---|---|
| Mean Return | 6.39 | 137.4 | 18.3 | 416.7 |
| Episodes > 50 return | 2–4% | 70% | 0% | 100% |
| Mean Episode Length | 944 | ~400 (fast success or fail) | ~1000 (timeout/idle) | ~300 (fast success) |

PPO's best checkpoint achieves a reasonable success rate (~70%), but the final model has regressed to near-random levels. SAC maintains high success throughout.

### Failure Case Analysis

From watching live rollouts (`run_agent_live.py`), we identified three primary failure modes:

1. **Going off-road:** The robot drifts laterally beyond `max_lateral_dist=5m` and the episode terminates. This typically happens when the agent overshoots a turn at the intersection. PPO's best model occasionally fails this way on sharp-angle scenarios. Penalty: `out_of_road_penalty=3.0`.

2. **Failing to turn (driving straight through the intersection):** In some scenarios, the goal is down a side street, but the policy drives straight through the intersection without steering. The robot either reaches the edge of the road and terminates, or continues until max steps. This is common in PPO's collapsed model — even when moving, the policy defaults to low-steering straight-line trajectories.

3. **Stopping/braking in front of obstacles:** The most common failure for PPO after collapse. The robot approaches a static obstacle (lamp post, tree, barrier), detects it via lidar, and applies full brake. With `no_negative_reward=True`, the clamped reward of 0 per step is "safer" than the −2.0 crash penalty for moving forward. The robot stays stopped until the 1000-step horizon ends. This is the dominant failure mode at evaluation and is the direct cause of the PPO collapse: once most rollouts are "stopped" data, PPO cannot explore its way out.

4. **Colliding with static objects:** Occasionally the robot drives into lamp posts, trees, or barriers, especially at speed. The crash penalty (−2.0) is applied, and in multi-collision scenarios the accumulated negative reward significantly hurts the return.

### Proposed Modifications

#### Modification 1: Resume from best checkpoint with lower learning rate

PPO collapses after its peak. We add `--resume_from` and `--resume_lr` flags to `train_ppo.py` so training can resume from `best_model` with a reduced learning rate (e.g., 1e-4 instead of 5e-4). Smaller gradient updates reduce the risk of overshooting into the braking local optimum, allowing the policy to fine-tune from its peak rather than destabilize.

#### Modification 2: Idle penalty wrapper for PPO

The stopping failure is caused by the asymmetry between the clamped-to-zero stop reward and the negative crash penalties. Following the approach that resolved the same issue for SAC, we propose wrapping the PPO environment with an `IdlePenaltyWrapper` that adds a −0.1 penalty per step when the robot's speed drops below 0.5 km/h. This makes stopping no longer "free," breaking the local optimum and encouraging the policy to keep moving. Combined with entropy regularization (`ent_coef > 0`), this should prevent the collapse pattern.

### Next Steps

1. **Apply idle penalty + entropy regularization to PPO** — add `IdlePenaltyWrapper` and set `ent_coef=0.005` (following the SocialNav reference), then retrain to verify the collapse is eliminated.
2. **Extended training budget** — increase PPO to 5–10M steps with the above fixes; the reference solution peaks at 576k out of 100M, suggesting that more steps alone help if the collapse is prevented.
3. **Hyperparameter tuning** — test lower LR (1e-4), lower `max_grad_norm` (0.5), and `no_negative_reward=False` for PPO, borrowing from the more stable SocialNav reference config.
4. **Dynamic environment** — transition from `SidewalkStaticMetaUrbanEnv` to `SidewalkDynamicMetaUrbanEnv` with pedestrians and other agents, which is the target for the final project.

---

## Output Structure

```
midterm_logs/
├── Random/
│   └── random_agent_results.json
├── PPO/
│   └── ppo_seed1/
│       ├── tb_logs/
│       ├── checkpoints/
│       ├── best_model/
│       └── eval_logs/evaluations.npz
├── SAC/
│   └── sac_seed0/
│       ├── tb_logs/
│       ├── checkpoints/
│       ├── best_model/
│       └── eval_logs/evaluations.npz
├── eval/
│   ├── ppo_eval_results.json
│   └── sac_eval_results.json
├── videos/
│   ├── random_episode_0.mp4
│   ├── ppo_episode_0.mp4
│   └── sac_episode_0.mp4
└── learning_curves.png
```
