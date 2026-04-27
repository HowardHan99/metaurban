# Final Project — Model-Based RL on MetaUrban PointNav

Model-Based Policy Optimization (MBPO, Janner et al. NeurIPS 2019) on the same
`SidewalkStaticMetaUrbanEnv` used for the midterm. MBPO learns a probabilistic
ensemble dynamics model from real transitions, then augments a SAC agent's
replay buffer with short k-step rollouts unrolled *inside* the learned model.
SAC updates sample mixed batches from real + synthetic buffers (≈5% real).

Why MBPO here: it is model-based RL that works natively on low-dim state+lidar
observations (no need for images, unlike Dreamer); the learned dynamics model
makes it roughly an order of magnitude more sample-efficient than the midterm
PPO/SAC, which gives us a fair shot at good performance in 500k real env steps.

## Files

| Script | Purpose |
|---|---|
| `env_config.py` | Shared env config (mirrors midterm) + `MBPO_ENV_OVERRIDES` reward rebalancing |
| `sac_agent.py` | Self-contained SAC (actor, twin-Q, learnable temperature) — clean `state_dict()` for checkpoints |
| `train_mbpo.py` | MBPO training loop with **resume-from-any-checkpoint** support |
| `evaluate.py` | Evaluate an MBPO checkpoint |
| `run_random.py` | Random-agent baseline under MBPO reward config (so the baseline is comparable) |
| `plot_results.py` | MBPO learning curve; optionally overlays midterm PPO/SAC |

## Setup

One-time install of the model-based RL library (if not already done):

```bash
pip install mbrl
# If it downgrades cloudpickle to 1.3.0, restore it:
pip install 'cloudpickle>=2.0'
```

mbrl 0.1.5 pulls in old `gym` 0.17.2 as a transitive dep; it coexists fine with
MetaUrban's gymnasium — we never call into old-gym from this project.

## Training

### Default 500k-step run

```bash
cd final
python train_mbpo.py --total_timesteps 500000 --seed 0
```

With ensemble of 7 × Gaussian-MLP (4 layers × 200 hidden), dynamics retrained
every 500 env steps, 10 SAC updates/step, rollout length k=1, a 500k-step run
takes roughly 6–10 hours on a single GPU (MBPO's SAC-update loop dominates).

Logs and checkpoints go to `./final_logs/MBPO/mbpo_seed0/`:
- `eval_log.jsonl` — one JSON line per eval (step, mean_return, q25/q75, success_rate, …)
- `checkpoints/ckpt_step_*.pt` — full-state checkpoints every 50k steps
- `checkpoints/ckpt_latest.pt` — symlink to most recent checkpoint
- `checkpoints/buffers_step_*/` — replay buffer side-cars (loaded automatically on resume)

### Resume from any checkpoint

A checkpoint contains everything needed to keep training without drift: dynamics
weights + normalizer stats, SAC actor/critic/target/optimizers/log_alpha, **both
replay buffers**, and RNG state.

```bash
python train_mbpo.py \
  --resume_from ./final_logs/MBPO/mbpo_seed0/checkpoints/ckpt_step_200000.pt \
  --total_timesteps 500000
```

To train *beyond* the original budget, just bump `--total_timesteps`:

```bash
python train_mbpo.py \
  --resume_from ./final_logs/MBPO/mbpo_seed0/checkpoints/ckpt_latest.pt \
  --total_timesteps 1000000
```

### Custom hyperparameters

All MBPO knobs exposed as CLI flags — see `python train_mbpo.py --help`. The
ones most worth tuning:

| Flag | Default | Effect |
|---|---|---|
| `--rollout_length` | 1 | Longer = more synthetic data but compounding model error. Janner's paper schedules 1→15. |
| `--sac_updates_per_step` | 10 | More = better SAC but slower. Paper uses 20–40. |
| `--model_train_freq` | 500 | How often to retrain dynamics (env steps). Lower = fresher model. |
| `--rollout_batch_size` | 10_000 | Size of synthetic rollout batch per retrain. |
| `--real_ratio` | 0.5 | Fraction of real (vs. synthetic) transitions in SAC batches. Janner uses 0.05; we bumped to 0.5 after a 500k-step run where 95% synthetic data drove α into a runaway. |
| `--ensemble_size` / `--num_elites` | 7 / 5 | Ensemble uncertainty; elites = best-validation members used for rollouts. |

## Evaluation

```bash
python evaluate.py \
  --checkpoint ./final_logs/MBPO/mbpo_seed0/checkpoints/ckpt_latest.pt \
  --episodes 50
```

Prints the same metric panel as `midterm/evaluate.py` (mean return, success
rate, crash rate, out-of-road rate, mean episode length) and writes a JSON.

## Random baseline

```bash
python run_random.py --num_runs 3 --episodes_per_run 50
```

Unlike `midterm/run_random.py`, this version uses `MBPO_ENV_OVERRIDES` so the
random baseline return is directly comparable to MBPO's eval returns.

## Plotting

```bash
python plot_results.py                           # MBPO only
python plot_results.py --include_midterm         # also overlays midterm PPO/SAC
```

Output: `final_logs/learning_curves.png`, `final_logs/success_rates.png`.

## MBPO loop reference

Adapted from `mbrl.algorithms.mbpo` (which is incompatible with gymnasium envs
out of the box) using mbrl's lower-level building blocks:

1. **Seed real buffer** — `--init_steps` random transitions.
2. **Every env step:**
   1. Act with current SAC policy, step real env, append to real buffer.
   2. If `step % model_train_freq == 0`: update normalizer on real buffer,
      retrain ensemble dynamics (early stopping on val loss), then unroll
      `rollout_batch_size` × `rollout_length` synthetic transitions via
      `ModelEnv` into the synthetic buffer.
   3. Run `sac_updates_per_step` SAC updates on mixed real+synthetic batches.
3. **Eval / checkpoint** on configured cadence.

Termination inside the learned-model rollouts uses a no-op termination function
(we can't recover MetaUrban's crash/out-of-road flags from obs alone). This is
safe as long as `rollout_length` stays short (1–5).
