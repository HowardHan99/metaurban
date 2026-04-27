# MBPO Run #6 — Operator's Notes

Active training run for MBPO on MetaUrban PointNav. This file is loaded by
Claude Code at session start so the assistant knows the current state.

## Run #6 snapshot

| field | value |
|---|---|
| pid | 557072 (restarted 22:13 after earlier 557072 died near step 201,500 with no traceback) |
| launched | 2026-04-18 ~22:13 local |
| resumed from | step 200000 (run #5 archive) |
| target | **12h wall time** — graceful stop at 2026-04-19 10:13 local via SIGTERM (expected ~step 920k; `--total_timesteps 1500000` won't bind). Scheduler pid=561310 (`nohup sleep && kill -TERM 557072`), log `/tmp/mbpo_run6_autostop.log`. |
| α floor | 0.25 (raised from run #5's 0.15) |
| shaping | `+20 · Δrc` (potential-based, unchanged from run #5) |
| logs dir | `./final_logs/MBPO/mbpo_seed0/` |
| stdout | `/tmp/mbpo_run6.log` |
| ETA | capped at 12h (see target row) — ~20–22h would be needed to reach 1.5M at run #5 throughput |

## Why these settings

- α floor 0.25: run #5 (floor=0.15) fell into a persistent idle attractor at
  step 210k+. α auto-tune then climbed from 0.15 → 0.63 by step 250k as it
  tried to push entropy back up. Raising the floor forces sustained
  exploration pressure before the collapse.
- Potential-based Δrc shaping (coeff=20): preserved — it stabilized the
  pre-collapse phase of run #5 (peak mean +150 at step 190k).
- Resumed from step 200k (peak region) rather than restart to reuse the
  dynamics model + replay buffers.

## Monitor progress

```bash
# is it alive?
ps -p 557072 -o pid,etime,%cpu,cmd

# latest evals
tail -f /home/howardhan/metaurban/final/final_logs/MBPO/mbpo_seed0/eval_log.jsonl

# key metric: any success_rate > 0
grep -o '"success_rate": [^,]*' \
  /home/howardhan/metaurban/final/final_logs/MBPO/mbpo_seed0/eval_log.jsonl \
  | sort -u

# α trajectory (run after each 50k checkpoint)
cd /home/howardhan/metaurban/final && python -c "
import torch, glob
for p in sorted(glob.glob('final_logs/MBPO/mbpo_seed0/checkpoints/ckpt_step_*.pt')):
    c = torch.load(p, map_location='cpu', weights_only=False)
    print(p.split('/')[-1], 'alpha=', c['sac']['log_alpha'].exp().item())"
```

## Stop / resume

```bash
# graceful stop (uses last 50k checkpoint)
kill -TERM 557072

# resume from latest checkpoint
cd /home/howardhan/metaurban/final
python train_mbpo.py \
  --resume_from ./final_logs/MBPO/mbpo_seed0/checkpoints/ckpt_latest.pt \
  --total_timesteps 1500000 \
  --seed 0 \
  --goal_proximity_coeff 20.0
```

Run is nohupped, so it survives terminal close. If the machine restarts,
relaunch from `ckpt_latest.pt` with the resume command.

## What to watch for

- **First eval** at step 210k (~10 min after launch). If α at floor 0.25 and
  mean_return not already idle → intervention worked.
- **Milestone**: `success_rate > 0` — first ever across all 6 runs.
- **Failure signal**: `mean_length=1000` across consecutive evals = idle
  attractor reforming. If it happens, kill and try one of:
  - Option B: distance-to-goal shaping `+k / (1 + d_to_goal)` instead of Δrc
  - Option C: imitation bootstrap from midterm PPO/SAC successful trajectories

## Run history (archived under `./final_logs/MBPO/`)

| dir | issue |
|---|---|
| `mbpo_seed0_run1_alpha_runaway/` | α → 103, pure-noise policy |
| `mbpo_seed0_run2_safe_idle/` | α auto-tuned to 0.05, creep-slowly local opt |
| `mbpo_seed0_run3_alpha_floor_0.15/` | oscillating, peak +210 at 220k, 0% success |
| `mbpo_seed0_run4_quadratic_rc_backfire/` | mid-rc cruising exploit |
| `mbpo_seed0_run5_idle_collapse/` | step 200k peak then idle attractor at 210k+ |
| `mbpo_seed0/` | **run #6 active** |

Full diagnostic: [diagnostic_report.md](diagnostic_report.md)

## Claude Code permissions (skip the accept/deny prompts)

All commands in this file are **read-only or reversible** (ps, tail, grep,
checkpoint reads, SIGTERM, relaunch). Running Claude Code with auto-approval
for these is safe.

### Option 1 — launch in "accept edits" mode

Safest option. Auto-approves file edits and non-destructive bash; still
prompts for anything it judges risky (rm -rf, git push, etc.).

```bash
cd /home/howardhan/metaurban/final
claude --permission-mode acceptEdits
```

### Option 2 — fully bypass prompts (this session only)

Use when you want zero prompts. Does not persist across sessions.

```bash
claude --dangerously-skip-permissions
```

### Option 3 — persistent per-command allowlist (recommended)

Create `.claude/settings.local.json` in the project root. Claude Code reads
it at launch and auto-approves only the listed commands. Everything else
still prompts.

```bash
mkdir -p /home/howardhan/metaurban/final/.claude
cat > /home/howardhan/metaurban/final/.claude/settings.local.json <<'EOF'
{
  "permissions": {
    "allow": [
      "Bash(ps:*)",
      "Bash(tail:*)",
      "Bash(cat:*)",
      "Bash(grep:*)",
      "Bash(wc:*)",
      "Bash(ls:*)",
      "Bash(pgrep:*)",
      "Bash(kill:*)",
      "Bash(python -c:*)",
      "Bash(python train_mbpo.py:*)",
      "Bash(nohup python:*)",
      "Bash(mv:*)",
      "Bash(ln:*)",
      "Bash(mkdir:*)",
      "Read",
      "Edit",
      "Write",
      "Glob",
      "Grep"
    ]
  }
}
EOF
```

The `Bash(cmd:*)` pattern allows any invocation starting with that command.
Still prompts for `rm`, `git push --force`, `sudo`, etc. Tune by adding
entries to the `allow` list as new commands come up.

### Switching mode mid-session

Type `/permissions` inside Claude Code to inspect/change rules, or edit
`.claude/settings.local.json` and relaunch.
