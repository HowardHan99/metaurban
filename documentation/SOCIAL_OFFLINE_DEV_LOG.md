# Social Offline Development Log

Owner: minghao
Branch: minghao-dev
Project: MetaUrban social offline reward learning

## 1) Goal
Build a complete offline learning pipeline for social navigation:
1. Generate socially rich simulation scenes.
2. Export replay clips (4-8s sliding windows).
3. Use LLM to annotate social issues.
4. Convert annotations into reward labels.
5. Train offline RL (IQL/CQL/TD3+BC) with social shaping.

---

## 2) Current System Snapshot

### Implemented modules
- `metaurban/social_reward/taxonomy.py`
  - 8 social issue labels, severity/confidence levels, schema, prompt, validation, penalty computation.
- `metaurban/social_reward/dataset_collector.py`
  - Episode recording, clip extraction, trajectory/frame export.
- `metaurban/social_reward/collect_dataset.py`
  - CLI for rollout collection + clip slicing.
- `metaurban/manager/social_scenario_manager.py`
  - Socialized pedestrian behavior manager (static bystanders, linger, group walk, ego-yield zone).
- `metaurban/envs/social_dynamic_env.py`
  - Environment wrapper that injects social scenario manager.
- `metaurban/social_reward/llm_annotator.py`
  - LLM clip annotator (openai/google/mock), validation, social penalty output.
- `metaurban/social_reward/annotate_clips.py`
  - CLI for batch annotation and reward write-back.

### Output artifacts
- `dataset/episodes/*.npz`
- `dataset/clips/*.npz`
- `dataset/annotations/*.json`
- Reward fields written back into clip files:
  - `social_penalty`
  - `social_present_mask`

---

## 3) Milestone Timeline

### 2026-03-15
- WSLg and runtime stack verified.
- MetaUrban run issues resolved (dependencies, ORCA build, assets).
- Social taxonomy implemented and cleaned to English comments only.

### 2026-03-16
- Offline dataset collection pipeline implemented.
- Social scene richness module added (`social_scenario_manager.py`).
- LLM annotation pipeline implemented (`llm_annotator.py`, `annotate_clips.py`).
- Smoke tests passed for:
  - clip extraction,
  - mock annotation,
  - reward write-back.

---

## 4) Experiment Log (append new rows)

| Date | Exp ID | Config Summary | Dataset Size | Label Noise Check | Offline RL Algo | Main Metrics | Result | Notes |
|---|---|---|---:|---|---|---|---|---|
| 2026-03-16 | E001 | mock annotation dry-run | 1 clip smoke test | pass | N/A | pipeline pass | success | end-to-end toolchain validated |

Recommended metric columns:
- Safety: collision rate, human-near-miss rate, min-distance distribution.
- Social: per-label frequency, weighted social penalty, intervention count.
- Task: success rate, route completion, episode return.

---

## 5) Active TODO

### Pipeline
- [x] Build social taxonomy.
- [x] Build dataset collector and clipper.
- [x] Build LLM annotation CLI.
- [x] Write social penalty into clips.
- [ ] Build offline RL trainer script (`train_offline_rl.py`).
- [ ] Add train/val/test split tool for clip dataset.
- [ ] Add annotation quality audit tool (confidence histogram + rule violations).
- [ ] Add benchmark report script for model comparison.

### Data quality
- [ ] Add automatic outlier detection for unrealistic clips.
- [ ] Add hard negative mining (high-risk but non-collision samples).
- [ ] Add manual review queue for low-confidence labels.

---

## 6) Known Risks / Issues

| ID | Severity | Issue | Status | Mitigation |
|---|---|---|---|---|
| R1 | High | LLM visual labels may be noisy in occlusion scenes | Open | confidence threshold + human review for low confidence |
| R2 | Medium | Social scene diversity still limited by map/traffic sampling | Open | increase scenario randomization and behavior composition |
| R3 | Medium | Reward over-penalization may hurt task completion | Open | lambda sweep and Pareto evaluation |
| R4 | Medium | Offline distribution shift at policy deployment | Open | conservative algorithm (IQL/CQL) + OOD checks |

---

## 7) Runbook (quick commands)

### 7.1 Collect dataset
```bash
conda run -n metaurban python metaurban/social_reward/collect_dataset.py \
  --num-episodes 200 \
  --out-dir dataset \
  --policy idm \
  --clip-len 6.0 \
  --stride 3.0 \
  --capture-rgb
```

### 7.2 Annotate clips
```bash
# mock backend (debug)
conda run -n metaurban python metaurban/social_reward/annotate_clips.py \
  --clips-dir dataset/clips \
  --backend mock \
  --out-dir dataset/annotations

# openai backend (real)
OPENAI_API_KEY=xxx conda run -n metaurban python metaurban/social_reward/annotate_clips.py \
  --clips-dir dataset/clips \
  --backend openai \
  --model gpt-4o \
  --out-dir dataset/annotations
```

### 7.3 Write rewards back to clips
```bash
conda run -n metaurban python metaurban/social_reward/annotate_clips.py \
  --write-rewards \
  --clips-dir dataset/clips \
  --annotations-dir dataset/annotations
```

---

## 8) Change Log Template (copy block)

```md
### YYYY-MM-DD
- What changed:
  -
- Why:
  -
- Validation:
  -
- Impact:
  -
- Next:
  -
```

---

## 9) Next Suggested Milestone

M2: Offline RL training baseline
1. Implement `train_offline_rl.py` with IQL baseline.
2. Load clips with `reward_shaped = reward - social_penalty`.
3. Evaluate against no-social-penalty baseline.
4. Generate first benchmark table and add to this log.
