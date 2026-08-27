# MetaUrban VLM Social-Reward Post-Training

This module implements reproducible supervised post-training for social-behavior classification. It is separate from the existing online VLM-as-reward PPO code: the existing path queries a base VLM during reinforcement learning, while this path trains and evaluates a PEFT adapter that can later be loaded by that reward path.

Current status: **pipeline implemented; large-model post-training run pending**. No checkpoint, loss, or evaluation result is claimed in this repository.

## Task and model-visible data

Each example is one post-transition observation:

- one egocentric RGB image;
- `ego_speed` in m/s;
- `ego_heading` in radians;
- a fixed classification instruction.

The target is exactly one of `NEGATIVE_SOCIAL`, `NEUTRAL`, or `POSITIVE_SOCIAL`, serialized as `{"label":"NEGATIVE_SOCIAL"}` (with the corresponding label substituted). The prompt contains no action, reward, collision or distance flag, completion/termination state, teacher metadata, scenario ID, episode ID, or target label.

`episode_id` and `step_index` remain in prepared JSONL only as non-model metadata for leakage prevention and auditability. The collator constructs prompts exclusively from `image_path`, `ego_speed`, and `ego_heading`.

## Model and QLoRA configuration

The default is [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), a 7B-class model published as **8B parameters**. The checked-in config uses 4-bit NF4 double quantization, bfloat16 compute, rank-16 LoRA (`alpha=32`, dropout `0.05`), and language attention/MLP projections as target modules. Base parameters remain frozen; only LoRA parameters are optimized.

See [configs/qwen_vl_qlora.yaml](configs/qwen_vl_qlora.yaml) for the complete, reviewable configuration: learning rate `2e-4`, 3 epochs, batch size 1, gradient accumulation 16, gradient checkpointing, seed 42, maximum sequence length 512, and 512x288 source RGB images.

## Expected label input

The raw collector writes `episodes/<episode_id>/records.jsonl`. Teacher labels are a separate JSONL and must contain a supported label field (`label`, `vlm_label_class`, `target`, or `vlm_label_id`) plus at least one identity:

```json
{"episode_id":"episode_000042","step_index":5,"label":"NEUTRAL"}
```

`record_id` or `image_path` can be used instead. Conflicting identities/labels fail closed. The preparation command joins labels without copying teacher explanations or simulator-only fields, then deterministically assigns complete episodes to train, validation, and test sets.

```bash
python RL/SocialNav/vlm_training/prepare_sft_dataset.py \
  --dataset_root /path/to/raw_dataset \
  --labels /path/to/human_reviewed_teacher_labels.jsonl \
  --output_dir experiments/vlm_social_reward/data \
  --seed 42
```

This writes `train.jsonl`, `val.jsonl`, `test.jsonl`, `dataset_statistics.json`, and `split_manifest.json`. At least three labeled episodes are required so every split is non-empty. Adjacent frames from an episode can never cross split boundaries.

## Training

Install the declared [requirements](requirements.txt) in a CUDA environment. They include PyTorch, Transformers, PEFT, Accelerate, `qwen-vl-utils`, PyYAML, Pillow, and bitsandbytes. A 4-bit run deliberately stops if CUDA is unavailable.

```bash
pip install -r RL/SocialNav/vlm_training/requirements.txt
```

```bash
python RL/SocialNav/vlm_training/train_vlm_lora.py \
  --config RL/SocialNav/vlm_training/configs/qwen_vl_qlora.yaml \
  --dataset_dir experiments/vlm_social_reward/data \
  --dataset_root /path/to/raw_dataset \
  --output_dir experiments/vlm_social_reward/run_001
```

The training loop performs multimodal forward passes, assistant-only supervised language-modeling loss, backward passes, optimizer/scheduler steps, and adapter checkpoints. Runtime outputs are ignored by Git.

For a real 20–100-example end-to-end verification:

```bash
python RL/SocialNav/vlm_training/train_vlm_lora.py \
  --dataset_dir experiments/vlm_social_reward/data \
  --dataset_root /path/to/raw_dataset \
  --output_dir experiments/vlm_social_reward/sanity_001 \
  --sanity_run --sanity_examples 64
```

Sanity mode fails unless trainable LoRA parameters exist, loss is finite, backward creates a non-zero LoRA gradient, an optimizer step changes a LoRA parameter, the adapter saves and reloads, and inference after reload returns non-empty output.

## Evaluation and inference

Evaluation loads the untouched base model first and the base plus adapter second, using the same held-out test examples. It writes accuracy, macro F1, per-class precision/recall/F1, a confusion matrix, structured-output validity, and raw predictions for both variants.

```bash
python RL/SocialNav/vlm_training/evaluate_vlm.py \
  --dataset_dir experiments/vlm_social_reward/data \
  --dataset_root /path/to/raw_dataset \
  --adapter_path experiments/vlm_social_reward/run_001/final_adapter \
  --output_dir experiments/vlm_social_reward/run_001
```

Single-example inference uses the same prompt and supports `--adapter_path`:

```bash
python RL/SocialNav/vlm_training/inference.py \
  --image frame.png --ego_speed 0.8 --ego_heading 1.2 \
  --adapter_path experiments/vlm_social_reward/run_001/final_adapter
```

To use the trained adapter as the PPO online reward model, pass `--vlm_adapter_path PATH` to [train_ppo_onlinevlm.py](../train_ppo_onlinevlm.py). Omitting it preserves the original base-model behavior.

## Experiment evidence layout

Real runs create the following ignored structure; model weights must never be committed:

```text
experiments/vlm_social_reward/
  run_001/
    config.json
    dataset_statistics.json
    split_manifest.json
    logs/
    checkpoints/
    final_adapter/
    evaluation_base.json
    evaluation_lora.json
```

Only small textual logs/results from a completed, reviewed run should ever be selectively added later. See [VLM_TRAINING_EVIDENCE.md](../../../VLM_TRAINING_EVIDENCE.md) for the public evidence boundary.
