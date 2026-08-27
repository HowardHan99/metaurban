# MetaUrban VLM Post-Training Evidence

## Existing Work

The existing project uses a Qwen vision-language model as a social-reward model inside PPO-based MetaUrban social-navigation training. That online inference/reward work predates and remains distinct from the post-training implementation below.

## Post-Training Task

The supervised task classifies one post-transition social-navigation observation as `NEGATIVE_SOCIAL`, `NEUTRAL`, or `POSITIVE_SOCIAL`. The assistant emits exactly one JSON object such as `{"label":"NEUTRAL"}`.

## Model

The intended base model is `Qwen/Qwen2.5-VL-7B-Instruct`.

## Model Size

This is a 7B-class Qwen VLM published on its official Hugging Face model page as **8B parameters**.

## Training Method

Supervised fine-tuning with QLoRA: 4-bit NF4 base-model loading and trainable low-rank adapters on language attention and MLP projections. A non-quantized LoRA mode is also supported by configuration.

## Data

Examples contain MetaUrban egocentric RGB observations plus ego speed and heading. Records are grouped and split by complete trajectory/episode; adjacent frames are never randomly divided across train, validation, and test sets. Simulator rewards, VLM rewards, actions, collision/distance/completion/terminal fields, teacher metadata, and labels are not exposed in the model prompt.

## Training Code

[RL/SocialNav/vlm_training/train_vlm_lora.py](RL/SocialNav/vlm_training/train_vlm_lora.py)

## Evaluation Code

[RL/SocialNav/vlm_training/evaluate_vlm.py](RL/SocialNav/vlm_training/evaluate_vlm.py)

## PPO Integration

[RL/SocialNav/train_ppo_onlinevlm.py](RL/SocialNav/train_ppo_onlinevlm.py)

## Current Status

Training pipeline implemented; large-model post-training run pending.

No large-model adapter, training loss, checkpoint, or base-versus-LoRA evaluation result is currently claimed. Evidence files should be populated only by a real run of the checked-in pipeline.
