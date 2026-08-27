#!/usr/bin/env python3
"""Real multimodal LoRA/QLoRA supervised training for MetaUrban social labels."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List

from common import assert_example_schema, build_messages, load_config, load_jsonl, resolve_image, save_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/qwen_vl_qlora.yaml")
    parser.add_argument("--dataset_dir", type=Path, required=True, help="Output from prepare_sft_dataset.py")
    parser.add_argument("--dataset_root", type=Path, required=True, help="Root containing the relative RGB paths")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sanity_run", action="store_true")
    parser.add_argument("--sanity_examples", type=int, default=64)
    return parser.parse_args()


def require_training_dependencies():
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            BitsAndBytesConfig,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Training requires torch, transformers, peft, accelerate, qwen-vl-utils, "
            "and bitsandbytes when use_4bit=true."
        ) from exc
    return {
        "torch": torch,
        "LoraConfig": LoraConfig,
        "PeftModel": PeftModel,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModel": AutoModelForImageTextToText,
        "AutoProcessor": AutoProcessor,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "get_scheduler": get_linear_schedule_with_warmup,
    }


class SFTCollator:
    """Tokenize one-image chats and mask every token except the assistant target."""

    def __init__(self, processor, dataset_root: Path, max_length: int):
        self.processor = processor
        self.dataset_root = dataset_root
        self.max_length = max_length

    def _render(self, example, image, include_target):
        messages = build_messages(example, image, include_target=include_target)
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not include_target,
        )

    def __call__(self, examples: List[Dict[str, Any]]):
        import torch

        images = [resolve_image(self.dataset_root, example) for example in examples]
        full_text = [self._render(example, image, True) for example, image in zip(examples, images)]
        prompt_text = [self._render(example, image, False) for example, image in zip(examples, images)]
        full = self.processor(
            text=full_text,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt = self.processor(
            text=prompt_text,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = full["input_ids"].clone()
        labels[full["attention_mask"] == 0] = -100
        for row in range(len(examples)):
            full_length = int(full["attention_mask"][row].sum())
            prompt_length = int(prompt["attention_mask"][row].sum())
            if prompt_length >= full_length:
                raise ValueError("Target was truncated; raise max_sequence_length")
            full_ids = full["input_ids"][row, :prompt_length]
            prompt_ids = prompt["input_ids"][row, :prompt_length]
            if not torch.equal(full_ids, prompt_ids):
                raise ValueError("Prompt tokens are not a prefix of the supervised conversation")
            labels[row, :prompt_length] = -100
            if not torch.any(labels[row] != -100):
                raise ValueError("No assistant target tokens remain after masking")
        full["labels"] = labels
        return full


def make_model(config, deps):
    torch = deps["torch"]
    model_cfg = config["model"]
    quant_cfg = config["quantization"]
    use_4bit = bool(quant_cfg["use_4bit"])
    if use_4bit and not torch.cuda.is_available():
        raise RuntimeError("4-bit QLoRA requires a CUDA GPU; set use_4bit=false for full-precision LoRA")
    compute_dtype = getattr(torch, str(quant_cfg["compute_dtype"]))
    load_kwargs = {
        "torch_dtype": compute_dtype,
        "device_map": model_cfg.get("device_map", "auto"),
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
    }
    if use_4bit:
        load_kwargs["quantization_config"] = deps["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg["quant_type"],
            bnb_4bit_use_double_quant=bool(quant_cfg["double_quant"]),
            bnb_4bit_compute_dtype=compute_dtype,
        )
    model = deps["AutoModel"].from_pretrained(model_cfg["base_model"], **load_kwargs)
    if use_4bit:
        model = deps["prepare_model_for_kbit_training"](
            model, use_gradient_checkpointing=bool(config["training"]["gradient_checkpointing"])
        )
    if config["training"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.config.use_cache = False
    lora_cfg = config["lora"]
    peft_config = deps["LoraConfig"](
        r=int(lora_cfg["rank"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        bias="none",
        target_modules=list(lora_cfg["target_modules"]),
        task_type="CAUSAL_LM",
    )
    model = deps["get_peft_model"](model, peft_config)
    return model


def model_input_device(model):
    embedding = model.get_input_embeddings()
    if embedding is not None and embedding.weight.device.type != "meta":
        return embedding.weight.device
    for parameter in model.parameters():
        if parameter.device.type not in {"meta", "cpu"}:
            return parameter.device
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("Could not determine the model input device")


def move_batch(batch, device):
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def generate_after_reload(model, processor, collator, example, torch):
    image = resolve_image(collator.dataset_root, example)
    messages = build_messages(example, image, include_target=False)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    device = model_input_device(model)
    inputs = move_batch(inputs, device)
    model.eval()
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=24, do_sample=False)
    trimmed = generated[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


def main():
    args = parse_args()
    config = load_config(args.config)
    deps = require_training_dependencies()
    torch = deps["torch"]
    seed = int(config["training"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Output directory must be new or empty; existing evidence is never overwritten")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir()
    (args.output_dir / "checkpoints").mkdir()

    train_rows = load_jsonl(args.dataset_dir / "train.jsonl")
    val_rows = load_jsonl(args.dataset_dir / "val.jsonl")
    assert_example_schema(train_rows)
    assert_example_schema(val_rows)
    if args.sanity_run:
        if not 20 <= args.sanity_examples <= 100:
            raise ValueError("--sanity_examples must be between 20 and 100")
        if len(train_rows) < 20:
            raise ValueError("Sanity mode requires at least 20 training examples")
        train_rows = train_rows[: min(args.sanity_examples, len(train_rows))]

    save_json(args.output_dir / "config.json", {**config, "sanity_run": args.sanity_run})
    for name in ("dataset_statistics.json", "split_manifest.json"):
        source = args.dataset_dir / name
        if source.exists():
            (args.output_dir / name).write_bytes(source.read_bytes())

    processor = deps["AutoProcessor"].from_pretrained(
        config["model"]["base_model"],
        min_pixels=int(config["data"]["min_pixels"]),
        max_pixels=int(config["data"]["max_pixels"]),
        trust_remote_code=bool(config["model"].get("trust_remote_code", False)),
    )
    model = make_model(config, deps)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable <= 0:
        raise RuntimeError("No trainable LoRA parameters were created")
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    collator = SFTCollator(processor, args.dataset_root, int(config["data"]["max_sequence_length"]))
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        train_rows,
        batch_size=int(config["training"]["per_device_batch_size"]),
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=int(config["data"].get("num_workers", 0)),
    )
    gradient_accumulation = int(config["training"]["gradient_accumulation_steps"])
    epochs = 1 if args.sanity_run else int(config["training"]["epochs"])
    update_steps = math.ceil(len(loader) / gradient_accumulation) * epochs
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = deps["get_scheduler"](
        optimizer,
        num_warmup_steps=int(update_steps * float(config["training"]["warmup_ratio"])),
        num_training_steps=update_steps,
    )

    trainable_parameters = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    tracked_name, tracked_parameter = next(
        ((name, parameter) for name, parameter in trainable_parameters if "lora_B" in name),
        trainable_parameters[0],
    )
    initial_parameter = tracked_parameter.detach().float().cpu().clone()
    finite_loss = True
    nonzero_lora_gradient = False
    global_step = 0
    optimizer_steps = 0
    log_path = args.output_dir / "logs" / "train.jsonl"
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulated_micro_steps = 0
    with log_path.open("w", encoding="utf-8") as log_handle:
        for epoch in range(epochs):
            for batch_index, batch in enumerate(loader):
                batch = move_batch(batch, model_input_device(model))
                output = model(**batch)
                loss = output.loss
                if not torch.isfinite(loss):
                    finite_loss = False
                    raise FloatingPointError(f"Non-finite loss at step {global_step}: {loss.item()}")
                (loss / gradient_accumulation).backward()
                global_step += 1
                accumulated_micro_steps += 1
                should_step = accumulated_micro_steps == gradient_accumulation or batch_index + 1 == len(loader)
                if should_step:
                    nonzero_lora_gradient = nonzero_lora_gradient or any(
                        parameter.grad is not None and bool(torch.any(parameter.grad.detach() != 0))
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    )
                    torch.nn.utils.clip_grad_norm_(
                        (parameter for parameter in model.parameters() if parameter.requires_grad),
                        float(config["training"]["max_grad_norm"]),
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    accumulated_micro_steps = 0
                row = {"epoch": epoch + 1, "micro_step": global_step, "loss": float(loss.detach().cpu())}
                log_handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                log_handle.flush()
                print(json.dumps(row))
            checkpoint = args.output_dir / "checkpoints" / f"checkpoint-epoch-{epoch + 1}"
            model.save_pretrained(checkpoint)

    parameter_changed = not torch.equal(initial_parameter, tracked_parameter.detach().float().cpu())
    if not nonzero_lora_gradient:
        raise RuntimeError("Backward completed but every LoRA gradient was zero")
    if not parameter_changed:
        raise RuntimeError(f"Optimizer did not change tracked LoRA parameter {tracked_name}")

    final_adapter = args.output_dir / "final_adapter"
    model.save_pretrained(final_adapter)
    processor.save_pretrained(final_adapter)
    if not (final_adapter / "adapter_config.json").is_file():
        raise RuntimeError("Adapter save verification failed")

    reload_inference = None
    adapter_reloaded = False
    if args.sanity_run:
        base = model.unload()
        reloaded = deps["PeftModel"].from_pretrained(base, final_adapter)
        adapter_reloaded = True
        probe = val_rows[0] if val_rows else train_rows[0]
        reload_inference = generate_after_reload(reloaded, processor, collator, probe, torch)
        if not reload_inference.strip():
            raise RuntimeError("Reloaded adapter produced an empty inference response")

    summary = {
        "trainable_lora_parameters": trainable,
        "loss_finite": finite_loss,
        "backward_succeeded": True,
        "nonzero_lora_gradients": nonzero_lora_gradient,
        "optimizer_steps": optimizer_steps,
        "lora_parameter_changed": parameter_changed,
        "adapter_saved": True,
        "adapter_reloaded": adapter_reloaded,
        "reload_inference_succeeded": bool(reload_inference and reload_inference.strip()),
    }
    save_json(args.output_dir / "training_verification.json", summary)
    print("Training verification summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
