#!/usr/bin/env python3
"""Run one leakage-safe social-classification inference with an optional adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import build_messages, load_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/qwen_vl_qlora.yaml")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--ego_speed", type=float, required=True)
    parser.add_argument("--ego_heading", type=float, required=True)
    parser.add_argument("--adapter_path", type=Path)
    parser.add_argument("--max_new_tokens", type=int, default=24)
    return parser.parse_args()


def load_model_and_processor(config, adapter_path=None):
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Inference requires torch and transformers") from exc

    model_cfg = config["model"]
    quant_cfg = config["quantization"]
    use_4bit = bool(quant_cfg["use_4bit"])
    if use_4bit and not torch.cuda.is_available():
        raise RuntimeError("The configured 4-bit load requires CUDA")
    dtype = getattr(torch, str(quant_cfg["compute_dtype"]))
    kwargs = {
        "torch_dtype": dtype,
        "device_map": model_cfg.get("device_map", "auto"),
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
    }
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg["quant_type"],
            bnb_4bit_use_double_quant=bool(quant_cfg["double_quant"]),
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForImageTextToText.from_pretrained(model_cfg["base_model"], **kwargs)
    if adapter_path is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("PEFT is required to load --adapter_path") from exc
        model = PeftModel.from_pretrained(model, adapter_path)
    processor = AutoProcessor.from_pretrained(
        model_cfg["base_model"],
        min_pixels=int(config["data"]["min_pixels"]),
        max_pixels=int(config["data"]["max_pixels"]),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    model.eval()
    return model, processor


def input_device(model):
    import torch

    embedding = model.get_input_embeddings()
    if embedding is not None and embedding.weight.device.type != "meta":
        return embedding.weight.device
    for parameter in model.parameters():
        if parameter.device.type != "meta" and parameter.device.type != "cpu":
            return parameter.device
    return torch.device("cpu")


def predict(model, processor, image, ego_speed, ego_heading, max_new_tokens=24):
    import torch

    example = {"ego_speed": ego_speed, "ego_heading": ego_heading}
    messages = build_messages(example, image, include_target=False)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    device = input_device(model)
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def main():
    args = parse_args()
    from PIL import Image

    config = load_config(args.config)
    model, processor = load_model_and_processor(config, args.adapter_path)
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    print(predict(model, processor, image, args.ego_speed, args.ego_heading, args.max_new_tokens))


if __name__ == "__main__":
    main()
