"""Supervised fine-tune (LoRA) on one of the three filtered training sets.

Usage:
  python sft_finetune.py --train-jsonl data/filtered/budget.jsonl --out-dir runs/sft_budget
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from model_io import BASE_MODEL, render_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=args.train_jsonl, split="train")
    dataset = dataset.map(lambda row: {"prompt": render_prompt(tokenizer, row["prompt"])})

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0}
    )

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    config = SFTConfig(
        output_dir=args.out_dir,
        per_device_train_batch_size=8,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        report_to="none",
        completion_only_loss=True,
        max_length=256,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        peft_config=lora_config,
    )
    train_result = trainer.train()

    adapter_dir = str(Path(args.out_dir) / "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"adapter saved to {adapter_dir}")
    print(train_result.metrics)


if __name__ == "__main__":
    main()
