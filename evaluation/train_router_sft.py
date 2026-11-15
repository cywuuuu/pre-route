#!/usr/bin/env python3
"""
Pre-Route Router Distillation Training
======================================

Fine-tune a small model (Qwen3-1.7B) to perform RAG vs LC routing
using knowledge distillation from a large teacher model.

Usage:
    python train_router_sft.py --tag exp1 --epochs 6

Author: Pre-Route Authors
"""

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import SFTTrainer, SFTConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    # Model
    "model_name": "Qwen3-1.7B",
    "model_path": "model_weights",

    # Data
    "train_file": "distillation_datasets/from_multi_answer_models_0.7_0.1_0.2_42/train_data_sft.json",
    "eval_file": "distillation_datasets/from_multi_answer_models_0.7_0.1_0.2_42/eval_data_sft.json",

    # Training - KEY HYPERPARAMETERS
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "epochs": 6,
    "batch_size": 1,
    "gradient_accumulation": 4,
    "max_length": 8000,
    "seed": 42,

    # Output
    "output_dir": "outputs",
}


# =============================================================================
# Callbacks
# =============================================================================

class EpochSaver(TrainerCallback):
    """Save checkpoint after each epoch, keep only last K."""

    def __init__(self, base_dir, tokenizer, keep_last=3):
        self.base_dir = Path(base_dir)
        self.tokenizer = tokenizer
        self.keep_last = keep_last
        (self.base_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        if getattr(args, "process_index", 0) != 0:
            return control

        epoch = int(state.epoch or 0)
        save_dir = self.base_dir / "checkpoints" / f"epoch-{epoch:03d}"
        save_dir.mkdir(exist_ok=True)

        # Save model
        if kwargs.get("model"):
            kwargs["model"].save_pretrained(str(save_dir), safe_serialization=True)

        # Save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(str(save_dir))

        print(f"[EpochSaver] Saved: {save_dir}")
        self._cleanup()
        return control

    def _cleanup(self):
        """Remove old checkpoints."""
        checkpoints = sorted((self.base_dir / "checkpoints").glob("epoch-*"))
        if len(checkpoints) > self.keep_last:
            for old in checkpoints[:-self.keep_last]:
                shutil.rmtree(old)
                print(f"[EpochSaver] Deleted: {old}")


# =============================================================================
# Training
# =============================================================================

def load_system_prompt():
    """Load system prompt from eval_utils."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("eval_utils",
                                                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_utils.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.system_prompt


def train(args):
    """Main training function."""

    # Setup
    set_seed(args.seed)
    model_path = Path(args.model_path) / args.model_name
    output_dir = Path(args.output_dir) / f"{args.model_name}_sft_{args.tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Pre-Route Router Training")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"LR: {args.learning_rate}, Epochs: {args.epochs}")
    print(f"Batch: {args.batch_size} x {args.gradient_accumulation} accum")
    print("="*60)

    # Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load model
    print("[2/5] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), trust_remote_code=True, torch_dtype="auto"
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # Load data
    print("[3/5] Loading data...")
    with open(args.train_file) as f:
        train_data = json.load(f)
    with open(args.eval_file) as f:
        eval_data = json.load(f)
    print(f"  Train: {len(train_data)}, Eval: {len(eval_data)}")

    # Tokenize
    system_prompt = load_system_prompt()

    def tokenize(example):
        thinking = example['thinking_trace']
        if isinstance(thinking, list):
            thinking = thinking[0]

        reason_prefix = "\n" + example.get('reason', '') + "\n" if args.use_reason else "\n"

        instruction = tokenizer(
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{example['decision_prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n",
            add_special_tokens=False
        )
        response = tokenizer(
            reason_prefix + "<thinking>" + thinking + "</thinking><|im_end|>",
            add_special_tokens=False
        )

        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

        if len(input_ids) > args.max_length:
            input_ids = input_ids[:args.max_length]
            attention_mask = attention_mask[:args.max_length]
            labels = labels[:args.max_length]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    train_dataset = Dataset.from_list(train_data).map(tokenize)
    eval_dataset = Dataset.from_list(eval_data).map(tokenize)

    # Training config
    print("[4/5] Configuring training...")
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch = world_size * args.batch_size * args.gradient_accumulation
    steps_per_epoch = math.ceil(len(train_data) / effective_batch)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    print(f"  Effective batch: {effective_batch}")
    print(f"  Steps/epoch: {steps_per_epoch}, Total: {total_steps}, Warmup: {warmup_steps}")

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=1.0,
        max_seq_length=args.max_length,
        bf16=True,
        save_strategy="no",  # Use callback instead
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
    )
    trainer.add_callback(EpochSaver(output_dir, tokenizer, keep_last=3))

    # Save config
    config_save = {
        "model_name": args.model_name,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "max_length": args.max_length,
        "effective_batch_size": effective_batch,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "seed": args.seed,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_save, f, indent=2)

    # Train
    print("[5/5] Training...")
    trainer.train()

    # Save final
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\n Done! Final model: {final_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-Route Router Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Key parameters only
    parser.add_argument("--tag", type=str, default="default",
                        help="Experiment tag")
    parser.add_argument("--model_name", type=str, default=DEFAULT_CONFIG["model_name"],
                        help="Model name")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"],
                        help="Number of epochs")
    parser.add_argument("--lr", "--learning_rate", dest="learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"],
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"],
                        help="Per-device batch size")
    parser.add_argument("--max_length", type=int, default=DEFAULT_CONFIG["max_length"],
                        help="Max sequence length")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"],
                        help="Random seed")

    # Paths
    parser.add_argument("--train_file", type=str, default=DEFAULT_CONFIG["train_file"])
    parser.add_argument("--eval_file", type=str, default=DEFAULT_CONFIG["eval_file"])
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"])

    # Flags
    parser.add_argument("--no_reason", action="store_true",
                        help="Exclude reason field")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.use_reason = not args.no_reason
    args.gradient_accumulation = DEFAULT_CONFIG["gradient_accumulation"]
    args.weight_decay = DEFAULT_CONFIG["weight_decay"]
    args.warmup_ratio = DEFAULT_CONFIG["warmup_ratio"]
    args.lr_scheduler_type = DEFAULT_CONFIG["lr_scheduler_type"]
    args.model_path = DEFAULT_CONFIG["model_path"]

    train(args)
