"""SFT a small open code model on filtered passing rollouts.

Recipe (mirrors MedAgentGym): take rollouts where mean_score >= threshold,
format as multi-turn chat conversations, train a LoRA adapter on top of
Qwen-2.5-Coder-1.5B-Instruct (configurable). 4-bit base via bitsandbytes
keeps it on a single 24 GB GPU.

Use --dry_run on a CPU box first to validate the data->tokens path without
pulling weights or installing trl/peft.
"""
import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


DEFAULT_BASE = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def load_sft_jsonl(path: str):
    ds = load_dataset("json", data_files=path, split="train")
    if "messages" not in ds.column_names:
        raise ValueError(f"expected 'messages' column in {path}; got {ds.column_names}")
    return ds


def render_with_chat_template(tokenizer, ds):
    def _render(ex):
        text = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}
    return ds.map(_render, remove_columns=[c for c in ds.column_names if c != "messages"])


def dry_run(args):
    ds = load_sft_jsonl(args.data)
    print(f"loaded {len(ds)} records from {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    rendered = render_with_chat_template(tokenizer, ds)
    lengths = []
    for i, ex in enumerate(rendered):
        ids = tokenizer(ex["text"], add_special_tokens=False).input_ids
        lengths.append(len(ids))
        if i < args.preview:
            print(f"\n--- record {i} (len={len(ids)} tokens) ---")
            print(ex["text"][:1200] + ("\n...[truncated]" if len(ex["text"]) > 1200 else ""))

    if not lengths:
        print("WARNING: no records rendered.")
        return
    import statistics
    print(f"\ntoken-length stats over {len(lengths)} records: "
          f"min={min(lengths)} mean={statistics.mean(lengths):.0f} "
          f"median={statistics.median(lengths):.0f} max={max(lengths)} "
          f"p95={sorted(lengths)[int(0.95*len(lengths))-1]}")
    print("dry-run OK.")


def train(args):
    try:
        from peft import LoraConfig, get_peft_model
        from transformers import (AutoModelForCausalLM, BitsAndBytesConfig)
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        print(f"missing training deps: {e}", file=sys.stderr)
        print("install with: pip install 'physagentgym[training]'  "
              "(or: pip install trl peft bitsandbytes accelerate)", file=sys.stderr)
        sys.exit(2)
    import torch

    ds = load_sft_jsonl(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = None
    if args.qlora and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=200,
        bf16=torch.cuda.is_available(),
        max_length=args.max_seq_len,
        packing=False,
        report_to="none",
        dataset_text_field=None,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved adapter to {args.output_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="SFT JSONL produced by build_sft_dataset.py")
    ap.add_argument("--base_model", default=DEFAULT_BASE)
    ap.add_argument("--output_dir", default="checkpoints/qwen-coder-1.5b-physagent")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--qlora", action="store_true", help="4-bit base via bitsandbytes")
    ap.add_argument("--seed", type=int, default=42, help="Training seed (data shuffle + init)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Load + tokenize only; don't pull weights or train.")
    ap.add_argument("--preview", type=int, default=2,
                    help="Dry-run: how many rendered examples to print.")
    args = ap.parse_args()

    if args.dry_run:
        dry_run(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
