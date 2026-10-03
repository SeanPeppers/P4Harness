"""QLoRA fine-tuning of qwen2.5-coder:14b on the p4harness P4 SFC dataset.

Setup (run once in a separate fine-tuning venv):
    python -m venv .venv-finetune
    source .venv-finetune/bin/activate
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install transformers peft bitsandbytes trl datasets accelerate

Usage:
    source .venv-finetune/bin/activate
    python experiments/finetune_qlora.py --data data/verified_corpus.jsonl --output models/qwen-p4-qlora

The fine-tuned model is exported in Hugging Face format and can be converted to
GGUF for Ollama deployment via llama.cpp convert_hf_to_gguf.py.

Design choices:
- r=8, alpha=32: standard QLoRA settings for 14B models
- 4-bit NF4 quantization: fits in 16GB VRAM with 14B model
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Dataset: RAG store (task, p4_code) pairs — 10k+ p4c-verified examples (expanded)
- Training format: instruction-following (system prompt + task → p4_code)
"""
from __future__ import annotations
import argparse
import json
import sys


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a P4_16 programmer for bmv2 Service Function Chaining. Output ONLY the modified table and action definitions in a ```p4 block. Keep under 40 lines. Follow NSH SFC rules: decrement SI, use NoAction() as default, snake_case field names."""


def load_rag_dataset(jsonl_path: str) -> list[dict]:
    """Load (task, p4_code) pairs from the RAG store JSONL file."""
    examples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            task = entry.get("task", "").strip()
            code = entry.get("p4_code", "").strip()
            if task and code:
                examples.append({"task": task, "p4_code": code})
    print(f"[dataset] Loaded {len(examples)} examples from {jsonl_path}")
    return examples


# A prior version of this file added 46 custom P4 domain tokens (bit<N>,
# MyIngress, headers_t, NoAction, ...) as atomic vocabulary to save context
# and speed convergence. Removed: their embed_tokens/lm_head rows were
# initialized as a mean of sub-token embeddings, which lands ~42% below the
# base vocabulary's norm and is undertrainable by peft's trainable_token_indices
# in a few epochs -- the new token systematically loses the argmax to a
# base-vocab fragment (e.g. p(' In') > p('MyIngress') after "control ").
# Worse, tokenizing "NoAction" as one atomic id destroyed the model's native
# sub-word spelling of it (base qwen completes "No"->"Action" at p=1.00, the
# fine-tune with p~0), so a bad generation couldn't even fall back to spelling
# the identifier out. Real eval scored 0/50 with every output containing a
# truncated/corrupted custom token (headers_t -> "_t", a bogus duplicate
# "No;" action, bit<N> dropped to a bare integer). Confirmed as the cause (not
# quantization, not GGUF conversion, not Ollama) by loading the merged model
# directly via transformers (bf16, no quantization at all) and reproducing
# identical corruption, and confirmed as fixable by truncating embed_tokens/
# lm_head back to base vocab size on the existing merged checkpoint with zero
# retraining: 12/14 scenarios passed (vs. 0/50 with the tokens), before this
# fix was applied for real. The base tokenizer already
# spells every one of these identifiers correctly; the custom vocabulary was
# never a claimed paper contribution, so dropping it costs nothing.


def format_instruction(task: str, p4_code: str, tokenizer) -> str:
    """Format a (task, p4_code) pair using the model's native chat template."""
    msgs = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: {task}"},
        {"role": "assistant", "content": f"```p4\n{p4_code}\n```"},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


def build_hf_dataset(examples: list[dict], tokenizer):
    """Convert examples to HuggingFace Dataset format."""
    from datasets import Dataset
    records = [{"text": format_instruction(e["task"], e["p4_code"], tokenizer)} for e in examples]
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# QLoRA fine-tuning
# ---------------------------------------------------------------------------

def train(data_path: str, output_dir: str, model_name: str, epochs: int, batch_size: int, lora_r: int = 8, resume_from: str | None = None, use_dora: bool = True) -> None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run: pip install torch transformers peft bitsandbytes trl datasets accelerate")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Fine-tuning requires GPU.")
        print("Install CUDA-enabled PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    print(f"[finetune] GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    print(f"[finetune] Loading model: {model_name}")

    # 4-bit quantization config (NF4). bfloat16, not float16: SFTTrainer.__init__
    # force-casts every trainable LoRA param to bfloat16 whenever the base
    # model is 4-bit (trl/trainer/sft_trainer.py, per the QLoRA paper). A
    # float16 base mixed with bfloat16 trainables blew up on step 1 once any
    # param was actually trainable: `fp16 * bf16` type-promotes to float32,
    # and that float32 stream hit a still-bf16 param elsewhere -- "expected
    # mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16".
    # Matching the base to bfloat16 keeps a single dtype end to end
    #.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 1024
    tokenizer.truncation_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 4,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        use_dora=use_dora,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset — 90/10 train/eval split
    import random as _random
    examples = load_rag_dataset(data_path)
    if len(examples) < 10:
        print(f"ERROR: Only {len(examples)} examples found. Need at least 10.")
        sys.exit(1)

    _random.seed(42)
    _random.shuffle(examples)
    split = max(1, int(len(examples) * 0.1))
    train_examples = examples[split:]
    eval_examples = examples[:split]

    train_dataset = build_hf_dataset(train_examples, tokenizer)
    eval_dataset = build_hf_dataset(eval_examples, tokenizer)
    print(f"[finetune] Train: {len(train_dataset)}, Eval: {len(eval_dataset)}, Epochs: {epochs}")
    print("[finetune] Sample prompt prefix:", repr(train_dataset[0]["text"][:120]))

    # Training arguments
    grad_accum = max(1, 8 // batch_size)
    # transformers 5.x dropped warmup_ratio from TrainingArguments (warmup_steps
    # only now); compute the equivalent step count ourselves to keep the same
    # ~3% warmup behavior this was tuned for.
    total_steps = -(-len(train_examples) // (batch_size * grad_accum)) * epochs
    warmup_steps = max(1, round(total_steps * 0.03))
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=1e-4,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        fp16=False,
        bf16=False,
        # DoRA's magnitude path holds more activations than plain LoRA, which at
        # batch 4 x 1024 tokens is enough to OOM a 22GB L4 on its own; trade the
        # activation memory for recompute.
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=300,
        per_device_eval_batch_size=1,
        eval_strategy="steps",
        eval_steps=300,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        processing_class=tokenizer,
    )

    print("[finetune] Starting training...")
    trainer.train(resume_from_checkpoint=resume_from)

    print(f"[finetune] Saving best LoRA adapters to {output_dir}/final")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    print("[finetune] Done.")
    print()
    print("To convert to GGUF for Ollama:")
    print(f"  python llama.cpp/convert_hf_to_gguf.py {output_dir}/final --outfile qwen-p4-qlora.gguf")
    print("  ollama create p4harness-p4-v2 -f Modelfile")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune qwen2.5-coder:14b on P4 SFC dataset")
    parser.add_argument("--data", default="data/verified_corpus.jsonl", help="Training dataset JSONL path")
    parser.add_argument("--output", default="models/qwen-p4-qlora", help="Output directory for adapters")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-14B-Instruct", help="HuggingFace model name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size (1-2 for 16GB VRAM)")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank (alpha will be rank*4)")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint directory (e.g. output_dir/checkpoint-600)")
    parser.add_argument("--no-dora", action="store_true", help="Plain QLoRA (use_dora=False) instead of DoRA")
    args = parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.output,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lora_r=args.lora_r,
        resume_from=args.resume,
        use_dora=not args.no_dora,
    )


if __name__ == "__main__":
    main()
