"""Merge a LoRA/DoRA adapter into its base model, HF format.

DoRA adapters carry an extra lora_magnitude_vector tensor that llama.cpp's
convert_lora_to_gguf.py doesn't understand, so a static
GGUF LoRA adapter isn't an option for DoRA configs -- this merges properly
instead. Loads the base in 4-bit on a single GPU (same as training) rather
than fp16/CPU: a disk-offloaded fp16 load hit three distinct failures in a
row (host RAM OOM at ~28GB needed vs. ~15GB QOS cap, then GPU VRAM OOM from
device_map="auto" filling the L4, then a real internal peft bug in its
disk-offload/adapter-loading interaction) before this simpler path worked.

Usage:
    python experiments/merge_adapter.py --base Qwen/Qwen2.5-Coder-14B-Instruct \
        --adapter models/qlora_r8/final --output models/qlora_r8/merged
"""
from __future__ import annotations
import argparse


def _stream_save_dequantized(model, output_dir: str, shard_bytes: int = 2 * 1024**3) -> int:
    """Write the model's weights to sharded safetensors on disk, dequantizing
    any bnb.nn.Linear4bit tensor one at a time, without ever holding more
    than one shard's worth of tensors in memory.

    peft's Linear4bit.merge() (peft/tuners/lora/bnb.py) dequantizes, adds the
    LoRA/DoRA delta, then REQUANTIZES back into bnb.nn.Params4bit -- by
    design, not a bug -- so merge_and_unload() on a 4-bit base leaves every
    adapted layer still bitsandbytes-quantized (convert_hf_to_gguf.py can't
    read that: "Can not map tensor '...absmax'", 2026-09-16 finding).
    Dequantizing all of them into a normal model.save_pretrained() call
    needs ~28GB for this 14B model at once, which fits neither this GPU's
    22GB VRAM nor the group's ~16GB host-RAM QOS cap (same finding) --
    so instead of materializing the whole model, dequantize and flush each
    tensor to disk immediately.
    """
    import gc
    import glob
    import json
    import os

    import bitsandbytes as bnb
    import torch
    from peft.utils.integrations import dequantize_bnb_weight
    from safetensors.torch import save_file

    # A retried/crashed prior run can leave stale shard files behind; a fresh
    # run with fewer shards would then mix old and new tensors under the same
    # index. Only remove tensor shards
    # and the index -- not the whole directory, in case a caller ever writes
    # config.json/tokenizer files here before calling this.
    for stale in glob.glob(os.path.join(output_dir, "*.safetensors")) + glob.glob(
        os.path.join(output_dir, "model.safetensors.index.json")
    ):
        os.remove(stale)
    os.makedirs(output_dir, exist_ok=True)
    linear4bit = {f"{name}.weight": mod for name, mod in model.named_modules() if isinstance(mod, bnb.nn.Linear4bit)}

    weight_map: dict[str, str] = {}
    shard: dict[str, torch.Tensor] = {}
    shard_size = 0
    shard_num = 0
    n_dequantized = 0

    def flush() -> None:
        nonlocal shard, shard_size, shard_num
        if not shard:
            return
        shard_num += 1
        fname = f"model-{shard_num:05d}.safetensors"
        save_file(shard, os.path.join(output_dir, fname), metadata={"format": "pt"})
        weight_map.update({name: fname for name in shard})
        shard = {}
        shard_size = 0
        gc.collect()
        torch.cuda.empty_cache()

    for module_name, module in model.named_modules():
        for pname, param in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{pname}" if module_name else pname
            if full_name in linear4bit:
                tensor = dequantize_bnb_weight(param, state=param.quant_state).cpu()
                n_dequantized += 1
            else:
                tensor = param.detach().cpu().clone()
            shard[full_name] = tensor
            shard_size += tensor.numel() * tensor.element_size()
            if shard_size >= shard_bytes:
                flush()
    flush()

    # Rename shard files now that the final count is known, and write the
    # standard HF sharded-checkpoint index so downstream tooling (including
    # convert_hf_to_gguf.py) finds every tensor.
    total = shard_num
    old_to_new = {
        old_fname: f"model-{int(old_fname[len('model-'):-len('.safetensors')]):05d}-of-{total:05d}.safetensors"
        for old_fname in set(weight_map.values())
    }
    for old_fname, new_fname in old_to_new.items():
        os.rename(os.path.join(output_dir, old_fname), os.path.join(output_dir, new_fname))
    renamed_map = {name: old_to_new[fname] for name, fname in weight_map.items()}

    total_bytes = sum(
        os.path.getsize(os.path.join(output_dir, f)) for f in os.listdir(output_dir) if f.endswith(".safetensors")
    )
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_bytes}, "weight_map": renamed_map}, f)

    return n_dequantized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Single-device 4-bit load, same bnb config training used: a disk-offloaded
    # fp16 load hit three distinct failures in a row (host RAM OOM, GPU VRAM
    # OOM, then a real internal peft bug in its disk-offload/adapter-loading
    # interaction -- _update_offload KeyError from module-name mangling,
    # 2026-09-16). 4-bit needs ~9GB, comfortably fits the 22GB L4 alone, no
    # offload machinery involved at all. peft supports merge_and_unload() on
    # a 4-bit base by dequantizing internally.
    # bfloat16, matching finetune_qlora.py's training dtype exactly (that
    # file's bfloat16 comment has the full story) -- PyTorch silently
    # promotes a float16+bfloat16 elementwise add to float32 rather than
    # erroring, so a mismatch here wouldn't crash, just quietly mix
    # precisions during the merge.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    print(f"[merge] Loading base model {args.base} (4-bit, single GPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb_config, device_map={"": 0}, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    print(f"[merge] Loading adapter from {args.adapter}...")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    # model.config still carries the base load's 4-bit quantization_config
    # (inherited from from_pretrained) even though the weights will be saved
    # as plain fp16 below -- left in place, convert_hf_to_gguf.py reads it
    # and tries to dequantize already-fp16 tensors via its own unsupported
    # bitsandbytes path. Strip it so the saved
    # config.json accurately describes plain fp16 weights.
    if hasattr(model.config, "quantization_config"):
        del model.config.quantization_config
    print(f"[merge] Saving to {args.output} (streaming dequantized weights to disk)...")
    n_dequantized = _stream_save_dequantized(model, args.output)
    print(f"[merge] Dequantized {n_dequantized} bitsandbytes-quantized tensors during save")
    model.config.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("[merge] Done.")


if __name__ == "__main__":
    main()
