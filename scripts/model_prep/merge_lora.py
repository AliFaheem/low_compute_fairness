"""Merge a LoRA adapter repo into its base model and save a full bf16 checkpoint under models/ (local, never uploaded).

    uv run python scripts/model_prep/merge_lora.py --model alif-1.0-8b

The model's entry in configs/models.yaml gives `adapter` (HF repo + revision of the adapter) and `base` (a model key
whose weights the adapter was trained on). Alif-1.0-8B-Instruct ships only a LoRA adapter on
unsloth/Meta-Llama-3.1-8B; that repo's tensors are identical to meta-llama/Llama-3.1-8B (checked on 5 tensors,
25 Oct 2026), so we merge onto the cached meta-llama copy. Done on CPU without peft:
    W += (B @ A) * scale,  scale = alpha / sqrt(r) with rsLoRA, else alpha / r
and `modules_to_save` tensors (e.g. embed_tokens, lm_head) replace the base tensors. The adapter's tokenizer is saved
with the model. Writes merge_meta.json next to the weights.
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
import common  # noqa: E402

PREFIX = "base_model.model."


def main():
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="alif-1.0-8b")
    args = ap.parse_args()
    common.load_env()
    models = common.load_models()["models"]
    spec = models[args.model]
    adapter, base = spec["adapter"], models[spec["base"]]
    out = spec["hf_id"]
    if os.path.exists(os.path.join(out, "merge_meta.json")):
        print(f"{out} already merged")
        return

    cfg = json.load(open(hf_hub_download(adapter["hf_id"], "adapter_config.json", revision=adapter["revision"])))
    weights = load_file(hf_hub_download(adapter["hf_id"], "adapter_model.safetensors", revision=adapter["revision"]))
    scale = cfg["lora_alpha"] / (math.sqrt(cfg["r"]) if cfg.get("use_rslora") else cfg["r"])
    print(f"adapter r={cfg['r']} alpha={cfg['lora_alpha']} rslora={cfg.get('use_rslora')} scale={scale:.4f}")

    model = AutoModelForCausalLM.from_pretrained(base["hf_id"], revision=base["revision"], dtype=torch.bfloat16,
                                                 device_map="cpu")
    params = dict(model.named_parameters())
    merged = replaced = 0
    with torch.no_grad():
        for key, a in weights.items():
            name = key[len(PREFIX):]
            if ".lora_A." in name:
                target = name.replace(".lora_A.weight", ".weight")
                b = weights[key.replace(".lora_A.", ".lora_B.")]
                w = params[target]
                w.copy_((w.float() + (b.float() @ a.float()) * scale).to(w.dtype))
                merged += 1
            elif ".lora_B." not in name:
                target = name.replace(".modules_to_save.default", "")
                if params[target].shape != a.shape:
                    raise SystemExit(f"{target}: shape {tuple(a.shape)} vs base {tuple(params[target].shape)}")
                params[target].copy_(a.to(params[target].dtype))
                replaced += 1
    print(f"merged {merged} LoRA layers, replaced {replaced} full tensors")
    if merged != 7 * model.config.num_hidden_layers:
        raise SystemExit(f"expected {7 * model.config.num_hidden_layers} LoRA layers")

    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(adapter["hf_id"], revision=adapter["revision"]).save_pretrained(out)
    json.dump({"adapter": adapter, "base": {"hf_id": base["hf_id"], "revision": base["revision"]},
               "adapter_base_declared": cfg["base_model_name_or_path"], "scale": scale,
               "lora_layers": merged, "replaced": replaced}, open(os.path.join(out, "merge_meta.json"), "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
