"""Diagnose a quantized variant that collapses (e.g. Gemma-4-E4B NF4): perplexity under several bitsandbytes configs.

    uv run python scripts/compression/check_quant.py --model gemma-4-e4b

Configs: bf16 (reference), int8, nf4 (as in run.py), and nf4 limited to the decoder's attention/MLP linears
(every other linear layer, e.g. Gemma 4's per-layer embedding projections, stays in bf16). Also lists which
linear layers outside attention/MLP exist, since those are the usual suspects.
Writes logs/quant_check_<model>.json.
"""
import argparse
import collections
import gc
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import wanda_prune as wp  # noqa: E402

CORE = re.compile(r"layers\.\d+\.(self_attn|mlp)\.")


def load(spec, qc, device):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    kw = dict(revision=spec["revision"], dtype=torch.bfloat16, device_map=device)
    if qc is not None:
        kw["quantization_config"] = qc
    try:
        return AutoModelForCausalLM.from_pretrained(spec["hf_id"], **kw).eval()
    except ValueError:
        return AutoModelForImageTextToText.from_pretrained(spec["hf_id"], **kw).eval()


def main():
    from transformers import AutoTokenizer, BitsAndBytesConfig
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-4-e4b")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    common.load_env()
    spec = common.load_models()["models"][args.model]
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=spec["revision"])
    evals = {"en": wp.c4_windows(tok, wp.C4_VALID, args.windows, args.seqlen, 0),
             "ur": wp.packed_windows(tok, wp.urdu_split(0)[1], args.windows, args.seqlen, 0)}

    # Names of linear layers outside decoder attention/MLP, from the bf16 model.
    model = load(spec, None, args.device)
    linears = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    non_core = [n for n in linears if not CORE.search(n) and not n.endswith("lm_head")]
    groups = collections.Counter(re.sub(r"\.\d+\.", ".N.", n) for n in non_core)
    print(f"{len(linears)} linear layers; {len(non_core)} outside decoder attention/MLP:")
    for g, c in groups.most_common(25):
        print(f"  {c:4d} x {g}")
    results = {"bf16": {lang: wp.perplexity(model, w, args.device) for lang, w in evals.items()}}
    print("bf16", results["bf16"], flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()

    configs = {
        "int8": BitsAndBytesConfig(load_in_8bit=True),
        "nf4": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False),
        "nf4_core_only": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False,
                                            llm_int8_skip_modules=non_core + ["lm_head"]),
    }
    for name, qc in configs.items():
        model = load(spec, qc, args.device)
        quantized = sum(type(m).__name__ in ("Linear4bit", "Linear8bitLt") for m in model.modules())
        results[name] = {lang: wp.perplexity(model, w, args.device) for lang, w in evals.items()}
        results[name]["quantized_linear_layers"] = quantized
        print(name, results[name], flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()

    out = f"logs/quant_check_{args.model}.json"
    os.makedirs("logs", exist_ok=True)
    json.dump({"model": args.model, "windows": args.windows, "seqlen": args.seqlen, "results": results,
               "non_core_linear_groups": dict(groups)}, open(out, "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
