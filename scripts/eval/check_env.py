"""Check that this machine can run the experiments: GPU, libraries, HF token, model access, data.

    uv run python scripts/eval/check_env.py            # all models in configs/models.yaml
    uv run python scripts/eval/check_env.py --machine lab-4090   # only the models assigned to it

Prints one line per check and exits non-zero if any check fails.
"""
import argparse
import os
import sys

import common
import hub

failed = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        failed.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine")
    ap.add_argument("--cpu", action="store_true", help="do not require a GPU (smoke tests only)")
    args = ap.parse_args()
    common.load_env()

    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            check(f"GPU {i}", True, f"{p.name}, {p.total_memory / 1e9:.0f} GB, CUDA {torch.version.cuda}")
        try:
            import bitsandbytes  # noqa: F401
            x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
            lin = bitsandbytes.nn.Linear4bit(64, 64, compute_dtype=torch.bfloat16, quant_type="nf4").cuda()
            lin(x)
            check("bitsandbytes", True, bitsandbytes.__version__)
        except Exception as e:
            check("bitsandbytes", False, f"{type(e).__name__}: {e}")
    else:
        check("GPU", args.cpu, "no CUDA GPU visible to torch" + (" (allowed with --cpu)" if args.cpu else
                                                                "; check nvidia-smi and the driver (>= 525)"))

    from huggingface_hub import HfApi, hf_hub_download
    try:
        user = HfApi().whoami()["name"]
        check("HF token", True, f"logged in as {user}")
    except Exception as e:
        check("HF token", False, f"set HF_TOKEN in .env ({type(e).__name__})")
        user = None

    for var in ("HF_DATA_REPO", "HF_RESULTS_REPO"):
        check(var, bool(os.environ.get(var)), os.environ.get(var, "not set in .env"))
    if user and hub.repo():
        try:
            hub.ensure_private(hub.repo())
            check("results repo private + writable", True, hub.repo())
        except BaseException as e:
            check("results repo private + writable", False, str(e))

    models = common.load_models()["models"]
    keys = list(models)
    if args.machine:
        keys = (common.load_runs()["machines"] or {}).get(args.machine) or []
        check(f"machine {args.machine} in configs/runs.yaml", bool(keys))
    for key in keys:
        spec = models[key]
        try:
            hf_hub_download(spec["hf_id"], "config.json", revision=spec["revision"])
            check(f"model access {key}", True, spec["hf_id"])
        except Exception as e:
            check(f"model access {key}", False, f"{spec['hf_id']}: {type(e).__name__}"
                  + (" (accept the license on huggingface.co)" if spec.get("gated") else ""))

    for lang in ("en", "ur", "roman_ur"):
        n = len(common.load_items(lang)) if common.data_files(lang) else 0
        check(f"data {lang}", n == 6432, f"{n} items" + ("" if n else "; run scripts/data_prep/hf_data.py pull"))

    print(f"\n{'All checks passed.' if not failed else f'{len(failed)} check(s) failed.'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
