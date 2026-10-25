"""Keep one version of each Wanda calibration set: c4_en (English C4) and mix (C4 en + FineWeb-2 Urdu + Roman Urdu).

    # 1. collect the per-model lists that wanda_prune.py writes (logs/calibration/<model>-wanda<NN>[mix].jsonl)
    #    from each GPU machine into one local folder, e.g. with scp
    # 2. regenerate lists for prunes that ran before recording existed (no GPU; tokenizer only)
    uv run python scripts/compression/sync_calibration.py regen-c4 --models qwen3-8b gemma-4-e4b --dir logs/calibration
    # 3. merge into the two files, upload them to the private HF data repo, write the git manifests
    uv run python scripts/compression/sync_calibration.py merge --dir logs/calibration
    uv run python scripts/compression/sync_calibration.py push
    uv run python scripts/compression/sync_calibration.py manifest

Each model tokenizes text differently, so each model draws a slightly different document set under the same
procedure and seed (sparsity does not matter). The merged files hold every document once, with the models that
used it. Texts are C4 and FineWeb-2 (ODC-By); the public repo only carries the manifests (no text).
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

OUT_DIR = "data/our_data/calibration"   # merged files (git-ignored data folder)
REMOTE = "calibration"                  # folder in HF_DATA_REPO
MANIFESTS = "docs/calibration"
SETS = ("c4_en", "mix")


def set_of(filename):
    return "mix" if filename.endswith("mix.jsonl") else "c4_en"


def regen_c4(models, directory):
    """English C4 calibration documents for prunes done before recording existed (same seed and procedure)."""
    from transformers import AutoTokenizer
    import wanda_prune
    os.makedirs(directory, exist_ok=True)
    for m in models:
        spec = common.load_models()["models"][m]
        tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=spec["revision"])
        rec = []
        wanda_prune.c4_windows(tok, wanda_prune.C4_TRAIN, 128, 2048, 0, rec)
        docs = [dict(r, source="en", file=wanda_prune.C4_TRAIN) for r in rec]
        wanda_prune.save_calibration(directory, f"{m}-wanda50", docs)


def merge(directory, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    for name in SETS:
        merged = {}
        for path in sorted(glob.glob(os.path.join(directory, "*.jsonl"))):
            if set_of(path) != name:
                continue
            model = os.path.basename(path).rsplit("-wanda", 1)[0]
            for line in open(path, encoding="utf-8"):
                d = json.loads(line)
                key = (d["source"], d["id"])
                if key not in merged:
                    merged[key] = {k: v for k, v in d.items() if k != "token_start"} | {"models": []}
                if model not in merged[key]["models"]:
                    merged[key]["models"].append(model)
        out = os.path.join(out_dir, f"{name}.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(d, ensure_ascii=False) + "\n" for d in merged.values())
        by_src = {}
        for (src, _), d in merged.items():
            by_src[src] = by_src.get(src, 0) + 1
        print(f"{out}: {len(merged)} documents {by_src}")


def push():
    from huggingface_hub import HfApi
    repo = os.environ["HF_DATA_REPO"]
    api = HfApi()
    if not api.repo_info(repo, repo_type="dataset").private:
        raise SystemExit(f"{repo} is public; refusing to upload")
    for name in SETS:
        api.upload_file(path_or_fileobj=os.path.join(OUT_DIR, f"{name}.jsonl"), path_in_repo=f"{REMOTE}/{name}.jsonl",
                        repo_id=repo, repo_type="dataset", commit_message=f"Wanda calibration set {name}")
        print(f"pushed {name}.jsonl to {repo}/{REMOTE}/")


def manifest():
    os.makedirs(MANIFESTS, exist_ok=True)
    for name in SETS:
        rows = [json.loads(line) for line in open(os.path.join(OUT_DIR, f"{name}.jsonl"), encoding="utf-8")]
        out = os.path.join(MANIFESTS, f"{name}.tsv")
        with open(out, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["source", "file", "id", "url", "chars", "models"])
            for r in rows:
                w.writerow([r["source"], r["file"], r["id"], r.get("url", ""), len(r["text"]), ",".join(r["models"])])
        print(f"{out}: {len(rows)} documents")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["regen-c4", "merge", "push", "manifest"])
    ap.add_argument("--dir", default="logs/calibration", help="folder with the per-model calibration lists")
    ap.add_argument("--models", nargs="+", default=[])
    ap.add_argument("--out", default=OUT_DIR, help="where merge writes c4_en.jsonl and mix.jsonl")
    args = ap.parse_args()
    common.load_env()
    if args.action == "regen-c4":
        regen_c4(args.models, args.dir)
    elif args.action == "merge":
        merge(args.dir, args.out)
    elif args.action == "push":
        push()
    else:
        manifest()


if __name__ == "__main__":
    main()
