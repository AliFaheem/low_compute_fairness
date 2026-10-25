"""Keep the Wanda calibration documents: full texts in the private HF data repo, ID manifests in git.

    uv run python scripts/compression/sync_calibration.py push       # GPU machine: logs/calibration/*.jsonl -> HF_DATA_REPO/calibration/
    uv run python scripts/compression/sync_calibration.py manifest   # any machine: HF -> docs/calibration/*.tsv (no text)

The texts come from C4 and FineWeb-2 (both ODC-By). The public code repo only carries the manifests:
source, file, document id, url and token offset, enough to rebuild the exact calibration set.
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
import common  # noqa: E402

LOCAL = "logs/calibration"
REMOTE = "calibration"
MANIFESTS = "docs/calibration"


def push():
    from huggingface_hub import HfApi
    repo = os.environ["HF_DATA_REPO"]
    api = HfApi()
    if not api.repo_info(repo, repo_type="dataset").private:
        raise SystemExit(f"{repo} is public; refusing to upload")
    files = sorted(glob.glob(os.path.join(LOCAL, "*.jsonl")))
    if not files:
        print("no calibration files to push")
        return
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=LOCAL, path_in_repo=REMOTE,
                      allow_patterns=["*.jsonl"], commit_message="Wanda calibration documents")
    print(f"pushed {len(files)} calibration files to {repo}/{REMOTE}/")


def manifest():
    from huggingface_hub import snapshot_download
    repo = os.environ["HF_DATA_REPO"]
    root = snapshot_download(repo, repo_type="dataset", allow_patterns=[f"{REMOTE}/*.jsonl"])
    os.makedirs(MANIFESTS, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(root, REMOTE, "*.jsonl"))):
        name = os.path.basename(path)[: -len(".jsonl")]
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        out = os.path.join(MANIFESTS, f"{name}.tsv")
        with open(out, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["source", "file", "id", "url", "token_start", "chars"])
            for r in rows:
                w.writerow([r["source"], r["file"], r["id"], r.get("url", ""), r.get("token_start", ""), len(r["text"])])
        print(f"{out}: {len(rows)} documents")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["push", "manifest"])
    args = ap.parse_args()
    common.load_env()
    push() if args.action == "push" else manifest()


if __name__ == "__main__":
    main()
