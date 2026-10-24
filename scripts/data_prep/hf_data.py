"""Sync data/our_data with a private Hugging Face dataset repo, so GPU machines get the data without git.

    uv run python scripts/data_prep/hf_data.py push   # upload en/, ur/, roman_ur/ and subset_ids.json
    uv run python scripts/data_prep/hf_data.py pull   # download them and run validate_splits.py

Reads HF_TOKEN and HF_DATA_REPO (e.g. "user/pakbbq-subset") from the environment or .env.
The repo is created private, and push refuses to upload to a repo that is public: PakBBQ has no
license file, so the data is not published until that is cleared.
"""
import argparse
import os
import subprocess
import sys

from huggingface_hub import HfApi, snapshot_download

DATA_DIR = "data/our_data"
PATTERNS = ["en/*.jsonl", "ur/*.jsonl", "roman_ur/*.jsonl", "subset_ids.json"]


def load_env(path=".env"):
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def repo_id():
    repo = os.environ.get("HF_DATA_REPO", "")
    if not repo:
        raise SystemExit("Set HF_DATA_REPO (e.g. user/pakbbq-subset) in .env")
    return repo


def push(api):
    repo = repo_id()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type="dataset").private:
        raise SystemExit(f"{repo} is public; refusing to upload. Make it private on huggingface.co first.")
    commit = api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=DATA_DIR,
                               allow_patterns=PATTERNS, commit_message="Update data")
    print(f"Uploaded {DATA_DIR} to private dataset {repo}: {commit.commit_url}")


def pull(api):
    repo = repo_id()
    path = snapshot_download(repo, repo_type="dataset", local_dir=DATA_DIR, allow_patterns=PATTERNS)
    print(f"Downloaded {repo} to {path}")
    subprocess.run([sys.executable, "scripts/data_prep/validate_splits.py"], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["push", "pull"])
    args = ap.parse_args()
    load_env()
    api = HfApi()
    push(api) if args.action == "push" else pull(api)


if __name__ == "__main__":
    main()
