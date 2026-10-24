"""Results storage on a private Hugging Face dataset (HF_RESULTS_REPO), mirroring results/runs/.

A run folder results/<kind>/<run_id>/ is stored at <kind>/<run_id>/ in the dataset, where kind is
`runs` (the grid) or `smoke` (--limit tests). While a run is
scoring, a CommitScheduler uploads the folder every few minutes, so a crashed machine loses at
most that much work and the run can continue elsewhere: before scoring, a run with no local
files is downloaded from the dataset first.
"""
import contextlib
import os

from huggingface_hub import CommitScheduler, HfApi, snapshot_download

UPLOAD_EVERY_MINUTES = 5


def repo():
    return os.environ.get("HF_RESULTS_REPO", "")


def ensure_private(name):
    api = HfApi()
    api.create_repo(name, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(name, repo_type="dataset").private:
        raise SystemExit(f"{name} is public; refusing to upload results. Make it private on huggingface.co.")


def fetch_run(name, out_dir, results_root="results"):
    """Download <kind>/<run_id>/ from the dataset into results/<kind>/<run_id>/ (no-op if absent)."""
    rel = os.path.relpath(out_dir, results_root)
    snapshot_download(name, repo_type="dataset", local_dir=results_root, allow_patterns=[f"{rel}/*"])


def fetch_meta(name, results_root="results"):
    """Download every run's meta.json (small), for status across machines."""
    snapshot_download(name, repo_type="dataset", local_dir=results_root, allow_patterns=["runs/*/meta.json"])


def fetch_all(name, results_root="results"):
    snapshot_download(name, repo_type="dataset", local_dir=results_root, allow_patterns=["runs/*/*"])


class Uploader:
    """Background uploads of one run folder; `lock` must be held while writing files in it."""

    def __init__(self, name, out_dir, results_root="results"):
        self.scheduler = CommitScheduler(repo_id=name, repo_type="dataset", folder_path=out_dir,
                                         path_in_repo=os.path.relpath(out_dir, results_root), every=UPLOAD_EVERY_MINUTES,
                                         private=True, allow_patterns=["preds.jsonl", "meta.json"])
        self.lock = self.scheduler.lock

    def finish(self):
        self.scheduler.trigger().result()  # final upload, waits until it is committed
        self.scheduler.stop()


class NoUploader:
    lock = contextlib.nullcontext()

    def finish(self):
        pass
