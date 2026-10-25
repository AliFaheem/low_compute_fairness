"""Shared helpers for the evaluation scripts: configs, run ids, data and prompts."""
import glob
import hashlib
import json
import os

import yaml

DATA_DIR = "data/our_data"
PROMPTS = "prompts/pakbbq_eval_prompts.json"
RUNS_DIR = "results/runs"
SMOKE_DIR = "results/smoke"
ORDERS = 3  # cyclic option orders per item
ANS = ["ans0", "ans1", "ans2"]


def load_env(path=".env"):
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load_models():
    return yaml.safe_load(open("configs/models.yaml", encoding="utf-8"))


def load_runs():
    return yaml.safe_load(open("configs/runs.yaml", encoding="utf-8"))


def run_id(model, variant, lang):
    return f"{model}__{variant}__{lang}"


def grid():
    """All (experiment, model, variant, lang) runs in configs/runs.yaml, in order."""
    cfg = load_runs()
    for exp, spec in cfg["experiments"].items():
        for model in spec["models"]:
            for variant in spec["variants"]:
                for lang in cfg["languages"]:
                    yield exp, model, variant, lang


def data_files(lang):
    """Data split of a prompt language; prompt variants such as ur_abc reuse another split (`data` key)."""
    split = json.load(open(PROMPTS, encoding="utf-8")).get(lang, {}).get("data", lang)
    return sorted(glob.glob(os.path.join(DATA_DIR, split, "*.jsonl")))


def load_items(lang):
    """All items of a split, in file order (files sorted by name)."""
    return [json.loads(line) for path in data_files(lang) for line in open(path, encoding="utf-8")]


def sample_items(items, n):
    """Deterministic, evenly spaced sample across all categories (for smoke tests)."""
    if not n or n >= len(items):
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def sha256_files(paths):
    h = hashlib.sha256()
    for path in paths:
        h.update(open(path, "rb").read())
    return h.hexdigest()


def load_template(lang):
    return json.load(open(PROMPTS, encoding="utf-8"))[lang]


def prompt_hash(lang):
    return hashlib.sha256(json.dumps(load_template(lang), ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def shown_options(order):
    """Original option index shown at each display position: cyclic rotation by `order`."""
    return [(pos + order) % 3 for pos in range(3)]


def build_prompt(item, t, order):
    """PakBBQ's Gemini plain-text layout plus the answer cue; see prompts/README.md."""
    labels = t["labels"]
    opts = "\n".join(f"{labels[pos]}. {item[ANS[orig]]}" for pos, orig in enumerate(shown_options(order)))
    return (f"{t['instruction']}\n"
            f"{t['context']}: {item['context']}\n\n"
            f"{t['question']}: {item['question']}\n\n"
            f"{t['options']}:\n{opts}\n\n"
            f"{t['respond']}\n"
            f"{t['answer_cue']}")
