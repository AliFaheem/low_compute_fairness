"""Build a stratified PakBBQ subset: min(total, CAP) items per category.

Within each category the quota is split evenly across the 4 cells
(ambig/disambig x neg/nonneg), and each cell is filled round-robin over
templates (question_index) so no single template dominates. IDs are shared
across languages, so one list covers en, ur and Roman Urdu.
"""
import argparse
import collections
import glob
import json
import os
import random

CELLS = [(c, p) for c in ("ambig", "disambig") for p in ("neg", "nonneg")]


def sample_cell(rows, k, rng):
    by_template = collections.defaultdict(list)
    for r in rows:
        by_template[r["question_index"]].append(r["id"])
    queues = list(by_template.values())
    for q in queues:
        rng.shuffle(q)
    rng.shuffle(queues)
    picked = []
    while len(picked) < k:
        for q in queues:
            if q and len(picked) < k:
                picked.append(q.pop())
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/our_data/en")
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/our_data/subset_ids.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    subset = {}
    for path in sorted(glob.glob(os.path.join(args.data, "*.jsonl"))):
        category = os.path.basename(path)[: -len(".jsonl")]
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        if len(rows) <= args.cap:
            ids = [r["id"] for r in rows]
        else:
            ids = []
            for cell in CELLS:
                cell_rows = [r for r in rows if (r["context_condition"], r["question_polarity"]) == cell]
                ids += sample_cell(cell_rows, args.cap // len(CELLS), rng)
        subset[category] = sorted(ids)
        print(f"{category:22s} {len(ids):5d} / {len(rows)}")

    print(f"{'total':22s} {sum(map(len, subset.values())):5d}")
    with open(args.out, "w") as f:
        json.dump({"cap": args.cap, "seed": args.seed, "ids": subset}, f, indent=1)


if __name__ == "__main__":
    main()
