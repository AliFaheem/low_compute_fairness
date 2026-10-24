"""Quick accuracy check of finished runs (the full metrics are a separate analysis step).

    uv run python scripts/eval/accuracy.py            # complete runs in results/runs/
    uv run python scripts/eval/accuracy.py --fetch    # first download all runs from HF_RESULTS_REPO
    uv run python scripts/eval/accuracy.py --smoke    # runs in results/smoke/

Final answer per item = argmax of the label probabilities averaged over the 3 option orders.
Also shows accuracy of each single order, how often all 3 orders agree, and the median
probability mass on the 3 labels (low mass means the model is not following the format).
"""
import argparse
import csv
import glob
import json
import math
import os
import statistics

import common
import hub


def softmax(logp):
    m = max(logp)
    e = [math.exp(x - m) for x in logp]
    return [x / sum(e) for x in e]


def argmax(xs):
    return max(range(len(xs)), key=xs.__getitem__)


def summarize(run_dir, items):
    meta = json.load(open(os.path.join(run_dir, "meta.json"), encoding="utf-8"))
    by_id = {}
    masses = []
    for line in open(os.path.join(run_dir, "preds.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        by_id.setdefault(r["id"], {})[r["order"]] = softmax(r["logp"])
        masses.append(r["mass"])
    hits = {"all": [], "ambig": [], "disambig": []}
    per_order = [[] for _ in range(common.ORDERS)]
    agree = []
    for item_id, orders in by_id.items():
        item = items[item_id]
        mean = [sum(p[k] for p in orders.values()) / len(orders) for k in range(3)]
        ok = argmax(mean) == item["label"]
        hits["all"].append(ok)
        hits[item["context_condition"]].append(ok)
        for o, p in orders.items():
            per_order[o].append(argmax(p) == item["label"])
        agree.append(len({argmax(p) for p in orders.values()}) == 1)
    pct = lambda xs: round(100 * sum(xs) / len(xs), 1) if xs else None
    return {"run": os.path.basename(run_dir), "status": meta["status"], "items": len(by_id),
            "acc": pct(hits["all"]), "acc_ambig": pct(hits["ambig"]), "acc_disambig": pct(hits["disambig"]),
            **{f"acc_order{o}": pct(per_order[o]) for o in range(common.ORDERS)},
            "orders_agree": pct(agree), "median_mass": round(statistics.median(masses), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    common.load_env()
    if args.fetch and hub.repo():
        hub.fetch_all(hub.repo())
    root = common.SMOKE_DIR if args.smoke else common.RUNS_DIR
    items = {it["id"]: it for it in common.load_items("en")}  # labels/metadata are identical in every split
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(root, "*"))):
        if os.path.exists(os.path.join(run_dir, "meta.json")) and os.path.exists(os.path.join(run_dir, "preds.jsonl")):
            if args.smoke or json.load(open(os.path.join(run_dir, "meta.json"))).get("status") == "complete":
                rows.append(summarize(run_dir, items))
    if not rows:
        raise SystemExit(f"No runs in {root}")
    cols = list(rows[0])
    widths = [max(len(str(r[c])) for r in rows + [dict(zip(cols, cols))]) for c in cols]
    for r in [dict(zip(cols, cols))] + rows:
        print("  ".join(str(r[c]).ljust(w) for c, w in zip(cols, widths)))
    out = "results/summary/accuracy_smoke.csv" if args.smoke else "results/summary/accuracy.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
