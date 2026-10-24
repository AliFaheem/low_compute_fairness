"""Show progress of the experiment grid across all machines.

    uv run python scripts/eval/status.py           # fetches every run's meta.json from HF_RESULTS_REPO first
    uv run python scripts/eval/status.py --local   # only what is on this machine
    uv run python scripts/eval/status.py --watch   # refresh every minute (Ctrl-C to stop)

Each cell is one run: `done`, rows scored so far for a partial run, or `-` if not started.
"""
import argparse
import json
import os
import time

import common
import hub


def cell(out_dir):
    path = os.path.join(out_dir, "meta.json")
    if not os.path.exists(path):
        return "-", None
    meta = json.load(open(path, encoding="utf-8"))
    host = meta["sessions"][-1]["host"] if meta.get("sessions") else meta.get("host")
    if meta.get("status") == "complete":
        return "done", host
    preds = os.path.join(out_dir, "preds.jsonl")
    rows = sum(1 for _ in open(preds, encoding="utf-8")) if os.path.exists(preds) else 0
    return f"{100 * rows // meta['n_rows_expected']}%", host


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--watch", type=int, nargs="?", const=60, metavar="SECONDS",
                    help="redraw every SECONDS (default 60)")
    args = ap.parse_args()
    common.load_env()
    if args.watch:
        while True:
            print("\033[2J\033[H" + time.strftime("%H:%M:%S"), flush=True)
            show(args)
            time.sleep(args.watch)
    show(args)


def show(args):
    if not args.local and hub.repo():
        hub.fetch_meta(hub.repo())
        hub.snapshot_download(hub.repo(), repo_type="dataset", local_dir="results",
                              allow_patterns=["runs/*/preds.jsonl"])  # partial counts; small files

    langs = common.load_runs()["languages"]
    machines = {m: mach for mach, models in (common.load_runs()["machines"] or {}).items() for m in models}
    rows, n_done, n_all = [], 0, 0
    seen = set()
    for exp, model, variant, _ in common.grid():
        if (model, variant) in seen:
            continue
        seen.add((model, variant))
        cells = [cell(os.path.join(common.RUNS_DIR, common.run_id(model, variant, l))) for l in langs]
        n_done += sum(c == "done" for c, _ in cells)
        n_all += len(cells)
        hosts = sorted({h for _, h in cells if h})
        rows.append([exp, model, variant] + [c for c, _ in cells] + [machines.get(model, ""), ", ".join(hosts)])

    header = ["exp", "model", "variant"] + langs + ["assigned", "ran on"]
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    for r in [header] + rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(r, widths)))
    print(f"\n{n_done}/{n_all} runs done")


if __name__ == "__main__":
    main()
