"""Write a local progress dashboard: CSVs of the run grid and the GPU machines, plus a static HTML page.

    uv run python scripts/eval/dashboard.py                 # update once
    uv run python scripts/eval/dashboard.py --every 60      # update every minute (Ctrl-C to stop)
    open results/dashboard/index.html

Grid status comes from every run's meta.json in HF_RESULTS_REPO (small files). The live percentage of the run each
machine is scoring, its current step, tmux sessions and GPU use come from SSH (hosts in MACHINES, via ~/.ssh/config).
Outputs, all under results/ (git-ignored):
  results/dashboard/progress.csv   one row per (experiment, model, variant, language)
  results/dashboard/extras.csv     runs outside the grid (pilots, noise-floor reruns)
  results/dashboard/machines.csv   one row per machine
  results/dashboard/data.js        the same CSV text, so the page works from file:// (browsers block fetch there)
  results/dashboard/index.html
"""
import argparse
import csv
import io
import json
import os
import re
import subprocess
import time

import common
import hub

MACHINES = {"vast": "M1", "vast2": "M2", "vast3": "M3"}
ON_HOLD = {"exp2", "exp2_optional"}
OUT = "results/dashboard"
REMOTE = r"""cd /root/low_compute_fairness 2>/dev/null || exit 3
echo "GPU|$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1)"
echo "TMUX|$(tmux ls 2>/dev/null | cut -d: -f1 | tr '\n' ' ')"
for f in $(ls -t logs/*.log 2>/dev/null | head -3); do
  echo "LOG|$f|$(stat -c %Y "$f")"
  echo "STEP|$(grep -a '^===' "$f" | tail -1)"
  echo "RUN|$(grep -a '^\[[^] ]*__' "$f" | tail -1)"
  echo "ROWS|$(tail -c 400 "$f" | tr '\r' '\n' | grep -a 'rows/s\|block ' | tail -1)"
done"""


def ssh(host):
    try:
        out = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", host, REMOTE],
                             capture_output=True, text=True, timeout=60).stdout
    except subprocess.TimeoutExpired:
        return None
    return out if "GPU|" in out else None


def machine_status(host, name):
    out = ssh(host)
    row = {"machine": name, "host": host, "reachable": "no", "gpu": "", "tmux": "", "step": "", "run": "",
           "progress": "", "log": "", "log_age_min": ""}
    if out is None:
        return row, None
    row["reachable"] = "yes"
    logs = []
    for line in out.splitlines():
        key, _, val = line.partition("|")
        if key == "GPU":
            row["gpu"] = val.strip()
        elif key == "TMUX":
            row["tmux"] = val.strip()
        elif key == "LOG":
            path, _, mtime = val.partition("|")
            logs.append({"log": path, "mtime": int(mtime or 0)})
        elif logs and key in ("STEP", "RUN", "ROWS"):
            logs[-1][key] = val.strip()
    # The active log is the newest one; waiting watchers write rarely, so it is the running queue.
    live = None
    if logs:
        cur = logs[0]
        row["log"] = cur["log"]
        row["log_age_min"] = round((time.time() - cur["mtime"]) / 60, 1)
        step = re.sub(r"^=== \S+ \S+ ", "", cur.get("STEP", ""))
        row["step"] = re.sub(r"uv run python -u scripts/\S+/", "", step)[:160]
        run = cur.get("RUN", "")
        row["run"] = run[:160]
        row["progress"] = cur.get("ROWS", "")
        m = re.match(r"\[([^\]]+)\] (\d+)/(\d+) rows done", run)
        p = re.match(r"\s*(\d+)/(\d+) rows", row["progress"])
        if m and p and "step exit" not in cur.get("STEP", ""):
            live = (m.group(1), int(p.group(1)), int(p.group(2)))
    return row, live


def run_state(run):
    path = os.path.join(common.RUNS_DIR, run, "meta.json")
    if not os.path.exists(path):
        return "pending", ""
    meta = json.load(open(path, encoding="utf-8"))
    host = meta["sessions"][-1]["host"] if meta.get("sessions") else meta.get("host", "")
    return ("done" if meta.get("status") == "complete" else "partial"), host


def to_csv(rows, fields):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def update():
    common.load_env()
    if hub.repo():
        hub.fetch_meta(hub.repo())
    machines, live = [], {}
    for host, name in MACHINES.items():
        row, lv = machine_status(host, name)
        machines.append(row)
        if lv:
            live[lv[0]] = (name, lv[1], lv[2])

    grid, in_grid = [], set()
    for exp, model, variant, lang in common.grid():
        run = common.run_id(model, variant, lang)
        in_grid.add(run)
        state, host = run_state(run)
        pct = "100" if state == "done" else ""
        where = ""
        if run in live and state != "done":
            name, done, total = live[run]
            state, pct, where = "running", str(100 * done // total), name
        elif state == "pending" and exp in ON_HOLD:
            state = "on hold"
        grid.append({"exp": exp, "model": model, "variant": variant, "lang": lang, "state": state,
                     "pct": pct, "machine": where})

    extras = []
    for run in sorted(os.listdir(common.RUNS_DIR)) if os.path.isdir(common.RUNS_DIR) else []:
        if run not in in_grid and os.path.exists(os.path.join(common.RUNS_DIR, run, "meta.json")):
            state, _ = run_state(run)
            if run in live and state != "done":
                state = f"running ({live[run][0]}, {100 * live[run][1] // live[run][2]}%)"
            extras.append({"run": run, "state": state})

    os.makedirs(OUT, exist_ok=True)
    texts = {"progress": to_csv(grid, ["exp", "model", "variant", "lang", "state", "pct", "machine"]),
             "extras": to_csv(extras, ["run", "state"]),
             "machines": to_csv(machines, list(machines[0]))}
    for name, text in texts.items():
        open(os.path.join(OUT, f"{name}.csv"), "w", encoding="utf-8").write(text)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    open(os.path.join(OUT, "data.js"), "w", encoding="utf-8").write(
        "window.DASH = " + json.dumps({"updated": stamp, **texts}) + ";\n")
    page = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(open(page, encoding="utf-8").read())
    n_done = sum(r["state"] == "done" for r in grid)
    print(f"{stamp}: {n_done}/{len(grid)} grid runs done; "
          + ", ".join(f"{m['machine']} {'up' if m['reachable'] == 'yes' else 'DOWN'}" for m in machines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, metavar="SECONDS", help="repeat every SECONDS")
    args = ap.parse_args()
    while True:
        try:
            update()
        except Exception as e:  # keep the loop alive through network errors
            print(f"{time.strftime('%H:%M:%S')}: update failed: {e!r}", flush=True)
        if not args.every:
            break
        time.sleep(args.every)


if __name__ == "__main__":
    main()
