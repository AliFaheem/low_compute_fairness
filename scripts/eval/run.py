"""Score a model variant on PakBBQ: one forward pass per prompt, probabilities of the 3 answer labels.

    uv run python scripts/eval/run.py --model llama-3.1-8b --variant nf4 --lang en ur roman_ur
    uv run python scripts/eval/run.py --machine lab-4090          # every grid run assigned to this machine
    uv run python scripts/eval/run.py --model smoke-qwen3-0.6b --variant bf16 --lang en --limit 30 --device cpu

Each run writes results/runs/<model>__<variant>__<lang>/ (results/smoke/... with --limit):
  preds.jsonl  one row per (item, option order): log-probs of the 3 labels in ORIGINAL option order
               (ans0, ans1, ans2), `mass`, the full-vocabulary probability on the 3 labels, and `top`,
               the 10 most likely next tokens [id, text, log-prob] (what the model wanted to say)
  meta.json    model revision, variant settings, library versions, GPU, data and prompt hashes, status

Results are uploaded to the private HF dataset HF_RESULTS_REPO every few minutes and at the end
(scripts/eval/hub.py; --no-upload to skip). A run with no local files first downloads whatever the
dataset has for it, so a run interrupted on one machine continues on another.
Runs are resumable: re-running the same command skips rows already in preds.jsonl. A run is
`complete` when it has every item in all 3 option orders. The model is loaded once per
(model, variant) and all requested languages are scored with it.
"""
import argparse
import datetime
import fcntl
import json
import os
import platform
import socket
import subprocess
import sys
import time

import torch

import common
import hub

MIN_BUDGET = 512
TOP_K = 10  # next tokens stored per row (token id, text, log-prob) for debugging off-label answers


def git_state():
    def git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    dirty = [l for l in git("status", "--porcelain").splitlines() if not l[3:].startswith("results/")]
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(dirty)}


def model_source(spec, variant):
    if variant.startswith("wanda"):  # wanda50, wanda75, ...: pruned copy listed under that key
        src = spec.get(variant)
        if not src:
            raise SystemExit(f"{spec['hf_id']}: no {variant} checkpoint in configs/models.yaml yet")
        return src["hf_id"], src["revision"]
    return spec["hf_id"], spec["revision"]


def quant_config(variant):
    from transformers import BitsAndBytesConfig
    if variant == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if variant == "nf4":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False)
    return None


def load_model(spec, variant, device):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    hf_id, revision = model_source(spec, variant)
    qc = quant_config(variant)
    kw = dict(revision=revision, dtype=torch.bfloat16, device_map=device)
    if qc is not None:
        kw["quantization_config"] = qc
    try:
        model = AutoModelForCausalLM.from_pretrained(hf_id, **kw)
    except ValueError:  # multimodal checkpoints (Gemma 4); only text is ever passed in
        model = AutoModelForImageTextToText.from_pretrained(hf_id, **kw)
    model.eval()
    tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok, qc


def check_quantized(model, variant):
    """Count linear layers by type, and fail if a quantized variant has no quantized layers."""
    counts = {}
    for m in model.modules():
        name = type(m).__name__
        if name in ("Linear", "Linear8bitLt", "Linear4bit"):
            counts[name] = counts.get(name, 0) + 1
    want = {"int8": "Linear8bitLt", "nf4": "Linear4bit"}.get(variant)
    if want and not counts.get(want):
        raise SystemExit(f"{variant}: no {want} layers after loading; quantization did not apply")
    return counts


def label_token_ids(tok, prompts, labels):
    """Token id of each label as the next token after the prompt; must be 1 token, distinct, stable."""
    for sep in (" ", ""):
        ids, ok = None, True
        for prompt in prompts:
            base = tok(prompt).input_ids
            conts = []
            for label in labels:
                full = tok(prompt + sep + label).input_ids
                if full[:len(base)] != base or len(full) != len(base) + 1:
                    ok = False
                    break
                conts.append(full[-1])
            if not ok or len(set(conts)) != 3 or (ids is not None and conts != ids):
                ok = False
                break
            ids = conts
        if ok:
            return sep, ids
    raise SystemExit(f"Labels {labels} are not single, stable tokens for this tokenizer")


def last_token_logprobs(model, enc):
    mask = enc["attention_mask"]
    pos = (mask.cumsum(-1) - 1).clamp(min=0)  # left padding: real tokens start at position 0
    try:
        out = model(**enc, position_ids=pos, logits_to_keep=1)
    except TypeError:
        out = model(**enc, position_ids=pos)
    return out.logits[:, -1, :].float().log_softmax(-1)


def batches(lengths, budget):
    """Index batches, longest first, with (batch size x longest prompt) <= budget tokens."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    i = 0
    while i < len(order):
        n = max(1, budget // lengths[order[i]])
        yield order[i:i + n]
        i += n


def read_done(path):
    """(id, order) pairs already scored; drops a half-written last line."""
    if not os.path.exists(path):
        return set()
    lines = open(path, encoding="utf-8").read().split("\n")
    rows, good = [], []
    for line in lines:
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
            good.append(line)
        except ValueError:
            break
    if len(good) != len([l for l in lines if l.strip()]):
        with open(path, "w", encoding="utf-8") as f:
            f.write("".join(l + "\n" for l in good))
    return {(r["id"], r["order"]) for r in rows}


def score_language(model, tok, lang, out_dir, args, meta_base):
    repo = "" if args.no_upload else hub.repo()
    os.makedirs(out_dir, exist_ok=True)
    lock = open(os.path.join(out_dir, ".lock"), "w")
    try:  # two processes appending to one preds.jsonl would duplicate rows
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"{out_dir} is being scored by another process on this machine")
    t = common.load_template(lang)
    items = common.sample_items(common.load_items(lang), args.limit)
    preds_path = os.path.join(out_dir, "preds.jsonl")
    meta_path = os.path.join(out_dir, "meta.json")
    done = read_done(preds_path)
    todo = [(it, o) for it in items for o in range(common.ORDERS) if (it["id"], o) not in done]
    total = len(items) * common.ORDERS

    sep, label_ids = label_token_ids(tok, [common.build_prompt(it, t, 0) for it in items[:5]], t["labels"])
    prompts = [common.build_prompt(it, t, o) for it, o in todo]
    lengths = [len(tok(p).input_ids) for p in prompts]

    meta = dict(meta_base, lang=lang, n_items=len(items), n_rows_expected=total,
                labels=t["labels"], label_sep=sep, label_token_ids=label_ids,
                data_sha256=common.sha256_files(common.data_files(lang)), prompt_sha256=common.prompt_hash(lang),
                max_prompt_tokens=max(lengths, default=None))
    if os.path.exists(meta_path):
        old = json.load(open(meta_path, encoding="utf-8"))
        for key in ("model_revision", "data_sha256", "prompt_sha256", "label_token_ids"):
            if old.get(key) != meta[key]:
                raise SystemExit(f"{out_dir}: {key} differs from the existing partial run; delete it to restart")
        meta["sessions"] = old.get("sessions", [])
    else:
        meta["sessions"] = []
    session = {"host": meta_base["host"], "started": now(), "rows_before": len(done), "token_budget": args.token_budget}
    meta["sessions"].append(session)
    meta["status"] = "partial"
    write_json(meta_path, meta)

    print(f"[{os.path.basename(out_dir)}] {len(done)}/{total} rows done, {len(todo)} to score", flush=True)
    uploader = hub.Uploader(repo, out_dir) if repo else hub.NoUploader()
    budget, written, t0 = args.token_budget, 0, time.time()
    try:
        budget, written = score_rows(model, tok, todo, prompts, lengths, label_ids, preds_path, uploader, budget,
                                     len(done), total, t0)
    finally:
        n_rows = len(read_done(preds_path))
        session.update(ended=now(), rows_after=n_rows, final_token_budget=budget, seconds=round(time.time() - t0, 1))
        if torch.cuda.is_available():
            session["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        meta["status"] = "complete" if n_rows == total else "partial"
        with uploader.lock:
            write_json(meta_path, meta)
        uploader.finish()  # final upload, also after a crash
        print(f"[{os.path.basename(out_dir)}] {meta['status']}: {n_rows}/{total} rows"
              + (f", uploaded to {repo}" if repo else ""), flush=True)


def score_rows(model, tok, todo, prompts, lengths, label_ids, preds_path, uploader, budget, n_done, total, t0):
    written = 0
    with open(preds_path, "a", encoding="utf-8") as f:
        pending = list(batches(lengths, budget))
        while pending:
            idx = pending.pop(0)
            try:
                enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=True)
                enc = {k: v.to(model.device) for k, v in enc.items()}
                with torch.inference_mode():
                    full = last_token_logprobs(model, enc)
                    lp = full[:, label_ids].cpu()
                    top_v, top_i = (t.cpu() for t in full.topk(TOP_K, dim=-1))  # what the model wanted to say
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if budget <= MIN_BUDGET or len(idx) == 1:
                    raise
                budget //= 2
                print(f"  out of memory; token budget -> {budget}", flush=True)
                rest = idx + [i for b in pending for i in b]
                sub = list(batches([lengths[i] for i in rest], budget))
                pending = [[rest[j] for j in b] for b in sub]
                continue
            lines = []
            for row, i in enumerate(idx):
                item, order = todo[i]
                shown = common.shown_options(order)
                orig = [0.0] * 3
                for pos in range(3):
                    orig[shown[pos]] = round(lp[row, pos].item(), 6)
                top = [[int(t), tok.decode([int(t)]), round(float(v), 4)] for v, t in zip(top_v[row], top_i[row])]
                lines.append(json.dumps({"id": item["id"], "order": order, "logp": orig,
                                         "mass": round(float(lp[row].exp().sum()), 6), "top": top},
                                        ensure_ascii=False) + "\n")
            with uploader.lock:  # never upload a half-written batch
                f.writelines(lines)
                f.flush()
            written += len(idx)
            if written % 2000 < len(idx) or not pending:
                rate = written / (time.time() - t0)
                print(f"  {n_done + written}/{total} rows, {rate:.1f} rows/s", flush=True)
    return budget, written


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def planned_runs(args):
    """(model, variant, [langs]) groups to run, from --machine or --model/--variant/--lang."""
    if args.machine:
        assigned = common.load_runs()["machines"].get(args.machine)
        if not assigned:
            raise SystemExit(f"No models assigned to machine {args.machine!r} in configs/runs.yaml")
        groups = {}
        for _, model, variant, lang in common.grid():
            if model in assigned:
                groups.setdefault((model, variant), []).append(lang)
        return [(m, v, langs) for (m, v), langs in groups.items()]
    if not (args.model and args.variant and args.lang):
        raise SystemExit("Give --machine, or all of --model, --variant and --lang")
    return [(args.model, args.variant, args.lang)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", help="run every grid run assigned to this machine in configs/runs.yaml")
    ap.add_argument("--model")
    ap.add_argument("--variant")
    ap.add_argument("--lang", nargs="+", choices=["en", "ur", "roman_ur"])
    ap.add_argument("--limit", type=int, help="score an evenly spaced sample of N items (smoke test)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-upload", action="store_true", help="do not upload results to HF_RESULTS_REPO")
    ap.add_argument("--token-budget", type=int, default=16384,
                    help="max (batch size x prompt length) tokens per forward pass; halved on out-of-memory")
    args = ap.parse_args()
    common.load_env()
    cfg = common.load_models()

    for model_key, variant, langs in planned_runs(args):
        spec = cfg["models"].get(model_key)
        if spec is None or variant not in cfg["variants"]:
            raise SystemExit(f"Unknown model {model_key!r} or variant {variant!r} (see configs/models.yaml)")
        root = common.SMOKE_DIR if args.limit else common.RUNS_DIR
        dirs = {l: os.path.join(root, common.run_id(model_key, variant, l) + (f"__n{args.limit}" if args.limit else ""))
                for l in langs}
        repo = "" if args.no_upload else hub.repo()
        if repo:
            hub.ensure_private(repo)
            for l in langs:
                if not os.path.exists(os.path.join(dirs[l], "preds.jsonl")):
                    hub.fetch_run(repo, dirs[l])  # finished or partial on another machine
        todo = [l for l in langs if not is_complete(dirs[l])]
        if not todo:
            print(f"{model_key} {variant}: all requested languages complete")
            continue
        print(f"Loading {model_key} ({variant}) ...", flush=True)
        t0 = time.time()
        model, tok, qc = load_model(spec, variant, args.device)
        hf_id, revision = model_source(spec, variant)
        import transformers
        versions = {"python": platform.python_version(), "torch": torch.__version__,
                    "transformers": transformers.__version__, "cuda": torch.version.cuda}
        try:
            import bitsandbytes
            versions["bitsandbytes"] = bitsandbytes.__version__
        except ImportError:
            pass
        meta_base = {
            "model": model_key, "variant": variant, "model_id": hf_id, "model_revision": revision,
            "quantization_config": qc.to_dict() if qc is not None else None,
            "linear_layers": check_quantized(model, variant),
            "memory_footprint_gb": round(model.get_memory_footprint() / 1e9, 3),
            "load_seconds": round(time.time() - t0, 1), "limit": args.limit,
            "device": args.device, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "host": socket.gethostname(), "versions": versions, "git": git_state(),
            "command": " ".join(sys.argv),
        }
        print(f"  loaded in {meta_base['load_seconds']}s, {meta_base['memory_footprint_gb']} GB, "
              f"layers {meta_base['linear_layers']}", flush=True)
        for lang in todo:
            score_language(model, tok, lang, dirs[lang], args, meta_base)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def is_complete(out_dir):
    path = os.path.join(out_dir, "meta.json")
    return os.path.exists(path) and json.load(open(path, encoding="utf-8")).get("status") == "complete"


if __name__ == "__main__":
    main()
