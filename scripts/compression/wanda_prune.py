"""Prune a model with Wanda (Sun et al., ICLR 2024) and optionally push it to a private HF repo.

    uv run python scripts/compression/wanda_prune.py --model llama-3.1-8b --push alifaheem94/llama-3.1-8b-wanda50
    uv run python scripts/compression/wanda_prune.py --model smoke-qwen3-0.6b --device cpu --nsamples 8 --seqlen 256

Wanda removes, in every linear layer of every decoder block, the weights with the smallest
|W_ij| * ||X_j||_2, where ||X_j|| is the L2 norm of input feature j over the calibration tokens.
Weights compete within each output row (50% of each row is removed), or in groups of 4 with
--structure 2:4. No retraining, no weight update.

Blocks are pruned in order, and the input statistics of block l are collected with blocks 0..l-1
already pruned, as in the official implementation. Statistics are collected by running the whole
model on the calibration data and stopping after block l, so no architecture-specific code is needed
(Gemma 4's per-layer inputs and shared KV caches work as-is). Only decoder blocks of the language
model are pruned: embeddings, lm_head, and vision/audio towers stay dense.

Calibration follows the paper: 128 random 2048-token windows from C4 (en, train shard 0), seed 0.
Perplexity on C4 validation is measured before and after as a sanity check.
The pruned model is saved in bf16 with its tokenizer and wanda_meta.json, then (with --push)
uploaded to a private model repo; put its hf_id and revision under `wanda50` in configs/models.yaml.
"""
import argparse
import datetime
import gzip
import json
import os
import random
import re
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
import common  # noqa: E402

C4_REPO = "allenai/c4"
C4_TRAIN = "en/c4-train.00000-of-01024.json.gz"
C4_VALID = "en/c4-validation.00000-of-00008.json.gz"
# FineWeb-2 Urdu: urd_Arab = Urdu script, urd_Latn = Roman Urdu. The urd_Arab train shard is 4.8 GB, so
# Urdu script uses its 71 MB test shard, split into disjoint calibration / perplexity halves.
FW2_REPO = "HuggingFaceFW/fineweb-2"
FW2_FILES = {"arab_test": "data/urd_Arab/test/000_00000.parquet",
             "latn_train": "data/urd_Latn/train/000_00000.parquet",
             "latn_test": "data/urd_Latn/test/000_00000.parquet"}
ROMAN_URDU_WORDS = {"hai", "hain", "ka", "ki", "ke", "aur", "mein", "main", "nahi", "nahin", "ko", "se", "ye", "yeh",
                    "wo", "woh", "kya", "bhi", "tha", "thi", "kar", "ho", "jo", "par", "sath", "apne"}


class StopForward(Exception):
    pass


def fw2_docs(key):
    """Texts of one FineWeb-2 Urdu shard; Roman Urdu keeps docs with >= 15% Roman Urdu function words."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    table = pq.read_table(hf_hub_download(FW2_REPO, FW2_FILES[key], repo_type="dataset"), columns=["id", "text"])
    docs = list(zip(table.column("id").to_pylist(), table.column("text").to_pylist()))
    if key.startswith("latn"):
        def ratio(s):
            w = re.findall(r"[a-zA-Z]+", s.lower())
            return sum(x in ROMAN_URDU_WORDS for x in w) / max(len(w), 1)
        docs = [d for d in docs if ratio(d[1]) >= 0.15]
    return docs  # [(fineweb-2 id, text)]


def packed_windows(tok, docs, n, seqlen, seed, record=None):
    """n seqlen-token windows from shuffled (id, text) docs joined with EOS (Urdu web docs are often short).
    If `record` is a list, the documents used are appended to it."""
    rng = random.Random(seed)
    docs = docs[:]
    rng.shuffle(docs)
    buf, out = [], []
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0
    for doc_id, text in docs:
        buf += tok(text, add_special_tokens=False).input_ids + [eos]
        if record is not None:
            record.append({"id": doc_id, "text": text})
        while len(buf) >= seqlen and len(out) < n:
            out.append(torch.tensor([buf[:seqlen]]))
            buf = buf[seqlen:]
        if len(out) == n:
            return out
    raise SystemExit(f"Not enough text for {n} windows of {seqlen} tokens (got {len(out)})")


def urdu_split(seed):
    """Urdu-script docs split into disjoint calibration and perplexity pools."""
    docs = fw2_docs("arab_test")
    idx = list(range(len(docs)))
    random.Random(seed).shuffle(idx)
    half = len(idx) // 2
    return [docs[i] for i in idx[:half]], [docs[i] for i in idx[half:]]


def c4_windows(tok, path, n, seqlen, seed, record=None):
    """n random seqlen-token windows from documents longer than seqlen (as in the Wanda code).
    If `record` is a list, each window's source document (url, token offset, text) is appended to it."""
    from huggingface_hub import hf_hub_download
    rows = [json.loads(l) for l in gzip.open(hf_hub_download(C4_REPO, path, repo_type="dataset"), "rt")]
    docs = [r["text"] for r in rows]
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        i = rng.randrange(len(docs))
        ids = tok(docs[i], return_tensors="pt").input_ids
        if ids.shape[1] <= seqlen:
            continue
        start = rng.randrange(ids.shape[1] - seqlen)
        out.append(ids[:, start:start + seqlen])
        if record is not None:
            record.append({"id": f"{path}#{i}", "url": rows[i].get("url"), "token_start": start, "text": docs[i]})
    return out


def save_calibration(directory, name, docs):
    """The exact calibration documents used, one JSON line each (source, file, id/url, text)."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(d, ensure_ascii=False) + "\n" for d in docs)
    print(f"  calibration documents: {len(docs)} -> {path}", flush=True)


def decoder_blocks(model):
    """The language model's list of decoder blocks: the longest ModuleList named `layers`."""
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and name.split(".")[-1] == "layers" \
                and not any(k in name for k in ("vision", "audio")):
            if best is None or len(mod) > len(best[1]):
                best = (name, mod)
    if best is None:
        raise SystemExit("Could not find the decoder blocks")
    return best


@torch.inference_mode()
def collect_norms(model, block, samples, device):
    """Sum over calibration tokens of x_j^2 for the input of every linear layer in `block`."""
    linears = {n: m for n, m in block.named_modules() if isinstance(m, torch.nn.Linear)}
    sums = {n: torch.zeros(m.in_features, dtype=torch.float32, device=device) for n, m in linears.items()}

    def hook(name):
        def fn(_, inputs, __):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
            sums[name] += (x * x).sum(0)
        return fn

    def stop(*_):
        raise StopForward

    handles = [m.register_forward_hook(hook(n)) for n, m in linears.items()]
    handles.append(block.register_forward_hook(stop))
    try:
        for ids in samples:
            try:
                model(input_ids=ids.to(device), use_cache=False)
            except StopForward:
                pass
    finally:
        for h in handles:
            h.remove()
    return linears, sums


@torch.inference_mode()
def prune_linear(lin, norm2, sparsity, structure):
    W = lin.weight.data
    metric = W.abs().float() * norm2.sqrt()[None, :]
    mask = torch.zeros_like(W, dtype=torch.bool)
    if structure == "2:4":
        groups = metric.reshape(W.shape[0], -1, 4)
        idx = groups.topk(2, dim=-1, largest=False).indices
        mask.view(W.shape[0], -1, 4).scatter_(-1, idx, True)
    else:
        k = int(W.shape[1] * sparsity)
        idx = metric.topk(k, dim=1, largest=False).indices
        mask.scatter_(1, idx, True)
    W[mask] = 0
    return mask.sum().item(), mask.numel()


@torch.inference_mode()
def perplexity(model, samples, device):
    nll, n = 0.0, 0
    for ids in samples:
        ids = ids.to(device)
        logits = model(input_ids=ids, use_cache=False).logits[:, :-1].float()
        nll += torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1),
                                                 reduction="sum").item()
        n += ids.shape[1] - 1
    return round(float(torch.exp(torch.tensor(nll / n))), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model key in configs/models.yaml")
    ap.add_argument("--sparsity", type=float, default=0.5)
    ap.add_argument("--structure", choices=["unstructured", "2:4"], default="unstructured")
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--eval-samples", type=int, default=16, help="windows per language for perplexity (0 = skip)")
    ap.add_argument("--calib", choices=["c4", "mix"], default="c4",
                    help="c4: English C4 (paper default); mix: 1/3 C4 en + 1/3 FineWeb-2 Urdu + 1/3 Roman Urdu")
    ap.add_argument("--eval-langs", nargs="+", default=["en", "ur", "roman_ur"],
                    help="perplexity languages; ur/roman_ur use FineWeb-2 held-out text and never fail the run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", help="output dir (default models/<model>-wanda<pct>)")
    ap.add_argument("--push", metavar="HF_REPO", help="upload to this private HF model repo")
    ap.add_argument("--no-save", action="store_true", help="only measure (write wanda_meta.json, not the weights)")
    ap.add_argument("--calib-log", default="logs/calibration", help="where to save the calibration documents used")
    args = ap.parse_args()
    if args.structure == "2:4":
        args.sparsity = 0.5
    common.load_env()
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    spec = common.load_models()["models"][args.model]
    tag = f"wanda{int(args.sparsity * 100)}" + ("-2of4" if args.structure == "2:4" else "") \
        + ("mix" if args.calib == "mix" else "")
    out = args.out or os.path.join("models", f"{args.model}-{tag}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=spec["revision"])
    kw = dict(revision=spec["revision"], dtype=torch.bfloat16, device_map=args.device)
    try:
        model = AutoModelForCausalLM.from_pretrained(spec["hf_id"], **kw)
    except ValueError:  # multimodal checkpoints (Gemma 4)
        model = AutoModelForImageTextToText.from_pretrained(spec["hf_id"], **kw)
    model.eval()

    ur_calib_pool = ur_eval_pool = None
    if args.calib == "mix":
        ur_calib_pool, ur_eval_pool = urdu_split(args.seed)
        n_en = args.nsamples - 2 * (args.nsamples // 3)
        rec_en, rec_ur, rec_ro = [], [], []
        calib = (c4_windows(tok, C4_TRAIN, n_en, args.seqlen, args.seed, rec_en)
                 + packed_windows(tok, ur_calib_pool, args.nsamples // 3, args.seqlen, args.seed, rec_ur)
                 + packed_windows(tok, fw2_docs("latn_train"), args.nsamples // 3, args.seqlen, args.seed, rec_ro))
        calib_docs = [dict(r, source="en", file=C4_TRAIN) for r in rec_en] \
            + [dict(r, source="ur", file=FW2_FILES["arab_test"]) for r in rec_ur] \
            + [dict(r, source="roman_ur", file=FW2_FILES["latn_train"]) for r in rec_ro]
        calib_record = {"mix": {"en": [C4_REPO, C4_TRAIN, n_en],
                                "ur": [FW2_REPO, FW2_FILES["arab_test"] + " (calibration half)", args.nsamples // 3],
                                "roman_ur": [FW2_REPO, FW2_FILES["latn_train"] + " (filtered)", args.nsamples // 3]}}
    else:
        rec_en = []
        calib = c4_windows(tok, C4_TRAIN, args.nsamples, args.seqlen, args.seed, rec_en)
        calib_record = {"dataset": C4_REPO, "file": C4_TRAIN}
        calib_docs = [dict(r, source="en", file=C4_TRAIN) for r in rec_en]
    save_calibration(args.calib_log, f"{args.model}-{tag}", calib_docs)

    evals = {}
    if args.eval_samples:
        evals["en"] = c4_windows(tok, C4_VALID, args.eval_samples, args.seqlen, args.seed)
        for lang in [l for l in args.eval_langs if l != "en"]:
            try:  # extra languages are best-effort: never let them fail the pruning
                if lang == "ur":
                    pool = ur_eval_pool if ur_eval_pool is not None else urdu_split(args.seed)[1]
                else:
                    pool = fw2_docs("latn_test")
                evals[lang] = packed_windows(tok, pool, args.eval_samples, args.seqlen, args.seed)
            except Exception as e:
                print(f"  perplexity for {lang} skipped: {e}", flush=True)
    ppl = {lang: {"before": perplexity(model, w, args.device)} for lang, w in evals.items()}
    ppl_before = ppl.get("en", {}).get("before")
    print(f"Loaded {args.model}; perplexity before: { {l: v['before'] for l, v in ppl.items()} }", flush=True)

    blocks_name, blocks = decoder_blocks(model)
    layers = {}
    for i, block in enumerate(blocks):
        linears, sums = collect_norms(model, block, calib, args.device)
        for name, lin in linears.items():
            zeros, total = prune_linear(lin, sums[name], args.sparsity, args.structure)
            layers[f"{blocks_name}.{i}.{name}"] = round(zeros / total, 4)
        print(f"  block {i + 1}/{len(blocks)}: pruned {len(linears)} linear layers", flush=True)

    for lang, w in evals.items():
        ppl[lang]["after"] = perplexity(model, w, args.device)
    ppl_after = ppl.get("en", {}).get("after")
    n_params = sum(p.numel() for p in model.parameters())
    n_pruned_layer_params = sum(m.weight.numel() for b in blocks for m in b.modules() if isinstance(m, torch.nn.Linear))
    zeros_total = sum(int((m.weight == 0).sum()) for b in blocks for m in b.modules() if isinstance(m, torch.nn.Linear))
    meta = {
        "source_model": spec["hf_id"], "source_revision": spec["revision"], "method": "wanda",
        "sparsity": args.sparsity, "structure": args.structure,
        "calibration": dict(calib_record, kind=args.calib, nsamples=args.nsamples, seqlen=args.seqlen, seed=args.seed),
        "perplexity": {lang: dict(v, windows=args.eval_samples, seqlen=args.seqlen) for lang, v in ppl.items()},
        "c4_validation_perplexity": {"before": ppl_before, "after": ppl_after, "windows": args.eval_samples,
                                     "seqlen": args.seqlen},
        "pruned_blocks": f"{blocks_name} ({len(blocks)} blocks)",
        "linear_layers_pruned": len(layers),
        "sparsity_in_pruned_layers": round(zeros_total / n_pruned_layer_params, 4),
        "sparsity_of_whole_model": round(zeros_total / n_params, 4),
        "min_max_layer_sparsity": [min(layers.values()), max(layers.values())],
        "seconds": round(time.time() - t0, 1), "device": args.device,
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__},
    }
    os.makedirs(out, exist_ok=True)
    if not args.no_save:
        model.save_pretrained(out)
        tok.save_pretrained(out)
    json.dump(dict(meta, layers=layers), open(os.path.join(out, "wanda_meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))
    print(f"Saved to {out}")

    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push, private=True, exist_ok=True)
        if not api.repo_info(args.push).private:
            raise SystemExit(f"{args.push} is public; refusing to upload")
        commit = api.upload_folder(repo_id=args.push, folder_path=out, commit_message=f"Wanda {tag} of {spec['hf_id']}")
        print(f"Pushed. In configs/models.yaml set:\n  {tag.replace('-', '_')}: {{hf_id: {args.push}, "
              f"revision: {commit.oid}}}")


if __name__ == "__main__":
    main()
