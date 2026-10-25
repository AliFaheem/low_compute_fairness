"""AWQ 4-bit quantization (group size 128) with the mixed en/ur/roman_ur calibration text, via gptqmodel.

Runs in the separate AWQ venv (gptqmodel 7.7; installing it in the main env would change the torch pins):

    /root/awq_test/.venv/bin/python scripts/compression/awq_quant.py --model qwen3-8b \
        --calib-file logs/calibration/qalb-1.0-8b-wanda50mix.jsonl

Calibration: the documents of a Wanda mixed-calibration log (1/3 C4 en, 1/3 FineWeb-2 Urdu, 1/3 Roman Urdu), so
AWQ and wanda*mix see the same text. Each source's documents are tokenized with the target model's tokenizer,
concatenated and cut into SEQLEN-token samples; NSAMPLES are taken, split evenly across the 3 sources.
Afterwards perplexity is measured on the same held-out windows as wanda_prune.py (16 x 2048, seed 0), so it
compares directly with the "before" values in logs/wanda_meta/. Writes models/<model>-awq/ (local, never
uploaded) with awq_meta.json; score it with run.py --variant awq under the same venv.
"""
import argparse
import collections
import datetime
import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import wanda_prune as wp  # noqa: E402

SOURCES = ("en", "ur", "roman_ur")


def calibration(tok, path, nsamples, seqlen):
    docs = collections.defaultdict(list)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        docs[r["source"]].append(r["text"])
    samples, counts = [], {}
    for i, src in enumerate(SOURCES):
        want = nsamples // len(SOURCES) + (i < nsamples % len(SOURCES))
        ids = []
        for text in docs[src]:
            ids += tok(text, add_special_tokens=False).input_ids + [tok.eos_token_id]
        chunks = [ids[j:j + seqlen] for j in range(0, len(ids) - seqlen + 1, seqlen)][:want]
        if len(chunks) < want:
            raise SystemExit(f"{path}: only {len(chunks)} x {seqlen} tokens of {src}, need {want}")
        samples += [{"input_ids": c, "attention_mask": [1] * seqlen} for c in chunks]
        counts[src] = len(chunks)
    return samples, counts


def main():
    from gptqmodel import GPTQModel, QuantizeConfig
    from gptqmodel.quantization import FORMAT, METHOD
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib-file", required=True, help="a logs/calibration/<model>-wanda*mix.jsonl file")
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--eval-samples", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", help="output dir (default models/<model>-awq)")
    args = ap.parse_args()
    common.load_env()
    spec = common.load_models()["models"][args.model]
    out = args.out or f"models/{args.model}-awq"
    if os.path.exists(os.path.join(out, "awq_meta.json")):
        print(f"{out} already quantized")
        return

    tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=spec["revision"])
    calib, counts = calibration(tok, args.calib_file, args.nsamples, args.seqlen)
    print(f"calibration: {counts} x {args.seqlen} tokens from {args.calib_file}", flush=True)

    qc = QuantizeConfig(bits=args.bits, group_size=args.group_size, method=METHOD.AWQ, format=FORMAT.GEMM)
    t0 = time.time()
    # GPTQModel.load passes extra kwargs to the model constructor (no `revision`), so load the pinned snapshot by path.
    src = spec["hf_id"]
    if spec["revision"]:
        from huggingface_hub import snapshot_download
        src = snapshot_download(spec["hf_id"], revision=spec["revision"])
    model = GPTQModel.load(src, qc)
    model.quantize(calib, batch_size=1, tokenizer=tok)
    model.save(out)
    tok.save_pretrained(out)
    minutes = round((time.time() - t0) / 60, 1)
    del model
    torch.cuda.empty_cache()
    print(f"quantized and saved to {out} in {minutes} min", flush=True)

    # Perplexity of the saved checkpoint, loaded the way run.py loads it.
    ppl = {}
    try:
        model = GPTQModel.load(out, device=args.device)
        hf = getattr(model, "model", model)
        evals = {"en": wp.c4_windows(tok, wp.C4_VALID, args.eval_samples, 2048, 0),
                 "ur": wp.packed_windows(tok, wp.urdu_split(0)[1], args.eval_samples, 2048, 0),
                 "roman_ur": wp.packed_windows(tok, wp.fw2_docs("latn_test"), args.eval_samples, 2048, 0)}
        with torch.no_grad():
            ppl = {lang: {"after": wp.perplexity(hf, w, args.device), "windows": args.eval_samples, "seqlen": 2048}
                   for lang, w in evals.items()}
        print(f"perplexity after: { {l: v['after'] for l, v in ppl.items()} }", flush=True)
    except Exception as e:  # best-effort, like wanda_prune's extra languages
        print(f"perplexity skipped: {e!r}", flush=True)

    import gptqmodel
    import transformers
    meta = {"source_model": spec["hf_id"], "source_revision": spec["revision"], "method": "awq",
            "bits": args.bits, "group_size": args.group_size, "format": "gemm",
            "calibration": {"file": args.calib_file,
                            "sha256": hashlib.sha256(open(args.calib_file, "rb").read()).hexdigest(),
                            "samples_per_source": counts, "seqlen": args.seqlen},
            "perplexity": ppl, "minutes": minutes,
            "versions": {"gptqmodel": gptqmodel.__version__, "torch": torch.__version__,
                         "transformers": transformers.__version__},
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    json.dump(meta, open(os.path.join(out, "awq_meta.json"), "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
