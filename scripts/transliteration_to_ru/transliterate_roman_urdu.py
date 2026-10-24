"""Transliterate the Urdu PakBBQ split into Roman Urdu with Gemini.

Output mirrors the input exactly: same file names, same records, same key
order; only the text fields (context, question, ans0-2) are replaced.

Calls Gemini on Vertex AI with the service account in credentials.json.

Resumable: every finished item is appended and flushed immediately, and a
re-run skips IDs already in the output. Retryable errors (rate limits, 5xx,
timeouts, bad model output) are retried, at most --retries times per item;
items that still fail are logged to failures.log and retried on later runs,
up to --max-failed-runs runs in total. Auth, permission or billing errors, or
--max-consecutive-failures failed items in a row, stop the run.

    uv run python scripts/transliteration_to_ru/transliterate_roman_urdu.py --limit 10   # test run
    uv run python scripts/transliteration_to_ru/transliterate_roman_urdu.py              # full run / resume
"""
import argparse
import collections
import concurrent.futures as cf
import glob
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request

from google.auth.transport.requests import Request
from google.oauth2 import service_account

FIELDS = ["context", "question", "ans0", "ans1", "ans2"]
ANSWERS = ["ans0", "ans1", "ans2"]
NON_LATIN = re.compile(r"[^\W\d_a-zA-Z]")  # any letter that is not plain a-z / A-Z
API_URL = ("https://aiplatform.googleapis.com/v1/projects/{project}/locations/global"
           "/publishers/google/models/{model}:generateContent")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PROMPT_FILE = os.path.join(REPO_ROOT, "prompts",
                           "roman_urdu_transliteration.txt")
SYSTEM_PROMPT = open(PROMPT_FILE, encoding="utf-8").read().strip()


class FatalError(Exception):
    """Stops the whole run (e.g. invalid API key)."""


class RetryableError(Exception):
    pass


class ItemFailed(Exception):
    """An item used up all its attempts."""

    def __init__(self, error, attempts):
        super().__init__(str(error))
        self.attempts = attempts


class VertexAuth:
    """Service-account credentials; refreshes the access token when it expires."""

    def __init__(self, path):
        if not os.path.exists(path):
            raise FatalError(f"credentials file not found: {path}")
        self.creds = service_account.Credentials.from_service_account_file(
            path, scopes=["https://www.googleapis.com/auth/cloud-platform"])
        self.project = self.creds.project_id
        self.lock = threading.Lock()

    def token(self):
        with self.lock:
            if not self.creds.valid:
                try:
                    self.creds.refresh(Request())
                except Exception as e:
                    raise FatalError(f"could not get access token: {e}")
            return self.creds.token


def call_gemini(fields, model, auth, temperature, feedback=None, timeout=120):
    text = json.dumps(fields, ensure_ascii=False)
    if feedback:
        text += f"\n\nYour previous answer was rejected: {feedback}. Return all five fields."
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {k: {"type": "STRING"} for k in FIELDS},
                "required": FIELDS,
            },
        },
        # PakBBQ is about stereotypes; default filters can block benign items.
        "safetySettings": [
            {"category": c, "threshold": "BLOCK_NONE"}
            for c in ("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                      "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")
        ],
    }
    req = urllib.request.Request(
        API_URL.format(project=auth.project, model=model),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {auth.token()}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            detail = json.loads(raw)["error"]["message"].strip()
        except (ValueError, KeyError, TypeError):
            detail = raw[:200].strip()
        if e.code in (401, 403):
            raise FatalError(f"auth/permission error (HTTP {e.code}): {detail}")
        if e.code == 402 or "billing" in detail.lower() or "credits" in detail.lower():
            raise FatalError(f"Billing problem (HTTP {e.code}): {detail}")
        if e.code == 404:
            raise FatalError(f"Model not found: {model}")
        raise RetryableError(f"HTTP {e.code}: {detail}")
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise RetryableError(f"network: {e}")

    try:
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        out = json.loads(text)
    except (KeyError, IndexError, json.JSONDecodeError):
        reason = data.get("promptFeedback", {}).get("blockReason") or \
            (data.get("candidates") or [{}])[0].get("finishReason")
        raise RetryableError(f"unusable response (reason: {reason})")

    for k in FIELDS:
        if not isinstance(out.get(k), str) or not out[k].strip():
            raise RetryableError(f"missing field {k}")
    return {k: out[k].strip() for k in FIELDS}


def non_latin_words(text):
    """Words still containing Urdu/Devanagari letters; accepted now, fixed by a later cleanup pass."""
    return [w for w in text.split() if NON_LATIN.search(w)]


class AnswerCache:
    """Same Urdu answer string -> same Roman spelling everywhere (persisted across runs)."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.map = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}

    def resolve(self, urdu, roman):
        """Cached spelling if any; otherwise cache this one, but never a spelling with non-Latin words."""
        with self.lock:
            if urdu not in self.map:
                if non_latin_words(roman):
                    return roman
                self.map[urdu] = roman
                tmp = self.path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self.map, f, ensure_ascii=False, indent=1)
                os.replace(tmp, self.path)
            return self.map[urdu]


def transliterate(row, model, auth, retries, cache, temperature, fallback=None):
    """Returns (new_row, attempts used, {field: non-Latin words}, model used). At most retries + 1 API calls.

    A rate limit (429) on the main model is retried at once on the fallback model, which has
    its own quota; if that is rate-limited too, wait and go back to the main model.
    """
    fields = {k: row[k] for k in FIELDS}
    feedback = None
    current = model
    for attempt in range(retries + 1):
        try:
            new = call_gemini(fields, current, auth, temperature, feedback)
            for k in ANSWERS:
                new[k] = cache.resolve(row[k], new[k])
            flags = {k: non_latin_words(new[k]) for k in FIELDS if non_latin_words(new[k])}
            return {k: new.get(k, v) for k, v in row.items()}, attempt + 1, flags, current
        except RetryableError as e:
            if attempt == retries:
                raise ItemFailed(e, attempt + 1)
            if str(e).startswith("missing field"):
                # Bad model output, not server load: retry at once with feedback.
                feedback = str(e).split(":")[0]
                print(f"  id={row['id']} attempt {attempt + 1} failed ({e}); retrying now", flush=True)
                continue
            if "HTTP 429" in str(e) and fallback and current != fallback:
                print(f"  id={row['id']} attempt {attempt + 1} rate-limited on {current}; "
                      f"retrying now on {fallback}", flush=True)
                current = fallback
                continue
            current = model
            wait = min(120, 2 ** attempt * (10 if "HTTP 429" in str(e) else 2)) + random.random()
            print(f"  id={row['id']} attempt {attempt + 1} failed ({e}); retry in {wait:.0f}s", flush=True)
            time.sleep(wait)


def read_done(path):
    """IDs already written; also drops a half-written last line."""
    if not os.path.exists(path):
        return set()
    with open(path, "rb+") as f:
        content = f.read()
        if content and not content.endswith(b"\n"):
            f.truncate(content.rfind(b"\n") + 1)
            content = content[: content.rfind(b"\n") + 1]
    return {json.loads(line)["id"] for line in content.decode().splitlines() if line.strip()}


def failed_runs(path):
    """How many runs each ID has already failed in (one failures.log line per failed run)."""
    counts = collections.Counter()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                counts[json.loads(line)["id"]] += 1
            except (ValueError, KeyError):
                pass
    return counts


def finalize(path, src_rows):
    """Once a file is complete, rewrite it in source order (workers finish out of order)."""
    by_id = {json.loads(line)["id"]: line for line in open(path, encoding="utf-8") if line.strip()}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(by_id[r["id"]] for r in src_rows)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/our_data/ur")
    ap.add_argument("--out", default="data/our_data/roman_ur")
    ap.add_argument("--work-dir", default="data/our_data/_build/roman_ur",
                    help="bookkeeping (meta, answer cache, failure/cleanup/fallback logs); keeps --out data-only")
    ap.add_argument("--files", default="*.jsonl", help="glob within --src, e.g. Regional.jsonl")
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--fallback-model", default="gemini-2.5-flash-lite",
                    help="used immediately when the main model is rate-limited; '' to disable")
    ap.add_argument("--round-robin", action="store_true",
                    help="alternate items between --model and --fallback-model (each falls back to the other)")
    ap.add_argument("--limit", type=int, default=None, help="max new items this run (for testing)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help=">0 gives natural spelling variety in context/question; answers stay consistent via the cache")
    ap.add_argument("--credentials", default="credentials.json")
    ap.add_argument("--log-every", type=int, default=10, help="print progress every N items")
    ap.add_argument("--max-failed-runs", type=int, default=3,
                    help="skip items that already failed in this many runs (see failures.log)")
    ap.add_argument("--retry-failed", action="store_true", help="ignore --max-failed-runs")
    ap.add_argument("--max-consecutive-failures", type=int, default=10,
                    help="stop if this many items in a row use up all attempts (systemic problem)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)
    with open(os.path.join(args.work_dir, "meta.json"), "w") as f:
        json.dump({"source": args.src, "model": args.model, "fallback_model": args.fallback_model,
                   "round_robin": args.round_robin,
                   "temperature": args.temperature,
                   "system_prompt": SYSTEM_PROMPT}, f, indent=1, ensure_ascii=False)

    try:
        auth = VertexAuth(args.credentials)
        auth.token()
    except FatalError as e:
        sys.exit(f"FATAL: {e}")

    budget = args.limit
    lock = threading.Lock()
    cache = AnswerCache(os.path.join(args.work_dir, "answer_cache.json"))
    failures_path = os.path.join(args.work_dir, "failures.log")
    past_failures = failed_runs(failures_path)
    failures = open(failures_path, "a", encoding="utf-8")
    # Items whose output still has Urdu/Devanagari words; a later pass replaces just those words.
    cleanup = open(os.path.join(args.work_dir, "needs_cleanup.log"), "a", encoding="utf-8")
    # Which items the fallback model produced (records themselves stay unchanged).
    models_log = open(os.path.join(args.work_dir, "fallback_items.log"), "a", encoding="utf-8")
    n_ok = n_fail = n_skipped = n_flagged = consecutive_fail = 0
    by_model = collections.Counter()
    attempts_hist = collections.Counter()

    for src_path in sorted(glob.glob(os.path.join(args.src, args.files))):
        name = os.path.basename(src_path)
        out_path = os.path.join(args.out, name)
        src_rows = [json.loads(line) for line in open(src_path, encoding="utf-8") if line.strip()]
        done = read_done(out_path)
        todo = [r for r in src_rows if r["id"] not in done]
        if not args.retry_failed:
            given_up = [r for r in todo if past_failures[r["id"]] >= args.max_failed_runs]
            todo = [r for r in todo if past_failures[r["id"]] < args.max_failed_runs]
            n_skipped += len(given_up)
        else:
            given_up = []
        if budget is not None:
            todo, budget = todo[:budget], budget - len(todo[:budget])
        print(f"{name}: {len(done)}/{len(src_rows)} done, {len(todo)} to do this run"
              + (f", {len(given_up)} skipped (failed in {args.max_failed_runs}+ runs)" if given_up else ""),
              flush=True)

        if todo:
            with open(out_path, "a", encoding="utf-8") as out, \
                    cf.ThreadPoolExecutor(args.workers) as pool:
                # With --round-robin, every other item starts on the fallback model (and falls back to
                # the main one), spreading load over both models' quotas.
                pairs = [(args.model, args.fallback_model or None),
                         (args.fallback_model, args.model)] if args.round_robin and args.fallback_model \
                    else [(args.model, args.fallback_model or None)]
                futures = {pool.submit(transliterate, r, *pairs[i % len(pairs)][:1], auth, args.retries, cache,
                                       args.temperature, pairs[i % len(pairs)][1]): r
                           for i, r in enumerate(todo)}
                start, n_file = time.time(), 0
                try:
                    for fut in cf.as_completed(futures):
                        row = futures[fut]
                        try:
                            new_row, attempts, flags, used_model = fut.result()
                        except ItemFailed as e:
                            n_fail += 1
                            consecutive_fail += 1
                            attempts_hist[e.attempts] += 1
                            failures.write(json.dumps({"file": name, "id": row["id"], "attempts": e.attempts,
                                                       "error": str(e)}) + "\n")
                            failures.flush()
                            if consecutive_fail >= args.max_consecutive_failures:
                                raise FatalError(f"{consecutive_fail} items in a row failed after all attempts; "
                                                 f"last error: {e}")
                            continue
                        consecutive_fail = 0
                        attempts_hist[attempts] += 1
                        by_model[used_model] += 1
                        if used_model != args.model:
                            models_log.write(json.dumps({"file": name, "id": row["id"], "model": used_model}) + "\n")
                            models_log.flush()
                        if flags:
                            n_flagged += 1
                            cleanup.write(json.dumps({"file": name, "id": row["id"], "words": flags},
                                                     ensure_ascii=False) + "\n")
                            cleanup.flush()
                        with lock:
                            out.write(json.dumps(new_row, ensure_ascii=False) + "\n")
                            out.flush()
                            os.fsync(out.fileno())
                        done.add(row["id"])
                        n_ok += 1
                        n_file += 1
                        if n_file % args.log_every == 0 or n_file == len(todo):
                            rate = n_file / (time.time() - start)
                            eta = (len(todo) - n_file) / rate if rate else 0
                            print(f"  {name}: {len(done)}/{len(src_rows)} "
                                  f"({rate:.1f} items/s, ~{eta / 60:.0f} min left in file)", flush=True)
                except FatalError as e:
                    pool.shutdown(cancel_futures=True)
                    sys.exit(f"FATAL: {e}\nProgress is saved; fix the problem and re-run to resume.")
                except KeyboardInterrupt:
                    # Every finished item is already flushed to disk; don't wait for in-flight
                    # calls (a second Ctrl-C during that wait used to raise a traceback).
                    print("Interrupted. Progress is saved; re-run to resume.", file=sys.stderr, flush=True)
                    os._exit(1)

        if len(done) == len(src_rows):
            finalize(out_path, src_rows)
        if budget == 0:
            break

    print(f"Finished: {n_ok} transliterated, {n_fail} failed this run"
          + (" (see failures.log; re-run to retry)" if n_fail else "")
          + (f", {n_skipped} skipped after failing in {args.max_failed_runs}+ runs (use --retry-failed)"
             if n_skipped else ""))
    if by_model:
        print("Items per model this run: " + ", ".join(f"{m}: {n}" for m, n in by_model.most_common()))
    if n_flagged:
        print(f"{n_flagged} items kept non-Latin words for later cleanup (needs_cleanup.log in --work-dir)")
    if attempts_hist:
        print("Attempts per item this run: "
              + ", ".join(f"{k}: {v}" for k, v in sorted(attempts_hist.items())))

    # Structure check against the source split (same IDs, keys, labels, metadata).
    sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "data_prep"))
    from validate_splits import check_split
    src_names = {os.path.basename(p) for p in glob.glob(os.path.join(args.src, "*.jsonl"))}
    complete = all(os.path.exists(os.path.join(args.out, n)) and
                   sum(1 for _ in open(os.path.join(args.out, n))) == sum(1 for _ in open(os.path.join(args.src, n)))
                   for n in src_names)
    n, errors = check_split(os.path.dirname(os.path.abspath(args.src)), os.path.basename(args.src),
                            os.path.relpath(os.path.abspath(args.out), os.path.dirname(os.path.abspath(args.src))),
                            partial=not complete)
    pending = [e for e in errors if "non-Latin" in e]
    errors = [e for e in errors if "non-Latin" not in e]
    print(f"Structure check vs {args.src}: {n} rows, {'complete' if complete else 'partial'} -> "
          + ("OK" if not errors else f"{len(errors)} problem(s)")
          + (f"; {len(pending)} fields still have non-Latin words (pending cleanup)" if pending else ""))
    for e in errors[:10]:
        print(f"  - {e}")


if __name__ == "__main__":
    main()
