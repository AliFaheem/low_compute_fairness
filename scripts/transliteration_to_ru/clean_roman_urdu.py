"""Replace words the transliteration model left in Urdu/other scripts, using a reviewed word list.

    python3 scripts/transliteration_to_ru/clean_roman_urdu.py extract   # add new leftover words to the review file
    # review/fill the `replacement` column in prompts/roman_urdu_word_fixes.csv
    python3 scripts/transliteration_to_ru/clean_roman_urdu.py apply     # replace reviewed words in completed files

Replacement is whole-word, only in the text fields, and keeps surrounding punctuation.
Rows whose `word` contains a space are exact phrases (matched on word boundaries), added by
hand for errors written in Latin letters that `extract` cannot detect, e.g. "Islamabad terutama".
Only files that are complete (same row count as the Urdu source) are rewritten, atomically,
so this is safe to run while transliteration is still working on a later file.
"""
import argparse
import collections
import csv
import glob
import json
import os
import re

FIELDS = ["context", "question", "ans0", "ans1", "ans2"]
NON_LATIN = re.compile(r"[^\W\d_a-zA-Z]")
PUNCT = ".,?!;:\"'()[]،۔؟"
COLUMNS = ["word", "replacement", "count", "example_roman", "example_urdu"]


def leftover_words(text):
    return [tok.strip(PUNCT) for tok in text.split() if NON_LATIN.search(tok.strip(PUNCT))]


def load_fixes(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {row["word"]: row for row in csv.DictReader(f)}


def save_fixes(path, fixes):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:  # BOM so Excel reads Urdu correctly
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for row in sorted(fixes.values(), key=lambda r: -int(r["count"] or 0)):
            w.writerow({k: row.get(k, "") for k in COLUMNS})


def scan(out_dir, src_dir):
    """word -> list of (file, id, field, roman text, urdu text)."""
    found = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(out_dir, "*.jsonl"))):
        name = os.path.basename(path)
        urdu = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(src_dir, name), encoding="utf-8")}
        for line in open(path, encoding="utf-8"):
            try:
                row = json.loads(line)
            except ValueError:
                continue  # half-written last line of a file still being built
            for k in FIELDS:
                for w in leftover_words(row[k]):
                    found[w].append((name, row["id"], k, row[k], urdu[row["id"]][k]))
    return found


def extract(args):
    fixes = load_fixes(args.fixes)
    found = scan(args.out, args.src)
    new = 0
    for word, occ in found.items():
        _, _, _, roman, urdu = occ[0]
        if word not in fixes:
            fixes[word] = {"word": word, "replacement": ""}
            new += 1
        fixes[word].update(count=len(occ), example_roman=roman, example_urdu=urdu)
    for word, row in fixes.items():
        if word not in found:
            row["count"] = 0  # already replaced everywhere
    save_fixes(args.fixes, fixes)
    todo = [w for w, r in fixes.items() if int(r["count"]) and not r["replacement"].strip()]
    print(f"{len(found)} leftover words in output ({new} new) -> {args.fixes}")
    print(f"{len(todo)} still need a replacement" + (f": {', '.join(todo)}" if todo else ""))


def apply(args):
    fixes = {w: r["replacement"].strip() for w, r in load_fixes(args.fixes).items() if r["replacement"].strip()}
    bad = {w: r for w, r in fixes.items() if NON_LATIN.search(r)}
    if bad:
        raise SystemExit(f"Replacements must be Latin-only: {bad}")
    phrases = {re.compile(r"(?<!\w)" + re.escape(w) + r"(?!\w)"): r for w, r in fixes.items() if " " in w}
    fixes = {w: r for w, r in fixes.items() if " " not in w}

    def fix_token(tok):
        core = tok.strip(PUNCT)
        if core not in fixes:
            return tok
        start = tok.index(core)
        return tok[:start] + fixes[core] + tok[start + len(core):]

    replaced, skipped = collections.Counter(), []
    for path in sorted(glob.glob(os.path.join(args.out, "*.jsonl"))):
        name = os.path.basename(path)
        n_src = sum(1 for l in open(os.path.join(args.src, name), encoding="utf-8") if l.strip())
        lines = [l for l in open(path, encoding="utf-8") if l.strip()]
        if len(lines) != n_src:
            skipped.append(f"{name} ({len(lines)}/{n_src})")
            continue
        rows = [json.loads(l) for l in lines]
        for row in rows:
            for k in FIELDS:
                for pattern, repl in phrases.items():
                    row[k], n = pattern.subn(repl, row[k])
                    replaced[name] += n
                new = " ".join(fix_token(t) for t in row[k].split(" "))
                if new != row[k]:
                    replaced[name] += sum(1 for t in row[k].split(" ") if t.strip(PUNCT) in fixes)
                    row[k] = new
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        os.replace(tmp, path)

    print(f"Replaced {sum(replaced.values())} words: {dict(replaced)}")
    if skipped:
        print(f"Skipped incomplete files (run again when done): {', '.join(skipped)}")
    left = scan(args.out, args.src)
    print(f"Leftover words remaining: {len(left)}" + (f" ({', '.join(left)})" if left else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["extract", "apply"])
    ap.add_argument("--out", default="data/our_data/roman_ur")
    ap.add_argument("--src", default="data/our_data/ur")
    ap.add_argument("--fixes", default="prompts/roman_urdu_word_fixes.csv")
    args = ap.parse_args()
    extract(args) if args.action == "extract" else apply(args)


if __name__ == "__main__":
    main()
