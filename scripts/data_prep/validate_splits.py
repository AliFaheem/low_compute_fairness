"""Check that language splits are structurally identical, so experiments can treat them interchangeably.

For every split vs the reference (en): same file names, same row count, same IDs in the
same order, same keys in the same order, and identical non-text fields. Text fields
(context, question, ans0-2) must be non-empty; for roman_ur they must be Latin-only.

answer_info is compared on group tags only (answer_info[ansN][1], used for scoring): the
source Urdu split leaves a few display names (answer_info[ansN][0]) in Urdu script.

    python3 scripts/data_prep/validate_splits.py                       # en, ur, roman_ur under data/our_data
    python3 scripts/data_prep/validate_splits.py --splits en ur --allow-partial roman_ur
"""
import argparse
import glob
import json
import os
import re
import sys

TEXT_FIELDS = ["context", "question", "ans0", "ans1", "ans2"]
NON_LATIN = re.compile(r"[^\W\d_a-zA-Z]")


def comparable(row):
    out = {k: v for k, v in row.items() if k not in TEXT_FIELDS}
    out["answer_info"] = {k: v[1] for k, v in row["answer_info"].items()}
    return out


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def check_split(root, ref, split, partial):
    errors = []
    ref_files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, ref, "*.jsonl")))
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, split, "*.jsonl")))
    if not partial and files != ref_files:
        errors.append(f"file names differ: missing={sorted(set(ref_files) - set(files))} extra={sorted(set(files) - set(ref_files))}")
    n_rows = 0
    for name in files:
        if name not in ref_files:
            errors.append(f"{name}: not in {ref}")
            continue
        ref_rows = load(os.path.join(root, ref, name))
        rows = load(os.path.join(root, split, name))
        n_rows += len(rows)
        if partial:
            # Unfinished files may be out of order: compare by ID, require only a subset.
            ref_by_id = {r["id"]: r for r in ref_rows}
            pairs = [(ref_by_id.get(r["id"]), r) for r in rows]
            if any(a is None for a, _ in pairs):
                errors.append(f"{name}: IDs not in {ref}")
                continue
        else:
            if len(rows) != len(ref_rows):
                errors.append(f"{name}: {len(rows)} rows vs {len(ref_rows)} in {ref}")
                continue
            if [r["id"] for r in rows] != [r["id"] for r in ref_rows]:
                errors.append(f"{name}: IDs or order differ from {ref}")
                continue
            pairs = list(zip(ref_rows, rows))
        for a, b in pairs:
            where = f"{name} id={b['id']}"
            if list(a) != list(b):
                errors.append(f"{where}: keys {list(b)} != {list(a)}")
                continue
            ca, cb = comparable(a), comparable(b)
            diff = [k for k in ca if ca[k] != cb[k]]
            if diff:
                errors.append(f"{where}: non-text fields differ: {diff}")
            for k in TEXT_FIELDS:
                if not isinstance(b[k], str) or not b[k].strip():
                    errors.append(f"{where}: empty {k}")
                elif split == "roman_ur" and NON_LATIN.search(b[k]):
                    errors.append(f"{where}: non-Latin text in {k}: {b[k]!r}")
    return n_rows, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/our_data")
    ap.add_argument("--ref", default="en")
    ap.add_argument("--splits", nargs="*", default=["ur", "roman_ur"])
    ap.add_argument("--allow-partial", nargs="*", default=[], help="splits still being built")
    args = ap.parse_args()

    ok = True
    for split in args.splits + args.allow_partial:
        if not os.path.isdir(os.path.join(args.root, split)):
            print(f"{split}: MISSING ({os.path.join(args.root, split)})")
            ok = False
            continue
        partial = split in args.allow_partial
        n, errors = check_split(args.root, args.ref, split, partial)
        status = "OK" if not errors else f"{len(errors)} problem(s)"
        print(f"{split}: {n} rows checked against {args.ref}{' (partial)' if partial else ''} -> {status}")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        ok &= not errors
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
