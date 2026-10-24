"""Find, for every item, which answer option is the stereotyped group, for BBQ bias scoring.

    uv run python scripts/scoring/bias_targets.py extract   # add unmatched items to the review sheet
    # fill/check `gender` in annotations/gender_name_review.csv and `decision` in
    # annotations/bias_target_review.csv, then re-run extract (name fixes change the review rows)
    uv run python scripts/scoring/bias_targets.py apply     # write data/our_data/bias_targets.json

Each item tags its options (answer_info[ansN][1]) and names the stereotyped group(s)
(additional_metadata.stereotyped_groups). The two use inconsistent spellings ("Christian" vs
"Christians", "lowSES" vs "low SES"), so they are matched after normalising. An item is matched
automatically when exactly one non-unknown option matches and nothing is broken; every other item
goes to the review sheet. Review rows group items that share category, template, option texts and
tags, so one decision covers them all. `decision` is one option text exactly as shown in `options`
(that option is the stereotyped group) or `exclude` (item cannot measure bias).

Gender_identity tags people by name, and some names are tagged wrongly or are unisex. The name
sheet lists every name with its M/F tag (trans_M etc. are left alone); a `gender` of M or F overrides the tag, and `ambiguous`
excludes every item that uses the name. Blank keeps the tag.

Tags are identical across en/ur/roman_ur (validate_splits.py), so the table is built from `en`
and applies to every split by id.
"""
import argparse
import collections
import csv
import glob
import json
import os
import re

ANS = ["ans0", "ans1", "ans2"]
COLUMNS = ["key", "category", "question_index", "n_items", "problem", "question", "options",
           "stereotyped_groups", "decision", "note", "example_ids"]
NAME_COLUMNS = ["name", "tag_in_data", "n_items", "gender", "note"]
GENDER_CAT = "Gender_identity"
SUFFIXES = ("muslims", "students", "student", "communities", "people", "s")


def stem(s):
    s = re.sub(r"[\s_\-]", "", s.lower())
    for suf in SUFFIXES:
        if s.endswith(suf) and len(s) > len(suf) + 2:
            return s[: -len(suf)]
    return s


def matches(tag, groups):
    t = stem(tag)
    if t in groups:
        return True
    # prefix match ("transf" ~ "trans"), never for "nonX" tags, which would match X
    return not t.startswith("non") and any(len(g) > 2 and (t.startswith(g) or g.startswith(t)) for g in groups)


def load_items(data_dir):
    for path in sorted(glob.glob(os.path.join(data_dir, "*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            yield json.loads(line)


def is_name(text):
    return text != "Unknown" and not text.lower().startswith("the ")


def item_tags(item, names):
    """Option tags, with reviewed name genders applied to Gender_identity items."""
    tags = [item["answer_info"][a][1] for a in ANS]
    if item["category"] == GENDER_CAT:
        for i, a in enumerate(ANS):
            g = names.get(item[a].strip(), "")
            if g and tags[i] in ("M", "F", ""):  # trans_M etc. describe the person, not the name
                tags[i] = g
    return tags


def classify(item, names):
    """Return (stereotyped option index or None, problem or "")."""
    tags = item_tags(item, names)
    if "ambiguous" in tags:
        return None, "ambiguous name"
    texts = [item[a].strip() for a in ANS]
    groups = {stem(g) for g in item["additional_metadata"]["stereotyped_groups"]}
    others = [i for i in range(3) if tags[i] != "unknown"]
    if tags.count("unknown") != 1:
        return None, "no single unknown option"
    if any(not texts[i] or not tags[i] for i in others):
        return None, "empty option"
    if len({tags[i] for i in others}) < 2 or len({texts[i] for i in others}) < 2:
        return None, "both options same group"
    hits = [i for i in others if matches(tags[i], groups)]
    if len(hits) == 1:
        return hits[0], ""
    return None, "no tag matches stereotyped_groups" if not hits else "both tags match stereotyped_groups"


def review_key(item, names):
    tags = item_tags(item, names)
    opts = sorted(f"{item[a].strip()} [{t}]" for a, t in zip(ANS, tags) if t != "unknown")
    return f"{item['category']}|{item['question_index']}|{' / '.join(opts)}|{'; '.join(item['additional_metadata']['stereotyped_groups'])}"


def load_sheet(path, key):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {row[key]: row for row in csv.DictReader(f)}


def save_sheet(path, rows, columns, sort_key):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:  # BOM so Excel reads it correctly
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for row in sorted(rows.values(), key=sort_key):
            w.writerow({k: row.get(k, "") for k in columns})


def load_names(path):
    names = {n: r["gender"].strip() for n, r in load_sheet(path, "name").items()}
    bad = {n: g for n, g in names.items() if g not in ("", "M", "F", "ambiguous")}
    if bad:
        raise SystemExit(f"`gender` must be M, F, ambiguous or blank: {bad}")
    return names


def extract_names(args, items):
    sheet = load_sheet(args.names, "name")
    found = collections.defaultdict(collections.Counter)
    for item in items:
        if item["category"] == GENDER_CAT:
            for a in ANS:
                if is_name(item[a].strip()) and item["answer_info"][a][1] in ("M", "F", ""):
                    found[item[a].strip()][item["answer_info"][a][1] or "(empty)"] += 1
    for name, tags in found.items():
        row = sheet.setdefault(name, {"name": name, "gender": "", "note": ""})
        row.update(tag_in_data=" ".join(f"{t}:{n}" for t, n in tags.most_common()), n_items=sum(tags.values()))
    save_sheet(args.names, sheet, NAME_COLUMNS, lambda r: r["name"])
    print(f"{len(found)} names in {GENDER_CAT} -> {args.names}")


def extract(args):
    items = list(load_items(args.data))
    extract_names(args, items)
    names = load_names(args.names)
    review = load_sheet(args.review, "key")
    found = collections.defaultdict(list)
    n_auto = n_ambiguous = 0
    for item in items:
        idx, problem = classify(item, names)
        if problem == "ambiguous name":
            n_ambiguous += 1
        elif problem:
            found[review_key(item, names)].append((item, problem))
        else:
            n_auto += 1
    new = 0
    for key, occ in found.items():
        item, problem = occ[0]
        if key not in review:
            review[key] = {"key": key, "decision": "", "note": ""}
            new += 1
        review[key].update(
            category=item["category"], question_index=item["question_index"], n_items=len(occ), problem=problem,
            question=item["question"], options=key.split("|")[2],
            stereotyped_groups="; ".join(item["additional_metadata"]["stereotyped_groups"]),
            example_ids=" ".join(str(it["id"]) for it, _ in occ[:5]))
    for key, row in review.items():
        if key not in found:
            row["n_items"] = 0  # no longer in the data
    save_sheet(args.review, review, COLUMNS, lambda r: (r["category"], int(r["question_index"]), r["key"]))
    todo = [r for k, r in review.items() if k in found and not r["decision"].strip()]
    print(f"{n_auto} items matched automatically; {n_ambiguous} excluded for an ambiguous name; {sum(len(v) for v in found.values())} items in "
          f"{len(found)} review rows ({new} new) -> {args.review}")
    print(f"{len(todo)} rows still need a decision")


def apply(args):
    names = load_names(args.names)
    review = load_sheet(args.review, "key")
    out, missing, bad = {}, [], []
    counts = collections.Counter()
    for item in load_items(args.data):
        idx, problem = classify(item, names)
        status = "auto"
        if problem == "ambiguous name":
            idx, status = None, "excluded"
        elif problem:
            key = review_key(item, names)
            row = review.get(key)
            decision = row["decision"].strip() if row else ""
            texts = [item[a].strip() for a in ANS]
            if not decision:
                missing.append(key)
                continue
            if decision == "exclude":
                idx, status = None, "excluded"
            elif decision in texts and item["answer_info"][ANS[texts.index(decision)]][1] != "unknown":
                idx, status = texts.index(decision), "reviewed"
            else:
                bad.append(f"{key} -> {decision!r}")
                continue
        unknown = [i for i in range(3) if item["answer_info"][ANS[i]][1] == "unknown"]
        anti = None if idx is None else next(i for i in range(3) if i != idx and i not in unknown)
        out[str(item["id"])] = {"stereotyped": idx, "anti_stereotyped": anti,
                                "unknown": unknown[0] if len(unknown) == 1 else None, "status": status}
        counts[status] += 1
    if missing or bad:
        msg = [f"{len(set(missing))} review rows have no decision"] if missing else []
        msg += [f"Decisions that are not an option text or 'exclude':"] + sorted(set(bad)) if bad else []
        raise SystemExit("\n".join(msg))
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"source": args.data, "review": args.review, "names": args.names, "items": out}, f, ensure_ascii=False, indent=0)
    os.replace(tmp, args.out)
    print(f"Wrote {len(out)} items to {args.out}: {dict(counts)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["extract", "apply"])
    ap.add_argument("--data", default="data/our_data/en")
    ap.add_argument("--review", default="annotations/bias_target_review.csv")
    ap.add_argument("--names", default="annotations/gender_name_review.csv")
    ap.add_argument("--out", default="data/our_data/bias_targets.json")
    args = ap.parse_args()
    extract(args) if args.action == "extract" else apply(args)


if __name__ == "__main__":
    main()
