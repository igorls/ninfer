"""Build the fine-tune corpus in task form (VM): JSON lines {id, source, split, family, state,
question{type, instructions, criteria}, gold} where gold is the Clef option id ("true"/"false" for a
noul, the option key for a choice, the level index string for a score).

Sources: the router synthetic corpus (exact oracle, 4 families), the seeded oracle cases (exact,
6 families), and public classification sets for breadth. The 231 JevBench public items and the
24 diagnostic cases are never included. Split: 10% dev by id hash."""
import hashlib
import json
import random
import re
from pathlib import Path

OUT = Path("/content/hx/train_records.jsonl")
rng = random.Random(20261003)
records = []


def add(source, ident, family, state, question, gold):
    split = "dev" if int(hashlib.sha256(ident.encode()).hexdigest(), 16) % 10 == 0 else "train"
    records.append({"id": f"{source}/{ident}", "source": source, "split": split, "family": family,
                    "state": state, "question": question, "gold": str(gold)})


# Router synthetic corpus (chat form -> task form).
for line in open("/content/router2048.jsonl", encoding="utf-8"):
    row = json.loads(line)
    user = row["messages"][-1]["content"]
    head, options_block = user.split("\n\nOptions:\n", 1)
    assert head.startswith("State:\n")
    body = head[len("State:\n"):]
    state, _, instructions = body.rpartition("\n\n")
    criteria = {}
    for opt in options_block.split("\n"):
        m = re.match(r"^([A-Z])\. (.*)$", opt)
        if not m:
            continue
        key = row["mapping"][m.group(1)]
        text = m.group(2)
        description = text[len(key) + 2:] if text.startswith(key + ": ") else text
        criteria[key] = description
    assert row["expected"] in criteria, row["id"]
    add("router", row["id"][:16], row["family"], state, {"type": "choice", "instructions": instructions, "criteria": criteria}, row["expected"])

# Seeded oracle cases (JevBench task format).
for line in open("/content/oracle_train.jsonl", encoding="utf-8"):
    row = json.loads(line)
    q = row["question"]
    expected = str(row["expected"])
    if q["type"] == "noul":
        gold = "true" if expected == "yes" else "false"
    elif q["type"] == "score":
        gold = expected
    else:
        gold = expected
    add("oracle", row["id"], row["family"], row["state"], q, gold)

# Public sets.
try:
    from datasets import load_dataset
    def sample(ds, n):
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        return [ds[i] for i in idx[:n]]
    try:
        ds = load_dataset("PolyAI/banking77", split="train")
        names = ds.features["label"].names
        criteria = {n: n.replace("_", " ") for n in names}
        for i, ex in enumerate(sample(ds, 900)):
            add("banking77", str(i), "intent", ex["text"], {"type": "choice", "instructions": "Which banking intent does the customer message express?", "criteria": criteria}, names[ex["label"]])
    except Exception as error:  # noqa: BLE001
        print("banking77 skipped:", repr(error)[:200])
    try:
        ds = load_dataset("google/boolq", split="train")
        for i, ex in enumerate(sample(ds, 900)):
            add("boolq", str(i), "boolq", f"Passage: {ex['passage']}\n\nQuestion: {ex['question']}?",
                {"type": "noul", "instructions": "Does the passage answer the question with yes?"}, "true" if ex["answer"] else "false")
    except Exception as error:  # noqa: BLE001
        print("boolq skipped:", repr(error)[:200])
    try:
        ds = load_dataset("facebook/anli", split="train_r1")
        names = ["entailment", "neutral", "contradiction"]
        criteria = {"entailment": "The hypothesis follows from the premise.", "neutral": "The premise neither supports nor contradicts the hypothesis.", "contradiction": "The hypothesis contradicts the premise."}
        for i, ex in enumerate(sample(ds, 700)):
            add("anli", str(i), "nli", f"Premise: {ex['premise']}\n\nHypothesis: {ex['hypothesis']}", {"type": "choice", "instructions": "What is the relation between premise and hypothesis?", "criteria": criteria}, names[ex["label"]])
    except Exception as error:  # noqa: BLE001
        print("anli skipped:", repr(error)[:200])
    try:
        ds = load_dataset("Yelp/yelp_review_full", split="train")
        levels = ["One star: very negative.", "Two stars: negative.", "Three stars: mixed or neutral.", "Four stars: positive.", "Five stars: very positive."]
        for i, ex in enumerate(sample(ds, 600)):
            add("yelp", str(i), "sentiment", ex["text"][:1500], {"type": "score", "instructions": "How many stars did the reviewer give?", "criteria": levels}, str(ex["label"]))
    except Exception as error:  # noqa: BLE001
        print("yelp skipped:", repr(error)[:200])
    try:
        ds = load_dataset("fancyzhx/ag_news", split="train")
        names = ["world", "sports", "business", "science_technology"]
        criteria = {"world": "World news.", "sports": "Sports.", "business": "Business and finance.", "science_technology": "Science and technology."}
        for i, ex in enumerate(sample(ds, 500)):
            add("ag_news", str(i), "topic", ex["text"], {"type": "choice", "instructions": "Which section does this news item belong to?", "criteria": criteria}, names[ex["label"]])
    except Exception as error:  # noqa: BLE001
        print("ag_news skipped:", repr(error)[:200])
    try:
        ds = load_dataset("clinc_oos", "plus", split="train")
        names = ds.features["intent"].names
        criteria = {n: n.replace("_", " ") for n in names}
        for i, ex in enumerate(sample(ds, 400)):
            add("clinc150", str(i), "intent150", ex["text"], {"type": "choice", "instructions": "Which intent does the utterance express?", "criteria": criteria}, names[ex["intent"]])
    except Exception as error:  # noqa: BLE001
        print("clinc150 skipped:", repr(error)[:200])
except Exception as error:  # noqa: BLE001
    print("datasets unavailable:", repr(error)[:200])

rng.shuffle(records)
with OUT.open("w", encoding="utf-8") as out:
    for r in records:
        out.write(json.dumps(r, ensure_ascii=False) + "\n")
from collections import Counter
print(json.dumps({"records": len(records), "by_source": Counter(r["source"] for r in records),
                  "by_type": Counter(r["question"]["type"] for r in records), "dev": sum(r["split"] == "dev" for r in records)}))
print("RECORDS_BUILT", flush=True)
