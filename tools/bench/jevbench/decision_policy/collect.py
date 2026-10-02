"""Collect decision outputs from a local ninfer-serve over the corpus and the eval sets (VM).

Serves the production-recipe artifact with ninfer-serve (text only, FP8 KV, prefill chunk 2048,
concurrency 8) on 127.0.0.1:8010 and runs, through decision_study's tested request builders:
  native            /v1/systemone, labels in dataset order            (all records)
  native_reversed   /v1/systemone, choice labels reversed             (choice records)
  rotated           chat letter readout with labels rotated by one    (choice records)
  reason_1024       /v1/messages with a 1,024-token thinking budget   (eval sets + dev subset)
Rows go to /content/hx/collect/<set>.jsonl with the full request/response for every call.
Sets: train (corpus, split train), dev (corpus, split dev), public (231), oracle (24).
"""
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/content")
sys.path.insert(0, "/content/clef/jevbench")
import decision_study as study  # noqa: E402
from jevbench.tasks import load_jsonl  # noqa: E402

R = Path("/content/hx")
OUT = R / "collect"
OUT.mkdir(exist_ok=True)
MODEL = "qwen3.8-27b"
ENDPOINT = "http://127.0.0.1:8010"
ARTIFACT = json.loads((R / "artifact.json").read_text())["path"]
REASON_DEV = int(sys.argv[1]) if len(sys.argv) > 1 else 400

server_log = (R / "serve.log").open("w")
server = subprocess.Popen([str(R / "build/apps/ninfer-serve"), ARTIFACT, "--max-context", "8192", "--kv-capacity", "65536",
                           "--max-concurrency", "8", "--prefill-chunk", "2048", "--kv-dtype", "fp8",
                           "--model-id", MODEL, "--host", "127.0.0.1", "--port", "8010"],
                          stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
for _ in range(240):
    try:
        urllib.request.urlopen(ENDPOINT + "/v1/models", timeout=5).read()
        break
    except Exception:  # noqa: BLE001
        if server.poll() is not None:
            raise SystemExit("server exited: " + (R / "serve.log").read_text()[-2000:])
        time.sleep(5)
else:
    raise SystemExit("server not ready")
print("server ready", flush=True)
client = study.Client(ENDPOINT)


def corpus_tasks(split):
    tasks = []
    for line in open(R / "train_records.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["split"] != split:
            continue
        q = r["question"]
        if q["type"] == "noul":
            labels, expected = ["no", "yes"], "yes" if r["gold"] == "true" else "no"
        elif q["type"] == "score":
            labels, expected = [str(i) for i in range(len(q["criteria"]))], r["gold"]
        else:
            labels, expected = list(q["criteria"]), r["gold"]
        if len(labels) > 26:
            continue
        row = {"id": r["id"], "family": r["family"], "state": r["state"], "question": q, "labels": labels,
               "expected": expected, "group": r["id"], "provenance": {}}
        try:
            tasks.append(study.Task.read(row, split))
        except ValueError as error:
            print("skip", r["id"], error, flush=True)
    return tasks


J = Path("/content/clef/jevbench")
sets = {
    "public": [study.Task.read(json.loads(l), c) for c in ("easy", "original", "hard")
               for l in open(J / "datasets/public" / f"{c}.jsonl", encoding="utf-8") if l.strip()
               and not json.loads(l).get("provenance", {}).get("exclude_reason") and json.loads(l).get("split") == "public"],
    "oracle": [study.Task.read(json.loads(l), "oracle_diagnostic") for l in open("/content/oracle.jsonl", encoding="utf-8") if l.strip()],
    "dev": corpus_tasks("dev"),
    "train": corpus_tasks("train"),
}
print({k: len(v) for k, v in sets.items()}, flush=True)
rng = random.Random(20261003)


def reversed_native(task):
    mapped = study.mapped_task(task, "native")
    mapped.labels = list(reversed(task.labels))
    row = study.evaluate(mapped, "native", MODEL, client)
    row["method"] = "native_reversed"
    row["task_id"] = task.id
    return row


def run_set(name, tasks, reasoning):
    path = OUT / f"{name}.jsonl"
    started = time.time()
    jobs = []
    for task in tasks:
        jobs.append(("native", task))
        if task.question["type"] == "choice":
            jobs.append(("native_reversed", task))
            jobs.append(("rotated", task))
    reason_tasks = tasks if reasoning == "all" else rng.sample(tasks, min(reasoning, len(tasks)))
    for task in reason_tasks:
        jobs.append(("reason_1024", task))
    print(name, "jobs", len(jobs), flush=True)

    def work(job):
        method, task = job
        if method == "native_reversed":
            return reversed_native(task)
        return study.evaluate(task, method, MODEL, client)

    done = 0
    with path.open("w", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=6) as pool:
        for row in pool.map(work, jobs):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            done += 1
            if done % 500 == 0:
                print(name, done, "/", len(jobs), round(time.time() - started), "s", flush=True)
    print(name, "done", len(jobs), round(time.time() - started), "s", flush=True)


run_set("public", sets["public"], "all")
run_set("oracle", sets["oracle"], "all")
run_set("dev", sets["dev"], REASON_DEV)
run_set("train", sets["train"], 0)
server.terminate()
print("COLLECT_DONE", flush=True)
