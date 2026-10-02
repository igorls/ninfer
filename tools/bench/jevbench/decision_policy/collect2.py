"""Second collection pass (VM): the Evidence/Criterion/Options framing and the native-token
('semantic') readout from decision_study, as extra ensemble members. Appends rows to the same
set files. Restarts the server (collect.py stopped it)."""
import json
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/content")
sys.path.insert(0, "/content/clef/jevbench")
import decision_study as study  # noqa: E402

R = Path("/content/hx")
OUT = R / "collect"
MODEL = "qwen3.8-27b"
ENDPOINT = "http://127.0.0.1:8010"
ARTIFACT = json.loads((R / "artifact.json").read_text())["path"]
server_log = (R / "serve2.log").open("w")
server = subprocess.Popen([str(R / "build/apps/ninfer-serve"), ARTIFACT, "--max-context", "8192", "--kv-capacity", "65536",
                           "--max-concurrency", "8", "--prefill-chunk", "2048", "--kv-dtype", "fp8",
                           "--model-id", MODEL, "--host", "127.0.0.1", "--port", "8010"],
                          stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
for _ in range(240):
    try:
        urllib.request.urlopen(ENDPOINT + "/v1/models", timeout=5).read()
        break
    except Exception:  # noqa: BLE001
        time.sleep(5)
client = study.Client(ENDPOINT)
# Rebuild the task sets from the first pass's native rows (they carry everything needed).
for name in ("public", "oracle", "dev", "train"):
    path = OUT / f"{name}.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    natives = [r for r in rows if r["method"] == "native" and r["ok"]]
    tasks = []
    for r in natives:
        body = r["requests"][0]["body"]
        q = body["questions"]["q"]
        labels = list(r["probs"])
        tasks.append(study.Task(r["task_id"], r["family"], body["state"], q, labels, r["expected"], r.get("group", r["task_id"]), r["cohort"]))
    jobs = [("framed", t) for t in tasks] + [("semantic", t) for t in tasks if t.question["type"] != "choice"]
    started = time.time()
    with path.open("a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=6) as pool:
        for row in pool.map(lambda j: study.evaluate(j[1], j[0], MODEL, client), jobs):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(name, "done", len(jobs), round(time.time() - started), "s", flush=True)
server.terminate()
print("COLLECT2_DONE", flush=True)
