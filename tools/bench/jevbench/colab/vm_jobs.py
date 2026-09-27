"""Foreground collection driver for a Colab VM (run with `colab exec -f`, short --timeout).

The CLI returns at its client-side timeout but this cell keeps running, which keeps the
kernel busy for the whole collection. Progress leaves the VM only through files that the
local side fetches with `colab download` (the Contents API, which does not use the kernel):

  /content/rr/export/progress.json        heartbeat, per-job row counts and state
  /content/rr/export/<job>/state.json     ordered delta parts with SHA-256
  /content/rr/export/<job>/part-*.jsonl.gz complete outcome lines only

Jobs come from /content/rr/inputs/jobs.json:
  [{"name": "pilot-c8", "requests": "pilot64.jsonl", "concurrency": 8, "derive_2048": false}, ...]
"direct_only": true collects only the direct action with its answer-letter logprobs.
Outcome files are /content/rr/out/<name>.jsonl. A restored job resumes from its rows.
"""
import gzip
import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

R = Path("/content/rr")
EXPORT = R / "export"
OUT = R / "out"
TOOL = R / "ninfer/tools/bench/jevbench/reasoning_router.py"
COLLECTOR = R / "build/apps/ninfer-reasoning-collect"
EXPORT_SECONDS = 120
OUT.mkdir(parents=True, exist_ok=True)
EXPORT.mkdir(parents=True, exist_ok=True)
jobs = json.loads((R / "inputs/jobs.json").read_text())
status = {job["name"]: {"state": "waiting", "rows": 0, "attempts": 0} for job in jobs}
started = time.time()
lock = threading.Lock()
# Terminal state: once set, every later publish repeats it and the exporter thread exits.
terminal = {"done": False, "error": None}


def complete_bytes(path):
    if not path.exists():
        return b""
    data = path.read_bytes()
    return data[: data.rfind(b"\n") + 1]


def export(job):
    """Append a gzip part holding the job's complete lines not yet exported."""
    name = job["name"]
    folder = EXPORT / name
    folder.mkdir(exist_ok=True)
    state_path = folder / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"lines": 0, "parts": []}
    data = complete_bytes(OUT / f"{name}.jsonl")
    lines = data.splitlines(keepends=True)
    if len(lines) > state["lines"]:
        part = f"part-{state['lines']:05d}-{len(lines):05d}.jsonl.gz"
        blob = gzip.compress(b"".join(lines[state["lines"]:]), mtime=0)
        (folder / part).write_bytes(blob)
        state["parts"].append({"name": part, "first": state["lines"], "end": len(lines),
                               "sha256": hashlib.sha256(blob).hexdigest()})
        state["lines"] = len(lines)
    state["complete_sha256"] = hashlib.sha256(data).hexdigest()
    tmp = folder / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(state_path)
    return len(lines)


def publish(final=False, error=None):
    with lock:
        if final:
            terminal["done"] = True
        if error:
            terminal["error"] = error
        for job in jobs:
            status[job["name"]]["rows"] = export(job)
        report = {"heartbeat": time.time(), "elapsed_s": round(time.time() - started, 1),
                  "done": terminal["done"], "error": terminal["error"], "jobs": status}
        tmp = EXPORT / "progress.json.tmp"
        tmp.write_text(json.dumps(report, indent=1))
        tmp.replace(EXPORT / "progress.json")


def exporter():
    while not terminal["done"]:
        time.sleep(EXPORT_SECONDS)
        if terminal["done"]:
            return
        try:
            publish()
        except Exception as error:  # noqa: BLE001 - keep exporting
            print("export failed:", error, flush=True)


def wait_ready():
    while True:
        build = (R / "build.log").read_text(errors="replace") if (R / "build.log").exists() else ""
        download = (R / "download.log").read_text(errors="replace") if (R / "download.log").exists() else ""
        if "BUILD_COMPLETE" in build and "VERIFIED" in download:
            return
        if "BUILD_FAILED" in build:
            raise SystemExit("collector build failed:\n" + build[-4000:])
        if "Traceback" in download or "AssertionError" in download:
            raise SystemExit("artifact download or verification failed:\n" + download[-2000:])
        time.sleep(15)


def run(job):
    name = job["name"]
    requests = R / "inputs" / job["requests"]
    total = sum(1 for line in requests.read_bytes().splitlines() if line.strip())
    status[name].update({"state": "running", "total": total, "started": time.time()})
    stalls = 0
    while True:
        before = len(complete_bytes(OUT / f"{name}.jsonl").splitlines())
        if before >= total:
            status[name]["state"] = "complete"
            return
        command = [sys.executable, str(TOOL), "collect", "--collector", str(COLLECTOR),
                   "--artifact", json.loads((R / "artifact.json").read_text())["path"],
                   "--requests", str(requests), "--out", str(OUT / f"{name}.jsonl"),
                   "--concurrency", str(job.get("concurrency", 1))]
        if job.get("derive_2048"):
            command.append("--derive-2048")
        if job.get("direct_only"):
            command.append("--direct-only")
        status[name]["attempts"] += 1
        with open(R / "logs" / f"{name}.log", "a") as log:
            log.write(f"\n=== attempt {status[name]['attempts']} at {time.ctime()}\n")
            log.flush()
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        after = len(complete_bytes(OUT / f"{name}.jsonl").splitlines())
        status[name]["last_exit"] = code
        if code == 0 and after >= total:
            status[name]["state"] = "complete"
            return
        stalls = 0 if after > before else stalls + 1
        if stalls >= 3:
            status[name]["state"] = "failed"
            return


threading.Thread(target=exporter, daemon=True).start()
publish()
try:
    wait_ready()
    for job in jobs:
        run(job)
        publish()
except BaseException as failure:  # SystemExit included: IPython keeps the kernel alive
    publish(final=True, error=f"{type(failure).__name__}: {failure}"[-4000:])
    raise
publish(final=True)
print(json.dumps(status, indent=1))
