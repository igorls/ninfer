"""Local mirror of a running Colab collection (runs on the workstation, uses the colab CLI).

Polls /content/rr/export through `colab download` (the Contents API, so it works while
vm_jobs.py keeps the kernel busy), verifies every part's SHA-256, and assembles
<run-dir>/<job>.jsonl from contiguous parts. Prints one line per poll. Exits 0 when the VM
reports done, 3 when the session stops answering (restore onto a new VM from <run-dir>).
"""
import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def fetch(session, remote, local):
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(local.suffix + ".part")
    done = subprocess.run(["colab", "download", "-s", session, remote, str(tmp)],
                          capture_output=True, text=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if done.returncode != 0 or not tmp.exists():
        return False
    tmp.replace(local)
    return True


def verified(path, sha256):
    return path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == sha256


def sync_job(session, job_dir, name, run_dir):
    """Fetch the remote state into staging, then every part it lists. The local state.json
    (the restore input) is replaced only once all listed parts are present and verified."""
    staged = job_dir / "state.json.staged"
    if not fetch(session, f"/content/rr/export/{name}/state.json", staged):
        return None
    state = json.loads(staged.read_text())
    for part in state["parts"]:
        path = job_dir / part["name"]
        if not verified(path, part["sha256"]):
            if not fetch(session, f"/content/rr/export/{name}/{part['name']}", path) or \
                    not verified(path, part["sha256"]):
                path.unlink(missing_ok=True)
                return None
    staged.replace(job_dir / "state.json")
    return assemble(job_dir, name, run_dir)


def assemble(job_dir, name, run_dir):
    state = json.loads((job_dir / "state.json").read_text())
    data, first = b"", 0
    for part in state["parts"]:
        path = job_dir / part["name"]
        if part["first"] != first or not verified(path, part["sha256"]):
            return None
        data += gzip.decompress(path.read_bytes())
        first = part["end"]
    if hashlib.sha256(data).hexdigest() != state["complete_sha256"]:
        return None
    target = run_dir / f"{name}.jsonl"
    tmp = target.with_suffix(".jsonl.tmp")
    tmp.write_bytes(data)
    tmp.replace(target)
    return state["lines"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()
    failures = 0
    while True:
        progress_path = args.run_dir / "progress.json"
        if not fetch(args.session, "/content/rr/export/progress.json", progress_path):
            failures += 1
            print(f"{time.strftime('%H:%M:%S')} progress download failed ({failures})", flush=True)
            if failures >= 3:
                print("SESSION_LOST", flush=True)
                sys.exit(3)
            time.sleep(60)
            continue
        failures = 0
        progress = json.loads(progress_path.read_text())
        summary, complete = [], True
        for name, job in progress["jobs"].items():
            local = sync_job(args.session, args.run_dir / "parts" / name, name, args.run_dir)
            complete &= local is not None and local == job["rows"] and job["state"] == "complete"
            summary.append(f"{name}:{job['state']} {job['rows']}/{job.get('total', '?')} local={local}")
        age = time.time() - progress["heartbeat"]
        print(f"{time.strftime('%H:%M:%S')} elapsed={progress['elapsed_s']:.0f}s heartbeat_age={age:.0f}s "
              + " | ".join(summary), flush=True)
        if progress.get("error"):
            print("VM_ERROR", progress["error"], flush=True)
        if progress.get("done"):
            if complete and not progress.get("error"):
                print("VM_DONE", flush=True)
                sys.exit(0)
            print("VM_DONE_INCOMPLETE: a job failed or its local copy is incomplete", flush=True)
            sys.exit(4)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
