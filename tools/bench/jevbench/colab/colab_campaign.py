"""Run a Colab G4 collection across reclaimed sessions (runs on the workstation, uses the colab CLI).

Each round creates a session, uploads the source archive, the prebuilt collector in ~30 MB chunks,
the request files and jobs.json, plus the verified progress already mirrored under
<run-dir>/parts/ as one import.tar.gz. It then runs vm_setup.py, vm_restore.py and vm_jobs.py, and
mirrors results with colab_sync.py until the jobs finish or the session is lost. The session is
stopped at the end of every round. The campaign ends when the jobs are done, a job fails, the
session cap is reached, or two consecutive sessions add no rows.

  python colab_campaign.py --run-dir DIR --source DIR/source.tar.gz --collector DIR/collector.tar.gz \
      --jobs DIR/inputs/jobs-full.json --requests DIR/inputs/full2048.jsonl --prefix rr27b-g4-c
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHUNK = 30 * 1024 * 1024
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "MSYS_NO_PATHCONV": "1"}


def say(message):
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def colab(*args, timeout=None):
    try:
        done = subprocess.run(["colab", *args], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", env=ENV, timeout=timeout)
        return done.returncode, done.stdout + done.stderr
    except subprocess.TimeoutExpired as expired:
        return 124, f"local timeout after {expired.timeout}s"


def upload(session, local, remote):
    for attempt in range(3):
        code, output = colab("upload", "-s", session, str(local), remote, timeout=900)
        if code == 0:
            return True
        say(f"upload {local.name} failed ({attempt + 1}/3): {output.strip()[-300:]}")
        time.sleep(10)
    return False


def execute(session, script, timeout, *, allow_timeout=False):
    """Run a VM script; returns its output, or None when it raised or the session failed."""
    code, output = colab("exec", "-s", session, "-f", str(script), "--timeout", str(timeout),
                         timeout=timeout + 300)
    say(f"{script.name} exit={code}\n{output.strip()[-2500:]}")
    if "Traceback" in output or "SystemExit" in output:
        return None
    if code != 0 and not allow_timeout:
        return None
    return output


def collector_chunks(collector, workdir):
    """Split the collector package into upload-sized chunks plus its SHA-256 file (cached)."""
    folder = workdir / "collector-chunks"
    digest = hashlib.sha256(collector.read_bytes()).hexdigest()
    stamp = folder / "collector.tar.gz.sha256"
    if not stamp.exists() or stamp.read_text().split()[0] != digest:
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("collector.tar.gz.part-*"):
            old.unlink()
        data = collector.read_bytes()
        for index, offset in enumerate(range(0, len(data), CHUNK)):
            (folder / f"collector.tar.gz.part-{index:03d}").write_bytes(data[offset:offset + CHUNK])
        stamp.write_text(f"{digest}  collector.tar.gz\n")
    return sorted(folder.glob("collector.tar.gz.part-*")), stamp


def import_bundle(run_dir, workdir, names):
    """Pack the named jobs' verified state.json and listed parts; None when nothing is mirrored yet."""
    members = []
    for state_path in sorted(run_dir / "parts" / name / "state.json" for name in names):
        if not state_path.exists():
            continue
        state = json.loads(state_path.read_text())
        if not state["parts"]:
            continue
        members.append((state_path, f"{state_path.parent.name}/state.json"))
        for part in state["parts"]:
            path = state_path.parent / part["name"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != part["sha256"]:
                raise SystemExit(f"{path} does not match its mirrored state; resolve before resuming")
            members.append((path, f"{state_path.parent.name}/{part['name']}"))
    if not members:
        return None
    bundle = workdir / "import.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for path, name in members:
            archive.add(path, arcname=name)
    return bundle


def mirrored_rows(run_dir):
    rows = {}
    for state_path in (run_dir / "parts").glob("*/state.json"):
        rows[state_path.parent.name] = json.loads(state_path.read_text())["lines"]
    return rows


def round_trip(args, session, workdir, log):
    """One session. Returns 'done', 'failed', 'lost' or 'unavailable' (no G4 granted)."""
    code, output = colab("new", "-s", session, "--gpu", "G4", timeout=900)
    say(f"new {session} exit={code}: {output.strip()[-400:]}")
    if code != 0:
        return "unavailable"
    chunks, stamp = collector_chunks(args.collector, workdir)
    bundle = import_bundle(args.run_dir, workdir, [job["name"] for job in json.loads(args.jobs.read_text())])
    uploads = [(args.source, "/content/source.tar.gz"), (args.jobs, "/content/jobs.json"),
               *[(path, f"/content/{path.name}") for path in args.requests],
               *[(path, f"/content/{path.name}") for path in chunks], (stamp, f"/content/{stamp.name}")]
    if bundle:
        uploads.append((bundle, "/content/import.tar.gz"))
    for local, remote in uploads:
        if not upload(session, local, remote):
            return "lost"
    say(f"uploaded {len(uploads)} files" + (f", import {bundle.stat().st_size} bytes" if bundle else ""))
    if execute(session, HERE / "vm_setup.py", 600) is None:
        return "lost"
    if bundle and (execute(session, HERE / "vm_restore.py", 600) or "").count("RESTORED") == 0:
        return "lost"
    # The CLI returns at its timeout while the cell keeps the kernel busy with the jobs.
    execute(session, HERE / "vm_jobs.py", 60, allow_timeout=True)
    sync = subprocess.run([sys.executable, str(HERE / "colab_sync.py"), "--session", session,
                           "--run-dir", str(args.run_dir), "--interval", str(args.interval)],
                          stdout=log, stderr=subprocess.STDOUT, env=ENV)
    say(f"sync exit {sync.returncode}")
    return {0: "done", 4: "failed"}.get(sync.returncode, "lost")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--collector", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--requests", type=Path, nargs="+", required=True)
    parser.add_argument("--prefix", required=True, help="session names are <prefix><round>")
    parser.add_argument("--max-sessions", type=int, default=8)
    parser.add_argument("--interval", type=int, default=120, help="colab_sync poll seconds")
    args = parser.parse_args()
    # The colab CLI prints box-drawing characters; a redirected Windows stdout defaults to cp1252.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    workdir = args.run_dir / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    idle, unavailable = 0, 0
    for index in range(1, args.max_sessions + 1):
        session = f"{args.prefix}{index}"
        before = mirrored_rows(args.run_dir)
        say(f"round {index}/{args.max_sessions} session {session} mirrored={before}")
        with open(args.run_dir / f"sync-{session}.log", "a", encoding="utf-8") as log:
            try:
                outcome = round_trip(args, session, workdir, log)
            finally:
                code, output = colab("stop", "-s", session, timeout=300)
                say(f"stop {session} exit={code}: {output.strip()[-200:]}")
        after = mirrored_rows(args.run_dir)
        say(f"round {index} {outcome}; mirrored={after}")
        if outcome in ("done", "failed"):
            print(f"CAMPAIGN_{outcome.upper()}", flush=True)
            sys.exit(0 if outcome == "done" else 4)
        if outcome == "unavailable":
            unavailable += 1
            if unavailable >= 3:
                print("CAMPAIGN_NO_G4", flush=True)
                sys.exit(5)
            time.sleep(300)
            continue
        unavailable = 0
        idle = idle + 1 if after == before else 0
        if idle >= 2:
            print("CAMPAIGN_STALLED: two sessions added no rows", flush=True)
            sys.exit(6)
    print("CAMPAIGN_SESSION_CAP", flush=True)
    sys.exit(7)


if __name__ == "__main__":
    main()
