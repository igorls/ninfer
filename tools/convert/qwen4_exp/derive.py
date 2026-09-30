"""Derive the Flash-Next v3 artifact from the public v2 download, unattended and resumable.

    python -m tools.convert.qwen4_exp.derive WORKDIR [--verify] [--file-digests]

Steps, each skipped when its result already exists:

1. download the pinned v2 file into WORKDIR/v2 with ``aria2c -c`` (a partial download resumes);
2. upgrade it in place into WORKDIR/v3 (``--release-input``: the consumed v2 range is punched
   out, so the peak disk is one artifact), checking the v2 SHA256 during that single read;
3. optionally verify the result (``verify``), then record WORKDIR/v3/DONE.json with timings.

An interrupted upgrade leaves a hole-punched v2 file that still has its full length, which
``aria2c -c`` would accept as complete. A started-but-unfinished upgrade therefore discards the
v2 file and any partial output and downloads again.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .source import MTP_BANKS, V2_BYTES, V2_FILENAME, V2_URL
from .upgrade import upgrade
from .verify import verify


def _log(message: str) -> None:
    print(f"[derive {time.strftime('%H:%M:%S')}] {message}", flush=True)


def _ensure_aria2c() -> str:
    found = shutil.which("aria2c")
    if found:
        return found
    _log("installing aria2")
    subprocess.run(["apt-get", "install", "-y", "-qq", "aria2"], check=True)
    return shutil.which("aria2c") or "aria2c"


def download(workdir: Path) -> float:
    target = workdir / "v2" / V2_FILENAME
    control = target.with_name(target.name + ".aria2")
    if target.exists() and not control.exists() and target.stat().st_size == V2_BYTES:
        return 0.0
    target.parent.mkdir(parents=True, exist_ok=True)
    aria2c = _ensure_aria2c()
    started = time.perf_counter()
    for attempt in range(1, 6):
        _log(f"download attempt {attempt}: {V2_URL}")
        result = subprocess.run(
            [
                aria2c,
                "-x16",
                "-s16",
                "-k64M",
                "-c",
                "--file-allocation=none",
                "--auto-file-renaming=false",
                "--allow-overwrite=true",
                "--summary-interval=60",
                "--console-log-level=warn",
                "-d",
                str(target.parent),
                "-o",
                target.name,
                V2_URL,
            ]
        )
        if result.returncode == 0 and target.stat().st_size == V2_BYTES:
            break
    else:
        raise RuntimeError("v2 download did not complete")
    seconds = time.perf_counter() - started
    _log(f"downloaded {V2_BYTES} bytes in {seconds:.1f} s")
    return seconds


def derive(
    workdir: Path,
    *,
    run_verify: bool = False,
    file_digests: bool = False,
    mtp_reference: dict[str, Path] | None = None,
    download_only: bool = False,
) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    source = workdir / "v2" / V2_FILENAME
    output_dir = workdir / "v3"
    output = output_dir / V2_FILENAME
    done = output_dir / "DONE.json"
    started_marker = workdir / "upgrade.started"
    if done.exists():
        _log(f"already derived: {done}")
        return json.loads(done.read_text(encoding="utf-8"))
    if started_marker.exists():
        _log("previous upgrade was interrupted; its v2 input is partly released")
        source.unlink(missing_ok=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        started_marker.unlink()
    record: dict = {"download_seconds": download(workdir)}
    if download_only:
        return record
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob(f".{V2_FILENAME}*.tmp"):
        stale.unlink()
    started_marker.write_text(str(os.getpid()), encoding="utf-8")
    _log("upgrading in place")
    report = upgrade(source, output, release_input=True, mtp_reference=mtp_reference)
    source.unlink()
    started_marker.unlink()
    record.update(
        upgrade_seconds=report["seconds"],
        upgrade_timings=report["timings"],
        artifact_id=report["artifact_id"],
        mtp_banks={k: v["sha256"] for k, v in report["mtp_banks"].items()},
    )
    if run_verify or file_digests:
        _log("verifying")
        summary = verify(output, report, file_digests=file_digests)
        record["verify"] = summary
        if not summary["passed"]:
            raise RuntimeError(f"verification failed: {summary['failures'][:5]}")
    done.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    _log(f"derived {output}: {json.dumps(record)}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--file-digests", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--mtp-reference", nargs=2, type=Path, metavar=("GATE_UP", "DOWN"))
    args = parser.parse_args()
    derive(
        args.workdir,
        run_verify=args.verify,
        file_digests=args.file_digests,
        mtp_reference=dict(zip(MTP_BANKS, args.mtp_reference)) if args.mtp_reference else None,
        download_only=args.download_only,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
