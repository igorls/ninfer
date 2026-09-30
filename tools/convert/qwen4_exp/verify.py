"""Verify an upgraded Flash-Next v3 artifact through the generic v3 reader.

    python -m tools.convert.qwen4_exp.verify V3.ninfer [--report R.json] [--file-digests]

Every object is opened through ``tools.artifact.reader.Artifact`` (which checks framing, the
directory, part headers and each object's encoded size and alignment) and hashed. Unchanged
objects must equal the v2 object digests recorded by the upgrade; the two baked MTP banks must
equal the upgrade's bank digests and the pinned loader-oracle digests. ``--file-digests`` also
hashes every file of the set and compares them with the pinned qualified digests.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from tools.artifact.file_io import READ_FLAGS, pread
from tools.artifact.reader import Artifact

from .inventory import V2_OBJECTS
from .source import MTP_BANK_SHA256, MTP_BANKS, V3_FILE_SHA256

READ_BYTES = 16 * 1024 * 1024


def _object_digest(artifact: Artifact, object_id: str) -> str:
    digest = hashlib.sha256()
    for chunk in artifact.iter_object(object_id, chunk_bytes=READ_BYTES):
        digest.update(chunk)
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    fd = os.open(path, READ_FLAGS)
    try:
        offset = 0
        while chunk := pread(fd, READ_BYTES, offset):
            digest.update(chunk)
            offset += len(chunk)
    finally:
        os.close(fd)
    return digest.hexdigest()


def verify(path: Path, report: dict, *, file_digests: bool = False) -> dict:
    started = time.perf_counter()
    workers = 1 if os.name == "nt" else 8
    failures: list[str] = []
    with Artifact(path) as artifact:
        for index in range(len(artifact.directory.files)):
            artifact._file(index)  # open every part before sharing descriptors across threads
        ids = [obj.id for obj in artifact.objects]
        expected = report["v2_objects"]
        if len(ids) != V2_OBJECTS or set(ids) != set(expected):
            failures.append("object ids differ from the v2 inventory")
        for object_id in ids:
            artifact.object(object_id)  # encoding, size and alignment
        with ThreadPoolExecutor(workers) as pool:
            digests = dict(zip(ids, pool.map(lambda i: _object_digest(artifact, i), ids)))
        banks = report["mtp_banks"]
        for object_id, digest in digests.items():
            if object_id in MTP_BANKS:
                wanted = {banks[object_id]["sha256"], MTP_BANK_SHA256.get(object_id, digest)}
                if wanted != {digest}:
                    failures.append(f"{object_id}: bank digest {digest} != {sorted(wanted)}")
            elif digest != expected.get(object_id):
                failures.append(f"{object_id}: v3 bytes differ from the v2 object")
        files = [path] + [path.parent / f.path for f in artifact.directory.files[1:]]
        summary = {
            "objects": len(ids),
            "bindings": len(artifact.directory.bindings),
            "uses": len(artifact.directory.uses),
            "files": [p.name for p in files],
            "payload_bytes": artifact.payload_bytes,
            "file_bytes": artifact.file_bytes,
            "artifact_id": artifact.artifact_id.hex(),
        }
    object_seconds = time.perf_counter() - started
    if file_digests:
        with ThreadPoolExecutor(len(files) if os.name != "nt" else 1) as pool:
            summary["file_sha256"] = list(pool.map(_file_digest, files))
        if V3_FILE_SHA256 and tuple(summary["file_sha256"]) != V3_FILE_SHA256:
            failures.append("file digests differ from the pinned qualified v3 artifact")
    summary.update(
        unchanged_objects_equal_v2=sum(
            1 for i in ids if i not in MTP_BANKS and digests[i] == expected.get(i)
        ),
        object_seconds=round(object_seconds, 3),
        seconds=round(time.perf_counter() - started, 3),
        failures=failures,
        passed=not failures,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--file-digests", action="store_true")
    args = parser.parse_args()
    report_path = args.report or args.artifact.with_name(args.artifact.name + ".upgrade.json")
    summary = verify(
        args.artifact,
        json.loads(report_path.read_text(encoding="utf-8")),
        file_digests=args.file_digests,
    )
    print(json.dumps(summary, indent=1))
    sys.exit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
