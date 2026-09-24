"""Rebuild outcome files on a replacement Colab VM from exported parts (run with `colab exec -f`).

Upload each job's state.json and parts to /content/rr/import/<job>/ first. Parts are checked
against state.json and must cover contiguous lines from zero; the restored file's SHA-256 must
equal the state's complete_sha256. Existing outcome files are never overwritten.
"""
import gzip
import hashlib
import json
from pathlib import Path

R = Path("/content/rr")
(R / "out").mkdir(parents=True, exist_ok=True)
for folder in sorted((R / "import").iterdir()):
    state = json.loads((folder / "state.json").read_text())
    data, expected_first = b"", 0
    for part in state["parts"]:
        blob = (folder / part["name"]).read_bytes()
        if hashlib.sha256(blob).hexdigest() != part["sha256"] or part["first"] != expected_first:
            raise SystemExit(f"{folder.name}: part {part['name']} is corrupt or out of order")
        data += gzip.decompress(blob)
        expected_first = part["end"]
    if hashlib.sha256(data).hexdigest() != state["complete_sha256"]:
        raise SystemExit(f"{folder.name}: restored bytes differ from the exported state")
    target = R / "out" / f"{folder.name}.jsonl"
    if target.exists():
        raise SystemExit(f"{target} already exists; restore only onto a fresh VM")
    target.write_bytes(data)
    # Continue the same part series so the local copy keeps assembling one contiguous file.
    export = R / "export" / folder.name
    export.mkdir(parents=True, exist_ok=True)
    for part in state["parts"]:
        (export / part["name"]).write_bytes((folder / part["name"]).read_bytes())
    (export / "state.json").write_text(json.dumps(state, indent=1))
    print("RESTORED", folder.name, state["lines"], hashlib.sha256(data).hexdigest())
