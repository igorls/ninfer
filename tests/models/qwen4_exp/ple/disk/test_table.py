"""PLE shard/page addressing and NF4 reconstruction using a small synthetic artifact."""

from pathlib import Path
import bisect
import json
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from nf4_reference import fp32, table as exact_table, rounded
from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorSpec
from tools.artifact.writer import ArtifactWriter


def check(executable, path):
    with Artifact(path) as artifact:
        binding = artifact.directory.bindings["text/layers/1/ple/table"]
        parts = binding["parts"]
        objects = [artifact.by_id[p["object"]] for p in parts]
        ends = []
        rows = 0
        ids = []
        for obj in objects:
            ids += [
                rows + r
                for r in sorted({0, 44, 45, obj.shape[0] - 1})
                if r < obj.shape[0]
            ]
            rows += obj.shape[0]
            ends.append(rows)
        result = subprocess.run(
            [executable, str(path)],
            input=struct.pack(f"<{len(ids)}I", *ids),
            check=True,
            stdout=subprocess.PIPE,
        )
        data = result.stdout
        assert struct.unpack_from("<Q", data)[0] == rows
        controls_id = artifact.directory.components["text"]["resources"][
            "ple_hash_controls"
        ]
        assert data[8:288] == artifact.read_object(controls_id)
        assert struct.unpack_from("<I", data, 288)[0] == len(objects)
        offset = 292
        tables = []
        for obj in objects:
            if obj.format == 'u4z8_g16_fp16':
                tables.append(None)
                continue
            pages = (obj.shape[0] + 44) // 45
            (word,) = struct.unpack(
                "<I", artifact.read_range(obj.offset + pages * 4096, 4)
            )
            expected = exact_table(fp32(word))
            actual = struct.unpack_from("<2048H", data, offset)
            assert list(actual) == expected, obj.id
            tables.append(expected)
            offset += 4096
        for gid in ids:
            shard = bisect.bisect_right(ends, gid)
            local = gid - (ends[shard - 1] if shard else 0)
            obj = objects[shard]
            if obj.format == 'u4z8_g16_fp16':
                expected = u4_reference(artifact.read_object(obj.id), obj.shape[0], local)
                assert list(struct.unpack_from('<160H', data, offset)) == expected, gid
                offset += 320
                continue
            encoded = artifact.read_range(
                obj.offset + local // 45 * 4096 + local % 45 * 90, 90
            )
            expected = [
                tables[shard][
                    encoded[80 + k // 16] * 16
                    + ((encoded[k // 2] >> (4 * (k % 2))) & 15)
                ]
                for k in range(160)
            ]
            assert list(struct.unpack_from("<160H", data, offset)) == expected, gid
            offset += 320
        assert len(data) == offset
    print(
        f"PLE disk source: {len(objects)} LUTs and {len(ids)} rows match exact reference"
    )


def u4_reference(raw, rows, row):
    """Exact stored FP16 multiplier times signed U4, independently rounded to BF16."""
    scales = struct.unpack_from('<10e', raw, ((rows * 80 + 255) // 256) * 256 + row * 20)
    words = raw[row * 80:(row + 1) * 80]
    values = [(((words[k // 2] >> (4 * (k % 2))) & 15) - 8) * scales[k // 16]
              for k in range(160)]
    return [(0x8000 if value < 0 else 0) | rounded(abs(value)) for value in values]


def fixture(path, divisor=2.0, invalid_controls=False, hash_controls=None, representation='nf4'):
    sizes = (46, 2)
    specs = [ResourceSpec("controls", 280)] + [
        TensorSpec(f"shard-{i}", (n, 160),
                   'u4z8_g16_fp16' if representation == 'u4' else 'nf4_g16_e4m3fn',
                   'packed_u4_g16_v1' if representation == 'u4' else 'row_paged_g16_4k_v1')
        for i, n in enumerate(sizes)
    ]
    controls = (
        list(hash_controls)
        if hash_controls is not None
        else [0x123456789123456, -77, 1234567890123456] + [1] * 16 + list(range(16))
    )
    if invalid_controls:
        controls[3] = 0
    with ArtifactWriter(
        path,
        specs,
        components={
            "text": {"config": {}, "resources": {"ple_hash_controls": "controls"}}
        },
        bindings={
            "text/layers/1/ple/table": {
                "parts": [
                    {"object": f"shard-{i}", "range": [0, n * 160]}
                    for i, n in enumerate(sizes)
                ]
            }
        },
        max_file_bytes=12288,
    ) as writer:
        writer.write_object("controls", struct.pack("<35q", *controls))
        for i, n in enumerate(sizes):
            if representation == 'u4':
                raw = bytearray(((n * 80 + 255) // 256) * 256 + n * 20)
                for r in range(n):
                    raw[r * 80:(r + 1) * 80] = bytes((j * 53 + r * 19 + i) % 256 for j in range(80))
                    scales = [(0.0, 2**-24, 0.125, 0.333251953125, 1.0, 4.0)[(r+j+i) % 6] for j in range(10)]
                    for group, scale in enumerate(scales):
                        if scale == 0:
                            raw[r * 80 + group * 8:r * 80 + (group + 1) * 8] = b'\x88' * 8
                    struct.pack_into('<10e', raw, ((n * 80 + 255) // 256) * 256 + r * 20, *scales)
                writer.write_object(f'shard-{i}', raw)
                continue
            raw = bytearray(((n + 44) // 45) * 4096)
            for r in range(n):
                begin = r // 45 * 4096 + r % 45 * 90
                raw[begin : begin + 80] = bytes(
                    (j * 53 + r * 19 + i) % 256 for j in range(80)
                )
                raw[begin + 80 : begin + 90] = bytes(
                    (j * 17 + r + i) % 127 for j in range(10)
                )
            raw.extend(struct.pack("<f", divisor if i == 0 else 3.25))
            writer.write_object(f"shard-{i}", raw)


def resegment(source, destination):
    # Split one 90-byte row across three physical files, independently of the producer's
    # usual 4 KiB volume boundaries. The logical payload and represented values stay intact.
    with Artifact(source) as artifact:
        payload = artifact.read_range(0, artifact.payload_bytes)
        shard = artifact.by_id["shard-0"]
        boundary = shard.offset + (44 * 80 if shard.format == 'u4z8_g16_fp16' else 44 * 90) + 37
        identity = artifact.artifact_id
    with source.open("rb") as f:
        header = f.read(32)
        root = json.loads(f.read(struct.unpack_from("<Q", header, 8)[0]))
    sizes = (boundary, 29, len(payload) - boundary - 29)
    root["files"] = [
        {"path": None if i == 0 else f"{destination.name}.part-{i}", "payload_bytes": n}
        for i, n in enumerate(sizes)
    ]
    directory = json.dumps(root).encode()
    prefix = ((32 + len(directory) + 4095) // 4096) * 4096
    offset = 0
    for i, n in enumerate(sizes):
        path = destination if i == 0 else destination.parent / root["files"][i]["path"]
        with path.open("wb") as f:
            if i == 0:
                f.write(
                    struct.pack("<8sQ16s", b"NINFER\x00\x03", len(directory), identity)
                )
                f.write(directory)
                f.write(bytes(prefix - 32 - len(directory)))
            else:
                f.write(struct.pack("<8sQ16s", b"NINPRT\x00\x03", i, identity))
                f.write(bytes(4096 - 32))
            f.write(payload[offset : offset + n])
        offset += n


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_table.py <test executable>")
    executable = sys.argv[1]
    with tempfile.TemporaryDirectory(prefix="ninfer-ple-table-") as directory:
        path = Path(directory) / "fixture.ninfer"
        fixture(path)
        check(executable, path)
        split = Path(directory) / "split.ninfer"
        resegment(path, split)
        check(executable, split)
        path = Path(directory) / 'u4.ninfer'
        fixture(path, representation='u4')
        check(executable, path)
        split = Path(directory) / 'u4-split.ninfer'
        resegment(path, split)
        check(executable, split)
        for name, options in (
            ("bad-divisor", {"divisor": 0}),
            ("bad-controls", {"invalid_controls": True}),
        ):
            path = Path(directory) / (name + ".ninfer")
            fixture(path, **options)
            result = subprocess.run(
                [executable, str(path)],
                input=b"",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            assert result.returncode == 1, (name, result.returncode, result.stderr)
