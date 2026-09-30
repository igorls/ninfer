from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct

import pytest
import torch

from tools.artifact.codecs.nvfp4 import encode_nvfp4_expert_bank
from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorSpec, plan_objects
from tools.artifact.writer import ArtifactWriter
from tools.convert.qwen4_exp.inventory import Plan, V2Object, plan_upgrade
from tools.convert.qwen4_exp.upgrade import artifact_id, stream_upgrade
from tools.convert.quantization.nvfp4 import encode_rows, weight_divisor

FIXTURE = Path(__file__).parents[1] / "fixtures" / "qwen4_exp" / "v2_directory.json.gz"


def _published_directory() -> dict:
    return json.loads(gzip.decompress(FIXTURE.read_bytes()))


def test_published_v2_directory_maps_to_the_closed_v3_inventory():
    plan = plan_upgrade(_published_directory())
    formats = Counter(getattr(spec, "format", "resource") for spec in plan.specs)
    assert formats == {
        "bf16": 1235,
        "nvfp4": 98,
        "fp8_e4m3fn_row_fp32": 96,
        "u4z8_g16_fp16": 128,
        "int64": 3,
        "resource": 6,
    }
    objects = plan_objects(plan.specs)
    assert objects[-1].offset + objects[-1].bytes == 109_680_552_704
    by_id = {obj.id: obj for obj in objects}
    # Objects ahead of the MTP banks keep their v2 offsets; the 342 behind them move down.
    moved = [o for o in objects if plan.sources[o.id].offset != o.offset]
    assert len(moved) == 343 and by_id["mtp/layer/mlp/experts/down"] in moved

    text = plan.components["text"]["config"]
    assert text["architectures"] == ["Qwen4ExpForCausalLM"]
    assert text["model_type"] == "qwen4_exp_text"
    assert plan.components["mtp"] == {
        "config": {"architectures": ["Qwen4ExpMTP"]},
        "target": "text",
    }
    assert plan.components["text"]["resources"]["chat_template.jinja"] == (
        "frontend/chat_template.jinja"
    )
    assert len(plan.bindings) == 1668
    assert plan.bindings["mtp/layers/0/mlp/experts/gate_up"] == {
        "object": "mtp/layer/mlp/experts/gate_up"
    }
    assert plan.bindings["vision/layers/0/attention/key"] == {
        "parts": [
            {"object": "vision/layers/0/attention/qkv", "range": [1327104, 2654208]}
        ]
    }
    uses = {(u["parameter"], u["input"]): u["activation_policy"] for u in plan.uses}
    assert uses[("text/layers/0/mlp/experts/gate_up", "text/layers/0/ffn_input")] == "AllowA4"
    assert uses[
        ("text/layers/3/attention/query_gate_key_value", "text/layers/3/mixer_input")
    ] == ("AllowA8")
    assert uses[
        ("mtp/layers/0/attention/output", "mtp/layers/0/attention/gated_output")
    ] == ("A16Only")
    assert uses[("text/output_head", "mtp/final_hidden")] == "A16Only"


def test_directory_rejects_a_changed_v2_inventory():
    directory = _published_directory()
    directory["objects"][100]["shape"] = [1, 1]
    with pytest.raises(ValueError, match="expected"):
        plan_upgrade(directory)


def _write_v2(path: Path, objects: list[tuple[dict, bytes]]) -> tuple[int, str]:
    """Write a v2-framed file; returns its payload offset and SHA256."""

    payload = bytearray()
    records = []
    for record, data in objects:
        if record["kind"] == "tensor":
            payload.extend(bytes(-len(payload) % 256))
        records.append({**record, "offset": len(payload), "bytes": len(data)})
        payload.extend(data)
    directory = json.dumps({"identity": {}, "objects": records}).encode()
    start = (16 + len(directory) + 4095) // 4096 * 4096
    raw = b"NINFER\x00\x02" + struct.pack("<Q", len(directory)) + directory
    raw += bytes(start - len(raw)) + payload
    path.write_bytes(raw)
    return start, hashlib.sha256(raw).hexdigest()


def test_stream_upgrade_copies_hashes_and_bakes_expert_banks(tmp_path):
    generator = torch.Generator().manual_seed(3)
    bank_shape = (3, 128, 64)
    bank = (torch.randn(bank_shape, generator=generator) * 0.02).to(torch.bfloat16)
    bank[1, 5, :16] = 0  # an all-zero group keeps a zero scale
    bank[2, 7, 3] = -0.0
    dense = torch.arange(15, dtype=torch.float32).to(torch.bfloat16).reshape(3, 5)
    table = torch.tensor([-1, 1 << 40, 3, 7], dtype=torch.int64)
    tensor = lambda name, value, fmt: (  # noqa: E731
        {"name": name, "kind": "tensor", "shape": list(value.shape), "format": fmt},
        value.contiguous().view(torch.uint8).numpy().tobytes(),
    )
    v2 = tmp_path / "v2.ninfer"
    start, sha = _write_v2(
        v2,
        [
            ({"name": "r", "kind": "resource"}, b"hello"),
            tensor("dense", dense, "BF16"),
            tensor("bank", bank, "BF16"),
            tensor("table", table, "I64"),
        ],
    )
    offsets = json.loads(v2.read_bytes()[16 : start].rstrip(b"\0"))["objects"]
    sources = {
        o["name"]: V2Object(
            o["name"], o["kind"], o["offset"], o["bytes"], tuple(o.get("shape", ()))
        )
        for o in offsets
    }
    plan = Plan(
        specs=(
            ResourceSpec("r", 5),
            TensorSpec("dense", (3, 5), "bf16", "contiguous_le_v1"),
            TensorSpec("bank", bank_shape, "nvfp4", "expert_block_scale_k16_m128x4_v1"),
            TensorSpec("table", (4,), "int64", "contiguous_le_v1"),
        ),
        sources=sources,
        components={"text": {"config": {"architectures": ["Qwen4ExpForCausalLM"]}}},
        bindings={},
        uses=(),
        metadata={},
        provenance={},
    )
    output = tmp_path / "v3.ninfer"
    identity = artifact_id(plan, sha)
    writer = ArtifactWriter(
        output, plan.specs, components=plan.components, bindings={}, artifact_id=identity
    )
    fd = os.open(v2, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        with writer:
            report = stream_upgrade(
                fd,
                v2.stat().st_size,
                start,
                plan,
                writer,
                banks=("bank",),
                expected_sha256=sha,
                pinned_banks={},
                chunk_bytes=1000,  # pieces straddle chunk boundaries
            )
    finally:
        os.close(fd)

    codes = torch.empty((3, 128, 32), dtype=torch.uint8)
    scales = torch.empty((3, 128, 4), dtype=torch.uint8)
    divisors = torch.empty(3, dtype=torch.float32)
    for expert in range(3):
        divisor = weight_divisor(float(bank[expert].float().abs().amax()))
        codes[expert], scales[expert] = encode_rows(bank[expert], divisor)
        divisors[expert] = struct.unpack("<f", divisor)[0]
    expected_bank = bytes(encode_nvfp4_expert_bank(codes, scales, divisors, bank_shape))

    with Artifact(output) as artifact:
        assert artifact.artifact_id == identity
        assert artifact.read_object("r") == b"hello"
        assert artifact.read_object("dense") == dense.view(torch.uint8).numpy().tobytes()
        assert artifact.read_object("table") == table.view(torch.uint8).numpy().tobytes()
        assert artifact.read_object("bank") == expected_bank
    assert report["v2_sha256"] == sha
    assert report["v2_objects"]["dense"] == hashlib.sha256(
        dense.view(torch.uint8).numpy().tobytes()
    ).hexdigest()
    assert report["mtp_banks"]["bank"]["sha256"] == hashlib.sha256(expected_bank).hexdigest()
    assert report["mtp_banks"]["bank"]["fp64_oracle"]["passed"]

    # A corrupted input is detected by the same pass, and the partial output is discarded.
    corrupt = bytearray(v2.read_bytes())
    corrupt[-1] ^= 1
    v2.write_bytes(corrupt)
    retry = tmp_path / "retry.ninfer"
    fd = os.open(v2, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        with pytest.raises(ValueError, match="SHA256"):
            with ArtifactWriter(
                retry, plan.specs, components=plan.components, bindings={}
            ) as writer:
                stream_upgrade(
                    fd,
                    len(corrupt),
                    start,
                    plan,
                    writer,
                    banks=("bank",),
                    expected_sha256=sha,
                    pinned_banks={},
                )
    finally:
        os.close(fd)
    assert not retry.exists()
