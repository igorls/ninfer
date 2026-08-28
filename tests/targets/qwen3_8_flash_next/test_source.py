from __future__ import annotations

import json
from pathlib import Path
import struct

import pytest

from tools.convert.qwen3_8_flash_next.source import SourceError, read_safetensor_directory


def _write_safetensor(path: Path, *, truncate: bool = False) -> None:
    directory = {
        "codes": {"dtype": "U8", "shape": [2, 2], "data_offsets": [0, 4]},
        "scale": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
    }
    header = json.dumps(directory, separators=(",", ":")).encode("utf-8")
    payload = b"\x01\x02\x03\x04" + struct.pack("<f", 1.0)
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload[:-1 if truncate else None])


def test_raw_safetensor_directory_exposes_exact_payload_slices(tmp_path):
    path = tmp_path / "fixture.safetensors"
    _write_safetensor(path)
    directory = read_safetensor_directory(path)
    assert directory.header_bytes > 0
    assert directory.tensors["codes"].dtype == "U8"
    assert directory.tensors["codes"].shape == (2, 2)
    assert directory.tensors["codes"].bytes == 4
    assert directory.tensors["scale"].absolute_offset == 8 + directory.header_bytes + 4


def test_raw_safetensor_directory_rejects_truncated_payload(tmp_path):
    path = tmp_path / "truncated.safetensors"
    _write_safetensor(path, truncate=True)
    with pytest.raises(SourceError, match="payload size mismatch"):
        read_safetensor_directory(path)
