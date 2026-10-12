"""Expert-routing trace format v1 (`.nxtr`): streaming reader and writer.

Layout (little-endian):

    "NXTRACE1" | u32 metadata_length | metadata JSON (UTF-8) | record*

    record header (24 bytes):
        u32 sentinel 0x43525452 | u8 kind | u8 reserved | u16 layer_count | u16 first_layer |
        u16 reserved | u32 columns | u64 timestamp_unix_us
    column table: columns x { u64 request_id, u32 position, u32 flags }
        request_id is the request-log request.request_id of the column's request, 0 when untagged
    expert ids:   columns x layer_count x top_k  u16, order [column][layer][rank]

A truncated or corrupt tail (an engine crash) ends the stream at the last complete record; the
reader reports that through `status` instead of raising.
"""

from __future__ import annotations

import json
import os
import struct
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import numpy as np

MAGIC = b"NXTRACE1"
SENTINEL = 0x43525452
FORMAT_NAME = "ninfer-expert-trace"
FORMAT_VERSION = 1

KIND_PREFILL = 1
KIND_DECODE = 2
KIND_MTP_PROPOSAL = 3
KIND_MTP_VERIFY = 4
KIND_NAMES = {
    KIND_PREFILL: "prefill",
    KIND_DECODE: "decode",
    KIND_MTP_PROPOSAL: "mtp-proposal",
    KIND_MTP_VERIFY: "mtp-verify",
}

RECORD_HEADER = struct.Struct("<IBBHHHIQ")
COLUMN_DTYPE = np.dtype([("request_id", "<u8"), ("position", "<u4"), ("flags", "<u4")])
ID_DTYPE = np.dtype("<u2")

assert RECORD_HEADER.size == 24
assert COLUMN_DTYPE.itemsize == 16


class TraceFormatError(ValueError):
    """The file is not a readable v1 expert trace (bad magic, metadata or writer arguments)."""


@dataclass(frozen=True)
class TraceMetadata:
    text_layers: int
    experts: int
    top_k: int
    mtp_layer: int | None = None
    server_instance_id: str | None = None
    artifact_id: str | None = None
    model_id: str | None = None
    created_unix_ms: int | None = None
    raw: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> TraceMetadata:
        if data.get("format") != FORMAT_NAME:
            raise TraceFormatError(f"metadata format is {data.get('format')!r}, expected {FORMAT_NAME!r}")
        if data.get("version") != FORMAT_VERSION:
            raise TraceFormatError(f"unsupported trace version {data.get('version')!r}")
        try:
            mtp = data.get("mtp_layer")
            return cls(
                text_layers=int(data["text_layers"]),
                experts=int(data["experts"]),
                top_k=int(data["top_k"]),
                mtp_layer=None if mtp is None else int(mtp),
                server_instance_id=_opt_str(data.get("server_instance_id")),
                artifact_id=_opt_str(data.get("artifact_id")),
                model_id=_opt_str(data.get("model_id")),
                created_unix_ms=None if data.get("created_unix_ms") is None else int(data["created_unix_ms"]),
                raw=dict(data),
            )
        except KeyError as error:
            raise TraceFormatError(f"metadata lacks required key {error.args[0]!r}") from None

    @property
    def total_layers(self) -> int:
        """Layers addressable by absolute layer index: text layers plus the MTP layer, if any."""
        return self.text_layers if self.mtp_layer is None else max(self.text_layers, self.mtp_layer + 1)

    @property
    def text_items(self) -> int:
        return self.text_layers * self.experts

    def to_json(self) -> dict[str, object]:
        out = dict(self.raw)
        out.update(
            {
                "format": FORMAT_NAME,
                "version": FORMAT_VERSION,
                "text_layers": self.text_layers,
                "mtp_layer": self.mtp_layer,
                "experts": self.experts,
                "top_k": self.top_k,
            }
        )
        for key in ("server_instance_id", "artifact_id", "model_id", "created_unix_ms"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out


def _opt_str(value: object) -> str | None:
    return None if value is None else str(value)


def make_metadata(
    *,
    text_layers: int = 48,
    experts: int = 512,
    top_k: int = 10,
    mtp_layer: int | None = None,
    server_instance_id: str | None = None,
    artifact_id: str | None = None,
    model_id: str | None = "qwen3.8-flash-next",
    created_unix_ms: int | None = None,
    **extra: object,
) -> dict[str, object]:
    """Build a metadata object for `TraceWriter` (tests, synthetic traces)."""
    out: dict[str, object] = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "text_layers": text_layers,
        "mtp_layer": mtp_layer,
        "experts": experts,
        "top_k": top_k,
    }
    for key, value in (
        ("server_instance_id", server_instance_id),
        ("artifact_id", artifact_id),
        ("model_id", model_id),
        ("created_unix_ms", created_unix_ms),
    ):
        if value is not None:
            out[key] = value
    out.update(extra)
    return out


@dataclass(slots=True)
class TraceRecord:
    """One forward unit. `ids` is `uint16[columns, layer_count, top_k]`, rank 0 first."""

    kind: int
    first_layer: int
    timestamp_us: int
    request_ids: np.ndarray  # uint64[columns]
    positions: np.ndarray  # uint32[columns]
    flags: np.ndarray  # uint32[columns]
    ids: np.ndarray  # uint16[columns, layer_count, top_k]

    @property
    def columns(self) -> int:
        return int(self.request_ids.shape[0])

    @property
    def layer_count(self) -> int:
        return int(self.ids.shape[1])

    @property
    def kind_name(self) -> str:
        return KIND_NAMES.get(self.kind, f"kind{self.kind}")


class TraceReader:
    """Streaming reader. Iterate it (or call `records()`) to get `TraceRecord`s.

    After iteration `status` is "complete", "truncated" (the file ends inside a record) or
    "corrupt" (a record header with a bad sentinel); `status_detail` explains and
    `records_read`/`valid_bytes` give the extent of the readable prefix.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self._file: BinaryIO = open(self.path, "rb", buffering=1 << 20)
        try:
            self.file_size = os.fstat(self._file.fileno()).st_size
            self.metadata = self._read_preamble()
        except BaseException:
            self._file.close()
            raise
        self.status = "open"
        self.status_detail = ""
        self.records_read = 0
        self.valid_bytes = self._file.tell()

    def _read_preamble(self) -> TraceMetadata:
        f = self._file
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise TraceFormatError(f"{self.path}: bad magic {magic!r}, expected {MAGIC!r}")
        length_bytes = f.read(4)
        if len(length_bytes) < 4:
            raise TraceFormatError(f"{self.path}: truncated before the metadata length")
        (length,) = struct.unpack("<I", length_bytes)
        payload = f.read(length)
        if len(payload) < length:
            raise TraceFormatError(f"{self.path}: truncated inside the metadata ({len(payload)}/{length} bytes)")
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TraceFormatError(f"{self.path}: metadata is not valid UTF-8 JSON: {error}") from None
        if not isinstance(decoded, dict):
            raise TraceFormatError(f"{self.path}: metadata is not a JSON object")
        return TraceMetadata.from_json(decoded)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> TraceReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[TraceRecord]:
        return self.records()

    def records(self) -> Iterator[TraceRecord]:
        f = self._file
        top_k = self.metadata.top_k
        size = self.file_size
        while True:
            start = f.tell()
            header = f.read(RECORD_HEADER.size)
            if not header:
                self.status = "complete"
                return
            if len(header) < RECORD_HEADER.size:
                self._stop("truncated", start, f"{len(header)} of {RECORD_HEADER.size} header bytes")
                return
            sentinel, kind, _r0, layer_count, first_layer, _r1, columns, ts_us = RECORD_HEADER.unpack(header)
            if sentinel != SENTINEL:
                self._stop("corrupt", start, f"bad record sentinel 0x{sentinel:08x}")
                return
            table_bytes = columns * COLUMN_DTYPE.itemsize
            ids_bytes = columns * layer_count * top_k * ID_DTYPE.itemsize
            if start + RECORD_HEADER.size + table_bytes + ids_bytes > size:
                self._stop("truncated", start, "record body extends past the end of the file")
                return
            table = np.frombuffer(f.read(table_bytes), dtype=COLUMN_DTYPE)
            raw_ids = np.frombuffer(f.read(ids_bytes), dtype=ID_DTYPE)
            record = TraceRecord(
                kind=kind,
                first_layer=first_layer,
                timestamp_us=ts_us,
                request_ids=np.ascontiguousarray(table["request_id"]),
                positions=np.ascontiguousarray(table["position"]),
                flags=np.ascontiguousarray(table["flags"]),
                ids=raw_ids.reshape(columns, layer_count, top_k),
            )
            self.records_read += 1
            self.valid_bytes = f.tell()
            yield record

    def _stop(self, status: str, offset: int, detail: str) -> None:
        self.status = status
        self.status_detail = f"{detail} at byte {offset} (after {self.records_read} complete records)"


def read_trace(path: str | os.PathLike[str]) -> tuple[TraceMetadata, list[TraceRecord], TraceReader]:
    """Load a whole trace into memory (tests and small traces). The reader carries the status."""
    reader = TraceReader(path)
    try:
        records = list(reader.records())
    finally:
        reader.close()
    return reader.metadata, records, reader


class TraceWriter:
    """Writes a v1 trace. Used by tests and the synthetic generator; the engine has its own writer."""

    def __init__(self, path: str | os.PathLike[str], metadata: Mapping[str, object] | TraceMetadata):
        meta = metadata.to_json() if isinstance(metadata, TraceMetadata) else dict(metadata)
        self.metadata = TraceMetadata.from_json(meta)
        payload = json.dumps(meta, separators=(",", ":")).encode("utf-8")
        self._file: BinaryIO = open(path, "wb")
        self._file.write(MAGIC)
        self._file.write(struct.pack("<I", len(payload)))
        self._file.write(payload)
        self.records_written = 0

    def write_record(
        self,
        kind: int,
        request_ids: Iterable[int] | np.ndarray,
        positions: Iterable[int] | np.ndarray,
        ids: np.ndarray,
        *,
        first_layer: int = 0,
        timestamp_us: int = 0,
        flags: Iterable[int] | np.ndarray | None = None,
    ) -> None:
        request_ids = np.asarray(request_ids, dtype="<u8")
        positions = np.asarray(positions, dtype="<u4")
        ids = np.asarray(ids)
        columns = request_ids.shape[0]
        if request_ids.ndim != 1 or positions.shape != request_ids.shape:
            raise TraceFormatError("request_ids and positions must be 1-D and the same length")
        if ids.ndim != 3 or ids.shape[0] != columns or ids.shape[2] != self.metadata.top_k:
            raise TraceFormatError(
                f"ids must have shape [columns={columns}, layers, top_k={self.metadata.top_k}], got {ids.shape}"
            )
        if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= self.metadata.experts):
            raise TraceFormatError(f"expert ids must be in [0, {self.metadata.experts})")
        layer_count = ids.shape[1]
        if layer_count >= 1 << 16 or first_layer >= 1 << 16 or columns >= 1 << 32:
            raise TraceFormatError("record geometry does not fit the v1 header")
        table = np.zeros(columns, dtype=COLUMN_DTYPE)
        table["request_id"] = request_ids
        table["position"] = positions
        if flags is not None:
            table["flags"] = np.asarray(flags, dtype="<u4")
        f = self._file
        f.write(RECORD_HEADER.pack(SENTINEL, kind, 0, layer_count, first_layer, 0, columns, timestamp_us))
        f.write(table.tobytes())
        f.write(np.ascontiguousarray(ids, dtype=ID_DTYPE).tobytes())
        self.records_written += 1

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
