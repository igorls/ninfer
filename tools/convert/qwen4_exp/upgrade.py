"""Upgrade the published Flash-Next v2 artifact to v3 in one sequential pass.

    python -m tools.convert.qwen4_exp.upgrade V2.ninfer V3.ninfer [--release-input]

Every v2 byte is read once, in order. That pass hashes the whole input (checked against the
pinned v2 SHA256) and every v2 object range, copies each object's payload into the v3 file set,
and bakes the two BF16 MTP expert banks into NVFP4 expert banks with the repository's
``NVFP4_MAXABS_DIVISOR_RNE_V1`` encoder, one divisor per expert. ``--release-input`` punches
the consumed input range out of the v2 file (Linux), so peak disk stays one artifact.

The artifact id is derived from the v2 digest and the directory, so the output is reproducible
byte for byte. A JSON report (``V3.ninfer.upgrade.json`` by default) records the v2 object
digests that ``verify`` compares against.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import json
import os
from pathlib import Path
import struct
import time

import torch

from tools.artifact.codecs.nvfp4 import (
    decode_nvfp4_expert_bank_words,
    encode_nvfp4_expert_bank,
)
from tools.artifact.file_io import READ_FLAGS, discard_cached_pages, pread
from tools.artifact.writer import ArtifactWriter
from tools.convert.quantization.nvfp4 import encode_rows, weight_divisor

from .inventory import Plan, plan_upgrade
from .source import (
    CHAT_TEMPLATE_SHA256,
    MTP_BANK_SHA256,
    MTP_BANKS,
    V2_BYTES,
    V2_SHA256,
)

CHUNK_BYTES = 64 * 1024 * 1024
V2_MAGIC = b"NINFER\x00\x02"
PAGE = 4096


def read_v2_directory(fd: int) -> tuple[dict, int]:
    """Return the v2 JSON directory and the file offset of its payload."""

    header = pread(fd, 16, 0)
    if len(header) != 16 or header[:8] != V2_MAGIC:
        raise ValueError("expected an NInfer v2 input")
    count = struct.unpack_from("<Q", header, 8)[0]
    directory = json.loads(pread(fd, count, 16))
    return directory, (16 + count + PAGE - 1) // PAGE * PAGE


def artifact_id(plan: Plan, v2_sha256: str) -> bytes:
    text = json.dumps(plan.description(), sort_keys=True, separators=(",", ":"))
    seed = b"ninfer-v3/qwen4_exp/v2-upgrade\x00" + v2_sha256.encode() + b"\x00" + text.encode()
    return hashlib.sha256(seed).digest()[:16]


class _Releaser:
    """Punch consumed input pages out of the v2 file (FALLOC_FL_PUNCH_HOLE|KEEP_SIZE)."""

    def __init__(self, fd: int) -> None:
        if os.name == "nt":
            raise OSError("--release-input needs Linux fallocate hole punching")
        self._libc = ctypes.CDLL(None, use_errno=True)
        self._libc.fallocate.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_int64,
        )
        self._fd = fd
        self.released = 0

    def upto(self, offset: int) -> None:
        end = offset // PAGE * PAGE
        if end <= self.released:
            return
        if self._libc.fallocate(self._fd, 0x03, self.released, end - self.released):
            error = ctypes.get_errno()
            raise OSError(error, f"fallocate punch hole failed: {os.strerror(error)}")
        self.released = end


class _BankBaker:
    """Accumulate one BF16 expert at a time and encode it into NVFP4 words."""

    def __init__(self, object_id: str, shape: tuple[int, ...]) -> None:
        self.id = object_id
        self.shape = shape
        experts, rows, columns = shape
        self.expert_bytes = rows * columns * 2
        # BF16 sources kept for the FP64 oracle check of the stored words.
        self.samples: dict[int, torch.Tensor] = {
            expert: torch.empty(0) for expert in sorted({0, experts // 2, experts - 1})
        }
        self.codes = torch.empty((experts, rows, columns // 2), dtype=torch.uint8)
        self.scales = torch.empty((experts, rows, columns // 16), dtype=torch.uint8)
        self.divisors = torch.empty((experts,), dtype=torch.float32)
        self._buffer = bytearray(self.expert_bytes)
        self._filled = 0
        self._expert = 0
        self.seconds = 0.0

    def feed(self, data: memoryview) -> None:
        while data:
            count = min(len(data), self.expert_bytes - self._filled)
            self._buffer[self._filled : self._filled + count] = data[:count]
            self._filled += count
            data = data[count:]
            if self._filled == self.expert_bytes:
                self._encode()

    def _encode(self) -> None:
        started = time.perf_counter()
        _, rows, columns = self.shape
        values = torch.frombuffer(self._buffer, dtype=torch.bfloat16).reshape(rows, columns)
        if self._expert in self.samples:
            self.samples[self._expert] = values.clone()
        divisor = weight_divisor(float(values.float().abs().amax()))
        codes, scales = encode_rows(values, divisor)
        self.codes[self._expert].copy_(codes)
        self.scales[self._expert].copy_(scales)
        self.divisors[self._expert] = struct.unpack("<f", divisor)[0]
        self._expert += 1
        self._filled = 0
        self.seconds += time.perf_counter() - started

    def payload(self) -> bytearray:
        if self._expert != self.shape[0] or self._filled:
            raise ValueError(f"{self.id}: incomplete BF16 bank")
        return encode_nvfp4_expert_bank(self.codes, self.scales, self.divisors, self.shape)


def _compare_reference(payload: bytearray, reference: Path) -> dict:
    """Byte comparison against a dumped v2 loader bank, summarized by plane."""

    data = reference.read_bytes()
    if len(data) != len(payload):
        return {"equal": False, "reason": f"length {len(data)} != {len(payload)}"}
    if data == payload:
        return {"equal": True}
    left = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    right = torch.frombuffer(payload, dtype=torch.uint8)
    differ = (left != right).nonzero().flatten()
    return {
        "equal": False,
        "differing_bytes": int(differ.numel()),
        "first_offsets": [int(v) for v in differ[:16]],
        "reference_bytes": [int(left[v]) for v in differ[:16]],
        "baked_bytes": [int(right[v]) for v in differ[:16]],
    }


def upgrade(
    input_path: Path,
    output_path: Path,
    *,
    release_input: bool = False,
    report_path: Path | None = None,
    mtp_reference: dict[str, Path] | None = None,
) -> dict:
    started = time.perf_counter()
    fd = os.open(input_path, os.O_RDWR if release_input else READ_FLAGS)
    try:
        size = os.fstat(fd).st_size
        if size != V2_BYTES:
            raise ValueError(f"v2 input has {size} bytes, expected {V2_BYTES}")
        directory, payload_start = read_v2_directory(fd)
        plan = plan_upgrade(directory)
        identity = artifact_id(plan, V2_SHA256)
        writer = ArtifactWriter(
            output_path,
            plan.specs,
            components=plan.components,
            bindings=plan.bindings,
            uses=plan.uses,
            metadata=plan.metadata,
            provenance=plan.provenance,
            artifact_id=identity,
        )
        with writer:
            result = stream_upgrade(
                fd,
                size,
                payload_start,
                plan,
                writer,
                banks=MTP_BANKS,
                expected_sha256=V2_SHA256,
                pinned_banks=MTP_BANK_SHA256,
                release_input=release_input,
                mtp_reference=mtp_reference,
            )
            # Decision 1: the official Flash-Next template is the one carried into v3.
            template = result["v2_objects"]["frontend/chat_template.jinja"]
            if template != CHAT_TEMPLATE_SHA256:
                raise ValueError(f"chat template {template} != {CHAT_TEMPLATE_SHA256}")
    finally:
        os.close(fd)
    result.update(
        input=str(input_path),
        output=str(output_path),
        artifact_id=identity.hex(),
        files=[f.path or output_path.name for f in writer.directory.files],
        payload_bytes=writer.directory.payload_bytes,
        seconds=round(time.perf_counter() - started, 3),
    )
    report_path = report_path or output_path.with_name(output_path.name + ".upgrade.json")
    report_path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(
        f"upgraded {input_path} -> {output_path}: {len(result['files'])} files, "
        f"{result['seconds']} s, report {report_path}",
        flush=True,
    )
    return result


def stream_upgrade(
    fd: int,
    size: int,
    payload_start: int,
    plan: Plan,
    writer: ArtifactWriter,
    *,
    banks: tuple[str, ...],
    expected_sha256: str,
    pinned_banks: dict[str, str],
    release_input: bool = False,
    mtp_reference: dict[str, Path] | None = None,
    chunk_bytes: int = CHUNK_BYTES,
) -> dict:
    """Copy, hash and bake the complete v2 file in one ordered read."""
    spans = sorted(
        (payload_start + obj.offset, payload_start + obj.offset + obj.bytes, obj.name)
        for obj in plan.sources.values()
    )
    object_hashes = {name: hashlib.sha256() for *_, name in spans}
    whole = hashlib.sha256()
    bakers = {name: _BankBaker(name, plan.sources[name].shape) for name in banks}
    releaser = _Releaser(fd) if release_input else None
    timings = {"read_wait": 0.0, "write": 0.0, "hash_wait": 0.0}
    bank_reports: dict[str, dict] = {}

    def hash_pieces(pieces: list[tuple[str, int, memoryview]]) -> None:
        for name, _, view in pieces:
            object_hashes[name].update(view)

    with ThreadPoolExecutor(3) as pool:
        position, span = 0, 0
        pending = pool.submit(pread, fd, min(chunk_bytes, size), 0)
        while position < size:
            clock = time.perf_counter()
            chunk = pending.result()
            timings["read_wait"] += time.perf_counter() - clock
            if not chunk:
                raise ValueError(f"v2 input ended at {position}")
            end = position + len(chunk)
            if end < size:
                pending = pool.submit(pread, fd, min(chunk_bytes, size - end), end)
            view = memoryview(chunk)
            whole_done = pool.submit(whole.update, view)
            pieces = []
            while span < len(spans) and spans[span][0] < end:
                begin, stop, name = spans[span]
                low, high = max(begin, position), min(stop, end)
                if low < high:
                    pieces.append((name, low - begin, view[low - position : high - position]))
                if stop <= end:
                    span += 1
                else:
                    break
            pieces_done = pool.submit(hash_pieces, pieces)
            clock = time.perf_counter()
            for name, offset, piece in pieces:
                baker = bakers.get(name)
                if baker is None:
                    writer.write_region(name, offset, piece)
                    continue
                baker.feed(piece)
                if offset + len(piece) == plan.sources[name].bytes:
                    payload = baker.payload()
                    digest = hashlib.sha256(payload).hexdigest()
                    bank_reports[name] = {
                        "sha256": digest,
                        "bytes": len(payload),
                        "encode_seconds": round(baker.seconds, 3),
                    }
                    pinned = pinned_banks.get(name)
                    if pinned is not None and digest != pinned:
                        raise ValueError(f"{name}: baked bank {digest} != pinned {pinned}")
                    if mtp_reference and name in mtp_reference:
                        bank_reports[name]["reference"] = _compare_reference(
                            payload, mtp_reference[name]
                        )
                    oracle = check_bank_semantics(payload, baker.shape, baker.samples)
                    bank_reports[name]["fp64_oracle"] = oracle
                    if not oracle["passed"]:
                        raise ValueError(f"{name}: baked bank violates the NVFP4 encoder rules")
                    writer.write_region(name, 0, payload)
                    del bakers[name]
            timings["write"] += time.perf_counter() - clock
            clock = time.perf_counter()
            whole_done.result()
            pieces_done.result()
            timings["hash_wait"] += time.perf_counter() - clock
            if releaser is not None:
                releaser.upto(end)
            else:
                discard_cached_pages(fd, position, len(chunk))
            position = end
    digest = whole.hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"v2 input SHA256 {digest} != published {expected_sha256}")
    if len(bank_reports) != len(banks):
        raise ValueError("the v2 input did not contain both MTP banks")
    return {
        "v2_sha256": digest,
        "v2_objects": {name: h.hexdigest() for name, h in object_hashes.items()},
        "mtp_banks": bank_reports,
        "timings": {key: round(value, 3) for key, value in timings.items()},
    }


_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)


def check_bank_semantics(
    payload: bytearray, shape: tuple[int, ...], sources: dict[int, torch.Tensor]
) -> dict:
    """Independent FP64 oracle on sampled experts, decoded from the stored bank words.

    From the BF16 source W of an expert: the divisor must be binary32(2688 / max|W|); each
    group's scale must lie within one E4M3FN step of max|W*d|/6; and every E2M1 code must be a
    nearest representable value of W*d/scale after saturation at 6 (a tie may go either way).
    """

    codes, scales, divisors = decode_nvfp4_expert_bank_words(payload, shape)
    _, rows, columns = shape
    worst_code, worst_scale, divisors_exact = 0.0, 0.0, True
    for expert, source in sources.items():
        w = source.double()
        amax = float(w.abs().max())
        expected = struct.unpack("<f", struct.pack("<f", 2688.0 / amax if amax else 1.0))[0]
        divisors_exact &= float(divisors[expert]) == expected
        y = (w * float(divisors[expert])).reshape(rows, columns // 16, 16)
        s = scales[expert].view(torch.float8_e4m3fn).double().unsqueeze(2)
        ideal = y.abs().amax(dim=2, keepdim=True) / 6.0
        step = torch.exp2(torch.floor(torch.log2(ideal.clamp(min=2.0**-6))) - 3)
        worst_scale = max(worst_scale, float(((s - ideal).abs() / step).max()))
        words = codes[expert]
        words = torch.stack((words & 0xF, words >> 4), dim=2).reshape(rows, columns // 16, 16)
        magnitude = _E2M1[(words & 0x7).long()]
        ratio = torch.where(s > 0, y / torch.where(s > 0, s, 1.0), 0.0)
        target = ratio.abs().clamp(max=6.0)
        nearest = (target.unsqueeze(3) - _E2M1).abs().amin(dim=3)
        excess = (magnitude - target).abs() - nearest
        sign_ok = ((words & 0x8) != 0) == (ratio < 0)
        sign_ok |= (magnitude == 0) & (ratio.abs() <= 0.25)
        worst_code = max(worst_code, float(excess.max()))
        divisors_exact &= bool(sign_ok.all())
    passed = divisors_exact and worst_scale <= 1.0 and worst_code <= 1e-5
    return {
        "experts": sorted(sources),
        "divisors_and_signs_exact": divisors_exact,
        "max_scale_error_e4m3_steps": worst_scale,
        "max_code_excess": worst_code,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--release-input", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--mtp-reference",
        nargs=2,
        type=Path,
        metavar=("GATE_UP", "DOWN"),
        help="v2 loader NVFP4 bank dumps to compare byte for byte",
    )
    args = parser.parse_args()
    reference = dict(zip(MTP_BANKS, args.mtp_reference)) if args.mtp_reference else None
    upgrade(
        args.input,
        args.output,
        release_input=args.release_input,
        report_path=args.report,
        mtp_reference=reference,
    )


if __name__ == "__main__":
    main()
