"""Exact U4Z8 G16 codes and binary16 group multipliers in packed_u4_g16_v1 layout."""

from __future__ import annotations

from typing import Sequence

import numpy
import torch

from ..layouts import packed_u4_geometry
from ._tensor_bytes import Payload, _payload_length, _payload_tensor


def validate_u4z8_words(codes: torch.Tensor, scales: torch.Tensor) -> None:
    if bool((codes > 15).any()):
        raise ValueError("U4Z8 codes must be integers in [0,15]")
    words = scales.view(torch.int16).to(torch.int32) & 0xFFFF
    if bool((((words & 0x8000) != 0) | ((words & 0x7C00) == 0x7C00)).any()):
        raise ValueError("U4Z8 scales must be nonnegative finite binary16 words")
    groups = codes.reshape(codes.shape[0], -1, 16)
    if bool(((words == 0).unsqueeze(2) & (groups != 8)).any()):
        raise ValueError("a zero U4Z8 scale admits only the zero-point code 8")


def encode_u4z8_g16(
    codes: torch.Tensor, scales: torch.Tensor, shape: Sequence[int]
) -> bytes:
    """Pack ``[N,K]`` unsigned nibbles low-first and append ``[N,K/16]`` FP16 multipliers."""

    geometry = packed_u4_geometry("u4z8_g16_fp16", shape)
    if codes.dtype != torch.uint8 or tuple(codes.shape) != (geometry.n, geometry.k):
        raise TypeError(f"U4Z8 codes must be uint8 with shape {(geometry.n, geometry.k)}")
    if scales.dtype != torch.float16 or tuple(scales.shape) != (
        geometry.n,
        geometry.groups_per_row,
    ):
        raise TypeError(
            f"U4Z8 scales must be FP16 with shape {(geometry.n, geometry.groups_per_row)}"
        )
    codes = codes.detach().contiguous().cpu()
    scales = scales.detach().contiguous().cpu()
    validate_u4z8_words(codes, scales)
    pairs = codes.reshape(geometry.n, geometry.k // 2, 2)
    packed = pairs[..., 0] | (pairs[..., 1] << 4)
    payload = bytearray(geometry.payload_bytes)
    payload[: geometry.code_plane_bytes] = packed.numpy().tobytes()
    payload[geometry.scale_plane_offset :] = scales.numpy().astype("<f2").tobytes()
    return bytes(payload)


def decode_u4z8_g16_words(
    payload: Payload, shape: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unsigned ``[N,K]`` codes and ``[N,K/16]`` FP16 multipliers."""

    geometry = packed_u4_geometry("u4z8_g16_fp16", shape)
    if _payload_length(payload) != geometry.payload_bytes:
        raise ValueError(
            f"U4Z8 payload has {_payload_length(payload)} bytes, "
            f"expected {geometry.payload_bytes}"
        )
    raw = _payload_tensor(payload, torch.device("cpu"))
    packed = raw[: geometry.code_plane_bytes].reshape(geometry.n, geometry.k // 2)
    codes = torch.stack((packed & 0xF, packed >> 4), dim=2).reshape(geometry.n, geometry.k)
    words = numpy.frombuffer(raw[geometry.scale_plane_offset :].numpy(), dtype="<f2")
    scales = torch.from_numpy(words.astype(numpy.float16)).reshape(
        geometry.n, geometry.groups_per_row
    )
    validate_u4z8_words(codes, scales)
    return codes, scales


def dequantize_u4z8_g16(payload: Payload, shape: Sequence[int]) -> torch.Tensor:
    """Reconstruct ``binary32(binary32(u - 8) * binary32(scale))``."""

    codes, scales = decode_u4z8_g16_words(payload, shape)
    q = codes.to(torch.float32) - 8.0
    return q * scales.to(torch.float32).repeat_interleave(16, dim=1)
