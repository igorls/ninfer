"""Exact NVFP4 codes, scale swizzling, and the stored FP32 weight divisor."""

from __future__ import annotations

import struct
from typing import Sequence

import torch

from ..formats import valid_positive_fp32_word
from ..layouts import block_scale_geometry, expert_block_scale_geometry
from ._tensor_bytes import (
    Payload,
    _exact_uint8_matrix,
    _payload_length,
    _payload_tensor,
)
from .direct import encode_direct


def swizzle_nvfp4_scales(
    natural_scales: torch.Tensor, shape: Sequence[int]
) -> torch.Tensor:
    """Map natural ``[N,K/16]`` E4M3FN words to the registered scale layout."""

    geometry = block_scale_geometry("nvfp4", shape)
    source = _exact_uint8_matrix(
        natural_scales,
        (geometry.n, geometry.groups_per_row),
        "NVFP4 scales",
    )
    return (
        source.reshape(geometry.n // 128, 4, 32, geometry.k_tiles, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(-1)
    )


def unswizzle_nvfp4_scales(
    stored_scales: torch.Tensor, shape: Sequence[int]
) -> torch.Tensor:
    """Recover natural ``[N,K/16]`` E4M3FN words from registered layout bytes."""

    geometry = block_scale_geometry("nvfp4", shape)
    if (
        stored_scales.dtype != torch.uint8
        or stored_scales.dim() != 1
        or stored_scales.numel() != geometry.scale_plane_bytes
    ):
        raise TypeError(
            "stored NVFP4 scales must be one-dimensional uint8 with "
            f"{geometry.scale_plane_bytes} elements"
        )
    source = stored_scales.detach().contiguous().cpu()
    return (
        source.reshape(geometry.n // 128, geometry.k_tiles, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(geometry.n, geometry.groups_per_row)
    )


def _positive_fp32_word(value: torch.Tensor | bytes | bytearray | memoryview) -> bytes:
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.float32 or value.numel() != 1:
            raise TypeError("NVFP4 weight divisor must be one FP32 word")
        raw = encode_direct(value.reshape(()), "fp32")
    else:
        raw = bytes(value)
        if len(raw) != 4:
            raise TypeError("NVFP4 weight divisor must contain exactly four bytes")
    word = struct.unpack("<I", raw)[0]
    if not valid_positive_fp32_word(word):
        raise ValueError("NVFP4 weight divisor must be finite and positive")
    return raw


def encode_nvfp4(
    packed_codes: torch.Tensor,
    natural_scales: torch.Tensor,
    weight_divisor: torch.Tensor | bytes | bytearray | memoryview,
    shape: Sequence[int],
) -> bytes:
    """Encode exact source NVFP4 words without numerical conversion."""

    geometry = block_scale_geometry("nvfp4", shape)
    codes = _exact_uint8_matrix(
        packed_codes,
        (geometry.n, geometry.k // 2),
        "NVFP4 packed codes",
    )
    scales = _exact_uint8_matrix(
        natural_scales,
        (geometry.n, geometry.groups_per_row),
        "NVFP4 scales",
    )
    invalid = ((scales & 0x80) != 0) | (scales == 0x7F)
    if bool(invalid.any()):
        raise ValueError("NVFP4 scales must be nonnegative finite E4M3FN words")
    divisor = _positive_fp32_word(weight_divisor)
    swizzled = swizzle_nvfp4_scales(scales, shape)
    payload = bytearray(geometry.payload_bytes)
    payload[: geometry.code_plane_bytes] = codes.numpy().tobytes()
    payload[
        geometry.scale_plane_offset : geometry.scale_plane_offset
        + geometry.scale_plane_bytes
    ] = swizzled.numpy().tobytes()
    payload[geometry.weight_divisor_offset :] = divisor
    return bytes(payload)


def decode_nvfp4_words(
    payload: Payload,
    shape: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode exact packed code, natural scale, and divisor words."""

    geometry = block_scale_geometry("nvfp4", shape)
    if _payload_length(payload) != geometry.payload_bytes:
        raise ValueError(
            f"NVFP4 payload has {_payload_length(payload)} bytes, "
            f"expected {geometry.payload_bytes}"
        )
    raw = _payload_tensor(payload, torch.device("cpu"))
    codes = (
        raw[: geometry.code_plane_bytes].clone().reshape(geometry.n, geometry.k // 2)
    )
    stored_scales = raw[
        geometry.scale_plane_offset : geometry.scale_plane_offset
        + geometry.scale_plane_bytes
    ]
    scales = unswizzle_nvfp4_scales(stored_scales, shape)
    invalid = ((scales & 0x80) != 0) | (scales == 0x7F)
    if bool(invalid.any()):
        raise ValueError("NVFP4 scales must be nonnegative finite E4M3FN words")
    divisor_bytes = bytes(raw[geometry.weight_divisor_offset :].numpy())
    divisor_word = struct.unpack("<I", divisor_bytes)[0]
    if not valid_positive_fp32_word(divisor_word):
        raise ValueError("NVFP4 weight divisor must be finite and positive")
    divisor = torch.frombuffer(bytearray(divisor_bytes), dtype=torch.float32).reshape(
        ()
    )
    return codes, scales, divisor


def _valid_scale_words(scales: torch.Tensor) -> None:
    if bool((((scales & 0x80) != 0) | (scales == 0x7F)).any()):
        raise ValueError("NVFP4 scales must be nonnegative finite E4M3FN words")


def _valid_divisors(divisors: torch.Tensor) -> None:
    words = divisors.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    if bool(((words & 0x80000000) != 0).any() | (words == 0).any()) or not bool(
        torch.isfinite(divisors).all()
    ):
        raise ValueError("NVFP4 expert divisors must be finite and positive")


def encode_nvfp4_expert_bank(
    packed_codes: torch.Tensor,
    natural_scales: torch.Tensor,
    weight_divisors: torch.Tensor,
    shape: Sequence[int],
) -> bytearray:
    """Encode an ``[E,N,K]`` bank from exact per-expert NVFP4 words.

    ``packed_codes`` is ``[E,N,K/2]``, ``natural_scales`` ``[E,N,K/16]`` and
    ``weight_divisors`` holds one FP32 divisor per expert.
    """

    geometry = expert_block_scale_geometry("nvfp4", shape)
    e, n, k = geometry.experts, geometry.n, geometry.k
    if packed_codes.dtype != torch.uint8 or tuple(packed_codes.shape) != (e, n, k // 2):
        raise TypeError(f"NVFP4 bank codes must be uint8 with shape {(e, n, k // 2)}")
    if natural_scales.dtype != torch.uint8 or tuple(natural_scales.shape) != (
        e,
        n,
        geometry.groups_per_row,
    ):
        raise TypeError(
            f"NVFP4 bank scales must be uint8 with shape {(e, n, geometry.groups_per_row)}"
        )
    if weight_divisors.dtype != torch.float32 or tuple(weight_divisors.shape) != (e,):
        raise TypeError(f"NVFP4 bank divisors must be FP32 with shape ({e},)")
    codes = packed_codes.detach().contiguous().cpu()
    scales = natural_scales.detach().contiguous().cpu()
    divisors = weight_divisors.detach().contiguous().cpu()
    _valid_scale_words(scales)
    _valid_divisors(divisors)
    swizzled = (
        scales.reshape(e, n // 128, 4, 32, geometry.k_tiles, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape(-1)
    )
    payload = bytearray(geometry.payload_bytes)
    view = memoryview(payload)
    view[: geometry.code_plane_bytes] = codes.reshape(-1).numpy()
    view[
        geometry.scale_plane_offset : geometry.scale_plane_offset
        + geometry.scale_plane_bytes
    ] = swizzled.numpy()
    view[geometry.weight_divisor_offset :] = encode_direct(divisors, "fp32")
    return payload


def decode_nvfp4_expert_bank_words(
    payload: Payload,
    shape: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode ``[E,N,K/2]`` codes, natural ``[E,N,K/16]`` scales and ``[E]`` divisors."""

    geometry = expert_block_scale_geometry("nvfp4", shape)
    if _payload_length(payload) != geometry.payload_bytes:
        raise ValueError(
            f"NVFP4 bank payload has {_payload_length(payload)} bytes, "
            f"expected {geometry.payload_bytes}"
        )
    e, n, k = geometry.experts, geometry.n, geometry.k
    raw = _payload_tensor(payload, torch.device("cpu"))
    codes = raw[: geometry.code_plane_bytes].reshape(e, n, k // 2)
    scales = (
        raw[
            geometry.scale_plane_offset : geometry.scale_plane_offset
            + geometry.scale_plane_bytes
        ]
        .reshape(e, n // 128, geometry.k_tiles, 32, 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape(e, n, geometry.groups_per_row)
    )
    divisors = raw[geometry.weight_divisor_offset :].clone().view(torch.float32)
    _valid_scale_words(scales)
    _valid_divisors(divisors)
    return codes, scales, divisors
