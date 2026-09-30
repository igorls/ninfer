"""Exact E4M3FN code words and row scales in the row-scale layouts.

``fp8_e4m3fn_row_bf16`` stores BF16 multipliers in ``row_scale_v1``;
``fp8_e4m3fn_row_fp32`` stores FP32 multipliers in ``row_scale_fp32_v1``.
"""

from __future__ import annotations

from typing import Sequence

import torch

from ..formats import Fp8RowFormat
from ..layouts import _format, row_scale_geometry
from ._tensor_bytes import (
    Payload,
    _exact_uint8_matrix,
    _payload_length,
    _payload_tensor,
)
from .direct import decode_direct, encode_direct


_SCALE_DIRECT = {2: "bf16", 4: "fp32"}
_SCALE_DTYPES = {2: torch.bfloat16, 4: torch.float32}


def _row_format(format: str | Fp8RowFormat) -> Fp8RowFormat:
    spec = _format(format)
    if not isinstance(spec, Fp8RowFormat):
        raise ValueError("row-scaled FP8 codec requires a row-scaled FP8 format")
    return spec


def _exact_scale_vector(
    tensor: torch.Tensor, spec: Fp8RowFormat, length: int, label: str
) -> torch.Tensor:
    dtype = _SCALE_DTYPES[spec.scale_bytes]
    if tensor.dtype != dtype or tuple(tensor.shape) != (length,):
        raise TypeError(f"{label} must be {dtype} with shape ({length},)")
    return tensor.detach().contiguous().cpu()


def validate_fp8_row_words(codes: torch.Tensor, scales: torch.Tensor) -> None:
    if bool(((codes & 0x7F) == 0x7F).any()):
        raise ValueError("row-scaled FP8 codes must be finite E4M3FN words")
    if scales.dtype == torch.float32:
        scale_words = scales.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        sign, exponent, name = 0x80000000, 0x7F800000, "FP32"
    else:
        scale_words = scales.view(torch.int16).to(torch.int32) & 0xFFFF
        sign, exponent, name = 0x8000, 0x7F80, "BF16"
    invalid_scales = ((scale_words & sign) != 0) | ((scale_words & exponent) == exponent)
    if bool(invalid_scales.any()):
        raise ValueError(f"row-scaled FP8 scales must be nonnegative finite {name} words")
    zero_scale = scale_words == 0
    nonzero_code = (codes & 0x7F) != 0
    if bool((zero_scale.unsqueeze(1) & nonzero_code).any()):
        raise ValueError("a zero row scale requires only signed-zero FP8 codes")


def encode_fp8_row_scaled(
    code_words: torch.Tensor,
    row_scales: torch.Tensor,
    shape: Sequence[int],
    format: str | Fp8RowFormat = "fp8_e4m3fn_row_bf16",
) -> bytes:
    """Encode exact E4M3FN code words and the format's row multipliers."""

    spec = _row_format(format)
    geometry = row_scale_geometry(spec, shape)
    codes = _exact_uint8_matrix(
        code_words,
        (geometry.n, geometry.k),
        "row-scaled FP8 codes",
    )
    scales = _exact_scale_vector(
        row_scales,
        spec,
        geometry.n,
        "row-scaled FP8 scales",
    )
    validate_fp8_row_words(codes, scales)
    payload = bytearray(geometry.payload_bytes)
    payload[: geometry.code_plane_bytes] = codes.numpy().tobytes()
    scale_begin = geometry.scale_plane_offset
    payload[scale_begin : scale_begin + geometry.scale_plane_bytes] = encode_direct(
        scales, _SCALE_DIRECT[spec.scale_bytes]
    )
    return bytes(payload)


def decode_fp8_row_scaled_words(
    payload: Payload,
    shape: Sequence[int],
    format: str | Fp8RowFormat = "fp8_e4m3fn_row_bf16",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode exact E4M3FN code words and the format's row multipliers."""

    spec = _row_format(format)
    geometry = row_scale_geometry(spec, shape)
    if _payload_length(payload) != geometry.payload_bytes:
        raise ValueError(
            f"row-scaled FP8 payload has {_payload_length(payload)} bytes, "
            f"expected {geometry.payload_bytes}"
        )
    raw = _payload_tensor(payload, torch.device("cpu"))
    codes = raw[: geometry.code_plane_bytes].clone().reshape(geometry.n, geometry.k)
    scale_begin = geometry.scale_plane_offset
    scale_bytes = raw[scale_begin : scale_begin + geometry.scale_plane_bytes]
    scales = decode_direct(scale_bytes, _SCALE_DIRECT[spec.scale_bytes], (geometry.n,))
    validate_fp8_row_words(codes, scales)
    return codes, scales


def dequantize_fp8_row_scaled(
    payload: Payload,
    shape: Sequence[int],
    dtype: torch.dtype = torch.float32,
    format: str | Fp8RowFormat = "fp8_e4m3fn_row_bf16",
) -> torch.Tensor:
    """Reconstruct ``binary32(code * scale)`` from the exact stored words."""

    codes, scales = decode_fp8_row_scaled_words(payload, shape, format)
    return (codes.view(torch.float8_e4m3fn).float() * scales.float().unsqueeze(1)).to(
        dtype
    )
