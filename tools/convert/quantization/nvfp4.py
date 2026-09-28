"""NVFP4 encoder profile NVFP4_MAXABS_DIVISOR_RNE_V1 for a complete packed parent.

With ``amax = max|W|`` over the complete parent and ``2688 = 6 * 448``:

    d_w = binary32(2688 / amax)          # 1 for an all-zero parent
    y   = binary32(W * d_w)
    per 16 consecutive K values:
        s = E4M3FN(min(binary32(max|y| / 6), 448))   # RNE; +0 for an all-zero group
        q = E2M1(y / decode(s))                      # RNE ties-to-even, saturating at +-6

so ``W ~= q * decode(s) / d_w``. The divisor belongs to the complete parent: grouped logical
inputs and streaming row chunks share it. Adapted from cometkim/ninfer (Apache-2.0),
``tools/convert/quantization/nvfp4.py`` on ``feat/qwen3.8-nvfp4full``.
"""

from __future__ import annotations

import struct

import torch

FULL_RANGE = 2688.0
_E2M1_BOUNDARIES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
# A tie rounds up exactly when the upper neighbour has the even code.
_E2M1_TIES_UP = (False, True, False, True, False, True, False)


def weight_divisor(amax: float) -> bytes:
    """The parent's FP32 divisor word."""
    if amax != amax or amax == float("inf") or amax < 0:
        raise ValueError("NVFP4 source maximum must be finite")
    return struct.pack("<f", FULL_RANGE / amax if amax else 1.0)


def e2m1_rne_codes(values: torch.Tensor) -> torch.Tensor:
    """Signed E2M1 codes of FP32 ratios, ties to even, magnitudes above 6 saturate to 6."""
    magnitude = values.abs()
    codes = torch.zeros_like(values, dtype=torch.uint8)
    for boundary, tie_up in zip(_E2M1_BOUNDARIES, _E2M1_TIES_UP):
        codes += ((magnitude > boundary) | ((magnitude == boundary) & tie_up)).to(
            torch.uint8
        )
    return codes | (torch.signbit(values).to(torch.uint8) << 3)


def encode_rows(values: torch.Tensor, divisor: bytes) -> tuple[torch.Tensor, torch.Tensor]:
    """Return packed ``[rows,K/2]`` codes and natural ``[rows,K/16]`` E4M3FN scale words."""
    rows, columns = values.shape
    if columns % 16:
        raise ValueError("NVFP4 rows need complete 16-value groups")
    d = struct.unpack("<f", divisor)[0]
    blocks = (values.float() * d).reshape(rows, columns // 16, 16)
    scales = (blocks.abs().amax(dim=2) / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn)
    decoded = scales.float()
    nonzero = decoded > 0
    ratios = torch.where(
        nonzero[..., None], blocks / torch.where(nonzero, decoded, 1.0)[..., None], 0.0
    )
    codes = e2m1_rne_codes(ratios).reshape(rows, columns // 2, 2)
    return codes[..., 0] | (codes[..., 1] << 4), scales.view(torch.uint8)
