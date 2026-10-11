"""U4Z8 encoder profile U4Z8_RANGE_RNE_V1 for ``u4z8_g16_fp16`` tables.

For every group of 16 consecutive K values ``x`` (binary32 of the source values):

    p = max(0, max x),  m = max(0, max -x)
    s = binary16_RNE(binary32(max(p / 7, m / 8)))      # +0 for an all-zero group
    u = 8 + clamp(RNE(binary32(x / decode(s))), -8, 7)  # u = 8 when s = 0

so ``x ~= (u - 8) * s``. The scale is the smallest that keeps the group's positive values within
code 15 (+7) and its negative values within code 0 (-8); rounding ``s`` to binary16 can push the
extreme value half a step out of range, where it saturates. Values whose scale underflows binary16
encode as zero.
"""

from __future__ import annotations

import torch


def encode_rows(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unsigned ``[rows,K]`` codes and ``[rows,K/16]`` binary16 multipliers."""

    if values.dim() != 2 or values.shape[1] % 16:
        raise ValueError("U4Z8 rows need complete 16-value groups")
    rows, columns = values.shape
    blocks = values.float().reshape(rows, columns // 16, 16)
    if not bool(torch.isfinite(blocks).all()):
        raise ValueError("U4Z8 source contains NaN or infinity")
    positive = blocks.amax(dim=2).clamp(min=0.0)
    negative = (-blocks).amax(dim=2).clamp(min=0.0)
    # abs() stores an all-zero group's multiplier as +0, never the invalid -0 word.
    scales = torch.maximum(positive / 7.0, negative / 8.0).abs().to(torch.float16)
    if not bool(torch.isfinite(scales).all()):
        raise ValueError("U4Z8 group multiplier exceeds binary16")
    decoded = scales.float()
    nonzero = decoded > 0
    ratios = torch.where(
        nonzero[..., None], blocks / torch.where(nonzero, decoded, 1.0)[..., None], 0.0
    )
    codes = (torch.round(ratios).clamp(-8.0, 7.0) + 8.0).to(torch.uint8)
    return codes.reshape(rows, columns), scales
