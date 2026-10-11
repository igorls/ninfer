from __future__ import annotations

import pytest
import torch

from tools.artifact.codecs.packed_u4 import dequantize_u4z8_g16, encode_u4z8_g16
from tools.convert.quantization.u4z8 import encode_rows


def test_u4z8_range_fit_uses_both_code_extremes_and_signed_zero_groups():
    rows = torch.stack(
        (
            torch.arange(-8, 8, dtype=torch.float32) * 0.5,  # p/7 = m/8 = 0.5
            torch.arange(0, 16, dtype=torch.float32) * (7 / 15),  # positive: p/7 = 1
            -torch.arange(0, 16, dtype=torch.float32) * 0.5,  # negative: m/8 = 0.9375
            torch.zeros(16),
            -torch.zeros(16),
        )
    ).to(torch.bfloat16)
    codes, scales = encode_rows(rows)
    assert scales[:, 0].tolist() == [0.5, 1.0, 0.9375, 0.0, 0.0]
    assert scales.view(torch.int16)[3:, 0].tolist() == [0, 0]  # +0, never the invalid -0 word
    assert codes[0].tolist() == list(range(16))
    assert codes[1, -1] == 15 and codes[1, 0] == 8
    assert codes[2, -1] == 0 and codes[2, 0] == 8
    assert codes[3:].eq(8).all()


def test_u4z8_saturates_after_multiplier_rounding_and_rejects_overflow():
    # A subnormal binary16 multiplier rounds 1.4 * 2^-24 down to 2^-24; x / s = 9.8 saturates.
    x = 7 * 1.4 * 2.0**-24
    codes, scales = encode_rows(torch.full((1, 16), x))
    assert float(scales[0, 0]) == 2.0**-24
    assert codes.eq(15).all()
    with pytest.raises(ValueError, match="binary16"):
        encode_rows(torch.full((1, 16), 1e6))
    with pytest.raises(ValueError, match="16-value groups"):
        encode_rows(torch.zeros(2, 8))


def test_u4z8_words_round_trip_through_the_layout_codec():
    generator = torch.Generator().manual_seed(5)
    values = (torch.randn(9, 32, generator=generator) * 0.3).to(torch.bfloat16)
    codes, scales = encode_rows(values)
    decoded = dequantize_u4z8_g16(encode_u4z8_g16(codes, scales, values.shape), values.shape)
    error = (decoded - values.float()).abs()
    bound = scales.float().repeat_interleave(16, dim=1) / 2
    assert bool((error <= bound * (1 + 1e-6)).all())
