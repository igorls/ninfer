from __future__ import annotations

import struct

import pytest
import torch

from tools.artifact.layouts import (
    block_scale_geometry,
    encoded_size,
    expert_block_scale_geometry,
)
from tools.artifact.codecs.direct import decode_direct, encode_direct
from tools.artifact.codecs.nvfp4 import (
    decode_nvfp4_expert_bank_words,
    decode_nvfp4_words,
    encode_nvfp4,
    encode_nvfp4_expert_bank,
)
from tools.artifact.codecs.packed_u4 import (
    decode_u4z8_g16_words,
    dequantize_u4z8_g16,
    encode_u4z8_g16,
)
from tools.artifact.codecs.row_split import decode_row_split_codes, encode_row_split


def _signed_word(word: int, bits: int) -> int:
    return word if word < 1 << (bits - 1) else word - (1 << bits)


@pytest.mark.parametrize(
    ("format_name", "tensor", "words", "word_format", "word_view"),
    [
        (
            "bf16",
            torch.tensor(
                [_signed_word(word, 16) for word in (0x0000, 0x8000, 0x0001, 0x7FC1)],
                dtype=torch.int16,
            ).view(torch.bfloat16),
            (0x0000, 0x8000, 0x0001, 0x7FC1),
            "H",
            torch.int16,
        ),
        (
            "fp32",
            torch.tensor(
                [
                    _signed_word(word, 32)
                    for word in (0x00000000, 0x80000000, 0x00000001, 0x7FC01234)
                ],
                dtype=torch.int32,
            ).view(torch.float32),
            (0x00000000, 0x80000000, 0x00000001, 0x7FC01234),
            "I",
            torch.int32,
        ),
        (
            "int32",
            torch.tensor((0, -1, -(1 << 31), (1 << 31) - 1), dtype=torch.int32),
            (0, -1, -(1 << 31), (1 << 31) - 1),
            "i",
            torch.int32,
        ),
        (
            "int64",
            torch.tensor((0, -1, -(1 << 63), (1 << 40) + 3), dtype=torch.int64),
            (0, -1, -(1 << 63), (1 << 40) + 3),
            "q",
            torch.int64,
        ),
    ],
)
def test_direct_layout_preserves_exact_little_endian_words(
    format_name, tensor, words, word_format, word_view
):
    expected = struct.pack("<" + word_format * len(words), *words)
    payload = encode_direct(tensor, format_name)
    assert payload == expected
    decoded = decode_direct(payload, format_name, tensor.shape)
    assert torch.equal(decoded.view(word_view), tensor.view(word_view))

    if format_name == "bf16":
        with pytest.raises(TypeError):
            encode_direct(tensor.float(), format_name)


@pytest.mark.parametrize(
    (
        "format_name",
        "k",
        "group_size",
        "k_pad",
        "scale_offset",
        "prefix",
        "base_prefix",
        "high_prefix",
    ),
    [
        pytest.param(
            "q4_g64_fp16",
            65,
            64,
            128,
            256,
            (-8, -7, -1, 0, 1, 7),
            b"\x98\x0f\x71",
            b"",
            id="q4",
        ),
        pytest.param(
            "q5_g64_fp16",
            130,
            64,
            256,
            512,
            (-16, -15, -1, 0, 1, 15),
            b"\x10\x0f\xf1",
            b"\x07",
            id="q5",
        ),
        pytest.param(
            "q6_g64_fp16",
            65,
            64,
            128,
            512,
            (-32, -31, -17, -16, -1, 0, 15, 31),
            b"\x10\x0f\x0f\xff",
            b"\xea\x43",
            id="q6",
        ),
        pytest.param(
            "q8_g32_fp16",
            33,
            32,
            128,
            256,
            (-127, -1, 0, 1, 127),
            b"\x81\xff\x00\x01\x7f",
            b"",
            id="q8",
        ),
    ],
)
def test_row_split_matches_known_packed_bytes(
    format_name, k, group_size, k_pad, scale_offset, prefix, base_prefix, high_prefix
):
    groups = k_pad // group_size
    codes = torch.zeros((1, groups, group_size), dtype=torch.int8)
    codes[0, 0, : len(prefix)] = torch.tensor(prefix, dtype=torch.int8)
    scales = torch.zeros((1, groups), dtype=torch.float16)
    scales[0, :2] = torch.tensor([1.5, 0.25], dtype=torch.float16)

    # Literal wire positions and words come from the format, independently of layout helpers.
    expected = bytearray(scale_offset + groups * 2)
    expected[: len(base_prefix)] = base_prefix
    expected[256 : 256 + len(high_prefix)] = high_prefix
    expected[scale_offset:] = struct.pack(
        "<" + "H" * groups, 0x3E00, 0x3400, *([0] * (groups - 2))
    )
    assert encode_row_split(codes, scales, format_name, (1, k)) == expected
    decoded_scales, decoded_codes = decode_row_split_codes(
        expected, format_name, (1, k)
    )
    assert torch.equal(decoded_scales, scales)
    assert torch.equal(decoded_codes, codes)


def test_nvfp4_known_vector_geometry_swizzle_tail_and_round_trip():
    shape = (128, 64)
    geometry = block_scale_geometry("nvfp4", shape)
    assert (
        geometry.code_plane_bytes,
        geometry.scale_plane_offset,
        geometry.scale_plane_bytes,
        geometry.weight_divisor_offset,
        geometry.payload_bytes,
    ) == (4096, 4096, 512, 4608, 4612)

    packed = (
        torch.arange(geometry.code_plane_bytes, dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(128, 32)
    )
    packed[0, 0] = 0x10
    scales = (
        torch.arange(128 * 4, dtype=torch.int64)
        .remainder(0x7F)
        .to(torch.uint8)
        .reshape(128, 4)
    )
    divisor = struct.pack("<f", 2.5)
    payload = encode_nvfp4(packed, scales, divisor, shape)

    assert len(payload) == 4612
    assert payload[0] == 0x10  # low nibble is K=0; high nibble is K=1.
    for row, lane in ((0, 0), (31, 3), (32, 0), (127, 3)):
        offset = geometry.scale_plane_offset + (row % 32) * 16 + (row // 32) * 4 + lane
        assert payload[offset] == int(scales[row, lane])
    assert payload[geometry.weight_divisor_offset :] == divisor

    decoded_packed, decoded_scales, decoded_divisor = decode_nvfp4_words(payload, shape)
    assert torch.equal(decoded_packed, packed)
    assert torch.equal(decoded_scales, scales)
    assert bytes(decoded_divisor.reshape(1).view(torch.uint8).numpy()) == divisor


def test_nvfp4_expert_bank_concatenates_independent_expert_planes():
    shape = (3, 128, 64)
    geometry = expert_block_scale_geometry("nvfp4", shape)
    assert (
        geometry.code_plane_bytes,
        geometry.scale_plane_offset,
        geometry.scale_plane_bytes,
        geometry.weight_divisor_offset,
        geometry.payload_bytes,
    ) == (12288, 12288, 1536, 13824, 13836)
    assert encoded_size("expert_block_scale_k16_m128x4_v1", "nvfp4", shape) == 13836
    with pytest.raises(ValueError):
        encoded_size("expert_block_scale_k16_m128x4_v1", "nvfp4", (128, 64))

    generator = torch.Generator().manual_seed(7)
    codes = torch.randint(0, 256, (3, 128, 32), dtype=torch.uint8, generator=generator)
    scales = torch.randint(0, 0x7F, (3, 128, 4), dtype=torch.uint8, generator=generator)
    divisors = torch.tensor([0.5, 2688.0, 3.25], dtype=torch.float32)
    payload = encode_nvfp4_expert_bank(codes, scales, divisors, shape)

    # Every expert is the registered rank-2 block-scale matrix, split by plane.
    for expert in range(3):
        matrix = encode_nvfp4(
            codes[expert], scales[expert], struct.pack("<f", divisors[expert]), (128, 64)
        )
        assert payload[expert * 4096 : (expert + 1) * 4096] == matrix[:4096]
        scale_begin = geometry.scale_plane_offset + expert * 512
        assert payload[scale_begin : scale_begin + 512] == matrix[4096:4608]
        divisor_begin = geometry.weight_divisor_offset + 4 * expert
        assert payload[divisor_begin : divisor_begin + 4] == matrix[4608:]

    decoded = decode_nvfp4_expert_bank_words(payload, shape)
    assert torch.equal(decoded[0], codes)
    assert torch.equal(decoded[1], scales)
    assert torch.equal(decoded[2], divisors)
    with pytest.raises(ValueError, match="finite and positive"):
        encode_nvfp4_expert_bank(codes, scales, torch.tensor([1.0, 0.0, 1.0]), shape)


def test_u4z8_packs_low_nibble_first_and_decodes_affine_values():
    shape = (2, 32)
    assert encoded_size("packed_u4_g16_v1", "u4z8_g16_fp16", shape) == 264
    with pytest.raises(ValueError):
        encoded_size("packed_u4_g16_v1", "u4z8_g16_fp16", (2, 24))

    codes = torch.full(shape, 8, dtype=torch.uint8)
    codes[0, :4] = torch.tensor([0, 15, 9, 7], dtype=torch.uint8)
    codes[1, 16] = 12
    scales = torch.tensor([[0.5, 0.0], [0.0, 0.25]], dtype=torch.float16)
    payload = encode_u4z8_g16(codes, scales, shape)

    assert payload[:2] == bytes((0xF0, 0x79))  # (k=0 low, k=1 high), (k=2 low, k=3 high)
    assert payload[24] == 0x8C  # row 1, k=16 low nibble 12, k=17 high nibble 8
    assert payload[32:256] == bytes(224)
    assert payload[256:] == struct.pack("<HHHH", 0x3800, 0, 0, 0x3400)

    decoded_codes, decoded_scales = decode_u4z8_g16_words(payload, shape)
    assert torch.equal(decoded_codes, codes)
    assert torch.equal(decoded_scales, scales)
    values = dequantize_u4z8_g16(payload, shape)
    assert values[0, :4].tolist() == [-4.0, 3.5, 0.5, -0.5]
    assert values[1, 16].item() == 1.0 and values[1, :16].abs().sum().item() == 0.0

    invalid = codes.clone()
    invalid[0, 16] = 9
    with pytest.raises(ValueError, match="zero-point code 8"):
        encode_u4z8_g16(invalid, scales, shape)
