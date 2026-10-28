"""Method requests/results and the built-in adapters to quantization or encoded import.

Methods own input traversal, chunking, and auxiliary values. Numerical algorithms
return codes/scales; artifact.tensor_output owns their physical byte placement.
User methods accept the same PrepareRequest and return a PreparedMethod.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from math import prod
import struct
from typing import Callable, Mapping

import torch

from tools.artifact.formats import (
    DirectFormat,
    QuantFormat,
    get_format,
    valid_positive_fp32_word,
)
from tools.artifact.layouts import BATCHED_BLOCK_SCALE_K16_M128X4_V1
from tools.artifact.schema import TensorSpec
from tools.artifact.tensor_output import TensorOutput

from .quantization.fp8_row import quantize_bf16_rows, quantize_bf16_rows_fp32_scale
from .quantization.groupwise import quantize_matrix
from .quantization.nvfp4 import encode_rows as encode_nvfp4_rows
from .quantization.nvfp4 import weight_divisor as nvfp4_weight_divisor
from .quantization.u4z8 import encode_rows as encode_u4z8_rows
from .quantization.stored_scale import quantize_stored_scale
from .sources.logical import EncodedRows, LogicalSource

UseKey = tuple[str, str]
AuxiliaryKey = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class MethodInput:
    parameter: str
    source: LogicalSource
    uses: tuple[UseKey, ...]


@dataclass(frozen=True, slots=True)
class AuxiliaryValue:
    format: str
    shape: tuple[int, ...]
    data: bytes

    @classmethod
    def activation_divisor(cls, value: bytes | float) -> AuxiliaryValue:
        raw = value if isinstance(value, bytes) else struct.pack("<f", value)
        if len(raw) != 4 or not valid_positive_fp32_word(struct.unpack("<I", raw)[0]):
            raise ValueError("activation input divisor must be positive finite FP32")
        return cls("fp32", (), raw)


@dataclass(frozen=True, slots=True)
class PreparedMethod:
    produce: Callable[[TensorOutput], None]
    auxiliaries: Mapping[AuxiliaryKey, AuxiliaryValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PrepareRequest:
    target: TensorSpec
    inputs: tuple[MethodInput, ...]
    policies: Mapping[UseKey, str]
    parameters: Mapping[str, object]
    device: str = "cuda"
    rows_per_chunk: int = 512
    auxiliary_overrides: Mapping[AuxiliaryKey, AuxiliaryValue] = field(
        default_factory=dict
    )
    source_offsets: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        values = [0]
        for item in self.inputs:
            values.append(values[-1] + prod(item.source.shape))
        object.__setattr__(self, "source_offsets", tuple(values))

    @property
    def source(self) -> LogicalSource:
        if len(self.inputs) != 1:
            raise ValueError("this method has multiple inputs; use request.inputs")
        return self.inputs[0].source

    def job(self, *, produce, auxiliaries=None) -> PreparedMethod:
        return PreparedMethod(produce, {} if auxiliaries is None else dict(auxiliaries))

    def values(self, begin: int, end: int) -> torch.Tensor:
        """Read the ordered logical inputs as one C-order parent element sequence."""
        pieces = []
        if not 0 <= begin <= end <= self.source_offsets[-1]:
            raise ValueError(f"{self.target.id}: requested values exceed method inputs")
        index = bisect_right(self.source_offsets, begin) - 1
        while begin < end:
            high = min(end, self.source_offsets[index + 1])
            pieces.append(
                self.inputs[index].source.values(
                    begin - self.source_offsets[index],
                    high - self.source_offsets[index],
                )
            )
            begin = high
            index += 1
        if not pieces:
            return torch.empty(0)
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces)

    def encoded_rows(self, begin: int, end: int) -> EncodedRows:
        pieces = []
        cursor = 0
        for item in self.inputs:
            source = item.source
            if len(source.shape) != 2 or source.shape[1] != self.target.shape[-1]:
                raise ValueError(
                    f"{item.parameter}: encoded grouping requires complete rows"
                )
            low, high = max(begin, cursor), min(end, cursor + source.shape[0])
            if low < high:
                if source.read_encoded is None:
                    raise ValueError(f"{item.parameter}: encoded rows are unavailable")
                pieces.append(source.read_encoded(low - cursor, high - cursor))
            cursor += source.shape[0]
        if not pieces or end > cursor:
            raise ValueError("encoded method input range is invalid")
        first = pieces[0]
        if any(
            (p.format, p.weight_divisor) != (first.format, first.weight_divisor)
            for p in pieces
        ):
            raise ValueError("encoded inputs cannot share one parent format/divisor")
        if len(pieces) == 1:
            return first
        return EncodedRows(
            first.format,
            torch.cat([p.codes for p in pieces]),
            torch.cat([p.scales for p in pieces]),
            first.weight_divisor,
        )


Method = Callable[[PrepareRequest], PreparedMethod]
_DIRECT_DTYPES = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "int32": torch.int32,
    "int64": torch.int64,
}
_FP8_ROW_QUANTIZERS = {
    "fp8_e4m3fn_row_bf16": quantize_bf16_rows,
    "fp8_e4m3fn_row_fp32": quantize_bf16_rows_fp32_scale,
}
# Tall, narrow tables (PLE shards are [2.5M,160]) read at least this many values per chunk.
_MIN_CHUNK_VALUES = 1 << 22


def _preflight(request: PrepareRequest, *, values: bool = True) -> None:
    if request.parameters:
        raise ValueError(
            f"{request.target.id}: this method accepts no numerical parameters"
        )
    if prod(request.target.shape) != sum(
        prod(item.source.shape) for item in request.inputs
    ):
        raise ValueError(
            f"{request.target.id}: parent and logical source counts differ"
        )
    if type(request.rows_per_chunk) is not int or request.rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive")
    if values:
        for item in request.inputs:
            item.source.values(0, 1)


def cast_direct(request: PrepareRequest) -> PreparedMethod:
    """Convert values at the explicit target BF16/FP32/INT32/INT64 boundary."""
    if not isinstance(get_format(request.target.format), DirectFormat):
        raise ValueError("cast_direct requires a direct target format")
    _preflight(request)
    dtype = _DIRECT_DTYPES[request.target.format]
    chunk = request.rows_per_chunk * (
        request.target.shape[-1] if len(request.target.shape) > 1 else 1
    )
    elements = prod(request.target.shape)

    def produce(output):
        for begin in range(0, elements, chunk):
            values = request.values(begin, min(elements, begin + chunk))
            if dtype == torch.int32 and (
                bool((values < -(1 << 31)).any())
                or bool((values > (1 << 31) - 1).any())
            ):
                raise ValueError(
                    "int32 conversion source is outside the representable range"
                )
            values = values.to(dtype=dtype)
            output.write_values(begin, values)

    return request.job(produce=produce)


def grouped_absmax(request: PrepareRequest) -> PreparedMethod:
    """Use the existing grouped max-abs, FP16-scale and code-rounding algorithm."""
    if (
        not isinstance(get_format(request.target.format), QuantFormat)
        or len(request.target.shape) != 2
    ):
        raise ValueError("grouped_absmax requires a grouped-integer matrix target")
    _preflight(request)
    n, k = request.target.shape

    def produce(output):
        for begin in range(0, n, request.rows_per_chunk):
            end = min(n, begin + request.rows_per_chunk)
            values = request.values(begin * k, end * k).reshape(end - begin, k)
            if not values.dtype.is_floating_point:
                raise TypeError(
                    "grouped_absmax source must provide floating-point values"
                )
            encoded = quantize_matrix(
                values, request.target.format, device=request.device
            )
            output.write_codes(begin, encoded.codes, encoded.scales)

    return request.job(produce=produce)


def fp8_row_maxabs(request: PrepareRequest) -> PreparedMethod:
    """Round inputs to BF16, then quantize to E4M3FN codes with one multiplier per row.

    The multiplier is ``binary32(max|w| / 448)``, rounded to BF16 for ``fp8_e4m3fn_row_bf16``
    and kept as FP32 for ``fp8_e4m3fn_row_fp32``.
    """
    quantize = _FP8_ROW_QUANTIZERS.get(request.target.format)
    if quantize is None or len(request.target.shape) != 2:
        raise ValueError("fp8_row_maxabs requires a row-scaled FP8 matrix format")
    _preflight(request)
    n, k = request.target.shape

    def produce(output):
        for begin in range(0, n, request.rows_per_chunk):
            end = min(n, begin + request.rows_per_chunk)
            values = (
                request.values(begin * k, end * k)
                .reshape(end - begin, k)
                .to(torch.bfloat16)
            )
            encoded = quantize(values)
            output.write_codes(begin, encoded.codes, encoded.scales)

    return request.job(produce=produce)


def import_encoded(request: PrepareRequest) -> PreparedMethod:
    """Preserve FP8/NVFP4 words, with independent divisors for bank matrices."""
    block_fp8 = request.target.format == "fp8_e4m3fn_block128x128_bf16"
    if (
        request.target.format
        not in ("nvfp4", "fp8_e4m3fn_row_bf16", "fp8_e4m3fn_block128x128_bf16")
        or len(request.target.shape) not in (2, 3)
        or (
            len(request.target.shape) == 3
            and request.target.format == "fp8_e4m3fn_row_bf16"
        )
    ):
        raise ValueError("import_encoded requires an encoded matrix or bank target")
    _preflight(request, values=False)
    n, k = request.target.shape[-2:]
    matrices = request.target.shape[0] if len(request.target.shape) == 3 else 1
    auxiliaries = {}
    divisors = {}
    cursor = 0
    for item in request.inputs:
        source = item.source
        if len(source.shape) != 2 or source.shape[1] != k:
            raise ValueError(
                f"{item.parameter}: encoded grouping requires complete rows"
            )
        if block_fp8 and source.shape[0] % 128:
            raise ValueError(
                f"{item.parameter}: FP8 source must contain whole 128-row blocks"
            )
        if source.read_encoded is None:
            raise ValueError(f"{item.parameter}: encoded rows are unavailable")
        first = source.read_encoded(0, 128 if block_fp8 else 1)
        if first.format != request.target.format:
            raise ValueError(
                f"{item.parameter}: source {first.format} differs from target {request.target.format}"
            )
        if first.format == "nvfp4":
            for matrix in range(cursor // n, (cursor + source.shape[0] - 1) // n + 1):
                if matrix in divisors and divisors[matrix] != first.weight_divisor:
                    raise ValueError("NVFP4 weight divisors differ within one matrix")
                divisors[matrix] = first.weight_divisor
            for parameter, input_name in item.uses:
                key = (parameter, input_name, "activation_input_divisor")
                if key in request.auxiliary_overrides:
                    value = request.auxiliary_overrides[key]
                elif request.policies[(parameter, input_name)] == "AllowA4":
                    if source.input_divisor is None:
                        raise ValueError(
                            f"{parameter}: supply an activation divisor for AllowA4"
                        )
                    value = AuxiliaryValue.activation_divisor(source.input_divisor())
                else:
                    continue
                if key in auxiliaries and auxiliaries[key] != value:
                    raise ValueError(
                        f"{parameter}: activation divisors differ within one Use"
                    )
                auxiliaries[key] = value
        cursor += source.shape[0]
    chunk = (
        max(128, request.rows_per_chunk // 128 * 128)
        if request.target.format == "nvfp4" or block_fp8
        else request.rows_per_chunk
    )

    def produce(output):
        for matrix in range(matrices):
            for begin in range(0, n, chunk):
                words = request.encoded_rows(
                    matrix * n + begin, matrix * n + min(n, begin + chunk)
                )
                output.write_codes(
                    begin,
                    words.codes,
                    words.scales,
                    words.weight_divisor,
                    matrix=matrix,
                )

    return request.job(produce=produce, auxiliaries=auxiliaries)


def nvfp4_maxabs(request: PrepareRequest) -> PreparedMethod:
    """Encode values with NVFP4_MAXABS_DIVISOR_RNE_V1 over the complete packed parent.

    Every AllowA4 Use needs a calibrated activation divisor from the recipe; the weight carries
    no source input scale to derive one from.
    """
    if request.target.format != "nvfp4" or len(request.target.shape) != 2:
        raise ValueError("nvfp4_maxabs requires an NVFP4 matrix target")
    _preflight(request)
    n, k = request.target.shape
    if n % 128 or k % 16:
        raise ValueError("NVFP4 parent requires complete 128-row and 16-column tiles")
    chunk = max(128, request.rows_per_chunk // 128 * 128)
    auxiliaries = {}
    for item in request.inputs:
        for use in item.uses:
            if request.policies[use] != "AllowA4":
                continue
            key = (*use, "activation_input_divisor")
            if key not in request.auxiliary_overrides:
                raise ValueError(f"{use[0]}: a calibrated activation divisor is required")
            auxiliaries[key] = request.auxiliary_overrides[key]

    def produce(output):
        # The divisor is a property of the whole parent, so it needs a first pass over every row.
        maximum = 0.0
        for begin in range(0, n, chunk):
            values = request.values(begin * k, min(n, begin + chunk) * k).float()
            if not bool(torch.isfinite(values).all()):
                raise ValueError("NVFP4 source contains NaN or infinity")
            maximum = max(maximum, float(values.abs().max()))
        divisor = nvfp4_weight_divisor(maximum)
        for begin in range(0, n, chunk):
            end = min(n, begin + chunk)
            values = request.values(begin * k, end * k).reshape(end - begin, k)
            codes, scales = encode_nvfp4_rows(values.to(request.device), divisor)
            output.write_codes(begin, codes.cpu(), scales.cpu(), divisor)

    return request.job(produce=produce, auxiliaries=auxiliaries)


def _supplied_divisors(request: PrepareRequest) -> dict:
    """Recipe-supplied activation divisors of the inputs' Uses, stored as given."""
    result = {}
    for item in request.inputs:
        for use in item.uses:
            key = (*use, "activation_input_divisor")
            if key in request.auxiliary_overrides:
                result[key] = request.auxiliary_overrides[key]
    return result


def nvfp4_batched_maxabs(request: PrepareRequest) -> PreparedMethod:
    """Encode an ``[E,N,K]`` NVFP4 expert bank with NVFP4_MAXABS_DIVISOR_RNE_V1 per expert.

    Expert ``e`` gets the divisor ``binary32(2688 / max|W_e|)`` of its complete ``[N,K]``
    matrix, so a fused gate/up bank shares one divisor between an expert's gate and up rows.
    Every AllowA4 Use supplies its stored global activation divisor. Recipes reproducing
    the fork's dynamic K16-group quantization explicitly supply a unit global divisor.
    """
    target = request.target
    if (
        target.format != "nvfp4"
        or len(target.shape) != 3
        or target.layout != BATCHED_BLOCK_SCALE_K16_M128X4_V1.name
    ):
        raise ValueError("nvfp4_batched_maxabs requires an NVFP4 expert-bank target")
    _preflight(request)
    experts, n, k = target.shape
    size = n * k
    auxiliaries = _supplied_divisors(request)
    for item in request.inputs:
        for use in item.uses:
            if request.policies[use] == "AllowA4" and (*use, "activation_input_divisor") not in auxiliaries:
                raise ValueError(f"{use[0]}: an explicit activation divisor is required")

    def produce(output):
        for expert in range(experts):
            values = request.values(expert * size, (expert + 1) * size).reshape(n, k)
            magnitude = values.float().abs()
            if not bool(torch.isfinite(magnitude).all()):
                raise ValueError(f"expert {expert}: NVFP4 source contains NaN or infinity")
            divisor = nvfp4_weight_divisor(float(magnitude.max()))
            codes, scales = encode_nvfp4_rows(values.to(request.device), divisor)
            output.write_codes(0, codes.cpu(), scales.cpu(), divisor, matrix=expert)

    return request.job(produce=produce, auxiliaries=auxiliaries)


def u4z8_maxabs(request: PrepareRequest) -> PreparedMethod:
    """Encode a table with U4Z8_RANGE_RNE_V1: per 16 values, the smallest binary16 multiplier
    that keeps positive values within +7 and negative values within -8, and round-to-nearest-even
    codes around the fixed zero point 8. Chunks hold at least 4 Mi values."""
    if request.target.format != "u4z8_g16_fp16" or len(request.target.shape) != 2:
        raise ValueError("u4z8_maxabs requires a U4Z8 matrix target")
    _preflight(request)
    n, k = request.target.shape
    rows = max(request.rows_per_chunk, -(-_MIN_CHUNK_VALUES // k))

    def produce(output):
        for begin in range(0, n, rows):
            end = min(n, begin + rows)
            values = request.values(begin * k, end * k).reshape(end - begin, k)
            codes, scales = encode_u4z8_rows(values.to(request.device))
            output.write_codes(begin, codes.cpu(), scales.cpu())

    return request.job(produce=produce)


def stored_scale_mse(request: PrepareRequest) -> PreparedMethod:
    """Two-pass object amax and exhaustive G16 stored-scale MSE encoding."""
    if (
        request.target.format not in ("nf4_g16_e4m3fn", "nvfp4")
        or len(request.target.shape) != 2
    ):
        raise ValueError("stored_scale_mse requires an NF4 or NVFP4 matrix")
    _preflight(request)
    n, k = request.target.shape
    chunk = request.rows_per_chunk
    if request.target.format == "nvfp4":
        chunk = max(128, chunk // 128 * 128)

    def produce(output):
        maximum = 0.0
        for begin in range(0, n, chunk):
            end = min(n, begin + chunk)
            values = request.values(begin * k, end * k).to(
                device=request.device, dtype=torch.float32
            )
            if not bool(torch.isfinite(values).all()):
                raise ValueError("stored-scale source contains NaN or infinity")
            maximum = max(maximum, values.abs().max().item())
        limit = 448.0 if request.target.format == "nf4_g16_e4m3fn" else 2688.0
        divisor = struct.pack("<f", limit / maximum if maximum else 1.0)
        if not valid_positive_fp32_word(struct.unpack("<I", divisor)[0]):
            raise ValueError("amax produces an invalid FP32 object divisor")
        for begin in range(0, n, chunk):
            end = min(n, begin + chunk)
            values = request.values(begin * k, end * k).reshape(end - begin, k)
            encoded = quantize_stored_scale(
                values, request.target.format, divisor, device=request.device
            )
            output.write_codes(begin, encoded.codes, encoded.scales, divisor)

    return request.job(produce=produce)


METHODS: dict[str, Method] = {
    "cast_direct": cast_direct,
    "grouped_absmax": grouped_absmax,
    "fp8_row_maxabs": fp8_row_maxabs,
    "import_encoded": import_encoded,
    "nvfp4_maxabs": nvfp4_maxabs,
    "nvfp4_batched_maxabs": nvfp4_batched_maxabs,
    "u4z8_maxabs": u4z8_maxabs,
    "stored_scale_mse": stored_scale_mse,
}
