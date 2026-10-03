#!/usr/bin/env python3
"""Independent YaRN check for the Qwen3.8-Flash-Next long-context table.

Reimplements the Hugging Face `_compute_yarn_parameters` defaults used when the
published rope override omits beta_fast, beta_slow, truncate, and
attention_factor: beta_fast 32, beta_slow 1, truncate true, and
attention_factor = 0.1 * ln(factor) + 1. The literals are the same anchors the
C++ host test locks.
"""

import math
import sys

DIM = 64
THETA = 10_000_000.0
NATIVE = 262_144
FACTOR = 4.0
PAIRS = 32


def correction_dim(num_rotations: float) -> float:
    return (DIM * math.log(NATIVE / (num_rotations * 2.0 * math.pi))) / (2.0 * math.log(THETA))


def correction_bounds() -> tuple[float, float]:
    low = math.floor(correction_dim(32.0))
    high = math.ceil(correction_dim(1.0))
    low = max(low, 0.0)
    high = min(high, float(DIM - 1))
    if low == high:
        high += 0.001
    return low, high


def default_inv_freq(pair: int) -> float:
    return THETA ** ((-2.0 * pair) / DIM)


def yarn_inv_freq(pair: int) -> float:
    low, high = correction_bounds()
    ramp = (pair - low) / (high - low)
    ramp = min(max(ramp, 0.0), 1.0)
    extrapolation = 1.0 / (THETA ** ((2.0 * pair) / DIM))
    interpolation = extrapolation / FACTOR
    # HF mask is 1 on the extrapolation side and falls across the ramp.
    mask = 1.0 - ramp
    return interpolation * (1.0 - mask) + extrapolation * mask


def attention_factor() -> float:
    return 0.1 * math.log(FACTOR) + 1.0


def rotate(first: float, second: float, position: int, inv_freq: float, scale: float):
    angle = position * inv_freq
    sine = math.sin(angle) * scale
    cosine = math.cos(angle) * scale
    return first * cosine - second * sine, second * cosine + first * sine


def near(got: float, want: float, rel: float = 1.0e-12) -> bool:
    return abs(got - want) <= rel * max(abs(want), 1.0)


def main() -> int:
    failures = 0

    def expect(condition: bool, message: str) -> None:
        nonlocal failures
        if not condition:
            print(f"FAIL: {message}", file=sys.stderr)
            failures += 1

    low, high = correction_bounds()
    expect(low == 14.0 and high == 22.0, f"bounds {low}, {high}")
    expect(near(attention_factor(), 1.1386294361119891), "attention factor")

    anchors = {
        0: (1.0000000000000000e00, 1.0),
        14: (8.6596432336006539e-04, 1.0),
        15: (4.7423982268010459e-04, 0.90625),
        16: (2.5693505988868084e-04, 0.8125),
        17: (1.3734974507600040e-04, 0.71875),
        18: (7.2173874043091133e-05, 0.625),
        19: (3.7072249820680398e-05, 0.53125),
        20: (1.8449222025000475e-05, 0.4375),
        21: (8.7597700711790042e-06, 0.34375),
        22: (3.8498163151487305e-06, 0.25),
        31: (4.1370427498579540e-08, 0.25),
    }
    for pair, (want, ratio) in anchors.items():
        got = yarn_inv_freq(pair)
        expect(near(got, want), f"pair {pair} inv {got}")
        expect(near(got / default_inv_freq(pair), ratio, 1.0e-9), f"pair {pair} ratio")

    factor = attention_factor()
    origin = rotate(1.0, 0.5, 0, yarn_inv_freq(0), factor)
    expect(near(origin[0], 1.1386294361119891) and near(origin[1], 0.5693147180559945),
           "position 0 scale")
    plain = rotate(1.0, 0.5, 0, default_inv_freq(0), 1.0)
    expect(near(plain[0], 1.0) and near(plain[1], 0.5), "position 0 default")

    rotations = [
        (100, 0, 1.0, 0.5, False, 1.1155016928425634, -0.07520620496591685),
        (100, 0, 1.0, 0.5, True, 1.270143063503297, -0.08563199875246463),
        (100, 31, 1.0, 0.5, False, 0.9999917257775797, 0.5000165481025381),
        (100, 31, 1.0, 0.5, True, 1.1386270808229184, 0.569319428609776),
        (262144, 31, 1.0, 0.5, False, 0.9773760172024913, 0.5428960499003426),
        (262144, 31, 1.0, 0.5, True, 1.132388374847845, 0.5816294434477498),
        (300000, 31, 1.0, -0.25, False, 1.011173995079255, -0.20006786767359366),
        (300000, 31, 1.0, -0.25, True, 1.1420745705075712, -0.27050412238900834),
        (999999, 22, 0.125, 0.5, False, -0.271007217613324, -0.4383834942165186),
        (999999, 22, 0.125, 0.5, True, 0.2622277521568631, -0.5249888638877341),
    ]
    for position, pair, x0, x1, yarn, want0, want1 in rotations:
        inv = yarn_inv_freq(pair) if yarn else default_inv_freq(pair)
        got0, got1 = rotate(x0, x1, position, inv, factor if yarn else 1.0)
        expect(near(got0, want0, 1.0e-9) and near(got1, want1, 1.0e-9),
               f"rotation {position}/{pair}")

    expect(PAIRS == 32, "pair count")
    if failures:
        print(f"{failures} YaRN checks failed", file=sys.stderr)
        return 1
    print("PASS: flash-next yarn rope")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
