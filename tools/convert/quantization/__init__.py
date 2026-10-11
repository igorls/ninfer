"""Numerical algorithms producing codes and stored scales from logical values.

groupwise covers Q4/Q5/Q6/Q8 using one parameterized algorithm; fp8_row covers
BF16-to-row-scaled FP8 with BF16 or FP32 multipliers; nvfp4 covers NVFP4 matrices and
per-expert banks; u4z8 covers U4Z8 G16 tables. Exact byte packing belongs to
tools.artifact.codecs.
"""
