"""Numerical algorithms producing codes and stored scales from logical values.

groupwise covers Q4/Q5/Q6/Q8; fp8_row and fp8_block share BF16-scale reduction
rules. fp8_row also provides FP32 row multipliers. nvfp4 covers matrices and
expert banks; u4z8 covers U4Z8 G16 tables. Exact packing belongs to
tools.artifact.codecs.
"""
