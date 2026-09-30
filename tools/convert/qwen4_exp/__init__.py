"""Qwen3.8-Flash-Next (`qwen4_exp`) artifact tooling.

The v3 artifact is derived from the published v2 NInfer artifact: `upgrade` copies every
v2 payload byte-for-byte into a new v3 directory and bakes the two BF16 MTP expert banks
into NVFP4 expert banks; `verify` re-reads the result through the generic v3 reader; and
`derive` downloads, upgrades and verifies unattended on a Linux VM.
"""
