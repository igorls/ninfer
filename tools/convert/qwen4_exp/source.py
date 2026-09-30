"""Pinned inputs and qualified outputs of the Flash-Next v2-to-v3 upgrade."""

from __future__ import annotations

V2_REPOSITORY = "igorls/Qwen3.8-Flash-Next-mixed-NInfer"
V2_REVISION = "5f0ee7e24279cbadf8d4a90c93c2ff5ea6b00688"
V2_FILENAME = "qwen3_8_flash_next_mixed.ninfer"
V2_BYTES = 113_298_397_952
V2_SHA256 = "3d383e51963aafd4318dfd04c8dc63ee7df11768de19d9ab58dbba44460d1d02"
V2_MODEL_ID = "qwen3.8-flash-next"
V2_WEIGHTS_ID = "mixed-nvfp4-fp8-ple-int4"
V2_URL = f"https://huggingface.co/{V2_REPOSITORY}/resolve/{V2_REVISION}/{V2_FILENAME}"

# Official Qwen/Qwen3.8-Flash-Next config.json, stored beside this module.
SOURCE_CONFIG_REPOSITORY = "Qwen/Qwen3.8-Flash-Next"
SOURCE_CONFIG_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"

# The v2 artifact's frontend resources are the official ones (chat template c3cf9e34...).
CHAT_TEMPLATE_SHA256 = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"

# MTP expert banks baked from the v2 BF16 banks. Each digest equals the SHA256 of the
# device buffer the v2 loader quantizes at startup (`ninfer_quantize_mtp` at 87812bc8 on G4).
MTP_BANKS = ("mtp/layer/mlp/experts/gate_up", "mtp/layer/mlp/experts/down")
MTP_BANK_SHA256: dict[str, str] = {}

# SHA256 of every file of the qualified v3 artifact, in `files` order.
V3_FILE_SHA256: tuple[str, ...] = ()
