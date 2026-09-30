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
MTP_BANK_SHA256 = {
    "mtp/layer/mlp/experts/gate_up": (
        "9e25663a0c5435ac498c8f3032db9c745909eeded17aba853f36685a2087b1f6"
    ),
    "mtp/layer/mlp/experts/down": (
        "fa78245bce9efe1073a7eaef0a4f6f294ce521ce1094d6d69f846f46fb8c9f60"
    ),
}

# The qualified v3 artifact (artifact_id 1e7e026e9f634926ae26b80f0fc8591e): SHA256 of every file,
# in `files` order (entry, part-0001, part-0002, part-0003).
V3_FILE_SHA256 = (
    "0de7b5f6c3ac9719e1c6e5811a765fda15fb4618444343502fc798dff2282714",
    "e5d74e6a4874ef270f000799db5e171786dd685fb4c0d14c0baa075cae13cda8",
    "67a30b1e718f2205d45ed22235ea93d1c0f9f0bf7cba963155002b5bccbf7baa",
    "8bc02c7328ff27ce22858ce447afeb5a60d4d01f1f3a12955f9e2a5220780565",
)
