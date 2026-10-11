"""Pinned inputs and qualified outputs of the Flash-Next v2-to-v3 upgrade."""

from __future__ import annotations

import hashlib
from pathlib import Path

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

# v3 carries the maintained template instead. The official Flash-Next template is byte-identical
# to the official Qwen3.8-27B template, which rejects developer messages, later system messages
# and tool histories without a user turn; tools/chat_templates/qwen3_8.jinja accepts them.
TEMPLATE_OBJECT = "frontend/chat_template.jinja"
MAINTAINED_TEMPLATE = "tools/chat_templates/qwen3_8.jinja"
MAINTAINED_TEMPLATE_SHA256 = "1080e714707a3a47815f1affa79772996b7c13436cf32ddaa21f21184059ac9e"


def maintained_template() -> bytes:
    """The maintained template's committed (LF) bytes, whatever the checkout's line endings."""

    data = (Path(__file__).resolve().parents[3] / MAINTAINED_TEMPLATE).read_bytes()
    data = data.replace(b"\r\n", b"\n")
    digest = hashlib.sha256(data).hexdigest()
    if digest != MAINTAINED_TEMPLATE_SHA256:
        raise ValueError(f"{MAINTAINED_TEMPLATE} {digest} != {MAINTAINED_TEMPLATE_SHA256}")
    return data

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

# SHA256 of every file of the v3 derivation (artifact_id 3cade505298b575e009fe78877b76a33, with
# text eos_token_id and the maintained chat template), in `files` order (entry, part-0001,
# part-0002, part-0003).
V3_FILE_SHA256: tuple[str, ...] = (
    "f05abe459b4062af844e6c4ba6585feb97b8e3c9f016bac02815d483e2d51797",
    "69425b82468c039cb837335289daa9c9eaea17216c957dae3f633bdc7aa92ea3",
    "a39946035aa14f8f9e910ba371662b1dfb33d6f3df27af41ec31b875a8e8216e",
    "1deec0ae25095afcd18f2e8c8e8471bf2dda631fc506960304b791d60a477be4",
)
