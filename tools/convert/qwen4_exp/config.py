"""Normalize the official Qwen3.8-Flash-Next config.json into v3 component configs.

The text component records the text-only ``Qwen4ExpForCausalLM`` / ``qwen4_exp_text`` pair,
mirroring how ``qwen3_5`` records ``Qwen3_5*ForCausalLM`` for a ``...ForConditionalGeneration``
checkpoint. Only facts the ``qwen4_exp`` architecture interprets are kept.
"""

from __future__ import annotations

import json
from pathlib import Path
import struct

SOURCE_CONFIG = Path(__file__).with_name("source_config.json")
TEXT_ARCHITECTURE = "Qwen4ExpForCausalLM"
TEXT_MODEL_TYPE = "qwen4_exp_text"
MTP_CONFIG = {"architectures": ["Qwen4ExpMTP"]}

_TEXT_INTEGERS = (
    "hidden_size",
    "vocab_size",
    "num_hidden_layers",
    "max_position_embeddings",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "linear_num_key_heads",
    "linear_key_head_dim",
    "linear_num_value_heads",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
    "hc_count",
    "hc_lowrank",
    "indexer_n_heads",
    "indexer_kv_heads",
    "indexer_head_dim",
    "indexer_compress_ratio",
    "indexer_budget",
    "ple_embed_dim",
    "ple_conv_kernel_size",
    "ngram_size",
    "heads_per_ngram",
    "split_ngram_parts",
)
_VISION_INTEGERS = (
    "depth",
    "hidden_size",
    "intermediate_size",
    "num_heads",
    "patch_size",
    "temporal_patch_size",
    "spatial_merge_size",
    "num_position_embeddings",
)


def _f32(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label}: expected a number")
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _positive(raw: dict, key: str, label: str) -> int:
    value = raw.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label}.{key}: expected a positive integer, got {value!r}")
    return value


def _fixed(raw: dict, key: str, value: object, label: str) -> None:
    if raw.get(key) != value:
        raise ValueError(f"{label}.{key}: expected {value!r}, got {raw.get(key)!r}")


def load_source() -> dict:
    return json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))


def text_config(source: dict) -> dict:
    if source.get("architectures") != ["Qwen4ExpForConditionalGeneration"]:
        raise ValueError(f"unsupported architecture {source.get('architectures')!r}")
    raw = source["text_config"]
    _fixed(raw, "model_type", TEXT_MODEL_TYPE, "text")
    _fixed(raw, "hidden_act", "silu", "text")
    _fixed(raw, "attention_bias", False, "text")
    _fixed(raw, "output_gate_type", "sigmoid", "text")
    _fixed(raw, "mamba_ssm_dtype", "float32", "text")
    _fixed(raw, "mtp_num_hidden_layers", 1, "text")
    _fixed(raw, "mtp_use_dedicated_embeddings", False, "text")
    _fixed(raw, "tie_word_embeddings", False, "text")
    result: dict = {"architectures": [TEXT_ARCHITECTURE], "model_type": TEXT_MODEL_TYPE}
    for key in _TEXT_INTEGERS:
        result[key] = _positive(raw, key, "text")
    result["tie_word_embeddings"] = False
    result["rms_norm_eps"] = _f32(raw.get("rms_norm_eps"), "text.rms_norm_eps")
    layers = raw.get("layer_types")
    if (
        not isinstance(layers, list)
        or len(layers) != result["num_hidden_layers"]
        or any(kind not in ("full_attention", "linear_attention") for kind in layers)
    ):
        raise ValueError("text.layer_types must describe every decoder layer")
    result["layer_types"] = list(layers)
    rope = raw.get("rope_parameters")
    if not isinstance(rope, dict) or rope.get("mrope_interleaved") is not True:
        raise ValueError("text.rope_parameters must select interleaved MRoPE")
    sections = rope.get("mrope_section")
    factor = _f32(rope.get("partial_rotary_factor"), "text.partial_rotary_factor")
    if (
        not isinstance(sections, list)
        or len(sections) != 3
        or sum(sections) * 2 != int(result["head_dim"] * factor)
    ):
        raise ValueError("MRoPE sections and rotary width disagree")
    result["rope_parameters"] = {
        "rope_theta": _f32(rope.get("rope_theta"), "text.rope_theta"),
        "partial_rotary_factor": factor,
        "mrope_section": list(sections),
    }
    # The source counts PLE layers from 1; its weights belong to 0-based decoder layer 1.
    if raw.get("ple_layer_ids") != [2]:
        raise ValueError("text.ple_layer_ids: expected [2]")
    result["ple_layer_ids"] = [2]
    if result["num_experts_per_tok"] > result["num_experts"]:
        raise ValueError("selected experts exceed the expert count")
    if result["num_attention_heads"] % result["num_key_value_heads"]:
        raise ValueError("attention heads must be divisible by KV heads")
    if result["linear_num_value_heads"] % result["linear_num_key_heads"]:
        raise ValueError("linear value heads must be divisible by key heads")
    return result


def vision_config(source: dict, text: dict) -> dict:
    raw = source["vision_config"]
    _fixed(raw, "in_channels", 3, "vision")
    _fixed(raw, "hidden_act", "gelu_pytorch_tanh", "vision")
    _fixed(raw, "deepstack_visual_indexes", [], "vision")
    _fixed(raw, "out_hidden_size", text["hidden_size"], "vision")
    result = {"model_type": "qwen4_exp_vision"}
    for key in _VISION_INTEGERS:
        result[key] = _positive(raw, key, "vision")
    result["out_hidden_size"] = text["hidden_size"]
    return result
