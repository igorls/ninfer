"""Native official-safetensors conversion of the Qwen4 family (qwen4_exp) to v3 artifacts.

Synthetic tiny checkpoints use the official Qwen3.8-Flash-Next tensor names; the read-back checks
use the artifact contract (docs/maintainer/qwen3.8-flash-next-artifact.md), the documented storage
layouts and numeric rules, decoded here independently of the converter's codecs.
"""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys

import numpy as np
import pytest
from safetensors.torch import save_file
import torch

from tools.artifact.codecs.nvfp4 import encode_nvfp4_expert_bank
from tools.artifact.reader import Artifact
from tools.artifact.schema import binding_parts
from tools.convert import methods
from tools.convert.official_recipes import RECIPES, qwen4_allocation
from tools.convert.pipeline import convert
from tools.convert.quantization.nvfp4 import encode_rows, weight_divisor
from tools.convert.qwen4_exp import build
from tools.convert.qwen4_exp.config import load_source, text_config, vision_config
from tools.convert.qwen4_exp.inventory import _inputs as upgrade_inputs
from tools.convert.qwen4_exp.inventory import plan_upgrade
from tools.convert.recipe import Recipe
from tools.convert.sources.safetensors import SafetensorsSource, TensorInfo

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "qwen4_exp"
MAINTAINED = ROOT / "tools" / "chat_templates" / "qwen3_8.jinja"
FP8_FP32 = "fp8_e4m3fn_row_fp32"


# --- synthetic official checkpoints -------------------------------------------------------


def _text(*, moe: bool) -> dict:
    text = {
        "model_type": "qwen4_exp_text",
        "hidden_act": "silu",
        "attention_bias": False,
        "output_gate_type": "sigmoid",
        "mamba_ssm_dtype": "float32",
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "tie_word_embeddings": False,
        "hidden_size": 128,
        "vocab_size": 8,
        "num_hidden_layers": 2,
        "max_position_embeddings": 256,
        "rms_norm_eps": 1e-6,
        "layer_types": ["linear_attention", "full_attention"],
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
        "linear_num_key_heads": 1,
        "linear_key_head_dim": 16,
        "linear_num_value_heads": 2,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 3,
        "hc_count": 2,
        "hc_lowrank": 16,
        "indexer_n_heads": 2,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 16,
        "indexer_compress_ratio": 4,
        "indexer_budget": 8,
        "ple_embed_dim": 64,
        "ple_conv_kernel_size": 3,
        "ngram_size": 3,
        "heads_per_ngram": 2,
        "split_ngram_parts": 2,
        "ple_layer_ids": [1],
        "eos_token_id": 5,
    }
    if moe:
        text.update(
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=32,
        )
    else:
        text["intermediate_size"] = 96
    return text


VISION = {
    "depth": 1,
    "hidden_size": 32,
    "intermediate_size": 48,
    "num_heads": 2,
    "patch_size": 2,
    "temporal_patch_size": 1,
    "spatial_merge_size": 2,
    "num_position_embeddings": 4,
    "out_hidden_size": 128,
    "in_channels": 3,
    "hidden_act": "gelu_pytorch_tanh",
    "deepstack_visual_indexes": [],
}
PLE_ROWS = 40


def _source_config(*, moe: bool, nested: bool) -> dict:
    text = _text(moe=moe)
    if not nested:
        return {"architectures": ["Qwen4ExpForCausalLM"], **text}
    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "text_config": text,
        "vision_config": dict(VISION),
        "tie_word_embeddings": False,
    }


def _official_tensors(text: dict, *, nested: bool, seed: int) -> dict[str, torch.Tensor]:
    """Random tensors under the official names and axis orders of every component."""
    generator = torch.Generator().manual_seed(seed)

    def r(*shape, scale=0.05):
        return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16)

    h, v = text["hidden_size"], text["vocab_size"]
    s, low, streams = text["hc_count"] * h, text["hc_lowrank"], text["hc_count"]
    d, q, kv = (
        text["head_dim"],
        text["num_attention_heads"] * text["head_dim"],
        text["num_key_value_heads"] * text["head_dim"],
    )
    kg = text["linear_num_key_heads"] * text["linear_key_head_dim"]
    vg = text["linear_num_value_heads"] * text["linear_value_head_dim"]
    nv, taps = text["linear_num_value_heads"], text["linear_conv_kernel_dim"]
    tp = "model.language_model." if nested else "model."
    t: dict[str, torch.Tensor] = {}

    def hyper(prefix, block):
        t[prefix + "hc_norm.weight"] = r(s)
        t[prefix + "input_mix_weight_down.weight"] = r(low, s)
        t[prefix + "input_mix_weight_up.weight"] = r(s, low)
        if block:
            t[prefix + "block_inject_weight.weight"] = r(streams, s)

    def block(prefix, mixer):
        hyper(prefix + "attn_hyper_connection.", True)
        hyper(prefix + "mlp_hyper_connection.", True)
        if mixer == "full_attention":
            a = prefix + "self_attn."
            t[a + "q_proj.weight"] = r(2 * q, h)
            t[a + "k_proj.weight"] = r(kv, h)
            t[a + "v_proj.weight"] = r(kv, h)
            t[a + "o_proj.weight"] = r(h, q)
            t[a + "q_norm.weight"] = r(d)
            t[a + "k_norm.weight"] = r(d)
            rows = (text["indexer_n_heads"] + text["indexer_kv_heads"]) * text["indexer_head_dim"]
            t[a + "indexer.index_qk_proj.weight"] = r(rows, h)
            t[a + "indexer.q_layernorm.weight"] = r(text["indexer_head_dim"])
            t[a + "indexer.k_layernorm.weight"] = r(text["indexer_head_dim"])
        else:
            g = prefix + "linear_attn."
            t[g + "A_log"] = r(nv, scale=1.0)
            t[g + "dt_bias"] = r(nv, scale=1.0)
            t[g + "conv1d.weight"] = r(2 * kg + vg, 1, taps)
            t[g + "in_proj_a.weight"] = r(nv, h)
            t[g + "in_proj_b.weight"] = r(nv, h)
            t[g + "in_proj_qkv.weight"] = r(2 * kg + vg, h)
            t[g + "in_proj_z.weight"] = r(vg, h)
            t[g + "norm.weight"] = r(text["linear_value_head_dim"])
            t[g + "out_proj.weight"] = r(h, vg)
        m = prefix + "mlp."
        if "num_experts" in text:
            e, i, sh = (
                text["num_experts"],
                text["moe_intermediate_size"],
                text["shared_expert_intermediate_size"],
            )
            t[m + "gate.weight"] = r(e, h)
            t[m + "shared_expert_gate.weight"] = r(1, h)
            t[m + "shared_expert.gate_proj.weight"] = r(sh, h)
            t[m + "shared_expert.up_proj.weight"] = r(sh, h)
            t[m + "shared_expert.down_proj.weight"] = r(h, sh)
            gate_up = r(e, 2 * i, h)
            gate_up[1] = 0  # an all-zero expert keeps divisor 1 and zero scales
            gate_up[2, 3, 5] = 4.0  # an outlier sets its expert's divisor
            t[m + "experts.gate_up_proj"] = gate_up
            t[m + "experts.down_proj"] = r(e, h, i)
        else:
            i = text["intermediate_size"]
            t[m + "gate_proj.weight"] = r(i, h)
            t[m + "up_proj.weight"] = r(i, h)
            t[m + "down_proj.weight"] = r(h, i)

    t[tp + "embed_tokens.weight"] = r(v, h)
    t["lm_head.weight"] = r(v, h)
    hyper(tp + "hyper_connection_mixer.", False)
    for index, kind in enumerate(text["layer_types"]):
        block(f"{tp}layers.{index}.", kind)
    heads = (text["ngram_size"] - 1) * text["heads_per_ngram"]
    width, embed = text["ple_embed_dim"] // heads, text["ple_embed_dim"]
    for layer in text["ple_layer_ids"]:
        p = f"{tp}layers.{layer - 1}.ple."
        t[p + "key_proj.weight"] = r(s, embed)
        t[p + "value_proj.weight"] = r(h, embed)
        t[p + "norm_query.weight"] = r(s)
        t[p + "norm_key.weight"] = r(s)
        t[p + "norm_conv.weight"] = r(s)
        t[p + "conv1d.weight"] = r(s, 1, text["ple_conv_kernel_size"])
        e = p + "ple_embedding."
        t[e + "layer_multipliers"] = torch.tensor([1, 1 << 40, 7], dtype=torch.int64)
        sizes = [PLE_ROWS * text["split_ngram_parts"] // heads] * heads
        t[e + "ngram_heads_vocab_sizes"] = torch.tensor(sizes, dtype=torch.int64)
        t[e + "ngram_heads_offsets"] = torch.tensor(np.cumsum([0] + sizes[:-1]), dtype=torch.int64)
        for shard in range(text["split_ngram_parts"]):
            table = r(PLE_ROWS, width, scale=0.5)
            table[3, :16] = 0  # an all-zero group keeps a zero multiplier
            table[4, :16] = torch.linspace(-1.0, 0.25, 16)  # negative-dominated group
            t[f"{e}ngram_embedding.shard_{shard}.weight"] = table
    t["mtp.fc_embedding.weight"] = r(h, h)
    t["mtp.fc_hidden.weight"] = r(h, h)
    t["mtp.pre_fc_norm_embedding.weight"] = r(h)
    t["mtp.pre_fc_norm_hidden.weight"] = r(s)
    hyper("mtp.hyper_connection_mixer.", False)
    block("mtp.layers.0.", "full_attention")
    if nested:
        vh, vi, ps, pt = (
            VISION["hidden_size"],
            VISION["intermediate_size"],
            VISION["patch_size"],
            VISION["temporal_patch_size"],
        )
        merger = VISION["spatial_merge_size"] ** 2 * vh
        t["model.visual.patch_embed.proj.weight"] = r(vh, 3, pt, ps, ps)
        t["model.visual.patch_embed.proj.bias"] = r(vh)
        t["model.visual.pos_embed.weight"] = r(VISION["num_position_embeddings"], vh)
        for i in range(VISION["depth"]):
            b = f"model.visual.blocks.{i}."
            for norm in ("norm1", "norm2"):
                t[b + norm + ".weight"] = r(vh)
                t[b + norm + ".bias"] = r(vh)
            t[b + "attn.qkv.weight"] = r(3 * vh, vh)
            t[b + "attn.qkv.bias"] = r(3 * vh)
            t[b + "attn.proj.weight"] = r(vh, vh)
            t[b + "attn.proj.bias"] = r(vh)
            t[b + "mlp.linear_fc1.weight"] = r(vi, vh)
            t[b + "mlp.linear_fc1.bias"] = r(vi)
            t[b + "mlp.linear_fc2.weight"] = r(vh, vi)
            t[b + "mlp.linear_fc2.bias"] = r(vh)
        t["model.visual.merger.norm.weight"] = r(vh)
        t["model.visual.merger.norm.bias"] = r(vh)
        t["model.visual.merger.linear_fc1.weight"] = r(merger, merger)
        t["model.visual.merger.linear_fc1.bias"] = r(merger)
        t["model.visual.merger.linear_fc2.weight"] = r(h, merger)
        t["model.visual.merger.linear_fc2.bias"] = r(h)
    return t


def _resources(path: Path, *, template: str = "synthetic {{ messages }}") -> None:
    files = {
        "tokenizer.json": {"model": {"vocab": {str(i): i for i in range(6)}}},
        "tokenizer_config.json": {},
        "generation_config.json": {"eos_token_id": 5},
    }
    for role in ("preprocessor_config.json", "video_preprocessor_config.json"):
        files[role] = {"patch_size": 2, "temporal_patch_size": 1, "merge_size": 2}
    for role, value in files.items():
        (path / role).write_text(json.dumps(value))
    (path / "chat_template.jinja").write_text(template)


def _checkpoint(path: Path, config: dict, tensors: dict[str, torch.Tensor]) -> Path:
    """Two safetensors shards with an index, beside config.json and frontend resources."""
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    names = list(tensors)
    shards = {
        "model-00001-of-00002.safetensors": names[::2],
        "model-00002-of-00002.safetensors": names[1::2],
    }
    weight_map = {}
    for file, members in shards.items():
        save_file({name: tensors[name].contiguous() for name in members}, path / file)
        weight_map.update({name: file for name in members})
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    _resources(path)
    return path


def _pattern(name: str) -> str:
    return re.sub(r"shard_\d+", "shard_N", re.sub(r"\.\d+\.", ".N.", name))


def test_synthetic_checkpoint_uses_the_official_tensor_names():
    official = json.loads(gzip.decompress((FIXTURES / "official_tensors.json.gz").read_bytes()))
    expected = {_pattern(name) for name in official["tensors"]}
    synthetic = _official_tensors(_text(moe=True), nested=True, seed=0)
    assert {_pattern(name) for name in synthetic} == expected
    for name, tensor in synthetic.items():
        dtype = official["tensors"].get(name, [None])[0]
        if dtype is not None:
            assert dtype == ("I64" if tensor.dtype == torch.int64 else "BF16"), name


# --- the logical values the contract assigns to official tensors ---------------------------


def _expected(name: str, t: dict[str, torch.Tensor], tp: str) -> tuple[torch.Tensor, set[str]]:
    """A v3 logical parameter's value from official tensors, and the tensors it reads."""
    used: set[str] = set()

    def get(source):
        used.add(source)
        return t[source]

    def cat(*sources):
        return torch.cat([get(source) for source in sources])

    def conv(source):
        return get(source)[:, 0, :].T.contiguous()

    hyper = {
        "norm": "hc_norm.weight",
        "input_mix/down": "input_mix_weight_down.weight",
        "input_mix/up": "input_mix_weight_up.weight",
        "block_inject": "block_inject_weight.weight",
    }
    if name == "text/token_embedding":
        return get(tp + "embed_tokens.weight"), used
    if name == "text/output_head":
        return get("lm_head.weight"), used
    match = re.fullmatch(r"(text|mtp)/hyper_connection/(.+)", name)
    if match:
        prefix = tp if match[1] == "text" else "mtp."
        return get(prefix + "hyper_connection_mixer." + hyper[match[2]]), used
    mtp = {
        "mtp/embedding_projection": "mtp.fc_embedding.weight",
        "mtp/hidden_projection": "mtp.fc_hidden.weight",
        "mtp/embedding_norm": "mtp.pre_fc_norm_embedding.weight",
        "mtp/hidden_norm": "mtp.pre_fc_norm_hidden.weight",
    }
    if name in mtp:
        return get(mtp[name]), used
    match = re.fullmatch(r"(?:text/layers/(\d+)|mtp/layers/0)/(.+)", name)
    if match:
        sb = f"{tp}layers.{match[1]}." if match[1] is not None else "mtp.layers.0."
        role = match[2]
        part = re.fullmatch(r"(attention|mlp)/hyper_connection/(.+)", role)
        if part:
            owner = "attn" if part[1] == "attention" else "mlp"
            return get(f"{sb}{owner}_hyper_connection.{hyper[part[2]]}"), used
        a, g, m, p = sb + "self_attn.", sb + "linear_attn.", sb + "mlp.", sb + "ple."
        direct = {
            "attention/query_norm": a + "q_norm.weight",
            "attention/key_norm": a + "k_norm.weight",
            "attention/output": a + "o_proj.weight",
            "attention/indexer/query_key": a + "indexer.index_qk_proj.weight",
            "attention/indexer/query_norm": a + "indexer.q_layernorm.weight",
            "attention/indexer/key_norm": a + "indexer.k_layernorm.weight",
            "gdn/a_log": g + "A_log",
            "gdn/dt_bias": g + "dt_bias",
            "gdn/norm": g + "norm.weight",
            "gdn/output": g + "out_proj.weight",
            "mlp/router": m + "gate.weight",
            "mlp/shared_expert_gate": m + "shared_expert_gate.weight",
            "mlp/shared_expert/gate": m + "shared_expert.gate_proj.weight",
            "mlp/shared_expert/up": m + "shared_expert.up_proj.weight",
            "mlp/shared_expert/down": m + "shared_expert.down_proj.weight",
            "mlp/experts/gate_up": m + "experts.gate_up_proj",
            "mlp/experts/down": m + "experts.down_proj",
            "mlp/gate": m + "gate_proj.weight",
            "mlp/up": m + "up_proj.weight",
            "mlp/down": m + "down_proj.weight",
            "ple/key_projection": p + "key_proj.weight",
            "ple/value_projection": p + "value_proj.weight",
            "ple/query_norm": p + "norm_query.weight",
            "ple/key_norm": p + "norm_key.weight",
            "ple/conv_norm": p + "norm_conv.weight",
            "ple/embedding/layer_multipliers": p + "ple_embedding.layer_multipliers",
            "ple/embedding/ngram_head_offsets": p + "ple_embedding.ngram_heads_offsets",
            "ple/embedding/ngram_head_vocab_sizes": p + "ple_embedding.ngram_heads_vocab_sizes",
        }
        if role in direct:
            return get(direct[role]), used
        if role == "attention/query_gate_key_value":
            return cat(a + "q_proj.weight", a + "k_proj.weight", a + "v_proj.weight"), used
        if role == "gdn/query_key_value_z":
            return cat(g + "in_proj_qkv.weight", g + "in_proj_z.weight"), used
        if role == "gdn/a_b_projection":
            return cat(g + "in_proj_a.weight", g + "in_proj_b.weight"), used
        if role == "gdn/convolution":
            return conv(g + "conv1d.weight"), used
        if role == "ple/convolution":
            return conv(p + "conv1d.weight"), used
        shard = re.fullmatch(r"ple/embedding/shards/(\d+)", role)
        if shard:
            return get(f"{p}ple_embedding.ngram_embedding.shard_{shard[1]}.weight"), used
    vision = {
        "vision/patch_embedding_bias": "model.visual.patch_embed.proj.bias",
        "vision/position_embedding": "model.visual.pos_embed.weight",
        "vision/merger/norm_weight": "model.visual.merger.norm.weight",
        "vision/merger/norm_bias": "model.visual.merger.norm.bias",
        "vision/merger/fc1": "model.visual.merger.linear_fc1.weight",
        "vision/merger/fc1_bias": "model.visual.merger.linear_fc1.bias",
        "vision/merger/fc2": "model.visual.merger.linear_fc2.weight",
        "vision/merger/fc2_bias": "model.visual.merger.linear_fc2.bias",
    }
    if name in vision:
        return get(vision[name]), used
    if name == "vision/patch_embedding":
        value = get("model.visual.patch_embed.proj.weight")
        return value.reshape(value.shape[0], -1), used
    match = re.fullmatch(r"vision/layers/(\d+)/(.+)", name)
    if match:
        b = f"model.visual.blocks.{match[1]}."
        role = match[2]
        qkv = re.fullmatch(r"attention/(query|key|value)(_bias)?", role)
        if qkv:
            source = get(b + ("attn.qkv.bias" if qkv[2] else "attn.qkv.weight"))
            return source.chunk(3)[("query", "key", "value").index(qkv[1])], used
        blocks = {
            "attention/output": "attn.proj.weight",
            "attention/output_bias": "attn.proj.bias",
            "mlp/fc1": "mlp.linear_fc1.weight",
            "mlp/fc1_bias": "mlp.linear_fc1.bias",
            "mlp/fc2": "mlp.linear_fc2.weight",
            "mlp/fc2_bias": "mlp.linear_fc2.bias",
            "norm1_weight": "norm1.weight",
            "norm1_bias": "norm1.bias",
            "norm2_weight": "norm2.weight",
            "norm2_bias": "norm2.bias",
        }
        return get(b + blocks[role]), used
    raise AssertionError(f"no contract value for {name}")


# --- independent decoders of the stored words and the encoders' numeric rules --------------


def _align(value: int) -> int:
    return (value + 255) // 256 * 256


def _e4m3(word: int) -> float:
    sign = -1.0 if word & 0x80 else 1.0
    exponent, fraction = (word >> 3) & 0xF, word & 0x7
    if exponent == 0:
        return sign * fraction * 2.0**-9
    return sign * (1 + fraction / 8) * 2.0 ** (exponent - 7)


E4M3 = np.array([_e4m3(w) for w in range(256)])
E4M3_POSITIVE = np.array([_e4m3(w) for w in range(0x7F)])
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _nearest_excess(values: np.ndarray, grid: np.ndarray, chosen: np.ndarray) -> float:
    """How much farther the chosen magnitudes are from the values than the nearest grid points."""
    nearest = np.abs(values[..., None] - grid).min(axis=-1)
    return float((np.abs(chosen - values) - nearest).max(initial=0.0))


def _check_fp8_row_fp32(payload: bytes, source: torch.Tensor) -> None:
    n, k = source.shape
    raw = np.frombuffer(payload, dtype=np.uint8)
    assert len(payload) == _align(n * k) + 4 * n
    codes = raw[: n * k].reshape(n, k)
    assert not raw[n * k : _align(n * k)].any()
    scales = np.frombuffer(payload, dtype="<f4", offset=_align(n * k), count=n).astype(np.float64)
    x = source.float().double().numpy()
    amax = np.abs(x).max(axis=1)
    assert np.array_equal(scales, (amax / 448.0).astype(np.float32).astype(np.float64))
    y = np.clip(x / np.where(scales > 0, scales, 1.0)[:, None], -448.0, 448.0)
    decoded = E4M3[codes]
    assert np.all(np.isfinite(decoded))
    assert _nearest_excess(np.abs(y), E4M3_POSITIVE, np.abs(decoded)) <= 1e-6 * 448
    assert np.all((decoded == 0) | (np.sign(decoded) == np.sign(y)))


def _scale_offset(n: int, g: int, k: int) -> int:
    """storage-layouts.md Section 4: the swizzled scale word of row n, group g."""
    tile, inner = divmod(n, 128)
    return (tile * (k // 64) + g // 4) * 512 + (inner % 32) * 16 + (inner // 32) * 4 + g % 4


def _check_nvfp4_bank(payload: bytes, source: torch.Tensor) -> None:
    e, n, k = source.shape
    codes_bytes = e * n * k // 2
    scale_begin = _align(codes_bytes)
    divisor_begin = scale_begin + e * n * k // 16
    assert len(payload) == divisor_begin + 4 * e
    raw = np.frombuffer(payload, dtype=np.uint8)
    assert not raw[codes_bytes:scale_begin].any()
    divisors = np.frombuffer(payload, dtype="<f4", offset=divisor_begin, count=e)
    rows, groups = np.meshgrid(np.arange(n), np.arange(k // 16), indexing="ij")
    offsets = np.vectorize(_scale_offset)(rows, groups, k)
    for expert in range(e):
        w = source[expert].float().double().numpy()
        amax = np.abs(w).max()
        assert divisors[expert] == np.float32(2688.0 / amax if amax else 1.0)
        packed = raw[expert * n * k // 2 : (expert + 1) * n * k // 2].reshape(n, k // 2)
        words = np.stack((packed & 0xF, packed >> 4), axis=2).reshape(n, k // 16, 16)
        scale_words = raw[scale_begin + expert * n * k // 16 + offsets]
        assert np.all(scale_words < 0x7F)
        s = E4M3[scale_words][..., None]
        y = (w * float(divisors[expert])).reshape(n, k // 16, 16)
        ideal = np.abs(y).max(axis=2, keepdims=True) / 6.0
        step = 2.0 ** (np.floor(np.log2(np.maximum(ideal, 2.0**-6))) - 3)
        assert np.all(np.abs(s - ideal) <= step)
        ratio = np.where(s > 0, y / np.where(s > 0, s, 1.0), 0.0)
        magnitude = E2M1[words & 7]
        # The encoder rounds y and y/s to binary32; near a code boundary that may pick the
        # other neighbour of an FP64 value by about one binary32 step.
        assert _nearest_excess(np.minimum(np.abs(ratio), 6.0), E2M1, magnitude) <= 1e-5
        signs = ((words & 8) != 0) == (ratio < 0)
        assert np.all(signs | ((magnitude == 0) & (np.abs(ratio) <= 0.25)))


def _check_u4z8(payload: bytes, source: torch.Tensor) -> None:
    n, k = source.shape
    raw = np.frombuffer(payload, dtype=np.uint8)
    begin = _align(n * k // 2)
    assert len(payload) == begin + n * (k // 16) * 2
    packed = raw[: n * k // 2].reshape(n, k // 2)
    codes = np.stack((packed & 0xF, packed >> 4), axis=2).reshape(n, k // 16, 16).astype(int)
    scales = np.frombuffer(payload, dtype="<f2", offset=begin).astype(np.float64)
    scales = scales.reshape(n, k // 16)
    x = source.float().double().numpy().reshape(n, k // 16, 16)
    ideal = np.maximum(x.max(axis=2).clip(min=0) / 7.0, (-x).max(axis=2).clip(min=0) / 8.0)
    ulp = np.spacing(ideal.astype(np.float16)).astype(np.float64)
    assert np.all(np.abs(scales - ideal) <= ulp)
    q = codes - 8
    zero = scales == 0
    assert np.all(q[zero] == 0)
    ratio = x / np.where(zero, 1.0, scales)[..., None]
    grid = np.arange(-8, 8, dtype=np.float64)
    excess = np.abs(q - ratio) - np.abs(ratio[..., None] - grid).min(axis=-1)
    assert float(excess[~zero].max(initial=0.0)) <= 1e-6


def _check_object(artifact: Artifact, object_id: str, value: torch.Tensor) -> str:
    obj = artifact.object(object_id)
    payload = artifact.read_object(object_id)
    assert tuple(obj.shape) == tuple(value.shape), object_id
    if obj.format == "bf16":
        assert payload == value.to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()
    elif obj.format == "int64":
        assert payload == value.to(torch.int64).numpy().astype("<i8").tobytes()
    elif obj.format == FP8_FP32:
        _check_fp8_row_fp32(payload, value)
    elif obj.format == "nvfp4":
        assert obj.layout == "expert_block_scale_k16_m128x4_v1"
        _check_nvfp4_bank(payload, value)
    elif obj.format == "u4z8_g16_fp16":
        _check_u4z8(payload, value)
    else:
        raise AssertionError(f"{object_id}: unexpected format {obj.format}")
    return obj.format


def _check_artifact(path: Path, tensors: dict, tp: str) -> tuple[Artifact, dict, set]:
    """Every Binding against its contract value; returns the parameter formats and the official
    tensors the Bindings cover."""
    artifact = Artifact(path)
    formats: dict[str, str] = {}
    used: set[str] = set()
    parents: dict[str, list] = {}
    for name, binding in artifact.directory.bindings.items():
        value, sources = _expected(name, tensors, tp)
        used |= sources
        parts = binding_parts(binding, artifact.by_id, name)
        if len(parts) == 1 and parts[0][1] == 0 and "object" in binding:
            formats[name] = _check_object(artifact, parts[0][0], value)
        else:
            assert len(parts) == 1, name
            parents.setdefault(parts[0][0], []).append((parts[0][1], parts[0][2], name, value))
    for object_id, members in parents.items():
        members.sort()
        cursor = 0
        for begin, end, name, value in members:
            assert (begin, end) == (cursor, cursor + value.numel()), name
            cursor = end
        obj = artifact.object(object_id)
        parent = torch.cat([value.reshape(-1) for *_, value in members]).reshape(obj.shape)
        format = _check_object(artifact, object_id, parent)
        formats.update({name: format for _, _, name, _ in members})
    return artifact, formats, used


# --- end-to-end conversions -----------------------------------------------------------------


def test_moe_checkpoint_converts_through_the_cli(tmp_path):
    config = _source_config(moe=True, nested=True)
    tensors = _official_tensors(config["text_config"], nested=True, seed=1)
    source = _checkpoint(tmp_path / "source", config, tensors)
    output = tmp_path / "tiny.ninfer"
    command = [
        sys.executable, "-B", "-m", "tools.convert",
        "--model", str(source),
        "--recipe", "qwen3_8_flash_next",
        "--components", "text,vision,mtp",
        "--resource", f"chat_template.jinja={MAINTAINED}",
        "--name", "tiny-qwen4",
        "--out", str(output),
        "--device", "cpu",
        "--rows-per-chunk", "5",
        "--max-file-bytes", "400000",
    ]  # fmt: skip
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr
    artifact, formats, used = _check_artifact(output, tensors, "model.language_model.")
    with artifact:
        directory = artifact.directory
        assert len(directory.files) > 1
        assert used == set(tensors)  # every official tensor is stored
        text = directory.components["text"]["config"]
        assert text == text_config(config)
        assert text["ple_layer_ids"] == [1] and text["num_experts"] == 4
        assert "intermediate_size" not in text
        assert directory.components["vision"]["config"] == vision_config(config, text)
        assert directory.components["mtp"] == {
            "config": {"architectures": ["Qwen4ExpMTP"]},
            "target": "text",
        }
        resources = directory.components["text"]["resources"]
        assert artifact.read_object(resources["chat_template.jinja"]) == MAINTAINED.read_bytes()
        assert set(directory.components["vision"]["resources"]) == {
            "preprocessor_config.json",
            "video_preprocessor_config.json",
        }
        assert directory.metadata == {"name": "tiny-qwen4"}
        assert directory.provenance["recipe"] == "qwen3_8_flash_next"
        assert directory.provenance["chat_template"] == {
            "selection": "explicit",
            "sha256": hashlib.sha256(MAINTAINED.read_bytes()).hexdigest(),
        }

        # The Flash-Next allocation: NVFP4 banks (Text and MTP), FP8 row-FP32 Text mixers,
        # U4Z8 PLE shards, INT64 PLE tables, BF16 everything else (MTP attention included).
        def expected_format(name):
            if re.search(r"/mlp/experts/(gate_up|down)$", name):
                return "nvfp4"
            if re.fullmatch(
                r"text/layers/\d+/(attention/(query_gate_key_value|output)|gdn/(query_key_value_z|output))",
                name,
            ):
                return FP8_FP32
            if "/ple/embedding/shards/" in name:
                return "u4z8_g16_fp16"
            if "/ple/embedding/" in name:
                return "int64"
            return "bf16"

        assert formats == {name: expected_format(name) for name in formats}
        assert Counter(f for f in formats.values() if f != "bf16") == {
            "nvfp4": 6,
            FP8_FP32: 4,
            "u4z8_g16_fp16": 2,
            "int64": 3,
        }
        # Fused parents are whole objects; Vision Q/K/V are Parts of one packed parent.
        for name in (
            "text/layers/1/attention/query_gate_key_value",
            "text/layers/0/gdn/query_key_value_z",
            "text/layers/0/gdn/a_b_projection",
            "mtp/layers/0/mlp/experts/gate_up",
        ):
            assert set(directory.bindings[name]) == {"object"}
        vision = [
            binding_parts(directory.bindings[f"vision/layers/0/attention/{role}"], artifact.by_id)
            for role in ("query", "key", "value")
        ]
        assert len({parts[0][0] for parts in vision}) == 1

        # One Use per mathematical input (the v2 upgrade's independent map); policy follows
        # the stored format and no activation divisors are stored.
        uses = {(u["parameter"], u["input"]): u for u in directory.uses}
        assert set(uses) == {(p, i) for p in directory.bindings for i in upgrade_inputs(p)}
        policy = {"nvfp4": "AllowA4", FP8_FP32: "AllowA8"}
        for (parameter, _), use in uses.items():
            assert use["activation_policy"] == policy.get(formats[parameter], "A16Only")
            assert "auxiliaries" not in use

        # The native expert-bank encoder is the one that baked the qualified MTP banks.
        bank = tensors["mtp.layers.0.mlp.experts.gate_up_proj"]
        codes, scales, divisors = [], [], []
        for expert in bank:
            divisor = weight_divisor(float(expert.float().abs().max()))
            c, s = encode_rows(expert, divisor)
            codes.append(c)
            scales.append(s)
            divisors.append(struct.unpack("<f", divisor)[0])
        baked = encode_nvfp4_expert_bank(
            torch.stack(codes), torch.stack(scales), torch.tensor(divisors), bank.shape
        )
        stored = directory.bindings["mtp/layers/0/mlp/experts/gate_up"]["object"]
        assert artifact.read_object(stored) == bytes(baked)
    report = json.loads(Path(str(output) + ".conversion.json").read_text())
    assert report["formats"]["nvfp4"] == 6


def test_dense_checkpoint_converts_with_qwen3_5_dense_mlp_names(tmp_path, monkeypatch):
    # Small chunks so the PLE quantizer streams each table in several row blocks.
    monkeypatch.setattr(methods, "_MIN_CHUNK_VALUES", 7 * 16)
    config = _source_config(moe=False, nested=False)
    tensors = _official_tensors(config, nested=False, seed=2)
    source = _checkpoint(tmp_path / "source", config, tensors)
    output = tmp_path / "dense.ninfer"
    with SafetensorsSource(source) as base:
        model = build.build_model(
            base,
            components=("text", "mtp"),
            resource_overrides={"chat_template.jinja": MAINTAINED},
        )
        recipe = Recipe(model)
        qwen4_allocation(
            model,
            recipe,
            mixer=FP8_FP32,
            dense_mlp=FP8_FP32,
            mtp_dense_mlp=FP8_FP32,
            experts="nvfp4",  # absent from a dense model: skipped
            ple="u4z8_g16_fp16",
        )
        with pytest.raises(ValueError, match="MoE"):
            RECIPES["qwen3_8_flash_next"](model, Recipe(model), {})
        convert(model, recipe, output, device="cpu", rows_per_chunk=3)
    artifact, formats, used = _check_artifact(output, tensors, "model.")
    with artifact:
        directory = artifact.directory
        assert used == set(tensors)
        text = directory.components["text"]["config"]
        assert text["intermediate_size"] == 96
        assert not {"num_experts", "num_experts_per_tok", "moe_intermediate_size"} & set(text)
        assert set(directory.components) == {"text", "mtp"}
        for block in ("text/layers/0/", "text/layers/1/", "mtp/layers/0/"):
            assert not any(name.startswith(block + "mlp/experts") for name in directory.bindings)
            assert formats[block + "mlp/down"] == FP8_FP32
            # Gate and up share one FP8 parent as consecutive Parts, as for qwen3_5 Dense.
            (gate,) = binding_parts(directory.bindings[block + "mlp/gate"], artifact.by_id)
            (up,) = binding_parts(directory.bindings[block + "mlp/up"], artifact.by_id)
            assert gate[0] == up[0]
            assert (gate[1:], up[1:]) == ((0, 96 * 128), (96 * 128, 2 * 96 * 128))
            assert artifact.object(gate[0]).format == FP8_FP32
        uses = {(u["parameter"], u["input"]): u["activation_policy"] for u in directory.uses}
        for block in ("text/layers/1/", "mtp/layers/0/"):
            assert uses[(block + "mlp/gate", block + "ffn_input")] == "AllowA8"
            assert uses[(block + "mlp/up", block + "ffn_input")] == "AllowA8"
            assert uses[(block + "mlp/down", block + "mlp/product")] == "AllowA8"
        assert uses[("text/output_head", "mtp/final_hidden")] == "A16Only"


# --- Flash-Next geometry: the native route reproduces the qualified artifact's contract -----


class _HeaderOnlySource:
    """The official Flash-Next tensor directory (names, dtypes, shapes) without payloads."""

    def __init__(self, root: Path, config: dict, tensors: dict):
        self.root = self.path = root
        self.config = config
        self.tensors = tensors
        self.weight_map = dict.fromkeys(tensors)

    def has(self, name):
        return name in self.tensors

    def describe(self, name):
        if name not in self.tensors:
            raise ValueError(f"missing source tensor {name!r}")
        dtype, shape = self.tensors[name]
        return TensorInfo(self.root, tuple(shape), dtype, 0, 0)

    def read_flat(self, name, begin=0, end=None):
        dtype, shape = self.tensors[name]
        end = int(np.prod(shape)) if end is None else end
        return torch.zeros(end - begin, dtype=torch.int64 if dtype == "I64" else torch.bfloat16)


def test_flash_next_official_tensors_map_to_the_qualified_v3_contract(tmp_path):
    official = json.loads(gzip.decompress((FIXTURES / "official_tensors.json.gz").read_bytes()))
    assert official["repository"] == "Qwen/Qwen3.8-Flash-Next"
    root = tmp_path / "flash-next"
    root.mkdir()
    _resources(root)
    for role in ("preprocessor_config.json", "video_preprocessor_config.json"):
        (root / role).write_text(
            json.dumps({"patch_size": 16, "temporal_patch_size": 2, "merge_size": 2})
        )
    source = _HeaderOnlySource(root, load_source(), official["tensors"])
    model = build.build_model(
        source,
        components=("text", "vision", "mtp"),
        resource_overrides={"chat_template.jinja": MAINTAINED},
    )
    recipe = Recipe(model)
    RECIPES["qwen3_8_flash_next"](model, recipe, {})
    prepared = recipe.prepare(device="cpu")

    upgraded = plan_upgrade(
        json.loads(gzip.decompress((FIXTURES / "v2_directory.json.gz").read_bytes()))
    )
    assert model.components["text"]["config"] == upgraded.components["text"]["config"]
    assert model.components["vision"]["config"] == upgraded.components["vision"]["config"]
    assert model.components["mtp"]["config"] == upgraded.components["mtp"]["config"]

    native = {job.spec.id: job.spec for job in prepared.weights}
    reference = {spec.id: spec for spec in upgraded.specs if hasattr(spec, "format")}
    assert Counter(s.format for s in native.values()) == Counter(
        s.format for s in reference.values()
    )
    assert set(prepared.bindings) == set(upgraded.bindings)
    for name, binding in prepared.bindings.items():
        target = upgraded.bindings[name]
        if "object" in target:
            ours, qualified = native[binding["object"]], reference[target["object"]]
            assert (ours.shape, ours.format, ours.layout) == (
                qualified.shape,
                qualified.format,
                qualified.layout,
            ), name
        else:
            # Vision Q/K/V and their biases: Parts of one packed parent at the same ranges.
            ranges = [part["range"] for part in target["parts"]]
            assert [part["range"] for part in binding["parts"]] == ranges, name
            ours = native[binding["parts"][0]["object"]]
            qualified = reference[target["parts"][0]["object"]]
            assert (ours.shape, ours.format) == (qualified.shape, qualified.format), name

    def uses(items):
        return {(u["parameter"], u["input"], u["activation_policy"]) for u in items}

    assert uses(prepared.uses) == uses(upgraded.uses)
    assert not prepared.auxiliaries


# --- adapter policies -------------------------------------------------------------------------


def test_chat_template_defaults_to_the_maintained_qwen3_8_template(tmp_path, monkeypatch):
    config = _source_config(moe=True, nested=True)
    source = _checkpoint(
        tmp_path / "source", config, _official_tensors(config["text_config"], nested=True, seed=3)
    )
    with SafetensorsSource(source) as base:
        with pytest.raises(ValueError, match="--resource chat_template.jinja"):
            build.build_model(base)
        official = hashlib.sha256((source / "chat_template.jinja").read_bytes()).hexdigest()
        monkeypatch.setattr(build, "MAINTAINED_TEMPLATES", {official: ("qwen3.8", "qwen3_8.jinja")})
        model = build.build_model(base)
        stored = model.resources[model.components["text"]["resources"]["chat_template.jinja"]]
        assert stored == MAINTAINED.read_bytes()
        assert model.provenance["chat_template"] == {
            "selection": "maintained",
            "official": "qwen3.8",
            "official_sha256": official,
            "installed": "tools/chat_templates/qwen3_8.jinja",
            "sha256": hashlib.sha256(MAINTAINED.read_bytes()).hexdigest(),
        }
    assert (build.TEMPLATE_DIRECTORY / "qwen3_8.jinja").is_file()


def test_unmapped_or_misshapen_official_tensors_fail_the_build(tmp_path):
    config = _source_config(moe=True, nested=True)
    tensors = _official_tensors(config["text_config"], nested=True, seed=4)
    overrides = {"chat_template.jinja": MAINTAINED}
    # Unselected components may carry tensors the selection does not read.
    with SafetensorsSource(_checkpoint(tmp_path / "complete", config, tensors)) as base:
        model = build.build_model(base, resource_overrides=overrides)
        assert not any(name.startswith(("mtp/", "vision/")) for name in model.parameters)
    extra = dict(tensors)
    extra["model.language_model.layers.0.post_attention_layernorm.weight"] = torch.ones(128)
    with SafetensorsSource(_checkpoint(tmp_path / "extra", config, extra)) as base:
        with pytest.raises(ValueError, match="no qwen4_exp mapping.*post_attention_layernorm"):
            build.build_model(base, resource_overrides=overrides)
    wrong = dict(tensors)
    wrong["model.language_model.layers.1.self_attn.k_proj.weight"] = torch.ones(64, 128)
    with SafetensorsSource(_checkpoint(tmp_path / "wrong", config, wrong)) as base:
        with pytest.raises(ValueError, match="k_proj.weight: expected source shape"):
            build.build_model(base, resource_overrides=overrides)
    with SafetensorsSource(_checkpoint(tmp_path / "flat", config, tensors)) as base:
        with pytest.raises(ValueError, match="selects text"):
            build.build_model(base, components=("text", "dflash"), resource_overrides=overrides)


def test_config_normalizer_accepts_dense_blocks_and_rejects_mixed_ones():
    dense = text_config(_source_config(moe=False, nested=False))
    assert dense["intermediate_size"] == 96 and "num_experts" not in dense
    keys = list(dense)
    assert keys.index("intermediate_size") == keys.index("linear_conv_kernel_dim") + 1
    mixed = _source_config(moe=True, nested=True)
    mixed["text_config"]["mlp_only_layers"] = [0]
    with pytest.raises(ValueError, match="same feed-forward kind"):
        text_config(mixed)
    bad = _source_config(moe=True, nested=True)
    bad["text_config"]["ple_layer_ids"] = [3]
    with pytest.raises(ValueError, match="ple_layer_ids"):
        text_config(bad)
