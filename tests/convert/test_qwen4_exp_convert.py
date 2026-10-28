"""Official Qwen4 sources through the public converter, checked with independent oracles."""
from __future__ import annotations
from collections import Counter
import gzip
import hashlib
import json
from math import prod
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
from tools.artifact.reader import Artifact
from tools.artifact.schema import binding_parts, TensorObject
from tools.artifact.layouts import encoded_size
from tools.convert import qwen4_exp
from tools.convert.official_recipes import RECIPES
from tools.convert.recipe import Recipe
from tools.convert.pipeline import convert
from tools.convert.sources.safetensors import SafetensorsSource, TensorInfo
ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "qwen4_exp"
MAINTAINED = ROOT / "tools" / "chat_templates" / "qwen3_8_flash_next.jinja"
FP8_FP32 = "fp8_e4m3fn_row_fp32"


def _text(*, moe: bool) -> dict:
    text = {
        "model_type": "qwen4_exp_text",
        "hidden_act": "silu",
        "attention_bias": False,
        "output_gate_type": "sigmoid",
        "mamba_ssm_dtype": "float32",
        "mtp_num_hidden_layers": 1,
        "mtp": {"num_hidden_layers": 1, "layer_types": ["full_attention"],
                "mtp_use_hidden_state_from_layer": None, "hybrid": True,
                "rope_theta": 10000000},
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


def _pattern(name):
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
        assert end - begin <= 65536, "header fixture read would materialize a large weight"
        return torch.zeros(end - begin, dtype=torch.int64 if dtype == "I64" else torch.bfloat16)


def _expected(name, tensors, cfg, tp):
    """Logical mathematics from official axes, independently of the adapter's sources."""
    if name in ("text/token_embedding", "mtp/token_embedding"):
        return tensors[tp + "embed_tokens.weight"]
    if name in ("text/output_head", "mtp/output_head"):
        return tensors["lm_head.weight"]
    mtp = {"fc_embedding": "fc_embedding", "fc_hidden": "fc_hidden",
           "pre_fc_embedding_norm": "pre_fc_norm_embedding",
           "pre_fc_hidden_norm": "pre_fc_norm_hidden"}
    if name.startswith("mtp/") and name[4:] in mtp:
        return tensors["mtp." + mtp[name[4:]] + ".weight"]
    hc = {"norm": "hc_norm.weight", "down": "input_mix_weight_down.weight",
          "up": "input_mix_weight_up.weight", "inject": "block_inject_weight.weight"}
    match = re.fullmatch(r"(text|mtp)/final_hc/(.+)", name)
    if match:
        return tensors[(tp if match[1] == "text" else "mtp.") + "hyper_connection_mixer." + hc[match[2]]]
    match = re.fullmatch(r"(?:text/layers/(\d+)|mtp/layers/0)/(.+)", name)
    if match:
        prefix = f"{tp}layers.{match[1]}." if match[1] is not None else "mtp.layers.0."
        role = match[2]
        hyper = re.fullmatch(r"(attention|moe)_hc/(.+)", role)
        if hyper:
            return tensors[prefix + ("attn" if hyper[1] == "attention" else "mlp") + "_hyper_connection." + hc[hyper[2]]]
        a, g, m, p = (prefix + x for x in ("self_attn.", "linear_attn.", "mlp.", "ple."))
        direct = {
            "attention/key": a + "k_proj.weight", "attention/value": a + "v_proj.weight",
            "attention/output": a + "o_proj.weight", "attention/query_norm": a + "q_norm.weight",
            "attention/key_norm": a + "k_norm.weight",
            "attention/index_query_norm": a + "indexer.q_layernorm.weight",
            "attention/index_key_norm": a + "indexer.k_layernorm.weight",
            "gdn/z": g + "in_proj_z.weight", "gdn/a_projection": g + "in_proj_a.weight",
            "gdn/b_projection": g + "in_proj_b.weight", "gdn/output": g + "out_proj.weight",
            "gdn/norm": g + "norm.weight", "gdn/a_log": g + "A_log", "gdn/dt_bias": g + "dt_bias",
            "moe/router": m + "gate.weight", "moe/shared_score": m + "shared_expert_gate.weight",
            "moe/shared/gate": m + "shared_expert.gate_proj.weight",
            "moe/shared/up": m + "shared_expert.up_proj.weight",
            "moe/shared/down": m + "shared_expert.down_proj.weight",
            "moe/experts/down": m + "experts.down_proj",
            "ple/key": p + "key_proj.weight", "ple/value": p + "value_proj.weight",
            "ple/norm_query": p + "norm_query.weight", "ple/norm_key": p + "norm_key.weight",
            "ple/norm_conv": p + "norm_conv.weight",
        }
        if role in direct:
            return tensors[direct[role]]
        if role in ("attention/query", "attention/output_gate"):
            raw = tensors[a + "q_proj.weight"].reshape(cfg["num_attention_heads"], 2, cfg["head_dim"], -1)
            return raw[:, int(role.endswith("output_gate"))].reshape(-1, cfg["hidden_size"])
        if role in ("attention/index_query", "attention/index_key"):
            split = cfg["indexer_n_heads"] * cfg["indexer_head_dim"]
            raw = tensors[a + "indexer.index_qk_proj.weight"]
            return raw[:split] if role.endswith("query") else raw[split:]
        if role in ("gdn/query", "gdn/key", "gdn/value"):
            q = cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
            raw = tensors[g + "in_proj_qkv.weight"]
            return {"gdn/query": raw[:q], "gdn/key": raw[q:2*q], "gdn/value": raw[2*q:]}[role]
        if role in ("gdn/convolution", "ple/convolution"):
            return tensors[(g if role.startswith("gdn/") else p) + "conv1d.weight"][:, 0, :].T.contiguous()
        if role in ("moe/experts/gate", "moe/experts/up"):
            bank = tensors[m + "experts.gate_up_proj"]
            i = cfg["moe_intermediate_size"]
            return bank[:, :i] if role.endswith("gate") else bank[:, i:]
        if role == "ple/table":
            return torch.cat([tensors[f"{p}ple_embedding.ngram_embedding.shard_{j}.weight"]
                              for j in range(cfg["split_ngram_parts"])])
    direct = {
        "vision/patch_embedding": "patch_embed.proj.weight", "vision/patch_embedding_bias": "patch_embed.proj.bias",
        "vision/position_embedding": "pos_embed.weight", "vision/merger/norm_weight": "merger.norm.weight",
        "vision/merger/norm_bias": "merger.norm.bias", "vision/merger/fc1": "merger.linear_fc1.weight",
        "vision/merger/fc1_bias": "merger.linear_fc1.bias", "vision/merger/fc2": "merger.linear_fc2.weight",
        "vision/merger/fc2_bias": "merger.linear_fc2.bias",
    }
    if name in direct:
        value = tensors["model.visual." + direct[name]]
        return value.flatten(1) if name == "vision/patch_embedding" else value
    match = re.fullmatch(r"vision/layers/(\d+)/(.+)", name)
    if match:
        b, role = f"model.visual.blocks.{match[1]}.", match[2]
        qkv = re.fullmatch(r"attention/(query|key|value)(_bias)?", role)
        if qkv:
            raw = tensors[b + ("attn.qkv.bias" if qkv[2] else "attn.qkv.weight")]
            return raw.chunk(3)[("query", "key", "value").index(qkv[1])]
        fields = {"attention/output": "attn.proj.weight", "attention/output_bias": "attn.proj.bias",
                  "mlp/fc1": "mlp.linear_fc1.weight", "mlp/fc1_bias": "mlp.linear_fc1.bias",
                  "mlp/fc2": "mlp.linear_fc2.weight", "mlp/fc2_bias": "mlp.linear_fc2.bias"}
        if role.startswith(("norm1_", "norm2_")):
            return tensors[b + role.replace("_", ".")]
        return tensors[b + fields[role]]
    raise AssertionError(f"no independent value for {name}")


def _check_artifact(path, tensors, cfg, tp):
    with Artifact(path) as artifact:
        parents, covered = {}, {}
        formats = {}
        for name, binding in artifact.directory.bindings.items():
            expected = _expected(name, tensors, cfg, tp).float().flatten()
            cursor = 0
            for oid, begin, end in binding_parts(binding, artifact.by_id, name):
                obj = artifact.object(oid)
                if oid not in parents:
                    parents[oid] = torch.empty(prod(obj.shape))
                    covered[oid] = torch.zeros(prod(obj.shape), dtype=torch.bool)
                count = end - begin
                selected = expected[cursor:cursor+count]
                assert selected.numel() == count
                overlap = covered[oid][begin:end]
                assert torch.equal(parents[oid][begin:end][overlap], selected[overlap])
                parents[oid][begin:end] = selected
                covered[oid][begin:end] = True
                cursor += count
                formats[name] = obj.format
            assert cursor == expected.numel(), name
        for oid, values in parents.items():
            assert covered[oid].all(), oid
            obj, payload = artifact.object(oid), artifact.read_object(oid)
            values = values.reshape(obj.shape)
            if obj.format == "bf16":
                assert payload == values.bfloat16().contiguous().view(torch.uint8).numpy().tobytes()
            elif obj.format == "fp32":
                assert payload == values.contiguous().numpy().astype('<f4').tobytes()
            elif obj.format == FP8_FP32:
                _check_fp8_row_fp32(payload, values)
            elif obj.format == "nvfp4":
                assert obj.layout == "batched_block_scale_k16_m128x4_v1"
                _check_nvfp4_bank(payload, values)
            elif obj.format == "u4z8_g16_fp16":
                _check_u4z8(payload, values)
            else:
                raise AssertionError(obj.format)
        uses = {(u['parameter'], u['input']): u for u in artifact.directory.uses}
        for (name, _), use in uses.items():
            expected = {'nvfp4': 'AllowA4', FP8_FP32: 'AllowA8'}.get(formats[name], 'A16Only')
            assert use['activation_policy'] == expected
            if expected == 'AllowA4':
                ref = use['auxiliaries']['activation_input_divisor']
                (oid, begin, end), = binding_parts(ref, artifact.by_id)
                assert end - begin == 1
                data = artifact.read_object(oid)
                assert struct.unpack_from('<f', data, begin * 4)[0] == 1.0
        resources = artifact.directory.components['text']['resources']
        layer = cfg['ple_layer_ids'][0] - 1
        controls = b''.join(tensors[tp + f'layers.{layer}.ple.ple_embedding.' + field].view(torch.uint8).numpy().tobytes()
                            for field in ('layer_multipliers', 'ngram_heads_vocab_sizes', 'ngram_heads_offsets'))
        assert artifact.read_object(resources['ple_hash_controls']) == controls
        assert artifact.read_object(resources['chat_template.jinja']) == MAINTAINED.read_bytes()
        assert artifact.directory.bindings['mtp/output_head'] == artifact.directory.bindings['text/output_head']
        assert artifact.directory.bindings['mtp/token_embedding'] == artifact.directory.bindings['text/token_embedding']
        return formats, len(artifact.directory.files)


@pytest.mark.parametrize('nested', [True, False])
def test_official_moe_checkpoint_converts_through_cli(tmp_path, nested):
    config = _source_config(moe=True, nested=nested)
    cfg = config.get('text_config', config)
    tensors = _official_tensors(cfg, nested=nested, seed=1)
    source = _checkpoint(tmp_path / 'source', config, tensors)
    output = tmp_path / 'tiny.ninfer'
    components = 'text,vision,mtp' if nested else 'text,mtp'
    result = subprocess.run([sys.executable, '-m', 'tools.convert', '--model', str(source),
        '--recipe', 'qwen3_8_flash_next_mixed', '--components', components,
        '--out', str(output), '--device', 'cpu', '--rows-per-chunk', '5',
        '--max-file-bytes', '400000'], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    formats, files = _check_artifact(output, tensors, cfg, 'model.language_model.' if nested else 'model.')
    assert files > 1
    assert formats['text/layers/0/gdn/query'] == FP8_FP32
    assert formats['text/layers/1/attention/key'] == FP8_FP32
    assert formats['text/layers/1/attention/index_query'] == 'bf16'
    assert formats['mtp/layers/0/attention/query'] == 'bf16'
    report = json.loads(Path(str(output) + '.conversion.json').read_text())
    assert report['formats']['nvfp4'] == 6
    assert report['formats']['u4z8_g16_fp16'] == 2


def test_flash_next_official_header_maps_every_selected_source_with_bounded_reads(tmp_path):
    official = json.loads(gzip.decompress((FIXTURES / 'official_tensors.json.gz').read_bytes()))
    config = json.loads((Path(__file__).parent / 'qwen4_exp_source_config.json').read_text())
    _resources(tmp_path)
    for role in ('preprocessor_config.json', 'video_preprocessor_config.json'):
        (tmp_path / role).write_text(json.dumps({'patch_size': 16, 'temporal_patch_size': 2, 'merge_size': 2}))
    model = qwen4_exp.build_model(_HeaderOnlySource(tmp_path, config, official['tensors']),
                                 components=('text', 'vision', 'mtp'))
    recipe = Recipe(model)
    RECIPES['qwen3_8_flash_next_mixed'](model, recipe, {})
    prepared = recipe.prepare(device='cpu')
    assert set(prepared.bindings) == set(model.parameters)
    weights = {j.spec.id: TensorObject(j.spec.id, j.spec.shape, j.spec.format, j.spec.layout,
                  0, encoded_size(j.spec.layout, j.spec.format, j.spec.shape)) for j in prepared.weights}
    for name in ('text/layers/3/', 'mtp/layers/0/'):
        def parts(role):
            return binding_parts(prepared.bindings[name + role], weights)
        router, score = parts('moe/router'), parts('moe/shared_score')
        assert router[0][0] == score[0][0] and router[0][2] == score[0][1]
        assert len(parts('moe/experts/gate')) == 512
    table = prepared.bindings['text/layers/1/ple/table']
    shards = binding_parts(table, weights)
    assert len(shards) == 128
    assert all(weights[oid].shape == (2500012, 160) for oid, _, _ in shards)
    assert Counter(j.spec.format for j in prepared.weights)['nvfp4'] == 98


def test_unmapped_or_misshapen_official_tensors_are_rejected(tmp_path):
    config = _source_config(moe=True, nested=True)
    tensors = _official_tensors(config['text_config'], nested=True, seed=4)
    with SafetensorsSource(_checkpoint(tmp_path / 'complete', config, tensors)) as base:
        model = qwen4_exp.build_model(base)
        assert not any(n.startswith(('mtp/', 'vision/')) for n in model.parameters)
    extra = {**tensors, 'model.language_model.layers.0.post_attention_layernorm.weight': torch.ones(128)}
    with SafetensorsSource(_checkpoint(tmp_path / 'extra', config, extra)) as base:
        with pytest.raises(ValueError, match='source coverage differs.*post_attention_layernorm'):
            qwen4_exp.build_model(base)
    wrong = {**tensors, 'model.language_model.layers.1.self_attn.k_proj.weight': torch.ones(64, 128)}
    with SafetensorsSource(_checkpoint(tmp_path / 'wrong', config, wrong)) as base:
        with pytest.raises(ValueError, match='k_proj.weight: expected BF16'):
            qwen4_exp.build_model(base)


def test_config_rejects_unimplemented_dense_mixed_and_invalid_ple():
    with pytest.raises(ValueError, match='num_experts'):
        qwen4_exp.text_config(_source_config(moe=False, nested=False))
    mixed = _source_config(moe=True, nested=True)
    mixed['text_config']['mlp_only_layers'] = [0]
    with pytest.raises(ValueError, match='mixed'):
        qwen4_exp.text_config(mixed)
    bad = _source_config(moe=True, nested=True)
    bad['text_config']['ple_layer_ids'] = [3]
    with pytest.raises(ValueError, match='PLE'):
        qwen4_exp.text_config(bad)
