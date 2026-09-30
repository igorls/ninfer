"""Closed Flash-Next v2 object inventory and its v3 directory.

Every v2 object keeps its id, shape and payload bytes; its v2 numeric format and layout map to
the equivalent v3 registrations. The two BF16 MTP expert banks become NVFP4 expert banks.
Bindings are one-to-one with the stored parents under the ``qwen4_exp`` logical names
(``mtp/layer/`` becomes ``mtp/layers/0/``); Vision uses the ``qwen3_5`` Vision names and splits.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from tools.artifact.layouts import encoded_size
from tools.artifact.schema import ResourceSpec, TensorSpec

from .config import MTP_CONFIG, load_source, text_config, vision_config
from .source import (
    MTP_BANKS,
    SOURCE_CONFIG_REPOSITORY,
    SOURCE_CONFIG_REVISION,
    V2_FILENAME,
    V2_MODEL_ID,
    V2_REPOSITORY,
    V2_REVISION,
    V2_SHA256,
    V2_WEIGHTS_ID,
)

V2_OBJECTS = 1566
V2_ENCODINGS = {
    ("BF16", "contiguous-le-v1"): ("bf16", "contiguous_le_v1"),
    ("I64", "contiguous-le-v1"): ("int64", "contiguous_le_v1"),
    ("NVFP4", "expert-blockscale-k16-m128x4-v1"): (
        "nvfp4",
        "expert_block_scale_k16_m128x4_v1",
    ),
    ("FP8_E4M3FN_ROW_F32S", "row-scale-f32-v1"): (
        "fp8_e4m3fn_row_fp32",
        "row_scale_fp32_v1",
    ),
    ("U4Z8G16_F16S", "packed-u4-g16-v1"): ("u4z8_g16_fp16", "packed_u4_g16_v1"),
}
TEXT_RESOURCES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
)
VISION_RESOURCES = ("preprocessor_config.json", "video_preprocessor_config.json")
BAKED_BANK = ("nvfp4", "expert_block_scale_k16_m128x4_v1")
POLICIES = {"nvfp4": "AllowA4", "fp8_e4m3fn_row_fp32": "AllowA8"}
PLE_LAYER = 1  # 0-based decoder layer holding the PLE weights (source ple_layer_ids=[2]).
PLE_SHARD_ROWS = 2_500_012


@dataclass(frozen=True, slots=True)
class V2Object:
    name: str
    kind: str
    offset: int
    bytes: int
    shape: tuple[int, ...] = ()
    format: str = ""
    layout: str = ""


@dataclass(frozen=True, slots=True)
class Plan:
    specs: tuple[TensorSpec | ResourceSpec, ...]
    sources: dict[str, V2Object]
    components: dict
    bindings: dict
    uses: tuple[dict, ...]
    metadata: dict
    provenance: dict

    def description(self) -> dict:
        """The directory content that determines the artifact, excluding its file table."""

        return {
            "components": self.components,
            "objects": [
                (
                    ["tensor", s.id, list(s.shape), s.format, s.layout]
                    if isinstance(s, TensorSpec)
                    else ["resource", s.id, s.encoding, s.bytes]
                )
                for s in self.specs
            ],
            "bindings": self.bindings,
            "uses": list(self.uses),
            "metadata": self.metadata,
            "provenance": self.provenance,
        }


def _block(expected: dict, prefix: str, text: dict, kind: str, mtp: bool) -> None:
    h, e, i = text["hidden_size"], text["num_experts"], text["moe_intermediate_size"]
    hc = text["hc_count"] * h
    low = text["hc_lowrank"]
    for part in ("attention", "mlp"):
        base = f"{prefix}{part}/hyper_connection/"
        expected[base + "block_inject"] = ("BF16", (text["hc_count"], hc))
        expected[base + "norm"] = ("BF16", (hc,))
        expected[base + "input_mix/down"] = ("BF16", (low, hc))
        expected[base + "input_mix/up"] = ("BF16", (hc, low))
    shared = text["shared_expert_intermediate_size"]
    expected[prefix + "mlp/router"] = ("BF16", (e, h))
    expected[prefix + "mlp/shared_expert_gate"] = ("BF16", (1, h))
    expected[prefix + "mlp/shared_expert/gate"] = ("BF16", (shared, h))
    expected[prefix + "mlp/shared_expert/up"] = ("BF16", (shared, h))
    expected[prefix + "mlp/shared_expert/down"] = ("BF16", (h, shared))
    bank = "BF16" if mtp else "NVFP4"
    expected[prefix + "mlp/experts/gate_up"] = (bank, (e, 2 * i, h))
    expected[prefix + "mlp/experts/down"] = (bank, (e, h, i))
    projection = "BF16" if mtp else "FP8_E4M3FN_ROW_F32S"
    if kind == "full_attention":
        q = text["num_attention_heads"] * text["head_dim"]
        kv = text["num_key_value_heads"] * text["head_dim"]
        indexer = (text["indexer_n_heads"] + text["indexer_kv_heads"]) * text[
            "indexer_head_dim"
        ]
        expected[prefix + "attention/query_gate_key_value"] = (
            projection,
            (2 * q + 2 * kv, h),
        )
        expected[prefix + "attention/output"] = (projection, (h, q))
        expected[prefix + "attention/query_norm"] = ("BF16", (text["head_dim"],))
        expected[prefix + "attention/key_norm"] = ("BF16", (text["head_dim"],))
        expected[prefix + "attention/indexer/query_key"] = ("BF16", (indexer, h))
        expected[prefix + "attention/indexer/query_norm"] = (
            "BF16",
            (text["indexer_head_dim"],),
        )
        expected[prefix + "attention/indexer/key_norm"] = (
            "BF16",
            (text["indexer_head_dim"],),
        )
    else:
        kg = text["linear_num_key_heads"] * text["linear_key_head_dim"]
        vg = text["linear_num_value_heads"] * text["linear_value_head_dim"]
        heads = text["linear_num_value_heads"]
        expected[prefix + "gdn/query_key_value_z"] = (projection, (2 * kg + 2 * vg, h))
        expected[prefix + "gdn/output"] = (projection, (h, vg))
        expected[prefix + "gdn/a_b_projection"] = ("BF16", (2 * heads, h))
        expected[prefix + "gdn/a_log"] = ("BF16", (heads,))
        expected[prefix + "gdn/dt_bias"] = ("BF16", (heads,))
        expected[prefix + "gdn/norm"] = ("BF16", (text["linear_value_head_dim"],))
        expected[prefix + "gdn/convolution"] = (
            "BF16",
            (text["linear_conv_kernel_dim"], 2 * kg + vg),
        )


def expected_tensors(text: dict, vision: dict) -> dict[str, tuple[str, tuple[int, ...]]]:
    """The complete v2 tensor inventory: name -> (v2 format, shape)."""

    h, r = text["hidden_size"], text["vocab_size"]
    hc, low = text["hc_count"] * h, text["hc_lowrank"]
    expected: dict[str, tuple[str, tuple[int, ...]]] = {
        "text/token_embedding": ("BF16", (r, h)),
        "text/output_head": ("BF16", (r, h)),
    }
    for prefix in ("text/", "mtp/"):
        expected[prefix + "hyper_connection/norm"] = ("BF16", (hc,))
        expected[prefix + "hyper_connection/input_mix/down"] = ("BF16", (low, hc))
        expected[prefix + "hyper_connection/input_mix/up"] = ("BF16", (hc, low))
    for index, kind in enumerate(text["layer_types"]):
        _block(expected, f"text/layers/{index}/", text, kind, mtp=False)
    ple = f"text/layers/{PLE_LAYER}/ple/"
    width = text["ple_embed_dim"]
    expected[ple + "key_projection"] = ("BF16", (hc, width))
    expected[ple + "value_projection"] = ("BF16", (h, width))
    expected[ple + "query_norm"] = ("BF16", (hc,))
    expected[ple + "key_norm"] = ("BF16", (hc,))
    expected[ple + "conv_norm"] = ("BF16", (hc,))
    expected[ple + "convolution"] = ("BF16", (text["ple_conv_kernel_size"], hc))
    heads = (text["ngram_size"] - 1) * text["heads_per_ngram"]
    expected[ple + "embedding/layer_multipliers"] = ("I64", (text["ngram_size"],))
    expected[ple + "embedding/ngram_head_offsets"] = ("I64", (heads,))
    expected[ple + "embedding/ngram_head_vocab_sizes"] = ("I64", (heads,))
    for shard in range(text["split_ngram_parts"]):
        expected[f"{ple}embedding/shards/{shard}"] = (
            "U4Z8G16_F16S",
            (PLE_SHARD_ROWS, width // heads),
        )
    expected["mtp/embedding_projection"] = ("BF16", (h, h))
    expected["mtp/hidden_projection"] = ("BF16", (h, h))
    expected["mtp/embedding_norm"] = ("BF16", (h,))
    expected["mtp/hidden_norm"] = ("BF16", (hc,))
    _block(expected, "mtp/layer/", text, "full_attention", mtp=True)
    vh, vi = vision["hidden_size"], vision["intermediate_size"]
    patch = 3 * vision["temporal_patch_size"] * vision["patch_size"] ** 2
    merger = vision["spatial_merge_size"] ** 2 * vh
    expected["vision/patch_embedding"] = ("BF16", (vh, patch))
    expected["vision/patch_embedding_bias"] = ("BF16", (vh,))
    expected["vision/position_embedding"] = ("BF16", (vision["num_position_embeddings"], vh))
    for index in range(vision["depth"]):
        p = f"vision/layers/{index}/"
        for name, shape in (
            ("attention/qkv", (3 * vh, vh)),
            ("attention/qkv_bias", (3 * vh,)),
            ("attention/output", (vh, vh)),
            ("attention/output_bias", (vh,)),
            ("mlp/fc1", (vi, vh)),
            ("mlp/fc1_bias", (vi,)),
            ("mlp/fc2", (vh, vi)),
            ("mlp/fc2_bias", (vh,)),
            ("norm1/weight", (vh,)),
            ("norm1/bias", (vh,)),
            ("norm2/weight", (vh,)),
            ("norm2/bias", (vh,)),
        ):
            expected[p + name] = ("BF16", shape)
    for name, shape in (
        ("fc1", (merger, merger)),
        ("fc1_bias", (merger,)),
        ("fc2", (h, merger)),
        ("fc2_bias", (h,)),
        ("norm/weight", (vh,)),
        ("norm/bias", (vh,)),
    ):
        expected["vision/merger/" + name] = ("BF16", shape)
    return expected


def _inputs(name: str) -> list[str]:
    """Mathematical input positions of a projection parameter (empty for non-projections)."""

    if name == "text/output_head":
        return ["text/final_hidden", "mtp/final_hidden"]
    if name == "mtp/embedding_projection":
        return ["mtp/normalized_embedding"]
    if name == "mtp/hidden_projection":
        return ["mtp/normalized_hidden"]
    match = re.fullmatch(
        r"((?:text/layers/\d+/|mtp/layers/0/)(?:attention|mlp)/|text/|mtp/)"
        r"hyper_connection/(input_mix/down|input_mix/up|block_inject)",
        name,
    )
    if match:
        owner, role = match.groups()
        state = "low_rank" if role == "input_mix/up" else "normalized_state"
        return [f"{owner}hyper_connection/{state}"]
    match = re.fullmatch(r"(text/layers/\d+/)ple/(key|value)_projection", name)
    if match:
        return [match[1] + "ple/embedding"]
    match = re.fullmatch(r"(text/layers/\d+/|mtp/layers/0/)(.+)", name)
    if match:
        block, role = match.groups()
        if role in (
            "attention/query_gate_key_value",
            "attention/indexer/query_key",
            "gdn/query_key_value_z",
            "gdn/a_b_projection",
        ):
            return [block + "mixer_input"]
        if role in ("attention/output", "gdn/output"):
            return [block + role.rsplit("/", 1)[0] + "/gated_output"]
        if role in (
            "mlp/router",
            "mlp/shared_expert_gate",
            "mlp/shared_expert/gate",
            "mlp/shared_expert/up",
            "mlp/experts/gate_up",
        ):
            return [block + "ffn_input"]
        if role in ("mlp/shared_expert/down", "mlp/experts/down"):
            return [block + role.rsplit("/", 1)[0] + "/product"]
        return []
    if name == "vision/patch_embedding":
        return ["vision/patch_input"]
    if name == "vision/merger/fc1":
        return ["vision/merger/input"]
    if name == "vision/merger/fc2":
        return ["vision/merger/activation"]
    match = re.fullmatch(r"(vision/layers/\d+/)(.+)", name)
    if match:
        block, role = match.groups()
        position = {
            "attention/query": "attention_input",
            "attention/key": "attention_input",
            "attention/value": "attention_input",
            "attention/output": "attention_output",
            "mlp/fc1": "mlp_input",
            "mlp/fc2": "mlp_activation",
        }.get(role)
        return [] if position is None else [block + position]
    return []


def parse_v2_objects(directory: dict) -> list[V2Object]:
    identity = directory.get("identity")
    if identity != {"model_id": V2_MODEL_ID, "weights_id": V2_WEIGHTS_ID}:
        raise ValueError(f"not the Flash-Next v2 artifact: {identity!r}")
    objects = []
    for value in directory["objects"]:
        if value["kind"] == "resource":
            if value.get("encoding") != "raw-bytes-v1":
                raise ValueError(f"{value['name']}: unsupported v2 resource encoding")
            objects.append(
                V2Object(value["name"], "resource", value["offset"], value["bytes"])
            )
        else:
            objects.append(
                V2Object(
                    value["name"],
                    "tensor",
                    value["offset"],
                    value["bytes"],
                    tuple(value["shape"]),
                    value["format"],
                    value["layout"],
                )
            )
    if len(objects) != V2_OBJECTS:
        raise ValueError(f"expected {V2_OBJECTS} v2 objects, got {len(objects)}")
    return objects


def plan_upgrade(directory: dict) -> Plan:
    """Map the v2 directory to v3 object specs, components, bindings and uses."""

    objects = parse_v2_objects(directory)
    source = load_source()
    text = text_config(source)
    vision = vision_config(source, text)
    expected = expected_tensors(text, vision)
    resources = {"frontend/" + name for name in (*TEXT_RESOURCES, *VISION_RESOURCES)}
    names = [o.name for o in objects]
    if len(set(names)) != len(names):
        raise ValueError("duplicate v2 object names")
    wanted = expected.keys() | resources
    if set(names) != wanted:
        missing = sorted(wanted - set(names))
        extra = sorted(set(names) - wanted)
        raise ValueError(f"v2 inventory differs: missing {missing[:5]}, extra {extra[:5]}")

    specs: list[TensorSpec | ResourceSpec] = []
    sources: dict[str, V2Object] = {}
    formats: dict[str, str] = {}
    previous_end = 0
    for obj in objects:
        if obj.offset < previous_end:
            raise ValueError(f"{obj.name}: v2 objects overlap or are unordered")
        previous_end = obj.offset + obj.bytes
        sources[obj.name] = obj
        if obj.kind == "resource":
            if obj.name not in resources:
                raise ValueError(f"{obj.name}: unexpected v2 resource")
            specs.append(ResourceSpec(obj.name, obj.bytes))
            continue
        want_format, want_shape = expected[obj.name]
        if (obj.format, obj.shape) != (want_format, want_shape):
            raise ValueError(
                f"{obj.name}: expected {want_format} {want_shape}, "
                f"got {obj.format} {obj.shape}"
            )
        if obj.name in MTP_BANKS:
            if (obj.format, obj.layout) != ("BF16", "contiguous-le-v1"):
                raise ValueError(f"{obj.name}: expected the BF16 v2 MTP bank")
            format, layout = BAKED_BANK
        else:
            try:
                format, layout = V2_ENCODINGS[(obj.format, obj.layout)]
            except KeyError:
                raise ValueError(
                    f"{obj.name}: unknown v2 encoding {obj.format}/{obj.layout}"
                ) from None
            if encoded_size(layout, format, obj.shape) != obj.bytes:
                raise ValueError(f"{obj.name}: v2 bytes differ from the v3 encoding")
        specs.append(TensorSpec(obj.name, obj.shape, format, layout))
        formats[obj.name] = format

    bindings: dict[str, dict] = {}
    parameter_formats: dict[str, str] = {}

    def put(name: str, binding: dict, format: str) -> None:
        if name in bindings:
            raise ValueError(f"duplicate logical parameter {name}")
        bindings[name] = binding
        parameter_formats[name] = format

    for name, format in formats.items():
        vision_layer = re.fullmatch(r"(vision/layers/\d+/)(.+)", name)
        if vision_layer:
            block, role = vision_layer.groups()
            width = vision["hidden_size"]
            if role in ("attention/qkv", "attention/qkv_bias"):
                size = width * width if role == "attention/qkv" else width
                suffix = "" if role == "attention/qkv" else "_bias"
                for index, part in enumerate(("query", "key", "value")):
                    put(
                        f"{block}attention/{part}{suffix}",
                        {"parts": [{"object": name, "range": [index * size, (index + 1) * size]}]},
                        format,
                    )
                continue
            role = role.replace("norm1/", "norm1_").replace("norm2/", "norm2_")
            put(block + role, {"object": name}, format)
            continue
        if name.startswith("vision/merger/norm/"):
            put(name.replace("merger/norm/", "merger/norm_"), {"object": name}, format)
            continue
        put(name.replace("mtp/layer/", "mtp/layers/0/", 1), {"object": name}, format)

    uses = []
    for name, format in parameter_formats.items():
        for input_name in _inputs(name):
            uses.append(
                {
                    "parameter": name,
                    "input": input_name,
                    "activation_policy": POLICIES.get(format, "A16Only"),
                }
            )

    components = {
        "text": {
            "config": text,
            "resources": {role: "frontend/" + role for role in TEXT_RESOURCES},
        },
        "vision": {
            "config": vision,
            "target": "text",
            "resources": {role: "frontend/" + role for role in VISION_RESOURCES},
        },
        "mtp": {"config": dict(MTP_CONFIG), "target": "text"},
    }
    provenance = {
        "upgraded_from": {
            "version": 2,
            "model_id": V2_MODEL_ID,
            "weights_id": V2_WEIGHTS_ID,
            "repository": V2_REPOSITORY,
            "revision": V2_REVISION,
            "file": V2_FILENAME,
            "sha256": V2_SHA256,
        },
        "config": {
            "repository": SOURCE_CONFIG_REPOSITORY,
            "revision": SOURCE_CONFIG_REVISION,
        },
        "sources": {
            "mixed": "primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8@"
            "a4e813ed3cfbbcc61e2929699eccb864a4dfa843 (without the BF16 PLE table)",
            "ple": "primitive-ai/Qwen3.8-Flash-Next-PLE-quant@"
            "da8b39586016d8325ac619be28ad77d6296625ec (ples_int4)",
            "recipe": "qwen3_8_flash_next_mixed-v1",
        },
        "mtp_expert_banks": {
            "objects": list(MTP_BANKS),
            "encoder": "NVFP4_MAXABS_DIVISOR_RNE_V1 with one FP32 divisor per expert",
            "source": "the v2 BF16 banks; bytes equal the v2 loader's NVFP4 device banks",
        },
    }
    return Plan(
        specs=tuple(specs),
        sources=sources,
        components=components,
        bindings=bindings,
        uses=tuple(uses),
        metadata={"name": V2_MODEL_ID},
        provenance=provenance,
    )
