"""Official representation recipes built from the same public conversion functions."""

from __future__ import annotations

from fnmatch import fnmatchcase
from math import prod
import json

from .methods import (
    AuxiliaryValue,
    cast_direct,
    fp8_row_maxabs,
    grouped_absmax,
    import_encoded,
    nvfp4_maxabs,
    nvfp4_batched_maxabs,
    stored_scale_mse,
    u4z8_maxabs,
)
from .recipe import AuxiliaryReference
from .sources.fp8_block import fp8_block_source
from .sources.modelopt_nvfp4 import ModelOptNvfp4
from .sources.safetensors import tensor_source
from .sources.logical import LogicalSource

FP8_FP32 = "fp8_e4m3fn_row_fp32"
U4Z8 = "u4z8_g16_fp16"
_QWEN4_ROLES = {
    "mixer": ("text/layers/*/attention/query", "text/layers/*/attention/key", "text/layers/*/attention/output_gate", "text/layers/*/attention/value", "text/layers/*/attention/output", "text/layers/*/gdn/query", "text/layers/*/gdn/key", "text/layers/*/gdn/value", "text/layers/*/gdn/z", "text/layers/*/gdn/output"),
    "experts": ("text/layers/*/moe/experts/gate", "text/layers/*/moe/experts/up", "text/layers/*/moe/experts/down"),
    "mtp_mixer": ("mtp/layers/0/attention/*",),
    "mtp_experts": ("mtp/layers/0/moe/experts/gate", "mtp/layers/0/moe/experts/up", "mtp/layers/0/moe/experts/down"),
    "ple": ("text/layers/*/ple/table",),
}

Q4 = "q4_g64_fp16"
Q5 = "q5_g64_fp16"
Q6 = "q6_g64_fp16"
Q8 = "q8_g32_fp16"
FP8 = "fp8_e4m3fn_row_bf16"


def _assign(recipe, name, format, *, source=None):
    method = grouped_absmax if format in (Q4, Q5, Q6, Q8) else cast_direct
    recipe.assign(name, format=format, method=method, source=source)


def _optional(model, recipe):
    for name, parameter in model.parameters.items():
        if not parameter.projection:
            continue
        if name.startswith("vision/"):
            if name == "vision/patch_embedding":
                format = Q6
            elif name.startswith("vision/merger/"):
                format = Q8
            elif name.endswith(
                ("/attention/query", "/attention/key", "/attention/value", "/mlp/fc1")
            ):
                format = Q4
            else:
                format = Q5
            _assign(recipe, name, format)
        elif name.startswith(("mtp/", "dflash/", "dflash2/")):
            if name.endswith(
                (
                    "/moe/router",
                    "/moe/shared_score",
                    "/attention_conv/kernel_projection",
                    "/mlp_conv/kernel_projection",
                    "/candidate_selector/hidden_projection",
                )
            ):
                continue
            _assign(recipe, name, Q8)
    for backend in ("dflash", "dflash2"):
        if backend not in model.components:
            continue
        layers = model.components[backend]["config"]["num_hidden_layers"]
        for layer in range(layers):
            prefix = f"{backend}/layers/{layer}/attention/"
            for role in ("key", "value"):
                recipe.share(prefix + "context_" + role, prefix + role)


def _dense_groupwise(model, recipe, vocabulary):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", vocabulary)
    _assign(recipe, "text/output_head", vocabulary)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        if name.endswith(
            (
                "/attention/query",
                "/attention/key",
                "/gdn/query",
                "/gdn/key",
                "/mlp/gate",
                "/mlp/up",
            )
        ):
            format = Q4
        else:
            format = Q5
        _assign(recipe, name, format)


def qwen3_6_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q6)


def qwen3_8_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q8)


def qwen3_6_35b_a3b(model, recipe, sources):
    if "num_experts" not in model.config:
        raise ValueError("this official recipe requires Qwen3.5 MoE mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q6)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(
            (
                "/gdn/a_projection",
                "/gdn/b_projection",
                "/moe/router",
                "/moe/shared_score",
            )
        ):
            continue
        if "/moe/experts/" in name:
            layer = int(name.split("/")[2])
            format = (
                (Q6 if layer in (34, 38, 39) else Q5) if name.endswith("/down") else Q4
            )
        else:
            format = Q8
        _assign(recipe, name, format)


def qwen3_6_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q8)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        layer = int(name.split("/")[2])
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        direct = (
            ("/attention/" in name and not name.endswith("/output") and layer < 24)
            or (name.endswith("/attention/output") and layer in (3, 7))
            or (name.endswith("/gdn/output") and layer == 4)
        )
        if direct:
            continue
        recipe.assign(
            name,
            format="nvfp4",
            method=import_encoded,
            source=model.source(name, quantized, "nvfp4"),
            activation_policy="AllowA4",
        )


def qwen3_8_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    recipe.assign("text/token_embedding", format=FP8, method=fp8_row_maxabs)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/") or name == "text/token_embedding":
            continue
        source = model.source(name, quantized)
        if not parameter.projection or name.endswith(
            ("/gdn/a_projection", "/gdn/b_projection")
        ):
            recipe.assign(name, source=source)
            continue
        layer = int(name.split("/")[2]) if name.startswith("text/layers/") else -1
        format = "nvfp4" if "/mlp/" in name and layer < 56 else FP8
        recipe.assign(
            name,
            format=format,
            method=import_encoded,
            source=model.source(name, quantized, format),
            activation_policy="AllowA4" if format == "nvfp4" else "AllowA8",
        )


def qwen3_8_flash_next(model, recipe, sources):
    """Qwen4Exp: NVIDIA NVFP4/FP8 expert banks, Q8 projections and paged NF4 PLE."""
    recipe.provenance.update(
        mapping="qwen4_exp_v1",
        nvidia_nvfp4="preserve_codes_scales; RN_FP32(1/multiplier) divisors",
        nvidia_mtp="preserve_e4m3fn_codes_bf16_block128x128_scales",
        stored_scale_mse="g16_fp64_exhaustive_e4m3fn_v1",
        q_gate_order="all_query_heads_then_all_gate_heads",
        convolution_order="tap_channel",
    )
    cfg = model.config
    expected = {
        "architectures": ["Qwen4ExpForCausalLM"],
        "hidden_size": 2560,
        "vocab_size": 248320,
        "num_hidden_layers": 48,
        "num_experts": 512,
        "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640,
        "hc_count": 4,
        "hc_lowrank": 320,
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "indexer_n_heads": 4,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 128,
        "ple_layer_ids": [1],
        "split_ngram_parts": 128,
        "ple_shard_shape": [2500012, 160],
        "ple_embed_dim": 2560,
        "layer_types": [
            "indexed_attention" if l % 4 == 3 else "linear_attention" for l in range(48)
        ],
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"qwen3_8_flash_next requires {key}={value!r}")
    recipe.place("text/*", (0,))
    for component in ("vision", "mtp"):
        if component in model.components:
            recipe.place(component + "/*", (1 if component == "vision" else 2, 0))
    for name in model.parameters:
        if name.startswith("text/layers/") and name.endswith(
            (
                "/gdn/query",
                "/gdn/key",
                "/gdn/value",
                "/gdn/z",
                "/gdn/output",
                "/attention/query",
                "/attention/output_gate",
                "/attention/output",
            )
        ):
            _assign(recipe, name, Q8)

    h, intermediate, experts = 2560, 640, 512
    width = h * intermediate
    nvidia = ModelOptNvfp4(sources["nvidia"])
    controls = []
    covered = set()
    for layer in range(48):
        p = f"text/layers/{layer}/moe/experts/"
        layer_inputs = None
        gate_up_parts = []
        down_parts = []
        for expert in range(experts):
            prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}."
            refs = {}
            scalars = {}
            for role in ("gate", "up", "down"):
                source = prefix + role + "_proj"
                refs[role] = nvidia.matrix(
                    source, (h, intermediate) if role == "down" else (intermediate, h)
                )
                scalars[role] = nvidia.scales(source)
                covered.update(
                    source + "." + field
                    for field in (
                        "weight",
                        "weight_scale",
                        "weight_scale_2",
                        "input_scale",
                    )
                )
            if scalars["gate"] != scalars["up"]:
                raise ValueError(f"{prefix}: gate/up multiplier words differ")
            input_words = (scalars["gate"][2], scalars["down"][2])
            if layer_inputs is None:
                layer_inputs = input_words
                controls.extend((scalars["gate"][3], scalars["down"][3]))
            elif layer_inputs != input_words:
                raise ValueError(
                    f"{prefix}: activation multipliers differ within the layer"
                )
            first, last = expert * width, (expert + 1) * width
            gate_up_parts.extend(
                (p + role, first, last, refs[role]) for role in ("gate", "up")
            )
            down_parts.append((p + "down", first, last, refs["down"]))
        for part, shape, order in (
            (gate_up_parts, (experts, 2 * intermediate, h), 0),
            (down_parts, (experts, h, intermediate), 1),
        ):
            recipe.parent(
                part,
                shape=shape,
                format="nvfp4",
                method=import_encoded,
                layout="batched_block_scale_k16_m128x4_v1",
                order=(3, layer, order),
            )
    expected_experts = {
        key
        for key in nvidia.store.weight_map
        if key.startswith("model.language_model.layers.") and ".mlp.experts." in key
    }
    if covered != expected_experts:
        raise ValueError("NVIDIA source must contain the complete main expert tensors")
    recipe.auxiliary_tensor(
        "expert_activation_divisors",
        AuxiliaryValue("fp32", (48, 2), b"".join(controls)),
    )
    for layer in range(48):
        for role in ("gate", "up", "down"):
            name = f"text/layers/{layer}/moe/experts/{role}"
            recipe.use(
                name,
                model.parameters[name].inputs[0],
                activation_policy="AllowA4",
                auxiliaries={
                    "activation_input_divisor": AuxiliaryReference(
                        "expert_activation_divisors", 2 * layer + int(role == "down")
                    )
                },
            )

    if "mtp" in model.components:
        for role in ("token_embedding", "output_head"):
            recipe.share("mtp/" + role, "text/" + role)
        for name, parameter in model.parameters.items():
            if name.startswith("mtp/") and parameter.source_factory is not None:
                recipe.assign(name, source=model.source(name, nvidia.store))
        p = "mtp/layers/0/moe/experts/"
        gate_up_parts = []
        down_parts = []
        for expert in range(experts):
            refs = {}
            for role in ("gate", "up", "down"):
                refs[role] = fp8_block_source(
                    nvidia.store,
                    f"mtp.layers.0.mlp.experts.{expert}.{role}_proj",
                    (h, intermediate) if role == "down" else (intermediate, h),
                )
            first, last = expert * width, (expert + 1) * width
            gate_up_parts.extend(
                (p + role, first, last, refs[role]) for role in ("gate", "up")
            )
            down_parts.append((p + "down", first, last, refs["down"]))
        for part, shape, order in (
            (gate_up_parts, (experts, 2 * intermediate, h), 0),
            (down_parts, (experts, h, intermediate), 1),
        ):
            recipe.parent(
                part,
                shape=shape,
                format="fp8_e4m3fn_block128x128_bf16",
                method=import_encoded,
                order=(2, 1, order),
            )
        for role in ("gate", "up", "down"):
            name = p + role
            recipe.use(
                name,
                model.parameters[name].inputs[0],
                activation_policy="AllowA8",
            )

    rows, k = cfg["ple_shard_shape"]
    for shard in range(cfg["split_ngram_parts"]):
        name = f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
        source = tensor_source(sources["base"], name, (rows, k))
        recipe.parent(
            [
                (
                    "text/layers/1/ple/table",
                    shard * rows * k,
                    (shard + 1) * rows * k,
                    source,
                )
            ],
            shape=(rows, k),
            format="nf4_g16_e4m3fn",
            method=stored_scale_mse,
            order=(4, shard),
        )


def qwen3_8_27b_orcarouter_nvfp4(model, recipe, sources):
    """OrcaRouter's compressed-tensors checkpoint is the base: its NVFP4/FP8 matrix codes are
    imported as published, and its BF16 token embedding and full output head stay BF16."""
    if "num_experts" in model.config:
        raise ValueError("this recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    base = sources["base"]
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            continue
        # The same allocation as the official Qwen3.8-27B NVFP4 checkpoint; a source matrix
        # stored in another encoding fails the import.
        layer = int(name.split("/")[2])
        format = "nvfp4" if "/mlp/" in name and layer < 56 else FP8
        recipe.assign(
            name,
            format=format,
            method=import_encoded,
            source=model.source(name, base, format),
            activation_policy="AllowA4" if format == "nvfp4" else "AllowA8",
        )


def _nvfp4full_allocation(layer: int, family: str, role: str) -> str | None:
    """"bf16", "imported" or "local" for one Text projection of the nvfp4full profile."""
    if family == "gdn" and role in ("a_projection", "b_projection"):
        return "bf16"
    # The nine BF16 exception parents of the Qwen3.6-27B NVFP4 profile.
    if (
        (family == "attention" and role != "output" and layer < 24)
        or (family == "attention" and role == "output" and layer in (3, 7))
        or (family == "gdn" and role == "output" and layer == 4)
    ):
        return "bf16"
    if family == "mlp" and layer < 56:
        return "imported"
    return "local"


def _nvfp4full_site(layer: int, family: str, role: str) -> str:
    if family == "mlp":
        parent = "down_projection" if role == "down" else "gate_up_projection"
    else:
        parent = "output_projection" if role == "output" else "input_projection"
    return f"text/layers/{layer}/{family}/{parent}/input_scale_divisor"


def nvfp4full_local_sites(layer_types) -> set[str]:
    """Activation-divisor sites of locally encoded nvfp4full parents."""
    sites = set()
    for layer, kind in enumerate(layer_types):
        mixer = "attention" if kind == "full_attention" else "gdn"
        for family, role in ((mixer, "query"), (mixer, "output"), ("mlp", "gate"), ("mlp", "down")):
            if _nvfp4full_allocation(layer, family, role) == "local":
                sites.add(_nvfp4full_site(layer, family, role))
    return sites


def qwen3_8_27b_nvfp4full(model, recipe, sources):
    """Fuller NVFP4 on the official BF16 checkpoint, adapted from cometkim/ninfer (Apache-2.0).

    Layers 0-55 MLP words are imported bit-exactly from the Unsloth NVFP4 checkpoint (source
    ``quantized``); every other Text projection except the nine BF16 exception parents of the
    Qwen3.6-27B NVFP4 profile and the GDN a/b controls is encoded locally from BF16 with
    NVFP4_MAXABS_DIVISOR_RNE_V1. Locally encoded sites take their activation divisors from the
    calibration document (source ``calibration``, written by tools.convert.calibrate_nvfp4). The
    vocabulary endpoints are W8G32; MTP, Vision and DFlash2 keep the official formats.
    """
    if "num_experts" in model.config or model.config.get("num_hidden_layers") != 64:
        raise ValueError("nvfp4full requires the 64-layer Qwen3.8-27B Dense model")
    _optional(model, recipe)
    quantized = sources["quantized"]
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q8)
    local = {}
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        _, _, layer, family, role = name.split("/")
        allocation = _nvfp4full_allocation(int(layer), family, role)
        if allocation == "bf16":
            continue
        if allocation == "imported":
            recipe.assign(
                name,
                format="nvfp4",
                method=import_encoded,
                source=model.source(name, quantized, "nvfp4"),
                activation_policy="AllowA4",
            )
            continue
        recipe.assign(name, format="nvfp4", method=nvfp4_maxabs, activation_policy="AllowA4")
        local[name] = _nvfp4full_site(int(layer), family, role)
    document = json.loads(sources.path("calibration").read_text(encoding="utf-8"))
    measured = document.get("measured_sites", {})
    if set(measured) != set(local.values()):
        raise ValueError("calibration site set differs from the nvfp4full local sites")
    for name, site in local.items():
        divisor = AuxiliaryValue.activation_divisor(
            float(measured[site]["input_scale_divisor"])
        )
        for input_name in model.parameters[name].inputs:
            recipe.use(name, input_name, auxiliaries={"activation_input_divisor": divisor})


def _require_qwen4(model, *, moe: bool | None = None):
    if model.config.get("architectures") != ["Qwen4ExpForCausalLM"]:
        raise ValueError("this recipe requires the qwen4_exp architecture")
    if moe is not None and ("num_experts" in model.config) != moe:
        kind = "MoE" if moe else "dense"
        raise ValueError(f"this recipe requires {kind} qwen4_exp feed-forward blocks")


def qwen4_allocation(model, recipe, **formats):
    """Assign formats to Qwen4 parameter roles; a recipe for a new checkpoint is one call.

    ``mixer`` selects Text projections, ``experts`` and ``mtp_experts`` select native
    batched NVFP4 banks, and ``ple`` selects the U4Z8 table. NVFP4 Uses store an explicit
    unit global activation divisor; their K16 UE4M3 scales remain dynamic.
    """
    _require_qwen4(model)
    for role, format in formats.items():
        if role not in _QWEN4_ROLES:
            raise ValueError(f"unknown qwen4 role {role!r}")
        names = [
            name
            for name in model.parameters
            if any(fnmatchcase(name, pattern) for pattern in _QWEN4_ROLES[role])
            and (role == "ple" or model.parameters[name].projection)
        ]
        if format is None or not names:
            continue
        if role == "ple":
            if format != U4Z8:
                raise ValueError("PLE tables use u4z8_g16_fp16")
            shape = tuple(model.config["ple_shard_shape"])
            size = prod(shape)
            for name in names:
                source = model.parameters[name].source
                for shard in range(model.config["split_ngram_parts"]):
                    origin = shard * size
                    fragment = LogicalSource(shape, source.label + f"/shard/{shard}",
                        lambda a, b, parent=source, origin=origin:
                            parent.values(origin + a, origin + b))
                    recipe.parent([(name, origin, origin + size, fragment)], shape=shape,
                                  format=U4Z8, method=u4z8_maxabs, order=(4, shard))
        elif role.endswith("experts"):
            if format != "nvfp4":
                raise ValueError("expert banks use nvfp4")
            roots = sorted({name.rsplit("/", 1)[0] + "/" for name in names})
            e, h, i = (model.config[key] for key in
                       ("num_experts", "hidden_size", "moe_intermediate_size"))
            width = i * h
            for root in roots:
                gate_up, down = [], []
                for expert in range(e):
                    first, last = expert * width, (expert + 1) * width
                    for part in ("gate", "up", "down"):
                        name = root + part
                        source = model.parameters[name].source
                        shape = (h, i) if part == "down" else (i, h)
                        fragment = LogicalSource(shape, source.label + f"/expert/{expert}",
                            lambda a, b, parent=source, origin=first:
                                parent.values(origin + a, origin + b))
                        (down if part == "down" else gate_up).append(
                            (name, first, last, fragment))
                for parts, shape in ((gate_up, (e, 2 * i, h)), (down, (e, h, i))):
                    recipe.parent(parts, shape=shape, format="nvfp4",
                                  method=nvfp4_batched_maxabs,
                                  layout="batched_block_scale_k16_m128x4_v1")
                for part in ("gate", "up", "down"):
                    name = root + part
                    for input_name in model.parameters[name].inputs:
                        recipe.use(name, input_name, activation_policy="AllowA4",
                            auxiliaries={"activation_input_divisor":
                                         AuxiliaryValue.activation_divisor(1.0)})
        elif format in (FP8, FP8_FP32):
            recipe.assign(names, format=format, method=fp8_row_maxabs, activation_policy="AllowA8")
            if role == "mixer":
                # Keep the fork's Q,G,K,V parent and separate BF16 index parent.
                # This is a native layout supported by the public split Op.
                for name in names:
                    if not name.endswith("/attention/query"):
                        continue
                    root = name.removesuffix("query")
                    for roles, stored, method in (
                        (("query", "output_gate", "key", "value"), format, fp8_row_maxabs),
                        (("index_query", "index_key"), "bf16", cast_direct),
                    ):
                        parameters = [model.parameters[root + part] for part in roles]
                        recipe.parent([(p.name, 0, prod(p.shape), p.source) for p in parameters],
                            shape=(sum(p.shape[0] for p in parameters), parameters[0].shape[1]),
                            format=stored, method=method)
        else:
            # NVFP4 matrices need calibrated activation divisors; assign them in a custom recipe.
            raise ValueError(f"{role}: unsupported qwen4 allocation format {format!r}")


def qwen3_8_flash_next_mixed(model, recipe, sources):
    """Qwen3.8-Flash-Next from the official BF16 checkpoint in the qualified v3 allocation.

    The 48 Text and the MTP expert banks are NVFP4 with one divisor per expert, the Text
    attention and GDN projections FP8 with FP32 row multipliers, the PLE table U4Z8 G16;
    everything else, including the MTP attention, Vision and both vocabulary weights, is BF16.
    The production artifact imported its FP8, NVFP4 and U4Z8 words from third-party
    quantizations; this recipe encodes them from the official BF16 weights.
    """
    _require_qwen4(model, moe=True)
    qwen4_allocation(
        model,
        recipe,
        mixer=FP8_FP32,
        experts="nvfp4",
        mtp_experts="nvfp4",
        ple=U4Z8,
    )
    if "mtp" in model.components:
        recipe.share("mtp/token_embedding", "text/token_embedding")
        recipe.share("mtp/output_head", "text/output_head")


RECIPES = {
    "qwen3_6_27b": qwen3_6_27b,
    "qwen3_6_27b_nvfp4": qwen3_6_27b_nvfp4,
    "qwen3_8_27b_orcarouter_nvfp4": qwen3_8_27b_orcarouter_nvfp4,
    "qwen3_8_27b_nvfp4full": qwen3_8_27b_nvfp4full,
    "qwen3_8_flash_next_mixed": qwen3_8_flash_next_mixed,
    "qwen3_8_27b": qwen3_8_27b,
    "qwen3_8_27b_nvfp4": qwen3_8_27b_nvfp4,
    "qwen3_6_35b_a3b": qwen3_6_35b_a3b,
    "qwen3_8_flash_next": qwen3_8_flash_next,
}
