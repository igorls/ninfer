"""Official representation recipes built from the same public conversion functions."""

from __future__ import annotations

import json

from .methods import (
    AuxiliaryValue,
    cast_direct,
    fp8_row_maxabs,
    grouped_absmax,
    import_encoded,
    nvfp4_maxabs,
)

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
    """Activation-divisor sites of the locally encoded nvfp4full parents (135 for Qwen3.8-27B)."""
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


RECIPES = {
    "qwen3_6_27b": qwen3_6_27b,
    "qwen3_6_27b_nvfp4": qwen3_6_27b_nvfp4,
    "qwen3_8_27b": qwen3_8_27b,
    "qwen3_8_27b_nvfp4": qwen3_8_27b_nvfp4,
    "qwen3_8_27b_nvfp4full": qwen3_8_27b_nvfp4full,
    "qwen3_8_27b_orcarouter_nvfp4": qwen3_8_27b_orcarouter_nvfp4,
    "qwen3_6_35b_a3b": qwen3_6_35b_a3b,
}
