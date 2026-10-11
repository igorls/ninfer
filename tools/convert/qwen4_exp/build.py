"""Official Qwen4-family safetensors to the logical parameters of the ``qwen4_exp`` architecture.

The native converter route: ``python -m tools.convert --model OFFICIAL_DIR --recipe ...`` selects
this adapter when ``config.json`` names a Qwen4Exp architecture (``config.SOURCE_ARCHITECTURES``).
It normalizes the config with the same ``config`` module as the v2 upgrade and declares every
logical parameter the ``qwen4_exp`` loader binds, under the names of
``docs/maintainer/qwen3.8-flash-next-artifact.md`` Section 5, with their mathematical Uses.

The loader binds fused parents as single parameters, so packing happens in the logical source:

- ``attention/query_gate_key_value`` is ``q_proj`` (per head ``[query, output_gate]``, the
  official row order), then ``k_proj`` and ``v_proj``;
- ``gdn/query_key_value_z`` is ``in_proj_qkv`` then ``in_proj_z``;
- ``gdn/a_b_projection`` is ``in_proj_a`` then ``in_proj_b``;
- ``mlp/experts/gate_up`` and ``mlp/experts/down`` are the official ``[E,2I,H]`` (rows
  ``[gate, up]`` per expert) and ``[E,H,I]`` banks.

Convolutions are stored ``[taps, channels]``; Vision reuses the ``qwen3_5`` tower mapping.
Dense feed-forward blocks (no ``num_experts``) use the ``qwen3_5`` dense names ``mlp/gate``,
``mlp/up`` (one packing group) and ``mlp/down``; the engine does not bind them yet (Qwen4
readiness Q3). PLE shard heights are storage facts read from the source tensors.

Every mapped tensor must exist with its expected shape, and every source tensor of a selected
component must be mapped: an official checkpoint with mathematics this adapter does not know
fails here instead of producing an artifact that silently drops it. ``model.source(name, store)``
reads the same official names from another store as values; encoded imports need an explicit
``LogicalSource``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

from ..model import Model, Parameter
from ..qwen3_5 import add_vision
from ..resources import load_resources
from ..sources.logical import concat_sources, transpose_source
from ..sources.safetensors import SafetensorsSource, tensor_source
from .config import MTP_CONFIG, accepts, source_text, text_config, vision_config
from .source import CHAT_TEMPLATE_SHA256

COMPONENTS = ("text", "vision", "mtp")
TEMPLATE_DIRECTORY = Path(__file__).resolve().parents[2] / "chat_templates"
# The official Qwen3.8 chat template: Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
# and Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540 ship byte-identical
# chat_template.jinja files (8,952 bytes, checked 2026-10-11). An official template with this
# digest is replaced by the maintained derivative; any other needs an explicit template.
MAINTAINED_TEMPLATES = {CHAT_TEMPLATE_SHA256: ("qwen3.8", "qwen3_8.jinja")}

__all__ = ["COMPONENTS", "MAINTAINED_TEMPLATES", "accepts", "build_model", "select_chat_template"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_chat_template(
    model_dir: Path, overrides: Mapping[str, str | Path]
) -> tuple[dict[str, Path | bytes], dict]:
    """Resource overrides with the chat template decided, and the decision's provenance record.

    An explicit ``chat_template.jinja`` override wins. Otherwise the official template must be a
    known official Qwen template, which installs the committed (LF) bytes of its maintained
    derivative from ``tools/chat_templates``, whatever the checkout's line endings.
    """
    result = {role: Path(path) for role, path in overrides.items()}
    if "chat_template.jinja" in result:
        return result, {
            "selection": "explicit",
            "sha256": _sha256(result["chat_template.jinja"]),
        }
    official = Path(model_dir) / "chat_template.jinja"
    digest = _sha256(official) if official.is_file() else None
    if digest not in MAINTAINED_TEMPLATES:
        state = f"has SHA256 {digest}" if digest else "is missing"
        raise ValueError(
            f"the official chat template {state} and is not a known official Qwen template; "
            "select one with --resource chat_template.jinja=PATH"
        )
    family, name = MAINTAINED_TEMPLATES[digest]
    maintained = (TEMPLATE_DIRECTORY / name).read_bytes().replace(b"\r\n", b"\n")
    result["chat_template.jinja"] = maintained
    return result, {
        "selection": "maintained",
        "official": family,
        "official_sha256": digest,
        "installed": f"tools/chat_templates/{name}",
        "sha256": hashlib.sha256(maintained).hexdigest(),
    }


class _Builder:
    def __init__(self, model: Model, store: SafetensorsSource, config: dict):
        self.model = model
        self.store = store
        self.config = config
        self.consumed: set[str] = set()
        self.stream_width = config["hc_count"] * config["hidden_size"]

    def add(self, name, sources, shape, *, inputs=(), direct="bf16", transpose=None):
        """Declare one logical parameter read from ``sources`` (official name, shape) pairs,
        concatenated along the leading axis, or one transposed source."""
        shape = tuple(shape)
        sources = tuple((source, tuple(dims)) for source, dims in sources)
        for source, dims in sources:
            actual = self.store.describe(source).shape
            if actual != dims:
                raise ValueError(f"{source}: expected source shape {dims}, got {actual}")
            self.consumed.add(source)

        def make(store, format=None):
            if format is not None:
                raise ValueError(f"{name}: provide an explicit encoded logical source")
            pieces = [tensor_source(store, source, dims) for source, dims in sources]
            if transpose is not None:
                return transpose_source(pieces[0], transpose, shape)
            return pieces[0] if len(pieces) == 1 else concat_sources(pieces)

        self.model.add(
            Parameter(
                name,
                shape,
                make(self.store),
                make,
                tuple(inputs),
                direct,
                residency=name.split("/", 1)[0],
            )
        )

    def hyper(self, prefix, source_prefix, *, block):
        width, low = self.stream_width, self.config["hc_lowrank"]
        state = prefix + "normalized_state"
        self.add(prefix + "norm", [(source_prefix + "hc_norm.weight", (width,))], (width,))
        self.add(
            prefix + "input_mix/down",
            [(source_prefix + "input_mix_weight_down.weight", (low, width))],
            (low, width),
            inputs=(state,),
        )
        self.add(
            prefix + "input_mix/up",
            [(source_prefix + "input_mix_weight_up.weight", (width, low))],
            (width, low),
            inputs=(prefix + "low_rank",),
        )
        if block:
            streams = self.config["hc_count"]
            self.add(
                prefix + "block_inject",
                [(source_prefix + "block_inject_weight.weight", (streams, width))],
                (streams, width),
                inputs=(state,),
            )

    def attention(self, prefix, source_prefix):
        c = self.config
        h, d = c["hidden_size"], c["head_dim"]
        q, kv = c["num_attention_heads"] * d, c["num_key_value_heads"] * d
        indexer = (c["indexer_n_heads"] + c["indexer_kv_heads"]) * c["indexer_head_dim"]
        p, sp = prefix + "attention/", source_prefix + "self_attn."
        mixer_input = prefix + "mixer_input"
        self.add(
            p + "query_gate_key_value",
            [
                (sp + "q_proj.weight", (2 * q, h)),
                (sp + "k_proj.weight", (kv, h)),
                (sp + "v_proj.weight", (kv, h)),
            ],
            (2 * q + 2 * kv, h),
            inputs=(mixer_input,),
        )
        self.add(p + "query_norm", [(sp + "q_norm.weight", (d,))], (d,))
        self.add(p + "key_norm", [(sp + "k_norm.weight", (d,))], (d,))
        self.add(
            p + "output",
            [(sp + "o_proj.weight", (h, q))],
            (h, q),
            inputs=(p + "gated_output",),
        )
        self.add(
            p + "indexer/query_key",
            [(sp + "indexer.index_qk_proj.weight", (indexer, h))],
            (indexer, h),
            inputs=(mixer_input,),
        )
        width = (c["indexer_head_dim"],)
        self.add(p + "indexer/query_norm", [(sp + "indexer.q_layernorm.weight", width)], width)
        self.add(p + "indexer/key_norm", [(sp + "indexer.k_layernorm.weight", width)], width)

    def gdn(self, prefix, source_prefix):
        c = self.config
        h = c["hidden_size"]
        kg = c["linear_num_key_heads"] * c["linear_key_head_dim"]
        vg = c["linear_num_value_heads"] * c["linear_value_head_dim"]
        heads, taps = c["linear_num_value_heads"], c["linear_conv_kernel_dim"]
        channels = 2 * kg + vg
        p, sp = prefix + "gdn/", source_prefix + "linear_attn."
        mixer_input = prefix + "mixer_input"
        self.add(
            p + "query_key_value_z",
            [(sp + "in_proj_qkv.weight", (channels, h)), (sp + "in_proj_z.weight", (vg, h))],
            (channels + vg, h),
            inputs=(mixer_input,),
        )
        self.add(
            p + "a_b_projection",
            [(sp + "in_proj_a.weight", (heads, h)), (sp + "in_proj_b.weight", (heads, h))],
            (2 * heads, h),
            inputs=(mixer_input,),
        )
        # The qwen4_exp loader binds BF16 GDN gating parameters (the official dtype).
        self.add(p + "a_log", [(sp + "A_log", (heads,))], (heads,))
        self.add(p + "dt_bias", [(sp + "dt_bias", (heads,))], (heads,))
        self.add(
            p + "convolution",
            [(sp + "conv1d.weight", (channels, 1, taps))],
            (taps, channels),
            transpose=(2, 0, 1),
        )
        self.add(
            p + "norm",
            [(sp + "norm.weight", (c["linear_value_head_dim"],))],
            (c["linear_value_head_dim"],),
        )
        self.add(
            p + "output",
            [(sp + "out_proj.weight", (h, vg))],
            (h, vg),
            inputs=(p + "gated_output",),
        )

    def moe(self, prefix, source_prefix):
        c = self.config
        h, e = c["hidden_size"], c["num_experts"]
        i, shared = c["moe_intermediate_size"], c["shared_expert_intermediate_size"]
        p, sp = prefix + "mlp/", source_prefix + "mlp."
        ffn_input = prefix + "ffn_input"
        self.add(p + "router", [(sp + "gate.weight", (e, h))], (e, h), inputs=(ffn_input,))
        self.add(
            p + "shared_expert_gate",
            [(sp + "shared_expert_gate.weight", (1, h))],
            (1, h),
            inputs=(ffn_input,),
        )
        for role in ("gate", "up"):
            self.add(
                p + "shared_expert/" + role,
                [(sp + f"shared_expert.{role}_proj.weight", (shared, h))],
                (shared, h),
                inputs=(ffn_input,),
            )
        self.add(
            p + "shared_expert/down",
            [(sp + "shared_expert.down_proj.weight", (h, shared))],
            (h, shared),
            inputs=(p + "shared_expert/product",),
        )
        self.add(
            p + "experts/gate_up",
            [(sp + "experts.gate_up_proj", (e, 2 * i, h))],
            (e, 2 * i, h),
            inputs=(ffn_input,),
        )
        self.add(
            p + "experts/down",
            [(sp + "experts.down_proj", (e, h, i))],
            (e, h, i),
            inputs=(p + "experts/product",),
        )

    def dense(self, prefix, source_prefix):
        h, i = self.config["hidden_size"], self.config["intermediate_size"]
        p, sp = prefix + "mlp/", source_prefix + "mlp."
        for role in ("gate", "up"):
            self.add(
                p + role,
                [(sp + f"{role}_proj.weight", (i, h))],
                (i, h),
                inputs=(prefix + "ffn_input",),
            )
        self.add(
            p + "down",
            [(sp + "down_proj.weight", (h, i))],
            (h, i),
            inputs=(p + "product",),
        )
        self.model.packing_groups.append((p + "gate", p + "up"))

    def block(self, prefix, source_prefix, mixer):
        self.hyper(
            prefix + "attention/hyper_connection/",
            source_prefix + "attn_hyper_connection.",
            block=True,
        )
        if mixer == "full_attention":
            self.attention(prefix, source_prefix)
        else:
            self.gdn(prefix, source_prefix)
        self.hyper(
            prefix + "mlp/hyper_connection/", source_prefix + "mlp_hyper_connection.", block=True
        )
        if "num_experts" in self.config:
            self.moe(prefix, source_prefix)
        else:
            self.dense(prefix, source_prefix)

    def ple(self, prefix, source_prefix):
        c = self.config
        h, width, embed = c["hidden_size"], self.stream_width, c["ple_embed_dim"]
        heads = (c["ngram_size"] - 1) * c["heads_per_ngram"]
        if embed % heads:
            raise ValueError("PLE heads must divide ple_embed_dim")
        p, sp = prefix + "ple/", source_prefix + "ple."
        self.add(
            p + "key_projection",
            [(sp + "key_proj.weight", (width, embed))],
            (width, embed),
            inputs=(p + "embedding",),
        )
        self.add(
            p + "value_projection",
            [(sp + "value_proj.weight", (h, embed))],
            (h, embed),
            inputs=(p + "embedding",),
        )
        for role, field in (
            ("query_norm", "norm_query"),
            ("key_norm", "norm_key"),
            ("conv_norm", "norm_conv"),
        ):
            self.add(p + role, [(sp + field + ".weight", (width,))], (width,))
        taps = c["ple_conv_kernel_size"]
        self.add(
            p + "convolution",
            [(sp + "conv1d.weight", (width, 1, taps))],
            (taps, width),
            transpose=(2, 0, 1),
        )
        ep, esp = p + "embedding/", sp + "ple_embedding."
        for role, field, count in (
            ("layer_multipliers", "layer_multipliers", c["ngram_size"]),
            ("ngram_head_offsets", "ngram_heads_offsets", heads),
            ("ngram_head_vocab_sizes", "ngram_heads_vocab_sizes", heads),
        ):
            self.add(ep + role, [(esp + field, (count,))], (count,), direct="int64")
        # One table split into split_ngram_parts equal row blocks; the height is a storage fact.
        first = esp + "ngram_embedding.shard_0.weight"
        rows = self.store.describe(first).shape[0]
        for shard in range(c["split_ngram_parts"]):
            self.add(
                f"{ep}shards/{shard}",
                [(f"{esp}ngram_embedding.shard_{shard}.weight", (rows, embed // heads))],
                (rows, embed // heads),
            )

    def mtp(self):
        h, width = self.config["hidden_size"], self.stream_width
        self.add(
            "mtp/embedding_projection",
            [("mtp.fc_embedding.weight", (h, h))],
            (h, h),
            inputs=("mtp/normalized_embedding",),
        )
        self.add(
            "mtp/hidden_projection",
            [("mtp.fc_hidden.weight", (h, h))],
            (h, h),
            inputs=("mtp/normalized_hidden",),
        )
        self.add("mtp/embedding_norm", [("mtp.pre_fc_norm_embedding.weight", (h,))], (h,))
        self.add("mtp/hidden_norm", [("mtp.pre_fc_norm_hidden.weight", (width,))], (width,))
        self.hyper("mtp/hyper_connection/", "mtp.hyper_connection_mixer.", block=False)
        self.block("mtp/layers/0/", "mtp.layers.0.", "full_attention")


def _require_coverage(store, consumed: set[str], selected: set[str]) -> None:
    unmapped = []
    for name in store.weight_map:
        if name in consumed:
            continue
        owner = (
            "vision"
            if name.startswith("model.visual.")
            else "mtp"
            if name.startswith("mtp.")
            else "text"
        )
        if owner in selected:
            unmapped.append(name)
    if unmapped:
        unmapped.sort()
        raise ValueError(
            f"{len(unmapped)} source tensors of the selected components have no qwen4_exp "
            f"mapping (the checkpoint carries mathematics this converter does not implement): "
            f"{unmapped[:8]}"
        )


def build_model(
    base: SafetensorsSource,
    *,
    components: tuple[str, ...] = ("text",),
    companions: Mapping[str, SafetensorsSource] | None = None,
    resource_overrides: Mapping[str, str | Path] | None = None,
) -> Model:
    selected = set(components)
    if "text" not in selected or selected - set(COMPONENTS):
        raise ValueError("qwen4_exp conversion selects text and optionally vision and mtp")
    if companions:
        raise ValueError("qwen4_exp conversion has no companion sources")
    config = text_config(base.config)
    records = {"text": {"config": config}}
    if "vision" in selected:
        records["vision"] = {"config": vision_config(base.config, config), "target": "text"}
    if "mtp" in selected:
        records["mtp"] = {"config": dict(MTP_CONFIG), "target": "text"}
    overrides, template = select_chat_template(base.root, resource_overrides or {})
    refs, resources, count, special = load_resources(
        base.root,
        vocab_size=config["vocab_size"],
        vision_config=records["vision"]["config"] if "vision" in selected else None,
        overrides=overrides,
    )
    for component, resource_refs in refs.items():
        records[component]["resources"] = resource_refs
    model = Model(
        records,
        resources=resources,
        token_count=count,
        special_token_ids=special,
        provenance={"chat_template": template},
    )
    builder = _Builder(model, base, config)
    h, r = config["hidden_size"], config["vocab_size"]
    tp = "model.language_model." if source_text(base.config) is not base.config else "model."
    builder.add("text/token_embedding", [(tp + "embed_tokens.weight", (r, h))], (r, h))
    head_inputs = ("text/final_hidden",) + (("mtp/final_hidden",) if "mtp" in selected else ())
    builder.add("text/output_head", [("lm_head.weight", (r, h))], (r, h), inputs=head_inputs)
    builder.hyper("text/hyper_connection/", tp + "hyper_connection_mixer.", block=False)
    for index, kind in enumerate(config["layer_types"]):
        builder.block(f"text/layers/{index}/", f"{tp}layers.{index}.", kind)
    for layer in config["ple_layer_ids"]:
        builder.ple(f"text/layers/{layer - 1}/", f"{tp}layers.{layer - 1}.")
    if "mtp" in selected:
        builder.mtp()
    if "vision" in selected:
        builder.consumed |= add_vision(model, base, records["vision"]["config"], h)
    _require_coverage(base, builder.consumed, selected)
    return model
