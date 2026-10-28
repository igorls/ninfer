"""Reframe the fork's packed Qwen4Exp v3 directory without requantizing weights.

The successor entry shares the original read-only continuation files. Both entries
therefore retain their container-set ID. Keep the entries and continuations in the
same directory; deleting the original entry does not remove the shared parts.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from math import prod
from pathlib import Path
import struct

from ..artifact.framing import HEADER, MAGIC, PAYLOAD_ALIGNMENT
from ..artifact.layouts import align_up
from ..artifact.reader import Artifact
from ..artifact.schema import ArtifactError, encode_directory

PREFIX_BYTES = PAYLOAD_ALIGNMENT
CONTROLS_ID = "resource/text/ple_hash_controls"
DIVISOR_ID = "reframe/unit_activation_input_divisor"


def reframe_directory(original: dict, controls: bytes, read_tensor) -> tuple[dict, bytes]:
    """Map stored physical operands to canonical logical bindings, exactly."""
    out = deepcopy(original)
    cfg = out["components"]["text"]["config"]
    if cfg.get("architectures") != ["Qwen4ExpForCausalLM"]:
        raise ArtifactError("expected the fork's packed Qwen4Exp v3 artifact")
    old = original["bindings"]
    if "text/hyper_connection/norm" not in old:
        raise ArtifactError("expected packed fork bindings; this entry is already canonical")
    if len(controls) != 35 * 8:
        raise ArtifactError("Qwen4Exp PLE controls must contain 35 little-endian int64 values")
    objects = {obj["id"]: obj for obj in original["objects"]}
    bindings = {name: deepcopy(value) for name, value in old.items() if name.startswith("vision/")}
    uses = [deepcopy(use) for use in original["uses"] if use["parameter"].startswith("vision/")]
    old_uses = {}
    prefix = bytearray(PREFIX_BYTES)
    prefix[:len(controls)] = controls
    struct.pack_into("<f", prefix, 512, 1.0)
    added_objects = []
    joined_objects, removed_objects = {}, set()
    for use in original["uses"]:
        old_uses.setdefault(use["parameter"], []).append(use)

    def stored(name):
        binding = old[name]
        if set(binding) != {"object"}:
            raise ArtifactError(f"{name}: expected one complete packed parent")
        return binding["object"]

    def add(name, previous, input_name=None, binding=None):
        bindings[name] = deepcopy(old[previous] if binding is None else binding)
        if input_name is None:
            return
        candidates = old_uses.get(previous, [])
        if len(candidates) > 1:
            candidates = [use for use in candidates if use["input"] == input_name]
        if len(candidates) != 1:
            raise ArtifactError(f"{previous}: expected one explicit activation Use")
        use = deepcopy(candidates[0])
        use.update(parameter=name, input=input_name)
        if use.get("activation_policy") == "AllowA4":
            parent = objects[stored(previous)]
            if parent["format"] != "nvfp4":
                raise ArtifactError(f"{previous}: unit A4 divisor applies only to native NVFP4")
            # The old kernels explicitly passed 1.0 as the global divisor. K16
            # UE4M3 scales remain dynamic in both implementations.
            use["auxiliaries"] = {"activation_input_divisor": {"object": DIVISOR_ID}}
        uses.append(use)

    def slices(parent, rows, object_id=None):
        object_id = stored(parent) if object_id is None else object_id
        obj = objects[object_id]
        if len(obj["shape"]) != 2 or sum(rows) != obj["shape"][0]:
            raise ArtifactError(f"{parent}: packed projection geometry differs")
        width = obj["shape"][1]
        result, begin = [], 0
        for row_count in rows:
            end = begin + row_count * width
            result.append({"parts": [{"object": object_id, "range": [begin, end]}]})
            begin = end
        return result

    def hc(prefix, previous, final=False):
        add(prefix + "norm", previous + "norm")
        add(prefix + "up", previous + "input_mix/up", prefix + "lowrank_activation")
        if final:
            add(prefix + "down", previous + "input_mix/down", prefix + "normalized_streams")
            return
        # Match the maintained recipe's native Down/injection packing. Preparing
        # two independent parents would select a separate injection traversal.
        down, injection = (objects[stored(previous + role)]
                           for role in ("input_mix/down", "block_inject"))
        if (any(obj["format"] != "bf16" or obj["layout"] != "contiguous_le_v1"
                or len(obj["shape"]) != 2 for obj in (down, injection))
                or down["shape"][1] != injection["shape"][1]):
            raise ArtifactError("HC Down/injection require compatible stored BF16 matrices")
        parts = [read_tensor(obj) for obj in (down, injection)]
        if any(len(data) != obj["bytes"] for obj, data in zip((down, injection), parts)):
            raise ArtifactError("truncated HC Down/injection tensor")
        parent = "reframe/" + prefix + "down_injection"
        offset = align_up(len(payload_prefix), 256)
        payload_prefix.extend(bytes(offset - len(payload_prefix)))
        data = b"".join(parts)
        payload_prefix.extend(data)
        added_objects.append(dict(kind="tensor", id=parent,
                                  shape=[down["shape"][0] + injection["shape"][0], down["shape"][1]],
                                  format="bf16", layout="contiguous_le_v1", offset=offset,
                                  bytes=len(data)))
        count = prod(down["shape"])
        for role, old_role, bounds in (("down", "input_mix/down", [0, count]),
                                       ("inject", "block_inject", [count, count + prod(injection["shape"])])):
            add(prefix + role, previous + old_role, prefix + "normalized_streams",
                {"parts": [{"object": parent, "range": bounds}]})

    hidden = cfg["hidden_size"]
    experts = cfg["num_experts"]
    intermediate = cfg["moe_intermediate_size"]
    q_rows = cfg["num_attention_heads"] * cfg["head_dim"]
    kv_rows = cfg["num_key_value_heads"] * cfg["head_dim"]
    index_rows = [cfg["indexer_n_heads"] * cfg["indexer_head_dim"], cfg["indexer_kv_heads"] * cfg["indexer_head_dim"]]

    def attention_parent(parent):
        obj = objects[stored(parent)]
        if obj["shape"] != [2 * q_rows + 2 * kv_rows, hidden]:
            raise ArtifactError(f"{parent}: packed attention geometry differs")
        encodings = {
            ("bf16", "contiguous_le_v1"): (2, 0),
            ("fp8_e4m3fn_row_fp32", "row_scale_fp32_v1"): (1, 4),
        }
        encoding = encodings.get((obj["format"], obj["layout"]))
        if encoding is None:
            raise ArtifactError(f"{parent}: unsupported packed attention encoding")
        code_size, scale_size = encoding
        rows, columns = obj["shape"]
        raw = read_tensor(obj)
        if len(raw) != rows * (columns * code_size + scale_size):
            raise ArtifactError(f"{parent}: truncated packed attention payload")
        head_dim = cfg["head_dim"]
        spans = [
            (head * 2 * head_dim + half * head_dim, head_dim)
            for half in range(2) for head in range(cfg["num_attention_heads"])
        ] + [(2 * q_rows, 2 * kv_rows)]

        def plane(begin, row_bytes):
            return b"".join(raw[begin + row * row_bytes:begin + (row + count) * row_bytes]
                            for row, count in spans)

        payload = plane(0, columns * code_size)
        if scale_size:
            payload += plane(rows * columns * code_size, scale_size)
        offset = align_up(len(prefix), PAYLOAD_ALIGNMENT)
        prefix.extend(bytes(offset - len(prefix)))
        prefix.extend(payload)
        packed = deepcopy(obj)
        packed.update(id="reframe/" + parent, offset=offset)
        added_objects.append(packed)
        objects[packed["id"]] = packed
        return packed["id"]

    def block(prefix, kind):
        hc(prefix + "attention_hc/", prefix + "attention/hyper_connection/")
        hc(prefix + "moe_hc/", prefix + "mlp/hyper_connection/")
        mixer_input = prefix + "attention_input"
        if kind == "linear_attention":
            p = prefix + "gdn/"
            key_width = cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
            value_width = cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
            for role, binding in zip(("query", "key", "value", "z"), slices(p + "query_key_value_z", [key_width, key_width, value_width, value_width])):
                add(p + role, p + "query_key_value_z", mixer_input, binding)
            for role, binding in zip(("a_projection", "b_projection"), slices(p + "a_b_projection", [cfg["linear_num_value_heads"]] * 2)):
                add(p + role, p + "a_b_projection", mixer_input, binding)
            for role in ("a_log", "dt_bias", "convolution", "norm"):
                add(p + role, p + role)
            add(p + "output", p + "output", p + "gated_output")
        else:
            p = prefix + "attention/"
            packed = attention_parent(p + "query_gate_key_value")
            for role, binding in zip(("query", "output_gate", "key", "value"), slices(p + "query_gate_key_value", [q_rows, q_rows, kv_rows, kv_rows], packed)):
                add(p + role, p + "query_gate_key_value", mixer_input, binding)
            for role, binding in zip(("index_query", "index_key"), slices(p + "indexer/query_key", index_rows)):
                add(p + role, p + "indexer/query_key", mixer_input, binding)
            for role in ("query_norm", "key_norm"):
                add(p + role, p + role)
                add(p + "index_" + role, p + "indexer/" + role)
            add(p + "output", p + "output", p + "gated_output")
        p, previous = prefix + "moe/", prefix + "mlp/"
        moe_input = prefix + "moe_input"
        control = "reframe/" + p + "control"
        control_parts = []
        for role, old_role in (("router", "router"), ("shared_score", "shared_expert_gate")):
            obj = objects[stored(previous + old_role)]
            if obj["format"] != "bf16" or obj["layout"] != "contiguous_le_v1":
                raise ArtifactError("routing controls require stored BF16 matrices")
            data = read_tensor(obj)
            if len(data) != obj["bytes"]:
                raise ArtifactError("truncated routing control tensor")
            control_parts.append((role, old_role, data))
        offset = align_up(len(payload_prefix), 256)
        payload_prefix.extend(bytes(offset - len(payload_prefix)))
        data = b"".join(part[2] for part in control_parts)
        payload_prefix.extend(data)
        added_objects.append(dict(kind="tensor", id=control, shape=[experts + 1, hidden],
                                  format="bf16", layout="contiguous_le_v1", offset=offset,
                                  bytes=len(data)))
        for role, old_role, _ in control_parts:
            begin, end = (0, experts * hidden) if role == "router" else (experts * hidden, (experts + 1) * hidden)
            add(p + role, previous + old_role, moe_input,
                {"parts": [{"object": control, "range": [begin, end]}]})
        gate, up = (objects[stored(previous + "shared_expert/" + role)] for role in ("gate", "up"))
        if any(obj["format"] != "bf16" or obj["layout"] != "contiguous_le_v1" for obj in (gate, up)) or gate["shape"] != up["shape"] or gate["offset"] + gate["bytes"] != up["offset"]:
            raise ArtifactError("packed fork shared gate/up require adjacent BF16 matrices")
        joined = deepcopy(gate)
        joined.update(id="reframe/" + p + "shared_gate_up",
                      shape=[2 * gate["shape"][0], hidden], bytes=gate["bytes"] + up["bytes"])
        joined_objects[gate["id"]] = joined
        removed_objects.add(up["id"])
        width = prod(gate["shape"])
        for i, role in enumerate(("gate", "up")):
            add(p + "shared/" + role, previous + "shared_expert/" + role, moe_input,
                {"parts": [{"object": joined["id"], "range": [i * width, (i + 1) * width]}]})
        add(p + "shared/down", previous + "shared_expert/down", p + "shared/product")
        parent = previous + "experts/gate_up"
        object_id = stored(parent)
        if objects[object_id]["shape"] != [experts, 2 * intermediate, hidden]:
            raise ArtifactError(f"{parent}: unexpected expert bank geometry")
        stride = intermediate * hidden
        for offset, role in enumerate(("gate", "up")):
            parts = [{"object": object_id, "range": [(2 * e + offset) * stride, (2 * e + offset + 1) * stride]} for e in range(experts)]
            add(p + "experts/" + role, parent, moe_input, {"parts": parts})
        add(p + "experts/down", previous + "experts/down", p + "experts/product")

    payload_prefix = prefix
    add("text/token_embedding", "text/token_embedding")
    add("text/output_head", "text/output_head", "text/final_hidden")
    hc("text/final_hc/", "text/hyper_connection/", True)
    for layer, kind in enumerate(cfg["layer_types"]):
        block(f"text/layers/{layer}/", kind)
    layers = cfg["ple_layer_ids"]
    if len(layers) != 1 or cfg["ngram_size"] != 3 or cfg["heads_per_ngram"] != 8:
        raise ArtifactError("unsupported original PLE geometry")
    layer = layers[0] - 1
    p = f"text/layers/{layer}/ple/"
    for role, previous in (("key", "key_projection"), ("value", "value_projection")):
        add(p + role, p + previous, p + "embedding")
    for role, previous in (("norm_key", "key_norm"), ("norm_query", "query_norm"), ("norm_conv", "conv_norm"), ("convolution", "convolution")):
        add(p + role, p + previous)
    shards = [stored(p + f"embedding/shards/{i}") for i in range(cfg["split_ngram_parts"])]
    shard_shape = objects[shards[0]]["shape"]
    if any(objects[name]["shape"] != shard_shape for name in shards):
        raise ArtifactError("PLE shards differ in shape")
    bindings[p + "table"] = {"parts": [{"object": name, "range": [0, prod(shard_shape)]} for name in shards]}
    cfg.update(hidden_act="silu", output_gate_type="sigmoid", attention_bias=False, mamba_ssm_dtype="float32", norm_topk_prob=True, ple_layer_ids=[layer], ple_conv_dilation=cfg["ngram_size"], ple_shard_shape=shard_shape)
    cfg["layer_types"] = ["indexed_attention" if value == "full_attention" else value for value in cfg["layer_types"]]
    cfg["rope_parameters"].update(rope_type="default", mrope_interleaved=True)
    out["components"]["text"].setdefault("resources", {})["ple_hash_controls"] = CONTROLS_ID
    if "mtp" in out["components"]:
        block("mtp/layers/0/", "indexed_attention")
        hc("mtp/final_hc/", "mtp/hyper_connection/", True)
        bindings["mtp/token_embedding"] = deepcopy(bindings["text/token_embedding"])
        add("mtp/output_head", "text/output_head", "mtp/final_hidden")
        add("mtp/pre_fc_embedding_norm", "mtp/embedding_norm")
        add("mtp/pre_fc_hidden_norm", "mtp/hidden_norm")
        add("mtp/fc_embedding", "mtp/embedding_projection", "mtp/normalized_embedding")
        add("mtp/fc_hidden", "mtp/hidden_projection", "mtp/normalized_hidden_streams")
        out["components"]["mtp"]["config"] = dict(architectures=["Qwen4ExpMTP"], model_type="qwen4_exp_mtp", num_hidden_layers=1, hybrid=True, layer_types=["indexed_attention"], mtp_use_hidden_state_from_layer=None, rope_theta=cfg["rope_parameters"]["rope_theta"])
    if "vision" in out["components"]:
        out["components"]["vision"]["config"].update(hidden_act="gelu_pytorch_tanh", in_channels=3, deepstack_visual_indexes=[])
    prefix_bytes = align_up(len(payload_prefix), PAYLOAD_ALIGNMENT)
    payload_prefix.extend(bytes(prefix_bytes - len(payload_prefix)))
    out["objects"] = [deepcopy(joined_objects.get(obj["id"], obj))
                      for obj in out["objects"] if obj["id"] not in removed_objects]
    for obj in out["objects"]:
        obj["offset"] += prefix_bytes
        if obj.get("layout") == "expert_block_scale_k16_m128x4_v1":
            obj["layout"] = "batched_block_scale_k16_m128x4_v1"
    out["objects"][:0] = [
        dict(kind="resource", id=CONTROLS_ID, encoding="raw_bytes_v1", offset=0, bytes=len(controls)),
        dict(kind="tensor", id=DIVISOR_ID, shape=[1], format="fp32", layout="contiguous_le_v1", offset=512, bytes=4),
    ] + added_objects
    out["files"][0]["payload_bytes"] += prefix_bytes
    out["bindings"], out["uses"] = bindings, uses
    out.setdefault("provenance", {})["qwen4_exp_reframe"] = {"source": "fork packed v3", "original_payload": "unchanged", "packed_hc_down_injection": "byte-exact native parents", "packed_router_controls": "byte-exact copies", "packed_attention_rows": "byte-exact per-head Q/G deinterleaving, including row scales", "nvfp4_global_activation_divisor": 1.0}
    encode_directory(out)
    return out, bytes(payload_prefix)


def reframe(source: Path, destination: Path) -> dict:
    source, destination = source.resolve(), destination.resolve()
    if source.parent != destination.parent or source == destination:
        raise ArtifactError("write a distinct entry beside its shared continuation files")
    with Artifact(source) as artifact:
        original = artifact.directory.to_json()
        cfg = original["components"]["text"]["config"]
        p = f"text/layers/{cfg['ple_layer_ids'][0] - 1}/ple/embedding/"
        controls = b"".join(b"".join(artifact.iter_range(artifact.by_id[original["bindings"][p + name]["object"]].offset, n * 8)) for name, n in (("layer_multipliers", 3), ("ngram_head_vocab_sizes", 16), ("ngram_head_offsets", 16)))
        directory, prefix = reframe_directory(original, controls,
            lambda obj: b"".join(artifact.iter_range(obj["offset"], obj["bytes"])))
        raw = encode_directory(directory)
        payload_offset = align_up(HEADER.size + len(raw), PAYLOAD_ALIGNMENT)
        digest = hashlib.sha256()
        # Exclusive creation keeps both the original and any previous result intact.
        created = False
        try:
            with destination.open("xb") as output:
                created = True
                output.write(HEADER.pack(MAGIC, len(raw), artifact.artifact_id))
                output.write(raw)
                output.write(bytes(payload_offset - HEADER.size - len(raw)))
                output.write(prefix)
                for chunk in artifact.iter_range(0, original["files"][0]["payload_bytes"]):
                    digest.update(chunk)
                    output.write(chunk)
            copied = hashlib.sha256()
            with destination.open("rb") as output:
                output.seek(payload_offset + len(prefix))
                while chunk := output.read(8 * 1024 * 1024):
                    copied.update(chunk)
            if digest.digest() != copied.digest():
                raise ArtifactError("successor entry payload verification failed")
            with Artifact(destination) as successor:
                for obj in successor.objects:
                    # Check geometry/codecs without reading every large tensor again.
                    from ..artifact.schema import validate_encoding
                    validate_encoding(obj)
        except BaseException:
            if created:
                destination.unlink()
            raise
        return {"entry": str(destination), "copied_payload_bytes": original["files"][0]["payload_bytes"], "unchanged_payload_sha256": digest.hexdigest(), "shared_continuation_files": len(original["files"]) - 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(reframe(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
