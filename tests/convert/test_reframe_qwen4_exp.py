"""Exact physical-range and continuation preservation for packed v3 reframing."""

from copy import deepcopy
import hashlib
from math import prod
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.artifact.framing import HEADER, MAGIC, PART_MAGIC
from tools.artifact.layouts import align_up
from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorSpec, encode_directory, plan_objects
from tools.convert.reframe_qwen4_exp import reframe, reframe_directory


def packed_fixture():
    specs, bindings, uses = [], {}, []

    def tensor(name, shape, input_name=None, format="bf16", layout="contiguous_le_v1"):
        specs.append(TensorSpec(name, tuple(shape), format, layout))
        bindings[name] = {"object": name}
        if input_name:
            uses.append(dict(parameter=name, input=input_name, activation_policy="AllowA4" if format == "nvfp4" else "A16Only"))

    def hc(p, final=False):
        tensor(p + "norm", [512])
        tensor(p + "input_mix/down", [16, 512], p + "normalized_state")
        tensor(p + "input_mix/up", [512, 16], p + "low_rank")
        if not final:
            tensor(p + "block_inject", [4, 512], p + "normalized_state")

    def block(p, gdn):
        hc(p + "attention/hyper_connection/")
        hc(p + "mlp/hyper_connection/")
        if gdn:
            q = p + "gdn/"
            tensor(q + "query_key_value_z", [192, 128], p + "mixer_input")
            tensor(q + "a_b_projection", [4, 128], p + "mixer_input")
            for role, shape in (("a_log", [2]), ("dt_bias", [2]), ("convolution", [4, 128]), ("norm", [32])):
                tensor(q + role, shape)
            tensor(q + "output", [128, 64], q + "gated_output")
        else:
            q = p + "attention/"
            tensor(q + "query_gate_key_value", [192, 128], p + "mixer_input")
            tensor(q + "indexer/query_key", [64, 128], p + "mixer_input")
            for role in ("query_norm", "key_norm", "indexer/query_norm", "indexer/key_norm"):
                tensor(q + role, [32])
            tensor(q + "output", [128, 64], q + "gated_output")
        q = p + "mlp/"
        for role, shape in (("router", [2, 128]), ("shared_expert_gate", [1, 128]), ("shared_expert/gate", [64, 128]), ("shared_expert/up", [64, 128])):
            tensor(q + role, shape, p + "ffn_input")
        tensor(q + "shared_expert/down", [128, 64], q + "shared_expert/product")
        for role, shape, input_name in (("gate_up", [2, 128, 128], p + "ffn_input"), ("down", [2, 128, 64], q + "experts/product")):
            tensor(q + "experts/" + role, shape, input_name, "nvfp4", "batched_block_scale_k16_m128x4_v1")

    cfg = dict(architectures=["Qwen4ExpForCausalLM"], hidden_size=128, num_experts=2, moe_intermediate_size=64, num_attention_heads=2, head_dim=32, num_key_value_heads=1, indexer_n_heads=1, indexer_head_dim=32, indexer_kv_heads=1, linear_num_key_heads=1, linear_key_head_dim=32, linear_num_value_heads=2, linear_value_head_dim=32, layer_types=["linear_attention", "full_attention"], ple_layer_ids=[2], split_ngram_parts=2, ngram_size=3, heads_per_ngram=8, rope_parameters=dict(rope_theta=10000, partial_rotary_factor=0.5, mrope_section=[3, 3, 2]))
    tensor("text/token_embedding", [16, 128])
    tensor("text/output_head", [16, 128], "text/final_hidden")
    uses.append(dict(parameter="text/output_head", input="mtp/final_hidden", activation_policy="A16Only"))
    hc("text/hyper_connection/", True)
    block("text/layers/0/", True)
    block("text/layers/1/", False)
    p = "text/layers/1/ple/"
    tensor(p + "key_projection", [512, 32], p + "embedding")
    tensor(p + "value_projection", [128, 32], p + "embedding")
    for role in ("key_norm", "query_norm", "conv_norm"):
        tensor(p + role, [512])
    tensor(p + "convolution", [4, 512])
    for role, size in (("layer_multipliers", 3), ("ngram_head_vocab_sizes", 16), ("ngram_head_offsets", 16)):
        tensor(p + "embedding/" + role, [size], format="int64")
    for i in range(2):
        tensor(p + f"embedding/shards/{i}", [3, 160], format="u4z8_g16_fp16", layout="packed_u4_g16_v1")
    block("mtp/layers/0/", False)
    hc("mtp/hyper_connection/", True)
    tensor("mtp/embedding_norm", [128])
    tensor("mtp/hidden_norm", [512])
    tensor("mtp/embedding_projection", [128, 128], "mtp/normalized_embedding")
    tensor("mtp/hidden_projection", [128, 128], "mtp/normalized_hidden")
    objects = [obj.to_json() for obj in plan_objects(specs)]
    for obj in objects:
        if obj["layout"] == "batched_block_scale_k16_m128x4_v1":
            obj["layout"] = "expert_block_scale_k16_m128x4_v1"
    directory = dict(components={"text": {"config": cfg}, "mtp": {"config": {"architectures": ["Qwen4ExpMTP"]}, "target": "text"}}, objects=objects, bindings=bindings, uses=uses, files=[dict(path=None, payload_bytes=objects[-1]["offset"] + objects[-1]["bytes"])])
    return directory


class ReframeTest(unittest.TestCase):
    def test_hc_down_and_injection_keep_the_native_parent(self):
        original = packed_fixture()
        data = {
            obj["id"]: bytes((i * 7 + index * 11) % 256 for i in range(obj["bytes"]))
            for index, obj in enumerate(original["objects"])
        }
        result, prefix = reframe_directory(original, bytes(280), lambda obj: data[obj["id"]])
        objects = {obj["id"]: obj for obj in result["objects"]}
        for block in ("text/layers/0/", "text/layers/1/", "mtp/layers/0/"):
            for name, previous in (("attention_hc/", "attention/hyper_connection/"),
                                   ("moe_hc/", "mlp/hyper_connection/")):
                with self.subTest(block=block, name=name):
                    down, = result["bindings"][block + name + "down"]["parts"]
                    injection, = result["bindings"][block + name + "inject"]["parts"]
                    self.assertEqual(down["object"], injection["object"])
                    self.assertEqual(down["range"], [0, 16 * 512])
                    self.assertEqual(injection["range"], [16 * 512, 20 * 512])
                    parent = objects[down["object"]]
                    self.assertEqual(parent["shape"], [20, 512])
                    self.assertEqual(parent["format"], "bf16")
                    self.assertEqual(parent["layout"], "contiguous_le_v1")
                    payload = prefix[parent["offset"]:parent["offset"] + parent["bytes"]]
                    self.assertEqual(payload,
                                     data[block + previous + "input_mix/down"]
                                     + data[block + previous + "block_inject"])

    def test_physical_operands_and_semantics(self):
        original = packed_fixture()
        before = deepcopy(original)
        controls = struct.pack("<35q", *range(-17, 18))
        result, prefix = reframe_directory(original, controls, lambda obj: bytes(obj["bytes"]))
        self.assertEqual(original, before)
        self.assertEqual(prefix[:280], controls)
        self.assertEqual(struct.unpack_from("<f", prefix, 512), (1.0,))
        self.assertTrue(result["components"]["text"]["config"]["rope_parameters"]["mrope_interleaved"])
        self.assertEqual(result["components"]["text"]["config"]["ple_layer_ids"], [1])
        b = result["bindings"]
        # Canonical slices group Q and G after per-head rows are deinterleaved.
        for p in ("text/layers/1/attention/", "mtp/layers/0/attention/"):
            self.assertEqual(b[p + "output_gate"]["parts"][0]["range"], [64 * 128, 128 * 128])
            self.assertEqual(b[p + "key"]["parts"][0]["range"], [128 * 128, 160 * 128])
        self.assertEqual(b["text/layers/0/gdn/value"]["parts"][0]["range"], [64 * 128, 128 * 128])
        self.assertEqual([p["range"] for p in b["text/layers/0/moe/experts/up"]["parts"]], [[8192, 16384], [24576, 32768]])
        self.assertEqual(b["mtp/token_embedding"], b["text/token_embedding"])
        a4_uses = [u for u in result["uses"] if u.get("activation_policy") == "AllowA4"]
        self.assertEqual(len(a4_uses), 9)
        self.assertTrue(all("activation_input_divisor" in u["auxiliaries"] for u in a4_uses))
        self.assertFalse(any("hyper_connection" in name or "gate_up" in name for name in b))

    def test_per_head_query_gate_rows_and_scales(self):
        for format, layout, code_bytes in (
            ("bf16", "contiguous_le_v1", 2),
            ("fp8_e4m3fn_row_fp32", "row_scale_fp32_v1", 1),
        ):
            with self.subTest(format=format):
                original = packed_fixture()
                data = {obj["id"]: bytes(obj["bytes"]) for obj in original["objects"]}
                for obj in original["objects"]:
                    if not obj["id"].endswith("attention/query_gate_key_value"):
                        continue
                    n, k = obj["shape"]
                    obj.update(format=format, layout=layout, bytes=n * k * code_bytes + (n * 4 if code_bytes == 1 else 0))
                    if code_bytes == 2:
                        data[obj["id"]] = b"".join(
                            struct.pack("<H", 0x3F00 + row + col % 8)
                            for row in range(n) for col in range(k)
                        )
                    else:
                        data[obj["id"]] = (
                            bytes((row * 7 + col * 3) % 112 for row in range(n) for col in range(k))
                            + b"".join(struct.pack("<f", (row + 1) / 1024) for row in range(n))
                        )
                result, prefix = reframe_directory(original, bytes(280), lambda obj: data[obj["id"]])
                objects = {obj["id"]: obj for obj in result["objects"]}
                for block in ("text/layers/1/attention/", "mtp/layers/0/attention/"):
                    old_id = original["bindings"][block + "query_gate_key_value"]["object"]
                    old_payload = data[old_id]
                    for role, old_rows in (
                        ("query", [head * 64 + d for head in range(2) for d in range(32)]),
                        ("output_gate", [head * 64 + 32 + d for head in range(2) for d in range(32)]),
                        ("key", list(range(128, 160))),
                        ("value", list(range(160, 192))),
                    ):
                        part, = result["bindings"][block + role]["parts"]
                        obj = objects[part["object"]]
                        packed = data[obj["id"]] if obj["id"] in data else prefix[obj["offset"]:obj["offset"] + obj["bytes"]]
                        first = part["range"][0] // 128
                        for row, old_row in enumerate(old_rows):
                            width = 128 * code_bytes
                            self.assertEqual(packed[(first + row) * width:(first + row + 1) * width],
                                             old_payload[old_row * width:(old_row + 1) * width],
                                             f"{block}{role} row {row}")
                            if code_bytes == 1:
                                scales = 192 * 128
                                self.assertEqual(packed[scales + (first + row) * 4:scales + (first + row + 1) * 4],
                                                 old_payload[scales + old_row * 4:scales + (old_row + 1) * 4])

    def test_shared_parts_and_tensor_crossing_entry_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = packed_fixture()
            size = directory["files"][0]["payload_bytes"]
            # An existing matrix straddles the root/continuation boundary.
            split = directory["objects"][10]["offset"] + 37
            directory["files"] = [dict(path=None, payload_bytes=split), dict(path="original.part-0001", payload_bytes=size - split)]
            payload = bytearray((i * 29 + 7) % 256 for i in range(size))
            controls = {}
            for obj in directory["objects"]:
                if obj["format"] == "int64":
                    words = list(range(1, prod(obj["shape"]) + 1))
                    data = struct.pack("<" + "q" * len(words), *words)
                    payload[obj["offset"]:obj["offset"] + len(data)] = data
                    controls[obj["id"].split("/")[-1]] = data
            artifact_id = bytes(range(16))
            raw = encode_directory(directory)
            start = align_up(HEADER.size + len(raw), 4096)
            source = root / "original.ninfer"
            source.write_bytes(HEADER.pack(MAGIC, len(raw), artifact_id) + raw + bytes(start - HEADER.size - len(raw)) + payload[:split])
            part = root / "original.part-0001"
            part.write_bytes(HEADER.pack(PART_MAGIC, 1, artifact_id) + bytes(4096 - HEADER.size) + payload[split:])
            before = hashlib.sha256(source.read_bytes()).digest(), hashlib.sha256(part.read_bytes()).digest()
            destination = root / "canonical.ninfer"
            report = reframe(source, destination)
            self.assertEqual(report["copied_payload_bytes"], split)
            with Artifact(destination) as result:
                self.assertEqual(b"".join(result.iter_range(result.by_id[directory["objects"][0]["id"]].offset, size)), payload)
                obj = result.by_id["resource/text/ple_hash_controls"]
                self.assertEqual(b"".join(result.iter_range(obj.offset, obj.bytes)), controls["layer_multipliers"] + controls["ngram_head_vocab_sizes"] + controls["ngram_head_offsets"])
                old_objects = {obj['id']: obj for obj in directory['objects']}
                for block in ('text/layers/0/', 'text/layers/1/', 'mtp/layers/0/'):
                    parent = result.by_id[result.directory.bindings[block + 'moe/router']['parts'][0]['object']]
                    expected = b''.join(payload[old_objects[block + 'mlp/' + role]['offset']:
                        old_objects[block + 'mlp/' + role]['offset'] + old_objects[block + 'mlp/' + role]['bytes']]
                        for role in ('router', 'shared_expert_gate'))
                    self.assertEqual(result.read_object(parent.id), expected)
            self.assertEqual(before, (hashlib.sha256(source.read_bytes()).digest(), hashlib.sha256(part.read_bytes()).digest()))
            with self.assertRaises(FileExistsError):
                reframe(source, destination)
            original_iter = Artifact.iter_range
            failed = root / 'failed.ninfer'
            def fail_copy(owner, offset, count, *args, **kwargs):
                if offset == 0 and count == split:
                    raise OSError(28, 'No space left on device')
                return original_iter(owner, offset, count, *args, **kwargs)
            with patch.object(Artifact, 'iter_range', fail_copy):
                with self.assertRaisesRegex(OSError, 'space'):
                    reframe(source, failed)
            self.assertFalse(failed.exists())
            self.assertEqual(before, (hashlib.sha256(source.read_bytes()).digest(), hashlib.sha256(part.read_bytes()).digest()))

    def test_reject_wrong_source_and_unsafe_destination(self):
        original = packed_fixture()
        with self.assertRaisesRegex(ValueError, "35"):
            reframe_directory(original, bytes(8), lambda obj: bytes(obj["bytes"]))
        canonical, _ = reframe_directory(original, bytes(280), lambda obj: bytes(obj["bytes"]))
        with self.assertRaisesRegex(ValueError, "canonical"):
            reframe_directory(canonical, bytes(280), lambda obj: bytes(obj["bytes"]))
        with self.assertRaisesRegex(ValueError, "distinct"):
            reframe(Path("original.ninfer"), Path("original.ninfer"))


if __name__ == "__main__":
    unittest.main()
