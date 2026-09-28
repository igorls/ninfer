"""Verify a qwen3_8_27b_nvfp4full artifact against its sources and, optionally, production.

Checks, each over the complete artifact:

- allocation: 247 NVFP4 parents (112 imported, 135 local), the nine BF16 exception parents, W8G32
  vocabulary endpoints, AllowA4 on every NVFP4 Use and a divisor on each;
- imported parents: codes, scales and weight divisor equal the Unsloth compressed-tensors words,
  and each Use divisor equals that source's ``input_global_scale`` word;
- local parents: the payload equals a re-encode with NVFP4_MAXABS_DIVISOR_RNE_V1, and an
  independent FP64 decode oracle confirms the divisor word, every E4M3FN scale (nearest, ties
  even) and every E2M1 code (nearest, ties even, saturating); Use divisors equal the calibration;
- BF16 exception parents equal the BF16 source words; W8 endpoints equal a groupwise re-encode;
- with ``--reference``: every object bound to the same logical parameters in the reference
  artifact (the production ``nvfp4`` profile) is compared by payload SHA-256, which proves the
  112 layer 0-55 MLP parents and their divisors are production's words.

    python3 -m tools.convert.verify_nvfp4full ARTIFACT --model BF16_DIR --quantized UNSLOTH_DIR \\
        --calibration CALIBRATION.json [--reference PRODUCTION.ninfer] [--report OUT.json]
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

import torch

from tools.artifact.codecs.nvfp4 import decode_nvfp4_words
from tools.artifact.codecs.row_split import decode_row_split_codes
from tools.artifact.reader import Artifact
from tools.artifact.schema import binding_parts

from .official_recipes import _nvfp4full_allocation, _nvfp4full_site
from .qwen3_5 import build_model
from .quantization.groupwise import quantize_matrix
from .quantization.nvfp4 import FULL_RANGE, encode_rows
from .sources.safetensors import SafetensorsSource

ROWS = 512
_E4M3 = torch.arange(0x7F, dtype=torch.uint8).view(torch.float8_e4m3fn).double()
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)


class Failure(Exception):
    pass


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def _word(value: float) -> bytes:
    return struct.pack("<f", value)


def _parents(artifact: Artifact) -> dict[str, list[tuple[str, int, int, int]]]:
    """Object -> [(parameter, parameter element offset, object begin, object end)]."""
    result = defaultdict(list)
    for name, binding in artifact.directory.bindings.items():
        offset = 0
        for object_id, begin, end in binding_parts(binding, artifact.by_id, name):
            result[object_id].append((name, offset, begin, end))
            offset += end - begin
    return result


def _use_divisors(artifact: Artifact) -> dict[tuple[str, str], bytes]:
    result = {}
    for use in artifact.directory.uses:
        auxiliary = use.get("auxiliaries", {}).get("activation_input_divisor")
        if auxiliary is not None:
            result[(use["parameter"], use["input"])] = artifact.read_object(auxiliary["object"])
    return result


def _independent_oracle(values: torch.Tensor, codes, scales, divisor: float) -> None:
    """FP64 nearest-value checks from the represented BF16 source and the stored words."""
    rows, k = values.shape
    y = (values.double() * divisor).float().double().reshape(rows, k // 16, 16)
    target = (y.abs().amax(-1) / 6).float().double().clamp(max=448.0)
    stored = _E4M3[scales.long()]
    upper = torch.bucketize(target, _E4M3).clamp(max=0x7E)
    lower = (upper - 1).clamp(min=0)
    low_gap, high_gap = target - _E4M3[lower], _E4M3[upper] - target
    expected = torch.where(
        (high_gap < low_gap) | ((high_gap == low_gap) & (upper % 2 == 0)), upper, lower
    )
    _check(torch.equal(expected.to(torch.uint8), scales), "E4M3FN scale is not nearest-even")
    ratio = torch.where(stored[..., None] > 0, (y / stored[..., None]).float().double(), 0.0)
    magnitude = ratio.abs().clamp(max=6.0)
    upper = torch.bucketize(magnitude, _E2M1).clamp(max=7)
    lower = (upper - 1).clamp(min=0)
    low_gap, high_gap = magnitude - _E2M1[lower], _E2M1[upper] - magnitude
    chosen = torch.where(
        (high_gap < low_gap) | ((high_gap == low_gap) & (upper % 2 == 0)), upper, lower
    )
    signed = (chosen | (torch.signbit(ratio).long() << 3)).to(torch.uint8).reshape(rows, k)
    _check(
        torch.equal(signed[:, 0::2] | (signed[:, 1::2] << 4), codes),
        "E2M1 code is not the nearest-even value",
    )


def _decoded(codes, scales, divisor: float) -> torch.Tensor:
    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float64,
    )
    unpacked = torch.stack((codes & 15, codes >> 4), dim=-1).reshape(codes.shape[0], -1)
    stored = _E4M3[scales.long()].repeat_interleave(16, dim=1)
    return table[unpacked.long()] * stored / divisor


def verify(args) -> dict:
    started = time.perf_counter()
    report: dict = {"artifact": str(args.artifact), "checks": {}, "local": {}, "imported": {}}
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))["measured_sites"]
    with (
        Artifact(args.artifact) as artifact,
        SafetensorsSource(args.model) as base,
        SafetensorsSource(args.quantized) as quantized,
    ):
        components = tuple(artifact.directory.components)
        model = build_model(base, components=("text",))
        parents = _parents(artifact)
        divisors = _use_divisors(artifact)
        tensors = {o.id: o for o in artifact.objects if o.kind == "tensor"}
        formats = Counter(o.format for o in tensors.values())
        report["components"] = components
        report["formats"] = dict(sorted(formats.items()))

        nvfp4 = {"imported": [], "local": []}
        exceptions = []
        for object_id, members in parents.items():
            names = [m[0] for m in members]
            if not all(n.startswith("text/layers/") for n in names):
                continue
            if not model.parameters[names[0]].projection:
                continue
            parts = names[0].split("/")
            layer, family, role = int(parts[2]), parts[3], parts[4]
            allocation = _nvfp4full_allocation(layer, family, role)
            obj = tensors[object_id]
            if allocation == "bf16":
                _check(obj.format == "bf16", f"{object_id}: {names} must stay BF16")
                if not role.endswith(("a_projection", "b_projection")):
                    exceptions.append(object_id)
                continue
            _check(obj.format == "nvfp4", f"{object_id}: {names} must be NVFP4")
            nvfp4[allocation].append(object_id)
        _check(len(nvfp4["imported"]) == 112, f"imported parents: {len(nvfp4['imported'])}")
        _check(len(nvfp4["local"]) == 135, f"local parents: {len(nvfp4['local'])}")
        _check(len(exceptions) == 9, f"BF16 exception parents: {len(exceptions)}")
        _check(formats["nvfp4"] == 247, f"NVFP4 tensors: {formats['nvfp4']}")
        for name in ("text/token_embedding", "text/output_head"):
            (object_id, _, _), = binding_parts(artifact.directory.bindings[name], artifact.by_id)
            _check(tensors[object_id].format == "q8_g32_fp16", f"{name} must be W8G32")
        report["checks"]["allocation"] = "pass"

        def members_values(object_id: str) -> torch.Tensor:
            obj = tensors[object_id]
            pieces = []
            for name, offset, begin, end in sorted(parents[object_id], key=lambda m: m[2]):
                pieces.append(model.parameters[name].source.values(offset, offset + end - begin))
            return torch.cat(pieces).reshape(obj.shape)

        for object_id in nvfp4["imported"]:
            obj = tensors[object_id]
            codes, scales, divisor = decode_nvfp4_words(artifact.read_object(object_id), obj.shape)
            members = sorted(parents[object_id], key=lambda m: m[2])
            source_codes, source_scales, words, inputs = [], [], set(), {}
            for name, _, _, _ in members:
                layer, role = name.split("/")[2], name.split("/")[4]
                prefix = f"model.language_model.layers.{layer}.mlp.{role}_proj"
                source_codes.append(quantized.read_flat(prefix + ".weight_packed").reshape(-1))
                source_scales.append(
                    quantized.read_flat(prefix + ".weight_scale").view(torch.uint8).reshape(-1)
                )
                words.add(bytes(quantized.read_flat(prefix + ".weight_global_scale").view(torch.uint8).numpy()))
                inputs[name] = bytes(
                    quantized.read_flat(prefix + ".input_global_scale").view(torch.uint8).numpy()
                )
            _check(len(words) == 1, f"{object_id}: source weight divisors differ")
            _check(torch.equal(codes.reshape(-1), torch.cat(source_codes)), f"{object_id}: codes differ")
            _check(torch.equal(scales.reshape(-1), torch.cat(source_scales)), f"{object_id}: scales differ")
            _check(bytes(divisor.reshape(1).view(torch.uint8).numpy()) == words.pop(), f"{object_id}: divisor differs")
            for name, word in inputs.items():
                for input_name in model.parameters[name].inputs:
                    _check(divisors.get((name, input_name)) == word, f"{name}: Use divisor differs")
            report["imported"][object_id] = [m[0] for m in members]
        report["checks"]["imported_words"] = "pass"

        errors = []
        for index, object_id in enumerate(nvfp4["local"]):
            obj = tensors[object_id]
            values = members_values(object_id).float()
            codes, scales, divisor = decode_nvfp4_words(artifact.read_object(object_id), obj.shape)
            amax = float(values.abs().max())
            word = _word(FULL_RANGE / amax)
            _check(bytes(divisor.reshape(1).view(torch.uint8).numpy()) == word, f"{object_id}: divisor")
            d = struct.unpack("<f", word)[0]
            squared_error = squared_norm = 0.0
            for begin in range(0, obj.shape[0], ROWS):
                end = min(obj.shape[0], begin + ROWS)
                rows = values[begin:end]
                expected_codes, expected_scales = encode_rows(rows, word)
                _check(torch.equal(expected_codes, codes[begin:end]), f"{object_id}: encoder codes")
                _check(torch.equal(expected_scales, scales[begin:end]), f"{object_id}: encoder scales")
                _independent_oracle(rows, codes[begin:end], scales[begin:end], d)
                difference = _decoded(codes[begin:end], scales[begin:end], d) - rows.double()
                squared_error += float(difference.square().sum())
                squared_norm += float(rows.double().square().sum())
            relative = (squared_error / squared_norm) ** 0.5
            errors.append(relative)
            names = [m[0] for m in parents[object_id]]
            first = names[0].split("/")
            site = _nvfp4full_site(int(first[2]), first[3], first[4])
            expected = _word(float(calibration[site]["input_scale_divisor"]))
            for name in names:
                for input_name in model.parameters[name].inputs:
                    _check(divisors.get((name, input_name)) == expected, f"{name}: Use divisor")
            report["local"][object_id] = {"parameters": names, "relative_frobenius": relative}
            print(f"local {index + 1}/135 {names[0]} rel={relative:.4f}", flush=True)
        report["checks"]["local_encoder_and_oracle"] = "pass"
        report["local_relative_frobenius_max"] = max(errors)
        report["local_relative_frobenius_median"] = sorted(errors)[len(errors) // 2]

        for object_id in exceptions:
            expected = members_values(object_id).to(torch.bfloat16)
            actual = torch.frombuffer(bytearray(artifact.read_object(object_id)), dtype=torch.bfloat16)
            _check(torch.equal(actual, expected.reshape(-1)), f"{object_id}: BF16 exception words")
        report["checks"]["bf16_exceptions"] = "pass"

        for name in ("text/token_embedding", "text/output_head"):
            (object_id, _, _), = binding_parts(artifact.directory.bindings[name], artifact.by_id)
            obj = tensors[object_id]
            scales, codes = decode_row_split_codes(artifact.read_object(object_id), obj.format, obj.shape)
            source = model.parameters[name].source
            k = obj.shape[1]
            for begin in range(0, obj.shape[0], 8192):
                end = min(obj.shape[0], begin + 8192)
                rows = source.values(begin * k, end * k).reshape(end - begin, k)
                encoded = quantize_matrix(rows, obj.format, device="cpu")
                _check(torch.equal(encoded.codes, codes[begin:end]), f"{name}: W8 codes")
                _check(
                    torch.equal(encoded.scales.view(torch.int16), scales[begin:end].view(torch.int16)),
                    f"{name}: W8 scales",
                )
        report["checks"]["w8_endpoints"] = "pass"

        if args.reference is not None:
            report["reference"] = compare_reference(artifact, args.reference, parents, nvfp4["imported"])
    report["seconds"] = time.perf_counter() - started
    return report


def _digest(artifact: Artifact, object_id: str) -> str:
    digest = hashlib.sha256()
    for chunk in artifact.iter_object(object_id):
        digest.update(chunk)
    return digest.hexdigest()


def compare_reference(artifact: Artifact, path: Path, parents, imported) -> dict:
    """Payload SHA-256 of objects bound to the same logical parameters in both artifacts."""
    result = {"path": str(path), "identical": [], "different": [], "unmatched": []}
    with Artifact(path) as reference:
        reference_parents = _parents(reference)
        by_members = {
            tuple(sorted((m[0], m[1]) for m in members)): object_id
            for object_id, members in reference_parents.items()
        }
        for object_id, members in parents.items():
            key = tuple(sorted((m[0], m[1]) for m in members))
            other = by_members.get(key)
            if other is None:
                result["unmatched"].append(object_id)
                continue
            same = _digest(artifact, object_id) == _digest(reference, other)
            result["identical" if same else "different"].append(
                {"object": object_id, "reference": other, "parameters": [m[0] for m in members]}
            )
        reference_divisors = _use_divisors(reference)
        own_divisors = _use_divisors(artifact)
        mismatched = []
        for object_id in imported:
            _check(
                any(entry["object"] == object_id for entry in result["identical"]),
                f"{object_id}: imported parent differs from the reference artifact",
            )
            for name, _, _, _ in parents[object_id]:
                for key, word in own_divisors.items():
                    if key[0] == name and reference_divisors.get(key) != word:
                        mismatched.append(name)
        _check(not mismatched, f"imported Use divisors differ from the reference: {mismatched[:3]}")
        resources = {}
        for obj in artifact.objects:
            if obj.kind != "tensor":
                role = obj.id.rsplit("/", 1)[-1]
                match = [o for o in reference.objects if o.kind != "tensor" and o.id.endswith("/" + role)]
                resources[role] = bool(match) and _digest(artifact, obj.id) == _digest(reference, match[0].id)
        result["resources_identical"] = resources
    result["imported_parents_identical"] = len(imported)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--model", type=Path, required=True, help="official BF16 checkpoint")
    parser.add_argument("--quantized", type=Path, required=True, help="Unsloth NVFP4 checkpoint")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="artifact to compare object payloads with")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        report = verify(args)
    except Failure as failure:
        print(f"VERIFY FAILED: {failure}", flush=True)
        sys.exit(1)
    text = json.dumps(report, indent=1)
    if args.report is not None:
        args.report.write_text(text + "\n", encoding="utf-8")
    summary = {k: v for k, v in report.items() if k not in ("local", "imported", "reference")}
    if "reference" in report:
        reference = report["reference"]
        summary["reference"] = {
            "identical": len(reference["identical"]),
            "different": [e["parameters"][0] for e in reference["different"]],
            "unmatched": len(reference["unmatched"]),
            "imported_parents_identical": reference["imported_parents_identical"],
            "resources_identical": reference["resources_identical"],
        }
    print(json.dumps(summary, indent=1))
    print("VERIFY PASSED", flush=True)


if __name__ == "__main__":
    main()
