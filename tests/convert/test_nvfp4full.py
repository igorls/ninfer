"""nvfp4full recipe: allocation, calibration, imported words, and the local NVFP4 encoder.

Adapted from cometkim/ninfer tests/convert/test_nvfp4full_profile.py (Apache-2.0).
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
import json
import struct
from types import SimpleNamespace

import pytest
import torch

from tools.artifact.schema import TensorSpec
from tools.convert.__main__ import SourceInputs
from tools.convert.methods import (
    MethodInput,
    PrepareRequest,
    cast_direct,
    grouped_absmax,
    import_encoded,
    nvfp4_maxabs,
)
from tools.convert.model import Model
from tools.convert.official_recipes import nvfp4full_local_sites, qwen3_8_27b_nvfp4full
from tools.convert.qwen3_5 import _Builder
from tools.convert.quantization.nvfp4 import e2m1_rne_codes
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import EncodedRows, LogicalSource

LAYER_TYPES = ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(64)]


def _logical_model():
    config = dict(
        num_hidden_layers=64,
        hidden_size=5120,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_num_value_heads=48,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        layer_types=LAYER_TYPES,
    )
    model = Model({"text": {"config": config}})
    # The real adapter's logical selectors and packing groups, without weights.
    store = SimpleNamespace(path="synthetic", config={})
    builder = _Builder(model)
    for i in range(64):
        prefix = f"text/layers/{i}/"
        mixer = builder.attention if LAYER_TYPES[i] == "full_attention" else builder.gdn
        mixer(prefix, f"layers.{i}.", store, config)
        builder.dense(prefix, f"layers.{i}.", store, 5120, 17408)
    for name in ("text/token_embedding", "text/output_head"):
        builder.add(name, store, name, (248320, 5120), inputs=("vocab",))
    return model, store


def _synthetic_sources(recipe):
    """Bounded zero reads at model shapes; imported rows carry one shared divisor word."""
    for name, selections in recipe.selections.items():

        def source_for(selection):
            shape = selection.source.shape
            k = shape[-1] if shape else 1

            def encoded(begin, end):
                return EncodedRows(
                    "nvfp4",
                    torch.zeros((end - begin, k // 2), dtype=torch.uint8),
                    torch.zeros((end - begin, k // 16), dtype=torch.uint8),
                    struct.pack("<f", 2.0),
                )

            return LogicalSource(
                shape,
                name,
                lambda a, b: torch.zeros(b - a),
                encoded,
                lambda: struct.pack("<f", 2.0),
                lambda: struct.pack("<f", 3.0),
            )

        recipe.selections[name] = [
            replace(selection, source=source_for(selection)) for selection in selections
        ]


class _Sources(dict):
    def __init__(self, calibration, **values):
        super().__init__(values)
        self.calibration = calibration

    def path(self, name):
        assert name == "calibration"
        return self.calibration


def test_local_site_set():
    sites = nvfp4full_local_sites(LAYER_TYPES)
    assert len(sites) == 135
    assert "text/layers/4/gdn/input_projection/input_scale_divisor" in sites
    assert "text/layers/4/gdn/output_projection/input_scale_divisor" not in sites
    assert "text/layers/23/attention/input_projection/input_scale_divisor" not in sites
    assert "text/layers/27/attention/input_projection/input_scale_divisor" in sites
    assert "text/layers/7/attention/output_projection/input_scale_divisor" not in sites
    assert "text/layers/11/attention/output_projection/input_scale_divisor" in sites
    assert "text/layers/55/mlp/down_projection/input_scale_divisor" not in sites
    assert "text/layers/56/mlp/gate_up_projection/input_scale_divisor" in sites


def test_allocation_packing_and_calibration(tmp_path):
    model, quantized = _logical_model()
    path = tmp_path / "calibration.json"
    sites = nvfp4full_local_sites(LAYER_TYPES)
    path.write_text(
        json.dumps({"measured_sites": {s: {"input_scale_divisor": 3.25} for s in sites}})
    )
    recipe = Recipe(model)
    qwen3_8_27b_nvfp4full(model, recipe, _Sources(path, quantized=quantized))
    for name, selections in recipe.selections.items():
        chosen = selections[0]
        if name in ("text/token_embedding", "text/output_head"):
            assert (chosen.format, chosen.method) == ("q8_g32_fp16", grouped_absmax)
        elif "/mlp/" in name:
            layer = int(name.split("/")[2])
            assert chosen.format == "nvfp4"
            assert chosen.method is (import_encoded if layer < 56 else nvfp4_maxabs)
    assert recipe.selections["text/layers/3/attention/query"][0].format == "bf16"
    assert recipe.selections["text/layers/4/gdn/output"][0].format == "bf16"
    assert recipe.selections["text/layers/5/gdn/a_projection"][0].method is cast_direct
    assert recipe.selections["text/layers/27/attention/query"][0].method is nvfp4_maxabs
    local_inputs = sum(
        len(parameter.inputs)
        for name, parameter in model.parameters.items()
        if recipe.selections[name][0].method is nvfp4_maxabs
    )
    assert len(recipe.auxiliary_overrides) == local_inputs
    assert all(v.data == struct.pack("<f", 3.25) for v in recipe.auxiliary_overrides.values())

    # Real v3 preparation at model shapes: default packing groups form every parent.
    _synthetic_sources(recipe)
    prepared = recipe.prepare(device="cpu")
    methods = [job.method_name for job in prepared.weights]
    assert methods.count("nvfp4_maxabs") == 135
    assert methods.count("import_encoded") == 112
    assert sum(job.spec.format == "nvfp4" for job in prepared.weights) == 247
    shapes = {job.spec.shape for job in prepared.weights if job.spec.format == "nvfp4"}
    assert shapes == {(14336, 5120), (16384, 5120), (34816, 5120), (5120, 6144), (5120, 17408)}
    # Excluding the 48 a/b control parents, the BF16 projection parents are exactly the six
    # attention inputs, two attention outputs and one GDN output of the Qwen3.6 exception set.
    exceptions = set()
    for name, parameter in model.parameters.items():
        if (
            name.startswith("text/layers/")
            and parameter.projection
            and not name.endswith(("/a_projection", "/b_projection"))
            and recipe.selections[name][0].format == "bf16"
        ):
            binding = prepared.bindings[name]
            exceptions.update(part["object"] for part in binding.get("parts", [binding]))
    assert len(exceptions) == 9
    nvfp4 = {
        name
        for name, selections in recipe.selections.items()
        if selections[0].format == "nvfp4"
    }
    assert all(
        use["activation_policy"] == "AllowA4" and "auxiliaries" in use
        for use in prepared.uses
        if use["parameter"] in nvfp4
    )

    path.write_text('{"measured_sites": {}}')
    with pytest.raises(ValueError, match="calibration site set"):
        qwen3_8_27b_nvfp4full(model, Recipe(model), _Sources(path, quantized=quantized))


def test_calibration_source_path_and_provenance(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text("{}")
    with ExitStack() as stack:
        sources = SourceInputs(SimpleNamespace(path=tmp_path / "base"), {"calibration": path}, stack)
        assert sources.provenance() == {"base": {"path": str(tmp_path / "base")}}
        assert sources.path("calibration") == path
        provenance = sources.provenance()
        assert provenance["calibration"]["path"] == str(path)
        assert len(provenance["calibration"]["sha256"]) == 64
        with pytest.raises(ValueError, match="provide --source missing=PATH"):
            sources.path("missing")


def test_e2m1_exact_ties_and_signed_zero():
    values = torch.tensor([0.0, -0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0, -0.75])
    assert e2m1_rne_codes(values).tolist() == [0, 8, 0, 2, 2, 4, 4, 6, 6, 7, 10]


def _encode(values, sources, chunk):
    spec = TensorSpec("parent", tuple(values.shape), "nvfp4", "block_scale_k16_m128x4_v1")
    inputs = tuple(MethodInput(str(i), source, ()) for i, source in enumerate(sources))
    request = PrepareRequest(spec, inputs, {}, {}, device="cpu", rows_per_chunk=chunk)
    result = []
    nvfp4_maxabs(request).produce(
        SimpleNamespace(write_codes=lambda start, c, s, d: result.append((c, s, d)))
    )
    return (
        torch.cat([x[0] for x in result]),
        torch.cat([x[1] for x in result]),
        {x[2] for x in result},
    )


def test_parent_divisor_is_global_and_codes_match_independent_oracle():
    # Unequal slice maxima: encoding each grouped logical input on its own would be incorrect.
    generator = torch.Generator().manual_seed(7)
    values = torch.randn(256, 64, generator=generator).to(torch.bfloat16).float()
    values[:128] *= 0.25
    values[3, 5] = 0.0
    values[200, 17] = -0.0
    sources = [
        LogicalSource((128, 64), str(i), lambda a, b, i=i: values[i * 128 : (i + 1) * 128].reshape(-1)[a:b])
        for i in range(2)
    ]
    a, b = _encode(values, sources, 128), _encode(values, sources, 512)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and a[2] == b[2]
    (divisor_word,) = a[2]
    divisor = struct.unpack("<f", divisor_word)[0]
    assert divisor_word == struct.pack("<f", 2688.0 / float(values.abs().max()))

    # Independent FP64 oracle: E4M3FN scale by exhaustive nearest search, E2M1 nearest-even code.
    e4m3 = torch.arange(0x7F, dtype=torch.uint8).view(torch.float8_e4m3fn).double()
    y = (values.double() * divisor).float().double().reshape(256, 4, 16)
    target = (y.abs().amax(-1) / 6).float().double().clamp(max=448)
    distance = (target[..., None] - e4m3).abs()
    nearest = distance.min(-1, keepdim=True).values
    tie_even = torch.where(distance == nearest, torch.arange(0x7F) + (torch.arange(0x7F) % 2) * 256, 10_000)
    scale_words = tie_even.argmin(-1).to(torch.uint8)
    assert torch.equal(a[1], scale_words)
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)
    decoded = e4m3[scale_words.long()]
    ratio = torch.where(decoded[..., None] > 0, (y / decoded[..., None]).float().double(), 0.0)
    distances = (ratio.abs()[..., None] - grid).abs()
    best = distances.min(-1, keepdim=True).values
    magnitude = torch.where(distances == best, torch.arange(8) + (torch.arange(8) % 2) * 8, 100).argmin(-1)
    codes = (magnitude | (torch.signbit(ratio).long() << 3)).to(torch.uint8).reshape(256, 64)
    assert torch.equal(a[0], codes[:, 0::2] | (codes[:, 1::2] << 4))
