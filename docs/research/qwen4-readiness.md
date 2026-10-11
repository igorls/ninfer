# Qwen4 readiness plan

Active plan, started 2026-10-11. Goal: serve official Qwen4 checkpoints, starting with the
expected Qwen4-27B, within days of their release, through the existing `qwen4_exp` package that
runs Qwen3.8-Flash-Next (a Qwen4-architecture model). Remove this document when the first official
Qwen4 artifact is qualified; its lasting content moves to the model and conversion references.

## What is known and what is assumed

Known: Flash-Next's text, MTP and Vision mathematics, its configuration keys and its v3 artifact.
The engine selects `qwen4_exp` from the text component's `Qwen4ExpForCausalLM` /
`qwen4_exp_text` pair, which the converter writes, so an official checkpoint with the same
mathematics needs no new engine registration.

Assumed until the release: a dense Qwen4-27B keeps the Qwen3.8-27B trunk (hidden 5120, 24 query /
4 KV heads at head dimension 256, GDN 16/48 heads, MLP 17408) and adds Qwen4's hyper-connections,
QSA indexer attention, PLE and MTP. Every assumption below is checked against the real
`config.json` on launch day.

## Hard-coded today

Code facts from `workstation` `c2bbf835`:

- Every block is MoE (`weights.h` `BlockWeights::moe`, `execution/text.cpp` `run_layers`), and the
  MoE configuration keys are required.
- One PLE layer only, `ple_layer_ids == [2]` in the converter, a constant PLE boundary token
  (248044), 16 PLE heads with 80-byte rows and history 9. (Read from config since Q1.)
- Op geometries accept only Flash-Next shapes, with no generic fallback: hyper-connection
  (H 2560, 4 streams, rank 320), PLE n-gram (2560/10240/160), QSA indexer (d128, 4 heads,
  projection 640, 512 selected), selected-block attention (24 query / 2 KV heads, d256), gated
  RMSNorm-RoPE (13312 rows), NVFP4 expert bank (2560, 512 experts, top-10, 640), BF16-control GDN
  gating ([96, 2560]) and FP8 row-FP32 GDN input ([16384, 2560]).
- Linear dispatch is an exact (n, k) table; unknown shapes throw at load.
- The only artifact route is the pinned v2-to-v3 upgrade; there is no converter from official
  safetensors.

## Work items

Each item keeps Flash-Next bit-identical (Engine real cases, chunk invariance, MTP target parity)
unless it states otherwise.

| Item | Scope | Acceptance |
|---|---|---|
| Q1 Configuration reads | PLE layer list, boundary token from `eos_token_id`/config, PLE head count and history from config, accepted Vision/MTP strings in one place | Flash-Next unchanged; config tests for list and dense cases |
| Q2 Op geometry | Template or runtime dimensions with dispatch tables for the ops above; add the assumed 27B geometry (H 5120, 4 KV heads) | FP64 oracle tests at Flash-Next and assumed-27B geometries; Flash-Next routes bit-identical |
| Q3 Dense FFN | MoE optional per layer in Text and MTP; dense blocks use `linear_swiglu` + `linear` | Flash-Next unchanged; dense path covered by Q5 |
| Q4 Native converter | Official safetensors to v3 for the Qwen4 family: config normalizer, tensor map, packing groups, NVFP4/FP8 methods, streamed PLE quantizer, maintained template through `--resource` | Synthetic small-checkpoint tests; rebuilding Flash-Next from official sources on a Colab G4 matches the oracle |
| Q5 Random-weight end to end | Converter-built random-weight artifact at the assumed 27B geometry; Engine against transformers' `qwen4_exp` with the same weights | Teacher-forced logits within the existing parity tolerance; loads, prefill, decode, MTP |
| Q6 Launch-day runbook | Compare config with assumptions, finish shapes and Linear routes, NVFP4 allocation and BF16 KL gate, template diff, VRAM fit beside the desktop, real Engine tests, oracle gate, Supervisor preset, model card | Written before release; executed on release |

Order: Q1 and Q4 start now and run in parallel; Q2 follows Q1; Q3 and Q5 need Q2; Q6 is drafted
alongside.

## Status

**Q1: qualified on Flash-Next.** The re-derived artifact (`artifact_id` 3cade505...) passes the
loading and all 19 Engine real cases; greedy serving text matches the previous build and artifact.

- [x] `ple_layer_ids` is a list end to end: config (ascending, distinct, GDN layers only), one PLE
  parameter set and table per layer, injection before each listed block, decode and MTP transfers
  planar by PLE layer, and one convolution history per PLE layer in every state image.
- [x] The PLE boundary is the text config's `eos_token_id`, which the converter now writes; a
  config without it is rejected, so the published Flash-Next derivation must be re-derived.
- [x] PLE head count, row bytes and history columns come from config; native preparation rejects
  a PLE geometry `ops::ple_ngram` does not execute. The companion identities are constants in
  `config.h` (C++) and `config.py` (converter).
- [x] Tests: `ninfer_qwen4_exp_loading_test` (Flash-Next config, lists, boundary rules, rejected
  geometries, a dense config's clear rejection, two-layer bindings) and the two-layer state image
  test; `tests/convert/test_qwen4_exp.py`.
- [x] Re-derived the Flash-Next artifact (it also carries the maintained chat template) and pinned
  its file digests in `source.py`; `verify --file-digests` passes. `ninfer_qwen4_exp_loading_real_test`
  and the 19 cases of `ninfer_qwen4_exp_engine_real_test` pass on it (Windows, RTX PRO 6000).
  Served with MTP K=3, greedy text equals the previous build serving the previous artifact with the
  same template on seven of eight workloads, including both logprob readouts; the eighth workload's
  prompt changed between the runs. Published on the Hugging Face repository (2026-10-11).
- [ ] An empty `ple_layer_ids` (a checkpoint without PLE) is still rejected; making PLE optional is
  a Q3-like change if an official checkpoint needs it.

**Q4: source integration done, synthetic-validated, not qualified.** On branch `qwen4/converter`,
`python -m tools.convert` selects the `qwen4_exp` adapter (`tools/convert/qwen4_exp/build.py`)
for a Qwen4Exp `config.json`. The adapter shares `config.py` with the v2 upgrade; the upgrade's
`artifact_id` is pinned in tests, so the shared normalizer cannot change the derivation silently.
The conversion guide documents the route.

| Deliverable | State |
|---|---|
| Config normalizer | Shared with the upgrade. MoE or dense (`intermediate_size`) blocks; `ple_layer_ids` as a list; mixed per-layer blocks rejected |
| Tensor map | Every official Flash-Next tensor maps to the loader's logical names and Uses. Fused parents are packed in the source. A missing, misshapen or unmapped tensor of a selected component fails |
| Methods | `nvfp4_expert_maxabs` (one divisor per expert, the MTP bake), `u4z8_maxabs` (`U4Z8_RANGE_RNE_V1`, streamed), `fp8_row_maxabs` with FP32 row multipliers, INT64 `cast_direct`; BF16 stays the default |
| Recipes | `qwen3_8_flash_next`; `qwen4_allocation` assigns a format per role, so a new checkpoint's recipe is one call |
| Dense blocks | Converted under the Qwen3.5 dense names `mlp/{gate,up,down}`; not loadable until Q3 |
| Template | The maintained `qwen3_8.jinja` is the default when the official template's SHA256 is `c3cf9e34...d7a81041`. That digest belongs to the official Qwen3.8 template, which the Qwen3.8-27B and Flash-Next pins share (checked 2026-10-11). Any other template needs `--resource chat_template.jinja=PATH` |

Validated:
- Synthetic tiny checkpoints with the official names convert end to end: MoE with Vision and MTP
  through the CLI, and dense text with MTP. Every binding is read back and checked by independent
  decoders.
- At Flash-Next geometry, the official headers (1,658 tensors) map to exactly the qualified
  artifact's 1,668 bindings, 959 Uses, and its object shapes, formats and layouts.

Not validated:
- No real official checkpoint has been converted.
- The Flash-Next official-source rebuild is planned on a Colab G4 (about 360 GB in, 102 GiB out).
  It must pass the transformers oracle gate. Its FP8, NVFP4 and U4Z8 words are encoded from BF16,
  so they differ from the production artifact's imported words.

## Template

The maintained `tools/chat_templates/qwen3_8.jinja` is the official Qwen3.8-27B template plus three
NInfer changes (positional system/developer turns, final-assistant continuation, tool-result
histories without a user turn). Flash-Next's official template is byte-identical to the official
Qwen3.8 one. On release, diff the official Qwen4 template against it: if identical, the recipe uses
`qwen3_8.jinja`; otherwise add `qwen4.jinja` carrying the same three changes.
