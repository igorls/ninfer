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
  (248044), 16 PLE heads with 80-byte rows and history 9.
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

## Template

The maintained `tools/chat_templates/qwen3_8.jinja` is the official Qwen3.8-27B template plus three
NInfer changes (positional system/developer turns, final-assistant continuation, tool-result
histories without a user turn). Flash-Next's official template is byte-identical to the official
Qwen3.8 one. On release, diff the official Qwen4 template against it: if identical, the recipe uses
`qwen3_8.jinja`; otherwise add `qwen4.jinja` carrying the same three changes.
