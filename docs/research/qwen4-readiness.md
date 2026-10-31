# Qwen4 readiness plan

Started 2026-10-11; reconciled 2026-10-30 against the fork at b387a949 and upstream's
[Op checklist](https://github.com/Neroued/ninfer/blob/1882c6adbc98043fcf89f3589aa5addb3742777a/docs/designs/qwen4-27b-op-checklist.md)
and [architecture proposal](https://github.com/Neroued/ninfer/blob/1882c6adbc98043fcf89f3589aa5addb3742777a/docs/designs/qwen4-27b-rtx5090.md).
This is the current preparation plan, not a declaration of released-checkpoint support.
Remove it when the first official Qwen4 artifact is qualified; move lasting facts into the
model, conversion and performance references.

The goal remains native v3 Engine Text, MTP and Vision on both RTX 5090 and RTX PRO 6000,
under Windows/MSVC and Linux. Reuse the Qwen4Exp architecture where the mathematics matches;
do not add checkpoint-specific execution registrations. Preserve resident GPU execution when
the selected weights fit and leave usable desktop headroom. Spark, tensor parallelism and
GLM5Next are outside this preparation scope.

This documentation refresh does not change the sync/upstream-64d10fbf acceptance candidate
b387a949. The [5090 acceptance gate](https://github.com/igorls/ninfer/issues/37) remains separate
from implementing and qualifying a future dense Qwen4 configuration.

## Evidence and assumptions

The October 11 plan extrapolated from Qwen3.8-27B and Flash-Next: FFN width 17408 plus PLE.
Intel's later kernel evidence gives a better preparation candidate, but does not supply an
official complete config or checkpoint.

| Detail | Evidence level | Preparation decision |
|---|---|---|
| Dense Qwen4, hidden 5120, four HC streams / residual width 20480 | Directly visible in Intel's dense kernel identifiers and HC constants | Retain the matching geometry groundwork |
| FFN intermediate 14336 | Inferred from Intel TP2 projection shapes, including K5120/N14336 and N5120/K7168 | Replace 17408 as the leading candidate; confirm the FFN interpretation against the released config and tensor headers |
| HC rank 640; QSA Q24/KV4, D256 | Derived from local TP2 geometry and adopted by the upstream proposal | Prepare explicit profiles; final rank/head counts still come from the checkpoint |
| GDN Q/K16, V48, D128; indexer Q4/KV1, D128, compression 4 | Reused architecture geometry in the upstream proposal | Validate each config field and represented state contract on release |
| 64 layers, 3:1 GDN/QSA schedule, 262144 context, no PLE, exact MTP layout | Upstream design assumptions; not established by the inspected Intel projection/HC code | Exercise candidate configurations synthetically without declaring these official facts |
| NVFP4 GDN/FFN, FP8 HC down/up, BF16 injection, Q4 draft head, AllowA8/AllowA4 | NInfer representation and compute-policy choices | Treat as candidate recipes requiring numerical, quality, performance and resource qualification |
| Final names, tokenizer, template, Vision interface and quantization metadata | Require official resources and tensor headers | Do not infer them from kernel dimensions |

### Source provenance

Upstream's proposal explicitly cites
[Intel XPU PR #22](https://github.com/analytics-zoo/vllm-xpu-kernels/pull/22) and its
[kernel inventory](https://raw.githubusercontent.com/analytics-zoo/vllm-xpu-kernels/downstream/v0.31.0/custom-kernels-inventory.md).
The public-source retrieval on October 30 exposed these concrete anchors:

- [fp8_dense.cpp](https://raw.githubusercontent.com/analytics-zoo/vllm-xpu-kernels/downstream/v0.31.0/csrc/qwen38/fp8_dense.cpp):
  the Dense Qwen4 TP2 description and splits() entries for the projection shapes above.
- [hc_dense_sycl.cpp](https://raw.githubusercontent.com/analytics-zoo/vllm-xpu-kernels/downstream/v0.31.0/csrc/qwen38/hc_dense_sycl.cpp):
  kHidden=5120, kStreams=4 and their product kWidth=20480.
- The inventory describes TP2 local HC down weights with 320 low-rank outputs. Scaling local
  dimensions to a full-model profile is an inference, not a supplied official config.
- [llama.cpp #31072](https://github.com/ggml-org/llama.cpp/issues/31072) discusses the same PR.
  It is corroborating analysis of that source, not independent confirmation.

These Intel URLs name a moving branch, not an immutable source revision. The PR/API returned
404 during inspection; the raw files were retrievable earlier in the investigation but also
returned 404 on recheck. Preserve the distinction between the observed code, the pinned
upstream proposal and unavailable provenance. Recover an immutable source revision or verify
against official files before claiming checkpoint conformance. Neither availability failures
nor detailed kernel dimensions establish that NInfer's author has private model access.

## Current source and preserved groundwork

Current-support statements below refer to the b387a949 sync candidate, not to the older
preparation branches. The [model](../maintainer/qwen4_exp-model.md) and
[execution](../maintainer/qwen4_exp-execution.md) references own current runtime behavior.

- Native Qwen4Exp config and execution still require MoE blocks and exactly one PLE layer.
  Dense FFN and a PLE-free model are not integrated.
- The current converter is tools/convert/qwen4_exp.py with canonical bindings and an explicit
  [offline reframe](../maintainer/qwen3.8-flash-next-artifact.md#1-download-and-derivation).
  A native safetensors route exists; the old claim that only a v2 upgrade exists is obsolete.
  Its current text normalizer requires MoE fields and its block builder unconditionally emits
  MoE. Historical dense converter support must be adapted rather than assumed present.
- Current HC operands remain H2560/R320. The six Q2 geometry commits on qwen4/ops are not
  ancestors of b387a949. Their old APIs and representations cannot be merged as current support.
- Exact dispatch tables, resource queries and supported weight views must agree; adding a
  shape to a registry alone does not complete a fused Op or model route.

| Preserved work | Recorded outcome | Limit and reuse decision |
|---|---|---|
| Q1 on qwen4/prep / converter history | Config-derived PLE fields and EOS, multiple-PLE preparation, re-derived Flash artifact and 19 Engine cases on native Windows / PRO 6000 were recorded on October 11 | Historical evidence for that source/artifact; current single-PLE implementation is narrower. Seven of eight served workload comparisons matched; the eighth prompt differed |
| Q2 on qwen4/ops at be9126f2 | H5120/R640 BF16 HC, four-KV attention/RoPE, PLE width 5120, BF16 GDN controls and FP8 input [16384,5120]; independent-oracle fixtures exist | Explicitly source-implemented, not qualified. The recorded 135 byte-identical outputs protect old Flash Op routes; Engine cases and dense performance were not run |
| Q4 on qwen4/converter at 88d9d7fb | Tiny MoE/Vision/MTP and dense/MTP conversions with independent binding decoding; official Flash headers mapped to 1668 bindings / 959 Uses | Synthetic/header evidence only; no real official-source rebuild or dense Engine parity. BF16 was the default, with per-role allocation |
| b387a949 native Flash integration | Later resident and bounded Host Engine, mathematical and build evidence is recorded in upstream ports and performance | Does not qualify dense Qwen4, physical RTX 5090, native Windows GPU execution on this source, or the proposed recipe |

Known Q2 costs remain unresolved: four-KV attention uses six of sixteen MMA rows per CTA,
dense GDN controls use a split-8 route, and the widened FP8 GDN input reuses schedules selected
for K2560. These are reasons to measure the eventual production routes, not established speedups.

## Work items and upstream checklist mapping

Q1-Q6 retain their original ownership. Checklist identifiers refer to the pinned upstream
document; they identify candidate work, not completed ports.

| Item | Revised work | Acceptance / state |
|---|---|---|
| Q1 Config | Reconcile official fields with the evidence table; model Dense/MoE and PLE presence explicitly | Historical Flash qualification only; dense/no-PLE config and native binding remain pending |
| Q2 Ops | L1-L5, H1-H5, A1-A5 and M1 below; reuse applicable GDN/Vision operators | Older branch has partial geometry groundwork; new formats, views and production dispatch remain pending |
| Q3 Model composition | Dense linear_swiglu + linear in Text/MTP; optional PLE with complete state/resource/graph semantics | Not integrated; same architecture entry, explicit finite composition |
| Q4 Converter | Adapt the existing canonical adapter for dense/no-PLE and selected packing/format policies, including shared weight views | Old dense synthetic tests are reusable requirements, not proof the current converter implements them |
| Q5 Synthetic end to end | Converter-produced dense fixtures, Text/MTP/Vision handoffs, prefill/decode/verification, eager/graphs and recovery | No completed dense Engine qualification recorded; independent mathematical reference required |
| Q6 Official checkpoint and devices | Config/header reconciliation, model quality, capacity, same-route performance and desktop coexistence | Execute on real artifacts and qualify each GPU, OS and route separately |

### Q2 and Q4 concrete deltas

- **L1/L2, dense FFN:** prepare NVFP4 gate/up [28672,5120] and down [5120,14336].
  These replace the inherited 17408 assumption for the candidate. Check complete SwiGLU
  mathematics, stored divisors, activation permission, dispatch and workspace together.
- **H1-H5, HC:** reuse the older H5120/R640 mathematics and fixtures, but add the proposed
  FP8/AllowA8 down/up with separate BF16/A16 injection. The old all-BF16 geometry does not
  satisfy that contract. Preserve per-stream norm and the public BF16 residual boundary.
- **A1-A5/L3, QSA:** prepare four-KV-head state and attention with one six-output FP8 parent
  [14976,5120], ordered K/V/index-K/Q/gate/index-Q. The old 14336-row gated projection plus
  separate indexer is partial groundwork. Deinterleave source Q/gate and move matching scales.
- **M1, MTP history:** expose [2176,5120] as a three-output K/V/index-K prefix of that same
  immutable parent. Scales use the full parent's real plane offset; retain backing through
  graph replay. Do not duplicate or repack history weights at execution.
- **L4/L5, MTP stem/head:** FP8 [5120,5120] stem projections include fc_hidden at 4M columns;
  Q4 [248320,5120] proposal output retains the full vocabulary. Main verification keeps its
  full head. Draft count is not a decoder-layer count.
- **GDN:** adapt the candidate NVFP4 [16384,5120] input and its update/snapshot/record paths.
  The old FP8 row-FP32 route is different represented input. Retain FP32 recurrent state and
  A_log/dt_bias, the actual control semantics and sigmoid output gating.
- **Vision:** reuse the existing shared encoder; verify the merger's 5120-wide text handoff.
  A matching tensor width does not establish checkpoint/config compatibility.
- **No PLE:** implement the model/converter absence case if confirmed, while preserving the
  existing Flash PLE route. Wider PLE kernels are useful historical work, not a prerequisite
  for the no-PLE candidate. Do not add a fallback for an unrecognized architecture.

Sequence: reconcile Q1/config and Q4 mappings first; bring applicable Q2 work into the current
Op contracts; then Q3 composition and Q5 complete-path fixtures. Q6 is the release gate.
Upstream's Spark/multi-rank infrastructure is not a dependency for these single-GPU operators.

## Numerical, performance and release gates

Follow [Op development](../maintainer/op-development.md), [tests](../../tests/README.md) and
[benchmarks](../../bench/README.md). Existing criteria do not change to accommodate a new path.

- Independent FP32/FP64 references consume independently decoded stored weights/scales and the
  selected activation quantization. Exact transforms have exact references. Preserve public
  rounding/state boundaries; agreement between two kernels is insufficient.
- Cover M=1/4/8, verification widths (d+1)*B for d=1..5 and B=1..8, and actual wide prefill
  workloads. fc_hidden uses 4M. State operators retain T/B semantics and ragged valid columns.
- Test six-output and prefix projection views with nonuniform scales at M>1/B>1. Outputs must
  be contiguous per segment; parent row slices are not automatically contiguous columns.
  Verify snapshot/replay and accepted-prefix records without rerunning a shorter physical block.
- Keep applicable old Flash exact/state invariance checks; use unchanged numerical/target
  criteria across different arithmetic paths rather than promising universal token identity.
- Synthetic qualification does not establish quantization quality. A Transformers reference
  that only implements MoE cannot serve as a dense oracle without a validated dense extension.
  Q5 must supply an independent dense reference matching the represented architecture.
- With official weights, compare the selected recipe to an appropriate high-precision reference:
  scoring/distribution error, long-context indexer/retrieval behavior beyond all-selected cases,
  HC error propagation, Vision and MTP target correctness. Report draft acceptance, round time
  and final generation speed separately. AllowA8/AllowA4 permits a route; report which ran.
- New shapes establish capability. Performance comparisons require the same device, represented
  weights, precision, placement and workload, and retain every adverse result and resource cost.
  Do not transfer G4 timings to RTX 5090 or infer native Windows GPU acceptance from a build.
- On release, inspect actual config/tensor headers before confirming layer count, FFN width,
  PLE, MTP or Vision. Diff tokenizer/template resources, choose a measured recipe, verify v3
  bindings and numerical/Engine gates, then document the model card and Supervisor preset.
  Official-source conversion/rebuilds and model benchmarks remain pending work, not this
  documentation refresh's claimed result.

## Hardware feasibility and acceptance

The existing mixed Flash artifact and a future dense Qwen4 artifact have different requirements.
The [Flash resource contract](../maintainer/qwen4_exp-model.md) retains the complete Main
expert source on Host even when a subset is cached on GPU. PLE is disk-backed.

The qualified mixed-artifact bounded runs registered 67948068864 bytes, or 63.28157 GiB, for
Host experts alone. Even a full 64 GiB of physical RAM leaves only 0.71843 GiB before the OS,
ordinary buffers, context, workers and staging. A smaller Device budget or GPU expert cache
does not subtract that full Host source allocation. Swap does not supply the physical RAM
needed for the complete registered backing.

A 32 GB GPU / 64 GB RAM machine therefore cannot safely run those current full-model Flash
acceptance cases under this placement contract. Mark them BLOCKED by hardware, not PASS or a
product-test failure. Continue fitting dense-model tests and affected mathematical/CPU/build
checks; record their actual memory fit. Keep the Flash gap open until a sufficiently provisioned
physical 5090 host runs it or the maintainer explicitly changes the merge acceptance scope.
G4 budget simulations retain their own evidence but do not fill that physical-device gap.

The upstream dense proposal's weight/KV arithmetic is a design estimate, not proof of 5090 fit.
Plan weights, Host backing, KV/state, workspace, graphs, media, worker CPU cost and free desktop
headroom together. Qualify RTX 5090 and the 96 GB PRO 6000 separately under native Windows and
Linux; no whole-GPU or whole-RAM assumptions.
