# Flash-Next v3 port (M3): phase 0 feasibility, design and plan

Date: 2026-09-29. Base: `workstation` at `28c40898` (v3). Source of the port: `research/qwen4-flash-next`
at `87812bc8` (v2). This is a temporary plan. Remove it when M3 closes; stable content moves to the
maintainer references it names.

Scope of M3: Qwen3.8-Flash-Next (transformers `Qwen4ExpForConditionalGeneration`,
text `model_type = qwen4_exp_text`) becomes a second v3 architecture package beside
`src/models/qwen3_5`, with a v3 artifact. All development and qualification runs on Colab G4. The
on-site RTX PRO 6000 (x870e) only receives the finished build and artifact.

## 1. Measured G4 facts (session of 2026-09-29, 19:42-20:08 UTC)

| Item | Measured |
|---|---|
| GPU | RTX PRO 6000 Blackwell **Server** Edition, 97,887 MiB, driver 580.82.07 (CUDA 13.0 driver API) |
| Toolkit | image ships CUDA 12.8 only; `apt-get install cuda-toolkit-13-3` plus build deps took **5.8 min** |
| CPU / RAM | AMD EPYC 9B45, 48 threads; **176 GiB** RAM (174 GiB available idle), no swap; `/dev/shm` 87 GiB |
| Disk | one 236 GiB overlay, **189 GiB free** at start (48 GiB image) |
| Disk speed | direct write 2.0 GB/s; cold artifact reads 0.31-0.45 GB/s (see below); page-cached reads ~11 GB/s |
| Pinned host | 30 GiB `cudaHostAlloc` + touch in 6.5 s; H2D 33.9 GB/s. 30 GiB pinned PLE fits with >100 GiB to spare |
| HF download, v2 artifact | `igorls/Qwen3.8-Flash-Next-mixed-NInfer@5f0ee7e2`: `aria2c -x16 -s16` **113.30 GB in 376.7 s = 300.7 MB/s (6.3 min)**. A single `curl` range stream ran 248 MB/s. `hf download` (xet, unauthenticated) stalled at 13.9 GB after a ~230 MB/s burst and was abandoned |
| Verification | `sha256sum -c SHA256SUMS`: **OK** (`3d383e51...0d1d02`), 363.8 s while a build ran |
| Old engine build | `research/qwen4-flash-next@87812bc8` builds on Linux (GCC 13.3, nvcc 13.3): `ninfer`, `ninfer-serve`, `ninfer-perplexity` in **~5.5 min** on 48 threads. The target-private `ninfer_qwen3_8_flash_next_reference` tool has bit-rotted (2 compile errors in `main.cpp:1730,1770`) |
| Old engine run | loads the downloaded v2 artifact and generates. Cold start 174.3 s (weights-materialize 167.5 s for 75.17 GB = 449 MB/s, PLE warm 5.3 s); warm start 12.6 s |
| Baseline, greedy | 63-token prompt, 115 tokens: prefill 1,214 tok/s, **decode 135.5 tok/s**, 70.01 GiB weights |
| Baseline, `--spec mtp --draft-tokens 3` | **decode 196.0 tok/s**, acceptance 64.96% (2.95 tok/round, by position 34/28/14); text identical to the greedy run |
| Disk after CUDA + clone + build + v2 artifact | 163 GiB used, 74 GiB free, so toolkit + build tree is ~9.5 GiB |
| Google Drive | `colab drivemount` needs an interactive OAuth consent in a browser for every session. It cannot be an unattended mirror |

The G4 is the Server Edition; x870e is the Workstation Edition. Compare performance only inside
one G4 session (same card, same clocks), never across machines.

## 2. Feasibility: yes

The whole port can be built and qualified on G4 without touching x870e.

- **Disk.** One 105.5 GiB artifact plus toolkit and build (~10 GiB) fits (116 of 189 GiB). The v2
  input and a separate v3 output do not fit together (211 GiB). The upgrade therefore runs in place
  with `fallocate --punch-hole`, which works on the Colab overlay (tested). Each copied chunk of
  the v2 file is released, so the peak is ~105.5 GiB plus one chunk.
- **Host RAM.** 176 GiB holds 30 GiB of PLE (mapped or pinned), the 4.7 GiB BF16 MTP banks if
  they were still host-read, and the page cache the loader needs.
- **Time.** The critical path of a fresh session is about 12 minutes: CUDA install then build
  (11.3 min), in parallel with download then upgrade (~6.3 + ~5 min). Cold engine start is ~3 min
  at the measured cold-read rate. That leaves 20-50 minutes of GPU work in a 37-67 minute window.
- **Cold-read caveat.** Cold reads of a just-downloaded file ran at 0.31-0.45 GB/s. The first
  engine start of a session costs ~3 minutes, and later starts ~13 s from page cache.

### Resumable G4 workflow

State never lives on the VM. Code is in git (the public `igorls/ninfer` fork, cloned by commit).
The artifact is on HF. Results are small files pulled every few minutes with `colab download`
(~10 MB/s). A reclaimed VM loses only the current step.

1. `colab sessions` first; at most 3 G4s exist across agents. Name sessions `m3-*`, stop only your
   own, and unassign a lost slot (`colab_cli Client.unassign(endpoint)`).
2. Bootstrap script (extend `E:\v3port-scratch\sync2\colab\vm_build2.py`), launched detached:
   - A: apt CUDA 13.3 + deps, clone at `COMMIT`, cmake/ninja build of the needed targets.
   - B, in parallel: `aria2c -x16 -s16 -k64M` of the pinned v2 artifact, then (for v3 arms) the
     punch-hole upgrade to v3. Run the SHA256 check once per new input revision, not every session.
3. Each experiment writes a self-describing result directory (commit, artifact id, command,
   output). Pull it after every step.
4. Fixtures produced once, such as v2 baseline tokens and top-k logprobs for the public prompt set,
   are committed to the repo. Later sessions compare against them without re-running the v2 arm.

Faster restarts need an HF write token on the VM (§8, decision 4). The qualified v3 artifact could
then be published once and downloaded directly (6.3 min instead of download + upgrade). Build
outputs could also be cached there, saving ~5.5 min. Drive cannot serve this purpose because it
needs a browser consent per session.

### If a single G4 could not hold everything

It can. The fallback reconversion still uses a streaming converter (§4.4), and PLE could
move to an SSD-backed row cache (Strata-style), but neither is needed for M3.

## 3. v3 architecture design: `src/models/qwen4_exp`

**Package name.** `qwen4_exp` matches the checkpoint's own identity (`Qwen4ExpForConditionalGeneration`,
`qwen4_exp_text`) and the vLLM/transformers module names, which is how v3 names architectures
(`qwen3_5` for `Qwen3_5*`). The registered artifact identity stays `qwen3.8-flash-next`.

### 3.1 What v3 requires of a second package

v3 has no model registry. The Engine is monomorphic:
- `runtime::ModelInstance::ModelContract = models::qwen3_5::RuntimeTypes`;
- `engine.cpp` holds `EngineCore<ModelInstance>` and `CausalScoreCore<ModelInstance>`;
- `construct_model()` calls the `qwen3_5` load/plan functions directly;
- `models/registry.h` only resolves two architecture strings.

The Engine is generic through templates, though. `EngineCore<Instance>`, `ResourceManager<ModelContract>`
and `RequestRecord<ModelContract>` only need a `ModelContract` bundle of about 32 types
(`qwen3_5/program/runtime_types.h`).

Design of the seam (the "closed Engine registry" that `qwen3_5/program/program.h:140` refers to
but that does not exist yet):

- `models/registry.h`: add `Architecture::Qwen4Exp`, resolved from `Qwen4ExpForCausalLM` /
  `qwen4_exp_text` in the artifact's `text` component config. This mirrors `qwen3_5`: the HF
  checkpoint's top-level class is `...ForConditionalGeneration`, while the v3 `text` component
  records the text-only `...ForCausalLM` class. The upgrader (§4.2) writes exactly this pair, so
  the registry and the artifact agree by construction.
- `runtime/engine/model_instance.*`: turn `ModelInstance` into one struct per package
  (`qwen3_5::Instance`, `qwen4_exp::Instance`), each with its own `ModelContract`.
  `construct_model` reads the architecture from the directory, then dispatches to that package's
  `plan_load / materialize_model / make_sequence_planner / create_program`. The capacity
  resolution between them (`resolve_kv_capacity`, `reconcile_concurrency`) stays shared.
- `engine.cpp`: `Engine::Impl` holds
  `std::variant<EngineCore<qwen3_5::Instance>, EngineCore<qwen4_exp::Instance>>` (and the same
  for `CausalScoreCore`). It is chosen once at construction, with no per-call runtime branching
  beyond one `std::visit` at the public entry points. Programs share no state.
- **Frontend is reused, not duplicated.** Flash-Next has the same tokenizer family, chat-template
  mechanics, MRoPE `[11,11,10]` prompt layout and the same 27/1152/4304/16 vision tower
  (merger output 2560). `qwen4_exp` therefore aliases `qwen3_5::Frontend`, `PreparedPrompt` and
  `OutputSession`, which keeps `engine.cpp`'s prompt path unchanged. The one coupling is
  `parse_resources(FrontendResources&, const qwen3_5::Config&)`, which reads only `vocab_size`,
  the vision patch geometry and a DFlash2 top-k. That becomes a small `FrontendGeometry` value
  both packages fill. `default_sampling(Architecture)` gains the Qwen4Exp entry.
- **Program types are package-owned.** The contract types (`SequencePlan`, `ResourcePlan`,
  `PendingBatch`, the handles, `PressurePlanningSession`, ...) are PIMPL wrappers declared in
  `qwen3_5/program/program.h` over `qwen3_5::detail` impls. `qwen4_exp` declares its own set with
  the same Engine-facing surface over its own impls. This duplicates about 900 lines of
  declarations, but it keeps upstream's `qwen3_5/program` untouched. Promoting the value handles to
  `runtime/contract` would be cleaner, but every upstream merge would then conflict in
  `program.h`, so it is not proposed now.

**Upstream merge-conflict surface** (Igor syncs by merging `origin/master`). Upstream files that M3
edits: `src/models/registry.{h,cpp}`, `src/models/CMakeLists.txt`,
`src/runtime/engine/{model_instance.h,model_instance.cpp,engine.cpp}`,
`src/models/qwen3_5/frontend/{resources.*,frontend.h}` (geometry seam),
`src/artifact/{formats,layouts,materializer,binder}.*`, `src/core/weight.h` (new QTypes),
`tools/artifact/{formats,layouts}.py` + `codecs/`, `tools/upgrade_ninfer_v2_to_v3.py`,
`src/ops/CMakeLists.txt` and the linear/rmsnorm_rope shape registries,
`docs/maintainer/{artifact-container,storage-layouts,tensor-formats}.md`. Everything else is new
directories (`src/models/qwen4_exp`, new `src/ops/*` families, `tests/models/qwen4_exp`,
`tests/ops/*`, `tools/convert/qwen4_exp*`).

### 3.2 Component map, v2 → v3

Sizes are v2 lines and an estimate of the v3 code to write or adapt (excluding tests).

| v2 component (`src/targets/qwen3_8_flash_next/...`) | v3 home | Shared or private | v2 lines | v3 est. |
|---|---|---|---:|---:|
| `package.cpp`, `export/.../package.h`, `runtime.h` | `qwen4_exp/{model,config,weights}.*`, `program/runtime_types.h`, `program/program.h` | private | 1,220 | 1,600 |
| `load/{bindings,loader,materialized}`, `expert_bank`, `model_view.h` | `qwen4_exp/load/*`, `load.cpp` on v3 `Binder`/`MaterializationPlan`/`WeightView` | private | 1,470 | 1,200 |
| `load/quantize_nvfp4_expert_bank.cu` (MTP banks at load) | **removed**; baked offline by the upgrader (§4.3); v3 forbids runtime repacking | converter | 262 | (moves to tool) |
| `load/quantize_output_head.cu` (FP8 head/embedding flags) | **removed**; a later artifact recipe choice if ever wanted (the flags added VRAM) | n/a | 334 | 0 |
| `moe*.{h,cpp,cu}`, `moe_route.cu`, `moe_shared_kernels.cu` | new Op family `src/ops/sparse_moe/nvfp4_e512_k10` (route + NVFP4 expert-bank gate_up/down + BF16 shared expert); v3 `sparse_moe` is closed to the 35B Q4/Q8 geometry | Op | 3,500 | 3,300 |
| `qsa_attention*` + `ops/softmax_attention/selected_block/*` + `include/ninfer/ops/selected_block_attention.h` (Astra's split decode, bb7b7305) | re-add the `selected_block_attention` Op (decode split + prefill MMA) and a `qsa_attention` execution file | Op + private glue | 1,760 | 1,700 |
| `qsa_indexer*` | new Op `src/ops/qsa_indexer` (block-key plane store, scores, top-k 512 blocks) | Op | 1,080 | 1,050 |
| `gdn*` | `qwen4_exp/execution/gdn.cpp` over the existing v3 `gated_delta_net`, `causal_conv1d_silu`, `gdn_gating` Ops (same core dims as the 27B: Hqk 16, Hv 48, d 128, FP32 state) | shared Ops, private glue | 490 | 400 |
| `hyper_connection*` | new Op `src/ops/hyper_connection` (norm/low-rank/mix, prepare/inject) | Op | 940 | 900 |
| `ple_{table,index,pipeline,decode}*` | new Op `src/ops/ple_ngram` (dequant + gated combine + dilated conv); host gather pipeline in `qwen4_exp/execution/ple*` over `core/host_worker_pool` | Op + private | 1,160 | 1,200 |
| `mtp_forward*` | `qwen4_exp/execution/mtp.cpp`, reusing `speculative_round`, `argmax`, `rmsnorm` | private | 530 | 550 |
| `vision_execute`, `vision_adapter.h` | reuse `qwen3_5/execution/vision.*` + `program/vision_control.h` with merger output 2560 | shared | 190 | 150 |
| `text_executor*`, `text_decode*` | `qwen4_exp/execution/text.cpp` (layer composition, prefill/decode, CUDA-graph decode) | private | 2,070 | 2,200 |
| `program.cpp`, `program_impl.h`, `runtime_plan*`, `runtime_state*`, `lane_ledger*`, `stage_ledger*`, `state_dumper.h` | `qwen4_exp/program/*` + `qwen4_exp/state/*`: lanes, slots (`slots_per_lane = draft+1`), KV + indexer planes, GDN/PLE/MTP state, checkpoints, pressure/capture, transactions against the v3 contract | private | 7,000 | 8,000 |
| frontend (`qwen3_6::make_frontend`) | `qwen3_5::Frontend` via the geometry seam | shared | 20 | 150 |
| Target registry entry (`src/targets/registry.*`) | Engine variant + `construct_model` dispatch (§3.1) | runtime | ~60 | 400 |
| FP8 F32-row-scale linear (`fp8_f32_a8*`, dispatch edits), Flash-Next `[N,K]` instances | `src/ops/linear/fp8` gains the `fp8_e4m3fn_row_fp32` codec path and the problems `[13312,2560]`, `[16384,2560]`, `[2560,6144]`; BF16 `[640,2560]`, `[10240,2560]`, `[2560,2560]`, `[248320,2560]`; `rmsnorm_rope` gains the 24/2 head geometry | Op | ~900 | 1,000 |

Total new or adapted engine code: about 22K lines, plus about 27K lines of v2 tests to port
selectively (`tests/targets/qwen3_8_flash_next`, 35 files).

**Ops that already exist on `workstation` and are reused unchanged:** `gated_delta_net`
(+ batch update), `causal_conv1d_silu`, `gdn_gating`, `rmsnorm`, `rope`/MRoPE, `embedding`
(needs the 2560 instance), `sampling`, `argmax`, `candidate_logprobs`, `speculative_round`,
`scatter`, `residual_add`, `kv_cache_append` (paged), and all vision Ops.

**Must be ported from the old branch:** `selected_block_attention`, QSA indexer, NVFP4 E512/top-10
MoE (route, expert banks, shared expert), hyper-connection, PLE n-gram decode, the FP8 F32-scale
linear path, and the Flash-Next linear/rope shape instances.

### 3.3 v3 APIs the old code relied on that changed

| Area | v2 | v3 | Port action |
|---|---|---|---|
| Container | `NINFER\0\2`, one file, mapped reader | `NINFER\0\3`, ≤32 GB parts, objects may straddle parts, no mmap | new artifact (§4) |
| Formats/layouts | `U4Z8G16_F16S`/`packed-u4-g16-v1`, `FP8_E4M3FN_ROW_F32S`/`row-scale-f32-v1`, `NVFP4`/`expert-blockscale-k16-m128x4-v1`, `I64` | none of these exist | add four generic registrations (§4.1) |
| Binder | `require_tensor(name, format, layout, shape)`, `retain_mapped_tensor` | `parameter(name, Shape, Residency, QType)`, `binding`, `use`, `resource`, `finish()`; `Residency {Device, Host, Values}`, Host = heap copy | rewrite load; add **`Residency::Mapped`** (read-only file mapping per object, with a copy fallback for the rare object that straddles two part files) so the 30 GiB PLE stays page-cache backed as in v2 |
| Materializer | `device_data(handle)`, `mapped_tensor_bytes`, host warm-up | `device_parent`/`host_parent` → `WeightParent`; no warm-up | add mapped parents + explicit warm-up before readiness |
| Typed helpers | `typed_binding.h`, `core/host_memory.h`, `QType` in `core/tensor.h` | gone; `core/weight.h` (`WeightView`, `native_weight`), `ops/weight_input.h` | mechanical |
| Registry | `ActiveTarget` variant, `construct_target` | gone; single `ModelInstance` | Engine variant (§3.1) |
| Frontend | `qwen3_6::make_frontend`, `take/bind_frontend_resources` | `qwen3_5/frontend`, `parse_resources(Config)` | geometry seam |
| Vision | `qwen3_vision::Encoder`, `qwen3_6/vision_control.h` | `qwen3_5/execution/vision.h`, `program/vision_control.h` | adapt |
| Runtime contract | `runtime/contract/types.h`, `resource_manager.h`, `context_cost.h`, `sampling_history.h` | split `contract/{request,execution,resources,timing}.h`; `context_cache/`; `prompt_token_presence` inline in `qwen3_5` | all ~45 `runtime::` symbols Flash-Next uses still exist; the pressure/capture/materialization protocol is richer (`start_resource_transaction`, `progress_context_transaction`, `prove_persistent_backfill`, capture-pressure plans), so the Program port is the largest item |
| Linear internals | `Nvfp4Gemv*`, `Bf16Mma*`, `bf16_gemm_mma_kernel` | renamed/restructured (`nvfp4_a16_gemv.cuh`, `nvfp4_a4_mma.cuh`, `bf16_a16_mma.cuh`, `bf16_schedule.cuh`) | rewrite the MoE/hyper kernels' includes against the new internals |
| `ops/common/warp.cuh::block_reduce_sum` | ended with `__syncthreads()` | no trailing barrier | **audit** the 3 uses in the hyper kernels and 5 in the PLE kernels for shared-memory reuse races |
| `ninfer/types.h` flags | `quantize_output_head_fp8`, `quantize_token_embedding_fp8`, `use_qsa_prefill_mma`, `gdn_state_storage` | removed | drop; QSA prefill uses the MMA route unconditionally; GDN state FP32 |
| `speculative_options.h` | Flash-Next rule `(draft+1)*concurrency ≤ 64` slots | fixed `[1,5]` draft check | validate in the `qwen4_exp` planner (slots are a Program fact) |

## 4. v3 artifact and converter plan

### 4.1 Four generic container registrations

These are exact re-statements of the v2 definitions, so payload bytes are preserved:

| v3 format | v3 layout | v2 origin | Objects |
|---|---|---|---:|
| `fp8_e4m3fn_row_fp32` | `row_scale_fp32_v1` (code plane, 256-align, FP32 row multipliers) | `FP8_E4M3FN_ROW_F32S` / `row-scale-f32-v1` | 96 |
| `nvfp4` (existing codec) | `expert_block_scale_k16_m128x4_v1` (rank 3 `[E,N,K]`: all codes, then per-expert swizzled scales, then E FP32 divisors) | `expert-blockscale-k16-m128x4-v1` | 96 (+2 MTP, §4.3) |
| `u4z8_g16_fp16` | `packed_u4_g16_v1` | `U4Z8G16_F16S` / `packed-u4-g16-v1` | 128 |
| `int64` | `contiguous_le_v1` | `I64` | 3 |

A v3 per-expert split is not possible without moving bytes, because the v2 bank interleaves planes
across experts. The rank-3 layout is required.

Each needs the matching entries in `src/artifact/{formats,layouts}.cpp`, `core/weight.h`,
`tools/artifact/{formats,layouts}.py` + a codec, and `storage-layouts.md` / `tensor-formats.md`.

### 4.2 Recommended route: upgrade the published v2 artifact

Extend `tools/upgrade_ninfer_v2_to_v3.py` with a Flash-Next branch: known identity
`("qwen3.8-flash-next", "mixed-nvfp4-fp8-ple-int4")` with 1,566 objects. It remains
standard-library only. The existing upgrader already works this way: the v3 payload is the v2
payload copied byte-for-byte at the same offsets, with a new directory, and the replacement chat
template appended. Flash-Next adds:

- **Directory synthesis.**
  - `components.text.config` is the real `qwen4_exp_text` config from the pinned
    `config.json`, not a stub. Its `architectures` is `["Qwen4ExpForCausalLM"]` and its
    `model_type` is `qwen4_exp_text`, the pair §3.1 resolves.
  - `components.vision` has the 27/1152/4304/16 config (merger output 2560).
  - `components.mtp` is `{architectures: ["Qwen4ExpMTP"], target: "text"}`.
  - Resources are the six frontend files.
- **Bindings.**
  - Bindings are 1:1 with the stored parents. The Flash-Next kernels consume the fused parents
    (`attention/query_gate_key_value [13312,2560]`, GDN `[q,k,v,z] [16384,2560]`,
    `[A,B] [96,2560]`, expert banks `[512,1280,2560]` / `[512,2560,640]`, PLE shards
    `[2500012,160]`) directly, so no Part splitting is needed.
  - `mtp/layer/` is renamed to `mtp/layers/0/` (v3 convention).
  - Uses carry `AllowA4` (NVFP4), `AllowA8` (FP8) and `A16Only` (BF16, int64, u4 PLE).
- **Streaming in place on G4.** An optional `--release-input` flag punches holes in the input
  behind the copy cursor, so the peak disk is one artifact.
- **Chat template.** The Flash-Next official template (`c3cf9e34...`, 8,952 B) differs from the
  maintained `tools/chat_templates/qwen3_8.jinja` (`c97bd026...`). The generic upgrader would
  install the latter, because the model id starts with `qwen3.8-`. See decision 1.

**Object inventory in v3 terms** (with MTP baked):

| Format | Count |
|---|---:|
| `bf16` | 1,235 |
| `nvfp4` | 98 |
| `fp8_e4m3fn_row_fp32` | 96 |
| `u4z8_g16_fp16` | 128 |
| `int64` | 3 |
| resources | 6 |
| **objects** | **1,566** |

About 1,560 bindings. The file is about 102.2 GiB (105.52 − 4.69 + 1.32), split into 4 files at
the 32 GB default. The device weights remain ~71 GiB.

### 4.3 MTP expert banks: bake NVFP4 offline

In v2 the two BF16 MTP banks (4.69 GiB, the last two payload objects at offsets 107.26-112.30 GB)
are quantized to NVFP4 at every engine start. v3's contract forbids runtime weight repacking, so
the upgrader replaces them with two NVFP4 `expert_block_scale` objects. All other objects keep
their offsets because the banks are the payload tail.

The quantizer must produce bytes identical to the v2 loader's device buffers, so v3 MTP matches
the v2 engine exactly. Two options:
- a Python/numpy transliteration of `quantize_nvfp4_expert_bank.cu`;
- the existing `ninfer_quantize_mtp` / `splice_mtp.py` from the old branch, run once on a G4.

The oracle is byte equality with the v2 loader's device banks, dumped from the old engine on G4.

The v3 file's SHA256 then differs from the published v2 file; the other 1,564 objects are
byte-identical. If Igor prefers byte-identical-plus-template, the alternative is keeping BF16
banks with a v3 load-time quantizer. That violates the v3 rule and costs 4.7 GiB of host reads per
start, so it is not recommended.

### 4.4 Fallback: reconvert from the pinned primitive-ai sources

Port `tools/convert/qwen3_8_flash_next/{source,recipe,inventory,convert}.py` (~1,700 lines) onto
the v3 writer as an official recipe `qwen3_8_flash_next_mixed`. Inputs:
- `primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8@a4e813ed` without `ple-bf16-*`: 75.1 GiB of
  171.2 GiB, 48 `ct-experts-layerNN` files of 1.32 GiB each, `carry-model-bf16-*`, `fp8-tail-00`.
- `primitive-ai/Qwen3.8-Flash-Next-PLE-quant@da8b3958` `ples_int4/*`: 29.8 GiB in 128 shards.
- The six official frontend resources.

Sources plus output (~210 GiB) do not fit on a G4, so the converter must stream: fetch one source
file, emit its objects, then delete it. The peak is then the output plus ~2 GiB. At ~300 MB/s this
is ~6 min of download plus conversion. That rate is extrapolated, not measured on the primitive-ai
repositories: it was measured on `igorls/...-NInfer`, which sits behind the same HF xet CDN. Only
the primitive-ai file listing (sizes) was fetched on the G4.

Verification: every emitted object must be byte-identical to the corresponding object of the
upgraded artifact, except the MTP banks and template when the recipe chooses them. This makes the
two routes cross-check each other. Use this route only if the v2 artifact is ever unavailable, or
when a new source revision is adopted.

### 4.5 Verifying the upgraded artifact

1. **Exact payload identity.** Stream-hash every v3 object range and compare it with the same
   range in the v2 file. This covers all 1,564 unchanged objects and the six resources.
2. **MTP banks.** Byte equality with the v2 loader's device buffers (§4.3). Also decode the
   NVFP4 banks against an FP64 oracle from the BF16 source to confirm rounding semantics.
3. **Container validation.** `python -m tools.artifact.inspect` and the C++ `Reader::validate_object`
   for every object. The `qwen4_exp` binder consumes every object exactly once and fails on any
   missing or extra object.
4. **End to end.** Parity against the v2 engine (§5).

**Time on G4** (upgrade route): download 6.3 min; SHA check ~6 min, only once per input revision;
in-place copy ~3-5 min (2 GB/s writes, reads partly from page cache); MTP bake <1 min on GPU.
Total ~10-12 min, peak disk ~106 GiB.

**x870e deployment.** The same tool upgrades Igor's local v2 copy on Windows, with no 105 GB
transfer. Windows has no punch-hole in the script; it needs 102 GiB free beside the input, or
`--release-input` implemented with `FSCTL_SET_ZERO_DATA` on a sparse file. The output SHA256 is
compared against the G4-produced v3 artifact.

## 5. Qualification plan on G4

**Public data only:** public prompt sets (the `eval/kld` corpus, public chat and code prompts,
public images), and synthetic cases.

**1. Op oracles.**
- **Scope.** Every model-private and newly ported Op gets an independent FP64 oracle at the real
  shapes, per `docs/maintainer/op-development.md`: `selected_block_attention` (BF16/FP8 KV,
  B 1/2/5/8, empty sets, graph replay), QSA indexer (block keys, scores, top-512 selection, tail),
  NVFP4 MoE (route softmax/top-10/renorm, gate_up/down banks, shared expert), hyper-connection
  (prepare/mix/inject), PLE n-gram (u4 dequant, gated combine, dilated conv, history reset on EOS),
  FP8 F32-scale linear at every registered `[N,K]`, and the 24/2 `rmsnorm_rope`.
- **Starting point.** The v2 tests (`test_moe*`, `test_qsa_*`, `test_hyper_connection`,
  `test_ple_*`, `test_fp8_f32`, `test_gdn`) already hold these oracles; port them to
  `tests/ops/*`.
- **The `block_reduce_sum` barrier change** gets a test that fails if the race exists.

**2. Artifact.**
- §4.5 checks 1-3.
- Load-only test: materialize on G4 and checksum the device weights against the v2 engine's
  materialized weights.

**3. Engine parity against v2.**
- **v2 build.** `research/qwen4-flash-next@87812bc8` on Linux. Measured working; only the
  reference tool needs a 2-line fix.
- **Prompt set.** A fixed public set of 24-48 prompts: short and long (≥18K), thinking on/off,
  tool call, JSON-schema structured output, 3 image prompts.
- **Greedy token parity.** Both engines at `--greedy`. Criterion: identical sequences. Any
  divergence must be at a near-tie, meaning the top-2 logprob gap at the first divergent position
  is < 1e-3 in v2.
- **Teacher-forced logprob parity.** The same corpus through prompt-logprob scoring
  (`/v1/score` or prompt logprobs, both lines have it) with top-20. Criteria: per-position top-1
  agreement ≥ 99.9% and mean KL(v2‖v3) ≤ 1e-4. Ported kernels are expected to be bit-close. A
  looser result must be explained by a named kernel-internals change, such as the new linear
  internals.
- **Baseline fixtures.** The v2 arm runs once. Its tokens and top-20 logprobs are committed as
  fixtures (a few MB), so later sessions need only the v3 arm and one artifact on disk.

**4. MTP.**
- Under greedy, the drafted and accepted token streams must equal v2's, not only the acceptance
  rate.
- Acceptance histogram by position within ±2 pt of the v2 arm on the same prompts.
- Baseline measured today: 64.96% at K=3 on a short prompt, decode 135.5 → 196.0 tok/s.
- Concurrency ≥ 4 with MTP on: a regression test for the `cache_slot = 2 * max_concurrency + c`
  aliasing bug class, which asserts `cache_slot base ≥ slots_per_lane * max_concurrency`.

**5. Vision.**
- Image and video prompts: token parity with v2 on the prompt set.
- The VRAM delta with `--vision` matches the envelope (~1.06-1.24 GiB); no second vision buffer.

**6. Prefix reuse and pressure.**
- Port the saturated-pool acceptance scripts: catalog exhaustion, depth-1 retention/newcomer
  thrash, and the thinking-toggle and tool-order probes (`tools/bench/prefix_reuse_probes`).
- On v3 these exercise the Engine's `ResourceManager` against the Flash-Next Program's physical
  slots. Each owner holds two physical checkpoint slots, so admission must see checkpoint-slot
  pressure. This was the v2 catalog-exhaustion defect; see risk 4.

**7. Performance.**
- **Harness recovery.** The eight-rule harness (`README-harness.md`, `night/id_probe.py`,
  `round_timing.py`) survives only in an old session scratchpad, and `E:\NInfer\bench-harness` no
  longer exists. Move it into `tools/bench/flash_next/` first.
- **Probe.** The 200-token greedy probe: 8 lanes × 3 reps, sequential then 8-way. Engine
  per-round timing comes from `--request-log-jsonl`.
- **Workloads.** 30K-context decode (split-attention), prefill at chunk 8192, and MTP K=3/4.
- **Report together:** single-stream tok/s, per-stream tok/s at C=8, and peak VRAM.
- **Method.** A/B v2 against v3 in the same G4 session. Target: v3 ≥ v2 on every row.

**8. x870e confirmation** (after G4 sign-off, with Igor's go): upgrade the local copy, compare its
SHA with the G4 v3 artifact, and run one short confirmation run in a window Igor approves. No
spare-port instances.

## 6. Milestones (implementation branches from `workstation` after `sync/upstream-d44ab584` lands)

| # | Milestone | Size (new/adapted lines, excluding tests) | G4 verification |
|---|---|---:|---|
| M3.1 | Container registrations (§4.1) + Flash-Next upgrader + MTP NVFP4 bake + docs | ~1,800 (+ ~600 tests) | download v2 → upgrade in place; §4.5 checks 1-3; MTP bytes equal the v2 loader dump |
| M3.2 | `qwen4_exp` skeleton: config, binder/load with `Residency::Mapped`, Engine variant seam, frontend geometry seam, `Architecture::Qwen4Exp` | ~3,000 | artifact loads through the public Engine; device-weight checksums equal v2's; qwen3_5 CTest unchanged |
| M3.3 | Op ports with FP64 oracles: selected-block attention, QSA indexer, NVFP4 E512/K10 MoE, hyper-connection, PLE n-gram, FP8-F32 linear + Flash-Next shapes, rmsnorm_rope 24/2; `block_reduce_sum` audit | ~8,000 (+ ~6,000 tests) | `ctest -R ops` on G4 |
| M3.4 | Text execution + Program on the v3 contract: prefill/decode, KV + indexer + GDN + PLE state, checkpoints/continuations/pressure, CUDA-graph decode, logprobs, structured output | ~10,000 (+ ~8,000 tests) | greedy + teacher-forced parity (§5.3); continuation/prefix tests; pressure scenarios |
| M3.5 | MTP + Vision | ~1,500 (+ ~1,500 tests) | §5.4, §5.5 |
| M3.6 | CLI/serve options, harness into `tools/bench/flash_next`, performance A/B, VRAM envelope, docs (`qwen3.8-flash-next-{artifact,model}.md` rewritten for v3, `upstream-ports.md`, `performance.md`, AGENTS product line), model card for the v3 artifact | ~1,000 | §5.6, §5.7; then the x870e confirmation (§5.8) |

M3.3 and the load half of M3.2 can proceed in parallel once M3.1 lands. M3.4 is the critical path.
Each milestone is its own branch, then fast-forwarded into `workstation` after its G4 checks,
following the one-worktree-per-issue rule.

## 7. Risks

1. **Program port size.** The v2 Program targeted a thinner v2 contract. The v3 contract adds
   resource transactions, persistent-backfill proofs and capture-pressure planning. Budget M3.4 as
   the largest milestone. Port behaviour, not structure.
2. **New linear internals.** Rebasing the MoE and hyper kernels onto the renamed NVFP4/BF16
   internals can change reduction order. Parity criteria (§5.3) allow only near-tie divergences and
   bounded KL, and FP64 oracles judge each Op.
3. **`block_reduce_sum` lost its trailing barrier.** This is a silent shared-memory race in
   ported kernels unless audited.
4. **Checkpoint-slot accounting.** The v3 `ResourceManager` owns the logical catalog, while
   Flash-Next's owners pair two physical slots. If the Program does not report that physical
   pressure at admission, the v2 catalog-exhaustion defect (0% reuse forever after ~8 owners)
   comes back.
5. **Mapped PLE on Windows.** x870e has 125.7 GiB RAM (46.9 GiB free at the time of writing),
   next to production and the desktop. Page-cache-backed PLE can be evicted under pressure and
   stall prefill. Pinning 30 GiB is safer for latency but takes the RAM from the desktop. M3 keeps
   v2's mapped behaviour. The SSD row cache (Strata `ple_reader.cpp` design) belongs to the
   hot-swap phase.
6. **G4 reclaim mid-step.** Mitigated by the stateless workflow (§2). The first engine start per
   session costs ~3 min of cold reads.
7. **Upstream drift during M3.** Every upstream merge may touch the seam files (§3.1). Keep the
   seam small and merge upstream between milestones, not inside one.
8. **Old reference tool and oracle environment.** The C++ reference runner does not compile at
   `87812bc8`. The vLLM-faithful Python oracle lived in `E:\NInfer\venv-qwen4exp`, which is gone.
   Rebuild the environment on a G4 from `tools/reference/qwen3_8_flash_next/oracle/README.md` if
   an end-to-end oracle beyond v2-engine parity is needed.

## 8. Decisions for Igor

1. **Chat template in the v3 artifact.**
   - Recommended: keep Flash-Next's official template (`c3cf9e34`). This gives parity with v2
     now; switch to a maintained template later as a separate, measured change.
   - Alternative: install `qwen3_8.jinja`, as the generic upgrader would.
2. **MTP banks.** Bake NVFP4 into the v3 artifact (recommended, required by v3's
   no-runtime-repacking rule; the SHA then differs from the published v2), or keep BF16 with a
   load-time quantizer.
3. **Drop the v2-only runtime flags.** These are the FP8 output head and FP8 embedding (both
   measured to add VRAM), the BF16 GDN state, and the QSA MMA switch. Recommended: yes.
4. **HF write token on Colab.** It allows publishing the qualified v3 artifact (and cached build
   outputs) once, so each session starts ~6 min faster. Without a token, every session re-derives
   the v3 artifact from the v2 download (~10-12 min, fully automatic). The v3 artifact could also be
   published from x870e after the local upgrade.
5. **Accept the upstream conflict surface of §3.1** (Engine variant, registry, frontend geometry
   seam, container registrations), or ask to propose these seams upstream first.
6. **PLE residency for M3:** mapped page cache (v2 behaviour, recommended) or pinned host memory.
