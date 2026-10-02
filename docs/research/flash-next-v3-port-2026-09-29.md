# Flash-Next v3 port (M3): phase 0 feasibility, design and plan

Date: 2026-09-29. Base: `workstation` at `28c40898` (v3). Source of the port: `research/qwen4-flash-next`
at `87812bc8` (v2). This is a temporary plan. Remove it when M3 closes; stable content moves to the
maintainer references it names.

Scope of M3: Qwen3.8-Flash-Next (transformers `Qwen4ExpForConditionalGeneration`,
text `model_type = qwen4_exp_text`) becomes a second v3 architecture package beside
`src/models/qwen3_5`, with a v3 artifact. All development and qualification runs on Colab G4. The
on-site RTX PRO 6000 Workstation Edition only receives the finished build and artifact.

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

The G4 is the Server Edition; the on-site card is the Workstation Edition. Compare performance only inside
one G4 session (same card, same clocks), never across machines.

## 2. Feasibility: yes

The whole port can be built and qualified on G4 without touching the workstation.

- **Disk.** One 105.5 GiB artifact plus toolkit and build (~10 GiB) fits (116 of 189 GiB). The v2
  input and a separate v3 output do not fit together (211 GiB). The upgrade therefore runs in place
  with `fallocate --punch-hole`, which works on the Colab overlay (tested). Each copied chunk of
  the v2 file is released, so the peak is ~105.5 GiB plus one chunk.
- **Host RAM.** 176 GiB holds 30 GiB of PLE (mapped or pinned), the 4.7 GiB BF16 MTP banks if
  they were still host-read, and the page cache the loader needs.
- **Time.** The critical path of a fresh session is about 13 minutes: CUDA install then build
  (11.3 min), in parallel with download then upgrade (6.5 + 6.7 min measured in M3.1). Cold engine start is ~3 min
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
   - B, in parallel: `python -m tools.convert.qwen4_exp.derive` (pinned v2 download, then the
     punch-hole upgrade to v3, which checks the v2 SHA256 during its single read).
3. Each experiment writes a self-describing result directory (commit, artifact id, command,
   output). Pull it after every step.
4. Fixtures produced once, such as v2 baseline tokens and top-k logprobs for the public prompt set,
   are committed to the repo. Later sessions compare against them without re-running the v2 arm.

There is no HF write token (§8, decision 4): each session re-derives the v3 artifact with
`python -m tools.convert.qwen4_exp.derive WORKDIR` (download with `aria2c -c`, in-place upgrade with
the v2 SHA256 checked in the same read, optional verify; an interrupted upgrade discards its
partly released input and downloads again). Drive cannot cache it because it needs a browser
consent per session.

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
`src/artifact/{formats,materializer,binder}.cpp`, `src/core/{weight.h,weight_view.cpp}` (new
QTypes/layouts; landed in M3.1), `tools/artifact/{formats,layouts,writer,tensor_output}.py` +
`codecs/`, `tools/convert/recipe.py`, `tools/upgrade_ninfer_v2_to_v3.py` (a Flash-Next redirect),
`src/ops/CMakeLists.txt` and the linear/rmsnorm_rope shape registries,
`docs/maintainer/{artifact-container,storage-layouts,tensor-formats}.md`. Everything else is new
directories (`src/models/qwen4_exp`, new `src/ops/*` families, `tests/models/qwen4_exp`,
`tests/ops/*`, `tools/convert/qwen4_exp`).

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

*Correction (M3.3):* v3 Ops are shape-registered for the 27B's 5120 width, so "unchanged" held
only after an inventory. `gdn_gating_proj` needed the `[96,2560]` parent (registered in M3.3);
`embedding` BF16 `[248320,2560]`, `rmsnorm` at D 2560/10240/256, `gated_rmsnorm`, the GDN Ops at
Hqk 16/Hv 48, `causal_conv1d_silu` at C 10240, the sampling/logprob Ops at vocab 248320,
`sigmoid_mul` and `kv_cache_append` D256/Hkv 2 were already admitted. `rope` is not needed: the
Flash-Next q/k path uses the new gated `rmsnorm_rope` form. Two gaps remain for M3.4 (§6.2).

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
| `ops/common/warp.cuh::block_reduce_sum` | ended with `__syncthreads()` | no trailing barrier | **audit** the 3 uses in the hyper kernels and 5 in the PLE kernels for shared-memory reuse races. *Resolved in M3.3 (§6.2): `warp.cuh` stays upstream's; every ported call site owns its barrier* |
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

**As built (M3.1):** the upgrade is `tools/convert/qwen4_exp` (`upgrade`, `verify`, `derive`), not
a branch of the standard-library `tools/upgrade_ninfer_v2_to_v3.py`. The MTP bake (§4.3) quantizes
2.5G BF16 values, which the repository's torch NVFP4 encoder does in ~20 s on CPU and a
standard-library script cannot do in session time; AGENTS.md places target-private inventories and
converter-side verification under `tools/convert/<target>`; and `tools/artifact/writer.py` is the
generic v3 writer, so the framing is not copied a third time. The generic upgrader rejects the
Flash-Next identity and points to `python -m tools.convert.qwen4_exp.upgrade`. The artifact
contract is recorded in `docs/maintainer/qwen3.8-flash-next-artifact.md`.

Original design, kept for the record: known identity
`("qwen3.8-flash-next", "mixed-nvfp4-fp8-ple-int4")` with 1,566 objects; the v3 payload is the v2
payload copied byte-for-byte with a new directory. Flash-Next adds:

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
  maintained `tools/chat_templates/qwen3_8.jinja` (`c97bd026...`). Decided (decision 1): the v3
  artifact keeps the official template, so the v2 resource is copied and nothing is appended.
- **One ordered read.** The single pass that copies the payload also hashes every v2 object
  range and the whole file. The whole-file digest replaces the separate `sha256sum -c` step, and
  the object digests are check 1's reference, since the in-place input no longer exists afterwards.
- **Reproducible output.** The v3 `artifact_id` is derived from the v2 digest and the directory
  (the container only recommends a fresh UUID), so G4 and the workstation produce identical files.

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

1,668 bindings (Vision `qkv` splits into three Parts, as in `qwen3_5`) and 959 Uses. The logical
payload is exactly 109,680,552,704 bytes (102.15 GiB), split into 4 files at the 32 GB default.
The device weights remain ~71 GiB.

### 4.3 MTP expert banks: bake NVFP4 offline

In v2 the two BF16 MTP banks (4.69 GiB at payload offsets 107.26-112.30 GB) are quantized to NVFP4
at every engine start. v3's contract forbids runtime weight repacking, so the upgrader replaces
them with two NVFP4 `expert_block_scale` objects.

*Correction (M3.1):* the banks are **not** the payload tail. They are v2 objects 1222 and 1223, and
342 objects (1.0 GB: the MTP attention, indexer and norms, then all of Vision) follow them. The
1,223 objects up to and including the gate/up bank keep their offsets; the down bank and the 342
objects behind it move down by the 3.62 GB the NVFP4 banks save. Check 1 compares by object id,
so the relocation costs nothing.

The quantizer must produce bytes identical to the v2 loader's device buffers, so v3 MTP matches
the v2 engine exactly. *As built:* the v2 kernel is exactly the repository's
`NVFP4_MAXABS_DIVISOR_RNE_V1` encoder (`tools/convert/quantization/nvfp4.py`) applied per expert,
so the upgrade reuses it on CPU (~20 s for both banks) and pins the digests of the result. The
oracle dump comes from `ninfer_quantize_mtp` at `87812bc8`, which calls the same
`quantize_bf16_expert_bank_to_nvfp4` on the same host-mapped BF16 bytes as `materialized.cpp:395`.

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

**M3.1 evidence (G4 session `m3-1`, 2026-09-30 01:07-01:31 UTC, commit `c7d5c6b9`).**

| Check | Result |
|---|---|
| v2 input | the upgrade's single read hashed the whole file: `3d383e51...0d1d02`, equal to `SHA256SUMS` |
| 1. payload identity | `verify` re-read all 1,566 objects through `tools.artifact.reader.Artifact`: 1,564 unchanged objects (1,558 tensors + 6 resources) equal their v2 object digests |
| 2. MTP banks | `ninfer_quantize_mtp` at `87812bc8` (the v2 loader's `quantize_bf16_expert_bank_to_nvfp4` on the host-mapped BF16 banks) dumped gate/up `9e25663a...2087b1f6` and down `fa78245b...6fb8c9f60`; the baked banks are byte-identical (`reference.equal`). FP64 oracle on experts 0/256/511: divisors and signs exact, scales within 0.50 E4M3FN step, code excess ≤ 3e-7 |
| 3. container | `python -m tools.artifact.inspect`: 1,566 objects, 1,668 bindings, 959 Uses, formats bf16 1,235 / nvfp4 98 / fp8 row fp32 96 / u4z8 128 / int64 3; C++ `ninfer_artifact_reader_test <v3>` ran `Reader::validate_object` on every object (`v3 objects=1566`) |
| v3 artifact | `artifact_id 1e7e026e9f634926ae26b80f0fc8591e`, 4 files, 109,681,105,664 bytes; entry `0de7b5f6...5fda15fb4618444343502fc798dff2282714`, part-0001 `e5d74e6a...`, part-0002 `67a30b1e...`, part-0003 `8bc02c73...` (full digests pinned in `tools/convert/qwen4_exp/source.py`) |
| regressions | `ctest -R '^ninfer_artifact|^ninfer_qwen3_5_loading_test$'` 4/4 on G4; `pytest tests/artifact tests/convert` 59 passed (Python 3.11.16, torch 2.14 CPU) |

**Time on G4** (measured): download 6.5 min (`aria2c -x16`, ~290 MB/s); in-place upgrade 7.3 min
(435 s: 81 s waiting on reads, 347 s writing, both banks baked in 5.5 s on CPU; the overlay disk
sustained ~300 MB/s reads plus ~300 MB/s writes, not the 2 GB/s of a pure write); optional
verify 2.7 min (object pass 22 s, file digests 138 s). Peak disk stayed one artifact (the
released v2 file shrank as the output grew; 161 GiB used at the end including the build trees).

**Unattended re-derivation (fresh G4 session `m3-2`, commit `8f7a0aaa`, digests pinned):**
`python -m tools.convert.qwen4_exp.derive /content/m3/work --verify --file-digests` took 15.5 min
wall: setup 7 s (Python 3.11 venv, torch CPU), download 6.5 min (392 s), upgrade 6.7 min (401 s),
verify 2.1 min (128 s). The four file digests equal `m3-1`'s, so the derivation is reproducible
byte for byte across VMs. Without `--verify`, a session has the artifact after ~13.3 min.

**Workstation upgrade.** The same tool upgrades Igor's local v2 copy on Windows, with no 105 GB
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
- **The `block_reduce_sum` barrier change** gets a test that fails if the race exists. *As built (M3.3):
  no deterministic failing test is constructible for a missing barrier; the evidence is the
  call-site audit plus `compute-sanitizer` racecheck/synccheck over every new route (§6.2).*

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

**8. Workstation confirmation** (after G4 sign-off, with Igor's go): upgrade the local copy, compare its
SHA with the G4 v3 artifact, and run one short confirmation run in a window Igor approves.

## 6. Milestones (implementation branches from `workstation` after `sync/upstream-d44ab584` lands)

| # | Milestone | Size (new/adapted lines, excluding tests) | G4 verification |
|---|---|---:|---|
| M3.1 | Container registrations (§4.1) + Flash-Next upgrader + MTP NVFP4 bake + docs | ~1,800 (+ ~600 tests) | download v2 → upgrade in place; §4.5 checks 1-3; MTP bytes equal the v2 loader dump. **Done** on `m3/1-container` (§4.5 evidence) |
| M3.2 | `qwen4_exp` skeleton: config, binder/load with `Residency::Mapped`, Engine dispatch seam, frontend geometry seam, `Architecture::Qwen4Exp` | ~3,000 (as built ~1,500) | artifact loads through the public Engine; device-weight checksums equal v2's; qwen3_5 CTest unchanged. **Done** on `m3/2-skeleton` (§6.1) |
| M3.3 | Op ports with FP64 oracles: selected-block attention, QSA indexer, NVFP4 E512/K10 MoE, hyper-connection, PLE n-gram, FP8-F32 linear + Flash-Next shapes, rmsnorm_rope 24/2; `block_reduce_sum` audit | ~8,000 (+ ~6,000 tests) (as built ~6,800 + ~2,200 tests) | `ctest -R ops` on G4. **Done** on `m3/3-ops` (§6.2) |
| M3.4 | Text execution + Program on the v3 contract: prefill/decode, KV + indexer + GDN + PLE state, checkpoints/continuations/pressure, CUDA-graph decode, logprobs, structured output | ~10,000 (+ ~8,000 tests) | greedy + teacher-forced parity (§5.3); continuation/prefix tests; pressure scenarios |
| M3.5 | MTP + Vision | ~1,500 (+ ~1,500 tests) | §5.4, §5.5 |
| M3.6 | CLI/serve options, harness into `tools/bench/flash_next`, performance A/B, VRAM envelope, docs (`qwen3.8-flash-next-{artifact,model}.md` rewritten for v3, `upstream-ports.md`, `performance.md`, AGENTS product line), model card for the v3 artifact | ~1,000 | §5.6, §5.7; then the workstation confirmation (§5.8) |

M3.3 and the load half of M3.2 can proceed in parallel once M3.1 lands.

### 6.1 M3.2 as built (`m3/2-skeleton`, 2026-09-30)

**Design decisions beyond §3.**
- **Engine seam = dispatch at construction, no variant yet.** `construct_model` resolves
  `models::resolve_architecture(reader.directory())` after `ArtifactInspect` and calls
  `construct_qwen3_5` (the old body) or `reject_qwen4_exp`. The latter runs the real load
  (`plan_load` with every Binding and Use checked, the VRAM headroom check, materialization with
  the PLE mapped and warmed, `qwen3_5::make_frontend` with `Architecture::Qwen4Exp`) and then
  throws inside `TargetFinalize` with the load facts in the message. `engine.cpp` is untouched: a
  `qwen4_exp::Instance` needs the ~32-type `ModelContract` (M3.4), and a core-less alternative
  would be a placeholder that lets `ninfer-serve` listen with a model that cannot answer.
  M3.4 replaces `reject_qwen4_exp` with instance creation and adds the `EngineCore` variant.
- **`Residency::Mapped`** (`src/artifact/{file_io,reader,binder,materializer,views}`): one
  read-only mapping per object (`FileMapping`, offset aligned down to the page size or the 64 KiB
  Windows allocation granularity; it outlives the Reader). A straddling object is owned as a copy
  (`mapped_copy_count`; the v3 artifact has exactly one straddling shard). Warm-up is v2's
  read-ahead-and-touch, inside `WeightsMaterialize` (its byte total now includes mapped bytes), so
  there is no new public `StartupPhase`. `LoadSummary` is unchanged until Flash-Next reaches it.
- **Package reuse.** `qwen4_exp` takes the Qwen3.5 weight-handle vocabulary (`WeightId`,
  `BoundWeight`, `VisionWeights`), leaf config structs (`AttentionConfig`, `RopeConfig`,
  `GdnConfig`, `MoeConfig`, `VisionConfig`) and `FrontendResources`; it owns its config parser,
  `loading::Bindings` (Device, Mapped and INT64 values, with `require_complete`), `bind_vision`
  (the Qwen3.5 one takes a `qwen3_5::TextConfig`) and a `Model` with a `PleTable`. The shard
  height (2,500,012) comes from the stored shape; the tables are checked as consecutive head
  ranges covered by the shards.
- **Geometry seam** is `FrontendGeometry{embedding_rows, vision patch, selector_top_k}`;
  `default_sampling` needed no change (Flash-Next's defaults equal the dense ones).
- **Gap (c)** is an Op entry, `ops::prepare_nvfp4_expert_bank_weight`, returning
  `Nvfp4ExpertBankWeight` (planes, E/N/K, per-expert plane bytes, policy). It rejects a stored
  activation divisor. M3.3's MoE Op consumes it.

**Conflict surface added to §3.1:** `src/artifact/{file_io,reader,binder,materializer,views}`,
`include/ninfer/ops/weight_input.h`, `src/ops/weight_input.cpp`,
`src/models/qwen3_5/load/resources.cpp`, `tests/CMakeLists.txt`,
`tests/artifact/test_materialization.cpp`, `docs/maintainer/engine-architecture.md`. Not touched:
`engine.cpp`, `model_instance.h`, `frontend.h`, `src/artifact/formats.cpp`.

**G4 evidence** (sessions `m3-2a` and `m3-2b`, 2026-09-30 08:37-09:25 UTC, commit `ee7b3d32`).

| Check | Result |
|---|---|
| v2 reference | `87812bc8` loader with Vision and MTP (scratch `ninfer_v2_dump`): 1,429 Device buffers, 77,667,520,224 bytes: a 71.02 GiB arena plus the two loader-quantized MTP banks outside it; cold load 161 s (PLE warm 56.7 s) |
| v3 load, Vision and MTP | `ninfer_qwen4_exp_loading_real_test`: 1,668 bindings and all 959 Uses consumed, 1,429 Device parents in one **72.33 GiB** arena (banks inside; the same bytes as v2's arena plus banks; the NVML delta of +72.88 GiB also counts the CUDA context), 128 mapped shards of 32,000,161,792 bytes (1 straddle copy); warm load: upload 5.2 s, PLE warm 1.8 s |
| v3 load, Text only | 1,198 bindings, 1,067 parents, **70.01 GiB** (v2: 70.01 GiB) |
| Device checksums | SHA256 of every Device buffer keyed by object: **1,429 of 1,429 equal** v2, including both MTP banks (`9e25663a...`, `fa78245b...`, the v2 loader's quantized NVFP4 buffers) |
| PLE | 128 of 128 mapped shards SHA256-equal v2's mappings; tables: 3 multipliers, 16 consecutive heads covering 320,001,446 of 320,001,536 rows; the shards are plain (unregistered) host mappings |
| Admission | every BF16 and FP8 projection Use prepared; all 98 banks admitted `AllowA4` with positive finite per-expert divisors; the Frontend renders the official template (56-token chat prompt) |
| Public Engine | `ninfer_qwen4_exp_engine_real_test` and `ninfer V3 --prompt`: `TargetPlan`, `WeightsMaterialize` and `FrontendInitialize` complete, `TargetFinalize` fails with "Qwen4ExpForCausalLM artifact 'qwen3.8-flash-next' loaded (1668 bindings, 72.33 GiB of Device weights, 29.80 GiB of mapped PLE table), but this build has no qwen4_exp execution Program; it cannot generate or score yet" |
| qwen3_5 unchanged | 27B NVFP4 (`neroued/Qwen3.8-27B-nvfp4-NInfer@f0b43ad4`): `loading_real --vision --speculative mtp` output identical on `93e76556` and the branch (1,004 parents, 21,122,608,640 bytes); CTest `artifact_*`, `qwen3_5_*`, `tool_call_parser` and `sampling_defaults` with the real artifact all pass except `qwen3_5_prefix_real_test`, which fails identically on the base ("Complete MTP checkpoint was not materialized from Host"), and `orcarouter_tokenizer` (not built) |
| Windows | MSVC 14.51 and CUDA 13.3: `ninfer`, `ninfer_engine`, both qwen4_exp tests and `ninfer_artifact_materialization_test` build with no warnings in the touched files; a scratch CPU-only program linked to the MSVC `ninfer_artifact.lib` maps a single-file fixture object (64 KiB align-down), warms it, reads it after Reader destruction and refuses to map the straddling one |

**Corrections for M3.3+.**
- G4 order is forced by disk: download v2, take every v2 reference dump, then `derive` in place,
  then run v3. Any v2 fixture a later milestone needs (M3.4 tokens and logprobs) must be produced
  in that window.
- A scratch tool that loads through the v2 `StandaloneLoadedModel` must link `ninfer_engine` (the
  Qwen3.6 frontend lives there at `87812bc8`).
- The startup log reports Device and mapped bytes in one `WeightsMaterialize` phase (the CLI
  prints "loading weights | 99.8 GiB" for a 70.01 GiB Text load plus the 29.8 GiB PLE). A separate
  public phase is a `types.h` change, left to M3.4 or M3.6 if wanted.
- v3 cold PLE warm-up is unmeasured (every G4 v3 load ran from the page cache `derive` had just
  filled); v2's cold 32 GB warm took 56.7 s. The workstation confirmation (§5.8) is the first cold
  Windows number.
- `qwen3_5_prefix_real_test` already fails on `workstation` with the 27B NVFP4 artifact; it is not
  an M3 signal.
- M3.3 inputs: expert banks arrive as `ops::Nvfp4ExpertBankWeight`; PLE shards are `WeightView`s
  over mapped `u4z8_g16_fp16` parents (`qwen4_exp::PleTable`); the shared-expert gate and up are
  two parents, so a fused SwiGLU needs a two-parent form or two projections.

**Inputs M3.1 fixed for M3.2+ (all closed in M3.2):**
- Logical names, Uses and input positions are those of
  `docs/maintainer/qwen3.8-flash-next-artifact.md` §5; the binder follows them (changing them is a
  converter change, cheap because every session re-derives).
- An NVFP4 expert bank's `WeightParent` has `weight_scale_divisor = 0`; its E divisors are the
  payload's divisor plane at `geometry.divisor_offset`. There is no native `Weight` form for it.
- `Binder::values` accepts `int64`; `HostValues::integers64()` reads the PLE tables, and
  `Residency::Mapped` holds the shards (decision 6).
- `ple_layer_ids = [2]` in the config counts from 1; the PLE weights are at 0-based layer 1.
- Expert banks carry `AllowA4` with no activation-divisor auxiliary: an A4 path scales
  activations dynamically, as v2 did (`prepare_nvfp4_expert_bank_weight`). M3.4 is the critical
  path.
Each milestone is its own branch, then fast-forwarded into `workstation` after its G4 checks,
following the one-worktree-per-issue rule.

### 6.2 M3.3 as built (`m3/3-ops`, 2026-09-30)

All M3.3 Ops live in `src/ops` behind contracts in `include/ninfer/ops/`, with the formula,
domain, effects and workspace in the header. No Op has an environment switch or a second
selectable kernel; each extent has one private route.

| Contract | Entries | Routes (private) |
|---|---|---|
| `sparse_moe.h` (NVFP4-bank profile) | `sparse_moe(x, SparseMoeNvfp4BankWeights, SparseMoeEpilogue::Store, out, ws, stream)`; capacity query over `(LinearPolicy gate_up, LinearPolicy down, min_T, max_T)` | T ≤ 8: fused router, NVFP4 A16 GEMV gate/up, per-row down. T 9-255 (or any T > 8 when a bank forbids A4): grouped A16 SIMT. T ≥ 256 with AllowA4 banks: grouped block-scaled FP4 MMA with dynamic per-16 activation scales |
| `selected_block_attention.h` | batched form (C 1..8, `PagedKVBatchLayerView` + `table_rows`, caller workspace, graph-safe); shared-row form (C 1..262144, `PagedKVLayerView`, no workspace) | tensor-core partitions + merge; KV profiles BFloat16 (BF16 K, FP16 V) and FP8-E4M3FN-row256 (the Op rotates q by the D256 Hadamard) |
| `qsa_indexer.h` | `qsa_indexer_append` (shared-row prefill form; snapshot form W ≤ 16, B ≤ 8); `qsa_indexer_select` with a `{max_complete_blocks}` envelope and a capacity query | identity route when complete ≤ 512; otherwise tiled scores + exact in-house radix top-512 (no CUB, no CCCL internals) |
| `rmsnorm_rope.h` (gated D256 form) | packed `[13312,T]` projection → q `[256,24,T]`, gate `[256,24,T]`, k `[256,2,T]`, v `[256,2,T]`; one-centered RMSNorm, interleaved MRoPE on dims 0..63, theta 1e7 | one kernel, any T |
| `hyper_connection.h` | `hyper_connection_prepare` (block input `[2560,T]` + FP32 injection `[4,T]`), `hyper_connection_mix` (final mixers), `hyper_connection_inject` (in place), capacity query | decode: v2's fused 9a/G3 chain; prefill: v3 split-K BF16 MMA with a fixed-order reduce |
| `ple_ngram.h` | `ple_ngram_decode` (exact u4z8_g16 row decode), `ple_ngram` (prefill, one slot pair, in-place allowed), `ple_ngram_snapshot` (B 1..8, one history snapshot per column, the `causal_conv1d_silu_snapshot` convention) | column-parallel dilated convolution (replaces v2's serial per-channel scan) |
| `linear.h` | FP8 admits both row-scale words; FP32-scaled problems `[13312,2560]`, `[16384,2560]`, `[2560,6144]`; BF16 adds `[640,2560]`, `[2560,2560]`, `[10240,2560]`, `[13312,2560]`, `[2560,6144]`, `[248320,2560]` | the scale word is a template parameter through every FP8 route (A16 GEMV/sliced/MMA, A8 MMA/TMA/split-K) |
| `gdn_gating_proj.h` | the Flash-Next parent `[96,2560]` | existing kernels |

**Design decisions beyond §3.**
- **KV codec.** Flash-Next attention consumes v3's paged KV exactly as `kv_cache_append` writes it
  (page-major `[256,64,2,N]`). v3's BFloat16 profile stores V as FP16 and its FP8 profile is
  row-scaled with a Hadamard-rotated K, where v2 stored BF16 V and unscaled FP8. This is a named
  codec change for the §5.3 parity criteria.
- **Indexer block keys** share the main KV page groups: a BF16 `[128,16,Npages]` plane (16 blocks
  per 64-token page) addressed through the main block table, allocatable as a `[32,64,1,N]` plane
  beside K/V. v2's separate plane pool and table are gone. The pooled block mean is cast to BF16
  before normalization, as the checkpoint does (a declared semantic seam).
- **v2's FP8-F32 kernel is not ported.** v3's A8 routes with FP32 row scales beat
  `fp8_f32_a8.cu` at every measured point.
- **MoE expert addressing** goes through one device helper `nvfp4_expert(bank, e)`, so a later
  VRAM expert cache replaces one function. No cache is implemented.
- **`Nvfp4ExpertBankWeight`** moved from `weight_input.h` to `sparse_moe.h`, next to its consumer.
- **PLE gate/combine** no longer rounds the gated value to BF16 between kernels (it put v2 2-5e-2
  from the FP64 oracle).
- **One shape translation unit per K** for shared kernel instances (`bf16/shapes/k2560.cu`,
  `fp8/shapes/k2560_fp32.cu`): memcheck reports "duplicate entry kernels" when two shape files
  instantiate the same kernel.

**`block_reduce_sum` audit.** Upstream's `warp.cuh` stays unchanged (no conflict surface). The two
hyper call sites make one reduction per launch with the broadcast behind its own barrier. v2's PLE
helper reused one shared buffer for four reductions; its replacement gives each call site its own
array, written once and read after one barrier. The MoE, attention and indexer kernels do not call
it; their shared-buffer reuses (indexer norm/rotate, snapshot append, select histogram and scan,
attention K/V tile, MoE grouping scan) each end with an explicit barrier. v3's existing call sites
already separate reuses with a barrier. A deterministic test cannot make a missing barrier fail, so
the evidence is `compute-sanitizer`: racecheck 0 hazards and memcheck/synccheck 0 errors on every
new route (MoE at T 1, 2, 9, 300; the attention, indexer and norm/rope routes; hyper, PLE and the
new linear problems).

**G4 evidence** (CUDA 13.3, RTX PRO 6000 Blackwell Server Edition; per-family sessions `m3-3a/b/c`
2026-09-30 17:02-18:45 UTC, integrated session `m3-3i` 18:41 UTC onward).

**Integrated suite** (`m3-3i`, the combined `m3/3-ops` tree): all 74 `tests/ops` targets built
and `ctest` over them (76 tests, including the two `kv_cache_append` aliases and every existing
`qwen3_5` Op test) passed 76/76, none skipped, in 1,482 s. Per-family details:

| Test | Criterion | Worst measured |
|---|---|---|
| `ninfer_sparse_moe_nvfp4_test`, A16 banks (T 1, 1 graph, 2, 8, 9, 64 graph, 255, 256, 1000) | rel-L2 ≤ 6e-3, gross ≤ 6e-3 of max ref | rel-L2 3.55e-3 |
| same, AllowA4 banks (T 1, 7, 8 graph, 9, 10, 255, 256, 257, 300 graph, 1000, 4097, 8192) | rel-L2 ≤ 0.12, gross ≤ 0.12 of max ref (dynamic FP4 activations) | rel-L2 7.49e-2 |
| same, exact 16-way score tie across the top-10 boundary; guards, NaN-prefilled output, input preservation, workspace high-water = query | exact | pass |
| `ninfer_selected_block_attention_test`: BF16 and FP8 profiles, batched B 1/2/5/8 with graph replay, shared row T 1..96, NaN in unselected cache, empty set | rel-L2 ≤ 4e-3, gross ≤ 2e-3 + 1.2e-2·max\|ref\|; empty set exact 0 | rel-L2 1.69e-3 (BF16), 1.90e-3 (FP8) |
| `ninfer_qsa_indexer_test`: four forming-block phases, snapshot W 1/4 up to B 8 with graph, select at 7.5K blocks over 8 rows and 65,575 blocks on one row, all-ties case | block keys pair-scaled 6.9e-3 + rel-L2 2.5e-3; state bytes, untouched plane regions and selections exact | rel-L2 1.87e-3; exact checks equal |
| `ninfer_rmsnorm_rope_test`, gated D256 form (T 1, 8, 2048; positions to 262100; graph) | pair-scaled 6.9e-3, rel-L2 1.85e-3; gate/v exact | rel-L2 1.73e-3 |
| `ninfer_hyper_connection_test` (T 1..9, 16, 31-33, 63-65, 256, 512, 8192; inject to T 65537; graph) | block input rel-L2 6e-3; injection 1e-3; inject pointwise 1.01·2^-8 | rel-L2 2.46e-3 |
| `ninfer_ple_ngram_test` (decode 1-65,536 rows incl. subnormal/zero scales; T 1..8192; B 1/5/8 snapshots; graph) | decode exact; output rel-L2 3e-3; state one BF16 step; untouched slots exact | decode exact; rel-L2 1.7e-3 |
| `ninfer_linear_fp8_fp32_test`, A16 and A8 (T 17..8192 incl. split-K); FP32-row-scale `native_weight` branch | Linear A16 / A8 (0.04) criteria; the prepared operand gives bit-identical output | rel-L2 2.1e-3 / 3.1e-2 |
| `ninfer_linear_bf16_a16_test`, new shapes at every route boundary, graph | Linear A16 | rel-L2 2.2e-3 |

**v2 (`87812bc8`) vs v3, same G4 session per family** (µs; decode points in CUDA-graph replay
where the family runs graphs, L2 flushed per call; medians).

| Op / point | v2 | v3 |
|---|---:|---:|
| MoE T=1 / 2 / 4 / 8 (graph) | 66.00 / 91.82 / 156.70 / 242.89 | 67.12 / 94.22 / 156.00 / 241.93 |
| MoE T=16 / 64 / 256 / 512 (graph) | 514.4 / 1135.6 / 1170.2 / 1256.9 | 504.2 / 1089.7 / 1159.2 / 1233.9 |
| MoE T=8192 (eager) | 8152 | 7839 |
| attention decode B=1 at 8K or 30K / B=8 at 30K | 28.2 / 79.4 | 11.8 / 22.7 |
| attention prefill, 8192 tokens at 30K | 16,672 | 9,064 |
| indexer decode B=1 / B=8 at 30K | 39.0 / 174 | 22.0 / 32.3 |
| indexer prefill, 8192 tokens at 30K | 4,489 | 640 |
| q/k norm+rope (+ KV write) T=1 / T=8192 | 5.66 / 355 | 5.63 / 315 |
| hyper prepare T=1 / 8 / 512 / 8192 | 31.7 / 39.9 / 123.9 / 1544 | 31.7 / 39.9 / 123.9 / 1547 |
| PLE layer T=1 / 8 / 512 / 8192 | 97.3 / 103.4 / 416.8 / 8228 | 94.2 / 96.3 / 217.1 / 2770 |
| FP8 FP32-scale `[13312,2560]` T=1 / 512 / 8192 | 40.9 / 97.3 / 1058 | 39.9 / 91.1 / 845 |
| FP8 FP32-scale `[16384,2560]` T=1 / 512 / 8192 | 47.1 / 115.7 / 1289 | 48.1 / 101.4 / 1027 |
| FP8 FP32-scale `[2560,6144]` T=1 / 512 / 8192 | 27.6 / 70.6 / 586 | 27.6 / 57.3 / 503 |
| BF16 `[640,2560]` T=1 / 64 / 8192 | 14.3 / 37.9 / 138 | 11.2 / 18.4 / 128 |
| BF16 `[248320,2560]` T=1 / 8 | 900.1 / 901.1 | 898.0 / 900.1 |

v3 is at or below v2 everywhere except: MoE T=1 and T=2, +1.1 and +2.4 µs. nsys shows v3's
kernels are faster (62.5 vs 63.4 µs at T=1); the difference is the 4-byte memset graph node that
zeroes the fused router's arrival counter in caller workspace. v2 kept that counter as a
module-global `__device__` variable across calls, which the v3 Op contract forbids. Four
hyper/linear points differed by one timer quantum (≈1 µs) and changed sign between passes. v2
had no route for BF16 `[2560,6144]` and `[248320,2560]` above T=8. Full per-point tables are in
the session scratch (`E:\v3port-scratch\m3\m33\{moe,attn,misc}`).

**Production FP8 regression A/B** (G4 session `m3-3ab`, 2026-09-30 19:30-19:55 UTC). The FP8
routes are now templated on the row-scale word, so the Qwen3.8-27B production FP8 (BF16-scale)
problems were compared between `workstation` `3a6d078b` and this branch in one session:
`linear` at all six registered `[N,K]`, `attn_input_proj` `[14336,5120]`, `gdn_input_proj` and
`gdn_input_proj_conv_snapshot` `[16384,5120]`, `linear_add` `[5120,6144]`/`[5120,17408]`,
`linear_swiglu` `[34816,5120]`, `linear_topk` and `embedding` `[248320,5120]`, policy AllowA8, at
T = 1, 2, 8, 512, 8192 (topk U = 1, 2, 8). Outputs: all 68 cases bit-identical (FNV-1a of every
output byte). Timing: the existing op benchmarks, cold L2, 100 samples per run, 5 rounds with
alternating arm order; every point is within ±0.25% of `workstation`, except points that sit on
the benchmark's two timer levels in both arms. The largest, `linear [34816,5120]` T=512 (+4.8% in
5 rounds, bimodal 256/268 µs in both arms), measured +0.00% (median) / −0.08% (mean) over 20
alternating rounds of 200 samples.

**Windows.** MSVC 14.51 + CUDA 13.3 build `ninfer_ops`, all 74 `tests/ops` targets, `ninfer`,
`ninfer-serve`, both `qwen4_exp` tests, `ninfer_artifact_materialization_test` and
`ninfer_qwen3_5_loading_test` with no errors and no new warnings in the touched files.

**Corrections for M3.4.**
- **Attention layer call order:** `linear [13312,2560]` (FP8 FP32-scale; BF16 for MTP) →
  gated `rmsnorm_rope` → `kv_cache_append` → `linear [640,2560]` → `qsa_indexer_append` →
  `qsa_indexer_select` (pass the graph context bucket as `max_complete_blocks`) →
  `selected_block_attention` → `sigmoid_mul(gate, attended)` in place → `linear [2560,6144]`.
- **Program state per QSA layer (12 target + MTP):** main K/V planes (plus scale planes for FP8),
  one indexer block-key plane in the same page groups, and per state slot the forming-block state
  (`[128,4,S]` BF16 + `[3,4,S]` I32) with the same source/destination/snapshot handling as the
  GDN convolution state. PLE: the 9-column BF16 history per state slot.
- **Gap: `kv_cache_append` has only a single-sequence D256 form.** Batched decode needs one launch
  per lane or a batched D256 append.
- **Gap: no fused GDN input projection at `[16384,2560]` FP32-scale.** Compose `linear` with the
  `causal_conv1d_silu` split/snapshot forms, or extend `gdn_input_proj` (the FP8 scale word is
  now a template parameter, so this is cheap).
- **Counter-free MoE router (decided by Igor, 2026-09-30).** M3.4 replaces the fused decode
  router's workspace arrival counter with a counter-free reduction: no module-global device
  state and no per-call memset node in the graph. This removes the MoE T=1/T=2 gap to v2 above.
- **Workspace at chunk 8192, 131,072 context:** MoE 590 MiB (AllowA4 banks, T 1..8192; 948 MiB
  if A16Only); indexer select ~72 MB; batched attention ≤ 2.6 MB.
- **Parity:** the MoE A4 prefill route (T ≥ 256) carries ~7.5% relative error on the routed term
  by design (v2's profile), so M3.4 expects near-v2, not bit-close, prefill; decode is A16.
- **G4 tooling:** concurrent `colab new` calls clobber `sessions.json`; create sessions one at a
  time and re-adopt a lost live VM (`E:\v3port-scratch\m3\m33\misc\recover.py`). `vm_ops.sh`
  setup must install the ffmpeg and curl dev packages for the v3 configure.

## 7. Risks

1. **Program port size.** The v2 Program targeted a thinner v2 contract. The v3 contract adds
   resource transactions, persistent-backfill proofs and capture-pressure planning. Budget M3.4 as
   the largest milestone. Port behaviour, not structure.
2. **New linear internals.** Rebasing the MoE and hyper kernels onto the renamed NVFP4/BF16
   internals can change reduction order. Parity criteria (§5.3) allow only near-tie divergences and
   bounded KL, and FP64 oracles judge each Op.
3. **`block_reduce_sum` lost its trailing barrier.** This is a silent shared-memory race in
   ported kernels unless audited. *Closed by M3.3 (§6.2).*
4. **Checkpoint-slot accounting.** The v3 `ResourceManager` owns the logical catalog, while
   Flash-Next's owners pair two physical slots. If the Program does not report that physical
   pressure at admission, the v2 catalog-exhaustion defect (0% reuse forever after ~8 owners)
   comes back.
5. **Mapped PLE on Windows.** The workstation has 125.7 GiB RAM, shared with the desktop.
   Page-cache-backed PLE can be evicted under pressure and stall prefill. Pinning 30 GiB is safer for latency but takes the RAM from the desktop. M3 keeps
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

## 8. Decisions (Igor, 2026-09-29)

1. **Chat template:** keep Flash-Next's official template (`c3cf9e34`) in the v3 artifact. A
   maintained template is a later, separately measured change.
2. **MTP banks:** bake NVFP4 offline into the v3 artifact (v3 forbids runtime repacking; the
   artifact SHA256 differs from v2's).
3. **v2-only runtime flags dropped:** FP8 output head, FP8 embedding, BF16 GDN state and the QSA
   MMA switch. Neither the artifact nor the converter carries them.
4. **No HF write token:** every Colab session re-derives the v3 artifact from the public v2
   download (`igorls/Qwen3.8-Flash-Next-mixed-NInfer@5f0ee7e2`, SHA256 `3d383e51...0d1d02`) with
   `python -m tools.convert.qwen4_exp.derive`, fully automatic and resumable.
5. **Upstream conflict surface of §3.1 accepted.**
6. **PLE residency for M3:** mapped page cache (v2 behaviour).
