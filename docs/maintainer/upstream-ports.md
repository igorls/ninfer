# Upstream ports

Updated: September 28, 2026.

This is the living record of how this workstation fork ([igorls/ninfer](https://github.com/igorls/ninfer))
tracks [Neroued/ninfer](https://github.com/Neroued/ninfer): the reviewed upstream point, the fork
features carried onto upstream v3, deliberate divergences, dropped and deferred work, qualification
evidence and deployment. The first priority is **Qwen3.8-27B NVFP4 on native Windows and the RTX PRO
6000 Blackwell**. RTX 5090 results remain useful evidence for that device; they do not qualify the
workstation.

## Sync state and method

| Item | Value |
|---|---|
| Fork line | `port/upstream-v3`, based on upstream [`bace20dc`](https://github.com/Neroued/ninfer/commit/bace20dc70249eed6402b66d4852c6c3f9612905) (September 24, 2026) |
| Last reviewed upstream commit | `bace20dc`, the merge-base |
| Pre-v3 fork line | `research/qwen4-flash-next` at `87812bc8` (September 26); the source for Flash-Next |
| Not yet reviewed | 15 upstream commits after `bace20dc` (September 26): Q4/Q5/Q6/Q8/FP8/NVFP4 and BF16 Linear template unification (`229c1832`, `42614c0e`, `fc62790a`, `502cd9d6`, `5d08cba8`, `fc3993d8`, `ecbc3357`), GDN two-stage kernels (`0784e76f`), KDA recurrent and chunked paths (`619e3f4c`, `6d333ce0`, `d4ea63ea`, `71a1cb0e`) and documentation rules (`b24a439f`, `2ddef207`, `e31bc99b`) |

The fork syncs by merging `origin/master` into the fork line, never by rebasing or cherry-picking
upstream work. Each merge records every upstream change it brings in here as integrated, deferred
or reverted, so the merge-base is the reviewed point. A rejected upstream change is reverted in the
fork line (or device-gated) with its measurement recorded; a deferred one names the condition that
would make it useful. The next merge must re-fit the fork's registered BF16 `[248320,5120]` shape
(OrcaRouter) into upstream's unified BF16 Linear templates.

## How to maintain this document

Update an entry when its implementation or evidence changes. Record the upstream source, affected
model and workload, the fork commit, qualification and remaining work. Status values:

- **Candidate:** reviewed; implementation and local qualification remain.
- **In progress:** adaptation or qualification has started; state the unfinished acceptance work.
- **Integrated:** present in the fork line, with the commit and the actual qualification scope.
- **Deferred:** keep the reason and the condition that would make the change useful.
- **Reverted:** removed or device-gated in the fork line, with the measurement.

Source integration, hardware qualification and deployment are separate claims. Record deployment
only after checking the installed build and service. Detailed performance results live in
[performance](../performance.md) and its [RTX PRO 6000 records](../performance/rtx-pro-6000.md);
link them here.

## Upstream through `bace20dc`

Everything upstream through `bace20dc` is integrated as the fork line's base, including the changes
the pre-v3 fork had ported selectively (the DFlash2 closure and state fixes `b8786751`, `03177b91`,
`e51b585c`; `b88c0f6f`; materialization budgets `d4929686`; the MoE L2 hints `ce954918`,
`7f14d963`; `9f0575bb`) and those it had deferred (real Jinja templates, v3 artifacts, shape-owned
dispatch, the Q4/Q5/Q8 tuning series). The review of the last 20 commits (`6cc95cc5..bace20dc`):

| Upstream | Subject | Verdict |
|---|---|---|
| `8eaed538` | preserve literal content in chat templates | Integrated: frontend correctness on the Jinja path |
| `bb844c43` `beedffa0` `d3c125ed` `5b4303c0` | Q4 dense/SwiGLU dispatch tuning | Integrated: groupwise-int routes; production is NVFP4 |
| `1d8587bc` | W4A4 activation scales one tile per TMA request | Integrated; measured with P1 below |
| `05507ab0` then `c4ae8a9c` | approximate SiLU, then accurate SiLU restored | Integrated; net result is accurate SiLU |
| `5f5fccab` | W4A4 TMA takes a partial last M tile | Integrated; measured with P1 below |
| `028eb61e` | predicated Q8 GEMM cache policy | Integrated; affects W8 DFlash2 drafter routes |
| `dc58675f` | Q5 routed-down Rows2 window | Integrated: 35B MoE only |
| `f9c4a04b` | state-cache working-set bench scenarios | Integrated |
| `cb30e070` `a9a0d10a` `b39de4d5` `f76e19c0` `9e163eee` `594930e7` | Q4/Q5 A16 routes | Integrated: groupwise-int only |
| `4c0fe48a` then `bace20dc` | blocking sync, then configurable `NINFER_CUDA_SYNC` with a spin default | Integrated with a fork divergence: the default is `blocking` (below) |

Upstream `ee9d5192` (token-fast W4A4 rasterisation, unconditional) is integrated on every device. The
pre-v3 fork had gated it to the RTX 5090 after a 7,680-token prefill regression on the RTX PRO 6000
(741.4 to 749.9 ms, September 10, older engine and configuration). On v3, with the tiled scale fetch
of `1d8587bc`, a weight-fast measurement build prefilled the same-size probe in 678.2 ms against
683.4 ms for token-fast: inside the round-to-round drift and with inconsistent sign across rounds, so
the device gate (fork `db1a3694`) is not carried
([measurement](../performance/rtx-pro-6000.md#v3-port-against-the-production-build-2026-09-28)).

## Fork features carried onto v3

Each group was transplanted onto the v3 structure as coherent commits naming their fork provenance.
Fork SHAs are on `research/qwen4-flash-next`.

| Group | Fork source | Fork-line commits | Notes |
|---|---|---|---|
| Native Windows/MSVC build | `3ca81ca8` `5a94cd1c` `0c2efc02` `11dcf0e6` `534ec75b` `d2289abf` | `173948fb` | FFmpeg/libcurl from `FFMPEG_ROOT`/`CURL_ROOT` (LGPL default), `/utf-8`, wide multiply for `__int128`, Windows file I/O, `alignas(64)` NVFP4 TMA descriptors, Windows test fixes, v2-to-v3 upgrade on Windows (`03b36e5c`) |
| Tool-call robustness | `b0c40582` `d45b1871` `b802e911` | none | Superseded by upstream's parser: schema-mismatch arguments pass as text, fallback reasons are logged |
| Serve protocol and sampling | `8264394c` `b137af03` `bb88a776` | `21e09cb3` `1b872d0b` `f9800e26` | top_k clamp, prefix discovery on by default, token masks and repetition penalty |
| Admin, telemetry, robustness | `7fc96721` `f704502a` `6a181abf` `00ad2b86` `f426d2c4` `2f375231` `3400f9c9` `039bef66` `ff20cd06` | `17e27079` | `/admin/vram`, `/admin/stats`, `/admin/quiesce`, `--api-key-file`, client attribution, tool-block digest, fatal exit, telemetry that never waits on execution |
| Memory sizing, desktop reserve | `9bf5e8b8` `5c23fe2e` | `57a277c9` | NVML device-wide sizing, `--desktop-reserve-gib`, slack floor |
| Prefix reuse under a full pool | `bb7b7305` `2b75ed5f` `aec32ee0` `f26ab57d` `529f85b8` `a29f97e2` `d2982875` | `352958d5`, clamp removed in `1c390ca6` | Observations survive consume-and-republish, newcomer protection, read-only retention, idle flush |
| Structured output | `bb7b7305` `2cbdb2b6` `bb88a776` | `ff78ceae` `f9800e26` | XGrammar JSON object/schema, `tool_choice: required`, MTP drafting under constraint; constrained DFlash rows verify no drafts |
| Token logprobs, `/v1/score` | `f596a68e` `2c307887` `da87a3f5` `c4288317` `a0b43ad4` `05797621` `3181627f` `e17813cf` `d49a8e0c` `282f013c` | `3bbb06fa` `737b570a` | `candidate_logprobs` Op, device readout, prompt positions, read-only participation |
| TypeSafe System One | `e20e7e23` `147370d7` `ff20cd06` `87812bc8` | `1833011e` | Jev drop-in contract |
| Reasoning feature readout | `15f0c5aa` and the collector at `87812bc8` | `f290c8e7` | `capture_reasoning_features`, `ninfer-reasoning-collect` |
| OrcaRouter NVFP4 | `91ce2f2c` | `e35b663d` `9a9a1823` `74c5343f` `7130a888` | BF16 `[248320,5120]` Linear and LinearTopK, Qwen2-style tokenizer resources, conversion recipe (re-converted, not upgraded) |
| Supervisor, Windows app, installer | SUP and W2 commits, `bbe3e16e` | `168bdf12` `fc9a1ebe` | |
| Arcade, JevBench, router, probes | ARC, JEV, RTR, BEN commits | `a1f8b2d9` `6dfd01b8` `7c8555cb` | |
| Docs, model card | DOC commits, `97200f2b` | `174fdb87` `b80c497b` | Fork README; the OrcaRouter card describes the published v2 artifact |

The DFlash2 residue of `ddeeec19` needed nothing: v3 rejects an unsettled StateImage Fork at seal
revalidation instead of the fork's early admission gate, and the fork-only all-constrained
width-one DFlash path is not carried.

## Deliberate divergences from upstream

| Behaviour | Upstream | Fork | Reason and evidence |
|---|---|---|---|
| CUDA synchronization default (`300ddb9f`) | `spin` | `blocking` when `NINFER_CUDA_SYNC` is unset; the variable still selects `spin`, `yield` or `auto` | Shared desktop. Production-flag A/B on the RTX PRO 6000: 7,680-token prefill 681.0 vs 683.4 ms, decode 152.9 vs 148.4 tok/s (medians, within noise), CPU during decode 0.03 vs 0.96 core ([measurement](../performance/rtx-pro-6000.md#v3-port-against-the-production-build-2026-09-28)). Igor, 2026-09-28 |
| Dense non-thinking presence penalty (`92ad4c47`) | `1.5` | `0`; MoE keeps `1.5` | Production parity: the pre-v3 27B package used 0 and production requests record 0 |
| `preserve_thinking` server default (`3cb5e718`) | unset, so the Qwen3.8 template keeps returned closed-turn reasoning | `false` unless a request or `--preserve-thinking` asks | Production parity (Igor, 2026-09-28). Keeping the template default is a later change to agree with bentokit |
| Continued final assistant turn (`737b570a`) | rendered without a think block | rendered behind the empty think block the generation prompt carries (thinking off) | `/v1/score` text form and assistant prefill are conditioned like a generated answer |

## Dropped

Approved by Igor, 2026-09-27 and 2026-09-28.

| Item | Reason |
|---|---|
| `84f0347f` tool-call recovery, POST `/admin/vram` | Lost in the 2026-08-30 merge; superseded by upstream's parser and `f704502a` |
| `3c6e41ec` merge integration fixes | Specific to the 2026-08-30 merge |
| `98fa9c14` explicit pimpl move operators | Targets the removed qwen3_6 template API |
| `a7c24564` `362aa76c` `7de1c559` KV-store lift | v3 owns the logical/host KV stores under `models/qwen3_5/program/storage` |
| `48c11931` Flash-Next MTP brief | Completed plan |
| `NINFER_BUILD_MEDIA=OFF` stubs | v3 makes FFmpeg and libcurl mandatory |
| `supervisor-logs-demo/`, `bench/d20_results*/` | Measurement output, not source |
| Private-catalog clamp from `aec32ee0` | Removed capacity that exists: v3 backs catalog entries with idle lanes' state slots as well. By its own account it only corrected a reported number |
| `docs/tribuno-production-evaluation-plan.md` | Completed 2026-09-07 handoff plan |
| `db1a3694` device-gated W4A4 raster | Not reproduced on v3 (above) |

`tools/freq_corpus/fixtures/ranking/*.i64` are upstream's own files (the default ranking of the
27B proposal head) and stay.

## Deferred

- **Qwen3.8-Flash-Next (M3), waiting for Igor's go-ahead.** Its source is the 140 Flash-Next commits
  of `research/qwen4-flash-next` (`6314ab02` to `4cd6b5f4`, interleaved with other work); the
  per-commit notes are in git
  history (`git show 27426227:docs/research/upstream-v3-port-2026-09-27.md`). It becomes a second
  architecture package beside `src/models/qwen3_5` with its own v3 converter recipe and artifact.
  Pieces outside the Flash-Next directories are easy to miss: the v2 artifact formats and mapped
  payloads (`6314ab02`, `8a824a23`, `e52495e4`, `b8619496`), Flash-Next Ops under `src/ops`
  (FP8 projections, QSA, PLE, hyper-connections, MoE, BF16 vision shapes), the shared vision
  encoder refactor (`f362fd0f`), Engine and serve options (FP8 head and embedding, BF16 GDN state,
  QSA MMA prefill, draft floor), the shrinkable device buffer (`00ad2b86`), selected-block split
  attention (`bb7b7305`), `841f8361`, the draft-head shortlists (`shortlist_32k.i32`,
  `shortlist_65k.i32`), `tools/reference/qwen3_8_flash_next`, its model card and performance
  records.
- **Matched Flash-Next versus Qwen3.8-27B prefill study.** Resume with M3: same workstation,
  matched application texts and supported contexts, verified cold and 90%-cached ratios, TTFT,
  uncached prefill and complete request latency reported separately.

## Open qualification questions

- `prefix_real`'s Host-restore check fails on the Qwen3.8-27B NVFP4 artifact identically on the
  pristine base and the fork line (the demoted turn closure is not the selected source): upstream
  behaviour on this artifact, not investigated further.
- A json_schema request whose prompt asks for prose produces whitespace until its output limit on
  both the pre-v3 production build and the fork line: the grammar admits unbounded leading
  whitespace. A bound on outer whitespace would fix it.
- On the pre-v3 line, exploratory T=1500 GDN input and LinearAdd 5120x6144 NVFP4 fixtures disagreed
  with their oracles on both baseline and candidate; not re-examined on v3.

## Qualification of the fork line

Windows, MSVC 19.51, CUDA 13.3, `sm_120a`, RTX PRO 6000 Blackwell (driver 616.92):

- Full CTest at `51474054`: 137 tests, 128 passed, 9 skipped (artifact- or source-gated), 0 failed.
  The pristine base `173948fb` had 127 tests, 120 passed, 7 skipped. Later changes ran their
  focused tests (serve options and schemas for `3cb5e718`, the device sync tests for `300ddb9f`).
- Every changed Op passes its independent oracle test: `candidate_logprobs`, BF16 `[248320,5120]`
  Linear, BF16 LinearTopK, and the NVFP4 Linear/LinearAdd/LinearSwiGLU tests on both rasters.
- Real-model tests on the v3 copies of the production Qwen3.8-27B NVFP4+DFlash2 artifact and the
  re-converted OrcaRouter artifact: loading with MTP and DFlash2, causal scoring, reasoning features
  and Vision workspace pass on both; `dflash2_real` (K=15 and K=7, 8 rows) and every `prefix_real`
  scenario except Host restore pass on the production copy.
- Spare-port serving smoke (production flags, MTP and DFlash2): chat with and without thinking,
  image input, reuse across turns, logprobs, `/v1/score`, `/v1/systemone`, required tool calls,
  structured output, Anthropic Messages. The supervisor manages the engine as a monitor-only twin.
- Production-flag A/B against the installed pre-v3 build: the fork line matches it within the
  round-to-round drift on a 7,680-token prefill and a 256-token MTP5 decode
  ([table](../performance/rtx-pro-6000.md#v3-port-against-the-production-build-2026-09-28)).
- A 48-entry private catalog is accepted and reported; retention follows the state-image backing
  (`--device-state-slots` and `--host-state-slots`), and evictions are counted in `/admin/stats`.

## Deployment

As of September 28, 2026 the x870e production service on :8010 runs the installed pre-v3 build
(`ninfer-serve` from fork `87812bc8`, supervisor from `bbe3e16e`, release `2026.09.24-alpha.1`
runtime files) with the v2 production artifact
`C:\models\Qwen3.8-27B\qwen3_8_27b_nvfp4_dflash2.ninfer`. The fork line is not deployed. Its v3
copy of that artifact is `E:\models\v3\Qwen3.8-27B\qwen3_8_27b_nvfp4_dflash2.ninfer`.
