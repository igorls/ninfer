# Upstream v3 port ledger — 2026-09-27

This is the tracking record for moving the fork onto upstream NInfer v3. Update the status
columns as work lands; remove the document when the port is complete and the fork syncs by
merging upstream.

## Scope

| Item | Value |
|---|---|
| Old fork line | `research/qwen4-flash-next`, pinned at `87812bc8` (2026-09-26) |
| Merge-base with upstream | `21a0e85f` (2026-09-02) |
| Fork commits since merge-base | 258 non-merge, 20 merges |
| Port branch | `port/upstream-v3`, based on upstream `bace20dc` (2026-09-24) |
| Previously reviewed upstream point | `6cc95cc5` (fork issue #21); `bace20dc` becomes the reviewed point |
| Priority | Qwen3.8-27B to production parity (M2), then Flash-Next (M3) |

Method: features are transplanted onto the v3 structure as coherent commits in dependency order,
not by rebasing or merging the 258 commits. Each commit builds on Windows/MSVC, passes its
relevant tests, and names its fork provenance ("Ported from fork ...").

## Classification summary

| Class | Meaning | Commits |
|---|---|---:|
| a | Already in upstream (patch-equal or superseded) | 11 |
| b | 27B/common: must port | 47 |
| c | Flash-Next only (M3) | 140 |
| d | No-conflict tooling, apps, docs | 53 |
| e | Obsolete: proposed drop | 7 |
| | Total | 258 |

The 20 merges carry no feature of their own. `7478c8d0` (2026-08-30) and `e650ee62` (2026-09-01)
resolved the upstream serve/frontend rewrites; the early fork serve work of 2026-08-27
(tool-call recovery, admin endpoints) did not survive them, so the fork's final state at
`87812bc8`, not the original commits, defines what is ported. `8d3cc1d8` and `933c46d1` merged
PR #20 (Windows Qwen3.8 setup, `534ec75b`); the D-series merges only integrate Flash-Next work
branches.

## Feature groups and port order

Status values: `todo`, `wip`, `done` (with port-branch SHA), `verify` (believed superseded,
confirm in M2), `dropped` (after approval).

| # | Group | Class | Fork SHAs | v3 target | Status |
|---:|---|---|---|---|---|
| 1 | W1 Windows/MSVC native build | b | `3ca81ca8` `5a94cd1c` `0c2efc02` `11dcf0e6` `534ec75b` `d2289abf` | `cmake/Dependencies.cmake`, top `CMakeLists.txt`, `src/core/platform.h`, `src/core/wide_multiply.h`, `src/artifact/file_io.cpp`, product logging, tests | done `173948fb`; benchmark targets not yet built on Windows |
| 2 | T1 Tool-call robustness | b | `b0c40582` `d45b1871` `b802e911` | `src/models/qwen3_5/frontend/tool_call_parser.cpp` and its tests | done: superseded by upstream (schema-mismatch arguments pass as text; fallback reasons named in logs). The opt-in NINFER_TOOL_CALLS_ALLOW_TRAILING_TEXT switch is not carried: nothing sets it, and the default (strict) matches production |
| 3 | S1 Serve protocol and sampling | b | `8264394c` `b137af03` `bb88a776` | `src/serve/*_request.cpp`, `request_validation.cpp` | done: `21e09cb3` (top_k clamp, prefix discovery #23), `1b872d0b` (sampler), `f9800e26` (repetition penalty end to end). `8264394c` is obsolete on v3: the artifact template receives the effort and there is no capability model to list |
| 4 | S2 Admin, telemetry, engine robustness | b | `7fc96721` `f704502a` `6a181abf` `00ad2b86` `f426d2c4` `2f375231` `3400f9c9` `039bef66` `ff20cd06` (`/admin/stats`) | `include/ninfer/engine.h`, `src/runtime/engine/engine_core.h`, `src/serve/http_server.cpp`, `src/serve/device_snapshot_cache.h`, `src/serve/request_log.cpp` | done `17e27079` |
| 5 | S3 Memory sizing, desktop reserve | b | `9bf5e8b8` `5c23fe2e` | `src/core/device_memory.*` (NVML), Engine KV capacity resolution, serve/CLI options | done `57a277c9` |
| 6 | R1 Prefix reuse, catalog stability | b | `bb7b7305` (resource manager part) `2b75ed5f` `aec32ee0` `f26ab57d` `529f85b8` `a29f97e2` `d2982875` | `src/runtime/engine/context_cache/*`, `src/models/qwen3_5/program/planning`, `src/models/qwen3_5/frontend/chat_template.cpp` | done `352958d5` (resource manager, idle flush); its catalog clamp from `aec32ee0` removed in `1c390ca6` (Igor, 2026-09-28); `2b75ed5f` (issue #13 tool-loop checkpoint) verify with the tool-loop reuse probe in the M2 smoke; `d2982875` with L1 |
| 7 | G1 Structured output (XGrammar) | b | `bb7b7305` (XGrammar part) `2cbdb2b6` `bb88a776` (required tool choice) `ddeeec19` (all-constrained width-one) | `third_party/xgrammar`, `src/runtime/contract/structured_output.*`, sampler/speculative mask plumbing in `src/ops`, `src/models/qwen3_5/program`, three protocol adapters | done `ff78ceae` (engine), `f9800e26` (protocols); the fork-only DFlash all-constrained width-one path is not carried (v3 allocates no ordinary buffers under speculation), constrained DFlash rows verify no drafts |
| 8 | L1 Logprobs, `/v1/score`, read-only participation | b | `f596a68e` `2c307887` `da87a3f5` `c4288317` `a0b43ad4` `05797621` `3181627f` `e17813cf` `d49a8e0c` `282f013c` | `src/ops/candidate_logprobs`, `src/runtime/contract/token_logprobs.h`, `src/models/qwen3_5/program` sampling and prefill sites, `src/serve/ninfer_score_http.cpp` | done `3bbb06fa` (candidate_logprobs Op), `737b570a` (host and device readout, prompt positions, read-only participation, /v1/score). Divergence from upstream: the maintained templates render a continued final assistant turn (thinking off) behind the empty think block, as the fork did; real-model check of /v1/score pending the v3 artifact copies |
| 9 | Y1 TypeSafe System One | b | `e20e7e23` `147370d7` `ff20cd06` `87812bc8` | `src/serve/typesafe_systemone*`, `tests/test_typesafe_schema.cpp` | done `1833011e`. Sources carried verbatim onto the v3 generation service; the decision-arcade client paragraph of `docs/serving.md` returns with the ARC pages. Real-model /v1/systemone check in the M2 smoke |
| 10 | Q1 Reasoning feature readout | b | `15f0c5aa` | `ExecutionOptions`, `src/models/qwen3_5` frontend frontier and program readout, `apps/reasoning-collect` | done `f290c8e7`. The frontier comes from the rendered layout's final open assistant block and is absent (capture refused at submit) when not an exact token frontier, instead of the fork's throw during every prepare. The collector is the 87812bc8 version (concurrency, derive-2048, direct-only); it checks `model_name` qwen3.8-27b plus an `nvfp4` weight format. Real test passes on both v3 artifacts. `reasoning_router.py` and the Colab tools move with RTR |
| 11 | O1 OrcaRouter NVFP4 | b | `91ce2f2c` | `tools/convert` recipe (BF16 embedding + full head), tokenizer resource handling, BF16 Linear/LinearTopK qualification, supervisor preset | done `e35b663d` (BF16 [248320,5120] Linear and LinearTopK, oracle tests), `9a9a1823` (letters-only split, optional added_tokens_decoder/add_bos_token, 17-case tokenizers fixture), `74c5343f` (recipe `qwen3_8_27b_orcarouter_nvfp4`). v3 artifact `E:\models\v3\OrcaRouter-Qwen3.8-27B-NVFP4\qwen3_8_27b_orcarouter_nvfp4.ninfer` (26,268,683,012 bytes, CPU conversion 127 s) loads with MTP and DFlash2. The v3 proposal already force-includes special tokens. Supervisor preset moves with SUP |
| 12 | D2 DFlash2 residue | b/a | `ddeeec19` | StateImage-fork admission block in the Program; 27B DFlash2 real test | verify. v3 rejects an unsettled Fork at seal revalidation (`StalePolicyState`, seal returns no plan) instead of the fork's early `inspect_admission` gate; not carried unless a test shows a stall. DFlash2 serving smoke passes; `ninfer_qwen3_5_dflash2_real_test` needs about 3.7 GB (4 rows) to 6.3 GB (8 rows) beyond weights under the fixed 8 GiB default desktop reserve and cannot run beside production |
| 13 | Tooling and apps | d | SUP, W2, RTR, JEV, ARC, BEN, DOC rows below | `apps/ninfer-supervisor`, `apps/windows`, `scripts/windows`, `tools/bench`, `docs`, `model-cards` | done: `168bdf12` supervisor (unit test passes; monitor-only twin on 8098 observed a v3 engine), `a1f8b2d9` arcade (12 Node tests pass), `fc9a1ebe` installer scripts, `6dfd01b8` JevBench and router (13 pytest pass), `7c8555cb` probes and CUDA 13.3 container, `174fdb87` fork README and `docs/performance/rtx-pro-6000.md` (pre-v3 DFlash2, OrcaRouter and W4A4 records), `b80c497b` OrcaRouter card carried as-is (describes the published v2 artifact; republishing v3 is a later decision, Igor 2026-09-28). `docs/tribuno-production-evaluation-plan.md` dropped as a completed plan (Igor, 2026-09-28). The fork's edits to `run_serve_*`/`run_ninfer_bench_matrix.py` are superseded by v3 (report schema v15, DFlash2 modes); Flash-Next tools, `shortlist_*.i32` and the Flash-Next card wait for M3 |
| 14 | P1 NVFP4 W4A4 schedule by device | b | `db1a3694` | `src/ops/linear/nvfp4` TMA raster selection | measure first |
| 15 | Flash-Next package + converter + v3 artifact | c | 140 rows (FN, FA) | second architecture package beside `src/models/qwen3_5`, v3 converter recipe | M3 |

Order rationale: tool-call, protocol and admin work (2-5) are self-contained and restore the
serving surface production clients use. Prefix-reuse stability (6) precedes structured output
(7), whose mask plumbing precedes logprobs (8): the mask-renormalised readout and read-only
participation build on both. System One (9) builds on the logprob readout and read-only
participation. The supervisor (13) needs the S2/S3 endpoints and flags. The W4A4 schedule (14) is
a measurement decision on the ported engine. v3 artifact copies, real-model tests, the serving
smoke, the supervisor twin and the production A/B close M2.

Already-in-upstream rows (`a`) need no port: `436b0dde` `b2513be7` `2b515a4d` `90ca8509` `f43f1794`
`5e4a66d0` are patch-equal; `7f6ae7cc` is upstream `d4929686`; `a5e9b7be` (content parts on tool
messages) and `35a27329` (`chat_template_kwargs`) are in upstream's request parsers; `e734dfdd`
(long-anchor slot identity) has identical checks in upstream's resource manager; `ddeeec19` is
mostly the upstream DFlash2 closure.

## Drops (approved by Igor, 2026-09-27)

| Item | Reason |
|---|---|
| `84f0347f` tool-call recovery, POST `/admin/vram` | Lost in merge `7478c8d0`; superseded by upstream parser rework and `f704502a` |
| `3c6e41ec` merge integration fixes | Specific to the 2026-08-30 merge |
| `98fa9c14` explicit pimpl move operators | Targets the removed qwen3_6 template API; v3 builds clean on MSVC without it |
| `a7c24564` `362aa76c` `7de1c559` D19 KV-store lift | v3 owns logical/host KV stores under `models/qwen3_5/program/storage`; Flash-Next decides its own store placement in M3 |
| `48c11931` Flash-Next MTP implementation brief | Completed plan |
| `NINFER_BUILD_MEDIA=OFF` stubs (`decode_stub.cpp`, `acquire_stub.cpp`) | v3 makes FFmpeg and libcurl mandatory; every Windows build here has them |
| Committed data: `supervisor-logs-demo/` (13 MB), `bench/d20_results*/` | Measurement output, not source; stays in the old branch history |

Correction (Igor, 2026-09-27): `tools/freq_corpus/fixtures/ranking/*.i64` are not Flash-Next data.
They are byte-identical to upstream's copies at `bace20dc`, and the v3 converter uses
`ranking.train.counts.i64` as `DEFAULT_RANKING` for the 27B optimized proposal head
(`tools/convert/proposal.py`). Nothing to drop. Only the fork's `shortlist_32k.i32` and
`shortlist_65k.i32` beside them are Flash-Next inputs (draft head, `load/materialized.cpp`); they
come with M3.

## Upstream `6cc95cc5..bace20dc` verdicts

All 20 commits are kept in the port base. Two need a production decision.

| Upstream | Subject | Verdict |
|---|---|---|
| `8eaed538` | preserve literal content in chat templates | Keep: frontend correctness on the Jinja path |
| `bb844c43` `beedffa0` `d3c125ed` `5b4303c0` | Q4 dense/SwiGLU dispatch tuning | Keep: groupwise-int routes only; production is NVFP4 |
| `1d8587bc` | W4A4 activation scales one tile per TMA request | Keep; its RTX PRO 6000 effect is measured with P1 |
| `05507ab0` then `c4ae8a9c` | approximate silu, then accurate silu restored | Keep both; net result is accurate silu, as on the fork |
| `5f5fccab` | W4A4 TMA takes a partial last M tile | Keep; measured with P1 |
| `028eb61e` | predicated Q8 GEMM cache policy | Keep; affects W8 DFlash2 drafter routes |
| `dc58675f` | Q5 routed-down Rows2 window | Keep: 35B MoE only |
| `f9c4a04b` | state-cache working-set bench scenarios | Keep |
| `cb30e070` `a9a0d10a` `b39de4d5` `f76e19c0` `9e163eee` `594930e7` | Q4/Q5 A16 routes | Keep: groupwise-int only |
| `4c0fe48a` then `bace20dc` | blocking sync, then spin default with `NINFER_CUDA_SYNC` | Keep upstream's spin default for parity (Igor, 2026-09-27). Spin equals production today: the fork never set device flags, and `cudaDeviceScheduleAuto` spins on a 32-thread host with one GPU. The close-out A/B also measures `NINFER_CUDA_SYNC=blocking` on the port build (TTFT, decode tok/s, CPU core freed) as data for a later production decision |

Earlier upstream `ee9d5192` (in the base, reviewed before `6cc95cc5`) made token-fast W4A4
rasterisation unconditional. The fork rejected it on the RTX PRO 6000 (7,680-token prefill
741.4 -> 749.9 ms, 2026-09-10) and kept it only for the RTX 5090 (`db1a3694`). It stays in the
base for now (Igor, 2026-09-27): the M2 NVFP4 A/B (group P1) measures it on this RTX PRO 6000
against the fork's 741.4 ms 7,680-token reference, with the 2026-09-10 probe, and decides whether
a device-selected weight-fast schedule comes back.

## Decisions and risks

Recorded decisions (Igor, 2026-09-27): proceed with M2 in the proposed order; drops approved as
listed above; keep the spin CUDA sync default; re-convert OrcaRouter; keep `ee9d5192` pending the
P1 A/B.

Recorded decisions (Igor, 2026-09-28):

- Remove the private-catalog clamp (fork `aec32ee0`, carried in `352958d5`): done `1c390ca6`, no
  option left behind. The three `prefix_real` pressure scenarios passed with the clamp disabled;
  their confirmation run on the final build needs about 30 GB free (21.34 GiB weights plus the
  test's fixed 8 GiB default reserve) and is in the GPU window.
- OrcaRouter keeps following its `tokenizer.json` (letters-only split), documented in the
  OrcaRouter section of `docs/weight-conversion.md` (`7130a888`), which replaces the fork's
  per-target artifact reference on v3.
- Drop `docs/tribuno-production-evaluation-plan.md`; carry the OrcaRouter model card as-is.
- The Codex NVFP4 session is abandoned (below).
- GPU window: production may be stopped for the port's exclusive measurements; the main session
  warns the dependents and gives the go.

1. System One scores change across builds (see the Tribuno rollout rule). Any v3 production build
   needs a bentokit recalibration before it enters the judge list.
2. v3 renders the artifact's maintained Jinja chat template. The fork's template behaviours
   (issue #13 checkpoint at the generation opener; the continued assistant turn opening with an
   empty think block for `/v1/score`) must be re-derived on that path and re-verified with the
   reuse probes and the score/System One tests.
3. `tools/upgrade_ninfer_v2_to_v3.py` knows only the seven official identities. The NVFP4+DFlash2
   artifact should upgrade (DFlash2 components are handled); the OrcaRouter artifact
   (`qwen3.8-27b-orcarouter/nvfp4`, BF16 embedding and full head) is not a known input. Decision
   (Igor, 2026-09-27): re-convert OrcaRouter from its local source with a v3 recipe, outputs under
   `E:\models\v3\`, never overwriting an existing artifact; the upgrade tool is not extended for
   it. The upgrade tool still gets its Windows patch for the NVFP4+DFlash2 production copies.
4. CUDA synchronization default and the W4A4 raster: decided above.

## Coordination notes (2026-09-27)

- Igor's main tree `P:\NInfer` holds an uncommitted `docs/maintainer/upstream-ports.md` and a matching
  AGENTS.md rule (another session, 2026-09-26; screened upstream through `bace20dc` against fork
  `bbe3e16e`). It is meant to become the fork's upstream-port authority. It is not copied into this
  branch; at M2 close-out this ledger's upstream verdicts and port record fold into it, and this
  ledger is removed as a completed plan.
- The Codex NVFP4 session is gone and its work is abandoned (Igor, 2026-09-28). Its worktree
  (`C:\Users\igorl\.codex\worktrees\nvfp4-prefill`) still holds 19 uncommitted edits adapting
  upstream `1d8587bc`/`5f5fccab` (tiled activation scales, partial final TMA tiles) to the old
  branch; it is left untouched. The v3 base carries both commits in upstream form, and P1 measures
  the v3 schedule.
- `tools/upgrade_ninfer_v2_to_v3.py` now runs on Windows (`03b36e5c`). The production copy is
  `E:\models\v3\Qwen3.8-27B\qwen3_8_27b_nvfp4_dflash2.ninfer` (`C:\models` is a junction to
  `E:\models`).

## M2 verification so far (2026-09-27)

Full CTest at Q1 (`f290c8e7`): 135 tests, 127 passed, 8 real-model skips, 0 failed (632 s, GPU
shared with production). Later groups ran their focused tests: BF16 Linear and LinearTopK oracles,
OrcaRouter tokenizer (17 cases), supervisor, frontend, serve schema tests.

Real-model tests (`NINFER_TEST_ARTIFACT`, one at a time beside production):

| Test | Production copy | OrcaRouter v3 |
|---|---|---|
| `loading_real` MTP and DFlash2, vision, optimized proposal | pass (21.5 / 23.3 GB device) | pass (24.0 / 25.8 GB) |
| `score_real` | pass | pass |
| `reasoning_features_real` | pass | pass (5,120 values at frontier 1,313) |
| `vision_workspace` | pass | pass |
| `prefix_real` | see below | not run |
| `dflash2_real` | blocked: memory under the 8 GiB default reserve | not run |
| `moe_real`, `dflash_real` | 35B-A3B only; no local 35B artifact | |

`prefix_real` on the production copy, per scenario, port against the pristine base `173948fb`
(`E:\NInfer.v3base`, golden fix only): vision, concurrent, anthropic-prefix-regression,
shared-rewrite-materialization, shared-replacement, private-long-anchor, rewrite-checkpoint,
rewrite-checkpoint-shared and stream-observations pass on both. Host restore (inside `all`) fails
identically on both (upstream behaviour here). pressure-resume, private-checkpoint-pressure and
source-pressure-protection fail only on the port; all three pass with the private-catalog clamp
(fork `aec32ee0`, carried in `352958d5`) disabled. The clamp is a no-op at v3 defaults (P=2C=16,
H+R=16) and at the production flags; it binds when P > H+R, as in these tests (P=4, H=2, R=0),
where upstream backs catalog entries with idle lanes' state slots. Removed (Igor, 2026-09-28).

Spare-port serving smoke (8021, production copy, production flags with 4 lanes and 32K KV, 6 GiB
reserve): MTP and DFlash2 each 10/11 — text thinking and non-thinking, image, reuse across turns
(1,247 of 1,271 prompt tokens cached), logprobs, `/v1/score` token and text form,
`/v1/systemone` noul/choice/score and 422 shape, required tool call, Anthropic Messages. The miss is
a json_schema request whose prompt asks for prose: the constrained answer stays whitespace until
the limit (the grammar admits leading whitespace, same code as the fork); with a prompt that asks
for JSON both json_schema and json_object return valid JSON in 25 tokens. The production binary
returns the same whitespace run for that request (one read-only request to :8010, 2026-09-28), so it
is pre-existing, not a port regression. MTP and DFlash2 give identical greedy outputs, logprobs and
System One values.

Production defaults the port has to keep (checked 2026-09-28 against the fork and the production
request log):

- Dense non-thinking presence penalty: the fork's 27B package uses 0.0 in both modes (`91ce2f2c`);
  v3's Dense non-thinking preset was 1.5 and production requests record 0.0. Restored in `92ad4c47`
  (MoE keeps 1.5).
- `preserve_thinking`: the fork's server default is `false`, passed explicitly into every prompt, so
  closed-turn `reasoning_content` a client sends back is dropped (production log:
  `preserve_thinking: false`). v3 leaves it unset and the Qwen3.8 template then retains that
  reasoning. Clients that return reasoning in their history would see longer prompts and different
  model input. Decision pending (Igor).
- Reasoning effort (xhigh when unset), `kDefaultMaxTokens` (8192) and the other serve defaults
  match.

Private catalog of 48 (the Concierge size) on a spare port, v3 production copy, 4 lanes, 16K KV,
no speculation, 40 distinct two-turn sessions then third-turn revisits of sessions 0-7 and 36-39:
the engine accepts and reports `private 48`. With the default 8 Host state slots, second-turn reuse
stays at 99% for all 40 sessions and there are no idle flushes, but retention is bounded by state
images: Host state slots 8/8, 34 private owners evicted (counted in `/admin/stats`), 3 of 12
revisits hit. With `--host-state-slots 44`, all 12 revisits hit at 99% (12 Host state restores) and
6 owners were evicted. A catalog of 48 therefore needs matching state slots to retain 48 sessions;
nothing fails silently.

## M1 — pristine Windows baseline

Base `bace20dc` plus `173948fb` (`build(windows): build upstream v3 natively with MSVC and CUDA
13.3`). Upstream as-is does not configure on Windows (pkg-config is mandatory for FFmpeg and
libcurl), so the baseline is the smallest change set that builds and runs its own tests there.

Configuration: `E:\NInfer.v3port\build-win`, generator `Visual Studio 18 2026` (MSVC 19.51), CUDA
13.3.33, `CMAKE_CUDA_ARCHITECTURES=120a`,
`FFMPEG_ROOT=P:/third_party/ffmpeg/ffmpeg-master-latest-win64-gpl-shared` (development build
only; the CMake default is the LGPL distribution), `CURL_ROOT=P:/third_party/curl-inst`,
`NINFER_BUILD_APPS=ON`, `BUILD_TESTING=ON` (upstream's name; the fork's `NINFER_BUILD_MEDIA` has
no v3 equivalent because media is mandatory). Built at below-normal priority with unrestricted
`-j`.

MSVC/Windows fixes in `173948fb`:

| Area | Fix |
|---|---|
| Dependencies | FFmpeg/libcurl from `FFMPEG_ROOT`/`CURL_ROOT` as `ninfer::ffmpeg`/`ninfer::libcurl`; LGPL default; CTest `PATH` for the CUDA/FFmpeg/curl DLLs |
| Compiler | `/utf-8` everywhere (a Jinja test has non-cp1252 literals), `/Zc:preprocessor` for CUDA host passes, `NOMINMAX`, module scanning off, `UTF8PROC_STATIC` |
| 128-bit arithmetic | `core/wide_multiply.h` replaces `unsigned __int128` in the prefill work model, context cost and planner comparisons; new exact test against a 16-bit-limb oracle |
| POSIX calls | `core/platform.h` (pid, isatty, `localtime`/`gmtime`), console width via `GetConsoleScreenBufferInfo`, Winsock in media acquisition, `localtime_s` in the vendored Jinja runtime |
| Artifact reader | `ReadFile` with offsets; unbuffered direct reads; full file sharing (POSIX semantics) |
| NVFP4 W4A4 TMA | descriptor block `alignas(64)` instead of 128 (MSVC rejects by-value parameters aligned beyond 64; cuda.h drops CUtensorMap's `alignas` under MSVC's `__cplusplus`). All NVFP4 Linear, LinearSwiGLU and input-projection op tests pass against their oracles |
| Python tools | binary descriptors, seek-based `pread`/`pwrite` fallback, `fsync` writeback; recipe paths with a drive letter |
| Tests | portable temp dirs, aligned `operator new`, CRT stderr capture, `const` `std::sqrt`, a missing `<array>`, spin instead of a 1 ms sleep, UTF-8 subprocess I/O, empty `NINFER_CUDA_SYNC` via `cmake -E env`, GNU ld `--wrap` fault injection Linux-only |

The fork's other pre-v3 Windows commits need nothing on v3: `98fa9c14` (pimpl move operators)
targets removed code, and the NVFP4 descriptor staging buffer of `3ca81ca8` (one static device
copy written per launch) is replaced by the alignment fix above.

### CTest baseline

`ctest --test-dir build-win -C Release --output-on-failure` (serial, about 630 s, GPU shared with
production serving):

| Result | Count | Tests |
|---|---:|---|
| Passed | 120 | core, artifact, runtime, serving-schema, frontend, op-oracle and Python interop tests |
| Skipped (exit 77) | 7 | `ninfer_qwen3_5_{loading_real,prefix_real,score_real,vision_workspace,dflash2_real,moe_real,dflash_real}_test`: `NINFER_TEST_ARTIFACT` unset |
| Failed | 0 | |

CTest lists the seven as skipped ("Not Run"), not passed. v3 real-model tests take one
`NINFER_TEST_ARTIFACT` path instead of the fork's per-identity variables.
`pytest tests/artifact tests/convert tests/test_serve_corpus.py`: 49 passed.

Windows findings to carry into M2:

- CTest applies `ENVIRONMENT` to its own process and, on Windows, does not remove an empty
  variable afterwards: every later test inherited `NINFER_CUDA_SYNC=""` and failed at device
  creation. Avoid empty values in test `ENVIRONMENT`.
- `tools/upgrade_ninfer_v2_to_v3.py` calls `os.posix_fadvise`/`os.fdatasync`; it needs the same
  Windows treatment before it can upgrade the production artifacts here.
- The Visual Studio generator compiles a project's `.cu` files one at a time; a full rebuild of
  this tree takes about 15 minutes at below-normal priority.

## Per-commit ledger

Class and group per fork commit, in fork order. Groups: W1 Windows build, W2 Windows app and
installer, T1 tool calls, S1 protocol/sampling, S2 admin/telemetry/robustness, S3 memory sizing,
G1 structured output, R1 prefix reuse, L1 logprobs/score, Y1 System One, Q1 reasoning readout,
O1 OrcaRouter, P1 W4A4 schedule, D2 DFlash2, UP carried upstream commit, FN Flash-Next, FA
Flash-Next v2 artifact formats, SUP supervisor, RTR router, JEV JevBench, ARC arcade, BEN bench
tools, DOC docs, OBS obsolete.

| Fork SHA | Date | Subject | Class | Group | Note |
|---|---|---|---|---|---|
| `3ca81ca8` | 2026-08-27 | feat(platform): native Windows MSVC+CUDA build and toolcall/admin fixes | b | W1 | platform half re-done in M1 on v3 (file_io, platform.h, wide multiply); its tool-call JSON form and admin routes were lost in merge 7478c8d0 (admin re-done by f704502a) |
| `7fc96721` | 2026-08-27 | feat(serve): tolerant tool calling, supervisor app, prefix observability, fatal executor handler | b | S2 | supervisor app -> SUP (d); fatal executor handler + test survive at 87812bc8 (port); tolerant tool-call parser lost in merge 7478c8d0 |
| `84f0347f` | 2026-08-27 | fix(serve): partial tool-call recovery, tag whitespace resilience, and POST /admin/vram endpoints | e | OBS | tool-call recovery and POST /admin/vram lost in merge 7478c8d0; superseded by upstream 3b50962b/0c5d570c/719d56ef and fork f704502a Dropped (Igor, 2026-09-27). |
| `5a94cd1c` | 2026-08-27 | feat(media): enable native Windows Vision support via direct FFmpeg and libcurl discovery | b | W1 | FFmpeg/libcurl discovery without pkg-config: re-done in M1 (cmake/Dependencies.cmake) |
| `a5e9b7be` | 2026-08-27 | fix(serve): accept structured multi-part content (including images) in tool messages | a | S1 | upstream parse_tool_message accepts content parts (openai_chat_request.cpp) |
| `2b75ed5f` | 2026-08-27 | fix(frontend): publish rewrite checkpoint at current generation opener during tool loops (issue #13) | b | R1 | issue #13: checkpoint at the current generation opener in tool loops; upstream still retains the first tail assistant; re-derive against the Jinja probe design |
| `ca3c7e26` | 2026-08-27 | feat(supervisor): detect prefix reuse collapse from host state slot saturation | d | SUP |  |
| `e734dfdd` | 2026-08-28 | fix(runtime): preserve long-anchor slot identity | a | R1 | upstream resource_manager.h has the identical long-anchor ordinal checks; its prefix real test covers the Program side |
| `35a27329` | 2026-08-28 | feat(serve): accept vllm template options | a | S1 | upstream normalizes chat_template_kwargs (openai_chat_request.cpp:805); fork schema tests re-checked in M2 |
| `8d7a4827` | 2026-08-28 | feat(supervisor): add production mission control | d | SUP |  |
| `6314ab02` | 2026-08-28 | feat(artifact): retain mapped tensor payloads | c | FA | v2 mapped tensor payloads; v3 needs its own Host-mapped path for PLE/MTP banks |
| `8a824a23` | 2026-08-28 | feat(artifact): register Flash-Next quantized formats | c | FA | Flash-Next quantized formats in the v2 registry |
| `823e0c08` | 2026-08-28 | docs(research): assess Flash-Next single-GPU route | d | DOC | Flash-Next single-GPU research note |
| `e52495e4` | 2026-08-28 | feat(artifact): pack NVFP4 expert banks | c | FA | NVFP4 expert bank layout |
| `f8b6414b` | 2026-08-28 | feat(convert): validate Flash-Next source bundle | c | FN |  |
| `6414b70c` | 2026-08-28 | feat(convert): stream Flash-Next mixed artifact | c | FN |  |
| `b8619496` | 2026-08-28 | fix(artifact): retain PLE indices as I64 | c | FA | PLE indices as I64 |
| `cbdbcd9c` | 2026-08-28 | test(artifact): smoke external native artifacts | c | FA | external artifact smoke test |
| `de357f46` | 2026-08-28 | feat(target): bind Flash-Next artifact and PLE | c | FN |  |
| `bc03fc80` | 2026-08-28 | feat(target): expose Flash-Next expert banks | c | FN |  |
| `f29b6b34` | 2026-08-28 | feat(target): materialize Flash-Next text weights | c | FN |  |
| `3a2b94b1` | 2026-08-28 | feat(target): pipeline Flash-Next PLE gathers | c | FN |  |
| `db2652e7` | 2026-08-28 | feat(target): execute Flash-Next routed MoE | c | FN |  |
| `12ea8ac5` | 2026-08-28 | feat(ops): execute Flash-Next FP8 projections | c | FN |  |
| `f2b82f03` | 2026-08-28 | feat(target): execute Flash-Next hyper connections | c | FN |  |
| `cf6294f7` | 2026-08-28 | feat(target): execute Flash-Next DeltaNet | c | FN |  |
| `2c9eade8` | 2026-08-28 | feat(target): select Flash-Next QSA blocks | c | FN |  |
| `4195ff8c` | 2026-08-28 | feat(target): execute Flash-Next QSA | c | FN |  |
| `90b61c93` | 2026-08-28 | feat(target): inject Flash-Next PLE | c | FN |  |
| `1573e96d` | 2026-08-28 | feat(target): compose Flash-Next text decode | c | FN |  |
| `7a673437` | 2026-08-28 | feat(target): own Flash-Next runtime capacity | c | FN |  |
| `993578cf` | 2026-08-28 | feat(target): transact Flash-Next text rounds | c | FN |  |
| `992e5b99` | 2026-08-28 | feat(target): load Flash-Next text artifacts | c | FN |  |
| `836d5607` | 2026-08-28 | feat(target): add Flash-Next full-residency probe | c | FN |  |
| `93ed50d6` | 2026-08-28 | feat(target): materialize Flash-Next vision weights | c | FN |  |
| `09274043` | 2026-08-28 | feat(ops): admit Flash-Next vision BF16 linear | c | FN | BF16 vision linear shape for Flash-Next |
| `51e549aa` | 2026-08-28 | fix(targets/qwen3_8_flash_next): pin PLE boundary ownership to token 248044 | c | FN |  |
| `5e39fc41` | 2026-08-28 | feat(targets/qwen3_8_flash_next): implement chat diagnostic reference tool mode | c | FN |  |
| `f362fd0f` | 2026-08-28 | refactor(vision): extract shared Qwen3 encoder | c | FN | shared Qwen3 vision encoder extraction for Flash-Next |
| `cda6759b` | 2026-08-28 | feat(qwen3_8_flash_next): implement multimodal vision execution and diagnostic | c | FN |  |
| `3b0fae41` | 2026-08-29 | fix(qwen3_8_flash_next): renormalize top-10 MoE routing probabilities (norm_topk_prob=true) | c | FN |  |
| `2fe5a855` | 2026-08-29 | feat(qwen3_8_flash_next): add teacher-forced first-divergence harness | c | FN |  |
| `ae578c1f` | 2026-08-29 | fix(qwen3_8_flash_next): gate the GDN output norm with sigmoid(z) | c | FN |  |
| `31c9db28` | 2026-08-29 | fix(qwen3_8_flash_next): make the oracle harness survive multi-token prompts | c | FN |  |
| `b6c54ab3` | 2026-08-29 | fix(qwen3_8_flash_next): synchronize before reading the sampled token | c | FN |  |
| `5d3d96cd` | 2026-08-29 | fix(qwen3_8_flash_next): fix B>1 planar MRoPE indexing and discriminate GDN sigmoid gate in test | c | FN |  |
| `0c2efc02` | 2026-08-29 | build: disable C++20 module dependency scanning | b | W1 | CMAKE_CXX_SCAN_FOR_MODULES OFF: re-done in M1 |
| `72522164` | 2026-08-29 | refactor(qwen3_8_flash_next): own the Qwen3.6-family frontend resources | c | FN |  |
| `ee55cbed` | 2026-08-29 | feat(qwen3_8_flash_next): register a compile-checked Package and Program skeleton | c | FN |  |
| `abe38e43` | 2026-08-29 | feat(qwen3_8_flash_next): serve text through the public Engine (cold-path Program) | c | FN |  |
| `2a3c364c` | 2026-08-29 | feat(qwen3_8_flash_next): chunked-prefill plumbing and BF16 T>8 linears | c | FN |  |
| `84e75ce8` | 2026-08-29 | fix(qwen3_8_flash_next): order the PLE host gather copy after compute-stream work | c | FN |  |
| `311cc0f5` | 2026-08-29 | test(qwen3_8_flash_next): run real-artifact cases only with NINFER_WEIGHTS set | c | FN |  |
| `e67974b5` | 2026-08-29 | feat(qwen3_8_flash_next): T-wide chunked prefill for the non-QSA blocks | c | FN |  |
| `4df6ccd4` | 2026-08-29 | fix(qwen3_8_flash_next): size the prefill-chunk workspace for the executor's staging tensors | c | FN |  |
| `e92e8224` | 2026-08-29 | fix(ninfer-supervisor): poll nvidia-smi without flashing a console window | d | SUP |  |
| `6d6fa410` | 2026-08-29 | feat(qwen3_8_flash_next): T-wide QSA indexer and attention in chunked prefill (5c-3) | c | FN |  |
| `252d9595` | 2026-08-30 | fix(qwen3_8_flash_next): barrier between the softmax max read and the probability write in QSA sparse attention | c | FN |  |
| `3f633e19` | 2026-08-30 | feat(qwen3_8_flash_next): CUDA-graph decode replay (sequence 7) | c | FN |  |
| `72f1d915` | 2026-08-30 | test(qwen3_8_flash_next): finite synthetic model, bit-exact graph equivalence, graph allowance in the memory report (7b) | c | FN |  |
| `b2d2af31` | 2026-08-30 | feat(qwen3_8_flash_next): vision on the served prefill path and continuation checkpoints (sequences 8 + 6) | c | FN |  |
| `25d304be` | 2026-08-30 | feat(qwen3_8_flash_next): turn-closure checkpoints for served multi-turn reuse (sequence 6e) | c | FN |  |
| `7e1acac5` | 2026-08-30 | fix(qwen3_8_flash_next): state-slot invariant and 64-round continuation check (sequence 6f) | c | FN |  |
| `0a495f93` | 2026-08-30 | fix(qwen3_8_flash_next): evict catalogued checkpoints under page-group pressure (sequence 6g, partial) | c | FN |  |
| `1e7eb28b` | 2026-08-30 | feat(qwen3_8_flash_next): Engine pressure-planning protocol for private continuations (sequence 6h) | c | FN |  |
| `3c6e41ec` | 2026-08-30 | fix(merge): make the upstream merge actually serve both targets (integration fixes) | e | OBS | integration fix for the 2026-08-30 merge; nothing to carry Dropped (Igor, 2026-09-27). |
| `ca7305e2` | 2026-08-30 | perf(qwen3_8_flash_next): high-occupancy fused hyper-connection chain (sequence 9a) | c | FN |  |
| `6243bcf3` | 2026-08-31 | fix(qwen3_8_flash_next): catalog-debt cleanup from 6h (sequence 6i) | c | FN |  |
| `dbc1e663` | 2026-08-31 | perf(qwen3_8_flash_next): warp-cooperative QSA prefill attention (sequence 9d) | c | FN |  |
| `c79c197f` | 2026-08-31 | perf(qwen3_8_flash_next): weight-stationary hyper-connection prefill (sequence 9e) | c | FN |  |
| `dff56e86` | 2026-08-31 | perf(qwen3_8_flash_next): grouped weight-stationary MoE prefill (sequence 9c) | c | FN |  |
| `24ee3d5a` | 2026-08-31 | perf(qwen3_8_flash_next): native-FP8 A8 prefill GEMMs on the FlashNext dispatch (sequence 9f) | c | FN |  |
| `b2eb5e81` | 2026-08-31 | test(qwen3_8_flash_next): repair vacuous QSA equivalence and the GDN stream race (sequence 9t) | c | FN |  |
| `693791b6` | 2026-08-31 | perf(qwen3_8_flash_next): PLE dequantization on the GPU for prefill chunks (sequence 9g) | c | FN |  |
| `e317af72` | 2026-09-01 | perf(qwen3_8_flash_next): QSA indexer identity bypass for fully-selected contexts (sequence G1) | c | FN |  |
| `4c2e62ad` | 2026-09-01 | perf(qwen3_8_flash_next): optional FP8 LM head behind EngineOptions (sequence G2) | c | FN |  |
| `3c808571` | 2026-09-01 | perf(qwen3_8_flash_next): Split-K hyper-connection down-projection (sequence G3) | c | FN |  |
| `1dc03c4f` | 2026-09-01 | feat(qwen3_8_flash_next): MTP runtime bring-up — materialization, draft step, parity oracle (sequence 10a) | c | FN |  |
| `92c82ae2` | 2026-09-01 | fix(qwen3_8_flash_next): up-projection partial-tile OOB stores + hardened HC gate (sequence G4) | c | FN |  |
| `a01fc7b8` | 2026-09-01 | perf(qwen3_8_flash_next): hybrid NVFP4 tensor-core MMA for MoE prefill (sequence 11) | c | FN |  |
| `79e862d0` | 2026-09-01 | perf(qwen3_8_flash_next): 2D warp-tiled MMA MoE schedule + overlap (sequence 11b) | c | FN |  |
| `0970406a` | 2026-09-01 | perf(qwen3_8_flash_next): deterministic DeviceTopK long-context indexer selection (sequence G5) | c | FN |  |
| `104d34dc` | 2026-09-01 | perf(qwen3_8_flash_next): context-bucketed decode CUDA graphs (sequence G6) | c | FN |  |
| `ddf69c87` | 2026-09-01 | perf(qwen3_8_flash_next): prefill select arm — segmented sort restored + identity bypass (sequence G7) | c | FN |  |
| `878d5d6c` | 2026-09-01 | test(qwen3_8_flash_next): synthetic 64k-context soak and determinism bisect harness (sequence G8) | c | FN |  |
| `144bc6af` | 2026-09-01 | fix(qwen3_8_flash_next): PLE reduction barrier race + prefill frontier selection (sequence G9) | c | FN |  |
| `177906f8` | 2026-09-01 | test(qwen3_8_flash_next): prefill select equivalence at long-context shapes (sequence G10) | c | FN |  |
| `7e8e6490` | 2026-09-01 | perf(qwen3_8_flash_next): tile prefill indexer scoring as a bf16 MMA GEMM (sequence G11) | c | FN |  |
| `1bb16ca1` | 2026-09-01 | test(qwen3_8_flash_next): per-token mask sentinel and score oracle for the prefill GEMM (sequence G12) | c | FN |  |
| `c41b2411` | 2026-09-01 | test(qwen3_8_flash_next): randomized-input score oracle for the prefill GEMM (sequence G13) | c | FN |  |
| `f9f60e10` | 2026-09-01 | test(qwen3_8_flash_next): 256k feasibility soak — full native context works (sequence G14) | c | FN |  |
| `d7ef1d55` | 2026-09-01 | feat(qwen3_8_flash_next): MTP speculative decoding verify/accept loop (sequence 10b) | c | FN |  |
| `e8c2c7a9` | 2026-09-01 | fix(qwen3_8_flash_next): reset the encoded-vision-item cache when a lane takes a new request (sequence G16) | c | FN |  |
| `52fe9062` | 2026-09-01 | feat(qwen3_8_flash_next): pressure-planning contract members and a load/admission/eviction soak (sequence M1) | c | FN |  |
| `3ea24b42` | 2026-09-01 | feat(qwen3_8_flash_next): NVFP4-quantize the MTP draft expert bank at load (sequence 12) | c | FN |  |
| `6c64eca4` | 2026-09-01 | perf(qwen3_8_flash_next): fuse the MoE down projection with a fixed-order reduction (sequence 14) | c | FN |  |
| `012b16e8` | 2026-09-01 | perf(qwen3_8_flash_next): tiled MMA QSA prefill attention behind a plan flag (sequence G17) | c | FN |  |
| `f379f3e6` | 2026-09-01 | chore(qwen3_8_flash_next): name the prefill shared-down kernel for what it is (sequence 15) | c | FN |  |
| `ef6c9df0` | 2026-09-01 | feat(qwen3_8_flash_next): expose the QSA MMA prefill flag to serve and the environment (sequence G18) | c | FN |  |
| `0ea7ecc3` | 2026-09-01 | test(qwen3_8_flash_next): arm the hyper-connection 25 us gate only under NINFER_PERF_GATES=1 (sequence G19) | c | FN |  |
| `4236f739` | 2026-09-01 | test(qwen3_8_flash_next): make the executor's synthetic model route like the real one (sequence 16) | c | FN |  |
| `9ce19e51` | 2026-09-01 | test(qwen3_8_flash_next): remove the remaining DC bias from the synthetic model so every layer routes to all experts (sequence 16b) | c | FN |  |
| `39322809` | 2026-09-01 | perf(qwen3_8_flash_next): unsplit the residual FP8 linear at prefill width, and an interleaved in-process A/B timer (sequence G20) | c | FN |  |
| `3abc3d02` | 2026-09-01 | feat(qwen3_8_flash_next): keep the MTP draft expert bank out of device memory and use a real per-expert divisor (sequence 12b) | c | FN |  |
| `6934481d` | 2026-09-02 | perf(qwen3_8_flash_next): env-gated CUDA-event stage ledger for the prefill chunk (sequence 18) | c | FN |  |
| `29c9fa76` | 2026-09-02 | perf(qwen3_8_flash_next): decide the QSA selection regime on the host instead of syncing twelve times per prefill chunk (sequence G21) | c | FN |  |
| `652e51c1` | 2026-09-02 | fix(qwen3_8_flash_next): bind the MTP expert banks as mapped tensors and always consume them (sequence 12b regression) | c | FN |  |
| `39ebfaa7` | 2026-09-02 | fix(qwen3_8_flash_next): restore the --qsa-prefill-mma serve flag dropped by the upstream merge | c | FN |  |
| `71bcb865` | 2026-09-02 | test(qwen3_8_flash_next): synthetic Flash-Next artifact fixture with the MTP objects, bound under every feature combination (sequence 12c) | c | FN |  |
| `b541f92e` | 2026-09-02 | perf(qwen3_8_flash_next): router, shared expert and grouping as tensor-core GEMMs on the prefill path (sequence G22) | c | FN |  |
| `b6069928` | 2026-09-02 | fix(qwen3_8_flash_next): report every checkpoint of an evicted owner from the pressure-planning session (sequence M3) | c | FN |  |
| `4d84cf8b` | 2026-09-02 | test(qwen3_8_flash_next): teacher-forced oracle logits through the prefill path, on both attention arms (sequence G23c) | c | FN |  |
| `8a304307` | 2026-09-03 | fix(qwen3_8_flash_next): make the capacity-curve rejection say what it wants, and let the oracle mode take a small context | c | FN |  |
| `ac25f5c6` | 2026-09-03 | perf(qwen3_8_flash_next): make the tensor-core QSA prefill attention the default | c | FN |  |
| `6ac70743` | 2026-09-03 | perf(qwen3_8_flash_next): pad the routed MoE scale rows to a stride coprime with the bank count (sequence 21 footnote) | c | FN |  |
| `11e0bb4e` | 2026-09-03 | perf(qwen3_8_flash_next): tensor-core PV and four-warp QK for the QSA prefill attention, behind a switch (sequence G24) | c | FN |  |
| `97d5ec2b` | 2026-09-03 | fix(qwen3_8_flash_next): order the speculative verifier's recurrent rows without serialising ordinary decode (sequence M2b) | c | FN |  |
| `cedd255c` | 2026-09-03 | perf(qwen3_8_flash_next): stage the routed MoE gate-up weights with cp.async and double-buffer the activations, behind a switch (sequence 21c) | c | FN |  |
| `98fa9c14` | 2026-09-03 | fix(qwen3_6): emit the pimpl move-assignment operators so every Engine link resolves | e | OBS | explicit pimpl move operators for the removed qwen3_6 template API; v3 has no such instantiations (M1 build shows no need) Dropped (Igor, 2026-09-27). |
| `11dcf0e6` | 2026-09-03 | build(third_party): compile spdlog with /utf-8 under MSVC | b | W1 | spdlog /utf-8: re-done in M1 |
| `b0c40582` | 2026-09-03 | fix(qwen3_6): name the tool-call parse fallbacks and allow trailing text behind a switch | b | T1 | named parse fallbacks + NINFER_TOOL_CALLS_ALLOW_TRAILING_TEXT |
| `fe724073` | 2026-09-03 | fix(qwen3_8_flash_next): add LRU eviction to continuation slots and inspect_capture feasibility (M6b) | c | FN |  |
| `a786be77` | 2026-09-03 | test(qwen3_8_flash_next): say what the 24k reuse test does and does not prove | c | FN |  |
| `5c23fe2e` | 2026-09-03 | fix(kv-capacity): enforce post-startup slack floor and pool coverage reconciliation (Sequence M4) | b | S3 | post-startup slack floor, pool coverage diagnostics, --kv-slack-floor-mib, --clamp-concurrency-to-pool |
| `bf1df85b` | 2026-09-03 | perf(qwen3_8_flash_next): measure device wait, which was being counted as host time | c | FN |  |
| `56ed7300` | 2026-09-03 | perf(qwen3_8_flash_next): optimize decode moe down kernel with vector loads, staged activations, up-front scales, and barrier-free loop (Sequence D1) | c | FN |  |
| `c46a328d` | 2026-09-03 | perf(moe): eliminate shared memory bank conflicts and staging overhead in flash_next_moe_down_kernel | c | FN |  |
| `93176f7c` | 2026-09-03 | fix(qwen3_8_flash_next): eliminate unilateral slot eviction and enforce invariant in pressure planning (D5) | c | FN |  |
| `35277b6b` | 2026-09-04 | fix(flash_next): fix remaining tokens argument in prefill work pricing and add 8-turn reuse regression test | c | FN |  |
| `9cb37a73` | 2026-09-04 | perf(qwen3_8_flash_next): raise prefill gate/up MMA occupancy with a 256-thread V2 kernel (D7) | c | FN |  |
| `c4f8f4c9` | 2026-09-04 | fix(qwen3_8_flash_next): name the state-slot ceiling and say what the operator can ask for | c | FN |  |
| `8264394c` | 2026-09-04 | fix(serve): dynamically format supported reasoning efforts in 400 error message (D11) | b | S1 | reasoning_effort 400 lists supported values |
| `6d7971af` | 2026-09-04 | feat(targets/qwen3_8_flash_next): implement cheap draft head for MTP speculative decoding | c | FN |  |
| `1fcd72b4` | 2026-09-04 | fix(targets/flash-next): clamp continuation capacity to state-slot ceiling and synchronize with Engine (D8) | c | FN |  |
| `eac4eba8` | 2026-09-04 | docs(runtime_plan): clarify defensive invariant comment following D8 clamp synchronization | c | FN |  |
| `35f22811` | 2026-09-04 | fix(qwen3_8_flash_next): derive draft floor slots from draft tokens and clamp catalog (D15) | c | FN |  |
| `a6ec8733` | 2026-09-04 | fix(qwen3_8_flash_next): eliminate vision workspace double-allocation | c | FN |  |
| `4eaf05f6` | 2026-09-04 | fix(qwen3_8_flash_next): cite Flash-Next 4u draft limit and add 0-continuation warning | c | FN |  |
| `10b7aeaa` | 2026-09-04 | feat(targets/flash-next): wire --kv-dtype fp8 through QSA attention and runtime plan (D13) | c | FN |  |
| `eaa21384` | 2026-09-04 | feat(flash-next): format-tolerant NVFP4/BF16 MTP expert binding (Sequence D17) | c | FN |  |
| `415572b0` | 2026-09-04 | feat(gdn): implement BF16 recurrent SSM state storage for Flash-Next (D18) | c | FN |  |
| `9bf5e8b8` | 2026-09-04 | feat(sizing): Sequence D21 - NVML device-wide memory sizing & desktop reserve floor | b | S3 | NVML device-wide memory query, desktop reserve floor (--desktop-reserve-gib/-mib), actionable admission memory errors |
| `a7950919` | 2026-09-04 | feat(spec): Sequence D20 - speculative telemetry wiring & acceptance fixtures | c | FN | Flash-Next MTP telemetry + acceptance fixtures |
| `a7c24564` | 2026-09-04 | feat(residency): Sequence D19 - lift logical KV and host extent stores to src/runtime/ | e | OBS | D19 lift of qwen3_6 KV stores into src/runtime; v3 owns these under models/qwen3_5/program/storage Dropped (Igor, 2026-09-27). |
| `b2c12970` | 2026-09-04 | feat(head): Sequence D17 - wire FP8 output head options & divergence harness | c | FN | FP8 output head flags; only Flash-Next honors them |
| `362aa76c` | 2026-09-04 | revert: back out the D19 residency lift until its own test passes | e | OBS | revert of a7c24564 Dropped (Igor, 2026-09-27). |
| `7de1c559` | 2026-09-04 | feat(runtime): lift logical and host KV stores into namespace ninfer::runtime (D19 re-delivery) | e | OBS | D19 re-delivery; structure superseded by v3. Its DeviceArena-backing fix is checked against v3 storage in M2 Dropped (Igor, 2026-09-27). |
| `eab7f1b8` | 2026-09-04 | feat(embed): wire FP8 token embedding quantizer & extend divergence harness | c | FN | FP8 token embedding flag; only Flash-Next honors it |
| `2a606c22` | 2026-09-05 | feat(supervisor): Docker-style tray menu with desktop reserve, idle unload and device-wide memory | d | SUP |  |
| `c1fb8c88` | 2026-09-05 | perf(flash-next): path-per-warp MoE down projection at decode, bitwise-identical | c | FN |  |
| `236ca4c1` | 2026-09-05 | perf(flash-next): fuse group_norm into the hyper-connection low-rank kernel at decode | c | FN |  |
| `12a571ce` | 2026-09-05 | perf(flash-next): drop the dead per-commit slot upload and inline small PLE gathers | c | FN |  |
| `b5ef28c4` | 2026-09-05 | fix(flash-next): guard the MTP draft's empty-selection attention and export prefill hyper-hidden | c | FN |  |
| `9ab7b05b` | 2026-09-05 | fix(flash-next): --output-head-fp8 / --token-embedding-fp8 now free memory instead of costing it | c | FN |  |
| `a14a320c` | 2026-09-05 | feat(supervisor): make the tray feature-complete and fix the reserve that never reached the engine | d | SUP |  |
| `aaa809bc` | 2026-09-05 | fix(flash-next): capture every decode graph at startup; the lazy path allocated 5 GiB at runtime (D23) | c | FN |  |
| `342388e4` | 2026-09-05 | perf(flash-next): fuse the MoE router projection and top-10 selection into one launch | c | FN |  |
| `f704502a` | 2026-09-05 | feat(serve): add GET /admin/vram, and delete the release handlers that never existed | b | S2 | GET /admin/vram with NVML device view |
| `e9503026` | 2026-09-05 | feat(supervisor): configuration editor in the dashboard, and remove what could not work | d | SUP |  |
| `6a181abf` | 2026-09-05 | fix(serve): authenticate the engine, and report a decode rate that exists | b | S2 | --api-key-file (upstream has --api-key only); decode-rate report; supervisor half -> SUP |
| `1cb3216d` | 2026-09-05 | wip(supervisor): dashboard rewrite, GPU process attribution, reserve budget | d | SUP | supervisor code carried; propose dropping committed data (supervisor-logs-demo/ 13 MB, bench/d20_results*/) |
| `00ad2b86` | 2026-09-05 | feat(engine): quiescence fence and a device buffer whose backing can shrink | b | S2 | Engine::run_at_quiescence + POST /admin/quiesce; SparseDeviceBuffer is used only by Flash-Next (carried in M3) |
| `bd9ca85f` | 2026-09-05 | diag(flash-next): report WHERE a prefix-reuse candidate stops matching | c | FN |  |
| `cd62f265` | 2026-09-05 | fix(supervisor): restart an engine that is alive but no longer answering | d | SUP |  |
| `f426d2c4` | 2026-09-05 | fix(serve): stop /admin/vram taking the engine's execution lock | b | S2 | Engine::try_memory_summary so telemetry never waits on execution |
| `aec32ee0` | 2026-09-05 | fix(engine): stop advertising continuation capacity no StateImage can back | b | R1 | continuation capacity bounded by backable StateImages |
| `b137af03` | 2026-09-05 | fix(serve): clamp top_k to the sampler's candidate domain instead of refusing | b | S1 | clamp top_k > 20 instead of 400 |
| `bdb78d84` | 2026-09-05 | fix(supervisor): record why a restart happened where the reason survives | d | SUP |  |
| `9a36594a` | 2026-09-05 | feat(supervisor): model catalog with availability checks and a switch that rolls back | d | SUP |  |
| `316a1b51` | 2026-09-05 | fix(supervisor): refuse to overwrite a config edited while it was held | d | SUP |  |
| `97166fd3` | 2026-09-05 | chore(supervisor): example config with a model catalog, and a browser test client | d | SUP | example config + browser console |
| `704860f4` | 2026-09-06 | fix(supervisor): report a model switch's outcome from observation, not intention | d | SUP |  |
| `9ed89699` | 2026-09-06 | feat(supervisor): a model picker in the dashboard, locked while a switch is in flight | d | SUP |  |
| `5c71c27d` | 2026-09-06 | fix(supervisor): stop the rollback message from naming a cause it does not know | d | SUP |  |
| `2f375231` | 2026-09-06 | feat(serve): record which client sent each request | b | S2 | request_start.client from User-Agent |
| `019ca3ab` | 2026-09-06 | feat(supervisor): report generation speed as a total, and chart it over time | d | SUP |  |
| `b283af4a` | 2026-09-06 | refactor(supervisor): give the overview three bands instead of nine equal slabs | d | SUP |  |
| `c7e64f07` | 2026-09-06 | fix(supervisor): refresh the config stamp after writing, so it stops refusing its own edits | d | SUP |  |
| `d45b1871` | 2026-09-06 | fix(qwen3_6): a badly written tool argument no longer discards the whole call | b | T1 | typed tool argument that does not parse is passed through as text |
| `3400f9c9` | 2026-09-06 | feat(serve): fingerprint the tool block on every request | b | S2 | request_start.tools_digest |
| `fd6880bf` | 2026-09-06 | feat(supervisor): show which apps are using the engine, and how well each reuses | d | SUP |  |
| `48c11931` | 2026-09-06 | docs: implementation brief for finishing MTP speculation on Flash-Next | e | OBS | completed Flash-Next MTP implementation brief (plan doc) Dropped (Igor, 2026-09-27). |
| `1941ff01` | 2026-09-06 | fix(flash-next): complete MTP state and improve decode and prefill | c | FN | MTP state completion; small host_memory/frontend-test helpers travel with it |
| `039bef66` | 2026-09-06 | fix(serve): avoid deadlock during cold telemetry refresh | b | S2 | device snapshot cache: telemetry never deadlocks HTTP workers |
| `bb7b7305` | 2026-09-07 | feat(flash-next): prefix-reuse stability, split-attention decode, native structured output | b | G1 | three concerns: XGrammar structured output (G1), checkpoint capacity as admission resource + observation carry-over + publication grace (R1), selected-block split attention (FN) |
| `fa3136fb` | 2026-09-07 | docs(flash-next): profiling results, structured-output serving notes, reuse acceptance probes | d | DOC | Flash-Next profiling + reuse probes |
| `c81c88f6` | 2026-09-07 | docs: describe workstation fork direction and capabilities | d | DOC | README fork direction |
| `497a051f` | 2026-09-07 | feat(supervisor): adapt KV capacity to the reservation the engine can get, show the running plan, bound the series file | d | SUP | KV adaptation to the available reservation |
| `8e1e71bb` | 2026-09-08 | feat(windows): install standalone NInfer app with native icons | d | W2 | standalone Windows app with icons |
| `436b0dde` | 2026-09-10 | fix(core): complete host uploads before returning | a | UP | patch-equal upstream b88c0f6f |
| `b2513be7` | 2026-09-10 | perf(text): skip NFC normalization for ASCII | a | UP | patch-equal upstream 641ef3e7 |
| `2b515a4d` | 2026-09-10 | perf(frontend): use a flat BPE merge table | a | UP | patch-equal upstream b158afe2 |
| `a09b91f0` | 2026-09-10 | perf(flash-next): remove the small-prefill cliff | c | FN |  |
| `b802e911` | 2026-09-10 | test(serve): align degraded tool argument expectation | b | T1 | test expectation for d45b1871 |
| `6d9bb536` | 2026-09-11 | docs(readme): link published Flash-Next Hugging Face artifact | d | DOC |  |
| `ddeeec19` | 2026-09-12 | feat(engine): integrate Qwen3.8-27B DFlash2 serving | a | D2 | upstream DFlash2 closure (385b30ce and ops through a16b6442) plus b8786751, 03177b91, e51b585c. Residue to carry: StateImage-fork admission block, all-constrained width-one verify (G1), 27B DFlash2 real test, supervisor presets (SUP) |
| `91ce2f2c` | 2026-09-12 | feat(models): support OrcaRouter Qwen3.8-27B NVFP4 | b | O1 | v3 needs no identity registration; carry the converter recipe (BF16 embedding + full head, all special tokens in the proposal head), tokenizer resource handling, BF16 Linear/LinearTopK qualification, supervisor preset |
| `534ec75b` | 2026-09-12 | fix(windows): unblock Qwen3.8 builds and media benchmarks | b | W1 | Patrick PR #20: bench support + test CMake on Windows; README/docs -> DOC |
| `673853f6` | 2026-09-15 | feat(supervisor): speculative draft-acceptance insight (issue #16) | d | SUP |  |
| `cb8723ef` | 2026-09-15 | feat(supervisor): context and KV capacity pressure insight (issue #18) | d | SUP |  |
| `7f6ae7cc` | 2026-09-15 | perf(runtime): port upstream materialization search and adaptive planning budgets | a | UP | port of upstream d4929686 (in base). Its Windows parts are re-done in M1; its Flash-Next pressure-session contract moves with M3 |
| `90ca8509` | 2026-09-06 | perf(ops): prefetch the shared-expert down weights from the D1 tail | a | UP | patch-equal upstream ce954918 |
| `f43f1794` | 2026-09-07 | perf(ops): warm L2 for the next projection from the MoE down tail, issuing the hint before the block barrier | a | UP | patch-equal upstream 7f14d963 |
| `5e4a66d0` | 2026-09-10 | fix(build): include bf16 definitions in q4 topk kernel | a | UP | patch-equal upstream 9f0575bb |
| `bb88a776` | 2026-09-18 | fix(serve): preserve json schema declaration order and enable automatic prefix cache reuse | b | S1 | schema declaration order (#24) is upstream too (request_json.h ordered_json); carry automatic prefix-cache discovery by default (#23) and the tool_choice=required grammar (G1) |
| `f26ab57d` | 2026-09-19 | fix(runtime): skip owners in publication grace when listing shared-capture pressure candidates | b | R1 | graced owners skipped in shared-capture pressure candidates |
| `f596a68e` | 2026-09-19 | feat(runtime): host readout of per-token log probabilities from a BF16 logit column | b | L1 | host logprob readout |
| `2c307887` | 2026-09-19 | feat(serve): per-token log probabilities on chat completions | b | L1 | chat logprobs/top_logprobs |
| `da87a3f5` | 2026-09-19 | feat(serve): closed-set scoring endpoint and read-only cache participation | b | L1 | /v1/score + prompt_cache_read_only |
| `db1a3694` | 2026-09-19 | perf(ops): token-fast NVFP4 W4A4 TMA schedule on the desktop RTX 5090 | b | P1 | upstream ee9d5192 (in base) made token-fast raster unconditional; the fork keeps weight-fast on RTX PRO 6000 (741.4 -> 749.9 ms regression). Re-measure on v3 |
| `6b147eef` | 2026-09-19 | fix(supervisor): keep the "Start at login" entry across installs | d | SUP |  |
| `529f85b8` | 2026-09-19 | fix(runtime): retain a reused private checkpoint for read-only and declared-boundary requests | b | R1 | retain reused private checkpoint for read-only/declared-boundary requests |
| `841f8361` | 2026-09-19 | feat(flash-next): token logprobs, read-only cache participation, declared-boundary checkpoints | c | FN |  |
| `c4288317` | 2026-09-19 | fix(serve): score branches one at a time from an explicit-only published prefix | b | L1 |  |
| `a0b43ad4` | 2026-09-19 | test(runtime): a read-only request leaves the catalog unchanged | b | L1 | read-only isolation test |
| `05797621` | 2026-09-19 | feat(ops): candidate_logprobs, raw and mask-renormalised log-probabilities of listed tokens | b | L1 | ops::candidate_logprobs + FP64 oracle test |
| `3181627f` | 2026-09-19 | perf(runtime): device readout of token logprobs, speculative rounds kept for scored requests | b | L1 | device readout, speculative rounds kept |
| `e17813cf` | 2026-09-20 | feat(runtime): log probabilities at prompt positions | b | L1 | prompt-position logprobs |
| `d49a8e0c` | 2026-09-20 | feat(serve): score candidates that span several tokens as continued assistant turns | b | L1 | multi-token score as continued assistant turns |
| `a29f97e2` | 2026-09-20 | fix(runtime): empty the context cache instead of terminating when an idle Engine cannot plan | b | R1 | idle-Engine planner failure empties the cache instead of terminating |
| `282f013c` | 2026-09-20 | test(frontend): continued assistant turn opens like the generation prompt | b | L1 | continued assistant turn renders the empty think block |
| `d2982875` | 2026-09-21 | perf(runtime): prefill read-only requests without rewrite-frontier splits | b | R1 | read-only prefill without rewrite-frontier splits |
| `60904080` | 2026-09-21 | bench(jevbench): native adapter, TypeSafe shim and public-set driver for JevBench | d | JEV |  |
| `b731bc06` | 2026-09-21 | build(flash-next): include <algorithm> where std::max over an initializer list is used | c | FN |  |
| `cafb09e5` | 2026-09-21 | build(flash-next): declare dynamic shared memory with __align__ as the other kernels do | c | FN |  |
| `a77d4ed5` | 2026-09-21 | build(linux): CUDA 13.3 container images and size_t-explicit checked arithmetic | d | BEN | Linux Dockerfile on CUDA 13.3; Flash-Next checked-arithmetic part goes with M3 |
| `3d6abdf8` | 2026-09-21 | bench(jevbench): the TypeSafe shim refuses to share its port | d | JEV |  |
| `df3d0bc7` | 2026-09-21 | docs(readme): describe the decision readout, read-only participation and the JevBench evaluation | d | DOC |  |
| `d6ecb8f2` | 2026-09-21 | docs(design): dual-purpose decode explainer page with its own design record | d | DOC |  |
| `97200f2b` | 2026-09-21 | docs(model-cards): Qwen3.8-Flash-Next mixed and Qwen3.8-27B OrcaRouter NVFP4 release cards | d | DOC | model cards |
| `edf0749f` | 2026-09-21 | feat(windows): installer build script, Inno Setup definition and first-launch application | d | W2 | installer script, Inno Setup, first-launch app |
| `21a49729` | 2026-09-21 | docs(research): R610 driver and CUDA Tile evaluation, upstream review of 2026-09-07 | d | DOC | R610/CUDA Tile research; its upstream review is superseded by this ledger |
| `12550e28` | 2026-09-21 | bench(vision): sweep input image resolutions against a served engine and read the request log | d | BEN | vision resolution sweep |
| `383ce91e` | 2026-09-21 | docs(design): decision-analyzer recording page for the closed-set readout | d | DOC |  |
| `e20e7e23` | 2026-09-21 | feat(serve): add TypeSafe AI System One decision endpoint and docs | b | Y1 |  |
| `147370d7` | 2026-09-21 | fix(serve): keep System One choices on native tokens | b | Y1 |  |
| `95f14958` | 2026-09-24 | perf(flash-next): batch hyper-connection decode across tokens | c | FN |  |
| `4cd6b5f4` | 2026-09-24 | perf(flash-next): pick the MoE decode down kernel by token count | c | FN |  |
| `ff20cd06` | 2026-09-24 | feat(serve): System One image observations and /admin/stats | b | Y1 | System One images; GET /admin/stats (S2) |
| `15f0c5aa` | 2026-09-24 | feat(engine): opt-in reasoning feature readout for router training | b | Q1 | capture_reasoning_features + ninfer-reasoning-collect app |
| `07b3152c` | 2026-09-24 | feat(arcade): System One decision arcade demos and evaluators | d | ARC |  |
| `496f7366` | 2026-09-24 | bench(jevbench): paired decision study and synthetic corpus tools | d | JEV |  |
| `fe00a432` | 2026-09-24 | chore: ignore Playwright MCP session logs | d | DOC | .gitignore |
| `d2289abf` | 2026-09-24 | build(windows): bundle an LGPL FFmpeg for Vision | b | W1 | LGPL FFmpeg default: re-done in M1 |
| `59fe4a4b` | 2026-09-24 | feat(router): concurrent, derive-2048 and prune-tolerant Colab collection | d | RTR |  |
| `7de49d47` | 2026-09-24 | fix(router): sticky terminal state, verified sync, newline-only JSONL; document Colab collection | d | RTR |  |
| `ebab451f` | 2026-09-24 | fix(router): install CUDA 13.3 on Colab images that ship 12.8 | d | RTR |  |
| `48a95e40` | 2026-09-24 | feat(router): reuse a prebuilt collector on replacement Colab VMs | d | RTR |  |
| `8e145703` | 2026-09-24 | fix(router): ship libcudart with the prebuilt collector | d | RTR |  |
| `491baac9` | 2026-09-24 | feat(router): resume Colab collection across reclaimed sessions | d | RTR |  |
| `1c2f6bff` | 2026-09-24 | fix(router): utf-8 campaign output on Windows | d | RTR |  |
| `d3ad3d97` | 2026-09-25 | docs(router): 27B outcomes and router study on the synthetic corpus | d | RTR |  |
| `99f11130` | 2026-09-25 | docs(router): count derived actions in the token total | d | RTR |  |
| `2cbdb2b6` | 2026-09-25 | perf(engine): keep MTP drafting under structured output | b | G1 | MTP drafting under structured output (per-column masks) |
| `9d2e6d11` | 2026-09-25 | feat(router): direct-answer letter confidence, fold rotation and a confidence gate | d | RTR |  |
| `2af5fb5b` | 2026-09-25 | docs(router): rotating holdouts and direct-answer confidence results | d | RTR |  |
| `bbe3e16e` | 2026-09-25 | perf(supervisor): read device memory through one persistent NVML session | d | SUP |  |
| `87812bc8` | 2026-09-26 | feat(serve): make /v1/systemone a drop-in for TypeSafe Jev clients | b | Y1 | Jev drop-in contract |
