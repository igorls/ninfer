# Upstream ports

Updated: September 29, 2026.

This is the living record of how this workstation fork ([igorls/ninfer](https://github.com/igorls/ninfer))
tracks [Neroued/ninfer](https://github.com/Neroued/ninfer): the reviewed upstream point, the fork
features carried onto upstream v3, deliberate divergences, dropped and deferred work, qualification
evidence. The first priority is **Qwen3.8-27B NVFP4 on native Windows and the RTX PRO
6000 Blackwell**. RTX 5090 results remain useful evidence for that device; they do not qualify the
workstation.

## Sync state and method

| Item | Value |
|---|---|
| Fork line | `workstation` (named `port/upstream-v3` until September 28, 2026), based on upstream [`bace20dc`](https://github.com/Neroued/ninfer/commit/bace20dc70249eed6402b66d4852c6c3f9612905) (September 24, 2026) and merged with upstream `e31bc99b` in `e2a149f1` and `d44ab584` in `d29866c9` |
| Last reviewed upstream commit | [`d44ab584`](https://github.com/Neroued/ninfer/commit/d44ab58408aa389728cd8b1ee50179527e1f3e0d) (September 29, 2026), the merge-base |
| Pre-v3 fork line | `research/qwen4-flash-next` at `87812bc8` (September 26); the source for Flash-Next |
| Not yet reviewed | `d44ab584..68c54356`: 11 upstream commits as of October 7, 2026, including the context-cache replacement `b9114396` |

The fork syncs by merging `origin/master` into the fork line, never by rebasing or cherry-picking
upstream work. Each merge records every upstream change it brings in here as integrated, deferred
or reverted, so the merge-base is the reviewed point. A rejected upstream change is reverted in the
fork line (or device-gated) with its measurement recorded; a deferred one names the condition that
would make it useful.

## How to maintain this document

Update an entry when its implementation or evidence changes. Record the upstream source, affected
model and workload, the fork commit, qualification and remaining work. Status values:

- **Candidate:** reviewed; implementation and local qualification remain.
- **In progress:** adaptation or qualification has started; state the unfinished acceptance work.
- **Integrated:** present in the fork line, with the commit and the actual qualification scope.
- **Deferred:** keep the reason and the condition that would make the change useful.
- **Reverted:** removed or device-gated in the fork line, with the measurement.

Source integration and hardware qualification are separate claims. Detailed performance results live in
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
([measurement](../performance/rtx-pro-6000.md#v3-port-against-the-previous-release-build-2026-09-28)).

## Upstream `bace20dc..e31bc99b` (merged September 28, 2026)

The fork line merged upstream `e31bc99b` in `e2a149f1`, the first real merge, so the merge-base now
records the reviewed point. The merge commit holds only the three conflict resolutions: upstream
moved `nvfp4_w4a4_tma.cuh` to `nvfp4_a4_tma.cuh`, and the fork's 64-byte descriptor alignment moved
with it; the GDN Replay Record and Replay Fold tests keep the fork's non-`constexpr` scale. Two
follow-ups made the merged tree build and run on the fork's surfaces: `24e04e99` gives upstream's
new BF16 TMA descriptors the same 64-byte alignment (MSVC error C2719), and `267a201a` routes the
fork's BF16 `[248320,5120]` head (OrcaRouter) through the unified BF16 schedules of `[14336,5120]`,
since the old per-shape templates are gone.

In the 27B production artifact, NVFP4 serves MLP layers 0–55; FP8 serves every attention and GDN
projection, MLP layers 56–63, the output head and the embedding; Q8 serves MTP and DFlash2; Q4, Q5
and Q6 serve Vision, and Q4 the optimized proposal head used by `--lm-head-draft`.

| Upstream | Subject | Verdict |
|---|---|---|
| `229c1832` | unify Q6 templates, add sliced-K MMA | Integrated: Vision patch embedding; Q6 Linear oracle test |
| `42614c0e` | unify Q4 schedules, tune sliced-K dispatch | Integrated: Vision and the optimized proposal head (in every MTP decode); Q4 Linear, LinearAdd and LinearSwiGLU oracle tests |
| `fc62790a` | unify Q5 templates, tune sliced-K Linear | Integrated: Vision; Q5 Linear and LinearAdd oracle tests |
| `502cd9d6` | unify Q8 templates, tune sliced-K schedules | Integrated: MTP and DFlash2 drafter; Q8 Linear, LinearAdd, LinearPair and LinearSwiGLU oracle tests, `dflash2_real` |
| `5d08cba8` | unify FP8 templates, preserve FP32 SwiGLU fusion | Integrated: attention/GDN projections, late MLP, head; FP8 A16/A8 Linear, LinearAdd and LinearSwiGLU oracle tests |
| `fc3993d8` | unify NVFP4 templates, add A16 MMA routes | Integrated with the 64-byte TMA alignment carried in the merge: MLP layers 0–55; NVFP4 A16/A4 Linear, LinearAdd, LinearSwiGLU, attention- and GDN-input oracle tests |
| `ecbc3357` | expand BF16 templates, unify epilogues | Integrated with `24e04e99` and `267a201a`: the OrcaRouter head (the production artifact's decode runs no BF16 Linear kernel); BF16 Linear (including `[248320,5120]`) and LinearAdd oracle tests, OrcaRouter real-model tests |
| `0784e76f` | two-stage GDN chunked path | Integrated: every 27B prefill (48 GDN layers); GDN, Replay Record and Replay Fold oracle tests |
| `619e3f4c` `6d333ce0` `d4ea63ea` `71a1cb0e` | KDA recurrent, batch and chunked paths | Integrated, not exercised: no fork model uses Kimi Delta Attention; its oracle test passes on MSVC and the PRO 6000 |
| `b24a439f` `2ddef207` | reporting and completion rules | Integrated into AGENTS.md beside the fork's upstream-ports rule |
| `e31bc99b` | Linear guidance, Q4 performance report | Integrated (documentation; the report is upstream's RTX 5090 data) |

Qualification on Windows (MSVC 19.51, CUDA 13.3, `sm_120a`, RTX PRO 6000, driver 616.92):

- Full CTest, 138 tests: 129 passed, 9 skipped (artifact- or source-gated), 0 failed. The pre-merge
  build (`b84c4d86`) had 137 tests: 128 passed, 9 skipped, 0 failed. The new test is KDA's.
- Every NVFP4, FP8, BF16 and Q-format Linear, LinearAdd, LinearPair and LinearSwiGLU oracle test,
  the attention- and GDN-input projection tests, and the GDN, Replay Record, Replay Fold and KDA
  tests pass against their independent oracles.
- Real-model tests on the v3 production artifact give the same results before and after the merge:
  loading (MTP with Vision, DFlash2, scoring), causal scoring, reasoning features, the Vision
  workspace, `dflash2_real` (K=15 and K=7 at 8 rows) and twelve `prefix_real` scenarios pass; the
  default `prefix_real` run stops at the known Host-restore check (below). On the OrcaRouter artifact, loading, causal
  scoring, reasoning features and stream observations pass.
- Production-flag A/B against the previous release build, clean rounds: 7,680-token
  prefill 663.4 vs 689.7 ms (−3.8%), 256-token MTP decode 149.1 vs 148.5 tok/s (equal within
  noise), with the same MTP acceptance
  ([measurement](../performance/rtx-pro-6000.md#upstream-e31bc99b-merge-against-the-previous-release-build-2026-09-28)).

## Upstream `e31bc99b..d44ab584` (merged September 29, 2026)

Branch `sync/upstream-d44ab584`. The merge commit `d29866c9` holds only two conflict resolutions,
both in the Program's prefill context. Upstream `4201b5d2` binds DFlash prefill controls per chunk
(a `DFlashPrefillIngress` with the chunk's KV table row, in place of the shared decode ingress), and
the fork had appended its first-token logits capture and first-token and prompt readouts
(`737b570a`, `f290c8e7`) to the same `PrefillContext`. The resolution keeps both, upstream's fields
first: `program/context.h` declares the union, and `program/prefill.cpp`'s `advance_prefill`
initialises the DFlash row as upstream does (0, rebound per chunk) and keeps the fork's readout
wiring. The other `PrefillContext` initialisers (causal scoring, forced-token append) merged
cleanly and match the field order. Follow-ups: `183cdca6` gives upstream's new FP8 TMA descriptors
the fork's 64-byte alignment (MSVC error C2719 in every FP8 Linear, LinearAdd, SwiGLU and
attention-input unit; Linux code unchanged), and `5482fd99` reverts `1cfdb4d6`.

In the production artifact, FP8 KV makes the FP8 causal attention routes the production attention
path; FP8 serves every attention and GDN projection, MLP layers 56–63 and the output head; NVFP4
serves MLP layers 0–55. The `nvfp4full` profile pairs with NVFP4 KV on an RTX 5090.

| Upstream | Subject | Verdict |
|---|---|---|
| `23b0997d` `98ba2dac` | reorganize and tune BF16 causal attention; stabilize its graphs | Integrated: BF16 KV (not a production configuration); attention oracle test (all KV dtypes), `prefix_real` attention scenario with BF16 KV |
| `4e8939d6` `5a15a166` | reorganize and tune FP8 causal attention; unify its graphs and query tiling | Integrated: the production FP8-KV attention; oracle test, `prefix_real` attention with FP8 KV (MTP3 and MTP5), G4 A/B |
| `20a36378` | organize and tune INT8 causal attention | Integrated: not a production KV; oracle test, `prefix_real` attention with INT8 KV |
| `1192ad76` | organize and tune NVFP4 causal attention | Integrated: the NVFP4-KV attention (`nvfp4full` profile); oracle test, `prefix_real` attention with NVFP4 KV, G4 `nvfp4full` A/B (neutral) |
| `a637f28f` | organize and tune K8V4 causal attention | Integrated: not a production KV; oracle test, `prefix_real` attention with K8V4 KV |
| `1737ca11` | share causal attention primitives | Integrated (refactor of the five above) |
| `a012e2bc` | align attention qualification and engine graph planning | Integrated: graph profiles and startup planning; the nvfp4/k8v4 attention CTest aliases fold into the one attention test; every `prefix_real` scenario, G4 A/B |
| `582c9a8f` | varied benchmark inputs, restored mutable operands | Integrated, not exercised (Op benchmarks are not built by the fork); `bench_fixtures` test passes |
| `909fb087` `344d69b8` `7489500d` `b3f018ab` `7ede9b44` | FP8 Linear TMA split-K and schedule tuning at `[34816,5120]`, `[14336,5120]`, `[16384,5120]`, `[5120,6144]`, `[5120,17408]` | Integrated with `183cdca6`: the production FP8 projections and late MLP; FP8 A16/A8 Linear oracle tests; the series brings most of the 7,680-token prefill gain |
| `40bfe7dc` | FP8 fused projections (attention/GDN input, LinearAdd, SwiGLU) with TMA split-K | Integrated: FP8 LinearAdd, LinearSwiGLU, attention- and GDN-input and conv-snapshot/record oracle tests |
| `7f6aafed` | split-KV prefill for FP8 and K8V4 KV | Integrated: long-context prefill with FP8 KV (−19.5% at 61,625 tokens); oracle test, G4 A/B; device memory unchanged at `--prefill-chunk 2048` |
| `c1c48a6a` | configurable KV dtype in the serve benchmark runners | Integrated, not exercised (upstream's benchmark tooling) |
| `d23835c1` | Qwen3.8 FP8 KV serving results | Integrated (documentation; upstream's RTX 5090 data) |
| `84cf93e4` | native FP8-to-BF16 conversion (CUDA 13.2+) | Integrated: exact; neutral on G4 decode (217.5 vs 217.7 tok/s with and without it); FP8 A16 oracle tests |
| `1cfdb4d6` | native NVFP4 A16 decoding (CUDA 13.2+) | **Reverted** (`5482fd99`): exact, but the production decode on the RTX PRO 6000 G4 falls from 230.9 to 217.7 tok/s (−5.7%) with it, prefill unchanged; the effect on the RTX 5090 is unmeasured and one build serves both devices |
| `4201b5d2` | bind DFlash prefill controls per chunk | Integrated through the merge resolution: `dflash_prefill_real` (new upstream test) and `dflash2_real` at K=15 (8 rows) and K=7 with FP8 and NVFP4 KV pass |
| `d44ab584` | extend grouped small prefill to every KV dtype | Integrated: attention oracle test, G4 A/B |

Qualification:

- **Windows** (MSVC 19.51, CUDA 13.3, `sm_120a`, RTX PRO 6000, driver 616.92): full CTest, 138 tests: 128 passed, 10 skipped (artifact- or source-gated),
  0 failed. The pre-merge build had 138 tests with 129 passed and 9 skipped. The merge removes the
  two attention aliases (`--nvfp4-only`, `--k8v4-only`; the main attention test now covers all
  five KV dtypes), and adds `bench_fixtures` (passes) and `dflash_prefill_real` (skipped without an
  artifact). The attention test passes against its FP64 oracle for BF16, FP8, INT8, NVFP4 and K8V4
  KV, as do every FP8 and NVFP4 Linear, LinearAdd and LinearSwiGLU test, the attention- and
  GDN-input projection tests, the KV append tests and the speculative-round and masked-block tests.
  The run was repeated on the tip with the revert.
- **Real-model tests on Colab G4** (Linux build of `183cdca6`, `NINFER_TEST_ARTIFACT` = the
  production artifact from Hugging Face): loading (MTP, Vision, optimized proposal head), causal
  scoring, reasoning features, `dflash2_real` (K=15 at 8 rows: 21 of 23 drafted tokens accepted;
  K=7 at 2 rows with FP8 and with NVFP4 KV: 20 of 20), `dflash_prefill_real` (DFlash2), the
  `prefix_real` attention scenario with each of the five KV dtypes (MTP3, and FP8 with MTP5), and
  the twelve other `prefix_real` scenarios run one by one all pass. The default `prefix_real` run
  stops at the known Host-restore check with the byte-identical message and counters recorded on
  the pristine base and the previous merge (open question below).
- **Speed** (G4, production flags, alternating arms): the tip prefills 10.5% faster at 7,680 tokens
  and 19.5% faster at 61,625 tokens and decodes 1.8% faster than the previous release build (`28c40898`); without the
  revert the merge decoded 4.0% slower. The `nvfp4full` configuration (`nvfp4full`, NVFP4 KV, MTP3)
  is unchanged within 0.2%, with identical device memory
  ([measurement](../performance/rtx-pro-6000.md#upstream-d44ab584-merge-against-the-previous-release-build-2026-09-29-colab-g4)).

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
| Hidden row export | fork-native on `research/decision-head` (October 2026) | pending | `CausalScoreReadout::capture_hidden_rows` on the scoring route, `ninfer-hidden-export`; training readout for the decision head plan |
| OrcaRouter NVFP4 | `91ce2f2c` | `e35b663d` `9a9a1823` `74c5343f` `7130a888` | BF16 `[248320,5120]` Linear and LinearTopK, Qwen2-style tokenizer resources, conversion recipe (re-converted, not upgraded) |
| Supervisor, Windows app, installer | SUP and W2 commits, `bbe3e16e` | `168bdf12` `fc9a1ebe` `e92c2078` | The model catalog reads the v3 header and `metadata.name`; a v2 artifact is listed as needing the upgrade |
| Arcade, JevBench, router, probes | ARC, JEV, RTR, BEN commits | `a1f8b2d9` `6dfd01b8` `7c8555cb` | |
| Docs, model card | DOC commits, `97200f2b` | `174fdb87` `b80c497b` | Fork README; the OrcaRouter card describes the published v2 artifact |

The port missed nine documents, which were carried unchanged on 2026-09-28 (only links to moved
v3 paths were updated): the research records `docs/research/jevbench-qwen-intelligence-dossier.md`,
`learned-reasoning-router.md`, `qwen3.8-27b-decision-study.md`,
`qwen3_8_27b_derivative_artifacts.md` and `r610-cuda-tile.md`; the Supervisor's design-system and
product documents `DESIGN.md` and `PRODUCT.md`; and the demo design sidecars
`docs/decide-demo.DESIGN.md` and `docs/dual-purpose.DESIGN.md`.

The DFlash2 residue of `ddeeec19` needed nothing: v3 rejects an unsettled StateImage Fork at seal
revalidation instead of the fork's early admission gate, and the fork-only all-constrained
width-one DFlash path is not carried.

## Adapted from other forks

### cometkim/ninfer `nvfp4full` weight profile

Status: **integrated as a conversion recipe; not qualified for production**
(September 28, 2026).

| Item | Value |
|---|---|
| Source | [cometkim/ninfer](https://github.com/cometkim/ninfer) (Apache-2.0): branch `feat/qwen3.8-nvfp4full` at `ac8e0b75` (recipe, `nvfp4_maxabs` encoder and tests over upstream `1d8587bc`), and the v2 line at `55152a4f` (`calibrate_nvfp4full.py`, `verify_nvfp4full.py`, fork artifact document section 14) |
| Fork-line commits | `3db1caee` recipe, encoder, calibration and verification; `07085988` the BF16-reference distribution readout used to qualify it; `0f25f279` the Linux build fix it needed |
| Artifact | `qwen3_8_27b_nvfp4full_dflash2.ninfer` (local, `broad-v1` calibration) |

What changed in the adaptation:

- The recipe is an official recipe, `qwen3_8_27b_nvfp4full`, beside `qwen3_8_27b_nvfp4`; the fork's
  shared profile helper and explicit parent grouping are gone because `nvfp4_maxabs` is a built-in
  method that the default packing groups accept.
- The DFlash2 companion is Q8 (the upstream schema), not the fork's NVFP4-encoded module, so no
  fork runtime is needed. Text, MTP and Vision need no runtime change.
- Calibration loads the whole BF16 model instead of streaming layers. The fork's ten-document corpus
  is kept as `cometkim-v1`; the recipe's committed calibration is `broad-v1`, which adds pinned public
  text and rendered chats. With the same corpus, 64 of the 135 divisors are bit-identical to the
  published fork artifact's and the rest within -2% to +21% (different BF16 forward paths).
- Verification reads the v3 artifact and compares object payloads with the production artifact:
  the 112 imported MLP parents and their Use divisors are production's words.

Qualification (Colab G4 and the workstation, [measurements](../performance/rtx-pro-6000.md#qwen38-27b-nvfp4full-against-the-production-profile-2026-09-28)):
the profile is 2.95 GiB smaller, prefills 22-26% faster and decodes 9-13% faster, with equal MTP
acceptance, but it fails the quality gate set before the full scoring (mean KL to BF16 at most 1.25x
production's, top-1 agreement at most 1 point lower): KL 1.56x, top-1 2.8 points lower. GDN
projections cause most of the shift; weight-only NVFP4 (A16 activations) halves the KL increase but
still fails. Of the
narrower allocations tried, only NVFP4 on MLP 56-63 alone passes (KL 1.09x, top-1 0.81 points lower,
0.73 GiB smaller); NVFP4 on GDN layers 8-55 misses the top-1 bound by 0.07 points. Those variants
were produced with recipe overrides and are not committed recipes.

## Deliberate divergences from upstream

| Behaviour | Upstream | Fork | Reason and evidence |
|---|---|---|---|
| CUDA synchronization default (`300ddb9f`) | `spin` | `blocking` when `NINFER_CUDA_SYNC` is unset; the variable still selects `spin`, `yield` or `auto` | Shared desktop. Production-flag A/B on the RTX PRO 6000: 7,680-token prefill 681.0 vs 683.4 ms, decode 152.9 vs 148.4 tok/s (medians, within noise), CPU during decode 0.03 vs 0.96 core ([measurement](../performance/rtx-pro-6000.md#v3-port-against-the-previous-release-build-2026-09-28)). Igor, 2026-09-28 |
| Dense non-thinking presence penalty (`92ad4c47`) | `1.5` | `0`; MoE keeps `1.5` | Parity with the pre-v3 27B package, which used 0 |
| `preserve_thinking` server default (`3cb5e718`) | unset, so the Qwen3.8 template keeps returned closed-turn reasoning | `false` unless a request or `--preserve-thinking` asks | Parity with the pre-v3 fork (Igor, 2026-09-28) |
| Unsupported reasoning effort | passed to the chat template, whose `raise_exception` comes back as HTTP 400 `invalid_prompt` on `messages` with the interpreter's source trace | checked before rendering against the efforts observed from the loaded template at startup: HTTP 400 `reasoning_effort_not_supported` on the effort field, listing the supported efforts; template errors carry no trace | Restores the pre-v3 fork contract (`8264394c`) on Chat Completions, Responses and Anthropic Messages; the protocol vocabulary is unchanged |
| Continued final assistant turn (`737b570a`) | rendered without a think block | rendered behind the empty think block the generation prompt carries (thinking off) | `/v1/score` text form and assistant prefill are conditioned like a generated answer |
| Admission planned while an active lane's StateImage Fork is unsettled (both models) | at `d44ab584`: a pressure target is Feasible without the seal's Fork check, so a burst of admissions with prefix reuse ends in the fatal `selected pressure target could not be sealed` | the target is deferred, and EngineCore retries a deferred admission once after the next execution unit | Reproduced on the 27B at concurrency 4 (October 7, 2026, [evidence](../research/flash-next-v3-port-2026-09-29.md#fatal-engine-failures-found-2026-10-06-fixed-2026-10-07)). Upstream `b9114396` (not yet reviewed) replaces this planner; the next sync reconciles the fix with it |
| `POST /v1/rerank` | no rerank route | Jina-shaped rerank scored by an in-process System One Choice; `GET /v1/models` advertises `ninfer-choice-rerank-v1`; missing or wrong API key is HTTP 401 with the OpenAI error object | Added 2026-10-02. `return_documents` defaults to true. Choice labels stay off the public response |

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
| An external evaluation handoff plan | Completed 2026-09-07 handoff plan |
| `db1a3694` device-gated W4A4 raster | Not reproduced on v3 (above) |

`tools/freq_corpus/fixtures/ranking/*.i64` are upstream's own files (the default ranking of the
27B proposal head) and stay.

## Flash-Next v3

**Source integrated in the working tree, October 4, 2026; qualification below.**
The `workstation` working tree carries the text port from
`m3/4-text@577a1bcd` over its `6ad46d87` base, with native MTP and Vision added on the current
workstation Engine contract. The original architecture source remains the pre-v3
`research/qwen4-flash-next` line. No generic model base class or alternate inference route was
introduced.

The package now includes text prefill/decode, QSA/GDN/PLE continuation state, causal scoring and
hidden-row readout, CUDA Graphs, prefix reuse, constrained output, recursive MTP drafting with
accepted-prefix replay, and the shared Vision encoder through explicit Vision parameters.
MTP keeps its own QSA forming-block state and KV frontier; verification snapshots are disjoint
from resident checkpoints. Vision BF16 projection shapes include the 4304-wide partial-K tail.
A disjoint MoE workspace fixes operand overlap in the recovered text port. Keeping the wide
hyper-connection up projection in FP32 fixes a scoring/readout discrepancy. Batched GDN
projection/convolution retains FP32 and the single-row accumulation order to prevent an
away-from-tie trajectory divergence.

Qualification uses a Colab G4 RTX PRO 6000 Blackwell Server Edition, CUDA 13.3 and GCC 13.3.
Focused checks pass for text graph/eager parity, strict B=2/4/8 comparison (remaining differences
at exact ties), scoring/readout, MTP K=1/3/5 graph replay and eight heterogeneous requests,
ragged budgets, context limits, FP8 KV across QSA selection, seeded sampling with penalties,
constrained JSON, image reuse and ordered video frames. Host pressure demonstrably transfers
both recurrent state and KV, including MTP KV; resumed tokens exactly match device-only resume.
Changed math passes independent FP64/codec oracles at the affected production shapes.

End-to-end v2 parity is **not accepted**: the 27-request BF16 comparison has 8 identical
trajectories, a 9.625-nat first-divergence gap on one tool prompt, and mean/maximum absolute prompt
logprob differences of 0.948762/16.452988 nats. The shared-prefix top-20 proxy meets its mean/p99 bounds;
the greedy-gap and prompt-logprob criteria fail. Causal-prefix independence passes exactly.

Numerical acceptance moved to an independent transformers oracle. That gate fails one of five
metrics, mean chosen-token difference, entirely on one long document.

A pre-registered 27-document long-context follow-up finds no v3 regression against v2. The
maintainer accepted v3 numerically on October 6, 2026, with the gate's failure kept on record.

The same study found v3 was not prefill-chunk invariant, while v2 was. The cause was M-dependent
split-K in the GDN control projection and final-wave split-K in the FP8 projections.

Both are now removed. Prompt readouts and generated tokens are bit-identical across chunk sizes,
except inside a final chunk shorter than 256 tokens, as with v2. The real-artifact Engine case
`prefill chunk invariance` checks this.

The [active plan](../research/flash-next-v3-port-2026-09-29.md) records the criteria, results and
attribution.

The strengthened MTP target comparison initially failed beyond its 0.05-nat tie bound.
Corrections preserve ordinary decode projection arithmetic across speculative widths and fix
QSA's batch-dependent FP16 probability partitioning. The K=1/3/5 eight-prompt comparison now
matches all 96 generated tokens per request. All 16 focused Engine cases pass after correcting
case-sensitive image/video assertions; current ordinary/MTP timings include the correction costs.
With the chunk-invariance case added, all 17 pass on the G4 as of October 6.

MSVC 19.51 and CUDA 13.3 compile and link the native Windows Engine test, CLI, server and Supervisor. Build outputs
were redirected after the original build volume exhausted disk space; the link reports a CRT
library conflict warning.

Windows GPU runtime qualification (October 6-7, 2026) ran on an RTX PRO 6000 Blackwell
Workstation Edition beside the desktop: the 14 Flash-Next Op tests, the qwen4_exp loading test
and all 17 real-artifact Engine cases pass, and `ninfer_bench` matches the G4. The serving
configuration the G4 used (context 131072, KV capacity 262144, concurrency 8) does not fit beside
the desktop: startup correctly refuses a 9.8 GB runtime reservation with 7.5 GB available. The
Windows serving check with a smaller KV capacity remains open.

Three fatal engine failures found on October 6 are fixed
([causes and evidence](../research/flash-next-v3-port-2026-09-29.md#fatal-engine-failures-found-2026-10-06-fixed-2026-10-07)):
- `selected pressure target could not be sealed`: a burst of admissions planned while an earlier
  admission's StateImage Fork was unsettled. It also affects the 27B; see the divergence above;
- `candidate token ledger does not match prompt length`: the Flash-Next MTP round borrowed a staged
  materialization's prompt ledger as scratch;
- `KV committed frontier is invalid`: after a capture, a Flash-Next request with MTP re-mapped only
  its Text KV.

Current and earlier G4 performance/resource results and their costs are in
[performance](../performance.md#flash-next-v3-on-colab-g4-2026-10-04); the
[active plan](../research/flash-next-v3-port-2026-09-29.md) distinguishes this implementation
from the wider historical M3 qualification criteria. The Supervisor accepts K=1..5 and rejects an optimized draft head for Flash-Next; its native
CPU test passes. This source integration is on `workstation`.

DFlash/DFlash2, v2-only draft-head shortlists and alternate GDN-state/attention modes are not
part of this architecture's v3 contract. MTP uses the full stored output head; BF16 and FP8
row-256 KV are the supported cache representations.

## Deferred

- **Matched Flash-Next versus Qwen3.8-27B prefill study.** Resume with M3: same workstation,
  matched application texts and supported contexts, verified cold and 90%-cached ratios, TTFT,
  uncached prefill and complete request latency reported separately.

## Open qualification questions

- `prefix_real`'s Host-restore check fails on the Qwen3.8-27B NVFP4 artifact identically on the
  pristine base and the fork line (the demoted turn closure is not the selected source): upstream
  behaviour on this artifact, not investigated further.
- A json_schema request whose prompt asks for prose produces whitespace until its output limit on
  both the pre-v3 build and the fork line: the grammar admits unbounded leading
  whitespace. A bound on outer whitespace would fix it.
- On the pre-v3 line, exploratory T=1500 GDN input and LinearAdd 5120x6144 NVFP4 fixtures disagreed
  with their oracles on both baseline and candidate; not re-examined on v3.
- `dflash2_real`'s default probe (K=15, one 24-token generation) accepted 21 of 23 drafted tokens
  before the upstream `e31bc99b` merge and 20 of 31 after it; the test passes both times. The merge
  changed the Q8, BF16 and GDN kernels the drafter and target run. DFlash2 acceptance on real
  requests was not re-measured; the production configuration uses MTP, whose acceptance is unchanged. After the
  `d44ab584` merge the same probe accepted 21 of 23 on a Colab G4 (Linux); the Windows figures
  are from the workstation, so the two are not a like-for-like pair.

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
- Serving smoke (production flags, MTP and DFlash2): chat with and without thinking,
  image input, reuse across turns, logprobs, `/v1/score`, `/v1/systemone`, required tool calls,
  structured output, Anthropic Messages.
- Production-flag A/B against the pre-v3 build: the fork line matches it within the
  round-to-round drift on a 7,680-token prefill and a 256-token MTP5 decode
  ([table](../performance/rtx-pro-6000.md#v3-port-against-the-previous-release-build-2026-09-28)).
- A 48-entry private catalog is accepted and reported; retention follows the state-image backing
  (`--device-state-slots` and `--host-state-slots`), and evictions are counted in `/admin/stats`.
- Release candidate `2026.09.28-v3port.1` (source `e92c2078`, LGPL FFmpeg; its runtime DLLs are
  byte-identical to those of `2026.09.24-alpha.1`, so only the four executables change).
  - Staged smoke: the release payload ran with nothing but its own DLLs on `PATH`, with
    production flags. It passed 11 of 11 checks: models list, chat with and without thinking, image,
    tool call, `json_object`, `json_schema`, `/v1/systemone`, `/admin/vram`, `/admin/stats` and
    `/health`.
  - Supervisor: on a v3 configuration, the new supervisor lists the four
    27B entries as available and Flash-Next as unavailable (it is still v2).
  - Rendered prompts are byte-identical to the pre-v3 build for 21 request shapes, read back through
    prompt-position logprobs. The shapes cover the default, every effort, thinking on and off, a
    system message, tools, a tool loop, multi-turn with reasoning, preserve, image and `json_object`.
  - Unsupported efforts (`minimal`, `high`, `max`) get HTTP 400 from both builds, but the error
    body differs. The fork line returns `invalid_prompt` on `messages`, with the template's message
    and a Jinja trace. The pre-v3 build returns `reasoning_effort_not_supported` on `reasoning_effort`.
    The fork line restored the pre-v3 contract afterwards (divergence table above).
  - A 24-question `/v1/systemone` replay (`noul`, `choice` and `score` on six states) is
    bit-identical to the pre-v3 build.
  - The upgrade tool, run from a Windows checkout (`core.autocrlf=true`), embeds the chat template
    with CRLF line endings. The vendored Jinja lexer normalizes line endings, so the rendered
    prompts are unchanged; only the embedded bytes differ from a Linux run.
