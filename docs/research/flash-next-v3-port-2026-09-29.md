# Flash-Next v3 port (M3): phase 0 feasibility, design and plan

Date: 2026-09-29. Base: `workstation` at `28c40898` (v3). Source of the port: `research/qwen4-flash-next`
at `87812bc8` (v2). This is a temporary plan. Remove it when M3 closes; stable content moves to the
maintainer references it names.

Scope of M3: Qwen3.8-Flash-Next (transformers `Qwen4ExpForConditionalGeneration`,
text `model_type = qwen4_exp_text`) becomes a second v3 architecture package beside
`src/models/qwen3_5`, with a v3 artifact. All development and qualification runs on Colab G4. The
on-site RTX PRO 6000 Workstation Edition only receives the finished build and artifact.

## Current workstation implementation (2026-10-04)

The current user deliverable is the native v3 Flash-Next package on `workstation`, including
**text, MTP and Vision in this pass**, with GPU validation only on Colab G4. No local GPU use,
commit, release or deployment is part of this pass. The current base is `2969beee`; text was
recovered from `m3/4-text@577a1bcd` relative to `6ad46d87`, then adapted to the current Engine.
MTP and Vision were implemented here. Earlier measurements and criteria below remain historical
evidence; they are not silently replaced by the current checks.

The selected implementation has package-owned Parameters, Program, StateImage, planning and
transactions, a closed Engine variant, and the shared frontend/Vision encoder through explicit
Vision parameters. Model code composes Ops and owns continuation semantics. Ops retain math;
there is no alternate inference route, runtime repacking, family base class or DFlash placeholder.
The package carries its explicit scheduling code; keeping it aligned with future Engine contract
changes is a maintenance cost. Shared-source changes include the Engine dispatch, Vision
constructor seam, registered Op geometries and finite kernel routes.

| Dimension | Current evidence / status |
|---|---|
| Text behavior | Native prefill/decode, CUDA Graphs, prefix reuse/catalog turnover, logprobs, structured output, causal scoring and final 2560-wide hidden export implemented. Focused real-artifact Engine checks pass. |
| Numerical semantics | Changed Ops use independent FP64/codec oracles and real shapes. Scoring versus prompt readout max absolute difference 9.53674e-07 nats, below unchanged 0.05. Text B=2/4/8 passes unchanged 0.05-nat near-tie criterion. Changing future tokens at fixed extents 64/1024/2200 leaves predictor prefixes exactly equal, including beyond QSA's sparse boundary. The saved-v2 comparison below fails the greedy-gap and prompt-logprob criteria. The independent-oracle gate that replaced it passes four of five metrics. It fails mean chosen-token difference (0.270212 v3 against an allowed 0.261764), entirely on `book-18k`'s long-prompt readouts. A pre-registered 27-document long-context follow-up finds no v3 regression at either chunk size. `book-18k`'s offset follows v3's chunk size, and unlike v2, v3 is not chunk-invariant. Numerical acceptance remains open. |
| MTP | Full-head K=1..5, target verification width K+1, recursive drafts, teacher extension, GDN accepted-prefix fold and private QSA/PLE snapshots implemented. K=1/3/5 graph/eager equality, ragged budgets, eight distinct active requests, context-tail limits, FP8 KV at the 2048-token QSA selection boundary, seeded stochastic penalties and constrained output pass. |
| Continuation state | MTP owns a separate indexer and KV frontier E-1 at target frontier E. Host pressure transfers state and both KV families; plain and MTP resumes match device-only tokens exactly. State-image clone/reset/isolation and incompatible host geometry tests pass. |
| Vision | Shared 27-layer encoder, 2560 merger, bounded startup workspace/handoff, chunked media scatter and three-axis continuation positions implemented. Red/blue image discrimination, prefix reuse and ordered red-then-blue video pass with ordinary and MTP generation. BF16 4304-wide partial-K projection tails pass the independent oracle. |
| Interfaces | Public Engine, CLI/serve and CausalScoring dispatch to the package. BF16/FP8 KV and full MTP head are supported; DFlash, DFlash2 and optimized draft shortlists are rejected before loading. Python conversion checks: 3 passed on Python 3.11.17. |
| Performance/resources | Current same-allocation G4 MTP0/3/5 corpus measurements, concurrency 2/4/8 ordinary serving and C=8 MTP3/5, numerical-fix costs and resource accounting are recorded in [performance](../performance.md#flash-next-v3-on-colab-g4-2026-10-04). MTP5 loses 23.7% decode throughput on the short B=1 input; at C=8 MTP3/5 lose 8.5%/25.0% versus ordinary decode. Longer B=1 continuations improve. These are workload-specific results. |
| Platform | Linux Release CUDA 13.3/GCC 13.3 built and ran on G4. Native MSVC 19.51/CUDA 13.3 builds compile and link the Engine test, CLI, server and Supervisor. Outputs were redirected after the original build volume exhausted space; the linker reports LNK4098 (CRT-library conflict). Windows GPU execution and release packaging remain unverified. No local GPU was used. |

Failures found and disposition:

- The recovered MoE route could overlap operands with the root scratch arena. A disjoint
  subarena fixes ownership; the original oracle tolerances were retained.
- The wide hyper up projection rounded before sigmoid and produced a 0.109504-nat scoring
  discrepancy. Keeping that private result in FP32 fixes it without relaxing the 0.05 criterion.
- Batched GDN rounded before convolution and diverged at a 0.125-nat reference gap. Retaining
  FP32 with the B=1 reduction order fixes it. A Tensor Core alternative passed its local FP64
  oracle but still failed the Engine criterion and was rejected. Performance costs of the
  initial per-row route and the selected shared-weight route remain reported.
- Initial host pressure fixtures selected device state, so they did not demonstrate host
  restore. The corrected fixture forces state and KV transfers and compares exact output.
  An older catalog test forced post-EOS output that could not be re-rendered; turnover now uses
  normal EOS behavior while the separate raw-token identity test keeps forced output.
- A stronger eight-request MTP target comparison failed on four rows, with a worst reference
  gap of 2.75 nats against the unchanged 0.05 bound. Verification's aggregate width had
  accidentally enabled prefill A8 projection. Enforcing decode A16 removes three failures;
  one 0.125-nat divergence remained. Fused recurrent projection windows pass the original
  comparison against the target-only MTP fallback. The strengthened comparison against
  ordinary decoding still fails at K=1/3/5 (0.125-nat gaps). Rounding intra-window history
  to its persistent BF16 form did not resolve it and exposed a 1.75-nat worst gap. Earlier
  graph equality, budget and first-token checks did not test this property. Subsequent
  short-width FP8, hyper and MoE route harmonization passed their independent Op oracles but
  did not close the Engine gap (worst intermediate gap 3.375 nats). The shared MoE projection
  now keeps private gate/up accumulators in FP32 before SwiGLU, increasing workspace by
  2,560 bytes per token for A16 and 3,840 for A4. Aggregate correction costs are measured below;
  the contribution of this individual change is not isolated.
- A stage trace found QSA changed softmax partitions from 32 to 16 at batch width four.
  Because probabilities are stored in FP16 relative to each partition maximum, that changed
  represented values even with identical Q/K/V and block selections. Fixed partitioning passes
  the FP64 oracle and makes the first verified token identical through all 48 model layers.
  That candidate still failed the full K=1/3/5 comparison (worst 1.0-nat gap).
  The next trace isolated a further batch-dependent difference in the layer-1 PLE injection:
  BF16 key/value projections switched reduction algorithms with column count. Aligning those
  short projections and the vocabulary readout with the ordinary reduction closes the full
  K=1/3/5, eight-prompt, 96-token comparison with no token divergences. BF16 projection FP64
  tests and QSA independent-query batch invariance both pass. All 16 real-artifact Engine cases
  pass after the final fixture correction below. Restoring the faster grouped MoE route above
  eight columns reintroduces 0.125-nat failures at all three draft lengths, so it was rejected.
- Two Vision assertions initially failed because the valid video response capitalized "Blue".
  Color checks now normalize case, also require red/blue discrimination for cold and reused
  images, and pass for ordinary and MTP image/video execution. The serving benchmark initially
  selected an unsupported optimized proposal head; an explicit full/optimized option now
  reaches both runners, is verified against server telemetry, and is checked during resume.
  Sixteen focused Python 3.11 harness tests pass; both concurrent MTP runs then complete.
- Missing replay geometry and unimplemented Vision/backend paths in intermediate candidates
  failed and were completed. A remote rebuild briefly retained stale instrumented objects due
  to tar timestamps; those Engine results were invalidated, sources touched and checks rerun.

This is source integration with focused G4 qualification, not closure of every historical M3
acceptance item. The original v2 greedy/logprob/MTP/Vision comparison criteria in §5 are preserved;
current results must be reported separately from those older baselines. Full v2/v3 same-session
performance, the full saturated-pool scenario campaign, large-media performance and Windows GPU
qualification are not established by these tests. Stable implementation contracts now live in
[the artifact/execution guide](../maintainer/qwen3.8-flash-next-artifact.md#7-execution) and
[upstream port status](../maintainer/upstream-ports.md#flash-next-v3).

### Saved-v2 comparison: acceptance gap

The unchanged September 30 M3.4 comparison uses v2 `87812bc8`, BF16 KV, context 131072,
KV capacity 262144, concurrency 8 and prefill chunk 8192. Its 27 greedy requests cover 16 short,
four thinking, two tool, two schema and three long-document prompts. Prompt readouts sample
1,433 aligned positions from the 23 requests without tools/schema. The reference's second run
uses chunk 2048 and collection concurrency 2 to measure its route variation. The current G4
candidate uses the original reference settings. No comparison tolerance was relaxed.

| Criterion | Reference route variation | Current candidate versus reference | Status |
|---|---:|---:|---|
| Maximum reference gap at first greedy divergence | 0.25 nats | 9.625 nats (`tool-1`, position 0) | Fails 0.50 bound |
| Shared-prefix top-20 KL proxy, mean / p99 | 0.000774 / 0.009950 | 0.002521 / 0.021637 | Within 0.01 / 0.10 bounds |
| Prompt absolute logprob difference, mean / maximum | 0.000567 / 0.374786 | 0.948762 / 16.452988 nats | Fails 0.03 / 1.124359 bounds |

Only 8/27 complete generated token sequences are identical (reference variation: 11/27).
The candidate's 98.8869% agreement is restricted to shared prefixes plus the first divergence,
with 1,688 shared positions; it is **not** teacher-forced agreement over every generated token.
The clipped top-20 statistic is a truncated proxy, not full-distribution KL; its maximum is
1.599525. There are no prefix-only length mismatches in the final comparison. Current schema
behavior has separate passing checks; that does not make its reference trajectory identical.

Before the final MTP arithmetic corrections, the same comparison had a 6.0-nat worst greedy
gap and 0.943660/18.951420 mean/maximum prompt difference, with proxy mean/p99/max
0.001742/0.027043/0.183858. Thus the final candidate improves some maxima but worsens the
greedy gap, mean prompt difference and maximum proxy; it does not close this gate.

That earlier candidate's 390 short, 275 thinking and 768 long sampled prompt positions had mean NLL changes of
+0.028182, -0.077530 and -0.631041 nats, respectively. These mixed outcomes do not replace the
failed absolute-difference criteria and are not a full perplexity or model-quality evaluation.

Diagnosis has ruled out the new FP32 hyper up projection as the sole cause: a temporary ablation
still gives 0.925685 mean and 20.28947 maximum prompt difference. The saved v2 MMA/non-MMA QSA
routes themselves differ by 0.797127 mean and 15.005175 maximum prompt nats, with a 9.625-nat
greedy divergence. v2 stages probabilities in BF16; the v3 attention Op stages them in FP16.
This is a material arithmetic difference, not proof that it accounts for the entire mismatch.
A live v2 rebuild on the current G4 reproduces the short-prompt and tool-prompt baseline tokens
and raw logprobs exactly. A 23-token stage trace starts with exact hyper initialization, then
relative L2 differences of 0.000731 at the first attention input and 0.015470 at its output.
Differences accumulate to 0.104884 after the final layer; no single catastrophic transition
has been established. An additional FP64 causal-attention calculation over the saved v2
53-token QSA operands gives relative L2 errors of approximately 0.0016–0.0018 across its 12
attention layers, consistent with BF16 output rounding; it does not establish a gross legacy
attention defect or explain the full-model discrepancy. The legacy and current MoE both switch
to A4 at 256 prefill tokens and quantize input and routed intermediate activations, so a newly
enabled A4 policy does not explain the long-prompt concentration. The comparison remains open.

### Acceptance by independent oracle (decided 2026-10-05, before any oracle result)

The saved-v2 comparison above stays on record unchanged, but it is no longer the acceptance gate:
v2 is not a stable yardstick, because its own MMA and non-MMA QSA routes differ by 0.797127 mean
and 15.005175 maximum prompt nats. Following the qualification contract (production routes are
judged against an independent oracle, not another kernel), v3 is accepted against the
transformers `qwen4_exp` reference, with v2 measured the same way as the comparison point.

- **Oracle.** transformers' `Qwen4ExpText*` modules with the pinned checkpoint's weights decoded
  independently (NVFP4 experts through the FP4 table and their stored scales, the u4 PLE table
  with its scales), as in the research line's `tools/reference/qwen3_8_flash_next/oracle`, in
  FP32. Its transformers version is recorded with the results.
- **Inputs.** The same 50 records as the saved comparison (27 greedy trajectories, 23 prompt
  readouts). Each engine's own tokens are teacher-forced through the oracle, so every engine
  distribution is compared with the oracle's at an identical context. Tokenization must
  reproduce each record's `prompt_tokens`.
- **Metrics, per engine against the oracle.** Mean and maximum absolute difference of the
  chosen token's logprob; top-1 agreement; KL(oracle || engine) over the union of both top-20
  lists, renormalized, mean and p99. Positions are pooled over all records.
- **Criteria.** v3 passes if, on every metric, it is no worse than v2 by more than the larger of
  10% of v2's value or the v2 reference route variation already measured (0.000567 mean and
  0.374786 maximum chosen-token difference). If v3 is worse on any metric, the gate fails,
  and the result is reported with the positions responsible.
- **Not covered.** This gate replaces only the full-model numerical comparison. The Op oracles,
  Engine cases, MTP equality and Windows GPU qualification are separate and still required.

### Oracle gate result (2026-10-05): fails on mean chosen-token difference

`tools/bench/flash_next/oracle.py` ran on a Colab G4 (RTX PRO 6000 Blackwell Server) with
transformers 5.18.0 and PyTorch 2.11.0+cu130, FP32 with TF32 off and SDPA attention, over the
saved v2 records and the current candidate's records (50 each). Sources:
`primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8@a4e813ed` without `ple-bf16-*` and
`primitive-ai/Qwen3.8-Flash-Next-PLE-quant@da8b3958` `ples_int4`. The 100 teacher-forced
forwards took about 15 minutes; the longest (21,141 tokens) took 77-91 s.

All 27 prompts reproduce `prompt_tokens` once tools are rendered the way the OpenAI route serves
them (`render_tool_definition`: name, parameters, `strict: false`, then description). Rendering
the client's tool object instead is 6 tokens shorter (296/297 versus 302/303). The first pass
used that rendering and excluded both tool records; it gave the same verdict (mean 0.243654 v2,
0.276054 v3, allowed 0.268019).

| Metric | v2 | v3 | Allowed for v3 | Status |
|---|---:|---:|---:|---|
| Chosen-token difference, mean | 0.237967 | 0.270212 | 0.261764 | **Fails** |
| Chosen-token difference, maximum | 16.913912 | 16.433323 | 18.605303 | Passes |
| Top-1 agreement (generated positions) | 0.976981 | 0.976558 | 0.879283 | Passes |
| KL mean (generated positions) | 0.089672 | 0.089330 | 0.098639 | Passes |
| KL p99 (generated positions) | 2.581029 | 2.594268 | 2.839132 | Passes |

Pooled positions: 4,865 for v2 and 4,803 for v3, none excluded.

**Positions responsible.** The `book-18k` prompt readouts (256 positions sampled across its 21,013
prompt tokens) account for all of the excess. v3's mean difference there is 1.8702 nats against
v2's 1.2398, which is +161.4 nats summed. Over every other record v3 sums 21 nats *less* than v2.
Without these 256 positions the means would be 0.1801 for v3 and 0.1823 for v2. That figure only
attributes the failure; it is not a revised gate.

| Records | Kind | v2 mean / max | v3 mean / max | Oracle draw-to-draw mean / max |
|---|---|---:|---:|---:|
| short (16) | prompt | 0.139 / 1.63 | 0.121 / 0.99 | 0.000 / 0.00 |
| thinking (4) | prompt | 0.183 / 2.01 | 0.182 / 1.61 | 0.000 / 0.00 |
| code-4k | prompt | 1.523 / 16.91 | 1.508 / 16.43 | 0.013 / 0.26 |
| wiki-18k | prompt | 1.159 / 10.29 | 1.109 / 8.74 | 0.216 / 7.45 |
| book-18k | prompt | 1.240 / 14.71 | 1.870 / 13.24 | 0.324 / 11.13 |
| book-18k | generated | 0.038 / 0.36, KL 0.0080 | 0.041 / 0.41, KL 0.0081 | — |

**What the diagnosis established.**

- The oracle is exact below the QSA budget. Prompts shorter than 2,048 tokens give bit-identical
  oracle readouts in the v2 and v3 forwards.
- Past the budget, the reference is itself route-chaotic. Its two forwards over the identical
  `book-18k` prompt differ only by the generated tokens appended after the readouts, which
  changes the reduction shapes. They still disagree by up to 11.13 nats at one position.
- Oracle-draw noise does not explain v3's `book-18k` offset. Scored against the v2 forward's
  oracle readouts, v3 is still at 1.950 mean.
- The offset is a whole-document bias, not a causality leak. v3 is more confident than the
  oracle in the true next token by 1.54 nats on average: 98 positions are more than 2 nats above
  it and 6 more than 2 nats below. The bias is flat across QSA block offsets: +1.41, +1.92,
  +1.56 and +1.67 nats for `p mod 4` = 0 to 3, past position 2,048.
- A block-granular leak of future tokens would be absent at offset 3, where the next token sits
  in the following block.
- The sign is not consistent across documents. Past position 2,048 v3 is −0.36 on `wiki-18k` and
  −0.54 on `code-4k`. v2 is +0.47 and +0.63 on wiki and book, and −0.41 on code.
- Generated positions decoded after the same 21K prefill agree with the oracle as well for v3 as
  for v2.

**Open.** These records cannot tell whether v3's per-document long-context offset is a v3 defect
or the long-context route sensitivity both engines show. The evidence for route sensitivity is
the same in each case:

- v2's MMA and non-MMA routes differ by 0.797127 mean prompt nats;
- both engines deviate from the oracle by 1.1-1.9 nats per long-document prompt position;
- v2 and v3 differ from each other there by 1.4-1.8 nats.

Separating the two needs more long documents per engine. The pre-registered gate itself stays
failed.

### Long-context follow-up (decided 2026-10-05, before any result)

This follow-up asks one question: is v3's `book-18k` offset a v3 long-context defect, or the
route sensitivity both engines show? It adds evidence and changes nothing about the gate above,
which stays failed. Accepting v3 remains the maintainer's decision.

- **Documents.** `parity.py --set long`, 27 documents:
  - 24 new documents from corpus files the gate set does not use. Each of pg19, wikitext,
    ninfer and zhwiki contributes files 01-03, in two consecutive windows of 80,000, 80,000,
    64,000 and 28,000 characters respectively.
  - The gate set's `code-4k`, `wiki-18k` and `book-18k`, as run-to-run controls.
- **Engines.** v2 `87812bc8` on its published artifact and the v3 candidate on the artifact
  derived from it. Server settings are the gate's: BF16 KV, context 131072, KV capacity 262144,
  concurrency 8. Each engine runs twice, at prefill chunk 8192 (the primary route) and 2048 (a
  second route). Prompt readouts use the gate's 256 evenly spaced positions.
- **Oracle.** `oracle.py prompts`, two draws per document. Draw 1 appends a fixed suffix after
  the prompt, which leaves every readout's context unchanged.
- **Statistic.** Per document, the mean |engine − oracle draw 0| over readouts at positions
  ≥ 2,048 (the QSA budget). D = v3 − v2 on the primary route.
- **Reading.** The margin is 10% of v2's mean over documents.
  - *Regression* if the 95% document-bootstrap interval of mean D lies above 0 and mean D
    exceeds the margin.
  - *No regression* if the interval's upper end is at or below the margin.
  - *Inconclusive* otherwise.
- **Reported with the reading:**
  - the per-document table and the number of documents where v3 is worse;
  - signed differences;
  - each engine's chunk-2048 route variation;
  - the oracle's draw-to-draw variation;
  - the controls against the gate records.
- **What follows.** A regression leads to a layer-by-layer trace of v3 against the oracle on the
  worst document. Either other reading is reported as it stands. No gate criterion or tolerance
  changes in any case.

### Long-context follow-up result (2026-10-06): no v3 regression; `book-18k` was v3's chunk route

**Setup.**
- Engines ran on one Colab G4: v2 `87812bc8` on the published artifact, then the v3 candidate.
  Its engine source is unchanged from the gate's; only tooling and docs differ. It ran on the
  derived artifact, which has the same artifact id as the gate's (`1e7e026e…`).
- The oracle ran on two G4 sessions with two shards each. A first oracle session was lost after
  about 46 of its 54 draws, before any download, and was rerun from the start. Nothing from the
  lost run is used.
- Every engine record aligned with the oracle's tokenization; no record was excluded.

**Controls.** Against the gate's records, v3 reproduces all 768 control readouts bit-identically.
v2 reproduces 764. The other 4 are on `wiki-18k`, with a maximum of 0.374786 nats, the same value
as v2's recorded route variation.

**Pre-registered reading.** Mean |engine − oracle| over readouts at positions ≥ 2,048, 27
documents:

| Route | v2 mean | v3 mean | Mean D (v3 − v2) | 95% document bootstrap | Margin | v3 worse on | Reading |
|---|---:|---:|---:|---:|---:|---:|---|
| Chunk 8192 (primary) | 1.4748 | 1.3974 | −0.0774 | [−0.2332, +0.0636] | 0.1475 | 11 / 27 | No regression |
| Chunk 2048 | 1.4753 | 1.3864 | −0.0889 | [−0.2337, +0.0497] | 0.1475 | 14 / 27 | No regression |

**Spread of D.** D varies by document with a standard deviation of 0.403. The oracle's own
draw-to-draw variation averages 0.185.

| Extremes of D at chunk 8192 | Document | D |
|---|---|---:|
| Worst for v3 | `book-18k` | +0.831 |
| | `pg19-02-1` | +0.588 |
| Worst for v2 | `pg19-01-1` | −1.368 |
| | `wikitext-02-0` | −0.914 |

**`book-18k` moves with v3's chunk size.** v3 scores 2.038 at chunk 8192 and 1.237 at chunk 2048.
v2 scores 1.207 at both. The positions that failed the gate are therefore one draw of v3's
chunk-dependent route; at the other chunk size v3 matches v2 on that document.

**v3 is not chunk-invariant; v2 is.** The two chunk sizes change readouts by different amounts:

| Engine | Change between chunk 8192 and 2048 |
|---|---|
| v2 | Bit-identical on 24 of 27 documents; 0.0024 mean |
| v3 | 1.56 nats per readout past position 2,048; per-document scores change by 0.28 mean, 0.80 max |

The v3 dependence begins inside the first chunk. Readouts below position 1,024 differ by 1.00 on
average, and only position 0 matches. Those tokens and their position in the chunk are identical
in both runs; only the chunk's token count differs.

The earlier fixed-extent test shows that the *contents* of later tokens do not affect earlier
positions. So the dependence is on the launch's token count, not a leak of future tokens.

A code reading found every activation quantizer to be per-row. It also found several Ops whose
reduction order or tile shape depends on the token count. The first candidate is the GDN gating
projection route table, which chooses split-K by column count and touches every row of every GDN
layer. NVFP4 activation quantization above 256 prefill tokens can amplify such rounding-level
differences. This cause is not yet verified by an experiment.

**Disposition.** The oracle gate stays failed as recorded. The follow-up finds no long-context
regression on either route. The gate's failure is attributable to v3's chunk-route variation
landing unfavorably on one document. Accepting v3 is the maintainer's decision. Still open: v3
has lost v2's chunk-size invariance.

**Decision (maintainer, 2026-10-06).** v3 is accepted numerically on this evidence, with the
oracle gate's failure kept on record. Chunk-size invariance is to be restored before landing.

### Prefill chunk invariance restored (2026-10-06)

**Cause.** Two Flash-Next routes changed a token's FP32 reduction order with the launch's token
count:
- The GDN control projection chose split 8/4/2/1 by column count.
- The FP8 projections (`[13312,2560]`, `[16384,2560]` including the fused GDN input,
  `[2560,6144]`) split the K loop of an underfilled final wave. Which tiles form that wave
  depends on the token count.

NVFP4 activation quantization above 256 prefill tokens amplifies those rounding-level
differences. Every activation quantizer is per-row, and no other prefill route on the large-chunk
path changes per-column order.

**Change.**
- The control projection uses split 8 for every column count. That is its existing decode route,
  and it already slices large launches to cooperative residency.
- The three FP8 shapes use the same tiles without the final-wave split.
- `[2560,6144]` takes 64x128 tiles through T=640 and 128x128 through T=1536. Tile shape does not
  change a column's K order.

**Verification** (Colab G4, base `9b662909` against the patched tree):

| Check | Base kernels | Patched |
|---|---|---|
| Op test, control projection: leading columns at T=1/9/1025/2049/4097/8192 vs T=64 | Changes from T=1025 | Bit-identical |
| Op test, FP8 A8 (all three shapes): shared columns of consecutive T=289...8192, uniform inputs | Changes at 4-7 of the T values per shape | Bit-identical |
| New Engine case `prefill chunk invariance` (3,599 scores, chunk 1024 and 1536 vs 4096) | 3,596 and 3,595 differ, max 4.59 / 7.68 nats | 0 differ |
| Long set (27 documents), chunk 8192 vs 2048: prompt readouts and generated tokens | Not invariant (see above) | 27/27 bit-identical |
| Retuned table at chunk 1024 vs the fix build at chunk 8192 | — | 23/27 bit-identical |

The four documents that differ in the last row each end in a final 1024-chunk shorter than 256
tokens (19, 166, 233, 236 tokens), which takes the A16 MoE route. Each first difference lies
inside that chunk. Final chunks of 274 and 283 tokens stayed identical.

The full real-artifact Engine suite passes on the patched tree: 17 of 17 cases, including the new
one. The touched targets compile and link with MSVC 19.51 and CUDA 13.3 on Windows.

**Numerics after the change.** The pre-registered long-context reading against the same oracle
draws stays "no regression":

| Chunk | v3 mean | v2 mean | Mean D | 95% interval | v3 worse on |
|---:|---:|---:|---:|---:|---:|
| 8192 | 1.2869 | 1.4748 | −0.1879 | [−0.3491, −0.0307] | 10 / 27 |
| 2048 | 1.2869 | 1.4753 | −0.1884 | [−0.3494, −0.0314] | 10 / 27 |

Both chunk sizes give the same v3 mean because the result is now chunk-invariant. This is a
different draw of the long-context route, so it does not show that the change improved accuracy.

**Performance** is recorded in [performance](../performance.md#prefill-chunk-invariance-2026-10-06).
The invariance fix alone cost up to 1.2% prefill. The selected table recovers it, gains 0.4-0.8%
at 512-1,024-token prompts and stays within ±0.13% at longer ones. Decode is unchanged, and the
chunk-1024 workspace drops by 12.5 MiB.

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

### Open: fatal engine failures (2026-10-06)

A Colab G4 evaluation of MTP draft policies exercised workloads the Engine cases do not cover:
long prompts at concurrency, and prefix reuse after a long request. It found three fatal engine
failures, all reproduced on `7053c7af` (this line plus NVTX ranges only).

| Workload | Fatal engine failure |
|---|---|
| concurrency 4: two 20K-token summaries and two code requests submitted together, no MTP | `selected pressure target could not be sealed` |
| the same workload with MTP K=2, 3 or 5 | `candidate token ledger does not match prompt length` |
| concurrency 1, MTP K=5: a 128K-token request, then a request reusing an earlier conversation's prefix | `KV committed frontier is invalid` |

The same reuse sequence without MTP completes. The Engine cases' concurrency coverage uses prompts
under 512 tokens, which is why they pass. These failures block landing on `workstation`. Each fix
should add an Engine case with long prompts at concurrency.
