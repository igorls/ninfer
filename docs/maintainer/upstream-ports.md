# Upstream ports

Updated: October 28, 2026.

This is the living record of how this workstation fork ([igorls/ninfer](https://github.com/igorls/ninfer))
tracks [Neroued/ninfer](https://github.com/Neroued/ninfer): the reviewed upstream point, the fork
features carried onto upstream v3, deliberate divergences, dropped and deferred work, qualification
evidence. **RTX 5090 and RTX PRO 6000 Blackwell are both first-class fork targets;
native Windows remains supported beside Linux.** Preserve the resident-GPU path
when the selected model fits, and assess bounded offload for larger working sets.
Qualification and performance evidence remain specific to each device and route.

## Sync state and method

| Item | Value |
|---|---|
| Fork line | `workstation` (named `port/upstream-v3` until September 28, 2026), based on upstream [`bace20dc`](https://github.com/Neroued/ninfer/commit/bace20dc70249eed6402b66d4852c6c3f9612905) (September 24, 2026) and merged with upstream `e31bc99b` in `e2a149f1` and `d44ab584` in `d29866c9` |
| Last merged upstream commit | [`64d10fbf`](https://github.com/Neroued/ninfer/commit/64d10fbf9f8e6bfc79ab0570b94674c5b9997929) on the local `sync/upstream-64d10fbf` branch; physical RTX 5090/native Windows GPU qualification remains outstanding |
| Pre-v3 fork line | `research/qwen4-flash-next` at `87812bc8` (September 26); the source for Flash-Next |
| Current sync | All reviewed incoming work through `64d10fbf` is recorded in the local sync merge with the fork adaptations and regression repairs. Linux G4 resident/bounded Engine qualification and Windows build/CPU checks pass |
| Current target | `64d10fbf` (October 16, 2026); source adaptation and G4 regression repairs are qualified, with retained measurement/resource costs and outstanding physical RTX 5090/native Windows GPU qualification |

### Final-tip adaptation (October 23–28)

The October 28 repair addresses dense MTP and ordinary Flash decode before the
merge commit. The original baseline and workload remain the comparison
authorities. Dense target verification accidentally repeated its final
normalization, vocabulary projection and argmax; the duplicate is removed.
The dense acceptance divergence begins at one BF16 SiLU-gated D128 norm output
in prefill. The specialization now uses the generic row's BF16x2 pair grouping,
restoring baseline tokens and acceptance in the focused probes. Plain and
Sigmoid-gated profiles retain their upstream grouping. Both SiLU profiles satisfy
the existing FP64 accuracy budget; this is a consistency/performance choice,
with no claim of improved general accuracy.

The Flash candidate uses a fixed local sorting network for the 512-expert,
top-10 router and a vectorized single-token NVFP4 Down projection that includes
the prepared shared term. Other router shapes retain their original sorting.
All NVFP4 vector widths use the same two FP32 accumulation chains, and separate
and fused shared Down use the same dot reduction. There is no new activation
cast, workspace allocation or runtime weight packing. The first fast fusion
passed its independent Op oracle but failed two of 26 real Engine cases at a
0.125-nat reference gap, beyond the unchanged 0.05 criterion. The split control
passed both. Aligning the vector reductions fixes those two cases, and all 26
resident Engine cases pass on the corrected source. The expanded independent
MoE residual-output oracle, RMSNorm/gated-norm suites and both dense MTP
integration checks pass. Windows CLI/server/Supervisor and affected test targets
build; an earlier stale-object link failure is resolved by a coherent rebuild,
with the existing LIBCMT warning retained. Ordinary/MTP K=3 Host execution passes
in eager and graph modes at both 32 and 28 GiB, registering 67,948,068,864 Host
expert bytes with 16 workers. The original paired method and baseline are retained.

The first complete repair comparison validates 48 process records and improves
15 of 16 wave medians. The remaining C1/P512 Flash ordinary median costs 0.453%
(-0.316 to +1.143%). A paired profile shows more GPU work despite less GPU idle
time. The follow-up splits single-query BF16 sparse-GQA value columns across two
CTAs, preserving the key partitions, probability scaling and warp reductions.
It duplicates key/cache tile work, with no new workspace. The independent BF16
oracle and cross-width invariance checks pass, as do all 26 resident Engine
cases on this selected source. Bounded Host checks also pass at both budgets.
Other query counts and quantized-cache schedules retain their existing implementation.
The final comparison validates 24 new Flash processes and the 24 unaffected dense
processes. Fifteen of 16 wave medians improve: dense MTP improves 0.12–1.49%,
and Flash ordinary C4 improves 2.97–3.30%. Flash C1/P2048 improves 1.55%; C1/P512
retains +0.021% (-1.070 to +0.236%), despite a lower mean wave time. Its paired
decode-rate median is 0.233% lower despite a higher mean rate. Independent
baseline/self pairs exhibit P512 wave drift of +0.032 to +0.133%; this quantifies
measurement limits without proving equivalence or removing the adverse A/B result.
Dense C4 MTP also retains small decode-rate losses and one +0.301% wave pair.
Short Flash MTP gains are smaller than the pre-repair candidate's, while long
MTP improves. All ranges, numerical tradeoffs and CPU/RSS/sequence/workspace/
startup GPU costs remain in the [repair comparison](../performance.md#final-repair-comparison).

Earlier shared-Down scalar/four-row fusions and shared Gate/Up fusion passed
their oracles but were slower and are removed. Sparse-attention query staging
passed all five cache-format oracles but added 8 KiB of shared memory without a
consistent timing gain and is removed. An iterative top-k alternative likewise
provided no consistent benefit over the sorting network and is removed. The
complete measurements and rejected approaches are retained in performance.
The shared Gate/Up two-chain, value-column-per-warp and four-CTA attention
alternatives also pass their affected oracles but give mixed timing; they are
removed. The selected two-CTA source is restored and rebuilt after these trials.

October 28 continuation: the prior managed checkout had been removed, but its
Git recovery snapshot preserved the complete pending merge. That source has been
restored into a fresh isolated checkout, with `64d10fbf` still the merge target.
The shared workstation checkout has not been changed. New build/test evidence
is saved outside the managed checkout. The recovered merge and qualified repairs
are committed on the local sync branch; publication and deployment are separate.

The initial matched campaign and focused profiles exposed resident launch and
HC injection costs; their complete workload table, unfavorable results and
resource costs are retained in [performance](../performance.md#initial-matched-comparison-before-resident-graph-optimization).
The follow-up captures the immutable resident traversal in two graphs around PLE
instead of per-layer cache/CPU handoffs, while Host placement keeps those handoffs.
HC injection assigns one 256-thread CTA to each stream with a uniform reduction
across token widths. The October 28 Linux candidate/baseline builds, affected
HC/MoE mathematical checks, 116 Python tests and independent template comparison
pass. Windows CLI/server/Supervisor and selected test targets also build; all
three Windows CPU NVFP4 dispatch tests pass. The new resident candidate passes
all 26 public Engine cases, including exact chunk invariance, ordinary/MTP
graph/eager and batch parity, Vision, scoring, trace neutrality, restoration and
concurrent reuse. Bounded Host execution also passes ordinary/MTP K=3 in eager
and graph modes under both 32 and 28 GiB Engine budgets, at the unchanged target
parity criterion. Each run registers 67,948,068,864 Host expert bytes. The complete
matched follow-up results and remaining costs are recorded in performance. The older
26-case pass below predates these two performance changes.

The first follow-up measurements still show a material ordinary-decode cost.
The maintained converter groups HC Down/injection into one native parent, but
the legacy reframer left them independent and therefore selected a separate
injection traversal. An exact-byte regression reproduces the missing parent for
Text and MTP. The reframer now copies those operands into the maintained packing
without requantization; its five focused Windows tests pass. Python 3.11 passes
117 tests and eight subtests on the G4, and the corrected artifact passes all
26 resident public Engine cases with unchanged criteria. Final bounded Host
ordinary/MTP K=3 eager/graph runs also pass at 32 and 28 GiB. Before the repair
above, the native-HC checkpoint used 24 new Flash processes and 24 unaffected
dense processes, with three paired processes per point. At that checkpoint,
Flash ordinary wave medians remained 0.14–7.56% slower;
short-prompt MTP improves 19.92–22.86%, while long-prompt wave medians are near
neutral and their per-request decode rates still fell. Dense MTP remained
1.54–12.74% slower. Full ranges, acceptance, CPU/RSS/workspace and startup costs
are retained in the [final measurement](../performance.md#final-native-hc-comparison-2026-10-28).

The same profile attributes 26.559 ms to the separate shared-expert Down kernel
at Flash C1K0 P512. A subsequent NVFP4 vector prototype folded that projection
into routed Down, preserving per-lane FMA order, warp reductions, expert-rank
summation and the final shared contribution. It built on Linux and Windows and
passed the unchanged block-scaled oracle. A one-process timing probe failed to
show a benefit at either P512 or P2048, with longer wave times than the preceding
HC-parent-only probe. This unpaired probe does not establish a stable slowdown,
but provides no performance reason to retain the added complexity. All seven
prototype files were restored at that checkpoint. The later selected single-row
vector route above uses a different geometry.

The adaptation through `64d10fbf` is committed on the local `sync/upstream-64d10fbf` branch.
Linux G4 runtime checks pass; physical RTX 5090 and native Windows GPU acceptance
remain outstanding.
The combined native Qwen4Exp package keeps resident CUDA experts and adds upstream's
bounded Host expert placement. Its public Engine route also carries the fork's
logprob/scoring readouts, repetition penalty with prompt membership, and routed-expert
observation. The common startup also restores optional concurrency clamping using
complete KV page groups, preserves an explicit pool's tail pages after clamping,
and owns the metadata needed to retry Host placement after discarding a resident
candidate. Their Linux resident/bounded runtime checks pass on the mixed artifact;
the other hardware and artifact qualification limits above remain applicable.

The converter now uses the canonical Qwen4Exp logical bindings, batched NVFP4 layout,
per-Use activation divisors, dedicated Flash-Next template and shared frontend.
`qwen3_8_flash_next` selects upstream's NVIDIA-backed representation;
`qwen3_8_flash_next_mixed` encodes the fork's mixed allocation from official BF16
sources. They are distinct recipes with distinct numerical inputs. The retired
private expert-layout alias is not accepted by the canonical loader or codecs.
Existing fork packed-v3 artifacts require the explicit
`python -m tools.convert.reframe_qwen4_exp SOURCE DESTINATION` command beside their
continuation files. It preserves the original payload and companions, packs byte-exact
BF16 HC Down/injection parents and router/shared-score controls, deinterleaves
per-head attention query/gate code rows and their FP8 scale words, joins adjacent
shared gate/up views, and stores the
previously implicit unit activation divisor. It does not quantize at load time.
The reframing tests cover cross-file tensors, exact control copies, preservation of
the original files, exclusive creation, and removal of partial output after I/O failure.

October 23 qualification history follows. Intermediate failures and incomplete
runs are retained; later explicit passes supersede only the affected checks.
The October 28 continuation above governs the resident graph/HC follow-up.

- Before the resident graph/HC injection follow-up, the resident candidate passes all 26 Flash-Next public Engine cases on
  the replacement G4, including ordinary/MTP Text and Vision, graph/eager, batch
  parity, scoring, exact 3,599-score chunk equality, tracing, context restoration
  and concurrent reuse. The five affected HC/MoE/FP8/QSA oracle suites pass.
  These supersede the intermediate failures below for the resident route.
  The broader run and focused corrections cover 179 selected CTest entries:
  177 pass, with OrcaRouter tokenizer and reasoning-feature real tests skipped
  for missing artifacts. Three script oracles initially
  failed because the test runner had deleted their native probes; rebuilding the
  probes makes all three pass. The maintained template failure below was a source
  defect and was fixed separately. The initial whole-inference timing results
  and their regressions are recorded in performance; follow-up acceptance is pending.
- The initial extension of bounded Host execution to MTP K=3 exposes a target-parity failure at
  both 32 and 28 GiB Engine budgets: the first differing decision has a 0.125-nat
  reference gap, beyond the unchanged 0.05 criterion. Ordinary graph/eager passes
  at both budgets. Each configuration registers 67,948,068,864 Host expert bytes;
  MTP leaves less Main expert-cache capacity. This blocks acceptance until the
  CPU and router corrections below.
- A direct CPU NVFP4 column-invariance regression reproduces batch-dependent
  expert sum order. Fixed route-rank accumulation closes the ordinary regression.
  A second regression with a tiny BF16 neighbour exposes a group-wide switch from
  BF16 dot products to FP32; the safety choice is now per column. All three CPU
  qualification routes (native dispatch, forced AVX2 and forced FMA) pass both
  exact-column checks and their unchanged independent FP64 criteria on the G4.
  The fixed-rank output tile uses 160 KiB of worker stack rather than 16 KiB;
  full 256-column grouping adds 10 KiB of member metadata. The three CPU checks
  and selected product rebuild also pass on Windows. This correction alone does
  not close the real 32 GiB MTP failure: its gap remains 0.125. The 28 GiB rerun
  now passes ordinary/MTP eager and graphs; its remaining first divergence has
  a zero reference gap. Timing is unmeasured.
- Fixing cache capacity at 22 GiB and waiting for all extra experts to become ready
  still leaves a 0.125-gap MTP failure after the longer prompt; those diagnostic
  policies were restored. The standalone 512-expert router separately used a
  vector reduction only at T=1 and MMA at wider windows. Its new exact-column
  regression fails across the short widths. The router now retains the complete
  NVFP4 route's continuous per-lane reduction through 48 columns. The independent
  block-scaled oracle and exact-column checks pass, as do integer MoE and common
  mixed-execution checks. The real 32 GiB ordinary/MTP eager/graph rerun now
  passes at the unchanged criterion, as does the Windows product and affected
  target build. The 28 GiB rerun also passes ordinary/MTP eager and graph
  execution. No cache cap or additional cache wait is retained. Router timing
  is unmeasured; these budget tests do not qualify physical RTX 5090 execution.
- Windows 11, MSVC 19.51 and CUDA 13.3: CLI, server and Supervisor built. Eleven
  focused CPU/public-header/I/O/PLE/schema/metrics tests passed, including NF4 and
  U4 split-plane disk reads, cache reuse, cancellation and corruption handling.
  CUDA event-record fault injection uses the Linux linker wrapper; it was not
  exercised on Windows. The `LIBCMT` linker warning remains.
- Python 3.11 on Windows: 113 artifact/converter/serving-corpus tests passed; three CUDA-only
  conversion tests were skipped. Synthetic text-only and multimodal official
  checkpoints are read back against independent BF16, FP8, NVFP4 and U4 oracles.
- Colab RTX PRO 6000 Blackwell Server Edition, CUDA 13.3: the first final-tip
  product build and HC/LinearSplit/U4 focused checks passed. Three A8 QSA K/V
  oracle failures were fixed by retaining A8 for Q/G and using A16 activations
  for the narrow K/V tail. Criteria were unchanged; the performance cost has
  not been measured. The initial real graph/eager run failed before execution
  because the old router/shared operands were not one contiguous native parent.
  The explicit reframe now prepares those parents; its real-model acceptance
  remains pending.
- On the October 23 replacement G4, both baseline and candidate selected products
  built. Eight focused HC/LinearSplit/U4/MoE/PLE checks passed, including the
  independent selected-expert ID oracles and NF4/U4 disk preparation. A final
  startup recompile and canonical real-model acceptance remain in progress.
- The explicit reframe and full loading/materialization retry now pass on that G4.
  The loader test's upstream-only FP8 MTP assertion was corrected to check the
  mixed artifact's NVFP4 operands and explicit unit activation divisors separately.
  The resident scheduler initially exhausted eight reader slots on 48 layers;
  its shared plan now sizes those slots from the traversal. The first real Engine
  run after that fix still fails graph/eager parity, batched near-tie criteria and
  Host/Device continuation parity, with malformed text. Differential probes found
  exact embedding and PLE values but a 171% relative-L2 jump at the first QSA output.
  The old packed attention parent interleaves query/gate rows per head; the initial
  reframe incorrectly treated them as two contiguous groups. Reframing now copies
  those rows and their FP8 scales into canonical order. An exact BF16/FP8 regression
  test failed before this correction and passes afterward. The corrected artifact
  restores coherent text. The first complete rerun passes 16 of 25 Engine cases:
  admission, page-based concurrency clamping, catalog turnover, conversation chains,
  ordinary tracing, ordinary/MTP structured output and logprobs, causal score,
  Vision/reuse, MTP/reuse, MTP graph/ragged execution, long FP8 MTP, ordinary/MTP
  long concurrent reuse, shared promotion and the materialization ledger. Graph/eager,
  B=2/4/8 target parity, ordinary/MTP Host continuation, MTP tracing, causal prefix
  independence, chunk invariance, hidden readout and MTP target parity fail.
  Bounded Host execution also fails graph/eager parity. Dense 27B prefix, preemption,
  tools and scoring pass; grammar fails with a KV page-reservation error.
  These are unresolved gates, not final-model acceptance or performance evidence.
- The NVFP4 GPU MoE dispatch restores the fork's A16 vector route through 48
  columns in both complete and routed expert Ops. An isolated two-prompt B=2
  check fails with upstream's T=1-only policy and passes with T<=48 at the same
  near-tie criterion. Independent FP64 oracle coverage now includes T=47/48/49;
  the NVFP4 oracle rerun passes. Both ordinary and MTP Host/Device continuation
  checks now pass, while graph/eager still differs on prompt 4. This route change can
  also affect short prefill; no performance or memory improvement is claimed.
- MTP expert tracing now maps valid logical columns to their physical mirror
  offsets and excludes inactive proposal rows and padded verification columns.
  The trace fixture permits the ordinary final decode when only one output remains
  and checks the actual proposal/verification extents. The short ragged-row
  regression fails before the fix and passes afterward, alongside ordinary and
  MTP trace output-neutrality checks on the real artifact.
- NVFP4 prefill uses fixed expert-rank reduction instead of atomic output
  accumulation. With the A16 policy held fixed, the causal-prefix check improves
  from a 4.86628-nat worst difference to zero, and exact hidden-readout equality
  passes. Graph/eager still fails with either reduction, so that remains a
  separate issue. Fixed reduction stores ten FP32 expert contributions per token
  instead of one routed sum; its buffer is 360 MiB larger at T=4096. Total planned
  workspace and performance costs remain to be measured. The oracle also checks
  repeated identical FP32-output prefill calls without relaxing numerical criteria.
- Runtime rechecks native context-copy dependencies immediately before executing
  a reserved unit. Concurrent grammar requests exposed a fresh admission pinning
  a shared KV tail after the resident's unit was reserved; executing the stale
  runnable decision then required an unreserved COW page. The corrected scheduling
  check passes the dense grammar run, including mixed constrained/unconstrained
  requests that previously failed. Flash-Next and broader recovery checks are pending.
- Eager and captured decode now share the same finite QSA context envelope.
  With all other arithmetic held fixed, the eight-prompt graph/eager test fails
  with exact-length eager envelopes and passes with the graph's rounded envelopes.
  MTP proposal and target calls use the same rule. The envelope-only prefill
  trial still fails chunk invariance (worst difference 7.6695 nats, versus 9.4893
  with the original envelopes); it was insufficient. The later joint HC/norm/QSA
  correction below also fixes the prefill envelope.
- The first new repeat-output oracle fixture incorrectly requested FP32 output
  from complete MoE, whose public output is BF16; it was rejected during graph
  capture. The fixture now checks routed FP32 and complete BF16 outputs at their
  actual public boundaries. The corrected NVFP4 oracle suite passes, including
  repeated identical prefill outputs and the 47/48/49-column boundary.
- The next complete Engine run passes 23 of 26 cases. Remaining failures are
  ordinary batching, MTP target parity and prefill chunk invariance. B=1 tokens
  and their top-2 readouts now come from the same generation, removing the earlier
  fixture's dependency on a separate graph run. Both ordinary/MTP Host restoration,
  all three trace cases, causal-prefix independence and exact hidden readout pass.
  Bounded Host execution also passes graph/eager parity under a 32 GiB budget;
  this is Server Edition correctness evidence, not RTX 5090 hardware qualification.
- HC prefill Down now uses four K partitions across the wide-window schedules.
  Its new exact-prefix regression fails on the variable-partition schedules and
  passes with fixed partitioning, alongside the unchanged FP64 HC oracle. This
  alone does not fix model chunk invariance: the worst score difference is still
  8.03798 nats. Boundary captures then find two differing BF16 values at the first
  HC mix, caused by the earlier normalization schedule changing with window size.
  The normalization prefix regression fails on ExpandNorm and CombineNorm before
  retaining the paired 256-thread reduction, then all three norm suites pass with
  unchanged FP64 criteria. With fixed HC partitioning, fixed normalization and an
  instance-wide QSA prefill envelope together, all 3,599 scores match exactly for
  1,024-, 1,536- and 4,096-token chunks. Fixed normalization alone still fails at
  up to 8.8505 nats. These are joint requirements; an envelope change alone was
  insufficient. The HC/QSA timing and resource tradeoffs have not been measured.
- The cleaned-up HC/MoE implementation passes all five affected CUDA suites on
  the replacement G4, with the existing FP64 numerical criteria. The complete
  dense 27B prefix, preemption, grammar, tools and scoring rerun passes, as does
  page-based concurrency clamping. Python 3.11 passes all 116 tests, including
  the three CUDA-only conversion tests skipped on Windows. The first Linux
  Python attempt lacked pytest; installing the test dependencies resolves it.
  The current Windows build and all eleven focused CPU checks pass again.
- Flash-Next ordinary and MTP batch parity still fail after the joint chunk fix.
  Ordinary B=2/4 first differs at a reference gap of 0.125; MTP K=3 also has a
  3.375-gap failure. These block acceptance. FP64 dot products against captured
  head weights locate the difference before the vocabulary head. Layer and
  attention captures then find differing BF16 query/gate values from identical
  attention inputs: single-column and segmented batched projection use different
  K reductions. Earlier small-HC and shared-expert vector-policy trials
  did not close parity and were restored; the combined trial also introduced
  larger MTP failures. No timing gain is claimed for these trials.
- The segmented FP8 attention projection now retains one A16 reduction through
  48 columns. Its new seeded finite-code regression fails at T=2/8/16 as well as
  the former A8 widths and passes after the fix, alongside independently decoded
  FP64 checks. Permissive-policy cases through 48 now use the existing A16
  criterion, matching the selected arithmetic, rather than the former A8 budget.
  Ordinary B=2/4/8 generation passes its 48-token check. The longer ordinary
  control still fails at a 0.875 reference gap, and MTP still fails at this stage.
- The joint short-window correction also keeps HC Down/Up schedules uniform,
  retains FP32 shared-expert activation through the vector route, and keeps A16
  activations in the FP8 output projection through 48 columns. The eight-prompt,
  96-token ordinary control and MTP K=1/3/5 now match exactly in the focused run.
  The HC, NVFP4 MoE and FP32-scale FP8 suites pass their FP64 oracles. Omitting
  shared-vector execution reintroduces a 1.375-gap ordinary failure and MTP
  failures; omitting output A16 reintroduces MTP gaps up to 2.75; omitting uniform
  HC reintroduces 0.125-gap ordinary/MTP failures. All three are necessary for
  this workload. Broader final-configuration acceptance and matched timing and
  resource comparisons remain pending.
- QSA short-window probability partitions restore the fork's
  width-independent policy. FP16 probabilities depend on each partition's
  maximum; varying the partition count changes represented values. The direct
  single-query versus 48-query regression fails on the upstream partition policy
  at 512- and 2,053-token histories and passes with fixed partitions. The full
  sparse-attention suite passes across all five KV encodings with unchanged FP64
  criteria, including the new exact-column check.
  Its first build exposed a test-helper argument mismatch, now corrected. The
  Windows fixture uses the existing independent software FP16 codec where MSVC
  lacks `_Float16`; Linux keeps its native FP16 oracle.
  The latest Windows CLI, server, Supervisor and affected Op test builds pass;
  native Windows GPU execution is still unqualified.
- The broader suite finds a maintained Flash-Next template continuation failure:
  thinking-off assistant prefill omitted the empty think block required by the
  serving contract. The template now matches generation's prologue while retaining
  exact content whitespace. The earlier dedicated whitespace fixture conflicted
  with that contract; it now checks both the required prologue and unchanged
  whitespace. All nine template tests, including the independent Python/C++
  renderer comparison, pass on Windows after the correction.
- A previous replacement Colab session expired before its final logs were saved.
  An unrestricted test/microbenchmark build exhausted its disk while copying the
  new entry. That copy and affected links are invalid and must be repeated after
  reclaiming task-owned build outputs. Completed focused results above are not
  a full-suite or final-model pass. The new instance builds selected targets and
  saves completed logs locally.

RTX 5090 and native Windows GPU execution remain qualification gates. Matched G4
resident results against the pre-sync fork are complete and retain the regressions
above. Server Edition results do not qualify the desktop PRO 6000 or an RTX 5090.

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

## Upstream `81c8ce09..64d10fbf` (assessment, October 19, 2026)

**Status: In progress; full merge and GPU qualification incomplete.** Implementation
was authorized after the assessment. The contained grammar trial is carried in
`58cd4222` on `sync/upstream-64d10fbf`; the first 24-commit upstream merge is
`d3ffa5b3`. The original assessment comparison is committed
`workstation` at `2b62e8e5` against upstream
[`64d10fbf`](https://github.com/Neroued/ninfer/commit/64d10fbf9f8e6bfc79ab0570b94674c5b9997929).
There are 77 upstream commits and 166 fork-only commits after the common ancestor
`81c8ce09`. A dry merge reports 187 unique unmerged paths, including 59 in Ops,
25 in the native Qwen4Exp package, 16 in Qwen3.5 and 27 in tests. Directory-rename
messages are additional to that path count. Similar features exist on both sides;
patch-equivalence filtering does not eliminate any of the 77 upstream commits.

The user's target is **both RTX 5090 and RTX PRO 6000 Blackwell, with native Windows
retained**, and a fork that can add capabilities beyond upstream. Preserve the
GPU-resident Flash-Next execution route. Also assess upstream's CPU/GPU expert
execution as a useful route for the 5090; do not blanket-reject it because the PRO
6000 has more VRAM. Runtime placement should follow the supported model's actual
resource plan, with separate qualification for resident and offloaded execution.

### What upstream's Flash-Next design addresses

The upstream model reference accounts for 63.281433 GiB of Main routed experts,
27.126954 GiB of PLE NF4 tables and about 10.257616 GiB of fixed GPU weights with
all components enabled. It places the complete Main expert source on Host, PLE on
disk, and a capacity-bounded expert subset on GPU. Small-column missing experts
run on CPU; larger columns stream complete bank windows to GPU.
`qwen4_exp::Startup::plan` rejects the non-Host candidate, and `materialize`
requires Host backing. This is a different execution/representation choice from
the fork's existing resident expert banks, mapped U4Z8 PLE and offline-baked MTP
banks.

This design is consistent with upstream's stated RTX 5090 target; that attribution
is an inference from the documented target and placement. It is not evidence of
a speed gain or loss. At this revision upstream's performance index explicitly
says that Flash-Next has no published result page in its serving campaign.
See the upstream [model/weight accounting](https://github.com/Neroued/ninfer/blob/64d10fbf/docs/maintainer/qwen4_exp-model.md)
and [execution reference](https://github.com/Neroued/ninfer/blob/64d10fbf/docs/maintainer/qwen4_exp-execution.md).

### Improvements worth evaluating

| Priority | Change | Recommendation and limits |
|---|---|---|
| First contained trial | `0a8530d4`: length-bounded JSON string masks | Bring forward if the native CPU grammar and independent schema/language checks pass. Counted character-unit state avoids expanding each remaining length into a separate matching state; upstream adds boundary, UTF-8/escape, speculative-preview and rollback cases. It applies to either GPU. No model throughput improvement is established. |
| High | `19367582`: calibrate the actual configured CUDA Graph set before final KV sizing | Adapt. Our startup queries device-wide free memory and enforces a desktop reserve, but that does not replace measuring graph allocation and retained driver memory. Carry both protections into the combined planner. Keep explicit KV capacity/features, bounded OOM recovery and truthful startup telemetry. The upstream commit itself discloses an unexplained 14 ms decode regression in one comparison, versus about 1 ms in reverse order; no inference gain is established. |
| High for 5090 Flash-Next | Main expert cache, CPU/GPU MoE and bounded PLE preparation | Adapt as additional native placement/execution capability, while retaining GPU-resident execution when it fits. This is a capacity feature, not yet a demonstrated speed improvement over resident execution. Account for full Host backing, registered ranges, worker CPU cost, GPU cache/windows, RAM, disk I/O and desktop headroom. |
| Kernel candidates | HC, QSA, fused PLE gate, 10240-wide offset RMSNorm, NVFP4/block-FP8 MoE | Compare against the fork's actual kernels and operands. The fork already carries optimized HC/PLE/QSA and recent FP8/NVFP4 dispatch tuning. A new upstream implementation is not automatically an improvement over those paths. Check supported widths, public BF16 rounding boundaries and independent numerical oracles before A/B timing on either GPU. |
| Architecture/maintenance | Op family organization, LinearSplit, shared Qwen frontend, complete continuation and KV-prefix ownership | Adopt deliberately across callers and both native architectures. Explicit ownership and fewer duplicated context implementations can benefit maintenance; the moved files alone are not a runtime speed result. Preserve recovery, scoring, grammar, readouts, expert tracing and fork prefix-admission behavior. |
| Representation/quality candidate | NF4 PLE, imported block-FP8 MTP, source-backed converter and dedicated Flash-Next template | Evaluate independently of offload. These alter actual stored operands, logical bindings and/or template behavior. Importing the MTP block-FP8 representation avoids the fork's NVFP4 bake, but changes memory requirements and numerical inputs; quality, acceptance rate and speed are unmeasured. Existing v3 framing does not make the two Flash-Next artifacts interchangeable. |

### Portability and fork protections

A straight merge does not satisfy native Windows:

- `src/core/CMakeLists.txt` unconditionally requires Linux x86-64, including for
  resident-only use. Core also links `hwloc>=2.5` and `liburing>=2.5`.
- CPU topology/ISA discovery uses `<cpuid.h>`, and MoE worker compilation uses
  GCC/Clang `-mavx*` flags. Port discovery, affinity, floating-point mode and
  per-ISA compilation/dispatch to MSVC without making CPU support a configure
  failure for an otherwise supported GPU route.
- The new Core file/read-queue layer replaces the fork's Windows-aware Artifact
  file I/O with POSIX direct reads, `io_uring`, `eventfd` and `poll`. Preserve
  alignment, exact-read/error handling, cancellation, buffer ownership and
  completion-before-reuse with a native Windows implementation.
- Graph calibration and new startup candidates must retain NVML device-wide
  accounting, the configured desktop reserve, optional concurrency clamping and
  the fork's blocking CUDA synchronization default.
- Forward-port fork tuning and correctness changes through renamed files. In
  particular, upstream's moved FP8 TMA kernel still releases an input stage once
  per consumer warp, while the fork counts/releases every consumer thread.
  Do not overwrite the fork's staged-input handoff fix. Re-run its alternating
  identical-weight/cold-buffer regression after migration.
- The comparison baseline excludes the shared checkout's 35 tracked modifications
  and five untracked path entries. Active HC fusion, MTP/proposal and FP8 work
  overlaps this sync. Reconcile those changes when their owner finishes; absence
  from this committed baseline does not mean their behavior is superseded.

### Complete change inventory

Every one of the 77 upstream commits appears once below. Candidate means reviewed
for purpose and integration constraints; it does not mean implementation,
numerical qualification or hardware performance qualification has passed.

| Area | Upstream commits | Initial disposition |
|---|---|---|
| References and planning | `84d1cf02` `dc8b58fb` `30e3a92f` `be9ad8a3` `fbd285b5` `c6f47b52` `f1ace85e` `6e772ee3` `93fb1874` `64d10fbf` | Candidate: reconcile active references with dual-GPU and Windows contracts; completed task plans should not become new permanent authorities. |
| Qwen4Exp Ops | `f2620157` `be12563a` `d6a8de0e` `1a9d676e` `30b489f6` `d02703e2` `fc4e3331` `e70fc17f` `47cdc08a` `fbcd8504` `6a4104fd` `f7d3090b` `70a11f12` `5989682a` `642d588f` | Candidate: compare GDN, HC, PLE and QSA with fork implementations and preserve observable rounding/state semantics. Benchmark relevant production shapes on both GPUs before replacing optimized kernels. |
| Conversion and representations | `d8c0fa6f` `7c4c5e3e` `9cdbf221` `4de9e425` | Candidate: review NF4 PLE, bank bindings, block-FP8 MTP and dedicated template. These change artifact operands/semantics; no checkpoint conversion or source download is authorized by this assessment. |
| Mixed-row KV | `a0d27b22` | Candidate: useful for QSA row densities; validate exact KV publication, prefix and Main/MTP frontiers. |
| Op family organization | `3124dffe` `90d25edd` `7ee63999` `20d3a13f` `1844d861` `4eb2c042` `d53eb7ae` `6733be72` `c012155c` `4366c7b6` `8cc45088` `e6585d15` `ae54c1a9` | Candidate: adopt explicit family ownership and LinearSplit where useful; migrate fork callers, tuning and correctness fixes rather than taking renamed upstream files wholesale. |
| Core memory/execution split | `6a42de35` | Candidate: separate physical backing from execution views while retaining native Windows allocation, device-wide memory accounting and the desktop reserve. |
| CPU/GPU expert execution | `74671a4e` `c18d72db` `610f79e9` `8cf26bb6` `8e3dff94` `c7c36e88` `fddba033` `c996baa8` `7ff78ee8` `545a171a` `17bea0bc` `7e691b0b` | Candidate: important for 5090 Flash-Next feasibility. Port platform/ISA contracts; retain an all-GPU route when the complete selected model and runtime fit. Offload is no longer blanket-deferred. |
| Graph startup calibration | `19367582` | Candidate, high priority: use actual configured graph allocation, bounded recovery and truthful memory diagnostics; preserve NVML accounting, explicit capacity and desktop headroom. |
| PLE preparation pipeline | `8cea711b` `333b4808` `5ac4d9f0` | Candidate: bounded async row preparation and cache can help RAM-limited execution. Adapt Linux io_uring to Windows completion semantics and measure I/O/CPU/latency costs. |
| PLE gate and RMSNorm | `1b50c72d` `569861a3` | Candidate: inspect fused PLE and 10240-wide norm at actual fork call sites; distinguish new format/semantic support from improvements to existing fused paths. |
| NVFP4 and block-FP8 GPU MoE | `69c098bf` `df4ff5fe` | Candidate: compare resident decode, speculative widths and prefill with the fork's NVFP4 bank kernels; block-FP8 expands supported stored operands, with additional memory cost. |
| Bounded JSON string masks | `0a8530d4` | Candidate, first isolated trial: CPU-only grammar change and independent language/schema regressions; applicable to both GPUs without changing model placement. |
| Shared Qwen and context ownership | `9cfd5646` `48d8d185` `e9a7de36` `ee5a5ffa` `4423e0ac` | Candidate: reduce duplicate frontend/context ownership and make complete continuation/KV-prefix lifetimes explicit; preserve fork scoring/readouts, prefix admission policy and pressure recovery. |
| Qwen4Exp integration and cache priorities | `088e8da4` `985121d4` `4f31ad4a` `afe43480` `439e1075` `3444964f` `f8a774f7` | Candidate: adapt into the native architecture rather than wholesale replacing the resident fork implementation with compulsory host backing. Qualify text, Vision, MTP, eager/graphs, scoring, reuse and trace hooks on each route. |

### Merge boundaries and acceptance

Dry merges from the same committed fork baseline give these candidate boundaries.
These are conflict counts for independent previews, not predicted effort or a
count of new failures after an earlier stage has been resolved.

| Upstream boundary | Cumulative incoming commits | Unique unmerged paths | Main concern |
|---|---:|---:|---|
| `f1ace85e` | 24 | 52 | Overlapping Qwen4 Ops, conversion and documentation |
| `6e772ee3` | 38 | 99 | Move fork Ops/callers/tests/tuning into the new family ownership |
| `6a42de35` | 39 | 111 | Explicit memory backing and CUDA execution views |
| `19367582` | 52 | 134 | Windows CPU execution, expert placement and graph-calibrated startup |
| `4423e0ac` | 68 | 157 | PLE I/O, new MoE/grammar, shared context and snapshot ownership |
| `64d10fbf` | 77 | 187 | Reconcile complete Qwen4Exp Text/Vision/MTP and graph composition |

Keep the merge-based sync method. A contained trial can apply an upstream diff in
an isolated checkout for evidence; that is not a cherry-picked sync or a claim
that upstream ancestry has advanced. Record integrated/deferred/reverted decisions
when the actual merges are prepared. Do not create a cosmetic zero-behind merge
that silently drops useful upstream work.

Completion requires the selected design across native Windows and Linux, with
both RTX 5090 and RTX PRO 6000 treated as supported targets. Report unavailable
hardware runs separately; neither upstream 5090 results nor PRO 6000 runs qualify
the other GPU. Use explicit artifacts and select resident/offload baselines and
workloads before measurement:

- Native builds and affected host/protocol tests, including Supervisor and the
  serving extensions. A CPU-only grammar trial does not qualify the full merge.
- Each changed production Op against its independent FP32/FP64 oracle or exact
  codec reference, with independently decoded stored scales and relevant model
  shapes. Preserve defined intermediate/state boundaries.
- Public Engine tests for Qwen3.8-27B and Flash-Next: ordinary/speculative execution,
  eager/graphs, ragged batches, Vision, scoring/logprobs, grammar/tools,
  prefix reuse, Snapshot/Replay, cancellation and resource release.
- Matched whole-Engine measurements at representative prompt lengths,
  cache-hit ratios, concurrency and speculative widths on each GPU. Compare to
  the fork's optimized baseline, with identical operands/configuration for a
  kernel comparison; report artifact/representation changes separately.
  Include TTFT/prefill, decode, complete latency/throughput, MTP acceptance,
  worst cases, VRAM, Host/pinned memory, CPU and I/O costs.
- Offload additionally needs capacity/working-set boundaries, cache-miss and
  streaming behavior, startup/graph OOM recovery, transfer completion and
  desktop-coexistence evidence. Performance and memory benefit remain
  unverified until these measurements exist.

### Contained grammar trial

Upstream `0a8530d4` applies cleanly to an isolated `2b62e8e5` checkout, preserving
the fork's existing MSVC adaptations. Release builds of `ninfer_grammar_test`,
`ninfer_json_schema_test` and `ninfer_regex_choice_test` pass with MSVC 19.51 and
CUDA toolkit 13.3.33 configured; these targets execute entirely on CPU.
All five selected CTest entries pass, including independent Python 3.11.15 JSON
Schema and regex/choice oracles. The baseline also passes the three native tests
and both Python oracles with the corrected environment. The candidate adds
counted-unit/rollback/speculative-mask tests, boundary lengths through 200,
UTF-8/escape checks and a dense-token mask oracle.

The first oracle invocation inherited Python 3.14 and Windows CP1252; Unicode pipe
encoding failed and the run was stopped. An attempted Python reconfiguration
also referenced a transient uv environment that no longer existed. Both setup
issues were corrected with a persistent task-local Python 3.11 environment and
`PYTHONUTF8=1`; the subsequent baseline and candidate runs pass. Existing MSVC
vendor conversion and nodiscard warnings remain. The unpaired Python schema
oracle runs took 3.44 s on baseline and 6.33 s on candidate; they are correctness
runs, not a controlled suite-performance comparison, and the elapsed increase
is unexplained.

A matched, alternating CPU probe comparison passes every independent acceptance
check. For `minLength=256,maxLength=512`, median cold-process/compile/mask time
falls from 94.36 to 22.48 ms (76.2% lower). The three shorter-bound workloads
instead rise by 1.6%, 7.2% and 7.6%; worst median increase is 1.01 ms. These
costs are retained, not dismissed as noise. This is a 257-token byte/EOS probe,
not the production tokenizer, serving TTFT or model decode. See the full
[method and results](../performance.md#bounded-string-grammar-trial-2026-10-19).
Recommendation: carry the candidate into the sync for bounded-string handling,
with the small-case costs and production-vocabulary uncertainty explicit.

Source/history review, dry merges and the contained Windows CPU trial are complete.
The trial was committed as `58cd4222` after implementation authorization. The full
integration, model numerical comparison, model conversion and dual-GPU qualification
remain open. The first source merge is recorded locally; nothing has been pushed or deployed.

Stage 1 carries the fork's FP32-row-scale FP8 input kernels, short-window schedules
and BF16 GDN control operands into the upstream prepared-operand interfaces. The
removed `gdn_gating_proj` and gated-RMSNorm aliases are migrated across native
callers. Independent GDN fixtures add the represented BF16 control words and FP32
row scales; upstream QSA and the fork's block-key indexer retain distinct test names.
The first Windows build exposed stale geometry constants in the moved FP8 output
helper and missing aggregate-template deduction on MSVC; both are corrected.
The Release runtime, GDN control projection and GDN snapshot targets build with
MSVC 19.51/CUDA 13.3.33. Default parallelism also exhausted `ptxas` allocation;
the successful retry used four jobs at below-normal priority. These are build
checks, not desktop GPU qualification. Linker CRT-library conflict warnings remain.
The Colab baseline full build also found
an existing DFlash benchmark using pointer access on the ModelInstance variant;
the same benchmark-only correction is applied to both trees, without changing the
baseline inference implementation.

The first boundary also brings generic paged-NF4 framing/codecs, mixed-row KV
spans, PLE state Ops and the independent upstream HC/QSA Ops into source. The
fork's resident Flash-Next composition continues to use its existing HC, PLE and
block-key QSA routes at this boundary. This does not replace their optimized
implementations or establish a performance gain. The upstream flat
`tools/convert/qwen4_exp.py` is deferred until the native binding/representation
merge: it would otherwise be shadowed by the active fork package of the same
name. The active converter, recipes and representation references stay together.
Generic NF4 encoding support is separate from an advertised runnable NF4 model.
The Linux qualification snapshot initially omitted a newly restored private
header and retained one old replay-record call signature; both were corrected.
Colab's disk then filled while holding two complete build trees and both explicit
artifacts. Link and synthetic-converter I/O failures from that condition are
invalidated checks, with reruns after removing disposable benchmark binaries and
download caches. Hardware/numerical qualification is still in progress.

The converter rerun then exposed an actual integration error: the U4Z8 method's
return was lost when the stored-scale method was added. Restoring it makes all
85 artifact/converter Python checks pass under Python 3.11.17; three CUDA-only
quantizer checks are skipped by the CPU Torch environment. The real-model
baseline also has two explicit prerequisite failures with the selected 27B
artifact: the 35B MoE test rejects its dense architecture, and the DFlash test
requires a missing `dflash` component. Neither is a candidate regression or a
passed qualification of those routes.

At the second boundary, Op-family ownership and plain segmented projections are
migrated. The fork's BF16 vocabulary TopK and LinearAdd profiles remain
under `linear/topk` and `linear/add`; source integration retains those profiles
rather than dropping them during the file moves. Shared operand validation
accounts for the actual BF16/FP32 FP8 scale words. The retained masked D256 KV
form uses the upstream shared writer with fork admission checks; its numerical
and performance acceptance remains pending. The FP8 TMA handoff still counts and
releases every consumer thread after its reads. Linux and Windows builds of
this second boundary pass for the model runtime. Linux CLI/server and Windows
CLI/server/Supervisor build successfully. Six selected Windows host schema,
serving-option and Supervisor checks pass. CUDA arithmetic qualification remains
in progress; these product builds do not establish its result or a speed gain.
The first GPU snapshot passes 35 of 38 selected checks, including its 27B Engine
cases, BF16 GDN control oracle and upstream HC/QSA/norm/PLE cases. All three GDN
input checks abort on the newly represented FP32-scale profile: admission was
extended but backing validation still called the BF16-only validator. This is
an integration failure, not a numerical pass. The first snapshot is corrected
for reruns; the second boundary consolidates both represented scale formats in
the shared validator. Its Linux Engine library compiles; linking the products
also found fork Chat fields still calling the boolean helper removed upstream.
Those fields migrate to the equivalent shared request validator, retaining
their null/default and boolean schema behavior.
The first second-boundary qualification build also exposed a malformed merge of
the TopK fixtures. The complete fork oracle, including Q8, FP8, BF16 and mapped
Q4 heads, is retained under the new test target and passes on Colab G4. The NVFP4
LinearAdd fixture also required its complete shared oracle runner, retaining the
fork's A16/A4 boundaries and graph/ragged cases. All 88 artifact/converter Python
checks pass with CUDA Torch enabled on the second G4.

The selected second-boundary matrix initially passed 26 of 29 entries. The three
failures were masked batched BF16 KV writes missing from the shared writer, an
FP32-scale GDN snapshot losing precision in its short A8 contraction, and a GDN
record fixture hardcoding input width 5120 for the 2560-wide parent. The shared
BF16 writer now handles masked request rows with the existing exact codec oracle;
the FP32-scale convolution route keeps BF16 activations through 32 aggregate
columns; the fixture follows its actual parent width. All affected GDN/KV checks
pass after correction. Performance of the convolution fallback is unmeasured.

The first complete Flash-Next candidate run also failed MTP batch target parity,
although its other cases passed. The merge had replaced the fork's fixed paired
split-8 control reduction with token-dependent schedules. Restoring the fixed
reduction makes the real-model parity case pass, and an added exact shared-column
regression passes for BF16 and FP32 controls over decode, speculative and prefill
widths. Control accuracy remains checked independently in FP64. Its capacity test
now verifies the entire interval instead of assuming the largest extent needs no
scratch. The original fork baseline passes all 24 Flash-Next real-model cases.

The Core boundary (`6a42de35`) is adapted with explicit physical Device tags,
DeviceBuffer backing, non-owning MemoryArena spans and borrowed CudaExecutionView
operands across the fork's native routes. NVML device-wide reserve enforcement
remains at physical allocation, and blocking synchronization remains the default.
Windows CLI/server/Supervisor builds and four selected native host memory/API
tests pass. The complete Linux CLI/server/test build passes; its 174-entry CTest
run and full real-model rerun are in progress. Earlier incomplete BF16 Linear and
FP8 SwiGLU fixture merges, remaining pointer/view migrations and retained stream
signatures were caught and corrected during compilation. Independent fork shape,
BF16 head and graph cases are retained with the shared mathematical runners.
This is source/build evidence, not full qualification.

The Colab hosts are RTX PRO 6000 Blackwell **Server Edition**, driver 580.82.07,
CUDA 13.3.33 and Python 3.11.17. They are not the workstation card or an RTX 5090.
The resident artifacts are pinned 27B NVFP4 and fork mixed Flash-Next v3 inputs.
No matched inference speed comparison, full-model offload result or physical
RTX 5090 qualification has been established by this sync yet.

As a separate baseline for later ports, upstream `64d10fbf` passes 14 selected
CPU topology/executor, asynchronous read-queue, expert-cache, NF4 and CPU MoE
checks on Linux with a Colab RTX PRO 6000 Blackwell Server Edition, CUDA 13.3
and Python 3.11.17. Its 71 artifact/converter checks pass with Torch 2.14.1+cu130,
including the CUDA quantizer cases. This is upstream primitive qualification on
that host, not integration into the fork or a full-model offload measurement.

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

## Upstream `d44ab584..81c8ce09` (October 8, 2026)

All 44 commits are merged from `origin/master` at `81c8ce09`. The sync adopts the upstream
preemptive runtime, with request-owned output/grammar state and Snapshot/Replay recovery.
The prior non-preemptive admission planner and its unsettled-StateImage workaround are
superseded together. The experimental hybrid scheduler remains separate; this merge makes
no claim of a latency or throughput improvement over the previous fork.

| Upstream | Change | Disposition |
|---|---|---|
| `a667efdd` `417eb3d6` `e621c7d6` | Device-SM-derived RoPE, normalization, MoE and attention launches | Integrated for the PRO 6000 as well as upstream's 5090 |
| `064965c7` `75a89050` `b9114396` `a8e212ac` `911d34db` | Explicit KV publication streams, cleanup synchronization, context replacement, preemption, preparation costs and cache-source preservation | Integrated; native Flash-Next storage, transactions, replay and MTP migrated to these contracts |
| `f854788b` `c772812b` `624dcdd4` | TTFT workloads, diagnostics, timing quantiles and contract documentation | Integrated |
| `abb7f14f` | Prometheus runtime/request metrics | Integrated alongside fork admin telemetry; Supervisor controls now use the shared host-context byte budget |
| `68c54356` | Preserve cache boundaries through template trimming | Integrated |
| `7e2973ee` `08902f73` `eb6696ae` `41e50d0d` `8ffa3131` `2734a56e` `81c8ce09` | XGrammar CPU core, GBNF, JSON/schema, constrained tools, choices, regex, composition, tuple schemas and bounded numbers | Integrated; old fork structured-output adapter removed; native Flash-Next MTP uses per-position target masks |
| `a643abcd` | Maintainer guidance | Integrated with fork contracts |
| `35e9b5c8` `04ded76f` `8f0aa859` `58ab3f22` `cd521ab7` | Five Q8 Linear shapes at K=2560/6144 | Integrated |
| `471924b0` `f58e32e0` `714c5149` `bcdcbd45` `8beb4cdd` `16bdb491` `2553e26e` `cf91c818` `07e1f8c3` `d0f91cd9` `070fa61a` | Eleven BF16 Linear shapes | Integrated for wide prefill; fork narrow full-vocabulary/square reductions retained through 48 columns after the new dispatch failed the existing Flash-Next target-parity gate |
| `7971ff18` `c02a07b4` `dd73c88e` `9879d052` `a64b5eea` `b21780f3` `15cba227` | Seven BF16 Vision/merger shapes | Integrated; supersede the fork's duplicate Vision shape file |

The fork retains desktop reserve and optional concurrency clamping, blocking CUDA synchronization,
serving logprobs/scoring/System One, Supervisor, and both native model packages. Active and paused
requests participate in quiescence. Request-owned prompt readouts and reasoning features survive
lane changes; restore rebinds device readouts and reinstalls sampling counts. Candidate-logprob
readout supports the upstream per-column grammar masks and excludes padded speculative columns.
Flash-Next replay preserves native QSA/GDN/PLE and continuation state, with its MTP KV frontier
one token behind the target; it does not use the Qwen3.5 MTP bridge.

MSVC adaptations use exact wide-product comparison, saturating prefill-cost arithmetic and exact
decimal MiB parsing without `__int128`; XGrammar receives the MSVC unreachable marker. These
adaptations preserve the Linux arithmetic semantics. The public header now includes `<array>`;
KV-directory ownership checks use the C++20-supported `use_count()` API. Supervisor pressure and
reuse displays now consume actual preemption/spill and root/checkpoint telemetry; occupied Host
bytes already include reserved bytes and are not added twice.

Qualification uses a Colab G4 RTX PRO 6000 Blackwell Server Edition (96 GB), CUDA 13.3.73,
GCC 13.3 and Python 3.11.17. The explicit artifacts are Qwen3.8-27B NVFP4 v3 and the native
Qwen3.8-Flash-Next mixed v3 artifact, including Vision and MTP. Native Windows verification uses
MSVC 14.51, CUDA 13.3, Release `sm_120a`, and the LGPL FFmpeg bundle. Windows builds run below
normal priority; GPU execution for this sync is on the G4, not the shared desktop.

- Linux/CUDA and native Windows full builds pass. Eleven focused Windows CPU tests cover public
  types, logprobs, serving schemas/options, telemetry and Supervisor. Python 3.11 syntax checks
  pass for the 24 changed Python files. The existing Windows CRT linker warning remains.
- The 158-entry G4 CTest run initially had six failures and 17 missing-artifact/source skips.
  Five failures were missing `jsonschema` dependencies or stale schema-adapter assertions;
  the focused rerun passes. The sixth was a superseded duplicate Vision BF16 fixture outside
  upstream's supported patch/merger domains. The adopted independent BF16 oracle covers those
  supported shapes and rejects invalid domains; its full rerun passes, including after restoring
  narrow decode dispatch. This is a full run plus affected reruns, not a single green full run.
- Qwen3.8-27B grammar (MTP/DFlash2), constrained tools (MTP, Snapshot and Replay), causal scoring,
  reasoning-feature readouts, and the complete prefix suite pass. Native transaction tests for
  ordinary, MTP and DFlash2 check exact ledger occurrence
  counts after lane rebinding/replay, preserved prompt readouts, and resumed sampled/candidate
  logprobs against independent FP64 normalization of the actual target BF16 column.
- Public pressure tests cover two requests with 192-token prompts, 256 output tokens, 128-token
  prefill chunks and a 512-token KV pool. Replay, 512-MiB Host Snapshot, paused cancellation and
  replay cancellation pass for 27B ordinary/MTP/DFlash2 and Flash-Next ordinary/MTP; constrained
  Flash-Next MTP passes the same cases. Checks include penalties, request-specific candidate IDs,
  durable prompt readouts, stream/metric accounting and resource release. The 27B MTP result
  initially included intermittent prompt-readout comparison failures; their FP8 TMA cause and
  the subsequent qualification are recorded below.
- Flash-Next passes all 21 cases across the final full run and one focused fixture rerun:
  ordinary/MTP text, graphs/eager, ragged batches, context spill/reuse, structured output,
  logprobs, causal scoring/hidden readout, chunk invariance, image/video and long concurrent
  reuse. The full run passed 20 cases and rejected the correct red answer `#FF0000` because the
  image fixture required the word `red`. Accepting that exact RGB representation makes the
  focused image/reuse/video case pass. This is not a new numerical acceptance tolerance.

Failures resolved during qualification include a native Flash-Next first replay permit using
`E` instead of its MTP backend's `E-1` frontier; invalid continuation-tail copying from a scoring
buffer; padded MTP candidate-readout columns; and duplicated response-format parsing. Tool-bearing
prefix fixtures now honor the adopted default-EOS constraint contract. A newly added native test
also needed the ordinary backend's implicit one-token row-count convention.

Targeted CUDA initialization checking identified a whole-frame MTP output transfer containing
uninitialized unused K/C array tails. Live outputs were written and consumers respected their
row/count bounds; these tails are separate from prompt-readout storage. Fixed-size output frames
are now initialized once at startup for both native packages. The full targeted MTP Replay
`compute-sanitizer --tool initcheck` rerun passes with zero errors; the initial run reported
4,045 accesses to those undefined tails. The separate prompt-comparison failure came from the
FP8 projection path described below.

The first Flash-Next run passed 20 of 21 cases. Its target-parity failure was a 0.25-nat first
difference at token 84, reproduced by ordinary B=8 as well as MTP K=1/3/5. Restoring the fork's
matched narrow BF16 reductions removes that failure without changing the 0.05-nat acceptance gate;
upstream's wider prefill schedules remain. No end-to-end performance improvement is claimed.

An initial added recovery check compared generated tokens/logprobs with sequential execution.
The first differences were on requests that were never preempted (27B ordinary token 82,
MTP/DFlash2 token 131; Flash-Next ordinary token 42), so that check confounded recovery with batch
arithmetic. The replacement checks per-request token/readout association, distinct candidate IDs,
probability normalization and durable prompt values, alongside the native actual-column oracle.
It does not claim exact generated-token equality across batch shapes. Original failed runs remain
part of the qualification record.

The unchanged 0.002-nat prompt-readout comparison failed intermittently in 27B MTP Replay,
including the sixth run of a later bounded repetition. Sequential fresh prefill reproduced it
with and without graphs. Captures showed changing hidden values while FP64 normalization of the
actual BF16 logits agreed with the readout kernel. The first divergence was a 32-row stripe in
the first token of an FP8 GDN projection tile, already present in its FP32 accumulators.

The existing FP8 TMA pipeline released staged inputs through one lane per consumer warp.
Every consumer now contributes its own release arrival, and the empty barrier counts all those
readers before the producer can reuse the stage. An isolated test alternating two identical
resident weights reproduces the original defect with both real and synthetic operands; simply
repeating one hot weight did not. Reverting the fix reproduces it again at iteration 1,371.
The selected handoff passes 16,384 alternating-weight repetitions for each operand set and
256 real-model requests with exact duplicate-projection comparisons. A full-consumer CTA barrier
also passed but was unnecessary once every reader supplied its own release edge. The regression
retains the independent mathematical oracle and adds 4,096 complete-output weight switches.
An initial fixture build used an incompatible byte-buffer helper overload; the direct byte
preservation check fixes that compilation error.

Final production builds pass on Linux and MSVC. All nine affected FP8 Op suites pass, including
the new repeated-weight regression. Another 256 fresh-prefill requests with graphs enabled have
no prompt-readout mismatches, and all 12 MTP Replay repetitions pass. The four public pressure
scenarios pass again for 27B ordinary/MTP/DFlash2, Flash-Next ordinary/MTP and constrained
Flash-Next MTP. Two initial Flash-Next rerun commands omitted its required full proposal head and
were rejected before inference; the correctly configured reruns pass. Targeted TMA `synccheck`
and `racecheck` report zero errors and zero hazards. The focused timing distributions and their
limitations are recorded in
[FP8 TMA qualification](../performance.md#fp8-tma-staged-input-handoff-2026-10-08).

Windows GPU execution, end-to-end speed changes, and desktop coexistence on this revision are
unmeasured. G4 correctness evidence does not establish deployment or a workstation speed gain.

Historical qualification in later sections describes its named revisions, not this runtime.

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
| Structured output | `bb7b7305` `2cbdb2b6` `bb88a776` | `ff78ceae` `f9800e26` | Originally XGrammar JSON object/schema and required tools; superseded by upstream OutputSession and all-backend per-column masks in the October 8 sync |
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

The DFlash2 residue of `ddeeec19` and the old all-constrained width-one path are not carried.
The October 8 sync uses upstream context reservations and per-column speculative grammar masks.

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
| Request model names (`d17f6a52`) | the public model id only | `--model-alias NAME` (repeatable) adds names that `GET /v1/models` lists; every server also accepts `default`; responses report the public id | One client configuration keeps working across artifacts and renamed ids. Added 2026-10-11 |
| Chat Completions `allowed_tools` entries (`d17f6a52`) | flat `{"type":"function","name":...}` only | also OpenAI's documented `{"type":"function","function":{"name":...}}` | The documented shape returned HTTP 400 `function name must be a string`; worth offering upstream |
| Routed-expert statistics and trace | no routed-expert observation | `--expert-stats` / `--expert-trace DIR`: the NVFP4-bank `sparse_moe` optionally publishes its top-10 selection, the `qwen4_exp` Program mirrors it per executed unit, and the server keeps live counts (`GET /admin/experts`) and writes `.nxtr` traces with request-log ids | Measures expert-activation skew on real traffic for an expert VRAM budget (keeping only part of the routed experts resident). Off by default; startup reserves the trace slices only when enabled. Op test checks the published ids against the FP64 oracle selection; Engine real cases check record shape, tags, positions and unchanged greedy output |
| Engine-discovered shared prefix on first sighting (`493b78c4`, `bb8c4068`) | saved whenever its coverage budget allows, ahead of the request's deeper points | takes free space only on its first sighting; the next request to reach it publishes it under the ordinary optional-capture budget (repeated demand still needs two committed requests) | As a fallback it shrank the same request's turn-closure budget to the tokens between them, so with a full context cache a plain multi-turn chat resumed only from the system prefix. Flash-Next v3 with a full cache (Host context 9.49 of 9.52 GB): turns 2 to 4 of a 5.2K-token-system chat reused 5,182 tokens each before and 5,189 / 5,226 / 5,263 after; with ~500-token answers turn 5 reused 5,182 of 7,292 before and 6,892 of 7,480 after. `493b78c4` gated on reused demand, which a capture sees only from the third request, so a full-cache shared system message (~5.7K tokens) asked different questions reused 0, 0, 0, then 5,738; `bb8c4068` gates on another request's crossing: 0, 0, then 5,740 from the third request. The cost against upstream, which reuses from the second request: with no free space, the second request over a new unmarked system message still prefills it cold; an explicit `prompt_cache_breakpoint` on that message reuses from the second request. Real case `conversation chain survives a full cache` also requires a third question over the conversation's system message to resume from it (1,850 of 1,871 tokens); Flash-Next reuse cases pass, and the 27B `prefix_real` and native-transaction tests passed with `493b78c4`. Proposed upstream |
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

**Source integrated October 4, 2026; migrated to upstream runtime October 8.**
The results below predate the runtime sync unless explicitly stated; see its qualification above.
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
the desktop: startup correctly refuses a 9.8 GB runtime reservation with 7.5 GB available.

Windows serving on this runtime (October 11, 2026, release build `27cd69b2`, LGPL FFmpeg) uses
context 131072, KV capacity 262144, concurrency 4, prefill chunk 8192, FP8 KV, Vision and a 6 GiB
desktop reserve; startup leaves 10.9 GB free without speculation and 8.7 GB with MTP K=3. An
HTTP API suite of about 120 checks across Chat Completions, Responses, Anthropic Messages, score,
rerank, System One, Vision, tools, structured output, logprobs, caching, concurrency, cancellation
and needles at 31K and 107K tokens passes on both, except the multi-turn reuse check described
under open questions and a greedy empty answer under `tool_choice:"none"` (model behaviour). Greedy
outputs are byte-identical with and without MTP; speeds are in
[performance](../performance.md#windows-serving-mtp-draft-count-2026-10-11). The artifact's official
template rejects developer and later system messages, so serving uses `--chat-template` with the
maintained `qwen3_8.jinja`.

Native Windows GPU tests at `27cd69b2`: 156 of 159 CTest entries pass without artifacts. The two
JSON Schema and tool-schema Python oracles time out in their `--probe` subprocess, as they do with
binaries built at `2f08a0b2`. `hyper_connection` failed once with a host `bad allocation` while
host RAM was nearly exhausted and passes on rerun. With the Flash-Next artifact, 17 of 21 Engine
cases pass in the full run. The other four failed in `cudaMallocHost`: the first reported out of
memory, the next three `cudaErrorAlreadyMapped`. In fresh processes `mtp long concurrent reuse`,
`mtp shared prefix promotion` and `mtp staged materialization ledger` pass alone, while
`long concurrent reuse` (no speculation) fails again with `cudaErrorAlreadyMapped`. That case passes
on the Linux G4 at this revision; its Windows cause is open. The Qwen3.8-27B real-artifact tests were
not run on Windows at this revision.

Three fatal engine failures found on October 6 are fixed
([causes and evidence](../research/flash-next-v3-port-2026-09-29.md#fatal-engine-failures-found-2026-10-06-fixed-2026-10-07)):
- `selected pressure target could not be sealed`: a burst of admissions planned while an earlier
  admission's StateImage Fork was unsettled. It also affected the 27B; the October 8 runtime replaces that planner;
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


- Before the October 8 runtime sync, `prefix_real`'s Host-restore check failed on the
  Qwen3.8-27B NVFP4 artifact identically on the
  pristine base and the fork line (the demoted turn closure is not the selected source): upstream
  behaviour on that revision. The complete suite passes after adopting upstream context
  replacement and preemption; the earlier failure is retained here as historical evidence.
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

## Historical qualification of the fork line

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
