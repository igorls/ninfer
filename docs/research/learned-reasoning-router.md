# Learned reasoning router for Qwen3.8-27B NVFP4

Design and experiments, 23 September 2026. Engine changes are explicitly in scope. This extends
the [decision study](qwen3.8-27b-decision-study.md). Feature collection and progressive head
training are implemented; the initial heads are research artifacts and are not qualified for serving.

## Proposed model

Train a small predictor of the expected benefit of additional reasoning, using frozen
Qwen3.8-27B NVFP4 features. Compare:

1. A classifier/regressor on candidate probabilities, entropy, margin, outside-candidate mass,
   option count, and task metadata available at inference.
2. A linear head on the final hidden vector at a precisely defined shared-prompt position.
3. A small nonlinear head on that vector, optionally combined with the decision-probe features.

The 27B artifact has a [5,120-wide hidden representation](../maintainer/qwen3.8-27b-artifact.md).
An illustrative `5120 -> 128 -> 3` network has 655,875 parameters including biases: about 1.31 MB
with 16-bit parameter storage. Three outputs could estimate the quality of direct, 1,024-cap,
and 2,048-cap actions. They are independent quality estimates, not a softmax over mutually
exclusive correctness events: multiple actions can be correct on the same item. Cost prediction
can use additional outputs or a separately fitted predictor. This size is an illustration,
not a measured accuracy or latency result.

The hidden-only head can route before an answer probe, potentially avoiding the probe's branch
and rollback. The feature-only model pays for that probe. Compare total cost and held-out benefit
before selecting either. A separately tokenizing language model is another possible router, but
its extra prefill and residency would need to earn their cost against these alternatives.

## Training target

For each independently labeled scenario, execute all supported actions under the exact target
artifact, template, precision and controlled cache policy. Record answer correctness, validity,
tokens and latency. Training must include examples where reasoning fixes, preserves, and breaks
the direct answer, and where every tested action fails. A harder question does not necessarily
benefit from more generation.

Estimate per-action correctness and compute cost, then select the best expected quality under a
declared latency/token budget, or use `expected_quality - lambda * expected_cost`. Actual future
reasoning length is a training label; it cannot be an inference input. Keeping cost separate lets
hardware-specific latency estimates change without pretending PRO 6000 timings apply to a 5090.

Our 32-case pilot is insufficient to train or qualify this predictor. Collect diverse training
scenarios and reserve separate calibration/test scenarios, grouping paraphrases and related
templates. Keep expected answers and rationales outside inference features. For stochastic
decoding, estimate outcomes across samples; greedy training still needs controlled execution and
repeatability. Bind the router to its qualified artifact/template/KV profile and re-evaluate when
those change.

The quality goal is to retain the useful accuracy of a strong fixed reasoning policy while
avoiding unnecessary reasoning. Compare always-direct, always-1,024, always-2,048, simple
confidence gates, and random routing at comparable compute. Measure false-direct decisions
(cases that reasoning would have fixed), paired accuracy, per-family regressions, validity, and
complete request latency including feature extraction and routing. Assess unseen families and
scenario sources before making a broad claim.

## Engine design

One public request would own `prefill -> route -> direct/reasoning -> finish`:

- **Frontend:** define one adaptive prompt with identical instructions up to the routing point,
  plus canonical direct and thinking continuations. Current `enable_thinking` and reasoning effort
  can change the leading system instructions, so toggling them after prefill cannot manufacture a
  shared prefix. Train on the exact new rendering and compare it with fixed-mode baselines.
- **Program:** expose the selected hidden representation internally, execute the qualified head,
  and retain the full state needed for the selected path. Existing
  [continuation hidden storage](../../src/targets/qwen3_6/impl/state/state_image.cpp) is useful
  infrastructure, not an existing router hook. If a private answer probe is used, preserve KV,
  recurrent GDN state, continuation hidden state, positions and applicable speculative state.
- **Engine:** own the route decision and request-phase transition before any tentative answer is
  published. Admit sufficient resources for the permitted reasoning path, preserve cancellation
  and total-budget semantics, and schedule the resulting work through the existing request loop.
- **Ops and artifact ownership:** put the small classifier's mathematical implementation under
  `src/ops`; bind its qualified coefficients and feature specification through the target's
  artifact/conversion contracts. Keep model-state pointers and policy decisions out of HTTP code.
- **Product interface:** expose an adaptive reasoning policy and caller budget, with actual mode,
  compute and decision metadata in observability. Final field names and schemas belong to the
  implementation; this proposal introduces no new HTTP contract.

The first version routes once. A later stop/continue predictor could re-evaluate at defined
reasoning checkpoints, but it needs training examples from partial reasoning states and labels
for the value of continuing. A prefill-only router cannot be assumed to generalize to that task.

The cache-sensitive answer changes found in the pilot are part of this engineering work. Stable
feature extraction and qualified state continuation are prerequisites for trustworthy routing
labels. The engine work should first provide a reproducible routing boundary and paired feature/
outcome collection, then integrate the learned policy after held-out evaluation. Engine changes
are expected; they are not a reason to restrict the design to HTTP request orchestration.

## Implemented training path

The initial training path now exists in the Engine and
[`reasoning_router.py`](../../tools/bench/jevbench/reasoning_router.py). The
[`ninfer-reasoning-collect`](../../apps/reasoning-collect/main.cpp) executable runs the frozen
Qwen3.8-27B NVFP4 artifact through the public Engine, collecting a 5,120-value normalized hidden
row immediately before `<think>` and the actual outcomes of direct / 1,024 / 2,048-token modes.
Each mode starts fresh with FP8 KV, a 1,024-token prefill chunk and speculation disabled. The
prefill split is explicit, and all three feature vectors must match exactly. This is a controlled
training profile, not qualification of cached or speculative serving.

The trainer fits three independent correctness logits and three nonnegative output-token cost
estimates. Its first baseline is a linear head with 30,726 parameters; a 128-unit MLP is also
available. The backbone stays frozen. Targets come from Qwen's own paired attempts checked
against exact-oracle labels. Invalid or budget-truncated answers are failures; interrupted or
context-exhausted collection is retried. Neither the gold answer nor completed reasoning enters
the feature vector.

Per-example RMS normalization stays fixed across rounds. Progressive training loads the previous
head and replays all earlier training examples. Artifact SHA-256, feature profile, split policy
and prior observation digests must agree. Paraphrases supplied from other sources must share a
group. Exact prompt deduplication and a held-out template family complement the grouped instance
split. Checkpoint selection uses validation routing utility, with prediction loss as a tie-break;
test sets do not select epochs. All exported heads are marked `qualified_for_serving: false`.

See the [collection and Colab commands](../../tools/bench/jevbench/README.md#progressive-reasoning-router-training).

### Initial progressive training run — 2026-09-23

Collection used the local RTX PRO 6000 Blackwell and the registered Qwen3.8-27B NVFP4 artifact
with its FP8 KV profile. Training used a Google Colab Tesla T4 and PyTorch 2.11.0+cu130. The
backbone stayed in NInfer locally; only the generated examples, features and head training code
went to Colab. The session was terminated after downloading all four checkpoints and reports.

The authored generator produced 135 unique prompts after deduplication. The initial experiment
collected the first 90 in deterministic shuffled order: 270 attempts and 45,628 output tokens.
There were 22 accounting, 18 inclusive-date, 19 temporal-lookup, 26 resource, three precedence
and two ordinal examples. No attempt was interrupted or output-truncated. The identical
pre-think feature check passed for every paired example. Across all 90 cases, direct generation
was correct on 65; both reasoning budgets were correct on 90. These are diagnostic labels,
not a benchmark score or an independent estimate of deployment accuracy.

Both heads first trained on a 40-example snapshot, then warm-started on the accumulated 90.
The latter split contains 41 training groups, 21 validation groups, nine instance-test groups,
and 19 temporal-lookup groups held out as an entire family. No prior observation was replaced.
Utility was fixed at correctness minus 0.02 per 1,024 generated tokens.

| Head | Parameters | Round 1 best epoch | Round 2 best epoch | Round 2 validation utility | Validation mean tokens |
|---|---:|---:|---:|---:|---:|
| Linear | 30,726 | 1 | 0 | 0.995018 | 255.1 |
| MLP, 128 hidden units | 656,262 | 24 | 3 | 0.994730 | 269.8 |

The linear head wins the predefined validation objective. Its second round correctly retained
the original weights; the additional training did not improve validation utility. The MLP's
second-round weights changed but did not beat the linear baseline. Both validation accuracies
were 21/21. Test results for the selected linear head:

| Policy | Instance test: correct / 9 | Mean output tokens | Family test: correct / 19 | Mean output tokens |
|---|---:|---:|---:|---:|
| Direct | 4 | 2.0 | 19 | 2.0 |
| Always reason, either budget | 9 | 253.8 | 19 | 219.9 |
| Selected linear router | 9 | 253.8 | 19 | 219.9 |

This establishes a working progressive training loop, **not an effective compute-saving router**.
On both test sets the router spent as much as always reasoning; the held-out family particularly
exposes unnecessary thinking. No reasoning attempt exceeded 1,024 tokens, no case distinguished
the two reasoning budgets, and no reasoning-harms example occurred. The next corpus needs varied
templates and independently verifiable cases spanning those missing outcomes. Increasing head
size did not solve that data gap. Do not enable adaptive serving from these results.

Raw outcomes, SHA-256 artifact identity, four checkpoints and reports are retained locally under
`profiles/bench/reasoning-router-20260923/`. `round2-data.jsonl` is the accumulated, digest-bound
dataset to resume; `round2/router.json` is the selected research checkpoint. The initial collector
started before the digest field was added: snapshots were bound to the exact source file after
hashing it during the resident run and checking its unchanged size/mtime. Subsequent collection
uses the built-in hash-and-resume wrapper. Nothing was installed into the running server.

Verification: focused C++ build; frontend suite; real-GPU feature tests covering a multi-chunk
prompt, interleaved request state, direct/reasoning equality, exact BF16 expansion, opt-out and
raw-prompt rejection; eight router data/replay tests; nine existing decision-study tests; and a
successful no-work resume through the final digest-validating collector. The cache-sensitive
production behavior from the earlier study remains unresolved; this collection avoids that
source of variation rather than claiming it fixed.

#### Standards review

Two material concerns were raised: path-only artifact provenance and incomplete generations
being accepted on resume. The collector now binds SHA-256 and retries interrupted/context-limited
examples. Exhausting the declared output budget intentionally remains a failed action label.
No documented standards violation or feature ownership/lifetime mismatch was found.

#### Spec review

Three concerns were raised: row counts standing in for independent groups, path-only replay
compatibility, and prediction-loss selection differing from the routing objective. These are
addressed by distinct-group gates, immutable observation/artifact checks, and validation-utility
selection. Standards: two concerns addressed; Spec: three concerns addressed. The principal
remaining limitation is the training corpus, not an unresolved review finding.

### Flash-Next corpus generation on Colab G4 — 2026-09-23

The current NInfer source snapshot built and served Flash-Next successfully in Google Colab.
This uses the C++ Engine and OpenAI serving route; Python only orchestrates requests, validates
scenario data and computes gold. The teacher supplies inputs for the Qwen3.8-27B NVFP4 study.
Its own output quality or internal features are not substituted for Qwen's paired outcome labels.

The runtime is Ubuntu 24.04, NVIDIA RTX PRO 6000 Blackwell **Server Edition** with 97,887 MiB
VRAM and 176 GiB host RAM, driver 580.82.07, nvcc 13.3.73, GCC 13.3 and CMake 3.31.10.
The source is the working snapshot of `research/qwen4-flash-next` based on `147370d7`, including
the current engine/serving changes. It is not a clean-release qualification. The text-only
Release build used `NINFER_BUILD_MEDIA=OFF`, `BUILD_TESTING=OFF` and target `ninfer-serve`.

The published Flash-Next artifact was downloaded at revision
`5f0ee7e24279cbadf8d4a90c93c2ff5ea6b00688`, and its complete 113,298,397,952-byte file matched
SHA-256 `3d383e51963aafd4318dfd04c8dc63ee7df11768de19d9ab58dbba44460d1d02` before loading.
The serving profile uses 8,192 context, 32,768 KV capacity, four active requests, 2,048-token
prefill chunks, FP8 KV, BF16 recurrent state, ordinary decode and prefix reuse disabled.
The known-answer HTTP smoke returned `42` for `17 + 25`; all four concurrent structured smoke
requests passed local schema and semantic checks on their first attempt. This establishes
working native execution with the installed driver/toolkit combination. It does not qualify
MTP, vision, all numerical operators or performance relative to another environment.

`synthetic_corpus.py` constrains the teacher to four data schemas: ordered integer operations,
first-match policies, blocked directed graphs and 0/1 allocation. Code renders the questions
and computes gold; free-text teacher titles cannot alter either. Sibling mutations share one
group, and exact logical duplicates are removed independently of formatting or option order.
Four size strata per family deliberately vary workload size, without asserting measured
difficulty. The generated data is a bounded diagnostic corpus, not a JevBench score or broad
coverage of natural-language reasoning. Paired Qwen collection must determine whether it adds
the missing budget distinctions and reasoning-harms cases.

The completed run produced **2,048 unique scenarios in 64 blueprint groups**: 512 per family
and 128 per family/size stratum. Sixty-five accepted teacher slots included one exact duplicate;
the extra slot restored 64 unique blueprints after deduplication. Seventy teacher requests
generated 49,887 output tokens in total. Four attempts were rejected for duplicate/self graph
edges and one for exhausting 6,144 tokens in a whitespace loop; retries succeeded, and no failed
slot remained. The 60-slot expansion after the four-case smoke took 180.4 seconds including
the worker's completion polling; the final replacement slot took 1.1 seconds. These are corpus
job timings, not a controlled benchmark. GPU samples held at 73,795 MiB (72.1 GiB), and engine
startup took 141.2 seconds, mostly weight materialization.

The exported archive passed SHA-256/ZIP verification, all 2,048 tasks passed the decision parser
and rendered-state gold check, and all 2,048 converted to Qwen collector inputs without further
deduplication. For a separate new study holding out `synthetic_allocation`, the fixed split has
32 train, nine validation, seven instance-test and 16 family-test groups. This illustrative split
does not change the existing progressive study's holdouts. **No Qwen outcomes have been collected
for this new corpus yet, and no router gain is claimed.** The Colab G4 session was terminated
after downloading the results.

The ten focused tests include independent all-pairs and exhaustive-subset oracles, floor/clamp
and rule-order cases, title exclusion, grouping/deduplication, manifestless resume rejection,
and wrong-model/timeout retries. The standards review raised one resume finding and one teacher
identity concern; both are addressed. The spec review raised teacher identity, resume count
semantics and size-versus-difficulty interpretation; all three are addressed. Both reviewers
confirmed the fixes. The HTTP API checks the model name; full-file verification in the launcher
binds this run to the actual artifact, since a response model name cannot attest to its weights.

Run evidence is retained locally under `profiles/bench/colab-corpus-20260923/`, including the
source snapshot, raw teacher attempts, executable-gold tasks and environment/launch records.
See the [generation commands](../../tools/bench/jevbench/README.md#flash-next-synthetic-corpus-on-colab-g4).

## Research evidence

[Think When Needed](https://arxiv.org/html/2601.18146v2) is the closest precedent: it trains a
small gradient-boosted router on the advantage of thinking, incorporating ranking features,
hidden-state pooling and diagnostic probes. Its results concern ranking/recommendation, so
transfer to NInfer decisions remains unproven. Its masked probe construction cannot be copied
into a recurrent GDN model without isolating recurrent state as well as attention.

[Thinkless](https://arxiv.org/abs/2505.13379) and
[AdaptThink](https://arxiv.org/abs/2505.13417) train the reasoning model itself to select modes.
They support the feasibility of learning compute allocation, but are different interventions
from training a small head on a frozen NVFP4 model. None of these papers establishes Qwen3.8-27B
NVFP4 performance. The proposed architecture and training study require their own measurements.
