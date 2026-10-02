# Decision head plan: learning from Cloudflare Clef

2 October 2026. Plan for Igor's goal: make `/v1/systemone` better than its current zero-shot
readout, win against [Cloudflare Clef](https://huggingface.co/Cloudflare/clef) where it matters,
and keep one resident model serving Chat and the decisions API together. Design follow-up to the
[decision study](qwen3.8-27b-decision-study.md) and the
[JevBench dossier](jevbench-qwen-intelligence-dossier.md). This is a plan, not authorization for
the engine work in phase 2; that is a product change Igor decides after phase 1.

## Verdict

The shipped Clef release is stock Qwen3.8-27B plus a 256 MB "joint schema head". Six backbone
tensors sampled by HTTP byte range across both attention types, an MLP projection, a norm and a
vision block are byte-identical to `Qwen/Qwen3.8-27B`; the only keys missing from Clef are the
`mtp.*` tensors. The blog's rank-256 LoRA is not in the release. So the head is a pure add-on to
the model already in VRAM: Chat requests never execute it, and decision requests reuse the same
weights, context cache and scheduler. Igor's constraint is satisfied by construction.

What the head buys, by design rather than by measurement:

- Options are scored by their text, not by a positional letter, so option order cannot move
  probabilities. Our route is positional; JevBench showed 11 argmax flips over 119 reordered
  public choice tasks.
- All questions of a record decide in one backbone pass. Our route runs one branch per question
  (300 questions took 6.9 s on an idle instance).
- Probabilities were trained with a Brier term. Ours are raw next-token mass with post-hoc
  temperature (hard ECE 0.133 at T=1).

What it cannot do: think. On TypeSafe's Decision Index Clef loses to Jev by 20 to 30 points on
GPQA, MMLU-Pro and BBH, while our decision study measured bounded reasoning fixing 5 to 6 of 20
hard public items. The route that wins where it matters is a head for fast decisions plus the
existing reasoning route for hard families, which Clef's non-autoregressive design cannot offer.

Unknown until phase 1: whether the shipped head beats our zero-shot route on JevBench. Clef has
never been run on it, and the README numbers may have been produced with a LoRA'd backbone the
release omits.

## The head, as shipped

Source: `joint_schema_model.py` and `joint_head_config.json` in the release.

- Input: `last_hidden_state` of the HF text model, which is the output of the final RMSNorm
  (`Qwen3_5TextModel.forward` applies `self.norm` before returning). This is the same quantity
  the Program already captures as one row for the reasoning router; the head needs every
  position of the record.
- Record prompt, tokenized segment by segment and concatenated: a fixed system prompt, `STATE:`,
  media placeholders before the state, the state, then `SCHEMA FIELDS:` with one block per
  question (`FIELD n`, `ID: <key>`, `TYPE:`, `INSTRUCTION:`, `ALLOWED OPTIONS:` with
  `OPTION n: {"option_id": ..., "description": ...}`, `END FIELD`), and the assistant opener with
  an empty think block ending in `JOINT SCHEMA DECISIONS:`. Question and option token spans are
  recorded while encoding. `max_length` 16,384; the schema is never truncated, only the state.
- Head (`hidden_size` 5120, `width` 1024, 16 heads, feed-forward 4096): LayerNorm on the hidden
  states; six bias-free projections to width 1024 (memory, question, option-question, global,
  option-context, option-lexical); a 3-entry type embedding; two evidence-routing layers
  (LayerNorm, torch `MultiheadAttention` from option queries to the projected sequence, GELU
  feed-forward); per-question softmax pooling of routed options into a field vector plus the
  global last-position vector; four `TransformerDecoderLayer`s (`norm_first`, GELU) over the
  field vectors with the sequence as memory; final LayerNorms; per option one logit =
  lexical cosine prior + sigmoid-gated (scaled cosine + residual MLP over
  [field, option, product, |difference|]).
- Option vectors mean-pool the option's hidden span and the mean of the output-embedding rows of
  its tokens (`lm_head.weight`). Question vectors mean-pool the instruction span. The global
  vector is the last position.
- Softmax per question. The reference `systemone_answer` reports confidence as the maximum
  probability; Jev and our contract use `(N·pmax−1)/(N−1)`. Score is the expected level index on
  both.

## Phase 1: does the shipped head win on our metric? (go/no-go)

Off-engine, on Colab G4 (BF16 backbone, transformers 5.10.2, the release code as published).
Local GPUs stay with production; no extra engine instance runs beside it.

Workload: the 231 JevBench public items through the TypeSafe adapter, paired with our native
route at criteria insertion order (the
[paired run](../../profiles/bench/jevbench-native-20260927/README.md) is the baseline: 189/231,
hard 76/111, Brier 0.2459, ECE 0.0674 at T=1). Also the 12 authored exact-answer diagnostics and
the 20 selected hard items from the decision study.

Metrics, declared before running: accuracy total and hard-only with paired fixed/broken counts
and a grouped bootstrap interval; multiclass Brier and ten-bin ECE; argmax agreement with our
route; invariance under a random option-order shuffle (the head sorts choice keys, so this
mostly checks the pipeline); per-family breakdown.

Acceptance: the head must beat 76/111 hard with an interval excluding zero, or improve Brier by
a margin that survives the T=1.5 rescaling of our route. Otherwise adoption is a latency and
ordering win only, and phase 4's hybrid becomes the main quality lever.

Phase 1b, quantization sensitivity, one day, before any engine work: fake-quantize the HF linears
on Colab with the converter's NVFP4 scheme for the layers the production artifact quantizes
(pin the set from the base artifact's conversion report), run the head on those hidden states,
and compare per-option logits against BF16: argmax agreement, max |Δp|, KL. This decides whether
phase 2 is "port" or "port, then fine-tune on our numerics" (phase 4).

### Phase 1 and 1b results (2 October 2026)

Run on a Colab G4 VM; evidence and method in
[the run directory](../../profiles/bench/jevbench-clef-20261002/README.md).

**Phase 1 is a go.** The shipped head answered 201/231 public items against our native route's
189/231 (fixed 21, broken 9, scenario bootstrap [+0.4, +10.1] pp). Hard tier 82/111 against
76/111 (fixed 14, broken 8, interval [-2.7, +13.5] pp includes zero). Brier 0.166 against 0.246,
ECE 0.030 against 0.067 at T=1; our route's predeclared T=1.5 rescale reaches only 0.240. Gains
concentrate in the probability family (9/10 against 4/10) and noul questions (65/74 against
57/74); ambiguous lost (5/7 against 7/7); temporal_numeric stays at 5/15. Reversing option order
changed no probability. T=1.5 hurts the head: it is already calibrated.

**Phase 1b changes phase 2's premise.** With the production artifact's weight words written into
the BF16 backbone (NVFP4 MLP 0-55 from the unsloth source, FP8 rows elsewhere, FP8 embedding),
the head collapses to 119/231 (Brier 0.62, 52% argmax agreement with BF16), reproduced twice,
while the backbone's own next-token NLL is unchanged. Each family alone is tolerated (NVFP4 MLP
only 200/231, FP8 projections only 206/231), and an all-FP8 backbone built with the converter's
row codec keeps the head at 206/231 with hard 86/111 and oracle 18/24. The activation proxy adds
nothing beyond the weight effect. The collapse is a loss of discrimination, not confident
errors: the head's median logit spread shrinks 2.55 (BF16), 1.64 (NVFP4 only), 1.36 (all-FP8,
simulated), 0.23 (production set), and answers become near-uniform. All-FP8 keeps the argmax but
its probabilities are already flatter (Brier 0.197 against 0.166).

Consequences for the plan (Igor's decision, 2 October): **no all-FP8 artifact.** NInfer stays
native Blackwell NVFP4, and the extra VRAM is not worth it. The all-FP8 row was in any case a
PyTorch simulation, never an engine run. So:

- The shipped head is not deployable on the production artifact as it is. The path is to
  **fine-tune the head on the production NVFP4 hidden states**: start from Clef's released head
  weights, feed it the final-normalized hidden states the engine actually produces for the Clef
  record encoding, and train against Clef-BF16's own per-option distributions as soft targets
  (Apache-2.0 permits this; Jev's terms would not) with the label-smoothed cross-entropy plus
  Brier objective the blog describes. The collapse is a loss of logit spread, not of ranking, so
  the first experiment is the cheapest one: fine-tune only the head's scales and norms, then the
  routing layers, then everything.
- Prerequisite, and now the first engine deliverable: a **hidden-state export route** that
  returns the final-normalized hidden sequence of a prepared prompt from the production artifact
  (an extension of the existing one-row reasoning-feature capture). It runs on Colab G4 through
  the Linux build with the NVFP4 artifact, so no on-site GPU is involved and the head trains on
  the exact production numerics, FP8 KV included.
- Acceptance for the fine-tuned head, before any serving work: on the engine's hidden states it
  must recover at least the BF16 reference on the 231 public items and its Brier (201/231,
  0.166), measured against our native route as in phase 1; held-out families and the authored
  diagnostics must not regress; Chat is untouched by construction.
- Calibration is the head's largest measured advantage over our route, and it is the quantity the
  fine-tune has to restore, since it is exactly what the quantization noise erased first.

### Phase 2 progress (2 October 2026)

The hidden-state export route exists on branch `research/decision-head` (uncommitted):
`CausalScoreReadout::capture_hidden_rows` on the scoring route and the `ninfer-hidden-export` app
([perplexity guide](../perplexity.md#hidden-row-export)). The real scoring test checks the rows
exactly and passed on Linux against the published production-recipe artifact. The shipped head
over the engine's own rows for the 255 records scores 121/231 public (hard 48/111, Brier 0.627,
oracle 10/24), confirming the phase 1b simulation (119/231) and fixing the fine-tune's starting
point. Details in [the run directory](../../profiles/bench/jevbench-clef-20261002/README.md).

The export also exposed a property of the quantized backbone that the fine-tune must respect:
hidden rows depend on how many tokens the prefill chunk holds. A record's first half run alone
differs from the same positions inside the full record from position 0 on (per-position cosine
minima 0.18 to 0.42, next-token log probability shifts up to 23 nats), while the engine is strictly
causal (an altered suffix at the same length leaves earlier rows bit-identical). PyTorch
reproduces it only with fake-quantized activations (minima 0.07 to 0.32; BF16 stays at 0.99 on
short records and 0.64 at one position of a 4,033-token record), so it is inherent to 4-bit and
8-bit activation quantization through 64 layers, not an engine defect. The scoring Engine now
honors the caller's prefill chunk so exports can match serving (2,048); the head scores the same
at 1,024 and 2,048 (121 and 120 of 231). Consequences: train the head on rows exported at several
chunk sizes and truncations so it tolerates the drift, evaluate it at serving's chunking, and
prefer readouts that are robust to it (the last position and next-token distributions drift less
than arbitrary span means) where the architecture allows.

### First fine-tune (2 October 2026)

`train_head.py` fine-tuned the released head on engine rows (production-recipe artifact, chunks
2,048 and 1,024 as augmentation) with Clef-BF16 soft targets plus gold labels over 5,468 records
(router synthetic corpus, seeded oracle cases, BoolQ, ANLI, AG News, Yelp; dev split of 539).
Public set at the serving chunk: 118 (shipped head on engine rows) to 154 after the norms-and-scales
stage alone, 172 after two full epochs, 176 after eight; Brier 0.63 to 0.35. Still below the native
route (189, Brier 0.246; paired interval against native excludes zero) and the BF16 head (201,
0.166), so the acceptance bar is not met. Chunk robustness held (1,024 and 2,048 within one item).
Dev KL saturates after a few epochs; the weak families (judge_hard, multi_hop, probability,
temporal_numeric) are absent from the corpus. Next levers, in order: in-distribution training data
for those families (a synthetic generator with exact oracles exists for four families; the others
need authoring), a larger head or a reset of its scoring layers, and training the head on BF16
and engine rows jointly so it learns the drift rather than only its endpoint. Details in
[the run directory](../../profiles/bench/jevbench-clef-20261002/README.md).

### Native-route policy study (2 October 2026): the route that beats Clef

With the head fine-tune short of the bar, the same corpus and Colab server were used to improve
the native route directly (`tools/bench/jevbench/decision_policy/`, evidence in
[the run directory](../../profiles/bench/jevbench-clef-20261002/README.md)). Public set, paired:

- **Letter-rotation averaging** (a second readout with rotated option letters, same state prefix):
  198/231, Brier 0.208, ECE 0.037; against native fixed 13, broken 4, [+0.9, +7.5] pp. No training,
  one extra branch per question.
- **Calibration fitted on the corpus does not transfer** to the public items (Brier worsens); the
  raw readout is already calibrated there. Calibrators belong to the deployment's workload.
- **Confidence-gated reasoning** (escalate the least confident or highest-entropy quarter of
  items to a 1,024-token thinking pass, thresholds from dev quantiles at a declared budget):
  207 to 208/231, hard 88/111, Brier 0.166 to 0.177, ECE 0.037; against native fixed 24, broken 5,
  [+3.8, +13.2] pp; against the Clef BF16 head fixed 15, broken 8, [-1.3, +7.1] pp. Estimated
  single-stream mean latency 0.5 to 0.7 s per decision at that budget.

This reaches Clef's accuracy and Brier from the native NVFP4 route, with the latency budget as an
explicit knob and reasoning as the capability Clef lacks. It supersedes the head as the product
path: the proposed serving change is (1) an averaged readout option for System One and (2) an
escalation policy with a per-request or per-server budget (`reasoning budget`, `escalate below`
signal and threshold), both measured here with the exact request shapes. Both are serving
changes for Igor's go; the head stays research.

## Phase 2: engine integration (product change)

Ownership, following [engine architecture](../maintainer/engine-architecture.md):

- Artifact and converter: a new optional component `decision_head` beside `vision`, `mtp` and
  `dflash2`, with the head's BF16 tensors and `joint_head_config.json` values; a converter recipe
  option that splices `joint_head.safetensors` into an existing artifact, like the DFlash2 splice.
  Artifacts without the component keep the current readout route (Flash-Next has no head).
- Model: binds the component as native Parameters and the output-embedding rows the lexical
  vectors need.
- Program: capture the final-normalized hidden states of the whole record. Two options to decide
  at implementation: a per-lane BF16 buffer of `[record_tokens, 5120]` (168 MB per lane at the
  16,384 cap), or per-chunk accumulation of the six width-1024 projections and the per-span sums,
  which is exact because the projections are linear and commute with span means. The head then
  runs as Ops on device after prefill. The existing frontier-row capture is the tap point.
- Ops under `src/ops`: LayerNorm with bias, biased Linear, torch-semantics multi-head attention
  over short query sets against a long memory, erf-GELU feed-forward, span mean-pool, embedding
  gather-mean, and the cosine/prior/residual scorer. Each gets the naive FP32 oracle required by
  [Op development](../maintainer/op-development.md); the published PyTorch module run in FP32 is
  the independent oracle for the composed head.
- Engine: a decision request kind that takes a prepared prompt plus question and option spans and
  returns per-option logits. Read-only in the context cache for v1 (no publication, no frontier
  split). Validation of spans and the 16,384 cap happens before admission, like prompt-position
  logprobs today.
- Serving: `/v1/systemone` keeps its contract. HTTP owns the record encoding and spans, and
  selects the head route when the resident artifact carries the component; a request extension
  selects the token readout explicitly for comparisons. Confidence stays Jev's formula.

Encoding fidelity the head was trained on, each a divergence from the current route that the
head route must follow exactly:

- Choice keys are sorted alphabetically before encoding; the response maps back to request keys.
- Objects and arrays render as compact JSON with sorted keys, not 2-space indented.
- The schema prints `ID: <question key>`, so question keys reach the model.
- Empty instructions become the question key; a Noul gets default true/false criteria text.
- Segments are tokenized separately and concatenated; tokenizing the joined string moves span
  boundaries. Verify NInfer's tokenizer matches HF on these segments.
- Media placeholders precede the state.

Qualification before deployment: at least 500 records (JevBench public, the vision probe set,
authored multi-question records) compared against the BF16 reference: argmax agreement, max |Δp|,
KL, and the phase 1 metrics; Chat and Anthropic route logits unchanged before and after the
Program change on the existing test binaries; VRAM delta on the production profile; latency per
the bench-harness protocol for 1, 8, 50 and 300 questions per record against the current route.

## Phase 3: serving advantages Clef does not have

- One pass per record replaces the serial per-question fan-out.
- The state sits between a fixed system prefix and the schema, so repeated-state workloads can
  publish the state prefix once the projected memory is stored with the checkpoint (v2; v1 is
  read-only).
- Decision prefill shares the one compact batch per round with Chat prefill under the existing
  concurrency and FIFO contracts; no new scheduling.
- Images take the existing vision route; the head reads vision tokens as memory positions.

## Phase 4: winning where it matters

- Hybrid: head for the fast path, bounded reasoning for hard families, selected by the
  [learned router](learned-reasoning-router.md) or by head confidence and task type with
  thresholds frozen before evaluation. Report error at fixed coverage and end-to-end compute.
- Beyond the recovery fine-tune above: continue training the head on the public classification
  sets from the Decision Index and on reasoning-route outputs for hard families. Sensitive
  workloads (Tribuno, YDUQS) collect and train on site only.
- A head for Flash-Next, trained from scratch with the same recipe, is last and optional; its
  zero-shot route already leads the 27B on JevBench hard.

## Evaluation rules

| Item | Rule |
|---|---|
| Baseline | native `/v1/systemone` at criteria insertion order, T=1 and the predeclared T=1.5 |
| Comparison systems | shipped Clef reference on Colab; NInfer head route on the production artifact |
| Workloads | JevBench public (paired), the authored diagnostics, the vision probe set, multi-question records |
| Aggregation | paired fixed/broken counts, grouped bootstrap by scenario, worst family reported |
| Latency | the 8-rule bench-harness protocol on the production machine, no other engine running |
| Official score | only Harold's sealed run; never present a public-subset number as official |

## Evidence recorded today

- Backbone identity: `model.language_model.layers.3.self_attn.{q_proj,o_proj}.weight`,
  `layers.0.linear_attn.out_proj.weight`, `layers.0.mlp.gate_proj.weight`,
  `layers.0.input_layernorm.weight` and `model.visual.blocks.0.attn.proj.weight` fetched by
  safetensors byte range from both repositories and compared byte-for-byte: all equal. Key sets
  differ only by Qwen's `mtp.*` tensors. Clef ships 12 shards (54.7 GB, 1,948 keys) against
  Qwen's 18 (55.6 GB, 1,456 keys); the key count differs because transformers 5.10 resaved
  fused projections split.
- Head size and config from `joint_head.safetensors` (256,125,024 bytes) and
  `joint_head_config.json`.
- Published latency (Clef median 209.3 ms, p95 238.6 ms on an H200, records with several
  questions) is not comparable with our single-question p50 of 51 to 61 ms on the shared
  production machine.
