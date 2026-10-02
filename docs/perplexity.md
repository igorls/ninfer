# Perplexity evaluation

`ninfer-perplexity` measures the causal perplexity produced by a v3 `.ninfer` artifact.
It uses the artifact's tokenizer, Text model, selected Main KV representation, final normalization,
and main output head. It is an offline evaluator, not a serving endpoint or a logits-export API.
Only Text weights and resources are loaded; Vision and speculative components are not required.

## Run the fixed corpus

The repository includes `ninfer-ppl-1m-v1`, a fixed set of 16 independent UTF-8 streams covering
English reference text, English long-form text, Chinese reference text, and NInfer C++/CUDA code.
`full` selects all streams; `--quick` selects one stream from each domain.

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b_nvfp4.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json \
  --quick \
  --kv-dtype fp8
```

The default evaluation uses a 4,096-token context and a 2,048-token stride. Use `--context` and
`--stride` to change that protocol, or score one UTF-8 file with `--text FILE`. The available Main
KV representations are `bf16`, `int8`, `fp8`, `nvfp4`, and `k8v4`.

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b.ninfer \
  --text notes.txt \
  --context 16384 --stride 8192 \
  --kv-dtype int8
```

Run `./build/apps/ninfer-perplexity --help` for the complete command surface. The evaluator loads
the model once, reads and tokenizes every selected stream before scoring, and writes readable
startup, corpus, scoring, and per-stream summaries to stderr. Interactive weight loading and
scoring use one transient progress line; redirected scoring emits persistent progress every ten
seconds. `--log-level debug` exposes internal startup and stream-begin detail. The final
domain/overall table remains product output on stdout; the independent full-precision machine
report is `report.json` under `profiles/perplexity/` unless `--output` supplies an empty directory.

For KV-format comparisons, the recommended long-context profile is the full corpus with
`--context 65536 --stride 32768` and without `--quick`.

## Metric

For a stream `x[0..N)`, every token after `x[0]` is scored exactly once. A window `[b,e)` with target
suffix `[s,e)` contributes:

```text
log p(x[i] | x[b], ..., x[i-1])  for i in [s,e)
```

Each window starts from empty State and Main KV, so history before `b` is deliberately excluded.
The reported metric is therefore fixed-window, truncated-context causal perplexity:

```text
mean_nll = -sum(logprob) / scored_tokens
perplexity = exp(mean_nll)
```

The first window scores `[1,min(context,N))`. Each later window advances by `stride` targets while
retaining up to `context-stride` preceding tokens as local context. Streams never share history.

## Comparing runs

For a numerical comparison, keep the corpus, context, stride, and execution settings fixed except
the variable being measured. Compare KV formats with the same artifact and weight formats with the
same KV format.

The corpus name is a workload scale, not an exact token count. Exact input and scored-token counts
are runtime results from the current artifact tokenizer and are recorded in each report. Reports
contain unrounded NLL/PPL values for every window, stream, domain, and the token-weighted overall
aggregate.

The schema-v2 report identifies the artifact's architecture, public name, actual weight formats
and prefill signature alongside the workload and numerical results.

## Distribution agreement with a reference model

`--reference FILE` compares an artifact's next-token distributions with a reference model's on
exactly the reference file's token ids, so tokenization and chat rendering cannot differ between
the two sides. For every scored position the reference stores its 32 most likely tokens with
their log probabilities and the log probability of the actual next token; the evaluator reads the
artifact's log probabilities of those same tokens, its own most likely token, and the actual
token's log probability through `Engine::score_tokens` with a `CausalScoreReadout`.

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b_nvfp4.ninfer \
  --reference kld-400k-v1.reference --context 4096 --kv-dtype fp8
```

The report gives, per sequence, domain and overall:

- `kl_mean` and the `kl_p50`/`kl_p90`/`kl_p99`/`kl_p999`/`kl_max` quantiles of
  KL(reference || artifact), computed over the reference top-32 tokens plus one bucket holding
  each side's remaining mass. Merging the tail makes it a lower bound on the full-vocabulary
  divergence; it is exact when the 32 tokens carry all of the reference's mass.
- `top1_agreement`: the fraction of positions where the artifact's most likely token is the
  reference's.
- `evaluated_mean_nll`, `reference_mean_nll` and their difference. NLL deltas are not a quality
  score on their own: a noisier model can predict text better at positions where the reference is
  confidently wrong.

`--dump-positions FILE` also writes, per scored position in reference order, the float32
divergence, the float32 target log probability and the int32 argmax token of the artifact.

`kld-400k-v1` is built by [`eval/kld/build_reference.py`](../eval/kld/build_reference.py) from
pinned public sources: the first 40,960 tokens of the `00` streams of `ninfer-ppl-1m-v1` (English
reference and long-form text, Chinese, NInfer C++/CUDA), 32,768 tokens of CPython 3.13 standard
library modules, 24,576 tokens of MATH-500 problems and solutions, 8,192 tokens of Belebele
passages in each of Portuguese, Spanish, French, German, Russian, Japanese, Arabic and Hindi, 20
s1K-1.1 reasoning conversations (thinking on) and 24 UltraChat conversations (thinking off),
rendered with the Qwen3.8 chat template. Every sequence is an independent window of at most 4,096
tokens; 114 sequences hold 400,006 tokens. The reference is the official BF16 Qwen3.8-27B
(revision `1d4bf0f2`) in Hugging Face Transformers, with log-softmax taken over FP32 logits of the
BF16 final hidden state. The builder records source revisions and SHA-256 digests in the corpus
manifest; the reference file itself is a local, regenerable artifact (about 106 MB).

On the UltraChat sequences the BF16 model assigns more than half its mass to `<|im_end|>` at about a
third of the positions inside answers; NInfer artifacts reproduce this, so their NLL there is often
lower than the reference's.

## Hidden row export

`ninfer-hidden-export` uses the same offline CausalScoring Engine to write the final-normalized
hidden state of every predictor position, the row the output head reads, for a list of token
sequences. It exists for training readouts over the artifact's actual numerics, such as a decision
head that consumes the hidden states of a quantized backbone.

```bash
./build/apps/ninfer-hidden-export models/qwen3_8_27b_nvfp4.ninfer \
  --input records.jsonl --output rows.safetensors \
  --context 16384 --prefill-chunk 2048 --kv-dtype fp8
```

`records.jsonl` holds one `{"id": "...", "tokens": [...]}` object per line, with 2 to `--context`
token ids per record. Rows cover predictor positions `0 .. tokens-2`: the last token of a sequence
is never a predictor, so a caller that needs every real position appends one token (for example
the pad token) and ignores nothing else. Each record is scored from fresh state with no context
cache. A row never depends on later tokens, but on the quantized artifacts it does depend on how
many tokens its prefill chunk holds: the same prefix exported at two lengths yields rows that can
differ substantially at some positions, because activation quantization amplifies the small
numerical differences of the chunk's GEMM shapes. The startup numerics are written into the file's
metadata: `prefill_chunk` and `kv_dtype` must match the serving configuration whose hidden states
the rows are meant to represent, and a consumer that must tolerate serving's variable chunk
occupancy should export at several chunk sizes and truncations.

The output is a standard safetensors file with, per record, `hidden/<id>` (BF16, `[tokens-1,
hidden_size]`) and `logprobs/<id>` (F32, `[tokens-1]`, the log probability of `tokens[i+1]` at
row `i`, the same value `score_tokens` returns). `safetensors.safe_open` reads it directly.

The underlying readout is `CausalScoreReadout::capture_hidden_rows` on `Engine::score_tokens`; it
leaves the scores unchanged, and `ninfer_qwen3_5_score_real_test` checks that the rows are
deterministic and positionally indexed.
