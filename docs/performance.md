# Single-GPU serving performance

The upstream measurements use one NVIDIA GeForce RTX 5090 through NInfer's public HTTP serving route.
Choose a model below for its detailed results, run conditions, output limitations, and reproduction
commands. These are recorded measurements; a model/backend being supported does not
mean every workload or concurrency has a published measurement.

Read the [measurement and publication rules](performance/methodology.md) for workload definitions,
metric formulas, statistics, comparison requirements, and the standard result-page format.

## Published coverage

Each cell links to the relevant result section. “Not published” describes measurement coverage,
not product support. C is configured request concurrency; K is the number of draft tokens.

| Model / weights | MTP0 context profile | Single-request speculative decode | Corpus makespan | MTP3 decode saturation |
|---|---|---|---|---|
| Qwen3.6-27B / `groupwise-int` | [8K–256K](performance/qwen3.6-27b.md#no-speculation-context-profile) | [MTP3](performance/qwen3.6-27b.md#single-request-speculative-decode) | Not published | [C=1, 2, 4, 8](performance/qwen3.6-27b.md#decode-saturation) |
| Qwen3.6-27B / `nvfp4` | [8K–256K](performance/qwen3.6-27b.md#no-speculation-context-profile) | [MTP3](performance/qwen3.6-27b.md#single-request-speculative-decode) | Not published | [C=1, 2, 4, 8](performance/qwen3.6-27b.md#decode-saturation) |
| Qwen3.6-35B-A3B / `groupwise-int` | [8K–256K](performance/qwen3.6-35b-a3b.md#no-speculation-context-profile) | [MTP3; DFlash K=7 stochastic/greedy](performance/qwen3.6-35b-a3b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash C=1](performance/qwen3.6-35b-a3b.md#corpus-makespan) | [C=1, 2, 4, 8](performance/qwen3.6-35b-a3b.md#decode-saturation) |
| Qwen3.8-27B / `groupwise-int` | [8K–256K](performance/qwen3.8-27b.md#no-speculation-context-profile) | [MTP3; DFlash2 K=7](performance/qwen3.8-27b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash2 C=1](performance/qwen3.8-27b.md#corpus-makespan) | [C=1, 2, 4, 8](performance/qwen3.8-27b.md#decode-saturation) |
| Qwen3.8-27B / `nvfp4` | [8K–256K](performance/qwen3.8-27b.md#no-speculation-context-profile) | [MTP3; DFlash2 K=7](performance/qwen3.8-27b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash2 C=1](performance/qwen3.8-27b.md#corpus-makespan) | [C=1, 2, 4, 8](performance/qwen3.8-27b.md#decode-saturation) |

Qwen3.8 and Qwen3.6-35B-A3B C=1 corpus points also supply their single-request phase tables.
The Qwen3.6-27B NVFP4 MTP3 phase table comes from a corpus C=1 point whose full makespan is
not published here. Qwen3.8 measurements use FP8 E4M3 row-256 KV; the Qwen3.6 measurements
use INT8 group-64 KV. Each model page records its build and run conditions.

The fork's [RTX PRO 6000 workstation records](performance/rtx-pro-6000.md) keep its pre-v3
DFlash2, OrcaRouter and NVFP4 prefill-schedule measurements with their own conditions.

Qwen3.8-Flash-Next is supported, but has no published result page in this campaign.
The Qwen3.8-27B measurements above describe a different architecture and artifact.
Flash Next's current execution scope is documented [here](maintainer/qwen4_exp-model.md).

## Reading the results

| Question | Metric to use |
|---|---|
| How fast is prompt processing or an individual decode phase? | Prefill phase, Server TTFT, Decode phase |
| How long does the full fixed request set take? | Corpus makespan, Corpus decode, Requests/s |
| What aggregate decode rate is sustained at a full batch? | Steady decode |

These rates use different time boundaries. Server TTFT is an internal phase sum; external
streaming TTFT has its [own benchmark contract](../tools/bench/ttft/README.md). Stochastic runs
can generate different token totals even with the same prompts and seeds. Output-limit and
repetition samples remain labeled in the measured corpus; throughput alone does not establish
successful task completion. See the [35B termination and anomalies](performance/qwen3.6-35b-a3b.md#termination-and-anomalies)
and [Qwen3.8 completion outcomes](performance/qwen3.8-27b.md#completion-outcomes).

## Related references

- [Serving benchmark runners](../tools/bench/README.md#serving-corpus-benchmark): usage and local report files.
- [Engine and Op benchmarks](../bench/README.md): their separate measurement scopes and commands.
- [Capability evaluation](../eval/README.md): evaluation workflow; published scores live in the
  [model cards](README.md#model-artifacts), with a [README results index](../README.md#performance-and-evaluation).
- [Perplexity](perplexity.md): offline causal-scoring measurement and comparison rules.

Model pages are the detailed result authority. Model-card performance tables are excerpts linked
to those pages; update them together when replacing an applicable measurement. The README links
to the result records and their qualification scope.

## YaRN long context: Qwen3.8-27B

Measured November 7, 2026 on one Colab G4: RTX PRO 6000 Blackwell **Server Edition** (97,887 MiB),
driver 580.82.07, CUDA 13.3, GCC 13.3, Release `sm_120a`, source `4fc972a0` of the
`--yarn-factor` change (the later commits only touch tests and documentation). The artifact is
`neroued/Qwen3.8-27B-nvfp4-NInfer@a107ba1b` (23,719,715,844 bytes; native 262,144 positions).
All runs use FP8 KV, CUDA Graphs, one execution lane, MTP K=3 with the full proposal head for the
server runs, an automatic context capacity (74.8 GB device pool) and a 2-GiB desktop reserve.
Arms ran one at a time on an otherwise idle GPU; each figure is a single run unless stated. These
results do not establish Windows, RTX PRO 6000 Workstation Edition or RTX 5090 behaviour, and the
OrcaRouter checkpoint and the 35B MoE route were not run.

**Op qualification.** `ninfer_rope_test` passes its FP64-oracle YaRN cases at the 27B and 35B
shapes: decode and 128- to 2,048-token prefill, Text 1-D and MRoPE with distinct axes, positions
from 0 across the native boundary to 1,048,575, single-tensor MTP forms, padded and BF16-unaligned
strides and Graph replay, all within the unchanged pair-relative BF16 profile scaled by the
attention factor. The oracle's correction range for the published parameters is pairs 14..22, as
derived by hand. With `NINFER_LONG_CONTEXT_TESTS=1`, the BF16 (136 cases, 129.5 s) and FP8 (159
cases, 197.2 s) causal-attention suites pass, including decode and 129-column prefill tails at
the full 1,048,576-key envelope against the FP64 oracle. A first attempt that ran them beside a
served 1M-context engine failed those two cases with `cudaMalloc: insufficient Device memory`;
they were rerun on an idle GPU.

### Short-context quality

`ninfer-perplexity` on the quick `ninfer-ppl-1m-v1` set (four streams, 261,167 scored tokens):

| Window / stride | YaRN | PPL | Mean NLL | NLL change | Score rate (tok/s) |
|---|---|---:|---:|---:|---:|
| 4,096 / 2,048 | off | 4.316363 | 1.462413 | — | 7,084.8 |
| 4,096 / 2,048 | 2 | 4.318466 | 1.462900 | +0.000487 | 7,022.1 |
| 4,096 / 2,048 | 4 | 4.325740 | 1.464583 | +0.002170 | 6,997.3 |
| 65,536 / 32,768 | off | 4.063292 | 1.401994 | — | 9,299.4 |
| 65,536 / 32,768 | 2 | 4.065197 | 1.402462 | +0.000468 | 9,291.8 |
| 65,536 / 32,768 | 4 | 4.068942 | 1.403383 | +0.001389 | 9,288.7 |

Static YaRN costs a little at every length, as Qwen's cards warn, and the cost grows with the
factor: +0.05% PPL at factor 2 and +0.22% at factor 4 with a 4K window, +0.05% and +0.14% with a
64K window. Per domain at 4K and factor 4 the NLL rises by 0.0039 (Chinese reference), 0.0029
(English long form), 0.0016 (code) and 0.0002 (English reference); at 64K English reference
improves by 0.0036 while the other three rise by 0.0025–0.0038.
Scoring is deterministic: two more runs of the 4K native and factor-4 arms reproduced every NLL
exactly. Their score rates were 7,084.8, 6,987.7 and 6,966.8 tok/s native against 6,997.3,
6,969.6 and 6,978.5 tok/s with factor 4 (means 7,013 and 6,982, -0.4%), inside the native arm's own
1.7% spread, so the YaRN RoPE route has no measurable throughput cost here.

### Retrieval, speed and memory

A synthetic RULER-style multi-query haystack: deterministic pseudo-English filler, eight asked
key/value facts at depths 3, 15, 30, 45, 60, 75, 88 and 97%, eight distractor facts with other
keys, and one request for all eight values as JSON (thinking off, greedy). The same seed gives the
same keys and values at every length. The 512K and 1M requests also ask for a 300-word summary so
that decode runs past the JSON.

| Server setting | Prompt tokens | Correct | TTFT (s) | Prompt rate (tok/s) | Decode (tok/s) | Output tokens | MTP accepted / drafted | Pool peak (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| native, 262,144 | 32,012 | 8/8 | 2.47 | 12,960 | 199.4 | 137 | 102/105 | 1.90 |
| native, 262,144 | 255,014 | 8/8 | 45.37 | 5,621 | 158.2 | 137 | 102/102 | 15.78 |
| YaRN 4, 262,144 | 32,012 | 8/8 | 2.49 | 12,836 | 199.6 | 137 | 102/105 | 1.90 |
| YaRN 4, 262,144 | 255,014 | 8/8 | 45.69 | 5,582 | 158.2 | 137 | 102/102 | 15.78 |
| YaRN 2, 524,288 | 515,010 | 7/8 | 153.57 | 3,354 | 90.1 | 539 | 353/561 | 27.94 |
| YaRN 2, 524,288 (repeat) | 515,010 | 7/8 | 153.52 | 3,355 | 90.1 | 539 | 353/561 | 27.94 |
| YaRN 4, 1,048,576 | 515,010 | 8/8 | 153.04 | 3,365 | 92.8 | 567 | 377/573 | 27.94 |
| YaRN 4, 1,048,576 | 1,040,006 | 7/8 | 560.32 | 1,856 | 62.2 | 542 | 352/573 | 56.22 |
| YaRN 4, 1,048,576 (after the 1M request) | 32,012 | 8/8 | 2.51 | 12,756 | 199.6 | 137 | 102/105 | 58.12 |

TTFT is client-measured to the first content token, so it includes tokenization and the whole
prefill. Decode is client-measured after the first token; the 137-token rows are only the JSON
answer, which MTP drafts almost perfectly, so they are not comparable with the 539- and 542-token
rows. Pool peak is the context pool's peak occupancy from `/admin/vram`: FP8 KV (about 33 KB per
token for the 16 attention layers), MTP history and the retained prefix checkpoints. The pool is
reserved at startup, so device memory in use stays at about 94 GB in every arm.

Both misses are key confusions with a similarly named distractor rather than lost needles: at
515K with factor 2 `cobalt-ibex` (45% depth) received the code of the distractor `cobalt-heron`,
in both runs, and at 1M `emerald-puffin` (75%) received `saffron-puffin`'s. Factor 4 retrieved
all eight on the same 515K prompt, and the 256K rows, native or with factor 4, retrieve all eight.
This is one seed and one haystack type, not a benchmark score, so it does not rank factor 2
against factor 4.

Prefill follows the attention-dominated quadratic profile: the published RTX 5090 NVFP4 context
profile (linear plus quadratic term), scaled to this card by the 255K native row, predicts about
152 s at 515K and 563 s at 1.04M, against 153.6 s and 560.3 s measured. The last 8,192-token
chunks of the 1M prompt took about 7.9 s each. The ceiling itself costs nothing measurable: the
same 515K prompt took 153.0 s under the 1M setting (factor 4) and 153.6 and 153.5 s in two runs of
the 512K setting (factor 2). Factor 4 at 255K changes TTFT by +0.7% in single runs and leaves
decode unchanged.

MTP is scaled with the Text RoPE and keeps drafting past the native length: with the summary,
acceptance was 62.9% at 515K with factor 2, 65.8% at 515K with factor 4 and 61.4% at 1M, with
per-position counts 151/119/83, 155/128/94 and 152/119/81. No same-task run below 256K was made,
so these rates have no short-context baseline; the JSON-only rows draft nearly perfectly at
every length and do not discriminate.

DFlash2 drafts are refused past their own 262,144 positions at startup: `--yarn-factor 2
--max-context 524288 --spec dflash2` stops with "max_context exceeds the draft model's position
capacity of 262144 tokens; use MTP or no speculation for longer contexts".

## Flash-Next v3 on Colab G4 (2026-10-04)

Current measurements include the MTP batch-arithmetic corrections. Earlier candidates and
rejected alternatives remain labeled below so their gains and regressions are visible.

These are qualification measurements of the `workstation` candidate based on `2969beee`, with
text, MTP and Vision source integration. They establish performance on the RTX PRO 6000
Blackwell **Server Edition** (97,887 MiB), driver 580.82.07, CUDA 13.3.73, GCC 13.3, Release
`sm_120a`. All arms ran sequentially on one G4 allocation with an exclusive GPU. They do not
establish workstation Windows performance or improvement over the v2 engine.

The mixed v3 artifact was derived from
`igorls/Qwen3.8-Flash-Next-mixed-NInfer@5f0ee7e2`, using BF16, FP32-row-scaled FP8 and NVFP4
expert banks. MTP uses the full stored output head and the offline-converted NVFP4 MTP banks.
The PLE table remains mapped in host memory.

### Single-request Engine benchmark

`ninfer_bench` uses the public Engine with `bench/fixtures/bench_corpus.ids`, prefix cache
disabled, BF16 KV, max context and KV capacity 16,384, prefill chunk 1024 and CUDA Graphs.
Each point has one warmup and three measured repetitions. It forces 129 output tokens:
the first is produced by prefill, leaving 128 in the timed decode phase. Values below are
arithmetic means of per-repetition rates; total time is mean request time, excluding load.
The fixed corpus and forced output count can produce easy or repetitive continuations;
these figures do not measure general task quality or representative chat MTP acceptance.

| Prompt tokens | Draft K | Prefill tok/s | Decode tok/s, mean ± sample SD | Total seconds | Accepted / drafted |
|---:|---:|---:|---:|---:|---:|
| 512 | 0 | 7,360 | 133.392 ± 0.679 | 1.02981 | — |
| 512 | 3 | 7,344 | 132.777 ± 0.017 | 1.05122 | 32.29% |
| 512 | 5 | 7,343 | 101.817 ± 0.026 | 1.34417 | 21.45% |
| 2,048 | 0 | 9,257 | 136.224 ± 0.131 | 1.16151 | — |
| 2,048 | 3 | 9,191 | 263.027 ± 0.021 | 0.71861 | 96.94% |
| 2,048 | 5 | 9,187 | 265.348 ± 0.078 | 0.71226 | 88.89% |
| 8,192 | 0 | 8,691 | 135.538 ± 0.023 | 1.88765 | — |
| 8,192 | 3 | 8,631 | 262.233 ± 0.301 | 1.44629 | 98.96% |
| 8,192 | 5 | 8,634 | 286.900 ± 0.038 | 1.40160 | 97.25% |

MTP3 decode changes versus ordinary are -0.46%, +93.08% and +93.47%; MTP5 changes are
-23.67%, +94.79% and +111.67%, respectively. At 512 tokens the total request is 2.08% slower
with K=3 and 30.53% slower with K=5. At 2,048/8,192 tokens total latency falls by
38.13%/23.38% for K=3 and 38.68%/25.75% for K=5. MTP prefill is 0.21–0.75% slower at every
point. Teacher extension, full-head drafting and host PLE gathering remain measured costs.

Before the stronger MTP target-equivalence fixes, the same methodology measured:

| Prompt tokens | K | Earlier prefill tok/s | Earlier decode tok/s, mean ± SD | Earlier total seconds | Earlier acceptance |
|---:|---:|---:|---:|---:|---:|
| 512 | 0 | 7,139 | 131.271 ± 0.019 | 1.04754 | — |
| 512 | 3 | 7,336 | 138.630 ± 0.030 | 1.01051 | 30.81% |
| 512 | 5 | 7,344 | 136.966 ± 0.017 | 1.01847 | 29.07% |
| 2,048 | 0 | 9,022 | 133.909 ± 0.081 | 1.18352 | — |
| 2,048 | 3 | 9,204 | 278.155 ± 0.052 | 0.69178 | 96.94% |
| 2,048 | 5 | 9,189 | 300.316 ± 0.135 | 0.65602 | 88.14% |
| 8,192 | 0 | 8,688 | 134.937 ± 1.164 | 1.89225 | — |
| 8,192 | 3 | 8,651 | 277.121 ± 0.051 | 1.41794 | 98.96% |
| 8,192 | 5 | 8,637 | 324.917 ± 0.074 | 1.34890 | 97.25% |

Current MTP decode rates are lower than those earlier rates at all six points, worst -25.66%
for K=5 at 512 tokens. That case also changes acceptance and trajectory; the difference cannot
be attributed entirely to kernel cost. These sequential, non-interleaved measurements do not
establish a noise bound. The earlier candidate fails the strengthened target-equivalence check.

Reproduce from a configured build and an explicitly selected v3 artifact:

```sh
build/bench/ninfer_bench --weights "$ARTIFACT" \
  --corpus bench/fixtures/bench_corpus.ids -pg '512,128;2048,128;8192,128' \
  -r 3 --warmup 1 --max-ctx 16384 --prefill-chunk 1024 --kv-dtype bf16 \
  -o json --output-file ordinary.json
# Repeat with --spec mtp --draft-tokens 3, then 5, in separate runs.
```

### Resource cost and numerical fixes

At the benchmark's B=1, chunk 1024 and 16K BF16 KV capacity:

| Selection | Device weights GiB | Sequence MiB | Workspace MiB | CUDA Graph allowance MiB | KV payload MiB |
|---|---:|---:|---:|---:|---:|
| Text | 70.01 | 510.38 | 135.23 | 32 | 396 |
| Text + MTP3 | 71.50 | 552.08 | 153.88 | 64 | 429 |
| Text + MTP5 | 71.50 | 556.15 | 153.88 | 64 | 429 |

KV payload is included in sequence resources, not an additional sum. MTP adds about 1.49 GiB
of device weights, one KV/indexer layer, transient verification/replay state and two graph
families. These text benchmark figures exclude Vision weights and workspace. Image/video
functional tests separately check startup-bounded Vision workspace, but Vision throughput and
large-media peak memory are not measured here. Current page-cache-warm load took 11.07–11.19
seconds, including 7.87–7.98 seconds of weight upload; earlier runs took 9.86–10.03 seconds and
6.85–7.04 seconds respectively. The observed increase was not isolated to an individual source
change. Neither measurement represents cold-storage startup.

Two precision corrections were required without changing acceptance thresholds. The wide
hyper-connection up projection now retains FP32 through sigmoid; prompt scoring versus prompt
logprob readout improved from maximum absolute error 0.109504 to 0.00000858307 nats, under the
unchanged 0.05 criterion. Its private up plane grows by 20,480 bytes per prefill token (20 MiB
at chunk 1024), but the whole Program workspace peak remains 135.23 MiB in the text benchmark
because another phase determines the peak.

An ordinary-decode A/B with that correction disabled measured prefill 7,162 / 9,056 / 8,715
tok/s and decode 131.558 / 133.698 / 135.411 tok/s for the same 512 / 2,048 / 8,192 inputs.
With the correction, prefill changes are -0.31% / -0.37% / -0.31%, decode changes -0.22% /
+0.16% / -0.35%, and total time changes +0.23% / -0.07% / +0.33%. This three-repeat,
non-interleaved comparison records the observed cost; it does not establish statistical
insignificance or a speed gain.

The later shared MoE gate/up correction adds 2,560 bytes per token for A16 or 3,840 for A4
to that Op's private workspace. The B=1 Program workspace capacities above remain measured
at 135.23/153.88 MiB because another phase sets their peak. Aligning short projection/MoE
reductions and QSA probability partitions across widths closes the ordinary-versus-MTP target
comparison. Re-enabling the faster grouped MoE route above eight columns reintroduces
0.125-nat divergences at K=1/3/5, beyond the unchanged 0.05 bound, and was rejected.

Batched GDN projection originally rounded its private output to BF16 before convolution,
while B=1 retained FP32. This caused a real-model batch divergence at a 0.125-nat reference
top-two gap, above the unchanged 0.05 threshold. The corrected fused projection/convolution
retains FP32 and is qualified against the FP64 Op oracle and the same batch criterion.

### Concurrent decode and correction costs

The HTTP `decode-saturation` fixture is `long_decode_aime26_15`: 335 prompt tokens per request,
512 output tokens, greedy, BF16 KV, max context 2048, KV capacity 2048 × C, prefill chunk 1024,
CUDA Graphs and prefix cache disabled. One simultaneous wave per concurrency is measured.
The rate counts committed tokens only in complete telemetry intervals at exactly C decode rows;
the corresponding per-row rate is aggregate / C. All requests reached the 512-token limit.
This is a saturation workload, not evidence that the mathematical task was completed.

| C | Before numerical fixes, aggregate tok/s | Initial per-row FP32 correction | Earlier shared-weight FP32 correction | Earlier per row tok/s | Earlier full-batch measured seconds | Earlier wave seconds | Earlier vs before |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 241.3 | 232.0 | 234.7 | 117.3 | 3.00 | 4.480 | -2.8% |
| 4 | 379.0 | 352.8 | 368.0 | 92.0 | 4.00 | 5.804 | -2.9% |
| 8 | 562.7 | 497.1 | 510.9 | 63.9 | 7.00 | 8.520 | -9.2% |

The before arm has the original BF16 GDN intermediate and fails the strict batch criterion.
The shared-weight arm retains FP32 and the single-request reduction order, sharing weight reads across
finite 2/4/8-column tiles. It passes both the independent FP64 Op oracle and the real-model
batch criterion. The throughput cost remains a regression relative to the numerically rejected
arm; these single waves do not support a confidence interval or a claim that the cost is noise.
They also do not compare v3 against v2.

After all MTP equivalence corrections, the same one-wave HTTP workload measures:

| C | K | Aggregate tok/s | Per row tok/s | Full-batch seconds | Wave seconds | Acceptance |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0 | 226.667 | 113.333 | 3.00 | 4.644 | — |
| 4 | 0 | 351.998 | 87.999 | 5.00 | 6.055 | — |
| 8 | 0 | 457.000 | 57.125 | 8.00 | 9.458 | — |
| 8 | 3 | 418.000 | 52.250 | 9.00 | 10.463 | 64.42% |
| 8 | 5 | 342.909 | 42.864 | 11.00 | 12.588 | 51.54% |

Current ordinary rates are 3.4%, 4.3% and 10.5% below the earlier shared-weight arm.
At C=8, MTP3/5 reduce throughput by 8.5%/25.0% versus current ordinary decode and increase
wave latency by 10.6%/33.1%. All requests reach the fixed output limit. These single-wave
measurements show that fewer target rounds do not guarantee a throughput gain; they do not
establish confidence intervals or task quality. Initial MTP runs failed before loading because
the benchmark forced an optimized draft head. Both runners now accept `--proposal-head full`,
verify the actual head and prevent incompatible serial resume; these results use the full head.

The saved-v2 27-request comparison still fails its registered greedy-gap and prompt-logprob
criteria; see the
[active comparison record](research/flash-next-v3-port-2026-09-29.md#saved-v2-comparison-acceptance-gap).
Numerical acceptance moved to an independent transformers oracle and was granted on
October 6, 2026. Neither current nor historical timings establish v2 parity.

The initial per-row correction cost 3.9%, 6.9% and 11.6%, respectively. A fused Tensor Core
alternative passed the local Op oracle but still failed the unchanged Engine criterion at a
0.125-nat gap and was rejected. A single eight-column shared-weight tile passed correctness but
measured only 224.0 / 352.0 / 509.7 tok/s at C=2/4/8; matching tile size to active concurrency
recovered the smaller-batch loss. All approaches and adverse results remain visible here.

```sh
python3.11 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve --artifact flash-next="$ARTIFACT" \
  --mode mtp0 --sampling greedy --suite decode-saturation --concurrency 8 \
  --decode-tokens 512 --max-context 2048 --kv-capacity 16384 \
  --prefill-chunk 1024 --kv-dtype bf16 --output results/flash-next-c8
# Repeat with --mode mtp3/mtp5 --proposal-head full and distinct output directories.
```

### Windows, RTX PRO 6000 Workstation Edition (2026-10-07)

The `30d07ae8` build was measured beside the desktop with the same `ninfer_bench` method as
above: chunk 1024, BF16 KV, max context 16,384, one warmup and three repetitions. MSVC 19.51 and
CUDA 13.3 were used.

| Prompt tokens | Draft K | Prefill tok/s | Decode tok/s | Acceptance |
|---:|---:|---:|---:|---:|
| 512 | 0 | 7,451 | 138.2 | — |
| 2,048 | 0 | 9,601 | 137.6 | — |
| 8,192 | 0 | 9,077 | 137.4 | — |
| 512 | 3 | 7,463 | 135.9 | 32.5% |
| 2,048 | 3 | 9,396 | 260.1 | 96.0% |
| 8,192 | 3 | 8,874 | 259.2 | 99.0% |

At chunk 8192 the plain-decode rows are 7,598 / 11,475 / 10,922 tok/s prefill and
141.7 / 139.1 / 137.1 tok/s decode. These are on par with, or slightly above, the G4 Server
Edition figures above. The fixed corpus makes the MTP acceptance unrepresentative, as noted for
the G4.

### Windows serving, MTP draft count (2026-10-11)

Release `2026.10.11-v3port.11` (`27cd69b2`, LGPL FFmpeg, MSVC 19.51, CUDA 13.3) on the RTX PRO 6000
Workstation Edition beside the desktop. Server: `--max-context 131072 --kv-capacity 262144
--max-concurrency 4 --prefill-chunk 8192 --kv-dtype fp8 --vision --desktop-reserve-gib 6`, with
and without `--spec mtp --draft-tokens 3`. Rates are the server's own `timings.predicted_per_second`
(decode only). Single-request rows ran on an otherwise idle server, three greedy repetitions each,
median shown; every repetition of a row produced the same text, and both servers produced
byte-identical text for every row.

| Workload | Output tokens | No speculation | MTP K=3 | Change | MTP acceptance |
|---|---:|---:|---:|---:|---:|
| Chat explanation | 400 | 108.6 | 116.2 | +7% | 46% |
| Code with tests | 493 | 108.8 | 177.4 | +63% | 89% |
| Documentation summary (1,842-token prompt) | 400 | 105.5 | 133.9 | +27% | 61% |
| JSON Schema extraction | 116 | 108.2 | 157.7 | +46% | 80% |
| Thinking arithmetic | 440 | 105.2 | 139.8 | +33% | 64% |
| `logprobs` with `top_logprobs: 3` | 200 | 83.8 | 44.7 | −47% | no drafts |
| `logprobs` without `top_logprobs` | 200 | 97.9 | 110.6 | +13% | 44% |
| Four concurrent 300-token chat requests, aggregate | 1,200 | 225.6 | 189.7 | −16% | — |

A `top_logprobs` request reads the full vocabulary on the host and is offered no drafts, but on an
MTP server it still pays the K+1-wide verification, hence the loss. Four concurrent low-acceptance
requests lose aggregate throughput for the same reason.

The draft count was chosen on a four-concurrent document-extraction workload with long outputs
(median 600-740 tokens) over six-minute windows; per-request decode medians come from the engine
log and aggregate rates from five-second windows with four decode-ready requests:

| Draft K | Requests | Per-request median tok/s | Aggregate median tok/s | Acceptance |
|---:|---:|---:|---:|---:|
| 0 | 109 | 80.4 | 298.6 | — |
| 2 | 134 | 95.6 | 334.7 | 89% |
| 3 | 136 / 476 | 103.0 / 104.1 | 332.9 / 350.5 | 82% |
| 4 | 162 | 98.0 | 329.7 | 77% |
| 5 | 152 | 96.7 | 336.1 | 72% |

K=3 has the best per-request median in both of its windows (the second figure is a longer window
that also carried API-suite traffic). The aggregate differences between K=2 and K=5 are within
the variation between K=3's two windows. The workload mix changes over time, so these are
indicative rather than paired measurements. MTP is the better default for interactive, code and
structured work and for high-acceptance batches; a server dominated by `top_logprobs` requests or
by concurrent low-acceptance chat is faster without speculation.

### Prefill chunk invariance (2026-10-06)

Removing M-dependent split-K (GDN control projection) and final-wave split-K (Flash-Next FP8
projections) makes prefill bit-identical across chunk sizes; see the
[active plan](research/flash-next-v3-port-2026-09-29.md#prefill-chunk-invariance-restored-2026-10-06).
The `[2560,6144]` FP8 projection then underfilled the GPU at mid-size launches. Its table now uses
64x128 tiles through T=640 and 128x128 tiles through T=1536. The tile shape does not change a
column's K order.

`ninfer_bench` measured prefill only (`-pg P,1`, one warmup, five measured repetitions, BF16 KV,
max context 16,384) on one G4 allocation. Two interleaved rounds ran per chunk size, in the order
base, fix, retuned, alternative. The table gives each arm's change against the base run of the
same round, for 512 / 1,024 / 2,048 / 8,192-token prompts:

| Chunk | Round | Base prefill tok/s | Invariance fix only | Fix + retuned table (selected) | Fix + retuned + `[13312,2560]` 64x256 (rejected) |
|---:|---:|---|---|---|---|
| 1024 | 1 | 7,261 / 9,215 / 8,987 / 8,500 | -0.34 / -0.58 / +0.46 / +1.17% | +1.62 / +3.27 / +3.23 / +3.01% | +2.79 / +3.09 / +3.18 / +3.05% |
| 1024 | 2 | 7,410 / 9,458 / 9,219 / 8,688 | -0.27 / -1.05 / -1.10 / -0.86% | +0.54 / +0.69 / +0.63 / +0.80% | +0.46 / +0.79 / +0.59 / +0.96% |
| 8192 | 1 | 7,403 / 9,464 / 10,927 / 10,480 | -0.29 / -1.10 / +0.07 / -0.06% | +0.75 / +0.80 / +0.08 / -0.04% | +0.73 / +0.70 / -0.09 / -0.06% |
| 8192 | 2 | 7,411 / 9,465 / 10,929 / 10,483 | -0.36 / -1.18 / +0.08 / -0.12% | +0.35 / +0.63 / +0.06 / -0.13% | +0.62 / +0.69 / -0.11 / -0.09% |

Round 1 at chunk 1024 is confounded by warm-up drift. The base itself moved +2% between rounds,
and the later arms in that round inherit the rise. Rounds at chunk 8192 agree within 0.4%.

Read from the consistent rounds:
- The invariance fix alone costs up to 1.2% prefill on short and mid prompts.
- The retuned table recovers that cost and gains 0.4-0.8% at 512-1,024-token prompts. The removed
  splits were tuned for 170 SMs, so they did not fit this 188-SM device.
- At 2,048-8,192-token prompts with chunk 8192, the retuned table is within ±0.13% of base.
- The `[13312,2560]` alternative shows no consistent further gain and was not adopted.

**Decode and memory.** An earlier `pp+tg128` run of the fix-only build left decode unchanged,
for example 135.87 against 135.84 tok/s. The retuned table does not touch the decode routes.
Peak Engine workspace at chunk 1024 drops from 135.2 to 122.7 MiB. At chunk 8192 it is 981.8 MiB
in both builds.

## Bounded-string grammar trial (2026-10-19)

This is a Windows CPU grammar-probe measurement in an isolated checkout, not a
model/serving benchmark. Baseline is committed `workstation` at `2b62e8e5`;
candidate is that tree plus upstream `0a8530d4` (length-bounded JSON string masks).
MSVC 19.51.36260, Release, Windows x64; the build is configured with CUDA
13.3.33 but the measured executables use no GPU. The machine's shared GPU and
serving process were not exercised by this experiment.

Each process runs `ninfer_json_schema_test --probe` with the same 257-token
byte/EOS vocabulary and 4-MiB grammar cache. Input is one bounded-string schema
and 20/37/31/37 candidate strings for the four rows below. Candidates cover
minimum/maximum neighbors, ASCII, multibyte Unicode, newline, backslash and NUL.
Python 3.11.15 with JSON Schema 4.26 independently determines acceptance; every
baseline and candidate result agrees. Native counted-unit/preview/rollback,
schema-boundary/dense-token and regex/choice checks also pass.

Timing covers process launch, cold grammar construction and all token masks for
that row. One warm-up per arm/row is excluded, followed by five repetitions with
alternating arm order. Reported values are per-row medians and observed ranges;
there is no average across workloads or assumed engine-speed acceptance gate.
The decision criteria are correct language/mask behavior and an explicit account
of both the intended large-bound benefit and small-bound costs.

| minLength / maxLength | Baseline median ms (min–max) | Candidate median ms (min–max) | Median time change |
|---|---:|---:|---:|
| 0 / 1 | 13.40 (12.48–14.50) | 14.37 (13.77–17.48) | +7.2% |
| 16 / 32 | 14.61 (13.32–16.51) | 14.85 (12.78–19.08) | +1.6% |
| 127 / 128 | 13.32 (13.21–13.63) | 14.33 (13.86–15.40) | +7.6% |
| 256 / 512 | 94.36 (90.59–96.55) | 22.48 (20.63–33.01) | -76.2% |

The large-bound workload improves; all three smaller-bound medians worsen.
The worst small-case median cost is 1.01 ms and the largest paired observed
candidate increase is 4.47 ms (16/32, one pair). These increases are not dismissed
as noise. Production vocabulary/cache residency, CPU utilization and peak memory
were not measured; neither GPU inference nor end-to-end serving improvement is
established. Correctness-oracle wall times and initial interpreter/encoding
setup failures are recorded in the [port assessment](maintainer/upstream-ports.md#contained-grammar-trial).

The grammar trial was carried in `58cd4222` and included in the staged sync.
These grammar-only measurements do not qualify the subsequent native runtime merge. See the [sync assessment](maintainer/upstream-ports.md#upstream-81c8ce0964d10fbf-assessment-october-19-2026)
for the remaining integration and dual-GPU qualification.

## FP8 TMA staged-input handoff (2026-10-08)

Upstream-sync qualification exposed intermittent first-token corruption in the existing FP8 TMA
pipeline. Changing each staged-input release from one representative lane per warp to every
consumer thread removes the alternating-weight reproducer. The separate
[integration record](maintainer/upstream-ports.md) covers the failure, mathematical checks and
real-model recovery qualification.

These focused Op timings use the same Colab G4 RTX PRO 6000 Blackwell Server Edition, driver
580.82.07, CUDA 13.3.73, GCC 13.3 and Release `sm_120a`. The public GDN-input Op uses
`FP8_E4M3FN_ROW_BF16`, N=16,384, K=5,120, T=128, `AllowA8`, and the existing seeded synthetic
BF16 activation and independently decoded packed-weight fixture. CUDA events enclose 512
complete Op launches per sample, including activation quantization. There are five successive
samples per arm; the table gives their individual means in microseconds, without dropping any.
The one-buffer case reuses a hot weight. The two-buffer case alternates identical 84-MB payloads
to exercise changing weight addresses and a larger cache footprint. GPU jobs run serially.

| Handoff | One weight: five sample means, microseconds | Alternating weights: five sample means, microseconds |
|---|---|---|
| Original representative arrivals, numerically unsafe | 52.63 / 52.78 / 52.72 / 51.44 / 47.10 | 69.65 / 69.59 / 69.61 / 69.60 / 69.42 |
| All-consumer CTA barrier, rejected as unnecessary synchronization | 53.14 / 53.08 / 50.86 / 47.34 / 47.11 | 69.68 / 69.60 / 69.37 / 69.22 / 69.31 |
| Every reader arrives, selected | 55.12 / 53.76 / 50.22 / 49.23 / 49.27 | 69.66 / 69.36 / 68.39 / 67.21 / 67.17 |

The selected arm's medians are 50.22 and 68.39 microseconds; the original arm's are 52.63 and
69.60. The warm distributions also contain slower selected samples: its range is 49.23–55.12
against 47.10–52.78. These short, non-interleaved runs show substantial within-run drift; they
do not establish a stable gain or regression. Selection follows the direct per-reader memory
ordering contract and correctness reproducer, not the timing median. Whole-inference speed,
Windows GPU performance and desktop coexistence on the merged revision remain unmeasured.
The handoff adds no production allocation or shared-memory storage; the regression test alone
allocates a second weight and performs its bounded repeat check.

## Final-tip upstream adaptation qualification (2026-10-23)

The following paragraphs retain the intermediate qualification history. The
[final October 28 comparison](#final-native-hc-comparison-2026-10-28) records
the latest candidate, completed checks, gains, regressions and resource costs.

The source merge through `64d10fbf` is still being qualified. Numerical checks of
the fork's FP8 QSA split found three K/V failures at T=129 and T=193 with A8
activations. Above 48 columns, Q/G retain A8; the narrow K/V tail uses A16 activations against
the same independently decoded FP8 weights and unchanged oracle criteria.
The extra activation path has not received matched performance measurements.

NVFP4 MoE retains A16 vector execution through 48 columns, including short
prefill and the largest supported MTP batch. Prefill output reduction uses a
fixed expert-rank order: atomic accumulation failed causal-prefix and repeated
scoring checks on the real mixed artifact. Fixed reduction passes those focused
checks but stores ten FP32 contributions per token instead of one routed sum,
an extra 360 MiB at T=4096. This is the contribution-buffer difference, not a
measurement of total Program workspace, peak VRAM or end-to-end speed. Those
resource and timing comparisons remain open.

Wide HC prefill retains four Down partitions and the paired normalization
reduction across window sizes. QSA prefill uses the instance's maximum-context
envelope. The combination passes exact score equality across 1,024-, 1,536- and
4,096-token chunks; the initial variable schedules differed by up to 9.4893 nats.
This replaces the affected upstream wave and normalization choices to preserve
the fork's chunk semantics. Their timing and total-workspace costs remain
unmeasured, and no speed gain is claimed.

Short-window target parity additionally requires one segmented attention-input
reduction, uniform HC Down/Up schedules, FP32 shared-expert activation and A16
FP8 output-projection activations through 48 columns. Together these pass the
eight-prompt ordinary and K=1/3/5 MTP comparison at 96 outputs per request;
removing any one of the latter three changes fails the same 0.05-nat criterion.
These choices replace faster short-window arithmetic choices to preserve target
decisions. Their timing, total workspace and B=1 costs remain unmeasured.
QSA also retains the single-query probability partitions through 48 columns;
the full sparse-attention oracle and exact column comparison pass across all
five KV encodings. Its partition workspace and timing cost remain unmeasured.

CPU NVFP4 expert grouping also retains route-rank reduction and chooses its tiny-input
BF16-dot fallback per column. Exact single-column/batch checks and independent FP64
oracles pass on the three CPU dispatch qualification routes. The output tile occupies
160 KiB of worker stack, an increase of 144 KiB per worker; full 256-column grouping
adds 10 KiB of member metadata to the planned Host workspace. Their timing cost is
unmeasured. Extended bounded MTP initially fails target parity at a 0.125-nat gap
under both 32 and 28 GiB Engine budgets, while ordinary graph/eager passes. With
the CPU correction, 28 GiB passes ordinary/MTP eager and graphs at the unchanged
criterion; at that stage 32 GiB still fails at 0.125. It blocks the benchmark
until the router correction below; it is not a measured slowdown.
The standalone Flash-Next router now retains the complete route's reduction through
48 columns; its prior vector-to-MMA transition fails exact coefficient invariance.
Independent router and MoE criteria pass after the correction, and ordinary/MTP
eager and graph execution now pass at both 32 and 28 GiB under the original
criterion. No fixed cache cap or extra cache wait is retained. Timing is unmeasured.

The matched Engine comparison uses the committed pre-sync fork
`2b62e8e5` and this candidate on the same G4: Qwen3.8-27B NVFP4 and the mixed
Flash-Next artifact, prompt lengths 512/2048, 64 timed decode tokens after the
prefill token, C=1/4 and K=0/3, BF16 KV, context 4096 and chunk 1024. Each point
has three process pairs in AB/BA/AB order, with one warmup and three measured
repetitions per process. All rows use the same fixed corpus prefix and force
65 outputs. Wave wall time covers submission through all results, after prompt
preparation; it excludes model loading and tokenization. Per-request prefill and
decode rates remain separate from wave throughput and latency. Comparisons use
means of the three process means and the median paired percentage change, with
the full three-pair range retained. Rows within a concurrent wave are not
independent repetitions. Load, graph/workspace storage, host CPU/RSS/I/O and GPU
temperature/clocks are recorded separately. This synthetic workload establishes
neither HTTP serving performance nor general MTP acceptance or response quality.

### Initial matched comparison, before resident graph optimization

The October 23 campaign completed all 48 processes on one Colab G4 RTX PRO 6000
Blackwell Server Edition (driver 580.82.07, CUDA 13.3, Release `sm_120a`). Baseline
is `2b62e8e5`; candidate includes the numerical, CPU and router corrections above,
before the subsequent resident graph and HC injection changes. The dense artifact
is the same in both arms. Flash-Next uses the original mixed v3 artifact in the
baseline and its exact canonical reframe in the candidate. These are resident
measurements; bounded Host execution has correctness evidence but no matched
performance result. Positive wave changes mean slower execution. Rates and wave
times are means of three process means; percentage changes are the median of the
three paired changes, with every pair represented in the range.

| Model | C | K | Prompt | Prefill tok/s baseline → candidate | Decode tok/s baseline → candidate | Wave seconds baseline → candidate | Paired wave change %, median (range) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense 27B | 1 | 0 | 512 | 13247.9 → 13548.5 | 68.52 → 69.68 | 0.9746 → 0.9579 | -1.57 (-2.03 to -1.52) |
| Dense 27B | 1 | 0 | 2048 | 15866.1 → 15973.1 | 68.22 → 68.70 | 1.0692 → 1.0617 | -0.65 (-0.80 to -0.64) |
| Dense 27B | 4 | 0 | 512 | 13303.4 → 13326.5 | 64.00 → 64.01 | 1.2015 → 1.2003 | -0.10 (-0.16 to -0.06) |
| Dense 27B | 4 | 0 | 2048 | 15820.1 → 15836.2 | 62.85 → 62.89 | 1.5844 → 1.5826 | -0.10 (-0.16 to -0.08) |
| Dense 27B | 1 | 3 | 512 | 12027.1 → 12250.0 | 117.61 → 106.83 | 0.5885 → 0.6424 | +9.17 (+9.11 to +9.22) |
| Dense 27B | 1 | 3 | 2048 | 15252.0 → 15346.9 | 211.84 → 204.04 | 0.4381 → 0.4486 | +2.42 (+2.34 to +2.43) |
| Dense 27B | 4 | 3 | 512 | 12050.8 → 12069.1 | 106.58 → 98.96 | 0.8186 → 0.9236 | +12.77 (+12.66 to +13.05) |
| Dense 27B | 4 | 3 | 2048 | 15223.9 → 15267.3 | 188.29 → 180.77 | 0.9397 → 0.9540 | +1.50 (+1.47 to +1.60) |
| Flash-Next | 1 | 0 | 512 | 7412.0 → 7197.0 | 135.46 → 106.28 | 0.5433 → 0.6748 | +24.29 (+23.87 to +24.49) |
| Flash-Next | 1 | 0 | 2048 | 9275.8 → 9259.6 | 132.85 → 103.71 | 0.7044 → 0.8397 | +19.16 (+18.99 to +19.49) |
| Flash-Next | 4 | 0 | 512 | 7475.1 → 7270.3 | 87.67 → 76.83 | 1.0319 → 1.1477 | +11.21 (+11.20 to +11.26) |
| Flash-Next | 4 | 0 | 2048 | 9285.1 → 9196.0 | 84.55 → 74.87 | 1.6737 → 1.7794 | +6.20 (+6.10 to +6.66) |
| Flash-Next | 1 | 3 | 512 | 7410.7 → 7186.2 | 114.32 → 117.44 | 0.6403 → 0.6175 | -3.57 (-3.61 to -3.53) |
| Flash-Next | 1 | 3 | 2048 | 9226.3 → 9223.7 | 254.38 → 229.88 | 0.4797 → 0.5017 | +4.59 (+4.53 to +4.68) |
| Flash-Next | 4 | 3 | 512 | 7426.6 → 7215.6 | 57.44 → 60.91 | 1.4790 → 1.3932 | -5.80 (-5.82 to -5.78) |
| Flash-Next | 4 | 3 | 2048 | 9218.8 → 9125.4 | 130.93 → 120.52 | 1.4508 → 1.4897 | +2.68 (+2.63 to +2.74) |

MTP acceptance changes with the target arithmetic. For the 512-token dense case,
accepted/drafted falls from 0.40698 to 0.36667 at C=1 and from 0.41716 to 0.39595
at C=4; both 2048-token dense cases accept every draft in this corpus. Flash-Next
512-token acceptance rises from 0.22523 to 0.30000 at both concurrency levels;
2048-token acceptance remains 0.95918. Dense C=1, P=512 uses 31 rounds instead of
29; the slowest C=4 row uses 35 instead of 31. These are corpus-specific
acceptance and scheduling costs, not evidence that all MTP slowdown comes from
slower kernels.

Flash-Next resident weight capacity is identical at 75,172,951,040 bytes, with no
Host experts. Peak process RSS falls from approximately 34.0 GB (31.66 GiB) to
2.45–2.61 GB (2.28–2.43 GiB), consistent with disk-backed PLE materialization.
Measured workspace peak rises from 128,684,032 to 167,378,944 bytes without MTP,
and from 161,357,824 to 167,378,944 with K=3. Free GPU memory after startup falls
by 77,594,624 / 140,509,184 / 94,371,840 / 234,881,024 bytes for C1K0 / C4K0 /
C1K3 / C4K3 respectively. These observations do not measure peak VRAM. Baseline
graph allowance and candidate actual preparation increment are different metrics
and must not be compared as actual graph storage. Candidate process CPU work,
including startup, is often about twice baseline: one C4K0 process uses
8.05 user + 6.94 system seconds against 3.38 + 3.59 seconds.

The first Flash-Next baseline load takes 345.458 seconds, versus 9.66–10.11 seconds
in subsequent baseline processes and 8.45–9.56 seconds in candidate processes.
The first dense baseline load also takes about 47 seconds. Cold I/O state was not
controlled; these retained outliers do not establish a causal load-time speedup.

Focused Nsight Systems captures of the measured repetition (one warmup, `-r 1`,
P=512, 64 decode tokens, CUDA profiler API boundary, no CPU sampling) explain a
resident Flash-Next cost. At C1K0, baseline launches 55,913 kernels with 497.681 ms
summed GPU duration and 64 graphs; candidate launches 61,601 kernels with
621.618 ms summed duration and 6,400 graphs. Graph-launch API time rises from
9.287 to 32.647 ms; candidate also makes 793,295 event queries taking 364.039 ms
of CPU API time. Separate HC injection consumes 60.510 ms over 6,240 calls;
candidate HC Down consumes 39.752 ms over 6,208 calls, against 37.923 ms for the
baseline fused Down/injection route. The selected vector Gate improves from
66.498 to 44.061 ms, while vector Down worsens from 34.738 to 39.127 ms and the
candidate also launches shared Down separately. Summed kernel time is not wave
latency. Dense C4K3 similarly increases from 29,269 kernels / 786.784 ms to
31,978 / 888.251 ms and 62 to 70 graph launches, alongside its extra MTP rounds.

The follow-up candidate captures each immutable resident traversal in two graphs
around PLE, retaining segmented cache/CPU handoffs for Host placement. HC injection
uses one 256-thread CTA per stream instead of 64 lanes per stream. The October 28
build, independent HC/MoE oracles and all 26 resident public Engine cases pass;
bounded Host ordinary/MTP eager/graph checks also pass at 32/28 GiB. Matched
timing acceptance of those changes is still in progress.
The original adverse results above remain
part of the assessment. Earlier logs were lost when the previous managed checkout
was removed; their recorded summaries are retained, and new evidence is saved
outside the checkout.

RAM, CPU work, disk preparation, graph preparation and desktop reserve must be
included when reporting the offload route. Earlier Core-stage model passes,
isolated Op passes, build times and test wall times establish no throughput or
latency improvement. Physical RTX 5090 and native Windows GPU acceptance remain
unverified on the final candidate.
The [port record](maintainer/upstream-ports.md#final-tip-adaptation-in-progress-october-23)
tracks source integration, failed attempts and remaining qualification separately.

### Resident graph and injection follow-up (2026-10-28)

A fresh G4 run completes all 48 processes with the same method, baseline and
workloads above, on driver 580.82.07 and CUDA 13.3.73. This candidate contains
the two resident traversal graphs and the 256-thread injection kernel, before
the subsequent HC-parent and shared-Down corrections. The GPU clocks are not
locked; before/after temperature and clock observations are retained with the
process records. Each row below retains all three paired observations.

Artifact sources are the mixed Flash-Next v3 bundle at
[`ef61b3c8`](https://huggingface.co/igorls/Qwen3.8-Flash-Next-mixed-NInfer/tree/ef61b3c856f91e58a58e41a5a6269b6c8648d266)
and dense 27B NVFP4 at
[`a107ba1b`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer/tree/a107ba1b5b0609d9ca90d5ed61439f5f5cc7d64d).
The Flash candidate is their canonical byte-preserving reframe with separate
HC parents. The official NVIDIA-backed recipe is a different representation
and has no real-artifact qualification in this comparison.

| Model | C | K | Prompt | Prefill tok/s base → candidate | Decode tok/s base → candidate | Wave s base → candidate | Paired wave change %, median (range) |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 1 | 0 | 512 | 13155.7 → 13420.9 | 68.63 → 69.70 | 0.9732 → 0.9579 | -1.54 (-1.77 to -1.39) |
| dense | 1 | 0 | 2048 | 15895.0 → 16010.7 | 68.27 → 68.74 | 1.0681 → 1.0607 | -0.59 (-0.94 to -0.56) |
| dense | 4 | 0 | 512 | 13305.9 → 13341.2 | 64.08 → 64.10 | 1.1997 → 1.1984 | -0.11 (-0.14 to -0.08) |
| dense | 4 | 0 | 2048 | 15936.4 → 15943.1 | 62.95 → 62.95 | 1.5785 → 1.5775 | -0.06 (-0.08 to -0.05) |
| dense | 1 | 3 | 512 | 12045.5 → 12292.6 | 117.94 → 107.12 | 0.5867 → 0.6405 | +9.17 (+9.10 to +9.24) |
| dense | 1 | 3 | 2048 | 15336.2 → 15417.1 | 212.30 → 204.44 | 0.4366 → 0.4473 | +2.49 (+2.39 to +2.50) |
| dense | 4 | 3 | 512 | 12054.8 → 12078.6 | 106.60 → 99.08 | 0.8180 → 0.9223 | +12.74 (+12.74 to +12.78) |
| dense | 4 | 3 | 2048 | 15328.5 → 15366.7 | 188.46 → 180.97 | 0.9352 → 0.9495 | +1.54 (+1.53 to +1.54) |
| flash | 1 | 0 | 512 | 7286.0 → 7108.8 | 134.23 → 110.72 | 0.5487 → 0.6513 | +18.66 (+18.65 to +18.79) |
| flash | 1 | 0 | 2048 | 9026.5 → 9052.7 | 131.23 → 109.06 | 0.7163 → 0.8144 | +13.70 (+13.51 to +13.89) |
| flash | 4 | 0 | 512 | 7414.9 → 7229.6 | 87.65 → 80.45 | 1.0343 → 1.1100 | +6.96 (+6.60 to +8.40) |
| flash | 4 | 0 | 2048 | 9277.2 → 9170.5 | 84.45 → 78.08 | 1.6697 → 1.7451 | +4.52 (+4.44 to +4.58) |
| flash | 1 | 3 | 512 | 7304.5 → 7077.1 | 112.71 → 139.99 | 0.6492 → 0.5308 | -18.43 (-18.67 to -17.64) |
| flash | 1 | 3 | 2048 | 9194.9 → 9159.1 | 252.69 → 236.11 | 0.4820 → 0.4960 | +3.07 (+2.47 to +3.14) |
| flash | 4 | 3 | 512 | 7406.3 → 7230.6 | 57.53 → 72.90 | 1.4760 → 1.2198 | -17.34 (-17.45 to -17.28) |
| flash | 4 | 3 | 2048 | 9222.8 → 9109.1 | 131.17 → 122.84 | 1.4481 → 1.4802 | +2.18 (+2.16 to +2.31) |

The dense result reproduces the earlier pattern: ordinary execution is near
neutral or faster, while MTP is slower in every measured row. Its acceptance
fractions remain 0.40698 → 0.36667 at C1P512, 0.41716 → 0.39595 at C4P512,
and 1.0 in both arms at P2048. Flash-Next short-prompt acceptance is now
0.22523 → 0.40698 at both concurrency levels, versus 0.30000 in the October 23
candidate; long-prompt acceptance remains 0.95918. Therefore the short-MTP
gains include changed acceptance as well as implementation cost. These figures
do not establish general acceptance or response-quality improvement.

Flash peak process RSS is 34.00–34.05 GB in the baseline and 2.44–2.54 GB in
the candidate. Weight capacities match: 75,172,951,040 bytes without MTP and
76,769,669,632 with K=3; Host expert bytes remain zero. Workspace capacities
remain 128,684,032 → 167,378,944 bytes without MTP and
161,357,824 → 167,378,944 with K=3. Candidate free GPU memory after startup is
54,525,952 / 41,943,040 / 46,137,344 / 39,845,888 bytes lower for C1K0 / C4K0 /
C1K3 / C4K3. These are startup observations, not peak VRAM measurements or pure
graph-allocation attribution. Whole-process CPU work, including startup and
warmup, remains higher, especially at C=4. Dense peak RSS also increases from
1.99–2.04 to 2.31–2.36 GB, despite matching weight/workspace capacities and
free GPU memory after startup.

The first Flash baseline load takes 254.652 seconds; its other loads span
10.518–10.947 seconds, against candidate 9.701–10.514. Dense baseline loads
span 3.214–79.828 seconds, against candidate 3.401–3.625. Cold state remains
uncontrolled, and the outliers remain included in these ranges.

The paired C1K0 P512 profiles contain 55,913 baseline kernels / 498.097 ms
summed duration and 61,601 candidate kernels / 598.076 ms. Candidate graph
launches are now 256 (10.739 ms API time), versus 64 (9.657 ms) baseline and
6,400 in the earlier segmented candidate. Busy event queries remain:
779,465 calls / 353.869 ms CPU API time versus 2 / 0.005 ms baseline.
These overlapping CPU waits are not additive GPU or wave latency.

Candidate HC Down takes 39.772 ms, with a further 37.488 ms in separate
injection; baseline fused HC Down/injection takes 38.012 ms. Candidate routed
vector Down takes 39.249 ms plus 26.559 ms for shared Down, against baseline
fused Down's 34.741 ms. Candidate vector Gate, shared Gate/control and route
selection take 44.040, 25.532 and 35.583 ms respectively; baseline Gate and
score kernels take 66.613 and 9.416 ms. These profiles identify remaining
fusion/packing costs without attributing the whole MTP result to kernel speed.

The subsequent HC-parent reframing is being qualified separately; the shared-Down
prototype below was rejected. This complete adverse/favorable result is retained as
the pre-correction measurement; its dense measurements can be reused when
only the Flash artifact or the H2560/E512/K10 MoE route changes.

### HC packing correction and rejected shared-Down trial (2026-10-28)

The legacy reframer missed the maintained converter's shared HC Down/injection
parent. The corrected offline reframe copies those BF16 operands byte for byte;
it preserves the original payload and all three companions. For the published
bundle, the canonical entry grows by 650,301,440 bytes to 33,260,744,704 bytes.
Resident weight capacity remains 75,172,951,040 bytes at C1K0, with no Host
experts; the entry retains the original unused payload as well as the native
copies. Fresh conversion directly from the supported checkpoint recipe already
uses the native packing.

The new exact-byte Text/MTP regression fails before the fix and passes afterward.
Python 3.11 passes 117 tests and eight subtests; HC retains its independent FP64
criteria. On the real artifact, exact 3,599-score chunk equality, ordinary/MTP
K=1/3/5 eight-prompt target parity and graph/eager checks pass. All 26 resident
public Engine cases pass on the final production build, as do ordinary/MTP K=3
eager/graph Host runs at 32/28 GiB. The final matched measurements follow below.

One three-repetition process with C1K0 gives wave means of 0.590085 s at P512
and 0.751975 s at P2048, with decode rates 123.50 and 120.94 tok/s. This is a
focused probe, not a matched final comparison. A subsequent NVFP4 prototype
fused shared Down into routed Down, retaining the per-lane FMA, warp and
expert-rank reductions. It builds on Linux/Windows and passes the block-scaled
mathematical suite, but its corresponding wave means are 0.602159 and
0.763346 s. The three wave ranges are 0.594097–0.612090 versus
0.589766–0.590562 at P512, and 0.755501–0.774531 versus 0.751501–0.752846
at P2048. These one-process, unpaired observations do not establish a stable
slowdown magnitude, but provide no reason to retain the added complexity.
The prototype and its workspace/interface changes were removed completely.

The final comparison reruns all 24 Flash processes with native HC parents,
using the same paired method, baseline, workload and numerical criteria. The
24 dense process measurements above remain applicable: dense 27B uses neither
the changed artifact nor the Flash HC path, and all MoE prototype changes have
been restored. No matched offload or physical RTX 5090 result is inferred.

### Final native-HC comparison (2026-10-28)

The final candidate keeps the resident graphs, width-invariant numerical
corrections and native HC-parent reframing. The shared-Down prototype is absent.
All 26 resident public Engine cases pass again, including Text/MTP/Vision,
graph/eager and batch target parity, scoring, tracing, restoration and concurrent
reuse. Ordinary/MTP K=3 eager and graph Host execution passes again at both
32 and 28 GiB, registering 67,948,068,864 Host expert bytes with 16 workers.
Windows CLI/server/Supervisor and affected test targets build on MSVC/ CUDA 13.3;
the existing LIBCMT warning remains. The three Windows CPU dispatch tests and
five focused reframing tests pass. Native Windows GPU execution is unqualified.

The table combines 24 new Flash processes with the 24 unaffected dense processes
above. Every point retains three process pairs in AB/BA/AB order, warmup 1,
three measured repetitions, the same P512/P2048 fixed corpus prefix, 65 forced
outputs (64 timed decode tokens), C1/ C4, K0/K3, BF16 KV, max context 4096 and
prefill chunk 1024. Model preparation/loading stays outside wave timing. Rates
are means of three process means; wave changes are medians of three paired
changes with their full range. Concurrent rows are not independent samples.
All 48 records validate requested counts and configuration; candidate Flash
Host expert bytes are zero in this resident comparison.

| Model | C | K | Prompt | Prefill tok/s base → candidate | Decode tok/s base → candidate | Wave s base → candidate | Paired wave change %, median (range) |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 1 | 0 | 512 | 13155.7 → 13420.9 | 68.63 → 69.70 | 0.9732 → 0.9579 | -1.541 (-1.768 to -1.390) |
| dense | 1 | 0 | 2048 | 15895.0 → 16010.7 | 68.27 → 68.74 | 1.0681 → 1.0607 | -0.588 (-0.941 to -0.563) |
| dense | 4 | 0 | 512 | 13305.9 → 13341.2 | 64.08 → 64.10 | 1.1997 → 1.1984 | -0.109 (-0.144 to -0.080) |
| dense | 4 | 0 | 2048 | 15936.4 → 15943.1 | 62.95 → 62.95 | 1.5785 → 1.5775 | -0.055 (-0.084 to -0.045) |
| dense | 1 | 3 | 512 | 12045.5 → 12292.6 | 117.94 → 107.12 | 0.5867 → 0.6405 | +9.168 (+9.098 to +9.244) |
| dense | 1 | 3 | 2048 | 15336.2 → 15417.1 | 212.30 → 204.44 | 0.4366 → 0.4473 | +2.494 (+2.388 to +2.498) |
| dense | 4 | 3 | 512 | 12054.8 → 12078.6 | 106.60 → 99.08 | 0.8180 → 0.9223 | +12.741 (+12.736 to +12.779) |
| dense | 4 | 3 | 2048 | 15328.5 → 15366.7 | 188.46 → 180.97 | 0.9352 → 0.9495 | +1.538 (+1.535 to +1.541) |
| flash | 1 | 0 | 512 | 7289.5 → 7277.8 | 134.26 → 123.47 | 0.5485 → 0.5900 | +7.563 (+7.528 to +7.567) |
| flash | 1 | 0 | 2048 | 9147.9 → 9301.5 | 131.38 → 118.99 | 0.7127 → 0.7607 | +5.441 (+4.998 to +9.761) |
| flash | 4 | 0 | 512 | 7462.4 → 7424.2 | 87.54 → 86.13 | 1.0334 → 1.0476 | +1.407 (+1.301 to +1.423) |
| flash | 4 | 0 | 2048 | 9284.2 → 9347.8 | 84.53 → 83.69 | 1.6684 → 1.6707 | +0.137 (+0.074 to +0.208) |
| flash | 1 | 3 | 512 | 7401.4 → 7347.8 | 114.36 → 151.37 | 0.6400 → 0.4937 | -22.864 (-22.893 to -22.820) |
| flash | 1 | 3 | 2048 | 9218.7 → 9394.2 | 254.60 → 248.46 | 0.4795 → 0.4769 | -0.516 (-0.594 to -0.506) |
| flash | 4 | 3 | 512 | 7418.7 → 7373.8 | 57.52 → 74.97 | 1.4755 → 1.1817 | -19.920 (-19.933 to -19.890) |
| flash | 4 | 3 | 2048 | 9219.7 → 9295.7 | 131.13 → 125.58 | 1.4486 → 1.4481 | -0.034 (-0.077 to +0.002) |

Eight of 16 point medians improve and eight regress; no average across different
workloads substitutes for this distribution. Dense ordinary execution improves
by 0.06–1.54% in observed wave medians, while all four dense MTP points regress
by 1.54–12.74% (worst individual pair 12.78%). Dense acceptance remains as above.
These are retained costs, not dismissed as noise or removed from the integration
assessment. Dense P512 MTP executes more speculative rounds than baseline;
changed acceptance accounts for part of that cost, without attributing it all
to slower kernels.

All four Flash ordinary points remain slower: 0.14–7.56% in wave medians.
The P2048 C1 range reaches 9.761%, despite a 5.441% median; the small C4 P2048
cost is also retained (0.074–0.208%). Both short-prompt MTP points improve
by 19.92–22.86%, including an acceptance change from 0.22523 to 0.42683.
Long-prompt acceptance remains 0.95918. Long-prompt MTP wave medians improve
by 0.516% at C1 and 0.034% at C4, but the C4 range includes one 0.001921%
slowdown. Per-request long-prompt MTP decode rates still fall from 254.60 to
248.46 tok/s at C1 and 131.13 to 125.58 at C4 (about 2.4–4.2%). Wave latency,
decode rate and acceptance describe different parts of execution; none is
substituted for another. This fixed corpus establishes no general MTP acceptance
or response-quality improvement and no HTTP serving result.

Resident Flash peak process RSS is 34.00–34.05 GB in the baseline and 2.43–2.53 GB
in the candidate. Weight capacities remain identical at 75,172,951,040 bytes
without MTP and 76,769,669,632 with K=3. Workspace capacity is
128,684,032 → 167,378,944 bytes without MTP and 161,357,824 → 167,378,944
with K=3. Free GPU memory after startup is 54,525,952 / 41,943,040 / 46,137,344 /
39,845,888 bytes lower at C1 K0 / C4 K0 / C1 K3 / C4 K3 respectively. These are
startup observations; neither peak VRAM nor pure graph storage is measured.
The earlier 360-MiB T4096 fixed-rank contribution-buffer increase and CPU worker
stack/metadata costs remain applicable outside this chunk 1024 resident matrix.

Whole-process CPU costs below include model startup, warmup and measured work;
GNU time supplies user/system CPU seconds and peak RSS. These costs remain
higher even though the candidate uses less Host RAM.

| Flash config | Baseline mean CPU user + system seconds | Candidate mean CPU user + system seconds |
|---|---:|---:|
| C1 K0 | 2.400 + 4.210 | 4.203 + 4.547 |
| C4 K0 | 3.367 + 4.147 | 7.403 + 6.910 |
| C1 K3 | 2.567 + 3.303 | 3.520 + 3.857 |
| C4 K3 | 3.530 + 4.197 | 7.573 + 6.653 |

The first final Flash baseline load takes 230.116 seconds; its other loads span
9.967–10.469 seconds, against candidate 9.183–9.841. The preceding campaigns'
254.652- and 345.458-second Flash outliers and the dense 79.828-second load remain
in their respective records. Cold I/O state was uncontrolled; no causal load
speedup is inferred from these differences. GPU clocks were not locked, and
boundary temperature/clock observations are retained with each process.

The native parent fixes the omitted maintained-recipe packing without runtime
weight repacking. Numerical invariance, generic cached/ CPU expert support and
the remaining split-kernel costs are retained together. Gains and regressions
remain specific to each measured workload. The rejected fusion trial provides no
validated faster replacement. Bounded Host correctness does not establish
offload throughput, desktop coexistence, physical RTX 5090 or Windows GPU
performance. The official NVIDIA-recipe artifact and OrcaRouter real-artifact
qualification remain separate. All completed evidence is saved outside the
managed checkout, and the Colab test instance was terminated after download.

### Dense MTP and Flash decode repair (2026-10-28)

The preceding native-HC campaign exposed the regressions that this repair
addresses; its unfavorable measurements remain above. The baseline remains the
pre-sync fork at `2b62e8e5`, with the same pinned mixed Flash and dense 27B
artifacts. The final repair is measured through the public Engine on a Colab
RTX PRO 6000 Blackwell Server Edition, CUDA 13.3, driver 580.82.07. It retains
the resident traversal graphs and native HC-parent artifact packing.

Dense target verification contained a duplicated final RMSNorm, vocabulary
projection and argmax. Removing it brings the focused P2048 K3 head launch
count from 84 to the baseline's 68, removing 13.3 ms of summed kernel time in
that profile. This profile establishes the removed work, not an end-to-end
speedup by itself.

The dense P512 acceptance loss has a separate numerical cause. Full prefill
outputs through layer 28 match baseline exactly. At layer 29, column 450,
GDN input controls and recurrent output still match; one BF16 SiLU-gated norm
value differs. The specialized D128 reduction grouped four squared values
left-to-right, whereas the generic gated row sums BF16x2 pairs. That one cast
changes 483 post-mixer values, then 4,277 post-MLP values and later recurrent
state. The specialization now preserves the generic SiLU pair grouping. Plain
and Sigmoid-gated norms retain their upstream arithmetic. Focused public Engine
C1 K0/K3 token sequences at P512/P2048 match baseline, and the C1/C4 dense MTP
acceptance and round counts return to baseline.

This is a consistency and performance decision, not a general accuracy gain.
At the differing element, the independent FP64 result is
-0.06420897734794798. Baseline BF16 -0.064453125 has absolute error
0.0002441476520520; upstream BF16 -0.06396484375 has error
0.0002441335979480 and is slightly closer to FP64 at this near midpoint. Both
satisfy the unchanged gated-norm accuracy budget. The RMSNorm and both gated
norm profiles pass their naive FP64 oracle suites.

Flash routing uses a fixed bitonic local sort for 512 experts/top-10, preserving
stable tie IDs and softmax arithmetic. Other supported geometries retain the
original sort. A focused routing profile falls from 35.430 to 16.615 ms summed
over 3,120 calls; this is kernel attribution, separate from whole-inference
results. Single-token NVFP4 vector Down combines the prepared shared projection
with the routed result, using vector loads and two independent FP32 FMA chains.
All NVFP4 vector widths now use that accumulation order, and separate/fused
shared Down use the same dot reduction. FP32 activations and the final BF16
output boundary are retained; there is no new runtime packing or workspace
allocation. Existing workspace remains allocated, so no VRAM saving is claimed.

The initial fast T1-only variant passed the independent MoE oracle but failed
2 of 26 real Engine cases: batched versus single and MTP batch target parity,
with a 0.125-nat reference gap beyond the unchanged 0.05 criterion. The other
24 cases passed. Timing was stopped before any final matrix record. Both
failures pass with the split control. Matching the expert/shared vector
reductions across widths fixes both failures; the acceptance threshold is
unchanged. The MoE oracle independently decodes packed weights and stored
scales, checks the complete residual/shared formula against FP64, and covers
BF16 residual output at T1/T8, guarded storage and CUDA Graph replay for NVFP4
and FP8. Real-model parity remains a separate qualification gate.

Rejected approaches remain part of the assessment. Four-row eleven-warp and
ten-warp shared-Down fusions passed their oracles but were slower; the revised
ten-warp version lost 4–7% decode in both process orders. A scalar one-row
variant lost about 12%. A shared Gate/Up fusion lost about 1%. Sparse-attention
query staging passed all five cache-format oracles, still used 255 registers,
added 8 KiB shared memory and gave no consistent gain (one pair cost 0.38%).
An iterative router alternative passed exact IDs and the oracle but gave mixed
results against the bitonic implementation. All are removed. The baseline
native MTP round benchmark failed its own no-fallback assertion with the full
head; no timings from that harness are used. Public Engine P2048 execution and
profiles supply the applicable evidence instead.

On the corrected source, all 26 resident Flash Engine cases pass, including
Text/MTP/Vision, eager/graph and batch target parity, scoring, exact prefill
chunk invariance, tracing, restoration and concurrent reuse. Ordinary/MTP K=3
Host execution passes in eager and graph modes under both 32- and 28-GiB Engine
budgets, registering 67,948,068,864 Host expert bytes with 16 workers. Both dense
MTP integration checks pass (native transaction recovery and batched attention
prefix execution). Linux CLI/server and affected targets build; the three
selected mathematical suites pass with the expanded residual tests. Windows
CLI/server/Supervisor and affected test targets build with MSVC and CUDA 13.3.
An earlier incremental build linked stale objects after a prototype restoration;
a coherent rebuild resolves that failure. The existing LIBCMT warning remains.
Native Windows GPU execution has not been tested in this repair.

The complete vector-repair campaign validates 48 process records under the
original method. Fifteen of 16 wave medians improve; C1/P512 ordinary Flash
retains +0.453% (-0.316 to +1.143%), with a -0.664% decode-rate median. C1/P2048
wave median improves 1.308%, but one pair costs 0.942%; its decode-rate median
improves 0.471% while the mean rate falls and one pair loses 2.458%. These
residuals are retained rather than dismissed as noise. Short Flash MTP
acceptance is 0.29293 at C1 and 0.31250 at C4, above the baseline's 0.22523 but
below the pre-repair candidate's 0.42683. Its short MTP wave gains shrink to
12.86–13.22%, versus the earlier 19.92–22.86%. Long MTP acceptance stays
0.95918. This tradeoff comes with the consistent vector profile; it is not a
general model-quality claim.

A fresh paired C1/P512 profile measures baseline/candidate GPU kernel spans of
546.310/549.975 ms, busy time 497.811/511.861 ms and idle time 48.500/38.115 ms.
It points to additional GPU work despite reduced idle time. The candidate's
643,842 event queries sum to 312.872 ms of API time overlapping execution;
that sum is not an elapsed-latency attribution. Sparse GQA occupies 15.449 ms
in its main kernel and 3.103 ms in merge, versus 4.876 and 1.680 ms in the
baseline's corresponding kernels. Profiled times establish an optimization
question; unprofiled paired Engine measurements decide the change.

A further shared Gate/Up two-chain trial preserves router order and passes the
MoE oracle but gives mixed AB/BA timing, so it is removed. A sparse-GQA
value-column split between warps passes the full five-format oracle (351.56 s),
reduces BF16 registers from 255 to 245 and improves P2048 about 2.5% in both
orders, but gives mixed P512 changes (-0.706% and +0.456%); it is not selected.

The selected follow-up instead partitions BF16 value columns across two CTAs
for a single-query partial MMA. Key partitions, per-partition FP16 probability
scaling, FMAs and warp reductions retain their original profile. Each CTA owns
128 output columns, and only one publishes the shared softmax statistics.
The selector owns this policy; other query counts and quantized-cache schedules
retain their existing implementation. It doubles CTAs and duplicates key-score
and cache-tile work, without adding workspace or a launch. BF16 vector-cache
register use is 253 rather than 255 (the scalar-cache specialization uses 191),
with no local-memory spills; no large occupancy improvement is claimed.
An upper-slice shared-memory read offset was missed in the first trial and
caught during source review. That owned trial was stopped before any timing;
the corrected variant passes the independent BF16 oracle. Two AB/BA C1 process
pairs give P512 wave improvements of 1.53% and 1.29% against the qualified
vector control, and P2048 improvements of 2.10% and 1.22%. The single-query
selection also passes the oracle's cross-width invariance checks. Full Engine
qualification and the final paired comparison follow; these probes alone do
not establish the final baseline comparison.

A four-CTA variant also passes the BF16 oracle but gives mixed AB/BA results:
P512 changes -0.179% and +0.046%; P2048 changes -1.905% and +1.912%. It is
removed. The selected two-CTA source is restored with fresh build dependencies
and rebuilt; its runtime and test sources match the passing Windows build.
No rejected trial or private instrumentation remains in the product.

### Final repair comparison

The selected source passes all 26 resident Flash Engine cases again, and
ordinary/MTP K=3 eager/graph Host execution at both 32 and 28 GiB. The independent
BF16 sparse-GQA suite passes across the single-query selection boundary;
quantized-cache schedules retain their existing mathematics and applicable
oracle evidence. MoE residual-output and RMSNorm/gated-norm oracles and both dense
MTP integrations above remain applicable. Windows CLI/server/Supervisor,
Flash Engine, MoE and sparse-GQA test targets build with the existing LIBCMT
warning. No Windows GPU test is inferred from those builds.

This table uses 24 new Flash processes and the 24 unaffected dense processes
from the complete vector-repair campaign. All 48 records validate configuration,
exit status and 65 requested/generated outputs (64 timed decode tokens).
The baseline, pinned artifacts and fixed corpus prefix remain unchanged.
Each point retains three process pairs in AB/BA/AB order, warmup 1, three
measured repetitions per process, P512/P2048, C1/C4, K0/K3, BF16 KV, max context
4096, prefill chunk 1024 and CUDA Graphs. Preparation/loading is outside wave
timing, and candidate resident Host expert bytes are zero. Rates and wave times
are means of three process means; percentage changes are medians of the three
paired changes, with the full range. Concurrent rows are not independent samples.

| Model | C | K | Prompt | Prefill tok/s base → candidate | Decode tok/s base → candidate | Wave s base → candidate | Paired wave change %, median (range) |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 1 | 0 | 512 | 13214.8 → 13533.5 | 68.68 → 69.84 | 0.9722 → 0.9557 | -1.552 (-1.989 to -1.547) |
| dense | 1 | 0 | 2048 | 15884.8 → 15992.9 | 68.37 → 68.80 | 1.0667 → 1.0599 | -0.614 (-0.686 to -0.604) |
| dense | 4 | 0 | 512 | 13293.8 → 13325.7 | 64.06 → 64.12 | 1.2001 → 1.1982 | -0.170 (-0.173 to -0.142) |
| dense | 4 | 0 | 2048 | 15878.9 → 15890.5 | 62.94 → 62.97 | 1.5805 → 1.5790 | -0.093 (-0.119 to -0.075) |
| dense | 1 | 3 | 512 | 12036.0 → 12266.2 | 117.91 → 119.64 | 0.5868 → 0.5780 | -1.485 (-1.527 to -1.483) |
| dense | 1 | 3 | 2048 | 15289.9 → 15382.4 | 212.33 → 213.55 | 0.4369 → 0.4342 | -0.611 (-0.655 to -0.578) |
| dense | 4 | 3 | 512 | 12056.3 → 12070.2 | 106.82 → 106.59 | 0.8168 → 0.8169 | -0.119 (-0.138 to +0.301) |
| dense | 4 | 3 | 2048 | 15287.8 → 15329.6 | 188.55 → 188.49 | 0.9365 → 0.9342 | -0.234 (-0.266 to -0.229) |
| flash | 1 | 0 | 512 | 7335.4 → 7347.4 | 134.92 → 135.23 | 0.5458 → 0.5443 | +0.021 (-1.070 to +0.236) |
| flash | 1 | 0 | 2048 | 9229.8 → 9484.4 | 132.40 → 133.30 | 0.7070 → 0.6976 | -1.550 (-2.175 to -0.287) |
| flash | 4 | 0 | 512 | 7477.3 → 7495.1 | 87.98 → 91.76 | 1.0291 → 0.9972 | -3.305 (-3.317 to -2.668) |
| flash | 4 | 0 | 2048 | 9282.9 → 9420.5 | 84.57 → 88.67 | 1.6681 → 1.6188 | -2.966 (-2.987 to -2.914) |
| flash | 1 | 3 | 512 | 7411.9 → 7404.7 | 114.33 → 131.81 | 0.6401 → 0.5559 | -13.169 (-13.174 to -13.110) |
| flash | 1 | 3 | 2048 | 9219.9 → 9445.7 | 254.45 → 260.39 | 0.4796 → 0.4639 | -3.247 (-3.374 to -3.220) |
| flash | 4 | 3 | 512 | 7428.6 → 7429.0 | 57.45 → 66.78 | 1.4769 → 1.2815 | -13.248 (-13.258 to -13.177) |
| flash | 4 | 3 | 2048 | 9222.4 → 9359.0 | 131.00 → 131.23 | 1.4490 → 1.4178 | -2.139 (-2.210 to -2.103) |

Fifteen of 16 wave medians improve. All four dense MTP medians improve
0.119–1.485%, restoring baseline acceptance (C1 P512 0.40698; C4 P512 0.41716;
P2048 1.0). Dense ordinary execution improves 0.093–1.552% by wave median.
Dense C4/P512 MTP still has one +0.301% wave pair and a +0.014% mean-wave change,
despite its -0.119% paired median. C4 MTP decode-rate medians remain lower by
0.061% at P512 and 0.036% at P2048; the worst pair loses 0.536%. These small
adverse results remain part of the assessment.

Flash ordinary C4 wave medians improve 3.305%/2.966% at P512/P2048, with
4.585%/4.853% decode-rate median gains. C1/P2048 improves 1.550% by wave median;
its decode-rate median gains 1.014%, but one pair loses 0.682%. C1/P512 retains
+0.021% wave median (-1.070 to +0.236%) and -0.233% decode-rate median
(-0.279 to +1.197%). Its mean wave and mean decode rate improve, as the table
shows; changing aggregation would change the sign, so neither replaces the
original paired metric. The large earlier ordinary-decode regression is removed,
but this is not a claim that every point or repetition is faster.

An independent baseline-versus-itself control uses six fresh C1/K0 processes,
the identical binary/artifact/workload and the same three-pair method. P512
wave changes are +0.133%, +0.032% and +0.115%; decode-rate changes are -0.148%,
+0.013% and -0.198%. P2048 wave changes span -0.717 to +0.239%, and decode-rate
changes span -0.309 to +0.281%. This demonstrates comparison drift at the
scale of the remaining P512 wave result. It does not prove equivalence,
remove the adverse A/B result or establish a causal explanation for its
decode-rate median. The original comparison and acceptance criteria are
unchanged; self-control samples are not pooled into the A/B table.

All four Flash MTP wave medians improve: 13.169%/13.248% at P512 C1/C4,
and 3.247%/2.139% at P2048. Short-prompt acceptance remains 0.29293/0.31250,
versus baseline 0.22523 and pre-repair candidate 0.42683. Thus short MTP gains
are smaller than the pre-repair candidate's 19.92–22.86%; that cost is retained.
Long-prompt acceptance stays 0.95918 and its decode-rate medians now improve
2.331%/0.180%. These results establish no general quality or acceptance gain.

Prefill-rate medians improve at 15 of 16 points; Flash C1/K3/P512 loses 0.143%
(range -0.172 to +0.025%). Decode-rate medians improve at 13 of 16 points;
the three negative medians are the two dense C4 MTP points and Flash C1/K0/P512
above. Wave latency, prefill, decode and acceptance are reported separately.

Whole-process resource measurements include startup, warmup and measured work;
CPU times and peak RSS come from GNU time. GB uses decimal bytes. GPU deltas
are free memory after startup, not peak VRAM or isolated CUDA Graph storage.

| Model/config | Mean CPU user + system s, base → candidate | Peak RSS GB, base → candidate | Startup free GPU delta bytes, candidate − base |
|---|---:|---:|---:|
| dense C1 K0 | 0.943 + 1.083 → 1.013 + 1.137 | 1.991–1.993 → 2.312–2.312 | 0 to 0 |
| dense C4 K0 | 1.070 + 1.057 → 1.160 + 1.143 | 2.020–2.021 → 2.339–2.341 | 0 to 0 |
| dense C1 K3 | 0.977 + 1.043 → 1.037 + 1.137 | 1.995–1.996 → 2.316–2.316 | 0 to 0 |
| dense C4 K3 | 1.120 + 1.117 → 1.203 + 1.193 | 2.036–2.036 → 2.356–2.357 | 0 to 0 |
| flash C1 K0 | 2.410 + 4.290 → 3.927 + 4.363 | 33.996–33.998 → 2.436–2.437 | -52,428,800 to -52,428,800 |
| flash C4 K0 | 3.383 + 4.187 → 7.177 + 6.720 | 34.041–34.043 → 2.474–2.475 | -41,943,040 to -41,943,040 |
| flash C1 K3 | 2.613 + 3.340 → 3.620 + 3.923 | 33.998–33.999 → 2.450–2.451 | -46,137,344 to -46,137,344 |
| flash C4 K3 | 3.547 + 4.257 → 7.647 + 6.913 | 34.049–34.049 → 2.532–2.534 | -39,845,888 to -39,845,888 |

Dense weight, sequence and workspace capacities and startup free GPU memory
match baseline; peak process RSS is about 0.32 GB higher and CPU costs rise.
Flash peak process RSS falls from about 34 GB to 2.44–2.53 GB, but CPU costs rise.
Flash weight capacities match at 75,172,951,040 bytes without MTP and
76,769,669,632 with K3. Workspace capacity is 128,684,032 → 167,378,944 bytes
without MTP and 161,357,824 → 167,378,944 with K3. Sequence-capacity deltas are
+2,579,968 / -3,910,912 / +22,968,832 / +14,576,640 bytes for C1K0 / C4K0 /
C1K3 / C4K3. The table retains the resulting startup free-GPU-memory costs.
The new kernels add no workspace allocation; this does not imply that the
whole merge has no memory cost. The earlier T4096 fixed-rank contribution-buffer
increase and Host worker stack/metadata costs also remain.

The first final Flash baseline load takes 222.188 seconds, then its loads span
10.553–10.981 seconds; candidate loads span 9.636–10.485. The vector-repair
campaign's 225.759-second load and the earlier 230.116/254.652/345.458-second
outliers remain in their records. Cold I/O and GPU clocks were uncontrolled;
no causal loading speedup or dismissal of small adverse timings is inferred.
Temperature/clock observations are retained at each process boundary.

The selected repair retains the public Engine route, native representations,
FP32 expert activations, per-input divisors and the existing numerical/parity
criteria. It introduces no checkpoint-specific route or runtime weight repacking.
G4 correctness and these resident timings do not qualify physical RTX 5090,
native Windows GPU execution, desktop coexistence, Host-offload throughput,
the official NVIDIA-recipe artifact or OrcaRouter real-artifact performance.
Those remain separate qualification work. No new HTTP serving result or broad
CTest claim is made. Complete raw evidence, rejected trials and source recovery
artifacts are saved outside the managed checkout. The owned Colab session is
terminated; the repair is included in the local upstream sync merge.

## Next upstream stage: original resident composition (2026-11-05)

This comparison evaluates the 52-commit source adaptation through upstream
`1408277a`, before its Flash graph repair. A is `98b6c010` (the `64d10fbf`
sync plus JSON whitespace repair); B is the initial adapted runtime. Neither is
an accepted workstation release. Both use the public Engine benchmark on the
same standard Colab G4: RTX PRO 6000 Blackwell Server Edition, driver 580.82.07,
CUDA 13.3.73, Linux, Python 3.11.18, about 176 GiB RAM. Clocks are not locked.
Both arms in the original matrix inherit Linux nice level 10.

The exact dense artifact is `neroued/Qwen3.8-27B-nvfp4-NInfer` at
`a107ba1b5b0609d9ca90d5ed61439f5f5cc7d64d`; Flash is the unchanged-payload
canonical reframe of `igorls/Qwen3.8-Flash-Next-mixed-NInfer` at
`ef61b3c856f91e58a58e41a5a6269b6c8648d266`. Both arms use identical files and
the committed `bench/fixtures/bench_corpus.ids`. All weights remain resident;
these timings do not measure Host-offload throughput.

Each profile uses `-pg "512,64;2048,64"` in that order, context 4096, chunk 1024,
BF16 KV, graphs, one warmup and three measured repetitions, with 65 forced
outputs and 64 timed decode tokens. KV capacity is `4096*C`, derived by the
benchmark; desktop reserve is the default 8 GiB. Dense K3 uses the optimized
proposal head, while Flash K3 uses its full shared head. Three fresh process
pairs per profile run AB/BA/AB: 48 processes and 720 measured requests, with
correlated concurrent rows. A and B have identical MTP accepted/drafted/round
and fallback counts in every corresponding report.

Rates and seconds are means of process means; changes are paired medians and
the complete three-pair range. Positive wave changes are slower; positive rate
changes are faster. Investigation flags were set before measurement at >2%
wave-median slowdown or >5% in any pair. Smaller adverse results remain visible
and do not establish equivalence.

| Model | C | K | P | Wave s A → B | Wave change % | Prefill tok/s A → B | Prefill change % | Decode tok/s A → B | Decode change % |
|---|---:|---:|---:|---|---|---|---|---|---|
| dense | 1 | 0 | 512 | 0.957778 → 0.956556 | -0.001 (-0.427 to +0.047) | 13450.011013 → 13554.511463 | -0.006 (-0.092 to +2.481) | 69.703798 → 69.776185 | +0.000 (-0.037 to +0.349) |
| dense | 1 | 0 | 2048 | 1.061036 → 1.060115 | -0.015 (-0.278 to +0.033) | 16000.499620 → 16031.500413 | +0.019 (-0.024 to +0.589) | 68.718583 → 68.769210 | +0.021 (-0.040 to +0.240) |
| dense | 1 | 3 | 512 | 0.541659 → 0.541495 | -0.013 (-0.075 to -0.003) | 12843.854077 → 12846.449697 | +0.011 (-0.011 to +0.061) | 127.900340 → 127.942563 | +0.022 (+0.009 to +0.068) |
| dense | 1 | 3 | 2048 | 0.437685 → 0.437695 | +0.007 (-0.020 to +0.020) | 15644.125930 → 15647.541980 | +0.031 (-0.022 to +0.057) | 209.610327 → 209.570151 | -0.018 (-0.025 to -0.014) |
| dense | 4 | 0 | 512 | 1.198308 → 1.198170 | -0.014 (-0.020 to +0.000) | 13367.427553 → 13373.719193 | +0.034 (+0.006 to +0.101) | 64.082990 → 64.090117 | +0.009 (+0.004 to +0.020) |
| dense | 4 | 0 | 2048 | 1.577096 → 1.577068 | -0.000 (-0.018 to +0.013) | 15959.765770 → 15962.664593 | +0.014 (-0.005 to +0.046) | 62.944617 → 62.942014 | -0.010 (-0.014 to +0.011) |
| dense | 4 | 3 | 512 | 0.823366 → 0.823309 | +0.007 (-0.038 to +0.010) | 12612.245187 → 12617.218817 | +0.046 (+0.016 to +0.056) | 112.546225 → 112.547576 | -0.007 (-0.023 to +0.034) |
| dense | 4 | 3 | 2048 | 0.931230 → 0.931005 | -0.021 (-0.065 to +0.013) | 15581.892970 → 15586.308590 | +0.038 (-0.027 to +0.074) | 182.599522 → 182.625928 | +0.018 (-0.023 to +0.048) |
| flash | 1 | 0 | 512 | 0.538654 → 0.525226 | -2.424 (-2.821 to -2.233) | 7369.687129 → 7429.662598 | +0.699 (+0.381 to +1.359) | 136.817629 → 140.653747 | +2.808 (+2.493 to +3.111) |
| flash | 1 | 0 | 2048 | 0.687086 → 0.699392 | +1.903 (+1.496 to +1.975) | 9503.074705 → 9523.604010 | +0.361 (-0.238 to +0.526) | 136.139877 → 132.512523 | -2.632 (-3.040 to -2.318) |
| flash | 1 | 3 | 512 | 0.562813 → 0.550758 | -2.145 (-2.154 to -2.127) | 7264.796829 → 7334.223838 | +0.838 (+0.665 to +1.367) | 130.329820 → 133.424883 | +2.412 (+2.291 to +2.422) |
| flash | 1 | 3 | 2048 | 0.466792 → 0.463688 | -0.434 (-1.224 to -0.327) | 9390.934138 → 9435.450717 | +0.175 (+0.095 to +1.165) | 258.698429 → 260.915383 | +0.685 (+0.562 to +1.332) |
| flash | 4 | 0 | 512 | 0.997427 → 1.012034 | +1.465 (+1.456 to +1.472) | 7478.569213 → 7492.717521 | +0.184 (+0.057 to +0.327) | 91.809839 → 89.903202 | -2.087 (-2.107 to -2.036) |
| flash | 4 | 0 | 2048 | 1.618760 → 1.640835 | +1.390 (+1.244 to +1.457) | 9422.571321 → 9420.062122 | -0.124 (-0.127 to +0.172) | 88.658471 → 86.128855 | -2.820 (-2.955 to -2.785) |
| flash | 4 | 3 | 512 | 1.280668 → 1.282682 | +0.156 (+0.149 to +0.167) | 7417.593480 → 7415.720616 | -0.007 (-0.078 to +0.009) | 66.863800 → 66.742616 | -0.178 (-0.197 to -0.168) |
| flash | 4 | 3 | 2048 | 1.417779 → 1.419185 | +0.099 (+0.065 to +0.133) | 9355.000238 → 9353.278294 | -0.008 (-0.076 to +0.028) | 131.344331 → 131.052921 | -0.216 (-0.237 to -0.212) |

The original stage improves both Flash C1/P512 wave medians by over 2%. Flash
ordinary C1/P2048 and C4/P512/P2048 regress by 1.390–1.903%; their decode rates
fall 2.087–2.820%. Flash C4 MTP has smaller adverse wave/rate medians, and dense
changes are small with mixed signs. These costs prompted the graph investigation
below rather than unconditional acceptance.

Whole-process CPU includes startup and warmup. Peak RSS is a process high-water
mark, not steady-state RAM. I/O/cache state was uncontrolled; load times do not
establish a causal loader speedup. Means and complete process ranges follow.

| Model | C | K | CPU s A → B | Peak RSS MiB A → B | Load s A → B |
|---|---:|---:|---|---|---|
| dense | 1 | 0 | 2.179 (2.143 to 2.216) → 2.193 (2.171 to 2.217) | 2206.452 (2206.059 to 2206.809) → 2211.951 (2211.121 to 2212.367) | 3.516 (3.418 to 3.684) → 3.480 (3.449 to 3.527) |
| dense | 1 | 3 | 2.225 (2.221 to 2.231) → 2.218 (2.207 to 2.232) | 2210.565 (2210.480 to 2210.613) → 2216.242 (2215.801 to 2216.875) | 3.557 (3.545 to 3.572) → 3.535 (3.524 to 3.548) |
| dense | 4 | 0 | 2.311 (2.302 to 2.323) → 2.298 (2.292 to 2.303) | 2233.827 (2233.434 to 2234.613) → 2239.138 (2238.672 to 2239.375) | 3.564 (3.546 to 3.577) → 3.542 (3.538 to 3.551) |
| dense | 4 | 3 | 2.456 (2.438 to 2.465) → 2.451 (2.439 to 2.470) | 2249.576 (2248.809 to 2250.363) → 2254.922 (2254.801 to 2255.000) | 3.706 (3.688 to 3.718) → 3.709 (3.694 to 3.723) |
| flash | 1 | 0 | 13.506 (8.307 to 23.778) → 8.361 (8.294 to 8.417) | 2324.302 (2323.188 to 2325.012) → 2332.018 (2331.875 to 2332.281) | 9.720 (9.690 to 9.764) → 9.728 (9.653 to 9.787) |
| flash | 1 | 3 | 8.126 (7.666 to 8.965) → 7.694 (7.681 to 7.715) | 2337.005 (2336.820 to 2337.273) → 2343.699 (2343.367 to 2344.223) | 10.032 (9.965 to 10.085) → 10.098 (9.996 to 10.232) |
| flash | 4 | 0 | 14.095 (14.054 to 14.156) → 14.242 (14.220 to 14.277) | 2360.206 (2359.820 to 2360.527) → 2383.803 (2383.438 to 2384.363) | 9.960 (9.908 to 9.996) → 10.272 (10.211 to 10.365) |
| flash | 4 | 3 | 14.735 (14.688 to 14.769) → 14.705 (14.689 to 14.721) | 2415.547 (2415.379 to 2415.672) → 2429.902 (2429.207 to 2430.762) | 10.574 (10.515 to 10.608) → 11.048 (11.003 to 11.086) |

Arena capacities match for corresponding profiles; unchanged arenas do not imply
unchanged process or CUDA-driver costs. Startup retained CUDA bytes and available
GPU memory follow; these boundary snapshots do not measure peak GPU residency.

| Model | C | K | Weight / sequence / workspace MiB (both) | Startup retained MiB A → B | Available after startup MiB A → B |
|---|---:|---:|---|---|---|
| dense | 1 | 0 | 19431.707 / 416.924 / 125.887 | 2.293 → 2.293 | 76711.875 → 76711.875 |
| dense | 1 | 3 | 20202.569 / 442.089 / 125.887 | 5.431 → 5.431 | 75911.875 → 75911.875 |
| dense | 4 | 0 | 19431.707 / 1636.265 / 125.887 | 10.293 → 10.293 | 75483.875 → 75483.875 |
| dense | 4 | 3 | 20202.569 / 1736.884 / 125.887 | 11.431 → 11.431 | 74611.875 → 74611.875 |
| flash | 1 | 0 | 71690.513 / 215.831 / 159.625 | 3.487 → 5.487 | 24605.875 → 24605.875 |
| flash | 1 | 3 | 73213.262 / 252.158 / 159.625 | 8.738 → 8.738 | 23039.875 → 23041.875 |
| flash | 4 | 0 | 71690.513 / 848.257 / 159.625 | 13.487 → 21.487 | 23961.875 → 23955.875 |
| flash | 4 | 3 | 73213.262 / 933.409 / 159.625 | 26.738 → 34.738 | 22341.875 → 22335.875 |

Initial harness errors were rejected before valid timing: `--kv-capacity` is not
a benchmark option, and Flash does not support `--lm-head-draft`. Failed setup
records are retained separately; the valid matrix uses the settings above.

Profiling the original mixed sequence and an isolated P2048 point shows the same
163,203 kernels and launch geometry at P2048. Decode kernel work is about
1,353 ms in B for either sequence, but inter-kernel gaps are 88.064 ms after
P512 versus 33.638 ms in isolation. PLE CPU waits are 28.556 and 30.177 ms.
These are instrumented durations, not replacement benchmark results. The
isolated three-pair P2048 loop improves 1.824% (-1.885 to -0.695%) instead of
the original matrix's slowdown: workload order is material, and the original
matrix remains the acceptance workload.

Explicit graph uploads after child updates do not repair the problem: B→U
wave changes are -0.001% (-0.240 to +0.124%) at P512 and +0.207%
(-0.120 to +0.286%) at P2048. Moving the resident PLE events outside the graph
gives only -0.313% (-0.612 to -0.201%) and -0.110% (-1.206 to -0.109%),
respectively. Both prototypes are removed; their complete records remain
outside the repository. No arithmetic, cache-format or tolerance change is
used to address graph dispatch.

Further three-pair probes retain the original A baseline and two-prompt order.
Returning only the Main traversal to split graphs (while keeping the new PLE
events, head placement and draft chains) gives -1.777% (-3.042 to -0.546%)
at P512 but +1.527% (+0.048 to +1.555%) at P2048. Waiting on the CUDA completion
event when no Host preparation or draft handoff can progress gives -2.081%
(-2.900 to -1.886%) and +0.988% (+0.752 to +1.625%). Moving the head projection
outside Main alone gives -1.656% (-2.367 to -0.108%) and +1.304%
(-1.289 to +2.209%). None repairs the long-prompt regression; all are removed.
These probes are diagnostic and do not replace the complete original matrix.

The next candidate rejects the coupled Qwen4Exp graph-composition change in
`1408277a`, restores the fork's resident prefix/PLE/suffix and separate
head/sampling graph, and retains the independent Engine admission dependency
fix from that commit. Its build, behavior and matched performance qualification
remain pending. This is an explicit implementation decision on top of reviewed
upstream ancestry, not a claim that the entire upstream range must be adopted.

### Selected rejection of the coupled graph rewrite

The selected source retains reviewed ancestry through `1408277a`, rejects its
coupled Flash Main/draft graph rewrite, and retains the independent Engine
admission dependency fix. It keeps the fork's resident prefix/PLE/suffix, separate
head/sampling graph and per-step draft handoffs while using the new expert
schedules. Proposal trace copies occur before the shared draft buffer is reused.
The precise CUDA dispatch mechanism is unresolved; the controlled rollback
supports this implementation decision without a driver-level causal claim.

The final matrix uses the same baseline, artifacts, two-prompt order, hardware,
metrics and criteria as above. Both arms inherit nice level 19 from the nested
worker, compared with 10 in the original matrix. Within each matrix the arms
match; the priority change and uncontrolled clocks limit cross-campaign causal
attribution. The selected matrix has 24 new Flash processes and
reuses the 24 dense processes whose execution was unaffected by this rejection:
48 reports and 720 measured requests in the combined matrix, not 48 new runs.
Every corresponding MTP acceptance/draft/round/fallback count matches. The eight
dense rows and their resource costs remain the preceding matrix's results.
The full Flash distributions follow, including smaller adverse changes.

| C | K | P | Wave s A → B | Wave change % | Prefill tok/s A → B | Prefill change % | Decode tok/s A → B | Decode change % |
|---:|---:|---:|---|---|---|---|---|---|
| 1 | 0 | 512 | 0.543652 → 0.538145 | -1.017 (-1.376 to -0.642) | 7331.662146 → 7336.104155 | +0.277 (-0.862 to +0.767) | 135.460450 → 137.070568 | +1.326 (+0.721 to +1.524) |
| 1 | 0 | 2048 | 0.695811 → 0.693570 | -0.965 (-1.849 to +1.891) | 9441.638336 → 9480.491717 | +0.329 (+0.117 to +0.795) | 134.081229 → 134.472534 | +1.095 (-2.810 to +2.672) |
| 1 | 3 | 512 | 0.556643 → 0.556117 | -0.112 (-0.371 to +0.200) | 7361.423378 → 7373.870015 | +0.260 (-0.266 to +0.515) | 131.734865 → 131.844833 | +0.087 (-0.186 to +0.350) |
| 1 | 3 | 2048 | 0.463604 → 0.463263 | -0.081 (-0.099 to -0.041) | 9453.174485 → 9471.473954 | +0.196 (+0.124 to +0.260) | 260.515197 → 260.441788 | -0.023 (-0.039 to -0.022) |
| 4 | 0 | 512 | 0.997807 → 0.997898 | +0.030 (-0.144 to +0.141) | 7477.275634 → 7476.978645 | +0.060 (-0.217 to +0.145) | 91.773238 → 91.767897 | +0.049 (-0.220 to +0.153) |
| 4 | 0 | 2048 | 1.619187 → 1.618986 | -0.013 (-0.073 to +0.048) | 9419.539070 → 9417.467659 | -0.033 (-0.176 to +0.143) | 88.631415 → 88.693521 | +0.084 (+0.003 to +0.123) |
| 4 | 3 | 512 | 1.280803 → 1.280442 | -0.024 (-0.056 to -0.005) | 7414.609731 → 7416.107583 | -0.016 (-0.073 to +0.150) | 66.860951 → 66.879876 | +0.028 (+0.019 to +0.038) |
| 4 | 3 | 2048 | 1.417756 → 1.417374 | -0.001 (-0.082 to +0.002) | 9355.510042 → 9360.569343 | +0.016 (+0.013 to +0.133) | 131.340253 → 131.324543 | -0.015 (-0.022 to +0.001) |

All four ordinary Flash decode-rate medians improve, by 0.049–1.326%, replacing
the original stage's long/C4 losses. This does not establish equivalence or a
universal gain: C4/K0/P512 retains +0.030% median wave time, C1/K0/P2048 has a
+1.891% wave pair, and long MTP retains -0.023% (C1) and -0.015% (C4) decode-rate
medians. C4/K0/P2048 and C4/K3/P512 retain small prefill-rate losses. The other
adverse phase/pair values are visible in the complete ranges. No final row meets
the predeclared investigation flags; smaller losses remain acceptance inputs.

The rejection gives up the original composition's larger short-prompt MTP
improvement (-2.145% wave median becomes -0.112%). It also avoids its C4 startup
GPU increment and most of its process-RSS/startup cost. Whole-process CPU,
peak-RSS and load means/ranges follow with the same interpretation as above.

| C | K | CPU s A → B | Peak RSS MiB A → B | Load s A → B |
|---:|---:|---|---|---|
| 1 | 0 | 8.365 (8.284 to 8.422) → 8.323 (8.296 to 8.347) | 2323.617 (2322.426 to 2324.578) → 2326.702 (2326.270 to 2327.219) | 9.754 (9.720 to 9.786) → 9.710 (9.685 to 9.753) |
| 1 | 3 | 7.681 (7.648 to 7.736) → 7.661 (7.600 to 7.699) | 2337.408 (2336.926 to 2337.680) → 2341.190 (2340.895 to 2341.461) | 10.061 (10.031 to 10.113) → 10.065 (10.031 to 10.105) |
| 4 | 0 | 14.053 (13.998 to 14.087) → 14.045 (14.023 to 14.081) | 2360.007 (2359.918 to 2360.156) → 2363.681 (2363.258 to 2363.926) | 10.036 (10.016 to 10.063) → 10.091 (10.071 to 10.105) |
| 4 | 3 | 14.649 (14.579 to 14.690) → 14.644 (14.600 to 14.677) | 2415.691 (2415.465 to 2416.039) → 2419.585 (2419.012 to 2420.062) | 10.573 (10.506 to 10.639) → 10.607 (10.572 to 10.670) |

Peak RSS increases 2.641–4.492 MiB in individual Flash pairs. Mean load changes
range from -0.043 to +0.055 s, with a worst paired increase of +0.077 s. CPU
means fall slightly, but individual pairs still increase by up to 0.099 s.
No steady-RAM, isolated loader or CPU-causality claim follows from these values.

| C | K | Weight / sequence / workspace MiB (both) | Startup retained MiB A → B | Available after startup MiB A → B |
|---:|---:|---|---|---|
| 1 | 0 | 71690.513 / 215.831 / 159.625 | 3.487 → 3.487 | 24605.875 → 24607.875 |
| 1 | 3 | 73213.262 / 252.158 / 159.625 | 8.738 → 6.738 | 23039.875 → 23043.875 |
| 4 | 0 | 71690.513 / 848.257 / 159.625 | 13.487 → 13.487 | 23961.875 → 23963.875 |
| 4 | 3 | 73213.262 / 933.409 / 159.625 | 26.738 → 26.738 | 22341.875 → 22343.875 |

Resident timings do not qualify Host-offload throughput, physical RTX 5090,
native Windows Flash execution or interactive desktop latency. The selected
source also passes all 26 resident Engine cases, four graph/PLE contracts,
ordinary/MTP eager/graph Host execution at 32 and 28 GiB, three Host route-trace
cases at 28 GiB and the initial startup profile at a 24 GiB Engine budget.
The issue #37 acceptance gate remains separate from these G4 results.

For both stage matrices, the reported arena capacities, startup-retained bytes
and available-after-startup values are identical across all three processes
within each arm/profile. These are startup boundaries, not peak-GPU measures.


## Initial native-session comparison (2026-11-05)

This evaluates the uncommitted adaptation through upstream `71cdd615` against
accepted `e1ab9af3`. All 48 processes complete and produce 720 measured request
records. **The original exact-proposal-count acceptance criterion fails** in one
of the 16 profiles; collecting the remaining profiles does not waive that failure.
Source integration, physical RTX 5090 acceptance and a stable release remain
separate decisions. The associated functional checks and repaired source failures
are recorded in [upstream ports](maintainer/upstream-ports.md#next-candidate-native-sessions-through-71cdd615).

Both arms run on one Colab G4 RTX PRO 6000 Blackwell Server Edition, Linux,
driver 580.82.07, CUDA 13.3.73, about 176 GiB RAM and absolute process nice 10.
The dense artifact is `neroued/Qwen3.8-27B-nvfp4-NInfer` at
`a107ba1b5b0609d9ca90d5ed61439f5f5cc7d64d`; Flash is
`igorls/Qwen3.8-Flash-Next-mixed-NInfer` at
`ef61b3c856f91e58a58e41a5a6269b6c8648d266`, canonically reframed without changing
payload words. Both use the committed 65,536-token benchmark corpus.

Each model uses C1/C4 × ordinary K0/MTP K3 × P512/P2048, BF16 KV, context 4096,
prefill chunk 1024, explicit KV capacity 4096 × C, the default 8 GiB desktop
reserve, one warmup and three repetitions per fresh process. There are three
process pairs per model/concurrency/backend, ordered AB/BA/AB. Each request
produces 65 tokens, with 64 timed decode tokens. Dense uses the optimized
proposal head and Flash the full head. Clocks, CPU affinity and filesystem-cache
state are not locked. No other qualification GPU job overlaps a measured process.

Rates and wave seconds are arithmetic means of process means. Percentage changes
are medians of the three paired changes with their complete range. Wave is
submit-to-completion wall time; concurrent request rows are correlated. The
predeclared investigation flags are >2% median wave slowdown or >5% in any pair,
plus any exact proposal-count mismatch. Smaller adverse results remain reported;
being below a timing flag does not establish equivalence or acceptance.

The original run stops at dense C4/K3/P512. After focused diagnostic probes,
collection resumes with the same frozen production and benchmark source and
retains every prior process, including the failing pair. Four remaining dense
and 24 Flash processes complete the original matrix. A setup attempt used an
unsupported `--kv-capacity` switch and failed before model loading; the benchmark
already derives and reports the required explicit capacity. A resume-harness
variable error also occurred before new GPU work and was corrected.

Native dense prefill defers proposal work that the baseline performs before
publishing its first token. Phase-rate differences can therefore include work
moving between prefill and decode; they do not establish isolated kernel gains.
The wave comparison covers the full request. Where proposal counts differ, it
does not isolate a speed difference at equal speculative work.

| Model | C | K | P | Prefill tok/s A → B | Decode tok/s A → B | Wave s A → B | Wave change %, median (range) |
|---|---:|---:|---:|---|---|---|---|
| dense | 1 | 0 | 512 | 13374.1 → 13316.4 | 69.54 → 68.92 | 0.9603 → 0.9673 | +0.804 (+0.559 to +0.821) |
| dense | 1 | 0 | 2048 | 15950.9 → 15961.4 | 68.56 → 68.58 | 1.0639 → 1.0619 | -0.075 (-0.454 to -0.024) |
| dense | 1 | 3 | 512 | 12789.4 → 13199.5 | 127.78 → 127.01 | 0.5424 → 0.5429 | +0.073 (-0.059 to +0.251) |
| dense | 1 | 3 | 2048 | 15583.8 → 15783.4 | 209.43 → 209.13 | 0.4386 → 0.4361 | -0.557 (-0.605 to -0.550) |
| dense | 4 | 0 | 512 | 13317.8 → 13340.5 | 63.99 → 64.21 | 1.2009 → 1.1950 | -0.511 (-0.515 to -0.439) |
| dense | 4 | 0 | 2048 | 15848.5 → 15871.7 | 62.85 → 63.07 | 1.5829 → 1.5763 | -0.423 (-0.449 to -0.388) |
| dense † | 4 | 3 | 512 | 12565.2 → 13071.4 | 112.34 → 113.26 | 0.8252 → 0.8152 | -1.375 (-1.686 to -0.589) |
| dense | 4 | 3 | 2048 | 15537.1 → 15758.3 | 182.52 → 182.14 | 0.9330 → 0.9249 | -0.876 (-1.096 to -0.633) |
| flash | 1 | 0 | 512 | 7327.3 → 7329.0 | 136.67 → 133.99 | 0.5396 → 0.5480 | +1.629 (+1.352 to +1.727) |
| flash | 1 | 0 | 2048 | 9435.6 → 9424.2 | 131.44 → 130.90 | 0.7054 → 0.7068 | +0.144 (-0.168 to +0.658) |
| flash | 1 | 3 | 512 | 7275.8 → 7269.7 | 129.76 → 131.13 | 0.5649 → 0.5588 | -1.060 (-1.316 to -0.845) |
| flash | 1 | 3 | 2048 | 9423.7 → 9391.5 | 258.51 → 258.43 | 0.4662 → 0.4660 | -0.125 (-0.131 to +0.154) |
| flash | 4 | 0 | 512 | 7464.1 → 7482.0 | 91.45 → 91.65 | 1.0008 → 0.9979 | -0.056 (-0.944 to +0.133) |
| flash | 4 | 0 | 2048 | 9415.9 → 9409.2 | 88.55 → 88.51 | 1.6203 → 1.6207 | +0.100 (-0.181 to +0.142) |
| flash | 4 | 3 | 512 | 7421.2 → 7424.3 | 66.77 → 66.80 | 1.2819 → 1.2802 | -0.118 (-0.170 to -0.098) |
| flash | 4 | 3 | 2048 | 9344.8 → 9330.5 | 131.19 → 131.19 | 1.4194 → 1.4197 | +0.013 (-0.050 to +0.103) |

Complete phase-rate change distributions:

| Model | C | K | P | Prefill change %, median (range) | Decode change %, median (range) |
|---|---:|---:|---:|---|---|
| dense | 1 | 0 | 512 | -1.437 (-1.483 to +1.706) | -0.938 (-0.950 to -0.801) |
| dense | 1 | 0 | 2048 | -0.104 (-0.200 to +0.503) | -0.060 (-0.132 to +0.281) |
| dense | 1 | 3 | 512 | +3.188 (+3.170 to +3.260) | -0.589 (-0.769 to -0.449) |
| dense | 1 | 3 | 2048 | +1.263 (+1.253 to +1.326) | -0.151 (-0.154 to -0.128) |
| dense | 4 | 0 | 512 | +0.177 (+0.151 to +0.185) | +0.373 (+0.304 to +0.373) |
| dense | 4 | 0 | 2048 | +0.113 (+0.058 to +0.268) | +0.337 (+0.327 to +0.391) |
| dense | 4 | 3 | 512 | +4.879 (+2.198 to +5.008) | +0.834 (+0.474 to +1.168) |
| dense | 4 | 3 | 2048 | +1.430 (+1.333 to +1.509) | -0.163 (-0.613 to +0.161) |
| flash | 1 | 0 | 512 | -0.023 (-0.187 to +0.280) | -1.999 (-2.178 to -1.714) |
| flash | 1 | 0 | 2048 | +0.050 (-0.553 to +0.141) | -0.437 (-0.855 to +0.066) |
| flash | 1 | 3 | 512 | -0.197 (-0.681 to +0.629) | +1.065 (+0.884 to +1.238) |
| flash | 1 | 3 | 2048 | -0.186 (-0.749 to -0.091) | -0.021 (-0.084 to +0.013) |
| flash | 4 | 0 | 512 | -0.107 (-0.232 to +1.070) | +0.033 (-0.176 to +0.822) |
| flash | 4 | 0 | 2048 | -0.190 (-0.292 to +0.272) | -0.048 (-0.076 to -0.011) |
| flash | 4 | 3 | 512 | -0.018 (-0.070 to +0.212) | +0.050 (+0.042 to +0.060) |
| flash | 4 | 3 | 2048 | -0.130 (-0.283 to -0.045) | -0.001 (-0.003 to +0.011) |

† Dense C4/K3/P512 fails exact counts in all three process pairs. Each process's
three measured repetitions total A → B: 357 → 354 rounds, 1,050 → 1,041 drafts,
408 → 411 accepted, three fallback steps in both, and accepted-per-position
[231, 132, 45] → [228, 135, 48]. Row four changes from 29 rounds/35 accepted
to 28/36 in every repetition; other rows and P2048 match. The other 15 profiles
match every row's rounds, drafted/accepted/fallback counts and position histogram.
This fails the original gate and is not reported as an equal-work throughput gain.

A focused P512/C4/K3 single-repetition probe reproduces the difference and finds
identical generated token IDs in both builds for every prime/warmup/measured row.
The first changed acceptance is row four at frontier 525, after both builds fully
accepted the preceding round; it precedes batch shrink. Source composition differs
in cached/batched versus regenerated one-token proposal work. The precise first
differing arithmetic remains unresolved. Private full-packed-KV, history-width,
combined-history and equal-text-RoPE-axis probes do not restore the baseline counts
and are rejected; none changes the candidate production source or benchmark binary.
This focused target-token match does not establish arbitrary-prompt equality.

### CPU, startup and memory costs

Whole-process user+system CPU time includes loading, warmup and measurements;
peak RSS is `ru_maxrss`, not steady-state Host memory. Means are shown with full
three-process ranges. Filesystem/cache state is uncontrolled; the first Flash
baseline process has a large CPU/startup sample, so these values do not isolate a
causal loader improvement. CPU inference, GPU execution and wall time are distinct.

| Model | C | K | CPU s A → B | Peak RSS MiB A → B | Load s A → B |
|---|---:|---:|---|---|---|
| dense | 1 | 0 | 2.143 (2.109–2.176) → 10.196 (10.193–10.201) | 2212.2 (2211.9–2212.7) → 2217.7 (2217.5–2218.0) | 3.663 (3.625–3.689) → 3.648 (3.638–3.658) |
| dense | 1 | 3 | 2.161 (2.147–2.174) → 6.095 (6.065–6.114) | 2216.0 (2215.7–2216.5) → 2226.8 (2226.0–2227.4) | 3.757 (3.742–3.775) → 3.831 (3.800–3.855) |
| dense | 4 | 0 | 2.286 (2.277–2.300) → 13.321 (13.310–13.342) | 2238.8 (2238.2–2239.4) → 2257.7 (2257.4–2258.2) | 3.800 (3.786–3.818) → 3.915 (3.898–3.934) |
| dense | 4 | 3 | 2.381 (2.371–2.388) → 9.377 (9.330–9.446) | 2255.0 (2254.7–2255.5) → 2296.6 (2296.1–2297.2) | 3.901 (3.897–3.906) → 4.287 (4.279–4.298) |
| flash | 1 | 0 | 13.725 (8.344–24.452) → 8.390 (8.305–8.437) | 2327.3 (2326.8–2327.5) → 2348.3 (2347.9–2348.7) | 8.819 (8.784–8.858) → 8.926 (8.902–8.950) |
| flash | 1 | 3 | 7.977 (7.566–8.675) → 7.639 (7.594–7.672) | 2340.6 (2340.1–2340.9) → 2362.3 (2362.2–2362.5) | 9.161 (9.078–9.247) → 9.312 (9.273–9.368) |
| flash | 4 | 0 | 14.005 (13.924–14.082) → 14.041 (14.011–14.078) | 2364.5 (2364.1–2364.8) → 2426.0 (2425.4–2426.5) | 9.173 (9.115–9.257) → 9.515 (9.475–9.538) |
| flash | 4 | 3 | 14.577 (14.507–14.658) → 14.696 (14.679–14.708) | 2419.2 (2418.7–2419.7) → 2481.2 (2481.1–2481.3) | 9.755 (9.698–9.832) → 10.181 (10.163–10.210) |

These are planned Device arena capacities in MiB, not independent whole-device
peak VRAM measurements. Weights and workspace capacities match per profile;
sequence capacities change with the native frame/state layout.

| Model | C | K | Weights MiB (both) | Sequence MiB A → B | Workspace MiB (both) |
|---|---:|---:|---:|---:|---|---:|
| dense | 1 | 0 | 19431.71 | 416.92 → 414.34 | 125.89 |
| dense | 1 | 3 | 20202.57 | 442.09 → 449.22 | 125.89 |
| dense | 4 | 0 | 19431.71 | 1636.27 → 1627.28 | 125.89 |
| dense | 4 | 3 | 20202.57 | 1736.88 → 1736.73 | 125.89 |
| flash | 1 | 0 | 71690.51 | 215.83 → 215.84 | 159.62 |
| flash | 1 | 3 | 73213.26 | 252.16 → 252.20 | 159.62 |
| flash | 4 | 0 | 71690.51 | 848.26 → 848.31 | 159.62 |
| flash | 4 | 3 | 73213.26 | 933.41 → 933.55 | 159.62 |

No profile crosses the predeclared wave-time flags, but adverse results include
Flash ordinary C1/P512 wave +1.629% median and decode −1.999% median, dense
ordinary C1/P512 wave +0.804%, plus the smaller losses in the tables. Dense process
CPU time rises materially. Counts, startup, RSS and layout costs remain part of
the disposition. The original count gate remains failed and this is not a stable-release
or physical RTX 5090 performance qualification.

### Ordinary Flash whole-inference profile

Nsight Systems 2026.1.3 profiles baseline and candidate at C1/K0/P512 with the same
artifact, corpus, BF16 KV, context/chunk/reserve and one measured repetition after
warmup. Graph-level captures omit measured graph device activity, even in a
process-start retry, and cannot establish GPU idle time. A final node-level
capture records 52,289 kernels in each arm with identical names/counts, 256 graph
launches, 523 asynchronous copies and 66 stream waits. The instrumented GPU kernel
sums are 510.500/510.498 ms; event records rise from 199 to 268. Graph nodes have
separate CUPTI stream IDs, so a primary-stream-only gap calculation is invalid.

The node capture's generation wall times are 552.238/551.171 ms and its union of
all GPU activity is 510.854/510.838 ms. That instrumented pair does not reproduce
the uninstrumented slowdown and does not replace the matched matrix. It finds no
changed kernel inventory, but does not establish a quantitative cause for the
1.629% wave regression. Profiling stops here; the adverse timing and unresolved
attribution remain in the disposition.

## Final-source comparison (2026-11-06)

This evaluates the uncommitted adaptation through upstream `a754fcf3`, including the
QSA envelope and NVFP4 MoE include repairs, against accepted `e1ab9af3`. The method is
the native-session comparison above: Linux RTX PRO 6000 Server Edition, resident GPU
experts, three AB/BA/AB pairs, nice 10, 16 profiles, 48 processes, 720 measured requests.
**Acceptance fails on both predeclared criteria.** Physical RTX 5090 and Windows
performance are not measured here, and this is not a stable-release qualification.

All generated token IDs match in all 720 requests. Exact proposal counts differ in 54
paired rows, in two groups:

- Flash K=3 P512 (45 rows): `rounds` is one lower with identical drafted and accepted
  counts. Each affected row has `fallback_steps=1`; rows with no fallback step match.
  The baseline incremented `rounds` on every Flash decode step, the candidate only on steps
  that drafted, so this is a counter definition change. It brings Flash in line with dense,
  which has counted only drafting steps in both arms.
- Dense C4/K3 P512 row 3 (9 rows): rounds 29 → 28, drafted 86 → 83, accepted 35 → 36,
  per-position `[21, 11, 3]` → `[20, 12, 4]`. This is the earlier exact-count failure.
  [Its cause is established below](#disposition-of-the-remaining-items-2026-11-06).

Paired wave-time change, B against A (median and range over three pairs, positive is
slower):

| Model | C | K | P512 % | P2048 % |
|---|---:|---:|---|---|
| dense | 1 | 0 | −0.20 (−0.22 to −0.20) | −0.19 (−0.20 to −0.15) |
| dense | 1 | 3 | −1.33 (−1.38 to −1.32) | −0.98 (−1.00 to −0.97) |
| dense | 4 | 0 | −0.08 (−0.11 to −0.08) | −0.15 (−0.15 to −0.11) |
| dense | 4 | 3 | −1.66 (−2.36 to −1.56) | −1.16 (−1.52 to −1.07) |
| flash | 1 | 0 | +0.88 (+0.07 to +1.67) | +0.47 (−0.69 to +0.97) |
| flash | 1 | 3 | −0.86 (−0.87 to −0.84) | −0.02 (−0.17 to −0.00) |
| flash | 4 | 0 | **+2.33 (+2.29 to +2.37)** | +1.38 (+1.32 to +1.47) |
| flash | 4 | 3 | −0.12 (−0.14 to −0.11) | −0.19 (−0.20 to −0.16) |

Twelve of the 16 are faster and four slower; the median is −0.17%, the worst +2.33%.
Flash C4/K0 P512 crosses the 2% investigation flag, and its cost is in decode (+3.4% at
P512, +3.1% at P2048) with prefill unchanged. A per-node graph trace of both arms shows
no kernel becoming slower. Launch counts rise about 2% across the same kernels, about 13
more forward passes per process. That count was read here as about 1.6 extra decode steps
per wave. [That reading is retracted](#disposition-of-the-remaining-items-2026-11-06):
measured waves run the same 64 steps in both arms.

Process costs match the earlier comparison. Dense CPU time stays well above the
baseline (C1/K0 2.0 → 10.1 s, C4/K0 2.1 → 13.3 s, C1/K3 2.1 → 6.0 s, C4/K3 2.3 → 9.3 s)
at unchanged wall time, almost all of it user time. Flash stays near parity (C4/K0 13.8 →
14.1 s) but moves from system to user time. The candidate coordinator yields in a loop
while work is outstanding; that fits the increase but has not been measured as its cause.
Engine construction is slower in every profile checked (dense C1/K0 3.32 → 3.45 s, Flash
C4/K0 8.86 → 9.27 s). Peak RSS rises by up to 0.1 GiB.

### Coordinator wait repair (2026-11-06)

Per-thread sampling attributed the dense CPU cost above: one coordinator thread ran at a
full core from the start of generation to exit, where the baseline slept in a blocking
stream synchronize. That undid the fork's blocking default on a shared desktop. After a
poll without progress the coordinator now blocks on the active stream through the same
blocking-synchronize default. Host experts and the Host PLE pipeline keep polling, because
the coordinator overlaps Host work with that stream.

The full comparison was repeated, same method and baseline. **Acceptance still fails on
the same two criteria.** All IDs match in all 720 requests, and the same 54 count rows
differ with the same values.

| Profile | CPU s before repair | CPU s after repair | Baseline CPU s |
|---|---:|---:|---:|
| dense C1/K0 | 10.1 | 2.2 | 2.1 |
| dense C1/K3 | 6.0 | 2.2 | 2.1 |
| dense C4/K0 | 13.3 | 2.3 | 2.2 |
| dense C4/K3 | 9.3 | 2.4 | 2.4 |
| flash C1/K0 | 8.3 | 8.3 | 8.2 |
| flash C1/K3 | 7.5 | 7.6 | 7.5 |
| flash C4/K0 | 14.1 | 14.2 | 14.0 |
| flash C4/K3 | 14.5 | 14.7 | 14.6 |

The dense cost is removed. **Flash is unchanged**: Flash runs the Host PLE pipeline, so it
still polls and holds a core while active, as the baseline already did. This is not a
regression, but it is also not a repair for that model.

The repair has a small dense timing cost, which is wake-up latency. Median wave change
against the baseline, before → after the repair:

| Model | C | K | P512 % | P2048 % |
|---|---:|---:|---|---|
| dense | 1 | 0 | −0.20 → +0.31 | −0.19 → +0.34 |
| dense | 1 | 3 | −1.33 → −0.62 | −0.98 → −0.47 |
| dense | 4 | 0 | −0.08 → +0.40 | −0.15 → +0.24 |
| dense | 4 | 3 | −1.66 → −1.11 | −1.16 → −0.72 |
| flash | 1 | 0 | +0.88 → −0.44 | +0.47 → +1.31 |
| flash | 1 | 3 | −0.86 → −0.89 | −0.02 → −0.01 |
| flash | 4 | 0 | **+2.33 → +2.21** | +1.38 → +1.36 |
| flash | 4 | 3 | −0.12 → −0.09 | −0.19 → +0.00 |

Eight profiles are now faster and eight slower, median 0.00%, worst +2.21%. Flash does not
take the new wait, so its movement between these runs shows campaign-to-campaign variation
of up to about 1.3 points on C1/K0; the other Flash rows moved by 0.2 points or less. Flash
C4/K0 P512 still crosses the 2% flag and is not addressed by this repair. Windows wake-up
latency for this blocking wait is unmeasured.

### Comparison after the dense repairs (2026-11-06)

The checkpoint metadata, reasoning prefix and JSON whitespace repairs changed runtime source,
so the full comparison was repeated on the final source with the same method and baseline.
**Acceptance still fails on the same two criteria.** All IDs match in all 720 requests, and
the same 54 count rows differ with the same values.

| Model | C | K | P512 % | P2048 % |
|---|---:|---:|---|---|
| dense | 1 | 0 | +0.36 (+0.35 to +0.38) | +0.34 (+0.26 to +0.37) |
| dense | 1 | 3 | −0.67 (−0.75 to −0.66) | −0.41 (−0.44 to −0.21) |
| dense | 4 | 0 | +0.36 (+0.35 to +0.47) | +0.31 (+0.13 to +0.37) |
| dense | 4 | 3 | −1.00 (−1.03 to −0.89) | −0.65 (−0.80 to −0.52) |
| flash | 1 | 0 | +0.89 (+0.41 to +1.76) | +1.59 (+0.14 to +1.71) |
| flash | 1 | 3 | −0.86 (−0.92 to −0.80) | −0.03 (−0.08 to +0.05) |
| flash | 4 | 0 | **+2.05 (+1.71 to +2.28)** | +1.37 (+1.26 to +1.43) |
| flash | 4 | 3 | −0.15 (−0.16 to −0.11) | −0.05 (−0.18 to −0.03) |

Eight of the 16 are faster and eight slower; the median is +0.14% and the worst +2.05%.
Flash C4/K0 P512 is the only investigation flag;
[its attribution follows below](#disposition-of-the-remaining-items-2026-11-06). Speculative
profiles are faster or level, and every ordinary profile
is slower, by up to +0.36% on dense and +2.05% on Flash.

In the dense C4/K3 P512 test, each row's counts are identical across all nine repetitions in
each arm, and the four rows give four different ID sequences in both arms. Only row 3 changes:
86 drafted and 35 accepted become 83 and 36, with its IDs unchanged. The drafter proposed
differently for that row without changing the output.
[The cause is established below](#disposition-of-the-remaining-items-2026-11-06).

Process costs are small but consistent. Peak RSS is 0.07 to 0.10 GiB higher in every profile.
Process wall time including loading is 0.2 to 0.9 s higher. CPU seconds match the baseline to
within 0.3 s; the baseline's Flash C1/K0 mean of 14.3 s contains one cold first process at
25.5 s, and its other two match the candidate at about 8.7 s.

Bounding JSON whitespace raises constraint compilation. On the Qwen3.8 vocabulary the first
JSON compile in a process took about 260 ms instead of about 9 ms, about 171 ms at a bound of
80 and 77 ms at 32. Later new grammars in that fixture took about 10 to 45 ms, and a cached
grammar under 1 ms. Decoding the same tokens takes the same time. Larger schemas were not
measured. Windows and physical RTX 5090 performance are not measured here.

### Disposition of the remaining items (2026-11-06)

The three items that fail acceptance were investigated on the same binaries, artifacts and
server. Host timing used an `LD_PRELOAD` shim on the unmodified executables; device records
used private object overrides at link time. None of this instrumentation is in the source.

**`rounds`.** In all 90 Flash K=3 rows, the baseline's `rounds` minus the candidate's equals
`fallback_steps` and every other field matches. The change is a counter definition only.
The old Flash counter was the outlier: dense counted only drafting steps in both arms.

**Dense C4/K3 P512 row 3.** The cause is a one-step BF16 tie in the drafter. Private
binaries that record each proposal reproduce the 86/35 → 83/36 change. At the first
differing proposal, the second draft position of one round:

| Arm | Top token | Logit | Second token | Logit | Proposal |
|---|---:|---:|---:|---:|---|
| baseline | 35935 | 19.125 | 1433 | 19.000 | wrong |
| candidate | 1433 | 19.125 | 35935 | 19.125 | matches the generated token |

An exact tie goes to the lower proposal-head row, which here is token 1433. Before that
proposal, hidden states already differ at rounding level in all four rows: 69%, 30% and 23%
of elements are bit-identical at draft positions one to three, cosine similarity stays at or
above 0.99996, and top-1 logits move by at most 0.125, one BF16 step at this size. One more
proposal differs in rows 0 to 2 later in the wave without changing any count. The two arms
compose the draft differently:
the baseline takes the first draft from a batched MTP forward of width K+1, while the
candidate runs a one-token forward after a KV-only history pass. That fits rounding-level
differences, but the first differing kernel was not isolated. Proposals only affect how
many tokens verification accepts, so generated IDs stay exact.

**Flash C4/K0 P512.** The attribution given above, about 1.6 extra decode steps per wave, is
**retracted**. Each measured wave runs 64 decode steps in both arms. The slowdown has two
parts, both measured without a tracer. Nsight Systems changes the result here: under its API
trace the baseline, not the candidate, was the slower arm.

| Part | Per step | Per wave | Evidence |
|---|---:|---:|---|
| Candidate launch order | about +0.1 ms | about +7 ms | every step, also in fast waves |
| Slow dispatch mode | about +0.3 ms | about +20 ms | on or off, in both arms |

The launch order comes from upstream's single resident main graph (`1408277a`). The baseline
launches its prefix graph at once and prepares PLE rows while it runs. The candidate waits
for those rows before its one launch. Host time between GPU completion and the next launch
rises from 18 to 82 µs, and first launch to GPU completion rises by about 40 µs.

The slow mode is a discrete adder. Step-by-step profiles are otherwise nearly identical
between waves. In the comparison above it is on for 8 of 9 candidate P512 waves and all
candidate P2048 waves, and for no baseline P512 wave, but the baseline reaches it too: whole
P512 waves in one process, and often the first 4 to 14 steps of P2048 waves. In the other
direction, Flash C1/K3 P512 reaches its faster mode only in the candidate.

| Hypothesis | Test | Result |
|---|---|---|
| Extra decode work | Count steps per wave | 64 in both arms |
| Clock or throttling | Sample clocks at 20 ms | SM about 2,400 MHz in both modes (one cold first wave dipped to 2,265), memory unchanged, no throttle flags |
| Clock transitions | Lock the SM clock at 2,407 MHz | The baseline still shows both modes; the candidate stays slow |
| GPU idle between steps | Delay each step by 0 to 600 µs | Step time net of delay unchanged: 10.95 ms baseline, 11.33 ms candidate |
| Context pool placement | Grow the candidate pool by 96 MiB to 2 GiB | Modes unchanged |

Device timestamps inside the candidate's graph locate the adder. It is spread evenly across
all 48 layers: per layer, about +4.4 µs in the mixer, +2.3 µs in MoE and +0.7 µs in the
single combine kernel, and only +1.4 µs in the 855 µs final stage. It follows kernel count
rather than kernel time, at 25% of the combine stage and 0.16% of the final stage. That
points to dispatch between graph nodes and agrees with the earlier finding that no kernel
became slower.

Graph structure changes how often the mode occurs. One extra one-thread kernel per layer
moved 15 of 20 P512 waves and 16 of 20 P2048 waves to the fast mode, so the instrumented
candidate ran at about 1,007 ms instead of 1,021 ms despite the added work. What selects the
mode remains unknown. The timestamps were taken on the candidate only. This is a real cost
to ordinary Flash decoding, not noise, and it is not repaired here. Flash MTP K=3 profiles
show no net cost: their wave changes are −0.86% to −0.03%. Physical RTX 5090 and Windows
behavior were not measured.

All three items were accepted for this merge with these findings on record. That covers
the source integration only; it is not release acceptance or RTX 5090 qualification.
