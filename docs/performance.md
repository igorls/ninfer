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
  [model cards](README.md#model-artifacts), with a [README summary](../README.md#evaluation).
- [Perplexity](perplexity.md): offline causal-scoring measurement and comparison rules.

Model pages are the detailed result authority. README and model-card performance tables are
excerpts linked to those pages; update them together when replacing an applicable measurement.

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
