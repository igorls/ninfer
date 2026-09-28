# RTX PRO 6000 workstation records (fork)

These results were measured on the fork's pre-v3 engine (`research/qwen4-flash-next`) on its
RTX PRO 6000 Blackwell 96 GB workstation, beside other desktop GPU work. They are historical
records, not v3 measurements, and they do not follow every rule of the
[publication methodology](methodology.md). Each section states its own conditions. Qwen3.8-Flash-Next
records stay with that line until its port.

## NVFP4 W4A4 prefill schedule (RTX 5090 record)

On the fork's pre-v3 engine, the desktop RTX 5090 selected token-fast CTA rasterization and one
activation-scale TMA fetch per K-tile pair for NVFP4 Linear and LinearSwiGLU, while the RTX PRO
6000 kept weight-fast rasterization and per-tile scale fetches: the combined change had regressed
its measured 7,680-token prefill from 741.4 to 749.9 ms. The v3 base makes token-fast rasterization
unconditional (upstream `ee9d5192`) and fetches activation scales one tile per TMA request
(`1d8587bc`); the RTX PRO 6000 decision is re-measured on v3.

[Issue #22](https://github.com/igorls/ninfer/issues/22#issuecomment-5741594802) and
its [raw reports and reproduction scripts](https://gist.github.com/patrickscd/a1f67f1b693962d6cd6a826748cc24a3)
record the September 19, 2026 RTX 5090 A/B: baseline `5e4a66d0` versus that baseline
plus the two-file patch from `1acfff7d`. Five alternating fresh-process pairs per
mode, one discarded warmup per prompt, and 256 timed decode tokens used
Qwen3.8-27B NVFP4, 32,768 context/KV capacity, FP8 KV, 1,024-token prefill chunks,
CUDA Graphs, disabled prefix reuse, and a 2 GiB desktop reserve. Windows Ninja
Release used nvcc 13.3.33, MSVC 14.50.35503, driver 616.92 and a 480 W power limit.
Desktop applications remained open.

| Prompt tokens | Prefill gain, MTP0 | Prefill gain, MTP3 |
|---:|---:|---:|
| 512 | 4.13% | 4.35% |
| 4,096 | 5.83% | 5.21% |
| 8,192 | 5.83% | 5.72% |
| 16,384 | 5.12% | 5.24% |

Mean decode differences ranged from -0.31% to +0.26%. The submitted Linear A4
and LinearSwiGLU numerical checks passed for both arms; a fixed greedy 512-token
prompt produced identical text and token IDs across arms in each speculation mode.
These gains apply to the measured workload, not all chunk sizes or other GPUs.
The optional 35B MoE cache-hint measurement was waived because the contributor
does not run that artifact.

## Qwen3.8-27B DFlash2 on RTX PRO 6000

Native Windows qualification on September 7–8, 2026 uses the RTX PRO 6000 Blackwell 96 GB,
MSVC 19.51, CUDA 13.3.33 and `sm_120a`. The target is the existing Qwen3.8-27B NVFP4 artifact
with every base payload preserved, extended with the
[incoai DFlash2 companion](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2).
v3 carries upstream's DFlash2 integration; see [DFlash](../maintainer/dflash.md) and
[weight conversion](../weight-conversion.md).

The matched serving campaign uses one final Release executable, FP8 KV, CUDA graphs, capacity
eight, a 32,768-token context limit and KV pool, 2,048-token prefill chunks, and no prefix reuse
or Vision allocation. Each process receives a 256-token warmup before two repetitions. The
Python and translation fixtures use non-thinking mode; the AIME fixture uses `xhigh` reasoning.
Sampling is greedy with no presence/frequency penalty and a 1,024-token output budget. Values
are committed output tokens divided by complete client wall time, including prefill. At eight
active requests, the numerator sums the wave and the denominator is its makespan. This is not
a steady-state decode-only measurement. Naturally completed translations remain valid samples.

| Active requests | Workload | Ordinary tok/s | MTP5 tok/s | DFlash2 K7 tok/s |
|---:|---|---:|---:|---:|
| 1 | Python | 64.7 | 183.7 | **195.6** |
| 1 | Translation | 64.5 | 169.0 | **175.7** |
| 1 | Mathematics reasoning | 65.1 | 159.4 | **187.0** |
| 8 | Python | 425.3 | **1,016.3** | 987.4 |
| 8 | Translation | 389.5 | **896.5** | 821.7 |
| 8 | Mathematics reasoning | 423.2 | 827.5 | **839.3** |

These are two-repetition means, not confidence intervals. DFlash2 improves the single-request
wall rate by 4–17% over MTP5 in these cases, and by 2.7–3.0x over ordinary decoding. At eight
requests, MTP5 leads on Python and translation; the mathematics difference is small. Start with
`--spec dflash2 --draft-tokens 7 --lm-head-draft` for interactive decode, and MTP5 with the
optimized head for concurrent Python/translation throughput. Backend selection remains fixed
at Engine startup; these observations do not establish a universal winner at other contexts.

Strict JSON reaches 64.0 / 61.2 / 61.9 tok/s with ordinary / MTP5 / DFlash2 at one active request,
and 383.8 / 374.9 / 377.5 aggregate tok/s at eight. Every result passes the schema validator;
the single-request JSON text is identical across the three modes. In a separate paired
capacity-one experiment, replacing padded zero-draft verification with the width-one target
route improved DFlash2 JSON from 52.9 to 63.0 tok/s, about 19%, without a clear unconstrained
throughput regression. Backend state maintenance still has a small cost.

For the 7,680-token long-context execution probe, mean prefill / server TTFT is 764.8 / 770.0 ms
with ordinary decoding, 811.6 / 816.5 ms with MTP5, and 812.6 / 817.6 ms with DFlash2. DFlash2
does not improve prefill in this comparison; its benefit is generated-token decode.

| Startup mode, capacity eight | Materialized weights GiB | Runtime reservation GiB |
|---|---:|---:|
| Ordinary | 18.976 | 2.578 |
| MTP5, optimized head | 19.729 | 3.390 |
| DFlash2 K7, optimized head | 21.383 | 6.128 |

The runtime reservation includes the KV pool, state/workspace and unallocated CUDA-graph
headroom. These values describe this 32K configuration; they are not measured peak process VRAM.

The fixtures are `scenario_code_python`, `scenario_translation_markdown`, and
`long_decode_aime26_15` under `examples/cli/messages/`. A separate strict JSON-schema case asks
for twelve job-runner implementation steps and is independently validated. The `long_niah_8k`
fixture exercises the longer attention path; its short answer and explicit answer in the prompt
make it an execution/TTFT probe, not retrieval-quality evidence.

An earlier capacity-one sweep tested draft counts 2, 3, 5, 7, 11 and 15 and both proposal-head
routes. K=7 with the optimized head gave the best observed DFlash2 results for these workloads.
Larger draft blocks did not compensate for their additional verification work. Keep that sweep
separate from the final capacity-eight comparison.

Profiling a K=7, capacity-one native benchmark with 512 prompt tokens and 256 output tokens
shows 2,038 ms of kernel execution inside a 2,131 ms decode range. Of summed kernel time,
89.8% is target verification and 8.5% is drafting plus selection. The largest contributors
are target NVFP4 gate/up (455 ms), fused FP8 GDN input (377 ms), and NVFP4 down (288 ms).
These measurements identify target verification as the next optimization focus; they do not
measure hardware occupancy or bandwidth saturation. A forced A8 GDN input route was slower,
and materialized A16 did not improve its fused baseline; both experiments were reverted.
Nsight Compute counters were unavailable (`ERR_NVGPUCTRPERM`).

Qualification passes the independent numerical Op checks, variable-width sparse acceptance,
graph and eager/full-head execution, K=7/K=15 with eight requests, partial terminal blocks,
cancellation, page/ring boundaries, Host restore, and image/video input. A reproduced concurrent
admission failure now waits for an unfinished StateImage fork to settle; the CPU regression and
repeated real pressure runs pass. MTP5 and DFlash2 K7 each pass all 21 live structured-output
protocol checks. All-constrained batches use width-one target execution while maintaining the
selected backend state; mixed batches retain masked speculative execution.

Actual response text was read manually. Translations retain the required table, code and
identifiers, with prose differences; Python and mathematics hit their output budgets and cannot
establish completed-code or final-answer quality. The JSON plans are valid and relevant to the
request. Greedy text can differ across verification widths because qualified FP8/NVFP4 arithmetic
routes differ. This integration is not advertised as bitwise lossless, and these performance
fixtures do not establish Tribuno legal-workflow acceptance. The real checkpoint qualification
here is NVFP4; it does not qualify every supported weight profile or production context length.

Local request text, timings and counters were kept under `profiles/bench/dflash2-20260907/final-*`
and the phase breakdown under `profiles/nsys/dflash2-20260907/` on the workstation.

## OrcaRouter NVFP4 integration probe

On 2026-09-12, the fork's v2 `qwen3.8-27b-orcarouter/nvfp4` artifact was exercised
on RTX PRO 6000 Blackwell with ECC enabled, driver 616.92, CUDA 13.3 and a native Windows Release
build. It preserves the pinned OrcaRouter source's BF16 embeddings and output head. These are
initial integration measurements, not a benchmark suite or an Unsloth Studio comparison.

Each mode received the same 169-token Python queue-repair prompt, greedy non-thinking sampling,
neutral penalties and a 1536-token output budget. Startup used four active lanes, a 32K per-request
limit, 64K shared FP8 KV, 2048-token prefill chunks, CUDA graphs and Vision enabled. The timed
coding request ran alone; separate checks exercised two concurrent requests. Responses terminated
naturally and differed in content and length, so elapsed-time ratios are not fixed-output speedups.
Decode rates below use engine-reported decode time and exclude the first output token.

| Backend | Completion tokens | Decode tok/s | Accepted / drafted | Generated tests passing |
|---|---:|---:|---:|---:|
| Ordinary | 1296 | 70.8 | — | 1/2 |
| MTP K5, optimized head | 1465 | 196.0 | 1068/1985 (53.8%) | 2/2 |
| DFlash2 K7, optimized head | 1310 | 208.8 | 975/2338 (41.7%) | 2/2 |
| DFlash2 K7, full BF16 head | 1295 | 206.4 | 975/2240 (43.5%) | 1/2 |

Actual answers were reviewed and their Python unittest examples executed with Python 3.14. The
ordinary response incorrectly calls `Task.exception()` expecting a returned `CancelledError`;
that method raises it. The full-head DFlash2 response references a nonexistent public
`Queue.unfinished_tasks` attribute. The optimized DFlash2 response's own tests pass, but its
worker re-raises a job failure and dies, so remaining queued jobs can still make `join()` hang.
Its passing tests therefore do not establish a successful production repair. These observations
are too small a sample to rank model quality or attribute errors to speculation.

All four modes passed ordinary Chat Completions, constrained JSON Schema,
low-thinking output and two-request concurrency checks. Named tool calls also passed in the
installed build, which included a separate, pending forced-tool implementation; that observation
does not qualify forced-tool support in the standalone OrcaRouter change. The installed ordinary-decoding route
also correctly read a synthetic image's counts/shapes, streamed SSE through `[DONE]`, and reused
2764 tokens through a Responses `previous_response_id` continuation. Source-specific tokenizer
checks cover 17 independent reference cases; BF16 Linear and LinearTopK pass their independent
numerical oracles, including the existing FP8 top-k regression. These establish integration,
not long-horizon coding quality. The launcher defaults this derivative to ordinary decoding and
exposes both speculative backends for deliberate testing.
