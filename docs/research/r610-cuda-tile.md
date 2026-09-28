# R610 driver and CUDA Tile development evaluation

Research date: 2026-09-08. Updated: 2026-09-11 after the driver update and reboot.
Status: serving smoke checks, the first Tile experiment, and its kernel-level
diagnosis are complete. Correcting the Tile implementation and tuning its
pipeline makes it 2.68x faster than the initial candidate at 2,048 tokens, but
it remains 1.67x slower than the existing Op. No candidate is integrated.
Extended production stability and driver-only performance effects remain
unmeasured.

## Decision

Use 27B for the initial dense prefill experiment; keep Flash-Next as the
production model and primary optimization target. CUDA Tile C++ AOT and JIT
execution work on the updated machine, including a small CUDA Graph probe.
The complete FP8 GDN projection candidates are numerically acceptable at the
sampled shapes but slower than the existing implementation. The first version
omitted alignment information needed for efficient Tile memory operations;
its poor result was an implementation result, not a limit established for
Tile. Retain the existing production kernels while evaluating stronger Tile
designs. Neither a Tile engine speedup nor a driver-only speedup has been
established.

This document is the active research and experiment handoff. Once qualification
and the selected experiment finish, integrate useful conclusions into the
existing performance/development references and retire this temporary document.

## Local baseline observed

| Component | Observed version or identity |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| Driver | 596.86, R595 U7 |
| CUDA compiler | CUDA 13.3, NVCC V13.3.33 |
| Nsight Compute | 2026.2.0 installed |
| Nsight Systems | 2026.1.3 installed |

These are the pre-upgrade observations. On 2026-09-11, `nvidia-smi` reports
driver **616.92**, ECC **Enabled**, and 97,887 MiB of GPU memory. NVCC remains
13.3.33; Nsight Compute and Systems remain 2026.2.0 and 2026.1.3. The isolated
GDN probe uses the repository's MSVC 19.51 toolchain and `sm_120a`.

No matched pre-upgrade timing series was collected for these workloads. The
new driver and ECC setting therefore cannot be assigned a performance effect
from the measurements below.

## Findings and relevance

### RTX New Feature Branch

NVIDIA's driver listing showed RTX R610 U2, version 610.88, released July 30,
2026. NFB exposes features outside the Production Branch schedule; NVIDIA
recommends Production Branch for enterprise stability. The listing did not
establish an inference throughput improvement for NInfer.

Source: [NVIDIA driver listing](https://www.nvidia.com/Download/processFind.aspx?dtcid=1&lang=en-us&lid=1&osid=57).

### CUDA Tile C++ and compatibility

CUDA Tile C++ requires CUDA Toolkit 13.3 and a supported GPU. NVIDIA distinguishes
ahead-of-time compilation from JIT: its documented general driver floor is
R580, while JIT requires a driver at least as new as the toolkit's corresponding
driver branch, R610 for CUDA 13.3. Nsight Compute likewise requires R610 or newer
to profile Tile C++ JIT kernels.

Sources: [CUDA Tile C++ introduction and requirements](https://developer.nvidia.com/blog/develop-high-performance-gpu-kernels-in-cpp-with-nvidia-cuda-tile/),
[Nsight Compute release notes](https://docs.nvidia.com/nsight-compute/ReleaseNotes/).

The pre-upgrade 596.86 driver is not inherently incompatible with the installed
CUDA 13.3 compiler. CUDA 13.x minor-version compatibility supports drivers R580
and newer, with restrictions on newer driver-dependent features and PTX JIT.
Do not interpret R610's Tile JIT requirement as a requirement for every CUDA
13.3 binary or every Tile experiment.

Source: [CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

### CUDA Graph launch-queue fix: conditional relevance

R610 data-center notes document a fix for Xid 32 crashes with many chained CUDA
Graph nodes when `CUDA_SCALE_LAUNCH_QUEUES` is set to `2x` or `4x`. NInfer uses
graph instantiate, update, upload, and launch in
[`src/core/decode_graph.cpp`](../../src/core/decode_graph.cpp).
No reference to that environment setting was found in the source paths checked;
this does not establish the environment of a running engine.

The fix is documented for the data-center release. Its presence in the RTX
workstation package was not independently confirmed. It is relevant to future
launch-queue experiments, not evidence of a current NInfer defect or speedup.

### Blackwell TMA caveat

The same data-center notes retain a tensor-map encoding issue that can cause
illegal memory accesses when backing allocations are smaller than 128 KiB and
tensors are not dense and non-overlapping. NInfer uses `cuTensorMapEncodeTiled`
in [`nvfp4_w4a4_tma.cuh`](../../src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh).
The inspected descriptors appear contiguous; exposure was not established.
Revisit the issue if a candidate introduces small, strided TMA layouts. No
descriptor workaround was applied.

Source for the graph fix and TMA issue:
[R610 data-center release notes, Linux 610.57.04 / Windows 610.88](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-610-57-04/index.html).

### Profiler improvements are a separate upgrade

Nsight Compute 2026.2.1 adds dynamic CUDA injection support, process-ID attachment,
and fixes involving late injection and replay attachment. These could improve
profiling of a persistent engine, but require a separate tool update. Its
general driver requirement is CUDA 13 compatibility; do not describe these
attachment improvements as exclusive to R610 or assume unrestricted attachment
to an unprepared process.

Source: [Nsight Compute 2026.2.1 release notes and requirements](https://docs.nvidia.com/nsight-compute/ReleaseNotes/).

## First Tile experiment and result

Hypothesis: Tile C++ may make better tensor-core tiling and pipelining choices
easier to explore. It does not automatically outperform specialized CUDA C++ or
inline PTX. The following sequence was used for the first candidate:

1. Use relevant profiling evidence to select one material prefill matrix
   multiplication bottleneck. Prefer a representative 27B operation for the first
   experiment; inspect actual artifact formats and dimensions before choosing it.
   Do not select an operation merely because Tile makes it easy to express.
2. Implement the complete operation with the same represented inputs and public
   quantization semantics. Check that Tile supports the needed format and
   operation efficiently before investing in tuning. Keep the candidate local to
   the operation's implementation and benchmark ownership.
3. Validate directly against an independent FP32/FP64 mathematical oracle,
   including exact decoding of stored quantized weights. Use output tolerances
   appropriate to the arithmetic profile; existing-kernel parity alone is
   insufficient.
4. Measure real model shapes and relevant prefill sizes against the current
   kernel. Collect latency first; use registers, occupancy, memory traffic, and
   instruction evidence only to resolve tuning decisions.
5. Integrate a winning candidate and measure end-to-end prefill before claiming
   an engine improvement. If it loses, record the useful limitation and stop or
   revise only when evidence supports a specific alternative.

Flash-Next MoE is a possible later target. Expert workloads can be small and
irregular, so the useful tile and batching strategy depends on actual token
routing. A dense prefill win does not prove a MoE or single-token decode win.

Ahead-of-time compilation can be explored without waiting for R610, subject to
the documented supported compilation path. R610 enables the CUDA 13.3 Tile JIT
and associated profiling route.

### Compiler and runtime probe

A 10,003-element Tile addition kernel passes exact CPU-oracle checks over eight
eager launches and eight graph replays with changing inputs. Both `sm_120a` AOT
and `compute_120a` Tile JIT binaries pass. AOT still passes with
`CUDA_DISABLE_JIT=1`; the JIT-only binary then fails with the expected disabled
JIT error. This qualifies the small probe, not arbitrary Tile kernels or the GDN
candidate's graph behavior.

### Selection from actual 27B prefill

Nsight Systems captured the first 2,048-token `prefill` step of a roughly
6.8K-token text request on the installed 27B NVFP4 engine. The FP8 GDN input
projection, with weight shape `[16384, 5120]`, accounts for **32.685 ms / 20.5%**
of summed GPU kernel time in that step. The NVFP4 gate/up SwiGLU accounts for
21.7%. These percentages describe the captured step, not the entire request
or its wall time.

FP8 GDN was selected because it is material and the installed Tile C++ MMA API
supports FP8. The inspected API does not expose direct FP4/scaled MMA, making
the NVFP4 operation a less suitable first Tile candidate.

### Complete-operation comparison

The isolated candidate includes BF16 activation quantization to FP8, the FP8
contraction, stored row and token scales, and BF16 QKV/Z output writes. It uses
the same production activation quantizer. The reference independently evaluates
the full dot product in FP64 from represented BF16 activations and exactly
decoded stored weights, without copying the private activation quantization.

The fixture checks 31 rows in each of Q, K, V and Z at seven token positions
per shape, using the existing A8 criterion separately for each projection.
All tested variants pass these sampled checks, output guards/full-write checks,
and input/weight preservation. This is sampled operator evidence, not complete
domain qualification. The GDN candidate was tested eagerly only.

Timing includes the complete operation, with three warmups and the median of
nine CUDA-event samples. A 256 MiB flush precedes each measured interval. The
production engine is stopped for these measurements. Four Tile configurations
were tried; the best is `64x64x128` at every sampled token count:

| Prefill tokens | Existing Op, microseconds | Best Tile Op, microseconds | Tile / existing |
| ---: | ---: | ---: | ---: |
| 33 | 70.368 | 154.240 | 2.19x |
| 512 | 187.328 | 825.600 | 4.41x |
| 2,048 | 646.592 | 2,934.560 | 4.54x |

**Decision: reject the initial candidate for production integration.** This
comparison alone did not establish a fundamental Tile limitation. The following
diagnosis identifies implementation weaknesses and tests concrete revisions.

Local probe sources, serving harnesses and measurements are under
`profiles/bench/r610-tile-20260911/` (ignored local experiment artifacts).
Production dispatch, the installed executable and saved Supervisor model
configuration are unchanged.

### Kernel-level diagnosis and controlled revisions

Nsight Compute captured the production Op and the original, aligned, and
aligned/structured-loop Tile candidates at 2,048 tokens. Each capture includes
the same activation quantization stage and one matrix kernel after warmup.
Counter access required an elevated profiler process; no persistent driver
permission setting was changed. Clocks were not locked, so these captures are
used for attribution; the comparison below uses separate CUDA-event timings.

The original Tile matrix kernel uses native FP8 tensor-core instructions. The
problem is not a fallback to scalar floating-point matrix multiplication.
However, its generated code contains byte-at-a-time global loads and extensive
byte rearrangement. It uses 214 registers per thread, versus 94 for the
production kernel. Nsight reports no local-memory spills for this candidate;
spilling does not explain its loss. Achieved occupancy is 16.48%, versus 32.76%
for production. Long-scoreboard waits for L1TEX operations occupy about 52.5%
of the original candidate's average interval between issued warp instructions.

Adding `ct::assume_aligned(..., 16_ic)` to the activation-code and weight-code
pointers enables `UTMALDG.2D` TMA loads in the generated code. Host checks verify
both pointer preconditions. Register use falls to 124. Occupancy remains about
16.4%, limited by shared memory, but the complete operation becomes 2.10x
faster. This isolates missing alignment information as a material defect in
the first implementation.

Changing only the loop to `ct::irange` then increases the generated shared
allocation from 50,476 to 67,964 bytes. Nsight confirms that the shared-memory
block limit falls from two to one and achieved occupancy falls to 8.30%.
The deeper generated pipeline is slower for this shape. Setting `latency=1`
on its two input loads reduces the allocation to 43,332 bytes and improves
timing. Adding an `occupancy=3` kernel hint reduces it further to 26,948 bytes
and yields the best tested large-prefill candidate. The last candidate's
resource allocation was inspected statically; its achieved occupancy was not
separately profiled.

These are documented Tile controls, not changes to the mathematical operation.
See [NVIDIA's Tile performance guidance](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html#c-performance-tips)
and [optimization hints](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html#optimization-hints).

The final comparison retains the same represented inputs, independent FP64
oracle, per-Q/K/V/Z A8 criteria, guard/preservation checks, and complete-operation
timing method. All 27 variant/shape cases pass at T=33, 512 and 2,048.

| Implementation at T=2,048 | Complete Op, microseconds |
| --- | ---: |
| Existing production Op | 658.208 |
| Original Tile, 64x64x128 | 2,938.752 |
| Add input alignment | 1,399.520 |
| Also use structured loop, default hints | 2,184.896 |
| Structured loop with load latency=1 | 1,223.264 |
| Also request occupancy=3 | **1,096.928** |
| Disable TMA on aligned loads, latency=1 | 1,193.056 |

The wider 64x128x128 aligned tile takes 1,886.912 microseconds. Reducing K to 64
with the structured/latency=1 variant takes 1,622.752 microseconds. Neither
improves on the selected candidate. Disabling TMA produces vectorized
`LDG.E.128` loads and is competitive with default-occupancy TMA, but also loses
to the selected candidate. TMA itself is therefore not a universal explanation
for which implementation wins.

At T=512, the selected occupancy=3 candidate takes 326.400 microseconds versus
189.216 for production; at T=33 it takes 99.040 versus 70.240. The best T=33
Tile variant instead disables TMA and takes 96.992 microseconds. There is no
tested shape where a Tile candidate beats the existing Op.

**Conclusion:** the initial kernel's implementation and compiler guidance
explain a substantial part of its loss. Targeted corrections make its
2,048-token operation 2.68x faster, reducing the gap to production from 4.46x
to 1.67x in the final comparison. This is a meaningful Tile prototype
improvement, not an engine improvement. The remaining gap has not been fully
attributed, and the tested controls do not establish Tile's performance ceiling.
A subsequent candidate needs a different work distribution or memory/epilogue
layout with evidence that it can close the remaining gap; further arbitrary
tile-size changes are not justified by these results. Production remains on
the existing kernels.

The final local executable is `gdn_tile_final_probe.exe`; run it through
`run_tile_comparison.ps1 -Probe gdn_tile_final_probe.exe` in the experiment
directory to include the production stop/restore window. `gdn-final-results.log`
contains the comparison; `ncu-*.ncu-rep` and the SASS/resource dumps support
the diagnosis. No GDN graph or end-to-end Tile qualification was performed.

## Serving smoke checks after the update

The installed serving binary completed 24 requests on 27B: eight each with
ordinary decoding, MTP5, and DFlash2 K7. Each mode ran two repetitions at roughly
2K and 6.8K input tokens, with a fresh root and a followup. Actual request events
confirm zero cached root tokens, positive followup reuse, and the intended
speculative backend. Requests used greedy non-thinking decoding, neutral
penalties, a 256-token output budget, FP8 KV, concurrency one and 2,048-token
prefill chunks.

Flash-Next completed the same eight-request smoke sequence with its resident
MTP4 production configuration. Those samples use different serving capacity and
prefill settings and are not a controlled comparison against 27B. Unique root
prefixes also differ between runs; the measurements are operational baselines,
not strict speculative-output parity or model-quality evaluations.

The 27B test server and standalone GPU probes run within Supervisor-controlled
stop/start windows. Flash-Next is restored afterward and checked through the
engine's direct `/health` endpoint. These short text checks do not qualify long
contexts, concurrency, Vision, long-running stability, or a driver-only speedup.

Remaining questions are the driver-only performance effect, broader production
stability, workstation applicability of the documented data-center graph fix,
and whether a stronger Tile execution layout can close the remaining Op gap.
The initial candidate, its diagnosis, the controlled revisions, and the decision
to retain the existing production kernel are complete.
