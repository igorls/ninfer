# RTX PRO 6000 workstation records (fork)

These results were measured on the fork's RTX PRO 6000 Blackwell 96 GB workstation and, where a
section says so, on Colab G4 VMs with the same GPU in its Server Edition. The first section compares
weight profiles; the next three compare the fork line with the previous release build; the others are records of the fork's
pre-v3 engine (`research/qwen4-flash-next`), taken beside other desktop GPU work. They do not
follow every rule of the [publication methodology](methodology.md); each section states its own
conditions. Qwen3.8-Flash-Next records stay with that line until its port.

## Qwen3.8-27B `nvfp4full` against the production profile (2026-09-28)

Candidate: the [`qwen3_8_27b_nvfp4full`](../weight-conversion.md#qwen38-27b-nvfp4full) profile
(recipe adapted from cometkim/ninfer), converted locally with the `broad-v1` activation calibration
and a Q8 DFlash2 companion. Current: the production artifact
`qwen3_8_27b_nvfp4_dflash2.ninfer` (Unsloth mixed NVFP4/FP8). Both arms ran the same build of the
task branch. Clean measurements ran on Colab G4 VMs (RTX PRO 6000 Blackwell Server Edition, driver
580.82, Linux build with CUDA 13.3); the G4 copy of the production weights is
`neroued/Qwen3.8-27B-nvfp4-NInfer` at `f0b43ad4`, whose 1,072 bound objects and Use divisors are
byte-identical to the local production artifact's (only the embedded chat template's line endings
differ). The Colab-converted candidate is object-for-object identical to the local conversion.

### Quality: distribution agreement with BF16

Criterion, written before any artifact was scored on the full corpus: a candidate passes when its
mean KL(BF16 || candidate) is at most 1.25 times production's, its top-1 agreement with BF16 is at
most 1.0 point below production's, and its MTP acceptance is at most 2.0 points below.

`ninfer-perplexity --reference` on `kld-400k-v1` (399,892 scored positions, FP8 KV,
[method](../perplexity.md#distribution-agreement-with-a-reference-model)). The G4 and the
workstation produce identical numbers for the same artifact.

| Profile | Text weights | KL mean | KL p50 | KL p99 | Top-1 | NLL - BF16 | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| Production `nvfp4` | 18.98 GiB | 0.0611 | 0.00616 | 0.976 | 93.10% | +0.0123 | reference |
| `nvfp4full`, `broad-v1` calibration | 16.03 GiB | 0.0954 (1.56x) | 0.01333 | 1.550 | 90.30% | +0.0208 | fails |
| `nvfp4full`, `cometkim-v1` calibration | 16.03 GiB | 0.0993 (1.63x) | 0.01359 | 1.628 | 90.30% | +0.0174 | fails |
| Variant: NVFP4 on MLP 56-63 only | 18.25 GiB | 0.0668 (1.09x) | 0.00823 | 0.999 | 92.29% | +0.0330 | passes |
| Variant: NVFP4 on MLP 56-63 and attention | 17.8 GiB | 0.0739 (1.21x) | 0.00972 | 1.131 | 91.74% | +0.0439 | fails (top-1) |
| Variant: NVFP4 on GDN layers 8-55 only | 17.43 GiB | 0.0759 (1.24x) | 0.00806 | 1.256 | 92.03% | +0.0105 | fails (top-1 by 0.07) |

The variants keep everything else of the production allocation (imported FP8), take the Q8
vocabulary endpoints and BF16-checkpoint norms of `nvfp4full`, and were made with recipe overrides.
Per domain, `nvfp4full` (`broad-v1`) roughly doubles mean KL on C++ code, the Belebele languages and
English prose (2.0-2.1x) and raises it least on UltraChat (1.35x) and MATH-500 (1.40x); top-1
agreement drops 1.2 (MATH-500) to 5.0 (UltraChat) points per domain.

Attribution on an 18-sequence development subset (69,144 positions; production 0.0721 mean KL,
92.98% top-1): the Q8 endpoints and exact norms alone improve on production (0.0636, 93.24%); NVFP4
on GDN alone gives 0.0975 and 91.58%, on MLP 56-63 alone 0.0708 and 92.08%, on attention alone
0.0699 and 92.62%. Keeping every locally encoded site except MLP gate/up at A16 (weight-only NVFP4;
the fused SwiGLU has no A16 route beyond 16 tokens) gives 0.0849 and 91.48% against the full
profile's 0.1006 and 90.55%: activation quantization accounts for about half of the KL increase and
two fifths of the top-1 loss, weight rounding for the rest. The calibration corpus changes little:
`broad-v1` lowers mean KL 4% against `cometkim-v1`.

### MTP acceptance

24 prompts (MATH-500 rows 0-11 and UltraChat `test_gen` rows 0-11), thinking on, temperature 0,
512 output tokens, one request at a time, production flags with MTP5 and the proposal head. The G4
and workstation runs of the same artifact generated identical tokens; the variants ran on the workstation.

| Profile | Completion tokens | Acceptance | Tokens per round | Accepted per draft position |
|---|---:|---:|---:|---|
| Production | 9,825 | 43.3% | 3.16 | 2342 / 1671 / 1177 / 870 / 647 |
| `nvfp4full`, `broad-v1` | 9,639 | 45.3% | 3.25 | 2270 / 1657 / 1197 / 885 / 674 |
| `nvfp4full`, `cometkim-v1` | 9,754 | 45.1% | 3.25 | 2272 / 1664 / 1213 / 911 / 695 |
| Variant: MLP 56-63 only | 9,670 | 45.2% | 3.25 | 2296 / 1649 / 1197 / 885 / 670 |
| Variant: GDN layers 8-55 only | 9,728 | 43.9% | 3.19 | 2347 / 1641 / 1177 / 874 / 637 |

No profile's MTP acceptance is lower than production's.

### GPQA-Diamond, paired (supporting evidence)

Production and `nvfp4full` (`broad-v1`) served side by side on one G4 (MTP5, FP8 KV, 8 concurrent
requests each), EvalScope 1.10.0 through `eval/ninfer_eval`, 0-shot rule scoring, thinking on with
`reasoning_effort: low`, temperature 1.0, top_p 0.95, top_k 20, 16,384 output tokens, one seed per
VM. The low effort and output bound keep a seed inside a Colab VM's lifetime; they lower both arms'
scores, so these are paired comparisons, not published-style GPQA numbers.

| Seed | Paired questions | Production | `nvfp4full` | Only production correct | Only `nvfp4full` correct | Difference |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 197 | 78.68% | 78.17% | 14 | 13 | -0.51 pts |
| 2 (VM reclaimed mid-run) | 61 | 88.52% | 90.91%* | 2 | 3 | +1.64 pts |
| 3 | 197 | 82.23% | 81.73% | 11 | 10 | -0.51 pts |
| Pooled | 455 answers | | | 27 | 26 | -0.22 pts, 95% CI [-3.52, +3.10] |

\* Over its 66 answered questions; the difference uses the 61 both arms answered. One question per
complete seed returned no scored answer. The interval is a question-level bootstrap over the pooled
paired answers.

The paired difference is inside the noise; GPQA at this size cannot resolve the token-level shift
the KL gate measures.

### Speed and memory

`tools/bench/serve_ab_probe.py`: a fresh `ninfer-serve` per arm and round with the production flags
except the KV capacity (`--max-context 131072 --kv-capacity 131072 --max-concurrency 8
--prefill-chunk 2048 --kv-dtype fp8 --vision --spec mtp --draft-tokens 5 --lm-head-draft
--desktop-reserve-gib 6`), alternating arm order; per arm and round one warmup, five cold
7,680-token prefills (a `long_64k_independent` prefix trimmed to exactly 7,680 prompt tokens, a
fresh salt per request, read-only cache, one output token) and five greedy 256-token counting
decodes (thinking off). Times are the server's request-log timings; decode excludes the first token.
Weights are the loaded weight arena (Text, MTP, Vision and proposal head).

| Machine | Arm | Samples | 7,680-token prefill ms, median [min-max] | Decode tok/s, median [min-max] | MTP acceptance | Weights | Device free |
|---|---|---:|---:|---:|---:|---:|---:|
| G4, 5 rounds | production | 25 | 574.5 [571.4-576.7] | 226.9 [226.8-227.1] | 67.6% | 20.00 GiB | 66.84 GiB |
| G4, 5 rounds | `nvfp4full` | 25 | 444.6 [444.2-445.1] | 248.4 [248.2-248.5] | 66.1% | 17.05 GiB | 69.79 GiB |
| G4, 5 rounds | NVFP4 on MLP 56-63 only | 25 | 537.8 [535.4-540.0] | 233.6 [233.5-233.8] | 67.6% | 19.28 GiB | 67.56 GiB |
| Workstation, 2 rounds | production | 10 | 651.4 [642.3-660.9] | 233.9 [231.0-236.5] | 67.6% | 20.00 GiB | 12.95 GiB |
| Workstation, 2 rounds | `nvfp4full` | 10 | 483.2 [476.5-487.3] | 264.3 [263.6-269.4] | 69.1% | 17.05 GiB | 15.90 GiB |

The G4 `nvfp4full` arm used the `cometkim-v1` conversion (same weights, different activation
divisors); the workstation arm the `broad-v1` one. The workstation rounds ran on Windows with driver 616.92.
`nvfp4full` prefills 22.6% (G4) and 25.8% (workstation) faster and decodes 9.5% and 13.0% faster, with 2.95
GiB more free device memory.

### RTX 5090 projection

The engine's own sizing on a G4, with the desktop reserve raised by the capacity difference so that
it sees what a 32,607 MiB RTX 5090 with a 1 GiB desktop reserve leaves (`--desktop-reserve-mib
66304`), `--kv-capacity auto`, FP8 KV, 4 concurrent requests. "Contexts" is the automatic KV capacity
divided by the context length; "free" is the emulated 5090's free memory after startup.

| Profile | Context | Configuration | Weights | Automatic KV tokens | Contexts | Free |
|---|---:|---|---:|---:|---:|---:|
| Production | 65,536 | MTP, Vision | 20.00 GiB | 192,960 | 2.94 | 2.29 GiB |
| Production | 65,536 | DFlash2, Vision | 21.33 GiB | 97,664 | 1.49 | 3.80 GiB |
| Production | 131,072 | MTP, Vision | 20.00 GiB | 192,960 | 1.47 | 2.29 GiB |
| Production | 131,072 | DFlash2, with or without Vision | - | does not start | - | - |
| `nvfp4full` | 65,536 | MTP, Vision | 17.05 GiB | 262,144 (4 x 65,536 cap) | 4.00 | 2.98 GiB |
| `nvfp4full` | 65,536 | DFlash2, Vision | 18.37 GiB | 193,600 | 2.95 | 3.80 GiB |
| `nvfp4full` | 131,072 | MTP, Vision | 17.05 GiB | 283,264 | 2.16 | 2.29 GiB |
| `nvfp4full` | 131,072 | DFlash2, Vision | 18.37 GiB | 193,600 | 1.48 | 3.80 GiB |
| MLP 56-63 only | 65,536 | MTP, Vision | 19.28 GiB | 215,104 | 3.28 | 2.29 GiB |
| MLP 56-63 only | 131,072 | DFlash2, Vision | - | does not start | - | - |

The projection does not model a
consumer card's WDDM or CUDA-context differences.

## Upstream `d44ab584` merge against the previous release build (2026-09-29, Colab G4)

All measurements in this section ran on Colab G4 VMs (RTX PRO 6000 Blackwell Server Edition,
Linux, CUDA 13.3), not on the workstation. Each arm is a Linux build of the fork: the previous
release build, source `28c40898` (`2026.09.29-v3port.3`), the merge `183cdca6` (upstream `d44ab584` merged
in `d29866c9`, plus the MSVC-only descriptor alignment that leaves Linux code unchanged), and
variants built on the VM by merging an intermediate upstream commit into `28c40898` or reverting
commits from `183cdca6`. The weights are the production artifact from
`neroued/Qwen3.8-27B-nvfp4-NInfer@f0b43ad4`, object-identical to the workstation's copy.

The harness (`vm_ab.py`, built on `tools/bench/serve_ab_probe.py`) starts a fresh `ninfer-serve` per
arm and round and alternates the arm order each round. Per arm and round: one warmup of each kind,
five cold 7,680-token prefills (a `long_64k_independent` prefix trimmed to exactly 7,680 prompt
tokens, a fresh salt per request, read-only cache, one output token), five greedy 256-token
counting decodes (thinking off), and long-context requests (the first 262,000 characters of the same
fixture plus a counting instruction: 61,625 prompt tokens, fresh salt, 128 greedy output tokens).
Times are the server's request-log timings; decode excludes the first token. Production flags:
`--max-context 131072 --kv-capacity 524288 --max-concurrency 8 --prefill-chunk 2048 --kv-dtype fp8
--vision --spec mtp --draft-tokens 5 --lm-head-draft --desktop-reserve-gib 6`.

### Merge against the previous release build (6 rounds, 30 short and 18 long samples per arm)

| Arm | 7,680-token prefill ms, median [min–max] | 256-token decode tok/s | MTP acceptance | 61,625-token prefill ms | 61,625-token context decode tok/s |
|---|---:|---:|---:|---:|---:|
| previous release `28c40898` | 585.8 [578.9–588.3] | 226.8 [226.6–227.0] | 67.6% | 7,264.8 [7,226.7–7,273.2] | 272.8 [206.8–272.9] |
| merge `183cdca6` | 524.9 [518.1–526.7] | 217.7 [217.4–217.9] | 69.1% | 5,834.9 [5,738.7–5,843.9] | 261.7 [191.4–261.9] |

The merge prefills 10.4% faster at 7,680 tokens and 19.7% faster at 61,625 tokens, and decodes 4.0%
slower at short context and 4.1% slower at long context (median). Device free memory after startup
is identical to the byte in both arms (53.99 GiB): at `--prefill-chunk 2048` the larger split-KV
prefill workspace of `7f6aafed` does not change the allocation. (The `prefix_real` attention
scenario, a different configuration, reports a workspace peak of 255 MB for FP8 and K8V4 KV
against 130 MB for BF16, INT8 and NVFP4.) Long-context decode samples fall into modes set by the
MTP acceptance of each salted request (the low values in every arm are requests with about 65%
acceptance instead of 95–99%), so the long-context medians carry the comparison, not the ranges.

### Attribution (4 rounds, 20 short samples per arm)

| Arm | 7,680-token prefill ms | 256-token decode tok/s |
|---|---:|---:|
| previous release `28c40898` | 585.7 [572.2–587.2] | 226.8 [226.5–227.0] |
| `28c40898` + upstream through `a012e2bc` (causal attention series) | 585.5 [577.7–587.2] | 230.1 [229.8–230.2] |
| `28c40898` + upstream through `40bfe7dc` (plus the FP8 Linear series) | 518.4 [516.4–519.9] | 226.6 [226.4–226.8] |
| merge without `84cf93e4` and `1cfdb4d6` | 525.7 [523.4–527.1] | 230.5 [230.4–230.8] |
| merge | 525.4 [523.6–526.8] | 217.8 [217.5–217.9] |

A second VM separated the two codec commits (4 rounds, 20 samples per arm):

| Arm | 7,680-token prefill ms | 256-token decode tok/s | 61,625-token prefill ms |
|---|---:|---:|---:|
| previous release `28c40898` | 587.9 [582.7–588.9] | 226.9 [226.4–227.1] | 7,277.8 |
| merge | 526.4 [524.8–527.3] | 217.7 [217.3–217.8] | 5,850.3 |
| merge without `1cfdb4d6` (native NVFP4 A16 decoding) | 526.2 [524.6–527.2] | 230.9 [230.8–231.1] | 5,857.8 |
| merge without `84cf93e4` (native FP8-to-BF16 conversion) | 526.2 [524.5–527.2] | 217.5 [217.4–217.7] | 5,854.2 |
| merge without both | 526.0 [524.4–527.1] | 230.5 [230.1–230.8] | 5,853.1 |

- The causal attention series speeds up decode by 1.5% and leaves prefill unchanged at these
  lengths. The FP8 Linear series brings the 7,680-token prefill gain (−11.5%) and costs about
  1.5% of decode. Split-KV FP8 prefill (`7f6aafed`) brings most of the long-context prefill gain
  (6,735 ms without it, 5,835 ms with it).
- The native NVFP4 A16 decoding of `1cfdb4d6` alone costs 5.7% of decode and nothing on prefill.
  The fork line reverts it (`5482fd99`); the tip is code-identical on Linux to the "merge without
  `1cfdb4d6`" arm: **prefill −10.5% at 7,680 tokens and −19.5% at 61,625 tokens, decode +1.8%**
  against the previous release build, and 276.2 against 272.7 tok/s at 61,625 tokens of context in
  matched high-acceptance samples. `84cf93e4` is neutral and stays.
- Both codec paths are exact: System One on a build without either codec gives the same answers
  and probabilities as the merge on all 390 questions of a synthetic System One set.

### `nvfp4full` configuration (6 rounds, 30 short and 18 long samples per arm)

The `nvfp4full` profile (converted on the G4 with the `workstation` recipe and the committed
`broad-v1` calibration; 17.05 GiB of weights), NVFP4 KV, MTP3. Flags: `--max-context 65536
--kv-capacity 262144 --kv-dtype nvfp4 --spec mtp --draft-tokens 3`, with the others as above.
`--max-concurrency 8` and `--lm-head-draft` are assumptions, and the G4 is not an RTX 5090. The long-context request is 240,000 characters (56,449
prompt tokens) to fit the 65,536-token context.

| Arm | 7,680-token prefill ms | 256-token decode tok/s | MTP acceptance | 56,449-token prefill ms | Device free |
|---|---:|---:|---:|---:|---:|
| previous release `28c40898` | 456.8 [454.4–457.2] | 232.6 [232.3–233.1] | 85.9% | 6,710.1 | 69.33 GiB |
| merge | 456.7 [453.4–457.2] | 232.2 [231.7–232.6] | 85.9% | 6,710.1 | 69.33 GiB |
| merge without both codec commits | 456.8 [455.0–457.4] | 231.9 [231.6–232.1] | 85.9% | 6,709.9 | 69.33 GiB |

The merge is neutral here: `nvfp4full` runs almost no FP8 Linear, split-KV prefill covers FP8 and
K8V4 KV only, and the codec revert changes nothing in this configuration. Device memory is
identical to the byte. At matched full acceptance, long-context decode is 241.1–241.6 tok/s in
every arm.

## Upstream `e31bc99b` merge against the previous release build (2026-09-28)

Measured on the RTX PRO 6000 Blackwell (driver 616.92). Arms: the previous release build
`2026.09.28-v3port.1` (source `e92c2078`) and the
`workstation` build at `267a201a` (upstream `e31bc99b` merged, with the fork's follow-ups), both
with the v3 production artifact and blocking CUDA sync. Each arm is a fresh `ninfer-serve` with the
production flags except the KV capacity: `--max-context 131072 --kv-capacity 131072
--max-concurrency 8 --prefill-chunk 2048 --kv-dtype fp8 --vision --spec mtp --draft-tokens 5
--lm-head-draft --desktop-reserve-gib 6`. The probe is the one of the next section (one warmup,
five cold 7,680-token prefills, five greedy 256-token MTP decodes per arm and round). Three sets of
four rounds alternate the arm order (ABBA). A round is clean when no other engine request
completed on the GPU during its window; such traffic reached six of the 24 rounds.

| Rounds | Arm | Rounds | 7,680-token prefill ms, median [min–max] | Decode tok/s, median [min–max] | MTP acceptance |
|---|---|---:|---:|---:|---:|
| All clean | release | 9 | 689.7 [661.0–794.0] | 148.5 [141.8–151.7] | 36.4–37.0% |
| All clean | merged | 9 | 663.4 [651.1–694.0] | 149.1 [139.5–152.6] | 36.5–37.0% |
| Set 1 (all clean) | release | 4 | 689.2 [661.0–794.0] | 148.3 [143.7–151.6] | 36.6–37.0% |
| Set 1 (all clean) | merged | 4 | 666.5 [651.1–694.0] | 145.7 [139.5–151.9] | 36.6–37.0% |
| Sets 2–3, clean | release | 5 | 689.7 [673.4–710.6] | 148.7 [141.8–151.7] | 36.4–37.0% |
| Sets 2–3, clean | merged | 5 | 663.2 [654.1–675.8] | 150.8 [148.2–152.6] | 36.5–36.9% |
| With other traffic | release | 3 | 698.2 [681.4–703.7] | 149.5 [44.4–151.1] | 36.7–36.9% |
| With other traffic | merged | 3 | 1042.7 [672.3–3754.1] | 149.7 [28.0–152.5] | 36.5–36.7% |

Every sample processed exactly 7,680 prompt tokens or produced 256 completion tokens. The engine's
CPU use during decode was 0.02–0.05 core in both arms.

- **Prefill.** The merged build prefills the probe 3.8% faster over the clean rounds (663.4 vs
  689.7 ms; a permutation test on the medians gives p < 0.001), and is faster in 8 of the 9 clean
  round pairs. The upstream changes on this path are the two-stage GDN chunked kernels
  (`0784e76f`) and the NVFP4 and FP8 template routes (`fc3993d8`, `5d08cba8`); the gain was not
  attributed further.
- **Decode.** Equal within the measurement: 149.1 vs 148.5 tok/s over the clean rounds (p = 0.35).
  The first set alone was 1.8% slower (145.7 vs 148.3 tok/s, lower in 3 of 4 rounds); the next two
  sets were 1.4% faster (150.8 vs 148.7). A kernel trace of six 256-token decodes per arm (Nsight
  Systems, CUDA graphs traced per node, same flags) favours the merge: GPU kernel time fell from
  10,133 to 9,928 ms (−2.0%) and the wall time from 10.81 to 10.56 s. Every rewritten Linear
  family is faster, the largest being the FP8 A16 projections (the sliced-K MMA takes 3,023 ms
  against 3,098 ms for the former K-split MMA and SIMT kernels). The GDN replay kernels took 7 ms
  more (per launch, `recurrent_fold` +4% and `recurrent_record` +1%).
- **Other traffic.** Six rounds overlapped other engine requests (1–4 each). They spread both
  arms' samples (prefill up to 3.75 s, decode down to 28 tok/s) and are excluded from the
  comparison above.

## v3 port against the previous release build (2026-09-28)

Measured on the RTX PRO 6000 Blackwell (driver 616.92, 600 W limit, ECC on), with other GPU
applications paused and idle. Each arm is a fresh `ninfer-serve` with the production flags: `--max-context 131072 --kv-capacity 524288 --max-concurrency 8
--prefill-chunk 2048 --kv-dtype fp8 --vision --spec mtp --draft-tokens 5 --lm-head-draft
--desktop-reserve-gib 6`. Two rounds alternate the four arms. Per arm and round: one warmup, five
cold 7,680-token prefills (the `long_niah_8k` fixture trimmed to exactly 7,680 prompt tokens, a
fresh salt per request, read-only cache, one output token) and five greedy 256-token decodes
(counting prompt, thinking off). Prefill and decode times are the server's own request-log
timings; decode tok/s excludes the first token. CPU cores are the process's CPU time over the
decode repetitions divided by their wall time. Values are over both rounds (ten samples).

| Arm | Build | 7,680-token prefill ms, median [min–max] | Decode tok/s, median [min–max] | MTP acceptance | CPU cores in decode | Presence penalty |
|---|---|---:|---:|---:|---:|---:|
| Previous release | pre-v3 fork build, v2 artifact, spin | 683.6 [673.5–696.0] | 153.8 [150.7–154.8] | 36.9% | 0.96 | 0 |
| Port | v3 `27426227`, v3 copy, spin (token-fast W4A4 raster) | 683.4 [664.6–698.2] | 148.4 [132.9–153.0] | 36.6% | 0.96–0.98 | 0 |
| Port, blocking sync | same, `NINFER_CUDA_SYNC=blocking` | 681.0 [670.8–693.9] | 152.9 [149.7–154.0] | 36.8% | 0.03 | 0 |
| Port, weight-fast raster | same, W4A4 TMA grid in weight-fast order | 678.2 [673.7–681.8] | 153.2 [151.8–154.9] | 36.8% | 0.96–0.97 | 0 |

Every arm processed exactly 7,680 prompt tokens and 256 completion tokens. Round-to-round drift is
as large as any arm difference: the production arm's prefill median moved from 676.5 to 694.3 ms
between rounds, and the token-fast and weight-fast ordering of prefill reversed between rounds
(674.0 vs 677.1 ms, then 689.4 vs 678.4 ms).

- **Parity.** The v3 port prefills the probe in the same time as the previous release build (683.4 vs
  683.6 ms). Its decode medians are 152.9–153.2 tok/s in the blocking and weight-fast arms, whose
  decode path is the port's own (a single token tile orders identically under both rasters), against
  153.8 for the previous release build. The token-fast spin arm's lower median comes from slow samples (132.9 tok/s
  in round one, 141.6–149.9 in round two), not from a code difference.
- **W4A4 raster (P1).** Weight-fast against the v3 default token-fast: −0.8% prefill median, inside
  the noise and with inconsistent sign across rounds. The fork's 2026-09-10 regression (741.4 to
  749.9 ms, a different configuration) does not reproduce on v3, which also carries upstream's
  tiled activation-scale fetch. v3 keeps upstream's unconditional token-fast raster.
- **Blocking CUDA sync.** Prefill and decode are unchanged within noise, and the engine's CPU use
  during decode falls from about 0.96 core to 0.03. This is data only; the default stays spin.

## NVFP4 W4A4 prefill schedule (RTX 5090 record)

On the fork's pre-v3 engine, the desktop RTX 5090 selected token-fast CTA rasterization and one
activation-scale TMA fetch per K-tile pair for NVFP4 Linear and LinearSwiGLU, while the RTX PRO
6000 kept weight-fast rasterization and per-tile scale fetches: the combined change had regressed
its measured 7,680-token prefill from 741.4 to 749.9 ms. The v3 base makes token-fast rasterization
unconditional (upstream `ee9d5192`) and fetches activation scales one tile per TMA request
(`1d8587bc`); re-measured on v3 in the section above, the RTX PRO 6000 shows no difference between
the two rasters, so v3 keeps token-fast on every device.

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
fixtures do not establish legal-workflow acceptance. The real checkpoint qualification
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
low-thinking output and two-request concurrency checks. Named tool calls also passed in a
build that included a separate, pending forced-tool implementation; that observation
does not qualify forced-tool support in the standalone OrcaRouter change. The ordinary-decoding route
also correctly read a synthetic image's counts/shapes, streamed SSE through `[DONE]`, and reused
2764 tokens through a Responses `previous_response_id` continuation. Source-specific tokenizer
checks cover 17 independent reference cases; BF16 Linear and LinearTopK pass their independent
numerical oracles, including the existing FP8 top-k regression. These establish integration,
not long-horizon coding quality. The launcher defaults this derivative to ordinary decoding and
exposes both speculative backends for deliberate testing.
