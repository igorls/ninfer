# Sequence D17 Item 2: Output-Head FP8 Quantizer & Greedy Divergence Benchmark

## 1. Overview & Rationale

During the Flash-Next architecture audit, a dormant quantizer was identified at `src/targets/qwen3_8_flash_next/impl/load/materialized.cpp:271`:
```cpp
quantize_bf16_output_head_to_fp8_e4m3_row_f32s(text.output_head, output_head_fp8, fp8_head, ...)
```
While the kernel implementation and unit verification (`tests/targets/qwen3_8_flash_next/test_output_head_fp8.cpp`) were already complete, the quantization pass was hardcoded to `false` and not exposed to user-facing CLI or server flags.

- **Storage Impact**: The Qwen3.8-Flash-Next output head is a $248,320 \times 2,560$ projection matrix. In BF16, this single unquantized tensor occupies **1,271,398,400 bytes (~1.18 GiB)** of device VRAM. In FP8 E4M3 with row-wise float32 scaling, it occupies **636,692,480 bytes (~607 MiB)**, yielding **~605 MiB** of net GPU memory reduction.
- **Critical Quality Seam**: Unlike speculative draft heads (where prediction errors are discarded during verification against the target model), the primary output head directly projects final hidden states to vocabulary logits. Quantization errors directly alter the emitted token distribution.
- **Acceptance Contract**: Rather than a crude binary pass/fail or perplexity proxy, acceptance is evaluated through **greedy divergence position ($P_{\text{div}}$)** against unquantized BF16 across identical prompts and seeds under temperature 0.0.

---

## 2. CLI & Server Interface

The following options have been wired into `ninfer` (CLI) and `ninfer-serve` (REST service), defaulting to `false`/`bf16` (preserving unquantized baseline behavior):

| Flag | Values | Default | Description |
|---|---|---|---|
| `--output-head-fp8` | boolean flag | `false` | Enable FP8 E4M3 row-scaled output head |
| `--no-output-head-fp8` | boolean flag | `false` | Explicitly disable FP8 output head |
| `--output-head-dtype` | `bf16` \| `fp8` | `bf16` | Explicit output head datatype selector |

---

## 3. Greedy Divergence Harness (`bench_output_head_divergence.py`)

The benchmark harness executes paired A/B evaluation across a diverse, curated 14-prompt corpus spanning five distinct workloads:
1. `code_generation`: C++ and Python systems tasks (CUDA reductions, SPSC ring buffer, prefix trie)
2. `technical_reasoning`: Low-level hardware, memory hierarchies, and floating-point representations (FP8 formats, NVLink vs PCIe, cache coherency)
3. `structured_json`: Schema-constrained tool invocation and observability telemetry
4. `mathematics_logic`: Multi-step analytical math, linear algebra, and combinatorics
5. `dialogue_multilingual`: Conversational English and localized Portuguese concierge dialogue

### Metrics Reported:
- **Divergence Position ($P_{\text{div}}$)**: The 1-based index of the first differing token ($t_i^{\text{BF16}} \ne t_i^{\text{FP8}}$). If output streams match completely up to maximum generated length, $P_{\text{div}} = \text{None}$ (100% agreement).
- **Prefix Match Ratio**: Length of identical prefix divided by total generated tokens.
- **First Divergent Token Pair**: Exact token representation from both arms at the point of divergence.
- **$P_{\text{div}}$ Statistical Distribution**: Min, P25, Median, P75, Max, and Mean across all divergent prompts.
- **Empirical VRAM Reduction**: Verified via `nvidia-smi` delta before and after quantization pass.

---

## 4. Execution Commands for Coordinator (Sole GPU Owner)

Under the active GPU stand-down protocol, only the coordinator (`windows:claude:ninfer`) runs GPU processes.

### A. Fully Automated Sequential Run
Starts BF16 server, evaluates all prompts, drains VRAM, starts FP8 server, evaluates all prompts, and outputs summary JSON and terminal table:
```powershell
python P:\NInfer\bench\d17\bench_output_head_divergence.py --port 8160 --max-tokens 256
```

### B. Evaluating Against Pre-Existing Servers
If coordinator manages server lifecycles manually:
```powershell
# Server 1 (BF16):
.\build-win\apps\Release\ninfer-serve.exe E:\models\Qwen3.8-Flash-Next\qwen3_8_flash_next_nvfp4_mtp.ninfer --port 8160 --output-head-dtype bf16 --max-context 32768 --kv-capacity 32768 --max-concurrency 1 --max-private-continuations 16 --greedy --no-thinking

# Server 2 (FP8):
.\build-win\apps\Release\ninfer-serve.exe E:\models\Qwen3.8-Flash-Next\qwen3_8_flash_next_nvfp4_mtp.ninfer --port 8161 --output-head-dtype fp8 --max-context 32768 --kv-capacity 32768 --max-concurrency 1 --max-private-continuations 16 --greedy --no-thinking

# Run divergence harness:
python P:\NInfer\bench\d17\bench_output_head_divergence.py --bf16-url http://127.0.0.1:8160 --fp8-url http://127.0.0.1:8161 --max-tokens 256
```

---

## 5. Artifact Output

The benchmark automatically generates `bench/d17/d17_output_head_divergence_summary.json` containing:
- Complete execution configuration, model path, and timestamp
- VRAM usage and savings measurements
- Overall statistics ($P_{\text{div}}$ distribution, zero-divergence rate, mean prefix match)
- Category breakdown
- Full per-prompt token streams, timing, and divergence analysis
