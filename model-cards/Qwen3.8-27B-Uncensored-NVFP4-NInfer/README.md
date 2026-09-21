---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - orcarouter/Qwen3.8-27B-Uncensored-NVFP4
  - incoai/Qwen3.8-27B-DFlash2
base_model_relation: quantized
language:
  - en
  - zh
tags:
  - ninfer
  - qwen3.8
  - uncensored
  - abliterated
  - nvfp4
  - fp8
  - bf16
  - dflash2
  - speculative-decoding
  - mtp
  - blackwell
  - cuda
  - windows
  - multimodal
  - conversational
---

# Qwen3.8-27B Uncensored NVFP4 for NInfer

A native `.ninfer` conversion of [OrcaRouter's Qwen3.8-27B-Uncensored-NVFP4](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-NVFP4), with the [Inco AI DFlash2 companion](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) included. It runs through the [igorls/NInfer fork](https://github.com/igorls/ninfer/tree/91ce2f2c73c27ffbff3bd7c43a319f295e9597cc).

OrcaRouter produced the abliterated model and its mixed NVFP4/FP8 weights. This release packages those weights for NInfer, preserving the source **BF16 token embeddings and full output head**. No additional fine-tuning or refusal-removal procedure was performed for this conversion.

The file contains the target model, Vision and MTP components, an optimized proposal head, the DFlash2 companion, and all six tokenizer/template/media resources. Separate source-checkpoint downloads or companion splicing are unnecessary for inference.

| Property | Value |
|---|---|
| File | `qwen3_8_27b_orcarouter_nvfp4.ninfer` |
| Size | 26,268,462,848 bytes (26.27 GB / 24.46 GiB) |
| Engine identity | `qwen3.8-27b-orcarouter/nvfp4` |
| Runtime | NInfer fork, revision `91ce2f2c73c27ffbff3bd7c43a319f295e9597cc` |
| CUDA target | `sm_120a` |
| Qualified workstation | RTX PRO 6000 Blackwell Workstation Edition, 96 GB, Windows |
| License | Apache 2.0 |

This is a NInfer artifact. Transformers, vLLM, Ollama and llama.cpp do not load the `.ninfer` format. Disk size is not total VRAM usage: enabled features, CUDA graphs, concurrency and context caches require additional memory. Smaller GPUs and maximum-context operation have not been qualified for this release.

## Download and verify

With the Hugging Face CLI available:

```powershell
hf download igorls/Qwen3.8-27B-Uncensored-NVFP4-NInfer --local-dir .\models\orcarouter
Get-FileHash .\models\orcarouter\qwen3_8_27b_orcarouter_nvfp4.ninfer -Algorithm SHA256
```

Expected SHA-256:

```text
003f8c65175e262e66f83a803f7b1d286f7226ede0e672d3b3e5c4b6c75f3e7f
```

The repository also provides [SHA256SUMS](SHA256SUMS), an [artifact manifest](artifact-manifest.json), and a [conversion report](qwen3_8_27b_orcarouter_nvfp4.ninfer.conversion.json).

## Windows quick start

Use an x64 Visual Studio developer shell with CUDA installed. The qualified toolchain is Visual Studio 2026 and CUDA 13.3. Build the matching source before downloading into its `models/orcarouter` directory:

```powershell
git clone https://github.com/igorls/ninfer.git
cd ninfer
git checkout 91ce2f2c73c27ffbff3bd7c43a319f295e9597cc

cmake -S . -B build-win -G "Visual Studio 18 2026" -A x64 -DNINFER_BUILD_MEDIA=OFF
cmake --build build-win --config Release --target ninfer-serve ninfer-supervisor -j

hf download igorls/Qwen3.8-27B-Uncensored-NVFP4-NInfer --local-dir .\models\orcarouter

.\build-win\apps\Release\ninfer-serve.exe .\models\orcarouter\qwen3_8_27b_orcarouter_nvfp4.ninfer `
  --host 127.0.0.1 --port 8010 --model-id qwen3.8-27b-orcarouter `
  --max-context 32768 --kv-capacity 65536 --max-concurrency 4 `
  --prefill-chunk 2048 --kv-dtype fp8 --desktop-reserve-gib 3
```

This starts ordinary decoding. Wait for the server-ready message before sending requests. The per-request limit is 32K tokens; the 64K KV pool is shared across requests.

The example builds text serving. For image/video input, build with `NINFER_BUILD_MEDIA=ON`, supply FFmpeg/libcurl and their runtime DLLs as described in the [fork's build instructions](https://github.com/igorls/ninfer/blob/91ce2f2c73c27ffbff3bd7c43a319f295e9597cc/README.md#build-on-windows), and add `--vision` at launch.

The Supervisor example contains a **27B OrcaRouter Uncensored** entry. Set its artifact and executable paths for your installation; its dashboard exposes ordinary decoding, MTP and DFlash2.

An OpenAI-compatible request:

```powershell
$body = @{
  model = "qwen3.8-27b-orcarouter"
  messages = @(@{role = "user"; content = "Explain how asyncio.Queue.task_done() and join() interact."})
  max_tokens = 512
  reasoning_effort = "none"
} | ConvertTo-Json -Depth 6

Invoke-RestMethod http://127.0.0.1:8010/v1/chat/completions `
  -Method Post -ContentType "application/json" -Body $body
```

## Optional speculative decoding

Restart the server with one of these additions:

| Backend | Additional launch arguments |
|---|---|
| MTP, optimized proposal head | `--spec mtp --draft-tokens 5 --lm-head-draft` |
| DFlash2, optimized proposal head | `--spec dflash2 --draft-tokens 7 --lm-head-draft` |
| DFlash2, full BF16 proposal head | `--spec dflash2 --draft-tokens 7` |

The DFlash2 companion was trained for canonical Qwen3.8-27B. Its acceptance on this derivative is workload dependent. Different verification widths select different floating-point kernel paths, so this integration does not promise byte-identical output to ordinary decoding. The target verifies proposed tokens; acceptance rate is not an answer-quality metric.

## Validation and initial performance

Integration checks cover source-specific tokenization, independent BF16 Linear/LinearTopK numerical oracles, ordinary/MTP/DFlash2 serving, JSON Schema output and two-request concurrency. The installed ordinary route also passed an image-reading check, SSE streaming and Responses continuation reuse. The matching source built in isolation and passed the focused converter, tokenizer, frontend, Supervisor and sampling-default checks.

A single 169-token Python queue-repair prompt measured **70.8 tok/s ordinary**, **196.0 tok/s MTP K5**, and **208.8 tok/s DFlash2 K7 with the optimized head** on the RTX PRO 6000, with ECC enabled, driver 616.92 and CUDA 13.3. It used greedy non-thinking sampling, neutral penalties, a 1536-token output limit, 32K context, 64K FP8 KV, four configured lanes, 2048-token prefill chunks, CUDA graphs and Vision. Only one request ran during each timed sample. Rates use engine decode time and exclude the first output token.

These are integration-probe results, not a benchmark-suite average or fixed-output speedup: responses differed in length and content. Manual review found coding errors, including in an answer whose generated tests passed. Long-horizon coding quality and equivalence to the source in other engines remain unevaluated. See the [full probe and its limitations](https://github.com/igorls/ninfer/blob/91ce2f2c73c27ffbff3bd7c43a319f295e9597cc/docs/performance.md#orcarouter-nvfp4-integration-probe).

## Sources and license

- Target, Vision, MTP and frontend: [OrcaRouter source revision `69d21348`](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-NVFP4/tree/69d21348b2d6c11439fb69368f40414c2256e44e).
- DFlash2 companion: [Inco AI source revision `dedf8df6`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/tree/dedf8df68adfb1afeaf7b7480c0a0243108177b4).
- Underlying base: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
- Conversion recipe: [`qwen3_8_27b_orcarouter_nvfp4-v1`](https://github.com/igorls/ninfer/blob/91ce2f2c73c27ffbff3bd7c43a319f295e9597cc/tools/convert/qwen3_8_27b/convert_orcarouter_nvfp4.py).

Apache 2.0; see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). This release retains the source model's refusal-removed behavior and limitations. It is an independent NInfer conversion and does not imply endorsement by the source authors.
