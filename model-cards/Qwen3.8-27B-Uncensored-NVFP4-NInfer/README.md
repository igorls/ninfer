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

A native **NInfer v3** `.ninfer` conversion of [OrcaRouter's Qwen3.8-27B-Uncensored-NVFP4](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-NVFP4), with the [Inco AI DFlash2 companion](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) included. It runs through the [igorls/NInfer fork](https://github.com/igorls/ninfer) on the current workstation line.

OrcaRouter produced the abliterated model and its mixed NVFP4/FP8 weights. This release packages those weights for NInfer, preserving the source **BF16 token embeddings and full output head**. No additional fine-tuning or refusal-removal procedure was performed for this conversion.

The file contains the target model, Vision and MTP components, an optimized proposal head, the DFlash2 companion, and all six tokenizer/template/media resources. Separate source-checkpoint downloads or companion splicing are unnecessary for inference.

| Property | Value |
|---|---|
| File | `qwen3_8_27b_orcarouter_nvfp4.ninfer` |
| Container | NInfer **v3** (`NINFER\x00\x03`) |
| Size | 26,268,683,012 bytes (26.27 GB / 24.46 GiB) |
| SHA-256 | `50f2e407a2a6ee6b049a41de2724119cb4bbcb189ba14445babdd2ff5d56a384` |
| Engine identity | `qwen3.8-27b-orcarouter/nvfp4` |
| Conversion recipe | `qwen3_8_27b_orcarouter_nvfp4` (re-convert; not `upgrade_ninfer_v2_to_v3.py`) |
| Runtime | NInfer fork, revision `6ad46d87fbeef4573e8ff7fcd59a534e5666fb0a` (workstation) |
| CUDA target | `sm_120a` |
| Qualified workstation | RTX PRO 6000 Blackwell Workstation Edition, 96 GB, Windows |
| License | Apache 2.0 |

This is a NInfer artifact. Transformers, vLLM, Ollama and llama.cpp do not load the `.ninfer` format. Disk size is not total VRAM usage: enabled features, CUDA graphs, concurrency and context caches require additional memory. Smaller GPUs and maximum-context operation have not been qualified for this release.

> **v2 note:** Earlier uploads of this repo were NInfer **v2**. Current workstation builds expect **v3**. Re-download and verify the SHA-256 above before serving.

## Download and verify

With the Hugging Face CLI available:

```powershell
hf download igorls/Qwen3.8-27B-Uncensored-NVFP4-NInfer --local-dir .\models\orcarouter
Get-FileHash .\models\orcarouter\qwen3_8_27b_orcarouter_nvfp4.ninfer -Algorithm SHA256
```

Expected SHA-256:

```text
50f2e407a2a6ee6b049a41de2724119cb4bbcb189ba14445babdd2ff5d56a384
```

The repository also provides [SHA256SUMS](SHA256SUMS), an [artifact manifest](artifact-manifest.json), and a [conversion report](qwen3_8_27b_orcarouter_nvfp4.ninfer.conversion.json).

## Windows quick start (current workstation)

Use an x64 Visual Studio developer shell with CUDA installed. The qualified toolchain is Visual Studio 2026 and CUDA 13.3. Build the matching source before downloading into its `models/orcarouter` directory:

```powershell
git clone https://github.com/igorls/ninfer.git
cd ninfer
git checkout 6ad46d87fbeef4573e8ff7fcd59a534e5666fb0a
# or: git checkout workstation

cmake -S . -B build-win -G "Visual Studio 18 2026" -A x64 -DNINFER_BUILD_MEDIA=OFF
cmake --build build-win --config Release --target ninfer-serve ninfer-supervisor -j

hf download igorls/Qwen3.8-27B-Uncensored-NVFP4-NInfer --local-dir .\models\orcarouter

.\build-win\apps\Release\ninfer-serve.exe .\models\orcarouter\qwen3_8_27b_orcarouter_nvfp4.ninfer `
  --host 127.0.0.1 --port 8010 --model-id qwen3.8-27b-orcarouter `
  --max-context 32768 --kv-capacity auto --max-concurrency 4 `
  --prefill-chunk 2048 --kv-dtype fp8 --desktop-reserve-gib 3 `
  --clamp-concurrency-to-pool
```

This starts ordinary decoding. Wait for the server-ready / listening message before sending requests. `--kv-capacity auto` sizes the pool from free VRAM after the desktop reserve; raise `--max-context` only when headroom allows.

The example builds text serving. For image/video input, build with `NINFER_BUILD_MEDIA=ON`, supply FFmpeg/libcurl and their runtime DLLs as described in the fork's build instructions, and add `--vision` at launch.

The Supervisor catalog reads the v3 header and `metadata.name`. Set its artifact and executable paths for your installation; its dashboard exposes ordinary decoding, MTP and DFlash2.

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

## Validation

On the qualified workstation (RTX PRO 6000, Windows), this v3 artifact:

- Has magic `NINFER\x00\x03`, inspectable via `python -m tools.artifact.inspect`
- Loads weights (21.3 GiB device upload) beside a live production engine
- Serves ordinary decode on a spare port (`--kv-capacity auto`) and returns a correct short chat completion

Integration checks for the recipe cover source-specific tokenization, BF16 Linear/LinearTopK oracles, ordinary/MTP/DFlash2 serving paths, and Vision when media is built. See the fork's performance notes for earlier probe numbers on this identity.

## Sources and license

- Target, Vision, MTP and frontend: [OrcaRouter source revision `69d21348`](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-NVFP4/tree/69d21348b2d6c11439fb69368f40414c2256e44e).
- DFlash2 companion: [Inco AI source revision `dedf8df6`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/tree/dedf8df68adfb1afeaf7b7480c0a0243108177b4).
- Underlying base: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
- Conversion recipe: `qwen3_8_27b_orcarouter_nvfp4` in the fork's `tools/convert/official_recipes.py`.

Apache 2.0; see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). This release retains the source model's refusal-removed behavior and limitations. It is an independent NInfer conversion and does not imply endorsement by the source authors.
