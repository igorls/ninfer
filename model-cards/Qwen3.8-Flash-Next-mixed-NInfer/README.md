---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
  - Qwen/Qwen3.8-Flash-Next
  - primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8
  - primitive-ai/Qwen3.8-Flash-Next-PLE-quant
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.8
  - flash-next
  - nvfp4
  - fp8
  - int4
  - mixed-precision
  - blackwell
  - rtx-pro-6000
  - windows
  - single-gpu
  - multimodal
  - speculative-decoding
---

# Qwen3.8-Flash-Next mixed quantization for NInfer

An **early-access engineering preview** of Qwen3.8-Flash-Next in the native `.ninfer`
format, prepared for the [NInfer workstation fork](https://github.com/igorls/ninfer).
This release distributes the artifact used by our Windows RTX PRO 6000 service,
including its text, Vision, MTP, tokenizer, and prompt-template data.

The artifact combines Primitive AI's NVFP4/FP8 compute weights with its INT4 PLE
table. It preserves all 512 routed experts per layer. This is a format conversion
and source splice, with no fine-tuning or expert pruning performed by this release.
Credit for the base model belongs to Qwen and for the source quantizations to
Primitive AI; see [NOTICE.md](NOTICE.md).

**Use the pinned NInfer fork below.** This file cannot be loaded by Transformers,
vLLM, llama.cpp, or Ollama. There is no Hugging Face hosted inference integration.

## Download and identity

| Property | Value |
|---|---|
| File | `qwen3_8_flash_next_mixed.ninfer` |
| Download size | 113,298,397,952 bytes — 113.30 GB / 105.52 GiB |
| Container | NInfer v2 |
| Model identity | `qwen3.8-flash-next` |
| Weight identity | `mixed-nvfp4-fp8-ple-int4` |
| Release revision | `preview-2026-09-07` |
| Compatible engine revision | [`c81c88f6d66652be4607fb775333a6c63132f3f5`](https://github.com/igorls/ninfer/tree/c81c88f6d66652be4607fb775333a6c63132f3f5) |

Using the Hugging Face CLI:

```powershell
hf download igorls/Qwen3.8-Flash-Next-mixed-NInfer `
  qwen3_8_flash_next_mixed.ninfer artifact-manifest.json SHA256SUMS `
  LICENSE NOTICE.md LICENSE-APACHE-2.0.txt README.md `
  --revision preview-2026-09-07 --local-dir .\models\flash-next

(Get-FileHash .\models\flash-next\qwen3_8_flash_next_mixed.ninfer -Algorithm SHA256).Hash
```

Expected SHA256:

```text
3d383e51963aafd4318dfd04c8dc63ee7df11768de19d9ab58dbba44460d1d02
```

The manifest and checksum identify the exact release bytes. Keep them with the
artifact when reproducing results.

## Storage and memory

The on-disk inventory has 1,566 objects: 1,560 tensors and six frontend resources.

| Component | Stored format | Runtime placement |
|---|---|---|
| Main routed expert banks | 96 NVFP4 tensors; K16 scales and expert divisors | GPU |
| Main QSA/GDN projection parents | 96 FP8 E4M3 tensors with FP32 row scales | GPU |
| PLE embedding table | 128 INT4 group-16 tensors with FP16 scales | Host file mapping / RAM page cache |
| PLE indexing metadata | Three I64 tensors | Host file mapping |
| MTP expert banks | Two BF16 tensors, included in the BF16 count below | Converted to NVFP4 by the loader when MTP is enabled |
| Other weights, including Vision and MTP tails | BF16; 1,237 BF16 tensors in total | GPU, subject to enabled features |
| Tokenizer, template and media configuration | Six embedded raw resources | Host |

**MTP is BF16 on disk and NVFP4 during execution.** The two stored MTP expert banks
occupy 5,033,164,800 bytes. The pinned loader retains those source banks in the file
mapping and creates 1,415,581,696 bytes of NVFP4 device payload using NInfer's
weights-only quantizer. No quantization calibration dataset was added in this
conversion. The optional separately spliced artifact with NVFP4 MTP already on disk
is not the file published here.

With Vision and MTP enabled and the default BF16 embedding/output head, the other
device-bound tensor payload totals 76,251,938,528 bytes (71.02 GiB), before the MTP
NVFP4 buffers, alignment, execution workspaces, CUDA graphs, recurrent state and KV
cache. This is a payload accounting figure, **not total VRAM usage**. The PLE table
plus its indexing metadata occupies 32,000,162,072 bytes (29.80 GiB) of host-mapped
data. The runtime warms the table's pages at startup; reclaiming this cache under
memory pressure can hurt latency.

The development workstation has one **RTX PRO 6000 Blackwell Workstation Edition
(96 GB)** and 128 GB-class system RAM (125.64 GiB visible to Windows). These are the
observed host specifications, not a measured minimum-RAM requirement. Use an SSD
with more than 114 GB free for this file and metadata; leave additional room for
the build and download cache. We have not qualified this artifact on smaller GPUs.

## Windows quick start

Build from an x64 Visual Studio developer shell with CUDA installed. This example
uses the workstation toolchain: Visual Studio 2026, CUDA 13.3, and CMake 4.3. NInfer
targets `sm_120a`; this release is intended for the RTX PRO 6000 Blackwell platform.

```powershell
git clone https://github.com/igorls/ninfer.git
cd ninfer
git checkout c81c88f6d66652be4607fb775333a6c63132f3f5

cmake -S . -B build-win -G "Visual Studio 18 2026" -A x64 -DNINFER_BUILD_MEDIA=OFF
cmake --build build-win --config Release -j
```

Download the model into `models/flash-next` inside that checkout using the command
above, then start a bounded text-serving profile:

```powershell
.\build-win\apps\Release\ninfer-serve.exe .\models\flash-next\qwen3_8_flash_next_mixed.ninfer `
  --host 127.0.0.1 --port 8010 --model-id qwen3.8-flash-next `
  --max-context 32768 --kv-capacity 65536 --max-concurrency 2 `
  --prefill-chunk 8192 --desktop-reserve-gib 3 `
  --kv-dtype fp8 --gdn-state-dtype bf16 --spec mtp --draft-tokens 4 `
  --preserve-thinking
```

This is a conservative starting profile, not the settings used for every published
measurement. Wait for the server-ready message: model materialization, host-table
warm-up and graph creation happen before serving. `--max-context` limits each
request; `--kv-capacity` is shared across active requests. The engine supports up
to eight active requests, but a KV pool sized for fewer full-context requests can
queue long requests. MTP currently speculates at decode batch size one; larger
batches use ordinary batched decode.

For image/video input, build with `NINFER_BUILD_MEDIA=ON` and the FFmpeg/libcurl
dependencies described in the [fork README](https://github.com/igorls/ninfer/blob/c81c88f6d66652be4607fb775333a6c63132f3f5/README.md),
then add `--vision`. The text-only build above does not enable image/video input.
The artifact always contains Vision and MTP weights, even when a feature is disabled.

An OpenAI-compatible structured-output request:

```powershell
$body = @{
  model = "qwen3.8-flash-next"
  messages = @(@{role = "user"; content = 'Return a JSON object with a "greeting" string in Portuguese.'})
  max_tokens = 128
  reasoning_effort = "none"
  response_format = @{type = "json_object"}
} | ConvertTo-Json -Depth 6

Invoke-RestMethod http://127.0.0.1:8010/v1/chat/completions `
  -Method Post -ContentType "application/json" -Body $body
```

The fork also serves OpenAI Responses and Anthropic Messages, tool calls, streaming,
and a supported JSON Schema subset using constrained decoding. Details and explicit
unsupported-schema behavior are in the [serving reference](https://github.com/igorls/ninfer/blob/c81c88f6d66652be4607fb775333a6c63132f3f5/docs/serving.md).
JSON conformance and application-level answer correctness are separate properties.

## Provenance and validation

| Input | Pinned revision | Contribution |
|---|---|---|
| [Primitive mixed NVFP4/FP8](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8/tree/a4e813ed3cfbbcc61e2929699eccb864a4dfa843) | `a4e813ed3cfbbcc61e2929699eccb864a4dfa843` | Compute backbone, BF16 MTP/Vision/tails and frontend; BF16 PLE shards omitted |
| [Primitive INT4 PLE](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant/tree/da8b39586016d8325ac619be28ad77d6296625ec) | `da8b39586016d8325ac619be28ad77d6296625ec` | All 128 `ples_int4` table shards |
| [Official Qwen reference](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540) | `de4b8e4d43b917e7706784d8bb445c9af86a3540` | Retained license and byte-identical frontend reference |

The converter aggregates expert banks, arranges projection rows and scale planes
for NInfer, substitutes the INT4 PLE table, and embeds the six frontend resources.
It consumes the 296,502 non-PLE source tensors recorded in the mixed checkpoint
index and replaces its 128 BF16 PLE tensors. The sidecar recorded recipe
`qwen3_8_flash_next_mixed-v1` and the two source revisions above, but did not record
the historical converter Git revision. That historical revision is unknown; the
compatible engine pin is not presented as the original converter revision. The
original BF16 checkpoint revision used by Primitive is also not independently
established by this release.

Release checks on September 7, 2026 read the actual file with NInfer's container
reader, validating the v2 directory, format/shape byte sizes, offsets, alignments
and file bounds. The full-file SHA256 was computed, and every embedded frontend
resource matched both the recorded conversion hashes and the pinned official
Qwen reference. This verifies artifact structure and identity, not full numerical
equivalence to Qwen's original BF16 inference implementation.

Earlier Windows RTX PRO 6000 engineering checks cover serving, multi-position MTP
acceptance, prefix reuse, Vision smoke requests, and constrained output. Workload
definitions, timings and limitations are in the [performance reference](https://github.com/igorls/ninfer/blob/c81c88f6d66652be4607fb775333a6c63132f3f5/docs/performance.md).
This publication adds no benchmark campaign or quality score. Real application
evaluation, including manual review of complex legal/evidence workflows, remains
in progress. This preview does not establish production readiness for those tasks.

## License

The Qwen-derived model weights are distributed with the original **Qwen Community
License 1.0**, reproduced verbatim in [LICENSE](LICENSE). Its commercial-use
conditions apply to derivatives; this is not an Apache-2.0-only model release.
Primitive's PLE repository separately advertises Apache-2.0 metadata; the
accompanying [Apache license text](LICENSE-APACHE-2.0.txt) and [attribution notice](NOTICE.md)
preserve that information without replacing Qwen's conditions. The NInfer engine
source has its own Apache-2.0 license.
