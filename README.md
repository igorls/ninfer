# NInfer

A C++/CUDA inference engine for local, single-GPU text and vision workloads. This is
[igorls/ninfer](https://github.com/igorls/ninfer), the workstation fork of
[Neroued/ninfer](https://github.com/Neroued/ninfer), with native Windows support, a Supervisor app,
and application-serving extensions.

The engine targets **`sm_120a` only**. The fork supports **RTX 5090** and
**RTX PRO 6000 Blackwell, 96 GB**, with native Windows and Linux builds.
Qualification and performance records identify the device and execution route actually tested.

[Models](#models) · [Quick start](#quick-start) · [Performance and evaluation](#performance-and-evaluation) · [Documentation](docs/README.md)

## What this fork adds

- **Windows Supervisor:** tray app, browser dashboard, installer, model switching, desktop memory
  reserve, and request/cache/GPU diagnostics. It can also host llama.cpp's browser chat UI.
- **Application APIs:** OpenAI Chat Completions and Responses Core, Anthropic Messages, streaming,
  tool calls, constrained JSON with a supported JSON Schema subset, and token log probabilities.
- **Decision and scoring APIs:** `/v1/score`, TypeSafe-compatible `/v1/systemone`, and Jina-shaped
  `/v1/rerank`, plus offline causal scoring and hidden-row export.
- **Long-session reuse:** exact-prefix caching with Device/Host retention, read-only cache
  participation, and pressure-driven pause/resume through snapshots or token replay.
- **Native Flash-Next:** a v3 architecture package for text, MTP, Vision, scoring, and prefix reuse.

## Models

The engine loads **v3 `.ninfer` artifacts**. All models below have published v3 downloads on
Hugging Face; no local upgrade is needed for those files.

| Model | Weight profiles | Get the artifact | Optional components |
|---|---|---|---|
| Qwen3.8-Flash-Next | Mixed NVFP4 / FP8 / INT4 PLE, BF16 elsewhere | [Download v3](https://huggingface.co/igorls/Qwen3.8-Flash-Next-mixed-NInfer#ninfer-v3-artifact-re-derived-2026-10-11) (four files) | Vision; MTP with full output head |
| Qwen3.8-27B | `groupwise-int`, `nvfp4` | [Groupwise](https://huggingface.co/neroued/Qwen3.8-27B-NInfer), [NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | Vision; MTP; DFlash2 with companion weights |
| Qwen3.6-27B | `groupwise-int`, `nvfp4` | [Groupwise](https://huggingface.co/neroued/Qwen3.6-27B-NInfer), [NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | Vision; MTP |
| Qwen3.6-35B-A3B | `groupwise-int` | [Download](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | Vision; MTP; DFlash with companion weights |
| Qwen3.8-27B OrcaRouter Uncensored | NVFP4, BF16 embedding and full head | [Download v3](https://huggingface.co/igorls/Qwen3.8-27B-Uncensored-NVFP4-NInfer) | Vision; MTP; DFlash2 |

Flash-Next supports BF16 or FP8 KV and MTP draft counts 1–5. It uses the full stored output head:
**omit `--lm-head-draft`**. Its PLE table stays host-resident, and the artifact is about 102 GiB
across four files. See the [artifact guide](docs/maintainer/qwen3.8-flash-next-artifact.md) for
storage, memory, and execution requirements.

The experimental Qwen3.8-27B [`nvfp4full` recipe](docs/weight-conversion.md#qwen38-27b-nvfp4full)
is also available; it is **not qualified for production**. For custom weight mixtures and sources,
see [weight conversion](docs/weight-conversion.md).

## Quick start

Build from source, prepare a v3 artifact, then start the server. The example uses Qwen3.8-27B NVFP4.

```bash
git clone https://github.com/igorls/ninfer.git
cd ninfer
```

### Build on Windows

Use Windows 11, a 64-bit Visual Studio 2026 C++ environment (MSVC 19.51), CUDA 13.3, and CMake.
FFmpeg and libcurl (>= 7.85) need headers, MSVC import libraries, and runtime DLLs in their
`include/`, `lib/`, and `bin/` directories. Release packages bundle only LGPL FFmpeg.

```powershell
cmake -S . -B build-win -G "Visual Studio 18 2026" -A x64 `
  -DFFMPEG_ROOT=C:/deps/ffmpeg-lgpl-shared -DCURL_ROOT=C:/deps/curl
cmake --build build-win --config Release -j
```

The CLI and server are in `build-win/apps/Release/`. Put the FFmpeg, libcurl, and CUDA runtime
DLLs on `PATH` when running from the build tree. For build options and tests, see the
[build guide](docs/maintainer/build-system.md#windows-msvc).

### Build on Linux

Use 64-bit Linux, a CUDA toolkit supporting `sm_120a`, CMake >= 3.28, a C++20 host compiler, Ninja,
`pkg-config`, FFmpeg development libraries (`libavformat`, `libavcodec`, `libavutil`, `libswscale`),
and libcurl >= 7.85. Upstream validates CUDA 13.1; the fork also builds with CUDA 13.3.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run binaries from the source build tree. Tests and benchmarks are excluded by default;
[developer presets and targets](docs/maintainer/build-system.md) enable them.

### Download the model

Use the Hugging Face CLI to download the current v3 artifact:

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer qwen3_8_27b_nvfp4.ninfer --local-dir models
```

For **Flash-Next**, use the [v3 download command](docs/maintainer/qwen3.8-flash-next-artifact.md#1-download-and-derivation)
to fetch the entry file and all three parts. Keep them in one directory and load
`qwen3_8_flash_next_mixed.v3.ninfer`. Existing v2 files have separate
[upgrade instructions](docs/weight-conversion.md#upgrade-an-existing-v2-artifact).

### Start the server

On Windows:

```powershell
.\build-win\apps\Release\ninfer-serve.exe models/qwen3_8_27b_nvfp4.ninfer `
  --max-context 32768 --kv-capacity 65536 --max-concurrency 2 `
  --kv-dtype fp8 --desktop-reserve-gib 6 `
  --spec mtp --draft-tokens 3 --lm-head-draft --preserve-thinking
```

On Linux:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 32768 --kv-capacity 65536 --max-concurrency 2 \
  --kv-dtype fp8 --desktop-reserve-gib 6 \
  --spec mtp --draft-tokens 3 --lm-head-draft --preserve-thinking
```

This gives each request a 32,768-token limit, two execution lanes, and a shared 65,536-token
KV pool, while reserving 6 GiB for the desktop. These are example settings; longer contexts and
retained prefixes need more capacity. See [server options](docs/serving.md#server-options).
Add `--vision` at startup for image/video input; speculative and Vision residency are fixed
for the process lifetime.

### Send a request

Point an OpenAI client at `http://127.0.0.1:8080/v1` with model `default`, or use HTTP directly.
For PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/v1/chat/completions `
  -Method Post -ContentType 'application/json' `
  -Body '{"model":"default","messages":[{"role":"user","content":"Reply with one short sentence."}],"max_tokens":64}'
```

For Bash:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","messages":[{"role":"user","content":"Reply with one short sentence."}],"max_tokens":64}'
```

For one-shot generation, use `ninfer` instead of `ninfer-serve`; the [CLI guide](docs/cli.md)
covers prompts, chat histories, image/video input, and output streams.

### Windows app and Docker

To install the Supervisor, edit a copy of the
[example configuration](apps/ninfer-supervisor/supervisor.example.json) with your executable,
v3 artifact, working directory, and API-key paths, then run:

```powershell
.\scripts\windows\install.ps1 -ConfigPath .\supervisor.local.json
```

The per-user app includes a tray icon and a dashboard at `http://127.0.0.1:8099`.
See [Windows app operations](docs/windows-app.md) for configuration, updates, and removal,
or [Docker serving](docs/serving.md#docker) for the container workflow.

## Performance and evaluation

Results are recorded by **model, hardware, and workload**. The upstream serving corpus runs on
an RTX 5090; the fork's workstation and Colab G4 records use RTX PRO 6000 Blackwell variants.
These are separate measurements. Full tables, run conditions, and limitations live in the linked
references.

| Model | Performance records | Capability evaluation |
|---|---|---|
| Qwen3.8-Flash-Next mixed | [Windows serving](docs/performance.md#windows-serving-mtp-draft-count-2026-10-11), [G4 Engine and concurrency](docs/performance.md#flash-next-v3-on-colab-g4-2026-10-04) | No published capability scores; [numerical and runtime qualification](docs/maintainer/upstream-ports.md#flash-next-v3) |
| Qwen3.8-27B | [RTX 5090: groupwise-int and NVFP4](docs/performance/qwen3.8-27b.md) | [Groupwise](model-cards/Qwen3.8-27B-NInfer/README.md#evaluation), [NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md#evaluation) |
| Qwen3.6-27B | [RTX 5090: groupwise-int and NVFP4](docs/performance/qwen3.6-27b.md) | [Groupwise](model-cards/Qwen3.6-27B-NInfer/README.md#evaluation), [NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md#evaluation) |
| Qwen3.6-35B-A3B | [RTX 5090: groupwise-int](docs/performance/qwen3.6-35b-a3b.md) | [Model card](model-cards/Qwen3.6-35B-A3B-NInfer/README.md#evaluation) |
| Qwen3.8-27B OrcaRouter | [Historical workstation results](docs/performance/rtx-pro-6000.md) | [Model card](model-cards/Qwen3.8-27B-Uncensored-NVFP4-NInfer/README.md) |

Flash-Next has recorded Windows and Linux text/MTP/Vision checks. Numerical acceptance was recorded
on October 6 with one independent-oracle metric failure retained; end-to-end v2 parity was not
accepted. The October 11 Windows Engine run still has an open non-speculative concurrent-reuse
failure (`cudaErrorAlreadyMapped`). The [port record](docs/maintainer/upstream-ports.md#flash-next-v3)
keeps the full qualification scope and exceptions.

MTP gains depend on the workload: the Windows serving study also records slowdowns for
`top_logprobs` and concurrent low-acceptance chat. See the [measurement rules](docs/performance/methodology.md),
[capability evaluation workflow](eval/README.md), and [JevBench adapter](tools/bench/jevbench/)
for interpretation and reproduction.

## Limits and documentation

One GPU and one resident model per Engine, with 1–8 execution lanes and bounded FIFO admission.
Supervisor model switching replaces the engine. Multi-GPU/distributed serving and priority/QoS
are outside the current implementation. Tool calls are returned to clients; NInfer does not execute
them. The in-tree C++ headers are not an installed SDK.

- [Documentation index](docs/README.md): user guides, model/artifact references, and maintainer contracts.
- [HTTP serving](docs/serving.md): protocol behavior, authentication, structured output, scoring, and telemetry.
- [Context reuse and scheduling](docs/maintainer/resource-scheduling-and-context-cache.md): retention and pressure recovery.
- [Performance index](docs/performance.md): all published coverage, including experimental profiles.
- [Contributing](CONTRIBUTING.md): development and review workflow.

Run the relevant executable's `--help` for exact options and defaults.

## Upstream and licensing

The original engine and upstream Qwen artifacts are the work of
[Neroued/ninfer](https://github.com/Neroued/ninfer) and its contributors. This fork maintains the
Windows, workstation, Flash-Next, and application-serving additions; upstream measurements retain
their own provenance.

NInfer is licensed under [Apache-2.0](LICENSE). Model artifacts retain their source licenses and
provenance; see their model cards and conversion references. Vendored dependencies retain their
license files under `third_party/`.
