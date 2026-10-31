# NInfer documentation

Start with the [project README](../README.md) to build NInfer, download a published artifact, and
run the CLI or HTTP server.

## User guides

| Document | Purpose |
|---|---|
| [CLI](cli.md) | text, chat-history, image/video input, output streams, sampling, MTP, and common runtime options |
| [HTTP serving](serving.md) | OpenAI Responses/Chat Completions, Anthropic Messages, TypeSafe System One, state, streaming, token counting, authentication, and tool calls |
| [Demos](index.html) | Arena Duel, Tetris, chess, Kitchen Rush, Beacon Runner, the cube speed run, the decision analyzer, and the dual-purpose readout |
| [Decision arcade](decision-arcade.md) | real-time Arena Duel combat, Tetris, chess, Kitchen Rush, Beacon Runner and Cube, with reproducible decision evaluation |
| [Windows app](windows-app.md) | install the standalone Supervisor, launch at sign-in, update, and uninstall |
| [Performance](performance.md) | RTX 5090 and RTX PRO 6000 measurement coverage, per-model results, methodology, and publication rules |
| [Weight conversion](weight-conversion.md) | official recipes, custom formats and sources, conversion methods, optional components and artifact output |
| [Perplexity](perplexity.md) | fixed-corpus and custom-text causal perplexity, comparison rules, progress, and reports |
| [CLI examples](../examples/cli/) | committed text, multimodal, thinking, long-decode, and long-context inputs |

The executable `--help` output is the exact source for command-line option spelling and defaults.

## Model artifacts

| Model | Weights | Download / preparation | Model card / reference |
|---|---|---|---|
| Qwen3.8-Flash-Next | Mixed NVFP4 / FP8 / INT4 PLE, BF16 elsewhere | [Hugging Face v3](https://huggingface.co/igorls/Qwen3.8-Flash-Next-mixed-NInfer#ninfer-v3-artifact-re-derived-2026-10-11), [download instructions](maintainer/qwen3.8-flash-next-artifact.md#1-download-and-derivation) | [Artifact contract](maintainer/qwen3.8-flash-next-artifact.md), [qualification](maintainer/upstream-ports.md#flash-next-v3) |
| Qwen3.6-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | [model card](../model-cards/Qwen3.6-27B-NInfer/README.md) |
| Qwen3.6-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) |
| Qwen3.8-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | [model card](../model-cards/Qwen3.8-27B-NInfer/README.md) |
| Qwen3.8-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) |
| Qwen3.6-35B-A3B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | [model card](../model-cards/Qwen3.6-35B-A3B-NInfer/README.md) |
| Qwen3.8-27B OrcaRouter Uncensored | NVFP4, BF16 embedding and full head | [Hugging Face v3](https://huggingface.co/igorls/Qwen3.8-27B-Uncensored-NVFP4-NInfer) | [Model card](../model-cards/Qwen3.8-27B-Uncensored-NVFP4-NInfer/README.md) |

The engine requires v3, and all models in this table have published v3 downloads. Only existing
v2 files need the [upgrade workflows](weight-conversion.md#upgrade-an-existing-v2-artifact).
The experimental [Qwen3.8-27B `nvfp4full` recipe](weight-conversion.md#qwen38-27b-nvfp4full)
is not qualified for production.

## Repository-local guides

- [Benchmarks](../bench/README.md)
- [Tests](../tests/README.md)
- [Tools](../tools/README.md)
- [Capability evaluation](../eval/README.md)

## Maintainer references

The active references under [`maintainer/`](maintainer/) record current architecture, model,
artifact, and maintenance contracts. These files are not additional user workflows or installed
API documentation.

[Engine architecture](maintainer/engine-architecture.md) is the single top-level reference. The
other references own narrower contracts:

| Document | Responsibility |
|---|---|
| [Engine architecture](maintainer/engine-architecture.md) | model/config/weight ownership, loading-to-execution flow, requests, scheduling, transactions and graphs |
| [Build system](maintainer/build-system.md) | CMake targets, explicit source ownership, CUDA compilation boundaries, presets and developer configuration |
| [Artifact container](maintainer/artifact-container.md) | v3 directory, objects, logical bindings, Uses, resources and file framing/sharding |
| [Numeric formats](maintainer/tensor-formats.md) | represented values, codes/scales, conversion arithmetic and numerical interpretation |
| [Storage layouts](maintainer/storage-layouts.md) | packing, plane offsets, padding, encoded sizes and view addressing |
| [Qwen3.8-Flash-Next artifact](maintainer/qwen3.8-flash-next-artifact.md) | Published packed-v3 source provenance, historical inventory and canonical reframe; current bindings and execution are in the Qwen4Exp references |
| [Qwen3.5 model](maintainer/qwen3_5-model.md) | Dense/MoE mathematics, instance config, logical parameters, MTP, Vision and state semantics |
| [Qwen4Exp model and weights](maintainer/qwen4_exp-model.md) | Flash Next architecture, quantization recipe, merged projections, expert banks and PLE representation |
| [Qwen4Exp execution and state](maintainer/qwen4_exp-execution.md) | current support, layer flow, PLE pipeline, MTP/Vision, continuation, resource planning and CUDA Graph |
| [Expert weight cache](maintainer/expert-weight-cache.md) | Host/GPU bank sources, replica cache, mixed CPU/GPU execution, streaming and graph handoffs |
| [DFlash and DFlash2](maintainer/dflash.md) | conditioning, masked draft computation, proposal distributions and backend state |
| [Constrained decoding](maintainer/constrained-decoding.md) | GBNF/JSON/choice/regex, tool policies, masks, speculative transactions and observations |
| [Resource scheduling and context cache](maintainer/resource-scheduling-and-context-cache.md) | core design for continuation, retention, incremental resources, preemption and recovery |
| [Paged KV context store](maintainer/paged-kv-cache.md) | typed pools, pages, replicas, address spaces, reservations and consumer views |
| [ReplaySSM GDN](maintainer/replayssm-gdn.md) | raw transition records and faithful commitment of the verified state prefix |
| [Op development](maintainer/op-development.md) | semantic boundaries, source ownership, numerical qualification and performance evidence |
| [Operational logging](maintainer/logging.md) | log ownership, presentation, severity and data policy |
| [Linear benchmarks](../bench/README.md) | pure Linear measurement, metrics and suites |
| [Linear tuning and reports](maintainer/linear-tuning.md) | tuning ranges, priority points, dispatch tradeoffs and final performance report format |
| [Upstream ports](maintainer/upstream-ports.md) | the fork's reviewed upstream point, carried features, divergences, dropped and deferred work, qualification and deployment |

Model cards contain official artifact facts and source provenance. The
[conversion guide](weight-conversion.md) is the entry point for making an artifact. Exact config
fields, parameter expansion and native supported domains are maintained by the code linked from
these references.

Qwen3.8-Flash-Next is the Qwen4Exp instance; Qwen3.8-27B uses the Qwen3.5 architecture.
The model references own mathematical and stored-representation facts, and the execution reference
owns integrated support and its limits. Concrete Op contracts live in `include/ninfer/ops/`.

## Planned work

[Qwen4 readiness](research/qwen4-readiness.md) separates Intel kernel evidence, upstream design
choices and release assumptions, and maps preserved fork work to pending dense Text/MTP/Vision
implementation and device qualification.

[Linear test maintenance](maintainer/linear-test-reorganization-plan.md) tracks the pending
consolidation of test runners and explicit case contracts.
