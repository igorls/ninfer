# NInfer documentation

Start with the [project README](../README.md) to build NInfer, download a published artifact, and
run the CLI or HTTP server.

## User guides

| Document | Purpose |
|---|---|
| [Startup configuration](startup-configuration.md) | shared CLI/serve TOML schema, defaults, overrides, resource controls and startup display |
| [Deployment configurations](../examples/configs/) | RTX 5090, single Spark and dual-Spark TOML examples |
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

## Internal architecture

内部文档记录跨模块设计、重要数值取舍和开发方法。具体类型、接口、支持几何与 codec
以所属代码为准，不在文档维护重复声明。

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
| [GLM5Next model semantics](maintainer/glm5next-model.md) | architecture, complete mathematics, logical weights, Main/MTP/Vision and history dependencies |
| [GLM5Next quantization and numerics](maintainer/glm5next-quantization.md) | selected source/weight/cache representations, exact codecs, activation policy and rounding boundaries; implementation status belongs to the design index |
| [Expert weight cache](maintainer/expert-weight-cache.md) | Host/GPU bank sources, replica cache, mixed CPU/GPU execution, streaming and graph handoffs |
| [DFlash and DFlash2](maintainer/dflash.md) | conditioning, masked draft computation, proposal distributions and backend state |
| [Constrained decoding](maintainer/constrained-decoding.md) | GBNF/JSON/choice/regex, tool policies, masks, speculative transactions and observations |
| [Resource scheduling and context cache](maintainer/resource-scheduling-and-context-cache.md) | core design for continuation, retention, incremental resources, preemption and recovery |
| [Paged KV context store](maintainer/paged-kv-cache.md) | typed pools, pages, replicas, address spaces, reservations and consumer views |
| [ReplaySSM GDN](maintainer/replayssm-gdn.md) | raw transition records and faithful commitment of the verified state prefix |
| [Op development](maintainer/op-development.md) | semantic boundaries, source ownership, numerical qualification and performance evidence |
| [Operational logging](maintainer/observability.md) | log ownership, presentation, severity and data policy |
| [Linear benchmarks](../bench/README.md) | pure Linear measurement, metrics and suites |
| [Linear tuning and reports](maintainer/linear-tuning.md) | tuning ranges, priority points, dispatch tradeoffs and final performance report format |
| [Upstream ports](maintainer/upstream-ports.md) | the fork's reviewed upstream point, carried features, divergences, dropped and deferred work, qualification and deployment |

## Development tools

- [Tests](../tests/README.md)：测试组织、筛选与真实模型检查。
- [Benchmarks](../bench/README.md)：Engine/Op 微测、roofline 与外部 profiling。
- [Tools](../tools/README.md)：转换检查、Serve runners 与带宽工具入口。
- [Capability evaluation](../eval/README.md)：API 评测、并发与结果恢复。

## Open work

[未完成事项](issue-checklist.md)保留待处理问题与 MLA 性能工作。
Qwen4-27B 的[架构与量化](designs/qwen4-27b-rtx5090.md)及
[Op 需求](designs/qwen4-27b-op-checklist.md)属于独立适配计划。

[GLM5Next FP4-Local Op 清单](designs/glm5next-fp4-local-op-checklist.md)记录原生格式的
完整计算接入、实际形状、代码组织与逐点优化要求。

Qwen3.8-Flash-Next 使用 Qwen4Exp；Qwen3.8-27B 使用 Qwen3.5。

Shared infrastructure references: [Frontend](maintainer/frontend.md), [Context storage](maintainer/context-management.md), [weights and loading](maintainer/weights-and-loading.md), [speculative execution](maintainer/speculative-decoding.md). Fork representation and platform contracts remain in the detailed references above.
