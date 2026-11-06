# Tools

工具使用显式 artifact、executable 和输出目录；Python 不提供另一条模型推理路径。
维护环境为 Python 3.11，各工具按自身 requirements 使用已选环境，不在运行时隐式下载模型。
临时日志和实验产物放在 `.agent-tmp/<task>/`，`out/` 留给权重。

| 任务 | 入口 |
|---|---|
| 权重转换 | [convert](convert/)与[转换指南](../docs/weight-conversion.md) |
| Artifact 检查 | [artifact.inspect](artifact/inspect.py) |
| 已有 v2 产物升级工具 | [upgrade_ninfer_v2_to_v3.py](upgrade_ninfer_v2_to_v3.py) |
| Engine 与 Op 微测 | [bench](../bench/README.md) |
| 固定语料、并发与 TTFT | [tools/bench](bench/README.md) |
| 已运行服务的协议检查 | [serve_contract](smoke/serve_contract.py) |
| Thinking/history 场景 | [serve_thinking_preservation](smoke/serve_thinking_preservation.py) |

Artifact 的表示与物化关系见[权重与加载](../docs/maintainer/weights-and-loading.md)。
各命令参数以 `--help` 为准；managed runner 拥有其启动的服务，resident runner 只访问给定地址。

## Memory bandwidth probe

持续带宽 probe 是 CMake benchmark target，使用所选架构：

```bash
cmake --build build/rtx5090-dev -j --target ninfer_memory_bandwidth_bench
./build/rtx5090-dev/bench/ninfer_memory_bandwidth_bench
```

## Artifact workflow

The [conversion guide](../docs/weight-conversion.md) owns commands for official/custom recipes,
mixed sources, optional components, resources and sharding. The converter writes a `.ninfer`
artifact plus its `.conversion.json` report. Inspect a result with:

```bash
python3 -m tools.artifact.inspect out/qwen3_6_27b.ninfer --objects
```

Numeric formats, layouts and framing are defined by the references in the
[documentation map](../docs/README.md).

## Benchmark orchestration

`tools/bench/run_ninfer_bench_matrix.py` builds and runs the public-Engine benchmark matrix and
writes ignored local reports below `profiles/bench/`:

```bash
python3 tools/bench/run_ninfer_bench_matrix.py --build-dir build/rtx5090-dev --preset core --dry-run
python3 tools/bench/run_ninfer_bench_matrix.py --build-dir build/rtx5090-dev --preset core
```

See [`tools/bench/README.md`](bench/README.md) and [`bench/README.md`](../bench/README.md) for the
orchestrator and executable contracts.

For request-arrival latency, use the managed Qwen3.8-27B NVFP4/FP8 TTFT campaign. Its measurement
runner remains an external-only HTTP client; the separate controller owns Serve lifecycle and
artifacts. See [`tools/bench/ttft/README.md`](bench/ttft/README.md).

## Serving smoke

After starting `ninfer-serve` in another terminal:

```bash
python3 -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 \
  --model qwen3.6-27b
```

The client exercises OpenAI, Anthropic, rerank, streaming, usage, multimodal, and tool-call
response surfaces against the resident process.

For typed rewrite-checkpoint and thinking-history behavior, the managed smoke script launches a
real server and consumes the repository fixture:

```bash
python3 tools/smoke/serve_thinking_preservation.py \
  --server-bin build/rtx5090-dev/apps/ninfer-serve \
  --artifact out/qwen3_6_27b.ninfer --backend mtp
```
