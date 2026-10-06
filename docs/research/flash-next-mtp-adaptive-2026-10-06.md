# Flash-Next adaptive MTP draft count (active, 2026-10-06)

This plan evaluates `--draft-policy adaptive` for Qwen3.8-Flash-Next MTP on v3. With that policy,
`--draft-tokens K` becomes a maximum: each round drafts, per request, the count with the most
expected committed tokens per unit of round cost under that request's decayed acceptance.

The idea came from an outside report: draft fewer tokens at long context, and bound the MTP
attention window. That code was read but not used. This implementation and its evaluation are
our own.

## Why the draft count, not the window

The baseline measurement used the Colab G4 (RTX PRO 6000 Server), branch `m3/6-mtp-adaptive` at
`7053c7af`, BF16 KV, greedy, thinking off.

**v3 decode is nearly flat in context.** Plain decode costs 7.47 ms per token at 2K and 7.51 ms at
32K. It runs at 127.8 tok/s at 128K against 135.3 at 20K. The MTP K=3 draft phase grows less
than 1% from 2K to 32K. Bounding the MTP window would therefore save little up to 128K. It is
deferred to the 1M-context work.

**The best fixed K depends on the task.** It does not depend on context length:

| Workload | K=0 | K=1 | K=2 | K=3 | K=4 | K=5 |
|---|---:|---:|---:|---:|---:|---:|
| iterative code editing, 8 turns to 21K context | 135.7 | 189.5 | 234.2 | 257.9 | 264.8 | **278.5** |
| short chat, 8 prompts | 134.4 | 176.4 | **196.1** | 192.3 | 178.2 | 170.2 |
| long-document summaries, 20K-128K | 135.3 | 157.5 | **160.3** | 149.6 | 134.0 | 121.7 |

Values are median decode tok/s.

Code editing keeps 0.93-0.99 acceptance even at 21K context. A schedule keyed on context length
would cut drafting where it pays most.

**Greedy MTP output equals ordinary output up to exact ties.** The first divergence in 3,000
tokens was at a tie of 0.0000 nats.

## Pre-registered evaluation (decided before any adaptive result)

**Build.** `m3/6-mtp-adaptive` with the adaptive policy, on one Colab G4, same artifact
(`1e7e026e...`), BF16 KV, context 147,456, prefill chunk 8192, CUDA Graphs on, prefix reuse on.

**Correctness**, required before the performance reading counts:
1. The real-artifact Engine suite passes, with the adaptive K≤5 arm in `mtp batch target parity`
   and in `mtp graphs and ragged batches`:
   - eager and graph output identical;
   - divergence from the ordinary target only at a tie of 0.05 nats or less.
2. The host policy test passes.
3. On the workloads, adaptive output equals K=0 output except where the first divergence lies at a
   K=0 tie of 0.05 nats or less. This is checked on the first code turn with top-2 logprobs.

**Performance.** One server per arm, run in the same session, on the three workloads above:
- arms: K=0, fixed K=1..5, adaptive K≤5;
- concurrency 1;
- plus a concurrency-4 mixed arm (two code-editing and two summary requests at once) for fixed
  K=2, K=3, K=5 and adaptive.

Metric: median decode tok/s per workload, and aggregate tok/s for the mixed arm.

**Acceptance criteria:**
- **A.** On each workload, adaptive is at least 98% of the best fixed K on that workload.
- **B.** Against fixed K=3, adaptive is at least 2% faster on at least two of the three workloads,
  and no more than 2% slower on any.
- **C.** In the mixed concurrency-4 arm, adaptive aggregate throughput is at least 98% of the best
  fixed K.

If A-C hold, `adaptive` becomes the documented recommendation for Flash-Next MTP. If any fails,
the result is reported as measured and `fixed` stays the default.

**Costs reported with the result:**
- extra graph memory (device memory after startup, adaptive against fixed K=5);
- extra startup time for capturing per-width verification graphs.
