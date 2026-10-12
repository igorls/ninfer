# Expert trace analysis

Offline analysis of expert-routing traces (`.nxtr`) from the MoE text model: how many routed experts
must stay in VRAM, which caching policy serves a given budget best, and whether different clients
use different experts. The tool reads a trace plus, optionally, the serve request log. It never
runs the engine.

```bash
python -m tools.bench.expert_trace.simulate TRACE.nxtr --budgets 4096,0.5,32GiB \
    --slots both --request-log requests.jsonl --json out.json
python -m tools.bench.expert_trace.synthetic demo.nxtr --requests 200 --clients 3 --mtp
python -m pytest -q tests/bench/test_expert_trace.py -p no:cacheprovider
```

Python 3.11 and numpy only.

## Trace format v1 (`.nxtr`, little-endian)

```text
"NXTRACE1" | u32 metadata_length | metadata JSON (UTF-8) | record*
```

Metadata carries `format: "ninfer-expert-trace"`, `version: 1`, `server_instance_id`, `artifact_id`,
`model_id`, `text_layers` (48), `mtp_layer` (48, or `null` without MTP), `experts` (512), `top_k`
(10) and `created_unix_ms`.

Record header, 24 bytes: `u32 sentinel 0x43525452 | u8 kind | u8 reserved | u16 layer_count |
u16 first_layer | u16 reserved | u32 columns | u64 timestamp_unix_us`. Kinds: 1 prefill chunk, 2
ordinary decode round, 3 MTP proposal step, 4 MTP verify. The layer range is read from the header
(`first_layer`, `layer_count`); text records normally cover layers 0..47 and MTP proposals layer 48.

Then `columns` x 16 bytes `{u64 request_id, u32 position, u32 flags}` (`request_id`: the request-log
`request.request_id` of the column's request, 0 when untagged), then `columns x layer_count x
top_k` `u16` expert ids in `[column][layer][rank]` order, rank 0 being the best router score. All
columns of a record ran in one forward unit.

A truncated or corrupt final record (a crash) ends the stream at the last complete record; the
reader reports `status` (`complete`, `truncated`, `corrupt`) and the report prints a warning.
`trace.TraceReader` streams records as numpy arrays; `trace.TraceWriter` writes the format for tests
and synthetic traces.

## Request-log join

The server tags every Engine request with its request-log id, and the Engine copies it into each
column, so the trace's `request_id` is the request log's `request.request_id` within the same
`server_instance_id`; the join is direct (no engine-id indirection):

| Log line | Fields used |
|---|---|
| every line | `server_instance_id` (only lines equal to the trace's id are used), `timestamp_unix_ms` |
| `request_start`, `request_done`, `request_error` | `request.request_id` = the trace `request_id`, `request.client` (HTTP `User-Agent`), `request.protocol`, `request.tools_digest`, `request.enable_thinking`, `request.tool_count`, ... |

`request_rejected` events are not used (their requests never reach the Engine and carry no
client), nor are `generation.engine_request_id` and `request_scheduling`. Request id 0 means
"untagged" and forms the client `(untagged)`; ids that are missing from the log are `(unjoined)`.
The client label is `--group-by` fields joined with `|` (default `client`; others: `protocol`,
`tools_digest`, `enable_thinking`, `tools`, ...); `--client-strip-version` removes the version from a
`User-Agent`. The log is read as a stream; a bisection on `timestamp_unix_ms` starts the scan shortly
before the trace's `created_unix_ms` instead of at the beginning of a multi-gigabyte file, and
`--request-log-tail-mib` reads only the last N MiB. Synthetic traces carry their client map in the
metadata and need no log.

## Simulation model

* A **pool** is a set of equal-size items (an expert of a layer) competing for `budget` slots.
  The text layers form one pool (24,576 items; budgets are in experts). The MTP layer is a second,
  separate pool: `--mtp-budget proportional` (default) gives it the same fraction of its 512
  experts, `resident` keeps it fully resident and reports no misses for it.
* **Slots:** `--slots global` lets any slot hold any layer's expert. `per-layer` fixes a quota per
  layer; quotas follow the training profile's best N (`--quota-from profile`) or are equal
  (`uniform`), and are the same for every policy.
* **Hit accounting:** a record needs the union of the experts its columns routed to, per layer. Each
  needed expert that is not resident is one miss per record and layer, however many columns use it.
  Misses are served from host memory (`miss MiB`). Residency changes only between records.
* **Tokens:** `tok` is a decode/verify/proposal *column* (an executed position). `ctok` is a
  *committed* token, estimated per request from the progress of the first position between its
  consecutive decode units (clipped to the unit width), so it needs no acceptance data.
* **Prefill** records never change residency (a large chunk touches almost every expert). They are
  measured against the resident set at that moment: `pf-miss%` is the share of the experts they
  touch that are not resident, `pf GiB/chunk` the bytes that would be streamed per chunk.
* **Window:** the first `--train-fraction` of the requests (by first appearance, default 0.3) train
  the static profiles; all policies are evaluated on the remaining records. Dynamic policies still
  run through the training part to warm up. With `--train-fraction 0` the whole trace is evaluated
  and `static-train`, `client-profile` and `prefill-informed` are skipped.
* Static policies start fully loaded and pay no load cost; dynamic policies start empty (LRU,
  Belady) or from the `static-train` set (`adaptive` and variants).
* `swap MiB` is the bytes promoted into VRAM: every admitted miss for LRU/Belady, every planned
  promotion for the adaptive family.

## Policies

| Policy | What it does |
|---|---|
| `static-oracle` | Top-N by record frequency (number of decode records needing the expert) over the **evaluated window**: a hindsight upper bound for any static profile on exactly that window. With a train split this deliberately differs from "the whole trace", so `static-oracle >= static-train` holds. |
| `static-train` | The same ranking from the training split only. |
| `lru` | Least recently used per pool (per layer in `per-layer` mode); every miss is admitted, evicting experts the current record does not need. |
| `adaptive` | Decayed usage counts: every round `count *= --decay` then `+1` per (column, expert) use; every `--interval` rounds at most `--max-swaps` promotions pair the highest-count absent experts with the lowest-count residents (empty slots first) when the candidate count is at least `--min-count` and beats the victim by `--min-gain`. Defaults 0.7, 4, 96, 2.0, 1.5. A promotion decided after round `k` is usable from round `k + 1 + --copy-latency-rounds` (default 0). Seeded from `static-train`. |
| `prefill-informed` | `adaptive`, plus at a request's first decode round the up-to `--prefill-promote` experts its own prefill used most are promoted, displacing the lowest-count residents the current round does not need (no gain test), usable immediately. Tests whether a prompt predicts its decode experts. |
| `client-profile` | `adaptive`, plus `--lock-fraction` of every layer's slots hold the top experts (summed per-client record frequency from the training split) of the clients active in the last `--client-window` rounds, re-evaluated at each decision; locked experts are never victims. Reports whether it beats `adaptive` (hit rate, swap traffic). |
| `belady` | Offline farthest-next-use replacement with bypass: after each record keep, among the resident and just-missed experts, those needed soonest. Optimal among demand-admitted caches (verified against exhaustive search), so a lower bound on LRU-class misses over the **whole** run. Policies that preload or prefetch (static, adaptive) are not bounded by it. It is the slowest policy; leave it out with `--no-belady` or `--policies`. |

## Reports

Per budget and slot mode, per pool: hit rate, misses per token, miss MiB per token, swap MiB per
token, prefill miss share and streaming GiB per chunk. With client labels: hit rate per client and
policy, and the Jaccard overlap of per-layer top-N expert sets between clients (diagonal: a client's
first half of traffic against its second half, the ceiling a per-client profile can reach; the
random-subset expectation is printed for scale). A skew report lists, per layer, the experts needed
to cover 50/80/95% of decode activations (min/median/max over layers) and the distinct experts per
record and layer, also by batch width. `--json OUT` holds everything.

## Cost

The compiled trace keeps the unique experts of every record (4 bytes per expert per record, about
2 KiB per single-column decode round). Dynamic policies cost tens to a few hundred microseconds per
record, Belady and `per-layer` mode the most; use `--max-records N` to bound a run.
