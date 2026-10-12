"""Synthetic expert-routing traces for tests and demos.

The generator models what real MoE routing looks like at the level the cache simulators care
about, not the model itself:

* every layer has a Zipf-like popularity over its experts (a random permutation per layer);
* clients prefer partially different experts: a fraction of each client's preferred set comes
  from the popular head, the rest from the tail, so clients overlap on the head and differ on
  the tail;
* a request has a small focus set per layer that drifts slowly (temporal locality within a
  request);
* each token's top-k experts are a Gumbel top-k sample of the resulting logits, rank 0 first.

The trace follows an engine-like schedule: requests arrive with exponential gaps, a request is
prefilled in chunks when admitted (up to `max_batch` concurrent), then decoded one column per
round, or with MTP as `draft_window` proposal steps plus one verify of `draft_window + 1`
columns per request.

    python -m tools.bench.expert_trace.synthetic OUT.nxtr --requests 200 --clients 3 --mtp
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .trace import (
    KIND_DECODE,
    KIND_MTP_PROPOSAL,
    KIND_MTP_VERIFY,
    KIND_PREFILL,
    TraceWriter,
    make_metadata,
)


@dataclass
class SyntheticConfig:
    text_layers: int = 48
    experts: int = 512
    top_k: int = 10
    mtp: bool = False
    draft_window: int = 3
    accept_rate: float = 0.65
    clients: int = 3
    requests: int = 60
    zipf_s: float = 1.0
    client_pref_size: int = 48
    client_pref_boost: float = 5.0
    client_shared: float = 0.5
    focus_size: int = 24
    focus_boost: float = 3.0
    focus_refresh_tokens: int = 48
    focus_refresh_fraction: float = 0.25
    noise: float = 1.0
    prefill_noise: float = 1.5
    prompt_len: tuple[int, int] = (128, 1024)
    decode_len: tuple[int, int] = (48, 256)
    prefill_chunk: int = 512
    max_batch: int = 4
    arrival_gap_rounds: float = 12.0
    seed: int = 0
    server_instance_id: str = "synthetic-0"
    created_unix_ms: int = 1_790_000_000_000


@dataclass
class _Request:
    request_id: int
    client: int
    prompt_len: int
    decode_len: int
    arrival: float
    pos: int = 0
    produced: int = 0
    focus: np.ndarray | None = None
    logits: np.ndarray | None = None  # [layers(+mtp), experts] base logits for this request
    tokens_since_refresh: int = 0


@dataclass
class SyntheticTrace:
    config: SyntheticConfig
    client_names: list[str]
    requests: list[_Request]
    base: np.ndarray = field(default=None, repr=False)  # [clients, layers(+mtp), experts] logits

    def metadata(self) -> dict:
        c = self.config
        return make_metadata(
            text_layers=c.text_layers,
            experts=c.experts,
            top_k=c.top_k,
            mtp_layer=c.text_layers if c.mtp else None,
            server_instance_id=c.server_instance_id,
            artifact_id="synthetic",
            model_id="synthetic",
            created_unix_ms=c.created_unix_ms,
            synthetic={
                "seed": c.seed,
                "clients": self.client_names,
                "request_clients": {str(r.request_id): self.client_names[r.client] for r in self.requests},
            },
        )

    def write(self, path: str | Path) -> int:
        with TraceWriter(path, self.metadata()) as writer:
            for kind, first_layer, req_ids, positions, ids, ts in self.records():
                writer.write_record(kind, req_ids, positions, ids, first_layer=first_layer, timestamp_us=ts)
            return writer.records_written

    def request_log_lines(self, decoy_instance: str | None = "serve-other-1") -> list[str]:
        """JSONL in the serve request-log shape: one start and one done event per request whose
        `request.request_id` is the trace's request id (an unrelated `engine_request_id` is
        included to show it is not used), plus decoy lines of another server instance."""
        base_ms = self.config.created_unix_ms + 1000
        lines = []
        for i, r in enumerate(self.requests):
            request = {
                "request_id": r.request_id,
                "http_request_id": f"req_{i:08x}",
                "client": self.client_names[r.client],
                "protocol": "openai_chat_completions",
                "tools_digest": "",
                "enable_thinking": False,
            }
            common = {"artifact_type": "ninfer_serve_request_log", "schema_version": 25}
            lines.append(
                json.dumps({**common, "event": "request_start", "request": request,
                            "server_instance_id": self.config.server_instance_id, "timestamp_unix_ms": base_ms + i})
            )
            lines.append(
                json.dumps({**common, "event": "request_done", "request": request,
                            "generation": {"engine_request_id": r.request_id + 5000},
                            "server_instance_id": self.config.server_instance_id, "timestamp_unix_ms": base_ms + i + 1})
            )
            if decoy_instance:
                other = dict(request, client="decoy/9")
                lines.append(
                    json.dumps({**common, "event": "request_done", "request": other,
                                "generation": {"engine_request_id": r.request_id + 5000},
                                "server_instance_id": decoy_instance, "timestamp_unix_ms": base_ms + i + 1})
                )
        return lines

    # ----------------------------------------------------------------------------- schedule

    def records(self) -> Iterator[tuple[int, int, np.ndarray, np.ndarray, np.ndarray, int]]:
        """Yield (kind, first_layer, request_ids, positions, ids, timestamp_us) in engine order."""
        c = self.config
        rng = np.random.default_rng(c.seed + 1)
        layers = c.text_layers + (1 if c.mtp else 0)
        self._reset(layers)
        clock_us = c.created_unix_ms * 1000
        queue = deque(sorted(self.requests, key=lambda r: r.arrival))
        active: list[_Request] = []
        round_no = 0.0
        while queue or active:
            while queue and queue[0].arrival <= round_no and len(active) < c.max_batch:
                request = queue.popleft()
                for rec in self._prefill(request, rng, layers):
                    clock_us += 90_000
                    yield (*rec, clock_us)
                active.append(request)
            if not active:
                round_no = queue[0].arrival
                continue
            if c.mtp:
                for step in range(c.draft_window):
                    pos = np.array([r.pos + 1 + step for r in active], dtype=np.uint32)
                    ids = self._sample(active, rng, layers, rows=slice(c.text_layers, layers), noise=c.noise)
                    clock_us += 3_000
                    yield (KIND_MTP_PROPOSAL, c.text_layers, self._ids(active), pos, ids, clock_us)
                width = c.draft_window + 1
                req_ids = np.repeat(self._ids(active), width)
                positions = np.concatenate([np.arange(r.pos, r.pos + width) for r in active]).astype(np.uint32)
                ids = self._sample(active, rng, layers, rows=slice(0, c.text_layers), noise=c.noise, repeat=width)
                clock_us += 28_000
                yield (KIND_MTP_VERIFY, 0, req_ids, positions, ids, clock_us)
                for r in active:
                    accepted = 0
                    while accepted < c.draft_window and rng.random() < c.accept_rate:
                        accepted += 1
                    self._advance(r, accepted + 1, rng, layers)
            else:
                pos = np.array([r.pos for r in active], dtype=np.uint32)
                ids = self._sample(active, rng, layers, rows=slice(0, c.text_layers), noise=c.noise)
                clock_us += 22_000
                yield (KIND_DECODE, 0, self._ids(active), pos, ids, clock_us)
                for r in active:
                    self._advance(r, 1, rng, layers)
            active = [r for r in active if r.produced < r.decode_len]
            round_no += 1

    def _reset(self, layers: int) -> None:
        """Restart every request, so that `records()` is repeatable."""
        rng = np.random.default_rng(self.config.seed + 2)
        for r in self.requests:
            r.pos = r.produced = r.tokens_since_refresh = 0
            r.focus = rng.integers(0, self.config.experts, size=(layers, self.config.focus_size))
            self._rebuild_logits(r, layers)

    @staticmethod
    def _ids(active: list[_Request]) -> np.ndarray:
        return np.array([r.request_id for r in active], dtype=np.uint64)

    def _advance(self, request: _Request, tokens: int, rng, layers: int) -> None:
        request.pos += tokens
        request.produced += tokens
        request.tokens_since_refresh += tokens
        if request.tokens_since_refresh >= self.config.focus_refresh_tokens:
            request.tokens_since_refresh = 0
            self._refresh_focus(request, rng, layers)

    def _refresh_focus(self, request: _Request, rng, layers: int) -> None:
        c = self.config
        swap = max(1, int(c.focus_size * c.focus_refresh_fraction))
        slots = rng.integers(0, c.focus_size, size=(layers, swap))
        request.focus[np.arange(layers)[:, None], slots] = rng.integers(0, c.experts, size=(layers, swap))
        self._rebuild_logits(request, layers)

    def _rebuild_logits(self, request: _Request, layers: int) -> None:
        c = self.config
        logits = self.base[request.client].copy()
        logits[np.arange(layers)[:, None], request.focus] += np.log(c.focus_boost)
        request.logits = logits

    def _sample(self, active, rng, layers: int, *, rows: slice, noise: float, repeat: int = 1) -> np.ndarray:
        """Gumbel top-k expert ids [len(active) * repeat, rows, top_k], rank 0 first."""
        base = np.stack([r.logits[rows] for r in active])
        if repeat > 1:
            base = np.repeat(base, repeat, axis=0)
        return self._topk(base, rng, noise)

    def _topk(self, base: np.ndarray, rng, noise: float) -> np.ndarray:
        k = self.config.top_k
        logits = base + noise * rng.gumbel(size=base.shape).astype(np.float32)
        part = np.argpartition(-logits, k - 1, axis=-1)[..., :k]
        vals = np.take_along_axis(logits, part, axis=-1)
        order = np.argsort(-vals, axis=-1)
        return np.take_along_axis(part, order, axis=-1).astype(np.uint16)

    def _prefill(self, request: _Request, rng, layers: int):
        c = self.config
        text = slice(0, c.text_layers)
        done = 0
        while done < request.prompt_len:
            n = min(c.prefill_chunk, request.prompt_len - done)
            parts = []
            for lo in range(0, n, 64):
                m = min(64, n - lo)
                base = np.broadcast_to(request.logits[text], (m, c.text_layers, c.experts))
                parts.append(self._topk(base, rng, c.prefill_noise))
            ids = np.concatenate(parts)
            positions = np.arange(done, done + n, dtype=np.uint32)
            yield (KIND_PREFILL, 0, np.full(n, request.request_id, dtype=np.uint64), positions, ids)
            done += n
        request.pos = request.prompt_len


def build_synthetic(config: SyntheticConfig | None = None, **overrides) -> SyntheticTrace:
    cfg = replace(config or SyntheticConfig(), **overrides)
    rng = np.random.default_rng(cfg.seed)
    layers = cfg.text_layers + (1 if cfg.mtp else 0)
    experts = cfg.experts

    # per-layer Zipf popularity over a random permutation of the experts
    rank = np.argsort(rng.random((layers, experts)), axis=1)  # rank[l, j] = expert at popularity rank j
    log_weight = np.empty((layers, experts), dtype=np.float32)
    zipf = -cfg.zipf_s * np.log(np.arange(1, experts + 1))
    np.put_along_axis(log_weight, rank, np.broadcast_to(zipf, (layers, experts)).astype(np.float32), axis=1)

    # client preference offsets: part from the popular head, part from the tail
    pref = min(cfg.client_pref_size, experts // 2)
    shared = int(round(cfg.client_shared * pref))
    head = min(4 * pref, experts)
    base = np.repeat(log_weight[None], max(cfg.clients, 1), axis=0)
    for ci in range(max(cfg.clients, 1)):
        for layer in range(layers):
            head_pick = rng.choice(rank[layer, :head], size=min(shared, head), replace=False)
            tail_pool = rank[layer, head:] if experts > head else rank[layer]
            tail_pick = rng.choice(tail_pool, size=min(pref - len(head_pick), len(tail_pool)), replace=False)
            base[ci, layer, np.concatenate([head_pick, tail_pick])] += np.log(cfg.client_pref_boost)

    # request plan
    names = [f"client-{chr(ord('a') + i % 26)}/{1 + i // 26}" for i in range(max(cfg.clients, 1))]
    assignment = np.arange(cfg.requests) % len(names)
    rng.shuffle(assignment)
    gaps = rng.exponential(cfg.arrival_gap_rounds, size=cfg.requests)
    arrivals = np.cumsum(gaps)
    ids = 1000 + np.cumsum(1 + rng.integers(0, 3, size=cfg.requests))
    requests = []
    for i in range(cfg.requests):
        requests.append(
            _Request(
                request_id=int(ids[i]),
                client=int(assignment[i]),
                prompt_len=int(rng.integers(cfg.prompt_len[0], cfg.prompt_len[1] + 1)),
                decode_len=int(rng.integers(cfg.decode_len[0], cfg.decode_len[1] + 1)),
                arrival=float(arrivals[i]),
            )
        )
    return SyntheticTrace(cfg, names, requests, base)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", help="output .nxtr path")
    p.add_argument("--request-log", help="also write a matching request log (JSONL)")
    d = SyntheticConfig()
    p.add_argument("--requests", type=int, default=d.requests)
    p.add_argument("--clients", type=int, default=d.clients)
    p.add_argument("--layers", type=int, default=d.text_layers)
    p.add_argument("--experts", type=int, default=d.experts)
    p.add_argument("--top-k", type=int, default=d.top_k)
    p.add_argument("--mtp", action="store_true")
    p.add_argument("--draft-window", type=int, default=d.draft_window)
    p.add_argument("--max-batch", type=int, default=d.max_batch)
    p.add_argument("--zipf", type=float, default=d.zipf_s)
    p.add_argument("--prompt-len", type=int, nargs=2, default=list(d.prompt_len))
    p.add_argument("--decode-len", type=int, nargs=2, default=list(d.decode_len))
    p.add_argument("--seed", type=int, default=d.seed)
    args = p.parse_args(argv)
    trace = build_synthetic(
        SyntheticConfig(
            text_layers=args.layers, experts=args.experts, top_k=args.top_k, mtp=args.mtp,
            draft_window=args.draft_window, clients=args.clients, requests=args.requests,
            max_batch=args.max_batch, zipf_s=args.zipf, prompt_len=tuple(args.prompt_len),
            decode_len=tuple(args.decode_len), seed=args.seed, server_instance_id=f"synthetic-{args.seed}",
        )
    )
    count = trace.write(args.out)
    print(f"wrote {count} records to {args.out}")
    if args.request_log:
        Path(args.request_log).write_text("\n".join(trace.request_log_lines()) + "\n", encoding="utf-8")
        print(f"wrote request log to {args.request_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
