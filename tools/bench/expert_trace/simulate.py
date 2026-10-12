"""Expert VRAM-budget cache simulator over `.nxtr` traces.

    python -m tools.bench.expert_trace.simulate TRACE [TRACE ...] [--budgets 4096,0.5,32GiB]
        [--policies ...] [--slots global|per-layer|both] [--request-log requests.jsonl] [--json OUT]

See README.md in this directory for the policies and the accounting rules.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import analysis, policies as pol
from .compiled import ClientLookup, CompiledTrace, PoolData, compile_trace
from .requestlog import load_request_log
from .trace import KIND_NAMES, TraceMetadata, TraceReader

GIB = 1 << 30
MIB = 1 << 20
DEFAULT_EXPERT_BYTES = 2_764_808
POLICY_ORDER = (
    "static-oracle",
    "static-train",
    "lru",
    "adaptive",
    "prefill-informed",
    "client-profile",
    "belady",
)
TEXT_ONLY = ("prefill-informed", "client-profile")


# ----------------------------------------------------------------------------- configuration


@dataclass
class SimConfig:
    budgets: list[int]
    policies: list[str] = field(default_factory=lambda: list(POLICY_ORDER))
    slots: list[str] = field(default_factory=lambda: ["global"])
    quota_from: str = "profile"
    train_fraction: float = 0.3
    expert_bytes: int = DEFAULT_EXPERT_BYTES
    mtp_budget: str = "proportional"
    params: pol.AdaptiveParams = field(default_factory=pol.AdaptiveParams)


def parse_budgets(spec: str, total_items: int, expert_bytes: int = DEFAULT_EXPERT_BYTES) -> list[int]:
    """Expert counts from a comma list of counts (`4096`), fractions (`0.5`, `50%`) or sizes
    (`32GiB`, `500MiB`, `20GB`)."""
    out: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        m = re.fullmatch(r"(\d*\.?\d+)\s*(gib|gb|mib|mb|%)?", token, re.IGNORECASE)
        if not m:
            raise ValueError(f"cannot parse budget {token!r} (use a count, a fraction, N%, or a size like 32GiB)")
        number, unit = float(m.group(1)), (m.group(2) or "").lower()
        if unit == "gib":
            count = number * GIB / expert_bytes
        elif unit == "gb":
            count = number * 1e9 / expert_bytes
        elif unit == "mib":
            count = number * MIB / expert_bytes
        elif unit == "mb":
            count = number * 1e6 / expert_bytes
        elif unit == "%":
            count = number / 100.0 * total_items
        elif "." in m.group(1):
            count = number * total_items
        else:
            count = number
        value = int(min(max(int(count), 1), total_items))
        if value not in out:
            out.append(value)
    if not out:
        raise ValueError("no budgets given")
    return out


def parse_policies(spec: str) -> list[str]:
    if spec.strip().lower() == "all":
        return list(POLICY_ORDER)
    names = [s.strip() for s in spec.split(",") if s.strip()]
    for name in names:
        if name not in POLICY_ORDER:
            raise ValueError(f"unknown policy {name!r}; choose from {', '.join(POLICY_ORDER)}")
    return [p for p in POLICY_ORDER if p in names]


# --------------------------------------------------------------------------------- simulation


def _metrics(t: pol.Tally, comp: CompiledTrace, split: int, expert_bytes: int, pool_name: str) -> dict:
    cols = t.decode_columns
    committed = int(comp.committed[split:].sum())
    misses = t.misses
    div = lambda x, d: (x / d) if d else None  # noqa: E731
    miss_bytes = misses * expert_bytes
    swap_bytes = t.swaps * expert_bytes
    out = {
        "decode_records": t.decode_records,
        "decode_columns": cols,
        "committed_tokens_est": committed,
        "needed": t.needed,
        "hits": t.hits,
        "misses": misses,
        "hit_rate": div(t.hits, t.needed),
        "misses_per_token": div(misses, cols),
        "misses_per_committed_token": div(misses, committed),
        "miss_mib_per_token": div(miss_bytes, cols * MIB) if cols else None,
        "miss_mib_per_committed_token": div(miss_bytes, committed * MIB) if committed else None,
        "swaps": t.swaps,
        "swap_mib_per_token": div(swap_bytes, cols * MIB) if cols else None,
        "swap_mib_per_committed_token": div(swap_bytes, committed * MIB) if committed else None,
        "prefill": {
            "records": t.prefill_records,
            "columns": t.prefill_columns,
            "needed": t.prefill_needed,
            "missing": t.prefill_missing,
            "missing_fraction": div(t.prefill_missing, t.prefill_needed),
            "missing_gib_per_chunk": div(t.prefill_missing * expert_bytes, t.prefill_records * GIB),
            "missing_mib_per_token": div(t.prefill_missing * expert_bytes, t.prefill_columns * MIB),
        },
    }
    if pool_name == "text" and comp.client_names:
        out["per_client"] = {
            comp.client_names[c]: {
                "decode_columns": int(t.client_columns[c]),
                "needed": int(t.client_needed[c]),
                "hits": int(t.client_hits[c]),
                "hit_rate": div(int(t.client_hits[c]), int(t.client_needed[c])),
                "prefill_tokens": int(t.client_pf_tokens[c]),
                "prefill_needed": int(t.client_pf_needed[c]),
                "prefill_missing": int(t.client_pf_missing[c]),
                "prefill_missing_fraction": div(int(t.client_pf_missing[c]), int(t.client_pf_needed[c])),
            }
            for c in range(len(comp.client_names))
            if t.client_needed[c] > 0 or t.client_pf_needed[c] > 0
        }
    return out


@dataclass
class _PoolContext:
    pool: PoolData
    train_freq: np.ndarray | None
    oracle_freq: np.ndarray
    client_freq: np.ndarray | None


def _pool_context(comp: CompiledTrace, pool: PoolData, split: int) -> _PoolContext:
    train = pol.decode_records(comp, pool, 0, split) if split > 0 else np.empty(0, dtype=np.int64)
    ev = pol.decode_records(comp, pool, split)
    return _PoolContext(
        pool=pool,
        train_freq=pol.record_frequency(pool, train) if len(train) else None,
        oracle_freq=pol.record_frequency(pool, ev),
        client_freq=pol.client_profiles(comp, pool, 0, split)
        if (comp.client_names and pool.name == "text" and split > 0)
        else None,
    )


def _run_policy(name, comp, ctx: _PoolContext, layout, split, params) -> pol.Tally | str:
    pool = ctx.pool
    if name == "static-oracle":
        return pol.run_static(comp, pool, pol.select_top(ctx.oracle_freq, layout), split)
    if name == "static-train":
        if ctx.train_freq is None:
            return "needs --train-fraction > 0"
        return pol.run_static(comp, pool, pol.select_top(ctx.train_freq, layout), split)
    if name == "lru":
        return pol.run_lru(comp, pool, layout, split)
    if name == "belady":
        return pol.run_belady(comp, pool, layout, split)
    seed = pol.select_top(ctx.train_freq, layout) if ctx.train_freq is not None else None
    if name == "adaptive":
        return pol.run_adaptive(comp, pool, layout, split, params, seed)
    if name in TEXT_ONLY and pool.name != "text":
        return "text pool only"
    if name == "prefill-informed":
        if not comp.prefill_lists:
            return "no prefill expert profiles in the trace (needs prefill records before decode)"
        return pol.run_adaptive(comp, pool, layout, split, params, seed, variant=name)
    if name == "client-profile":
        if not comp.client_names:
            return "needs client labels (--request-log or synthetic metadata)"
        if ctx.client_freq is None:
            return "needs --train-fraction > 0 for per-client profiles"
        return pol.run_adaptive(comp, pool, layout, split, params, seed, variant=name, client_freq=ctx.client_freq)
    raise ValueError(name)


def run_simulation(comp: CompiledTrace, cfg: SimConfig, log=lambda *_: None) -> dict:
    split = comp.split_index(cfg.train_fraction)
    total_text = comp.text.n_items
    contexts = {pool.name: _pool_context(comp, pool, split) for pool in comp.pools()}
    mtp_resident = cfg.mtp_budget == "resident"
    results = []
    for mode in cfg.slots:
        for budget in cfg.budgets:
            entry = {
                "slots": mode,
                "budget_experts": budget,
                "budget_fraction": budget / total_text,
                "budget_gib": budget * cfg.expert_bytes / GIB,
                "pools": {},
            }
            for pool in comp.pools():
                ctx = contexts[pool.name]
                if pool.name == "mtp":
                    if mtp_resident:
                        entry["pools"]["mtp"] = {"resident": True, "pool_items": pool.n_items, "budget_slots": pool.n_items, "policies": {}}
                        continue
                    pool_budget = max(1, round(budget / total_text * pool.n_items))
                else:
                    pool_budget = budget
                scores = ctx.train_freq if cfg.quota_from == "profile" else None
                layout = pol.make_layout(pool, mode, pool_budget, scores)
                block = {
                    "resident": False,
                    "pool_items": pool.n_items,
                    "budget_slots": layout.budget,
                    "quota": layout.quota.tolist() if layout.n_groups > 1 else None,
                    "policies": {},
                }
                for name in cfg.policies:
                    started = time.perf_counter()
                    outcome = _run_policy(name, comp, ctx, layout, split, cfg.params)
                    if isinstance(outcome, str):
                        block["policies"][name] = {"skipped": outcome}
                    else:
                        block["policies"][name] = _metrics(outcome, comp, split, cfg.expert_bytes, pool.name)
                    elapsed = time.perf_counter() - started
                    if elapsed >= 2.0:
                        log(f"  {mode} budget={budget} {pool.name}/{name}: {elapsed:.1f}s")
                entry["pools"][pool.name] = block
            results.append(entry)
            log(f"simulated {mode} budget={budget}")

    ev_text = pol.decode_records(comp, comp.text, split)
    return {
        "window": {
            "records_total": comp.n_records,
            "requests_total": comp.n_requests,
            "train_fraction": cfg.train_fraction,
            "split_record": split,
            "train_records": split,
            "eval_records": comp.n_records - split,
            "eval_text_decode_rounds": int(len(ev_text)),
            "eval_text_decode_columns": int(comp.ncols[ev_text].sum()) if len(ev_text) else 0,
            "eval_committed_tokens_est": int(comp.committed[split:].sum()),
        },
        "budgets": results,
    }


# -------------------------------------------------------------------------------- trace input


def _kind_summary(comp: CompiledTrace) -> dict:
    out = {}
    for kind, name in KIND_NAMES.items():
        sel = comp.kinds == kind
        out[name] = {"records": int(sel.sum()), "columns": int(comp.ncols[sel].sum())}
    return out


def _iter_traces(paths: Sequence[str], statuses: list[tuple[str, str, str]]) -> tuple[TraceMetadata, Iterator]:
    first = TraceReader(paths[0])
    meta = first.metadata

    def generate():
        readers = [first] + [TraceReader(p) for p in paths[1:]]
        try:
            for reader in readers:
                m = reader.metadata
                same = (m.text_layers, m.experts, m.top_k, m.mtp_layer, m.server_instance_id) == (
                    meta.text_layers, meta.experts, meta.top_k, meta.mtp_layer, meta.server_instance_id,
                )
                if not same:
                    raise SystemExit(f"{reader.path}: geometry or server_instance_id differs from {paths[0]}")
                yield from reader.records()
                statuses.append((str(reader.path), reader.status, reader.status_detail))
        finally:
            for reader in readers:
                reader.close()

    return meta, generate()


def synthetic_clients(meta: TraceMetadata) -> dict[int, str]:
    info = meta.raw.get("synthetic")
    if not isinstance(info, dict):
        return {}
    return {int(k): str(v) for k, v in info.get("request_clients", {}).items()}


def load_compiled(args, log=lambda *_: None) -> tuple[CompiledTrace, dict]:
    statuses: list[tuple[str, str, str]] = []
    meta, stream = _iter_traces(args.trace, statuses)
    info: dict = {"traces": list(args.trace), "metadata": {k: v for k, v in meta.raw.items() if k != "synthetic"}}
    mapping: dict[int, str] = {}
    if args.request_log:
        instance = args.server_instance_id or meta.server_instance_id
        if not instance:
            raise SystemExit("the trace has no server_instance_id; pass --server-instance-id")
        since = meta.created_unix_ms - 300_000 if meta.created_unix_ms else None
        started = time.perf_counter()
        join = load_request_log(
            args.request_log,
            instance,
            group_by=[s for s in args.group_by.split(",") if s],
            strip_version=args.client_strip_version,
            since_unix_ms=since,
            tail_bytes=int(args.request_log_tail_mib * MIB) if args.request_log_tail_mib else None,
        )
        mapping = join.labels
        info["request_log"] = {"path": str(args.request_log), "server_instance_id": instance, **join.summary()}
        log(f"request log: {len(mapping)} requests labelled ({time.perf_counter() - started:.1f}s)")
    else:
        mapping = synthetic_clients(meta)
        if mapping:
            info["request_log"] = {"source": "synthetic metadata", "requests_labelled": len(mapping)}
    lookup = ClientLookup(mapping) if (mapping or args.request_log) else None
    prefill_topk = args.prefill_promote if "prefill-informed" in args.policy_list else 0
    started = time.perf_counter()
    # the reader status is only known once the stream is drained, so patch it in afterwards
    comp = compile_trace(
        stream, meta, client_lookup=lookup, prefill_topk=prefill_topk, max_records=args.max_records
    )
    problems = [s for s in statuses if s[1] != "complete"]
    if problems:
        comp.status = "truncated" if any(s[1] == "truncated" for s in problems) else "corrupt"
        comp.status_detail = "; ".join(f"{path}: {detail}" for path, _s, detail in problems)
    log(f"compiled {comp.n_records} records in {time.perf_counter() - started:.1f}s")
    if lookup is not None:
        info["clients"] = list(comp.client_names)
        tagged = comp.req_ids[comp.req_ids != 0]  # request id 0 is an untagged column
        info["requests_tagged"] = int(tagged.size)
        info["requests_joined"] = int(np.isin(tagged, np.fromiter(mapping, dtype=np.uint64, count=len(mapping))).sum())
    info["status"] = comp.status
    info["status_detail"] = comp.status_detail
    info["records_by_kind"] = _kind_summary(comp)
    info["requests"] = comp.n_requests
    return comp, info


# ----------------------------------------------------------------------------------- output


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"{type(obj)} is not JSON serialisable")


def _fmt(value, spec: str, none: str = "-") -> str:
    return none if value is None else format(value, spec)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h)) for i, h in enumerate(headers)]
    line = lambda cells: "  ".join(  # noqa: E731
        str(c).ljust(widths[i]) if i == 0 else str(c).rjust(widths[i]) for i, c in enumerate(cells)
    )
    return [line(headers), "  ".join("-" * w for w in widths), *(line(r) for r in rows)]


def render_text(report: dict) -> str:
    lines: list[str] = []
    info = report["trace"]
    meta = info["metadata"]
    lines.append("== Expert trace simulation ==")
    lines.append(
        f"trace: {', '.join(info['traces'])}  model={meta.get('model_id')}  "
        f"{meta.get('text_layers')} text layers x {meta.get('experts')} experts, top-{meta.get('top_k')}, "
        f"mtp_layer={meta.get('mtp_layer')}"
    )
    lines.append(f"server_instance_id={meta.get('server_instance_id')}  artifact_id={str(meta.get('artifact_id'))[:16]}")
    if info["status"] != "complete":
        lines.append(f"WARNING: trace is {info['status']}: {info['status_detail']} (analysis uses the complete records)")
    kinds = ", ".join(f"{k}={v['records']} rec/{v['columns']} col" for k, v in info["records_by_kind"].items() if v["records"])
    lines.append(f"records: {kinds}; {info['requests']} requests")
    if "request_log" in info:
        rl = info["request_log"]
        if "path" in rl:
            lines.append(
                f"request log: {info.get('requests_joined', 0)} of {info.get('requests_tagged', 0)} tagged traced requests joined "
                f"({rl['requests_labelled']} requests in the log for this instance; "
                f"group by {','.join(rl['group_by'])}; scanned {rl['lines_scanned']} lines)"
            )
        else:
            lines.append(f"clients: {rl['source']} ({rl['requests_labelled']} requests)")
    w = report["window"]
    lines.append(
        f"window: records [0,{w['split_record']}) train / [{w['split_record']},{w['records_total']}) evaluated "
        f"(train fraction {w['train_fraction']}); evaluated text decode rounds={w['eval_text_decode_rounds']} "
        f"columns={w['eval_text_decode_columns']} committed tokens (est.)={w['eval_committed_tokens_est']}"
    )
    cfg = report["config"]
    lines.append(
        f"expert bytes={cfg['expert_bytes']}  adaptive: decay={cfg['params']['decay']} interval={cfg['params']['interval']} "
        f"max-swaps={cfg['params']['max_swaps']} min-count={cfg['params']['min_count']} min-gain={cfg['params']['min_gain']} "
        f"latency={cfg['params']['latency']}"
    )
    lines.append(
        "tok = decode/verify/proposal column (token position executed); ctok = committed token (estimated from position progress);"
    )
    lines.append("miss MiB = host-served expert bytes; swap MiB = bytes promoted into VRAM; pf = prefill chunks (streamed, not cached).")

    if report.get("skew"):
        lines += ["", "== Skew (decode activations, whole trace) =="]
        for pool_name, s in report["skew"].items():
            cov = s["coverage"]
            lines.append(f"[{pool_name}] {s['records']} decode records, experts per layer needed to cover:")
            rows = [[k, v["min"], f"{v['median']:.0f}", v["max"], f"{v['mean']:.1f}"] for k, v in cov.items()]
            lines += ["  " + t for t in _table(["coverage", "min", "median", "max", "mean"], rows)]
            d = s["distinct_per_record_layer"]
            lines.append(
                f"  distinct experts per (record, layer): mean {d['mean']:.2f}, median {d['median']}, p95 {d['p95']}, max {d['max']}; "
                f"per-layer mean min/median/max {d['per_layer_mean']['min']:.2f}/{d['per_layer_mean']['median']:.2f}/{d['per_layer_mean']['max']:.2f}"
            )
            if d["by_columns"]:
                parts = [f"{c} col: {v['mean']:.1f} (p95 {v['p95']})" for c, v in d["by_columns"].items()]
                lines.append("  by columns per record: " + "; ".join(parts))

    if report.get("prefill"):
        lines += ["", "== Prefill chunks (whole trace) =="]
        for pool_name, s in report["prefill"].items():
            kinds = ", ".join(f"{k}: {v['chunks']} chunks/{v['tokens']} tok" for k, v in s["by_kind"].items())
            d = s["distinct_per_record_layer"]
            lines.append(
                f"[{pool_name}] {s['chunks']} chunks, {s['tokens']} tokens ({kinds}); distinct experts touched per "
                f"(chunk, layer) of {s['experts_per_layer']}: mean {d['mean']:.1f}, median {d['median']}, p95 {d['p95']}, max {d['max']}"
            )
            for k, v in d["by_kind"].items():
                lines.append(f"  {k}: mean {v['mean']:.1f} (p95 {v['p95']}) over {v['cells']} chunk-layers")

    for entry in report["simulation"]["budgets"]:
        header = (
            f"budget {entry['budget_experts']} experts "
            f"({entry['budget_fraction'] * 100:.1f}% of text experts, {entry['budget_gib']:.1f} GiB), slots={entry['slots']}"
        )
        for pool_name, block in entry["pools"].items():
            lines += ["", f"== {header} - {pool_name} layers =="]
            if block["resident"]:
                lines.append("MTP layer kept fully resident (--mtp-budget resident): no misses by construction.")
                continue
            if pool_name == "mtp":
                lines.append(f"MTP pool: {block['budget_slots']} of {block['pool_items']} slots (same fraction as the text pool)")
            rows = []
            show_ctok = any(
                (m.get("committed_tokens_est") or 0) not in (0, m.get("decode_columns"))
                for m in block["policies"].values()
                if "skipped" not in m
            )
            skipped = []
            for name, m in block["policies"].items():
                if "skipped" in m:
                    skipped.append(f"{name}: {m['skipped']}")
                    continue
                pf = m["prefill"]
                row = [
                    name,
                    _fmt(None if m["hit_rate"] is None else m["hit_rate"] * 100, ".2f"),
                    _fmt(m["misses_per_token"], ".3f"),
                ]
                if show_ctok:
                    row.append(_fmt(m["misses_per_committed_token"], ".3f"))
                row += [
                    _fmt(m["miss_mib_per_token"], ".1f"),
                    _fmt(m["swap_mib_per_token"], ".2f"),
                    _fmt(None if pf["missing_fraction"] is None else pf["missing_fraction"] * 100, ".1f"),
                    _fmt(pf["missing_gib_per_chunk"], ".2f"),
                ]
                rows.append(row)
            headers = ["policy", "hit%", "miss/tok"] + (["miss/ctok"] if show_ctok else [])
            headers += ["missMiB/tok", "swapMiB/tok", "pf-miss%", "pf GiB/chunk"]
            lines += _table(headers, rows)
            lines += [f"skipped - {note}" for note in skipped]
            if "adaptive" in block["policies"] and "client-profile" in block["policies"]:
                a, c = block["policies"]["adaptive"], block["policies"]["client-profile"]
                if "skipped" not in a and "skipped" not in c and a["hit_rate"] is not None:
                    delta = (c["hit_rate"] - a["hit_rate"]) * 100
                    verdict = "beats" if delta > 0.05 else "ties" if delta >= -0.05 else "does not beat"
                    lines.append(
                        f"client-profile {verdict} adaptive: hit rate {delta:+.2f} points, "
                        f"swap MiB/tok {c['swap_mib_per_token']:.2f} vs {a['swap_mib_per_token']:.2f}"
                    )

        text = entry["pools"].get("text")
        if text and not text["resident"]:
            clients = sorted(
                {c for m in text["policies"].values() if "per_client" in m for c in m["per_client"]},
                key=lambda c: -max(m["per_client"][c]["decode_columns"] for m in text["policies"].values() if "per_client" in m and c in m["per_client"]),
            )[: report["config"]["max_clients"]]
            if clients:
                names = [n for n, m in text["policies"].items() if "per_client" in m]
                rows = []
                for c in clients:
                    first = next(m["per_client"][c] for m in text["policies"].values() if "per_client" in m and c in m["per_client"])
                    row = [c[:40], first["decode_columns"]]
                    for n in names:
                        pc = text["policies"][n]["per_client"].get(c)
                        row.append(_fmt(None if not pc or pc["hit_rate"] is None else pc["hit_rate"] * 100, ".2f"))
                    rows.append(row)
                lines += ["", f"-- per-client hit% (text pool, {header}) --"]
                lines += _table(["client", "tokens"] + names, rows)

    overlap = report.get("client_overlap")
    if overlap and overlap.get("clients"):
        lines += ["", "== Cross-client overlap of per-layer top-N expert sets (Jaccard, mean over layers) =="]
        lines.append("clients: " + "; ".join(f"[{i}] {c[:40]} ({n} tok)" for i, (c, n) in enumerate(zip(overlap["clients"], overlap["decode_columns"]))))
        lines.append("diagonal = the client's first half vs second half of its traffic (self-similarity ceiling)")
        for n, s in overlap["by_top_n"].items():
            lines.append(
                f"top-{n}: within-client {_fmt(s['within_client_split_half_mean'], '.3f')}  "
                f"cross-client {_fmt(s['cross_client_mean'], '.3f')}  random {s['random_expectation']:.3f}"
            )
            rows = [[f"[{i}]"] + [_fmt(v, ".3f") for v in row] for i, row in enumerate(s["matrix"])]
            lines += ["  " + t for t in _table([""] + [f"[{j}]" for j in range(len(s["matrix"]))], rows)]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tools.bench.expert_trace.simulate",
        description="Simulate expert VRAM budgets and caching policies on an expert-routing trace.",
    )
    p.add_argument("trace", nargs="+", help=".nxtr trace file(s) of one server instance, read in order")
    p.add_argument("--budgets", default="0.25,0.5,0.75", help="expert counts, fractions of the text experts or sizes, e.g. 4096,0.5,32GiB")
    p.add_argument("--policies", default="all", help=f"comma list of {', '.join(POLICY_ORDER)}")
    p.add_argument("--no-belady", action="store_true", help="skip the offline Belady policy (the slowest)")
    p.add_argument("--slots", choices=("global", "per-layer", "both"), default="global")
    p.add_argument("--quota-from", choices=("profile", "uniform"), default="profile", help="per-layer quotas: from the training profile, or equal")
    p.add_argument("--train-fraction", type=float, default=0.3, help="first fraction of requests (by first appearance) trains profiles; the rest is evaluated")
    p.add_argument("--expert-bytes", type=int, default=DEFAULT_EXPERT_BYTES)
    p.add_argument("--mtp-budget", choices=("proportional", "resident"), default="proportional", help="simulate the MTP layer at the same fraction, or keep it fully resident")
    ad = pol.AdaptiveParams()
    p.add_argument("--decay", type=float, default=ad.decay)
    p.add_argument("--interval", type=int, default=ad.interval)
    p.add_argument("--max-swaps", type=int, default=ad.max_swaps)
    p.add_argument("--min-count", type=float, default=ad.min_count)
    p.add_argument("--min-gain", type=float, default=ad.min_gain)
    p.add_argument("--copy-latency-rounds", type=int, default=ad.latency, help="extra rounds before a promoted expert is usable (adaptive family)")
    p.add_argument("--prefill-promote", type=int, default=ad.prefill_promote, help="experts promoted per new request by prefill-informed")
    p.add_argument("--lock-fraction", type=float, default=ad.lock_fraction, help="client-profile: pool fraction locked to the active clients' top experts")
    p.add_argument("--client-window", type=int, default=ad.client_window, help="client-profile: a client is active for this many rounds after its last record")
    p.add_argument("--request-log", help="serve request log (JSONL) to label requests by client")
    p.add_argument("--server-instance-id", help="override the trace's server_instance_id for the log join")
    p.add_argument("--group-by", default="client", help="request-log fields forming the client label, comma separated (client, protocol, tools_digest, enable_thinking, tools, model, ...)")
    p.add_argument("--client-strip-version", action="store_true", help="drop the version after '/' in User-Agent labels")
    p.add_argument("--request-log-tail-mib", type=float, default=0, help="read only the last N MiB of the request log")
    p.add_argument("--max-records", type=int, help="use only the first N trace records")
    p.add_argument("--max-clients", type=int, default=8)
    p.add_argument("--min-client-columns", type=int, default=200, help="minimum decode tokens for a client to enter the overlap analysis")
    p.add_argument("--overlap-top", default="8,16,32,64", help="top-N sizes per layer for the overlap analysis")
    p.add_argument("--no-skew", action="store_true", help="skip the skew report")
    p.add_argument("--json", metavar="OUT", help="write the full result as JSON")
    p.add_argument("--quiet", action="store_true", help="no progress on stderr")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    log = (lambda *_: None) if args.quiet else (lambda msg: print(msg, file=sys.stderr, flush=True))
    try:
        args.policy_list = [p for p in parse_policies(args.policies) if not (args.no_belady and p == "belady")]
    except ValueError as error:
        parser.error(str(error))
    comp, info = load_compiled(args, log)
    if comp.n_records == 0:
        print("trace has no complete records", file=sys.stderr)
        return 1
    try:
        budgets = parse_budgets(args.budgets, comp.text.n_items, args.expert_bytes)
    except ValueError as error:
        parser.error(str(error))
    params = pol.AdaptiveParams(
        decay=args.decay, interval=args.interval, max_swaps=args.max_swaps, min_count=args.min_count,
        min_gain=args.min_gain, latency=args.copy_latency_rounds, prefill_promote=args.prefill_promote,
        lock_fraction=args.lock_fraction, client_window=args.client_window,
    )
    cfg = SimConfig(
        budgets=budgets, policies=args.policy_list,
        slots=["global", "per-layer"] if args.slots == "both" else [args.slots],
        quota_from=args.quota_from, train_fraction=args.train_fraction, expert_bytes=args.expert_bytes,
        mtp_budget=args.mtp_budget, params=params,
    )
    report: dict = {"trace": info}
    report["config"] = {
        "budgets": budgets, "policies": cfg.policies, "slots": cfg.slots, "quota_from": cfg.quota_from,
        "train_fraction": cfg.train_fraction, "expert_bytes": cfg.expert_bytes, "mtp_budget": cfg.mtp_budget,
        "params": params.__dict__, "max_clients": args.max_clients,
    }
    if not args.no_skew:
        started = time.perf_counter()
        report["skew"] = analysis.skew_report(comp)
        report["prefill"] = analysis.prefill_report(comp)
        log(f"skew report: {time.perf_counter() - started:.1f}s")
    if comp.client_names:
        tops = tuple(int(x) for x in args.overlap_top.split(",") if x)
        report["client_overlap"] = analysis.client_overlap(comp, tops, args.min_client_columns, args.max_clients)
    simulation = run_simulation(comp, cfg, log)
    report["window"] = simulation["window"]
    report["simulation"] = simulation
    print(render_text(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=_json_default), encoding="utf-8")
        log(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
