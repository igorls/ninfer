"""Policy-independent trace statistics: expert skew and cross-client overlap."""

from __future__ import annotations

import numpy as np

from .compiled import MIXED, CompiledTrace, PoolData
from .policies import _blocks, decode_records, flat_entry_index, prefill_records
from .trace import KIND_NAMES, PREFILL_KINDS


def _usage(pool: PoolData, recs: np.ndarray, limit: int = 1 << 23) -> np.ndarray:
    """(column, expert) use counts per item over the given records."""
    off = pool.entries.offsets
    usage = np.zeros(pool.n_items, dtype=np.float64)
    if len(recs) == 0:
        return usage
    for a, b in _blocks(off[recs + 1] - off[recs], limit):
        idx, _ = flat_entry_index(off, recs[a:b])
        usage += np.bincount(
            pool.entries.items[idx], weights=pool.entries.counts[idx].astype(np.float64), minlength=pool.n_items
        )
    return usage


def _quantile_from_hist(hist: np.ndarray, q: float) -> int:
    total = int(hist.sum())
    if total == 0:
        return 0
    return int(np.searchsorted(np.cumsum(hist), q * total, side="left"))


def _distinct_per_record_layer(comp: CompiledTrace, pool: PoolData, recs: np.ndarray, group_values: np.ndarray, label) -> dict:
    """Distribution of distinct experts per (record, layer) over `recs`, overall and grouped by
    `label(group_values[record])` (at most the 8 largest groups)."""
    layers, experts = pool.n_layers, pool.experts
    hist = np.zeros(experts + 1, dtype=np.int64)
    by_group: dict[int, np.ndarray] = {}
    layer_sum = np.zeros(layers, dtype=np.float64)
    layer_cells = np.zeros(layers, dtype=np.float64)
    off = pool.entries.offsets
    lo_layer = comp.first_layer.astype(np.int64) - pool.layer_lo
    hi_layer = lo_layer + comp.layer_count.astype(np.int64)
    for a, b in _blocks(off[recs + 1] - off[recs], 1 << 22):
        block = recs[a:b]
        idx, lens = flat_entry_index(off, block)
        layer = pool.entries.items[idx].astype(np.int64) // experts
        seg = np.repeat(np.arange(len(block)), lens)
        matrix = np.bincount(seg * layers + layer, minlength=len(block) * layers).reshape(len(block), layers)
        lane = np.arange(layers)[None, :]
        covered = (lane >= lo_layer[block][:, None]) & (lane < hi_layer[block][:, None])
        hist += np.bincount(matrix[covered], minlength=experts + 1)
        layer_sum += np.where(covered, matrix, 0).sum(axis=0)
        layer_cells += covered.sum(axis=0)
        groups = group_values[block]
        for g in np.unique(groups).tolist():
            sel = groups == g
            h = by_group.setdefault(int(g), np.zeros(experts + 1, dtype=np.int64))
            h += np.bincount(matrix[sel][covered[sel]], minlength=experts + 1)
    cells = int(hist.sum())
    mean_all = float((hist * np.arange(experts + 1)).sum() / cells) if cells else 0.0
    per_layer_mean = np.divide(layer_sum, layer_cells, out=np.zeros(layers), where=layer_cells > 0)
    distinct = {
        "mean": mean_all,
        "median": _quantile_from_hist(hist, 0.5),
        "p95": _quantile_from_hist(hist, 0.95),
        "max": int(np.flatnonzero(hist).max()) if cells else 0,
        "per_layer_mean": {
            "min": float(per_layer_mean.min()),
            "median": float(np.median(per_layer_mean)),
            "max": float(per_layer_mean.max()),
        },
        "by_group": {},
    }
    ranked = sorted(by_group.items(), key=lambda kv: -int(kv[1].sum()))[:8]
    for g, h in sorted(ranked):
        n = int(h.sum())
        distinct["by_group"][label(g)] = {
            "cells": n,
            "mean": float((h * np.arange(experts + 1)).sum() / n),
            "median": _quantile_from_hist(h, 0.5),
            "p95": _quantile_from_hist(h, 0.95),
        }
    return distinct


def skew_report(comp: CompiledTrace, coverage=(0.5, 0.8, 0.95)) -> dict[str, dict]:
    """Per pool: experts needed to cover X% of decode activations, and distinct experts per record."""
    out: dict[str, dict] = {}
    for pool in comp.pools():
        recs = decode_records(comp, pool)
        if len(recs) == 0:
            continue
        layers, experts = pool.n_layers, pool.experts
        usage = _usage(pool, recs).reshape(layers, experts)
        totals = usage.sum(axis=1)
        ordered = -np.sort(-usage, axis=1)
        cum = np.cumsum(ordered, axis=1)
        cover: dict[str, dict] = {}
        for p in coverage:
            with np.errstate(invalid="ignore", divide="ignore"):
                need = (cum < (p * totals)[:, None]).sum(axis=1) + 1
            need = np.where(totals > 0, need, 0)
            used = need[totals > 0]
            cover[f"{int(round(p * 100))}%"] = {
                "min": int(used.min()) if used.size else 0,
                "median": float(np.median(used)) if used.size else 0.0,
                "max": int(used.max()) if used.size else 0,
                "mean": float(used.mean()) if used.size else 0.0,
                "per_layer": need.tolist(),
            }

        distinct = _distinct_per_record_layer(comp, pool, recs, comp.ncols, str)
        distinct["by_columns"] = distinct.pop("by_group")
        out[pool.name] = {
            "records": int(len(recs)),
            "activations": float(totals.sum()),
            "coverage": cover,
            "distinct_per_record_layer": distinct,
        }
    return out


def prefill_report(comp: CompiledTrace) -> dict[str, dict]:
    """Per pool: prefill chunks by kind (per-token rows or histogram) and the distinct experts
    each chunk touches per layer, which is what streaming a chunk from host memory has to move."""
    out: dict[str, dict] = {}
    for pool in comp.pools():
        recs = prefill_records(comp, pool)
        if len(recs) == 0:
            continue
        kinds = comp.kinds[recs]
        distinct = _distinct_per_record_layer(
            comp, pool, recs, comp.kinds, lambda kind: KIND_NAMES.get(int(kind), f"kind{kind}")
        )
        out[pool.name] = {
            "chunks": int(len(recs)),
            "tokens": int(comp.ncols[recs].sum()),
            "by_kind": {
                KIND_NAMES[k]: {"chunks": int((kinds == k).sum()), "tokens": int(comp.ncols[recs][kinds == k].sum())}
                for k in PREFILL_KINDS
                if (kinds == k).any()
            },
            "distinct_per_record_layer": {**distinct, "by_kind": distinct["by_group"]},
            "experts_per_layer": pool.experts,
        }
        del out[pool.name]["distinct_per_record_layer"]["by_group"]
    return out


# ---------------------------------------------------------------------------------- clients


def _client_usage(comp: CompiledTrace, min_columns: int, max_clients: int):
    pool = comp.text
    nc = len(comp.client_names)
    dec = decode_records(comp, pool)
    rc = comp.rec_client[dec]
    recs_of: dict[int, list[int]] = {c: dec[rc == c].tolist() for c in range(nc)}
    seg_of: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {c: [] for c in range(nc)}
    columns = {c: int(comp.ncols[dec[rc == c]].sum()) for c in range(nc)}
    for r in dec[rc == MIXED].tolist():
        for c, items, counts, ccols in pool.mixed.get(r, ()):
            seg_of[c].append((r, items, counts))
            recs_of[c].append(r)
            columns[c] += ccols
    ranked = [c for c in sorted(range(nc), key=lambda c: -columns[c]) if columns[c] >= min_columns and len(recs_of[c]) >= 2]
    ranked = ranked[:max_clients]
    usage: dict[int, dict[str, np.ndarray]] = {}
    for c in ranked:
        all_recs = np.asarray(sorted(recs_of[c]), dtype=np.int64)
        mid = int(all_recs[len(all_recs) // 2])
        single = dec[rc == c]
        halves = {"first": single[single < mid], "second": single[single >= mid]}
        u = {name: _usage(pool, rr) for name, rr in halves.items()}
        for r, items, counts in seg_of[c]:
            u["first" if r < mid else "second"][items] += counts
        u["all"] = u["first"] + u["second"]
        usage[c] = u
    return ranked, usage, columns


def _top_masks(usage: np.ndarray, layers: int, experts: int, n: int) -> np.ndarray:
    rows = usage.reshape(layers, experts)
    n = min(n, experts)
    idx = np.argpartition(-rows, n - 1, axis=1)[:, :n]
    mask = np.zeros((layers, experts), dtype=bool)
    np.put_along_axis(mask, idx, True, axis=1)
    return mask & (rows > 0)


def _jaccard_layers(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Jaccard index of two [layers, experts] masks per layer (NaN where both are empty)."""
    inter = (a & b).sum(axis=1)
    union = (a | b).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / np.maximum(union, 1), np.nan)


def client_overlap(comp: CompiledTrace, top_ns=(8, 16, 32, 64), min_columns: int = 200, max_clients: int = 8) -> dict:
    """Jaccard overlap of per-layer top-N expert sets between clients.

    The diagonal is the same client's first half versus second half (by time): the stationarity
    ceiling a per-client profile can reach. Off-diagonal values near it mean the clients are not
    distinguishable by their expert use. `random` is the expected Jaccard of random N-subsets.
    """
    if not comp.client_names:
        return {}
    pool = comp.text
    top_ns = [n for n in top_ns if 0 < n < pool.experts]
    ranked, usage, columns = _client_usage(comp, min_columns, max_clients)
    result = {
        "clients": [comp.client_names[c] for c in ranked],
        "decode_columns": [columns[c] for c in ranked],
        "min_columns": min_columns,
        "by_top_n": {},
    }
    layers, experts = pool.n_layers, pool.experts
    for n in top_ns:
        masks = {c: _top_masks(usage[c]["all"], layers, experts, n) for c in ranked}
        halves = {c: (_top_masks(usage[c]["first"], layers, experts, n), _top_masks(usage[c]["second"], layers, experts, n)) for c in ranked}
        per_layer = [
            [_jaccard_layers(*halves[ci]) if i == j else _jaccard_layers(masks[ci], masks[cj]) for j, cj in enumerate(ranked)]
            for i, ci in enumerate(ranked)
        ]
        matrix = [[float(np.nanmean(v)) if np.isfinite(v).any() else float("nan") for v in row] for row in per_layer]
        off_diag = [matrix[i][j] for i in range(len(ranked)) for j in range(len(ranked)) if i != j and matrix[i][j] == matrix[i][j]]
        diag = [matrix[i][i] for i in range(len(ranked)) if matrix[i][i] == matrix[i][i]]
        result["by_top_n"][str(n)] = {
            "matrix": matrix,
            "per_layer": [[[None if np.isnan(x) else float(x) for x in v] for v in row] for row in per_layer],
            "within_client_split_half_mean": float(np.mean(diag)) if diag else None,
            "cross_client_mean": float(np.mean(off_diag)) if off_diag else None,
            "random_expectation": n / (2 * experts - n),
        }
    return result
