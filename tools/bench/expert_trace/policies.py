"""Expert-cache policies over compiled traces.

Model. A *pool* is a set of items (an expert of a layer) of equal size competing for `budget`
slots. In `global` slot mode any slot holds any item; in `per-layer` mode every layer owns a fixed
quota. A record is a lookup of the union of its columns' experts: needed items that are resident
are hits, the others are misses (served from host memory), each counted once per record. Residency
changes only between records; an expert promoted after record `k` is usable from record
`k + 1 + latency`. Prefill records are streamed through: they are measured against the resident
set but never change it.

All metrics accumulate only for records at or after `split` (the evaluation window); earlier
records warm the dynamic policies up and produce the training profiles.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .compiled import MIXED, CompiledTrace, PoolData
from .trace import KIND_PREFILL

NEG_INF = float("-inf")
POS_INF = float("inf")


# ----------------------------------------------------------------------------------- layout


@dataclass(frozen=True, eq=False)
class SlotLayout:
    mode: str  # "global" | "per-layer"
    n_items: int
    n_groups: int
    width: int  # items per group; item = group * width + offset
    quota: np.ndarray  # int64 [n_groups]

    @property
    def budget(self) -> int:
        return int(self.quota.sum())

    @property
    def group_of_item(self) -> np.ndarray:
        return (np.arange(self.n_items) // self.width).astype(np.int32)


def make_layout(
    pool: PoolData, mode: str, budget: int, quota_scores: np.ndarray | None = None
) -> SlotLayout:
    """Slot layout for `budget` slots. Per-layer quotas follow `quota_scores` (the number of the
    globally best-scoring items that fall into each layer) or are uniform without scores."""
    n = pool.n_items
    budget = int(max(0, min(budget, n)))
    if mode == "global" or pool.n_layers == 1:
        return SlotLayout(mode, n, 1, n, np.asarray([budget], dtype=np.int64))
    groups, width = pool.n_layers, pool.experts
    if quota_scores is not None and np.any(quota_scores > 0):
        best = np.argsort(-quota_scores, kind="stable")[:budget]
        quota = np.bincount(best // width, minlength=groups).astype(np.int64)
    else:
        base, extra = divmod(budget, groups)
        quota = base + (np.arange(groups) < extra).astype(np.int64)
    return SlotLayout(mode, n, groups, width, quota)


def select_top(scores: np.ndarray, layout: SlotLayout) -> np.ndarray:
    """Boolean mask of the `quota[g]` best-scoring items of every group (ties by lower index)."""
    g, w = layout.n_groups, layout.width
    rows = scores.reshape(g, w)
    order = np.argsort(-rows, axis=1, kind="stable")
    take = np.arange(w)[None, :] < layout.quota[:, None]
    mask = np.zeros((g, w), dtype=bool)
    np.put_along_axis(mask, order, take, axis=1)
    return mask.reshape(-1)


# ----------------------------------------------------------------------------------- entries


def _blocks(lengths: np.ndarray, limit: int):
    """Split a record list into runs whose entry total stays near `limit`."""
    cum = np.cumsum(lengths)
    start, count = 0, len(lengths)
    while start < count:
        base = int(cum[start - 1]) if start else 0
        end = int(np.searchsorted(cum, base + limit, side="right"))
        end = max(end, start + 1)
        yield start, min(end, count)
        start = end


def flat_entry_index(offsets: np.ndarray, recs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = offsets[recs]
    lens = offsets[recs + 1] - starts
    total = int(lens.sum())
    shift = starts - (np.cumsum(lens) - lens)
    return np.repeat(shift, lens) + np.arange(total), lens


def record_hits(pool: PoolData, resident: np.ndarray, recs: np.ndarray, limit: int = 1 << 23):
    """(hits, needed) per record in `recs` against a fixed resident mask."""
    off = pool.entries.offsets
    hits = np.empty(len(recs), dtype=np.int64)
    needed = np.empty(len(recs), dtype=np.int64)
    lens_all = off[recs + 1] - off[recs]
    for a, b in _blocks(lens_all, limit):
        idx, lens = flat_entry_index(off, recs[a:b])
        hit = resident[pool.entries.items[idx]]
        seg = np.repeat(np.arange(b - a), lens)
        hits[a:b] = np.bincount(seg, weights=hit, minlength=b - a).astype(np.int64)
        needed[a:b] = lens
    return hits, needed


def record_frequency(pool: PoolData, recs: np.ndarray, limit: int = 1 << 23) -> np.ndarray:
    """Number of the given records that need each item."""
    off = pool.entries.offsets
    freq = np.zeros(pool.n_items, dtype=np.float64)
    if len(recs) == 0:
        return freq
    lens_all = off[recs + 1] - off[recs]
    for a, b in _blocks(lens_all, limit):
        idx, _ = flat_entry_index(off, recs[a:b])
        freq += np.bincount(pool.entries.items[idx], minlength=pool.n_items)
    return freq


def decode_records(comp: CompiledTrace, pool: PoolData, lo: int = 0, hi: int | None = None) -> np.ndarray:
    """Record indices (decode, verify and MTP proposal) that touch the pool, within [lo, hi)."""
    lens = pool.entries.lengths()
    hi = comp.n_records if hi is None else hi
    sel = (lens > 0) & (comp.kinds != KIND_PREFILL)
    sel[:lo] = False
    sel[hi:] = False
    return np.flatnonzero(sel)


def prefill_records(comp: CompiledTrace, pool: PoolData, lo: int = 0) -> np.ndarray:
    sel = (pool.entries.lengths() > 0) & (comp.kinds == KIND_PREFILL)
    sel[:lo] = False
    return np.flatnonzero(sel)


# ----------------------------------------------------------------------------------- tally


class Tally:
    """Counters of one policy run over the evaluation window."""

    def __init__(self, n_clients: int):
        self.decode_records = 0
        self.decode_columns = 0
        self.needed = 0
        self.hits = 0
        self.swaps = 0  # experts copied into the pool (promotions)
        self.prefill_records = 0
        self.prefill_columns = 0
        self.prefill_needed = 0
        self.prefill_missing = 0
        self.client_columns = np.zeros(n_clients, dtype=np.int64)
        self.client_needed = np.zeros(n_clients, dtype=np.int64)
        self.client_hits = np.zeros(n_clients, dtype=np.int64)

    @property
    def misses(self) -> int:
        return self.needed - self.hits


class _Recorder:
    """Per-record accounting shared by the sequential policies."""

    def __init__(self, comp: CompiledTrace, pool: PoolData, split: int, tally: Tally):
        self.ncols = comp.ncols.tolist()
        self.kinds = comp.kinds.tolist()
        self.client = comp.rec_client.tolist()
        self.mixed = pool.mixed
        self.split = split
        self.tally = tally
        self.has_clients = len(comp.client_names) > 0

    def decode(self, r: int, resident: np.ndarray, hits: int, needed: int) -> None:
        if r < self.split:
            return
        t = self.tally
        cols = self.ncols[r]
        t.decode_records += 1
        t.decode_columns += cols
        t.needed += needed
        t.hits += hits
        if self.has_clients:
            c = self.client[r]
            if c >= 0:
                t.client_columns[c] += cols
                t.client_needed[c] += needed
                t.client_hits[c] += hits
            elif c == MIXED and r in self.mixed:
                for cc, items, _counts, ccols in self.mixed[r]:
                    t.client_columns[cc] += ccols
                    t.client_needed[cc] += len(items)
                    t.client_hits[cc] += int(np.count_nonzero(resident[items]))

    def prefill(self, r: int, hits: int, needed: int) -> None:
        if r < self.split:
            return
        t = self.tally
        t.prefill_records += 1
        t.prefill_columns += self.ncols[r]
        t.prefill_needed += needed
        t.prefill_missing += needed - hits


# ----------------------------------------------------------------------------------- static


def run_static(comp: CompiledTrace, pool: PoolData, resident: np.ndarray, split: int) -> Tally:
    """A fixed resident set, evaluated vectorised over the evaluation window."""
    tally = Tally(len(comp.client_names))
    dec = decode_records(comp, pool, split)
    if len(dec):
        hits, needed = record_hits(pool, resident, dec)
        tally.decode_records = len(dec)
        tally.decode_columns = int(comp.ncols[dec].sum())
        tally.needed = int(needed.sum())
        tally.hits = int(hits.sum())
        nc = len(comp.client_names)
        if nc:
            rc = comp.rec_client[dec].astype(np.int64)
            single = rc >= 0
            if single.any():
                tally.client_needed += np.bincount(rc[single], weights=needed[single], minlength=nc).astype(np.int64)
                tally.client_hits += np.bincount(rc[single], weights=hits[single], minlength=nc).astype(np.int64)
                tally.client_columns += np.bincount(
                    rc[single], weights=comp.ncols[dec][single], minlength=nc
                ).astype(np.int64)
            for r in dec[rc == MIXED].tolist():
                for c, items, _counts, ccols in pool.mixed.get(r, ()):
                    tally.client_columns[c] += ccols
                    tally.client_needed[c] += len(items)
                    tally.client_hits[c] += int(np.count_nonzero(resident[items]))
    pre = prefill_records(comp, pool, split)
    if len(pre):
        hits, needed = record_hits(pool, resident, pre)
        tally.prefill_records = len(pre)
        tally.prefill_columns = int(comp.ncols[pre].sum())
        tally.prefill_needed = int(needed.sum())
        tally.prefill_missing = int((needed - hits).sum())
    return tally


# ----------------------------------------------------------------------------------- planning


def plan_swaps(
    cand_score: np.ndarray,
    cand_ok: np.ndarray,
    vict_score: np.ndarray,
    vict_ok: np.ndarray,
    free: np.ndarray,
    layout: SlotLayout,
    *,
    max_swaps: int,
    min_cand: float,
    min_gain: float,
    free_score: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair the best candidates with the worst victims inside every group.

    Free slots act as victims with score `free_score` and come first. A pair is kept when the
    candidate scores at least `min_cand` and beats its victim by at least `min_gain`; at most
    `max_swaps` pairs (largest gain first) are returned across all groups.
    Returns (candidate items, victim items); a victim of -1 means an empty slot.
    """
    g, w = layout.n_groups, layout.width
    k = min(int(max_swaps), w)
    if k <= 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    cs = np.where(cand_ok, cand_score, NEG_INF).reshape(g, w)
    if k < w:
        ci = np.argpartition(-cs, k - 1, axis=1)[:, :k]
    else:
        ci = np.broadcast_to(np.arange(w), (g, w)).copy()
    csel = np.take_along_axis(cs, ci, axis=1)
    order = np.argsort(-csel, axis=1, kind="stable")
    ci = np.take_along_axis(ci, order, axis=1)
    csel = np.take_along_axis(csel, order, axis=1)

    vs = np.where(vict_ok, vict_score, POS_INF).reshape(g, w)
    if k < w:
        vi = np.argpartition(vs, k - 1, axis=1)[:, :k]
    else:
        vi = np.broadcast_to(np.arange(w), (g, w)).copy()
    vsel = np.take_along_axis(vs, vi, axis=1)
    order = np.argsort(vsel, axis=1, kind="stable")
    vi = np.take_along_axis(vi, order, axis=1)
    vsel = np.take_along_axis(vsel, order, axis=1)

    j = np.arange(k)[None, :]
    free_row = np.minimum(free, k)[:, None]
    src = np.clip(j - free_row, 0, k - 1)
    is_free = j < free_row
    v_score = np.where(is_free, free_score, np.take_along_axis(vsel, src, axis=1))
    v_idx = np.where(is_free, -1, np.take_along_axis(vi, src, axis=1))

    with np.errstate(invalid="ignore"):
        gain = csel - v_score
        valid = np.isfinite(csel) & (csel >= min_cand) & (v_score < POS_INF) & (gain >= min_gain)
    base = (np.arange(g) * w)[:, None]
    cand_flat = (base + ci)[valid].astype(np.int64)
    vict_flat = np.where(v_idx >= 0, base + v_idx, -1)[valid].astype(np.int64)
    if len(cand_flat) > max_swaps:
        best = np.argpartition(-gain[valid], max_swaps - 1)[:max_swaps]
        cand_flat, vict_flat = cand_flat[best], vict_flat[best]
    return cand_flat, vict_flat


def _group_counts(mask: np.ndarray, layout: SlotLayout) -> np.ndarray:
    if layout.n_groups == 1:
        return np.asarray([int(np.count_nonzero(mask))], dtype=np.int64)
    return np.bincount(np.flatnonzero(mask) // layout.width, minlength=layout.n_groups).astype(np.int64)


# ----------------------------------------------------------------------------------- LRU


class _GlobalLru:
    """Exact LRU over one pool of `capacity` slots using an append-only touch queue.

    Each touch appends (item, stamp); an entry is current only while the item's last stamp still
    equals it. Victims are popped from the head, skipping stale entries, in vectorised chunks.
    """

    def __init__(self, n_items: int, capacity: int):
        self.capacity = capacity
        self.last = np.full(n_items, -1, dtype=np.int64)
        self.resident = np.zeros(n_items, dtype=bool)
        self.count = 0
        size = max(8 * capacity, 1 << 14)
        self.qi = np.empty(size, dtype=np.int32)
        self.qs = np.empty(size, dtype=np.int64)
        self.head = 0
        self.tail = 0

    def touch(self, items: np.ndarray, stamp: int) -> None:
        m = len(items)
        if self.tail + m > len(self.qi):
            self._compact(m)
        self.last[items] = stamp
        self.qi[self.tail : self.tail + m] = items
        self.qs[self.tail : self.tail + m] = stamp
        self.tail += m

    def _compact(self, extra: int) -> None:
        live = np.flatnonzero(self.resident)
        live = live[np.argsort(self.last[live], kind="stable")]
        need = len(live) + extra
        if need > len(self.qi):
            size = need * 2
            self.qi = np.empty(size, dtype=np.int32)
            self.qs = np.empty(size, dtype=np.int64)
        self.qi[: len(live)] = live
        self.qs[: len(live)] = self.last[live]
        self.head, self.tail = 0, len(live)

    def evict(self, count: int, stamp: int) -> np.ndarray:
        """Pop up to `count` least-recently-used residents last touched before `stamp`."""
        out: list[np.ndarray] = []
        need = count
        while need > 0 and self.head < self.tail:
            end = min(self.tail, self.head + max(64, 4 * need))
            qi, qs = self.qi[self.head : end], self.qs[self.head : end]
            ok = self.resident[qi] & (self.last[qi] == qs) & (qs < stamp)
            idx = np.flatnonzero(ok)
            if len(idx) >= need:
                take = idx[:need]
                out.append(qi[take])
                self.head += int(take[-1]) + 1
                need = 0
            else:
                out.append(qi[idx])
                need -= len(idx)
                if end == self.tail or int(qs[-1]) >= stamp:
                    self.head += int(np.searchsorted(qs, stamp, side="left"))
                    break
                self.head = end
        victims = np.concatenate(out).astype(np.int64) if out else np.empty(0, np.int64)
        self.resident[victims] = False
        self.count -= len(victims)
        return victims


def run_lru(comp: CompiledTrace, pool: PoolData, layout: SlotLayout, split: int) -> Tally:
    tally = Tally(len(comp.client_names))
    rec = _Recorder(comp, pool, split, tally)
    ent = pool.entries
    off = ent.offsets.tolist()
    items_all = ent.items
    n = layout.n_items
    quota = layout.quota
    grouped = layout.n_groups > 1
    group_of = layout.group_of_item
    glru = None if grouped else _GlobalLru(n, int(quota[0]))
    resident = np.zeros(n, dtype=bool) if grouped else glru.resident
    last = np.full(n, -1, dtype=np.int64) if grouped else glru.last
    group_count = np.zeros(layout.n_groups, dtype=np.int64)
    big = np.iinfo(np.int64).max
    width = layout.width

    for r in np.flatnonzero(ent.lengths() > 0).tolist():
        a, b = off[r], off[r + 1]
        items = items_all[a:b]
        hit = resident[items]
        n_hit = int(np.count_nonzero(hit))
        needed = b - a
        if rec.kinds[r] == KIND_PREFILL:
            rec.prefill(r, n_hit, needed)
            continue
        rec.decode(r, resident, n_hit, needed)
        miss = items[~hit].astype(np.int64) if n_hit < needed else np.empty(0, np.int64)
        if not grouped:
            glru.touch(items, r)
            if len(miss):
                room = int(quota[0]) - glru.count
                if len(miss) > room:
                    glru.evict(len(miss) - room, r)
                admit = miss[: max(0, int(quota[0]) - glru.count)]
                resident[admit] = True
                glru.count += len(admit)
                if r >= split:
                    tally.swaps += len(admit)
            continue
        last[items] = r
        if not len(miss):
            continue
        miss_group = group_of[miss]
        want = np.bincount(miss_group, minlength=layout.n_groups)
        evict_n = np.maximum(want - (quota - group_count), 0)
        rows = np.flatnonzero(evict_n)
        if len(rows):
            key = np.where(resident & (last < r), last, big).reshape(layout.n_groups, width)[rows]
            order = np.argsort(key, axis=1, kind="stable")
            take = np.arange(width)[None, :] < evict_n[rows][:, None]
            victims = (rows[:, None] * width + order)[take]
            real = np.take_along_axis(key, order, axis=1)[take] < big
            victims = victims[real]
            resident[victims] = False
            group_count -= np.bincount(group_of[victims], minlength=layout.n_groups)
        room = quota - group_count
        first = np.searchsorted(miss_group, miss_group)
        admit = miss[(np.arange(len(miss)) - first) < room[miss_group]]
        resident[admit] = True
        group_count += np.bincount(group_of[admit], minlength=layout.n_groups)
        if r >= split:
            tally.swaps += len(admit)
    return tally


# ----------------------------------------------------------------------------------- adaptive


@dataclass
class AdaptiveParams:
    decay: float = 0.7
    interval: int = 4
    max_swaps: int = 96
    min_count: float = 2.0
    min_gain: float = 1.5
    latency: int = 0  # extra rounds before a promoted expert is usable
    prefill_promote: int = 256
    lock_fraction: float = 0.3
    client_window: int = 16


def run_adaptive(
    comp: CompiledTrace,
    pool: PoolData,
    layout: SlotLayout,
    split: int,
    params: AdaptiveParams,
    seed: np.ndarray | None,
    *,
    variant: str = "adaptive",
    client_freq: np.ndarray | None = None,
    round_log: list | None = None,
) -> Tally:
    """Decayed-usage promotion; `variant` adds prefill-informed or client-locked promotion."""
    if variant not in ("adaptive", "prefill-informed", "client-profile"):
        raise ValueError(variant)
    tally = Tally(len(comp.client_names))
    rec = _Recorder(comp, pool, split, tally)
    ent = pool.entries
    off = ent.offsets.tolist()
    items_all, counts_all = ent.items, ent.counts
    n = layout.n_items
    quota = layout.quota
    resident = np.zeros(n, dtype=bool) if seed is None else seed.copy()
    pending_mask = np.zeros(n, dtype=bool)
    pending: deque[tuple[int, np.ndarray]] = deque()
    locked = np.zeros(n, dtype=bool)
    counts = np.zeros(n, dtype=np.float64)
    nc = len(comp.client_names)
    lock_quota = np.floor(params.lock_fraction * quota).astype(np.int64)
    use_locks = variant == "client-profile" and client_freq is not None and nc > 0 and lock_quota.sum() > 0
    last_seen = np.full(max(nc, 1), -(10**9), dtype=np.int64)
    active_clients: frozenset[int] = frozenset()
    client_of = comp.rec_client.tolist()
    round_no = 0

    def free_slots() -> np.ndarray:
        used = _group_counts(resident | pending_mask, layout)
        return np.maximum(quota - used, 0)

    def schedule(cands: np.ndarray, victims: np.ndarray, ready: int, r: int) -> None:
        real = victims[victims >= 0]
        resident[real] = False
        if len(cands):
            pending.append((ready, cands))
            pending_mask[cands] = True
            if r >= split:
                tally.swaps += len(cands)

    def activate(now: int) -> None:
        while pending and pending[0][0] <= now:
            batch = pending.popleft()[1]
            resident[batch] = True
            pending_mask[batch] = False

    def apply_locks(active: frozenset[int], r: int) -> None:
        locked[:] = False
        if not active:
            return
        score = client_freq[sorted(active)].sum(axis=0)
        rows = score.reshape(layout.n_groups, layout.width)
        order = np.argsort(-rows, axis=1, kind="stable")
        keep = (np.arange(layout.width)[None, :] < lock_quota[:, None]) & (
            np.take_along_axis(rows, order, axis=1) > 0
        )
        chosen = (np.arange(layout.n_groups)[:, None] * layout.width + order)[keep]
        locked[chosen] = True
        need = chosen[~resident[chosen] & ~pending_mask[chosen]]
        if not len(need):
            return
        cand_ok = np.zeros(n, dtype=bool)
        cand_ok[need] = True
        cands, victims = plan_swaps(
            score, cand_ok, counts, resident & ~locked, free_slots(), layout,
            max_swaps=len(need), min_cand=NEG_INF, min_gain=NEG_INF, free_score=0.0,
        )
        schedule(cands, victims, round_no + 1 + params.latency, r)

    for r in np.flatnonzero(ent.lengths() > 0).tolist():
        a, b = off[r], off[r + 1]
        items = items_all[a:b]
        needed = b - a
        if rec.kinds[r] == KIND_PREFILL:
            n_hit = int(np.count_nonzero(resident[items]))
            rec.prefill(r, n_hit, needed)
            continue

        activate(round_no)
        if variant == "prefill-informed" and r in comp.prefill_lists:
            lists = comp.prefill_lists[r]
            promote = params.prefill_promote
            cand_score = np.zeros(n, dtype=np.float64)
            cand_ok = np.zeros(n, dtype=bool)
            for top, top_counts in lists:
                top, top_counts = top[:promote], top_counts[:promote]
                fresh = ~resident[top] & ~pending_mask[top]
                top, top_counts = top[fresh], top_counts[fresh]
                cand_score[top] = np.maximum(cand_score[top], top_counts)
                cand_ok[top] = True
            protect = np.zeros(n, dtype=bool)
            protect[items] = True
            cands, victims = plan_swaps(
                cand_score, cand_ok, counts, resident & ~locked & ~protect, free_slots(), layout,
                max_swaps=promote * len(lists), min_cand=NEG_INF, min_gain=NEG_INF, free_score=0.0,
            )
            schedule(cands, victims, round_no + params.latency, r)
            activate(round_no)

        n_hit = int(np.count_nonzero(resident[items]))
        rec.decode(r, resident, n_hit, needed)
        counts *= params.decay
        counts[items] += counts_all[a:b]

        if use_locks:
            c = client_of[r]
            if c >= 0:
                last_seen[c] = round_no
            elif c == MIXED:
                for cc, *_ in pool.mixed.get(r, ()):
                    last_seen[cc] = round_no

        decided = 0
        if (round_no + 1) % params.interval == 0:
            if use_locks:
                now_active = frozenset(np.flatnonzero(round_no - last_seen < params.client_window).tolist())
                if now_active != active_clients:
                    active_clients = now_active
                    apply_locks(active_clients, r)
            cands, victims = plan_swaps(
                counts, ~resident & ~pending_mask, counts, resident & ~locked, free_slots(), layout,
                max_swaps=params.max_swaps, min_cand=params.min_count, min_gain=params.min_gain,
                free_score=0.0,
            )
            schedule(cands, victims, round_no + 1 + params.latency, r)
            decided = len(cands)
        if round_log is not None:
            occupied = int(np.count_nonzero(resident | pending_mask))
            round_log.append((round_no, r, n_hit, needed, decided, occupied))
        round_no += 1
    return tally


# ----------------------------------------------------------------------------------- Belady


def run_belady(comp: CompiledTrace, pool: PoolData, layout: SlotLayout, split: int) -> Tally:
    """Farthest-next-use replacement with bypass: after every record keep, among the resident and
    just-missed items, the `quota` that are needed soonest (an offline lower bound on misses)."""
    tally = Tally(len(comp.client_names))
    rec = _Recorder(comp, pool, split, tally)
    ent = pool.entries
    off = ent.offsets.tolist()
    items_all = ent.items
    n = layout.n_items
    never = np.int64(2**62)
    nxt = np.empty(len(items_all), dtype=np.int64)
    upcoming = np.full(n, never, dtype=np.int64)
    dec = decode_records(comp, pool).tolist()
    for r in reversed(dec):
        a, b = off[r], off[r + 1]
        it = items_all[a:b]
        nxt[a:b] = upcoming[it]
        upcoming[it] = r
    next_use = upcoming.copy()  # first use of every item (after the loop `upcoming` holds it)
    resident = np.zeros(n, dtype=bool)
    quota = layout.quota

    for r in np.flatnonzero(ent.lengths() > 0).tolist():
        a, b = off[r], off[r + 1]
        items = items_all[a:b]
        hit = resident[items]
        n_hit = int(np.count_nonzero(hit))
        needed = b - a
        if rec.kinds[r] == KIND_PREFILL:
            rec.prefill(r, n_hit, needed)
            continue
        rec.decode(r, resident, n_hit, needed)
        use = nxt[a:b]
        next_use[items] = use
        if n_hit == needed:
            continue
        missing = ~hit & (use < never)
        if not missing.any():
            continue
        cand_ok = np.zeros(n, dtype=bool)
        cand_ok[items[missing]] = True
        cands, victims = plan_swaps(
            -next_use.astype(np.float64), cand_ok, -next_use.astype(np.float64), resident,
            np.maximum(quota - _group_counts(resident, layout), 0), layout,
            max_swaps=int(missing.sum()), min_cand=NEG_INF, min_gain=0.5, free_score=NEG_INF,
        )
        resident[victims[victims >= 0]] = False
        resident[cands] = True
        if r >= split:
            tally.swaps += len(cands)
    return tally


# ----------------------------------------------------------------------------------- profiles


def client_profiles(comp: CompiledTrace, pool: PoolData, lo: int, hi: int) -> np.ndarray:
    """Per-client record frequency (fraction of the client's decode records that need each item)
    over records [lo, hi). Shape [clients, items]; clients without records stay all-zero."""
    nc = len(comp.client_names)
    prof = np.zeros((nc, pool.n_items), dtype=np.float32)
    if nc == 0 or hi <= lo:
        return prof
    dec = decode_records(comp, pool, lo, hi)
    rc = comp.rec_client[dec].astype(np.int64)
    for c in range(nc):
        recs = dec[rc == c]
        total = len(recs)
        freq = record_frequency(pool, recs) if total else np.zeros(pool.n_items)
        for r in dec[rc == MIXED].tolist():
            for cc, items, _counts, _cols in pool.mixed.get(r, ()):
                if cc == c:
                    freq[items] += 1
                    total += 1
        if total:
            prof[c] = (freq / total).astype(np.float32)
    return prof
