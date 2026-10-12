"""Compile a trace stream into flat per-record arrays for the cache simulators.

For every record and every slot pool the compiled form keeps the *union* of the experts the
record's columns routed to (`items`) with the number of (column, layer-expert) uses (`counts`).
Hit accounting works on the union (a needed expert is one lookup per record and layer, however many
columns use it); the decayed-usage policies add the multiplicities.

Two slot pools exist: the text layers (items `layer * experts + expert`) and, if the trace has
one, the MTP layer (items `expert`). Prefill records are kept in the same arrays so that streaming
cost can be measured against whatever is resident when the chunk runs.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

from .requestlog import UNJOINED, UNTAGGED
from .trace import KIND_DECODE, KIND_MTP_VERIFY, KIND_PREFILL, TraceFormatError, TraceMetadata, TraceRecord

MAX_COUNT = 65535
MIXED = -2  # `rec_client` value for records whose columns belong to several clients


@dataclass
class EntrySet:
    """Per-record unique expert items; `offsets` has one entry per record plus one."""

    offsets: np.ndarray  # int64 [R + 1]
    items: np.ndarray  # uint16/uint32 [E], pool-local item ids
    counts: np.ndarray  # uint16 [E], (column, expert) uses, clipped

    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)


@dataclass
class PoolData:
    name: str  # "text" or "mtp"
    layer_lo: int
    n_layers: int
    experts: int
    entries: EntrySet
    # records with several clients: record -> [(client, items int32, counts int32, columns)]
    mixed: dict[int, list[tuple[int, np.ndarray, np.ndarray, int]]] = field(default_factory=dict)

    @property
    def n_items(self) -> int:
        return self.n_layers * self.experts


@dataclass
class CompiledTrace:
    meta: TraceMetadata
    kinds: np.ndarray  # uint8 [R]
    first_layer: np.ndarray
    layer_count: np.ndarray
    ts_us: np.ndarray  # uint64 [R]
    ncols: np.ndarray  # int32 [R]
    committed: np.ndarray  # int32 [R], estimated committed tokens of text decode/verify records
    rec_client: np.ndarray  # int16 [R]: client index, -1 no client table, MIXED for several
    client_names: list[str]
    text: PoolData
    mtp: PoolData | None
    req_first_rec: np.ndarray  # int64, record index of each request's first appearance, in order
    req_ids: np.ndarray  # uint64, the request ids in the same order
    prefill_lists: dict[int, list[tuple[np.ndarray, np.ndarray]]]
    status: str = "complete"
    status_detail: str = ""

    @property
    def n_records(self) -> int:
        return int(self.kinds.shape[0])

    @property
    def n_requests(self) -> int:
        return int(self.req_first_rec.shape[0])

    def split_index(self, train_fraction: float) -> int:
        """First record of the evaluation window: the first appearance of the first test request."""
        if train_fraction <= 0 or self.n_requests == 0:
            return 0
        n_train = math.ceil(train_fraction * self.n_requests)
        if n_train >= self.n_requests:
            return self.n_records
        return int(self.req_first_rec[n_train])

    def pools(self) -> list[PoolData]:
        return [self.text] + ([self.mtp] if self.mtp is not None else [])


class ClientLookup:
    """Vectorised request id -> client index.

    Request id 0 (a column the server did not tag) maps to the `(untagged)` client and ids missing
    from the mapping to the `(unjoined)` client.
    """

    def __init__(self, mapping: Mapping[int, str], unknown_label: str = UNJOINED, untagged_label: str = UNTAGGED):
        names = sorted(set(mapping.values()) - {unknown_label, untagged_label})
        self.names = names + [unknown_label, untagged_label]
        self.unknown = len(names)
        self.untagged = len(names) + 1
        index = {name: i for i, name in enumerate(names)}
        items = [(int(k), index[v]) for k, v in mapping.items() if int(k) != 0 and v in index]
        keys = np.fromiter((k for k, _ in items), dtype=np.uint64, count=len(items))
        values = np.fromiter((v for _, v in items), dtype=np.int16, count=len(items))
        order = np.argsort(keys, kind="stable")
        self._keys = keys[order]
        self._values = values[order]

    def indices(self, request_ids: np.ndarray) -> np.ndarray:
        if self._keys.size == 0:
            out = np.full(request_ids.shape, self.unknown, dtype=np.int16)
        else:
            pos = np.minimum(np.searchsorted(self._keys, request_ids), self._keys.size - 1)
            hit = self._keys[pos] == request_ids
            out = np.where(hit, self._values[pos], self.unknown).astype(np.int16)
        out[request_ids == 0] = self.untagged
        return out


class _EntryBuilder:
    """Collects raw per-record item arrays and dedupes them in large vectorised batches."""

    FLUSH_ITEMS = 1 << 22
    DIRECT_ITEMS = 1 << 17

    def __init__(self, n_items: int):
        self.n_items = n_items
        self.dtype = np.uint16 if n_items <= 0xFFFF else np.uint32
        self._raw: list[np.ndarray] = []
        self._recs: list[int] = []
        self._pending = 0
        self._items: list[np.ndarray] = []
        self._counts: list[np.ndarray] = []
        self._lens: list[tuple[np.ndarray, np.ndarray]] = []

    def add(self, rec: int, raw: np.ndarray) -> None:
        if raw.size > self.DIRECT_ITEMS:
            # prefill-sized records: a histogram is far cheaper than sorting millions of keys
            self._flush()
            hist = np.bincount(raw, minlength=self.n_items)
            nz = np.flatnonzero(hist)
            self._items.append(nz.astype(self.dtype))
            self._counts.append(np.minimum(hist[nz], MAX_COUNT).astype(np.uint16))
            self._lens.append((np.asarray([rec], dtype=np.int64), np.asarray([nz.size], dtype=np.int64)))
            return
        self._raw.append(raw)
        self._recs.append(rec)
        self._pending += raw.size
        if self._pending >= self.FLUSH_ITEMS:
            self._flush()

    def _flush(self) -> None:
        if not self._recs:
            return
        recs = np.asarray(self._recs, dtype=np.int64)
        lens = np.fromiter((x.size for x in self._raw), dtype=np.int64, count=len(self._raw))
        keys = np.repeat(recs, lens) * self.n_items + np.concatenate(self._raw).astype(np.int64)
        uniq, cnt = np.unique(keys, return_counts=True)
        rec_of = uniq // self.n_items
        self._items.append((uniq - rec_of * self.n_items).astype(self.dtype))
        self._counts.append(np.minimum(cnt, MAX_COUNT).astype(np.uint16))
        per_rec = np.bincount(np.searchsorted(recs, rec_of), minlength=recs.size)
        self._lens.append((recs, per_rec))
        self._raw, self._recs, self._pending = [], [], 0

    def finish(self, n_records: int) -> EntrySet:
        self._flush()
        lens = np.zeros(n_records, dtype=np.int64)
        for recs, per_rec in self._lens:
            lens[recs] = per_rec
        offsets = np.zeros(n_records + 1, dtype=np.int64)
        np.cumsum(lens, out=offsets[1:])
        items = np.concatenate(self._items) if self._items else np.empty(0, dtype=self.dtype)
        counts = np.concatenate(self._counts) if self._counts else np.empty(0, dtype=np.uint16)
        return EntrySet(offsets, items, counts)


def _unique_counts(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uniq, cnt = np.unique(raw, return_counts=True)
    return uniq.astype(np.int32), cnt.astype(np.int32)


def compile_trace(
    records: Iterable[TraceRecord],
    meta: TraceMetadata,
    *,
    client_lookup: ClientLookup | None = None,
    prefill_topk: int = 0,
    max_records: int | None = None,
    status: str = "complete",
    status_detail: str = "",
) -> CompiledTrace:
    experts, text_layers, mtp_layer = meta.experts, meta.text_layers, meta.mtp_layer
    text_items = text_layers * experts
    text_builder = _EntryBuilder(text_items)
    mtp_builder = _EntryBuilder(experts) if mtp_layer is not None else None

    kinds: list[int] = []
    first_layers: list[int] = []
    layer_counts: list[int] = []
    stamps: list[int] = []
    ncols_l: list[int] = []
    committed: list[int] = []
    rec_client: list[int] = []
    mixed: dict[int, list] = {}
    req_first: dict[int, int] = {}
    last_unit: dict[int, tuple[int, int, int]] = {}
    prefill_acc: dict[int, np.ndarray] = {}
    prefill_lists: dict[int, list] = {}

    for record in records:
        r = len(kinds)
        if max_records is not None and r >= max_records:
            break
        cols = record.columns
        lo = record.first_layer
        hi = lo + record.layer_count
        ids = record.ids
        if ids.size and int(ids.max()) >= experts:
            raise TraceFormatError(f"record {r}: expert id {int(ids.max())} >= experts {experts}")
        text_hi = min(hi, text_layers)
        has_text = lo < text_hi
        has_mtp = mtp_layer is not None and lo <= mtp_layer < hi
        covered = (text_hi - lo if has_text else 0) + (1 if has_mtp else 0)
        if covered != record.layer_count:
            raise TraceFormatError(
                f"record {r}: layers {lo}..{hi - 1} are neither text layers (<{text_layers}) nor the MTP layer ({mtp_layer})"
            )

        kinds.append(record.kind)
        first_layers.append(lo)
        layer_counts.append(record.layer_count)
        stamps.append(record.timestamp_us)
        ncols_l.append(cols)
        committed.append(0)

        req = record.request_ids
        uniq_req = np.unique(req) if cols > 1 else req
        for u in uniq_req.tolist():
            if u not in req_first:
                req_first[u] = r

        raw_text2d = None
        if has_text:
            layer_ids = np.arange(lo, text_hi, dtype=np.int32) * experts
            raw_text2d = (ids[:, : text_hi - lo, :].astype(np.int32) + layer_ids[None, :, None]).reshape(cols, -1)
            text_builder.add(r, raw_text2d.reshape(-1))
        if has_mtp:
            mtp_builder.add(r, ids[:, mtp_layer - lo, :].astype(np.int32).reshape(-1))

        # clients
        if client_lookup is None:
            rec_client.append(-1)
        else:
            ci = client_lookup.indices(req) if cols else np.empty(0, dtype=np.int16)
            cu = np.unique(ci) if cols > 1 else ci
            if cu.size <= 1:
                rec_client.append(int(cu[0]) if cu.size else client_lookup.unknown)
            else:
                rec_client.append(MIXED)
                if has_text:
                    segments = []
                    for c in cu.tolist():
                        m = ci == c
                        items_c, counts_c = _unique_counts(raw_text2d[m].reshape(-1))
                        segments.append((c, items_c, counts_c, int(m.sum())))
                    mixed[r] = segments

        if not has_text:
            continue
        positions = record.positions
        if record.kind == KIND_PREFILL and prefill_topk > 0:
            for u in uniq_req.tolist():
                rows = raw_text2d if cols == 1 or uniq_req.size == 1 else raw_text2d[req == u]
                acc = prefill_acc.get(u)
                if acc is None:
                    acc = prefill_acc[u] = np.zeros(text_items, dtype=np.uint32)
                    if len(prefill_acc) > 64:
                        prefill_acc.pop(next(iter(prefill_acc)))
                acc += np.bincount(rows.reshape(-1), minlength=text_items).astype(np.uint32)
        elif record.kind in (KIND_DECODE, KIND_MTP_VERIFY):
            if cols == 1 or uniq_req.size == 1:
                groups = [(int(uniq_req[0]), int(positions.min()), cols)]
            else:
                groups = []
                for u in uniq_req.tolist():
                    m = req == u
                    groups.append((u, int(positions[m].min()), int(m.sum())))
            for u, p0, n in groups:
                prev = last_unit.get(u)
                if prev is None:
                    acc = prefill_acc.pop(u, None) if prefill_topk > 0 else None
                    if acc is not None:
                        nz = np.flatnonzero(acc)
                        if nz.size:
                            order = np.argsort(-acc[nz].astype(np.int64), kind="stable")[:prefill_topk]
                            top = nz[order]
                            prefill_lists.setdefault(r, []).append((top.astype(np.int32), acc[top].astype(np.int64)))
                else:
                    prev_rec, prev_p0, prev_n = prev
                    committed[prev_rec] += min(max(p0 - prev_p0, 1), prev_n)
                last_unit[u] = (r, p0, n)

    for prev_rec, _p0, _n in last_unit.values():
        committed[prev_rec] += 1  # the final unit of a request: at least one token

    n_records = len(kinds)
    text = PoolData("text", 0, text_layers, experts, text_builder.finish(n_records), mixed)
    mtp = None
    if mtp_builder is not None:
        mtp = PoolData("mtp", mtp_layer, 1, experts, mtp_builder.finish(n_records))
    return CompiledTrace(
        meta=meta,
        kinds=np.asarray(kinds, dtype=np.uint8),
        first_layer=np.asarray(first_layers, dtype=np.uint16),
        layer_count=np.asarray(layer_counts, dtype=np.uint16),
        ts_us=np.asarray(stamps, dtype=np.uint64),
        ncols=np.asarray(ncols_l, dtype=np.int32),
        committed=np.asarray(committed, dtype=np.int32),
        rec_client=np.asarray(rec_client, dtype=np.int16),
        client_names=list(client_lookup.names) if client_lookup is not None else [],
        text=text,
        mtp=mtp,
        req_first_rec=np.fromiter(req_first.values(), dtype=np.int64, count=len(req_first)),
        req_ids=np.fromiter(req_first.keys(), dtype=np.uint64, count=len(req_first)),
        prefill_lists=prefill_lists,
        status=status,
        status_detail=status_detail,
    )
