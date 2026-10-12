"""Expert-trace tooling: format round trip, hit accounting, cache policies, request-log join."""

from __future__ import annotations

import itertools
import json
import random
from collections import Counter, OrderedDict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tools.bench.expert_trace import analysis, policies as pol, simulate as sim
from tools.bench.expert_trace.compiled import ClientLookup, compile_trace
from tools.bench.expert_trace.requestlog import NO_USER_AGENT, load_request_log
from tools.bench.expert_trace.synthetic import SyntheticConfig, build_synthetic
from tools.bench.expert_trace.trace import (
    KIND_DECODE,
    KIND_MTP_PROPOSAL,
    KIND_MTP_VERIFY,
    KIND_PREFILL,
    KIND_PREFILL_HISTOGRAM,
    RECORD_HEADER,
    HistogramRecord,
    TraceFormatError,
    TraceReader,
    TraceWriter,
    ids_to_histogram,
    make_metadata,
    read_trace,
)

SERVER = "serve-1-100"


# ------------------------------------------------------------------------------------ helpers


def write_trace(path: Path, records, *, text_layers=1, experts=4, top_k=1, mtp_layer=None, **extra) -> Path:
    meta = make_metadata(
        text_layers=text_layers, experts=experts, top_k=top_k, mtp_layer=mtp_layer,
        server_instance_id=SERVER, created_unix_ms=1_000_000, **extra,
    )
    with TraceWriter(path, meta) as writer:
        for rec in records:
            if rec["kind"] == KIND_PREFILL_HISTOGRAM:
                writer.write_histogram(**{k: v for k, v in rec.items() if k != "kind"})
            else:
                writer.write_record(**rec)
    return path


def prefill_pair(request, first_position, ids, experts):
    """The same prefill chunk as per-token rows (kind 1) and as a histogram (kind 5)."""
    ids = np.asarray(ids)
    tokens = ids.shape[0]
    rows = dict(kind=KIND_PREFILL, request_ids=[request] * tokens,
                positions=range(first_position, first_position + tokens), ids=ids)
    summary = dict(kind=KIND_PREFILL_HISTOGRAM, request_id=request, first_position=first_position, tokens=tokens,
                   counts=ids_to_histogram(ids, experts))
    return rows, summary


def decode(request, position, *experts, kind=KIND_DECODE, first_layer=0, ts=0):
    """One column, one layer, top_k = len(experts)."""
    return dict(kind=kind, request_ids=[request], positions=[position],
                ids=np.array([[list(experts)]]), first_layer=first_layer, timestamp_us=ts)


def compile_path(path: Path, mapping=None, prefill_topk=0, **kw):
    reader = TraceReader(path)
    try:
        lookup = ClientLookup(mapping) if mapping else None
        return compile_trace(reader.records(), reader.metadata, client_lookup=lookup,
                             prefill_topk=prefill_topk, **kw)
    finally:
        reader.close()


def run(comp, *, budgets, policies, train_fraction=0.0, slots=("global",), mtp_budget="proportional",
        quota_from="profile", **params):
    cfg = sim.SimConfig(budgets=list(budgets), policies=list(policies), slots=list(slots),
                        train_fraction=train_fraction, mtp_budget=mtp_budget, quota_from=quota_from,
                        params=pol.AdaptiveParams(**params))
    return sim.run_simulation(comp, cfg)


def policy_result(result, name, *, budget_index=0, pool="text"):
    return result["budgets"][budget_index]["pools"][pool]["policies"][name]


def pool_sets(pool):
    off = pool.entries.offsets
    return [pool.entries.items[off[r]:off[r + 1]].astype(int).tolist() for r in range(len(off) - 1)]


# -------------------------------------------------------------------------------- reader/writer


def test_round_trip_all_kinds(tmp_path):
    rng = np.random.default_rng(1)
    path = tmp_path / "t.nxtr"
    meta = make_metadata(text_layers=3, experts=16, top_k=4, mtp_layer=3, server_instance_id=SERVER,
                         artifact_id="a" * 8, created_unix_ms=5)
    expected = []
    with TraceWriter(path, meta) as writer:
        for kind, first_layer, layers, cols in [(KIND_PREFILL, 0, 3, 5), (KIND_DECODE, 0, 3, 2),
                                                (KIND_MTP_PROPOSAL, 3, 1, 2), (KIND_MTP_VERIFY, 0, 3, 8)]:
            ids = rng.integers(0, 16, size=(cols, layers, 4)).astype(np.uint16)
            req = rng.integers(1, 1 << 40, size=cols).astype(np.uint64)
            pos = rng.integers(0, 1 << 20, size=cols).astype(np.uint32)
            writer.write_record(kind, req, pos, ids, first_layer=first_layer, timestamp_us=1000 + kind,
                                flags=np.arange(cols))
            expected.append((kind, first_layer, req, pos, ids))
    metadata, records, reader = read_trace(path)
    assert reader.status == "complete" and reader.records_read == 4
    assert (metadata.text_layers, metadata.mtp_layer, metadata.experts, metadata.top_k) == (3, 3, 16, 4)
    assert metadata.server_instance_id == SERVER and metadata.created_unix_ms == 5
    for rec, (kind, first_layer, req, pos, ids) in zip(records, expected):
        assert (rec.kind, rec.first_layer, rec.timestamp_us) == (kind, first_layer, 1000 + kind)
        assert rec.layer_count == ids.shape[1] and rec.columns == ids.shape[0]
        np.testing.assert_array_equal(rec.request_ids, req)
        np.testing.assert_array_equal(rec.positions, pos)
        np.testing.assert_array_equal(rec.flags, np.arange(ids.shape[0]))
        np.testing.assert_array_equal(rec.ids, ids)
    assert RECORD_HEADER.size == 24


def test_null_mtp_layer_and_extra_metadata(tmp_path):
    path = write_trace(tmp_path / "t.nxtr", [decode(1, 0, 2)], mtp_layer=None, custom={"x": 1})
    metadata, _records, _reader = read_trace(path)
    assert metadata.mtp_layer is None and metadata.total_layers == 1
    assert metadata.raw["custom"] == {"x": 1}


def test_truncated_tail_stops_at_last_complete_record(tmp_path):
    path = write_trace(tmp_path / "t.nxtr", [decode(1, i, i % 4) for i in range(3)])
    data = path.read_bytes()
    record_size = 24 + 16 + 2
    for cut in (1, 10, 24, 30, record_size - 1):
        short = tmp_path / f"cut{cut}.nxtr"
        short.write_bytes(data[: len(data) - record_size + cut])
        _meta, records, reader = read_trace(short)
        assert len(records) == 2, cut
        assert reader.status == "truncated" and "2 complete records" in reader.status_detail
        assert reader.valid_bytes == len(data) - record_size
    exact = tmp_path / "exact.nxtr"
    exact.write_bytes(data[: len(data) - record_size])
    assert read_trace(exact)[2].status == "complete"


def test_corrupt_tail_and_bad_preamble(tmp_path):
    path = write_trace(tmp_path / "t.nxtr", [decode(1, 0, 1), decode(1, 1, 2)])
    data = bytearray(path.read_bytes())
    data[-(24 + 16 + 2)] ^= 0xFF  # break the second record's sentinel
    path.write_bytes(bytes(data))
    _meta, records, reader = read_trace(path)
    assert len(records) == 1 and reader.status == "corrupt"
    bad = tmp_path / "bad.nxtr"
    bad.write_bytes(b"NOTATRACE" + b"\0" * 32)
    with pytest.raises(TraceFormatError):
        TraceReader(bad)
    cut_meta = tmp_path / "cutmeta.nxtr"
    cut_meta.write_bytes(path.read_bytes()[:20])
    with pytest.raises(TraceFormatError):
        TraceReader(cut_meta)


def test_writer_rejects_out_of_range_expert(tmp_path):
    with TraceWriter(tmp_path / "t.nxtr", make_metadata(text_layers=1, experts=4, top_k=1)) as writer:
        with pytest.raises(TraceFormatError):
            writer.write_record(KIND_DECODE, [1], [0], np.array([[[4]]]))


# --------------------------------------------------------------------------- compile / accounting


def test_compile_matches_naive_union_and_counts(tmp_path):
    rng = np.random.default_rng(3)
    records, expected = [], []
    for i in range(12):
        cols = int(rng.integers(1, 5))
        ids = rng.integers(0, 8, size=(cols, 2, 3))
        ids[:, :, 1] = (ids[:, :, 0] + 1) % 8  # keep top-k entries distinct per column and layer
        ids[:, :, 2] = (ids[:, :, 0] + 2) % 8
        records.append(dict(kind=KIND_DECODE, request_ids=np.full(cols, 5 + i % 3), positions=np.arange(cols), ids=ids))
        naive = Counter(layer * 8 + int(e) for c in range(cols) for layer in range(2) for e in ids[c, layer])
        expected.append(naive)
    comp = compile_path(write_trace(tmp_path / "t.nxtr", records, text_layers=2, experts=8, top_k=3))
    off = comp.text.entries.offsets
    for r, naive in enumerate(expected):
        items = comp.text.entries.items[off[r]:off[r + 1]].astype(int).tolist()
        counts = comp.text.entries.counts[off[r]:off[r + 1]].astype(int).tolist()
        assert items == sorted(naive) and counts == [naive[i] for i in sorted(naive)]


def test_hit_is_counted_once_per_record_and_layer_whatever_the_columns(tmp_path):
    # two columns share expert 2 in layer 0 and expert 5 in layer 1
    rec = dict(kind=KIND_DECODE, request_ids=[1, 2], positions=[0, 0],
               ids=np.array([[[1, 2], [5, 6]], [[2, 3], [5, 7]]]))
    comp = compile_path(write_trace(tmp_path / "t.nxtr", [rec], text_layers=2, experts=8, top_k=2),
                        mapping={1: "a", 2: "b"})
    resident = np.zeros(16, dtype=bool)
    resident[[1, 2, 8 + 5]] = True  # (l0,1) (l0,2) (l1,5)
    tally = pol.run_static(comp, comp.text, resident, split=0)
    assert (tally.needed, tally.hits, tally.misses) == (6, 3, 3)  # union {1,2,3} + {5,6,7}
    assert tally.decode_columns == 2
    # per-client: each client's own needs; the shared (l0,2) and (l1,5) count for both
    names = comp.client_names
    a, b = names.index("a"), names.index("b")
    assert (tally.client_needed[a], tally.client_hits[a]) == (4, 3)
    assert (tally.client_needed[b], tally.client_hits[b]) == (4, 2)
    assert comp.rec_client[0] == -2  # mixed record


def test_mtp_records_go_to_their_own_pool_and_prefill_is_separate(tmp_path):
    recs = [
        dict(kind=KIND_PREFILL, request_ids=[1, 1, 1], positions=[0, 1, 2],
             ids=np.array([[[0, 1]], [[2, 3]], [[0, 3]]])),
        decode(1, 3, 0, 1),
        decode(1, 4, 6, 7, kind=KIND_MTP_PROPOSAL, first_layer=1),
        decode(1, 4, 2, 3, kind=KIND_MTP_VERIFY),
    ]
    comp = compile_path(write_trace(tmp_path / "t.nxtr", recs, text_layers=1, experts=8, top_k=2, mtp_layer=1))
    assert pool_sets(comp.text) == [[0, 1, 2, 3], [0, 1], [], [2, 3]]
    assert pool_sets(comp.mtp) == [[], [], [6, 7], []]
    resident = np.zeros(8, dtype=bool)
    resident[[0, 1]] = True
    tally = pol.run_static(comp, comp.text, resident, split=0)
    assert (tally.prefill_records, tally.prefill_columns, tally.prefill_needed, tally.prefill_missing) == (1, 3, 4, 2)
    assert (tally.decode_records, tally.needed, tally.hits) == (2, 4, 2)


def test_committed_token_estimate_from_position_progress(tmp_path):
    def verify(request, first, width=3):
        return dict(kind=KIND_MTP_VERIFY, request_ids=[request] * width, positions=range(first, first + width),
                    ids=np.zeros((width, 1, 1), dtype=int))
    comp = compile_path(write_trace(tmp_path / "t.nxtr", [verify(7, 10), verify(7, 12), verify(7, 13), verify(7, 20)]))
    # 10->12: 2 committed, 12->13: 1, 13->20 clipped to the unit width 3, last unit counts 1
    assert comp.committed.tolist() == [2, 1, 3, 1]


# ---------------------------------------------------------------------- known-answer policies

SEQUENCE = [0, 0, 1, 0, 2, 0, 1, 3, 0, 1]


def sequence_trace(tmp_path, sequence=SEQUENCE, request=1):
    return compile_path(write_trace(tmp_path / "seq.nxtr", [decode(request, i, e) for i, e in enumerate(sequence)]))


def test_static_lru_belady_known_answers(tmp_path):
    comp = sequence_trace(tmp_path)
    result = run(comp, budgets=[2], policies=["static-oracle", "lru", "belady"])
    oracle, lru, belady = (policy_result(result, p) for p in ("static-oracle", "lru", "belady"))
    assert (oracle["needed"], oracle["hits"], oracle["swaps"]) == (10, 8, 0)  # keeps experts 0 and 1
    assert (lru["hits"], lru["swaps"]) == (3, 7)  # hand-simulated LRU(2)
    assert (belady["hits"], belady["swaps"]) == (6, 2)  # expert 2 and 3 are bypassed, never reused
    assert lru["misses"] == 7 and belady["misses"] == 4
    assert lru["miss_mib_per_token"] == pytest.approx(7 * sim.DEFAULT_EXPERT_BYTES / sim.MIB / 10)
    assert lru["swap_mib_per_token"] == pytest.approx(7 * sim.DEFAULT_EXPERT_BYTES / sim.MIB / 10)


def test_static_train_uses_only_the_training_split(tmp_path):
    seq = [(1, e) for e in (0, 0, 1)] + [(2, e) for e in (0, 1, 0)] + [(3, e) for e in (2, 2, 3)] + [(4, e) for e in (2, 3, 2)]
    comp = compile_path(write_trace(tmp_path / "t.nxtr", [decode(r, i, e) for i, (r, e) in enumerate(seq)]))
    assert comp.split_index(0.5) == 6 and comp.split_index(0.0) == 0
    result = run(comp, budgets=[2], policies=["static-oracle", "static-train"], train_fraction=0.5)
    assert result["window"]["split_record"] == 6 and result["window"]["eval_records"] == 6
    assert policy_result(result, "static-train")["hits"] == 0  # trained on {0,1}, evaluated on {2,3}
    assert policy_result(result, "static-oracle")["hits"] == 6  # hindsight on the evaluated window


def test_static_train_needs_a_training_split(tmp_path):
    result = run(sequence_trace(tmp_path), budgets=[2], policies=["static-train", "client-profile", "prefill-informed"])
    for name in ("static-train", "client-profile", "prefill-informed"):
        assert "skipped" in policy_result(result, name)


# ------------------------------------------------------------------------------------ adaptive


def three_hot_trace(tmp_path, rounds=12):
    """Three columns per round using experts 1, 2, 3 of an 8-expert layer."""
    rec = lambda i: dict(kind=KIND_DECODE, request_ids=[1, 1, 1], positions=[i] * 3, ids=np.array([[[1]], [[2]], [[3]]]))
    return compile_path(write_trace(tmp_path / "hot.nxtr", [rec(i) for i in range(rounds)], experts=8))


def adaptive_log(comp, budget, **kw):
    layout = pol.make_layout(comp.text, "global", budget)
    log = []
    tally = pol.run_adaptive(comp, comp.text, layout, 0, pol.AdaptiveParams(**kw), None, round_log=log)
    return tally, log


def test_adaptive_admits_only_after_the_interval_and_respects_max_swaps(tmp_path):
    comp = three_hot_trace(tmp_path)
    tally, log = adaptive_log(comp, 3, interval=4, max_swaps=2)
    decided = [entry[4] for entry in log]
    assert decided == [0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0]  # at most 2 per decision, then the third
    hits = [entry[2] for entry in log]
    # nothing is resident until the first decision lands; a promotion is usable from the next round
    assert hits == [0] * 4 + [2] * 4 + [3] * 4
    assert max(entry[5] for entry in log) <= 3  # slots never exceeded (promotions in flight count)
    assert tally.swaps == 3


def test_adaptive_copy_latency_delays_use_by_whole_rounds(tmp_path):
    comp = three_hot_trace(tmp_path)
    _tally, log = adaptive_log(comp, 3, interval=4, max_swaps=3, latency=2)
    assert [entry[2] for entry in log] == [0] * 6 + [3] * 6


def test_adaptive_waits_for_min_count_and_min_gain(tmp_path):
    comp = three_hot_trace(tmp_path)
    # a hot expert used once per round has decayed count 2.53 after 4 rounds and 3.14 after 8
    _t, log = adaptive_log(comp, 3, interval=4, max_swaps=3, min_count=3.0)
    assert [entry[4] for entry in log] == [0] * 7 + [3] + [0] * 4
    _t, log = adaptive_log(comp, 3, interval=4, max_swaps=3, min_gain=2.6)  # a free slot "gains" the count itself
    assert [entry[4] for entry in log] == [0] * 7 + [3] + [0] * 4


def test_adaptive_swaps_a_cold_resident_for_a_hot_candidate(tmp_path):
    # budget 1: expert 0 is seeded and cold; expert 1 is used every round
    comp = compile_path(write_trace(tmp_path / "t.nxtr", [decode(1, i, 1) for i in range(8)], experts=4))
    layout = pol.make_layout(comp.text, "global", 1)
    seed = np.zeros(4, dtype=bool)
    seed[0] = True
    log = []
    pol.run_adaptive(comp, comp.text, layout, 0, pol.AdaptiveParams(interval=4), seed, round_log=log)
    assert [entry[2] for entry in log] == [0, 0, 0, 0, 1, 1, 1, 1]


@pytest.mark.parametrize("histogram", [False, True])
def test_prefill_informed_promotes_what_the_prompt_used(tmp_path, histogram):
    rows, summary = prefill_pair(1, 0, np.array([[[10, 11]]] * 6), 16)
    prefill = summary if histogram else rows
    decodes = [decode(1, 6 + i, 10, 11) for i in range(5)]
    comp = compile_path(write_trace(tmp_path / "t.nxtr", [prefill, *decodes], experts=16, top_k=2), prefill_topk=2)
    assert list(comp.prefill_lists) == [1]
    result = run(comp, budgets=[2], policies=["adaptive", "prefill-informed"], interval=1000, prefill_promote=2)
    assert policy_result(result, "adaptive")["hits"] == 0
    informed = policy_result(result, "prefill-informed")
    assert (informed["hits"], informed["needed"], informed["swaps"]) == (10, 10, 2)
    # the streamed prefill itself is measured against the resident set at that time
    assert informed["prefill"]["needed"] == 2 and informed["prefill"]["missing"] == 2
    assert informed["prefill"]["columns"] == 6  # tokens, whichever form the chunk has


def test_client_profile_locks_the_active_clients_experts(tmp_path):
    def rec(request, position, experts):
        return dict(kind=KIND_DECODE, request_ids=[request] * 4, positions=[position] * 4,
                    ids=np.array([[[e]] for e in experts]))
    a, b = [0, 1, 2, 3], [8, 9, 10, 11]
    records = [rec(1, i, a) for i in range(10)] + [rec(2, i, b) for i in range(10)]
    records += [rec(3, i, b) for i in range(8)] + [rec(4, i, a) for i in range(8)] + [rec(5, i, b) for i in range(8)]
    mapping = {1: "app-a", 2: "app-b", 3: "app-b", 4: "app-a", 5: "app-b"}
    comp = compile_path(write_trace(tmp_path / "t.nxtr", records, experts=16), mapping)
    params = dict(interval=1, min_count=1000.0, lock_fraction=0.5, client_window=2)
    result = run(comp, budgets=[4], policies=["adaptive", "client-profile"], train_fraction=0.4, **params)
    adaptive, locked = policy_result(result, "adaptive"), policy_result(result, "client-profile")
    # adaptive never promotes (min-count) and stays on the training-profile seed; the locks follow the clients
    assert locked["hits"] > adaptive["hits"] and locked["swaps"] > 0 and adaptive["swaps"] == 0
    assert set(locked["per_client"]) == {"app-a", "app-b"}


# ---------------------------------------------------------- randomized checks against references


def random_compiled(tmp_path, seed, *, layers=1, experts=6, top_k=1, records=14, max_cols=2):
    rng = random.Random(seed)
    recs = []
    for i in range(records):
        cols = rng.randint(1, max_cols)
        ids = np.array([[[rng.randrange(experts)] * top_k for _ in range(layers)] for _ in range(cols)])
        recs.append(dict(kind=KIND_DECODE, request_ids=[1] * cols, positions=[i] * cols, ids=ids))
    return compile_path(write_trace(tmp_path / f"r{seed}.nxtr", recs, text_layers=layers, experts=experts, top_k=top_k))


def naive_lru(records, quota, width):
    caches = [OrderedDict() for _ in quota]
    hits = swaps = 0
    for items in records:
        touched = set(items)
        fresh = [[] for _ in quota]
        for i in items:
            g = i // width
            if i in caches[g]:
                caches[g].move_to_end(i)
                hits += 1
            else:
                caches[g][i] = None
                fresh[g].append(i)
        for g, cache in enumerate(caches):
            over = len(cache) - quota[g]
            while over > 0:
                victim = next((k for k in cache if k not in touched), None)
                if victim is None:
                    break
                del cache[victim]
                over -= 1
            for i in sorted(fresh[g], reverse=True)[: max(over, 0)]:
                del cache[i]
                fresh[g].remove(i)
            swaps += len(fresh[g])
    return hits, swaps


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("quota", [[3], [2, 1, 3]])
def test_lru_matches_a_naive_reference(tmp_path, seed, quota):
    layers = len(quota)
    comp = random_compiled(tmp_path, seed, layers=layers, experts=6, top_k=2, records=40, max_cols=3)
    layout = pol.SlotLayout("per-layer" if layers > 1 else "global", comp.text.n_items, layers,
                            comp.text.n_items // layers, np.asarray(quota))
    tally = pol.run_lru(comp, comp.text, layout, 0)
    hits, swaps = naive_lru(pool_sets(comp.text), quota, layout.width)
    assert (tally.hits, tally.swaps) == (hits, swaps)


def test_lru_with_a_record_larger_than_the_pool_keeps_the_pool_full(tmp_path):
    rec = dict(kind=KIND_DECODE, request_ids=[1] * 4, positions=[0] * 4, ids=np.array([[[0]], [[1]], [[2]], [[3]]]))
    comp = compile_path(write_trace(tmp_path / "t.nxtr", [rec, rec, decode(1, 1, 3)], experts=8))
    layout = pol.make_layout(comp.text, "global", 2)
    tally = pol.run_lru(comp, comp.text, layout, 0)
    # four items compete for two slots in one record: the lowest two are kept and the rest stay
    # host-served; the later single-expert record then evicts the older of the two
    assert tally.hits == 2 and tally.swaps == 3
    assert (tally.hits, tally.swaps) == naive_lru(pool_sets(comp.text), [2], 8)


def optimal_misses(records, capacity):
    states = {frozenset(): 0}
    for rec in records:
        rec = frozenset(rec)
        nxt: dict[frozenset, int] = {}
        for state, cost in states.items():
            cost += len(rec - state)
            pool = sorted(state | rec)
            for k in range(min(capacity, len(pool)) + 1):
                for kept in itertools.combinations(pool, k):
                    key = frozenset(kept)
                    if nxt.get(key, 1 << 30) > cost:
                        nxt[key] = cost
        states = nxt
    return min(states.values())


@pytest.mark.parametrize("seed", range(12))
def test_belady_reaches_the_offline_optimum_and_never_loses_to_lru(tmp_path, seed):
    comp = random_compiled(tmp_path, seed, experts=6, records=14, max_cols=2)
    for capacity in (1, 2, 3):
        layout = pol.make_layout(comp.text, "global", capacity)
        belady = pol.run_belady(comp, comp.text, layout, 0)
        lru = pol.run_lru(comp, comp.text, layout, 0)
        assert belady.misses == optimal_misses(pool_sets(comp.text), capacity)
        assert belady.misses <= lru.misses


def test_belady_per_layer_is_the_sum_of_independent_optima(tmp_path):
    comp = random_compiled(tmp_path, 5, layers=2, experts=4, top_k=1, records=14, max_cols=2)
    layout = pol.SlotLayout("per-layer", 8, 2, 4, np.asarray([2, 1]))
    belady = pol.run_belady(comp, comp.text, layout, 0)
    sets = pool_sets(comp.text)
    expected = sum(
        optimal_misses([[i for i in s if i // 4 == g] for s in sets], q) for g, q in enumerate((2, 1))
    )
    assert belady.misses == expected


def naive_plan(cand_score, cand_ok, vict_score, vict_ok, free, groups, width, *, max_swaps, min_cand, min_gain, free_score):
    pairs = []
    for g in range(groups):
        lo = g * width
        cands = sorted(((cand_score[i], i) for i in range(lo, lo + width) if cand_ok[i]), reverse=True)
        victims = [(free_score, -1)] * int(free[g]) + sorted((vict_score[i], i) for i in range(lo, lo + width) if vict_ok[i])
        for (cs, ci), (vs, vi) in zip(cands, victims):
            if cs >= min_cand and cs - vs >= min_gain:
                pairs.append((cs - vs, ci, vi))
    pairs.sort(reverse=True)
    return {(ci, vi) for _gain, ci, vi in pairs[:max_swaps]}


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("groups", [1, 3])
def test_plan_swaps_pairs_best_candidates_with_worst_victims_per_group(seed, groups):
    rng = np.random.default_rng(seed)
    width = 12
    n = groups * width
    resident = rng.random(n) < 0.4
    cand_score = rng.random(n) * 6
    vict_score = rng.random(n) * 3
    quota = resident.reshape(groups, width).sum(axis=1) + rng.integers(0, 3, size=groups)
    layout = pol.SlotLayout("per-layer" if groups > 1 else "global", n, groups, width, quota)
    free = quota - resident.reshape(groups, width).sum(axis=1)
    kw = dict(max_swaps=int(rng.integers(1, 7)), min_cand=1.0, min_gain=0.8, free_score=0.0)
    cands, victims = pol.plan_swaps(cand_score, ~resident, vict_score, resident, free, layout, **kw)
    expected = naive_plan(cand_score, ~resident, vict_score, resident, free, groups, width, **kw)
    assert set(zip(cands.tolist(), victims.tolist())) == expected
    assert len(cands) == len(set(cands.tolist())) <= kw["max_swaps"]
    for c, v in zip(cands.tolist(), victims.tolist()):
        assert v < 0 or v // width == c // width  # a swap never moves a slot between layers


def test_adaptive_per_layer_never_exceeds_the_total_budget(synthetic_path):
    comp = synthetic_compiled(synthetic_path)
    split = comp.split_index(0.3)
    ctx = sim._pool_context(comp, comp.text, split)
    layout = pol.make_layout(comp.text, "per-layer", comp.text.n_items // 3, ctx.train_freq)
    seed = pol.select_top(ctx.train_freq, layout)
    log = []
    pol.run_adaptive(comp, comp.text, layout, split, pol.AdaptiveParams(), seed, round_log=log)
    assert log and max(entry[5] for entry in log) <= layout.budget
    assert all(entry[4] <= 96 for entry in log)


# ----------------------------------------------------------------------------- layout / budgets


def test_budget_parsing():
    total = 24576
    assert sim.parse_budgets("4096,0.5,32GiB", total) == [4096, 12288, int(32 * sim.GIB / sim.DEFAULT_EXPERT_BYTES)]
    assert sim.parse_budgets("25%, 1.0, 1, 999999", total) == [6144, 24576, 1]
    assert sim.parse_budgets("100MiB", total)[0] == int(100 * sim.MIB / sim.DEFAULT_EXPERT_BYTES)
    with pytest.raises(ValueError):
        sim.parse_budgets("lots", total)


def test_per_layer_quotas_follow_the_profile_or_are_uniform(tmp_path):
    comp = compile_path(write_trace(tmp_path / "t.nxtr", [decode(1, 0, 0)], text_layers=3, experts=4, top_k=1))
    uniform = pol.make_layout(comp.text, "per-layer", 7)
    assert uniform.quota.tolist() == [3, 2, 2]
    scores = np.zeros(12)
    scores[[0, 1, 2, 4]] = [9, 8, 7, 6]  # layer 0 holds three of the best four
    profile = pol.make_layout(comp.text, "per-layer", 4, scores)
    assert profile.quota.tolist() == [3, 1, 0] and profile.budget == 4
    top = pol.select_top(scores, profile)
    assert np.flatnonzero(top).tolist() == [0, 1, 2, 4]
    assert pol.make_layout(comp.text, "global", 4).quota.tolist() == [4]


# ------------------------------------------------------------------------------- request log


def log_line(event, instance, service_id, *, client=None, engine_id=None, ts=0, protocol="openai_chat_completions", **extra):
    request = {"request_id": service_id, "protocol": protocol}
    if client is not None:
        request["client"] = client
    obj = {"artifact_type": "ninfer_serve_request_log", "event": event, "server_instance_id": instance,
           "timestamp_unix_ms": ts, "request": request, **extra}
    if engine_id is not None:
        obj["generation"] = {"engine_request_id": engine_id}
    return json.dumps(obj)


def test_request_log_join_uses_the_request_id_and_filters_the_server_instance(tmp_path):
    lines = [
        json.dumps({"event": "server_start", "server_instance_id": SERVER, "timestamp_unix_ms": 1}),
        '{"server_instance_id": "' + SERVER + '", "event": "request_done", "request": {"request_i',  # torn line
        log_line("request_start", SERVER, 20, client="app-a/1", ts=2),
        # the engine id is a different counter and must play no part in the join
        log_line("request_done", SERVER, 20, client="app-a/1", engine_id=7, ts=3),
        log_line("request_start", SERVER, 21, client="app-b/2", ts=4),
        log_line("request_done", SERVER, 21, client="app-b/2", engine_id=8, ts=5, protocol="anthropic_messages"),
        log_line("request_done", SERVER, 22, client="", engine_id=9, ts=6),
        # the same ids from another server instance must not leak in
        log_line("request_done", "serve-2-200", 20, client="intruder/1", engine_id=7, ts=7),
        log_line("request_done", "serve-2-200", 23, client="intruder/1", engine_id=10, ts=8),
        # a request that failed after it started
        log_line("request_start", SERVER, 24, client="app-a/1", ts=9),
        log_line("request_error", SERVER, 24, client="app-a/1", ts=10),
        # scheduling events and rejections are not part of the join
        json.dumps({"event": "request_scheduling", "server_instance_id": SERVER, "timestamp_unix_ms": 11,
                    "request": {"request_id": 26}, "engine_request_id": 11, "transition": "paused"}),
        log_line("request_rejected", SERVER, 25, client="app-c/1", ts=12),
    ]
    path = tmp_path / "requests.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    join = load_request_log(path, SERVER)
    assert join.labels == {20: "app-a/1", 21: "app-b/2", 22: NO_USER_AGENT, 24: "app-a/1"}
    stripped = load_request_log(path, SERVER, strip_version=True)
    assert stripped.labels[20] == "app-a" and stripped.labels[21] == "app-b"
    by_protocol = load_request_log(path, SERVER, group_by=["client", "protocol"])
    assert by_protocol.labels[21] == "app-b/2|anthropic_messages"
    assert by_protocol.labels[20] == "app-a/1|openai_chat_completions"


def test_request_log_scan_starts_near_the_trace_in_a_large_log(tmp_path):
    pad = "x" * 200
    lines = [log_line("request_done", SERVER, i, client="app/1", ts=i, pad=pad) for i in range(4000)]
    path = tmp_path / "requests.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    join = load_request_log(path, SERVER, since_unix_ms=3000)
    assert join.lines_scanned < 2000  # bisected, not read from the start
    assert 3000 in join.labels and 3999 in join.labels and 10 not in join.labels
    tail = load_request_log(path, SERVER, tail_bytes=100_000)
    assert 3999 in tail.labels and 10 not in tail.labels
    assert len(load_request_log(path, SERVER).labels) == 4000


def test_client_lookup_groups_untagged_and_unjoined_requests():
    lookup = ClientLookup({10: "x", 11: "y", 12: "x", 0: "never"})
    idx = lookup.indices(np.array([10, 11, 99, 12, 0], dtype=np.uint64))
    assert [lookup.names[i] for i in idx] == ["x", "y", "(unjoined)", "x", "(untagged)"]
    assert ClientLookup({}).indices(np.array([1, 0], dtype=np.uint64)).tolist() == [0, 1]


def test_untagged_columns_are_their_own_client(tmp_path):
    recs = [decode(0, 0, 1), decode(5, 0, 2), dict(kind=KIND_DECODE, request_ids=[0, 5], positions=[1, 1],
                                                   ids=np.array([[[1]], [[3]]]))]
    comp = compile_path(write_trace(tmp_path / "t.nxtr", recs), mapping={5: "app"})
    names = comp.client_names
    assert names == ["app", "(unjoined)", "(untagged)"]
    assert comp.rec_client.tolist() == [2, 0, -2]  # untagged, app, a record mixing both
    resident = np.zeros(4, dtype=bool)
    resident[1] = True
    tally = pol.run_static(comp, comp.text, resident, split=0)
    untagged, app = names.index("(untagged)"), names.index("app")
    assert (tally.client_needed[untagged], tally.client_hits[untagged]) == (2, 2)
    assert (tally.client_needed[app], tally.client_hits[app]) == (2, 0)


# ------------------------------------------------------------------------ prefill histograms


def test_histogram_round_trip(tmp_path):
    rng = np.random.default_rng(5)
    ids = rng.integers(0, 16, size=(9, 3, 4))
    counts = ids_to_histogram(ids, 16)
    assert counts.dtype == np.uint16 and counts.shape == (3, 16) and (counts.sum(axis=1) == 9 * 4).all()
    path = tmp_path / "t.nxtr"
    meta = make_metadata(text_layers=3, experts=16, top_k=4, mtp_layer=None, server_instance_id=SERVER)
    with TraceWriter(path, meta) as writer:
        writer.write_record(KIND_DECODE, [7], [3], ids[:1], timestamp_us=10)
        writer.write_histogram(77, 1000, 9, counts, timestamp_us=20)
        writer.write_histogram(0, 4, 2, counts[1:], first_layer=1, timestamp_us=30)
        writer.write_record(KIND_DECODE, [7], [4], ids[1:2], timestamp_us=40)
    _meta, records, reader = read_trace(path)
    assert reader.status == "complete" and [r.kind for r in records] == [KIND_DECODE, 5, 5, KIND_DECODE]
    first, second = records[1], records[2]
    assert isinstance(first, HistogramRecord)
    assert (first.request_id, first.first_position, first.tokens, first.first_layer, first.timestamp_us) == (77, 1000, 9, 0, 20)
    assert first.columns == 1 and first.layer_count == 3 and first.kind_name == "prefill-histogram"
    np.testing.assert_array_equal(first.counts, counts)
    assert (second.request_id, second.first_layer, second.layer_count) == (0, 1, 2)
    np.testing.assert_array_equal(second.counts, counts[1:])
    np.testing.assert_array_equal(records[3].ids, ids[1:2])  # the per-token record after it still parses


def test_histogram_counts_saturate_at_u16():
    ids = np.zeros((70000, 1, 1), dtype=np.uint16)
    assert ids_to_histogram(ids, 4).tolist() == [[65535, 0, 0, 0]]


def test_writer_rejects_malformed_histograms(tmp_path):
    with TraceWriter(tmp_path / "t.nxtr", make_metadata(text_layers=1, experts=4, top_k=1)) as writer:
        with pytest.raises(TraceFormatError):
            writer.write_histogram(1, 0, 3, np.zeros((1, 5)))  # wrong expert axis
        with pytest.raises(TraceFormatError):
            writer.write_histogram(1, 0, 3, np.full((1, 4), 70000))  # does not fit u16


def test_truncated_histogram_tail_is_tolerated(tmp_path):
    ids = np.array([[[1, 2]], [[2, 3]], [[0, 3]]])
    _rows, summary = prefill_pair(4, 10, ids, 8)
    path = write_trace(tmp_path / "t.nxtr", [decode(4, 9, 1, 2), summary], experts=8, top_k=2)
    data = path.read_bytes()
    record_size = 24 + 16 + 1 * 8 * 2
    for cut in (1, 12, 24, 30, 40, record_size - 1):  # inside the header, the summary column, the counts
        short = tmp_path / f"cut{cut}.nxtr"
        short.write_bytes(data[: len(data) - record_size + cut])
        _meta, records, reader = read_trace(short)
        assert [r.kind for r in records] == [KIND_DECODE], cut
        assert reader.status == "truncated" and "1 complete records" in reader.status_detail
    assert read_trace(path)[2].status == "complete"
    # a histogram that claims several summary columns is corrupt, not truncated
    bad = bytearray(data)
    bad[len(data) - record_size + 12 : len(data) - record_size + 16] = (2).to_bytes(4, "little")
    corrupt = tmp_path / "corrupt.nxtr"
    corrupt.write_bytes(bytes(bad))
    _meta, records, reader = read_trace(corrupt)
    assert len(records) == 1 and reader.status == "corrupt"


def chunked_trace(tmp_path, name, *, histogram):
    """Four requests, each prefilled (request 2 in two chunks) and then decoded. Layers 0..1 are
    text layers and layer 2 the MTP layer, which the first chunk of every request also covers."""
    rng = np.random.default_rng(11)
    records = []
    position = {}
    for request in (1, 2, 3, 4):
        for chunk in range(2 if request == 2 else 1):
            n = int(rng.integers(6, 14))
            ids = rng.integers(0, 16, size=(n, 3 if chunk == 0 else 2, 3))
            rows, summary = prefill_pair(request, position.get(request, 0), ids, 16)
            records.append(summary if histogram else rows)
            position[request] = position.get(request, 0) + n
        for step in range(5):
            experts = rng.integers(0, 16, size=3).tolist()
            records.append(decode(request, position[request] + step, *experts))
    return compile_path(
        write_trace(tmp_path / name, records, text_layers=2, experts=16, top_k=3, mtp_layer=2),
        mapping={1: "app-a", 2: "app-b", 3: "app-a", 4: "app-b"}, prefill_topk=6,
    )


def test_histogram_chunk_equals_the_per_token_chunk(tmp_path):
    rows = chunked_trace(tmp_path, "rows.nxtr", histogram=False)
    summary = chunked_trace(tmp_path, "hist.nxtr", histogram=True)
    for pool in ("text", "mtp"):
        a, b = getattr(rows, pool), getattr(summary, pool)
        np.testing.assert_array_equal(a.entries.offsets, b.entries.offsets)
        np.testing.assert_array_equal(a.entries.items, b.entries.items)
        np.testing.assert_array_equal(a.entries.counts, b.entries.counts)
    np.testing.assert_array_equal(rows.ncols, summary.ncols)  # a histogram chunk counts its tokens
    np.testing.assert_array_equal(rows.rec_client, summary.rec_client)
    np.testing.assert_array_equal(rows.committed, summary.committed)
    assert rows.prefill_lists.keys() == summary.prefill_lists.keys() and rows.prefill_lists
    for r, expected in rows.prefill_lists.items():
        for (items_a, counts_a), (items_b, counts_b) in zip(expected, summary.prefill_lists[r]):
            np.testing.assert_array_equal(items_a, items_b)
            np.testing.assert_array_equal(counts_a, counts_b)
    kwargs = dict(budgets=[8, 20], policies=list(sim.POLICY_ORDER), train_fraction=0.4, slots=("global", "per-layer"),
                  interval=2, min_count=1.0, prefill_promote=6)
    result_rows, result_summary = run(rows, **kwargs), run(summary, **kwargs)
    assert result_rows == result_summary
    text = result_rows["budgets"][0]["pools"]["text"]["policies"]
    streaming = text["static-oracle"]["prefill"]
    assert streaming["records"] > 0 and streaming["needed"] > 0 and streaming["missing"] > 0
    informed = text["prefill-informed"]
    assert informed["swaps"] > 0 and set(informed["per_client"]) == {"app-a", "app-b"}
    assert all(c["prefill_tokens"] > 0 and c["prefill_needed"] > 0 for c in informed["per_client"].values())


def test_synthetic_prefill_histograms_do_not_change_any_result(tmp_path):
    def compiled(histograms):
        config = replace(synthetic_config(), requests=14, prefill_histograms=histograms)
        path = tmp_path / f"s{histograms}.nxtr"
        build_synthetic(config).write(path)
        return synthetic_compiled(path), read_trace(path)[1]

    (with_hist, records_hist), (with_rows, records_rows) = compiled(True), compiled(False)
    kinds_hist, kinds_rows = Counter(r.kind for r in records_hist), Counter(r.kind for r in records_rows)
    assert kinds_hist[KIND_PREFILL_HISTOGRAM] > 0 and kinds_rows[KIND_PREFILL_HISTOGRAM] == 0
    assert kinds_hist[KIND_PREFILL] + kinds_hist[KIND_PREFILL_HISTOGRAM] == kinds_rows[KIND_PREFILL]
    kwargs = dict(budgets=[32, 64], policies=list(sim.POLICY_ORDER), train_fraction=0.3, slots=("global", "per-layer"),
                  prefill_promote=16)
    assert run(with_hist, **kwargs) == run(with_rows, **kwargs)
    assert analysis.prefill_report(with_hist)["text"]["tokens"] == analysis.prefill_report(with_rows)["text"]["tokens"]


def test_prefill_report_covers_both_kinds(synthetic_path):
    report = analysis.prefill_report(synthetic_compiled(synthetic_path))
    text = report["text"]
    assert set(text["by_kind"]) == {"prefill", "prefill-histogram"}
    assert text["chunks"] == sum(v["chunks"] for v in text["by_kind"].values())
    assert text["tokens"] == sum(v["tokens"] for v in text["by_kind"].values())
    distinct = text["distinct_per_record_layer"]
    assert set(distinct["by_kind"]) == {"prefill", "prefill-histogram"}
    # a long chunk touches more experts per layer than a short tail does
    assert distinct["by_kind"]["prefill-histogram"]["mean"] > distinct["by_kind"]["prefill"]["mean"]
    assert distinct["max"] <= text["experts_per_layer"]


# --------------------------------------------------------------------------- synthetic traces


def synthetic_config():
    return SyntheticConfig(text_layers=4, experts=32, top_k=4, mtp=True, clients=3, requests=36, seed=7,
                           prompt_len=(40, 120), decode_len=(30, 80), prefill_chunk=64, client_pref_size=6,
                           client_shared=0.0, client_pref_boost=12.0, focus_size=4, server_instance_id=SERVER)


@pytest.fixture(scope="module")
def synthetic_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("synthetic") / "s.nxtr"
    trace = build_synthetic(synthetic_config())
    trace.write(path)
    path.with_suffix(".jsonl").write_text("\n".join(trace.request_log_lines()) + "\n", encoding="utf-8")
    return path


def test_synthetic_trace_is_well_formed_and_repeatable(synthetic_path, tmp_path):
    metadata, records, reader = read_trace(synthetic_path)
    assert reader.status == "complete" and metadata.mtp_layer == 4
    kinds = Counter(r.kind for r in records)
    # long prompt chunks (tokens * top_k > experts) are histograms, short tails stay per-token rows
    assert set(kinds) == {KIND_PREFILL, KIND_PREFILL_HISTOGRAM, KIND_MTP_PROPOSAL, KIND_MTP_VERIFY}
    for rec in records:
        if rec.kind == KIND_MTP_PROPOSAL:
            assert (rec.first_layer, rec.layer_count) == (4, 1)
        else:
            assert (rec.first_layer, rec.layer_count) == (0, 4)
        if rec.kind == KIND_PREFILL_HISTOGRAM:
            assert isinstance(rec, HistogramRecord) and rec.tokens * 4 > 32
            assert (rec.counts.sum(axis=1) == rec.tokens * 4).all()  # every token picked top_k experts per layer
            continue
        assert rec.ids.max() < 32
        if rec.kind == KIND_PREFILL:
            assert rec.columns * 4 <= 32
        assert all(len(set(row)) == 4 for row in rec.ids.reshape(-1, 4).tolist())  # distinct experts per token and layer
    again = tmp_path / "again.nxtr"
    build_synthetic(synthetic_config()).write(again)
    assert again.read_bytes() == synthetic_path.read_bytes()


def synthetic_compiled(path, *, prefill_topk=64):
    reader = TraceReader(path)
    try:
        mapping = sim.synthetic_clients(reader.metadata)
        return compile_trace(reader.records(), reader.metadata, client_lookup=ClientLookup(mapping),
                             prefill_topk=prefill_topk)
    finally:
        reader.close()


def test_synthetic_policy_orderings(synthetic_path):
    comp = synthetic_compiled(synthetic_path)
    total = comp.text.n_items
    budgets = [total // 4, total // 2]
    split = run(comp, budgets=budgets, policies=list(sim.POLICY_ORDER), train_fraction=0.3, slots=("global", "per-layer"))
    for entry in split["budgets"]:
        text = entry["pools"]["text"]["policies"]
        assert text["static-oracle"]["hits"] >= text["static-train"]["hits"]
        assert text["static-oracle"]["needed"] == text["lru"]["needed"] == text["belady"]["needed"]
        for name, m in text.items():
            assert "skipped" not in m, name
            assert 0 <= m["hits"] <= m["needed"], name
    whole = run(comp, budgets=budgets, policies=["static-oracle", "lru", "belady", "adaptive"], train_fraction=0.0,
                slots=("global", "per-layer"))
    by_mode = {}
    for entry in whole["budgets"]:
        for pool in ("text", "mtp"):
            p = entry["pools"][pool]["policies"]
            assert p["belady"]["misses"] <= p["lru"]["misses"], (entry["slots"], pool)
        by_mode[(entry["slots"], entry["budget_experts"])] = entry["pools"]["text"]["policies"]["static-oracle"]["hits"]
    for budget in budgets:  # fixed per-layer quotas can only cost hits against a free global choice
        assert by_mode[("global", budget)] >= by_mode[("per-layer", budget)]
    assert by_mode[("global", budgets[0])] <= by_mode[("global", budgets[1])]


def test_mtp_pool_can_be_kept_resident(synthetic_path):
    comp = synthetic_compiled(synthetic_path)
    result = run(comp, budgets=[comp.text.n_items // 4], policies=["lru"], mtp_budget="resident")
    assert result["budgets"][0]["pools"]["mtp"]["resident"] is True


def test_join_by_request_log_matches_the_synthetic_metadata(synthetic_path):
    reader = TraceReader(synthetic_path)
    truth = sim.synthetic_clients(reader.metadata)
    reader.close()
    join = load_request_log(synthetic_path.with_suffix(".jsonl"), SERVER)
    assert join.labels == truth and len(truth) == 36


def test_skew_and_client_overlap(synthetic_path):
    comp = synthetic_compiled(synthetic_path)
    skew = analysis.skew_report(comp)
    cover = skew["text"]["coverage"]
    assert cover["50%"]["max"] <= cover["80%"]["max"] <= cover["95%"]["max"] <= 32
    assert 1 <= skew["text"]["distinct_per_record_layer"]["mean"] <= 32
    for columns, stats in skew["mtp"]["distinct_per_record_layer"]["by_columns"].items():
        assert stats["mean"] <= int(columns) * 4  # one layer, top-4 per column
    overlap = analysis.client_overlap(comp, top_ns=(4,), min_columns=50, max_clients=3)
    stats = overlap["by_top_n"]["4"]
    assert len(overlap["clients"]) == 3
    # distinct clients prefer distinct experts here, so they overlap less than a client with itself
    assert stats["cross_client_mean"] < stats["within_client_split_half_mean"]


def test_cli_end_to_end_writes_json(synthetic_path, tmp_path, capsys):
    out = tmp_path / "out.json"
    code = sim.main([str(synthetic_path), "--budgets", "32,0.5", "--policies", "static-oracle,lru,adaptive,belady",
                     "--request-log", str(synthetic_path.with_suffix(".jsonl")), "--json", str(out), "--quiet",
                     "--min-client-columns", "50", "--overlap-top", "4"])
    assert code == 0
    text = capsys.readouterr().out
    assert "Skew" in text and "per-client hit%" in text and "Cross-client overlap" in text
    report = json.loads(out.read_text())
    assert [b["budget_experts"] for b in report["simulation"]["budgets"]] == [32, 64]
    assert report["trace"]["requests_joined"] == 36
    assert report["simulation"]["budgets"][0]["pools"]["text"]["policies"]["lru"]["swap_mib_per_token"] > 0


def test_cli_reports_a_truncated_trace(synthetic_path, tmp_path, capsys):
    data = synthetic_path.read_bytes()
    cut = tmp_path / "cut.nxtr"
    cut.write_bytes(data[:-5])
    assert sim.main([str(cut), "--budgets", "32", "--policies", "static-oracle", "--quiet", "--no-skew"]) == 0
    assert "trace is truncated" in capsys.readouterr().out
