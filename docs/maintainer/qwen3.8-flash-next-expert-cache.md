# Qwen3.8-Flash-Next single-GPU expert cache

This is the residency plan for the 48 text routed-expert banks of
`qwen3.8-flash-next/mixed-nvfp4-fp8-ple-int4`. It is the concurrency lever for one
RTX PRO 6000-class card that already holds the model and still needs eight 256K FP8 KV
planes. It is not a 32 GB product, not an RTX 5090 recipe, and not a dual-GPU or
tensor-parallel path.

The mathematical contract is unchanged: the cache copies the stored NVFP4 code, K16
scale, and FP32 divisor words. It does not requantize them. Router, shared expert, and
the independent shared-expert gate stay on device with the rest of the non-expert weights.

## Why the resident set is large

Packed text routed experts are `kExpertCacheGeometry.text_bank_bytes` =
67,947,921,408 bytes (63.280 GiB): 48 layers, 512 experts, gate/up `[512,1280,2560]`
and down `[512,2560,640]`. One expert's raw words are 2,764,808 bytes. A device slot
aligns each plane to 256 bytes, so a slot is 2,765,312 bytes. The extra bytes are
padding, not new weights.

The released artifact's device payload with Text, MTP, and Vision, and with the BF16
MTP expert banks left mapped, is 76,251,938,528 bytes (71.02 GiB). Subtracting the text
banks leaves 8,304,017,120 bytes of non-expert device payload, including vision, the
routers, and the shared experts. The loader's NVFP4 MTP banks add one more layer,
1,415,581,696 bytes. Together that is the ~72.33 GiB stand-in.

A Colab G4 reports 101,974,081,536 bytes. One 256K FP8 KV plane with MTP is
3,707,764,736 bytes (13 QSA layers). Eight planes are 29,662,117,888 bytes. Full expert
residency plus those eight planes overshoots that card by about 5.0 GiB of payload.
Six planes fit, and the payload remainder is 2,059,972,896 bytes, which is less than a
seventh plane. That matches the empty stand-in that placed six planes and failed the
seventh with about 1.39 GiB of driver-reported free memory; the gap between 2.06 GiB of
payload remainder and 1.39 GiB free is context, alignment, and allocator overhead, which
this arithmetic does not measure.

Freeing 10 GiB of text-expert payload leaves room for planes 7 and 8 (6.91 GiB) and
about 5 GiB of payload headroom on that G4. The 12–18 GiB cache from issue #27 is the
sketch for a 32 GB card: a 16 GiB resident budget would drop more than 40 GiB of experts
and is the wrong default here.

## Budget

`ExpertCachePlan` is the clamp.

| Request | Result |
|---|---|
| `0`, or at least the packed text banks | Cache off. The 96 text banks stay in the artifact device arena. |
| Anything smaller | `resident_slots = floor(request / slot_bytes)`, device bytes = slots × 2,765,312, never above the request. |

The PRO 6000 request is packed banks minus 10 GiB (`flash_next_pro6000_expert_cache_budget_bytes`).
Slot flooring realizes a slightly larger release, still under 12 GiB, and a resident cache
well above 18 GiB. `FlashNextRuntimePlan::pro6000_expert_cache` always carries that plan.
`expert_cache` is the plan this process actually bound.

The process default is full residency. The contiguous-bank MoE kernels are still the
execution path, and launching them against file-mapped banks would capture host
addresses. Opt in with either of:

```text
NINFER_FLASH_NEXT_EXPERT_CACHE=pro6000
NINFER_FLASH_NEXT_EXPERT_CACHE_BUDGET_BYTES=<bytes|full|<n>GiB>
```

The byte variable wins when both are set. `12GiB` is a resident budget of 12 GiB, not a
request to free 12 GiB.

## What stays on device

- Token embedding, output head, hyper-connections, GDN, QSA, PLE projections, final mixer.
- Every layer's router and shared expert, including `shared_expert_gate`.
- Vision, when the load asked for it.
- MTP, including its NVFP4 expert banks (or the loader's BF16-to-NVFP4 buffers). MTP is
  one layer and is not part of the 63.28 GiB text cache.
- The expert-cache slot arena, when the cache is on.
- KV, graphs, and workspaces, unchanged by this plan.

## Miss path

Text `experts/gate_up` and `experts/down` are `retain_mapped_tensor` when the cache is
on, the same file-mapping placement as the PLE table. They are not pinned and they are
not read from an SSD cache. The operating system can reclaim the pages.

`ExpertCacheDevice` owns:

- the device slot arena;
- a pinned staging window of at most 16 slots (44,244,992 bytes at the production slot
  size);
- a pointer table, 48 layers × 6 planes × 512 device pointers.

`prepare(layer, local_expert_ids, host_banks, stream)` is the miss path:

1. `ExpertCacheDirectory::touch` admits an absent id or refreshes a hit. The id is
   `layer * 512 + expert` and does not change when the slot assignment does.
2. A miss packs that expert's code, scale, and divisor bytes into one pinned slot.
   Padding is zero. The NVFP4 words are copied, not recomputed.
3. The slot is copied to the device arena on the caller's stream. A list longer than
   the window waits on that stream once per extra wave, then reuses a staging slot.
   Decode's 10 experts fit in one wave and do not wait inside that prepare. The next
   `prepare` waits on one event for the previous prepare's copies before it writes the
   staging window or the pointer mirror again. There is no poll loop.
4. Each touched layer's pointer table is published on the same stream after the copies.

Eviction is least-recently-used. A hit moves the id to the newest end. Capacity zero
records a miss and admits nothing.

## CUDA graphs

The pointer-table allocation is stable for the life of the cache. Replay must load
`plane_table(layer, plane)[expert]` rather than `bank_base + expert * stride`. Updating
the table before replay changes the resident set without recapture and without baking a
payload address into the graph.

This change does not rewrite `moe_kernels.cu` or the decode graph in `text_executor`.
`flash_next_moe` refuses a layer whose `routed_expert_cache` is set, so a cache-enabled
load cannot silently run the contiguous kernels. The execution Program, when it lands,
calls `prepare` for the ids it is about to consume and gathers through the table.
Ids produced inside a captured full-model graph are not visible to the host until the
graph returns; the Program has to publish the table for those ids before replay, or
split the capture at the MoE boundary. That Program work is not in this change.

## What a Colab G4 still has to prove

No GPU and no Flash-Next artifact were available for this change. The host test checks
the budget clamp, LRU hit/miss, stable ids, and byte-identical slot packing. It does
not measure:

- the hit rate of this LRU (or of a later routing profile) at the PRO 6000 slot count;
- the PCIe cost of a miss, which moves 2,765,312 bytes per expert and up to 10 experts
  per layer per token;
- that eight 256K FP8+MTP KV planes plus the reduced weight footprint, the slot arena,
  and the graph/activation workspace actually malloc on a 101,974,081,536-byte G4.

Payload arithmetic says that configuration fits with at least 4 GiB left before context
and allocator overhead. The malloc is the proof.

## Cherry-pick onto workstation

Do not rebase this branch onto workstation, and do not port the Flash-Next execution
Program as part of this change. Cherry-pick the new files first, then reapply the
integration hunks after the execution port lands. Expect conflicts in the files that
port is already editing.

New files, no existing-history conflict:

- `src/targets/qwen3_8_flash_next/impl/expert_cache.h`
- `src/targets/qwen3_8_flash_next/impl/expert_cache.cpp`
- `src/targets/qwen3_8_flash_next/impl/expert_cache_device.h`
- `src/targets/qwen3_8_flash_next/impl/expert_cache_device.cu`
- `tests/targets/qwen3_8_flash_next/test_expert_cache.cpp`
- `docs/maintainer/qwen3.8-flash-next-expert-cache.md`

Integration hunks, likely to collide with the execution port:

- `src/targets/qwen3_8_flash_next/impl/load/bindings.h`
- `src/targets/qwen3_8_flash_next/impl/load/bindings.cpp`
- `src/targets/qwen3_8_flash_next/impl/load/loader.h`
- `src/targets/qwen3_8_flash_next/impl/load/loader.cpp`
- `src/targets/qwen3_8_flash_next/impl/load/materialized.h`
- `src/targets/qwen3_8_flash_next/impl/load/materialized.cpp`
- `src/targets/qwen3_8_flash_next/impl/expert_bank.h`
- `src/targets/qwen3_8_flash_next/impl/expert_bank.cpp`
- `src/targets/qwen3_8_flash_next/impl/model_view.h`
- `src/targets/qwen3_8_flash_next/impl/runtime_plan.h`
- `src/targets/qwen3_8_flash_next/impl/runtime_plan.cpp`
- `src/targets/qwen3_8_flash_next/impl/package.cpp`
- `src/targets/qwen3_8_flash_next/impl/moe.h`
- `src/targets/qwen3_8_flash_next/impl/moe.cpp`
- `src/targets/qwen3_8_flash_next/CMakeLists.txt`
- `tests/CMakeLists.txt`
- `docs/maintainer/qwen3.8-flash-next-model.md`
- `docs/maintainer/qwen3.8-flash-next-artifact.md`
- `docs/README.md`

The execution port is also likely to edit these, which this change does not touch.
Rebase against them rather than assuming they stayed still:

- `src/targets/qwen3_8_flash_next/impl/program.cpp`
- `src/targets/qwen3_8_flash_next/impl/program_impl.h`
- `src/targets/qwen3_8_flash_next/impl/text_executor.cpp`
- `src/targets/qwen3_8_flash_next/impl/text_executor.h`
- `src/targets/qwen3_8_flash_next/impl/text_decode.cpp`
- `src/targets/qwen3_8_flash_next/impl/text_decode.h`
- `src/targets/qwen3_8_flash_next/impl/text_decode_kernels.cu`
- `src/targets/qwen3_8_flash_next/impl/moe_kernels.cu`
- `src/targets/qwen3_8_flash_next/impl/moe_kernels.h`
- `src/targets/qwen3_8_flash_next/impl/moe_shared_kernels.cu`
- `src/targets/qwen3_8_flash_next/impl/mtp_forward.cpp`
- `src/targets/qwen3_8_flash_next/impl/runtime_state.cpp`
- `src/targets/qwen3_8_flash_next/impl/runtime_state.h`

There is no `docs/maintainer/upstream-ports.md` on this research branch. This file is
the cherry-pick list.
