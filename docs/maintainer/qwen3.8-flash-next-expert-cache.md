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
- The expert-cache slot arena and the one-layer gather bank, when the cache is on.
- KV, graphs, and workspaces. The decode graph is split at each text MoE when the cache is on; full residency keeps one graph.

## Miss path

Text `experts/gate_up` and `experts/down` are `retain_mapped_tensor` when the cache is
on, the same file-mapping placement as the PLE table. They are not pinned and they are
not read from an SSD cache. The operating system can reclaim the pages.

`ExpertCacheDevice` owns:

- the device slot arena;
- a pinned staging window of at most 16 slots (44,244,992 bytes at the production slot
  size);
- a pointer table, 48 layers × 6 planes × 512 device pointers;
- one contiguous gate bank and one contiguous down bank, together
  `layer_bank_bytes`, filled by the gather kernel from that table.

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

## Gather

The MoE math kernels still address a contiguous 512-expert bank
(`codes + expert * stride`). They are not retargeted. When a layer's
`routed_expert_cache` is set, `flash_next_moe` does not launch them on the
file-mapped banks.

`prepare` admits the routed ids and publishes `plane_table(layer, plane)[expert]`.
A gather kernel then loads those pointers and copies each selected expert's code,
scale, and divisor words into a one-layer device bank owned by `ExpertCacheDevice`.
The math kernels run on that bank. Full residency never allocates it and never
enters this path. The bank is `layer_bank_bytes` (1,415,581,696). On the G4
8-plane payload that leaves 3,968,008,480 bytes (3.70 GiB) before CUDA context,
graphs, and activation workspace. The same fit without the launch bank was
5,383,590,176 bytes (5.01 GiB).

Eager prefill and eager decode publish and gather on the calling stream before
the math kernels. That publish synchronizes the stream so the host can read the
route ids. It must not run while the stream is capturing.

## CUDA graphs

A captured full-model graph does not execute its route kernels, so the host
cannot see those ids until a segment returns. Decode capture therefore splits
on the text-MoE boundary. `ExpertGatherCapture` records 49 segments
(`kExpertCacheDecodeSegments`): segment `i` ends at the route of layer `i`, and
segment `i + 1` begins with that layer's expert kernels. Replay launches segment
`i`, copies the shared routing-id buffer back, calls `prepare` and the gather,
then launches segment `i + 1`. The math kernels capture the launch-bank
addresses, which do not change when the resident set does. The pointer table is
what moves.

Every text layer's route writes the same device id buffer: the MoE workspace is
scoped inside `flash_next_moe` and the surrounding round allocations rewind
before the next layer. Capture rejects a round where those addresses differ.
MTP's expert banks stay device-resident, so its captured draft steps are not
split.

Unset `NINFER_FLASH_NEXT_EXPERT_CACHE` keeps the single full-round graph.

## What a Colab G4 serve must show

Cache off at `1ad843e` already served: full residency,
`host_to_device_bytes=75172939520`, listen on `127.0.0.1:8010`, and
"Reply with exactly: pong" returned "pong" (17 prompt + 2 completion, TTFT
28.7 ms).

Cache on (`NINFER_FLASH_NEXT_EXPERT_CACHE=pro6000`) must now get past startup
and answer that same prompt. The serve log should show:

- `flash_next expert_cache slots=20688 device_bytes=57208774656 released_bytes=10739146752 pinned_staging_bytes=44244992 gather_bank_bytes=1415581696`
- no `capture failed` line and no refusal that the routed experts are cache-resident
- `ninfer-serve` reaches listen
- the same prompt returns a completion. The text to compare with cache off is `pong`. A different completion means the gather did not feed the math kernels the routed NVFP4 words
- weight `host_to_device_bytes` stays near the non-expert upload (`7225018112` on the previous cache-on attempt). The slot arena and the launch bank are separate device allocations, not part of that weight upload

Startup still runs the eager warmup forward before capture, so the first routing
set is admitted before the server listens. A short generate then pays a host
sync and any misses at every text layer (48) on prefill and on each decode
token. This machine has no GPU, so none of that serve path was executed here.

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
- `src/targets/qwen3_8_flash_next/impl/expert_gather.h`
- `src/targets/qwen3_8_flash_next/impl/expert_gather.cpp`
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
- `src/targets/qwen3_8_flash_next/impl/text_executor.h`
- `src/targets/qwen3_8_flash_next/impl/text_executor.cpp`
- `src/core/decode_graph.h`
- `src/core/decode_graph.cpp`
- `src/targets/qwen3_8_flash_next/CMakeLists.txt`
- `tests/CMakeLists.txt`
- `docs/maintainer/qwen3.8-flash-next-model.md`
- `docs/maintainer/qwen3.8-flash-next-artifact.md`
- `docs/README.md`

The math kernels are unchanged. The execution port is also likely to edit these,
which this change does not touch. Rebase against them rather than assuming they
stayed still:

- `src/targets/qwen3_8_flash_next/impl/program.cpp`
- `src/targets/qwen3_8_flash_next/impl/program_impl.h`
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
