#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * QSA indexer persistent state.
 *
 * Forming-block state, one entry per state slot s:
 *   raw_keys      BF16 [128,4,S]: raw_keys[:,r,s] is the raw indexer key of the r-th token of the
 *                 block currently being formed;
 *   raw_positions I32  [3,4,S]:   the three MRoPE position axes of that token.
 * After a token at cache position p has been appended into slot s, entries r < (p+1) mod 4 of s
 * hold the tokens 4*floor((p+1)/4) + r of the incomplete block; the remaining entries are
 * unspecified.
 */
struct QsaIndexerKeyState {
    Tensor raw_keys;
    Tensor raw_positions;
};

/**
 * Paged plane of completed block keys, addressed through the Main Text KV block table.
 *
 * keys is BF16 [128,16,Npages]: page g holds the 16 blocks of its 64 tokens (the same 64-token
 * page as the main K/V). Block b of the sequence on table row r lives at
 * keys[:, b mod 16, block_tables[floor(b/16), r]]. block_tables is contiguous I32 [L,R]; the
 * shared-row forms take a single-row table (R = 1). An allocator can provide the plane as a
 * [32,64,1,Npages] BF16 page-major plane beside the main K/V planes; the Op views it as
 * [128,16,Npages].
 */
struct QsaIndexerBlockKeys {
    Tensor keys;
    Tensor block_tables;
};

/**
 * Op: QSA indexer key append (stateful).
 *
 * Math / indexing:
 *   The indexer projection column holds 4 query heads x 128 followed by one raw key
 *   (rows [512,640)). Appending the token at cache position p with raw key k and MRoPE positions
 *   (a0,a1,a2) sets forming entry r = p mod 4 to (k, a). When r = 3 the block b = floor(p/4) is
 *   complete and its key is published:
 *
 *     m[d]      = BF16_RNE( (k0[d] + k1[d] + k2[d] + k3[d]) / 4 )      (FP32 mean, BF16 cast)
 *     inv       = 1 / sqrt(sum_d m[d]^2 / 128 + 1e-6)
 *     n[d]      = m[d] * inv * (1 + key_norm[d])
 *     phi(i)    = a_first[i mod 3] * (1e7)^(-2i/64),  0 <= i < 32, a_first = positions of k0
 *     out[i]    = n[i] cos phi(i) - n[i+32] sin phi(i)
 *     out[i+32] = n[i+32] cos phi(i) + n[i] sin phi(i)
 *     out[d]    = n[d],  64 <= d < 128
 *     keys[:, b mod 16, table[b/16]] = BF16(out)
 *
 *   The BF16 cast of the block mean is the checkpoint's semantic seam (its pooled key is a BF16
 *   tensor); normalization and rotation have no observable intermediate rounding.
 *
 * Logical shapes / forms:
 *   Shared-row form: projected BF16 [640,T], positions I32 [T] consecutive (p0, p0+1, ...),
 *   rope_positions I32 [T,3] planar, one host source slot and destination slot, a single-row
 *   block table; T >= 1. The source slot holds the forming block of p0 (entries r < p0 mod 4);
 *   after the call the destination slot holds the forming block after the last token. Source and
 *   destination may be the same slot.
 *
 *   Speculative / decode form: projected BF16 [640,W,B], positions I32 [W,B] consecutive along W,
 *   rope_positions I32 [W*B,3] planar with column index w + W*b, table_rows, initial_slots and
 *   snapshot_base_slots I32 [B], W in [1,16], B in [1,8]. Row b starts from initial_slots[b]; after
 *   column w its forming state is written to slot snapshot_base_slots[b] + w and every block
 *   completed at column w is published. The caller reserves each row's [base, base+W) interval,
 *   the intervals of different rows are disjoint and do not contain another row's initial slot;
 *   a row's own initial slot may lie inside its interval. Optional valid_columns I32 [B]
 *   bounds each row to [1,W] columns; omitted means W. Invalid suffixes neither publish keys
 *   nor write state snapshots and may contain arbitrary data.
 *
 * Supported domain: key_norm BF16 [128]; state and plane tensors as described above; all tensors
 *   contiguous and 16-byte aligned; every published block's page must be materialized.
 *
 * Numeric: the oracle evaluates the formula above in FP64 after the BF16 mean cast (exact
 *   comparison of that cast is not required when the FP32 sum is inexact). Raw keys and positions
 *   are copied bit for bit.
 *
 * Effects: writes the destination/snapshot slots' entries and the published block keys only. A
 *   block published from a speculative column that the schedule later rejects lies beyond the
 *   committed frontier and is republished when that block completes again. The Op neither
 *   decides nor publishes a frontier.
 *
 * Workspace: none. Execution: graph capturable; no synchronization.
 */
void qsa_indexer_append(const Tensor& projected, const Tensor& positions,
                        const Tensor& rope_positions, const Tensor& key_norm,
                        std::int32_t source_slot, std::int32_t destination_slot,
                        const QsaIndexerKeyState& state, const QsaIndexerBlockKeys& blocks,
                        cudaStream_t stream);

void qsa_indexer_append(const Tensor& projected, const Tensor& positions,
                        const Tensor& rope_positions, const Tensor& table_rows,
                        const Tensor& initial_slots, const Tensor& snapshot_base_slots,
                        const Tensor& valid_columns,
                        const Tensor& key_norm, const QsaIndexerKeyState& state,
                        const QsaIndexerBlockKeys& blocks, cudaStream_t stream);

/**
 * Host execution envelope of qsa_indexer_select: every column's complete-block count
 * floor((positions[c]+1)/4) is at most max_complete_blocks (>= 0). It bounds workspace and route
 * selection; the device positions still define the result.
 */
struct QsaIndexerSelectEnvelope {
    std::int32_t max_complete_blocks = 0;
};

/**
 * Op: QSA indexer block selection.
 *
 * Math / indexing:
 *   For column c with cache position p = positions[c], complete = floor((p+1)/4). The query
 *   heads h in [0,4) are rows [128h,128h+128) of projected[:,c], one-centered RMS-normalized with
 *   query_norm and rotated exactly like a block key (same formula with the column's own
 *   rope_positions). For every complete block b < complete with key K_b (published by
 *   qsa_indexer_append):
 *
 *     score(b) = sum_h max(0, q_h . K_b) / sqrt(128)
 *
 *   The selected set S is all complete blocks when complete <= 512, otherwise the 512 blocks with
 *   the largest scores, ordered by descending score and then ascending block id (an exact score
 *   tie keeps the lower id). selections[0:n,c] lists S in ascending block id, n = min(complete,
 *   512) is written to counts[c], and selections[n:512,c] = -1. The causal tail
 *   [4*complete, p] is implicit and is appended by selected_block_attention.
 *
 * Logical shapes: projected BF16 [640,C]; positions I32 [C]; rope_positions I32 [C,3] planar;
 *   query_norm BF16 [128]; selections I32 [512,C]; counts I32 [C]. Batched form: table_rows I32
 *   [C] select a row of blocks.block_tables for each column, C in [1,128]. Shared-row form: a
 *   single-row table, C >= 1.
 *
 * Numeric: the oracle computes the ideal scores in FP64 from the represented query projection
 *   and stored block keys. Production scores are FP32 values within the suite criterion of the
 *   ideal (query rotation, BF16 staging and reduction association are private); the selection is
 *   exact with respect to the production scores, so it equals the ideal selection whenever the
 *   ideal 512th/513th scores differ by more than that criterion. Blocks with bitwise-identical
 *   keys receive identical scores.
 *
 * Effects: writes selections and counts only. Workspace: caller-owned, sized by
 *   qsa_indexer_select_workspace_capacity_bytes() for the column interval and envelope (zero when
 *   max_complete_blocks <= 512). Execution: graph capturable for a fixed C and envelope.
 */
[[nodiscard]] std::size_t
qsa_indexer_select_workspace_capacity_bytes(std::int32_t min_columns, std::int32_t max_columns,
                                            QsaIndexerSelectEnvelope envelope);

void qsa_indexer_select(const Tensor& projected, const Tensor& positions,
                        const Tensor& rope_positions, const Tensor& table_rows,
                        const Tensor& query_norm, const QsaIndexerBlockKeys& blocks,
                        QsaIndexerSelectEnvelope envelope, WorkspaceArena& workspace,
                        Tensor& selections, Tensor& counts, cudaStream_t stream);

void qsa_indexer_select(const Tensor& projected, const Tensor& positions,
                        const Tensor& rope_positions, const Tensor& query_norm,
                        const QsaIndexerBlockKeys& blocks, QsaIndexerSelectEnvelope envelope,
                        WorkspaceArena& workspace, Tensor& selections, Tensor& counts,
                        cudaStream_t stream);

} // namespace ninfer::ops
