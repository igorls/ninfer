#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

struct SparseMoeWeights {
    Weight router_shared_gate;
    Weight routed_gate_up;
    Weight routed_down;
    Weight shared_gate_up;
    Weight shared_down;
};

enum class SparseMoeEpilogue : std::uint8_t {
    AddResidual, ///< destination = moe(x) + incoming destination.
    Store,       ///< destination = moe(x); the incoming destination is not read.
};

/**
 * One complete NVFP4 expert bank `[E,N,K]` in the `expert_block_scale_k16_m128x4_v1` layout.
 *
 * `codes` holds E expert-major code planes of `N*K/2` bytes (two E2M1 codes per byte, even column
 * in the low nibble), `scales` holds E expert-major K16 block-scale planes of `N*K/16` UE4M3 bytes
 * in the M128x4 swizzle of `storage-layouts.md`, and `weight_scale_divisors` holds E FP32 weight
 * divisors. Logical weight `W[e,n,k] = E2M1(code) * UE4M3(scale[e,n,k/16]) / divisor[e]`. Expert e
 * starts at `e*code_bytes_per_expert` and `e*scale_bytes_per_expert`. A bank Use stores no
 * activation divisor: an A4 route quantizes each activation row with a dynamic scale.
 */
struct Nvfp4ExpertBankWeight {
    const std::byte* codes               = nullptr;
    const std::byte* scales              = nullptr;
    const float* weight_scale_divisors   = nullptr;
    std::int32_t experts                 = 0;
    std::int32_t n                       = 0;
    std::int32_t k                       = 0;
    std::uint64_t code_bytes_per_expert  = 0;
    std::uint64_t scale_bytes_per_expert = 0;
    LinearPolicy policy                  = LinearPolicy::A16Only;
};

/**
 * Weights of the 512-expert, top-10 sparse MoE with NVFP4 expert banks and a BF16 sigmoid-gated
 * shared expert. Every BF16 weight is contiguous row-major `[N,K]` (QType::BF16, Contiguous).
 */
struct SparseMoeNvfp4BankWeights {
    Weight router;                 ///< BF16 [512,2560]
    Weight shared_expert_gate;     ///< BF16 [1,2560]
    Nvfp4ExpertBankWeight gate_up; ///< NVFP4 [512,1280,2560], rows [gate_640, up_640]
    Nvfp4ExpertBankWeight down;    ///< NVFP4 [512,2560,640]
    Weight shared_gate;            ///< BF16 [640,2560]
    Weight shared_up;              ///< BF16 [640,2560]
    Weight shared_down;            ///< BF16 [2560,640]
};

/**
 * Optional per-call execution hints. Every field is a pure cache hint with no numeric effect:
 * the same call with a default-constructed SparseMoeHints produces bit-identical output.
 *
 * next_weight_prefetch names a weight span the next decode-step consumer will stream; the decode
 * D4 epilogue issues fire-and-forget L2 prefetches over its first bytes. The span is caller-owned
 * and read once, inside the call: the Op keeps no state between calls, and no hidden channel
 * carries it.
 */
struct SparseMoeHints {
    const void* next_weight_prefetch       = nullptr;
    std::size_t next_weight_prefetch_bytes = 0;
};

/**
 * Returns the transient capacity required by SparseMoe for every T in the inclusive
 * [min_tokens,max_tokens] interval. The routed QTypes are the fixed implementation profile.
 * Invalid profiles or intervals throw.
 */
[[nodiscard]] std::size_t sparse_moe_workspace_capacity_bytes(QType routed_gate_up,
                                                              QType routed_down,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

/**
 * Closed sparse-MoE Op for the exact future 35B-A3B geometry.
 *
 * For contiguous BF16 x [2048,T] and destination [2048,T] with T>0, 256 routed experts, top-8
 * selection, and one always-on shared expert, this Op owns router projection and selection,
 * selected routed and shared SwiGLU projections, down projections, their merge, and the
 * AddResidual epilogue independently for every token column. At an exact top-8 boundary tie the
 * lower expert id wins. destination is the only observable mutation: its incoming value is the
 * residual and its outgoing value is the BF16 sparse-MoE result plus that residual.
 *
 * The complete mathematical oracle starts from represented BF16 inputs, exact stored-weight
 * decode, and evaluates the logical formula naively in FP32/FP64. Scores, route weights, expert
 * activations, workspace representation, reduction association, and scale placement are private
 * execution choices rather than semantic rounding boundaries.
 *
 * The five weights have the exact registered shapes: BF16 router/shared gate [257,2048], routed
 * gate/up [256*1024,2048], routed down [256*2048,512], shared gate/up [1024,2048], and shared down
 * [2048,512]. Admitted codec profiles are Q4+Q5, Q4+Q6, and Q8+Q8 for the two routed banks; both
 * shared banks are Q8. Expert e directly selects its stored row spans; no selected-weight gather
 * or repack occurs.
 *
 * Every positive T is supported.
 *
 * x, destination, all weight planes, and live workspace must be pairwise non-overlapping.
 * Execution is enqueued on stream without host synchronization. Workspace is caller-owned,
 * graph-stable transient storage and carries no state beyond the call.
 */
void sparse_moe(const Tensor& x, const SparseMoeWeights& weights, SparseMoeEpilogue epilogue,
                Tensor& destination, WorkspaceArena& workspace, cudaStream_t stream);

/**
 * The same Op with caller-supplied execution hints. Semantics, workspace requirement and output
 * are exactly those of the overload above; hints only steer cache warming.
 */
void sparse_moe(const Tensor& x, const SparseMoeWeights& weights, SparseMoeEpilogue epilogue,
                Tensor& destination, const SparseMoeHints& hints, WorkspaceArena& workspace,
                cudaStream_t stream);

/**
 * Returns the transient capacity required by the NVFP4-bank SparseMoe overload for every T in the
 * inclusive `[min_tokens,max_tokens]` interval under the two banks' activation policies. Invalid
 * policies or intervals throw.
 */
[[nodiscard]] std::size_t sparse_moe_workspace_capacity_bytes(LinearPolicy gate_up_policy,
                                                              LinearPolicy down_policy,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

/**
 * Op: 512-expert top-10 sparse MoE with NVFP4 expert banks and a sigmoid-gated BF16 shared expert.
 *
 * Math / indexing, independently for every token column t with x_t = x[:,t]:
 *   s[e]     = dot(router[e,:], x_t)                      e in [0,512)
 *   p[e]     = exp(s[e]) / sum_j exp(s[j])                FP32-or-better softmax over all 512
 *   S        = the 10 experts of largest p (equivalently largest s); at an exact score tie the
 *              lower expert id ranks first
 *   w[e]     = p[e] / sum_{j in S} p[j]                   top-10 renormalization, e in S
 *   h_e      = silu(G_e x_t) * (U_e x_t)                  G_e = gate_up[e,0:640,:],
 *                                                         U_e = gate_up[e,640:1280,:]
 *   routed   = sum_{e in S} w[e] * (down[e] h_e)
 *   h_s      = silu(shared_gate x_t) * (shared_up x_t)
 *   shared   = sigmoid(dot(shared_expert_gate[0,:], x_t)) * (shared_down h_s)
 *   moe(x_t) = routed + shared
 *   silu(v) = v / (1 + exp(-v)), sigmoid(v) = 1 / (1 + exp(-v)).
 *
 * Logical shapes:
 *   x and destination are contiguous, 16-byte-aligned BF16 [2560,T]; T is every positive token
 *   extent. Weight shapes are those of SparseMoeNvfp4BankWeights.
 *
 * Supported domain:
 *   The two banks are complete NVFP4 expert_block_scale_k16_m128x4_v1 parents with 512 experts,
 *   finite positive divisors, and 16-byte-aligned planes. The five BF16 weights are contiguous.
 *   The epilogue is Store.
 *
 * Numeric:
 *   The oracle decodes every bank weight from its stored code, UE4M3 scale and divisor, uses the
 *   represented BF16 values of x and the BF16 weights, and evaluates the complete formula in FP64,
 *   including the expert selection. The BF16 destination store is the only semantic rounding
 *   boundary. Score, probability, activation and partial-sum precision, the reduction order, and
 *   private activation quantization are implementation choices within the policy: when both banks
 *   permit AllowA4, a route may quantize the expert inputs x_t and h_e to NVFP4 with one dynamic
 *   UE4M3 scale per 16 consecutive values (no stored activation divisor); otherwise every routed
 *   product keeps the represented BF16 activation. Each such arithmetic profile has its own named
 *   criterion. Selection is a semantic step: callers must not rely on a particular outcome where
 *   the 10th and 11th scores differ only by FP32 rounding.
 *
 * Effects:
 *   destination is the only observable mutation: it is overwritten with BF16(moe(x)) and its
 *   incoming value is not read. x, weights and banks are unchanged. x, destination, every weight
 *   plane and the live workspace are pairwise non-overlapping.
 *
 * Workspace:
 *   Caller-owned transient storage sized by the NVFP4-bank sparse_moe_workspace_capacity_bytes()
 *   for the call's T and the banks' policies. It carries no state beyond the call.
 *
 * Execution:
 *   Enqueued on `stream` with no host synchronization and no device allocation; the call is
 *   CUDA-Graph capturable and replayable with graph-stable workspace. Expert data are addressed
 *   only through the bank views; no gather or repack of expert weights occurs.
 */
void sparse_moe(const Tensor& x, const SparseMoeNvfp4BankWeights& weights,
                SparseMoeEpilogue epilogue, Tensor& destination, WorkspaceArena& workspace,
                cudaStream_t stream);

} // namespace ninfer::ops
