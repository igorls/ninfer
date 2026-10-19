#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

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
 * computation may prefetch its first bytes into L2. The span is caller-owned
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
 * Sparse mixture of SwiGLU experts with a shared expert and residual addition.
 *
 * For each contiguous BF16 x[:,t] and destination[:,t], both [2048,T], T>0:
 *   scores = router_shared_gate * x
 *   E = top8(scores[0:256]), breaking ties by lower expert id
 *   alpha = softmax(scores[E]); beta = sigmoid(scores[256])
 *   h_e = silu(gate_e * x) * (up_e * x)
 *   destination' = BF16(destination + sum_e_in_E alpha_e * (down_e * h_e)
 *                                   + beta * (shared_down * h_shared))
 *
 * Stored weights are interpreted through their exact numeric formats. Internal staging,
 * accumulation and scale placement are private; they add no semantic rounding boundaries.
 * destination is the only observable mutation. All inputs remain unchanged.
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
 * Execution is enqueued on execution.stream without host synchronization. The positive physical
 * SM count in execution selects the persistent prefill grid. Workspace is caller-owned,
 * graph-stable transient storage and carries no state beyond the call; its capacity does not
 * depend on the grid.
 */
void sparse_moe(const Tensor& x, const SparseMoeWeights& weights, SparseMoeEpilogue epilogue,
                Tensor& destination, WorkspaceArena& workspace, DeviceExecutionView execution);

/**
 * The same Op with caller-supplied execution hints. Semantics, workspace requirement and output
 * are exactly those of the overload above; hints only steer cache warming.
 */
void sparse_moe(const Tensor& x, const SparseMoeWeights& weights, SparseMoeEpilogue epilogue,
                Tensor& destination, const SparseMoeHints& hints, WorkspaceArena& workspace,
                DeviceExecutionView execution);

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
 *   destination is overwritten with BF16(moe(x)) and its incoming value is not read. When
 *   selected_experts is non-null it is overwritten with S of every column as contiguous I32
 *   [10,T]: column t holds its ten expert ids in rank order (largest score first, the lower id
 *   first at an exact tie), the same selection that produced destination. These are the only
 *   observable mutations. x, weights and banks are unchanged. x, destination, selected_experts,
 *   every weight plane and the live workspace are pairwise non-overlapping.
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
                cudaStream_t stream, Tensor* selected_experts = nullptr);

} // namespace ninfer::ops
