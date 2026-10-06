#include "core/weight.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"

#include <cstdint>
#include <initializer_list>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Fp8GdnInputRoute : std::uint8_t {
    A16,
    A8,
};

Fp8GdnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("fp8 gdn_input_proj: T must be positive"); }
    if (policy == LinearPolicy::A16Only) { return Fp8GdnInputRoute::A16; }
    if (!allows_a8(policy)) {
        throw std::invalid_argument("fp8 gdn_input_proj: unsupported policy");
    }
    return tokens >= 17 ? Fp8GdnInputRoute::A8 : Fp8GdnInputRoute::A16;
}

void a16_k5120(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
               cudaStream_t stream) {
    if (x.ne[1] == 1) {
        fp8_gdn_input_decode_launch(x, weight, qkv, z, stream);
    } else {
        fp8_gdn_input_matrix_launch(x, weight, qkv, z, stream);
    }
}

} // namespace

const Fp8GdnInputProfile kFp8GdnInputBf16K5120{
    QType::FP8_E4M3FN_ROW_BF16,           5120,
    a16_k5120,                            fp8_gdn_input_a8_launch,
    fp8_gdn_input_partial_capacity_bytes, fp8_gdn_snapshot_fused_launch,
    fp8_gdn_record_fused_launch};

const Fp8GdnInputProfile* find_fp8_gdn_input_profile(QType qtype, std::int32_t parent_rows,
                                                     std::int32_t input_rows) noexcept {
    if (parent_rows != 16384) { return nullptr; }
    for (const Fp8GdnInputProfile* profile : {&kFp8GdnInputBf16K5120, &kFp8GdnInputFp32K2560}) {
        if (profile->qtype == qtype && profile->input_rows == input_rows) { return profile; }
    }
    return nullptr;
}

std::size_t fp8_gdn_input_workspace_capacity_bytes(const Fp8GdnInputProfile& profile,
                                                   LinearPolicy policy, std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 gdn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    return resolve_route(policy, max_tokens) == Fp8GdnInputRoute::A8
               ? fp8_a8_workspace_capacity_bytes(max_tokens, profile.input_rows,
                                                 profile.a8_partial_bytes(max_tokens))
               : 0;
}

void fp8_gdn_input_a8_dispatch(const Fp8GdnInputProfile& profile, const Tensor& x,
                               const Weight& weight, Tensor& qkv, Tensor& z,
                               WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope                   = workspace.scope();
    const Fp8A8Workspace scratch = allocate_fp8_a8_workspace(
        workspace, x.ne[1], weight.k, profile.a8_partial_bytes(x.ne[1]));
    profile.a8(x, weight, qkv, z, scratch, stream);
}

void fp8_gdn_input_dispatch(const Fp8GdnInputProfile& profile, const Tensor& x,
                            const Weight& weight, Tensor& qkv, Tensor& z, LinearPolicy policy,
                            WorkspaceArena* workspace, cudaStream_t stream) {
    if (resolve_route(policy, x.ne[1]) == Fp8GdnInputRoute::A16) {
        profile.a16(x, weight, qkv, z, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("fp8 A8 gdn_input_proj requires caller workspace");
    }
    fp8_gdn_input_a8_dispatch(profile, x, weight, qkv, z, *workspace, stream);
}

} // namespace ninfer::ops::detail
