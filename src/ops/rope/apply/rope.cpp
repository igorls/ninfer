#include "ninfer/ops/rope.h"
#include "ops/rope/apply/dispatch.h"
#include "ops/common/host/operands.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kTextHeadDim = 256;

int position_axes(const Tensor& positions, std::int32_t tokens) {
    if (positions.ne[0] != tokens || positions.ne[2] != 1 || positions.ne[3] != 1 ||
        (positions.ne[1] != 1 && positions.ne[1] != 2 && positions.ne[1] != 3)) {
        throw std::invalid_argument("rope: positions must have shape [T], [T,2], or [T,3]");
    }
    return positions.ne[1];
}

void require_tensor_layout(const Tensor& tensor, const char* label, std::int32_t head_dim,
                           std::int32_t heads, std::int32_t tokens) {
    if (heads <= 0) {
        throw std::invalid_argument(std::string("rope: ") + label + " must have positive heads");
    }
    if (tensor.ne[0] != head_dim || tensor.ne[1] != heads || tensor.ne[2] != tokens ||
        tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("rope: invalid ") + label + " shape");
    }
    constexpr std::int64_t elem = 2;
    if (tensor.nb[0] != elem || tensor.nb[1] != elem * head_dim ||
        tensor.nb[2] < elem * static_cast<std::int64_t>(head_dim) * heads ||
        (tensor.nb[2] % elem) != 0) {
        throw std::invalid_argument(std::string("rope: invalid ") + label + " strides");
    }
}

void require_common(const Tensor& positions, int rotary_dim, float theta) {
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument("rope: positions must be I32");
    }
    if (!(theta > 0.0f) || !std::isfinite(theta)) {
        throw std::invalid_argument("rope: theta must be positive and finite");
    }
    if (rotary_dim <= 0 || (rotary_dim & 1) != 0) {
        throw std::invalid_argument("rope: rotary_dim must be positive and even");
    }
}

void require_positions_storage(const Tensor& positions) {
    if (!positions.is_contiguous()) {
        throw std::invalid_argument("rope: positions must be contiguous");
    }
    if (positions.data == nullptr) {
        throw std::invalid_argument("rope: positions data must be non-null");
    }
}

void require_rotary_geometry(int axes, int rotary_dim, std::int32_t head_dim) {
    if (axes == 2) {
        if ((head_dim != 64 && head_dim != 72) || rotary_dim != head_dim) {
            throw std::invalid_argument(
                "rope: 2-D Vision mode requires head_dim=rotary_dim=64 or 72");
        }
        return;
    }
    if (axes == 1 && head_dim == 128 && rotary_dim == 128) { return; }
    if (head_dim != kTextHeadDim || rotary_dim > kTextHeadDim) {
        throw std::invalid_argument("rope: Text mode requires D256 or one-dimensional D128/R128");
    }
    if (axes == 3 && rotary_dim != 64) {
        throw std::invalid_argument("rope: 3-D Text MRoPE requires rotary_dim=64");
    }
}

bool finite_positive(float value) { return std::isfinite(value) && value > 0.0F; }

// Resolves the documented YaRN frequencies in FP64 from the public parameters.
detail::YarnR64Table yarn_table(int rotary_dim, float theta, const RopeYarn& yarn) {
    if (!finite_positive(yarn.factor) || !(yarn.factor > 1.0F) ||
        yarn.original_max_position_embeddings <= 0 || !finite_positive(yarn.beta_fast) ||
        !finite_positive(yarn.beta_slow) || !(yarn.beta_slow < yarn.beta_fast) ||
        !finite_positive(yarn.attention_factor)) {
        throw std::invalid_argument("rope: invalid YaRN parameters");
    }
    constexpr double kPi        = 3.14159265358979323846264338327950288;
    const double dimension      = rotary_dim;
    const double log_theta      = std::log(static_cast<double>(theta));
    const double original       = yarn.original_max_position_embeddings;
    const auto correction       = [&](double rotations) {
        return dimension * std::log(original / (rotations * 2.0 * kPi)) / (2.0 * log_theta);
    };
    const double low  = std::max(std::floor(correction(yarn.beta_fast)), 0.0);
    const double high = std::min(std::ceil(correction(yarn.beta_slow)), dimension - 1.0);
    const double span = high == low ? 0.001 : high - low;
    detail::YarnR64Table table{};
    for (int pair = 0; pair < rotary_dim / 2; ++pair) {
        const double base = std::pow(static_cast<double>(theta), -2.0 * pair / dimension);
        const double ramp = std::clamp((pair - low) / span, 0.0, 1.0);
        table.frequency[pair] = (1.0 - ramp) * base + ramp * base / yarn.factor;
    }
    table.magnitude = yarn.attention_factor;
    return table;
}

void require_yarn_geometry(int axes, int rotary_dim, std::int32_t head_dim) {
    if ((axes != 1 && axes != 3) || head_dim != kTextHeadDim || rotary_dim != 64) {
        throw std::invalid_argument(
            "rope: YaRN requires Text 1-D or MRoPE with head_dim=256 and rotary_dim=64");
    }
}

} // namespace

void rope(const Tensor& positions, int rotary_dim, float theta, const RopeYarn& yarn, Tensor& q,
          Tensor& k, CudaExecutionView execution) {
    require_common(positions, rotary_dim, theta);
    if (q.dtype != DType::BF16 || k.dtype != DType::BF16) {
        throw std::invalid_argument("rope: q/k must be BF16");
    }
    (void)common::operand::numel(positions, "rope");
    const std::int64_t q_numel = common::operand::numel(q, "rope");
    (void)common::operand::numel(k, "rope");
    const std::int32_t tokens   = q.ne[2];
    const int axes              = position_axes(positions, tokens);
    const std::int32_t head_dim = q.ne[0];
    require_yarn_geometry(axes, rotary_dim, head_dim);
    require_tensor_layout(q, "q", head_dim, q.ne[1], tokens);
    require_tensor_layout(k, "k", head_dim, k.ne[1], tokens);
    const auto table = yarn_table(rotary_dim, theta, yarn);
    if (q_numel == 0) { return; }
    common::operand::device(positions.device, execution.device(), "rope", "positions");
    require_positions_storage(positions);
    common::operand::device(q.device, execution.device(), "rope", "query");
    common::operand::device(k.device, execution.device(), "rope", "key");
    if (q.data == nullptr || k.data == nullptr) {
        throw std::invalid_argument("rope: q/k data must be non-null");
    }
    detail::rope_yarn_dispatch(positions, table, q, &k, execution);
}

void rope(const Tensor& positions, int rotary_dim, float theta, const RopeYarn& yarn, Tensor& x,
          CudaExecutionView execution) {
    require_common(positions, rotary_dim, theta);
    if (x.dtype != DType::BF16) { throw std::invalid_argument("rope: tensor must be BF16"); }
    (void)common::operand::numel(positions, "rope");
    const std::int64_t x_numel  = common::operand::numel(x, "rope");
    const std::int32_t tokens   = x.ne[2];
    const int axes              = position_axes(positions, tokens);
    const std::int32_t head_dim = x.ne[0];
    require_yarn_geometry(axes, rotary_dim, head_dim);
    require_tensor_layout(x, "tensor", head_dim, x.ne[1], tokens);
    const auto table = yarn_table(rotary_dim, theta, yarn);
    if (x_numel == 0) { return; }
    common::operand::device(positions.device, execution.device(), "rope", "positions");
    require_positions_storage(positions);
    common::operand::device(x.device, execution.device(), "rope", "input");
    if (x.data == nullptr) { throw std::invalid_argument("rope: tensor data must be non-null"); }
    detail::rope_yarn_dispatch(positions, table, x, nullptr, execution);
}

void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
          CudaExecutionView execution) {
    require_common(positions, rotary_dim, theta);
    if (q.dtype != DType::BF16 || k.dtype != DType::BF16) {
        throw std::invalid_argument("rope: q/k must be BF16");
    }
    (void)common::operand::numel(positions, "rope");
    const std::int64_t q_numel = common::operand::numel(q, "rope");
    (void)common::operand::numel(k, "rope");
    const std::int32_t tokens   = q.ne[2];
    const int axes              = position_axes(positions, tokens);
    const std::int32_t head_dim = q.ne[0];
    const std::int32_t q_heads  = q.ne[1];
    const std::int32_t k_heads  = k.ne[1];
    require_rotary_geometry(axes, rotary_dim, head_dim);
    require_tensor_layout(q, "q", head_dim, q_heads, tokens);
    require_tensor_layout(k, "k", head_dim, k_heads, tokens);
    if (q_numel == 0) { return; }
    common::operand::device(positions.device, execution.device(), "rope", "positions");
    require_positions_storage(positions);
    common::operand::device(q.device, execution.device(), "rope", "query");
    common::operand::device(k.device, execution.device(), "rope", "key");
    if (q.data == nullptr || k.data == nullptr) {
        throw std::invalid_argument("rope: q/k data must be non-null");
    }
    detail::rope_dispatch(positions, rotary_dim, theta, q, k, execution);
}

void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
          CudaExecutionView execution) {
    require_common(positions, rotary_dim, theta);
    if (x.dtype != DType::BF16) { throw std::invalid_argument("rope: tensor must be BF16"); }
    (void)common::operand::numel(positions, "rope");
    const std::int64_t x_numel  = common::operand::numel(x, "rope");
    const std::int32_t tokens   = x.ne[2];
    const int axes              = position_axes(positions, tokens);
    const std::int32_t head_dim = x.ne[0];
    const std::int32_t heads    = x.ne[1];
    require_rotary_geometry(axes, rotary_dim, head_dim);
    require_tensor_layout(x, "tensor", head_dim, heads, tokens);
    if (x_numel == 0) { return; }
    common::operand::device(positions.device, execution.device(), "rope", "positions");
    require_positions_storage(positions);
    common::operand::device(x.device, execution.device(), "rope", "input");
    if (x.data == nullptr) { throw std::invalid_argument("rope: tensor data must be non-null"); }
    detail::rope_dispatch(positions, rotary_dim, theta, x, execution);
}

} // namespace ninfer::ops
