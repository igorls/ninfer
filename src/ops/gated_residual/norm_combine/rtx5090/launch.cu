#include "ninfer/ops/gated_residual.h"
#include "ops/gated_residual/kernel/norm.cuh"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail::hc {
namespace {
constexpr int kDisjointMaterializeStart = 1841;
constexpr int kDisjointMaterializeEnd   = 3072;
constexpr int kInPlaceMaterializeStart  = 1824;
constexpr int kInPlaceMaterializeEnd    = 3680;

// One reduction tree preserves the BF16 normalization boundary across prefill windows.
template <Form Kind>
void launch_norm_rows(Operands o, CudaExecutionView e) {
    const std::int64_t rows = std::int64_t(o.columns) * kStreams;
    for (std::int64_t first = 0; first < rows;) {
        const auto count = std::min(rows - first, std::int64_t(std::numeric_limits<int>::max()));
        norm_kernel<Kind, 256><<<unsigned(count), 256, 0, e.stream()>>>(o, first);
        CUDA_CHECK(cudaGetLastError());
        first += count;
    }
}

template <Form Kind>
void launch_combine(Operands o, CudaExecutionView e) {
    constexpr int block                = 256;
    const std::int64_t vectors         = std::int64_t(o.columns) * kStreams * (kWidth / 8);
    constexpr std::int64_t max_vectors = std::int64_t(std::numeric_limits<int>::max()) * block;
    for (std::int64_t first = 0; first < vectors;) {
        const auto count = std::min(vectors - first, max_vectors);
        combine_vector_kernel<Kind, block>
            <<<unsigned((count + block - 1) / block), block, 0, e.stream()>>>(o, first, count);
        CUDA_CHECK(cudaGetLastError());
        first += count;
    }
}

template <Form Kind>
void launch_materialized_norm(Operands o, CudaExecutionView execution) {
    launch_combine<Kind == Form::AddNorm ? Form::AddNorm : Form::Combine>(o, execution);
    o.input = o.residual;
    // The freshly produced residual uses the paired norm, without input eviction.
    launch_norm_rows<Form::GroupedNorm>(o, execution);
}

using Launch = void (*)(Operands, CudaExecutionView);

template <Form Kind>
Launch select_combine_norm(Operands o) {
    const bool in_place = o.input == o.residual;
    const bool materialize =
        in_place ? o.columns >= kInPlaceMaterializeStart && o.columns <= kInPlaceMaterializeEnd
                 : o.columns >= kDisjointMaterializeStart && o.columns <= kDisjointMaterializeEnd;
    return materialize ? launch_materialized_norm<Kind> : launch_norm_rows<Kind>;
}

Launch select_implementation(Form form, Operands o) {
    switch (form) {
    case Form::ExpandNorm:
        return launch_norm_rows<Form::ExpandNorm>;
    case Form::GroupedNorm:
        return launch_norm_rows<Form::GroupedNorm>;
    case Form::Combine:
        return launch_combine<Form::Combine>;
    case Form::CombineNorm:
        return select_combine_norm<Form::CombineNorm>(o);
    case Form::AddNorm:
        return select_combine_norm<Form::AddNorm>(o);
    }
    throw std::logic_error("gated residual: invalid form");
}

} // namespace

void launch_norm_combine(Form form, Operands operands, CudaExecutionView execution) {
    select_implementation(form, operands)(operands, execution);
}
} // namespace ninfer::ops::detail::hc
