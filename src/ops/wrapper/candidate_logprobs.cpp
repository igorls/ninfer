// ninfer::ops - candidate_logprobs wrapper: public contract validation and launcher dispatch.
#include "ninfer/ops/candidate_logprobs.h"

#include "ops/launcher/candidate_logprobs.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

[[noreturn]] void fail(const std::string& what) {
    throw std::invalid_argument("candidate_logprobs: " + what);
}

void require_accessible(const Tensor& tensor, std::size_t alignment, const char* label) {
    if (!tensor.is_contiguous()) { fail(std::string(label) + " must be contiguous"); }
    if (tensor.data == nullptr) { fail(std::string(label) + " data must be non-null"); }
    if ((reinterpret_cast<std::uintptr_t>(tensor.data) & (alignment - 1)) != 0) {
        fail(std::string(label) + " data is not naturally aligned");
    }
}

void require_shape(const Tensor& tensor, std::int32_t ne0, std::int32_t ne1, std::int32_t ne2,
                   const char* label) {
    if (tensor.ne[0] != ne0 || tensor.ne[1] != ne1 || tensor.ne[2] != ne2 || tensor.ne[3] != 1) {
        fail(std::string(label) + " has the wrong shape");
    }
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    if (lhs_begin <= rhs_begin) { return rhs_begin - lhs_begin < lhs.bytes(); }
    return lhs_begin - rhs_begin < rhs.bytes();
}

} // namespace

void candidate_logprobs(const Tensor& logits, std::int32_t valid_rows, const Tensor& sampled_ids,
                        const Tensor* candidate_ids, const Tensor* allowed, Tensor& sampled_out,
                        Tensor* candidates_out, cudaStream_t stream) {
    if (logits.dtype != DType::BF16) { fail("logits must be BF16"); }
    if (logits.ne[0] <= 0 || logits.ne[1] <= 0 || logits.ne[2] != 1 || logits.ne[3] != 1) {
        fail("logits must be rank-2 with positive dimensions");
    }
    const std::int32_t columns = logits.ne[1];
    if (valid_rows <= 0 || valid_rows > logits.ne[0]) {
        fail("valid_rows must be in [1, physical_rows]");
    }
    if (sampled_ids.dtype != DType::I32) { fail("sampled_ids must be I32"); }
    require_shape(sampled_ids, columns, 1, 1, "sampled_ids");
    if (sampled_out.dtype != DType::FP32) { fail("sampled_out must be FP32"); }
    require_shape(sampled_out, columns, 2, 1, "sampled_out");
    if ((candidate_ids == nullptr) != (candidates_out == nullptr)) {
        fail("candidate_ids and candidates_out are given together");
    }
    if (candidate_ids != nullptr) {
        if (candidate_ids->dtype != DType::I32) { fail("candidate_ids must be I32"); }
        if (candidate_ids->ne[0] <= 0) { fail("candidate_ids must be nonempty"); }
        require_shape(*candidate_ids, candidate_ids->ne[0], 1, 1, "candidate_ids");
        if (candidates_out->dtype != DType::FP32) { fail("candidates_out must be FP32"); }
        require_shape(*candidates_out, columns, candidate_ids->ne[0], 2, "candidates_out");
    }
    if (allowed != nullptr) {
        if (allowed->dtype != DType::I32) { fail("allowed must be I32"); }
        if (allowed->ne[0] <= 0 || static_cast<std::int64_t>(allowed->ne[0]) * 32 < valid_rows) {
            fail("allowed must cover every valid row");
        }
        require_shape(*allowed, allowed->ne[0], 1, 1, "allowed");
    }

    (void)logits.bytes();
    require_accessible(logits, alignof(std::uint16_t), "logits");
    require_accessible(sampled_ids, alignof(std::int32_t), "sampled_ids");
    require_accessible(sampled_out, alignof(float), "sampled_out");
    if (candidate_ids != nullptr) {
        require_accessible(*candidate_ids, alignof(std::int32_t), "candidate_ids");
        require_accessible(*candidates_out, alignof(float), "candidates_out");
    }
    if (allowed != nullptr) { require_accessible(*allowed, alignof(std::int32_t), "allowed"); }

    const Tensor* inputs[] = {&logits, &sampled_ids, candidate_ids, allowed};
    const Tensor* outputs[] = {&sampled_out, candidates_out};
    for (const Tensor* output : outputs) {
        if (output == nullptr) { continue; }
        for (const Tensor* input : inputs) {
            if (input != nullptr && overlaps(*output, *input)) {
                fail("outputs must not overlap inputs");
            }
        }
    }
    if (candidates_out != nullptr && overlaps(sampled_out, *candidates_out)) {
        fail("outputs must not overlap each other");
    }

    detail::candidate_logprobs_launch(logits, valid_rows, sampled_ids, candidate_ids, allowed,
                                      sampled_out, candidates_out, stream);
}

} // namespace ninfer::ops
