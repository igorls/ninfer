#include "models/qwen4_exp/program/mtp_frame.h"
#include <stdexcept>

namespace ninfer::models::qwen4_exp {
MtpFrameLayout plan_mtp_frame(const TextConfig& c, std::int32_t width, std::int32_t batch) {
    if (width < 2 || width > kMtpMaximumWidth || batch < 1 || batch > kMaximumConcurrency) {
        throw std::invalid_argument("MTP frame geometry is outside the supported domain");
    }
    LayoutBuilder b;
    MtpFrameLayout l;
    l.width          = width;
    l.batch_capacity = batch;
    l.ingress        = b.add(sizeof(MtpIngress), 256, "MTP ingress");
    const auto add   = [&](DType dtype, std::initializer_list<std::int32_t> shape) {
        return b.add_tensor(dtype, shape, 256, "MTP frame");
    };
    const auto h        = static_cast<std::int32_t>(c.stream_width());
    const auto v        = static_cast<std::int32_t>(c.vocab_size);
    l.hidden            = add(DType::BF16, {h, width, batch});
    l.logits            = add(DType::BF16, {v, width, batch});
    l.argmax            = add(DType::I32, {width, batch});
    l.drafts            = add(DType::I32, {width - 1, batch});
    l.licensed          = add(DType::I32, {width, batch});
    l.counts            = add(DType::I32, {batch});
    l.accepted          = add(DType::I32, {batch});
    l.previous          = add(DType::BF16, {h, width, batch});
    l.teacher_output    = add(DType::BF16, {h, width, batch});
    l.teacher_positions = add(DType::I32, {width, batch});
    l.teacher_rope      = add(DType::I32, {width * batch, 3});
    l.ar_hidden         = add(DType::BF16, {h, batch});
    l.next_hidden       = add(DType::BF16, {h, batch});
    l.embedding       = add(DType::BF16, {static_cast<std::int32_t>(c.hidden_size), width * batch});
    l.proposal_logits = add(DType::BF16, {v, batch});
    l.proposal_ids    = add(DType::I32, {batch});
    l.proposal_positions = add(DType::I32, {batch});
    l.proposal_rope      = add(DType::I32, {batch, 3});
    l.proposal_sources   = add(DType::I32, {batch});
    l.records            = plan_gdn_replay_records(
        b, {.layers          = static_cast<std::int32_t>(c.linear_attention_layers),
                       .record_capacity = batch,
                       .width           = width,
                       .conv_channels   = static_cast<std::int32_t>(c.gdn.conv_channels()),
                       .qk_heads        = static_cast<std::int32_t>(c.gdn.linear_num_key_heads),
                       .value_heads     = static_cast<std::int32_t>(c.gdn.linear_num_value_heads),
                       .key_dim         = static_cast<std::int32_t>(c.gdn.linear_key_head_dim),
                       .value_dim       = static_cast<std::int32_t>(c.gdn.linear_value_head_dim)});
    l.bytes = b.finish(256, "MTP frame");
    return l;
}

#define NINFER_MTP_TENSORS(F)                                                                      \
    F(hidden)                                                                                      \
    F(logits) F(argmax) F(drafts) F(licensed) F(counts) F(accepted) F(previous) F(teacher_output)  \
        F(teacher_positions) F(teacher_rope) F(ar_hidden) F(next_hidden) F(embedding)              \
            F(proposal_logits) F(proposal_ids) F(proposal_positions) F(proposal_rope)              \
                F(proposal_sources)

MtpFrame::MtpFrame(DeviceSpan backing, const MtpFrameLayout& l)
    : ingress(l.ingress.bind(backing)), records(backing, l.records), width(l.width),
      batch_capacity(l.batch_capacity) {
#define BIND(name) name = l.name.bind(backing);
    NINFER_MTP_TENSORS(BIND)
#undef BIND
}

MtpFrame MtpFrame::batch(std::int32_t count) const {
    if (count < 1 || count > batch_capacity) { throw std::invalid_argument("MTP active batch"); }
    MtpFrame f = *this;
    for (Tensor* t : {&f.hidden, &f.logits, &f.previous, &f.teacher_output}) {
        *t = t->slice(2, 0, count);
    }
    for (Tensor* t : {&f.argmax, &f.drafts, &f.licensed, &f.teacher_positions, &f.ar_hidden,
                      &f.next_hidden, &f.proposal_logits}) {
        *t = t->slice(1, 0, count);
    }
    for (Tensor* t :
         {&f.counts, &f.accepted, &f.proposal_ids, &f.proposal_positions, &f.proposal_sources}) {
        *t = t->slice(0, 0, count);
    }
    // Planar controls are repacked for the current B, with a fixed maximum backing.
    f.teacher_rope  = Tensor(teacher_rope.data, DType::I32, {width * count, 3});
    f.proposal_rope = Tensor(proposal_rope.data, DType::I32, {count, 3});
    f.embedding     = Tensor(embedding.data, DType::BF16, {embedding.ne[0], width * count});
    return f;
}

Tensor MtpFrame::i32(std::size_t offset, std::int32_t count) const {
    return Tensor(static_cast<std::byte*>(ingress.data) + offset, DType::I32, {count});
}

const ops::SamplingConfig* MtpFrame::sampling() const {
    return reinterpret_cast<const ops::SamplingConfig*>(
        static_cast<const std::byte*>(ingress.data) + offsetof(MtpIngress, sampling));
}

#undef NINFER_MTP_TENSORS
} // namespace ninfer::models::qwen4_exp
