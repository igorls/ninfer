#include "models/qwen4_exp/program/graphs.h"
#include "models/qwen4_exp/program/internal.h"
#include "core/startup.h"
#include "core/transfer.h"
#include "core/nvtx.h"

#include <algorithm>
#include <thread>

namespace ninfer::models::qwen4_exp::detail {
namespace {

constexpr std::array kContextBounds{128U, 512U, 2048U, 8192U, 32768U, 131072U};

std::vector<std::uint32_t> context_bounds(std::uint32_t capacity) {
    std::vector<std::uint32_t> out;
    for (const auto bound : kContextBounds) {
        if (bound >= capacity) break;
        out.push_back(bound);
    }
    out.push_back(capacity);
    return out;
}

void write_i32(const Tensor& target, std::span<const std::int32_t> values, CudaExecutionView cuda) {
    copy_bytes({target.data, target.bytes(), cuda.device()},
               {values.data(), values.size_bytes(), Device::cpu()}, cuda);
}

// Preparation precedes Session binding. One physical page per row is repeated across the
// context envelope; mappings and images are private to this preparation lifetime.
class CaptureInputs {
public:
    explicit CaptureInputs(RankProgramImpl& owner) : p_(owner) {}

    void prepare() {
        auto& r = p_.resources;
        if (r.states->storage().slot_count() < static_cast<std::int32_t>(p_.shape.max_concurrency))
            throw std::logic_error("Graph preparation requires one StateImage per row");
        for (std::uint32_t b = 0; b < p_.shape.max_concurrency; ++b) {
            slots_.push_back(static_cast<std::int32_t>(b));
            r.states->storage().clear(slots_.back(), p_.cuda);
        }
        for (std::uint32_t pool = 0; pool < r.pool_count(); ++pool) {
            auto& cache  = r.pool(pool);
            auto& pages  = cache.pages();
            auto& tables = cache.tables();
            if (pages.capacity_pages() < p_.shape.max_concurrency ||
                tables.row_count() < static_cast<std::int32_t>(p_.shape.max_concurrency))
                throw std::logic_error("Graph preparation requires one KV page and table per row");
            const auto capacity  = tables.logical_page_capacity();
            const auto row_bytes = static_cast<std::size_t>(capacity) * sizeof(std::int32_t);
            mappings_[pool]      = PinnedHostBuffer(row_bytes * p_.shape.max_concurrency);
            for (std::uint32_t b = 0; b < p_.shape.max_concurrency; ++b) {
                const auto page = static_cast<std::int32_t>(b);
                pages.zero_pages({&page, 1}, p_.cuda);
                const auto staging = mappings_[pool].span().subspan(b * row_bytes, row_bytes);
                auto* indices      = static_cast<std::int32_t*>(staging.data);
                std::fill_n(indices, capacity, page);
                tables.publish(page, 0, {indices, capacity}, staging, p_.cuda);
            }
        }
        for (std::uint32_t slot = 0; slot < 2; ++slot) {
            auto embedding = r.ple_upload->device_slot(slot, r.plan.ple.max_positions, 1);
            CUDA_CHECK(cudaMemsetAsync(embedding.data, 0, embedding.bytes(), p_.cuda.stream()));
        }
        const auto& buffers = r.plan.persistent.buffers;
        if (buffers.mtp) {
            for (const auto* region :
                 {&buffers.mtp->base_conditioning, &buffers.mtp->hidden, &buffers.mtp->tail_keys,
                  &buffers.mtp->tail_coordinates, &buffers.mtp->drafts, &buffers.mtp->extents,
                  &buffers.mtp->anchors, &buffers.mtp->frontiers}) {
                const auto value = r.tensor(*region);
                CUDA_CHECK(cudaMemsetAsync(value.data, 0, value.bytes(), p_.cuda.stream()));
            }
        }
        std::vector<ops::SamplingConfig> configs(p_.shape.max_concurrency);
        for (auto& config : configs) config.temperature = 0;
        const auto target = r.tensor(*buffers.sampling);
        copy_bytes({target.data, target.bytes(), p_.cuda.device()},
                   {configs.data(), configs.size() * sizeof(configs[0]), Device::cpu()}, p_.cuda);
    }

    ~CaptureInputs() {
        // prepare() reports execution failures; unwinding still drains the staging readers.
        (void)cudaStreamSynchronize(p_.cuda.stream());
    }

    void batch(execution::BatchView call, std::uint32_t visible) {
        const auto t = call.tokens, b = call.batch;
        const auto valid = std::min<std::uint32_t>(t, visible);
        const auto base  = visible - valid;
        std::vector<std::int32_t> ids(t * b, p_.parameters.profile->config.text.ple.eos);
        std::vector<std::int32_t> positions(t * b), rope(t * b * 3), rows(b), bases(b, base);
        for (int row = 0; row < b; ++row) {
            rows[row] = row;
            for (int j = 0; j < t; ++j) {
                const auto position    = base + std::min<std::uint32_t>(j, valid - 1);
                positions[j + t * row] = position;
                for (int axis = 0; axis < 3; ++axis) rope[j + t * (axis + 3 * row)] = position;
            }
        }
        write_i32(call.ids, ids, p_.cuda);
        write_i32(call.positions, positions, p_.cuda);
        write_i32(call.rope_positions, rope, p_.cuda);
        write_i32(call.source_slots, std::span(slots_).first(b), p_.cuda);
        write_i32(call.destination_slots, std::span(slots_).first(b), p_.cuda);
        write_i32(call.table_rows, rows, p_.cuda);
        write_i32(call.base_positions, bases, p_.cuda);
        std::vector<std::int32_t> counts(b, valid), offsets(b, 0);
        write_i32(call.valid_columns, counts, p_.cuda);
        write_i32(call.record_offsets, offsets, p_.cuda);
    }

private:
    RankProgramImpl& p_;
    std::vector<std::int32_t> slots_;
    std::array<PinnedHostBuffer, 2> mappings_;
};
} // namespace

std::uint32_t context_envelope(std::uint32_t visible, std::uint32_t capacity) {
    if (!visible || visible > capacity)
        throw std::logic_error("Qwen4Exp visible context exceeds its capacity");
    for (const auto bound : kContextBounds)
        if (visible <= bound) return std::min(bound, capacity);
    return capacity;
}

Graphs::Graphs(RankProgramImpl& owner)
    : owner_(owner), bounds_(context_bounds(owner.shape.max_context)) {}

TextGraph& Graphs::main(Batch& graph, MainGraph kind) {
    switch (kind) {
    case MainGraph::Control:
        return graph.control;
    case MainGraph::Ordinary:
        return graph.ordinary;
    case MainGraph::Verify:
        return graph.verify;
    }
    throw std::logic_error("invalid Main graph kind");
}

void Graphs::prepare_main(Batch& graph, std::uint32_t batch, MainGraph kind) {
    auto& p             = owner_;
    auto& r             = p.resources;
    const auto& buffers = r.plan.persistent.buffers;
    const auto verify   = kind == MainGraph::Verify;
    const auto tokens   = verify ? p.shape.draft_tokens + 1 : 1;
    const auto call     = r.main_batch(tokens, batch, bounds_.back());
    const auto& c       = p.parameters.profile->config.text;
    const auto t = static_cast<std::int32_t>(tokens), b = static_cast<std::int32_t>(batch);
    execution::TextOutputs output{r.packed(buffers.pending_sample, {dimension(c.hidden), t, b}),
                                  {}};
    if (buffers.pending_conditioning)
        output.conditioning =
            r.packed(*buffers.pending_conditioning, {dimension(c.residual_width()), t, b});
    const execution::TextExecution execution{r.workspace,
                                             *r.states,
                                             *r.main,
                                             r.experts.get(),
                                             r.tensor(buffers.main_qsa.keys),
                                             r.tensor(buffers.main_qsa.coordinates),
                                             r.tensor(buffers.ple_records),
                                             p.route_trace ? &*p.route_trace : nullptr};
    execution::StateUpdate update = execution::StepUpdate{};
    if (verify) update.emplace<execution::VerifyRecords>(*r.gdn_records);
    const std::array ple{r.ple_upload->binding(0, tokens, batch),
                         r.ple_upload->binding(1, tokens, batch)};
    auto& selected = main(graph, kind);
    {
        // Warm the actual bodies before capture. No disk request is needed for these bindings.
        execution::TextRun run(p.parameters, execution, call, update, p.cuda);
        run.stem();
        for (std::uint32_t layer = 0; layer < c.ple.layer; ++layer)
            while (!run.advance_layer(layer)) std::this_thread::yield();
        run.consume_ple(ple[0].values);
        for (std::uint32_t layer = c.ple.layer; layer < p.parameters.text.layers.size(); ++layer)
            while (!run.advance_layer(layer)) std::this_thread::yield();
        run.finish(output);
        p.device.synchronize();
        selected.prepare(run, p.parameters, r.experts.get(), bounds_,
                         {ple[0].values, ple[1].values}, output,
                         tokens * batch, r.workspace, p.device);
    }
    selected.select(bounds_.back());
    execution::TextRun run(p.parameters, execution, call, update, p.cuda);
    for (std::uint32_t slot : {1U, 0U}) {
        selected.begin_prefix(p.cuda);
        while (!selected.advance_prefix(p.cuda)) std::this_thread::yield();
        selected.consume_ple(slot, p.cuda);
        while (!selected.advance_suffix(p.cuda)) std::this_thread::yield();
    }
    p.device.synchronize();
}

void Graphs::prepare_drafts(Batch& graph, std::uint32_t batch) {
    auto& p = owner_;
    graph.drafts.resize(p.shape.draft_tokens);
    for (std::uint32_t step = 0; step < p.shape.draft_tokens; ++step) {
        auto& call = graph.drafts[step];
        p.draft_forward(step, batch, bounds_.back());
        p.device.synchronize();
        for (const auto bound : bounds_) {
            p.resources.workspace.reset_peak();
            call.definitions.emplace_back();
            call.definitions.back().capture(p.cuda.stream(),
                [&] { p.draft_forward(step, batch, bound); });
            call.workspace_bytes.push_back(p.resources.workspace.peak_used());
        }
        call.executable.instantiate(call.definitions.front());
        for (std::size_t i = 0; i < call.definitions.size(); ++i) {
            call.executable.update(call.definitions[i]);
            call.executable.upload(p.cuda.stream());
            p.device.synchronize();
        }
        call.installed = call.definitions.size() - 1;
    }
}

void Graphs::prepare() {
    auto& p = owner_;
    auto& r = p.resources;
    std::optional<moe::Schedule::Preparation> preparation;
    if (r.experts) preparation.emplace(*r.experts);
    CaptureInputs inputs(p);
    inputs.prepare();
    batches_.resize(p.shape.max_concurrency);
    for (std::uint32_t batch = 1; batch <= p.shape.max_concurrency; ++batch) {
        auto& graph = batches_[batch - 1];
        inputs.batch(r.main_batch(1, batch, bounds_.back()), bounds_.back());
        prepare_main(graph, batch, MainGraph::Control);
        prepare_main(graph, batch, MainGraph::Ordinary);
        p.project_samples(batch, false);
        p.sample_forward(batch, false);
        p.device.synchronize();
        r.workspace.reset_peak();
        graph.sample.prepare(p.cuda, [&] {
            p.project_samples(batch, false);
            p.sample_forward(batch, false);
        });
        graph.sample_workspace = r.workspace.peak_used();
        if (p.parameters.mtp) {
            const auto& frame = *r.plan.persistent.buffers.mtp;
            for (std::uint32_t step = 0; step < p.shape.draft_tokens; ++step)
                inputs.batch(r.mtp_batch(frame.draft_steps[step], 1, batch, bounds_.back()),
                             bounds_.back());
            prepare_drafts(graph, batch);
            inputs.batch(r.main_batch(p.shape.draft_tokens + 1, batch, bounds_.back()),
                         bounds_.back());
            prepare_main(graph, batch, MainGraph::Verify);
            p.project_samples(batch, true);
            p.device.synchronize();
            r.workspace.reset_peak();
            graph.head.prepare(p.cuda, [&] { p.project_samples(batch, true); });
            graph.head_workspace = r.workspace.peak_used();
            p.accept_forward(batch);
            p.device.synchronize();
            r.workspace.reset_peak();
            graph.accept.prepare(p.cuda, [&] { p.accept_forward(batch); });
            graph.accept_workspace = r.workspace.peak_used();
        }
    }
    if (r.experts) r.experts->restore_cache_after_preparation();
    p.device.synchronize();
    r.workspace.reset_peak();
}

TextGraph& Graphs::main(std::uint32_t batch, MainGraph kind, std::uint32_t visible) {
    auto& group = batches_.at(batch - 1);
    auto& graph = main(group, kind);
    graph.select(visible);
    workspace_peak_ = std::max(workspace_peak_, graph.workspace_bytes());
    return graph;
}

void Graphs::draft(std::uint32_t batch, std::span<const execution::BatchView> calls,
                   ple::Ticket ticket, std::span<const std::uint32_t> extents) {
    auto& group = batches_.at(batch - 1);
    for (std::size_t step = 0; step < calls.size(); ++step) {
        auto& call = group.drafts.at(step);
        const auto found = std::lower_bound(bounds_.begin(), bounds_.end(), calls[step].max_visible_tokens);
        if (found == bounds_.end()) throw std::logic_error("MTP graph context exceeds capacity");
        const auto selected = static_cast<std::size_t>(found - bounds_.begin());
        workspace_peak_ = std::max(workspace_peak_, call.workspace_bytes[selected]);
        if (call.installed != selected) {
            call.executable.update(call.definitions[selected]);
            call.installed = selected;
        }
        call.executable.launch(owner_.cuda.stream());
        owner_.save_draft(static_cast<std::uint32_t>(step), batch);
        const auto& frame = *owner_.resources.plan.persistent.buffers.mtp;
        auto ids = owner_.device_tensor(frame.draft_steps[step + 1].ids).slice(0, 0, batch);
        owner_.ple.append_device(ticket, ids, extents, static_cast<std::uint32_t>(step));
    }
}

void Graphs::verify_head(std::uint32_t batch) {
    auto& group = batches_.at(batch - 1);
    workspace_peak_ = std::max(workspace_peak_, group.head_workspace);
    group.head.executable.launch(owner_.cuda.stream());
}

void Graphs::sample(std::uint32_t batch) {
    workspace_peak_ = std::max(workspace_peak_, batches_.at(batch - 1).sample_workspace);
    batches_.at(batch - 1).sample.executable.launch(owner_.cuda.stream());
}

void Graphs::accept(std::uint32_t batch) {
    workspace_peak_ = std::max(workspace_peak_, batches_.at(batch - 1).accept_workspace);
    batches_.at(batch - 1).accept.executable.launch(owner_.cuda.stream());
}

void RankProgramImpl::initialize_graphs() {
    if (!shape.cuda_graph) return;
    nvtx::ScopedRange range(nvtx::Name::CudaGraphPrepare, nvtx::Category::Graph);
    const auto before = device_memory_snapshot(cuda.device());
    graphs            = std::make_unique<Graphs>(*this);
    graphs->prepare();
    graph_memory.emplace();
    graph_memory->preparation_increment_bytes =
        device_memory_growth(before, device_memory_snapshot(cuda.device()));
    resources.states->storage().clear_all(cuda);
    device.synchronize();
}

void RankProgramImpl::release_graphs() {
    resources.synchronize();
    graphs.reset();
}

} // namespace ninfer::models::qwen4_exp::detail
