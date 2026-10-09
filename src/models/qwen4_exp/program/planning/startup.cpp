#include "ninfer/ops/speculative_round.h"
#include "models/qwen4_exp/program/planning/startup.h"

#include "core/device.h"
#include "models/qwen4_exp/execution/workspace.h"
#include "models/qwen4_exp/execution/text.h"
#include "models/qwen4_exp/program/internal.h"
#include "models/qwen4_exp/program/planning/graph_profiles.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/ple_ngram.h"
#include "ninfer/ops/qsa_indexer.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/selected_block_attention.h"
#include "ninfer/ops/sparse_moe.h"

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen4_exp::detail {
namespace {

using execution::dimension;
namespace workspace = execution::workspace;

constexpr std::size_t kMiB        = 1024ULL * 1024ULL;
constexpr std::size_t kArenaAlign = 256ULL;
// The PLE dilated convolution (four taps, dilation three) keeps nine history columns
// (ops::ple_ngram state [channels,9]).
constexpr std::int32_t kPleHistoryColumns = 9;
// Driver/module state of one instantiated ordinary decode executable (one per B and topology).
constexpr std::size_t kGraphTopologyAllowance = 16ULL * kMiB;

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

std::int32_t checked_i32(std::uint64_t value, const char* label) {
    if (value == 0 ||
        value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<std::int32_t>(value);
}

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

TensorLayout add_tensor(LayoutBuilder& builder, DType dtype,
                        std::initializer_list<std::int32_t> shape, const char* label) {
    return builder.add_tensor(dtype, shape, kArenaAlign, label);
}

PersistentLayout persistent_layout(const SequencePlanImpl& plan) {
    const auto& parameters = *plan.parameters;
    const auto& config     = parameters.model.config().text;
    if (!plan.context_cache.device_state_slots) {
        throw std::logic_error("Qwen4Exp context cache options are not normalized");
    }
    const std::int32_t state_image_slots = checked_i32(
        static_cast<std::uint64_t>(plan.max_concurrency) + *plan.context_cache.device_state_slots,
        "Qwen4Exp StateImage slot count exceeds int32");
    const auto effective_prefill_chunk =
        static_cast<std::int32_t>(std::min(plan.prefill_chunk, plan.capacity));
    LayoutBuilder builder;
    PersistentLayout out;
    out.decoder = qwen4_exp::plan_decoder_state(
        builder,
        qwen4_exp::DecoderStateSpec{
            .full_attention_layers     = config.full_attention_layers,
            .mtp_layers                = 1,
            .capacity                  = plan.capacity,
            .kv_heads                  = dimension(config.attention.num_key_value_heads),
            .attention_head_dim        = dimension(config.attention.head_dim),
            .indexer_dim               = dimension(config.indexer.head_dim),
            .indexer_block             = dimension(config.indexer.compress_ratio),
            .kv_storage                = plan.kv_storage,
            .enable_mtp                = plan.speculative_backend == SpeculativeBackend::Mtp,
            .kv_table_rows             = static_cast<std::int32_t>(plan.max_concurrency),
            .text_physical_page_groups = plan.main_page_groups,
            .mtp_physical_page_groups =
                plan.speculative_backend == SpeculativeBackend::Mtp ? plan.main_page_groups : 0,
        });
    out.state_images =
        qwen4_exp::plan_state_image_device_pool(
            builder,
            qwen4_exp::StateImageSpec{
                .linear =
                    {
                        .layers         = config.linear_attention_layers,
                        .conv_channels  = dimension(config.gdn.conv_channels()),
                        .conv_width     = dimension(config.gdn.linear_conv_kernel_dim - 1),
                        .value_heads    = dimension(config.gdn.linear_num_value_heads),
                        .value_head_dim = dimension(config.gdn.linear_value_head_dim),
                        .key_head_dim   = dimension(config.gdn.linear_key_head_dim),
                        .slot_count     = state_image_slots,
                        .conv_dtype     = DType::BF16,
                    },
                .token_mixer =
                    {
                        .indexer_layers =
                            config.full_attention_layers +
                            (plan.speculative_backend == SpeculativeBackend::Mtp ? 1U : 0U),
                        .indexer_dim   = dimension(config.indexer.head_dim),
                        .indexer_block = dimension(config.indexer.compress_ratio),
                        .ple_channels  = dimension(config.stream_width()),
                        .ple_history   = kPleHistoryColumns,
                        .transient_slots =
                            plan
                                        .speculative_backend == SpeculativeBackend::Mtp
                                ? static_cast<std::int32_t>(plan.max_concurrency *
                                                            (plan.draft_window + 1))
                                : 0,
                    },
                .hidden = dimension(config.stream_width()),
            });
    out.round = qwen4_exp::plan_round_state_layout(
        builder, qwen4_exp::RoundStateSpec{.stream_hidden  = dimension(config.stream_width()),
                                           .output_rows    = dimension(config.vocab_size),
                                           .batch_capacity = plan.max_concurrency,
                                           .causal_scoring = plan.causal_scoring});
    if (plan.speculative_backend == SpeculativeBackend::Mtp) {
        out.mtp         = plan_mtp_frame(config, plan.draft_window + 1, plan.max_concurrency);
        out.mtp_storage = builder.add(out.mtp->bytes, kArenaAlign, "MTP persistent frame");
    }
    if (plan.causal_scoring) {
        out.prefill_hidden = add_tensor(builder, DType::BF16,
                                        {dimension(config.hidden_size), effective_prefill_chunk},
                                        "scoring prefill mixed hidden");
        out.score_hidden =
            add_tensor(builder, DType::BF16,
                       {dimension(config.hidden_size), static_cast<std::int32_t>(kCausalScoreTile)},
                       "causal score hidden staging");
    } else {
        const auto public_tokens = dimension(parameters.model.resources().public_token_count);
        const auto lanes         = static_cast<std::int32_t>(plan.max_concurrency);
        out.token_counts =
            add_tensor(builder, DType::I32, {public_tokens, lanes}, "sampling token counts");
        const auto config_words = static_cast<std::int32_t>(
            (sizeof(ops::SamplingConfig) + sizeof(std::int32_t) - 1) / sizeof(std::int32_t));
        out.sampling_config =
            add_tensor(builder, DType::I32, {config_words, lanes}, "sampling config");
        out.grammar_masks = add_tensor(
            builder, DType::I32,
            {(public_tokens + 31) / 32, static_cast<std::int32_t>(plan.draft_window + 1), lanes},
            "grammar token masks");
        out.prompt_presence = add_tensor(builder, DType::I32, {(public_tokens + 31) / 32, lanes},
                                         "sampling prompt presence");
        out.logprob_candidate_ids = add_tensor(
            builder, DType::I32, {static_cast<std::int32_t>(kMaximumLogprobCandidates), lanes},
            "token logprob candidate ids");
        out.logprob_readout         = add_tensor(builder, DType::FP32,
                                                 {static_cast<std::int32_t>(kLogprobReadoutFloats), lanes},
                                                 "token logprob readout");
        out.logprob_prompt_next_ids = add_tensor(
            builder, DType::I32, {static_cast<std::int32_t>(kMaximumPromptReadouts), lanes},
            "prompt logprob next ids");
        out.logprob_prompt_readout = add_tensor(
            builder, DType::FP32, {static_cast<std::int32_t>(kLogprobPromptReadoutFloats), lanes},
            "prompt logprob readout");
    }
    out.bytes            = builder.finish(kArenaAlign, "persistent layout");
    out.kv_payload_bytes = out.decoder.kv_payload_bytes();
    return out;
}

// Mirrors the TextContext allocation sequence: phase roots, then per layer the PLE injection,
// the hyper preparation, the mixer stage with its Op scratch, the MoE and the final projection.
// Sequentially exclusive scratch takes the maximum; live roots are counted for their lifetime.
WorkspacePlan build_workspace_plan(const SequencePlanImpl& plan) {
    const auto& parameters        = *plan.parameters;
    const auto& config            = parameters.model.config().text;
    const std::uint32_t chunk_u32 = std::min(plan.prefill_chunk, plan.capacity);
    if (chunk_u32 == 0 ||
        chunk_u32 > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("sequence workspace dimensions are invalid");
    }
    const auto chunk         = static_cast<std::int32_t>(chunk_u32);
    const auto public_tokens = dimension(parameters.model.resources().public_token_count);

    const auto scratch = [](WorkspaceLayoutBuilder& layout, std::size_t bytes) {
        if (bytes == 0) { return; }
        auto scope = layout.scope();
        (void)layout.alloc_bytes(bytes);
    };
    const auto finish = [](const WorkspaceLayoutBuilder& layout) { return layout.peak_bytes(1); };
    const auto linear_bytes = [](const execution::LinearParameters& p, std::int32_t tokens) {
        return ops::linear_workspace_capacity_bytes(p.weight.qtype, p.weight.n, p.weight.k,
                                                    p.policy, tokens, tokens);
    };
    const auto& ple = parameters.text.ple;
    const ops::QsaIndexerSelectEnvelope full_envelope{
        indexer_envelope_blocks(plan.capacity - 1U, config.indexer.compress_ratio)};

    // One traversal of the 48 blocks at T columns (B rows when decoding).
    const auto body = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens, bool prefill,
                          std::int32_t width = 1) {
        for (std::uint32_t layer = 0; layer < parameters.text.layers.size(); ++layer) {
            const auto& block = parameters.text.layers[layer];
            if (layer == ple.layer) {
                auto stage = layout.scope();
                (void)workspace::ple_stage(layout, config, tokens);
                scratch(layout, linear_bytes(ple.key_projection, tokens));
                scratch(layout, linear_bytes(ple.value_projection, tokens));
                scratch(layout, std::max<std::size_t>(
                                    1, ops::ple_ngram_workspace_capacity_bytes(tokens, tokens)));
            }
            scratch(layout, ops::hyper_connection_workspace_capacity_bytes(tokens, tokens));
            {
                auto stage = layout.scope();
                if (const auto* a = std::get_if<execution::AttentionParameters>(&block.mixer)) {
                    (void)workspace::attention_stage(layout, config, tokens);
                    scratch(layout, linear_bytes(a->projection, tokens));
                    scratch(layout, linear_bytes(a->indexer_projection, tokens));
                    std::size_t attention_scratch =
                        ops::qsa_indexer_select_workspace_capacity_bytes(tokens, tokens,
                                                                         full_envelope);
                    if (!prefill) {
                        attention_scratch = std::max(
                            attention_scratch,
                            ops::selected_block_attention_workspace_capacity_bytes(tokens, tokens));
                    }
                    scratch(layout, attention_scratch);
                    scratch(layout, linear_bytes(a->output, tokens));
                } else {
                    const auto& g = std::get<execution::GdnParameters>(block.mixer);
                    (void)workspace::gdn_stage(layout, config, tokens, prefill);
                    const auto& w = g.projection.weight;
                    if (prefill) {
                        scratch(layout,
                                ops::gdn_input_proj_workspace_capacity_bytes(
                                    w.qtype, w.n, w.k, g.projection.policy, tokens, tokens));
                    } else {
                        scratch(layout,
                                std::max<std::size_t>(
                                    1, ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                                           w.qtype, w.n, w.k, g.projection.policy, tokens / width,
                                           width, width)));
                    }
                    scratch(layout, ops::gdn_gating_proj_workspace_capacity_bytes(
                                        dimension(config.gdn.linear_num_value_heads),
                                        dimension(config.hidden_size), tokens, tokens));
                    if (prefill) {
                        scratch(layout,
                                ops::gated_delta_net_workspace_capacity_bytes(
                                    dimension(config.gdn.linear_num_key_heads),
                                    dimension(config.gdn.linear_num_value_heads), tokens, tokens));
                    }
                    scratch(layout, linear_bytes(g.output, tokens));
                }
            }
            scratch(layout, ops::hyper_connection_workspace_capacity_bytes(tokens, tokens));
            scratch(layout, ops::sparse_moe_workspace_capacity_bytes(
                                block.moe.gate_up.policy, block.moe.down.policy, tokens, tokens));
        }
    };
    // Final mixer and output head over `columns` four-stream rows.
    const auto project_streams = [&](WorkspaceLayoutBuilder& layout, std::int32_t columns) {
        auto stage = layout.scope();
        (void)workspace::mixed_hidden(layout, config, columns);
        scratch(layout, ops::hyper_connection_workspace_capacity_bytes(columns, columns));
        scratch(layout, linear_bytes(parameters.text.output_head, columns));
    };

    const auto mtp_body = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens, bool teacher) {
        const auto& m = *parameters.mtp;
        (void)workspace::layer_roots(layout, config, tokens, Tensor{});
        {
            auto stem = layout.scope();
            (void)workspace::mixed_hidden(layout, config, tokens);
            (void)workspace::mixed_hidden(layout, config, tokens);
            (void)workspace::stream_hidden(layout, config, tokens);
            (void)workspace::stream_hidden(layout, config, tokens);
            scratch(layout, linear_bytes(m.embedding_projection, tokens));
            scratch(layout, linear_bytes(m.hidden_projection, 4 * tokens));
        }
        scratch(layout, ops::hyper_connection_workspace_capacity_bytes(tokens, tokens));
        {
            auto stage    = layout.scope();
            const auto& a = std::get<execution::AttentionParameters>(m.layer.mixer);
            (void)workspace::attention_stage(layout, config, tokens);
            scratch(layout, linear_bytes(a.projection, tokens));
            scratch(layout, linear_bytes(a.indexer_projection, tokens));
            if (!teacher) {
                scratch(layout, ops::qsa_indexer_select_workspace_capacity_bytes(tokens, tokens,
                                                                                 full_envelope));
                scratch(layout,
                        ops::selected_block_attention_workspace_capacity_bytes(tokens, tokens));
                scratch(layout, linear_bytes(a.output, tokens));
            }
        }
        if (!teacher) {
            scratch(layout, ops::sparse_moe_workspace_capacity_bytes(m.layer.moe.gate_up.policy,
                                                                     m.layer.moe.down.policy,
                                                                     tokens, tokens));
            (void)workspace::mixed_hidden(layout, config, tokens);
            scratch(layout, ops::hyper_connection_workspace_capacity_bytes(tokens, tokens));
            scratch(layout, linear_bytes(m.output_head, tokens));
        }
    };
    WorkspacePlan out;
    {
        WorkspaceLayoutBuilder layout;
        (void)workspace::prefill_controls(layout, chunk);
        const Tensor hidden = workspace::stream_hidden(layout, config, chunk);
        (void)workspace::layer_roots(layout, config, chunk, hidden);
        (void)workspace::ple_rows(layout, config, chunk);
        {
            auto stage = layout.scope();
            (void)workspace::mixed_hidden(layout, config, chunk);
            if (plan.features.vision) { (void)layout.alloc(DType::I32, {chunk}); }
        }
        body(layout, chunk, true);
        if (parameters.mtp) {
            auto teacher = layout.scope();
            (void)workspace::mixed_hidden(layout, config, chunk);
            if (plan.features.vision) { (void)layout.alloc(DType::I32, {chunk}); }
            (void)workspace::stream_hidden(layout, config, chunk);
            (void)workspace::stream_hidden(layout, config, chunk);
            (void)layout.alloc(DType::I32, {chunk});
            (void)layout.alloc(DType::I32, {chunk, 3});
            mtp_body(layout, chunk, true);
        }
        // Prompt readout tiles one column through the round logits; the reasoning feature mixes
        // one column; scoring mixes the whole chunk into its persistent hidden.
        {
            auto stage = layout.scope();
            (void)workspace::stream_hidden(layout, config, 1);
            project_streams(layout, 1);
        }
        if (plan.causal_scoring) {
            scratch(layout, ops::hyper_connection_workspace_capacity_bytes(chunk, chunk));
        } else {
            project_streams(layout, 1);
            scratch(layout, ops::sampling_workspace_capacity_bytes(public_tokens, 1, 1));
        }
        out.text_prefill = finish(layout);
    }
    if (plan.causal_scoring) {
        WorkspaceLayoutBuilder layout;
        constexpr auto tile = static_cast<std::int32_t>(kCausalScoreTile);
        constexpr auto rows = static_cast<std::int32_t>(kCausalScoreReadoutRows);
        (void)layout.alloc(DType::BF16, {dimension(config.vocab_size), tile});
        (void)layout.alloc(DType::I32, {rows, tile});
        (void)layout.alloc(DType::FP32, {rows, tile});
        (void)layout.alloc(DType::I32, {tile});
        (void)layout.alloc(DType::FP32, {tile});
        scratch(layout, linear_bytes(parameters.text.output_head, tile));
        out.causal_score = finish(layout);
    } else {
        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            WorkspaceLayoutBuilder layout;
            const Tensor hidden{};
            (void)workspace::layer_roots(layout, config, batch, hidden);
            {
                auto stage = layout.scope();
                (void)workspace::mixed_hidden(layout, config, batch);
            }
            body(layout, batch, false);
            project_streams(layout, batch);
            scratch(layout, ops::sampling_workspace_capacity_bytes(public_tokens, batch, batch));
            out.ordinary_round = std::max(out.ordinary_round, finish(layout));
        }
        // An exact prefix hit samples from its stored continuation hidden.
        WorkspaceLayoutBuilder exact_hit;
        project_streams(exact_hit, 1);
        scratch(exact_hit, ops::sampling_workspace_capacity_bytes(public_tokens, 1, 1));
        out.ordinary_round = std::max(out.ordinary_round, finish(exact_hit));
    }
    if (parameters.mtp) {
        const auto width = static_cast<std::int32_t>(plan.draft_window + 1);
        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            const auto columns = width * batch;
            WorkspaceLayoutBuilder verify;
            (void)workspace::layer_roots(verify, config, columns, Tensor{});
            {
                auto scope = verify.scope();
                (void)workspace::mixed_hidden(verify, config, columns);
            }
            body(verify, columns, false, width);
            project_streams(verify, columns);
            WorkspaceLayoutBuilder teacher;
            mtp_body(teacher, columns, true);
            WorkspaceLayoutBuilder proposal;
            mtp_body(proposal, batch, false);
            const auto accept = ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                public_tokens, width - 1, width - 1, batch, batch);
            out.mtp_round = std::max(
                {out.mtp_round, finish(verify), finish(teacher), finish(proposal), accept});
        }
    }
    out.general_capacity =
        std::max({out.text_prefill, out.ordinary_round, out.causal_score, out.mtp_round});
    out.capacity = out.general_capacity;
    if (plan.features.vision) {
        const auto merged = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(plan.capacity, kMaximumVisionItemTokens));
        out.vision = execution::VisionContext::plan_workspace(
            *parameters.model.config().vision, *parameters.vision, merged, out.general_capacity);
        out.capacity = std::max(out.capacity, out.vision->capacity_bytes);
    }
    return out;
}

void validate_target_options(const execution::Parameters& parameters, DeviceContext& device,
                             const EngineOptions& options) {
    const auto& config = parameters.model.config().text;
    if (config.full_attention_layers == 0 || config.linear_attention_layers == 0) {
        throw std::invalid_argument("Qwen4Exp Program requires attention and GDN layers");
    }
    if (parameters.model.options() != models::load_options(options)) {
        throw std::invalid_argument(
            "loaded components do not match the requested execution options");
    }
    require_supported_engine_options(options);
    if (options.max_context == 0 || options.max_context > config.max_position_embeddings) {
        throw std::invalid_argument("max_context exceeds the configured position capacity");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % kPrefillChunkAlignment != 0) {
        throw std::invalid_argument("prefill_chunk must be a nonzero multiple of 128");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("max_concurrency must be in [1,8]");
    }
    const std::uint32_t logical_pages = page_count(options.max_context);
    const std::uint32_t minimum_pages = std::max(logical_pages, options.max_concurrency);
    const std::uint64_t maximum_pages64 =
        static_cast<std::uint64_t>(options.max_concurrency) * logical_pages;
    if (maximum_pages64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("maximum Main KV page count exceeds uint32");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit: {
        if (options.kv_capacity.explicit_tokens < options.max_context) {
            throw std::invalid_argument("kv_capacity must be at least max_context");
        }
        const std::uint32_t requested_pages = page_count(options.kv_capacity.explicit_tokens);
        if (requested_pages < minimum_pages || requested_pages > maximum_pages64) {
            throw std::invalid_argument(
                "kv_capacity is outside the usable range for max_context and max_concurrency");
        }
        break;
    }
    case KvCapacityMode::Automatic:
        break;
    default:
        throw std::invalid_argument("unknown kv_capacity policy");
    }
    if (device.compute_capability() != 120) {
        throw std::invalid_argument("Qwen4Exp runtime requires compute capability 12.0");
    }
}

std::unique_ptr<SequencePlanImpl> build_sequence_candidate(const SequencePlanningInputs& inputs,
                                                           std::uint32_t main_page_groups) {
    if (main_page_groups == 0) {
        throw std::invalid_argument("Main KV physical page count must be positive");
    }
    auto impl                  = std::make_unique<SequencePlanImpl>();
    impl->parameters           = inputs.parameters;
    impl->capacity             = inputs.capacity;
    impl->main_page_groups     = main_page_groups;
    impl->kv_capacity          = static_cast<std::uint32_t>(checked_i32(
        static_cast<std::uint64_t>(main_page_groups) * static_cast<std::uint32_t>(kPagedKVPageSize),
        "resolved Paged KV capacity exceeds int32"));
    impl->max_concurrency      = inputs.max_concurrency;
    impl->prefill_chunk        = inputs.prefill_chunk;
    impl->draft_window         = inputs.draft_window;
    impl->speculative_backend  = inputs.speculative_backend;
    impl->features             = inputs.features;
    impl->use_cuda_graph       = inputs.use_cuda_graph;
    impl->causal_scoring       = inputs.causal_scoring;
    impl->device               = inputs.device;
    impl->context_cache        = inputs.context_cache;
    impl->kv_storage           = inputs.kv_storage;
    impl->multiprocessor_count = inputs.multiprocessor_count;
    impl->persistent           = persistent_layout(*impl);
    if (!impl->context_cache.host_capacity_bytes) {
        impl->context_cache.host_capacity_bytes =
            checked_add(8ULL * 1024 * 1024 * 1024,
                        checked_mul(8, impl->persistent.state_images.host.image_bytes,
                                    "Host context state default overflow"),
                        "Host context default overflow");
    }
    impl->workspace = build_workspace_plan(*impl);
    if (impl->use_cuda_graph) {
        // One executable per (B, topology class); profiles of a class update it in place.
        const auto& config = inputs.parameters->model.config().text;
        const auto profiles =
            ordinary_graph_profiles(impl->capacity, config.indexer.compress_ratio,
                                    config.indexer.budget / config.indexer.compress_ratio);
        std::uint32_t topologies = 0;
        for (const GraphExecutionProfile& profile : profiles) {
            topologies = std::max(topologies, profile.topology_class + 1U);
        }
        impl->graph_allowance_bytes =
            checked_mul(checked_mul(kGraphTopologyAllowance, topologies, "graph allowance"),
                        impl->max_concurrency *
                            (impl->speculative_backend == SpeculativeBackend::Mtp ? 2U : 1U),
                        "decode exact-b graph allowance");
    }
    impl->device_reservation_bytes = checked_add(
        checked_add(impl->persistent.bytes, impl->workspace.capacity, "sequence memory plan"),
        impl->graph_allowance_bytes, "sequence graph allowance");
    return impl;
}

} // namespace

} // namespace ninfer::models::qwen4_exp::detail

namespace ninfer::models::qwen4_exp {

void require_supported_engine_options(const EngineOptions& options) {
    if ((options.speculative.backend != SpeculativeBackend::None &&
         options.speculative.backend != SpeculativeBackend::Mtp) ||
        (options.speculative.backend == SpeculativeBackend::None &&
         options.speculative.draft_tokens != 0) ||
        (options.speculative.backend == SpeculativeBackend::Mtp &&
         (options.speculative.draft_tokens < 1 ||
          options.speculative.draft_tokens > kMtpMaximumDrafts)) ||
        options.speculative.proposal_head != ProposalHead::Full) {
        throw std::invalid_argument(
            "Qwen3.8-Flash-Next supports MTP with 1..5 greedy drafts and the full proposal head");
    }
    if (options.kv_cache != KvCacheStorage::BFloat16 &&
        options.kv_cache != KvCacheStorage::Fp8E4M3Row256) {
        throw std::invalid_argument(
            "Qwen3.8-Flash-Next supports the BF16 and FP8 row-scaled KV caches");
    }
}

} // namespace ninfer::models::qwen4_exp

namespace ninfer::models::qwen4_exp::detail {

std::unique_ptr<SequencePlannerImpl>
make_sequence_planner_impl(const execution::Parameters& parameters, DeviceContext& device,
                           const EngineOptions& options) {
    validate_target_options(parameters, device, options);
    SequencePlanningInputs inputs{
        .parameters           = &parameters,
        .capacity             = options.max_context,
        .max_concurrency      = options.max_concurrency,
        .prefill_chunk        = std::min(options.prefill_chunk, options.max_context),
        .draft_window         = options.speculative.draft_tokens,
        .speculative_backend  = options.speculative.backend,
        .kv_storage           = options.kv_cache,
        .features             = models::load_options(options),
        .use_cuda_graph       = options.use_cuda_graph,
        .causal_scoring       = options.purpose == EnginePurpose::CausalScoring,
        .device               = options.device,
        .multiprocessor_count = device.multiprocessor_count(),
        .context_cache        = options.context_cache,
    };
    const std::uint32_t logical_pages = page_count(inputs.capacity);
    const std::uint32_t minimum_pages = std::max(logical_pages, inputs.max_concurrency);
    const std::uint64_t maximum_pages64 =
        static_cast<std::uint64_t>(inputs.max_concurrency) * logical_pages;
    if (maximum_pages64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("maximum Main KV page count exceeds uint32");
    }
    const auto maximum_pages = static_cast<std::uint32_t>(maximum_pages64);

    auto planner     = std::make_unique<SequencePlannerImpl>();
    planner->inputs  = inputs;
    planner->minimum = build_sequence_candidate(inputs, minimum_pages);
    planner->curve   = runtime::SequenceCapacityCurve{
          .main_page_tokens                     = static_cast<std::uint32_t>(kPagedKVPageSize),
          .minimum_main_page_groups             = minimum_pages,
          .maximum_main_page_groups             = maximum_pages,
          .minimum_device_reservation_bytes     = planner->minimum->device_reservation_bytes,
          .bytes_per_additional_main_page_group = 0,
    };
    if (minimum_pages < maximum_pages) {
        auto adjacent = build_sequence_candidate(inputs, minimum_pages + 1U);
        if (adjacent->device_reservation_bytes <= planner->minimum->device_reservation_bytes) {
            throw std::logic_error("Qwen4Exp sequence layout has a nonpositive KV capacity stride");
        }
        planner->curve.bytes_per_additional_main_page_group =
            adjacent->device_reservation_bytes - planner->minimum->device_reservation_bytes;
    }
    return planner;
}

std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<SequencePlannerImpl> planner,
                            std::uint32_t main_page_groups) {
    if (planner == nullptr || planner->minimum == nullptr) {
        throw std::invalid_argument("Qwen4Exp sequence planner is empty");
    }
    const std::size_t expected = planner->curve.reservation_bytes(main_page_groups);
    std::unique_ptr<SequencePlanImpl> plan;
    if (main_page_groups == planner->curve.minimum_main_page_groups) {
        plan = std::move(planner->minimum);
    } else {
        plan = build_sequence_candidate(planner->inputs, main_page_groups);
    }
    if (plan->device_reservation_bytes != expected) {
        throw std::logic_error(
            "Qwen4Exp physical sequence layout is not affine in Main KV page capacity");
    }
    return plan;
}

} // namespace ninfer::models::qwen4_exp::detail
