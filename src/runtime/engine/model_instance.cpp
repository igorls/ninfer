#include "runtime/engine/model_instance.h"
#include "artifact/reader.h"
#include "artifact/formats.h"
#include "core/device_memory.h"
#include "core/startup.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/measurement.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <set>
#include <stdexcept>
#include <utility>

namespace ninfer::runtime {
namespace {
using Clock = std::chrono::steady_clock;

void validate_options(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
}

// Sizing reads device-wide free memory (NVML, every process), not this process's view: on a
// shared desktop GPU other applications hold memory that cudaMemGetInfo does not attribute.
std::size_t runtime_bytes_after_weights(int device, std::size_t desktop_reserve_bytes) {
    const DeviceMemorySnapshot memory = query_device_memory(device);
    if (memory.free_bytes < desktop_reserve_bytes) {
        throw std::invalid_argument(
            "Insufficient device memory after loading weights: free memory is " +
            format_device_memory_bytes(memory.free_bytes) +
            ", which is less than the required desktop reserve floor of " +
            format_device_memory_bytes(desktop_reserve_bytes) + ".\n" +
            "Another process or OS component holds " +
            format_device_memory_bytes(memory.used_bytes) + " on device " + std::to_string(device) +
            (memory.device_name.empty() ? "" : " (" + memory.device_name + ")") + ".\n" +
            "Telemetry source: " +
            (memory.is_nvml ? "NVML device-wide query" : "cudaMemGetInfo fallback") + ".");
    }
    return memory.free_bytes - desktop_reserve_bytes;
}

// Reports how much of max_concurrency full-context sequences the resolved pool backs, and lowers
// the effective concurrency to the backed count when the options ask for that.
std::uint32_t reconcile_concurrency(const EngineOptions& options,
                                    const SequenceCapacityCurve& curve,
                                    KvCapacityResolution& resolution) {
    const std::uint32_t groups_per_sequence =
        (options.max_context + curve.main_page_tokens - 1U) / curve.main_page_tokens;
    const std::uint32_t full_demand = options.max_concurrency * groups_per_sequence;
    const double coverage           = static_cast<double>(resolution.main_page_groups) /
                            static_cast<double>(std::max(1U, full_demand));
    std::uint32_t effective = options.max_concurrency;
    if (options.clamp_concurrency_to_pool && resolution.main_page_groups < full_demand) {
        effective = std::max(1U, resolution.main_page_groups / std::max(1U, groups_per_sequence));
    }
    resolution.requested_concurrency      = options.max_concurrency;
    resolution.effective_concurrency      = effective;
    resolution.unbacked_concurrency_ratio = coverage < 1.0 ? 1.0 / std::max(coverage, 1e-9) : 1.0;
    if (effective < options.max_concurrency) {
        std::fprintf(stderr,
                     "[kv-sizer] concurrency clamped from %u to %u lanes so every lane can hold "
                     "a full %u-token context (--clamp-concurrency-to-pool)\n",
                     options.max_concurrency, effective, options.max_context);
    } else if (coverage < 1.0) {
        std::fprintf(stderr,
                     "[kv-sizer] KV pool backs %.1f%% of %u full-context sequences; concurrent "
                     "long requests beyond it wait for admission\n",
                     coverage * 100.0, options.max_concurrency);
    }
    return effective;
}

// After startup the device must still hold the desktop reserve. Engine growth beyond its planned
// reservation is a defect; foreign growth during the load is only reported.
void check_post_startup_reserve(const EngineOptions& options,
                                const KvCapacityResolution& resolution,
                                std::size_t available_before_program,
                                std::size_t available_after_startup) {
    if (options.desktop_reserve_bytes == 0 ||
        available_after_startup >= options.desktop_reserve_bytes) {
        return;
    }
    const std::size_t engine_growth             = available_before_program > available_after_startup
                                                      ? available_before_program - available_after_startup
                                                      : 0;
    constexpr std::size_t kGrowthToleranceBytes = 32ULL * 1024ULL * 1024ULL;
    if (engine_growth > resolution.runtime_reservation_bytes + kGrowthToleranceBytes) {
        throw std::runtime_error(
            "Engine resident memory growth (" + format_device_memory_bytes(engine_growth) +
            ") exceeded its planned runtime reservation (" +
            format_device_memory_bytes(resolution.runtime_reservation_bytes) +
            ") and breached the desktop reserve floor (" +
            format_device_memory_bytes(options.desktop_reserve_bytes) +
            ", free remaining: " + format_device_memory_bytes(available_after_startup) + ")");
    }
    std::fprintf(stderr,
                 "[kv-sizer] WARNING: device free memory after startup (%zu MiB) is below the "
                 "desktop reserve floor (%zu MiB); the Engine stayed within its planned %zu MiB, "
                 "so another process took memory during the load\n",
                 available_after_startup >> 20U, options.desktop_reserve_bytes >> 20U,
                 resolution.runtime_reservation_bytes >> 20U);
}

} // namespace

EngineOptions normalize_engine_options(EngineOptions options) {
    switch (options.purpose) {
    case EnginePurpose::Generation:
        break;
    case EnginePurpose::CausalScoring:
        options.max_concurrency      = 1;
        options.max_pending_requests = 1;
        options.prefill_chunk        = 1024;
        options.kv_capacity          = KvCapacityPolicy::explicit_capacity(options.max_context);
        options.speculative          = {};
        options.enable_vision        = false;
        options.use_cuda_graph       = false;
        options.context_cache        = ContextCacheOptions{.enabled = false};
        break;
    default:
        throw std::invalid_argument("Engine purpose is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }

    ContextCacheOptions& cache      = options.context_cache;
    const std::uint32_t concurrency = options.max_concurrency;
    if (!cache.enabled) {
        if ((cache.device_state_slots && *cache.device_state_slots != 0) ||
            (cache.max_private_continuations && *cache.max_private_continuations != concurrency) ||
            (cache.max_shared_prefixes && *cache.max_shared_prefixes != 0) ||
            (cache.max_long_anchors_per_continuation &&
             *cache.max_long_anchors_per_continuation != 0)) {
            throw std::invalid_argument("disabled context cache accepts only root-only capacities");
        }
        cache.device_state_slots                = 0;
        cache.host_state_slots                  = 0;
        cache.host_kv_capacity_bytes            = 0;
        cache.max_private_continuations         = concurrency;
        cache.max_shared_prefixes               = 0;
        cache.max_long_anchors_per_continuation = 0;
        return options;
    }

    cache.device_state_slots            = cache.device_state_slots.value_or(concurrency);
    const std::uint64_t default_private = 2ULL * concurrency;
    cache.max_private_continuations =
        cache.max_private_continuations.value_or(static_cast<std::uint32_t>(default_private));
    cache.max_shared_prefixes = cache.max_shared_prefixes.value_or(
        std::max(concurrency, static_cast<std::uint32_t>(kMaximumExplicitPromptCacheMarkers)));
    cache.max_long_anchors_per_continuation = cache.max_long_anchors_per_continuation.value_or(2U);

    if (*cache.max_private_continuations < concurrency) {
        throw std::invalid_argument(
            "context cache max_private_continuations must cover every active request");
    }
    // A catalogued continuation needs a StateImage to hold it, and the active lanes consume
    // their own; extra device plus Host state slots are what can back checkpoints. Advertising a
    // larger catalog reports capacity that cannot exist (the 27B once claimed 64 against 32).
    const std::uint64_t backing =
        static_cast<std::uint64_t>(*cache.device_state_slots) + cache.host_state_slots;
    if (backing >= concurrency && *cache.max_private_continuations > backing) {
        cache.max_private_continuations = static_cast<std::uint32_t>(backing);
    }
    const std::uint64_t total_device_state_slots =
        static_cast<std::uint64_t>(concurrency) + *cache.device_state_slots;
    if (total_device_state_slots > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache Device state capacity exceeds uint32");
    }
    const std::uint64_t address_spaces =
        static_cast<std::uint64_t>(*cache.max_private_continuations) + *cache.max_shared_prefixes;
    if (address_spaces > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache address-space capacity exceeds uint32");
    }
    if (*cache.max_long_anchors_per_continuation != 0 &&
        *cache.max_private_continuations >
            std::numeric_limits<std::size_t>::max() / *cache.max_long_anchors_per_continuation) {
        throw std::overflow_error("context cache long-anchor capacity exceeds size_t");
    }
    return options;
}

ModelInstance::ModelInstance(std::unique_ptr<models::qwen3_5::Model> source,
                             const EngineOptions& options)
    : model(std::move(source)), parameters(*model),
      frontend(models::qwen3_5::make_frontend(
          model->resources(), {.chat_template_path       = options.chat_template_path,
                               .architecture             = model->config().text.architecture,
                               .vision_enabled           = options.enable_vision,
                               .max_context              = options.max_context,
                               .media_cache_bytes        = options.media_cache_bytes,
                               .media_live_bytes         = options.media_live_bytes,
                               .media_preprocess_threads = options.media_preprocess_threads})),
      capacity(options.max_context) {}

ModelInstance::~ModelInstance() = default;

ConstructedModel construct_model(EngineOptions& options, DeviceContext& device) {
    validate_options(options);
    const auto start = Clock::now();
    set_runtime_desktop_reserve_floor(0);
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    artifact::Reader reader(options.artifact_path);
    inspect.complete();
    StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
    auto plan = models::qwen3_5::plan_load(reader, models::load_options(options));
    {
        // Fail before a multi-GiB upload when the weights alone cannot leave the reserve.
        const DeviceMemorySnapshot memory = query_device_memory(device.device);
        const std::size_t weights =
            static_cast<std::size_t>(plan.materialization().device_capacity_bytes);
        if (memory.free_bytes < weights + options.desktop_reserve_bytes) {
            throw std::invalid_argument(format_insufficient_memory_error(
                memory, device.device, weights, options.desktop_reserve_bytes));
        }
    }
    binding.complete();
    auto model =
        models::qwen3_5::materialize_model(std::move(plan), device, &options.startup_observer);
    device.synchronize();
    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    auto instance = std::make_unique<ModelInstance>(std::move(model), options);
    frontend.complete();
    StartupPhaseScope planning(options.startup_observer, StartupPhase::TargetFinalize);
    const auto signature = models::qwen3_5::prefill_signature(*instance->model);
    auto context_cost    = resolve_context_machine_cost(
        {.hardware_class =
                context_cost_hardware_class(device.props.name, device.props.major, device.props.minor),
            .prefill_signature = signature},
        options.context_cost.preset_path);
    auto planner = models::qwen3_5::make_sequence_planner(instance->parameters, device, options);
    const SequenceCapacityCurve curve = planner.capacity_curve();
    auto resolution                   = resolve_kv_capacity(
        options.kv_capacity, curve,
        runtime_bytes_after_weights(device.device, options.desktop_reserve_bytes));
    resolution.desktop_reserve_bytes          = options.desktop_reserve_bytes;
    const std::uint32_t effective_concurrency = reconcile_concurrency(options, curve, resolution);
    if (effective_concurrency != options.max_concurrency) {
        options.max_concurrency = effective_concurrency;
        options                 = normalize_engine_options(std::move(options));
        planner = models::qwen3_5::make_sequence_planner(instance->parameters, device, options);
    }
    auto sequence = std::move(planner).finalize(resolution.main_page_groups);
    if (sequence.device_reservation_bytes() != resolution.runtime_reservation_bytes ||
        sequence.kv_capacity() != resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized Program plan");
    }
    instance->kv_capacity_resolution = resolution;
    planning.complete();
    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    const std::size_t available_before_program = query_device_memory(device.device).free_bytes;
    instance->program = models::qwen3_5::create_program(instance->parameters, std::move(sequence),
                                                        device, options.startup_observer);
    device.synchronize();
    program.complete();
    instance->kv_capacity_resolution.available_after_startup_bytes =
        query_device_memory(device.device).free_bytes;
    check_post_startup_reserve(options, instance->kv_capacity_resolution, available_before_program,
                               instance->kv_capacity_resolution.available_after_startup_bytes);
    set_runtime_desktop_reserve_floor(options.desktop_reserve_bytes);
    const auto& stats = instance->model->storage_stats();
    LoadSummary summary;
    summary.architecture = models::architecture_name(instance->model->config().text.architecture);
    summary.model_name   = instance->model->info().name;
    summary.prefill_signature = signature;
    std::set<std::string> formats;
    for (const auto& weight : instance->model->weight_data()) {
        for (const auto& part : weight.view.parts) {
            formats.emplace(artifact::format_name(part.parent->geometry.format));
        }
    }
    summary.weight_formats.assign(formats.begin(), formats.end());
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.read_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.device_object_count  = stats.device_object_count;
    summary.host_object_count    = stats.host_object_count;
    summary.context_cost         = std::move(context_cost.summary);
    return {std::move(instance), std::move(summary), std::move(context_cost.model)};
}

} // namespace ninfer::runtime
