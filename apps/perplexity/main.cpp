#include "corpus.h"
#include "evaluation.h"
#include "reference.h"
#include "options.h"
#include "report.h"

#include "core/platform.h"
#include "ninfer/engine.h"
#include "product/logging/configuration.h"
#include "product/logging/pretty_format.h"
#include "product/logging/startup_log.h"

#include <spdlog/logger.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using namespace ninfer::perplexity;
using json = nlohmann::json;
std::string kv_name(ninfer::KvCacheStorage value) {
    switch (value) {
    case ninfer::KvCacheStorage::BFloat16:
        return "bf16";
    case ninfer::KvCacheStorage::Int8Group64:
        return "int8-g64";
    case ninfer::KvCacheStorage::Fp8E4m3fnGroup128Ue8m0:
        return "fp8-e4m3-g128-ue8m0";
    case ninfer::KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case ninfer::KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    }
    throw std::logic_error("unknown KV dtype");
}


double seconds_since(Clock::time_point begin) {
    return std::chrono::duration<double>(Clock::now() - begin).count();
}

struct EvaluationStream {
    ninfer::perplexity::CorpusStream source;
    std::vector<ninfer::TokenId> tokens;
    std::vector<WindowPlan> windows;
};

json distribution_json(const ninfer::perplexity::DistributionAggregate& value) {
    return json{{"positions", value.positions()},
                {"kl_mean", value.mean_kl()},
                {"kl_p50", value.kl_quantile(0.50)},
                {"kl_p90", value.kl_quantile(0.90)},
                {"kl_p99", value.kl_quantile(0.99)},
                {"kl_p999", value.kl_quantile(0.999)},
                {"kl_max", value.kl_quantile(1.0)},
                {"top1_agreement", value.top1_agreement()},
                {"evaluated_mean_nll", value.mean_evaluated_nll()},
                {"reference_mean_nll", value.mean_reference_nll()},
                {"delta_mean_nll", value.mean_evaluated_nll() - value.mean_reference_nll()}};
}

// Scores the reference file's exact token sequences and compares every next-token distribution
// with the reference model's top-K distribution.
int run_reference(const Options& options, const std::shared_ptr<spdlog::logger>& logger,
                  ninfer::product::StartupLogRenderer& startup_log,
                  const std::shared_ptr<ninfer::product::TerminalProgress>& progress) {
    using ninfer::perplexity::DistributionAggregate;
    const Clock::time_point total_started = Clock::now();
    const Clock::time_point read_started  = Clock::now();
    const ninfer::perplexity::ReferenceCorpus reference =
        ninfer::perplexity::load_reference(*options.reference);
    if (reference.top_k > ninfer::kMaximumCausalReferenceTokens) {
        throw std::runtime_error("reference top_k exceeds kMaximumCausalReferenceTokens");
    }
    std::uint64_t total_positions = 0;
    for (const auto& sequence : reference.sequences) {
        if (sequence.tokens.size() > options.context) {
            throw std::runtime_error("reference sequence " + sequence.id +
                                     " is longer than --context");
        }
        total_positions += sequence.tokens.size() - 1;
    }
    const double read_seconds = seconds_since(read_started);

    auto engine_options = options.launch.engine;
    engine_options.startup_observer = startup_log.observer();
    ninfer::Engine engine(std::move(engine_options));
    const ninfer::LoadSummary load = engine.startup_report().load;
    startup_log.engine_ready(load);
    logger->info("reference ready | {} | {} sequences | {} positions | top-{}",
                 reference.corpus_id, reference.sequences.size(),
                 ninfer::product::format_pretty_count(total_positions), reference.top_k);

    CorpusSelection selection{.corpus_id = reference.corpus_id,
                              .mode      = "reference",
                              .source    = reference.source,
                              .streams   = {}};
    const std::filesystem::path output_directory =
        prepare_output_directory(options,
            RunDescription{ninfer::product::configuration_view(options.launch,
                ninfer::product::EntryPoint::Perplexity), engine.description(),
                engine.startup_report(), engine.snapshot()}, selection);
    const Clock::time_point scoring_started = Clock::now();
    Clock::time_point next_progress         = scoring_started + std::chrono::seconds(10);
    const std::uint32_t k                   = reference.top_k;
    DistributionAggregate overall;
    std::map<std::string, DistributionAggregate> domains;
    json sequence_reports = json::array();
    // Per scored target, in reference order: float32 KL, float32 evaluated target log
    // probability, int32 evaluated argmax token.
    std::ofstream positions;
    if (options.dump_positions) {
        positions.open(*options.dump_positions, std::ios::binary | std::ios::trunc);
        if (!positions) {
            throw std::runtime_error("cannot create " + options.dump_positions->string());
        }
    }
    for (std::size_t index = 0; index < reference.sequences.size(); ++index) {
        const auto& sequence = reference.sequences[index];
        const std::size_t targets = sequence.tokens.size() - 1;
        ninfer::CausalScoreReadout readout{.reference_count  = k,
                                           .reference_tokens = sequence.top_tokens};
        const Clock::time_point started = Clock::now();
        ninfer::CausalScores scores;
        try {
            scores = engine.score_tokens(sequence.tokens, 1, readout);
        } catch (const std::exception& error) {
            throw std::runtime_error("scoring reference sequence " + sequence.id +
                                     " failed: " + error.what());
        }
        DistributionAggregate local;
        for (std::size_t target = 0; target < targets; ++target) {
            const std::span<const float> p(sequence.top_logprobs.data() + target * k, k);
            const std::span<const float> q(scores.reference_logprobs.data() + target * k, k);
            const double kl = ninfer::perplexity::top_k_kl_divergence(p, q);
            local.add(kl, scores.argmax_tokens[target] == sequence.top_tokens[target * k],
                      scores.target_logprobs[target], sequence.target_logprobs[target]);
            if (positions.is_open()) {
                const float values[2] = {static_cast<float>(kl), scores.target_logprobs[target]};
                positions.write(reinterpret_cast<const char*>(values), sizeof(values));
                positions.write(reinterpret_cast<const char*>(&scores.argmax_tokens[target]),
                                sizeof(ninfer::TokenId));
            }
        }
        overall.add(local);
        domains[sequence.domain].add(local);
        json item      = distribution_json(local);
        item["id"]     = sequence.id;
        item["domain"] = sequence.domain;
        item["seconds"] = seconds_since(started);
        sequence_reports.push_back(std::move(item));
        if (Clock::now() >= next_progress) {
            std::ostringstream line;
            line << "scoring | " << ninfer::product::format_pretty_count(overall.positions())
                 << '/' << ninfer::product::format_pretty_count(total_positions)
                 << " positions | mean KL " << std::scientific << std::setprecision(4)
                 << overall.mean_kl();
            if (progress->enabled()) {
                progress->update("  " + line.str());
            } else {
                logger->info("{}", line.str());
            }
            next_progress = Clock::now() + std::chrono::seconds(10);
        }
    }
    const double scoring_seconds = seconds_since(scoring_started);
    progress->clear();

    json domain_reports = json::array();
    for (const auto& [domain, aggregate] : domains) {
        json item      = distribution_json(aggregate);
        item["domain"] = domain;
        domain_reports.push_back(std::move(item));
    }
    json report{
        {"schema_version", 2},
        {"metric",
         {{"name", "next-token distribution agreement with a reference"},
          {"divergence", "KL(reference || evaluated) over the reference top-K plus a tail bucket"},
          {"log_base", "natural"}}},
        {"artifact",
         {{"path", std::filesystem::absolute(options.launch.engine.artifact_path).lexically_normal().string()},
          {"architecture", load.architecture},
          {"name", load.model_name},
          {"prefill_signature", load.prefill_signature},
          {"formats", load.weight_formats}}},
        {"reference",
         {{"path", std::filesystem::absolute(reference.source).lexically_normal().string()},
          {"corpus_id", reference.corpus_id},
          {"model", json::parse(reference.model)},
          {"top_k", k},
          {"sequences", reference.sequences.size()}}},
        {"execution",
         {{"purpose", "causal_scoring"},
          {"device", options.launch.engine.device},
          {"context_tokens", options.context},
          {"kv_dtype", kv_name(options.launch.engine.kv_cache)}}},
        {"timing",
         {{"load_seconds", load.load_seconds},
          {"read_seconds", read_seconds},
          {"score_seconds", scoring_seconds},
          {"total_seconds", seconds_since(total_started)}}},
        {"sequences", std::move(sequence_reports)},
        {"domains", std::move(domain_reports)},
        {"overall", distribution_json(overall)},
    };
    const std::filesystem::path temporary = output_directory / "report.json.tmp";
    const std::filesystem::path final     = output_directory / "report.json";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) { throw std::runtime_error("cannot create report: " + temporary.string()); }
        output << std::setw(2) << report << '\n';
        output.flush();
        if (!output) { throw std::runtime_error("cannot write report: " + temporary.string()); }
    }
    std::filesystem::rename(temporary, final);

    std::cout << "Reference distribution result\n"
              << "artifact: " << load.model_name << ", kv: " << kv_name(options.launch.engine.kv_cache)
              << ", reference: " << reference.corpus_id << " (top-" << k << ")\n\n";
    std::cout << std::left << std::setw(20) << "domain" << std::right << std::setw(10)
              << "positions" << std::setw(12) << "KL mean" << std::setw(12) << "KL p50"
              << std::setw(12) << "KL p99" << std::setw(10) << "top-1" << std::setw(12)
              << "dNLL" << '\n';
    const auto row = [](const std::string& label, const DistributionAggregate& value) {
        std::cout << std::left << std::setw(20) << label << std::right << std::setw(10)
                  << value.positions() << std::scientific << std::setprecision(3)
                  << std::setw(12) << value.mean_kl() << std::setw(12) << value.kl_quantile(0.5)
                  << std::setw(12) << value.kl_quantile(0.99) << std::fixed
                  << std::setprecision(4) << std::setw(10) << value.top1_agreement()
                  << std::setprecision(5) << std::setw(12)
                  << value.mean_evaluated_nll() - value.mean_reference_nll() << '\n';
    };
    for (const auto& [domain, aggregate] : domains) { row(domain, aggregate); }
    row("overall", overall);
    std::cout << "\nreport: " << final << '\n';
    return 0;
}

int run(const Options& options, const std::shared_ptr<spdlog::logger>& logger,
        ninfer::product::StartupLogRenderer& startup_log,
        const std::shared_ptr<ninfer::product::TerminalProgress>& progress) {
    const Clock::time_point total_started = Clock::now();
    auto engine_options                   = options.launch.engine;
    engine_options.startup_observer       = startup_log.observer();
    ninfer::Engine engine(std::move(engine_options));
    const RunDescription description{ninfer::product::configuration_view(
                                         options.launch, ninfer::product::EntryPoint::Perplexity),
                                     engine.description(), engine.startup_report(),
                                     engine.snapshot()};
    startup_log.engine_ready(description.startup.load);
    ninfer::product::log_resolved_configuration(*logger, description.engine);
    logger->info("scoring | window {} | stride {}", options.context, options.stride);

    const Clock::time_point preflight_started = Clock::now();
    logger->info("preparing corpus");
    CorpusSelection corpus = options.corpus
                                 ? ninfer::perplexity::load_corpus(*options.corpus, options.quick)
                                 : ninfer::perplexity::load_custom_text(*options.text);
    std::vector<EvaluationStream> streams;
    streams.reserve(corpus.streams.size());
    std::uint64_t total_scored_tokens = 0;
    std::uint64_t total_input_tokens  = 0;
    std::uint64_t total_windows       = 0;
    for (auto& source : corpus.streams) {
        std::vector<ninfer::TokenId> tokens = engine.tokenize_text(source.text);
        if (tokens.size() < 2) {
            throw std::runtime_error("stream tokenized to fewer than two tokens: " + source.id);
        }
        std::vector<WindowPlan> windows =
            ninfer::perplexity::plan_windows(tokens.size(), options.context, options.stride);
        total_input_tokens += static_cast<std::uint64_t>(tokens.size());
        total_scored_tokens += static_cast<std::uint64_t>(tokens.size() - 1);
        total_windows += static_cast<std::uint64_t>(windows.size());
        streams.push_back(EvaluationStream{.source  = std::move(source),
                                           .tokens  = std::move(tokens),
                                           .windows = std::move(windows)});
    }
    const double preflight_seconds = seconds_since(preflight_started);
    logger->info("corpus ready | {} streams | {} input tokens | {} scored tokens | {} windows | {}",
                 ninfer::product::format_pretty_count(streams.size()),
                 ninfer::product::format_pretty_count(total_input_tokens),
                 ninfer::product::format_pretty_count(total_scored_tokens),
                 ninfer::product::format_pretty_count(total_windows),
                 ninfer::product::format_pretty_duration(preflight_seconds));

    const std::filesystem::path output_directory =
        prepare_output_directory(options, description, corpus);
    const Clock::time_point scoring_started = Clock::now();
    logger->info("scoring | {} streams | {} tokens | {} windows",
                 ninfer::product::format_pretty_count(streams.size()),
                 ninfer::product::format_pretty_count(total_scored_tokens),
                 ninfer::product::format_pretty_count(total_windows));
    Clock::time_point next_progress = scoring_started + std::chrono::seconds(10);
    EvaluationResult result;
    result.preflight_seconds = preflight_seconds;
    result.streams.reserve(streams.size());
    auto& overall                   = result.overall;
    std::uint64_t completed_windows = 0;

    for (std::size_t stream_index = 0; stream_index < streams.size(); ++stream_index) {
        EvaluationStream& stream = streams[stream_index];
        std::ostringstream stream_status;
        stream_status << "  scoring [" << stream_index + 1 << '/' << streams.size() << "] "
                      << ninfer::product::format_pretty_text(stream.source.id) << " | "
                      << ninfer::product::format_pretty_count(stream.tokens.size()) << " tokens | "
                      << ninfer::product::format_pretty_count(stream.windows.size()) << " windows";
        if (progress->enabled()) {
            progress->update(stream_status.str());
        } else {
            logger->debug("{}", stream_status.str());
        }
        const Clock::time_point stream_started = Clock::now();
        StreamResult stream_result{.id           = stream.source.id,
                                   .domain       = stream.source.domain,
                                   .path         = stream.source.path,
                                   .input_tokens = stream.tokens.size()};
        stream_result.windows.reserve(stream.windows.size());
        auto& stream_score = stream_result.score;
        for (std::size_t window_index = 0; window_index < stream.windows.size(); ++window_index) {
            const WindowPlan& window = stream.windows[window_index];
            std::vector<ninfer::TokenId> input(
                stream.tokens.begin() + static_cast<std::ptrdiff_t>(window.input_begin),
                stream.tokens.begin() + static_cast<std::ptrdiff_t>(window.input_end));
            const Clock::time_point window_started = Clock::now();
            std::vector<float> logprobs;
            try {
                logprobs = engine.score_tokens(std::move(input), window.first_target);
            } catch (const std::exception& error) {
                throw std::runtime_error("scoring " + stream.source.id + " window " +
                                         std::to_string(window_index) + " failed: " + error.what());
            }
            const std::size_t expected = window.target_end - window.target_begin;
            if (logprobs.size() != expected) {
                throw std::runtime_error("scoring returned an invalid target count for " +
                                         stream.source.id);
            }
            ScoreAggregate window_score;
            window_score.add(logprobs);
            stream_score.add(window_score);
            overall.add(window_score);
            result.domains[stream.source.domain].add(window_score);
            ++completed_windows;
            stream_result.windows.push_back({window, window_score, seconds_since(window_started)});

            if (Clock::now() >= next_progress) {
                const double elapsed = seconds_since(scoring_started);
                const double rate    = static_cast<double>(overall.scored_tokens) / elapsed;
                const std::uint64_t remaining = total_scored_tokens - overall.scored_tokens;
                const double eta = rate > 0 ? static_cast<double>(remaining) / rate : 0.0;
                std::ostringstream line;
                line << "scoring | " << ninfer::product::format_pretty_count(overall.scored_tokens)
                     << '/' << ninfer::product::format_pretty_count(total_scored_tokens)
                     << " tokens | " << completed_windows << '/' << total_windows
                     << " windows | PPL " << std::fixed << std::setprecision(4) << overall.ppl()
                     << " | " << ninfer::product::format_pretty_rate(rate, "tok") << " | elapsed "
                     << ninfer::product::format_pretty_duration(elapsed) << " | ETA "
                     << ninfer::product::format_pretty_duration(eta);
                if (progress->enabled()) {
                    progress->update("  " + line.str());
                } else {
                    logger->info("{}", line.str());
                }
                next_progress = Clock::now() + std::chrono::seconds(10);
            }
        }
        const double stream_seconds = seconds_since(stream_started);
        progress->clear();
        logger->info("[{}/{}] {} | {} scored tokens | PPL {:.6g} | {}", stream_index + 1,
                     streams.size(), ninfer::product::format_pretty_text(stream.source.id),
                     ninfer::product::format_pretty_count(stream_score.scored_tokens),
                     stream_score.ppl(), ninfer::product::format_pretty_duration(stream_seconds));
        stream_result.seconds = stream_seconds;
        result.streams.push_back(std::move(stream_result));
    }

    const double scoring_seconds = seconds_since(scoring_started);
    progress->clear();
    logger->info("scoring complete | {} tokens | {} windows | PPL {:.6g} | {} | {}",
                 ninfer::product::format_pretty_count(overall.scored_tokens), completed_windows,
                 overall.ppl(), ninfer::product::format_pretty_duration(scoring_seconds),
                 ninfer::product::format_pretty_rate(
                     static_cast<double>(overall.scored_tokens) / scoring_seconds, "tok"));
    result.scoring_seconds = scoring_seconds;
    result.total_seconds   = seconds_since(total_started);
    write_report(output_directory, make_report(options, description, corpus, result), std::cout);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    ninfer::perplexity::Options options;
    try {
        options = ninfer::perplexity::parse_options(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "ninfer-perplexity: " << error.what() << '\n';
        std::cerr << ninfer::perplexity::usage_text();
        return 1;
    }
    if (options.help_requested) {
        std::cout << ninfer::perplexity::usage_text();
        return 0;
    }

    ninfer::product::LoggingRuntime logging(
        {.logger_name  = "ninfer-perplexity",
         .level        = options.launch.logging.level,
         .presentation = ninfer::product::LogPresentation::Tool});
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
    ninfer::product::StartupLogRenderer startup_log(logging);
    try {
        ninfer::product::log_configuration(
            *logger, ninfer::product::configuration_view(options.launch,
                                                         ninfer::product::EntryPoint::Perplexity));
        return options.reference
                   ? run_reference(options, logger, startup_log, logging.terminal_progress())
                   : run(options, logger, startup_log, logging.terminal_progress());
    } catch (const std::exception& error) {
        logging.terminal_progress()->clear();
        logger->error("{}", ninfer::product::format_pretty_text(error.what()));
        return 1;
    }
}
