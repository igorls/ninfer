// ninfer-hidden-export: final-normalized hidden rows of token sequences from a .ninfer artifact,
// written as one safetensors file for training readouts (decision heads). Uses the offline
// CausalScoring Engine, so every record is scored from fresh state with no context cache.
//
// Input: JSON lines {"id": "...", "tokens": [...]} with 2 <= tokens <= --context. The rows cover
// predictor positions 0 .. tokens-2 (the last token is never a predictor): a caller that needs
// every real position appends one token itself. Output tensors: "hidden/<id>" BF16 [tokens-1, H]
// and "logprobs/<id>" F32 [tokens-1], the log probability of tokens[i+1] at row i.

#include "ninfer/engine.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using json = nlohmann::json;

struct Record {
    std::string id;
    std::vector<ninfer::TokenId> tokens;
};

struct Settings {
    std::string artifact;
    std::string input;
    std::string output;
    std::uint32_t context       = 16384;
    std::uint32_t prefill_chunk = 2048;
    std::string kv_dtype        = "fp8";
    int device                  = 0;
    std::size_t limit           = 0;
};

ninfer::KvCacheStorage parse_kv_dtype(const std::string& value) {
    if (value == "bf16") { return ninfer::KvCacheStorage::BFloat16; }
    if (value == "int8") { return ninfer::KvCacheStorage::Int8Group64; }
    if (value == "fp8") { return ninfer::KvCacheStorage::Fp8E4M3Row256; }
    if (value == "nvfp4") { return ninfer::KvCacheStorage::Nvfp4Group16; }
    if (value == "k8v4") { return ninfer::KvCacheStorage::Fp8KeyNvfp4Value; }
    throw std::invalid_argument("--kv-dtype must be bf16, int8, fp8, nvfp4 or k8v4");
}

void usage() {
    std::cerr << "usage: ninfer-hidden-export <artifact.ninfer> --input records.jsonl "
                 "--output rows.safetensors\n"
                 "       [--context N=16384] [--prefill-chunk N=2048] "
                 "[--kv-dtype bf16|int8|fp8|nvfp4|k8v4 (fp8)] [--device N] [--limit N]\n"
                 "Rows are the final-normalized BF16 hidden states of predictor positions "
                 "0..tokens-2.\n";
}

std::vector<Record> read_records(const std::string& path, std::uint32_t context, std::size_t limit) {
    std::ifstream in(path);
    if (!in) { throw std::runtime_error("cannot open " + path); }
    std::vector<Record> records;
    std::string line;
    std::size_t number = 0;
    while (std::getline(in, line)) {
        ++number;
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) { continue; }
        const json row = json::parse(line);
        Record record;
        record.id = row.at("id").get<std::string>();
        if (record.id.empty() || record.id.find('"') != std::string::npos) {
            throw std::runtime_error("line " + std::to_string(number) + ": invalid id");
        }
        record.tokens = row.at("tokens").get<std::vector<ninfer::TokenId>>();
        if (record.tokens.size() < 2 || record.tokens.size() > context) {
            throw std::runtime_error("line " + std::to_string(number) + " (" + record.id +
                                     "): token count must be in [2, --context]");
        }
        records.push_back(std::move(record));
        if (limit != 0 && records.size() == limit) { break; }
    }
    if (records.empty()) { throw std::runtime_error("no records in " + path); }
    return records;
}

void write_u64(std::ostream& out, std::uint64_t value) {
    unsigned char bytes[8];
    for (int i = 0; i < 8; ++i) { bytes[i] = static_cast<unsigned char>((value >> (8 * i)) & 0xFFU); }
    out.write(reinterpret_cast<const char*>(bytes), 8);
}

} // namespace

int main(int argc, char** argv) try {
    Settings settings;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto value                 = [&](const char* name) -> std::string {
            if (i + 1 >= argc) { throw std::invalid_argument(std::string(name) + " needs a value"); }
            return argv[++i];
        };
        if (arg == "--input") {
            settings.input = value("--input");
        } else if (arg == "--output") {
            settings.output = value("--output");
        } else if (arg == "--context") {
            settings.context = static_cast<std::uint32_t>(std::stoul(value("--context")));
        } else if (arg == "--prefill-chunk") {
            settings.prefill_chunk = static_cast<std::uint32_t>(std::stoul(value("--prefill-chunk")));
        } else if (arg == "--kv-dtype") {
            settings.kv_dtype = value("--kv-dtype");
        } else if (arg == "--device") {
            settings.device = std::stoi(value("--device"));
        } else if (arg == "--limit") {
            settings.limit = std::stoul(value("--limit"));
        } else if (arg == "--help" || arg == "-h") {
            usage();
            return 0;
        } else if (!arg.empty() && arg[0] == '-') {
            throw std::invalid_argument("unknown option " + std::string(arg));
        } else if (settings.artifact.empty()) {
            settings.artifact = arg;
        } else {
            throw std::invalid_argument("unexpected argument " + std::string(arg));
        }
    }
    if (settings.artifact.empty() || settings.input.empty() || settings.output.empty()) {
        usage();
        return 2;
    }
    const ninfer::KvCacheStorage kv = parse_kv_dtype(settings.kv_dtype);
    if (settings.prefill_chunk == 0 || settings.prefill_chunk % 128 != 0) {
        throw std::invalid_argument("--prefill-chunk must be a positive multiple of 128");
    }
    if (settings.context < 2) { throw std::invalid_argument("--context must be at least 2"); }
    const std::vector<Record> records = read_records(settings.input, settings.context, settings.limit);

    ninfer::EngineOptions options;
    options.artifact_path = settings.artifact;
    options.purpose       = ninfer::EnginePurpose::CausalScoring;
    options.device        = settings.device;
    options.max_context   = settings.context;
    options.prefill_chunk = settings.prefill_chunk;
    options.kv_cache      = kv;
    const auto load_started = std::chrono::steady_clock::now();
    ninfer::Engine engine(std::move(options));
    const ninfer::LoadSummary load = engine.load_summary();
    const auto& effective          = engine.options();
    std::cerr << "engine ready in "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - load_started).count()
              << " s | model " << load.model_name << " | prefill chunk " << effective.prefill_chunk
              << " | kv " << settings.kv_dtype << " | " << records.size() << " records\n";

    // The safetensors header needs every offset up front; record lengths are known from the input.
    ninfer::CausalScoreReadout readout;
    readout.capture_hidden_rows = true;
    std::uint32_t hidden_size   = 0;
    {
        const ninfer::CausalScores probe = engine.score_tokens({records.front().tokens[0], records.front().tokens[1]}, 1, readout);
        hidden_size                      = probe.hidden_size;
    }
    if (hidden_size == 0) { throw std::runtime_error("the artifact reported no hidden size"); }
    json header = json::object();
    std::uint64_t offset = 0;
    for (const Record& record : records) {
        const std::uint64_t rows = record.tokens.size() - 1;
        const std::uint64_t hidden_bytes = rows * hidden_size * 2;
        header["hidden/" + record.id] = {{"dtype", "BF16"}, {"shape", {rows, hidden_size}},
                                         {"data_offsets", {offset, offset + hidden_bytes}}};
        offset += hidden_bytes;
        const std::uint64_t logprob_bytes = rows * 4;
        header["logprobs/" + record.id] = {{"dtype", "F32"}, {"shape", {rows}},
                                           {"data_offsets", {offset, offset + logprob_bytes}}};
        offset += logprob_bytes;
    }
    header["__metadata__"] = {{"format", "ninfer-hidden-export"},
                              {"artifact", settings.artifact},
                              {"model_id", load.model_name},
                              {"hidden_size", std::to_string(hidden_size)},
                              {"rows", "final-normalized hidden state of predictor positions 0..tokens-2"},
                              {"prefill_chunk", std::to_string(effective.prefill_chunk)},
                              {"kv_dtype", settings.kv_dtype},
                              {"context", std::to_string(effective.max_context)},
                              {"records", std::to_string(records.size())}};
    std::string header_text = header.dump();
    while (header_text.size() % 8 != 0) { header_text.push_back(' '); }

    std::ofstream out(settings.output, std::ios::binary);
    if (!out) { throw std::runtime_error("cannot write " + settings.output); }
    write_u64(out, header_text.size());
    out.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));

    const auto started = std::chrono::steady_clock::now();
    std::uint64_t written = 0;
    for (std::size_t index = 0; index < records.size(); ++index) {
        const Record& record = records[index];
        const ninfer::CausalScores scores = engine.score_tokens(record.tokens, 1, readout);
        const std::size_t rows = record.tokens.size() - 1;
        if (scores.hidden_size != hidden_size || scores.hidden_rows.size() != rows * hidden_size ||
            scores.target_logprobs.size() != rows) {
            throw std::runtime_error("unexpected readout shape for " + record.id);
        }
        out.write(reinterpret_cast<const char*>(scores.hidden_rows.data()),
                  static_cast<std::streamsize>(scores.hidden_rows.size() * sizeof(std::uint16_t)));
        out.write(reinterpret_cast<const char*>(scores.target_logprobs.data()),
                  static_cast<std::streamsize>(scores.target_logprobs.size() * sizeof(float)));
        written += scores.hidden_rows.size() * 2 + scores.target_logprobs.size() * 4;
        if (!out) { throw std::runtime_error("write failed at " + record.id); }
        if (index % 25 == 0 || index + 1 == records.size()) {
            std::cerr << "exported " << index + 1 << "/" << records.size() << " ("
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
                      << " s)\n";
        }
    }
    if (written != offset) { throw std::runtime_error("data section size mismatch"); }
    out.close();
    std::cout << json{{"output", settings.output},
                      {"records", records.size()},
                      {"hidden_size", hidden_size},
                      {"bytes", 8 + header_text.size() + offset}}
                     .dump()
              << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "ninfer-hidden-export: " << error.what() << '\n';
    return 1;
}
