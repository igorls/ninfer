// ExpertRouteRecorder: the .nxtr trace bytes, live counts and the bounded-writer contract.

#include "serve/expert_route_recorder.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

template <class T>
T read_at(const std::vector<char>& bytes, std::size_t offset) {
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return value;
}

} // namespace

int main() {
    int failures = 0;
    const auto directory =
        std::filesystem::temp_directory_path() /
        ("ninfer-expert-trace-test-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::path trace_file;
    {
        ninfer::serve::ExpertRouteRecorder recorder(directory);
        failures += check(recorder.snapshot(false).value("enabled", true) == false,
                          "a recorder without geometry reported itself enabled");
        recorder.expert_route_started(
            {.text_layers = 2, .mtp_layer = 2, .experts = 512, .top_k = 10});

        // A record before configure() counts live but is not written.
        std::array<ninfer::ExpertRouteColumn, 1> early{{{.request_tag = 1, .position = 0}}};
        std::vector<std::uint16_t> early_ids(2 * 10);
        for (std::size_t i = 0; i < early_ids.size(); ++i) {
            early_ids[i] = static_cast<std::uint16_t>(i % 10);
        }
        recorder.expert_route({.kind        = ninfer::ExpertRouteKind::Prefill,
                               .first_layer = 0,
                               .layer_count = 2,
                               .top_k       = 10,
                               .columns     = early,
                               .experts     = early_ids});

        recorder.configure("serve-test-1", "test-model", "C:/models/example.ninfer");

        // Two decode columns over both text layers: expert 7 is selected in every slot of layer 0.
        std::array<ninfer::ExpertRouteColumn, 2> columns{
            {{.request_tag = 42, .position = 100}, {.request_tag = 43, .position = 7}}};
        std::vector<std::uint16_t> ids(2 * 2 * 10);
        for (std::size_t column = 0; column < 2; ++column) {
            for (std::size_t layer = 0; layer < 2; ++layer) {
                for (std::size_t rank = 0; rank < 10; ++rank) {
                    ids[(column * 2 + layer) * 10 + rank] =
                        static_cast<std::uint16_t>(layer == 0 && rank == 0 ? 7 : 100 + rank);
                }
            }
        }
        recorder.expert_route({.kind        = ninfer::ExpertRouteKind::Decode,
                               .first_layer = 0,
                               .layer_count = 2,
                               .top_k       = 10,
                               .columns     = columns,
                               .experts     = ids});
        // One MTP proposal column on the MTP slice.
        std::array<ninfer::ExpertRouteColumn, 1> proposal{{{.request_tag = 42, .position = 101}}};
        std::vector<std::uint16_t> proposal_ids{511, 1, 2, 3, 4, 5, 6, 8, 9, 10};
        recorder.expert_route({.kind        = ninfer::ExpertRouteKind::MtpProposal,
                               .first_layer = 2,
                               .layer_count = 1,
                               .top_k       = 10,
                               .columns     = proposal,
                               .experts     = proposal_ids});
        // A malformed record (ids do not match the shape) is ignored.
        recorder.expert_route({.kind        = ninfer::ExpertRouteKind::Decode,
                               .first_layer = 0,
                               .layer_count = 2,
                               .top_k       = 10,
                               .columns     = columns,
                               .experts     = proposal_ids});

        const nlohmann::json live = recorder.snapshot(true);
        failures += check(live.at("enabled") == true, "a configured recorder reported disabled");
        failures += check(
            live.at("records").at("decode") == 1 && live.at("records").at("prefill") == 1 &&
                live.at("records").at("mtp_proposal") == 1 && live.at("columns").at("decode") == 2,
            "unit and column counts are wrong");
        const auto& decode = live.at("decode").at("total");
        failures += check(decode.size() == 3 && decode.at(0).at(7) == 2 &&
                              decode.at(1).at(100) == 2 && decode.at(2).at(511) == 1,
                          "decode totals did not count every slice and the MTP row");
        failures += check(live.at("prefill").at("total").at(0).at(0) == 1,
                          "prefill totals did not count the early record");
        failures += check(live.at("decode").at("experts_for_80_percent").size() == 3,
                          "per-slice coverage is missing");
        for (int attempt = 0; attempt < 100; ++attempt) {
            if (recorder.snapshot(false).at("trace").at("bytes_queued") == 0) { break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        trace_file = directory / "expert-trace-serve-test-1.nxtr";
    }

    std::ifstream input(trace_file, std::ios::binary);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                                  std::istreambuf_iterator<char>());
    failures += check(bytes.size() > 12 && std::memcmp(bytes.data(), "NXTRACE1", 8) == 0,
                      "the trace does not start with its magic");
    if (bytes.size() > 12) {
        const auto metadata_bytes = read_at<std::uint32_t>(bytes, 8);
        const nlohmann::json metadata =
            nlohmann::json::parse(std::string(bytes.data() + 12, metadata_bytes));
        failures += check(metadata.at("server_instance_id") == "serve-test-1" &&
                              metadata.at("text_layers") == 2 && metadata.at("mtp_layer") == 2 &&
                              metadata.at("experts") == 512 && metadata.at("top_k") == 10 &&
                              metadata.at("artifact") == "example.ninfer",
                          "trace metadata is wrong");
        std::size_t offset = 12 + metadata_bytes;
        // Decode record: header, two 16-byte columns, 2 x 2 x 10 ids.
        failures += check(read_at<std::uint32_t>(bytes, offset) == 0x43525452U &&
                              read_at<std::uint8_t>(bytes, offset + 4) == 2 &&
                              read_at<std::uint16_t>(bytes, offset + 6) == 2 &&
                              read_at<std::uint16_t>(bytes, offset + 8) == 0 &&
                              read_at<std::uint32_t>(bytes, offset + 12) == 2,
                          "decode record header is wrong");
        failures += check(read_at<std::uint64_t>(bytes, offset + 24) == 42 &&
                              read_at<std::uint32_t>(bytes, offset + 32) == 100 &&
                              read_at<std::uint64_t>(bytes, offset + 40) == 43 &&
                              read_at<std::uint32_t>(bytes, offset + 48) == 7,
                          "decode record columns are wrong");
        failures += check(read_at<std::uint16_t>(bytes, offset + 56) == 7 &&
                              read_at<std::uint16_t>(bytes, offset + 56 + 2 * 10) == 100,
                          "decode record ids are not [column][layer][rank]");
        offset += 24 + 2 * 16 + 2 * 2 * 10 * 2;
        failures += check(read_at<std::uint8_t>(bytes, offset + 4) == 3 &&
                              read_at<std::uint16_t>(bytes, offset + 6) == 1 &&
                              read_at<std::uint16_t>(bytes, offset + 8) == 2 &&
                              read_at<std::uint16_t>(bytes, offset + 24 + 16) == 511,
                          "MTP proposal record is wrong");
        offset += 24 + 16 + 10 * 2;
        failures += check(offset == bytes.size(),
                          "the trace holds records it should not (early or malformed)");
    }
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
    if (failures == 0) { std::cout << "PASS\n"; }
    return failures == 0 ? 0 : 1;
}
