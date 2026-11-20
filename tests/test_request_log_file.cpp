#include "serve/request_log_file.h"

#include <chrono>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

using ninfer::serve::RequestLogFile;
using ninfer::serve::RequestLogWrite;
namespace fs = std::filesystem;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct TestLog {
    fs::path directory = fs::temp_directory_path() /
        ("ninfer-request-log-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::path path = directory / "requests.jsonl";
    fs::path staging = directory / "requests.jsonl.prune.tmp";

    TestLog() { require(fs::create_directory(directory), "cannot create temporary directory"); }

    ~TestLog() {
        std::error_code error;
        fs::remove(path, error);
        fs::remove(staging, error);
        fs::remove(directory, error);
    }

    void seed(std::string_view bytes) const {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output << bytes;
    }

    std::string read() const {
        std::ifstream input(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }
};

void test_prune_startup_and_live() {
    TestLog log;
    constexpr std::uint64_t cap = 256;
    const std::string startup = R"({"event":"startup_config"})";
    const std::string server  = R"({"event":"server_start"})";
    log.seed(std::string(2 * cap, 'x') + "\r\n{\"legacy\":1}\r\n{torn");
    RequestLogFile writer(log.path, cap);
    require(writer.append(startup, true) == RequestLogWrite::Written, "startup prune failed");
    require(log.read() == "{\"legacy\":1}\r\n" + startup + "\n",
            "startup prune lost the complete legacy tail or kept a partial line");
    require(writer.append(server, true) == RequestLogWrite::Written, "server metadata write failed");

    for (int i = 0; i < 30; ++i) {
        const std::string line = "{\"id\":" + std::to_string(i) + ",\"event\":\"request_done\"}";
        require(writer.append(line) == RequestLogWrite::Written, "live append or prune failed");
        require(fs::file_size(log.path) <= cap, "log exceeded its cap");
    }
    const std::string recent = log.read();
    require(recent.starts_with(startup + "\n" + server + "\n") &&
                recent.find(startup, startup.size()) == std::string::npos &&
                recent.find(server, startup.size() + server.size() + 1) == std::string::npos,
            "startup metadata was lost or duplicated across prunes");
    require(recent.find("\"id\":0,") == std::string::npos &&
                recent.ends_with("{\"id\":28,\"event\":\"request_done\"}\n"
                                 "{\"id\":29,\"event\":\"request_done\"}\n"),
            "pruning did not preserve the newest complete entries");
    require(writer.append(std::string(cap, 'x')) == RequestLogWrite::Oversized &&
                log.read() == recent && writer.append("{}") == RequestLogWrite::Written,
            "oversized record disrupted subsequent writes");
    require(!fs::exists(log.staging), "prune leaked its staging file");
}

void test_full_history() {
    TestLog log;
    log.seed("{\"legacy\":1}\n");
    {
        RequestLogFile first(log.path, 0);
        require(first.append("{\"run\":1}") == RequestLogWrite::Written, "first append failed");
    }
    {
        RequestLogFile second(log.path, 0);
        require(second.append("{\"run\":2}") == RequestLogWrite::Written, "second append failed");
    }
    std::ifstream input(log.path);
    std::string line;
    for (const auto expected : {"{\"legacy\":1}", "{\"run\":1}", "{\"run\":2}"}) {
        require(static_cast<bool>(std::getline(input, line)) && line == expected,
                "full history lost a record or changed record order");
    }
    require(!std::getline(input, line), "full history added an unexpected record");
}

void test_failed_prune_keeps_original() {
    TestLog log;
    const std::string original = std::string(256, 'x') + '\n';
    log.seed(original);
    fs::create_directory(log.staging);
    RequestLogFile writer(log.path, 128);
    require(writer.append("{}") == RequestLogWrite::Failed && log.read() == original &&
                fs::is_directory(log.staging),
            "prune failure changed the original log or the obstructing directory");
}

#ifdef _WIN32
void test_prune_replaces_file_with_shared_reader() {
    TestLog log;
    const std::string original = std::string(256, 'x') + "\n{}\n";
    log.seed(original);
    RequestLogFile writer(log.path, 128);
    const HANDLE reader = CreateFileW(log.path.c_str(), GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL, nullptr);
    require(reader != INVALID_HANDLE_VALUE, "failed to open with supervisor sharing flags");
    const auto result = writer.append("{\"new\":1}");
    std::string held(original.size(), '\0');
    DWORD received = 0;
    const bool read = ReadFile(reader, held.data(), static_cast<DWORD>(held.size()), &received,
                               nullptr) != FALSE;
    CloseHandle(reader);
    require(result == RequestLogWrite::Written && log.read() == "{}\n{\"new\":1}\n" &&
                read && received == held.size() && held == original,
            "prune failed while a supervisor reader held the original log");
}
#endif

} // namespace

int main() {
    try {
        test_prune_startup_and_live();
        test_full_history();
        test_failed_prune_keeps_original();
#ifdef _WIN32
        test_prune_replaces_file_with_shared_reader();
#endif
        std::cout << "request log retention tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
