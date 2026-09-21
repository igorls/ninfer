#include <winsock2.h>
#include <windows.h>
#include <commdlg.h>
#include <shellapi.h>
#include <shlobj.h>
#include <winhttp.h>
#include <cuda.h>
#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {
std::string utf8(const std::wstring& value) {
    if (value.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, value.data(), int(value.size()), nullptr, 0, nullptr, nullptr);
    std::string out(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, value.data(), int(value.size()), out.data(), n, nullptr, nullptr);
    return out;
}
std::wstring wide(const std::string& value) {
    if (value.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), int(value.size()), nullptr, 0);
    if (!n) throw std::runtime_error("Invalid UTF-8 text.");
    std::wstring out(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, value.data(), int(value.size()), out.data(), n);
    return out;
}
std::string path_text(const fs::path& p) { return utf8(p.wstring()); }
fs::path executable_dir() {
    std::array<wchar_t, 32768> path{};
    DWORD n = GetModuleFileNameW(nullptr, path.data(), DWORD(path.size()));
    if (!n || n >= path.size()) throw std::runtime_error("Cannot locate NInfer.");
    return fs::path(std::wstring(path.data(), n)).parent_path();
}
fs::path default_data_dir() {
    PWSTR path = nullptr;
    if (FAILED(SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, nullptr, &path)))
        throw std::runtime_error("Cannot locate your application data folder.");
    fs::path result = fs::path(path) / L"NInfer";
    CoTaskMemFree(path);
    return result;
}
void print(const std::string& value) {
    DWORD written = 0;
    auto handle = GetStdHandle(STD_OUTPUT_HANDLE);
    if (handle && handle != INVALID_HANDLE_VALUE)
        WriteFile(handle, value.data(), DWORD(value.size()), &written, nullptr);
}
struct Gpu { std::string name; std::size_t bytes = 0; int driver = 0; };
Gpu check_gpu() {
    HMODULE module = LoadLibraryExW(L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!module) throw std::runtime_error("Install a current NVIDIA driver before running NInfer.");
    struct Unload { HMODULE h; ~Unload() { FreeLibrary(h); } } unload{module};
#define CUDA_FUNCTION(name) auto name##_fn = reinterpret_cast<decltype(&name)>(GetProcAddress(module, #name)); \
    if (!name##_fn) throw std::runtime_error("The NVIDIA driver is missing a required CUDA function.")
    CUDA_FUNCTION(cuInit);
    CUDA_FUNCTION(cuDeviceGet);
    CUDA_FUNCTION(cuDeviceGetName);
    CUDA_FUNCTION(cuDeviceGetAttribute);
    CUDA_FUNCTION(cuDriverGetVersion);
    auto memory_fn = reinterpret_cast<decltype(&cuDeviceTotalMem)>(GetProcAddress(module, "cuDeviceTotalMem_v2"));
    if (!memory_fn) throw std::runtime_error("Update your NVIDIA driver.");
    CUdevice device{};
    Gpu gpu;
    int major = 0, minor = 0;
    char name[256]{};
    if (cuInit_fn(0) != CUDA_SUCCESS || cuDeviceGet_fn(&device, 0) != CUDA_SUCCESS ||
        cuDeviceGetName_fn(name, sizeof(name), device) != CUDA_SUCCESS ||
        memory_fn(&gpu.bytes, device) != CUDA_SUCCESS ||
        cuDeviceGetAttribute_fn(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) != CUDA_SUCCESS ||
        cuDeviceGetAttribute_fn(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) != CUDA_SUCCESS ||
        cuDriverGetVersion_fn(&gpu.driver) != CUDA_SUCCESS)
        throw std::runtime_error("Cannot initialize the NVIDIA GPU. Check the driver installation.");
    if (major != 12 || minor != 0)
        throw std::runtime_error("This build requires an sm_120 Blackwell GPU, such as RTX 5090 or RTX PRO 6000 Blackwell. RTX 3090 is not supported.");
    if (gpu.driver < 13030)
        throw std::runtime_error("This preview requires an NVIDIA driver supporting CUDA 13.3 or newer. Update the driver and restart Windows.");
    gpu.name = name;
    return gpu;
}

std::string model_identity(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    unsigned char header[16]{};
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!in || std::memcmp(header, "NINFER\0\2", 8))
        throw std::runtime_error("Choose an NInfer v2 .ninfer artifact. GGUF and safetensors files cannot be loaded.");
    std::uint64_t size = 0;
    for (unsigned i = 0; i != 8; ++i) size |= std::uint64_t(header[8 + i]) << (8 * i);
    if (size == 0 || size > 16 * 1024 * 1024 || fs::file_size(path) <= size + 16)
        throw std::runtime_error("The model file is incomplete or has an invalid directory.");
    std::string directory(size, '\0');
    in.read(directory.data(), std::streamsize(size));
    if (!in) throw std::runtime_error("The model directory could not be read.");
    auto identity = Json::parse(directory).at("identity");
    auto model = identity.at("model_id").get<std::string>();
    if (model != "qwen3.8-27b" && model != "qwen3.6-27b" &&
        model != "qwen3.6-35b-a3b" && model != "qwen3.8-flash-next")
        throw std::runtime_error("This model identity is not registered in this build.");
    return model;
}

void check_port(int port) {
    SOCKET socket = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (socket == INVALID_SOCKET) throw std::runtime_error("Cannot check local network ports.");
    BOOL exclusive = TRUE;
    setsockopt(socket, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, reinterpret_cast<const char*>(&exclusive), sizeof(exclusive));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(static_cast<u_short>(port));
    int status = bind(socket, reinterpret_cast<sockaddr*>(&address), sizeof(address));
    closesocket(socket);
    if (status) throw std::runtime_error("Local port " + std::to_string(port) + " is already in use. Close the other server before setting up NInfer.");
}

Json create_config(const fs::path& model_path, const fs::path& bin, const fs::path& data,
                   const Gpu& gpu, int engine_port, int dashboard_port) {
    const std::string model = model_identity(model_path);
    const bool flash = model == "qwen3.8-flash-next";
    if (flash && gpu.bytes < 90ULL * 1024 * 1024 * 1024)
        throw std::runtime_error("Flash-Next needs the 96 GB workstation configuration and substantial system RAM. Choose a 27B model on RTX 5090.");
    auto serve = bin / L"ninfer-serve.exe";
    if (!fs::exists(serve)) throw std::runtime_error("NInfer engine is missing. Reinstall NInfer.");
    std::vector<std::string> args{
        "--host", "127.0.0.1", "--port", std::to_string(engine_port),
        "--model-id", model, "--max-context", "16384", "--kv-capacity", "32768",
        "--max-concurrency", "2", "--prefill-chunk", "2048", "--desktop-reserve-gib", "3",
        "--kv-dtype", "fp8", "--spec", "mtp", "--draft-tokens", "3",
        "--request-log-jsonl", path_text(data / L"logs/requests.jsonl")};
    if (flash) { args.insert(args.end(), {"--gdn-state-dtype", "bf16"}); }
    auto engine_args = args;
    engine_args.insert(engine_args.begin(), path_text(fs::absolute(model_path)));
    return {
        {"engine", {{"executable", path_text(serve)}, {"args", engine_args},
            {"workdir", path_text(data)}, {"engine_host", "127.0.0.1"},
            {"engine_port", engine_port}, {"device", 0},
            {"request_log", path_text(data / L"logs/requests.jsonl")}}},
        {"supervisor", {{"host", "127.0.0.1"}, {"port", dashboard_port},
            {"logs_dir", path_text(data / L"logs")}, {"run_at_login", false}}},
        {"models", Json::array({{{"id", model}, {"artifact", path_text(fs::absolute(model_path))}, {"args", args}}})},
        {"active_model", model}};
}

void write_new_config(const fs::path& file, const Json& config) {
    fs::create_directories(file.parent_path() / L"logs");
    auto temp = file;
    temp += L".setup-" + std::to_wstring(GetCurrentProcessId());
    {
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        out << config.dump(2) << '\n';
        if (!out) throw std::runtime_error("Cannot save the NInfer configuration.");
    }
    // MoveFile without REPLACE_EXISTING preserves a configuration created by another launch.
    if (!MoveFileExW(temp.c_str(), file.c_str(), MOVEFILE_WRITE_THROUGH)) {
        DeleteFileW(temp.c_str());
        throw std::runtime_error("A configuration already exists, or its folder is not writable. Existing settings were preserved.");
    }
}

struct Setup {
    fs::path bin, data, model;
    Gpu gpu;
    int engine_port = 8010, dashboard_port = 8099;
};
INT_PTR CALLBACK dialog_proc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam) {
    auto* setup = reinterpret_cast<Setup*>(GetWindowLongPtrW(hwnd, DWLP_USER));
    if (message == WM_INITDIALOG) {
        setup = reinterpret_cast<Setup*>(lparam);
        SetWindowLongPtrW(hwnd, DWLP_USER, lparam);
        auto gpu = wide(setup->gpu.name + " | " + std::to_string(setup->gpu.bytes / (1024ULL * 1024 * 1024)) + " GiB VRAM");
        SetDlgItemTextW(hwnd, 201, gpu.c_str());
        SendMessageW(hwnd, WM_SETICON, ICON_SMALL, reinterpret_cast<LPARAM>(LoadIconW(GetModuleHandleW(nullptr), MAKEINTRESOURCEW(101))));
        EnableWindow(GetDlgItem(hwnd, IDOK), FALSE);
        return TRUE;
    }
    if (message != WM_COMMAND || !setup) return FALSE;
    switch (LOWORD(wparam)) {
    case IDCANCEL: EndDialog(hwnd, IDCANCEL); return TRUE;
    case 204:
    case 205:
        ShellExecuteW(hwnd, L"open", LOWORD(wparam) == 204
            ? L"https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer"
            : L"https://huggingface.co/igorls/Qwen3.8-Flash-Next-mixed-NInfer", nullptr, nullptr, SW_SHOWNORMAL);
        return TRUE;
    case 203: {
        std::array<wchar_t, 32768> path{};
        OPENFILENAMEW ofn{sizeof(ofn)};
        ofn.hwndOwner = hwnd;
        ofn.lpstrFilter = L"NInfer models (*.ninfer)\0*.ninfer\0\0";
        ofn.lpstrFile = path.data();
        ofn.nMaxFile = DWORD(path.size());
        ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
        if (GetOpenFileNameW(&ofn)) {
            setup->model = path.data();
            SetDlgItemTextW(hwnd, 202, path.data());
            EnableWindow(GetDlgItem(hwnd, IDOK), TRUE);
        }
        return TRUE;
    }
    case IDOK:
        try {
            auto config = create_config(setup->model, setup->bin, setup->data, setup->gpu,
                                        setup->engine_port, setup->dashboard_port);
            check_port(setup->engine_port);
            check_port(setup->dashboard_port);
            write_new_config(setup->data / L"supervisor.json", config);
            EndDialog(hwnd, IDOK);
        } catch (const std::exception& e) {
            MessageBoxW(hwnd, wide(e.what()).c_str(), L"NInfer setup", MB_OK | MB_ICONWARNING);
        }
        return TRUE;
    }
    return FALSE;
}

void launch(const Setup& setup) {
    auto file = setup.data / L"supervisor.json";
    std::ifstream in(file);
    const auto config = Json::parse(in);
    const int port = config.at("supervisor").value("port", 8099);
    auto executable = setup.bin / L"ninfer-supervisor.exe";
    auto cmd = L"\"" + executable.wstring() + L"\" --config \"" + file.wstring() + L"\" --background";
    STARTUPINFOW startup{sizeof(startup)};
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(executable.c_str(), cmd.data(), nullptr, nullptr, FALSE,
                        CREATE_NO_WINDOW, nullptr, setup.data.c_str(), &startup, &process))
        throw std::runtime_error("Cannot start the NInfer Supervisor. Reinstall the application if files are missing.");
    CloseHandle(process.hThread);
    WaitForSingleObject(process.hProcess, 5000);
    DWORD code = 0;
    GetExitCodeProcess(process.hProcess, &code);
    CloseHandle(process.hProcess);
    if (code != 0 && code != STILL_ACTIVE)
        throw std::runtime_error("NInfer could not start. Check the logs folder in your NInfer application data.");
    // Wait for the dashboard HTTP listener, independently of model loading.
    HINTERNET session = WinHttpOpen(L"NInfer Setup", WINHTTP_ACCESS_TYPE_NO_PROXY, nullptr, nullptr, 0);
    if (session) {
        WinHttpSetTimeouts(session, 250, 250, 250, 250);
        HINTERNET connection = WinHttpConnect(session, L"127.0.0.1", INTERNET_PORT(port), 0);
        for (int i = 0; connection && i < 30; ++i) {
            HINTERNET request = WinHttpOpenRequest(connection, L"GET", L"/api/state", nullptr, nullptr, nullptr, 0);
            bool ready = request && WinHttpSendRequest(request, nullptr, 0, nullptr, 0, 0, 0) && WinHttpReceiveResponse(request, nullptr);
            if (request) WinHttpCloseHandle(request);
            if (ready) break;
            Sleep(200);
        }
        if (connection) WinHttpCloseHandle(connection);
        WinHttpCloseHandle(session);
    }
    auto url = L"http://127.0.0.1:" + std::to_wstring(port) + L"/";
    ShellExecuteW(nullptr, L"open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}
} // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int) {
    bool quiet = false;
    try {
        Setup setup{executable_dir(), default_data_dir()};
        bool no_start = false, probe = false;
        int argc = 0;
        auto argv = CommandLineToArgvW(GetCommandLineW(), &argc);
        for (int i = 1; i < argc; ++i) {
            std::wstring arg = argv[i];
            auto value = [&]() -> std::wstring {
                if (++i >= argc) throw std::runtime_error("Missing command-line value.");
                return argv[i];
            };
            if (arg == L"--no-start") { no_start = true; quiet = true; }
            else if (arg == L"--probe") { probe = true; quiet = true; }
            else if (arg == L"--model") { setup.model = value(); quiet = true; }
            else if (arg == L"--data-dir") setup.data = fs::absolute(value());
            else if (arg == L"--engine-port") setup.engine_port = std::stoi(value());
            else if (arg == L"--dashboard-port") setup.dashboard_port = std::stoi(value());
            else if (arg == L"--help") {
                print("NInfer launcher [--data-dir DIR] [--model FILE --no-start] [--probe]\n"
                      "Optional --engine-port and --dashboard-port select initial local ports.\n");
                LocalFree(argv);
                return 0;
            } else throw std::runtime_error("Unknown launcher argument.");
        }
        LocalFree(argv);
        if (setup.engine_port < 1024 || setup.engine_port > 65535 ||
            setup.dashboard_port < 1024 || setup.dashboard_port > 65535 || setup.engine_port == setup.dashboard_port)
            throw std::runtime_error("Choose distinct local ports between 1024 and 65535.");
        setup.gpu = check_gpu();
        if (probe) {
            print(Json{{"gpu", setup.gpu.name}, {"memory_bytes", setup.gpu.bytes},
                       {"cuda_driver", setup.gpu.driver}, {"supported", true}}.dump() + "\n");
            return 0;
        }
        WSADATA sockets{};
        if (WSAStartup(MAKEWORD(2, 2), &sockets)) throw std::runtime_error("Cannot initialize local networking.");
        struct Cleanup { ~Cleanup() { WSACleanup(); } } cleanup;
        CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
        if (!fs::exists(setup.data / L"supervisor.json")) {
            if (!setup.model.empty()) {
                auto config = create_config(setup.model, setup.bin, setup.data, setup.gpu,
                                            setup.engine_port, setup.dashboard_port);
                check_port(setup.engine_port);
                check_port(setup.dashboard_port);
                write_new_config(setup.data / L"supervisor.json", config);
            } else {
                if (no_start) throw std::runtime_error("Choose a model with --model FILE.");
                auto result = DialogBoxParamW(instance, MAKEINTRESOURCEW(200), nullptr, dialog_proc,
                                              reinterpret_cast<LPARAM>(&setup));
                if (result == -1) throw std::runtime_error("Cannot open the model setup window.");
                if (result != IDOK) return 0;
            }
        } else if (!setup.model.empty()) {
            throw std::runtime_error("NInfer is already configured. Existing settings were preserved.");
        }
        if (!no_start) launch(setup);
        CoUninitialize();
        return 0;
    } catch (const std::exception& e) {
        print(std::string(e.what()) + "\n");
        if (!quiet) MessageBoxW(nullptr, wide(e.what()).c_str(), L"NInfer", MB_OK | MB_ICONERROR);
        return 1;
    }
}
