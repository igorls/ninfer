#pragma once

#include "logic.hpp"

#include <windows.h>
#include <nvml.h>

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>

namespace ninfer::supervisor {

// Device-wide memory through one NVML session held for the supervisor's lifetime.
// nvidia-smi was spawned once per second before: ~51 ms of process start per sample, a
// window-station-dependent child that fails with 0xc0000142 during logoff, and one new NVML
// session per call. Opening NVML once and reusing the device handle costs a few microseconds
// per sample. nvml.dll is resolved from System32 at runtime, so the binary keeps no link
// dependency on it; when it cannot load, the caller falls back to nvidia-smi.
class NvmlMemory {
public:
    NvmlMemory() = default;
    NvmlMemory(const NvmlMemory&)            = delete;
    NvmlMemory& operator=(const NvmlMemory&) = delete;

    // ok=false with `error` set when NVML is unavailable or the device cannot be read.
    [[nodiscard]] NvidiaMemory query(int device) {
        NvidiaMemory out;
        std::lock_guard lock(mu_);
        if (!ensure_session(out.error)) { return out; }
        nvmlDevice_t handle = nullptr;
        if (auto it = handles_.find(device); it != handles_.end()) {
            handle = it->second;
        } else {
            const nvmlReturn_t rc = get_handle_(static_cast<unsigned>(device), &handle);
            if (rc != NVML_SUCCESS) {
                out.error = "nvmlDeviceGetHandleByIndex: " + error_text(rc);
                return out;
            }
            handles_.emplace(device, handle);
        }
        nvmlMemory_t memory{};
        const nvmlReturn_t rc = get_memory_(handle, &memory);
        if (rc != NVML_SUCCESS) {
            out.error = "nvmlDeviceGetMemoryInfo: " + error_text(rc);
            return out;
        }
        out.ok        = true;
        out.index     = device;
        out.used_mib  = memory.used / (1024ull * 1024ull);
        out.total_mib = memory.total / (1024ull * 1024ull);
        return out;
    }

    [[nodiscard]] bool available() {
        std::lock_guard lock(mu_);
        std::string ignored;
        return ensure_session(ignored);
    }

private:
    using FnInit      = nvmlReturn_t (*)();
    using FnHandle    = nvmlReturn_t (*)(unsigned int, nvmlDevice_t*);
    using FnMemory    = nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_t*);
    using FnErrorText = const char* (*)(nvmlReturn_t);

    static constexpr auto kRetry = std::chrono::minutes(10); // every attempt opens a session

    bool ensure_session(std::string& error) {
        if (initialized_) { return true; }
        const auto now = std::chrono::steady_clock::now();
        if (attempted_ && now - failed_at_ < kRetry) {
            error = failure_;
            return false;
        }
        attempted_ = true;
        failed_at_ = now;
        if (library_ == nullptr) {
            library_ = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        }
        if (library_ == nullptr) {
            failure_ = "nvml.dll not found";
            error    = failure_;
            return false;
        }
        const auto init = reinterpret_cast<FnInit>(GetProcAddress(library_, "nvmlInit_v2"));
        get_handle_ = reinterpret_cast<FnHandle>(GetProcAddress(library_, "nvmlDeviceGetHandleByIndex_v2"));
        get_memory_ = reinterpret_cast<FnMemory>(GetProcAddress(library_, "nvmlDeviceGetMemoryInfo"));
        error_text_ = reinterpret_cast<FnErrorText>(GetProcAddress(library_, "nvmlErrorString"));
        if (init == nullptr || get_handle_ == nullptr || get_memory_ == nullptr) {
            failure_ = "nvml.dll lacks the required entry points";
            error    = failure_;
            return false;
        }
        const nvmlReturn_t rc = init();
        if (rc != NVML_SUCCESS) {
            failure_ = "nvmlInit: " + error_text(rc);
            error    = failure_;
            return false;
        }
        initialized_ = true; // never shut down: the session lives as long as the supervisor
        return true;
    }

    std::string error_text(nvmlReturn_t rc) const {
        if (error_text_ != nullptr) {
            if (const char* text = error_text_(rc)) { return text; }
        }
        return "NVML error " + std::to_string(static_cast<int>(rc));
    }

    std::mutex mu_;
    HMODULE library_ = nullptr;
    FnHandle get_handle_     = nullptr;
    FnMemory get_memory_     = nullptr;
    FnErrorText error_text_  = nullptr;
    bool initialized_        = false;
    bool attempted_          = false;
    std::chrono::steady_clock::time_point failed_at_{};
    std::string failure_;
    std::unordered_map<int, nvmlDevice_t> handles_;
};

} // namespace ninfer::supervisor
