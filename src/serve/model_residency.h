#pragma once

// Model residency for the HTTP server: the listening socket outlives the loaded model.
//
// The server keeps answering while the model is unloaded. A request that needs the model takes
// a use lease; the first lease after an unload runs the loader, and every lease that arrives
// while that load is running waits for the same load. A failed load fails the requests that
// waited for it and leaves the model unloaded, so the next request starts a new attempt. With a
// nonzero idle limit, a model that has had no use lease for that long is destroyed. Use leases
// hold it resident, so an unload never overlaps a request; status leases (health, metrics) only
// delay the destruction until the reads in progress finish, so polling cannot keep it loaded.
//
// The service type is a template parameter so the state machine can be qualified with a fake
// loader; serving instantiates it with GenerationService.

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace ninfer::serve {

enum class ResidencyState : std::uint8_t {
    Loading,   // the loader is running; leases wait for it
    Loaded,    // the service is attached and leases are granted
    Unloading, // the idle service is being destroyed; leases wait, then reload
    Unloaded,  // no service; the next use lease loads one
};

[[nodiscard]] constexpr const char* residency_state_name(ResidencyState state) noexcept {
    switch (state) {
    case ResidencyState::Loading:
        return "loading";
    case ResidencyState::Loaded:
        return "loaded";
    case ResidencyState::Unloading:
        return "unloading";
    case ResidencyState::Unloaded:
        return "unloaded";
    }
    return "unknown";
}

// Why a load ran: the server's startup load, or a request that found the model unloaded.
enum class LoadCause : std::uint8_t { Startup, Request };

struct ResidencyStatus {
    ResidencyState state         = ResidencyState::Unloaded;
    std::uint32_t idle_unload_ms = 0; // 0 keeps the model loaded
    std::uint64_t loads          = 0; // completed loads, the startup load included
    std::uint64_t unloads        = 0;
    int leases                   = 0; // use and status leases currently held
    double last_load_seconds     = 0.0;
    double last_unload_seconds   = 0.0;
    std::string last_load_error; // the most recent failed load; cleared by a successful one
};

// Raised to a request whose load failed, or that arrived while the server is stopping.
class ModelUnavailable final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

template <class Service> class ModelResidency {
public:
    using Clock  = std::chrono::steady_clock;
    using Loader = std::function<std::unique_ptr<Service>(LoadCause)>;

    // Called without the residency lock. attach runs after a load and before any lease can see
    // the service; detach runs after the last lease is gone and before the service is destroyed.
    struct Hooks {
        std::function<void(Service&, LoadCause)> attach;
        std::function<void(Service&)> detach;
        std::function<void(LoadCause)> load_started;
        std::function<void(LoadCause, double seconds)> loaded;
        std::function<void(LoadCause, double seconds, const std::string& error)> load_failed;
        std::function<void(double idle_seconds, double unload_seconds)> unloaded;
    };

    // Holds the service resident. A use lease restarts the idle clock when it is released; a
    // status lease only reads the service and leaves the clock alone.
    class Lease {
    public:
        Lease() = default;
        Lease(const Lease&)            = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept { *this = std::move(other); }
        Lease& operator=(Lease&& other) noexcept {
            if (this != &other) {
                release();
                owner_   = std::exchange(other.owner_, nullptr);
                service_ = std::exchange(other.service_, nullptr);
                use_     = other.use_;
            }
            return *this;
        }
        ~Lease() { release(); }

        [[nodiscard]] explicit operator bool() const noexcept { return service_ != nullptr; }
        [[nodiscard]] Service& operator*() const noexcept { return *service_; }
        [[nodiscard]] Service* operator->() const noexcept { return service_; }

    private:
        friend class ModelResidency;
        Lease(ModelResidency* owner, Service* service, bool use) noexcept
            : owner_(owner), service_(service), use_(use) {}

        void release() noexcept {
            if (owner_ != nullptr) owner_->release(use_);
            owner_   = nullptr;
            service_ = nullptr;
        }

        ModelResidency* owner_ = nullptr;
        Service* service_      = nullptr;
        bool use_              = false;
    };

    ModelResidency(Loader loader, Hooks hooks, std::uint32_t idle_unload_ms)
        : loader_(std::move(loader)), hooks_(std::move(hooks)), idle_unload_ms_(idle_unload_ms) {}

    ModelResidency(const ModelResidency&)            = delete;
    ModelResidency& operator=(const ModelResidency&) = delete;

    ~ModelResidency() {
        stop();
        if (idle_thread_.joinable()) idle_thread_.join();
        // HTTP workers have finished by now; the service goes without a detach hook because
        // its observers are being destroyed with the server.
        std::unique_ptr<Service> service;
        {
            std::lock_guard lock(mutex_);
            service = std::move(service_);
        }
    }

    // Startup load on the calling thread. A failure propagates and leaves the model unloaded.
    void load_at_startup() {
        std::unique_lock lock(mutex_);
        if (state_ != ResidencyState::Unloaded || service_)
            throw std::logic_error("model residency is already loaded");
        run_load(lock, LoadCause::Startup);
        if (state_ != ResidencyState::Loaded) std::rethrow_exception(last_failure_);
        if (!idle_thread_.joinable()) idle_thread_ = std::thread([this] { run_idle_unloader(); });
    }

    // Blocks until the model is resident, loading it when it is not, and returns a use lease.
    // Throws ModelUnavailable when the load this request waited for failed or the server stops.
    [[nodiscard]] Lease acquire() {
        std::unique_lock lock(mutex_);
        for (;;) {
            if (stopping_) throw ModelUnavailable("the server is stopping");
            switch (state_) {
            case ResidencyState::Loaded:
                ++leases_;
                ++use_leases_;
                return Lease(this, service_.get(), true);
            case ResidencyState::Unloaded:
                run_load(lock, LoadCause::Request);
                if (state_ != ResidencyState::Loaded) throw_load_failure();
                continue;
            case ResidencyState::Loading: {
                // Wait for this attempt only: a failure fails its waiters even when another
                // request has already started the next attempt.
                const std::uint64_t attempt = load_attempt_;
                changed_.wait(lock, [&] {
                    return stopping_ || load_attempt_ != attempt ||
                           state_ != ResidencyState::Loading;
                });
                if (!stopping_ && failed_attempt_ == attempt) throw_load_failure();
                continue;
            }
            case ResidencyState::Unloading:
                changed_.wait(lock, [&] {
                    return stopping_ || state_ != ResidencyState::Unloading;
                });
                continue;
            }
        }
    }

    // A status lease when the model is resident right now, otherwise empty. Never loads or waits,
    // and still answers while stopping so shutdown can read final statistics.
    [[nodiscard]] Lease peek() {
        std::lock_guard lock(mutex_);
        if (state_ != ResidencyState::Loaded) return {};
        ++leases_;
        return Lease(this, service_.get(), false);
    }

    [[nodiscard]] ResidencyStatus status() const {
        std::lock_guard lock(mutex_);
        return ResidencyStatus{.state               = state_,
                               .idle_unload_ms      = idle_unload_ms_,
                               .loads               = loads_,
                               .unloads             = unloads_,
                               .leases              = leases_,
                               .last_load_seconds   = last_load_seconds_,
                               .last_unload_seconds = last_unload_seconds_,
                               .last_load_error     = last_load_error_};
    }

    // Takes effect immediately; the idle clock keeps counting from the last released use lease.
    void set_idle_unload_ms(std::uint32_t milliseconds) {
        {
            std::lock_guard lock(mutex_);
            idle_unload_ms_ = milliseconds;
        }
        changed_.notify_all();
    }

    // Wakes waiting requests with ModelUnavailable and ends the idle unloader. A resident service
    // stays attached so in-flight handlers can finish; `on_resident` lets the caller wake them.
    void stop(const std::function<void(Service&)>& on_resident = {}) {
        Service* resident = nullptr;
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
            if (state_ == ResidencyState::Loaded) resident = service_.get();
        }
        changed_.notify_all();
        if (resident != nullptr && on_resident) on_resident(*resident);
    }

private:
    // Runs the loader with the lock released. The caller observes the outcome in state_.
    void run_load(std::unique_lock<std::mutex>& lock, LoadCause cause) {
        state_                      = ResidencyState::Loading;
        const std::uint64_t attempt = ++load_attempt_;
        lock.unlock();
        if (hooks_.load_started) hooks_.load_started(cause);
        const Clock::time_point started = Clock::now();
        std::unique_ptr<Service> service;
        std::exception_ptr failure;
        std::string error;
        try {
            service = loader_(cause);
            if (!service) throw std::runtime_error("the model loader returned no service");
            if (hooks_.attach) hooks_.attach(*service, cause);
        } catch (const std::exception& exception) {
            failure = std::current_exception();
            error   = exception.what();
        } catch (...) {
            failure = std::current_exception();
            error   = "unknown error";
        }
        if (failure && service && hooks_.detach) {
            try {
                hooks_.detach(*service);
            } catch (...) {
            }
        }
        if (failure) service.reset();
        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        if (failure) {
            if (hooks_.load_failed) hooks_.load_failed(cause, seconds, error);
        } else if (hooks_.loaded) {
            hooks_.loaded(cause, seconds);
        }
        lock.lock();
        if (failure) {
            state_           = ResidencyState::Unloaded;
            failed_attempt_  = attempt;
            last_failure_    = failure;
            failed_error_    = error;
            last_load_error_ = error;
        } else {
            service_           = std::move(service);
            state_             = ResidencyState::Loaded;
            last_use_          = Clock::now();
            last_load_seconds_ = seconds;
            last_load_error_.clear();
            ++loads_;
        }
        changed_.notify_all();
    }

    [[noreturn]] void throw_load_failure() const {
        throw ModelUnavailable("the model failed to load: " + failed_error_);
    }

    void release(bool use) noexcept {
        {
            std::lock_guard lock(mutex_);
            --leases_;
            if (use) {
                --use_leases_;
                last_use_ = Clock::now();
            }
        }
        changed_.notify_all();
    }

    void run_idle_unloader() {
        std::unique_lock lock(mutex_);
        while (!stopping_) {
            const bool idle = state_ == ResidencyState::Loaded && use_leases_ == 0 &&
                              idle_unload_ms_ != 0;
            if (!idle) {
                changed_.wait(lock);
                continue;
            }
            const Clock::time_point due =
                last_use_ + std::chrono::milliseconds(idle_unload_ms_);
            if (Clock::now() < due) {
                changed_.wait_until(lock, due);
                continue;
            }
            const double idle_seconds =
                std::chrono::duration<double>(Clock::now() - last_use_).count();
            // No lease is granted from here on; status reads already in progress finish first.
            state_ = ResidencyState::Unloading;
            changed_.wait(lock, [&] { return leases_ == 0; });
            std::unique_ptr<Service> service = std::move(service_);
            lock.unlock();
            const Clock::time_point started = Clock::now();
            if (hooks_.detach) {
                try {
                    hooks_.detach(*service);
                } catch (...) {
                }
            }
            service.reset();
            const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
            if (hooks_.unloaded) hooks_.unloaded(idle_seconds, seconds);
            lock.lock();
            state_               = ResidencyState::Unloaded;
            last_unload_seconds_ = seconds;
            ++unloads_;
            changed_.notify_all();
        }
    }

    Loader loader_;
    Hooks hooks_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::unique_ptr<Service> service_;
    ResidencyState state_        = ResidencyState::Unloaded;
    std::uint32_t idle_unload_ms_ = 0;
    int leases_                   = 0; // use and status leases
    int use_leases_               = 0;
    bool stopping_                = false;
    Clock::time_point last_use_{};
    std::uint64_t load_attempt_   = 0;
    std::uint64_t failed_attempt_ = 0;
    std::exception_ptr last_failure_;
    std::string failed_error_;    // error of failed_attempt_, kept for its late waiters
    std::string last_load_error_; // cleared by the next successful load
    std::uint64_t loads_         = 0;
    std::uint64_t unloads_       = 0;
    double last_load_seconds_    = 0.0;
    double last_unload_seconds_  = 0.0;
    std::thread idle_thread_;
};

} // namespace ninfer::serve
