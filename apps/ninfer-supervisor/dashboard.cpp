#include "dashboard.hpp"
#include "dashboard_resource.h"

#include <windows.h>
#include <stdexcept>

namespace ninfer::supervisor {

std::string_view dashboard_html() {
    static const auto html = [] {
        const auto module = ::GetModuleHandleW(nullptr);
        const auto resource = ::FindResourceW(
            module, MAKEINTRESOURCEW(IDR_NINFER_DASHBOARD), MAKEINTRESOURCEW(10));
        if (!resource) throw std::runtime_error("Supervisor dashboard resource is missing");
        const auto loaded = ::LoadResource(module, resource);
        const auto bytes = ::SizeofResource(module, resource);
        const auto* data = loaded ? static_cast<const char*>(::LockResource(loaded)) : nullptr;
        if (!data || !bytes) throw std::runtime_error("Supervisor dashboard resource cannot load");
        return std::string_view(data, bytes);
    }();
    return html;
}

} // namespace ninfer::supervisor
