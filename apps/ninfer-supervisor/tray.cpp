#include "tray.hpp"

#include <windows.h>
#include <shellapi.h>

#include <array>
#include <cstdio>
#include <string>

namespace ninfer::supervisor {
namespace {

constexpr UINT kTrayMsg    = WM_APP + 1;
constexpr UINT kIdOpen     = 1;
constexpr UINT kIdStart    = 2;
constexpr UINT kIdStop     = 3;
constexpr UINT kIdRestart  = 4;
constexpr UINT kIdQuit     = 5;
constexpr UINT kIdReserveBase  = 100;  // + GiB
constexpr UINT kIdIdleBase     = 200;  // + minutes
constexpr UINT kIdReleaseCache = 300;

// The reserve choices. The floor is measured rather than chosen: on this card a
// desktop starts missing its 16.7 ms frame budget below ~1.7 GiB free and shows
// nothing measurable above ~3 GiB. So 2 GiB is the smallest honest option, and 4
// is the one to recommend -- twice the measured cliff, leaving room for a real
// desktop's several applications rather than the one synthetic canary that
// produced the number.
constexpr std::array<int, 6> kReserveChoices{2, 3, 4, 6, 8, 12};

// A duration, not a toggle. Unloading after five minutes would punish anyone who
// steps away mid-task; an hour reliably catches "finished for the day".
constexpr std::array<int, 4> kIdleChoices{0, 15, 60, 240};
constexpr UINT_PTR kTimer  = 1;
constexpr UINT kTrayUid    = 1;
constexpr wchar_t kClass[] = L"NInferSupervisorTray";

COLORREF status_fill(TrayStatus status) {
    switch (status) {
    case TrayStatus::Working: return RGB(0x2F, 0x9E, 0x54);
    case TrayStatus::Pending: return RGB(0xD1, 0x8B, 0x12);
    case TrayStatus::Failed: return RGB(0xC5, 0x3B, 0x33);
    case TrayStatus::Idle: break;
    }
    return RGB(0x6B, 0x72, 0x80);
}

const wchar_t* status_word(TrayStatus status) {
    switch (status) {
    case TrayStatus::Working: return L"running";
    case TrayStatus::Pending: return L"pending";
    case TrayStatus::Failed: return L"failed";
    case TrayStatus::Idle: break;
    }
    return L"idle";
}


std::wstring widen(const std::string& in) {
    if (in.empty()) { return {}; }
    const int n = MultiByteToWideChar(CP_UTF8, 0, in.c_str(), static_cast<int>(in.size()), nullptr, 0);
    std::wstring out(static_cast<std::size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, in.c_str(), static_cast<int>(in.size()), out.data(), n);
    return out;
}

// A status dot for the header line, matching the tray icon's colour. Same 24bpp
// DIB reasoning as make_tray_icon: a screen-compatible DDB is 32bpp and menus
// read its alpha, which GDI never writes, so the dot would come out invisible.
HBITMAP make_status_dot(TrayStatus status) {
    const int d = GetSystemMetrics(SM_CXMENUCHECK);
    const int size = d > 0 ? d : 16;
    HDC screen = GetDC(nullptr);
    if (screen == nullptr) { return nullptr; }
    BITMAPINFO bi{};
    bi.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
    bi.bmiHeader.biWidth       = size;
    bi.bmiHeader.biHeight      = size;
    bi.bmiHeader.biPlanes      = 1;
    bi.bmiHeader.biBitCount    = 24;
    bi.bmiHeader.biCompression = BI_RGB;
    void* bits = nullptr;
    HBITMAP bmp = CreateDIBSection(screen, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
    if (bmp != nullptr) {
        HDC dc = CreateCompatibleDC(screen);
        HGDIOBJ old = SelectObject(dc, bmp);
        // Menu background varies with theme; COLOR_MENU is the closest system
        // answer and keeps the dot from sitting on a wrong-coloured tile.
        HBRUSH back = GetSysColorBrush(COLOR_MENU);
        RECT all{0, 0, size, size};
        FillRect(dc, &all, back);
        HBRUSH fill = CreateSolidBrush(status_fill(status));
        HGDIOBJ oldb = SelectObject(dc, fill);
        HGDIOBJ oldp = SelectObject(dc, GetStockObject(NULL_PEN));
        const int inset = size / 4;
        Ellipse(dc, inset, inset, size - inset, size - inset);
        SelectObject(dc, oldp);
        SelectObject(dc, oldb);
        DeleteObject(fill);
        SelectObject(dc, old);
        DeleteDC(dc);
    }
    ReleaseDC(nullptr, screen);
    return bmp;
}

// Free memory leads, because that is the number a person can act on; the split is
// secondary. Reads device-wide truth -- see query_device_memory_smi.
std::wstring memory_line(const NvidiaSmiMemory& m) {
    if (!m.ok || m.total_mib == 0) { return L"GPU memory unavailable"; }
    wchar_t buf[128];
    const double freeg = static_cast<double>(m.total_mib - m.used_mib) / 1024.0;
    const double tot   = static_cast<double>(m.total_mib) / 1024.0;
    _snwprintf_s(buf, _TRUNCATE, L"%.1f GB free of %.0f GB", freeg, tot);
    return buf;
}

void append_caption(HMENU menu, const std::wstring& text, HBITMAP dot = nullptr) {
    AppendMenuW(menu, MF_STRING | MF_DISABLED | MF_GRAYED, 0, text.c_str());
    if (dot != nullptr) {
        MENUITEMINFOW mii{};
        mii.cbSize     = sizeof(mii);
        mii.fMask      = MIIM_BITMAP;
        mii.hbmpItem   = dot;
        SetMenuItemInfoW(menu, GetMenuItemCount(menu) - 1, TRUE, &mii);
    }
}

int small_icon_size() {
    const int size = GetSystemMetrics(SM_CXSMICON);
    return size > 0 ? size : 16;
}

// Drawn rather than shipped as an .ico resource: no build-system change, no
// binary asset in the tree, correct at whatever SM_CXSMICON the display scaling
// reports, and the fill colour can carry status.
//
// The colour bitmap is an explicit 24bpp DIB section, NOT CreateCompatibleBitmap.
// A screen-compatible DDB is 32bpp on any modern display, and CreateIconIndirect
// then reads its alpha channel as per-pixel alpha. GDI never writes alpha, so
// every pixel would come out fully transparent and the icon would vanish. At
// 24bpp there is no alpha channel to misread and the 1bpp mask decides shape.
HICON make_tray_icon(TrayStatus status, int size) {
    HDC screen = GetDC(nullptr);
    if (screen == nullptr) { return nullptr; }

    BITMAPINFO bi{};
    bi.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
    bi.bmiHeader.biWidth       = size;
    bi.bmiHeader.biHeight      = -size; // top-down
    bi.bmiHeader.biPlanes      = 1;
    bi.bmiHeader.biBitCount    = 24;
    bi.bmiHeader.biCompression = BI_RGB;

    void* bits        = nullptr;
    HBITMAP color_bmp = CreateDIBSection(screen, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
    HBITMAP mask_bmp  = CreateBitmap(size, size, 1, 1, nullptr);
    HDC color_dc      = CreateCompatibleDC(screen);
    HDC mask_dc       = CreateCompatibleDC(screen);
    if (color_bmp == nullptr || mask_bmp == nullptr || color_dc == nullptr || mask_dc == nullptr) {
        if (color_dc != nullptr) { DeleteDC(color_dc); }
        if (mask_dc != nullptr) { DeleteDC(mask_dc); }
        if (color_bmp != nullptr) { DeleteObject(color_bmp); }
        if (mask_bmp != nullptr) { DeleteObject(mask_bmp); }
        ReleaseDC(nullptr, screen);
        return nullptr;
    }
    auto* old_color = static_cast<HBITMAP>(SelectObject(color_dc, color_bmp));
    auto* old_mask  = static_cast<HBITMAP>(SelectObject(mask_dc, mask_bmp));

    // Full-bleed fill; the rounded corners are cut by the mask, so the glyph
    // antialiases against the fill rather than fringing against a background.
    RECT rc{0, 0, size, size};
    HBRUSH fill = CreateSolidBrush(status_fill(status));
    FillRect(color_dc, &rc, fill);
    DeleteObject(fill);

    LOGFONTW lf{};
    lf.lfHeight  = -(size * 3 / 4);
    lf.lfWeight  = FW_BOLD;
    lf.lfQuality = ANTIALIASED_QUALITY;
    lf.lfCharSet = DEFAULT_CHARSET;
    lstrcpyW(lf.lfFaceName, L"Segoe UI");
    HFONT font = CreateFontIndirectW(&lf);
    if (font != nullptr) {
        auto* old_font = static_cast<HFONT>(SelectObject(color_dc, font));
        SetBkMode(color_dc, TRANSPARENT);
        SetTextColor(color_dc, RGB(0xFF, 0xFF, 0xFF));
        DrawTextW(color_dc, L"N", 1, &rc, DT_CENTER | DT_VCENTER | DT_SINGLELINE | DT_NOCLIP);
        SelectObject(color_dc, old_font);
        DeleteObject(font);
    }

    // 1bpp mask: white (1) transparent, black (0) opaque.
    PatBlt(mask_dc, 0, 0, size, size, WHITENESS);
    HPEN pen         = CreatePen(PS_SOLID, 1, RGB(0, 0, 0));
    auto* old_brush  = static_cast<HBRUSH>(SelectObject(mask_dc, GetStockObject(BLACK_BRUSH)));
    auto* old_pen    = static_cast<HPEN>(SelectObject(mask_dc, pen));
    const int radius = size / 3 < 2 ? 2 : size / 3;
    RoundRect(mask_dc, 0, 0, size, size, radius, radius);
    SelectObject(mask_dc, old_brush);
    SelectObject(mask_dc, old_pen);
    DeleteObject(pen);

    SelectObject(color_dc, old_color);
    SelectObject(mask_dc, old_mask);

    ICONINFO info{};
    info.fIcon    = TRUE;
    info.hbmMask  = mask_bmp;
    info.hbmColor = color_bmp;
    HICON icon    = CreateIconIndirect(&info);

    DeleteObject(color_bmp);
    DeleteObject(mask_bmp);
    DeleteDC(color_dc);
    DeleteDC(mask_dc);
    ReleaseDC(nullptr, screen);
    return icon;
}

LRESULT CALLBACK tray_wnd(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    TrayIcon* self = nullptr;
    if (msg == WM_NCCREATE) {
        auto* cs = reinterpret_cast<CREATESTRUCTW*>(lparam);
        self     = static_cast<TrayIcon*>(cs->lpCreateParams);
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
    } else {
        self = reinterpret_cast<TrayIcon*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    }
    if (self == nullptr) { return DefWindowProcW(hwnd, msg, wparam, lparam); }
    if (msg == WM_TIMER && wparam == kTimer) {
        self->refresh_icon();
        self->tick_idle_unload();
        return 0;
    }
    if (msg == kTrayMsg && (LOWORD(lparam) == WM_RBUTTONUP || LOWORD(lparam) == WM_LBUTTONUP)) {
        POINT pt{};
        GetCursorPos(&pt);
        HMENU menu = CreatePopupMenu();
        const EngineStatus est   = self->child().status();
        const TrayStatus tstatus = self->menu_status();
        const bool running       = est.state == EngineState::Running;

        HBITMAP dot = make_status_dot(tstatus);
        std::wstring head = L"NInfer is ";
        head += status_word(tstatus);
        if (!self->manages_engine()) { head += L" (observing)"; }
        append_caption(menu, head, dot);
        append_caption(menu, L"    " + memory_line(query_device_memory_smi(self->device_index())));
        AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);

        AppendMenuW(menu, MF_STRING, kIdOpen, L"Go to the dashboard");
        AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);

        HMENU reserve = CreatePopupMenu();
        for (const int gib : kReserveChoices) {
            wchar_t label[64];
            const wchar_t* note = gib == 2 ? L"   measured floor" : (gib == 4 ? L"   recommended" : L"");
            _snwprintf_s(label, _TRUNCATE, L"%d GB%s", gib, note);
            AppendMenuW(reserve, MF_STRING | (gib == self->reserve_gib() ? MF_CHECKED : 0),
                        kIdReserveBase + static_cast<UINT>(gib), label);
        }
        AppendMenuW(menu, MF_POPUP, reinterpret_cast<UINT_PTR>(reserve), L"Reserve for the desktop");

        HMENU idle = CreatePopupMenu();
        for (const int mins : kIdleChoices) {
            wchar_t label[64];
            if (mins == 0)        { _snwprintf_s(label, _TRUNCATE, L"Never"); }
            else if (mins < 60)   { _snwprintf_s(label, _TRUNCATE, L"After %d minutes", mins); }
            else                  { _snwprintf_s(label, _TRUNCATE, L"After %d hour%s", mins / 60,
                                                 mins == 60 ? L"" : L"s"); }
            AppendMenuW(idle, MF_STRING | (mins == self->idle_minutes() ? MF_CHECKED : 0),
                        kIdIdleBase + static_cast<UINT>(mins), label);
        }
        AppendMenuW(menu, MF_POPUP, reinterpret_cast<UINT_PTR>(idle), L"Unload when idle");

        // Releasing only the reclaimable caches needs a residency path the engine
        // does not have yet. Shown disabled with the reason rather than hidden --
        // Docker greys Kubernetes Context the same way when it is unavailable.
        AppendMenuW(menu, MF_STRING | MF_DISABLED | MF_GRAYED, kIdReleaseCache,
                    L"Release cache\tneeds engine support");
        AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);

        AppendMenuW(menu, MF_STRING, kIdRestart, L"Restart engine");
        AppendMenuW(menu, MF_STRING, running ? kIdStop : kIdStart,
                    running ? L"Unload model\tfrees ~70 GB" : L"Load model");
        AppendMenuW(menu, MF_STRING, kIdQuit, L"Quit supervisor");
        SetForegroundWindow(hwnd);
        const int cmd =
            TrackPopupMenu(menu, TPM_RETURNCMD | TPM_NONOTIFY, pt.x, pt.y, 0, hwnd, nullptr);
        DestroyMenu(menu);
        if (dot != nullptr) { DeleteObject(dot); }
        if (cmd == kIdOpen) { self->open_dashboard(); }
        if (cmd == kIdStart) { self->child().start(); }
        if (cmd == kIdStop) { self->child().stop(); }
        if (cmd == kIdRestart) { self->child().restart(); }
        if (cmd == kIdQuit) { PostQuitMessage(0); }
        if (cmd >= kIdReserveBase && cmd < kIdIdleBase) {
            self->on_reserve_chosen(static_cast<int>(cmd - kIdReserveBase));
        }
        if (cmd >= kIdIdleBase && cmd < kIdReleaseCache) {
            self->on_idle_chosen(static_cast<int>(cmd - kIdIdleBase));
        }
        return 0;
    }
    if (msg == WM_DESTROY) {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wparam, lparam);
}

} // namespace

TrayIcon::TrayIcon(EngineChild& child, EngineSpec& spec, std::string dashboard_url,
                   bool manages_engine, std::string prefs_path)
    : child_(child), spec_(spec), dashboard_url_(std::move(dashboard_url)),
      manages_engine_(manages_engine), prefs_path_(std::move(prefs_path)),
      prefs_(load_tray_prefs(prefs_path_)) {
    // The stored reserve is the user's standing choice, so make the engine's
    // arguments agree with it at startup rather than only after they touch the
    // menu. Without this the menu would show one number and the engine would run
    // with another, which is exactly the kind of quiet disagreement that made the
    // desktop unusable twice today.
    apply_reserve(prefs_.desktop_reserve_gib);
}

TrayIcon::~TrayIcon() {
    if (hwnd_ != nullptr) { DestroyWindow(static_cast<HWND>(hwnd_)); }
    if (hicon_ != nullptr) { DestroyIcon(static_cast<HICON>(hicon_)); }
}

EngineChild& TrayIcon::child() { return child_; }

// Rewrites --desktop-reserve-gib in the engine's argument vector, or appends it
// when absent. Takes effect on the next engine start; the menu says so rather
// than implying a running engine can be resized, which it cannot yet -- the
// reserve is a sizing input and there is no runtime enforcement (see D23).
void TrayIcon::apply_reserve(int gib) {
    auto& args = spec_.args;
    for (std::size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == "--desktop-reserve-gib") {
            args[i + 1] = std::to_string(gib);
            return;
        }
        if (args[i] == "--desktop-reserve-mib") {
            // A MiB-valued reserve is more specific than anything this menu can
            // express, so leave it alone rather than silently coarsening it.
            return;
        }
    }
    args.emplace_back("--desktop-reserve-gib");
    args.emplace_back(std::to_string(gib));
}

void TrayIcon::on_reserve_chosen(int gib) {
    if (gib == prefs_.desktop_reserve_gib) { return; }
    prefs_.desktop_reserve_gib = gib;
    save_tray_prefs(prefs_path_, prefs_);
    apply_reserve(gib);

    // Ask before restarting. Reserving memory is cheap to choose and expensive to
    // apply -- a restart costs a full model load -- so the cost is stated and the
    // decision is the user's.
    if (child_.status().state == EngineState::Running) {
        wchar_t msg[256];
        _snwprintf_s(msg, _TRUNCATE,
                     L"Reserve set to %d GB.\n\nThis applies when the engine next starts. "
                     L"Restart now? The model reloads, which takes about a minute.",
                     gib);
        if (MessageBoxW(nullptr, msg, L"NInfer", MB_YESNO | MB_ICONQUESTION) == IDYES) {
            child_.restart();
        }
    }
}

void TrayIcon::on_idle_chosen(int minutes) {
    prefs_.idle_unload_minutes = minutes;
    save_tray_prefs(prefs_path_, prefs_);
    last_busy_ = std::chrono::steady_clock::now();
}

// Unloads an idle engine so a workstation is not holding ~70 GB of weights while
// nobody is using it. "Idle" is deliberately coarse: the engine reports running
// and healthy but the supervisor has seen no state change, which is enough for a
// timer measured in tens of minutes and avoids plumbing per-request activity
// through for a feature whose shortest useful setting is 15 minutes.
void TrayIcon::tick_idle_unload() {
    if (prefs_.idle_unload_minutes <= 0 || !manages_engine_) { return; }
    const EngineStatus st = child_.status();
    if (st.state != EngineState::Running) {
        last_busy_ = std::chrono::steady_clock::now();
        return;
    }
    const auto idle_for = std::chrono::steady_clock::now() - last_busy_;
    if (idle_for >= std::chrono::minutes(prefs_.idle_unload_minutes)) {
        child_.stop();
        last_busy_ = std::chrono::steady_clock::now();
    }
}

// Device-wide memory. nvidia-smi rather than cudaMemGetInfo or DXGI, both of
// which report an empty card while another process holds 70 GiB.
NvidiaSmiMemory query_device_memory_smi(int device) {
    NvidiaSmiMemory out;
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE r = nullptr;
    HANDLE w = nullptr;
    if (!CreatePipe(&r, &w, &sa, 0)) { return out; }
    SetHandleInformation(r, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW si{};
    si.cb         = sizeof(si);
    si.dwFlags    = STARTF_USESTDHANDLES;
    si.hStdOutput = w;
    si.hStdError  = w;

    wchar_t cmd[] = L"nvidia-smi --query-gpu=index,memory.used,memory.total "
                    L"--format=csv,noheader,nounits";
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, cmd, nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr,
                        &si, &pi)) {
        CloseHandle(r);
        CloseHandle(w);
        return out;
    }
    CloseHandle(w);
    CloseHandle(pi.hThread);

    std::string buf;
    char chunk[512];
    DWORD got = 0;
    while (ReadFile(r, chunk, sizeof(chunk), &got, nullptr) && got > 0) {
        buf.append(chunk, got);
    }
    CloseHandle(r);
    // Bounded: the menu is being drawn and must not hang if nvidia-smi wedges.
    WaitForSingleObject(pi.hProcess, 2000);
    CloseHandle(pi.hProcess);
    return parse_nvidia_smi_memory_csv(buf, device);
}


void TrayIcon::open_dashboard() const {
    ShellExecuteA(nullptr, "open", dashboard_url_.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

void TrayIcon::request_quit() {
    if (hwnd_ != nullptr) { PostMessageW(static_cast<HWND>(hwnd_), WM_CLOSE, 0, 0); }
}

TrayStatus TrayIcon::current_status() {
    const EngineStatus status = child_.status();
    if (!manages_engine_) {
        // Unmanaged: no process of ours has a state, so observed health is the
        // only thing actually measured. EngineChild keeps it current for
        // unmanaged engines through observe_health().
        if (status.health == "ok") { return TrayStatus::Working; }
        if (status.health == "unhealthy") { return TrayStatus::Failed; }
        if (status.health == "unreachable") { return TrayStatus::Pending; }
        return TrayStatus::Idle;
    }
    if (status.crash_loop_halted) { return TrayStatus::Failed; }
    switch (status.state) {
    case EngineState::Running:
        return status.health == "unhealthy" ? TrayStatus::Failed : TrayStatus::Working;
    case EngineState::Starting:
    case EngineState::Stopping:
    case EngineState::BackingOff: return TrayStatus::Pending;
    case EngineState::Halted: return TrayStatus::Failed;
    case EngineState::Stopped: break;
    }
    return TrayStatus::Idle;
}

std::wstring TrayIcon::tooltip(TrayStatus status) const {
    std::wstring tip = L"NInfer supervisor - ";
    tip += status_word(status);
    if (!manages_engine_) { tip += L" (monitor-only)"; }
    return tip;
}

void TrayIcon::refresh_icon() {
    if (hwnd_ == nullptr) { return; }
    const TrayStatus status = current_status();
    if (static_cast<int>(status) == last_status_) { return; }
    last_status_ = static_cast<int>(status);

    HICON icon = make_tray_icon(status, small_icon_size());
    if (icon == nullptr) { return; }

    const std::wstring tip = tooltip(status);
    NOTIFYICONDATAW nid{};
    nid.cbSize = sizeof(nid);
    nid.hWnd   = static_cast<HWND>(hwnd_);
    nid.uID    = kTrayUid;
    nid.uFlags = NIF_ICON | NIF_TIP;
    nid.hIcon  = icon;
    lstrcpynW(nid.szTip, tip.c_str(), ARRAYSIZE(nid.szTip));
    Shell_NotifyIconW(NIM_MODIFY, &nid);

    if (hicon_ != nullptr) { DestroyIcon(static_cast<HICON>(hicon_)); }
    hicon_ = icon;
}

void TrayIcon::run() {
    WNDCLASSEXW wc{};
    wc.cbSize        = sizeof(wc);
    wc.lpfnWndProc   = tray_wnd;
    wc.hInstance     = GetModuleHandleW(nullptr);
    wc.lpszClassName = kClass;
    RegisterClassExW(&wc);
    HWND hwnd = CreateWindowExW(0, kClass, L"NInfer supervisor", 0, 0, 0, 0, 0, HWND_MESSAGE,
                                nullptr, wc.hInstance, this);
    hwnd_     = hwnd;

    const TrayStatus status = current_status();
    last_status_            = static_cast<int>(status);
    HICON icon              = make_tray_icon(status, small_icon_size());
    hicon_                  = icon;
    const std::wstring tip  = tooltip(status);

    NOTIFYICONDATAW nid{};
    nid.cbSize           = sizeof(nid);
    nid.hWnd             = hwnd;
    nid.uID              = kTrayUid;
    nid.uFlags           = NIF_MESSAGE | NIF_TIP | NIF_ICON;
    nid.uCallbackMessage = kTrayMsg;
    nid.hIcon            = icon != nullptr ? icon : LoadIconW(nullptr, MAKEINTRESOURCEW(32512));
    lstrcpynW(nid.szTip, tip.c_str(), ARRAYSIZE(nid.szTip));
    Shell_NotifyIconW(NIM_ADD, &nid);
    SetTimer(hwnd, kTimer, 1000, nullptr);

    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    KillTimer(hwnd, kTimer);
    Shell_NotifyIconW(NIM_DELETE, &nid);
}

} // namespace ninfer::supervisor
