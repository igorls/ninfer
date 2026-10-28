#pragma once

#include <windows.h>

namespace ninfer::io::windows {
// Native syscall boundary, also used by the fault-injection executable at link time.
BOOL issue_read(HANDLE, void*, DWORD, OVERLAPPED*);
BOOL completion(HANDLE, DWORD*, ULONG_PTR*, OVERLAPPED**, DWORD);
BOOL signal(HANDLE);
} // namespace ninfer::io::windows
