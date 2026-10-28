#include "core/io/windows_calls.h"

namespace ninfer::io::windows {
BOOL issue_read(HANDLE file, void* destination, DWORD bytes, OVERLAPPED* operation) {
    return ReadFile(file, destination, bytes, nullptr, operation);
}
BOOL completion(HANDLE port, DWORD* bytes, ULONG_PTR* key, OVERLAPPED** operation, DWORD timeout) {
    return GetQueuedCompletionStatus(port, bytes, key, operation, timeout);
}
BOOL signal(HANDLE port) { return PostQueuedCompletionStatus(port, 0, 0, nullptr); }
} // namespace ninfer::io::windows
