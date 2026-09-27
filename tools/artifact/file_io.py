"""Keep large offline transfers from retaining whole artifacts in the OS page cache."""

from __future__ import annotations

import os

IO_CHUNK_BYTES = 8 * 1024 * 1024
WRITEBACK_BYTES = 64 * 1024 * 1024

# Windows opens descriptors in text mode unless told otherwise.
READ_FLAGS = os.O_RDONLY | getattr(os, "O_BINARY", 0)

if hasattr(os, "pread"):
    pread = os.pread
    pwrite = os.pwrite
else:
    # Windows has no positional descriptor I/O. Each descriptor here is used by one thread, so
    # seeking before the transfer is equivalent.
    def pread(fd: int, count: int, offset: int) -> bytes:
        os.lseek(fd, offset, os.SEEK_SET)
        chunks = []
        while count > 0:
            chunk = os.read(fd, count)
            if not chunk:
                break
            chunks.append(chunk)
            count -= len(chunk)
        return b"".join(chunks)

    def pwrite(fd: int, data, offset: int) -> int:
        os.lseek(fd, offset, os.SEEK_SET)
        return os.write(fd, data)

if os.name == "nt":
    # Windows has no per-range page-cache advice; its memory manager trims the standby list of
    # written file pages on its own, so writeback is only bounded by syncing.
    def discard_cached_pages(fd: int, offset: int = 0, count: int | None = None) -> None:
        del fd, offset, count

    _sync = os.fsync
else:
    _PAGE_BYTES = os.sysconf("SC_PAGE_SIZE")

    def discard_cached_pages(fd: int, offset: int = 0, count: int | None = None) -> None:
        if count is None:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        elif count > 0:
            begin = offset // _PAGE_BYTES * _PAGE_BYTES
            end = (offset + count + _PAGE_BYTES - 1) // _PAGE_BYTES * _PAGE_BYTES
            os.posix_fadvise(fd, begin, end - begin, os.POSIX_FADV_DONTNEED)

    _sync = os.fdatasync


class Writeback:
    """Bound dirty output across all open shards; release clean pages after writeback."""

    def __init__(self) -> None:
        self._bytes = 0
        self._fds: set[int] = set()

    def written(self, fd: int, count: int) -> None:
        self._fds.add(fd)
        self._bytes += count
        if self._bytes >= WRITEBACK_BYTES:
            self.flush()

    def flush(self) -> None:
        for fd in self._fds:
            _sync(fd)
            discard_cached_pages(fd)
        self._fds.clear()
        self._bytes = 0
