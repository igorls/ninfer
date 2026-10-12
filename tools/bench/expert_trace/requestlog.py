"""Join trace request ids to clients through the serve request log (`--request-log-jsonl`).

Every trace column carries the request-log id of its request: the server passes its
`request.request_id` to the Engine as an opaque route tag and the Engine copies it into the column
(0 means untagged). The join is therefore direct: trace `request_id` -> the `request` object of the
`request_start` / `request_done` / `request_error` events with that `request.request_id`
(`request_rejected` events carry no client and their requests never reach the Engine). The client (HTTP `User-Agent`), protocol, tools digest and thinking mode are
fields of that object. Request ids are only unique within one server instance and every log line
carries `server_instance_id`, so lines are filtered on it.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

NO_USER_AGENT = "(no-user-agent)"
UNJOINED = "(unjoined)"
UNTAGGED = "(untagged)"

_TS_RE = re.compile(rb'"timestamp_unix_ms"\s*:\s*(\d+)')
_REQUEST_EVENTS = ("request_start", "request_done", "request_error")
_REQUEST_EVENT_MARKS = tuple(event.encode() for event in _REQUEST_EVENTS)


@dataclass
class RequestLogJoin:
    """Request id -> label, plus how the scan went."""

    labels: dict[int, str] = field(default_factory=dict)
    group_by: tuple[str, ...] = ("client",)
    lines_scanned: int = 0
    lines_matched: int = 0

    def summary(self) -> dict[str, object]:
        return {
            "group_by": list(self.group_by),
            "lines_scanned": self.lines_scanned,
            "lines_matched": self.lines_matched,
            "requests_labelled": len(self.labels),
        }


def request_label(request: Mapping[str, object], group_by: Sequence[str], *, strip_version: bool) -> str:
    parts = []
    for name in group_by:
        value = request.get(name)
        if name == "client":
            text = str(value or "").strip()
            if not text:
                text = NO_USER_AGENT
            elif strip_version:
                text = text.split("/", 1)[0].strip() or NO_USER_AGENT
            parts.append(text)
        elif name == "tools":
            parts.append("tools" if request.get("tool_count") else "no-tools")
        else:
            parts.append("(none)" if value in (None, "") else str(value))
    return "|".join(parts)


def _seek_to_time(f, size: int, target_ms: int) -> int:
    """Byte offset of a line boundary at or before the first line with timestamp >= target_ms.

    Request logs are appended chronologically, so a bisection on line timestamps avoids scanning
    a multi-gigabyte log from the start. Lines are read at probe offsets only.
    """
    lo, hi = 0, size
    for _ in range(48):
        if hi - lo < (1 << 16):
            break
        mid = (lo + hi) // 2
        f.seek(mid)
        f.readline()  # discard the partial line
        stamp = None
        for _ in range(64):
            line = f.readline()
            if not line:
                break
            match = _TS_RE.search(line)
            if match:
                stamp = int(match.group(1))
                break
        if stamp is None or stamp >= target_ms:
            hi = mid
        else:
            lo = mid
    return lo


def load_request_log(
    path: str | os.PathLike[str],
    server_instance_id: str,
    *,
    group_by: Sequence[str] = ("client",),
    strip_version: bool = False,
    since_unix_ms: int | None = None,
    until_unix_ms: int | None = None,
    tail_bytes: int | None = None,
) -> RequestLogJoin:
    """Scan the request log (streaming) and map request ids to labels.

    `since_unix_ms` bisects to the first relevant byte instead of reading from the start;
    `until_unix_ms` stops the scan once lines are later than that. `tail_bytes` reads only the last
    N bytes (a cheaper, cruder alternative).
    """
    group_by = tuple(group_by)
    needle = server_instance_id.encode("utf-8")
    join = RequestLogJoin(group_by=group_by)
    requests: dict[int, dict] = {}

    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        start = 0
        if tail_bytes is not None and tail_bytes < size:
            start = size - tail_bytes
        elif since_unix_ms is not None:
            start = _seek_to_time(f, size, since_unix_ms)
        f.seek(start)
        if start:
            f.readline()  # drop the partial first line
        for line in f:
            join.lines_scanned += 1
            if needle not in line:
                continue
            if until_unix_ms is not None:
                match = _TS_RE.search(line)
                if match and int(match.group(1)) > until_unix_ms:
                    break
            if not any(mark in line for mark in _REQUEST_EVENT_MARKS):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn final line
            if obj.get("server_instance_id") != server_instance_id or obj.get("event") not in _REQUEST_EVENTS:
                continue
            request = obj.get("request")
            if not isinstance(request, dict) or "request_id" not in request:
                continue
            join.lines_matched += 1
            request_id = int(request["request_id"])
            # a later event of the same request refines the shape (start -> done)
            requests[request_id] = {**requests.get(request_id, {}), **request}

    for request_id, request in requests.items():
        join.labels[request_id] = request_label(request, group_by, strip_version=strip_version)
    return join
