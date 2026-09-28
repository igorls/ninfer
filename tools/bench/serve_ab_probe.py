"""Alternating-arm ninfer-serve probe: cold 7,680-token prefill and 256-token greedy decode.

Each arm is an artifact (optionally with its own executable) served with the same flags. Every
round starts a fresh ``ninfer-serve`` per arm in alternating order (ABBA...), records
``/admin/vram``, sends one warmup, ``--samples`` cold prefills (a fresh salt per request,
read-only prompt cache, one output token) and ``--samples`` greedy decodes (counting prompt,
thinking off), then stops the server. Prefill and decode times are the server's own request-log
timings; decode tok/s excludes the first token.

    python tools/bench/serve_ab_probe.py --exe build/apps/ninfer-serve --port 8021 \\
        --arm current=models/a.ninfer --arm nvfp4full=models/b.ninfer --rounds 5 \\
        --out profiles/bench/ab --flags "--max-context 131072 --kv-capacity 131072 ..."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

FIXTURE = Path(__file__).resolve().parents[2] / "bench/fixtures/ttft/text/long_64k_independent.json"
DECODE_PROMPT = "Count from 1 to 400 in words, one number per line, with no other text."


def _request(port: int, path: str, body: dict | None = None, timeout: float = 600.0) -> dict:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _wait_ready(port: int, process: subprocess.Popen, timeout: float = 900.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"ninfer-serve exited with {process.returncode} during startup")
        try:
            _request(port, "/v1/models", timeout=5)
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(1.0)
    raise TimeoutError("ninfer-serve did not become ready")


def _prefill_messages(document: str, salt: int) -> list[dict]:
    # Fixed-width decimal salt: every request has the same token count and no shared prefix.
    return [
        {"role": "system", "content": f"Run {salt:06d}. Answer the supplied document independently."},
        {"role": "user", "content": document},
    ]


def _chat(port: int, messages: list[dict], max_tokens: int, **extra) -> dict:
    body = {
        "model": _request(port, "/v1/models")["data"][0]["id"],
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        **extra,
    }
    return _request(port, "/v1/chat/completions", body)


def _fit_document(port: int, target_tokens: int) -> str:
    """Longest prefix of the fixture's user text whose rendered prompt has target_tokens."""
    source = json.loads(FIXTURE.read_text(encoding="utf-8"))
    text = next(m["content"] for m in source if m["role"] == "user")
    # About four characters per token; the bound keeps every probe prefill short.
    low, high = 0, min(len(text), 8 * target_tokens)
    count = lambda n: _chat(port, _prefill_messages(text[:n], 0), 1, prompt_cache_read_only=True)[
        "usage"
    ]["prompt_tokens"]
    if count(high) < target_tokens:
        raise ValueError("fixture is shorter than the target prompt")
    while low < high:
        middle = (low + high + 1) // 2
        if count(middle) <= target_tokens:
            low = middle
        else:
            high = middle - 1
    if count(low) != target_tokens:
        raise ValueError(f"no fixture prefix renders exactly {target_tokens} tokens")
    return text[:low]


def _records(log: Path, begin: int) -> list[dict]:
    lines = log.read_text(encoding="utf-8").splitlines()[begin:]
    return [r for r in map(json.loads, lines) if r.get("event") == "request_done"]


def run_arm(args, label: str, artifact: str, exe: str, round_index: int, salt: int, document):
    out = Path(args.out)
    log = out / f"requests-{label}-r{round_index}.jsonl"
    command = [exe, artifact, "--port", str(args.port), "--request-log-jsonl", str(log)]
    command += shlex.split(args.flags)
    stderr = open(out / f"serve-{label}-r{round_index}.log", "w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=stderr, stderr=subprocess.STDOUT)
    try:
        _wait_ready(args.port, process)
        vram = _request(args.port, "/admin/vram")
        if document is None:
            document = _fit_document(args.port, args.prompt_tokens)
        _chat(args.port, _prefill_messages(document, salt), 1, prompt_cache_read_only=True)
        _chat(args.port, [{"role": "user", "content": DECODE_PROMPT}], 16)
        begin = len(log.read_text(encoding="utf-8").splitlines())
        for sample in range(args.samples):
            _chat(args.port, _prefill_messages(document, salt + 1 + sample), 1,
                  prompt_cache_read_only=True)
        for _ in range(args.samples):
            _chat(args.port, [{"role": "user", "content": DECODE_PROMPT}], args.decode_tokens)
        records = _records(log, begin)
    finally:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
        stderr.close()
    prefill = [r for r in records if r["result"]["completion_tokens"] == 1]
    decode = [r for r in records if r["result"]["completion_tokens"] == args.decode_tokens]
    if len(prefill) != args.samples or len(decode) != args.samples:
        raise RuntimeError(f"{label}: expected {args.samples} prefill and decode records")
    if any(r["result"]["computed_prefill_tokens"] != args.prompt_tokens for r in prefill):
        raise RuntimeError(f"{label}: a prefill sample did not compute {args.prompt_tokens} tokens")
    result = {
        "arm": label,
        "round": round_index,
        "artifact": artifact,
        "prefill_ms": [1000 * r["timings_seconds"]["prefill"] for r in prefill],
        "decode_tok_s": [
            (r["result"]["completion_tokens"] - 1) / r["timings_seconds"]["decode"] for r in decode
        ],
        "acceptance": [
            r["speculative"]["accepted_tokens"] / r["speculative"]["drafted_tokens"]
            for r in decode
            if r["speculative"]["drafted_tokens"]
        ],
        "accepted_per_position": [r["speculative"]["accepted_per_position"] for r in decode],
        "window": [prefill[0]["timestamp_unix_ms"], decode[-1]["timestamp_unix_ms"]],
        "weights_bytes": vram["arenas"]["weights"]["capacity_bytes"],
        "device_free_bytes": vram["device"]["free_bytes"],
        "kv": vram["kv"],
        "plan": vram["plan"],
    }
    return result, document


def summarize(results: list[dict]) -> dict:
    summary = {}
    for label in dict.fromkeys(r["arm"] for r in results):
        rows = [r for r in results if r["arm"] == label]
        prefill = [v for r in rows for v in r["prefill_ms"]]
        decode = [v for r in rows for v in r["decode_tok_s"]]
        acceptance = [v for r in rows for v in r["acceptance"]]
        summary[label] = {
            "rounds": len(rows),
            "prefill_ms_median": statistics.median(prefill),
            "prefill_ms_range": [min(prefill), max(prefill)],
            "decode_tok_s_median": statistics.median(decode),
            "decode_tok_s_range": [min(decode), max(decode)],
            "acceptance_range": [min(acceptance), max(acceptance)] if acceptance else None,
            "weights_gib": rows[0]["weights_bytes"] / 2**30,
            "device_free_gib": [r["device_free_bytes"] / 2**30 for r in rows],
        }
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--exe", required=True, help="default ninfer-serve executable")
    parser.add_argument("--arm", action="append", required=True, metavar="LABEL=ARTIFACT[@EXE]")
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--flags", required=True, help="ninfer-serve flags shared by every arm")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--prompt-tokens", type=int, default=7680)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    arms = []
    for item in args.arm:
        label, _, rest = item.partition("=")
        artifact, _, exe = rest.partition("@")
        arms.append((label, artifact, exe or args.exe))
    results_path = Path(args.out) / "results.jsonl"
    results, document, salt = [], None, 100000
    for round_index in range(1, args.rounds + 1):
        order = arms if round_index % 2 else list(reversed(arms))
        for label, artifact, exe in order:
            salt += 1000
            result, document = run_arm(args, label, artifact, exe, round_index, salt, document)
            results.append(result)
            with results_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result) + "\n")
            print(
                f"round {round_index} {label}: prefill {statistics.median(result['prefill_ms']):.1f} ms, "
                f"decode {statistics.median(result['decode_tok_s']):.1f} tok/s, "
                f"weights {result['weights_bytes'] / 2**30:.2f} GiB",
                flush=True,
            )
    summary = summarize(results)
    (Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.exit(main())
