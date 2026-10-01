"""Flash-Next engine parity: collect greedy/prompt log probabilities from a ninfer-serve and compare.

The request set is built only from committed public text (fixed short prompts and the
eval/corpora/perplexity-1m streams). Every request is greedy with thinking as listed.

    collect  run the set against a running server and write one JSON record per request:
             python -m tools.bench.flash_next.parity collect --port 18090 --out v2.jsonl
               [--concurrency 2] [--prompt-positions]
    compare  compare a candidate against a reference, against a noise floor measured on the
             reference engine itself (a second run with a different but equally valid route):
             python -m tools.bench.flash_next.parity compare --reference v2.jsonl
               --candidate v3.jsonl [--noise v2_b2.jsonl] [--out report.json]

Generated positions report the top-20 raw log probabilities (before any sampling adjustment);
prompt positions report the raw log probability of the prompt's own next token. Comparison:

- greedy: per request, the first position where the token sequences differ and the reference's
  top-2 gap (nats) there; a divergence away from a near tie is a parity failure;
- shared trajectory: on positions both sequences share, KL(reference || candidate) restricted
  to the reference top-20 (a candidate token outside its own top-20 takes its 20th log
  probability, an upper bound on its mass) and top-1 agreement;
- prompt positions: |delta log p(next token)| statistics.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
from pathlib import Path
import statistics
import sys
import time
import urllib.error
import urllib.request

REPOSITORY = Path(__file__).resolve().parents[3]
CORPUS = REPOSITORY / "eval/corpora/perplexity-1m/data"

SHORT = [
    "Explain in two sentences why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "List three prime numbers greater than one hundred and show how you checked each one.",
    "Translate to Portuguese: The library opens at nine in the morning and closes at five.",
    "What is the capital of Australia, and why is it not Sydney?",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
    "Give two arguments for and two against daylight saving time.",
    "Describe how a hash table resolves collisions, with one example.",
    "Write a haiku about a lighthouse in winter.",
    "Solve for x: 3x + 7 = 25. Show the steps.",
    "用三句话介绍长城的历史。",
    "Explain the difference between TCP and UDP for a beginner.",
    "Write a SQL query that returns the five most recent orders per customer.",
    "What causes the seasons on Earth? Answer briefly.",
    "Rewrite this sentence in the passive voice: The committee approved the new budget.",
    "Name four renewable energy sources and one limitation of each.",
]
THINKING = [
    "A train leaves at 14:35 and arrives at 18:10. How long is the trip in minutes?",
    "Is 221 a prime number? Explain.",
    "How many times does the letter r appear in the word strawberry?",
    "If a rectangle has perimeter 30 and area 56, what are its sides?",
]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                "required": ["city"],
            },
        },
    }
]
SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "city",
        "schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "country": {"type": "string"},
                           "population_millions": {"type": "number"}},
            "required": ["city", "country", "population_millions"],
            "additionalProperties": False,
        },
    },
}


def _document(domain: str, characters: int) -> str:
    return (CORPUS / domain / "00.txt").read_text(encoding="utf-8")[:characters]


def request_set() -> list[dict]:
    """Deterministic list of {id, messages, max_tokens, thinking, extra}."""
    out = []
    for i, text in enumerate(SHORT):
        out.append({"id": f"short-{i:02d}", "messages": [{"role": "user", "content": text}],
                    "max_tokens": 192, "thinking": False})
    for i, text in enumerate(THINKING):
        out.append({"id": f"thinking-{i}", "messages": [{"role": "user", "content": text}],
                    "max_tokens": 256, "thinking": True})
    out.append({"id": "tool-0", "messages": [{"role": "user", "content": "What is the weather in Lisbon right now?"}],
                "max_tokens": 96, "thinking": False, "extra": {"tools": TOOLS}})
    out.append({"id": "tool-1", "messages": [{"role": "user", "content": "Compare the weather in Oslo and Cairo in Fahrenheit."}],
                "max_tokens": 128, "thinking": False, "extra": {"tools": TOOLS}})
    out.append({"id": "schema-0", "messages": [{"role": "user", "content": "Describe Tokyo as JSON."}],
                "max_tokens": 96, "thinking": False, "extra": {"response_format": SCHEMA}})
    out.append({"id": "schema-1", "messages": [{"role": "user", "content": "Describe Nairobi as JSON."}],
                "max_tokens": 96, "thinking": False, "extra": {"response_format": SCHEMA}})
    for domain, characters, tag in (("ninfer", 16000, "code-4k"), ("wikitext", 80000, "wiki-18k"),
                                    ("pg19", 80000, "book-18k")):
        out.append({"id": tag, "messages": [{"role": "user", "content":
                    "Read the text and summarize it in three sentences.\n\n" + _document(domain, characters)}],
                    "max_tokens": 128, "thinking": False})
    return out


def _post(port: int, body: dict, timeout: float = 1800.0) -> dict:
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _model(port: int) -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=30) as response:
        return json.loads(response.read())["data"][0]["id"]


def _generated(port: int, model: str, item: dict) -> dict:
    body = {"model": model, "messages": item["messages"], "max_tokens": item["max_tokens"],
            "temperature": 0, "logprobs": True, "top_logprobs": 20,
            "chat_template_kwargs": {"enable_thinking": item["thinking"]},
            "prompt_cache_read_only": True, **item.get("extra", {})}
    started = time.time()
    response = _post(port, body)
    content = response["choices"][0]["logprobs"]["content"]
    return {
        "id": item["id"], "kind": "generated", "seconds": time.time() - started,
        "prompt_tokens": response["usage"]["prompt_tokens"],
        "finish_reason": response["choices"][0]["finish_reason"],
        "tokens": [entry["token_id"] for entry in content],
        "raw": [entry["raw_logprob"] for entry in content],
        "top": [[[t["token_id"], t["raw_logprob"]] for t in entry["top_logprobs"]] for entry in content],
    }


def _prompt_positions(port: int, model: str, item: dict, prompt_tokens: int) -> dict:
    count = min(256, prompt_tokens - 1)
    positions = sorted({round(i * (prompt_tokens - 2) / max(1, count - 1)) for i in range(count)})
    body = {"model": model, "messages": item["messages"], "max_tokens": 1, "temperature": 0,
            "logprobs": True, "top_logprobs": 0, "logprob_prompt_positions": positions,
            "chat_template_kwargs": {"enable_thinking": item["thinking"]},
            "prompt_cache_read_only": True}
    response = _post(port, body)
    entries = response["choices"][0]["logprobs"]["prompt"]
    return {"id": item["id"], "kind": "prompt", "positions": positions,
            "tokens": [e["token_id"] for e in entries], "raw": [e["raw_logprob"] for e in entries]}


def collect(args: argparse.Namespace) -> None:
    model = _model(args.port)
    items = request_set()
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {(r["id"], r["kind"]) for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())}
    with open(out, "a", encoding="utf-8") as sink, \
            concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        pending = [item for item in items if (item["id"], "generated") not in done]
        for record in pool.map(lambda item: _generated(args.port, model, item), pending):
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            print(f"{record['id']}: {len(record['tokens'])} tokens in {record['seconds']:.1f}s", flush=True)
        if args.prompt_positions:
            lengths = {r["id"]: r["prompt_tokens"] for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())
                       if r["kind"] == "generated"}
            for item in items:
                if (item["id"], "prompt") in done or item["id"] not in lengths or item.get("extra"):
                    continue
                record = _prompt_positions(args.port, model, item, lengths[item["id"]])
                sink.write(json.dumps(record) + "\n")
                sink.flush()
                print(f"{item['id']}: {len(record['positions'])} prompt positions", flush=True)


def _load(path: str) -> dict:
    out = {}
    for record in map(json.loads, Path(path).read_text(encoding="utf-8").splitlines()):
        out[(record["id"], record["kind"])] = record
    return out


def _kl(reference_top: list, candidate_top: list) -> float:
    candidate = {token: logprob for token, logprob in candidate_top}
    floor = min(candidate.values())
    total = 0.0
    for token, logprob in reference_top:
        p = math.exp(logprob)
        total += p * (logprob - candidate.get(token, floor))
    return max(total, 0.0)


def _pair(reference: dict, candidate: dict) -> dict:
    report = {"requests": 0, "identical": 0, "divergences": [], "kl": [], "agree": 0, "positions": 0,
              "prompt_delta": []}
    for key, ref in reference.items():
        cand = candidate.get(key)
        if cand is None:
            continue
        if key[1] == "prompt":
            if ref["positions"] == cand["positions"]:
                report["prompt_delta"].extend(abs(a - b) for a, b in zip(ref["raw"], cand["raw"]))
            continue
        report["requests"] += 1
        shared = 0
        for a, b in zip(ref["tokens"], cand["tokens"]):
            if a != b:
                break
            shared += 1
        for position in range(shared):
            report["kl"].append(_kl(ref["top"][position], cand["top"][position]))
        report["positions"] += shared
        report["agree"] += shared
        if shared == min(len(ref["tokens"]), len(cand["tokens"])):
            report["identical"] += 1
        else:
            top = ref["top"][shared]
            gap = top[0][1] - top[1][1] if len(top) > 1 else float("inf")
            report["positions"] += 1
            report["divergences"].append({"id": key[0], "position": shared, "reference_gap": gap})
    return report


def _summary(report: dict) -> dict:
    kl = report["kl"]
    delta = report["prompt_delta"]
    return {
        "requests": report["requests"], "identical": report["identical"],
        "top1_agreement": report["agree"] / report["positions"] if report["positions"] else None,
        "shared_positions": len(kl), "kl_mean": statistics.fmean(kl) if kl else None,
        "kl_p99": sorted(kl)[int(0.99 * (len(kl) - 1))] if kl else None, "kl_max": max(kl) if kl else None,
        "divergences": report["divergences"],
        "max_divergence_gap": max((d["reference_gap"] for d in report["divergences"]), default=0.0),
        "prompt_positions": len(delta), "prompt_delta_mean": statistics.fmean(delta) if delta else None,
        "prompt_delta_max": max(delta) if delta else None,
    }


def compare(args: argparse.Namespace) -> None:
    reference = _load(args.reference)
    result = {"candidate": _summary(_pair(reference, _load(args.candidate)))}
    if args.noise:
        result["noise_floor"] = _summary(_pair(reference, _load(args.noise)))
    text = json.dumps(result, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--port", type=int, required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--concurrency", type=int, default=1)
    c.add_argument("--prompt-positions", action="store_true")
    d = sub.add_parser("compare")
    d.add_argument("--reference", required=True)
    d.add_argument("--candidate", required=True)
    d.add_argument("--noise")
    d.add_argument("--out")
    args = parser.parse_args()
    collect(args) if args.command == "collect" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
