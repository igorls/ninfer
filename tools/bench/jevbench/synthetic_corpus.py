"""NInfer teacher -> bounded scenario specifications -> executable-gold decision corpus.

The teacher authors data, never executable code or trusted answer labels. Every
numeric answer is recomputed by the oracle, then rendered from that same data.
All mutations of one blueprint share a split group. Python 3.11, stdlib only.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import heapq
import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

KINDS = ("ledger", "policy", "route", "allocation")
DEPTHS = {"ledger": (3, 8, 18, 32), "policy": (3, 8, 16, 28),
          "route": (5, 8, 12, 16), "allocation": (4, 8, 14, 22)}
OPS = ("add", "subtract", "multiply", "floor_divide", "min", "max")
COMPARISONS = ("lt", "le", "eq", "ne", "ge", "gt")
CAP = 1_000_000_000


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def integer(low, high):
    return {"type": "integer", "minimum": low, "maximum": high}


def enum(values):
    return {"type": "string", "enum": list(values)}


def obj(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def array(items, low, high=None):
    return {"type": "array", "items": items, "minItems": low, "maxItems": low if high is None else high}


def schema(kind, depth):
    title = {"title": {"type": "string"}}
    if kind == "ledger":
        return obj({**title, "initial": integer(-100, 800),
                    "steps": array(obj({"op": enum(OPS), "value": integer(1, 50)}), depth)})
    if kind == "policy":
        condition = obj({"field": integer(0, 7), "comparison": enum(COMPARISONS), "threshold": integer(0, 50)})
        return obj({**title, "facts": array(integer(0, 50), 8),
                    "rules": array(obj({"tests": array(condition, 1, 3), "action": integer(0, 3)}), depth),
                    "fallback": integer(0, 3)})
    if kind == "route":
        edge = obj({"source": integer(0, depth - 1), "target": integer(0, depth - 1), "cost": integer(1, 90)})
        return obj({**title, "nodes": {"type": "integer", "const": depth},
                    "edges": array(edge, depth * 2, depth * 3),
                    "blocked": array(integer(1, depth - 2), 0, 2)})
    return obj({**title, "capacity": integer(20, 160),
                "items": array(obj({"cost": integer(1, 50), "reward": integer(1, 100)}), depth)})


def validate_schema(value, contract):
    """Independent local validation of the deliberately small emitted schema subset."""
    kind = contract["type"]
    expected = {"integer": int, "object": dict, "array": list, "string": str}[kind]
    if type(value) is not expected:
        raise ValueError(f"expected {kind}")
    if "const" in contract and value != contract["const"]:
        raise ValueError("constant mismatch")
    if "enum" in contract and value not in contract["enum"]:
        raise ValueError("enum mismatch")
    if kind == "integer" and not contract.get("minimum", value) <= value <= contract.get("maximum", value):
        raise ValueError("integer out of bounds")
    if kind == "object":
        if set(value) != set(contract["properties"]):
            raise ValueError("missing or unexpected fields")
        for key, child in contract["properties"].items():
            validate_schema(value[key], child)
    if kind == "array":
        if not contract["minItems"] <= len(value) <= contract["maxItems"]:
            raise ValueError("array size out of bounds")
        for child in value:
            validate_schema(child, contract["items"])


def validate(kind, spec, depth):
    validate_schema(spec, schema(kind, depth))
    if len(spec["title"]) > 160:
        raise ValueError("title too long")
    if kind == "route":
        pairs = [(edge["source"], edge["target"]) for edge in spec["edges"]]
        if len(set(pairs)) != len(pairs) or any(a == b for a, b in pairs):
            raise ValueError("duplicate or self graph edge")
        if len(set(spec["blocked"])) != len(spec["blocked"]):
            raise ValueError("duplicate blocked node")


def compare(a, operation, b):
    return {"lt": a < b, "le": a <= b, "eq": a == b,
            "ne": a != b, "ge": a >= b, "gt": a > b}[operation]


def solve(kind, spec):
    if kind == "ledger":
        value = spec["initial"]
        for step in spec["steps"]:
            operand, operation = step["value"], step["op"]
            if operation == "add": value += operand
            elif operation == "subtract": value -= operand
            elif operation == "multiply": value *= operand
            elif operation == "floor_divide": value //= operand
            elif operation == "min": value = min(value, operand)
            elif operation == "max": value = max(value, operand)
            else: raise ValueError("unknown ledger operation")
            value = max(-CAP, min(CAP, value))
        return str(value)
    if kind == "policy":
        for rule in spec["rules"]:
            if all(compare(spec["facts"][t["field"]], t["comparison"], t["threshold"]) for t in rule["tests"]):
                return str(rule["action"])
        return str(spec["fallback"])
    if kind == "route":
        adjacency = [[] for _ in range(spec["nodes"])]
        blocked = set(spec["blocked"])
        for edge in spec["edges"]:
            if edge["source"] not in blocked and edge["target"] not in blocked:
                adjacency[edge["source"]].append((edge["target"], edge["cost"]))
        distances, queue = {0: 0}, [(0, 0)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != distances[node]: continue
            if node == spec["nodes"] - 1: return str(distance)
            for target, cost in adjacency[node]:
                candidate = distance + cost
                if candidate < distances.get(target, float("inf")):
                    distances[target] = candidate
                    heapq.heappush(queue, (candidate, target))
        return "unreachable"
    if kind != "allocation": raise ValueError("unknown scenario kind")
    best = [0] * (spec["capacity"] + 1)
    for item in spec["items"]:
        for available in range(spec["capacity"], item["cost"] - 1, -1):
            best[available] = max(best[available], best[available - item["cost"]] + item["reward"])
    return str(best[-1])


def mutate(kind, original, variant, rng):
    spec = copy.deepcopy(original)
    if variant == 0: return spec
    if kind == "ledger":
        spec["initial"] = rng.randint(-100, 800)
        for step in spec["steps"]:
            step["value"] = rng.randint(1, 5 if step["op"] in ("multiply", "floor_divide") else 50)
    elif kind == "policy":
        spec["facts"] = [rng.randint(0, 50) for _ in range(8)]
        # Boundary-focused siblings, keeping every relative in the same split group.
        rule = spec["rules"][variant % len(spec["rules"])]
        for test in rule["tests"]:
            spec["facts"][test["field"]] = max(0, min(50, test["threshold"] + (-1, 0, 1)[variant % 3]))
    elif kind == "route":
        for edge in spec["edges"]: edge["cost"] = rng.randint(1, 90)
        spec["blocked"] = rng.sample(range(1, spec["nodes"] - 1), variant % 3)
    else:
        spec["capacity"] = rng.randint(20, 160)
        for item in spec["items"]:
            item["cost"], item["reward"] = rng.randint(1, 50), rng.randint(1, 100)
    return spec


def render(kind, spec, variant):
    data = {key: value for key, value in spec.items() if key != "title"}
    # The teacher's free-text title is audit metadata only, never an instruction or factual premise.
    compact = variant % 2 == 0
    state = json.dumps(data, indent=None if compact else 2)
    instructions = {
        "ledger": "Start with initial. Apply steps in their listed order. Each step updates the current value using its op and value. add/subtract/multiply have ordinary integer meanings; floor_divide divides then rounds toward negative infinity; min/max takes the smaller/larger of the current value and the operand. Immediately after EVERY step clamp the result to [-1000000000,1000000000]. What is the final integer?",
        "policy": "facts is an array indexed 0 through 7. Examine rules in listed order. A rule matches only if ALL its tests hold. A test compares facts[field] with threshold: lt <, le <=, eq ==, ne !=, ge >=, gt >. Return the action of the FIRST matching rule; later rules cannot override it. If none matches return fallback. Which action applies?",
        "route": "The directed graph has nodes 0 through nodes-1. Each edge goes source -> target and adds cost. Blocked nodes and ALL edges touching them are unavailable. Find the minimum total cost from node 0 to node nodes-1; a route may use multiple edges. If no route exists choose unreachable. What is the minimum cost?",
        "allocation": "Each item is a distinct project and may be chosen at most once, wholly or not at all. Any subset (including empty) is allowed. Its total cost must be <= capacity. Rewards add. Maximize total reward; unused capacity earns nothing. What is the maximum achievable reward?",
    }
    return state, instructions[kind]


def to_task(kind, spec, blueprint, variant, seed):
    expected = solve(kind, spec)
    rng = random.Random(fingerprint([seed, blueprint, variant]))
    if kind == "policy":
        options = ["0", "1", "2", "3"]
    else:
        anchor = int(expected) if expected != "unreachable" else rng.randint(10, 150)
        options = [expected]
        if kind == "route" and expected != "unreachable": options.append("unreachable")
        while len(options) < 4:
            candidate = str(anchor + rng.choice([-1, 1]) * rng.randint(1, max(12, abs(anchor) // 3)))
            if kind != "ledger" and int(candidate) < 0: continue
            if candidate not in options: options.append(candidate)
    rng.shuffle(options)
    state, instructions = render(kind, spec, variant)
    identity = fingerprint([kind, state, instructions, options])
    return {"id": f"synthetic-{identity}", "family": f"synthetic_{kind}",
            "group": f"blueprint-{blueprint}", "source_group": f"blueprint-{blueprint}",
            "state": state, "question": {"type": "choice", "instructions": instructions,
                "criteria": {value: f"Result is {value}." for value in options}},
            "labels": options, "expected": expected,
            "provenance": {"source": "NInfer Flash-Next scenario author with executable oracle",
                "seed": seed, "blueprint": blueprint, "variant": variant,
                "oracle": f"synthetic_corpus.solve/{kind}", "license": "MIT"}}


def request_json(endpoint, body, timeout=240):
    request = urllib.request.Request(endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def generate_one(endpoint, model, slot, seed):
    kind = KINDS[slot % len(KINDS)]
    size_level = (slot // len(KINDS)) % 4
    depth = DEPTHS[kind][size_level]
    guidance = {
        "ledger": "Make order matter. Mix add, subtract, multiply and floor_divide. Avoid repeated min/max resets that erase the earlier work.",
        "policy": "Design competing overlapping rules and near-threshold facts. The first matching rule wins; include exception-like early rules and plausible later distractors.",
        "route": "Use distinct directed non-self edges, branches, cycles and competing multi-hop routes. Do not repeat a source/target pair. Block at most two intermediate nodes, without duplicate blocks.",
        "allocation": "Make the best portfolio require tradeoffs: reward/cost greedy ranking should not trivially solve every instance. Mix costs and rewards rather than sorting them.",
    }
    base = {"model": model, "temperature": 0.9, "max_tokens": 6144,
            "chat_template_kwargs": {"enable_thinking": False}, "prompt_cache_read_only": True,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "scenario", "strict": True, "schema": schema(kind, depth)}}}
    attempts = []
    for attempt in range(3):
        body = {**base, "seed": seed + slot * 3 + attempt, "messages": [
            {"role": "system", "content": "Author diverse synthetic decision scenarios as the exact requested JSON schema. Return data only. Do not include answers, code, explanations, or instructions in the title."},
            {"role": "user", "content": f"Scenario kind {kind}; size {depth}; variant seed {seed + slot * 3 + attempt}. {guidance[kind]} Create an original coherent instance with a short descriptive title."}]}
        started = time.monotonic()
        record = {"request": body}
        try:
            response = request_json(endpoint, body)
            record["response"] = response
            if response["model"] != model: raise ValueError("served model differs from requested teacher")
            choice = response["choices"][0]
            if choice["finish_reason"] != "stop": raise ValueError("incomplete teacher output")
            spec = json.loads(choice["message"]["content"])
            validate(kind, spec, depth)
            identity = fingerprint([kind, {k: v for k, v in spec.items() if k != "title"}])
            return {"slot": slot, "seed": seed, "kind": kind, "depth": depth, "id": identity,
                    "spec": spec, "attempts": attempts, "ok": True}
        except (ValueError, KeyError, urllib.error.URLError, TimeoutError) as error:
            record["error"] = str(error)
        finally:
            record["seconds"] = time.monotonic() - started
            attempts.append(record)
    return {"slot": slot, "seed": seed, "kind": kind, "depth": depth, "attempts": attempts, "ok": False}


def generate(args):
    args.out.mkdir(parents=True, exist_ok=True)
    contract = {"schema": 1, "model": args.model, "teacher_artifact_sha256": args.teacher_artifact_sha256,
                "seed": args.seed, "variants": args.variants,
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "blueprints": args.blueprints}
    manifest = args.out / "manifest.json"
    path = args.out / "blueprints.jsonl"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        previous_count = previous.pop("blueprints")
        if previous != {key: value for key, value in contract.items() if key != "blueprints"}:
            raise ValueError("resume teacher, seed, generator or variant contract differs")
        if previous_count > args.blueprints:
            raise ValueError("blueprint count may only increase on resume")
    elif path.exists() and path.stat().st_size:
        raise ValueError("existing blueprints require their original manifest")
    existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    if any(row["seed"] != args.seed for row in existing): raise ValueError("resume seed differs")
    manifest.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    done = {row["slot"] for row in existing if row["ok"]}
    pending = [slot for slot in range(args.blueprints) if slot not in done]
    with path.open("a", encoding="utf-8") as output, concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        jobs = {pool.submit(generate_one, args.endpoint, args.model, slot, args.seed): slot for slot in pending}
        for future in concurrent.futures.as_completed(jobs):
            result = future.result()
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            print(json.dumps({"slot": result["slot"], "kind": result["kind"], "ok": result["ok"]}), flush=True)
    compile_corpus(path, args.out, args.variants, args.seed)


def compile_corpus(path, out, variants, seed):
    blueprints, tasks, seen, kinds, sizes, slots = {}, [], set(), Counter(), Counter(), {}
    attempts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "slot" in row: slots[row["slot"]] = row["ok"]
        attempts.extend(row.get("attempts", []))
        if row["ok"]: blueprints[row["id"]] = row
    for identity, row in sorted(blueprints.items()):
        kind = row["kind"]
        for variant in range(variants):
            spec = mutate(kind, row["spec"], variant, random.Random(fingerprint([seed, identity, variant])))
            validate(kind, spec, row["depth"])
            task = to_task(kind, spec, identity, variant, seed)
            # Option order and formatting do not turn an identical logical scenario into new data.
            signature = fingerprint([kind, {k: v for k, v in spec.items() if k != "title"}])
            if signature in seen: continue
            seen.add(signature)
            tasks.append(task)
            kinds[kind] += 1
            sizes[f"{kind}:{row['depth']}"] += 1
    tasks.sort(key=lambda row: fingerprint([seed, row["id"]]))
    with (out / "tasks.jsonl").open("w", encoding="utf-8") as output:
        for task in tasks: output.write(json.dumps(task, ensure_ascii=False) + "\n")
    report = {"schema": 1, "seed": seed, "teacher_blueprints": len(blueprints), "scenarios": len(tasks),
              "groups": len({r["group"] for r in tasks}), "variants_per_blueprint": variants,
              "family_counts": dict(kinds), "size_counts": dict(sizes),
              "teacher_requests": len(attempts),
              "rejected_attempts": sum("error" in attempt for attempt in attempts),
              "failed_slots": sorted(slot for slot, ok in slots.items() if not ok),
              "teacher_output_tokens": sum(attempt.get("response", {}).get("usage", {}).get("completion_tokens", 0) for attempt in attempts),
              "teacher_request_seconds_sum": sum(attempt["seconds"] for attempt in attempts),
              "distinct_answers_per_family": {
                  kind: len({r["expected"] for r in tasks if r["family"] == f"synthetic_{kind}"}) for kind in KINDS},
              "target_labels_collected": False,
              "limitation": "Generated inputs and executable gold only; Qwen NVFP4 outcomes and routing gains require paired collection."}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8010")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--teacher-artifact-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blueprints", type=int, default=64)
    parser.add_argument("--variants", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    if args.blueprints < 1 or args.variants < 1 or not 1 <= args.concurrency <= 8:
        parser.error("positive counts and concurrency 1..8 required")
    if len(args.teacher_artifact_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.teacher_artifact_sha256):
        parser.error("teacher artifact SHA-256 must be 64 lowercase hex digits")
    generate(args)


if __name__ == "__main__":
    main()
