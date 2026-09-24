"""Paired decision experiments against an existing NInfer HTTP server (Python 3.11).

Reuses the submitted JevBench prompt without importing or modifying a JevBench checkout.
Inputs use JevBench's JSONL task schema. No labels/provenance are sent to the model.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ninfer_native import NInferNativeAdapter

METHODS = ("submitted", "shim", "native", "framed", "semantic", "rotated",
           "two_order", "short", "anthropic_short", "reason_1024", "reason_2048")


@dataclass
class Task:
    id: str
    family: str
    state: object
    question: dict
    labels: list[str]
    expected: str
    group: str
    cohort: str

    @classmethod
    def read(cls, row, cohort):
        labels = [str(x) for x in row["labels"]]
        expected = str(row["expected"])
        question = row["question"]
        kind = question["type"]
        if (not 2 <= len(labels) <= 26 or len(set(labels)) != len(labels)
                or expected not in labels):
            raise ValueError(f"{row['id']}: invalid labels or ground truth")
        if kind == "noul" and set(labels) != {"no", "yes"}:
            raise ValueError("noul labels must be no/yes")
        if kind == "score" and labels != [str(i) for i in range(len(labels))]:
            raise ValueError("score labels must preserve ascending level order")
        if kind == "score" and len(labels) > 10:
            raise ValueError("native score supports at most ten levels")
        criteria = question.get("criteria")
        if kind == "choice" and (not isinstance(criteria, dict) or set(criteria) != set(labels)):
            raise ValueError("choice criteria must match labels")
        if kind == "score" and (not isinstance(criteria, list) or len(criteria) != len(labels)):
            raise ValueError("score criteria must match levels")
        if kind not in {"noul", "choice", "score"}:
            raise ValueError(f"unknown question type {kind}")
        if row.get("provenance", {}).get("exclude_reason"):
            raise ValueError(f"{row['id']}: excluded task cannot enter this labeled study")
        return cls(row["id"], row["family"], row["state"], question, labels,
                   expected, row.get("group") or row["id"], cohort)


def load_tasks(inputs):
    tasks, sources = [], []
    for spec in inputs:
        cohort, filename = spec.split("=", 1)
        if not cohort:
            raise ValueError("use --tasks COHORT=path.jsonl")
        data = Path(filename).read_bytes()
        sources.append({"cohort": cohort, "path": str(Path(filename).resolve()),
                        "sha256": hashlib.sha256(data).hexdigest()})
        tasks.extend(Task.read(json.loads(line), cohort)
                     for line in data.decode("utf-8-sig").splitlines() if line.strip())
    keys = [(t.cohort, t.id) for t in tasks]
    if not tasks or len(set(keys)) != len(keys):
        raise ValueError("empty data or duplicate task IDs within a cohort")
    return tasks, sources


def select_tasks(tasks, groups_per_family, seed):
    """Sample whole scenario groups within cohort/family; never choose by model outcomes."""
    if groups_per_family is None:
        return tasks
    buckets = defaultdict(dict)
    group_families = {}
    for task in tasks:
        group_key = (task.cohort, task.group)
        previous = group_families.setdefault(group_key, task.family)
        if previous != task.family:
            raise ValueError("a scenario group crosses families; cannot stratify it")
        buckets[(task.cohort, task.family)].setdefault(task.group, []).append(task)
    chosen = []
    for key, groups in sorted(buckets.items()):
        ordered = sorted(groups, key=lambda g: hashlib.sha256(
            f"{seed}:{key}:{g}".encode()).digest())
        chosen.extend(t for group in ordered[:groups_per_family] for t in groups[group])
    return chosen


def mapped_task(task, method):
    result = copy.deepcopy(task)
    if method == "shim" and task.question["type"] == "choice":
        result.labels = list(task.question["criteria"])
    if method == "rotated":
        result.labels = task.labels[1:] + task.labels[:1]
    return result


def chat_request(task, method, model):
    mapped = mapped_task(task, method)
    body = NInferNativeAdapter(model=model).build_request(mapped)
    tokens = body["logprob_candidates"]
    if method in {"framed", "semantic"}:
        opts = NInferNativeAdapter.options(mapped)
        if method == "semantic":
            tokens = [{"yes": "Yes", "no": "No"}[label] for label in mapped.labels] \
                if task.question["type"] == "noul" else mapped.labels
        state = task.state if isinstance(task.state, str) else json.dumps(task.state, ensure_ascii=False, indent=1)
        rows = [f"{token}. {label}: {description}" for token, (label, description) in zip(tokens, opts)]
        ordinal = " (levels ordered lowest to highest)" if task.question["type"] == "score" else ""
        body["messages"] = [
            {"role": "system", "content": "Evaluate the evidence against the criterion. Reply with only the selected option token."},
            {"role": "user", "content": f"Evidence:\n{state}\n\nCriterion:\n{task.question['instructions']}\n\nOptions{ordinal}:\n"
             + "\n".join(rows) + "\n\nSelected option token:"},
        ]
        body["logprob_candidates"] = tokens
    if method == "short":
        body["max_tokens"] = 64
        for field in ("logprobs", "top_logprobs", "logprob_candidates"):
            body.pop(field)
    return body, dict(zip(tokens, mapped.labels))


def native_request(task, model):
    question = copy.deepcopy(task.question)
    if question["type"] == "choice":
        question["criteria"] = {label: question["criteria"][label] for label in task.labels}
    return {"model": model, "state": task.state, "questions": {"q": question}, "temperature": 1.0}


def anthropic_request(task, method, model):
    chat, mapping = chat_request(task, "submitted", model)
    budget = int(method.split("_")[1]) if method.startswith("reason_") else None
    return {
        "model": model, "system": chat["messages"][0]["content"],
        "messages": chat["messages"][1:], "temperature": 0,
        "max_tokens": budget + 256 if budget else 64,
        "thinking": {"type": "enabled", "budget_tokens": budget} if budget else {"type": "disabled"},
    }, mapping


def distribution(values, labels):
    if set(values) != set(labels):
        raise ValueError("probability labels differ from declared labels")
    result = {label: float(values[label]) for label in labels}
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in result.values()):
        raise ValueError("invalid probability")
    if not math.isclose(sum(result.values()), 1, abs_tol=1e-5):
        raise ValueError("probabilities do not sum to one")
    return result


def read_candidates(response, mapping):
    entry = response["choices"][0]["logprobs"]["content"][0]
    if entry.get("forced"):
        raise ValueError("forced position has no model distribution")
    candidates = entry["candidate_logprobs"]
    if [c["token"] for c in candidates] != list(mapping):
        raise ValueError("candidate tokens/order differ from request")
    logits = [float(c["raw_logprob"]) for c in candidates]
    if any(not math.isfinite(x) or x > 1e-6 for x in logits) or max(logits) <= -9999:
        raise ValueError("invalid or unavailable candidate logprobs")
    weights = [math.exp(x - max(logits)) for x in logits]
    total = sum(weights)
    mass = sum(math.exp(x) for x in logits)
    if mass > 1.00001:
        raise ValueError("candidate vocabulary mass exceeds one")
    return {label: weight / total for label, weight in zip(mapping.values(), weights)}, max(0, 1 - mass)


def parse_answer(text, mapping):
    # Accept only a complete answer token, optionally followed by ordinary option punctuation.
    match = re.fullmatch(r"\s*([A-Z])(?:[.)])?\s*", text)
    if not match or match[1] not in mapping:
        raise ValueError(f"answer is not exactly one declared option: {text[:120]!r}")
    return mapping[match[1]]


class Client:
    def __init__(self, endpoint, key_env="", timeout=180):
        self.endpoint = endpoint.rstrip("/").removesuffix("/v1")
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
        if key_env:
            self.headers["Authorization"] = "Bearer " + os.environ[key_env]

    def request(self, route, body=None):
        request = urllib.request.Request(self.endpoint + route, headers=self.headers,
                                         data=json.dumps(body, ensure_ascii=False).encode() if body is not None else None)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)


def evaluate(task, method, model, client):
    row = {"task_id": task.id, "group": task.group, "family": task.family,
           "cohort": task.cohort, "method": method, "expected": task.expected,
           "ok": False, "correct": False, "probs": None, "requests": []}
    if (method in {"rotated", "two_order"} and task.question["type"] != "choice"
            or method == "semantic" and task.question["type"] == "choice"):
        return {**row, "skipped": True, "skip_reason": "method not applicable to this question type"}
    started = time.perf_counter()
    try:
        if method == "native":
            route, body, mapping = "/v1/systemone", native_request(task, model), None
        elif method.startswith("reason_") or method == "anthropic_short":
            body, mapping = anthropic_request(task, method, model)
            route = "/v1/messages"
        else:
            body, mapping = chat_request(task, method, model)
            route = "/v1/chat/completions"
        call = {"route": route, "body": body}
        row["requests"].append(call)
        response = client.request(route, body)
        call["response"] = response
        if response.get("model") != model:
            raise ValueError(f"response model differs from target: {response.get('model')!r}")
        row["usage"] = response.get("usage", {})
        if method == "native":
            answer = response["answers"]["q"]
            values = {"yes": answer["noul"], "no": 1 - answer["noul"]} \
                if task.question["type"] == "noul" else answer["probabilities"]
            row["probs"] = distribution(values, task.labels)
        elif method == "short":
            choice = response["choices"][0]
            row["finish_reason"] = choice["finish_reason"]
            if choice["finish_reason"] != "stop":
                raise ValueError("short answer did not complete normally")
            row["answer"] = parse_answer(choice["message"].get("content") or "", mapping)
        elif route == "/v1/messages":
            row["finish_reason"] = response["stop_reason"]
            if response["stop_reason"] != "end_turn":
                raise ValueError("generated answer did not complete normally")
            text = "".join(block["text"] for block in response["content"] if block["type"] == "text")
            row["answer"] = parse_answer(text, mapping)
        else:
            row["probs"], row["outside_mass"] = read_candidates(response, mapping)
        if row["probs"] is not None:
            row["probs"] = distribution(row["probs"], task.labels)
            row["answer"] = max(task.labels, key=row["probs"].get)
        row["ok"] = True
        row["correct"] = row["answer"] == task.expected
    except (ValueError, KeyError, IndexError, TypeError, OSError) as error:
        row["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, urllib.error.HTTPError):
            row["http_error"] = error.read().decode("utf-8", errors="replace")[:4000]
    row["latency_s"] = time.perf_counter() - started
    return row


def average_orders(task, first, second):
    """Derived two-pass method; charge both measured passes, including any failure."""
    row = {k: first[k] for k in ("task_id", "group", "family", "cohort", "expected")}
    row.update(method="two_order", ok=False, correct=False, probs=None,
               latency_s=first["latency_s"] + second["latency_s"],
               components=["submitted", "rotated"])
    usages = [r.get("usage", {}) for r in (first, second)]
    row["usage"] = {key: sum(u[key] for u in usages) if all(isinstance(u.get(key), int) for u in usages) else None
                    for key in ("prompt_tokens", "completion_tokens")}
    if first["ok"] and second["ok"]:
        row["probs"] = {label: (first["probs"][label] + second["probs"][label]) / 2 for label in task.labels}
        row["answer"] = max(task.labels, key=row["probs"].get)
        row["ok"] = True
        row["correct"] = row["answer"] == task.expected
        row["order_flip"] = first["answer"] != second["answer"]
    else:
        row["error"] = "one or both component reads failed"
    return row


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * fraction
    low = math.floor(index)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)


def metrics(rows):
    rows = [row for row in rows if not row.get("skipped")]
    probabilities = [row for row in rows if row["ok"] and row.get("probs") is not None]
    bins = [[] for _ in range(10)]
    for row in probabilities:
        confidence = max(row["probs"].values())
        bins[min(9, int(confidence * 10))].append((confidence, row["correct"]))
    def tokens(chat_key, other_key):
        values = []
        for row in rows:
            usage = row.get("usage", {})
            value = usage.get(chat_key, usage.get(other_key))
            if other_key == "input_tokens" and "prompt_tokens" not in usage and isinstance(value, int):
                # Anthropic reports uncached input separately; Chat prompt_tokens includes it.
                value += sum(usage.get(key) or 0 for key in
                             ("cache_read_input_tokens", "cache_creation_input_tokens"))
            values.append(value)
        return [value for value in values if isinstance(value, int) and value >= 0]
    input_tokens = tokens("prompt_tokens", "input_tokens")
    output_tokens = tokens("completion_tokens", "output_tokens")
    return {
        "n": len(rows), "valid": sum(r["ok"] for r in rows), "correct": sum(r["correct"] for r in rows),
        "accuracy_failures_wrong": statistics.fmean(r["correct"] for r in rows) if rows else None,
        "probability_n": len(probabilities),
        "nll_valid_only": statistics.fmean(-math.log(max(r["probs"][r["expected"]], 1e-300)) for r in probabilities) if probabilities else None,
        "brier_valid_only": statistics.fmean(sum((p - (label == r["expected"])) ** 2 for label, p in r["probs"].items()) for r in probabilities) if probabilities else None,
        "ece_10_bins_valid_only": sum(len(bucket) * abs(statistics.fmean(x[0] for x in bucket) - statistics.fmean(x[1] for x in bucket)) for bucket in bins if bucket) / len(probabilities) if probabilities else None,
        "p50_s": percentile([r["latency_s"] for r in rows], .5),
        "p95_s": percentile([r["latency_s"] for r in rows], .95),
        "input_usage_n": len(input_tokens), "output_usage_n": len(output_tokens),
        "mean_input_tokens": statistics.fmean(input_tokens) if input_tokens else None,
        "mean_output_tokens": statistics.fmean(output_tokens) if output_tokens else None,
    }


def paired(rows, baseline="submitted"):
    lookup = {(r["cohort"], r["task_id"], r["method"]): r for r in rows if not r.get("skipped")}
    result = {}
    for method in sorted({r["method"] for r in rows} - {baseline}):
        pairs = [(lookup[(r["cohort"], r["task_id"], baseline)], r) for r in rows
                 if r["method"] == method and not r.get("skipped")
                 and (r["cohort"], r["task_id"], baseline) in lookup]
        if not pairs:
            continue
        groups = defaultdict(list)
        for base, candidate in pairs:
            groups[(base["cohort"], base["group"])].append(int(candidate["correct"]) - int(base["correct"]))
        clusters = list(groups.values())
        rng = random.Random(0)
        deltas = []
        for _ in range(2000):
            sample = [value for cluster in rng.choices(clusters, k=len(clusters)) for value in cluster]
            deltas.append(statistics.fmean(sample))
        result[method] = {
            "baseline": baseline, "n": len(pairs), "scenario_groups": len(groups),
            "fixed": sum(not a["correct"] and b["correct"] for a, b in pairs),
            "broken": sum(a["correct"] and not b["correct"] for a, b in pairs),
            "delta_accuracy": statistics.fmean(int(b["correct"]) - int(a["correct"]) for a, b in pairs),
            "scenario_bootstrap_95": [percentile(deltas, .025), percentile(deltas, .975)],
        }
    return result


def summarize(rows):
    cohorts = {}
    for cohort in sorted({r["cohort"] for r in rows}):
        selected = [r for r in rows if r["cohort"] == cohort]
        methods = sorted({r["method"] for r in selected})
        cohorts[cohort] = {
            "methods": {m: metrics([r for r in selected if r["method"] == m]) for m in methods},
            "paired_vs_submitted": paired(selected),
            "controls": {"reasoning_vs_anthropic_short": paired(selected, "anthropic_short"),
                         "semantic_vs_framed": paired(selected, "framed")},
            "families": {family: {m: metrics([r for r in selected if r["family"] == family and r["method"] == m]) for m in methods}
                         for family in sorted({r["family"] for r in selected})},
        }
    return {"cohorts": cohorts, "note": "Pilot, not an official JevBench score. Probability metrics exclude invalid reads; accuracy counts every applicable failure as wrong. Generated answers have no probability metrics. Two-order latency sums separately measured passes."}


def write_report(summary, path):
    lines = ["# Paired decision study", "", summary["note"], ""]
    for cohort, data in summary["cohorts"].items():
        lines += [f"## {cohort}", "", "| Method | Correct / attempted | Invalid | p50 / p95 seconds | Fixed / broken vs submitted |", "|---|---:|---:|---:|---:|"]
        for method, m in data["methods"].items():
            if not m["n"]:
                continue
            p = data["paired_vs_submitted"].get(method)
            changes = f"{p['fixed']} / {p['broken']}" if p else "—"
            lines.append(f"| {method} | {m['correct']} / {m['n']} | {m['n'] - m['valid']} | {m['p50_s']:.3f} / {m['p95_s']:.3f} | {changes} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", action="append", required=True, metavar="COHORT=PATH")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8010")
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifact", required=True, help="explicit descriptive artifact identity/format for provenance")
    parser.add_argument("--out", type=Path, required=True, help="new run directory")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--groups-per-family", type=int)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--key-env", default="")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--plan", action="store_true", help="print selection and request count without writing or contacting a server")
    args = parser.parse_args(argv)
    if args.groups_per_family is not None and args.groups_per_family < 1:
        parser.error("groups-per-family must be positive")
    tasks, sources = load_tasks(args.tasks)
    tasks = select_tasks(tasks, args.groups_per_family, args.seed)
    methods = list(dict.fromkeys(args.methods))
    if "two_order" in methods:
        methods = list(dict.fromkeys(["submitted", "rotated"] + methods))
    planned = [(t, m) for t in tasks for m in methods if m != "two_order"
               and not (m == "rotated" and t.question["type"] != "choice")
               and not (m == "semantic" and t.question["type"] == "choice")]
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "endpoint": args.endpoint,
                "model": args.model, "artifact": args.artifact, "sources": sources, "seed": args.seed,
                "methods": methods, "requests_planned": len(planned), "concurrency": 1,
                "selection": [{"cohort": t.cohort, "id": t.id, "group": t.group, "family": t.family} for t in tasks],
                "reasoning_readout": "strict parsed option; per-request Engine cap via Anthropic; no fabricated probabilities",
                "latency_scope": "serial client wall time on the existing service; no exclusive GPU claim"}
    if args.plan:
        print(json.dumps(manifest, indent=2))
        return 0
    client = Client(args.endpoint, args.key_env, args.timeout)
    models = client.request("/v1/models")
    if args.model not in [m["id"] for m in models["data"]]:
        raise ValueError("requested model is not served by this endpoint")
    manifest["server_models"] = models
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows = []
    rng = random.Random(args.seed)
    rng.shuffle(tasks)
    with (args.out / "results.jsonl").open("w", encoding="utf-8") as output:
        for index, task in enumerate(tasks):
            task_methods = [m for m in methods if m != "two_order"]
            rng.shuffle(task_methods)
            task_rows = {}
            for method in task_methods:
                row = evaluate(task, method, args.model, client)
                rows.append(row)
                task_rows[method] = row
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
            if "two_order" in methods and task.question["type"] == "choice":
                row = average_orders(task, task_rows["submitted"], task_rows["rotated"])
                rows.append(row)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
            print(f"[{index + 1}/{len(tasks)}] {task.cohort}/{task.id}: " + ", ".join(f"{m}={'ok' if r['correct'] else 'wrong' if r['ok'] else 'skip' if r.get('skipped') else 'INVALID'}" for m, r in task_rows.items()), flush=True)
    summary = summarize(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, args.out / "report.md")
    print(f"Report: {args.out / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
