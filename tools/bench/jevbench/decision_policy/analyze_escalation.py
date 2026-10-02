"""Escalation analysis with budgets and alternative uncertainty signals (local).

Base distribution: letter-rotation average (avg_rot) when available, else native. Signals:
  conf        max probability of the base distribution
  margin      top-1 minus top-2
  disagree    largest per-option difference among native, reversed-order and rotated readouts
  entropy     normalized entropy
For each signal and each escalation budget (10, 20, 30, 50, 100% of items), the threshold is the
dev-split quantile that meets the budget; public is then reported at that threshold. Escalated
items take the reasoning answer (one-hot mixed with the base distribution at weight w chosen on
dev). Also reports the reasoning route alone. Latency: VM-measured and a single-stream estimate
(native 0.06 s, reasoning 2.3 s from the decision study)."""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path.cwd()
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools/bench/jevbench"))
import decision_study as study  # noqa: E402

COLLECT = Path(sys.argv[1] if len(sys.argv) > 1 else "run_collect")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "run_calibration")
BASELINE = Path(sys.argv[3]) if len(sys.argv) > 3 else REPO / "profiles/bench/jevbench-native-20260927/run/results.jsonl"
REFERENCE = Path(sys.argv[4]) if len(sys.argv) > 4 else REPO / "profiles/bench/jevbench-clef-20261002/run/results.jsonl"


def load(name):
    rows = []
    for l in (COLLECT / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            try:
                rows.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return rows


def items(name):
    table = defaultdict(dict)
    for r in load(name):
        table[r["task_id"]][r["method"]] = r
    out = []
    for tid, e in table.items():
        nat = e.get("native")
        if not nat or not nat["ok"]:
            continue
        labels = list(nat["probs"])
        v = {"native": [nat["probs"][l] for l in labels]}
        for m in ("native_reversed", "rotated", "framed", "semantic"):
            r = e.get(m)
            if r and r["ok"] and r.get("probs") and set(r["probs"]) == set(labels):
                v[m] = [r["probs"][l] for l in labels]
        base = v["native"] if "rotated" not in v else [(a + b) / 2 for a, b in zip(v["native"], v["rotated"])]
        s = sorted(base, reverse=True)
        n = len(base)
        disagree = max([max(abs(a - b) for a, b in zip(v["native"], v[m])) for m in v if m != "native"] + [0.0])
        reason = e.get("reason_1024")
        out.append({"id": tid, "cohort": nat["cohort"], "group": nat.get("group", tid), "family": nat["family"], "labels": labels,
                    "expected": nat["expected"], "base": base, "variants": v,
                    "signals": {"conf": s[0], "margin": s[0] - (s[1] if n > 1 else 0.0), "disagree": disagree,
                                "entropy": -sum(x * math.log(max(x, 1e-12)) for x in base) / math.log(max(n, 2))},
                    "reason": reason["answer"] if reason and reason["ok"] and reason["answer"] in labels else None,
                    "reason_ok": bool(reason and reason["ok"]), "lat_native": nat["latency_s"],
                    "lat_reason": reason["latency_s"] if reason and reason["ok"] else None})
    return out


def score(probs, it):
    gold = it["labels"].index(it["expected"])
    answer = max(range(len(probs)), key=lambda i: probs[i])
    brier = sum((p - (1.0 if i == gold else 0.0)) ** 2 for i, p in enumerate(probs))
    return answer == gold, brier


def evaluate(selected, signal, threshold, weight, lower_is_uncertain):
    correct, brier, esc, lat_vm, lat_est = 0, 0.0, 0, 0.0, 0.0
    for it in selected:
        value = it["signals"][signal]
        uncertain = value < threshold if lower_is_uncertain else value > threshold
        probs = list(it["base"])
        lat_vm += it["lat_native"]
        lat_est += 0.06
        if uncertain and it["reason"] is not None:
            esc += 1
            lat_vm += it["lat_reason"] or 0.0
            lat_est += 2.3
            gi = it["labels"].index(it["reason"])
            probs = [(1 - weight) * p + (weight if i == gi else 0.0) for i, p in enumerate(probs)]
        ok, b = score(probs, it)
        correct += ok
        brier += b
    n = len(selected)
    return {"correct": correct, "n": n, "accuracy": correct / n, "brier": brier / n, "rate": esc / n,
            "latency_vm_s": lat_vm / n, "latency_est_s": lat_est / n}


sets = {k: items(k) for k in ("dev", "public", "oracle")}
dev = [it for it in sets["dev"] if it["reason_ok"]]
public = sets["public"]
print({k: len(v) for k, v in sets.items()}, "dev with reasoning", len(dev), "public with reasoning", sum(it["reason_ok"] for it in public))
# Reasoning alone and base alone.
for name, selected in (("public", public), ("oracle", sets["oracle"])):
    base = evaluate(selected, "conf", -1.0, 0.0, True)
    reason_only = {"correct": sum(it["reason"] == it["expected"] for it in selected if it["reason"] is not None),
                   "n": sum(it["reason"] is not None for it in selected)}
    hard = [it for it in selected if it["cohort"] == "hard"]
    print(name, "base avg_rot", f"{base['correct']}/{base['n']}", "brier", round(base["brier"], 4),
          "| reasoning alone", f"{reason_only['correct']}/{reason_only['n']}",
          "| hard reasoning alone", f"{sum(it['reason'] == it['expected'] for it in hard if it['reason'] is not None)}/{sum(it['reason'] is not None for it in hard)}",
          "| hard base", f"{sum(score(it['base'], it)[0] for it in hard)}/{len(hard)}")
report = {}
for signal, lower in (("conf", True), ("margin", True), ("disagree", False), ("entropy", False)):
    values = sorted(it["signals"][signal] for it in dev)
    for budget in (0.1, 0.2, 0.3, 0.5, 1.0):
        # Threshold = dev quantile that escalates `budget` of dev items.
        k = int(round(budget * (len(values) - 1)))
        threshold = values[k] if lower else values[len(values) - 1 - k]
        if budget >= 1.0:
            threshold = float("inf") if lower else float("-inf")
        best = None
        for weight in (0.5, 0.7, 0.85, 1.0):
            d = evaluate(dev, signal, threshold, weight, lower)
            key = d["accuracy"] - d["brier"]
            if best is None or key > best[0]:
                best = (key, weight, d)
        weight, d = best[1], best[2]
        p = evaluate(public, signal, threshold, weight, lower)
        h = evaluate([it for it in public if it["cohort"] == "hard"], signal, threshold, weight, lower)
        report[f"{signal}@{budget}"] = {"threshold": threshold, "weight": weight, "dev": d, "public": p, "hard": h}
        print(f"{signal:9s} budget {budget:<4} thr {threshold:7.3f} w {weight:<4} | dev {d['correct']}/{d['n']} brier {d['brier']:.3f} rate {d['rate']:.2f} "
              f"| public {p['correct']}/231 brier {p['brier']:.3f} rate {p['rate']:.2f} lat_est {p['latency_est_s']:.2f}s | hard {h['correct']}/111")
(OUT / "escalation_report.json").write_text(json.dumps(report, indent=1))
print("ESCALATION_DONE")

# Paired statistics for the dev-chosen gates at the 0.5 dev budget (entropy and confidence),
# plus the base average alone, against the native baseline rows and the Clef BF16 reference.
baseline = [r for r in (json.loads(l) for l in BASELINE.read_text(encoding="utf-8").splitlines() if l.strip()) if r["method"] == "native"]
reference = [dict(r, method="clef_bf16") for r in (json.loads(l) for l in REFERENCE.read_text(encoding="utf-8").splitlines() if l.strip()) if r["method"] == "clef" and r["cohort"] in ("easy", "original", "hard")]


def rows_for(name, signal, threshold, weight, lower):
    out = []
    for it in public:
        value = it["signals"][signal]
        uncertain = value < threshold if lower else value > threshold
        probs = list(it["base"])
        if uncertain and it["reason"] is not None:
            gi = it["labels"].index(it["reason"])
            probs = [(1 - weight) * p + (weight if i == gi else 0.0) for i, p in enumerate(probs)]
        p = dict(zip(it["labels"], probs))
        answer = max(it["labels"], key=p.get)
        out.append({"cohort": it["cohort"], "task_id": it["id"], "group": it["group"], "family": it["family"], "question_type": "",
                    "method": name, "expected": it["expected"], "ok": True, "correct": answer == it["expected"], "probs": p,
                    "answer": answer, "latency_s": 0.0, "usage": {}, "request": None, "response": None, "error": None})
    with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    return out


configs = {"avg_rot_only": ("conf", -1.0, 0.0, True)}
for signal, lower in (("entropy", False), ("conf", True)):
    e = report[f"{signal}@0.5"]
    configs[f"escalate_{signal}_b50"] = (signal, e["threshold"], e["weight"], lower)
paired = {}
for name, (signal, threshold, weight, lower) in configs.items():
    cand = rows_for(name, signal, threshold, weight, lower)
    rows = baseline + reference + cand
    for cohort in ("all", "hard"):
        sel = rows if cohort == "all" else [r for r in rows if r["cohort"] == cohort]
        m = {k: study.metrics([r for r in sel if r["method"] == k]) for k in ("native", "clef_bf16", name)}
        pn, pc = study.paired(sel, "native").get(name, {}), study.paired(sel, "clef_bf16").get(name, {})
        line = {"correct": {k: f"{v['correct']}/{v['n']}" for k, v in m.items()}, "brier": {k: round(v["brier_valid_only"], 4) for k, v in m.items()},
                "ece": {k: round(v["ece_10_bins_valid_only"], 4) for k, v in m.items()},
                "vs_native": [pn.get("fixed"), pn.get("broken"), [round(x, 3) for x in pn.get("scenario_bootstrap_95", [])]],
                "vs_clef_bf16": [pc.get("fixed"), pc.get("broken"), [round(x, 3) for x in pc.get("scenario_bootstrap_95", [])]]}
        paired[f"{name}/{cohort}"] = line
        print(name, cohort, json.dumps(line))
(OUT / "escalation_paired.json").write_text(json.dumps(paired, indent=1))
print("PAIRED_DONE")
