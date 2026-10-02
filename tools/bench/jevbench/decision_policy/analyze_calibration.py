"""Offline analysis of the collected native-route outputs: calibration, averaging, escalation.

Inputs: collect/{train,dev,public,oracle}.jsonl from collect.py. Fit on train, select on dev,
report public (and oracle) paired against the native baseline and the Clef BF16 reference.
Python 3.11 with torch (CPU is enough)."""
from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path.cwd()
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools/bench/jevbench"))
import decision_study as study  # noqa: E402

COLLECT = Path(sys.argv[1] if len(sys.argv) > 1 else "run_collect")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "run_calibration")
BASELINE = Path(sys.argv[3]) if len(sys.argv) > 3 else REPO / "profiles/bench/jevbench-native-20260927/run/results.jsonl"
REFERENCE = Path(sys.argv[4]) if len(sys.argv) > 4 else REPO / "profiles/bench/jevbench-clef-20261002/run/results.jsonl"
OUT.mkdir(exist_ok=True)
torch.manual_seed(0)
random.seed(0)
K = 10  # sorted-probability feature width
TYPES = {"noul": 0, "choice": 1, "score": 2}


def load(name):
    p = COLLECT / f"{name}.jsonl"
    rows = []
    if p.exists():
        for l in p.read_text(encoding="utf-8").splitlines():
            if l.strip():
                try:
                    rows.append(json.loads(l))
                except json.JSONDecodeError:
                    pass  # a line still being appended by a running collector
    return rows


def by_task(rows):
    table = defaultdict(dict)
    for r in rows:
        table[r["task_id"]][r["method"]] = r
    return table


def labels_of(row):
    return list(row["probs"])


def item(table_entry):
    """One decision: the native distribution plus optional variants and the reasoning answer."""
    native = table_entry.get("native")
    if native is None or not native["ok"]:
        return None
    labels = labels_of(native)
    kind = "noul" if set(labels) == {"no", "yes"} else ("score" if labels == [str(i) for i in range(len(labels))] else "choice")
    variants = {"native": [native["probs"][l] for l in labels]}
    for name in ("native_reversed", "rotated"):
        r = table_entry.get(name)
        if r and r["ok"] and r["probs"] is not None and set(r["probs"]) == set(labels):
            variants[name] = [r["probs"][l] for l in labels]
    reason = table_entry.get("reason_1024")
    return {"id": native["task_id"], "cohort": native["cohort"], "group": native.get("group", native["task_id"]),
            "family": native["family"], "labels": labels, "kind": kind, "expected": native["expected"],
            "variants": variants, "latency_native": native["latency_s"],
            "reason": (reason["answer"] if reason and reason["ok"] else None), "reason_ok": bool(reason and reason["ok"]),
            "reason_latency": reason["latency_s"] if reason else None}


def combine(it, mode):
    v = it["variants"]
    if mode == "native" or mode not in ("avg_rev", "avg_rot", "avg_all"):
        p = v["native"]
    elif mode == "avg_rev":
        p = v["native"] if "native_reversed" not in v else [(a + b) / 2 for a, b in zip(v["native"], v["native_reversed"])]
    elif mode == "avg_rot":
        p = v["native"] if "rotated" not in v else [(a + b) / 2 for a, b in zip(v["native"], v["rotated"])]
    else:
        parts = [v[k] for k in ("native", "native_reversed", "rotated") if k in v]
        p = [sum(x) / len(parts) for x in zip(*parts)]
    return p


def features(it, p):
    s = sorted(p, reverse=True)
    n = len(p)
    ent = -sum(x * math.log(max(x, 1e-12)) for x in p) / math.log(max(n, 2))
    v = it["variants"]
    disagreement = 0.0
    for k in ("native_reversed", "rotated"):
        if k in v:
            disagreement = max(disagreement, max(abs(a - b) for a, b in zip(v["native"], v[k])))
    f = [s[i] if i < n else 0.0 for i in range(K)] + [ent, s[0] - (s[1] if n > 1 else 0.0), math.log(n), disagreement]
    f += [1.0 if TYPES[it["kind"]] == t else 0.0 for t in range(3)]
    return f


class Calibrator(torch.nn.Module):
    """Features -> (temperature, uniform mixing); argmax preserved."""

    def __init__(self, width=32):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(K + 7, width), torch.nn.Tanh(), torch.nn.Linear(width, 2))

    def forward(self, f, logp, n_mask):
        out = self.net(f)
        temperature = torch.exp(out[:, :1].clamp(-2.5, 2.5))
        eps = torch.sigmoid(out[:, 1:2] - 3.0)
        z = logp / temperature
        z = z.masked_fill(~n_mask, -1e9)
        q = torch.softmax(z, -1)
        n = n_mask.float().sum(-1, keepdim=True)
        return q * (1 - eps) + eps * n_mask.float() / n


def tensors(items, mode):
    f, logp, mask, gold = [], [], [], []
    for it in items:
        p = combine(it, mode)
        n = len(p)
        f.append(features(it, p))
        row = [math.log(max(x, 1e-12)) for x in p] + [0.0] * (26 - n)
        logp.append(row)
        mask.append([True] * n + [False] * (26 - n))
        gold.append(it["labels"].index(it["expected"]))
    return (torch.tensor(f), torch.tensor(logp), torch.tensor(mask), torch.tensor(gold))


def metrics(probs, gold, mask):
    n = mask.float().sum(-1)
    correct = (probs.argmax(-1) == gold).float()
    onehot = torch.nn.functional.one_hot(gold, 26).float() * mask.float()
    brier = ((probs - onehot) ** 2 * mask.float()).sum(-1)
    nll = -torch.log(probs.gather(1, gold[:, None]).squeeze(1).clamp_min(1e-12))
    conf = probs.max(-1).values
    bins = torch.clamp((conf * 10).long(), max=9)
    ece = 0.0
    for b in range(10):
        sel = bins == b
        if sel.any():
            ece += sel.float().sum().item() * abs(conf[sel].mean().item() - correct[sel].mean().item())
    return {"n": int(len(gold)), "accuracy": correct.mean().item(), "correct": int(correct.sum().item()),
            "brier": brier.mean().item(), "nll": nll.mean().item(), "ece": ece / len(gold)}


def temperature_scale(logp, mask, t):
    z = (logp / t).masked_fill(~mask, -1e9)
    return torch.softmax(z, -1)


def fit_temperature(logp, mask, gold):
    best = (None, None)
    for t in [x / 20 for x in range(5, 80)]:
        nll = -torch.log(temperature_scale(logp, mask, t).gather(1, gold[:, None]).clamp_min(1e-12)).mean().item()
        if best[0] is None or nll < best[0]:
            best = (nll, t)
    return best[1]


def fit_calibrator(train, dev, epochs=60):
    model = Calibrator()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    f, logp, mask, gold = train
    fd, lpd, md, gd = dev
    best = (None, None)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(gold))
        for i in range(0, len(gold), 64):
            idx = perm[i:i + 64]
            q = model(f[idx], logp[idx], mask[idx])
            onehot = torch.nn.functional.one_hot(gold[idx], 26).float() * mask[idx].float()
            loss = (-torch.log(q.gather(1, gold[idx][:, None]).clamp_min(1e-12)).mean()
                    + ((q - onehot) ** 2 * mask[idx].float()).sum(-1).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            q = model(fd, lpd, md)
            score = metrics(q, gd, md)
            s = score["nll"] + score["brier"]
            if best[0] is None or s < best[0]:
                best = (s, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    model.eval()
    return model


sets = {name: [it for it in (item(e) for e in by_task(load(name)).values()) if it] for name in ("train", "dev", "public", "oracle")}
print({k: len(v) for k, v in sets.items()}, flush=True)
report = {"sets": {k: len(v) for k, v in sets.items()}, "results": {}}
modes = ["native", "avg_rev", "avg_rot", "avg_all"]
for mode in modes:
    tr, dv = tensors(sets["train"], mode), tensors(sets["dev"], mode)
    pub, orc = tensors(sets["public"], mode), tensors(sets["oracle"], mode)
    entry = {}
    # Raw
    for name, t in (("public", pub), ("oracle", orc), ("dev", dv)):
        entry[f"raw_{name}"] = metrics(temperature_scale(t[1], t[2], 1.0), t[3], t[2])
    # Global temperature fit on train.
    T = fit_temperature(tr[1], tr[2], tr[3])
    entry["temperature"] = T
    for name, t in (("public", pub), ("oracle", orc), ("dev", dv)):
        entry[f"temp_{name}"] = metrics(temperature_scale(t[1], t[2], T), t[3], t[2])
    # Per-type temperature.
    per_type = {}
    for kind, code in TYPES.items():
        sel = tr[0][:, K + 4 + code] > 0.5
        per_type[kind] = fit_temperature(tr[1][sel], tr[2][sel], tr[3][sel]) if sel.sum() > 20 else T
    entry["per_type_temperature"] = per_type
    for name, t in (("public", pub), ("oracle", orc), ("dev", dv)):
        ts = torch.tensor([[per_type[k] for k, c in TYPES.items() if t[0][i, K + 4 + c] > 0.5][0] for i in range(len(t[3]))]).unsqueeze(1)
        entry[f"pertype_{name}"] = metrics(temperature_scale(t[1], t[2], ts), t[3], t[2])
    # Feature-conditioned calibrator.
    model = fit_calibrator(tr, dv)
    with torch.no_grad():
        for name, t in (("public", pub), ("oracle", orc), ("dev", dv)):
            entry[f"calib_{name}"] = metrics(model(t[0], t[1], t[2]), t[3], t[2])
    torch.save(model.state_dict(), OUT / f"calibrator_{mode}.pt")
    report["results"][mode] = entry
    print(mode, "T", round(T, 2), "per-type", {k: round(v, 2) for k, v in per_type.items()}, flush=True)
    for stage in ("raw", "temp", "pertype", "calib"):
        e = entry[f"{stage}_public"]
        print(f"  {stage:8s} public {e['correct']}/{e['n']} brier {e['brier']:.4f} ece {e['ece']:.4f} nll {e['nll']:.4f} | dev brier {entry[f'{stage}_dev']['brier']:.4f}", flush=True)

# Escalation: gate on calibrated confidence of the best mode; thresholds chosen on dev.
best_mode = min(modes, key=lambda m: report["results"][m]["calib_dev"]["brier"])
report["best_mode"] = best_mode
model = Calibrator()
model.load_state_dict(torch.load(OUT / f"calibrator_{best_mode}.pt", weights_only=True))
model.eval()


def escalate(items, tau, weight):
    t = tensors(items, best_mode)
    with torch.no_grad():
        q = model(t[0], t[1], t[2])
    conf = q.max(-1).values
    rows, latency, escalated = [], 0.0, 0
    out = q.clone()
    for i, it in enumerate(items):
        latency += it["latency_native"]
        if conf[i] < tau and it["reason_ok"] and it["reason"] in it["labels"]:
            escalated += 1
            latency += it["reason_latency"]
            onehot = torch.zeros(26)
            onehot[it["labels"].index(it["reason"])] = 1.0
            out[i] = (1 - weight) * q[i] + weight * onehot
    m = metrics(out, t[3], t[2])
    m.update({"tau": tau, "weight": weight, "escalated": escalated, "rate": escalated / max(len(items), 1), "mean_latency_s": latency / max(len(items), 1)})
    return m, out


dev_reason = [it for it in sets["dev"] if it["reason_ok"]]
print("dev items with reasoning:", len(dev_reason), "public with reasoning:", sum(it["reason_ok"] for it in sets["public"]), flush=True)
grid = []
for tau in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]:
    for weight in (0.6, 0.8, 0.9):
        m, _ = escalate(dev_reason, tau, weight)
        grid.append((m["brier"] - 0.5 * m["accuracy"], tau, weight, m))
grid.sort(key=lambda x: x[0])
report["escalation_dev_grid"] = [{"tau": g[1], "weight": g[2], **{k: g[3][k] for k in ("accuracy", "brier", "rate", "mean_latency_s")}} for g in grid]
chosen = grid[0]
report["escalation"] = {"tau": chosen[1], "weight": chosen[2]}
for tau in sorted({g[1] for g in grid}):
    m, _ = escalate(sets["public"], tau, chosen[2])
    report["results"].setdefault("escalation_public", {})[str(tau)] = m
    print(f"escalation tau {tau:<5} weight {chosen[2]} public {m['correct']}/{m['n']} brier {m['brier']:.4f} ece {m['ece']:.4f} rate {m['rate']:.2f} latency {m['mean_latency_s']:.2f}s", flush=True)
m_pub, probs_pub = escalate(sets["public"], chosen[1], chosen[2])
report["results"]["escalation_public_chosen"] = m_pub

# Per-item rows for the chosen configuration (and calibrated-only), in the phase-1 schema.
def write_rows(name, probs, items):
    with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as out:
        for i, it in enumerate(items):
            p = {l: probs[i, j].item() for j, l in enumerate(it["labels"])}
            answer = max(it["labels"], key=p.get)
            out.write(json.dumps({"cohort": it["cohort"], "task_id": it["id"], "group": it["group"], "family": it["family"],
                                  "question_type": it["kind"], "method": name, "expected": it["expected"], "ok": True,
                                  "correct": answer == it["expected"], "probs": p, "answer": answer, "latency_s": 0.0,
                                  "usage": {}, "request": None, "response": None, "error": None}) + "\n")


with torch.no_grad():
    t = tensors(sets["public"], best_mode)
    write_rows("native_calibrated", model(t[0], t[1], t[2]), sets["public"])
write_rows("native_escalated", probs_pub, sets["public"])

# Paired against the native baseline rows and the Clef BF16 reference.
baseline = [r for r in (json.loads(l) for l in BASELINE.read_text(encoding="utf-8").splitlines() if l.strip()) if r["method"] == "native"]
reference = [dict(r, method="clef_bf16") for r in (json.loads(l) for l in REFERENCE.read_text(encoding="utf-8").splitlines() if l.strip()) if r["method"] == "clef" and r["cohort"] in ("easy", "original", "hard")]
for name in ("native_calibrated", "native_escalated"):
    cand = [json.loads(l) for l in (OUT / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = baseline + reference + cand
    for cohort in ("all", "hard"):
        sel = rows if cohort == "all" else [r for r in rows if r["cohort"] == cohort]
        p_nat = study.paired(sel, "native").get(name, {})
        p_ref = study.paired(sel, "clef_bf16").get(name, {})
        m = {k: study.metrics([r for r in sel if r["method"] == k]) for k in ("native", "clef_bf16", name)}
        line = {"cohort": cohort, "correct": {k: f"{v['correct']}/{v['n']}" for k, v in m.items()},
                "brier": {k: round(v["brier_valid_only"], 4) for k, v in m.items()},
                "vs_native": {"fixed": p_nat.get("fixed"), "broken": p_nat.get("broken"), "ci": p_nat.get("scenario_bootstrap_95")},
                "vs_clef_bf16": {"fixed": p_ref.get("fixed"), "broken": p_ref.get("broken"), "ci": p_ref.get("scenario_bootstrap_95")}}
        report.setdefault("paired", {})[f"{name}/{cohort}"] = line
        print(name, json.dumps(line), flush=True)
(OUT / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
print("ANALYSIS_DONE")
