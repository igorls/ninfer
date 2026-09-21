"""Run the public JevBench decisions against a live ninfer-serve and score them the JevBench way.

    python tools/bench/jevbench/run_public.py --jevbench <clone of fstandhartinger/jevbench> \
        --endpoint http://127.0.0.1:8010/v1 --model qwen3.8-27b --out profiles/bench/jevbench-<date>

The script installs ``ninfer_native.py`` into the clone's adapter package (a copy, so the clone
stays a plain checkout), runs the three public files (easy, original = "standard", hard) through
JevBench's own serial no-retry ``Runner``, and then applies ``jevbench.composite_v12`` to the
public subset: Intelligence over the tiers present (the judge tier has no public items, so its
weight is renormalised away exactly as the scorer does for a tier a system never ran),
Calibration from the hard tier (ECE and total-variation distance to the gold distributions of the
public probability items), Speed from the standard-tier p50/p95 with the self-hosted adjustment,
and Cost at the hosted size-class list price the board uses for a public 27B dense model
($0.09 per million input tokens, output free) times the measured input tokens.

The same public items are then compared, one for one, with every published system's per-task
outcomes in ``results/v1.2/jevbench-v1.2-per-task.json`` so the accuracy comparison is on identical
questions. Held-out and imported items are not public, so the printed JevBench Score is an
estimate over 231 of the 534 decisions, never the official number.

Outputs (all under --out, which must not exist yet): ``<tier>.results.jsonl`` per tier,
``raw/`` bodies, ``ledger.jsonl``, ``manifest.json`` and ``summary.json``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shutil
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIERS = {"easy": "easy.jsonl", "standard": "original.jsonl", "hard": "hard.jsonl"}
PRICE_IN_PER_M = 0.09   # board: OpenRouter Gemma 4 26B-A4B size-class reference for a public 27B
PRICE_OUT_PER_M = 0.0


def install_adapter(clone: Path) -> None:
    target = clone / "jevbench" / "adapters" / "ninfer_native.py"
    shutil.copyfile(HERE / "ninfer_native.py", target)


def main() -> int:
    global PRICE_IN_PER_M
    ap = argparse.ArgumentParser()
    ap.add_argument("--jevbench", required=True, help="path to a jevbench checkout")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8010/v1")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--out", required=True, help="new output directory")
    ap.add_argument("--limit", type=int, default=None, help="first N tasks of each tier")
    ap.add_argument("--request-options", default=None, help="JSON overrides of the request body")
    ap.add_argument("--logit-temperature", type=float, default=None,
                    help="adapter-side divisor of the option logits (calibration only; default 1.0)")
    ap.add_argument("--endpoint-kind", default="gpu", choices=["gpu", "api"],
                    help="gpu = self-hosted (x2 + 0.15 s latency adjustment), api = production")
    ap.add_argument("--price-in-per-m", type=float, default=PRICE_IN_PER_M,
                    help="hosted size-class list price per million input tokens for the Cost axis")
    args = ap.parse_args()
    PRICE_IN_PER_M = args.price_in_per_m

    clone = Path(args.jevbench).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        print(f"refusing to reuse {out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True)
    install_adapter(clone)
    sys.path.insert(0, str(clone))
    if os.name == "nt":
        # JevBench's ledger serialises concurrent runs with POSIX flock. This driver is one serial
        # process on Windows, where the module does not exist; a no-op lock keeps the ledger's
        # reserve/settle semantics and only drops the cross-process exclusion.
        import types
        sys.modules.setdefault("fcntl", types.SimpleNamespace(
            LOCK_SH=1, LOCK_EX=2, LOCK_UN=8, flock=lambda *a, **k: None))
    from jevbench import composite_v12 as C  # noqa: E402
    from jevbench.adapters.ninfer_native import NInferNativeAdapter  # noqa: E402
    from jevbench.budget import Ledger  # noqa: E402
    from jevbench.runner import Runner  # noqa: E402
    from jevbench.summarize import summarize  # noqa: E402
    from jevbench.tasks import dataset_hash, load_jsonl  # noqa: E402

    adapter = NInferNativeAdapter(endpoint=args.endpoint, model=args.model,
                                  price_input_per_m=PRICE_IN_PER_M,
                                  price_output_per_m=PRICE_OUT_PER_M)
    if args.request_options:
        adapter.request_options = json.loads(args.request_options)
    if args.logit_temperature is not None:
        from jevbench.adapters.ninfer_native import LOGIT_TEMPERATURE_OPTION  # noqa: E402
        adapter.request_options[LOGIT_TEMPERATURE_OPTION] = args.logit_temperature
    ledger = Ledger(str(out / "ledger.jsonl"), cap_usd=5.0)
    runner = Runner(adapter, ledger, raw_dir=str(out / "raw"), default_reserve_usd=0.0)

    started = dt.datetime.now(dt.timezone.utc).isoformat()
    tiers = {}
    for tier, fname in TIERS.items():
        tasks = load_jsonl(str(clone / "datasets" / "public" / fname))
        if args.limit:
            tasks = tasks[: args.limit]
        print(f"[{tier}] {len(tasks)} decisions", flush=True)
        records = runner.run_all(tasks, results_path=str(out / f"{tier}.results.jsonl"))
        tiers[tier] = {"tasks": tasks, "records": records, "hash": dataset_hash(tasks)}
    finished = dt.datetime.now(dt.timezone.utc).isoformat()

    # --- JevBench Score on the public subset -------------------------------------------------
    by_id = {t.id: t for tier in tiers.values() for t in tier["tasks"]}
    acc = {}
    per_tier = {}
    for tier, d in tiers.items():
        s = summarize(d["tasks"], d["records"])
        per_tier[tier] = {k: s[k] for k in ("n_planned", "n_attempted", "n_valid", "n_correct",
                                            "accuracy", "schema_validity", "brier_mean",
                                            "latency", "per_family")}
        per_tier[tier]["ece"] = s["ece"]["ece"] if s["ece"] else None
        # a failed request counts as wrong, as in the board
        acc[tier] = s["n_correct"] / s["n_planned"] if s["n_planned"] else None

    hard = tiers["hard"]
    hard_pairs = []
    tvds = []
    for r in hard["records"]:
        t = by_id[r["task_id"]]
        p = r.get("probs") or {}
        if p:
            hard_pairs.append((max(p.values()), r.get("correct") is True))
            gold = t.provenance.get("gold_probs")
            if gold:
                tvds.append(C.tvd(p, gold, t.labels))
    from jevbench.metrics import ece_top_label  # noqa: E402
    ece_hard = ece_top_label(hard_pairs)["ece"] if hard_pairs else None
    mean_tvd = statistics.fmean(tvds) if tvds else None

    std_lat = [r["latency_s"] for r in tiers["standard"]["records"]]
    all_records = [r for d in tiers.values() for r in d["records"]]
    all_lat = [r["latency_s"] for r in all_records]

    def pct(xs, q):
        from jevbench.metrics import percentile
        return percentile(xs, q)

    tokens = [r["usage"].get("input_tokens") for r in all_records
              if r.get("ok") and isinstance(r["usage"].get("input_tokens"), (int, float))]
    mean_in = statistics.fmean(tokens) if tokens else None
    usd_per_1000 = mean_in * 1000 * PRICE_IN_PER_M / 1e6 if mean_in else None

    axes = {
        "intelligence": C.intelligence({"easy": acc["easy"], "standard": acc["standard"],
                                        "judge": None, "hard": acc["hard"]}),
        "calibration": C.calibration(ece_hard, mean_tvd),
        "speed": C.speed(pct(std_lat, 0.5), pct(std_lat, 0.95), args.endpoint_kind),
        "cost": C.cost(usd_per_1000) if usd_per_1000 else None,
    }
    score = C.jevbench_score(axes)

    # --- same-item comparison with the published board ---------------------------------------
    board = json.loads((clone / "results/v1.2/jevbench-v1.2-per-task.json").read_text("utf-8"))
    results_json = json.loads((clone / "results/v1.2/jevbench-v1.2-results.json").read_text("utf-8"))
    board_axes = {s["key"]: s for s in results_json["systems"]}
    ours = {r["task_id"]: r for r in all_records}
    comparison = []
    for key, sysrow in board["systems"].items():
        pt = sysrow.get("public_tasks") or {}
        row = {"key": key, "display": sysrow["display"], "partial": sysrow.get("partial")}
        for tier, d in tiers.items():
            ids = [t.id for t in d["tasks"]]
            theirs = [pt.get(i) for i in ids]
            attempted = [x for x in theirs if x and x[0] != "n"]
            if len(attempted) < len(ids):
                row[tier] = None
                continue
            row[tier] = {"them": sum(1 for x in attempted if x[0] == "c") / len(ids),
                         "us": sum(1 for i in ids if ours.get(i, {}).get("correct")) / len(ids),
                         "them_p50_s": statistics.median(x[1] for x in attempted if x[1] is not None)
                         if any(x[1] is not None for x in attempted) else None}
        b = board_axes.get(key)
        if b:
            row["board"] = {"jevbench_score": b.get("jevbench_score"), "axes": b.get("axes"),
                            "ranked": b.get("ranked")}
        comparison.append(row)

    summary = {
        "endpoint": args.endpoint, "model": args.model, "endpoint_kind": args.endpoint_kind,
        "request_options": adapter.request_options, "started_utc": started,
        "finished_utc": finished,
        "dataset_hashes": {k: v["hash"] for k, v in tiers.items()},
        "public_tier_accuracy_failures_wrong": acc,
        "per_tier": per_tier,
        "hard_calibration": {"ece": ece_hard, "n": len(hard_pairs), "mean_tvd": mean_tvd,
                             "n_probability_items": len(tvds)},
        "latency_s": {"standard_p50": pct(std_lat, 0.5), "standard_p95": pct(std_lat, 0.95),
                      "all_p50": pct(all_lat, 0.5), "all_p95": pct(all_lat, 0.95),
                      "hard_p50": pct([r["latency_s"] for r in hard["records"]], 0.5),
                      "hard_p95": pct([r["latency_s"] for r in hard["records"]], 0.95),
                      "adjusted_standard_p50": C.adjusted_latency(pct(std_lat, 0.5),
                                                                  args.endpoint_kind),
                      "adjusted_standard_p95": C.adjusted_latency(pct(std_lat, 0.95),
                                                                  args.endpoint_kind)},
        "cost": {"price_in_per_m": PRICE_IN_PER_M, "price_out_per_m": PRICE_OUT_PER_M,
                 "mean_input_tokens": mean_in, "usd_per_1000_est": usd_per_1000,
                 "basis": "ESTIMATE: hosted size-class list price x measured input tokens; "
                          "self-hosted GPU has no tariff"},
        "axes_public_subset": axes,
        "jevbench_score_public_subset": score,
        "note": "231 public decisions of 534; judge tier absent (weights renormalised). "
                "Not the official score.",
        "comparison_same_public_items": comparison,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), "utf-8")
    manifest = {"adapter": "ninfer_native", "endpoint": args.endpoint, "model": args.model,
                "request_options": adapter.request_options, "price_in_per_m": PRICE_IN_PER_M,
                "price_out_per_m": PRICE_OUT_PER_M, "cost_basis": "self_hosted_gpu",
                "dataset_hashes": summary["dataset_hashes"], "started_utc": started,
                "finished_utc": finished, "jevbench_checkout": str(clone)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")

    print(json.dumps({k: summary[k] for k in ("public_tier_accuracy_failures_wrong",
                                              "hard_calibration", "latency_s", "cost",
                                              "axes_public_subset",
                                              "jevbench_score_public_subset")},
                     indent=2))
    print(f"\n{'system':<48} {'easy':>6} {'std':>6} {'hard':>6}   board")
    for row in sorted(comparison, key=lambda r: -(r.get("board") or {}).get("jevbench_score", 0)):
        cells = []
        for tier in TIERS:
            c = row.get(tier)
            cells.append(f"{c['them']*100:5.1f}%" if c else "   -  ")
        b = row.get("board") or {}
        print(f"{row['display'][:48]:<48} {' '.join(cells)}   "
              f"{b.get('jevbench_score', float('nan')):5.1f}")
    print(f"{'NInfer ' + args.model:<48} "
          + " ".join(f"{acc[t]*100:5.1f}%" for t in TIERS)
          + f"   {score:5.1f} (public subset)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
