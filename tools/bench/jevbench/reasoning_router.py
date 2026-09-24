"""Prepare oracle tasks and progressively train a frozen-Qwen reasoning router.

Preparation needs Python 3.11 only. Training needs PyTorch (provided by Colab).
Labels are actual paired Qwen outcomes checked against independent gold; neither
gold nor completed reasoning is part of the router's input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

BUDGETS = [0, 1024, 2048]
PROFILE = "qwen3.8-27b/nvfp4:fp8:chunk1024:cache-off:spec-none:medium:frontier-split"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prepare(args):
    from decision_study import Task, chat_request
    from make_decision_cases import make_cases
    rows = ([json.loads(line) for line in args.tasks.read_text(encoding="utf-8-sig").splitlines()
             if line.strip()] if args.tasks else make_cases(args.seed, args.scenarios))
    seen, prepared = set(), []
    for row in rows:
        task = Task.read(row, "oracle")
        body, mapping = chat_request(task, "submitted", "qwen3.8-27b")
        fingerprint = digest(body["messages"])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        # Identical prompts across seeds get the same identity and cannot cross splits.
        # External paraphrases must supply a shared group; authored templates also receive
        # a separate family holdout to expose the limits of instance-level generalization.
        prepared.append({"id": fingerprint, "group": row.get("source_group", row.get("group", fingerprint)) if args.tasks else fingerprint,
                         "family": task.family, "messages": body["messages"],
                         "mapping": mapping, "expected": task.expected,
                         "provenance": row.get("provenance", {})})
    prepared.sort(key=lambda row: digest([args.seed, row["id"]]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as out:
        for row in prepared:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"prepared": len(prepared), "deduplicated": len(rows) - len(prepared)}))


def split_for(row, holdout_family):
    if row["family"] == holdout_family:
        return "family_test"
    bucket = int(digest(row["group"])[:8], 16) % 10
    return "test" if bucket == 0 else "validation" if bucket in (1, 2) else "train"


def collect(args):
    def version():
        stat = args.artifact.stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ino

    before = version()
    with args.artifact.open("rb") as stream:
        artifact_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if version() != before:
        raise ValueError("artifact changed while hashing")
    subprocess.run([str(args.collector.resolve()), str(args.artifact.resolve()),
                    str(args.requests.resolve()), str(args.out.resolve()), artifact_digest], check=True)
    if version() != before:
        raise ValueError("artifact changed during collection; discard this collection output")


def labels(row):
    if [action["budget"] for action in row["actions"]] != BUDGETS:
        raise ValueError("paired actions must be complete and ordered 0/1024/2048")
    correct, costs = [], []
    for action in row["actions"]:
        answer = re.fullmatch(r"\s*([A-Z])(?:[.)])?\s*", action["content"])
        selected = row["input"]["mapping"].get(answer[1]) if answer else None
        correct.append(float(action["stopped"] and selected == row["input"]["expected"]))
        cost = action["output_tokens"]
        if not isinstance(cost, int) or cost < 1:
            raise ValueError("invalid observed output-token cost")
        costs.append(float(cost))
    return correct, costs


def load_rows(paths, holdout_family):
    rows, seen, groups, prompts, artifacts = [], set(), {}, {}, set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["schema"] != 1 or row["profile"] != PROFILE:
                raise ValueError("incompatible collection schema/profile")
            item = row["input"]
            if item["id"] in seen:
                raise ValueError("duplicate observation: pass each accumulated file only once")
            seen.add(item["id"])
            if len(row["features"]) != 5120 or not all(math.isfinite(x) for x in row["features"]):
                raise ValueError("invalid backbone feature vector")
            split = split_for(item, holdout_family)
            for key, table in ((item["group"], groups), (digest(item["messages"]), prompts)):
                if key in table and table[key] != split:
                    raise ValueError("related or identical prompts cross data splits")
                table[key] = split
            row["split"] = split
            labels(row)
            artifact_digest = row["artifact_sha256"]
            if not re.fullmatch(r"[0-9a-f]{64}", artifact_digest):
                raise ValueError("invalid frozen artifact digest")
            artifacts.add(artifact_digest)
            rows.append(row)
    if len(artifacts) != 1:
        raise ValueError("one explicitly pinned backbone artifact is required")
    return rows


def validate_replay(parent, contract, observations, training_ids):
    if parent["contract"] != contract:
        raise ValueError("previous router has a different feature/training contract")
    if not set(parent["training_ids"]).issubset(training_ids):
        raise ValueError("progressive training must retain every previous training example")
    if any(observations.get(key) != value for key, value in parent["observation_digests"].items()):
        raise ValueError("progressive training must retain unchanged prior observations and holdouts")


def train(args):
    import copy
    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows(args.data, args.holdout_family)
    indices = {split: [i for i, row in enumerate(rows) if row["split"] == split]
               for split in ("train", "validation", "test", "family_test")}
    group_counts = {split: len({rows[i]["input"]["group"] for i in index})
                    for split, index in indices.items()}
    if group_counts["train"] < 8 or group_counts["validation"] < 3:
        raise ValueError("need at least eight training and three validation groups")
    x = torch.tensor([r["features"] for r in rows], dtype=torch.float32, device=device)
    # Per-example RMS scaling has no fitted corpus statistics and stays stable between rounds.
    x = x / x.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    y = torch.tensor([labels(r)[0] for r in rows], dtype=torch.float32, device=device)
    costs = torch.tensor([labels(r)[1] for r in rows], dtype=torch.float32, device=device)
    net = (torch.nn.Linear(5120, 6) if args.hidden == 0 else
           torch.nn.Sequential(torch.nn.Linear(5120, args.hidden), torch.nn.Tanh(),
                               torch.nn.Linear(args.hidden, 6))).to(device)
    contract = {"schema": 1, "profile": PROFILE, "artifact_sha256": rows[0]["artifact_sha256"],
                "budgets": BUDGETS, "feature": "pre-think-final-norm-bf16-as-f32",
                "normalization": "per-row-rms", "hidden": args.hidden,
                "holdout_family": args.holdout_family, "cost_weight": args.cost_weight,
                "split_policy": "sha256-group-10:test0:validation1,2:train3-9"}
    parent = None
    observations = {r["input"]["id"]: digest({k: v for k, v in r.items() if k != "artifact"}) for r in rows}
    if args.previous:
        parent = json.loads(args.previous.read_text())
        validate_replay(parent, contract, observations, {rows[i]["input"]["id"] for i in indices["train"]})
        net.load_state_dict({key: torch.tensor(value, device=device)
                             for key, value in parent["state_dict"].items()})
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=0.1)

    def loss(index):
        prediction = net(x[index])
        classification = F.binary_cross_entropy_with_logits(prediction[:, :3], y[index])
        cost_loss = F.mse_loss(F.softplus(prediction[:, 3:]), costs[index] / 2048)
        return classification + 0.2 * cost_loss

    best, state, best_epoch = (float("inf"), float("inf")), None, 0
    for epoch in range(args.epochs + 1):
        net.eval()
        with torch.no_grad():
            val = indices["validation"]
            prediction = net(x[val])
            actions = (prediction[:, :3].sigmoid() - args.cost_weight *
                       F.softplus(prediction[:, 3:]) * 2).argmax(-1)
            iv = torch.arange(len(val), device=device)
            utility = (y[val][iv, actions] - args.cost_weight * costs[val][iv, actions] / 1024).mean()
            validation = (-float(utility), float(loss(val)))
        if validation < best:
            best, state, best_epoch = validation, copy.deepcopy(net.state_dict()), epoch
        if epoch - best_epoch >= 40 or epoch == args.epochs:
            break
        net.train()
        optimizer.zero_grad(set_to_none=True)
        loss(indices["train"]).backward()
        optimizer.step()
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        logits = net(x)
        probabilities = logits[:, :3].sigmoid()
        predicted_cost = F.softplus(logits[:, 3:]) * 2048
        chosen = (probabilities - args.cost_weight * predicted_cost / 1024).argmax(-1)

    def score(index, actions):
        if not index:
            return None
        ix = torch.tensor(index, device=device)
        picked = actions[ix]
        correct = y[ix, picked]
        paid = costs[ix, picked]
        return {"n": len(index), "correct": int(correct.sum()),
                "accuracy": float(correct.mean()), "mean_output_tokens": float(paid.mean()),
                "utility": float((correct - args.cost_weight * paid / 1024).mean()),
                "action_counts": {str(b): int((picked == j).sum()) for j, b in enumerate(BUDGETS)}}

    metrics = {}
    oracle = (y - args.cost_weight * costs / 1024).argmax(-1)
    for split, index in indices.items():
        metrics[split] = {"router": score(index, chosen), "oracle_upper_bound": score(index, oracle)}
        for j, budget in enumerate(BUDGETS):
            metrics[split][f"always_{budget}"] = score(index, torch.full_like(chosen, j))
        if index:
            metrics[split]["brier"] = float((probabilities[index] - y[index]).square().mean())
            metrics[split]["reasoning_helps"] = int(((y[index, 0] == 0) & (y[index, 1:].max(-1).values == 1)).sum())
            metrics[split]["reasoning_harms"] = int(((y[index, 0] == 1) & (y[index, 1:].min(-1).values == 0)).sum())
    args.out.mkdir(parents=True, exist_ok=False)
    checkpoint = {"contract": contract, "qualified_for_serving": False,
                  "observation_digests": observations,
                  "training_ids": [rows[i]["input"]["id"] for i in indices["train"]],
                  "parent": digest(parent) if parent else None,
                  "state_dict": {k: v.cpu().tolist() for k, v in state.items()}}
    (args.out / "router.json").write_text(json.dumps(checkpoint), encoding="utf-8")
    report = {"device": str(device), "torch": torch.__version__, "examples": len(rows),
              "parameters": sum(p.numel() for p in net.parameters()), "best_epoch": best_epoch,
              "validation_utility": -best[0], "validation_loss": best[1],
              "groups": group_counts, "seed": args.seed, "metrics": metrics,
              "qualified_for_serving": False,
              "limitation": "Small authored diagnostic templates; not a general capability or production qualification."}
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.out / "predictions.jsonl").open("w", encoding="utf-8") as out:
        for i, row in enumerate(rows):
            out.write(json.dumps({"id": row["input"]["id"], "split": row["split"],
                                  "family": row["input"]["family"], "budget": BUDGETS[int(chosen[i])],
                                  "probabilities": probabilities[i].cpu().tolist(),
                                  "predicted_cost": predicted_cost[i].cpu().tolist()}) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--tasks", type=Path)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--scenarios", type=int, default=32)
    p.add_argument("--out", type=Path, required=True)
    c = commands.add_parser("collect")
    c.add_argument("--artifact", type=Path, required=True)
    c.add_argument("--collector", type=Path, required=True)
    c.add_argument("--requests", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    t = commands.add_parser("train")
    t.add_argument("--data", type=Path, nargs="+", required=True)
    t.add_argument("--previous", type=Path)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--holdout-family", default="temporal_lookup")
    t.add_argument("--hidden", type=int, choices=(0, 128), default=0)
    t.add_argument("--cost-weight", type=float, default=0.02, help="accuracy utility cost per 1024 output tokens")
    t.add_argument("--learning-rate", type=float, default=0.0003)
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.scenarios < 1:
            parser.error("scenarios must be positive")
        prepare(args)
    elif args.command == "collect":
        collect(args)
    else:
        if args.epochs < 1 or args.cost_weight < 0 or args.learning_rate <= 0:
            parser.error("invalid training hyperparameters")
        train(args)


if __name__ == "__main__":
    main()
