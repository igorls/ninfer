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
import statistics
import subprocess
from pathlib import Path

BUDGETS = [0, 1024, 2048]
PROFILE = "qwen3.8-27b/nvfp4:fp8:chunk1024:cache-off:spec-none:medium:frontier-split"
# The collector appends ":cN" when N rows decode concurrently (labels depend on batch
# composition) and ":derive2048" when an uncapped 1024 action stands in for the 2048 one.
# A training set always holds exactly one profile.
PROFILE_PATTERN = re.compile(re.escape(PROFILE) + r"(?::c[2-8])?(?::derive2048)?")
# A --direct-only collection re-observes the direct action with its answer-letter distribution.
CONFIDENCE_PROFILE_PATTERN = re.compile(re.escape(PROFILE) + r"(?::c[2-8])?:direct-only")


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


def split_for(row, holdout_family, fold=0):
    if row["family"] == holdout_family:
        return "family_test"
    # Fold k rotates the ten group buckets, so over folds 0-9 every group is tested once.
    bucket = (int(digest(row["group"])[:8], 16) - fold) % 10
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
    command = [str(args.collector.resolve()), str(args.artifact.resolve()),
               str(args.requests.resolve()), str(args.out.resolve()), artifact_digest,
               "--concurrency", str(args.concurrency)]
    if args.derive_2048:
        command.append("--derive-2048")
    if args.direct_only:
        command.append("--direct-only")
    subprocess.run(command, check=True)
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


def jsonl_lines(path):
    # JSONL records end at "\n" only; str.splitlines would also split on U+2028, U+2029 and
    # U+0085, which the collector writes raw inside model text.
    return [line.rstrip("\r") for line in path.read_text(encoding="utf-8").split("\n")]


def load_rows(paths, holdout_family, fold=0):
    rows, seen, groups, prompts, artifacts, profiles = [], set(), {}, {}, set(), set()
    for path in paths:
        for line in jsonl_lines(path):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["schema"] != 1 or not PROFILE_PATTERN.fullmatch(row["profile"]):
                raise ValueError("incompatible collection schema/profile")
            profiles.add(row["profile"])
            item = row["input"]
            if item["id"] in seen:
                raise ValueError("duplicate observation: pass each accumulated file only once")
            seen.add(item["id"])
            if len(row["features"]) != 5120 or not all(math.isfinite(x) for x in row["features"]):
                raise ValueError("invalid backbone feature vector")
            split = split_for(item, holdout_family, fold)
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
    if len(profiles) != 1:
        raise ValueError("incompatible collection schema/profile: one collection profile per training set")
    return rows


def load_confidence(path, rows):
    """Answer-letter confidence of each row's direct action, from a --direct-only collection.

    The row must observe the identical prompt state (input, artifact and exact features) and
    produce the same direct answer as the paired collection. Returns per row: top letter
    probability, margin to the second, entropy normalized by log(options), and the raw
    probability mass the whole vocabulary places on the option letters."""
    observed = {}
    for line in jsonl_lines(path):
        if line.strip():
            row = json.loads(line)
            if not CONFIDENCE_PROFILE_PATTERN.fullmatch(row["profile"]):
                raise ValueError("confidence rows must come from a --direct-only collection")
            observed[row["input"]["id"]] = row
    values = []
    for row in rows:
        other = observed.get(row["input"]["id"])
        if other is None:
            raise ValueError("confidence collection lacks an observation")
        if (other["input"] != row["input"] or other["artifact_sha256"] != row["artifact_sha256"]
                or other["features"] != row["features"]):
            raise ValueError("confidence row observes a different prompt state")
        direct = other["actions"][0]
        if direct["budget"] != 0 or direct["content"] != row["actions"][0]["content"]:
            raise ValueError("confidence direct answer differs from the paired direct action")
        letters = direct["answer_logprobs"]
        if set(letters) != set(row["input"]["mapping"]) or len(letters) < 2:
            raise ValueError("confidence must cover exactly the option letters")
        p = sorted((math.exp(v) for v in letters.values()), reverse=True)
        entropy = -sum(q * math.log(q) for q in p if q > 0) / math.log(len(p))
        mass = sum(math.exp(v) for v in direct["answer_raw_logprobs"].values())
        values.append([p[0], p[0] - p[1], entropy, mass])
    return values


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
    rows = load_rows(args.data, args.holdout_family, args.fold)
    indices = {split: [i for i, row in enumerate(rows) if row["split"] == split]
               for split in ("train", "validation", "test", "family_test")}
    group_counts = {split: len({rows[i]["input"]["group"] for i in index})
                    for split, index in indices.items()}
    if group_counts["train"] < 8 or group_counts["validation"] < 3:
        raise ValueError("need at least eight training and three validation groups")
    if args.inputs != "hidden" and not args.confidence:
        raise ValueError("confidence inputs need --confidence")
    confidence = (torch.tensor(load_confidence(args.confidence, rows), dtype=torch.float32, device=device)
                  if args.confidence else None)
    parts = []
    if args.inputs != "confidence":
        hidden = torch.tensor([r["features"] for r in rows], dtype=torch.float32, device=device)
        # Per-example RMS scaling has no fitted corpus statistics and stays stable between rounds.
        parts.append(hidden / hidden.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6))
    if args.inputs != "hidden":
        parts.append(confidence)
    x = torch.cat(parts, -1)
    y = torch.tensor([labels(r)[0] for r in rows], dtype=torch.float32, device=device)
    costs = torch.tensor([labels(r)[1] for r in rows], dtype=torch.float32, device=device)
    width = x.shape[1]
    net = (torch.nn.Linear(width, 6) if args.hidden == 0 else
           torch.nn.Sequential(torch.nn.Linear(width, args.hidden), torch.nn.Tanh(),
                               torch.nn.Linear(args.hidden, 6))).to(device)
    contract = {"schema": 1, "profile": rows[0]["profile"], "artifact_sha256": rows[0]["artifact_sha256"],
                "budgets": BUDGETS, "feature": "pre-think-final-norm-bf16-as-f32",
                "normalization": "per-row-rms", "hidden": args.hidden, "inputs": args.inputs,
                "holdout_family": args.holdout_family, "cost_weight": args.cost_weight,
                "split_policy": f"sha256-group-10-rotate{args.fold}:test0:validation1,2:train3-9"}
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

    gate = None
    if confidence is not None:
        # Baseline with one fitted number: answer directly when the top option letter's
        # probability reaches the threshold, otherwise reason with the 1,024 budget. The
        # threshold maximizes train+validation utility; tests never influence it.
        fit = torch.tensor(indices["train"] + indices["validation"], device=device)
        top, iv = confidence[fit, 0], torch.arange(len(fit), device=device)
        best_gate = None
        for threshold in sorted(set(top.tolist())) + [float("inf")]:
            actions = torch.where(top >= threshold, 0, 1)
            utility = float((y[fit][iv, actions] - args.cost_weight * costs[fit][iv, actions] / 1024).mean())
            if best_gate is None or utility > best_gate[0]:
                best_gate = (utility, threshold)
        gate = {"threshold": best_gate[1], "fit_utility": best_gate[0],
                "actions": torch.where(confidence[:, 0] >= best_gate[1], 0, 1)}
    metrics = {}
    oracle = (y - args.cost_weight * costs / 1024).argmax(-1)
    for split, index in indices.items():
        metrics[split] = {"router": score(index, chosen), "oracle_upper_bound": score(index, oracle)}
        for j, budget in enumerate(BUDGETS):
            metrics[split][f"always_{budget}"] = score(index, torch.full_like(chosen, j))
        if gate:
            metrics[split]["confidence_gate"] = score(index, gate["actions"])
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
              "confidence_gate": {k: v for k, v in gate.items() if k != "actions"} if gate else None,
              "groups": group_counts, "seed": args.seed, "metrics": metrics,
              "qualified_for_serving": False,
              "limitation": "Small authored diagnostic templates; not a general capability or production qualification."}
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.out / "predictions.jsonl").open("w", encoding="utf-8") as out:
        for i, row in enumerate(rows):
            out.write(json.dumps({"id": row["input"]["id"], "split": row["split"],
                                  "family": row["input"]["family"], "budget": BUDGETS[int(chosen[i])],
                                  "gate_budget": BUDGETS[int(gate["actions"][i])] if gate else None,
                                  "probabilities": probabilities[i].cpu().tolist(),
                                  "predicted_cost": predicted_cost[i].cpu().tolist()}) + "\n")
    print(json.dumps(report, indent=2))


def read_outcomes(path):
    rows = {}
    for line in jsonl_lines(path):
        if line.strip():
            row = json.loads(line)
            if row["input"]["id"] in rows:
                raise ValueError(f"duplicate observation in {path}")
            rows[row["input"]["id"]] = row
    return rows


def quantiles(values):
    if not values:
        return None
    ordered = sorted(values)
    return {"median": statistics.median(ordered), "p90": ordered[int(0.9 * (len(ordered) - 1))],
            "max": ordered[-1]}


def analyze(args):
    """Collection diagnostics: label yield by stratum, cap use, budget identity, decode rate,
    and (with --against) feature and label agreement between two collections of the same ids."""
    rows = {}
    for path in args.data:
        for key, row in read_outcomes(path).items():
            if key in rows:
                raise ValueError("duplicate observation across --data files")
            rows[key] = row
    strata = {}
    if args.blueprints:
        for line in args.blueprints.read_text(encoding="utf-8").splitlines():
            if line.strip():
                blueprint = json.loads(line)
                if "id" in blueprint:  # failed teacher slots carry no accepted blueprint
                    strata[blueprint["id"]] = f"{blueprint['kind']}-{blueprint['depth']}"

    def stratum(row):
        blueprint = row["input"].get("provenance", {}).get("blueprint")
        return strata.get(blueprint, row["input"]["family"])

    table, identity, rate_points = {}, {"compared": 0, "identical": 0}, []
    for row in rows.values():
        correct, _ = labels(row)
        actions = row["actions"]
        entry = table.setdefault(stratum(row), {"n": 0, "correct": [0, 0, 0], "reasoning_helps": 0,
                                                "reasoning_harms": 0, "2048_beats_1024": 0,
                                                "1024_beats_2048": 0, "all_fail": 0, "cap_applied": [0, 0],
                                                "derived_2048": 0, "reasoning_tokens": []})
        entry["n"] += 1
        entry["correct"] = [a + b for a, b in zip(entry["correct"], correct)]
        entry["reasoning_helps"] += int(correct[0] == 0 and max(correct[1:]) == 1)
        entry["reasoning_harms"] += int(correct[0] == 1 and min(correct[1:]) == 0)
        entry["2048_beats_1024"] += int(correct[2] > correct[1])
        entry["1024_beats_2048"] += int(correct[1] > correct[2])
        entry["all_fail"] += int(max(correct) == 0)
        for j in (1, 2):
            entry["cap_applied"][j - 1] += int(actions[j].get("thinking", {}).get("cap_applied", False))
            entry["reasoning_tokens"].append(actions[j].get("reasoning_tokens", 0))
        derived = "derived_from_budget" in actions[2]
        entry["derived_2048"] += int(derived)
        uncapped = not actions[1].get("thinking", {}).get("cap_applied", False) and actions[1]["stopped"]
        if uncapped and not derived and "thinking" in actions[1]:
            identity["compared"] += 1
            identity["identical"] += int(all(actions[1][k] == actions[2][k] for k in
                                             ("content", "reasoning", "output_tokens", "finish_reason")))
        for action in actions[1:]:
            if "derived_from_budget" not in action and action.get("seconds"):
                # Forced cap-close tokens are committed in one step, not decoded one by one.
                decoded = action["output_tokens"] - action.get("thinking", {}).get("injected_tokens", 0)
                rate_points.append((decoded, action["seconds"]))
    for entry in table.values():
        n = entry["n"]
        entry["accuracy"] = [round(c / n, 3) for c in entry["correct"]]
        entry["reasoning_tokens"] = quantiles(entry["reasoning_tokens"])
    rate = None
    if len(rate_points) >= 2:
        mean_t = statistics.fmean(t for t, _ in rate_points)
        mean_s = statistics.fmean(s for _, s in rate_points)
        slope = (sum((t - mean_t) * (s - mean_s) for t, s in rate_points) /
                 max(1e-9, sum((t - mean_t) ** 2 for t, _ in rate_points)))
        rate = {"decode_tokens_per_second": round(1 / slope, 2) if slope > 0 else None,
                "intercept_seconds": round(mean_s - slope * mean_t, 4), "actions": len(rate_points),
                "note": "OLS of action seconds on decoded tokens (output minus injected cap-close); concurrency 1 only"}
    report = {"rows": len(rows), "profiles": sorted({str(r.get("profile")) for r in rows.values()}),
              "artifacts": sorted({str(r.get("artifact_sha256")) for r in rows.values()}),
              "strata": dict(sorted(table.items())), "budget_identity_when_1024_uncapped": identity,
              "decode_rate": rate}
    if args.against:
        other = read_outcomes(args.against)
        shared = sorted(set(rows) & set(other))
        agreement = {"shared_ids": len(shared), "features_bitwise_equal": 0, "max_abs_feature_diff": 0.0,
                     "label_agreement": [0, 0, 0], "content_equal": [0, 0, 0], "disagreements": []}
        for key in shared:
            a, b = rows[key], other[key]
            diff = max(abs(x - y) for x, y in zip(a["features"], b["features"]))
            agreement["features_bitwise_equal"] += int(a["features"] == b["features"])
            agreement["max_abs_feature_diff"] = max(agreement["max_abs_feature_diff"], diff)
            ca, cb = labels(a)[0], labels(b)[0]
            for j in range(3):
                agreement["label_agreement"][j] += int(ca[j] == cb[j])
                agreement["content_equal"][j] += int(a["actions"][j]["content"] == b["actions"][j]["content"] and
                                                     a["actions"][j]["output_tokens"] == b["actions"][j]["output_tokens"])
            if ca != cb:
                agreement["disagreements"].append({"id": key, "stratum": stratum(a), "data": ca, "against": cb})
        report["against"] = {"path": str(args.against), **agreement}
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


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
    c.add_argument("--concurrency", type=int, choices=range(1, 9), default=1,
                   help="rows decoded concurrently; >1 marks labels as a non-repeatable :cN profile")
    c.add_argument("--derive-2048", action="store_true",
                   help="reuse an uncapped 1024 action as the 2048 action (profile :derive2048)")
    c.add_argument("--direct-only", action="store_true",
                   help="collect only the direct action with its answer-letter logprobs (profile :direct-only)")
    a = commands.add_parser("analyze")
    a.add_argument("--data", type=Path, nargs="+", required=True)
    a.add_argument("--blueprints", type=Path, help="synthetic_corpus blueprints.jsonl for kind-depth strata")
    a.add_argument("--against", type=Path, help="second collection of the same ids to compare")
    a.add_argument("--out", type=Path)
    t = commands.add_parser("train")
    t.add_argument("--data", type=Path, nargs="+", required=True)
    t.add_argument("--previous", type=Path)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--holdout-family", default="temporal_lookup", help="a family absent from the data holds none out")
    t.add_argument("--fold", type=int, choices=range(10), default=0, help="rotation of the group buckets")
    t.add_argument("--confidence", type=Path, help="--direct-only collection of the same rows")
    t.add_argument("--inputs", choices=("hidden", "confidence", "both"), default="hidden")
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
    elif args.command == "analyze":
        analyze(args)
    else:
        if args.epochs < 1 or args.cost_weight < 0 or args.learning_rate <= 0:
            parser.error("invalid training hyperparameters")
        train(args)


if __name__ == "__main__":
    main()
