"""Flash-Next acceptance against the independent transformers reference (the oracle gate).

The oracle is transformers' own `qwen4_exp` text model with the pinned source checkpoint's weights
decoded independently of NInfer: NVFP4 experts through the FP4 table and their stored scales, FP8
projections with their row scales, the u4 PLE table with its scales. It runs in FP32 (TF32 off).
Each engine's own tokens from a `parity.py collect` record set are teacher-forced through it, so
every engine distribution is compared with the oracle's at an identical context.

    run   python -m tools.bench.flash_next.oracle run --model-dir <mixed> --ple-dir <ples_int4>
            --records v2=parity-R.jsonl --records v3=parity-v3.jsonl --out oracle.jsonl
    gate  python -m tools.bench.flash_next.oracle gate --oracle oracle.jsonl
            --reference v2=parity-R.jsonl --candidate v3=parity-v3.jsonl [--out report.json]

The gate's metrics and criteria are fixed in docs/research/flash-next-v3-port-2026-09-29.md
("Acceptance by independent oracle"). Sources: primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8
(without ple-bf16-*) and primitive-ai/Qwen3.8-Flash-Next-PLE-quant ples_int4, at the revisions
the artifact's conversion record pins. Adapted from the research line's state oracle.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

FP4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
FP8_TARGETS = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
               "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
PLE_ROWS_PER_SHARD = 2500012


def _served_tools(tools: list) -> list:
    """The tool definitions as the engine's OpenAI route hands them to the chat template
    (src/serve/translate.cpp render_tool_definition): name, parameters, strict=false, then description."""
    out = []
    for tool in tools:
        function = tool["function"]
        served = {"name": function["name"],
                  "parameters": function.get("parameters") or {"type": "object", "properties": {}}, "strict": False}
        if function.get("description"):
            served["description"] = function["description"]
        out.append({"type": "function", "function": served})
    return out


def _load_records(spec: str) -> tuple[str, dict]:
    label, _, path = spec.partition("=")
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        out[(record["id"], record["kind"])] = record
    return label, out


# ---------------------------------------------------------------- the reference model


def _build(model_dir: str, ple_dir: str, device: str, attn: str):
    import safetensors
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers.activations import ACT2FN
    from transformers.models.qwen4_exp import modeling_qwen4_exp as M
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    lut = torch.tensor(FP4, dtype=torch.float32, device=device)

    class Experts(nn.Module):
        """One layer's routed experts, decoded from the packed NVFP4 source when a forward needs them."""

        def __init__(self, config, layer: int):
            super().__init__()
            self.num_experts, self.layer = config.num_experts, layer
            self.act = ACT2FN[config.hidden_act]
            self.file = safetensors.safe_open(os.path.join(model_dir, f"ct-experts-layer{layer:02d}.safetensors"),
                                              framework="pt", device="cpu")

        def _decode(self, prefix: str):
            packed = self.file.get_tensor(prefix + ".weight_packed").to(device)
            scale = self.file.get_tensor(prefix + ".weight_scale").to(device).float()
            glob = self.file.get_tensor(prefix + ".weight_global_scale").float().item()
            rows, half = packed.shape
            values = torch.stack([lut[(packed & 0x0F).long()], lut[(packed >> 4).long()]], dim=-1).reshape(rows, half * 2)
            return values * (scale.repeat_interleave(16, dim=-1) / glob)

        def forward(self, hidden, top_k_index, top_k_weights):
            out = torch.zeros_like(hidden)
            mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            for expert in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist():
                if expert == self.num_experts:
                    continue
                pos, tok = torch.where(mask[expert])
                base = f"model.language_model.layers.{self.layer}.mlp.experts.{expert}"
                gate_up = torch.cat([self._decode(base + ".gate_proj"), self._decode(base + ".up_proj")], dim=0)
                gate, up = F.linear(hidden[tok], gate_up).chunk(2, dim=-1)
                h = F.linear(self.act(gate) * up, self._decode(base + ".down_proj"))
                out.index_add_(0, tok, h * top_k_weights[tok, pos, None])
            return out

    class PLE(nn.Module):
        """The u4 PLE table held in host memory once, gathered per forward."""

        def __init__(self):
            super().__init__()
            # the reference reads `ngram_embedding.weight.device` to place the ids; the rows live in host memory
            self.register_buffer("weight", torch.empty(0), persistent=False)
            self.i4, self.scale = [], []
            shards = sorted(Path(ple_dir).glob("shard_*.safetensors"), key=lambda p: int(p.stem.split("_")[1]))
            for path in shards:
                with safetensors.safe_open(str(path), framework="pt", device="cpu") as f:
                    self.i4.append(f.get_tensor("weight_i4"))
                    self.scale.append(f.get_tensor("weight_scale"))

        def forward(self, ngram_ids):
            flat = ngram_ids.reshape(-1).cpu()
            rows = torch.empty(flat.numel(), 160, dtype=torch.float32)
            shard, local = flat // PLE_ROWS_PER_SHARD, flat % PLE_ROWS_PER_SHARD
            for s in shard.unique().tolist():
                where = (shard == s).nonzero().flatten()
                i4 = self.i4[s][local[where]].to(torch.int32)
                scale = self.scale[s][local[where]].float()
                values = torch.stack([(i4 & 0x0F).float() - 8.0, (i4 >> 4).float() - 8.0], dim=-1).reshape(-1, 160)
                rows[where] = values * scale.repeat_interleave(16, dim=-1)
            return rows.reshape(*ngram_ids.shape, 160).to(device)

    root = json.loads(Path(model_dir, "config.json").read_text(encoding="utf-8"))
    cfg = Qwen4ExpTextConfig(**root["text_config"])
    cfg._attn_implementation = attn
    with torch.device("meta"):
        model = M.Qwen4ExpTextModel(cfg)
    for layer in range(cfg.num_hidden_layers):
        model.layers[layer].mlp.experts = Experts(cfg, layer)
    ple_layer = next(l for l in model.layers if getattr(l, "ple", None) is not None)
    ple_layer.ple.ple_embedding.ngram_embedding = PLE()
    model.to_empty(device=device)
    # non-persistent buffers (RoPE inv_freq) are not in the checkpoint: rebuild them on the device
    for name, module in list(model.named_modules()):
        if isinstance(module, M.Qwen4ExpTextRotaryEmbedding):
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            setattr(parent, name.rsplit(".", 1)[-1], M.Qwen4ExpTextRotaryEmbedding(config=cfg, device=device))
    model.float().eval()

    index = json.loads(Path(model_dir, "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    files = {}

    def tensor(name):
        if index[name] not in files:
            files[index[name]] = safetensors.safe_open(os.path.join(model_dir, index[name]), framework="pt", device="cpu")
        return files[index[name]].get_tensor(name)

    state, head = {}, None
    for key, file in index.items():
        if file.startswith("ple-bf16-") or "ngram_embedding.weight" in key:
            continue
        if key == "lm_head.weight":
            head = tensor(key).float().to(device)
            continue
        if not key.startswith("model.language_model.") or ".mlp.experts." in key:
            continue
        target = key[len("model.language_model."):]
        if any(target.endswith(t + ".weight_scale") for t in FP8_TARGETS):
            continue
        if any(target.endswith(t + ".weight") for t in FP8_TARGETS):
            scale = tensor(key[: -len(".weight")] + ".weight_scale").float()
            weight = tensor(key).to(torch.float32)
            state[target] = weight * (scale.unsqueeze(1) if scale.ndim == 1 else scale)
        else:
            t = tensor(key)
            state[target] = t.float() if t.dtype in (torch.bfloat16, torch.float16) else t
    result = model.load_state_dict(state, strict=False)
    assert not result.missing_keys and not result.unexpected_keys, (result.missing_keys, result.unexpected_keys)
    emb = ple_layer.ple.ple_embedding
    assert emb.layer_multipliers.tolist() == [23703573157769, 20109073645365, 8052911324071], "PLE multipliers"
    assert emb.ngram_heads_offsets.tolist()[:3] == [0, 20000003, 40000026], "PLE offsets"
    return model, head


# ---------------------------------------------------------------- run


def _prompt_ids(tokenizer, item: dict) -> list[int]:
    extra = {"tools": _served_tools(item["extra"]["tools"])} if "tools" in item.get("extra", {}) else {}
    prompt = tokenizer.apply_chat_template(item["messages"], add_generation_prompt=True, tokenize=True,
                                           enable_thinking=item["thinking"], **extra)
    prompt = prompt["input_ids"] if hasattr(prompt, "keys") else prompt  # BatchEncoding in transformers 5
    return [int(t) for t in (prompt[0] if prompt and isinstance(prompt[0], list) else prompt)]


def run(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer
    from tools.bench.flash_next import parity

    sets = [_load_records(spec) for spec in args.records]
    items = {item["id"]: item for item in parity.request_set()}
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    started = time.time()
    model, head = _build(args.model_dir, args.ple_dir, args.device, args.attn)
    print(f"oracle built in {time.time() - started:.0f} s", flush=True)

    def readout(hidden, positions):
        logp = torch.log_softmax(torch.nn.functional.linear(hidden[0, positions], head), dim=-1)
        return logp

    done = set()
    out_path = Path(args.out)
    if out_path.exists():
        done = {(r["engine"], r["id"], r["kind"]) for r in map(json.loads, out_path.read_text(encoding="utf-8").splitlines())}
    with out_path.open("a", encoding="utf-8") as out:
        only = set(filter(None, args.only.split(",")))
        for item_id, item in items.items():
            if only and item_id not in only:
                continue
            prompt = _prompt_ids(tokenizer, item)
            for label, records in sets:
                gen = records.get((item_id, "generated"))
                pro = records.get((item_id, "prompt"))
                if gen is None or (label, item_id, "generated") in done:
                    continue
                t0 = time.time()
                ids = prompt + gen["tokens"]
                with torch.no_grad():
                    hidden = model(input_ids=torch.tensor([ids], device=args.device), use_cache=False).last_hidden_state
                    P, T = len(prompt), len(gen["tokens"])
                    logp = readout(hidden, list(range(P - 1, P + T - 1)))
                    chosen = logp[torch.arange(T), torch.tensor(gen["tokens"], device=args.device)].tolist()
                    top_v, top_i = logp.topk(20, dim=-1)
                    record = {"engine": label, "id": item_id, "kind": "generated", "prompt_tokens": P,
                              "engine_prompt_tokens": gen["prompt_tokens"], "tokens": gen["tokens"], "chosen": chosen,
                              "top": [[[int(i), float(v)] for i, v in zip(ri, rv)] for ri, rv in zip(top_i.tolist(), top_v.tolist())]}
                    out.write(json.dumps(record) + "\n")
                    if pro is not None and (label, item_id, "prompt") not in done:
                        positions = pro["positions"]
                        lp = readout(hidden, positions)
                        nxt = [prompt[p + 1] for p in positions]
                        out.write(json.dumps({"engine": label, "id": item_id, "kind": "prompt", "positions": positions,
                                              "tokens": nxt, "engine_tokens": pro["tokens"],
                                              "chosen": lp[torch.arange(len(positions)), torch.tensor(nxt, device=args.device)].tolist()}) + "\n")
                    out.flush()
                del hidden
                torch.cuda.empty_cache() if args.device.startswith("cuda") else None
                engine_prompt = gen["prompt_tokens"]
                mismatch = "" if P == engine_prompt else f"  (prompt {P} tokens vs engine {engine_prompt})"
                print(f"{label} {item_id}: {len(ids)} tokens in {time.time() - t0:.1f} s{mismatch}", flush=True)


# ---------------------------------------------------------------- gate


def _metrics(oracle: dict, records: dict, label: str) -> dict:
    """Engine against oracle at identical contexts, positions pooled over records."""
    diffs, kls, agree, total, excluded = [], [], 0, 0, []
    for (engine, item_id, kind), o in oracle.items():
        if engine != label:
            continue
        r = records.get((item_id, kind))
        if r is None:
            continue
        if kind == "generated":
            if o["prompt_tokens"] != r["prompt_tokens"] or o["tokens"] != r["tokens"]:
                excluded.append(item_id)
                continue
            for e_lp, o_lp, e_top, o_top in zip(r["raw"], o["chosen"], r["top"], o["top"]):
                diffs.append(abs(e_lp - o_lp))
                total += 1
                agree += int(e_top[0][0] == o_top[0][0])
                kls.append(_kl(o_top, e_top))
        else:
            if o["tokens"] != o["engine_tokens"]:
                excluded.append(item_id + "/prompt")
                continue
            diffs.extend(abs(a - b) for a, b in zip(r["raw"], o["chosen"]))
    kls.sort()
    return {"positions": len(diffs), "chosen_mean": statistics.fmean(diffs), "chosen_max": max(diffs),
            "top1_agreement": agree / max(1, total), "kl_mean": statistics.fmean(kls),
            "kl_p99": kls[min(len(kls) - 1, int(0.99 * len(kls)))], "excluded": excluded}


def _kl(oracle_top: list, engine_top: list) -> float:
    """KL(oracle || engine) over the union of both top-20 lists, each side renormalized on the union.
    A token missing from one side's top-20 takes that side's 20th log probability (an upper bound)."""
    o = {t: v for t, v in oracle_top}
    e = {t: v for t, v in engine_top}
    union = set(o) | set(e)
    o_floor, e_floor = min(o.values()), min(e.values())
    po = {t: math.exp(o.get(t, o_floor)) for t in union}
    pe = {t: math.exp(e.get(t, e_floor)) for t in union}
    zo, ze = sum(po.values()), sum(pe.values())
    return sum((po[t] / zo) * (math.log(po[t] / zo) - math.log(pe[t] / ze)) for t in union)


def gate(args: argparse.Namespace) -> None:
    oracle = {}
    for line in Path(args.oracle).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        oracle[(r["engine"], r["id"], r["kind"])] = r
    ref_label, ref = _load_records(args.reference)
    cand_label, cand = _load_records(args.candidate)
    m_ref, m_cand = _metrics(oracle, ref, ref_label), _metrics(oracle, cand, cand_label)
    # pre-registered: no worse than the reference by more than max(10% of the reference, route variation)
    route = {"chosen_mean": 0.000567, "chosen_max": 0.374786}
    rows, passed = [], True
    for key, higher_better in (("chosen_mean", False), ("chosen_max", False), ("top1_agreement", True),
                               ("kl_mean", False), ("kl_p99", False)):
        r, c = m_ref[key], m_cand[key]
        slack = max(0.10 * abs(r), route.get(key, 0.0))
        ok = c >= r - slack if higher_better else c <= r + slack
        passed &= ok
        rows.append({"metric": key, ref_label: r, cand_label: c, "allowed": (r - slack) if higher_better else (r + slack), "pass": ok})
    report = {"reference": ref_label, "candidate": cand_label, ref_label + "_metrics": m_ref,
              cand_label + "_metrics": m_cand, "criteria": rows, "pass": passed}
    for row in rows:
        print(f"{row['metric']:15s} {ref_label} {row[ref_label]:.6f}  {cand_label} {row[cand_label]:.6f}  "
              f"allowed {row['allowed']:.6f}  {'PASS' if row['pass'] else 'FAIL'}")
    print(f"positions {ref_label} {m_ref['positions']}  {cand_label} {m_cand['positions']}; "
          f"excluded {ref_label} {m_ref['excluded']}  {cand_label} {m_cand['excluded']}")
    print("GATE", "PASS" if passed else "FAIL")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- long-context study

QSA_BUDGET = 2048  # indexer_budget: below it QSA keeps every block and the oracle is reproducible


def prompts(args: argparse.Namespace) -> None:
    """Oracle readout of every prompt position of a request set. Draw k appends k copies of a fixed
    suffix after the prompt: causally invisible to the readouts, it changes only the reduction
    shapes, so the draws measure the reference's own variation at identical contexts."""
    import torch
    from transformers import AutoTokenizer
    from tools.bench.flash_next import parity

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    suffix = tokenizer.encode(" The text describes", add_special_tokens=False)
    model, head = _build(args.model_dir, args.ple_dir, args.device, args.attn)
    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        done = {(r["id"], r["draw"]) for r in map(json.loads, out_path.read_text(encoding="utf-8").splitlines())}
    shard, shards = (int(x) for x in args.shard.split("/"))
    with out_path.open("a", encoding="utf-8") as out:
        for index, item in enumerate(parity.SETS[args.set]()):
            if index % shards != shard:
                continue
            prompt = _prompt_ids(tokenizer, item)
            for draw in range(args.draws):
                if (item["id"], draw) in done:
                    continue
                t0 = time.time()
                ids = prompt + suffix * draw
                chosen = []
                with torch.no_grad():
                    hidden = model(input_ids=torch.tensor([ids], device=args.device), use_cache=False).last_hidden_state[0]
                    following = torch.tensor(prompt[1:], device=args.device)
                    for start in range(0, len(prompt) - 1, 1024):
                        stop = min(len(prompt) - 1, start + 1024)
                        logp = torch.log_softmax(torch.nn.functional.linear(hidden[start:stop], head), dim=-1)
                        chosen += logp[torch.arange(stop - start, device=args.device), following[start:stop]].tolist()
                del hidden
                out.write(json.dumps({"id": item["id"], "kind": "prompt-all", "draw": draw, "prompt": prompt,
                                      "chosen": chosen}) + "\n")
                out.flush()
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                print(f"{item['id']} draw {draw}: {len(ids)} tokens in {time.time() - t0:.1f} s", flush=True)


def _long_scores(oracle: dict, records: dict, item_id: str) -> dict | None:
    """Mean |engine - oracle| and mean signed difference over the record's positions past the
    QSA budget, or None when the record is missing or misaligned with the oracle's prompt."""
    from tools.bench.flash_next import parity

    o, r = oracle.get((item_id, 0)), records.get((item_id, "prompt"))
    if o is None or r is None:
        return None
    prompt = o["prompt"]
    if r["positions"] != parity.prompt_positions(len(prompt)) or r["tokens"] != [prompt[p + 1] for p in r["positions"]]:
        return None
    pairs = [(e, o["chosen"][p]) for p, e in zip(r["positions"], r["raw"]) if p >= QSA_BUDGET]
    return {"n": len(pairs), "abs": statistics.fmean(abs(e - x) for e, x in pairs),
            "signed": statistics.fmean(e - x for e, x in pairs),
            "values": {p: e for p, e in zip(r["positions"], r["raw"]) if p >= QSA_BUDGET}}


def long_study(args: argparse.Namespace) -> None:
    """The pre-registered long-context reading (docs/research/flash-next-v3-port-2026-09-29.md,
    "Long-context follow-up"): per document, D = mean|v3 - oracle| - mean|v2 - oracle| past the QSA
    budget; a 95% document bootstrap of mean D against a margin of 10% of v2's mean."""
    import random

    oracle = {}
    for line in Path(args.oracle).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        oracle[(r["id"], r["draw"])] = r
    labelled = [_load_records(spec) for spec in [args.reference, args.candidate] + (args.route or [])]
    (ref_label, ref), (cand_label, cand) = labelled[0], labelled[1]
    rows, excluded = [], []
    for item_id in sorted({i for i, _ in oracle}):
        scores = {label: _long_scores(oracle, records, item_id) for label, records in labelled}
        if scores[ref_label] is None or scores[cand_label] is None:
            excluded.append(item_id)
            continue
        row = {"id": item_id, "prompt_tokens": len(oracle[(item_id, 0)]["prompt"]), "positions": scores[ref_label]["n"]}
        for label, s in scores.items():
            if s is not None:
                row[label] = s["abs"]
                row[label + "_signed"] = s["signed"]
        row["D"] = row[cand_label] - row[ref_label]
        positions = sorted(scores[ref_label]["values"])
        if (item_id, 1) in oracle:
            o0, o1 = oracle[(item_id, 0)]["chosen"], oracle[(item_id, 1)]["chosen"]
            row["oracle_draws"] = statistics.fmean(abs(o0[p] - o1[p]) for p in positions)
        for label, s in scores.items():
            if s is not None and label not in (ref_label, cand_label):
                base = scores[ref_label] if label.startswith(ref_label) else scores[cand_label]
                row[label + "_route"] = statistics.fmean(abs(s["values"][p] - base["values"][p]) for p in positions)
        rows.append(row)
    d = [row["D"] for row in rows]
    rng = random.Random(0)
    means = sorted(statistics.fmean(rng.choices(d, k=len(d))) for _ in range(10000))
    low, high = means[249], means[9749]
    margin = 0.10 * statistics.fmean(row[ref_label] for row in rows)
    mean_d = statistics.fmean(d)
    if low > 0 and mean_d > margin:
        reading = "v3 regression on long context"
    elif high <= margin:
        reading = "no v3 regression on long context"
    else:
        reading = "inconclusive"
    for row in rows:
        extra = "  ".join(f"{k} {v:.3f}" for k, v in row.items() if k.endswith("_route") or k == "oracle_draws")
        print(f"{row['id']:15s} P {row['prompt_tokens']:6d} n {row['positions']:3d}  {ref_label} {row[ref_label]:.3f} "
              f"({row[ref_label + '_signed']:+.2f})  {cand_label} {row[cand_label]:.3f} ({row[cand_label + '_signed']:+.2f})  "
              f"D {row['D']:+.3f}  {extra}")
    summary = {"documents": len(rows), "excluded": excluded, "mean_D": mean_d, "ci95": [low, high], "margin": margin,
               "candidate_worse": sum(x > 0 for x in d), "reading": reading,
               ref_label + "_mean": statistics.fmean(row[ref_label] for row in rows),
               cand_label + "_mean": statistics.fmean(row[cand_label] for row in rows)}
    print(json.dumps(summary, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "documents": rows}, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model-dir", required=True)
    r.add_argument("--ple-dir", required=True)
    r.add_argument("--records", action="append", required=True, help="label=path of a parity.py record set")
    r.add_argument("--out", required=True)
    r.add_argument("--device", default="cuda")
    r.add_argument("--attn", default="sdpa", choices=("sdpa", "eager"))
    r.add_argument("--only", default="", help="comma-separated request ids (default: all)")
    g = sub.add_parser("gate")
    g.add_argument("--oracle", required=True)
    g.add_argument("--reference", required=True, help="label=path (v2)")
    g.add_argument("--candidate", required=True, help="label=path (v3)")
    g.add_argument("--out")
    p = sub.add_parser("prompts")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--ple-dir", required=True)
    p.add_argument("--set", default="long")
    p.add_argument("--draws", type=int, default=2)
    p.add_argument("--shard", default="0/1", help="K/N: only documents whose set index is K modulo N")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--attn", default="sdpa", choices=("sdpa", "eager"))
    s = sub.add_parser("long")
    s.add_argument("--oracle", required=True, help="output of `prompts`")
    s.add_argument("--reference", required=True, help="label=path (v2, primary route)")
    s.add_argument("--candidate", required=True, help="label=path (v3, primary route)")
    s.add_argument("--route", action="append", help="label=path of a second route; the label starts with its engine's label")
    s.add_argument("--out")
    args = parser.parse_args()
    {"run": run, "gate": gate, "prompts": prompts, "long": long_study}[args.command](args)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    main()
