"""Build the kld-400k-v1 corpus and its BF16 next-token reference distributions.

The corpus is assembled from pinned public sources and tokenized with the official Qwen3.8-27B
tokenizer. Every sequence is an independent window of at most 4,096 tokens. The official BF16
checkpoint (Hugging Face Transformers) then evaluates each sequence once; for every predictor
position i the reference stores the 32 most likely next tokens with their FP32 log-probabilities
(computed from FP32 logits of the BF16 final hidden state) and the log-probability of the actual
token i+1. ``ninfer-perplexity --reference`` scores an artifact on exactly these token ids and
reports KL(BF16 || artifact), top-1 agreement and NLL.

The same pass records, per corpus domain, the maximum absolute input of every Text projection
site, and writes the NVFP4 activation calibration of ``tools.convert.calibrate_nvfp4`` from its
fixed corpus, so one BF16 load serves both.

    python3 -m eval.kld.build_reference --model /path/to/Qwen3.8-27B --out OUT_DIR

Output: OUT_DIR/corpus.json, OUT_DIR/parts/NNNNN.bin (one record per sequence),
OUT_DIR/site_amax.json, OUT_DIR/qwen3_8_27b_nvfp4full_calibration.json. ``--assemble`` joins
verified parts into OUT_DIR/kld-400k-v1.reference.

Reference file (little endian): b"NINFKLD1", uint64 header bytes, UTF-8 JSON header, then one
record per header sequence: int32 tokens[T], int32 top_ids[T-1][K], float32 top_logprobs[T-1][K],
float32 target_logprobs[T-1].
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import time
import urllib.request

CORPUS_ID = "kld-400k-v1"
MAGIC = b"NINFKLD1"
TOP_K = 32
SEQUENCE_TOKENS = 4096
MODEL = {"repository": "Qwen/Qwen3.8-27B", "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"}
REPOSITORY = Path(__file__).resolve().parents[2]

# (domain, repository text file, tokens): the first tokens of each committed perplexity stream.
REPOSITORY_STREAMS = (
    ("english_reference", "eval/corpora/perplexity-1m/data/wikitext/00.txt", 40960),
    ("english_long_form", "eval/corpora/perplexity-1m/data/pg19/00.txt", 40960),
    ("chinese_reference", "eval/corpora/perplexity-1m/data/zhwiki/00.txt", 40960),
    ("cpp_code", "eval/corpora/perplexity-1m/data/ninfer/00.txt", 40960),
)
CPYTHON = {
    "repository": "python/cpython",
    "tag": "v3.13.0",
    "files": (
        "Lib/heapq.py",
        "Lib/bisect.py",
        "Lib/textwrap.py",
        "Lib/functools.py",
        "Lib/dataclasses.py",
        "Lib/statistics.py",
        "Lib/fractions.py",
        "Lib/argparse.py",
    ),
    "tokens": 32768,
}
MATH = {
    "repository": "HuggingFaceH4/MATH-500",
    "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    "file": "test.jsonl",
    "tokens": 24576,
}
BELEBELE = {
    "repository": "facebook/belebele",
    "revision": "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2",
    "languages": (
        "por_Latn",
        "spa_Latn",
        "fra_Latn",
        "deu_Latn",
        "rus_Cyrl",
        "jpn_Jpan",
        "arb_Arab",
        "hin_Deva",
    ),
    "tokens": 8192,
}
S1K = {
    "repository": "simplescaling/s1K-1.1",
    "revision": "96c411f1fe4c49d20f0e2a1565f61e1a28b0b84d",
    "file": "data/train-00000-of-00001.parquet",
    "rows": 20,
}
ULTRACHAT = {
    "repository": "HuggingFaceH4/ultrachat_200k",
    "revision": "8049631c405ae6576f93f445c6b8166f76f5505a",
    "file": "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",
    "rows": 24,
}


def build_corpus(tokenizer) -> tuple[list[dict], dict]:
    """Deterministic sequences [{id, domain, ids}] plus source provenance."""
    sequences: list[dict] = []
    provenance: dict = {}

    def add(domain: str, windows: list[list[int]], label: str) -> None:
        for index, ids in enumerate(windows):
            sequences.append({"id": f"{label}-{index:02d}", "domain": domain, "ids": ids})

    def encode(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    from tools.convert.calibrate_nvfp4 import hf_dataset_file, token_windows

    for domain, relative, tokens in REPOSITORY_STREAMS:
        data = (REPOSITORY / relative).read_bytes()
        provenance[relative] = {"sha256": _sha256(data), "tokens": tokens}
        add(domain, token_windows(encode(data.decode("utf-8")), tokens), Path(relative).parent.name)

    texts = []
    tag = CPYTHON["tag"]
    for relative in CPYTHON["files"]:
        url = f"https://raw.githubusercontent.com/{CPYTHON['repository']}/{tag}/{relative}"
        data = urllib.request.urlopen(url, timeout=60).read()
        provenance[f"cpython/{tag}/{relative}"] = {"url": url, "sha256": _sha256(data)}
        texts.append(data.decode("utf-8"))
    add("python_code", token_windows(encode("\n\n".join(texts)), CPYTHON["tokens"]), "cpython")

    path = hf_dataset_file(MATH["repository"], MATH["revision"], MATH["file"])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    provenance[MATH["repository"]] = {"revision": MATH["revision"], "sha256": _sha256(path.read_bytes())}
    text = "\n\n".join(f"Problem: {row['problem']}\nSolution: {row['solution']}" for row in rows)
    add("math", token_windows(encode(text), MATH["tokens"]), "math500")

    for language in BELEBELE["languages"]:
        path = hf_dataset_file(BELEBELE["repository"], BELEBELE["revision"], f"data/{language}.jsonl")
        passages = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                passage = json.loads(line)["flores_passage"]
                if passage not in passages:
                    passages.append(passage)
        provenance[f"{BELEBELE['repository']}/{language}"] = {
            "revision": BELEBELE["revision"],
            "sha256": _sha256(path.read_bytes()),
        }
        add("multilingual", token_windows(encode("\n\n".join(passages)), BELEBELE["tokens"]), language)

    import pandas

    path = hf_dataset_file(S1K["repository"], S1K["revision"], S1K["file"])
    provenance[S1K["repository"]] = {"revision": S1K["revision"], "sha256": _sha256(path.read_bytes())}
    frame = pandas.read_parquet(path)
    windows = []
    for row in frame.head(S1K["rows"]).itertuples():
        messages = [
            {"role": "user", "content": row.question},
            {
                "role": "assistant",
                "content": row.deepseek_attempt,
                "reasoning_content": row.deepseek_thinking_trajectory,
            },
        ]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
        windows.append(list(ids)[:SEQUENCE_TOKENS])
    add("chat_reasoning", windows, "s1k")

    path = hf_dataset_file(ULTRACHAT["repository"], ULTRACHAT["revision"], ULTRACHAT["file"])
    provenance[ULTRACHAT["repository"]] = {
        "revision": ULTRACHAT["revision"],
        "sha256": _sha256(path.read_bytes()),
    }
    frame = pandas.read_parquet(path)
    windows = []
    for row in frame.head(ULTRACHAT["rows"]).itertuples():
        messages = [{"role": m["role"], "content": m["content"]} for m in row.messages]
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=False, enable_thinking=False
        )
        windows.append(list(ids)[:SEQUENCE_TOKENS])
    add("chat_general", windows, "ultrachat")
    return sequences, provenance


def reference_record(model_parts, ids: list[int]) -> bytes:
    """One sequence's reference record."""
    import torch

    language_model, head_fp32 = model_parts
    device = head_fp32.device
    tokens = torch.tensor([ids], device=language_model.device)
    with torch.inference_mode():
        hidden = language_model(input_ids=tokens, use_cache=False).last_hidden_state[0]
        predictors = len(ids) - 1
        targets = torch.tensor(ids[1:], device=device)
        top_ids, top_lp, target_lp = [], [], []
        for begin in range(0, predictors, 1024):
            end = min(predictors, begin + 1024)
            logits = hidden[begin:end].to(device).float() @ head_fp32.T
            logprobs = torch.log_softmax(logits, dim=-1)
            values, indices = torch.topk(logprobs, TOP_K, dim=-1)
            top_ids.append(indices.to(torch.int32))
            top_lp.append(values)
            target_lp.append(logprobs.gather(1, targets[begin:end, None])[:, 0])
    buffer = io.BytesIO()
    buffer.write(torch.tensor(ids, dtype=torch.int32).numpy().tobytes())
    buffer.write(torch.cat(top_ids).cpu().numpy().tobytes())
    buffer.write(torch.cat(top_lp).cpu().numpy().tobytes())
    buffer.write(torch.cat(target_lp).cpu().numpy().tobytes())
    return buffer.getvalue()


def header(corpus: dict) -> dict:
    return {
        "format": "ninfer-kld-reference-v1",
        "corpus_id": corpus["corpus_id"],
        "model": corpus["model"],
        "top_k": corpus["top_k"],
        "logprob_source": corpus["logprob_source"],
        "sequences": [
            {"id": s["id"], "domain": s["domain"], "tokens": s["tokens"]}
            for s in corpus["sequences"]
        ],
    }


def assemble(out: Path) -> Path:
    corpus = json.loads((out / "corpus.json").read_text(encoding="utf-8"))
    target = out / f"{CORPUS_ID}.reference"
    encoded = json.dumps(header(corpus), ensure_ascii=False).encode("utf-8")
    with target.open("xb") as stream:
        stream.write(MAGIC + struct.pack("<Q", len(encoded)) + encoded)
        for index, sequence in enumerate(corpus["sequences"]):
            data = (out / "parts" / f"{index:05d}.bin").read_bytes()
            t, k = sequence["tokens"], corpus["top_k"]
            if len(data) != 4 * (t + 2 * (t - 1) * k + (t - 1)):
                raise ValueError(f"part {index} has {len(data)} bytes")
            if _sha256(data) != sequence["record_sha256"]:
                raise ValueError(f"part {index} does not match its recorded SHA-256")
            stream.write(data)
    return target


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", type=Path, help="official BF16 checkpoint directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--assemble", action="store_true", help="join verified parts only")
    parser.add_argument("--calibration-corpus", default="broad-v1",
                        choices=("cometkim-v1", "broad-v1"))
    args = parser.parse_args(argv)
    if args.assemble:
        print(assemble(args.out))
        return
    import torch

    sys.path.insert(0, str(REPOSITORY))
    from tools.convert.calibrate_nvfp4 import (
        SiteMaxima,
        all_sites,
        calibration_document,
        corpus_windows,
        load_bf16,
        measure_corpus,
    )
    from tools.convert.official_recipes import nvfp4full_local_sites

    started = time.perf_counter()
    (args.out / "parts").mkdir(parents=True, exist_ok=True)
    model, tokenizer, language_model, lm_head = load_bf16(args.model)
    print(f"loaded BF16 model in {time.perf_counter() - started:.0f}s", flush=True)
    layer_types = language_model.config.layer_types
    sites = all_sites(layer_types)

    calibration = args.out / "qwen3_8_27b_nvfp4full_calibration.json"
    if not calibration.exists():
        local = nvfp4full_local_sites(layer_types)
        windows, provenance = corpus_windows(args.calibration_corpus, tokenizer)
        maxima = measure_corpus(windows, language_model, sites, args.calibration_corpus)
        document = calibration_document(
            {name: maxima[name] for name in local},
            local,
            model=MODEL,
            corpus=args.calibration_corpus,
            tokens=sum(len(window) for window in windows),
            provenance=provenance,
        )
        calibration.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        (args.out / "site_amax_calibration.json").write_text(
            json.dumps(maxima, indent=1) + "\n", encoding="utf-8"
        )
        print(f"calibration: {len(local)} sites, {document['corpus_tokens']} tokens", flush=True)

    corpus_path = args.out / "corpus.json"
    if corpus_path.exists():
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    else:
        sequences, provenance = build_corpus(tokenizer)
        tokenizer_file = Path(args.model) / "tokenizer.json"
        corpus = {
            "corpus_id": CORPUS_ID,
            "model": MODEL,
            "tokenizer_sha256": _sha256(tokenizer_file.read_bytes()),
            "top_k": TOP_K,
            "logprob_source": "log_softmax of FP32 logits: BF16 final hidden x FP32(lm_head)",
            "provenance": provenance,
            "sequences": [
                {
                    "id": s["id"],
                    "domain": s["domain"],
                    "tokens": len(s["ids"]),
                    "ids": s["ids"],
                }
                for s in sequences
            ],
        }
        corpus_path.write_text(json.dumps(corpus) + "\n", encoding="utf-8")
    total = sum(s["tokens"] for s in corpus["sequences"])
    print(f"corpus: {len(corpus['sequences'])} sequences, {total} tokens", flush=True)

    head_fp32 = lm_head.weight.detach().float()
    hooks = SiteMaxima(language_model, sites)
    amax_path = args.out / "site_amax.json"
    # A resumed run keeps the maxima of sequences an earlier session already evaluated.
    earlier = json.loads(amax_path.read_text(encoding="utf-8")) if amax_path.exists() else {}

    def merged_amax() -> dict:
        result = {bucket: dict(values) for bucket, values in earlier.items()}
        for bucket, values in hooks.results().items():
            target = result.setdefault(bucket, {})
            for name, value in values.items():
                target[name] = max(value, target.get(name, 0.0))
        return result

    for index, sequence in enumerate(corpus["sequences"]):
        part = args.out / "parts" / f"{index:05d}.bin"
        if part.exists() and "record_sha256" in sequence:
            continue
        begin = time.perf_counter()
        hooks.bucket = sequence["domain"]
        data = reference_record((language_model, head_fp32), sequence["ids"])
        hooks.bucket = None
        temporary = part.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(part)
        sequence["record_sha256"] = _sha256(data)
        corpus_path.write_text(json.dumps(corpus) + "\n", encoding="utf-8")
        amax_path.write_text(json.dumps(merged_amax(), indent=1) + "\n", encoding="utf-8")
        print(
            f"[{index + 1}/{len(corpus['sequences'])}] {sequence['id']} {sequence['tokens']} "
            f"tokens {time.perf_counter() - begin:.1f}s",
            flush=True,
        )
    hooks.close()
    print(f"REFERENCE_COMPLETE {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
