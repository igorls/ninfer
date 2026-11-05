"""Opt-in /v1/score contract check against a running real-model server."""

import argparse
import json
import math
from pathlib import Path
import urllib.request


def validate_score(value):
    assert value["object"] == "score"
    assert [row["id"] for row in value["results"]] == ["token", "text"]
    for index, row in enumerate(value["results"]):
        assert "error" not in row, row
        assert row["form"] == ("token" if index == 0 else "text")
        candidates = row["candidate_logprobs"]
        assert len(candidates) == 2
        raw = [candidate["raw_logprob"] for candidate in candidates]
        assert all(math.isfinite(x) and x <= 0 for x in raw)
        maximum = max(raw)
        normalizer = maximum + math.log(sum(math.exp(x - maximum) for x in raw))
        assert abs(sum(math.exp(c["logprob"]) for c in candidates) - 1) < 2e-5
        for candidate in candidates:
            assert abs(candidate["logprob"] - (candidate["raw_logprob"] - normalizer)) < 2e-5
            if index:
                tokens = candidate["tokens"]
                actual = bytes(byte for token in tokens for byte in token["bytes"])
                assert actual == candidate["token"].encode("utf-8"), (actual, candidate)
                assert abs(sum(t["logprob"] for t in tokens) - candidate["raw_logprob"]) < 2e-5
                positions = [t["position"] for t in tokens]
                assert positions == list(range(positions[0], positions[0] + len(positions)))
        if not index:
            assert [c["token_id"] for c in candidates] == [32, 33]
            assert abs(row["outside_mass"] - (1 - sum(math.exp(x) for x in raw))) < 2e-5
        assert math.isfinite(row["entropy"]) and row["entropy"] >= 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-file", type=Path)
    args = parser.parse_args()
    headers = {"content-type": "application/json"}
    if args.api_key_file:
        headers["Authorization"] = "Bearer " + args.api_key_file.read_text(encoding="utf-8").rstrip("\r\n \t")
    # Short options catch candidates previously rejected after subtracting the thinking opener;
    # long options catch silent truncation to the final token. UTF-8 checks byte ownership too.
    for candidates in [["Yes", "No."], ["The invoice is overdue.", "The invoice was paid."],
                       ["Pagamento pendente: não.", "Pagamento recebido: sim."]]:
        body = {
            "model": args.model,
            "messages": [{"role": "system", "content": "The invoice is unpaid."}],
            "questions": [
                {"id": "token", "content": "Is it overdue? A for yes, B for no.",
                 "candidates": [32, 33]},
                {"id": "text", "content": "State the payment status.", "candidates": candidates},
            ],
            "chat_template_kwargs": {"enable_thinking": False},
            "top_logprobs": 2,
        }
        for _ in range(2):
            request = urllib.request.Request(args.url.rstrip("/") + "/v1/score",
                                             data=json.dumps(body).encode(),
                                             headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                validate_score(json.load(response))
    print("PASS token and complete text candidates, repeated requests, UTF-8 and normalization")


if __name__ == "__main__":
    main()
