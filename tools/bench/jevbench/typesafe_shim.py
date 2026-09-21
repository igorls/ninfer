"""TypeSafe-compatible ``POST /v1/systemone`` shim in front of a running ninfer-serve.

JevBench's stock ``typesafe`` adapter speaks the TypeSafe wire format. This shim translates that
request into the same NInfer Chat Completions request the ``ninfer_native`` adapter builds (lettered
options, one greedy token, ``logprob_candidates``, thinking off, read-only cache participation) and
returns the answer in TypeSafe's shape, so the benchmark can be run against NInfer without a new
adapter:

    python tools/bench/jevbench/typesafe_shim.py --ninfer http://127.0.0.1:8010/v1 \
        --ninfer-model qwen3.8-27b --port 8000

    python -m jevbench.cli run --adapter typesafe --endpoint http://127.0.0.1:8000 --key-env '' \
        --model ninfer-qwen3.8-27b ...

The requested ``model`` selects the logit temperature: ``<name>`` reads the raw distribution,
``<name>-t1.5`` divides the option logits by 1.5 before the softmax (any positive number after
``-t``). Both are reported back in the ``model`` field and in ``runtime``.

Answer mapping (the inverse of JevBench's ``typesafe`` adapter):
  noul   -> {"type": "noul", "noul": p_yes}
  choice -> {"type": "choice", "choice": argmax, "probabilities": {option: p}}
  score  -> {"type": "score", "probabilities": {"0": p0, "1": p1, ...}}
Usage reports NInfer's prompt tokens and zero output tokens: the decision is a readout, not a
generation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ninfer_native  # noqa: E402  (sibling module; the adapter's prompt is the one source of truth)


class _Task:
    """The subset of jevbench.tasks.Task that ninfer_native.build_request reads."""

    def __init__(self, state, question):
        self.id = "shim"
        self.state = state
        self.question = question
        qtype = question["type"]
        crit = question.get("criteria")
        if qtype == "noul":
            self.labels = ["no", "yes"]
        elif qtype == "score":
            if not isinstance(crit, list) or not crit:
                raise ValueError("score question needs a list of level criteria")
            self.labels = [str(i) for i in range(len(crit))]
        elif qtype == "choice":
            if not isinstance(crit, dict) or not crit:
                raise ValueError("choice question needs a criteria object")
            self.labels = list(crit.keys())
        else:
            raise ValueError(f"unsupported question type {qtype!r}")


class Shim:
    def __init__(self, ninfer_endpoint, ninfer_model, api_key=""):
        self.adapter = ninfer_native.NInferNativeAdapter(endpoint=ninfer_endpoint, model=ninfer_model)
        self.api_key = api_key

    @staticmethod
    def temperature_of(model_name):
        m = re.search(r"-t([0-9]*\.?[0-9]+)$", model_name or "")
        return float(m.group(1)) if m else 1.0

    def decide(self, body):
        questions = body.get("questions") or {}
        if not isinstance(questions, dict) or len(questions) != 1:
            raise ValueError("exactly one question is supported per request")
        (key, question), = questions.items()
        task = _Task(body.get("state"), question)
        temperature = self.temperature_of(body.get("model"))
        if not temperature > 0:
            raise ValueError("temperature must be positive")
        request = self.adapter.build_request(task)
        data = json.dumps(request).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.adapter.endpoint}/chat/completions", data=data,
                                     headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            out = json.loads(resp.read())
        entry = out["choices"][0]["logprobs"]["content"][0]
        cands = entry["candidate_logprobs"]
        if [c["token"] for c in cands] != request["logprob_candidates"]:
            raise RuntimeError("candidate order differs from the request")
        scaled = [float(c["raw_logprob"]) / temperature for c in cands]
        peak = max(scaled)
        weights = [math.exp(x - peak) for x in scaled]
        total = sum(weights)
        probs = {lab: w / total for lab, w in zip(task.labels, weights)}
        qtype = question["type"]
        if qtype == "noul":
            answer = {"type": "noul", "noul": probs["yes"]}
        elif qtype == "choice":
            answer = {"type": "choice", "choice": max(task.labels, key=lambda k: probs[k]),
                      "probabilities": probs}
        else:
            answer = {"type": "score", "probabilities": probs}
        usage = out.get("usage") or {}
        return {
            "model": body.get("model") or self.adapter.model,
            "answers": {key: answer},
            "usage": {"input_tokens": usage.get("prompt_tokens"), "output_tokens": 0},
            "runtime": {"engine": "ninfer", "engine_model": out.get("model"),
                        "probability_origin": "native-option-letter-softmax",
                        "logit_temperature": temperature,
                        "outside_candidate_mass": 1.0 - sum(math.exp(float(c["raw_logprob"]))
                                                          for c in cands)},
        }


def make_handler(shim):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, status, payload):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/systemone":
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                self._send(200, shim.decide(body))
            except urllib.error.HTTPError as e:
                self._send(502, {"error": f"ninfer HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace')}"})
            except (ValueError, KeyError, TypeError, RuntimeError) as e:
                self._send(400, {"error": str(e)})
    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ninfer", default="http://127.0.0.1:8010/v1")
    ap.add_argument("--ninfer-model", default="qwen3.8-27b")
    ap.add_argument("--ninfer-api-key", default="")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(Shim(args.ninfer, args.ninfer_model,
                                                                            args.ninfer_api_key)))
    print(f"typesafe shim on http://{args.host}:{args.port}/v1/systemone -> {args.ninfer} ({args.ninfer_model})",
          flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
