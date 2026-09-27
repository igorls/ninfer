"""JevBench adapter for NInfer: the model's own next-token distribution over option letters.

One non-streaming Chat Completions request per decision against a running ``ninfer-serve``.
The prompt renders the state, the instructions and the exact label set as lettered options
(A, B, ...). The request asks for one greedy token with ``logprobs`` and NInfer's
``logprob_candidates`` extension set to the option letters, so the response carries the model's
log-softmax renormalised over exactly those letters at the first generated position. Nothing is
parsed out of generated text; the distribution is ``native``. Alternatives beyond the candidate
list are not requested (``top_logprobs: 0``), which keeps the readout on the device.

``prompt_cache_read_only: true`` (NInfer extension) lets the request reuse an already published
prefix but publish nothing itself, so a benchmark sweep does not evict other conversations'
cached state on a shared server. Thinking is disabled per request through
``chat_template_kwargs``; a reasoning run is a different entrant and would carry its own label.

Mapping from the canonical record (fixed before any run):
  * noul   -> options in label order ["no", "yes"]; descriptions from criteria "false"/"true"
  * choice -> options in label order; description = criteria[label] (or the bare label)
  * score  -> options "0".."k-1" in label order; description = the level text
The letters follow label order, so the first label is always "A".

Cost: a self-hosted GPU has no tariff. When the runner is given ``--price-in-per-m`` and
``--price-out-per-m``, the usual "est." hosted size-class price applies to the measured input
tokens; output tokens are reported as 0 because the decision is a readout, not a generation
(the single greedy token is discarded, as for the other direct-logit entrants).
"""

from __future__ import annotations

import json
import math
import os
import string

try:
    from .base import DecisionResult, http_post_json
except ImportError:  # imported outside the jevbench package (typesafe_shim.py uses build_request only)
    DecisionResult = None
    http_post_json = None

# Request option handled by the adapter, not the server: divide the option logits by this value
# before the softmax. Fixed before a run and recorded in the manifest; it changes calibration only,
# never the argmax. 1.0 reads the model's raw distribution.
LOGIT_TEMPERATURE_OPTION = "logit_temperature"

_SYSTEM = (
    "You are a decision engine. You read a state and answer one bounded question by "
    "choosing exactly one of the given options. Reply with the option letter only."
)


class NInferNativeAdapter:
    name = "ninfer_native"
    cost_basis = "self_hosted_gpu"

    def __init__(self, endpoint=None, model=None, key_env="", timeout_s=120.0,
                 price_input_per_m=None, price_output_per_m=None):
        self.endpoint = (endpoint or os.environ.get("NINFER_ENDPOINT")
                         or "http://127.0.0.1:8080/v1").rstrip("/")
        self.model = model or os.environ.get("NINFER_MODEL") or ""
        self.key_env = key_env
        self.timeout_s = timeout_s
        self.price_input_per_m = price_input_per_m
        self.price_output_per_m = price_output_per_m
        self.request_options = {}

    @staticmethod
    def options(task):
        """(label, description) pairs in label order."""
        qtype = task.question["type"]
        crit = task.question.get("criteria")
        labels = [str(x) for x in task.labels]
        if qtype == "noul":
            crit = crit or {}
            key = {"yes": "true", "no": "false"}
            return [(lab, crit.get(key.get(lab, lab)) or lab) for lab in labels]
        if qtype == "score":
            return [(lab, crit[int(lab)] if crit and int(lab) < len(crit) else lab)
                    for lab in labels]
        crit = crit if isinstance(crit, dict) else {}
        return [(lab, crit.get(lab) or lab) for lab in labels]

    def build_request(self, task) -> dict:
        opts = self.options(task)
        if len(opts) > len(string.ascii_uppercase):
            raise ValueError(f"{task.id}: {len(opts)} options exceed the letter range")
        letters = string.ascii_uppercase[: len(opts)]
        lines = [f"{letter}. {label}: {desc}" if desc != label else f"{letter}. {label}"
                 for letter, (label, desc) in zip(letters, opts)]
        state = task.state if isinstance(task.state, str) else json.dumps(
            task.state, ensure_ascii=False, indent=1)
        question = task.question["instructions"]
        question += ("\n\nLevels, lowest to highest:" if task.question["type"] == "score"
                     else "\n\nOptions:")
        user = (f"State:\n{state}\n\n{question}\n" + "\n".join(lines)
                + "\n\nAnswer with the letter of the best option.")
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user},
            ],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 0,
            "logprob_candidates": list(letters),
            "prompt_cache_read_only": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        for k, v in (self.request_options or {}).items():
            if k == LOGIT_TEMPERATURE_OPTION:
                continue  # adapter-side calibration, never sent to the server
            if v is None:
                body.pop(k, None)
            else:
                body[k] = v
        return body

    @property
    def logit_temperature(self) -> float:
        """Divisor applied to the option logits before the softmax; 1.0 is the raw distribution."""
        value = float((self.request_options or {}).get(LOGIT_TEMPERATURE_OPTION, 1.0))
        if not value > 0:
            raise ValueError(f"{LOGIT_TEMPERATURE_OPTION} must be positive")
        return value

    def run(self, task) -> DecisionResult:
        key = os.environ.get(self.key_env, "") if self.key_env else ""
        if self.key_env and not key:
            return DecisionResult(adapter=self.name, ok=False, probs_source="native",
                                  model=self.model, error=f"missing env key {self.key_env}")
        body = self.build_request(task)
        headers = {"Content-Type": "application/json",
                   **({"Authorization": f"Bearer {key}"} if key else {})}
        try:
            status, parsed, latency = http_post_json(
                f"{self.endpoint}/chat/completions", body, headers, self.timeout_s)
        except ConnectionError as e:
            return DecisionResult(adapter=self.name, ok=False, error=str(e),
                                  probs_source="native", model=self.model, request_body=body)
        res = DecisionResult(adapter=self.name, ok=False, status=status, latency_s=latency,
                             probs_source="native", model=self.model, raw=parsed,
                             request_body=body)
        if status != 200 or not isinstance(parsed, dict):
            res.error = f"HTTP {status}: {str(parsed)[:300]}"
            return res
        res.model = parsed.get("model") or self.model
        usage = parsed.get("usage") or {}
        res.usage = {"input_tokens": usage.get("prompt_tokens"), "output_tokens": 0,
                     "cached_tokens": (usage.get("prompt_tokens_details") or {}).get(
                         "cached_tokens")}
        try:
            entry = parsed["choices"][0]["logprobs"]["content"][0]
            cands = entry["candidate_logprobs"]
            letters = body["logprob_candidates"]
            if [c["token"] for c in cands] != letters:
                raise ValueError("candidate order differs from the request")
            labels = [str(x) for x in task.labels]
            temperature = self.logit_temperature
            scaled = [float(c["raw_logprob"]) / temperature for c in cands]
            peak = max(scaled)
            weights = [math.exp(x - peak) for x in scaled]
            total = sum(weights)
            probs = {lab: w / total for lab, w in zip(labels, weights)}
            outside = 1.0 - sum(math.exp(float(c["raw_logprob"])) for c in cands)
        except (KeyError, IndexError, TypeError, ValueError) as e:
            res.error = f"native distribution missing: {type(e).__name__}: {e}"
            return res
        res.probs = probs
        # The runner copies a top-level "runtime" block of the raw body into the per-item record.
        parsed["runtime"] = {"probability_origin": "native-option-letter-softmax",
                             "readout": "logprob_candidates at the first generated position",
                             "outside_candidate_mass": outside,
                             "logit_temperature": temperature,
                             "greedy_token": entry.get("token"),
                             "thinking": False}
        res.ok = True
        return res

    def reserve_estimate(self, task):
        if self.price_input_per_m is None or self.price_output_per_m is None:
            return 0.0
        return 8_192 * self.price_input_per_m / 1e6
