# JevBench on NInfer

[JevBench](https://github.com/fstandhartinger/jevbench) measures Jev-class decision models: the
system reads a state and a bounded rubric and returns a probability for every option. The board's
JevBench Score is the geometric mean of Intelligence, Calibration, Speed and Cost.

## Files

- `ninfer_native.py`: a JevBench adapter in the repository's adapter contract. One non-streaming
  Chat Completions request per decision against a running `ninfer-serve`: the state, instructions
  and lettered options in one user message, `max_tokens 1`, `logprobs` with NInfer's
  `logprob_candidates` set to the option letters, thinking off, `prompt_cache_read_only` so the
  sweep publishes nothing into the server's context cache. The distribution is the model's own
  (`native`), renormalised over the letters. An adapter-side request option `logit_temperature`
  divides the option logits before the softmax; it changes calibration only, never the argmax, and
  is recorded in the run manifest. Default 1.0.
- `typesafe_shim.py`: a `POST /v1/systemone` server in TypeSafe's wire format in front of
  `ninfer-serve`, so JevBench's stock `typesafe` adapter runs unchanged. It builds exactly the
  request `ninfer_native.py` builds. The requested `model` name selects the temperature:
  `ninfer-qwen3.8-27b` reads the raw distribution, `ninfer-qwen3.8-27b-t1.5` divides the logits
  by 1.5.
- `run_public.py`: copies the adapter into a JevBench checkout, runs the 231 public decisions
  (easy, original = standard, hard) through JevBench's own serial runner, scores the public subset
  with the board's `composite_v12`, and compares accuracy item by item with every published
  system's per-task outcomes. The judge tier and the held-out items are not public, so the printed
  score is an estimate; only the benchmark author can produce the official number.

## Running

```powershell
git clone --depth 1 https://github.com/fstandhartinger/jevbench.git $env:TEMP\jevbench
python tools/bench/jevbench/run_public.py --jevbench $env:TEMP\jevbench `
    --endpoint http://127.0.0.1:8010/v1 --model qwen3.8-27b `
    --out profiles/bench/jevbench-<date>-public [--logit-temperature 1.5] [--price-in-per-m 0.15]
```

Through the shim with the author's adapter (the ledger needs POSIX `fcntl`; on Windows stub it as
`run_public.py` does):

```sh
python tools/bench/jevbench/typesafe_shim.py --ninfer http://127.0.0.1:8010/v1 \
    --ninfer-model qwen3.8-27b --port 8000
python -m jevbench.cli run --tasks datasets/public/hard.jsonl --adapter typesafe \
    --endpoint http://127.0.0.1:8000 --key-env '' --model ninfer-qwen3.8-27b-t1.5 \
    --reserve-usd 0 --cost-basis self_hosted_gpu --results RUN/results.jsonl --raw-dir RUN/raw
```

Cost follows the board's rule for self-hosted systems: the base model's public hosted tariff times
the measured input tokens, output free. Latency is caller wall time; the board multiplies
self-hosted latency by two and adds 0.15 s.

## Results on the public items (2026-09-21, RTX PRO 6000 Blackwell, one request at a time)

| system | easy | standard | hard | hard ECE | standard p50 | score estimate |
|---|---|---|---|---|---|---|
| Qwen3.8-27B NVFP4, raw | 100 % | 97.2 % | 70.3 % | 0.113 | 0.047 s | 71.2 |
| Qwen3.8-27B NVFP4, temperature 1.5 | 100 % | 97.2 % | 70.3 % | 0.077 | 0.047 s | 72.8 |
| Qwen3.8-Flash-Next mixed, raw | 100 % | 98.6 % | 79.3 % | 0.096 | 0.090 s | 71.5 |

Jev 1.13.0 scores 100 / 98.6 / 73.0 % on the same public items. The 27B rows are the fork at
`d2982875` (read-only requests prefill in one pass); the temperature was chosen on these public
items after the raw run and is declared as such. Flash-Next's temperature sweep found 1.0 already
optimal. Detailed notes live beside each run under `profiles/bench/` (not tracked).
