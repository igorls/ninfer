# JevBench on NInfer

[JevBench](https://github.com/fstandhartinger/jevbench) measures Jev-class decision models: the
system reads a state and a bounded rubric and returns a probability for every option. The current
v1.4 score combines Intelligence, Calibration, Speed and Cost with gates and sealed-task adjustments;
see the [v1.4 method](https://github.com/fstandhartinger/jevbench/blob/main/docs/METHOD-v1.4.md).

## Paired Qwen3.8-27B NVFP4 study

`decision_study.py` implements the first comparisons in the
[research dossier](../../../docs/research/jevbench-qwen-intelligence-dossier.md) against an existing
HTTP server. The target is **Qwen3.8-27B NVFP4**. It makes serial requests and does not start,
restart, reconfigure, or switch the model service. It uses Python 3.11's standard library and the
existing submitted adapter; no EvalScope dependency or new inference implementation is needed for
these task-level protocol comparisons.

The study keeps these methods separate:

| Method | Intervention |
|---|---|
| `submitted` | Exact `ninfer_native.py` prompt, canonical dataset label order, raw first-token probabilities |
| `shim` | Same prompt, but choice order follows the criteria object's insertion order, as the submitted TypeSafe shim does |
| `native` | Actual `/v1/systemone` call, temperature 1, choices explicitly ordered by dataset labels |
| `framed` | Evidence/Criterion/Options framing with the same letter mapping |
| `semantic` | Same framed prompt with Yes/No or numeric tokens; only binary/ordinal tasks |
| `rotated` | Rotate choice descriptions and semantic IDs by one position, keeping A/B/etc. as the output labels; only unordered choices |
| `two_order` | Average submitted and rotated probabilities after aligning semantic IDs; charge both passes |
| `short` | Same submitted prompt, thinking off, up to 64 generated tokens; strict answer parsing |
| `anthropic_short` | Same non-thinking generation via `/v1/messages`, to control for the reasoning route's protocol |
| `reason_1024`, `reason_2048` | Same decision prompt, per-request Engine thinking cap, then a generated answer |

Anthropic's supported per-request cap starts at 1,024 tokens. The preliminary dossier's 128/512
caps would require a differently configured server; this study uses 1,024/2,048 without changing
the resident service. The full output allowance adds 256 tokens for Engine control suffix and
answer. Thinking and answer blocks are separate. An incomplete generation or an answer other
than a single declared letter (optional trailing `.` or `)`) is invalid and counts as wrong.
Generated answers have **no probability distribution**: the study does not invent one-hot
confidence or treat verbal confidence as native logits. Post-reasoning native probability
measurement is a separate follow-up requiring a verified continuation boundary.

Example on Windows (set `$studyPython` to the installed Python 3.11 interpreter):

```powershell
$studyPython = uv python find 3.11
& $studyPython tools/bench/jevbench/make_decision_cases.py `
    --out profiles/bench/decision-study/oracle.jsonl
& $studyPython tools/bench/jevbench/decision_study.py `
    --tasks "public_hard=$env:TEMP/jevbench/datasets/public/hard.jsonl" `
    --tasks oracle_diagnostic=profiles/bench/decision-study/oracle.jsonl `
    --endpoint http://127.0.0.1:8010 --model qwen3.8-27b `
    --artifact 'Qwen3.8-27B NVFP4; state the actual KV format and hardware here' `
    --groups-per-family 2 --out profiles/bench/decision-study/pilot --plan
```

Remove `--plan` to run. The output directory must be new. `--tasks` accepts multiple
`COHORT=path.jsonl` inputs in JevBench's task format. A source checkout supplies public data only;
this runner neither modifies it nor reports an official v1.4 score. `--methods` selects explicit
comparisons, and `--key-env` reads an optional endpoint credential without saving it. The served
model ID must match the requested target. Artifact and KV format are operator-supplied provenance,
not discovered or validated by the model-list endpoint.

Selection hashes scenario group IDs within each cohort/family, before any request or outcome.
Related examples in one group are kept together. Both task order and method order are randomized
with the recorded seed. The authored oracle cohort covers integer accounting, inclusive dates,
rule precedence, temporal lookup, ordinal rules, and resource constraints. Gold labels are
calculated in `make_decision_cases.py`; provenance and gold explanations are never prompt inputs.
This small synthetic diagnostic does not establish broad generalization.

Each run writes a selection/source manifest, full requests and responses in `results.jsonl`,
`summary.json`, and `report.md`. Reports include failures in applicable-task accuracy, paired
fixed/broken counts, scenario-bootstrap intervals, family results, and p50/p95 caller latency.
NLL, multiclass Brier and ten-bin ECE are computed only on valid native distributions, with that
denominator explicit. They use the task's expected label, not a soft gold distribution.
The two-order method reuses its separately measured component reads; its summed latency is a
two-pass cost estimate, not a directly measured combined endpoint. Latency is from the existing
service and makes no exclusive-GPU or RTX 5090 performance claim.

Cache policy is recorded in each exact request. Chat readouts use read-only participation;
Anthropic requests use that endpoint's default publication policy. Randomized ordering does not
make those cache histories identical. The initial pilot found greedy answer changes after prefix
publication, so its scores describe these observed service routes rather than a cache-controlled
isolation of reasoning. See the [pilot findings](../../../docs/research/qwen3.8-27b-decision-study.md).

Verification without a model:

```powershell
& $studyPython -m unittest discover -s tests -p test_decision_study.py -v
```

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

## Progressive reasoning router training

This is separate from a JevBench score. It trains a tiny head on frozen Qwen3.8-27B NVFP4
features and independently checked outcomes of direct, 1,024-token and 2,048-token reasoning.
The Engine uses a fixed pre-think boundary, no cache and no speculation for collection.
The collector rejects other registered model identities. See the
[design and current experiment](../../../docs/research/learned-reasoning-router.md).

Build the collector, prepare authored exact-oracle examples, and collect through the public Engine:

```powershell
$py = uv python find 3.11
cmake --build build-win --config Release --target ninfer-reasoning-collect -j
& $py tools/bench/jevbench/reasoning_router.py prepare --scenarios 32 --out requests.jsonl
& $py tools/bench/jevbench/reasoning_router.py collect `
  --collector build-win/apps/Release/ninfer-reasoning-collect.exe `
  --artifact C:/models/Qwen3.8-27B/qwen3_8_27b_nvfp4_dflash2.ninfer `
  --requests requests.jsonl --out outcomes.jsonl
```

`collect` hashes the artifact, verifies it did not change, and resumes matching completed IDs.
An interrupted partial final JSONL line must be removed before resuming; complete rows remain
usable. Use a new outcome file for a different artifact or numerical profile. `prepare --tasks`
accepts existing task JSONL with independent gold; related variants must share `group` or
`source_group`. Never use the teacher's unsupported self-assessment as the correctness label.
The included generated cohort is a small diagnostic, not a general training corpus.

Training runs locally with PyTorch or on Colab with the same script. A Colab CLI example:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
colab new -s ninfer-router --gpu T4
colab upload -s ninfer-router tools/bench/jevbench/reasoning_router.py /content/reasoning_router.py
colab upload -s ninfer-router outcomes.jsonl /content/outcomes.jsonl
@'
import runpy, sys, shutil
sys.argv = ['reasoning_router.py', 'train', '--data', '/content/outcomes.jsonl', '--out', '/content/router']
runpy.run_path('/content/reasoning_router.py', run_name='__main__')
shutil.make_archive('/content/router', 'zip', '/content/router')
'@ | colab exec -s ninfer-router --timeout 180
colab download -s ninfer-router /content/router.zip router.zip
colab stop -s ninfer-router
```

Download before stopping; stop the session even when training fails. Training outputs are
`router.json` (portable weights, profile, replay identities), `report.json` and `predictions.jsonl`.
Default selection utility is correctness minus `0.02 * output_tokens / 1024`; set it before
evaluating. The default head has 30,726 parameters; `--hidden 128` trains the larger MLP.

For the next round, collect new groups into the accumulated outcomes, upload the updated file and
the previous `router.json`, then add `--previous /content/previous-router.json` to the train command.
Every prior observation and holdout must remain unchanged. The head warm-starts and trains on
the full replay set. Instance test and held-out-family test results remain separate. Neither a
successful training run nor a small diagnostic accuracy gain enables adaptive serving; these
exports explicitly remain unqualified until representative evaluation supports integration.

### Flash-Next synthetic corpus on Colab G4

`synthetic_corpus.py` sends ordinary OpenAI Chat Completions requests to NInfer. Flash-Next
authors bounded JSON scenario data; Python computes exact answers and renders the decision
questions. It never runs generated code or accepts a teacher answer as gold. The four families
are ordered integer operations, first-match policies, directed shortest paths, and 0/1 allocation.
Each has four **size** strata; difficulty is established later by the target model's outcomes.

Use a full 96 GB G4 runtime, enough host RAM for Flash-Next's mapped PLE table, and at least
114 GB free disk for its artifact plus build/cache headroom. Upload the current source and
build the C++ engine in the Colab environment (CUDA 13.3 is needed for the current Flash-Next
top-k dependency):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNINFER_BUILD_MEDIA=OFF -DBUILD_TESTING=OFF
cmake --build build --target ninfer-serve -j
./build/apps/ninfer-serve /content/models/flash-next/qwen3_8_flash_next_mixed.ninfer \
  --host 127.0.0.1 --port 8010 --model-id qwen3.8-flash-next \
  --max-context 8192 --kv-capacity 32768 --max-concurrency 4 --prefill-chunk 2048 \
  --desktop-reserve-gib 1 --kv-dtype fp8 --gdn-state-dtype bf16 --no-prefix-reuse
```

Download the [published artifact](https://huggingface.co/igorls/Qwen3.8-Flash-Next-mixed-NInfer)
at revision `5f0ee7e24279cbadf8d4a90c93c2ff5ea6b00688` and verify its full-file SHA-256 before
starting the server. The CLI digest below is provenance supplied by the operator; the HTTP
model name is checked but cannot attest to the loaded weights.

```bash
python3 tools/bench/jevbench/synthetic_corpus.py \
  --teacher-artifact-sha256 3d383e51963aafd4318dfd04c8dc63ee7df11768de19d9ab58dbba44460d1d02 \
  --out /content/corpus --blueprints 64 --variants 32 --concurrency 4
```

Outputs are raw requests/responses and accepted specifications (`blueprints.jsonl`), deduplicated
tasks with executable gold (`tasks.jsonl`), a resume manifest and counts/rejection report.
All 32 siblings of a blueprint share one split group; 2,048 tasks therefore represent at most
64 teacher blueprints. The four canonical question templates are a bounded diagnostic corpus,
not coverage of all JevBench decisions. No benchmark inputs are sent to the teacher.

Resume with the same model, verified artifact, generator, seed and variant count. The requested
blueprint count may only increase; failed slots retry. Preserve the manifest with the raw rows.
An interrupted partial final JSONL line must be removed before resuming. Download results before
stopping the named Colab session, and stop it after use to release the GPU.

Prepare these tasks with `reasoning_router.py prepare --tasks /path/to/tasks.jsonl --out requests.jsonl`,
then collect direct/1,024/2,048-token outcomes using **Qwen3.8-27B NVFP4**. Flash is the scenario
author, while Qwen's frozen features and measured outcomes train its router. Select an entire
synthetic family for a new study's holdout; progressive training must retain the original
holdouts and replay data. Generating this corpus alone does not establish routing quality.

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
