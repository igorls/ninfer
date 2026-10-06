# HTTP serving

`build/apps/ninfer-serve` loads one v3 `.ninfer` artifact and exposes OpenAI- and
Anthropic-compatible HTTP endpoints over one resident NInfer Engine.

## Start the server

See [CUDA synchronization](cli.md#cuda-synchronization) for the shared `NINFER_CUDA_SYNC` setting.

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 \
  --port 8080 \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

The command uses Qwen3.8-27B NVFP4. Each request has a 240,000-token logical ceiling. A shared
240,000-token Main Text KV pool serves admitted requests; either request may use the full capacity
when running alone, and two requests run concurrently when their complete reservations fit.

With `C=2` and two extra Device checkpoint slots, the process owns two active StateImage guarantees
plus a global pool of two Device-resident checkpoints. Eight pinned Host State slots and 8 GiB of
pinned Host KV retain inactive continuations under Device pressure. Active request capacity is two.

Other artifacts use the same command shape with their own path. For 35B-A3B DFlash, replace the MTP
selection with `--spec dflash --draft-tokens 7 --lm-head-draft`. Qwen3.8-27B
artifacts with DFlash2 companion weights also support `--spec dflash2 --draft-tokens 7`, with
`--lm-head-draft` optional. DFlash2 accepts draft counts 1..15 and supports the same sampling,
concurrency, prefix reuse, and image/video request surfaces. It may remain combined with
`--vision`.

When `--model-id` is omitted, the server advertises and accepts the artifact's `metadata.name`,
falling back to its architecture name when no name is stored. An explicit `--model-id` is a public
HTTP alias override and does not select or alter model execution.

Vision is disabled by default: its weights and Vision-specific unified-workspace extent are not
allocated, and media requests and token-count requests fail with HTTP 400 `vision_disabled`. Add
`--vision` when the server must accept image or video input. Speculative residency is likewise
frozen by `--spec mtp|dflash|dflash2` and `--draft-tokens`; omitting `--spec` loads no speculative backend.
`--lm-head-draft` additionally loads the optimized proposal head. DFlash on 35B-A3B and DFlash2 on Qwen3.8-27B can be combined
with `--vision`; each accelerates generated-text decode after multimodal prefill, while Vision encode
and prefill remain outside speculative acceleration. A later request cannot enable a capability
omitted at startup. The artifact need only contain the Text backbone and the optional components
selected for this process.

## Endpoints

| Method and path | Behavior |
|---|---|
| `GET /health` | Engine readiness |
| `GET /v1/models` | configured OpenAI model alias, advertised rerank id, and effective `max_model_len` |
| `GET /v1/models/{id}` | lookup of the configured alias or advertised rerank id, and effective `max_model_len` |
| `GET /props` | llama.cpp-compatible server properties for llama.cpp's web UI |
| `POST /v1/chat/completions` | OpenAI-style chat generation |
| `POST /v1/score` | closed-set scoring of isolated questions against one shared prefix |
| `POST /v1/rerank` | Jina-shaped document reranking scored by an in-process System One Choice |
| `POST /v1/responses` | OpenAI Responses Core generation, state, typed Items, and SSE |
| `POST /v1/responses/input_tokens` | Responses prompt-token count without generation |
| `GET /v1/responses/{id}` | retrieve a locally stored terminal Response |
| `DELETE /v1/responses/{id}` | delete a locally stored Response |
| `GET /v1/responses/{id}/input_items` | list that Response's normalized input Items |
| `POST /v1/messages` | Anthropic-style message generation |
| `POST /v1/messages/count_tokens` | checkpoint-native expanded input-token count |
| `POST /v1/systemone` | TypeSafe System One structured decision, classification, and scoring |
| `POST /systemone` | alias for `POST /v1/systemone` |
| `GET /admin/vram` | Engine memory plan beside cached device-wide memory diagnostics |
| `GET /admin/stats` | cumulative runtime and Host-work statistics |
| `POST /admin/quiesce` | hold the Engine still at an execution boundary (diagnostic) |

`GET /health` returns HTTP 200 with `{"status":"ok"}` while the Engine can accept work. After an
Engine-wide failure it returns HTTP 503 with `{"status":"unavailable"}`. Temporary queue
saturation does not make the Engine unavailable. The endpoint remains unauthenticated.

Every OpenAI-compatible response carries a unique `x-request-id` header, including streaming and
error responses. Anthropic endpoints use their separate `request-id` contract.

`/admin/vram` reports the Engine's planned reservation, arenas and KV beside device-wide free
memory from NVML (`device.source` is `cudaMemGetInfo` when NVML is unavailable), the desktop
reserve, and the compute processes on the device. Driver queries refresh at most once every two
seconds; while one is in progress, concurrent requests receive the previous snapshot with its
`device.age_ms`, or HTTP 503 before the first snapshot. The Engine memory plan is read without
waiting for an execution unit, so `/admin/vram` stays responsive under load. `/admin/quiesce`
refuses admission until active requests drain, holds the Engine at an idle boundary for the
requested `hold_ms` (default 0, at most 10000), then resumes; requests that arrive meanwhile wait
and their admission deadlines are extended by the hold. It changes no capacity.

All three generation SSE endpoints emit the standard `: keep-alive` comment after five seconds
without a protocol event. The comment is transport-only: SSE clients ignore it, and it does not
change generated text, event ordering, usage, stored Responses, or request logs. On Linux, accepted
connections also use TCP keepalive and a 15-second `TCP_USER_TIMEOUT`; together with the heartbeat,
a dead or unacknowledging peer is normally cancelled within about 20 seconds, including while the
request is waiting or prefilling. A peer whose TCP stack remains connected and acknowledges data
cannot be distinguished from a reading application; proxies must close their upstream NInfer
connection when the downstream client disappears.

## OpenAI Chat Completions

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "system", "content": "Answer concisely."},
      {"role": "user", "content": "What is speculative decoding?"}
    ],
    "max_tokens": 128
  }'
```

The endpoint supports:

- `model` naming the configured public model ID; an omitted or null `model` selects it, as
  llama.cpp does, and any other name is `model_not_found`;
- `system`, `developer`, `user`, `assistant`, and `tool` history, plus legacy `function` history;
- string content and ordered text/refusal parts; adjacent parts are preserved without inserted
  separators, and empty wire content remains an empty turn;
- User `image_url` parts, tool-result `image_url` parts used by compatible clients, and the User
  `video_url` extension using HTTP(S) or data URIs; image detail is omitted or `auto`;
- nonnegative `max_completion_tokens` and the legacy `max_tokens` spelling; zero performs prompt
  processing without generation, and llama.cpp's `-1` removes the output limit so the request
  stops at its context capacity;
- `temperature`, `top_p`, presence/frequency penalties, and signed integer `seed`;
- the compatible `top_k` (`0..20`) and `min_p` (`0..1`) sampler extensions;
- up to four non-empty stop strings, applied to both reasoning and answer output;
- `n:1`, text-only `modalities`, and `response_format` `text`, `json_object` or `json_schema`
  (see [Structured output](#structured-output));
- non-streaming responses and server-sent event streams;
- `stream_options.include_usage`;
- llama.cpp-compatible terminal `timings`, plus opt-in `timings_per_token` and
  streaming `return_progress` observations;
- non-strict function tools with `tool_choice` `auto`, `none`, `required`, a named function, or
  `allowed_tools` in `auto` or `required` mode, parallel calls enabled, assistant tool-call
  history, tool-result messages, and legacy function-call history;
- the top-level `reasoning_effort` field, and llama.cpp's `thinking_budget_tokens` (a positive
  reasoning budget, or `-1` for none) for requests that think;
- `enable_thinking` and `preserve_thinking`, either at top level or in
  `chat_template_kwargs`;
- Assistant `reasoning_content` and `reasoning` history aliases.

Options whose observable behavior the Engine cannot provide are rejected when they request that
behavior. This includes nonzero `logit_bias`, requested log probabilities,
audio/file input or audio output, `strict:true`,
`parallel_tool_calls:false` with enabled tools, explicit low/high image detail, web search,
moderation, low/high verbosity, stored Chat Completions, and non-empty legacy `functions`.
Each capability rejection identifies the affected field and the guarantee NInfer cannot provide.
Known constrained-decoding aliases (`grammar`, `structured_outputs`, `guided_json`, `guided_regex`,
`guided_choice`, and `guided_grammar`) receive the same explicit rejection instead of being treated
as unknown hints.

Semantically neutral fields do not make an otherwise executable request fail. All-zero
`logit_bias`, `logprobs:false`, `top_logprobs:0`, `verbosity:"medium"`, empty legacy tool controls,
text-only `audio` configuration, and `prediction` are accepted without changing Engine execution.
Metadata, user/safety identifiers, service-tier and prompt-cache hints are likewise advisory.
Unknown top-level fields are ignored.

### Token log probabilities

`logprobs: true` on a non-streaming Chat Completions request returns
`choices[0].logprobs.content`, one entry per generated token (reasoning tokens and the final stop
token included), in the OpenAI shape: `token`, `logprob`, `bytes`, and `top_logprobs` with up to
`top_logprobs` (0 to 20) alternatives. Every entry also carries `token_id` and `raw_logprob`.

All values are read from the target model's logits **before any sampling adjustment**: no
repetition/presence/frequency penalty, no temperature, no `top_k`/`top_p`/`min_p` truncation. They
are the model's own distribution at that position however the token was drawn, so a client can
apply its own temperature scaling for calibration.

- `raw_logprob` is the log softmax over the whole vocabulary.
- `logprob` is the log softmax over the tokens the structured-output grammar allows at that
  position, which is the distribution the sampler could actually draw from. Without
  `response_format` constraints the two are equal. Under a mask `top_logprobs` lists allowed tokens
  only and their probabilities sum to one over the allowed set.
- `logprob_candidates` (NInfer extension): an array of up to 1024 closed-set options, each a string
  that encodes to exactly one token or an integer token id. Every position then also reports
  `candidate_logprobs`, the distribution renormalised over exactly that list, in request order and
  not limited to 20. A candidate the grammar forbids reports the `-9999` floor. Check that each
  option is one token in the context where it appears: a leading space changes the token.
- A thinking-budget control token inserted by the engine is reported with `"forced": true`,
  `logprob` 0 and no alternatives.
- `-inf` is not valid JSON; vanishing or forbidden probabilities report `-9999`.
- `logprob_prompt_positions` (NInfer extension): ascending 0-based indices into the rendered
  prompt's tokens, at most 256, needing `top_logprobs` 0. The response adds
  `choices[0].logprobs.prompt`, one entry per position: the distribution over the token after that
  position, whose `token` is the prompt's own next token with its `logprob` (so summed entries
  score a continuation), plus `candidate_logprobs` when candidates are given. A position beyond the
  prompt is rejected. Every listed position is computed by the request: prefix reuse is limited to
  frontiers at or below the first position. A history assistant turn renders as the selected
  template writes it; where that omits the empty think block the generation prompt carries, the
  model expects `<think>` at the position before its content, so write the block into the content
  (`"<think>\n\n</think>\n\n" + answer`) to read an answer distribution there.

A request with `logprobs: true` and `top_logprobs` above 0 reads a full vocabulary column on the
host and decodes one token per round: speculative drafts are not offered for it. Without
`top_logprobs` the readout runs on the device, costs a few floats per position, and speculative
decoding stays on. `logprobs` with `stream: true` is rejected with `logprobs_stream_not_supported`;
the Responses API does not report log probabilities.

With thinking off, the maintained Qwen templates render a continued final assistant message
(assistant prefill) with the same empty think block the generation prompt carries before its
content, so the continuation is conditioned exactly like an answer the model produces.

### Read-only cache participation

`prompt_cache_read_only: true` (NInfer extension, Chat Completions) lets a request start from an
already published prefix while capturing no checkpoint and publishing no continuation of its own.
Use it for one-shot requests, such as single-token classification over a shared document, whose
continuation will never be reused: a burst of them otherwise turns the bounded continuation
catalog over and evicts other conversations' cached state. A read-only request never creates the
shared prefix it reads; some earlier request has to publish it. Because nothing can later resume
from it, a read-only request also skips the prefill split that publishing requests make at the
prompt's rewrite execution frontiers, so a short one-shot prompt prefills in one pass over the
weights instead of two.

### Closed-set scoring: `POST /v1/score`

Scores many isolated questions against one shared prefix in a single call, for closed-set
classification with calibrated probabilities.

```json
{
  "model": "qwen3.8-27b",
  "messages": [{"role": "system", "content": "<instructions and document>"}],
  "questions": [
    {"id": "q1", "content": "Is the invoice overdue? Answer A for yes, B for no.", "candidates": ["A", "B"]},
    {"id": "q2", "content": "...", "candidates": ["A", "B", "C", "D"]}
  ],
  "candidates": ["A", "B"],
  "top_logprobs": 0,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

`messages` is the shared prefix in Chat Completions format. Each of the 1 to 256 `questions`
becomes one user message appended after it, so questions never see each other. `candidates` are
strings or integer token ids; a question without its own list uses the top-level default. Each
question takes one of two forms, reported as `form`:

- `token`: every candidate is one token. One greedy request reads the distribution over the
  candidates at the first generated position (`candidate_logprobs`, `top_logprobs`,
  `outside_mass`). A question may carry its own `response_format`; the scored position is always
  the first generated token, so a format whose first token is punctuation (a JSON string quote) is
  not useful here.
- `text`: some candidate spans several tokens. One request per candidate renders it as the
  continued final assistant turn, opened exactly as generation opens an answer, and reads the
  log-probability of each of its tokens at its prompt position. `candidate_logprobs[].raw_logprob`
  is their sum, the candidate's conditional log-probability; `logprob` renormalises the sums over
  the list; `tokens` lists the per-token entries with their positions, so a client can apply its
  own length normalisation. Token ids cannot be mixed into such a list, `response_format` does not
  apply, and the form needs thinking off. Cost is one prefill of the question and candidate per
  candidate, all reading the shared prefix.

Every result also carries `entropy` (nats) and `margin` (top-two probability ratio) of its
distribution.

The response lists results in question order:

```json
{
  "object": "score", "model": "qwen3.8-27b",
  "results": [
    {"id": "q1", "index": 0, "token": "A", "token_id": 32,
     "candidate_logprobs": [{"token": "A", "token_id": 32, "logprob": -0.02, "raw_logprob": -0.03, "bytes": [65]}],
     "top_logprobs": [], "outside_mass": 0.004, "cached_tokens": 0}
  ],
  "usage": {"prompt_tokens": 12400, "cached_tokens": 6100, "completion_tokens": 2}
}
```

`candidate_logprobs[].logprob` is renormalised over the candidate list, `raw_logprob` is
vocabulary-wide, and `outside_mass` is the vocabulary-wide probability the model put outside the
list. A question that fails reports `{"id", "index", "error"}` in its slot and does not fail the
call.

The endpoint is orchestration over the ordinary Engine route, not a fused batch: the first question
carries an explicit shared-prefix boundary at the end of `messages` (and no implicit write
candidate), which prefills and publishes the prefix once; the remaining questions then run one at
a time against that prefix, read-only in the context cache. The published prefix stays in the
cache under normal retention, so a later call over the same `messages` starts warm.

A string `name` on a `tool` message is accepted as an ignored, output-neutral compatibility
extension for clients that mirror the function name onto tool results. It does not participate in
tool identity, prompt rendering, or output. Non-string values are malformed; non-empty names on
other message roles remain unsupported because they carry participant identity that the loaded chat
template cannot represent.

For commonly generated OpenAI-compatible payloads, `repetition_penalty` is accepted only at its
neutral value `1` on Chat Completions (Responses and the `--repetition-penalty` server flag apply
it), and `mm_processor_kwargs` when empty or containing only null values. String-form
image/video URLs are also accepted.

Malformed protocol values return field-specific HTTP 400 errors. Invalid media sources, bytes, or
decoded content use `invalid_media`; remote fetch and timeout failures retain their dedicated
server-error codes. Failures in the normalized prompt contract use `invalid_prompt`; typed capacity
and availability failures retain their dedicated codes. Internal invariant failures are not
relabeled as client input errors. When the chat template itself rejects a request, the message is
the template's own `raise_exception` text (or the interpreter's cause) after the template's source
name, without a template trace.

The request `model` must equal the public model ID: the artifact `identity.model_id` by default, or
the explicit `--model-id` override. Reasoning is returned separately as `reasoning_content`; answer
text remains in `content`.

Across Chat Completions, Responses, and Anthropic Messages, a direct top-level tool-parameter
`type`, or an `anyOf`/`oneOf` composed entirely of explicit primitive types, guides conversion of
Qwen's untyped parameter text. It does not decide whether structurally complete markup is a tool
call. String-admitting values remain strings, including the empty string. An empty block for a
declared non-string parameter is omitted. Admitted JSON values retain their JSON type;
case-insensitive boolean text is normalized to `true` or `false`. A nonempty schema mismatch remains
a structured call: valid JSON retains its represented type and other text becomes a JSON string so
the tool consumer can report the validation error and continue the agent loop. Schemas without a
supported explicit type retain untyped inference. NInfer does not apply defaults, enforce required
properties, perform recursive JSON Schema validation, or use constrained decoding.

String parameters preserve function/tool-call markers and balanced nested
`<parameter=...>...</parameter>` text as value bytes. The Qwen wire format has no delimiter escape,
so an unmatched nested parameter opener or a standalone `</parameter>` cannot be represented
unambiguously; either causes the complete tool-call region to fall back to ordinary content.

Messages enter the selected template in their input order. The maintained Qwen templates keep
system/developer messages at their original positions.

Prompt-bearing JSON objects retain their received member order through request parsing and prompt
rendering, including tool schemas and historical tool inputs. Canonical model-origin tool arguments
retain that member order in aggregate and streaming responses, so an unmodified replay reconstructs
the same ordered tool call. NInfer does not canonicalize semantically equivalent JSON: if a client
reorders members, inserts defaults, or otherwise rewrites a tool object, the changed rendered input
does not match the model-held endpoint and can reuse only an earlier exact checkpoint.

`--chat-template FILE` selects a local Jinja template; by default, the server uses the template
stored in the artifact. See the [CLI guide](cli.md#text-input) for an example.

Control-token spellings quoted in message content, tool data or ordinary template kwargs are
encoded as text. Media placeholders come from the template and bind to actual image/video inputs.

`chat_template_kwargs` passes a JSON object to the template in Chat Completions, Responses and
Anthropic Messages. Values duplicated in typed request fields must agree. Null standard options
mean unspecified; other null values remain `none`. Messages, tools, generation mode and tokenizer
special tokens cannot be overridden through kwargs.

`--default-thinking-budget N` sets a positive default thinking-token cap for requests that start
in thinking mode. Non-thinking requests receive no cap. It may coexist with `--no-thinking`
because requests can explicitly enable thinking. Anthropic
`thinking:{"type":"enabled","budget_tokens":N}` overrides this default for that request.

Add `--default-thinking-budget 512` to the startup command to cap model-origin thinking at 512
tokens for every thinking-enabled request.

At the cap boundary, Engine first honors a natural `</think>`, stop condition, cancellation, or
total output/context limit. If thinking remains open, it commits Qwen's canonical early-close
guidance and close marker to the same model sequence without sampling, streams the guidance as a
reasoning delta, and continues normal content or tool-call generation. Inserted tokens count in
completion usage and the request's `max_tokens`/`max_output_tokens` budget. If the effective output
capacity extends past the cap but cannot fit the complete tokenizer-derived control suffix plus one
post-close model token, preparation is rejected with HTTP 400 code
`thinking_budget_capacity_insufficient` rather than partially inserting control. The server does
not promise that the model will emit nonempty content or a tool call after the marker.

For Chat Completions, top-level `reasoning_effort` and `chat_template_kwargs.reasoning_effort` name
the same option and must agree. `reasoning_effort: "none"` requests disabled thinking with any
template. The selected template decides which other standard values (`minimal`, `low`, `medium`,
`high`, `xhigh`, `max`) exist: at startup, NInfer renders one user turn with each effort and records
the efforts the template accepts and the one it renders by default. The maintained Qwen3.8 template
accepts `low`, `medium` and `xhigh` (default). Any other effort returns HTTP 400 with code
`reasoning_effort_not_supported` on `reasoning_effort` before the prompt is rendered, and the
message lists the accepted efforts, for example
`Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.` A template
that renders every effort like an unspecified one (the Qwen3.6 template) does not interpret
reasoning effort; any effort other than `none` then returns
`the loaded chat template does not support reasoning effort` with the same code. Conflicting explicit
`enable_thinking` and effort values return `conflicting_template_option`.

`preserve_thinking` controls reasoning retention according to the selected template. Request
options override server defaults set with `--no-thinking` and `--preserve-thinking`. Unless a
request or `--preserve-thinking` asks to keep it, closed-turn assistant reasoning sent back in the
history is dropped, even where the template would keep it by default (Qwen3.8). Unspecified
thinking and effort options use the template's defaults.

Streaming begins with an assistant-role chunk, sends separate reasoning and content deltas, then a
finish-reason chunk and `[DONE]`. When `stream_options.include_usage` is true, a final empty
`choices` chunk contains completed usage. Aggregate and streamed usage include cached prompt tokens
and reasoning-token details; choices carry `logprobs: null` when log probabilities were not
requested, and aggregate assistant messages carry `refusal: null` because refusal output is not
supported.

### llama.cpp-compatible request observations

Every successful Chat Completions response includes a top-level `timings` object. This is a
llama.cpp-compatible response extension, not an OpenAI field. In a stream it is attached to the
last JSON chunk before `[DONE]`: the empty `choices` usage chunk when
`stream_options.include_usage` is true, otherwise the finish-reason chunk.

```json
{
  "timings": {
    "cache_n": 4096,
    "prompt_n": 4096,
    "prompt_ms": 83.0,
    "prompt_per_token_ms": 0.020263671875,
    "prompt_per_second": 49349.39759036145,
    "predicted_n": 129,
    "predicted_ms": 1140.0,
    "predicted_per_token_ms": 8.90625,
    "predicted_per_second": 112.28070175438596
  }
}
```

`cache_n` is the exact Engine-proven reused prompt prefix and `prompt_n` is the remaining prompt
suffix, so `cache_n + prompt_n` equals `usage.prompt_tokens`. Prompt time starts when admission
commits that exact reuse choice and ends when the first output token is committed. Generation time
starts at that first token and ends at the last committed output token. Accordingly, generation
speed uses `max(predicted_n - 1, 0)` token intervals; the first token belongs to prompt latency and
is not counted again as a decode interval. Zero-token, one-token, zero-duration, and exact-cache-hit
cases report finite zero rates rather than `NaN` or infinity. Speculative requests additionally
include terminal `draft_n` and `draft_n_accepted` when draft work occurred.

Set top-level `timings_per_token: true` on a streaming request to attach the latest cumulative
timing snapshot to each visible reasoning or content chunk. This does not enable terminal timings,
which are always present. A model commit that is temporarily hidden by UTF-8, stop-string,
reasoning, or tool-call buffering still advances the cumulative token count; the next visible chunk
observes that committed frontier. The option increases response serialization and transport volume
and is off by default.

Set top-level `return_progress: true` together with `stream: true` to receive prompt-processing
chunks:

```json
{
  "prompt_progress": {
    "total": 8192,
    "cache": 4096,
    "processed": 6144,
    "time_ms": 41
  }
}
```

The initial event has `processed == cache`. Later cumulative events are published only after the
corresponding prefill unit commits, may be coalesced when the consumer is slower than prefill, and
never move backwards. The final event has `processed == total` and precedes the first output delta.
For an exact full-prefix hit, the initial event already has `cache == processed == total` and no
synthetic prompt work is reported. `time_ms` is elapsed wall time since committed admission;
clients may calculate actual suffix progress as `(processed-cache)/(total-cache)` when the
denominator is nonzero.

### llama.cpp web UI

llama.cpp's web UI (`llama-ui`, shipped with llama.cpp 0.5.0) runs against NInfer as a single-model
llama.cpp server. At startup it reads `GET /props`; NInfer answers with the values its loaded Engine
determines:

```json
{
  "default_generation_settings": {"n_ctx": 131072},
  "total_slots": 8,
  "model_alias": "qwen3.8-27b",
  "model_path": "qwen3_8_27b_nvfp4.ninfer",
  "role": "model",
  "modalities": {"vision": true, "audio": false, "video": false},
  "chat_template": "{%- set image_count = namespace(value=0) %}..."
}
```

`n_ctx` is `--max-context`, `total_slots` the effective `--max-concurrency`, `model_alias` the
public model ID, `model_path` the artifact file name, and `vision` follows `--vision`. The chat
template is the Jinja source prompts are rendered with (the artifact's own, or `--chat-template`);
the UI inspects it to offer its thinking toggle. llama.cpp's sampling `params`, special tokens and
`build_info` are omitted, so the UI's settings show no server defaults: NInfer's sampling defaults
depend on whether the request thinks. A `?model=` query returns the same properties.

The UI calls relative paths, so it must be served from the same origin as the API. On Windows the
Supervisor serves it as a [web frontend](windows-app.md#web-frontends). Elsewhere, use a reverse
proxy that serves its files and forwards `/v1/*`, `/props`, `/health`, `/slots`, `/tools`, and
`/models*` to NInfer without buffering streams. `--cors` does not affect it. Its chat requests
omit `model`, send `timings_per_token`, and put its thinking toggle in
`chat_template_kwargs.enable_thinking`; answers stream `content` and `reasoning_content` deltas
with the timings above. Its reasoning-effort menu sends `thinking_budget_tokens`. Every request
also carries the UI's browser tools (`get_datetime`, `get_info`); tool calls stream as ordinary
`tool_calls` deltas, the UI asks the user before running one in the browser, and it returns the
result as a `tool` message.

llama.cpp-specific features without an NInfer route stay unavailable, and the UI degrades as it does
against a llama.cpp server started without them: `/slots` (the UI then assumes idle slots), server
`/tools` and MCP proxying (the UI logs the 404 and keeps its browser tools), `/v1/chat/completions/control` behind the "Skip reasoning" button, the
resumable-stream routes (`/v1/stream`, `/v1/streams/lookup`) that reattach a stream after a dropped
connection or reload, and router-mode `/models` loading. `reasoning_format` is ignored: reasoning
is always returned separately. Audio and llama.cpp's `input_video` content part are not accepted.

### Multimodal request

Start the server with `--vision` before sending media:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "text", "text": "Describe this image."}
      ]
    }],
    "max_tokens": 128
  }'
```

OpenAI image and video sources may be HTTP(S) URLs or base64 data URLs.

Text and media requests use one complete-prompt context contract. After chat-template rendering and
media-token expansion, the result must fit Engine `--max-context`. The current Vision runtime also
has a 32,768 merged-token envelope (131,072 raw patches); the effective Vision limit is therefore
`min(--max-context, 32768)`. There is no fixed image/video item-count limit: item count is admitted
through aggregate source-byte, decoded-pixel, raw-patch, Vision-token, and live-memory budgets.

Media cache misses run as independent decode → resize → BF16-pack tasks on a bounded host worker
pool. Prepared payloads are keyed by SHA-256 of the acquired bytes plus modality, so repeated media
in later requests reuses the exact immutable BF16 patch input; concurrent identical misses use one
single-flight build. `--media-cache-mib` bounds LRU-retained payloads, while
`--media-live-mib` bounds every cache-, request-, or runtime-referenced payload. Cache eviction does
not invalidate a request reference, and live bytes are returned only when the final reference is
released. A request-level preparation gate derived from the live limit prevents concurrent partial
builds from deadlocking the memory account.

An expanded prompt beyond `--max-context` returns HTTP 400 `context_length_exceeded`, including
the prepared token count and configured context ceiling. A media preprocessing resource rejection
returns HTTP 400 `media_budget_exceeded`. HTTP 413 `request_too_large` is reserved for a raw request
body that exceeds `--max-request-mib` before JSON parsing; it is not used for model-context or media
resource errors.

## OpenAI prompt caching

Chat Completions and Responses translate OpenAI cache hints into optional shared-prefix write
candidates:

- omitted `prompt_cache_options` creates a default implicit candidate at the latest representable
  content boundary;
- `mode:"implicit"` requests the same automatic candidate explicitly;
- `mode:"explicit"` disables that implicit write for the request;
- `prompt_cache_breakpoint:{"mode":"explicit"}` on supported content creates an explicit
  candidate.

One request carries at most four distinct writes. An implicit target occupies one slot unless it
coincides with an explicit target; the remaining slots contain the latest explicit boundaries.
Earlier schema-valid historical breakpoints are accepted but are not new write candidates. Exact
reads of already-published prefixes do not require the request to repeat a marker.

These fields are optimization hints. A legal boundary that cannot be represented as an exact
rendered-token frontier is ignored without changing prompt content. `prompt_cache_key` is not an
Engine session key or prefix identity. Valid TTL/retention values are accepted, but NInfer does not
promise their wall-clock residency; physical retention follows the resource scheduler.

## OpenAI Responses Core

NInfer implements the typed-Item and semantic-event core of the OpenAI
[Responses API](https://developers.openai.com/api/reference/resources/responses/overview). All
supported model instances use this same adapter and Engine route. It is intentionally not
advertised as full parity with OpenAI-hosted tools, durable cloud storage, background jobs,
Conversations, or compaction.

### Create a Response

```bash
curl http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "instructions": "Answer concisely.",
    "input": "What is speculative decoding?",
    "max_output_tokens": 128,
    "store": true
  }'
```

The same endpoint works with OpenAI SDKs by replacing their base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local-secret")
response = client.responses.create(
    model="qwen3.8-27b",
    instructions="Answer concisely.",
    input="What is speculative decoding?",
    max_output_tokens=128,
)
print(response.output_text)  # SDK helper derived from response.output
```

`output_text` is an SDK convenience property. It is not emitted as a top-level wire field; the
wire response contains typed `output` Items.

### Create request fields

| Field | NInfer Responses Core contract |
|---|---|
| `model` | required non-empty string; must equal the artifact-derived public model ID or explicit `--model-id` override |
| `input` | string or typed Item array; it may be omitted or empty only when `previous_response_id` already supplies a user query |
| `instructions` | optional string, inserted before the reconstructed conversation for this request only |
| `previous_response_id` | optional ID of a retained local Response |
| `max_output_tokens` | non-negative integer; omission executes with `--default-max-tokens` but remains `null` in the Response object |
| `stream` | boolean; `true` selects Responses SSE rather than a JSON body |
| `store` | boolean, default `true`; controls local retrieval and continuation state |
| `temperature` | finite number in `[0,2]` |
| `top_p` | finite number in `[0,1]` |
| `metadata` | at most 16 string pairs; keys at most 64 characters and values at most 512 |
| `client_metadata` | Codex client extension; an object or `null`, accepted as opaque tracing metadata with no generation effect |
| `reasoning.effort` | `none` requests disabled thinking; another standard effort must be one the selected template accepts, otherwise `reasoning_effort_not_supported` on `reasoning.effort` (see [Chat Completions](#openai-chat-completions)) |
| `chat_template_kwargs` | template parameters as a JSON object; standard options merge with typed fields |
| `preserve_thinking` | alias for `chat_template_kwargs.preserve_thinking`; conflicting values are rejected |
| `text.format` | `text`, `json_object`, or flat `json_schema` (`name`, `schema`, optional `strict`, `description`); echoed on the Response object |
| `tools` | direct function definitions or namespace groups containing function definitions; see below |
| `tool_choice` | `auto`, `none`, `required`, a named function, or function-only `allowed_tools` with mode `auto` or `required`; a namespaced selection carries both `namespace` and `name` |
| `parallel_tool_calls` | `true` by default; `false` is accepted only when no effective tool is callable |
| `max_tool_calls` | non-negative integer accepted as a hosted-tool no-op; NInfer does not execute hosted tools |
| `truncation` | omitted or `disabled`; overlong input fails instead of silently dropping Items |
| `top_logprobs` | omitted or `0` |
| `service_tier` | omitted, `auto`, or `default`; the response reports `default` |
| `background` | omitted or `false` |
| `include` | omitted or an empty array |
| `stream_options.include_obfuscation` | optional boolean; accepted as a transport hint, but this local server emits no padding |
| cache and client hints | `prompt_cache_key`, `prompt_cache_options`, `prompt_cache_retention`, and explicit breakpoints follow [OpenAI prompt caching](#openai-prompt-caching); `safety_identifier` and `user` are accepted as client hints |

Unknown top-level fields fail with `unknown_parameter`. Recognized but unsupported features fail
with a field-specific 400 error instead of being silently ignored.

### Input Item contract

String `input` is normalized to one user `message` with an `input_text` part. Array input accepts:

| Item | Supported form |
|---|---|
| `message` | roles `user`, `assistant`, `system`, and `developer`; string content or typed content array |
| `input_text` | message content part containing string `text` |
| `output_text` | assistant-message replay part containing string `text` |
| `refusal` | assistant-message replay part; its text enters assistant history |
| `input_image` | user- or assistant-message part with HTTP(S) or data-URI `image_url`; detail omitted or `auto`; requires server `--vision` |
| `input_video` | NInfer extension with HTTP(S) or data-URI `video_url`; requires server `--vision` |
| `reasoning` | raw replay Item with `reasoning_text` content; summary/encrypted metadata may accompany raw text but cannot replace it |
| `function_call` | completed assistant call with optional `id` and namespace, plus required `call_id`, `name`, and JSON-object string `arguments` |
| `function_call_output` | completed result with required `call_id` and optional matching name/namespace assertion; `output` may be a string or a non-empty array of `input_text`/`input_image` parts |

Contiguous assistant-owned Items form one assistant history turn in the representable order
`reasoning` -> assistant message content -> `function_call`. Multiple message Items append their
content parts, multiple calls retain declaration order, and a reasoning-only turn is retained. A
user, system, developer, or `function_call_output` Item ends the group; an order that would require
rearranging assistant content fails with `invalid_assistant_history`. Results are validated by
`call_id` and reordered to call declaration order before prompt rendering; unknown, duplicate, or
unrepresentable partial result sets fail with `invalid_tool_history`. Canonical input Items retain
client order. Input Item IDs are preserved when supplied and generated otherwise; duplicate IDs
fail.

System and developer message Items retain their positions in the input array. Top-level
`instructions` is represented as a leading developer turn for the current request; target-specific
role lowering occurs only in the Qwen family frontend.

An `input_text`, `input_image`, or tool-result part may carry
`prompt_cache_breakpoint:{"mode":"explicit"}`. Write selection follows
[OpenAI prompt caching](#openai-prompt-caching); boundaries affect reuse opportunities, not prompt
identity or output semantics. String message status/phase metadata is accepted but has no Qwen
prompt representation.

`input_file`, `input_audio`, image `file_id`, non-`auto` image detail, reasoning metadata without raw
reasoning text, partial tool Items, and other Item/content types are not supported. HTTP media URLs
stored in a response chain are fetched again when that chain is continued; use data URIs when the
historical media bytes must be immutable.

### Function tools

Responses function definitions may be declared directly rather than inside Chat Completions'
nested `function` object:

```json
{
  "type": "function",
  "name": "get_weather",
  "description": "Get current weather",
  "parameters": {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"]
  },
  "strict": false
}
```

They may also be grouped in a Responses namespace:

```json
{
  "type": "namespace",
  "name": "mcp__weather",
  "description": "Weather service",
  "tools": [{"type": "function", "name": "get_current"}]
}
```

NInfer gives each namespace/function pair a distinct internal Engine identity and restores the
separate `namespace` and `name` fields in aggregate output, SSE events, and replayed Items. The same
function name may therefore appear in different namespaces. Namespace members remain ordinary
client-executed functions; this does not add a remote MCP executor.

NInfer renders these definitions in the Qwen prompt and parses model output into separate
`function_call` output Items. Each output has a protocol Item `id` (`fc_...`) and a distinct
`call_id` (`call_...`). The client executes the function and sends a `function_call_output` Item in
a later request. Only functions in the current effective tool set can become structured calls;
undeclared model output remains ordinary text. `allowed_tools` with mode `auto` filters that set
without changing declaration order, while `tool_choice:"none"` disables structured tool output even
when the history contains earlier calls.

NInfer does not execute functions or enforce JSON Schema through constrained decoding, so
`strict:true`, required or named tool choice, hosted tools, remote MCP tools, and custom free-form
tools are rejected. Deferred loading, output schemas, and caller restrictions that exclude direct
invocation are also rejected because their semantics cannot be honored.

### Response object and usage

A terminal wire response has `object: "response"`, one of `completed`, `incomplete`, or
`cancelled` in `status`, and a typed `output` array. NInfer may emit:

- a `reasoning` Item containing raw `reasoning_text` and an empty summary;
- an assistant `message` containing an `output_text` part;
- one or more `function_call` Items.

Ordinary model/string stops produce `completed`. Output-token or context-capacity exhaustion
produces `incomplete` with `incomplete_details.reason: "max_output_tokens"`. Errors accepted after
an SSE response has started produce `response.failed`; validation and preparation errors remain
normal HTTP error responses. `completed_at` is populated only for completed Responses. A
reasoning-only incomplete result contains no invented empty assistant message.

Usage is checkpoint-native:

```json
{
  "input_tokens": 42,
  "input_tokens_details": {"cached_tokens": 17},
  "output_tokens": 12,
  "output_tokens_details": {"reasoning_tokens": 5},
  "total_tokens": 54
}
```

`input_tokens` includes the chat template and expanded media tokens. `cached_tokens` is the exact
checkpoint-proven prompt prefix reused by Engine. `output_tokens` is the count of accepted generated token
IDs, including a withheld stop token when applicable. `reasoning_tokens` is counted in the Qwen
output decoder while accepted tokens are still in the reasoning channel; it is not estimated by
re-tokenizing decoded text.

### Responses streaming

Set `stream:true` for semantic Server-Sent Events. Every frame uses both the SSE event name and a
matching JSON `type`, and every JSON event has a monotonically increasing `sequence_number`:

```text
event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":7,...}

```

The normal lifecycle is:

1. `response.created`, then `response.in_progress`;
2. `response.output_item.added` and `response.content_part.added`;
3. zero or more `response.reasoning_text.delta` or `response.output_text.delta` events;
4. matching `*.done`, `response.content_part.done`, and `response.output_item.done` events;
5. exactly one `response.completed`, `response.incomplete`, or `response.failed` terminal event.

Function arguments use `response.function_call_arguments.delta` and `.done`. IDs, output indices,
and content indices remain stable, and concatenated deltas equal the terminal Item. Responses SSE
does not emit the Chat Completions `[DONE]` sentinel. With tools enabled, ordinary answer text still
streams immediately; only an ambiguous `<tool_call>` suffix or the structured tool region is held.
Malformed tool markup is flushed back as ordinary text without losing bytes.

### Local response state and resources

`store` defaults to `true`. Stored Responses live only in this server process and are bounded by an
LRU store. They are lost on restart and are not OpenAI's durable cloud retention service.

`previous_response_id` reconstructs the complete stored input/output Item history before the new
input. The current `instructions` value is placed first but is not saved into the continuation
context, matching the Responses rule that previous top-level instructions do not carry forward.
Function definitions are request configuration rather than conversation Items and must be sent
again on tool-result turns. The reconstructed prompt follows the ordinary Engine path, so compatible
checkpoint reuse applies naturally.

A stored Response also retains its resolved `preserve_thinking` value. A child which omits the
field inherits the parent value. An explicit different value creates a new semantic branch; prompt
rendering and identity still determine reuse. Changing the boolean alone never invalidates an exact
checkpoint already proved compatible by the model runtime.

For Engine-local reuse, a stored root Response receives one bounded session key derived from its
response ID, and every `previous_response_id` child inherits that key. `store:false` roots remain
anonymous; a `store:false` child may read its inherited session checkpoint but does not replace the
stored chain's latest endpoint. Response-store eviction or deletion removes the HTTP object, not an
independently retained Engine checkpoint; the latter remains bounded by the Engine's own retention
and pressure policy. No session key or cache marker is added to the HTTP schema.

Resource behavior:

| Endpoint | Contract |
|---|---|
| `GET /v1/responses/{id}` | returns the stored terminal object, or 404 `response_not_found`; stream recovery and non-empty `include` are rejected rather than ignored |
| `DELETE /v1/responses/{id}` | removes public retrieval and returns `response.deleted`; descendant contexts already retained by other Responses remain usable |
| `GET /v1/responses/{id}/input_items` | returns normalized Items supplied to that request; supports `after`, `limit` `1..100` (default `20`), and `order` `asc|desc` (default `desc`); image URLs are redacted unless `include=message.input_image.image_url` |
| `POST /v1/responses/{id}/cancel` | explicitly fails because background execution is unsupported |
| `POST /v1/responses/compact` | explicitly fails with `compaction_not_supported` |

`store:false` Responses cannot be retrieved or used as `previous_response_id`. LRU eviction and
explicit deletion also make an ID unavailable. A single Response larger than the configured store
capacity fails with `response_store_capacity_exceeded` rather than silently pretending it was
stored.

### Responses input token count

`POST /v1/responses/input_tokens` uses the same prompt path as Create and does not run generation.
It accepts `model`, `input`, `instructions`, `previous_response_id`, reasoning, function tools and
tool choice, supported text/truncation values, and the `preserve_thinking` extension. Parent lookup,
call-ID normalization, template rendering, and media expansion are therefore identical to the
corresponding Create request:

```bash
curl http://127.0.0.1:8080/v1/responses/input_tokens \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","input":"Count this prompt."}'
```

```json
{"object":"response.input_tokens","input_tokens":11}
```

Unsupported Create fields include Conversations, prompt templates, context management, hosted
moderation, Structured Outputs/JSON mode, non-empty `include`, background execution, compaction,
files/audio, and OpenAI-hosted/MCP/custom tools. These are compatibility boundaries, not silently
accepted placeholders.

## Anthropic Messages

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "Explain prefix reuse in one sentence."}
    ]
  }'
```

The endpoint accepts top-level System text, ordered User/Assistant/System history, text and image
blocks, Thinking history, tool-use history, tool results, user-defined tools, aggregate responses,
and Anthropic SSE. Consecutive User or Assistant messages are joined without adding separators.
Mid-conversation System messages retain their input position. A final text-only Assistant message
is an Assistant prefill: generation continues its existing text instead of opening another turn.
Assistant prefill cannot contain media, Thinking, or tool calls and cannot start with Thinking
enabled.

Claude Code may place its attribution metadata in the first block of a top-level System array. If
that block is a text block beginning exactly with `x-anthropic-billing-header:`, NInfer consumes the
whole block before token counting, prompt preparation, and cache identity construction. The rule is
positional: a string-form System value, a later array block, or an inline System message with the
same text remains ordinary prompt content. A `cache_control` marker attached to the consumed block
is consumed with it rather than moved to adjacent content.

`max_tokens` is optional for local clients and otherwise uses `--default-max-tokens`; a positive
value is the complete output budget. `max_tokens:0` is rejected because NInfer does not expose a
completed zero-output cache-prewarm lifecycle. `temperature`, `top_p`, `top_k`, and
`stop_sequences` enter Engine execution. A matched custom stop is returned as
`stop_reason:"stop_sequence"` together with the actual `stop_sequence`; context exhaustion returns
`model_context_window_exceeded`.

Thinking supports `disabled`, `adaptive`, and `enabled`. Enabled Thinking requires
`budget_tokens >= 1024` and less than `max_tokens`, and that budget is passed to Engine. Visible
Thinking is returned with an opaque compatibility signature; SSE emits its `signature_delta`
before closing the block. Request lowering reconstructs the local prompt from the visible
`thinking` text and treats `signature` as non-semantic transport metadata, so retained history
remains usable across serve restarts.
`display:"omitted"` is rejected because NInfer cannot provide Anthropic's
encrypted hidden-reasoning restore semantics. `preserve_thinking` remains a NInfer extension for
closed-turn reasoning history. `output_config.effort` follows the Chat Completions effort rules: an
effort the selected template does not accept returns `invalid_request_error` with code
`reasoning_effort_not_supported` on `output_config.effort` and the list of accepted efforts.

User-defined, non-strict tools support `name`, `description`, object `input_schema`, and
`input_examples`. `tool_choice` `auto`, `none`, `any` and `tool` (one named tool) are
executable; `output_config.format` accepts `json_schema`
([Structured output](#structured-output)). `strict:true`, active single-call enforcement, deferred tools, tools that exclude direct model
calls, Anthropic-provided/server tools, toolsets, MCP, and containers are rejected because their
required constraint or executor is absent. `tool_result` preserves text/image order and marks
`is_error:true` explicitly in the model prompt. For a visible Assistant tool-use turn, the next
User turn must provide exactly one leading result for every declared ID; valid results are matched
by ID and normalized to call order. A history that begins with results remains valid as a truncated
or imported conversation.

Block-level ephemeral `cache_control` on tools and supported System/User/Assistant/tool-history
blocks creates explicit shared-prefix candidates. At most four distinct block-level breakpoints are
accepted. Request-level `cache_control` targets the last cacheable block: it merges with an explicit
breakpoint at the same target and TTL, conflicts at the same target with a different TTL, and needs
an available fifth slot when four different explicit targets already exist. TTL must be `5m` or
`1h`; it is a protocol hint, not a wall-clock residency guarantee.

NInfer maps representable boundaries to exact prompt frontiers and ignores a legal but
unrepresentable advisory boundary without changing the prompt. Reuse still requires exact rendered
identity and can read an existing owner without another `cache_control`. Aggregate usage reports
verified reused tokens in `cache_read_input_tokens` and leaves cache creation unknown. Streaming
emits `message_start` after Engine admission commits the prefix selection and before
transfer/prefill output, so its uncached/cache-read split is already exact; terminal cumulative
usage matches the aggregate response.

Documents, Search Results, Files, Structured Outputs, server-tool results, container uploads, and
other execution-dependent blocks are rejected with the missing capability identified. Metadata,
service tier, inference geography, protocol-version/beta headers, cache TTL, and unknown advisory
fields do not block an otherwise executable request. The request `model` is any non-empty local
proxy label and is echoed in the response; it does not select the resident artifact.

Every Messages response carries a `request-id` header; error bodies also carry `request_id` and use
Anthropic error categories. Local admission overload maps to HTTP 529 and queue/media timeouts to
HTTP 504. Streaming owns the full Anthropic block lifecycle for Thinking, text, and tool use.

`POST /v1/messages/count_tokens` uses the artifact's tokenizer, chat template, and media expansion
without generation. It shares the same prompt normalization, tools, Thinking mode, Assistant
prefill, media processing, and cache-marker interpretation as Messages; output-only sampling and
streaming fields do not affect the count:

```bash
curl http://127.0.0.1:8080/v1/messages/count_tokens \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Count this prompt."}]
  }'
```

## TypeSafe System One

`POST /v1/systemone` (alias `POST /systemone`) serves TypeSafe's System One API from the loaded
model. Requests, answers and errors follow TypeSafe's published contract (OpenAPI 0.2.0) and the
observed behaviour of its hosted `jev-1.13` service, so a client written for Jev, including the
official `typesafe-sdk` (Python) and `@typesafe-ai/sdk` (JavaScript), runs unchanged against this
server once its base URL points here. Every answer is read from next-token probabilities in one
forward step per question; nothing is generated. Independent questions over one `state` share its
prefill and never see each other.

### Request

```json
{
  "model": "jev-latest",
  "state": "Customer: 'I was charged twice for order #9482. Please refund immediately!'",
  "questions": {
    "is_urgent": {
      "type": "noul",
      "instructions": "Is this customer inquiry urgent?",
      "criteria": {
        "true": "Customer demands immediate resolution or financial correction",
        "false": "Routine informational inquiry"
      }
    },
    "department": {
      "type": "choice",
      "instructions": "Which department should handle this ticket?",
      "criteria": {
        "billing": "Charges, invoices, duplicate payments, refunds",
        "shipping": "Tracking, delivery delays, lost packages",
        "general": null
      }
    },
    "customer_frustration": {
      "type": "score",
      "instructions": "How frustrated is the customer?",
      "criteria": ["Calm and polite", "Mildly annoyed", "Very upset / demanding",
                   {"level": "Hostile", "examples": ["threatens a chargeback"]}]
    }
  }
}
```

| Field | Contract |
|---|---|
| `model` | Required nonempty string. `jev-latest`, `jev-preview`, a versioned `jev-*`, the served id and any other name are accepted and none changes execution; an empty name is `400 Unknown model: `. A trailing `-t<T>` (for example `qwen3.8-27b-t1.5`) sets the temperature when `temperature` is absent. |
| `state` | Required: a string (it may be empty), an object or an array. Objects and arrays reach the model as 2-space indented JSON. |
| `questions` | Required map of 1 to 8,192 questions. Keys come back in `answers`, are never shown to the model and cannot be empty. |
| `temperature` | NInfer extension. A finite positive number, default 1; candidate logits are divided by it before normalisation. |
| `images` | NInfer extension. Image URLs or base64 data URIs shared by all questions, taking the same acquisition, preprocessing and Engine vision route as Chat Completions; the server needs `--vision`. This is not a TypeSafe feature, and a client should require nonzero `usage.vision_tokens` before treating an answer as image-grounded. |
| `policy` | NInfer extension. `{"readout": "single" or "averaged", "reasoning_budget": N, "escalate_entropy": F, "answer_weight": F}`, every field optional; omitted fields take the server's `--systemone-*` defaults, and an omitted object takes them all. See [Decision policy](#decision-policy). |

Any other top-level field, including `stream`, is
`400 {"detail": {"error_type": "api_usage_error", "message": "Invalid request."}}`, as on Jev.
Unknown fields inside a question are ignored. `instructions`, option and level descriptions and
Noul criteria each accept a string, an object or an array; objects and arrays reach the model as
2-space indented JSON, and field names inside them are the caller's own.

| `type` | `instructions` | `criteria` | Answer |
|---|---|---|---|
| `noul` | optional when a criterion describes yes or no | optional `{"true": ..., "false": ...}`; either may be null (undescribed); other keys are ignored | `noul` |
| `choice` | optional | required object of 1 to 255 options; a description may be null | `choice`, `confidence`, `probabilities` |
| `score` | optional | required array of 1 to 10 levels; a level cannot be null | `score`, `confidence`, `legend`, `probabilities` |

A Noul reads the candidates `Yes` and `No` and answers P(Yes). A Score reads the level digits
`0` to `N-1`.

#### Choice labels

Options are listed for the model with a label, and the answer is read at that label:

- If every key is one printable ASCII character, the key is its own label (`- A: description`).
- Otherwise up to 62 options are labelled `A`-`Z`, `a`-`z`, `0`-`9` in request order, with the key
  in brackets (`- A: [billing] description`). Each label is one native token, read at the first
  answer position.
- 63 to 255 options are labelled with a letter and a digit, `A0` to `Z4`. The first answer position
  gives P(letter); for every letter, an answer continued with that letter gives P(digit | letter).
  A label's log-probability is the sum of the two, and labels are normalised over the options
  exactly like one-token labels. The question's prefix is published once, so each
  letter branch prefills only the answer opener and its letter. The server checks per request that
  the tokenizer splits each label into its letter and its digit.

Labels are positional, so listing the same options in another order relabels them and can move
their probabilities.

### Response

```json
{
  "model": "qwen3.8-27b",
  "answers": {
    "is_urgent": {"type": "noul", "noul": 0.942},
    "department": {
      "type": "choice",
      "choice": "billing",
      "confidence": 0.8275,
      "probabilities": {"billing": 0.885, "shipping": 0.082, "general": 0.033}
    },
    "customer_frustration": {
      "type": "score",
      "score": 2.15,
      "confidence": 0.57,
      "legend": {"0": "Calm and polite", "1": "Mildly annoyed", "2": "Very upset / demanding",
                 "3": {"level": "Hostile", "examples": ["threatens a chargeback"]}},
      "probabilities": {"0": 0.01, "1": 0.12, "2": 0.58, "3": 0.29}
    }
  },
  "usage": {"input_tokens": 348, "output_tokens": 0}
}
```

- `model` is what answered: the served id, with `-t<T>` when the logits were scaled. TypeSafe
  reports the versioned model the same way, so a client that logs it records the substitution.
- Answer fields appear in Jev's order: `type`, `choice`, `confidence`, `probabilities` for a Choice
  and `type`, `score`, `confidence`, `legend`, `probabilities` for a Score.
- `probabilities` are exact renormalised values that sum to 1 (Jev rounds to two decimals), and
  `choice` is their first maximum.
- `legend` returns every level exactly as sent, including objects and arrays.
- `usage.input_tokens` counts the shared state once plus each branch's tokens past a prefix-cache
  hit. `usage.output_tokens` is 0: the answer is a readout, not generated text. `vision_tokens`
  (NInfer extension, image observations only) is the expanded vision-token count of the shared
  observation, already included in `input_tokens`.
- Every System One response, successful or not, carries `x-typesafe-request-id`, the same value as
  `x-request-id`.

#### Probabilities, score and confidence

Let $\ell_0, \dots, \ell_{N-1}$ be the candidate log-probabilities at the answer position and
$T > 0$ the temperature. With $\ell_{\max}$ the largest finite candidate,

$$P_i = \frac{\exp((\ell_i - \ell_{\max}) / T)}{\sum_j \exp((\ell_j - \ell_{\max}) / T)}$$

over finite candidates; a candidate that is not finite gets 0, and if none is finite the
distribution is uniform. Subtracting before dividing keeps the winner at very small temperatures.

$$\text{score} = \sum_{i=0}^{N-1} i \cdot P_i$$

Confidence is one minus the expected distance from the most probable answer, divided by that
distance for a uniform distribution, clamped to $[0, 1]$; one option or level has confidence 1.

- Choice, where every other option is at distance 1:
  $\text{confidence} = \dfrac{N \max_i P_i - 1}{N - 1}$.
- Score, where levels are $|i - k|$ apart and $k$ is the first most probable level:
  $\text{confidence} = 1 - \dfrac{\sum_i P_i\,|i - k|}{\lfloor N^2/4 \rfloor / N}$.
  Probability on the levels next to $k$ lowers it less than probability far away.

Both reproduce the confidences jev-1.13 returns for Choice answers and for Score answers of 1 to 10
levels, within their two-decimal rounding.

#### Decision policy

The default answers each question from one readout of the native prompt above, and its prompt
bytes do not change. The `policy` extension, or the server defaults `--systemone-readout`,
`--systemone-reasoning-budget` and `--systemone-escalate-entropy`, add two measured
improvements for questions with at most 26 options (a Noul, a Choice of up to 26 keys, a Score):

- `"readout": "averaged"` reads the question through three prompts and averages their
  renormalised distributions: the native prompt, a lettered decision prompt with the option
  order rotated by one (Choice only), and an evidence/criterion framing with the options listed as
  `A. key: description`. The lettered prompts render the state and the question themselves, so
  each costs one more prefill per question and publishes nothing to the prefix cache.
- `"reasoning_budget": N` (0 disables) escalates a question whose averaged distribution has
  normalised entropy above `escalate_entropy` (default 0.66) to one bounded-thinking generation
  over the lettered prompt, with `N` thinking tokens and 256 answer tokens. When the answer is
  exactly one declared letter, it is mixed into the distribution at `answer_weight` (default 0.7);
  any other outcome leaves the distribution as it was. An escalated question costs seconds, so
  the budget is a per-request or per-deployment latency decision.

`usage.policy` reports `readout`, the number of `escalated` questions and their
`reasoning_tokens` whenever a policy was active. `usage.input_tokens` bills every readout and
escalation prompt. On the 231 public JevBench items the averaged readout alone moved the
Qwen3.8-27B NVFP4 artifact from 189 to 204 correct (Brier 0.246 to 0.174) and escalating the
highest-entropy quarter with a 1,024-token budget to 210 (Brier 0.149); the study behind the
defaults is recorded in [the decision head plan](research/decision-head-plan.md).

### Errors

Errors use FastAPI's `{"detail": ...}` body with the statuses Jev returns:

| Status | `detail` | When |
|---|---|---|
| 422 | list of `{"type", "loc", "msg", "input"}` (and `"ctx"`) | the body is not valid JSON or is empty, or violates the schema: a missing or mistyped field, empty `questions`, a Score without levels, a null level. Every violation is listed. |
| 400 | message | `Question key cannot be empty.`, `Noul question must have criteria or instructions: <id>`, `Choice question must have at least one choice: <id>`, `Too many choices. Must have at most 255 choices.`, `Too many score levels. Must have at most 10 levels.` |
| 400 | `{"error_type": "api_usage_error", "message"}` | an unknown top-level field or question type, an empty `model`, an invalid `temperature` or `images`, a media failure |
| 400 | `{"error_type": "max_tokens_exceeded"}` | a branch longer than `--max-context`, or more than 8,192 questions |
| 401, 403 | `{"error_type": "authentication_error", "message"}` | with `--api-key`: a wrong key (401) or none (403) |
| 405 | `"Method Not Allowed"` | any method but POST; the response carries `Allow: POST` |
| 429, 503 | `{"error_type": "rate_limit_error"` or `"api_error", "message"}` | the request queue is full or its wait timed out; both SDKs retry these |
| 500 | `{"error_type": "api_error", "message"}` | an internal failure |

### Execution and prefix reuse

1. `state` becomes the system prompt `"You are an evaluation assistant. State to evaluate:\n" +
   state`; each question is one user turn after it, answered with thinking off.
2. A single question runs read-only in the context cache and publishes nothing: its prefix is
   never reused.
3. With two or more questions, the first branch places an explicit prefix boundary after the state,
   prefills it once and publishes it. Later branches read it and prefill only their own question.
4. A Choice with letter-digit labels also publishes its question, so each letter branch prefills
   only the answer opener and one letter.
5. Branches run one after another as separate Engine requests, so a call's latency grows with its
   question count and with the letters of a large Choice. Jev evaluates a whole call in one pass.
   The Python SDK's default timeout is 10 seconds; raise it for calls with hundreds of questions.

### Differences from Jev

- The loaded model answers, not Jev: probabilities differ, and thresholds tuned on Jev need
  revalidation.
- Probabilities are exact rather than rounded, `usage.output_tokens` is 0 and `input_tokens`
  counts this server's prompt.
- Option order matters, because labels are positional.
- Latency grows with the number of questions (see above).
- Model names are not validated.
- Beyond Jev's limits: `temperature`, `images`, state up to `--max-context` tokens per branch (Jev
  allows 32k tokens of state and longest question, 64k in total) and up to 8,192 questions.

### Client usage examples

The [decision arcade](decision-arcade.md) provides interactive [Tetris](tetris-demo.html),
[Chess](chess-demo.html), [Kitchen Rush](kitchen-demo.html) and [Cube](rubiks-demo.html) clients. They expose the full choice
distribution, measured client latency, exact requests/responses and a separately labeled local
reference policy. Kitchen Rush compares paired kitchens with identical seeded orders and either
real-time or paused decision clocks. Its cooperative mode assigns independent model actors to
two chefs in one shared kitchen, exposing handoffs and resource conflicts; stale responses never
substitute a different job.
Chess retains every legal move; positions above its 62-code alphabet use piece selection followed
by move selection, with both conditional distributions shown. The guide covers controls,
measurement boundaries and reproducible evaluation commands.

#### Existing Jev clients

Both TypeSafe SDKs append `/v1/systemone` and `/v1/models` to their base URL, so the base URL is the
server root, not `/v1`. An application already built on Jev switches with two environment variables
and no code change:

```bash
TYPESAFE_BASE_URL=http://127.0.0.1:8080
TYPESAFE_API_KEY=local-secret   # the server's --api-key; any printable value when it has none
```

`GET /v1/models` answers both SDK families from one body: `data` for OpenAI clients and `models`
(`name`, `description`, `release_date`) for TypeSafe's `models.list()`, listing the served id and
the `jev-latest` and `jev-preview` aliases. `data` lists the served model first and then the
advertised rerank id (`ninfer-choice-rerank-v1` unless `--rerank-model-id` replaces it) when that
id differs from the served model; a client that discovers the chat model reads `data[0]`.
`GET /v1/models/{id}` resolves either id.

#### cURL

```bash
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "jev-latest",
    "state": "Customer: '\''I was charged twice for order #9482. Please refund immediately!'\''",
    "questions": {
      "is_urgent": {"type": "noul", "instructions": "Is this customer inquiry urgent?"},
      "department": {
        "type": "choice",
        "instructions": "Which department should handle this ticket?",
        "criteria": {"billing": "Charges and refunds", "shipping": "Delivery", "general": null}
      }
    }
  }'
```

#### Python (`typesafe-sdk`)

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

with TypeSafeClient(base_url="http://127.0.0.1:8080", api_key="local-secret") as client:
    response = client.system_one(
        state="Customer: 'I was charged twice for order #9482. Please refund immediately!'",
        questions={
            "is_urgent": Noul(instructions="Is this customer inquiry urgent?"),
            "department": Choice(
                instructions="Which department should handle this ticket?",
                criteria={"billing": "Charges and refunds", "shipping": "Delivery", "general": None},
            ),
            "customer_frustration": Score(
                instructions="How frustrated is the customer?",
                criteria=["Calm and polite", "Mildly annoyed", "Very upset / demanding"],
            ),
        },
    )

print(response.model)  # the served model, e.g. qwen3.8-27b
print(response.answers["is_urgent"].noul)
department = response.answers["department"]
print(department.choice, department.confidence, department.probabilities)
print(response.answers["customer_frustration"].score)
```

#### JavaScript / TypeScript (`@typesafe-ai/sdk`)

```typescript
import { TypeSafeClient, choice, noul } from "@typesafe-ai/sdk";

const client = new TypeSafeClient({ baseURL: "http://127.0.0.1:8080", apiKey: "local-secret" });

const response = await client.systemOne({
  state: "Customer: 'I was charged twice for order #9482. Please refund immediately!'",
  questions: {
    is_urgent: noul("Is this customer inquiry urgent?"),
    department: choice("Which department should handle this ticket?", {
      billing: "Charges and refunds",
      shipping: "Delivery",
      general: null,
    }),
  },
});

console.log(response.model, response.answers.is_urgent.noul, response.answers.department.choice);
```

## Document reranking: `POST /v1/rerank`

Jina-shaped reranking. Each document is one System One Choice over the query, executed in-process
on the loaded model. The route does not call `POST /v1/systemone`. The four Choice labels stay
inside that scoring step; the response does not name them. Like System One questions, the
documents run as consecutive Engine requests, one per document, so latency grows with the
document count. Unlike System One, rerank only reads the prompt cache: the prefix its documents
share is the short query, so it publishes and captures nothing and cannot evict cached
conversations.

```json
{
  "model": "ninfer-choice-rerank-v1",
  "query": "where is the invoice?",
  "documents": ["The invoice is in the drawer.", {"text": "The weather is clear."}],
  "top_n": 2,
  "return_documents": true
}
```

`query` is a required nonempty string. `documents` is a required array of 1 to
`--rerank-max-documents` entries (256 by default, the same cap as `POST /v1/score`). An entry is a
string or an object with a nonempty `text` string. Other fields on a document object are ignored.
An empty string is HTTP 400.

`model` is optional, and `null` counts as omitted, as on `POST /v1/chat/completions`. Any other
value must be a nonempty string equal to the served model id or the advertised rerank id
(`--rerank-model-id`, `ninfer-choice-rerank-v1` by default). Either name runs the loaded model. An
empty string or a non-string value is HTTP 400. Any other id is HTTP 404 `model_not_found`.

`top_n` is an optional integer. Omitted or `null`, it equals the document count. Any integer
outside `1..N`, including one beyond the 32-bit range, is clamped into that range; a non-integer is
HTTP 400. Unknown top-level fields are ignored.

`return_documents` defaults to true. Each result then includes `document.text`. `false` omits
`document`.

```json
{
  "model": "ninfer-choice-rerank-v1",
  "results": [
    {"index": 0, "relevance_score": 0.91, "document": {"text": "The invoice is in the drawer."}}
  ],
  "usage": {"total_tokens": 1842}
}
```

`results` are best-first by `relevance_score`, descending. Equal scores keep the lower original
`index` first. `index` is the position in the request `documents` array. `model` is always the
advertised rerank id. `usage.total_tokens` is the billed prompt tokens of the Choice batch, counted
the same way System One counts `input_tokens`: the first question bills its whole prompt and each
later question bills only the tokens past the shared query.

The relevance score is the expected value of the four options, in this order:

| Option | Prompt criteria | Default weight |
|---|---|---:|
| exact | Document answers the query directly / is an exact match | `1` |
| substitute | Document is a useful near-match or substitute answer | `0.6` |
| complement | Document is related / complementary but not sufficient alone | `0.25` |
| irrelevant | Document is unrelated to the query | `0` |

`relevance_score = 1.0·P(exact) + 0.6·P(substitute) + 0.25·P(complement) + 0·P(irrelevant)` with
the defaults. `--rerank-weight-exact`, `--rerank-weight-substitute`, `--rerank-weight-complement`
and `--rerank-weight-irrelevant` replace those coefficients. Each question's instruction is exactly
`How relevant is this document to the query?`, a blank line, `Document:`, and the document text.

With `--api-key` set, a missing or wrong bearer token or `x-api-key` is HTTP 401 and the body is
the OpenAI error object `{"error":{"message","type","param","code"}}`. A request that fails
validation is HTTP 400 in that same object. A generation failure is HTTP 500 `internal_error`.

## Authentication and CORS

Pass `--api-key VALUE` to require the same value as an OpenAI/TypeSafe bearer token or Anthropic
`x-api-key` header. `--api-key-file PATH` reads the key from a file instead, so it does not appear
in the process command line that other local processes can read; startup fails if the file is
unreadable or empty. `GET /health` and CORS preflight requests remain unauthenticated.

```bash
curl http://127.0.0.1:8080/v1/models \
  -H 'Authorization: Bearer local-secret'
```

`--cors` adds permissive browser CORS headers. It is disabled by default.

## Server options

The table lists executable defaults. The startup example selects a long-context FP8/MTP3 profile.

| Option | Meaning | Default |
|---|---|---:|
| `--host H` | listen address | `127.0.0.1` |
| `--port N` | listen port | `8080` |
| `--api-key KEY` | required bearer or `x-api-key` value | unset |
| `--api-key-file PATH` | read the required key from a file | unset |
| `--model-id ID` | override the public OpenAI model alias | artifact `identity.model_id` |
| `--rerank-model-id ID` | model id advertised for `POST /v1/rerank` and accepted beside the served id | `ninfer-choice-rerank-v1` |
| `--rerank-max-documents N` | maximum `documents` array length for `POST /v1/rerank` | `256` |
| `--rerank-weight-exact F` | relevance weight of the exact-match Choice option | `1` |
| `--rerank-weight-substitute F` | relevance weight of the substitute Choice option | `0.6` |
| `--rerank-weight-complement F` | relevance weight of the complement Choice option | `0.25` |
| `--rerank-weight-irrelevant F` | relevance weight of the irrelevant Choice option | `0` |
| `--max-context N` | logical context ceiling of each sequence | `8192` |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context` | `8192` |
| `--max-concurrency N` | maximum admitted requests; valid range `1..8` | `1` |
| `--clamp-concurrency-to-pool` | lower concurrency to the number of full-context sequences the KV pool backs | off |
| `--desktop-reserve-gib N`, `--desktop-reserve-mib N` | device memory kept free for the desktop and other applications; sizing uses device-wide (NVML) free memory, and no Engine allocation may take the device below the reserve after startup | `8` GiB |
| `--kv-slack-floor-mib N` | device memory `--kv-capacity auto` leaves unreserved | `1024` |
| `--max-pending-requests N` | additional requests allowed to wait for admission | `16` |
| `--pending-timeout-ms N` | maximum preparation-plus-admission wait | `30000` |
| `--prefill-chunk N` | text-prefill chunk | `1024` |
| `--log-stats-interval-ms N` | aggregate throughput report interval; `0` disables it | `5000` |
| `--log-level trace\|debug\|info\|warning\|error\|critical\|off` | pretty stderr verbosity | `info` |
| `--device N` | CUDA device index | `0` |
| `--context-cost-presets FILE` | optional runtime context-cost preset registry | generic + compiled defaults |
| `--max-request-mib N` | body-size limit before JSON parsing | `384` |
| `--media-cache-mib N` | LRU-retained prepared BF16 media payloads; `0` disables retention | `1024` |
| `--media-live-mib N` | all live prepared BF16 media payloads | `2048` |
| `--media-preprocess-threads N` | bounded media preprocessing workers; `0` selects at most 16 from host concurrency | `0` |
| `--request-log-jsonl FILE` | append full-precision server/request records | disabled |
| `--response-store-max-records N` | maximum locally retained Responses objects | `1024` |
| `--response-store-max-mib N` | total local Response envelope/Item/context budget | `256` |
| `--kv-dtype bf16\|int8\|fp8\|nvfp4\|k8v4` | KV-cache storage | `bf16` |
| `--spec mtp\|dflash\|dflash2` | speculative backend | off |
| `--draft-tokens N` | MTP `1..5`; DFlash/DFlash2 `1..15` | unset |
| `--draft-policy fixed\|adaptive` | MTP drafts per round: always `--draft-tokens`, or a count in `1..--draft-tokens` chosen per request from its acceptance (Qwen3.8-Flash-Next) | `fixed` |
| `--lm-head-draft` | optimized proposal head | off |
| `--default-max-tokens N` | output limit when omitted by a request | `8192` |
| `--default-thinking-budget N` | positive thinking cap inherited by thinking-enabled requests | unset |
| `--vision` | enable media input and load Vision GPU allocations | off |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--no-prefix-reuse` | disable compatible-prefix caching | prefix reuse on |
| `--device-state-slots N` | extra Device checkpoint StateImages beyond the active-lane guarantee | `max-concurrency` |
| `--host-state-slots N` | pinned Host StateImage capacity | `8` |
| `--host-kv-mib N` | shared pinned Host Main/Backend KV byte capacity in MiB | `8192` |
| `--max-private-continuations N` | private continuation descriptor capacity | `2 * max-concurrency` |
| `--max-shared-prefixes N` | Engine-wide shared stable-prefix descriptor capacity | `max(max-concurrency, 4)` |
| `--max-long-anchors-per-continuation N` | private long-anchor limit per continuation | `2` |
| `--no-thinking` | disable thinking by default | thinking on |
| `--preserve-thinking` | preserve closed-turn assistant reasoning by default | off |
| `--cors` | permissive browser CORS headers | off |
| `--systemone-readout single|averaged` | System One readout for requests without a `policy` ([decision policy](#decision-policy)) | single |
| `--systemone-reasoning-budget N` | System One thinking budget for escalated questions; 0 disables escalation | 0 |
| `--systemone-escalate-entropy F` | normalised-entropy threshold above which a System One question is escalated | 0.66 |
| `--temperature F` | process-level temperature override | unset |
| `--top-p F` | process-level top-p override | unset |
| `--top-k N` | process-level top-k override (`0..20`; zero selects the top-20 cap) | unset |
| `--min-p F` | process-level min-p override | unset |
| `--presence-penalty F` | process-level presence-penalty override | unset |
| `--frequency-penalty F` | process-level frequency-penalty override | unset |
| `--seed N` | fixed seed when a request omits one | fresh random seed per request |
| `--greedy` | force exact argmax for all requests | off |

Context-cost coefficients resolve once at startup from generic defaults, matching compiled values,
and optional transfer or prefill entries from `--context-cost-presets FILE`. Prefill entries match
the hardware and a signature derived from the actual Text/Vision configuration, bindings and Uses.
A new representation without a matching measurement uses generic prefill coefficients. A malformed
file aborts startup; the operational context-cost record and JSONL `server_start` identify the
selected source.

Engine selects sampling defaults from the loaded architecture and the request's resolved thinking mode.
Qwen3.6-27B and Qwen3.8-27B use `1.0/0.95/20/0/0` for
temperature/top-p/top-k/min-p/presence penalty in thinking mode and `0.7/0.80/20/0/0` in
non-thinking mode. Qwen3.6-35B-A3B uses a presence penalty of `1.5` in both modes.
Frequency penalty is `0` for all registered presets. Process flags override registered values,
request fields override process flags, and `--greedy` finally forces temperature `0`.

For `C=--max-concurrency` and `H=--device-state-slots`, total Device StateImage capacity is `C+H`:
`C` slots guarantee active requests and `H` is a global checkpoint pool. Host State and Host KV are
independent startup-fixed pinned-memory capacities; Host KV is shared by Main and the selected
Backend pool and is consumed in physical page extents. `--no-prefix-reuse` selects root-only Engine
mode and cannot be combined with any of the seven explicit context-cache capacity flags, including
zero-valued flags.

Run `./build/apps/ninfer-serve --help` for the exact option contract.

Serve writes human-readable operational records to stderr using
`YYYY-MM-DD HH:MM:SS.mmm  LEVEL  message`. Normal output covers material startup milestones,
readiness, request lifecycle, fixed-interval throughput, and shutdown; `--log-level debug` exposes
internal startup and resource-planning detail. A terminal may use one transient line during startup,
but Serve throughput is always a persistent record. Redirected stderr contains no terminal control
sequences. Pretty values use readable units and rounded rates; use the independent request JSONL for
complete fields and full precision. Operational records never contain prompts, generated text,
request bodies, credentials, or arbitrary client error messages.
If a tool marker is returned to text because its structure or tool identity cannot be represented,
Serve emits one warning with only the failure classification, never the generated markup.

## Structured request log

`--request-log-jsonl FILE` enables the machine-readable measurement log. The server opens `FILE`
in append mode and flushes every event, so successive model or MTP blocks may share one campaign
file. The parent directory must already exist. Failure to open the file aborts startup; the log path
is also rejected if it resolves to the model artifact.

Every line is one `ninfer_serve_request_log` schema-v21 JSON object. All events carry
`timestamp_unix_ms` and a process-unique `server_instance_id`; request IDs are monotonic only within
that server instance. Successful request-start records include request-scoped acquisition,
media-preprocessing wall/work, tokenizer, cache hit/miss/single-flight, and payload-size fields;
they do not infer request behavior from process-global counter deltas.

| Event | Contents |
|---|---|
| `server_start` | artifact path, architecture, public name, actual formats and prefill signature; resolved Engine and context-cache capacities, thinking/non-thinking sampler defaults plus process overrides, thinking-history and thinking-budget defaults, Device arenas, the optional non-additive Vision layout inside the unified workspace, Host State/KV capacity and occupancy, KV sizing ledger, CUDA Graph allowance, CUDA/GPU environment, and redacted argv |
| `request_start` | protocol, resolved sampler and seed, requested reasoning effort, actual initial thinking mode and optional budget, Responses semantic-change flag, output budget, stream/message/tool shape, `client` (the `User-Agent` header), and `tools_digest` (an FNV-1a fingerprint of the ordered tool definitions; a change between two turns of one conversation invalidates its whole cached prefix) |
| `request_rejected` | parsed request shape, requested reasoning effort, media-item count, `phase: "prepare"`, and the exact HTTP status/type/code/parameter/message for a synchronous preparation rejection |
| `request_done` | finish reason, prompt/completion/cache/computed-prefill tokens, prefix reuse path, tool-call parse diagnostics, request-owned materialization cost/search diagnostics, thinking-budget application counters, unrounded request-stage seconds, per-request Engine Host exposure, and complete speculative-decoding counters |
| `request_error` | the resolved request configuration and the generation, cancellation, or pre-outcome transport terminal message |
| `throughput` | interval token/decode/context-cache pressure counter deltas, authoritative worker Host-work deltas, current scheduler/resource gauges, and decode-round batch statistics |

`requested_reasoning_effort` records the explicit option, or `null` when unspecified;
`preserve_thinking` records the resolved value, including the server default.
`enable_thinking` records whether the response starts in thinking mode.

`request_done.result.tool_call_parse` records whether a complete marker was seen, the structured
call count, empty non-string arguments omitted during normalization, schema-mismatched arguments
preserved for consumer validation, and a stable text-fallback reason. Fallback reasons are `none`,
`malformed_structure`, `duplicate_parameter`, `invalid_tool_name`, `undeclared_tool`, and
`trailing_content`. These counters contain no tool arguments or generated text.

`request_done.timings_seconds` contains `prepare`, `ttft`, `vision`, `prefill`, `decode`, and `total`
as full-precision JSON numbers. Its `speculative` object contains `backend`, `draft_window`, `rounds`,
`drafted_tokens`, `accepted_tokens`, `fallback_steps`, and `accepted_per_position`. Rates can be
derived downstream from raw token counts and seconds instead of rounded stderr strings.

For `server_start.memory`, `workspace.capacity_bytes` is the only physical workspace allocation.
When Vision is enabled, `vision_workspace` reports the aggregate prompt and maximum-item token
bounds plus encode peak and handoff layout/usage within that same allocation; these bytes must not
be added to `workspace.capacity_bytes`. The field is `null` when Vision is disabled.

`request_done.engine_timing` separates FIFO `queue_wait_seconds`, blocking
`device_wait_exposed_seconds`, and five mutually exclusive Host-active exposure phases under
`host_exposed_seconds`: `engine_boundary`, `program_submit`, `program_post`,
`engine_commit_output`, and `engine_maintenance`. `total` is exactly their sum and excludes Device
wait. The nested `decode` object reports the request's decode-class Host exposure, Device wait, and
round count; `units` reports its prefill/control unit counts. In a compact batch every participating
request is delayed by the full round, so these values explain request latency but **must not be
summed across concurrent requests**.

The JSONL file contains no generated response text and never records an API-key value; `argv`
replaces that value with `<redacted>`. Operational stderr summaries are rounded and are not the
aggregation source. OpenAI Responses, OpenAI Chat, and Anthropic generation requests receive a
request ID when they enter synchronous preparation. Successful preparation produces
`request_start`; a preparation failure produces `request_rejected` without a matching start. Each
started generation transaction then has exactly one machine terminal: `request_done` when Engine
returns its outcome, or `request_error` when generation fails before an outcome exists. Later
response rendering, Responses storage, or terminal transport failures are operational response
events only and do not add a second JSONL terminal. Schema/model validation rejections before
preparation and token-count-only calls are not measurement requests and do not receive request IDs.

By default the server persistently reports aggregate activity every five seconds. `prefill` counts
prompt suffix tokens actually computed during the interval, excluding prefix-cache hits; `decode`
counts tokens finally committed by decode rounds, excluding the first token produced by prefill.
For MTP, DFlash and DFlash2 this is the accepted committed output, not draft or rejected tokens.
Pretty `batch` and JSONL `average_size` are decode row-rounds divided by decode rounds during the
same interval. The
`running`, `prefilling`, `decode_ready`, `waiting`, `materializing`, `capture_pending`, and
`terminal_pending` fields are the Engine scheduler snapshot at the end of the interval. The JSONL
`context_cache` object reports selection, capture, transfer, COW, pressure spill, private/shared
owner degradation and eviction, checkpoint drop, pressure search, budget exhaustion, maximal fallback, and historical-fork
counters as interval deltas; `occupancy` and `last_selection` are end-of-interval gauges. Materialization predictions are
request-owned and appear only on the corresponding `request_done` event.
`pressure.searches` counts plans accepted into Program resource transactions, including a transaction that later ends in
request-local abort; committed victim counters likewise report the resulting stable cache changes.

The JSONL `throughput.host_work` object is the aggregation authority: the Engine worker counts each
wall-time segment once, independent of batch size. `elapsed_seconds` contains the same five
mutually exclusive Host phases and their `total`; `device_wait_seconds` is separate.
`work_class_seconds` splits Host and Device-wait time into decode, prefill, and control classes.
`detail_subset_seconds` and `detail_invocations` expose admission, context-transaction, replica, and
stats-publication slow paths; these detail values are already contained in a top-level Host phase
and must not be added to `total`. Per-round, per-row-round, and per-invocation normalized values are
`null` when their denominator is zero. Pretty throughput contains nonzero token rates and counts,
the current running/prefill/decode-ready composition, nonzero waiting/materialization/terminal
states, average decode batch, and Host-active time plus its fraction of the interval. Use JSONL for
complete measurement analysis.
Intervals with context materialization or retention activity are retained even when they contain no
token execution; only fully idle intervals are omitted. Downstream measurement should prefer the
raw counters and seconds over rounded stderr rates.

## Execution behavior

### Structured output

The Engine constrains answer tokens before greedy selection or stochastic top-k/top-p sampling.
Chat Completions accepts `response_format: {"type":"json_object"}` for a JSON object, or:

```json
{"response_format":{"type":"json_schema","json_schema":{"name":"result","strict":true,"schema":{"type":"object","properties":{"status":{"type":"string","enum":["confirmed","unknown"]}},"required":["status"],"additionalProperties":false}}}}
```

Responses uses the flat equivalent under `text.format`:
`{"type":"json_schema","name":"result","schema":{...}}`. Anthropic Messages accepts
`output_config.format: {"type":"json_schema","schema":{...}}`. The C++ interface is
`ExecutionOptions::structured_output`, with `JsonObject` or `JsonSchema` and a serialized schema.
Schema constraints apply regardless of the optional `strict` flag.

Supported assertions are explicit types (including nullable type arrays), `properties`,
`required`, `additionalProperties`, `items`, `prefixItems`, `minItems`, `maxItems`, numeric bounds,
`enum`, `const`, `anyOf`, and unescaped document-local `$ref` paths into `$defs`/`definitions`.
Typed assertions require an explicit `type`, and required names must be declared in `properties`.
`$ref` and `anyOf` cannot have sibling assertions; `enum` and `const` may additionally specify a
matching type. Ordinary schema annotations are accepted. Unsupported assertions return HTTP 400
`unsupported_json_schema`; this includes `pattern`, string length bounds, `format`, `oneOf`,
`allOf`, and `uniqueItems`. Unspecified `additionalProperties` retains its JSON Schema meaning.

Reasoning remains separate and unconstrained; the answer grammar activates after `</think>`. An
end token is allowed only after a complete matching value. Output/context limits and cancellation
can still produce an incomplete JSON prefix: inspect the finish reason before parsing. Streaming
chunks are prefixes. Structured output cannot be combined with active function tools, custom stops,
raw output, or special-token preservation.

A required tool choice (`tool_choice:"required"`, a named function, `allowed_tools` in `required`
mode, or Anthropic `any`/`tool`) constrains the answer to exactly one complete native call to an
eligible function; argument text stays non-strict. It cannot be combined with custom stops.

Compiled grammars are cached per model frontend. Each request owns fresh matcher state, including
requests that reuse a prompt prefix. MTP keeps speculating: the Program walks a fork of the
matcher along the drafts, truncates each draft at its first token the grammar rejects or that
completes it, and gives verification column j the grammar state after drafts 0..j-1. DFlash
proposes inside the round, so its constrained lanes verify no drafts. Masks occupy stable
Program-owned device storage for graph replay.

The server owns one resident Engine with a startup-fixed capacity of `1..8` active generation
requests. At each decode boundary, every decode-ready request is compacted into one batch and
processed by one model traversal and, when graphs are enabled, one exact-batch CUDA Graph replay. A
request joins that batch only after its single-request prefill finishes; when it completes or is
cancelled, the next boundary rebuilds the batch without an empty row.

`--max-pending-requests` bounds the requests waiting behind the active set. The total generation
request lifetime capacity is `max_concurrency + max_pending_requests`, including requests still in
CPU/media preparation and completed model results whose response has not yet been released. A full
capacity returns HTTP 429 with code `server_overloaded`. The absolute
`--pending-timeout-ms` deadline starts before preparation, covers media acquisition and Engine FIFO
waiting, and returns HTTP 503 with code `request_queue_timeout` if admission does not occur in time.
There is no admission ETA or unbounded overflow queue.

Input memory is bounded by the outstanding-request count and the per-request
`--max-request-mib` limit. Media requests additionally share one preparation permit, so a waiting
media request retains the same cancellation and timeout deadline. Model output is bounded by the
same finite request count and each request's effective output-token limit; output callbacks and
network serialization run outside the GPU executor and do not delay formation of the next batch.

`--max-context` is each sequence's logical ceiling. `--kv-capacity` fixes the shared Main Text KV
pool used by active requests and retained prefixes. `auto` accounts for the complete enabled runtime
and leaves 1 GiB of sizing headroom; omitting the option makes it follow `--max-context`. Capacity
resolves once at startup.

Admission reserves the full prompt-plus-effective-output page entitlement through request
completion. A request remains queued until a legal resource plan can satisfy that entitlement.

Each reusable checkpoint contains KV and complete continuation state. At admission, capture, and
finish boundaries, resource pressure may keep it on Device, move its StateImage and/or KV replicas
to pinned Host memory, or evict it. The planner compares incoming-request work with the later
recovery cost imposed on retained checkpoints. Active requests retain their state and completion
reservations, and placement choices preserve model semantics. The full policy and invariants are
defined in [Resource scheduling and context cache](maintainer/resource-scheduling-and-context-cache.md).

Compatible prefixes are reused for both text and multimodal histories unless the server starts with
`--no-prefix-reuse`. A multimodal hit additionally requires matching token types, three-axis MRoPE
positions, encoded-media digest, grid, and consumer spans. Media wholly inside a matched prefix
skips Vision execution, while new suffix media is encoded normally. The pretty completion record
shows `cache N (P%, path)` using readable path labels; JSONL retains the exact
`prefix_cache_hit_tokens` and `prefix_reuse_path` fields. Machine paths are `root`,
`private_endpoint`, `private_turn_closure`, `private_response_replay`, `private_long_anchor`, and
`shared_stable_prefix`. Reuse validation covers KV, recurrent state, hidden state, selected-backend
state, and the exact prompt frontier. With stable `preserve_thinking=true`, the auxiliary checkpoint
rolls to the message frontier immediately before the current response's deterministic generation
prologue. A normalized response, compact-summary instruction, or replacement user suffix therefore
replays the small generation prologue and only the changed suffix while retaining the complete
stable conversation prefix. Stable `false` places the turn-closure checkpoint before the first
assistant opener in the open turn, so closing that turn can recompute its opener and omit its
reasoning without discarding the preceding conversation.

`preserve_thinking` selects the capture frontier for newly created checkpoints. Existing exact
checkpoints remain reusable across a mode change. If the desired boundary is behind the selected
reuse frontier and has no snapshot, the Engine keeps the valid hit and defers the new checkpoint. A
later request that diverges before every retained checkpoint starts from root. The JSONL completion
record exposes the restored checkpoint as `prefix_reuse_path`. Reasoning-effort changes participate
in rendered-token identity and exact-prefix selection.

An appended mid-conversation system message is an ordinary prompt suffix, so an unchanged prior
history remains eligible for `private_endpoint`. If the client modifies, removes, or moves a
historical system message, the token prefix genuinely differs and a miss/reset is correct.

Speculative backends preserve protocol output shapes, stop behavior, and usage accounting. If a stop
truncates a multi-token MTP, DFlash or DFlash2 round, the Engine commits the exact accepted target prefix so
a following compatible turn can reuse it. Output-limit and context-capacity finishes map to
`length`/ `max_tokens`; ordinary model or string stops map to `stop`/ `end_turn`.

Function tools are rendered into the model prompt and generated calls are parsed into protocol
responses. NInfer does not execute tools and does not enforce client JSON Schema through constrained
decoding.

Prompt-token usage includes chat-template and expanded media tokens. Generated-token usage comes
from accepted output token IDs, including a stop token whose decoded text may be withheld.
