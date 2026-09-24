# System One decision arcade

[Play Tetris](tetris-demo.html) · [Play Chess](chess-demo.html) · [Play Kitchen Rush](kitchen-demo.html) · [Play Beacon Runner](beacon-demo.html)

The arcade demonstrates the [TypeSafe-compatible System One API](serving.md#typesafe-system-one):
the application supplies an observation and a finite set of legal actions, NInfer scores the
candidate tokens, and the application applies the selected action. There is no generated answer
text and no local override of the model's choice.

Open a game HTML page with the adjacent `arcade/` assets present. The default NInfer endpoint is
`http://127.0.0.1:8010`; change it under **Connection & pace** in Tetris or **Qwen connection** in
Chess and Kitchen Rush. Beacon Runner defaults to the separate vision candidate at port 8014.
Cross-origin NInfer requests require the server's `--cors` option. NInfer's model field is
a request label, not a resident-model switch. Bearer
tokens stay in memory and are excluded from exports. Fonts have local fallbacks; chess rules are
vendored locally. Human and local-reference modes work without a model server.

## Four games, one decision interface

| Game | Legal actions supplied by the client | Local reference | Details |
|---|---|---|---|
| Tetris | Every supported rotation/column straight-drop landing and resulting board metrics | Lexicographic lines, holes, height, bumpiness | [Controls, method and measured comparison](tetris-demo.md) |
| Chess | Every legal move, including castling, en passant and all four promotions | A one-ply material heuristic with a small development tie-break | Controls and limits below |
| Kitchen Rush | Every available chef/order/station job, plus waiting | A greedy scheduler favoring completed dishes, urgency and the selected goal | Paired clocks and rules below |
| Beacon Runner | Fixed eight screen directions plus brake; continuous collision physics | None; Qwen vision or human input | Camera-only control and ablations below |

The instruments separate live inference, human input and local reference play. They display
measured request P50/P95, warm-up separately, token usage and exact wire JSON. Tetris and chess
track decision budgets; Kitchen Rush tracks expired actions and missed service deadlines.
Probabilities describe model preferences among supplied options, not likelihoods of winning.
Reference agreement is a policy comparison, not optimal-play accuracy or a chess-engine rating.

The shared implementation is `arcade/system-one.js`: validated closed-set responses, transport,
full-response timing, cancellation, strict deadline races and measurement summaries.
`arcade/instrument.css` supplies the common console. Each game owns its rules/observation adapter
and controller in `<game>-game.js` and `<game>-ui.js`. The HTML pages own their controls and markup.
Kitchen Rush also owns a DOM-independent `kitchen-match.js` for its paired clock and concurrent
decision lifecycle. The CLI evaluators load these same adapters and transport; they do not reimplement the games.

An additional game should enumerate its complete supported legal action set, attach factual
outcomes, build its question, and apply only the returned action. It should expose its reference
policy separately, name its timing/deadline semantics and export the request/response evidence.
Keep game semantics in its adapter and reusable API behavior in the shared transport.

## Jev vs Qwen

The chess page supports separate **Jev · TypeSafe** and **Qwen · NInfer** players. Both receive the
same position representation, legal actions, selection instructions and 62-option grouping policy.
No move is silently replaced by the local reference.

1. Keep NInfer serving Qwen at `http://127.0.0.1:8010` with `--cors`.
2. Start the API relay from the repository root:

   ```sh
   node tools/arcade/jev-relay.mjs
   ```

3. Open Chess, click **Jev vs Qwen**, and enter your TypeSafe key under **Jev connection**.
   The defaults are relay `http://127.0.0.1:8012` and model `jev-latest`.
4. Press **Run** to watch or **Step** to inspect a move. **Swap sides** resets the starting
   position with Qwen as White and Jev as Black, preserving each provider's connection and key.

TypeSafe's official endpoint is `https://api.typesafe.ai/v1/systemone`, with bearer authentication;
see its [API reference](https://api.typesafe.ai/docs). Its browser preflight rejected the local
file origin (`Origin: null`) during validation, so the relay supplies the local browser boundary.
It serves only API requests, forwards to that fixed upstream, binds to `127.0.0.1`, and logs no
requests or credentials. It accepts local browser origins and cancels upstream work when a client
disconnects. It does not host the game pages.

The page holds separate keys for Jev and Qwen. A Jev key is sent only to the Jev relay, which passes
it to TypeSafe. Every request must supply its own key; the relay never reads an environment key or
grants access based on a browser origin. Use `--port 8013` and update the page's relay URL if 8012 is occupied. Stop the relay
with Ctrl+C when launched in a terminal. Never put keys in URL parameters, PGNs or shared exports.

Each model connection warms once per run before its first live turn; self-play shares that warm-up.
The comparison table separates applied moves, P50/P95 milliseconds and input/output tokens by color;
warm-ups are excluded. Latency samples include completed decisions even if strict mode rejects a
late answer, but exclude incomplete expired decisions. Token totals include every retained completed
stage, including earlier stages of an expired decision. PGNs identify the players and exports retain
the requested model, response model, endpoint and per-stage evidence. Jev's schema permits nonzero
output-token usage; the instrument displays the returned counts.

TypeSafe describes choice probabilities as summing to approximately one. A live Jev 1.13.0 response
contained every legal option but totaled 0.99, which the original exact-sum check rejected. The
shared client accepts totals from 0.98 through 1.02; this is the arcade's explicit tolerance, not
a provider guarantee. It still requires every requested option, finite values in [0, 1], and a
selected choice from the supplied legal actions. Raw values are never normalized, and the chess inspector
calls out approximate totals. Invalid responses pause play and remain inspectable/exportable as
`failure`, separate from completed move timings. Authentication errors identify the key; response
validation errors direct the user to the receipt and retry controls.

The returned `choice` controls the action even when the provider's probability ranking disagrees.
An observed Jev 1.13.0 response selected `D` (Kf3, 12%) while reporting `O` (Rg1) at 13%.
The arcade preserves and applies `D`; it neither substitutes the probability leader nor issues
another inference request. This provider inconsistency appears as a non-blocking `choiceWarning`
in the decision readout, chess stage inspector and export. It does not change the raw response or
relax legal-action and numeric validation. Replaying the supplied position/response through the
controller verifies the exact Kf3 transition and retention of both reported probabilities.

Latency includes the network path to each provider, so this compares the end-to-end experience,
not pure model compute speed. Play both colors before comparing results; a single game does not
establish model strength. The default preset leaves strict deadlines off so a network delay does
not prevent a move. Authentication, network and API failures pause the position without declaring
a winner or switching providers. A real Jev match requires your local key.

Authenticated validation on September 22, 2026 completed 24 legal plies through the local relay:
12 continuing the reported failing position with Jev as White (ending in Qwen checkmate), and
12 from the opening with sides swapped. A second live 0.99-total Jev response was accepted and
played unchanged. Across 12 timed moves each, Jev request P50/P95 was 258.5/302.6 ms and Qwen was
127.1/151.5 ms; each connection's warm-up was excluded. These short HTTP checks establish response
compatibility and legal move application, not relative chess strength or browser behavior.

## Chess controls and behavior

- Choose **You**, **Qwen · NInfer**, **Jev · TypeSafe**, or **Local reference** independently for
  White and Black. The default is human White against Qwen. Choose the same model twice for self-play.
- Click a piece and a highlighted destination. For promotion, explicitly select Queen, Rook,
  Bishop or Knight. Arrow keys navigate the board; Enter selects. **Flip** reverses orientation.
- **Run / Pause / Resume** controls continuous play. **Step** makes one move and opens its
  inspector. **Reset** restores the starting position. Changing session settings resets the run.
- Choose an opening, mate-in-one, material-capture or promotion fixture, or load a custom FEN.
  Move history and **Previous / Next / Live position** support replay without changing the game.
- **Save PGN** downloads the move sequence, initial position and terminal result. **Export** also
  includes recorded decision-stage requests/responses, legal options, choices, timings and reference
  results. Warm-up is retained as a separate duration, without its request/response.
  Runs stop at 300 plies to bound inference and retained evidence.

Played pieces glide to their destination over 220 ms. Captured pieces fade, castling moves the
king and rook together, and promotion reveals the selected piece on arrival. This presentation
runs after the move is committed and measured, uses the configured pause between moves, and
does not enter API or full-decision timings. With a shorter pause, movement finishes before the
next turn begins. Reduced-motion preferences keep immediate placement and last-move highlights.
Pause, Reset, Flip, replay navigation, or hiding the tab settle any active animation immediately;
replay positions themselves change instantly.

Legality and end conditions come from [chess.js 1.4.0](arcade/vendor/README.md). The client attaches
checkmate and material facts to each move. Recapture net subtracts the moved piece's value when its
destination is attacked; it does not search reply sequences and can overstate danger from pinned
pieces or miss compensation. The reference adds small centralization/check tie-breaks. Neither
that reference nor the model is a strong chess engine; the model can still blunder.

At most 62 legal moves fit one System One choice question. A larger position first selects a piece,
then one of that piece's moves. **All legal moves remain available.** This hierarchy changes the
decision distribution: the second request is conditional on the selected piece. The inspector
shows both stages separately; a second-stage probability is not a probability over the whole board.

**API request time** sums the measured durations of all requests for a completed move, including
response-body reads, parsing and validation. **Full decision time** also includes legal-action
enumeration, request construction and stage transitions. Each selected model connection receives
one warm-up before its first live move, excluded from both distributions. Presentation pace and
human turns are untimed.

The chess decision budget covers the complete move decision. Normally a late response is played
and counted as late. **Pause on a missed deadline** aborts and leaves the board unchanged; resume
retries the position. Completed earlier stages remain in the export, while an expired move is
excluded from completed latency samples. Pause/reset and hiding the tab cancel outstanding work;
stale responses cannot move pieces in a new session.

Optional chess URL parameters: `qwen-endpoint`, `qwen-model`, `jev-endpoint`, `jev-model`, `white`
and `black` (`human`, `qwen`, `jev`, `reference`), `match=jev-qwen`,
`scenario` (`opening`, `mate`, `fork`, `promotion`), `fen`, `pace` (0–2000 ms), `budget-ms`
(100–3000 ms), and `strict=1`.

## Kitchen Rush controls and behavior

[Kitchen Rush](kitchen-demo.html) offers **Compare · two kitchens** and **Cooperate · one kitchen**.
Comparison assigns two chefs to each independent kitchen; both kitchens
receive the same seeded orders, recipes, prices, VIP flags and deadlines. The model sees arrived
orders and current jobs, not the future schedule. The full schedule is available in the export
for reproduction. A kitchen holds up to six active tickets; excess arrivals count as missed.

- Press **Local demo**, then **Run**, for two greedy local players without API calls. The default
  is local reference versus Qwen. Choose **You**, **Qwen**, **Jev** or **Local reference** for either
  kitchen. **Jev vs Qwen** selects the two providers and opens Jev's connection controls; use the
  same API-only relay and separate keys described above.
- **Real time** advances both kitchens from one wall clock while requests are in flight. Food
  cooks and burns, and tickets expire, while a model decides. A returned job is rechecked by its
  stable action identity; if it is no longer legal, it is recorded as stale and nothing replaces it.
- **Paused decisions** freezes both clocks while choices are pending, then advances both kitchens
  by 0.5 seconds. Each kitchen with an available job gets one choice per round. **Step** runs one
  round. This separates scheduling choices from request latency; the display pauses between rounds
  for readability. Retrying an interrupted round does not grant an extra decision to its completed side.
- Change the shared goal to **Maximize revenue**, **Reduce waste**, or **Prioritize VIP orders**
  during service. The new instructions affect subsequent decisions; old receipts and in-flight
  requests retain their original goal. Changing players, connection settings, seed, clock mode or
  duration resets the shift. **P**, **Pause**, and hiding the tab stop the simulation and cancel outstanding requests.
- In human mode, choose a job below your kitchen. Walking and work are automatic. Chef movement
  follows the simulation clock, and reduced motion removes spatial transitions. Human and reference
  decisions never enter inference latency samples or invent probabilities.

The pipeline is collect, chop, cook, plate, serve. Salad skips cooking. A job reserves its chef,
order and destination station immediately. Each station supports one job at a time. Loading the
stove frees the chef while food cooks; cooked food must finish plating within five seconds or it
burns. Burned food blocks the stove until cleared. Clearing a still-live ticket allows it to be
started again. Started food discarded by expiry counts as waste; burned food is counted once.
Serving exactly at the service deadline succeeds. The shift ends at its selected duration and
counts unfinished tickets as missed. Rules advance in deterministic 100 ms ticks.

The local reference is a disclosed greedy scheduler, not an optimal planner. Compare revenue,
served and missed orders, waste, and the goal being pursued. A single seed does not establish model
strength. Both live players use the same observation, goal instructions and complete legal job set;
there is no local substitution for a model's choice.

Each provider warms once before either simulation clock starts. P50/P95 include completed valid
responses, including stale jobs, and exclude warm-ups, failed responses and cancellations. Latency
includes network travel, response reads, parsing and validation. The inspector preserves all
candidate jobs and the exact request/response, including approximate probabilities and provider
choice/ranking disagreements. API errors pause the pair and retain the failure receipt. **Export**
saves the schedule, states, events, goal changes, decisions, warm-ups and measurements without keys.

Optional Kitchen Rush URL parameters: `left` and `right` (`human`, `qwen`, `jev`, `reference`),
`layout` (`comparison`, `cooperative`), `mode` (`realtime`, `paused`), `goal` (`revenue`, `waste`, `vip`), `seed` (0–4294967295), and
`match=jev-qwen`. Enter connection details and keys in the page.

### One kitchen, two independent agents

Select **Cooperate · one kitchen**. The first player controls Ada and the second controls Bo.
**Jev + Qwen** assigns the two providers; **Local demo** assigns two local references. Qwen + Qwen,
human + model, and other combinations also work. The shared board, order tickets and team score
appear above the two individual decision readouts. The shared clock advances once for the world.

Each agent receives only its own chef's legal jobs, the current shared kitchen, its partner's
committed work, and the eight latest activity entries. Agents coordinate through visible work
claims and changes to shared state. They may continue a dish the other chef has worked on; this
counts as a handoff when the new job is assigned. Revenue belongs to the team. These observable
handoffs establish task sharing, not evidence that the models negotiated a plan or developed a
persistent relationship. The activity log describes actual simulation events, not generated dialogue.

In real time, both agents can have a request in flight. The first valid answer claims the order
and station atomically. An answer that collides with a teammate's active claim is a **conflict**,
included in the stale count; an expired job is stale without being a resource conflict. Neither
case applies a substitute action. The chef observes the updated world on its next decision.
Resources are reserved by the simulation, so collisions expose coordination failures without
allowing two chefs to cook or collect the same dish simultaneously.

Paused mode gathers both choices against the frozen kitchen before applying either. Conflicting
claims rotate priority between Ada and Bo on successive rounds; provider response order does not
decide the winner. Nonconflicting choices can both apply, then the world advances 0.5 seconds.
If a round is interrupted after one valid reply, that reply remains inspectable as waiting for
the paired round and is reused on Resume. Reset discards it. This completed request remains a
latency sample even before its action is committed; waiting for the teammate is excluded from
the request duration. A provider error pauses the team and preserves its failing receipt.

The version 2 export contains one shared kitchen snapshot, separate `chefs` decision records,
per-chef latency and conflict counts, and the team handoff count. Comparison exports retain two
independent kitchen results. Keys remain outside both formats. The evaluator's `--layout
cooperative --player qwen --partner qwen` exercises two independent actors against the resident
model; `--partner reference` uses the local policy as teammate.

## Reproduce and interpret results

Run the rules, wire-contract and controller checks with Node's built-in test runner:

```sh
node --test tests/test_system_one.mjs tests/test_tetris_demo.mjs tests/test_tetris_ui.mjs tests/test_chess_demo.mjs tests/test_chess_ui.mjs tests/test_jev_relay.mjs tests/test_kitchen_demo.mjs tests/test_kitchen_match.mjs tests/test_kitchen_cooperative.mjs tests/test_kitchen_ui.mjs
```

The controller suites use a minimal DOM port; they do not establish rendering, browser input,
accessibility-tree or browser compatibility behavior.

For the September 22 validation, browser navigation to the local HTML was blocked. The user's
screenshot exposed the player table overflowing the narrow measurements column. The table now
spans the console above the board, nested grids can shrink, and board/measurement columns stack
according to the console's available width. No post-fix desktop/mobile captures or visual approval
are claimed; those browser checks remain open.

For real inference against an already resident server:

```sh
node tools/bench/tetris-eval.mjs --endpoint http://127.0.0.1:8010 --compare --pieces 100 --seed 1 --output tetris-evaluation.json
node tools/bench/chess-eval.mjs --endpoint http://127.0.0.1:8010 --plies 24 --output chess-evaluation.json
node tools/bench/chess-eval.mjs --endpoint http://127.0.0.1:8010 --scenario mate --plies 1
node tools/bench/kitchen-eval.mjs --player qwen --mode paused --seconds 30 --seed 1 --output kitchen-evaluation.json
node tools/bench/kitchen-eval.mjs --player reference --mode paused --seconds 90 --seed 1
node tools/bench/kitchen-eval.mjs --layout cooperative --player qwen --partner qwen --mode realtime --seconds 30 --seed 1 --output kitchen-cooperative.json
```

Set `NINFER_API_KEY` when authentication is required. All harnesses exclude warm-up and do not
render a browser. Tetris and chess make sequential HTTP requests and do not simulate strict
deadline cancellation. The Kitchen Rush harness runs the shipped match controller with selectable
players and layout; paused mode omits presentation delays, while real-time mode uses wall-clock time.

Kitchen Rush's September 22, 2026 HTTP check against the resident `qwen3.8-27b` server on port 8010
completed a 30-second simulated shift, seed 1, revenue goal, paused mode: 41 valid decisions,
request P50 **62.0 ms**, P95 **77.9 ms**, and no stale choices. Qwen served 2 orders for $52 with
3 missed and 3 wasted; the local reference served 3 for $64 with 2 missed and 2 wasted. A separate
30-second real-time run with the same seed and goal finished both kitchens at exactly 30 seconds:
31 valid Qwen decisions, P50 **93.9 ms**, P95 **269.1 ms**, no stale choices, and the same service
and revenue totals. These are integration checks, not optimality or comparative-strength claims. Jev's Kitchen Rush routing
and receipts were checked with controlled responses; no live Jev kitchen run is claimed.
The complete focused suite passes 87 tests, including cooperative ownership, concurrent claims,
latency-independent paused arbitration, handoffs and interrupted paired rounds. Kitchen visuals and animation still need current
browser captures; source and DOM-port checks do not establish on-screen layout or motion quality.

A live cooperative check used two independent Qwen actors against that resident server for
30 seconds of real time, seed 1 and the revenue goal. The team served 2 orders for $52, missed 3
and wasted 3, with 2 handoffs. Ada completed 34 requests (P50/P95 83.5/105.9 ms) and Bo completed
21 (85.8/140.4 ms). Five choices conflicted with the teammate's active claim and were rejected
without substitution. This demonstrates concurrent interaction and observable coordination failures;
it does not establish effective planning. The requests identify each chef separately; the model
and server are shared. No live Jev cooperative check is claimed.

On September 22, 2026, Qwen3.8-27B NVFP4 on an RTX PRO 6000 Blackwell Workstation Edition completed
24 plies of chess self-play with request P50 **121.8 ms** and P95 **158.3 ms**, all selected moves
legal. The separate candidate server used FP8 KV, prefill chunk 2048, concurrency capacity 8 and
4096 context. A mate-in-one fixture and a 73-legal-move fixture both ended in checkmate; the latter
used two visible decision stages. These bounded checks establish the integration, not chess strength.
The [Tetris guide](tetris-demo.md#reproduce-an-evaluation) records its paired prompt comparison.

The backend change also fixes temperature normalization: subtract the maximum finite candidate
score before division by temperature. The previous implementation could turn a decisive low-
temperature choice into a uniform distribution. The candidate passed this live regression and
the serving schema tests. Candidate token strings are validated/resolved once per request and
reused across question branches; no measured backend speedup is claimed for this CPU optimization.

## Beacon Runner: camera-only control

[Open Beacon Runner](beacon-demo.html). Pilot a cyan drone into yellow beacons, avoid solid gray
walls and a moving red hazard, and drag the world while it runs. Keyboard-accessible object
selectors and nudge controls provide the same edits. Human mode supports arrow keys with the arena
focused and nine touch buttons. Qwen mode requires the image-enabled NInfer build, `--vision` and
`--cors`; the page defaults to the separate local candidate at `http://127.0.0.1:8014`.

Every request contains constant instructions, the same nine unfiltered actions, and one PNG in
the NInfer `images` extension. No game coordinates, object lists, collision flags, score, history,
seed or route hints enter the prompt. No local policy chooses or replaces an action. The exact
camera image is visible beside the live arena; export retains the full images and raw responses.
Geometry is rendered by the same pixel renderer in the page and evaluation harness.

Run first performs an excluded warm-up without moving and requires actual `usage.vision_tokens > 0`.
A server that silently ignores images fails visibly. Accepted commands drive 220 ms pulses; one
request is outstanding at a time, and the world continues during inference. Answers beyond the
selected request deadline are discarded. Request time and frame age are separate measurements.
Camera changes cancel the pending request and brake the drone; Resume starts a new observation.
Hiding the tab pauses. Runs last 60 or 90 seconds; Reset starts again. API keys are excluded from
exports. Warm-up and failed/cancelled requests do not enter P50/P95; completed late drops do.

Use **Blackout** to remove all scene pixels, **Freeze** to hold the first observation, or **Delay**
to send frames at least one second old. Freeze and Delay deliberately permit old camera frames;
only request duration determines a late drop. This prevents camera experiments from being replaced
by a client-side stop policy. Delayed mode buffers before its first request.

Reproduce a real-time run using the shipped simulation, renderer and request client:

```sh
node tools/bench/beacon-eval.mjs --endpoint http://127.0.0.1:8014 --seconds 20 --mode live --width 320 --output out/arcade/beacon-live.json
node tools/bench/beacon-eval.mjs --endpoint http://127.0.0.1:8014 --seconds 5 --mode blackout --output out/arcade/beacon-blackout.json
```

The September 22 candidate used Qwen3.8-27B NVFP4, RTX PRO 6000 Blackwell Workstation Edition,
FP8 KV, 8192 context/KV capacity, one active request, prefill chunk 2048, vision enabled and no
speculation. The existing server remained resident: GPU ownership was not exclusive. A 20-second
320 × 240 run with the shipped relative-position instructions collected **1 beacon**, recorded
39 contacts and 74 completed decisions, with 3 late drops at a 500 ms deadline. Request P50/P95
was **70.4 / 445.3 ms**, excluding warm-up. An earlier prompt collected none and drove into the
border; explicitly comparing the ring to the drone improved this run, but obstacle avoidance
remains weak. This is an inspectable visual-control experiment, not a claim of reliable navigation.

Five-second ablation checks used the same shipped prompt and nine actions. Blackout returned
Brake on all 18 decisions (zero beacons, zero contacts). Freeze kept choosing up-right from its
held image, with frame age reaching about six seconds. Delay returned actions from observations
over one second old. These controls change only camera pixels/timing; they do not inject a stop
or navigation action. The harness drains an outstanding response before exiting, so elapsed
simulation time can exceed the requested duration.

The controller/rules checks and live HTTP harness do not establish browser layout or interaction
quality. Current browser captures remain blocked by the session URL policy; visual finish review
requires recapture. The local candidate is separate from the installed server.

## Native System One vision measurements

System One accepts image observations via the NInfer `images` extension. The following probe uses
that endpoint directly; `--route chat` selects the native Chat Completions baseline instead.

```sh
node tools/bench/vision-decision-probe.mjs --endpoint http://127.0.0.1:8014 --route systemone --widths 320,640,960 --samples 12 --output out/arcade/systemone-vision --log PATH_TO_SERVER_REQUESTS_JSONL
```

On the candidate configuration above, all 36 directional scenes and all four target/decoy swaps
were correct. All four blackouts selected wait. Removing the image produced the same up choice
each time (1/4 correct). Actual vision-token counts were returned by the API and confirmed in
server logs. Text and actions remained fixed; only PNG pixels changed.

| PNG size | Correct directions | Vision tokens | Request P50 | Request P95 |
|---|---|---|---|---|
| 320 × 240 | 12 / 12 | 80 | 57.5 ms | 68.3 ms |
| 640 × 480 | 12 / 12 | 300 | 71.1 ms | 659.6 ms |
| 960 × 720 | 12 / 12 | 660 | 123.2 ms | 1344.7 ms |

One warm-up per resolution was excluded. With 12 observations, P95 is the maximum; the shared-GPU
tail is not a production latency guarantee. Multi-question image requests returned the correct
choice for both questions and counted the 80-token image once. Malformed image arrays returned
422, corrupt PNG data returned 400, and text-only requests retained their previous usage shape.

### Pre-extension Chat Completions baseline

Before the image extension, System One formatted structured `state` as text and had no native image-input field.
The engine's multimodal Chat Completions route can already produce a one-token action-candidate
distribution with `logprobs`, `logprob_candidates`, `max_tokens: 1` and thinking disabled. That
route establishes an engine baseline; its timings are not System One vision API measurements.

Reproduce the bounded image-grounding probe against a vision-enabled resident server:

```sh
node tools/bench/vision-decision-probe.mjs --route chat --widths 320,640,960 --samples 12 --output out/arcade/vision-probe --log PATH_TO_SERVER_REQUESTS_JSONL
```

The script generates PNG observations with a cyan triangular drone, a yellow ring target and a
purple decoy. Every native-vision request uses identical instructions and fixed left/right/up/down/
wait choices. Only the pixels change; no coordinates, object state, filenames or expected answers
enter the model prompt. It includes opposite target/decoy placements, blank frames and no-image
controls. The retained report contains the exact returned distributions and joins server vision
tokens, encoder time, preparation, prefill and queue timing using per-request client tags.

On September 22, 2026, the resident Qwen3.8-27B NVFP4 server on an RTX PRO 6000 Blackwell Workstation
Edition used FP8 KV, capacity 8, prefill chunk 2048, vision enabled and MTP enabled. The probe sent
one request at a time; exclusive GPU ownership was not established. Each resolution used 12
changing scenes after one excluded warm-up, measuring through the complete HTTP response.

| PNG size | Correct directions | Vision tokens | Request P50 | Request P95 |
|---|---|---|---|---|
| 320 × 240 | 12 / 12 | 80 | 45.3 ms | 218.8 ms |
| 640 × 480 | 12 / 12 | 300 | 70.9 ms | 271.8 ms |
| 960 × 720 | 12 / 12 | 660 | 124.5 ms | 687.6 ms |

All four target/decoy swaps produced the opposite correct direction. All four blank-frame controls
selected wait. Removing the image while preserving the direction tasks produced the same up choice
every time, correct in 1 of 4 cases. The logs confirmed nonzero vision tokens and encoder work for
image requests. The largest request outliers were reflected in prefill time, with about 1.3–1.4 ms
of ingress queue wait; their underlying cause was not diagnosed. With 12 samples, P95 is the sample
maximum, not a stable tail-latency estimate.

In that pre-extension build, two attempted System One encodings—image content parts inside `state`, and an experimental top-level
`images` field—both returned HTTP 200 but logged zero vision tokens and zero encoder time. Neither
was an advertised image-input contract. The former serialized the image URI as text; the latter
was ignored. That transport gap is now addressed by the explicit image field. The baseline controls
established basic visual grounding, not navigation ability or rich-scene accuracy.
