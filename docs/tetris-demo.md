# Tetris decision arcade

[Open Tetris](tetris-demo.html) or [Chess](chess-demo.html) in the [decision arcade](decision-arcade.md).
Tetris is a static HTML client with adjacent `arcade/` scripts and styles for
`POST /v1/systemone`, using the same TypeSafe-compatible `choice` contract as any other application.
The API base URL defaults to `http://127.0.0.1:8010`; set it under **Connection & pace**.
Cross-origin use requires the server's existing `--cors` option. For an authenticated server, enter
its bearer token in the page. Tokens are held only in memory and excluded from exports.
The model field is a request label; inference uses the server's resident artifact.

## Play and inspect

- **System One** enumerates all legal straight-drop landings, sends the resulting board metrics,
  and applies the model's selected option. There is no local heuristic override.
- **You** provides manual placement: left/right changes the column, up rotates, and Space places
  the piece. On-screen controls provide the same actions. This mode is untimed.
- **Reference** runs the stated policy locally. It makes no inference request and displays no
  synthetic model probabilities or API latency.
- **Run / Pause / Resume** controls continuous play. **Step** executes one piece and opens its
  inspector. **Reset** replays the seed. Changing session settings starts a fresh run.
- **Piece speed** controls the live or reference choice's steer-and-settle animation from 0.5×
  to 3× and can change during a run. At 1× the movement takes about 620 ms. Animation and the pause between
  pieces are presentation time, separate from request latency and the gravity deadline. Pause or
  hiding the tab settles an in-progress placement; Reset starts a fresh run. Reduced-motion
  preferences skip the movement. Manual play already previews the selected landing and locks it
  immediately when you press Place.
- **Export run** downloads observations, legal actions, selected actions, requests, responses,
  timings and outcomes as JSON. A failed live request opens the inspector with its request and
  response, leaves the board unchanged, and remains exportable even before the first completed
  decision. Retrying the piece clears that failure after a valid choice. Failed requests do not
  enter latency percentiles. Runs stop after 500 pieces to bound inference and retained evidence.

The seven-bag sequence is seeded, including seed zero. Choose an empty well, a four-line well
starting with an I piece, or a staircase. The four-line fixture moves the first I to the start of
the first bag; the bag still contains all seven pieces.

Watch mode records a missed budget and still plays the response. **Stop on a missed deadline**
enables the challenge: expiry cancels the request, locks the spawn-column landing and ends the run.
Reset and pause cancel in-flight work; late responses cannot change the next session. Hiding the
tab pauses play because a background tab is not a valid real-time browser measurement.

This is a placement challenge, not guideline-complete Tetris. Legal actions are rotations and
columns reachable by a top-down drop, without wall kicks, tucks, hold, or movement-path search.
The falling piece visualizes the decision budget. The later move toward the selected landing is
also a visual explanation of the choice, not proof that the piece could follow that physical path.

Optional URL parameters are `endpoint`, `model`, `seed`, `row` (4–40 milliseconds), `pace`
(0–800 milliseconds), `player` (`live`, `human`, `reference`), `scenario`
(`empty`, `trench`, `stairs`), `motion` (0.5–3×), and `strict=1`.

## What the instrument measures

The policy is lexicographic: maximize lines cleared, minimize buried holes, minimize maximum
column height, then minimize the sum of neighboring column-height differences. Equivalent outcomes
count as agreement. This is a transparent local reference, not an optimal Tetris oracle.

The browser computes candidate outcomes. The model chooses among those outcomes in one categorical
logit readout. The compact prompt preserves named outcome fields but omits the redundant board
drawing and placement coordinates. Every legal landing remains available; options are not ranked,
filtered or annotated with the reference's preferred answer.

- **Request latency** starts before fetch and ends after the full body has been read, parsed and
  validated. P50 and P95 use nearest-rank percentiles of completed live decisions in the run.
- **Warm-up** uses a real request with the first board's candidate set. It is shown separately and
  excluded from latency statistics.
- **Deadline** covers enumeration and request preparation through acceptance of a validated answer.
  The budget is the spawn-column drop distance multiplied by milliseconds per row, with a one-row
  minimum lock interval. Strictly timed-out requests have no completed latency sample.
- **Presentation pace** happens between decisions and does not enter request latency.
- **Probability** is the model's raw preference over the supplied candidates. Accepted rounded
  totals can range from 98% to 102%; the page shows an off-100% total without changing the values.
  It is not a calibrated likelihood of winning. The inspector exposes every candidate and the
  request/response JSON.
- **Tokens** come from response usage. NInfer reports zero output tokens for its logit readout;
  another compatible provider may report nonzero output usage. The instrument preserves that count.

## Reproduce an evaluation

Node's built-in test runner checks the actual game/transport code and the UI controller:

```sh
node --test tests/test_tetris_demo.mjs tests/test_tetris_ui.mjs
```

The controller suite uses a minimal DOM port. It verifies controls, cancellation, retry, deadlines
and exports; it does not provide rendering, accessibility-tree or browser compatibility coverage.
Browser capture and visual approval remain open as recorded in the
[arcade validation notes](decision-arcade.md#reproduce-and-interpret-results).

For a resident model, run:

```sh
node tools/bench/tetris-eval.mjs --endpoint http://127.0.0.1:8010 --model qwen3.8-27b --compare --pieces 100 --seed 1 --output tetris-evaluation.json
```

Use `NINFER_API_KEY` for authentication. The comparison alternates request order between the
original and compact encodings. Each encoding owns its board but receives the same seeded piece
sequence. Warm-ups are excluded. This direct HTTP harness measures completed requests; it does not
render a browser or cancel late requests.

On September 22, 2026, one such local run against Qwen3.8-27B NVFP4, on an NVIDIA RTX PRO 6000
Blackwell Workstation Edition, produced:

| 100-piece run, seed 1 | Original prompt | Compact named fields |
|---|---:|---:|
| Request P50 | 84.6 ms | 64.5 ms |
| Request P95 | 121.9 ms | 95.1 ms |
| Mean input tokens | 801.5 | 548.3 |
| Policy agreement | 83/100 | 87/100 |
| Lines cleared | 34 | 37 |
| Completed requests exceeding 12 ms/row budget | 8 | 0 |

These are observed results for this run and hardware, not a latency guarantee or a general
model-accuracy claim. The running server used FP8 KV, prefill chunk 2048 and concurrency capacity 8;
the evaluator submitted requests sequentially. The comparison preceded the separate System One
temperature-normalization fix and candidate-token resolution change. No backend speedup is
attributed to that fix.
