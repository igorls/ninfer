import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFile} from 'node:fs/promises';
import {loadRubiks} from '../tools/bench/rubiks-eval.mjs';

const R = await loadRubiks();
const {Cube, CubeSolver, RubiksSession} = R;
const state = ids => Cube.faceletString(Cube.applyAll(Cube.SOLVED, ids ? ids.split(' ') : []));

// Published facelet strings for standard Singmaster turns, in Kociemba's U R F D L B order.
test('face turns follow standard notation: clockwise as you look at the face', () => {
  assert.equal(state('U'), 'UUUUUUUUUBBBRRRRRRRRRFFFFFFDDDDDDDDDFFFLLLLLLLLLBBBBBB');
  assert.equal(state('R'), 'UUFUUFUUFRRRRRRRRRFFDFFDFFDDDBDDBDDBLLLLLLLLLUBBUBBUBB');
  assert.equal(state('F'), 'UUUUUULLLURRURRURRFFFFFFFFFRRRDDDDDDLLDLLDLLDBBBBBBBBB');
  assert.equal(state("U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2"), 'UBULURUFURURFRBRDRFUFLFRFDFDFDLDRDBDLULBLFLDLBUBRBLBDB');
  assert.equal(state('U2 D2 F2 B2 L2 R2'), 'UDUDUDUDURLRLRLRLRFBFBFBFBFDUDUDUDUDLRLRLRLRLBFBFBFBFB');
});

test('cubie tables read back from the facelets match the reference two-phase tables', () => {
  const reference = {
    U: [[3, 0, 1, 2, 4, 5, 6, 7], [0, 0, 0, 0, 0, 0, 0, 0], [3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    R: [[4, 1, 2, 0, 7, 5, 6, 3], [2, 0, 0, 1, 1, 0, 0, 2], [8, 1, 2, 3, 11, 5, 6, 7, 4, 9, 10, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    F: [[1, 5, 2, 3, 0, 4, 6, 7], [1, 2, 0, 0, 2, 1, 0, 0], [0, 9, 2, 3, 4, 8, 6, 7, 1, 5, 10, 11], [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0]],
    D: [[0, 1, 2, 3, 5, 6, 7, 4], [0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 5, 6, 7, 4, 8, 9, 10, 11], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    L: [[0, 2, 6, 3, 4, 1, 5, 7], [0, 1, 2, 0, 0, 2, 1, 0], [0, 1, 10, 3, 4, 5, 9, 7, 8, 2, 6, 11], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    B: [[0, 1, 3, 7, 4, 5, 2, 6], [0, 0, 1, 2, 0, 0, 2, 1], [0, 1, 2, 11, 4, 5, 6, 10, 8, 9, 3, 7], [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1]]
  };
  for (const [face, [cp, co, ep, eo]] of Object.entries(reference)) {
    const cube = Cube.MOVE_CUBIES[Cube.move(face).index];
    assert.deepEqual([Array.from(cube.cp), Array.from(cube.co), Array.from(cube.ep), Array.from(cube.eo)], [cp, co, ep, eo], face);
  }
});

test('move orders: every face cycles in four, R U in 105 and R U\' in 63', () => {
  const order = ids => { let colors = Cube.SOLVED, n = 0; do { colors = Cube.applyAll(colors, ids.split(' ')); n++; } while (!Cube.solved(colors)); return n; };
  for (const face of Cube.FACES) { assert.equal(order(face), 4); assert.equal(order(face + '2'), 2); assert.equal(order(face + ' ' + face + "'"), 1); }
  assert.equal(order('R U'), 105);
  assert.equal(order("R U'"), 63);
});

// The renderer turns a layer with THREE's right-handed axis-angle rotation. Rotating every sticker
// that way must land it exactly where the facelet permutation says, or the cube would snap.
test('the rendered layer rotation lands every sticker where the model moves it', async () => {
  const three = await readFile(new URL('../docs/arcade/vendor/three.min.js', import.meta.url), 'utf8');
  const {THREE} = vm.runInNewContext(three + ';({THREE})', {console});
  const key = (p, n) => p.map(Math.round).join() + '|' + n.map(Math.round).join();
  const index = new Map(Cube.FACELETS.map(f => [key(f.pos, f.n), f.index]));
  for (const turn of Cube.MOVES) {
    const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(...turn.axis), turn.angle);
    const onLayer = f => f.pos[0] * turn.axis[0] + f.pos[1] * turn.axis[1] + f.pos[2] * turn.axis[2] === 1;
    for (const f of Cube.FACELETS) {
      const target = onLayer(f)
        ? index.get(key(new THREE.Vector3(...f.pos).applyQuaternion(q).toArray(), new THREE.Vector3(...f.n).applyQuaternion(q).toArray()))
        : f.index;
      assert.equal(turn.perm[target], f.index, turn.id + ' facelet ' + f.index);
    }
  }
});

// Independent oracle: breadth-first distances over facelet strings, joined in the middle.
function oracle() {
  const key = colors => String.fromCharCode(...colors), depth = new Map([[key(Cube.SOLVED), 0]]);
  let frontier = [Cube.SOLVED];
  for (let d = 1; d <= 3; d++) {
    const next = [];
    for (const colors of frontier) for (const turn of Cube.MOVES) { const n = Cube.apply(colors, turn), k = key(n); if (!depth.has(k)) { depth.set(k, d); next.push(n); } }
    frontier = next;
  }
  return colors => {
    let best = Infinity;
    const walk = (c, g) => { const k = depth.get(key(c)); if (k != null) best = Math.min(best, g + k); if (g < 3 && g + 1 < best) for (const t of Cube.MOVES) walk(Cube.apply(c, t), g + 1); };
    walk(colors, 0);
    return best;
  };
}

test('exact search agrees with a breadth-first oracle and its lines solve', () => {
  const distance = oracle(), random = R.mulberry32(7);
  for (let length = 1; length <= 6; length++) for (let trial = 0; trial < 5; trial++) {
    const colors = Cube.applyAll(Cube.SOLVED, R.canonicalTurns(length, random)), found = CubeSolver.optimal(Cube.cubieOf(colors));
    assert.equal(found.distance, distance(colors));
    assert.equal(Cube.solved(Cube.applyAll(colors, found.moves)), true);
  }
});

test('turn analysis grades each neighbor by its exact distance', () => {
  const game = new R.Game({distance: 6, seed: 11}), rows = game.analysis();
  assert.equal(rows.length, 18);
  for (const [i, row] of rows.entries()) {
    const next = Cube.apply(game.colors, Cube.MOVES[i]);
    assert.equal(row.distance, CubeSolver.optimal(Cube.cubieOf(next)).distance, row.id);
    assert.equal(row.delta, row.distance - 6);
  }
  assert.ok(rows.some(row => row.delta === -1));
});

test('two-phase solves random states in about twenty turns', () => {
  const random = R.mulberry32(3);
  for (let i = 0; i < 12; i++) {
    const cube = CubeSolver.randomCube(random), found = CubeSolver.twoPhase(cube);
    assert.equal(Cube.solved(Cube.applyAll(Cube.colorsOf(cube), found.moves)), true);
    assert.ok(found.length <= 24, 'length ' + found.length);
  }
});

test('scrambles are seeded, proven exactly par turns deep, and random states are reproducible', () => {
  for (const distance of R.DISTANCES) for (const seed of [1, 2, 3]) {
    const game = new R.Game({distance, seed});
    assert.equal(game.scramble.length, distance);
    assert.equal(game.par, distance);
    assert.equal(CubeSolver.optimal(game.cube()).distance, distance);
    assert.deepEqual([...new R.Game({distance, seed}).scramble], [...game.scramble]);
  }
  const a = new R.Game({distance: 'random', seed: 5}), b = new R.Game({distance: 'random', seed: 5});
  assert.deepEqual([...a.scramble], [...b.scramble]);
  assert.equal(a.par, null);
  assert.ok(a.scramble.length >= 16);
});

test('the choice request lists all eighteen turns and never leaks the solver or the scramble', () => {
  const game = new R.Game({distance: 4, seed: 3});
  const {options, body} = R.requestFor(game, 'qwen3.8-27b'), wire = JSON.stringify(body);
  assert.equal(options.length, 18);
  assert.equal(new Set(options.map(option => option.code)).size, 18);
  assert.deepEqual([...Object.keys(body.questions.move.criteria)], [...options.map(option => option.code)]);
  assert.equal(wire.includes(game.scramble.join(' ')), false);
  for (const word of ['distance', 'par ', 'optimal', 'delta']) assert.equal(wire.toLowerCase().includes(word), false, word);
  assert.match(body.questions.move.criteria[options.find(o => o.id === "R'").code], /counterclockwise quarter turn of the right face/);
});

// Session: a fake view and a controllable clock stand in for the page.
function harness({answer = keys => keys[0], fail = null, hold = false, finish = () => 'stop'} = {}) {
  let now = 0, release = null;
  const calls = [], updates = [];
  const fetch = async (url, init) => {
    const request = JSON.parse(init.body);
    calls.push(request);
    if (hold) await new Promise((resolve, reject) => { release = resolve; init.signal?.addEventListener('abort', () => reject(Object.assign(new Error('aborted'), {name: 'AbortError'}))); });
    now += 50;
    if (fail) return {ok: false, status: 500, text: async () => JSON.stringify({error: {message: fail}})};
    const reasoning = !!request.messages;
    const keys = reasoning ? request.response_format.json_schema.schema.properties.move.enum : Object.keys(request.questions.move.criteria), choice = answer(keys, request);
    if (reasoning) return {ok: true, status: 200, text: async () => JSON.stringify({choices: [{finish_reason: finish(calls.length), message: {content: JSON.stringify({move: choice}), reasoning_content: 'A short proposed plan.'}}],
      usage: {prompt_tokens: 900, completion_tokens: 100, completion_tokens_details: {reasoning_tokens: 92}}})};
    return {ok: true, status: 200, text: async () => JSON.stringify({answers: {move: {type: 'choice', choice, confidence: 1, probabilities: Object.fromEntries(keys.map(k => [k, k === choice ? 1 : 0]))}}, usage: {input_tokens: 900, output_tokens: 0}})};
  };
  return {fetch, calls, updates, tick: ms => { now += ms; }, now: () => now, release: () => release?.()};
}
async function sessionWith(options = {}) {
  const h = harness(options), S = await loadRubiks({fetch: h.fetch, performance: {now: h.now}});
  const presented = [];
  // Model and search turns block on their animation; a person's animation runs alongside their clock.
  const view = {scramble: () => Promise.resolve(), present: (turn, info) => { presented.push(turn.id); if (info.source !== 'human') h.tick(1000); return Promise.resolve(); }, settle() {},
    update: event => h.updates.push(event), connection: () => ({endpoint: 'http://test', model: 'resident', apiKey: 'secret-key'})};
  const session = new S.RubiksSession.Session({view, now: h.now, defer: fn => fn()});
  return {S, h, session, presented};
}

test('a live solve applies each returned turn, keeps warm-up and animation off the clock, and grades turns', async () => {
  // Answer with the solver's own next move so the run finishes; grading must then be all optimal.
  let game;
  const {session, h, presented} = await sessionWith({answer: () => R.SystemOne.CODES[Cube.MOVES.findIndex(t => t.id === game.solution().moves[0])]});
  await session.reset({distance: 4, seed: 2, player: 'qwen'});
  game = session.game;
  await session.start();
  assert.equal(session.mode, 'SOLVED');
  assert.equal(h.calls.length, 1 + 4);
  assert.deepEqual(presented, [...game.history]);
  assert.equal(session.clock.read(), 4 * 50);
  assert.equal(session.warmup.ms, 50);
  const summary = session.summary();
  assert.equal(summary.optimal, 4);
  assert.equal(summary.graded, 4);
  assert.equal(summary.optimalMass, 1);
  const exported = JSON.stringify(session.exportData());
  assert.equal(exported.includes('secret-key'), false);
  assert.equal(session.exportData().decisions.length, 4);
});

test('a failed request neither turns the cube nor adds a sample, and export keeps the failure', async () => {
  const {session, h} = await sessionWith({fail: 'engine offline'});
  await session.reset({distance: 3, seed: 1, player: 'qwen'});
  const before = Cube.faceletString(session.game.colors);
  await session.start();
  assert.equal(session.mode, 'REQUEST FAILED');
  assert.equal(h.calls.length, 1);
  assert.equal(Cube.faceletString(session.game.colors), before);
  assert.equal(session.records.length, 0);
  assert.equal(session.clock.read(), 0);
  assert.match(session.exportData().failure.message, /engine offline/);
  assert.equal(session.exportData().failure.warmup, true);
});

test('pausing cancels the pending request and takes its time back off the clock', async () => {
  const {session, h} = await sessionWith({hold: true});
  await session.reset({distance: 3, seed: 1, player: 'qwen'});
  session.warmup = {ms: 1};
  const run = session.start();
  await new Promise(resolve => setImmediate(resolve));
  h.tick(400);
  assert.equal(session.clock.read(), 400);
  session.pause();
  await run;
  assert.equal(session.mode, 'PAUSED');
  assert.equal(session.clock.read(), 0);
  assert.equal(session.game.history.length, 0);
});

test('local search solves at par on its search clock, and a person can take over a paused solve', async () => {
  const {session} = await sessionWith();
  await session.reset({distance: 7, seed: 4, player: 'reference'});
  await session.start(true);
  assert.equal(session.mode, 'PAUSED');
  assert.equal(session.game.history.length, 1);
  session.setPlayer('human');
  const line = session.game.solution().moves;
  for (const id of line) assert.equal(session.humanTurn(id), true);
  assert.equal(session.mode, 'SOLVED');
  assert.equal(session.game.history.length, 7);
  assert.equal(session.summary().optimal, 7);
  assert.equal(session.humanTurn('U'), false);
});

test('a person\'s clock starts at the first turn and the model turn limit does not apply to them', async () => {
  const {session, h} = await sessionWith();
  await session.reset({distance: 3, seed: 9, player: 'human'});
  h.tick(5000);
  assert.equal(session.clock.read(), 0);
  session.humanTurn('U');
  h.tick(700);
  assert.equal(session.clock.read(), 700);
  assert.equal(session.game.trace[0].optimal != null, true);
  for (let i = 0; i < 20; i++) session.humanTurn(i % 2 ? 'R' : "R'");
  assert.equal(session.mode, 'SOLVING');
});

test('reasoning receives the same observation and options, without reference-solver information', () => {
  const game = new R.Game({distance: 5, seed: 1});
  game.commit("U'");
  game.analysis();
  game.solution();
  const decision = R.requestFor(game, 'resident').body;
  const {body, options} = R.RubiksReasoning.requestFor(game, 'resident', {maxTokens: 8192});
  assert.ok(body.messages[1].content.startsWith(decision.state + '\n\n'));
  for (const [code, description] of Object.entries(decision.questions.move.criteria)) assert.ok(body.messages[1].content.includes(code + ': ' + description));
  assert.equal(JSON.stringify(body).includes(game.scramble.join(' ')), false);
  assert.deepEqual([...body.response_format.json_schema.schema.properties.move.enum], [...options.map(o => o.code)]);
  assert.equal(body.enable_thinking, true);
  assert.equal(body.max_tokens, 8192);
});

test('reasoning uses the shared grading and clocks without fabricating choice probabilities', async () => {
  let game;
  const {session, h, presented} = await sessionWith({answer: () => R.SystemOne.CODES[Cube.MOVES.findIndex(t => t.id === game.solution().moves[0])]});
  await session.reset({distance: 3, seed: 1, player: 'reasoning'});
  game = session.game;
  await session.start();
  assert.equal(session.mode, 'SOLVED');
  assert.equal(h.calls.length, 4);
  assert.deepEqual(presented, [...game.history]);
  assert.equal(session.clock.read(), 150);
  assert.equal(session.warmup.ms, 50);
  assert.equal(session.summary().optimal, 3);
  assert.equal(session.summary().optimalMass, null);
  assert.equal(session.latest.probabilities, null);
  assert.equal(session.latest.usage.output_tokens, 100);
  assert.equal(session.latest.usage.reasoning_tokens, 92);
  assert.equal(session.latest.method, 'reasoning');
  assert.equal(session.summary().allRequestMs, 200);
  assert.equal(session.summary().allOutputTokens, 400);
  assert.equal(JSON.stringify(session.exportData()).includes('secret-key'), false);
});

test('a truncated reasoning answer never moves the cube and retains its failed request cost', async () => {
  const {session} = await sessionWith({finish: call => call === 1 ? 'stop' : 'length'});
  await session.reset({distance: 5, seed: 1, player: 'reasoning'});
  const before = Cube.faceletString(session.game.colors);
  await session.start();
  assert.equal(session.mode, 'REQUEST FAILED');
  assert.match(session.failure.message, /output-token limit/);
  assert.equal(Cube.faceletString(session.game.colors), before);
  assert.equal(session.records.length, 0);
  assert.equal(session.summary().totalRequestMs, 50);
  assert.equal(session.summary().allRequestMs, 100);
  assert.equal(session.summary().allOutputTokens, 200);
  assert.equal(session.failure.receipt.response.usage.completion_tokens, 100);
});

test('pausing reasoning cancels transport and a later response cannot turn the cube', async () => {
  const {session, h} = await sessionWith({hold: true});
  await session.reset({distance: 5, seed: 1, player: 'reasoning'});
  session.warmup = {ms: 1};
  const run = session.start();
  await new Promise(resolve => setImmediate(resolve));
  h.tick(700);
  session.pause();
  h.release();
  await run;
  assert.equal(session.game.history.length, 0);
  assert.equal(session.records.length, 0);
  assert.equal(session.clock.read(), 0);
  assert.equal(session.mode, 'PAUSED');
});

test('model choices exclude immediate reversals and longer returns to a visited cube', () => {
  const game = new R.Game({distance: 5, seed: 1});
  game.commit('R');
  let request = R.requestFor(game, 'resident');
  assert.equal(request.options.some(o => o.id === "R'"), false, 'the inverse must not return to the starting cube');
  assert.ok(request.options.some(o => o.id === 'R'), 'a further quarter turn may reach a new state');
  game.commit('L');
  game.commit("R'");
  request = R.requestFor(game, 'resident');
  assert.equal(request.options.some(o => o.id === "L'"), false, 'commuting moves can close a longer cycle');
  const raw = R.requestFor(game, 'resident', {avoidRepeats: false});
  assert.equal(raw.options.length, 18, 'raw comparison still offers every legal turn');
  assert.ok(raw.options.some(o => o.id === "L'"));
  for (const option of request.options) assert.equal(option.code, raw.options.find(o => o.id === option.id).code);
});

test('a model run cannot ping-pong even when it always prefers the first available option', async () => {
  const {session} = await sessionWith();
  await session.reset({distance: 5, seed: 1, player: 'qwen'});
  let colors = session.game.colors.slice();
  const visited = new Set([Cube.faceletString(colors)]);
  await session.start();
  for (const id of session.game.history) {
    colors = Cube.apply(colors, Cube.move(id));
    const key = Cube.faceletString(colors);
    assert.equal(visited.has(key), false, 'revisited a cube after ' + id);
    visited.add(key);
  }
});

test('human reversals remain legal and a model stops if every neighboring position was visited', async () => {
  const {session, h} = await sessionWith();
  await session.reset({distance: 5, seed: 1, player: 'human'});
  const initial = Cube.faceletString(session.game.colors);
  for (const turn of Cube.MOVES) {
    assert.equal(session.humanTurn(turn.id), true);
    assert.equal(session.humanTurn(Cube.inverse(turn.id)), true);
  }
  assert.equal(Cube.faceletString(session.game.colors), initial);
  session.setPlayer('qwen');
  await session.start();
  assert.equal(session.mode, 'NO NEW TURN');
  assert.equal(h.calls.length, 0);
  assert.equal(session.running, false);
  session.view.connection = () => ({endpoint: 'http://test', model: 'resident', avoidRepeats: false});
  assert.equal(session.hasModelTurn, true);
  await session.reset({distance: 5, seed: 1, player: 'qwen'});
  assert.equal(R.requestFor(session.game, 'resident').options.length, 18, 'reset forgets visited states');
});
