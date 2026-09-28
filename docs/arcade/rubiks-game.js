// Seeded cube sessions and the System One turn request. The page, the tests and the bench share it.
// The solver measures every position; the model never sees those measurements.
const Rubiks = (() => {
  const {FACES, FACE_NAME, MOVES, SOLVED} = Cube;
  const DISTANCES = [3, 4, 5, 6, 7, 8];
  // Exact search depth and node budget for a live position. Past either, the page shows a proven
  // lower bound instead of a distance.
  const HORIZON = 11, BUDGET = 4e5;

  function mulberry32(seed) {
    let state = seed >>> 0;
    return () => {
      state = state + 0x6D2B79F5 | 0;
      let value = Math.imul(state ^ state >>> 15, 1 | state);
      value = value + Math.imul(value ^ value >>> 7, 61 | value) ^ value;
      return ((value ^ value >>> 14) >>> 0) / 4294967296;
    };
  }
  // Random canonical turns: never the same face twice in a row, and commuting opposite faces
  // only in U-before-D order, so no turn trivially cancels or merges with another.
  function canonicalTurns(count, random) {
    const ids = [];
    let last = -1;
    while (ids.length < count) {
      const turn = MOVES[Math.floor(random() * 18)];
      if (turn.faceIndex === last || turn.faceIndex === last - 3) continue;
      ids.push(turn.id);
      last = turn.faceIndex;
    }
    return ids;
  }
  // `distance` 3..8 draws canonical sequences until the solver proves one is exactly that many
  // turns from solved, so the scramble length is also par. 'random' draws a uniformly random
  // state and scrambles with the inverse of a two-phase solution; par is then unproven.
  function scramble(distance, seed) {
    const random = mulberry32(seed + Math.imul(distance === 'random' ? 0 : distance, 0x9E3779B1));
    if (distance === 'random') {
      const solution = CubeSolver.twoPhase(CubeSolver.randomCube(random));
      return {moves: Cube.invert(solution.moves), par: null};
    }
    if (!DISTANCES.includes(distance)) throw new Error('Scramble distance must be 3 to 8 turns, or random.');
    for (;;) {
      const moves = canonicalTurns(distance, random);
      if (CubeSolver.optimal(Cube.cubieOf(Cube.applyAll(SOLVED, moves)), {maxDepth: distance}).distance === distance) return {moves, par: distance};
    }
  }
  function describe(option) {
    const turn = option.quarters === 2 ? 'half turn' : option.quarters === 1 ? 'clockwise quarter turn' : 'counterclockwise quarter turn';
    return option.id + ': ' + turn + ' of the ' + FACE_NAME[option.face] + ' face, as you look at it. Misplaced stickers afterward: ' + option.misplaced + '.';
  }
  // Exact distance within the horizon, else the proven lower bound. A quick measure that runs out
  // of budget stays pending; settle() finishes it with the full budget when the page is idle.
  const QUICK = 3e4;
  function measure(colors, budget = BUDGET) {
    const found = CubeSolver.optimal(Cube.cubieOf(colors), {maxDepth: HORIZON, budget});
    return {distance: found.distance, lower: found.lower, line: found.moves, pending: !!found.exhausted && budget < BUDGET};
  }

  class Game {
    constructor({distance = 5, seed = 1} = {}) {
      this.setting = distance;
      this.seed = seed >>> 0;
      const drawn = scramble(distance, this.seed);
      this.scramble = drawn.moves;
      this.par = drawn.par;
      this.colors = Cube.applyAll(SOLVED, this.scramble);
      this.history = [];
      this.visited = new Set([Cube.faceletString(this.colors)]);
      // One entry per committed turn, with exact distances when the solver proved them.
      this.trace = [];
      this.position = drawn.par == null ? measure(this.colors, QUICK) : {distance: drawn.par, lower: drawn.par, line: null};
    }
    get distance() { return this.position.distance; }
    solved() { return Cube.solved(this.colors); }
    misplaced() { return Cube.misplaced(this.colors); }
    face(name) { return Cube.faceText(this.colors, name); }
    cube() { return Cube.cubieOf(this.colors); }
    options() {
      if (this.solved()) return [];
      return MOVES.map(turn => {
        const next = Cube.apply(this.colors, turn);
        return {id: turn.id, face: turn.face, quarters: turn.quarters, misplaced: Cube.misplaced(next), revisits: this.visited.has(Cube.faceletString(next))};
      });
    }
    // Solver distance after each turn. It stays local: requestFor never sends it.
    analysis() {
      this.settle();
      if (this.solved() || this.distance == null) return null;
      return CubeSolver.analyze(this.cube(), this.distance, {budget: BUDGET});
    }
    // `known` is a distance the caller already proved. A quick commit spends at most a few
    // milliseconds measuring, so fast input never waits for a long search.
    commit(id, {known = null, quick = false} = {}) {
      const turn = Cube.move(id);
      if (this.solved()) throw new Error('The cube is already solved.');
      const before = this.position;
      this.colors = Cube.apply(this.colors, turn);
      this.history.push(turn.id);
      this.visited.add(Cube.faceletString(this.colors));
      const line = before.line?.[0] === turn.id ? before.line.slice(1) : null;
      this.position = this.solved() ? {distance: 0, lower: 0, line: []}
        : known != null ? {distance: known, lower: known, line}
        : line ? {distance: before.distance - 1, lower: before.distance - 1, line}
        : measure(this.colors, quick ? QUICK : BUDGET);
      this.trace.push({turn: turn.id, before: before.distance, lowerBefore: before.lower, after: this.position.distance, lower: this.position.lower, optimal: null});
      this.#grade(this.trace.length - 1);
      return turn;
    }
    // Measures a position committed without measurement; true when that changed the trace.
    settle() {
      if (!this.position.pending) return false;
      this.position = measure(this.colors);
      const last = this.trace.at(-1);
      if (last) { last.after = this.position.distance; last.lower = this.position.lower; this.#grade(this.trace.length - 1); }
      return true;
    }
    #grade(index) {
      const entry = this.trace[index];
      entry.optimal = entry.before != null && entry.after != null ? entry.after === entry.before - 1 : null;
    }
    // An optimal line inside the horizon, otherwise a two-phase line of about twenty turns.
    solution() {
      this.settle();
      if (this.solved()) return {moves: [], optimal: true};
      if (this.position.line) return {moves: this.position.line.slice(), optimal: true};
      if (this.distance != null) {
        const found = CubeSolver.optimal(this.cube(), {exactly: this.distance, budget: BUDGET * 4});
        if (found.moves) { this.position.line = found.moves; return {moves: found.moves.slice(), optimal: true}; }
      }
      return {moves: CubeSolver.twoPhase(this.cube()).moves, optimal: false};
    }
    state() {
      const faces = FACES.map(face => FACE_NAME[face] + ' ' + this.face(face)).join('\n');
      return 'Rubik cube, 3x3. ' + (this.solved() ? 'Solved.' : 'Unsolved.') +
        '\nRead each face while looking at it, left to right, top to bottom. The up face has its back edge on the first row. The down face has its front edge on the first row.' +
        '\nCenters stay put and name the face: up white, right red, front green, down yellow, left orange, back blue.' +
        '\nLetters: W white, R red, G green, Y yellow, O orange, B blue.' +
        '\n' + faces +
        '\nMisplaced stickers, centers excluded: ' + this.misplaced() + ' of 48.' +
        '\nTurns already made in this solve: ' + (this.history.join(' ') || 'none') + '.';
    }
  }

  const codeOptions = options => options.map((option, index) => ({...option, code: SystemOne.CODES[index]}));
  const INSTRUCTIONS = 'Choose the face turn that leaves the cube the fewest turns from solved. A face letter alone is a clockwise quarter turn of that face as you look at it; a prime is counterclockwise; a 2 is a half turn. Each option states how many stickers would be misplaced afterward. Return the option token.';
  function candidates(game, {avoidRepeats = true} = {}) {
    const all = codeOptions(game.options());
    return {options: avoidRepeats ? all.filter(option => !option.revisits) : all,
      excluded: avoidRepeats ? all.filter(option => option.revisits).map(option => option.id) : []};
  }
  function requestFor(game, model, {avoidRepeats = true} = {}) {
    const {options, excluded} = candidates(game, {avoidRepeats});
    if (!options.length) throw new Error('No unvisited next position remains. Reset or allow repeated positions to continue.');
    const criteria = Object.fromEntries(options.map(option => [option.code, describe(option)]));
    const instructions = INSTRUCTIONS + (avoidRepeats ? ' Turns returning to an already visited position are omitted; choose among the offered turns.' : '');
    return {options, excluded, body: {model, state: game.state(), questions: {move: {type: 'choice', instructions, criteria}}}};
  }
  async function choose(game, connection, {signal} = {}) {
    const {options, excluded, body} = requestFor(game, connection.model, connection);
    const result = await SystemOne.decide(connection.endpoint, body, options, {signal, apiKey: connection.apiKey || ''});
    return {option: result.option, ms: result.ms, request: body, response: result.body, options, probabilities: result.probabilities,
      usage: result.usage, choiceWarning: result.choiceWarning, excluded};
  }

  return {DISTANCES, mulberry32, canonicalTurns, Game, candidates, requestFor, choose};
})();
