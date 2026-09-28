// DOM-free run controller for the cube page: players, the solver clock, decisions, records and
// exports. The page supplies a view that animates turns and repaints; tests supply a fake view.
// The solver clock runs only while the solver is deciding: request time for Qwen, search time for
// local search, and wall time from the first turn for a person. Warm-up, instrument analysis and
// turn animation stay off it, so turn speed never changes a result.
const RubiksSession = (() => {
  class Clock {
    constructor(now) { this.now = now; this.elapsed = 0; this.since = null; }
    get running() { return this.since != null; }
    start() { if (this.since == null) this.since = this.now(); return this.elapsed; }
    stop() { if (this.since != null) { this.elapsed += this.now() - this.since; this.since = null; } }
    // A cancelled or discarded decision takes its time back off the clock.
    revert(checkpoint) { this.since = null; this.elapsed = checkpoint; }
    read() { return this.elapsed + (this.since == null ? 0 : this.now() - this.since); }
  }
  const PLAYERS = ['reference', 'qwen', 'reasoning', 'human'];
  const isModel = player => player === 'qwen' || player === 'reasoning';
  // Qwen stops at a turn limit; local search always finishes and people stop when they like.
  const turnLimit = game => game.par ? game.par * 3 + 6 : 60;

  class Session {
    constructor({view, now = () => performance.now(), defer = fn => setTimeout(fn, 0)}) {
      Object.assign(this, {view, now, defer});
      this.game = null;
      this.player = 'reference';
      this.mode = 'READY';
      this.running = false;
      this.epoch = 0;
      this.controller = null;
      this.clock = new Clock(now);
      this.records = [];
      this.latest = null;
      this.failure = null;
      this.warmup = null;
      this.plan = null;
      this.presenting = null;
      this.settling = false;
    }
    get limit() { return this.game ? turnLimit(this.game) : null; }
    get modelPlayer() { return isModel(this.player); }
    get hasModelTurn() { return Rubiks.candidates(this.game, this.view.connection()).options.length > 0; }
    set(mode, event = 'mode') { this.mode = mode; this.view.update(event); }
    // A new seeded scramble. The view plays it from solved; input waits until it lands.
    reset({distance, seed, player}) {
      if (!PLAYERS.includes(player)) throw new Error('Unknown solver ' + player + '.');
      const game = new Rubiks.Game({distance, seed});
      this.halt();
      Object.assign(this, {game, player, records: [], latest: null, failure: null, warmup: null, plan: null, clock: new Clock(this.now)});
      const token = ++this.epoch;
      this.mode = 'SCRAMBLING';
      this.view.update('reset');
      this.presenting = Promise.resolve(this.view.scramble(game)).then(() => {
        if (token !== this.epoch) return;
        this.presenting = null;
        this.set('READY');
      });
      return this.presenting;
    }
    // Stops decisions and cancels a pending request without changing the mode.
    halt() {
      this.running = false;
      this.epoch++;
      this.controller?.abort();
      this.controller = null;
      this.clock.stop();
    }
    pause(mode = 'PAUSED') {
      if (!this.running) return;
      this.halt();
      this.set(mode);
    }
    // The scramble never depends on the solver, so a paused solve can change hands: local search
    // can finish what Qwen started, or a person can take over.
    setPlayer(player) {
      if (!PLAYERS.includes(player)) throw new Error('Unknown solver ' + player + '.');
      if (this.running) this.halt();
      this.player = player;
      this.warmup = null;
      this.plan = null;
      this.latest = null;
      if (this.game && !this.presenting && !this.game.solved()) this.mode = this.game.history.length ? 'PAUSED' : 'READY';
      this.view.update('player');
    }
    toggle(step = false) {
      if (this.running) { this.pause(); return Promise.resolve(); }
      return this.start(step);
    }
    async start(step = false) {
      if (!this.game || this.game.solved()) return;
      if (this.presenting) {
        const epoch = this.epoch;
        this.view.settle();
        await this.presenting;
        if (epoch !== this.epoch) return;
      }
      this.failure = null;
      this.running = true;
      const token = ++this.epoch;
      if (this.player === 'human') { this.clock.start(); this.set('SOLVING'); return; }
      if (this.modelPlayer && !this.hasModelTurn) { this.finish('NO NEW TURN'); return; }
      if (this.modelPlayer && !this.warmup) {
        this.set('WARM-UP');
        if (!(await this.warm(token))) return;
      }
      this.set('SOLVING');
      while (this.running && token === this.epoch && !this.game.solved()) {
        if (this.modelPlayer && this.game.history.length >= this.limit) { this.finish('TURN LIMIT'); return; }
        if (this.modelPlayer && !this.hasModelTurn) { this.finish('NO NEW TURN'); return; }
        if (!(await this.turn(token))) return;
        if (step) break;
      }
      if (token !== this.epoch) return;
      if (this.game.solved()) this.finish('SOLVED');
      else { this.running = false; this.set('PAUSED'); }
    }
    finish(mode) {
      this.running = false;
      this.epoch++;
      this.clock.stop();
      this.set(mode, mode === 'SOLVED' ? 'solved' : 'mode');
    }
    fail(error, warmup = false) {
      this.failure = {message: error.message, kind: error.kind || null, status: error.status ?? null, warmup, receipt: error.receipt || null};
      this.running = false;
      this.epoch++;
      this.set('REQUEST FAILED', 'failure');
    }
    async warm(token) {
      const connection = this.view.connection();
      this.controller = new AbortController();
      try {
        const result = await this.choose(connection, this.controller.signal);
        if (token !== this.epoch) return false;
        this.warmup = {ms: result.ms, usage: result.usage, method: this.player, request: result.request, response: result.response};
        return true;
      } catch (error) {
        if (token === this.epoch && error.name !== 'AbortError') this.fail(error, true);
        return false;
      } finally {
        if (token === this.epoch) this.controller = null;
      }
    }
    choose(connection, signal) {
      return (this.player === 'reasoning' ? RubiksReasoning : Rubiks).choose(this.game, connection, {signal});
    }
    // One decision and its presentation. False when the run stopped meanwhile.
    async turn(token) {
      let decision;
      if (this.player === 'reference') decision = this.searchTurn();
      else {
        decision = await this.liveTurn(token);
        if (!decision) return false;
      }
      this.records.push(decision.record);
      this.latest = decision.latest;
      this.view.update('decision');
      this.follow();
      await this.view.present(Cube.move(decision.record.turn), {source: decision.record.source});
      return token === this.epoch;
    }
    // Local search plans once and follows its line; its search time is its clock. A two-phase
    // line is replaced as soon as the position is close enough to prove a shorter one.
    searchTurn() {
      this.clock.start();
      const started = this.now(), current = this.plan, distance = this.game.distance;
      let searched = false;
      if (!current?.moves.length || (!current.optimal && distance != null && distance < current.moves.length)) { this.plan = this.game.solution(); searched = true; }
      const id = this.plan.moves.shift();
      const ms = this.now() - started;
      this.clock.stop();
      this.game.commit(id, {quick: true});
      const plan = {moves: this.plan.moves.slice(), optimal: this.plan.optimal};
      return {record: {source: 'reference', turn: id, ms, searched, index: this.game.trace.length - 1, planOptimal: plan.optimal},
        latest: {source: 'reference', turn: id, ms, searched, plan}};
    }
    // Instrument analysis runs before the request and stays off the clock and out of the prompt.
    async liveTurn(token) {
      const analysis = this.game.analysis(), connection = this.view.connection();
      this.controller = new AbortController();
      const checkpoint = this.clock.start();
      let result;
      try {
        result = await this.choose(connection, this.controller.signal);
      } catch (error) {
        this.clock.revert(checkpoint);
        if (token === this.epoch && error.name !== 'AbortError') this.fail(error);
        return null;
      }
      if (token !== this.epoch) { this.clock.revert(checkpoint); return null; }
      this.clock.stop();
      this.controller = null;
      const {option, probabilities} = result;
      const row = analysis?.find(item => item.id === option.id) || null;
      const optimalMass = analysis && probabilities ? result.options.reduce((sum, item) => analysis.find(row => row.id === item.id)?.delta === -1 ? sum + probabilities[item.code] : sum, 0) : null;
      this.game.commit(option.id, {known: row?.distance ?? null, quick: true});
      const record = {source: 'live', method: this.player, turn: option.id, code: option.code, ms: result.ms, index: this.game.trace.length - 1, optimalMass,
        avoidRepeats: connection.avoidRepeats !== false, excluded: result.excluded,
        probabilities, usage: result.usage, choiceWarning: result.choiceWarning, request: result.request, response: result.response};
      return {record, latest: {...record, options: result.options, analysis}};
    }
    // A person's turn commits at input time. The view animates it, or already has while dragging.
    humanTurn(id, {presented = false} = {}) {
      if (this.player !== 'human' || !this.game || this.game.solved() || this.presenting || this.mode === 'REQUEST FAILED') return false;
      if (!this.running) { this.running = true; this.epoch++; }
      this.clock.start();
      this.game.commit(id, {quick: true});
      this.records.push({source: 'human', turn: id, ms: null, index: this.game.trace.length - 1, atMs: this.clock.read()});
      this.latest = {source: 'human', turn: id};
      if (!presented) this.view.present(Cube.move(id), {source: 'human'});
      if (this.game.solved()) this.finish('SOLVED');
      else { this.follow(); this.set('SOLVING', 'turn'); }
      return true;
    }
    // Finishes a pending distance measurement when the page is idle.
    follow() {
      if (this.settling || !this.game.position.pending) return;
      this.settling = true;
      const game = this.game;
      this.defer(() => {
        this.settling = false;
        if (game === this.game && game.settle()) this.view.update('trace');
      });
    }
    summary() {
      const trace = this.game?.trace || [], graded = trace.filter(entry => entry.optimal != null);
      const masses = this.records.filter(record => record.optimalMass != null).map(record => record.optimalMass);
      const seconds = this.clock.read() / 1000;
      const failedRequestMs = this.failure && !this.failure.warmup ? this.failure.receipt?.ms || 0 : 0;
      const warmupMs = this.warmup?.ms ?? (this.failure?.warmup ? this.failure.receipt?.ms || 0 : 0);
      const failedUsage = this.failure?.receipt?.response?.usage;
      const inputTokens = this.records.reduce((sum, record) => sum + (record.usage?.input_tokens || 0), 0);
      const outputTokens = this.records.reduce((sum, record) => sum + (record.usage?.output_tokens || 0), 0);
      return {...SystemOne.summary(this.records), turns: trace.length, optimal: graded.filter(entry => entry.optimal).length, graded: graded.length,
        failedRequestMs, totalRequestMs: this.clock.read() + failedRequestMs,
        warmupMs, allRequestMs: this.clock.read() + failedRequestMs + warmupMs, inputTokens, outputTokens,
        allInputTokens: inputTokens + (this.warmup?.usage?.input_tokens || 0) + (failedUsage?.prompt_tokens ?? failedUsage?.input_tokens ?? 0),
        allOutputTokens: outputTokens + (this.warmup?.usage?.output_tokens || 0) + (failedUsage?.completion_tokens ?? failedUsage?.output_tokens ?? 0),
        clockMs: this.clock.read(), tps: seconds > 0 && trace.length ? trace.length / seconds : null,
        optimalMass: masses.length ? masses.reduce((a, b) => a + b, 0) / masses.length : null};
    }
    exportData() {
      const game = this.game;
      return {game: 'rubiks', version: 3, player: this.player, scramble: {setting: game.setting, seed: game.seed, moves: game.scramble, par: game.par},
        solved: game.solved(), mode: this.mode, history: game.history, trace: game.trace, summary: this.summary(), warmup: this.warmup,
        decisions: this.records, failure: this.failure};
    }
  }
  return {Session, Clock, PLAYERS, turnLimit};
})();
