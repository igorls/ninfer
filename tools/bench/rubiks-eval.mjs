#!/usr/bin/env node
// Runs the cube page's session controller without a browser: the same scramble, solver, System One
// requests and grading, with presentation reduced to nothing.
import {readFile, writeFile} from 'node:fs/promises';
import vm from 'node:vm';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';

export const SOURCES = ['system-one.js', 'rubiks-cube.js', 'rubiks-solver.js', 'rubiks-game.js', 'rubiks-reasoning.js', 'rubiks-session.js'];
export async function loadRubiks(overrides = {}, {transform = source => source} = {}) {
  const sources = await Promise.all(SOURCES.map(path => readFile(new URL('../../docs/arcade/' + path, import.meta.url), 'utf8')));
  return vm.runInNewContext(transform(sources.join('\n')) + ';({...Rubiks, Rubiks, Cube, CubeSolver, RubiksSession, RubiksReasoning, SystemOne})', {
    fetch, performance, AbortController, AbortSignal, setTimeout, clearTimeout, ...overrides
  });
}

export async function evaluate({endpoint, model = 'qwen3.8-27b', player = 'live', distance = 5, seed = 1, maxTokens = 4096, avoidRepeats = true, apiKey = '', R = null, onUpdate = () => {}} = {}) {
  R ??= await loadRubiks();
  if (!['local', 'live', 'reasoning'].includes(player)) throw new Error('--player must be local, live or reasoning');
  if (player !== 'local' && !endpoint) throw new Error('--endpoint is required for a model player');
  const view = {scramble: () => Promise.resolve(), present: () => Promise.resolve(), settle() {}, update() {},
    connection: () => ({endpoint, model, apiKey, maxTokens, avoidRepeats})};
  const session = new R.RubiksSession.Session({view, defer: fn => fn()});
  view.update = event => onUpdate(event, session);
  await session.reset({distance, seed, player: player === 'local' ? 'reference' : player === 'reasoning' ? 'reasoning' : 'qwen'});
  const started = performance.now();
  await session.start();
  const data = session.exportData();
  return {schema: 'ninfer.rubiks.eval.v3', at: new Date().toISOString(),
    config: {endpoint: endpoint || null, model, player, distance, seed, avoidRepeats,
      ...(player === 'reasoning' ? {maxTokens, effort: 'low', temperature: 0, observation: 'systemone-v1'} : {})},
    methodology: player === 'local'
      ? 'Local search proves an optimal line inside an eleven-turn horizon and otherwise follows a two-phase line. Its clock is search time only.'
      : 'One excluded warm-up, then one ' + (player === 'reasoning' ? 'thinking-enabled Chat Completions' : 'System One') + ' request per turn. Both modes use the same state, history and candidate policy. ' + (avoidRepeats ? 'Turns revisiting a prior cube are excluded before requesting a choice; probabilities are conditional on the offered turns.' : 'All eighteen turns are offered, including repeats.') + ' Each returned legal turn is applied. Solver distances, grades and the scramble stay out of the prompt. The run stops at three times par plus six turns or when no unvisited next position remains. Probabilities and optimal mass are only available for System One. Failed-request time is retained separately; warm-up and grading are excluded from the solve clock.',
    wallMs: performance.now() - started, ...data};
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    const args = {};
    const usage = 'Expected [--player local|live|reasoning] [--endpoint URL] [--model label] [--distance 3..8|random] [--seed n] [--runs n] [--max-tokens n] [--repeats avoid|allow] [--output path]';
    for (let i = 2; i < process.argv.length; i += 2) {
      const key = process.argv[i];
      if (!['--endpoint', '--model', '--player', '--distance', '--seed', '--runs', '--max-tokens', '--repeats', '--output'].includes(key) || !process.argv[i + 1]) throw new Error(usage);
      args[key.slice(2)] = process.argv[i + 1];
    }
    const player = args.player || 'live';
    if (!['local', 'live', 'reasoning'].includes(player)) throw new Error('--player must be local, live or reasoning');
    const distance = args.distance === 'random' ? 'random' : Number(args.distance || 5);
    if (distance !== 'random' && !(Number.isInteger(distance) && distance >= 3 && distance <= 8)) throw new Error('--distance must be 3..8 or random');
    const seed = Number(args.seed || 1), runs = Number(args.runs || 1), maxTokens = Number(args['max-tokens'] || 4096);
    if (!Number.isInteger(seed) || seed < 0) throw new Error('--seed must be a non-negative integer');
    if (!Number.isInteger(runs) || runs < 1) throw new Error('--runs must be a positive integer');
    if (args.repeats && !['avoid', 'allow'].includes(args.repeats)) throw new Error('--repeats must be avoid or allow');
    const R = await loadRubiks();
    const reports = [];
    for (let i = 0; i < runs; i++) reports.push(await evaluate({endpoint: args.endpoint, model: args.model, player, distance, seed: seed + i, maxTokens, avoidRepeats: args.repeats !== 'allow', apiKey: process.env.NINFER_API_KEY || '', R,
      onUpdate: (event, session) => {
        if (event === 'decision') console.error('seed ' + session.game.seed + ' · turn ' + session.game.history.length + ' · ' + session.latest.turn + ' · distance ' + session.game.distance + ' · ' + Math.round(session.latest.ms) + ' ms');
      }}));
    const report = runs === 1 ? reports[0] : {schema: 'ninfer.rubiks.eval-set.v3', runs: reports};
    if (args.output) await writeFile(args.output, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(reports.map(r => ({seed: r.scramble.seed, par: r.scramble.par, solved: r.solved, mode: r.mode,
      turns: r.summary.turns, optimal: r.summary.optimal + '/' + r.summary.graded, optimalMass: r.summary.optimalMass, p50: r.summary.p50,
      clockMs: r.summary.clockMs, allRequestMs: r.summary.allRequestMs, outputTokens: r.summary.allOutputTokens,
      ...(r.failure ? {failure: r.failure.message, warmupFailed: r.failure.warmup} : {})})), null, 2));
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
