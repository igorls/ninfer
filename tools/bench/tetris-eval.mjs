#!/usr/bin/env node
// Runs the demo's actual rules and request builder without a browser.
// Example: node tools/bench/tetris-eval.mjs --endpoint http://127.0.0.1:8010 --compare --pieces 100 --seed 1
import { readFile, writeFile } from 'node:fs/promises';
import vm from 'node:vm';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';

export async function loadTetris() {
  const source = await readFile(new URL('../../docs/arcade/tetris-game.js', import.meta.url), 'utf8');
  const shared = await readFile(new URL('../../docs/arcade/system-one.js', import.meta.url), 'utf8');
  return vm.runInNewContext(shared + '\n' + source + '; Tetris', {
    fetch, performance, AbortController, AbortSignal, setTimeout, clearTimeout,
  });
}

export async function evaluate({ endpoint, model = 'qwen3.8-27b', seed = 1, pieces = 100, compare = false, scenario = 'empty', apiKey = '', row = 12 }) {
  const T = await loadTetris();
  const encodings = compare ? ['original', 'compact'] : ['compact'];
  const runs = encodings.map(encoding => ({ encoding, game: new T.Game(seed, scenario), decisions: [], warmupMs: null }));
  for (const run of runs) {
    const options = run.game.options();
    const warm = await T.decide(endpoint, T.requestFor(run.game, options, model, run.encoding), options, {
      apiKey, signal: AbortSignal.timeout(10000),
    });
    run.warmupMs = warm.ms;
  }
  // Alternate request order each round to reduce order/thermal bias. Each policy owns its own board.
  for (let i = 0; i < pieces; i++) {
    for (const run of (i % 2 ? runs.slice().reverse() : runs)) {
      const game = run.game, spawn = T.spawn(game.board, game.kind);
      if (!spawn) continue;
      const options = game.options();
      if (!options.length) continue;
      const request = T.requestFor(game, options, model, run.encoding);
      const result = await T.decide(endpoint, request, options, { apiKey, signal: AbortSignal.timeout(10000) });
      const best = T.reference(options), board = game.board.map(r => r.slice());
      const budgetMs = Math.max(1, spawn.rows) * row;
      const decision = {
        piece: game.pieces + 1, kind: game.kind, board, choice: result.option.code, source: 'live',
        ms: result.ms, budgetMs, deadlineMet: result.ms <= budgetMs,
        policyMatch: T.compare(result.option, best) === 0,
        lines: result.option.lines, holes: result.option.holes, height: result.option.height,
        request, response: result.body,
      };
      game.commit(result.option);
      run.decisions.push(decision);
    }
    if (runs.every(run => !T.spawn(run.game.board, run.game.kind))) break;
  }
  return {
    schema: 'ninfer.tetris.eval.v1',
    at: new Date().toISOString(),
    config: { endpoint, modelLabel: model, seed, pieces, scenario, rowMs: row },
    methodology: 'Sequential local HTTP requests; warm-up excluded; encodings interleaved on independent boards with the same seven-bag sequence. Latency includes body read and validation. No browser rendering or strict-deadline cancellation. Policy agreement is not optimal-play accuracy.',
    runs: runs.map(run => ({
      encoding: run.encoding, warmupMs: run.warmupMs,
      summary: {
        ...T.summary(run.decisions), pieces: run.game.pieces, lines: run.game.lines, score: run.game.score,
        finalHoles: T.metrics(run.game.board).holes, finalHeight: T.metrics(run.game.board).height,
        meanInputTokens: run.decisions.reduce((s, d) => s + d.response.usage.input_tokens, 0) / (run.decisions.length || 1),
      },
      decisions: run.decisions,
    })),
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    const args = process.argv.slice(2), values = {};
    for (let i = 0; i < args.length; i++) {
      if (args[i] === '--compare') values.compare = true;
      else if (['--endpoint', '--model', '--seed', '--pieces', '--scenario', '--output'].includes(args[i])) {
        if (!args[i + 1] || args[i + 1].startsWith('--')) throw new Error('Missing value for ' + args[i]);
        values[args[i].slice(2)] = args[++i];
      } else throw new Error('Unknown argument: ' + args[i]);
    }
    if (!values.endpoint) throw new Error('Pass --endpoint http://127.0.0.1:8010. Requests use the resident model. Optional: --compare --seed 1 --pieces 100 --scenario empty --output report.json. NINFER_API_KEY supplies auth.');
    const seed = Number(values.seed ?? 1), pieces = Number(values.pieces ?? 100);
    if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295 || !Number.isInteger(pieces) || pieces < 1 || pieces > 500) throw new Error('Use an integer seed 0..4294967295 and piece count 1..500.');
    if (values.scenario && !['empty', 'trench', 'stairs'].includes(values.scenario)) throw new Error('Unknown scenario.');
    const report = await evaluate({ ...values, seed, pieces, apiKey: process.env.NINFER_API_KEY || '' });
    if (values.output) await writeFile(values.output, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ config: report.config, runs: report.runs.map(({ encoding, warmupMs, summary }) => ({ encoding, warmupMs, ...summary })) }, null, 2));
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
