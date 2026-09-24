import test from 'node:test';
import assert from 'node:assert/strict';
import { loadTetris } from '../tools/bench/tetris-eval.mjs';

const T = await loadTetris();
const plain = value => JSON.parse(JSON.stringify(value));
const filled = grid => grid.flat().filter(Boolean).length;

test('two adjacent full rows clear together and preserve the row above', () => {
  const grid = T.emptyBoard();
  grid[17][0] = 'J';
  for (const y of [18, 19]) for (let x = 0; x < 10; x++) if (x !== 4 && x !== 5) grid[y][x] = 'G';
  const before = plain(grid);
  const result = T.place(grid, [[4,18],[5,18],[4,19],[5,19]], 'O');
  assert.equal(result.lines, 2);
  assert.equal(result.grid[19][0], 'J');
  assert.equal(filled(result.grid), 1);
  assert.deepEqual(plain(grid), before, 'placement must not mutate the observation');
});

test('nonadjacent line clears shift survivors by the correct number of rows', () => {
  const grid = T.emptyBoard();
  for (const y of [17, 19]) for (let x = 0; x < 10; x++) if (x !== 4) grid[y][x] = 'G';
  grid[16][0] = 'J'; grid[18][9] = 'L';
  const result = T.place(grid, [[4,16],[4,17],[4,18],[4,19]], 'I');
  assert.equal(result.lines, 2);
  assert.equal(result.grid[18][0], 'J');
  assert.equal(result.grid[18][4], 'I');
  assert.equal(result.grid[19][9], 'L');
  assert.equal(result.grid[19][4], 'I');
});

test('four-line fixture clears exactly four lines through a real legal landing', () => {
  const game = new T.Game(1, 'trench');
  const option = T.reference(game.options());
  assert.equal(game.kind, 'I');
  assert.equal(option.lines, 4);
  game.commit(option);
  assert.equal(game.lines, 4);
  assert.equal(game.score, 800);
  assert.equal(filled(game.board), 0);
});

test('all seven pieces enumerate distinct supported top-drop landings', () => {
  const expected = { I:17, O:9, T:34, J:34, L:34, S:17, Z:17 };
  for (const [kind, count] of Object.entries(expected)) {
    const grid = T.emptyBoard(), options = T.landings(grid, kind);
    assert.equal(options.length, count);
    assert.equal(new Set(options.map(o => o.cells.map(([x,y]) => x+y*10).sort((a,b)=>a-b).join(','))).size, count);
    for (const option of options) {
      assert.equal(filled(T.place(grid, option.cells, kind).grid), 4);
      assert.ok(T.collides(grid, T.SHAPES[kind][option.rot], option.column, option.y+1));
    }
  }
});

test('seed zero is reproducible and every bag contains all seven pieces', () => {
  const a = new T.Game(0), b = new T.Game(0), observed = [];
  for (let i = 0; i < 35; i++) {
    observed.push(a.kind);
    assert.equal(a.kind, b.kind);
    a.commit(T.reference(a.options())); b.commit(T.reference(b.options()));
  }
  for (let i = 0; i < 35; i += 7) assert.equal(new Set(observed.slice(i,i+7)).size, 7);
  assert.deepEqual(plain(a.board), plain(b.board));
});

test('reference comparisons use exact priority and accept equivalent outcomes', () => {
  const a = { lines:1, holes:10, height:19, bump:60 };
  const b = { lines:0, holes:0, height:1, bump:0 };
  assert.ok(T.compare(a,b)<0);
  assert.ok(T.compare({ ...b,height:2,bump:0 },{ ...b,height:1,bump:30 })>0);
  assert.equal(T.compare({ ...b, column:0 },{ ...b,column:9 }),0);
});

test('long reference runs preserve exact board occupancy through clears', () => {
  for (const seed of [0,1,7,42]) {
    const game = new T.Game(seed);
    for (let i = 0; i < 150; i++) {
      if (!T.spawn(game.board, game.kind)) break;
      const before = filled(game.board), option = T.reference(game.options());
      game.commit(option);
      assert.equal(filled(game.board), before+4-option.lines*10);
      assert.equal(game.board.length,20);
      assert.ok(game.board.every(r=>r.length===10));
    }
  }
});

function response(options, winner = options[0].code) {
  return { model:'qwen3.8-27b', answers:{move:{type:'choice',choice:winner,confidence:1,probabilities:Object.fromEntries(options.map(o=>[o.code,o.code===winner?1:0]))}}, usage:{input_tokens:200,output_tokens:0} };
}

test('wire contract carries every landing and decoding cannot silently substitute a heuristic', () => {
  const game = new T.Game(1), options = game.options(), request = T.requestFor(game,options,'test-model');
  assert.deepEqual(Object.keys(request.questions.move.criteria), plain(options.map(o=>o.code)));
  assert.equal(request.questions.move.type,'choice');
  const chosen = options.at(-1);
  assert.equal(T.decode(response(options,chosen.code),options).option.code,chosen.code);
  for (const change of [
    body=>body.answers.move.choice='unknown',
    body=>body.answers.move.probabilities[options[0].code]=NaN,
    body=>body.answers.move.probabilities.extra=0,
    body=>body.answers.move.probabilities[options[0].code]=0,
    body=>body.answers.move.confidence=2,
    body=>body.usage.output_tokens=-1,
    body=>body.usage.output_tokens=1.5,
    body=>body.answers.move.type='score',
  ]) {
    const body=response(options);change(body);assert.throws(()=>T.decode(body,options));
  }
  const inconsistent=response(options);inconsistent.answers.move.choice=options[1].code;
  const selected=T.decode(inconsistent,options);
  assert.equal(selected.option,options[1]);assert.match(selected.choiceWarning,/reported leader/);
  const jev=response(options);jev.model='jev-latest';jev.usage.output_tokens=12;
  assert.equal(T.decode(jev,options).usage.output_tokens,12,'TypeSafe reports output usage even without generated answer text');
});

test('latency includes the complete response body and parsing/validation', async () => {
  const game = new T.Game(1), options = game.options();
  let time=0;
  const result=await T.decide('http://localhost:8010/',T.requestFor(game,options,'model'),options,{
    now:()=>time,apiKey:'test-secret',
    fetchImpl:async(url,init)=>{
      assert.equal(url,'http://localhost:8010/v1/systemone');
      assert.equal(init.headers.Authorization,'Bearer test-secret');
      time=20;
      return {ok:true,text:async()=>{time=90;return JSON.stringify(response(options));}};
    },
  });
  assert.equal(result.ms,90);
});

test('HTTP and malformed responses fail explicitly', async () => {
  const game=new T.Game(1),options=game.options();
  await assert.rejects(T.decide('http://localhost',{},options,{fetchImpl:async()=>({ok:false,status:401,text:async()=>'{"error":{"message":"bad token"}}'})}),/HTTP 401: bad token/);
  await assert.rejects(T.decide('http://localhost',{},options,{fetchImpl:async()=>({ok:true,text:async()=>'<html>not JSON'})}),/not valid JSON/);
});

test('deadline expiry aborts a stalled request and discards its late result', async () => {
  let signal,resolveLate;
  const result=await T.withDeadline(s=>{signal=s;return new Promise(resolve=>{resolveLate=resolve;});},5);
  assert.equal(result.expired,true);
  assert.equal(signal.aborted,true);
  resolveLate({choice:'late'});
  assert.equal(result.expired,true);
});

test('reset/pause cancellation propagates through an outstanding request', async () => {
  const controller=new AbortController();
  const pending=T.withDeadline(signal=>new Promise((resolve,reject)=>signal.addEventListener('abort',()=>reject(new Error('cancelled')))),10000,controller.signal);
  controller.abort();
  await assert.rejects(pending,/cancelled/);
});

test('measured percentiles exclude reference, warm-up and cancelled decisions', () => {
  const stats=T.summary([
    {source:'live',ms:50,policyMatch:true,deadlineMet:true},
    {source:'live',ms:100,policyMatch:false,deadlineMet:false},
    {source:'live',ms:null,policyMatch:null,deadlineMet:false},
    {source:'reference',ms:1,policyMatch:true,deadlineMet:null},
  ]);
  assert.equal(stats.samples,2);assert.equal(stats.p50,50);assert.equal(stats.p95,100);
  assert.equal(stats.deadlines,3);assert.equal(stats.misses,2);assert.equal(stats.evaluated,3);
});
