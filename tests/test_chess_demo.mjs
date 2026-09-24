import test from 'node:test';
import assert from 'node:assert/strict';
import {loadChess} from '../tools/bench/chess-eval.mjs';
const C=await loadChess();
const highMobility='7k/8/8/3Q4/8/8/2Q5/K3Q3 w - - 0 1';

test('opening actions preserve the exact position and history, then commit a legal move',()=>{
  const game=new C.Game(),fen=game.fen;
  assert.equal(game.options().length,20);assert.equal(game.fen,fen);assert.equal(game.ply,0);
  game.commit(game.options().find(o=>o.id==='e2e4'));
  assert.equal(game.turn,'b');assert.equal(game.ply,1);assert.equal(game.rules.history()[0],'e4');
  assert.ok(game.options().every(o=>o.color==='b'));assert.equal(game.ply,1);assert.equal(game.rules.history()[0],'e4');
  assert.throws(()=>game.commit({from:'e7',to:'e4'}),/Invalid move/);
});

test('special moves retain castling, en passant and all four promotion choices',()=>{
  const castle=new C.Game('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1');
  assert.ok(castle.options().some(o=>o.san==='O-O'));assert.ok(castle.options().some(o=>o.san==='O-O-O'));
  castle.commit(castle.options().find(o=>o.san==='O-O'));assert.equal(castle.rules.get('f1').type,'r');
  const ep=new C.Game();for(const san of ['e4','a6','e5','d5'])ep.rules.move(san);
  const capture=ep.options().find(o=>o.id==='e5d6');assert.equal(capture.captured,'p');ep.commit(capture);assert.equal(ep.rules.get('d5'),undefined);
  const promotion=new C.Game(C.positions.promotion),options=promotion.options().filter(o=>o.from==='a7');
  assert.deepEqual([...options.map(o=>o.promotion)].sort(),['b','n','q','r']);
  promotion.commit(options.find(o=>o.promotion==='n'));assert.equal(promotion.rules.get('a8').type,'n');
});

test('checkmate, stalemate, repetition and insufficient material end the game',()=>{
  const mate=new C.Game(C.positions.mate),option=mate.options().find(o=>o.mate);assert.equal(option.san,'Rd8#');mate.commit(option);
  assert.match(mate.status().text,/White wins by checkmate/);assert.equal(mate.options().length,0);
  assert.match(new C.Game('7k/5K2/6Q1/8/8/8/8/8 b - - 0 1').status().text,/stalemate/);
  assert.match(new C.Game('7k/8/8/8/8/8/8/K7 w - - 0 1').status().text,/insufficient/);
  const repetition=new C.Game();for(const san of ['Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8'])repetition.rules.move(san);
  repetition.options();assert.match(repetition.status().text,/threefold/);
});

test('a position above the API limit retains every move through unique piece groups',()=>{
  const game=new C.Game(highMobility),options=game.options(),plan=C.plan(game,options);
  assert.equal(options.length,73);assert.equal(plan.kind,'piece');
  assert.equal(new Set(plan.options.map(o=>o.code)).size,plan.options.length);
  assert.deepEqual([...plan.options.flatMap(o=>o.moves.map(m=>m.id))].sort(),[...options.map(o=>o.id)].sort());
  assert.ok(plan.options.every(o=>o.moves.length<=62));
  const body=C.requestFor(game,plan,'qwen3.8-27b');assert.equal(Object.keys(body.questions.move.criteria).length,plan.options.length);
  assert.ok(!JSON.stringify(body).includes('referenceScore'));
});

function responder(requests,{controller}={}) {
  return async(url,init)=>{
    const request=JSON.parse(init.body),keys=Object.keys(request.questions.move.criteria),choice=keys[0];requests.push(request);
    if(controller)controller.abort();
    return {ok:true,text:async()=>JSON.stringify({answers:{move:{type:'choice',choice,confidence:1,probabilities:Object.fromEntries(keys.map(k=>[k,k===choice?1:0]))}},usage:{input_tokens:200,output_tokens:0}})};
  };
}
test('two-stage choice records both exact responses and applies only the selected legal move',async()=>{
  const requests=[],D=await loadChess({fetch:responder(requests)}),game=new D.Game(highMobility),fen=game.fen;
  const result=await D.choose(game,game.options(),{endpoint:'http://test',model:'resident'});
  assert.equal(requests.length,2);assert.equal(result.stages.length,2);assert.equal(game.fen,fen);
  assert.equal(result.stages[0].kind,'piece');assert.equal(result.stages[1].kind,'move');
  assert.equal(result.ms,result.stages[0].ms+result.stages[1].ms);
  assert.ok(result.stages[0].options[0].moves.some(o=>o.id===result.option.id));game.commit(result.option);assert.equal(game.ply,1);
});

test('cancellation between stages prevents a second inference request',async()=>{
  const requests=[],controller=new AbortController(),D=await loadChess({fetch:responder(requests,{controller})}),game=new D.Game(highMobility);
  await assert.rejects(D.choose(game,game.options(),{endpoint:'http://test',model:'resident'},{signal:controller.signal}),{name:'AbortError'});
  assert.equal(requests.length,1);assert.equal(game.ply,0);
});
