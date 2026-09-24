import test from 'node:test';
import assert from 'node:assert/strict';
import {loadKitchen,evaluate} from '../tools/bench/kitchen-eval.mjs';
const K=await loadKitchen(),plain=value=>JSON.parse(JSON.stringify(value));
const ticket=(id=1,recipe='soup',extra={})=>({id,recipe,at:0,deadline:60000,price:K.recipes[recipe].price,vip:false,...extra});
function perform(game,type,order=1,chef=0){const option=game.options().find(option=>option.type===type&&option.order===order&&option.chef===chef);assert.ok(option,'missing '+type);assert.ok(game.apply(option));game.advanceTo(game.chefs[chef].job.finish);}
test('seeded order schedules are identical and independent of kitchen actions',()=>{
  const a=new K.Game({seed:0}),b=new K.Game({seed:0});
  assert.deepEqual(plain(a.schedule),plain(b.schedule));assert.notDeepEqual(plain(a.schedule),plain(new K.Game({seed:1}).schedule));
  perform(a,'collect');a.advanceTo(30000);b.advanceTo(30000);
  assert.deepEqual(plain(a.schedule),plain(b.schedule));assert.equal(a.arrival,b.arrival);
  assert.ok(a.active().length<=6);assert.ok(b.active().length<=6);
  assert.ok(!a.state().includes('#'+a.schedule.at(-1).id+' '),'observation does not expose future orders');
});
test('a soup completes the full pipeline, releases its chef during cooking and scores once',()=>{
  const game=new K.Game({orders:[ticket()]});
  for(const type of ['collect','chop','cook'])perform(game,type);
  assert.equal(game.orders[0].stage,'cooking');assert.equal(game.chefs[0].job,null);
  game.advanceTo(game.orders[0].readyAt);perform(game,'plate');perform(game,'serve');
  assert.equal(game.stats.served,1);assert.equal(game.stats.revenue,20);assert.equal(game.stats.waste,0);
  assert.equal(game.orders[0].status,'served');assert.ok(game.options().every(option=>option.order!==1));
});
test('salad skips cooking, station reservations and order claims prevent duplicate jobs',()=>{
  const game=new K.Game({orders:[ticket(1,'salad'),ticket(2,'soup')]});
  const collect=game.options().find(option=>option.chef===0&&option.order===1);game.apply(collect);
  assert.ok(game.options().every(option=>option.order!==1));assert.ok(game.options().every(option=>option.type!=='collect'));
  assert.equal(game.apply(collect),false);game.advanceTo(game.chefs[0].job.finish);
  perform(game,'chop');assert.ok(!game.options().some(option=>option.type==='cook'&&option.order===1));
  perform(game,'plate');perform(game,'serve');assert.equal(game.stats.served,1);
});
test('burned food blocks the stove until cleared, without counting its waste twice on expiry',()=>{
  const game=new K.Game({orders:[ticket()]});for(const type of ['collect','chop','cook'])perform(game,type);
  game.advanceTo(game.orders[0].burnAt);assert.equal(game.orders[0].stage,'burnt');assert.equal(game.stats.burned,1);assert.equal(game.stats.waste,1);
  game.advanceTo(60000);assert.equal(game.stats.missed,1);assert.equal(game.stats.waste,1);assert.equal(game.available('stove'),false);
  perform(game,'clear');assert.equal(game.available('stove'),true);assert.equal(game.orders[0].stage,'none');
});
test('food can burn during a plate job; arrival cannot resurrect it as a plated dish',()=>{
  const game=new K.Game({orders:[ticket()]});for(const type of ['collect','chop','cook'])perform(game,type);
  game.advanceTo(game.orders[0].burnAt-100);perform(game,'plate');
  assert.equal(game.orders[0].stage,'burnt');assert.equal(game.stats.waste,1);assert.ok(!game.options().some(option=>option.type==='serve'));
});
test('serving exactly at the deadline succeeds; expired jobs cannot resurrect orders',()=>{
  const game=new K.Game({orders:[ticket(1,'salad')]});for(const type of ['collect','chop','plate'])perform(game,type);
  const serve=game.options().find(option=>option.type==='serve'&&option.chef===0);game.orders[0].deadline=game.time+serve.duration;perform(game,'serve');assert.equal(game.stats.served,1);assert.equal(game.stats.missed,0);
  const expired=new K.Game({orders:[ticket(1,'soup',{deadline:500})]});const option=expired.options()[0];expired.apply(option);expired.advanceTo(5000);
  assert.equal(expired.orders[0].status,'missed');assert.equal(expired.orders[0].stage,'none');assert.equal(expired.chefs[0].job,null);assert.equal(expired.apply(option),false);
});
test('the adapter exposes every legal job and separates policy instructions from observations',()=>{
  const game=new K.Game(),options=game.options(),request=K.requestFor(game,options,'qwen3.8-27b','vip');
  assert.deepEqual(Object.keys(request.questions.move.criteria),plain(options.map(option=>option.code)));
  assert.match(request.questions.move.instructions,/VIP/);assert.equal(request.state,game.state());assert.ok(options.length<=62);
});
test('a complete local shift is deterministic, productive and untimed in both kitchens',async()=>{
  const report=await evaluate({seconds:60});assert.equal(report.failure,null);
  assert.equal(report.kitchens[0].time,60000);assert.ok(report.kitchens[0].stats.served>0);
  assert.deepEqual(report.kitchens[0].stats,report.kitchens[1].stats);assert.equal(report.kitchens[0].summary.samples,0);
  assert.ok(report.kitchens[0].decisions.every(record=>record.applied&&!record.request&&!record.response));
});
