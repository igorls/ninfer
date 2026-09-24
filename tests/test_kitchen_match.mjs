import test from 'node:test';
import assert from 'node:assert/strict';
import {loadKitchen} from '../tools/bench/kitchen-eval.mjs';
const config=(extra={})=>({seed:1,duration:30000,mode:'paused',goal:'revenue',players:['qwen','reference'],profiles:{qwen:{endpoint:'http://127.0.0.1:8010',model:'qwen3.8-27b',apiKey:'qwen-secret'},jev:{endpoint:'http://127.0.0.1:8012',model:'jev-latest',apiKey:'jev-secret'}},...extra});
function response(init){const request=JSON.parse(init.body),keys=Object.keys(request.questions.move.criteria);return {model:request.model,answers:{move:{type:'choice',choice:keys[0],confidence:1,probabilities:Object.fromEntries(keys.map((key,i)=>[key,i===0?1:0]))}},usage:{input_tokens:200,output_tokens:request.model==='jev-latest'?18:0}};}
const ok=body=>({ok:true,status:200,text:async()=>JSON.stringify(body)});
const turn=()=>new Promise(setImmediate);
function heldSleep(ms,signal){return new Promise((resolve,reject)=>{if(signal.aborted)reject(new DOMException('Paused','AbortError'));else signal.addEventListener('abort',()=>reject(new DOMException('Paused','AbortError')),{once:true});});}

test('provider profiles, warm-ups, completed samples and exports remain independent and key-free',async()=>{
  const calls=[];let wall=0;
  const K=await loadKitchen({performance:{now:()=>wall},fetch:async(url,init)=>{
    const body=response(init),jev=body.model==='jev-latest';calls.push({url,init});wall+=jev?200:50;
    assert.equal(url,jev?'http://127.0.0.1:8012/v1/systemone':'http://127.0.0.1:8010/v1/systemone');
    assert.equal(init.headers.Authorization,jev?'Bearer jev-secret':'Bearer qwen-secret');return ok(body);
  }});
  const match=new K.Match(config({players:['jev','qwen']}));await match.start(true);
  assert.equal(calls.length,4);assert.equal(match.phase,'STEP COMPLETE');
  const report=match.report();assert.equal(report.kitchens[0].time,500);assert.equal(report.kitchens[1].time,500);
  assert.equal(report.warmups.jev.ms,200);assert.equal(report.warmups.qwen.ms,50);
  assert.equal(report.kitchens[0].summary.samples,1);assert.equal(report.kitchens[1].summary.samples,1);
  assert.equal(report.kitchens[0].decisions[0].response.usage.output_tokens,18);
  assert.ok(!JSON.stringify(report).includes('secret'));
});

test('real-time clocks advance during requests and reject an expired action without substitution',async()=>{
  let wall=0,calls=0,release;
  const K=await loadKitchen({performance:{now:()=>wall},fetch:async(url,init)=>++calls===1?ok(response(init)):new Promise(resolve=>{release=()=>resolve(ok(response(init)));})});
  const match=new K.Match(config({mode:'realtime',duration:3000,orders:[{id:1,recipe:'soup',at:0,deadline:500,price:20,vip:false}]}),{now:()=>wall,sleep:heldSleep});
  const running=match.start();await turn();assert.ok(release);wall=1000;release();await turn();
  assert.equal(match.lanes[0].game.time,1000);assert.equal(match.lanes[1].game.time,1000);
  assert.equal(match.lanes[0].last.stale,true);assert.equal(match.lanes[0].last.applied,false);
  assert.equal(match.lanes[0].game.chefs[0].job,null);assert.equal(match.lanes[0].last.ms,1000);
  match.pause();await running;
});

test('paused decisions freeze both kitchens until all answers arrive, then advance exactly one round',async()=>{
  let wall=0,calls=0,release;
  const K=await loadKitchen({performance:{now:()=>wall},fetch:async(url,init)=>++calls===1?ok(response(init)):new Promise(resolve=>{release=()=>resolve(ok(response(init)));})});
  const match=new K.Match(config(),{now:()=>wall}),running=match.start(true);await turn();wall=5000;
  assert.equal(match.lanes[0].game.time,0);assert.equal(match.lanes[1].game.time,0);
  release();await running;assert.equal(match.lanes[0].game.time,500);assert.equal(match.lanes[1].game.time,500);
  assert.equal(match.lanes[0].last.applied,true);assert.equal(match.lanes[0].last.ms,5000);
});

test('reset cancels pending work and an eventual old response cannot affect the new shift',async()=>{
  let calls=0,release,lastSignal;
  const K=await loadKitchen({fetch:async(url,init)=>{if(++calls===1)return ok(response(init));lastSignal=init.signal;return new Promise(resolve=>{release=()=>resolve(ok(response(init)));});}});
  const match=new K.Match(config()),running=match.start(true);await turn();match.reset(config({seed:2}));
  assert.ok(lastSignal.aborted);release();await running;
  assert.equal(match.phase,'READY');assert.equal(match.lanes[0].game.time,0);assert.equal(match.lanes[0].records.length,0);assert.equal(match.lanes[1].records.length,0);
});

test('a failed response pauses the pair and preserves the receipt; retry does not change providers',async()=>{
  let calls=0,fail=true;
  const K=await loadKitchen({fetch:async(url,init)=>{calls++;const body=response(init);if(calls===2&&fail)body.answers.move.choice='unknown';return ok(body);}});
  const match=new K.Match(config({orders:[{id:1,recipe:'soup',at:0,deadline:25000,price:20,vip:false},{id:2,recipe:'salad',at:0,deadline:25000,price:12,vip:false}]}));
  // A second independent job is available, but the completed side must not receive an extra turn on retry.
  match.lanes[1].game.orders[1].stage='raw';await match.start(true);
  assert.equal(match.phase,'REQUEST FAILED');assert.equal(match.lanes[0].records.length,0);assert.equal(match.lanes[0].game.time,0);
  assert.equal(match.failure.receipt.response.answers.move.choice,'unknown');assert.equal(match.failure.player,'qwen');
  fail=false;await match.start(true);assert.equal(match.phase,'STEP COMPLETE');assert.equal(match.failure,null);assert.equal(match.lanes[0].records.length,1);
  assert.equal(match.lanes[1].records.length,1);
});

test('goal changes affect new instructions without resetting the world or changing in-flight evidence',async()=>{
  const requests=[];
  const K=await loadKitchen({fetch:async(url,init)=>{requests.push(JSON.parse(init.body));return ok(response(init));}});
  const match=new K.Match(config());await match.start(true);match.setGoal('vip');
  const first=match.lanes[0].records[0];assert.equal(first.goal,'revenue');assert.equal(match.lanes[0].game.time,500);
  for(let i=0;i<10&&match.lanes[0].records.length<2;i++)await match.start(true);
  assert.match(requests.at(-1).questions.move.instructions,/VIP/);assert.equal(match.lanes[0].records[1].goal,'vip');
  assert.equal(match.goalHistory.length,2);assert.ok(match.lanes[0].game.time>500);
});

test('human choices remain manual, and pause releases a waiting human round',async()=>{
  const K=await loadKitchen({fetch:()=>{throw new Error('No API in local play');}}),match=new K.Match(config({players:['human','reference']}));
  const first=match.start(true);await turn();assert.equal(match.lanes[0].game.time,0);
  const option=match.lanes[0].game.options()[1];assert.ok(match.choose(0,option.id));await first;
  assert.equal(match.lanes[0].last.action,option.id);assert.equal(match.lanes[0].last.source,'human');assert.equal(match.lanes[0].last.ms,null);
  match.reset(config({players:['human','reference']}));const next=match.start(true);await turn();match.pause();await next;assert.equal(match.lanes[0].human,null);
});
