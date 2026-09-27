import test from 'node:test';
import assert from 'node:assert/strict';
import {loadKitchen} from '../tools/bench/kitchen-eval.mjs';
const config=(extra={})=>({layout:'cooperative',seed:1,duration:30000,mode:'paused',goal:'revenue',players:['qwen','jev'],profiles:{qwen:{endpoint:'http://127.0.0.1:8010',model:'qwen3.8-27b',apiKey:'qwen-test'},jev:{endpoint:'http://127.0.0.1:8012',model:'jev-latest',apiKey:'jev-test'}},...extra});
const tick=()=>new Promise(setImmediate);
function reply(init,invalid=false){const request=JSON.parse(init.body),keys=Object.keys(request.questions.move.criteria);return {ok:true,status:200,text:async()=>JSON.stringify({model:request.model,answers:{move:{type:'choice',choice:invalid?'invalid':keys[0],confidence:1,probabilities:Object.fromEntries(keys.map((key,i)=>[key,i===0?1:0]))}},usage:{input_tokens:200,output_tokens:0}})};}
function holdSleep(ms,signal){return new Promise((resolve,reject)=>signal.addEventListener('abort',()=>reject(new DOMException('Paused','AbortError')),{once:true}));}

test('each actor sees only its own legal jobs and the partner state; one shared world advances once',async()=>{
  const K=await loadKitchen(),match=new K.Match(config({players:['reference','reference']}));
  assert.equal(match.games.length,1);assert.equal(match.lanes[0].game,match.lanes[1].game);
  for(const lane of match.lanes){
    const options=match.options(lane);assert.ok(options.every(option=>option.type==='wait'||option.chef===lane.chefId));
    assert.equal(new Set(options.map(option=>option.code)).size,options.length);
    const request=K.requestFor(lane.game,options,'test','revenue',lane.chefId);
    assert.match(request.questions.move.instructions,new RegExp('only '+lane.game.chefs[lane.chefId].name));assert.match(request.state,/Recent shared activity/);
  }
  await match.start(true);assert.equal(match.games[0].time,500);
  assert.equal(match.lanes[0].last.applied,true);assert.equal(match.lanes[1].last.conflict,true);
  assert.equal(match.games[0].chefs.filter(chef=>chef.job).length,1);assert.match(match.lanes[1].last.reason,/Ada already claimed/);
  assert.match(match.games[0].events.at(-1).text,/Bo rejected/);
});

test('paused simultaneous claims use rotating priority, independent of which provider answers first',async()=>{
  async function round(priority,fast){
    const held=[],K=await loadKitchen({fetch:async(url,init)=>new Promise(resolve=>held.push({resolve,init}))});
    const match=new K.Match(config());match.warmups={qwen:{ms:0},jev:{ms:0}};match.round=priority;
    const running=match.start(true);await tick();assert.equal(held.length,2);
    held[fast].resolve(reply(held[fast].init));await tick();assert.equal(match.games[0].chefs.filter(chef=>chef.job).length,0);assert.equal(match.games[0].time,0);
    assert.equal(match.lanes[fast].last.applied,null);assert.equal(match.lanes[fast].status,'WAITING FOR PARTNER');
    held[1-fast].resolve(reply(held[1-fast].init));await running;
    assert.equal(match.lanes[priority].last.applied,true);assert.equal(match.lanes[1-priority].last.conflict,true);
    assert.equal(match.lanes[priority].last.appliedAt,0);assert.equal(match.games[0].time,500);
  }
  await round(0,1);await round(1,0);
});

test('real time launches both decisions together and the first valid answer atomically claims the shared job',async()=>{
  let wall=0;const held=[],K=await loadKitchen({performance:{now:()=>wall},fetch:async(url,init)=>new Promise(resolve=>held.push({resolve,init}))});
  const match=new K.Match(config({mode:'realtime'}),{now:()=>wall,sleep:holdSleep});match.warmups={qwen:{ms:0},jev:{ms:0}};
  const running=match.start();await tick();assert.equal(held.length,2);
  wall=100;held[1].resolve(reply(held[1].init));await tick();assert.equal(match.games[0].orders[0].claimedBy,1);
  wall=300;held[0].resolve(reply(held[0].init));await tick();
  assert.equal(match.lanes[1].last.applied,true);assert.equal(match.lanes[0].last.conflict,true);assert.equal(match.games[0].time,300);
  assert.equal(match.games[0].chefs[0].job,null);assert.equal(match.lanes[0].records.length,1);
  match.pause();await running;
});

test('a chef can continue a partner dish; the handoff appears in shared state, activity and the receipt',async()=>{
  const K=await loadKitchen(),match=new K.Match(config({players:['reference','reference']})),game=match.games[0];
  assert.ok(game.apply(game.options(0).find(option=>option.type==='collect')));game.advanceTo(game.chefs[0].job.finish);
  const options=game.options(1),chosen=options.find(option=>option.type==='chop'),record={};
  match.lanes[1].prepared={option:chosen,record};match.commit(match.lanes[1]);
  assert.equal(record.handoffFrom,0);assert.equal(game.stats.handoffs,1);assert.equal(game.orders[0].claimedBy,1);
  assert.ok(game.events.some(event=>event.text.includes('Bo takes over #1 from Ada')));
  const request=K.requestFor(game,game.options(0),'test','revenue',0);
  assert.match(request.state,/assigned to Bo/);assert.match(request.state,/Bo takes over/);
});

test('expired jobs are stale, not resource conflicts, and a chef cannot take over its partner actor',async()=>{
  const K=await loadKitchen(),match=new K.Match(config({players:['human','reference']})),game=match.games[0];
  const option=match.options(match.lanes[0])[0];game.advanceTo(30000);
  const record={};match.lanes[0].prepared={option,record};match.commit(match.lanes[0]);
  assert.equal(record.stale,true);assert.equal(record.conflict,false);assert.equal(game.chefs[0].job,null);
  match.reset(config({players:['human','human']}));const running=match.start(true);await tick();
  const partnerOption=match.options(match.lanes[1])[0];assert.equal(match.choose(0,partnerOption.id),false);
  match.pause();await running;assert.ok(match.lanes.every(lane=>!lane.human));
});

test('pause retains a completed paused proposal for retry, while reset cancels and discards the whole round',async()=>{
  const held=[],K=await loadKitchen({fetch:async(url,init)=>new Promise((resolve,reject)=>{
    init.signal.addEventListener('abort',()=>reject(new DOMException('Paused','AbortError')),{once:true});held.push({resolve,init});
  })});
  const match=new K.Match(config());match.warmups={qwen:{ms:0},jev:{ms:0}};
  const first=match.start(true);await tick();held[0].resolve(reply(held[0].init));await tick();match.pause();await first;
  assert.equal(match.lanes[0].records.length,1);assert.equal(match.lanes[0].last.applied,null);assert.ok(match.lanes[0].prepared);
  const retry=match.start(true);await tick();assert.equal(held.length,3);held[2].resolve(reply(held[2].init));await retry;
  assert.equal(match.lanes[0].records.length,1);assert.equal(match.lanes[0].last.applied,true);assert.equal(match.games[0].time,500);
  match.reset(config());match.warmups={qwen:{ms:0},jev:{ms:0}};
  const next=match.start(true);await tick();held[3].resolve(reply(held[3].init));await tick();match.reset(config({seed:2}));await next;
  assert.equal(match.games[0].time,0);assert.equal(match.games.length,1);assert.ok(match.lanes.every(lane=>lane.last===null&&lane.prepared===null&&lane.records.length===0));
});

test('a failed chef pauses the shared round; a valid teammate receipt is reused without applying early',async()=>{
  let fail=true;const counts={};const K=await loadKitchen({fetch:async(url,init)=>{const model=JSON.parse(init.body).model;counts[model]=(counts[model]||0)+1;return reply(init,model==='jev-latest'&&fail);}});
  const match=new K.Match(config());match.warmups={qwen:{ms:0},jev:{ms:0}};
  await match.start(true);assert.equal(match.phase,'REQUEST FAILED');assert.equal(match.failure.chefId,1);assert.equal(match.games[0].time,0);
  assert.equal(match.lanes[0].last.applied,null);assert.ok(match.games[0].chefs.every(chef=>!chef.job));
  fail=false;await match.start(true);assert.equal(match.games[0].time,500);assert.equal(counts['qwen3.8-27b'],1);assert.equal(counts['jev-latest'],2);
});

test('a full local cooperative shift exports a single team result and separate untimed actor records',async()=>{
  const K=await loadKitchen({fetch:()=>{throw new Error('Local only');}}),match=new K.Match(config({players:['reference','reference'],duration:60000}),{sleep:async()=>{}});
  await match.start();const report=match.report();assert.equal(report.kitchens.length,1);assert.equal(report.chefs.length,2);assert.equal(report.kitchens[0].time,60000);
  assert.equal(report.config.layout,'cooperative');assert.ok(report.kitchens[0].stats.served>0);assert.ok(report.kitchens[0].stats.handoffs>0);
  for(const chef of report.chefs){assert.equal(chef.summary.samples,0);assert.ok(chef.decisions.every(record=>record.options.every(option=>option.type==='wait'||option.chef===chef.chefId)));}
  assert.ok(!JSON.stringify(report).includes('qwen-test'));assert.ok(!JSON.stringify(report).includes('jev-test'));
});
