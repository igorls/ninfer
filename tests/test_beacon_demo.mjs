import test from 'node:test';
import assert from 'node:assert/strict';
import {loadBeacon,observation,png} from '../tools/bench/beacon-eval.mjs';
const {Game:G,Session:S}=await loadBeacon();
const plain=value=>JSON.parse(JSON.stringify(value));
test('free movement is continuous, diagonals preserve speed and pulses stop',()=>{
  const a=G.create(),b=G.create();G.command(a,'B');G.command(b,'F');G.tick(a,200);G.tick(b,200);
  assert.ok(Math.abs(a.drone.x-128)<.01);assert.ok(Math.abs(Math.hypot(b.drone.x-100,b.drone.y-240)-28)<.01);
  G.tick(a,1000);const x=a.drone.x;G.tick(a,1000);assert.equal(a.drone.x,x);
});
test('walls stop motion without tunnelling, contacts count once and rings collect',()=>{
  const g=G.create();g.drone={x:250,y:300};G.command(g,'B',3000);G.tick(g,2500);assert.ok(g.drone.x<=268);assert.equal(g.hits,1);
  g.drone={...g.beacon};G.brake(g);G.tick(g,10);assert.equal(g.collected,1);assert.ok(g.beacon.y>380);
});
test('moving scene objects is bounded and cannot cover the drone or beacon',()=>{
  const g=G.create();assert.equal(G.moveObject(g,'wall-0',90,230),false);assert.equal(G.moveObject(g,'beacon',300,320),false);
  assert.equal(G.moveObject(g,'beacon',100,110),true);assert.equal(G.hitObject(g,100,110),'beacon');
});
test('requests carry pixels with identical text and fixed unfiltered actions across opposite worlds',()=>{
  const a=G.create(),b=G.create();b.beacon={x:50,y:300};
  const ra=S.request('qwen',observation(G,a).dataURL),rb=S.request('qwen',observation(G,b).dataURL);
  assert.notEqual(ra.images[0],rb.images[0]);delete ra.images;delete rb.images;assert.deepEqual(plain(ra),plain(rb));
  assert.equal(Object.keys(ra.questions.move.criteria).length,9);assert.deepEqual(Object.keys(ra).sort(),['model','questions','state']);
  assert.equal(png(G.render(a)).subarray(0,8).toString('hex'),'89504e470d0a1a0a');
});
test('freeze retains exact pixels, delay selects an older frame, blackout removes all objects',()=>{
  const camera=new S.Camera(),g=G.create(),first=G.render(g);camera.push(first,0);
  const held=camera.sample('freeze',0,320);g.drone.x=500;camera.push(G.render(g),1100);
  assert.equal(camera.sample('freeze',1100,320),held);assert.equal(camera.sample('delay',1100,320).capturedAt,0);
  assert.notDeepEqual(camera.sample('live',1100,320).pixels,held.pixels);
  const black=camera.sample('blackout',1100,320).pixels;assert.equal(new Set(Array.from({length:black.length/4},(_,i)=>Array.from(black.slice(i*4,i*4+4)).join())).size,1);
  camera.reset();assert.equal(camera.sample('delay',1100,320),null);
});
function answer(vision=80){return {option:G.actions[1],ms:60,usage:{vision_tokens:vision},body:{usage:{vision_tokens:vision}}};}
const cfg={model:'qwen',endpoint:'http://localhost',mode:'live',deadline:500,key:'test-secret'};
test('missing image-processing evidence fails closed and retains raw response without credentials',async()=>{
  const client=new S.Client({send:async()=>answer(0)});
  await assert.rejects(client.decide(observation(G,G.create()),cfg),/did not confirm/);
  assert.equal(client.records[0].status,'failed');assert.ok(client.records[0].response);assert.ok(!JSON.stringify(client.records).includes('test-secret'));
});
test('cancellation discards an outstanding decision even when transport ignores abort',async()=>{
  let release;const client=new S.Client({send:()=>new Promise(r=>{release=r;})});
  const pending=client.decide(observation(G,G.create()),cfg);assert.equal(client.pending,true);
  assert.equal(await client.decide(observation(G,G.create()),cfg),null);client.cancel();release(answer());
  assert.equal(await pending,null);assert.equal(client.records.length,0);assert.equal(client.pending,false);
});
test('request deadlines drop late commands but intentional camera age does not',async()=>{
  let now=0;const client=new S.Client({now:()=>now,send:async()=>{now+=600;return answer();}});
  const late=await client.decide({...observation(G,G.create()),capturedAt:-5000},cfg);assert.equal(late.dropped,true);
  client.send=async()=>{now+=40;return answer();};const frozen=await client.decide({...observation(G,G.create()),capturedAt:-5000},{...cfg,mode:'freeze'});
  assert.equal(frozen.dropped,false);assert.equal(frozen.frameAge,5640);
  await client.decide(observation(G,G.create()),cfg,{warmup:true});assert.equal(S.summary(client.records).n,2);
});
