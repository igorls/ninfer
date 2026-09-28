import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFile} from 'node:fs/promises';
const context=vm.createContext({console,AbortController,AbortSignal,setTimeout,clearTimeout,performance,fetch});
for(const file of ['system-one','arena-game','arena-session'])vm.runInContext(await readFile(new URL(`../docs/arcade/${file}.js`,import.meta.url),'utf8'),context);
const {Game,actions,request,reference}=vm.runInContext('ArenaGame',context),{Session,summary}=vm.runInContext('ArenaSession',context);
function incoming(type='cut',remaining=120){const g=new Game();g.player.x=-1;g.enemy.x=1;g.enemy.action='telegraph';g.enemy.attack={...vm.runInContext(`ArenaGame.attacks.${type}`,context),type,at:remaining,windup:remaining,started:0};g.enemy.until=remaining;return g;}
test('parry timing, ordinary block, and unblockable crush have different outcomes',()=>{
 const parry=incoming();parry.act('E');parry.tick(130);assert.equal(parry.player.hp,100);assert.equal(parry.stats.parries,1);
 const block=incoming('cut',300);block.act('E');block.tick(310);assert.equal(block.player.hp,97);assert.equal(block.stats.blocks,1);
 const crush=incoming('crush');crush.act('E');crush.tick(130);assert.equal(crush.player.hp,70);assert.equal(crush.stats.blocks,0);
});
test('dodge covers its invulnerability window, and an early dodge can be punished at the wall',()=>{
 const g=incoming('crush');g.act('F');g.tick(130);assert.equal(g.player.hp,100);assert.equal(g.stats.dodges,1);
 const early=incoming('crush',410);early.player.x=-5;early.enemy.x=-3;early.act('F');early.tick(420);assert.equal(early.player.hp,70);
});
test('attack range, commitment and recovery counter bonus are observable',()=>{
 const miss=new Game();miss.act('C');assert.equal(miss.act('F'),false);miss.tick(160);assert.equal(miss.enemy.hp,150);assert.equal(miss.stats.misses,1);
 const counter=new Game();counter.player.x=-1;counter.enemy.x=1;counter.enemy.action='recovery';counter.enemy.until=1000;counter.act('D');counter.tick(370);assert.equal(counter.enemy.hp,115);assert.equal(counter.stats.dealt,35);
});
test('legal choices respect stamina and observations expose facts without policy advice',()=>{
 const g=incoming();g.player.stamina=8;assert.deepEqual(Array.from(g.legal(),a=>a.code),['A','B','E','G']);const wire=request('test',g);assert.equal(wire.questions.move.criteria.F,undefined);assert.match(wire.state,/impact_in_ms/);assert.ok(!wire.state.includes('best_action'));assert.equal(actions.length,7);
});
test('seeded reference bouts are deterministic, bounded and complete',()=>{
 function play(){const g=new Game({seed:12});while(!g.over){const code=reference(g);if(code)g.act(code);g.tick(140);}return JSON.stringify({time:g.time,result:g.result,stats:g.stats,player:g.player,enemy:g.enemy});}
 assert.equal(play(),play());const a=JSON.parse(play());assert.ok(a.time<=45000);assert.ok(a.stats.dealt>0);assert.ok(a.stats.parries+a.stats.dodges>0);assert.ok(a.player.x>=-5&&a.enemy.x<=5);
});
test('simulation consumes full elapsed time; time verdict uses health fraction',()=>{
 const g=new Game({duration:1000});g.player.hp=80;g.enemy.hp=90;g.tick(2000);assert.equal(g.time,1000);assert.equal(g.result,'victory');
});
test('a lethal strike ends the attack and damage totals stop at remaining health',()=>{
 const g=incoming('crush',150);g.enemy.hp=5;g.act('C');g.tick(160);assert.equal(g.result,'victory');assert.equal(g.player.hp,100);assert.equal(g.stats.dealt,5);
 const block=incoming('cut',300);block.player.hp=1;block.act('E');block.tick(310);assert.equal(block.player.hp,0);assert.equal(block.stats.taken,1);
});
const answer=(options,code='E')=>({option:options.find(o=>o.code===code)||options[0],body:{answers:{}},ms:10,probabilities:{},usage:{input_tokens:12,output_tokens:0}});
test('warm-up leaves the world still, live latency advances it before the choice applies',async()=>{
 let now=0,resolve;const s=new Session({}, {now:()=>now,send:async(_e,_r,options)=>{now+=100;return answer(options);}});await s.start();assert.equal(s.game.time,0);assert.equal(s.records[0].status,'warmup');
 s.send=(_e,_r,options)=>new Promise(r=>resolve=()=>r(answer(options)));const pending=s.decide(false);now+=120;resolve();await pending;assert.equal(s.game.time,120);assert.equal(s.game.player.since,120);assert.equal(s.records.at(-1).status,'applied');
});
test('pause cancels in-flight work; a late answer cannot act or overwrite a new run',async()=>{
 let now=0,resolve;const s=new Session({}, {now:()=>now,send:(_e,_r,options)=>new Promise(r=>resolve=()=>r(answer(options)))});s.warmed=true;await s.start();const pending=s.decide(false);now=40;s.pause();await s.start();resolve();await pending;assert.equal(s.game.stats.actions,0);assert.equal(s.records.length,0);assert.equal(s.status,'running');
});
test('a cancelled warm-up cannot start the clock during its replacement',async()=>{
 const resolves=[];const s=new Session({}, {send:(_e,_r,options)=>new Promise(resolve=>resolves.push(()=>resolve(answer(options))))});
 const first=s.start();s.pause();const second=s.start();resolves[0]();await first;assert.equal(s.status,'warming');assert.equal(s.warmed,false);assert.equal(s.game.time,0);
 resolves[1]();await second;assert.equal(s.status,'running');assert.equal(s.records.length,1);s.pause();
});
test('real deadline aborts transport and never applies late work',async()=>{
 let aborted=false;const s=new Session({deadline:8},{send:(_e,_r,options,{signal})=>new Promise(resolve=>{signal.addEventListener('abort',()=>{aborted=true;resolve(answer(options));});})});s.warmed=true;await s.start();await s.decide(false);assert.equal(aborted,true);assert.equal(s.records[0].status,'expired');assert.equal(s.game.stats.actions,0);s.pause();
});
test('a newly stunned fighter rejects an otherwise valid response',async()=>{
 let now=0;const s=new Session({}, {now:()=>now,send:async(_e,_r,options)=>{now=130;return answer(options,'C');}});s.game=incoming('crush');s.warmed=true;await s.start();await s.decide(false);assert.equal(s.records[0].status,'unavailable');assert.equal(s.game.stats.actions,0);
});
test('wire failures stay visible; export omits the key and summaries omit warm-up',async()=>{
 const s=new Session({key:'never-export-this'},{send:async()=>{throw new Error('HTTP 500');}});await s.start();assert.equal(s.status,'error');assert.equal(s.records[0].status,'failed');assert.ok(!JSON.stringify(s.export()).includes('never-export-this'));
 const stats=summary([{source:'live',warmup:true,ms:800,status:'warmup'},{source:'live',ms:90,status:'applied',usage:{input_tokens:20,output_tokens:0}},{source:'live',status:'expired'}]);assert.equal(stats.p50,90);assert.equal(stats.expired,1);assert.equal(stats.input,20);
});
