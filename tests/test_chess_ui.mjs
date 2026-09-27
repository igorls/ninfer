// Controller/DOM-port tests. These deliberately make no browser rendering claim.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
const html=await readFile(new URL('../docs/chess-demo.html',import.meta.url),'utf8');
const scripts=await Promise.all([...html.matchAll(/<script src="([^"]+)"/g)].map(m=>readFile(new URL('../docs/'+m[1],import.meta.url),'utf8')));
class Element {
  constructor(){this.children=[];this.listeners={};this.attributes={};this.dataset={};this.style={};this.className='';this.value='';this.textContent='';this.hidden=false;this.disabled=false;
    this.classList={add:(...names)=>{this.className=[...new Set([...this.className.split(' ').filter(Boolean),...names])].join(' ');},remove:(...names)=>{this.className=this.className.split(' ').filter(n=>!names.includes(n)).join(' ');},toggle:(name,on)=>{on?this.classList.add(name):this.classList.remove(name);}};}
  setAttribute(k,v){this.attributes[k]=String(v);}
  append(...nodes){for(const node of nodes)node.parentNode=this;this.children.push(...nodes);}
  replaceChildren(...nodes){for(const node of this.children)node.parentNode=null;this.children=[];this.append(...nodes);}
  remove(){if(this.parentNode){this.parentNode.children=this.parentNode.children.filter(node=>node!==this);this.parentNode=null;}}
  addEventListener(event,handler){(this.listeners[event]||=[]).push(handler);}
  async fire(event,detail={}){for(const handler of this.listeners[event]||[])await handler(detail);}
  querySelectorAll(){return this.children;}
  closest(){return null;}
  focus(){}
  click(){}
}
function goodFetch(url,init){
  const criteria=JSON.parse(init.body).questions.move.criteria,keys=Object.keys(criteria),choice=keys.find(k=>criteria[k].includes('; mate yes'))||keys[0];
  return Promise.resolve({ok:true,text:async()=>JSON.stringify({answers:{move:{type:'choice',choice,confidence:1,probabilities:Object.fromEntries(keys.map(k=>[k,k===choice?1:0]))}},usage:{input_tokens:500,output_tokens:0}})});
}
function app(search='',fetchImpl=goodFetch,clock=performance,{reducedMotion=true}={}){
  const animations=[],motionPreference=new Element();motionPreference.matches=reducedMotion;
  const createElement=()=>{
    const node=new Element();
    node.animate=(frames,options)=>{
      let complete,reject;
      const finished=new Promise((resolve,fail)=>{complete=resolve;reject=fail;});
      const animation={node,frames,options,finished,complete,cancelled:false,cancel(){this.cancelled=true;reject(new DOMException('Cancelled','AbortError'));}};
      animations.push(animation);return animation;
    };
    return node;
  };
  const elements=new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(m=>[m[1],createElement()]));
  for(const m of html.matchAll(/<input\b[^>]*id="([^"]+)"[^>]*>/g))elements.get(m[1]).value=m[0].match(/\bvalue="([^"]*)"/)?.[1]||'';
  for(const m of html.matchAll(/<select\b[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g))elements.get(m[1]).value=m[2].match(/value="([^"]+)"/)[1];
  for(const type of ['q','r','b','n']){const button=new Element();button.dataset.promotion=type;elements.get('promotion').append(button);}
  const document=new Element();document.getElementById=id=>{assert.ok(elements.has(id),'Missing element '+id);return elements.get(id);};document.createElement=createElement;
  const blobs=[],timers=new Set();class LocalURL extends URL{static createObjectURL(blob){blobs.push(blob);return 'blob:test';}static revokeObjectURL(){}}
  const context=vm.createContext({document,location:{search},URL:LocalURL,URLSearchParams,Blob,DOMException,AbortController,AbortSignal,performance:clock,fetch:fetchImpl,
    matchMedia:()=>motionPreference,
    requestAnimationFrame:()=>0,cancelAnimationFrame:()=>{},
    setTimeout:(fn,ms)=>{const timer=setTimeout(()=>{timers.delete(timer);fn();},ms);timers.add(timer);return timer;},clearTimeout:timer=>{clearTimeout(timer);timers.delete(timer);}});
  scripts.forEach(source=>vm.runInContext(source,context));
  return {$:id=>elements.get(id),square:id=>elements.get('board').children.find(node=>node.dataset.square===id),document,blobs,animations,motionPreference,dispose:()=>{for(const timer of timers)clearTimeout(timer);for(const animation of animations)animation.cancel();}};
}

test('live chess Step warms once, delivers checkmate, and exports exact evidence and a finished PGN',async()=>{
  let requests=0;const ui=app('?white=qwen&scenario=mate',(...args)=>{requests++;return goodFetch(...args);});
  try{
    ui.$('qwen-key').value='secret-test-token';await ui.$('step').fire('click');
    assert.equal(requests,2);assert.equal(ui.$('ply-count').textContent,'PLY 1');assert.equal(ui.$('mode').textContent,'GAME COMPLETE');
    assert.equal(ui.$('samples').textContent,1);assert.equal(ui.$('board-banner').hidden,false);
    await ui.$('export').fire('click');const text=await ui.blobs[0].text(),report=JSON.parse(text);
    assert.ok(!text.includes('secret-test-token'));assert.equal(report.decisions[0].san,'Rd8#');assert.equal(report.decisions[0].stages.length,1);
    assert.match(report.pgn,/\[Result "1-0"\]/);assert.match(report.pgn,/Rd8# 1-0/);
  }finally{ui.dispose();}
});

test('a human explicitly chooses underpromotion and makes no inference call',async()=>{
  const ui=app('?scenario=promotion',()=>{throw new Error('Unexpected inference');});
  try{
    const playing=ui.$('step').fire('click');await ui.square('a7').fire('click');await ui.square('a8').fire('click');
    assert.equal(ui.$('promotion').hidden,false);assert.equal(ui.$('ply-count').textContent,'PLY 0');
    await ui.$('promotion').children.find(b=>b.dataset.promotion==='n').fire('click');await playing;
    assert.equal(ui.$('ply-count').textContent,'PLY 1');assert.match(ui.$('current-fen').textContent,/^N6k/);
    assert.equal(ui.$('samples').textContent,0);assert.equal(ui.$('dt').textContent,'—');assert.equal(ui.$('on-time').textContent,'—');
  }finally{ui.dispose();}
});

test('Reset cancels warm-up and an eventual stale response cannot apply a move',async()=>{
  let release,signal;const ui=app('?white=qwen',(url,init)=>{signal=init.signal;return new Promise(resolve=>{release=()=>goodFetch(url,init).then(resolve);});});
  try{
    const old=ui.$('run').fire('click');assert.equal(ui.$('mode').textContent,'WARMING UP');await ui.$('reset').fire('click');
    assert.ok(signal.aborted);await release();await old;
    assert.equal(ui.$('ply-count').textContent,'PLY 0');assert.equal(ui.$('mode').textContent,'STANDBY');assert.equal(ui.$('samples').textContent,0);
  }finally{ui.dispose();}
});

test('a strict two-request timeout retains the completed first receipt and leaves the position unchanged',async()=>{
  let requests=0,lastSignal;const fen='7k/8/8/3Q4/8/8/2Q5/K3Q3 w - - 0 1';
  const ui=app('?white=qwen&strict=1&budget-ms=300&fen='+encodeURIComponent(fen),(url,init)=>{requests++;if(requests<3)return goodFetch(url,init);lastSignal=init.signal;return new Promise(()=>{});});
  try{
    await ui.$('step').fire('click');assert.equal(requests,3);assert.ok(lastSignal.aborted);
    assert.equal(ui.$('current-fen').textContent,fen);assert.equal(ui.$('ply-count').textContent,'PLY 0');assert.equal(ui.$('mode').textContent,'DEADLINE MISSED');
    assert.equal(ui.$('samples').textContent,0);assert.equal(ui.$('on-time').textContent,'0 / 1');assert.equal(ui.$('calls').textContent,1);
    await ui.$('export').fire('click');const report=JSON.parse(await ui.blobs[0].text());assert.equal(report.decisions[0].stages[0].kind,'piece');assert.equal(report.decisions[0].expired,true);
  }finally{ui.dispose();}
});

test('strict mode rejects a fully completed late response even before the timer callback runs',async()=>{
  let requests=0,tick=0;
  const ui=app('?white=qwen&strict=1&budget-ms=100',(url,init)=>{if(++requests===2)tick+=140;return goodFetch(url,init);},{now:()=>tick});
  try{
    const fen=ui.$('current-fen').textContent;await ui.$('step').fire('click');
    assert.equal(ui.$('mode').textContent,'DEADLINE MISSED');assert.equal(ui.$('current-fen').textContent,fen);
    assert.equal(ui.$('decision-title').textContent,'DEADLINE MISSED');assert.match(ui.$('decision-note').textContent,/No move applied/);
    await ui.$('export').fire('click');const record=JSON.parse(await ui.blobs[0].text()).decisions[0];
    assert.equal(record.expired,false);assert.equal(record.rejectedLate,true);assert.equal(record.placed,false);
    assert.equal(record.ms,140);assert.equal(record.stages.length,1);assert.ok(record.choice);
  }finally{ui.dispose();}
});

test('P pauses with a board button focused, without intercepting editable or modified shortcuts',async()=>{
  const ui=app();const target={closest:selector=>selector.split(',').includes('button')?{}:null};let prevented=0;
  try{
    const playing=ui.$('step').fire('click');
    await ui.document.fire('keydown',{key:'p',ctrlKey:true,target,preventDefault:()=>prevented++});assert.equal(ui.$('mode').textContent,'YOUR TURN');
    await ui.document.fire('keydown',{key:'p',target:{closest:()=>({})},preventDefault:()=>prevented++});assert.equal(ui.$('mode').textContent,'YOUR TURN');
    await ui.document.fire('keydown',{key:'p',target,preventDefault:()=>prevented++});await playing;
    assert.equal(ui.$('mode').textContent,'PAUSED');assert.equal(prevented,1);
  }finally{ui.dispose();}
});

test('local replay does not mutate the live position or invent inference measurements',async()=>{
  const ui=app('?white=reference&black=reference',()=>{throw new Error('Unexpected inference');});
  try{
    await ui.$('step').fire('click');await ui.$('step').fire('click');const fen=ui.$('current-fen').textContent;
    await ui.$('previous').fire('click');assert.equal(ui.$('ply-count').textContent,'PLY 1');assert.notEqual(ui.$('current-fen').textContent,fen);
    await ui.$('return-live').fire('click');assert.equal(ui.$('current-fen').textContent,fen);assert.equal(ui.$('samples').textContent,0);
    await ui.$('save-pgn').fire('click');assert.match(await ui.blobs[0].text(),/1\. /);
    await ui.$('flip').fire('click');assert.equal(ui.$('board').children[0].dataset.square,'h1');
  }finally{ui.dispose();}
});

test('a failed request can be retried and hiding the tab cancels a human turn',async()=>{
  let fail=true;const ui=app('?white=qwen',(...args)=>fail?Promise.resolve({ok:false,status:503,text:async()=>'unavailable'}):goodFetch(...args));
  try{
    await ui.$('step').fire('click');assert.match(ui.$('fault').textContent,/503/);assert.equal(ui.$('ply-count').textContent,'PLY 0');
    fail=false;await ui.$('step').fire('click');assert.equal(ui.$('ply-count').textContent,'PLY 1');assert.equal(ui.$('fault').textContent,'');
    ui.$('white').value='human';await ui.$('white').fire('change');const playing=ui.$('run').fire('click');
    ui.document.hidden=true;await ui.document.fire('visibilitychange');await playing;assert.equal(ui.$('mode').textContent,'PAUSED · TAB HIDDEN');
  }finally{ui.dispose();}
});

test('invalid custom FEN leaves a usable board and can be corrected',async()=>{
  const ui=app('?fen=broken');
  try{
    assert.match(ui.$('fault').textContent,/Invalid starting position/);assert.equal(ui.$('board').children.length,64);
    ui.$('start-fen').value='7k/8/8/8/8/8/8/K7 w - - 0 1';await ui.$('load-fen').fire('click');
    assert.equal(ui.$('fault').textContent,'');assert.match(ui.$('turn-state').textContent,/insufficient material/);
  }finally{ui.dispose();}
});

test('Jev and Qwen use separate endpoints, keys, warm-ups, per-player timings and returned model names',async()=>{
  let time=0;const calls=[];
  const ui=app('?match=jev-qwen',async(url,init)=>{
    const request=JSON.parse(init.body),jev=request.model==='jev-latest';
    assert.equal(url,jev?'http://127.0.0.1:8012/v1/systemone':'http://127.0.0.1:8010/v1/systemone');
    assert.equal(init.headers.Authorization,jev?'Bearer jev-secret':'Bearer qwen-secret');
    calls.push({url,request});time+=jev?100:20;
    const body=JSON.parse(await(await goodFetch(url,init)).text());body.model=jev?'jev-1.13.0':'qwen3.8-27b';body.usage.output_tokens=jev?12:0;
    return {ok:true,text:async()=>JSON.stringify(body)};
  },{now:()=>time});
  try{
    ui.$('jev-key').value='jev-secret';ui.$('qwen-key').value='qwen-secret';
    await ui.$('step').fire('click');await ui.$('step').fire('click');
    assert.equal(calls.length,4);assert.deepEqual(calls.map(c=>c.request.model),['jev-latest','jev-latest','qwen3.8-27b','qwen3.8-27b']);
    assert.equal(ui.$('ply-count').textContent,'PLY 2');assert.equal(ui.$('player-stats').children.length,2);
    assert.equal(ui.$('player-stats').children[0].children[2].textContent,'100.0');assert.equal(ui.$('player-stats').children[1].children[2].textContent,'20.0');
    assert.equal(ui.$('player-stats').children[0].children[4].textContent,'500 / 12');
    await ui.$('export').fire('click');const text=await ui.blobs[0].text(),report=JSON.parse(text);
    assert.ok(!text.includes('jev-secret'));assert.ok(!text.includes('qwen-secret'));assert.equal(report.players.w.samples,1);assert.equal(report.players.b.samples,1);
    assert.equal(report.decisions[0].player,'jev');assert.equal(report.decisions[1].player,'qwen');
    assert.equal(report.warmups.jev.ms,100);assert.equal(report.warmups.qwen.ms,20);assert.match(report.pgn,/\[White "jev-1.13.0"\]/);
    await ui.$('step').fire('click');assert.equal(calls.length,5,'a returning player does not warm again');
  }finally{ui.dispose();}
});

test('match preset and swap preserve provider credentials but reset position and measurements',async()=>{
  const ui=app('?white=reference&black=reference');
  try{
    ui.$('jev-key').value='jev-secret';ui.$('qwen-key').value='qwen-secret';await ui.$('step').fire('click');
    await ui.$('jev-v-qwen').fire('click');assert.equal(ui.$('white').value,'jev');assert.equal(ui.$('black').value,'qwen');assert.equal(ui.$('jev-connection').open,true);
    assert.equal(ui.$('ply-count').textContent,'PLY 0');const starting=ui.$('current-fen').textContent;
    await ui.$('swap-sides').fire('click');assert.equal(ui.$('white').value,'qwen');assert.equal(ui.$('black').value,'jev');
    assert.equal(ui.$('jev-key').value,'jev-secret');assert.equal(ui.$('qwen-key').value,'qwen-secret');assert.equal(ui.$('current-fen').textContent,starting);assert.equal(ui.$('samples').textContent,0);
  }finally{ui.dispose();}
});

test('failed Jev authorization pauses without awarding a move or substituting Qwen',async()=>{
  let requests=0;const ui=app('?match=jev-qwen',async(url)=>{requests++;assert.match(url,/:8012/);return {ok:false,status:401,text:async()=>'{"error":{"message":"Enter your TypeSafe key"}}'};});
  try{
    await ui.$('run').fire('click');assert.equal(requests,1);assert.equal(ui.$('ply-count').textContent,'PLY 0');assert.match(ui.$('fault').textContent,/JEV.*401.*TypeSafe key/);assert.equal(ui.$('mode').textContent,'REQUEST FAILED');
  }finally{ui.dispose();}
});

test('Jev rounded totals play successfully and the inspector preserves and explains them',async()=>{
  const captured=JSON.parse(await readFile(new URL('./fixtures/arcade/jev-rounded-choice.json',import.meta.url),'utf8'));
  const fen='r1bqk2r/pppp1ppp/3b4/4n3/3Pn3/8/PPP2PPP/RNBQKB1R w KQkq - 1 7';
  const ui=app('?match=jev-qwen&fen='+encodeURIComponent(fen),async()=>({ok:true,status:200,text:async()=>JSON.stringify(captured)}));
  try{
    await ui.$('step').fire('click');assert.equal(ui.$('ply-count').textContent,'PLY 1');assert.equal(ui.$('fault').textContent,'');
    assert.match(ui.$('distribution-note').textContent,/99\.00%/);
    assert.deepEqual(JSON.parse(ui.$('response-wire').textContent),captured);
  }finally{ui.dispose();}
});

test('an invalid Jev response replaces the previous receipt, keeps the board and timings, and can be retried',async()=>{
  let calls=0,fail=true;
  const ui=app('?white=jev&black=jev',async(url,init)=>{
    const response=await goodFetch(url,init);
    if(++calls===3&&fail){const body=JSON.parse(await response.text());for(const k in body.answers.move.probabilities)body.answers.move.probabilities[k]=0;return {ok:true,status:200,text:async()=>JSON.stringify(body)};}
    return response;
  });
  try{
    ui.$('jev-key').value='fixture-secret';await ui.$('step').fire('click');const before=ui.$('current-fen').textContent;
    await ui.$('step').fire('click');assert.equal(ui.$('current-fen').textContent,before);assert.equal(ui.$('ply-count').textContent,'PLY 1');
    assert.equal(ui.$('samples').textContent,1);assert.equal(ui.$('decision-title').textContent,'RESPONSE REJECTED');
    assert.match(ui.$('fault').textContent,/total 0\.00%/);assert.doesNotMatch(ui.$('fault').textContent,/key|relay/);
    assert.equal(ui.$('probabilities').children.length,0);assert.equal(ui.$('dt').textContent,'—');
    assert.ok(Object.values(JSON.parse(ui.$('response-wire').textContent).answers.move.probabilities).every(p=>p===0));
    await ui.$('export').fire('click');const text=await ui.blobs[0].text(),report=JSON.parse(text);
    assert.equal(report.failure.phase,'move');assert.equal(report.failure.beforeFen,before);assert.equal(report.decisions.length,1);assert.ok(!text.includes('fixture-secret'));
    fail=false;await ui.$('step').fire('click');assert.equal(ui.$('ply-count').textContent,'PLY 2');assert.equal(ui.$('fault').textContent,'');
  }finally{ui.dispose();}
});

test('Jev choice/probability disagreement plays Kf3, reports the discrepancy and preserves the receipt',async()=>{
  const body=JSON.parse(await readFile(new URL('./fixtures/arcade/jev-choice-disagreement.json',import.meta.url),'utf8'));
  const fen='3r2k1/3p2pp/3p4/4pp2/8/6K1/1PqP1P1P/2BR4 w - - 0 23';
  let requests=0;
  const ui=app('?match=jev-qwen&fen='+encodeURIComponent(fen),async(url,init)=>{
    const criteria=JSON.parse(init.body).questions.move.criteria;
    assert.match(criteria.D,/^Kf3 /);assert.match(criteria.O,/^Rg1 /);
    return ++requests===1?goodFetch(url,init):{ok:true,status:200,text:async()=>JSON.stringify(body)};
  });
  try{
    await ui.$('step').fire('click');
    assert.equal(requests,2,'no retry or replacement request for an inconsistent probability ranking');
    assert.equal(ui.$('ply-count').textContent,'PLY 1');assert.equal(ui.$('mode').textContent,'STEP COMPLETE');
    assert.equal(ui.$('fault').textContent,'');assert.equal(ui.$('decision-title').textContent,'Kf3');
    assert.equal(ui.$('current-fen').textContent,'3r2k1/3p2pp/3p4/4pp2/8/5K2/1PqP1P1P/2BR4 b - - 1 23');
    assert.match(ui.$('decision-note').textContent,/D \(12\.0%\).*O.*13\.0%/);
    assert.match(ui.$('distribution-note').textContent,/returned legal choice/);
    assert.equal(ui.$('candidate-table').children.length,16);
    assert.equal(ui.$('candidate-table').children.find(row=>row.className==='chosen').children[0].textContent,'D');
    assert.deepEqual(JSON.parse(ui.$('response-wire').textContent),body);
    await ui.$('export').fire('click');const report=JSON.parse(await ui.blobs[0].text());
    assert.equal(report.failure,null);assert.equal(report.decisions[0].choice,'g3f3');assert.equal(report.summary.samples,1);
    assert.match(report.decisions[0].stages[0].choiceWarning,/D \(12\.0%\)/);
    assert.equal(report.decisions[0].stages[0].response.answers.move.probabilities.O,.13);
  }finally{ui.dispose();}
});

test('a live move animates after commit, finishes before Step completes, and leaves inference timings unchanged',async()=>{
  let tick=0;const ui=app('?white=qwen&scenario=mate',goodFetch,{now:()=>tick},{reducedMotion:false});
  try{
    const playing=ui.$('step').fire('click');await new Promise(setImmediate);
    assert.equal(ui.$('ply-count').textContent,'PLY 1');assert.equal(ui.$('board-banner').hidden,true);
    assert.equal(ui.animations.length,1);assert.equal(ui.animations[0].frames[0].transform,'translate(0%, 700%)');
    assert.equal(ui.animations[0].options.duration,220);
    assert.match(ui.square('d8').className,/piece-in-flight/);
    tick=500;ui.animations[0].complete();await playing;
    assert.equal(ui.$('mode').textContent,'GAME COMPLETE');assert.equal(ui.$('board-banner').hidden,false);
    assert.equal(ui.$('board').children.length,64);assert.doesNotMatch(ui.square('d8').className,/piece-in-flight/);
    await ui.$('export').fire('click');const record=JSON.parse(await ui.blobs[0].text()).decisions[0];
    assert.equal(record.ms,0);assert.equal(record.decisionMs,0);assert.equal(record.placed,true);
  }finally{ui.dispose();}
});

test('captures fade the removed piece, including the separate en-passant square',async()=>{
  for(const fixture of [
    {fen:'k7/8/8/3r4/4P3/8/8/7K w - - 0 1',from:'e4',to:'d5',captureTop:'37.5%'},
    {fen:'k7/8/8/3pP3/8/8/8/7K w - d6 0 1',from:'e5',to:'d6',captureTop:'37.5%'},
  ]){
    const ui=app('?fen='+encodeURIComponent(fixture.fen),goodFetch,performance,{reducedMotion:false});
    try{
      const playing=ui.$('step').fire('click');await ui.square(fixture.from).fire('click');await ui.square(fixture.to).fire('click');
      await new Promise(setImmediate);assert.equal(ui.animations.length,2);
      const capture=ui.animations.find(a=>a.frames[0].opacity===1);
      assert.equal(capture.node.style.left,'37.5%');assert.equal(capture.node.style.top,fixture.captureTop);
      assert.equal(capture.frames[1].opacity,0);assert.equal(capture.options.delay+capture.options.duration,220);
      assert.match(capture.node.innerHTML,/piece black/);
      for(const animation of ui.animations)animation.complete();await playing;
      assert.equal(ui.$('board').children.length,64);assert.match(ui.square(fixture.to).attributes['aria-label'],/White pawn/);
    }finally{ui.dispose();}
  }
});

test('castling moves king and rook together in either board orientation',async()=>{
  for(const [to,flipped,kingShift,rookShift]of [['g1',false,-200,200],['c1',true,-200,300]]){
    const ui=app('?fen='+encodeURIComponent('4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1'),goodFetch,performance,{reducedMotion:false});
    try{
      if(flipped)await ui.$('flip').fire('click');
      const playing=ui.$('step').fire('click');await ui.square('e1').fire('click');await ui.square(to).fire('click');await new Promise(setImmediate);
      assert.equal(ui.animations.length,2);
      assert.equal(ui.animations[0].frames[0].transform,'translate('+kingShift+'%, 0%)');
      assert.equal(ui.animations[1].frames[0].transform,'translate('+rookShift+'%, 0%)');
      for(const animation of ui.animations)animation.complete();await playing;
      assert.equal(ui.$('board').children.length,64);assert.match(ui.square(to).attributes['aria-label'],/White king/);
      assert.match(ui.square(to==='g1'?'f1':'d1').attributes['aria-label'],/White rook/);
    }finally{ui.dispose();}
  }
});

test('promotion travels as a pawn and reveals the explicitly selected knight on arrival',async()=>{
  const ui=app('?scenario=promotion',goodFetch,performance,{reducedMotion:false});
  try{
    const pawn=ui.square('a7').innerHTML,playing=ui.$('step').fire('click');
    await ui.square('a7').fire('click');await ui.square('a8').fire('click');
    await ui.$('promotion').children.find(button=>button.dataset.promotion==='n').fire('click');await new Promise(setImmediate);
    assert.equal(ui.animations.length,1);assert.equal(ui.animations[0].node.innerHTML,pawn);
    assert.match(ui.square('a8').className,/piece-in-flight/);assert.match(ui.square('a8').attributes['aria-label'],/White knight/);
    ui.animations[0].complete();await playing;
    assert.doesNotMatch(ui.square('a8').className,/piece-in-flight/);assert.notEqual(ui.square('a8').innerHTML,pawn);
  }finally{ui.dispose();}
});

test('reset, pause, flip, replay, hiding the tab and a reduced-motion change settle active movement safely',async()=>{
  for(const action of ['reset','pause','flip','replay','hidden','reduced']){
    const ui=app('?white=reference&black=reference',goodFetch,performance,{reducedMotion:false});
    try{
      const playing=ui.$('step').fire('click');await new Promise(setImmediate);assert.ok(ui.animations.length);
      if(action==='hidden'){ui.document.hidden=true;await ui.document.fire('visibilitychange');}
      else if(action==='reduced'){ui.motionPreference.matches=true;await ui.motionPreference.fire('change');}
      else await ui.$(action==='pause'?'run':action==='replay'?'previous':action).fire('click');
      await playing;assert.ok(ui.animations.every(animation=>animation.cancelled));
      assert.equal(ui.$('board').children.length,64);assert.ok(ui.$('board').children.every(square=>!square.className.includes('piece-in-flight')));
      assert.equal(ui.$('ply-count').textContent,action==='reset'||action==='replay'?'PLY 0':'PLY 1');
      if(action==='reset')assert.equal(ui.$('mode').textContent,'STANDBY');
      if(action==='flip')assert.equal(ui.$('board').children[0].dataset.square,'h1');
    }finally{ui.dispose();}
  }
});

test('reduced motion keeps the immediate move and highlights without spatial animation',async()=>{
  const ui=app('?white=reference&black=reference');
  try{
    await ui.$('step').fire('click');assert.equal(ui.animations.length,0);assert.equal(ui.$('ply-count').textContent,'PLY 1');
    assert.equal(ui.$('board').children.filter(square=>square.className.includes('last-move')).length,2);
  }finally{ui.dispose();}
});
