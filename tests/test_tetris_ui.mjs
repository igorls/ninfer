// Controller tests with a minimal DOM port; these do not claim visual/browser coverage.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

const html = await readFile(new URL('../docs/tetris-demo.html', import.meta.url), 'utf8');
const scripts = await Promise.all([...html.matchAll(/<script src="([^"]+)"/g)].map(m=>readFile(new URL('../docs/'+m[1], import.meta.url),'utf8')));
class Element {
  constructor() {
    this.children = []; this.listeners = {}; this.attributes = {}; this.className = ''; this.style = {};
    this.value = ''; this.textContent = ''; this.disabled = false; this.hidden = false;
    this.classList = {
      add: (...names) => { this.className = [...new Set([...this.className.split(' ').filter(Boolean), ...names])].join(' '); },
      remove: (...names) => { this.className = this.className.split(' ').filter(n => !names.includes(n)).join(' '); },
      toggle: (name, on) => { on ? this.classList.add(name) : this.classList.remove(name); },
    };
  }
  setAttribute(k,v) { this.attributes[k] = String(v); }
  appendChild(child) { this.children.push(child); }
  append(...children) { this.children.push(...children); }
  insertBefore(child) { this.children.unshift(child); }
  replaceChildren(...children) { this.children = children; }
  addEventListener(event,handler) { (this.listeners[event] ||= []).push(handler); }
  async fire(event,detail = {}) { for (const handler of this.listeners[event] || []) await handler(detail); }
  closest() { return null; }
  click() {}
}

function goodFetch(url, init) {
  const request = JSON.parse(init.body), criteria = request.questions.move.criteria;
  const metrics = Object.entries(criteria).map(([key,desc])=>[key,desc.match(/\d+/g).map(Number)]);
  metrics.sort(([,a],[,b])=>b[0]-a[0]||a[1]-b[1]||a[2]-b[2]||a[3]-b[3]);
  const winner = metrics[0][0];
  const body = { answers:{move:{type:'choice',choice:winner,confidence:1,probabilities:Object.fromEntries(metrics.map(([key])=>[key,key===winner?1:0]))}},usage:{input_tokens:200,output_tokens:0}};
  return Promise.resolve({ ok:true,text:async()=>JSON.stringify(body) });
}

function app(search = '', fetchImpl = goodFetch, {motion = false} = {}) {
  const elements = new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(m=>[m[1],new Element()]));
  for (const m of html.matchAll(/<input\b[^>]*id="([^"]+)"[^>]*>/g)) elements.get(m[1]).value = m[0].match(/\bvalue="([^"]*)"/)?.[1] || '';
  for (const m of html.matchAll(/<select\b[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)) elements.get(m[1]).value = m[2].match(/value="([^"]+)"/)[1];
  const document = new Element();
  document.getElementById = id => { assert.ok(elements.has(id), 'Missing element '+id); return elements.get(id); };
  document.createElement = () => new Element();
  const blobs = [];
  class LocalURL extends URL {
    static createObjectURL(blob) { blobs.push(blob); return 'blob:test'; }
    static revokeObjectURL() {}
  }
  const timers = new Set();
  const frames = new Map();let frameId=0,virtualNow=0,reduced=!motion;
  const context = vm.createContext({
    document, location:{search}, URL:LocalURL, URLSearchParams, Blob, DOMException,
    AbortController, AbortSignal, performance:motion?{now:()=>virtualNow}:performance, fetch:fetchImpl,
    matchMedia:()=>({matches:reduced}),
    requestAnimationFrame:callback=>{
      if(!motion)return 0;
      const id=++frameId,timer=setTimeout(()=>{frames.delete(id);virtualNow+=16;callback(virtualNow);},1);
      frames.set(id,timer);return id;
    },
    cancelAnimationFrame:id=>{clearTimeout(frames.get(id));frames.delete(id);},
    setTimeout:(fn,ms)=>{const id=setTimeout(()=>{timers.delete(id);fn();},ms);timers.add(id);return id;},
    clearTimeout:id=>{clearTimeout(id);timers.delete(id);},
  });
  for (const script of scripts) vm.runInContext(script,context);
  return { $:id=>elements.get(id), document, blobs, setReduced:value=>{reduced=value;}, dispose:()=>{
    for(const id of timers)clearTimeout(id);
    for(const timer of frames.values())clearTimeout(timer);
  } };
}

async function waitFor(predicate) {
  for(let i=0;i<100;i++) {
    if(predicate())return;
    await new Promise(resolve=>setTimeout(resolve,2));
  }
  assert.fail('Timed out waiting for the expected UI state');
}

test('live Step warms once, places the chosen move, and exports exact wire evidence without the key', async () => {
  let requests=0;
  const ui=app('?scenario=trench',(...args)=>{requests++;return goodFetch(...args);});
  try {
    ui.$('api-key').value='private-test-key';
    await ui.$('step').fire('click');
    assert.equal(requests,2);
    assert.equal(ui.$('scoreboard').textContent,'4 LINES · 1 PIECES');
    assert.equal(ui.$('samples').textContent,1);
    assert.equal(ui.$('mode').textContent,'STEP COMPLETE');
    assert.equal(ui.$('inspector').open,true);
    await ui.$('step').fire('click');
    assert.equal(requests,3,'resume must not warm again');
    await ui.$('export').fire('click');
    const text=await ui.blobs[0].text(),report=JSON.parse(text);
    assert.ok(!text.includes('private-test-key'));
    assert.equal(report.decisions.length,2);
    assert.equal(report.decisions[0].cleared,4);
    assert.equal(report.decisions[0].response.answers.move.type,'choice');
  } finally { ui.dispose(); }
});

test('chosen piece moves through intermediate cells; speed changes presentation without changing the decision', async () => {
  async function atSpeed(speed,checkMotion=false) {
    const ui=app('?scenario=trench',goodFetch,{motion:true});
    try {
      ui.$('motion-speed').value=String(speed);
      await ui.$('motion-speed').fire('input');
      const playing=ui.$('step').fire('click');
      await waitFor(()=>ui.$('piece-motion').hidden===false);
      assert.equal(ui.$('motion-speed').disabled,false);
      if(checkMotion) {
        const before=ui.$('piece-motion').children.map(tile=>tile.style.transform).join(';');
        await new Promise(resolve=>setTimeout(resolve,8));
        const during=ui.$('piece-motion').children.map(tile=>tile.style.transform).join(';');
        assert.notEqual(during,before,'the selected piece should visibly travel before locking');
      }
      await playing;
      assert.equal(ui.$('piece-motion').hidden,true);
      assert.equal(ui.$('scoreboard').textContent,'4 LINES · 1 PIECES');
      await ui.$('export').fire('click');
      return JSON.parse(await ui.blobs[0].text()).decisions[0];
    } finally { ui.dispose(); }
  }
  const slow=await atSpeed(.5,true),fast=await atSpeed(3);
  assert.equal(slow.choice,fast.choice);
  assert.equal(slow.motionSpeedAtStart,.5);
  assert.equal(slow.motionSpeedAtEnd,.5);
  assert.equal(fast.motionSpeedAtStart,3);
  assert.equal(fast.motionSpeedAtEnd,3);
  assert.ok(slow.presentationMs>fast.presentationMs*3);
});

test('pausing during a landing settles the committed piece and resume starts the next one', async () => {
  const ui=app('?scenario=trench',goodFetch,{motion:true});
  try {
    ui.$('motion-speed').value='0.5';await ui.$('motion-speed').fire('input');
    const playing=ui.$('step').fire('click');
    await waitFor(()=>ui.$('piece-motion').hidden===false);
    await ui.$('run').fire('click');await playing;
    assert.equal(ui.$('mode').textContent,'PAUSED');
    assert.equal(ui.$('piece-motion').hidden,true);
    assert.equal(ui.$('scoreboard').textContent,'4 LINES · 1 PIECES');
    await ui.$('step').fire('click');
    await ui.$('export').fire('click');
    const report=JSON.parse(await ui.blobs[0].text());
    assert.equal(report.decisions.length,2);
    assert.equal(report.decisions[0].placed,true);
    assert.equal(report.decisions[0].presentationMs,null);
    assert.equal(report.decisions[1].piece,2);
  } finally { ui.dispose(); }
});

test('enabling reduced motion during a landing settles the committed board', async () => {
  const ui=app('?scenario=trench',goodFetch,{motion:true});
  try {
    ui.$('motion-speed').value='0.5';await ui.$('motion-speed').fire('input');
    const playing=ui.$('step').fire('click');
    await waitFor(()=>ui.$('piece-motion').hidden===false);
    ui.setReduced(true);await playing;
    assert.equal(ui.$('piece-motion').hidden,true);
    assert.equal(ui.$('scoreboard').textContent,'4 LINES · 1 PIECES');
    await ui.$('export').fire('click');
    const record=JSON.parse(await ui.blobs[0].text()).decisions[0];
    assert.equal(record.placed,true);
    assert.ok(record.presentationMs<620);
  } finally { ui.dispose(); }
});

test('mid-landing speed changes are labeled as initial and final settings in the export', async () => {
  const ui=app('',goodFetch,{motion:true});
  try {
    ui.$('motion-speed').value='0.5';await ui.$('motion-speed').fire('input');
    const playing=ui.$('step').fire('click');
    await waitFor(()=>ui.$('piece-motion').hidden===false);
    ui.$('motion-speed').value='3';await ui.$('motion-speed').fire('input');
    await playing;await ui.$('export').fire('click');
    const record=JSON.parse(await ui.blobs[0].text()).decisions[0];
    assert.equal(record.motionSpeedAtStart,.5);
    assert.equal(record.motionSpeedAtEnd,3);
    assert.equal(ui.$('motion-value').textContent,'3×');
  } finally { ui.dispose(); }
});

test('a rounded probability total is disclosed without changing the returned values', async () => {
  const ui=app('',async(...args)=>{
    const response=await goodFetch(...args),body=JSON.parse(await response.text());
    const probabilities=body.answers.move.probabilities;
    probabilities[body.answers.move.choice]=.99;
    return {ok:true,text:async()=>JSON.stringify(body)};
  });
  try {
    await ui.$('step').fire('click');
    assert.match(ui.$('decision-note').textContent,/Raw probabilities total 99\.0%/);
    await ui.$('export').fire('click');
    const record=JSON.parse(await ui.blobs[0].text()).decisions[0];
    assert.ok(Math.abs(record.probabilityTotal-.99)<1e-8);
    assert.equal(record.probabilities[record.choice],.99);
  } finally { ui.dispose(); }
});

test('Reset aborts warm-up; a late response cannot resurrect the old session', async () => {
  let release,signal;
  const ui=app('',(url,init)=>{signal=init.signal;return new Promise(resolve=>{release=()=>goodFetch(url,init).then(resolve);});});
  try {
    const old=ui.$('run').fire('click');
    assert.equal(ui.$('mode').textContent,'WARMING UP');
    await ui.$('reset').fire('click');
    assert.equal(signal.aborted,true);
    await release();await old;
    assert.equal(ui.$('scoreboard').textContent,'0 LINES · 0 PIECES');
    assert.equal(ui.$('mode').textContent,'STANDBY');
    assert.equal(ui.$('run').textContent,'RUN');
  } finally { ui.dispose(); }
});

test('a failed warm-up recovers on retry without poisoning a serialized run queue', async () => {
  let fail=true;
  const ui=app('',(...args)=>fail?Promise.resolve({ok:false,status:401,text:async()=>'{"error":{"message":"bad token"}}'}):goodFetch(...args));
  try {
    ui.$('api-key').value='private-test-key';
    await ui.$('step').fire('click');
    assert.match(ui.$('fault').textContent,/HTTP 401/);
    assert.equal(ui.$('mode').textContent,'REQUEST FAILED');
    assert.equal(ui.$('power-label').textContent,'REQUEST FAILED');
    assert.equal(ui.$('banner-title').textContent,'REQUEST FAILED');
    assert.equal(ui.$('inspector').open,true);
    assert.equal(ui.$('inspector-label').textContent,'INSPECT FAILED REQUEST');
    assert.equal(ui.$('wire-details').open,true);
    assert.match(ui.$('inspector-count').textContent,/FAILED REQUEST/);
    assert.match(ui.$('response-wire').textContent,/bad token/);
    assert.equal(ui.$('scoreboard').textContent,'0 LINES · 0 PIECES');
    assert.equal(ui.$('samples').textContent,0);
    assert.equal(ui.$('export').disabled,false);
    await ui.$('export').fire('click');
    const text=await ui.blobs[0].text(),report=JSON.parse(text);
    assert.ok(!text.includes('private-test-key'));
    assert.equal(report.decisions.length,0);
    assert.equal(report.failure.phase,'warmup');
    assert.equal(report.failure.status,401);
    assert.equal(report.failure.response.error.message,'bad token');
    assert.equal(ui.$('step').disabled,false);
    fail=false;await ui.$('step').fire('click');
    assert.equal(ui.$('scoreboard').textContent,'0 LINES · 1 PIECES');
    assert.equal(ui.$('fault').textContent,'');
    assert.equal(ui.$('power-label').textContent,'READY TO PLAY');
    assert.equal(ui.$('inspector-label').textContent,'INSPECT LAST DECISION');
    await ui.$('export').fire('click');
    assert.equal(JSON.parse(await ui.blobs[1].text()).failure,null);
  } finally { ui.dispose(); }
});

test('a failed decision preserves earlier measurements and replaces the stale inspector', async () => {
  let requests=0;
  const ui=app('',(...args)=>++requests===3
    ?Promise.resolve({ok:true,text:async()=>'{"answers":{"move":{"type":"choice","choice":"?"}}}'})
    :goodFetch(...args));
  try {
    await ui.$('step').fire('click');
    assert.equal(ui.$('samples').textContent,1);
    await ui.$('step').fire('click');
    assert.equal(ui.$('scoreboard').textContent,'0 LINES · 1 PIECES');
    assert.equal(ui.$('samples').textContent,1);
    assert.equal(ui.$('decision-title').textContent,'PIECE 2 · REQUEST FAILED');
    assert.equal(ui.$('selection-prob').textContent,'—');
    assert.equal(ui.$('response-wire').textContent.includes('"choice": "?"'),true);
    await ui.$('export').fire('click');
    const report=JSON.parse(await ui.blobs[0].text());
    assert.equal(report.decisions.length,1);
    assert.equal(report.failure.phase,'decision');
    assert.equal(report.failure.kind,'response');
    assert.equal(report.failure.piece,2);
  } finally { ui.dispose(); }
});

test('manual placement is playable with controls, remains untimed and sends no request', async () => {
  const ui=app('?player=human&scenario=trench',()=>{throw new Error('Manual mode must not call the API');});
  try {
    const playing=ui.$('step').fire('click');
    assert.equal(ui.$('mode').textContent,'YOUR TURN');
    assert.match(ui.$('manual-status').textContent,/Landing .* column .* rotation 0 degrees/);
    const firstPreview=ui.$('manual-status').textContent;
    await ui.$('rotate').fire('click');
    assert.notEqual(ui.$('manual-status').textContent,firstPreview);
    assert.match(ui.$('manual-status').textContent,/rotation \d+ degrees/);
    await ui.$('right').fire('click');
    await ui.$('drop').fire('click');
    await playing;
    assert.equal(ui.$('scoreboard').textContent,'4 LINES · 1 PIECES');
    assert.equal(ui.$('dt').textContent,'—');
    assert.equal(ui.$('on-time').textContent,'—');
    assert.equal(ui.$('source').textContent,'HUMAN PLAYER');
    assert.equal(ui.$('manual-status').textContent,'');
  } finally { ui.dispose(); }
});

test('reference play visibly identifies itself and produces no model probabilities', async () => {
  const ui=app('?player=reference',()=>{throw new Error('Reference must not call the API');});
  try {
    await ui.$('step').fire('click');
    assert.equal(ui.$('source').textContent,'LOCAL REFERENCE');
    assert.equal(ui.$('dt').textContent,'—');
    assert.equal(ui.$('selection-prob').textContent,'—');
    assert.equal(ui.$('samples').textContent,0);
    assert.match(ui.$('session-note').textContent,/NO INFERENCE/);
  } finally { ui.dispose(); }
});

test('strict challenge cancels late inference and locks only the spawn column', async () => {
  let requests=0,signal;
  const ui=app('?strict=1&row=4',(url,init)=>{
    requests++;
    if(requests===1)return goodFetch(url,init);
    signal=init.signal;
    return new Promise((resolve,reject)=>signal.addEventListener('abort',()=>reject(new DOMException('Aborted','AbortError'))));
  });
  try {
    await ui.$('step').fire('click');
    assert.equal(signal.aborted,true);
    assert.equal(ui.$('mode').textContent,'DEADLINE MISSED');
    assert.equal(ui.$('samples').textContent,0);
    assert.equal(ui.$('scoreboard').textContent,'0 LINES · 1 PIECES');
    await ui.$('export').fire('click');
    const report=JSON.parse(await ui.blobs[0].text());
    assert.equal(report.decisions[0].placed,false);
    assert.equal(report.decisions[0].choice,null);
    assert.equal(report.decisions[0].fallback.cells.length,4);
  } finally { ui.dispose(); }
});

test('manual keyboard controls preserve Space activation on focused buttons',async()=>{
  const ui=app('?player=human&scenario=trench');let prevented=0;
  const boardTarget={closest:()=>null};
  const buttonTarget={closest:selector=>selector.split(',').includes('button')?{}:null};
  try{
    const playing=ui.$('step').fire('click');
    await ui.document.fire('keydown',{key:' ',target:buttonTarget,preventDefault:()=>prevented++});
    assert.equal(prevented,0,'Space should activate the focused button natively');
    assert.equal(ui.$('mode').textContent,'YOUR TURN');
    for(const key of ['ArrowUp','ArrowRight',' '])await ui.document.fire('keydown',{key,target:boardTarget,preventDefault:()=>prevented++});
    await playing;assert.equal(ui.$('scoreboard').textContent,'4 LINES · 1 PIECES');assert.equal(prevented,3);
    const second=ui.$('step').fire('click');await ui.document.fire('keydown',{key:'p',target:buttonTarget,preventDefault:()=>prevented++});await second;
    assert.equal(ui.$('mode').textContent,'PAUSED');assert.equal(prevented,4);
  }finally{ui.dispose();}
});

test('hiding a tab pauses manual play; focused inputs do not trigger global shortcuts', async () => {
  const ui=app('?player=human');
  try {
    const playing=ui.$('run').fire('click');
    await ui.document.fire('keydown',{key:'r',target:{closest:()=>true}});
    assert.equal(ui.$('mode').textContent,'YOUR TURN');
    ui.document.hidden=true;
    await ui.document.fire('visibilitychange');
    await playing;
    assert.equal(ui.$('mode').textContent,'PAUSED · TAB HIDDEN');
    assert.equal(ui.$('scoreboard').textContent,'0 LINES · 0 PIECES');
  } finally { ui.dispose(); }
});
