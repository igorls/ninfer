// Exercise the shipped controller through a minimal DOM port, not a browser renderer.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
const html=await readFile(new URL('../docs/kitchen-demo.html',import.meta.url),'utf8');
const scripts=await Promise.all([...html.matchAll(/<script src="([^"]+)"/g)].map(m=>readFile(new URL('../docs/'+m[1],import.meta.url),'utf8')));
class Element{
  constructor(){this.children=[];this.listeners={};this.attributes={};this.style={};this.className='';this.value='';this.textContent='';this.disabled=false;this.hidden=false;
    this.classList={toggle:(name,on)=>{const names=new Set(this.className.split(' ').filter(Boolean));on?names.add(name):names.delete(name);this.className=[...names].join(' ');}};}
  setAttribute(key,value){this.attributes[key]=String(value);}
  append(...nodes){for(const node of nodes){node.remove();node.parentNode=this;this.children.push(node);}}
  replaceChildren(...nodes){for(const node of this.children)node.parentNode=null;this.children=[];this.append(...nodes);}
  remove(){if(this.parentNode){this.parentNode.children=this.parentNode.children.filter(node=>node!==this);this.parentNode=null;}}
  addEventListener(event,handler){(this.listeners[event]||=[]).push(handler);}
  async fire(event,detail={}){if(event==='click'&&this.disabled)return;for(const handler of this.listeners[event]||[])await handler(detail);}
  closest(){return null;}
  click(){}
}
function goodFetch(url,init){
  const request=JSON.parse(init.body),keys=Object.keys(request.questions.move.criteria);
  return Promise.resolve({ok:true,status:200,text:async()=>JSON.stringify({model:request.model,answers:{move:{type:'choice',choice:keys[0],confidence:1,probabilities:Object.fromEntries(keys.map((key,i)=>[key,i===0?1:0]))}},usage:{input_tokens:200,output_tokens:7}})});
}
function app(search='',fetch=goodFetch){
  const elements=new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(m=>[m[1],new Element()]));
  for(const m of html.matchAll(/<input\b[^>]*id="([^"]+)"[^>]*>/g))elements.get(m[1]).value=m[0].match(/\bvalue="([^"]*)"/)?.[1]||'';
  for(const m of html.matchAll(/<select\b[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)){
    const options=[...m[2].matchAll(/<option\b([^>]*)>/g)],selected=options.find(option=>/\bselected\b/.test(option[1]))||options[0];elements.get(m[1]).value=selected[1].match(/value="([^"]+)"/)[1];
  }
  const document=new Element();document.getElementById=id=>{assert.ok(elements.has(id),'Missing element '+id);return elements.get(id);};document.createElement=()=>new Element();
  const blobs=[],timers=new Set();class LocalURL extends URL{static createObjectURL(blob){blobs.push(blob);return 'blob:test';}static revokeObjectURL(){}}
  const context=vm.createContext({document,location:{search},URL:LocalURL,URLSearchParams,Blob,DOMException,AbortController,AbortSignal,performance,fetch,
    setTimeout:(fn,ms)=>{const timer=setTimeout(()=>{timers.delete(timer);fn();},ms);timers.add(timer);return timer;},clearTimeout:timer=>{clearTimeout(timer);timers.delete(timer);}});
  scripts.forEach(source=>vm.runInContext(source,context));
  return {$:id=>elements.get(id),lane:index=>elements.get('kitchens').children[index],document,blobs,dispose:()=>{for(const timer of timers)clearTimeout(timer);}};
}
const turn=()=>new Promise(setImmediate),find=(root,cls)=>root.children.find(node=>node.className.split(' ').includes(cls));
const textTree=node=>[node.textContent,...node.children.map(textTree)].join(' ');
async function report(ui){await ui.$('export').fire('click');return JSON.parse(await ui.blobs.at(-1).text());}

test('local preset is playable without HTTP, steps both kitchens, inspects jobs and exports equal untimed results',async()=>{
  const ui=app('?mode=paused',()=>{throw new Error('Unexpected API call');});
  try{
    assert.equal(ui.$('duration').value,'90000');assert.equal(ui.$('kitchens').children.length,2);
    await ui.$('local-demo').fire('click');await ui.$('step').fire('click');
    assert.equal(ui.$('mode').textContent,'STEP COMPLETE');assert.equal(ui.$('shift-progress').value,500);
    assert.ok(ui.$('candidate-table').children.length>0);assert.equal(ui.$('request-wire').textContent,'No API request.');
    const result=await report(ui);assert.equal(result.kitchens[0].time,500);assert.equal(result.kitchens[1].time,500);
    assert.deepEqual(result.kitchens[0].stats,result.kitchens[1].stats);assert.equal(result.kitchens[0].summary.samples,0);
    assert.match(textTree(find(ui.lane(0),'kitchen-probs')),/Greedy local policy/);
    await ui.$('reset').fire('click');assert.equal(ui.$('mode').textContent,'READY');assert.equal(ui.$('export').disabled,true);assert.equal(ui.$('candidate-table').children.length,0);
  }finally{ui.dispose();}
});

test('Jev preset keeps profiles separate, shows raw receipts, usage and probabilities, and excludes keys from export',async()=>{
  const calls=[],ui=app('?mode=paused',(url,init)=>{calls.push({url,init});return goodFetch(url,init);});
  try{
    await ui.$('jev-match').fire('click');ui.$('jev-key').value='jev-test-secret';await ui.$('jev-key').fire('change');ui.$('qwen-key').value='qwen-test-secret';await ui.$('qwen-key').fire('change');
    await ui.$('step').fire('click');assert.equal(calls.length,4);
    for(const {url,init}of calls)assert.equal(init.headers.Authorization,url.includes('8012')?'Bearer jev-test-secret':'Bearer qwen-test-secret');
    assert.equal(ui.$('jev-connection').open,true);assert.equal(JSON.parse(ui.$('response-wire').textContent).model,'jev-latest');
    assert.match(textTree(find(ui.lane(0),'lane-timings')),/TOKENS IN\/OUT.*200 \/ 7/);
    assert.ok(ui.$('candidate-table').children.some(row=>row.className==='chosen'));
    ui.$('inspect-lane').value='1';await ui.$('inspect-lane').fire('change');assert.equal(JSON.parse(ui.$('request-wire').textContent).model,'qwen3.8-27b');
    const result=await report(ui);assert.equal(result.kitchens[0].summary.samples,1);assert.equal(result.kitchens[1].summary.samples,1);assert.ok(!JSON.stringify(result).includes('secret'));
  }finally{ui.dispose();}
});

test('paused human mode renders legal job buttons before waiting, preserves them through goal changes and commits the chosen job',async()=>{
  const ui=app('?mode=paused&left=human&right=reference',()=>{throw new Error('Unexpected API call');});
  try{
    const playing=ui.$('step').fire('click');await turn();const manual=find(ui.lane(0),'manual-jobs');
    assert.equal(manual.hidden,false);assert.ok(manual.children.length>1);assert.equal(ui.$('shift-progress').value,0);
    const button=manual.children[1];ui.$('goal').value='vip';await ui.$('goal').fire('change');assert.equal(manual.children[1],button);
    const label=button.textContent;await button.fire('click');await playing;
    const result=await report(ui);assert.equal(result.kitchens[0].decisions[0].label,label);assert.equal(result.kitchens[0].decisions[0].source,'human');assert.equal(result.kitchens[0].summary.samples,0);
    assert.equal(result.goalHistory.at(-1).goal,'vip');assert.equal(manual.hidden,true);
  }finally{ui.dispose();}
});

test('hiding the tab pauses a human round and removes job controls without advancing the clock',async()=>{
  const ui=app('?mode=paused&left=human&right=reference');
  try{
    const running=ui.$('run').fire('click');await turn();ui.document.hidden=true;await ui.document.fire('visibilitychange');await running;
    assert.equal(ui.$('mode').textContent,'PAUSED · TAB HIDDEN');assert.equal(ui.$('shift-progress').value,0);assert.equal(find(ui.lane(0),'manual-jobs').hidden,true);
    assert.equal(ui.$('session-settings').disabled,false);
  }finally{ui.dispose();}
});

test('failed provider response pauses both kitchens with the exact receipt and can be retried',async()=>{
  let count=0;const ui=app('?mode=paused&left=qwen&right=reference',async(url,init)=>{
    if(++count===2)return {ok:true,status:200,text:async()=>JSON.stringify({answers:{move:{choice:'invalid'}}})};return goodFetch(url,init);
  });
  try{
    await ui.$('step').fire('click');assert.equal(ui.$('mode').textContent,'REQUEST FAILED');assert.equal(ui.$('inspector').open,true);
    assert.equal(JSON.parse(ui.$('response-wire').textContent).answers.move.choice,'invalid');assert.match(ui.$('fault').textContent,/QWEN/);assert.equal(ui.$('shift-progress').value,0);
    const failed=await report(ui);assert.equal(failed.kitchens[0].summary.samples,0);assert.ok(failed.failure);
    await ui.$('step').fire('click');assert.equal(ui.$('fault').textContent,'');assert.equal(ui.$('shift-progress').value,500);
  }finally{ui.dispose();}
});

test('invalid seed disables play and keyboard start until corrected without breaking page controls',async()=>{
  let requests=0;const ui=app('?seed=wrong&mode=paused',(...args)=>{requests++;return goodFetch(...args);});
  try{
    assert.match(ui.$('fault').textContent,/Choose a seed/);assert.equal(ui.$('run').disabled,true);
    await ui.document.fire('keydown',{key:'p',target:new Element(),preventDefault(){}});assert.equal(requests,0);
    ui.$('seed').value='0';await ui.$('seed').fire('change');await ui.$('local-demo').fire('click');await ui.$('step').fire('click');
    assert.equal(ui.$('fault').textContent,'');assert.equal(ui.$('shift-progress').value,500);assert.equal((await report(ui)).config.seed,0);
  }finally{ui.dispose();}
});

test('cooperative mode renders one shared world, two actor readouts and conflicts, then restores comparison cleanly',async()=>{
  const ui=app('?layout=cooperative&mode=paused&left=reference&right=reference');
  try{
    assert.equal(ui.$('shared-kitchen').hidden,false);assert.equal(ui.$('left-label').textContent,'Ada · first chef');assert.equal(ui.$('right-label').textContent,'Bo · second chef');
    assert.equal(ui.$('jev-match').textContent,'Jev + Qwen');assert.ok(find(ui.$('shared-world'),'kitchen-map'));assert.equal(find(ui.lane(0),'kitchen-map'),undefined);assert.equal(find(ui.lane(1),'kitchen-map'),undefined);
    await ui.$('step').fire('click');assert.equal(ui.$('shift-progress').value,500);assert.match(ui.$('team-metrics').textContent,/CONFLICTS 1/);
    assert.match(textTree(ui.$('activity-log')),/Bo rejected/);assert.match(textTree(find(ui.lane(0),'kitchen-heading')),/ADA/);assert.match(textTree(find(ui.lane(1),'kitchen-heading')),/BO/);
    ui.$('inspect-lane').value='1';await ui.$('inspect-lane').fire('change');assert.match(ui.$('decision-note').textContent,/Conflict/);
    const data=await report(ui);assert.equal(data.kitchens.length,1);assert.equal(data.chefs.length,2);
    ui.$('layout').value='comparison';await ui.$('layout').fire('change');assert.equal(ui.$('shared-kitchen').hidden,true);assert.equal(ui.$('team-activity').hidden,true);
    assert.ok(find(ui.lane(0),'kitchen-map'));assert.ok(find(ui.lane(1),'kitchen-map'));assert.equal(ui.$('left-label').textContent,'Left kitchen');assert.equal(ui.$('export').disabled,true);
  }finally{ui.dispose();}
});

test('cooperative human controls only its assigned chef and the paired receipt updates after commit',async()=>{
  const ui=app('?layout=cooperative&mode=paused&left=human&right=reference');
  try{
    const running=ui.$('step').fire('click');await turn();const manual=find(ui.lane(0),'manual-jobs');
    assert.ok(manual.children.every(button=>button.textContent.startsWith('Ada:')||button.textContent.startsWith('Wait')));
    ui.$('inspect-lane').value='1';await ui.$('inspect-lane').fire('change');assert.match(ui.$('decision-note').textContent,/Waiting for the paired round/);
    await manual.children[0].fire('click');await running;assert.match(ui.$('decision-note').textContent,/Conflict/);assert.equal(ui.$('shift-progress').value,500);
    assert.equal(find(ui.lane(1),'manual-jobs').hidden,true);
  }finally{ui.dispose();}
});
