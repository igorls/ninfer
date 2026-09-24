// Minimal DOM ports test controller behavior, not layout or browser rendering.
import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFile} from 'node:fs/promises';
const html=await readFile(new URL('../docs/beacon-demo.html',import.meta.url),'utf8');
const scripts=await Promise.all([...html.matchAll(/<script src="([^"]+)"/g)].map(m=>readFile(new URL('../docs/'+m[1],import.meta.url),'utf8')));
class Element {
  constructor(){this.value='';this.textContent='';this.children=[];this.listeners={};this.classList={toggle(){}};this.hidden=false;this.disabled=false;}
  append(...nodes){this.children.push(...nodes);}replaceChildren(...nodes){this.children=nodes;}setAttribute(){}
  addEventListener(name,fn){(this.listeners[name]??=[]).push(fn);}
  fire(name,event={}){this['on'+name]?.(event);for(const fn of this.listeners[name]||[])fn(event);}
  getContext(){return {putImageData:data=>{this.data=data;}};}
  toDataURL(){return 'data:image/png;base64,'+Buffer.from(this.data.pixels).toString('base64');}
  click(){this.fire('click');}
}
function app({vision=80,hold=false}={}){
  const nodes=new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(m=>[m[1],new Element()]));
  for(const m of html.matchAll(/<input\b[^>]*id="([^"]+)"[^>]*>/g))nodes.get(m[1]).value=m[0].match(/\bvalue="([^"]*)"/)?.[1]||'';
  for(const m of html.matchAll(/<select\b[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)){const opts=[...m[2].matchAll(/<option\b([^>]*)>/g)];nodes.get(m[1]).value=(opts.find(o=>/\bselected\b/.test(o[1]))||opts[0])[1].match(/value="([^"]+)"/)[1];}
  const document=new Element();document.getElementById=id=>{assert.ok(nodes.has(id),id);return nodes.get(id);};document.createElement=()=>new Element();document.querySelectorAll=()=>[];
  const calls=[],blobs=[];let raf,now=0,release;
  class LocalURL{static createObjectURL(b){blobs.push(b);return 'blob:test';}static revokeObjectURL(){}}
  const context=vm.createContext({document,ImageData:class{constructor(pixels,width,height){Object.assign(this,{pixels,width,height});}},performance:{now:()=>now},requestAnimationFrame:fn=>{raf=fn;},AbortSignal,AbortController,Blob,URL:LocalURL,setTimeout:fn=>fn(),clearTimeout,
    fetch:async(url,init)=>{calls.push({url,init});if(hold)await new Promise(r=>{release=r;});const request=JSON.parse(init.body);return {ok:true,status:200,text:async()=>JSON.stringify({model:'qwen',answers:{move:{type:'choice',choice:'B',confidence:1,probabilities:Object.fromEntries(Object.keys(request.questions.move.criteria).map(k=>[k,k==='B'?1:0]))}},usage:{input_tokens:100,output_tokens:0,vision_tokens:vision}})};}});
  scripts.forEach(s=>vm.runInContext(s,context));
  return {$:id=>nodes.get(id),document,calls,blobs,frame:async(ms=100)=>{now+=ms;raf(now);await new Promise(setImmediate);},release:()=>release?.()};
}
async function exported(ui){ui.$('export').click();return JSON.parse(await ui.blobs.at(-1).text());}
test('run verifies vision, moves only after a decision, shows exact sent image and exports no key',async()=>{
  const ui=app();ui.$('key').value='do-not-export';ui.$('run').click();await ui.frame();
  assert.equal(ui.calls.length,1);assert.equal(ui.$('vision-tokens').textContent,80);assert.match(ui.$('action-detail').textContent,/Warm-up/);
  assert.equal(ui.$('observation').src,JSON.parse(ui.calls[0].init.body).images[0]);
  await ui.frame(250);assert.equal(ui.calls.length,2);assert.equal(ui.$('action-detail').textContent,'applied');
  const data=await exported(ui);assert.equal(data.records.length,2);assert.ok(!JSON.stringify(data).includes('do-not-export'));assert.equal(data.records[0].warmup,true);
  ui.$('reset').click();assert.equal(ui.$('mode').textContent,'READY');assert.equal(ui.$('export').disabled,true);assert.equal(ui.$('observation').hidden,true);
});
test('an old server fails visibly with raw response and no world advance',async()=>{
  const ui=app({vision:0});ui.$('run').click();await ui.frame();assert.equal(ui.$('mode').textContent,'REQUEST FAILED');assert.match(ui.$('fault').textContent,/native image processing/);
  const data=await exported(ui);assert.equal(data.score.elapsed,0);assert.equal(data.records[0].status,'failed');assert.equal(data.records[0].response.usage.vision_tokens,0);
});
test('camera change cancels pending response and hidden tabs pause human play',async()=>{
  const ui=app({hold:true});ui.$('run').click();await ui.frame();ui.$('camera-mode').value='freeze';ui.$('camera-mode').fire('change');ui.release();await new Promise(setImmediate);
  assert.equal(ui.$('mode').textContent,'SETTINGS CHANGED');assert.equal((await exported(ui)).records.length,0);
  ui.$('pilot').value='human';ui.$('pilot').fire('change');ui.$('run').click();await ui.frame();ui.$('manual-actions').children[1].click();assert.equal(ui.$('export').disabled,false);
  ui.document.hidden=true;ui.document.fire('visibilitychange');assert.equal(ui.$('mode').textContent,'TAB HIDDEN');assert.equal(ui.$('run').textContent,'RESUME');
});
test('invalid seed prevents requests and can be corrected',async()=>{
  const ui=app();ui.$('seed').value='-1';ui.$('run').click();await ui.frame();assert.equal(ui.calls.length,0);assert.match(ui.$('fault').textContent,/whole-number seed/);
  ui.$('seed').value='0';ui.$('run').click();await ui.frame();assert.equal(ui.calls.length,1);
});
