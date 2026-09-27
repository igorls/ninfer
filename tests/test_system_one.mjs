import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {loadChess} from '../tools/bench/chess-eval.mjs';

const {SystemOne:S}=await loadChess();
// Real Jev 1.13.0 response, September 22: all 35 options present, total 0.99.
const captured=JSON.parse(await readFile(new URL('./fixtures/arcade/jev-rounded-choice.json',import.meta.url),'utf8'));
const options=Object.keys(captured.answers.move.probabilities).map(code=>({code}));
const request={model:'jev-latest',questions:{move:{type:'choice',criteria:Object.fromEntries(options.map(o=>[o.code,o.code]))}}};
const reply=body=>async()=>({ok:true,status:200,text:async()=>JSON.stringify(body)});

test('accepts Jev approximate probabilities without changing the choice or wire values',async()=>{
  const decoded=await S.decide('http://localhost',request,options,{fetchImpl:reply(captured)});
  assert.equal(decoded.option.code,'B');
  assert.equal(decoded.choiceWarning,null);
  assert.ok(Math.abs(decoded.probabilityTotal-.99)<1e-10);
  assert.deepEqual(JSON.parse(JSON.stringify(decoded.probabilities)),captured.answers.move.probabilities);
  assert.deepEqual(JSON.parse(JSON.stringify(decoded.body)),captured);
});

test('approximate totals do not excuse missing options, invalid numbers or materially wrong mass',()=>{
  for(const total of [.98,1.02]){
    const body=structuredClone(captured);body.answers.move.probabilities.B=total-.05;
    assert.doesNotThrow(()=>S.decodeChoice(body,options));
  }
  for(const mutate of [
    p=>{delete p.Q;},p=>{delete p.Q;p.unknown=0;},p=>{p.Q=-.01;},p=>{p.Q=NaN;},
    p=>{p.B=1.01;},p=>{p.B=.85;},p=>{p.B=.99;},p=>{p.B=0;},
  ]){const body=structuredClone(captured);mutate(body.answers.move.probabilities);assert.throws(()=>S.decodeChoice(body,options));}
});

test('a legal returned choice remains authoritative when Jev probabilities disagree',async()=>{
  const body=JSON.parse(await readFile(new URL('./fixtures/arcade/jev-choice-disagreement.json',import.meta.url),'utf8'));
  const options=Object.keys(body.answers.move.probabilities).sort().map(code=>({code}));
  const capturedRequest={model:'jev-latest',questions:{move:{type:'choice',criteria:Object.fromEntries(options.map(o=>[o.code,o.code]))}}};
  const decoded=await S.decide('http://localhost',capturedRequest,options,{fetchImpl:reply(body)});
  assert.equal(decoded.option.code,'D');
  assert.match(decoded.choiceWarning,/D \(12\.0%\).*O.*13\.0%/);
  assert.deepEqual(JSON.parse(JSON.stringify(decoded.body)),body);
  const tied=structuredClone(body);tied.answers.move.probabilities.O=.12;tied.answers.move.probabilities.A=.10;
  assert.equal(S.decodeChoice(tied,options).choiceWarning,null);
  body.answers.move.choice='unknown';assert.throws(()=>S.decodeChoice(body,options),/unknown action/);
});

test('a rejected response carries an inspectable receipt without authentication headers',async()=>{
  const body=structuredClone(captured);body.answers.move.probabilities.B=.5;
  await assert.rejects(S.decide('http://localhost',request,options,{apiKey:'fixture-secret',fetchImpl:reply(body)}),error=>{
    assert.equal(error.kind,'response');assert.equal(error.status,200);
    assert.deepEqual(JSON.parse(JSON.stringify(error.receipt.response)),body);
    assert.deepEqual(JSON.parse(JSON.stringify(error.receipt.request)),request);
    assert.ok(!JSON.stringify(error).includes('fixture-secret'));return true;
  });
});
