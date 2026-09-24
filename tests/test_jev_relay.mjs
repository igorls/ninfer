import test from 'node:test';
import assert from 'node:assert/strict';
import {once} from 'node:events';
import {createJevRelay} from '../tools/arcade/jev-relay.mjs';

async function relay(config,run){const server=createJevRelay(config);server.listen(0,'127.0.0.1');await once(server,'listening');try{await run('http://127.0.0.1:'+server.address().port);}finally{server.closeAllConnections();await new Promise(resolve=>server.close(resolve));}}
test('local page preflight succeeds, unknown origins/routes are rejected, and no-key requests fail without inference',async()=>{
  await relay({fetchImpl:()=>{throw new Error('Unexpected upstream');}},async url=>{
    const options=await fetch(url+'/v1/systemone',{method:'OPTIONS',headers:{Origin:'null','Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'authorization,content-type'}});
    assert.equal(options.status,204);assert.equal(options.headers.get('access-control-allow-origin'),'null');
    assert.equal((await fetch(url+'/v1/systemone',{method:'POST',headers:{Origin:'https://unrelated.example'}})).status,403);
    assert.equal((await fetch(url+'/arbitrary-route')).status,404);
    const missing=await fetch(url+'/v1/systemone',{method:'POST',body:'{}'});assert.equal(missing.status,401);assert.match((await missing.json()).error.message,/TypeSafe key/);
  });
});

test('relay forwards an exact decision to official Jev and preserves response usage/status without exposing the key',async()=>{
  const response=JSON.stringify({model:'jev-1.13.0',answers:{move:{type:'choice',choice:'A',confidence:1,probabilities:{A:1}}},usage:{input_tokens:120,output_tokens:12}}),payload=JSON.stringify({model:'jev-latest',state:'chess',questions:{move:{type:'choice',criteria:{A:'a4'}}}});
  await relay({fetchImpl:async(url,init)=>{
    assert.equal(url,'https://api.typesafe.ai/v1/systemone');assert.equal(init.body,payload);assert.equal(init.headers.Authorization,'Bearer page-key');assert.equal(init.redirect,'error');
    return new Response(response,{status:200,headers:{'Content-Type':'application/json'}});
  }},async url=>{
    const result=await fetch(url+'/v1/systemone',{method:'POST',headers:{Origin:'null',Authorization:'Bearer page-key'},body:payload});
    assert.equal(result.status,200);assert.equal(await result.text(),response);
    const health=await(await fetch(url+'/health')).text();assert.ok(!health.includes('environment-key'));assert.ok(!health.includes('page-key'));
  });
});

test('opaque origins cannot borrow an environment credential; caller-key failures remain actionable',async()=>{
  const previous=process.env.TYPESAFE_API_KEY;process.env.TYPESAFE_API_KEY='ambient-test-key';let calls=0;
  try{
    await relay({fetchImpl:async(url,init)=>{calls++;assert.equal(init.headers.Authorization,'Bearer caller-key');return new Response('{"error":{"message":"Invalid key"}}',{status:401});}},async url=>{
      const missing=await fetch(url+'/v1/systemone',{method:'POST',headers:{Origin:'null','Content-Type':'text/plain'},body:'{}'});assert.equal(missing.status,401);assert.equal(calls,0);
      const result=await fetch(url+'/v1/systemone',{method:'POST',headers:{Authorization:'Bearer caller-key'},body:'{}'});assert.equal(result.status,401);assert.match(await result.text(),/Invalid key/);assert.equal(calls,1);
    });
  }finally{if(previous==null)delete process.env.TYPESAFE_API_KEY;else process.env.TYPESAFE_API_KEY=previous;}
});

test('client disconnect aborts the upstream decision',async()=>{
  let entered;const started=new Promise(resolve=>{entered=resolve;});let canceled;
  const aborted=new Promise(resolve=>{canceled=resolve;});
  await relay({fetchImpl:async(url,{signal})=>{entered();return new Promise((resolve,reject)=>signal.addEventListener('abort',()=>{canceled();reject(signal.reason);},{once:true}));}},async url=>{
    const controller=new AbortController(),pending=fetch(url+'/v1/systemone',{method:'POST',headers:{Authorization:'Bearer caller-key'},body:'{}',signal:controller.signal});
    await started;controller.abort();await assert.rejects(pending,{name:'AbortError'});await aborted;
  });
});
