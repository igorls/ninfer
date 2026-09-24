#!/usr/bin/env node
// API-only local relay. TypeSafe rejects browser requests from file:// origins.
import {createServer} from 'node:http';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';

const upstream='https://api.typesafe.ai/v1/systemone';
function localOrigin(origin){
  if(!origin||origin==='null')return true;
  try{const url=new URL(origin);return ['http:','https:'].includes(url.protocol)&&['localhost','127.0.0.1','[::1]'].includes(url.hostname);}
  catch{return false;}
}
export function createJevRelay({fetchImpl=fetch}={}){
  return createServer(async(req,res)=>{
    const json=(status,message)=>{res.writeHead(status,{'Content-Type':'application/json'});res.end(JSON.stringify({error:{message}}));};
    if(!localOrigin(req.headers.origin)){json(403,'The Jev relay accepts local browser origins only.');return;}
    res.setHeader('Access-Control-Allow-Origin',req.headers.origin||'*');res.setHeader('Vary','Origin');
    res.setHeader('Access-Control-Allow-Methods','POST, OPTIONS');res.setHeader('Access-Control-Allow-Headers','Authorization, Content-Type');
    res.setHeader('Cache-Control','no-store');
    if(req.url==='/health'&&req.method==='GET'){res.writeHead(200,{'Content-Type':'application/json'});res.end(JSON.stringify({status:'ok',upstream}));return;}
    if(req.url!=='/v1/systemone'){json(404,'Use POST /v1/systemone.');return;}
    if(req.method==='OPTIONS'){res.writeHead(204);res.end();return;}
    if(req.method!=='POST'){json(405,'Use POST /v1/systemone.');return;}
    // A null browser origin also includes sandboxed remote pages. Never grant ambient credentials
    // based on Origin; every caller supplies its own key, including a local file page.
    const authorization=req.headers.authorization;
    if(!authorization||!/^Bearer\s+\S+$/i.test(authorization)){json(401,'Enter your TypeSafe key under Jev connection. Every request must supply its own bearer key.');return;}
    const controller=new AbortController();
    const cancel=()=>{if(!res.writableEnded)controller.abort();};
    res.on('close',cancel);req.on('aborted',cancel);
    try{
      const chunks=[];let size=0;
      for await(const chunk of req){size+=chunk.length;if(size>262144){json(413,'Decision request exceeds 256 KiB.');return;}chunks.push(chunk);}
      const body=Buffer.concat(chunks).toString('utf8');
      try{JSON.parse(body);}catch{json(400,'Request body must be JSON.');return;}
      const response=await fetchImpl(upstream,{method:'POST',headers:{'Content-Type':'application/json',Authorization:authorization},body,
        redirect:'error',signal:AbortSignal.any([controller.signal,AbortSignal.timeout(30000)])});
      const text=await response.text();
      if(res.destroyed)return;
      res.writeHead(response.status,{'Content-Type':response.headers.get('content-type')||'application/json'});res.end(text);
    }catch(error){
      if(res.destroyed)return;
      json(error.name==='TimeoutError'?504:502,error.name==='TimeoutError'?'TypeSafe did not complete the request within 30 seconds.':'The relay could not reach TypeSafe. Check its network connection.');
    }finally{res.off('close',cancel);req.off('aborted',cancel);}
  });
}

if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href){
  let port=8012;
  if(process.argv.length>2){
    if(process.argv.length!==4||process.argv[2]!=='--port'||!/^\d+$/.test(process.argv[3])){console.error('Usage: node tools/arcade/jev-relay.mjs [--port 8012]');process.exit(1);}
    port=Number(process.argv[3]);if(port<1||port>65535){console.error('Port must be 1..65535.');process.exit(1);}
  }
  const server=createJevRelay();
  server.on('error',error=>{console.error('Jev relay could not listen: '+error.message);process.exitCode=1;});
  server.listen(port,'127.0.0.1',()=>console.log('Jev API relay: http://127.0.0.1:'+port+' -> '+upstream+' (no request or credential logging)'));
  for(const signal of ['SIGINT','SIGTERM'])process.on(signal,()=>{server.close();server.closeAllConnections();});
}
