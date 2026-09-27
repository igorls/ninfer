#!/usr/bin/env node
import {readFile,writeFile} from 'node:fs/promises';
import vm from 'node:vm';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
export async function loadKitchen(overrides={}){
  const sources=await Promise.all(['system-one.js','kitchen-game.js','kitchen-match.js'].map(path=>readFile(new URL('../../docs/arcade/'+path,import.meta.url),'utf8')));
  return vm.runInNewContext(sources.join('\n')+'; ({...KitchenGame,...KitchenMatch,SystemOne})',{fetch,performance,URL,DOMException,AbortController,AbortSignal,setTimeout,clearTimeout,...overrides});
}
export async function evaluate({endpoint='http://127.0.0.1:8010',model='qwen3.8-27b',player='reference',partner='reference',layout='comparison',mode='paused',seed=1,seconds=90,goal='revenue',apiKey=''}){
  const K=await loadKitchen(),config={seed,duration:seconds*1000,goal,mode,layout,players:[player,partner],profiles:{qwen:{endpoint,model,apiKey}}};
  const match=new K.Match(config,mode==='paused'?{sleep:async()=>{}}:{});await match.start();
  return match.report();
}
if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href){
  try{
    const args={};for(let i=2;i<process.argv.length;i+=2){const key=process.argv[i].slice(2);if(!['endpoint','model','player','partner','layout','mode','seed','seconds','goal','output'].includes(key)||!process.argv[i+1])throw new Error('Expected --player reference|qwen --partner reference|qwen --layout comparison|cooperative --mode paused|realtime --seconds 30..120 --seed N --goal revenue|waste|vip [--endpoint URL] [--model LABEL] [--output FILE]');args[key]=process.argv[i+1];}
    for(const key of ['seed','seconds'])if(key in args)args[key]=Number(args[key]);
    if(args.player&&!['reference','qwen'].includes(args.player)||args.partner&&!['reference','qwen'].includes(args.partner)||args.layout&&!['comparison','cooperative'].includes(args.layout)||args.mode&&!['paused','realtime'].includes(args.mode)||args.goal&&!['revenue','waste','vip'].includes(args.goal))throw new Error('Unknown player, layout, mode or goal.');
    if(args.seed!=null&&(!Number.isInteger(args.seed)||args.seed<0||args.seed>4294967295)||args.seconds!=null&&(!Number.isInteger(args.seconds)||args.seconds<30||args.seconds>120))throw new Error('Invalid seed or duration.');
    const report=await evaluate({...args,apiKey:process.env.NINFER_API_KEY||''});
    if(args.output)await writeFile(args.output,JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify({config:report.config,failure:report.failure,kitchens:report.kitchens.map(lane=>({player:lane.player,time:lane.time,stats:lane.stats,summary:lane.summary,stale:lane.stale})),chefs:report.chefs?.map(({decisions,...actor})=>actor)},null,2));
    if(report.failure)process.exitCode=1;
  }catch(error){console.error(error.message);process.exitCode=1;}
}
