#!/usr/bin/env node
// Executes the browser's exact chess adapter and System One requests, without rendering a browser.
import {readFile,writeFile} from 'node:fs/promises';
import vm from 'node:vm';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';

export async function loadChess(overrides={}) {
  const sources=await Promise.all(['vendor/chess.js','system-one.js','chess-game.js'].map(path=>readFile(new URL('../../docs/arcade/'+path,import.meta.url),'utf8')));
  return vm.runInNewContext(sources.join('\n')+'; ({...ChessGame,SystemOne,ChessRules})',{
    fetch,performance,AbortController,AbortSignal,setTimeout,clearTimeout,...overrides,
  });
}

export async function evaluate({endpoint,model='qwen3.8-27b',scenario='opening',fen,plies=20,apiKey=''}) {
  const C=await loadChess(),game=new C.Game(fen||C.positions[scenario]),connection={endpoint,model,apiKey},records=[];
  const options=game.options();
  let warmupMs=null;
  if(options.length){const plan=C.plan(game,options);const warm=await C.SystemOne.decide(endpoint,C.requestFor(game,plan,model),plan.options,{apiKey,signal:AbortSignal.timeout(30000)});warmupMs=warm.ms;}
  for(let ply=0;ply<plies&&!game.status().over;ply++){
    const started=performance.now(),beforeFen=game.fen,options=game.options(),reference=C.reference(options);
    const result=await C.choose(game,options,connection,{signal:AbortSignal.timeout(30000)});
    game.commit(result.option);
    records.push({source:'live',beforeFen,afterFen:game.fen,legalMoves:options.length,choice:result.option.id,san:result.option.san,
      stages:result.stages,ms:result.ms,decisionMs:performance.now()-started,policyMatch:result.option.referenceScore===reference.referenceScore});
  }
  return {schema:'ninfer.chess.eval.v1',at:new Date().toISOString(),config:{endpoint,model,scenario,initialFen:game.initialFen,plies},warmupMs,
    methodology:'Sequential local HTTP self-play. Warm-up excluded. API time sums all requests for one move. Every legal move is retained; positions above 62 use piece then move selection. Reference agreement is a one-ply material heuristic, not chess accuracy.',
    summary:{...C.SystemOne.summary(records),plies:game.ply,requests:records.reduce((n,r)=>n+r.stages.length,0),status:game.status().text},decisions:records,pgn:game.rules.pgn()};
}

if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href){
  try {
    const args={};
    for(let i=2;i<process.argv.length;i+=2){const key=process.argv[i];if(!['--endpoint','--model','--scenario','--fen','--plies','--output'].includes(key)||!process.argv[i+1])throw new Error('Expected --endpoint URL [--model label] [--scenario opening|mate|fork|promotion] [--fen FEN] [--plies 1..300] [--output path]');args[key.slice(2)]=process.argv[i+1];}
    if(!args.endpoint)throw new Error('--endpoint is required');
    if(args.scenario&&!['opening','mate','fork','promotion'].includes(args.scenario))throw new Error('Unknown scenario');
    const plies=Number(args.plies||20);if(!Number.isInteger(plies)||plies<1||plies>300)throw new Error('--plies must be 1..300');
    const report=await evaluate({...args,plies,apiKey:process.env.NINFER_API_KEY||''});
    if(args.output)await writeFile(args.output,JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify({config:report.config,summary:report.summary,pgn:report.pgn},null,2));
  }catch(error){console.error(error.message);process.exitCode=1;}
}
