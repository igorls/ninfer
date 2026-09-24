#!/usr/bin/env node
// Exercises the shipped simulation and camera pixels directly; this is not browser UI rendering.
import vm from 'node:vm';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {deflateSync} from 'node:zlib';
import {resolve,dirname} from 'node:path';
import {pathToFileURL} from 'node:url';
export async function loadBeacon(){
  const context=vm.createContext({fetch,performance,AbortController,AbortSignal,setTimeout,clearTimeout});
  for(const file of ['system-one','beacon-game','beacon-session'])vm.runInContext(await readFile(new URL(`../../docs/arcade/${file}.js`,import.meta.url),'utf8'),context);
  return vm.runInContext('({Game:BeaconGame,Session:BeaconSession,SystemOne})',context);
}
const crcTable=Array.from({length:256},(_,v)=>{for(let k=0;k<8;k++)v=v&1?0xedb88320^(v>>>1):v>>>1;return v>>>0;});
function chunk(type,data){const tag=Buffer.from(type),size=Buffer.alloc(4),crc=Buffer.alloc(4);size.writeUInt32BE(data.length);let v=0xffffffff;for(const b of Buffer.concat([tag,data]))v=crcTable[(v^b)&255]^(v>>>8);crc.writeUInt32BE((v^0xffffffff)>>>0);return Buffer.concat([size,tag,data,crc]);}
export function png({pixels,width,height}){
  const rows=Buffer.alloc(height*(width*4+1));for(let y=0;y<height;y++)rows.set(pixels.subarray(y*width*4,(y+1)*width*4),y*(width*4+1)+1);
  const ihdr=Buffer.alloc(13);ihdr.writeUInt32BE(width);ihdr.writeUInt32BE(height,4);ihdr[8]=8;ihdr[9]=6;
  return Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),chunk('IHDR',ihdr),chunk('IDAT',deflateSync(rows)),chunk('IEND',Buffer.alloc(0))]);
}
export function observation(Game,game,width=320,blank=false){return {width,height:width*3/4,capturedAt:performance.now(),dataURL:'data:image/png;base64,'+png(Game.render(game,width,blank)).toString('base64')};}
export async function evaluate({endpoint='http://127.0.0.1:8014',model='qwen3.8-27b',width=320,seconds=15,mode='live',seed=1}={}){
  const {Game,Session}=await loadBeacon(),game=Game.create(seed),client=new Session.Client(),camera=new Session.Camera();
  const config={endpoint,model,mode,deadline:500,key:process.env.NINFER_API_KEY||''};
  await client.decide(observation(Game,game,width,mode==='blackout'),config,{warmup:true});
  let previous=performance.now(),lastSample=0;
  const timer=setInterval(()=>{const now=performance.now();Game.tick(game,now-previous);previous=now;if(now-lastSample>=80){camera.push(Game.render(game,width),now);lastSample=now;}},10);
  try {
    const end=performance.now()+seconds*1000;
    while(performance.now()<end){
      const frame=camera.sample(mode,performance.now(),width);
      if(!frame){await new Promise(r=>setTimeout(r,50));continue;}
      const row=await client.decide({...frame,dataURL:'data:image/png;base64,'+png(frame).toString('base64')},config);
      if(row&&!row.dropped)Game.command(game,row.action);else Game.brake(game);
      await new Promise(r=>setTimeout(r,Math.max(30,220-(row?.ms||0))));
    }
  }finally{clearInterval(timer);Game.brake(game);}
  return {config:{endpoint,model,width,seconds,mode,seed},score:{beacons:game.collected,contacts:game.hits,elapsed:game.time},summary:Session.summary(client.records),records:client.records};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href){
  try {
    const args={};for(let i=2;i<process.argv.length;i+=2){const key=process.argv[i].replace(/^--/,'');if(!['endpoint','model','width','seconds','mode','seed','output'].includes(key)||!process.argv[i+1])throw new Error('Expected --endpoint URL --model LABEL --width 320|640|960 --seconds 5..90 --mode live|blackout|freeze|delay --seed UINT32 --output FILE');args[key]=process.argv[i+1];}
    for(const k of ['width','seconds','seed'])if(k in args)args[k]=Number(args[k]);
    if(args.width&&!([320,640,960].includes(args.width))||args.seconds!=null&&(!Number.isInteger(args.seconds)||args.seconds<5||args.seconds>90)||args.mode&&!['live','blackout','freeze','delay'].includes(args.mode)||args.seed!=null&&(!Number.isInteger(args.seed)||args.seed<0||args.seed>4294967295))throw new Error('Invalid evaluation settings.');
    const report=await evaluate(args);if(args.output){await mkdir(dirname(resolve(args.output)),{recursive:true});await writeFile(args.output,JSON.stringify(report,null,2)+'\n');}
    console.log(JSON.stringify({config:report.config,score:report.score,summary:report.summary},null,2));
  }catch(error){console.error(error.message);process.exitCode=1;}
}
