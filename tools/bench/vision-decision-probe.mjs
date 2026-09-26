#!/usr/bin/env node
// Synthetic image-only grounding probe. No browser, game state, coordinates or answer labels enter the prompt.
import {deflateSync} from 'node:zlib';
import {mkdir,writeFile,readFile} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {randomUUID} from 'node:crypto';

const args={};for(let i=2;i<process.argv.length;i+=2)args[process.argv[i].replace(/^--/,'')]=process.argv[i+1];
const endpoint=args.endpoint||'http://127.0.0.1:8010',model=args.model||'qwen3.8-27b';
const route=args.route||'systemone';
if(!['systemone','chat'].includes(route))throw new Error('Use --route systemone or chat.');
const out=resolve(args.output||'out/arcade/vision-probe'),run=randomUUID();
const widths=(args.widths||'320,640,960').split(',').map(Number),count=Number(args.samples||12);
if(widths.some(width=>!Number.isInteger(width)||width<128||width>1280)||!Number.isInteger(count)||count<4||count>48)throw new Error('Use widths 128..1280 and 4..48 samples.');
await mkdir(out,{recursive:true});
const criteria={A:'Move left',B:'Move right',C:'Move up',D:'Move down',E:'Wait: no visible target'};
const instruction='Choose one movement command taking the cyan triangular drone toward the yellow circular ring. Ignore purple square decoys. Directions are screen-relative. If there is no visible yellow ring choose E. Use only the image as your observation.';
const fixedText=instruction+'\n'+Object.entries(criteria).map(([key,value])=>key+': '+value).join('\n')+'\nReturn only the option letter.';
const crcTable=Array.from({length:256},(_,value)=>{for(let k=0;k<8;k++)value=value&1?0xedb88320^(value>>>1):value>>>1;return value>>>0;});
function chunk(type,data){const tag=Buffer.from(type),length=Buffer.alloc(4),crc=Buffer.alloc(4);length.writeUInt32BE(data.length);let value=0xffffffff;for(const byte of Buffer.concat([tag,data]))value=crcTable[(value^byte)&255]^(value>>>8);crc.writeUInt32BE((value^0xffffffff)>>>0);return Buffer.concat([length,tag,data,crc]);}
function frame(width,index,{blank=false,swap=false}={}){
  const height=width*3/4|0,rows=Buffer.alloc(height*(width*3+1)),side=index%4;
  const cx=.43+((index*7)%9)/100,cy=.43+((index*3)%7)/100;
  const offsets=[[-.29,0],[.29,0],[0,-.29],[0,.29]],delta=offsets[side],tx=cx+delta[0],ty=cy+delta[1];
  const px=cx-delta[0],py=cy-delta[1];
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const u=x/width,v=y/height;let color=[17,25,32];
    if(!blank){
      if(x<width*.025||x>width*.975||y<height*.03||y>height*.97)color=[65,83,93];
      const ringX=swap?px:tx,ringY=swap?py:ty,squareX=swap?tx:px,squareY=swap?ty:py;
      const d=Math.hypot((u-ringX)*width,(v-ringY)*height);
      if(d<width*.047&&d>width*.027)color=[255,215,50];
      if(Math.abs(u-squareX)<.038&&Math.abs(v-squareY)<.05)color=[181,102,239];
      const dy=(v-cy)*height,dx=(u-cx)*width;
      if(dy>-width*.045&&dy<width*.035&&Math.abs(dx)<(dy+width*.045)*.52)color=[64,224,231];
    }
    const at=y*(width*3+1)+1+x*3;rows[at]=color[0];rows[at+1]=color[1];rows[at+2]=color[2];
  }
  const ihdr=Buffer.alloc(13);ihdr.writeUInt32BE(width);ihdr.writeUInt32BE(height,4);ihdr[8]=8;ihdr[9]=2;
  return {bytes:Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),chunk('IHDR',ihdr),chunk('IDAT',deflateSync(rows)),chunk('IEND',Buffer.alloc(0))]),width,height,expected:blank?'E':(swap?['B','A','D','C']:['A','B','C','D'])[side]};
}
const results=[];let sequence=0;
async function probe(route,kind,image,index,warmup=false){
  const tag='NInfer-VisionDecision-Probe/'+run+'/'+(++sequence),data='data:image/png;base64,'+image.bytes.toString('base64');
  const imagePart={type:'image_url',image_url:{url:data}};
  let body;
  if(route==='systemone'){
    body={model,state:kind==='state-parts'?[imagePart,{type:'text',text:instruction}]:instruction,questions:{move:{type:'choice',instructions:instruction,criteria}}};
    if(kind!=='state-parts'&&kind!=='no-image')body.images=[data];
  }else body={model,messages:[{role:'user',content:[...(kind==='no-image'?[]:[imagePart]),{type:'text',text:fixedText}]}],max_tokens:1,temperature:0,logprobs:true,logprob_candidates:Object.keys(criteria),prompt_cache_read_only:true,chat_template_kwargs:{enable_thinking:false}};
  const started=performance.now(),response=await fetch(endpoint+'/v1/'+(route==='systemone'?'systemone':'chat/completions'),{
    method:'POST',headers:{'Content-Type':'application/json','User-Agent':tag,...(process.env.NINFER_API_KEY?{Authorization:'Bearer '+process.env.NINFER_API_KEY}:{})},body:JSON.stringify(body),signal:AbortSignal.timeout(120000)});
  const wire=await response.json(),ms=performance.now()-started;
  if(!response.ok)throw new Error(JSON.stringify({route,kind,status:response.status,error:wire.detail??wire.error}));
  const answer=wire.answers?.move,position=wire.choices?.[0]?.logprobs?.content?.[0];
  const candidates=position?.candidate_logprobs;
  let choice=answer?.choice,probabilities=answer?.probabilities;
  if(!answer){
    if(!candidates?.length)throw new Error('Missing candidate distribution: '+JSON.stringify(wire));
    const scores=candidates.map(candidate=>candidate.raw_logprob??candidate.logprob),max=Math.max(...scores),weights=scores.map(value=>Math.exp(value-max)),sum=weights.reduce((a,b)=>a+b,0);
    probabilities=Object.fromEntries(candidates.map((candidate,i)=>[candidate.token,weights[i]/sum]));choice=Object.keys(probabilities).reduce((a,b)=>probabilities[a]>=probabilities[b]?a:b);
  }
  const filename=route+'-'+kind+'-'+image.width+'-'+index+'.png';
  await writeFile(join(out,filename),image.bytes);
  const row={tag,route,kind,index,warmup,width:image.width,height:image.height,frame:filename,expected:image.expected,choice,correct:choice===image.expected,ms,probabilities,usage:wire.usage,response:wire};
  results.push(row);console.log(JSON.stringify({route,kind,width:image.width,index,warmup,expected:row.expected,choice,ms:Math.round(ms)}));
}

for(const width of widths){
  await probe(route,'pixels',frame(width,100+width%7),-1,true);
  for(let i=0;i<count;i++)await probe(route,'pixels',frame(width,i),i);
}
// Same text and options, altered pixels. Correct answers must change with the ring, not the decoy or label.
for(let i=0;i<4;i++)await probe(route,'swapped-target',frame(320,i,{swap:true}),i);
for(let i=0;i<4;i++)await probe(route,'blank',frame(320,i,{blank:true}),i);
for(let i=0;i<4;i++)await probe(route,'no-image',frame(320,i),i);

if(args.log){
  const events=(await readFile(args.log,'utf8')).split('\n').flatMap(line=>{try{return [JSON.parse(line)];}catch{return [];}});
  for(const row of results){
    const own=events.filter(event=>event.request?.client===row.tag),start=own.find(event=>event.event==='request_start'),done=own.find(event=>event.event==='request_done');
    row.server={requestId:start?.request?.request_id,preparation:start?.preparation_seconds,timings:done?.timings_seconds,engine:done?.engine_timing};
  }
}
const groups={};for(const row of results.filter(row=>!row.warmup))(groups[row.route+'/'+row.kind+'/'+row.width]||=[]).push(row);
const summary=Object.entries(groups).map(([group,rows])=>{
  const sorted=rows.map(row=>row.ms).sort((a,b)=>a-b),p=q=>sorted[Math.ceil(q*sorted.length)-1];
  return {group,n:rows.length,correct:rows.filter(row=>row.correct).length,p50:p(.5),p95:p(.95),visionTokens:[...new Set(rows.map(row=>row.usage?.vision_tokens??row.server?.preparation?.vision_tokens))],visionMs:rows.map(row=>row.server?.timings?.vision!=null?row.server.timings.vision*1000:null)};
});
await writeFile(join(out,'report.json'),JSON.stringify({run,endpoint,model,instruction,criteria,method:'One-token closed-set decisions, full-response wall timing. Independent changing PNG frames. No game state in text. System One uses the NInfer images extension.',summary,results},null,2)+'\n');
console.log(JSON.stringify({output:out,summary},null,2));
