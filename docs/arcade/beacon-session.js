// Only encoded pixels cross this boundary. No game object enters request construction.
const BeaconSession = (() => {
  const state='Pilot the cyan triangular drone to the yellow circular beacon. Gray rectangles and the border are solid walls. Red disks are moving hazards. Screen directions: up is toward the top of the image. The camera image is your only observation.';
  const instructions='First locate the CYAN TRIANGLE (your drone), then the YELLOW RING (your target). Compare their positions relative to EACH OTHER, never to the image center. Which screen direction goes FROM THE TRIANGLE TO THE RING? Up means decreasing screen y, down increasing screen y. If the ring is below the triangle, choose down or a down diagonal, never up. If it is above, choose up or an up diagonal, never down. Steer around solid gray walls and away from red hazards. The triangle is not a direction arrow. If either object is absent, brake. Return the option token.';
  function request(model,image){return {model,state,images:[image],questions:{move:{type:'choice',instructions,
    criteria:Object.fromEntries(BeaconGame.actions.map(a=>[a.code,a.label==='Brake'?'Brake / wait':`Move ${a.label.toLowerCase()} for 220 milliseconds`]))}}};}
  class Camera {
    constructor(){this.reset();}
    reset(){this.frames=[];this.frozen=null;this.last=-Infinity;}
    push(frame,now){if(now-this.last<80)return;this.last=now;this.frames.push({...frame,capturedAt:now});while(this.frames.length&&this.frames[0].capturedAt<now-1600)this.frames.shift();}
    sample(mode,now,width){
      if(mode==='blackout')return {...BeaconGame.render(null,width,true),capturedAt:now};
      if(mode==='delay')return this.frames.findLast(f=>f.capturedAt<=now-1000)||null;
      const latest=this.frames.at(-1);if(!latest)return null;
      if(mode==='freeze'){this.frozen??=latest;return this.frozen;}this.frozen=null;return latest;
    }
  }
  class Client {
    constructor({send=SystemOne.decide,now=()=>performance.now()}={}){this.send=send;this.now=now;this.records=[];this.pending=false;this.epoch=0;}
    cancel(){this.epoch++;this.controller?.abort();}
    async decide(frame,config,{warmup=false}={}){
      if(this.pending)return null;
      this.pending=true;const epoch=this.epoch,started=this.now();this.controller=new AbortController();
      const wire=request(config.model,frame.dataURL);
      const row={request:wire,mode:config.mode,width:frame.width,height:frame.height,capturedAt:frame.capturedAt,started,warmup};
      try {
        const answer=await this.send(config.endpoint,wire,BeaconGame.actions,{apiKey:config.key,signal:AbortSignal.any([this.controller.signal,AbortSignal.timeout(warmup?30000:5000)])});
        if(epoch!==this.epoch)return null;
        Object.assign(row,{response:answer.body,ms:answer.ms,received:this.now(),action:answer.option.code});
        row.frameAge=row.received-frame.capturedAt;
        if(!Number.isInteger(answer.usage.vision_tokens)||answer.usage.vision_tokens<=0)
          throw new Error('This server did not confirm native image processing. Use the vision-enabled System One build; expected usage.vision_tokens > 0.');
        row.visionTokens=answer.usage.vision_tokens;
        row.dropped=!warmup&&row.received-started>config.deadline;
        row.status=warmup?'warmup':row.dropped?'late — dropped':'applied';
        return row;
      } catch(error){
        if(epoch!==this.epoch)return null;
        row.status='failed';row.error=error.message;row.response??=error.receipt?.response??null;row.ms??=this.now()-started;
        throw Object.assign(error,{record:row});
      } finally {
        if(epoch===this.epoch){this.records.push(row);if(this.records.length>600)this.records.shift();}
        this.pending=false;
      }
    }
  }
  function summary(records){
    const rows=records.filter(r=>!r.warmup&&r.status!=='failed'),values=rows.map(r=>r.ms).sort((a,b)=>a-b);
    const p=q=>values.length?values[Math.ceil(q*values.length)-1]:null;
    return {n:rows.length,p50:p(.5),p95:p(.95),dropped:rows.filter(r=>r.dropped).length};
  }
  return {request,Camera,Client,summary};
})();
