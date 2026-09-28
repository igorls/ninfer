// One outstanding decision, a real running clock, and an epoch per pause/reset.
const ArenaSession = (() => {
  'use strict';
  class Session {
    constructor(config={}, {send=SystemOne.decide,now=()=>performance.now()}={}){
      this.config={mode:'qwen',endpoint:'http://127.0.0.1:8010',model:'qwen3.8-27b',key:'',deadline:300,seed:1,difficulty:'duel',...config};
      this.game=new ArenaGame.Game(this.config);this.send=send;this.now=now;this.status='ready';this.records=[];this.epoch=0;this.pending=null;this.warmed=false;this.next=0;this.error=null;
    }
    advance(){if(this.status!=='running')return;const now=this.now();this.game.tick(Math.max(0,now-this.last));this.last=now;if(this.game.over){this.status='finished';this.cancel();}}
    cancel(){this.epoch++;this.pending?.controller.abort();this.pending=null;}
    pause(){this.advance();if(this.status==='finished')return;this.status='paused';this.cancel();}
    async start(){
      if(this.status==='running'||this.status==='warming'||this.status==='finished')return;
      this.error=null;
      if(this.config.mode==='qwen'&&!this.warmed){const epoch=this.epoch;this.status='warming';await this.decide(true);if(this.status!=='warming'||this.epoch!==epoch)return;this.warmed=true;}
      this.status='running';this.last=this.now();this.next=this.now();
    }
    pump(){
      this.advance();if(this.status!=='running'||this.now()<this.next)return;
      if(this.config.mode==='qwen'){if(!this.pending&&this.game.legal().length)void this.decide(false);}
      else if(this.config.mode==='reference'){
        const code=ArenaGame.reference(this.game);if(code){this.game.act(code);this.records.push({source:'reference',at:this.game.time,action:code,status:'applied'});this.next=this.now()+140;}
      }
    }
    human(code){this.advance();if(this.status!=='running'||this.config.mode!=='human')return false;const applied=this.game.act(code);if(applied)this.records.push({source:'human',at:this.game.time,action:code,status:'applied'});return applied;}
    async decide(warmup){
      if(this.pending)return;const options=this.game.legal();if(!options.length)return;
      const job={controller:new AbortController(),epoch:this.epoch};this.pending=job;
      const wire=ArenaGame.request(this.config.model,this.game,options),start=this.now();
      const row={source:'live',warmup,at:this.game.time,request:wire,status:'pending'};
      try{
        const answer=await SystemOne.withDeadline(signal=>this.send(this.config.endpoint,wire,options,{signal,apiKey:this.config.key}),warmup?30000:this.config.deadline,job.controller.signal);
        if(job.epoch!==this.epoch)return;
        this.advance();if(job.epoch!==this.epoch)return;
        row.elapsed=this.now()-start;
        if(answer.expired){row.status='expired';if(warmup)throw new Error('Warm-up exceeded 30 seconds. Check the server and retry.');}
        else{
          Object.assign(row,{response:answer.body,ms:answer.ms,action:answer.option.code,probabilities:answer.probabilities,warning:answer.choiceWarning,usage:answer.usage});
          // Recheck elapsed time as well: a busy browser can delay the deadline timer callback.
          row.status=warmup?'warmup':row.elapsed>this.config.deadline?'expired':this.game.act(answer.option.code)?'applied':'unavailable';
        }
        this.records.push(row);this.next=this.now()+40;
      }catch(error){
        if(job.epoch!==this.epoch)return;
        row.status='failed';row.error=error.message;row.response=error.receipt?.response??null;row.elapsed=this.now()-start;this.records.push(row);
        this.error=error.message;this.status='error';this.cancel();
      }finally{if(this.pending===job)this.pending=null;}
    }
    export(){const {key,...config}=this.config;return {game:'Arena Duel',version:1,config,status:this.status,result:this.game.result,elapsed:this.game.time,health:{player:this.game.player.hp,enemy:this.game.enemy.hp},stats:{...this.game.stats},events:this.game.events,records:this.records};}
  }
  function summary(records){
    const live=records.filter(r=>r.source==='live'&&!r.warmup),done=live.filter(r=>Number.isFinite(r.ms)),times=done.map(r=>r.ms).sort((a,b)=>a-b);
    const p=q=>times.length?times[Math.ceil(q*times.length)-1]:null;
    return {samples:done.length,applied:live.filter(r=>r.status==='applied').length,expired:live.filter(r=>r.status==='expired').length,unavailable:live.filter(r=>r.status==='unavailable').length,p50:p(.5),p95:p(.95),input:done.reduce((n,r)=>n+(r.usage?.input_tokens||0),0),output:done.reduce((n,r)=>n+(r.usage?.output_tokens||0),0)};
  }
  return {Session,summary};
})();
