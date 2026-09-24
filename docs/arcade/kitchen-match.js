// Shared or paired kitchen clocks and independent decision lifecycles, without DOM ownership.
const KitchenMatch=(()=>{
  const live=player=>player==='qwen'||player==='jev';
  const publicProfile=profile=>({endpoint:profile.endpoint,model:profile.model});
  class Match{
    constructor(config,{onChange=()=>{},now=()=>performance.now(),sleep=(ms,signal)=>new Promise((resolve,reject)=>{
      const abort=()=>{clearTimeout(timer);reject(new DOMException('Paused','AbortError'));};
      const timer=setTimeout(()=>{signal.removeEventListener('abort',abort);resolve();},ms);
      signal.addEventListener('abort',abort,{once:true});
    })}={}){
      this.onChange=onChange;this.now=now;this.sleep=sleep;this.epoch=0;this.running=false;this.reset(config);
    }
    reset(config=this.config){
      this.pause();this.config={...config,players:[...config.players]};
      this.cooperative=config.layout==='cooperative';this.round=0;
      this.games=Array.from({length:this.cooperative?1:2},()=>new KitchenGame.Game(config));
      this.lanes=config.players.map((player,index)=>({player,chefId:this.cooperative?index:null,game:this.games[this.cooperative?0:index],records:[],last:null,prepared:null,pending:false,status:'READY',nextAt:0,attempts:0,cancelled:0,human:null}));
      this.warmups={};this.failure=null;this.goalHistory=[{at:0,goal:config.goal}];this.phase='READY';this.notify();
    }
    notify(){this.onChange(this);}
    setGoal(goal){if(!KitchenGame.goals[goal])throw new Error('Unknown kitchen goal.');this.config.goal=goal;this.goalHistory.push({at:this.lanes[0].game.time,goal});this.notify();}
    pause(phase='PAUSED'){
      if(this.running&&this.config.mode==='realtime'&&this.phase==='RUNNING')this.sync();
      this.running=false;this.epoch++;this.controller?.abort();this.controller=null;
      for(const lane of this.lanes||[]){if(lane.pending&&live(lane.player))lane.cancelled++;lane.pending=false;lane.human=null;}
      this.phase=phase;if(this.lanes)this.notify();
    }
    sync(){
      if(this.phase!=='RUNNING'||this.config.mode!=='realtime')return;
      const target=this.anchorSim+Math.max(0,this.now()-this.anchor);
      for(const game of this.games)game.advanceTo(target);
    }
    fail(error,run,lane=null){
      if(run!==this.epoch||error.name==='AbortError')return;
      this.failure={player:lane?.player??null,chefId:lane?.chefId??null,message:error.message,kind:error.kind??'request',status:error.status??null,receipt:error.receipt??null};
      this.pause('REQUEST FAILED');
    }
    async warm(run,signal){
      for(const lane of this.lanes){
        if(!live(lane.player)||this.warmups[lane.player])continue;
        const profile=this.config.profiles[lane.player];
        const url=new URL(profile.endpoint);
        if(!['http:','https:'].includes(url.protocol)||url.username||url.password||url.search||url.hash)throw new Error('Use an HTTP API base URL without embedded credentials or query parameters.');
        if(lane.player==='jev'&&!profile.apiKey)throw new Error('Enter a TypeSafe key under Jev connection, or select Local reference.');
        const options=this.options(lane),request=KitchenGame.requestFor(lane.game,options,profile.model,this.config.goal,lane.chefId);
        lane.status='WARMING UP';this.notify();
        try{
          const result=await SystemOne.decide(profile.endpoint,request,options,{apiKey:profile.apiKey,signal:AbortSignal.any([signal,AbortSignal.timeout(30000)])});
          if(run!==this.epoch)return;
          this.warmups[lane.player]={ms:result.ms,model:result.body.model||profile.model};lane.status='READY';
        }catch(error){this.fail(error,run,lane);throw error;}
      }
    }
    options(lane){return lane.game.options(lane.chefId);}
    choose(index,id){
      const lane=this.lanes[index];if(!lane?.human)return false;
      this.sync();
      const options=this.options(lane),option=options.find(option=>option.id===id);
      if(!option)return false;
      lane.human({option,options,observedAt:lane.game.time});return true;
    }
    async decide(lane,run,signal,defer=false){
      let options=this.options(lane),observedAt=lane.game.time;
      if(options.length<=1||lane.pending||lane.prepared||lane.game.over||lane.game.time<lane.nextAt)return;
      const goal=this.config.goal,player=lane.player;
      let request=null,result;
      lane.pending=true;lane.status=player==='human'?'YOUR CALL':live(player)?'DECIDING':'LOCAL POLICY';this.notify();
      try{
        if(player==='human'){
          result=await new Promise((resolve,reject)=>{
            const abort=()=>{lane.human=null;reject(new DOMException('Paused','AbortError'));};
            lane.human=choice=>{signal.removeEventListener('abort',abort);lane.human=null;resolve(choice);};
            signal.addEventListener('abort',abort,{once:true});
            this.notify();
          });options=result.options;observedAt=result.observedAt;
        }else if(player==='reference')result={option:KitchenGame.reference(lane.game,options,goal)};
        else{
          const profile=this.config.profiles[player];request=KitchenGame.requestFor(lane.game,options,profile.model,goal,lane.chefId);lane.attempts++;
          result=await SystemOne.decide(profile.endpoint,request,options,{apiKey:profile.apiKey,signal:AbortSignal.any([signal,AbortSignal.timeout(30000)])});
        }
        if(run!==this.epoch||signal.aborted)return;
        this.sync();
        const record={source:live(player)?'live':player,player,chefId:lane.chefId,goal,observedAt,appliedAt:null,
          choice:result.option.code,action:result.option.id,label:result.option.label,options,request,response:result.body??null,
          ms:result.ms??null,probabilities:result.probabilities??null,choiceWarning:result.choiceWarning??null,applied:null,stale:false,conflict:false,reason:null};
        lane.records.push(record);lane.last=record;lane.prepared={record,option:result.option};
        if(defer)lane.status='WAITING FOR PARTNER';else this.commit(lane);
      }finally{if(run===this.epoch){lane.pending=false;lane.human=null;this.notify();}}
    }
    commit(lane){
      if(!lane.prepared)return;
      const {record,option}=lane.prepared,game=lane.game;
      const order=game.orders.find(order=>order.id===option.order);
      const claimant=game.chefs.find(chef=>chef.id!==lane.chefId&&(chef.job?.order===option.order||chef.job?.station===option.station));
      const allowed=option.type==='wait'||lane.chefId==null||option.chef===lane.chefId;
      const applied=allowed&&game.apply(option),conflict=!applied&&this.cooperative&&!game.over&&order?.status==='waiting'&&!!claimant;
      Object.assign(record,{applied,appliedAt:game.time,stale:!applied,conflict,
        reason:applied?null:conflict?claimant.name+' already claimed this order or station.':'The job is no longer available.',
        handoffFrom:applied&&option.type!=='wait'?game.chefs[option.chef].job?.handoffFrom??null:null});
      lane.prepared=null;lane.nextAt=game.time+(option.type==='wait'?500:100);
      lane.status=applied?option.label:conflict?'CONFLICT · CHOOSE AGAIN':'STALE · NO ACTION APPLIED';
      if(!applied&&this.cooperative)game.log(game.chefs[lane.chefId].name+' rejected: '+option.label+'. '+record.reason);
    }
    async start(single=false){
      if(this.running){this.pause();return;}
      if(this.lanes.every(lane=>lane.game.over))return;
      if(single&&this.config.mode!=='paused')return;
      const run=++this.epoch;this.running=true;this.controller=new AbortController();const signal=this.controller.signal;
      this.failure=null;this.phase='WARMING UP';this.notify();
      try{
        await this.warm(run,signal);if(run!==this.epoch)return;
        this.anchor=this.now();this.anchorSim=this.lanes[0].game.time;this.phase='RUNNING';this.notify();
        while(this.running&&run===this.epoch){
          if(this.config.mode==='realtime'){
            this.sync();
            if(this.lanes.every(lane=>lane.game.over)){this.pause('SHIFT COMPLETE');break;}
            for(const lane of this.lanes)if(!lane.pending&&lane.game.time>=lane.nextAt)
              void this.decide(lane,run,signal).catch(error=>this.fail(error,run,lane));
          }else{
            await Promise.all(this.lanes.map(lane=>this.decide(lane,run,signal,this.cooperative).catch(error=>{this.fail(error,run,lane);throw error;})));
            if(run!==this.epoch)return;
            if(this.cooperative){for(const index of this.round%2?[1,0]:[0,1])this.commit(this.lanes[index]);this.round++;}
            for(const game of this.games)game.advanceTo(game.time+KitchenGame.ROUND);
            if(this.lanes.every(lane=>lane.game.over)){this.pause('SHIFT COMPLETE');break;}
            if(single){this.pause('STEP COMPLETE');break;}
          }
          this.notify();await this.sleep(this.config.mode==='realtime'?50:80,signal);
        }
      }catch(error){this.fail(error,run);}
    }
    report(){
      const actors=this.lanes.map(lane=>({player:lane.player,chefId:lane.chefId,attempts:lane.attempts,cancelled:lane.cancelled,
        summary:SystemOne.summary(lane.records),stale:lane.records.filter(record=>record.stale).length,conflicts:lane.records.filter(record=>record.conflict).length,decisions:lane.records}));
      return {game:'kitchen',version:2,config:{layout:this.cooperative?'cooperative':'comparison',seed:this.config.seed,duration:this.config.duration,mode:this.config.mode,players:this.config.players,
        profiles:Object.fromEntries(Object.entries(this.config.profiles).map(([name,profile])=>[name,publicProfile(profile)]))},
        goalHistory:this.goalHistory,warmups:this.warmups,failure:this.failure,schedule:this.lanes[0].game.schedule,
        ...(this.cooperative?{kitchens:this.games.map(game=>game.snapshot()),chefs:actors}:
          {kitchens:this.games.map((game,index)=>({...actors[index],...game.snapshot()}))})};
    }
  }
  return {Match,live};
})();
