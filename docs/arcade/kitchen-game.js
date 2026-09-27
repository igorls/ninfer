// Deterministic kitchen rules. The model chooses jobs; movement and job execution are mechanical.
const KitchenGame=(()=>{
  const TICK=100,ROUND=500;
  const recipes={salad:{name:'Garden salad',prep:1800,cook:0,price:12},soup:{name:'Tomato soup',prep:2200,cook:4000,price:20},toast:{name:'Mushroom toast',prep:1400,cook:2500,price:16}};
  const stations={pantry:{name:'Pantry',x:1,y:1},prep:{name:'Chop',x:4,y:1},stove:{name:'Stove',x:7,y:1},plate:{name:'Plate',x:7,y:4},pass:{name:'Serve',x:4,y:4}};
  const goals={revenue:'Maximize revenue from completed orders. Favor valuable orders when their deadlines are achievable.',waste:'Minimize food waste while serving as many orders as possible. Finish started food, rescue cooked dishes before they burn, and avoid starting food you cannot serve in time.',vip:'Prioritize serving VIP orders before their deadlines, then maximize total orders served.'};
  const seconds=ms=>(ms/1000).toFixed(1)+'s';
  function schedule(seed=1,duration=90000){
    let state=seed>>>0;const random=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return state/4294967296;};
    const result=[];let at=0;
    while(at<duration-12000){
      const recipe=Object.keys(recipes)[Math.floor(random()*3)],vip=random()<.25;
      result.push({id:result.length+1,recipe,vip,at,deadline:at+22000+Math.floor(random()*50)*100,price:recipes[recipe].price+(vip?8:0)});
      at+=result.length===1?1500:3500+Math.floor(random()*20)*100;
    }
    return result;
  }
  class Game{
    constructor({seed=1,duration=90000,orders=schedule(seed,duration)}={}){
      this.seed=seed;this.duration=duration;this.time=0;this.schedule=orders.map(order=>({...order}));this.arrival=0;this.orders=[];
      this.chefs=[{id:0,name:'Ada',x:3,y:3,job:null},{id:1,name:'Bo',x:5,y:3,job:null}];
      this.stats={served:0,missed:0,waste:0,burned:0,revenue:0,vipServed:0,handoffs:0};this.events=[];this.arrive();
    }
    get over(){return this.time>=this.duration;}
    log(text){this.events.push({at:this.time,text});if(this.events.length>80)this.events.shift();}
    active(){return this.orders.filter(order=>order.status==='waiting');}
    arrive(){
      while(this.arrival<this.schedule.length&&this.schedule[this.arrival].at<=this.time){
        const order={...this.schedule[this.arrival++],stage:'none',status:'waiting',claimedBy:null,readyAt:null,burnAt:null};
        if(this.active().length>=6){order.status='missed';this.stats.missed++;this.log('Order #'+order.id+' lost: ticket rail full.');}
        else this.log((order.vip?'VIP ':'')+'Order #'+order.id+' · '+recipes[order.recipe].name);
        this.orders.push(order);
      }
    }
    position(chef){
      if(!chef.job)return {x:chef.x,y:chef.y};
      const job=chef.job,to=stations[job.station],distance=Math.abs(to.x-job.from.x)+Math.abs(to.y-job.from.y);
      const moved=Math.min(distance,Math.max(0,(this.time-job.start)/350)),dx=to.x-job.from.x;
      return {x:job.from.x+Math.sign(dx)*Math.min(Math.abs(dx),moved),y:job.from.y+Math.sign(to.y-job.from.y)*Math.max(0,moved-Math.abs(dx))};
    }
    available(station,ownOrder=null){
      if(this.chefs.some(chef=>chef.job?.station===station))return false;
      return station!=='stove'||!this.orders.some(order=>order.id!==ownOrder&&['cooking','ready','burnt'].includes(order.stage));
    }
    options(chefId=null){
      if(this.over)return [];
      const result=[];
      for(const chef of this.chefs){
        if(chef.job||(chefId!=null&&chef.id!==chefId))continue;
        for(const order of this.orders){
          if(order.claimedBy!=null||(order.status!=='waiting'&&order.stage!=='burnt'))continue;
          const recipe=recipes[order.recipe];let type,station,work;
          if(order.stage==='none'){type='collect';station='pantry';work=700;}
          else if(order.stage==='raw'){type='chop';station='prep';work=recipe.prep;}
          else if(order.stage==='prepped'&&recipe.cook){type='cook';station='stove';work=600;}
          else if(order.stage==='ready'||(order.stage==='prepped'&&!recipe.cook)){type='plate';station='plate';work=800;}
          else if(order.stage==='plated'){type='serve';station='pass';work=500;}
          else if(order.stage==='burnt'){type='clear';station='stove';work=1000;}
          else continue;
          if(!this.available(station,type==='clear'?order.id:null))continue;
          const target=stations[station],travel=(Math.abs(target.x-chef.x)+Math.abs(target.y-chef.y))*350;
          const duration=Math.ceil((travel+work)/TICK)*TICK;
          result.push({id:chef.id+':'+type+':'+order.id,chef:chef.id,order:order.id,type,station,duration,
            label:chef.name+': '+type+' #'+order.id+' '+recipe.name,
            detail:seconds(duration)+' job; due in '+seconds(order.deadline-this.time)+'; value $'+order.price+(order.vip?'; VIP':'')+(order.stage==='ready'?'; burns in '+seconds(order.burnAt-this.time):'')});
        }
      }
      result.push({id:'wait',type:'wait',label:'Wait 0.5s',detail:'Let current jobs and cooking advance.'});
      if(result.length>62)throw new Error('Kitchen action set exceeds the choice limit.');
      return result.map((option,i)=>({...option,code:SystemOne.CODES[i]}));
    }
    apply(option){
      const current=this.options().find(candidate=>candidate.id===option.id);
      if(!current)return false;
      if(current.type==='wait')return true;
      const chef=this.chefs[current.chef],order=this.orders.find(order=>order.id===current.order);
      order.claimedBy=chef.id;
      chef.job={...current,start:this.time,finish:this.time+current.duration,from:{x:chef.x,y:chef.y}};
      if(current.type!=='clear'&&order.lastChef!=null&&order.lastChef!==chef.id){
        this.stats.handoffs++;chef.job.handoffFrom=order.lastChef;
        this.log(chef.name+' takes over #'+order.id+' from '+this.chefs[order.lastChef].name+'.');
      }
      this.log(current.label);return true;
    }
    complete(chef){
      const job=chef.job,order=this.orders.find(order=>order.id===job.order),station=stations[job.station];
      chef.x=station.x;chef.y=station.y;chef.job=null;order.claimedBy=null;
      if(job.type==='clear'){order.stage='none';order.lastChef=null;this.log(chef.name+' cleared the stove.');return;}
      if(order.status!=='waiting')return;
      if(job.type==='collect')order.stage='raw';
      if(job.type==='chop')order.stage='prepped';
      if(job.type==='cook'){order.stage='cooking';order.readyAt=this.time+recipes[order.recipe].cook;order.burnAt=order.readyAt+5000;}
      if(job.type==='plate'){
        if(order.stage==='burnt'){this.log('Too late to plate #'+order.id+'.');return;}
        order.stage='plated';
      }
      order.lastChef=chef.id;
      if(job.type!=='serve')this.log(chef.name+' finished '+job.type+' #'+order.id+'.');
      if(job.type==='serve'){order.status='served';order.stage='none';this.stats.served++;this.stats.revenue+=order.price;this.stats.vipServed+=Number(order.vip);this.log('Served #'+order.id+' · +$'+order.price);}
    }
    advanceTo(target){
      const end=Math.min(this.duration,Math.floor(target/TICK)*TICK);
      while(this.time<end){
        this.time+=TICK;
        for(const chef of this.chefs)if(chef.job&&chef.job.finish<=this.time)this.complete(chef);
        for(const order of this.orders){
          if(order.stage==='cooking'&&order.readyAt<=this.time){order.stage='ready';this.log('#'+order.id+' ready to plate.');}
          if(order.stage==='ready'&&order.burnAt<=this.time){order.stage='burnt';this.stats.burned++;this.stats.waste++;this.log('#'+order.id+' burned. Clear the stove.');}
          if(order.status==='waiting'&&(order.deadline<=this.time||this.over)){
            order.status='missed';this.stats.missed++;
            if(!['none','burnt'].includes(order.stage)){this.stats.waste++;order.stage='none';}
            this.log('#'+order.id+' expired.');
          }
        }
        this.arrive();
      }
    }
    state(){
      return 'Kitchen Rush at '+seconds(this.time)+' / '+seconds(this.duration)+'. Two chefs share one station of each type. Jobs include automatic walking. Cooking continues without a chef and burns 5s after ready. Plate jobs must finish by the burn time. Orders must be served by their deadlines. Reservations prevent duplicate work.\n'+
        'Recipe times excluding walking: salad chop 1.8s, no cooking; soup chop 2.2s, cook 4s; toast chop 1.4s, cook 2.5s. Every recipe: collect 0.7s, load stove 0.6s if cooking, plate 0.8s, serve 0.5s.\n'+
        this.chefs.map(chef=>chef.name+': '+(chef.job?chef.job.label+', finishes in '+seconds(chef.job.finish-this.time):'idle')).join('\n')+'\n'+
        this.orders.filter(order=>order.status==='waiting'||order.stage==='burnt').map(order=>'#'+order.id+' '+recipes[order.recipe].name+' '+order.stage+'; due '+seconds(order.deadline-this.time)+'; $'+order.price+(order.vip?' VIP':'')+(order.claimedBy!=null?'; assigned to '+this.chefs[order.claimedBy].name:'')+(order.stage==='cooking'||order.stage==='ready'?'; ready '+seconds(order.readyAt-this.time)+', burns '+seconds(order.burnAt-this.time):'')).join('\n');
    }
    snapshot(){return {time:this.time,stats:{...this.stats},orders:this.orders.map(order=>({...order})),chefs:this.chefs.map(chef=>({...chef,job:chef.job?{...chef.job}:null})),events:this.events.map(event=>({...event}))};}
  }
  function requestFor(game,options,model,goal='revenue',chefId=null){
    const teamwork=chefId==null?'Choose one useful job for an idle chef.':
      'You control only '+game.chefs[chefId].name+'. '+game.chefs[1-chefId].name+' is an independent teammate in this SAME kitchen. Share the score and cooperate. Your teammate may claim an order or station while you decide; a conflicting action will be rejected. Continue each other\'s unfinished dishes and choose complementary work. Choose only your own listed jobs.';
    const recent=chefId==null?'':'\nRecent shared activity:\n'+game.events.slice(-8).map(event=>seconds(event.at)+' '+event.text).join('\n');
    return {model,state:game.state()+recent,questions:{move:{type:'choice',instructions:goals[goal]+' '+teamwork+' Prevent food from burning; allow the other chef to progress a different order. Pipeline: collect, chop, cook if needed, plate, serve. Salad skips cooking. Clear burnt food to reopen the stove. Waiting is allowed when useful.',criteria:Object.fromEntries(options.map(option=>[option.code,option.label+'; '+option.detail]))}}};
  }
  function reference(game,options,goal='revenue'){
    // Inspectable greedy baseline, not an optimal scheduling oracle.
    const score=option=>{
      if(option.type==='wait')return -1e6;
      const order=game.orders.find(order=>order.id===option.order),slack=order.deadline-game.time;
      let value={serve:100,plate:80,clear:65,cook:45,chop:30,collect:10}[option.type];
      if(goal==='vip'&&order.vip)value+=45;
      if(goal==='revenue')value+=order.price*.6;
      if(goal==='waste'&&option.type!=='collect')value+=30;
      if(order.stage==='ready')value+=Math.max(0,40-(order.burnAt-game.time)/100);
      if(slack<option.duration&&option.type!=='clear')value-=300;
      return value-option.duration/500-Math.max(0,slack)/5000;
    };
    return options.reduce((best,option)=>score(option)>score(best)?option:best,options[0]);
  }
  return {Game,recipes,stations,goals,schedule,requestFor,reference,TICK,ROUND,seconds};
})();
