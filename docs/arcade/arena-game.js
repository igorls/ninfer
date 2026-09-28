// Arena Duel's simulation. Time is milliseconds; positions and ranges are arena metres.
// Rendering, network latency and the optional reference policy do not own combat rules.
const ArenaGame = (() => {
  'use strict';
  const actions = [
    {code:'A',id:'advance',label:'Close distance',key:'D',cost:0,description:'Advance for 220 ms at 4 m/s; no protection.'},
    {code:'B',id:'retreat',label:'Backstep',key:'A',cost:0,description:'Retreat for 220 ms at 3.5 m/s; no invulnerability.'},
    {code:'C',id:'strike',label:'Quick strike',key:'J',cost:10,description:'10 stamina. Hit after 150 ms, range 2.25 m, 12 damage; recovery 230 ms. Cannot cancel.'},
    {code:'D',id:'heavy',label:'Power strike',key:'K',cost:24,description:'24 stamina. Step in 0.65 m; hit after 360 ms, range 2.5 m, 25 damage; recovery 340 ms. Cannot cancel.'},
    {code:'E',id:'guard',label:'Guard / parry',key:'L',cost:4,description:'4 stamina. Guard for 500 ms. Parries blockable hits in first 190 ms; later blocks cost 12 stamina and take 3 damage. CRUSH breaks guard.'},
    {code:'F',id:'dodge',label:'Dodge',key:'Space',cost:20,description:'20 stamina. Retreat 0.9 m over 360 ms; invulnerable for 360 ms. Recovery ends at 430 ms. Cannot cancel.'},
    {code:'G',id:'recover',label:'Catch breath',key:'S',cost:0,description:'Stand still for 260 ms to recover stamina at 50/s; no protection. Normal regeneration is 14/s.'}
  ];
  const attacks = {
    cut:{name:'SLASH',windup:540,range:2.7,damage:19,recovery:650,blockable:true},
    rush:{name:'LUNGE',windup:640,range:3.4,damage:22,recovery:760,blockable:true},
    crush:{name:'CRUSH',windup:780,range:2.9,damage:30,recovery:950,blockable:false}
  };
  const levels={sparring:1.3,duel:1,blitz:.72};
  const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
  class Game {
    constructor({seed=1,difficulty='duel',duration=45000}={}) {
      this.seed=seed>>>0;this.randomState=this.seed;this.difficulty=difficulty;this.duration=duration;
      this.time=0;this.over=false;this.result=null;this.serial=0;this.events=[];
      this.player={x:-1.65,hp:100,stamina:100,action:'idle',since:0,until:0,lockedUntil:0,attack:null,hurtAt:-10000};
      this.enemy={x:1.65,hp:150,action:'idle',since:0,until:700,attack:null,hurtAt:-10000};
      this.stats={dealt:0,taken:0,parries:0,blocks:0,dodges:0,misses:0,actions:0};
    }
    random(){let x=this.randomState+=0x6D2B79F5;x=Math.imul(x^x>>>15,x|1);x^=x+Math.imul(x^x>>>7,x|61);return ((x^x>>>14)>>>0)/4294967296;}
    get distance(){return Math.abs(this.enemy.x-this.player.x);}
    event(type,extra={}){this.events.push({id:++this.serial,time:this.time,type,...extra});if(this.events.length>400)this.events.shift();}
    legal(){return this.over||this.time<this.player.lockedUntil?[]:actions.filter(a=>a.cost<=this.player.stamina);}
    act(code){
      const a=this.legal().find(a=>a.code===code);if(!a)return false;
      const p=this.player,t=this.time;p.stamina-=a.cost;p.action=a.id;p.since=t;p.attack=null;this.stats.actions++;
      const duration={advance:220,retreat:220,strike:380,heavy:700,guard:500,dodge:430,recover:260}[a.id];
      p.until=t+duration;p.lockedUntil=t+({advance:140,retreat:140,guard:180,recover:180}[a.id]??duration);
      if(a.id==='strike'||a.id==='heavy')p.attack={at:t+(a.id==='strike'?150:360),done:false};
      this.event('action',{actor:'player',action:a.id});return true;
    }
    tick(ms){
      // Bounded integration steps, without throwing away elapsed time during a slow frame.
      if(!Number.isFinite(ms)||ms<0)throw new Error('Invalid elapsed time');
      while(ms>0&&!this.over){const dt=Math.min(10,ms,this.duration-this.time);this.step(dt);ms-=dt;}
    }
    step(dt){
      this.time+=dt;const t=this.time,p=this.player,e=this.enemy;
      p.stamina=clamp(p.stamina+dt*(p.action==='recover'?.05:.014),0,100);
      const dir=e.x>p.x?1:-1;
      if(t<p.until){
        if(p.action==='advance')p.x+=dir*dt*.004;
        if(p.action==='retreat')p.x-=dir*dt*.0035;
        if(p.action==='dodge'&&t-p.since<=360)p.x-=dir*dt*.0025;
        if(p.action==='heavy'&&t-p.since<=260)p.x+=dir*dt*.0025;
      } else {p.action='idle';p.attack=null;}
      p.x=clamp(p.x,-5,5);
      if(e.action==='idle'){
        if(this.distance>2.2)e.x-=dir*dt*.0026;
        if(t>=e.until&&this.distance<=3.1){
          const type=['cut','rush','crush'][Math.floor(this.random()*3)],spec=attacks[type];
          const windup=Math.round(spec.windup*levels[this.difficulty]);
          e.action='telegraph';e.since=t;e.until=t+windup;e.attack={type,...spec,at:e.until,started:t,windup};
          this.event('telegraph',{actor:'enemy',attack:type,impactAt:e.until});
        }
      } else if(e.action==='recovery'&&t>=e.until){e.action='idle';e.since=t;e.until=t+350+this.random()*350;e.attack=null;}
      // Fighters keep their own side and cannot overlap or pass through each other.
      e.x=clamp(e.x,-4.7,4.7);
      if(e.x-p.x<1.15){const mid=(e.x+p.x)/2;p.x=clamp(mid-.575,-5,3.85);e.x=p.x+1.15;}
      if(p.attack&&!p.attack.done&&t>=p.attack.at){p.attack.done=true;this.playerHit();}
      if(e.hp>0&&e.action==='telegraph'&&t>=e.until)this.enemyHit();
      if(p.hp<=0||e.hp<=0||t>=this.duration){
        this.over=true;this.result=p.hp<=0?'defeat':e.hp<=0?'victory':p.hp/100>e.hp/150?'victory':p.hp/100<e.hp/150?'defeat':'draw';
        this.event('finish',{result:this.result,reason:p.hp<=0||e.hp<=0?'knockout':'time'});
      }
    }
    playerHit(){
      const p=this.player,e=this.enemy,heavy=p.action==='heavy';
      if(this.distance>(heavy?2.5:2.25)){this.stats.misses++;this.event('miss',{actor:'player'});return;}
      const counter=e.action==='recovery',damage=Math.min(e.hp,Math.round((heavy?25:12)*(counter?1.4:1)));
      e.hp=Math.max(0,e.hp-damage);e.hurtAt=this.time;this.stats.dealt+=damage;
      this.event('hit',{actor:'player',damage,counter,heavy,x:e.x});
    }
    enemyHit(){
      const p=this.player,e=this.enemy,a=e.attack,t=this.time;
      e.action='recovery';e.since=t;e.until=t+a.recovery;
      const evading=p.action==='dodge'&&t-p.since<=360;
      if(this.distance>a.range||evading){
        if(evading)this.stats.dodges++;
        this.event(evading?'dodge':'miss',{actor:'enemy',x:p.x});return;
      }
      if(p.action==='guard'&&a.blockable){
        if(t-p.since<=190){this.stats.parries++;e.until=t+1050;p.stamina=clamp(p.stamina+12,0,100);this.event('parry',{x:p.x});return;}
        if(p.stamina>=12){const damage=Math.min(3,p.hp);p.stamina-=12;p.hp-=damage;this.stats.taken+=damage;this.stats.blocks++;this.event('block',{x:p.x});return;}
      }
      const damage=Math.min(p.hp,a.damage);p.hp-=damage;this.stats.taken+=damage;p.hurtAt=t;p.action='stagger';p.since=t;p.until=t+260;p.lockedUntil=p.until;p.attack=null;
      this.event('hit',{actor:'enemy',damage,heavy:a.type==='crush',x:p.x});
    }
    observation(){
      const p=this.player,e=this.enemy;
      return {remaining_ms:Math.max(0,Math.round(this.duration-this.time)),distance_m:+this.distance.toFixed(2),
        you:{hp:p.hp,stamina:Math.floor(p.stamina),action:p.action,retreat_room_m:+(p.x+5).toFixed(2)},
        opponent:{hp:e.hp,state:e.action,attack:e.action==='telegraph'?e.attack.name:null,
          impact_in_ms:e.action==='telegraph'?Math.ceil(e.until-this.time):null,
          attack_range_m:e.action==='telegraph'?e.attack.range:null,
          blockable:e.action==='telegraph'?e.attack.blockable:null,
          recovery_remaining_ms:e.action==='recovery'?Math.ceil(e.until-this.time):0}};
    }
  }
  // Explicit offline opponent to compare with; never called by the model controller.
  function reference(game){
    const p=game.player,e=game.enemy,d=game.distance,options=game.legal();if(!options.length)return null;
    const has=id=>options.find(a=>a.id===id)?.code;
    if(e.action==='telegraph'&&e.until-game.time<260&&d<=e.attack.range){
      if(e.attack.blockable&&has('guard'))return has('guard');
      return has('dodge')||has('retreat');
    }
    if(e.action==='telegraph')return has('recover');
    if(d>2.2)return has('advance');
    if(p.stamina<27)return has('recover');
    if(e.action==='recovery'&&e.until-game.time>500)return has('heavy')||has('strike');
    return has('strike');
  }
  function request(model,game,options=game.legal()){
    return {model,state:'Real-time duel. You are the cyan fighter. Defeat the red fighter; at 45 seconds higher remaining health fraction wins. Actions happen after network latency while the enemy keeps moving. A telegraphed attack hits when impact_in_ms reaches zero if you remain in range. Defense must cover the impact. Attacks cannot be cancelled; attacking into an imminent enemy hit is risky. Enemy recovery is vulnerable: your hits do 40% extra damage. Distances are centre to centre. Current observation: '+JSON.stringify(game.observation()),
      questions:{move:{type:'choice',instructions:'Choose the best immediate combat action. Survive incoming attacks, manage stamina, close distance when safe, and punish enemy recovery. Allow for roughly 100 ms before this action starts. Choose only the option token.',criteria:Object.fromEntries(options.map(a=>[a.code,a.label+'. '+a.description]))}}};
  }
  return {Game,actions,attacks,levels,reference,request};
})();
