(() => {
  'use strict';
  const $=id=>document.getElementById(id),reduced=matchMedia('(prefers-reduced-motion: reduce)'),names={qwen:'QWEN / SYSTEM ONE',human:'YOU / HUMAN',reference:'LOCAL / REFERENCE'};
  const controls=['pilot','difficulty','seed','deadline','endpoint','model','key'];
  let session,view,lastRow=null,eventCursor=0,callUntil=0,sound=false,audio=null,frame=0,lastPaint='';
  const ms=n=>Number.isFinite(n)?n.toFixed(1):'—';
  function config(){return {mode:$('pilot').value,difficulty:$('difficulty').value,seed:Number($('seed').value)>>>0,deadline:Number($('deadline').value),endpoint:$('endpoint').value.trim(),model:$('model').value.trim(),key:$('key').value};}
  function newBout(){
    session?.pause();session=new ArenaSession.Session(config());lastRow=null;eventCursor=0;lastPaint='';view?.reset();
    $('combat-call').classList.remove('show');$('request-wire').textContent='No request yet.';$('response-wire').textContent='No response yet.';$('probabilities').replaceChildren();$('history').replaceChildren();
    $('receipt-note').textContent='Warm-up and full validated response timings are recorded separately. Probabilities express action preference, not a chance of winning.';
    $('action').textContent='READY TO REACT';$('action-status').textContent='Waiting for the bell';$('latency').replaceChildren(document.createTextNode('—'));$('reaction-fill').style.width='0%';
    $('manual').hidden=session.config.mode!=='human';$('player-tag').textContent=names[session.config.mode];$('mode-label').textContent=names[session.config.mode];
    $('intro-pilot').replaceChildren(document.createTextNode({qwen:'Qwen steps into the ring.',human:'Your fighter. Your reflexes.',reference:'The local policy takes the controls.'}[session.config.mode]),document.createElement('br'),document.createTextNode('Every opening is a decision.'));
    $('bezel-source').textContent=names[session.config.mode]+' · SEED '+session.config.seed;
    $('deadline-marker').style.left=(session.config.deadline/500*100)+'%';
    update();
  }
  function tone(event){
    if(!sound||!audio)return;
    const now=audio.currentTime,parry=event.type==='parry',hit=event.type==='hit',defense=event.type==='block'||event.type==='dodge';if(!parry&&!hit&&!defense)return;
    const o=audio.createOscillator(),gain=audio.createGain();o.type=parry?'sine':'triangle';o.frequency.setValueAtTime(parry?1350:hit?event.heavy?115:185:350,now);o.frequency.exponentialRampToValueAtTime(parry?600:45,now+.15);gain.gain.setValueAtTime(.0001,now);gain.gain.exponentialRampToValueAtTime(.07,now+.004);gain.gain.exponentialRampToValueAtTime(.0001,now+.22);o.connect(gain).connect(audio.destination);o.start(now);o.stop(now+.23);
  }
  function flash(text,color){const el=$('combat-call');el.textContent=text;el.style.color=color;el.classList.remove('show');void el.offsetWidth;el.classList.add('show');callUntil=performance.now()+800;}
  function receipt(row){
    if(!row)return;const action=ArenaGame.actions.find(a=>a.code===row.action);
    $('action').textContent=row.warmup?'WARM-UP COMPLETE':action?.label.toUpperCase()||row.status.toUpperCase();
    $('action-status').textContent=row.warmup?'Not applied to the fight':({applied:'Applied to the live fight',expired:'Late reply discarded',unavailable:'No longer legal · discarded',failed:'Request failed'}[row.status]||row.status);
    $('latency').textContent=row.source==='live'?ms(row.ms):'—';
    const unit=document.createElement('small');unit.textContent=row.source==='live'?' ms':' offline';$('latency').append(unit);
    $('reaction-fill').style.width=(row.source==='live'?Math.min(100,(row.ms??row.elapsed??0)/5):0)+'%';
    $('reaction-fill').style.background=row.status==='expired'?'#ff795f':'#ffbf45';
    $('request-wire').textContent=row.request?JSON.stringify(row.request,null,2):'Offline action; no API request.';
    $('response-wire').textContent=row.response?JSON.stringify(row.response,null,2):row.error||'No API response.';
    $('receipt-note').textContent=row.warning||`${row.warmup?'Warm-up':row.source==='live'?'Live request':'Offline action'} · ${row.status}${row.source==='live'?' · elapsed '+ms(row.elapsed)+' ms':''}. Full responses are preserved in the export; keys are excluded.`;
    $('probabilities').replaceChildren();
    if(row.probabilities)for(const action of ArenaGame.actions){const p=row.probabilities[action.code];if(p==null)continue;const el=document.createElement('span');el.textContent=action.label+' '+(p*100).toFixed(1)+'%';if(action.code===row.action)el.className='picked';$('probabilities').append(el);}
    const recent=session.records.slice(-32);$('history').replaceChildren(...recent.map(r=>{const el=document.createElement('i');el.className=r.status;el.title=(r.warmup?'Warm-up':ArenaGame.actions.find(a=>a.code===r.action)?.label||'Decision')+' · '+r.status+(r.ms?' · '+ms(r.ms)+' ms':'');return el;}));
  }
  function update(){
    const g=session.game,p=g.player,e=g.enemy,status=session.status,running=status==='running',active=running||status==='warming',stats=ArenaSession.summary(session.records);
    $('run').textContent=running?'PAUSE':status==='warming'?'CANCEL WARM-UP':status==='finished'?'FIGHT AGAIN':status==='paused'?'RESUME FIGHT':status==='error'?'RETRY FIGHT':'START FIGHT';
    $('run').disabled=!view;$('export').disabled=!session.records.length;
    controls.forEach(id=>$(id).disabled=active);
    $('power-label').textContent={ready:'STANDING BY',warming:'WARMING MODEL',running:'FIGHT IN PROGRESS',paused:'PAUSED',finished:'BOUT COMPLETE',error:'CONNECTION ERROR'}[status];
    $('led').classList.toggle('on',running);$('round-clock').textContent=Math.max(0,Math.ceil((g.duration-g.time)/1000)).toString().padStart(2,'0');
    $('round-state').textContent=running?'LIVE':status.toUpperCase();
    $('player-health').style.width=p.hp+'%';$('enemy-health').style.width=e.hp/150*100+'%';$('stamina').style.width=p.stamina+'%';
    $('player-hp').textContent=p.hp+' / 100';$('enemy-hp').textContent=e.hp+' / 150';
    $('stage-intro').hidden=status!=='ready';$('bout-result').hidden=status!=='finished';
    if(status==='finished'){$('bout-result').dataset.result=g.result;$('result-title').textContent=g.result.toUpperCase();$('result-reason').textContent=g.events.at(-1)?.reason==='knockout'?'KNOCKOUT':'TIME · HEALTH FRACTION';$('result-detail').textContent=`${p.hp}% health left · ${g.stats.dealt} damage dealt · ${g.stats.parries} parries`;}
    const tell=e.action==='telegraph'&&!g.over;$('tell').hidden=!tell;
    if(tell){const a=e.attack;$('tell').dataset.heavy=String(!a.blockable);$('tell-name').textContent=a.name;$('tell-kind').textContent=a.blockable?'BLOCKABLE':'UNBLOCKABLE';$('tell-time').textContent=Math.max(0,Math.ceil(e.until-g.time))+' ms TO IMPACT';$('tell-fill').style.width=Math.max(0,(e.until-g.time)/a.windup*100)+'%';}
    $('stage-message').textContent=status==='warming'?'FIRST REQUEST · THE CLOCK IS HELD':status==='paused'?'PAUSED · THE CLOCK IS HELD':status==='ready'?'PRESS START FIGHT':status==='finished'?'BOUT COMPLETE':status==='error'?'CHECK CONNECTION · RETRY WHEN READY':e.action==='recovery'?'MAUL IS RECOVERING':session.pending?'QWEN IS DECIDING · WORLD IS LIVE':session.config.mode==='human'?'A / D MOVE · J / K STRIKE · L GUARD · SPACE DODGE':'WORLD IS LIVE';
    $('damage').textContent=g.stats.dealt+' : '+g.stats.taken;$('damage').title='Damage dealt : damage taken';$('parries').textContent=g.stats.parries;$('dodges').textContent=g.stats.dodges;
    $('latency-range').textContent='P50 '+ms(stats.p50)+' / P95 '+ms(stats.p95);$('missed').textContent=stats.expired+' LATE';
    const applied=session.config.mode==='qwen'?stats.applied:session.records.filter(r=>r.status==='applied').length;$('decision-count').textContent=applied+' APPLIED';
    $('inspector-count').textContent=stats.samples?`${stats.samples} RESPONSES / ${stats.unavailable} UNAVAILABLE`:'STRUCTURED STATE → ONE ACTION';
    $('fault').textContent=session.error?session.error+' Check Qwen connection, then Retry fight.':'';
    const row=session.records.at(-1);if(row!==lastRow){lastRow=row;receipt(row);}
    for(const event of g.events){if(event.id<=eventCursor)continue;eventCursor=event.id;tone(event);
      if(event.type==='parry')flash('PARRY','#baffff');
      else if(event.type==='dodge')flash('EVADED','#baffff');
      else if(event.type==='block')flash('GUARD','#e9d59b');
      else if(event.type==='hit')flash(event.actor==='player'?(event.counter?'COUNTER ':'HIT ')+event.damage:'−'+event.damage,event.actor==='player'?'#a3f8ff':'#ff9477');
    }
    if(performance.now()>callUntil)$('combat-call').classList.remove('show');
    if(session.config.mode==='human'){const legal=g.legal();$('manual-actions').querySelectorAll('button').forEach(b=>b.disabled=!running||!legal.some(a=>a.code===b.dataset.code));}
  }
  function loop(now){
    session.pump();const stamp=[session.status,session.records.length,!!session.pending,Math.floor(session.game.time/50)].join(':');
    if(stamp!==lastPaint){lastPaint=stamp;update();}
    if(now>callUntil&&$('combat-call').classList.contains('show'))$('combat-call').classList.remove('show');
    view?.render(session.game,now,reduced.matches);frame=requestAnimationFrame(loop);
  }
  $('run').addEventListener('click',async()=>{if(['running','warming'].includes(session.status)){session.pause();return;}if(session.status==='finished')newBout();await session.start();if(session.config.mode==='human')$('arena').focus();});
  $('reset').addEventListener('click',newBout);controls.forEach(id=>$(id).addEventListener('change',newBout));
  $('sound').addEventListener('click',async()=>{if(!audio)audio=new (window.AudioContext||window.webkitAudioContext)();await audio.resume();sound=!sound;$('sound').setAttribute('aria-pressed',String(sound));$('sound').textContent=sound?'SOUND ON':'SOUND OFF';});
  for(const action of ArenaGame.actions){const b=document.createElement('button');b.dataset.code=action.code;const key=document.createElement('b');key.textContent=action.key;b.append(key,document.createTextNode(action.label));b.title=action.description;b.addEventListener('click',()=>session.human(action.code));$('manual-actions').append(b);}
  const keys={a:'B',d:'A',j:'C',k:'D',l:'E',' ':'F',s:'G',ArrowLeft:'B',ArrowRight:'A'};
  $('arena').addEventListener('keydown',event=>{if(event.key.toLowerCase()==='p'){event.preventDefault();$('run').click();return;}const code=keys[event.key]||keys[event.key.toLowerCase()];if(code&&session.config.mode==='human'){event.preventDefault();session.human(code);}});
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&['running','warming'].includes(session.status))session.pause();});
  $('export').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(session.export(),null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`arena-${session.config.mode}-seed-${session.config.seed}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
  try{view=new ArenaView.View($('arena'));}catch(error){$('fault').textContent='The 3D arena could not start. Enable WebGL and reload. '+error.message;}
  newBout();
  if(!view){session.error='The 3D arena could not start. Enable WebGL and reload.';session.status='error';}
  window.ArenaApp=Object.freeze({snapshot:()=>session.export()});
  frame=requestAnimationFrame(loop);
  window.addEventListener('pagehide',()=>{session.pause();cancelAnimationFrame(frame);audio?.suspend();});
})();
