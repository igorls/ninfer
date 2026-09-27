(()=>{
  'use strict';
  const $=id=>document.getElementById(id),K=KitchenGame;
  const names={reference:'LOCAL REFERENCE',human:'YOU',qwen:'QWEN',jev:'JEV'};
  const stageNames={none:'NEW',raw:'COLLECTED',prepped:'CHOPPED',cooking:'COOKING',ready:'PLATE NOW',plated:'SERVE NOW',burnt:'BURNED'};
  const el=(tag,text,cls)=>{const node=document.createElement(tag);if(text!=null)node.textContent=text;if(cls)node.className=cls;return node;};
  const fmt=value=>Number.isFinite(value)?value.toFixed(1):'—';
  const time=ms=>Math.floor(ms/60000).toString().padStart(2,'0')+':'+Math.floor(ms/1000%60).toString().padStart(2,'0');
  const stationIcons={pantry:'<path d="M5 7h22v19H5zM5 13h22M11 7v19M21 7v19M12 3h8v4"/>',prep:'<path d="M3 22h25v6H3zM6 18 21 3l6 6-15 15M20 4l6 6"/>',stove:'<path d="M5 10h22v11H5zM9 6c-3-4 3-2 0-6M17 6c-3-4 3-2 0-6M24 6c-3-4 3-2 0-6M2 25h28M9 29h14"/>',plate:'<circle cx="16" cy="16" r="12"/><circle cx="16" cy="16" r="7"/>',pass:'<path d="M3 26h26M5 22a11 11 0 0 1 22 0zM14 8h4M16 8v3"/>'};
  const chefSVG='<svg viewBox="0 0 36 48" aria-hidden="true"><path class="apron" d="M8 28h20l4 18H4z"/><circle cx="18" cy="24" r="9" fill="#efd1a1" stroke="#30291b" stroke-width="1.4"/><path class="hat" d="M9 20V13C-1 8 7-1 14 4c5-8 15-2 13 4 10 1 7 11 0 11v4H9z"/><path d="M14 25h1m6 0h1" stroke="#30291b" stroke-width="2"/></svg>';
  let match=null,inspectionKey='',recordMenuKey='',configError='',activityKey='';
  const views=[0,1].map(index=>{
    const root=el('section',null,'kitchen-lane');root.setAttribute('aria-label',(index?'Right':'Left')+' kitchen');
    const heading=el('header',null,'kitchen-heading'),title=el('h2'),model=el('small');heading.append(title,model);
    const tickets=el('div',null,'ticket-rail');tickets.tabIndex=0;tickets.setAttribute('aria-label','Active order tickets');
    const map=el('div',null,'kitchen-map');map.setAttribute('role','img');
    const stations={};
    for(const [id,station]of Object.entries(K.stations)){
      const node=el('div',null,'station');node.style.left=station.x/8*100+'%';node.style.top=station.y/5*100+'%';
      node.innerHTML='<svg viewBox="0 0 32 32" aria-hidden="true">'+stationIcons[id]+'</svg>';
      const label=el('b',station.name.toUpperCase()),state=el('small');node.append(label,state);map.append(node);stations[id]={node,state};
    }
    const roster=el('div',null,'chef-roster'),chefs=[],jobs=[];
    for(const [id,name]of ['Ada','Bo'].entries()){
      const node=el('div',null,'chef'+(id?' bo':''));node.innerHTML=chefSVG;node.append(el('span',name));map.append(node);chefs.push(node);
      const row=el('div',null,'chef-job'+(id?' bo':'')),label=el('span','Idle'),remaining=el('small');row.append(el('b',name),label,remaining);roster.append(row);jobs.push({label,remaining});
    }
    const stats=el('dl',null,'kitchen-stats'),statNodes={};
    for(const [id,label]of [['revenue','REVENUE'],['served','SERVED'],['missed','MISSED'],['waste','WASTE']]){const group=el('div'),value=el('dd','0');group.append(el('dt',label),value);stats.append(group);statNodes[id]=value;}
    const decision=el('div',null,'lane-decision'),choice=el('b','Ready for service'),detail=el('span');decision.append(choice,detail);
    const probs=el('div',null,'kitchen-probs'),timings=el('div',null,'lane-timings'),event=el('p',null,'lane-event'),manual=el('div',null,'manual-jobs');
    root.append(heading,tickets,map,roster,stats,decision,probs,timings,event,manual);$('kitchens').append(root);
    return {root,heading,roster,stats,decision,title,model,tickets,map,stations,chefs,jobs,statNodes,choice,detail,probs,timings,event,manual,ticketNodes:new Map(),manualNodes:new Map(),lastRecord:undefined};
  });
  function config(){
    const seed=Number($('seed').value),duration=Number($('duration').value);
    if(!$('seed').value.trim()||!Number.isInteger(seed)||seed<0||seed>4294967295)throw new Error('Choose a seed from 0 to 4294967295.');
    if(![60000,90000,120000].includes(duration))throw new Error('Choose a supported shift length.');
    return {seed,duration,layout:$('layout').value,goal:$('goal').value,mode:$('clock-mode').value,players:[$('left-player').value,$('right-player').value],
      profiles:Object.fromEntries(['qwen','jev'].map(player=>[player,{endpoint:$(player+'-endpoint').value.trim(),model:$(player+'-model').value.trim(),apiKey:$(player+'-key').value}]))};
  }
  function reset(){
    try{
      const next=config();configError='';inspectionKey='';recordMenuKey='';activityKey='';$('inspect-record').value='latest';
      for(const view of views){view.lastRecord=undefined;view.probs.replaceChildren();view.ticketNodes.clear();view.tickets.replaceChildren();view.empty=null;view.manualNodes.clear();view.manual.replaceChildren();}
      if(match)match.reset(next);else match=new KitchenMatch.Match(next,{onChange:render});mount();render();
    }catch(error){configError=error.message;match?.pause();$('fault').textContent=configError;$('run').disabled=true;$('step').disabled=true;}
  }
  function mount(){
    const coop=match.cooperative;
    $('shared-kitchen').hidden=!coop;$('team-activity').hidden=!coop;$('kitchens').classList.toggle('cooperative',coop);
    $('shared-world').replaceChildren();$('activity-log').replaceChildren();
    for(const [index,view]of views.entries()){
      const world=[view.tickets,view.map,view.roster,view.stats];
      view.root.replaceChildren(view.heading,...(coop?[]:world),view.decision,view.probs,view.timings,view.event,view.manual);
      if(coop&&index===0)$('shared-world').append(...world);
      view.root.setAttribute('aria-label',coop?['Ada','Bo'][index]+' decisions':(index?'Right':'Left')+' kitchen');
      $(index?'right-label':'left-label').textContent=coop?['Ada · first chef','Bo · second chef'][index]:index?'Right kitchen':'Left kitchen';
    }
    $('inspect-lane').replaceChildren(...views.map((view,index)=>{const option=el('option',coop?['Ada','Bo'][index]:index?'Right kitchen':'Left kitchen');option.value=String(index);return option;}));$('inspect-lane').value='0';
    $('jev-match').textContent=coop?'Jev + Qwen':'Jev vs Qwen';
  }
  function renderWorld(view,game){
    const active=game.active();
    const ids=new Set(active.map(order=>order.id));
    for(const [id,ticket]of view.ticketNodes)if(!ids.has(id)){ticket.node.remove();view.ticketNodes.delete(id);}
    if(active.length&&view.empty){view.empty.remove();view.empty=null;}
    if(!active.length&&!view.empty){view.empty=el('p',game.over?'Service finished.':'Waiting for the next ticket…','ticket-empty');view.tickets.append(view.empty);}
    for(const order of active){
      let ticket=view.ticketNodes.get(order.id);
      if(!ticket){
        const node=el('article',null,'ticket'),header=el('header'),stage=el('span'),remaining=el('span'),line=el('div',null,'ticket-stage'),progress=el('progress');
        header.append(el('span','#'+order.id),el('span',order.vip?'VIP · $'+order.price:'$'+order.price));line.append(stage,remaining);
        progress.max=order.deadline-order.at;progress.setAttribute('aria-label','Time to serve order '+order.id);
        node.append(header,el('strong',K.recipes[order.recipe].name),line,progress);ticket={node,stage,remaining,progress};view.ticketNodes.set(order.id,ticket);view.tickets.append(node);
      }
      ticket.node.classList.toggle('urgent',order.deadline-game.time<5000||order.stage==='burnt');ticket.stage.textContent=stageNames[order.stage];ticket.remaining.textContent=Math.ceil((order.deadline-game.time)/1000)+'s';ticket.progress.value=Math.max(0,order.deadline-game.time);
    }
    for(const [id,station]of Object.entries(view.stations)){
      const job=game.chefs.find(chef=>chef.job?.station===id)?.job;
      const food=id==='stove'?game.orders.find(order=>['cooking','ready','burnt'].includes(order.stage)):null;
      station.node.classList.toggle('hot',!!food&&food.stage!=='burnt');station.node.classList.toggle('burnt',food?.stage==='burnt');
      station.state.textContent=food?food.stage==='burnt'?'CLEAR PAN':food.stage==='ready'?'READY '+Math.ceil((food.burnAt-game.time)/1000)+'s':'COOK '+Math.ceil((food.readyAt-game.time)/1000)+'s':job?'#'+job.order:'FREE';
    }
    for(const chef of game.chefs){
      const pos=game.position(chef);view.chefs[chef.id].style.left=pos.x/8*100+'%';view.chefs[chef.id].style.top=pos.y/5*100+'%';
      view.jobs[chef.id].label.textContent=chef.job?chef.job.label.split(': ')[1]:'Idle · ready for a job';view.jobs[chef.id].remaining.textContent=chef.job?K.seconds(chef.job.finish-game.time):'';
    }
    view.map.setAttribute('aria-label','Kitchen: '+game.chefs.map(chef=>chef.name+' '+(chef.job?chef.job.label:'idle')).join('; '));
    for(const [id,node]of Object.entries(view.statNodes))node.textContent=(id==='revenue'?'$':'')+game.stats[id];
  }
  function resultText(record){return record.applied==null?'Waiting for the paired round':record.applied?'Assigned':record.conflict?'Conflict · '+record.reason:'Stale · '+record.reason;}
  function renderLane(lane,index){
    const view=views[index],game=lane.game;
    if(!match.cooperative||index===0)renderWorld(view,game);
    view.title.textContent=(match.cooperative?game.chefs[index].name.toUpperCase()+' · ':index?'RIGHT · ':'LEFT · ')+names[lane.player];view.model.textContent=match.warmups[lane.player]?.model||(KitchenMatch.live(lane.player)?match.config.profiles[lane.player].model:'NO API CALLS');
    view.choice.textContent=lane.pending||lane.prepared?lane.status:lane.last?.label||'Ready for service';
    view.detail.textContent=lane.last?resultText(lane.last)+' · observed '+time(lane.last.observedAt)+(lane.last.choiceWarning?' · '+lane.last.choiceWarning:''):lane.player==='human'?'Press Run, then choose a job below.':'Collect, chop, cook, plate, serve.';
    if(view.lastRecord!==lane.last){
      view.lastRecord=lane.last;view.probs.replaceChildren();
      if(lane.last?.probabilities){
        for(const option of lane.last.options.slice().sort((a,b)=>lane.last.probabilities[b.code]-lane.last.probabilities[a.code]).slice(0,3)){
          const row=el('div',null,'prob-row'+(option.code===lane.last.choice?' chosen':'')),track=el('div',null,'prob-track'),fill=el('div',null,'prob-fill');
          fill.style.width=lane.last.probabilities[option.code]*100+'%';track.append(fill);row.append(el('span',option.label),track,el('span',(lane.last.probabilities[option.code]*100).toFixed(0)+'%'));view.probs.append(row);
        }
      }else view.probs.append(el('p',lane.player==='reference'?'Greedy local policy · no model probabilities.':lane.player==='human'?'Your choices · no inference measurements.':'The next response will show action preferences.','screen-note'));
    }
    const stats=SystemOne.summary(lane.records);view.timings.replaceChildren();
    const tokens=lane.records.reduce((sum,record)=>[sum[0]+(record.response?.usage?.input_tokens||0),sum[1]+(record.response?.usage?.output_tokens||0)],[0,0]);
    for(const [label,value]of [['P50',fmt(stats.p50)+' ms'],['P95',fmt(stats.p95)+' ms'],['N',stats.samples],['STALE',lane.records.filter(record=>record.stale).length],...(match.cooperative?[['CONFLICTS',lane.records.filter(record=>record.conflict).length]]:[]),['TOKENS IN/OUT',tokens.join(' / ')],['WARM-UP',match.warmups[lane.player]?fmt(match.warmups[lane.player].ms)+' ms':'—']]){
      const node=el('span',label+' ');node.append(el('b',String(value)));view.timings.append(node);
    }
    const event=game.events.at(-1);view.event.textContent=event?time(event.at)+' · '+event.text:'';
    const options=lane.player==='human'&&lane.human?match.options(lane):[],optionIds=new Set(options.map(option=>option.id));
    for(const [id,button]of view.manualNodes)if(!optionIds.has(id)){button.remove();view.manualNodes.delete(id);}
    for(const option of options){
      let button=view.manualNodes.get(option.id);
      if(!button){button=el('button',option.label);button.type='button';button.addEventListener('click',()=>match.choose(index,option.id));view.manualNodes.set(option.id,button);view.manual.append(button);}
      button.title=option.detail;
    }
    view.manual.hidden=!options.length;
  }
  function inspect(){
    const index=Number($('inspect-lane').value),lane=match.lanes[index];
    const menuKey=index+':'+lane.records.length;
    if(menuKey!==recordMenuKey){
      const previous=$('inspect-record').value;recordMenuKey=menuKey;$('inspect-record').replaceChildren();
      const latest=el('option','Latest');latest.value='latest';$('inspect-record').append(latest);
      lane.records.forEach((record,i)=>{const option=el('option','#'+(i+1)+' · '+time(record.observedAt)+' · '+record.label);option.value=String(i);$('inspect-record').append(option);});
      $('inspect-record').value=previous==='latest'||Number(previous)<lane.records.length?previous:'latest';
    }
    const selected=$('inspect-record').value==='latest'?lane.records.length-1:Number($('inspect-record').value),record=lane.records[selected];
    const key=menuKey+':'+selected+':'+record?.applied+':'+(match.failure?.message||'');if(key===inspectionKey)return;inspectionKey=key;
    $('candidate-table').replaceChildren();$('inspector-count').textContent=lane.records.length+' DECISIONS';
    if(match.failure){
      $('decision-note').textContent='Match paused: '+match.failure.message;
      $('request-wire').textContent=JSON.stringify(match.failure.receipt?.request??null,null,2);$('response-wire').textContent=JSON.stringify(match.failure.receipt?.response??null,null,2);return;
    }
    $('request-wire').textContent=record?.request?JSON.stringify(record.request,null,2):'No API request.';$('response-wire').textContent=record?.response?JSON.stringify(record.response,null,2):'No API response.';
    $('decision-note').textContent=record?resultText(record)+'. '+(record.applied===false?'No substitute was applied. ':'')+'Goal: '+record.goal+'. '+(record.choiceWarning||''):'Run or Step to record the first decision.';
    if(!record)return;
    if(record.probabilities){const total=Object.values(record.probabilities).reduce((sum,p)=>sum+p,0);if(Math.abs(total-1)>1e-5)$('decision-note').textContent+=' Raw probabilities total '+(total*100).toFixed(2)+'%.';}
    for(const option of record.options){const row=el('tr');if(option.code===record.choice)row.className='chosen';row.append(...[option.code,option.label,option.detail,record.probabilities?(record.probabilities[option.code]*100).toFixed(1)+'%':'—'].map(value=>el('td',value)));$('candidate-table').append(row);}
  }
  function render(){
    if(!match)return;
    $('mode').textContent=match.phase;$('power-label').textContent=match.phase;$('led').classList.toggle('on',match.running);
    $('run').textContent=match.running?'PAUSE':match.phase==='READY'?'RUN':'RESUME';$('run').disabled=!!configError||match.lanes.every(lane=>lane.game.over);
    $('step').disabled=!!configError||match.running||match.config.mode!=='paused'||match.lanes.every(lane=>lane.game.over);$('step').title='Advance the simulation by 0.5s in paused decision mode.';
    $('session-settings').disabled=match.running;
    for(const provider of ['qwen','jev'])for(const field of ['endpoint','model','key'])$(provider+'-'+field).disabled=match.running;
    $('clock').textContent=time(match.lanes[0].game.time)+' / '+time(match.config.duration);$('shift-progress').max=match.config.duration;$('shift-progress').value=match.lanes[0].game.time;
    $('goal-label').textContent={revenue:'MAXIMIZE REVENUE',waste:'REDUCE WASTE',vip:'PRIORITIZE VIPS'}[match.config.goal];
    $('clock-help').textContent=match.config.mode==='realtime'?(match.cooperative?'Both chefs decide independently. First valid answer claims the job. Food keeps aging.':'Orders and food keep aging while a model decides. Expired actions are rejected.'):(match.cooperative?'The kitchen freezes for both choices. Conflicting claims rotate priority each round.':'Both kitchens freeze during decisions, then advance 0.5s together.')+' Step runs one round.';
    $('match-note').textContent=match.phase==='SHIFT COMPLETE'?(match.cooperative?'Service finished. One team, one result.':'Shift complete. Compare service, revenue and waste below.'):match.cooperative?'One kitchen. Independent chefs. Shared work and consequences.':match.config.mode==='realtime'?'Same tickets. Same deadlines. The clock keeps running.':'Same tickets. Equal simulated time. The clock waits for both decisions.';
    $('export').disabled=!match.lanes.some(lane=>lane.records.length)&&!match.failure;
    const error=match.failure;
    $('fault').textContent=configError||(error?(error.chefId!=null?['Ada','Bo'][error.chefId]+' · ':'')+(error.player?names[error.player]+': ':'')+error.message+' '+(error.status===401||error.status===403?'Check the provider key.':error.kind==='network'?'Check the provider endpoint and relay.':'Inspect the receipt, then Resume to retry.') : '');
    if(error)$('inspector').open=true;
    match.lanes.forEach(renderLane);
    if(match.cooperative){
      const game=match.games[0];$('team-owners').textContent=match.lanes.map(lane=>game.chefs[lane.chefId].name+' · '+names[lane.player]).join(' / ');
      $('team-metrics').textContent='HANDOFFS '+game.stats.handoffs+' · CONFLICTS '+match.lanes.reduce((sum,lane)=>sum+lane.records.filter(record=>record.conflict).length,0)+' · A handoff continues a dish another chef worked on.';
      const key=game.events.length+':'+JSON.stringify(game.events.at(-1));
      if(activityKey!==key){activityKey=key;$('activity-log').replaceChildren(...game.events.slice(-10).reverse().map(event=>{const row=el('li',null,event.text.includes('rejected:')?'conflict':'');row.append(el('time',time(event.at)),el('span',event.text));return row;}));}
    }
    inspect();
  }
  for(const id of ['layout','left-player','right-player','clock-mode','seed','duration','qwen-endpoint','qwen-model','qwen-key','jev-endpoint','jev-model','jev-key'])$(id).addEventListener('change',reset);
  $('goal').addEventListener('change',()=>match?.setGoal($('goal').value));
  $('local-demo').addEventListener('click',()=>{$('left-player').value='reference';$('right-player').value='reference';reset();});
  $('jev-match').addEventListener('click',()=>{$('left-player').value='jev';$('right-player').value='qwen';$('jev-connection').open=true;reset();});
  $('run').addEventListener('click',()=>{if(!configError)return match?.start();});$('step').addEventListener('click',()=>{if(!configError)return match?.start(true);});$('reset').addEventListener('click',reset);
  for(const id of ['inspect-lane','inspect-record'])$(id).addEventListener('change',()=>{inspectionKey='';if(match)inspect();});
  $('export').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(match.report(),null,2)],{type:'application/json'})),link=el('a');link.href=url;link.download='ninfer-kitchen.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),0);});
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&match?.running)match.pause('PAUSED · TAB HIDDEN');});
  document.addEventListener('keydown',event=>{if(configError||event.repeat||event.ctrlKey||event.metaKey||event.altKey||event.target.closest('input,select,textarea,[contenteditable]'))return;if(event.key?.toLowerCase()==='p'){event.preventDefault();void match?.start();}});
  const params=new URLSearchParams(location.search);
  for(const side of ['left','right'])if(names[params.get(side)])$(side+'-player').value=params.get(side);
  if(['paused','realtime'].includes(params.get('mode')))$('clock-mode').value=params.get('mode');
  if(['cooperative','comparison'].includes(params.get('layout')))$('layout').value=params.get('layout');
  if(K.goals[params.get('goal')])$('goal').value=params.get('goal');
  if(params.has('seed'))$('seed').value=params.get('seed');
  if(params.get('match')==='jev-qwen'){$('left-player').value='jev';$('right-player').value='qwen';$('jev-connection').open=true;}
  reset();
})();
