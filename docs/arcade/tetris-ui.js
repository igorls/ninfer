(() => {
  const $=id=>document.getElementById(id), T=Tetris;
  const params=new URLSearchParams(location.search);
  for (const [key,id] of [['endpoint','endpoint'],['model','model'],['seed','seed'],['row','row'],['pace','pace'],['motion','motion-speed'],['player','controller'],['scenario','scenario']]) {
    if (params.has(key)) $(id).value=params.get(key);
  }
  $('strict').checked=params.get('strict')==='1';
  const cells=Array.from({length:200},()=>{
    const cell=document.createElement('div'); cell.className='cell'; cell.setAttribute('aria-hidden','true');
    $('well').insertBefore(cell,$('banner')); return cell;
  });
  const motionTiles=Array.from({length:4},()=>{
    const tile=document.createElement('i');$('piece-motion').appendChild(tile);return tile;
  });
  const sourceNames={live:'LIVE MODEL',human:'HUMAN PLAYER',reference:'LOCAL REFERENCE'};
  let config,game,records=[],last=null,failure=null,warmup=null,running=false,epoch=0,aborter=null,frame=0,manual=null;
  let activeRequest=null,activeOptions=null,activePhase=null,activeSince=0;
  let sessionStarted=new Date().toISOString();
  const fmt=n=>n==null?'—':Math.round(n).toString();
  const pct=p=>p==null?'—':(100*p).toFixed(1)+'%';
  const speedLabel=n=>Number(n).toFixed(2).replace(/\.?0+$/,'')+'×';
  function mode(text) { $('mode').textContent=text; }
  function banner(title,detail,error=false) {
    $('banner-title').textContent=title; $('banner-detail').textContent=detail;
    $('banner').classList.toggle('error',error);$('banner').classList.add('show');
  }
  function fault(error) {
    $('fault').textContent=(error.message||String(error))+' Check the session settings and retry.';
    banner('CHECK SETTINGS','Correct the session or connection values, then press Run.',true);
    mode('CHECK SETTINGS');$('power-label').textContent='CHECK SETTINGS';
  }
  function inspectFailure(error) {
    const receipt=error.receipt;
    failure={phase:activePhase,piece:game.pieces+1,kind:error.kind||'request',status:error.status??null,
      message:error.name==='TimeoutError'?'No complete answer within 10 seconds.':error.message||String(error),
      ms:receipt?.ms??performance.now()-activeSince,request:receipt?.request??activeRequest,
      response:receipt?.response??null};
    $('dt').textContent='—';$('latency-label').textContent='FAILED REQUEST · NOT A SAMPLE';
    $('dt-wrap').classList.remove('late');$('tokens').textContent='—';$('selection-prob').textContent='—';
    $('decision-title').textContent=activePhase==='warmup'?'WARM-UP FAILED':'PIECE '+failure.piece+' · REQUEST FAILED';
    $('decision-note').textContent=failure.message+' The board has not advanced.';
    $('probabilities').replaceChildren();
    const note=document.createElement('p');note.className='screen-note';
    note.textContent='No valid model choice or probability distribution. Failed requests are excluded from match timings.';
    $('probabilities').appendChild(note);
    $('landing-table').replaceChildren();
    for(const o of activeOptions||[]) {
      const tr=document.createElement('tr');
      for(const value of [o.code,o.column+1,o.rot,o.lines,o.holes,o.height,o.bump,'—']) {
        const td=document.createElement('td');td.textContent=value;tr.appendChild(td);
      }
      $('landing-table').appendChild(tr);
    }
    $('inspector-count').textContent='FAILED REQUEST · '+(activeOptions?.length||0)+' LANDINGS';
    $('inspector-label').textContent='INSPECT FAILED REQUEST';
    $('request-wire').textContent=failure.request?JSON.stringify(failure.request,null,2):'No request was sent.';
    $('response-wire').textContent=failure.response==null?'No response body.':typeof failure.response==='string'?failure.response:JSON.stringify(failure.response,null,2);
    $('inspector').open=true;$('wire-details').open=true;
    $('fault').textContent=failure.message+' Press Step or Run to retry this piece. The failed request is available below and in Export run.';
    banner('REQUEST FAILED','Piece '+failure.piece+' is unchanged. Inspect the response, then retry.',true);
    mode('REQUEST FAILED');
  }
  function readConfig() {
    const seed=Number($('seed').value),row=Number($('row').value),pace=Number($('pace').value),motionSpeed=Number($('motion-speed').value);
    if (!$('seed').value.trim()||!Number.isInteger(seed)||seed<0||seed>4294967295) throw new Error('Seed must be an integer from 0 to 4294967295.');
    if (!Number.isFinite(row)||row<4||row>40||!Number.isFinite(pace)||pace<0||pace>800||!Number.isFinite(motionSpeed)||motionSpeed<0.5||motionSpeed>3) throw new Error('Choose a valid gravity, pace and piece speed.');
    const source=$('controller').value,scenario=$('scenario').value;
    if (!sourceNames[source]||!['empty','trench','stairs'].includes(scenario)) throw new Error('Choose a player and starting board.');
    const model=$('model').value.trim();
    let endpoint=$('endpoint').value.trim().replace(/\/+$/,'');
    if (source==='live') {
      const url=new URL(endpoint);
      if (!['http:','https:'].includes(url.protocol)||url.username||url.password||url.search||url.hash) throw new Error('Use an HTTP API base URL without credentials, query parameters, or a fragment.');
      if (!model) throw new Error('Enter the model label.');
      endpoint=url.href.replace(/\/+$/,'');
    }
    return {seed,row,pace,motionSpeed,source,scenario,model,endpoint,strict:$('strict').checked};
  }
  function controls() {
    $('run').textContent=running?'PAUSE':game?.pieces?'RESUME':'RUN';
    $('step').disabled=running;
    $('session-settings').disabled=running;
    for(const id of ['endpoint','model','api-key','pace']) $(id).disabled=running;
    $('export').disabled=!records.length&&!failure;
    $('led').classList.toggle('on',running);
    $('power-label').textContent=running?'SESSION ACTIVE':failure?'REQUEST FAILED':'READY TO PLAY';
    $('manual-controls').hidden=config?.source!=='human';
    for(const id of ['left','right','rotate','drop']) $(id).disabled=!manual;
  }
  function stage(name) {
    for(const s of ['enumerate','decide','place']) $('stage-'+s).classList.toggle('active',s===name);
  }
  function paint({active=null,ghost=null,grid=game.board}={}) {
    const classes=grid.flat().map(k=>'cell'+(k?' '+k:''));
    if(ghost) for(const [x,y] of ghost.cells) classes[y*10+x]='cell ghost';
    if(active) for(const [x,y] of active.cells) if(y>=0&&y<20) classes[y*10+x]='cell '+active.kind+' live';
    cells.forEach((cell,i)=>{if(cell.className!==classes[i])cell.className=classes[i];});
    const m=T.metrics(grid);
    $('well').setAttribute('aria-label','Tetris board: '+game.lines+' lines cleared, '+game.pieces+' pieces, stack '+m.height+' of 20 rows, '+m.holes+' holes.');
    $('height').textContent=m.height+' / 20'; $('holes').textContent=m.holes;
  }
  function renderGame() {
    paint();
    $('scoreboard').textContent=game.lines+' LINES · '+game.pieces+' PIECES';
    $('score').textContent='SCORE '+game.score;
    $('piece-label').textContent='PIECE '+game.kind+' · SEED '+config.seed;
    $('queue').replaceChildren();
    for(const kind of game.queue.slice(1,4)) {
      const next=document.createElement('div');next.className='next';next.setAttribute('aria-label','Upcoming '+kind);
      const shape=T.SHAPES[kind][0],occupied=new Set(shape.map(([x,y])=>x+y*4));
      for(let i=0;i<12;i++) {const c=document.createElement('div');if(occupied.has(i))c.className=kind;next.appendChild(c);}
      $('queue').appendChild(next);
    }
  }
  function renderStats() {
    const s=T.summary(records);
    $('p50').textContent=fmt(s.p50);$('p95').textContent=fmt(s.p95);$('samples').textContent=s.samples;
    $('agreement').textContent=s.evaluated?s.matches+' / '+s.evaluated+' · '+pct(s.matches/s.evaluated):'—';
    $('on-time').textContent=s.deadlines?(s.deadlines-s.misses)+' / '+s.deadlines:'—';
    const data=records.filter(r=>r.source==='live'&&Number.isFinite(r.ms)).slice(-40).map(r=>r.ms);
    const max=Math.max(1,...data);
    $('latency-line').setAttribute('d',data.map((v,i)=>(i?'L':'M')+(2+i*316/Math.max(1,data.length-1)).toFixed(1)+','+(57-v/max*52).toFixed(1)).join(' '));
    $('latency-chart').setAttribute('aria-label',data.length?'Last '+data.length+' request latencies, maximum '+fmt(max)+' milliseconds.':'No live request timings yet.');
    controls();
  }
  function inspect(record) {
    last=record;failure=null;
    $('inspector-label').textContent='INSPECT LAST DECISION';
    const chosen=record.options.find(o=>o.code===record.choice);
    $('dt').textContent=record.source==='live'?fmt(record.ms):'—';
    $('latency-label').textContent=record.source==='live'?'REQUEST → VALID ANSWER':record.source==='human'?'HUMAN · UNTIMED':'LOCAL POLICY · NO API CALL';
    $('dt-wrap').classList.toggle('late',record.deadlineMet===false);
    $('decision-title').textContent=chosen?(record.deadlineMet===false&&config.strict?'NOT PLACED · ':'LANDING ')+record.choice+' · COLUMN '+(chosen.column+1):'DEADLINE EXPIRED';
    $('selection-prob').textContent=record.probabilities?pct(record.probabilities[record.choice]):'—';
    $('decision-note').textContent=chosen?'+'+chosen.lines+' lines · '+chosen.holes+' holes · height '+chosen.height+' · '+(record.policyMatch?'matches reference policy':'differs from reference policy'):'Request cancelled at the deadline. No valid answer was applied.';
    if(record.choiceWarning)$('decision-note').textContent+=' '+record.choiceWarning;
    if(Number.isFinite(record.probabilityTotal)&&Math.abs(record.probabilityTotal-1)>1e-8)
      $('decision-note').textContent+=' Raw probabilities total '+pct(record.probabilityTotal)+'; values are shown without normalization.';
    $('tokens').textContent=record.response?record.response.usage.input_tokens+' / '+record.response.usage.output_tokens:'—';
    $('probabilities').replaceChildren();
    if(record.probabilities) {
      for(const o of record.options.slice().sort((a,b)=>record.probabilities[b.code]-record.probabilities[a.code]).slice(0,5)) {
        const row=document.createElement('div');row.className='prob-row'+(o.code===record.choice?' chosen':'');
        const code=document.createElement('span');code.textContent=o.code;
        const track=document.createElement('div');track.className='prob-track';
        const fill=document.createElement('div');fill.className='prob-fill';fill.style.width=(record.probabilities[o.code]*100)+'%';track.appendChild(fill);
        const value=document.createElement('span');value.textContent=pct(record.probabilities[o.code]);
        row.append(code,track,value);$('probabilities').appendChild(row);
      }
    } else {
      const note=document.createElement('p');note.className='screen-note';
      note.textContent=record.source==='live'?'No complete response before the deadline. This is not a latency sample.':record.source==='reference'?'Deterministic reference policy. No model probabilities or inference timings are produced.':'Your placement. No API request or model probability.';
      $('probabilities').appendChild(note);
    }
    $('landing-table').replaceChildren();
    for(const o of record.options) {
      const tr=document.createElement('tr');if(o.code===record.choice)tr.className='selected';
      for(const value of [o.code+(o.code===record.choice?' ✓':''),o.column+1,o.rot,o.lines,o.holes,o.height,o.bump,record.probabilities?pct(record.probabilities[o.code]):'—']) {
        const td=document.createElement('td');td.textContent=value;tr.appendChild(td);
      }
      $('landing-table').appendChild(tr);
    }
    $('inspector-count').textContent=record.options.length+' LANDINGS';
    $('request-wire').textContent=record.request?JSON.stringify(record.request,null,2):'No API request in '+record.source+' mode.';
    $('response-wire').textContent=record.response?JSON.stringify(record.response,null,2):'No API response.';
  }
  function stop(status='PAUSED') {
    epoch++;aborter?.abort();aborter=null;cancelAnimationFrame(frame);manual=null;running=false;
    activeRequest=null;activeOptions=null;activePhase=null;
    $('piece-motion').hidden=true;
    $('manual-status').textContent='';
    if(game)renderGame();
    mode(status);controls();
  }
  function reset() {
    stop('STANDBY');
    try {config=readConfig();} catch(error) {fault(error);return;}
    game=new T.Game(config.seed,config.scenario);records=[];last=null;failure=null;warmup=null;
    activeRequest=null;activeOptions=null;activePhase=null;sessionStarted=new Date().toISOString();
    $('source').textContent=sourceNames[config.source];
    $('session-note').textContent=config.source==='live'?'LIVE API · NO HEURISTIC OVERRIDE':config.source==='reference'?'LOCAL REFERENCE · NO INFERENCE':'MANUAL PLACEMENT · NO INFERENCE';
    $('run-id').textContent='SEED '+config.seed;
    $('row-value').textContent=config.row+' ms / row';$('pace-value').textContent=config.pace+' ms';
    $('motion-value').textContent=speedLabel(config.motionSpeed);
    $('fault').textContent='';$('banner').classList.remove('show');$('piece-motion').hidden=true;
    for(const id of ['dt','p50','p95','selection-prob','tokens','warmup','budget'])$(id).textContent='—';
    $('dt-wrap').classList.remove('late');$('decision-title').textContent='WAITING FOR A DECISION';
    $('decision-note').textContent='Run continuously, or use Step to inspect one piece.';
    $('latency-label').textContent=config.source==='live'?'REQUEST → VALID ANSWER':'NO API TIMING';
    $('budget-note').textContent=config.source==='live'?'Warm-up is measured separately from timed decisions.':'Reference and human play do not measure API latency.';
    $('budget-label').textContent=config.source==='human'?'MANUAL PLACEMENT · UNTIMED':'SPAWN-TO-LOCK BUDGET';
    $('probabilities').replaceChildren();$('landing-table').replaceChildren();
    $('request-wire').textContent='No request yet.';$('response-wire').textContent='No response yet.';
    $('inspector-label').textContent='INSPECT LAST DECISION';$('wire-details').open=false;
    $('inspector-count').textContent='';$('deadline').value=1;$('option-count').textContent='—';
    stage('');renderGame();renderStats();
    banner(config.source==='human'?'YOUR TURN':'READY',config.source==='human'?'Run, choose a landing, then Place.':'Run the session or inspect one step.');
  }
  function delay(ms,signal) {
    return new Promise((resolve,reject)=>{
      if(signal.aborted){reject(new DOMException('Aborted','AbortError'));return;}
      const finish=()=>{signal.removeEventListener('abort',cancel);resolve();};
      const timer=setTimeout(finish,ms);
      const cancel=()=>{clearTimeout(timer);reject(new DOMException('Aborted','AbortError'));};
      signal.addEventListener('abort',cancel,{once:true});
    });
  }
  function animateFall(start,budget,spawn,kind,my,options) {
    $('deadline').value=1;
    if(matchMedia('(prefers-reduced-motion: reduce)').matches)return;
    const tick=()=>{
      if(my!==epoch)return;
      const elapsed=performance.now()-start,rows=Math.min(spawn.rows,Math.floor(elapsed/config.row));
      $('deadline').value=Math.max(0,1-elapsed/budget);
      paint({active:{kind,cells:spawn.shape.map(([x,y])=>[x+spawn.x,y+rows])}});
      if(elapsed<budget)frame=requestAnimationFrame(tick);
    };
    tick();
  }
  function animatePlacement(kind,option,spawn,before,decisionStart,my,signal) {
    if(config.source==='human'||matchMedia('(prefers-reduced-motion: reduce)').matches)return Promise.resolve(0);
    const start=performance.now();
    const row=config.source==='live'?Math.min(spawn.rows,Math.floor((start-decisionStart)/config.row)):0;
    const from=spawn.shape.map(([x,y])=>[x+spawn.x,y+row]),to=option.cells;
    const motion=$('piece-motion');motion.className='piece-motion '+kind;motion.hidden=false;
    motionTiles.forEach((tile,i)=>{tile.style.left=(from[i][0]*10)+'%';tile.style.top=(from[i][1]*5)+'%';});
    paint({grid:before});
    const ease=t=>1-(1-t)**3;
    const draw=progress=>{
      const steer=ease(Math.min(1,progress/.42));
      const drop=ease(Math.max(0,(progress-.12)/.88));
      motionTiles.forEach((tile,i)=>{
        const x=(to[i][0]-from[i][0])*steer*100,y=(to[i][1]-from[i][1])*drop*100;
        tile.style.transform='translate3d('+x+'%,'+y+'%,0)';
      });
    };
    draw(0);
    return new Promise((resolve,reject)=>{
      let previous=start,progress=0;
      const cancel=()=>{
        cancelAnimationFrame(frame);motion.hidden=true;signal.removeEventListener('abort',cancel);
        reject(new DOMException('Aborted','AbortError'));
      };
      if(signal.aborted){cancel();return;}
      signal.addEventListener('abort',cancel,{once:true});
      const tick=()=>{
        if(my!==epoch||signal.aborted){cancel();return;}
        if(matchMedia('(prefers-reduced-motion: reduce)').matches){
          draw(1);signal.removeEventListener('abort',cancel);resolve(performance.now()-start);return;
        }
        const now=performance.now(),delta=Math.min(64,Math.max(0,now-previous));previous=now;
        const speed=Math.max(.5,Math.min(3,Number($('motion-speed').value)||1));
        progress=Math.min(1,progress+delta*speed/620);draw(progress);
        if(progress<1)frame=requestAnimationFrame(tick);
        else {signal.removeEventListener('abort',cancel);resolve(now-start);}
      };
      frame=requestAnimationFrame(tick);
    });
  }
  function manualChoice(options,signal) {
    return new Promise((resolve,reject)=>{
      const cancel=()=>{manual=null;reject(new DOMException('Aborted','AbortError'));};
      signal.addEventListener('abort',cancel,{once:true});
      const first=options.find(o=>o.rot===0&&o.column>=3)||options[0];
      manual={options,index:options.indexOf(first),resolve:o=>{signal.removeEventListener('abort',cancel);manual=null;resolve({option:o});}};
      mode('YOUR TURN');controls();previewManual(first);
    });
  }
  function previewManual(option) {
    paint({ghost:option});
    $('decision-title').textContent='PREVIEW '+option.code+' · COLUMN '+(option.column+1);
    $('decision-note').textContent='+'+option.lines+' lines · '+option.holes+' holes · height '+option.height;
    $('manual-status').textContent='Landing '+option.code+', column '+(option.column+1)+', rotation '+(option.rot*90)+' degrees. Clears '+option.lines+' lines; leaves '+option.holes+' holes and stack height '+option.height+'.';
  }
  function selectManual(action) {
    if(!manual)return;
    const current=manual.options[manual.index];
    if(action==='drop'){manual.resolve(current);controls();return;}
    let selected;
    if(action==='rotate'){
      const rotations=[...new Set(manual.options.map(o=>o.rot))];
      const next=rotations[(rotations.indexOf(current.rot)+1)%rotations.length];
      selected=manual.options.filter(o=>o.rot===next).sort((a,b)=>Math.abs(a.column-current.column)-Math.abs(b.column-current.column))[0];
    } else {
      const options=manual.options.filter(o=>o.rot===current.rot);
      const index=options.indexOf(current);
      selected=options[Math.max(0,Math.min(options.length-1,index+(action==='left'?-1:1)))];
    }
    manual.index=manual.options.indexOf(selected);previewManual(selected);
  }
  async function playPiece(my,signal) {
    const start=performance.now();
    const kind=game.kind,options=game.options(),spawn=T.spawn(game.board,kind);
    if(!spawn||!options.length){banner('TOPPED OUT',game.lines+' lines · '+game.pieces+' pieces. Reset to replay the seed.');return false;}
    const before=game.board.map(r=>r.slice()),budget=Math.max(1,spawn.rows)*config.row;
    const reference=T.reference(options);
    $('option-count').textContent=options.length;$('banner').classList.remove('show');$('fault').textContent='';
    $('budget').textContent=config.source==='human'?'UNTIMED':fmt(budget)+' ms';
    $('budget-note').textContent=config.source==='human'?'Choose a column and rotation. Space or Place locks it.':config.strict?'Deadline challenge: a late response ends this run.':'Watch mode: late answers count as misses and still play.';
    stage('decide');mode(config.source==='human'?'YOUR TURN':config.source==='reference'?'LOCAL POLICY':'DECIDING');
    let result,request=null;
    if(config.source==='live'){
      request=T.requestFor(game,options,config.model);
      activeRequest=request;activeOptions=options;activePhase='decision';activeSince=performance.now();
      animateFall(start,budget,spawn,kind,my,options);
      const timedSignal=AbortSignal.any([signal,AbortSignal.timeout(10000)]);
      const ask=s=>T.decide(config.endpoint,request,options,{signal:s,apiKey:$('api-key').value});
      result=config.strict?await T.withDeadline(ask,budget-(performance.now()-start),timedSignal):await ask(timedSignal);
      activeRequest=null;activeOptions=null;activePhase=null;
    } else if(config.source==='reference') {
      result={option:reference};
      await delay(80,signal);
    } else result=await manualChoice(options,signal);
    if(my!==epoch)return false;
    cancelAnimationFrame(frame);
    // Deadline covers enumeration, request construction, full body parsing and validation.
    const elapsed=performance.now()-start;
    const deadlineMet=config.source==='live'?!result.expired&&elapsed<=budget:null;
    const record={piece:game.pieces+1,kind,next:game.queue[1],source:config.source,board:before,
      options,choice:result.option?.code??null,reference:reference.code,policyMatch:result.option?T.compare(result.option,reference)===0:null,
      ms:result.ms??null,decisionMs:config.source==='live'?elapsed:null,budgetMs:config.source==='live'?budget:null,
        deadlineMet,probabilities:result.probabilities??null,probabilityTotal:result.probabilityTotal??null,
        choiceWarning:result.choiceWarning??null,request,response:result.body??null,placed:false};
    records.push(record);inspect(record);
    $('deadline').value=deadlineMet==null?1:Math.max(0,1-elapsed/budget);
    if(deadlineMet===false&&config.strict) {
      const fallback=spawn.shape.map(([x,y])=>[x+spawn.x,y+spawn.rows]);
      const locked=T.place(game.board,fallback,kind);
      game.board=locked.grid;game.lines+=locked.lines;game.pieces++;game.score+=[0,100,300,500,800][locked.lines];
      record.fallback={cells:fallback,lines:locked.lines};record.boardAfter=game.board.map(r=>r.slice());
      renderGame();renderStats();mode('DEADLINE MISSED');
      banner('DEADLINE MISSED',fmt(elapsed)+' ms decision / '+fmt(budget)+' ms budget. Spawn column locked.');
      return false;
    }
    stage('place');mode('LANDING');
    $('manual-status').textContent='';
    const committed=game.commit(result.option);record.placed=true;record.cleared=committed.lines;record.boardAfter=game.board.map(r=>r.slice());
    record.motionSpeedAtStart=config.motionSpeed;record.presentationMs=null;
    renderStats();
    record.presentationMs=await animatePlacement(kind,result.option,spawn,before,start,my,signal);
    record.motionSpeedAtEnd=config.motionSpeed;
    if(my!==epoch)return false;
    $('piece-motion').hidden=true;renderGame();mode('PLACED');
    if(!committed.lines)paint({active:{kind,cells:result.option.cells}});
    if(committed.lines) {
      $('piece-label').textContent='+'+committed.lines+' LINE'+(committed.lines>1?'S':'')+' · '+game.kind+' NEXT';
      if(!matchMedia('(prefers-reduced-motion: reduce)').matches) {
        for(const y of committed.clearedRows)for(let x=0;x<10;x++)cells[y*10+x].classList.add('clearing');
      }
    }
    return true;
  }
  async function start(single=false) {
    if(running){stop();return;}
    if($('banner-title').textContent==='TOPPED OUT'||$('banner-title').textContent==='DEADLINE MISSED'||records.length>=500)reset();
    try{config=readConfig();}catch(error){fault(error);return;}
    if(!game)reset();
    const my=++epoch;aborter=new AbortController();const signal=aborter.signal;
    running=true;controls();$('fault').textContent='';$('banner').classList.remove('show');
    try {
      if(config.source==='live'&&warmup==null) {
        mode('WARMING UP');stage('enumerate');
        const options=game.options();
        if(options.length){
          activeRequest=T.requestFor(game,options,config.model);activeOptions=options;activePhase='warmup';activeSince=performance.now();
          const result=await T.decide(config.endpoint,activeRequest,options,{
            signal:AbortSignal.any([signal,AbortSignal.timeout(10000)]),apiKey:$('api-key').value});
          if(my!==epoch)return;
          activeRequest=null;activeOptions=null;activePhase=null;
          warmup=result.ms;$('warmup').textContent=fmt(warmup)+' ms · excluded';
        }
      }
      while(my===epoch&&records.length<500) {
        const keep=await playPiece(my,signal);
        if(!keep||single)break;
        await delay(config.pace,signal);
      }
      if(my===epoch) {
        if(records.length>=500){mode('RUN COMPLETE');banner('500 PIECES',game.lines+' lines. Export this run or reset.');}
        else if($('mode').textContent==='PLACED')mode(single?'STEP COMPLETE':'PAUSED');
      }
    } catch(error) {
      if(my===epoch){cancelAnimationFrame(frame);paint();activeRequest?inspectFailure(error):fault(error);}
    } finally {
      if(my===epoch){activeRequest=null;activeOptions=null;activePhase=null;running=false;manual=null;aborter=null;controls();}
    }
  }
  function exportRun() {
    const report={schema:'ninfer.tetris.run.v1',createdAt:sessionStarted,exportedAt:new Date().toISOString(),
      config,policy:T.requestFor(game,game.options(),config.model).questions.move.instructions,
      timing:'Client request through body read, JSON parsing and validation. Warm-up and presentation excluded. decisionMs includes request preparation.',
      warmupMs:warmup,summary:{...T.summary(records),lines:game.lines,pieces:game.pieces,score:game.score},decisions:records,failure};
    const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));
    const a=document.createElement('a');a.href=url;a.download='ninfer-tetris-'+config.source+'-seed-'+config.seed+'.json';a.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  $('run').addEventListener('click',()=>start());
  $('step').addEventListener('click',async()=>{await start(true);if(last)$('inspector').open=true;});
  $('reset').addEventListener('click',reset);
  $('export').addEventListener('click',exportRun);
  for(const id of ['left','right','rotate','drop'])$(id).addEventListener('click',()=>selectManual(id));
  for(const id of ['controller','scenario','seed','row','strict','endpoint','model','pace'])$(id).addEventListener('change',reset);
  $('row').addEventListener('input',()=>{$('row-value').textContent=$('row').value+' ms / row';});
  $('pace').addEventListener('input',()=>{$('pace-value').textContent=$('pace').value+' ms';});
  $('motion-speed').addEventListener('input',()=>{
    $('motion-value').textContent=speedLabel($('motion-speed').value);
    if(config)config.motionSpeed=Number($('motion-speed').value);
  });
  document.addEventListener('keydown',event=>{
    if(event.repeat||event.ctrlKey||event.metaKey||event.altKey||event.target.closest('input,select,textarea,[contenteditable]:not([contenteditable="false"])'))return;
    const actions={ArrowLeft:'left',ArrowRight:'right',ArrowUp:'rotate',' ':'drop'};
    if(manual&&actions[event.key]&&!event.target.closest('summary,a')&&
      !(event.key===' '&&event.target.closest('button'))){event.preventDefault();selectManual(actions[event.key]);}
    else if(event.key.toLowerCase()==='p'){event.preventDefault();running?stop():start();}
    else if(event.key.toLowerCase()==='r'){event.preventDefault();reset();}
  });
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&running)stop('PAUSED · TAB HIDDEN');});
  reset();
})();
