(() => {
  'use strict';
  const $=id=>document.getElementById(id);
  const icons={
    p:'<circle cx="24" cy="12" r="6"/><path d="M19 19h10l-2 5 3 10H18l3-10zM16 35h16v5H16z"/>',
    n:'<path d="M14 35c0-8 9-10 12-15l-8 4-6-4 7-10 2-6 6 4c11 3 12 14 7 27zM13 36h23v5H13z"/><path d="m20 16 3-2M27 11l3 4" fill="none"/>',
    b:'<path d="M24 5c-3 4-10 9-10 15 0 5 4 8 7 9l-4 6h14l-4-6c4-2 7-5 7-9 0-6-7-11-10-15zM14 36h20v5H14z"/><path d="m25 12-5 10" fill="none"/>',
    r:'<path d="M12 7h6v6h4V7h4v6h4V7h6v12l-6 5 2 11H16l2-11-6-5zM13 36h22v5H13z"/><path d="M16 20h16" fill="none"/>',
    q:'<path d="m12 15 5 7 7-12 7 12 5-7-5 19H17zM14 35h20v6H14z"/><circle cx="11" cy="12" r="3"/><circle cx="24" cy="7" r="3"/><circle cx="37" cy="12" r="3"/><path d="M18 29h12" fill="none"/>',
    k:'<path d="M24 4v10m-5-6h10" fill="none"/><path d="M24 16c-10-9-18 1-11 9l5 9h12l5-9c7-8-1-18-11-9zM14 35h20v6H14z"/><path d="M18 28h12" fill="none"/>'
  };
  const sourceName={qwen:'QWEN',jev:'JEV',human:'YOU',reference:'LOCAL REFERENCE'};
  const isLive=player=>player==='qwen'||player==='jev';
  let game,records=[],moves=[],running=false,epoch=0,controller=null,warmups={};
  let selected=null,promotionMoves=null,humanResolve=null,orientation='w',replay=null,focused='e2',preview=null;
  let latest=null,failure=null,stageIndex=0,deadlineStart=null,raf=null,motion=null;
  const reducedMotion=matchMedia('(prefers-reduced-motion: reduce)');
  reducedMotion.addEventListener('change',()=>{if(reducedMotion.matches)motion?.finish();});
  const settings=['white','black','scenario','qwen-endpoint','qwen-model','qwen-key','jev-endpoint','jev-model','jev-key','budget-ms','strict','load-fen','jev-v-qwen','swap-sides'];
  const connection=player=>({endpoint:$(player+'-endpoint').value.trim(),model:$(player+'-model').value.trim(),apiKey:$(player+'-key').value});
  const publicConnection=player=>{const {endpoint,model}=connection(player);return {endpoint,model};};
  const source=()=>$(game.turn==='w'?'white':'black').value;
  const ms=value=>Number.isFinite(value)?value.toFixed(1):'—';
  const element=(tag,text,cls)=>{const node=document.createElement(tag);if(text!=null)node.textContent=text;if(cls)node.className=cls;return node;};
  const pieceMarkup=piece=>'<svg viewBox="0 0 48 48" aria-hidden="true" class="piece '+(piece.color==='w'?'white':'black')+'">'+icons[piece.type]+'</svg>';
  function setMode(text){$('mode').textContent=text;$('power-label').textContent=text;$('led').classList.toggle('on',running);}
  function controls(){
    $('run').textContent=running?'PAUSE':game?.ply?'RESUME':'RUN';
    $('step').disabled=running;
    for(const id of settings)$(id).disabled=running;
    $('start-fen').disabled=running;
    $('export').disabled=!records.length&&!failure;
    $('save-pgn').disabled=!moves.length;
  }
  function stop(text='PAUSED'){
    running=false;epoch++;controller?.abort();controller=null;humanResolve=null;
    selected=null;promotionMoves=null;deadlineStart=null;
    if(raf!=null)cancelAnimationFrame(raf);raf=null;
    $('promotion').hidden=true;setMode(text);controls();
    if(game)renderBoard();
  }
  function reset(){
    let next;
    try {next=new ChessGame.Game($('scenario').value==='custom'?$('start-fen').value.trim():ChessGame.positions[$('scenario').value]);}
    catch(error){$('fault').textContent='Invalid starting position: '+error.message;return;}
    stop('STANDBY');game=next;records=[];moves=[];latest=null;failure=null;replay=null;preview=null;
    warmups={};stageIndex=0;deadlineStart=null;
    $('fault').textContent='';$('warmup').textContent='—';$('deadline').value=1;
    $('fen-input').hidden=$('scenario').value!=='custom';
    $('budget').textContent=$('budget-ms').value+' ms';$('budget-value').textContent=$('budget').textContent;
    $('session-note').textContent=sourceName[$('white').value]+' WHITE / '+sourceName[$('black').value]+' BLACK';
    game.rules.setHeader('White',isLive($('white').value)?connection($('white').value).model:sourceName[$('white').value]);
    game.rules.setHeader('Black',isLive($('black').value)?connection($('black').value).model:sourceName[$('black').value]);
    render();renderDecision();controls();
  }
  function displayedRules(){return replay==null?game.rules:new ChessRules.Chess(replay===0?game.initialFen:moves[replay-1].afterFen);}
  function renderBoard(){
    motion?.finish();
    const rules=displayedRules(),legal=humanResolve&&replay==null?game.options():[];
    const files=orientation==='w'?'abcdefgh':'hgfedcba',ranks=orientation==='w'?'87654321':'12345678';
    const last=replay==null?moves.at(-1):moves[replay-1];
    const board=$('board');board.replaceChildren();
    board.setAttribute('aria-label','Chessboard, '+ChessGame.label(orientation)+' at the bottom');
    for(let row=0;row<8;row++)for(let column=0;column<8;column++){
      const square=files[column]+ranks[row],piece=rules.get(square),destinations=legal.filter(o=>o.from===selected&&o.to===square);
      const button=element('button',null,'square '+((square.charCodeAt(0)+Number(square[1]))%2?'light':'dark'));
      button.type='button';button.dataset.square=square;button.tabIndex=square===focused?0:-1;
      button.setAttribute('aria-label',square+(piece?' '+ChessGame.label(piece.color)+' '+ChessGame.names[piece.type]:' empty')+(destinations.length?', legal destination':''));
      button.setAttribute('aria-pressed',square===selected?'true':'false');
      for(const [name,on]of Object.entries({'has-piece':!!piece,selected:square===selected,destination:!!destinations.length,'last-move':last&&(last.from===square||last.to===square),preview:preview&&(preview.from===square||preview.to===square),check:piece?.type==='k'&&piece.color===rules.turn()&&rules.isCheck()}))button.classList.toggle(name,!!on);
      if(piece)button.innerHTML=pieceMarkup(piece);
      if(column===0)button.append(element('span',ranks[row],'coordinate rank'));
      if(row===7)button.append(element('span',files[column],'coordinate file'));
      button.addEventListener('click',()=>selectSquare(square));
      button.addEventListener('keydown',event=>{
        const direction={ArrowLeft:[0,-1],ArrowRight:[0,1],ArrowUp:[-1,0],ArrowDown:[1,0]}[event.key];
        if(!direction)return;event.preventDefault();
        focused=files[Math.max(0,Math.min(7,column+direction[1]))]+ranks[Math.max(0,Math.min(7,row+direction[0]))];
        renderBoard();focusSquare();
      });
      board.append(button);
    }
    $('current-fen').textContent=rules.fen();
    const status=game.status();
    $('board-banner').hidden=replay!=null||!status.over;
    $('board-banner').textContent=status.text;
    $('board-note').textContent=replay!=null?'REPLAY · PLY '+replay:humanResolve?'SELECT A PIECE, THEN A DESTINATION':running?'DECISION IN PROGRESS':status.over?'GAME COMPLETE':'PRESS RUN OR SELECT YOUR PIECE';
    $('ply-count').textContent='PLY '+(replay??game.ply);
    $('previous').disabled=(replay??moves.length)<=0;
    $('next-position').disabled=replay==null||replay>=moves.length;
    $('return-live').disabled=replay==null;
  }
  // Presentation only: the rules, history and measured decision are committed before movement.
  function animateMove(move,beforeFen){
    const board=$('board');
    if(reducedMotion.matches||document.hidden||typeof board.animate!=='function')return Promise.resolve();
    const before=new ChessRules.Chess(beforeFen),layer=element('div',null,'piece-motion');
    layer.setAttribute('aria-hidden','true');
    const animations=[],hiddenSquares=[],bannerHidden=$('board-banner').hidden;
    let resolve,finished=false;
    const done=new Promise(complete=>{resolve=complete;});
    const state={finish:()=>{
      if(finished)return;finished=true;
      for(const animation of animations)animation.cancel();
      for(const square of hiddenSquares)square.classList.remove('piece-in-flight');
      layer.remove();$('board-banner').hidden=bannerHidden;
      if(motion===state)motion=null;
      resolve();
    }};
    motion=state;$('board-banner').hidden=true;board.append(layer);
    const point=square=>{
      const file=square.charCodeAt(0)-97,rank=Number(square[1])-1;
      return orientation==='w'?[file,7-rank]:[7-file,rank];
    };
    const sprite=(piece,square)=>{
      const node=element('div',null,'motion-piece'),[x,y]=point(square);
      node.innerHTML=pieceMarkup(piece);node.style.left=(x*12.5)+'%';node.style.top=(y*12.5)+'%';
      layer.append(node);return node;
    };
    if(move.captured){
      const square=move.flags.includes('e')?move.to[0]+move.from[1]:move.to;
      const captured=sprite(before.get(square),square);
      animations.push(captured.animate([{opacity:1},{opacity:0}],{duration:140,delay:80,easing:'ease-out',fill:'both'}));
    }
    const journeys=[[move.from,move.to]];
    if(move.flags.includes('k'))journeys.push(['h'+move.from[1],'f'+move.from[1]]);
    if(move.flags.includes('q'))journeys.push(['a'+move.from[1],'d'+move.from[1]]);
    for(const [from,to]of journeys){
      const target=[...board.children].find(square=>square.dataset.square===to);
      target.classList.add('piece-in-flight');hiddenSquares.push(target);
      // Travel with the original pawn on promotion; reveal the promoted piece on arrival.
      const moving=sprite(before.get(from),to),[x0,y0]=point(from),[x1,y1]=point(to);
      animations.push(moving.animate([{transform:'translate('+((x0-x1)*100)+'%, '+((y0-y1)*100)+'%)'},{transform:'translate(0%, 0%)'}],
        {duration:220,easing:'cubic-bezier(0.16, 1, 0.3, 1)',fill:'both'}));
    }
    Promise.all(animations.map(animation=>animation.finished.catch(()=>{}))).then(state.finish);
    return done;
  }
  function focusSquare(){for(const button of $('board').children)if(button.dataset.square===focused){button.focus();break;}}
  function selectSquare(square){
    if(replay!=null||game.status().over)return;
    if(!running&&source()==='human')void start(false);
    if(!humanResolve||promotionMoves)return;
    const legal=game.options(),destinations=legal.filter(o=>o.from===selected&&o.to===square);
    focused=square;
    if(destinations.length){
      if(destinations.some(o=>o.promotion)){promotionMoves=destinations;$('promotion').hidden=false;$('board-note').textContent='CHOOSE A PROMOTION PIECE';return;}
      humanResolve(destinations[0]);return;
    }
    selected=legal.some(o=>o.from===square)?square:null;
    renderBoard();focusSquare();
  }
  function renderHistory(){
    const history=$('history');history.replaceChildren();
    if(!moves.length){history.append(element('span','—'));return;}
    const initial=new ChessRules.Chess(game.initialFen),start=Number(game.initialFen.split(' ')[5]);
    if(initial.turn()==='b'){history.append(element('span',start+'.'),element('span','…'));}
    moves.forEach((move,i)=>{
      if(move.color==='w')history.append(element('span',(start+Math.floor((i+(initial.turn()==='b'?1:0))/2))+'.'));
      const button=element('button',move.san);button.type='button';button.setAttribute('aria-label','Replay after '+move.san);
      button.setAttribute('aria-current',replay===i+1?'true':'false');button.addEventListener('click',()=>showReplay(i+1));history.append(button);
    });
    if(replay==null)history.scrollTop=history.scrollHeight;
  }
  function render(){
    $('turn-state').textContent=game.status().text;
    $('source').textContent=ChessGame.label(game.turn).toUpperCase()+' · '+sourceName[source()];
    $('option-count').textContent=game.options().length;
    renderBoard();renderHistory();
    const stats=SystemOne.summary(records);
    $('p50').textContent=ms(stats.p50);$('p95').textContent=ms(stats.p95);$('samples').textContent=stats.samples;
    $('agreement').textContent=stats.evaluated?stats.matches+' / '+stats.evaluated:'—';
    $('on-time').textContent=stats.deadlines?(stats.deadlines-stats.misses)+' / '+stats.deadlines:'—';
    const status=game.status(),whiteName=sourceName[$('white').value],blackName=sourceName[$('black').value];
    $('match-result').textContent=status.over?(game.rules.isCheckmate()?sourceName[$(game.turn==='b'?'white':'black').value]+' WINS · CHECKMATE':status.text.toUpperCase()):whiteName+' vs '+blackName;
    $('player-stats').replaceChildren();
    for(const [side,color]of [['white','w'],['black','b']]){
      const player=$(side).value,turns=records.filter(r=>r.color===color),stats=SystemOne.summary(turns),row=element('tr');
      const name=element('td',ChessGame.label(color)+' · '+sourceName[player]);
      if(isLive(player))name.append(element('small',warmups[player]?.responseModel||connection(player).model));
      const stages=turns.flatMap(r=>r.stages),input=stages.reduce((n,s)=>n+s.response.usage.input_tokens,0),output=stages.reduce((n,s)=>n+s.response.usage.output_tokens,0);
      row.append(name,element('td',turns.filter(r=>r.placed).length),element('td',ms(stats.p50)),element('td',ms(stats.p95)),element('td',stages.length?input+' / '+output:'—'));
      $('player-stats').append(row);
    }
    $('warmup').textContent=Object.entries(warmups).map(([player,warm])=>sourceName[player]+' '+ms(warm.ms)+' ms').join(' · ')||'—';
    controls();
  }
  function renderDecision(){
    const record=latest,stages=record?.stages||[],stage=stages[stageIndex],answer=stage?.response.answers.move;
    $('dt').textContent=ms(record?.ms);$('decision-ms').textContent=Number.isFinite(record?.decisionMs)?ms(record.decisionMs)+' ms':'—';
    $('calls').textContent=stages.length||'—';
    $('tokens').textContent=stages.length?stages.reduce((n,s)=>n+s.response.usage.input_tokens,0)+' / '+stages.reduce((n,s)=>n+s.response.usage.output_tokens,0):'—';
    $('decision-title').textContent=record?.rejectedLate?'DEADLINE MISSED':record?.san|| (source()==='human'?'YOUR MOVE':'READY TO DECIDE');
    $('selection-prob').textContent=answer?(answer.probabilities[answer.choice]*100).toFixed(1)+'%':'—';
    $('decision-note').textContent=record?.rejectedLate?'No move applied. Resume to retry this position.':record?.source==='reference'?'Local material heuristic · no inference.':record?.source==='human'?'Human move · untimed.':stages.length>1?'Two requests for one move. The displayed probabilities are conditional on this stage.':stages.length?'Model choice among all legal moves. Probabilities are preferences, not win probabilities.':'Select your piece, or press Run to start the configured players.';
    $('inspector-count').textContent=stages.length?'· '+stages.length+' REQUEST'+(stages.length===1?'':'S'):'';
    const tabs=$('stage-tabs');tabs.replaceChildren();
    stages.forEach((s,i)=>{const button=element('button',(i+1)+'. '+(s.kind==='piece'?'Piece':'Move')+' · '+ms(s.ms)+' ms');button.type='button';button.setAttribute('aria-pressed',i===stageIndex?'true':'false');button.addEventListener('click',()=>{stageIndex=i;preview=null;renderDecision();renderBoard();});tabs.append(button);});
    $('distribution-note').textContent=stages.length>1?'Stage '+(stageIndex+1)+' of '+stages.length+'. These probabilities apply only to '+(stage?.kind==='piece'?'piece selection.':'moves of the selected piece.'):'All probabilities belong to this move request.';
    $('request-wire').textContent=stage?JSON.stringify(stage.request,null,2):'No request.';
    $('response-wire').textContent=stage?JSON.stringify(stage.response,null,2):'No response.';
    $('probabilities').replaceChildren();$('candidate-table').replaceChildren();
    if(failure){
      $('decision-title').textContent=failure.kind==='response'?'RESPONSE REJECTED':'REQUEST FAILED';
      $('decision-note').textContent='No move applied. '+(failure.phase==='warm-up'?'Warm-up failed. ':'')+'Press Resume or Step to retry this position.';
      $('distribution-note').textContent='Failed '+failure.phase+' request. The raw response is retained below; rejected data is excluded from match timings.';
      $('inspector-count').textContent='· FAILED '+failure.phase.toUpperCase();
      $('request-wire').textContent=JSON.stringify(failure.receipt?.request??null,null,2);
      $('response-wire').textContent=JSON.stringify(failure.receipt?.response??null,null,2);
      return;
    }
    if(!stage)return;
    if(stage.choiceWarning){
      $('decision-note').textContent+=' '+stage.choiceWarning;
      $('distribution-note').textContent+=' '+stage.choiceWarning;
    }
    const total=Object.values(answer.probabilities).reduce((sum,p)=>sum+p,0);
    if(Math.abs(total-1)>1e-5)$('distribution-note').textContent+=' API probabilities total '+(total*100).toFixed(2)+'%; raw values are shown without normalization.';
    const ordered=[...stage.options].sort((a,b)=>answer.probabilities[b.code]-answer.probabilities[a.code]);
    ordered.forEach((option,i)=>{
      const probability=answer.probabilities[option.code],label=option.san||option.from;
      if(i<5){
        const row=element('div',null,'prob-row'+(option.code===stage.choice?' chosen':''));
        const button=element('button',label);button.type='button';button.setAttribute('aria-label','Highlight '+label);
        button.addEventListener('click',()=>{preview=option;renderBoard();});
        const bar=element('div',null,'prob-track'),fill=element('div',null,'prob-fill');fill.style.width=(probability*100)+'%';bar.append(fill);
        row.append(button,bar,element('span',(probability*100).toFixed(1)+'%'));$('probabilities').append(row);
      }
      const row=element('tr');if(option.code===stage.choice)row.className='chosen';
      const detail=option.moves?option.moves.map(o=>o.san).join(', '):'capture '+(option.captured?ChessGame.names[option.captured]:'none')+' · '+(option.mate?'checkmate':option.attacked?'moved piece attacked':'moved piece safe');
      [option.code,label,detail,(probability*100).toFixed(2)+'%'].forEach(text=>row.append(element('td',text)));$('candidate-table').append(row);
    });
  }
  function animateBudget(){
    if(deadlineStart==null)return;
    const remaining=Math.max(0,1-(performance.now()-deadlineStart)/Number($('budget-ms').value));
    $('deadline').value=remaining;
    if(running&&remaining>0)raf=requestAnimationFrame(animateBudget);
  }
  function waitHuman(signal){
    return new Promise((resolve,reject)=>{
      const abort=()=>{humanResolve=null;reject(new DOMException('Paused','AbortError'));};
      signal.addEventListener('abort',abort,{once:true});
      humanResolve=option=>{signal.removeEventListener('abort',abort);humanResolve=null;promotionMoves=null;selected=null;$('promotion').hidden=true;resolve(option);};
      setMode('YOUR TURN');renderBoard();
    });
  }
  function delay(duration,signal){
    if(signal.aborted)return Promise.reject(new DOMException('Paused','AbortError'));
    return new Promise((resolve,reject)=>{
      const abort=()=>{clearTimeout(timer);reject(new DOMException('Paused','AbortError'));};
      const timer=setTimeout(()=>{signal.removeEventListener('abort',abort);resolve();},duration);
      signal.addEventListener('abort',abort,{once:true});
    });
  }
  async function start(single){
    if(running){stop();return;}
    if(game.status().over||game.ply>=300)return;
    replay=null;preview=null;running=true;const run=++epoch;controller=new AbortController();const signal=controller.signal;
    failure=null;$('fault').textContent='';controls();renderBoard();
    try {
      while(running&&run===epoch&&!game.status().over&&game.ply<300){
        const player=source(),color=game.turn,live=isLive(player),beforeFen=game.fen,playerConnection=live?connection(player):null;
        if(live&&!warmups[player]){
          setMode('WARMING UP');$('decision-note').textContent='Warming '+sourceName[player]+' · excluded from match timings.';
          const step=ChessGame.plan(game,game.options());
          const result=await SystemOne.decide(playerConnection.endpoint,ChessGame.requestFor(game,step,playerConnection.model),step.options,{signal:AbortSignal.any([signal,AbortSignal.timeout(30000)]),apiKey:playerConnection.apiKey});
          if(run!==epoch)return;
          warmups[player]={ms:result.ms,requestedModel:playerConnection.model,responseModel:result.body.model||null};
          for(const side of ['white','black'])if($(side).value===player)game.rules.setHeader(side==='white'?'White':'Black',result.body.model||playerConnection.model);
          render();
        }
        const startTime=performance.now(),options=game.options(),expected=ChessGame.reference(options);
        let option,result=null;const completedStages=[];
        $('stage-enumerate').classList.add('active');$('option-count').textContent=options.length;
        if(player==='human')option=await waitHuman(signal);
        else if(player==='reference'){setMode('LOCAL REFERENCE');option=expected;}
        else {
          setMode('DECIDING');deadlineStart=startTime;animateBudget();$('stage-decide').classList.add('active');
          const task=combined=>ChessGame.choose(game,options,playerConnection,{signal:AbortSignal.any([combined,AbortSignal.timeout(30000)]),onResult:stage=>{if(run===epoch)completedStages.push(stage);},onStage:s=>{if(run===epoch)$('stage-decide').textContent=(s.kind==='piece'?'PIECE':'MOVE')+' CHOICE · '+s.number;}});
          result=$('strict').checked?await SystemOne.withDeadline(task,Number($('budget-ms').value)-(performance.now()-startTime),signal):await task(signal);
          if(run!==epoch)return;
          option=result.option;
        }
        if(run!==epoch)return;
        const decisionMs=performance.now()-startTime,budget=Number($('budget-ms').value);
        deadlineStart=null;if(raf!=null)cancelAnimationFrame(raf);raf=null;
        const record={ply:game.ply+1,source:live?'live':player,player,color,connection:live?publicConnection(player):null,beforeFen,budgetMs:live?budget:null,
          decisionMs:live?decisionMs:null,ms:result?.ms??null,stages:result?.stages||completedStages.slice(),
          expired:!!result?.expired,deadlineMet:live?(!result.expired&&decisionMs<=budget):null,
          policyMatch:live&&option?option.referenceScore===expected.referenceScore:null,
          reference:expected.id,legalMoves:options.length,choice:option?.id??null,san:option?.san??null,placed:false};
        record.rejectedLate=record.deadlineMet===false&&$('strict').checked;
        records.push(record);latest=record;stageIndex=record.stages.length-1;
        if(record.rejectedLate){render();renderDecision();stop('DEADLINE MISSED');$('deadline').value=0;break;}
        const move=game.commit(option);Object.assign(record,{placed:true,san:move.san,afterFen:game.fen});
        moves.push({from:move.from,to:move.to,san:move.san,color:move.color,afterFen:game.fen});
        $('stage-place').classList.add('active');$('deadline').value=live?Math.max(0,1-decisionMs/budget):1;
        const presentationStart=performance.now();
        render();renderDecision();await animateMove(move,beforeFen);
        if(run!==epoch)return;
        if(single){$('inspector').open=true;stop(game.status().over?'GAME COMPLETE':'STEP COMPLETE');break;}
        if(game.status().over||game.ply>=300){stop(game.status().over?'GAME COMPLETE':'300 PLY LIMIT');break;}
        await delay(Math.max(0,Number($('pace').value)-(performance.now()-presentationStart)),signal);
      }
    }catch(error){
      if(run!==epoch||error.name==='AbortError')return;
      const player=source();
      failure={player,phase:$('mode').textContent==='WARMING UP'?'warm-up':'move',beforeFen:game.fen,kind:error.kind||'request',status:error.status??null,message:error.message,receipt:error.receipt??null};
      latest=null;stageIndex=0;renderDecision();$('inspector').open=true;
      const advice=error.status===401||error.status===403
        ? 'Check the '+(player==='jev'?'TypeSafe API key under Jev connection.':'bearer token under Qwen connection.')
        : error.kind==='network' ? (player==='jev'?'Check the Jev relay address and that the relay is running.':'Check the endpoint, resident model and CORS setting.')
        : error.status===429 ? 'The service is rate limited. Wait before retrying.'
        : 'Inspect the response below, then press Resume or Step to retry.';
      $('fault').textContent=sourceName[player]+': '+error.message+' '+advice;stop('REQUEST FAILED');
    }finally{
      if(run===epoch){running=false;controller=null;deadlineStart=null;controls();}
      if(!running){$('stage-enumerate').classList.remove('active');$('stage-decide').classList.remove('active');$('stage-place').classList.remove('active');}
    }
  }
  function showReplay(ply){stop('REPLAY');replay=Math.max(0,Math.min(moves.length,ply));preview=null;renderBoard();renderHistory();}
  function download(name,type,content){const url=URL.createObjectURL(new Blob([content],{type})),a=element('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),0);}
  $('run').addEventListener('click',()=>start(false));$('step').addEventListener('click',()=>start(true));$('reset').addEventListener('click',reset);
  $('flip').addEventListener('click',()=>{orientation=orientation==='w'?'b':'w';renderBoard();});
  $('previous').addEventListener('click',()=>showReplay((replay??moves.length)-1));$('next-position').addEventListener('click',()=>showReplay((replay??moves.length)+1));
  $('return-live').addEventListener('click',()=>{replay=null;preview=null;setMode('PAUSED');renderBoard();renderHistory();});
  for(const button of $('promotion').querySelectorAll('button'))button.addEventListener('click',()=>{const option=promotionMoves?.find(o=>o.promotion===button.dataset.promotion);if(option&&humanResolve)humanResolve(option);});
  for(const id of settings.filter(id=>!['load-fen','jev-v-qwen','swap-sides'].includes(id)))$(id).addEventListener('change',reset);
  $('jev-v-qwen').addEventListener('click',()=>{$('white').value='jev';$('black').value='qwen';$('strict').checked=false;$('jev-connection').open=true;reset();});
  $('swap-sides').addEventListener('click',()=>{const white=$('white').value;$('white').value=$('black').value;$('black').value=white;reset();});
  $('load-fen').addEventListener('click',reset);
  $('scenario').addEventListener('change',()=>{$('fen-input').hidden=$('scenario').value!=='custom';});
  $('pace').addEventListener('input',()=>{$('pace-value').textContent=$('pace').value+' ms';});
  $('budget-ms').addEventListener('input',()=>{$('budget-value').textContent=$('budget-ms').value+' ms';});
  $('save-pgn').addEventListener('click',()=>download('ninfer-chess.pgn','application/x-chess-pgn',game.rules.pgn()));
  $('export').addEventListener('click',()=>download('ninfer-chess.json','application/json',JSON.stringify({game:'chess',version:2,initialFen:game.initialFen,
    config:{white:$('white').value,black:$('black').value,connections:{qwen:publicConnection('qwen'),jev:publicConnection('jev')},budgetMs:Number($('budget-ms').value),strict:$('strict').checked},
    warmups,failure,summary:SystemOne.summary(records),players:Object.fromEntries(['w','b'].map(color=>[color,SystemOne.summary(records.filter(r=>r.color===color))])),decisions:records,pgn:game.rules.pgn(),finalFen:game.fen},null,2)));
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&running)stop('PAUSED · TAB HIDDEN');});
  document.addEventListener('keydown',event=>{
    if(event.repeat||event.ctrlKey||event.metaKey||event.altKey||event.target.closest('input,select,textarea,[contenteditable]:not([contenteditable="false"])'))return;
    if(event.key?.toLowerCase()==='p'){event.preventDefault();if(running)stop();else void start(false);}
  });
  const params=new URLSearchParams(location.search);
  for(const id of ['qwen-endpoint','qwen-model','jev-endpoint','jev-model'])if(params.has(id))$(id).value=params.get(id);
  for(const id of ['white','black'])if(['human','qwen','jev','reference'].includes(params.get(id)))$(id).value=params.get(id);
  if(params.get('match')==='jev-qwen'){$('white').value='jev';$('black').value='qwen';$('jev-connection').open=true;}
  if(ChessGame.positions[params.get('scenario')])$('scenario').value=params.get('scenario');
  if(params.has('fen')){$('scenario').value='custom';$('start-fen').value=params.get('fen');}
  else $('start-fen').value=ChessGame.positions.opening;
  for(const [id,min,max]of [['pace',0,2000],['budget-ms',100,3000]])if(params.has(id)&&Number.isFinite(Number(params.get(id))))$(id).value=Math.max(min,Math.min(max,Number(params.get(id))));
  $('strict').checked=params.get('strict')==='1';$('pace-value').textContent=$('pace').value+' ms';
  // An invalid URL FEN must leave a usable board, with the error visible for correction.
  game=new ChessGame.Game();reset();if(!$('board').children.length){render();renderDecision();controls();}
})();
