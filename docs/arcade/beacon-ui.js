(() => {
  const $=id=>document.getElementById(id),arena=$('arena'),ctx=arena.getContext('2d');
  const encodeCanvas=document.createElement('canvas');
  const camera=new BeaconSession.Camera(),client=new BeaconSession.Client();
  let game=BeaconGame.create(1),running=false,warming=false,verified=false,previous=performance.now(),nextDecision=0,lastPaint=0,epoch=0,drag=null;
  const events=[];
  const config=()=>({endpoint:$('endpoint').value.trim(),model:$('model').value.trim(),key:$('key').value,mode:$('camera-mode').value,deadline:Number($('deadline').value)});
  const status=value=>{$('mode').textContent=value;$('power-label').textContent=value;$('led').classList.toggle('on',running);};
  const stamp=ms=>`${String(Math.floor(ms/60000)).padStart(2,'0')}:${String(Math.floor(ms/1000)%60).padStart(2,'0')}`;
  function encode(frame){encodeCanvas.width=frame.width;encodeCanvas.height=frame.height;encodeCanvas.getContext('2d').putImageData(new ImageData(frame.pixels,frame.width,frame.height),0,0);return {...frame,dataURL:encodeCanvas.toDataURL('image/png')};}
  function pause(label='PAUSED'){
    running=false;warming=false;epoch++;client.cancel();BeaconGame.brake(game);$('run').textContent='RESUME';status(label);$('setup').disabled=game.time>0;
  }
  function show(row){
    if(!row)return;
    $('export').disabled=false;
    $('action').textContent=BeaconGame.actions.find(a=>a.code===row.action)?.label||'No action';
    $('action-detail').textContent=row.warmup?'Warm-up · no movement':row.status;
    $('latency').textContent=row.ms.toFixed(1)+' ms';$('frame-age').textContent=Number.isFinite(row.frameAge)?Math.round(row.frameAge)+' ms':'—';
    $('vision-tokens').textContent=row.visionTokens??'Unconfirmed';
    const totals=BeaconSession.summary(client.records);
    $('percentiles').textContent=totals.n?`${totals.p50.toFixed(1)} / ${totals.p95.toFixed(1)} ms`:'—';$('samples').textContent=`${totals.n} / ${totals.dropped}`;
    $('receipt-note').textContent=`${row.status} · ${row.mode} · ${row.width} × ${row.height}`;
    $('request-wire').textContent=JSON.stringify({...row.request,images:row.request.images.map(s=>s.slice(0,32)+'… [exact image in export]')},null,2);
    $('response-wire').textContent=JSON.stringify(row.response,null,2);
    $('probabilities').replaceChildren(...BeaconGame.actions.map(a=>{const div=document.createElement('div'),p=row.response?.answers?.move?.probabilities?.[a.code],label=document.createElement('span'),bar=document.createElement('progress');label.textContent=`${a.label} · ${p==null?'—':(p*100).toFixed(1)+'%'}`;bar.max=1;bar.value=p||0;bar.setAttribute('aria-label',a.label+' probability');div.append(label,bar);return div;}));
  }
  async function decide(now){
    if(client.pending||!running||$('pilot').value==='human')return;
    const width=Number($('resolution').value),frame=camera.sample($('camera-mode').value,now,width);if(!frame){status('BUFFERING CAMERA');return;}
    const observation=encode(frame),token=epoch,wasWarmup=!verified;
    $('observation').src=observation.dataURL;$('observation').hidden=false;$('empty-camera').hidden=true;
    $('frame-note').textContent=`${width} × ${width*3/4} · exact sent frame · ${$('camera-mode').value}`;
    warming=wasWarmup;status(wasWarmup?'VERIFYING VISION':'DECIDING');
    try {
      const row=await client.decide(observation,config(),{warmup:wasWarmup});
      if(token!==epoch||!row)return;
      show(row);verified=true;warming=false;
      if(!row.warmup&&!row.dropped)BeaconGame.command(game,row.action);else BeaconGame.brake(game);
      nextDecision=performance.now()+Math.max(30,220-(row.ms||0));status('RUNNING');
    }catch(error){if(token!==epoch)return;show(error.record);$('fault').textContent=error.message+' Correct the connection, then Resume to retry.';verified=false;pause('REQUEST FAILED');}
  }
  function paint(now){
    const elapsed=now-previous;previous=now;
    // Browser suspension is not simulated in a large jump or allowed to apply an old response.
    if(elapsed>1000&&running)pause('PAUSED AFTER SUSPENSION');
    if(running&&!warming){BeaconGame.tick(game,elapsed);if(game.time>=Number($('duration').value))pause('RUN COMPLETE');}
    if(now-lastPaint>=33){
      lastPaint=now;const frame=BeaconGame.render(game,Number($('resolution').value));camera.push(frame,now);
      const world=BeaconGame.render(game,640);ctx.putImageData(new ImageData(world.pixels,world.width,world.height),0,0);
      $('clock').textContent=stamp(game.time)+' / '+stamp(Number($('duration').value));$('score').textContent=`${game.collected} BEACONS · ${game.hits} CONTACTS`;
      $('run-progress').max=Number($('duration').value);$('run-progress').value=game.time;
    }
    if(running&&now>=nextDecision)void decide(now);
    requestAnimationFrame(paint);
  }
  $('run').onclick=()=>{
    if(running){pause();return;}
    if(game.time>=Number($('duration').value)){$('fault').textContent='Run complete. Reset to start another run.';return;}
    if(client.pending){$('fault').textContent='Waiting for the cancelled request to settle. Press Resume again.';return;}
    const seed=Number($('seed').value);
    if(!Number.isInteger(seed)||seed<0||seed>4294967295||!$('seed').value){$('fault').textContent='Choose a whole-number seed between 0 and 4294967295.';return;}
    if(game.time===0&&!verified){game=BeaconGame.create(seed);camera.reset();}
    $('fault').textContent='';running=true;warming=$('pilot').value==='qwen'&&!verified;previous=performance.now();nextDecision=0;$('run').textContent='PAUSE';$('setup').disabled=true;status('RUNNING');
  };
  $('reset').onclick=()=>{
    pause('READY');game=BeaconGame.create(Number($('seed').value));camera.reset();client.records.length=0;events.length=0;verified=false;$('setup').disabled=false;$('run').textContent='RUN';$('fault').textContent='';$('export').disabled=true;
    $('observation').hidden=true;$('empty-camera').hidden=false;$('frame-note').textContent='No observation yet.';$('action').textContent='—';$('action-detail').textContent='Waiting for a decision';
    for(const id of ['latency','frame-age','vision-tokens','percentiles'])$(id).textContent='—';$('samples').textContent='0 / 0';$('receipt-note').textContent='No decision recorded.';$('probabilities').replaceChildren();$('request-wire').textContent='No request yet.';$('response-wire').textContent='No response yet.';
  };
  for(const id of ['camera-mode','resolution','deadline','endpoint','model','key','pilot'])$(id).addEventListener('change',()=>{
    pause('SETTINGS CHANGED');verified=false;camera.reset();$('camera-label').textContent=$('camera-mode').value.toUpperCase();$('manual').hidden=$('pilot').value!=='human';
    events.push({at:game.time,type:'settings',camera:$('camera-mode').value,width:Number($('resolution').value),deadline:Number($('deadline').value)});
  });
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&running)pause('TAB HIDDEN');});
  function manual(code){if(running&&$('pilot').value==='human'){BeaconGame.command(game,code);events.push({at:game.time,type:'human-action',code});$('export').disabled=false;}}
  for(const a of BeaconGame.actions){const b=document.createElement('button');b.textContent=a.label;b.onclick=()=>manual(a.code);$('manual-actions').append(b);}
  arena.addEventListener('keydown',e=>{const code={ArrowLeft:'A',ArrowRight:'B',ArrowUp:'C',ArrowDown:'D',' ':'I'}[e.key];if(code){e.preventDefault();manual(code);}if(e.key.toLowerCase()==='p'&&running)pause();});
  const position=e=>{const box=arena.getBoundingClientRect();return {x:(e.clientX-box.left)/box.width*640,y:(e.clientY-box.top)/box.height*480};};
  function edit(id,x,y){const ok=BeaconGame.moveObject(game,id,x,y);$('edit-status').textContent=ok?'Scene changed. Qwen receives no edit notification.':'That position overlaps the drone, a wall or the boundary.';if(ok)events.push({at:game.time,type:'edit',id,x,y});}
  arena.addEventListener('pointerdown',e=>{const p=position(e),id=BeaconGame.hitObject(game,p.x,p.y);if(!id)return;const obj=id==='beacon'?game.beacon:game.walls[Number(id.slice(5))];drag={id,dx:p.x-obj.x,dy:p.y-obj.y};$('object').value=id;arena.setPointerCapture(e.pointerId);});
  arena.addEventListener('pointermove',e=>{if(!drag)return;const p=position(e);edit(drag.id,p.x-drag.dx,p.y-drag.dy);});
  for(const name of ['pointerup','pointercancel','lostpointercapture'])arena.addEventListener(name,()=>{drag=null;});
  document.querySelectorAll('[data-nudge]').forEach(b=>b.onclick=()=>{const id=$('object').value,obj=id==='beacon'?game.beacon:game.walls[Number(id.slice(5))],[dx,dy]=b.dataset.nudge.split(',').map(Number);edit(id,obj.x+dx,obj.y+dy);});
  $('export').onclick=()=>{
    const blob=new Blob([JSON.stringify({version:1,game:'Beacon Runner',pilot:$('pilot').value,seed:Number($('seed').value),score:{beacons:game.collected,contacts:game.hits,elapsed:game.time},events,records:client.records},null,2)],{type:'application/json'});
    const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download='beacon-run.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
  requestAnimationFrame(paint);
})();
