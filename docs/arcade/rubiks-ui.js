(() => {
  'use strict';
  const $=id=>document.getElementById(id);
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  const sourceName={reference:'LOCAL SEARCH',qwen:'QWEN · SYSTEM ONE',reasoning:'QWEN · REASONING',human:'YOU'};
  const clockName={reference:'SEARCH CLOCK',qwen:'REQUEST CLOCK',reasoning:'REQUEST CLOCK',human:'SOLVE CLOCK'};
  // Sticker colors by home face U R F D L B, shared by the WebGL cube and the model-view net.
  const PALETTE=['#f2efe6','#c8142f','#00a255','#ffcf00','#ff6a00','#0b56c9'];
  PALETTE.forEach((hex,i)=>document.documentElement.style.setProperty('--c'+i,hex));
  const FACE_OF_AXIS={'0,1,0':'U','0,-1,0':'D','1,0,0':'R','-1,0,0':'L','0,0,1':'F','0,0,-1':'B'};
  const settings=['player','distance','seed','endpoint','model','key','max-tokens','repeat-policy'];
  const ms=value=>Number.isFinite(value)?value.toFixed(1):'—';
  const pct=value=>Number.isFinite(value)?(value*100).toFixed(1)+'%':'—';
  const easeInOut=t=>t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2;
  const easeOut=t=>1-Math.pow(1-t,3);
  const gradeOf=(before,after)=>before==null||after==null?'unknown':after<before?'closer':after===before?'level':'farther';
  const gradeText={closer:'−1',level:'±0',farther:'+1',unknown:''};
  const stage=$('stage'),canvas=$('cube');
  let session,lastInteraction=performance.now();

  // ---------- The instrument: a speedcube on a phosphor turntable ----------
  const ARC_VERTEX='varying vec2 vPos;void main(){vPos=position.xy;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}';
  // A protractor ring around the turning layer: faint graduations, a lit arc from the start to the
  // current angle, brightest at its head, and a soft phosphor bloom.
  const ARC_FRAGMENT=`
    uniform float uStart,uSweep,uGlow,uRadius,uRing;uniform vec3 uColor;varying vec2 vPos;
    const float TAU=6.28318530718;
    void main(){
      float r=length(vPos),d=abs(r-uRadius),a=atan(vPos.y,vPos.x);
      float line=exp(-d*d/0.00022),halo=exp(-d*d/0.009)*0.3;
      float len=abs(uSweep),dir=uSweep<0.0?-1.0:1.0,rel=mod(dir*(a-uStart),TAU);
      float lit=(len>0.001&&rel<=len)?0.2+0.8*smoothstep(len-1.3,len,rel):0.0;
      float stepA=TAU/24.0,off=abs(mod(a-uStart+0.5*stepA,stepA)-0.5*stepA)*r;
      float quarter=abs(mod(a-uStart+TAU/8.0,TAU/4.0)-TAU/8.0)*r;
      float tickLen=quarter<0.03?0.17:0.085;
      float tick=(1.0-smoothstep(0.005,0.014,off))*(1.0-smoothstep(tickLen-0.02,tickLen,d));
      float v=uGlow*(uRing*(line*0.3+tick*0.45)+lit*(line*1.7+halo));
      gl_FragColor=vec4(uColor*v,min(v,1.0));
    }`;

  function roundedBox(size,radius,segments){
    const segs=segments*2+1,geo=new THREE.BoxGeometry(1,1,1,segs,segs,segs).toNonIndexed(),half=size/2-radius,h=.5/segs;
    const position=geo.attributes.position,normal=geo.attributes.normal,v=new THREE.Vector3(),n=new THREE.Vector3();
    for(let i=0;i<position.count;i++){
      v.fromBufferAttribute(position,i);
      n.set(v.x-Math.sign(v.x)*h,v.y-Math.sign(v.y)*h,v.z-Math.sign(v.z)*h).normalize();
      position.setXYZ(i,half*Math.sign(v.x)+n.x*radius,half*Math.sign(v.y)+n.y*radius,half*Math.sign(v.z)+n.z*radius);
      normal.setXYZ(i,n.x,n.y,n.z);
    }
    geo.deleteAttribute('uv');
    return geo;
  }
  function tileGeometry(){
    const s=.4,r=.09,shape=new THREE.Shape();
    shape.moveTo(-s+r,-s);shape.lineTo(s-r,-s);shape.quadraticCurveTo(s,-s,s,-s+r);shape.lineTo(s,s-r);shape.quadraticCurveTo(s,s,s-r,s);
    shape.lineTo(-s+r,s);shape.quadraticCurveTo(-s,s,-s,s-r);shape.lineTo(-s,-s+r);shape.quadraticCurveTo(-s,-s,-s+r,-s);
    const geo=new THREE.ExtrudeGeometry(shape,{depth:.004,bevelEnabled:true,bevelThickness:.009,bevelSize:.012,bevelSegments:3,curveSegments:7});
    geo.translate(0,0,.009);
    return geo;
  }
  // A small procedural studio for reflections: overhead softbox, warm and cool strips, amber floor bounce.
  function studio(renderer){
    const env=new THREE.Scene();
    env.add(new THREE.Mesh(new THREE.BoxGeometry(14,14,14),new THREE.MeshBasicMaterial({color:0x140c06,side:THREE.BackSide})));
    const light=(w,h,hex,k,x,y,z)=>{const mesh=new THREE.Mesh(new THREE.PlaneGeometry(w,h),new THREE.MeshBasicMaterial({color:new THREE.Color(hex).multiplyScalar(k),side:THREE.DoubleSide}));mesh.position.set(x,y,z);mesh.lookAt(0,0,0);env.add(mesh);};
    light(8,4,0xfff4e4,3,0,6.9,1.5);
    light(3,7,0xffd7a1,2.2,-6.9,1.5,2.5);
    light(3,7,0xcfe0ff,1.5,6.9,1,-2.5);
    light(12,1.4,0xffa000,.45,0,-3,6.9);
    const pmrem=new THREE.PMREMGenerator(renderer),texture=pmrem.fromScene(env,.04).texture;
    pmrem.dispose();
    return texture;
  }
  function glyphTexture(text){
    const c=document.createElement('canvas');c.width=c.height=128;
    const texture=new THREE.CanvasTexture(c);texture.colorSpace=THREE.SRGBColorSpace;texture.anisotropy=4;
    const draw=()=>{const x=c.getContext('2d');x.clearRect(0,0,128,128);x.fillStyle='#ffb000';x.font='700 92px "Kode Mono", ui-monospace, monospace';x.textAlign='center';x.textBaseline='middle';x.fillText(text,64,70);texture.needsUpdate=true;};
    draw();document.fonts?.load('700 92px "Kode Mono"').then(draw,()=>{});
    return texture;
  }

  function instrument(){
    const renderer=new THREE.WebGLRenderer({canvas,antialias:true,alpha:true,powerPreference:'high-performance'});
    renderer.setClearColor(0x000000,0);
    renderer.outputColorSpace=THREE.SRGBColorSpace;
    renderer.toneMapping=THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure=1.05;
    renderer.shadowMap.enabled=true;
    renderer.shadowMap.type=THREE.PCFSoftShadowMap;
    const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(24,1,.1,80);
    scene.environment=studio(renderer);
    scene.add(new THREE.HemisphereLight(0xffe9c9,0x241406,.45));
    const key=new THREE.DirectionalLight(0xfff0dc,2.3);
    key.position.set(3.2,9,4.8);key.castShadow=true;key.shadow.mapSize.set(1024,1024);
    Object.assign(key.shadow.camera,{left:-3.6,right:3.6,top:3.6,bottom:-3.6,near:3,far:20});
    key.shadow.bias=-.0004;key.shadow.normalBias=.02;
    scene.add(key);
    const rim=new THREE.DirectionalLight(0xffb040,1.35);rim.position.set(-6,2.5,-5.5);scene.add(rim);

    const GAP=1,SIZE=.94,FLOOR=-1.5*GAP-.012,Z=new THREE.Vector3(0,0,1);
    const root=new THREE.Group();scene.add(root);
    const bodyMat=new THREE.MeshStandardMaterial({color:0x0e0c0a,roughness:.36,metalness:0,envMapIntensity:.7});
    const tileMats=PALETTE.map(hex=>new THREE.MeshPhysicalMaterial({color:hex,roughness:.36,metalness:0,clearcoat:.35,clearcoatRoughness:.24,envMapIntensity:.7}));
    const bodyGeo=roundedBox(SIZE,.085,3),tileGeo=tileGeometry();
    const cubies=[],tiles=new Array(54),pickables=[];
    for(const facelet of Cube.FACELETS){
      let cubie=cubies.find(item=>item.userData.grid.join()===facelet.pos.join());
      if(!cubie){
        cubie=new THREE.Group();cubie.userData.grid=facelet.pos.slice();
        const body=new THREE.Mesh(bodyGeo,bodyMat);body.castShadow=true;cubie.add(body);pickables.push(body);
        root.add(cubie);cubies.push(cubie);
      }
      const tile=new THREE.Mesh(tileGeo,tileMats[facelet.face]),normal=new THREE.Vector3(...facelet.n);
      tile.quaternion.setFromUnitVectors(Z,normal);tile.position.copy(normal).multiplyScalar(SIZE/2);
      cubie.add(tile);tiles[facelet.index]=tile;pickables.push(tile);
    }

    // Turntable: shadow catcher, contact shade, phosphor rings, bearing ticks and face letters.
    const catcher=new THREE.Mesh(new THREE.CircleGeometry(4.6,96),new THREE.ShadowMaterial({color:0x000000,opacity:.52}));
    catcher.rotation.x=-Math.PI/2;catcher.position.y=FLOOR;catcher.receiveShadow=true;scene.add(catcher);
    const shade=document.createElement('canvas');shade.width=shade.height=128;
    {const x=shade.getContext('2d'),g=x.createRadialGradient(64,64,0,64,64,64);g.addColorStop(0,'rgba(0,0,0,.62)');g.addColorStop(.5,'rgba(0,0,0,.3)');g.addColorStop(1,'rgba(0,0,0,0)');x.fillStyle=g;x.fillRect(0,0,128,128);}
    const contact=new THREE.Mesh(new THREE.PlaneGeometry(4.3,4.3),new THREE.MeshBasicMaterial({map:new THREE.CanvasTexture(shade),transparent:true,depthWrite:false,toneMapped:false}));
    contact.rotation.x=-Math.PI/2;contact.position.y=FLOOR+.002;scene.add(contact);
    const table=new THREE.Group();table.position.y=FLOOR+.004;scene.add(table);
    const phosphor=opacity=>new THREE.MeshBasicMaterial({color:0xffb000,transparent:true,opacity,depthWrite:false,toneMapped:false,blending:THREE.AdditiveBlending,side:THREE.DoubleSide});
    const ringMat=phosphor(.62),faintMat=phosphor(.26),tickMat=phosphor(.5);
    const flat=mesh=>{mesh.rotation.x=-Math.PI/2;table.add(mesh);return mesh;};
    flat(new THREE.Mesh(new THREE.RingGeometry(3.02,3.045,256),ringMat));
    flat(new THREE.Mesh(new THREE.RingGeometry(2.5,2.512,256),faintMat));
    {
      const vertices=[];
      for(let k=0;k<72;k++){
        const a=k*Math.PI/36,cardinal=k%18===0,major=k%3===0,inner=3.09,outer=inner+(cardinal?.34:major?.17:.085),w=(cardinal?.034:.018)/2;
        const s=Math.sin(a),c=Math.cos(a),px=c*w,pz=-s*w;
        const p=[[s*inner-px,c*inner-pz],[s*inner+px,c*inner+pz],[s*outer+px,c*outer+pz],[s*outer-px,c*outer-pz]];
        for(const i of [0,1,2,0,2,3])vertices.push(p[i][0],0,p[i][1]);
      }
      const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));
      const ticks=new THREE.Mesh(geo,tickMat);table.add(ticks);
    }
    [['F',0],['R',Math.PI/2],['B',Math.PI],['L',-Math.PI/2]].forEach(([letter,a])=>{
      const label=new THREE.Mesh(new THREE.PlaneGeometry(.46,.46),new THREE.MeshBasicMaterial({map:glyphTexture(letter),transparent:true,depthWrite:false,toneMapped:false,opacity:.85}));
      label.position.set(Math.sin(a)*3.78,.001,Math.cos(a)*3.78);label.rotation.set(-Math.PI/2,a,0,'YXZ');table.add(label);
    });

    // Protractor arcs: a small pool for turns, one for the hover preview, one sweeping the floor.
    function makeArc(radius){
      const material=new THREE.ShaderMaterial({vertexShader:ARC_VERTEX,fragmentShader:ARC_FRAGMENT,transparent:true,depthWrite:false,
        blending:THREE.CustomBlending,blendSrc:THREE.OneFactor,blendDst:THREE.OneFactor,side:THREE.DoubleSide,
        uniforms:{uStart:{value:0},uSweep:{value:0},uGlow:{value:0},uRadius:{value:radius},uRing:{value:1},uColor:{value:new THREE.Vector3(1,.69,0)}}});
      const mesh=new THREE.Mesh(new THREE.RingGeometry(radius-.24,radius+.24,256,1),material);
      mesh.visible=false;mesh.renderOrder=3;scene.add(mesh);
      return {mesh,u:material.uniforms,peak:0,fadeFrom:null,fade:520};
    }
    const arcs=[makeArc(2.3),makeArc(2.3),makeArc(2.3),makeArc(2.3)],preview=makeArc(2.3),sweep=makeArc(3.032);
    let arcCursor=0;
    // Orients an arc around the layer whose outward normal is `normal`, centered on the camera side.
    function place(arc,normal,span){
      const n=new THREE.Vector3(...normal);
      arc.mesh.position.copy(n).multiplyScalar(GAP);arc.mesh.quaternion.setFromUnitVectors(Z,n);arc.mesh.updateMatrixWorld();
      const eye=arc.mesh.worldToLocal(camera.position.clone());
      arc.u.uStart.value=Math.atan2(eye.y,eye.x)-span/2;
    }
    function arcFor(normal,span,glow){
      const arc=arcs[arcCursor++%arcs.length];
      place(arc,normal,span);
      Object.assign(arc,{peak:glow,fadeFrom:null});arc.u.uGlow.value=glow;arc.u.uSweep.value=0;arc.u.uRing.value=1;arc.mesh.visible=true;
      return arc;
    }
    const fadeArc=(arc,now,ms=520)=>{arc.fadeFrom=now;arc.fade=ms;};
    function tickArcs(now){
      let live=false;
      for(const arc of [...arcs,sweep]){
        if(!arc.mesh.visible||arc.fadeFrom==null){live||=arc.mesh.visible&&arc===sweep;continue;}
        const k=Math.exp(-(now-arc.fadeFrom)/(arc.fade/3));
        arc.u.uGlow.value=arc.peak*k;
        if(k<.02){arc.mesh.visible=false;arc.fadeFrom=null;}else live=true;
      }
      return live;
    }

    // Visual state and the turn queue. The model state is authoritative; the cube catches up.
    const pivot=new THREE.Group();root.add(pivot);
    let visual=Cube.SOLVED.slice(),active=null,instant=false;
    const queue=[];
    function layout(colors){
      while(pivot.children.length)root.attach(pivot.children[0]);
      pivot.quaternion.identity();
      for(const cubie of cubies){cubie.position.set(...cubie.userData.grid).multiplyScalar(GAP);cubie.quaternion.identity();}
      for(let i=0;i<54;i++)tiles[i].material=tileMats[colors[i]];
      invalidate();
    }
    function grab(k,layer){
      pivot.quaternion.identity();
      for(const cubie of cubies)if(cubie.userData.grid[k]===layer)pivot.attach(cubie);
    }
    function duration(turn,source){
      if(instant||reduced.matches)return 0;
      const pace=Number($('pace').value)||1,base=source==='scramble'?95:turn.quarters===2?260:180;
      return base/pace/(1+queue.length*.6);
    }
    function present(turn,info={}){return new Promise(resolve=>{queue.push({turn,info,resolve});pump();});}
    function pump(){
      if(active||!queue.length||pointer.mode==='turn')return;
      const job=queue.shift(),span=job.turn.angle,time=duration(job.turn,job.info.source),scrambling=job.info.source==='scramble';
      showTurn(job.turn,job.info);
      const arc=arcFor(job.turn.axis,span,scrambling?.5:1);
      const done=()=>{visual=Cube.apply(visual,job.turn);layout(visual);arc.u.uSweep.value=span;fadeArc(arc,performance.now(),scrambling?260:560);job.resolve();pump();};
      if(!time){arc.u.uSweep.value=span;done();return;}
      const k=job.turn.axis.findIndex(v=>v!==0);
      grab(k,job.turn.axis[k]);
      active={axis:new THREE.Vector3(...job.turn.axis),from:0,to:span,start:performance.now(),ms:time,ease:job.info.source==='human'?easeOut:easeInOut,
        onAngle:angle=>{arc.u.uSweep.value=angle;},done};
      invalidate();
    }
    function advance(now){
      if(!active)return;
      const t=Math.min(1,(now-active.start)/active.ms),angle=active.from+(active.to-active.from)*active.ease(t);
      pivot.quaternion.setFromAxisAngle(active.axis,angle);active.onAngle(angle);
      if(t>=1){const job=active;active=null;job.done();}
    }
    function settle(){
      cancelDrag();
      instant=true;
      while(active){const job=active;active=null;job.done();}
      pump();
      instant=false;
    }
    function scramble(game){
      settle();
      visual=Cube.SOLVED.slice();layout(visual);
      return game.scramble.map(id=>present(Cube.move(id),{source:'scramble'})).at(-1)||Promise.resolve();
    }

    // Camera: orbit with inertia; a slow drift only while nothing is happening.
    const orbit={yaw:.62,pitch:.47,vy:0,vp:0,distance:16.2};
    const clampPitch=p=>Math.max(-.2,Math.min(1.25,p));
    function placeCamera(){
      const {yaw,pitch,distance}=orbit;
      camera.position.set(distance*Math.cos(pitch)*Math.sin(yaw),distance*Math.sin(pitch)-.4,distance*Math.cos(pitch)*Math.cos(yaw));
      camera.lookAt(0,-.42,0);
    }
    // The drift starts after four idle seconds and stops after forty-five, so an untouched page goes quiet.
    const drifting=()=>{const idle=performance.now()-lastInteraction;return !reduced.matches&&pointer.id==null&&!session?.running&&idle>4000&&idle<45000;};

    // Solved: one full turn of the cube while a sweep runs round the turntable.
    let spin=null;
    function celebrate(){
      if(reduced.matches)return;
      spin={start:performance.now(),ms:1500};
      sweep.mesh.position.set(0,FLOOR+.006,0);sweep.mesh.quaternion.setFromAxisAngle(new THREE.Vector3(1,0,0),-Math.PI/2);
      sweep.u.uStart.value=-Math.PI/2;sweep.u.uRing.value=0;sweep.peak=1;sweep.fadeFrom=null;sweep.u.uGlow.value=1;sweep.u.uSweep.value=0;sweep.mesh.visible=true;
      invalidate();
    }
    function tickSpin(now){
      if(!spin)return false;
      const t=Math.min(1,(now-spin.start)/spin.ms),e=easeInOut(t);
      root.rotation.y=e*Math.PI*2;sweep.u.uSweep.value=-e*Math.PI*2+.0001;
      if(t>=1){spin=null;root.rotation.y=0;fadeArc(sweep,now,900);}
      return true;
    }

    // Drawing on demand: each change asks for a frame; motion keeps asking until it settles.
    let queued=false,last=performance.now(),onScreen=true;
    function invalidate(){if(!queued){queued=true;requestAnimationFrame(frame);}}
    function frame(now){
      queued=false;
      const dt=Math.min(64,now-last);last=now;
      let busy=false;
      if(pointer.id==null&&(Math.abs(orbit.vy)>2e-6||Math.abs(orbit.vp)>2e-6)){
        orbit.yaw+=orbit.vy*dt;orbit.pitch=clampPitch(orbit.pitch+orbit.vp*dt);
        const k=Math.pow(.92,dt/16);orbit.vy*=k;orbit.vp*=k;busy=true;
      }
      if(drifting()){orbit.yaw+=.00008*dt;busy=true;}
      advance(now);
      const glowing=tickArcs(now),spinning=tickSpin(now);
      busy=glowing||spinning||busy||!!active||queue.length>0||pointer.mode==='turn';
      placeCamera();
      if(onScreen)renderer.render(scene,camera);
      if(busy)invalidate();
      else wakeForDrift();
    }
    // Drawing stops when nothing moves; one timer wakes the loop when the idle drift is due.
    let wake=null;
    function wakeForDrift(){
      const idle=performance.now()-lastInteraction;
      if(wake||reduced.matches||session?.running||idle>=4000)return;
      wake=setTimeout(()=>{wake=null;invalidate();},4000-idle+20);
    }
    new ResizeObserver(()=>{
      const w=Math.max(1,stage.clientWidth),h=Math.max(1,stage.clientHeight);
      renderer.setPixelRatio(Math.min(devicePixelRatio||1,2));renderer.setSize(w,h,false);
      camera.aspect=w/h;camera.updateProjectionMatrix();invalidate();
    }).observe(stage);
    new IntersectionObserver(entries=>{onScreen=entries.at(-1).isIntersecting;if(onScreen)invalidate();}).observe(stage);
    canvas.addEventListener('webglcontextlost',event=>{event.preventDefault();$('fault').textContent='The graphics context was lost. Reload the page to restore the cube view.';});

    // Pointer: drag a face to turn its layer, tap to turn it clockwise, drag elsewhere to orbit.
    const raycaster=new THREE.Raycaster(),ndc=new THREE.Vector2();
    const pointer={id:null,mode:null,x0:0,y0:0,x:0,y:0,hit:null,turn:null,samples:[],shift:false,alt:false};
    function pick(event){
      const rect=canvas.getBoundingClientRect();
      ndc.set((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1);
      raycaster.setFromCamera(ndc,camera);
      const hit=raycaster.intersectObjects(pickables,false)[0];
      if(!hit)return null;
      const point=root.worldToLocal(hit.point.clone()),abs=[Math.abs(point.x),Math.abs(point.y),Math.abs(point.z)];
      const k=abs.indexOf(Math.max(...abs)),normal=[0,0,0];normal[k]=Math.sign(point.getComponent(k));
      return {point,normal,grid:hit.object.parent.userData.grid};
    }
    function screenDelta(point,direction){
      const a=point.clone().project(camera),b=point.clone().addScaledVector(direction,.5).project(camera);
      return new THREE.Vector2((b.x-a.x)*canvas.clientWidth/2,-(b.y-a.y)*canvas.clientHeight/2);
    }
    function beginTurn(dx,dy){
      const {point,normal,grid}=pointer.hit,n=new THREE.Vector3(...normal);
      let best=null;
      for(let i=0;i<3;i++){
        if(normal[i])continue;
        const t=new THREE.Vector3();t.setComponent(i,1);
        const s=screenDelta(point,t),len=s.length();
        if(len<2)continue;
        const along=(dx*s.x+dy*s.y)/len;
        if(!best||Math.abs(along)>Math.abs(best.along))best={t,s,len,along};
      }
      if(!best){pointer.mode='orbit';return;}
      // Rotating about n x t moves the grabbed point along +t, so the drag sign is the angle sign.
      const axis=new THREE.Vector3().crossVectors(n,best.t),k=[0,1,2].find(i=>axis.getComponent(i)!==0),layer=grid[k];
      if(layer===0){pointer.mode='orbit';return;}
      const radius=Math.max(1,Math.hypot(...[0,1,2].filter(i=>i!==k).map(i=>point.getComponent(i))));
      const face=[0,0,0];face[k]=layer;
      const sense=axis.getComponent(k)*layer;
      grab(k,layer);
      // The layer leads the pointer a little, so a short swipe carries a whole quarter turn.
      pointer.turn={k,layer,axis,face,sense,dir:best.s.clone().divideScalar(best.len),scale:1.6*.5/best.len/radius,angle:0,arc:arcFor(face,0,1)};
      pointer.mode='turn';
      hideHover();
    }
    function dragAngle(dx,dy){const {dir,scale}=pointer.turn;return (dx*dir.x+dy*dir.y)*scale;}
    function cancelDrag(){
      if(pointer.mode!=='turn')return;
      fadeArc(pointer.turn.arc,performance.now(),200);
      pointer.mode='orbit';pointer.turn=null;layout(visual);
    }
    // A release past about a third of a quarter turn, momentum included, commits the next quarter.
    function endTurn(){
      const turn=pointer.turn,recent=pointer.samples.filter(s=>s.t>performance.now()-90);
      let velocity=0;
      if(recent.length>1){const a=recent[0],b=recent.at(-1);velocity=(dragAngle(b.x-pointer.x0,b.y-pointer.y0)-dragAngle(a.x-pointer.x0,a.y-pointer.y0))/Math.max(8,b.t-a.t);}
      const thrown=(turn.angle+velocity*110)/(Math.PI/2),quarters=Math.max(-2,Math.min(2,Math.sign(thrown)*Math.floor(Math.abs(thrown)+.68))),target=quarters*Math.PI/2;
      const time=reduced.matches?0:Math.max(70,Math.abs(target-turn.angle)/(Math.PI/2)*170);
      const clockwise=((-quarters*turn.sense)%4+4)%4,id=clockwise?FACE_OF_AXIS[turn.face.join()]+['','','2',"'"][clockwise]:null;
      const done=()=>{
        pointer.mode=null;pointer.turn=null;
        if(!id){layout(visual);fadeArc(turn.arc,performance.now(),240);pump();return;}
        visual=Cube.apply(visual,Cube.move(id));layout(visual);
        turn.arc.u.uSweep.value=target*turn.sense;fadeArc(turn.arc,performance.now(),560);
        if(session.humanTurn(id,{presented:true}))showTurn(Cube.move(id),{source:'human'});
        else{visual=Cube.apply(visual,Cube.move(Cube.inverse(id)));layout(visual);}
        pump();
      };
      pointer.mode='snap';
      if(!time){pivot.quaternion.setFromAxisAngle(turn.axis,target);done();return;}
      active={axis:turn.axis,from:turn.angle,to:target,start:performance.now(),ms:time,ease:easeOut,onAngle:angle=>{turn.arc.u.uSweep.value=angle*turn.sense;},done};
      invalidate();
    }
    const canTurn=()=>session?.player==='human'&&session.game&&!session.game.solved()&&!session.presenting&&session.mode!=='REQUEST FAILED';
    function hover(event){
      if(!canTurn()||pointer.id!=null||event.pointerType==='touch'){hideHover();return;}
      const hit=pick(event);
      if(!hit){hideHover();return;}
      const span=event.altKey?-Math.PI:event.shiftKey?.55:-.55;
      place(preview,hit.normal,span);
      Object.assign(preview,{peak:.5,fadeFrom:null});preview.u.uGlow.value=.5;preview.u.uRing.value=.7;preview.u.uSweep.value=span;preview.mesh.visible=true;
      stage.classList.add('can-turn');invalidate();
    }
    function hideHover(){if(preview.mesh.visible){preview.mesh.visible=false;invalidate();}stage.classList.remove('can-turn');}
    stage.addEventListener('pointerdown',event=>{
      if(event.button!==0||pointer.id!=null)return;
      lastInteraction=performance.now();
      const hit=canTurn()?pick(event):null;
      if(hit)settle();
      Object.assign(pointer,{id:event.pointerId,mode:hit?'press':'orbit',x0:event.clientX,y0:event.clientY,x:event.clientX,y:event.clientY,hit,turn:null,
        samples:[{t:performance.now(),x:event.clientX,y:event.clientY}],shift:event.shiftKey,alt:event.altKey});
      orbit.vy=orbit.vp=0;
      stage.setPointerCapture(event.pointerId);stage.classList.add('grabbing');hideHover();
    });
    stage.addEventListener('pointermove',event=>{
      if(pointer.id!==event.pointerId){hover(event);return;}
      const dx=event.clientX-pointer.x,dy=event.clientY-pointer.y;
      pointer.x=event.clientX;pointer.y=event.clientY;lastInteraction=performance.now();
      pointer.samples.push({t:performance.now(),x:event.clientX,y:event.clientY});
      if(pointer.samples.length>8)pointer.samples.shift();
      const tx=event.clientX-pointer.x0,ty=event.clientY-pointer.y0;
      if(pointer.mode==='press'&&tx*tx+ty*ty>=64)beginTurn(tx,ty);
      if(pointer.mode==='turn'){
        pointer.turn.angle=dragAngle(tx,ty);
        pivot.quaternion.setFromAxisAngle(pointer.turn.axis,pointer.turn.angle);
        pointer.turn.arc.u.uSweep.value=pointer.turn.angle*pointer.turn.sense;invalidate();
      } else if(pointer.mode==='orbit'){
        orbit.yaw-=dx*.0085;orbit.pitch=clampPitch(orbit.pitch+dy*.0085);
        const dt=Math.max(8,event.timeStamp-(pointer.at||event.timeStamp-16));pointer.at=event.timeStamp;
        orbit.vy=-dx*.0085/dt;orbit.vp=dy*.0085/dt;invalidate();
      }
    });
    const release=event=>{
      if(pointer.id!==event.pointerId)return;
      stage.classList.remove('grabbing');
      if(pointer.mode==='press'&&event.type==='pointerup'){
        const face=FACE_OF_AXIS[pointer.hit.normal.join()];
        session.humanTurn(face+(pointer.alt?'2':pointer.shift?"'":''));
        pointer.mode=null;
      } else if(pointer.mode==='turn')endTurn();
      else if(pointer.mode!=='snap')pointer.mode=null;
      if(pointer.mode==='orbit'||pointer.mode==='press')pointer.mode=null;
      if(performance.now()-(pointer.at||0)>60){orbit.vy=orbit.vp=0;}
      pointer.id=null;pointer.at=null;invalidate();
    };
    stage.addEventListener('pointerup',release);
    stage.addEventListener('pointercancel',event=>{if(pointer.mode==='turn')cancelDrag();release(event);});
    stage.addEventListener('pointerleave',hideHover);
    stage.addEventListener('contextmenu',event=>event.preventDefault());
    const orbitBy=(yaw,pitch)=>{orbit.yaw+=yaw;orbit.pitch=clampPitch(orbit.pitch+pitch);lastInteraction=performance.now();invalidate();};
    placeCamera();invalidate();
    // A key turn waits while a layer is in the hand, so the model and the view keep one order.
    const holding=()=>pointer.mode==='turn'||pointer.mode==='snap';
    return {scramble,present,settle,celebrate,orbitBy,invalidate,hideHover,holding};
  }

  // Without WebGL the net still shows the whole cube and keys still turn it.
  const cube=(()=>{
    try{return instrument();}
    catch(error){
      stage.classList.add('no-webgl');$('stage-fallback').hidden=false;canvas.hidden=true;
      const done=()=>Promise.resolve();
      return {scramble:done,present:(turn,info)=>{showTurn(turn,info);return done();},settle(){},celebrate(){},orbitBy(){},invalidate(){},hideHover(){},holding:()=>false};
    }
  })();

  // ---------- Readouts ----------
  function showTurn(turn,info){
    const move=$('hud-move');
    move.textContent=turn.id;move.classList.remove('fresh');void move.offsetWidth;move.classList.add('fresh');
    const count=session?.game?.history.length||0;
    $('hud-caption').textContent=info.source==='scramble'?'SCRAMBLE':(sourceName[session.player]+' · TURN '+count);
    setStage(info.source==='scramble'?'scramble':info.source==='human'?null:'turn');
  }
  function setStage(name=null){for(const id of ['scramble','choose','turn'])$('stage-'+id).classList.toggle('active',id===name);}
  const net=$('net'),netCells=[];
  for(const face of Cube.FACES){
    const box=document.createElement('div');box.className='face '+face;
    for(let i=0;i<9;i++){const cell=document.createElement('i');box.append(cell);netCells.push(cell);}
    net.append(box);
  }
  function renderNet(colors){
    for(let i=0;i<54;i++)netCells[i].style.background='var(--c'+colors[i]+')';
    net.setAttribute('aria-label',Cube.FACES.map(face=>Cube.FACE_NAME[face]+' '+Cube.faceText(colors,face)).join(', '));
  }
  function renderTape(){
    const g=session.game,tape=$('tape');
    const items=g.scramble.map(id=>'<li class="scramble">'+id+'</li>');
    items.push('<li class="divider" aria-hidden="true">SOLVE</li>');
    g.trace.forEach((entry,i)=>items.push('<li class="'+gradeOf(entry.before,entry.after)+(i===g.trace.length-1?' now':'')+'">'+entry.turn+'</li>'));
    tape.innerHTML=items.join('');
  }
  function origin(g){return g.trace[0]?{distance:g.trace[0].before,lower:g.trace[0].lowerBefore}:g.position;}
  function renderDescent(){
    const g=session.game,svg=$('descent'),W=320,H=112,left=34,right=10,top=8,bottom=20;
    const start=origin(g),points=[{i:0,d:start.distance,lower:start.lower}];
    g.trace.forEach((entry,i)=>points.push({i:i+1,d:entry.after,lower:entry.lower,grade:gradeOf(entry.before,entry.after)}));
    const values=points.map(p=>p.d??p.lower).concat(g.par||0,4),top0=Math.max(...values),every=top0>12?5:top0>6?2:1,maxD=Math.ceil(top0/every)*every,maxX=Math.max(points.length-1,g.par||0,6);
    const x=i=>left+i*(W-left-right)/maxX,y=d=>top+(1-d/maxD)*(H-top-bottom);
    const parts=[];
    for(let d=0;d<=maxD;d+=every)parts.push('<line class="grid" x1="'+left+'" x2="'+(W-right)+'" y1="'+y(d)+'" y2="'+y(d)+'"/><text class="axis" x="'+(left-12)+'" y="'+(y(d)+3.5)+'" text-anchor="end">'+d+'</text>');
    parts.push('<text class="axis" x="'+left+'" y="'+(H-4)+'">0</text><text class="axis" x="'+(W-right)+'" y="'+(H-4)+'" text-anchor="end">'+maxX+' TURNS</text>');
    if(g.par)parts.push('<line class="ideal" x1="'+x(0)+'" y1="'+y(g.par)+'" x2="'+x(g.par)+'" y2="'+y(0)+'"/>');
    for(let i=1;i<points.length;i++){
      const a=points[i-1],b=points[i],known=a.d!=null&&b.d!=null;
      parts.push('<line class="'+(known?'path':'bound')+'" x1="'+x(a.i)+'" y1="'+y(a.d??a.lower)+'" x2="'+x(b.i)+'" y2="'+y(b.d??b.lower)+'"/>');
    }
    points.forEach((p,i)=>{
      const cx=x(p.i),cy=y(p.d??p.lower);
      if(i===0)parts.push('<circle class="start" cx="'+cx+'" cy="'+cy+'" r="2.6"/>');
      else if(p.d==null)parts.push('<rect class="unknown" x="'+(cx-2.6)+'" y="'+(cy-2.6)+'" width="5.2" height="5.2"/>');
      else parts.push('<circle class="'+p.grade+'" cx="'+cx+'" cy="'+cy+'" r="3"/>');
    });
    const tip=points.at(-1);parts.push('<circle class="now" cx="'+x(tip.i)+'" cy="'+y(tip.d??tip.lower)+'" r="6.5"/>');
    svg.innerHTML='<title id="descent-label">Distance to solved after each turn</title>'+parts.join('');
  }
  const distanceText=position=>position.distance!=null?String(position.distance):'≥ '+position.lower;
  function renderDecision(){
    const latest=session.latest,rows=$('probabilities');
    rows.replaceChildren();
    $('selection-prob').textContent='—';
    if(!latest){
      $('decision-title').textContent=session.player==='human'?'YOUR SOLVE':'NEXT TURN';
      return;
    }
    if(latest.source==='reference'){
      $('decision-title').textContent='LOCAL SEARCH';
      $('selection-prob').textContent=latest.turn;
      const list=document.createElement('ol');list.className='plan';list.setAttribute('aria-label','Remaining line');
      for(const id of latest.plan.moves){const li=document.createElement('li');li.textContent=id;list.append(li);}
      rows.append(list);
      return;
    }
    if(latest.source==='human'){
      $('decision-title').textContent='YOUR TURN';
      $('selection-prob').textContent=latest.turn;
      return;
    }
    $('decision-title').textContent=(latest.method==='reasoning'?'REASONING':'SYSTEM ONE')+' · '+latest.turn;
    if(!latest.probabilities){
      const note=document.createElement('p');note.className='screen-note';
      note.textContent='Generated answer · '+latest.usage.output_tokens.toLocaleString()+' output tokens. Move probabilities are unavailable.';
      rows.append(note);
      return;
    }
    $('selection-prob').textContent=pct(latest.probabilities[latest.code]);
    const ranked=latest.options.map(option=>({option,p:latest.probabilities[option.code],row:latest.analysis?.find(row=>row.id===option.id)})).sort((a,b)=>b.p-a.p).slice(0,6);
    for(const {option,p,row} of ranked){
      const line=document.createElement('div'),grade=row?gradeOf(0,row.delta):'unknown';
      line.className='prob-row'+(option.id===latest.turn?' chosen':'');
      line.innerHTML='<span>'+option.id+'</span><span class="prob-track"><span class="prob-fill" style="width:'+(p*100)+'%"></span></span><span>'+(p*100).toFixed(1)+'</span><span class="grade '+grade+'">'+gradeText[grade]+'</span>';
      rows.append(line);
    }
  }
  function renderNote(){
    const g=session.game,latest=session.latest,note=$('decision-note'),mode=session.mode;
    if(mode==='SCRAMBLING'){note.textContent='The scramble plays from solved. The clock waits.';return;}
    if(mode==='WARM-UP'){note.textContent='Waiting for the first complete '+(session.player==='reasoning'?'reasoning':'System One')+' answer. This warm-up is excluded from the solve clock.';return;}
    if(mode==='REQUEST FAILED'){note.textContent='The request failed; the cube did not move. Run retries.';return;}
    if(g.solved()){note.textContent=summaryLine()+' Reset replays this seed; change the seed for a new scramble.';return;}
    if(mode==='TURN LIMIT'){note.textContent='Stopped at the '+session.limit+'-turn limit. Switch the solver to local search or You to finish from here.';return;}
    if(mode==='NO NEW TURN'){note.textContent='Every next turn returns to a visited position. Reset, allow repeats, or switch the solver to continue.';return;}
    if(session.player==='human'){
      const entry=g.trace.at(-1);
      note.textContent=entry?describeGrade(entry)+' Drag or tap a face to keep going.':'Drag a face to turn it, or tap it for a clockwise turn. Your clock starts with the first turn.';
      return;
    }
    if(latest?.source==='reference'){
      const left=latest.plan.moves.length;
      note.textContent=(latest.plan.optimal?'Optimal line':'Two-phase line, not proven shortest')+' · '+left+' turn'+(left===1?'':'s')+' left · '+(latest.searched?'searched in '+ms(latest.ms)+' ms':'following the line')+'.';
      return;
    }
    if(latest?.source==='live'){
      const entry=g.trace[latest.index];
      note.textContent=(latest.choiceWarning?latest.choiceWarning+' ':'')+(entry?describeGrade(entry):'')+(latest.excluded?.length?' '+latest.excluded.length+' returning turns excluded.':'');
      return;
    }
    note.textContent=session.modelPlayer?'Run sends one warm-up request, then one request per turn. '+(session.player==='reasoning'?'Thinking can take several seconds per request. ':'')+'The solver grades each answer; Qwen never sees the grade.':'Run starts local search. It proves the shortest line, then follows it.';
  }
  function describeGrade(entry){
    const grade=gradeOf(entry.before,entry.after);
    return entry.turn+(grade==='closer'?' was optimal: one turn closer.':grade==='level'?' kept the distance.':grade==='farther'?' moved one turn farther.':' is past the exact-search horizon; its grade is unknown.');
  }
  function summaryLine(){
    const s=session.summary(),g=session.game,time=clockText(s.clockMs);
    return time.value+' '+time.unit+' · '+s.turns+' turns'+(g.par?' · par '+g.par:'')+(s.graded?' · '+s.optimal+' of '+s.graded+' graded turns optimal':'')+'.';
  }
  function renderInspector(){
    const latest=session.latest,body=$('candidate-table');
    body.replaceChildren();
    $('excluded-turns').hidden=!latest?.excluded?.length;
    $('excluded-turns').textContent=latest?.excluded?.length?'Excluded because they revisit a position: '+latest.excluded.join(' ')+'. Probabilities are conditional on the offered turns.':'';
    $('inspector-count').textContent=latest?.source==='live'?ms(latest.ms)+' ms':'';
    if(latest?.source==='live'){
      latest.options.forEach(option=>{
        const tr=document.createElement('tr'),row=latest.analysis?.find(row=>row.id===option.id),grade=row?gradeOf(0,row.delta):'unknown';
        if(option.id===latest.turn)tr.className='selected';
        tr.innerHTML='<td>'+option.code+'</td><td>'+option.id+'</td><td>'+option.misplaced+'</td><td class="'+grade+'">'+(row?.distance!=null?row.distance+' ('+gradeText[grade]+')':'—')+'</td><td>'+pct(latest.probabilities?.[option.code])+'</td>';
        body.append(tr);
      });
    }
    const failure=session.failure;
    $('request-wire').textContent=failure?.receipt?.request?JSON.stringify(failure.receipt.request,null,2):latest?.request?JSON.stringify(latest.request,null,2):'No request yet.';
    $('response-wire').textContent=failure?.receipt?JSON.stringify(failure.receipt.response,null,2):latest?.response?JSON.stringify(latest.response,null,2):latest?.source==='reference'?'Local search does not call the API.':'No response yet.';
  }
  // Search and request clocks often finish inside a second, so they read in milliseconds there.
  function clockText(value=session.clock.read()){
    const small=value>0&&value<1000&&session.player!=='human';
    return {value:small?value.toFixed(value<10?2:1):(value/1000).toFixed(2),unit:small?'ms':'s'};
  }
  const rate=n=>n>=1000?Math.round(n).toLocaleString('en-US'):n>=100?n.toFixed(0):n.toFixed(2);
  function paintClock(){
    if(!session?.game)return;
    const value=session.clock.read(),turns=session.game.history.length,text=clockText(value);
    $('clock').textContent=text.value;$('clock-unit').textContent=text.unit;
    $('tps').textContent=value>0&&turns?rate(turns/(value/1000)):'—';
  }
  let clockLoop=false;
  function runClock(){
    if(clockLoop)return;
    clockLoop=true;
    const tick=()=>{paintClock();if(session.clock.running)requestAnimationFrame(tick);else clockLoop=false;};
    requestAnimationFrame(tick);
  }
  function controls(){
    const g=session.game,busy=session.running,human=session.player==='human';
    $('run').textContent=busy?'PAUSE':!g||g.solved()?'AGAIN':g.history.length||session.mode==='PAUSED'?'RESUME':human?'START':'RUN';
    $('run').disabled=!g||session.mode==='INDEXING'||(session.mode==='TURN LIMIT'&&session.modelPlayer);
    $('step').disabled=busy||human||!g||g.solved()||session.mode==='TURN LIMIT';
    for(const id of settings)$(id).disabled=busy;
    $('reasoning-settings').hidden=session.player!=='reasoning';
    $('repeat-settings').hidden=!session.modelPlayer;
    $('export').disabled=!g||(!session.records.length&&!session.failure);
  }
  function showBanner(title,detail,kind=''){
    const banner=$('banner');banner.className='banner cube-banner show '+kind;
    banner.replaceChildren();const b=document.createElement('b'),span=document.createElement('span');b.textContent=title;span.textContent=detail;banner.append(b,span);
  }
  function hideBanner(){$('banner').className='banner cube-banner';$('banner').replaceChildren();}

  const view={
    scramble:game=>cube.scramble(game),
    present:(turn,info)=>cube.present(turn,info).then(()=>setStage(session.running&&session.player!=='human'?'choose':null)),
    settle:()=>cube.settle(),
    connection:()=>({endpoint:$('endpoint').value.trim(),model:$('model').value.trim(),apiKey:$('key').value,maxTokens:Number($('max-tokens').value),avoidRepeats:$('repeat-policy').value==='avoid'}),
    update(event){
      const g=session.game;
      if(!g)return;
      const mode=session.mode;
      $('mode').textContent=mode;$('power-label').textContent=mode;
      $('led').classList.toggle('on',session.running||mode==='SOLVED'||mode==='SCRAMBLING');
      $('source').textContent=sourceName[session.player];$('session-note').textContent=sourceName[session.player]+' · SEED '+g.seed;
      // After a hand-off the clock holds more than one solver's time.
      $('clock-label').textContent=new Set(session.records.map(r=>r.method||r.source)).size>1?'SOLVER CLOCK':clockName[session.player];
      $('solve-state').textContent=g.solved()?'Solved':g.misplaced()+' stickers out';
      $('moves').textContent=String(g.history.length);
      $('par').textContent=g.par??'—';$('legend-par').hidden=!g.par;
      const s=session.summary();
      $('p50').textContent=s.p50==null?'—':s.p50.toFixed(0);
      $('optimal-turns').textContent=s.graded?s.optimal+' / '+s.graded+(s.turns>s.graded?' · '+(s.turns-s.graded)+' ungraded':''):'—';
      $('optimal-mass').textContent=pct(s.optimalMass);
      const usage=session.records.filter(r=>r.usage);
      $('tokens').textContent=usage.length?usage.reduce((a,r)=>a+r.usage.input_tokens,0)+' / '+usage.reduce((a,r)=>a+r.usage.output_tokens,0):'—';
      $('warmup').textContent=session.warmup?ms(session.warmup.ms)+' ms':'—';
      $('decision-ms').textContent=session.latest?.ms!=null?ms(session.latest.ms)+' ms':'—';
      const position=g.position,dist=$('hud-distance');
      dist.textContent=g.solved()?'0':distanceText(position);dist.classList.toggle('unknown',position.distance==null&&!g.solved());
      $('hud-par').textContent=g.par?'PAR '+g.par:'PAR UNPROVEN';
      $('distance-now').textContent=g.solved()?'0':distanceText(position);
      const left=g.misplaced();$('misplaced').value=left;$('misplaced-label').textContent=left+' / 48';
      renderNet(g.colors);
      renderTape();renderDescent();renderDecision();renderNote();paintClock();controls();
      if(session.clock.running)runClock();
      if(mode==='SCRAMBLING')setStage('scramble');
      else if(!session.running)setStage(null);
      else if(event==='mode')setStage('choose');
      if(event==='mode'&&mode==='READY'&&!g.history.length)$('hud-caption').textContent='SCRAMBLED';
      if(['decision','failure','reset','player'].includes(event))renderInspector();
      $('fault').textContent=session.failure?session.failure.message:'';
      if(event==='reset'){hideBanner();$('hud-move').textContent='—';$('hud-caption').textContent='SCRAMBLE';}
      if(event==='failure'){showBanner('REQUEST FAILED',session.failure.message,'error');$('inspector').open=true;}
      if(event==='solved'){showBanner('SOLVED',summaryLine());cube.celebrate();}
      if(event==='solved'||event==='reset'){lastInteraction=performance.now();cube.invalidate();}
      if(mode==='TURN LIMIT'&&event==='mode')showBanner('TURN LIMIT','Stopped after '+g.history.length+' turns.','limit');
      if(mode==='NO NEW TURN'&&event==='mode')showBanner('NO NEW TURN','All next positions have already been visited.','limit');
      if((mode==='SOLVING'&&event==='mode')||(event==='player'&&!g.solved()))hideBanner();
    }
  };
  session=new RubiksSession.Session({view,defer:fn=>window.requestIdleCallback?requestIdleCallback(fn,{timeout:250}):setTimeout(fn,40)});

  function settingsNow(){
    const raw=$('distance').value,seed=Number($('seed').value);
    return {distance:raw==='random'?'random':Number(raw),seed,player:$('player').value};
  }
  async function reset(){
    const next=settingsNow();
    if(!Number.isInteger(next.seed)||next.seed<0||next.seed>4294967295){$('fault').textContent='The seed must be a whole number from 0 to 4294967295.';return;}
    $('fault').textContent='';
    $('scramble-note').textContent=next.distance==='random'?'A uniformly random position, scrambled by reversing a two-phase solution. It is about twenty turns deep; par is unproven.':'The solver proves each scramble is exactly this many turns from solved, so its length is par.';
    if(next.distance==='random'&&!CubeSolver.ready('two-phase')){session.halt();$('mode').textContent='INDEXING';await CubeSolver.prepareAsync('two-phase');}
    await session.reset(next);
  }
  $('run').addEventListener('click',()=>{
    lastInteraction=performance.now();
    if(session.game?.solved()){reset().then(()=>{if(session.player!=='human')session.start();});return;}
    session.toggle(false);
  });
  $('step').addEventListener('click',()=>session.start(true));
  $('reset').addEventListener('click',reset);
  for(const id of ['distance','seed'])$(id).addEventListener('change',reset);
  $('player').addEventListener('change',()=>{if(session.game)session.setPlayer($('player').value);});
  $('pace').addEventListener('input',()=>{$('pace-value').textContent=Number($('pace').value).toFixed(2).replace(/0$/,'').replace(/\.$/,'')+'×';});
  $('export').addEventListener('click',()=>{
    const blob=new Blob([JSON.stringify(session.exportData(),null,2)],{type:'application/json'});
    const url=URL.createObjectURL(blob),link=document.createElement('a');
    link.href=url;link.download='ninfer-cube-'+session.game.seed+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  const KEYS={KeyU:'U',KeyD:'D',KeyL:'L',KeyR:'R',KeyF:'F',KeyB:'B'};
  addEventListener('keydown',event=>{
    lastInteraction=performance.now();cube.invalidate();
    if(event.ctrlKey||event.metaKey||event.target.matches('input, textarea, select, [contenteditable]'))return;
    if(event.code==='KeyP'){if(!$('run').disabled){event.preventDefault();$('run').click();}return;}
    const face=KEYS[event.code];
    if(face&&session.player==='human'){
      if(!cube.holding()&&session.humanTurn(face+(event.altKey?'2':event.shiftKey?"'":'')))event.preventDefault();
      return;
    }
    if(document.activeElement===stage&&event.key.startsWith('Arrow')){
      event.preventDefault();
      const step=.16;cube.orbitBy(event.key==='ArrowLeft'?step:event.key==='ArrowRight'?-step:0,event.key==='ArrowUp'?-step:event.key==='ArrowDown'?step:0);
    }
  });
  document.addEventListener('visibilitychange',()=>{if(document.hidden){session.pause();cube.settle();}});
  reduced.addEventListener('change',()=>{if(reduced.matches)cube.settle();});

  $('mode').textContent='INDEXING';$('power-label').textContent='INDEXING';
  $('decision-note').textContent='Building the solver tables.';
  CubeSolver.prepareAsync('exact').then(reset).then(()=>CubeSolver.prepareAsync('two-phase'));
})();
