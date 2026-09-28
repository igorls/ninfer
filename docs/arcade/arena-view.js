// Procedural articulated fighters and a floodlit training pit. No rendering effect changes time.
const ArenaView = (() => {
  'use strict';
  const T=THREE, clamp=(n,a,b)=>Math.max(a,Math.min(b,n)), mix=(a,b,t)=>a+(b-a)*t;
  const cache=new Map();
  function bevel(w,h,d,r=.035){
    const key=[w,h,d,r].join(',');if(cache.has(key))return cache.get(key);
    const g=new T.BoxGeometry(w,h,d,4,4,4),p=g.attributes.position,v=new T.Vector3(),c=new T.Vector3();
    for(let i=0;i<p.count;i++){v.fromBufferAttribute(p,i);c.set(clamp(v.x,-w/2+r,w/2-r),clamp(v.y,-h/2+r,h/2-r),clamp(v.z,-d/2+r,d/2-r));v.sub(c).normalize().multiplyScalar(r).add(c);p.setXYZ(i,v.x,v.y,v.z);}g.computeVertexNormals();cache.set(key,g);return g;
  }
  const mat=(color,metalness=.6,roughness=.3)=>new T.MeshStandardMaterial({color,metalness,roughness});
  function mesh(parent,geometry,material,x=0,y=0,z=0){const m=new T.Mesh(geometry,material);m.position.set(x,y,z);m.castShadow=true;m.receiveShadow=true;parent.add(m);return m;}
  function box(p,w,h,d,m,x=0,y=0,z=0,r=.035){return mesh(p,bevel(w,h,d,r),m,x,y,z);}
  function ring(p,radius,tube,m,y=0){const n=mesh(p,new T.TorusGeometry(radius,tube,6,96),m,0,y,0);n.rotation.x=Math.PI/2;return n;}
  function fighter(color,brute=false){
    const root=new T.Group(),body=new T.Group();root.add(body);body.position.y=1.02;
    const shell=mat(brute?0x4c3633:0xb5c7c6,.7,.3),dark=mat(0x17232b,.72,.3),trim=mat(brute?0x993d27:0x286b77,.7,.27),joint=mat(0x080f16,.65,.27);
    const glow=new T.MeshStandardMaterial({color,emissive:color,emissiveIntensity:1.1,roughness:.25,metalness:.2});
    const chest=box(body,.67,.72,.41,shell,0,.42,0,.075);chest.rotation.x=.06;
    box(body,.42,.28,.16,dark,0,.45,.25);box(body,.22,.07,.04,glow,0,.48,.35,.01);
    box(body,.48,.21,.33,dark,0,-.05,0,.04);
    for(const side of [-1,1]){box(body,.10,.4,.08,trim,side*.235,.34,.25,.014);box(body,.16,.2,.34,shell,side*.26,-.17,0,.035);}
    for(let i=0;i<3;i++)box(body,.38,.055,.09,joint,0,.09+i*.09,.218,.01);
    const head=new T.Group();body.add(head);head.position.set(0,.98,.04);
    mesh(head,new T.CylinderGeometry(.10,.11,.16,12),joint,0,-.20,0);
    box(head,.40,.34,.37,shell,0,0,0,.055);box(head,.36,.105,.08,joint,0,.01,.185,.02);box(head,.29,.035,.018,glow,0,.019,.235,.008);
    box(head,.21,.09,.12,dark,0,-.13,.18,.02);
    if(brute){for(const side of [-1,1])box(head,.09,.27,.09,trim,side*.23,.08,-.06,.015);}
    const arms=[],legs=[];
    for(const side of [-1,1]){
      const arm=new T.Group();arm.position.set(side*.46,.67,0);body.add(arm);
      mesh(arm,new T.SphereGeometry(.16,12,10),joint);
      box(arm,brute?.43:.30,brute?.30:.22,.43,shell,side*.02,-.02,0,.05);
      box(arm,.13,.07,.37,trim,side*.13,.09,.02,.014);
      box(arm,.22,.39,.25,dark,0,-.24,0,.04);
      const fore=new T.Group();fore.position.y=-.45;arm.add(fore);mesh(fore,new T.SphereGeometry(.105,10,8),joint);
      box(fore,.25,.37,.3,shell,0,-.22,.02,.045);box(fore,.09,.24,.035,glow,side*.08,-.24,.184,.013);
      box(fore,.21,.18,.21,joint,0,-.45,.02,.035);
      if(brute&&side===1){
        mesh(fore,new T.CylinderGeometry(.047,.047,.93,10),trim,0,-.66,.10);
        box(fore,.87,.36,.39,dark,0,-1.06,.10,.055);
        for(const s of [-1,1]){box(fore,.16,.43,.45,shell,s*.38,-1.06,.1,.03);box(fore,.045,.33,.34,glow,s*.48,-1.06,.1,.013);}
      } else if(!brute){
        box(fore,.075,.64,.10,dark,side*.18,-.27,.16,.014);
        box(fore,.035,.55,.055,glow,side*.23,-.26,.20,.01);
      }
      arms.push({upper:arm,fore});
      const leg=new T.Group();leg.position.set(side*.21,-.20,0);body.add(leg);
      mesh(leg,new T.SphereGeometry(.13,12,10),joint);
      box(leg,.27,.41,.30,shell,0,-.24,0,.045);box(leg,.09,.22,.04,trim,side*.075,-.24,.165,.012);
      const shin=new T.Group();shin.position.y=-.48;leg.add(shin);
      mesh(shin,new T.SphereGeometry(.12,12,10),joint);box(shin,.23,.34,.27,dark,0,-.20,0,.04);
      box(shin,.18,.26,.10,shell,0,-.2,.15,.02);box(shin,.06,.20,.025,glow,0,-.2,.213,.008);
      box(shin,.29,.16,.46,shell,0,-.40,.095,.04);legs.push({upper:leg,shin});
    }
    const shield=mesh(root,new T.CircleGeometry(.75,48),new T.MeshBasicMaterial({color,transparent:true,opacity:.2,side:T.DoubleSide,depthWrite:false}),0,1.35,.68);shield.visible=false;
    const rim=mesh(shield,new T.RingGeometry(.69,.71,48),new T.MeshBasicMaterial({color,transparent:true,opacity:.8,side:T.DoubleSide}),0,0,.002);
    root.scale.setScalar(brute?1.12:1);
    return {root,body,head,arms,legs,shield,brute,color};
  }
  function environment(renderer){
    const s=new T.Scene();s.add(new T.Mesh(new T.BoxGeometry(20,20,20),new T.MeshBasicMaterial({color:0x172333,side:T.BackSide})));
    for(const [x,y,z,w,h,c,p] of [[0,9,0,8,5,0xffe9d0,4],[-9,3,2,4,6,0x80e8ff,3],[9,3,-1,4,6,0xff8464,3]]){
      const m=new T.Mesh(new T.PlaneGeometry(w,h),new T.MeshBasicMaterial({color:new T.Color(c).multiplyScalar(p),side:T.DoubleSide}));m.position.set(x,y,z);m.lookAt(0,1,0);s.add(m);
    }
    const gen=new T.PMREMGenerator(renderer),target=gen.fromScene(s,.025);gen.dispose();s.traverse(o=>{if(o.isMesh){o.geometry.dispose();o.material.dispose();}});return target.texture;
  }
  function floorTexture(){
    const c=document.createElement('canvas');c.width=c.height=1536;const x=c.getContext('2d');
    x.fillStyle='#18242c';x.fillRect(0,0,1536,1536);x.translate(768,768);x.strokeStyle='#31444c';x.lineWidth=2;
    for(const r of [170,510,650]){x.beginPath();x.arc(0,0,r,0,Math.PI*2);x.stroke();}
    x.lineWidth=2;for(let i=0;i<64;i++){const a=i*Math.PI/32;x.save();x.rotate(a);x.beginPath();x.moveTo(620,0);x.lineTo(i%4===0?647:631,0);x.stroke();x.restore();}
    for(const s of [-1,1]){x.fillStyle=s<0?'#426d79':'#785349';x.fillRect(s*300-45,-95,90,8);x.fillRect(s*300-45,85,90,8);}
    x.textAlign='center';x.fillStyle='#52616a';x.font='600 65px "Barlow Condensed", sans-serif';x.fillText('N I N F E R',0,-20);x.font='500 25px "Kode Mono", monospace';x.fillText('A R E N A   D U E L',0,29);
    x.fillStyle='#61717c';x.font='22px monospace';x.fillText('REACTION / IMPACT / RECOVERY',0,555);
    const tex=new T.CanvasTexture(c);tex.colorSpace=T.SRGBColorSpace;tex.anisotropy=4;return tex;
  }
  class View {
    constructor(canvas){
      this.canvas=canvas;this.renderer=new T.WebGLRenderer({canvas,antialias:true,powerPreference:'high-performance'});this.renderer.setPixelRatio(Math.min(devicePixelRatio,1.7));
      this.renderer.outputColorSpace=T.SRGBColorSpace;this.renderer.toneMapping=T.ACESFilmicToneMapping;this.renderer.toneMappingExposure=1.2;
      this.renderer.shadowMap.enabled=true;this.renderer.shadowMap.type=T.PCFSoftShadowMap;
      this.scene=new T.Scene();this.scene.background=new T.Color(0x0b121b);this.scene.fog=new T.FogExp2(0x0b121b,.047);this.scene.environment=environment(this.renderer);
      this.camera=new T.PerspectiveCamera(32,1,.1,80);this.camera.position.set(0,4.8,10.8);this.camera.lookAt(0,.8,0);
      this.scene.add(new T.HemisphereLight(0xc3e7ff,0x111621,1.5));
      const fill=new T.DirectionalLight(0xc5eaff,1.8);fill.position.set(1,3,6);this.scene.add(fill);
      const light=new T.DirectionalLight(0xffead0,3.8);light.position.set(-3,8,4);light.castShadow=true;light.shadow.mapSize.set(2048,2048);Object.assign(light.shadow.camera,{left:-7,right:7,top:5,bottom:-5,near:1,far:24});light.shadow.normalBias=.04;light.shadow.bias=-.0003;this.scene.add(light);
      const cyan=new T.PointLight(0x30cfff,24,13,2);cyan.position.set(-4,3,-1);this.scene.add(cyan);const red=new T.PointLight(0xff5630,26,13,2);red.position.set(4,3,-1);this.scene.add(red);
      const steel=mat(0x16222b,.85,.31),edge=mat(0x35434b,.8,.24),black=mat(0x080e17,.5,.5),amber=new T.MeshBasicMaterial({color:0xecb56c});
      mesh(this.scene,new T.CylinderGeometry(6.5,6.7,.32,8),steel,0,-.21,0);
      const floor=mesh(this.scene,new T.CircleGeometry(6.48,8),new T.MeshStandardMaterial({map:floorTexture(),metalness:.52,roughness:.46}),0,-.043,0);floor.rotation.x=-Math.PI/2;floor.rotation.z=Math.PI/8;
      ring(this.scene,5.8,.015,edge,-.035);ring(this.scene,6.32,.013,amber,-.03);
      const ground=mesh(this.scene,new T.PlaneGeometry(100,100),black,0,-.40,0);ground.rotation.x=-Math.PI/2;
      for(let i=0;i<8;i++){
        const a=i*Math.PI/4+Math.PI/8,g=new T.Group();g.position.set(Math.sin(a)*6.25,0,Math.cos(a)*6.25);g.rotation.y=a;this.scene.add(g);
        box(g,.30,.34,.6,edge,0,-.12,0,.025);box(g,.18,.035,.35,amber,0,.06,0,.01);
      }
      // Lighting gantry, safety barrier, and the pit's structural hardware.
      for(const side of [-1,1]){
        const gantry=new T.Group();gantry.position.set(side*4.5,0,-3.8);this.scene.add(gantry);
        box(gantry,.15,3.25,.15,steel,0,1.6,0);box(gantry,1.35,.13,.23,edge,0,3.2,0);
        for(let i=0;i<4;i++)box(gantry,.24,.1,.17,new T.MeshBasicMaterial({color:0xf9e6cd}),-.48+i*.32,3.1,.1,.014);
        box(gantry,.6,.17,.6,edge,0,.05,0);
        for(let i=0;i<3;i++){const post=box(this.scene,.07,.75,.07,edge,side*(3.5+i*.9),.2,-4.8);}
        box(this.scene,2.9,.055,.055,edge,side*4.4,.60,-4.8,.01);box(this.scene,2.9,.04,.04,steel,side*4.4,.22,-4.8,.009);
      }
      this.hero=fighter(0x63eafa);this.foe=fighter(0xff6743,true);this.scene.add(this.hero.root,this.foe.root);
      this.hero.root.rotation.y=Math.PI/2-.23;this.foe.root.rotation.y=-Math.PI/2+.23;
      this.markers=[];
      for(const [r,color] of [[.75,0x70e6ed],[.85,0xf77250]]){const group=new T.Group();this.scene.add(group);ring(group,r,.017,new T.MeshBasicMaterial({color}),-.025);this.markers.push(group);}
      const tg=new T.RingGeometry(.2,1,64,1,-Math.PI*.3,Math.PI*.6);
      this.telegraph=mesh(this.scene,tg,new T.MeshBasicMaterial({color:0xff6b3e,transparent:true,opacity:.20,side:T.DoubleSide,depthWrite:false}),0,-.022,0);this.telegraph.rotation.x=-Math.PI/2;this.telegraph.rotation.z=Math.PI;
      this.impactRing=ring(this.scene,1,.022,new T.MeshBasicMaterial({color:0xafffff,transparent:true,opacity:0,depthWrite:false}),.08);this.impactRing.visible=false;
      this.particles=[];this.eventCursor=0;this.lastHit=-1000;this.lastTime=0;this.shake=0;
      const pg=new T.BufferGeometry();this.positions=new Float32Array(160*3);this.colors=new Float32Array(160*3);pg.setAttribute('position',new T.BufferAttribute(this.positions,3));pg.setAttribute('color',new T.BufferAttribute(this.colors,3));pg.setDrawRange(0,0);
      this.sparkMesh=new T.Points(pg,new T.PointsMaterial({size:.055,vertexColors:true,transparent:true,opacity:.95,depthWrite:false,blending:T.AdditiveBlending}));this.scene.add(this.sparkMesh);
      this.trails=[];for(let i=0;i<2;i++){const arc=mesh(this.scene,new T.TorusGeometry(.95,.025,5,36,Math.PI*1.1),new T.MeshBasicMaterial({color:i?0xff8757:0x83efff,transparent:true,opacity:0,depthWrite:false}));this.trails.push(arc);}
      this.resize();new ResizeObserver(()=>this.resize()).observe(canvas.parentElement);
    }
    resize(){const r=this.canvas.parentElement.getBoundingClientRect();if(!r.width||!r.height)return;this.renderer.setSize(r.width,r.height,false);this.camera.aspect=r.width/r.height;this.mobile=r.width<600;this.camera.fov=this.mobile?38:32;this.camera.updateProjectionMatrix();this.needsRender=true;}
    reset(){this.eventCursor=0;this.particles=[];this.lastHit=-1000;this.shake=0;this.impactRing.visible=false;this.needsRender=true;}
    pose(f,state,time,enemy=false,reduced=false){
      const elapsed=time-state.since,a=state.action,idle=Math.sin(time*.003),walk=a==='advance'||a==='retreat'||(enemy&&a==='idle');
      let lean=.08,drop=0,armL=-.35,armR=-.6,foreL=-.95,foreR=-1.1;
      if(a==='guard'){armL=-1.05;armR=-1.05;foreL=-1.5;foreR=-1.5;lean=.18;}
      if(a==='dodge'){drop=-.32;lean=.55;armL=.4;armR=.4;foreL=-.9;foreR=-.9;}
      if(a==='strike'||a==='heavy'){
        const hit=a==='strike'?150:360,duration=a==='strike'?380:700,progress=elapsed/hit;
        const extension=progress<1?Math.pow(progress,3):Math.max(0,1-(elapsed-hit)/(duration-hit));
        armR=mix(a==='heavy'?.5:-.3,-1.65,extension);foreR=mix(-1.5,-.12,extension);lean=.12+extension*.26;
      }
      if(enemy&&a==='telegraph'){
        const progress=clamp(elapsed/state.attack.windup,0,1);
        if(state.attack.type==='crush'){armR=-2.7*progress;foreR=-.5;armL=-.6;lean=-.12*progress;}
        else{armR=.7*progress;foreR=-1.2;lean=-.14*progress;}
      }
      if(enemy&&a==='recovery'){
        const follow=Math.max(0,1-elapsed/450);armR=-1.6*follow;foreR=-.2;lean=.45*follow;
      }
      if(a==='stagger'){lean=-.30;armL=.3;armR=.3;}
      f.root.position.x=state.x;f.body.position.y=1.04+drop+(reduced?0:idle*.018);f.body.rotation.x=lean;
      f.arms[0].upper.rotation.set(armL,0,-.17);f.arms[1].upper.rotation.set(armR,0,.17);f.arms[0].fore.rotation.x=foreL;f.arms[1].fore.rotation.x=foreR;
      const step=walk&&!reduced?Math.sin(time*.013)*.32:0;
      f.legs.forEach((l,i)=>{l.upper.rotation.x=(i?step:-step)-.08;l.shin.rotation.x=.16+(i?Math.max(0,-step):Math.max(0,step));});
      f.head.rotation.x=-lean*.4;f.shield.visible=a==='guard';f.shield.material.opacity=.13+(reduced?0:Math.sin(time*.02)*.04);
      const hurt=clamp(1-(time-state.hurtAt)/220,0,1);f.body.rotation.z=hurt*.13;
      if(state.hp<=0){f.root.rotation.z=enemy?-.95:.95;f.body.position.y=.6;}else f.root.rotation.z=0;
    }
    impact(event,time,reduced){
      if(!['hit','parry','block','dodge'].includes(event.type))return;
      const color=new T.Color(event.type==='parry'?0xc4ffff:event.actor==='player'?0x6ae5f5:0xff955e);
      const count=reduced?8:event.type==='parry'?45:30;
      for(let i=0;i<count;i++){const a=i*2.39996,v=1+(i%7)*.45;this.particles.push({x:event.x??0,y:1.4,z:.1,vx:Math.cos(a)*v,vy:1.5+(i%5)*.8,vz:Math.sin(a)*v,life:.3+(i%5)*.09,color});}
      this.particles=this.particles.slice(-160);this.lastHit=time;this.shake=reduced?0:event.heavy?.07:.035;
      this.impactRing.position.x=event.x??0;this.impactRing.material.color.copy(color);this.impactRing.visible=true;
    }
    render(game,clock,reduced=false){
      // A settled ready/paused/result screen does not need to compete with inference for the GPU.
      if(!this.needsRender&&this.drawnTime===game.time&&!this.particles.length&&clock-this.lastHit>500)return;
      this.drawnTime=game.time;this.needsRender=false;
      const dt=Math.min(.05,Math.max(0,(clock-this.lastTime)/1000));this.lastTime=clock;
      this.pose(this.hero,game.player,game.time,false,reduced);this.pose(this.foe,game.enemy,game.time,true,reduced);
      this.markers[0].position.x=game.player.x;this.markers[1].position.x=game.enemy.x;
      const incoming=game.enemy.action==='telegraph';this.telegraph.visible=incoming;
      if(incoming){const a=game.enemy.attack,progress=clamp((game.time-a.started)/a.windup,0,1);this.telegraph.position.x=game.enemy.x;this.telegraph.scale.setScalar(a.range);this.telegraph.material.opacity=.07+progress*.27;this.telegraph.material.color.set(a.blockable?0xffbb52:0xff542b);}
      for(const event of game.events){if(event.id<=this.eventCursor)continue;this.eventCursor=event.id;this.impact(event,clock,reduced);}
      this.particles=this.particles.filter(p=>{p.life-=dt;p.x+=p.vx*dt;p.y+=p.vy*dt;p.z+=p.vz*dt;p.vy-=12*dt;return p.life>0&&p.y>0;});
      this.particles.forEach((p,i)=>{this.positions.set([p.x,p.y,p.z],i*3);this.colors.set([p.color.r,p.color.g,p.color.b],i*3);});
      this.sparkMesh.geometry.setDrawRange(0,this.particles.length);this.sparkMesh.geometry.attributes.position.needsUpdate=true;this.sparkMesh.geometry.attributes.color.needsUpdate=true;
      const age=clock-this.lastHit;this.impactRing.visible=age<430;this.impactRing.scale.setScalar(.3+age*.004);this.impactRing.material.opacity=Math.max(0,.8-age/500);
      [game.player,game.enemy].forEach((s,i)=>{const arc=this.trails[i];const strike=i?s.action==='recovery'&&game.time-s.since<180:(s.action==='strike'||s.action==='heavy')&&Math.abs(game.time-s.attack?.at)<100;arc.visible=!!strike&&!reduced;if(arc.visible){arc.position.set(s.x+(i?-.7:.7),1.2,.1);arc.rotation.set(.5,i?Math.PI:0,game.time*.022);arc.material.opacity=.6;}});
      const centre=(game.player.x+game.enemy.x)*.5,shake=age<200?this.shake*(1-age/200):0;
      const fit=(game.distance+2)/(2*Math.tan(this.camera.fov*Math.PI/360)*this.camera.aspect);
      this.camera.position.x=centre+(reduced?0:Math.sin(clock*.19)*shake);this.camera.position.y=4.8+(reduced?0:Math.cos(clock*.15)*shake);this.camera.position.z=Math.max(this.mobile?8.8:10.8,fit);this.camera.lookAt(centre,.85,0);
      this.renderer.render(this.scene,this.camera);
    }
  }
  return {View};
})();
