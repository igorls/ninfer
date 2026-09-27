// Continuous physics and observation rendering. This module never chooses an action.
const BeaconGame = (() => {
  const W=640,H=480,R=12,SPEED=140;
  const actions=Object.freeze([
    ['A','Left',-1,0],['B','Right',1,0],['C','Up',0,-1],['D','Down',0,1],
    ['E','Up-left',-1,-1],['F','Up-right',1,-1],['G','Down-left',-1,1],['H','Down-right',1,1],['I','Brake',0,0]
  ].map(([code,label,x,y])=>Object.freeze({code,label,x,y})));
  const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
  function create(seed=1){
    return {seed:seed>>>0,time:0,drone:{x:100,y:240},beacon:{x:520,y:120},
      walls:[{x:280,y:280,w:90,h:100},{x:280,y:60,w:55,h:85}],
      hazard:{x:455,y:370,r:18},collected:0,hits:0,contact:false,command:'I',until:0,trail:[]};
  }
  function blocked(g,x,y){return x<R+8||y<R+8||x>W-R-8||y>H-R-8||g.walls.some(w=>
    Math.hypot(x-clamp(x,w.x,w.x+w.w),y-clamp(y,w.y,w.y+w.h))<R);}
  function command(g,code,duration=220){
    if(!actions.some(a=>a.code===code))throw new Error('Unknown movement command');
    g.command=code;g.until=g.time+duration;
  }
  function brake(g){g.command='I';g.until=g.time;}
  function nextBeacon(g){
    const spots=[[520,400],[100,390],[150,95],[520,120]];
    const point=spots[(g.collected-1)%spots.length];
    g.seed=(Math.imul(g.seed,1664525)+1013904223)>>>0;
    g.beacon={x:point[0]+((g.seed>>>16)%31)-15,y:point[1]+((g.seed>>>8)%21)-10};
  }
  function tick(g,ms){
    // Substeps prevent tunnelling after an uneven animation frame.
    for(let remaining=Math.max(0,ms);remaining>0;){
      const dt=Math.min(remaining,10);remaining-=dt;g.time+=dt;
      g.hazard.x=455+110*Math.sin(g.time/1500);g.hazard.y=370;
      const a=actions.find(a=>a.code===(g.time<=g.until?g.command:'I'));
      const norm=Math.hypot(a.x,a.y)||1,dx=a.x/norm*SPEED*dt/1000,dy=a.y/norm*SPEED*dt/1000;
      let collision=false;
      if(!blocked(g,g.drone.x+dx,g.drone.y))g.drone.x+=dx;else if(dx)collision=true;
      if(!blocked(g,g.drone.x,g.drone.y+dy))g.drone.y+=dy;else if(dy)collision=true;
      if(Math.hypot(g.drone.x-g.hazard.x,g.drone.y-g.hazard.y)<R+g.hazard.r)collision=true;
      if(collision&&!g.contact)g.hits++;g.contact=collision;
      if(Math.hypot(g.drone.x-g.beacon.x,g.drone.y-g.beacon.y)<25){g.collected++;nextBeacon(g);}
    }
  }
  function moveObject(g,id,x,y){
    const target=id==='beacon'?g.beacon:g.walls[Number(id.replace('wall-',''))];
    if(!target)return false;
    const old={x:target.x,y:target.y};
    target.x=clamp(x,25,W-(target.w||25)-15);target.y=clamp(y,25,H-(target.h||25)-15);
    if(blocked(g,g.drone.x,g.drone.y)||g.walls.some(w=>g.beacon.x>w.x-24&&g.beacon.x<w.x+w.w+24&&g.beacon.y>w.y-24&&g.beacon.y<w.y+w.h+24)){
      Object.assign(target,old);return false;
    }return true;
  }
  function hitObject(g,x,y){
    if(Math.hypot(x-g.beacon.x,y-g.beacon.y)<28)return 'beacon';
    const i=g.walls.findIndex(w=>x>=w.x&&x<=w.x+w.w&&y>=w.y&&y<=w.y+w.h);return i<0?null:'wall-'+i;
  }
  // Identical pixels in the browser camera and the live API harness. No labels or coordinates.
  function render(g,width=320,blank=false){
    const height=width*3/4,pixels=new Uint8ClampedArray(width*height*4),s=width/W;
    const bg=[17,25,32],wall=[99,116,124],cyan=[64,224,231],yellow=[255,215,50],red=[255,93,82];
    for(let i=0;i<pixels.length;i+=4){pixels[i]=bg[0];pixels[i+1]=bg[1];pixels[i+2]=bg[2];pixels[i+3]=255;}
    const set=(x,y,c)=>{const i=(y*width+x)*4;pixels[i]=c[0];pixels[i+1]=c[1];pixels[i+2]=c[2];};
    const rect=(x,y,w,h,c)=>{for(let py=Math.max(0,Math.floor(y*s));py<Math.min(height,Math.ceil((y+h)*s));py++)for(let px=Math.max(0,Math.floor(x*s));px<Math.min(width,Math.ceil((x+w)*s));px++)set(px,py,c);};
    const circle=(x,y,r,c,inner=0)=>{for(let py=Math.max(0,Math.floor((y-r)*s));py<Math.min(height,Math.ceil((y+r)*s));py++)for(let px=Math.max(0,Math.floor((x-r)*s));px<Math.min(width,Math.ceil((x+r)*s));px++){const d=Math.hypot(px/s-x,py/s-y);if(d<=r&&d>=inner)set(px,py,c);}};
    if(!blank){
      rect(0,0,W,8,wall);rect(0,H-8,W,8,wall);rect(0,0,8,H,wall);rect(W-8,0,8,H,wall);
      for(const w of g.walls){rect(w.x,w.y,w.w,w.h,wall);rect(w.x+4,w.y+4,w.w-8,3,[161,174,178]);}
      circle(g.beacon.x,g.beacon.y,23,yellow,15);circle(g.beacon.x,g.beacon.y,4,yellow);
      circle(g.hazard.x,g.hazard.y,g.hazard.r,red);rect(g.hazard.x-10,g.hazard.y-2,20,4,bg);
      // A triangular drone always points up; movement directions are screen-relative.
      for(let dy=-18;dy<=14;dy++)rect(g.drone.x-(dy+18)*.55,g.drone.y+dy,(dy+18)*1.1+1,1,cyan);
      circle(g.drone.x,g.drone.y+4,3,bg);
    }
    return {pixels,width,height};
  }
  return {W,H,actions,create,tick,command,brake,render,moveObject,hitObject};
})();
