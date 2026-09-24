// Pure rules and wire contract. The UI and the Node evaluation harness use this same code.
const Tetris = (() => {
  const W = 10, H = 20;
  const SHAPES = {
    I: [[[0,0],[1,0],[2,0],[3,0]], [[0,0],[0,1],[0,2],[0,3]]],
    O: [[[0,0],[1,0],[0,1],[1,1]]],
    T: [[[0,0],[1,0],[2,0],[1,1]], [[1,0],[0,1],[1,1],[1,2]], [[0,1],[1,1],[2,1],[1,0]], [[0,0],[0,1],[1,1],[0,2]]],
    J: [[[0,0],[0,1],[1,1],[2,1]], [[0,0],[1,0],[0,1],[0,2]], [[0,0],[1,0],[2,0],[2,1]], [[1,0],[1,1],[0,2],[1,2]]],
    L: [[[2,0],[0,1],[1,1],[2,1]], [[0,0],[0,1],[0,2],[1,2]], [[0,0],[1,0],[2,0],[0,1]], [[0,0],[1,0],[1,1],[1,2]]],
    S: [[[1,0],[2,0],[0,1],[1,1]], [[0,0],[0,1],[1,1],[1,2]]],
    Z: [[[0,0],[1,0],[1,1],[2,1]], [[1,0],[0,1],[1,1],[0,2]]]
  };
  const KINDS = Object.keys(SHAPES);
  const CODES = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  const INSTRUCTIONS = 'Choose most lines cleared; among ties choose fewest holes; then lowest height; then lowest bump. Compare every option in this priority order.';
  const emptyBoard = () => Array.from({length:H}, () => Array(W).fill(0));
  function random(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function collides(grid, shape, x, y) {
    return shape.some(([cx,cy]) => x+cx<0 || x+cx>=W || y+cy<0 || y+cy>=H || grid[y+cy][x+cx]);
  }
  function metrics(grid) {
    let holes = 0;
    const heights = Array.from({length:W}, (_,x) => {
      const first = grid.findIndex(row => row[x]);
      if (first<0) return 0;
      for (let y=first+1;y<H;y++) if (!grid[y][x]) holes++;
      return H-first;
    });
    return {holes, height:Math.max(...heights), bump:heights.slice(1).reduce((s,h,i)=>s+Math.abs(h-heights[i]),0), heights};
  }
  function place(grid, cells, kind) {
    if (cells.length!==4 || new Set(cells.map(([x,y])=>x+y*W)).size!==4 ||
      cells.some(([x,y])=>!Number.isInteger(x)||!Number.isInteger(y)||x<0||x>=W||y<0||y>=H||grid[y][x])) {
      throw new Error('Invalid placement');
    }
    const next = grid.map(row=>row.slice());
    for (const [x,y] of cells) next[y][x]=kind;
    const clearedRows = next.flatMap((row,y)=>row.every(Boolean)?[y]:[]);
    const kept = next.filter(row=>row.some(c=>!c));
    while (kept.length<H) kept.unshift(Array(W).fill(0));
    return {grid:kept, lines:clearedRows.length, clearedRows};
  }
  function landings(grid, kind) {
    const found=[], seen=new Set();
    SHAPES[kind].forEach((shape,rot)=>{
      const width=Math.max(...shape.map(c=>c[0]))+1;
      for (let x=0;x<=W-width;x++) {
        if (collides(grid,shape,x,0)) continue;
        let y=0;
        while (!collides(grid,shape,x,y+1)) y++;
        const cells=shape.map(([cx,cy])=>[cx+x,cy+y]);
        const key=cells.map(([px,py])=>px+py*W).sort((a,b)=>a-b).join(',');
        if (seen.has(key)) continue;
        seen.add(key);
        const after=place(grid,cells,kind);
        found.push({code:CODES[found.length], rot, column:x, y, cells, lines:after.lines, ...metrics(after.grid)});
      }
    });
    return found;
  }
  // Exact lexicographic reference, including ties. This is a policy oracle, not optimal Tetris.
  const compare = (a,b) => b.lines-a.lines || a.holes-b.holes || a.height-b.height || a.bump-b.bump;
  const reference = options => options.reduce((best,o)=>compare(o,best)<0?o:best,options[0]);
  function spawn(grid,kind) {
    const shape=SHAPES[kind][0], x=Math.floor((W-Math.max(...shape.map(c=>c[0]))-1)/2);
    if (collides(grid,shape,x,0)) return null;
    let rows=0;
    while (!collides(grid,shape,x,rows+1)) rows++;
    return {shape,x,rows};
  }
  class Game {
    constructor(seed=1, scenario='empty') {
      this.seed=seed; this.rng=random(seed); this.board=emptyBoard(); this.queue=[];
      this.lines=0; this.pieces=0; this.score=0; this.refill();
      if (scenario==='trench') {
        for(let y=16;y<H;y++) this.board[y]=Array.from({length:W},(_,x)=>x===4?0:'G');
        const i=this.queue.indexOf('I'); [this.queue[0],this.queue[i]]=[this.queue[i],this.queue[0]];
      } else if (scenario==='stairs') {
        for(let x=0;x<W;x++) for(let y=H-Math.floor(x/2);y<H;y++) this.board[y][x]='G';
      }
    }
    refill() {
      while(this.queue.length<8) {
        const bag=KINDS.slice();
        for(let i=bag.length-1;i>0;i--) { const j=Math.floor(this.rng()*(i+1)); [bag[i],bag[j]]=[bag[j],bag[i]]; }
        this.queue.push(...bag);
      }
    }
    get kind() { return this.queue[0]; }
    options() { return landings(this.board,this.kind); }
    commit(option) {
      const result=place(this.board,option.cells,this.kind);
      this.board=result.grid; this.lines+=result.lines; this.pieces++; this.score+=[0,100,300,500,800][result.lines];
      this.queue.shift(); this.refill();
      return result;
    }
  }
  function requestFor(game,options,model,encoding='compact') {
    const criteria=Object.fromEntries(options.map(o=>[o.code,
      'lines '+o.lines+', holes '+o.holes+', height '+o.height+', bump '+o.bump+
      (encoding==='original'?', rotation '+o.rot+', column '+o.column:'')
    ]));
    const state=encoding==='original'
      ? 'board, top row first, # filled . empty:\n'+game.board.map(r=>r.map(c=>c?'#':'.').join('')).join('\n')+'\npiece '+game.kind+'\nnext '+game.queue[1]
      : 'Tetris. Piece '+game.kind+'. Metrics describe the board AFTER each legal landing and line clear.';
    return {model,state,questions:{move:{type:'choice',
      instructions:encoding==='original'?'Choose one legal Tetris landing. If any option clears a line, pick the one that clears the most. Otherwise prefer fewer holes, then a lower stack, then less bumpiness.':INSTRUCTIONS,
      criteria}}};
  }
  const {decodeChoice:decode,decide,withDeadline,summary}=SystemOne;
  return {W,H,SHAPES,KINDS,Game,emptyBoard,collides,metrics,place,landings,compare,reference,spawn,requestFor,decode,decide,withDeadline,summary};
})();
  
