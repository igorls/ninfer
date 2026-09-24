// Rules are supplied by chess.js; this adapter owns observations, legal actions and a local reference.
const ChessGame = (() => {
  const {Chess, DEFAULT_POSITION} = ChessRules;
  const names={p:'pawn',n:'knight',b:'bishop',r:'rook',q:'queen',k:'king'};
  const value={p:1,n:3,b:3,r:5,q:9,k:0};
  const positions={
    opening:DEFAULT_POSITION,
    mate:'6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1',
    fork:'4k3/pp3ppp/8/3q4/8/2N5/PP3PPP/4K3 w - - 0 1',
    promotion:'7k/P7/8/8/8/8/7p/4K3 w - - 0 1'
  };
  const label=color=>color==='w'?'White':'Black';
  const id=move=>move.from+move.to+(move.promotion||'');
  function describe(move) {
    return move.san+' ('+move.from+'-'+move.to+(move.promotion?'='+move.promotion:'')+
      '); mate '+(move.mate?'yes':'no')+'; gain '+move.materialGain+'; recapture net '+move.materialAfterRecapture;
  }
  class Game {
    constructor(fen=DEFAULT_POSITION) { this.rules=new Chess(fen); this.initialFen=fen; }
    get turn() { return this.rules.turn(); }
    get fen() { return this.rules.fen(); }
    get ply() { return this.rules.history().length; }
    options() {
      if(this.rules.isGameOver())return [];
      const fen=this.fen;
      if(this.cachedFen===fen)return this.cachedOptions;
      const side=this.turn,opponent=side==='w'?'b':'w';
      const options=this.rules.moves({verbose:true}).map(move=>{
        this.rules.move({from:move.from,to:move.to,promotion:move.promotion});
        const attacked=this.rules.isAttacked(move.to,opponent);
        this.rules.undo();
        const mate=move.san.endsWith('#'),check=move.san.endsWith('+');
        const gain=(value[move.captured]||0)+(move.promotion?value[move.promotion]-1:0);
        const materialAfterRecapture=gain-(attacked?(value[move.promotion||move.piece]||2):0);
        const center=3.5-Math.abs(3.5-(move.to.charCodeAt(0)-97))+3.5-Math.abs(3.5-(Number(move.to[1])-1));
        // One-ply illustrative material policy, not a chess engine score or an optimal-play oracle.
        const referenceScore=mate?10000:materialAfterRecapture+center*.02+(check?.03:0);
        return {id:id(move),from:move.from,to:move.to,san:move.san,piece:move.piece,
          promotion:move.promotion||null,captured:move.captured||null,color:side,mate,check,attacked,
          materialGain:gain,materialAfterRecapture,referenceScore};
      });
      this.cachedFen=fen;this.cachedOptions=options;
      return options;
    }
    commit(option) {
      const move=this.rules.move({from:option.from,to:option.to,...(option.promotion?{promotion:option.promotion}:{})});
      if(!move)throw new Error('The selected move is no longer legal.');
      if(this.rules.isGameOver())this.rules.setHeader('Result',this.rules.isCheckmate()?(this.turn==='b'?'1-0':'0-1'):'1/2-1/2');
      return move;
    }
    status() {
      if(this.rules.isCheckmate())return {over:true,text:label(this.turn==='w'?'b':'w')+' wins by checkmate'};
      if(this.rules.isStalemate())return {over:true,text:'Draw · stalemate'};
      if(this.rules.isThreefoldRepetition())return {over:true,text:'Draw · threefold repetition'};
      if(this.rules.isInsufficientMaterial())return {over:true,text:'Draw · insufficient material'};
      if(this.rules.isDrawByFiftyMoves())return {over:true,text:'Draw · fifty-move rule'};
      return {over:false,text:label(this.turn)+' to move'+(this.rules.isCheck()?' · check':'')};
    }
    state() {
      const pieces=this.rules.board().flat().filter(Boolean).map(p=>label(p.color)+' '+names[p.type]+' '+p.square).join(', ');
      return 'Chess. '+this.status().text+'. FEN: '+this.fen+'\nPieces: '+pieces+
        '\nRecent moves: '+this.rules.history().slice(-12).join(' ');
    }
  }
  const reference=options=>options.reduce((a,b)=>b.referenceScore>a.referenceScore?b:a,options[0]);
  const codeOptions=options=>options.map((o,i)=>({...o,code:SystemOne.CODES[i]}));
  function plan(game,options) {
    if(options.length<=62)return {kind:'move',options:codeOptions(options)};
    // All moves are retained. First select a piece, then one legal move of that piece.
    const byPiece=new Map();
    for(const move of options){const group=byPiece.get(move.from)||[];group.push(move);byPiece.set(move.from,group);}
    const groups=[...byPiece].map(([from,moves],i)=>({code:SystemOne.CODES[i],from,moves}));
    if(groups.length>62||groups.some(g=>g.moves.length>62))throw new Error('Chess move grouping exceeds the API choice limit.');
    return {kind:'piece',options:groups};
  }
  function requestFor(game,step,model) {
    const instructions=step.kind==='piece'
      ? 'Choose the piece whose listed legal moves contain the strongest chess move.'
      : 'Choose the strongest legal chess move.';
    const criteria=Object.fromEntries(step.options.map(o=>[o.code,step.kind==='piece'
      ? names[o.moves[0].piece]+' on '+o.from+': '+o.moves.map(describe).join(' | ')
      : describe(o)]));
    return {model,state:game.state(),questions:{move:{type:'choice',instructions:instructions+
      ' Priority: checkmate first. Otherwise maximize net material after immediate recapture, then prefer profitable captures and safe central development. Material is measured in pawns. Recapture is an approximate risk, not a forced continuation. A check alone does not justify losing material. Return the option token.',criteria}}};
  }
  async function choose(game,options,connection,{signal,onStage=()=>{},onResult=()=>{}}={}) {
    let step=plan(game,options);
    const stages=[];
    while(true) {
      signal?.throwIfAborted();
      const request=requestFor(game,step,connection.model);
      onStage({kind:step.kind,options:step.options.length,number:stages.length+1});
      const result=await SystemOne.decide(connection.endpoint,request,step.options,{signal,apiKey:connection.apiKey||''});
      stages.push({kind:step.kind,request,response:result.body,ms:result.ms,options:step.options,choice:result.option.code,choiceWarning:result.choiceWarning});
      onResult(stages[stages.length-1]);
      if(step.kind==='move')return {option:result.option,stages,ms:stages.reduce((sum,s)=>sum+s.ms,0)};
      step={kind:'move',options:codeOptions(result.option.moves)};
    }
  }
  return {Game,positions,names,value,label,describe,id,reference,codeOptions,plan,requestFor,choose};
})();
