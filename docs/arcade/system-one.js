// Shared closed-set decision transport for the decision arcade. No game-specific policy lives here.
const SystemOne = (() => {
  const CODES='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
      function decode(body,options,questionId='move') {
        const answer=body?.answers?.[questionId], keys=options.map(o=>o.code);
        if (answer?.type!=='choice' || !keys.includes(answer.choice)) throw new Error('The API returned an unknown action or answer type.');
        const p=answer.probabilities;
        if (!p || Array.isArray(p) || Object.keys(p).length!==keys.length || keys.some(k=>!Object.hasOwn(p,k)))
          throw new Error('The API probabilities do not match the requested options.');
        if (keys.some(k=>typeof p[k]!=='number'||!Number.isFinite(p[k])||p[k]<0||p[k]>1))
          throw new Error('The API returned a probability outside the numeric range 0–1.');
        const probabilityTotal=keys.reduce((s,k)=>s+p[k],0);
        // TypeSafe specifies approximately 1; live Jev can return rounded totals of 0.99.
        // Allow two percentage points of drift, preserve raw values, reject larger errors.
        if (Math.abs(probabilityTotal-1)>.02+1e-8)
          throw new Error('The API probabilities total '+(probabilityTotal*100).toFixed(2)+'%; expected 98–102%.');
        // The explicit legal choice is authoritative. Jev can report a different probability
        // leader; expose that discrepancy without substituting our own move or retrying inference.
        const leader=keys.reduce((a,b)=>p[b]>p[a]?b:a);
        const choiceWarning=p[answer.choice]+1e-8<p[leader]
          ? 'API chose '+answer.choice+' ('+(p[answer.choice]*100).toFixed(1)+'%); '+leader+' is a reported leader ('+(p[leader]*100).toFixed(1)+'%). The returned legal choice is retained.'
          : null;
        if (!Number.isFinite(answer.confidence)||answer.confidence<0||answer.confidence>1) throw new Error('The API returned invalid confidence.');
        if (!Number.isInteger(body.usage?.input_tokens)||body.usage.input_tokens<0||!Number.isInteger(body.usage?.output_tokens)||body.usage.output_tokens<0) throw new Error('Unexpected System One token usage.');
        return {option:options.find(o=>o.code===answer.choice), probabilities:p, probabilityTotal, choiceWarning, confidence:answer.confidence, usage:body.usage};
      }
      async function decide(endpoint,request,options,{signal,apiKey='',fetchImpl=fetch,now=()=>performance.now()}={}) {
        const headers={'Content-Type':'application/json'};
        if (apiKey) headers.Authorization='Bearer '+apiKey;
        const start=now();
        let response,text;
        const fail=(message,kind,body)=>Object.assign(new Error(message),{kind,status:response?.status??null,receipt:{request,response:body??null,ms:now()-start}});
        try {
          response=await fetchImpl(endpoint.replace(/\/+$/,'')+'/v1/systemone',{method:'POST',headers,body:JSON.stringify(request),signal});
          text=await response.text();
        } catch(error) {
          if(signal?.aborted||error.name==='AbortError')throw error;
          throw fail('Could not complete the API request.','network');
        }
        let body;
        try {body=JSON.parse(text);} catch {}
        if (!response.ok) {
          // TypeSafe's FastAPI `detail` is a message, an {error_type, message} object or a list of
          // validation errors; OpenAI-shaped relays use error.message.
          const detail=body?.detail;
          const message=(typeof detail==='string'?detail:null)||detail?.message||detail?.error_type||
            (Array.isArray(detail)&&detail[0]?.msg?detail.map(d=>(d.loc||[]).join('.')+': '+d.msg).join('; '):null)||
            body?.error?.message||body?.message||text.slice(0,200);
          throw fail('HTTP '+response.status+': '+message,'http',body??text);
        }
        if(body===undefined)throw fail('The API response is not valid JSON.','response',text);
        try {
          const decoded=decode(body,options,Object.keys(request.questions)[0]);
          return {...decoded,ms:now()-start,body};
        } catch(error) {throw fail(error.message,'response',body);}
      }
      // A deadline is a real race: expiry aborts transport and a late answer cannot mutate the game.
      async function withDeadline(task,ms,signal) {
        if(ms<=0)return {expired:true};
        const controller=new AbortController();
        let timer;
        const combined=signal?AbortSignal.any([signal,controller.signal]):controller.signal;
        try {
          return await Promise.race([
            task(combined),
            new Promise(resolve=>{timer=setTimeout(()=>{resolve({expired:true});controller.abort();},Math.max(0,ms));})
          ]);
        } finally { clearTimeout(timer); }
      }
      function summary(records) {
        const timed=records.filter(r=>r.source==='live' && Number.isFinite(r.ms));
        const sorted=timed.map(r=>r.ms).sort((a,b)=>a-b);
        const percentile=q=>sorted.length?sorted[Math.max(0,Math.ceil(q*sorted.length)-1)]:null;
        const evaluated=records.filter(r=>r.policyMatch!=null);
        return {samples:timed.length,p50:percentile(.5),p95:percentile(.95),
          matches:evaluated.filter(r=>r.policyMatch).length,evaluated:evaluated.length,
          deadlines:records.filter(r=>r.deadlineMet!=null).length,
          misses:records.filter(r=>r.deadlineMet===false).length};
      }

  return {CODES, decodeChoice:decode, decide, withDeadline, summary};
})();
