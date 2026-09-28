// Same game and client as the browser. Real elapsed time, including every HTTP request.
import vm from 'node:vm';
import {readFile,writeFile} from 'node:fs/promises';
import {setTimeout as sleep} from 'node:timers/promises';
const flags=Object.fromEntries(process.argv.slice(2).map(arg=>{const [key,...value]=arg.replace(/^--/,'').split('=');return [key,value.join('=')];}));
if(flags.help!==undefined){console.log('node tools/bench/arena-eval.mjs --mode=qwen|reference --seed=1 --difficulty=duel --endpoint=http://127.0.0.1:8010 --model=qwen3.8-27b --deadline=300 --out=report.json');process.exit(0);}
const context=vm.createContext({console,performance,fetch,AbortController,AbortSignal,setTimeout,clearTimeout});
for(const file of ['system-one','arena-game','arena-session'])vm.runInContext(await readFile(new URL(`../../docs/arcade/${file}.js`,import.meta.url),'utf8'),context);
const {Session,summary}=vm.runInContext('ArenaSession',context);
const config={mode:flags.mode||'qwen',seed:Number(flags.seed||1),difficulty:flags.difficulty||'duel',deadline:Number(flags.deadline||300),endpoint:flags.endpoint||'http://127.0.0.1:8010',model:flags.model||'qwen3.8-27b'};
if(!['qwen','reference'].includes(config.mode)||!['sparring','duel','blitz'].includes(config.difficulty)||!Number.isFinite(config.deadline)||config.deadline<=0)throw new Error('Invalid mode, difficulty or deadline');
const session=new Session(config);await session.start();
while(session.status==='running'){session.pump();await sleep(8);}
const report=session.export();if(flags.out)await writeFile(flags.out,JSON.stringify(report,null,2));
console.log(JSON.stringify({config,status:report.status,result:report.result,elapsed:report.elapsed,health:report.health,combat:report.stats,requests:summary(report.records),error:session.error},null,2));
if(session.status==='error')process.exitCode=1;
