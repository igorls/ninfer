#pragma once
#include <string_view>
namespace ninfer::supervisor {
inline constexpr std::string_view kDashboardHtml =
R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark light">
<title>NInfer · Your local engine</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Crect width='16' height='16' rx='3' fill='%231d2124'/%3E%3Ctext x='2.5' y='12.5' font-family='Bahnschrift,sans-serif' font-weight='700' font-size='12' fill='%23e9edef'%3En%3C/text%3E%3Ccircle cx='12' cy='11.2' r='1.6' fill='%234fd98d'/%3E%3C/svg%3E">
<script>(function(){var r=document.documentElement,m='auto';try{m=localStorage.getItem('ninfer:display')||'auto';}catch(e){}r.dataset.display=m;r.dataset.theme=m==='day'||m==='night'?m:(window.matchMedia&&matchMedia('(prefers-color-scheme: light)').matches?'day':'night');})();</script>
<style>:root{
  --panel:#343a3f;--panel-hi:#3d444a;--panel-lo:#1d2124;--seam:#121517;--bezel:#262b2f;--bezel-top:#2e3439;--bezel-low:#1f2326;
  --key:#1e2225;--key-top:#4a5157;--key-hover:#262b2f;--legend:#e9edef;--legend-dim:#aab4ba;
  --glass:#040708;--glass-2:#0a0f12;--rule:rgba(233,240,244,.15);--rule-2:rgba(233,240,244,.075);
  --ink:#f1f4f5;--dim:#a8b2b8;--faint:#8592a0;--off:#76828a;
  --green:#4fd98d;--cyan:#41cdea;--magenta:#ee78de;--amber:#f5b53d;--red:#ff6257;--ice:#98bfe0;
  --amber-glass:#1c1406;--red-glass:#200807;
  --on-panel:#e9edef;--on-panel-dim:#aab4ba;--on-amber:#1b1305;--key-edge:#0f1113;--key-stop:#ffb2ab;
  --red-on-glass:#ffb2ab;--bezel-glint:#4a5157;--well-edge:#1d262b;--well-ink:#c9d3d8;
  --field-edge:#2a343a;--field-edge-hover:#3d4a52;--field-edge-off:#1f272c;--placeholder:#7d8a92;
  --track:#1a2226;--track-edge:#2b353b;--green-share:#2f6b4c;--hover-glass:#0f171b;--selection:#41cdea55;
  --scroll-thumb:#4b555c;--readout-bg:#0d1418ee;--backdrop:#0b0e10b0;--dialog-shadow:#000000aa;
  --log-dim:#94a3ab;--log-info:#d7e0e4;--log-error:#ff9d94;
  --pole-dark:#2a0a08;--sheen-a:rgba(255,255,255,.045);--sheen-b:rgba(255,255,255,.012);
  --tape-body:rgba(128,138,146,.16);--tick:rgba(241,244,245,.55);--axis:rgba(241,244,245,.45);--envelope:rgba(241,244,245,.35);
  --trace-fill:rgba(241,244,245,.07);--gen-fill:rgba(79,217,141,.12);--reserve-label:#ff8a80;
  --du-shadow:0 0 0 1px var(--seam),0 1px 0 1px var(--bezel-glint),inset 0 0 0 1px #000,inset 0 2px 9px rgba(0,0,0,.9);
  --f-panel:Bahnschrift,'DIN Alternate','Barlow Semi Condensed','Arial Narrow',sans-serif;
  --f-text:'Segoe UI Variable Text','Segoe UI',system-ui,sans-serif;
  --f-mono:'Cascadia Mono','Cascadia Code',Consolas,monospace;
  --r-glass:7px;--gap:8px;
  --slew:.38s cubic-bezier(.33,1,.68,1);
  color-scheme:dark;
}
:root[data-theme=day]{
  --panel:#cdd2d6;--panel-hi:#d6dade;--panel-lo:#b7bec3;--seam:#98a1a8;--bezel:#aeb6bc;--bezel-top:#bcc3c8;--bezel-low:#a1aab0;
  --glass:#fbfbfa;--glass-2:#eef1f2;--rule:rgba(18,24,28,.17);--rule-2:rgba(18,24,28,.085);
  --ink:#11171a;--dim:#44515a;--faint:#56636c;--off:#86919a;
  --green:#0a7a43;--cyan:#006f99;--magenta:#9e2189;--amber:#8f5400;--red:#b3241c;--ice:#2b5fa3;
  --amber-glass:#fff1d4;--red-glass:#fde6e3;
  --on-panel:#141a1d;--on-panel-dim:#3d4950;--red-on-glass:#a01d16;--bezel-glint:#e4e8eb;
  --well-edge:#d3d9dd;--well-ink:#1f282d;--field-edge:#b6c0c6;--field-edge-hover:#7f8c94;--field-edge-off:#d6dce0;--placeholder:#69757d;
  --track:#e2e7ea;--track-edge:#c9d1d6;--green-share:#8fcfaf;--hover-glass:#e8eef1;--selection:#006f9940;
  --scroll-thumb:#a7b1b7;--readout-bg:#ffffffee;--backdrop:#3a434980;--dialog-shadow:#2a333a55;
  --log-dim:#56636c;--log-info:#1f282d;--log-error:#b3241c;
  --pole-dark:#ffffff;--sheen-a:transparent;--sheen-b:transparent;
  --tape-body:rgba(18,24,28,.055);--tick:rgba(17,23,26,.62);--axis:rgba(17,23,26,.5);--envelope:rgba(17,23,26,.3);
  --trace-fill:rgba(17,23,26,.06);--gen-fill:rgba(10,122,67,.12);--reserve-label:#b3241c;
  --du-shadow:0 0 0 1px var(--seam),0 1px 0 1px var(--bezel-glint),inset 0 0 0 1px rgba(40,48,54,.3),inset 0 2px 6px rgba(40,48,54,.16);
  color-scheme:light;
}
/* Keys are hardware: the caps and their legends do not change with the display. */
:root[data-theme=day] :is(.key,.fkey,.master){--ink:#f1f4f5;--green:#4fd98d;--cyan:#41cdea;--magenta:#ee78de;--amber:#f5b53d;--red:#ff6257;--off:#76828a;--dim:#a8b2b8}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;display:flex;flex-direction:column;height:100dvh;background:var(--panel);color:var(--ink);font:13px/1.5 var(--f-text);-webkit-font-smoothing:antialiased;overflow:hidden}
button,input,select{font:inherit;color:inherit}
button,a,input,select,summary{-webkit-tap-highlight-color:transparent}
button,a{touch-action:manipulation}
a{color:var(--cyan)}
button{cursor:pointer}
h1,h2,h3,p,dl,dd,ul,ol{margin:0}
ul,ol{padding:0;list-style:none}
::selection{background:var(--selection);color:var(--ink)}
input,select,textarea{caret-color:var(--cyan);accent-color:var(--cyan)}
:focus-visible{outline:2px solid var(--cyan);outline-offset:2px}
*{scrollbar-color:var(--scroll-thumb) transparent;scrollbar-width:thin}
[hidden]{display:none!important}
.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap}
.skip-link{position:fixed;top:-60px;left:12px;z-index:200;padding:10px 16px;background:var(--glass);color:var(--cyan);font:600 13px var(--f-panel)}
.skip-link:focus{top:8px}
.icon-defs{position:absolute;width:0;height:0;overflow:hidden}
.icon{width:14px;height:14px;fill:none;stroke:currentColor;stroke-width:1.4;stroke-linecap:round;stroke-linejoin:round;flex-shrink:0}

/* Panel lettering: DIN (Bahnschrift) in caps for every label, key and annunciation;
   Segoe stays the voice of explanations. */
.fkey,.key,.prompt,.fma-label,.fma-mode,.fma-value,.fma-sub,.fma-copy,.display-head,.display-label,.ro dt,.band-label,.cdu-title,.cdu-label,.cdu-tab,.lsk-prompt,.cfg-label,.traffic th,.cdu-facts dt,.cdu-subhead,.cdu-tag,.cdu-tmpy,.alert-sev,.finding-sev,.mtp-col span,.reserve-scale,.page-head h2,.link-state,.brand-name,.cfg-status,.cdu-unit{font-family:var(--f-panel);font-stretch:87.5%}
.fkey,.key,.prompt,.fma-label,.display-head h2,.display-label,.ro dt,.band-label,.cdu-label,.cdu-tab,.lsk-prompt,.cfg-label,.traffic th,.cdu-facts dt,.cdu-subhead,.cdu-tag,.cdu-tmpy,.alert-sev,.finding-sev,.link-state,.page-head h2,.cdu-title h1{text-transform:uppercase;letter-spacing:.06em}
b,strong{font-weight:600}

/* Glareshield: function keys and the link annunciator. */
.glareshield{display:flex;align-items:center;gap:14px;padding:8px 12px;background:var(--panel-lo);border-bottom:1px solid var(--seam);flex-shrink:0;min-width:0}
.brand{display:flex;align-items:center;gap:7px;text-decoration:none;color:var(--on-panel);flex-shrink:0}
.brand-mark{font:700 24px/1 var(--f-panel);letter-spacing:-.04em;color:var(--on-panel)}
.brand-mark span{color:var(--green)}
.brand-name{font-weight:600;font-size:15px;letter-spacing:.02em}
.fkeys{display:flex;gap:6px;min-width:0;flex:1}
.fkey,.key{position:relative;display:inline-flex;align-items:center;justify-content:center;gap:7px;min-height:32px;padding:0 13px;background:var(--key);border:1px solid var(--key-edge);border-top-color:var(--key-top);border-radius:4px;color:var(--legend);font-size:12.5px;font-weight:600;text-decoration:none;white-space:nowrap;transition:background-color .15s ease-out,color .15s ease-out}
.fkey:hover,.key:hover:not(:disabled){background:var(--key-hover);color:#fff}
.fkey:active,.key:active:not(:disabled){transform:translateY(1px);border-top-color:#353b40}
.key:disabled{color:var(--off);cursor:not-allowed;border-top-color:#2f3438}
.fkey-lamp{position:absolute;left:9px;right:9px;top:3px;height:2px;border-radius:1px;background:#3a4146}
.fkey[aria-current=page]{color:#fff}
.fkey[aria-current=page] .fkey-lamp{background:var(--green)}
.fkey-count{min-width:18px;padding:0 5px;border-radius:3px;background:var(--amber);color:var(--on-amber);font-size:11px;line-height:17px;text-align:center}
.fkey-dirty{padding:0 5px;border-radius:3px;border:1px solid var(--amber);color:var(--amber);font-size:10px;line-height:15px}
.fkey-small{min-height:28px;padding:0 10px;font-size:11.5px}
.fkey-small[aria-pressed=true]{color:var(--cyan);border-color:var(--key-edge);border-top-color:var(--cyan)}
.glare-end{display:flex;align-items:center;gap:12px;margin-left:auto;flex-shrink:0}
.master{display:grid;place-items:center;gap:0;min-width:70px;min-height:32px;padding:2px 8px;border:1px solid var(--key-edge);border-top-color:#3f3524;border-radius:4px;background:#241b0b;color:#6e5726;font:700 11px/1.15 var(--f-panel);font-stretch:87.5%;letter-spacing:.08em;text-transform:uppercase;transition:background-color .12s ease-out,color .12s ease-out}
.master.lit{background:var(--amber);border-top-color:#ffd88a;color:var(--on-amber)}
.master.lit.warning{background:var(--red);border-top-color:#ffb2ab;color:#200807}
.master:active{transform:translateY(1px)}
.link-state{display:flex;align-items:center;gap:7px;font-size:11.5px;color:var(--on-panel-dim)}
.lamp{display:inline-block;width:9px;height:9px;border-radius:2px;background:var(--green);box-shadow:inset 0 0 0 1px #00000055}
.lamp.offline{background:var(--red)}
.lamp.paused{background:var(--amber)}

/* Mode annunciator row: the band that never leaves the screen. */
.fma{position:relative;display:grid;grid-template-columns:minmax(250px,auto) minmax(0,1.3fr) minmax(0,1.45fr) minmax(0,1.1fr);align-items:stretch;margin:var(--gap) var(--gap) 0;background:var(--glass);border-radius:10px;border:4px solid var(--bezel);border-top-color:var(--bezel-top);border-bottom-color:var(--bezel-low);background-clip:padding-box;box-shadow:var(--du-shadow);flex-shrink:0}
.fma-col{display:flex;flex-direction:column;justify-content:center;gap:1px;min-width:0;padding:7px 12px;border-left:1px solid var(--rule)}
.fma-col:first-child{border-left:0}
.fma-label{font-size:11px;color:var(--faint)}
.fma-mode,.fma-value{display:block;width:fit-content;max-width:100%;font-size:16px;line-height:1.2;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:0 3px;margin-left:-3px;border-radius:2px;outline:1px solid transparent;outline-offset:0}
.fma-mode{font-size:18px;text-transform:uppercase;letter-spacing:.04em}
.fma-value{text-transform:uppercase;letter-spacing:.03em}
.fma-armed{font:600 11.5px var(--f-panel);font-stretch:87.5%;color:var(--cyan);text-transform:uppercase;letter-spacing:.05em;white-space:nowrap}
.fma-sub{font-size:12px;color:var(--dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.fma-sub b{font-weight:500;color:var(--ink);font-variant-numeric:tabular-nums}
/* A mode that just changed is boxed in white for ten seconds, as flight-mode
   annunciators do, so a glance catches the change rather than the state. */
.annunciated{outline-color:var(--ink)}
.fma-copy{display:flex;align-items:center;gap:7px;max-width:100%;min-width:0;padding:0;border:0;background:none;color:var(--cyan);font-size:14.5px;text-align:left}
.fma-copy span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.fma-copy:hover span{text-decoration:underline;text-underline-offset:3px}
.fma .prompt{font-size:11.5px}
.fma-engine{flex-direction:row;align-items:center;justify-content:space-between;gap:12px}
.fma-engine-state{display:flex;flex-direction:column;gap:1px;flex-shrink:0;min-width:88px}
.fma-engine .fma-mode{overflow:visible}
.engine-keys{display:flex;align-items:center;gap:6px}
.engine-keys .key{min-height:30px;padding:0 11px}
.key-stop{color:var(--key-stop)}
.key-go{color:var(--green)}

.prompt{display:inline-flex;align-items:baseline;gap:1px;color:var(--cyan);font-weight:600;font-size:12px;text-decoration:none;white-space:nowrap}
.prompt:hover{text-decoration:underline;text-underline-offset:3px}

/* Scratchpad: one line per message, cleared with CLR. */
.scratchpad{display:flex;flex-direction:column;gap:4px;margin:0 var(--gap);flex-shrink:0}
.scratchpad:has(.pad-line:not([hidden])){margin-top:6px}
.pad-line{display:flex;align-items:center;gap:10px;padding:6px 12px;background:var(--glass);border-radius:5px;font-size:12.5px;color:var(--ink);box-shadow:0 0 0 1px var(--seam)}
.pad-line.amber{color:var(--amber);background:var(--amber-glass)}
.pad-line.red,.pad-line.error{color:var(--red-on-glass);background:var(--red-glass)}
)HTML"
R"HTML(.pad-action span{flex:1;min-width:0;overflow-wrap:anywhere}
.key-tiny{min-height:24px;padding:0 8px;font-size:11px}

/* Live band: memory and rates stay in view on every page that is not Monitor. */
.live-band{position:relative;display:flex;align-items:center;gap:6px 20px;flex-wrap:wrap;margin:6px var(--gap) 0;padding:6px 12px;background:var(--glass);border-radius:10px;border:4px solid var(--bezel);border-top-color:var(--bezel-top);border-bottom-color:var(--bezel-low);background-clip:padding-box;box-shadow:var(--du-shadow);flex-shrink:0;font-variant-numeric:tabular-nums}
body.on-monitor .live-band{display:none}
.band-item{display:flex;align-items:baseline;gap:7px;white-space:nowrap}
.band-label{font-size:11px;color:var(--faint)}
.band-item b{font:600 14px var(--f-panel);font-stretch:87.5%}
.band-sub{font-size:12px;color:var(--green)}
.band-bar{align-self:center;width:84px;height:6px;background:var(--track);border-radius:1px;overflow:hidden}
.band-bar span{display:block;height:100%;background:var(--ink);width:0;transition:width var(--slew)}
.band-return{margin-left:auto}

/* Cover glass: anti-glare coating catches a little of the room. A specific
   effect on display glass only, never on panels or keys. */
.display::after,.fma::after,.cdu::after,.live-band::after{content:'';position:absolute;inset:0;border-radius:6px;pointer-events:none;background:linear-gradient(162deg,var(--sheen-a) 0,var(--sheen-b) 28%,transparent 46%)}

/* Workspace */
.workspace{flex:1;min-height:0;overflow:auto;padding:var(--gap);outline:none}
.view-monitor{height:100%;min-height:440px}
.monitor-grid{display:grid;height:100%;gap:var(--gap);grid-template-columns:minmax(0,1fr);grid-template-rows:minmax(232px,1.3fr) minmax(206px,1fr) auto auto}
.display{position:relative;display:flex;flex-direction:column;min-width:0;min-height:0;background:var(--glass);border-radius:10px;border:4px solid var(--bezel);border-top-color:var(--bezel-top);border-bottom-color:var(--bezel-low);background-clip:padding-box;box-shadow:var(--du-shadow);padding:7px 11px 9px}
.display-head{display:flex;align-items:baseline;gap:12px;min-width:0;margin-bottom:6px}
.display-head h2{font-size:12px;font-weight:600;color:var(--ink);flex-shrink:0}
.display-source{min-width:0;font-size:12px;color:var(--dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.display-range{margin-left:auto;font-size:11px;color:var(--faint);white-space:nowrap}
.display-label{font-size:11px;color:var(--faint);white-space:nowrap;flex-shrink:0}
.note{display:none;margin-top:7px;padding-top:7px;border-top:1px solid var(--rule-2);font-size:12px;line-height:1.55;color:var(--dim);max-width:95ch}
body.explain .note{display:block}
body.explain .view-monitor{height:auto}
body.explain .monitor-grid{height:auto;grid-template-rows:auto}
body.explain .display-memory .display-body,body.explain .display-rates .display-body{min-height:220px}
body.explain .display-traffic{max-height:none}
.explain-only{display:none}
body.explain .explain-only{display:block}

.display-body{flex:1;min-height:0;display:grid;grid-template-columns:66px minmax(0,1fr) 156px;gap:10px}
.tape{position:relative;min-height:0}
.tape canvas{position:absolute;inset:0;width:100%;height:100%;display:block}
.trace{position:relative;min-height:0;min-width:0}
.trace canvas{position:absolute;inset:0;width:100%;height:100%;display:block;border-radius:3px}
.trace canvas:focus-visible{outline-offset:-2px}
.trace-empty{position:absolute;left:0;right:0;top:40%;text-align:center;font:600 12px var(--f-panel);font-stretch:87.5%;letter-spacing:.06em;text-transform:uppercase;color:var(--faint);pointer-events:none}
.cursor-readout{display:none;position:absolute;top:2px;left:50px;z-index:2;padding:4px 8px;background:var(--readout-bg);border:1px solid var(--rule);border-radius:3px;font:500 12px/1.45 var(--f-panel);font-stretch:87.5%;color:var(--ink);font-variant-numeric:tabular-nums;pointer-events:none;white-space:nowrap}
.cursor-readout .t{color:var(--faint)}

.readouts{display:grid;align-content:start;gap:7px;font-variant-numeric:tabular-nums}
.ro{min-width:0}
.ro dt{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--faint)}
.ro dd{display:flex;align-items:baseline;gap:6px;flex-wrap:wrap;font-family:var(--f-panel);font-stretch:87.5%}
.ro dd b{font-size:17px;font-weight:600;line-height:1.2}
.ro-major dd b{font-size:24px;line-height:1.1;letter-spacing:-.01em}
.ro-unit{font-size:12px;color:var(--dim)}
.ro-sub{display:block!important;font-family:var(--f-text)!important;font-stretch:normal!important;font-size:12px;color:var(--faint);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ro .prompt{font-size:11.5px}
.readouts{border-left:1px solid var(--rule-2);padding-left:12px;min-height:0;overflow:hidden}
/* Secondary memory readouts are single label-value lines, so the column never outgrows a short display. */
.memory-readouts .ro:not(.ro-major){display:flex;align-items:baseline;justify-content:space-between;gap:8px}
.memory-readouts .ro:not(.ro-major) dd b{font-size:15px}
.ro-pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
.swatch{display:inline-block;width:12px;height:3px;border-radius:1px}
.swatch-gen{background:var(--green)}
.swatch-read{background:var(--ice)}

.gpu-apps{display:flex;align-items:baseline;gap:12px;margin-top:7px;padding-top:6px;border-top:1px solid var(--rule-2);min-width:0}
.gpu-app-list{display:flex;flex-wrap:wrap;gap:2px 16px;min-width:0;font-size:12px;color:var(--dim);font-variant-numeric:tabular-nums}
.gpu-app{white-space:nowrap}
.gpu-app b{font:600 12.5px var(--f-panel);font-stretch:87.5%;color:var(--ink);margin-left:5px}
.gpu-app.own b{color:var(--green)}


.monitor-side{display:contents}
.display-traffic{max-height:31vh;min-height:118px}
.col-app{width:38%}
.traffic-scroll{min-height:0;overflow:auto;margin:0 -12px;padding:0 12px}
.traffic{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
.traffic th{position:sticky;top:0;z-index:1;background:var(--glass);padding:3px 10px 5px 0;border-bottom:1px solid var(--rule);font-size:11px;font-weight:600;color:var(--faint);text-align:left;white-space:nowrap}
.traffic th.num,.traffic td.num{text-align:right}
.traffic th:last-child,.traffic td:last-child{padding-right:0}
.traffic td{padding:6px 10px 6px 0;border-bottom:1px solid var(--rule-2);vertical-align:top}
.traffic tbody tr.client:last-of-type td,.traffic tr.evidence-row td{border-bottom-color:var(--rule-2)}
.traffic .v{display:block;font:600 15px/1.25 var(--f-panel);font-stretch:87.5%;color:var(--ink);white-space:nowrap}
.traffic .s{display:block;font-size:12px;color:var(--faint);white-space:nowrap}
.client-name{display:flex;align-items:center;gap:8px;font:600 14px/1.25 var(--f-panel);font-stretch:87.5%;color:var(--ink);overflow-wrap:anywhere}
.client-counts{font-size:12px;color:var(--faint);font-variant-numeric:tabular-nums}
.client-counts b{font:600 12.5px var(--f-panel);font-stretch:87.5%;color:var(--ink)}
.client-tag{padding:0 5px;border:1px solid var(--rule);border-radius:3px;font-size:11px;font-weight:500;color:var(--dim);white-space:nowrap}
.client-evidence{margin-top:2px;padding:0;border:0;background:none;color:var(--cyan);font:600 11.5px var(--f-panel);font-stretch:87.5%;text-transform:uppercase;letter-spacing:.06em}
.client-evidence:hover{text-decoration:underline;text-underline-offset:3px}
tr.caution .client-name{color:var(--amber)}
tr.caution td,tr.caution-line td{background:var(--amber-glass)}
.caution-line td{padding-top:0;color:var(--amber);font-size:12px}
.caution-line td::before{content:'CAUTION';margin-right:10px;font:600 11px var(--f-panel);font-stretch:87.5%;letter-spacing:.06em}
.evidence-row td{padding:4px 0 10px;font-size:12px;color:var(--dim)}
.evidence-row p{max-width:95ch;margin-top:3px}
.traffic-empty td{padding:12px 0;color:var(--faint);font-size:12.5px}

.alert-list{display:grid;gap:3px}
.alert-list li{display:flex;align-items:baseline;gap:10px;min-width:0;font-size:12.5px}
.alert-sev{flex-shrink:0;width:66px;font-size:11px;font-weight:600}
.alert-title{flex-shrink:0;font-weight:600}
.alert-text{min-width:0;color:var(--dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.alert-list li.alert-quiet,.display-alerts .alert-list li.alert-quiet{display:block;color:var(--dim)}
.alert-quiet b{color:var(--green);font:600 12.5px var(--f-panel);font-stretch:87.5%;text-transform:uppercase;letter-spacing:.05em}

/* Control display unit: every non-live page. */
.view-cdu{display:flex;justify-content:center;min-height:100%;padding:0 34px}
.view-cdu>.cdu{min-height:100%}
.cdu{position:relative;width:100%;max-width:980px;background:var(--glass);border-radius:10px;border:4px solid var(--bezel);border-top-color:var(--bezel-top);border-bottom-color:var(--bezel-low);background-clip:padding-box;box-shadow:var(--du-shadow);padding:12px 18px 18px}
.cdu-wide{max-width:1240px}
.cdu-title{display:grid;grid-template-columns:minmax(0,1fr) auto minmax(0,1fr);grid-template-areas:'left title right';align-items:baseline;gap:12px;padding-bottom:8px;margin-bottom:10px;border-bottom:1px solid var(--rule)}
.cdu-title h1{grid-area:title;text-align:center}
.cdu-title h1{font-size:17px;font-weight:600;line-height:1.2}
.cdu-page{grid-area:right;justify-self:end;font-size:12px;color:var(--faint);white-space:nowrap}
.cdu-file{font-family:var(--f-mono);font-stretch:normal;font-size:11.5px;max-width:45%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.cdu-tmpy{grid-area:left;justify-self:start;padding:1px 7px;border:1px solid var(--amber);border-radius:3px;color:var(--amber);font-size:11.5px;font-weight:600}
.cdu-help{font-size:12.5px;color:var(--dim);max-width:88ch;margin-bottom:12px}
.cdu-small{font-size:12px;color:var(--faint);max-width:90ch}
.cdu-data{font-size:13px;color:var(--ink);max-width:90ch}
.cdu-label{display:block;font-size:11px;font-weight:600;color:var(--faint);margin-bottom:3px}
.cdu-lines{display:grid;gap:0}
.cdu-line{display:grid;gap:4px;padding:10px 0;border-bottom:1px solid var(--rule-2)}
.cdu-line:first-child{padding-top:2px}
.cdu-field-row{display:flex;gap:8px;align-items:center;max-width:620px}
.cdu-input,.cfg-input,.cfg-select{min-height:34px;min-width:0;padding:6px 10px;background:var(--glass-2);border:1px solid var(--field-edge);border-radius:4px;color:var(--cyan);font:500 14px var(--f-panel);font-stretch:87.5%;font-variant-numeric:tabular-nums}
.cdu-input{flex:1}
.cdu-input-mono{font-family:var(--f-mono);font-stretch:normal;font-size:13px}
.cdu-input-num{flex:0 0 96px;text-align:right}
.cdu-unit{font-size:13px;color:var(--dim)}
input[readonly]{color:var(--ink)}
.cdu-input:hover,.cfg-input:hover:not([readonly]),.cfg-select:hover{border-color:var(--field-edge-hover)}
.cdu-input::placeholder,.cfg-input::placeholder{color:var(--placeholder)}
.frontend-list{display:grid;gap:4px;margin-top:2px}
.frontend-list li{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 12px;font-size:13px}
.frontend-list code{font:12px var(--f-mono);color:var(--faint)}
.frontend-list .fe-down{color:var(--amber)}
.cdu-glossary{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:8px 24px;margin-top:12px}
.cdu-glossary div{display:flex;gap:10px;font-size:12px;color:var(--faint)}
.cdu-glossary dt{flex-shrink:0;width:44px;font:600 11px var(--f-panel);font-stretch:87.5%;color:var(--dim);letter-spacing:.06em;text-transform:uppercase;padding-top:1px}

.cdu-health{display:flex;align-items:baseline;flex-wrap:wrap;gap:4px 14px;margin-bottom:10px}
.cdu-health .cdu-data{font:600 14px var(--f-panel);font-stretch:87.5%;text-transform:uppercase;letter-spacing:.05em}
.cdu-tabs{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:12px}
.cdu-tab{display:inline-flex;align-items:center;gap:7px;min-height:30px;padding:0 12px;border:1px solid var(--rule);border-radius:4px;color:var(--cyan);font-size:12px;font-weight:600;text-decoration:none}
.cdu-tab:hover{background:var(--hover-glass)}
)HTML"
R"HTML(.cdu-tab[aria-current=page]{color:var(--ink);border-color:var(--ink)}
.cdu-tab-count{color:var(--faint);font-weight:500}
.cdu-subhead{display:flex;align-items:center;gap:10px;margin:16px 0 4px;font-size:12px;font-weight:600;color:var(--ink)}
.cdu-tag{padding:0 6px;border:1px solid var(--rule);border-radius:3px;font-size:11px;color:var(--dim);font-weight:500}
.cdu-pre{margin-top:6px;padding:10px 12px;background:var(--glass-2);border:1px solid var(--well-edge);border-radius:4px;font:11.5px/1.7 var(--f-mono);color:var(--well-ink);white-space:pre-wrap;overflow-wrap:anywhere;max-height:40vh;overflow:auto}
.cdu-facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:0 28px}
.cdu-facts div{display:grid;grid-template-columns:minmax(150px,.9fr) minmax(0,1.1fr);gap:14px;padding:7px 0;border-bottom:1px solid var(--rule-2);font-size:12.5px}
.cdu-facts dt{font-size:11px;color:var(--faint);padding-top:1px}
.cdu-facts dd{font:500 14px var(--f-panel);font-stretch:87.5%;overflow-wrap:anywhere;font-variant-numeric:tabular-nums}

.launch-plan{margin-top:14px}
.launch-plan dl{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:6px 20px}
.launch-plan dt{font:600 11px var(--f-panel);font-stretch:87.5%;color:var(--faint);text-transform:uppercase;letter-spacing:.06em}
.launch-plan dd{font:500 14px var(--f-panel);font-stretch:87.5%;font-variant-numeric:tabular-nums}
.plan-compromise{grid-column:1/-1;padding:6px 10px;background:var(--amber-glass);color:var(--amber);border-radius:4px;font-size:12.5px}
.launch-plan code{display:block;margin-top:6px;padding:10px 12px;background:var(--glass-2);border:1px solid var(--well-edge);border-radius:4px;font:11.5px/1.7 var(--f-mono);color:var(--well-ink);white-space:pre-wrap;word-break:break-all}

.findings{display:grid}
@media (min-width:1280px){.findings{display:block;columns:2;column-gap:36px}.finding{break-inside:avoid}.finding:first-child{border-top:0}}
.finding{display:grid;grid-template-columns:96px minmax(0,1fr);gap:2px 14px;padding:9px 0;border-bottom:1px solid var(--rule-2)}
.finding:first-child{border-top:1px solid var(--rule-2)}
.finding-sev{grid-row:1/span 4;font-size:11.5px;font-weight:600;padding-top:1px}
.finding-sev.critical{color:var(--red)}.finding-sev.warning{color:var(--amber)}.finding-sev.notice{color:var(--ink)}.finding-sev.info{color:var(--dim)}
.finding-title{font-weight:600;font-size:13.5px}
.finding-body{color:var(--dim);font-size:12.5px;max-width:95ch;overflow-wrap:anywhere}
.finding-action{color:var(--ink);font-size:12.5px;max-width:95ch}
.finding details summary{width:fit-content;color:var(--cyan);font:600 11.5px var(--f-panel);font-stretch:87.5%;text-transform:uppercase;letter-spacing:.06em;cursor:pointer;list-style:none;padding:2px 0}
.finding details summary::-webkit-details-marker{display:none}
.finding details summary::after{content:'>';margin-left:2px}
.finding details[open] summary::after{content:'<'}
.finding details p{color:var(--dim);font-size:12.5px;margin-top:4px;max-width:95ch}
.finding .cdu-pre{max-height:32vh}
.finding.compact{grid-template-columns:96px minmax(0,auto) minmax(0,1fr) auto;align-items:baseline;padding:6px 0}
.finding.compact .finding-sev{grid-row:auto}
.finding.compact .finding-title{font-size:13px;white-space:nowrap}
.finding.compact .finding-body{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.finding.compact.open .finding-body{white-space:normal;grid-column:2/-1}
.finding.compact.open details{grid-column:2/-1}

.diag-log{display:flex;flex-direction:column}
.view-cdu:has(:is(.diag-log,.diag-experts):not([hidden])){height:100%}
.view-cdu:has(:is(.diag-log,.diag-experts):not([hidden])) .cdu{display:flex;flex-direction:column;height:100%;min-height:420px}
.diag-log:not([hidden]){flex:1;min-height:0}
.log-toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin-bottom:8px}
.log-filters{display:flex;gap:4px;flex-wrap:wrap}
.log-toolbar .cdu-input{flex:1 1 160px;min-height:32px;font-size:13px}
.key-toggle{min-height:30px;padding:0 11px;font-size:12px}
.key-toggle.active{color:var(--cyan);border-top-color:var(--cyan)}
.log-pane{flex:1;min-height:240px;overflow:auto;padding:8px 10px;background:var(--glass-2);border:1px solid var(--well-edge);border-radius:4px;font:11.5px/1.75 var(--f-mono);color:var(--well-ink);white-space:pre-wrap;overflow-wrap:anywhere}
.log-line-dim{color:var(--log-dim)}.log-line-info{color:var(--log-info)}.log-line-req{color:var(--green)}.log-line-warn{color:var(--amber)}.log-line-error{color:var(--log-error)}

.mtp-bars{display:flex;gap:14px;max-width:420px;margin-top:8px}
.mtp-col{flex:1;text-align:center;font-size:11px;color:var(--faint)}
.mtp-track{position:relative;height:72px;margin-bottom:4px;background:var(--track);border-bottom:1px solid var(--rule)}
.mtp-bar{position:absolute;inset:0;background:var(--green);transform:scaleY(0);transform-origin:bottom;transition:transform var(--slew)}

/* Expert use: one row per text layer, one column per expert, on display glass. The map is
   the generation channel (green) for decode and the prompt channel (ice) for prefill, drawn
   from the root tokens; the bars at its right are measured (white); the budget is a target
   (magenta). One screen; the map takes the height the rest leaves. */
.view-cdu:has(.diag-experts:not([hidden])) .cdu{min-height:470px}
.diag-experts:not([hidden]){flex:1;min-height:0;display:flex;flex-direction:column}
.ex-live{flex:1;min-height:0;display:flex;flex-direction:column;gap:8px}
.ex-bar{display:flex;flex-wrap:wrap;align-items:center;gap:6px 12px}
.ex-tabs{display:flex;gap:4px}
.ex-bar .cdu-tab{min-height:26px;padding:0 10px;font-size:11.5px}
.ex-scale{display:flex;align-items:center;gap:6px;font:600 10.5px var(--f-panel);font-stretch:87.5%;letter-spacing:.06em;text-transform:uppercase;color:var(--faint);white-space:nowrap}
.ex-scale canvas{display:block;width:96px;height:10px;border-radius:1px;box-shadow:0 0 0 1px var(--rule-2)}
.ex-read{display:flex;align-items:baseline;flex-wrap:wrap;gap:0 10px;flex:1 1 300px;max-width:640px;min-width:0;margin-left:auto;min-height:26px;padding:3px 9px;border:1px solid var(--rule-2);border-radius:3px;font:500 12.5px/1.4 var(--f-panel);font-stretch:87.5%;font-variant-numeric:tabular-nums;color:var(--ink)}
.ex-read.live{border-color:var(--rule)}
.ex-read i{font-style:normal;font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--faint);margin-right:4px}
.ex-hint{font:12px var(--f-text);font-stretch:normal;color:var(--faint)}
.ex-facts{display:flex;flex-wrap:wrap;align-items:baseline;gap:2px 22px;font-variant-numeric:tabular-nums}
.ex-facts>div{display:flex;align-items:baseline;gap:7px;min-width:0}
.ex-facts dt{font:600 11px var(--f-panel);font-stretch:87.5%;letter-spacing:.06em;text-transform:uppercase;color:var(--faint)}
.ex-facts dd{font:500 13.5px var(--f-panel);font-stretch:87.5%;color:var(--ink);overflow-wrap:anywhere}
.ex-file{font:11.5px var(--f-mono);font-stretch:normal;color:var(--dim);overflow-wrap:anywhere}
.ex-facts .ex-bad{color:var(--amber)}
.ex-tag{padding:1px 7px;border:1px solid var(--amber);border-radius:3px;color:var(--amber);font:600 11.5px var(--f-panel);font-stretch:87.5%;letter-spacing:.06em;text-transform:uppercase}
.ex-map{position:relative;flex:1;min-height:150px}
.ex-map canvas{position:absolute;inset:0;width:100%;height:100%;display:block;border-radius:3px;transition:opacity var(--slew)}
.ex-map canvas:focus-visible{outline-offset:-2px}
.ex-map.stale canvas{opacity:.4}
.ex-skew{display:flex;flex-wrap:wrap;align-items:baseline;gap:2px 16px;font-variant-numeric:tabular-nums}
.ex-skew i,.ex-lg{font:600 11px var(--f-panel);font-stretch:87.5%;font-style:normal;letter-spacing:.06em;text-transform:uppercase;color:var(--faint)}
.ex-skew i{margin-right:6px}
.ex-skew b{font:600 15px var(--f-panel);font-stretch:87.5%;color:var(--ink)}
.ex-sub{font-size:12px;color:var(--faint)}
.ex-budget{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(0,1fr);gap:6px 28px;align-items:end;padding-top:8px;border-top:1px solid var(--rule-2)}
.ex-budget-set{min-width:0}
.ex-budget-row{display:flex;align-items:center;gap:6px 8px;flex-wrap:wrap}
.ex-budget .cdu-input{min-height:30px;padding:3px 8px;color:var(--magenta)}
.ex-slider{flex:1 1 150px;width:auto;min-width:120px}
.ex-budget-of{margin-left:10px;font:400 12px var(--f-text);letter-spacing:0;text-transform:none;color:var(--faint)}
.ex-budget-note{grid-column:1/-1;max-width:none}
.ex-out{display:flex;flex-wrap:wrap;gap:4px 26px}
.ex-out .ro dd b{font-size:22px;line-height:1.1}
.ex-out .ro dt{font-size:11px}
.ex-out .ro dt b{font:600 11px var(--f-panel);font-stretch:87.5%;color:var(--magenta)}
.ex-state{max-width:76ch;padding:6px 2px}
.ex-state .cdu-subhead{margin-top:6px}
.ex-state code{font:12px var(--f-mono);color:var(--well-ink)}
body.explain .view-cdu:has(.diag-experts:not([hidden])){height:auto}
body.explain .view-cdu:has(.diag-experts:not([hidden])) .cdu{height:auto}
body.explain .diag-experts:not([hidden]){flex:none}
body.explain .ex-live{flex:none}
body.explain .ex-map{flex:none;height:clamp(300px,52vh,460px)}
@media (max-height:780px) and (min-width:821px){
  .ex-live{gap:5px}
  .ex-budget{padding-top:5px;gap:3px 28px}
  .ex-budget .cdu-input{min-height:26px}
  .ex-out .ro dd b{font-size:19px}
  .ex-out .ro dt{font-size:10.5px}
  .ex-bar .cdu-tab{min-height:24px}
}
@media (max-width:820px){
  .ex-budget{grid-template-columns:minmax(0,1fr)}
  .ex-read{flex-basis:100%;max-width:none;margin-left:0}
}

/* API test bench: endpoint index beside the request and its answer; one screen. */
.view-bench{height:100%}
.view-bench .cdu{display:flex;flex-direction:column;height:100%;min-height:520px}
.bench-panes{flex:1;min-height:0;display:grid;grid-template-columns:minmax(220px,280px) minmax(0,1fr);gap:16px}
.bench-index{display:flex;flex-direction:column;gap:6px;min-height:0;padding-right:14px;border-right:1px solid var(--rule-2)}
.bench-index .cdu-input{flex:none;min-height:30px;font-size:13px}
.bench-list{flex:1;min-height:0;overflow:auto;margin:0 -6px;padding:0 6px}
.bench-group{margin:8px 0 2px;font:600 11px var(--f-panel);font-stretch:87.5%;letter-spacing:.06em;text-transform:uppercase;color:var(--faint)}
.bench-ep{display:grid;grid-template-columns:46px minmax(0,1fr);gap:0 8px;width:100%;padding:4px 6px;border:0;border-radius:3px;background:none;color:var(--ink);text-align:left;cursor:pointer}
.bench-ep:hover{background:var(--hover-glass)}
.bench-ep[aria-current=true]{background:var(--hover-glass);box-shadow:inset 0 0 0 1px var(--cyan)}
.bench-m{grid-row:1/span 2;font:600 11px var(--f-panel);font-stretch:87.5%;letter-spacing:.05em;color:var(--dim);padding-top:2px}
.bench-p{font:12px var(--f-mono);color:var(--cyan);overflow-wrap:anywhere}
.bench-t{font-size:12px;color:var(--faint)}
.bench-history-head{display:flex;align-items:baseline;justify-content:space-between;margin-top:6px;padding-top:6px;border-top:1px solid var(--rule-2)}
.bench-history{max-height:24vh;overflow:auto;display:grid;gap:1px}
.bench-hist{display:grid;grid-template-columns:34px minmax(0,1fr) auto;gap:6px;width:100%;padding:3px 4px;border:0;border-radius:3px;background:none;color:var(--ink);font:12px var(--f-mono);text-align:left;cursor:pointer}
.bench-hist:hover{background:var(--hover-glass)}
.bench-hist span:nth-child(2){overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bench-work{display:flex;flex-direction:column;gap:8px;min-width:0;min-height:0}
.bench-request{display:flex;flex-direction:column;gap:6px}
.bench-line{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.bench-method{min-width:62px;padding:6px 8px;border:1px solid var(--rule);border-radius:4px;text-align:center;font:600 13px var(--f-panel);font-stretch:87.5%;letter-spacing:.05em}
.bench-line #bench-path{flex:1 1 280px}
.bench-options{gap:6px 10px}
.bench-examples{display:flex;flex-wrap:wrap;gap:4px;margin-right:auto}
)HTML"
R"HTML(button.cdu-tab{background:none;cursor:pointer;font-family:var(--f-panel);font-stretch:87.5%;text-transform:uppercase;letter-spacing:.06em}
button.cdu-tab[aria-pressed=true]{color:var(--ink);border-color:var(--ink)}
.bench-stream{min-height:28px;font-size:13px}
.bench-body{width:100%;height:clamp(96px,22vh,280px);resize:vertical;padding:8px 10px;background:var(--glass-2);border:1px solid var(--field-edge);border-radius:4px;color:var(--ink);font:12.5px/1.55 var(--f-mono);tab-size:2;white-space:pre}
.bench-body:focus-visible{outline-offset:0}
.bench-body.invalid{border-color:var(--red)}
.bench-body-state{font-size:12px;color:var(--faint)}
.bench-response{flex:1;min-height:0;display:flex;flex-direction:column;padding-top:8px;border-top:1px solid var(--rule)}
.bench-status{display:flex;align-items:baseline;gap:6px 14px;flex-wrap:wrap;margin-bottom:6px}
.bench-code{font:600 16px var(--f-panel);font-stretch:87.5%;letter-spacing:.04em;text-transform:uppercase}
.bench-timing{min-width:0;font-size:12px;color:var(--dim);font-variant-numeric:tabular-nums}
.bench-tabs{display:flex;gap:4px;margin-left:auto}
.bench-out{flex:1;min-height:110px;overflow:auto;padding:8px 10px;background:var(--glass-2);border:1px solid var(--well-edge);border-radius:4px;color:var(--well-ink);font:12px/1.65 var(--f-mono);white-space:pre-wrap;overflow-wrap:anywhere}
.bench-out .j-k{color:var(--faint)}.bench-out .j-s{color:var(--ink)}.bench-out .j-n{color:var(--ink);font-weight:600}.bench-out .j-l{color:var(--dim)}
.bench-label{display:block;margin-bottom:2px;font:600 11px var(--f-panel);font-stretch:87.5%;letter-spacing:.06em;text-transform:uppercase;color:var(--faint)}
.bench-reasoning{color:var(--dim)}
.bench-empty{color:var(--faint);font-family:var(--f-text)}
.bench-ev{display:grid;grid-template-columns:62px minmax(64px,120px) minmax(0,1fr);gap:8px;padding:1px 0;border-bottom:1px solid var(--rule-2)}
.bench-ev>span:last-child{overflow-wrap:break-word;word-break:normal}
.bench-ev-t{color:var(--faint);text-align:right}
.bench-ev-n{color:var(--dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

/* Chat: a message log in the datalink manner. Your messages ride the prompt channel (ice),
   NInfer's replies the generation channel (green); what a reply measured is white and what
   it was sent is magenta, the colour of a configured target. One screen; the log scrolls. */
.view-chat{height:100%;padding:0}
.view-chat .cdu{display:flex;flex-direction:column;height:100%;min-height:440px;max-width:none}
.chat-bar{display:flex;align-items:center;gap:8px 14px;flex-wrap:wrap;padding-bottom:8px;border-bottom:1px solid var(--rule)}
.chat-head-id{display:flex;align-items:baseline;gap:10px;padding-right:14px;border-right:1px solid var(--rule-2);min-width:0}
.chat-title{font:600 16px/1.2 var(--f-panel);font-stretch:87.5%;letter-spacing:.06em;text-transform:uppercase;color:var(--ink)}
.chat-model-badge{font:11.5px var(--f-mono);color:var(--faint);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:30ch}
.chat-context{display:flex;align-items:center;gap:8px;flex:1 1 200px;min-width:0}
.chat-context .cdu-label,.chat-think .cdu-label{margin:0}
.chat-meter{flex:1;max-width:220px;height:6px;border-radius:1px;background:var(--track);box-shadow:inset 0 0 0 1px var(--track-edge);overflow:hidden}
.chat-meter span{display:block;width:0;height:100%;background:var(--ink);transition:width var(--slew)}
.chat-meter .amber-fill{background:var(--amber)}
.chat-context-text{font:500 13px var(--f-panel);font-stretch:87.5%;font-variant-numeric:tabular-nums;white-space:nowrap}
.chat-think{display:flex;align-items:center;gap:4px}
.chat-think .cdu-label{margin-right:4px}
.chat-think .cdu-tab{min-height:26px;padding:0 9px;font-size:11.5px}
.chat-param-count{color:var(--magenta)}
.chat-budget-now{display:flex;align-items:baseline;gap:7px;white-space:nowrap}
.chat-budget-now .cdu-label{margin:0}
#chat-budget-now{font:500 13px var(--f-panel);font-stretch:87.5%;font-variant-numeric:tabular-nums;color:var(--magenta)}
#chat-budget-now.engine{color:var(--ink)}
.chat-panes{flex:1;min-height:0;display:grid;grid-template-columns:minmax(0,1fr) clamp(350px,30vw,430px);gap:18px}
.chat-main{display:flex;flex-direction:column;min-width:0;min-height:0}
.chat-side{display:flex;flex-direction:column;gap:10px;min-width:0;min-height:0;padding-top:10px;padding-left:16px;border-left:1px solid var(--rule)}
.chat-side-head{display:flex;align-items:baseline;justify-content:space-between;gap:8px}
.chat-side-head .client-evidence{margin-top:0}
.chat-side-title{font:600 12px/1.2 var(--f-panel);font-stretch:87.5%;letter-spacing:.06em;text-transform:uppercase;color:var(--ink)}
.chat-side-meta{font:500 11.5px var(--f-panel);font-stretch:87.5%;font-variant-numeric:tabular-nums;color:var(--faint)}
.chat-inspector{flex:0 0 auto;display:flex;flex-direction:column;gap:6px;padding-bottom:10px;border-bottom:1px solid var(--rule)}
.chat-inspect-body{padding:8px 10px;background:var(--glass-2);border:1px solid var(--well-edge);border-radius:4px}
.chat-inspect-body .msg-telemetry{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px 14px}
.chat-settings{position:relative;flex:1 1 auto;min-height:0;overflow:auto;display:flex;flex-direction:column;gap:12px;padding-right:4px}
.chat-settings>*{flex-shrink:0}
.chat-settings-note{font-size:12px;color:var(--faint)}
.chat-back{display:none;align-self: flex-start;padding:0;border:0;background:none;font-size:12.5px}
.chat-group{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px 10px;align-content:start;min-width:0;margin:0;padding:0;border:0}
.chat-group legend{padding:0 0 5px;font:600 11.5px var(--f-panel);font-stretch:87.5%;letter-spacing:.06em;text-transform:uppercase;color:var(--dim)}
.chat-group-wide{grid-column:1/-1}
.chat-setting{display:flex;flex-direction:column;min-width:0}
.chat-setting-wide{grid-column:1/-1}
.chat-setting .cdu-input,.chat-setting .cfg-select{flex:none;min-height:30px;padding:4px 8px;font-size:13.5px}
.chat-settings .cfg-check{flex-direction:row;min-height:28px;font-size:13px}
.chat-settings .cfg-help{margin-top:0}
.chat-settings :is(.cdu-input:not(:placeholder-shown),.cfg-select.set){color:var(--magenta)}
.chat-settings :disabled{opacity:.55;cursor:not-allowed}
.chat-budget{display:flex;gap:6px}
.chat-budget .cfg-select{flex:1;min-width:0}
.chat-text{resize:vertical;min-height:52px;font:13px/1.45 var(--f-text)}
.chat-code{font:12px/1.5 var(--f-mono)}
.chat-settings .invalid{border-color:var(--red)}
.chat-settings .cfg-help.red{color:var(--red)}
.chat-settings-keys{flex-direction:row}
.chat-log{position:relative;flex:1;min-height:150px;overflow:auto;display:flex;flex-direction:column;gap:0;padding:12px 6px 12px 0}
.chat-log>*{flex-shrink:0}
.chat-empty{margin:auto;max-width:58ch;text-align:center;font-size:13px;color:var(--dim)}
.chat-empty-title{margin-bottom:6px;font:600 15px var(--f-panel);font-stretch:87.5%;letter-spacing:.06em;text-transform:uppercase;color:var(--ink)}
.msg+.msg{margin-top:18px}
.msg-user+.msg-model{margin-top:8px}
.msg-user{padding:8px 12px 10px;background:var(--glass-2);border:1px solid var(--well-edge);border-left-color:var(--ice);border-radius:4px}
.msg-model{padding-left:2px}
.msg-model+.msg-model{padding-top:12px;border-top:1px solid var(--rule-2)}
.msg-dir{font-size:8px;letter-spacing:0;transform:translateY(-1px)}
.msg-user .msg-dir{color:var(--ice)}
.msg-model .msg-dir{color:var(--green)}
.msg-error .msg-dir{color:var(--red)}
.msg-head{display:flex;align-items:baseline;flex-wrap:wrap;gap:2px 10px;margin-bottom:4px;font:600 11.5px var(--f-panel);font-stretch:87.5%;letter-spacing:.06em;text-transform:uppercase;color:var(--faint)}
.msg-head .client-evidence{margin-left:auto;margin-top:0}
.msg-who{color:var(--ice)}
.msg-head time,.msg-live{font-variant-numeric:tabular-nums}
.msg-model-name{font:11.5px var(--f-mono);letter-spacing:0;text-transform:none}
.msg-state{color:var(--dim)}
.msg-live{color:var(--green)}
.msg-body,.msg-reasoning,.msg-images{max-width:100ch}
.msg-body{font:14px/1.6 var(--f-text);color:var(--ink);overflow-wrap:anywhere}
/* Markdown inside a message (.md): blocks are separated by one small gap, never by blank lines.
   Gaps are in em so they follow the text size of the body (14px) and the reasoning (13px). */
.md>*{margin:0}
.md>*+*{margin-top:.5em}
.md-user p{white-space:pre-wrap}
.md strong{font-weight:700}
.md del{color:var(--dim);text-decoration-thickness:1px}
.md a{color:var(--cyan);text-decoration-thickness:1px;text-underline-offset:2px}
.md code{font-family:var(--f-mono);font-size:.89em;padding:.05em .3em;border-radius:3px;background:var(--glass-2);box-shadow:inset 0 0 0 1px var(--well-edge)}
.md-h{font-weight:650;line-height:1.3;color:var(--ink)}
.md>.md-h{margin-top:.85em}
.md>.md-h:first-child{margin-top:0}
.md>.md-h+*{margin-top:.3em}
.md>.md-h+.md-h{margin-top:.5em}
.md-h1{font-size:1.4em}
.md-h2{font-size:1.2em}
.md-h3{font-size:1.07em}
.md-h5,.md-h6{font-size:.93em}
.md-h6{color:var(--dim)}
.md ul,.md ol{padding-left:1.45em}
.md ul{list-style:disc}
.md ul ul{list-style:circle}
.md ul ul ul{list-style:square}
.md ol{list-style:decimal}
.md li::marker{color:var(--faint)}
.md ol>li::marker{font-variant-numeric:tabular-nums}
.md li+li{margin-top:.2em}
.md li>*+*{margin-top:.3em}
.md li>ul,.md li>ol{margin-top:.2em}
.md li.md-task-item{position:relative;list-style:none}
.md-task{position:absolute;left:-1.45em;top:.3em;width:.95em;height:.95em;border:1.5px solid var(--faint);border-radius:2px}
.md-task.done::after{content:"";position:absolute;left:.25em;top:.02em;width:.28em;height:.55em;border:solid var(--ink);border-width:0 1.6px 1.6px 0;transform:rotate(40deg)}
:where(.md) blockquote{margin:0}
.md blockquote{padding-left:11px;border-left:2px solid var(--faint);color:var(--dim)}
.md blockquote>*+*{margin-top:.5em}
.md hr{height:0;margin:.8em 0;border:0;border-top:1px solid var(--rule)}
.md-code{background:var(--glass-2);border:1px solid var(--well-edge);border-radius:4px;color:var(--well-ink)}
.md-code-head{display:flex;align-items:center;justify-content:space-between;gap:8px;min-height:20px;padding:2px 8px 0 10px}
.md-code-lang{font:600 10.5px/1.2 var(--f-panel);font-stretch:87.5%;letter-spacing:.06em;text-transform:uppercase;color:var(--faint)}
.md-copy{flex:none;width:46px;margin-left:auto;padding:0;border:0;background:none;color:var(--cyan);font:600 10.5px/1.2 var(--f-panel);font-stretch:87.5%;letter-spacing:.06em;text-align:right;text-transform:uppercase;white-space:nowrap;user-select:none}
.md-copy:hover{text-decoration:underline;text-underline-offset:3px}
.md-copy.failed{color:var(--amber)}
.md-code pre{margin:0;padding:3px 10px 8px;overflow-x:auto;font:12.5px/1.5 var(--f-mono);white-space:pre;overflow-wrap:normal;tab-size:4}
.md-code code{padding:0;border-radius:0;background:none;box-shadow:none;font:inherit}
.md-table{max-width:100%;overflow-x:auto}
.md-table table{border-collapse:collapse;font-size:13px;line-height:1.45;font-variant-numeric:tabular-nums;overflow-wrap:normal;word-break:normal}
.md-table th,.md-table td{padding:3px 14px 3px 0;border-bottom:1px solid var(--rule-2);text-align:left;vertical-align:top}
.md-table th{font-weight:650;border-bottom-color:var(--rule)}
.md-table :is(th,td):last-child{padding-right:2px}
.md-table .md-al-r{text-align:right}
.md-table .md-al-c{text-align:center}
.msg-images{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:6px}
.msg-images img{max-width:220px;max-height:140px;border:1px solid var(--rule);border-radius:3px}
.msg-reasoning{margin:2px 0 6px;color:var(--dim)}
.msg-reasoning summary{width:fit-content;cursor:pointer;font:600 11.5px var(--f-panel);font-stretch:87.5%;letter-spacing:.06em;text-transform:uppercase;color:var(--cyan)}
.msg-reasoning-body{max-height:40vh;overflow:auto;margin-top:4px;padding-left:10px;border-left:1px dotted var(--rule);font-size:13px;line-height:1.55;overflow-wrap:anywhere}
.msg-progress{display:flex;align-items:center;gap:10px;margin:2px 0 4px;font-size:12px;color:var(--ice);font-variant-numeric:tabular-nums}
.msg-bar{flex:none;width:160px;height:5px;background:var(--track);box-shadow:inset 0 0 0 1px var(--track-edge)}
.msg-bar span{display:block;height:100%;background:var(--ice);transition:width .25s ease-out}
.msg-foot{display:flex;align-items:baseline;flex-wrap:wrap;gap:4px 14px;margin-top:6px;padding-top:5px;border-top:1px solid var(--rule-2)}
.msg-vitals{display:flex;align-items:baseline;flex-wrap:wrap;gap:2px 10px;font:500 12px/1.4 var(--f-panel);font-stretch:87.5%;font-variant-numeric:tabular-nums;color:var(--faint)}
.vital-v{font-weight:600;color:var(--dim)}
.vital-v.amber{color:var(--amber)}
.vital-sep{color:var(--off)}
.msg-more{margin-top:0}
.msg-more[aria-pressed=true]{color:var(--ink);text-decoration:underline;text-underline-offset:3px}
.msg-sent-tag{color:var(--magenta)}
.msg-actions{display:flex;align-items:baseline;gap:14px;margin-left:auto}
.msg-actions .client-evidence{margin-top:0}
.msg-details{margin-top:6px;padding:6px 10px 8px;background:var(--glass-2);border:1px solid var(--well-edge);border-radius:4px}
.msg-telemetry{display:flex;flex-wrap:wrap;gap:4px 18px}
.msg-telemetry div{display:flex;flex-direction:column;min-width:0}
.msg-telemetry dt{font:600 10.5px var(--f-panel);font-stretch:87.5%;letter-spacing:.06em;text-transform:uppercase;color:var(--faint)}
.msg-telemetry dd{font:500 13px var(--f-panel);font-stretch:87.5%;font-variant-numeric:tabular-nums;color:var(--ink)}
.msg-telemetry dd.amber{color:var(--amber)}
.msg-sent{margin-top:5px;padding-top:5px;border-top:1px solid var(--rule-2);font:12px/1.5 var(--f-mono);color:var(--magenta);overflow-wrap:anywhere}
.msg-details>.msg-sent:first-child,.chat-inspect-body>.msg-sent:first-child{margin-top:0;padding-top:0;border-top:0}
.msg-sent .cdu-label{display:inline;margin:0 6px 0 0}
.msg-note{margin-top:4px;font-size:12.5px;color:var(--amber)}
.msg-tokens{font:13px/1.8 var(--f-mono);white-space:pre-wrap;overflow-wrap:anywhere}
.tok{border-radius:2px;cursor:default}
.tok:is(:hover,:focus-visible){outline:1px solid var(--cyan);outline-offset:0}
.tok-mid{box-shadow:inset 0 -2px 0 var(--amber)}
.tok-low{background:var(--amber-glass);box-shadow:inset 0 -2px 0 var(--amber);color:var(--amber)}
.tok-forced{box-shadow:inset 0 -1px 0 var(--magenta);color:var(--magenta)}
.tok-special{color:var(--faint);font-size:11.5px}
.msg-token-bar{display:flex;align-items:baseline;justify-content:space-between;flex-wrap:wrap;gap:4px 14px;margin-top:6px;padding:4px 8px;background:var(--glass-2);border:1px solid var(--well-edge);border-radius:3px;font-size:12px}
.msg-token-readout{font:500 12.5px/1.4 var(--f-panel);font-stretch:87.5%;font-variant-numeric:tabular-nums;color:var(--ink);min-width:0;overflow-wrap:anywhere}
.msg-token-readout code{font:12px var(--f-mono);color:var(--cyan)}
.msg-token-readout i{font-style:normal;font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--faint);margin:0 4px}
.msg-token-key{font-size:11.5px;color:var(--faint);white-space:nowrap}
.chat-composer{display:flex;flex-direction:column;gap:6px;padding-top:8px;border-top:1px solid var(--rule)}
)HTML"
R"HTML(.chat-input{width:100%;min-height:58px;max-height:30vh;resize:vertical;padding:8px 10px;background:var(--glass-2);border:1px solid var(--field-edge);border-radius:4px;color:var(--ink);font:14px/1.5 var(--f-text)}
.chat-input:hover{border-color:var(--field-edge-hover)}
.chat-input::placeholder{color:var(--placeholder)}
.chat-composer.drop .chat-input{border-color:var(--cyan)}
.chat-keys{display:flex;flex-wrap:wrap;align-items:center;gap:8px}
.chat-keys .cfg-status{flex:1}
.chat-attachments{display:flex;flex-wrap:wrap;gap:8px}
.chat-thumb{display:flex;flex-direction:column;align-items:flex-start;gap:4px}
.chat-thumb img{height:64px;border:1px solid var(--rule);border-radius:3px}
@media (min-width:981px){
  #chat-settings-toggle,.msg-details{display:none}
}
@media (max-width:980px){
  .chat-panes{grid-template-columns:minmax(0,1fr)}
  .chat:not(.params-open) .chat-side{display:none}
  .chat.params-open .chat-main{display:none}
  .chat.params-open .chat-side{padding-left:0;border-left:0}
  .chat.params-open .chat-back{display:inline-flex}
}
@media (max-width:520px){
  .chat-group{grid-template-columns:minmax(0,1fr)}
  .chat-bar{gap:6px 10px}
}

/* Settings: an index of line-select prompts, then one page per category. */
.lsk-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 28px}
.lsk{display:flex;flex-direction:column;gap:1px;padding:9px 0;border-bottom:1px solid var(--rule-2);text-decoration:none;min-width:0}
.lsk-r{align-items:flex-end;text-align:right}
.lsk{position:relative}
.lsk-l::before,.lsk-r::after{content:'';position:absolute;top:calc(50% - 8px);width:24px;height:15px;border-radius:3px;background:var(--key);border:1px solid #0f1113;border-top-color:var(--key-top);transition:border-color .12s ease-out}
.lsk-l::before{left:-56px}.lsk-r::after{right:-56px}
.lsk:hover::before,.lsk:hover::after,.lsk:focus-visible::before,.lsk:focus-visible::after{border-top-color:var(--cyan)}
.lsk-prompt{color:var(--cyan);font-size:14px;font-weight:600}
.lsk-sum{max-width:100%;font-size:12.5px;color:var(--dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-variant-numeric:tabular-nums}
.lsk:hover .lsk-prompt{text-decoration:underline;text-underline-offset:3px}
.lsk-sum.amber{color:var(--amber)}
.page-head{display:flex;align-items:baseline;gap:14px;margin-bottom:6px}
.page-head h2{font-size:14px;font-weight:600}
.page-head .key{margin-left:auto}
.key-small{min-height:28px;padding:0 10px;font-size:11.5px}

.cfg-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px 28px}
.cfg-field{min-width:0}
.cfg-field-top{display:flex;justify-content:space-between;align-items:baseline;gap:12px;margin-bottom:4px}
.cfg-label{font-size:11.5px;font-weight:600;color:var(--ink);text-transform:uppercase;letter-spacing:.05em}
.cfg-flag{font:11px var(--f-mono);color:var(--faint);overflow-wrap:anywhere;max-width:50%;text-align:right}
.cfg-input,.cfg-select{width:100%}
.cfg-select{color:var(--cyan)}
.cfg-select option{background:var(--glass-2);color:var(--ink)}
.cfg-input[readonly]{color:var(--dim);background:transparent;border-style:dashed}
.cfg-input.dirty,.cfg-select.dirty{border-color:var(--amber);color:var(--amber);background:var(--amber-glass)}
.cfg-input:invalid{border-color:var(--red)}
.cfg-input:disabled,.cfg-select:disabled{color:var(--off);border-color:var(--field-edge-off);cursor:not-allowed}
.cfg-help{font-size:12px;color:var(--faint);line-height:1.55;margin-top:4px}
.cfg-check{display:flex;align-items:center;gap:9px;min-height:34px;font:500 14px var(--f-panel);font-stretch:87.5%;color:var(--cyan)}
.cfg-check input{width:16px;height:16px}
.cfg-check:has(input.dirty){color:var(--amber)}
.cfg-note{grid-column:1/-1;padding:8px 0 0;border-top:1px solid var(--rule-2);font-size:12px;color:var(--faint);line-height:1.55;max-width:95ch}
.cfg-token-controls{margin-top:6px}
.cfg-token-summary{display:flex;justify-content:space-between;gap:12px;font-size:12px;color:var(--faint)}
.cfg-token-summary output{font:600 14px var(--f-panel);font-stretch:87.5%;color:var(--ink);font-variant-numeric:tabular-nums}
.cfg-token-slider,.reserve-slider{display:block;width:100%;height:30px;margin:0;accent-color:var(--magenta);cursor:pointer}
.cfg-token-slider:disabled{cursor:not-allowed;opacity:.45}
.cfg-token-scale{position:relative;height:16px;font:11px var(--f-panel);font-stretch:87.5%;color:var(--faint);font-variant-numeric:tabular-nums}
.cfg-token-scale span{position:absolute;transform:translateX(-50%)}
.cfg-token-scale span:first-child{transform:none}.cfg-token-scale span:last-child{transform:translateX(-100%)}
.cfg-capacity-pair{grid-column:1/-1;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px 28px;padding-bottom:12px;border-bottom:1px solid var(--rule-2)}
.cfg-capacity-preview{grid-column:1/-1}
.cfg-capacity-preview p{max-width:95ch;font-size:12.5px;color:var(--dim)}
.cfg-capacity-slots{display:flex;gap:4px;height:12px;margin:0 0 6px;max-width:520px}
.cfg-capacity-slots span{flex:1;min-width:0;border-radius:1px;background:var(--track);box-shadow:inset 0 0 0 1px var(--track-edge)}
.cfg-capacity-slots span.fit{background:var(--green);box-shadow:none}
.cfg-capacity-slots span.share{background:var(--green-share);box-shadow:none}
.config-advanced{grid-column:1/-1;border-top:1px solid var(--rule-2);padding-top:8px}
.config-advanced>summary{width:fit-content;color:var(--cyan);font:600 12px var(--f-panel);font-stretch:87.5%;text-transform:uppercase;letter-spacing:.06em;cursor:pointer}
.config-advanced .cfg-grid{margin-top:12px}
.fe-editor{grid-column:1/-1;display:grid;gap:8px}
.fe-row{display:grid;grid-template-columns:minmax(120px,1fr) minmax(200px,2fr) 100px auto;gap:8px 10px;align-items:end;padding:10px 0;border-bottom:1px solid var(--rule-2)}
.fe-row label{display:grid;gap:3px;font:600 11px var(--f-panel);font-stretch:87.5%;color:var(--faint);text-transform:uppercase;letter-spacing:.05em}
.fe-row input{min-height:34px;padding:6px 10px;background:var(--glass-2);border:1px solid var(--field-edge);border-radius:4px;color:var(--cyan);font:500 14px var(--f-panel);font-stretch:87.5%;min-width:0;text-transform:none;letter-spacing:0}
.fe-meta{grid-column:1/-1;display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--faint)}
.fe-actions{display:flex;flex-wrap:wrap;gap:8px;align-items:center;justify-content:space-between;margin-top:4px}
.empty-state{font-size:12.5px;color:var(--faint)}
.control-group{display:flex;gap:6px;flex-wrap:wrap;align-items:center}

.cfg-actions{position:sticky;bottom:0;z-index:3;display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:14px -18px -18px;padding:10px 18px;background:var(--glass);border-top:1px solid var(--rule);border-radius:0 0 var(--r-glass) var(--r-glass)}
.cfg-status{flex:1;min-width:120px;font-size:12.5px;color:var(--faint)}
.cfg-status.dirty{color:var(--amber)}
.cfg-status.saved{color:var(--green)}
.key-insert{color:var(--cyan)}
.cfg-errors{margin-top:12px;padding:10px 12px;background:var(--red-glass);color:var(--red-on-glass);border-radius:4px;font-size:12.5px}
.cfg-errors ul{margin:4px 0 0;padding-left:18px;list-style:disc}
/* A staged revision turns the page amber, the way a temporary flight plan does,
   until it is inserted or erased. */
.cdu.tmpy{box-shadow:0 0 0 1px var(--amber),0 1px 0 1px var(--bezel-glint)}
.cdu.tmpy .cdu-title h1{color:var(--amber)}

.presets{border:0;margin:0;padding:0;min-width:0}
.preset{display:grid;grid-template-columns:22px minmax(0,1fr) auto;gap:1px 10px;padding:7px 0;border-bottom:1px solid var(--rule-2);cursor:pointer}
.preset input{margin:3px 0 0;width:15px;height:15px}
.preset-name{font:600 14.5px var(--f-panel);font-stretch:87.5%;color:var(--cyan)}
.preset-detail{grid-column:2;font-size:12.5px;color:var(--dim)}
.preset code{grid-column:2;font:11.5px var(--f-mono);color:var(--faint);overflow-wrap:anywhere}
.preset-tag{grid-row:1;grid-column:3;font:600 11px var(--f-panel);font-stretch:87.5%;color:var(--green);text-transform:uppercase;letter-spacing:.06em}
.preset.unavailable{cursor:not-allowed}
.preset.unavailable .preset-name{color:var(--off)}
.preset.selected:not(.current) .preset-name{color:var(--amber)}
.preset:has(input:focus-visible){outline:2px solid var(--cyan);outline-offset:2px}

.reserve-layout{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px 32px;align-items:start}
.reserve-entry .cdu-field-row{max-width:220px;margin-bottom:6px}
.reserve-scale{display:flex;justify-content:space-between;font-size:11px;color:var(--faint);text-transform:uppercase;letter-spacing:.05em}
.reserve-preview{display:flex;height:12px;margin-top:10px;overflow:hidden;border-radius:1px;background:var(--track)}
.reserve-preview span{background:var(--ink);transition:width .15s ease-out}
.reserve-preview span+span{background:repeating-linear-gradient(-45deg,var(--red) 0 5px,var(--pole-dark) 5px 10px)}
.reserve-facts{display:grid;gap:8px}

dialog{border:0;padding:16px 18px;max-width:440px;width:calc(100% - 32px);background:var(--glass);color:var(--ink);border-radius:var(--r-glass);box-shadow:0 0 0 1px var(--rule),0 18px 50px var(--dialog-shadow)}
dialog::backdrop{background:var(--backdrop)}
dialog h2{font:600 16px var(--f-panel);font-stretch:87.5%;text-transform:uppercase;letter-spacing:.05em;padding-bottom:8px;border-bottom:1px solid var(--rule)}
.dialog-path{font:600 14px var(--f-mono);text-transform:none;letter-spacing:0}
dialog p{margin:12px 0 16px;font-size:13px;color:var(--dim)}
.dialog-keys{display:flex;justify-content:space-between;gap:8px}

/* State colours last, so a state always wins over a component's resting colour. */
.green,.cdu-data.green,.fma-mode.green{color:var(--green)}
.cyan{color:var(--cyan)}
.magenta{color:var(--magenta)}
.amber{color:var(--amber)}
.red{color:var(--red)}
.ice{color:var(--ice)}
.dim{color:var(--dim)}
@media (min-width:1180px){
  .monitor-grid{position:relative;grid-template-columns:minmax(0,var(--col-a,1.55fr)) minmax(380px,var(--col-b,1fr));grid-template-rows:minmax(232px,var(--row-a,1fr)) minmax(206px,var(--row-b,1fr));grid-template-areas:"mem side" "rates side"}
  .monitor-side{grid-area:side;display:flex;flex-direction:column;gap:var(--gap);min-height:0}
  .monitor-side .display-traffic{flex:0 1 auto;min-height:118px;max-height:none}
  .monitor-side .display-alerts{flex:1 0 auto}
  .display-alerts .alert-list li{display:grid;grid-template-columns:66px minmax(0,1fr);gap:1px 10px;padding:4px 0}
  .display-alerts .alert-title,.display-alerts .alert-text{grid-column:2;white-space:normal}
  .display-memory{grid-area:mem}.display-rates{grid-area:rates}
  .display-traffic .traffic-scroll{flex:1}
}
/* Seams: the panel gaps between displays, draggable in the wide layout. A short grip shows
   on hover; the focused seam lights cyan, the colour of what can be selected. */
.seam{display:none}
@media (min-width:1180px){
  body:not(.explain) .seam{display:block;position:absolute;z-index:3;touch-action:none}
  .seam-col{top:0;bottom:0;width:var(--gap);cursor:col-resize}
  .seam-row{height:var(--gap);cursor:row-resize}
  .seam::after{content:'';position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);border-radius:1px;background:var(--on-panel-dim);opacity:.35;transition:opacity .15s ease-out}
  .seam:is(:focus-visible,.dragging)::before{content:'';position:absolute;background:var(--cyan)}
  .seam-col:is(:focus-visible,.dragging)::before{left:calc(50% - 1px);top:0;bottom:0;width:2px}
  .seam-row:is(:focus-visible,.dragging)::before{top:calc(50% - 1px);left:0;right:0;height:2px}
  .seam-col::after{width:2px;height:32px}
  .seam-row::after{width:32px;height:2px}
  .seam:hover::after,.seam.dragging::after{opacity:1}
  .seam.dragging::after{background:var(--cyan)}
  .seam:focus-visible{outline:none}
  .seam:focus-visible::after{opacity:1;background:var(--cyan);box-shadow:0 0 0 2px var(--cyan)}
  body.resizing-col,body.resizing-col *{cursor:col-resize!important;user-select:none}
  body.resizing-row,body.resizing-row *{cursor:row-resize!important;user-select:none}
}
@media (max-width:880px){
  .view-bench,.view-bench .cdu{height:auto}
)HTML"
R"HTML(  .bench-panes{grid-template-columns:minmax(0,1fr)}
  .bench-index{padding-right:0;border-right:0;padding-bottom:8px;border-bottom:1px solid var(--rule-2)}
  .bench-list{max-height:30vh}
  .bench-out{min-height:260px}
  .bench-ev{grid-template-columns:54px 84px minmax(0,1fr)}
}
@media (max-width:820px){
  .view-cdu{padding:0}
  .lsk-l::before,.lsk-r::after{display:none}
  .glareshield{flex-wrap:wrap;gap:8px 10px;padding:7px 8px}
  .brand-name{display:none}
  .fkeys{order:3;flex-basis:100%;gap:4px}
  .fkeys .fkey{flex:1;padding:0 6px}
  .fma{grid-template-columns:repeat(2,minmax(0,1fr))}
  .fma-col{border-left:0;border-top:1px solid var(--rule)}
  .fma-engine{grid-column:1/-1}
  .fma-col:nth-child(2){border-top:1px solid var(--rule)}
  .fma-col:nth-child(odd):not(.fma-engine){border-left:1px solid var(--rule)}
  .display-body{grid-template-columns:52px minmax(0,1fr);grid-template-rows:auto minmax(100px,1fr)}
  .readouts{grid-column:1/-1;grid-row:1;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border-left:0;padding-left:0}
  .memory-readouts .ro:not(.ro-major){display:block}
  .memory-readouts .ro:not(.ro-major) dd b{font-size:17px}
  .ro-pair{display:contents}
  .monitor-grid{grid-template-rows:minmax(250px,1.3fr) minmax(230px,1fr) auto auto}
  .display-traffic{max-height:none}
  .lsk-grid,.cfg-grid,.cfg-capacity-pair,.reserve-layout{grid-template-columns:minmax(0,1fr)}
  .lsk-r{align-items:flex-start;text-align:left}
  .lsk-r .lsk-prompt{flex-direction:row-reverse}
  .fe-row{grid-template-columns:1fr}
}
@media (max-width:560px){
  .rate-readouts,.memory-readouts{grid-template-columns:repeat(2,minmax(0,1fr))}
  .ro-major dd b{font-size:22px}
  .traffic thead{display:none}
  .traffic,.traffic tbody,.traffic tr,.traffic td{display:block;width:100%}
  .traffic tr.client{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:4px 10px;padding:8px 0;border-bottom:1px solid var(--rule-2)}
  .traffic tr.client td{border:0;padding:0;text-align:left}
  .traffic tr.client td:first-child{grid-column:1/-1}
  .traffic td.num::before{content:attr(data-label);display:block;font:600 11px var(--f-panel);font-stretch:87.5%;color:var(--faint);text-transform:uppercase;letter-spacing:.06em}
  .view-monitor{height:auto;min-height:0}
  .monitor-grid{height:auto;grid-template-rows:auto}
  .display-body{flex:none;grid-template-rows:auto 170px}
  .display-traffic{max-height:none}
  .fma-sub{white-space:normal}
  .alert-list li{display:grid;grid-template-columns:66px minmax(0,1fr);gap:0 8px}
  .alert-title,.alert-text{grid-column:2;white-space:normal}
  .traffic .s,.traffic .v{white-space:normal}
  .gpu-apps{flex-direction:column;gap:2px}
  .finding,.finding.compact{grid-template-columns:1fr}
  .finding-sev{grid-row:auto}
  .finding.compact .finding-body{white-space:normal}
  .cdu{padding:10px 12px 14px}
  .cdu-title{grid-template-columns:auto minmax(0,1fr);grid-template-areas:'title right' 'left left'}
  .cdu-title h1{text-align:left}
  .cdu-page{overflow:hidden;text-overflow:ellipsis;max-width:100%}
  .cfg-actions{margin:12px -12px -14px;padding:10px 12px}
  .cdu-facts div{grid-template-columns:1fr;gap:2px}
}
@media (max-height:780px) and (min-width:821px){
  .readouts{gap:3px}
  .ro-major dd b{font-size:20px}
  .ro dd b{font-size:15px}
  .ro-sub{display:none!important}
  .gpu-apps{margin-top:4px;padding-top:4px}
  .display{padding:5px 10px 7px}
  .monitor-grid{grid-template-rows:minmax(196px,1.3fr) minmax(176px,1fr) auto auto}
  .alert-list li{font-size:12px}
}
@media (max-height:780px) and (min-width:1180px){
  .monitor-grid{grid-template-rows:minmax(196px,var(--row-a,1fr)) minmax(176px,var(--row-b,1fr))}
}
@media (min-width:561px){
  .display-traffic .traffic{table-layout:fixed}
  .display-traffic .client-name span:first-child{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}
  .display-traffic .traffic .s{white-space:normal}
}
@media (min-width:821px){
  body:not(.explain) .display-memory .display-body,body:not(.explain) .display-rates .display-body{container:dbody/size}
}
@container dbody (max-height:190px){
  .readouts{gap:3px}
  .ro dt{font-size:11px}
  .ro-major dd b{font-size:20px}
  .ro-sub{font-size:11px!important}
  .ro-pair dd b{font-size:14px}
}
@container dbody (max-height:150px){
  .ro-major dd b{font-size:17px}
  .ro-sub{line-height:1.2}
}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{transition:none!important;animation:none!important;scroll-behavior:auto!important}}
</style>
</head>
<body>
<a href="#main" class="skip-link">Skip to content</a>
<svg class="icon-defs" aria-hidden="true" xmlns="http://www.w3.org/2000/svg"><defs>
<symbol id="i-copy" viewBox="0 0 16 16"><rect x="5.5" y="5.5" width="8" height="8" rx="1"/><path d="M10.5 5.5v-3h-8v8h3"/></symbol>
<symbol id="i-out" viewBox="0 0 16 16"><path d="M9 2.5h4.5V7M13.5 2.5 7.5 8.5M11.5 9.5v4h-9v-9h4"/></symbol>
<symbol id="i-display" viewBox="0 0 16 16"><circle cx="8" cy="8" r="5.5"/><path d="M8 2.5v11a5.5 5.5 0 0 0 0-11z" fill="currentColor"/></symbol>
<symbol id="i-explain" viewBox="0 0 16 16"><circle cx="8" cy="8" r="6"/><path d="M8 7.2v4.3M8 4.6v.1"/></symbol>
</defs></svg>

<header class="glareshield">
  <a class="brand" href="#overview" aria-label="NInfer monitor"><span class="brand-mark" aria-hidden="true">n<span>.</span></span><span class="brand-name">NInfer</span></a>
  <nav class="fkeys" aria-label="Main navigation">
    <a class="fkey" href="#overview" data-view="overview" aria-current="page"><span class="fkey-lamp" aria-hidden="true"></span>Monitor</a>
    <a class="fkey" href="#chat" data-view="chat"><span class="fkey-lamp" aria-hidden="true"></span>Chat</a>
    <a class="fkey" href="#connect" data-view="connect"><span class="fkey-lamp" aria-hidden="true"></span>Connect</a>
    <a class="fkey" href="#api" data-view="api"><span class="fkey-lamp" aria-hidden="true"></span>API</a>
    <a class="fkey" href="#diagnostics" data-view="diagnostics"><span class="fkey-lamp" aria-hidden="true"></span>Diagnose<span id="nav-issues" class="fkey-count" hidden></span></a>
    <a class="fkey" href="#settings" data-view="settings"><span class="fkey-lamp" aria-hidden="true"></span>Settings<span id="settings-dirty" class="fkey-dirty" hidden>Unsaved</span></a>
  </nav>
  <div class="glare-end">
    <button type="button" id="master-caution" class="master" aria-label="Master caution: no new findings" title="Lights for new caution or warning findings. Press to acknowledge and open Diagnose."><span>Master</span><span id="master-word">Caution</span></button>
    <button type="button" id="display-mode" class="fkey fkey-small" title="Display mode: Auto follows Windows; Day and Night stay as chosen"><svg class="icon"><use href="#i-display"/></svg><span id="display-mode-label">Auto</span></button>
    <button type="button" id="explain-toggle" class="fkey fkey-small" aria-pressed="false" title="Show plain-language explanations on every page"><svg class="icon"><use href="#i-explain"/></svg>Explain</button>
    <span class="link-state" role="status"><span id="live-dot" class="lamp paused" aria-hidden="true"></span><span id="live-label">Connecting…</span></span>
  </div>
</header>

<section class="fma" aria-label="Engine status">
  <div class="fma-col fma-engine">
    <div class="fma-engine-state">
      <span class="fma-label">Engine</span>
      <strong id="kpi-state-badge" class="fma-mode amber" data-annunciate>Connecting</strong>
      <span class="fma-sub">Up <b id="kpi-uptime">—</b></span>
      <span id="fma-armed" class="fma-armed" hidden></span>
    </div>
    <div id="controls" class="engine-keys" role="group" aria-label="Engine controls">
      <button class="key key-go" data-act="start" id="btn-start" disabled>Start</button>
      <button class="key" data-act="restart" id="btn-restart" disabled>Restart</button>
      <button class="key key-stop" data-act="stop" id="btn-stop" disabled>Stop</button>
    </div>
  </div>
  <div class="fma-col fma-model">
    <span class="fma-label">Model</span>
    <strong id="model-label" class="fma-value" data-annunciate>Loading…</strong>
    <a class="prompt" href="#settings/presets">Presets<span aria-hidden="true">&gt;</span></a>
  </div>
  <div class="fma-col fma-plan">
    <span class="fma-label">Running plan</span>
    <strong id="plan-mode" class="fma-value" data-annunciate>—</strong>
    <span id="plan-brief" class="fma-sub">Waiting for launch details</span>
  </div>
  <div class="fma-col fma-api">
    <span class="fma-label">API address</span>
    <button type="button" id="fma-api-copy" class="fma-copy" title="Copy the configured base URL"><span id="overview-endpoint">Loading…</span><svg class="icon"><use href="#i-copy"/></svg></button>
    <span id="overview-connection-pending" class="fma-sub amber" hidden>Restart to apply</span>
  </div>
</section>

<div class="scratchpad" role="status" aria-live="polite">
  <p id="connection-alert" class="pad-line amber" hidden>Live updates are disconnected. Showing the last received values; reconnecting automatically.</p>
  <p id="engine-memo" class="pad-line" hidden></p>
  <p id="monitor-note" class="pad-line" hidden>This dashboard monitors an engine started elsewhere. Start and stop it in its original application.</p>
  <div id="action-status-line" class="pad-line pad-action" hidden><span id="action-status"></span><button type="button" id="action-clear" class="key key-tiny" aria-label="Clear message">Clr</button></div>
</div>

<div class="live-band" aria-label="Live engine readings">
  <span class="band-item"><span class="band-label">Memory</span><b id="band-mem">—</b><span class="band-bar"><span id="band-mem-bar"></span></span><span id="band-free" class="band-sub">—</span></span>
  <span class="band-item"><span class="band-label">Generating</span><b id="band-gen">—</b></span>
  <span class="band-item"><span class="band-label">Reading</span><b id="band-read">—</b></span>
  <span class="band-item"><span class="band-label">Running</span><b id="band-run">—</b></span>
  <a class="prompt band-return" href="#overview"><span aria-hidden="true">&lt;</span>Monitor</a>
</div>

<main id="main" class="workspace" tabindex="-1">

<section id="view-overview" class="view view-monitor" aria-labelledby="overview-title">
  <h1 id="overview-title" class="sr-only">Monitor</h1>
  <div class="monitor-grid">
    <section class="display display-memory" aria-labelledby="memory-title">
      <header class="display-head">
        <h2 id="memory-title">GPU memory</h2>
        <span id="dt-gpu-name" class="display-source">Graphics card</span>
        <span id="chart-range" class="display-range">Live history</span>
      </header>
      <div class="display-body">
        <div class="tape"><canvas id="memory-tape" aria-hidden="true"></canvas></div>
        <div class="trace">
          <canvas id="timeline-canvas" tabindex="0" role="img" aria-label="GPU memory history. Use the left and right arrow keys to move the time cursor; Escape clears it."></canvas>
          <div id="chart-tooltip" class="cursor-readout" aria-hidden="true"></div>
          <div id="chart-empty" class="trace-empty">Waiting for memory samples…</div>
        </div>
        <dl class="readouts memory-readouts">
          <div class="ro ro-major"><dt>Used</dt><dd><b id="kpi-vram-main">—</b><span id="kpi-vram-pct" class="ro-unit">—</span></dd></div>
          <div class="ro"><dt>Free</dt><dd><b id="kpi-vram-free" class="green">—</b></dd></div>
          <div class="ro"><dt>Card</dt><dd><b id="kpi-vram-total">—</b></dd></div>
          <div class="ro"><dt>Reserve</dt><dd><b id="ro-reserve" class="magenta">—</b><a class="prompt" href="#settings/reserve">Set<span aria-hidden="true">&gt;</span></a></dd></div>
        </dl>
      </div>
      <div class="gpu-apps"><span class="display-label">Using the card</span><div id="gpu-app-list" class="gpu-app-list">Waiting for app readings…</div></div>
)HTML"
R"HTML(      <p class="note">A loaded model holds GPU memory even while idle, so high usage alone is not a problem. The red-striped band is the desktop reserve: memory NInfer leaves free for your other apps when it starts. Windows reports each app's dedicated memory; shared allocations appear in several apps, so the list does not add up to the card total.</p>
      <p id="memory-guidance" class="pad-line amber" hidden></p>
    </section>

    <section class="display display-rates" aria-labelledby="throughput-title">
      <header class="display-head">
        <h2 id="throughput-title">Rates</h2>
        <span id="kpi-thr-window" class="display-source">—</span>
        <span id="throughput-range" class="display-range">Live history</span>
      </header>
      <div class="display-body">
        <div class="tape"><canvas id="rate-tape" aria-hidden="true"></canvas></div>
        <div class="trace">
          <canvas id="throughput-canvas" tabindex="0" role="img" aria-label="Generation and prompt-reading speed over time, on separate scales. Arrow keys move the time cursor."></canvas>
          <div id="throughput-tooltip" class="cursor-readout" aria-hidden="true"></div>
          <div id="throughput-empty" class="trace-empty">Waiting for request samples…</div>
        </div>
        <dl class="readouts rate-readouts">
          <div class="ro ro-major"><dt><i class="swatch swatch-gen"></i>Generating</dt><dd><b id="thr-decode" class="green">—</b></dd><dd id="kpi-decode-explainer" class="ro-sub">all requests together</dd></div>
          <div class="ro ro-major"><dt><i class="swatch swatch-read"></i>Reading</dt><dd><b id="thr-prefill" class="ice">—</b></dd></div>
          <div class="ro-pair">
            <div class="ro"><dt>First token</dt><dd><b id="kpi-ttft">—</b></dd></div>
            <div class="ro"><dt>Done</dt><dd><b id="kpi-done">—</b></dd></div>
          </div>
        </dl>
      </div>
      <p class="note">Measured by the engine while work happens, so long answers count as they stream. Every generated token counts, reasoning included. Prompt reading runs far faster than generation, so the two traces use separate scales: generation on the green tape at the left, prompt reading on the blue figures at the right edge. First token is the average wait before a response begins; Done counts completed requests in the request log.</p>
    </section>

    <div class="monitor-side">
    <section class="display display-traffic" aria-labelledby="clients-title" id="clients-panel">
      <header class="display-head">
        <h2 id="clients-title">Calling apps</h2>
        <span id="clients-sub" class="display-source">No requests observed yet</span>
        <span id="clients-count" class="display-range">—</span>
      </header>
      <div class="traffic-scroll">
        <table class="traffic">
          <colgroup><col class="col-app"><col span="4"></colgroup>
          <thead><tr><th scope="col">App · requests</th><th scope="col" class="num">First token</th><th scope="col" class="num">Queue</th><th scope="col" class="num">Prefill</th><th scope="col" class="num">Reuse</th></tr></thead>
          <tbody id="clients-list"><tr class="traffic-empty"><td colspan="5">No app has sent a request in the observation window. Apps appear here as they call the engine.</td></tr></tbody>
        </table>
      </div>
      <p class="note">Times are average, with the 95th percentile beneath. An app in amber averages a second or more in the queue. Queue is time spent waiting for a free lane, measured separately from prefill; prefill is the engine reading the prompt. Reuse is the share of prompt tokens restored from cache instead of read again; a full prefill alone does not show why reuse was unavailable. fail = failed, cxl = cancelled, cut = unfinished when the engine restarted, open = still awaiting an outcome.</p>
    </section>

    <section class="display display-alerts" aria-labelledby="alerts-title">
      <header class="display-head">
        <h2 id="alerts-title">Alerts</h2>
        <span id="attention-title" class="display-source">Checking findings…</span>
        <a class="prompt display-range" href="#diagnostics">Diagnose<span aria-hidden="true">&gt;</span></a>
      </header>
      <ul id="alert-list" class="alert-list"><li class="alert-quiet">Findings appear as the engine records activity.</li></ul>
    </section>
    </div>
    <div id="seam-col" class="seam seam-col" role="separator" tabindex="0" aria-orientation="vertical" aria-label="Width of memory and rates against calling apps and alerts" aria-valuemin="0" aria-valuemax="100" title="Drag to resize · double-click to restore"></div>
    <div id="seam-row" class="seam seam-row" role="separator" tabindex="0" aria-orientation="horizontal" aria-label="Height of memory against rates" aria-valuemin="0" aria-valuemax="100" title="Drag to resize · double-click to restore"></div>
  </div>
</section>

<section id="view-chat" class="view view-cdu view-chat" aria-labelledby="chat-title" hidden>
  <div class="cdu cdu-wide chat">
    <header class="chat-bar">
      <div class="chat-head-id"><h1 id="chat-title" class="chat-title">Chat</h1><span id="chat-model" class="chat-model-badge">Checking the model…</span></div>
      <div class="chat-context" title="Tokens the conversation holds, against the model's context limit">
        <span class="cdu-label">Context</span>
        <span class="chat-meter" aria-hidden="true"><span id="chat-meter-fill"></span></span>
        <span id="chat-context-text" class="chat-context-text">—</span>
      </div>
      <div class="chat-think" role="group" aria-label="Thinking">
        <span class="cdu-label">Thinking</span>
        <button type="button" class="cdu-tab" data-think="default" aria-pressed="true" title="Use the chat template's default">Default</button>
        <button type="button" class="cdu-tab" data-think="on" aria-pressed="false">On</button>
        <button type="button" class="cdu-tab" data-think="off" aria-pressed="false">Off</button>
      </div>
      <div class="chat-budget-now" title="The output budget the next request carries"><span class="cdu-label">Output</span><span id="chat-budget-now"></span></div>
      <button type="button" id="chat-settings-toggle" class="key key-small" aria-expanded="false" aria-controls="chat-side">Parameters<span id="chat-param-count-bar" class="chat-param-count"></span></button>
      <button type="button" id="chat-new" class="key key-small">New chat</button>
    </header>
    <div class="chat-panes">
      <div class="chat-main">
        <p id="chat-think-note" class="pad-line amber" hidden>Changing thinking in the middle of a conversation makes the engine read the whole conversation again instead of reusing it. Keep it the same for a conversation, or start a new chat.</p>
        <div id="chat-log" class="chat-log" role="log" aria-label="Conversation" tabindex="0">
          <div id="chat-empty" class="chat-empty">
            <p class="chat-empty-title">Talk to your model</p>
            <p>Each reply shows how long it took to start, how fast it generated, how much of the conversation the engine reused instead of reading it again, and the parameters it sent. An answer ends when the model finishes, at a stop string, or at the output budget shown above, which Parameters sets.</p>
          </div>
        </div>
        <form id="chat-form" class="chat-composer" autocomplete="off">
          <div id="chat-attachments" class="chat-attachments" hidden></div>
          <label class="sr-only" for="chat-input">Message</label>
          <textarea id="chat-input" class="chat-input" rows="2" placeholder="Message your model · Enter sends, Shift+Enter adds a line"></textarea>
          <div class="chat-keys">
            <button type="button" id="chat-attach" class="key key-small" hidden>Attach image</button>
            <input type="file" id="chat-file" accept="image/*" multiple hidden>
            <span id="chat-hint" class="cfg-status"></span>
            <button type="button" id="chat-inspect" class="key key-small" title="Open this conversation's next request on the API page">Open in API</button>
            <button type="submit" id="chat-send" class="key key-insert">Send<span aria-hidden="true">&gt;</span></button>
            <button type="button" id="chat-stop" class="key key-stop" hidden>Stop</button>
          </div>
        </form>
      </div>
      <aside id="chat-side" class="chat-side" aria-label="Reply telemetry and request parameters">
        <button type="button" id="chat-back" class="prompt chat-back">&lt;Chat</button>
        <section id="chat-inspector" class="chat-inspector" aria-labelledby="chat-inspect-title">
          <header class="chat-side-head">
            <h2 id="chat-inspect-title" class="chat-side-title">Reply inspector</h2>
            <span id="chat-inspect-meta" class="chat-side-meta">No replies yet</span>
          </header>
          <div id="chat-inspect-body" class="chat-inspect-body"><p class="cdu-small">Send a message to inspect its timings, cache reuse, speculative drafts, and sent parameters.</p></div>
        </section>
        <div id="chat-settings" class="chat-settings">
          <div class="chat-side-head chat-params-head">
            <h2 class="chat-side-title">Parameters<span id="chat-param-count" class="chat-param-count"></span></h2>
            <button type="button" id="chat-params-reset" class="client-evidence">Reset defaults</button>
          </div>
          <p class="chat-settings-note">An empty field sends nothing, so the engine's own default applies. Each reply lists what it sent.</p>
          <fieldset class="chat-group">
            <legend>Output</legend>
            <label class="chat-setting"><span class="cdu-label">Output budget</span>
              <span class="chat-budget"><select id="chat-budget" class="cfg-select"><option value="unlimited">No limit</option><option value="custom">Custom</option><option value="default">Engine default</option></select><input id="chat-max" class="cdu-input cdu-input-num" type="number" min="0" step="1" placeholder="tokens" aria-label="Output tokens" hidden></span></label>
            <p id="chat-budget-help" class="cfg-help chat-setting-wide"></p>
            <label class="cfg-check chat-setting"><input type="checkbox" id="chat-stream" checked><span>Stream the answer</span></label>
            <label class="cfg-check chat-setting"><input type="checkbox" id="chat-show-reasoning"><span>Open reasoning while it streams</span></label>
            <label class="chat-setting chat-setting-wide"><span class="cdu-label">System prompt</span><textarea id="chat-system" class="cdu-input chat-text" rows="2" placeholder="Optional instructions for the whole conversation"></textarea></label>
            <label class="chat-setting chat-setting-wide"><span class="cdu-label">Stop strings · one per line, up to four</span><textarea id="chat-stops" class="cdu-input chat-text" rows="2" placeholder="None"></textarea></label>
            <label class="chat-setting"><span class="cdu-label">Response format</span><select id="chat-format" class="cfg-select"><option value="">Free text</option><option value="json_object">JSON object</option><option value="json_schema">JSON schema</option></select></label>
            <label id="chat-schema-row" class="chat-setting chat-setting-wide" hidden><span class="cdu-label">JSON schema</span><textarea id="chat-schema" class="cdu-input chat-text chat-code" rows="4" spellcheck="false" placeholder='{"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}'></textarea><span id="chat-schema-state" class="cfg-help"></span></label>
          </fieldset>
          <fieldset class="chat-group">
            <legend>Sampling</legend>
            <label class="chat-setting"><span class="cdu-label">Temperature</span><input name="temperature" data-param="temperature" class="cdu-input" type="number" min="0" max="2" step="0.05" placeholder="Engine default"></label>
            <label class="chat-setting"><span class="cdu-label">Top p</span><input name="top_p" data-param="top_p" class="cdu-input" type="number" min="0" max="1" step="0.01" placeholder="Engine default"></label>
            <label class="chat-setting"><span class="cdu-label">Top k · 0 to 20</span><input name="top_k" data-param="top_k" data-int class="cdu-input" type="number" min="0" max="20" step="1" placeholder="Engine default"></label>
            <label class="chat-setting"><span class="cdu-label">Min p · 0 to 1</span><input name="min_p" data-param="min_p" class="cdu-input" type="number" min="0" max="1" step="0.01" placeholder="Engine default"></label>
            <label class="chat-setting"><span class="cdu-label">Presence penalty</span><input name="presence_penalty" data-param="presence_penalty" class="cdu-input" type="number" min="-2" max="2" step="0.1" placeholder="Engine default"></label>
            <label class="chat-setting"><span class="cdu-label">Frequency penalty</span><input name="frequency_penalty" data-param="frequency_penalty" class="cdu-input" type="number" min="-2" max="2" step="0.1" placeholder="Engine default"></label>
            <label class="chat-setting"><span class="cdu-label">Seed</span><input name="seed" data-param="seed" data-int class="cdu-input" type="number" step="1" placeholder="New each request"></label>
          </fieldset>
          <fieldset class="chat-group">
            <legend>Reasoning</legend>
            <label class="chat-setting"><span class="cdu-label">Reasoning effort</span><select name="reasoning_effort" data-param="reasoning_effort" class="cfg-select"><option value="">Template default</option><option>none</option><option>minimal</option><option>low</option><option>medium</option><option>high</option><option>xhigh</option><option>max</option></select></label>
            <label class="chat-setting"><span class="cdu-label">Thinking budget · tokens</span><input name="thinking_budget_tokens" data-param="thinking_budget_tokens" data-int class="cdu-input" type="number" min="-1" step="1" placeholder="Unbounded"></label>
            <label class="cfg-check chat-setting chat-setting-wide"><input type="checkbox" name="preserve_thinking" data-param="preserve_thinking" data-bool><span>Keep earlier reasoning in the history</span></label>
            <p id="chat-reasoning-help" class="cfg-help chat-setting-wide" hidden></p>
          </fieldset>
          <fieldset class="chat-group">
            <legend>NInfer</legend>
            <label class="cfg-check chat-setting"><input type="checkbox" id="chat-logprobs"><span>Token probabilities</span></label>
            <label class="chat-setting"><span class="cdu-label">Alternatives per token · 0 to 20</span><input id="chat-top-logprobs" class="cdu-input" type="number" min="0" max="20" step="1" placeholder="0"></label>
            <p id="chat-logprobs-help" class="cfg-help chat-setting-wide">Each reply gets a token view that marks the tokens the model was unsure of and shows the top alternatives on hover.</p>
)HTML"
R"HTML(            <label class="cfg-check chat-setting chat-setting-wide"><input type="checkbox" name="prompt_cache_read_only" data-param="prompt_cache_read_only" data-bool><span>Read the prompt cache without adding to it</span></label>
            <p class="cfg-help chat-setting-wide">For tests that should not evict other apps' cached conversations. The request still starts from a prefix already in the cache but publishes nothing of its own, so the next turn cannot pick up where this one ended.</p>
          </fieldset>
          <fieldset class="chat-group chat-group-wide">
            <legend>Extra request JSON</legend>
            <label class="chat-setting chat-setting-wide"><span class="sr-only">Extra request JSON</span><textarea id="chat-extra" class="cdu-input chat-text chat-code" rows="3" spellcheck="false" placeholder='Merged into the request last, so it can set or override any field, for example {"tool_choice": "none"}'></textarea><span id="chat-extra-state" class="cfg-help"></span></label>
          </fieldset>
        </div>
      </aside>
    </div>
  </div>
</section>

<section id="view-connect" class="view view-cdu" aria-labelledby="connect-title" hidden>
  <div class="cdu">
    <header class="cdu-title"><h1 id="connect-title">Connect an app</h1><span class="cdu-page">1/1</span></header>
    <p class="cdu-help">Use your local model in any app that speaks the OpenAI API.</p>
    <ol class="cdu-lines connect-lines">
      <li class="cdu-line"><span class="cdu-label">1 · Engine</span><span id="connect-readiness" class="cdu-data">Checking engine status…</span></li>
      <li class="cdu-line">
        <label class="cdu-label" for="api-address">2 · Base URL</label>
        <span class="cdu-small">Paste it into your app's OpenAI-compatible provider settings.</span>
        <span class="cdu-field-row"><input id="api-address" class="cdu-input cdu-input-mono" readonly value="Loading…"><button type="button" class="key" id="copy-address"><svg class="icon"><use href="#i-copy"/></svg>Copy</button></span>
        <span id="connection-pending" class="cdu-data amber" hidden>Connection settings were saved but may not be active yet. Restart the engine and confirm it is ready before using this address.</span>
        <span class="cdu-small">This is the saved address and works on this computer. Reaching it from another device needs network configuration.</span>
      </li>
      <li class="cdu-line"><span class="cdu-label">3 · API key</span><span id="auth-guidance" class="cdu-data">Loading API authentication details…</span></li>
      <li class="cdu-line"><span class="cdu-label">4 · Model</span><span id="model-instruction" class="cdu-data">Refresh your app's model list, choose the model NInfer serves, then send a short message. Activity appears on Monitor; if the app cannot connect, check that the engine is ready and the base URL matches.</span></li>
      <li class="cdu-line" id="frontends-step">
        <span class="cdu-label">Or chat in your browser</span>
        <span class="cdu-small">Served on this computer; no URL or key to set.</span>
        <ul id="frontend-list" class="frontend-list"></ul>
        <a href="#settings/frontends" class="prompt" id="connect-frontends-settings"><span aria-hidden="true">&lt;</span>Web frontends</a>
      </li>
    </ol>
    <dl class="cdu-glossary">
      <div><dt>API</dt><dd>The connection your app uses to talk to the engine. NInfer writes the responses; your app provides the conversation.</dd></div>
      <div><dt>Token</dt><dd>A small piece of text. Generation speed counts how many the engine produces each second.</dd></div>
    </dl>
  </div>
</section>

<section id="view-api" class="view view-cdu view-bench" aria-labelledby="api-title" hidden>
  <div class="cdu cdu-wide bench">
    <header class="cdu-title"><h1 id="api-title">API test bench</h1><span id="bench-auth" class="cdu-page">Checking authentication…</span></header>
    <div class="bench-panes">
      <nav class="bench-index" aria-label="Endpoints">
        <label class="sr-only" for="bench-filter">Filter endpoints</label>
        <input id="bench-filter" class="cdu-input" type="search" placeholder="Filter endpoints" autocomplete="off">
        <div id="bench-list" class="bench-list"></div>
        <div class="bench-history-head"><span class="cdu-label">Recent</span><button type="button" id="bench-history-clear" class="client-evidence" hidden>Clear</button></div>
        <ol id="bench-history" class="bench-history"><li class="cdu-small">Requests you send appear here.</li></ol>
      </nav>
      <section class="bench-work" aria-label="Request and response">
        <div class="bench-request">
          <div class="bench-line">
            <span id="bench-method" class="bench-method">GET</span>
            <label class="sr-only" for="bench-path">Path</label>
            <input id="bench-path" class="cdu-input cdu-input-mono" spellcheck="false" autocomplete="off">
            <button type="button" id="bench-send" class="key key-insert">Send<span aria-hidden="true">&gt;</span></button>
            <button type="button" id="bench-stop" class="key key-stop" hidden>Stop</button>
          </div>
          <p id="bench-note" class="cdu-small"></p>
          <div class="bench-line bench-options">
            <div id="bench-examples" class="bench-examples" role="group" aria-label="Examples"></div>
            <label id="bench-stream-wrap" class="cfg-check bench-stream"><input type="checkbox" id="bench-stream"><span>Stream</span></label>
            <button type="button" id="bench-format" class="key key-small">Format</button>
            <button type="button" id="bench-curl" class="key key-small"><svg class="icon"><use href="#i-copy"/></svg>Copy as curl</button>
          </div>
          <label class="sr-only" for="bench-body">Request body (JSON)</label>
          <textarea id="bench-body" class="bench-body" spellcheck="false" autocomplete="off" aria-describedby="bench-body-state"></textarea>
          <span id="bench-body-state" class="bench-body-state" role="status"></span>
        </div>
        <div class="bench-response" aria-live="off">
          <div class="bench-status">
            <span id="bench-code" class="bench-code">—</span>
            <span id="bench-timing" class="bench-timing">Send a request to see the engine's answer.</span>
            <nav class="bench-tabs" aria-label="Response views">
              <button type="button" class="cdu-tab" data-bench-tab="text" aria-pressed="false">Text</button>
              <button type="button" class="cdu-tab" data-bench-tab="body" aria-pressed="true">Body</button>
              <button type="button" class="cdu-tab" data-bench-tab="events" aria-pressed="false">Events <span id="bench-event-count" class="cdu-tab-count"></span></button>
              <button type="button" class="cdu-tab" data-bench-tab="headers" aria-pressed="false">Headers</button>
            </nav>
          </div>
          <div id="bench-out" class="bench-out" tabindex="0" aria-label="Response"></div>
        </div>
      </section>
    </div>
  </div>
</section>

<section id="view-diagnostics" class="view view-cdu" aria-labelledby="diagnostics-title" hidden>
  <div class="cdu cdu-wide">
    <header class="cdu-title"><h1 id="diagnostics-title">Diagnose</h1><span id="diag-page-label" class="cdu-page">Findings</span></header>
    <div class="cdu-health"><span id="diagnostic-health-label" class="cdu-data">Checking health…</span><span id="diagnostic-health-detail" class="cdu-small">Health details appear when the supervisor responds.</span></div>
    <nav class="cdu-tabs" aria-label="Diagnose pages">
      <a href="#diagnostics" data-diag="findings" class="cdu-tab">Findings <span id="insights-count-badge" class="cdu-tab-count">—</span></a>
      <a href="#diagnostics/log" data-diag="log" class="cdu-tab">Engine log</a>
      <a href="#diagnostics/engine" data-diag="engine" class="cdu-tab">Engine data</a>
      <a href="#diagnostics/accel" data-diag="accel" class="cdu-tab">Acceleration</a>
      <a href="#diagnostics/experts" data-diag="experts" class="cdu-tab">Experts</a>
    </nav>

    <section class="diag-page" data-diag-page="findings" aria-label="Findings">
      <p class="cdu-help explain-only">Findings from the available engine logs, most important first. Open the evidence for the measurements behind each one.</p>
      <div id="insights-container" class="findings"><p class="cdu-small">Waiting for diagnostic information…</p></div>
    </section>

    <section class="diag-page diag-log" data-diag-page="log" aria-labelledby="logs-title" hidden>
      <h2 id="logs-title" class="sr-only">Engine log</h2>
      <div class="log-toolbar">
        <div class="log-filters" role="group" aria-label="Log categories"><button class="key key-toggle active" data-filter="all" aria-pressed="true">All</button><button class="key key-toggle" data-filter="req" aria-pressed="false">Requests</button><button class="key key-toggle" data-filter="throughput" aria-pressed="false">Speed</button><button class="key key-toggle" data-filter="warn" aria-pressed="false">Warnings</button></div>
        <input type="search" id="log-search" class="cdu-input" placeholder="Search this log" aria-label="Search engine log">
        <button id="btn-autoscroll" class="key key-toggle active" aria-pressed="true">Follow latest</button>
        <button id="btn-copylog" class="key"><svg class="icon"><use href="#i-copy"/></svg>Copy log</button>
      </div>
      <div id="log-content" class="log-pane" tabindex="0" aria-label="Engine log output">Waiting for engine activity…</div>
    </section>

    <section class="diag-page" data-diag-page="engine" aria-label="Engine data" hidden>
      <dl class="cdu-facts">
        <div><dt>Process state</dt><dd id="kpi-state-main">—</dd></div>
        <div><dt>Process ID</dt><dd id="kpi-pid">—</dd></div>
        <div><dt>API port</dt><dd id="kpi-engine-port">—</dd></div>
        <div><dt>Health response</dt><dd id="dt-health-check">—</dd></div>
        <div><dt>Last engine event</dt><dd id="dt-last-event">—</dd></div>
)HTML"
R"HTML(        <div><dt>Memory reserved for your desktop</dt><dd id="dt-reserve">—</dd></div>
        <div><dt>Engine memory plan</dt><dd id="dt-reservation">—</dd></div>
        <div><dt>KV capacity</dt><dd id="dt-kv-capacity">—</dd></div>
      </dl>
      <div id="launch-plan" class="launch-plan" hidden></div>
      <h2 class="cdu-subhead">Startup capacity report</h2>
      <p class="cdu-small">The engine's own memory and request-capacity report from this start.</p>
      <pre id="dt-capacity-text" class="cdu-pre">No capacity report is available yet.</pre>
    </section>

    <section class="diag-page" data-diag-page="accel" aria-label="Generation acceleration" hidden>
      <p class="cdu-help">Speculative decoding drafts tokens ahead and checks them against the model. It is optional; a disabled backend is not an error.</p>
      <dl class="cdu-facts">
        <div><dt>Backend</dt><dd id="dt-mtp-backend">—</dd></div>
        <div><dt>Status</dt><dd id="kpi-mtp-badge">—</dd></div>
        <div><dt>Draft window</dt><dd id="dt-mtp-window">—</dd></div>
        <div><dt>Acceptance rate</dt><dd id="kpi-mtp-pct">—</dd></div>
        <div><dt>Accepted / drafted</dt><dd id="kpi-mtp-ratio">—</dd></div>
        <div><dt>Fallback steps</dt><dd id="kpi-mtp-fallbacks">—</dd></div>
        <div><dt>Prefix reuse</dt><dd id="dt-reuse-mix">—</dd></div>
      </dl>
      <h2 class="cdu-subhead">Accepted tokens by draft position <span id="dt-mtp-rate-badge" class="cdu-tag">—</span></h2>
      <div id="mtp-bars-container" class="mtp-bars"><div class="mtp-col"><div class="mtp-track"><div class="mtp-bar"></div></div><span>P1</span></div><div class="mtp-col"><div class="mtp-track"><div class="mtp-bar"></div></div><span>P2</span></div><div class="mtp-col"><div class="mtp-track"><div class="mtp-bar"></div></div><span>P3</span></div><div class="mtp-col"><div class="mtp-track"><div class="mtp-bar"></div></div><span>P4</span></div><div class="mtp-col"><div class="mtp-track"><div class="mtp-bar"></div></div><span>P5</span></div></div>
      <p class="cdu-small">Conversation-prefix reuse can avoid processing the same opening text again. Seed, append and reset describe how the engine reused that state.</p>
    </section>

    <section class="diag-page diag-experts" data-diag-page="experts" aria-labelledby="experts-title" hidden>
      <h2 id="experts-title" class="sr-only">Expert use</h2>
      <p class="cdu-help explain-only">Each text layer of the model holds a pool of experts, small sub-networks, and every token is sent to only the few best suited in each layer. This map shows how busy each expert has been lately: brighter is busier, scaled inside its own layer, and the counts fade with the half-life shown so the picture follows what the engine is doing now. Decode is the engine writing answers, including the speculative drafts and their checks; Prefill is it reading prompts. By expert keeps every expert in its own column, so a bright column is an expert that stays busy. Sorted lines each layer up from busiest to quietest, so the slope shows how lopsided the use is: a steep fall means a few experts do most of the work. The bars at the right give, per layer, how many experts it takes to cover 80% of that layer's activity.</p>
      <div id="ex-live" class="ex-live" hidden>
        <div class="ex-bar">
          <div class="ex-tabs" role="group" aria-label="Activity shown">
            <button type="button" class="cdu-tab" data-ex-kind="decode" aria-pressed="true">Decode</button>
            <button type="button" class="cdu-tab" data-ex-kind="prefill" aria-pressed="false">Prefill</button>
          </div>
          <div class="ex-tabs" role="group" aria-label="Order of experts">
            <button type="button" class="cdu-tab" data-ex-order="id" aria-pressed="true">By expert</button>
            <button type="button" class="cdu-tab" data-ex-order="sorted" aria-pressed="false">Sorted</button>
          </div>
          <span class="ex-scale" aria-hidden="true"><span>0</span><canvas id="ex-key" width="96" height="10"></canvas><span>Row peak</span></span>
          <output id="ex-read" class="ex-read" aria-live="off"></output>
        </div>
        <dl class="ex-facts" aria-label="Counters">
          <div><dt>Decode</dt><dd id="ex-f-decode">—</dd></div>
          <div><dt>Prefill</dt><dd id="ex-f-prefill">—</dd></div>
          <div id="ex-f-draft-wrap"><dt>Draft</dt><dd id="ex-f-draft">—</dd></div>
          <div id="ex-f-verify-wrap"><dt>Verify</dt><dd id="ex-f-verify">—</dd></div>
          <div><dt>Up</dt><dd id="ex-f-up">—</dd></div>
          <div><dt>Half-life</dt><dd id="ex-f-half">—</dd></div>
          <div><dt>Trace</dt><dd id="ex-f-trace">—</dd></div>
          <span id="ex-stale" class="ex-tag" hidden></span>
        </dl>
        <p class="note">Steps are passes of the model over a batch of tokens and tokens are the positions in them; the counts are since the engine started. Draft is the cheap speculative step that guesses ahead and Verify is the check of its guesses; both feed the Decode map. The trace file records every routing choice for later analysis; dropped means the writer fell behind and skipped some records.</p>
        <div id="ex-map" class="ex-map">
          <canvas id="ex-canvas" tabindex="0" role="img" aria-label="Expert use map. Rows are layers and columns are experts. Use the arrow keys to move between experts and layers, Home and End to jump along a layer, Escape to clear."></canvas>
          <div id="ex-idle" class="trace-empty" hidden>No expert activity in the recent window</div>
        </div>
        <p id="ex-skew" class="ex-skew">
          <span class="ex-lg">Experts for 80%</span>
          <span><i>min</i><b id="ex-sk-min">—</b></span>
          <span><i>median</i><b id="ex-sk-med">—</b></span>
          <span><i>max</i><b id="ex-sk-max">—</b></span>
          <span class="ex-sub">of <span id="ex-sk-of">512</span> per text layer</span>
          <span id="ex-sk-mtp-wrap"><i>MTP layer</i><b id="ex-sk-mtp">—</b></span>
        </p>
        <div class="ex-budget">
          <div class="ex-budget-set">
            <label class="cdu-label" for="ex-n">Expert VRAM budget<span id="ex-budget-of" class="ex-budget-of">of all 24,576 text-layer experts · 63.3 GiB</span></label>
            <div class="ex-budget-row">
              <input id="ex-slider" class="reserve-slider ex-slider" type="range" min="0" max="24576" step="48" value="8192" aria-label="Expert budget in experts" aria-describedby="ex-budget-note">
              <input id="ex-n" class="cdu-input cdu-input-num" type="number" min="0" max="24576" step="48" value="8192" inputmode="numeric"><span class="cdu-unit">experts</span>
              <input id="ex-gib" class="cdu-input cdu-input-num" type="number" min="0" max="63.3" step="0.1" value="21.1" inputmode="decimal" aria-label="Expert budget in GiB"><span class="cdu-unit">GiB</span>
            </div>
          </div>
          <dl class="ex-out">
            <div class="ro ro-major"><dt id="ex-dt-global">Hottest experts anywhere</dt><dd><b id="ex-share-global">—</b></dd></div>
            <div class="ro ro-major"><dt id="ex-dt-even">Even per layer</dt><dd><b id="ex-share-even">—</b></dd></div>
          </dl>
          <p id="ex-budget-note" class="cdu-small ex-budget-note">Share of recent decode activations those experts hold: an upper bound for a fixed set chosen from today's traffic, not a measured hit rate.</p>
        </div>
        <p class="note">The budget asks how much of the recent decode work a limited set of experts would cover if only they lived in GPU memory. Hottest anywhere picks the busiest experts across all layers; Even per layer gives every layer the same number. Both are best cases for a fixed set picked from the traffic on screen, so a cache that must adapt does worse. The MTP layer's experts are left out of the budget. In Sorted order the magenta marks show where each choice would cut: a line per layer for the global pick, a dashed line for the even one.</p>
      </div>
      <div id="ex-state" class="ex-state" role="status">
        <h3 id="ex-state-title" class="cdu-subhead">Reading expert use…</h3>
        <p id="ex-state-text" class="cdu-help">Asking the engine for its expert counters.</p>
        <p id="ex-state-how" class="cdu-small" hidden></p>
        <a id="ex-state-link" class="prompt" href="#settings/raw" hidden>Launch details<span aria-hidden="true">&gt;</span></a>
      </div>
    </section>
  </div>
</section>

<section id="view-settings" class="view view-cdu" aria-labelledby="settings-title" hidden>
  <div class="cdu cdu-wide" id="settings-cdu">
    <header class="cdu-title"><h1 id="settings-title">Settings</h1><span id="settings-tmpy" class="cdu-tmpy" hidden>Temporary</span><span id="cfg-path" class="cdu-page cdu-file">Loading…</span></header>

    <section class="settings-page" data-settings-page="index" aria-label="Settings index">
      <p class="cdu-help">Choose a page. Changes are staged as a temporary revision: Insert saves it, Erase discards it. Engine settings take effect the next time the engine starts.</p>
      <ul class="lsk-grid" id="settings-index">
        <li><a class="lsk lsk-l" href="#settings/network"><span class="lsk-prompt"><span aria-hidden="true">&lt;</span>Connection</span><span class="lsk-sum" id="sum-network">—</span></a></li>
        <li><a class="lsk lsk-r" href="#settings/capacity"><span class="lsk-prompt">Request capacity<span aria-hidden="true">&gt;</span></span><span class="lsk-sum" id="sum-capacity">—</span></a></li>
        <li><a class="lsk lsk-l" href="#settings/api"><span class="lsk-prompt"><span aria-hidden="true">&lt;</span>API &amp; access</span><span class="lsk-sum" id="sum-api">—</span></a></li>
        <li><a class="lsk lsk-r" href="#settings/memory"><span class="lsk-prompt">Memory &amp; precision<span aria-hidden="true">&gt;</span></span><span class="lsk-sum" id="sum-memory">—</span></a></li>
        <li><a class="lsk lsk-l" href="#settings/frontends"><span class="lsk-prompt"><span aria-hidden="true">&lt;</span>Web frontends</span><span class="lsk-sum" id="sum-frontends">—</span></a></li>
        <li><a class="lsk lsk-r" href="#settings/features"><span class="lsk-prompt">Model features<span aria-hidden="true">&gt;</span></span><span class="lsk-sum" id="sum-features">—</span></a></li>
        <li><a class="lsk lsk-l" href="#settings/presets"><span class="lsk-prompt"><span aria-hidden="true">&lt;</span>Model presets</span><span class="lsk-sum" id="sum-presets">—</span></a></li>
        <li><a class="lsk lsk-r" href="#settings/reserve"><span class="lsk-prompt">Desktop reserve<span aria-hidden="true">&gt;</span></span><span class="lsk-sum" id="sum-reserve">—</span></a></li>
        <li><a class="lsk lsk-l" href="#settings/raw"><span class="lsk-prompt"><span aria-hidden="true">&lt;</span>Launch details</span><span class="lsk-sum" id="sum-raw">read-only</span></a></li>
      </ul>
    </section>

    <section class="settings-page" data-settings-page="config" aria-labelledby="settings-section-title" hidden>
      <header class="page-head"><a class="prompt" href="#settings"><span aria-hidden="true">&lt;</span>Index</a><h2 id="settings-section-title">Connection</h2></header>
      <p id="settings-section-help" class="cdu-help">Addresses and ports used by the engine and this dashboard.</p>
      <div id="cfg-body" class="cfg-grid"><p class="cdu-small">Loading settings…</p></div>
      <div id="cfg-errors" class="cfg-errors" role="alert" hidden></div>
      <div class="cfg-actions">
        <button id="cfg-revert" class="key" disabled><span aria-hidden="true">&lt;</span>Erase</button>
        <span id="cfg-status" class="cfg-status" role="status">No unsaved changes</span>
        <button id="cfg-save" class="key key-insert" disabled>Insert<span aria-hidden="true">&gt;</span></button>
)HTML"
R"HTML(        <button id="cfg-save-restart" class="key" disabled>Insert + restart<span aria-hidden="true">&gt;</span></button>
      </div>
    </section>

    <section class="settings-page" data-settings-page="presets" id="model-catalog" aria-labelledby="catalog-title" hidden>
      <header class="page-head"><a class="prompt" href="#settings"><span aria-hidden="true">&lt;</span>Index</a><h2 id="catalog-title">Model presets</h2><button id="catalog-refresh" class="key key-small">Refresh</button></header>
      <p id="catalog-current" class="cdu-help">Checking the current model…</p>
      <fieldset id="catalog-options" class="presets" aria-label="Model to load"></fieldset>
      <p class="cdu-small">Artifact size is disk space, not GPU memory use. A file can be available and still need different launch settings.</p>
      <div class="cfg-actions">
        <span id="catalog-hint" class="cfg-status"></span>
        <button id="model-switch" class="key key-insert" disabled>Load and restart<span aria-hidden="true">&gt;</span></button>
      </div>
      <p id="catalog-status" class="pad-line" role="status" hidden></p>
    </section>

    <section class="settings-page" data-settings-page="reserve" id="desktop-reserve" aria-labelledby="reserve-title" hidden>
      <header class="page-head"><a class="prompt" href="#settings"><span aria-hidden="true">&lt;</span>Index</a><h2 id="reserve-title">Desktop reserve</h2></header>
      <p class="cdu-help">How much GPU memory NInfer leaves free when it starts, on top of what other apps already use. More room helps your desktop and GPU-heavy apps but leaves less for the model's conversations. It is a startup target, not memory locked away.</p>
      <div class="reserve-layout">
        <div class="reserve-entry">
          <label class="cdu-label" for="reserve-amount">Free memory target</label>
          <span class="cdu-field-row"><input id="reserve-amount" class="cdu-input cdu-input-num" type="number" min="1" max="64" step="1" value="8" aria-describedby="reserve-preview-note"><span class="cdu-unit">GiB</span></span>
          <input id="reserve-slider" class="reserve-slider" type="range" min="1" max="64" step="1" value="8" aria-label="Free GPU memory target in GiB" aria-describedby="reserve-preview-note">
          <div class="reserve-scale" aria-hidden="true"><span>More engine capacity</span><span>More room for apps</span></div>
          <div class="reserve-preview" aria-hidden="true"><span id="reserve-engine-preview"></span><span id="reserve-free-preview"></span></div>
        </div>
        <div class="reserve-facts">
          <p id="reserve-current" class="cdu-data">Checking the running reserve…</p>
          <p id="reserve-preview-note" class="cdu-small"></p>
          <p id="reserve-status" class="cdu-data" role="status"></p>
        </div>
      </div>
      <div class="cfg-actions">
        <button class="key" id="reserve-discard" hidden><span aria-hidden="true">&lt;</span>Erase</button>
        <button class="key" id="reserve-default">Engine default</button>
        <button class="key key-insert" id="reserve-save" disabled>Insert for next start<span aria-hidden="true">&gt;</span></button>
      </div>
    </section>
  </div>
</section>
</main>

<dialog id="confirm-modal" aria-labelledby="modal-title" aria-describedby="modal-desc"><h2 id="modal-title">Confirm action</h2><p id="modal-desc"></p><div class="dialog-keys"><button id="modal-cancel-btn" class="key"><span aria-hidden="true">&lt;</span>Cancel</button><button id="modal-confirm-btn" class="key key-insert">Confirm<span aria-hidden="true">&gt;</span></button></div></dialog>
<div id="a11y-live-region" class="sr-only" aria-live="polite"></div>
<script>(function() {
  'use strict';

  // ---------------------------------------------------------------------
  // Formatting
  // ---------------------------------------------------------------------
  const GiB = 1073741824, MiB = 1048576;
  function gib(bytes) {
    if (bytes == null || isNaN(bytes)) return '—';
    return (bytes / GiB).toFixed(2) + ' GiB';
  }
  function mib(bytes) {
    if (bytes == null || isNaN(bytes)) return '—';
    return (bytes / MiB).toFixed(1) + ' MiB';
  }
  function appBytes(bytes) {
    if (bytes >= GiB) return (bytes / GiB).toFixed(2) + ' GiB';
    return (bytes < MiB ? '<1' : (bytes / MiB).toFixed(0)) + ' MiB';
  }
  function formatUptime(seconds) {
    if (seconds == null || isNaN(seconds)) return '—';
    const h = Math.floor(seconds / 3600), m = Math.floor((seconds % 3600) / 60), s = Math.floor(seconds % 60);
    if (h > 0) return `${h}h ${m}m`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
  }
  function esc(str) {
    return String(str == null ? '' : str).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }
  function shortTokens(n) {
    n = Number(n);
    if (!Number.isFinite(n)) return '—';
    if (n >= 1048576 && n % 1048576 === 0) return (n / 1048576) + 'M';
    if (n >= 1024 && n % 1024 === 0) return (n / 1024) + 'K';
    return n.toLocaleString();
  }
  function rate(v) {
    if (!Number.isFinite(v)) return '—';
    return v >= 10000 ? (v / 1000).toFixed(1) + 'k' : Math.round(v).toLocaleString();
  }
  function duration(value) {
    return Number.isFinite(value) ? (value >= 1000 ? `${(value / 1000).toFixed(1)} s` : `${Math.round(value)} ms`) : '—';
  }
  const $ = id => document.getElementById(id);

  // ---------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------
  let lastState = null;
  let connected = false, actionBusy = false, configBusy = false;
  let modelBusy = false, modelCatalog = null, modelSelection = '', catalogLoading = false;
  let catalogEpoch = 0, modelSwitchStarted = 0, modelUncertain = false;
  let activeView = 'overview', diagPage = 'findings', settingsPage = 'index';
  let isDocumentVisible = !document.hidden;
  let autoScroll = true, activeLogFilter = 'all', logSearchQuery = '', rawLogTail = '';
  let lastInsightsJson = '';
  let launchFlags = {};
  const loadedAt = Date.now();
  let cfgData = null, cfgTab = 'network';
  const cfgEdits = {params: {}, engine: {}, supervisor: {}};
  let feDraft = null, feBusy = false;
  let pendingAction = null, lastFocusedTrigger = null;
  let pendingRestartPid = null;
  const expandedClients = new Set();
  let gpuAppsExpanded = false;

  const liveDot = $('live-dot'), liveLabel = $('live-label');
  const btnStart = $('btn-start'), btnStop = $('btn-stop'), btnRestart = $('btn-restart');
  const btnAutoscroll = $('btn-autoscroll'), btnCopyLog = $('btn-copylog');
  const logContent = $('log-content'), logSearch = $('log-search');
  const a11yLive = $('a11y-live-region');
  const confirmModal = $('confirm-modal'), modalTitle = $('modal-title'), modalDesc = $('modal-desc');
  const modalCancelBtn = $('modal-cancel-btn'), modalConfirmBtn = $('modal-confirm-btn');
  const catalogOptions = $('catalog-options'), modelSwitch = $('model-switch'), catalogRefresh = $('catalog-refresh');

  // ---------------------------------------------------------------------
  // Annunciation: a value that changes is boxed for ten seconds so a glance
  // catches the change itself, not only the new state.
  // ---------------------------------------------------------------------
  const annunciateTimers = new WeakMap();
  function annunciate(el, text, cls) {
    if (!el) return;
    if (cls !== undefined) el.className = el.className.split(' ').filter(c => !['green','amber','red','dim','white'].includes(c)).concat(cls ? [cls] : []).join(' ');
    if (el.textContent === text) return;
    const hadText = el.dataset.seen === '1';
    el.textContent = text;
    el.dataset.seen = '1';
    if (!hadText || Date.now() - loadedAt < 5000) return;
    el.classList.add('annunciated');
    clearTimeout(annunciateTimers.get(el));
    annunciateTimers.set(el, setTimeout(() => el.classList.remove('annunciated'), 10000));
  }

  // ---------------------------------------------------------------------
  // Explain: plain-language notes on demand.
  // ---------------------------------------------------------------------
  const explainToggle = $('explain-toggle');
  function setExplain(on) {
    document.body.classList.toggle('explain', on);
    explainToggle.setAttribute('aria-pressed', String(on));
    try { localStorage.setItem('ninfer:explain', on ? '1' : '0'); } catch (_) {}
    requestAnimationFrame(drawAll);
  }
  explainToggle.onclick = () => setExplain(!document.body.classList.contains('explain'));
  try { setExplain(localStorage.getItem('ninfer:explain') === '1'); } catch (_) { setExplain(false); }

  // ---------------------------------------------------------------------
  // Display mode: Auto follows the Windows app theme; Day and Night stay put.
  // ---------------------------------------------------------------------
  const displayKey = $('display-mode');
  const lightQuery = window.matchMedia ? window.matchMedia('(prefers-color-scheme: light)') : null;
  const displayNames = {auto: 'Auto', day: 'Day', night: 'Night'};
  function applyDisplay(mode) {
    const r = document.documentElement;
    r.dataset.display = mode;
    r.dataset.theme = mode === 'auto' ? (lightQuery && lightQuery.matches ? 'day' : 'night') : mode;
    $('display-mode-label').textContent = mode === 'auto' ? 'Auto · ' + displayNames[r.dataset.theme] : displayNames[mode];
    displayKey.setAttribute('aria-label', `Display mode: ${displayNames[mode]}${mode === 'auto' ? ', following Windows (' + displayNames[r.dataset.theme] + ')' : ''}. Press to change.`);
    readPalette();
    requestAnimationFrame(() => { drawAll(); if (ex.frame) { exDrawKey(exKind(ex.frame)); exDraw(); } });
  }
  displayKey.onclick = () => {
    const next = {auto: 'day', day: 'night', night: 'auto'}[document.documentElement.dataset.display] || 'auto';
    try { localStorage.setItem('ninfer:display', next); } catch (_) {}
    applyDisplay(next);
  };
  if (lightQuery) lightQuery.addEventListener('change', () => { if (document.documentElement.dataset.display === 'auto') applyDisplay('auto'); });

  // ---------------------------------------------------------------------
  // Scratchpad messages
  // ---------------------------------------------------------------------
  function notify(message, error = false) {
    const line = $('action-status-line');
    $('action-status').textContent = message;
    line.className = 'pad-line pad-action' + (error ? ' error' : '');
    line.hidden = !message;
  }
  $('action-clear').onclick = () => { $('action-status-line').hidden = true; };
  function catalogMessage(message, error = false) {
    const el = $('catalog-status');
    el.textContent = message;
    el.className = 'pad-line' + (error ? ' error' : '');
    el.hidden = !message;
  }

  // ---------------------------------------------------------------------
  // Routing: four function keys, each with its own pages.
  // ---------------------------------------------------------------------
  const viewNames = ['overview', 'chat', 'connect', 'api', 'diagnostics', 'settings'];
  const diagPages = ['findings', 'log', 'engine', 'accel', 'experts'];
  const settingsPages = ['network', 'api', 'frontends', 'capacity', 'memory', 'features', 'raw', 'presets', 'reserve'];
  const viewLabels = {overview:'Monitor', chat:'Chat', connect:'Connect', api:'API test bench', diagnostics:'Diagnose', settings:'Settings'};
  const diagLabels = {findings:'Findings', log:'Engine log', engine:'Engine data', accel:'Acceleration', experts:'Experts'};
  const tabLabels = {network:'Connection', api:'API & access', frontends:'Web frontends', capacity:'Request capacity', memory:'Memory & precision', features:'Model features', raw:'Launch details', presets:'Model presets', reserve:'Desktop reserve'};

  function selectView() {
    if (location.hash === '#main') { $('main').focus(); return; }
)HTML"
R"HTML(    const [head, sub] = location.hash.slice(1).split('/');
    activeView = viewNames.includes(head) ? head : 'overview';
    document.querySelectorAll('.view').forEach(el => { el.hidden = el.id !== 'view-' + activeView; });
    document.querySelectorAll('[data-view]').forEach(el => {
      if (el.dataset.view === activeView) el.setAttribute('aria-current', 'page'); else el.removeAttribute('aria-current');
    });
    document.body.classList.toggle('on-monitor', activeView === 'overview');
    let title = viewLabels[activeView];
    if (activeView === 'diagnostics') {
      diagPage = diagPages.includes(sub) ? sub : 'findings';
      document.querySelectorAll('[data-diag-page]').forEach(el => { el.hidden = el.dataset.diagPage !== diagPage; });
      document.querySelectorAll('[data-diag]').forEach(el => {
        if (el.dataset.diag === diagPage) el.setAttribute('aria-current', 'page'); else el.removeAttribute('aria-current');
      });
      $('diag-page-label').textContent = diagLabels[diagPage];
      title += ' · ' + diagLabels[diagPage];
      if (diagPage === 'log' && autoScroll) logContent.scrollTop = logContent.scrollHeight;
    }
    if (activeView === 'settings') {
      settingsPage = settingsPages.includes(sub) ? sub : 'index';
      const kind = settingsPage === 'index' ? 'index' : settingsPage === 'presets' ? 'presets' : settingsPage === 'reserve' ? 'reserve' : 'config';
      document.querySelectorAll('[data-settings-page]').forEach(el => { el.hidden = el.dataset.settingsPage !== kind; });
      if (kind === 'config') { cfgTab = settingsPage; renderConfig(); }
      if (kind === 'presets' && !modelCatalog && !catalogLoading) refreshModelCatalog();
      if (settingsPage !== 'index') title += ' · ' + tabLabels[settingsPage];
      updateTmpy();
    }
    if (activeView === 'api') benchInit();
    if (activeView === 'chat') chatInit();
    exSync();
    document.title = title + ' · NInfer';
    if (activeView === 'overview') requestAnimationFrame(drawAll);
  }
  window.addEventListener('hashchange', () => {
    setPendingRestart(pendingRestartPid);
    selectView();
    $('main').focus({preventScroll: true});
    $('main').scrollTop = 0;
  });

  // ---------------------------------------------------------------------
  // Connection and engine controls
  // ---------------------------------------------------------------------
  function setConnection(value) {
    connected = value;
    liveDot.className = 'lamp' + (value ? '' : ' offline');
    liveLabel.textContent = value ? 'Live' : 'Reconnecting…';
    $('connection-alert').hidden = value;
    updateEngineControls();
  }
  function managedEngine() {
    return cfgData ? cfgData.manages_engine !== false : !(lastState || {}).monitor_only;
  }
  function updateEngineControls() {
    const engine = (lastState || {}).engine || {};
    const managed = managedEngine();
    const changing = ['Starting', 'Stopping', 'BackingOff'].includes(engine.state);
    $('controls').hidden = !managed;
    $('monitor-note').hidden = managed;
    btnStart.hidden = engine.state === 'Running' || engine.state === 'Stopping';
    btnStop.hidden = engine.state === 'Stopped' || engine.state === 'Halted';
    btnRestart.hidden = engine.state !== 'Running';
    btnStart.disabled = !connected || !managed || actionBusy || modelBusy || modelUncertain || changing;
    btnStop.disabled = !connected || !managed || actionBusy || modelBusy || modelUncertain || engine.state === 'Stopping';
    btnRestart.disabled = !connected || !managed || actionBusy || modelBusy || modelUncertain || changing;
    updateModelControls();
  }
  async function copyText(button, value, labelEl) {
    const target = labelEl || button;
    const before = target.textContent;
    try {
      await navigator.clipboard.writeText(value);
      target.textContent = 'Copied';
      setTimeout(() => { target.textContent = before; }, 1600);
    } catch (_) { notify('Could not copy. Select the text and copy it manually.', true); }
  }
  $('copy-address').onclick = e => copyText(e.currentTarget, $('api-address').value, e.currentTarget.lastChild);
  $('fma-api-copy').onclick = () => copyText($('fma-api-copy'), $('api-address').value, $('overview-endpoint'));

  try { pendingRestartPid = JSON.parse(sessionStorage.getItem('ninfer:pending-restart') || 'null'); } catch (_) {}
  function setPendingRestart(pid) {
    pendingRestartPid = pid;
    try { sessionStorage.setItem('ninfer:pending-restart', JSON.stringify(pid)); } catch (_) {}
    $('connection-pending').hidden = pid == null;
    $('overview-connection-pending').hidden = pid == null;
    updateArmed();
  }

  // ---------------------------------------------------------------------
  // Running plan: the flags the process was launched with, after every
  // supervisor adjustment, and the engine's own capacity report. A reduced KV
  // plan is the one compromise the supervisor makes on its own.
  // ---------------------------------------------------------------------
  function renderLaunchPlan(eng, s) {
    const el = $('launch-plan');
    const args = Array.isArray(eng.launch_args) ? eng.launch_args : [];
    const flags = {};
    for (let i = 0; i < args.length; i++) {
      if (!args[i].startsWith('--')) continue;
      const next = args[i + 1];
      if (next !== undefined && !next.startsWith('--')) { flags[args[i]] = next; i++; } else flags[args[i]] = true;
    }
    launchFlags = flags;
    if (!args.length) {
      el.hidden = true;
      annunciate($('plan-mode'), '—', 'dim');
      $('plan-brief').textContent = 'No launch details yet';
      return;
    }
    const cap = {};
    (s.engine_capacity_line || '').split(' ').forEach(tok => { const eq = tok.indexOf('='); if (eq > 0) cap[tok.slice(0, eq)] = tok.slice(eq + 1); });
    const num = v => Number.isFinite(Number(v)) ? Number(v).toLocaleString() : (v || '—');
    const gibv = v => Number.isFinite(Number(v)) ? (Number(v) / GiB).toFixed(2) + ' GiB' : '—';
    const reduced = eng.kv_capacity_effective > 0;
    const kvRunning = cap.kv_capacity_tokens || (reduced ? eng.kv_capacity_effective : flags['--kv-capacity']);
    const spec = flags['--spec'] ? `${flags['--spec']}${flags['--draft-tokens'] ? ' K' + flags['--draft-tokens'] : ''}` : 'off';
    annunciate($('plan-mode'), reduced ? 'Reduced by supervisor' : 'As configured', reduced ? 'amber' : 'green');
    const brief = [kvRunning ? 'KV ' + shortTokens(kvRunning) : null, flags['--max-context'] ? 'ctx ' + shortTokens(flags['--max-context']) : null,
      (flags['--max-concurrency'] || 1) + ' lane' + (Number(flags['--max-concurrency'] || 1) === 1 ? '' : 's'), 'spec ' + spec, flags['--kv-dtype'] ? 'KV ' + flags['--kv-dtype'] : null].filter(Boolean);
    $('plan-brief').textContent = brief.join(' · ');
    $('plan-brief').title = reduced ? (eng.kv_capacity_note || 'The configured KV capacity did not fit beside other apps on the card.') : '';
    const rows = [
      ['KV capacity', reduced ? `${num(kvRunning)} of ${num(eng.kv_capacity_configured)} configured` : `${num(kvRunning)} tokens`],
      ['Max context', num(flags['--max-context'])],
      ['Concurrent requests', num(flags['--max-concurrency'])],
      ['Speculation', flags['--spec'] ? `${flags['--spec']}${flags['--draft-tokens'] ? ', ' + flags['--draft-tokens'] + ' drafts' : ''}` : 'off'],
      ['KV cache', flags['--kv-dtype'] || 'default'],
      ['Desktop reserve', flags['--desktop-reserve-gib'] ? flags['--desktop-reserve-gib'] + ' GiB' : 'engine default'],
      ['Engine reservation', gibv(cap.runtime_reservation_bytes)],
      ['Left after startup', gibv(cap.available_after_startup_bytes)],
    ];
    let html = '<h2 class="cdu-subhead">Running plan <span class="cdu-tag ' + (reduced ? 'amber' : 'green') + '">' + (reduced ? 'Reduced by supervisor' : 'As configured') + '</span></h2><dl>';
    rows.forEach(([k, v]) => { html += `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`; });
    if (reduced) html += `<div class="plan-compromise"><strong>Compromise:</strong> ${esc(eng.kv_capacity_note || 'the configured KV capacity did not fit beside other apps on the card.')}</div>`;
    html += '</dl><h2 class="cdu-subhead">Exact launch command</h2><code>' + esc(args.join(' ')) + '</code>';
    el.innerHTML = html;
    el.hidden = false;
  }

  function renderFrontends(list) {
    const target = $('frontend-list');
    if (!list.length) {
      const none = document.createElement('li');
      none.textContent = 'No web frontend is configured yet.';
      target.replaceChildren(none);
      return;
    }
    const reasons = {port_in_use: 'port in use', assets_missing: 'files not found', stopped: 'stopped'};
    target.replaceChildren(...list.map(fe => {
      const li = document.createElement('li');
      if (fe.state === 'serving') {
        const link = document.createElement('a');
        link.href = fe.url; link.target = '_blank'; link.rel = 'noopener'; link.className = 'prompt';
        link.textContent = 'Open ' + fe.name;
        const arrow = document.createElement('span'); arrow.setAttribute('aria-hidden', 'true'); arrow.textContent = '>';
        link.append(arrow);
        li.append(link);
      } else {
        const name = document.createElement('strong'); name.textContent = fe.name;
        const why = document.createElement('span'); why.className = 'fe-down';
        const whyText = fe.reason || reasons[fe.state] || fe.state;
        why.textContent = whyText.charAt(0).toUpperCase() + whyText.slice(1);
        li.append(name, why);
      }
      const url = document.createElement('code'); url.textContent = fe.url;
      li.append(url);
      return li;
    }));
  }

  // ---------------------------------------------------------------------
  // Model presets
  // ---------------------------------------------------------------------
  function modelBlockReason() {
    if (modelBusy) return 'Switch in progress. Other engine changes are paused.';
    if (modelUncertain) return 'The switch result is unknown. Verify completion before reloading this page to enable changes.';
    if (catalogLoading) return 'Checking available files…';
    if (!connected) return 'Reconnect to the supervisor before switching.';
    if (!cfgData || cfgData.manages_engine === false || (lastState || {}).monitor_only) return 'Switch models in the application that starts this engine.';
    if (!cfgData.writable) return 'The supervisor configuration is read-only.';
    if (configBusy || reserveBusy || actionBusy) return 'Wait for the current change to finish.';
    if (cfgDirtyCount() || reserveDraft != null) return 'Insert or erase your staged settings before switching.';
    if (['Starting', 'Stopping', 'BackingOff'].includes(((lastState || {}).engine || {}).state)) return 'Wait for the engine to finish its current transition.';
    return '';
  }
  function updateModelControls() {
    const reason = modelBlockReason();
    const selected = modelCatalog && modelCatalog.models.find(m => m.id === modelSelection);
    modelSwitch.disabled = !!reason || !selected || !selected.available || selected.active;
    catalogOptions.disabled = modelBusy || catalogLoading || !modelCatalog;
    catalogRefresh.disabled = modelBusy || catalogLoading;
    $('catalog-hint').textContent = reason || (selected && selected.active ? 'This preset is the current launch.' : selected ? 'Loading a preset restarts the engine and interrupts active requests.' : 'Choose a preset.');
    $('catalog-hint').className = 'cfg-status' + (selected && !selected.active && !reason ? ' dirty' : '');
    updateTmpy();
  }
  function modelDisplayName(engine) {
    const active = modelCatalog && modelCatalog.models.find(m => m.active);
    if (active) return active.name || active.id;
    if (modelCatalog && modelCatalog.active_artifact_label) return modelCatalog.active_artifact_label;
    const artifact = engine && engine.artifact;
)HTML"
R"HTML(    return artifact ? artifact.split(/[\\/]/).pop().replace(/\.ninfer$/, '').replace(/_/g, ' ') : 'Local model';
  }
  function renderModelCatalog(data) {
    modelCatalog = data;
    const chosen = data.models.find(m => m.id === modelSelection && m.available);
    if (!chosen) modelSelection = (data.models.find(m => m.active && m.available) || {}).id || '';
    catalogOptions.replaceChildren();
    if (!data.models.length) {
      const none = document.createElement('p'); none.className = 'cdu-small';
      none.textContent = 'No presets are configured. Add model presets to the supervisor configuration to switch models here.';
      catalogOptions.append(none);
    }
    data.models.forEach((model, index) => {
      const label = document.createElement('label');
      label.className = 'preset' + (model.available ? '' : ' unavailable') + (model.active ? ' current' : '') + (model.id === modelSelection ? ' selected' : '');
      const input = document.createElement('input');
      input.type = 'radio'; input.name = 'catalog-model'; input.value = model.id;
      input.checked = model.id === modelSelection; input.disabled = !model.available;
      input.setAttribute('aria-describedby', 'catalog-detail-' + index);
      input.onchange = () => {
        modelSelection = model.id;
        catalogOptions.querySelectorAll('.preset').forEach(p => p.classList.toggle('selected', p.querySelector('input').value === modelSelection));
        updateModelControls();
      };
      const name = document.createElement('span'); name.className = 'preset-name'; name.textContent = model.name || model.id;
      const detail = document.createElement('span'); detail.className = 'preset-detail'; detail.id = 'catalog-detail-' + index;
      detail.textContent = [model.description, (Number.isFinite(model.size_bytes) && model.size_bytes > 0 ? (model.size_bytes / GiB).toFixed(1) + ' GiB on disk' : 'Artifact size unavailable'), model.available ? '' : 'Unavailable: ' + model.reason].filter(Boolean).join(' · ');
      const artifact = document.createElement('code'); artifact.textContent = model.artifact;
      label.append(input, name);
      if (model.active) { const tag = document.createElement('span'); tag.className = 'preset-tag'; tag.textContent = 'Current launch'; label.append(tag); }
      label.append(detail, artifact);
      catalogOptions.append(label);
    });
    // The catalog describes launch args; health and saved active_model do not prove model identity.
    const active = data.models.filter(m => m.active).map(m => m.name || m.id).join(', ');
    $('catalog-current').textContent = 'Current launch: ' + (active || data.active_artifact_label || 'not identified') + '. Select another preset, then load it.';
    $('sum-presets').textContent = active || data.active_artifact_label || 'not identified';
    annunciate($('model-label'), modelDisplayName(cfgData && cfgData.engine));
    if (data.active_artifact_label) $('model-label').title = data.active_artifact_label;
    updateModelControls();
  }
  async function refreshModelCatalog() {
    if (modelBusy || catalogLoading) return;
    const epoch = ++catalogEpoch;
    catalogLoading = true; updateModelControls();
    try {
      const response = await fetch('/api/models');
      if (!response.ok) throw new Error('HTTP ' + response.status);
      const data = await response.json();
      if (epoch === catalogEpoch) renderModelCatalog(data);
    } catch (err) {
      if (epoch === catalogEpoch) {
        modelCatalog = null;
        catalogMessage('Could not refresh presets: ' + err.message + '. Refresh before switching.', true);
      }
    } finally { if (epoch === catalogEpoch) { catalogLoading = false; updateModelControls(); } }
  }
  catalogRefresh.onclick = refreshModelCatalog;
  modelSwitch.onclick = () => {
    if (modelSwitch.disabled) return;
    const selected = modelCatalog.models.find(m => m.id === modelSelection) || {};
    showConfirmModal('model', modelSwitch);
    pendingAction.modelId = modelSelection;
    modalTitle.textContent = 'Load ' + (selected.name || modelSelection) + '?';
    modalDesc.textContent = 'This restarts the engine and interrupts all active requests. Loading can take up to 3 minutes; you can keep using the dashboard while you wait. Saved launch settings take effect.';
    modalConfirmBtn.firstChild.textContent = 'Load and restart';
  };
  async function switchModel(id) {
    if (modelBlockReason()) return;
    modelBusy = true; ++catalogEpoch; modelSwitchStarted = Date.now();
    $('settings-cdu').inert = true;
    updateEngineControls(); renderMemoryControls(lastState || {});
    const progress = () => { const message = 'Loading ' + id + '… ' + Math.floor((Date.now() - modelSwitchStarted) / 1000) + 's elapsed. This can take up to 3 minutes; rollback may take longer. Keep this page open.'; catalogMessage(message); notify(message); };
    progress(); const timer = setInterval(progress, 1000);
    try {
      // Do not impose a browser timeout shorter than the server's 180-second readiness wait.
      const response = await fetch('/api/model', {method:'POST', headers:{'Content-Type':'application/json','X-NInfer-Supervisor':'1'}, body:JSON.stringify({id})});
      const data = await response.json();
      if (Array.isArray(data.models)) renderModelCatalog(data);
      clearInterval(timer);
      let message;
      if (response.ok && data.serving === true) message = 'Now serving ' + data.switched_to + '. Refresh the model list in your connected app.';
      else if (data.rolled_back_to) message = id + ' did not start. ' + (data.serving === true ? 'Still serving ' + data.rolled_back_to + '.' : 'The previous model, ' + data.rolled_back_to + ', did not recover. The engine is not serving; check Diagnose.') + ' ' + (data.error || '');
      else message = 'Could not switch to ' + id + ': ' + (data.error || 'HTTP ' + response.status) + '. Review the configuration and refresh presets before retrying.';
      catalogMessage(message, !response.ok); notify(message, !response.ok);
      // A switch replaces launch args. Refresh the editor's data while mutations
      // are still locked, so later saves use the new model's settings.
      if (data.switched_to || data.rolled_back_to) {
        try {
          const configResponse = await fetch('/api/config');
          if (!configResponse.ok) throw new Error();
          cfgData = await configResponse.json();
          updateConnectionGuide(cfgData); renderConfig();
        } catch (_) {
          modelUncertain = true;
          const more = message + ' Settings could not be refreshed. Reload this page before making further changes.';
          catalogMessage(more, true); notify(more, true);
        }
      }
    } catch (err) {
      modelUncertain = true;
      const message = 'The switch result is unknown: ' + err.message + '. The engine may still be loading. Changes remain paused. Check Diagnose and engine readiness; reload this page only after verifying completion.';
      catalogMessage(message, true); notify(message, true);
    } finally {
      clearInterval(timer); modelBusy = false;
      $('settings-cdu').inert = modelUncertain;
      updateEngineControls(); renderMemoryControls(lastState || {});
      await refreshModelCatalog();
    }
  }

  // ---------------------------------------------------------------------
  // Displays: tapes and traces on one shared time window, read by one cursor.
  // ---------------------------------------------------------------------
  const rateTape = $('rate-tape');
  const memTape = $('memory-tape'), memTrace = $('timeline-canvas'), rateTrace = $('throughput-canvas');
  const memTip = $('chart-tooltip'), rateTip = $('throughput-tooltip');
  // The canvases draw from the same tokens as the page, so a display-mode change
  // repaints the instruments in the new palette.
  const C = {};
  function readPalette() {
    const cs = getComputedStyle(document.documentElement);
    const v = name => cs.getPropertyValue(name).trim();
    Object.assign(C, {ink: v('--ink'), dim: v('--dim'), faint: v('--faint'), rule: v('--rule'), rule2: v('--rule-2'), green: v('--green'), ice: v('--ice'),
      magenta: v('--magenta'), red: v('--red'), amber: v('--amber'), glass: v('--glass'), glass2: v('--glass-2'), tapeBody: v('--tape-body'), tick: v('--tick'), axis: v('--axis'),
      envelope: v('--envelope'), traceFill: v('--trace-fill'), genFill: v('--gen-fill'), reserveLabel: v('--reserve-label'), poleDark: v('--pole-dark')});
  }
  readPalette();
  const PAD_TOP = 8, PAD_BOTTOM = 18;
  // Right gutter of both traces: the prompt-reading scale lives there, clear of the plot.
  const GUTTER = 40;
  let cursorT = null;

  function prep(canvas) {
    const r = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const c = canvas.getContext('2d');
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, r.width, r.height);
    return {c, W: r.width, H: r.height};
  }
  function niceStep(range, px, minPx) {
    const raw = range / Math.max(1, Math.floor(px / minPx));
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    for (const m of [1, 2, 4, 5, 8, 10]) if (m * mag >= raw) return m * mag;
    return 10 * mag;
  }
  function niceMax(v) {
    if (!(v > 0)) return 1;
    const mag = Math.pow(10, Math.floor(Math.log10(v)));
    for (const m of [1, 1.5, 2, 3, 4, 5, 6, 8, 10]) if (m * mag >= v) return m * mag;
    return 10 * mag;
  }
  function timeWindow(s) {
    const a = (s.series || {}).t_ms || [], b = (s.throughput || {}).t_ms || [];
    let t0 = Infinity, t1 = -Infinity;
    if (a.length > 1) { t0 = Math.min(t0, a[0]); t1 = Math.max(t1, a[a.length - 1]); }
    if (b.length > 1) { t0 = Math.min(t0, b[0]); t1 = Math.max(t1, b[b.length - 1]); }
    return Number.isFinite(t0) && t1 > t0 ? {t0, t1} : null;
  }
  function nearest(ts, t) {
    let lo = 0, hi = ts.length - 1;
    if (hi < 0) return -1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (ts[mid] < t) lo = mid; else hi = mid; }
    return Math.abs(ts[lo] - t) <= Math.abs(ts[hi] - t) ? lo : hi;
  }
  function relTime(ms) {
    const s = Math.round(ms / 1000);
    if (s <= 0) return 'now';
    return s >= 120 ? `−${Math.floor(s / 60)}m ${s % 60}s` : `−${s}s`;
  }
  function drawTimeAxis(c, W, H, win, xOf) {
    const span = win.t1 - win.t0;
    const step = niceStep(span / 1000, W, 70) * 1000;
    c.font = '500 11px Bahnschrift, sans-serif';
    c.fillStyle = C.faint; c.textBaseline = 'alphabetic';
    c.strokeStyle = C.axis; c.lineWidth = 1;
    c.beginPath(); c.moveTo(0, H - PAD_BOTTOM + .5); c.lineTo(W, H - PAD_BOTTOM + .5); c.stroke();
    for (let t = win.t1; t >= win.t0; t -= step / 2) {
      const xm = Math.round(xOf(t)) + .5;
      c.beginPath(); c.moveTo(xm, H - PAD_BOTTOM); c.lineTo(xm, H - PAD_BOTTOM + 3); c.stroke();
    }
    for (let t = win.t1; t >= win.t0; t -= step) {
      const x = xOf(t);
      c.strokeStyle = C.rule; c.lineWidth = 1; c.setLineDash([1, 4]);
      c.beginPath(); c.moveTo(Math.round(x) + .5, PAD_TOP); c.lineTo(Math.round(x) + .5, H - PAD_BOTTOM); c.stroke();
      c.setLineDash([]);
      c.strokeStyle = C.axis;
      c.beginPath(); c.moveTo(Math.round(x) + .5, H - PAD_BOTTOM); c.lineTo(Math.round(x) + .5, H - PAD_BOTTOM + 5); c.stroke();
      const label = relTime(win.t1 - t);
      c.textAlign = t === win.t1 ? 'right' : 'center';
      if (x > 18) c.fillText(label, t === win.t1 ? x - 2 : x, H - 5);
    }
  }
  function drawEvents(c, H, ser, win, xOf, W) {
    (ser.events || []).forEach(ev => {
      if (ev.t_ms < win.t0 || ev.t_ms > win.t1) return;
      const x = Math.round(xOf(ev.t_ms)) + .5;
      const col = /up/.test(ev.kind) ? C.green : /stop/.test(ev.kind) ? C.ink : C.amber;
)HTML"
R"HTML(      c.strokeStyle = col; c.globalAlpha = .55; c.lineWidth = 1; c.setLineDash([2, 3]);
      c.beginPath(); c.moveTo(x, PAD_TOP + 6); c.lineTo(x, H - PAD_BOTTOM); c.stroke();
      c.setLineDash([]); c.globalAlpha = 1;
      c.fillStyle = col;
      c.beginPath(); c.moveTo(x - 4, PAD_TOP); c.lineTo(x + 4, PAD_TOP); c.lineTo(x, PAD_TOP + 6); c.closePath(); c.fill();
    });
  }
  function eventNear(ser, t, span) {
    let best = null;
    (ser.events || []).forEach(ev => { const d = Math.abs(ev.t_ms - t); if (d < span * 0.012 && (!best || d < best.d)) best = {d, ev}; });
    return best ? best.ev : null;
  }

  function memoryScale(s) {
    const total = (s.device_memory || {}).total_bytes || 0;
    let max = total;
    const used = (s.series || {}).nvidia_used_bytes || [];
    for (const u of used) if (u > max) max = u;
    return max || 1;
  }
  function reserveFloorBytes(s) {
    return (s.admin_vram && s.admin_vram.desktop_reserve) ? s.admin_vram.desktop_reserve.runtime_floor_bytes || 0 : 0;
  }

  function tapeBody(c, W, plotH) {
    c.fillStyle = C.tapeBody;
    c.fillRect(0, PAD_TOP, W - 1, plotH);
    c.strokeStyle = C.tick; c.lineWidth = 1;
    c.beginPath(); c.moveTo(W - 1.5, PAD_TOP); c.lineTo(W - 1.5, PAD_TOP + plotH); c.stroke();
  }
  // A readout window points at the tape; the last digit sits on a drum that
  // rolls with the value, so the direction of change shows before the digit flips.
  const drum = {shown: null, from: 0, to: 0, start: 0};
  const reducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  // Slew: a pointer, a scale or a figure that changes travels to its new reading over 380 ms
  // on an ease-out, as a damped mechanical pointer does, instead of jumping. Only the travel
  // is drawn in between; every reading at rest is the measured value. Reduced motion jumps.
  const SLEW_MS = 380;
  const slews = new Map();
  let slewFrame = 0;
  const easeOut = p => 1 - Math.pow(1 - p, 3);
  function slew(key, target) {
    if (!Number.isFinite(target)) { slews.delete(key); return target; }
    const now = performance.now();
    const st = slews.get(key);
    if (!st || reducedMotion || document.hidden) { slews.set(key, {from: target, to: target, start: now, v: target}); return target; }
    if (target !== st.to) { st.from = st.v; st.to = target; st.start = now; }
    const p = Math.min(1, (now - st.start) / SLEW_MS);
    st.v = p < 1 ? st.from + (st.to - st.from) * easeOut(p) : st.to;
    if (p < 1 && !slewFrame) slewFrame = requestAnimationFrame(() => { slewFrame = 0; drawAll(); });
    return st.v;
  }
  // The same travel for a figure written as text.
  function countTo(el, target, fmt) {
    if (!Number.isFinite(target)) { el._count = null; el.textContent = '—'; return; }
    const prev = el._count;
    if (!prev || reducedMotion || document.hidden) { el._count = {v: target, to: target}; el.textContent = fmt(target); return; }
    if (prev.to === target) { if (!prev.moving) el.textContent = fmt(target); return; }
    const from = prev.v, start = performance.now();
    const run = el._count = {v: from, to: target, moving: true};
    const step = now => {
      if (el._count !== run) return;
      const p = Math.min(1, (now - start) / SLEW_MS);
      run.v = p < 1 ? from + (target - from) * easeOut(p) : target;
      el.textContent = fmt(run.v);
      if (p < 1) requestAnimationFrame(step); else run.moving = false;
    };
    requestAnimationFrame(step);
  }
  // Tenths shown on the drum: animates 250 ms toward the new reading, then rests
  // on a whole digit so the neighbours show only as clipped edges.
  function drumTenths(value) {
    const target = Math.round(value * 10);
    const now = performance.now();
    if (drum.shown == null || reducedMotion) { drum.shown = drum.from = drum.to = target; return target; }
    if (target !== drum.to) {
      drum.from = drum.shown; drum.to = target; drum.start = now;
      setTimeout(() => { if (lastState && activeView === 'overview') drawMemoryTape(lastState); }, 270);
    }
    const p = Math.min(1, (now - drum.start) / 250);
    drum.shown = drum.from + (drum.to - drum.from) * (1 - Math.pow(1 - p, 3));
    if (p < 1) requestAnimationFrame(() => { if (lastState && activeView === 'overview') drawMemoryTape(lastState); });
    else drum.shown = drum.to;
    return drum.shown;
  }
  function readoutWindow(c, W, y, value, colour, drum) {
    const boxW = W - 9, h = 11;
    c.fillStyle = C.glass; c.strokeStyle = colour; c.lineWidth = 1.25;
    const drumW = drum ? 11 : 0, drumX = boxW - drumW;
    c.beginPath();
    if (drum) {
      c.moveTo(.5, y - h); c.lineTo(drumX, y - h); c.lineTo(drumX, y - h - 7); c.lineTo(boxW, y - h - 7); c.lineTo(boxW, y - 5);
      c.lineTo(boxW + 6, y); c.lineTo(boxW, y + 5); c.lineTo(boxW, y + h + 7); c.lineTo(drumX, y + h + 7); c.lineTo(drumX, y + h); c.lineTo(.5, y + h);
    } else {
      c.moveTo(.5, y - h); c.lineTo(boxW, y - h); c.lineTo(boxW, y - 5); c.lineTo(boxW + 6, y); c.lineTo(boxW, y + 5); c.lineTo(boxW, y + h); c.lineTo(.5, y + h);
    }
    c.closePath(); c.fill(); c.stroke();
    c.font = '600 14px Bahnschrift, sans-serif'; c.textBaseline = 'middle'; c.fillStyle = colour;
    if (!drum) { c.textAlign = 'right'; c.fillText(value, boxW - 4, y + .5); return; }
    const t = drumTenths(value), d = Math.floor(t + 1e-6), f = Math.max(0, t - d);
    c.textAlign = 'right';
    c.fillText(Math.floor((d + 1e-6) / 10) + '.', drumX - 1, y + .5);
    c.save();
    c.beginPath(); c.rect(drumX, y - h - 6, drumW, 2 * h + 12); c.clip();
    c.textAlign = 'center';
    for (let k = -1; k <= 2; k++) {
      c.globalAlpha = k === 0 ? 1 : .4;
      c.fillText(String(((d + k) % 10 + 10) % 10), drumX + drumW / 2, y + .5 + (f - k) * 15);
    }
    c.restore();
  }
  function drawMemoryTape(s) {
    const {c, W, H} = prep(memTape);
    const nv = s.device_memory || {};
    const max = memoryScale(s);
    const plotH = H - PAD_TOP - PAD_BOTTOM;
    if (plotH <= 10) return;
    const yOf = v => PAD_TOP + plotH - plotH * (v / max);
    const edge = W - 1.5;
    tapeBody(c, W, plotH);
    const stepGiB = niceStep(max / GiB, plotH, 26);
    const nvNow = (s.device_memory || {}).used_bytes;
    const readY = Number.isFinite(nvNow) ? Math.max(PAD_TOP + 18, Math.min(PAD_TOP + plotH - 18, yOf(slew('mem.used', nvNow)))) : null;
    c.font = '500 11px Bahnschrift, sans-serif'; c.textAlign = 'right'; c.textBaseline = 'middle';
    for (let g = 0; g * GiB <= max + 1; g += stepGiB / 2) {
      const y = Math.round(yOf(g * GiB)) + .5;
      const major = Math.abs((g / stepGiB) - Math.round(g / stepGiB)) < 1e-6;
      c.strokeStyle = major ? C.ink : C.dim; c.lineWidth = 1;
      c.beginPath(); c.moveTo(edge - (major ? 8 : 5), y); c.lineTo(edge, y); c.stroke();
      if (major && (readY == null || Math.abs(y - readY) > 24)) { c.fillStyle = C.ink; c.fillText(String(Math.round(g)), edge - 11, y); }
    }
    // The desktop reserve as a barber pole: memory above this line is memory the
    // desktop was promised.
    const floor = reserveFloorBytes(s), total = nv.total_bytes || 0;
    if (floor > 0 && total > 0) {
      const yTop = yOf(total), yLim = yOf(total - floor);
      c.save();
      c.beginPath(); c.rect(edge - 5, yTop, 5, yLim - yTop); c.clip();
      c.fillStyle = C.poleDark; c.fillRect(edge - 5, yTop, 5, yLim - yTop);
      c.strokeStyle = C.red; c.lineWidth = 2.5;
      for (let y = yTop - 10; y < yLim + 10; y += 7) { c.beginPath(); c.moveTo(edge - 6, y + 6); c.lineTo(edge + 1, y - 1); c.stroke(); }
      c.restore();
    }
    // The next start's saved target, as a magenta bug.
    const next = (s.desktop_reserve || {}).next_gib;
    const draftGiB = reserveDraft != null && Number.isFinite(reserveDraft) ? (reserveDraft === 0 ? 8 : reserveDraft) : next;
    if (total > 0 && draftGiB > 0) {
      const y = yOf(total - draftGiB * GiB);
      c.fillStyle = C.magenta;
      c.beginPath(); c.moveTo(edge - 13, y - 4); c.lineTo(edge - 9, y - 4); c.lineTo(edge - 6, y); c.lineTo(edge - 9, y + 4); c.lineTo(edge - 13, y + 4); c.closePath(); c.fill();
    }
    // Current reading in a readout window, with a trend vector for the next ten seconds.
    const ser = s.series || {}, u = ser.nvidia_used_bytes || [], t = ser.t_ms || [];
    const used = Number.isFinite(nv.used_bytes) ? nv.used_bytes : u[u.length - 1];
    if (!Number.isFinite(used)) return;
    const y = Math.max(PAD_TOP + 9, Math.min(PAD_TOP + plotH - 9, yOf(slew('mem.used', used))));
    if (u.length > 20 && t.length === u.length) {
      const i0 = nearest(t, t[t.length - 1] - 10000);
      const dt = (t[t.length - 1] - t[i0]) / 1000;
      if (dt > 2) {
        const proj = used + (u[u.length - 1] - u[i0]) / dt * 10;
        if (Math.abs(proj - used) > 0.25 * GiB) {
          const yp = Math.max(PAD_TOP, Math.min(PAD_TOP + plotH, yOf(proj)));
          c.strokeStyle = C.green; c.lineWidth = 2;
          c.beginPath(); c.moveTo(edge - 2, y); c.lineTo(edge - 2, yp); c.stroke();
        }
      }
    }
    const breached = s.admin_vram && s.admin_vram.desktop_reserve && s.admin_vram.desktop_reserve.holding === false;
    readoutWindow(c, W, Math.max(PAD_TOP + 18, Math.min(PAD_TOP + plotH - 18, y)), used / GiB, breached ? C.red : C.ink, true);
  }

  function drawMemoryTrace(s, win) {
    const {c, W, H} = prep(memTrace);
    const ser = s.series || {};
    const u = ser.nvidia_used_bytes || [], t = ser.t_ms || [];
    const n = Math.min(t.length, u.length);
    $('chart-empty').hidden = n >= 2;
    if (n < 2 || !win || W <= 0) { memTip.style.display = 'none'; return; }
    const max = memoryScale(s);
    const plotH = H - PAD_TOP - PAD_BOTTOM;
    const PW = W - GUTTER;
    const xOf = tt => 1 + (PW - 2) * ((tt - win.t0) / (win.t1 - win.t0));
    const yOf = v => PAD_TOP + plotH - plotH * (Math.max(0, v || 0) / max);
    const stepGiB = niceStep(max / GiB, plotH, 26);
    c.strokeStyle = C.rule; c.lineWidth = 1; c.setLineDash([1, 4]);
    for (let g = stepGiB; g * GiB <= max + 1; g += stepGiB) { const y = Math.round(yOf(g * GiB)) + .5; c.beginPath(); c.moveTo(0, y); c.lineTo(PW, y); c.stroke(); }
    c.setLineDash([]);
    drawTimeAxis(c, PW, H, win, xOf);
    // One column per pixel: the 10 Hz series is denser than the canvas.
    const cols = Math.max(2, Math.floor(W));
    const hiY = new Array(cols).fill(null), loY = new Array(cols).fill(null);
    for (let i = 0; i < n; i++) {
      const x = Math.min(cols - 1, Math.max(0, Math.floor(xOf(t[i]))));
      const y = yOf(u[i]);
      if (hiY[x] == null || y < hiY[x]) hiY[x] = y;
      if (loY[x] == null || y > loY[x]) loY[x] = y;
    }
    const pts = [];
    for (let x = 0; x < cols; x++) if (hiY[x] != null) pts.push([x + .5, (hiY[x] + loY[x]) / 2, hiY[x], loY[x]]);
    if (pts.length > 1) {
      c.beginPath(); c.moveTo(pts[0][0], PAD_TOP + plotH);
      pts.forEach(p => c.lineTo(p[0], p[1]));
      c.lineTo(pts[pts.length - 1][0], PAD_TOP + plotH); c.closePath();
      c.fillStyle = C.traceFill; c.fill();
      c.strokeStyle = C.ink; c.lineWidth = 1.5; c.lineJoin = 'round';
      c.beginPath(); pts.forEach((p, i) => i ? c.lineTo(p[0], p[1]) : c.moveTo(p[0], p[1])); c.stroke();
      c.strokeStyle = C.envelope; c.lineWidth = 1;
      pts.forEach(p => { if (p[3] - p[2] > 2) { c.beginPath(); c.moveTo(p[0], p[2]); c.lineTo(p[0], p[3]); c.stroke(); } });
    }
    const floor = reserveFloorBytes(s), total = (s.device_memory || {}).total_bytes || 0;
    if (floor > 0 && total > 0) {
      const y = Math.round(yOf(total - floor)) + .5;
      c.strokeStyle = C.red; c.lineWidth = 1; c.setLineDash([6, 4]);
      c.beginPath(); c.moveTo(0, y); c.lineTo(PW, y); c.stroke(); c.setLineDash([]);
    }
    drawEvents(c, H, ser, win, xOf, W);
    if (floor > 0 && total > 0) {
      const y = Math.round(yOf(total - floor)) + .5;
      c.font = '600 11px Bahnschrift, sans-serif'; c.textAlign = 'left'; c.textBaseline = 'alphabetic';
)HTML"
R"HTML(      const w = c.measureText('DESKTOP RESERVE').width;
      c.fillStyle = C.glass; c.fillRect(3, y - 15, w + 6, 12);
      c.fillStyle = C.reserveLabel; c.fillText('DESKTOP RESERVE', 6, y - 5);
    }
    if (cursorT != null) {
      const i = nearest(t, cursorT);
      const x = Math.round(xOf(t[i])) + .5;
      c.strokeStyle = C.dim; c.lineWidth = 1;
      c.beginPath(); c.moveTo(x, PAD_TOP); c.lineTo(x, H - PAD_BOTTOM); c.stroke();
      c.fillStyle = C.ink; c.beginPath(); c.arc(x, yOf(u[i]), 3.5, 0, Math.PI * 2); c.fill();
      const ev = eventNear(ser, cursorT, win.t1 - win.t0);
      memTip.innerHTML = `<span class="t">${esc(relTime(win.t1 - t[i]))}</span> · ${esc(gib(u[i]))}${ev ? ` · <span class="${/up/.test(ev.kind) ? 'green' : 'amber'}">${esc(ev.label || ev.kind)}</span>` : ''}`;
      memTip.style.display = 'block';
      memTip.style.left = (x > W * .6 ? Math.max(4, x - memTip.offsetWidth - 10) : x + 10) + 'px';
    } else memTip.style.display = 'none';
  }

  // Generation and prompt reading share a time axis but not a value axis: prompt
  // reading runs an order of magnitude faster, so on one scale the line people
  // care about would sit on the floor. Generation reads off the tape at the left,
  // prompt reading off the blue figures at the right edge.
  // Both channels share N divisions so their figures sit on the same lines, and
  // each step is a whole nice number so every figure is exact. A floor keeps an
  // idle instrument on a scale that means something.
  function niceUp(v) {
    const mag = Math.pow(10, Math.floor(Math.log10(v)));
    for (const m of [1, 2, 2.5, 4, 5, 8, 10]) if (m * mag >= v - 1e-9) return m * mag;
    return 10 * mag;
  }
  function rateScales(s, plotH) {
    const ser = s.throughput || {};
    const dec = ser.decode_tok_s || [], pre = ser.prefill_tok_s || [];
    const now = (s.requests || {}).decode_tok_s_total || 0;
    const decRaw = Math.max(40, now, ...dec) * 1.1, preRaw = Math.max(400, ...pre) * 1.1;
    const n = Math.max(2, Math.min(5, Math.floor((plotH || 120) / 26)));
    let decStep = niceUp(decRaw / n), preStep = niceUp(preRaw / n);
    if (decStep < 1) decStep = 1;
    return {steps: n, decStep, preStep, decMax: decStep * n, preMax: preStep * n};
  }
  function drawRateTape(s) {
    const {c, W, H} = prep(rateTape);
    const plotH = H - PAD_TOP - PAD_BOTTOM;
    if (plotH <= 10) return;
    const {decMax, steps} = rateScales(s, plotH);
    // Re-ranging slides the tape: each figure keeps its exact value and travels to its new
    // place, so a changed scale reads as the same tape moving rather than a new one.
    const shownMax = slew('rate.decMax', decMax);
    const edge = W - 1.5;
    const yOf = v => PAD_TOP + plotH - plotH * (Math.max(0, Math.min(v, shownMax)) / shownMax);
    tapeBody(c, W, plotH);
    const nowRate = slew('rate.dec', (s.requests || {}).decode_tok_s_total || 0);
    const readY = Math.max(PAD_TOP + 11, Math.min(PAD_TOP + plotH - 11, yOf(nowRate)));
    c.font = '500 11px Bahnschrift, sans-serif'; c.textAlign = 'right'; c.textBaseline = 'middle';
    const half = decMax / (steps * 2);
    for (let k = 0; k * half <= shownMax + 1e-6; k++) {
      const v = half * k, y = Math.round(yOf(v)) + .5, major = k % 2 === 0;
      c.strokeStyle = major ? C.ink : C.dim;
      c.beginPath(); c.moveTo(edge - (major ? 8 : 5), y); c.lineTo(edge, y); c.stroke();
      if (major && Math.abs(y - readY) > 16) { c.fillStyle = C.green; c.fillText(Math.round(v).toLocaleString(), edge - 11, y); }
    }
    const reqs = s.requests || {};
    if (reqs.log_available === false) return;
    readoutWindow(c, W, readY, rate(nowRate), C.green, false);
  }
  function drawRateTrace(s, win) {
    const {c, W, H} = prep(rateTrace);
    const ser = s.throughput || {};
    const t = ser.t_ms || [], dec = ser.decode_tok_s || [], pre = ser.prefill_tok_s || [];
    const n = Math.min(t.length, dec.length, pre.length);
    $('throughput-empty').hidden = n >= 2;
    // The history comes from the engine's 5-second throughput reports; the running count and
    // the current rate come from the request log. An empty window says only what it knows.
    const busy = ((s.requests || {}).running_requests || 0) > 0 || ((s.requests || {}).decode_tok_s_total || 0) > 0;
    $('throughput-empty').textContent = busy ? 'Collecting rate samples…' : 'No rate samples in this window';
    if (!win || W <= 0) { rateTip.style.display = 'none'; return; }
    const plotH = H - PAD_TOP - PAD_BOTTOM;
    const scale = rateScales(s, plotH);
    const decMax = slew('rate.decMax', scale.decMax), preMax = slew('rate.preMax', scale.preMax);
    const PW = W - GUTTER;
    const xOf = tt => 1 + (PW - 2) * ((tt - win.t0) / (win.t1 - win.t0));
    const yAt = (v, max) => Math.round(PAD_TOP + plotH - plotH * (v / max)) + .5;
    c.font = '500 11px Bahnschrift, sans-serif'; c.textBaseline = 'middle';
    c.strokeStyle = C.rule; c.lineWidth = 1; c.setLineDash([1, 4]);
    for (let k = 1; k * scale.decStep <= decMax + 1e-6; k++) { const y = yAt(k * scale.decStep, decMax); c.beginPath(); c.moveTo(0, y); c.lineTo(PW, y); c.stroke(); }
    c.setLineDash([]);
    c.textAlign = 'right'; c.fillStyle = C.ice;
    for (let k = 1; k * scale.preStep <= preMax + 1e-6; k++) {
      const y = yAt(k * scale.preStep, preMax);
      if (y >= PAD_TOP - .5) c.fillText(Math.round(k * scale.preStep).toLocaleString(), W - 2, y);
    }
    c.textBaseline = 'alphabetic';
    drawTimeAxis(c, PW, H, win, xOf);
    if (n < 2) { rateTip.style.display = 'none'; drawEvents(c, H, (s.series || {}), win, xOf, W); return; }
    const line = (vals, max, colour, fill) => {
      const yOf = v => PAD_TOP + plotH - plotH * (Math.max(0, v || 0) / max);
      if (fill) {
        c.beginPath(); c.moveTo(xOf(t[0]), PAD_TOP + plotH);
        for (let i = 0; i < n; i++) c.lineTo(xOf(t[i]), yOf(vals[i]));
        c.lineTo(xOf(t[n - 1]), PAD_TOP + plotH); c.closePath();
        c.fillStyle = C.genFill; c.fill();
      }
      c.beginPath(); c.lineWidth = fill ? 1.75 : 1.25; c.strokeStyle = colour; c.lineJoin = 'round';
      for (let i = 0; i < n; i++) { if (i === 0) c.moveTo(xOf(t[i]), yOf(vals[i])); else c.lineTo(xOf(t[i]), yOf(vals[i])); }
      c.stroke();
      return yOf;
    };
    line(pre, preMax, C.ice, false);
    const yDec = line(dec, decMax, C.green, true);
    drawEvents(c, H, (s.series || {}), win, xOf, W);
    if (cursorT != null) {
      const i = nearest(t, cursorT);
      const x = Math.round(xOf(t[i])) + .5;
      c.strokeStyle = C.dim; c.lineWidth = 1;
      c.beginPath(); c.moveTo(x, PAD_TOP); c.lineTo(x, H - PAD_BOTTOM); c.stroke();
      c.fillStyle = C.green; c.beginPath(); c.arc(x, yDec(dec[i]), 3.5, 0, Math.PI * 2); c.fill();
      rateTip.innerHTML = `<span class="t">${esc(relTime(win.t1 - t[i]))}</span> · <span class="green">${esc(rate(dec[i]))} tok/s generating</span> · <span class="ice">${esc(rate(pre[i]))} tok/s reading</span>`;
      rateTip.style.display = 'block';
      rateTip.style.left = (x > W * .55 ? Math.max(4, x - rateTip.offsetWidth - 10) : x + 10) + 'px';
    } else rateTip.style.display = 'none';
  }

  function drawAll() {
    if (!lastState || !isDocumentVisible || activeView !== 'overview') return;
    const win = timeWindow(lastState);
    drawMemoryTape(lastState);
    drawMemoryTrace(lastState, win);
    drawRateTape(lastState);
    drawRateTrace(lastState, win);
    if (win) {
      const span = Math.round((win.t1 - win.t0) / 1000);
      const label = `Last ${span >= 120 ? Math.round(span / 60) + ' min' : span + ' s'}`;
      $('chart-range').textContent = label + ' · GiB';
      $('throughput-range').textContent = label + ' · tok/s';
    }
  }
  if (window.ResizeObserver) new ResizeObserver(() => requestAnimationFrame(drawAll)).observe($('view-overview'));
  window.addEventListener('resize', () => requestAnimationFrame(drawAll));

  // Seams: in the wide layout the gaps between the Monitor displays drag (or move with the
  // arrow keys) to give memory, rates or the side column more room. Double-click or Enter
  // restores the default split; the split is kept in this browser.
  const SPLIT_KEY = 'ninfer:monitor-split';
  const monitorGrid = document.querySelector('.monitor-grid');
  const memDisplay = document.querySelector('.display-memory'), rateDisplay = document.querySelector('.display-rates');
  const seams = {col: $('seam-col'), row: $('seam-row')};
  let split = {col: null, row: null};
  try { Object.assign(split, JSON.parse(localStorage.getItem(SPLIT_KEY) || '{}')); } catch (_) {}
  const gridGap = () => parseFloat(getComputedStyle(monitorGrid).columnGap) || 8;
  function placeSeams() {
    if (getComputedStyle(seams.col).display === 'none') return;
    const gr = monitorGrid.getBoundingClientRect(), mr = memDisplay.getBoundingClientRect(), rr = rateDisplay.getBoundingClientRect();
    const gap = gridGap();
    seams.col.style.left = (mr.right - gr.left) + 'px';
    Object.assign(seams.row.style, {top: (mr.bottom - gr.top) + 'px', left: (mr.left - gr.left) + 'px', width: mr.width + 'px'});
    seams.col.setAttribute('aria-valuenow', String(Math.round(100 * mr.width / Math.max(1, gr.width - gap))));
    seams.row.setAttribute('aria-valuenow', String(Math.round(100 * mr.height / Math.max(1, rr.bottom - mr.top - gap))));
  }
  function applySplit() {
    const g = monitorGrid.style;
    [['col', '--col-a', '--col-b'], ['row', '--row-a', '--row-b']].forEach(([k, a, b]) => {
      if (split[k] != null) { g.setProperty(a, split[k] + 'fr'); g.setProperty(b, (1 - split[k]) + 'fr'); }
      else { g.removeProperty(a); g.removeProperty(b); }
    });
    placeSeams();
    requestAnimationFrame(drawAll);
  }
  function saveSplit() { try { localStorage.setItem(SPLIT_KEY, JSON.stringify(split)); } catch (_) {} }
  // The split a pointer at (x, y) asks for, kept clear of each side's floor.
  function splitAt(kind, x, y) {
    const gr = monitorGrid.getBoundingClientRect(), gap = gridGap();
    if (kind === 'col') {
      const avail = gr.width - gap;
      return Math.min(avail - 380, Math.max(Math.min(520, avail / 2), x - gr.left - gap / 2)) / avail;
    }
    const mr = memDisplay.getBoundingClientRect(), rr = rateDisplay.getBoundingClientRect();
    const avail = rr.bottom - mr.top - gap;
    return Math.min(avail - 180, Math.max(190, y - mr.top - gap / 2)) / avail;
  }
  Object.entries(seams).forEach(([kind, el]) => {
    const reset = () => { split[kind] = null; applySplit(); saveSplit(); };
    el.addEventListener('pointerdown', e => {
      if (e.button !== 0) return;
      e.preventDefault();
      el.setPointerCapture(e.pointerId);
      el.classList.add('dragging'); document.body.classList.add('resizing-' + kind);
    });
    el.addEventListener('pointermove', e => {
      if (!el.classList.contains('dragging')) return;
      split[kind] = splitAt(kind, e.clientX, e.clientY); applySplit();
    });
    const end = () => {
      if (!el.classList.contains('dragging')) return;
      el.classList.remove('dragging'); document.body.classList.remove('resizing-' + kind);
      saveSplit();
    };
    ['pointerup', 'pointercancel', 'lostpointercapture'].forEach(type => el.addEventListener(type, end));
    el.addEventListener('dblclick', reset);
    el.addEventListener('keydown', e => {
      if (e.key === 'Enter') { e.preventDefault(); reset(); return; }
      const dir = (kind === 'col' ? {ArrowLeft: -1, ArrowRight: 1} : {ArrowUp: -1, ArrowDown: 1})[e.key];
      if (!dir) return;
      e.preventDefault();
      const r = el.getBoundingClientRect(), step = (e.shiftKey ? 80 : 16) * dir;
      split[kind] = splitAt(kind, r.left + r.width / 2 + (kind === 'col' ? step : 0), r.top + r.height / 2 + (kind === 'row' ? step : 0));
      applySplit(); saveSplit();
    });
  });
  if (window.ResizeObserver) { const ro = new ResizeObserver(placeSeams); ro.observe(monitorGrid); ro.observe(memDisplay); }
  applySplit();

)HTML"
R"HTML(  function cursorFromPointer(canvas, e) {
    const win = lastState && timeWindow(lastState);
    if (!win) return;
    const r = canvas.getBoundingClientRect();
    const f = Math.min(1, Math.max(0, (e.clientX - r.left - 1) / (r.width - 2 - GUTTER)));
    cursorT = win.t0 + f * (win.t1 - win.t0);
    drawAll();
  }
  [memTrace, rateTrace].forEach(canvas => {
    canvas.addEventListener('mousemove', e => cursorFromPointer(canvas, e));
    canvas.addEventListener('mouseleave', () => { cursorT = null; drawAll(); });
    canvas.addEventListener('keydown', e => {
      const win = lastState && timeWindow(lastState);
      if (!win) return;
      const step = (win.t1 - win.t0) / 100;
      if (cursorT == null) cursorT = win.t1;
      if (e.key === 'ArrowLeft') cursorT = Math.max(win.t0, cursorT - step);
      else if (e.key === 'ArrowRight') cursorT = Math.min(win.t1, cursorT + step);
      else if (e.key === 'Home') cursorT = win.t0;
      else if (e.key === 'End') cursorT = win.t1;
      else if (e.key === 'Escape') cursorT = null;
      else return;
      e.preventDefault();
      drawAll();
      if (cursorT != null) {
        const ser = lastState.series || {}, thr = lastState.throughput || {};
        const i = nearest(ser.t_ms || [], cursorT), j = nearest(thr.t_ms || [], cursorT);
        a11yLive.textContent = `${relTime(win.t1 - cursorT)}: memory ${i >= 0 ? gib(ser.nvidia_used_bytes[i]) : 'unavailable'}, generating ${j >= 0 ? rate(thr.decode_tok_s[j]) : '—'} tokens per second, reading ${j >= 0 ? rate(thr.prefill_tok_s[j]) : '—'} tokens per second.`;
      }
    });
    canvas.addEventListener('blur', () => { cursorT = null; drawAll(); });
  });

  // ---------------------------------------------------------------------
  // Calling apps: one row per app from the engine's own record. Columns never
  // move; a row under queue pressure earns a caution line, never bigger type.
  // ---------------------------------------------------------------------
  function renderClients(reqs) {
    const list = $('clients-list');
    const clients = Array.isArray(reqs.clients) ? reqs.clients : [];
    const focused = document.activeElement && document.activeElement.matches('.client-evidence') ? document.activeElement.dataset.client : null;
    const mins = reqs.clients_window_minutes || 15;
    const restarts = reqs.client_engine_starts || 0;
    $('clients-count').textContent = clients.length ? `${clients.length} app${clients.length === 1 ? '' : 's'}` : '—';
    $('clients-sub').textContent = `Last ${mins} min${restarts ? ` · ${restarts} engine start${restarts === 1 ? '' : 's'}` : ''}${reqs.client_window_limited ? ' · observation limit reached' : ''}`;
    if (!clients.length) {
      list.innerHTML = '<tr class="traffic-empty"><td colspan="5">No app has sent a request in the last ' + mins + ' minutes. Apps appear here as they call the engine.</td></tr>';
      return;
    }
    const tokens = value => Number(value).toLocaleString();
    const reasons = {source_invalid:'Selected checkpoint was invalid', source_revoked:'Selected checkpoint was revoked', cost_changed:'Restore cost changed', capacity_limit:'Restore exceeded capacity', isolated_capacity:'Restore could not fit in isolated capacity'};
    const rows = [];
    for (const c of clients) {
      const key = c.name || '';
      const pressure = (c.queue_ms?.mean || 0) >= 1000 || c.admission_timeouts > 0;
      const tr = document.createElement('tr');
      tr.className = 'client' + (pressure ? ' caution' : '');
      const app = document.createElement('td');
      const name = document.createElement('div'); name.className = 'client-name';
      // An app that sends no User-Agent cannot be named; saying so beats inventing a label.
      const label = document.createElement('span'); label.textContent = c.name || 'Unidentified app'; label.title = c.name || 'This app sends no User-Agent';
      name.append(label);
      if (c.tool_count > 0) { const tag = document.createElement('span'); tag.className = 'client-tag'; tag.textContent = `${c.tool_count} tool${c.tool_count === 1 ? '' : 's'}`; name.append(tag); }
      const counts = document.createElement('div'); counts.className = 'client-counts';
      counts.title = `${c.completed} completed, ${c.failed} failed, ${c.cancelled} cancelled, ${c.interrupted} unfinished at restart, ${c.active} awaiting outcome`;
      counts.innerHTML = `<b>${(c.completed || 0).toLocaleString()}</b> done · ${c.failed || 0} fail · ${c.cancelled || 0} cxl · ${c.interrupted || 0} cut · ${c.active || 0} open`;
      const ev = document.createElement('button');
      ev.type = 'button'; ev.className = 'client-evidence'; ev.dataset.client = key;
      ev.setAttribute('aria-expanded', String(expandedClients.has(key)));
      ev.textContent = expandedClients.has(key) ? 'Evidence <' : 'Evidence >';
      ev.onclick = () => { if (expandedClients.has(key)) expandedClients.delete(key); else expandedClients.add(key); renderClients((lastState || {}).requests || {}); };
      app.append(name, counts, ev);
      const cell = (labelText, v, sub, title) => {
        const td = document.createElement('td'); td.className = 'num'; td.dataset.label = labelText;
        const vv = document.createElement('span'); vv.className = 'v'; vv.textContent = v;
        const ss = document.createElement('span'); ss.className = 's'; ss.textContent = sub;
        if (title) td.title = title;
        td.append(vv, ss); return td;
      };
      const timing = (labelText, metric, title) => cell(labelText, Number.isFinite(metric?.mean) ? duration(metric.mean) : '—', Number.isFinite(metric?.p95) ? `p95 ${duration(metric.p95)}` : 'unavailable', title);
      const queueIsFirst = c.queue_ms?.samples > 0 && c.queue_first_output_samples === c.queue_ms.samples;
      const reuse = cell('Reuse', Number.isFinite(c.reuse_percent) ? `${c.reuse_percent.toFixed(0)}%` : '—', 'of prompt tokens');
      const queueCell = timing('Queue', c.queue_ms, queueIsFirst ? 'Queue wait before the first token' : 'Request queue wait');
      if ((c.queue_ms?.mean || 0) >= 1000) { queueCell.querySelector('.v').classList.add('amber'); queueCell.title += ` averages ${duration(c.queue_ms.mean)}; queue wait and prefill are measured separately.`; }
      tr.append(app, timing('First token', c.ttft_ms), queueCell, timing('Prefill', c.prefill_ms), reuse);
      rows.push(tr);
      // Requests that never reached a lane are the one thing the columns cannot show.
      if (c.admission_timeouts > 0) {
        const line = document.createElement('tr'); line.className = 'caution-line';
        const td = document.createElement('td'); td.colSpan = 5;
        td.textContent = `${c.admission_timeouts} request${c.admission_timeouts === 1 ? '' : 's'} expired before admission${Number.isFinite(c.queue_ms?.mean) ? `; recorded queue wait averages ${duration(c.queue_ms.mean)}` : ''}.`;
        line.append(td); rows.push(line);
      }
      if (expandedClients.has(key)) {
        const evidence = [];
        if (c.computed_samples > 0) evidence.push(`${tokens(c.computed_tokens)} prompt tokens computed across ${c.computed_samples} requests; this includes new input and recovery work.`);
        evidence.push(`Cache-hit observations: ${c.cache_samples}. Timing observations: ${c.ttft_ms?.samples || 0} first-token, ${c.queue_ms?.samples || 0} queue, ${c.prefill_ms?.samples || 0} prefill.`);
        if (c.full_prefills) evidence.push(`${c.full_prefills} full prefill${c.full_prefills === 1 ? '' : 's'}; ${c.unexplained_full_prefills} with no recorded cause${c.reuse_disabled_prefills ? `, ${c.reuse_disabled_prefills} with prefix reuse disabled` : ''}.`);
        for (const [reason, count] of Object.entries(c.fallbacks || {})) evidence.push(`${reasons[reason] || `Admission fallback: ${reason}`}: ${count} request${count === 1 ? '' : 's'}.`);
        if (c.engine_instances > 1) evidence.push(`Observations span ${c.engine_instances} engine instances. Each engine start clears its resident cache.`);
        evidence.push('Tool-definition changes and conversation changes are not diagnosed by these counters.');
        const er = document.createElement('tr'); er.className = 'evidence-row';
        const td = document.createElement('td'); td.colSpan = 5;
        evidence.forEach(text => { const p = document.createElement('p'); p.textContent = text; td.append(p); });
        er.append(td); rows.push(er);
      }
    }
    list.replaceChildren(...rows);
    if (focused != null) { const b = list.querySelector(`.client-evidence[data-client="${CSS.escape(focused)}"]`); if (b) b.focus({preventScroll: true}); }
  }

  // ---------------------------------------------------------------------
  // Findings
  // ---------------------------------------------------------------------
  function describeInsight(it) {
    const id = it.id || '';
    const e = it.evidence || {};
    const fallback = {title:it.title || id, body:it.statement || '', action:it.recommendation || ''};
    if (id.startsWith('latency.ttft_')) return {title:'How long responses take to begin', body:Number.isFinite(e.mean_ttft_s) ? `Responses began after an average of ${Math.round(e.mean_ttft_s * 1000)} ms in the recorded requests. Processing your prompt, preparing media, and waiting for the engine can all contribute.` : 'The engine is collecting response-start timing. Open the evidence for the available measurements.'};
    if (id.startsWith('latency.')) return {title:'Where response time is spent', body:e.queued > 0 ? `${e.queued} recorded requests spent most of their time waiting for engine capacity.` : 'The recorded requests do not show queueing as the main source of delay. Open the evidence for processing and generation timings.', action:e.queued > 0 ? 'Try fewer simultaneous requests. Advanced capacity settings can help if enough GPU memory is available.' : ''};
    if (id === 'speculative.draft_acceptance') {
      const pct = r => Math.round((r || 0) * 100) + '%';
      if (it.availability === 'unavailable') return {title:'Draft-token acceptance is not measurable yet', body:e.backend ? `Speculative decoding (${e.backend}) is configured, but the recorded requests have no draft rounds yet.` : 'The recorded requests carry no speculative decoding activity. This is expected when speculation is turned off.'};
      const curve = (e.per_position_rate || []).map((r, i) => `P${i + 1} ${pct(r)}`).join(', ');
      const actions = [];
      if (it.severity === 'warning') actions.push('Many tokens were generated without a draft. Open the evidence to see which requests, and compare them with requests that drafted normally.');
      if (Number.isInteger(e.inferred_draft_window)) actions.push(`Later draft positions are rarely accepted. A draft window of ${e.inferred_draft_window} may generate faster; this is inferred from the curve, so measure before keeping it.`);
      return {title:'How many drafted tokens were accepted', body:`${pct(e.acceptance_ratio)} of ${(e.drafted_tokens || 0).toLocaleString()} drafted tokens were accepted across ${e.requests_with_drafts || 0} requests${curve ? ` (by position: ${curve})` : ''}. The engine decoded without a draft on ${pct(e.fallback_share)} of steps.`, action:actions.join(' ')};
    }
    if (id === 'capacity.context_kv_pressure') {
      const pct = r => Math.round((r || 0) * 100) + '%';
      const cfg = e.config || {}, req = e.requests || {}, kv = e.kv || {}, sch = e.scheduler || {};
      if (it.availability === 'unavailable') return {title:'Capacity limits are not measurable yet', body:cfg.max_context ? 'The engine has not reported its memory use yet. This appears after the first activity sample.' : 'The log does not record this engine start, so traffic cannot be compared with its limits.'};
)HTML"
R"HTML(      const actions = {large_prompt:'A request came within 10% of the context limit. Raise Max context in Request capacity if GPU memory allows, or shorten long conversations.', kv_admission:'Requests waited because the conversation memory was full. Raise KV capacity if GPU memory allows, otherwise lower Max context to bound memory growth per request.', lanes_full:'Requests waited because every concurrent slot was busy. Raise Max concurrency only if KV capacity and GPU memory leave room for another request.', cache_churn:'KV pages moved to host memory to make room. Restoring them adds transfer work. Raise KV capacity if GPU memory allows.', preempted:'Requests were paused to make room and needed recovery. Raise KV capacity if GPU memory allows, or reduce concurrent demand.'};
      return {title:'How close you are to the context and memory limits', body:`The largest request used ${pct(req.largest_context_ratio)} of the ${(cfg.max_context || 0).toLocaleString()}-token context limit across ${req.count || 0} requests. Conversation memory peaked at ${pct(kv.utilization_high_water)} of capacity${kv.includes_retained_checkpoints ? ' (including saved conversation state)' : ''}, with up to ${sch.peak_running || 0} of ${cfg.max_concurrency || 0} requests running at once.`, action:actions[e.kind] || ''};
    }
    if (id === 'prefix.reuse_mix') return {title:'Reusing earlier conversation text', body:e.multi_turn_prompt_tokens > 0 ? `${((e.multi_turn_hit_ratio || 0) * 100).toFixed(0)}% of conversation prompt tokens were reused in the available log. Reuse can make follow-up responses start sooner.` : 'No multi-turn prompt samples are available yet. Reuse can make follow-up responses start sooner.'};
    if (id === 'client.narrated_tool_intent') return {title:'Tool behavior needs a conversation sample', body:'The log records whether tools were provided, but not what the model said. It cannot explain a promised tool action that did not happen.'};
    if (id === 'client.content_fields') return {title:'Reply text is not stored in this log', body:'To investigate an empty or unexpected reply, compare the conversation in your app with the engine log.'};
    if (id === 'vram.admin') return {title:'Detailed engine memory data is unavailable', body:'The supervisor could not retrieve the engine’s internal memory report. GPU-wide readings may still be available.', action:'If this persists after the engine is ready, check the API connection and authentication settings.'};
    if (id === 'prefix.reuse_collapsed' || id === 'prefix.multiturn_root') return {title:'Earlier conversation text is being processed again', body:'The logs show repeated work instead of reusing conversation state. Follow-up responses may take longer.', action:'Copy the engine log before restarting so the issue can be investigated.'};
    if (id === 'client.output_limit_while_thinking') return {title:'Some responses reached their length limit', body:'Requests with reasoning enabled stopped at the output limit. The log cannot confirm whether your app received a visible answer.', action:'Try a higher response-token limit in your app and compare the result.'};
    return fallback;
  }
  const master = $('master-caution');
  let masterIds = [];
  function ackedFindings() { try { return new Set(JSON.parse(localStorage.getItem('ninfer:acknowledged') || '[]')); } catch (_) { return new Set(); } }
  function updateMaster(problems) {
    masterIds = problems.map(it => it.id || it.title || '');
    const acked = ackedFindings();
    const fresh = problems.filter(it => !acked.has(it.id || it.title || ''));
    const warning = fresh.some(it => it.severity === 'critical');
    master.classList.toggle('lit', fresh.length > 0);
    master.classList.toggle('warning', warning);
    $('master-word').textContent = warning ? 'Warning' : 'Caution';
    master.setAttribute('aria-label', fresh.length ? `Master ${warning ? 'warning' : 'caution'}: ${fresh.length} new finding${fresh.length === 1 ? '' : 's'} to review. Press to acknowledge and open Diagnose.` : 'Master caution: no new findings. Opens Diagnose.');
  }
  master.onclick = () => {
    try { localStorage.setItem('ninfer:acknowledged', JSON.stringify(masterIds)); } catch (_) {}
    master.classList.remove('lit', 'warning');
    location.hash = '#diagnostics';
  };
  const sevRank = {critical:0, warning:1, notice:2, info:3};
  const sevWord = {critical:'Warning', warning:'Caution', notice:'Note', info:'Info'};
  function renderInsights(rep) {
    const items = (rep.insights || []).slice().sort((a, b) => (sevRank[a.severity] ?? 4) - (sevRank[b.severity] ?? 4));
    // After a start the Supervisor reads the existing request log once before it reports on it.
    const src = rep.source || {};
    const reading = src.request_log === 'reading';
    $('insights-count-badge').textContent = reading ? `reading ${src.bytes_total ? Math.floor(100 * src.bytes_read / src.bytes_total) : 0}%` : String(items.length);
    const problems = items.filter(it => ['critical', 'warning'].includes(it.severity));
    if (!reading) updateMaster(problems);
    const navIssues = $('nav-issues');
    navIssues.hidden = !problems.length;
    navIssues.textContent = problems.length;
    navIssues.setAttribute('aria-label', `${problems.length} to review`);
    const attention = $('attention-title');
    attention.textContent = reading ? 'Reading the request log…' : problems.length ? `${problems.length} to review` : `Nothing to review · ${items.length} note${items.length === 1 ? '' : 's'}`;
    attention.className = 'display-source' + (problems.length ? ' amber' : '');
    const alertList = $('alert-list');
    if (!problems.length) {
      alertList.innerHTML = reading ? '<li class="alert-quiet">Reading the existing request log before reporting on it.</li>'
        : `<li class="alert-quiet"><b>Normal</b> No finding needs review.${items.length ? ` ${items.length} informational note${items.length === 1 ? '' : 's'} on Diagnose.` : ''}</li>`;
    } else {
      alertList.innerHTML = problems.slice(0, 4).map(it => {
        const f = describeInsight(it);
        return `<li><span class="alert-sev ${it.severity === 'critical' ? 'red' : 'amber'}">${esc(sevWord[it.severity])}</span><span class="alert-title">${esc(f.title)}</span><span class="alert-text">${esc(f.action || f.body)}</span></li>`;
      }).join('') + (problems.length > 4 ? `<li class="alert-quiet">${problems.length - 4} more on Diagnose.</li>` : '');
    }
    const json = JSON.stringify(items);
    if (json === lastInsightsJson) return;
    lastInsightsJson = json;
    const openIds = new Set([...document.querySelectorAll('.finding details[open]')].map(el => el.dataset.id));
    const box = $('insights-container');
    if (!items.length) { box.innerHTML = '<p class="cdu-small">No diagnostic findings are available yet. Findings appear as the engine records activity.</p>'; return; }
    box.innerHTML = items.map((it, idx) => {
      const sev = it.severity || 'info';
      const f = describeInsight(it);
      const did = esc(it.id || idx);
      const open = openIds.has(String(it.id || idx));
      const compact = (sev === 'notice' || sev === 'info') && !f.action;
      return `<article class="finding${compact ? ' compact' : ''}${compact && open ? ' open' : ''}">
        <span class="finding-sev ${esc(sev)}">${esc(sevWord[sev] || sev)}</span>
        <h3 class="finding-title">${esc(f.title)}</h3>
        <p class="finding-body" title="${esc(f.body)}">${esc(f.body)}</p>
        ${f.action ? `<p class="finding-action">${esc(f.action)}</p>` : ''}
        <details data-id="${did}" ${open ? 'open' : ''}><summary>Evidence</summary><p>${esc(it.statement || '')}</p>${it.recommendation ? `<p>${esc(it.recommendation)}</p>` : ''}<pre class="cdu-pre">${esc(JSON.stringify(it.evidence || {}, null, 2))}</pre></details>
      </article>`;
    }).join('');
    box.querySelectorAll('.finding.compact details').forEach(d => d.addEventListener('toggle', () => d.closest('.finding').classList.toggle('open', d.open)));
  }

  // ---------------------------------------------------------------------
  // Desktop reserve
  // ---------------------------------------------------------------------
  let reserveDraft = null, reserveBusy = false, gpuAppsKey = '';
  const reserveSlider = $('reserve-slider'), reserveAmount = $('reserve-amount'), reserveSave = $('reserve-save');
  function renderMemoryControls(s) {
    const info = s.desktop_reserve || {};
    const saved = info.next_gib == null ? 8 : info.next_gib;
    const value = reserveDraft == null ? saved : reserveDraft === 0 ? 8 : reserveDraft;
    const nv = s.device_memory || {};
    const total = nv.ok ? nv.total_bytes / GiB : null;
    const budget = s.reserve_budget || {};
    const max = budget.ok ? budget.max_gib : 0;
    reserveSlider.max = reserveAmount.max = Math.max(1, max);
    reserveSlider.min = reserveAmount.min = saved === 0 && reserveDraft == null ? 0 : 1;
    if (document.activeElement !== reserveAmount) reserveAmount.value = value;
    reserveSlider.value = value;
    reserveSlider.setAttribute('aria-valuetext', reserveSlider.value + ' GiB free memory target');
    const disabled = !connected || s.monitor_only || info.pinned || reserveBusy || modelBusy || modelUncertain || !budget.ok || max < 1;
    reserveSlider.disabled = reserveAmount.disabled = disabled;
    $('reserve-default').disabled = disabled || max < 8;
    $('reserve-default').title = budget.ok && max < 8 ? 'The 8 GiB engine default exceeds this model’s current limit.' : '';
    reserveSave.disabled = disabled || reserveDraft == null || !Number.isInteger(value) || value < 1 || value > max;
    $('reserve-discard').hidden = reserveDraft == null;
    $('reserve-discard').disabled = reserveBusy;
    const part = total ? Math.min(100, value / total * 100) : 0;
    $('reserve-free-preview').style.width = part + '%';
    $('reserve-engine-preview').style.width = (100 - part) + '%';
    $('reserve-preview-note').textContent = budget.ok ? `Up to ${max} GiB can be left free with this model and its current settings. Model weights: ${(budget.weights_bytes / GiB).toFixed(1)} GiB. Runtime and conversation capacity: ${(budget.runtime_bytes / GiB).toFixed(1)} GiB. ${budget.automatic_capacity ? 'Conversation capacity adjusts automatically at startup. ' : 'Your configured conversation capacity is preserved. '}The limit also keeps startup slack and graph headroom, and accounts for other apps.` : budget.reason || 'Waiting for this model’s memory plan. Reserve editing is unavailable until it can be measured.';
    const active = s.admin_vram && s.admin_vram.desktop_reserve;
    const running = s.engine && s.engine.state === 'Running' && s.health && s.health.status === 200;
    $('reserve-current').textContent = running && active ? `Running target ${(active.configured_bytes / GiB).toFixed(1)} GiB · saved for next start ${saved} GiB` : `Saved for next start ${saved} GiB · running target unavailable`;
    const pending = running && active && Math.abs(active.configured_bytes / GiB - saved) > .01;
    const status = $('reserve-status');
    status.textContent = info.pinned ? 'This target is set in MiB in your launch configuration. Edit that configuration to change it.' : s.monitor_only ? 'Change the reserve in the application that starts this engine.' : !budget.ok ? budget.reason || 'Waiting for a model memory plan.' : value > max ? `This ${value} GiB target exceeds the current ${max} GiB limit. Lower it before saving or restarting.` : reserveBusy ? 'Saving reserve…' : reserveDraft != null ? 'Temporary revision. Inserting keeps your engine running; it applies at the next start.' : pending ? 'Saved. Restart the engine when convenient to apply this target.' : 'Changes take effect on the next engine start.';
    status.className = 'cdu-data' + (value > max && budget.ok ? ' red' : reserveDraft != null ? ' amber' : '');
    const ro = $('ro-reserve');
)HTML"
R"HTML(    ro.textContent = active && running ? `${(active.configured_bytes / GiB).toFixed(0)} GiB` : `${saved} GiB`;
    ro.title = `Running ${active && running ? (active.configured_bytes / GiB).toFixed(1) + ' GiB' : 'unavailable'}; next start ${saved} GiB`;
    $('sum-reserve').textContent = (reserveDraft != null ? `staged ${value} GiB · ` : '') + `${saved} GiB next start` + (active && running ? ` · ${(active.configured_bytes / GiB).toFixed(1)} GiB running` : '');
    $('sum-reserve').className = 'lsk-sum' + (reserveDraft != null ? ' amber' : '');
    updateTmpy();
    renderGpuApps(s);
  }
  function renderGpuApps(s) {
    const processes = s.gpu_processes || {};
    const key = JSON.stringify([processes, (s.engine || {}).pid, gpuAppsExpanded]);
    if (key === gpuAppsKey) return;
    gpuAppsKey = key;
    const list = $('gpu-app-list');
    list.replaceChildren();
    if (!processes.ok) { list.textContent = processes.error || 'Waiting for Windows GPU readings…'; return; }
    const groups = new Map();
    for (const app of processes.apps || []) {
      const own = app.pid === (s.engine || {}).pid;
      const name = own ? 'NInfer engine' : app.name.replace(/\.exe$/i, '');
      const group = groups.get(name) || {bytes: 0, count: 0, own};
      group.bytes += app.dedicated_bytes; group.count++; groups.set(name, group);
    }
    const ordered = [...groups].sort((a, b) => b[1].bytes - a[1].bytes);
    if (!ordered.length) { list.textContent = 'Windows reports no dedicated GPU allocations on this card.'; return; }
    const shown = gpuAppsExpanded ? ordered : ordered.slice(0, 5);
    for (const [name, group] of shown) {
      const item = document.createElement('span'); item.className = 'gpu-app' + (group.own ? ' own' : '');
      item.textContent = name + (group.count > 1 ? ` ×${group.count}` : '');
      const v = document.createElement('b'); v.textContent = appBytes(group.bytes);
      item.append(v); list.append(item);
    }
    if (ordered.length > 5) {
      const more = document.createElement('button'); more.type = 'button'; more.className = 'client-evidence';
      more.textContent = gpuAppsExpanded ? 'Fewer <' : `${ordered.length - 5} more >`;
      more.setAttribute('aria-expanded', String(gpuAppsExpanded));
      more.onclick = () => { gpuAppsExpanded = !gpuAppsExpanded; renderGpuApps(lastState || {}); };
      list.append(more);
    }
  }
  function changeReserve(value) { reserveDraft = value; renderMemoryControls(lastState || {}); updateModelControls(); drawAll(); }
  reserveSlider.oninput = () => { reserveAmount.value = reserveSlider.value; changeReserve(Number(reserveSlider.value)); };
  reserveAmount.oninput = () => changeReserve(reserveAmount.value === '' ? NaN : Number(reserveAmount.value));
  $('reserve-default').onclick = () => changeReserve(0);
  $('reserve-discard').onclick = () => { reserveDraft = null; renderMemoryControls(lastState || {}); updateModelControls(); drawAll(); };
  reserveSave.onclick = async () => {
    if (modelBusy || modelUncertain || reserveBusy || reserveDraft == null || !reserveAmount.reportValidity()) return;
    reserveBusy = true; renderMemoryControls(lastState || {});
    try {
      const response = await fetch('/api/desktop-reserve', {method:'POST', headers:{'Content-Type':'application/json','X-NInfer-Supervisor':'1'}, body:JSON.stringify({gib:reserveDraft})});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Could not save the reserve.');
      reserveDraft = null; render(data); notify('Reserve inserted. It applies when the engine next starts.');
    } catch (error) { notify(error.message || 'Could not reach the supervisor. Try again.', true); }
    finally { reserveBusy = false; renderMemoryControls(lastState || {}); }
  };

  // ---------------------------------------------------------------------
  // State updates
  // ---------------------------------------------------------------------
  function render(s) {
    if (!s) return;
    lastState = s;
    renderFrontends(s.frontends || []);
    if (cfgTab === 'frontends' && activeView === 'settings') feRefreshStatus();
    renderMemoryControls(s);

    const eng = s.engine || {};
    const state = eng.state || 'Unknown';
    const healthy = s.health && s.health.status === 200;
    const ready = state === 'Running' && healthy;
    const managed = managedEngine();
    if (ready && managed && pendingRestartPid != null && eng.pid && eng.pid !== pendingRestartPid) setPendingRestart(null);
    const labels = {Stopped:'Engine is stopped', Starting:'Starting your engine', Stopping:'Stopping your engine', BackingOff:'Waiting to try again', Halted:'Your engine needs attention'};
    const guidance = {Stopped:'Start the engine to load your model and connect an app.', Starting:'Your model is loading. The first start can take a little while.', Stopping:'The engine is shutting down. Connected apps stop receiving responses.', BackingOff:'The engine stopped unexpectedly. The supervisor will try to restart it.', Halted:'Automatic restarts have paused. Check Diagnose before starting again.'};
    let heading = ready ? 'Your engine is ready' : labels[state] || (state === 'Running' ? 'Engine is not ready yet' : 'Checking your engine');
    let guide = ready ? 'Your model is loaded and ready for requests.' : guidance[state] || 'The engine process is running, but its health check has not confirmed readiness. Check Diagnose if this continues.';
    if (!managed && !ready) { heading = 'Engine is not reachable'; guide = 'Check that the engine is running in its original application and that your connection settings match.'; }
    const modeText = ready ? 'Ready' : !managed && state !== 'Running' ? 'Not reachable' : state === 'Running' ? 'Not ready' : state === 'BackingOff' ? 'Retrying' : state;
    annunciate($('kpi-state-badge'), modeText, ready ? 'green' : state === 'Halted' ? 'red' : state === 'Stopped' ? '' : 'amber');
    const memo = $('engine-memo');
    memo.hidden = ready;
    memo.textContent = heading + '. ' + guide;
    memo.className = 'pad-line' + (state === 'Halted' ? ' red' : state === 'Stopped' ? '' : ' amber');
    $('kpi-state-main').textContent = state;
    const readiness = $('connect-readiness');
    readiness.textContent = pendingRestartPid != null ? 'Settings were saved. Restart the engine to apply them, then confirm it is ready at the configured address.' : ready ? 'Ready. Continue with your app’s connection settings below.' : heading + '. Check Monitor before connecting.';
    readiness.className = 'cdu-data ' + (ready && pendingRestartPid == null ? 'green' : 'amber');
    $('diagnostic-health-label').textContent = ready ? 'Engine health check passed' : 'Engine is not ready';
    $('diagnostic-health-label').className = 'cdu-data ' + (ready ? 'green' : 'amber');
    $('diagnostic-health-detail').textContent = ready ? 'The engine answered the latest health check.' : guide;
    updateEngineControls();

    $('kpi-pid').textContent = eng.pid || '—';
    $('kpi-uptime').textContent = formatUptime(eng.uptime_s);
    renderLaunchPlan(eng, s);

    // Memory readings
    const nv = s.device_memory || {};
    const dxgi = s.dxgi || {};
    const usedBytes = Number.isFinite(nv.used_bytes) && nv.total_bytes > 0 ? nv.used_bytes : null;
    const totalBytes = nv.total_bytes > 0 ? nv.total_bytes : null;
    const frac = totalBytes && usedBytes != null ? usedBytes / totalBytes : 0;
    const freeBytes = totalBytes != null && usedBytes != null ? totalBytes - usedBytes : null;
    const av = s.admin_vram || null;
    const breached = av && av.desktop_reserve && av.desktop_reserve.holding === false;
    countTo($('kpi-vram-main'), usedBytes == null ? NaN : usedBytes / GiB, v => v.toFixed(2));
    $('kpi-vram-pct').textContent = usedBytes == null ? 'unavailable' : `GiB · ${(frac * 100).toFixed(1)}%`;
    $('kpi-vram-free').textContent = gib(freeBytes);
    $('kpi-vram-free').className = breached ? 'red' : 'green';
    $('kpi-vram-total').textContent = gib(totalBytes);
    $('dt-gpu-name').textContent = dxgi.adapter_name || 'NVIDIA GPU';
    $('band-mem').textContent = usedBytes == null ? '—' : `${(usedBytes / GiB).toFixed(1)} / ${(totalBytes / GiB).toFixed(1)} GiB`;
    $('band-mem-bar').style.width = (frac * 100) + '%';
    $('band-free').textContent = freeBytes == null ? '' : `${(freeBytes / GiB).toFixed(1)} free`;
    $('band-free').className = 'band-sub' + (breached ? ' red' : '');
    const guidanceEl = $('memory-guidance');
    const memMessage = usedBytes == null ? 'Memory readings are unavailable. Open Diagnose for details.' : breached ? 'Available memory is below the desktop reserve. Review memory settings or close other GPU-heavy apps.' : '';
    guidanceEl.textContent = memMessage;
    guidanceEl.hidden = !memMessage;
    const reserveEl = $('dt-reserve'), reservationEl = $('dt-reservation');
    if (av && av.desktop_reserve) {
      const r = av.desktop_reserve;
      reserveEl.textContent = r.runtime_floor_bytes ? `${gib(r.runtime_floor_bytes)} floor · ${gib(r.free_bytes)} free` : `not enforced · ${gib(r.free_bytes)} free`;
      reserveEl.className = r.holding !== false ? 'green' : 'red';
      const p = av.plan || {};
      reservationEl.textContent = `${gib(p.runtime_reservation_bytes)} planned · ${mib(p.cuda_graph_rebuild_estimate_bytes)} graph rebuild estimate`;
    } else {
      reserveEl.textContent = s.admin_vram_note || '—';
      reserveEl.className = 'dim';
      reservationEl.textContent = '—';
    }
    $('dt-last-event').textContent = eng.last_event || '—';
    const kvEl = $('dt-kv-capacity');
    if (eng.kv_capacity_effective > 0) kvEl.textContent = eng.kv_capacity_effective.toLocaleString() + ' of ' + eng.kv_capacity_configured.toLocaleString() + ' tokens configured. ' + (eng.kv_capacity_note || '');
    else if (eng.kv_capacity_configured > 0) kvEl.textContent = eng.kv_capacity_configured.toLocaleString() + ' tokens, as configured';
    else kvEl.textContent = '—';
    $('dt-health-check').textContent = s.health ? `HTTP ${s.health.status} (${s.health.body || 'OK'})` : '—';
    $('dt-capacity-text').textContent = s.engine_capacity_line || 'No capacity report is available yet.';

    // Rates. The headline is the aggregate: with several requests at once it is
    // what the card produces; the per-response mean is what one reply feels like.
    const reqs = s.requests || {};
    const haveThr = reqs.log_available !== false;
    const thrTotal = reqs.decode_tok_s_total || 0;
    $('kpi-decode-explainer').textContent = reqs.decode_tok_s_mean > 0 ? `${reqs.decode_tok_s_mean.toFixed(1)} tok/s per response` : 'all requests together';
    // Zero is a reading, not a missing value; only an absent request log is unknown.
    const perSecond = v => `${rate(v)} tok/s`;
    const readTotal = haveThr ? reqs.prefill_tok_s_total || 0 : NaN;
    countTo($('thr-decode'), haveThr ? thrTotal : NaN, perSecond);
    countTo($('thr-prefill'), readTotal, perSecond);
    countTo($('band-gen'), haveThr ? thrTotal : NaN, perSecond);
    countTo($('band-read'), readTotal, perSecond);
    renderClients(reqs);
    const running = reqs.running_requests || 0;
    const lanes = Number(launchFlags['--max-concurrency']) || null;
    // Both figures come from the engine's latest 5-second report: the running count is taken
    // at the report, the rates over its window, so requests that just finished still count.
    $('kpi-thr-window').textContent = running > 0 || thrTotal > 0
      ? `${running}${lanes ? ' of ' + lanes : ''} running`
      : `Idle${lanes ? ' · ' + lanes + ' lane' + (lanes === 1 ? '' : 's') : ''}`;
    $('kpi-thr-window').className = 'display-source' + (running > 0 ? ' green' : '');
    $('band-run').textContent = `${running}${lanes ? ' / ' + lanes : ''}`;
    $('kpi-ttft').textContent = Number.isFinite(reqs.ttft_ms_mean) ? duration(reqs.ttft_ms_mean) : 'No data yet';
)HTML"
R"HTML(    $('kpi-done').textContent = reqs.done != null ? reqs.done.toLocaleString() : '—';

    // Speculative telemetry. A configured backend with no drafts yet is a
    // different state from no backend at all, and neither is an acceptance of 0.
    const specBackend = reqs.mtp_backend || '';
    const specOff = !specBackend || specBackend === 'none';
    $('dt-mtp-backend').textContent = specOff ? 'none — speculation disabled' : specBackend;
    $('dt-mtp-window').textContent = reqs.mtp_draft_window ? `${reqs.mtp_draft_window} draft token${reqs.mtp_draft_window === 1 ? '' : 's'}` : '—';
    $('kpi-mtp-badge').textContent = specOff ? 'Off' : specBackend.toUpperCase();
    if (specOff) {
      $('kpi-mtp-pct').textContent = '—'; $('kpi-mtp-ratio').textContent = 'not configured'; $('kpi-mtp-fallbacks').textContent = '—'; $('dt-mtp-rate-badge').textContent = 'Off';
    } else if (!reqs.mtp_drafted) {
      $('kpi-mtp-pct').textContent = '—'; $('kpi-mtp-ratio').textContent = 'no drafts yet'; $('kpi-mtp-fallbacks').textContent = '—'; $('dt-mtp-rate-badge').textContent = 'Idle';
    }
    $('mtp-bars-container').hidden = specOff || !reqs.mtp_drafted;
    if (!specOff && reqs.mtp_drafted) {
      const ratePct = ((reqs.mtp_last_accept_rate || 0) * 100).toFixed(1);
      $('kpi-mtp-pct').textContent = ratePct + '%';
      $('kpi-mtp-ratio').textContent = `${(reqs.mtp_accepted || 0).toLocaleString()} / ${(reqs.mtp_drafted || 0).toLocaleString()}`;
      $('kpi-mtp-fallbacks').textContent = reqs.mtp_fallback_steps || 0;
      $('dt-mtp-rate-badge').textContent = `${ratePct}% accepted`;
      // One bar per configured draft position; extra columns are hidden rather
      // than drawn empty, which would read as "accepted nothing at position 5".
      const pos = reqs.mtp_accepted_per_position || [];
      const maxPos = Math.max(1, ...pos);
      const bars = document.querySelectorAll('#mtp-bars-container .mtp-bar');
      document.querySelectorAll('#mtp-bars-container .mtp-col').forEach((col, idx) => { col.style.display = idx < Math.max(pos.length, reqs.mtp_draft_window || 0) ? '' : 'none'; });
      pos.forEach((pVal, idx) => {
        if (!bars[idx]) return;
        bars[idx].style.transform = `scaleY(${Math.max(0.04, pVal / maxPos)})`;
        bars[idx].parentElement.title = `Position ${idx + 1}: ${pVal} accepted (${((pVal / (reqs.mtp_accepted || 1)) * 100).toFixed(0)}%)`;
      });
    }
    $('dt-reuse-mix').textContent = `Seed ${reqs.reuse_seed || 0} · Append ${reqs.reuse_append || 0} · Reset ${reqs.reuse_full_reset || 0}`;

    // Preserve selection and scroll position when the log has not changed.
    const nextLog = s.log_tail || '';
    if (nextLog !== rawLogTail) { rawLogTail = nextLog; logContent.innerHTML = formatLogText(rawLogTail); }
    if (autoScroll && isDocumentVisible) logContent.scrollTop = logContent.scrollHeight;

    renderInsights(s.insights || {});
    if (isDocumentVisible) drawAll();
  }

  function formatLogText(rawText) {
    if (!rawText) return 'No engine activity yet. Start the engine to see its log here.';
    const filtered = rawText.split('\n').filter(l => {
      if (!l.trim()) return false;
      if (logSearchQuery && !l.toLowerCase().includes(logSearchQuery.toLowerCase())) return false;
      if (activeLogFilter === 'req') return /\[req |request id=|req#\d+/.test(l);
      if (activeLogFilter === 'throughput') return l.includes('throughput');
      if (activeLogFilter === 'warn') return /\[warn(?:ing)?\]|\[error\]|\bWARN\b|\bERROR\b|warning|failed|halt/i.test(l);
      return true;
    });
    if (!filtered.length) return 'No log entries match these filters. Choose All or clear the search.';
    return filtered.map(l => {
      let cls = 'log-line-dim';
      if (/\[error\]|\bERROR\b|failed|HALT/.test(l)) cls = 'log-line-error';
      else if (/\[warn\]|\bWARN(ING)?\b/.test(l)) cls = 'log-line-warn';
      else if (/\[req |request id=|req#\d+/.test(l)) cls = 'log-line-req';
      else if (/\[info\]|\bINFO\b/.test(l)) cls = 'log-line-info';
      return `<div class="${cls}">${esc(l)}</div>`;
    }).join('');
  }

  // ---------------------------------------------------------------------
  // Confirmation and lifecycle actions
  // ---------------------------------------------------------------------
  function showConfirmModal(actionName, btn) {
    lastFocusedTrigger = btn;
    pendingAction = {actionName, btn};
    const restart = actionName === 'restart' || actionName === 'save-restart';
    modalTitle.textContent = restart ? 'Restart your engine?' : 'Stop your engine?';
    modalDesc.textContent = restart ? 'Active requests are interrupted while the model reloads. Connected apps can send requests again when the engine is ready.' : 'Active requests are interrupted. Your model is unloaded until you start the engine again.';
    modalConfirmBtn.firstChild.textContent = actionName === 'save-restart' ? 'Insert and restart' : restart ? 'Restart engine' : 'Stop engine';
    confirmModal.showModal();
    modalCancelBtn.focus();
  }
  function hideConfirmModal() {
    confirmModal.close();
    pendingAction = null;
    if (lastFocusedTrigger) lastFocusedTrigger.focus();
  }
  confirmModal.addEventListener('cancel', () => { pendingAction = null; });
  modalCancelBtn.onclick = hideConfirmModal;
  modalConfirmBtn.onclick = () => {
    if (!pendingAction) return;
    const action = pendingAction.actionName, modelId = pendingAction.modelId;
    hideConfirmModal();
    if (action === 'bench') benchSend(true);
    else if (action === 'model') switchModel(modelId);
    else if (action === 'save-restart') cfgApply(true);
    else runAction(action);
  };
  async function runAction(actionName) {
    if (modelBusy || modelUncertain) return;
    actionBusy = true; updateEngineControls();
    notify(actionName === 'start' ? 'Starting your engine…' : actionName === 'stop' ? 'Stopping your engine…' : 'Restarting your engine…');
    try {
      const resp = await fetch(`/api/${actionName}`, {method:'POST', headers:{'X-NInfer-Supervisor':'1'}});
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || 'HTTP ' + resp.status);
      render(data);
      notify(actionName === 'start' ? 'Start requested. Readiness updates when your model finishes loading.' : actionName === 'stop' ? 'Stop requested. Watch the engine mode for completion.' : 'Restart requested. Your model will reload.');
    } catch (err) { notify('Could not ' + actionName + ' the engine: ' + err.message + '. Check the connection and try again.', true); }
    finally { actionBusy = false; updateEngineControls(); }
  }
  btnStart.onclick = () => runAction('start');
  btnStop.onclick = () => showConfirmModal('stop', btnStop);
  btnRestart.onclick = () => showConfirmModal('restart', btnRestart);

  // Log controls
  btnAutoscroll.onclick = () => {
    autoScroll = !autoScroll;
    btnAutoscroll.classList.toggle('active', autoScroll);
    btnAutoscroll.textContent = autoScroll ? 'Follow latest' : 'Following paused';
    btnAutoscroll.setAttribute('aria-pressed', String(autoScroll));
    if (autoScroll) logContent.scrollTop = logContent.scrollHeight;
  };
  btnCopyLog.onclick = () => copyText(btnCopyLog, rawLogTail, btnCopyLog.lastChild);
  document.querySelectorAll('.key-toggle[data-filter]').forEach(b => {
    b.onclick = () => {
      document.querySelectorAll('.key-toggle[data-filter]').forEach(x => { x.classList.remove('active'); x.setAttribute('aria-pressed', 'false'); });
      b.classList.add('active'); b.setAttribute('aria-pressed', 'true');
      activeLogFilter = b.dataset.filter;
      logContent.innerHTML = formatLogText(rawLogTail);
    };
  });
  logSearch.oninput = e => { logSearchQuery = e.target.value; logContent.innerHTML = formatLogText(rawLogTail); };

  document.addEventListener('visibilitychange', () => {
    isDocumentVisible = !document.hidden;
    exSync();
    if (isDocumentVisible) fetch('/api/state').then(r => { if (!r.ok) throw new Error(); return r.json(); }).then(s => { setConnection(true); render(s); }).catch(() => setConnection(false));
  });

  // ---------------------------------------------------------------------
  // Configuration editor.
  //
  // Every field is generated from /api/config's schema, which the C++ parameter
  // table produces. Nothing about a parameter -- its label, bounds, choices or
  // help text -- is written twice, so a knob added in the engine appears here
  // with the engine's own description rather than a copy that drifts.
  //
  // Two things are shown and deliberately not editable: the executable and the
  // working directory. The control gate on this page is a loopback check plus a
  // header, which stops a random web page from driving it -- it is not an
  // authorization system, and it should not be what stands between a browser and
  // choosing which binary runs.
  // ---------------------------------------------------------------------
  const cfgBody = $('cfg-body'), cfgErrors = $('cfg-errors'), cfgStatus = $('cfg-status');
  const cfgSave = $('cfg-save'), cfgSaveRestart = $('cfg-save-restart'), cfgRevert = $('cfg-revert'), cfgPath = $('cfg-path');
  const tabFields = {network: {engine: ['engine_host', 'engine_port', 'device'], supervisor: ['port', 'host', 'bind_any']}, api: {engine: ['api_key_file', 'request_log']}};

  function cfgDirtyCount() {
    return Object.keys(cfgEdits.params).length + Object.keys(cfgEdits.engine).length + Object.keys(cfgEdits.supervisor).length;
  }
  function tabDirtyCount(tab) {
    let n = 0;
    const f = tabFields[tab] || {};
    (f.engine || []).forEach(k => { if (k in cfgEdits.engine) n++; });
    (f.supervisor || []).forEach(k => { if (k in cfgEdits.supervisor) n++; });
    if (cfgData) (cfgData.schema || []).forEach(spec => { if (spec.group === tab && spec.key in cfgEdits.params) n++; });
    return n;
  }
  function updateArmed() {
    const el = $('fma-armed');
    const staged = cfgDirtyCount() > 0 || reserveDraft != null || feDirty();
    const text = modelBusy ? 'Loading preset' : pendingRestartPid != null ? 'Restart to apply' : staged ? 'Settings staged' : '';
    el.textContent = text;
    el.hidden = !text;
  }
  function updateTmpy() {
    updateArmed();
    const cdu = $('settings-cdu');
    let staged = false;
    if (settingsPage === 'presets') staged = !!(modelCatalog && modelCatalog.models.find(m => m.id === modelSelection && !m.active));
    else if (settingsPage === 'reserve') staged = reserveDraft != null;
    else if (settingsPage === 'frontends') staged = feDirty();
    else if (settingsPage === 'index') staged = cfgDirtyCount() > 0 || reserveDraft != null || feDirty();
    else staged = tabDirtyCount(settingsPage) > 0;
    cdu.classList.toggle('tmpy', staged);
    $('settings-tmpy').hidden = !staged;
  }
  function cfgSetDirtyUi() {
    const n = cfgDirtyCount();
    cfgStatus.className = 'cfg-status' + (n ? ' dirty' : '');
    cfgStatus.textContent = n ? `Temporary revision · ${n} staged change${n === 1 ? '' : 's'}` : 'No staged changes';
    $('settings-dirty').hidden = !n && reserveDraft == null && !feDirty();
    const writable = cfgData && cfgData.writable;
    cfgSave.disabled = !n || !writable || configBusy;
    cfgSaveRestart.disabled = !n || !writable || configBusy || cfgData.manages_engine === false;
    cfgSaveRestart.hidden = !!cfgData && cfgData.manages_engine === false;
    cfgRevert.disabled = !n || configBusy;
    cfgBody.inert = configBusy;
    renderSettingsIndex();
    updateModelControls();
  }
  function cfgEdit(section, key, value, original) {
    // An edit returned to its original value stops being an edit, so INSERT
    // reflects what would actually change rather than what was touched.
    const same = typeof original === 'boolean' ? value === original : String(value) === String(original == null ? '' : original);
    if (same) delete cfgEdits[section][key]; else cfgEdits[section][key] = value;
)HTML"
R"HTML(    cfgSetDirtyUi();
  }

  function tokenFieldValue(el) {
    if (!el) return {kind: 'missing'};
    const raw = el.value;
    if (raw === 'auto') return {kind: 'auto'};
    if (raw === '') return {kind: 'default'};
    if (/^\d+$/.test(raw)) { const n = Number(raw); if (Number.isInteger(n)) return {kind: 'num', n}; }
    return {kind: 'invalid'};
  }
  function parsedConcurrency() {
    const el = $('param-max_concurrency');
    const raw = el ? el.value : ((cfgData && cfgData.params || {}).max_concurrency || '');
    if (el && el.value === '') return {n: 1, explicit: false};
    if (/^\d+$/.test(String(raw))) { const n = Number(raw); if (n >= 1 && n <= 8) return {n, explicit: true}; }
    return {n: 1, explicit: false};
  }
  let capacityPairBusy = false;
  function updateCapacityPreview() {
    const preview = $('capacity-preview');
    if (!preview) return;
    const slotsEl = $('capacity-slots'), noteEl = $('capacity-preview-note');
    const ctx = tokenFieldValue($('param-max_context')), kv = tokenFieldValue($('param-kv_capacity'));
    const lanes = parsedConcurrency();
    const marks = [];
    let note = '';
    const laneWords = lanes.explicit ? (lanes.n + ' active lane' + (lanes.n === 1 ? '' : 's')) : 'the engine default of 1 active request';
    if (kv.kind === 'auto') {
      note = ctx.kind === 'num'
        ? 'Auto sizes the shared pool from free GPU memory at startup. Each request is still limited to ' + shortTokens(ctx.n) + '. Up to 8 requests can be active; that 8 is the Engine lane cap, not a memory estimate.'
        : 'Auto sizes the shared pool from free GPU memory at startup. Up to 8 requests can be active; that 8 is the Engine lane cap, not a memory estimate.';
    } else if (kv.kind === 'default') {
      note = 'The pool follows max context, so one full-length request fills it. Extra concurrent requests share that length. ' + (lanes.explicit ? lanes.n + ' of 8 Engine lanes are enabled.' : 'Omitting max concurrency uses 1 of 8 Engine lanes.');
      marks.push('fit');
      for (let i = 1; i < lanes.n; i++) marks.push('share');
    } else if (kv.kind === 'num' && ctx.kind === 'num') {
      const fit = ctx.n > 0 ? Math.floor(kv.n / ctx.n) : 0;
      const usable = Math.min(fit, lanes.n);
      for (let i = 0; i < usable; i++) marks.push('fit');
      for (let i = usable; i < lanes.n; i++) marks.push('share');
      if (fit < 1) note = 'This pool cannot hold one request at ' + shortTokens(ctx.n) + ' tokens. Raise KV capacity or lower max context.';
      else if (fit >= lanes.n) note = 'This pool holds ' + fit + ' full-length request' + (fit === 1 ? '' : 's') + ' at ' + shortTokens(ctx.n) + '. ' + laneWords + ' can each take a full-length request. 8 is the Engine maximum, not GPU memory.';
      else note = 'This pool holds ' + fit + ' full-length request' + (fit === 1 ? '' : 's') + ' at ' + shortTokens(ctx.n) + '. ' + laneWords + ' share the pool, so extra concurrent requests must be shorter. 8 is the Engine maximum.';
    } else if (kv.kind === 'num') {
      note = 'Shared pool: ' + shortTokens(kv.n) + ' tokens. Set max context to see how many full-length requests fit. Max concurrency is how many of the Engine’s 8 lanes are enabled.';
    } else {
      note = 'Set the KV pool first, then max context inside it. Max concurrency chooses how many of the Engine’s 8 lanes may be active.';
    }
    while (marks.length < 8) marks.push('');
    slotsEl.replaceChildren();
    marks.slice(0, 8).forEach(kind => { const tick = document.createElement('span'); if (kind) tick.className = kind; slotsEl.appendChild(tick); });
    noteEl.textContent = note;
  }
  function enforceCapacityPair(sourceId, reason) {
    if (capacityPairBusy) return;
    const ctxEl = $('param-max_context'), kvEl = $('param-kv_capacity');
    if (!ctxEl || !kvEl) return;
    const ctx = tokenFieldValue(ctxEl), kv = tokenFieldValue(kvEl);
    const clamp = reason === 'slider' || reason === 'blur' || reason === 'sync';
    const lanes = parsedConcurrency();
    const poolCap = ctx.kind === 'num' ? lanes.n * ctx.n : null;
    if (kv.kind === 'num' && ctx.kind === 'num' && ctx.n > kv.n) {
      if (clamp) { capacityPairBusy = true; ctxEl.value = String(kv.n); ctxEl.oninput(); capacityPairBusy = false; }
      else if (sourceId === 'param-max_context') ctxEl.setCustomValidity('Max context cannot exceed the KV pool (' + kv.n.toLocaleString() + ' tokens). Enlarge the pool first.');
      else if (sourceId === 'param-kv_capacity') kvEl.setCustomValidity('KV pool must cover max context (' + ctx.n.toLocaleString() + ' tokens).');
    } else if (kv.kind === 'num' && poolCap != null && kv.n > poolCap) {
      if (clamp) { capacityPairBusy = true; kvEl.value = String(poolCap); kvEl.oninput(); capacityPairBusy = false; }
      else kvEl.setCustomValidity('KV pool cannot exceed ' + lanes.n + ' × max context (' + poolCap.toLocaleString() + ' tokens). Raise max context or max concurrency.');
    }
    if (ctxEl._capacitySync) ctxEl._capacitySync(true);
    if (kvEl._capacitySync) kvEl._capacitySync(true);
    updateCapacityPreview();
  }
  function capacitySlider(spec, input) {
    const controls = document.createElement('div'); controls.className = 'cfg-token-controls';
    const summary = document.createElement('div'); summary.className = 'cfg-token-summary';
    const caption = document.createElement('span'); caption.textContent = spec.key === 'kv_capacity' ? 'Shared pool' : 'Per request';
    const output = document.createElement('output'); output.htmlFor = input.id;
    summary.append(caption, output);
    const slider = document.createElement('input');
    slider.type = 'range'; slider.className = 'cfg-token-slider'; slider.id = input.id + '-slider'; slider.min = 0; slider.step = 1;
    slider.setAttribute('aria-label', spec.label + ' common sizes');
    slider.setAttribute('aria-describedby', input.id + '-help');
    const scale = document.createElement('div'); scale.className = 'cfg-token-scale'; scale.setAttribute('aria-hidden', 'true');
    const automatic = spec.kind === 'int_or_auto';
    const specials = automatic ? ['', 'auto'] : [''];
    const presets = [];
    for (let n = automatic ? 8192 : 4096; n <= spec.max; n *= 2) if (n >= spec.min) presets.push(n);
    let stops = [];
    const stopLabel = value => value === '' ? 'Default' : value === 'auto' ? 'Auto' : Number.isFinite(Number(value)) ? shortTokens(Number(value)) : value;
    function numericMax() {
      if (spec.key === 'max_context') { const kv = tokenFieldValue($('param-kv_capacity')); return kv.kind === 'num' ? Math.min(spec.max, kv.n) : spec.max; }
      if (spec.key === 'kv_capacity') { const ctx = tokenFieldValue($('param-max_context')); if (ctx.kind === 'num') return Math.min(spec.max, parsedConcurrency().n * ctx.n); }
      return spec.max;
    }
    function sync(rebuild = true) {
      const raw = input.value, n = Number(raw), maxN = numericMax();
      const inRange = /^\d+$/.test(raw) && Number.isInteger(n) && n >= spec.min && n <= spec.max;
      const numeric = inRange && n <= maxN;
      const isAuto = automatic && raw === 'auto';
      let valid = numeric || isAuto || raw === '';
      let validity = valid ? '' : 'Enter a whole token count from ' + spec.min + ' to ' + spec.max + (automatic ? ', or auto.' : '.');
      if (inRange && n > maxN) {
        valid = false;
        validity = spec.key === 'kv_capacity' ? 'KV pool cannot exceed max concurrency × max context (' + maxN.toLocaleString() + ' tokens).' : 'Max context cannot exceed the KV pool (' + maxN.toLocaleString() + ' tokens). Enlarge the pool first.';
      }
      input.setCustomValidity(validity);
      if (rebuild) {
        const sizes = presets.filter(p => p <= maxN);
        if (numeric && !sizes.includes(n)) sizes.push(n);
        sizes.sort((a, b) => a - b);
        stops = [...specials, ...sizes.map(String)];
        slider.max = Math.max(0, stops.length - 1);
      }
      slider.disabled = input.disabled;
      const idx = numeric ? stops.indexOf(String(n)) : isAuto ? stops.indexOf('auto') : 0;
      slider.value = idx < 0 ? 0 : idx;
      const custom = numeric && !presets.includes(n);
      output.textContent = numeric ? (custom ? 'Custom · ' + n.toLocaleString() : shortTokens(n)) + ' tokens' : isAuto ? 'Auto' : raw === '' ? 'Engine default' : 'Check token count';
      slider.setAttribute('aria-valuetext', numeric ? n.toLocaleString() + ' tokens' + (custom ? ' (custom)' : '') : isAuto ? 'Auto: size from free memory' : raw === '' ? (automatic ? 'Engine default: follow max context' : 'Engine default') : 'Choose a valid size');
      scale.replaceChildren();
      const last = stops.length - 1;
      const tickAt = last <= 0 ? [0] : [0, Math.ceil(last / 2), last];
      const seen = new Set();
      for (const index of tickAt) {
        if (seen.has(index) || !stops[index] && stops[index] !== '') continue;
        seen.add(index);
        const tick = document.createElement('span');
        tick.textContent = stopLabel(stops[index]);
        tick.style.left = last === 0 ? '0%' : (index / last * 100) + '%';
        scale.appendChild(tick);
      }
    }
    input._capacitySync = sync;
    slider.oninput = () => { input.value = stops[Number(slider.value)]; input.oninput(); enforceCapacityPair(input.id, 'slider'); sync(false); };
    input.addEventListener('input', () => { sync(); if (!capacityPairBusy) enforceCapacityPair(input.id, 'type'); });
    input.addEventListener('blur', () => enforceCapacityPair(input.id, 'blur'));
    sync();
    controls.append(summary, slider, scale);
    return controls;
  }

  function fieldRow(spec, value) {
    spec = {...spec};
    const backend = Object.prototype.hasOwnProperty.call(cfgEdits.params, 'spec') ? cfgEdits.params.spec : (cfgData.params || {}).spec;
    if (spec.key === 'draft_tokens') spec.max = backend === 'mtp' ? 5 : 15;
    const original = value;
    if (Object.prototype.hasOwnProperty.call(cfgEdits.params, spec.key)) value = String(cfgEdits.params[spec.key]);
    const wrap = document.createElement('div'); wrap.className = 'cfg-field';
    const top = document.createElement('div'); top.className = 'cfg-field-top';
    const label = document.createElement('span'); label.className = 'cfg-label'; label.textContent = spec.label;
    const flag = document.createElement('span'); flag.className = 'cfg-flag'; flag.textContent = spec.flag;
    top.append(label, flag); wrap.appendChild(top);
    let input;
    if (spec.kind === 'flag') {
      input = document.createElement('input'); input.type = 'checkbox'; input.checked = value === 'true';
      const line = document.createElement('label'); line.className = 'cfg-check'; line.appendChild(input);
      const t = document.createElement('span'); t.textContent = input.checked ? 'Enabled' : 'Disabled'; line.appendChild(t);
      input.onchange = () => {
        t.textContent = input.checked ? 'Enabled' : 'Disabled';
        cfgEdit('params', spec.key, input.checked, original === 'true');
        input.classList.toggle('dirty', input.checked !== (original === 'true'));
      };
      wrap.appendChild(line);
    } else if (spec.kind === 'enum') {
      input = document.createElement('select'); input.className = 'cfg-select';
      // A parameter with choices is still optional: "not set" is a real state and
      // is not the same as the engine's default, which the engine picks itself.
      const none = document.createElement('option'); none.value = ''; none.textContent = spec.key === 'spec' ? 'Off — ordinary decoding' : '(not set)'; input.appendChild(none);
      spec.choices.forEach(choice => { const o = document.createElement('option'); o.value = choice; o.textContent = choice; input.appendChild(o); });
      input.value = value == null ? '' : value;
      input.onchange = () => {
        cfgEdit('params', spec.key, input.value, original || '');
        input.classList.toggle('dirty', input.value !== (original || ''));
)HTML"
R"HTML(        if (spec.key === 'spec') {
          if (!input.value) {
            cfgEdit('params', 'draft_tokens', '', (cfgData.params || {}).draft_tokens || '');
            cfgEdit('params', 'lm_head_draft', false, (cfgData.params || {}).lm_head_draft === 'true');
          } else {
            const limit = input.value === 'mtp' ? 5 : 15;
            const draft = Object.prototype.hasOwnProperty.call(cfgEdits.params, 'draft_tokens') ? cfgEdits.params.draft_tokens : (cfgData.params || {}).draft_tokens;
            if (!draft || Number(draft) > limit) cfgEdit('params', 'draft_tokens', String(input.value === 'mtp' ? limit : 7), (cfgData.params || {}).draft_tokens || '');
          }
          renderConfig();
          $('param-spec').focus();
        }
      };
      wrap.appendChild(input);
    } else {
      input = document.createElement('input'); input.className = 'cfg-input';
      input.type = spec.kind === 'int' ? 'number' : 'text';
      if (spec.kind === 'int') { input.min = spec.min; input.max = spec.max; }
      input.placeholder = spec.kind === 'int_or_auto' ? 'Engine default · ' + spec.min + '..' + spec.max + ' or auto' : 'Engine default';
      input.value = value == null ? '' : value;
      input.oninput = () => { cfgEdit('params', spec.key, input.value, original || ''); input.classList.toggle('dirty', input.value !== (original || '')); };
      wrap.appendChild(input);
    }
    const help = document.createElement('div'); help.className = 'cfg-help';
    input.id = 'param-' + spec.key;
    if (spec.key === 'draft_tokens' || spec.key === 'lm_head_draft') input.disabled = !backend;
    label.id = input.id + '-label';
    input.setAttribute('aria-labelledby', label.id);
    input.setAttribute('aria-describedby', input.id + '-help');
    input.classList.toggle('dirty', Object.prototype.hasOwnProperty.call(cfgEdits.params, spec.key));
    if (spec.key === 'max_context' || spec.key === 'kv_capacity') wrap.appendChild(capacitySlider(spec, input));
    help.id = input.id + '-help';
    help.textContent = (spec.kind === 'int' || spec.kind === 'int_or_auto') ? spec.help + ' Range ' + spec.min + '-' + spec.max + '.' : spec.help;
    if (spec.key === 'max_context' || spec.key === 'kv_capacity') {
      help.textContent += ' Pick a common size or enter an exact count. K = 1,024 tokens; M = 1,048,576.';
      help.textContent += spec.key === 'kv_capacity' ? ' This is the shared pool. It must cover one full-length request and cannot exceed max concurrency × max context. Engine default omits the flag and follows max context. Auto sizes from free memory.' : ' Cannot exceed an explicit KV pool; enlarge the pool first.';
    }
    wrap.appendChild(help);
    return wrap;
  }

  function plainField(section, key, label, value, opts) {
    opts = opts || {};
    const original = value;
    if (Object.prototype.hasOwnProperty.call(cfgEdits[section], key)) value = cfgEdits[section][key];
    const wrap = document.createElement('div'); wrap.className = 'cfg-field';
    const top = document.createElement('div'); top.className = 'cfg-field-top';
    const l = document.createElement('span'); l.className = 'cfg-label'; l.textContent = label; top.appendChild(l);
    if (opts.hint) { const h = document.createElement('span'); h.className = 'cfg-flag'; h.textContent = opts.hint; top.appendChild(h); }
    wrap.appendChild(top);
    let input;
    if (opts.type === 'bool') {
      input = document.createElement('input'); input.type = 'checkbox'; input.checked = !!value;
      const line = document.createElement('label'); line.className = 'cfg-check'; line.appendChild(input);
      const t = document.createElement('span'); t.textContent = input.checked ? 'Enabled' : 'Disabled'; line.appendChild(t);
      input.onchange = () => { t.textContent = input.checked ? 'Enabled' : 'Disabled'; cfgEdit(section, key, input.checked, !!original); input.classList.toggle('dirty', input.checked !== !!original); };
      wrap.appendChild(line);
    } else {
      input = document.createElement('input'); input.className = 'cfg-input';
      input.type = opts.type === 'int' ? 'number' : 'text';
      if (key === 'engine_port' || key === 'port') { input.min = 1; input.max = 65535; input.required = true; }
      if (key === 'device') { input.min = 0; input.required = true; }
      input.value = value == null ? '' : value;
      if (opts.readonly) input.readOnly = true;
      else input.oninput = () => {
        const v = opts.type === 'int' ? parseInt(input.value, 10) : input.value;
        cfgEdit(section, key, v, original);
        input.classList.toggle('dirty', String(input.value) !== String(original == null ? '' : original));
      };
      wrap.appendChild(input);
    }
    input.id = section + '-' + key;
    l.id = input.id + '-label';
    input.setAttribute('aria-labelledby', l.id);
    input.classList.toggle('dirty', Object.prototype.hasOwnProperty.call(cfgEdits[section], key));
    if (opts.help) {
      input.setAttribute('aria-describedby', input.id + '-help');
      const help = document.createElement('div'); help.className = 'cfg-help'; help.id = input.id + '-help'; help.textContent = opts.help;
      wrap.appendChild(help);
    }
    return wrap;
  }
  function cfgNote(text) { const n = document.createElement('div'); n.className = 'cfg-note'; n.textContent = text; return n; }

  // Web frontends editor. Edits stay local until INSERT, which posts the whole list
  // to /api/frontends; the Supervisor saves it and starts or stops listeners without
  // touching the engine.
  const feStateText = {serving: 'Serving', assets_missing: 'Files not found', port_in_use: 'Port in use', stopped: 'Stopped'};
  function feSaved() { return (cfgData && cfgData.frontends) || []; }
  function feDirty() { return feDraft !== null && JSON.stringify(feDraft) !== JSON.stringify(feSaved()); }
  function feStatusFor(fe) { return ((lastState || {}).frontends || []).find(s => s.name === fe.name && s.port === Number(fe.port)); }
  function feInput(label, value, attrs, onInput) {
    const wrap = document.createElement('label'); wrap.textContent = label;
    const input = document.createElement('input');
    Object.entries(attrs).forEach(([k, v]) => input.setAttribute(k, v));
    input.value = value == null ? '' : value;
    input.addEventListener('input', () => { onInput(input.value); feSetActions(); });
    wrap.appendChild(input);
    return wrap;
  }
  function feRefreshStatus() {
    if (!feDraft) return;
    feDraft.forEach((fe, i) => {
      const meta = $('fe-meta-' + i);
      if (!meta) return;
      const saved = feSaved().some(s => s.name === fe.name && s.dir === fe.dir && s.port === Number(fe.port));
      const st = saved ? feStatusFor(fe) : null;
      meta.replaceChildren();
      const state = document.createElement('span');
      state.textContent = !saved ? 'Not inserted yet' : st ? (feStateText[st.state] || st.state) + (st.reason ? ': ' + st.reason : '') : 'Starting';
      state.className = !saved ? 'amber' : st && st.state === 'serving' ? 'green' : 'amber';
      meta.appendChild(state);
      if (st && st.state === 'serving') {
        const open = document.createElement('a'); open.href = st.url; open.target = '_blank'; open.rel = 'noopener'; open.className = 'prompt';
        open.textContent = 'Open ' + st.url;
        meta.appendChild(open);
      }
    });
  }
  function feSetActions() {
    const save = $('fe-save');
    if (save) save.disabled = feBusy || !feDirty() || !(cfgData && cfgData.writable);
    const status = $('fe-status');
    if (status && !feBusy) { status.textContent = feDirty() ? 'Temporary revision · not inserted' : status.textContent || 'No staged changes'; status.className = 'cfg-status' + (feDirty() ? ' dirty' : ''); }
    $('settings-dirty').hidden = !cfgDirtyCount() && reserveDraft == null && !feDirty();
    updateTmpy();
  }
  function renderFrontendEditor() {
    if (feDraft === null) feDraft = JSON.parse(JSON.stringify(feSaved()));
    const editor = document.createElement('div'); editor.className = 'fe-editor';
    feDraft.forEach((fe, i) => {
      const row = document.createElement('div'); row.className = 'fe-row';
      row.appendChild(feInput('Name', fe.name, {required: '', maxlength: '64', placeholder: 'llama.cpp web UI'}, v => { fe.name = v; }));
      row.appendChild(feInput('Folder', fe.dir, {required: '', placeholder: 'C:\\path\\to\\built\\app', spellcheck: 'false'}, v => { fe.dir = v; }));
      row.appendChild(feInput('Port', fe.port, {required: '', type: 'number', min: '1', max: '65535'}, v => { fe.port = v === '' ? '' : Number(v); }));
      const remove = document.createElement('button'); remove.className = 'key'; remove.type = 'button'; remove.textContent = 'Remove';
      remove.onclick = () => { feDraft.splice(i, 1); renderConfig(); };
      row.appendChild(remove);
      const meta = document.createElement('div'); meta.className = 'fe-meta'; meta.id = 'fe-meta-' + i;
      row.appendChild(meta);
      editor.appendChild(row);
    });
    if (!feDraft.length) { const none = document.createElement('p'); none.className = 'empty-state'; none.textContent = 'No web frontend yet. Add one to chat with your engine in a browser.'; editor.appendChild(none); }
    const actions = document.createElement('div'); actions.className = 'cfg-actions';
    const discard = document.createElement('button'); discard.className = 'key'; discard.type = 'button'; discard.innerHTML = '<span aria-hidden="true">&lt;</span>Erase';
    discard.onclick = () => { feDraft = null; showCfgErrors(null); renderConfig(); };
    const status = document.createElement('span'); status.id = 'fe-status'; status.className = 'cfg-status'; status.setAttribute('role', 'status');
    status.textContent = feDirty() ? 'Temporary revision · not inserted' : 'No staged changes';
    const add = document.createElement('button'); add.className = 'key'; add.type = 'button'; add.textContent = 'Add frontend';
    add.disabled = feDraft.length >= 16;
    add.onclick = () => {
      const used = new Set(feDraft.map(f => Number(f.port)));
      // One past the highest configured port: the lowest free number is often another app's.
      let port = feDraft.length ? Math.max(...feDraft.map(f => Number(f.port) || 8099)) + 1 : 8100;
      while (used.has(port)) port++;
      feDraft.push({name: '', dir: '', port: port});
      renderConfig();
    };
    const save = document.createElement('button'); save.className = 'key key-insert'; save.type = 'button'; save.id = 'fe-save'; save.innerHTML = 'Insert and apply<span aria-hidden="true">&gt;</span>';
    save.onclick = feSave;
    actions.append(discard, status, add, save);
    editor.appendChild(cfgNote('Each folder holds a built web app with index.html at its root. llama.cpp publishes its web UI prebuilt on Hugging Face (ggml-org/llama-ui). Frontends listen on this computer only, and removing or changing one ends the chats open in it.'));
    editor.appendChild(actions);
    cfgBody.appendChild(editor);
    feRefreshStatus();
    feSetActions();
  }
  async function feSave() {
    if (feBusy) return;
    const invalid = [...cfgBody.querySelectorAll('.fe-row input')].find(el => !el.checkValidity());
    if (invalid) { invalid.reportValidity(); return; }
    feBusy = true; feSetActions(); showCfgErrors(null);
    const status = $('fe-status');
    status.textContent = 'Inserting…';
    try {
      const res = await fetch('/api/frontends', {method: 'POST', headers: {'Content-Type': 'application/json', 'X-NInfer-Supervisor': '1'}, body: JSON.stringify({frontends: feDraft.map(f => ({name: f.name.trim(), dir: f.dir.trim(), port: Number(f.port)}))})});
      let body = {};
      try { body = await res.json(); } catch (_) {}
      if (!res.ok) { showCfgErrors(body.details || [body.error || ('HTTP ' + res.status)]); status.textContent = 'Not inserted'; return; }
      cfgData.frontends = body.frontends || [];
      if (lastState) lastState.frontends = body.status || [];
)HTML"
R"HTML(      feDraft = null;
      renderConfig();
      renderFrontends((lastState || {}).frontends || []);
      $('fe-status').textContent = 'Inserted and applied. The engine kept running.';
      $('fe-status').className = 'cfg-status saved';
    } catch (err) {
      showCfgErrors([err.message || 'Could not save the frontends. Check the connection and try again.']);
      status.textContent = 'Not inserted';
    } finally { feBusy = false; feSetActions(); }
  }

  function renderConfig() {
    if (!cfgData || activeView !== 'settings' || !tabLabels[cfgTab] || cfgTab === 'presets' || cfgTab === 'reserve') { renderSettingsIndex(); return; }
    cfgBody.innerHTML = '';
    const help = {network:'Set the address apps use to reach your engine. Most local setups keep these values.', api:'Control API authentication and request logging.', capacity:'Size the shared KV pool first. Max context is how long one request may be inside that pool. Max concurrency chooses how many of the Engine’s 8 lanes may be active.', memory:'Advanced: these settings can change both memory use and generated responses.', features:'Advanced: available options depend on the model and engine build.', raw:'Read-only information about the model and the command used to launch it.', frontends:'Serve a web app, such as llama.cpp’s web UI, on its own local port beside your engine’s API. Inserting applies the change right away; the engine keeps running.'}[cfgTab];
    document.querySelector('[data-settings-page="config"] .cfg-actions').hidden = cfgTab === 'frontends' || cfgTab === 'raw';
    $('settings-section-title').textContent = tabLabels[cfgTab];
    $('settings-section-help').textContent = help;
    const eng = cfgData.engine || {}, sup = cfgData.supervisor || {};
    if (cfgTab === 'frontends') { renderFrontendEditor(); renderSettingsIndex(); return; }
    if (cfgTab === 'network') {
      cfgBody.appendChild(plainField('engine', 'engine_host', 'Engine host', eng.engine_host, {help: 'Address the engine listens on. The supervisor polls health here.'}));
      cfgBody.appendChild(plainField('engine', 'engine_port', 'Engine port', eng.engine_port, {type: 'int', help: 'Port the engine serves its OpenAI-compatible API on.'}));
      cfgBody.appendChild(plainField('engine', 'device', 'GPU device index', eng.device, {type: 'int', help: 'Which CUDA device the engine runs on.'}));
      cfgBody.appendChild(plainField('supervisor', 'port', 'Dashboard port', sup.port, {type: 'int', help: 'This page. Changing it needs the supervisor restarted, not just the engine.'}));
      cfgBody.appendChild(plainField('supervisor', 'host', 'Dashboard host', sup.host, {help: 'Must be a loopback address unless binding beyond loopback is enabled.'}));
      cfgBody.appendChild(plainField('supervisor', 'bind_any', 'Bind beyond loopback', sup.bind_any, {type: 'bool', help: 'Exposes engine start, stop and this configuration form to your network.'}));
      cfgBody.appendChild(cfgNote('The dashboard reaches the engine over loopback. Binding beyond loopback exposes engine lifecycle control and this form to anything that can reach the port; the only checks are a loopback peer test and a request header, which stop a stray web page rather than a determined caller.'));
      const advanced = document.createElement('details'); advanced.className = 'config-advanced';
      const summary = document.createElement('summary'); summary.textContent = 'Advanced connection settings'; advanced.appendChild(summary);
      const group = document.createElement('div'); group.className = 'cfg-grid';
      [...cfgBody.children].slice(2).forEach(el => group.appendChild(el));
      advanced.appendChild(group); cfgBody.appendChild(advanced);
      if (tabDirtyCount('network') > [...['engine_host', 'engine_port']].filter(k => k in cfgEdits.engine).length) advanced.open = true;
    } else if (cfgTab === 'api') {
      cfgBody.appendChild(plainField('engine', 'api_key_file', 'API key file', eng.api_key_file, {help: eng.api_key_present ? 'A key was read from this file. The key itself is never sent to this page.' : 'No key could be read from this path, so the engine accepts unauthenticated requests.'}));
      cfgBody.appendChild(plainField('engine', 'request_log', 'Request log path', eng.request_log, {help: 'Per-request JSONL. This is what fills the rates, calling apps, reuse and speculation readings.'}));
      (cfgData.schema || []).filter(x => x.group === 'api').forEach(spec => cfgBody.appendChild(fieldRow(spec, (cfgData.params || {})[spec.key])));
      cfgBody.appendChild(cfgNote('The API key is read from the file above by the supervisor and handed to the engine. It never appears on this page, in /api/config, or in the launch command.'));
    } else if (cfgTab === 'raw') {
      cfgBody.appendChild(plainField('engine', '_executable', 'Executable', eng.executable, {readonly: true, hint: 'file-only', help: 'Not editable here by design: this decides which binary runs.'}));
      cfgBody.appendChild(plainField('engine', '_workdir', 'Working directory', eng.workdir, {readonly: true, hint: 'file-only'}));
      cfgBody.appendChild(plainField('engine', '_artifact', 'Model artifact', eng.artifact, {readonly: true, hint: 'file-only'}));
      const pre = document.createElement('pre'); pre.className = 'cdu-pre'; pre.style.gridColumn = '1/-1'; pre.textContent = (eng.args || []).join(' ');
      cfgBody.appendChild(pre);
      if ((eng.passthrough || []).length) cfgBody.appendChild(cfgNote('Passed through unchanged because this build does not describe them: ' + eng.passthrough.join(' ') + '. They survive an edit made here.'));
    } else {
      const group = cfgTab;
      const specs = (cfgData.schema || []).filter(x => x.group === group && x.key !== 'desktop_reserve_gib');
      if (group === 'capacity') {
        const pair = document.createElement('div'); pair.className = 'cfg-capacity-pair';
        const kvSpec = specs.find(sp => sp.key === 'kv_capacity'), ctxSpec = specs.find(sp => sp.key === 'max_context');
        if (kvSpec) pair.appendChild(fieldRow(kvSpec, (cfgData.params || {}).kv_capacity));
        if (ctxSpec) pair.appendChild(fieldRow(ctxSpec, (cfgData.params || {}).max_context));
        const preview = document.createElement('div'); preview.className = 'cfg-capacity-preview'; preview.id = 'capacity-preview';
        const slots = document.createElement('div'); slots.className = 'cfg-capacity-slots'; slots.id = 'capacity-slots'; slots.setAttribute('aria-hidden', 'true');
        const note = document.createElement('p'); note.id = 'capacity-preview-note'; note.setAttribute('role', 'status');
        preview.append(slots, note); pair.appendChild(preview);
        cfgBody.appendChild(pair);
        specs.filter(sp => sp.key !== 'kv_capacity' && sp.key !== 'max_context').forEach(spec => cfgBody.appendChild(fieldRow(spec, (cfgData.params || {})[spec.key])));
        const conc = $('param-max_concurrency');
        if (conc) { conc.addEventListener('input', () => enforceCapacityPair(conc.id, 'type')); conc.addEventListener('blur', () => enforceCapacityPair(conc.id, 'blur')); }
        enforceCapacityPair('', 'sync');
      } else {
        specs.forEach(spec => cfgBody.appendChild(fieldRow(spec, (cfgData.params || {})[spec.key])));
        if (group === 'memory') {
          const reserveLink = document.createElement('a'); reserveLink.href = '#settings/reserve'; reserveLink.className = 'prompt';
          reserveLink.innerHTML = 'Desktop reserve<span aria-hidden="true">&gt;</span>';
          cfgBody.appendChild(reserveLink);
          cfgBody.appendChild(cfgNote('Quantization changes what the model outputs, not only what it costs. On this model the FP8 output head diverges from BF16 after about 95 greedy tokens and the FP8 KV cache diverges later. Measure before trusting a saving.'));
        }
      }
    }
    cfgSetDirtyUi();
  }

  // The settings index shows each page's current values, so most visits end there.
  function renderSettingsIndex() {
    if (!cfgData) return;
    const eng = cfgData.engine || {}, sup = cfgData.supervisor || {}, p = cfgData.params || {};
    const val = (section, key, base) => Object.prototype.hasOwnProperty.call(cfgEdits[section], key) ? cfgEdits[section][key] : base;
    const pv = key => Object.prototype.hasOwnProperty.call(cfgEdits.params, key) ? cfgEdits.params[key] : p[key];
    const set = (id, text, tab) => {
      const el = $(id); if (!el) return;
      const n = tab ? tabDirtyCount(tab) : 0;
      el.textContent = (n ? `${n} staged · ` : '') + text;
      el.className = 'lsk-sum' + (n ? ' amber' : '');
    };
    set('sum-network', `engine ${val('engine', 'engine_host', eng.engine_host)}:${val('engine', 'engine_port', eng.engine_port)} · dashboard ${val('supervisor', 'host', sup.host)}:${val('supervisor', 'port', sup.port)}`, 'network');
    set('sum-api', `${eng.api_key_present ? 'API key required' : 'No API key'} · ${eng.request_log ? 'request log on' : 'no request log'}`, 'api');
    const kv = pv('kv_capacity'), ctx = pv('max_context'), conc = pv('max_concurrency');
    set('sum-capacity', `KV ${kv ? (kv === 'auto' ? 'auto' : shortTokens(kv)) : 'default'} · ctx ${ctx ? shortTokens(ctx) : 'default'} · ${conc || 1} lane${Number(conc || 1) === 1 ? '' : 's'}`, 'capacity');
    const groupSummary = group => {
      const parts = (cfgData.schema || []).filter(sp => sp.group === group && sp.key !== 'desktop_reserve_gib').map(sp => {
        const v = pv(sp.key);
        if (v == null || v === '' || v === false || v === 'false') return null;
        return sp.kind === 'flag' ? sp.flag.replace(/^--/, '') : `${sp.flag.replace(/^--/, '')} ${v}`;
      }).filter(Boolean);
      return parts.length ? parts.join(' · ') : 'engine defaults';
    };
    set('sum-memory', groupSummary('memory'), 'memory');
    set('sum-features', groupSummary('features'), 'features');
    const fe = feSaved(), serving = ((lastState || {}).frontends || []).filter(f => f.state === 'serving').length;
    const feEl = $('sum-frontends');
    feEl.textContent = (feDirty() ? 'staged · ' : '') + (fe.length ? `${fe.length} configured · ${serving} serving` : 'none configured');
    feEl.className = 'lsk-sum' + (feDirty() ? ' amber' : '');
    $('sum-raw').textContent = eng.artifact ? 'read-only · ' + eng.artifact.split(/[\\/]/).pop() : 'read-only';
  }

  async function cfgPost(payload) {
    const res = await fetch('/api/config', {method: 'POST', headers: {'Content-Type': 'application/json', 'X-NInfer-Supervisor': '1'}, body: JSON.stringify(payload)});
    let body = {};
    try { body = await res.json(); } catch (_) {}
    return {ok: res.ok, status: res.status, body};
  }
  function showCfgErrors(list) {
    if (!list || !list.length) { cfgErrors.hidden = true; return; }
    cfgErrors.hidden = false;
    cfgErrors.innerHTML = '<strong>The configuration was not saved.</strong><ul>' + list.map(e => '<li>' + esc(e) + '</li>').join('') + '</ul>';
  }
  async function cfgApply(thenRestart) {
    if (configBusy || modelBusy || modelUncertain) return;
    const invalid = [...cfgBody.querySelectorAll('input, select')].find(el => !el.checkValidity());
    if (invalid) { invalid.reportValidity(); return; }
    configBusy = true; cfgSetDirtyUi();
    showCfgErrors(null);
    try {
      const r = await cfgPost({params: cfgEdits.params, engine: cfgEdits.engine, supervisor: cfgEdits.supervisor});
      if (!r.ok) { showCfgErrors(r.body.details || [r.body.error || ('HTTP ' + r.status)]); return; }
      cfgData = r.body;
      updateConnectionGuide(cfgData);
      updateEngineControls();
      cfgEdits.params = {}; cfgEdits.engine = {}; cfgEdits.supervisor = {};
      renderConfig();
      cfgStatus.className = 'cfg-status saved';
      const needsEngine = r.body.engine_restart_required, needsSup = r.body.supervisor_restart_required;
)HTML"
R"HTML(      cfgStatus.textContent = 'Inserted' + (needsEngine ? ' · takes effect when the engine restarts' : '') + (needsSup ? ' · dashboard host or port needs the supervisor restarted' : '');
      if (needsEngine) setPendingRestart(((lastState || {}).engine || {}).pid || 0);
      if (needsEngine || needsSup) notify('Settings inserted. Some changes need a restart before they take effect.');
      if (thenRestart && needsEngine) {
        const restart = await fetch('/api/restart', {method: 'POST', headers: {'X-NInfer-Supervisor': '1'}});
        if (!restart.ok) throw new Error('Settings were inserted, but the engine could not restart. Try Restart on the mode row.');
        cfgStatus.textContent = 'Inserted. Restart requested; wait for the engine to become ready.';
      }
    } catch (err) { showCfgErrors([err.message || 'Could not save settings. Check the connection and try again.']); }
    finally {
      configBusy = false; cfgBody.inert = false;
      cfgRevert.disabled = !cfgDirtyCount();
      cfgSave.disabled = !cfgDirtyCount() || !cfgData.writable;
      cfgSaveRestart.disabled = !cfgDirtyCount() || !cfgData.writable || cfgData.manages_engine === false;
      updateTmpy();
    }
  }
  cfgSave.onclick = () => cfgApply(false);
  cfgSaveRestart.onclick = () => showConfirmModal('save-restart', cfgSaveRestart);
  cfgRevert.onclick = () => { cfgEdits.params = {}; cfgEdits.engine = {}; cfgEdits.supervisor = {}; showCfgErrors(null); renderConfig(); };

  function updateConnectionGuide(d) {
    const host = d.engine.engine_host;
    const localHost = ['0.0.0.0', '::', 'localhost', '127.0.0.1', '::1'].includes(host) ? '127.0.0.1' : host;
    const apiUrl = 'http://' + (localHost.includes(':') ? '[' + localHost + ']' : localHost) + ':' + d.engine.engine_port + '/v1';
    $('api-address').value = apiUrl;
    $('overview-endpoint').textContent = apiUrl.replace(/^http:\/\//, '');
    $('fma-api-copy').title = 'Copy ' + apiUrl;
    annunciate($('model-label'), modelDisplayName(d.engine));
    $('model-label').title = (modelCatalog && modelCatalog.active_artifact_label) || d.engine.artifact || '';
    $('auth-guidance').textContent = d.engine.api_key_present ? 'Required. Enter the key configured for this engine into your app. The dashboard never displays or copies that key.' : 'None configured. If your app insists on an API key, enter a placeholder such as local; this does not enable authentication.';
  }

  // ---------------------------------------------------------------------
  // API test bench. Every route the engine exposes, sent through the
  // supervisor's /api/engine bench: same origin, the supervisor's key, streams
  // unbuffered. The supervisor's own read routes are called directly.
  // ---------------------------------------------------------------------
  let benchServedModel = '';
  let benchLastResponseId = '';
  let benchCurrent = null, benchExample = 0, benchAbort = null, benchTab = 'body', benchReady = false;
  const benchHistory = [];
  let benchView = {status: null, timing: '', text: null, body: '', events: [], headers: [], kind: ''};
  const benchModel = () => benchServedModel || 'local';
  const chatMessages = () => [{role: 'system', content: 'Answer concisely.'}, {role: 'user', content: 'What is speculative decoding?'}];
  const ep = (group, method, path, title, extra) => Object.assign({group, method, path, title, examples: []}, extra || {});
  const benchEndpoints = [
    ep('Status', 'GET', '/health', 'Engine readiness', {note: 'Always open, even when an API key is set. 200 when ready; 503 after an engine-wide failure.'}),
    ep('Status', 'GET', '/metrics', 'Prometheus metrics', {note: 'Counters, gauges and histograms in the Prometheus text format.'}),
    ep('Status', 'GET', '/v1/models', 'Served models', {note: 'The served id, its aliases and the rerank id, with the effective max_model_len.'}),
    ep('Status', 'GET', '/v1/models/{model}', 'One model'),
    ep('Status', 'GET', '/props', 'Server properties', {note: 'llama.cpp-compatible properties used by its web UI.'}),
    ep('OpenAI', 'POST', '/v1/chat/completions', 'Chat completions', {stream: 'openai', examples: [
      {name: 'Basic', body: () => ({model: benchModel(), messages: chatMessages(), max_tokens: 128})},
      {name: 'Streaming', body: () => ({model: benchModel(), messages: chatMessages(), max_tokens: 256, stream: true, stream_options: {include_usage: true}})},
      {name: 'No thinking', body: () => ({model: benchModel(), messages: chatMessages(), max_tokens: 128, chat_template_kwargs: {enable_thinking: false}})},
      {name: 'JSON schema', body: () => ({model: benchModel(), messages: [{role: 'user', content: 'Name a city and its country.'}], max_tokens: 128, response_format: {type: 'json_schema', json_schema: {name: 'place', schema: {type: 'object', properties: {city: {type: 'string'}, country: {type: 'string'}}, required: ['city', 'country']}}}})},
      {name: 'Choice', body: () => ({model: benchModel(), messages: [{role: 'user', content: 'Is the sky blue on a clear day? Answer yes or no.'}], max_tokens: 4, structured_outputs: {choice: ['yes', 'no']}})},
      {name: 'Logprobs', body: () => ({model: benchModel(), messages: [{role: 'user', content: 'Say hello.'}], max_tokens: 8, logprobs: true, top_logprobs: 3})},
      {name: 'Tool call', body: () => ({model: benchModel(), messages: [{role: 'user', content: 'What is the weather in Lisbon?'}], max_tokens: 256, tools: [{type: 'function', function: {name: 'get_weather', description: 'Current weather for a city', parameters: {type: 'object', properties: {city: {type: 'string'}}, required: ['city']}}}]})}]}),
    ep('OpenAI', 'POST', '/v1/responses', 'Responses', {stream: 'responses', examples: [
      {name: 'Basic', body: () => ({model: benchModel(), instructions: 'Answer concisely.', input: 'What is speculative decoding?', max_output_tokens: 128, store: true})},
      {name: 'Streaming', body: () => ({model: benchModel(), instructions: 'Answer concisely.', input: 'What is speculative decoding?', max_output_tokens: 256, stream: true})}]}),
    ep('OpenAI', 'POST', '/v1/responses/input_tokens', 'Count input tokens', {examples: [{name: 'Basic', body: () => ({model: benchModel(), input: 'Count this prompt.'})}]}),
    ep('OpenAI', 'GET', '/v1/responses/{response_id}', 'Stored response', {note: 'A response stored by an earlier Responses call; the last id the bench saw is filled in.'}),
    ep('OpenAI', 'GET', '/v1/responses/{response_id}/input_items?limit=20&order=desc', 'Stored input items'),
    ep('OpenAI', 'DELETE', '/v1/responses/{response_id}', 'Delete stored response'),
    ep('OpenAI', 'POST', '/v1/responses/{response_id}/cancel', 'Cancel response', {note: 'Always answers with an error: background execution is not supported.', examples: [{name: 'Empty', body: () => ({})}]}),
    ep('OpenAI', 'POST', '/v1/responses/compact', 'Compact', {note: 'Always answers compaction_not_supported.', examples: [{name: 'Basic', body: () => ({model: benchModel()})}]}),
    ep('Anthropic', 'POST', '/v1/messages', 'Messages', {stream: 'anthropic', examples: [
      {name: 'Basic', body: () => ({model: benchModel(), max_tokens: 128, messages: [{role: 'user', content: 'Explain prefix reuse in one sentence.'}]})},
      {name: 'Streaming', body: () => ({model: benchModel(), max_tokens: 256, stream: true, messages: [{role: 'user', content: 'Explain prefix reuse in one sentence.'}]})},
      {name: 'Thinking', body: () => ({model: benchModel(), max_tokens: 512, system: 'Answer concisely.', thinking: {type: 'enabled', budget_tokens: 256}, messages: [{role: 'user', content: 'Why does prompt caching help?'}]})}]}),
    ep('Anthropic', 'POST', '/v1/messages/count_tokens', 'Count tokens', {examples: [{name: 'Basic', body: () => ({model: benchModel(), messages: [{role: 'user', content: 'Count this prompt.'}]})}]}),
    ep('NInfer', 'POST', '/v1/score', 'Score answers', {note: 'Closed-set scoring of 1 to 256 questions against one shared prefix. No streaming.', examples: [{name: 'Basic', body: () => ({model: benchModel(), messages: [{role: 'system', content: 'Invoice 1042 was due on 2026-09-30 and is unpaid. Today is 2026-10-11.'}], questions: [{id: 'q1', content: 'Is the invoice overdue? Answer A for yes, B for no.', candidates: ['A', 'B']}], candidates: ['A', 'B'], top_logprobs: 0, chat_template_kwargs: {enable_thinking: false}})}]}),
    ep('NInfer', 'POST', '/v1/rerank', 'Rerank documents', {note: 'Scores each document; latency grows with the number of documents.', examples: [{name: 'Basic', body: () => ({query: 'where is the invoice?', documents: ['The invoice is in the drawer.', {text: 'The weather is clear.'}], top_n: 2, return_documents: true})}]}),
    ep('NInfer', 'POST', '/v1/systemone', 'System One decisions', {note: 'Structured decisions read from next-token probabilities; no text is generated. POST /systemone is the same route.', examples: [{name: 'Ticket triage', body: () => ({model: 'jev-latest', state: "Customer: 'I was charged twice for order #9482. Please refund immediately!'", questions: {is_urgent: {type: 'noul', instructions: 'Is this customer inquiry urgent?'}, department: {type: 'choice', instructions: 'Which department should handle this ticket?', criteria: {billing: 'Charges and refunds', shipping: 'Delivery', general: null}}, frustration: {type: 'score', instructions: 'How frustrated is the customer?', criteria: ['Calm', 'Mildly annoyed', 'Very upset']}}})}]}),
    ep('Admin', 'GET', '/admin/vram', 'Memory plan and device memory'),
    ep('Admin', 'GET', '/admin/stats', 'Runtime counters'),
    ep('Admin', 'GET', '/admin/experts', 'Expert use', {note: 'How often each routed expert was chosen, per layer, decayed with a half-life. Answers 404 expert_stats_disabled unless the engine was started with --expert-stats. Add ?totals=1 for the counts since start.'}),
    ep('Admin', 'POST', '/admin/quiesce', 'Quiesce', {disruptive: true, note: 'Diagnostic: stops admission, drains active requests and holds the engine idle, then resumes. Your apps’ requests wait while it runs.', examples: [{name: 'Hold 0 ms', body: () => ({hold_ms: 0})}]}),
    ep('Supervisor', 'GET', '/api/state', 'Supervisor state', {local: true, note: 'This dashboard’s own state, as Monitor reads it every second.'}),
    ep('Supervisor', 'GET', '/api/insights', 'Findings report', {local: true}),
    ep('Supervisor', 'GET', '/api/config', 'Configuration', {local: true, note: 'The API key itself is never included.'}),
    ep('Supervisor', 'GET', '/api/models', 'Model presets', {local: true}),
  ];
  benchEndpoints.forEach((e, i) => { e.id = i; });
  const benchBody = $('bench-body'), benchPath = $('bench-path'), benchOut = $('bench-out');

  function benchFillPath(path) {
    return path.replace('{model}', encodeURIComponent(benchModel())).replace('{response_id}', benchLastResponseId || '{response_id}');
  }
  function renderBenchList() {
    const q = $('bench-filter').value.trim().toLowerCase();
    const list = $('bench-list');
    list.replaceChildren();
    let group = '';
    benchEndpoints.forEach(e => {
      if (q && !(e.path + ' ' + e.title + ' ' + e.method + ' ' + e.group).toLowerCase().includes(q)) return;
      if (e.group !== group) {
        group = e.group;
        const h = document.createElement('h2'); h.className = 'bench-group'; h.textContent = group; list.append(h);
      }
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'bench-ep'; b.dataset.ep = e.id;
      if (benchCurrent && benchCurrent.id === e.id) b.setAttribute('aria-current', 'true');
      b.innerHTML = `<span class="bench-m">${esc(e.method)}</span><span class="bench-p">${esc(e.path.split('?')[0]).replace(/\//g, '/<wbr>')}</span><span class="bench-t">${esc(e.title)}${e.disruptive ? ' <span class="amber">· disruptive</span>' : ''}</span>`;
      b.onclick = () => benchSelect(e, 0);
      list.append(b);
    });
)HTML"
R"HTML(    if (!list.children.length) list.innerHTML = '<p class="cdu-small">No endpoint matches this filter.</p>';
  }
  function benchSetBody(obj) {
    benchBody.value = obj == null ? '' : JSON.stringify(obj, null, 2);
    benchValidate();
  }
  function benchSelect(e, exampleIndex) {
    benchCurrent = e; benchExample = exampleIndex;
    $('bench-method').textContent = e.method;
    $('bench-method').className = 'bench-method' + (e.disruptive ? ' amber' : '');
    benchPath.value = benchFillPath(e.path);
    $('bench-note').textContent = e.note || '';
    $('bench-note').hidden = !e.note;
    const exBox = $('bench-examples');
    exBox.replaceChildren();
    e.examples.forEach((x, i) => {
      const k = document.createElement('button');
      k.type = 'button'; k.className = 'cdu-tab'; k.textContent = x.name;
      k.setAttribute('aria-pressed', String(i === exampleIndex));
      k.onclick = () => benchSelect(e, i);
      exBox.append(k);
    });
    const hasBody = e.method !== 'GET' && e.method !== 'DELETE';
    benchBody.hidden = $('bench-body-state').hidden = $('bench-format').hidden = !hasBody;
    $('bench-stream-wrap').hidden = !e.stream;
    benchSetBody(hasBody && e.examples[exampleIndex] ? e.examples[exampleIndex].body() : hasBody ? {} : null);
    renderBenchList();
  }
  function benchValidate() {
    const state = $('bench-body-state');
    const raw = benchBody.value.trim();
    let ok = true, msg = raw ? 'Valid JSON' : 'Empty body';
    if (raw) { try { const v = JSON.parse(raw); if (benchCurrent && benchCurrent.stream) $('bench-stream').checked = v && v.stream === true; } catch (err) { ok = false; msg = 'Not valid JSON: ' + err.message; } }
    benchBody.classList.toggle('invalid', !ok);
    state.textContent = msg + ' · Ctrl+Enter sends';
    state.className = 'bench-body-state' + (ok ? '' : ' red');
    return ok;
  }
  benchBody.addEventListener('input', benchValidate);
  benchBody.addEventListener('keydown', e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); benchSend(); } });
  benchPath.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); benchSend(); } });
  $('bench-stream').onchange = () => {
    let v = {};
    try { v = JSON.parse(benchBody.value || '{}'); } catch (_) { return; }
    if ($('bench-stream').checked) v.stream = true; else delete v.stream;
    if (!$('bench-stream').checked && v.stream_options) delete v.stream_options;
    benchSetBody(v);
  };
  $('bench-format').onclick = () => { try { benchSetBody(JSON.parse(benchBody.value || '{}')); } catch (_) { benchValidate(); } };
  $('bench-filter').addEventListener('input', renderBenchList);
  $('bench-curl').onclick = e => {
    if (!benchCurrent) return;
    const path = benchPath.value.trim();
    const base = benchCurrent.local ? location.origin : ($('api-address').value || '').replace(/\/v1\/?$/, '');
    const parts = [`curl${benchCurrent.stream && $('bench-stream').checked ? ' -N' : ''} -X ${benchCurrent.method} '${base}${path}'`];
    if (!benchCurrent.local && cfgData && cfgData.engine && cfgData.engine.api_key_present) parts.push(`-H "Authorization: Bearer $NINFER_API_KEY"`);
    if (!benchBody.hidden && benchBody.value.trim()) {
      let compact = benchBody.value.trim();
      try { compact = JSON.stringify(JSON.parse(compact)); } catch (_) {}
      parts.push(`-H 'Content-Type: application/json'`, `-d '${compact.replace(/'/g, "'\\''")}'`);
    }
    copyText(e.currentTarget, parts.join(' \\\n  '), e.currentTarget.lastChild);
  };

  // Stream dialects: OpenAI chat chunks, Responses semantic events, Anthropic message events.
  function benchDelta(kind, name, data, acc) {
    if (kind === 'openai') {
      const d = data && data.choices && data.choices[0] && data.choices[0].delta;
      if (d) { if (d.reasoning_content) acc.reasoning += d.reasoning_content; if (d.content) acc.text += d.content; if (d.tool_calls) acc.tools += JSON.stringify(d.tool_calls); }
    } else if (kind === 'responses') {
      const type = (data && data.type) || name;
      if (type === 'response.output_text.delta') acc.text += data.delta || '';
      else if (/reasoning.*delta/.test(type)) acc.reasoning += data.delta || '';
      if (data && data.response && data.response.id) benchLastResponseId = data.response.id;
    } else if (kind === 'anthropic') {
      if (data && data.type === 'content_block_delta' && data.delta) {
        if (data.delta.type === 'text_delta') acc.text += data.delta.text || '';
        else if (data.delta.type === 'thinking_delta') acc.reasoning += data.delta.thinking || '';
        else if (data.delta.type === 'input_json_delta') acc.tools += data.delta.partial_json || '';
      }
    }
  }
  // The readable answer of a finished, non-streamed call, when the shape is a known one.
  function benchTextOf(json) {
    if (!json || typeof json !== 'object') return null;
    if (Array.isArray(json.choices) && json.choices[0] && json.choices[0].message) {
      const m = json.choices[0].message;
      return {reasoning: m.reasoning_content || '', text: m.content || '', tools: m.tool_calls ? JSON.stringify(m.tool_calls, null, 2) : ''};
    }
    if (json.object === 'response' && Array.isArray(json.output)) {
      if (json.id) benchLastResponseId = json.id;
      const acc = {reasoning: '', text: '', tools: ''};
      json.output.forEach(item => {
        if (item.type === 'message') (item.content || []).forEach(c => { if (c.text) acc.text += c.text; });
        else if (item.type === 'reasoning') (item.content || item.summary || []).forEach(c => { if (c.text) acc.reasoning += c.text; });
        else if (item.type === 'function_call') acc.tools += `${item.name}(${item.arguments || ''})\n`;
      });
      return acc;
    }
    if (json.type === 'message' && Array.isArray(json.content)) {
      const acc = {reasoning: '', text: '', tools: ''};
      json.content.forEach(b => { if (b.type === 'text') acc.text += b.text; else if (b.type === 'thinking') acc.reasoning += b.thinking || ''; else if (b.type === 'tool_use') acc.tools += `${b.name}(${JSON.stringify(b.input)})\n`; });
      return acc;
    }
    return null;
  }
  // Compact JSON has no break opportunities; a space after each separator gives the wrap
  // points at structure boundaries instead of mid-token.
  function benchWrapJson(text) {
    let v;
    try { v = JSON.parse(text); } catch (_) { return text; }
    return JSON.stringify(v).replace(/("(?:[^"\\]|\\.)*")|([,:])/g, (m, str, sep) => str ? str : sep + ' ');
  }
  function benchJsonHtml(text) {
    let json;
    try { json = JSON.parse(text); } catch (_) { return esc(text); }
    return esc(JSON.stringify(json, null, 2)).replace(/(&quot;(?:[^&]|&(?!quot;))*?&quot;)(\s*:)?|\b(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b|\b(true|false|null)\b/g,
      (m, str, colon, num, lit) => str ? (colon ? `<span class="j-k">${str}</span>${colon}` : `<span class="j-s">${str}</span>`) : num ? `<span class="j-n">${num}</span>` : `<span class="j-l">${lit}</span>`);
  }
  function renderBenchOut() {
    document.querySelectorAll('[data-bench-tab]').forEach(t => t.setAttribute('aria-pressed', String(t.dataset.benchTab === benchTab)));
    $('bench-event-count').textContent = benchView.events.length ? benchView.events.length.toLocaleString() : '';
    const code = $('bench-code');
    code.textContent = benchView.status == null ? '—' : benchView.status === 0 ? 'Failed' : 'HTTP ' + benchView.status;
    code.className = 'bench-code' + (benchView.status == null ? '' : benchView.status >= 200 && benchView.status < 300 ? ' green' : benchView.status >= 500 || benchView.status === 0 ? ' red' : ' amber');
    $('bench-timing').textContent = benchView.timing;
    let html = '';
    if (benchTab === 'text') {
      const t = benchView.text;
      if (!t) html = '<span class="bench-empty">No readable text for this answer. See Body.</span>';
      else html = (t.reasoning ? `<span class="bench-label">Reasoning</span><span class="bench-reasoning">${esc(t.reasoning)}</span>\n\n` : '') + (t.tools ? `<span class="bench-label">Tool calls</span>${esc(t.tools)}\n\n` : '') + `<span class="bench-label">Answer</span>${esc(t.text || '')}`;
    } else if (benchTab === 'events') {
      html = benchView.events.length ? benchView.events.map(ev => `<span class="bench-ev"><span class="bench-ev-t">+${ev.t} ms</span><span class="bench-ev-n" title="${esc(ev.name)}">${esc(ev.name)}</span><span>${esc(benchWrapJson(ev.data))}</span></span>`).join('') : '<span class="bench-empty">This answer was not a stream.</span>';
    } else if (benchTab === 'headers') {
      html = benchView.headers.length ? benchView.headers.map(([k, v]) => `<span class="j-k">${esc(k)}</span>: ${esc(v)}`).join('\n') : '<span class="bench-empty">No response yet.</span>';
    } else {
      html = benchView.body ? (benchView.kind === 'json' ? benchJsonHtml(benchView.body) : esc(benchView.body)) : '<span class="bench-empty">No body.</span>';
    }
    const stick = benchOut.scrollTop + benchOut.clientHeight >= benchOut.scrollHeight - 8;
    benchOut.innerHTML = html;
    if (benchAbort && stick) benchOut.scrollTop = benchOut.scrollHeight;
  }
  document.querySelectorAll('[data-bench-tab]').forEach(t => { t.onclick = () => { benchTab = t.dataset.benchTab; renderBenchOut(); }; });

  function benchMs(ms) { return ms >= 1000 ? (ms / 1000).toFixed(2) + ' s' : Math.round(ms) + ' ms'; }
  async function benchSend(confirmed) {
    if (!benchCurrent || benchAbort) return;
    const e = benchCurrent;
    const path = benchPath.value.trim();
    const placeholder = path.match(/\{[^}]+\}/);
    if (!path.startsWith('/')) { benchView = {...benchView, status: null, timing: 'The path must start with /.'}; renderBenchOut(); return; }
    if (placeholder) { benchView = {...benchView, status: null, timing: `Replace ${placeholder[0]} in the path first.`}; renderBenchOut(); return; }
    const hasBody = !benchBody.hidden;
    if (hasBody && !benchValidate()) { benchView = {...benchView, status: null, timing: 'Fix the JSON body before sending.'}; renderBenchOut(); return; }
    if (e.disruptive && !confirmed) {
      showConfirmModal('bench', $('bench-send'));
      modalTitle.innerHTML = `Send <span class="dialog-path">${esc(e.method)} ${esc(path.split('?')[0])}</span>?`;
      modalDesc.textContent = e.note || 'This request changes the engine’s state.';
      modalConfirmBtn.firstChild.textContent = 'Send';
      return;
    }
    const url = (e.local ? '' : '/api/engine') + path;
    const ctl = new AbortController();
    benchAbort = ctl;
    $('bench-send').hidden = true; $('bench-stop').hidden = false;
    const kind = e.stream || '';
    benchView = {status: null, timing: 'Waiting for the engine…', text: null, body: '', events: [], headers: [], kind: ''};
    renderBenchOut();
    const t0 = performance.now();
    let tHead = 0, tFirst = 0, bytes = 0;
    const record = {endpoint: e.id, method: e.method, path, body: hasBody ? benchBody.value : '', example: benchExample};
    try {
      const res = await fetch(url, {method: e.method, signal: ctl.signal, headers: Object.assign({'X-NInfer-Supervisor': '1'}, hasBody ? {'Content-Type': 'application/json'} : {}), body: hasBody && benchBody.value.trim() ? benchBody.value : undefined});
      tHead = performance.now() - t0;
      benchView.status = res.status;
      benchView.headers = [...res.headers.entries()];
      const type = res.headers.get('content-type') || '';
      const rid = res.headers.get('x-request-id') || res.headers.get('request-id') || '';
      const timing = () => [`headers ${benchMs(tHead)}`, tFirst ? `first event ${benchMs(tFirst)}` : '', `total ${benchMs(performance.now() - t0)}`, `${bytes.toLocaleString()} B`, rid ? `id ${rid}` : ''].filter(Boolean).join(' · ');
      if (type.includes('text/event-stream') && res.body) {
        const acc = {reasoning: '', text: '', tools: ''};
        benchView.text = acc; benchView.kind = 'sse';
)HTML"
R"HTML(        if (benchTab === 'body') benchTab = 'text';
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '', last = 0;
        for (;;) {
          const {value, done} = await reader.read();
          if (done) break;
          bytes += value.length;
          buffer += decoder.decode(value, {stream: true}).replace(/\r\n/g, '\n');
          let cut;
          while ((cut = buffer.indexOf('\n\n')) >= 0) {
            const block = buffer.slice(0, cut); buffer = buffer.slice(cut + 2);
            let name = 'message'; const data = [];
            block.split('\n').forEach(line => {
              if (line.startsWith(':')) { name = 'comment'; data.push(line.slice(1).trim()); }
              else if (line.startsWith('event:')) name = line.slice(6).trim();
              else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
            });
            const payload = data.join('\n');
            if (!tFirst && name !== 'comment') tFirst = performance.now() - t0;
            benchView.events.push({t: Math.round(performance.now() - t0), name, data: payload});
            if (payload && payload !== '[DONE]' && name !== 'comment') { try { benchDelta(kind, name, JSON.parse(payload), acc); } catch (_) {} }
          }
          benchView.body += decoder.decode();
          benchView.timing = timing();
          if (performance.now() - last > 80) { last = performance.now(); renderBenchOut(); }
        }
        benchView.body = benchView.events.map(ev => (ev.name !== 'message' ? 'event: ' + ev.name + '\n' : '') + 'data: ' + ev.data).join('\n\n');
      } else {
        const text = await res.text();
        bytes = new TextEncoder().encode(text).length;
        benchView.body = text;
        benchView.kind = type.includes('json') ? 'json' : 'text';
        let parsed = null;
        if (benchView.kind === 'json') { try { parsed = JSON.parse(text); } catch (_) {} }
        benchView.text = benchTextOf(parsed);
        if (benchTab === 'events' || (benchTab === 'text' && !benchView.text)) benchTab = 'body';
      }
      benchView.timing = timing();
    } catch (err) {
      if (err.name === 'AbortError') benchView.timing = (benchView.timing ? benchView.timing + ' · ' : '') + 'stopped';
      else { benchView.status = 0; benchView.timing = 'The request did not complete: ' + err.message; }
    } finally {
      benchAbort = null;
      $('bench-send').hidden = false; $('bench-stop').hidden = true;
      record.view = JSON.parse(JSON.stringify(benchView));
      record.when = new Date();
      benchHistory.unshift(record);
      benchHistory.length = Math.min(benchHistory.length, 12);
      renderBenchHistory();
      renderBenchOut();
    }
  }
  $('bench-send').onclick = () => benchSend(false);
  $('bench-stop').onclick = () => { if (benchAbort) benchAbort.abort(); };
  function renderBenchHistory() {
    const list = $('bench-history');
    $('bench-history-clear').hidden = !benchHistory.length;
    if (!benchHistory.length) { list.innerHTML = '<li class="cdu-small">Requests you send appear here.</li>'; return; }
    list.replaceChildren(...benchHistory.map(r => {
      const li = document.createElement('li');
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'bench-hist';
      const st = r.view.status;
      b.innerHTML = `<span class="${st >= 200 && st < 300 ? 'green' : st == null ? 'dim' : st >= 500 || st === 0 ? 'red' : 'amber'}">${st == null ? '—' : st === 0 ? 'ERR' : st}</span><span>${esc(r.method)} ${esc(r.path.split('?')[0])}</span><span class="dim">${r.when.toLocaleTimeString()}</span>`;
      b.onclick = () => {
        const e = benchEndpoints[r.endpoint];
        benchSelect(e, r.example);
        benchPath.value = r.path;
        if (r.body) { benchBody.value = r.body; benchValidate(); }
        benchView = JSON.parse(JSON.stringify(r.view));
        renderBenchOut();
      };
      li.append(b);
      return li;
    }));
  }
  $('bench-history-clear').onclick = () => { benchHistory.length = 0; renderBenchHistory(); };
  async function benchInit() {
    if (benchReady) return;
    benchReady = true;
    benchSelect(benchEndpoints.find(e => e.path === '/v1/chat/completions'), 0);
    try {
      const res = await fetch('/api/engine/v1/models', {headers: {'X-NInfer-Supervisor': '1'}});
      const data = await res.json();
      if (res.ok && data && Array.isArray(data.data) && data.data[0]) {
        benchServedModel = data.data[0].id;
        if (benchCurrent && benchCurrent.examples.length && benchExample >= 0) benchSelect(benchCurrent, benchExample);
      }
    } catch (_) {}
    renderBenchAuth();
  }
  function renderBenchAuth() {
    const el = $('bench-auth');
    if (!cfgData) { el.textContent = 'Checking authentication…'; return; }
    el.textContent = cfgData.engine && cfgData.engine.api_key_present ? 'The supervisor adds the engine’s API key' : 'No API key configured';
  }

  // ---------------------------------------------------------------------
  // Chat: a plain conversation with the served model through the bench route,
  // with what NInfer measured about each reply: time to start, prompt reading,
  // generation rate, exact reused prefix and speculative drafts.
  // ---------------------------------------------------------------------
  const chatLog = $('chat-log'), chatInput = $('chat-input');
  let chatReady = false, chatAbort = null, chatVision = false, chatCtx = 0;
  let chat = {messages: [], think: 'default', system: '', thinkUsed: null, selectedReply: null};
  try { const saved = JSON.parse(sessionStorage.getItem('ninfer:chat') || 'null'); if (saved && Array.isArray(saved.messages)) chat = Object.assign(chat, saved, {selectedReply: null}); } catch (_) {}
  let chatPending = [];
  function chatSave() {
    // Images and token probabilities stay out of storage; the message keeps how many images
    // it carried and its confidence summary.
    const slim = Object.assign({}, chat, {selectedReply: undefined, messages: chat.messages.map(m => Object.assign({}, m, {images: undefined, tokens: undefined, tokenView: undefined, imageCount: (m.images || []).length || m.imageCount || 0}))});
    try { sessionStorage.setItem('ninfer:chat', JSON.stringify(slim)); } catch (_) {}
  }
  // <markdown>
  // Chat markdown: a small, dependency-free subset of CommonMark and GFM. Model text is never
  // passed through as HTML: every character of it is escaped, and the only tags in the output
  // are the ones generated here. Links are made only for http, https and mailto addresses;
  // images are never fetched (an image becomes a link to its address). Blocks are paragraphs,
  // ATX headings, lists (nested, ordered with a start number, task items), block quotes, fenced
  // code, thematic breaks and GFM tables. Inline: code spans, strong, emphasis, strike, links,
  // bare addresses, escapes and <br>. Parsing is one left-to-right pass per block; nesting,
  // delimiter and link scans are bounded, so no input costs more than a few times its length.
  // blocks() returns one HTML string per top-level block with the character offset where it
  // starts, and live() uses that to re-parse only the end of a growing reply: all but the last
  // two blocks are final once more text has arrived after them.
  const MD = (() => {
    const MAX_DEPTH = 8, MAX_OPEN = 32, MAX_LINK_TEXT = 1000, MAX_URL = 2000;
    const RE_FENCE = /^( {0,3})(`{3,}|~{3,})[ \t]*(.*)$/;
    const RE_ATX = /^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$/;
    const RE_HR = /^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$/;
    const RE_QUOTE = /^ {0,3}>[ \t]?(.*)$/;
    const RE_BULLET = /^( *)([-*+])([ \t]+)(\S.*)$/;
    const RE_ORDERED = /^( *)(\d{1,9})([.)])([ \t]+)(\S.*)$/;
    const RE_CELL_SEP = /^:?-+:?$/;
    const RE_PUNCT = /[\p{P}\p{S}]/u, RE_SPACE = /\s/, RE_ALNUM = /[\p{L}\p{N}]/u;
    const RE_SPECIAL = /[\\`\[<*_~!]|https?:\/\//gi;
    const RE_BR = /<br[ \t]*\/?>/iy, RE_AUTOLINK = /<((?:https?:\/\/|mailto:)[^\s<>]+)>/iy;
    const RE_URL = new RegExp('https?:\\/\\/[^\\s<>`]{1,' + MAX_URL + '}', 'iy');
    const RE_SAFE_URL = /^(?:https?:\/\/[^\s\u0000-\u001f\u007f]+|mailto:[^\s\u0000-\u001f\u007f]+)$/i;

    const isBlank = l => l.trim() === '';
    // Lines keep their own tabs (code is shown and copied as written). Structure is read from
    // norm(line), which has the leading whitespace expanded to spaces at tab stops of four.
    const indentOf = l => {
      let n = 0;
      for (let i = 0; i < l.length; i++) { const c = l.charCodeAt(i); if (c === 32) n++; else if (c === 9) n += 4 - n % 4; else break; }
      return n;
    };
    const norm = l => { const c = l.charCodeAt(0); return c === 9 || (c === 32 && /^ +\t/.test(l)) ? ' '.repeat(indentOf(l)) + l.replace(/^[ \t]+/, '') : l; };
    // Removes up to `cols` columns of leading whitespace; a tab that straddles the cut leaves its remainder as spaces.
    function stripIndent(l, cols) {
      let col = 0, i = 0;
      while (i < l.length && col < cols) {
        const c = l.charCodeAt(i);
        if (c === 32) { col++; i++; }
        else if (c === 9) { const w = 4 - col % 4; if (col + w > cols) return ' '.repeat(col + w - cols) + l.slice(i + 1); col += w; i++; }
        else break;
      }
      return l.slice(i);
    }
    const normalize = t => String(t == null ? '' : t).replace(/\r\n?/g, '\n');
    const gapOf = ws => ws.indexOf('\t') >= 0 || ws.length > 4 ? 1 : ws.length;

    function fenceOpen(line) {
      const m = RE_FENCE.exec(norm(line));
      return m && !(m[2][0] === '`' && m[3].indexOf('`') >= 0) ? m : null;
    }
    // Tracks whether a line sits inside an open fence (no lazy continuation there).
    function fenceStep(fence, line) {
      if (!fence) { const m = fenceOpen(line); return m ? {ch: m[2][0], len: m[2].length} : null; }
      const m = /^ {0,3}(`{3,}|~{3,})[ \t]*$/.exec(norm(line));
      return m && m[1][0] === fence.ch && m[1].length >= fence.len ? null : fence;
    }
    function matchItem(line) {
      line = norm(line);
      let m = RE_BULLET.exec(line);
      if (m) return {ordered: false, indent: m[1].length, mark: 1, gap: gapOf(m[3]), text: m[4], num: 1};
      m = RE_ORDERED.exec(line);
      return m ? {ordered: true, indent: m[1].length, mark: m[2].length + 1, gap: gapOf(m[4]), text: m[5], num: +m[2]} : null;
    }

    // ---- tables -------------------------------------------------------------------
    function splitRow(line) {
      let s = line.trim();
      if (s[0] === '|') s = s.slice(1);
      if (s.length && s[s.length - 1] === '|' && s[s.length - 2] !== '\\') s = s.slice(0, -1);
      const cells = [];
      let cur = '';
      for (let i = 0; i < s.length; i++) {
        const c = s[i];
        if (c === '\\' && s[i + 1] === '|') { cur += '|'; i++; }
        else if (c === '\\' && s[i + 1] === '\\') { cur += '\\\\'; i++; }
        else if (c === '|') { cells.push(cur.trim()); cur = ''; }
        else cur += c;
      }
      cells.push(cur.trim());
      return cells;
    }
    function tableAligns(line, columns) {
      if (line == null || line.indexOf('|') < 0) return null;
      const cells = splitRow(line);
      if (cells.length !== columns || !cells.every(c => RE_CELL_SEP.test(c))) return null;
      return cells.map(c => c[0] === ':' ? (c[c.length - 1] === ':' ? 'c' : 'l') : c[c.length - 1] === ':' ? 'r' : '');
    }
    function parseTable(lines, i) {
      const head = splitRow(lines[i]);
      const aligns = tableAligns(lines[i + 1], head.length);
      if (!aligns) return null;
      const cell = (tag, text, k) => `<${tag}${aligns[k] ? ` class="md-al-${aligns[k]}"` : ''}>${inline(text, false)}</${tag}>`;
      let html = '<div class="md-table"><table><thead><tr>' + head.map((c, k) => cell('th', c, k)).join('') + '</tr></thead><tbody>';
      let j = i + 2;
      for (; j < lines.length; j++) {
        const l = lines[j];
        if (isBlank(l) || l.indexOf('|') < 0 || /^ {0,3}(?:>|#{1,6}[ \t]|`{3}|~{3})/.test(l)) break;
)HTML"
R"HTML(        const row = splitRow(l);
        html += '<tr>' + head.map((_, k) => cell('td', row[k] || '', k)).join('') + '</tr>';
      }
      return [html + '</tbody></table></div>', j];
    }

    // ---- blocks -------------------------------------------------------------------
    // Would this line end the paragraph above it?
    function interrupts(line, next, depth) {
      if (fenceOpen(line)) return true;
      line = norm(line);
      if (RE_HR.test(line)) return true;
      const h = RE_ATX.exec(line);
      if (h && h[2]) return true;
      if (depth >= MAX_DEPTH) return false;
      if (RE_QUOTE.test(line)) return true;
      const it = matchItem(line);
      if (it && it.indent <= 3 && (!it.ordered || it.num === 1)) return true;
      return line.indexOf('|') >= 0 && next != null && tableAligns(next, splitRow(line).length) != null;
    }
    // A code block: a header row (language legend, Copy key) over the code. The Copy key reads the
    // text of its own <code> element, which is the block's source exactly as it is displayed.
    // `code` is returned with the HTML so a block that is still filling in can be updated in place.
    function codeBlock(info, code) {
      const lang = (info.split(/\s+/)[0] || '').replace(/^\{\.?/, '').replace(/[^\w.+#-]/g, '').slice(0, 24);
      const html = '<div class="md-code"><div class="md-code-head">' + (lang ? `<span class="md-code-lang">${esc(lang)}</span>` : '') +
        `<button type="button" class="md-copy" aria-label="Copy ${lang ? esc(lang) + ' ' : ''}code">Copy</button></div><pre><code>${esc(code)}</code></pre></div>`;
      return [html, {lang, text: code}];
    }
    function parseFence(lines, i, m) {
      const indent = m[1].length, ch = m[2][0], len = m[2].length;
      const close = new RegExp('^ {0,3}' + (ch === '`' ? '`' : '~') + '{' + len + ',}[ \\t]*$');
      const body = [];
      let j = i + 1;
      for (; j < lines.length && !close.test(norm(lines[j])); j++) body.push(indent ? stripIndent(lines[j], indent) : lines[j]);
      while (body.length && isBlank(body[body.length - 1])) body.pop();
      const block = codeBlock(m[3].trim(), body.join('\n'));
      return [block[0], j < lines.length ? j + 1 : j, block[1]];
    }
    function parseQuote(lines, i, o, depth) {
      const inner = [];
      let fence = null, j = i;
      for (; j < lines.length; j++) {
        const l = lines[j], q = RE_QUOTE.exec(norm(l));
        if (q) { inner.push(q[1]); fence = fenceStep(fence, q[1]); continue; }
        const last = inner[inner.length - 1];
        if (isBlank(l) || fence || isBlank(last) || RE_HR.test(norm(last)) || (RE_ATX.exec(norm(last)) || [])[2] || interrupts(l, lines[j + 1], depth)) break;
        inner.push(l.trim());
      }
      const kids = [];
      parseBlocks(inner, o, depth + 1, kids, null);
      return ['<blockquote>' + kids.map(b => b.html).join('') + '</blockquote>', j];
    }
    function parseList(lines, i, o, depth) {
      const first = matchItem(lines[i]), n = lines.length, items = [];
      let j = i;
      while (j < n) {
        const it = matchItem(lines[j]);
        if (!it || it.ordered !== first.ordered || it.indent >= first.indent + 2 || RE_HR.test(norm(lines[j]))) break;
        const thr = it.indent + 2, cin = it.indent + it.mark + it.gap;   // child threshold, content column
        const body = [it.text];
        let fence = fenceStep(null, it.text), k = j + 1;
        while (k < n) {
          const l = lines[k];
          if (isBlank(l)) {
            let p = k;
            while (p < n && isBlank(lines[p])) p++;
            if (p < n && indentOf(lines[p]) >= thr) { for (; k < p; k++) body.push(''); continue; }
            break;
          }
          const ind = indentOf(l);
          if (ind >= thr) { const t = stripIndent(l, Math.min(ind, cin)); body.push(t); fence = fenceStep(fence, t); k++; continue; }
          if (!fence && !isBlank(body[body.length - 1]) && !matchItem(l) && !interrupts(l, lines[k + 1], depth)) { body.push(l.trim()); k++; continue; }
          break;
        }
        const task = /^\[([ xX])\][ \t]+(?=\S)/.exec(body[0]);
        if (task) body[0] = body[0].slice(task[0].length);
        const kids = [];
        parseBlocks(body, o, depth + 1, kids, null);
        items.push('<li' + (task ? ' class="md-task-item"' : '') + '>' +
          (task ? `<span class="md-task${task[1] === ' ' ? '' : ' done'}" role="img" aria-label="${task[1] === ' ' ? 'Not done' : 'Done'}"></span>` : '') +
          kids.map(b => b.html).join('') + '</li>');
        j = k;
        let p = j;
        while (p < n && isBlank(lines[p])) p++;
        if (p < n && p > j) {
          const nx = matchItem(lines[p]);
          if (nx && nx.ordered === first.ordered && nx.indent < first.indent + 2 && !RE_HR.test(norm(lines[p]))) j = p; else break;
        }
      }
      const tag = first.ordered ? 'ol' : 'ul';
      return [`<${tag}${first.ordered && first.num !== 1 ? ` start="${first.num}"` : ''}>` + items.join('') + `</${tag}>`, j];
    }
    function parseParagraph(lines, i, o, depth) {
      const part = l => o.pre ? l : l.trim();
      const parts = [part(lines[i])];
      let j = i + 1;
      for (; j < lines.length; j++) {
        if (isBlank(lines[j]) || interrupts(lines[j], lines[j + 1], depth)) break;
        parts.push(part(lines[j]));
      }
      let html = inline(parts.join('\n'), false);
      if (!o.pre) html = html.replace(/\n/g, '<br>');
      return ['<p>' + html + '</p>', j];
    }
    // offs: character offset of each line, kept for the top level only.
    function parseBlocks(lines, o, depth, out, offs) {
      const n = lines.length;
      let i = 0;
      while (i < n) {
        const line = lines[i];
        if (isBlank(line)) { i++; continue; }
        const nl = norm(line);
        let m, r = null;
        if ((m = fenceOpen(line))) r = parseFence(lines, i, m);
        else if ((m = RE_ATX.exec(nl)) && m[2]) { const lv = m[1].length; r = [`<div class="md-h md-h${lv}" role="heading" aria-level="${Math.min(6, lv + 2)}">${inline(m[2], false)}</div>`, i + 1]; }
        else if (RE_HR.test(nl)) r = ['<hr>', i + 1];
        else if (depth < MAX_DEPTH && RE_QUOTE.test(nl)) r = parseQuote(lines, i, o, depth);
        else if (depth < MAX_DEPTH && matchItem(line)) r = parseList(lines, i, o, depth);
        else if (line.indexOf('|') >= 0) r = parseTable(lines, i);
        if (!r) r = parseParagraph(lines, i, o, depth);
        out.push({html: r[0], at: offs ? offs[i] : 0, code: r[2]});
        i = r[1];
      }
    }
    function blocks(text, o) {
      const src = normalize(text), out = [];
      if (!src.trim()) return out;
      const lines = src.split('\n'), offs = new Array(lines.length);
      let at = 0;
      for (let i = 0; i < lines.length; i++) {
        offs[i] = at;
        at += lines[i].length + 1;
      }
      parseBlocks(lines, o || {}, 0, out, offs);
      return out;
    }

    // ---- inline -------------------------------------------------------------------
    function linkHtml(url, innerHtml) {
      const web = url[0] === 'h' || url[0] === 'H';
      return `<a href="${esc(url)}"${web ? ' target="_blank"' : ''} rel="noopener noreferrer" title="${esc(url)}">${innerHtml}</a>`;
    }
    // [text](destination "title"): the link, or null when it is not a link to a safe address.
    function parseLink(src, open, ctx) {
      const limit = Math.min(src.length, open + 1 + MAX_LINK_TEXT);
      let depth = 0, p = open + 1;
      for (; p < limit; p++) {
        const c = src.charCodeAt(p);
        if (c === 92) { p++; continue; }
        if (c === 96) {
          let k = 1;
          while (src.charCodeAt(p + k) === 96) k++;
          const cl = ctx.tick(k, p + k);
          p = (cl >= 0 && cl < limit ? cl + k : p + k) - 1;
          continue;
        }
        if (c === 91) depth++;
        else if (c === 93) { if (depth === 0) break; depth--; }
      }
      ctx.budget -= p - open;
      if (p >= limit || ctx.budget < 0 || src[p] !== ']' || src[p + 1] !== '(') return null;
      let q = p + 2;
      while (q < src.length && /[ \t\n]/.test(src[q])) q++;
      let url;
      if (src[q] === '<') {
        const e = src.indexOf('>', q);
        if (e < 0 || e - q > MAX_URL) return null;
        url = src.slice(q + 1, e); q = e + 1;
      } else {
        const s = q;
        let d = 0;
        for (; q < src.length && q - s <= MAX_URL; q++) {
          const c = src[q];
          if (c === '\\' && q + 1 < src.length) { q++; continue; }
          if (c === '(') d++;
          else if (c === ')') { if (d === 0) break; d--; }
          else if (/\s/.test(c)) break;
        }
        url = src.slice(s, q);
      }
      url = url.replace(/\\([!-\/:-@\[-`{-~])/g, '$1');
      while (q < src.length && /[ \t\n]/.test(src[q])) q++;
      const quote = src[q];
      if (quote === '"' || quote === "'" || quote === '(') {
        const end = quote === '(' ? ')' : quote;
        let e = q + 1;
        while (e < src.length && e - q < 300 && src[e] !== end) e += src[e] === '\\' ? 2 : 1;
        if (src[e] !== end) return null;
        q = e + 1;
        while (q < src.length && /[ \t\n]/.test(src[q])) q++;
      }
      if (src[q] !== ')' || !RE_SAFE_URL.test(url)) return null;
      return {text: src.slice(open + 1, p), url, end: q + 1};
    }
    // Delimiter runs of * _ ~ that can open or close emphasis: the CommonMark flanking rules,
    // plus two chat-friendly exceptions so arithmetic (2*3*4) and file globs (src/*.js) stay literal.
    function runFlags(src, i, k, ch) {
      const prev = i > 0 ? src[i - 1] : ' ', next = i + k < src.length ? src[i + k] : ' ';
      const pw = RE_SPACE.test(prev), nw = RE_SPACE.test(next), pp = RE_PUNCT.test(prev), np = RE_PUNCT.test(next);
      const left = !nw && (!np || pw || pp), right = !pw && (!pp || nw || np);
      let open = left, close = right;
      if (ch === '_') { open = left && (!right || pp); close = right && (!left || np); }
      if (ch === '*' && k === 1) {
        if (RE_ALNUM.test(prev) && RE_ALNUM.test(next)) open = close = false;
        else if (!pw && /^[.\/\\]\w/.test(src.slice(i + 1, i + 3))) open = close = false;
      }
      return {open, close};
    }
    // Pairs delimiter runs into <em>, <strong> and <del> (CommonMark's process-emphasis, with the
    // search for an opener capped so a long run of unmatched markers stays linear).
    function emphasis(nodes) {
      const out = [], stack = [];
      for (const nd of nodes) {
        if (nd.t !== 'd') { out.push(nd); continue; }
        let n = nd.n;
        if (nd.close) {
          while (n > 0) {
            let f = -1;
            for (let s = stack.length - 1, steps = 0; s >= 0 && steps < 50; s--, steps++) {
              const o = stack[s];
              if (o.ch !== nd.ch || !o.count) continue;
              if ((o.close || nd.open) && (o.orig + nd.n) % 3 === 0 && !(o.orig % 3 === 0 && nd.n % 3 === 0)) continue;
              f = s; break;
            }
            if (f < 0) break;
            const o = stack[f];
            const use = nd.ch === '~' ? 2 : (o.count >= 2 && n >= 2 ? 2 : 1);
            const children = out.splice(o.pos + 1);
            o.count -= use; n -= use;
            out[o.pos].s = o.ch.repeat(o.count);
            out.push({t: 'e', tag: nd.ch === '~' ? 'del' : use === 2 ? 'strong' : 'em', c: children});
            stack.length = f + 1;
            if (!o.count) stack.pop();
          }
        }
        if (n > 0) {
          out.push({t: 't', s: nd.ch.repeat(n)});
          if (nd.open && stack.length < MAX_OPEN) stack.push({ch: nd.ch, count: n, orig: nd.n, close: nd.close, pos: out.length - 1});
        }
      }
      return out;
    }
    const nodesHtml = list => {
      let s = '';
      for (const nd of list) s += nd.t === 't' ? esc(nd.s) : nd.t === 'h' ? nd.s : '<' + nd.tag + '>' + nodesHtml(nd.c) + '</' + nd.tag + '>';
      return s;
    };
)HTML"
R"HTML(    // A bare address ends before trailing punctuation and an unbalanced closing bracket.
    function trimUrl(u) {
      for (;;) {
        const last = u[u.length - 1];
        if (/[.,;:!?*_~'"’”]/.test(last)) u = u.slice(0, -1);
        else if (last === ')' || last === ']') {
          const open = last === ')' ? '(' : '[';
          if (u.split(last).length > u.split(open).length) u = u.slice(0, -1); else return u;
        } else return u;
      }
    }
    function inline(src, nolink) {
      const nodes = [], runs = {};
      let buf = '', i = 0;
      const flush = () => { if (buf) { nodes.push({t: 't', s: buf}); buf = ''; } };
      if (src.indexOf('`') >= 0) {
        const re = /`+/g;
        let m;
        while ((m = re.exec(src))) (runs[m[0].length] || (runs[m[0].length] = [])).push(m.index);
      }
      // First backtick run of exactly this length that starts at or after `from`.
      const tick = (k, from) => {
        const a = runs[k];
        if (!a) return -1;
        let lo = 0, hi = a.length;
        while (lo < hi) { const mid = (lo + hi) >> 1; if (a[mid] < from) lo = mid + 1; else hi = mid; }
        return lo < a.length ? a[lo] : -1;
      };
      const ctx = {tick, budget: 4096 + 8 * src.length};
      while (i < src.length) {
        RE_SPECIAL.lastIndex = i;
        const m = RE_SPECIAL.exec(src);
        if (!m) { buf += src.slice(i); break; }
        if (m.index > i) buf += src.slice(i, m.index);
        i = m.index;
        const c = src[i];
        if (c === '\\') {
          const nx = src[i + 1];
          // Brackets and braces keep their backslash: \( x \) and \[ x \] are LaTeX, shown as typed.
          if (nx !== undefined && /[!-\/:-@\\^-`|~]/.test(nx) && !/[()\[\]{}]/.test(nx)) { buf += nx; i += 2; }
          else { if (nx !== '\n') buf += '\\'; i += 1; }
        } else if (c === '`') {
          let k = 1;
          while (src[i + k] === '`') k++;
          const close = tick(k, i + k);
          if (close >= 0) {
            let code = src.slice(i + k, close).replace(/\n/g, ' ');
            if (code.length > 2 && code[0] === ' ' && code[code.length - 1] === ' ' && code.trim()) code = code.slice(1, -1);
            flush(); nodes.push({t: 'h', s: '<code>' + esc(code) + '</code>'});
            i = close + k;
          } else { buf += '`'.repeat(k); i += k; }
        } else if (c === '[' || (c === '!' && src[i + 1] === '[')) {
          const image = c === '!', open = image ? i + 1 : i;
          const lk = nolink || ctx.budget < 0 ? null : parseLink(src, open, ctx);
          if (lk) {
            flush();
            nodes.push({t: 'h', s: linkHtml(lk.url, image ? esc('Image: ' + (lk.text.trim() || lk.url)) : inline(lk.text, true))});
            i = lk.end;
          } else { buf += c; i += 1; }
        } else if (c === '<') {
          RE_BR.lastIndex = i; RE_AUTOLINK.lastIndex = i;
          let a;
          if (RE_BR.exec(src)) { flush(); nodes.push({t: 'h', s: '<br>'}); i = RE_BR.lastIndex; }
          else if (!nolink && (a = RE_AUTOLINK.exec(src))) { flush(); nodes.push({t: 'h', s: linkHtml(a[1], esc(a[1]))}); i = RE_AUTOLINK.lastIndex; }
          else { buf += '<'; i += 1; }
        } else if (c === '*' || c === '_' || c === '~') {
          let k = 1;
          while (src[i + k] === c) k++;
          const f = c === '~' && k !== 2 ? {open: false, close: false} : runFlags(src, i, k, c);
          if (f.open || f.close) { flush(); nodes.push({t: 'd', ch: c, n: k, open: f.open, close: f.close}); }
          else buf += src.slice(i, i + k);
          i += k;
        } else if (c === '!') { buf += '!'; i += 1; }
        else {   // http:// or https://
          let u = null;
          if (!nolink && !(i > 0 && RE_ALNUM.test(src[i - 1]))) {
            RE_URL.lastIndex = i;
            const mm = RE_URL.exec(src);
            if (mm) { u = trimUrl(mm[0]); if (!/^https?:\/\/[^:\/]/i.test(u)) u = null; }
          }
          if (u) { flush(); nodes.push({t: 'h', s: linkHtml(u, esc(u))}); i += u.length; }
          else { buf += m[0]; i += m[0].length; }
        }
      }
      flush();
      return nodesHtml(emphasis(nodes));
    }

    // ---- public -------------------------------------------------------------------
    function render(text, o) { return blocks(text, o).map(b => b.html).join(''); }
    // A growing reply inside `container`: only the blocks at its end are parsed and replaced.
    function live(container, o) {
      let text = '', list = [], nodes = [];
      return {
        update(raw) {
          const t = normalize(raw);
          if (t === text) return;
          const keep = t.startsWith(text) ? Math.max(0, list.length - 2) : 0;
          const off = keep ? list[keep].at : 0;
          const fresh = blocks(off ? t.slice(off) : t, o).map(b => ({html: b.html, at: b.at + off, code: b.code}));
          const next = list.slice(0, keep).concat(fresh);
          let i = keep;
          while (i < next.length && i < list.length) {
            const a = list[i], b = next[i];
            if (a.html !== b.html) {
              // A code block that is still filling in keeps its node, so its Copy key keeps its
              // feedback and the block keeps its scroll position.
              if (!(a.code && b.code && a.code.lang === b.code.lang && nodes[i])) break;
              nodes[i].querySelector('code').textContent = b.code.text;
            }
            i++;
          }
          for (let k = nodes.length - 1; k >= i; k--) nodes[k].remove();
          nodes.length = Math.min(nodes.length, i);
          if (i < next.length) {
            const tpl = document.createElement('template');
            tpl.innerHTML = next.slice(i).map(b => b.html).join('');
            const added = [...tpl.content.children];
            if (added.length === next.length - i) { container.append(...added); nodes.push(...added); }
            else { container.innerHTML = next.map(b => b.html).join(''); nodes = [...container.children]; }
          }
          list = next; text = t;
        }
      };
    }
    return {blocks, render, live};
  })();
  function chatMarkdown(text, opts) { return MD.render(text, opts); }
  // </markdown>
  // The Copy key of a code block. One listener on the log serves every block, so it keeps
  // working when a block is re-rendered while the reply streams. It copies the text of the
  // block's own <code> element, which is the code exactly as displayed (never HTML).
  async function writeClipboard(text) {
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) { await navigator.clipboard.writeText(text); return true; }
    } catch (_) { /* denied or unfocused: try the selection route */ }
    // A page served over plain http from another address has no async clipboard.
    const area = document.createElement('textarea'), active = document.activeElement;
    area.value = text;
    area.setAttribute('readonly', '');
    area.setAttribute('aria-hidden', 'true');
    area.style.cssText = 'position:fixed;top:0;left:0;width:1px;height:1px;opacity:0;pointer-events:none';
    document.body.append(area);
    area.select();
    area.setSelectionRange(0, text.length);
    let ok = false;
    try { ok = document.execCommand('copy'); } catch (_) {}
    area.remove();
    if (active && active.focus) active.focus();
    return ok;
  }
  chatLog.addEventListener('click', async e => {
    const key = e.target.closest && e.target.closest('.md-copy');
    const block = key && key.closest('.md-code');
    if (!key || !block || !chatLog.contains(key)) return;
    const ok = await writeClipboard(block.querySelector('code').textContent);
    clearTimeout(key._reset);
    key.textContent = ok ? 'Copied' : 'Failed';
    key.classList.toggle('failed', !ok);
    key._reset = setTimeout(() => { key.textContent = 'Copy'; key.classList.remove('failed'); }, 1500);
  });
  function chatTime(d) { return new Date(d).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', second: '2-digit'}); }
  function chatPct(a, b) { return b > 0 ? Math.round(100 * a / b) + '%' : '—'; }
  function chatVitals(st, m) {
    if (!st) return '';
    const items = [];
    const add = (val, unit, title, cls) => items.push(`<span${title ? ` title="${esc(title)}"` : ''}><span class="vital-v${cls ? ' ' + cls : ''}">${esc(val)}</span>${unit ? ' ' + esc(unit) : ''}</span>`);
    const t = st.timings || {}, u = st.usage || {};
    if (Number.isFinite(t.predicted_per_second) && t.predicted_n > 1) add(rate(t.predicted_per_second), 'tok/s', 'Generation rate', 'green');
    if (Number.isFinite(st.ttft)) {
      const pRead = Number.isFinite(t.prompt_ms) ? ` · engine prompt read ${duration(t.prompt_ms)}` : '';
      add(duration(st.ttft), st.whole ? 'resp' : 'TTFT', (st.whole ? 'Browser wall time to complete answer' : 'Browser wall time to first token') + pRead);
    }
    const out = u.completion_tokens != null ? u.completion_tokens : t.predicted_n;
    if (out != null) {
      const reasoning = (u.completion_tokens_details || {}).reasoning_tokens;
      add(Number(out).toLocaleString(), reasoning ? `tok (${reasoning.toLocaleString()} think)` : 'tok', 'Output tokens');
    }
    const prompt = u.prompt_tokens != null ? u.prompt_tokens : (t.cache_n || 0) + (t.prompt_n || 0);
    if (prompt) add(chatPct(t.cache_n || 0, prompt), 'cache', `${(t.cache_n || 0).toLocaleString()} of ${prompt.toLocaleString()} prompt tokens reused from prefix cache`);
    if (t.draft_n > 0) add(chatPct(t.draft_n_accepted || 0, t.draft_n), 'drafts', `${(t.draft_n_accepted || 0).toLocaleString()} of ${t.draft_n.toLocaleString()} speculative drafts accepted`);
    if (st.confidence) add((100 * st.confidence.mean).toFixed(1) + '%', 'conf', `Mean token probability · ${st.confidence.unsure.toLocaleString()} of ${st.confidence.n.toLocaleString()} tokens under 50%`);
    if (st.finish === 'length') {
      const sent = new Map((m && m.sent) || []);
      const limit = sent.has('max_tokens') ? sent.get('max_tokens') : sent.get('max_completion_tokens');
      add(limit === -1 ? 'context full' : 'budget reached', '', limit === -1 ? 'With no limit an answer ends when the conversation fills the context' : 'Output budget reached · open Details or raise the budget under Parameters', 'amber');
    } else if (st.finish && st.finish !== 'stop') add(st.finish, '', 'Finish reason', 'amber');
    return items.length ? `<div class="msg-vitals">${items.join('')}</div>` : '';
  }
  function chatTelemetry(st, m) {
    if (!st) return '';
    const items = [];
    const add = (label, value, title, cls) => items.push(`<div${title ? ` title="${esc(title)}"` : ''}><dt>${esc(label)}</dt><dd${cls ? ` class="${cls}"` : ''}>${esc(value)}</dd></div>`);
    if (Number.isFinite(st.ttft)) {
      if (st.whole) add('Response', duration(st.ttft), 'Measured in the browser from sending to the complete answer, including queue time');
      else add('First token', duration(st.ttft), 'Measured in the browser from sending to the first visible token, including queue time');
    }
    const t = st.timings || {}, u = st.usage || {};
    if (Number.isFinite(t.prompt_ms)) add('Prompt read', duration(t.prompt_ms), 'Engine time from admission to the first output token');
    if (Number.isFinite(t.predicted_per_second) && t.predicted_n > 1) add('Generating', rate(t.predicted_per_second) + ' tok/s');
    const out = u.completion_tokens != null ? u.completion_tokens : t.predicted_n;
    if (out != null) {
      const reasoning = (u.completion_tokens_details || {}).reasoning_tokens;
      add('Output', `${Number(out).toLocaleString()} tokens${reasoning ? ` · ${reasoning.toLocaleString()} reasoning` : ''}`);
    }
    const prompt = u.prompt_tokens != null ? u.prompt_tokens : (t.cache_n || 0) + (t.prompt_n || 0);
    if (prompt) add('Prompt', `${prompt.toLocaleString()} · ${(t.cache_n || 0).toLocaleString()} reused (${chatPct(t.cache_n || 0, prompt)})`, 'Reused is the exact prefix the engine kept from earlier turns instead of reading it again');
    if (t.draft_n > 0) add('Drafts', `${(t.draft_n_accepted || 0).toLocaleString()} of ${t.draft_n.toLocaleString()} accepted (${chatPct(t.draft_n_accepted || 0, t.draft_n)})`, 'Speculative decoding: drafted tokens the model accepted');
    if (st.finish === 'length') {
      const sent = new Map((m && m.sent) || []);
      const limit = sent.has('max_tokens') ? sent.get('max_tokens') : sent.get('max_completion_tokens');
      add('Finish', limit === -1 ? 'context full' : limit != null ? `output budget of ${Number(limit).toLocaleString()} reached` : 'engine default budget reached',
          limit === -1 ? 'With no limit an answer ends when the conversation fills the context' : 'Raise the budget under Parameters, or choose No limit', 'amber');
    } else if (st.finish) add('Finish', st.finish === 'stop' ? 'complete' : st.finish);
    if (st.confidence) add('Confidence', `mean ${(100 * st.confidence.mean).toFixed(1)}% · ${st.confidence.unsure.toLocaleString()} of ${st.confidence.n.toLocaleString()} under 50%`, 'Mean probability the model gave each token it produced, read before any sampling adjustment');
    return items.length ? `<dl class="msg-telemetry">${items.join('')}</dl>` : '';
  }
  // Sent: the request's own parameters, so a test always shows what produced it.
  function chatSentLine(m) {
    if (!m.sent) return '';
    return `<p class="msg-sent"><span class="cdu-label">Sent</span> ${m.sent.length ? esc(chatParamText(m.sent)) : 'engine defaults only'}</p>`;
  }
  // Token view: each generated token with its probability; unsure tokens are marked and the
  // alternatives are shown in the instant token readout bar and on the token's tooltip.
  const CHAT_SPECIAL_TOKENS = new Set(['<|im_end|>', '<|endoftext|>', '<|im_start|>', '<think>', '</think>']);
  const chatProb = lp => (Number.isFinite(lp) && lp > -9999 ? Math.exp(lp) : 0);
  function chatConfidence(tokens) {
    const real = tokens.filter(t => !t.forced && !CHAT_SPECIAL_TOKENS.has(t.token));
    if (!real.length) return null;
    const ps = real.map(t => chatProb(t.logprob));
    return {n: ps.length, mean: ps.reduce((a, b) => a + b, 0) / ps.length, unsure: ps.filter(p => p < .5).length};
  }
  function chatSplitTokens(tokens) {
    if (!Array.isArray(tokens) || !tokens.length) return {think: [], answer: []};
    const cut = tokens.findIndex(t => t && (t.token === '</think>' || (typeof t.token === 'string' && t.token.includes('</think>'))));
    return cut >= 0 ? {think: tokens.slice(0, cut + 1), answer: tokens.slice(cut + 1)} : {think: [], answer: tokens};
  }
  function chatTokenView(tokens, offset = 0) {
    return tokens.map((t, idx) => {
      const p = chatProb(t.logprob);
      const special = CHAT_SPECIAL_TOKENS.has(t.token);
      const band = t.forced ? 'tok-forced' : special ? 'tok-special' : p < .2 ? 'tok-low' : p < .5 ? 'tok-mid' : '';
      const alts = (t.top_logprobs || []).map(a => `${JSON.stringify(a.token)} ${(100 * chatProb(a.logprob)).toFixed(1)}%`).join('\n');
)HTML"
R"HTML(      const tip = t.forced ? 'Inserted by the thinking budget' : special ? `Special token · ${(100 * p).toFixed(1)}%` : `${(100 * p).toFixed(1)}%${alts ? '\n' + alts : ''}`;
      return `<span class="tok ${band}" data-tok="${offset + idx}" tabindex="0" title="${esc(tip)}">${esc(t.token)}</span>`;
    }).join('');
  }
  function chatTokenReadoutHtml(t) {
    if (!t) return '<span class="dim">Hover or focus any token to inspect its probability and top alternatives</span>';
    const p = chatProb(t.logprob);
    const special = CHAT_SPECIAL_TOKENS.has(t.token);
    const pCls = t.forced ? 'magenta' : special ? 'dim' : p < .2 ? 'red' : p < .5 ? 'amber' : 'green';
    const pText = t.forced ? 'forced by budget' : `${(100 * p).toFixed(1)}%`;
    const alts = (t.top_logprobs || []).slice(0, 5).map(a => `<span><code>${esc(JSON.stringify(a.token))}</code> <span class="vital-v">${(100 * chatProb(a.logprob)).toFixed(1)}%</span></span>`).join(' · ');
    return `<span class="cdu-label">${special ? 'Special' : 'Token'}</span> <code>${esc(JSON.stringify(t.token))}</code> <span class="vital-v ${pCls}">${esc(pText)}</span>${alts ? ` <span class="cdu-label">Alts</span> ${alts}` : ''}`;
  }
  const CHAT_STATES = {waiting: 'Waiting for the engine…', reading: 'Reading the prompt…', thinking: 'Thinking…', generating: 'Writing…', whole: 'Writing the whole answer…', stopped: 'Stopped', error: 'Failed'};
  const CHAT_LIVE = new Set(['waiting', 'reading', 'thinking', 'generating', 'whole']);
  function chatHead(m) {
    const showModel = m.model && (!benchServedModel || m.model !== benchServedModel);
    const who = m.role === 'user' ? '<span class="msg-dir" aria-hidden="true">▲</span><span class="msg-who">You</span>'
      : `<span class="msg-dir" aria-hidden="true">▼</span><span class="msg-who green">NInfer</span>${showModel ? `<span class="msg-model-name">${esc(m.model)}</span>` : ''}`;
    return `<header class="msg-head">${who}<time>${esc(chatTime(m.time))}</time>`;
  }
  // The live skeleton of a reply being written: its text grows by appended deltas, so a long
  // answer costs only its new text per paint, stays selectable, and the reasoning box keeps
  // its scroll position.
  function chatLiveMessage(m, el) {
    el.innerHTML = `${chatHead(m)}<span class="msg-state"></span><span class="msg-live"></span></header><div class="msg-progress" hidden><span class="msg-bar"><span></span></span><span></span></div><details class="msg-reasoning" hidden${m.reasoningOpen ? ' open' : ''}><summary>Reasoning</summary><div class="msg-reasoning-body md"></div></details><div class="msg-body md"></div>`;
    const det = el.querySelector('.msg-reasoning');
    det.addEventListener('toggle', () => { m.reasoningOpen = det.open; });
    const prog = el.querySelector('.msg-progress'), rbody = det.lastChild;
    el._live = {state: el.querySelector('.msg-state'), rate: el.querySelector('.msg-live'), prog, bar: prog.firstChild.firstChild, ptext: prog.lastChild, det, rbody,
      rview: MD.live(rbody), cview: MD.live(el.querySelector('.msg-body')), rLen: 0, cLen: 0};
    chatLiveUpdate(m, el);
    return el;
  }
  function chatActiveReplyIndex() {
    if (Number.isInteger(chat.selectedReply) && chat.messages[chat.selectedReply] && chat.messages[chat.selectedReply].role === 'assistant') {
      return chat.selectedReply;
    }
    for (let i = chat.messages.length - 1; i >= 0; i--) {
      if (chat.messages[i] && chat.messages[i].role === 'assistant') return i;
    }
    return -1;
  }
  function chatRenderInspector() {
    const metaEl = $('chat-inspect-meta'), bodyEl = $('chat-inspect-body');
    if (!metaEl || !bodyEl) return;
    const replies = [];
    chat.messages.forEach((m, idx) => { if (m && m.role === 'assistant') replies.push({m, idx}); });
    const activeIdx = chatActiveReplyIndex();
    chatLog.querySelectorAll('.msg-model').forEach(node => {
      const idx = Number(node.dataset.index);
      const btn = node.querySelector('[data-act="details"]');
      if (btn) btn.setAttribute('aria-pressed', String(idx === activeIdx));
    });
    if (!replies.length || activeIdx < 0) {
      metaEl.textContent = 'No replies yet';
      bodyEl.innerHTML = '<p class="cdu-small">Send a message to inspect its timings, cache reuse, speculative drafts, and sent parameters.</p>';
      return;
    }
    const pos = replies.findIndex(r => r.idx === activeIdx) + 1;
    const isLatest = activeIdx === replies[replies.length - 1].idx;
    const m = chat.messages[activeIdx], st = m.stats || {};
    if (m.tokens && !st.confidence) st.confidence = chatConfidence(m.tokens);
    metaEl.textContent = `${isLatest ? 'Latest · ' : ''}Turn ${pos} of ${replies.length} · ${chatTime(m.time)}`;
    const parts = [];
    if (CHAT_LIVE.has(m.state)) {
      const liveRate = m.state === 'generating' && Number.isFinite(st.liveRate) ? ` · <span class="vital-v green">${esc(rate(st.liveRate))}</span> tok/s` : '';
      parts.push(`<p class="cdu-small"><span class="green">${esc(CHAT_STATES[m.state] || 'Working…')}</span>${liveRate}</p>`);
    } else if (m.error) {
      parts.push(`<p class="cdu-small red">${esc(m.error)}</p>`);
    }
    const tel = chatTelemetry(st, m);
    if (tel) parts.push(tel);
    const sent = chatSentLine(m);
    if (sent) parts.push(sent);
    bodyEl.innerHTML = parts.join('') || '<p class="cdu-small">Waiting for reply metrics…</p>';
  }
  function chatLiveUpdate(m, el) {
    const v = el._live, st = m.stats || {};
    v.state.textContent = CHAT_STATES[m.state] || '';
    v.rate.textContent = m.state === 'generating' && Number.isFinite(st.liveRate) ? rate(st.liveRate) + ' tok/s' : '';
    const pr = m.state === 'reading' ? st.progress : null;
    v.prog.hidden = !pr;
    if (pr) {
      v.bar.style.width = Math.min(100, 100 * (pr.processed || 0) / Math.max(1, pr.total || 1)) + '%';
      v.ptext.textContent = `Reading prompt ${(pr.processed || 0).toLocaleString()} / ${(pr.total || 0).toLocaleString()}${pr.cache ? ` · ${pr.cache.toLocaleString()} reused` : ''}`;
    }
    if (m.reasoning.length > v.rLen) {
      v.det.hidden = false;
      const pinned = v.rbody.scrollTop + v.rbody.clientHeight >= v.rbody.scrollHeight - 12;
      v.rview.update(m.reasoning); v.rLen = m.reasoning.length;
      if (pinned) v.rbody.scrollTop = v.rbody.scrollHeight;
    }
    if (m.content.length > v.cLen) { v.cview.update(m.content); v.cLen = m.content.length; }
    chatRenderInspector();
  }
  function chatRenderMessage(m, i) {
    const el = document.createElement('article');
    el.className = 'msg ' + (m.role === 'user' ? 'msg-user' : 'msg-model') + (m.error ? ' msg-error' : '');
    el.dataset.index = i;
    if (m.role === 'user') {
      const imgs = (m.images || []).map(src => `<img src="${esc(src)}" alt="Attached image">`).join('');
      const lost = !m.images && m.imageCount ? `<span class="cdu-small">${m.imageCount} image${m.imageCount === 1 ? '' : 's'} attached (not kept after reload)</span>` : '';
      const copyBtn = m.content ? '<span class="msg-actions"><button type="button" class="client-evidence" data-act="copy">Copy</button></span>' : '';
      el.innerHTML = `${chatHead(m)}${copyBtn}</header>${imgs || lost ? `<div class="msg-images">${imgs}${lost}</div>` : ''}<div class="msg-body md md-user">${chatMarkdown(m.content, {pre: true})}</div>`;
      const b = el.querySelector('[data-act="copy"]');
      if (b) b.onclick = () => copyText(b, m.content || '', b.firstChild);
      return el;
    }
    if (CHAT_LIVE.has(m.state)) return chatLiveMessage(m, el);
    const st = m.stats || {};
    if (m.tokens && !st.confidence) st.confidence = chatConfidence(m.tokens);
    const state = CHAT_STATES[m.state] || '';
    const split = m.tokenView && m.tokens ? chatSplitTokens(m.tokens) : null;
    const reasoningBody = split && split.think.length ? `<div class="msg-tokens">${chatTokenView(split.think, 0)}</div>` : chatMarkdown(m.reasoning);
    const reasoning = m.reasoning || (split && split.think.length) ? `<details class="msg-reasoning"${m.reasoningOpen ? ' open' : ''}><summary>Reasoning${st.usage && st.usage.completion_tokens_details && st.usage.completion_tokens_details.reasoning_tokens ? ' · ' + st.usage.completion_tokens_details.reasoning_tokens.toLocaleString() + ' tokens' : ''}</summary><div class="msg-reasoning-body md">${reasoningBody}</div></details>` : '';
    const ended = m.state === 'done' || m.state === 'stopped' || m.state === 'error';
    const tokenKey = m.tokens && m.tokens.length ? `<button type="button" class="client-evidence" data-act="tokens" aria-pressed="${m.tokenView ? 'true' : 'false'}">${m.tokenView ? 'Text view' : 'Token view'}</button>` : '';
    const actions = ended ? `<div class="msg-actions"><button type="button" class="client-evidence" data-act="copy">Copy</button>${tokenKey}${i === chat.messages.length - 1 ? '<button type="button" class="client-evidence" data-act="regen">Regenerate</button>' : ''}</div>` : '';
    const answerTokens = split ? (split.answer.length ? split.answer : m.tokens) : null;
    const answerOffset = split && split.answer.length ? split.think.length : 0;
    const body = m.error ? `<span class="red">${esc(m.error)}</span>`
      : m.tokenView && m.tokens ? `<div class="msg-tokens">${chatTokenView(answerTokens, answerOffset)}</div><div class="msg-token-bar"><div class="msg-token-readout">${chatTokenReadoutHtml(null)}</div><span class="msg-token-key"><span class="tok tok-mid">under 50%</span> <span class="tok tok-low">under 20%</span></span></div>`
      : chatMarkdown(m.content);
    const vitals = m.state === 'done' || m.state === 'stopped' ? chatVitals(st, m) : '';
    const telemetry = m.state === 'done' || m.state === 'stopped' ? chatTelemetry(st, m) : '';
    const sentLine = ended ? chatSentLine(m) : '';
    const hasDetails = Boolean(telemetry || sentLine);
    const sentCount = m.sent ? m.sent.length : 0;
    const isInspected = i === chatActiveReplyIndex();
    const detailsBtn = hasDetails ? `<button type="button" class="client-evidence msg-more" data-act="details" aria-pressed="${isInspected ? 'true' : 'false'}" aria-expanded="${m.detailsOpen ? 'true' : 'false'}" title="Inspect full telemetry and sent parameters">Inspect${sentCount ? `<span class="msg-sent-tag"> · ${sentCount} sent</span>` : ''}</button>` : '';
    const foot = ended ? `<div class="msg-foot">${vitals}${detailsBtn}${actions}</div>${hasDetails ? `<div class="msg-details"${m.detailsOpen ? '' : ' hidden'}>${telemetry}${sentLine}</div>` : ''}` : '';
    el.innerHTML = `${chatHead(m)}${state ? `<span class="msg-state${m.state === 'error' ? ' red' : m.state === 'stopped' ? ' amber' : ''}">${esc(state)}</span>` : ''}</header>${reasoning}<div class="msg-body md">${body}</div>${m.note ? `<p class="msg-note">${esc(m.note)}</p>` : ''}${foot}`;
    const det = el.querySelector('.msg-reasoning');
    if (det) det.addEventListener('toggle', () => { m.reasoningOpen = det.open; });
    const readout = el.querySelector('.msg-token-readout');
    if (readout && m.tokens) {
      const showTok = e => {
        const tokEl = e.target.closest && e.target.closest('.tok[data-tok]');
        if (!tokEl || !el.contains(tokEl)) return;
        readout.innerHTML = chatTokenReadoutHtml(m.tokens[Number(tokEl.dataset.tok)]);
      };
      el.addEventListener('mouseover', showTok);
      el.addEventListener('focusin', showTok);
    }
    el.addEventListener('click', e => {
      if (e.target.closest && e.target.closest('button, a, summary, .tok')) return;
      if (chat.selectedReply !== i) { chat.selectedReply = i; chatRenderInspector(); }
    });
    el.querySelectorAll('[data-act]').forEach(b => {
      b.onclick = () => {
        if (b.dataset.act === 'copy') copyText(b, m.content || '', b.firstChild);
        else if (b.dataset.act === 'tokens') { m.tokenView = !m.tokenView; chatRender(i); }
        else if (b.dataset.act === 'details') {
          chat.selectedReply = i;
          m.detailsOpen = !m.detailsOpen;
          const d = el.querySelector('.msg-details');
          if (d) d.hidden = !m.detailsOpen;
          b.setAttribute('aria-expanded', String(m.detailsOpen));
          chatRenderInspector();
        }
        else if (b.dataset.act === 'regen') chatRegenerate();
      };
    });
    return el;
  }
  function chatRender(only) {
    $('chat-empty').hidden = chat.messages.length > 0;
    const stick = chatLog.scrollTop + chatLog.clientHeight >= chatLog.scrollHeight - 24;
    if (only != null) {
      const old = chatLog.querySelector(`.msg[data-index="${only}"]`);
      const next = chatRenderMessage(chat.messages[only], only);
      if (old) old.replaceWith(next); else chatLog.append(next);
    } else {
      chatLog.querySelectorAll('.msg').forEach(n => n.remove());
      chat.messages.forEach((m, i) => chatLog.append(chatRenderMessage(m, i)));
    }
    if (stick || only == null) chatLog.scrollTop = chatLog.scrollHeight;
    const last = [...chat.messages].reverse().find(m => m.stats && m.stats.usage);
    const used = last ? (last.stats.usage.prompt_tokens || 0) + (last.stats.usage.completion_tokens || 0) : 0;
    $('chat-context-text').textContent = chatCtx ? `${used.toLocaleString()} / ${chatCtx.toLocaleString()}` : used ? used.toLocaleString() : '—';
    $('chat-meter-fill').style.width = chatCtx ? Math.min(100, 100 * used / chatCtx) + '%' : '0';
    $('chat-meter-fill').className = chatCtx && used / chatCtx > .9 ? 'amber-fill' : '';
    $('chat-think-note').hidden = !(chat.thinkUsed != null && chat.thinkUsed !== chat.think && chat.messages.length);
    document.querySelectorAll('[data-think]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.think === chat.think)));
    chatRenderInspector();
  }
  // Parameters: an empty field is omitted from the request, so the engine's own default
  // applies. They persist in this browser; the conversation stays in the tab.
  const PARAMS_KEY = 'ninfer:chat-params';
  const chatFields = () => [...document.querySelectorAll('#chat-settings [data-param]')];
  function chatSaveParams() {
    const v = {budget: $('chat-budget').value, max: $('chat-max').value, stream: $('chat-stream').checked, stops: $('chat-stops').value,
      format: $('chat-format').value, schema: $('chat-schema').value, logprobs: $('chat-logprobs').checked, top: $('chat-top-logprobs').value, extra: $('chat-extra').value};
    chatFields().forEach(el => { v['p:' + el.dataset.param] = el.type === 'checkbox' ? el.checked : el.value; });
    try { localStorage.setItem(PARAMS_KEY, JSON.stringify(v)); } catch (_) {}
  }
  function chatLoadParams() {
    let v = {};
    try { v = JSON.parse(localStorage.getItem(PARAMS_KEY) || '{}') || {}; } catch (_) {}
    $('chat-budget').value = ['default', 'custom', 'unlimited'].includes(v.budget) ? v.budget : 'unlimited';
    $('chat-max').value = v.max || '';
    $('chat-stream').checked = v.stream !== false;
    $('chat-stops').value = v.stops || '';
    $('chat-format').value = ['', 'json_object', 'json_schema'].includes(v.format) ? v.format : '';
    $('chat-schema').value = v.schema || '';
    $('chat-logprobs').checked = !!v.logprobs;
    $('chat-top-logprobs').value = v.top || '';
    $('chat-extra').value = v.extra || '';
    chatFields().forEach(el => { const x = v['p:' + el.dataset.param]; if (el.type === 'checkbox') el.checked = !!x; else el.value = x == null ? '' : x; });
    const effortEl = document.querySelector('#chat-settings [data-param="reasoning_effort"]');
    if (chat.think === 'on' && effortEl && effortEl.value === 'none') effortEl.value = '';
    chatParamsUi();
  }
  // The engine's limit for a request that names none: known when the Supervisor launched it.
  function chatDefaultBudget() {
    const flag = Number(launchFlags['--default-max-tokens']);
    return Number.isInteger(flag) && flag > 0 ? {n: flag, known: true} : {n: 8192, known: false};
  }
  function chatJsonField(id, stateId, wantObject) {
    const raw = $(id).value.trim();
    let r = {ok: true, value: undefined};
    if (raw) {
      try {
        const v = JSON.parse(raw);
        r = !wantObject || (v && typeof v === 'object' && !Array.isArray(v)) ? {ok: true, value: v} : {ok: false, error: 'must be a JSON object'};
      } catch (e) { r = {ok: false, error: e.message}; }
    }
    const state = $(stateId);
    state.textContent = !raw ? '' : r.ok ? (wantObject ? `Valid · ${Object.keys(r.value).length} field${Object.keys(r.value).length === 1 ? '' : 's'}` : 'Valid') : 'Not valid JSON: ' + r.error;
    state.classList.toggle('red', !r.ok);
    $(id).classList.toggle('invalid', !r.ok);
    return r;
  }
  function chatParamsUi() {
    const mode = $('chat-budget').value;
    $('chat-max').hidden = mode !== 'custom';
    const def = chatDefaultBudget();
    const defText = def.known ? `${def.n.toLocaleString()} tokens` : `${def.n.toLocaleString()} tokens unless the engine was started with another value`;
    $('chat-budget-help').textContent = mode === 'default' ? `Sends no limit, so the engine applies its --default-max-tokens: ${defText}, reasoning included. This is the limit chat apps hit when they leave it out.`
      : mode === 'unlimited' ? 'Sends max_tokens -1. The answer ends only at a stop condition or when the conversation fills the context.'
      : 'The whole output budget, reasoning included. 0 reads the prompt without generating.';
    const now = $('chat-budget-now'), max = Number($('chat-max').value);
    now.textContent = mode === 'unlimited' ? 'No limit' : mode === 'custom' ? ($('chat-max').value !== '' && Number.isInteger(max) && max >= 0 ? max.toLocaleString() + ' tokens' : 'Enter a budget') : `Engine default${def.known ? ' · ' + def.n.toLocaleString() : ''}`;
    now.className = mode === 'default' ? 'engine' : '';
    now.parentElement.title = mode === 'default' && !def.known ? 'The engine was started elsewhere, so its --default-max-tokens is not known here; 8,192 is the engine default' : 'The output budget the next request carries';
    $('chat-schema-row').hidden = $('chat-format').value !== 'json_schema';
    const lp = $('chat-logprobs').checked;
)HTML"
R"HTML(    $('chat-top-logprobs').disabled = !lp;
    const effortEl = document.querySelector('#chat-settings [data-param="reasoning_effort"]');
    const budgetEl = document.querySelector('#chat-settings [data-param="thinking_budget_tokens"]');
    const thinkOff = chat.think === 'off';
    const effortNone = !thinkOff && effortEl && effortEl.value === 'none';
    if (effortEl) effortEl.disabled = thinkOff;
    if (budgetEl) budgetEl.disabled = thinkOff || effortNone;
    const rHelp = $('chat-reasoning-help');
    if (rHelp) {
      rHelp.hidden = !thinkOff && !effortNone;
      rHelp.textContent = thinkOff ? 'Thinking is Off in the bar above, so reasoning effort and thinking budget are not sent.'
        : effortNone ? 'Reasoning effort is none, which turns thinking off; thinking budget is not sent.' : '';
    }
    const schema = chatJsonField('chat-schema', 'chat-schema-state', true);
    if ($('chat-format').value === 'json_schema' && schema.ok && schema.value === undefined) {
      $('chat-schema-state').textContent = 'Enter the schema the answer must follow';
      $('chat-schema').classList.add('invalid');
    }
    chatJsonField('chat-extra', 'chat-extra-state', true);
    document.querySelectorAll('#chat-settings select').forEach(s => s.classList.toggle('set', !s.disabled && s.value !== '' && s.value !== 'default'));
    const sent = chatSentParams(chatRequestBody());
    const countText = sent.length ? ' · ' + sent.length : '';
    $('chat-param-count').textContent = countText;
    const barCount = $('chat-param-count-bar');
    if (barCount) barCount.textContent = countText;
  }
  // The parameters a request sets beyond the conversation and the page's own observations.
  const CHAT_PLUMBING = new Set(['model', 'messages', 'stream_options', 'return_progress', 'timings_per_token']);
  function chatSentParams(body) {
    return Object.keys(body).filter(k => !CHAT_PLUMBING.has(k) && body[k] !== undefined && !(k === 'stream' && body.stream === true))
      .map(k => [k, body[k]]);
  }
  function chatRequestBody() {
    const msgs = [];
    const system = $('chat-system').value.trim();
    if (system) msgs.push({role: 'system', content: system});
    // A turn whose reply failed or came back empty stays out of the history, question and all,
    // so the next request never carries two questions in a row. Replies carry their reasoning;
    // the template decides whether to keep it (preserve_thinking).
    const dropped = m => m && m.role === 'assistant' && (m.error || (!m.content && !CHAT_LIVE.has(m.state)));
    chat.messages.forEach((m, i) => {
      if (m.role === 'assistant') {
        if (m.content && !m.error) msgs.push(Object.assign({role: 'assistant', content: m.content}, m.reasoning ? {reasoning_content: m.reasoning} : {}));
        return;
      }
      if (dropped(chat.messages[i + 1])) return;
      if (m.images && m.images.length) msgs.push({role: 'user', content: [{type: 'text', text: m.content || ''}, ...m.images.map(url => ({type: 'image_url', image_url: {url}}))]});
      else msgs.push({role: 'user', content: m.content || ''});
    });
    const logprobs = $('chat-logprobs').checked;
    const stream = $('chat-stream').checked;
    const body = {model: benchServedModel || undefined, messages: msgs, stream};
    if (stream) Object.assign(body, {stream_options: {include_usage: true}, return_progress: true, timings_per_token: true});
    const budget = $('chat-budget').value, max = Number($('chat-max').value);
    if (budget === 'unlimited') body.max_tokens = -1;
    else if (budget === 'custom' && $('chat-max').value !== '' && Number.isInteger(max) && max >= 0) body.max_tokens = max;
    chatFields().forEach(el => {
      if (el.disabled) return;
      const key = el.dataset.param;
      if (el.type === 'checkbox') { if (el.checked) body[key] = true; return; }
      if (el.value === '') return;
      if (el.tagName === 'SELECT') { body[key] = el.value; return; }
      const n = Number(el.value);
      if (Number.isFinite(n)) body[key] = el.hasAttribute('data-int') ? Math.trunc(n) : n;
    });
    const stops = $('chat-stops').value.split('\n').filter(x => x.length).slice(0, 4);
    if (stops.length) body.stop = stops;
    const format = $('chat-format').value;
    if (format === 'json_object') body.response_format = {type: 'json_object'};
    else if (format === 'json_schema') {
      let schema;
      try { schema = JSON.parse($('chat-schema').value); } catch (_) {}
      if (schema && typeof schema === 'object') body.response_format = {type: 'json_schema', json_schema: {name: 'chat', schema}};
    }
    if (logprobs) {
      body.logprobs = true;
      const top = Number($('chat-top-logprobs').value);
      if (Number.isInteger(top) && top > 0) body.top_logprobs = Math.min(20, top);
    }
    if (chat.think !== 'default' && !(chat.think === 'on' && body.reasoning_effort === 'none')) {
      body.chat_template_kwargs = {enable_thinking: chat.think === 'on'};
    }
    let extra;
    try { extra = $('chat-extra').value.trim() ? JSON.parse($('chat-extra').value) : null; } catch (_) {}
    if (extra && typeof extra === 'object' && !Array.isArray(extra)) Object.assign(body, extra);
    return body;
  }
  function chatParamText(entries) {
    return entries.map(([k, v]) => `${k} ${typeof v === 'object' ? JSON.stringify(v) : String(v)}`).join(' · ');
  }
  async function chatSend(text) {
    if (chatAbort) return;
    const content = (text != null ? text : chatInput.value).trim();
    if (!content && !chatPending.length && text == null) return;
    const bad = $('chat-extra').classList.contains('invalid') ? 'chat-extra'
      : $('chat-format').value === 'json_schema' && $('chat-schema').classList.contains('invalid') ? 'chat-schema' : null;
    if (bad) {
      chatShowParams(true);
      $(bad).focus(); $('chat-hint').textContent = 'Fix the JSON under Parameters first';
      return;
    }
    if (text == null) {
      chat.messages.push({role: 'user', content, images: chatPending.slice(), time: Date.now()});
      chatPending = []; renderChatAttachments();
      chatInput.value = '';
    }
    const reply = {role: 'assistant', content: '', reasoning: '', model: benchServedModel, time: Date.now(), state: 'waiting', stats: {}, reasoningOpen: $('chat-show-reasoning').checked};
    chat.messages.push(reply);
    chat.selectedReply = null;
    const index = chat.messages.length - 1;
    const body = chatRequestBody();
    reply.sent = chatSentParams(body);
    chat.thinkUsed = chat.thinkUsed == null ? chat.think : chat.thinkUsed;
    chatRender();
    const ctl = new AbortController();
    chatAbort = ctl;
    $('chat-send').hidden = true; $('chat-stop').hidden = false;
    const t0 = performance.now();
    let last = 0;
    const paint = force => {
      const now = performance.now();
      if (!force && now - last <= 60) return;
      last = now;
      if (chat.messages[index] !== reply) return;
      const el = chatLog.querySelector(`.msg[data-index="${index}"]`);
      if (!el || !el._live) { chatRender(index); return; }
      const stick = chatLog.scrollTop + chatLog.clientHeight >= chatLog.scrollHeight - 24;
      chatLiveUpdate(reply, el);
      if (stick) chatLog.scrollTop = chatLog.scrollHeight;
    };
    try {
      const res = await fetch('/api/engine/v1/chat/completions', {method: 'POST', signal: ctl.signal, headers: {'Content-Type': 'application/json', 'X-NInfer-Supervisor': '1'}, body: JSON.stringify(body)});
      const streamed = (res.headers.get('content-type') || '').includes('text/event-stream');
      if (!res.ok) {
        let message = 'HTTP ' + res.status;
        try { const j = await res.json(); message = (j.error && (j.error.message || j.error)) || j.detail || message; } catch (_) {}
        throw new Error(typeof message === 'string' ? message : JSON.stringify(message));
      }
      if (!streamed) {
        reply.state = 'whole'; paint(true);
        const j = await res.json();
        const ch = (j.choices || [])[0] || {};
        const msg = ch.message || {};
        reply.stats.ttft = performance.now() - t0;
        reply.content = msg.content || '';
        reply.reasoning = msg.reasoning_content || msg.reasoning || '';
        if (msg.tool_calls) reply.content += (reply.content ? '\n\n' : '') + '```json\n' + JSON.stringify(msg.tool_calls, null, 2) + '\n```';
        reply.stats.finish = ch.finish_reason; reply.stats.usage = j.usage; reply.stats.timings = j.timings;
        reply.stats.whole = true;
        if (ch.logprobs && Array.isArray(ch.logprobs.content)) reply.tokens = ch.logprobs.content;
      } else await readSse(res, (name, payload) => {
        if (!payload || payload === '[DONE]') return;
        let d;
        try { d = JSON.parse(payload); } catch (_) { return; }
        if (d.error) throw new Error((d.error && d.error.message) || JSON.stringify(d.error));
        if (d.prompt_progress) { reply.state = 'reading'; reply.stats.progress = d.prompt_progress; paint(); }
        const ch = d.choices && d.choices[0];
        const delta = ch && ch.delta;
        if (delta && (delta.content || delta.reasoning_content)) {
          if (!Number.isFinite(reply.stats.ttft)) reply.stats.ttft = performance.now() - t0;
          if (delta.reasoning_content) { reply.reasoning += delta.reasoning_content; reply.state = 'thinking'; }
          if (delta.content) { reply.content += delta.content; reply.state = 'generating'; }
        }
        if (ch && ch.logprobs && Array.isArray(ch.logprobs.content) && ch.logprobs.content.length) {
          reply.tokens = (reply.tokens || []).concat(ch.logprobs.content);
        }
        if (ch && ch.finish_reason) reply.stats.finish = ch.finish_reason;
        if (d.timings) { reply.stats.timings = d.timings; if (Number.isFinite(d.timings.predicted_per_second) && d.timings.predicted_n > 1) reply.stats.liveRate = d.timings.predicted_per_second; }
        if (d.usage) reply.stats.usage = d.usage;
        paint();
      });
      if (reply.stats.finish) reply.state = 'done';
      else { reply.state = 'stopped'; reply.note = 'The answer ended without a finish reason, so it may be incomplete.'; }
    } catch (err) {
      if (err.name === 'AbortError') reply.state = 'stopped';
      else {
        ctl.abort();
        reply.state = 'error';
        reply.error = 'The model did not answer: ' + err.message;
        reply.note = 'This turn stays out of the conversation; Regenerate sends it again.';
      }
    } finally {
      chatAbort = null;
      $('chat-send').hidden = false; $('chat-stop').hidden = true;
      chatRender(); chatSave();
      chatInput.focus();
    }
  }
  // Server-sent events from a fetch response, one callback per event.
  async function readSse(res, onEvent) {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true}).replace(/\r\n/g, '\n');
      let cut;
      while ((cut = buffer.indexOf('\n\n')) >= 0) {
        const block = buffer.slice(0, cut); buffer = buffer.slice(cut + 2);
        let name = 'message'; const data = [];
        block.split('\n').forEach(line => {
          if (line.startsWith(':')) name = 'comment';
          else if (line.startsWith('event:')) name = line.slice(6).trim();
          else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
        });
        if (name !== 'comment') onEvent(name, data.join('\n'));
      }
    }
  }
  function chatRegenerate() {
    if (chatAbort) return;
    while (chat.messages.length && chat.messages[chat.messages.length - 1].role === 'assistant') chat.messages.pop();
    if (!chat.messages.length) return;
    chatSend('');
  }
  $('chat-form').onsubmit = e => { e.preventDefault(); chatSend(); };
  chatInput.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); chatSend(); } });
  $('chat-stop').onclick = () => { if (chatAbort) chatAbort.abort(); };
  $('chat-new').onclick = () => {
    if (chatAbort) chatAbort.abort();
    chat.messages = []; chat.thinkUsed = null; chat.selectedReply = null; chatPending = []; renderChatAttachments();
    chatSave(); chatRender(); chatInput.focus();
  };
  function chatShowParams(open) {
    $('chat-settings-toggle').setAttribute('aria-expanded', String(open));
    $('view-chat').querySelector('.chat').classList.toggle('params-open', open);
    if (open) chatParamsUi();
  }
  $('chat-settings-toggle').onclick = () => chatShowParams(!$('view-chat').querySelector('.chat').classList.contains('params-open'));
  $('chat-back').onclick = () => { chatShowParams(false); chatInput.focus(); };
  ['input', 'change'].forEach(type => $('chat-settings').addEventListener(type, e => {
    if (e.target.id === 'chat-system') return;
    if (e.target.dataset && e.target.dataset.param === 'reasoning_effort' && e.target.value === 'none' && chat.think === 'on') {
      chat.think = 'default';
      chatSave();
      chatRender();
    }
    chatParamsUi(); chatSaveParams();
    $('chat-hint').textContent = '';
  }));
  $('chat-params-reset').onclick = () => { try { localStorage.removeItem(PARAMS_KEY); } catch (_) {} chatLoadParams(); };
)HTML"
R"HTML(  $('chat-system').addEventListener('input', () => { chat.system = $('chat-system').value; chatSave(); chatParamsUi(); });
  document.querySelectorAll('[data-think]').forEach(b => {
    b.onclick = () => {
      chat.think = b.dataset.think;
      const effortEl = document.querySelector('#chat-settings [data-param="reasoning_effort"]');
      if (chat.think === 'on' && effortEl && effortEl.value === 'none') {
        effortEl.value = '';
        chatSaveParams();
      }
      chatParamsUi();
      chatSave();
      chatRender();
    };
  });
  $('chat-inspect').onclick = () => {
    const body = chatRequestBody();
    const draft = chatInput.value.trim();
    if (draft) body.messages.push({role: 'user', content: draft});
    const ep = benchEndpoints.find(e => e.path === '/v1/chat/completions');
    benchReady = true;
    benchSelect(ep, -1);
    benchSetBody(body);
    renderBenchAuth();
    location.hash = '#api';
  };

  // Images: attach, paste or drop. Large images are scaled to 1600 px on the long side,
  // which keeps the request small without changing what the model can read.
  async function chatImage(file) {
    if (!file || !file.type.startsWith('image/')) return;
    const url = await new Promise((resolve, reject) => { const r = new FileReader(); r.onload = () => resolve(r.result); r.onerror = reject; r.readAsDataURL(file); });
    const img = await new Promise((resolve, reject) => { const i = new Image(); i.onload = () => resolve(i); i.onerror = reject; i.src = url; });
    const scale = Math.min(1, 1600 / Math.max(img.width, img.height));
    if (scale >= 1 && file.size < 2_000_000) { chatPending.push(url); return; }
    const c = document.createElement('canvas');
    c.width = Math.round(img.width * scale); c.height = Math.round(img.height * scale);
    c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
    chatPending.push(c.toDataURL('image/jpeg', 0.9));
  }
  async function chatAddFiles(files) {
    for (const f of files) { try { await chatImage(f); } catch (_) {} }
    renderChatAttachments();
  }
  function renderChatAttachments() {
    const box = $('chat-attachments');
    box.hidden = !chatPending.length;
    box.replaceChildren(...chatPending.map((src, i) => {
      const wrap = document.createElement('span'); wrap.className = 'chat-thumb';
      const img = document.createElement('img'); img.src = src; img.alt = 'Image to send';
      const rm = document.createElement('button'); rm.type = 'button'; rm.className = 'key key-tiny'; rm.textContent = 'Remove';
      rm.onclick = () => { chatPending.splice(i, 1); renderChatAttachments(); };
      wrap.append(img, rm); return wrap;
    }));
    $('chat-hint').textContent = chatPending.length ? `${chatPending.length} image${chatPending.length === 1 ? '' : 's'} attached` : '';
  }
  $('chat-attach').onclick = () => $('chat-file').click();
  $('chat-file').onchange = e => { chatAddFiles([...e.target.files]); e.target.value = ''; };
  chatInput.addEventListener('paste', e => { const files = [...(e.clipboardData || {}).files || []].filter(f => f.type.startsWith('image/')); if (files.length && chatVision) { e.preventDefault(); chatAddFiles(files); } });
  ['dragover', 'dragleave', 'drop'].forEach(type => $('chat-form').addEventListener(type, e => {
    if (!chatVision) return;
    e.preventDefault();
    $('chat-form').classList.toggle('drop', type === 'dragover');
    if (type === 'drop') chatAddFiles([...e.dataTransfer.files]);
  }));

  async function chatInit() {
    if (!chatReady) {
      chatReady = true;
      $('chat-system').value = chat.system || '';
      chatLoadParams();
      chatRender();
    } else chatParamsUi();
    chatInput.focus();
    // Asked again on each visit until the engine answers: it may start after the page.
    if (benchServedModel && chatCtx) return;
    try {
      const props = await (await fetch('/api/engine/props', {headers: {'X-NInfer-Supervisor': '1'}})).json();
      chatVision = !!(props.modalities && props.modalities.vision);
      chatCtx = (props.default_generation_settings || {}).n_ctx || 0;
    } catch (_) {}
    if (!benchServedModel) {
      try {
        const data = await (await fetch('/api/engine/v1/models', {headers: {'X-NInfer-Supervisor': '1'}})).json();
        if (data && Array.isArray(data.data) && data.data[0]) benchServedModel = data.data[0].id;
      } catch (_) {}
    }
    $('chat-attach').hidden = !chatVision;
    $('chat-model').textContent = benchServedModel ? benchServedModel + (chatVision ? ' · text and images' : '') : 'Model not reachable';
    chatRender();
  }

  // ---------------------------------------------------------------------
  // Diagnose > Experts. The engine counts how often each routed expert is chosen
  // (GET /admin/experts, present only when it was started with --expert-stats). While the
  // page is on screen it asks every 2 s and draws one map: a row per layer, a column per
  // expert, colour = heat scaled inside its own row. Decode is the generation channel
  // (green), prefill the prompt channel (ice); the bars are measured (white); the budget
  // and its cut marks are a target (magenta). Nothing here is polled while it is hidden.
  // ---------------------------------------------------------------------
  const EXPERT_BYTES = 2764808;            // one routed expert of this model, packed
  const EX_POLL_MS = 2000;
  const EX_VIEW_KEY = 'ninfer:experts-view', EX_BUDGET_KEY = 'ninfer:expert-budget';
  const EX_LEFT = 36, EX_TOP = 17, EX_BOTTOM = 19, EX_GAP = 8;
  const exCanvas = $('ex-canvas'), exMap = $('ex-map'), exKeyCanvas = $('ex-key');
  const ex = {frame: null, kind: 'decode', order: 'id', budget: 8192, cur: null, at: 0, target: null, shown: null,
    from: null, animStart: 0, raf: 0, version: 0, lay: null, img: null, imgKey: '', cuts: null, lut: null, lutKey: '', cfgKey: ''};
  let exTimer = 0, exBusy = false, exSeq = 0, exAbort = null;
  try {
    const v = JSON.parse(localStorage.getItem(EX_VIEW_KEY) || '{}');
    if (v.kind === 'prefill') ex.kind = 'prefill';
    if (v.order === 'sorted') ex.order = 'sorted';
    const b = parseInt(localStorage.getItem(EX_BUDGET_KEY), 10);
    if (Number.isFinite(b) && b >= 0) ex.budget = b;
  } catch (_) {}

  const exInt = n => Number.isFinite(n) ? Math.round(n).toLocaleString() : '—';
  const exPct = x => !Number.isFinite(x) ? '—' : (x >= 0.1 ? (x * 100).toFixed(1) : (x * 100).toFixed(2)) + '%';
  function exBytes(n) {
    if (!Number.isFinite(n)) return '—';
    if (n >= GiB) return (n / GiB).toFixed(2) + ' GiB';
    if (n >= MiB) return (n / MiB).toFixed(1) + ' MiB';
    if (n >= 1024) return (n / 1024).toFixed(1) + ' KiB';
    return Math.round(n) + ' B';
  }
  const exKind = f => f.kinds[ex.kind] ? ex.kind : 'decode';

  // The reply, as typed arrays: one flat row-major [row][expert] table per kind.
  function exIngest(d) {
    const g = d.geometry || {};
    const L = g.text_layers | 0, E = g.experts | 0, K = g.top_k | 0;
    const R = L + (Number.isInteger(g.mtp_layer) ? 1 : 0);
    if (!(L > 0 && E > 1 && E <= 65536)) throw new Error('geometry');
    const f = {L, E, K, R, mtp: R > L, uptime: d.uptime_seconds, half: d.half_life_seconds, records: d.records || {}, columns: d.columns || {}, trace: d.trace || null, engine80: null, kinds: {}};
    for (const kind of ['decode', 'prefill']) {
      const rows = d[kind] && d[kind].heat;
      if (!Array.isArray(rows) || rows.length !== R) { if (kind === 'decode') throw new Error('heat'); continue; }
      const heat = new Float64Array(R * E);
      for (let r = 0; r < R; r++) {
        const row = rows[r];
        if (!Array.isArray(row) || row.length !== E) throw new Error('row');
        for (let i = 0; i < E; i++) heat[r * E + i] = row[i] > 0 ? +row[i] : 0;
      }
      f.kinds[kind] = {heat, derived: null};
    }
    const e80 = d.decode.experts_for_80_percent;
    if (Array.isArray(e80) && e80.length === R) f.engine80 = e80;
    return f;
  }

  // Order, prefix sums and coverage per row; the busiest text cells overall for the budget.
  function exDerive(f, kind) {
    const kd = f.kinds[kind] || f.kinds.decode;
    if (kd.derived) return kd.derived;
    const {L, E, R} = f, heat = kd.heat;
    const order = new Uint16Array(R * E), rank = new Uint16Array(R * E), prefix = new Float64Array(R * (E + 1));
    const sum = new Float64Array(R), peak = new Float64Array(R), cover = new Array(R).fill(null);
    const idx = new Uint16Array(E);
    for (let r = 0; r < R; r++) {
      const base = r * E, pb = r * (E + 1);
      for (let i = 0; i < E; i++) idx[i] = i;
      idx.sort((a, b) => heat[base + b] - heat[base + a] || a - b);
      order.set(idx, base);
      let acc = 0;
      for (let i = 0; i < E; i++) { rank[base + idx[i]] = i; acc += heat[base + idx[i]]; prefix[pb + i + 1] = acc; }
      sum[r] = acc; peak[r] = heat[base + idx[0]];
      if (acc > 0) {
        const goal = acc * 0.8 * (1 - 1e-12);
        let k = 1;
        while (k < E && prefix[pb + k] < goal) k++;
        cover[r] = k;
      }
    }
    if (kind === 'decode' && f.engine80) f.engine80.forEach((v, r) => { cover[r] = Number.isInteger(v) ? v : null; });
    let gAsc = null, gPrefix = null, textSum = 0;
    if (kind === 'decode') {
      const n = L * E;
      gAsc = heat.slice(0, n).sort();
      gPrefix = new Float64Array(n + 1);
      for (let i = 0; i < n; i++) gPrefix[i + 1] = gPrefix[i] + gAsc[n - 1 - i];
      textSum = gPrefix[n];
    }
    return kd.derived = {order, rank, prefix, sum, peak, cover, gAsc, gPrefix, textSum};
  }

  // What the map shows: each row's heat on a square-root scale of its own peak, in the chosen order.
  function exMatrix(f, kind, order) {
    const d = exDerive(f, kind), heat = (f.kinds[kind] || f.kinds.decode).heat, E = f.E;
    const m = new Float32Array(f.R * E);
    for (let r = 0; r < f.R; r++) {
      const base = r * E, pk = d.peak[r];
      if (!(pk > 0)) continue;
      for (let i = 0; i < E; i++) {
        const h = order === 'sorted' ? heat[base + d.order[base + i]] : heat[base + i];
        m[base + i] = h > 0 ? Math.sqrt(h / pk) : 0;
      }
    }
    return m;
  }

  function rgbOf(css) {
    css = String(css).trim();
    let m = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(css);
    if (m) {
      let h = m[1];
      if (h.length === 3) h = h.replace(/./g, '$&$&');
      const n = parseInt(h, 16);
      return [n >> 16 & 255, n >> 8 & 255, n & 255];
    }
    m = /rgba?\(([^)]+)\)/.exec(css);
    if (m) { const p = m[1].split(',').map(Number); return [p[0] | 0, p[1] | 0, p[2] | 0]; }
    return [128, 128, 128];
  }
  // 256 steps from the empty well to the channel colour; any activity is lifted off the floor.
  function exLut(kind) {
    const chan = kind === 'prefill' ? C.ice : C.green, key = chan + '|' + C.glass2;
    if (ex.lutKey === key) return ex.lut;
    const a = rgbOf(C.glass2), b = rgbOf(chan), lut = new Uint32Array(256);
    for (let k = 0; k < 256; k++) {
      const w = k === 0 ? 0 : 0.14 + 0.86 * ((k - 1) / 254);
      const r = Math.round(a[0] + (b[0] - a[0]) * w), g = Math.round(a[1] + (b[1] - a[1]) * w), bl = Math.round(a[2] + (b[2] - a[2]) * w);
      lut[k] = ((255 << 24) | (bl << 16) | (g << 8) | r) >>> 0;
    }
    ex.lutKey = key;
    return ex.lut = lut;
  }
  function exDrawKey(kind) {
    const dpr = window.devicePixelRatio || 1, w = Math.round(96 * dpr), h = Math.round(10 * dpr);
    if (exKeyCanvas.width !== w || exKeyCanvas.height !== h) { exKeyCanvas.width = w; exKeyCanvas.height = h; }
    const c = exKeyCanvas.getContext('2d'), lut = exLut(kind), img = c.createImageData(w, h), px = new Uint32Array(img.data.buffer);
    for (let x = 0; x < w; x++) {
      const t = x / (w - 1), v = lut[t > 0 ? 1 + ((t * 254 + .5) | 0) : 0];
      for (let y = 0; y < h; y++) px[y * w + x] = v;
    }
    c.putImageData(img, 0, 0);
  }

  // Geometry in CSS pixels for the layout, in device pixels for the cells, so a cell edge is never blurred.
  function exLayout(W, H, dpr, f) {
    const key = [W, H, dpr, f.R, f.L, f.E].join('|');
    if (ex.lay && ex.lay.key === key) return ex.lay;
    const barW = W < 560 ? 52 : 76, barGap = 10;
    const plotX = EX_LEFT, plotW = Math.max(60, W - EX_LEFT - barGap - barW - 2);
    const gap = f.mtp ? EX_GAP : 0;
)HTML"
R"HTML(    const plotH = Math.max(f.R, H - EX_TOP - EX_BOTTOM - gap);
    const rowH = plotH / f.R;
    const X0 = Math.round(plotX * dpr), PW = Math.max(1, Math.round(plotW * dpr));
    const ys = new Int32Array(f.R), ye = new Int32Array(f.R);
    for (let r = 0; r < f.R; r++) {
      const top = EX_TOP + r * rowH + (r >= f.L ? gap : 0);
      ys[r] = Math.round(top * dpr); ye[r] = Math.round((top + rowH) * dpr);
    }
    const Y0 = ys[0], PH = ye[f.R - 1] - Y0;
    const colLo = new Uint16Array(PW), colHi = new Uint16Array(PW);
    for (let x = 0; x < PW; x++) {
      const lo = Math.floor(x * f.E / PW);
      colLo[x] = lo; colHi[x] = Math.max(lo + 1, Math.floor((x + 1) * f.E / PW));
    }
    return ex.lay = {key, W, H, dpr, plotX, plotW, barX: plotX + plotW + barGap, barW, gap, rowH, X0, PW, Y0, PH, ys, ye, colLo, colHi, bottom: (Y0 + PH) / dpr};
  }
  // More experts than pixels: each pixel column shows the busiest expert under it.
  function exBuildImage(f, lay, m, kind) {
    const lut = exLut(kind), E = f.E, PW = lay.PW;
    const img = new ImageData(PW, lay.PH), px = new Uint32Array(img.data.buffer), line = new Uint32Array(PW);
    for (let r = 0; r < f.R; r++) {
      const base = r * E;
      for (let x = 0; x < PW; x++) {
        let t = 0;
        for (let e = lay.colLo[x], e1 = lay.colHi[x]; e < e1; e++) { const v = m[base + e]; if (v > t) t = v; }
        line[x] = lut[t > 0 ? 1 + ((t * 254 + .5) | 0) : 0];
      }
      const y0 = lay.ys[r] - lay.Y0, y1 = lay.ye[r] - lay.Y0 - (lay.ye[r] - lay.ys[r] >= 4 ? 1 : 0);
      for (let y = y0; y < y1; y++) px.set(line, y * PW);
    }
    return img;
  }

  function exDraw() {
    const f = ex.frame;
    if (!f || $('ex-live').hidden || !ex.shown) return;
    const rect = exCanvas.getBoundingClientRect();
    if (rect.width < 120 || rect.height < 60) return;
    const dpr = window.devicePixelRatio || 1;
    const cw = Math.round(rect.width * dpr), ch = Math.round(rect.height * dpr);
    if (exCanvas.width !== cw || exCanvas.height !== ch) { exCanvas.width = cw; exCanvas.height = ch; ex.imgKey = ''; }
    const c = exCanvas.getContext('2d');
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.clearRect(0, 0, cw, ch);
    const lay = exLayout(rect.width, rect.height, dpr, f), kind = exKind(f), kd = exDerive(f, kind), sel = ex.cur;
    const key = [ex.version, cw, ch, kind, ex.order, C.glass2, C.green, C.ice].join('|');
    if (ex.imgKey !== key) { ex.img = exBuildImage(f, lay, ex.shown, kind); ex.imgKey = key; }
    c.putImageData(ex.img, lay.X0, lay.Y0);
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    const d = v => v / dpr, sep = r => (lay.ye[r] - lay.ys[r] >= 4 ? 1 : 0);
    const barOf = r => kd.cover[r] == null ? 0 : lay.barW * kd.cover[r] / f.E;

    // Layer labels, sparse enough to read at any row height; the pointed layer is always named.
    const step = lay.rowH >= 11 ? 2 : lay.rowH >= 6 ? 4 : 8;
    c.font = '500 10.5px Bahnschrift, sans-serif'; c.textBaseline = 'alphabetic'; c.textAlign = 'right';
    for (let r = 0; r < f.R; r++) {
      const on = sel && sel.row === r, mtp = r >= f.L;
      if (!(mtp || r % step === 0 || on)) continue;
      if (!on && sel && sel.row < f.L && Math.abs(r - sel.row) * lay.rowH < 11) continue;   // the pointed layer's label has the room
      c.fillStyle = on ? C.ink : C.faint;
      c.fillText(mtp ? 'MTP' : String(r), EX_LEFT - 6, d(lay.ys[r] + lay.ye[r]) / 2 + 3.5);
    }
    c.font = '600 10.5px Bahnschrift, sans-serif'; c.fillStyle = C.faint;
    c.textAlign = 'right'; c.fillText('LAYER', EX_LEFT - 6, EX_TOP - 6);
    c.textAlign = 'left'; c.fillText(ex.order === 'sorted' ? 'RANK, BUSIEST FIRST' : 'EXPERT NUMBER', lay.plotX + 6, EX_TOP - 6);
    c.fillText('80% IN', lay.barX, EX_TOP - 6);

    // Per-row bars: how many experts cover 80% of that row's activity, on a 0..E track.
    for (let r = 0; r < f.R; r++) {
      const y = d(lay.ys[r]), h = d(lay.ye[r] - lay.ys[r] - sep(r)), on = sel && sel.row === r;
      c.fillStyle = C.rule2; c.fillRect(lay.barX, y, lay.barW, h);
      if (kd.cover[r] != null) { c.globalAlpha = on ? 1 : .78; c.fillStyle = C.ink; c.fillRect(lay.barX, y, barOf(r), h); c.globalAlpha = 1; }
    }

    // Axes.
    const ay = Math.round(lay.bottom) + 3.5;
    c.strokeStyle = C.axis; c.lineWidth = 1; c.fillStyle = C.faint; c.font = '500 10.5px Bahnschrift, sans-serif';
    const ticks = ex.order === 'sorted' ? [1, f.E / 8, f.E / 4, f.E / 2, f.E] : [0, f.E / 4, f.E / 2, 3 * f.E / 4, f.E - 1];
    ticks.forEach((v, i) => {
      v = Math.round(v);
      const x = Math.round(lay.plotX + (ex.order === 'sorted' ? v - .5 : v + .5) / f.E * lay.plotW) + .5;
      c.beginPath(); c.moveTo(x, ay); c.lineTo(x, ay + 3); c.stroke();
      c.textAlign = i === 0 ? 'left' : i === ticks.length - 1 ? 'right' : 'center';
      c.fillText(String(v), i === 0 ? x - .5 : i === ticks.length - 1 ? x + .5 : x, ay + 14);
    });
    c.beginPath(); c.moveTo(lay.plotX, ay); c.lineTo(lay.plotX + lay.plotW, ay); c.moveTo(lay.barX, ay); c.lineTo(lay.barX + lay.barW, ay); c.stroke();
    c.textAlign = 'left'; c.fillText('0', lay.barX, ay + 14);
    c.textAlign = 'right'; c.fillText(String(f.E), lay.barX + lay.barW, ay + 14);

    // The budget's cuts, in sorted decode order only: each layer's own cut for the global pick
    // (solid), the one rank every layer would get under the even split (dashed).
    if (ex.order === 'sorted' && kind === 'decode' && ex.cuts) {
      c.fillStyle = C.magenta;
      if (ex.cuts.perRow) {
        for (let r = 0; r < f.L; r++) {
          const p = ex.cuts.perRow[r];
          if (p <= 0 || p >= f.E) continue;
          c.fillRect(d(lay.X0 + Math.round(p / f.E * lay.PW) - 1), d(lay.ys[r]), d(2), d(lay.ye[r] - lay.ys[r] - sep(r)));
        }
      }
      if (ex.cuts.even > 0 && ex.cuts.even < f.E) {
        const x = d(lay.X0 + Math.round(ex.cuts.even / f.E * lay.PW)) + .5;
        c.strokeStyle = C.magenta; c.lineWidth = 1; c.setLineDash([4, 3]);
        c.beginPath(); c.moveTo(x, EX_TOP - 2); c.lineTo(x, d(lay.ye[f.L - 1]) + 2); c.stroke(); c.setLineDash([]);
      }
    }

    // The pointed cell: a white box (measured), with the row bracketed so a thin row is findable.
    if (sel && sel.row < f.R) {
      const ys = lay.ys[sel.row], ye = lay.ye[sel.row];
      const x0 = lay.X0 + Math.floor(sel.col * lay.PW / f.E), x1 = lay.X0 + Math.max(Math.floor(sel.col * lay.PW / f.E) + 1, Math.floor((sel.col + 1) * lay.PW / f.E));
      const half = Math.max((x1 - x0) / 2, 1.5 * dpr), cx = (x0 + x1) / 2;
      c.strokeStyle = C.dim; c.lineWidth = 1; c.globalAlpha = .7;
      c.beginPath();
      c.moveTo(lay.plotX, d(ys) - .5); c.lineTo(lay.plotX + lay.plotW, d(ys) - .5);
      c.moveTo(lay.plotX, d(ye) + .5); c.lineTo(lay.plotX + lay.plotW, d(ye) + .5);
      c.stroke(); c.globalAlpha = 1;
      c.strokeStyle = C.ink; c.lineWidth = 1.5;
      c.strokeRect(d(cx - half), d(ys) - 1, d(2 * half), d(ye - ys) + 2);
    }
  }

  // Slew: after each reading the map travels to the new values like every other instrument,
  // and rests on the measured ones. Reduced motion and a hidden page jump.
  function exSetTarget(jump) {
    const f = ex.frame;
    if (!f) return;
    const t = exMatrix(f, exKind(f), ex.order);
    const keep = ex.shown && ex.target && ex.target.length === t.length && !jump && !reducedMotion && !document.hidden;
    const from = keep ? ex.shown.slice() : null;
    if (ex.raf) { cancelAnimationFrame(ex.raf); ex.raf = 0; }
    ex.target = t;
    const idle = !f.kinds[exKind(f)] || !exDerive(f, exKind(f)).sum.some(v => v > 0);
    $('ex-idle').hidden = !idle;
    if (!from) { ex.shown = t; ex.version++; exDraw(); return; }
    ex.from = from; ex.shown = new Float32Array(t.length); ex.animStart = performance.now();
    exStep();
  }
  function exStep() {
    const p = Math.min(1, (performance.now() - ex.animStart) / SLEW_MS), e = easeOut(p);
    if (p >= 1) ex.shown = ex.target;
    else for (let i = 0; i < ex.shown.length; i++) ex.shown[i] = ex.from[i] + (ex.target[i] - ex.from[i]) * e;
    ex.version++; exDraw();
    ex.raf = p < 1 ? requestAnimationFrame(exStep) : 0;
  }

  function exPaintFacts() {
    const f = ex.frame, rec = f.records, col = f.columns;
    const pair = k => `${exInt(rec[k])} steps · ${exInt(col[k])} tokens`;
    $('ex-f-decode').textContent = pair('decode');
    $('ex-f-prefill').textContent = pair('prefill');
    const mtpOn = f.mtp || rec.mtp_proposal > 0 || rec.mtp_verify > 0;
    $('ex-f-draft-wrap').hidden = $('ex-f-verify-wrap').hidden = !mtpOn;
    $('ex-f-draft').textContent = pair('mtp_proposal');
    $('ex-f-verify').textContent = pair('mtp_verify');
    $('ex-f-up').textContent = formatUptime(f.uptime);
    $('ex-f-half').textContent = Number.isFinite(f.half) ? `${f.half} s` : '—';
    const t = f.trace;
    $('ex-f-trace').innerHTML = !t ? '<span class="dim">Not recording</span>'
      : `<span class="ex-file">${esc(t.file)}</span> ${esc(exBytes(t.bytes_written))} written · ${esc(exBytes(t.bytes_queued))} queued · <span class="${t.records_dropped > 0 ? 'ex-bad' : ''}">${esc(exInt(t.records_dropped))} dropped</span>`;
  }
  function exPaintSkew() {
    const f = ex.frame, kind = exKind(f), d = exDerive(f, kind);
    const v = [];
    for (let r = 0; r < f.L; r++) if (d.cover[r] != null) v.push(d.cover[r]);
    v.sort((a, b) => a - b);
    const n = v.length, med = !n ? NaN : n % 2 ? v[(n - 1) / 2] : Math.round((v[n / 2 - 1] + v[n / 2]) / 2);
    const put = (id, x) => countTo($(id), x, q => String(Math.round(q)));
    put('ex-sk-min', n ? v[0] : NaN); put('ex-sk-med', med); put('ex-sk-max', n ? v[n - 1] : NaN);
    $('ex-sk-mtp-wrap').hidden = !f.mtp;
    put('ex-sk-mtp', f.mtp && d.cover[f.L] != null ? d.cover[f.L] : NaN);
    $('ex-sk-of').textContent = String(f.E);
    document.querySelector('#ex-skew .ex-lg').textContent = `Experts for 80% · ${kind}`;
  }

  // Budget: the hottest N text-layer experts against the decayed decode activations on screen,
  // as one pool (global) and as an even N/layers per layer. Both are best cases for a static
  // set chosen from this traffic, never a measured hit rate.
  function exBudgetSync(from) {
    const f = ex.frame, max = f ? f.L * f.E : 24576;
    const n = Math.max(0, Math.min(max, Math.round(ex.budget)));
    ex.budget = n;
    if (from !== 'slider') $('ex-slider').value = n;
    if (from !== 'n') $('ex-n').value = n;
    if (from !== 'gib') $('ex-gib').value = (n * EXPERT_BYTES / GiB).toFixed(1);
  }
  function exBudgetUpdate() {
    const f = ex.frame;
    if (!f) return;
    const d = exDerive(f, 'decode'), n = Math.min(ex.budget, f.L * f.E), k = Math.min(f.E, Math.round(n / f.L));
    $('ex-dt-global').innerHTML = `Hottest <b>${n.toLocaleString()}</b> anywhere`;
    $('ex-dt-even').innerHTML = `Even <b>${k}</b> per layer`;
    $('ex-dt-even').title = `${k} per layer is ${(k * f.L).toLocaleString()} experts (${(k * f.L * EXPERT_BYTES / GiB).toFixed(1)} GiB)`;
    const gEl = $('ex-share-global'), eEl = $('ex-share-even'), fmt = v => v.toFixed(1) + '%';
    if (!(d.textSum > 0)) { countTo(gEl, NaN, fmt); countTo(eEl, NaN, fmt); ex.cuts = null; return; }
    let even = 0;
    for (let r = 0; r < f.L; r++) even += d.prefix[r * (f.E + 1) + k];
    countTo(gEl, 100 * d.gPrefix[n] / d.textSum, fmt);
    countTo(eEl, 100 * even / d.textSum, fmt);
    // Where the global pick cuts each layer; ties at the threshold are shared out evenly.
    let perRow = null;
    if (n > 0 && n < f.L * f.E) {
      const heat = f.kinds.decode.heat, T = d.gAsc[d.gAsc.length - n];
      const gt = new Int32Array(f.L), eq = new Int32Array(f.L);
      let gtSum = 0, eqSum = 0;
      for (let r = 0; r < f.L; r++) {
        const base = r * f.E;
        for (let p = 0; p < f.E; p++) {
          const h = heat[base + d.order[base + p]];
          if (h > T) gt[r]++; else if (h === T) eq[r]++; else break;
        }
        gtSum += gt[r]; eqSum += eq[r];
      }
)HTML"
R"HTML(      const share = eqSum > 0 ? (n - gtSum) / eqSum : 0;
      perRow = new Int32Array(f.L);
      for (let r = 0; r < f.L; r++) perRow[r] = gt[r] + Math.round(eq[r] * share);
    }
    ex.cuts = {perRow, even: k};
  }
  function exBudgetSet(n, from) {
    ex.budget = n;
    exBudgetSync(from);
    try { localStorage.setItem(EX_BUDGET_KEY, String(ex.budget)); } catch (_) {}
    exBudgetUpdate(); exDraw();
  }
  $('ex-slider').addEventListener('input', e => exBudgetSet(+e.target.value, 'slider'));
  $('ex-n').addEventListener('input', e => { if (e.target.value !== '' && Number.isFinite(+e.target.value)) exBudgetSet(+e.target.value, 'n'); });
  $('ex-gib').addEventListener('input', e => { if (e.target.value !== '' && Number.isFinite(+e.target.value)) exBudgetSet(+e.target.value * GiB / EXPERT_BYTES, 'gib'); });
  ['ex-n', 'ex-gib'].forEach(id => $(id).addEventListener('change', () => exBudgetSet(ex.budget, '')));

  // What the pointer or the arrow keys are on: layer, expert, rank, heat, share of the layer.
  function exReadout(announce) {
    const el = $('ex-read'), f = ex.frame, c = ex.cur;
    if (!f || !c) { el.className = 'ex-read'; el.innerHTML = '<span class="ex-hint">Point at the map, or focus it and use the arrow keys.</span>'; return; }
    const kind = exKind(f), d = exDerive(f, kind), base = c.row * f.E;
    const id = ex.order === 'sorted' ? d.order[base + c.col] : c.col, rank = d.rank[base + id];
    const h = f.kinds[kind].heat[base + id], sum = d.sum[c.row];
    const layer = c.row < f.L ? String(c.row) : 'MTP', cover = d.cover[c.row];
    const share = sum > 0 ? h / sum : NaN, cum = sum > 0 ? d.prefix[c.row * (f.E + 1) + rank + 1] / sum : NaN;
    el.className = 'ex-read live';
    el.innerHTML = `<span><i>Layer</i>${esc(layer)}</span><span><i>Expert</i>${id}</span><span><i>Rank</i>${rank + 1}</span><span><i>Heat</i>${esc(exInt(h))}</span><span><i>Share</i>${esc(exPct(share))}</span><span><i>Cumulative</i>${esc(exPct(cum))}</span>${cover != null ? `<span><i>80% in</i>${cover}</span>` : ''}`;
    if (announce) a11yLive.textContent = `${c.row < f.L ? 'Layer ' + layer : 'MTP layer'}, expert ${id}, rank ${rank + 1}, heat ${exInt(h)}, ${exPct(share)} of the layer, cumulative ${exPct(cum)}${cover != null ? ', 80 percent of the layer in ' + cover + ' experts' : ''}.`;
  }
  function exSetCursor(cur, announce) {
    const same = (!cur && !ex.cur) || (cur && ex.cur && cur.row === ex.cur.row && cur.col === ex.cur.col);
    if (same) return;
    ex.cur = cur; exReadout(announce); exDraw();
  }
  function exPoint(e) {
    const f = ex.frame, lay = ex.lay;
    if (!f || !lay || !ex.target) return null;
    const r = exCanvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    if (mx < lay.plotX || mx >= lay.plotX + lay.plotW || my < EX_TOP) return null;
    const y = my - EX_TOP, textH = f.L * lay.rowH;
    let row;
    if (y < textH) row = Math.floor(y / lay.rowH);
    else if (f.mtp && y >= textH + lay.gap && y < textH + lay.gap + lay.rowH) row = f.L;
    else return null;
    const x = Math.min(lay.PW - 1, Math.max(0, Math.floor((mx - lay.plotX) * lay.dpr)));
    let col = lay.colLo[x], best = -1;
    for (let q = lay.colLo[x]; q < lay.colHi[x]; q++) { const v = ex.target[row * f.E + q]; if (v > best) { best = v; col = q; } }
    return {row, col};
  }
  exCanvas.addEventListener('mousemove', e => exSetCursor(exPoint(e), false));
  exCanvas.addEventListener('mouseleave', () => exSetCursor(null));
  exCanvas.addEventListener('blur', () => exSetCursor(null));
  exCanvas.addEventListener('keydown', e => {
    const f = ex.frame;
    if (!f) return;
    if (e.key === 'Escape') { exSetCursor(null); return; }
    const big = e.shiftKey ? 16 : 1;
    let {row, col} = ex.cur || {row: 0, col: 0};
    if (e.key === 'ArrowLeft') col -= big;
    else if (e.key === 'ArrowRight') col += big;
    else if (e.key === 'ArrowUp') row -= e.shiftKey ? 8 : 1;
    else if (e.key === 'ArrowDown') row += e.shiftKey ? 8 : 1;
    else if (e.key === 'Home') col = 0;
    else if (e.key === 'End') col = f.E - 1;
    else return;
    e.preventDefault();
    row = Math.max(0, Math.min(f.R - 1, row)); col = Math.max(0, Math.min(f.E - 1, col));
    ex.cur = null;
    exSetCursor({row, col}, true);
  });

  document.querySelectorAll('[data-ex-kind],[data-ex-order]').forEach(btn => btn.addEventListener('click', () => {
    const f = ex.frame, c = ex.cur;
    if (btn.dataset.exKind) ex.kind = btn.dataset.exKind;
    if (btn.dataset.exOrder && btn.dataset.exOrder !== ex.order) {
      // Keep the pointed expert when the columns change meaning (id <-> rank).
      if (f && c) {
        const d = exDerive(f, exKind(f)), base = c.row * f.E;
        ex.cur = {row: c.row, col: btn.dataset.exOrder === 'sorted' ? d.rank[base + c.col] : d.order[base + c.col]};
      }
      ex.order = btn.dataset.exOrder;
    }
    document.querySelectorAll('[data-ex-kind]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.exKind === ex.kind)));
    document.querySelectorAll('[data-ex-order]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.exOrder === ex.order)));
    try { localStorage.setItem(EX_VIEW_KEY, JSON.stringify({kind: ex.kind, order: ex.order})); } catch (_) {}
    if (!f) return;
    exDrawKey(exKind(f)); exSetTarget(true); exPaintSkew(); exReadout();
  }));
  document.querySelectorAll('[data-ex-kind]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.exKind === ex.kind)));
  document.querySelectorAll('[data-ex-order]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.exOrder === ex.order)));
  if (window.ResizeObserver) new ResizeObserver(() => requestAnimationFrame(exDraw)).observe(exMap);

  // States other than a reading. Last good values stay (dimmed, tagged) through a short outage;
  // anything else replaces the map, because a reading that no longer applies is worse than none.
  function exStateFor(kind) {
    const engineState = (lastState && lastState.engine && lastState.engine.state) || '';
    const configFile = cfgData && cfgData.config_path;
    if (kind === 'disabled') {
      return {title: 'Expert counters are off',
        text: 'The engine is not counting which experts it uses. Start it with --expert-stats to see live use here, or with --expert-trace and a folder to count and also record every routing choice to a file for later analysis.',
        how: managedEngine()
          ? 'The Supervisor passes engine arguments it has no setting for straight through: add the flag to the engine arguments in its configuration file' + (configFile ? ' (' + configFile + ')' : '') + ', then restart the engine. Launch details shows the arguments the engine runs with today.'
          : 'This dashboard only watches an engine started elsewhere. Add the flag where that engine is launched, then restart it. Launch details shows the arguments this dashboard knows of.',
        link: true};
    }
    if (kind === 'absent') return {title: 'This engine does not report expert use', text: 'It answered, but has no expert counters. They need a build of the engine that has the --expert-stats option.', link: false};
    if (kind === 'waiting') return {title: 'No request has run yet', text: 'Expert counters are on, but the engine has not routed anything since it started. This page fills in with the first request.', link: false};
    if (kind === 'denied') return {title: 'The engine refused the request', text: 'It answered with an access error. Check the API key under Settings, API and access.', link: false};
    if (kind === 'error') return {title: 'Unexpected answer', text: 'The engine answered, but not with expert counters. Try again; the page keeps asking every 2 seconds.', link: false};
    const why = !connected ? 'The dashboard lost its connection to the Supervisor and is reconnecting.'
      : !managedEngine() && engineState !== 'Running' ? 'The engine is not reachable.'
      : {Stopped: 'The engine is stopped.', Starting: 'The engine is starting; the model is still loading.', Stopping: 'The engine is shutting down.', BackingOff: 'The engine stopped unexpectedly and will be restarted.', Halted: 'Automatic restarts have paused. Check Diagnose.'}[engineState] || 'The engine did not answer.';
    return {title: 'Engine not reachable', text: why + ' Expert use appears here once the engine is ready.', link: false};
  }
  function exSetStale(on) {
    exMap.classList.toggle('stale', on);
    const tag = $('ex-stale');
    tag.hidden = !on;
    if (on) tag.textContent = `Engine not answering · last reading ${formatUptime((Date.now() - ex.at) / 1000)} ago`;
  }
  function exApply(o) {
    const live = $('ex-live'), box = $('ex-state');
    if (o.kind === 'data') {
      const first = !ex.frame || ex.frame.R !== o.frame.R || ex.frame.E !== o.frame.E;
      ex.frame = o.frame; ex.at = Date.now();
      live.hidden = false; box.hidden = true;
      exSetStale(false);
      const total = o.frame.L * o.frame.E, key = total + '|' + o.frame.R;
      if (ex.cfgKey !== key) {
        ex.cfgKey = key; ex.cur = null;
        const gMax = (total * EXPERT_BYTES / GiB).toFixed(1);
        ['ex-slider', 'ex-n'].forEach(id => { $(id).max = total; });
        $('ex-gib').max = gMax;
        $('ex-budget-of').textContent = `of all ${total.toLocaleString()} text-layer experts · ${gMax} GiB`;
        exBudgetSync('');
      }
      exPaintFacts();
      exBudgetUpdate();
      exDrawKey(exKind(o.frame));
      exSetTarget(first);
      exPaintSkew();
      exReadout();
      return;
    }
    if (o.kind === 'down' && ex.frame) { exSetStale(true); return; }
    ex.frame = null; ex.target = ex.shown = null; ex.cur = null;
    live.hidden = true; box.hidden = false;
    const s = exStateFor(o.kind);
    $('ex-state-title').textContent = s.title;
    $('ex-state-text').textContent = s.text;
    $('ex-state-how').textContent = s.how || '';
    $('ex-state-how').hidden = !s.how;
    $('ex-state-link').hidden = !s.link;
  }

  async function exPoll() {
    const mine = ++exSeq, ctl = new AbortController();
    exBusy = true; exAbort = ctl;
    let outcome;
    try {
      const res = await fetch('/api/engine/admin/experts', {headers: {'X-NInfer-Supervisor': '1'}, signal: ctl.signal, cache: 'no-store'});
      const text = await res.text();
      let body = null;
      try { body = JSON.parse(text); } catch (_) {}
      if (res.status === 200 && body && body.object === 'expert_utilization') {
        if (body.enabled === false) outcome = {kind: 'waiting'};
        else { try { outcome = {kind: 'data', frame: exIngest(body)}; } catch (_) { outcome = {kind: 'error'}; } }
      } else if (res.status === 404) {
        const code = body && ((body.error && body.error.code) || body.code);
        outcome = {kind: code === 'expert_stats_disabled' ? 'disabled' : 'absent'};
      } else if (res.status === 401 || res.status === 403) outcome = {kind: 'denied'};
      else if (res.status === 502 || res.status === 503 || res.status === 504) outcome = {kind: 'down'};
      else outcome = {kind: 'error'};
    } catch (err) {
      if (mine !== exSeq) return;
      outcome = {kind: 'down'};
    }
    if (mine !== exSeq) return;
    exBusy = false; exAbort = null;
    exApply(outcome);
    if (exWanted()) exTimer = setTimeout(() => { exTimer = 0; exSync(); }, EX_POLL_MS);
  }
  const exWanted = () => activeView === 'diagnostics' && diagPage === 'experts' && isDocumentVisible;
  function exStop() {
    exSeq++;
    clearTimeout(exTimer); exTimer = 0;
    if (exAbort) exAbort.abort();
    exAbort = null; exBusy = false;
    if (ex.raf) { cancelAnimationFrame(ex.raf); ex.raf = 0; if (ex.target) { ex.shown = ex.target; ex.version++; } }
  }
  function exSync() {
    if (exWanted()) { if (!exTimer && !exBusy) exPoll(); }
    else if (exTimer || exBusy || exAbort) exStop();
  }

  fetch('/api/config').then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); }).then(d => {
)HTML"
R"HTML(    cfgData = d;
    updateEngineControls();
    updateConnectionGuide(d);
    cfgPath.textContent = d.config_path ? d.config_path.split(/[\\/]/).pop() : 'no config file';
    cfgPath.title = d.config_path || '';
    $('kpi-engine-port').textContent = (d.engine && d.engine.engine_port) ? ':' + d.engine.engine_port : '—';
    renderConfig();
    renderSettingsIndex();
    renderBenchAuth();
  }).catch(() => {
    cfgBody.innerHTML = '<p class="cdu-small">Settings could not be loaded. Check that you opened the dashboard on the engine’s computer, then reload the page.</p>';
  });

  window.addEventListener('beforeunload', e => { if (cfgDirtyCount() || reserveDraft != null || feDirty()) { e.preventDefault(); e.returnValue = ''; } });
  setPendingRestart(pendingRestartPid);
  applyDisplay(document.documentElement.dataset.display || 'auto');
  selectView();
  // One stream powers every view; transport failures are visible and recover automatically.
  refreshModelCatalog();
  const es = new EventSource('/api/events');
  es.onmessage = e => {
    try { const s = JSON.parse(e.data); setConnection(true); if (isDocumentVisible) render(s); }
    catch (_) { setConnection(false); }
  };
  es.onerror = () => setConnection(false);
  fetch('/api/state').then(r => { if (!r.ok) throw new Error(); return r.json(); }).then(s => { setConnection(true); render(s); }).catch(() => setConnection(false));
})();
</script>
</body></html>
)HTML";
} // namespace ninfer::supervisor
