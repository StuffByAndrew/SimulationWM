"""Researcher-only, self-contained HTML view of one fixture evaluation (data from fixture.evaluate embedded; no network)."""
import hashlib
import json

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fixture evaluation</title><style>
:root{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--pred:#2a78d6;--meas:#eb6834;--band:rgba(11,11,11,.08);--border:rgba(11,11,11,.10);--neutral:#f0efec;--neg:#e34948}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme=light])){color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#f0efec;
--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--pred:#3987e5;--meas:#d95926;--band:rgba(255,255,255,.10);--border:rgba(255,255,255,.10);--neutral:#383835;--neg:#e66767}}
:root[data-theme=dark]{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;
--pred:#3987e5;--meas:#d95926;--band:rgba(255,255,255,.10);--border:rgba(255,255,255,.10);--neutral:#383835;--neg:#e66767}
body{margin:0;padding:16px;background:var(--page);color:var(--ink);font:14px system-ui,-apple-system,"Segoe UI",sans-serif}
h1{font-size:18px;margin:0 0 4px}h2{font-size:15px;margin:18px 0 8px}.sub{color:var(--ink2);font-size:12px}
.row{display:flex;flex-wrap:wrap;gap:12px}.card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:10px;min-width:0;flex:1 1 380px}
.stat{font-size:12px;color:var(--ink2)}.stat b{color:var(--ink);font-size:15px;font-variant-numeric:tabular-nums}
svg{display:block;width:100%;height:auto}svg text{font:11px system-ui;fill:var(--ink2)}.axis{stroke:var(--axis)}.gridl{stroke:var(--grid)}
.legend{display:flex;gap:14px;font-size:12px;color:var(--ink2);margin:4px 0}.sw{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:-2px;margin-right:4px}
button{font:inherit;font-size:12px;color:var(--ink);background:var(--surface);border:1px solid var(--axis);border-radius:5px;padding:4px 9px;cursor:pointer}
button[aria-pressed=true]{border-color:var(--pred);box-shadow:inset 0 0 0 1px var(--pred)}.controls{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:6px}
#tip{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--axis);border-radius:5px;padding:5px 8px;font-size:12px;display:none;z-index:9}
table{border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}td,th{padding:3px 8px;border-bottom:1px solid var(--grid);text-align:right}th:first-child,td:first-child{text-align:left}
</style></head><body><div id="tip"></div>
<h1>Positioning-fixture evaluation</h1><div class="sub">Researcher-only. Stopping positions are ruler readings in mm; dial settings in scale divisions (0 to 10). Predicted values come from the agent's predictor program run in isolation; measured values are repeated releases of the source fixture.</div>
<div id="summary" class="row" style="margin-top:10px"></div>
<h2>Chosen settings per target</h2><div id="chosen" class="row"></div>
<h2>Held-out settings map</h2><div class="row"><div class="card"><div class="controls" id="mapcontrols"></div><svg id="map" viewBox="0 0 420 380"></svg>
<div class="legend" id="maplegend"></div><div class="sub">Each square is a held-out setting; stars are the agent's chosen settings. Click any mark to replay its trials.</div></div>
<div class="card" id="detail"><div class="controls"><b id="dtitle"></b><button id="play">Play</button></div><svg id="dhist" viewBox="0 0 460 210"></svg><svg id="traj" viewBox="0 0 460 260"></svg><svg id="rail" viewBox="0 0 460 70"></svg></div></div>
<h2>Prediction scores</h2><div class="card" id="scores"></div>
<script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),NS='http://www.w3.org/2000/svg',$=id=>document.getElementById(id);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim(),tip=$('tip');
const el=(p,tag,a={},text)=>{const e=document.createElementNS(NS,tag);for(const k in a)e.setAttribute(k,a[k]);if(text!==undefined)e.textContent=text;p.appendChild(e);return e};
const fmt=(v,d=2)=>v===null||v===undefined?'—':(+v).toFixed(d),pct=v=>v===null||v===undefined?'—':(100*v).toFixed(0)+'%';
const hover=(e,html)=>{e.addEventListener('mousemove',ev=>{tip.innerHTML=html;tip.style.display='block';tip.style.left=ev.clientX+12+'px';tip.style.top=ev.clientY+12+'px'});e.addEventListener('mouseleave',()=>tip.style.display='none')};
const hits=(xs,t)=>xs.length?xs.filter(x=>x>=t.lo_mm&&x<=t.hi_mm).length/xs.length:null;
const stops=s=>s.trials.map(t=>t.stop_mm);
const settings=[...D.grid.map((s,i)=>({...s,label:'held-out #'+(i+1)})),...Object.entries(D.chosen).map(([k,s])=>({...s,label:'chosen for '+k,chosen:k}))];
function axes(svg,x0,x1,y0,y1,L,R,T,B,W,H,xl,yl){const sx=v=>L+(v-x0)/(x1-x0)*(W-L-R),sy=v=>H-B-(v-y0)/(y1-y0)*(H-T-B);
 for(const v of ticks(x0,x1)){el(svg,'line',{x1:sx(v),x2:sx(v),y1:T,y2:H-B,class:'gridl'});el(svg,'text',{x:sx(v),y:H-B+13,'text-anchor':'middle'},v)}
 for(const v of ticks(y0,y1)){el(svg,'line',{x1:L,x2:W-R,y1:sy(v),y2:sy(v),class:'gridl'});el(svg,'text',{x:L-4,y:sy(v)+4,'text-anchor':'end'},v)}
 el(svg,'line',{x1:L,x2:W-R,y1:H-B,y2:H-B,class:'axis'});el(svg,'text',{x:(L+W-R)/2,y:H-2,'text-anchor':'middle'},xl);
 el(svg,'text',{x:12,y:(T+H-B)/2,'text-anchor':'middle',transform:`rotate(-90 12 ${(T+H-B)/2})`},yl);return [sx,sy]}
function ticks(a,b){const s=Math.pow(10,Math.floor(Math.log10((b-a)/4))),m=(b-a)/s>20?5:(b-a)/s>8?2:1,st=s*m,o=[];for(let v=Math.ceil(a/st)*st;v<=b+1e-9;v+=st)o.push(+v.toFixed(6));return o}
function histogram(svg,s,targets){svg.replaceChildren();const meas=stops(s),pred=s.predicted_mm||[],all=meas.concat(pred).concat(targets.flatMap(t=>[t.lo_mm,t.hi_mm]));
 let lo=Math.min(...all),hi=Math.max(...all);const pad=Math.max(2,(hi-lo)*.05);lo-=pad;hi+=pad;const nb=40,w=(hi-lo)/nb,bin=xs=>{const c=new Array(nb).fill(0);for(const x of xs)c[Math.min(nb-1,Math.max(0,Math.floor((x-lo)/w)))]+=1/xs.length;return c};
 const cm=bin(meas),cp=pred.length?bin(pred):null,top=Math.max(...cm,...(cp||[0]))*1.1||1,[sx,sy]=axes(svg,lo,hi,0,top,44,10,10,30,460,210,'stopping position (ruler mm)','fraction');
 for(const t of targets){const r=el(svg,'rect',{x:sx(t.lo_mm),y:10,width:Math.max(1.5,sx(t.hi_mm)-sx(t.lo_mm)),height:170,fill:'var(--band)',stroke:'var(--axis)','stroke-dasharray':'3 2'});hover(r,`${t.name}: ${t.lo_mm}–${t.hi_mm} mm`);el(svg,'text',{x:sx(t.lo_mm)+2,y:20},t.name.replace('_',' '))}
 cm.forEach((c,i)=>{if(!c)return;const r=el(svg,'rect',{x:sx(lo+i*w)+1,y:sy(c),width:Math.max(1,sx(lo+w)-sx(lo)-2),height:sy(0)-sy(c),fill:'var(--meas)',rx:1});hover(r,`${fmt(lo+i*w,1)}–${fmt(lo+(i+1)*w,1)} mm<br>measured ${pct(c)}`)});
 if(cp){let d='';cp.forEach((c,i)=>{d+=`${i?'L':'M'}${sx(lo+i*w)},${sy(c)}L${sx(lo+(i+1)*w)},${sy(c)}`});el(svg,'path',{d,fill:'none',stroke:'var(--pred)','stroke-width':2})}}
function legend(){return `<div class="legend"><span><span class="sw" style="background:var(--meas)"></span>measured releases</span><span><span class="sw" style="background:var(--pred)"></span>predicted distribution</span><span><span class="sw" style="background:var(--band);border:1px dashed var(--axis)"></span>target region</span></div>`}
// Summary and chosen-setting panels
const P=D.scores.prediction;$('summary').innerHTML=`<div class="card stat">Prediction CRPS (held-out settings)<br><b>${fmt(P.crps_mm)} mm</b> · oracle ${fmt(P.oracle_crps_mm)} · mean-only ${fmt(P.mean_only_crps_mm)} · settings-blind ${fmt(P.pooled_crps_mm)}${P.error?'<br>'+P.error:''}</div>`+
 D.targets.map(t=>{const s=D.scores.targets[t.name];return `<div class="card stat">${t.name.replace('_',' ')} (${t.lo_mm}–${t.hi_mm} mm)<br><b>${s.hits===null?'—':s.hits+'/'+s.trials} hits</b> measured ${pct(s.measured_hit_fraction)} · predicted ${pct(s.predicted_hit_fraction)}${s.error?'<br>'+s.error:''}</div>`}).join('');
for(const t of D.targets){const s=D.chosen[t.name],sc=D.scores.targets[t.name],card=document.createElement('div');card.className='card';$('chosen').append(card);
 if(!s){card.innerHTML=`<b>${t.name}</b><div class="sub">${sc.error||'no chosen setting'}</div>`;continue}
 card.innerHTML=`<b>${t.name.replace('_',' ')}</b> <span class="sub">compression ${fmt(s.compression)}, pressure ${fmt(s.pressure)}</span>${legend()}`;
 const svg=el(card,'svg',{viewBox:'0 0 460 210'});histogram(svg,s,[t]);const bars=el(card,'svg',{viewBox:'0 0 460 64'});
 [['predicted hit rate',sc.predicted_hit_fraction,'var(--pred)',null],['measured hit rate',sc.measured_hit_fraction,'var(--meas)',sc.wilson95]].forEach(([n,v,c,ci],i)=>{const y=6+i*28;
  el(bars,'text',{x:0,y:y+13},n);el(bars,'rect',{x:130,y,width:300,height:18,fill:'var(--band)',rx:3});if(v!==null)el(bars,'rect',{x:130,y,width:Math.max(2,300*v),height:18,fill:c,rx:3});
  if(ci)el(bars,'line',{x1:130+300*ci[0],x2:130+300*ci[1],y1:y+9,y2:y+9,stroke:'var(--ink)','stroke-width':1.5});el(bars,'text',{x:434,y:y+13},pct(v))})}
// Settings map
let mode='measured',target=D.targets[0].name,selected=settings[0];const modes={predicted:'Predicted success',measured:'Measured success',difference:'Predicted − measured'};
const ramp=['#cde2fb','#9ec5f4','#6da7ec','#3987e5','#256abf','#184f95','#0d366b'],mix=(a,b,f)=>{const p=h=>[1,3,5].map(i=>parseInt(h.slice(i,i+2),16)),x=p(a),y=p(b);return 'rgb('+x.map((v,i)=>Math.round(v+(y[i]-v)*f)).join(',')+')'};
const seq=v=>{const f=v*(ramp.length-1),i=Math.min(ramp.length-2,Math.floor(f));return mix(ramp[i],ramp[i+1],f-i)};
const div=v=>{const mid=css('--neutral')||'#f0efec';return v>=0?mix(mid,'#e34948',Math.min(1,v)):mix(mid,'#2a78d6',Math.min(1,-v))};
const value=(s,t)=>{const p=s.predicted_mm?hits(s.predicted_mm,t):null,m=hits(stops(s),t);return {p,m,d:p===null?null:p-m}};
function controls(){const c=$('mapcontrols');c.replaceChildren();for(const k in modes){const b=document.createElement('button');b.textContent=modes[k];b.setAttribute('aria-pressed',mode===k);b.onclick=()=>{mode=k;controls();map()};c.append(b)}
 const sel=document.createElement('select');for(const t of D.targets){const o=document.createElement('option');o.value=t.name;o.textContent=`${t.name} (${t.lo_mm}–${t.hi_mm} mm)`;sel.append(o)}sel.value=target;sel.onchange=()=>{target=sel.value;map()};c.append(sel)}
function map(){const svg=$('map');svg.replaceChildren();const t=D.targets.find(x=>x.name===target),[sx,sy]=axes(svg,0,10,0,10,40,14,10,34,420,380,'compression dial','pressure dial');
 for(const s of settings){const v=value(s,t),q=mode==='predicted'?v.p:mode==='measured'?v.m:v.d,fill=q===null?'var(--band)':mode==='difference'?div(q):seq(q),x=sx(s.compression),y=sy(s.pressure);
  const e=s.chosen?el(svg,'path',{d:star(x,y,13),fill,stroke:'var(--ink)','stroke-width':1.2}):el(svg,'rect',{x:x-14,y:y-14,width:28,height:28,rx:4,fill,stroke:s===selected?'var(--ink)':'var(--surface)','stroke-width':s===selected?2:2});
  e.style.cursor='pointer';e.onclick=()=>{selected=s;map();detail()};hover(e,`${s.label}<br>compression ${fmt(s.compression)}, pressure ${fmt(s.pressure)}<br>predicted ${pct(v.p)} · measured ${pct(v.m)} (${s.trials.length} releases)`)}
 $('maplegend').innerHTML=mode==='difference'?'<span><span class="sw" style="background:#2a78d6"></span>predicted lower</span><span><span class="sw" style="background:var(--neutral)"></span>equal</span><span><span class="sw" style="background:#e34948"></span>predicted higher</span>':
  `<span>0%</span>${ramp.map(c=>`<span class="sw" style="background:${c};margin:0"></span>`).join('')}<span>100% of releases in ${target.replace('_',' ')}</span>`}
const star=(x,y,r)=>{let d='';for(let i=0;i<10;i++){const a=-Math.PI/2+i*Math.PI/5,q=i%2?r*.45:r;d+=(i?'L':'M')+(x+q*Math.cos(a))+','+(y+q*Math.sin(a))}return d+'Z'};
// Selected setting: histogram, overlaid carriage trajectories, replay
let timer=null,now=0;
function detail(){clearInterval(timer);timer=null;$('play').textContent='Play';now=0;$('dtitle').textContent=`${selected.label}: compression ${fmt(selected.compression)}, pressure ${fmt(selected.pressure)}`;
 histogram($('dhist'),selected,D.targets);const svg=$('traj');svg.replaceChildren();const tr=selected.trials,tmax=Math.max(...tr.map(t=>t.t_s[t.t_s.length-1])),xs=tr.flatMap(t=>t.x_mm);
 const [sx,sy]=axes(svg,0,tmax,Math.min(...xs,0),Math.max(...xs,...D.targets.map(t=>t.hi_mm))+5,44,10,10,30,460,260,'time since release (s)','carriage index (mm)');
 for(const t of D.targets)el(svg,'rect',{x:44,y:sy(t.hi_mm),width:406,height:Math.max(1.5,sy(t.lo_mm)-sy(t.hi_mm)),fill:'var(--band)'});
 for(const t of tr)el(svg,'path',{d:t.t_s.map((v,i)=>(i?'L':'M')+sx(v)+','+sy(t.x_mm[i])).join(''),fill:'none',stroke:'var(--meas)','stroke-width':1.2,'stroke-opacity':.55});
 const cursor=el(svg,'line',{x1:44,x2:44,y1:10,y2:230,stroke:'var(--ink)','stroke-dasharray':'4 3'});
 const rail=$('rail');rail.replaceChildren();const lo=Math.min(...xs,-50),hi=Math.max(...xs,250),rx=v=>10+(v-lo)/(hi-lo)*440;el(rail,'line',{x1:10,x2:450,y1:40,y2:40,stroke:'var(--axis)','stroke-width':4});
 for(const t of D.targets)el(rail,'rect',{x:rx(t.lo_mm),y:30,width:Math.max(1.5,rx(t.hi_mm)-rx(t.lo_mm)),height:20,fill:'var(--band)',stroke:'var(--axis)'});
 for(let v=Math.ceil(lo/50)*50;v<=hi;v+=50)el(rail,'text',{x:rx(v),y:66,'text-anchor':'middle'},v);
 const dots=tr.map(()=>el(rail,'circle',{r:4,cy:40,fill:'var(--meas)','fill-opacity':.6,stroke:'var(--surface)','stroke-width':1}));
 const draw=()=>{cursor.setAttribute('x1',sx(now));cursor.setAttribute('x2',sx(now));tr.forEach((t,i)=>{let k=t.t_s.findIndex(v=>v>now);k=k<0?t.t_s.length-1:Math.max(0,k-1);dots[i].setAttribute('cx',rx(t.x_mm[k]))})};draw();
 $('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='Play';return}if(now>=tmax)now=0;$('play').textContent='Pause';const start=performance.now()-now*1000;
  timer=setInterval(()=>{now=Math.min(tmax,(performance.now()-start)/1000);draw();if(now>=tmax){clearInterval(timer);timer=null;$('play').textContent='Replay'}},30)}}
// Scores table
$('scores').innerHTML=`<table><tr><th>target</th><th>Brier predicted</th><th>Brier mean-only</th><th>Brier oracle</th><th>best held-out hit rate</th><th>hit rate at mean-nearest held-out setting</th></tr>`+
 D.targets.map(t=>{const b=P.brier[t.name],s=D.scores.targets[t.name];return `<tr><td>${t.name}</td><td>${fmt(b.predicted,3)}</td><td>${fmt(b.mean_only,3)}</td><td>${fmt(b.oracle,3)}</td><td>${pct(s.grid_best_hit_fraction)}</td><td>${pct(s.mean_choice_hit_fraction)}</td></tr>`}).join('')+
 `</table><div class="sub" style="margin-top:6px">CRPS skill vs settings-blind: ${fmt(P.crps_skill_vs_pooled,3)}. ${P.failed_queries} of ${P.queries} held-out queries failed.</div>`;
controls();map();detail();
</script></body></html>"""


def write_view(data, path):
    """Write the view with `data` embedded; returns its sha256."""
    page = PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")).encode()
    path.write_bytes(page)
    return hashlib.sha256(page).hexdigest()
