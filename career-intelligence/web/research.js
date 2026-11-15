"use strict";
const $=id=>document.getElementById(id),esc=x=>String(x??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&apos;"}[c])),fmt=(v,d=3)=>v==null?"—":Number(v).toFixed(d),clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const state={model:null,draft:null,records:[],selected:null,scenario:null,search:""};
function predict(p){
 if(!state.model||!p)return null;const m=state.model,inputs=[p.min,p.pts,p.gp,p.reb,p.ast,p.stl,p.blk,p.tov,p.fg,p.threePa,p.ft,p.eff,p.age];
 if(inputs.some(x=>!Number.isFinite(Number(x))))return null;
 const z=m.coef.reduce((acc,w,i)=>acc+w*clamp((Number(inputs[i])-m.mean[i])/m.std[i],-m.clipZ,m.clipZ),m.intercept);
 return 1/(1+Math.exp(-clamp(z,-25,25)));
}
function profileList(){
 const q=state.search.toLowerCase(),records=state.records.filter(r=>!q||r.name.toLowerCase().includes(q)).slice(0,100);
 $("res-player-select").innerHTML=records.map((r,i)=>'<option value="'+esc(String(r.id))+'" '+(state.selected?.id===r.id?"selected":"")+'>'+esc(r.name)+" · "+r.year+'</option>').join("")||'<option value="">No matching historical players</option>';
}
function riskGroup(probability){
 const m=state.model;if(!m||probability===null)return null;
 return m.survival.find(s=>probability>=s.pMin&&probability<=s.pMax)||m.survival.reduce((best,s)=>Math.abs((s.pMin+s.pMax)/2-probability)<Math.abs((best.pMin+best.pMax)/2-probability)?s:best,m.survival[0]);
}
function drawSurvival(){
 const prob=predict(state.selected?.rookie),risk=riskGroup(prob);if(!risk){$("res-survival-chart").innerHTML='<text x="390" y="164" fill="#8d8ca1" text-anchor="middle">Historical cohort unavailable</text>';return}
 const W=790,H=325,left=51,right=26,top=25,bottom=49,innerW=W-left-right,innerH=H-top-bottom,ys=v=>top+innerH*(1-v),xs=i=>left+innerW*i/9;
 const ticks=[0,.25,.5,.75,1].map(y=>'<line x1="'+left+'" y1="'+ys(y)+'" x2="'+(W-right)+'" y2="'+ys(y)+'" stroke="#37354b" stroke-width="1"/><text x="'+(left-14)+'" y="'+(ys(y)+5)+'" text-anchor="end" fill="#9993a7" font-size="13">'+(y*100)+'%</text>').join("");
 const coords=risk.curve.map((v,i)=>(i?"L":"M")+xs(i)+" "+ys(v)).join(" "),fill=coords+"L"+xs(9)+" "+ys(0)+"L"+xs(0)+" "+ys(0)+"Z";
 $("res-survival-chart").innerHTML='<defs><linearGradient id="res-fill" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#ffcb29" stop-opacity=".33"/><stop offset="1" stop-color="#ffcb29" stop-opacity="0"/></linearGradient></defs>'+ticks+'<path d="'+fill+'" fill="url(#res-fill)"/><path d="'+coords+'" fill="none" stroke="#ffbe2e" stroke-width="3" stroke-linejoin="round"/>'+risk.curve.map((v,i)=>'<circle cx="'+xs(i)+'" cy="'+ys(v)+'" r="3.5" fill="#ffca34"/>').join("")+
 [0,2,4,6,8,9].map(i=>'<text x="'+xs(i)+'" y="'+(H-20)+'" fill="#aaa7ba" text-anchor="middle" font-size="13">'+(i+1)+'</text>').join("")+
 '<text x="'+(W/2)+'" y="'+(H-1)+'" fill="#7a758c" text-anchor="middle" font-size="11">Career calendar-year span</text>';
 $("res-chart-note").textContent="Observed 1–10-year survival of "+risk.n+" historical players in a similar model risk band. This peer curve is empirical and does not forecast the selected player year by year.";
}
const editable=[
 {key:"pts",label:"Points per game",min:0,max:35,step:.5},
 {key:"ast",label:"Assists per game",min:0,max:12,step:.5},
 {key:"reb",label:"Rebounds per game",min:0,max:17,step:.5},
 {key:"gp",label:"Games played",min:1,max:82,step:1},
 {key:"age",label:"Rookie age",min:18,max:33,step:1}
];
function sliders(){
 const v=state.scenario;if(!v)return;
 $("res-sliders").innerHTML=editable.map(f=>{
 const number=Number(v[f.key]),safe=clamp(number,f.min,f.max);
 return '<label class="res-slider-item"><div><b>'+esc(f.label)+'</b><span id="res-value-'+f.key+'">'+safe+'</span></div><input type="range" data-scenario="'+f.key+'" min="'+f.min+'" max="'+f.max+'" step="'+f.step+'" value="'+safe+'"></label>';
 }).join("");
 $("res-scenario-estimate").textContent=fmt((predict(v)||0)*100,1)+"%";
}
function selectProfile(rec){
 if(!rec)return;state.selected=rec;state.scenario={...rec.rookie};
 $("res-player-info").innerHTML='<strong>'+esc(rec.name)+'</strong><span>Rookie season '+esc(String(rec.year))+' · '+esc(rec.position||"position unavailable")+'</span><div class="res-player-figures"><b>'+fmt(rec.rookie.pts,1)+' PPG</b><b>'+fmt(rec.rookie.reb,1)+' RPG</b><b>'+fmt(rec.rookie.ast,1)+' APG</b></div>';
 $("res-prediction").textContent=fmt((predict(rec.rookie)||0)*100,1)+"%";
 profileList();drawSurvival();sliders();
}
function setup(){
 const m=state.model,d=state.draft;if(!m||!d)return;
 $("res-auc").textContent=fmt(m.metrics.heldOut.auc);
 $("res-brier").textContent=fmt(m.metrics.heldOut.brier);
 $("res-rookies").textContent=m.rowCounts.rookieRecords.toLocaleString("en");
 $("res-test").textContent=m.rowCounts.test;
 $("res-model-auc").textContent=fmt(m.metrics.heldOut.auc);
 $("res-model-brier").textContent=fmt(m.metrics.heldOut.brier);
 $("res-draft-brier").textContent=fmt(d.horizons.five.test.brier,4);
 $("res-ten-brier").textContent=fmt(d.horizons.ten.test.brier,4);
 $("res-model-comparison").innerHTML='<div><span>Historical model test Brier</span><strong>'+fmt(m.metrics.heldOut.brier,4)+'</strong></div><div><span>Prevalence baseline</span><strong>'+fmt(m.metrics.baselineHeldOut.brier,4)+'</strong></div><div><span>Historical ROC-AUC</span><strong>'+fmt(m.metrics.heldOut.auc,4)+'</strong></div><div><span>Test cases</span><strong>'+m.metrics.heldOut.n+'</strong></div>';
 $("res-player-search").addEventListener("input",e=>{state.search=e.target.value.trim();profileList()});
 $("res-player-select").addEventListener("change",e=>{const r=state.records.find(p=>p.id===e.target.value);if(r)selectProfile(r)});
 $("res-sliders").addEventListener("input",e=>{const key=e.target.dataset.scenario;if(!key||!state.scenario)return;state.scenario[key]=Number(e.target.value);$("res-value-"+key).textContent=e.target.value;$("res-scenario-estimate").textContent=fmt((predict(state.scenario)||0)*100,1)+"%"});
 $("res-reset-scenario").addEventListener("click",()=>{if(state.selected){state.scenario={...state.selected.rookie};sliders()}});
 profileList();selectProfile(state.records.find(r=>r.name==="Stephen Curry")||state.records[0]);
}
async function init(){
 try{
 const [m,d,rookies]=await Promise.all(["/data/model.json","/data/draft-path.json","/data/rookies.json"].map(url=>fetch(url).then(r=>{if(!r.ok)throw Error("Source unavailable: "+url);return r.json()})));
 state.model=m;state.draft=d;state.records=rookies.map((r,i)=>({...r,id:String(i)}));setup();
 }catch(e){$("res-player-info").textContent="Historical archive unavailable: "+e.message;}
}
init();
