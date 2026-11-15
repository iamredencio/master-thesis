"use strict";
const $=id=>document.getElementById(id);
const esc=x=>String(x??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const clamp=(n,a,b)=>Math.max(a,Math.min(b,n));
const fmt=(x,d=1)=>x==null||!Number.isFinite(Number(x))?"—":Number(x).toFixed(d);
const upper=x=>String(x??"").toUpperCase();
const skills=["spacing","creation","defense","rebounding","ballsecurity"];
const skillNames={spacing:"Three-point spacing",creation:"Scoring & playmaking",defense:"Defensive activity",rebounding:"Rebounding",ballsecurity:"Ball security"};
const skillDesc={spacing:"Shooting accuracy and three-point volume",creation:"Self-created shots and passing",defense:"Steals, blocks, perimeter and rim activity",rebounding:"Rebounding production and role",ballsecurity:"Protecting possessions and making passes"};
const featured=[
{id:"1966",name:"LeBron James",league:"nba",headshot:"https://a.espncdn.com/i/headshots/nba/players/full/1966.png",teamName:"NBA player · team lookup",position:"F",teamId:""},
{id:"3975",name:"Stephen Curry",league:"nba",headshot:"https://a.espncdn.com/i/headshots/nba/players/full/3975.png",teamName:"NBA player",position:"G",teamId:"9"},
{id:"3202",name:"Kevin Durant",league:"nba",headshot:"https://a.espncdn.com/i/headshots/nba/players/full/3202.png",teamName:"NBA player",position:"F",teamId:""},
{id:"3059318",name:"Joel Embiid",league:"nba",headshot:"https://a.espncdn.com/i/headshots/nba/players/full/3059318.png",teamName:"NBA player",position:"C",teamId:""},
{id:"3032977",name:"Giannis Antetokounmpo",league:"nba",headshot:"https://a.espncdn.com/i/headshots/nba/players/full/3032977.png",teamName:"NBA player",position:"F",teamId:""}
];
const S={section:"players",league:"nba",playerTab:"overview",metric:"points",range:"3",selected:featured[0],selectedKey:"nba:1966",
teams:{nba:[],college:[]},rosters:{nba:new Map(),college:new Map()},historical:[],model:null,draft:null,
teamId:"9",collegeId:"150",candidateId:"150",draftPlayer:null,draftPick:15,
stats:new Map(),games:new Map(),history:new Map(),teamStats:new Map(),
needs:{spacing:50,creation:50,defense:50,rebounding:50,ballsecurity:50},ratings:{},sources:{},shortlist:[],loadingTeams:false};
function photo(p,cls=""){const img=p?.headshot;const valid=typeof img==="string"&&/^https:\/\/[^/]*\.(?:espncdn|espn)\.com\//i.test(img);
 return '<div class="dash-avatar '+cls+'"><span>'+esc((p?.name||"PL").split(/\s+/).map(x=>x[0]).slice(0,2).join("").toUpperCase())+'</span>'+(valid?'<img src="'+esc(img)+'" alt="" loading="lazy" onerror="this.remove()">':"")+'</div>';
}
function playerKey(p){return p.league+":"+p.id}
async function endpoint(type,league,id){const target="/api/scout?type="+encodeURIComponent(type)+"&league="+encodeURIComponent(league)+(id?"&"+(["player","games","history"].includes(type)?"id":"team")+"="+encodeURIComponent(id):"");
 const response=await fetch(target,{headers:{Accept:"application/json"}});const j=await response.json().catch(()=>({}));
 if(!response.ok||!j.ok)throw Error(j.reason||j.error||"Data feed unavailable");return j;
}
async function jsonFile(path){const r=await fetch(path);if(!r.ok)throw Error("Cannot load "+path);return r.json()}
function statOf(p){return S.stats.get(playerKey(p))?.stats||null}
function gameOf(p){return S.games.get(playerKey(p))?.games||[]}
function historyOf(p){return S.history.get(playerKey(p))?.history||[]}
function teamName(id,league="nba"){return S.teams[league].find(t=>t.id===String(id))?.name||""}
function teamAbbrev(id,league="nba"){return S.teams[league].find(t=>t.id===String(id))?.abbreviation||""}
function fromRoster(raw,league,team){return{...raw,league,teamId:String(team),teamName:teamName(team,league)||raw.teamName||"ESPN roster",year:2027}}
function getByKey(key){if(!key)return null;const [league,id]=key.split(":");
 if(league==="historical")return S.historical.find(p=>p.id===id)||null;
 for(const roster of S.rosters[league]?.values()||[]){const match=roster.find(p=>p.id===id);if(match)return match}
 return featured.find(p=>p.id===id&&league==="nba")||null;
}
function rosterOf(league,id){return S.rosters[league].get(id)||[]}
function getPool(league){
 if(league==="historical")return S.historical;
 if(league==="college")return rosterOf("college",S.collegeId);
 const teamSelect=$("players-team")?.value||"all",all=new Map();
 if(teamSelect==="all")featured.forEach(p=>all.set(playerKey(p),p));
 if(teamSelect==="all"){for(const players of S.rosters.nba.values())players.forEach(p=>all.set(playerKey(p),p));}
 else rosterOf("nba",teamSelect).forEach(p=>all.set(playerKey(p),p));
 return [...all.values()];
}
function cached(p){return S.stats.has(playerKey(p))}
async function loadRoster(league,id,force=false){
 if(!id)return[];if(!force&&S.rosters[league].has(id))return rosterOf(league,id);
 const response=await endpoint("roster",league,id);
 const rows=response.players.map(p=>fromRoster(p,league,id));
 S.rosters[league].set(id,rows);
 if(league==="college"&&id===S.collegeId){if(!S.draftPlayer||!rows.some(x=>x.id===S.draftPlayer.id))S.draftPlayer=rows[0]||null;renderDraftPlayers()}
 if(league==="nba"){renderTeamRoster();if(id===S.teamId)void prefetchRosterStats(league,id,7)}
 renderPlayerStrip();return rows;
}
async function fetchStats(p){
 if(!p||p.league==="historical")return null;
 const key=playerKey(p);if(S.stats.has(key))return S.stats.get(key);
 try{const j=await endpoint("player",p.league,p.id);S.stats.set(key,j);return j}catch(e){S.stats.set(key,null);return null}
}
async function fetchGames(p){
 if(!p||p.league==="historical")return null;
 const key=playerKey(p);if(S.games.has(key))return S.games.get(key);
 try{const j=await endpoint("games",p.league,p.id);S.games.set(key,j);return j}catch(e){S.games.set(key,null);return null}
}
async function fetchHistory(p){
 if(!p||p.league==="historical")return null;
 const key=playerKey(p);if(S.history.has(key))return S.history.get(key);
 try{const j=await endpoint("history",p.league,p.id);S.history.set(key,j);const latest=j.history?.at(-1);if(p.league==="nba"&&latest?.year===2027&&latest.teamId){const known=teamName(latest.teamId,"nba");const fp=featured.find(x=>x.id===p.id);if(fp){fp.teamId=latest.teamId;fp.teamName=known||"NBA team"}if(S.selectedKey===key){S.selected.teamId=latest.teamId;S.selected.teamName=known||"NBA team"}}return j}catch(e){S.history.set(key,null);return null}
}
function persist(){try{sessionStorage.setItem("hoop-metrics-scout-v1",JSON.stringify({needs:S.needs,ratings:S.ratings,sources:S.sources,shortlist:S.shortlist}))}catch(e){}}
function restore(){try{const v=JSON.parse(sessionStorage.getItem("hoop-metrics-scout-v1")||"{}");if(v.needs)for(const k of skills)if(Number.isFinite(+v.needs[k]))S.needs[k]=clamp(+v.needs[k],0,100);if(v.ratings&&typeof v.ratings==="object")S.ratings=v.ratings;if(v.sources&&typeof v.sources==="object")S.sources=v.sources;if(Array.isArray(v.shortlist))S.shortlist=v.shortlist.slice(0,100)}catch(e){}}
function nav(section){if(!["players","team","prospects"].includes(section))section="players";S.section=section;
 document.querySelectorAll(".dash-section").forEach(x=>x.classList.toggle("show",x.id==="section-"+section));
 document.querySelectorAll("[data-section]").forEach(x=>x.classList.toggle("on",x.dataset.section===section));
 history.replaceState(null,"","#"+section);window.scrollTo({top:0,behavior:"smooth"});
 if(section==="team"){renderTeam();ensureTeamStats(S.teamId)}
 if(section==="prospects")renderDraft();
}
function choices(league,active,withFeatured=false){
 let list=S.teams[league],html=withFeatured?'<option value="all">Featured & loaded NBA teams</option>':"";
 html+=list.map(t=>'<option value="'+esc(t.id)+'" '+(t.id===active?"selected":"")+'>'+esc(t.name)+'</option>').join("");
 return html||'<option>No teams available</option>';
}
function setupTeams(){
 $("team-franchise").innerHTML=choices("nba",S.teamId);
 $("players-team").innerHTML=S.league==="historical"?'<option value="all">All archived years</option>':choices(S.league,S.league==="college"?S.collegeId:"all",S.league==="nba");
 $("draft-school").innerHTML=choices("college",S.collegeId);
 $("candidate-school").innerHTML=choices("college",S.candidateId);
}
function changeLeague(league){
 S.league=league;$("players-search").value="";$("players-min-pts").value=0;$("players-min-ast").value=0;$("players-min-reb").value=0;
 document.querySelectorAll("[data-league]").forEach(x=>x.classList.toggle("on",x.dataset.league===league));
 setupTeams();S.selected=null;S.selectedKey="";if(league==="college")loadRoster("college",S.collegeId).catch(()=>renderPlayerStrip());else renderPlayerStrip();
}
function filtered(){
 let list=getPool(S.league),q=$("players-search").value.trim().toLowerCase(),position=$("players-position").value;
 const pts=Math.max(0,Number($("players-min-pts").value)||0),ast=Math.max(0,Number($("players-min-ast").value)||0),reb=Math.max(0,Number($("players-min-reb").value)||0);
 list=list.filter(p=>{
  if(q&&!(p.name+" "+(p.teamName||"")+" "+(p.position||"")).toLowerCase().includes(q))return false;
  if(position!=="all"&&!upper(p.position).includes(position))return false;
  const stats=statOf(p),rookie=p.rookie;
  if(pts&&!(Number(stats?.avgPoints??rookie?.pts)>=pts))return false;
  if(ast&&!(Number(stats?.avgAssists??rookie?.ast)>=ast))return false;
  if(reb&&!(Number(stats?.avgRebounds??rookie?.reb)>=reb))return false;
  return true;
 });
 const sort=$("players-sort").value;return list.sort((a,b)=>{
  if(sort==="name"){if(S.league==="nba"&&!q){const ai=featured.findIndex(x=>x.id===a.id),bi=featured.findIndex(x=>x.id===b.id);if(ai>=0&&bi>=0)return ai-bi;if(ai>=0)return-1;if(bi>=0)return 1;}return a.name.localeCompare(b.name)}
  if(sort==="team")return (a.teamName||"").localeCompare(b.teamName||"")||a.name.localeCompare(b.name);
  const map={pts:"avgPoints",ast:"avgAssists",reb:"avgRebounds"},rookie={pts:"pts",ast:"ast",reb:"reb"},key=map[sort];
  return Number(statOf(b)?.[key]??statOf(b)?.stats?.[key]??b.rookie?.[rookie[sort]]??-1)-Number(statOf(a)?.[key]??statOf(a)?.stats?.[key]??a.rookie?.[rookie[sort]]??-1);
 });
}
function renderPlayerStrip(){
 const list=filtered(),selected=playerKey(S.selected||{})||"";
 const max=S.league==="historical"?45:65;
 $("player-strip").innerHTML=list.length?list.slice(0,max).map(p=>{
  const team=teamAbbrev(p.teamId,p.league)||(p.league==="historical"?String(p.year):"NBA");
  return '<button type="button" role="listitem" data-player="'+esc(playerKey(p))+'" class="dash-player-chip '+(playerKey(p)===selected?"active":"")+'">'+photo(p)+'<span><strong>'+esc(p.name)+'</strong><small><span class="dash-team-tag">'+esc(team)+'</span>'+esc(p.jersey?"#"+p.jersey:p.position||"PLAYER")+'</small></span></button>';
 }).join(""):'<div class="dash-no-results">No players match the current filters. For statistical filters, open a player profile to load the observed season values first.</div>';
 const pool=getPool(S.league);$("player-filter-status").textContent=list.length+" shown from "+pool.length+" loaded profiles"+(S.league==="historical"?" · historic archive":" · ESPN");
 if(list.length&&(!S.selected||!list.some(p=>playerKey(p)===S.selectedKey)))selectPlayer(list[0]);
 else if(!list.length){S.selected=null;S.selectedKey="";renderEmptyPlayer()}
}
function renderEmptyPlayer(){
 $("player-identity").innerHTML='<div class="dash-empty">Select a player with available data to inspect their profile.</div>';
 $("player-impact").innerHTML='<h3>Advanced Metrics</h3><div class="dash-empty">Select a player.</div>';
 $("stat-grid").innerHTML='<div class="dash-skeleton">No player selected.</div>';
 $("performance-svg").innerHTML='<text x="450" y="150" fill="#9c9db0" text-anchor="middle">No performance data</text>';
 $("player-team-analysis").innerHTML='<h2>Team Analysis</h2><div class="dash-empty">Select a player to begin.</div>';
}
async function selectPlayer(p){
 if(!p)return;S.selected=p;S.selectedKey=playerKey(p);renderPlayerStrip();renderPlayer();
 const key=playerKey(p);
 if(p.league!=="historical"){
  const requests=[fetchStats(p),fetchGames(p),fetchHistory(p)];
  await Promise.allSettled(requests);
  if(S.selectedKey===key){const actual=getByKey(key);if(actual)S.selected=actual;renderPlayer();renderPlayerStrip();renderTeamRoster();}
 }
}
function ring(value,label,max,color="#ffbb2d"){
 const exists=value!==null&&value!==undefined&&Number.isFinite(Number(value)),progress=exists?clamp(Math.round(Number(value)/max*100),0,100):0;
 return '<div class="dash-ring" style="--ring-progress:'+progress+';--ring-color:'+color+'"><div class="dash-ring-inner"><strong class="dash-ring-value">'+(exists?fmt(value):"—")+'</strong><span class="dash-ring-label">'+esc(label)+'</span></div></div>';
}
function rookieProbability(p){
 if(!p?.rookie||!S.model)return null;
 const m=S.model,x=[p.rookie.min,p.rookie.pts,p.rookie.gp,p.rookie.reb,p.rookie.ast,p.rookie.stl,p.rookie.blk,p.rookie.tov,p.rookie.fg,p.rookie.threePa,p.rookie.ft,p.rookie.eff,p.rookie.age];
 const z=m.coef.reduce((s,c,i)=>s+c*clamp((x[i]-m.mean[i])/m.std[i],-m.clipZ,m.clipZ),m.intercept);
 return 1/(1+Math.exp(-clamp(z,-25,25)));
}
function analysisMetrics(p){const s=statOf(p);if(!s)return{};
 const made=s.avgFieldGoalsMade,attempt=s.avgFieldGoalsAttempted,three=s.avgThreePointFieldGoalsMade,fta=s.avgFreeThrowsAttempted,pts=s.avgPoints;
 return{efg:attempt>0&&made!=null&&three!=null?(made+.5*three)/attempt*100:null,
 ts:attempt>0&&fta!=null&&pts!=null?pts/(2*(attempt+.44*fta))*100:null,
 at:s.avgAssists!=null&&s.avgTurnovers>0?s.avgAssists/s.avgTurnovers:null};
}
function renderPlayer(){
 const p=S.selected;if(!p){renderEmptyPlayer();return}
 const s=statOf(p),r=p.rookie||{},m=analysisMetrics(p),gp=s?.gamesPlayed??r.gp,pts=s?.avgPoints??r.pts,reb=s?.avgRebounds??r.reb,ast=s?.avgAssists??r.ast;
 const team=teamAbbrev(p.teamId,p.league)||(p.league==="historical"?"ARCHIVE":p.league.toUpperCase());
 $("player-identity").innerHTML='<div class="dash-player-id-head">'+photo(p)+'<div><h2>'+esc(p.name)+'</h2><span class="dash-team-tag">'+esc(team)+'</span><span class="dash-jersey">'+esc(p.jersey?"#"+p.jersey:p.league==="historical"?String(p.year):"")+'</span><p>'+esc([p.position,p.height,p.weight].filter(Boolean).join(" · ")||"Historical rookie profile")+'</p></div></div>'+
 '<div class="dash-ring-grid">'+ring(pts,"PTS",40)+ring(reb,"REB",16)+ring(ast,"AST",13)+'</div>'+
 '<div class="dash-identity-facts"><div><span>Games Played</span><strong>'+fmt(gp,0)+'</strong></div><div><span>'+(p.league==="college"?"College":p.league==="historical"?"Rookie year":"Current Team")+'</span><strong>'+esc(p.league==="historical"?String(p.year):p.teamName||"—")+'</strong></div></div>';
 const active=p.league!=="historical"&&!!s;
 const adv=active?'<div class="dash-impact-section"><div class="dash-impact-title">Advanced Metrics</div><div class="dash-ring-grid">'+ring(m.efg,"eFG%",100)+ring(m.ts,"TS%",100)+ring(m.at,"AST/TO",5)+'</div></div><div class="dash-impact-section"><div class="dash-impact-title">Defensive Impact</div><div class="dash-ring-grid two">'+ring(s.avgBlocks,"BLK",4,"#733db1")+ring(s.avgSteals,"STL",4,"#733db1")+'</div></div><div class="dash-data-warning">Advanced values eFG%, TS% and AST/TO are calculated from observed ESPN season averages. PER and usage rate need additional verified inputs and are intentionally not invented.</div>':
 p.league==="historical"?'<div class="dash-impact-section"><div class="dash-impact-title">Archived Rookie Production</div><div class="dash-ring-grid">'+ring(r.stl,"STL",4)+ring(r.blk,"BLK",4)+ring(r.eff,"EFF",25)+'</div></div><div class="dash-data-warning">Archive statistics reflect '+esc(String(p.year))+' rookie season, not current performance.</div>':'<p class="dash-muted">Season averages will appear when verified ESPN data is available for this player.</p>';
 $("player-impact").innerHTML=adv;
 renderChart();renderStatGrid();renderPlayerAnalysis();
}
function renderStatGrid(){
 const p=S.selected;if(!p)return;const s=statOf(p)||{},r=p.rookie||{},m=analysisMetrics(p);
 const cells=S.playerTab==="advanced"?[
 ["eFG%",m.efg,"pct"],["TS%",m.ts,"pct"],["AST/TO",m.at,"ratio"],
 ["STL",s.avgSteals??r.stl,"one"],["BLK",s.avgBlocks??r.blk,"one"],["TOV",s.avgTurnovers??r.tov,"one"]]:
 [["PPG",s.avgPoints??r.pts,"one"],["RPG",s.avgRebounds??r.reb,"one"],["APG",s.avgAssists??r.ast,"one"],
 ["FG%",s.fieldGoalPct??r.fg,"one"],["3P%",s.threePointFieldGoalPct,"one"],["MPG",s.avgMinutes??r.min,"one"]];
 if(S.playerTab==="highlights"){
  const games=gameOf(p),high=games.length?Math.max(...games.map(g=>g.points??0)):null;
  const all=[["Season High Points",high===null?null:high,"Recorded single-game high"],["Season Average",s.avgPoints??r.pts,"Points per game"],
   ["Effective FG%",m.efg,"Calculated shooting efficiency"],["Games Played",s.gamesPlayed??r.gp,"Current-season appearances"]];
  $("stat-grid").innerHTML='<div class="dash-highlights-grid">'+all.map(([name,val,desc])=>'<div class="dash-panel dash-highlight-tile"><span>☆</span><div><h3>'+esc(name)+'</h3><small>'+esc(desc)+'</small><strong>'+fmt(val,name==="Games Played"||name==="Season High Points"?0:1)+(name==="Effective FG%"&&val!=null?"%":"")+'</strong></div></div>').join("")+'</div>';return;
 }
 $("stat-grid").innerHTML=cells.map(([title,value,format])=>'<div class="dash-stat-cell"><span>'+title+'</span><strong>'+fmt(value,format==="pct"?1:format==="ratio"?2:1)+(format==="pct"&&value!=null?"%":"")+'</strong></div>').join("");
}
function dateLabel(iso,short=false){const d=new Date(iso);return Number.isFinite(+d)?d.toLocaleDateString("en-US",short?{month:"short",day:"numeric"}:{month:"short",day:"numeric",year:"numeric"}):""}
function chartGames(p){
 const games=gameOf(p);if(S.range==="all")return games;
 const now=Date.now(),months=Number(S.range);const cutoff=new Date(now);cutoff.setMonth(cutoff.getMonth()-months);
 return games.filter(g=>Date.parse(g.date)>=cutoff.getTime());
}
function renderChart(){
 const p=S.selected;if(!p)return;
 $("chart-heading").textContent=S.playerTab==="highlights"?"Performance Highlights":"Season Performance";
 $("chart-subhead").textContent=p.league==="historical"?"The thesis archive contains season aggregates only":"Recorded 2026–27 regular-season games";
 document.querySelectorAll("[data-metric]").forEach(x=>x.classList.toggle("on",x.dataset.metric===S.metric));
 document.querySelectorAll("[data-range]").forEach(x=>x.classList.toggle("on",x.dataset.range===S.range));
 const games=chartGames(p),field=S.metric;
 if(!games.length){
  $("performance-svg").innerHTML='<text x="450" y="145" fill="#a9a8bd" font-size="23" text-anchor="middle">No recorded games for this period</text><text x="450" y="184" fill="#6e7186" font-size="15" text-anchor="middle">'+(p.league==="historical"?"Archived rookie seasons have no per-game feed":"Select ALL or another player to check available game logs")+'</text>';
  $("chart-caption").textContent=p.league==="historical"?"Historical rookie data · no real game logs":"Only observed ESPN games are plotted · no interpolation or fabricated results";return;
 }
 const vals=games.map(g=>Number(g[field])||0),top=Math.max(5,Math.ceil(Math.max(...vals)/5)*5),W=900,H=330,left=50,right=22,up=24,bottom=44,cw=W-left-right,ch=H-up-bottom;
 const coords=vals.map((v,i)=>({x:left+(vals.length<=1?cw/2:i*cw/(vals.length-1)),y:up+ch*(1-v/top),v,i}));
 const line=coords.map((pt,i)=>(i?"L":"M")+pt.x.toFixed(2)+" "+pt.y.toFixed(2)).join(" "),fill=line+" L"+coords.at(-1).x.toFixed(2)+" "+(H-bottom)+" L"+coords[0].x.toFixed(2)+" "+(H-bottom)+" Z";
 const ticks=[0,.25,.5,.75,1].map(n=>{const y=up+ch*(1-n);return '<line x1="'+left+'" y1="'+y+'" x2="'+(W-right)+'" y2="'+y+'" stroke="#353448" stroke-opacity=".48"/><text x="'+(left-15)+'" y="'+(y+5)+'" text-anchor="end" font-size="14" fill="#74788c">'+Math.round(top*n)+'</text>'}).join("");
 const labels=[0,Math.floor((games.length-1)/2),games.length-1].filter((v,i,a)=>a.indexOf(v)===i).map(i=>'<text x="'+coords[i].x+'" y="'+(H-10)+'" fill="#808299" text-anchor="middle" font-size="13">'+esc(dateLabel(games[i].date,true))+'</text>').join("");
 const dots=coords.map((c,i)=>'<circle cx="'+c.x+'" cy="'+c.y+'" r="11" fill="transparent" data-chart-dot="'+i+'" style="cursor:crosshair"/><circle cx="'+c.x+'" cy="'+c.y+'" r="3.5" fill="#ffbb2d" stroke="#1b1b30" stroke-width="1.5"><title>'+esc(dateLabel(games[i].date)+" vs "+games[i].opponent+": "+c.v+" "+field)+'</title></circle>').join("");
 $("performance-svg").innerHTML='<defs><linearGradient id="dh-areagrad" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#ffbe2f" stop-opacity=".34"/><stop offset="100%" stop-color="#ffbe2f" stop-opacity="0"/></linearGradient></defs>'+ticks+'<path d="'+fill+'" fill="url(#dh-areagrad)"/><path d="'+line+'" fill="none" stroke="#ffb82e" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>'+dots+labels;
 $("chart-caption").textContent=games.length+" verified games · "+S.metric+" · ESPN 2026–27 regular-season game log";
 $("performance-svg").querySelectorAll("[data-chart-dot]").forEach(node=>{const game=games[Number(node.dataset.chartDot)],v=game[field];
  node.addEventListener("pointerenter",()=>showTooltip(node,game,v));node.addEventListener("click",()=>showTooltip(node,game,v));
  node.addEventListener("pointerleave",()=>{$("chart-tooltip").hidden=true});
 });
}
function showTooltip(circle,game,v){
 const svg=$("performance-svg"),tip=$("chart-tooltip"),rect=svg.getBoundingClientRect(),r=circle.getBoundingClientRect(),base=$("chart-panel").getBoundingClientRect();
 tip.innerHTML='<span>'+esc(dateLabel(game.date))+' · '+esc(game.opponent)+'</span><strong>'+fmt(v,0)+' '+esc(S.metric.toUpperCase())+'</strong><small>'+esc(game.result||"")+'</small>';
 tip.hidden=false;tip.style.left=clamp(r.left-base.left-32,8,rect.width-120)+"px";tip.style.top=clamp(r.top-base.top-105,65,rect.height-45)+"px";
}
function playerSignals(p){
 const s=statOf(p),m=analysisMetrics(p);if(!s)return{good:[],gaps:[]};const good=[],gaps=[];
 if(s.threePointFieldGoalPct>=38&&s.avgThreePointFieldGoalsAttempted>=3)good.push("Three-point shooting");
 if(s.avgAssists>=6)good.push("Playmaking & creation");
 if(s.avgRebounds>=8)good.push("Rebounding production");
 if(s.avgPoints>=20)good.push("Scoring production");
 if((s.avgSteals||0)+(s.avgBlocks||0)>=2)good.push("Defensive box-score activity");
 if(s.threePointFieldGoalPct<31&&s.avgThreePointFieldGoalsAttempted>=3)gaps.push("Three-point efficiency vs 31% reference");
 if(s.avgTurnovers>=3.3)gaps.push("Turnover volume vs 3.3/g reference");
 if(m.efg!==null&&m.efg<48)gaps.push("eFG% below 48% reference");
 return{good,gaps};
}
function renderPlayerAnalysis(){
 const p=S.selected;if(!p)return;
 const s=statOf(p),metrics=playerSignals(p),ready=!!s;
 let html='<div class="dash-panel-top"><h2>Team Analysis</h2><a href="#team" data-go="team">Open team lab ↗</a></div>';
 if(S.playerTab==="highlights"||S.playerTab==="advanced"){
  const hist=historyOf(p);
  html+='<div class="dash-highlight-frame"><strong>'+(S.playerTab==="advanced"?"Advanced data provenance":"Career & season record")+'</strong><p>'+(S.playerTab==="advanced"?"True shooting, effective shooting and AST/TO are derived from actual ESPN season averages. PER, usage and RAPM are not available from this feed.":hist.length?"Available year-by-year seasons: "+hist.length+". Most recent season: "+esc(hist.at(-1).label)+".":"Historical career trend data is not available for the selected player.")+'</p></div>';
 }else if(ready){
  html+='<div class="dash-highlight-frame"><strong>What this player contributes</strong><p>'+esc(p.name)+' has '+fmt(s.avgPoints)+' points, '+fmt(s.avgRebounds)+' rebounds and '+fmt(s.avgAssists)+' assists per game through '+fmt(s.gamesPlayed,0)+' recorded appearances. These are individual box-score signals, not proof of synergy with a specific team.</p><div class="dash-mini-meter"><span>Scoring output visual scale (0–40 PPG)</span><span>'+fmt(s.avgPoints)+' PPG</span></div><div class="dash-progress"><span style="width:'+clamp((s.avgPoints||0)/40*100,0,100)+'%"></span></div></div>';
  html+='<div class="dash-analysis-grid"><div><h3>Observed Strengths</h3>'+(metrics.good.length?'<ul class="dash-signal-list">'+metrics.good.map(x=>'<li>'+esc(x)+'</li>').join("")+'</ul>':'<p class="dash-muted">No selected threshold exceeded. This does not imply a lack of ability.</p>')+'</div><div><h3>Areas to Investigate</h3>'+(metrics.gaps.length?'<ul class="dash-signal-list negative">'+metrics.gaps.map(x=>'<li>'+esc(x)+'</li>').join("")+'</ul>':'<p class="dash-muted">No low-output flags based on the displayed planning references.</p>')+'</div></div>';
 }else if(p.rookie){
  const probability=rookieProbability(p);
  html+='<div class="dash-highlight-frame"><strong>Thesis career-span model · historical</strong><p>'+esc(p.name)+' has '+fmt(p.rookie.pts)+' PPG in the archived rookie season. Model-inferred probability of a 5+ calendar-year career span: <strong>'+ (probability==null?"—":fmt(100*probability)+"%")+'</strong>. This is an in-sample archived assessment, not a forecast for the current NBA.</p></div>';
 }else html+='<div class="dash-empty">Current verified season statistics are not available yet. We do not generate artificial performance or team synergy.</div>';
 $("player-team-analysis").innerHTML=html;
}
async function prefetchRosterStats(league,id,limit=7){const players=rosterOf(league,id).slice(0,limit);for(let i=0;i<players.length;i+=3){await Promise.allSettled(players.slice(i,i+3).map(fetchStats));if(league==="nba"&&id===S.teamId){renderTeamRoster();renderPlayerStrip()}}}
function historical(p,index){return{...p,id:"rookie-"+index,league:"historical",teamName:"Historical NBA rookie",headshot:null,year:Number(p.year),position:p.position||"—"}}
async function ensureTeamStats(id){
 if(!id)return;const key=String(id);if(S.teamStats.has(key)){if(S.section==="team")renderTeamFactors();return}
 $("team-data-status").textContent="Fetching observed 2026–27 team box scores…";
 try{const j=await endpoint("team-stats","nba",key);S.teamStats.set(key,j)}catch(e){S.teamStats.set(key,null)}
 if(S.teamId===key){$("team-data-status").textContent=S.teamStats.get(key)?"Current season · ESPN":"Team stats temporarily unavailable";renderTeamFactors();renderRecommendation()}
}
function teamData(){return S.teamStats.get(S.teamId)?.fourFactors||null}
function renderTeamRoster(){
 if(!$("team-roster-list"))return;
 const list=rosterOf("nba",S.teamId);
 $("team-roster-list").innerHTML=list.length?list.map(p=>{
  const s=statOf(p);return '<button data-team-player="'+esc(playerKey(p))+'" class="dash-roster-row" style="width:100%;text-align:left;border:0;color:inherit">'+photo(p)+'<div><strong>'+esc(p.name)+'</strong><span>'+esc([p.position,p.jersey?"#"+p.jersey:""].filter(Boolean).join(" · "))+'</span></div><div class="dash-roster-summary"><strong>'+(s?fmt(s.avgPoints)+" PPG":"Stats ↗")+'</strong><span>'+(s?fmt(s.avgRebounds)+" RPG":"View profile")+'</span></div></button>';
 }).join(""):'<div class="dash-empty">Loading the roster, or ESPN data is temporarily unavailable.</div>';
}
function observedStats(){
 const f=teamData();const cols=[["eFG%",f?.efgPct!=null?fmt(f.efgPct*100)+"%":"—","Ref ≥54%"],
 ["TOV%",f?.tovPct!=null?fmt(f.tovPct*100)+"%":"—","Ref ≤14.5%"],
 ["OREB / G",f?.offRebPerGame!=null?fmt(f.offRebPerGame):"—","Proxy · ref ≥11.5"],
 ["FTM / FGA",f?.ftmPerFga!=null?fmt(f.ftmPerFga,3):"—","Ref ≥0.20"]];
 return cols.map(([k,n,t])=>'<div class="dash-factor"><small>'+k+'</small><strong>'+n+'</strong><span>'+t+'</span></div>').join("");
}
function renderTeamFactors(){
 const f=teamData();$("team-factors").innerHTML=observedStats();
 if(!f){$("team-strengths").innerHTML='<p class="dash-muted">Measured team statistics not available. Choose another NBA franchise or refresh.</p>';return}
 const tests=[
 {name:"Effective shooting",good:f.efgPct>=.54,text:"eFG% "+fmt(f.efgPct*100)+"% vs 54% reference"},
 {name:"Possession security",good:f.tovPct<=.145,text:"TOV% "+fmt(f.tovPct*100)+"% vs 14.5% reference"},
 {name:"Offensive boards",good:f.offRebPerGame>=11.5,text:"OREB/G "+fmt(f.offRebPerGame)+" vs 11.5 reference"},
 {name:"Free throw conversion",good:f.ftmPerFga>=.2,text:"FTM/FGA "+fmt(f.ftmPerFga,3)+" vs 0.20 reference"}];
 const good=tests.filter(x=>x.good),bad=tests.filter(x=>!x.good);
 $("team-strengths").innerHTML='<div><h3>Above Planning References</h3>'+(good.length?'<ul class="dash-signal-list">'+good.map(v=>'<li title="'+esc(v.text)+'">'+esc(v.name)+'</li>').join("")+'</ul>':'<p class="dash-muted">None of the four selected thresholds exceeded.</p>')+'</div><div><h3>Potential Gaps</h3>'+(bad.length?'<ul class="dash-signal-list negative">'+bad.map(v=>'<li title="'+esc(v.text)+'">'+esc(v.name)+'</li>').join("")+'</ul>':'<p class="dash-muted">No flags under these editorial references.</p>')+'</div>';
}
function renderTeamNeeds(){
 $("team-need-sliders").innerHTML=skills.map(k=>'<label class="dash-need-item"><div><span>'+esc(skillNames[k])+'</span><b id="need-'+k+'">'+S.needs[k]+'</b></div><input data-need="'+k+'" min="0" max="100" step="5" value="'+S.needs[k]+'" type="range"><small>'+esc(skillDesc[k])+'</small></label>').join("");
}
function suggestNeeds(){
 const f=teamData();if(!f)return;
 const step=x=>clamp(Math.round(x/5)*5,10,95);
 if(f.efgPct!=null)S.needs.spacing=step(65+(f.efgPct-.54)*480);
 if(f.ftmPerFga!=null)S.needs.creation=step(65+(f.ftmPerFga-.2)*230);
 if(f.offRebPerGame!=null)S.needs.rebounding=step(65+(f.offRebPerGame-11.5)*7);
 if(f.tovPct!=null)S.needs.ballsecurity=step(65-(f.tovPct-.145)*420);
 S.needs.defense=50;persist();renderTeamNeeds();renderCandidateList();renderRecommendation();
}
function teamFit(p){
 const ratings=S.ratings[playerKey(p)];if(!ratings||!skills.some(k=>Number.isFinite(ratings[k])))return null;
 let n=0,d=0;for(const k of skills){if(!Number.isFinite(Number(ratings[k])))continue;const w=Math.max(5,100-S.needs[k]);n+=w*ratings[k];d+=w}
 return d?Math.round(n/d):null;
}
function candidates(){const all=new Map();for(const p of rosterOf("college",S.candidateId))all.set(playerKey(p),p);
 for(const key of S.shortlist){const p=getByKey(key);if(p?.league==="college")all.set(key,p)}
 return[...all.values()];
}
function renderCandidateList(){
 const list=candidates().map(p=>({p,fit:teamFit(p)})).sort((a,b)=>(b.fit??-1)-(a.fit??-1));
 $("team-candidates").innerHTML=list.length?list.slice(0,40).map(({p,fit})=>'<div class="dash-candidate-row">'+photo(p)+'<div><strong>'+esc(p.name)+'</strong><small>'+esc(p.position||"NCAA")+' · '+(fit==null?"Not yet rated":S.sources[playerKey(p)]==="stats"?"Observed box-score proxy":"Manual scout rating")+'</small></div><strong>'+(fit==null?"—":fit+"/100")+'</strong><button data-evaluate="'+esc(playerKey(p))+'">Assess ↗</button></div>').join(""):'<div class="dash-empty">Load a college roster to build a candidate shortlist.</div>';
}
function renderRecommendation(){
 const judged=candidates().map(p=>({p,fit:teamFit(p)})).filter(x=>x.fit!=null).sort((a,b)=>b.fit-a.fit);
 if(!judged.length){$("team-recommendation").innerHTML='<div class="dash-recommendation-box"><p>No evaluated recruits yet. Load a college roster, select <strong>Assess</strong> on an athlete, and rate their skills—or opt into a box-score proxy. We will not make up a synergy score.</p></div>';return}
 const x=judged[0],source=S.sources[playerKey(x.p)]==="stats"?"box-score proxy":"manual scouting scores";
 $("team-recommendation").innerHTML='<div class="dash-recommendation-box"><p><strong>'+esc(x.p.name)+'</strong> currently ranks highest in your evaluated NCAA candidates with a <strong>'+x.fit+'/100</strong> illustrative fit index, based on '+source+' and the editable needs of <strong>'+esc(teamName(S.teamId)||"your NBA team")+'</strong>. This is a scouting-priority ranking, not predicted NBA wins or roster synergy.</p><div class="dash-mini-meter"><span>Weighted fit / 100 · heuristic</span><span>'+x.fit+'/100</span></div><div class="dash-progress"><span style="width:'+x.fit+'%"></span></div></div>';
}
function renderTeam(){
 $("team-franchise").value=S.teamId;renderTeamRoster();renderTeamFactors();renderTeamNeeds();renderCandidateList();renderRecommendation();
}
async function chooseTeam(id){
 if(!id)return;S.teamId=id;renderTeam();const roster=loadRoster("nba",id).catch(()=>[]);
 await Promise.allSettled([roster,ensureTeamStats(id)]);if(S.teamId===id)renderTeam();
}
async function loadCandidates(id){
 if(!id)return;S.candidateId=id;$("candidate-school").value=id;$("team-candidates").innerHTML='<div class="dash-skeleton">Loading college prospects…</div>';
 try{await loadRoster("college",id)}catch(e){$("team-candidates").innerHTML='<div class="dash-empty">ESPN roster unavailable for this program.</div>'}
 renderCandidateList();renderRecommendation();
}
function gradeCollege(p){
 const s=statOf(p);if(!s||s.gamesPlayed<3||!s.avgMinutes)return null;
 const per=36/Math.max(s.avgMinutes,8),pts=(s.avgPoints||0)*per,reb=(s.avgRebounds||0)*per,ast=(s.avgAssists||0)*per,stl=(s.avgSteals||0)*per,blk=(s.avgBlocks||0)*per,to=(s.avgTurnovers||0)*per,three=(s.avgThreePointFieldGoalsAttempted||0)*per;
 const step=n=>clamp(Math.round(n/5)*5,0,100);
 return{spacing:step(20+((s.threePointFieldGoalPct||0)-28)*2+three*3),creation:step(10+pts*2.3+ast*1.8),
  defense:step(10+(stl+blk)*24),rebounding:step(10+reb*6),ballsecurity:step(58+ast*6-to*8)};
}
async function showAssess(key){
 const p=getByKey(key);if(!p)return;
 const modal=$("dash-assess-modal");if(!modal)return;
 modal.hidden=false;const body=$("dash-assess-body");
 body.innerHTML='<h2>Scout '+esc(p.name)+'</h2><p>Adjust these subjective 0–100 scout ratings. They are not model-predicted NBA skills.</p>'+
 skills.map(k=>'<label class="dash-need-item"><div><span>'+esc(skillNames[k])+'</span><b id="assess-label-'+k+'">'+(S.ratings[key]?.[k]??50)+'</b></div><input data-assess="'+k+'" min="0" max="100" step="5" type="range" value="'+(S.ratings[key]?.[k]??50)+'"></label>').join("")+
 '<div class="dash-modal-actions"><button id="assess-save" class="dash-button dash-button-yellow">Save ratings</button><button id="assess-auto" class="dash-button dash-button-purple">Use observed box-score proxy</button></div><p id="assess-note" class="dash-tiny-help">Box-score proxy requires at least three observed college games and is not validated against NBA performance.</p>';
 const working={...Object.fromEntries(skills.map(k=>[k,50])),...S.ratings[key]};
 body.querySelectorAll("[data-assess]").forEach(x=>x.addEventListener("input",e=>{working[x.dataset.assess]=Number(e.target.value);$("assess-label-"+x.dataset.assess).textContent=e.target.value}));
 $("assess-save").addEventListener("click",()=>{S.ratings[key]=working;S.sources[key]="manual";if(!S.shortlist.includes(key))S.shortlist.push(key);persist();modal.hidden=true;renderCandidateList();renderRecommendation()});
 $("assess-auto").addEventListener("click",async()=>{const b=$("assess-auto");b.disabled=true;b.textContent="Retrieving season averages…";await fetchStats(p);const data=gradeCollege(p);
  if(!data){$("assess-note").textContent="Not enough verified college games (minimum three) or missing minutes to form a box-score proxy.";b.disabled=false;b.textContent="Use observed box-score proxy";return}
  Object.assign(working,data);S.ratings[key]={...data};S.sources[key]="stats";if(!S.shortlist.includes(key))S.shortlist.push(key);persist();modal.hidden=true;renderCandidateList();renderRecommendation();});
}
function renderDraftPlayers(){
 const list=rosterOf("college",S.collegeId);if(!list.length){$("draft-player").innerHTML='<option value="">No roster found</option>';return}
 if(!S.draftPlayer||!list.some(p=>p.id===S.draftPlayer.id))S.draftPlayer=list[0];
 $("draft-player").innerHTML=list.map(p=>'<option value="'+esc(p.id)+'" '+(S.draftPlayer?.id===p.id?"selected":"")+'>'+esc(p.name)+'</option>').join("");
 renderDraft();
}
function renderDraft(){
 const m=S.draft,p=S.draftPlayer;if(!m)return;
 const five=m.horizons.five.bands.find(b=>S.draftPick>=b.low&&S.draftPick<=b.high),ten=m.horizons.ten.bands.find(b=>S.draftPick>=b.low&&S.draftPick<=b.high);
 if(!five||!ten)return;$("draft-pick-range").value=S.draftPick;$("draft-pick-label").textContent="#"+S.draftPick;
 $("draft-pick-band").textContent="Assumed "+five.name+" · picks "+five.range;
 $("draft-five").textContent=fmt(five.prob*100)+"%";$("draft-ten").textContent=fmt(ten.prob*100)+"%";
 $("draft-five-bar").style.width=five.prob*100+"%";$("draft-ten-bar").style.width=ten.prob*100+"%";
 $("draft-outcome-details").innerHTML='<span>Training group: '+five.n+' drafted players</span><span>Median calendar span: '+five.median+' years</span>';
 const s=p?statOf(p):null,college=teamName(S.collegeId,"college")||"College program";
 $("draft-selected").innerHTML=p?'<div style="display:flex;align-items:center;gap:15px">'+photo(p)+'<div><strong>'+esc(p.name)+'</strong><small>'+esc(college)+' · '+esc(p.position||"NCAA player")+'</small><small>'+(s?fmt(s.avgPoints)+" PPG · "+fmt(s.avgRebounds)+" RPG · "+fmt(s.gamesPlayed,0)+" GP":"Loading observed college stats…")+'</small></div></div>':'<p class="dash-muted">Select an athlete to begin.</p>';
 const rows=[["01","NCAA college player",p?p.name+" · "+college:"College program","OBSERVED"],["02","Draft selection scenario","Assumed overall pick #"+S.draftPick,"ASSUMED"],["03","NBA career-span outlook",fmt(five.prob*100)+"% 5+ years · "+fmt(ten.prob*100)+"% 10+ years","HISTORICAL"]];
 $("draft-timeline").innerHTML=rows.map(x=>'<div class="dash-timeline-row"><span class="dash-timeline-node">'+x[0]+'</span><div><b>'+esc(x[1])+'</b><p>'+esc(x[2])+'</p></div><span>'+x[3]+'</span></div>').join("");
}
async function chooseCollege(id){
 if(!id)return;S.collegeId=id;S.draftPlayer=null;$("draft-school").value=id;$("draft-player").innerHTML='<option>Loading NCAA roster…</option>';
 try{await loadRoster("college",id);renderDraftPlayers();if(S.draftPlayer){await fetchStats(S.draftPlayer);renderDraft()}}
 catch(e){$("draft-player").innerHTML='<option>College roster unavailable</option>';S.draftPlayer=null;renderDraft()}
}
function setPlayerTab(tab){
 if(!["overview","advanced","highlights"].includes(tab))return;S.playerTab=tab;
 document.querySelectorAll("[data-player-tab]").forEach(b=>b.classList.toggle("on",b.dataset.playerTab===tab));
 renderPlayer();
}
function bind(){
 document.querySelectorAll("[data-section]").forEach(b=>b.addEventListener("click",()=>nav(b.dataset.section)));
 document.querySelectorAll("[data-go]").forEach(a=>a.addEventListener("click",e=>{e.preventDefault();nav(a.dataset.go)}));
 document.querySelectorAll("[data-league]").forEach(b=>b.addEventListener("click",()=>changeLeague(b.dataset.league)));
 document.querySelectorAll("[data-player-tab]").forEach(b=>b.addEventListener("click",()=>setPlayerTab(b.dataset.playerTab)));
 document.querySelectorAll("[data-metric]").forEach(b=>b.addEventListener("click",()=>{S.metric=b.dataset.metric;renderChart()}));
 document.querySelectorAll("[data-range]").forEach(b=>b.addEventListener("click",()=>{S.range=b.dataset.range;renderChart()}));
 for(const id of ["players-search","players-position","players-min-pts","players-min-ast","players-min-reb","players-sort"]){
  $(id).addEventListener(["players-search","players-min-pts","players-min-ast","players-min-reb"].includes(id)?"input":"change",renderPlayerStrip);
 }
 $("players-team").addEventListener("change",async e=>{
  const value=e.target.value;if(S.league==="historical"){renderPlayerStrip();return}
  if(S.league==="college"){S.collegeId=value;await loadRoster("college",value).catch(()=>[])}
  else if(value!=="all")await loadRoster("nba",value).catch(()=>[]);
  S.selected=null;S.selectedKey="";renderPlayerStrip();
 });
 $("players-refresh").addEventListener("click",async()=>{
  const p=S.selected;if(p?.league==="nba"||p?.league==="college"){S.stats.delete(playerKey(p));S.games.delete(playerKey(p));S.history.delete(playerKey(p));await selectPlayer(p)}
  else renderPlayerStrip();
 });
 $("team-franchise").addEventListener("change",e=>chooseTeam(e.target.value));
 $("team-suggest-needs").addEventListener("click",suggestNeeds);
 $("team-reset-needs").addEventListener("click",()=>{for(const k of skills)S.needs[k]=50;persist();renderTeamNeeds();renderCandidateList();renderRecommendation()});
 $("candidate-school").addEventListener("change",e=>loadCandidates(e.target.value));
 $("candidate-load").addEventListener("click",()=>loadCandidates($("candidate-school").value));
 $("draft-school").addEventListener("change",e=>chooseCollege(e.target.value));
 $("draft-player").addEventListener("change",async e=>{S.draftPlayer=rosterOf("college",S.collegeId).find(p=>p.id===e.target.value)||null;renderDraft();if(S.draftPlayer){await fetchStats(S.draftPlayer);renderDraft()}});
 $("draft-pick-range").addEventListener("input",e=>{S.draftPick=clamp(Number(e.target.value),1,60);renderDraft()});
 document.addEventListener("click",e=>{
  const node=e.target.closest("[data-player]");if(node){const p=getByKey(node.dataset.player);if(p)selectPlayer(p);return}
  const roster=e.target.closest("[data-team-player]");if(roster){const p=getByKey(roster.dataset.teamPlayer);if(p){S.league="nba";document.querySelectorAll("[data-league]").forEach(x=>x.classList.toggle("on",x.dataset.league==="nba"));setupTeams();nav("players");selectPlayer(p)}return}
  const assess=e.target.closest("[data-evaluate]");if(assess){showAssess(assess.dataset.evaluate);return}
  if(e.target.id==="dash-modal-close"||e.target.id==="dash-modal-backdrop")$("dash-assess-modal").hidden=true;
 });
 document.addEventListener("input",e=>{const k=e.target.dataset.need;if(k){S.needs[k]=Number(e.target.value);$("need-"+k).textContent=e.target.value;persist();renderCandidateList();renderRecommendation()}});
 document.addEventListener("keydown",e=>{if(e.key==="Escape"&&$("dash-assess-modal"))$("dash-assess-modal").hidden=true});
 window.addEventListener("hashchange",()=>{const v=location.hash.slice(1);if(["players","team","prospects"].includes(v)&&v!==S.section)nav(v)});
}
async function init(){
 restore();bind();const results=await Promise.allSettled([endpoint("teams","nba"),endpoint("teams","college"),jsonFile("/data/rookies.json"),jsonFile("/data/model.json"),jsonFile("/data/draft-path.json")]);
 if(results[0].status==="fulfilled")S.teams.nba=results[0].value.teams.sort((a,b)=>a.name.localeCompare(b.name));
 if(results[1].status==="fulfilled")S.teams.college=results[1].value.teams.sort((a,b)=>a.name.localeCompare(b.name));
 if(results[2].status==="fulfilled")S.historical=results[2].value.map(historical);
 if(results[3].status==="fulfilled")S.model=results[3].value;
 if(results[4].status==="fulfilled")S.draft=results[4].value;
 setupTeams();renderTeamNeeds();renderDraft();renderPlayerStrip();
 const v=location.hash.slice(1);nav(["players","team","prospects"].includes(v)?v:"players");
 await Promise.allSettled([loadRoster("nba","13"),loadRoster("nba","9"),loadRoster("college","150")]);
 const expanded=["PHX","PHI","MIL"].map(a=>S.teams.nba.find(t=>t.abbreviation===a)?.id).filter(Boolean);
 Promise.allSettled(expanded.map(id=>loadRoster("nba",id))).then(()=>renderPlayerStrip());
 await Promise.allSettled([chooseTeam(S.teamId),chooseCollege(S.collegeId),loadCandidates(S.candidateId)]);
 const p=getByKey(S.selectedKey)||featured[0];await selectPlayer(p);
}
const modal=document.createElement("div");modal.id="dash-assess-modal";modal.hidden=true;modal.innerHTML='<div class="dash-modal-backdrop" id="dash-modal-backdrop"></div><div class="dash-modal-panel" role="dialog" aria-modal="true" aria-label="College player scouting assessment"><button class="dash-modal-close" id="dash-modal-close" aria-label="Close dialog">×</button><div id="dash-assess-body"></div></div>';document.body.append(modal);
init();
