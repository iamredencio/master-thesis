"use strict";
const $=id=>document.getElementById(id);
const clean=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const numeral=(v,d=1)=>Number(v).toFixed(d);
const needsKeys=["spacing","creation","defense","rebounding","ballsecurity"];
const needNames={spacing:"Floor spacing",creation:"Shot creation",defense:"Rim & perimeter defense",rebounding:"Rebounding",ballsecurity:"Ball security"};
const needNotes={spacing:"Efficient 3-point threat and off-ball gravity",creation:"Self-created offense and passing",defense:"Stops, steals, blocks and rotations",rebounding:"Second chances and box-out coverage",ballsecurity:"Passing accuracy and turnover control"};
const state={tab:"players",league:"college",teams:{college:[],nba:[]},rosters:{college:new Map(),nba:new Map()},historical:[],model:null,draft:null,collegeTeam:"150",nbaTeam:"13",selected:null,selectedProspect:null,pick:15,rosterStatus:"loading",shortlist:[],assessments:{},needs:{spacing:50,creation:50,defense:50,rebounding:50,ballsecurity:50},statCache:new Map(),teamStats:new Map(),assessmentSources:{}};
function initials(name){return String(name||"CL").split(/\s+/).map(x=>x[0]||"").slice(0,2).join("").toUpperCase()||"CL"}
function isPortrait(u){return typeof u==="string"&&/^https:\/\/[^/]*\.(?:espncdn|espn)\.com\//i.test(u)}
function portrait(p,size="card"){const cls=size==="large"?"sc-face sc-face-lg":"sc-face";return '<div class="'+cls+'"><span>'+clean(initials(p.name))+'</span>'+(isPortrait(p.headshot)?'<img src="'+clean(p.headshot)+'" alt="" loading="lazy" referrerpolicy="no-referrer" onerror="this.remove()">':"")+'</div>'}
function keyOf(p){return p.league+":"+p.id}
function api(type,league,id){let query="type="+encodeURIComponent(type)+"&league="+encodeURIComponent(league);if(id)query+="&"+(type==="player"?"id":"team")+"="+encodeURIComponent(id);return fetch("/api/scout?"+query,{headers:{"Accept":"application/json"}}).then(async res=>{const j=await res.json().catch(()=>({}));if(!res.ok||!j.ok)throw Error(j.error||"Data source unavailable");return j})}
function showTab(tab){if(!["players","prospects","team"].includes(tab))tab="players";state.tab=tab;document.querySelectorAll(".sc-view").forEach(v=>v.classList.toggle("show",v.id==="sc-view-"+tab));document.querySelectorAll("[data-tab]").forEach(b=>b.classList.toggle("active",b.dataset.tab===tab));$("sc-crumb").textContent={players:"Player cards",prospects:"Draft lab",team:"Team fit lab"}[tab];history.replaceState(null,"","#"+tab);if(tab==="prospects")renderDraft();if(tab==="team")renderTeam();window.scrollTo({top:0,behavior:"instant"})}
function showError(message){$("sc-player-grid").innerHTML='<div class="sc-empty"><b>Players temporarily unavailable</b><p>'+clean(message)+'</p><button class="sc-action sc-dark" id="sc-error-retry">Retry feed ↻</button></div>';const b=$("sc-error-retry");if(b)b.addEventListener("click",()=>loadRoster(state.league,true))}
function convertRoster(p,league,teamName){return {...p,league,teamName:teamName||p.teamName||"",source:"ESPN",year:2027,id:String(p.id||"")}}
async function loadTeams(league){try{const j=await api("teams",league);state.teams[league]=j.teams.slice().sort((a,b)=>a.name.localeCompare(b.name));return true}catch(e){return false}}
function teamOptions(league,current){let options=state.teams[league].map(t=>'<option value="'+clean(t.id)+'" '+(String(t.id)===String(current)?"selected":"")+'>'+clean(t.name)+'</option>').join("");return options||'<option value="">Teams temporarily unavailable</option>'}
function populateTeams(){$("sc-team-pick").innerHTML=teamOptions(state.league,state.league==="college"?state.collegeTeam:state.nbaTeam);$("sc-college-select").innerHTML=teamOptions("college",state.collegeTeam);$("sc-nba-select").innerHTML=teamOptions("nba",state.nbaTeam);$("sc-teamlabel").textContent=state.league==="college"?"COLLEGE PROGRAM":"NBA FRANCHISE"}
function rosterFor(league){if(league==="historical")return state.historical;return state.rosters[league].get(league==="college"?state.collegeTeam:state.nbaTeam)||[]}
async function loadRoster(league,force=false){if(league==="historical"){state.rosterStatus="ready";renderCards();return}
 const id=league==="college"?state.collegeTeam:state.nbaTeam,cache=state.rosters[league];
 if(cache.has(id)&&!force){state.rosterStatus="ready";renderCards();if(league==="college")updateProspectOptions();if(league==="nba")renderTeam();return}
 state.rosterStatus="loading";if(league===state.league){$("sc-player-grid").innerHTML='<div class="sc-empty">Loading '+(league==="college"?"college":"NBA")+' player cards…</div>'}
 try{const data=await api("roster",league,id);const players=data.players.map(p=>convertRoster(p,league,data.team?.name||""));cache.set(id,players);state.rosterStatus="ready";
  if(league==="college"){if(!state.selectedProspect||!players.some(p=>p.id===state.selectedProspect.id))state.selectedProspect=players[0]||null;updateProspectOptions();}
  if(league==="nba")renderTeam();
  if(league===state.league)renderCards();
 }catch(e){state.rosterStatus="error";if(league===state.league)showError(e.message);if(league==="nba")renderTeam()}
}
function filterPlayers(){let players=rosterFor(state.league);const term=$("sc-search").value.trim().toLowerCase(),order=$("sc-sort").value;let selected=players.filter(p=>!term||(p.name+" "+p.position+" "+(p.teamName||"")).toLowerCase().includes(term));selected=selected.slice().sort((a,b)=>order==="year"?(Number(b.year||0)-Number(a.year||0))||a.name.localeCompare(b.name):order==="position"?(a.position||"").localeCompare(b.position||"")||a.name.localeCompare(b.name):a.name.localeCompare(b.name));return selected.slice(0,state.league==="historical"?180:120)}
function historicalChance(p){const m=state.model;if(!m||!p.rookie)return null;const x=[p.rookie.min,p.rookie.pts,p.rookie.gp,p.rookie.reb,p.rookie.ast,p.rookie.stl,p.rookie.blk,p.rookie.tov,p.rookie.fg,p.rookie.threePa,p.rookie.ft,p.rookie.eff,p.rookie.age];const z=m.coef.reduce((s,v,i)=>s+v*clamp((x[i]-m.mean[i])/m.std[i],-m.clipZ,m.clipZ),m.intercept);return 1/(1+Math.exp(-clamp(z,-25,25)))}
function statChip(label,value){return '<div><b>'+clean(value)+'</b><small>'+label+'</small></div>'}
function card(p){const saved=state.shortlist.includes(keyOf(p)),isHistoric=p.league==="historical";const pr=isHistoric?historicalChance(p):null;const known=state.statCache.get(keyOf(p))?.stats;const foot=isHistoric?statChip("PPG",numeral(p.rookie.pts))+statChip("REB",numeral(p.rookie.reb))+statChip("MIN",numeral(p.rookie.min)):known&&known.gamesPlayed?statChip("PPG",numeral(known.avgPoints??0))+statChip("REB",numeral(known.avgRebounds??0))+statChip("AST",numeral(known.avgAssists??0)):statChip("POSITION",p.position||"—")+statChip(p.league==="college"?"CLASS":"AGE",p.league==="college"?(p.classYear||"—"):(p.age??"—"))+statChip("HEIGHT",p.height||"—");
 const tag=p.league==="college"?"NCAA 2026–27":p.league==="nba"?"NBA 2026–27":"THESIS ARCHIVE";
 return '<article class="sc-player-card" data-card-id="'+clean(keyOf(p))+'"><div class="sc-cardart"><div class="sc-artpattern"></div><span class="sc-cardlabel">'+tag+'</span><span class="sc-cardnum">'+clean(p.jersey?"#"+p.jersey:String(p.year||""))+'</span>'+portrait(p)+'</div><div class="sc-cardinfo"><div class="sc-cardmeta">'+clean(p.teamName||"HISTORICAL NBA ROOKIE")+'</div><h3 title="'+clean(p.name)+'">'+clean(p.name)+'</h3><div class="sc-cardstats">'+foot+'</div><div class="sc-cardbottom"><button class="sc-cardopen" data-act="detail" data-id="'+clean(keyOf(p))+'">View scouting card <span>↗</span></button><button class="sc-star '+(saved?"selected":"")+'" data-act="shortlist" data-id="'+clean(keyOf(p))+'" aria-label="'+(saved?"Remove":"Add")+" shortlist" title="Toggle shortlist">'+(saved?"★":"☆")+'</button></div></div></article>';
}
function renderCards(){populateTeams();$("sc-team-selector").style.display=state.league==="historical"?"none":"";$("sc-source-note").textContent=state.league==="historical"?"Archived 1980–2016 rookie seasons · model not current":state.league==="nba"?"Live ESPN NBA roster · not player forecasts":"Live ESPN NCAA team roster · draft estimates require assumed pick";const list=filterPlayers();$("sc-card-count").textContent=list.length;$("sc-card-source").textContent=state.league==="historical"?"Thesis archive":state.league==="nba"?"NBA roster · ESPN":"NCAA roster · ESPN";
 if(state.rosterStatus==="error"&&state.league!=="historical")return;
 $("sc-player-grid").innerHTML=list.length?list.map(card).join(""):'<div class="sc-empty">No players match your search. Try another team or remove the filter.</div>'}
function findPlayer(key){if(!key)return null;let [league,id]=key.split(":");if(league==="historical")return state.historical.find(p=>keyOf(p)===key);if(league==="college"||league==="nba")for(const roster of state.rosters[league].values()){let p=roster.find(p=>p.id===id);if(p)return p}return null}
function shortlistToggle(p){const key=keyOf(p);const ix=state.shortlist.indexOf(key);if(ix>=0)state.shortlist.splice(ix,1);else state.shortlist.push(key);persist();renderCards();renderCandidateList();if($("sc-player-modal").hidden===false)openDetails(p)}
function setAssessment(p,key,value){if(!state.assessments[keyOf(p)])state.assessments[keyOf(p)]={};state.assessments[keyOf(p)][key]=clamp(Number(value),0,100);state.assessmentSources[keyOf(p)]="manual";persist();renderCandidateList()}
function persist(){try{sessionStorage.setItem("courtline-scout-v2",JSON.stringify({shortlist:state.shortlist,assessments:state.assessments,assessmentSources:state.assessmentSources,needs:state.needs}))}catch(e){}}
function recover(){try{const j=JSON.parse(sessionStorage.getItem("courtline-scout-v2")||"{}");if(Array.isArray(j.shortlist))state.shortlist=j.shortlist.filter(v=>typeof v==="string").slice(0,120);if(j.assessments&&typeof j.assessments==="object")state.assessments=j.assessments;if(j.assessmentSources&&typeof j.assessmentSources==="object")state.assessmentSources=j.assessmentSources;if(j.needs&&typeof j.needs==="object")for(const key of needsKeys)if(Number.isFinite(Number(j.needs[key])))state.needs[key]=clamp(Number(j.needs[key]),0,100)}catch(e){}}
function closeDetails(){$("sc-player-modal").hidden=true;document.body.classList.remove("sc-lock")}

async function fetchStats(p){
 if(!p||!["nba","college"].includes(p.league))return null;
 const key=keyOf(p);
 if(!state.statCache.has(key)){
  try{const result=await api("player",p.league,p.id);state.statCache.set(key,result)}
  catch(e){state.statCache.set(key,null)}
 }
 return state.statCache.get(key);
}
function gradeBoxScore(p){
 const result=state.statCache.get(keyOf(p)),s=result?.stats;
 if(p.league!=="college"||!s||!s.avgMinutes||s.gamesPlayed<3)return null;
 const factor=36/Math.max(8,s.avgMinutes),pts=(s.avgPoints||0)*factor,reb=(s.avgRebounds||0)*factor,ast=(s.avgAssists||0)*factor,stl=(s.avgSteals||0)*factor,blk=(s.avgBlocks||0)*factor,tov=(s.avgTurnovers||0)*factor,three=(s.avgThreePointFieldGoalsAttempted||0)*factor,pct=s.threePointFieldGoalPct||0;
 const round=x=>clamp(Math.round(x/5)*5,0,100);
 return{spacing:round(20+(pct-28)*2+three*3),creation:round(10+pts*2.3+ast*1.8),defense:round(10+(stl+blk)*24),rebounding:round(10+reb*6),ballsecurity:round(58+ast*6-tov*8)};
}
async function openDetails(p){
 if(!p)return;
 state.selected=p;$("sc-player-modal").hidden=false;document.body.classList.add("sc-lock");
 renderDetails(p);$("sc-close-modal").focus();
 if(p.league==="nba"||p.league==="college"){
  await fetchStats(p);
  if(state.selected===p&&!$("sc-player-modal").hidden)renderDetails(p);
  renderCards();if(state.selectedProspect?.id===p.id)renderDraft();
 }
}
function renderDetails(p){
 const result=state.statCache.get(keyOf(p)),stats=result?.stats||{},hasStats=!!stats.gamesPlayed;
 const assessed=state.assessments[keyOf(p)]||{},isCollege=p.league==="college",saved=state.shortlist.includes(keyOf(p));
 const kpis=hasStats?[["PPG","avgPoints"],["RPG","avgRebounds"],["APG","avgAssists"],["MIN","avgMinutes"],["3P%","threePointFieldGoalPct"],["GP","gamesPlayed"]]:[];
 const blocks=hasStats?kpis.map(([label,key])=>statChip(label,stats[key]===undefined?"—":numeral(stats[key],key==="gamesPlayed"?0:1))).join(""):p.rookie?statChip("POINTS",numeral(p.rookie.pts))+statChip("REBOUNDS",numeral(p.rookie.reb))+statChip("ASSISTS",numeral(p.rookie.ast))+statChip("MINUTES",numeral(p.rookie.min)):'<p class="sc-stats-empty">Current-season individual statistics are not available for this player yet. Only observed roster identity and measurements are displayed.</p>';
 const provenance=p.league==="historical"?'<div class="sc-data-note">Historical rookie career-span model estimate: '+numeral(100*historicalChance(p))+'% for 5+ calendar years. This is an archived in-sample profile, not a current prediction.</div>':'<div class="sc-data-note">'+(hasStats?("Observed "+(result.seasonDisplay||"2026–27")+" ESPN box-score averages through "+stats.gamesPlayed+" games. Small samples are volatile. "):"Verified ESPN roster. ")+"Skill ratings below are editable scouting assessments; automatic estimates are transparent box-score heuristics, not a validated NBA transition model.</div>';
 const badge=state.assessmentSources[keyOf(p)]==="box-score"?"BOX-SCORE PROXY":"MANUAL ASSESSMENT";
 const suggested=isCollege&&gradeBoxScore(p)?'<button class="sc-action sc-light" data-act="auto-grade" data-id="'+clean(keyOf(p))+'">Use observed box-score proxy ↻</button>':"";
 $("sc-drawer-content").innerHTML='<div class="sc-drawerhero">'+portrait(p,"large")+'<span>'+clean(p.league==="historical"?"HISTORICAL RECORD":"ESPN ROSTER · 2026–27")+'</span><h2>'+clean(p.name)+'</h2><p>'+clean(p.teamName||"NBA archive")+" · "+clean(p.position||"—")+'</p></div><div class="sc-drawerbody"><div class="sc-drawer-statbar"><span>PLAYER FACT SHEET</span><span>OBSERVED FIELDS</span></div><div class="sc-drawer-facts">'+statChip("HEIGHT",p.height||"—")+statChip(isCollege?"CLASS":"AGE",isCollege?(p.classYear||"—"):(p.age??"—"))+statChip("POSITION",p.position||"—")+'</div><h3>Available basketball data</h3><div class="sc-drawer-stats">'+blocks+'</div>'+provenance+'<h3>Scouting skill assessment <small>'+badge+'</small></h3><div class="sc-assessment">'+needsKeys.map(k=>'<label><span>'+clean(needNames[k])+' <b id="sc-assess-value-'+k+'">'+(assessed[k]??50)+'</b></span><input data-assess="'+k+'" data-for="'+clean(keyOf(p))+'" type="range" min="0" max="100" value="'+(assessed[k]??50)+'" step="5"></label>').join("")+'</div><div class="sc-draweractions">'+suggested+'<button class="sc-action sc-dark" data-act="shortlist" data-id="'+clean(keyOf(p))+'">'+(saved?"Remove from":"Add to")+" shortlist ★</button>'+ (isCollege?'<button class="sc-action sc-light" data-act="draft" data-id="'+clean(keyOf(p))+'">Draft path ↗</button>':"")+'</div></div>';
}

function switchLeague(league){state.league=league;document.querySelectorAll("[data-league]").forEach(b=>b.classList.toggle("selected",b.dataset.league===league));$("sc-search").value="";populateTeams();loadRoster(league)}
function updateProspectOptions(){const players=state.rosters.college.get(state.collegeTeam)||[];$("sc-prospect-select").innerHTML=players.map(p=>'<option value="'+clean(p.id)+'" '+(state.selectedProspect?.id===p.id?"selected":"")+'>'+clean(p.name)+' · '+clean(p.position)+'</option>').join("")||'<option value="">No roster available</option>';renderDraft();if(state.selectedProspect)fetchStats(state.selectedProspect).then(()=>renderDraft())}
function renderDraft(){const m=state.draft;if(!m)return;const p=state.selectedProspect;const bands=m.horizons.five.bands,band=bands.find(b=>state.pick>=b.low&&state.pick<=b.high)||bands[0],bandTen=m.horizons.ten.bands.find(b=>state.pick>=b.low&&state.pick<=b.high)||m.horizons.ten.bands[0];
 $("sc-pick-number").innerHTML=state.pick+'<span>'+ (state.pick%10===1&&state.pick%100!==11?"st":state.pick%10===2&&state.pick%100!==12?"nd":state.pick%10===3&&state.pick%100!==13?"rd":"th")+'</span>';$("sc-pick-band").textContent=band.name.toUpperCase();$("sc-draft-pick").value=state.pick;
 $("sc-prob-five").textContent=numeral(band.prob*100)+"%";$("sc-prob-ten").textContent=numeral(bandTen.prob*100)+"%";$("sc-bar-label-five").textContent=numeral(band.prob*100)+"%";$("sc-bar-label-ten").textContent=numeral(bandTen.prob*100)+"%";$("sc-bar-five").style.width=band.prob*100+"%";$("sc-bar-ten").style.width=bandTen.prob*100+"%";$("sc-median").textContent=String(band.median)+" years";$("sc-cohort-count").textContent=String(band.n);$("sc-timeline-draft").textContent="Assumed overall pick #"+state.pick+" · "+band.name;
 const school=state.teams.college.find(t=>t.id===state.collegeTeam)?.name||"College program";$("sc-timeline-school").textContent=p?p.name+" · "+school:school;
 $("sc-timeline-outcome").textContent="Historic "+band.name+" · "+numeral(band.prob*100)+"% for 5+ years";
 const record=p?state.statCache.get(keyOf(p)):null;const current=record?.stats;const measured=current?.gamesPlayed?" · "+current.gamesPlayed+" GP · "+numeral(current.avgPoints??0)+" PPG · "+numeral(current.avgRebounds??0)+" RPG (observed)":" · No verified current stats";
 $("sc-selected-prospect").innerHTML=p?'<div class="sc-prospect-mini">'+portrait(p)+'<div><strong>'+clean(p.name)+'</strong><span>'+clean(school)+'</span><small>'+clean([p.position,p.classYear,p.height].filter(Boolean).join(" · ")+measured)+'</small></div></div>':'<p>Select a player from a college roster.</p>';
}
function needSliders(){return needsKeys.map(k=>'<label class="sc-skillrange"><div><span>'+needNames[k]+'</span><b id="sc-need-value-'+k+'">'+state.needs[k]+'</b></div><input data-need="'+k+'" type="range" value="'+state.needs[k]+'" min="0" max="100" step="5"><small>'+needNotes[k]+'</small></label>').join("")}
function renderNeeds(skipSliders=false){if(!skipSliders)$("sc-need-sliders").innerHTML=needSliders();const arr=needsKeys.map(k=>({key:k,gap:100-state.needs[k],value:state.needs[k]})).sort((a,b)=>b.gap-a.gap);$("sc-primary-need").textContent=needNames[arr[0].key];
 $("sc-gaps").innerHTML=arr.map(x=>'<div class="sc-gapitem"><div><b>'+clean(needNames[x.key])+'</b><span>'+x.gap+' / 100 NEED</span></div><div class="sc-gapbar"><i style="width:'+x.gap+'%"></i></div></div>').join("")}

function observedFour(){
 const panel=$("sc-observed-four"),entry=state.teamStats.get(state.nbaTeam),f=entry?.fourFactors;
 $("sc-suggest-needs").disabled=!f;
 if(!f){panel.innerHTML='<div class="sc-helper">Observed team Four Factors unavailable. Edit scouting assumptions below to assess roster fit.</div>';return}
 const items=[
  ["eFG%",f.efgPct==null?null:(f.efgPct*100).toFixed(1)+"%","Reference ≥ 54.0%"],
  ["TOV%",f.tovPct==null?null:(f.tovPct*100).toFixed(1)+"%","Reference ≤ 14.5%"],
  ["OREB / G",f.offRebPerGame==null?null:f.offRebPerGame.toFixed(1),"Proxy · reference ≥ 11.5"],
  ["FTM / FGA",f.ftmPerFga==null?null:f.ftmPerFga.toFixed(3),"Reference ≥ 0.20"]
 ];
 panel.innerHTML='<div class="sc-observed-title"><strong>OBSERVED TEAM PRODUCTION · 2026–27</strong><small>'+String(f.games||0)+' games · ESPN</small></div><div class="sc-observed-cells">'+items.map(([label,value,reference])=>'<div><span>'+label+'</span><b>'+clean(value??"—")+'</b><small>'+clean(reference)+'</small></div>').join("")+'</div><p>Measured shooting, turnovers and free-throw rate; offensive rebounding is per game because opponent defensive rebounds are not provided. References are planning targets, not league averages. '+(Number(f.games)<15?"Early-season sample; treat gaps cautiously.":"")+'</p>';
}
async function loadTeamStats(id){
 try{const stats=await api("team-stats","nba",id);state.teamStats.set(id,stats)}catch(e){state.teamStats.set(id,null)}
 if(state.nbaTeam===id){observedFour()}
}
function suggestNeedsFromTeamStats(){
 const f=state.teamStats.get(state.nbaTeam)?.fourFactors;if(!f)return;
 const scale=v=>clamp(Math.round(v/5)*5,10,95);
 if(f.efgPct!==null)state.needs.spacing=scale(65+(f.efgPct-.54)*480);
 if(f.ftmPerFga!==null)state.needs.creation=scale(65+(f.ftmPerFga-.20)*230);
 if(f.offRebPerGame!==null)state.needs.rebounding=scale(65+(f.offRebPerGame-11.5)*7);
 if(f.tovPct!==null)state.needs.ballsecurity=scale(65-(f.tovPct-.145)*420);
 state.needs.defense=50;
 persist();renderNeeds();renderCandidateList();
}

function renderTeam(){if(!state.teams.nba.length)return;$("sc-nba-select").innerHTML=teamOptions("nba",state.nbaTeam);const team=state.teams.nba.find(t=>t.id===state.nbaTeam);const roster=state.rosters.nba.get(state.nbaTeam)||[];$("sc-nba-team-info").innerHTML='<div class="sc-teamlogo">'+(isPortrait(team?.logo)?'<img src="'+clean(team.logo)+'" alt="" onerror="this.remove()">':'NBA')+'</div><div><b>'+clean(team?.name||"NBA team")+'</b><small>Current roster · '+roster.length+' players via ESPN</small></div>';
 $("sc-nba-roster").innerHTML=roster.length?roster.slice(0,9).map(p=>'<div class="sc-rosterline">'+portrait(p)+'<div><b>'+clean(p.name)+'</b><small>'+clean(p.position||"NBA")+(p.age?" · age "+p.age:"")+'</small></div><span>#'+clean(p.jersey||"—")+'</span></div>').join(""):'<div class="sc-empty">Current roster not loaded.</div>';renderNeeds();observedFour();renderCandidateList()}
function candidatePlayers(){const all=new Map();for(const roster of state.rosters.college.values())for(const p of roster)all.set(keyOf(p),p);for(const k of state.shortlist){const p=findPlayer(k);if(p&&p.league==="college")all.set(k,p)}return [...all.values()]}
function fitScore(p){const rating=state.assessments[keyOf(p)];if(!rating||!needsKeys.some(k=>rating[k]!==undefined))return null;let weighted=0,w=0;for(const k of needsKeys){if(rating[k]===undefined)continue;const weight=Math.max(5,100-state.needs[k]);weighted+=weight*clamp(Number(rating[k]),0,100);w+=weight}return w?weighted/w:null}
function renderCandidateList(){if(!$("sc-candidate-list"))return;const list=candidatePlayers().map(p=>({p,score:fitScore(p)})).sort((a,b)=>(b.score??-1)-(a.score??-1));$("sc-candidate-list").innerHTML=list.length?list.slice(0,38).map(({p,score})=>'<div class="sc-candidate">'+portrait(p)+'<div class="sc-candidate-text"><b>'+clean(p.name)+'</b><small>'+clean(p.teamName||"NCAA")+" · "+clean(p.position)+'</small><span>'+ (score===null?"Not assessed · open player card":state.assessmentSources[keyOf(p)]==="box-score"?"ESPN box-score proxy fit":"User-rated planning fit")+'</span></div><strong class="sc-candidate-score">'+(score===null?"—":Math.round(score)+'/100')+'</strong><button data-act="detail" data-id="'+clean(keyOf(p))+'">Assess ↗</button></div>').join(""):'<div class="sc-empty">Visit a college roster in Player cards to populate this recruiting board.</div>'}
function bind(){
 document.querySelectorAll("[data-tab]").forEach(b=>b.addEventListener("click",()=>showTab(b.dataset.tab)));
 document.querySelectorAll("[data-league]").forEach(b=>b.addEventListener("click",()=>switchLeague(b.dataset.league)));
 $("sc-open-draft").addEventListener("click",()=>showTab("prospects"));$("sc-method-link").addEventListener("click",e=>{e.preventDefault();showTab("team")});
 $("sc-search").addEventListener("input",renderCards);$("sc-sort").addEventListener("change",renderCards);$("sc-refresh").addEventListener("click",()=>loadRoster(state.league,true));
 $("sc-team-pick").addEventListener("change",e=>{if(state.league==="college"){state.collegeTeam=e.target.value;$("sc-college-select").value=state.collegeTeam}else state.nbaTeam=e.target.value;loadRoster(state.league)});
 $("sc-college-select").addEventListener("change",e=>{state.collegeTeam=e.target.value;if(state.league==="college")$("sc-team-pick").value=state.collegeTeam;loadRoster("college")});
 $("sc-nba-select").addEventListener("change",e=>{state.nbaTeam=e.target.value;loadRoster("nba");loadTeamStats(state.nbaTeam)});
 $("sc-prospect-select").addEventListener("change",e=>{state.selectedProspect=(state.rosters.college.get(state.collegeTeam)||[]).find(p=>p.id===e.target.value)||null;renderDraft();if(state.selectedProspect)fetchStats(state.selectedProspect).then(()=>renderDraft())});
 $("sc-draft-pick").addEventListener("input",e=>{state.pick=clamp(Number(e.target.value),1,60);renderDraft()});
 $("sc-reset-needs").addEventListener("click",()=>{needsKeys.forEach(k=>state.needs[k]=50);persist();renderNeeds();renderCandidateList()});
 $("sc-load-candidates").addEventListener("click",()=>{showTab("players");switchLeague("college")});$("sc-clear-shortlist").addEventListener("click",()=>{state.shortlist=[];persist();renderCandidateList();renderCards()});
 $("sc-suggest-needs").addEventListener("click",suggestNeedsFromTeamStats);$("sc-close-modal").addEventListener("click",closeDetails);document.querySelectorAll("[data-close-modal]").forEach(x=>x.addEventListener("click",closeDetails));
 document.addEventListener("click",e=>{const b=e.target.closest("[data-act]");if(!b)return;const p=findPlayer(b.dataset.id);if(!p)return;if(b.dataset.act==="detail")openDetails(p);if(b.dataset.act==="auto-grade"){const ratings=gradeBoxScore(p);if(ratings){state.assessments[keyOf(p)]=ratings;state.assessmentSources[keyOf(p)]="box-score";if(!state.shortlist.includes(keyOf(p)))state.shortlist.push(keyOf(p));persist();renderDetails(p);renderCandidateList();renderCards()}}if(b.dataset.act==="shortlist")shortlistToggle(p);if(b.dataset.act==="draft"){state.selectedProspect=p;state.collegeTeam=[...state.rosters.college.entries()].find(([id,list])=>list.some(x=>x.id===p.id))?.[0]||state.collegeTeam;$("sc-college-select").value=state.collegeTeam;updateProspectOptions();closeDetails();showTab("prospects")}});
 document.addEventListener("input",e=>{const k=e.target.dataset?.need;if(k){state.needs[k]=Number(e.target.value);persist();$("sc-need-value-"+k).textContent=String(state.needs[k]);renderNeeds(true);renderCandidateList()}const assess=e.target.dataset?.assess;if(assess){const p=findPlayer(e.target.dataset.for);if(p){setAssessment(p,assess,e.target.value);const out=$("sc-assess-value-"+assess);if(out)out.textContent=e.target.value;}}});
 document.addEventListener("keydown",e=>{if(e.key==="Escape"&&!$("sc-player-modal").hidden)closeDetails();if(e.key==="/"&&!["INPUT","TEXTAREA","SELECT"].includes(document.activeElement?.tagName)){e.preventDefault();showTab("players");$("sc-search").focus()}});window.addEventListener("hashchange",()=>{const v=location.hash.slice(1);if(v&&v!==state.tab)showTab(v)});
}
async function init(){recover();bind();$("sc-date").textContent=new Date().toLocaleDateString("en-GB",{day:"numeric",month:"short",year:"numeric"});
 const results=await Promise.allSettled([fetch("/data/rookies.json").then(r=>r.json()),fetch("/data/model.json").then(r=>r.json()),fetch("/data/draft-path.json").then(r=>r.json()),loadTeams("college"),loadTeams("nba")]);
 if(results[0].status==="fulfilled")state.historical=results[0].value.map((p,i)=>({...p,id:String(p.year)+"-"+String(i),league:"historical",teamName:"NBA rookie archive",headshot:null}));
 if(results[1].status==="fulfilled")state.model=results[1].value;
 if(results[2].status==="fulfilled")state.draft=results[2].value;
 populateTeams();showTab(["players","prospects","team"].includes(location.hash.slice(1))?location.hash.slice(1):"players");
 await Promise.allSettled([loadRoster("college"),loadRoster("nba")]);renderDraft();renderTeam();renderCards();await loadTeamStats(state.nbaTeam);
}
init();
