/** COURTLINE / ESPN read-only adapter.
 * Unofficial public data: rate limits, licensing and upstream availability apply.
 * Endpoint parameters are numeric ESPN IDs only; callers cannot choose a URL.
 */
const BASE="https://site.api.espn.com/apis/site/v2/sports/basketball/";
const WEB="https://site.web.api.espn.com/apis/common/v3/sports/basketball/";
const LEAGUES={college:"mens-college-basketball",nba:"nba"};
function season(){const now=new Date();return now.getUTCFullYear()+(now.getUTCMonth()>=6?1:0)}
async function request(url){
 const response=await fetch(url,{signal:AbortSignal.timeout(7000),headers:{"Accept":"application/json","User-Agent":"CourtlineResearch/1.2 (basketball scouting dashboard)"}});
 if(!response.ok)throw Error("Upstream ESPN "+response.status);
 return response.json();
}
function image(v){return typeof v==="string"&&/^https:\/\/[^/]*\.(?:espn\.com|espncdn\.com)\//i.test(v)?v:null}
function roster(root){
 const list=Array.isArray(root.athletes)?root.athletes:[];
 return list.flatMap(a=>Array.isArray(a.items)?a.items:[a]).map(wrapper=>{
  const p=wrapper.athlete||wrapper;
  return{id:String(p.id||""),name:String(p.displayName||p.fullName||p.name||""),position:String(p.position?.abbreviation||"—"),jersey:String(p.jersey||""),height:String(p.displayHeight||""),weight:String(p.displayWeight||""),classYear:String(p.experience?.abbreviation||p.year||""),age:Number.isFinite(Number(p.age))?Number(p.age):null,headshot:image(p.headshot?.href||p.headshot?.url||""),source:"ESPN roster",updatedAt:new Date().toISOString()};
 }).filter(p=>p.id&&p.name).slice(0,70);
}
function teams(raw){
 const list=raw.sports?.[0]?.leagues?.[0]?.teams||raw.teams||[];
 return list.map(t=>t.team||t).map(t=>({id:String(t.id||""),name:String(t.displayName||t.name||""),abbreviation:String(t.abbreviation||""),logo:image(t.logos?.[0]?.href||"")})).filter(t=>t.id&&t.name);
}
function numeric(s){const n=Number(s);return Number.isFinite(n)?n:null}
function teamNumbers(data){
 const entries=data?.results?.stats?.categories||data?.categories||[];
 const out={};
 for(const c of entries)for(const v of c.stats||[]){
  if(v.name&&numeric(v.value)!==null)out[v.name]=Number(v.value);
 }
 return out;
}
function fourFactors(stats){
 const fg=stats.fieldGoalsMade,fga=stats.fieldGoalsAttempted,three=stats.threePointFieldGoalsMade,
  tov=stats.turnovers,fta=stats.freeThrowsAttempted,ftm=stats.freeThrowsMade,gp=stats.gamesPlayed,oreb=stats.offensiveRebounds;
 if(!fga||!gp)return null;
 const rate=(numerator,denom)=>numerator!=null&&denom>0?Number((numerator/denom).toFixed(4)):null;
 return{
  efgPct:rate(fg+0.5*three,fga),
  tovPct:rate(tov,fga+0.44*fta+tov),
  ftmPerFga:rate(ftm,fga),
  offRebPerGame:rate(oreb,gp),
  games:gp,
  note:"OREB per game is a proxy only; actual offensive rebounding percentage needs opponent defensive rebounds."
 };
}
function playerAverages(json,year){
 const avg=(json.categories||[]).find(c=>c.name==="averages");
 if(!avg||!Array.isArray(avg.names))return{stats:{},sample:0};
 const selected=(avg.statistics||[]).find(row=>Number(row.season?.year)===year);
 if(!selected)return{stats:{},sample:0};
 const stats={};for(let i=0;i<avg.names.length;i++){
  const name=avg.names[i],display=selected.stats?.[i],n=numeric(display);
  if(n!==null)stats[name]=n;
  else if(typeof display==="string"&&/^\d+(?:\.\d+)?-\d+(?:\.\d+)?$/.test(display)){
   const parts=display.split("-").map(Number),names=name.split("-");
   if(names.length===2){stats[names[0]]=parts[0];stats[names[1]]=parts[1]}
  }
 }
 return{stats,sample:stats.gamesPlayed||0,seasonDisplay:selected.season?.displayName||String(year)};
}
export default async function handler(req,res){
 if(req.method!=="GET"){res.setHeader("Allow","GET");return res.status(405).json({ok:false,error:"Method not allowed"})}
 const {type,league="college",team,id}=req.query||{};
 if(!LEAGUES[league]||!["teams","roster","player","team-stats","games","history"].includes(type))return res.status(400).json({ok:false,error:"Invalid resource"});
 const teamId=String(team||""),athleteId=String(id||"");
 if(["roster","team-stats"].includes(type)&&!/^[0-9]{1,7}$/.test(teamId))return res.status(400).json({ok:false,error:"Invalid team ID"});
 if(["player","games","history"].includes(type)&&!/^[0-9]{1,10}$/.test(athleteId))return res.status(400).json({ok:false,error:"Invalid athlete ID"});
 const slug=LEAGUES[league],year=season();
 res.setHeader("Cache-Control","s-maxage=180, stale-while-revalidate=600");
 try{
  if(type==="teams"){const raw=await request(BASE+slug+"/teams?limit=500&group=50"),items=teams(raw);if(!items.length)throw Error("Empty team list");return res.status(200).json({ok:true,season:year,league,source:"ESPN",teams:items})}
  if(type==="roster"){const raw=await request(BASE+slug+"/teams/"+teamId+"/roster?season="+year),players=roster(raw);if(!players.length)throw Error("Empty roster");return res.status(200).json({ok:true,season:year,league,source:"ESPN",team:{id:teamId,name:String(raw.team?.displayName||raw.team?.name||"")},players,fetchedAt:new Date().toISOString()})}
  if(type==="team-stats"){
   const raw=await request(BASE+slug+"/teams/"+teamId+"/statistics?season="+year),stats=teamNumbers(raw);
   return res.status(200).json({ok:true,season:year,league,source:"ESPN",stats,fourFactors:league==="nba"?fourFactors(stats):null,fetchedAt:new Date().toISOString()});
  }
  if(type==="games"){
   const raw=await request(WEB+slug+"/athletes/"+athleteId+"/gamelog?season="+year);
   const names=Array.isArray(raw.names)?raw.names:[],index=name=>names.indexOf(name);
   const num=(values,name)=>{const i=index(name);if(i<0)return null;const v=Number(values?.[i]);return Number.isFinite(v)?v:null};
   const pair=(values,name,half)=>{
    const i=index(name),value=i<0?null:values?.[i];
    if(typeof value!=="string"||!/^\\d+(?:\\.\\d+)?-\\d+(?:\\.\\d+)?$/.test(value))return null;
    return Number(value.split("-")[half]);
   };

   const groups=(raw.seasonTypes||[]).flatMap(seas=>(seas.categories||[])
      .filter(group=>/regular season|regular season stats/i.test(seas.displayName||"")||/regular season|regular season stats/i.test(group.displayName||"")||String(group.splitType)==="2")
      .flatMap(group=>group.events||[]));
   const observed=groups.map(row=>{
      const event=raw.events?.[String(row.eventId)]||{},when=Date.parse(event.gameDate||event.date||"");
      if(!Number.isFinite(when)||when>Date.now()+86400000)return null;
      const stats=row.stats||[];
      return{id:String(row.eventId),date:new Date(when).toISOString(),opponent:String(event.opponent?.abbreviation||event.opponent?.displayName||"—"),
       result:String(event.gameResult||"—"),minutes:num(stats,"minutes"),points:num(stats,"points"),
       rebounds:num(stats,"totalRebounds")??num(stats,"rebounds"),assists:num(stats,"assists"),
       steals:num(stats,"steals"),blocks:num(stats,"blocks"),turnovers:num(stats,"turnovers"),threeMade:pair(stats,"threePointFieldGoalsMade-threePointFieldGoalsAttempted",0),threeAttempts:pair(stats,"threePointFieldGoalsMade-threePointFieldGoalsAttempted",1),freeThrowsMade:pair(stats,"freeThrowsMade-freeThrowsAttempted",0)};
   }).filter(row=>row&&row.points!==null).sort((a,b)=>Date.parse(a.date)-Date.parse(b.date));
   const unique=[...new Map(observed.map(v=>[v.id,v])).values()].slice(-110);
   return res.status(200).json({ok:true,id:athleteId,season:year,league,source:"ESPN game log",games:unique,sample:unique.length,fetchedAt:new Date().toISOString()});
  }
  if(type==="history"){
   const raw=await request(WEB+slug+"/athletes/"+athleteId+"/stats");
   const avg=(raw.categories||[]).find(c=>c.name==="averages"),list=(avg?.statistics||[]).map(row=>{
    const stats={};for(let i=0;i<(avg?.names||[]).length;i++){const value=Number(row.stats?.[i]);if(Number.isFinite(value)&&row.stats?.[i]!=="")stats[avg.names[i]]=value}
    return{year:Number(row.season?.year),label:String(row.season?.displayName||""),teamId:String(row.teamId||""),stats}
   }).filter(s=>Number.isFinite(s.year)).sort((a,b)=>a.year-b.year).slice(-25);
   return res.status(200).json({ok:true,id:athleteId,season:year,league,source:"ESPN",history:list});
  }
  const raw=await request(WEB+slug+"/athletes/"+athleteId+"/stats?season="+year),summary=playerAverages(raw,year);
  return res.status(200).json({ok:true,season:year,league,source:"ESPN",id:athleteId,...summary,fetchedAt:new Date().toISOString()});
 }catch(e){return res.status(503).json({ok:false,season:year,league,error:"ESPN data temporarily unavailable",reason:String(e?.message||e).slice(0,120)})}
}
export { fourFactors, playerAverages };
