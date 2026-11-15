/** CURATED ESPN scouting adapter. Unofficial public ESPN endpoints; read-only.
 * Only known numeric ESPN IDs are accepted; no arbitrary remote URLs.
 * A rate-limited upstream can return a transparent error with no fabricated data.
 */
const BASE="https://site.api.espn.com/apis/site/v2/sports/basketball/";
const WEB="https://site.web.api.espn.com/apis/common/v3/sports/basketball/";
const leagues={college:"mens-college-basketball",nba:"nba"};
function currentSeason(){const n=new Date();return n.getUTCFullYear()+(n.getUTCMonth()>=6?1:0)}
async function readJson(url){
 const response=await fetch(url,{signal:AbortSignal.timeout(6500),headers:{"Accept":"application/json","User-Agent":"CourtlineResearch/1.2 (NBA research dashboard)"}});
 if(!response.ok)throw new Error("ESPN HTTP "+response.status);
 return response.json();
}
function safeImg(v){return typeof v==="string"&&/^https:\/\/[^/]*\.(?:espn\.com|espncdn\.com)\//i.test(v)?v:null}
function safeAthlete(a){
 const p=a.athlete||a;const statistics=Array.isArray(a.statistics)?a.statistics:[];
 return{id:String(p.id||""),name:String(p.displayName||p.fullName||p.name||""),teamName:String(p.team?.displayName||""),position:String(p.position?.abbreviation||p.position?.name||"—"),jersey:String(p.jersey||""),height:String(p.displayHeight||""),weight:String(p.displayWeight||""),classYear:String(p.experience?.abbreviation||p.experience?.displayValue||p.year||""),age:Number.isFinite(Number(p.age))?Number(p.age):null,headshot:safeImg(p.headshot?.href||p.headshot?.url||""),stats:statistics.slice(0,30),teamId:String(p.team?.id||""),source:"ESPN roster",updatedAt:new Date().toISOString()}
}
function flattenRoster(root){
 const list=Array.isArray(root.athletes)?root.athletes:[];
 return list.flatMap(v=>Array.isArray(v.items)?v.items:[v]).map(safeAthlete).filter(v=>v.id&&v.name).slice(0,60);
}
function flattenTeams(json){
 const groups=json.sports?.[0]?.leagues?.[0]?.teams||json.teams||[];
 return groups.map(t=>t.team||t).map(t=>({id:String(t.id||""),name:String(t.displayName||t.name||""),abbreviation:String(t.abbreviation||""),logo:safeImg(t.logos?.[0]?.href||"")})).filter(t=>t.id&&t.name);
}
function statRows(value,out=[],depth=0){
 if(depth>9||!value)return out;
 if(Array.isArray(value)){for(const v of value.slice(0,600))statRows(v,out,depth+1);return out}
 if(typeof value!=="object")return out;
 if(typeof value.name==="string"&&(typeof value.value==="number"||typeof value.displayValue==="string")&&out.length<550){
  out.push({name:value.name,label:String(value.displayName||value.label||value.name),value:Number.isFinite(Number(value.value))?Number(value.value):Number(value.displayValue),display:String(value.displayValue??value.value)});
 }
 for(const k of ["statistics","stats","categories","splits","totals","averages","entries","groups","items","season","regularSeason"]){
  if(value[k]!=null)statRows(value[k],out,depth+1);
 }
 return out;
}
function extractStats(json){
 const result={};
 for(const s of statRows(json)){
  const key=String(s.name).toLowerCase().replace(/[^a-z0-9]/g,"");
  if(!key||!Number.isFinite(s.value))continue;
  if(result[key]===undefined)result[key]=s.value;
 }
 return result;
}
export default async function handler(req,res){
 if(req.method!=="GET"){res.setHeader("Allow","GET");return res.status(405).json({ok:false,error:"Method not allowed"})}
 res.setHeader("Cache-Control","s-maxage=360, stale-while-revalidate=1200");
 const {type,league="college",team,id}=req.query||{};
 if(!leagues[league]||!["teams","roster","player","team-stats"].includes(type))return res.status(400).json({ok:false,error:"Invalid resource"});
 const teamId=String(team||""),athleteId=String(id||"");
 if(["roster","team-stats"].includes(type)&&!/^[0-9]{1,7}$/.test(teamId))return res.status(400).json({ok:false,error:"Invalid team id"});
 if(type==="player"&&!/^[0-9]{1,10}$/.test(athleteId))return res.status(400).json({ok:false,error:"Invalid athlete id"});
 const slug=leagues[league],season=currentSeason();
 try{
  if(type==="teams"){
   const data=await readJson(BASE+slug+"/teams?limit=500&group=50");
   const teams=flattenTeams(data);
   if(!teams.length)throw new Error("No team catalogue returned");
   return res.status(200).json({ok:true,league,season,teams,source:"ESPN"});
  }
  if(type==="roster"){
   const data=await readJson(BASE+slug+"/teams/"+teamId+"/roster?season="+season);
   const players=flattenRoster(data);
   if(!players.length)throw new Error("Roster is empty");
   return res.status(200).json({ok:true,league,season,team:{id:teamId,name:String(data.team?.displayName||data.team?.name||"")},players,source:"ESPN",fetchedAt:new Date().toISOString()});
  }
  if(type==="team-stats"){
   const data=await readJson(BASE+slug+"/teams/"+teamId+"/statistics?season="+season);
   if(req.query.inspect==="1")return res.status(200).json({ok:true,keys:Object.keys(data),peek:JSON.stringify(data).slice(0,22000)});\n   const stats=extractStats(data);
   return res.status(200).json({ok:true,league,season,stats,availableKeys:Object.keys(stats).slice(0,110),source:"ESPN",fetchedAt:new Date().toISOString()});
  }
  const data=await readJson(WEB+slug+"/athletes/"+athleteId+"/stats?season="+season);
  if(req.query.inspect==="1")return res.status(200).json({ok:true,keys:Object.keys(data), statsPreview:JSON.stringify(data.statistics||data.stats||data.splits||data.categories||{}).slice(0,24000), subObjects:Object.keys(data).filter(k=>k!=="teams").slice(0,80).map(k=>({key:k,type:typeof data[k],preview:JSON.stringify(data[k]).slice(0,400)}))});
  const stats=extractStats(data);
  return res.status(200).json({ok:true,id:athleteId,league,season,stats,availableKeys:Object.keys(stats).slice(0,100),source:"ESPN",fetchedAt:new Date().toISOString()});
 }catch(err){return res.status(503).json({ok:false,league,season,error:"ESPN data unavailable",reason:String(err?.message||err).slice(0,130)})}
}
