/** Read-only current-season NBA league benchmark aggregator.
 * Computes league-relative medians ONLY from observed ESPN team statistics.
 * For partial data, reports coverage and refuses unsupported comparisons.
 */
import {extractTeamMetrics,summarizeBenchmarkRows} from "../forecast-engine.js";
import {fourFactors} from "./scout.js";
const BASE="https://site.api.espn.com/apis/site/v2/sports/basketball/nba/";
async function get(url,timeout=5300){
 const r=await fetch(url,{headers:{"Accept":"application/json","User-Agent":"HoopMetricsResearch/0.3"},signal:AbortSignal.timeout(timeout)});
 if(!r.ok)throw Error("ESPN upstream "+r.status);
 return r.json();
}
function season(){const d=new Date();return d.getUTCFullYear()+(d.getUTCMonth()>=6?1:0)}
function teamNumbers(data){
 const list=data?.results?.stats?.categories||data?.categories||[],out={};
 for(const category of list)for(const row of category.stats||[]){
  const value=Number(row.value);
  if(row.name&&row.value!==null&&row.value!==""&&Number.isFinite(value))out[row.name]=value;
 }
 return out;
}
async function parallel(items,worker,limit=6){
 let i=0;const result=new Array(items.length);
 await Promise.all(Array.from({length:Math.min(items.length,limit)},async()=>{
  while(i<items.length){const index=i++;try{result[index]=await worker(items[index])}catch(e){result[index]=null}}
 }));
 return result;
}
export default async function handler(req,res){
 if(req.method!=="GET"){res.setHeader("Allow","GET");return res.status(405).json({ok:false,error:"GET required"})}
 res.setHeader("Cache-Control","s-maxage=1800, stale-while-revalidate=1800");
 const year=season();
 try{
  const directory=await get(BASE+"teams?limit=100&group=50");
  const teams=(directory.sports?.[0]?.leagues?.[0]?.teams||[]).map(x=>x.team||x).filter(x=>/^\d+$/.test(String(x.id))).slice(0,40);
  if(teams.length<20)throw Error("Insufficient team directory");
  const rows=(await parallel(teams,async t=>{
   const raw=await get(BASE+"teams/"+t.id+"/statistics?season="+year);
   const stats=teamNumbers(raw);
   if(!Number.isFinite(stats.gamesPlayed)||stats.gamesPlayed<2)return null;
   const factors=fourFactors(stats),metrics=extractTeamMetrics(stats,factors);
   return {teamId:String(t.id),games:stats.gamesPlayed,metrics};
  })).filter(Boolean);
  const benchmarks=summarizeBenchmarkRows(rows);
  const covered=rows.length;
  // Coverage threshold protects league-level claims when upstream is degraded.
  if(covered<20||Object.keys(benchmarks).length<6)throw Error("Insufficient observed team coverage ("+covered+"/"+teams.length+")");
  return res.status(200).json({ok:true,season:year,source:"ESPN NBA team season totals",leagueSample:covered,totalTeams:teams.length,benchmarks,observedAt:new Date().toISOString(),note:"League medians computed from recorded NBA games. Per-game team metrics are pace sensitive. Limited early-season samples are volatile."});
 }catch(e){
  return res.status(503).json({ok:false,season:year,error:"League benchmarks temporarily unavailable",reason:String(e?.message||e).slice(0,140)});
 }
}
