/**
 * Hoop Metrics / forecasting research 0.3.
 * Every probability here is EXPERIMENTAL, not calibrated or held-out validated.
 * This module never invents games, teams, injuries, or market availability.
 */
export const SKILL_DEFS={
  spacing:{label:"Three-point volume",unit:"3PM/G",delta:0.45,direction:1,teamLabel:"Three-point accuracy",metric:"threePointPct",teamUnit:"%",better:1},
  creation:{label:"Playmaking",unit:"AST/G",delta:0.75,direction:1,teamLabel:"Ball movement",metric:"avgAssists",teamUnit:"AST/G",better:1},
  defense:{label:"Defensive events",unit:"STL+BLK/G",delta:0.35,direction:1,teamLabel:"Defensive events (proxy)",metric:"defensiveEvents",teamUnit:"STL+BLK/G",better:1},
  rebounding:{label:"Rebounding",unit:"REB/G",delta:0.8,direction:1,teamLabel:"Offensive rebounding",metric:"avgOffensiveRebounds",teamUnit:"OREB/G",better:1},
  ballsecurity:{label:"Turnover reduction",unit:"TOV/G",delta:0.3,direction:-1,teamLabel:"Possession security",metric:"tovPct",teamUnit:"TOV%",better:-1},
  scoring:{label:"Scoring output",unit:"PTS/G",delta:1.6,direction:1,teamLabel:"Shooting efficiency",metric:"efgPct",teamUnit:"eFG%",better:1},
  rim:{label:"Rim protection",unit:"BLK/G",delta:0.22,direction:1,teamLabel:"Blocked shots (proxy)",metric:"avgBlocks",teamUnit:"BLK/G",better:1},
  pressure:{label:"Free-throw pressure",unit:"FTM/G",delta:0.5,direction:1,teamLabel:"Free-throw rate",metric:"ftmPerFga",teamUnit:"FTM/FGA",better:1}
};
export const CORE_SKILLS=["spacing","creation","defense","rebounding","ballsecurity","scoring","rim","pressure"];
const finite=(x)=>x!==null&&x!==undefined&&x!==""&&Number.isFinite(Number(x));
const n=x=>finite(x)?Number(x):null;
export function extractTeamMetrics(stats={},factors={}){
 const v=(key)=>n(stats[key]);
 const steals=v("avgSteals"),blocks=v("avgBlocks");
 return {
  threePointPct:v("threePointPct")??v("threePointFieldGoalPct"),
  avgAssists:v("avgAssists"),
  defensiveEvents:steals!==null&&blocks!==null?steals+blocks:null,
  avgOffensiveRebounds:v("avgOffensiveRebounds"),
  tovPct:n(factors?.tovPct)!==null?Number(factors.tovPct)*100:null,
  efgPct:n(factors?.efgPct)!==null?Number(factors.efgPct)*100:null,
  avgBlocks:blocks,
  ftmPerFga:v("fieldGoalsAttempted")&&v("freeThrowsMade")!==null?v("freeThrowsMade")/v("fieldGoalsAttempted"):n(factors?.ftmPerFga)
 };
}
export function summarizeBenchmarkRows(rows=[]){
 const valid=rows.filter(r=>r&&r.metrics&&typeof r.metrics==="object");
 const summaries={};
 for(const key of CORE_SKILLS){
  const def=SKILL_DEFS[key],values=valid.map(r=>n(r.metrics[def.metric])).filter(x=>x!==null).sort((a,b)=>a-b);
  if(values.length<12)continue;
  const median=values.length%2?(values[(values.length-1)/2]):(values[values.length/2-1]+values[values.length/2])/2;
  const mean=values.reduce((s,x)=>s+x,0)/values.length;
  const sd=Math.sqrt(values.reduce((s,x)=>s+(x-mean)**2,0)/Math.max(1,values.length-1));
  summaries[key]={median:Number(median.toFixed(4)),sd:Number(sd.toFixed(4)),observations:values.length};
 }
 return summaries;
}
export function computeTeamGaps(stats,factors,benchmarks){
 const observed=extractTeamMetrics(stats,factors),reference=benchmarks||{},gaps=[];
 for(const key of CORE_SKILLS){
  const def=SKILL_DEFS[key],b=reference[key],value=n(observed[def.metric]);
  if(value===null||!b||!finite(b.median)||!finite(b.sd)||b.observations<12)continue;
  const delta=def.better*(value-b.median),sd=Math.max(Math.abs(b.sd),0.001),z=delta/sd;
  const intensity=clamp(Math.round(Math.max(0,-z)*42),0,100);
  gaps.push({key,label:def.teamLabel,unit:def.teamUnit,current:value,median:b.median,sd:b.sd,z:Number(z.toFixed(3)),gap:intensity,status:z<-.4?"Below median":z>.4?"Above median":"Near median",observations:b.observations,needsAttention:z<-.4});
 }
 return gaps.sort((a,b)=>b.gap-a.gap);
}
const clamp=(x,lo,hi)=>Math.max(lo,Math.min(hi,x));
function average(vals){return vals.length?vals.reduce((a,b)=>a+b,0)/vals.length:null}
function variance(vals){const avg=average(vals);return vals.length>=2?vals.reduce((s,x)=>s+(x-avg)**2,0)/(vals.length-1):null}
// Standard normal CDF, Abramowitz–Stegun approximation, max absolute error ≈ 7.5e-8.
export function normalCdf(z){
 const x=Math.abs(Number(z));if(!Number.isFinite(x))return null;
 const t=1/(1+0.2316419*x);
 const density=Math.exp(-x*x/2)/Math.sqrt(2*Math.PI);
 const tail=density*(0.319381530*t-0.356563782*t*t+1.781477937*t**3-1.821255978*t**4+1.330274429*t**5);
 return clamp(z>=0?1-tail:tail,0,1);
}
export function skillGameValue(game,skill){
 if(!game)return null;
 const fields={spacing:["threeMade"],creation:["assists"],defense:["steals","blocks"],rebounding:["rebounds"],ballsecurity:["turnovers"],scoring:["points"],rim:["blocks"],pressure:["freeThrowsMade"]};
 const k=fields[skill];if(!k||!k.every(field=>finite(game[field])))return null;
 return k.reduce((sum,field)=>sum+Number(game[field]),0);
}
/**
 * Experimental next-10-game mean-threshold probability.
 * Likelihood has observed season game standard deviation, shrinkage to season mean,
 * and variance from forecasted average plus mean estimation uncertainty.
 * NOT fit or calibrated against future seasons; do not use for automated scouting decisions.
 */
export function forecastImprovement(games,skill,{window=10}={}){
 const def=SKILL_DEFS[skill];if(!def)return null;
 const rows=(Array.isArray(games)?games:[])
  .filter(g=>g&&Number.isFinite(Date.parse(g.date))&&Date.parse(g.date)<=Date.now())
  .sort((a,b)=>Date.parse(a.date)-Date.parse(b.date));
 const vals=rows.map(r=>skillGameValue(r,skill)).filter(finite);
 if(vals.length<6)return null;
 const recent=vals.slice(-Math.min(window,vals.length)),last=recent.slice(-Math.min(5,recent.length));
 const baseline=average(recent),seasonMean=average(vals),lastMean=average(last);
 const observedVar=variance(recent)??0;
 const empiricalSd=Math.max(Math.sqrt(observedVar),0.35);
 // Shrink recent trend toward observed season mean; cap half-season jumps in small data.
 const trend=clamp((lastMean-baseline)*0.45,-1.25*empiricalSd,1.25*empiricalSd);
 const predictedMean=baseline+trend;
 const threshold=baseline+def.direction*def.delta;
 const sdMean=Math.sqrt(empiricalSd**2/10+empiricalSd**2/Math.max(6,vals.length)+0.12**2);
 const z=def.direction*(predictedMean-threshold)/sdMean;
 const probability=clamp(normalCdf(z),.025,.975);
 return {skill,n:vals.length,metric:def.unit,baseline:Number(baseline.toFixed(3)),recentFive:Number(lastMean.toFixed(3)),predictedMean:Number(predictedMean.toFixed(3)),threshold:Number(threshold.toFixed(3)),probability:Number(probability.toFixed(4)),delta:def.delta,direction:def.direction,confidence:vals.length<12?"Low sample":"Provisional",horizonGames:10,method:"Shrinkage normal approximation, v0.3; not backtested or calibrated"};
}
export function scoreExternalSkill(stats,skill){
 if(!stats)return null;
 const values={
  spacing:[n(stats.avgThreePointFieldGoalsMade),4],
  creation:[n(stats.avgAssists),9],
  defense:[finite(stats.avgSteals)&&finite(stats.avgBlocks)?Number(stats.avgSteals)+Number(stats.avgBlocks):null,4],
  rebounding:[n(stats.avgRebounds),12],
  ballsecurity:[n(stats.avgTurnovers),4],
  scoring:[n(stats.avgPoints),30],
  rim:[n(stats.avgBlocks),3],
  pressure:[n(stats.avgFreeThrowsMade),7]
 };
 const pair=values[skill];if(!pair||pair[0]===null)return null;
 return Math.round(clamp(skill==="ballsecurity"?100*(1-pair[0]/pair[1]):100*pair[0]/pair[1],0,100));
}
export function liveForecast(game){
 if(!game)return null;
 const home=Number(game.home?.score),away=Number(game.away?.score);
 if(game.status===3&&finite(game.home?.score)&&finite(game.away?.score))return{probability:home===away?null:home>away?1:0,mode:"final",remainingMinutes:0};
 if(game.status!==2||!finite(game.home?.score)||!finite(game.away?.score))return null;
 const period=Number(game.period),clock=String(game.clock||"");
 const match=clock.match(/^(\d+):(\d{1,2})(?:\.\d+)?$/);
 if(!Number.isInteger(period)||period<1||period>12||!match)return null;
 const clockMin=Number(match[1])+Number(match[2])/60;
 if(clockMin<0||clockMin>12.01)return null;
 const remaining=period<=4?Math.max(0,(4-period)*12+clockMin):clockMin;
 if(remaining<=0)return null;
 // Brownian point spread approximation, neutral 50% prior; experimental and not calibrated.
 // 2.8 points/sqrt(minute) is a fixed hypothesis, not a fitted current-era sigma.
 const sigma=2.8*Math.sqrt(remaining)+0.4;
 const probability=clamp(normalCdf((home-away)/sigma),.01,.99);
 return{probability:Number(probability.toFixed(4)),mode:"experimental-live",remainingMinutes:Number(remaining.toFixed(2)),sigma:Number(sigma.toFixed(3)),margin:home-away,method:"Neutral-prior Brownian score-margin approximation, uncalibrated"};
}
export function validatedLiveText(game,forecast){
 if(!game||!game.home||!game.away)return null;
 const home=String(game.home.code||"HOME"),away=String(game.away.code||"AWAY");
 if(!/^[-A-Za-z0-9 ]{1,12}$/.test(home)||!/^[-A-Za-z0-9 ]{1,12}$/.test(away))return null;
 if(game.status===1)return home+" vs "+away+" — scheduled; no live probability before tip-off.";
 if(!finite(game.home.score)||!finite(game.away.score))return null;
 if(game.status===3)return home+" "+game.home.score+", "+away+" "+game.away.score+" — final score confirmed by the live feed.";
 if(!forecast||forecast.mode!=="experimental-live")return home+" "+game.home.score+"–"+away+" "+game.away.score+" — live; timing insufficient for a model estimate.";
 const winner=game.home.score===game.away.score?"The game is tied.":game.home.score>game.away.score?home+" leads by "+(game.home.score-game.away.score)+".":away+" leads by "+(game.away.score-game.home.score)+".";
 return home+" "+game.home.score+"–"+away+" "+game.away.score+". "+winner+" Experimental "+home+" win estimate: "+Math.round(forecast.probability*100)+"% (not calibrated).";
}
