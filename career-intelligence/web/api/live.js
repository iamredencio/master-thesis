/** Two independent read-only NBA scoreboard providers. No fabricated scores. */
function nbaClock(value){
 if(typeof value==="string"&&/^PT\\d+M\\d+(?:\\.\\d+)?S$/.test(value)){
  const m=value.match(/^PT(\\d+)M(\\d+)(?:\\.\\d+)?S$/);
  return String(m[1]).padStart(2,"0")+":"+String(m[2]).padStart(2,"0");
 }
 return typeof value==="string"&&/^\\d+:\\d{1,2}$/.test(value)?value:null;
}
const fmtET = () => {
 const parts = Object.fromEntries(new Intl.DateTimeFormat("en-US", {
   timeZone:"America/New_York",year:"numeric",month:"2-digit",day:"2-digit"
 }).formatToParts(new Date()).map(p=>[p.type,p.value]));
 return parts.year+parts.month+parts.day;
};
async function fetchJSON(url){
 const response=await fetch(url,{signal:AbortSignal.timeout(4500),headers:{"Accept":"application/json","User-Agent":"CourtlineResearch/1.0 (basketball data visualization)"}});
 if(!response.ok)throw new Error("Upstream HTTP "+response.status);
 return response.json();
}
function fromNBA(data){
 if(!data?.scoreboard?.games)throw Error("Unexpected NBA schema");
 return{source:"NBA liveData",games:data.scoreboard.games.slice(0,25).map(g=>({
 id:String(g.gameId||""),status:Number(g.gameStatus||1),statusText:String(g.gameStatusText||""),period:Number(g.period||0),clock:nbaClock(g.gameClock),
 timeUTC:g.gameTimeUTC||null,away:{code:String(g.awayTeam?.teamTricode||"—"),score:g.awayTeam?.score??null},
 home:{code:String(g.homeTeam?.teamTricode||"—"),score:g.homeTeam?.score??null}
 }))};
}
function fromESPN(data){
 if(!Array.isArray(data?.events))throw Error("Unexpected ESPN schema");
 return{source:"ESPN scoreboard",games:data.events.slice(0,25).map(event=>{
   const competition=event.competitions?.[0]||{};
   const teams=competition.competitors||[];
   const away=teams.find(team=>team.homeAway==="away")||{};
   const home=teams.find(team=>team.homeAway==="home")||{};
   const status=event.status?.type||{};
   const gameStatus=status.completed?3:status.state==="in"?2:1;
   const toTeam=t=>({code:String(t.team?.abbreviation||"—"),score:gameStatus===1?null:(t.score==null?null:Number(t.score))});
   return{id:String(event.id||""),status:gameStatus,period:Number(event.status?.period||0),clock:String(event.status?.displayClock||"")||null,
     statusText:String(status.shortDetail||status.detail||status.description||"Scheduled"),
     timeUTC:event.date||null,away:toTeam(away),home:toTeam(home)};
 })};
}
export default async function handler(req,res){
 if(req.method!=="GET"){res.setHeader("Allow","GET");return res.status(405).json({ok:false,message:"Method not allowed"})}
 res.setHeader("Cache-Control","s-maxage=30, stale-while-revalidate=45");
 const providers=[
  {url:"https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json",parse:fromNBA},
  {url:"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates="+fmtET()+"&limit=100",parse:fromESPN}
 ];
 for(const p of providers){
  try{
   const data=await fetchJSON(p.url);
   const normalized=p.parse(data);
   return res.status(200).json({ok:true,...normalized,updatedAt:new Date().toISOString()});
  }catch(err){/* Retry against next provider. No invented data. */}
 }
 return res.status(503).json({ok:false,message:"Live NBA score providers temporarily unavailable",games:[]});
}
