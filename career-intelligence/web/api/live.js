export default async function handler(req,res){
 if(req.method!=="GET"){res.setHeader("Allow","GET");return res.status(405).json({ok:false,message:"Method not allowed"})}
 res.setHeader("Cache-Control","s-maxage=45, stale-while-revalidate=60");
 try{
  const url="https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json";
  const response=await fetch(url,{signal:AbortSignal.timeout(7500),headers:{"Accept":"application/json"}});
  if(!response.ok)throw new Error("Feed upstream HTTP "+response.status);
  const body=await response.json();
  const games=(body?.scoreboard?.games||[]).slice(0,25).map(g=>({
    id:String(g.gameId||""),status:Number(g.gameStatus||1),statusText:String(g.gameStatusText||""),
    timeUTC:g.gameTimeUTC||null,
    away:{code:String(g.awayTeam?.teamTricode||"—"),score:g.awayTeam?.score??null},
    home:{code:String(g.homeTeam?.teamTricode||"—"),score:g.homeTeam?.score??null}
  }));
  return res.status(200).json({ok:true,source:"NBA public liveData scoreboard",updatedAt:body?.scoreboard?.gameDate||null,games});
 }catch(err){
  return res.status(503).json({ok:false,message:"NBA live-data feed temporarily unavailable",games:[]});
 }
}