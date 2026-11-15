import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {fourFactors,playerAverages} from "../api/scout.js";
test("Four Factors computes exact metrics and flags OREB per game as proxy",()=>{
 const r=fourFactors({fieldGoalsMade:40,fieldGoalsAttempted:80,threePointFieldGoalsMade:10,turnovers:12,
   freeThrowsAttempted:20,freeThrowsMade:16,offensiveRebounds:100,gamesPlayed:10});
 assert.ok(r);
 assert.equal(r.efgPct,0.5625);
 assert.equal(r.tovPct,0.119);
 assert.equal(r.ftmPerFga,0.2);
 assert.equal(r.offRebPerGame,10);
 assert.match(r.note,/proxy/i);
});
test("Player averages uses only requested season and maps names to numbers",()=>{
 const data={categories:[{name:"averages",names:["gamesPlayed","avgPoints","avgThreePointFieldGoalsMade-avgThreePointFieldGoalsAttempted"],
 statistics:[{season:{year:2026},stats:["31","11.2","1.0-3.0"]},
 {season:{year:2027,displayName:"2026-27"},stats:["4","15.5","2.0-5.0"]}]}]};
 const r=playerAverages(data,2027);
 assert.equal(r.seasonDisplay,"2026-27");
 assert.equal(r.stats.gamesPlayed,4);
 assert.equal(r.stats.avgPoints,15.5);
 assert.equal(r.stats.avgThreePointFieldGoalsMade,2);
 assert.equal(r.stats.avgThreePointFieldGoalsAttempted,5);
 assert.deepEqual(playerAverages(data,2028).stats,{});
});
test("Classic scouting UI retained with evidence boundaries",()=>{
 const source=readFileSync(new URL("../scouting-classic.html",import.meta.url),"utf8");
 for(const id of ["sc-player-grid","sc-prob-five","sc-prob-ten","sc-observed-four","sc-need-sliders","sc-candidate-list"]){
  assert.match(source,new RegExp('id="'+id+'"'));
 }
 assert.match(source,/NOT DRAFT ODDS/);
 assert.match(source,/NOT A WIN FORECAST/);
});
