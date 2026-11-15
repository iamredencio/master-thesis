import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {SKILL_DEFS,normalCdf,extractTeamMetrics,summarizeBenchmarkRows,computeTeamGaps,forecastImprovement,scoreExternalSkill,liveForecast,validatedLiveText} from "../forecast-engine.js";
test("normal CDF is symmetric, monotonic, and bounded",()=>{
 assert.ok(Math.abs(normalCdf(0)-.5)<1e-7);
 assert.ok(Math.abs(normalCdf(1)-.841345)<1e-4);
 assert.ok(Math.abs(normalCdf(-1)-.158655)<1e-4);
 assert.ok(normalCdf(2)>normalCdf(1));
});
test("team gap labels use observed team totals and genuine benchmark distributions",()=>{
 const stats={threePointPct:31,avgAssists:20,avgSteals:6,avgBlocks:3,avgOffensiveRebounds:10,fieldGoalsAttempted:900,freeThrowsMade:180};
 const f={tovPct:.16,efgPct:.5,ftmPerFga:.2};
 const metrics=extractTeamMetrics(stats,f);
 assert.equal(metrics.threePointPct,31);assert.equal(metrics.defensiveEvents,9);
 assert.equal(metrics.tovPct,16);assert.equal(metrics.efgPct,50);
 const rows=Array.from({length:30},(_,i)=>({teamId:String(i),metrics:{...metrics,threePointPct:35+i*.1,avgAssists:25+i*.1,
  defensiveEvents:13+i*.1,avgOffensiveRebounds:11+i*.1,tovPct:12+i*.1,efgPct:54+i*.1,avgBlocks:4+i*.1,ftmPerFga:.2+i*.001}}));
 const medians=summarizeBenchmarkRows(rows);assert.equal(medians.spacing.observations,30);
 const gaps=computeTeamGaps(stats,f,medians);
 assert.ok(gaps.length>=6);assert.equal(gaps[0].status,"Below median");
 assert.ok(gaps.some(g=>g.key==="spacing"&&g.gap>0));
 assert.ok(gaps.some(g=>g.key==="ballsecurity"&&g.gap>0));
});
test("improvement probability cannot be computed without six real game logs",()=>{
 const date="2026-10-01T12:00:00.000Z";
 assert.equal(forecastImprovement([{date,assists:2}],"creation"),null);
 assert.equal(forecastImprovement(Array(6).fill({date}),"creation"),null);
});
test("observed skill improvement estimates respond to trends but stay provisional",()=>{
 const date=i=>new Date(Date.UTC(2026,9,i+1)).toISOString();
 const rising=Array.from({length:12},(_,i)=>({date:date(i),assists:i<5?2:i<9?5:8,points:12+i,rebounds:4,turnovers:2,steals:1,blocks:1,threeMade:2,freeThrowsMade:3}));
 const falling=Array.from({length:12},(_,i)=>({...rising[i],assists:12-i}));
 const up=forecastImprovement(rising,"creation"),down=forecastImprovement(falling,"creation");
 assert.ok(up&&down);
 assert.ok(up.probability>down.probability);assert.ok(up.probability<=.975&&up.probability>=.025);
 assert.equal(up.method.includes("not backtested"),true);
 assert.equal(up.horizonGames,10);
});
test("external candidates use descriptive current-season skill ratings, not win forecasts",()=>{
 assert.ok(scoreExternalSkill({avgAssists:8},"creation")>scoreExternalSkill({avgAssists:3},"creation"));
 assert.equal(scoreExternalSkill({},"spacing"),null);
});
test("live estimate requires real period, clock and scores",()=>{
 const game={status:2,period:3,clock:"5:30",home:{code:"GS",score:91},away:{code:"LAL",score:85}};
 const estimate=liveForecast(game);
 assert.ok(estimate.probability>.5);
 assert.equal(estimate.mode,"experimental-live");
 assert.match(validatedLiveText(game,estimate),/not calibrated/i);
 assert.equal(liveForecast({...game,clock:null}),null);
 assert.equal(liveForecast({...game,status:1}),null);
 assert.equal(liveForecast({...game,home:{code:"GS",score:null}}),null);
 assert.equal(liveForecast({...game,status:3}).probability,1);
 assert.equal(liveForecast({...game,status:3,home:{code:"GS",score:81}}).probability,0);
});
test("new Research page uses Hoop Metrics look and retains original at the archive path",()=>{
 const root=new URL("../",import.meta.url),research=readFileSync(new URL("research.html",root),"utf8"),
 legacy=readFileSync(new URL("research-legacy.html",root),"utf8"),
 dash=readFileSync(new URL("scouting.html",root),"utf8");
 assert.match(research,/The Research/);
 assert.match(research,/research\\.css/);
 assert.match(legacy,/COURTLINE/);
 for(const id of ["team-skill-gaps","team-development-list","external-players","signal-games","team-priority-skill"]){
  assert.match(dash,new RegExp('id="'+id+'"'));
 }
 assert.match(dash,/EXPERIMENTAL PROBABILITIES/);
 assert.match(dash,/JEV-INSPIRED/);
});
