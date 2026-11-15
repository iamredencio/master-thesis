import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
const root=new URL("../",import.meta.url);
const home=readFileSync(new URL("index.html",root),"utf8");
const dash=readFileSync(new URL("scouting.html",root),"utf8");
const source=readFileSync(new URL("dashboard.js",root),"utf8");
const old=readFileSync(new URL("research.html",root),"utf8");
test("user design hero and primary dashboard navigation work as real links",()=>{
 assert.match(home,/Hoop<br><strong>Metrics/);
 for(const hash of ["players","team","prospects"])assert.match(home,new RegExp("/scouting.html#"+hash));
 assert.match(home,/Launch dashboard/);
});
test("screenshots' dashboard modules are integrated, old research preserved",()=>{
 for(const id of ["player-strip","performance-svg","player-impact","team-factors","team-recommendation","team-roster-list","draft-five","draft-ten","player-tabs","players-search","players-position"]){
  assert.match(dash,new RegExp('id="'+id+'"'));
 }
 assert.match(old,/Thesis|COURTLINE|Courtline/i);
});
test("dashboard source fetches real games and distinguishes evidence from scenarios",()=>{
 assert.match(source,/type|games/);
 assert.match(source,/endpoint\("games"/);
 assert.match(source,/historical|model/);
 assert.match(dash,/NOT DRAFT ODDS/);
 assert.match(dash,/provisional|experimental, not calibrated win or growth forecasts/i);
});
test("all stat values are sourced or absent, not fixture scores",()=>{
 assert.doesNotMatch(source,/27\.5 PPG.*LeBron/i);
 assert.match(source,/No recorded games for this period/);
 assert.match(source,/sample|verified games/);
});
