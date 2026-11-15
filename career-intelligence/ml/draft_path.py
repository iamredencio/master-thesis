#!/usr/bin/env python3
"""Rebuild the historical draft-slot conditional career model.

From repository root:
  python career-intelligence/ml/draft_path.py --check
  python career-intelligence/ml/draft_path.py --write

The input population includes DRAFTED players only. Output cannot estimate
the chance a college player is drafted or the specific pick they will receive.
Career outcome = last NBA calendar year minus first NBA calendar year + one;
it is not unique games played, minutes, or seasons rostered.
"""
from __future__ import annotations
import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
DRAFT=ROOT/"master thesis/Deeplearning articles/data/gmoney-nba-drafts-2016-1989/gmoney-nba-drafts-2016-1989/data/nba_drafts.csv"
CAREER=ROOT/"nba-players-stats/player_data.csv"
OUTPUT=ROOT/"career-intelligence/web/data/draft-path.json"
BANDS=[("Top 5","1–5",1,5),("Lottery","6–14",6,14),("Late first","15–30",15,30),("Early second","31–45",31,45),("Late second","46–60",46,60)]
PRIOR_STRENGTH=14

def name_key(value):
    value=unicodedata.normalize("NFKD",str(value or "")).lower()
    return re.sub(r"[^a-z]","", "".join(c for c in value if not unicodedata.combining(c)))

def load():
    draft=pd.read_csv(DRAFT).fillna("")
    careers=pd.read_csv(CAREER).fillna("")
    by_name={}
    for p in careers.to_dict(orient="records"):
        by_name.setdefault(name_key(p["name"]),[]).append(p)
    matched=[]
    for d in draft.to_dict(orient="records"):
        found=by_name.get(name_key(d["player"]),[])
        if len(found)!=1: continue
        try:
            year=int(d["draft_year"]);pick=int(d["pick"])
            span=int(found[0]["year_end"])-int(found[0]["year_start"])+1
        except (TypeError,ValueError):continue
        if 1989<=year<=2012 and 1<=pick<=60 and 1<=span<=40:
            matched.append({"year":year,"pick":pick,"span":span})
    return matched, len(draft), len(careers)

def brier(rows, label, prediction):
    if not rows:raise ValueError("Empty cohort")
    return round(sum((prediction(r)-(r["span"]>=label))**2 for r in rows)/len(rows),4)

def estimate(rows,target,train_years,val_years,test_years):
    tr=[r for r in rows if train_years[0]<=r["year"]<=train_years[1]]
    val=[r for r in rows if val_years[0]<=r["year"]<=val_years[1]]
    test=[r for r in rows if test_years[0]<=r["year"]<=test_years[1]]
    prevalence=sum(r["span"]>=target for r in tr)/len(tr)
    bands=[]
    for name,rng,lo,hi in BANDS:
        group=[r for r in tr if lo<=r["pick"]<=hi]
        good=sum(r["span"]>=target for r in group)
        prob=(good+prevalence*PRIOR_STRENGTH)/(len(group)+PRIOR_STRENGTH)
        sorted_span=sorted(r["span"] for r in group)
        bands.append({"name":name,"range":rng,"low":lo,"high":hi,
                      "n":len(group),"positive":good,"prob":round(prob,4),
                      "median":sorted_span[len(sorted_span)//2]})
    def predict(row):
        return next(b["prob"] for b in bands if b["low"]<=row["pick"]<=b["high"])
    return {
        "target":target,
        "cohorts":{"train":f"{train_years[0]}–{train_years[1]}",
                   "validation":f"{val_years[0]}–{val_years[1]}",
                   "test":f"{test_years[0]}–{test_years[1]}"},
        "n":{"train":len(tr),"val":len(val),"test":len(test)},
        "baseline":round(prevalence,4),
        "bands":bands,
        "validation":{"brier":brier(val,target,predict),"baseline":brier(val,target,lambda _:prevalence)},
        "test":{"brier":brier(test,target,predict),"baseline":brier(test,target,lambda _:prevalence)}
    }

def make():
    rows, draft_size, career_size=load()
    return {
      "version":"0.1.0",
      "trainedAt":"2026-11-15",
      "description":"Historical draft-pick conditional NBA career-span rates, smoothed by cohort prevalence. Not draft probability; not college stats model.",
      "source":{"draft":"master thesis/Deeplearning articles/data/gmoney-nba-drafts-2016-1989/gmoney-nba-drafts-2016-1989/data/nba_drafts.csv",
                "career":"nba-players-stats/player_data.csv"},
      "cohortLimit":"Career source ends in 2018. Historic pick records end in 2016. Labels after safe horizons excluded.",
      "matched":len(rows),
      "horizons":{
        "five":estimate(rows,5,(1989,2004),(2005,2007),(2008,2011)),
        "ten":estimate(rows,10,(1989,2002),(2003,2004),(2005,2008))
      },
      "limitations":[
        "These are conditional on already being selected at this draft pick; they are not P(drafted).",
        "Five and ten years refer to calendar span (last recorded season year minus first plus one), not unique active seasons.",
        "Players without unambiguous names in both archives are excluded (selection bias).",
        "Historical draft cohort rates can differ substantially from current NBA roster dynamics.",
        "Scouting scenarios are conditional comparisons, not independently trained NCAA-to-NBA projections."
      ]
    }

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--write",action="store_true")
    args=parser.parse_args()
    result=make()
    print(json.dumps({"matched":result["matched"],"metrics":{k:v["test"] for k,v in result["horizons"].items()}},indent=2))
    if args.check:
        assert result["matched"]==1160, result["matched"]
        for key in ("five","ten"):
            section=result["horizons"][key]
            assert section["test"]["brier"]<section["test"]["baseline"]
        saved=json.loads(OUTPUT.read_text())
        assert saved["matched"]==result["matched"]
        for key in ("five","ten"):
            for name in ("brier","baseline"):
                assert abs(saved["horizons"][key]["test"][name]-result["horizons"][key]["test"][name])<.001
            for b_old,b_new in zip(saved["horizons"][key]["bands"],result["horizons"][key]["bands"]):
                assert b_old["n"]==b_new["n"]
                assert abs(b_old["prob"]-b_new["prob"])<.001
        print("Passed: as-of historical label window, held-out Brier and artifact consistency.")
    if args.write:
        OUTPUT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
        print("Wrote", OUTPUT)

if __name__=="__main__":
    main()
