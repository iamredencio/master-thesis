#!/usr/bin/env python3
"""Reproducible, time-split NBA rookie longevity study.

Run from repository root:
    pip install -r career-intelligence/ml/requirements.txt
    python career-intelligence/ml/train.py --check
    python career-intelligence/ml/train.py --write

No inference-time API keys; training uses original immutable thesis CSVs.
Labels mean >=5 calendar years from first to last recorded NBA year, NOT
five distinct played seasons. All records after rookie cohort 2011 censored.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO / "career-intelligence/web/data/model.json"
ROOKIES_PATH = REPO / "data/used/nba_rookies_by_year.csv"
CAREERS_PATH = REPO / "nba-players-stats/player_data.csv"
FEATURES = ["min", "pts", "gp", "reb", "ast", "stl", "blk", "tov",
            "fg", "3pa", "ft", "eff", "age"]
LAMBDAS = (0.002, 0.01, 0.025, 0.05, 0.1, 0.25)
TRAIN_END = 2004
VALID_END = 2007
LABEL_END = 2011
CLIP_Z = 5.0

def normalize_name(value: str) -> str:
    ascii_text = unicodedata.normalize("NFD", str(value or ""))
    return re.sub(r"[^a-z]", "", "".join(
        c for c in ascii_text.lower() if not unicodedata.combining(c)))

def number(value) -> float:
    try:
        n = float(value)
        return n if np.isfinite(n) else 0.0
    except (TypeError, ValueError):
        return 0.0

def rookie_age(birth_value, rookie_year: int) -> int:
    try:
        birth = datetime.strptime(str(birth_value), "%B %d, %Y")
        return max(17, min(38, rookie_year + 1 - birth.year - (1 if birth.month > 7 else 0)))
    except (TypeError, ValueError):
        return 23

def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    rookies = pd.read_csv(ROOKIES_PATH).fillna("")
    careers = pd.read_csv(CAREERS_PATH).fillna("")
    career_map: dict[str, list[dict]] = {}
    for record in careers.to_dict(orient="records"):
        career_map.setdefault(normalize_name(record["name"]), []).append(record)
    features, labels, years, catalog = [], [], [], []
    for r in rookies.to_dict(orient="records"):
        year = int(number(r["year_drafted"]))
        candidate = career_map.get(normalize_name(r["name"]), [])
        career = candidate[0] if len(candidate) == 1 else None
        age = rookie_age(career["birth_date"], year) if career is not None else 23
        vector = [float(age) if key == "age" else number(r.get(key)) for key in FEATURES]
        span = (int(number(career["year_end"])) - int(number(career["year_start"])) + 1) if career else 0
        valid = 1 <= span <= 45
        catalog.append({
            "name": str(r["name"]), "year": year,
            "position": career["position"] if career is not None else "N/A",
            "rookie": {
                "gp": number(r["gp"]), "min": number(r["min"]),
                "pts": number(r["pts"]), "reb": number(r["reb"]),
                "ast": number(r["ast"]), "stl": number(r["stl"]),
                "blk": number(r["blk"]), "tov": number(r["tov"]),
                "fg": number(r["fg"]), "threePa": number(r["3pa"]),
                "ft": number(r["ft"]), "eff": number(r["eff"]), "age": age
            },
            "observedFive": (span >= 5) if valid and year <= LABEL_END else None,
        })
        # Do not pretend late-cohort career outcomes are known in the 2018 snapshot.
        if valid and 1980 <= year <= LABEL_END:
            features.append(vector)
            labels.append(int(span >= 5))
            years.append(year)
    return (np.asarray(features, dtype=float),
            np.asarray(labels, dtype=int),
            np.asarray(years, dtype=int), catalog)

def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -25, 25)))

def scale(x: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return np.clip((x - means) / stds, -CLIP_Z, CLIP_Z)

def train_logistic(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[float, np.ndarray]:
    weights = np.zeros(X.shape[1], dtype=float)
    bias = 0.0
    for step in range(1600):
        error = sigmoid(bias + X @ weights) - y
        rate = .2 / (1 + step / 450)
        bias -= rate * error.mean()
        weights -= rate * (X.T @ error / len(y) + lam * weights)
    return bias, weights

def evaluate(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "n": len(y),
        "auc": round(float(roc_auc_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "logloss": round(float(log_loss(y, p)), 4),
    }

def run() -> tuple[dict, list[dict]]:
    X, y, year, catalog = load_data()
    a = year <= TRAIN_END
    b = (year > TRAIN_END) & (year <= VALID_END)
    c = (year > VALID_END) & (year <= LABEL_END)
    assert a.sum() and b.sum() and c.sum()
    assert not np.any(a & b) and not np.any(b & c)
    assert len(X) == len(y) == len(year)
    means = X[a].mean(axis=0)
    stds = np.maximum(.05, X[a].std(axis=0))
    Xscaled = scale(X, means, stds)
    best, best_val = None, float("inf")
    for lam in LAMBDAS:
        bias, coef = train_logistic(Xscaled[a], y[a], lam)
        validation = evaluate(y[b], sigmoid(bias + Xscaled[b] @ coef))
        if validation["brier"] < best_val:
            best, best_val = (lam, bias, coef, validation), validation["brier"]
    lam, bias, coef, validation = best
    predicted = sigmoid(bias + Xscaled @ coef)
    baseline = float(y[a].mean())
    # Survival curves describe *observed* training-cohort survival by risk
    # quartile; they are not independent multi-horizon ML predictions.
    train_rows = np.where(a)[0]
    ranked = train_rows[np.argsort(predicted[a])]
    # Career duration lookup is explicitly independent from the predictor inputs.
    by_name = pd.read_csv(CAREERS_PATH).fillna("")
    archived = pd.read_csv(ROOKIES_PATH).fillna("")
    duration_map = {}
    career_by_name = {}
    for r in by_name.to_dict(orient="records"):
        career_by_name.setdefault(normalize_name(r["name"]), []).append(r)
    # rebuild durations in EXACT same filtered sample order
    durations = []
    for r in archived.to_dict(orient="records"):
        yr = int(number(r["year_drafted"]))
        matches = career_by_name.get(normalize_name(r["name"]), [])
        if len(matches) != 1 or not 1980 <= yr <= LABEL_END:
            continue
        length = int(number(matches[0]["year_end"])) - int(number(matches[0]["year_start"])) + 1
        if 1 <= length <= 45:
            durations.append(length)
    duration = np.asarray(durations, dtype=int)
    assert len(duration) == len(X)
    bands = []
    for band in range(4):
        ids = ranked[int(band*len(ranked)/4):int((band+1)*len(ranked)/4)]
        bands.append({
            "band": band, "n": len(ids),
            "pMin": round(float(predicted[ids].min()), 4),
            "pMax": round(float(predicted[ids].max()), 4),
            "curve": [round(float(np.mean(duration[ids] >= horizon)), 4)
                      for horizon in range(1, 11)],
        })
    model = {
        "version": "1.0.0",
        "generatedAt": datetime.utcnow().date().isoformat(),
        "modelType": "L2-regularized logistic regression",
        "target": "NBA career year span >= 5 (last - first + 1)",
        "trainingYears": "1980–2004", "validationYears": "2005–2007",
        "testYears": "2008–2011", "archiveEnd": 2018,
        "lambda": lam,
        "featureNames": FEATURES,
        "mean": means.tolist(), "std": stds.tolist(),
        "coef": coef.tolist(), "intercept": float(bias),
        "clipZ": CLIP_Z, "survival": bands,
        "metrics": {
            "validation": validation,
            "heldOut": evaluate(y[c], predicted[c]),
            "baselineHeldOut": evaluate(y[c], np.full(c.sum(), baseline)),
            "baselineProbability": baseline,
        },
        "rowCounts": {
            "rookieRecords": len(catalog),
            "careerRecords": len(by_name),
            "matchedTotal": len(X),
            "train": int(a.sum()), "validation": int(b.sum()), "test": int(c.sum()),
        },
        "limitations": [
            "Archived rookie records end in 2016; career records end in 2018.",
            "Outcome is career calendar-year span, not distinct seasons played.",
            "Future cohorts were excluded to avoid censored labels.",
            "Name normalization matching can exclude/ambiguously join some players.",
            "Historical out-of-time validation, not 2026 performance proof.",
            "Coefficients are associational, not causal.",
        ],
    }
    return model, catalog

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Enforce no-leakage, held-out improvement, and saved model consistency.")
    parser.add_argument("--write", action="store_true", help="Regenerate deployed JSON assets.")
    args = parser.parse_args()
    model, catalog = run()
    print(json.dumps({"metrics": model["metrics"], "cohorts": model["rowCounts"]}, indent=2))
    if args.check:
        assert model["rowCounts"]["matchedTotal"] == 1280
        assert model["rowCounts"]["test"] == 172
        assert model["metrics"]["heldOut"]["auc"] > .7
        assert model["metrics"]["heldOut"]["brier"] < model["metrics"]["baselineHeldOut"]["brier"]
        saved = json.loads(MODEL_PATH.read_text())
        assert abs(saved["metrics"]["heldOut"]["auc"] - model["metrics"]["heldOut"]["auc"]) < .01
        assert abs(saved["metrics"]["heldOut"]["brier"] - model["metrics"]["heldOut"]["brier"]) < .01
        print("CHECK PASSED: time-safe evaluation and model artifact consistency.")
    if args.write:
        out = REPO / "career-intelligence/web/data"
        out.mkdir(parents=True, exist_ok=True)
        (out / "model.json").write_text(json.dumps(model, indent=2) + "\n")
        (out / "rookies.json").write_text(json.dumps(catalog, separators=(",",":")) + "\n")
        print("Updated:", out)

if __name__ == "__main__":
    main()
