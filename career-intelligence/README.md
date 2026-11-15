# COURTLINE — basketball career intelligence

A professionally presented research application built on top of the original master's-thesis repository. The source thesis experiments and original CSVs remain unchanged.

## What is actually working

- **Trained model** for probability of an NBA career with a **calendar span of five or more years**, using first-rookie-season information only.
- **Time-split validation:** train 1980–2004 (988), validation 2005–2007 (120), held-out test 2008–2011 (172).
- **Held-out test:** ROC-AUC ~0.819; Brier ~0.149 vs prevalence-only baseline ~0.194.
- **1,538 searchable historical rookies** (1980–2016), with normalized data, position and approximate rookie age where available.
- **Interactive scouting profile**, player comparison, feature contributions, scenario analysis, and risk-banded historical career-span survival curves.
- **Live NBA scoreboard** from the NBA liveData CDN with an ESPN data fallback, refreshed every minute when available; reports errors instead of fabricated scores.
- Static frontend and one serverless endpoint; no API key needed for the historical application.

**Important:** The scoreboard is real-time; career predictions are **not** trained on live 2026 player-season data. It would be misleading to call the predictions themselves real-time until a new ingestion, backfill, training and drift-validation pipeline exists.

## Run

The app lives in `career-intelligence/web`. Configure this directory as the root in Vercel. It contains `index.html`, `app.js`, `styles.css`, `api/live.js`, `api/health.js`, and `data/*.json`.

For a quick local static UI check: `cd career-intelligence/web && python -m http.server 3000`. The static preview will not run `/api/live` locally; that function runs on Vercel.

### Retraining and checks

```bash
pip install -r career-intelligence/ml/requirements.txt
python career-intelligence/ml/train.py --check
python -m unittest discover -s career-intelligence/ml/tests -v
python career-intelligence/ml/train.py --write
```

## Prediction definition and limitations

- **Outcome:** last NBA year − first NBA year + 1 >= 5, based on the archive's `player_data.csv`. This is *career calendar-year span*, not 5 seasons played, not career salary, and not future PER.
- **Features:** first-season games, minutes per game, points, rebounds, assists, steals, blocks, turnovers, FG percentage, 3PA, FT percentage, efficiency, estimated rookie age.
- **No leakage:** chronology is fixed, normalization is learned from training only, duplicated-name joins are excluded, seasons after 2011 not used as 5-year labels, and no information from subsequent playing years appears in model inputs.
- **Validation:** L2 coefficient penalty chosen by validation Brier score, final evaluation is on untouched later cohorts. Dataset sample sizes are small and many names may not join cleanly.
- **Confidence:** percentage is an estimated probability, not a player-specific guarantee. Historical risk-band curves use *observed outcomes* from the training cohorts, not a second validated model. Input slider responses are associational sensitivity, not causal interventions.
- **Data coverage:** rookie records end 2016; career table ends 2018. NBA scoring and career environments have changed. Results should not inform hiring or financial decisions without more recent validation.

## Improving to a production-grade 2026 model

1. Ingest official/current player-season and per-game stats with licensed or authorized access, plus stable NBA person IDs. Do not merge identities by names alone.
2. Add injury availability, height/position changes, college/overseas histories, team context and draft pick where provenance is strong.
3. Train on *player-season rows with as-of timestamps*: (a) next-season per-game production and minutes conditional on continued participation; (b) yearly career-exit hazard with survival analysis and right-censoring; (c) conditional paths using lagged observations and hierarchical cohort effects.
4. Benchmark against persistence, cohort mean, survival Kaplan–Meier, regularized generalized linear models, gradient-boosted trees and sequence architectures where justified.
5. Hold out entire recent rookie cohorts and later calendar years; fit transformations and tune parameters **only within training**. Report calibration plots, horizon-specific Brier, MAE and interval coverage plus subgroup errors.
6. Deploy refreshed data and models as versioned artifacts, keep data as-of date, sources, audit trail and rollback. Retrain on a deliberately controlled cadence rather than every game.

## Provenance / resources

- Archived rookies: [data/used/nba_rookies_by_year.csv](../../data/used/nba_rookies_by_year.csv).
- Career start/end: [nba-players-stats/player_data.csv](../../nba-players-stats/player_data.csv).
- Previous experiments: `LSTM basketball data.py`, `SVR linear test.py`, `svr_test_rfe.py`, `feature_Extraction_pca.py`, notebooks and archived outputs.
- Optional current basketball integration: [nba_api](https://github.com/swar/nba_api) wraps NBA.com stats/live endpoints; upstream availability and terms must be verified.
- Commercial current player statistics: [BALLDONTLIE API](https://docs.balldontlie.io/) — season averages require a paid plan and API key; free plan does not suffice.
- Live games: public [NBA CDN liveData scoreboard](https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json), with fallback to [ESPN scoreboard](https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard) when the CDN rejects serverless requests. Both feeds are third-party interfaces; uptime, usage rights and commercial redistribution are not guaranteed.

The software is intended for research and visualization. Before public commercialization, review the provenance, copyright and API terms for the original archived datasets and all live feeds.

## Scouting expansion: Player cards, NCAA drafts and team fit

- [Scouting workspace](web/scouting.html): **Player cards**, **Draft lab** and **Team fit lab**. The original overview remains at /.
- Player cards use current **ESPN 2026–27 rosters and portraits** (via /api/scout) for NCAA and NBA. Individual statistics are fetched when a player is inspected. Archived rookie cards remain available and are not mixed with current players' predictions.
- Draft path shows empirical NBA career-span probabilities **conditional on a user-chosen draft pick**, using only players drafted in the thesis' 1989–2016 archive. It does **not** estimate whether a student will be drafted, current scouting rank, draft salary or NBA success from NCAA stats.
- Draft-path reproducibility: `python career-intelligence/ml/draft_path.py --check`; output `web/data/draft-path.json`; year-separated held-out tests for the 5-year and 10-year targets.
- The team fit lab loads the selected NBA team's current roster and measured box-score team totals. It computes eFG%, turnover rate, FT makes per FGA, and *offensive rebounds per game* (not true offensive rebounding percentage). Compare with clearly labeled planning targets, and optionally seed editable priorities from measured team stats. The app does not claim to predict wins.
- Candidate skills can be assessed manually or populated from an **explicit opt-in box-score proxy** using observed current-season NCAA PPG, RPG, APG, STL, BLK, TO and 3PT attempts/percentage. Ratings are a transparent heuristic, not a calibrated NCAA-to-NBA conversion. The fit index weights user assessed skills by user set roster needs.
- Data provider: public ESPN site APIs via official-looking but **unofficial undocumented access**, subject to provider restrictions. Commercial redistribution, especially player headshots, must be licensed/reviewed before public launch. Data is served to the authenticated Vercel preview and may not be permanently available.

See [SCOUTING_METHODS.md](SCOUTING_METHODS.md) for data methodology, provenance and known limitations.

## Hoop Metrics — screenshot-led redesign (November 2026)

The six screenshot references define the design direction: cinematic black hero with dark athlete portraits, yellow headline and CTA; dark blue-black dashboard background, purple/indigo panels, rounded controls, selectable player portrait chips, yellow observed performance chart, six numeric season cards, advanced/defensive ring metrics, and a two-column NBA Moneyball analysis.

### Routes

| Route | Purpose |
| --- | --- |
| `/` | Screenshot-led Hoop Metrics landing page |
| `/scouting.html#players` | Player explorer: NBA / NCAA / archived rookies, search, observed stats, chart, highlights |
| `/scouting.html#team` | Team analytics: real 2026–27 NBA roster & Four Factors, scouting candidate ratings |
| `/scouting.html#prospects` | NCAA-to-draft conditional *historical scenario* lab |
| `/research.html` | Original master's thesis research dashboard, fully preserved |
| `/scouting-classic.html` | Original v0.2 COURTLINE scouting prototype, fully preserved |

The layout is designed for desktop and mobile and contains no illustrative/fabricated box scores. The API `/api/scout` now supports `games` (ESPN regular-season athlete game log, current season) and `history` (historical player season averages). The front end never extrapolates future games or treats draft-slot conditional estimates as player-specific selection odds.

The gallery uses ESPN remote athlete images, **not copies of the user's screenshots as backgrounds**. Source headshot image licenses and API access terms require review before commercial public release.

### Verification

The GitHub Actions workflow runs reproducible Python model checks, Python tests, JS syntax validation and Node tests for the dashboard and the classic scouting API. Preview and production deployments must additionally smoke-test a real NBA player's game log, a real NCAA player's game log, an NBA team's current stats and all three application routes.
