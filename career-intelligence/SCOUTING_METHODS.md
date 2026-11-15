# COURTLINE Scouting v0.2 — data and methods

## Current and historical populations

* **Current NCAA basketball:** ESPN 2026–27 college roster identifiers and player images, player season statistics per athlete from the ESPN web API. No invented athlete records. Player statistical sample sizes are shown and may be tiny early in the season.
* **Current NBA basketball:** ESPN 2026–27 league team catalogue, current roster and per-team season box-score totals; refresh depends on upstream cache and provider access.
* **Historic rookies:** thesis archived rookies (1980–2016), with historical 5+ calendar-year span model trained on pre-2008 cohorts.
* **Historical draft trajectory:** the repo's `nba_drafts.csv` (1989–2016) joined to `nba-players-stats/player_data.csv` (ends 2018). Names are normalized and ambiguous matches excluded, so sample-selection bias remains.

### Historical draft-position scenarios

Outcome: `end_nba_year - start_nba_year + 1 >= horizon` with horizon 5 or 10. To avoid temporal censoring, only cohorts whose outcome could be observed in the 2018 archive enter training/validation/test. Every draft band gets an empirical success rate with a cohort-prior shrinkage of 14:

`p_band = (success_in_training_band + 14 * success_prevalence_training) / (N_training_band + 14)`

Draft bands: picks 1–5, 6–14, 15–30, 31–45, 46–60. Their bands are **historical**, not an athlete-specific inference. The chosen pick is a user assumption; we do not calculate draft selection likelihood.

| Horizon | Training | Validation | Test | Test N | Brier | Cohort baseline Brier |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 5-year span | 1989–2004 | 2005–2007 | 2008–2011 | 201 | 0.1778 | 0.2239 |
| 10-year span | 1989–2002 | 2003–2004 | 2005–2008 | 199 | 0.2128 | 0.2391 |

Note: this table relies on calendar span, not unique seasons played. It does not establish modern career prediction accuracy and it excludes undrafted players.

### Observed NBA team features

We compute three actual basketball factors and one rebounding *proxy* from the selected current NBA team's season totals:

- `eFG% = (FGM + 0.5 * 3PM) / FGA`
- `TOV% = TOV / (FGA + 0.44 * FTA + TOV)`
- `Free throw rate = FTM / FGA` (Oliver/Basketball Reference makes-based definition, not attempt-based `FTA/FGA`)
- `Offensive rebounds per game = OREB / GP`, which is **not ORB%**. True ORB% requires opponent defensive rebounds, not currently available in the selected team endpoint.

The visual shows **editorial planning targets** (eFG 54%, TOV 14.5%, OREB/game 11.5, FTM/FGA 0.20), not calculated league averages. There is no claim that surpassing these targets is causally sufficient for wins. The historically popular Four Factors weight proportions (40 / 25 / 20 / 15) are context, not a fitted 2026 winning formula.

### Player scouting skill and team fit

The user rates each candidate on five 0–100 dimensions: spacing, creation, defensive activity, rebounding and ball security. Manual ratings are stored in browser session storage (never sent to the server). If the user explicitly activates box-score grading for a college athlete with at least three observed games, the UI seeds editable ratings with a visible heuristic calculated from per-game statistics converted to per-36-minute equivalents. The 0–100 rubric is an **unvalidated descriptive proxy**, not a normative estimate or causal player talent measure. It cannot account for opposition strength, role, injuries or professional league translation.

The illustrative fit score uses `sum((max(5, 100 - teamCoverage_i) * athleteRating_i)) / sum(max(5, 100 - teamCoverage_i))` across rated dimensions, where `teamCoverage_i` is the manual or scenario-derived scouting value. It is explicitly **not** a predicted win percentage, true roster marginal value, or salary efficiency.

## What is needed for professional predictive Moneyball

1. Stable athlete identity crosswalk across ESPN, NCAA, draft and NBA; training examples must include **eligible undrafted college players**, not only those drafted.
2. Licensed historical NCAA per-season stats, opposition-adjusted pace, college production with cohort exposure, international prospects and injury/availability context.
3. Right-censored competing-risk pathway model: draft, G League, overseas, NBA roster, rotation, starter; train and hold out **by later draft classes**.
4. Lineup on/off impact, player pair interactions, salary cap and team contracts to estimate marginal fit; no optimizer can accurately predict additional wins without lineup outcome data.
5. Multi-year league calibration, selection-bias audits, small-sample shrinkage and subgroup evaluation; feature provenance and model version displayed in every forecast.
6. Review data licenses and privacy before any production scouting or commercial release.

### Source links

- ESPN APIs explained: https://hoopr.sportsdataverse.org/articles/espn-endpoints.html
- NCAA/college historical stats resource: https://github.com/sportsdataverse/hoopR-mbb-data
- Four Factors definitions: https://www.basketball-reference.com/about/factors.html
- Lineup complementarity research: https://wsb.wharton.upenn.edu/wp-content/uploads/2023/12/Brill_2023_Q.pdf

Current public ESPN access is not a guarantee of commercial data rights, reliability or API stability. Images are loaded from ESPN CDN without copying to the repository.
