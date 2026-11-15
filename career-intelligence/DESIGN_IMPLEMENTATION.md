# Hoop Metrics design implementation — six visual references

**Design input:** Six user-created screenshots supplied 15 November 2026 at ~03:17. The screenshots describe appearance and screen organization; they are not embedded as flattened images because the user needs a functional app.

| Reference | Implemented screen |
| --- | --- |
| Screenshot 1: filters, portrait player chips, overview & yellow chart | `scouting.html#players`: search, team, position, minimum box-score stats, chart and player profiles |
| Screenshot 2: advanced ring metrics and player/team diagnostics | Selected-player advanced box-score evidence, source-aware strength and gap cards |
| Screenshot 3: NBA roster, strengths/needs, recommended addition | `scouting.html#team`: real live roster, observed team efficiency, scored scouting shortlist |
| Screenshot 4: dark photo-collage hero, yellow Hoop Metrics type, feature tiles | `index.html`: visually faithful responsive landing and dashboard CTA |
| Screenshot 5: detailed player tabs, points/assists/rebounds, 1M/3M/6M/ALL | `scouting.html#players` with NBA and NCAA recorded game logs |
| Screenshot 6: highlights cards (high points, average, efficiency, GP) | Highlights player tab, measured season performance and games played |

## Visual tokens

- **Background:** `#070a16` player dashboard, `#111115` marketing landing
- **Panel:** `#1b1b30` dark indigo with smaller `#151526` metric tiles
- **Yellow:** `#ffbc2d` statistics and plots; `#ffd007` marketing accent and call-to-action
- **Purple:** `#572383` active tabs, plus indigo gradients for recruitment
- **Typography:** Manrope headings + DM Sans body, sensible responsive fallbacks
- **Interaction:** accessible interactive filters, ring indicators, real measured plots, touch-friendly portrait chips, responsive one-column mobile layouts

## Evidence boundary

Metrics originate from original thesis archived data or actual ESPN NBA/NCAA season data. We explicitly decline to show unsupported usage rate or PER, misleading team synergy scores, unavailable win probabilities, or unmeasured prospective NCAA-to-NBA draft probabilities. Suggested candidate fit remains an editable transparent scouting heuristic. All probabilities shown for drafted players are historical cohort-conditioned outcomes, not individualized current prospect predictions.

## Known limitations

The initial search corpus is composed of featured NBA athlete records and loaded NBA team rosters; it is not advertised as an exhaustive full-league index. NCAA search is restricted to a selected college program. ESPN is an unofficial public source without a commercial license/SLA. Season stats may be incomplete or missing for some rosters. A professional scouting platform would add a licensed league-wide roster index, on/off lineup data, RAPM, contract economics, position-neutral prospect projections and longitudinal college-to-NBA calibration.
