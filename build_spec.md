# Build Spec: xG From Scratch & The Finishing Skill Question (R)

**Project codename:** `xg-lab`
**Owner:** Alejandro Izaguirre
**Target audience:** Big tech DS internship recruiters/interviewers (ML, probabilistic prediction, applied stats roles)
**Stack:** R (>= 4.3), VS Code + R extension (`languageserver`), Quarto for reporting, `renv` for reproducibility
**Executor:** Claude Code — follow phases in order; each phase has acceptance criteria that must pass before moving on.
**Sibling project:** `uplift-lab` (experimentation/causal inference). Keep repo conventions identical: {targets} pipeline, testthat, lintr-clean, gitignored data dir.

---

## 1. Problem Statement

Expected goals (xG) is the canonical probabilistic model in soccer analytics: given a shot's context, what is the probability it becomes a goal? This is structurally identical to click/conversion prediction at tech companies — rare binary outcomes, heavy class imbalance, and a hard requirement that predicted probabilities be **calibrated**, not just well-ranked.

This project (a) builds an xG model from scratch on StatsBomb event data and benchmarks it against StatsBomb's own production xG, and (b) uses the model to answer a genuinely contested statistical question: **is finishing skill real and persistent, or do players who beat their xG mostly regress to the mean?** Part (b) demonstrates hierarchical modeling, shrinkage, and skill-vs-luck decomposition — the statistical reasoning interviewers probe hardest.

## 2. Goals

1. Train ≥ 3 xG models (calibrated logistic baseline, spatial GAM, gradient boosting) with leakage-safe cross-validation, and beat the logistic baseline on held-out log loss.
2. Demonstrate calibration rigor: reliability diagrams, expected calibration error (ECE), Brier decomposition; come within a documented margin of StatsBomb's own `shot_statsbomb_xg` on log loss (parity not required — honest benchmarking is the point).
3. Quantify finishing skill with a hierarchical/shrinkage model: player-level goals-above-xG with proper uncertainty, plus a split-sample persistence test (does period-1 over-performance predict period-2?).
4. Produce a ranked "true finishing skill" leaderboard with credible/confidence intervals that visibly shrinks low-shot-count players toward the mean — the figure that explains shrinkage in one image.
5. Ship a recruiter-skimmable README and a technically deep Quarto report.

## 3. Non-Goals

- **No tracking-data models** (full player positioning beyond freeze frames). Event + freeze-frame data only.
- **No live scraping of commercial data** (Opta, Wyscout). StatsBomb Open Data only — it is licensed for non-commercial use; include their attribution requirements in the README.
- **No win probability or match prediction model.** Separate project; would dilute the calibration story.
- **No deep learning.** GAM + GBM covers the model-complexity spectrum this dataset supports; tabular DL adds nothing here.
- **No Shiny app in v1** (P2 stretch only).

## 4. Dataset

- **Source:** StatsBomb Open Data — https://github.com/statsbomb/open-data (free JSON event data; competitions include World Cups, multiple Messi-era La Liga seasons, FA WSL, Euros, Champions League finals, and the Big-5-league 2015/16 release).
- **Access:** R package `StatsBombR` (install from GitHub: `statsbomb/StatsBombR`); fall back to reading the raw JSON from the repo with `jsonlite` + `purrr` if the package breaks.
- **Unit of analysis:** shot events. Expect on the order of tens of thousands of shots after pooling competitions — verify exact counts in code; do not assume.
- **Exclusions (document each):** penalties (model separately or exclude — default exclude, they're a different process), own goals, shots from direct free kicks flagged separately.
- **Key raw fields:** shot location (x, y), body part, technique, shot type (open play / set piece), under_pressure, first_time, `shot.freeze_frame` (positions of all players at shot moment), `shot_statsbomb_xg` (their model's value — the benchmark; NEVER a feature).
- **Engineered features (P0):** distance to goal center, shot angle (visible goal-mouth angle), body part, open-play vs set-piece, under pressure, first time; from freeze frame: number of defenders inside the shooter-to-goal triangle, distance to nearest defender, goalkeeper position/distance from goal line.
- Cache parsed events as parquet (`arrow`) in `data/`; raw JSON pulls are slow — never re-parse inside the pipeline once cached.

## 5. Repo Structure

```
xg-lab/
├── README.md                 # recruiter-facing: problem, headline results, 3 key figures, StatsBomb attribution
├── build_spec.md             # this file
├── renv.lock
├── _targets.R
├── R/
│   ├── 00_ingest.R           # pull events via StatsBombR, filter shots, cache parquet
│   ├── 01_features.R         # geometry + freeze-frame feature engineering
│   ├── 02_eda.R              # shot maps, goal-rate-by-zone, feature distributions
│   ├── 03_models.R           # logistic, GAM (mgcv), GBM (xgboost via tidymodels)
│   ├── 04_calibration.R      # reliability, ECE, Brier decomposition, benchmark vs StatsBomb xG
│   ├── 05_finishing_skill.R  # hierarchical model, shrinkage leaderboard, persistence test
│   └── utils.R               # geometry funcs, ECE, Brier decomposition — all unit tested
├── analysis/
│   └── report.qmd
├── figures/
├── data/                     # gitignored
└── tests/
    └── test_utils.R
```

## 6. Tech Requirements

- **Packages (P0):** `StatsBombR` (GitHub), `tidyverse`, `arrow`, `mgcv`, `tidymodels` + `xgboost`, `lme4`, `fixest`, `gt`, `targets`, `testthat`, `quarto`, `renv`. Optional (P1): `brms` for a fully Bayesian finishing-skill model if Stan installs cleanly; otherwise `lme4` + parametric bootstrap is sufficient.
- **Leakage discipline:** cross-validation must be **grouped by match** (shots from one match never span train/test); final holdout = entire competitions/seasons held out, not random rows. Document the split table in the report.
- **Reproducibility:** `set.seed(7406)`; pipeline runs end-to-end with `targets::tar_make()`; `renv::snapshot()` per phase.
- **Style/QA:** identical to `uplift-lab` — lintr-clean, roxygen comments + testthat coverage for every function in `utils.R` (geometry functions especially: angle/distance must be tested against hand-computed values).

## 7. Phases & Acceptance Criteria

### Phase 1 — Ingest & Features (P0)
- [ ] All available open-data competitions pulled; shots cached to parquet with a competition/season manifest table (counts of shots and goals per competition) rendered via `gt`.
- [ ] Geometry functions (distance, visible angle) unit-tested against ≥ 3 hand-computed cases each, including edge cases (shot from goal line, shot from center spot).
- [ ] Freeze-frame features computed; % of shots missing freeze frames documented and a missing-indicator strategy chosen and justified.
- [ ] Penalties/own goals excluded with counts reported.

### Phase 2 — EDA (P0)
- [ ] Shot map: hexbin goal rate over pitch coordinates (this is the project's visual hook — make it good; 300 dpi, proper pitch overlay).
- [ ] Empirical goal rate vs. distance and vs. angle curves with binomial CIs.
- [ ] Class balance and base rate by competition reported.

### Phase 3 — Models (P0)
- [ ] Model A: logistic regression (distance + angle + body part) — the interpretable baseline.
- [ ] Model B: `mgcv` GAM with 2-D spatial smooth `s(x, y)` + categorical/contextual terms.
- [ ] Model C: xgboost on the full feature set, tuned via grouped CV (document grid + budget).
- [ ] All models trained on identical grouped-CV folds; out-of-fold predictions saved for Phase 4.
- [ ] Held-out log loss: C ≤ B ≤ A expected; if ordering differs, investigate and explain in the report rather than silently accepting.

### Phase 4 — Calibration & Benchmark (P0)
- [ ] Reliability diagrams (out-of-fold) for all models + StatsBomb's xG on the same shots, on one figure.
- [ ] ECE and Brier score decomposition (reliability / resolution / uncertainty) implemented in `utils.R`, unit-tested on a toy example with a hand-computed answer.
- [ ] If GBM is miscalibrated, apply isotonic or Platt recalibration on a validation fold and show before/after.
- [ ] Benchmark table: log loss + Brier vs. `shot_statsbomb_xg` on the final holdout. Report the gap honestly; the README narrative is "how close can open features get to a production model," not "I beat StatsBomb."

### Phase 5 — Finishing Skill (P0)
- [ ] Per-player residual: goals minus model-xG, restricted to players above a minimum shot threshold (document the threshold sensitivity).
- [ ] Hierarchical model: shot-level logistic with player random intercept (`lme4::glmer`, offset/control = model xG on the link scale), producing shrunken player effects with intervals. P1 alternative: `brms` for full posteriors.
- [ ] The shrinkage figure: raw goals-above-xG vs. shrunken estimate, point size = shot count — low-volume players visibly pulled to zero.
- [ ] Persistence test: split each player's shots into two halves (odd/even chronological); correlation of over-performance across halves with a permutation-based null. State the conclusion plainly: how much of finishing over-performance is signal vs. noise in this sample.
- [ ] Leaderboard table (`gt`): top/bottom 10 by shrunken skill with intervals.

### Phase 6 — Report & README (P0)
- [ ] `report.qmd` renders clean: data & exclusions → EDA → models → calibration/benchmark → finishing skill → limitations (open-data selection bias toward elite competitions, no tracking data, ITT-style caveats on freeze-frame quality).
- [ ] README: 1-screen summary, the shot map, the reliability diagram, the shrinkage figure, headline numbers, 3-command reproduce instructions, StatsBomb attribution + non-commercial license note, and the interview-mapping table (Section 10).
- [ ] All README numbers regenerate from `tar_make()` output.

### Stretch (P2, only after P0)
- `brms` Bayesian version of Phase 5 with posterior predictive checks.
- Interactive shot-map explorer (Quarto OJS or Shiny).
- Per-competition xG transfer test: train on men's competitions, evaluate calibration on FA WSL (domain shift demo).

## 8. Success Metrics (for the project itself)

- Pipeline reproduces end-to-end from a clean clone in one command after data pull.
- A DS interviewer can get the two headline stories (calibration benchmark; finishing-skill persistence verdict) from the README in < 90 seconds.
- Every claim has a number, an interval or test, and a figure or table behind it.

## 9. Open Questions (resolve during build, non-blocking)

- **Data:** Is `StatsBombR` currently maintained/installable? If not, go straight to the raw-JSON fallback and note it in the README.
- **Stats:** Pool all competitions in one model, or include competition fixed effects? Default: pool with competition as a feature; check calibration by competition in Phase 4.
- **Stats:** Minimum shot threshold for the finishing-skill analysis (50? 100?) — run the sensitivity check and pick based on stability.
- **Engineering:** Stan/`brms` install on the build machine — attempt once, time-box to 30 minutes, fall back to `lme4` if it fights back.

## 10. Interview Mapping (keep in README)

| Project component | Interview question it answers |
|---|---|
| Grouped CV by match | "How do you prevent leakage with grouped/temporal data?" |
| Reliability + ECE + Brier decomposition | "Your classifier has great AUC — is it calibrated, and why does that matter?" |
| Benchmark vs StatsBomb xG | "How do you evaluate against a production system you can't inspect?" |
| GAM spatial smooth vs GBM | "Interpretability vs performance — how do you choose?" |
| Hierarchical shrinkage leaderboard | "How do you rank entities with wildly different sample sizes?" |
| Persistence / split-half test | "How do you tell skill from luck in a metric?" |
