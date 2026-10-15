# Handoff: continuing VAR-Lab on your own machine

Written 2026-10-15 at the end of the cloud session that built layers 1 and 2. Everything below was checked in that session; items that could not be checked are marked as such.

## 1. State of the work

- Branch: `docs/var-vision-scope` on `https://github.com/aizaguirre3/xg-lab` (public repository), seven commits on top of `main`, nothing merged, no pull request opened.
- Built and verified: milestones M0 to M3 of [`var_vision_scope.md`](var_vision_scope.md). Results and reproduce commands are in [`../analysis/varlab_report.md`](../analysis/varlab_report.md).
- Not built: M4 and M5 (video pipeline and offside line from broadcast clips), M6 (foul classifier), readers for the DFL IDSSE and SkillCorner tracking formats, a Quarto section (the write-up is markdown), and an xgboost variant of the decision models.
- A clean clone of the branch was tested in the cloud session: all tests pass, and a 4-match data pull from scratch worked end to end.

## 2. Safety audit of what the cloud session did

| Check | Result |
|---|---|
| Secrets, tokens, keys in tracked files | None. The only pattern hit is minified JavaScript inside the old rendered `analysis/report.html`. |
| Personal data in tracked files | None in file contents. Referee names in `analysis/decision_models.md` come from the public StatsBomb dataset and are aggregate analysis. |
| Plain-HTTP links in code or docs | None. Every data pull used HTTPS to the official host (see section 4). |
| Raw data committed | None. `data/` (including `data/sb_raw`, `data/external`, `data/parity`) and Python build artefacts are gitignored; `git ls-files` shows no JSON, CSV, RDS or JSONL except two tiny hand-written test fixtures. |
| Code executed from downloaded repositories | None. The Metrica and SkillCorner clones were read as data only; no script imports or runs their code. The R scripts parse JSON/CSV and never evaluate downloaded content. |
| Third-party software installed | Ubuntu packages for R 4.3.3 and CRAN libraries via apt (signed), Python packages via pip from PyPI (numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.1, statsmodels 0.15.0, pytest 9.1.1). |
| Credentials used | None. Every source was anonymous public access. Attaching SkillCorner with credentials was requested and denied by the session's permission classifier; nothing was granted. |
| Snowflake loader (`snowflake/load_to_snowflake.sql`) | Contains role and warehouse names only, no account identifier or password. |
| Git author identity on the new commits | **Your personal email address** (taken from the session's account context) is the commit author on the first six; the seventh (this handoff) uses the GitHub noreply form. Commits on `main` used a non-routable placeholder (`alejandro@alejandros-air.lan`). The repository is public, so the address is visible in commit metadata. See section 7 for the fix; the branch is unmerged, so rewriting it is cheap. |
| Licence statements | StatsBomb and SkillCorner terms were read from the licence files. Metrica publishes no licence file (its README asks for responsible use and acknowledgement); an earlier "CC BY 4.0" label for Metrica was wrong and has been corrected. |

## 3. Local setup

Prerequisites: git, R 4.3 or newer, Python 3.10 or newer, about 6 GB of free disk, internet access to `github.com` and `raw.githubusercontent.com`. Optional: Quarto (to re-render the original xG report), a GPU and SoccerNet access (only for the unbuilt video layers).

```bash
git clone -b docs/var-vision-scope https://github.com/aizaguirre3/xg-lab
cd xg-lab
git log --oneline origin/main..HEAD        # expect 7 commits, the top one touching docs/HANDOFF.md

# R packages (all from CRAN; mgcv ships with R)
Rscript -e 'install.packages(c("jsonlite","dplyr","tidyr","purrr","tibble","ggplot2","lme4","xgboost","here","testthat"))'

# Python package + tests (no data needed)
python -m venv .venv && source .venv/bin/activate
pip install -e . pytest
pytest tests/python -q                      # expect 40 passed, 4 skipped

# R tests (no data needed)
Rscript -e 'source("R/utils.R"); for (f in list.files("R", pattern = "^0[6-9]|^1[01]", full.names = TRUE)) source(f); testthat::test_dir("tests")'
```

Set a non-personal git identity before committing from your machine if you want to keep it that way:

```bash
git config user.email "aizaguirre3@users.noreply.github.com"
```

## 4. Data sources: what, where, how to verify

All downloads are over HTTPS from the project's official host. Pin the revision so you get the same files the session used.

| Dataset | Official source | Terms | Revision used | Size | Used by |
|---|---|---|---|---|---|
| StatsBomb Open Data (events + 360 frames) | `https://github.com/statsbomb/open-data`, files via `https://raw.githubusercontent.com/statsbomb/open-data/<ref>/data/` | Non-commercial only; no redistribution of the data; attribution and the StatsBomb logo on published analysis (licence 1.2.1, 1.2.2, 1.4) | `master` at commit `4b73468fc5b0f1950f9f66fada70ad3a4f9327cb` (as of 2026-10-15; the session pulled on 2026-10-14) | 852 files, 4.3 GB | `R/06_ingest_decisions.R` |
| Metrica Sports sample data | `https://github.com/metrica-sports/sample-data` | No licence file; README: "be responsible with the use of this data", acknowledge the source if used publicly; games are anonymised | commit `e706dd506b360d69d9d123d5b8026e7294b13996` (2021-04-15) | 234 MB | `R/11_offside_tracking.R` |
| SkillCorner open data (A-League 2024/25 tracking, 20 matches) | `https://github.com/SkillCorner/opendata` | MIT; SkillCorner asks for credit | commit `4340d274572876239c154c90bc507a9b3250a656` (2026-09-14) | ~100 MB per match via Git LFS (not fetched in the cloud) | not yet supported: needs a JSONL reader |
| SkillCorner body pose (2 matches, 25 Hz) | `https://huggingface.co/datasets/SkillCorner/opendata-bodypose` | MIT | not fetched (host blocked in the cloud) | ~3.3 GB per match | not yet supported |
| DFL / Sportec IDSSE (7 Bundesliga matches, TRACAB positions + events) | `https://doi.org/10.6084/m9.figshare.28196177` (figshare lists an MD5 per file) | CC BY 4.0; name the DFL and cite Bassek et al. 2025 | not fetched (host blocked in the cloud) | unknown | not yet supported: needs an XML/CSV reader |
| SoccerNet (video, GSR, MV-Foul) | `https://www.soccer-net.org` and `https://github.com/SoccerNet` | Gated: access form / NDA for video; read the Hugging Face terms before publishing anything | not fetched | tens of GB | M4 and M5 (unbuilt) |

### Pull the StatsBomb data, pinned

```bash
XG_SB_REF=4b73468fc5b0f1950f9f66fada70ad3a4f9327cb Rscript R/06_ingest_decisions.R   # ~30 min on 3 cores, ~4.3 GB into data/sb_raw/
```

Verify against the cloud session (same revision should give identical numbers):

| What | Expected |
|---|---|
| `ls data/sb_raw/events | wc -l` and `ls data/sb_raw/three-sixty | wc -l` | 426 and 426 |
| `du -sh data/sb_raw` | about 4.3 GB |
| `sha256sum data/sb_raw/competitions.json` | starts with `e6cd42f5d8956d6a` |
| `analysis/decisions_counts.md` TOTAL row | 426 matches, 10,540 fouls, 1,103 carded, 143 penalties, 1,348 Pass Offside, 144 Offside, frame coverage 84.9% / 87.3% |

If the TOTAL row differs, StatsBomb changed the data between revisions; re-pin to the commit above or document the new counts. Never type counts by hand: they are regenerated by the script.

### Pull the Metrica data, pinned

```bash
git clone https://github.com/metrica-sports/sample-data.git data/external/metrica
git -C data/external/metrica checkout e706dd506b360d69d9d123d5b8026e7294b13996
Rscript R/11_offside_tracking.R          # ~4 min; expect 1,763 passes, 1 flagged offside
```

### Fetch what the cloud could not (optional, for the unbuilt readers)

```bash
# SkillCorner tracking (Git LFS) and pose (Hugging Face). Keep any HF token in an env var, never in a file in the repo.
git lfs install
git clone https://github.com/SkillCorner/opendata.git data/external/skillcorner
git -C data/external/skillcorner checkout 4340d274572876239c154c90bc507a9b3250a656
git -C data/external/skillcorner lfs pull --include "data/matches/*/*_tracking_extrapolated.jsonl"
export HF_TOKEN=...   # only if the pose dataset asks for it; do not commit it
# IDSSE: download the archive from the figshare DOI above and compare each file's MD5 with the values figshare displays.
```

Do not run any code shipped inside these repositories (SkillCorner has a `src/` directory) without reading it first; the project only reads their data files.

## 5. Run order and expected outputs

```bash
Rscript R/06_ingest_decisions.R    # data/decisions.rds, data/pass_frames.rds, analysis/decisions_counts.md
Rscript R/07_decision_models.R     # card GAM: OOF log loss 0.305 vs 0.335 base rate; analysis/decision_models.md
Rscript R/08_referee_effects.R     # referee SD 0.17 logit, persistence r 0.20; figures/referee_shrinkage.png
Rscript R/10_offside_360.R         # 91% agreement beyond 2 yd; analysis/offside_360.md
Rscript R/11_offside_tracking.R    # 0.2% false positives with the true receiver; analysis/offside_tracking.md
Rscript tools/export_parity.R data/decision_oof.rds data/parity && pytest tests/python -q   # parity needs columns p_xgb, is_goal, fold; see analysis/varlab_report.md
# or, after the original xG pipeline: XG_VARLAB=1 Rscript run_all.R
```

Random seeds are fixed (7406), so numbers reproduce exactly on the same data; small differences across R or package versions are possible in the GAM fits.

## 6. Safe-handling rules for this project

- HTTPS only, official hosts only (GitHub, raw.githubusercontent.com, figshare, Hugging Face, soccer-net.org). Pin revisions with `XG_SB_REF` and `git checkout <sha>`; compare sizes, counts and the hashes above after each pull.
- Treat every downloaded file as data. Nothing in the pipeline executes downloaded content, and nothing should.
- Keep `data/` out of git (already ignored). StatsBomb forbids redistribution and commercial use of the data or any derived analysis; publish code and aggregates only, and add the StatsBomb logo to the README before publicising the results (attribution text is there, the logo is not yet).
- Credentials: none are needed for anything built so far. For SoccerNet or Hugging Face, keep tokens in environment variables or the tools' own credential stores, never in the repo; `.gitignore` already excludes `*.env` and `.env*`.
- Broadcast footage: never scrape Premier League or La Liga video. SoccerNet is the only lawful video source identified; read its gated terms before publishing weights or clips.
- The models here are analytics, not officiating. Nothing in this repository should be presented as making or recommending a refereeing decision.

## 7. Fixing the author email on the branch (optional, your call)

Six of the seven commits on `docs/var-vision-scope` are authored as `alejandro_gody@hotmail.es`. If you would rather not have that address in a public history, rewrite the unmerged branch and force-push it. Nobody else has based work on it.

```bash
git config user.email "aizaguirre3@users.noreply.github.com"
git config user.name "Alejandro Izaguirre"
git rebase origin/main --exec 'git commit --amend --no-edit --reset-author'
git log --format='%h %ae' origin/main..HEAD      # all lines should show the noreply address
git push --force-with-lease origin docs/var-vision-scope
```

Deleting the branch and re-pushing from a fresh clone also works. GitHub keeps the old commits reachable by SHA for a while even after a force-push, so do this before the branch is shared further if it matters.

## 8. What to do next, in order

1. Decide on section 7, then open a pull request to `main` if you want the work merged.
2. Add the StatsBomb logo to the README (licence 1.4).
3. Write readers for SkillCorner (`*_tracking_extrapolated.jsonl`, 10 fps, metres, centre origin) and IDSSE (TRACAB positions, 25 Hz) so `R/11_offside_tracking.R` can run on them; the engine's `to_attacking_frame()` already handles centre-origin coordinates. SkillCorner's pose files give the limb-level offset measurement the scope asked for.
4. M4/M5 only after SoccerNet access and a 12 to 24 GB GPU are in hand; the stack is listed in `var_vision_scope.md` section 6.
5. Optional: add xgboost to `R/07_decision_models.R` now that CRAN is reachable, and re-render `analysis/report.qmd` with Quarto if you want the xG report and VAR-Lab in one document.

## 9. Things the cloud session could not verify

- The StatsBomb revision actually pulled on 2026-10-14 was `master` at that time; the SHA above is `master` on 2026-10-15. If the TOTAL counts in section 4 match after your pull, the data is the same.
- Official-system figures (camera counts, frame rates, tolerances) are tagged in `var_vision_scope.md`; several remain unverified press figures.
- IDSSE file formats and sizes were not inspected because figshare was unreachable.
