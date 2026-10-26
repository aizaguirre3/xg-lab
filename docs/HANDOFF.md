# Handoff: continuing VAR-Lab on your own machine

Written 2026-10-15 at the end of the cloud session that built layers 1 and 2. Everything below was checked in that session; items that could not be checked are marked as such.

## 1. State of the work

- Branch: `docs/var-vision-scope` on `https://github.com/aizaguirre3/xg-lab` (public repository), eight commits on top of `main`, nothing merged, no pull request opened.
- Built and verified: milestones M0 to M3 of [`var_vision_scope.md`](var_vision_scope.md). Results and reproduce commands are in [`../analysis/varlab_report.md`](../analysis/varlab_report.md).
- Not built: M4 and M5 (video pipeline and offside line from broadcast clips), M6 (foul classifier), a reader for the DFL IDSSE tracking format, a Quarto section (the write-up is markdown), and an xgboost variant of the decision models.
- Added locally on 2026-10-17 (after this handoff was written): a SkillCorner reader (`R/12_read_skillcorner.R`) and a provider-agnostic `evaluate_tracking_match()` in `R/11_offside_tracking.R`, so the Law 11 engine runs on the 20 A-League matches as well as on Metrica; results in `analysis/offside_tracking.md`.
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
| Git author identity on the new commits | The first six commits were originally authored with your personal email address (taken from the session's account context); the repository is public. On 2026-10-15, with your approval, the branch was rewritten so every commit uses `aizaguirre3@users.noreply.github.com` as author and committer (dates, messages and trees unchanged) and force-pushed. GitHub can keep the superseded commits reachable by their old SHA for a while, so the address may remain visible via those SHAs until GitHub garbage-collects them; nothing in the current branch references them. Commits on `main` use a non-routable placeholder (`alejandro@alejandros-air.lan`). |
| Licence statements | StatsBomb and SkillCorner terms were read from the licence files. Metrica publishes no licence file (its README asks for responsible use and acknowledgement); an earlier "CC BY 4.0" label for Metrica was wrong and has been corrected. |

## 3. Local setup

Prerequisites: git, R 4.3 or newer, Python 3.10 or newer, about 6 GB of free disk, internet access to `github.com` and `raw.githubusercontent.com`. Optional: Quarto (to re-render the original xG report), a GPU and SoccerNet access (only for the unbuilt video layers).

```bash
git clone -b docs/var-vision-scope https://github.com/aizaguirre3/xg-lab
cd xg-lab
git log --oneline origin/main..HEAD        # expect 8 commits, the top one touching docs/HANDOFF.md
git log --format='%ae' origin/main..HEAD | sort -u   # expect only aizaguirre3@users.noreply.github.com

# R packages (all from CRAN; mgcv ships with R)
Rscript -e 'install.packages(c("jsonlite","dplyr","tidyr","purrr","tibble","ggplot2","lme4","xgboost","here","testthat"))'

# Python package + tests (no data needed)
python -m venv .venv && source .venv/bin/activate
pip install -e . pytest
pytest tests/python -q                      # expect 40 passed, 4 skipped

# R tests (no data needed)
Rscript -e 'source("R/utils.R"); for (f in list.files("R", pattern = "^0[6-9]|^1[0-2]", full.names = TRUE)) source(f); testthat::test_dir("tests")'
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
| SkillCorner open data (A-League 2024/25 tracking, 20 matches) | `https://github.com/SkillCorner/opendata` | MIT; SkillCorner asks for credit | commit `4340d274572876239c154c90bc507a9b3250a656` (2026-09-14) | 1.83 GB for the 20 `*_tracking_extrapolated.jsonl` files via Git LFS (85 to 102 MB each); the clone without them is about 150 MB | `R/12_read_skillcorner.R` -> `R/11_offside_tracking.R` |
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

With the SkillCorner data fetched as well (next subsection), the same command then takes about 30 more minutes and should report
`Passes evaluated: 16055 | flagged offside: 62` for the 20 matches; `analysis/offside_tracking.md` gets a SkillCorner section with a
false-positive rate of 0.012 for the targeted receiver (0.002 on Metrica) and a 1-frame (100 ms) median margin shift of 0.11 m.
If those differ, SkillCorner changed the data between revisions; re-pin to the commit above or document the new counts.

### Pull the SkillCorner data, pinned

```bash
git clone https://github.com/SkillCorner/opendata.git data/external/skillcorner      # pointers only if git-lfs is absent
git -C data/external/skillcorner checkout 4340d274572876239c154c90bc507a9b3250a656
# With git-lfs installed:
git -C data/external/skillcorner lfs pull --include "data/matches/*/*_tracking_extrapolated.jsonl"
Rscript R/11_offside_tracking.R          # Metrica (if present) then the 20 SkillCorner matches
```

Without git-lfs (the 2026-10-17 local run did this, so nothing had to be installed): each pointer file names the object's
sha256 and size, and GitHub serves the object over HTTPS at
`https://media.githubusercontent.com/media/SkillCorner/opendata/<commit>/<path>`. Download each
`data/matches/<id>/<id>_tracking_extrapolated.jsonl` from that URL, check `shasum -a 256` against the pointer's `oid`
and the byte count against its `size`, and replace the pointer with the verified file. All 20 verified on 2026-10-17.
`read_skillcorner_tracking()` refuses a file that is still a pointer. The `*_dynamic_events.csv` and `*_match.json`
files are ordinary git blobs and come with the clone.

Reader facts checked on the real data (also pinned by `tests/test_skillcorner.R` on a hand-written fixture):
tracking frames are contiguous at 10 fps with 22 players whenever the ball is in play; a pass is a `player_possession`
row with `end_type == "pass"`, kicked at `frame_end`; event coordinates are normalised so the possession team attacks
toward +x (for `attacking_side == "right_to_left"` they are the raw tracking coordinates rotated by 180 degrees), which
the reader undoes; `pass_outcome == "offside"` is the provider's offside flag.

### Fetch what the cloud could not (optional, for the unbuilt readers)

```bash
# SkillCorner pose (Hugging Face). Keep any HF token in an env var, never in a file in the repo.
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
Rscript R/11_offside_tracking.R    # Metrica: 0.2% false positives with the true receiver; SkillCorner (if fetched): 1.2%, 62 flags in 52 stoppages; analysis/offside_tracking.md
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

## 7. Author email on the branch (done)

The rewrite below was applied on 2026-10-15 and force-pushed; every commit on the branch now uses the GitHub noreply address. If you ever need to repeat it on another unmerged branch:

```bash
git config user.email "aizaguirre3@users.noreply.github.com"
git config user.name "Alejandro Izaguirre"
git rebase origin/main --exec 'git commit --amend --no-edit --reset-author'
git log --format='%h %ae' origin/main..HEAD      # all lines should show the noreply address
git push --force-with-lease origin docs/var-vision-scope
```

If you had cloned the branch before 2026-10-15, discard that clone or reset it to the rewritten remote (`git fetch && git reset --hard origin/docs/var-vision-scope`); do not merge or rebase the old commits back in, or the old identity returns. GitHub keeps superseded commits reachable by SHA for a while after a force-push.

## 8. What to do next, in order

1. Open a pull request to `main` if you want the work merged (the author rewrite in section 7 is done).
2. Add the StatsBomb logo to the README (licence 1.4).
3. SkillCorner reader: done (`R/12_read_skillcorner.R`, 2026-10-17). Still to write: an IDSSE reader (TRACAB positions, 25 Hz) that builds the same match bundle for `evaluate_tracking_match()`, and a pose reader for SkillCorner's body-pose files, which give the limb-level offset measurement the scope asked for.
4. M4/M5 only after SoccerNet access and a 12 to 24 GB GPU are in hand; the stack is listed in `var_vision_scope.md` section 6.
5. Optional: add xgboost to `R/07_decision_models.R` now that CRAN is reachable, and re-render `analysis/report.qmd` with Quarto if you want the xG report and VAR-Lab in one document.

### M4 on Colab (added 2026-10-17)

The vision scaffold lives in `varlab_vision/` (loaders, GS-HOTA adapter,
synthetic clip, detection -> tracking -> team -> pose pipeline), pinned in
`requirements-vision.txt` with licences in `varlab_vision/LICENSES.md`; the
CPU-only tests are `pytest tests/vision` (CI job `vision`, about a minute of
installs, no torch, no weights). Nothing in it needs SoccerNet files.

Workflow on the user's Colab Pro via `~/.local/bin/colab` (always pass `-s`):

```bash
# 1. environment probe: fresh L4, installs, weights, synthetic-clip timings, then releases the VM
colab run --gpu L4 --timeout 600 -s varlab-vision-env colab/vision_env_probe.py --ref <commit-sha>
# 2. once GSR access arrives: clips in batches under 45 min, each result downloaded as it is written
HF_TOKEN=... python colab/run_clips.py --clips SNGS-021 SNGS-022 --split valid --ref <commit-sha> \
    --out data/gsr_pred/valid --session varlab-clips --gpu L4 --budget-min 45 [--resume]
colab sessions   # must print "No active sessions" when done
```

`run_clips.py` does `colab new` -> upload the token as a 0600 file (never in a
command line or exec payload) -> `colab install -r requirements-vision.txt` ->
`colab restart-kernel` -> `colab exec -f colab/vision_worker.py` -> `setup(ref=...)`
-> per clip `process_clip(...)` then `colab download` -> `colab stop` in a
`finally`. Do not pin numpy on the VM (the kernel has it pre-imported) and do
not install inside a kernel you then import from; both failure modes are
written up in `analysis/vision_env.md`. It has not been run on real clips.

Measured on the L4 on 2026-10-17 (full table in `analysis/vision_env.md`):
clone 2 s, pip 40 s + 7 s for the GPU ONNX Runtime, weights 1.21 GB (RF-DETR
Medium 386 MB, SigLIP 775 MB, RTMPose-m 52 MB); on 1920x1080 synthetic frames
RF-DETR Medium 46 fps, ByteTrack 823 fps, SigLIP 326 crops/s, RTMPose-m 19.6 fps
at 11 people (216 people/s); peak GPU memory 1.2 GB reserved (1.6 GB on
`nvidia-smi`); smoke-pipeline GS-HOTA on the synthetic clip 0.88 (0.92 players
only). Expect 1.5 to 2 min per real clip, so the 49 validation clips are two to
three 45-minute runs. Still missing for M4 proper: jersey recognition, a
referee/goalkeeper role classifier, and the dataset's folder layout on Hugging
Face.

Pitch calibration was added on 2026-10-19 (`varlab_vision/pitch_model.py` and
`varlab_vision/calibration.py`; numbers and limits in `analysis/vision_env.md`).
It is classical and needs no weights. `production_pipeline` builds a
`FrameCalibrator` by default. Each object then gets `pitch_x` and `pitch_y` in
metres (bottom-middle of the box; null on an uncalibrated frame), `calib_error`
(the frame's RMS line-fit error in pixels) and `calib_status` (`calibrated`,
`tracked`, `interpolated` or `uncalibrated`); `bbox_pitch` is null on uncalibrated frames. Each
frame gets a camera record whose `calibration` key holds the status, the
homography and the line and inlier counts, and `has_labeled_camera` is true only
on frames with a usable calibration (status `calibrated` or `tracked`). `loaders.objects_frame` exposes `calib_error` and
`calib_status`, and `loaders.camera_frame(clip)` returns the per-frame
diagnostics. To switch it off, use `setup(with_calibration=False)` in
`colab/vision_worker.py` or `--no-calibration` in `colab/run_clips.py`. The
probe times it as a `calib` stage. On the Mac CPU a global search takes 0.6 to
0.84 s and a fitted frame about 82 ms. On the Colab L4 VM (measured 2026-10-20)
a fitted frame takes 0.264 s, about 3.3 minutes per clip if every frame is
fitted, which made calibration the slowest stage there and put a real clip at
4.5 to 5 minutes (49 clips: about four hours of L4 time). That run also exposed,
and `a4a231c` fixed, a search bug that failed when players hid the markings.

Since 2026-10-22 the calibrator applies a frame stride. The pipeline feeds it
through `FrameCalibrator.push` / `flush`. During steady tracking it fits about
one frame in six and interpolates the rest (`calib_status` `interpolated`,
`calib_error` null). The guards:
- the shot-cut test runs on every frame;
- a stretch is interpolated only when its closing fit is clean, otherwise it is
  replayed frame by frame;
- every interpolated frame is checked against its markings and fitted if it
  does not line up.

Measured back to back on the same 150-frame 1080p pan:
- Mac: 82.1 ms to 17.1 ms per frame (4.8x).
- Colab L4 VM (probe pinned to `1896146`): 0.269 s to 0.056 s (4.81x), a
  750-frame clip from 202 s to 42 s.

Accuracy is unchanged or better on the synthetic checks: clean sweep 0.042 m;
stress test 240/240 with 0 wrong locks and worst 0.129 m. On accelerating,
oscillating or shaky camera motion the stride falls back to about the cost of
fitting every frame, so the gain on real clips depends on how steady the camera
is. `FrameCalibrator(frame_stride=1)` or `update()` fits every frame. Full
tables are in `analysis/vision_env.md`. Accuracy on real broadcast frames is not
measured: every number so far comes from synthetic clips.

## 9. Things the cloud session could not verify

- The StatsBomb revision actually pulled on 2026-10-14 was `master` at that time; the SHA above is `master` on 2026-10-15. If the TOTAL counts in section 4 match after your pull, the data is the same.
- Official-system figures (camera counts, frame rates, tolerances) are tagged in `var_vision_scope.md`; several remain unverified press figures.
- IDSSE file formats and sizes were not inspected because figshare was unreachable.
