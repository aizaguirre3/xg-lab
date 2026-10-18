#!/usr/bin/env Rscript
# run_all.R -------------------------------------------------------------------
# Deterministic end-to-end pipeline driver for xg-lab.
#
# The build spec calls for a {targets} pipeline; this lightweight sourced driver
# is used instead to keep the fresh-machine dependency/install footprint small
# (no igraph/targets toolchain) while preserving the same phase boundaries and
# caching. Each phase reads the previous phase's cached .rds and writes its own,
# so re-runs are cheap and a failed phase doesn't lose upstream work.
#
# Usage:
#   Rscript run_all.R            # run all phases (skips ingest if cache exists)
#   XG_FORCE_INGEST=1 Rscript run_all.R   # re-pull StatsBomb data
#   XG_MAX_MATCHES=300 Rscript run_all.R  # smaller/faster data pull
#   XG_VARLAB=1 Rscript run_all.R         # also run the VAR-Lab phases (6-10)
# -----------------------------------------------------------------------------
set.seed(7406)
options(stringsAsFactors = FALSE)
stopifnot(file.exists("R/utils.R"))
source("R/utils.R")

# Each phase script defines its function(s) and only auto-runs when executed
# standalone (sys.nframe()==0). Sourcing here just loads the functions; we call
# each one explicitly, so every phase runs exactly once.
for (f in sprintf("R/%s", c("00_ingest.R", "01_features.R", "02_eda.R",
                            "03_models.R", "04_calibration.R", "05_finishing_skill.R",
                            "06_ingest_decisions.R", "07_decision_models.R",
                            "08_referee_effects.R", "09_offside_engine.R",
                            "10_offside_360.R", "11_offside_tracking.R")))
  source(f)

# Phase 0: unit tests (fail fast on a broken helper)
if (requireNamespace("testthat", quietly = TRUE)) {
  message("== Phase 0: unit tests ==")
  testthat::test_dir("tests", reporter = "summary", stop_on_failure = TRUE)
}

# Phase 1: ingest (cached -> skip) + features
if (!file.exists("data/shots_raw.rds") || identical(Sys.getenv("XG_FORCE_INGEST"), "1")) {
  message("== Phase 1a: ingest =="); ingest_shots()
} else message("Phase 1a: data/shots_raw.rds exists; skipping ingest.")
message("== Phase 1b: features =="); build_features()

message("== Phase 2: EDA ==");          run_eda()
message("== Phase 3: models ==");       run_models()
message("== Phase 4: calibration =="); run_calibration()
message("== Phase 5: finishing ==");    run_finishing()

# ---- VAR-Lab (docs/var_vision_scope.md): refereeing decisions + offside ----
# Skipped unless XG_VARLAB=1: it pulls ~4 GB of StatsBomb events + 360 frames
# (426 matches) into data/sb_raw/ and takes ~30 minutes on 3 cores.
if (identical(Sys.getenv("XG_VARLAB"), "1")) {
  if (!file.exists("data/decisions.rds") || identical(Sys.getenv("XG_FORCE_INGEST"), "1")) {
    message("== Phase 6: decisions ingest + 360 join =="); ingest_decisions()
  } else message("Phase 6: data/decisions.rds exists; skipping decisions ingest.")
  message("== Phase 7: decision models ==");   run_decision_models()
  message("== Phase 8: referee effects ==");   run_referee_effects()
  message("== Phase 9: offside engine on 360 frames =="); run_offside_360()
  if (dir.exists("data/external/metrica/data/Sample_Game_1")) {
    message("== Phase 10: offside engine on Metrica tracking =="); run_offside_tracking()
  } else message("Phase 10: Metrica sample data not found under data/external/metrica; skipping.",
                 " (git clone --depth 1 https://github.com/metrica-sports/sample-data.git data/external/metrica)")
}

message("\nPipeline complete. Figures in figures/, tables in data/*.rds.")
message("Render the report with:  quarto render analysis/report.qmd")