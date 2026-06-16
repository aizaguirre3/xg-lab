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
# -----------------------------------------------------------------------------
set.seed(7406)
options(stringsAsFactors = FALSE)
stopifnot(file.exists("R/utils.R"))
source("R/utils.R")

# Each phase script defines its function(s) and only auto-runs when executed
# standalone (sys.nframe()==0). Sourcing here just loads the functions; we call
# each one explicitly, so every phase runs exactly once.
for (f in sprintf("R/%s", c("00_ingest.R", "01_features.R", "02_eda.R",
                            "03_models.R", "04_calibration.R", "05_finishing_skill.R")))
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

message("\nPipeline complete. Figures in figures/, tables in data/*.rds.")
message("Render the report with:  quarto render analysis/report.qmd")