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

run_tests <- function() {
  if (requireNamespace("testthat", quietly = TRUE)) {
    message("== Phase 0: unit tests ==")
    res <- testthat::test_dir("tests", reporter = "summary", stop_on_failure = TRUE)
  }
}

phase <- function(label, file, fn) {
  message("\n== ", label, " ==")
  source(file, local = new.env())
  fn()
}

run_tests()

# Phase 1: ingest (cached) + features
source("R/00_ingest.R", local = TRUE)   # auto-skips if cache exists
source("R/01_features.R", local = TRUE)
build_features()

# Phase 2: EDA
source("R/02_eda.R", local = TRUE); run_eda()

# Phase 3: models (grouped CV, OOF preds)
source("R/03_models.R", local = TRUE); run_models()

# Phase 4: calibration + benchmark
source("R/04_calibration.R", local = TRUE); run_calibration()

# Phase 5: finishing skill
source("R/05_finishing_skill.R", local = TRUE); run_finishing()

message("\nPipeline complete. Figures in figures/, tables in data/*.rds.")
message("Render the report with:  quarto render analysis/report.qmd")