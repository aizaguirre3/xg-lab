# tests/test_decisions.R ------------------------------------------------------
# Unit tests for R/06_ingest_decisions.R: pure helpers plus an end-to-end parse
# of a tiny hand-written StatsBomb-shaped fixture (tests/fixtures/).
# Run standalone:
#   Rscript -e 'source("R/06_ingest_decisions.R"); testthat::test_file("tests/test_decisions.R")'
library(testthat)
if (!exists("parse_match_decisions")) source(file.path("R", "06_ingest_decisions.R"))

FIX <- testthat::test_path("fixtures")   # resolves whether run from the repo root or tests/

test_that("polygon_area is the shoelace area", {
  # 60 x 80 rectangle (closed polygon, first point repeated)
  expect_equal(polygon_area(c(0, 0, 60, 0, 60, 80, 0, 80, 0, 0)), 4800)
  # right triangle legs 3 and 4 -> 6
  expect_equal(polygon_area(c(0, 0, 3, 0, 0, 4)), 6)
  # degenerate / missing
  expect_true(is.na(polygon_area(NULL)))
  expect_true(is.na(polygon_area(c(1, 2, 3, 4))))
})

test_that("in_own_box uses the StatsBomb own-box rectangle", {
  expect_true(in_own_box(10, 40))      # inside
  expect_false(in_own_box(19, 40))     # just beyond the 18-yard line
  expect_false(in_own_box(10, 17))     # outside the box width
  expect_false(in_own_box(NA, 40))     # missing location
  expect_equal(in_own_box(c(10, 110), c(40, 40)), c(TRUE, FALSE))
})

test_that("compact_frame + frame_summary read flags from the actor's perspective", {
  ff <- data.frame(teammate = c(TRUE, FALSE, FALSE), actor = c(TRUE, FALSE, FALSE),
                   keeper = c(FALSE, FALSE, TRUE))
  ff$location <- list(c(10, 40), c(12, 42), c(118, 40))
  fr <- compact_frame(ff)
  expect_equal(nrow(fr), 3)
  expect_equal(fr$x, c(10, 12, 118))
  s <- frame_summary(fr)
  expect_equal(s$has_frame, 1L)
  expect_equal(s$n_teammates, 1L)
  expect_equal(s$n_opponents, 2L)
  expect_true(s$keeper_opp_in_frame)
  expect_false(s$keeper_own_in_frame)
  expect_true(s$actor_in_frame)
  # missing frame
  s0 <- frame_summary(NULL)
  expect_equal(s0$has_frame, 0L)
  expect_true(is.na(s0$n_frame))
})

test_that("parse_match_decisions extracts decisions and joins 360 frames", {
  row <- tibble::tibble(match_id = 1L, competition = "Test Cup", season = "2026",
                        referee = "A. Ref", referee_id = 9L)
  out <- parse_match_decisions(row, raw_dir = FIX, n_neg = 5)
  dec <- out$decisions
  expect_equal(out$n_events, 8L)
  expect_equal(out$n_frames, 3L)

  # 5 decision rows: Foul Committed, Foul Won, Pass Offside, Bad Behaviour, Offside
  expect_equal(nrow(dec), 5)
  expect_setequal(dec$type, c("Foul Committed", "Foul Won", "Pass", "Bad Behaviour", "Offside"))

  foul <- dec[dec$type == "Foul Committed", ]
  expect_equal(foul$card, "Yellow Card")
  expect_true(foul$penalty)
  expect_equal(foul$foul_type, "Handball")
  expect_true(foul$in_own_box)                  # (10, 40) is inside the own box
  expect_equal(foul$referee, "A. Ref")
  expect_equal(foul$has_frame, 1L)
  expect_equal(foul$n_frame, 4L)
  expect_true(foul$keeper_own_in_frame)         # own keeper at (2, 40)
  expect_false(foul$keeper_opp_in_frame)
  expect_equal(foul$visible_area_frac, 4800 / 9600)

  bad <- dec[dec$type == "Bad Behaviour", ]
  expect_equal(bad$card, "Red Card")
  expect_equal(bad$has_frame, 0L)               # no frame for this event
  expect_true(is.na(bad$x))                     # no location on Bad Behaviour

  po <- dec[dec$type == "Pass", ]
  expect_equal(po$pass_outcome, "Pass Offside")
  expect_equal(po$recipient_id, 201)
  expect_equal(po$pass_end_x, 100)
  expect_equal(po$has_frame, 1L)
  expect_true(po$keeper_opp_in_frame)

  # join rate over the fixture: 2 of 5 decision events have frames
  expect_equal(sum(dec$has_frame), 2L)
})

test_that("pass_frames keeps every Pass Offside and only forward completed negatives", {
  row <- tibble::tibble(match_id = 1L, competition = "Test Cup", season = "2026",
                        referee = "A. Ref", referee_id = 9L)
  pf <- parse_match_decisions(row, raw_dir = FIX, n_neg = 5)$pass_frames
  # e-pass-off (positive) + e-pass-ok (forward into opp half); e-pass-back excluded
  expect_equal(nrow(pf), 2)
  expect_equal(sum(pf$offside_call), 1L)
  expect_setequal(pf$event_id, c("e-pass-off", "e-pass-ok"))
  neg <- pf[pf$offside_call == 0, ]
  expect_true(neg$pass_end_x > neg$x && neg$pass_end_x > 60)
  expect_equal(neg$n_frame, 2L)
})

test_that("decision_counts tallies cards, penalties and frame coverage", {
  row <- tibble::tibble(match_id = 1L, competition = "Test Cup", season = "2026",
                        referee = "A. Ref", referee_id = 9L)
  dec <- parse_match_decisions(row, raw_dir = FIX)$decisions
  cnt <- decision_counts(dec, row)
  tc <- cnt[cnt$competition == "Test Cup", ]
  expect_equal(tc$matches, 1L)
  expect_equal(tc$fouls, 1L)
  expect_equal(tc$carded_fouls, 1L)
  expect_equal(tc$yellow, 1L)
  expect_equal(tc$penalties, 1L)
  expect_equal(tc$handballs, 1L)
  expect_equal(tc$bad_behaviour, 1L)
  expect_equal(tc$pass_offside, 1L)
  expect_equal(tc$offside_events, 1L)
  expect_equal(tc$frame_pct_fouls, 100)
  expect_equal(tc$frame_pct_pass_offside, 100)
  tot <- cnt[cnt$competition == "TOTAL", ]
  expect_equal(tot$fouls, 1L)
})

test_that("read_sb_json returns NULL for a missing file without a network", {
  expect_null(read_sb_json("events/does-not-exist.json", raw_dir = FIX,
                           base = "https://invalid.invalid/nothing"))
})
