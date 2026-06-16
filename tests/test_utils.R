# tests/test_utils.R ----------------------------------------------------------
# Unit tests for the geometry + calibration helpers in R/utils.R.
# Every expected value is hand-computed in the comment beside it.
# Run standalone:  Rscript -e 'source("R/utils.R"); testthat::test_file("tests/test_utils.R")'
# Or via the pipeline / testthat::test_dir("tests").
library(testthat)
if (!exists("distance_to_goal")) source(file.path("R", "utils.R"))

test_that("distance_to_goal matches hand-computed values", {
  # penalty spot (108,40): 12 yards straight out
  expect_equal(distance_to_goal(108, 40), 12)
  # (114,30): sqrt(6^2 + 10^2) = sqrt(136)
  expect_equal(distance_to_goal(114, 30), sqrt(136))
  # on the goal centre: 0
  expect_equal(distance_to_goal(120, 40), 0)
  # vectorised
  expect_equal(distance_to_goal(c(108, 120), c(40, 40)), c(12, 0))
})

test_that("shot_angle matches hand-computed subtended angles (radians)", {
  # straight-on at 12 yards: 2*atan(4/12)
  expect_equal(shot_angle(108, 40), 2 * atan(1 / 3), tolerance = 1e-9)
  # straight-on at 6 yards: 2*atan(4/6)
  expect_equal(shot_angle(114, 40), 2 * atan(2 / 3), tolerance = 1e-9)
  # off-centre (114,30): atan2(48,120) = atan(0.4)
  expect_equal(shot_angle(114, 30), atan(0.4), tolerance = 1e-9)
})

test_that("shot_angle edge cases are well-defined", {
  # exactly on the goal line between the posts -> pi (widest possible)
  expect_equal(shot_angle(120, 40), pi, tolerance = 1e-9)
  # closer + more central => larger angle than farther + central
  expect_gt(shot_angle(114, 40), shot_angle(108, 40))
  # central => larger angle than the same distance but off to the side
  expect_gt(shot_angle(114, 40), shot_angle(114, 20))
})

test_that("point_in_triangle classifies known points", {
  # shooting cone: shooter (108,40), posts (120,36) & (120,44)
  # a point on the line to goal centre is inside
  expect_true(point_in_triangle(115, 40, 108, 40, 120, 36, 120, 44))
  # a point off to the side, outside the cone
  expect_false(point_in_triangle(115, 20, 108, 40, 120, 36, 120, 44))
  # a vertex counts as inside (edge-inclusive)
  expect_true(point_in_triangle(108, 40, 108, 40, 120, 36, 120, 44))
  # behind the shooter is outside
  expect_false(point_in_triangle(100, 40, 108, 40, 120, 36, 120, 44))
})

test_that("log_loss is correct on a toy example", {
  # p=.5,.5 ; y=1,0  -> -log(0.5)
  expect_equal(log_loss(c(0.5, 0.5), c(1, 0)), -log(0.5), tolerance = 1e-9)
  # confident + correct beats unsure
  expect_lt(log_loss(c(0.99), c(1)), log_loss(c(0.5), c(1)))
})

test_that("brier_score is correct on a toy example", {
  # p=.8,.2 ; y=1,0 -> mean(.04,.04)=.04
  expect_equal(brier_score(c(0.8, 0.2), c(1, 0)), 0.04, tolerance = 1e-12)
})

test_that("expected_calibration_error matches a hand-computed case", {
  # two bins, each off by 0.1, equal weight -> ECE = 0.10
  expect_equal(
    expected_calibration_error(c(0.1, 0.1, 0.9, 0.9), c(0, 0, 1, 1)),
    0.10, tolerance = 1e-9
  )
  # perfectly calibrated -> 0
  expect_equal(
    expected_calibration_error(c(0.5, 0.5, 0.5, 0.5), c(1, 0, 1, 0)),
    0, tolerance = 1e-9
  )
})

test_that("brier_decomposition recomposes and matches hand values", {
  # p=.2,.2,.8,.8 ; y=0,0,1,1 (perfect within-bin):
  #   uncertainty = .5*.5 = .25
  #   reliability = (2*.04 + 2*.04)/4 = .04
  #   resolution  = (2*.25 + 2*.25)/4 = .25
  #   recomposed  = .04 - .25 + .25 = .04 = brier
  d <- brier_decomposition(c(0.2, 0.2, 0.8, 0.8), c(0, 0, 1, 1))
  expect_equal(d$uncertainty, 0.25, tolerance = 1e-12)
  expect_equal(d$reliability, 0.04, tolerance = 1e-12)
  expect_equal(d$resolution,  0.25, tolerance = 1e-12)
  expect_equal(d$brier,       0.04, tolerance = 1e-12)
  expect_equal(d$recomposed,  d$brier, tolerance = 1e-12)
})
