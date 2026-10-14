# tests/test_offside.R --------------------------------------------------------
# Hand-computed Law 11 cases for R/09_offside_engine.R.
# Run standalone:
#   Rscript -e 'source("R/09_offside_engine.R"); testthat::test_file("tests/test_offside.R")'
library(testthat)
if (!exists("offside_margin")) source(file.path("R", "09_offside_engine.R"))

test_that("second_last_defender_x picks the second-furthest-back opponent", {
  expect_equal(second_last_defender_x(c(118, 95, 80, 70)), 95)   # keeper at 118 is last
  expect_equal(second_last_defender_x(c(95, 118)), 95)           # order irrelevant
  expect_equal(second_last_defender_x(c(100, 100)), 100)         # tie: level pair
  expect_true(is.na(second_last_defender_x(c(118))))             # one defender -> NA
  expect_true(is.na(second_last_defender_x(numeric(0))))
})

test_that("offside_reference_x is the furthest forward of defender, ball and halfway", {
  expect_equal(offside_reference_x(c(118, 95), ball_x = 70), 95)   # defender line governs
  expect_equal(offside_reference_x(c(118, 95), ball_x = 105), 105) # ball ahead of the line
  expect_equal(offside_reference_x(c(118, 50), ball_x = 40), 60)   # halfway line governs
  expect_true(is.na(offside_reference_x(c(118), ball_x = 70)))
})

test_that("offside_margin matches hand-computed cases and 'level is onside'", {
  # receiver at 100, defenders 118 (GK) and 95, ball at 70 -> margin +5 (offside)
  expect_equal(offside_margin(100, c(118, 95), 70), 5)
  # level with the second-last defender -> margin 0 -> onside
  expect_equal(offside_margin(95, c(118, 95), 70), 0)
  expect_false(offside_margin(95, c(118, 95), 70) > 0)
  # in own half (x = 55) with a high defensive line at 50 -> halfway line rule, -5
  expect_equal(offside_margin(55, c(118, 50), 40), -5)
  # ball played from ahead of the receiver (back pass): ball at 105, receiver 100 -> -5
  expect_equal(offside_margin(100, c(118, 95), 105), -5)
  # vectorised over receivers
  expect_equal(offside_margin(c(100, 90), c(118, 95), 70), c(5, -5))
  # custom pitch length (metres): halfway at 52.5
  expect_equal(offside_margin(55, c(100, 40), 30, pitch_length = 105), 2.5)
})

test_that("to_attacking_frame flips and recentres correctly", {
  # StatsBomb corner origin, team attacking left: (10, 30) -> (110, 50)
  f <- to_attacking_frame(10, 30, attacking_right = FALSE)
  expect_equal(c(f$x, f$y), c(110, 50))
  # centre-origin tracking (105 x 68), attacking right: (0, 0) -> (52.5, 34)
  g <- to_attacking_frame(0, 0, TRUE, 105, 68, origin = "centre")
  expect_equal(c(g$x, g$y), c(52.5, 34))
  # centre-origin, attacking left: (10, 5) -> (105 - 62.5, 68 - 39) = (42.5, 29)
  h <- to_attacking_frame(10, 5, FALSE, 105, 68, origin = "centre")
  expect_equal(c(h$x, h$y), c(42.5, 29))
})

frame1 <- tibble::tibble(
  x = c(70, 100, 85, 118, 95, 80),
  y = c(40, 33, 50, 40, 40, 20),
  teammate = c(TRUE, TRUE, TRUE, FALSE, FALSE, FALSE),
  actor    = c(TRUE, FALSE, FALSE, FALSE, FALSE, FALSE),
  keeper   = c(FALSE, FALSE, FALSE, TRUE, FALSE, FALSE))

test_that("offside_frame evaluates a frame with both receiver rules", {
  r <- offside_frame(frame1, end_xy = c(104, 34))      # nearest teammate to the end point
  expect_true(r$ok)
  expect_equal(r$receiver_x, 100)
  expect_equal(r$second_last_def_x, 95)
  expect_equal(r$ball_x, 70)                           # ball = actor location
  expect_equal(r$reference_x, 95)
  expect_equal(r$margin, 5)
  expect_equal(r$n_def, 3L); expect_equal(r$n_att, 2L)
  m <- offside_frame(frame1, receiver = "max")
  expect_equal(m$receiver_x, 100)                      # furthest-forward attacker
  expect_equal(m$margin, 5)
  # explicit ball position overrides the actor
  b <- offside_frame(frame1, end_xy = c(104, 34), ball_xy = c(101, 40))
  expect_equal(b$margin, -1)                           # ball ahead of the receiver
})

test_that("offside_frame refuses frames it cannot judge", {
  expect_false(offside_frame(NULL)$ok)
  one_def <- frame1[c(1, 2, 4), ]                      # only one defender visible
  expect_false(offside_frame(one_def, end_xy = c(104, 34))$ok)
  no_actor <- frame1; no_actor$actor <- FALSE
  expect_false(offside_frame(no_actor, end_xy = c(104, 34))$ok)
  expect_true(offside_frame(no_actor, end_xy = c(104, 34), ball_xy = c(70, 40))$ok)
  expect_false(offside_frame(frame1, receiver = "nearest_end")$ok)   # no end point given
})

test_that("offside_mc propagates positional noise into P(offside) and a call", {
  mc <- offside_mc(frame1, end_xy = c(104, 34), sigma = 1, n = 400)
  # true margin +5 with sd ~ sqrt(2) on the difference -> essentially certain
  expect_gt(mc$p_offside, 0.99)
  expect_lt(abs(mc$margin_median - 5), 0.5)
  expect_equal(offside_call(mc), "offside")
  # a level receiver is a coin flip under noise -> refer
  level <- frame1; level$x[2] <- 95
  mcl <- offside_mc(level, end_xy = c(99, 34), sigma = 1, n = 400)
  expect_gt(mcl$p_offside, 0.3); expect_lt(mcl$p_offside, 0.7)
  expect_equal(offside_call(mcl), "refer")
  # zero noise reproduces the point estimate exactly
  mc0 <- offside_mc(frame1, end_xy = c(104, 34), sigma = 0, n = 50)
  expect_equal(mc0$margin_median, 5); expect_equal(mc0$p_offside, 1)
  # timing uncertainty: receiver sprinting forward at 8 yd/s, sd 0.04 s -> wider
  vx <- c(0, 8, 0, 0, 0, 0); vy <- rep(0, 6)
  mct <- offside_mc(frame1, end_xy = c(104, 34), sigma = 0, sigma_t = 0.04, vx = vx, vy = vy, n = 400)
  expect_gt(mct$margin_q95 - mct$margin_q05, 0.5)
  expect_lt(abs(mct$margin_median - 5), 0.3)
  # reproducible
  expect_equal(offside_mc(frame1, end_xy = c(104, 34), n = 100)$p_offside,
               offside_mc(frame1, end_xy = c(104, 34), n = 100)$p_offside)
})

test_that("point_in_polygon handles a rectangle and a concave polygon", {
  rect <- c(0, 0, 60, 0, 60, 80, 0, 80)
  expect_true(point_in_polygon(30, 40, rect))
  expect_false(point_in_polygon(61, 40, rect))
  expect_false(point_in_polygon(30, -1, rect))
  # L-shape: (0,0)-(10,0)-(10,5)-(5,5)-(5,10)-(0,10)
  L <- c(0, 0, 10, 0, 10, 5, 5, 5, 5, 10, 0, 10)
  expect_true(point_in_polygon(2, 8, L))
  expect_false(point_in_polygon(8, 8, L))
  expect_true(is.na(point_in_polygon(NA, 1, rect)))
  expect_true(is.na(point_in_polygon(1, 1, NULL)))
})

test_that("offside_mc_fast agrees with offside_mc", {
  f0 <- offside_mc_fast(frame1, end_xy = c(104, 34), sigma = 0, n = 50)
  expect_equal(f0$margin_median, 5); expect_equal(f0$p_offside, 1)
  f1 <- offside_mc_fast(frame1, end_xy = c(104, 34), sigma = 1, n = 2000)
  s1 <- offside_mc(frame1, end_xy = c(104, 34), sigma = 1, n = 2000)
  expect_lt(abs(f1$margin_median - s1$margin_median), 0.2)
  expect_lt(abs(f1$p_offside - s1$p_offside), 0.02)
  level <- frame1; level$x[2] <- 95
  fl <- offside_mc_fast(level, end_xy = c(99, 34), sigma = 1, n = 2000)
  expect_gt(fl$p_offside, 0.4); expect_lt(fl$p_offside, 0.6)
  expect_false(offside_mc_fast(NULL)$p_offside %in% c(0, 1))
})
