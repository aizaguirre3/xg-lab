# tests/test_skillcorner.R ----------------------------------------------------
# The SkillCorner reader (R/12_read_skillcorner.R) and the shared tracking
# evaluator (R/11_offside_tracking.R) on a hand-written fixture under
# tests/fixtures/skillcorner/900001: a 100 x 60 m pitch, 7 + 7 players, six
# frames at 10 fps (frame 5 empty), two passes with hand-computed margins.
# Run standalone:
#   Rscript -e 'source("R/09_offside_engine.R"); source("R/11_offside_tracking.R"); testthat::test_file("tests/test_skillcorner.R")'
library(testthat)
if (!exists("offside_margin")) source(file.path("R", "09_offside_engine.R"))
if (!exists("evaluate_tracking_match")) source(file.path("R", "11_offside_tracking.R"))

FX <- if (dir.exists(file.path("tests", "fixtures", "skillcorner"))) file.path("tests", "fixtures", "skillcorner") else
  testthat::test_path("fixtures", "skillcorner")

test_that("match metadata: pitch, teams, lineup roles and attacking directions", {
  meta <- read_skillcorner_meta("900001", dir = FX)
  expect_equal(meta$match_id, "900001")
  expect_equal(c(meta$pitch_length, meta$pitch_width), c(100, 60))
  expect_equal(c(meta$home_team_id, meta$away_team_id), c(1, 2))
  expect_equal(nrow(meta$players), 14)
  expect_equal(sum(meta$players$team == "Home"), 7)
  expect_equal(meta$players$player_id[meta$players$keeper], c("107", "201"))
  expect_equal(meta$players$team[meta$players$player_id == "207"], "Away")
  expect_equal(meta$periods$period, 1:2)
  # home attacks left_to_right (toward +x) in period 1 and right_to_left in period 2
  d <- meta$dir
  expect_true(d$right[d$team == "Home" & d$period == 1]); expect_false(d$right[d$team == "Home" & d$period == 2])
  expect_false(d$right[d$team == "Away" & d$period == 1]); expect_true(d$right[d$team == "Away" & d$period == 2])
})

test_that("an unfetched Git LFS pointer is recognised and refused", {
  ptr <- file.path(FX, "lfs_pointer.jsonl")
  expect_true(is_lfs_pointer(ptr))
  expect_false(is_lfs_pointer(file.path(FX, "900001", "900001_tracking_extrapolated.jsonl")))
  expect_false(is_lfs_pointer(file.path(FX, "does_not_exist.jsonl")))
  expect_error(read_skillcorner_tracking_long(ptr), "Git LFS pointer")
  expect_equal(skillcorner_matches(FX), "900001")
})

test_that("tracking JSONL -> frame vectors and wide per-team matrices", {
  meta <- read_skillcorner_meta("900001", dir = FX)
  tr <- read_skillcorner_tracking("900001", meta, dir = FX)
  expect_equal(tr$home$frame, 0:5)
  expect_equal(tr$home$period, c(1, 1, 1, 1, 1, NA))
  expect_equal(dim(tr$home$X), c(6, 7)); expect_equal(dim(tr$away$D), c(6, 7))
  expect_equal(tr$home$players, as.character(101:107)); expect_equal(tr$away$players, as.character(201:207))
  expect_equal(tr$home$keeper, c(rep(FALSE, 6), TRUE))
  # the runner moves +1 m per frame; the empty frame 5 is all NA
  expect_equal(unname(tr$home$X[, "102"]), c(28, 29, 30, 31, 32, NA))
  expect_equal(unname(tr$home$Y[, "102"]), c(5, 5, 5, 5, 5, NA))
  expect_equal(unname(tr$away$X[, "201"]), c(rep(45, 5), NA))
  expect_true(all(is.na(tr$home$X[6, ])))
  # detected flags: 202 extrapolated at frame 2 only; empty frame -> FALSE
  expect_equal(unname(tr$away$D[, "202"]), c(TRUE, TRUE, FALSE, TRUE, TRUE, FALSE))
  expect_true(all(tr$home$D[1:5, ]))
  # ball: (10, 0) up to frame 2, (10, 2) after, NA when out of play
  expect_equal(tr$home$ball_x, c(10, 10, 10, 10, 10, NA)); expect_equal(tr$home$ball_y, c(0, 0, 0, 2, 2, NA))
  expect_equal(tr$home$ball_detected, c(rep(TRUE, 5), FALSE))
})

test_that("passes: player_possession rows ending in a pass with a targeted receiver", {
  meta <- read_skillcorner_meta("900001", dir = FX)
  p <- read_skillcorner_passes("900001", meta, dir = FX)
  expect_equal(nrow(p), 2)                                 # carry, off-ball run, passing option, no-target pass excluded
  expect_equal(attr(p, "n_no_target"), 1L)
  expect_equal(p$event_id, c("8_0", "8_1"))
  expect_equal(p$team, c("Home", "Away")); expect_equal(p$period, c(1L, 1L))
  expect_equal(p$start_frame, c(2L, 3L))                   # frame_end of the possession = the kick frame
  expect_equal(p$from, c("101", "206")); expect_equal(p$to, c("102", "207"))
  expect_equal(p$offside_flag, c(1L, 0L)); expect_equal(p$pass_outcome, c("offside", "successful"))
  # SkillCorner stores event coordinates with the possession team attacking +x; the
  # right_to_left row (stored as x_end -10, reception 33/-1, line 30) comes back in
  # the raw tracking frame, where the away passer really stands at (10, 2).
  expect_equal(p$end_x, c(NA, -33)); expect_equal(p$end_y, c(NA, 1))   # reception point only when received
  expect_equal(p$sc_last_def_line_x, c(28, -30))
  expect_equal(p$passer_x_event, c(10, 10)); expect_equal(p$passer_y_event, c(0, 2))
  expect_equal(p$target_x_event, c(30, -32)); expect_equal(p$target_y_event, c(5, 0))
  raw_csv <- utils::read.csv(file.path(FX, "900001", "900001_dynamic_events.csv"), stringsAsFactors = FALSE)
  expect_equal(raw_csv$x_end[raw_csv$event_id == "8_1"], -10)
  expect_equal(p$fully_extrapolated, c(FALSE, FALSE)); expect_equal(p$reception_matched, c(TRUE, TRUE))
  # an attacking_side that contradicts match.json's home_team_side is reported
  flipped <- meta; flipped$dir$right <- !flipped$dir$right
  expect_warning(read_skillcorner_passes("900001", flipped, dir = FX), "attacking_side disagrees")
})

test_that("the bundle runs through the shared evaluator with hand-computed margins", {
  m <- load_skillcorner_match("900001", dir = FX)
  expect_equal(m$origin, "centre"); expect_equal(m$fps, 10); expect_equal(c(m$pl, m$pw), c(100, 60))
  expect_equal(m$passer_agreement, 1); expect_equal(m$n_no_target, 1L)   # both passers sit where the events say
  r <- evaluate_tracking_match(m, n_mc = 400)
  expect_equal(nrow(r), 2)
  expect_equal(r$offside_flag, c(1L, 0L))
  # Pass 1, home attacking +x, centre origin -> x + 50: receiver 102 at 30 -> 80,
  # second-last opponent 202 at 28 -> 78, ball (10, 0) -> 60, halfway 50: margin +2.
  expect_equal(r$margin_true[1], 2); expect_equal(r$second_last_def_x[1], 78); expect_equal(r$ball_x[1], 60)
  expect_equal(r$margin_max[1], 2)                        # the runner is also the furthest-forward team-mate
  expect_true(is.na(r$margin_near[1]))                     # no reception point for an offside pass
  expect_false(r$near_rule_correct_receiver[1])
  expect_equal(r$n_def[1], 7L); expect_true(r$ball_from_tracking[1])
  # Pass 2, away attacking -x -> x' = 50 - x: receiver 207 at -32 -> 82, second-last
  # opponent 106 at -30 -> 80, ball (10, 2) -> 40: margin +2 again, reception (-33, 1) -> (83, 29).
  expect_equal(r$margin_true[2], 2); expect_equal(r$second_last_def_x[2], 80); expect_equal(r$ball_x[2], 40)
  expect_equal(r$receiver_x[2], 82)
  expect_equal(r$margin_near[2], 2); expect_true(r$near_rule_correct_receiver[2])
  # kick-frame shifts: the runner gains 1 m per frame; the away receiver is static and frame 5 is empty
  expect_equal(c(r$margin_m2[1], r$margin_m1[1], r$margin_p1[1], r$margin_p2[1]), c(0, 1, 3, 4))
  expect_equal(c(r$margin_m2[2], r$margin_m1[2], r$margin_p1[2]), c(2, 2, 2)); expect_true(is.na(r$margin_p2[2]))
  # detection and keeper bookkeeping
  expect_equal(r$n_detected, c(13L, 14L))
  expect_equal(r$sld_detected, c(FALSE, TRUE)); expect_equal(r$n_def_detected, c(6L, 7L))
  expect_equal(r$sld_is_keeper, c(FALSE, FALSE)); expect_equal(r$last_is_keeper, c(TRUE, TRUE))
  expect_true(r$receiver_detected[1]); expect_true(r$passer_detected[1])
  # velocities by central difference over +/- 2 frames at 10 fps: 4 m in 0.4 s
  expect_equal(r$receiver_speed[1], 10); expect_equal(r$receiver_speed[2], 0)
  # Monte Carlo (sigma 0.3 m, sigma_t half a frame = 0.05 s): a 2 m margin is a clear offside
  expect_gt(r$p_offside[1], 0.95); expect_gt(r$p_offside[2], 0.99)
  expect_equal(r$call3, c("offside", "offside"))
  # SkillCorner's own line agrees with the engine after the same conversion
  right <- m$dir$right[match(paste(r$team, r$period), paste(m$dir$team, m$dir$period))]
  sc_line <- ifelse(right, r$sc_last_def_line_x + m$pl / 2, m$pl - (r$sc_last_def_line_x + m$pl / 2))
  expect_equal(sc_line, r$second_last_def_x)
  # Metrica-style columns are carried through unchanged
  expect_equal(r$from, c("101", "206")); expect_equal(r$game, c("900001", "900001"))
})

test_that("the passer-agreement check catches events that do not line up with the tracking", {
  meta <- read_skillcorner_meta("900001", dir = FX)
  tr <- read_skillcorner_tracking("900001", meta, dir = FX)
  p <- read_skillcorner_passes("900001", meta, dir = FX)
  expect_equal(skillcorner_passer_agreement(tr, p), 1)
  wrong <- p; wrong$passer_x_event <- -wrong$passer_x_event                # as if the normalisation were not undone
  expect_equal(skillcorner_passer_agreement(tr, wrong), 0)
  gone <- p; gone$start_frame <- 5L                                        # empty frame: nothing to check
  expect_true(is.na(skillcorner_passer_agreement(tr, gone)))
})

test_that("summary measures and the markdown section are computed from the results", {
  m <- load_skillcorner_match("900001", dir = FX)
  r <- evaluate_tracking_match(m, n_mc = 200); r$pl <- m$pl
  s <- summarise_tracking_eval(r)
  expect_equal(c(s$n, s$n_pos), c(2L, 1L))
  expect_equal(s$fp$rate[s$fp$receiver_rule == "true"], 1); expect_equal(s$fp$n[s$fp$receiver_rule == "true"], 1)
  expect_equal(s$sens$rate[s$sens$receiver_rule == "true"], 1)
  expect_equal(unname(s$frame_shift$median_abs_shift_1frame), 0.5)       # median of |3-2| and |2-2|
  expect_equal(unname(s$frame_shift$median_abs_shift_2frames), 2)        # the NA shift is dropped
  expect_equal(s$frame_shift$share_sign_flip_1frame, 0); expect_equal(s$frame_shift$share_gt5cm_1frame, 0.5)
  expect_equal(s$near_ok, 1); expect_equal(s$n_near, 1L)
  expect_equal(sum(s$mc_tab$flag_0) + sum(s$mc_tab$flag_1), 2L)
  out <- list(provider = "skillcorner", passes = r, fps = 10, sigma_t = 0.05, generated = as.Date("2026-01-01"),
              matches = "900001", n_matches = 1L, n_passes_total = 2L, dir = m$dir %>% dplyr::mutate(game = "900001"))
  lines <- skillcorner_section_lines(out)
  expect_true(any(grepl("^## SkillCorner", lines)))
  expect_true(any(grepl("1 frame \\(100 ms\\)", lines)))                 # 10 fps -> one frame is 100 ms
  expect_true(any(grepl("\\| true \\| 1.000 \\| ", lines)))
  expect_true(any(grepl("within 1 cm on 100.0% of 2 passes", lines)))
})
