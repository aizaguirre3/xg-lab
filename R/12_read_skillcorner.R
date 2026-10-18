# R/12_read_skillcorner.R -----------------------------------------------------
# Reader for the SkillCorner open broadcast-tracking data (A-League 2024/25,
# 20 matches): https://github.com/SkillCorner/opendata, MIT licence, SkillCorner
# asks to be credited. Turns one match directory into the bundle that
# R/11_offside_tracking.R evaluates (see evaluate_tracking_match() there):
#
#   data/external/skillcorner/data/matches/<id>/
#     <id>_match.json                   lineups, pitch size, which side the home
#                                       team attacks in each period
#     <id>_tracking_extrapolated.jsonl  one JSON object per frame, 10 fps, metres,
#                                       centre origin; 22 players + ball whenever
#                                       the ball is in play; each player carries
#                                       is_detected (seen by the broadcast camera)
#                                       or is extrapolated
#     <id>_dynamic_events.csv           SkillCorner Game Intelligence events; a pass
#                                       is a player_possession row whose end_type is
#                                       "pass", with the passer, the targeted receiver
#                                       and pass_outcome in {successful, unsuccessful,
#                                       offside}. Event coordinates are normalised to
#                                       the attacking direction (the possession team
#                                       always attacks toward +x): for a
#                                       right_to_left possession they are the raw
#                                       tracking coordinates rotated by 180 degrees,
#                                       (x, y) -> (-x, -y). The reader undoes that so
#                                       every coordinate it returns is in the raw frame.
#
# The tracking file is stored with Git LFS: before it is fetched the working copy
# holds a 130-byte pointer, which read_skillcorner_tracking() refuses with a clear
# message. Pure parsing, no network, nothing executed from the data repository.
# Unit-tested in tests/test_skillcorner.R on a hand-written fixture.
# -----------------------------------------------------------------------------
suppressPackageStartupMessages({ library(dplyr); library(tibble) })

SKILLCORNER_DIR <- file.path("data", "external", "skillcorner", "data", "matches")
SKILLCORNER_FPS <- 10

skillcorner_file <- function(match_id, what, dir = SKILLCORNER_DIR) {
  file.path(dir, match_id, sprintf("%s_%s", match_id, what))
}

#' List the match ids that have a materialised (non-pointer) tracking file.
skillcorner_matches <- function(dir = SKILLCORNER_DIR) {
  ids <- sort(basename(list.dirs(dir, recursive = FALSE)))
  ids[vapply(ids, function(id) {
    f <- skillcorner_file(id, "tracking_extrapolated.jsonl", dir)
    file.exists(f) && !is_lfs_pointer(f)
  }, logical(1))]
}

#' TRUE when a file is an unfetched Git LFS pointer rather than the data.
is_lfs_pointer <- function(path) {
  if (!file.exists(path) || file.size(path) > 1024) return(FALSE)
  first <- readLines(path, n = 1, warn = FALSE)
  length(first) == 1 && startsWith(first, "version https://git-lfs.github.com/spec/")
}

null_na <- function(x, as = NA_real_) if (is.null(x)) as else x
# Player ids as character keys without scientific notation (ids can exceed 1e6).
id_chr <- function(x) if (is.null(x)) NA_character_ else if (is.numeric(x)) sprintf("%.0f", x) else as.character(x)

#' Match metadata: pitch size, teams, lineup with roles, periods and the
#' attacking direction of each team in each period (from home_team_side).
read_skillcorner_meta <- function(match_id, dir = SKILLCORNER_DIR) {
  m <- jsonlite::fromJSON(skillcorner_file(match_id, "match.json", dir), simplifyVector = FALSE)
  home_id <- as.numeric(m$home_team$id); away_id <- as.numeric(m$away_team$id)
  players <- purrr::map_dfr(m$players, function(p) {
    tid <- as.numeric(null_na(p$team_id))
    tibble(player_id = id_chr(p$id), team_id = tid,
           team = if (isTRUE(tid == home_id)) "Home" else if (isTRUE(tid == away_id)) "Away" else NA_character_,
           role = as.character(null_na(p$player_role$acronym, NA_character_)),
           keeper = identical(p$player_role$acronym, "GK"),
           number = as.integer(null_na(p$number, NA_integer_)))
  })
  if (anyNA(players$team)) stop("match ", match_id, ": players whose team_id is neither home nor away")
  periods <- purrr::map_dfr(m$match_periods, ~ tibble(period = as.integer(.x$period),
                                                       start_frame = as.integer(.x$start_frame),
                                                       end_frame = as.integer(.x$end_frame)))
  sides <- unlist(m$home_team_side)
  if (length(sides) != nrow(periods)) stop("match ", match_id, ": home_team_side has ", length(sides),
                                           " entries for ", nrow(periods), " periods")
  if (!all(sides %in% c("left_to_right", "right_to_left"))) stop("match ", match_id, ": unknown home_team_side value")
  # Centre origin, +x toward the right: the team attacking left_to_right attacks toward +x.
  dir_tbl <- tibble(team = rep(c("Home", "Away"), each = nrow(periods)), period = rep(periods$period, 2),
                    right = c(sides == "left_to_right", sides != "left_to_right"))
  list(match_id = as.character(m$id), pitch_length = as.numeric(m$pitch_length), pitch_width = as.numeric(m$pitch_width),
       home_team_id = home_id, away_team_id = away_id, players = players, periods = periods, dir = dir_tbl)
}

#' Stream the JSONL tracking file into frame-level vectors plus one long table of
#' player positions (frame, player_id, x, y, detected). Frames with no players
#' (ball out of play, before kick-off) are kept in `frames` and absent from `long`.
read_skillcorner_tracking_long <- function(path, pagesize = 2000) {
  if (is_lfs_pointer(path)) stop(path, " is a Git LFS pointer, not the tracking data. Fetch it as docs/HANDOFF.md section 4 describes.")
  frames <- list(); longs <- list()
  con <- file(path, open = "r"); on.exit(close(con))
  jsonlite::stream_in(con, pagesize = pagesize, verbose = FALSE, simplifyVector = FALSE, handler = function(page) {
    fr <- tibble(
      frame = vapply(page, function(d) as.numeric(null_na(d$frame)), numeric(1)),
      period = vapply(page, function(d) as.numeric(null_na(d$period)), numeric(1)),
      ball_x = vapply(page, function(d) as.numeric(null_na(d$ball_data$x)), numeric(1)),
      ball_y = vapply(page, function(d) as.numeric(null_na(d$ball_data$y)), numeric(1)),
      ball_z = vapply(page, function(d) as.numeric(null_na(d$ball_data$z)), numeric(1)),
      ball_detected = vapply(page, function(d) isTRUE(d$ball_data$is_detected), logical(1)))
    n <- vapply(page, function(d) length(d$player_data), integer(1))
    pd <- unlist(lapply(page, `[[`, "player_data"), recursive = FALSE)
    lg <- tibble(frame = rep(fr$frame, n),
                 player_id = vapply(pd, function(p) id_chr(p$player_id), character(1)),
                 x = vapply(pd, function(p) as.numeric(null_na(p$x)), numeric(1)),
                 y = vapply(pd, function(p) as.numeric(null_na(p$y)), numeric(1)),
                 detected = vapply(pd, function(p) isTRUE(p$is_detected), logical(1)))
    frames[[length(frames) + 1]] <<- fr; longs[[length(longs) + 1]] <<- lg
  })
  frames <- bind_rows(frames); long <- bind_rows(longs)
  if (nrow(frames) == 0) stop(path, ": no frames")
  if (is.unsorted(frames$frame, strictly = TRUE)) stop(path, ": frames are not strictly increasing")
  list(frames = frames, long = long)
}

#' One team's tracking in the shape R/11 consumes: wide matrices (frame x player)
#' of x, y and the detected flag, plus the frame, period and ball vectors.
skillcorner_team_tracking <- function(tr, meta, team) {
  pl <- meta$players[meta$players$team == team, ]
  keep <- tr$long$player_id %in% pl$player_id
  fi <- match(tr$long$frame[keep], tr$frames$frame); pi <- match(tr$long$player_id[keep], pl$player_id)
  mk <- function(v, fill) { M <- matrix(fill, nrow(tr$frames), nrow(pl)); M[cbind(fi, pi)] <- v; colnames(M) <- pl$player_id; M }
  list(team = team, players = pl$player_id, keeper = pl$keeper,
       X = mk(tr$long$x[keep], NA_real_), Y = mk(tr$long$y[keep], NA_real_), D = mk(tr$long$detected[keep], FALSE),
       frame = tr$frames$frame, period = tr$frames$period,
       ball_x = tr$frames$ball_x, ball_y = tr$frames$ball_y, ball_detected = tr$frames$ball_detected)
}

read_skillcorner_tracking <- function(match_id, meta, dir = SKILLCORNER_DIR) {
  tr <- read_skillcorner_tracking_long(skillcorner_file(match_id, "tracking_extrapolated.jsonl", dir))
  unknown <- setdiff(unique(tr$long$player_id), meta$players$player_id)
  if (length(unknown)) warning("match ", match_id, ": ", length(unknown), " tracked player id(s) not in the lineup; dropped")
  list(home = skillcorner_team_tracking(tr, meta, "Home"), away = skillcorner_team_tracking(tr, meta, "Away"),
       frames = tr$frames)
}

SKILLCORNER_EVENT_COLS <- c("event_id", "index", "frame_end", "period", "attacking_side", "event_type", "end_type",
                            "player_id", "team_id", "x_end", "y_end", "pass_outcome", "player_targeted_id",
                            "player_targeted_x_pass", "player_targeted_y_pass",
                            "player_targeted_x_reception", "player_targeted_y_reception",
                            "last_defensive_line_x_end", "fully_extrapolated", "is_pass_reception_matched")

#' Passes from the dynamic-events CSV: player_possession rows that end in a pass
#' and name a targeted receiver. All coordinates are returned in the raw tracking
#' frame (metres, centre origin): SkillCorner stores event coordinates with the
#' possession team attacking toward +x, so right_to_left rows are rotated back by
#' (x, y) -> (-x, -y). `start_frame` is the frame the ball leaves the passer
#' (frame_end of the possession), `end_x`/`end_y` the targeted player's position
#' at reception (present only when the pass was received), and `offside_flag`
#' is 1 when SkillCorner's pass_outcome is "offside".
read_skillcorner_passes <- function(match_id, meta, dir = SKILLCORNER_DIR) {
  ev <- utils::read.csv(skillcorner_file(match_id, "dynamic_events.csv", dir), stringsAsFactors = FALSE,
                        check.names = FALSE, na.strings = c("", "NA"))
  miss <- setdiff(SKILLCORNER_EVENT_COLS, names(ev))
  if (length(miss)) stop("match ", match_id, ": dynamic events lack column(s) ", paste(miss, collapse = ", "))
  p <- ev[ev$event_type %in% "player_possession" & ev$end_type %in% "pass", SKILLCORNER_EVENT_COLS]
  n_no_target <- sum(is.na(p$player_targeted_id))
  p <- p[!is.na(p$player_targeted_id), ]
  team <- ifelse(p$team_id == meta$home_team_id, "Home", ifelse(p$team_id == meta$away_team_id, "Away", NA_character_))
  if (anyNA(team)) stop("match ", match_id, ": pass by a team that is neither home nor away")
  if (!all(p$attacking_side %in% c("left_to_right", "right_to_left"))) stop("match ", match_id, ": unknown attacking_side value")
  # Undo SkillCorner's attack-direction normalisation: right_to_left rows -> raw frame.
  sgn <- ifelse(p$attacking_side == "right_to_left", -1, 1)
  raw <- function(v) sgn * as.numeric(v)
  out <- tibble(game = as.character(match_id), idx = as.integer(p$index), event_id = as.character(p$event_id),
                team = team, period = as.integer(p$period), start_frame = as.integer(p$frame_end),
                from = id_chr(p$player_id), to = id_chr(p$player_targeted_id),
                end_x = raw(p$player_targeted_x_reception), end_y = raw(p$player_targeted_y_reception),
                pass_outcome = p$pass_outcome, offside_flag = as.integer(p$pass_outcome == "offside"),
                attacking_side = p$attacking_side,
                sc_last_def_line_x = raw(p$last_defensive_line_x_end),
                passer_x_event = raw(p$x_end), passer_y_event = raw(p$y_end),
                target_x_event = raw(p$player_targeted_x_pass), target_y_event = raw(p$player_targeted_y_pass),
                fully_extrapolated = as.logical(p$fully_extrapolated),
                reception_matched = as.logical(p$is_pass_reception_matched))
  # The event's attacking_side must agree with match.json's home_team_side.
  side_right <- out$attacking_side == "left_to_right"
  meta_right <- meta$dir$right[match(paste(out$team, out$period), paste(meta$dir$team, meta$dir$period))]
  n_bad <- sum(side_right != meta_right, na.rm = TRUE)
  if (n_bad) warning("match ", match_id, ": ", n_bad, " pass(es) whose attacking_side disagrees with home_team_side")
  attr(out, "n_no_target") <- n_no_target
  out
}

#' Share of passes whose passer sits, in the tracking, within `tol` metres of the
#' event's (de-normalised) passer position at the kick frame: a check that the
#' event frame index and coordinate frame line up with the tracking. In half of
#' the open-data matches the event coordinates are bit-identical to the tracking;
#' in the other half they sit a median of about 9 cm away (events derived from a
#' differently processed tracking pass), hence the 0.5 m default. NA when no pass
#' can be checked.
skillcorner_passer_agreement <- function(tr, passes, tol = 0.5) {
  ok <- vapply(seq_len(nrow(passes)), function(i) {
    t <- if (passes$team[i] == "Home") tr$home else tr$away
    k <- match(passes$start_frame[i], t$frame); j <- match(passes$from[i], t$players)
    if (is.na(k) || is.na(j) || is.na(t$X[k, j])) return(NA)
    abs(t$X[k, j] - passes$passer_x_event[i]) <= tol && abs(t$Y[k, j] - passes$passer_y_event[i]) <= tol
  }, logical(1))
  if (all(is.na(ok))) NA_real_ else mean(ok, na.rm = TRUE)
}

#' Everything R/11 needs for one SkillCorner match.
load_skillcorner_match <- function(match_id, dir = SKILLCORNER_DIR) {
  meta <- read_skillcorner_meta(match_id, dir)
  tr <- read_skillcorner_tracking(match_id, meta, dir)
  passes <- read_skillcorner_passes(match_id, meta, dir)
  agree <- skillcorner_passer_agreement(tr, passes)
  if (!is.na(agree) && agree < 0.95) warning("match ", match_id, ": passer position in the events is within 0.5 m of the tracking on only ",
                                             round(100 * agree, 1), "% of passes")
  list(id = as.character(match_id), provider = "skillcorner", home = tr$home, away = tr$away, passes = passes,
       dir = meta$dir, pl = meta$pitch_length, pw = meta$pitch_width, fps = SKILLCORNER_FPS, origin = "centre",
       meta = meta, passer_agreement = agree, passer_agreement_exact = skillcorner_passer_agreement(tr, passes, tol = 0.011),
       n_no_target = attr(passes, "n_no_target"))
}
