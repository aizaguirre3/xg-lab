# R/11_offside_tracking.R -----------------------------------------------------
# VAR-Lab layer 2a on continuous tracking: the Law 11 engine on
#   (a) Metrica Sports sample games 1 and 2 (25 Hz optical tracking, all 22
#       players + ball, events with the receiver identity and explicit
#       BALL LOST / OFFSIDE flags), and
#   (b) SkillCorner open data, A-League 2024/25, 20 matches (10 fps broadcast
#       tracking, 22 players + ball whenever the ball is in play, each player
#       flagged detected or extrapolated; dynamic events give the passer, the
#       targeted receiver and a pass_outcome of "offside"). The SkillCorner
#       reader lives in R/12_read_skillcorner.R.
#
# For every PASS event: positions at the kick frame -> torso margin of the TRUE
# receiver (Metrica's "To" player; SkillCorner's targeted player), plus the
# nearest-to-end-point and furthest-forward rules used on StatsBomb frames;
# velocities from finite differences feed a Monte Carlo over positional noise
# (sigma = 0.3 m) and kick timing (sigma_t = half a frame: 0.02 s at 25 Hz,
# 0.05 s at 10 fps). Frame-ambiguity sensitivity: the margin recomputed at kick
# frame +/- 1 and +/- 2 frames (one frame is 40 ms at 25 Hz, 100 ms at 10 fps).
#
# Both providers are evaluated by the same evaluate_tracking_match(); each has a
# loader that builds the match bundle it consumes (load_metrica_game(),
# load_skillcorner_match()).
# Data: data/external/metrica/data/Sample_Game_{1,2} (Metrica publishes these
# anonymised sample games without a formal licence; its README asks for
# responsible use and acknowledgement) and data/external/skillcorner (MIT).
# Writes data/offside_tracking_eval.rds, data/offside_tracking_skillcorner.rds,
#        figures/offside_tracking_margins.png,
#        figures/offside_tracking_margins_skillcorner.png,
#        analysis/offside_tracking.md (one section per provider present)
# -----------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(dplyr); library(purrr); library(tibble); library(tidyr); library(ggplot2)
})
if (!exists("offside_frame")) source(file.path("R", "09_offside_engine.R"))
if (!exists("load_skillcorner_match")) source(file.path("R", "12_read_skillcorner.R"))

METRICA_DIR <- file.path("data", "external", "metrica", "data")
PL <- 105; PW <- 68          # Metrica sample pitch, metres
FPS <- 25

# ---- Metrica loader ----------------------------------------------------------

read_metrica_tracking <- function(game, team) {
  path <- file.path(METRICA_DIR, game, sprintf("%s_RawTrackingData_%s_Team.csv", game, team))
  hdr <- read.csv(path, skip = 2, nrows = 1, header = FALSE, stringsAsFactors = FALSE)
  base <- as.character(unlist(hdr[1, ]))
  base[is.na(base)] <- ""
  # x,y pairs: a name followed by an empty cell -> Name_x, Name_y
  nm <- base
  for (i in seq_along(base)) {
    if (base[i] == "") nm[i] <- paste0(base[i - 1], "_y") else if (i > 3) nm[i] <- paste0(base[i], "_x")
  }
  dat <- read.csv(path, skip = 3, header = FALSE, stringsAsFactors = FALSE)
  names(dat) <- nm
  players <- setdiff(sub("_x$", "", grep("_x$", nm, value = TRUE)), "Ball")
  # wide matrices: one row per frame, one column per player (metres)
  X <- as.matrix(dat[, paste0(players, "_x")]) * PL
  Y <- as.matrix(dat[, paste0(players, "_y")]) * PW
  colnames(X) <- players; colnames(Y) <- players
  list(team = team, players = players, X = X, Y = Y, frame = dat$Frame, period = dat$Period,
       ball_x = dat$Ball_x * PL, ball_y = dat$Ball_y * PW)
}

# All on-pitch players (both teams) at one frame, as a small tibble; NULL if absent.
# `detected` is the provider's detected/extrapolated flag (matrix D); TRUE when the
# provider has none (optical tracking). `keeper` comes from the lineup when known.
positions_at <- function(home, away, f) {
  i <- match(f, home$frame); j <- match(f, away$frame)
  if (is.na(i) || is.na(j)) return(NULL)
  det <- function(t, k) if (is.null(t$D)) rep(TRUE, length(t$players)) else as.logical(t$D[k, ])
  kp <- function(t) if (is.null(t$keeper)) rep(FALSE, length(t$players)) else as.logical(t$keeper)
  out <- tibble(team = rep(c("Home", "Away"), c(length(home$players), length(away$players))),
                player = c(home$players, away$players),
                x = unname(c(home$X[i, ], away$X[j, ])), y = unname(c(home$Y[i, ], away$Y[j, ])),
                detected = unname(c(det(home, i), det(away, j))), keeper = unname(c(kp(home), kp(away))))
  out[!is.na(out$x) & !is.na(out$y), ]
}

read_metrica_events <- function(game) {
  ev <- read.csv(file.path(METRICA_DIR, game, sprintf("%s_RawEventsData.csv", game)), stringsAsFactors = FALSE)
  names(ev) <- c("team", "type", "subtype", "period", "start_frame", "start_time", "end_frame", "end_time",
                 "from", "to", "start_x", "start_y", "end_x", "end_y")
  ev %>% mutate(idx = row_number(), game = game)
}

# Attacking direction per team and period from the kick-off frame: a team whose
# players stand in the left half at kick-off attacks to the right.
attacking_right <- function(home, away, team, period, pl = PL) {
  f0 <- home$frame[which(home$period == period)[1]]
  p0 <- positions_at(home, away, f0)
  mean(p0$x[p0$team == team]) < pl / 2
}

#' Metrica game -> the bundle evaluate_tracking_match() consumes. Pass end
#' points are scaled to metres here so every provider hands over raw-frame metres.
load_metrica_game <- function(game) {
  home <- read_metrica_tracking(game, "Home"); away <- read_metrica_tracking(game, "Away")
  ev <- read_metrica_events(game)
  passes <- ev %>% filter(type == "PASS", to != "", !is.na(start_frame))
  # label: the next event is an OFFSIDE ball loss by the same team
  nxt <- ev[match(passes$idx + 1, ev$idx), ]
  passes$offside_flag <- as.integer(!is.na(nxt$subtype) & nxt$subtype == "OFFSIDE" & nxt$team == passes$team)
  passes <- passes %>%
    transmute(game, idx, team, period, start_frame, from, to, end_x = end_x * PL, end_y = end_y * PW, offside_flag)
  dir <- expand_grid(team = c("Home", "Away"), period = unique(home$period)) %>%
    mutate(right = map2_lgl(team, period, ~ attacking_right(home, away, .x, .y)))
  list(id = game, provider = "metrica", home = home, away = away, passes = as_tibble(passes), dir = dir,
       pl = PL, pw = PW, fps = FPS, origin = "corner")
}

# ---- Provider-agnostic evaluation -------------------------------------------

#' Evaluate every pass of one match bundle. The bundle is a list with
#'   home, away : per-team tracking (players, X, Y, frame, period; optional D =
#'                detected flags; home also carries ball_x, ball_y)
#'   passes     : tibble with game, idx, team ("Home"/"Away"), period, start_frame,
#'                from, to (player keys as in `players`), end_x, end_y (raw-frame
#'                metres, NA allowed), offside_flag, plus any columns to carry through
#'   dir        : tibble(team, period, right) attacking direction
#'   pl, pw     : pitch size in metres; fps; origin "corner" or "centre"
#' sigma_t defaults to half a frame.
evaluate_tracking_match <- function(m, sigma = 0.3, sigma_t = 0.5 / m$fps, n_mc = 300, min_players = 14) {
  home <- m$home; away <- m$away; passes <- m$passes; dir <- m$dir
  pl <- m$pl; pw <- m$pw; fps <- m$fps; origin <- m$origin

  one <- function(i, shift = 0) {
    p <- passes[i, ]; f <- p$start_frame + shift
    fr <- positions_at(home, away, f)
    if (is.null(fr) || nrow(fr) < min_players) return(NULL)
    right <- dir$right[dir$team == p$team & dir$period == p$period]
    if (length(right) != 1) return(NULL)
    xy <- to_attacking_frame(fr$x, fr$y, right, pl, pw, origin = origin)
    fr$x <- xy$x; fr$y <- xy$y
    bi <- match(f, home$frame)
    ball_known <- !is.na(bi) && !is.na(home$ball_x[bi])
    bxy <- if (ball_known) { t <- to_attacking_frame(home$ball_x[bi], home$ball_y[bi], right, pl, pw, origin = origin); c(t$x, t$y) } else NULL
    frame <- tibble(x = fr$x, y = fr$y, teammate = fr$team == p$team,
                    actor = fr$team == p$team & fr$player == p$from, keeper = FALSE)
    if (!any(frame$actor)) return(NULL)
    if (is.null(bxy)) bxy <- c(frame$x[frame$actor][1], frame$y[frame$actor][1])
    rec <- which(fr$team == p$team & fr$player == p$to)
    if (!length(rec)) return(NULL)
    def_idx <- which(!frame$teammate)
    def_x <- frame$x[def_idx]
    m_true <- offside_margin(frame$x[rec], def_x, bxy[1], pl)
    e <- to_attacking_frame(p$end_x, p$end_y, right, pl, pw, origin = origin)
    near <- offside_frame(frame, end_xy = c(e$x, e$y), ball_xy = bxy, receiver = "nearest_end", pitch_length = pl)
    mx <- offside_frame(frame, ball_xy = bxy, receiver = "max", pitch_length = pl)
    ord <- def_idx[order(def_x, decreasing = TRUE)]
    sld_i <- if (length(ord) >= 2) ord[2] else NA_integer_
    out <- tibble(margin_true = m_true, margin_near = near$margin, margin_max = mx$margin,
                  receiver_x = frame$x[rec], second_last_def_x = second_last_defender_x(def_x),
                  ball_x = bxy[1], n_def = length(def_idx), ball_from_tracking = ball_known,
                  near_rule_correct_receiver = isTRUE(all.equal(near$receiver_x, frame$x[rec])),
                  n_players = nrow(fr), n_detected = sum(fr$detected), n_def_detected = sum(fr$detected[def_idx]),
                  sld_detected = if (is.na(sld_i)) NA else fr$detected[sld_i],
                  sld_is_keeper = if (is.na(sld_i)) NA else fr$keeper[sld_i],
                  last_is_keeper = if (length(ord)) fr$keeper[ord[1]] else NA,
                  receiver_detected = fr$detected[rec], passer_detected = fr$detected[frame$actor][1])
    if (shift == 0) {
      # velocities by central difference over +/- 2 frames, for timing MC
      f1 <- positions_at(home, away, f - 2); f2 <- positions_at(home, away, f + 2)
      if (is.null(f1)) f1 <- fr[0, ]
      if (is.null(f2)) f2 <- fr[0, ]
      key <- paste(fr$team, fr$player)
      k1 <- match(key, paste(f1$team, f1$player)); k2 <- match(key, paste(f2$team, f2$player))
      sgn <- if (right) 1 else -1
      vx <- sgn * (f2$x[k2] - f1$x[k1]) / (4 / fps); vy <- sgn * (f2$y[k2] - f1$y[k1]) / (4 / fps)
      vx[is.na(vx)] <- 0; vy[is.na(vy)] <- 0
      frame_r <- frame; frame_r$x[rec] <- frame$x[rec]   # receiver fixed as the true one
      mc <- offside_mc_fast(frame_r, end_xy = c(frame$x[rec], frame$y[rec]), ball_xy = bxy, receiver = "nearest_end",
                            sigma = sigma, sigma_t = sigma_t, vx = vx, n = n_mc, seed = 7406 + i, pitch_length = pl)
      out$p_offside <- mc$p_offside; out$margin_q05 <- mc$margin_q05; out$margin_q95 <- mc$margin_q95
      out$call3 <- offside_call(mc)
      out$receiver_speed <- sqrt(vx[rec]^2 + vy[rec]^2)
    }
    out
  }
  res <- map_dfr(seq_len(nrow(passes)), function(i) {
    r0 <- one(i, 0); if (is.null(r0)) return(NULL)
    shifts <- map_dbl(c(-2, -1, 1, 2), function(s) { r <- one(i, s); if (is.null(r)) NA_real_ else r$margin_true })
    bind_cols(passes[i, ], r0,
              tibble(margin_m2 = shifts[1], margin_m1 = shifts[2], margin_p1 = shifts[3], margin_p2 = shifts[4]))
  })
  res
}

#' Metrica entry point kept for run_all.R and the handoff: one sample game.
evaluate_game <- function(game, sigma = 0.3, sigma_t = 0.02, n_mc = 300) {
  evaluate_tracking_match(load_metrica_game(game), sigma = sigma, sigma_t = sigma_t, n_mc = n_mc)
}

wilson <- function(k, n, z = 1.96) {
  p <- k / n; d <- 1 + z^2 / n
  c <- (p + z^2 / (2 * n)) / d; h <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / d
  c(lo = c - h, hi = c + h)
}

#' The measures reported for every provider: false-positive rate per receiver
#' rule on unflagged passes, sensitivity on flagged passes (true receiver),
#' kick-frame shift statistics, the three-way MC call table and the
#' nearest-to-end-point hit rate.
summarise_tracking_eval <- function(res) {
  neg <- res %>% filter(offside_flag == 0); pos <- res %>% filter(offside_flag == 1)
  rate <- function(x) { k <- sum(x > 0, na.rm = TRUE); n <- sum(!is.na(x)); c(rate = k / n, wilson(k, n), n = n) }
  fp <- bind_rows(true = rate(neg$margin_true), nearest_end = rate(neg$margin_near), furthest_forward = rate(neg$margin_max), .id = "receiver_rule")
  sens <- if (nrow(pos) > 0) bind_rows(true = rate(pos$margin_true), furthest_forward = rate(pos$margin_max), .id = "receiver_rule") else NULL
  frame_shift <- res %>% summarise(
    median_abs_shift_1frame = median(abs(margin_p1 - margin_true), na.rm = TRUE),
    p90_abs_shift_1frame = quantile(abs(margin_p1 - margin_true), 0.9, na.rm = TRUE),
    median_abs_shift_2frames = median(abs(margin_p2 - margin_true), na.rm = TRUE),
    p90_abs_shift_2frames = quantile(abs(margin_p2 - margin_true), 0.9, na.rm = TRUE),
    share_sign_flip_1frame = mean(sign(margin_p1) != sign(margin_true), na.rm = TRUE),
    share_sign_flip_2frames = mean(sign(margin_p2) != sign(margin_true), na.rm = TRUE),
    share_gt5cm_1frame = mean(abs(margin_p1 - margin_true) > 0.05, na.rm = TRUE),
    share_gt10cm_1frame = mean(abs(margin_p1 - margin_true) > 0.10, na.rm = TRUE),
    share_gt5cm_2frames = mean(abs(margin_p2 - margin_true) > 0.05, na.rm = TRUE))
  mc_tab <- res %>% count(call3, offside_flag) %>%
    pivot_wider(names_from = offside_flag, values_from = n, values_fill = 0, names_prefix = "flag_")
  if (!"flag_1" %in% names(mc_tab)) mc_tab$flag_1 <- 0L
  near_eval <- res %>% filter(!is.na(margin_near))
  list(n = nrow(res), n_pos = nrow(pos), fp = fp, sens = sens, frame_shift = frame_shift, mc_tab = mc_tab,
       near_ok = mean(near_eval$near_rule_correct_receiver), n_near = nrow(near_eval))
}

margin_histogram <- function(res, title) {
  neg <- res %>% filter(offside_flag == 0); pos <- res %>% filter(offside_flag == 1)
  ggplot(neg, aes(margin_true)) +
    geom_histogram(binwidth = 1, fill = "#9ecae1", colour = "white") +
    geom_vline(xintercept = 0, linetype = 2) +
    geom_vline(data = pos, aes(xintercept = margin_true), colour = "#d95f0e", linewidth = 0.6, alpha = 0.8) +
    labs(title = title,
         subtitle = sprintf("Blue: %d passes not flagged offside; orange: %d pass%s flagged offside", nrow(neg), nrow(pos), if (nrow(pos) == 1) "" else "es"),
         x = "Margin beyond the offside line (metres; + = offside position)", y = "Passes") +
    theme_minimal(base_size = 12)
}

# ---- Markdown sections -------------------------------------------------------

# Shared tables for one provider's results; `fps` fixes the millisecond labels.
common_section_lines <- function(s, fps, mc_label) {
  ms1 <- round(1000 / fps); ms2 <- 2 * ms1
  c("### False-positive rate on passes that were NOT flagged offside", "",
    "| Receiver rule | Share with margin > 0 | 95% CI | n |", "|---|---|---|---|",
    sprintf("| %s | %.3f | [%.3f, %.3f] | %d |", s$fp$receiver_rule, s$fp$rate, s$fp$lo, s$fp$hi, as.integer(s$fp$n)), "",
    sprintf("The nearest-to-end-point rule picked the true receiver in %.1f%% of the %d passes with a known end point.", 100 * s$near_ok, s$n_near), "",
    "### Kick-frame ambiguity (what one frame of timing error does to the margin)", "",
    sprintf("- Shifting the kick frame by 1 frame (%d ms) moves the margin by a median of %.2f m (90th percentile %.2f m); by 2 frames (%d ms): median %.2f m, p90 %.2f m.",
            ms1, s$frame_shift$median_abs_shift_1frame, s$frame_shift$p90_abs_shift_1frame, ms2,
            s$frame_shift$median_abs_shift_2frames, s$frame_shift$p90_abs_shift_2frames),
    sprintf("- The sign of the margin flips in %.1f%% of passes with a 1-frame shift and %.1f%% with a 2-frame shift.",
            100 * s$frame_shift$share_sign_flip_1frame, 100 * s$frame_shift$share_sign_flip_2frames),
    sprintf("- A 1-frame shift moves the margin by more than 5 cm (the reported manual-line tolerance of the official systems, unverified) in %.1f%% of passes and by more than 10 cm in %.1f%%; a 2-frame shift exceeds 5 cm in %.1f%%. This is why the official systems need a 500 Hz ball sensor or 100 fps cameras to fix the kick point.",
            100 * s$frame_shift$share_gt5cm_1frame, 100 * s$frame_shift$share_gt10cm_1frame, 100 * s$frame_shift$share_gt5cm_2frames), "",
    sprintf("### Three-way Monte-Carlo call (%s) vs. provider flag", mc_label), "",
    "| Engine call | Not flagged | Flagged offside |", "|---|---|---|",
    sprintf("| %s | %d | %d |", s$mc_tab$call3, s$mc_tab$flag_0, s$mc_tab$flag_1), "")
}

metrica_section_lines <- function(out) {
  res <- out$passes; s <- summarise_tracking_eval(res); pos <- res %>% filter(offside_flag == 1)
  fps <- if (is.null(out$fps)) FPS else out$fps
  generated <- if (is.null(out$generated)) "an earlier run" else format(out$generated)
  c("## Metrica Sports sample games 1 and 2 (25 Hz optical tracking)", "",
    sprintf("Generated by `R/11_offside_tracking.R` on %s. Passes evaluated: %d (%d flagged offside by the data provider). Positions at the kick frame, 25 Hz, metres.", generated, s$n, s$n_pos), "",
    common_section_lines(s, fps, "sigma = 0.3 m, sigma_t = 0.02 s"),
    "### The flagged offsides (case studies)", "",
    "| Game | Period | Kick frame | Passer | Receiver | Margin (m) | Second-last defender x | Ball x | P(offside) MC | Call |", "|---|---|---|---|---|---|---|---|---|---|",
    sprintf("| %s | %d | %d | %s | %s | %.2f | %.2f | %.2f | %.2f | %s |", pos$game, pos$period, pos$start_frame, pos$from, pos$to, pos$margin_true, pos$second_last_def_x, pos$ball_x, pos$p_offside, pos$call3), "",
    "![Margins](../figures/offside_tracking_margins.png)", "",
    "### Caveats", "",
    sprintf("- Only %d provider-flagged offside(s) follow a PASS event in these two games (a second flagged offside followed a ball recovery, not a pass, and is excluded): sensitivity cannot be estimated here; the false-positive rate can.", s$n_pos),
    "- A positive margin on an unflagged pass is not necessarily an error: an offside-positioned receiver who is not interfering is not flagged, and some passes are intercepted before any offence.",
    "- Tracking centroids are torso-level; limb-level offsets are not measurable without pose data.",
    "- Attacking direction is inferred from kick-off positions; ball position comes from tracking when available, else the passer's location.", "")
}

skillcorner_section_lines <- function(out) {
  res <- out$passes; s <- summarise_tracking_eval(res); pos <- res %>% filter(offside_flag == 1)
  neg <- res %>% filter(offside_flag == 0)
  # Detection strata: the second-last defender seen by the camera vs extrapolated.
  strat <- neg %>% filter(!is.na(margin_true), !is.na(sld_detected)) %>%
    mutate(stratum = ifelse(sld_detected, "second-last defender detected", "second-last defender extrapolated")) %>%
    group_by(stratum) %>% summarise(n = n(), fp_true = mean(margin_true > 0), fp_max = mean(margin_max > 0),
                                    median_detected = median(n_detected), .groups = "drop")
  pos_strat <- pos %>% summarise(n = n(), detected = sum(sld_detected), positive = sum(margin_true > 0),
                                 positive_detected = sum(margin_true > 0 & sld_detected))
  # Reader cross-check: SkillCorner's own last_defensive_line_x_end (raw frame) vs the
  # engine's second-last opponent, both expressed in the attacking frame.
  right <- out$dir$right[match(paste(res$game, res$team, res$period), paste(out$dir$game, out$dir$team, out$dir$period))]
  sc_line <- ifelse(right, res$sc_last_def_line_x + res$pl / 2, res$pl - (res$sc_last_def_line_x + res$pl / 2))
  line_diff <- sc_line - res$second_last_def_x
  line_ok <- !is.na(line_diff)
  line_agree_1cm <- mean(abs(line_diff[line_ok]) <= 0.01); line_agree_50cm <- mean(abs(line_diff[line_ok]) <= 0.5)
  far <- line_ok & abs(line_diff) > 0.5
  far_keeper <- if (any(far)) mean(res$sld_is_keeper[far] | !res$last_is_keeper[far], na.rm = TRUE) else NA_real_
  # Per match: volume, flags, false positives, detection and the event-vs-tracking check.
  info <- out$match_info
  if (is.null(info)) info <- tibble(game = out$matches, pitch_length = NA_real_, n_passes = NA_integer_,
                                    passer_agreement = NA_real_, passer_agreement_exact = NA_real_)
  per_match <- res %>% group_by(game) %>%
    summarise(evaluated = n(), flagged = sum(offside_flag), fp_true = mean(margin_true[offside_flag == 0] > 0, na.rm = TRUE),
              sld_detected_share = mean(sld_detected, na.rm = TRUE), .groups = "drop") %>%
    left_join(info, by = "game")
  n_exact <- sum(per_match$passer_agreement_exact >= 0.99, na.rm = TRUE)
  # Flagged passes come in runs around one stoppage, by BOTH teams (the offending
  # pass and the defending team's restart a few seconds later): cluster them
  # irrespective of team (gap <= 10 s) and judge each stoppage once.
  clus <- pos %>% arrange(game, start_frame) %>% group_by(game) %>%
    mutate(gap = start_frame - lag(start_frame), cluster = cumsum(is.na(gap) | gap > 10 * out$fps)) %>%
    group_by(game, cluster) %>%
    summarise(size = n(), both_teams = n_distinct(team) > 1, first_true = first(margin_true) > 0,
              any_true = any(margin_true > 0, na.rm = TRUE), any_max = any(margin_max > 0, na.rm = TRUE), .groups = "drop")
  n_clusters <- nrow(clus); share_clustered <- sum(clus$size[clus$size > 1]) / sum(clus$size)
  share_both_teams <- mean(clus$both_teams[clus$size > 1])
  sens_cluster <- c(first = mean(clus$first_true, na.rm = TRUE), any_true = mean(clus$any_true), any_max = mean(clus$any_max))
  mc_label <- sprintf("sigma = 0.3 m, sigma_t = %.2f s", out$sigma_t)
  c(sprintf("## SkillCorner A-League 2024/25 (10 fps broadcast tracking, %d matches)", out$n_matches), "",
    sprintf("Generated by `R/11_offside_tracking.R` (reader `R/12_read_skillcorner.R`) on %s. Passes evaluated: %d of %d pass-ending possessions with a targeted receiver (%d flagged `offside` by SkillCorner's pass_outcome). Positions at the kick frame, 10 fps, metres, centre origin; the \"true receiver\" is SkillCorner's targeted player. Matches: %s.",
            format(out$generated), s$n, out$n_passes_total, s$n_pos, paste(out$matches, collapse = ", ")), "",
    "### Per match", "",
    "| Match | Pitch length (m) | Passes with a target | Evaluated | Flagged offside | FP rate, targeted receiver | Second-last defender detected | Passer event position within 0.5 m of tracking | ... within 1 cm |",
    "|---|---|---|---|---|---|---|---|---|",
    sprintf("| %s | %s | %s | %d | %d | %.3f | %.2f | %s | %s |", per_match$game,
            ifelse(is.na(per_match$pitch_length), "", format(per_match$pitch_length)), ifelse(is.na(per_match$n_passes), "", per_match$n_passes),
            per_match$evaluated, per_match$flagged, per_match$fp_true, per_match$sld_detected_share,
            ifelse(is.na(per_match$passer_agreement), "", sprintf("%.3f", per_match$passer_agreement)),
            ifelse(is.na(per_match$passer_agreement_exact), "", sprintf("%.3f", per_match$passer_agreement_exact))), "",
    sprintf("The last two columns check the reader: the passer's position in the dynamic events, after undoing SkillCorner's attack-direction normalisation, is compared with the tracking at the event's kick frame. In %d of %d matches the two are bit-identical; in the others the events sit a few centimetres from the tracking (a different processing pass of the same positions), and the agreement peaks at the kick frame rather than at a neighbouring one.", n_exact, nrow(per_match)), "",
    common_section_lines(s, out$fps, mc_label),
    "### Sensitivity on the passes SkillCorner flagged offside", "",
    "| Receiver rule | Share with margin > 0 | 95% CI | n |", "|---|---|---|---|",
    sprintf("| %s | %.3f | [%.3f, %.3f] | %d |", s$sens$receiver_rule, s$sens$rate, s$sens$lo, s$sens$hi, as.integer(s$sens$n)), "",
    sprintf("With the targeted receiver, %d of %d flagged passes have a positive margin; %d of the %d flagged passes have the second-last defender detected by the camera, and %d of those %d are positive.",
            pos_strat$positive, pos_strat$n, pos_strat$detected, pos_strat$n, pos_strat$positive_detected, pos_strat$detected), "",
    sprintf("The flag is not a frame-precise label of the offending pass: the %d flagged passes form %d stoppages (runs of flagged passes less than 10 s apart, whichever team passed), %.0f%% of flagged passes belong to a run of two or more, and %.0f%% of those runs contain passes by both teams, i.e. the attacking team's pass and the defending team's restart are both labelled `offside`. Judging each stoppage once: the first flagged pass has its targeted receiver beyond the line in %.1f%% of stoppages; at least one flagged pass of the stoppage has the targeted receiver beyond the line in %.1f%% and the furthest-forward team-mate beyond it in %.1f%%.",
            s$n_pos, n_clusters, 100 * share_clustered, 100 * share_both_teams, 100 * sens_cluster["first"], 100 * sens_cluster["any_true"], 100 * sens_cluster["any_max"]), "",
    "### Detected vs. extrapolated second-last defender (unflagged passes)", "",
    "| Stratum | n | FP rate, true receiver | FP rate, furthest forward | Median players detected in frame |", "|---|---|---|---|---|",
    sprintf("| %s | %d | %.3f | %.3f | %.0f |", strat$stratum, strat$n, strat$fp_true, strat$fp_max, strat$median_detected), "",
    "### Reader cross-check against SkillCorner's own defensive line", "",
    sprintf("- SkillCorner publishes `last_defensive_line_x_end` on every pass. After the reader's conversion to the attacking frame it equals the engine's second-last opponent to within 1 cm on %.1f%% of %d passes and to within 50 cm on %.1f%% (the centimetre-level differences are the same event-vs-tracking processing offset seen in the per-match table). Of the %d passes further apart than 50 cm, %.1f%% have a goalkeeper who is not the deepest opponent or who is the second-last one, where SkillCorner's outfield line and the Law 11 second-last opponent legitimately differ; the margins here use the Law 11 definition.",
            100 * line_agree_1cm, sum(line_ok), 100 * line_agree_50cm, sum(far), 100 * far_keeper), "",
    "### Flagged offsides with a negative margin for the targeted receiver (case list)", "",
    "| Match | Period | Kick frame | Passer id | Targeted id | Margin, targeted (m) | Margin, furthest forward (m) | Second-last defender detected | Players detected | Call |", "|---|---|---|---|---|---|---|---|---|---|",
    { q <- pos %>% filter(margin_true <= 0)
      if (nrow(q)) sprintf("| %s | %d | %d | %s | %s | %.2f | %.2f | %s | %d | %s |", q$game, q$period, q$start_frame, q$from, q$to, q$margin_true, q$margin_max, ifelse(q$sld_detected, "yes", "no"), q$n_detected, q$call3) else "| (none) | | | | | | | | | |" }, "",
    "![Margins](../figures/offside_tracking_margins_skillcorner.png)", "",
    "### Caveats", "",
    "- Broadcast tracking: players off camera are extrapolated by the provider, and SkillCorner states that about 97% of player identities are correct. An extrapolated second-last defender fixes the offside line from a guess, which is why the detection strata are reported separately.",
    "- One frame is 100 ms at 10 fps, two and a half times the 40 ms of the 25 Hz Metrica data, so the kick-frame ambiguity figures above are not comparable across providers without that factor in mind; the Monte Carlo uses sigma_t = half a frame for both.",
    "- SkillCorner's pass_outcome `offside` is the provider's label for the pass, and the \"true receiver\" is the provider's targeted player, not the player the referee flagged. The label marks the passes of both teams around the stoppage (see the sensitivity section), so sensitivity against it is a lower bound at best; a flagged pass whose targeted player is onside may also mean another team-mate was the offside player (furthest-forward rule, case list).",
    "- The positional noise sigma = 0.3 m is the same nominal value used for the optical data; SkillCorner publishes no accuracy figure for broadcast tracking, so it is not a calibrated error.",
    "- Attacking direction comes from match.json's home_team_side per period and is checked against each event's attacking_side; the ball position comes from tracking when available, else the passer's location.", "")
}

#' Assemble analysis/offside_tracking.md from whichever provider results exist.
write_offside_tracking_md <- function() {
  secs <- character(0)
  if (file.exists(file.path("data", "offside_tracking_eval.rds"))) {
    out <- readRDS(file.path("data", "offside_tracking_eval.rds"))
    if (!is.null(out$passes)) secs <- c(secs, metrica_section_lines(out))
  }
  if (file.exists(file.path("data", "offside_tracking_skillcorner.rds")))
    secs <- c(secs, skillcorner_section_lines(readRDS(file.path("data", "offside_tracking_skillcorner.rds"))))
  lines <- c("# Law 11 engine on continuous tracking", "",
             "The pure Law 11 engine (`R/09_offside_engine.R`) evaluated at the kick frame of every pass in two tracking datasets of different kinds: 25 Hz optical tracking (Metrica Sports) and 10 fps broadcast tracking (SkillCorner). Every number below is computed by `R/11_offside_tracking.R`; nothing is typed by hand. Positive margin = the receiver is beyond the offside line (an offside position, not necessarily an offence).", "",
             secs)
  dir.create("analysis", showWarnings = FALSE)
  writeLines(lines, file.path("analysis", "offside_tracking.md"))
  invisible(lines)
}

# ---- Runners -----------------------------------------------------------------

run_offside_tracking <- function(games = c("Sample_Game_1", "Sample_Game_2")) {
  res <- map_dfr(games, function(g) { message("Evaluating ", g, "..."); evaluate_game(g) })
  message("Passes evaluated: ", nrow(res), " | flagged offside: ", sum(res$offside_flag))
  s <- summarise_tracking_eval(res)
  dir.create("figures", showWarnings = FALSE)
  ggsave(file.path("figures", "offside_tracking_margins.png"),
         margin_histogram(res, "Torso margin of the true receiver at the kick frame (Metrica games 1 and 2)"),
         width = 8, height = 5, dpi = 300)
  dir.create("data", showWarnings = FALSE)
  saveRDS(list(provider = "metrica", passes = res, fps = FPS, sigma_t = 0.02, generated = Sys.Date(),
               false_positive = s$fp, frame_shift = s$frame_shift, mc = s$mc_tab),
          file.path("data", "offside_tracking_eval.rds"))
  write_offside_tracking_md()
  invisible(res)
}

run_offside_skillcorner <- function(matches = skillcorner_matches(), sigma = 0.3, n_mc = 300) {
  if (!length(matches)) stop("No materialised SkillCorner tracking files under ", SKILLCORNER_DIR)
  bundles <- list()
  res <- map_dfr(matches, function(id) {
    message("Evaluating SkillCorner match ", id, "...")
    m <- load_skillcorner_match(id)
    bundles[[id]] <<- list(dir = m$dir %>% mutate(game = id), pl = m$pl,
                           info = tibble(game = id, pitch_length = m$pl, n_passes = nrow(m$passes), n_no_target = m$n_no_target,
                                         passer_agreement = m$passer_agreement, passer_agreement_exact = m$passer_agreement_exact))
    r <- evaluate_tracking_match(m, sigma = sigma, n_mc = n_mc)
    r$pl <- m$pl
    r
  })
  message("Passes evaluated: ", nrow(res), " | flagged offside: ", sum(res$offside_flag))
  s <- summarise_tracking_eval(res)
  dir.create("figures", showWarnings = FALSE)
  ggsave(file.path("figures", "offside_tracking_margins_skillcorner.png"),
         margin_histogram(res, sprintf("Torso margin of the targeted receiver at the kick frame (SkillCorner, %d A-League matches)", length(matches))),
         width = 8, height = 5, dpi = 300)
  dir.create("data", showWarnings = FALSE)
  saveRDS(list(provider = "skillcorner", passes = res, fps = SKILLCORNER_FPS, sigma_t = 0.5 / SKILLCORNER_FPS,
               generated = Sys.Date(), matches = matches, n_matches = length(matches),
               n_passes_total = sum(map_int(bundles, ~ .x$info$n_passes)), dir = bind_rows(map(bundles, "dir")),
               match_info = bind_rows(map(bundles, "info")),
               false_positive = s$fp, sensitivity = s$sens, frame_shift = s$frame_shift, mc = s$mc_tab),
          file.path("data", "offside_tracking_skillcorner.rds"))
  write_offside_tracking_md()
  invisible(res)
}

if (sys.nframe() == 0) {
  if (dir.exists(file.path(METRICA_DIR, "Sample_Game_1"))) run_offside_tracking() else message("Metrica data not found under ", METRICA_DIR, "; skipping.")
  if (length(skillcorner_matches())) run_offside_skillcorner() else message("No SkillCorner tracking under ", SKILLCORNER_DIR, "; skipping.")
}
