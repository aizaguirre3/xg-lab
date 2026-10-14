# R/11_offside_tracking.R -----------------------------------------------------
# VAR-Lab layer 2a on continuous tracking: the Law 11 engine on Metrica Sports
# sample games 1 and 2 (25 Hz, all 22 players + ball, events with the receiver
# identity and explicit BALL LOST / OFFSIDE flags).
#
# For every PASS event: positions at the kick frame -> torso margin of the TRUE
# receiver (the event's "To" player), plus the nearest-to-end-point and
# furthest-forward rules used on StatsBomb frames; velocities from finite
# differences feed a Monte Carlo over positional noise (sigma = 0.3 m) and kick
# timing (sigma_t = 0.02 s, half a frame). Frame-ambiguity sensitivity: the
# margin recomputed at kick frame +/- 1 and +/- 2 frames.
#
# Only two flagged offsides exist in these two games, so positives are case
# studies; the statistically useful number is the false-positive rate on the
# ~1,700 passes that were not flagged.
# Data: data/external/metrica/data/Sample_Game_{1,2} (CC BY 4.0, Metrica Sports).
# Writes data/offside_tracking_eval.rds, figures/offside_tracking_margins.png,
#        analysis/offside_tracking.md
# -----------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(dplyr); library(purrr); library(tibble); library(tidyr); library(ggplot2)
})
if (!exists("offside_frame")) source(file.path("R", "09_offside_engine.R"))

METRICA_DIR <- file.path("data", "external", "metrica", "data")
PL <- 105; PW <- 68          # Metrica sample pitch, metres
FPS <- 25

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
  players <- sub("_x$", "", grep("_x$", nm, value = TRUE))
  players <- setdiff(players, "Ball")
  long <- map_dfr(players, function(p) tibble(frame = dat$Frame, period = dat$Period, player = p, team = team,
                                              x = dat[[paste0(p, "_x")]] * PL, y = dat[[paste0(p, "_y")]] * PW))
  ball <- tibble(frame = dat$Frame, period = dat$Period, ball_x = dat$Ball_x * PL, ball_y = dat$Ball_y * PW)
  list(players = long %>% filter(!is.na(x)), ball = ball)
}

read_metrica_events <- function(game) {
  ev <- read.csv(file.path(METRICA_DIR, game, sprintf("%s_RawEventsData.csv", game)), stringsAsFactors = FALSE)
  names(ev) <- c("team", "type", "subtype", "period", "start_frame", "start_time", "end_frame", "end_time",
                 "from", "to", "start_x", "start_y", "end_x", "end_y")
  ev %>% mutate(idx = row_number(), game = game)
}

# Attacking direction per team and period from the kick-off frame: a team whose
# players stand in the left half at kick-off attacks to the right.
attacking_right <- function(players, team, period) {
  f0 <- min(players$frame[players$period == period])
  mean(players$x[players$team == team & players$period == period & players$frame == f0]) < PL / 2
}

evaluate_game <- function(game, sigma = 0.3, sigma_t = 0.02, n_mc = 300) {
  home <- read_metrica_tracking(game, "Home"); away <- read_metrica_tracking(game, "Away")
  players <- bind_rows(home$players, away$players)
  ball <- home$ball
  ev <- read_metrica_events(game)
  passes <- ev %>% filter(type == "PASS", to != "", !is.na(start_frame))
  # label: the next event is an OFFSIDE ball loss by the same team
  nxt <- ev[match(passes$idx + 1, ev$idx), ]
  passes$offside_flag <- as.integer(!is.na(nxt$subtype) & nxt$subtype == "OFFSIDE" & nxt$team == passes$team)
  dir <- expand_grid(team = c("Home", "Away"), period = unique(players$period)) %>%
    mutate(right = map2_lgl(team, period, ~ attacking_right(players, .x, .y)))

  one <- function(i, shift = 0) {
    p <- passes[i, ]; f <- p$start_frame + shift
    fr <- players %>% filter(frame == f)
    if (nrow(fr) < 14) return(NULL)
    right <- dir$right[dir$team == p$team & dir$period == p$period]
    xy <- to_attacking_frame(fr$x, fr$y, right, PL, PW, origin = "corner")
    fr$x <- xy$x; fr$y <- xy$y
    b <- ball[ball$frame == f, ]
    bxy <- if (nrow(b) && !is.na(b$ball_x)) { t <- to_attacking_frame(b$ball_x, b$ball_y, right, PL, PW); c(t$x, t$y) } else NULL
    frame <- tibble(x = fr$x, y = fr$y, teammate = fr$team == p$team,
                    actor = fr$team == p$team & fr$player == p$from, keeper = FALSE)
    if (!any(frame$actor)) return(NULL)
    if (is.null(bxy)) bxy <- c(frame$x[frame$actor][1], frame$y[frame$actor][1])
    rec <- which(fr$team == p$team & fr$player == p$to)
    if (!length(rec)) return(NULL)
    def_x <- frame$x[!frame$teammate]
    m_true <- offside_margin(frame$x[rec], def_x, bxy[1], PL)
    e <- to_attacking_frame(p$end_x * PL, p$end_y * PW, right, PL, PW)
    near <- offside_frame(frame, end_xy = c(e$x, e$y), ball_xy = bxy, receiver = "nearest_end", pitch_length = PL)
    mx <- offside_frame(frame, ball_xy = bxy, receiver = "max", pitch_length = PL)
    out <- tibble(margin_true = m_true, margin_near = near$margin, margin_max = mx$margin,
                  receiver_x = frame$x[rec], second_last_def_x = second_last_defender_x(def_x),
                  ball_x = bxy[1], n_def = sum(!frame$teammate), ball_from_tracking = nrow(b) > 0 && !is.na(b$ball_x),
                  near_rule_correct_receiver = isTRUE(all.equal(near$receiver_x, frame$x[rec])))
    if (shift == 0) {
      # velocities by central difference over +/- 2 frames, for timing MC
      f1 <- players %>% filter(frame == f - 2); f2 <- players %>% filter(frame == f + 2)
      v <- fr %>% select(team, player) %>%
        left_join(f1 %>% select(team, player, x1 = x, y1 = y), by = c("team", "player")) %>%
        left_join(f2 %>% select(team, player, x2 = x, y2 = y), by = c("team", "player"))
      sgn <- if (right) 1 else -1
      vx <- sgn * (v$x2 - v$x1) / (4 / FPS); vy <- sgn * (v$y2 - v$y1) / (4 / FPS)
      vx[is.na(vx)] <- 0; vy[is.na(vy)] <- 0
      frame_r <- frame; frame_r$x[rec] <- frame$x[rec]   # receiver fixed as the true one
      mc <- offside_mc(frame_r, end_xy = c(frame$x[rec], frame$y[rec]), ball_xy = bxy, receiver = "nearest_end",
                       sigma = sigma, sigma_t = sigma_t, vx = vx, vy = vy, n = n_mc, seed = 7406 + i, pitch_length = PL)
      out$p_offside <- mc$p_offside; out$margin_q05 <- mc$margin_q05; out$margin_q95 <- mc$margin_q95
      out$call3 <- offside_call(mc)
      out$receiver_speed <- sqrt(vx[rec]^2 + vy[rec]^2)
    }
    out
  }
  res <- map_dfr(seq_len(nrow(passes)), function(i) {
    r0 <- one(i, 0); if (is.null(r0)) return(NULL)
    shifts <- map_dbl(c(-2, -1, 1, 2), function(s) { r <- one(i, s); if (is.null(r)) NA_real_ else r$margin_true })
    bind_cols(passes[i, c("game", "idx", "team", "period", "start_frame", "from", "to", "offside_flag")], r0,
              tibble(margin_m2 = shifts[1], margin_m1 = shifts[2], margin_p1 = shifts[3], margin_p2 = shifts[4]))
  })
  res
}

wilson <- function(k, n, z = 1.96) {
  p <- k / n; d <- 1 + z^2 / n
  c <- (p + z^2 / (2 * n)) / d; h <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / d
  c(lo = c - h, hi = c + h)
}

run_offside_tracking <- function(games = c("Sample_Game_1", "Sample_Game_2")) {
  res <- map_dfr(games, function(g) { message("Evaluating ", g, "..."); evaluate_game(g) })
  message("Passes evaluated: ", nrow(res), " | flagged offside: ", sum(res$offside_flag))
  neg <- res %>% filter(offside_flag == 0); pos <- res %>% filter(offside_flag == 1)
  fpr <- function(col) { k <- sum(neg[[col]] > 0, na.rm = TRUE); n <- sum(!is.na(neg[[col]])); c(rate = k / n, wilson(k, n), n = n) }
  fp <- bind_rows(true = fpr("margin_true"), nearest_end = fpr("margin_near"), furthest_forward = fpr("margin_max"), .id = "receiver_rule")
  frame_shift <- res %>% summarise(
    median_abs_shift_1frame = median(abs(margin_p1 - margin_true), na.rm = TRUE),
    p90_abs_shift_1frame = quantile(abs(margin_p1 - margin_true), 0.9, na.rm = TRUE),
    median_abs_shift_2frames = median(abs(margin_p2 - margin_true), na.rm = TRUE),
    p90_abs_shift_2frames = quantile(abs(margin_p2 - margin_true), 0.9, na.rm = TRUE),
    share_sign_flip_1frame = mean(sign(margin_p1) != sign(margin_true), na.rm = TRUE),
    share_sign_flip_2frames = mean(sign(margin_p2) != sign(margin_true), na.rm = TRUE))
  mc_tab <- res %>% count(call3, offside_flag) %>%
    pivot_wider(names_from = offside_flag, values_from = n, values_fill = 0, names_prefix = "flag_")
  near_ok <- mean(res$near_rule_correct_receiver)

  p <- ggplot(neg, aes(margin_true)) +
    geom_histogram(binwidth = 1, fill = "#9ecae1", colour = "white") +
    geom_vline(xintercept = 0, linetype = 2) +
    geom_vline(data = pos, aes(xintercept = margin_true), colour = "#d95f0e", linewidth = 1) +
    labs(title = "Torso margin of the true receiver at the kick frame (Metrica games 1 and 2)",
         subtitle = sprintf("Blue: %d passes not flagged offside; orange lines: the %d flagged offsides", nrow(neg), nrow(pos)),
         x = "Margin beyond the offside line (metres; + = offside position)", y = "Passes") +
    theme_minimal(base_size = 12)
  dir.create("figures", showWarnings = FALSE)
  ggsave(file.path("figures", "offside_tracking_margins.png"), p, width = 8, height = 5, dpi = 300)

  saveRDS(list(passes = res, false_positive = fp, frame_shift = frame_shift, mc = mc_tab), file.path("data", "offside_tracking_eval.rds"))

  lines <- c("# Law 11 engine on continuous tracking (Metrica Sports sample games 1 and 2)", "",
             sprintf("Generated by `R/11_offside_tracking.R` on %s. Passes evaluated: %d (%d flagged offside by the data provider). Positions at the kick frame, 25 Hz, metres.", format(Sys.Date()), nrow(res), nrow(pos)), "",
             "## False-positive rate on passes that were NOT flagged offside", "",
             "| Receiver rule | Share with margin > 0 | 95% CI | n |", "|---|---|---|---|",
             sprintf("| %s | %.3f | [%.3f, %.3f] | %d |", fp$receiver_rule, fp$rate, fp$lo, fp$hi, as.integer(fp$n)), "",
             sprintf("The nearest-to-end-point rule picked the true receiver in %.1f%% of passes.", 100 * near_ok), "",
             "## The flagged offsides (case studies)", "",
             "| Game | Period | Kick frame | Passer | Receiver | Margin (m) | Second-last defender x | Ball x | P(offside) MC | Call |", "|---|---|---|---|---|---|---|---|---|---|",
             sprintf("| %s | %d | %d | %s | %s | %.2f | %.2f | %.2f | %.2f | %s |", pos$game, pos$period, pos$start_frame, pos$from, pos$to, pos$margin_true, pos$second_last_def_x, pos$ball_x, pos$p_offside, pos$call3), "",
             "## Kick-frame ambiguity (what one frame of timing error does to the margin)", "",
             sprintf("- Shifting the kick frame by 1 frame (40 ms) moves the margin by a median of %.2f m (90th percentile %.2f m); by 2 frames: median %.2f m, p90 %.2f m.",
                     frame_shift$median_abs_shift_1frame, frame_shift$p90_abs_shift_1frame, frame_shift$median_abs_shift_2frames, frame_shift$p90_abs_shift_2frames),
             sprintf("- The sign of the margin flips in %.1f%% of passes with a 1-frame shift and %.1f%% with a 2-frame shift.",
                     100 * frame_shift$share_sign_flip_1frame, 100 * frame_shift$share_sign_flip_2frames), "",
             "## Three-way Monte-Carlo call (sigma = 0.3 m, sigma_t = 0.02 s) vs. provider flag", "",
             "| Engine call | Not flagged | Flagged offside |", "|---|---|---|",
             sprintf("| %s | %d | %d |", mc_tab$call3, mc_tab$flag_0, if ("flag_1" %in% names(mc_tab)) mc_tab$flag_1 else 0L), "",
             "![Margins](../figures/offside_tracking_margins.png)", "",
             "## Caveats", "",
             "- Two flagged offsides in two games: sensitivity cannot be estimated here; the false-positive rate can.",
             "- A positive margin on an unflagged pass is not necessarily an error: an offside-positioned receiver who is not interfering is not flagged, and some passes are intercepted before any offence.",
             "- Tracking centroids are torso-level; limb-level offsets are not measurable without pose data.",
             "- Attacking direction is inferred from kick-off positions; ball position comes from tracking when available, else the passer's location.", "")
  writeLines(lines, file.path("analysis", "offside_tracking.md"))
  invisible(res)
}

if (sys.nframe() == 0) run_offside_tracking()
