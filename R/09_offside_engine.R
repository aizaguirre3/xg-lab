# R/09_offside_engine.R -------------------------------------------------------
# VAR-Lab layer 2: a pure Law 11 geometry engine with Monte-Carlo uncertainty.
#
# Law 11 (offside position): a player is in an offside position if any part of
# the head, body or feet is in the opponents' half (excluding the halfway line)
# AND nearer to the opponents' goal line than both the ball and the second-last
# opponent. Being level is onside. Whether an offside POSITION becomes an
# offside OFFENCE (interfering with play / an opponent, gaining an advantage) is
# a human judgement and is not modelled here.
#
# Everything is expressed in an "attacking-right" frame: the attacking team
# shoots toward x = pitch_length, the halfway line is x = pitch_length / 2.
# Positions are 2D torso-level points (StatsBomb 360 frames, tracking
# centroids), so the engine measures a torso margin, not a limb margin.
#
# Pure functions, no I/O. Unit-tested in tests/test_offside.R.
# -----------------------------------------------------------------------------

#' x of the second-last defender (keeper counts like any other opponent).
#' Returns NA when fewer than two defenders are supplied.
second_last_defender_x <- function(def_x) {
  def_x <- def_x[!is.na(def_x)]
  if (length(def_x) < 2) return(NA_real_)
  sort(def_x, decreasing = TRUE)[2]
}

#' The line a receiver must not be beyond: the furthest-forward of the
#' second-last defender, the ball, and the halfway line.
offside_reference_x <- function(def_x, ball_x, pitch_length = 120) {
  sld <- second_last_defender_x(def_x)
  if (is.na(sld) || is.na(ball_x)) return(NA_real_)
  max(sld, ball_x, pitch_length / 2)
}

#' Offside margin for one or more receivers: receiver x minus the reference
#' line. Positive = in an offside position; zero or negative = onside.
#' @param rec_x numeric vector of receiver x positions (attacking right)
#' @param def_x numeric vector of ALL defending players' x (keeper included)
#' @param ball_x x of the ball at the moment it is played
offside_margin <- function(rec_x, def_x, ball_x, pitch_length = 120) {
  ref <- offside_reference_x(def_x, ball_x, pitch_length)
  rec_x - ref
}

#' Convert raw coordinates into the attacking-right frame.
#' @param x,y raw coordinates; @param attacking_right TRUE if the team attacks
#'   toward +x already; @param origin "corner" (StatsBomb: x in [0,L]) or
#'   "centre" (tracking: x in [-L/2, L/2])
to_attacking_frame <- function(x, y, attacking_right, pitch_length = 120, pitch_width = 80,
                               origin = c("corner", "centre")) {
  origin <- match.arg(origin)
  if (origin == "centre") { x <- x + pitch_length / 2; y <- y + pitch_width / 2 }
  if (!attacking_right) { x <- pitch_length - x; y <- pitch_width - y }
  list(x = x, y = y)
}

#' Evaluate a compacted frame (tibble with x, y, teammate, actor, keeper, already
#' in the passer's attacking-right frame) at the moment of a pass.
#'
#' The ball is taken to be at the passer (actor) location, or `ball_xy` when
#' given. The receiver is chosen by `receiver`:
#'   "nearest_end": the teammate (not the actor) nearest to `end_xy`
#'   "max":         the teammate with the largest margin (the furthest forward)
#' Returns a one-row tibble; `ok` is FALSE when the frame cannot be evaluated
#' (no actor/ball, fewer than two defenders, or no candidate receiver).
offside_frame <- function(frame, end_xy = NULL, ball_xy = NULL,
                          receiver = c("nearest_end", "max"), pitch_length = 120) {
  receiver <- match.arg(receiver)
  bad <- tibble::tibble(ok = FALSE, receiver_x = NA_real_, receiver_y = NA_real_,
                        second_last_def_x = NA_real_, ball_x = NA_real_,
                        reference_x = NA_real_, margin = NA_real_,
                        n_def = NA_integer_, n_att = NA_integer_, receiver_rule = receiver)
  if (is.null(frame) || nrow(frame) == 0) return(bad)
  if (is.null(ball_xy)) {
    act <- frame[frame$actor, ]
    if (nrow(act) == 0) return(bad)
    ball_xy <- c(act$x[1], act$y[1])
  }
  def <- frame[!frame$teammate, ]
  att <- frame[frame$teammate & !frame$actor, ]
  if (nrow(def) < 2 || nrow(att) == 0) return(bad)
  ref <- offside_reference_x(def$x, ball_xy[1], pitch_length)
  if (is.na(ref)) return(bad)
  m <- att$x - ref
  i <- if (receiver == "max") which.max(m) else {
    if (is.null(end_xy) || any(is.na(end_xy))) return(bad)
    which.min((att$x - end_xy[1])^2 + (att$y - end_xy[2])^2)
  }
  tibble::tibble(ok = TRUE, receiver_x = att$x[i], receiver_y = att$y[i],
                 second_last_def_x = second_last_defender_x(def$x), ball_x = ball_xy[1],
                 reference_x = ref, margin = m[i],
                 n_def = nrow(def), n_att = nrow(att), receiver_rule = receiver)
}

#' Monte-Carlo uncertainty for a frame: jitter every position (and the ball)
#' with isotropic Gaussian noise of sd `sigma` (pitch units), optionally shift
#' the kick time by N(0, sigma_t) seconds along supplied velocities, and
#' re-evaluate. Returns P(offside), the median margin and a 90% interval.
#' @param vx,vy optional per-player velocities (pitch units per second) in the
#'   same row order as `frame`; NULL = no timing uncertainty
offside_mc <- function(frame, end_xy = NULL, ball_xy = NULL, receiver = "nearest_end",
                       sigma = 1, sigma_t = 0, vx = NULL, vy = NULL,
                       n = 500, seed = 7406, pitch_length = 120) {
  base <- offside_frame(frame, end_xy, ball_xy, receiver, pitch_length)
  if (!base$ok) return(tibble::tibble(p_offside = NA_real_, margin_median = NA_real_,
                                      margin_q05 = NA_real_, margin_q95 = NA_real_, n = n))
  set.seed(seed)
  k <- nrow(frame)
  ms <- vapply(seq_len(n), function(j) {
    dt <- if (sigma_t > 0) stats::rnorm(1, 0, sigma_t) else 0
    fr <- frame
    fr$x <- frame$x + stats::rnorm(k, 0, sigma) + if (!is.null(vx)) vx * dt else 0
    fr$y <- frame$y + stats::rnorm(k, 0, sigma) + if (!is.null(vy)) vy * dt else 0
    b <- if (is.null(ball_xy)) NULL else ball_xy + stats::rnorm(2, 0, sigma)
    r <- offside_frame(fr, end_xy, b, receiver, pitch_length)
    if (r$ok) r$margin else NA_real_
  }, numeric(1))
  ms <- ms[!is.na(ms)]
  tibble::tibble(p_offside = mean(ms > 0), margin_median = stats::median(ms),
                 margin_q05 = unname(stats::quantile(ms, 0.05)),
                 margin_q95 = unname(stats::quantile(ms, 0.95)), n = length(ms))
}

#' Three-way call with abstention: "offside" / "onside" when the 90% interval
#' is clear of zero, else "refer" (too close to call at this precision).
offside_call <- function(mc) {
  dplyr::case_when(
    is.na(mc$p_offside) ~ "no_frame",
    mc$margin_q05 > 0 ~ "offside",
    mc$margin_q95 <= 0 ~ "onside",
    TRUE ~ "refer")
}

#' Ray-casting point-in-polygon for a flat polygon vector c(x1,y1,x2,y2,...),
#' as StatsBomb 360 `visible_area` is stored. Points on an edge count as inside
#' for practical purposes (ties are rare with real coordinates).
point_in_polygon <- function(px, py, poly) {
  if (is.null(poly) || length(poly) < 6 || is.na(px) || is.na(py)) return(NA)
  x <- poly[seq(1, length(poly), 2)]; y <- poly[seq(2, length(poly), 2)]
  n <- length(x); inside <- FALSE; j <- n
  for (i in seq_len(n)) {
    if (((y[i] > py) != (y[j] > py)) &&
        (px < (x[j] - x[i]) * (py - y[i]) / (y[j] - y[i]) + x[i])) inside <- !inside
    j <- i
  }
  inside
}

#' Vectorised Monte Carlo with a FIXED receiver (chosen once from the unjittered
#' frame): much faster than offside_mc for thousands of frames. Jitters every
#' player and the ball with N(0, sigma) in x (only x matters for the margin) and,
#' when velocities and sigma_t are supplied, shifts every player along its
#' velocity by a kick-time error dt ~ N(0, sigma_t) shared within a draw.
#' Returns the same columns as offside_mc.
#' @param vx optional per-player x velocity (pitch units per second), same row
#'   order as `frame`; NULL = no timing uncertainty
offside_mc_fast <- function(frame, end_xy = NULL, ball_xy = NULL, receiver = "nearest_end",
                            sigma = 1, sigma_t = 0, vx = NULL,
                            n = 500, seed = 7406, pitch_length = 120) {
  base <- offside_frame(frame, end_xy, ball_xy, receiver, pitch_length)
  if (!base$ok) return(tibble::tibble(p_offside = NA_real_, margin_median = NA_real_,
                                      margin_q05 = NA_real_, margin_q95 = NA_real_, n = n))
  set.seed(seed)
  is_def <- !frame$teammate
  def_x <- frame$x[is_def]
  att <- which(frame$teammate & !frame$actor)
  rec_i <- att[which.min(abs(frame$x[att] - base$receiver_x) + abs(frame$y[att] - base$receiver_y))]
  dt <- if (sigma_t > 0 && !is.null(vx)) stats::rnorm(n, 0, sigma_t) else rep(0, n)
  vdef <- if (!is.null(vx)) vx[is_def] else rep(0, length(def_x))
  vrec <- if (!is.null(vx)) vx[rec_i] else 0
  D <- matrix(def_x, nrow = n, ncol = length(def_x), byrow = TRUE) +
    matrix(stats::rnorm(n * length(def_x), 0, sigma), nrow = n) + outer(dt, vdef)
  sld <- apply(D, 1, function(r) sort(r, decreasing = TRUE)[2])
  ball <- base$ball_x + stats::rnorm(n, 0, sigma)
  rec  <- base$receiver_x + stats::rnorm(n, 0, sigma) + dt * vrec
  ms <- rec - pmax(sld, ball, pitch_length / 2)
  tibble::tibble(p_offside = mean(ms > 0), margin_median = stats::median(ms),
                 margin_q05 = unname(stats::quantile(ms, 0.05)),
                 margin_q95 = unname(stats::quantile(ms, 0.95)), n = n)
}
