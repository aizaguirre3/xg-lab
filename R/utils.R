# utils.R ---------------------------------------------------------------------
# Pure, side-effect-free helpers: pitch geometry and probabilistic-forecast
# scoring. Everything here is unit-tested in tests/test_utils.R against
# hand-computed values. No I/O, no global state.
# -----------------------------------------------------------------------------

# StatsBomb pitch convention: 120 (length) x 80 (width). The attacked goal is at
# x = 120, mouth centred at y = 40, 8 yards wide -> posts at y = 36 and y = 44.
GOAL_X <- 120
GOAL_Y <- 40
POST_LEFT  <- c(GOAL_X, 36)
POST_RIGHT <- c(GOAL_X, 44)

#' Euclidean distance from a shot location to the goal centre.
#' @param x,y numeric vectors of pitch coordinates (StatsBomb frame).
#' @return numeric vector of distances in yards.
distance_to_goal <- function(x, y) {
  sqrt((GOAL_X - x)^2 + (GOAL_Y - y)^2)
}

#' Visible goal-mouth angle subtended by the two posts at the shot location.
#'
#' Computed from the two post vectors via atan2(|cross|, dot), which is robust
#' across the whole pitch (no division-by-zero, correct in the obtuse region
#' behind the goal line). Returns radians in [0, pi].
#'
#' Degenerate cases (documented + tested): exactly on the goal line between the
#' posts -> pi; exactly on a post -> 0 (zero-length vector).
#' @param x,y numeric vectors of pitch coordinates.
#' @return numeric vector of angles in radians.
shot_angle <- function(x, y) {
  ax <- POST_LEFT[1]  - x; ay <- POST_LEFT[2]  - y
  bx <- POST_RIGHT[1] - x; by <- POST_RIGHT[2] - y
  dot   <- ax * bx + ay * by
  cross <- ax * by - ay * bx
  atan2(abs(cross), dot)
}

#' Is point P inside the triangle (A,B,C)? Used for "defenders in the shooting
#' cone" = opponents inside the triangle shooter -> left post -> right post.
#' Uses the same-sign-of-cross-products test; points on an edge count as inside.
#' Scalar P, scalar triangle vertices (length-2 numeric each).
#' @return logical scalar.
point_in_triangle <- function(px, py, ax, ay, bx, by, cx, cy) {
  d1 <- (px - bx) * (ay - by) - (ax - bx) * (py - by)
  d2 <- (px - cx) * (by - cy) - (bx - cx) * (py - cy)
  d3 <- (px - ax) * (cy - ay) - (cx - ax) * (py - ay)
  has_neg <- (d1 < 0) || (d2 < 0) || (d3 < 0)
  has_pos <- (d1 > 0) || (d2 > 0) || (d3 > 0)
  !(has_neg && has_pos)
}

#' Log loss (binary cross-entropy), the primary xG ranking-and-calibration loss.
#' @param p predicted probabilities in (0,1); @param y 0/1 outcomes.
#' @param eps clamp to avoid log(0).
#' @return mean negative log-likelihood.
log_loss <- function(p, y, eps = 1e-15) {
  p <- pmin(pmax(p, eps), 1 - eps)
  -mean(y * log(p) + (1 - y) * log(1 - p))
}

#' Brier score = mean squared error of the probability forecast.
brier_score <- function(p, y) mean((p - y)^2)

#' Expected Calibration Error with equal-width bins.
#'
#' ECE = sum_k (n_k / N) * |mean_pred_k - mean_obs_k|, over occupied bins.
#' @param p predicted probabilities; @param y 0/1 outcomes; @param n_bins bins.
#' @return scalar ECE in [0,1].
expected_calibration_error <- function(p, y, n_bins = 10) {
  breaks <- seq(0, 1, length.out = n_bins + 1)
  bin <- cut(p, breaks = breaks, include.lowest = TRUE, labels = FALSE)
  N <- length(p)
  ece <- 0
  for (k in unique(bin)) {
    idx <- bin == k
    ece <- ece + (sum(idx) / N) * abs(mean(p[idx]) - mean(y[idx]))
  }
  ece
}

#' Murphy decomposition of the Brier score via binning.
#'
#' Brier ~= reliability - resolution + uncertainty, where
#'   uncertainty = obar (1 - obar),                       obar = overall base rate
#'   reliability = (1/N) sum_k n_k (pbar_k - obar_k)^2     (lower = better calibrated)
#'   resolution  = (1/N) sum_k n_k (obar_k - obar)^2       (higher = more informative)
#' The identity is exact only in the binning limit; we return the binned
#' components plus the raw Brier so the approximation gap is visible.
#' @return named list: reliability, resolution, uncertainty, brier, recomposed.
brier_decomposition <- function(p, y, n_bins = 10) {
  breaks <- seq(0, 1, length.out = n_bins + 1)
  bin <- cut(p, breaks = breaks, include.lowest = TRUE, labels = FALSE)
  N <- length(p)
  obar <- mean(y)
  reliability <- 0
  resolution  <- 0
  for (k in unique(bin)) {
    idx <- bin == k
    nk <- sum(idx)
    pbar_k <- mean(p[idx])
    obar_k <- mean(y[idx])
    reliability <- reliability + nk * (pbar_k - obar_k)^2
    resolution  <- resolution  + nk * (obar_k - obar)^2
  }
  reliability <- reliability / N
  resolution  <- resolution  / N
  uncertainty <- obar * (1 - obar)
  list(
    reliability = reliability,
    resolution  = resolution,
    uncertainty = uncertainty,
    brier       = brier_score(p, y),
    recomposed  = reliability - resolution + uncertainty
  )
}
