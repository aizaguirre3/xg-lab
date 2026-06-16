# R/04_calibration.R ----------------------------------------------------------
# Calibration rigor on the OOF predictions: reliability diagram, ECE, Murphy
# (Brier) decomposition, and an honest benchmark vs StatsBomb's own xG. If the
# GBM is miscalibrated, apply leakage-safe isotonic recalibration and show the
# before/after. Writes figures/reliability.png and data/*.rds tables.
# -----------------------------------------------------------------------------
suppressPackageStartupMessages({ library(dplyr); library(ggplot2); library(tidyr) })
if (!exists("expected_calibration_error")) source(file.path("R", "utils.R"))

MODELS <- c(logistic = "p_logistic", gam = "p_gam",
            xgboost = "p_xgb", statsbomb_xg = "statsbomb_xg")

# Per-model calibration scores.
calib_scores <- function(oof) {
  purrr_map <- function(nm, col) {
    p <- oof[[col]]; y <- oof$is_goal
    d <- brier_decomposition(p, y, n_bins = 10)
    tibble::tibble(model = nm,
                   log_loss = log_loss(p, y), brier = d$brier,
                   ece = expected_calibration_error(p, y, 10),
                   reliability = d$reliability, resolution = d$resolution,
                   uncertainty = d$uncertainty)
  }
  do.call(rbind, Map(purrr_map, names(MODELS), MODELS))
}

# Reliability-diagram data: equal-count (decile) bins with Wilson-ish CIs.
reliability_data <- function(oof, n_bins = 10) {
  do.call(rbind, lapply(names(MODELS), function(nm) {
    p <- oof[[MODELS[nm]]]; y <- oof$is_goal
    q <- unique(quantile(p, probs = seq(0, 1, length.out = n_bins + 1), na.rm = TRUE))
    b <- cut(p, q, include.lowest = TRUE, labels = FALSE)
    tibble::tibble(model = nm, p = p, y = y, bin = b) %>%
      group_by(model, bin) %>%
      summarise(mean_pred = mean(p), obs_rate = mean(y), n = dplyr::n(),
                se = sqrt(pmax(obs_rate * (1 - obs_rate), 1e-6) / n),
                .groups = "drop")
  }))
}

# Leakage-safe isotonic recalibration of xgboost: for each fold, fit the
# isotonic map on the OTHER folds' OOF preds, apply to this fold.
isotonic_recalibrate <- function(oof) {
  oof$p_xgb_iso <- NA_real_
  for (f in sort(unique(oof$fold))) {
    cal <- oof[oof$fold != f, ]; tst <- which(oof$fold == f)
    ord <- order(cal$p_xgb)
    fit <- stats::isoreg(cal$p_xgb[ord], cal$is_goal[ord])
    # step-function interpolation of the fitted isotonic curve
    iso_fun <- stats::approxfun(cal$p_xgb[ord], fit$yf,
                                method = "linear", rule = 2, ties = "ordered")
    oof$p_xgb_iso[tst] <- pmin(pmax(iso_fun(oof$p_xgb[tst]), 0), 1)
  }
  oof
}

run_calibration <- function(in_path = file.path("data", "oof_preds.rds")) {
  oof <- readRDS(in_path)

  scores <- calib_scores(oof)
  message("Calibration / benchmark scores (OOF):")
  print(as.data.frame(scores), digits = 4)
  saveRDS(scores, file.path("data", "calibration_scores.rds"))

  # reliability diagram
  rd <- reliability_data(oof)
  p <- ggplot(rd, aes(mean_pred, obs_rate, colour = model)) +
    geom_abline(slope = 1, intercept = 0, linetype = 2, colour = "grey50") +
    geom_errorbar(aes(ymin = obs_rate - 1.96 * se, ymax = obs_rate + 1.96 * se),
                  width = 0, alpha = 0.5) +
    geom_line() + geom_point(aes(size = n), alpha = 0.8) +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(title = "Reliability diagram (out-of-fold)",
         subtitle = "Closer to the dashed diagonal = better calibrated",
         x = "Mean predicted xG", y = "Observed goal rate",
         colour = "Model", size = "Shots in bin") +
    theme_minimal(base_size = 12)
  ggsave(file.path("figures", "reliability.png"), p, width = 7, height = 6, dpi = 300)
  message("Wrote figures/reliability.png")

  # isotonic recalibration of xgboost (before/after)
  oof <- isotonic_recalibrate(oof)
  before <- expected_calibration_error(oof$p_xgb, oof$is_goal)
  after  <- expected_calibration_error(oof$p_xgb_iso, oof$is_goal)
  recal <- tibble::tibble(
    metric = c("ECE", "log_loss", "brier"),
    xgb_before = c(before, log_loss(oof$p_xgb, oof$is_goal), brier_score(oof$p_xgb, oof$is_goal)),
    xgb_after  = c(after,  log_loss(oof$p_xgb_iso, oof$is_goal), brier_score(oof$p_xgb_iso, oof$is_goal)))
  message("Isotonic recalibration of xgboost (before/after):")
  print(as.data.frame(recal), digits = 4)
  saveRDS(recal, file.path("data", "recalibration.rds"))
  saveRDS(oof, file.path("data", "oof_preds_recal.rds"))
  invisible(scores)
}

if (sys.nframe() == 0 || identical(environment(), globalenv())) run_calibration()