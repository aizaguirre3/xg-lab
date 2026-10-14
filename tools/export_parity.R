#!/usr/bin/env Rscript
# tools/export_parity.R -------------------------------------------------------
# Export R reference values for the Python port (varlab) parity test.
#
#   data/parity/oof.csv            p = p_xgb, y = is_goal, fold, p_iso
#   data/parity/metrics_r.csv      metric, value  (log_loss, brier, ece, Murphy
#                                  components, n, base_rate, *_iso)
#   data/parity/reliability_r.csv  bin, mean_pred, obs_rate, n, se (decile bins)
#
# All numbers come from the reference implementations in R/utils.R; the
# equal-count bins and the leakage-safe isotonic map mirror reliability_data()
# and isotonic_recalibrate() in R/04_calibration.R line for line (re-stated
# here in base R so this script needs no dplyr/ggplot2).
#
# Usage:  Rscript tools/export_parity.R [in_path] [out_dir]
#         defaults: data/oof_preds.rds  data/parity
# If the input is absent (data/ is gitignored) the script says so and exits 0.
# Then run:  pytest tests/python/test_parity.py -q
# -----------------------------------------------------------------------------
args    <- commandArgs(trailingOnly = TRUE)
in_path <- if (length(args) >= 1) args[1] else file.path("data", "oof_preds.rds")
out_dir <- if (length(args) >= 2) args[2] else file.path("data", "parity")

if (!file.exists(in_path)) {
  cat(sprintf("export_parity: %s not found (it is produced by R/03_models.R and gitignored); nothing exported.\n", in_path))
  quit(save = "no", status = 0)
}
if (!file.exists(file.path("R", "utils.R")))
  stop("run from the repository root (R/utils.R not found)")
source(file.path("R", "utils.R"))

oof <- readRDS(in_path)
needed <- c("p_xgb", "is_goal", "fold")
if (!all(needed %in% names(oof)))
  stop("input lacks columns: ", paste(setdiff(needed, names(oof)), collapse = ", "))

p    <- as.numeric(oof$p_xgb)
y    <- as.integer(oof$is_goal)
fold <- as.integer(oof$fold)
ok   <- is.finite(p) & !is.na(y) & !is.na(fold)
if (any(!ok)) {
  cat(sprintf("export_parity: dropping %d rows with NA/NaN p, y or fold\n", sum(!ok)))
  p <- p[ok]; y <- y[ok]; fold <- fold[ok]
}
N <- length(p)

# -- leakage-safe isotonic map: verbatim logic of isotonic_recalibrate() ------
p_iso <- rep(NA_real_, N)
for (f in sort(unique(fold))) {
  cal_p <- p[fold != f]; cal_y <- y[fold != f]; tst <- which(fold == f)
  ord <- order(cal_p)
  fit <- stats::isoreg(cal_p[ord], cal_y[ord])
  iso_fun <- stats::approxfun(cal_p[ord], fit$yf,
                              method = "linear", rule = 2, ties = "ordered")
  p_iso[tst] <- pmin(pmax(iso_fun(p[tst]), 0), 1)
}

# -- scores -------------------------------------------------------------------
d <- brier_decomposition(p, y, n_bins = 10)
metrics <- data.frame(
  metric = c("n", "base_rate", "log_loss", "brier", "ece",
             "reliability", "resolution", "uncertainty", "recomposed",
             "ece_iso", "log_loss_iso", "brier_iso"),
  value  = c(N, mean(y), log_loss(p, y), brier_score(p, y),
             expected_calibration_error(p, y, n_bins = 10),
             d$reliability, d$resolution, d$uncertainty, d$recomposed,
             expected_calibration_error(p_iso, y, n_bins = 10),
             log_loss(p_iso, y), brier_score(p_iso, y)),
  stringsAsFactors = FALSE)

# -- equal-count bins: verbatim logic of reliability_data() -------------------
q <- unique(quantile(p, probs = seq(0, 1, length.out = 11), na.rm = TRUE))
b <- cut(p, q, include.lowest = TRUE, labels = FALSE)
bins <- sort(unique(b))
rel <- data.frame(
  bin       = bins,
  mean_pred = vapply(bins, function(k) mean(p[b == k]), numeric(1)),
  obs_rate  = vapply(bins, function(k) mean(y[b == k]), numeric(1)),
  n         = vapply(bins, function(k) sum(b == k), integer(1)))
rel$se <- sqrt(pmax(rel$obs_rate * (1 - rel$obs_rate), 1e-6) / rel$n)

# -- write (17 significant digits so the CSV round-trips the doubles) ---------
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
fmt <- function(x) formatC(x, digits = 17, format = "g")
oof_out <- data.frame(p = fmt(p), y = y, fold = fold, p_iso = fmt(p_iso))
metrics_out <- data.frame(metric = metrics$metric, value = fmt(metrics$value))
rel_out <- data.frame(bin = rel$bin, mean_pred = fmt(rel$mean_pred),
                      obs_rate = fmt(rel$obs_rate), n = rel$n, se = fmt(rel$se))
write.csv(oof_out,     file.path(out_dir, "oof.csv"),            row.names = FALSE, quote = FALSE)
write.csv(metrics_out, file.path(out_dir, "metrics_r.csv"),      row.names = FALSE, quote = FALSE)
write.csv(rel_out,     file.path(out_dir, "reliability_r.csv"),  row.names = FALSE, quote = FALSE)

cat(sprintf("export_parity: wrote %s (%d rows), %s and %s\n",
            file.path(out_dir, "oof.csv"), N,
            file.path(out_dir, "metrics_r.csv"), file.path(out_dir, "reliability_r.csv")))
print(metrics, digits = 10, row.names = FALSE)
