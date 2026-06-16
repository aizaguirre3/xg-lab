# R/03_models.R ---------------------------------------------------------------
# Three xG models on IDENTICAL match-grouped CV folds (shots from one match
# never span train/test -> leakage-safe). Saves out-of-fold (OOF) predictions
# for every shot, so Phase 4 calibration is honest.
#   A logistic  : interpretable baseline (distance + angle + body part)
#   B GAM       : mgcv 2-D spatial smooth s(x,y) + context
#   C xgboost   : full feature set, nrounds chosen by inner CV
# -----------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(dplyr); library(mgcv); library(xgboost)
})
if (!exists("log_loss")) source(file.path("R", "utils.R"))

N_FOLDS <- 5
SEED    <- 7406

# Numeric feature set for xgboost (statsbomb_xg is NEVER included).
XGB_FEATURES <- c("distance", "angle", "x", "y",
                  "is_open_play", "is_free_kick", "under_pressure_i",
                  "first_time_i", "n_def_cone", "dist_nearest_def",
                  "gk_dist_goal", "ff_missing", "body_foot", "body_head")

prep <- function(df) {
  df %>% mutate(
    body_cat        = factor(body_cat, levels = c("foot", "head", "other")),
    under_pressure_i = as.integer(under_pressure),
    first_time_i     = as.integer(first_time),
    body_foot        = as.integer(body_cat == "foot"),
    body_head        = as.integer(body_cat == "head")
  )
}

# Assign each MATCH (not row) to a fold -> grouped CV.
make_folds <- function(df, k = N_FOLDS, seed = SEED) {
  set.seed(seed)
  m <- unique(df$match_id)
  fold_of <- setNames(sample(rep_len(seq_len(k), length(m))), m)
  unname(fold_of[as.character(df$match_id)])
}

fit_predict_fold <- function(train, test) {
  # A: logistic
  m_log <- glm(is_goal ~ distance + angle + body_cat,
               data = train, family = binomial())
  p_log <- predict(m_log, test, type = "response")

  # B: GAM with spatial smooth + context
  m_gam <- mgcv::gam(
    is_goal ~ s(x, y) + body_cat + is_open_play + is_free_kick +
      under_pressure_i + first_time_i + s(dist_nearest_def) + s(n_def_cone, k = 5),
    data = train, family = binomial(), method = "REML")
  p_gam <- as.numeric(predict(m_gam, test, type = "response"))

  # C: xgboost, nrounds via inner CV (early stopping) on the training fold
  dtrain <- xgb.DMatrix(as.matrix(train[, XGB_FEATURES]), label = train$is_goal)
  dtest  <- xgb.DMatrix(as.matrix(test[, XGB_FEATURES]))
  params <- list(objective = "binary:logistic", eval_metric = "logloss",
                 max_depth = 4, eta = 0.05, subsample = 0.8,
                 colsample_bytree = 0.8, min_child_weight = 5)
  set.seed(SEED)
  cv <- xgb.cv(params, dtrain, nrounds = 600, nfold = 4,
               early_stopping_rounds = 30, verbose = 0)
  # xgboost >= 3 leaves cv$best_iteration empty; derive it from the eval log.
  best <- cv$best_iteration
  if (is.null(best) || length(best) == 0 || is.na(best)) {
    best <- which.min(cv$evaluation_log$test_logloss_mean)
  }
  best <- max(as.integer(best), 1L)
  m_xgb <- xgb.train(params, dtrain, nrounds = best, verbose = 0)
  p_xgb <- predict(m_xgb, dtest)

  list(p_log = p_log, p_gam = p_gam, p_xgb = p_xgb, xgb_rounds = best)
}

run_models <- function(in_path  = file.path("data", "shots_model.rds"),
                       out_path = file.path("data", "oof_preds.rds")) {
  df <- prep(readRDS(in_path))
  df$fold <- make_folds(df)
  message("Shots: ", nrow(df), " | matches: ", length(unique(df$match_id)),
          " | folds: ", N_FOLDS, " | base rate: ", round(mean(df$is_goal), 3))

  oof <- df %>% transmute(shot_id, match_id, competition, player_id, player_name,
                          is_goal, statsbomb_xg, fold,
                          p_logistic = NA_real_, p_gam = NA_real_, p_xgb = NA_real_)

  for (f in seq_len(N_FOLDS)) {
    tr <- df[df$fold != f, ]; te_idx <- which(df$fold == f)
    res <- fit_predict_fold(tr, df[te_idx, ])
    oof$p_logistic[te_idx] <- res$p_log
    oof$p_gam[te_idx]      <- res$p_gam
    oof$p_xgb[te_idx]      <- res$p_xgb
    message(sprintf("  fold %d: n_test=%d  xgb_rounds=%d", f, length(te_idx), res$xgb_rounds))
  }

  # quick OOF leaderboard (log loss + Brier), incl. StatsBomb benchmark
  loss_tbl <- tibble::tibble(
    model = c("logistic", "gam", "xgboost", "statsbomb_xg"),
    log_loss = c(log_loss(oof$p_logistic, oof$is_goal),
                 log_loss(oof$p_gam,      oof$is_goal),
                 log_loss(oof$p_xgb,      oof$is_goal),
                 log_loss(oof$statsbomb_xg, oof$is_goal)),
    brier = c(brier_score(oof$p_logistic, oof$is_goal),
              brier_score(oof$p_gam,      oof$is_goal),
              brier_score(oof$p_xgb,      oof$is_goal),
              brier_score(oof$statsbomb_xg, oof$is_goal))
  ) %>% arrange(log_loss)
  print(loss_tbl)

  saveRDS(oof, out_path)
  saveRDS(loss_tbl, file.path("data", "model_leaderboard.rds"))

  # also fit full-data models for EDA / inspection (not used for OOF metrics)
  full_gam <- mgcv::gam(
    is_goal ~ s(x, y) + body_cat + is_open_play + is_free_kick +
      under_pressure_i + first_time_i + s(dist_nearest_def) + s(n_def_cone, k = 5),
    data = df, family = binomial(), method = "REML")
  saveRDS(full_gam, file.path("data", "gam_full.rds"))
  invisible(oof)
}

if (sys.nframe() == 0) run_models()