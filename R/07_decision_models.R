# R/07_decision_models.R ------------------------------------------------------
# VAR-Lab layer 1: calibrated refereeing-decision models.
#
#   Target A  P(card | foul committed)          all Foul Committed events
#   Target B  P(penalty | foul in own box)      Foul Committed with in_own_box
#
# Two models per target on IDENTICAL grouped CV folds (logistic baseline and an
# mgcv GAM with a spatial smooth + context + 360-frame features), evaluated two
# ways: match-grouped folds (the xg-lab discipline) and referee-grouped folds
# (how well the model transports to an unseen referee). Out-of-fold predictions
# are scored with the xg-lab calibration helpers (log loss, Brier, ECE, Murphy
# decomposition), with cluster-bootstrap CIs over matches, leakage-safe per-fold
# isotonic recalibration, and an abstention ("refer to review") curve.
#
# Writes data/decision_oof_<target>.rds, data/decision_scores.rds,
#        figures/decision_reliability.png, figures/decision_abstention.png,
#        analysis/decision_models.md
# xgboost is deliberately absent: CRAN is unreachable from the build sandbox and
# the GAM already covers the flexible end for ~10k rows.
# -----------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(dplyr); library(purrr); library(tibble); library(tidyr); library(mgcv); library(ggplot2)
})
if (!exists("log_loss")) source(file.path("R", "utils.R"))

N_FOLDS <- 5
SEED    <- 7406
N_BOOT  <- as.integer(Sys.getenv("XG_DEC_BOOT", "500"))

logit <- function(p, eps = 1e-6) { p <- pmin(pmax(p, eps), 1 - eps); log(p / (1 - p)) }

# ---- feature engineering ------------------------------------------------------

position_group <- function(pos) {
  dplyr::case_when(
    is.na(pos) ~ "Other",
    grepl("Goalkeeper", pos) ~ "GK",
    grepl("Back", pos) ~ "DEF",
    grepl("Midfield", pos) ~ "MID",
    grepl("Wing|Forward|Striker", pos) ~ "FWD",
    TRUE ~ "Other")
}

# 360-frame features for one foul, from the FOULING team's perspective
# (own goal at x = 0). NA when the frame is missing.
foul_frame_features <- function(fr, fx, fy) {
  na_out <- tibble(n_teammates_goalside = NA_real_, dist_nearest_opp = NA_real_,
                   n_opp_within_5 = NA_real_)
  if (is.null(fr) || is.na(fx) || is.na(fy)) return(na_out)
  tm <- fr[fr$teammate & !fr$actor, ]
  op <- fr[!fr$teammate, ]
  d_op <- if (nrow(op)) sqrt((op$x - fx)^2 + (op$y - fy)^2) else numeric(0)
  tibble(
    # teammates (incl. keeper) between the foul and the fouling team's own goal:
    # 0 or 1 means the fouler was (nearly) the last man -> DOGSO territory
    n_teammates_goalside = sum(tm$x < fx),
    dist_nearest_opp = if (length(d_op)) min(d_op) else NA_real_,
    n_opp_within_5 = sum(d_op <= 5))
}

# Build the modelling table for the card target (all fouls).
build_foul_table <- function(dec) {
  f <- dec %>% filter(type == "Foul Committed", !is.na(x), !is.na(y))
  ff <- map_dfr(seq_len(nrow(f)), function(i) foul_frame_features(f$frame[[i]], f$x[i], f$y[i]))
  f <- bind_cols(f %>% select(-any_of(c("n_teammates_goalside", "dist_nearest_opp", "n_opp_within_5"))), ff)
  med <- function(v) ifelse(is.na(v), stats::median(v, na.rm = TRUE), v)
  f %>% mutate(
    y_card    = as.integer(!is.na(card)),
    y_penalty = as.integer(penalty),
    dist_own_goal = sqrt(x^2 + (y - 40)^2),
    foul_type_f = dplyr::coalesce(foul_type, "Normal"),
    pos_group   = position_group(position),
    is_women    = as.integer(grepl("Women", competition)),
    is_international = as.integer(grepl("World Cup|Euro|Cup of Nations", competition)),
    advantage_i = as.integer(advantage), offensive_i = as.integer(offensive),
    under_pressure_i = as.integer(under_pressure),
    minute = pmin(minute, 125),
    ff_missing = as.integer(has_frame == 0 | is.na(n_teammates_goalside)),
    n_teammates_goalside = med(n_teammates_goalside),
    dist_nearest_opp = med(dist_nearest_opp),
    n_opp_within_5 = med(n_opp_within_5),
    referee_key = dplyr::coalesce(as.character(referee_id), paste0("m", match_id)))
}

# Lump rare factor levels (seen fewer than min_n times) into "Other".
lump <- function(x, min_n = 30) {
  tab <- table(x); keep <- names(tab)[tab >= min_n]
  factor(ifelse(x %in% keep, x, "Other"))
}

# ---- grouped folds --------------------------------------------------------------

make_group_folds <- function(groups, k = N_FOLDS, seed = SEED) {
  set.seed(seed)
  g <- unique(groups)
  fold_of <- setNames(sample(rep_len(seq_len(k), length(g))), g)
  unname(fold_of[as.character(groups)])
}

# ---- models ---------------------------------------------------------------------

fit_predict_fold <- function(train, test, target) {
  yv <- train[[target]]
  base <- rep(mean(yv), nrow(test))
  # A: logistic baseline (location + type + simple context)
  f_log <- if (target == "y_penalty") {
    as.formula(paste(target, "~ dist_own_goal + foul_type_f + offensive_i"))
  } else as.formula(paste(target, "~ dist_own_goal + foul_type_f + offensive_i + advantage_i + minute + is_women + is_international"))
  m_log <- glm(f_log, data = train, family = binomial())
  p_log <- as.numeric(predict(m_log, test, type = "response"))
  # B: GAM: spatial smooth + context + 360 features. The penalty target has only
  # a handful of negatives (own-box fouls that were indirect-free-kick offences),
  # so it gets a deliberately small model; anything richer separates or fails.
  f_gam <- if (target == "y_penalty") {
    as.formula(paste(target, "~ foul_type_f + s(dist_own_goal, k = 4) + offensive_i"))
  } else as.formula(paste(target,
    "~ s(x, y) + foul_type_f + offensive_i + advantage_i + under_pressure_i + s(minute, k = 6) +",
    "pos_group + is_women + is_international + s(n_teammates_goalside, k = 4) +",
    "s(dist_nearest_opp, k = 6) + n_opp_within_5 + ff_missing"))
  m_gam <- mgcv::gam(f_gam, data = train, family = binomial(), method = "REML")
  p_gam <- as.numeric(predict(m_gam, test, type = "response"))
  list(p_base = base, p_log = p_log, p_gam = p_gam)
}

run_cv <- function(df, target, fold) {
  out <- df %>% select(event_id, match_id, referee_key, competition, season, all_of(target))
  out$fold <- fold; out$p_base <- NA_real_; out$p_log <- NA_real_; out$p_gam <- NA_real_
  for (f in sort(unique(fold))) {
    tr <- df[fold != f, ]; te <- df[fold == f, ]
    # factor levels must match between train and test
    lv <- levels(tr$foul_type_f)
    te$foul_type_f <- factor(ifelse(te$foul_type_f %in% lv, as.character(te$foul_type_f), "Other"), levels = lv)
    lvp <- levels(tr$pos_group)
    te$pos_group <- factor(ifelse(te$pos_group %in% lvp, as.character(te$pos_group), "Other"), levels = lvp)
    p <- fit_predict_fold(tr, te, target)
    idx <- which(fold == f)
    out$p_base[idx] <- p$p_base; out$p_log[idx] <- p$p_log; out$p_gam[idx] <- p$p_gam
  }
  out
}

# Leakage-safe per-fold isotonic recalibration of column `col`.
isotonic_by_fold <- function(oof, col, y) {
  res <- rep(NA_real_, nrow(oof))
  for (f in sort(unique(oof$fold))) {
    cal <- oof[oof$fold != f, ]; tst <- which(oof$fold == f)
    ord <- order(cal[[col]])
    fit <- stats::isoreg(cal[[col]][ord], cal[[y]][ord])
    iso_fun <- stats::approxfun(cal[[col]][ord], fit$yf, method = "linear", rule = 2, ties = "ordered")
    res[tst] <- pmin(pmax(iso_fun(oof[[col]][tst]), 0), 1)
  }
  res
}

# ---- scoring with cluster bootstrap over matches ----------------------------------

score_one <- function(p, y) {
  d <- brier_decomposition(p, y, 10)
  c(log_loss = log_loss(p, y), brier = d$brier, ece = expected_calibration_error(p, y, 10),
    reliability = d$reliability, resolution = d$resolution)
}

cluster_boot <- function(oof, cols, y, n_boot = N_BOOT, seed = SEED) {
  m <- unique(oof$match_id); idx_by_m <- split(seq_len(nrow(oof)), oof$match_id)
  set.seed(seed)
  reps <- replicate(n_boot, {
    draw <- sample(m, length(m), replace = TRUE)
    idx <- unlist(idx_by_m[as.character(draw)], use.names = FALSE)
    yy <- oof[[y]][idx]
    sapply(cols, function(cn) {
      s <- score_one(oof[[cn]][idx], yy)
      c(s, ll_vs_base = unname(s["log_loss"] - log_loss(oof$p_base[idx], yy)))
    })
  }, simplify = "array")                    # metrics x cols x n_boot
  # point estimates on the full data
  point <- sapply(cols, function(cn) {
    s <- score_one(oof[[cn]], oof[[y]])
    c(s, ll_vs_base = unname(s["log_loss"] - log_loss(oof$p_base, oof[[y]])))
  })
  map_dfr(cols, function(cn) {
    tibble(model = cn, metric = rownames(point),
           estimate = point[, cn],
           lo = apply(reps[, cn, ], 1, quantile, 0.025),
           hi = apply(reps[, cn, ], 1, quantile, 0.975))
  })
}

# Abstention curve: for symmetric thresholds t, cover the fouls with p <= t or
# p >= 1 - t, and report accuracy + log loss on the covered set.
abstention_curve <- function(p, y, ts = seq(0.05, 0.5, by = 0.025)) {
  map_dfr(ts, function(t) {
    cov <- p <= t | p >= 1 - t
    tibble(threshold = t, coverage = mean(cov),
           accuracy = if (any(cov)) mean((p[cov] >= 0.5) == (y[cov] == 1)) else NA_real_,
           log_loss = if (any(cov)) log_loss(p[cov], y[cov]) else NA_real_,
           n = sum(cov))
  })
}

# ---- driver ----------------------------------------------------------------------

run_decision_models <- function(in_path = file.path("data", "decisions.rds")) {
  dec <- readRDS(in_path)
  fouls <- build_foul_table(dec)
  fouls$foul_type_f <- lump(fouls$foul_type_f, 30)
  fouls$pos_group   <- factor(fouls$pos_group)
  message("Fouls: ", nrow(fouls), " | carded: ", sum(fouls$y_card),
          sprintf(" (%.3f)", mean(fouls$y_card)), " | matches: ", n_distinct(fouls$match_id),
          " | referees: ", n_distinct(fouls$referee_key))

  targets <- list(
    card = list(df = fouls, y = "y_card"),
    penalty = list(df = fouls %>% filter(in_own_box), y = "y_penalty"))
  message("Own-box fouls: ", nrow(targets$penalty$df), " | penalties: ",
          sum(targets$penalty$df$y_penalty), sprintf(" (%.3f)", mean(targets$penalty$df$y_penalty)))

  scores <- list(); oofs <- list(); abst <- list(); rel <- list()
  for (tn in names(targets)) {
    df <- targets[[tn]]$df; yv <- targets[[tn]]$y
    for (grouping in c("match", "referee")) {
      grp <- if (grouping == "match") df$match_id else df$referee_key
      oof <- run_cv(df, yv, make_group_folds(grp))
      oof$p_gam_iso <- isotonic_by_fold(oof, "p_gam", yv)
      oof$p_log_iso <- isotonic_by_fold(oof, "p_log", yv)
      cols <- c("p_base", "p_log", "p_gam", "p_log_iso", "p_gam_iso")
      sc <- cluster_boot(oof, cols, yv) %>% mutate(target = tn, cv = grouping)
      scores[[paste(tn, grouping)]] <- sc
      oofs[[paste(tn, grouping)]] <- oof %>% mutate(target = tn, cv = grouping)
      message(sprintf("[%s | %s-grouped CV] log loss base %.4f | logistic %.4f | GAM %.4f | GAM+iso %.4f | ECE GAM %.4f -> iso %.4f",
                      tn, grouping,
                      sc$estimate[sc$model == "p_base" & sc$metric == "log_loss"],
                      sc$estimate[sc$model == "p_log" & sc$metric == "log_loss"],
                      sc$estimate[sc$model == "p_gam" & sc$metric == "log_loss"],
                      sc$estimate[sc$model == "p_gam_iso" & sc$metric == "log_loss"],
                      sc$estimate[sc$model == "p_gam" & sc$metric == "ece"],
                      sc$estimate[sc$model == "p_gam_iso" & sc$metric == "ece"]))
      if (grouping == "match") {
        abst[[tn]] <- abstention_curve(oof$p_gam_iso, oof[[yv]]) %>% mutate(target = tn)
        q <- unique(quantile(oof$p_gam_iso, probs = seq(0, 1, 0.1)))
        b <- cut(oof$p_gam_iso, q, include.lowest = TRUE, labels = FALSE)
        rel[[tn]] <- tibble(p = oof$p_gam_iso, y = oof[[yv]], bin = b) %>%
          group_by(bin) %>%
          summarise(mean_pred = mean(p), obs_rate = mean(y), n = n(),
                    se = sqrt(pmax(obs_rate * (1 - obs_rate), 1e-6) / n), .groups = "drop") %>%
          mutate(target = tn)
      }
    }
  }
  scores <- bind_rows(scores); oof_all <- bind_rows(oofs)
  abst <- bind_rows(abst); rel <- bind_rows(rel)

  dir.create("data", showWarnings = FALSE); dir.create("figures", showWarnings = FALSE)
  saveRDS(scores, file.path("data", "decision_scores.rds"))
  saveRDS(oof_all, file.path("data", "decision_oof.rds"))
  saveRDS(fouls, file.path("data", "fouls_model.rds"))
  saveRDS(abst, file.path("data", "decision_abstention.rds"))

  # reliability diagram (match-grouped, isotonic GAM)
  p1 <- ggplot(rel, aes(mean_pred, obs_rate)) +
    geom_abline(slope = 1, intercept = 0, linetype = 2, colour = "grey50") +
    geom_errorbar(aes(ymin = obs_rate - 1.96 * se, ymax = obs_rate + 1.96 * se), width = 0, alpha = 0.5) +
    geom_line(colour = "#2c7fb8") + geom_point(aes(size = n), colour = "#2c7fb8", alpha = 0.8) +
    facet_wrap(~ target, labeller = labeller(target = c(card = "P(card | foul)", penalty = "P(penalty | foul in own box)"))) +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(title = "Refereeing-decision models: reliability (out-of-fold, match-grouped CV)",
         subtitle = "GAM with leakage-safe isotonic recalibration; decile bins with binomial CIs",
         x = "Mean predicted probability", y = "Observed rate", size = "Fouls in bin") +
    theme_minimal(base_size = 12)
  ggsave(file.path("figures", "decision_reliability.png"), p1, width = 9, height = 5, dpi = 300)

  # abstention curve
  p2 <- ggplot(abst, aes(coverage, accuracy, colour = target)) +
    geom_line() + geom_point() +
    labs(title = "Abstention: accuracy on the fouls the model is willing to call",
         subtitle = "Fouls with p <= t or p >= 1 - t are 'called'; the rest are 'refer to review'",
         x = "Coverage (share of fouls called)", y = "Accuracy on called fouls", colour = "Target") +
    theme_minimal(base_size = 12)
  ggsave(file.path("figures", "decision_abstention.png"), p2, width = 7, height = 5, dpi = 300)

  write_decision_markdown(scores, abst, fouls, targets)
  invisible(list(scores = scores, oof = oof_all, abstention = abst))
}

write_decision_markdown <- function(scores, abst, fouls, targets, path = file.path("analysis", "decision_models.md")) {
  fmt <- function(e, lo, hi) sprintf("%.4f [%.4f, %.4f]", e, lo, hi)
  lines <- c("# Refereeing-decision models (VAR-Lab layer 1)", "",
             sprintf("Generated by `R/07_decision_models.R` on %s. All numbers are out-of-fold; intervals are 95%% cluster-bootstrap CIs over matches (%d resamples).",
                     format(Sys.Date()), N_BOOT), "",
             sprintf("Fouls: %d (carded %d, %.1f%%) in %d matches, %d referees. Own-box fouls: %d (penalties %d, %.1f%%).",
                     nrow(fouls), sum(fouls$y_card), 100 * mean(fouls$y_card), n_distinct(fouls$match_id),
                     n_distinct(fouls$referee_key), nrow(targets$penalty$df), sum(targets$penalty$df$y_penalty),
                     100 * mean(targets$penalty$df$y_penalty)), "")
  for (tn in unique(scores$target)) {
    for (cv in c("match", "referee")) {
      s <- scores %>% filter(target == tn, cv == !!cv)
      lines <- c(lines, sprintf("## %s, %s-grouped CV", tn, cv), "",
                 "| Model | Log loss | Log loss minus base rate | Brier | ECE (10 bins) | Reliability | Resolution |",
                 "|---|---|---|---|---|---|---|")
      for (m in c("p_base", "p_log", "p_gam", "p_log_iso", "p_gam_iso")) {
        g <- function(met) { r <- s[s$model == m & s$metric == met, ]; fmt(r$estimate, r$lo, r$hi) }
        lines <- c(lines, sprintf("| %s | %s | %s | %s | %s | %s | %s |", m,
                                  g("log_loss"), g("ll_vs_base"), g("brier"), g("ece"), g("reliability"), g("resolution")))
      }
      lines <- c(lines, "")
    }
    a <- abst %>% filter(target == tn)
    lines <- c(lines, sprintf("### %s: abstention curve (match-grouped, GAM + isotonic)", tn), "",
               "| Threshold t | Coverage | Accuracy on called | Log loss on called | n called |", "|---|---|---|---|---|",
               sprintf("| %.3f | %.3f | %.3f | %.4f | %d |", a$threshold, a$coverage, a$accuracy, a$log_loss, a$n), "")
  }
  lines <- c(lines, "Model names: p_base = training-fold base rate; p_log = logistic baseline; p_gam = GAM; *_iso = leakage-safe per-fold isotonic recalibration.", "",
             "Note on ECE intervals: ECE is a positively biased statistic in finite samples, so its percentile-bootstrap interval can sit entirely above the full-sample point estimate; read the interval as the plausible range under resampling, not as a symmetric error bar.", "")
  writeLines(lines, path)
  invisible(path)
}

if (sys.nframe() == 0) run_decision_models()
