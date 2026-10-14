# R/08_referee_effects.R ------------------------------------------------------
# VAR-Lab layer 1: do referees differ in how often a foul becomes a card, once
# the foul's context is accounted for? Mirrors the finishing-skill analysis:
#   1. Raw cards-above-expected per referee (cards minus sum of OOF model P(card)).
#   2. Hierarchical shrinkage: glmer with a referee random intercept and the
#      match-grouped OOF GAM probability as a fixed offset on the logit scale.
#   3. Split-half persistence (odd/even matches per referee) with a permutation
#      null: does a referee's strictness in half 1 predict half 2?
#   4. Out-of-sample value of referee identity: match-grouped CV of the referee
#      adjustment on top of the context model (log loss with vs without).
# Writes figures/referee_shrinkage.png and data/referee_*.rds, and appends a
# section to analysis/decision_models.md.
# -----------------------------------------------------------------------------
suppressPackageStartupMessages({ library(dplyr); library(lme4); library(ggplot2); library(tibble); library(tidyr) })
if (!exists("log_loss")) source(file.path("R", "utils.R"))
# Referee names are UTF-8; a C locale would write them as <U+00E9> escapes.
invisible(suppressWarnings(try(Sys.setlocale("LC_CTYPE", "C.UTF-8"), silent = TRUE)))

MIN_FOULS <- as.integer(Sys.getenv("XG_REF_MIN_FOULS", "60"))
SEED <- 7406
logit <- function(p, eps = 1e-6) { p <- pmin(pmax(p, eps), 1 - eps); log(p / (1 - p)) }
inv_logit <- function(z) 1 / (1 + exp(-z))

run_referee_effects <- function(oof_path = file.path("data", "decision_oof.rds"),
                                idx_path = file.path("data", "match_index_360.rds")) {
  oof <- readRDS(oof_path) %>% filter(target == "card", cv == "match")
  idx <- readRDS(idx_path) %>% select(match_id, match_date, referee)
  oof <- oof %>% left_join(idx, by = "match_id") %>%
    filter(!grepl("^m", referee_key)) %>%            # drop matches with no referee recorded
    mutate(off = logit(p_gam_iso))
  message("Fouls with a known referee: ", nrow(oof), " | referees: ", n_distinct(oof$referee_key))

  # ---- 1. raw cards-above-expected per referee --------------------------------
  per_ref <- oof %>% group_by(referee_key, referee) %>%
    summarise(fouls = n(), matches = n_distinct(match_id), cards = sum(y_card),
              expected = sum(p_gam_iso), cae = cards - expected, .groups = "drop") %>%
    filter(fouls >= MIN_FOULS) %>% mutate(cae_per_foul = cae / fouls)
  message("Referees with >= ", MIN_FOULS, " fouls: ", nrow(per_ref))

  # ---- 2. shrinkage model -----------------------------------------------------
  keep <- oof %>% filter(referee_key %in% per_ref$referee_key) %>% mutate(referee_key = factor(referee_key))
  glmm <- lme4::glmer(y_card ~ 1 + (1 | referee_key), data = keep, family = binomial(),
                      offset = off, control = glmerControl(optimizer = "bobyqa"))
  sd_ref <- sqrt(as.numeric(lme4::VarCorr(glmm)$referee_key[1, 1]))
  re <- lme4::ranef(glmm, condVar = TRUE)$referee_key
  pv <- attr(re, "postVar")[1, 1, ]
  shrunk <- tibble(referee_key = rownames(re), strictness_logit = re[, 1], se = sqrt(pv)) %>%
    left_join(per_ref, by = "referee_key") %>%
    mutate(lo = strictness_logit - 1.96 * se, hi = strictness_logit + 1.96 * se) %>%
    arrange(desc(strictness_logit))
  message(sprintf("Referee random-intercept SD (logit): %.3f | fixed intercept: %.3f", sd_ref, fixef(glmm)[1]))

  pfig <- ggplot(shrunk, aes(cae_per_foul, strictness_logit, size = fouls)) +
    geom_hline(yintercept = 0, linetype = 2, colour = "grey50") +
    geom_vline(xintercept = 0, linetype = 2, colour = "grey50") +
    geom_point(alpha = 0.6, colour = "#d95f0e") +
    labs(title = "Referee strictness: shrinkage pulls low-volume referees toward zero",
         subtitle = "Raw cards-above-expected per foul vs. shrunken random-intercept effect (context model as offset)",
         x = "Raw cards above expected, per foul", y = "Shrunken strictness (log-odds)", size = "Fouls") +
    theme_minimal(base_size = 12)
  dir.create("figures", showWarnings = FALSE)
  ggsave(file.path("figures", "referee_shrinkage.png"), pfig, width = 7, height = 5.5, dpi = 300)

  # ---- 3. split-half persistence over matches ---------------------------------
  halves <- keep %>% arrange(referee_key, match_date, match_id) %>%
    group_by(referee_key) %>% mutate(m_rank = dense_rank(match_id),
                                     half = if_else(m_rank %% 2 == 1, "h1", "h2")) %>%
    group_by(referee_key, half) %>%
    summarise(cae_pf = (sum(y_card) - sum(p_gam_iso)) / n(), n = n(), .groups = "drop") %>%
    pivot_wider(names_from = half, values_from = c(cae_pf, n)) %>%
    filter(!is.na(n_h1), !is.na(n_h2), n_h1 >= 25, n_h2 >= 25)
  r_obs <- cor(halves$cae_pf_h1, halves$cae_pf_h2)
  set.seed(SEED)
  perm <- replicate(2000, cor(halves$cae_pf_h1, sample(halves$cae_pf_h2)))
  p_perm <- mean(abs(perm) >= abs(r_obs))
  message(sprintf("Persistence: r = %.3f over %d referees, permutation p = %.3f", r_obs, nrow(halves), p_perm))

  # ---- 4. out-of-sample value of referee identity ------------------------------
  # Match-grouped folds already in oof$fold. For each fold, fit the referee
  # random effect on the other folds and predict held-out fouls (unseen referees
  # get the population effect 0), then compare log loss with vs without.
  p_adj <- rep(NA_real_, nrow(keep))
  for (f in sort(unique(keep$fold))) {
    tr <- keep[keep$fold != f, ]; te_idx <- which(keep$fold == f)
    m <- lme4::glmer(y_card ~ 1 + (1 | referee_key), data = tr, family = binomial(),
                     offset = off, control = glmerControl(optimizer = "bobyqa"))
    re_f <- lme4::ranef(m)$referee_key
    eff <- re_f[as.character(keep$referee_key[te_idx]), 1]
    eff[is.na(eff)] <- 0
    p_adj[te_idx] <- inv_logit(keep$off[te_idx] + fixef(m)[1] + eff)
  }
  keep$p_ref_adj <- p_adj
  gain <- tibble(
    log_loss_context = log_loss(keep$p_gam_iso, keep$y_card),
    log_loss_context_plus_referee = log_loss(keep$p_ref_adj, keep$y_card),
    ece_context = expected_calibration_error(keep$p_gam_iso, keep$y_card),
    ece_context_plus_referee = expected_calibration_error(keep$p_ref_adj, keep$y_card))
  # cluster bootstrap over matches for the log-loss difference
  set.seed(SEED)
  ms <- unique(keep$match_id); idx_by_m <- split(seq_len(nrow(keep)), keep$match_id)
  diffs <- replicate(500, {
    draw <- sample(ms, length(ms), replace = TRUE)
    i <- unlist(idx_by_m[as.character(draw)], use.names = FALSE)
    log_loss(keep$p_ref_adj[i], keep$y_card[i]) - log_loss(keep$p_gam_iso[i], keep$y_card[i])
  })
  gain$ll_diff_lo <- quantile(diffs, 0.025); gain$ll_diff_hi <- quantile(diffs, 0.975)
  message(sprintf("Referee identity OOS: log loss %.4f -> %.4f (diff 95%% CI [%.4f, %.4f])",
                  gain$log_loss_context, gain$log_loss_context_plus_referee, gain$ll_diff_lo, gain$ll_diff_hi))

  persistence <- tibble(n_referees = nrow(halves), split_half_r = r_obs, perm_p = p_perm, sd_logit = sd_ref)
  saveRDS(shrunk, file.path("data", "referee_shrunk.rds"))
  saveRDS(persistence, file.path("data", "referee_persistence.rds"))
  saveRDS(gain, file.path("data", "referee_oos_gain.rds"))

  # ---- markdown section --------------------------------------------------------
  lb <- shrunk %>% transmute(referee, matches, fouls, cards, expected = round(expected, 1),
                              strictness = round(strictness_logit, 3), ci = sprintf("[%.2f, %.2f]", lo, hi))
  top <- head(lb, 8); bot <- tail(lb, 8)
  row_md <- function(d) sprintf("| %s | %d | %d | %d | %.1f | %.3f | %s |", d$referee, d$matches, d$fouls, d$cards, d$expected, d$strictness, d$ci)
  lines <- c("", "## Referee effects (cards per foul, context-adjusted)", "",
             sprintf("Referees with >= %d fouls: %d. Random-intercept SD on the logit scale: %.3f.", MIN_FOULS, nrow(shrunk), sd_ref),
             sprintf("Split-half persistence over matches: r = %.3f across %d referees, permutation p = %.3f.", r_obs, nrow(halves), p_perm),
             sprintf("Out-of-sample value of referee identity (match-grouped CV): log loss %.4f without vs %.4f with the referee effect; difference 95%% CI [%.4f, %.4f] (negative = referee identity helps).",
                     gain$log_loss_context, gain$log_loss_context_plus_referee, gain$ll_diff_lo, gain$ll_diff_hi), "",
             "| Referee | Matches | Fouls | Cards | Expected cards | Shrunken strictness (logit) | 95% CI |", "|---|---|---|---|---|---|---|",
             "| **Strictest 8** | | | | | | |", row_md(top), "| **Most lenient 8** | | | | | | |", row_md(bot), "",
             "![Referee shrinkage](../figures/referee_shrinkage.png)", "")
  con <- file(file.path("analysis", "decision_models.md"), open = "a", encoding = "UTF-8")
  writeLines(enc2utf8(lines), con); close(con)
  invisible(list(shrunk = shrunk, persistence = persistence, gain = gain))
}

if (sys.nframe() == 0) run_referee_effects()
