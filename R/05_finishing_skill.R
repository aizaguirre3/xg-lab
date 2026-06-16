# R/05_finishing_skill.R ------------------------------------------------------
# Is finishing skill real and persistent, or mostly regression to the mean?
#   1. Per-player goals-above-xG (raw residual vs our OOF model xG).
#   2. Hierarchical shrinkage: glmer with a player random intercept and the
#      model's xG as a fixed offset on the logit scale -> shrunken player
#      effects with intervals; low-volume players pulled toward zero.
#   3. Split-half persistence test (odd/even shots) with a permutation null:
#      does period-1 over-performance predict period-2?
# Writes figures/shrinkage.png and data/finishing_*.rds.
# -----------------------------------------------------------------------------
suppressPackageStartupMessages({ library(dplyr); library(lme4); library(ggplot2) })
if (!exists("log_loss")) source(file.path("R", "utils.R"))

MIN_SHOTS <- as.integer(Sys.getenv("XG_MIN_SHOTS", "40"))  # threshold (sens-checked)
MODEL_XG  <- "p_xgb"                                        # our best OOF model

logit <- function(p, eps = 1e-6) { p <- pmin(pmax(p, eps), 1 - eps); log(p / (1 - p)) }

run_finishing <- function(in_path = file.path("data", "oof_preds.rds")) {
  oof <- readRDS(in_path)
  oof <- oof %>% filter(!is.na(player_id), !is.na(.data[[MODEL_XG]]))
  oof$model_xg <- oof[[MODEL_XG]]

  # ---- 1. raw per-player goals-above-xG ------------------------------------
  per_player <- oof %>%
    group_by(player_id, player_name) %>%
    summarise(shots = dplyr::n(), goals = sum(is_goal),
              xg = sum(model_xg), gax = goals - xg, .groups = "drop") %>%
    filter(shots >= MIN_SHOTS) %>%
    mutate(gax_per_shot = gax / shots)
  message("Players with >= ", MIN_SHOTS, " shots: ", nrow(per_player))

  # ---- 2. hierarchical shrinkage model -------------------------------------
  # logit(P(goal)) = offset(logit(model_xg)) + (1 | player). The random
  # intercept IS the player's finishing skill on the log-odds scale, shrunk
  # toward 0 by volume. Restrict to the same min-shot players for stability.
  keep <- oof %>% filter(player_id %in% per_player$player_id) %>%
    mutate(off = logit(model_xg), player_id = factor(player_id))
  glmm <- lme4::glmer(is_goal ~ 1 + (1 | player_id), data = keep,
                      family = binomial(), offset = off,
                      control = glmerControl(optimizer = "bobyqa"))

  re <- lme4::ranef(glmm, condVar = TRUE)$player_id
  pv <- attr(re, "postVar")[1, 1, ]
  shrunk <- tibble::tibble(
    player_id = as.numeric(rownames(re)),
    skill_logit = re[, 1],
    skill_se = sqrt(pv)) %>%
    left_join(per_player, by = "player_id") %>%
    mutate(lo = skill_logit - 1.96 * skill_se,
           hi = skill_logit + 1.96 * skill_se)
  saveRDS(shrunk, file.path("data", "finishing_shrunk.rds"))

  # the shrinkage figure: raw GAX/shot vs shrunken skill, size = shots
  pfig <- ggplot(shrunk, aes(gax_per_shot, skill_logit, size = shots)) +
    geom_hline(yintercept = 0, linetype = 2, colour = "grey50") +
    geom_vline(xintercept = 0, linetype = 2, colour = "grey50") +
    geom_point(alpha = 0.6, colour = "#2c7fb8") +
    labs(title = "Shrinkage pulls low-volume finishers toward zero",
         subtitle = "Raw goals-above-xG per shot vs. shrunken random-intercept skill",
         x = "Raw goals-above-xG per shot", y = "Shrunken finishing skill (log-odds)",
         size = "Shots") +
    theme_minimal(base_size = 12)
  ggsave(file.path("figures", "shrinkage.png"), pfig, width = 7, height = 5.5, dpi = 300)
  message("Wrote figures/shrinkage.png")

  # leaderboard: top/bottom 10 by shrunken skill
  lb <- shrunk %>% arrange(desc(skill_logit)) %>%
    transmute(player_name, shots, goals, xg = round(xg, 1),
              gax = round(gax, 1), skill_logit = round(skill_logit, 3),
              ci = sprintf("[%.2f, %.2f]", lo, hi))
  saveRDS(lb, file.path("data", "finishing_leaderboard.rds"))

  # ---- 3. split-half persistence test --------------------------------------
  # odd/even shots per player; correlate over-performance across halves.
  oof2 <- oof %>% filter(player_id %in% per_player$player_id) %>%
    group_by(player_id) %>% mutate(idx = row_number(),
                                   half = if_else(idx %% 2 == 1, "h1", "h2")) %>%
    ungroup()
  halves <- oof2 %>% group_by(player_id, half) %>%
    summarise(gax_ps = (sum(is_goal) - sum(model_xg)) / dplyr::n(),
              n = dplyr::n(), .groups = "drop") %>%
    tidyr::pivot_wider(names_from = half, values_from = c(gax_ps, n)) %>%
    filter(n_h1 >= 15, n_h2 >= 15)            # need enough shots in each half
  r_obs <- cor(halves$gax_ps_h1, halves$gax_ps_h2)

  set.seed(SEED <- 7406)
  perm <- replicate(2000, cor(halves$gax_ps_h1, sample(halves$gax_ps_h2)))
  p_perm <- mean(abs(perm) >= abs(r_obs))
  persistence <- tibble::tibble(
    n_players = nrow(halves), split_half_r = r_obs, perm_p = p_perm)
  message(sprintf("Persistence: r = %.3f over %d players, permutation p = %.3f",
                  r_obs, nrow(halves), p_perm))
  saveRDS(persistence, file.path("data", "finishing_persistence.rds"))

  invisible(list(shrunk = shrunk, persistence = persistence))
}

if (sys.nframe() == 0) run_finishing()