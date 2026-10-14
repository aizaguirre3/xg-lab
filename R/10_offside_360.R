# R/10_offside_360.R ----------------------------------------------------------
# VAR-Lab layer 2 on StatsBomb 360 frames: does the Law 11 geometry engine agree
# with the referee's offside calls, and how noisy are 360 positions?
#
# Data: data/pass_frames.rds = every "Pass Offside" (positive) plus 15 sampled
# completed forward passes per match (negatives), each with its 360 frame.
# Frames carry no player ids, so the receiver is inferred two ways:
#   nearest_end: the teammate nearest to the pass end location
#   max:         the furthest-forward teammate (any attacker in an offside position)
# For every frame: torso margin (yards, + = offside), Monte-Carlo P(offside) at
# sigma = 1 yard, a three-way call (offside / onside / refer), and visibility
# strata (opposing keeper in frame, receiver inside the camera's visible area).
# A probit psychometric fit P(call | margin) gives the effective positional noise
# of the whole chain (360 precision + receiver inference + kick-point timing).
#
# Negatives are a per-match SAMPLE, so the intercept of the psychometric fit (the
# point of subjective equality) is sampling-dependent; the slope / width is not.
# Writes data/offside_360_eval.rds, figures/offside_psychometric.png,
#        analysis/offside_360.md
# -----------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(dplyr); library(purrr); library(tibble); library(tidyr); library(ggplot2)
})
if (!exists("log_loss")) source(file.path("R", "utils.R"))
if (!exists("offside_frame")) source(file.path("R", "09_offside_engine.R"))

SIGMA_YD <- as.numeric(Sys.getenv("XG_OFF_SIGMA", "1"))
N_MC     <- as.integer(Sys.getenv("XG_OFF_NMC", "300"))
SEED     <- 7406

wilson <- function(k, n, z = 1.96) {
  p <- k / n; d <- 1 + z^2 / n
  c <- (p + z^2 / (2 * n)) / d; h <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / d
  c(lo = c - h, hi = c + h)
}

evaluate_frames <- function(pf) {
  map_dfr(seq_len(nrow(pf)), function(i) {
    fr <- pf$frame[[i]]
    ball <- c(pf$x[i], pf$y[i]); end <- c(pf$pass_end_x[i], pf$pass_end_y[i])
    a <- offside_frame(fr, end_xy = end, ball_xy = ball, receiver = "nearest_end")
    b <- offside_frame(fr, ball_xy = ball, receiver = "max")
    mc <- offside_mc_fast(fr, end_xy = end, ball_xy = ball, receiver = "nearest_end",
                          sigma = SIGMA_YD, n = N_MC, seed = SEED + i)
    inside <- if (a$ok) point_in_polygon(a$receiver_x, a$receiver_y, pf$visible_area[[i]]) else NA
    tibble(event_id = pf$event_id[i], ok = a$ok,
           margin_near = a$margin, margin_max = b$margin,
           receiver_x = a$receiver_x, second_last_def_x = a$second_last_def_x,
           reference_x = a$reference_x, n_def = a$n_def, n_att = a$n_att,
           receiver_in_visible = inside,
           p_offside = mc$p_offside, margin_q05 = mc$margin_q05, margin_q95 = mc$margin_q95,
           call3 = offside_call(mc))
  })
}

agreement_row <- function(d, label, col = "margin_near") {
  m <- d[[col]]; y <- d$offside_call
  ok <- !is.na(m)
  agree <- (m[ok] > 0) == (y[ok] == 1)
  far <- ok & abs(m) > 2
  agree_far <- (m[far] > 0) == (y[far] == 1)
  pos <- ok & y == 1; neg <- ok & y == 0
  tibble(stratum = label, n = sum(ok),
         n_pos = sum(pos), sensitivity = mean(m[pos] > 0),
         n_neg = sum(neg), false_positive_rate = mean(m[neg] > 0),
         agreement = mean(agree), agree_lo = wilson(sum(agree), length(agree))[1], agree_hi = wilson(sum(agree), length(agree))[2],
         n_beyond_2yd = sum(far), agreement_beyond_2yd = mean(agree_far),
         beyond_lo = if (sum(far)) wilson(sum(agree_far), length(agree_far))[1] else NA_real_,
         beyond_hi = if (sum(far)) wilson(sum(agree_far), length(agree_far))[2] else NA_real_)
}

run_offside_360 <- function(in_path = file.path("data", "pass_frames.rds")) {
  pf <- readRDS(in_path) %>% filter(has_frame == 1)
  message("Pass frames with 360: ", nrow(pf), " | offside calls: ", sum(pf$offside_call))
  ev <- evaluate_frames(pf)
  d <- bind_cols(pf %>% select(event_id, match_id, competition, season, offside_call, pass_end_x,
                               keeper_opp_in_frame, n_frame, visible_area_frac, pass_height, pass_length, play_pattern),
                 ev %>% select(-event_id))
  message("Evaluable frames (actor + >= 2 defenders + a candidate receiver): ", sum(d$ok), " / ", nrow(d))

  # ---- agreement tables -----------------------------------------------------------
  dd <- d %>% filter(ok)
  strata <- bind_rows(
    agreement_row(dd, "all evaluable frames"),
    agreement_row(dd %>% filter(keeper_opp_in_frame), "opposing keeper in frame"),
    agreement_row(dd %>% filter(!keeper_opp_in_frame), "opposing keeper NOT in frame"),
    agreement_row(dd %>% filter(receiver_in_visible %in% TRUE), "inferred receiver inside visible area"),
    agreement_row(dd %>% filter(receiver_in_visible %in% FALSE), "inferred receiver outside visible area"),
    agreement_row(dd %>% filter(n_def >= 8), "8+ defenders visible"),
    agreement_row(dd %>% filter(n_def < 8), "< 8 defenders visible"),
    agreement_row(dd %>% filter(pass_height %in% "Ground Pass"), "ground passes"),
    agreement_row(dd %>% filter(!pass_height %in% "Ground Pass"), "lofted / high passes"))
  strata_max <- bind_rows(
    agreement_row(dd, "all evaluable frames (furthest-forward attacker)", "margin_max"),
    agreement_row(dd %>% filter(keeper_opp_in_frame), "opposing keeper in frame (furthest-forward)", "margin_max"))

  # ---- psychometric fit: P(call | margin), probit ---------------------------------
  # Fitted twice: on every evaluable frame (dominated by frames where the
  # defensive line is out of shot) and on "well-observed" frames (opposing
  # keeper in frame and >= 8 defenders visible), where the width is closer to
  # a genuine positional-noise estimate.
  fit_psych <- function(d) {
    fit <- glm(offside_call ~ margin_near, data = d, family = binomial(link = "probit"))
    slope <- coef(fit)[["margin_near"]]
    set.seed(SEED)
    ms <- unique(d$match_id); idx_by_m <- split(seq_len(nrow(d)), d$match_id)
    boots <- replicate(300, {
      draw <- sample(ms, length(ms), replace = TRUE)
      i <- unlist(idx_by_m[as.character(draw)], use.names = FALSE)
      f <- suppressWarnings(glm(offside_call ~ margin_near, data = d[i, ], family = binomial(link = "probit")))
      c(sigma = 1 / coef(f)[["margin_near"]], pse = -coef(f)[[1]] / coef(f)[["margin_near"]])
    })
    pos <- d$margin_near[d$offside_call == 1]; neg <- d$margin_near[d$offside_call == 0]
    list(fit = fit, row = tibble(
      n = nrow(d), sigma_eff_yd = 1 / slope,
      sigma_lo = quantile(boots["sigma", ], 0.025), sigma_hi = quantile(boots["sigma", ], 0.975),
      pse_yd = -coef(fit)[[1]] / slope, pse_lo = quantile(boots["pse", ], 0.025), pse_hi = quantile(boots["pse", ], 0.975),
      auc = mean(outer(pos, neg, ">") + 0.5 * outer(pos, neg, "=="))))
  }
  well <- dd %>% filter(keeper_opp_in_frame, n_def >= 8)
  ps_all <- fit_psych(dd); ps_well <- fit_psych(well)
  psych <- bind_rows(ps_all$row %>% mutate(subset = "all evaluable frames"),
                     ps_well$row %>% mutate(subset = "well-observed: keeper in frame and 8+ defenders"))
  message(sprintf("Psychometric width sigma: all frames %.2f yd [%.2f, %.2f]; well-observed %.2f yd [%.2f, %.2f] (n = %d); AUC %.3f / %.3f",
                  ps_all$row$sigma_eff_yd, ps_all$row$sigma_lo, ps_all$row$sigma_hi,
                  ps_well$row$sigma_eff_yd, ps_well$row$sigma_lo, ps_well$row$sigma_hi, nrow(well),
                  ps_all$row$auc, ps_well$row$auc))

  # empirical curves by 1-yard bins, faceted by subset
  mk_curve <- function(d, lab) d %>% mutate(bin = pmin(pmax(round(margin_near), -12), 12)) %>%
    group_by(bin) %>% summarise(n = n(), rate = mean(offside_call),
                                lo = wilson(sum(offside_call), n())[1], hi = wilson(sum(offside_call), n())[2], .groups = "drop") %>%
    mutate(subset = lab)
  curve <- bind_rows(mk_curve(dd, "all evaluable frames"), mk_curve(well, "well-observed: keeper in frame and 8+ defenders"))
  grid <- bind_rows(
    tibble(margin_near = seq(-12, 12, 0.25), subset = "all evaluable frames") %>% mutate(fit = predict(ps_all$fit, ., type = "response")),
    tibble(margin_near = seq(-12, 12, 0.25), subset = "well-observed: keeper in frame and 8+ defenders") %>% mutate(fit = predict(ps_well$fit, ., type = "response")))
  p <- ggplot(curve, aes(bin, rate)) +
    geom_line(data = grid, aes(margin_near, fit), colour = "#2c7fb8") +
    geom_errorbar(aes(ymin = lo, ymax = hi), width = 0, alpha = 0.5) +
    geom_point(aes(size = n), alpha = 0.8) +
    geom_vline(xintercept = 0, linetype = 2, colour = "grey50") +
    facet_wrap(~ subset) +
    labs(title = "Referee offside calls vs. the engine's torso margin on 360 frames",
         subtitle = sprintf("Probit width: all frames %.1f yd; well-observed frames %.1f yd (95%% CI %.1f to %.1f). Negatives are a per-match sample.",
                            ps_all$row$sigma_eff_yd, ps_well$row$sigma_eff_yd, ps_well$row$sigma_lo, ps_well$row$sigma_hi),
         x = "Margin of inferred receiver beyond the offside line (yards; + = offside position)",
         y = "Share of passes flagged offside", size = "Passes") +
    theme_minimal(base_size = 12)
  dir.create("figures", showWarnings = FALSE)
  ggsave(file.path("figures", "offside_psychometric.png"), p, width = 11, height = 5.5, dpi = 300)

  # ---- three-way Monte-Carlo call vs referee ------------------------------------------
  conf <- dd %>% count(call3, offside_call) %>%
    pivot_wider(names_from = offside_call, values_from = n, values_fill = 0, names_prefix = "ref_") %>%
    mutate(total = ref_0 + ref_1)
  coverage <- mean(dd$call3 != "refer")
  called <- dd %>% filter(call3 != "refer")
  acc_called <- mean((called$call3 == "offside") == (called$offside_call == 1))

  saveRDS(list(frames = d, strata = strata, strata_max = strata_max, psych = psych, curve = curve, confusion = conf),
          file.path("data", "offside_360_eval.rds"))
  write_offside_md(d, strata, strata_max, psych, conf, coverage, acc_called)
  invisible(list(frames = d, strata = strata, psych = psych))
}

write_offside_md <- function(d, strata, strata_max, psych, conf, coverage, acc_called,
                             path = file.path("analysis", "offside_360.md")) {
  f3 <- function(x) ifelse(is.na(x), "n/a", sprintf("%.3f", x))
  row <- function(s) sprintf("| %s | %d | %d | %s | %d | %s | %s [%s, %s] | %d | %s [%s, %s] |",
                             s$stratum, s$n, s$n_pos, f3(s$sensitivity), s$n_neg, f3(s$false_positive_rate),
                             f3(s$agreement), f3(s$agree_lo), f3(s$agree_hi), s$n_beyond_2yd,
                             f3(s$agreement_beyond_2yd), f3(s$beyond_lo), f3(s$beyond_hi))
  hdr <- c("| Stratum | n | Offside calls | Sensitivity (margin > 0) | Non-calls | False-positive rate | Agreement [95% CI] | n beyond 2 yd | Agreement beyond 2 yd [95% CI] |",
           "|---|---|---|---|---|---|---|---|---|")
  lines <- c("# Law 11 engine vs. referee offside calls on StatsBomb 360 frames (VAR-Lab layer 2b)", "",
             sprintf("Generated by `R/10_offside_360.R` on %s. Frames: %d with 360 data, %d evaluable (actor present, >= 2 defenders, a candidate receiver). Monte Carlo: sigma = %.1f yd on every position, %d draws.",
                     format(Sys.Date()), nrow(d), sum(d$ok), SIGMA_YD, N_MC), "",
             "Receiver rule: nearest teammate to the pass end location (frames carry no player ids). Negatives: 15 completed forward passes into the opponent's half sampled per match; positives: every pass with outcome Pass Offside.", "",
             "## Agreement by visibility stratum (inferred receiver)", "", hdr, sapply(seq_len(nrow(strata)), function(i) row(strata[i, ])), "",
             "## Furthest-forward attacker rule (any attacker in an offside position)", "", hdr, sapply(seq_len(nrow(strata_max)), function(i) row(strata_max[i, ])), "",
             "## Psychometric fit", "",
             "| Subset | n | Width sigma (yd) [95% CI over matches] | Point of subjective equality (yd) | AUC of margin |", "|---|---|---|---|---|",
             sprintf("| %s | %d | %.2f [%.2f, %.2f] | %.2f [%.2f, %.2f] | %.3f |", psych$subset, psych$n, psych$sigma_eff_yd, psych$sigma_lo, psych$sigma_hi,
                     psych$pse_yd, psych$pse_lo, psych$pse_hi, psych$auc), "",
             "The all-frames width bundles 360 positional precision, receiver mis-identification, defenders outside the camera view, torso-vs-limb offsets, and kick-point timing; it is a failure measure of the whole chain, not a precision estimate. The well-observed width is the closer proxy for 360 positional noise plus receiver inference. The point of subjective equality is sampling-dependent (negatives are a per-match sample).", "",
             "![Psychometric curve](../figures/offside_psychometric.png)", "",
             "## Three-way Monte-Carlo call vs. referee", "",
             "| Engine call | Referee: no call | Referee: offside | Total |", "|---|---|---|---|",
             sprintf("| %s | %d | %d | %d |", conf$call3, conf$ref_0, conf$ref_1, conf$total), "",
             sprintf("Coverage (frames the engine is willing to call at sigma = %.1f yd): %.3f. Accuracy on called frames: %.3f.", SIGMA_YD, coverage, acc_called), "",
             "## Caveats", "",
             "- 360 frames are 2D torso-level snapshots of the players the camera could see; the second-last defender may be out of frame (see keeper strata).",
             "- Whether the Pass Offside frame is sampled at the true kick point is undocumented by StatsBomb.",
             "- A non-call is not proof of an onside position: a player in an offside position who does not interfere is not flagged (Law 11.2), which inflates the furthest-forward rule's false-positive rate by design.", "")
  writeLines(lines, path)
  invisible(path)
}

if (sys.nframe() == 0) run_offside_360()
