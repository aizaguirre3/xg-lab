# R/02_eda.R ------------------------------------------------------------------
# Exploratory figures: the shot-map hexbin (the project's visual hook), empirical
# goal rate vs distance and vs angle with binomial CIs, and class balance by
# competition. Writes figures/*.png and data/eda_*.rds.
# -----------------------------------------------------------------------------
suppressPackageStartupMessages({ library(dplyr); library(ggplot2) })

# Minimal attacking-half pitch overlay in StatsBomb coords (120x80).
pitch_layer <- function() {
  list(
    annotate("rect", xmin = 60, xmax = 120, ymin = 0, ymax = 80,
             fill = NA, colour = "grey70"),
    annotate("rect", xmin = 102, xmax = 120, ymin = 18, ymax = 62,
             fill = NA, colour = "grey70"),                       # penalty box
    annotate("rect", xmin = 114, xmax = 120, ymin = 30, ymax = 50,
             fill = NA, colour = "grey70"),                       # six-yard box
    annotate("segment", x = 120, xend = 120, y = 36, yend = 44,
             colour = "black", linewidth = 1.2)                   # goal mouth
  )
}

run_eda <- function(in_path = file.path("data", "shots_model.rds")) {
  d <- readRDS(in_path)

  # (1) shot-map hexbin: goal rate over pitch coordinates
  pmap <- ggplot(d %>% filter(x >= 60), aes(x, y, z = is_goal)) +
    stat_summary_hex(fun = mean, bins = 30) +
    scale_fill_viridis_c(name = "Goal rate", option = "C", limits = c(0, NA)) +
    pitch_layer() +
    coord_equal(xlim = c(60, 121), ylim = c(0, 80)) +
    labs(title = "Goal rate by shot location",
         subtitle = "Hexbin mean over attacking half (StatsBomb coordinates)",
         x = NULL, y = NULL) +
    theme_minimal(base_size = 12) +
    theme(axis.text = element_blank(), panel.grid = element_blank())
  ggsave(file.path("figures", "shot_map.png"), pmap, width = 8, height = 6, dpi = 300)

  # (2) goal rate vs distance, binomial CIs
  by_dist <- d %>% mutate(db = cut(distance, breaks = seq(0, 40, 2))) %>%
    filter(!is.na(db)) %>% group_by(db) %>%
    summarise(mid = mean(distance), rate = mean(is_goal), n = dplyr::n(),
              se = sqrt(pmax(rate * (1 - rate), 1e-6) / n), .groups = "drop")
  pd <- ggplot(by_dist, aes(mid, rate)) +
    geom_ribbon(aes(ymin = pmax(rate - 1.96 * se, 0), ymax = rate + 1.96 * se),
                alpha = 0.2) +
    geom_line() + geom_point(aes(size = n)) +
    labs(title = "Empirical goal rate vs distance", x = "Distance to goal (yd)",
         y = "Goal rate", size = "Shots") + theme_minimal(base_size = 12)
  ggsave(file.path("figures", "goalrate_distance.png"), pd, width = 7, height = 5, dpi = 300)

  # (3) goal rate vs angle
  by_ang <- d %>% mutate(ab = cut(angle, breaks = seq(0, pi, length.out = 16))) %>%
    filter(!is.na(ab)) %>% group_by(ab) %>%
    summarise(mid = mean(angle), rate = mean(is_goal), n = dplyr::n(), .groups = "drop")
  pa <- ggplot(by_ang, aes(mid, rate)) + geom_line() + geom_point(aes(size = n)) +
    labs(title = "Empirical goal rate vs shot angle", x = "Visible goal angle (rad)",
         y = "Goal rate", size = "Shots") + theme_minimal(base_size = 12)
  ggsave(file.path("figures", "goalrate_angle.png"), pa, width = 7, height = 5, dpi = 300)

  # (4) class balance by competition
  bal <- d %>% group_by(competition) %>%
    summarise(shots = dplyr::n(), goals = sum(is_goal),
              base_rate = mean(is_goal), .groups = "drop") %>% arrange(desc(shots))
  saveRDS(bal, file.path("data", "eda_balance.rds"))
  message("Class balance by competition:"); print(as.data.frame(bal), digits = 3)
  message("Wrote figures/shot_map.png, goalrate_distance.png, goalrate_angle.png")
  invisible(bal)
}

if (sys.nframe() == 0) run_eda()