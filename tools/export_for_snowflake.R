#!/usr/bin/env Rscript
# tools/export_for_snowflake.R ------------------------------------------------
# Flatten the pipeline outputs into two dashboard-ready CSVs for Snowflake:
#   snowflake/shots.csv            shot-level, with our xG (p_xgb) + StatsBomb xG
#   snowflake/finishing_skill.csv  player leaderboard (shrunken finishing skill)
# Run after run_all.R. NULLs are written as empty strings (Snowflake NULL_IF).
# -----------------------------------------------------------------------------
suppressPackageStartupMessages({ library(dplyr) })

dir.create("snowflake", showWarnings = FALSE)
shots <- readRDS("data/shots_model.rds")
oof   <- readRDS("data/oof_preds.rds")

# join our out-of-fold model xG onto each shot
shots_out <- shots %>%
  left_join(oof %>% select(shot_id, our_xg = p_xgb), by = "shot_id") %>%
  transmute(
    shot_id, match_id, competition, season, minute,
    player_id, player_name, team,
    x = round(x, 2), y = round(y, 2),
    body_cat, shot_type, is_open_play, set_piece,
    under_pressure = as.integer(under_pressure),
    first_time     = as.integer(first_time),
    distance = round(distance, 3), angle = round(angle, 4),
    n_def_cone, dist_nearest_def = round(dist_nearest_def, 2),
    our_xg          = round(our_xg, 5),
    statsbomb_xg    = round(statsbomb_xg, 5),
    is_goal
  )

fin <- readRDS("data/finishing_leaderboard.rds")  # player, shots, goals, xg, gax, ...

w <- function(df, f) write.csv(df, f, row.names = FALSE, na = "")
w(shots_out, "snowflake/shots.csv")
w(fin,       "snowflake/finishing_skill.csv")
cat(sprintf("wrote snowflake/shots.csv (%d rows) and snowflake/finishing_skill.csv (%d rows)\n",
            nrow(shots_out), nrow(fin)))