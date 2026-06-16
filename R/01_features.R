# R/01_features.R -------------------------------------------------------------
# Turn raw shot events (data/shots_raw.rds) into a modeling table:
# geometry, context, and freeze-frame features. Penalties excluded; direct
# free kicks flagged. Caches data/shots_model.rds.
# -----------------------------------------------------------------------------
suppressPackageStartupMessages({ library(dplyr); library(purrr); library(tibble) })
if (!exists("distance_to_goal")) source(file.path("R", "utils.R"))

# Pull [x,y] out of a StatsBomb location list cell; NA if absent.
loc_xy <- function(cell) {
  if (is.null(cell) || length(cell) < 2 || any(is.na(unlist(cell)[1:2])))
    return(c(NA_real_, NA_real_))
  as.numeric(unlist(cell))[1:2]
}

# Freeze-frame features for one shot: defenders in the shooting cone, distance
# to nearest defender, and goalkeeper distance from goal. ff is the (flattened)
# freeze-frame data.frame for this shot; (sx,sy) is the shooter location.
freeze_features <- function(ff, sx, sy) {
  na_out <- list(n_def_cone = NA_real_, dist_nearest_def = NA_real_,
                 gk_dist_goal = NA_real_, has_freeze = 0L)
  if (is.null(ff) || !is.data.frame(ff) || nrow(ff) == 0) return(na_out)
  if (!"location" %in% names(ff) || !"teammate" %in% names(ff)) return(na_out)

  xy <- t(vapply(ff$location, loc_xy, numeric(2)))
  opp <- !isTRUE_vec(ff$teammate)             # opponents = not teammate
  ok  <- stats::complete.cases(xy) & opp
  if (!any(ok)) return(list(n_def_cone = 0, dist_nearest_def = NA_real_,
                            gk_dist_goal = NA_real_, has_freeze = 1L))
  ox <- xy[ok, 1]; oy <- xy[ok, 2]

  # defenders inside the shooter -> left post -> right post triangle
  in_cone <- mapply(point_in_triangle, ox, oy,
                    MoreArgs = list(ax = sx, ay = sy,
                                    bx = POST_LEFT[1],  by = POST_LEFT[2],
                                    cx = POST_RIGHT[1], cy = POST_RIGHT[2]))
  n_cone <- sum(in_cone)

  # distance from shooter to nearest opponent
  d_near <- min(sqrt((ox - sx)^2 + (oy - sy)^2))

  # goalkeeper distance from the goal line (centre); NA if GK not in frame
  gk_dist <- NA_real_
  if ("position.name" %in% names(ff)) {
    gk_idx <- which(ff$position.name == "Goalkeeper" & opp)
    if (length(gk_idx) >= 1) {
      g <- loc_xy(ff$location[[gk_idx[1]]])
      if (!any(is.na(g))) gk_dist <- distance_to_goal(g[1], g[2])
    }
  }
  list(n_def_cone = n_cone, dist_nearest_def = d_near,
       gk_dist_goal = gk_dist, has_freeze = 1L)
}

# robust logical coercion (StatsBomb omits FALSE -> NA in flattened frames)
isTRUE_vec <- function(x) !is.na(x) & x

build_features <- function(in_path  = file.path("data", "shots_raw.rds"),
                           out_path = file.path("data", "shots_model.rds")) {
  raw <- readRDS(in_path)
  message("Raw shots: ", nrow(raw))

  # shooter location
  xy <- t(vapply(raw$location, loc_xy, numeric(2)))
  raw$x <- xy[, 1]; raw$y <- xy[, 2]

  gv <- function(nm) if (nm %in% names(raw)) raw[[nm]] else rep(NA, nrow(raw))

  df <- tibble(
    shot_id     = gv("id"),
    match_id    = raw$match_id,
    competition = raw$competition,
    season      = raw$season,
    period      = gv("period"),
    minute      = gv("minute"),
    player_id   = gv("player.id"),
    player_name = gv("player.name"),
    team        = gv("team.name"),
    x = raw$x, y = raw$y,
    body_part   = gv("shot.body_part.name"),
    technique   = gv("shot.technique.name"),
    shot_type   = gv("shot.type.name"),
    play_pattern = gv("play_pattern.name"),
    under_pressure = isTRUE_vec(gv("under_pressure")),
    first_time     = isTRUE_vec(gv("shot.first_time")),
    statsbomb_xg   = as.numeric(gv("shot.statsbomb_xg")),   # BENCHMARK only
    outcome        = gv("shot.outcome.name")
  )

  # outcome label + exclusions
  df <- df %>%
    mutate(is_goal = as.integer(outcome == "Goal")) %>%
    filter(!is.na(x), !is.na(y))

  n_pre <- nrow(df)
  is_pen <- df$shot_type == "Penalty"
  message("Excluding ", sum(is_pen, na.rm = TRUE), " penalties.")
  df <- df %>% filter(is.na(shot_type) | shot_type != "Penalty")

  # engineered geometry + context
  df <- df %>%
    mutate(
      distance   = distance_to_goal(x, y),
      angle      = shot_angle(x, y),
      body_part  = dplyr::coalesce(body_part, "Unknown"),
      body_cat   = dplyr::case_when(
        body_part == "Head" ~ "head",
        body_part %in% c("Left Foot", "Right Foot") ~ "foot",
        TRUE ~ "other"),
      is_open_play = as.integer(dplyr::coalesce(shot_type, "Open Play") == "Open Play"),
      is_free_kick = as.integer(dplyr::coalesce(shot_type, "") == "Free Kick"),
      set_piece    = as.integer(is_open_play == 0)
    )

  # freeze-frame features (row-wise)
  ff_col <- if ("shot.freeze_frame" %in% names(raw)) raw[["shot.freeze_frame"]] else
    vector("list", nrow(raw))
  # align ff_col to the filtered df via shot_id
  keep_idx <- match(df$shot_id, gv("id"))
  ff_list <- ff_col[keep_idx]

  ff <- map_dfr(seq_len(nrow(df)), function(i)
    as_tibble(freeze_features(ff_list[[i]], df$x[i], df$y[i])))
  df <- bind_cols(df, ff)

  # missing freeze frame handling: indicator + median impute the numeric fields
  pct_missing <- mean(df$has_freeze == 0 | is.na(df$has_freeze)) * 100
  message(sprintf("Shots missing freeze frame: %.1f%%", pct_missing))
  med <- function(v) ifelse(is.na(v), median(v, na.rm = TRUE), v)
  df <- df %>%
    mutate(
      ff_missing       = as.integer(is.na(has_freeze) | has_freeze == 0),
      n_def_cone       = med(n_def_cone),
      dist_nearest_def = med(dist_nearest_def),
      gk_dist_goal     = med(gk_dist_goal)
    )

  message("Final modeling shots: ", nrow(df),
          " | goals: ", sum(df$is_goal),
          sprintf(" | base rate: %.3f", mean(df$is_goal)))
  saveRDS(df, out_path)
  invisible(df)
}

if (sys.nframe() == 0 || identical(environment(), globalenv())) {
  build_features()
}