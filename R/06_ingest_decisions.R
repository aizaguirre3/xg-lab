# R/06_ingest_decisions.R -----------------------------------------------------
# VAR-Lab layer 1 ingest: refereeing-decision events + StatsBomb 360 frames.
#
# Pulls, for every open-data match that has 360 data available, the events that
# encode a refereeing decision (Foul Committed, Foul Won, Bad Behaviour,
# Offside, and passes whose outcome is "Pass Offside"), joins each one to its
# 360 freeze frame on event_uuid, attaches the match referee, and writes:
#   data/match_index_360.rds   one row per match with 360 data (referee, teams)
#   data/decisions.rds         one row per decision event (+ nested frame)
#   data/pass_frames.rds       all Pass Offside + sampled forward passes with
#                              frames, for the offside geometry layer (M2b)
#   analysis/decisions_counts.md  counts table (regenerated every run)
#
# Raw JSON is cached under data/sb_raw/{matches,events,three-sixty}/ so the
# pipeline never re-downloads; delete the cache to force a fresh pull.
#
# StatsBomb Open Data licence (https://github.com/statsbomb/open-data):
#   non-commercial use only, no redistribution of the data, attribution and
#   the StatsBomb logo required on published analysis. Only code and aggregate
#   results are committed; data/ is gitignored.
#
# Config via env vars:
#   XG_DEC_MAX_MATCHES  cap on matches (default: all); "" = no cap
#   XG_DEC_NEG_PER_MATCH negatives sampled per match for pass_frames (default 15)
#   XG_DEC_CORES        parallel workers for parsing (default: cores - 1)
#   XG_SB_REF           statsbomb/open-data git ref to pull from (default master)
# -----------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(jsonlite); library(dplyr); library(purrr); library(tibble); library(tidyr)
})

# Pin the StatsBomb open-data revision with XG_SB_REF (a branch or a commit
# SHA) so a later pull reproduces the same files; "master" follows upstream.
SB_REF  <- Sys.getenv("XG_SB_REF", "master")
SB_BASE <- sprintf("https://raw.githubusercontent.com/statsbomb/open-data/%s/data", SB_REF)
RAW_DIR <- file.path("data", "sb_raw")
DECISION_TYPES <- c("Foul Committed", "Foul Won", "Bad Behaviour", "Offside")
PITCH_AREA <- 120 * 80

# ---- small pure helpers (unit-tested in tests/test_decisions.R) --------------

# Shoelace polygon area from a flat numeric vector c(x1,y1,x2,y2,...).
polygon_area <- function(v) {
  if (is.null(v) || length(v) < 6) return(NA_real_)
  x <- v[seq(1, length(v), 2)]; y <- v[seq(2, length(v), 2)]
  n <- length(x)
  j <- c(2:n, 1)
  abs(sum(x * y[j] - x[j] * y)) / 2
}

# Is a Foul Committed location inside the fouling team's OWN penalty box?
# StatsBomb coordinates are always from the acting team's perspective
# (attacking right, goal at x = 120), so the team's own box is x <= 18.
in_own_box <- function(x, y) !is.na(x) & !is.na(y) & x <= 18 & y >= 18 & y <= 62

# Pull [x, y] out of a StatsBomb location list cell; NA if absent.
loc_xy <- function(cell) {
  if (is.null(cell) || length(cell) < 2) return(c(NA_real_, NA_real_))
  v <- suppressWarnings(as.numeric(unlist(cell)))
  if (length(v) < 2 || any(is.na(v[1:2]))) return(c(NA_real_, NA_real_))
  v[1:2]
}

# robust logical coercion (StatsBomb omits FALSE -> NA in flattened frames)
isTRUE_vec <- function(x) !is.na(x) & as.logical(x)

# Compact one 360 freeze frame (data.frame with teammate/actor/keeper/location)
# into a tibble with numeric x, y and logical flags. NULL -> NULL.
compact_frame <- function(ff) {
  if (is.null(ff) || !is.data.frame(ff) || nrow(ff) == 0) return(NULL)
  xy <- t(vapply(ff$location, loc_xy, numeric(2)))
  tibble(x = xy[, 1], y = xy[, 2],
         teammate = isTRUE_vec(ff$teammate),
         actor    = isTRUE_vec(ff$actor),
         keeper   = isTRUE_vec(ff$keeper))
}

# Summary statistics of a compacted frame, from the acting team's perspective.
frame_summary <- function(fr) {
  if (is.null(fr)) {
    return(tibble(has_frame = 0L, n_frame = NA_integer_, n_teammates = NA_integer_,
                  n_opponents = NA_integer_, keeper_opp_in_frame = NA,
                  keeper_own_in_frame = NA, actor_in_frame = NA))
  }
  tibble(has_frame = 1L, n_frame = nrow(fr),
         n_teammates = sum(fr$teammate), n_opponents = sum(!fr$teammate),
         keeper_opp_in_frame = any(fr$keeper & !fr$teammate),
         keeper_own_in_frame = any(fr$keeper & fr$teammate),
         actor_in_frame = any(fr$actor))
}

# ---- cached JSON access -------------------------------------------------------

# Read a StatsBomb JSON file from the local raw cache, downloading it first if
# missing. rel_path is e.g. "events/3895292.json". Returns NULL on failure.
read_sb_json <- function(rel_path, raw_dir = RAW_DIR, base = SB_BASE) {
  local <- file.path(raw_dir, rel_path)
  if (!file.exists(local) || file.size(local) == 0) {
    dir.create(dirname(local), showWarnings = FALSE, recursive = TRUE)
    ok <- tryCatch({
      utils::download.file(file.path(base, rel_path), local, quiet = TRUE, mode = "wb")
      TRUE
    }, error = function(e) FALSE, warning = function(w) FALSE)
    if (!ok) { unlink(local); return(NULL) }
  }
  tryCatch(jsonlite::fromJSON(local, flatten = TRUE), error = function(e) NULL)
}

# One row per match whose 360 frames are available, with referee + teams.
build_match_index_360 <- function(raw_dir = RAW_DIR, max_matches = NA) {
  comps <- read_sb_json("competitions.json", raw_dir)
  stopifnot(!is.null(comps))
  sel <- comps %>%
    filter(!is.na(match_available_360)) %>%
    distinct(competition_id, season_id, competition_name, season_name)
  message("Competition-seasons with 360 data: ", nrow(sel))

  idx <- pmap_dfr(sel, function(competition_id, season_id, competition_name, season_name) {
    m <- read_sb_json(sprintf("matches/%d/%d.json", competition_id, season_id), raw_dir)
    if (is.null(m) || !"match_id" %in% names(m)) return(tibble())
    gv <- function(nm) if (nm %in% names(m)) m[[nm]] else rep(NA, nrow(m))
    tibble(match_id = m$match_id,
           competition = competition_name, season = season_name,
           competition_id = competition_id, season_id = season_id,
           match_date = gv("match_date"),
           home_team = gv("home_team.home_team_name"),
           away_team = gv("away_team.away_team_name"),
           home_score = gv("home_score"), away_score = gv("away_score"),
           referee_id = gv("referee.id"), referee = gv("referee.name"),
           stadium = gv("stadium.name"),
           match_status_360 = gv("match_status_360"))
  }) %>% distinct(match_id, .keep_all = TRUE)

  n_all <- nrow(idx)
  idx <- idx %>% filter(match_status_360 == "available") %>%
    arrange(competition, season, match_date, match_id)
  message("Matches listed: ", n_all, " | with 360 available: ", nrow(idx))
  if (!is.na(max_matches) && nrow(idx) > max_matches) {
    set.seed(7406)
    idx <- idx[sort(sample(nrow(idx), max_matches)), ]
    message("Capped to ", nrow(idx), " matches (XG_DEC_MAX_MATCHES).")
  }
  idx
}

# ---- per-match parsing --------------------------------------------------------

# Flattened events data.frame -> tidy tibble of the columns we keep.
tidy_events <- function(ev, match_id) {
  gv <- function(nm) if (nm %in% names(ev)) ev[[nm]] else rep(NA, nrow(ev))
  xy  <- t(vapply(gv("location"), loc_xy, numeric(2)))
  exy <- t(vapply(gv("pass.end_location"), loc_xy, numeric(2)))
  tibble(
    event_id   = ev$id,
    match_id   = match_id,
    index      = gv("index"),
    period     = gv("period"),
    minute     = gv("minute"),
    second     = gv("second"),
    timestamp  = gv("timestamp"),
    type       = gv("type.name"),
    team       = gv("team.name"),
    team_id    = gv("team.id"),
    possession_team = gv("possession_team.name"),
    player_id  = gv("player.id"),
    player_name = gv("player.name"),
    position   = gv("position.name"),
    play_pattern = gv("play_pattern.name"),
    under_pressure = isTRUE_vec(gv("under_pressure")),
    x = xy[, 1], y = xy[, 2],
    card       = dplyr::coalesce(as.character(gv("foul_committed.card.name")),
                                 as.character(gv("bad_behaviour.card.name"))),
    penalty    = isTRUE_vec(gv("foul_committed.penalty")),
    advantage  = isTRUE_vec(gv("foul_committed.advantage")),
    offensive  = isTRUE_vec(gv("foul_committed.offensive")),
    foul_type  = as.character(gv("foul_committed.type.name")),
    foul_won_defensive = isTRUE_vec(gv("foul_won.defensive")),
    foul_won_penalty   = isTRUE_vec(gv("foul_won.penalty")),
    pass_outcome   = as.character(gv("pass.outcome.name")),
    pass_end_x = exy[, 1], pass_end_y = exy[, 2],
    recipient_id   = gv("pass.recipient.id"),
    recipient_name = gv("pass.recipient.name"),
    pass_length    = gv("pass.length"),
    pass_angle     = gv("pass.angle"),
    pass_height    = as.character(gv("pass.height.name")),
    pass_body_part = as.character(gv("pass.body_part.name")),
    pass_type      = as.character(gv("pass.type.name")),
    related_events = gv("related_events")
  )
}

# Join a tidy event table to the 360 frames of the same match.
attach_frames <- function(df, frames) {
  if (is.null(frames) || !is.data.frame(frames) || nrow(frames) == 0 ||
      !"event_uuid" %in% names(frames)) {
    df$frame <- replicate(nrow(df), NULL, simplify = FALSE)
    df$visible_area <- replicate(nrow(df), NULL, simplify = FALSE)
    return(bind_cols(df, map_dfr(seq_len(nrow(df)), ~ frame_summary(NULL))))
  }
  pos <- match(df$event_id, frames$event_uuid)
  df$frame <- lapply(pos, function(i) if (is.na(i)) NULL else compact_frame(frames$freeze_frame[[i]]))
  df$visible_area <- lapply(pos, function(i)
    if (is.na(i)) NULL else suppressWarnings(as.numeric(unlist(frames$visible_area[[i]]))))
  summ <- map_dfr(df$frame, frame_summary)
  summ$visible_area_frac <- vapply(df$visible_area, function(v) polygon_area(v) / PITCH_AREA, numeric(1))
  bind_cols(df, summ)
}

# Decision events + frames for one match (row of the match index).
parse_match_decisions <- function(row, raw_dir = RAW_DIR, n_neg = 15, seed = 7406) {
  ev <- read_sb_json(sprintf("events/%d.json", row$match_id), raw_dir)
  if (is.null(ev) || !"type.name" %in% names(ev)) return(NULL)
  te <- tidy_events(ev, row$match_id)
  frames <- read_sb_json(sprintf("three-sixty/%d.json", row$match_id), raw_dir)

  is_decision <- te$type %in% DECISION_TYPES |
    (te$type == "Pass" & !is.na(te$pass_outcome) & te$pass_outcome == "Pass Offside")
  dec <- attach_frames(te[is_decision, ], frames) %>%
    mutate(competition = row$competition, season = row$season,
           referee = row$referee, referee_id = row$referee_id,
           in_own_box = in_own_box(x, y))

  # Pass frames for the offside layer: every Pass Offside plus a per-match
  # sample of completed forward passes into the opponent's half (the population
  # in which an offside call is possible). Negatives are implicit "no call".
  pass <- te[te$type == "Pass" & !is.na(te$x) & !is.na(te$pass_end_x), ]
  pos_rows <- pass[!is.na(pass$pass_outcome) & pass$pass_outcome == "Pass Offside", ]
  cand <- pass[is.na(pass$pass_outcome) & pass$pass_end_x > pass$x &
                 pass$pass_end_x > 60 & !is.na(pass$recipient_id), ]
  set.seed(seed + row$match_id)
  neg_rows <- cand[sample(nrow(cand), min(n_neg, nrow(cand))), ]
  pf <- attach_frames(bind_rows(pos_rows, neg_rows), frames) %>%
    mutate(competition = row$competition, season = row$season,
           referee = row$referee, referee_id = row$referee_id,
           offside_call = as.integer(!is.na(pass_outcome) & pass_outcome == "Pass Offside"))

  list(decisions = dec, pass_frames = pf,
       n_events = nrow(te), n_frames = if (is.null(frames)) 0L else nrow(frames))
}

# ---- counts table -------------------------------------------------------------

decision_counts <- function(dec, idx) {
  by_cs <- dec %>%
    group_by(competition, season) %>%
    summarise(
      fouls        = sum(type == "Foul Committed"),
      carded_fouls = sum(type == "Foul Committed" & !is.na(card)),
      yellow       = sum(type == "Foul Committed" & card %in% "Yellow Card"),
      second_yellow = sum(type == "Foul Committed" & card %in% "Second Yellow"),
      red          = sum(type == "Foul Committed" & card %in% "Red Card"),
      penalties    = sum(type == "Foul Committed" & penalty),
      handballs    = sum(type == "Foul Committed" & foul_type %in% "Handball"),
      bad_behaviour = sum(type == "Bad Behaviour"),
      pass_offside = sum(type == "Pass" & pass_outcome %in% "Pass Offside"),
      offside_events = sum(type == "Offside"),
      frame_pct_fouls = 100 * mean(has_frame[type == "Foul Committed"]),
      frame_pct_pass_offside = 100 * mean(has_frame[type == "Pass" & pass_outcome %in% "Pass Offside"]),
      .groups = "drop")
  m <- idx %>% count(competition, season, name = "matches")
  out <- left_join(m, by_cs, by = c("competition", "season"))
  tot <- out %>% summarise(across(c(matches, fouls, carded_fouls, yellow, second_yellow, red,
                                    penalties, handballs, bad_behaviour, pass_offside,
                                    offside_events), sum),
                           frame_pct_fouls = 100 * mean(dec$has_frame[dec$type == "Foul Committed"]),
                           frame_pct_pass_offside = 100 * mean(dec$has_frame[dec$type == "Pass" &
                                                                   dec$pass_outcome %in% "Pass Offside"])) %>%
    mutate(competition = "TOTAL", season = "")
  bind_rows(out, tot)
}

counts_to_markdown <- function(cnt, path) {
  hdr <- c("Competition", "Season", "Matches", "Fouls", "Carded", "Yellow", "2nd yellow", "Red",
           "Penalties", "Handballs", "Bad behaviour", "Pass offside", "Offside events",
           "Frame % fouls", "Frame % pass offside")
  rows <- apply(cnt, 1, function(r) paste0("| ", paste(c(
    r[["competition"]], r[["season"]], r[["matches"]], r[["fouls"]], r[["carded_fouls"]],
    r[["yellow"]], r[["second_yellow"]], r[["red"]], r[["penalties"]], r[["handballs"]],
    r[["bad_behaviour"]], r[["pass_offside"]], r[["offside_events"]],
    sprintf("%.1f", as.numeric(r[["frame_pct_fouls"]])),
    sprintf("%.1f", as.numeric(r[["frame_pct_pass_offside"]]))), collapse = " | "), " |"))
  txt <- c("# Refereeing-decision events in StatsBomb Open Data (matches with 360 frames)",
           "",
           sprintf("Generated by `R/06_ingest_decisions.R` on %s. Counts are computed from the data, never typed in.",
                   format(Sys.Date())),
           "Data: StatsBomb Open Data, non-commercial licence, attribution required.",
           "",
           paste0("| ", paste(hdr, collapse = " | "), " |"),
           paste0("|", paste(rep("---", length(hdr)), collapse = "|"), "|"),
           rows, "")
  writeLines(txt, path)
  invisible(path)
}

# ---- driver -------------------------------------------------------------------

ingest_decisions <- function(
    raw_dir     = RAW_DIR,
    max_matches = suppressWarnings(as.integer(Sys.getenv("XG_DEC_MAX_MATCHES", ""))),
    n_neg       = as.integer(Sys.getenv("XG_DEC_NEG_PER_MATCH", "15")),
    cores       = suppressWarnings(as.integer(Sys.getenv("XG_DEC_CORES", ""))),
    out_dir     = "data") {

  idx <- build_match_index_360(raw_dir, max_matches)
  if (is.na(cores)) cores <- max(1L, parallel::detectCores() - 1L)
  message("Parsing ", nrow(idx), " matches on ", cores, " core(s)...")

  rows <- split(idx, seq_len(nrow(idx)))
  worker <- function(r) tryCatch(parse_match_decisions(r, raw_dir, n_neg),
                                 error = function(e) { message("  match ", r$match_id, ": ", conditionMessage(e)); NULL })
  res <- if (cores > 1 && .Platform$OS.type != "windows") {
    parallel::mclapply(rows, worker, mc.cores = cores)
  } else lapply(rows, worker)
  ok <- !vapply(res, is.null, logical(1))
  message("Parsed ", sum(ok), "/", length(res), " matches.")

  dec <- bind_rows(map(res[ok], "decisions"))
  pf  <- bind_rows(map(res[ok], "pass_frames"))
  per_match <- tibble(match_id = idx$match_id[ok],
                      n_events = map_int(res[ok], "n_events"),
                      n_frames = map_int(res[ok], "n_frames"))
  idx <- left_join(idx, per_match, by = "match_id")

  cnt <- decision_counts(dec, idx[ok, ])
  message("Decision counts (TOTAL row):")
  print(as.data.frame(cnt[cnt$competition == "TOTAL", ]), digits = 4)

  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  saveRDS(idx, file.path(out_dir, "match_index_360.rds"))
  saveRDS(dec, file.path(out_dir, "decisions.rds"))
  saveRDS(pf,  file.path(out_dir, "pass_frames.rds"))
  saveRDS(cnt, file.path(out_dir, "decisions_counts.rds"))
  counts_to_markdown(cnt, file.path("analysis", "decisions_counts.md"))
  message("Wrote data/decisions.rds (", nrow(dec), " rows), data/pass_frames.rds (",
          nrow(pf), " rows), analysis/decisions_counts.md")
  invisible(list(index = idx, decisions = dec, pass_frames = pf, counts = cnt))
}

if (sys.nframe() == 0) ingest_decisions()
