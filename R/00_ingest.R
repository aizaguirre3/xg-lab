# R/00_ingest.R ---------------------------------------------------------------
# Pull StatsBomb Open Data shot events via raw JSON (no StatsBombR dependency,
# which the build spec flags as possibly unmaintained). Filters to shots, keeps
# the freeze frame, and caches a tidy shots table to data/shots_raw.rds.
#
# StatsBomb Open Data: https://github.com/statsbomb/open-data
#   Free, licensed for NON-COMMERCIAL use. Attribution required (see README).
#
# Config via env vars (so the same script does a tractable subset now and the
# full pull later):
#   XG_COMPETITIONS  comma-sep competition names (default: curated set below)
#   XG_MAX_MATCHES   cap on total matches downloaded (default 600; "" = no cap)
# -----------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(jsonlite); library(dplyr); library(purrr); library(tibble)
})

SB_BASE <- "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# Curated default: competitions that together yield tens of thousands of shots
# while keeping the download bounded. Overridable via XG_COMPETITIONS.
DEFAULT_COMPETITIONS <- c(
  "FIFA World Cup", "UEFA Euro", "Women's World Cup",
  "La Liga", "FA Women's Super League", "Champions League"
)

# Parse the XG_COMPETITIONS env var (comma-sep) into a clean vector, or NULL.
env_competitions <- function() {
  raw <- Sys.getenv("XG_COMPETITIONS", "")
  if (!nzchar(raw)) return(NULL)
  parts <- trimws(strsplit(raw, ",")[[1]])
  parts <- parts[nzchar(parts)]
  if (length(parts) == 0) NULL else parts
}

# Politely fetch + parse a JSON URL, with one retry. Returns NULL on failure.
fetch_json <- function(url) {
  for (attempt in 1:2) {
    out <- tryCatch(jsonlite::fromJSON(url, flatten = TRUE),
                    error = function(e) NULL)
    if (!is.null(out)) return(out)
    Sys.sleep(0.5)
  }
  NULL
}

ingest_shots <- function(
    competitions = NULL,
    max_matches  = suppressWarnings(as.integer(Sys.getenv("XG_MAX_MATCHES", "600"))),
    cache_path   = file.path("data", "shots_raw.rds")) {

  if (is.null(competitions)) competitions <- env_competitions()
  if (is.null(competitions)) competitions <- DEFAULT_COMPETITIONS
  message("Competitions requested: ", paste(competitions, collapse = ", "))

  comps <- fetch_json(file.path(SB_BASE, "competitions.json"))
  stopifnot(!is.null(comps))
  sel <- comps %>%
    filter(competition_name %in% competitions) %>%
    distinct(competition_id, season_id, competition_name, season_name)
  message("Matched ", nrow(sel), " competition-seasons.")

  # gather match ids across selected comp-seasons
  match_index <- pmap_dfr(sel, function(competition_id, season_id,
                                        competition_name, season_name) {
    m <- fetch_json(sprintf("%s/matches/%d/%d.json", SB_BASE, competition_id, season_id))
    if (is.null(m) || !"match_id" %in% names(m)) return(tibble())
    tibble(match_id = m$match_id,
           competition = competition_name, season = season_name)
  })
  match_index <- distinct(match_index, match_id, .keep_all = TRUE)
  message("Found ", nrow(match_index), " matches.")

  if (!is.na(max_matches) && nrow(match_index) > max_matches) {
    set.seed(7406)
    match_index <- match_index[sort(sample(nrow(match_index), max_matches)), ]
    message("Capped to ", nrow(match_index), " matches (XG_MAX_MATCHES).")
  }

  # pull events per match, keep only shots
  cols_keep <- c("id", "minute", "second", "period", "under_pressure",
                 "play_pattern.name", "team.name", "player.id", "player.name",
                 "location", "shot.statsbomb_xg", "shot.outcome.name",
                 "shot.body_part.name", "shot.technique.name", "shot.type.name",
                 "shot.first_time", "shot.freeze_frame")

  shots <- map_dfr(seq_len(nrow(match_index)), function(i) {
    row <- match_index[i, ]
    ev <- fetch_json(sprintf("%s/events/%d.json", SB_BASE, row$match_id))
    if (is.null(ev) || !"type.name" %in% names(ev)) return(tibble())
    s <- ev[ev$type.name == "Shot", , drop = FALSE]
    if (nrow(s) == 0) return(tibble())
    present <- intersect(cols_keep, names(s))
    s <- s[, present, drop = FALSE]
    s$match_id    <- row$match_id
    s$competition <- row$competition
    s$season      <- row$season
    if (i %% 50 == 0) message("  ...", i, "/", nrow(match_index), " matches")
    as_tibble(s)
  })

  message("Pulled ", nrow(shots), " raw shot events.")
  dir.create(dirname(cache_path), showWarnings = FALSE, recursive = TRUE)
  saveRDS(shots, cache_path)
  saveRDS(match_index, file.path("data", "match_index.rds"))
  invisible(shots)
}

# Run when sourced directly (the pipeline calls ingest_shots()).
if (sys.nframe() == 0) {
  if (!file.exists(file.path("data", "shots_raw.rds")) ||
      identical(Sys.getenv("XG_FORCE_INGEST"), "1")) {
    ingest_shots()
  } else {
    message("data/shots_raw.rds exists; skipping ingest (set XG_FORCE_INGEST=1 to refresh).")
  }
}
