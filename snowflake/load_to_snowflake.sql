-- snowflake/load_to_snowflake.sql ---------------------------------------------
-- Land the xg-lab outputs in Snowflake as dashboard-ready tables.
--   DORA.XG.SHOTS            shot-level (our xG + StatsBomb xG + outcome)
--   DORA.XG.FINISHING_SKILL  player finishing-skill leaderboard
-- Reuses the dev_xs XS warehouse + trial_guard credit cap from snowflake-layer.
-- Load is tiny (~20k rows); a few seconds of XS time.
--
-- Run order: PUT the two CSVs to the stage (from the CLI), then this file.
-- =============================================================================
USE ROLE ACCOUNTADMIN;
USE WAREHOUSE dev_xs;
CREATE DATABASE IF NOT EXISTS dora;
CREATE SCHEMA  IF NOT EXISTS dora.xg;
USE SCHEMA dora.xg;

CREATE FILE FORMAT IF NOT EXISTS csv_q
  TYPE = 'CSV' SKIP_HEADER = 1
  FIELD_OPTIONALLY_ENCLOSED_BY = '"'     -- names contain commas/accents
  NULL_IF = ('') EMPTY_FIELD_AS_NULL = TRUE;

CREATE STAGE IF NOT EXISTS stg_xg FILE_FORMAT = csv_q;

-- ---- tables -----------------------------------------------------------------
CREATE OR REPLACE TABLE shots (
  shot_id          STRING,
  match_id         INTEGER,
  competition      STRING,
  season           STRING,
  minute           INTEGER,
  player_id        INTEGER,
  player_name      STRING,
  team             STRING,
  x                FLOAT,
  y                FLOAT,
  body_cat         STRING,
  shot_type        STRING,
  is_open_play     INTEGER,
  set_piece        INTEGER,
  under_pressure   INTEGER,
  first_time       INTEGER,
  distance         FLOAT,
  angle            FLOAT,
  n_def_cone       INTEGER,
  dist_nearest_def FLOAT,
  our_xg           FLOAT,   -- our out-of-fold xgboost xG
  statsbomb_xg     FLOAT,   -- StatsBomb production xG (benchmark)
  is_goal          INTEGER
);

CREATE OR REPLACE TABLE finishing_skill (
  player_name STRING,
  shots       INTEGER,
  goals       INTEGER,
  xg          FLOAT,
  gax         FLOAT,        -- goals above expected
  skill_logit FLOAT,        -- shrunken finishing skill (log-odds)
  ci          STRING        -- 95% interval
);

-- ---- load (PUT runs from the CLI; see the snow PUT commands in the README) ---
COPY INTO shots           FROM @stg_xg PATTERN='.*shots.*\.csv.*'           ON_ERROR='ABORT_STATEMENT';
COPY INTO finishing_skill FROM @stg_xg PATTERN='.*finishing_skill.*\.csv.*' ON_ERROR='ABORT_STATEMENT';

-- ---- validate ---------------------------------------------------------------
SELECT 'shots' AS tbl, COUNT(*) AS row_count, SUM(is_goal) AS goals,
       ROUND(AVG(is_goal),3) AS base_rate, COUNT(DISTINCT player_id) AS players,
       COUNT(DISTINCT competition) AS competitions
FROM shots
UNION ALL
SELECT 'finishing_skill', COUNT(*), NULL, NULL, COUNT(DISTINCT player_name), NULL
FROM finishing_skill;