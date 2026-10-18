Hand-written `Labels-GameState.json` fixtures in the SoccerNet GSR layout
(field names from Figure S3 of arXiv:2404.11335). They are not SoccerNet data:
the numbers are invented, the images are rendered by the tests. SNGS-001 and
SNGS-003 share `game_id` 11 so the game-disjoint check has a leak to catch.
