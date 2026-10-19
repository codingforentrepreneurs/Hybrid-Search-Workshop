-- Hybrid search with RRF. Set the inputs first, then run: \i workshop/hybrid.sql
--   \set query 'treasure hunt'
--   \set likeStatement 'Indiana Jones and the Raiders of the Lost Ark'

WITH keyword AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY score) AS rank
    FROM (
        SELECT id, search_text <@> :'query' AS score
        FROM movies ORDER BY score LIMIT 50
    ) k
), semantic AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY distance) AS rank
    FROM (
        SELECT id, embedding <=> (SELECT embedding FROM movies WHERE title = :'likeStatement') AS distance
        FROM movies ORDER BY distance LIMIT 50
    ) s
)
SELECT m.title, k.rank AS keyword_rank, s.rank AS semantic_rank,
       round(COALESCE(1.0 / (60 + k.rank), 0) + COALESCE(1.0 / (60 + s.rank), 0), 4) AS rrf_score
FROM keyword k
FULL OUTER JOIN semantic s USING (id)
JOIN movies m ON m.id = COALESCE(k.id, s.id)
ORDER BY rrf_score DESC
LIMIT 8;
