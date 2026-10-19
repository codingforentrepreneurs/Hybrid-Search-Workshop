-- Load the 200 movies (text only, no embeddings) straight from the JSONL file.
--   psql "$DATABASE_URL" -f workshop/load_data.sql
-- Run from the repository root. `uv run workshop/load.py` fills in embeddings later.

CREATE TEMP TABLE staging (doc jsonb);

-- One JSON document per line. The CSV quote/delimiter characters are set to
-- bytes that never appear in the file, so each line arrives untouched.
\copy staging FROM 'workshop/movies_sample.jsonl' WITH (FORMAT csv, QUOTE e'\x01', DELIMITER e'\x02')

INSERT INTO movies (id, title, release_year, genres, plot, search_text)
SELECT (doc->>'wikipedia_id')::bigint,
       doc->>'title',
       (doc->>'release_year')::int,
       ARRAY(SELECT jsonb_array_elements_text(doc->'genres')),
       doc->>'plot',
       concat_ws(E'\n', doc->>'title',
                 array_to_string(ARRAY(SELECT jsonb_array_elements_text(doc->'genres')), ', '),
                 '', doc->>'plot')
FROM staging
ON CONFLICT (id) DO NOTHING;

SELECT count(*) AS movies, count(embedding) AS with_embeddings FROM movies;
