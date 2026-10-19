-- Catch-up file: the two indexes from Steps 5 and 6 of WORKSHOP.md.
--   psql "$DATABASE_URL" -f workshop/indexes.sql

CREATE INDEX IF NOT EXISTS movies_bm25_idx ON movies
    USING bm25 (search_text) WITH (text_config = 'english');

CREATE INDEX IF NOT EXISTS movies_embedding_idx ON movies
    USING diskann (embedding vector_cosine_ops);

ANALYZE movies;
