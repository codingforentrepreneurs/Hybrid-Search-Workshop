-- Catch-up file: everything from Steps 2 and 3 of WORKSHOP.md in one go.
--   psql "$DATABASE_URL" -f workshop/schema.sql

CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;  -- pulls in pgvector too
CREATE EXTENSION IF NOT EXISTS pg_textsearch;

CREATE TABLE IF NOT EXISTS movies (
    id           bigint PRIMARY KEY,   -- Wikipedia ID from the CMU corpus
    title        text NOT NULL,
    release_year int,
    genres       text[] NOT NULL DEFAULT '{}',
    plot         text NOT NULL,
    search_text  text NOT NULL,        -- title + genres + plot, what BM25 indexes
    embedding    vector(384)           -- BAAI/bge-small-en-v1.5
);
