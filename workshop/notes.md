# Hybrid search run sheet

```sh
git clone https://github.com/codingforentrepreneurs/Hybrid-Search-Workshop.git
cd Hybrid-Search-Workshop
```

Run every terminal command from this folder (the repository root).

## 0. Getting started (before the session)

- Docker Desktop (only if not using Tiger Cloud): https://www.docker.com/products/docker-desktop/
- uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Tiger Cloud account: https://console.cloud.timescale.com

Pre-download the embedding model:

```sh
uv run workshop/load.py --download-only
```

psql client (skip if `psql --version` works):

```sh
brew install libpq && brew link --force libpq    # macOS
sudo apt install -y postgresql-client           # Ubuntu / Debian
```

## 1. Create a database, then set the connection string

Create your `.env` file:

```sh
cp .env.sample .env
```

**Option A: Tiger Cloud.** Create a service, copy its connection string, and paste it into `.env`:

```sh
DATABASE_URL=paste-your-tiger-cloud-connection-string-here
```

**Option B: Docker** (extensions preinstalled). Start the database:

```sh
docker run -d --name hybrid-search -p 5432:5432 -e POSTGRES_PASSWORD=password timescale/timescaledb-ha:pg17 -c shared_preload_libraries=timescaledb,pg_textsearch
```

Then put this in `.env`:

```sh
DATABASE_URL=postgres://postgres:password@localhost:5432/postgres
```

## 2. Load .env and open psql

```sh
set -a; source .env; set +a
psql "$DATABASE_URL"
```

`set -a; source .env; set +a` exports everything in `.env` into this terminal (the same as a `use dotenv` shortcut). Windows PowerShell: run `$env:DATABASE_URL = "..."` instead.

```sql
\timing on
```

## 3. Extensions

```sql
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_textsearch;
```

## 4. Create the table

```sql
DROP TABLE IF EXISTS movies;

CREATE TABLE movies (
    id           bigint PRIMARY KEY,   -- Wikipedia ID from the CMU corpus
    title        text NOT NULL,
    release_year int,
    genres       text[] NOT NULL DEFAULT '{}',
    plot         text NOT NULL,
    search_text  text NOT NULL,        -- title + genres + plot: what BM25 indexes
    embedding    vector(384)           -- what semantic search compares
);
```

## 5. Load the sample data (text only, no embeddings yet)

```sql
\i workshop/load_data.sql
```

Basic query:

```sql
SELECT title, release_year, genres FROM movies LIMIT 5;
```

## 6. Keyword search (BM25)

```sql
CREATE INDEX movies_bm25_idx ON movies
    USING bm25 (search_text) WITH (text_config = 'english');
```

```sql
SELECT title, release_year, search_text <@> 'Indiana Jones' AS score
FROM movies ORDER BY score LIMIT 5;
-- The 4 Indiana Jones films. More negative = more relevant.
```

```sql
SELECT title, release_year FROM movies
ORDER BY search_text <@> 'shark beach town' LIMIT 3;
-- Jaws, The Life Aquatic, Finding Nemo.
```

```sql
SELECT title, release_year FROM movies
ORDER BY search_text <@> 'someone reliving the same day' LIMIT 5;
-- 50 First Dates beats Groundhog Day: keywords miss the concept.
```

## 7. Embeddings (second terminal)

Open a second terminal in the repository folder. The script reads `.env` on its own:

```sh
uv run workshop/load.py
```

Expect `Loaded 200 movies (200 with embeddings).` in about 15 seconds.

Back in psql:

```sql
SELECT title, vector_dims(embedding), left(embedding::text, 60) FROM movies LIMIT 3;
```

## 8. Semantic search (DiskANN)

```sql
CREATE INDEX movies_embedding_idx ON movies
    USING diskann (embedding vector_cosine_ops);
```

```sql
SELECT title, round((embedding <=> (SELECT embedding FROM movies WHERE title = 'Groundhog Day'))::numeric, 3) AS distance
FROM movies ORDER BY distance LIMIT 6;
-- Big, The Truman Show, Before Sunrise, 50 First Dates. No shared keywords.
```

```sql
SELECT title, round((embedding <=> (SELECT embedding FROM movies WHERE title = 'Jaws'))::numeric, 3) AS distance
FROM movies ORDER BY distance LIMIT 5;
-- Black Water, Finding Nemo, Dead Calm, The Life Aquatic.
```

```sql
SELECT title, round((embedding <=> (SELECT embedding FROM movies WHERE title = 'The Matrix'))::numeric, 3) AS distance
FROM movies ORDER BY distance LIMIT 5;
-- The Terminator, Alien, Lara Croft, Source Code.
```

```sql
SELECT title, round((embedding <=> (SELECT embedding FROM movies WHERE title = 'Indiana Jones and the Raiders of the Lost Ark'))::numeric, 3) AS distance
FROM movies ORDER BY distance LIMIT 6;
-- The other three Indy films, then Lara Croft and The Lost World.
```

Prove both indexes are used:

```sql
ANALYZE movies;
EXPLAIN SELECT title FROM movies ORDER BY search_text <@> 'Indiana Jones' LIMIT 5;
EXPLAIN SELECT title FROM movies
ORDER BY embedding <=> (SELECT embedding FROM movies WHERE title = 'Jaws') LIMIT 5;
-- Look for: Index Scan using movies_bm25_idx / movies_embedding_idx
```

## 9. Hybrid search (RRF)

```sql
\set query 'treasure hunt'
\set likeStatement 'Indiana Jones and the Raiders of the Lost Ark'

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
-- National Treasure, King Solomon's Mines, Cradle of Life, The Mummy.
-- The Goonies is keyword #2 but semantic #29, so it drops.
```

Try other inputs. Each pair reruns the same query from `workshop/hybrid.sql`:

```sql
\set query 'reliving the same day'
\set likeStatement 'Groundhog Day'
\i workshop/hybrid.sql
-- Groundhog Day #1 (keyword 2, semantic 1).
```

```sql
\set query 'ocean'
\set likeStatement 'Jaws'
\i workshop/hybrid.sql
-- Jaws, Finding Nemo, Titanic. Blank ranks = found by only one method.
```

## 10. Free-text hybrid search (exit psql or use a second terminal)

```sh
uv run workshop/ask.py "someone reliving the same day"
uv run workshop/ask.py "an archaeologist racing to recover an ancient artifact"
uv run workshop/ask.py "a shark terrorizes a beach town"
uv run workshop/ask.py "giant killer fish attacks swimmers at a seaside resort"
```

- The last query shows hybrid hurting: semantic search gets Jaws #1, but "fish" pulls Finding Nemo to the top of the hybrid list.
- Key line: Python only turns words into numbers. Postgres does all the searching and ranking.

## Catch up (if you fall behind)

From the repository folder:

```sh
set -a; source .env; set +a
psql "$DATABASE_URL" -f workshop/schema.sql       # Steps 3-4
psql "$DATABASE_URL" -f workshop/load_data.sql    # Step 5
uv run workshop/load.py                           # Step 7
psql "$DATABASE_URL" -f workshop/indexes.sql      # Steps 6 + 8 indexes
```

## Quick fixes

- `relation "movies" does not exist`: run Step 4.
- `no BM25 index found` or `index "movies_bm25_idx" does not exist`: run the CREATE INDEX in Step 6.
- `pg_textsearch library not loaded`: Docker needs the `-c shared_preload_libraries=...` flag from Step 1.
- `\i` or `\copy` can't find a file: start psql from the repository folder.
- `uv` can't find Python: `uv python install 3.13`.

## Reset

```sql
DROP TABLE IF EXISTS movies;
```

---

Dataset: [CMU Movie Summary Corpus](https://www.cs.cmu.edu/~ark/personas/) by David Bamman, Brendan O'Connor and Noah Smith ([CC BY-SA 3.0 US](https://creativecommons.org/licenses/by-sa/3.0/us/)). `movies_sample.jsonl` is a 200-movie subset.
