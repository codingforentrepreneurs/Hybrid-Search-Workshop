# Build hybrid search in PostgreSQL: follow-along guide

In about 45 minutes you will build keyword search (BM25 with `pg_textsearch`),
semantic search (StreamingDiskANN with `pgvectorscale`), and hybrid search
(Reciprocal Rank Fusion) in one PostgreSQL database.

We use a **200-movie sample** of the CMU Movie Summary Corpus so that embedding
takes seconds. The full demo app (42,000 movies, Django UI) runs on the
same foundation. At the end, the presenter switches to it.

| Step | What you do | Time |
|---|---|---|
| 0 | Before the webinar: install tools, start a database, download the model | 10 min (beforehand) |
| 1 | Connect | 2 min |
| 2 | Enable extensions | 2 min |
| 3 | Create the table | 3 min |
| 4 | Load and embed 200 movies | 2 min |
| 5 | Keyword search with BM25 | 7 min |
| 6 | Semantic search with DiskANN | 7 min |
| 7 | Hybrid search with RRF | 10 min |
| 8 | Compare all three from a script | 5 min |
| 9 | Presenter: the same thing at full scale | 5 min |

> **Fell behind?** Every step has a **Catch up** line that gets you to the
> finished state of that step. Run it and rejoin.

---

## Step 0 — Before the webinar

You need three things.

**1. `uv`** (runs the Python scripts and installs their dependencies automatically)

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh     # macOS / Linux
# Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**2. A PostgreSQL database with `pg_textsearch`, `pgvector`, and `pgvectorscale`.** Pick one:

- **Option A: Tiger Cloud (no install).** Create a free service at
  [console.cloud.timescale.com](https://console.cloud.timescale.com) and copy
  its connection string.
- **Option B: Docker (local).** These extensions are preinstalled in this image,
  which already preloads `pg_textsearch`. The `-c` flag just makes that explicit:

  ```sh
  docker run -d --name hybrid-search -p 5432:5432 \
    -e POSTGRES_PASSWORD=password \
    timescale/timescaledb-ha:pg17 \
    -c shared_preload_libraries=timescaledb,pg_textsearch
  ```

  Your connection string is `postgres://postgres:password@localhost:5432/postgres`.

  > **Windows (PowerShell):** put each command on one line, or replace each
  > trailing `\` with a backtick (`` ` ``). Use `$env:DATABASE_URL = "postgres://..."`
  > instead of `export`.

**3. This repository, plus the embedding model downloaded in advance** (~70 MB, so it doesn't compete for webinar bandwidth):

```sh
git clone <repo-url> hybrid-search-demo
cd hybrid-search-demo
uv run workshop/load.py --download-only
```

✅ You should see: `Model downloaded. You're ready for the workshop.`

The 200-movie sample is already in the repository. You can also download it on
its own from [this gist](https://gist.github.com/codingforentrepreneurs/eede18e3a7a203d9bcaff5c975aac2e9):

```sh
curl -L -o workshop/movies_sample.jsonl \
  https://gist.githubusercontent.com/codingforentrepreneurs/eede18e3a7a203d9bcaff5c975aac2e9/raw/movies_sample.jsonl
```

You also need a SQL client. `psql` is used below, but any client works (TablePlus,
DBeaver, the Tiger Cloud SQL editor). If you don't have `psql`, install it with
`brew install libpq` on macOS or `apt install postgresql-client` on Linux.

---

## Step 1 — Connect

Set your connection string once. Both the scripts and `psql` read it.

```sh
export DATABASE_URL="postgres://postgres:password@localhost:5432/postgres"   # or your Tiger Cloud URL
psql "$DATABASE_URL"
```

✅ You should see a `postgres=#` prompt on Docker (`tsdb=>` on Tiger Cloud).

---

## Step 2 — Enable the extensions

```sql
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;  -- CASCADE also installs pgvector
CREATE EXTENSION IF NOT EXISTS pg_textsearch;

SELECT extname, extversion FROM pg_extension
WHERE extname IN ('vector', 'vectorscale', 'pg_textsearch');
```

✅ You should see three rows.

| Extension | What it gives us |
|---|---|
| `vector` (pgvector) | the `vector` type and distance operators like `<=>` (cosine distance) |
| `vectorscale` (pgvectorscale) | the `diskann` index: fast approximate nearest neighbour search |
| `pg_textsearch` | the `bm25` index and the `<@>` relevance operator |

---

## Step 3 — Create the table

```sql
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

Each movie is stored **twice**, as text and as meaning:

- `search_text` holds the words. BM25 matches the query's exact terms against it.
- `embedding` holds 384 numbers that encode what the plot is *about*. Movies with
  similar plots get nearby vectors, even when they share no words.

`384` must match the embedding model. We use `BAAI/bge-small-en-v1.5`, a small
model that runs on a laptop CPU with no API key.

**Catch up (Steps 2–3):** `psql "$DATABASE_URL" -f workshop/schema.sql`

---

## Step 4 — Load and embed 200 movies

Exit `psql` (`\q`) or open a second terminal, then run:

```sh
uv run workshop/load.py
```

This script ([`load.py`](load.py), about 60 lines) does three things:

1. Reads `workshop/movies_sample.jsonl`, 200 movies (0.5 MB). It contains
   40 films that the demo queries rely on, plus 160 chosen at random with a
   fixed seed. [`make_sample.py`](make_sample.py) rebuilds the identical file
   from the original CMU archive.
2. Embeds each movie's text locally with `bge-small-en-v1.5`.
3. Inserts the rows, with the vectors, into `movies`.

✅ You should see: `Loaded 200 movies (200 with embeddings).` Embedding takes
about 10 seconds on a recent laptop.

Back in `psql`, look at what you stored:

```sql
SELECT title, release_year, genres, left(plot, 80) AS plot FROM movies LIMIT 5;
SELECT title, vector_dims(embedding), left(embedding::text, 60) FROM movies LIMIT 3;
```

> **Note:** In production, you'd generate embeddings once in a background job,
> or with an embeddings API. The full app supports both. Nothing here depends
> on the embedding model running in the database.

---

## Step 5 — Keyword search with BM25

Create the index:

```sql
CREATE INDEX movies_bm25_idx ON movies
    USING bm25 (search_text) WITH (text_config = 'english');
```

`text_config = 'english'` applies stemming ("racing" → "race") and drops stop
words. BM25 then scores documents by term frequency, how rare each term is, and
document length.

Search:

```sql
SELECT title, release_year,
       search_text <@> 'Indiana Jones' AS score
FROM movies
ORDER BY search_text <@> 'Indiana Jones'
LIMIT 5;
```

✅ You should see the four Indiana Jones films. Only 4 rows come back, even
with `LIMIT 5`, because BM25 returns only rows that contain a query term.

**Read the score:** `<@>` returns a **negative** number, and **more negative
means more relevant**. This lets a plain `ORDER BY ... ASC` put the best match
first, the same way distances sort.

Confirm that the index is being used. `ANALYZE` first refreshes the planner's
statistics, so it knows the table is worth an index scan:

```sql
ANALYZE movies;
EXPLAIN SELECT title FROM movies ORDER BY search_text <@> 'Indiana Jones' LIMIT 5;
```

✅ Look for `Index Scan using movies_bm25_idx`.

Now try a query that describes a plot rather than naming a film:

```sql
SELECT title, release_year
FROM movies
ORDER BY search_text <@> 'someone reliving the same day'
LIMIT 5;
```

🤔 *Groundhog Day* appears, but *50 First Dates* outranks it. BM25 matches words like "day"
and "same", and doesn't understand the concept. This is the gap that semantic
search fills.

> **Tip:** Inside functions and prepared statements, use the explicit form
> `search_text <@> to_bm25query('query', 'movies_bm25_idx')`. The scripts and
> the full app use this form.

---

## Step 6 — Semantic search with StreamingDiskANN

Create the index:

```sql
CREATE INDEX movies_embedding_idx ON movies
    USING diskann (embedding vector_cosine_ops);
```

`<=>` is **cosine distance**: `0` means identical direction, and smaller is
closer. A query has to be embedded with the same model as the documents. `psql`
can't do that, so for now let's use a vector that's already in the table:
**"find movies like *Groundhog Day*"**.

```sql
SELECT title, release_year,
       embedding <=> (SELECT embedding FROM movies WHERE title = 'Groundhog Day') AS distance
FROM movies
ORDER BY distance
LIMIT 6;
```

✅ You should see *Groundhog Day* at distance 0 (itself), then films such as
*Big*, *The Truman Show*, and *50 First Dates*. No keywords are shared; the
match comes from meaning.

```sql
EXPLAIN SELECT title FROM movies
ORDER BY embedding <=> (SELECT embedding FROM movies WHERE title = 'Groundhog Day')
LIMIT 5;
```

✅ Look for `Index Scan using movies_embedding_idx`.

**Why DiskANN?** It is a graph index that stores compressed vectors and keeps
search fast and accurate on data larger than RAM. With 200 rows any index is
fast. The difference matters for millions of rows.

**Catch up (Steps 5–6):** `psql "$DATABASE_URL" -f workshop/indexes.sql`

---

## Step 7 — Hybrid search with Reciprocal Rank Fusion

We now have two ranked lists with incompatible scores. BM25 returns `-15.8`;
cosine returns `0.31`. Adding them together would be meaningless.

**RRF ignores the scores and uses only the ranks:**

```
rrf_score(movie) = 1 / (60 + rank_in_keyword) + 1 / (60 + rank_in_semantic)
```

A movie ranked well in **both** lists beats one ranked first in only one list.
A movie missing from a list contributes 0 for that list. `k = 60` is the
standard constant: it dampens the advantage of ranking first.

To keep this in `psql`, use *Groundhog Day*'s vector as the semantic input,
paired with a keyword query:

```sql
WITH keyword AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY score) AS rank
    FROM (
        SELECT id, search_text <@> 'reliving the same day' AS score
        FROM movies ORDER BY score LIMIT 50
    ) k
), semantic AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY distance) AS rank
    FROM (
        SELECT id, embedding <=> (SELECT embedding FROM movies WHERE title = 'Groundhog Day') AS distance
        FROM movies ORDER BY distance LIMIT 50
    ) s
)
SELECT m.title,
       k.rank AS keyword_rank,
       s.rank AS semantic_rank,
       round(COALESCE(1.0 / (60 + k.rank), 0) + COALESCE(1.0 / (60 + s.rank), 0), 4) AS rrf_score
FROM keyword k
FULL OUTER JOIN semantic s USING (id)
JOIN movies m ON m.id = COALESCE(k.id, s.id)
ORDER BY rrf_score DESC
LIMIT 10;
```

How to read it:

1. **Two CTEs, two retrievals.** Each one uses its own index and fetches the
   top 50 candidates, then numbers them 1, 2, 3, …
2. **`FULL OUTER JOIN`** keeps movies found by only one method.
3. **`COALESCE(..., 0)`** gives a missing rank no credit.
4. The `keyword_rank` and `semantic_rank` columns show *why* each movie landed
   where it did.

✅ *Groundhog Day* is now **#1**. It was 2nd by keyword and 1st by meaning,
and that beats *50 First Dates* (1st and 5th). *The Truman Show* climbs to
3rd: it ranked only 11th by keyword, but 3rd by meaning. Rows where both rank
columns are filled in are the movies both methods agree on.

This is one SQL statement in one database. There is no separate search engine
to keep in sync.

---

## Step 8 — Compare all three with real text queries

[`ask.py`](ask.py) embeds your query text with the same model, then runs the
three queries from Steps 5–7 and prints the results side by side with timings.

```sh
uv run workshop/ask.py "Indiana Jones"
uv run workshop/ask.py "an archaeologist racing to recover an ancient artifact"
uv run workshop/ask.py "Indiana Jones supernatural artifact"
uv run workshop/ask.py "someone reliving the same day"
```

What to look for:

| Query | Lesson |
|---|---|
| `Indiana Jones` | An exact name. Keyword search succeeds; semantic search also finds the films, but returns looser matches further down. |
| `an archaeologist racing to recover…` | A description. Semantic search finds *Romancing the Stone* and *King Solomon's Mines*, although they share few words with the query. |
| `Indiana Jones supernatural artifact` | A name plus a theme. Both lists agree on the films, and hybrid keeps them at the top. |
| `someone reliving the same day` | Keyword search puts *50 First Dates* first. Hybrid puts *Groundhog Day* first, because both lists rank it near the top. |

> **Honest note:** Hybrid search does not win every query. In the
> "archaeologist" query on this sample, it pulls in some odd keyword-only
> matches. Hybrid search improves the *typical* result and reduces the worst
> misses. It doesn't guarantee a perfect ranking.

Now try your own queries. Plot descriptions work best: *"a robot falls in love"*,
*"heist goes wrong"*, *"shark terrorizes a beach town"*.

---

## Step 9 — Presenter: the same foundation at full scale

The presenter switches to the full Django app in this repository, already
connected to a preloaded database. Compared with this workshop, it adds:

| Workshop | Full app |
|---|---|
| 200 movies | 10,000 movies (the loader can load all 42,306) |
| One vector per movie (truncated at the model limit) | Long plots split into chunks, embedded, and pooled |
| Local `bge-small` (384 dims) | OpenRouter `text-embedding-3-small` (1,536 dims), with FastEmbed as a fallback |
| Hand-written SQL | The same SQL in [`src/movies/search.py`](../src/movies/search.py), plus genre and year filters |
| Terminal output | A side-by-side web UI with ranks, snippets, and timing |

The core is the same: one table, a `bm25` index, a `diskann` index, and a query
that fuses ranks.

To run the full app yourself after the webinar, follow the [repository README](../readme.md).
Loading and embedding the whole corpus takes a while, so start it before you
need it.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `extension "pg_textsearch" is not available` | Your PostgreSQL doesn't include it. Use one of the two options in Step 0. |
| `pg_textsearch library not loaded` (or `access method "bm25" does not exist`, or `operator does not exist: text <@> unknown`) | The extension library isn't preloaded. Docker: use the image and flags from Step 0. Self-hosted: add `pg_textsearch` to `shared_preload_libraries` in `postgresql.conf` and **restart** the server (a reload isn't enough). |
| `relation "movies" does not exist` | Run the Catch up for Steps 2–3. |
| `no BM25 index found for text <@> text expression`, `index "movies_bm25_idx" does not exist`, or `EXPLAIN` shows `Seq Scan` | Create the BM25 index first (Step 5), or run `psql "$DATABASE_URL" -f workshop/indexes.sql`. |
| `different vector dimensions 384 and 1536` or `expected 384 dimensions, not 1536` | That vector came from a different model (the full app uses 1,536 dimensions). Workshop vectors must come from `bge-small-en-v1.5`: use `ask.py`, or a vector already in the table. |
| `load.py` is slow | It's CPU-bound. Older laptops may take up to a minute. |
| `uv` can't find a matching Python | Run `uv python install 3.13`. The scripts need Python 3.10–3.13. |

**Start over:** `DROP TABLE movies;` then go back to Step 3.

---

*Dataset: [CMU Movie Summary Corpus](https://www.cs.cmu.edu/~ark/personas/) by
David Bamman, Brendan O'Connor, and Noah Smith, licensed under
[CC BY-SA 3.0 US](https://creativecommons.org/licenses/by-sa/3.0/us/).
`movies_sample.jsonl` is a 200-movie subset of it. The approach follows the
[Tiger Data hybrid search tutorial](https://www.tigerdata.com/docs/learn/tutorials/hybrid-search).*
