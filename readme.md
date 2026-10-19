# Movie hybrid search demo

A Django web app for comparing keyword, semantic, and hybrid search over movie
plots, with PostgreSQL on Tiger Cloud. This repository also provides shared
context for a separate session preparing webinar notes.

## Agreed direction

- **Web app:** Django, managed with `uv`.
- **Database:** Tiger Cloud PostgreSQL with `pg_textsearch`, `pgvector`, and
  `pgvectorscale`.
- **Keyword retrieval:** BM25 through `pg_textsearch`.
- **Semantic retrieval:** embeddings generated through OpenRouter, or locally
  with FastEmbed when the OpenRouter key is blank; stored in PostgreSQL and
  searched with a StreamingDiskANN index.
- **Hybrid ranking:** Reciprocal Rank Fusion (RRF) combines the two ranked lists.
- **Dataset:** CMU Movie Summary Corpus.
- **Demo experience:** compare BM25, semantic, and hybrid results side by side,
  with plot snippets, ranks, and timing.

The implementation follows the approach in the
[Tiger Data hybrid search tutorial](https://www.tigerdata.com/docs/learn/tutorials/hybrid-search).

## Dataset

[Download MovieSummaries.tar.gz](https://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz)
(approximately 46 MB; no signup or API key).

- [Dataset page](https://www.cs.cmu.edu/~ark/personas/)
- [Column descriptions and source README](https://www.cs.cmu.edu/~ark/personas/data/README.txt)

The corpus contains 42,306 Wikipedia plot summaries and movie metadata including
title, release date, genre, language, and runtime. Plot summaries join to metadata
by Wikipedia movie ID. Some metadata records do not have a corresponding plot.
The source snapshots date to 2012, so this is a historical movie collection.

Start with up to 10,000 movies and support loading the full corpus. Preserve
original plots and source IDs, and keep downloaded data out of version control.

Credit: David Bamman, Brendan O'Connor, and Noah Smith, *CMU Movie Summary Corpus*.
The dataset page links to
[Creative Commons Attribution-ShareAlike 3.0 US](https://creativecommons.org/licenses/by-sa/3.0/us/).
Preserve dataset attribution and source links in the app and derived data.

## Project layout

```text
pyproject.toml / uv.lock   Python dependencies managed by uv
rav.yaml                  Command shortcuts (rav run <command>)
.env.sample               Configuration template (copy to .env)
.env                      Local configuration and secrets; ignored by Git
readme.md                 Project context and setup
scripts/load_movies.py    Convenience loader entrypoint
src/manage.py             Django command entrypoint
src/config/               Django settings and URLs
src/movies/               Models, search, ingestion, templates, static assets
data/                     Downloaded corpus and local models; ignored by Git
```

Settings explicitly load the repository-root `.env` using `python-dotenv`.
Existing environment variables take precedence. The sample file documents
configuration; it is not loaded as credentials. Commands below run from the
repository root.

## Quick start

```sh
uv sync
cp .env.sample .env
uv run python src/manage.py runserver
```

Open `http://127.0.0.1:8000/`. Without `DATABASE_URL`, the app displays its setup
screen. SQLite is used only for offline checks; actual search requires the
Tiger Cloud PostgreSQL extensions.

The PostgreSQL driver uses the system `libpq` library. Install it if needed
(`brew install libpq` on macOS or your system's libpq package), or use
`uv add 'psycopg[binary]'` for the bundled driver.

To enable local FastEmbed inference, install its optional runtime once:

```sh
uv add fastembed==0.9.0
```

This records FastEmbed in the uv project and updates its lockfile. Its first use
also downloads the model into `data/models/`. A supplied OpenRouter key uses the
remote provider by default and does not require FastEmbed.

## Command shortcuts

[Rav](https://github.com/jmitchel3/rav) shortcuts are defined in `rav.yaml`.
Install the CLI with `uv tool install rav` if needed, then run commands from the
repository root:

```sh
rav list                       # Show all commands
rav run setup                  # Sync dependencies; create .env if missing
rav run install-fastembed      # Optional local embedding runtime
rav run server                 # Start Django
rav run prepare                # Download and prepare movies as JSONL
rav run migrate                # Apply database migrations
rav run load-sample             # Import up to 1,000 movies
rav run embed-plan             # Count outstanding embeddings
rav run embed                  # Embed with 4 concurrent OpenRouter requests
rav run embed-sample            # Embed up to 100 outstanding movies
rav run indexes                # Build BM25 and DiskANN indexes
rav run verify                 # System checks, migration checks, and tests
```

Use `load` for the default 10,000 movies, `load-all` for the full corpus, and
`embed` for all outstanding embeddings. For custom flags such as `--archive`,
use the direct `uv` commands below or edit `rav.yaml`: the locally installed
Rav 0.4.3 consumes command-line keyword options instead of forwarding them.
`setup` preserves an existing `.env`. Both `test` and `verify` use the isolated
SQLite test settings.

## Load movies and enable search

Inspect the dataset before connecting to a database:

```sh
uv run scripts/load_movies.py --prepare-only --limit 10000
```

This downloads and caches the CMU archive, joins titles/metadata to plots, and
writes `data/movies.jsonl`. It does not generate embeddings. If the download is
unavailable, supply a local archive:

```sh
uv run scripts/load_movies.py --prepare-only --archive /path/to/MovieSummaries.tar.gz
```

Set `DATABASE_URL` in the root `.env` to your Tiger Cloud connection string,
then run:

```sh
uv run python src/manage.py migrate
uv run scripts/load_movies.py --limit 10000
uv run python src/manage.py embed_movies --dry-run
uv run python src/manage.py embed_movies
uv run python src/manage.py setup_search_indexes
uv run python src/manage.py runserver
```

Use `--limit 0` on the loader for the full corpus. Imports upsert by Wikipedia ID;
unchanged search text retains its embeddings. Embedding batches are saved as they
finish, so rerunning `embed_movies` resumes work. `--limit 100` on that command
is useful for an initial small embedding run. OpenRouter requests use account
credits; the loader does not initiate them.

Embedding imports process 128 movies per commit, send up to 32 text chunks per
OpenRouter request, and run up to four requests concurrently. Database writes are
batched, and progress reports movies per second plus embedding and save times.
The provider, model, and preprocessing are unchanged, so completed embeddings
remain valid. To pick up these settings during an older run, stop it with Ctrl-C
and restart `rav run embed`; previously committed batches are skipped. A request
interrupted before its batch is committed may be repeated.

For custom throughput settings, use the management command directly:

```sh
uv run python src/manage.py embed_movies --batch-size 128 --workers 4 --request-batch-size 32
```

If OpenRouter rate limits the run, use fewer workers (or `rav run embed-serial`).
Concurrency affects OpenRouter requests only; FastEmbed retains its local
inference batching. Actual throughput depends on provider limits, plot lengths,
and database latency; no fixed speedup is guaranteed.

## Embedding providers

Use the [OpenRouter embeddings API](https://openrouter.ai/docs/api/api-reference/embeddings/create-embeddings)
at `https://openrouter.ai/api/v1/embeddings`.

With `EMBEDDING_PROVIDER=auto`, a nonblank `OPENROUTER_API_KEY` selects OpenRouter's
`openai/text-embedding-3-small` at 1,536 dimensions. A blank key selects FastEmbed's
`BAAI/bge-small-en-v1.5` at 384 dimensions. Set `EMBEDDING_PROVIDER=openrouter` or
`fastembed` to choose explicitly. A failed OpenRouter request does not silently
switch providers.

The database records the provider/model identity with each embedding, and queries
only vectors from the active model and dimension. Switching providers requires
rerunning `embed_movies` and `setup_search_indexes`. One active embedding is
stored per movie, so switching back also requires re-embedding. The base vector
column supports different dimensions; each model gets a matching typed column
and DiskANN index during setup.

Long text is split into bounded UTF-8 chunks, embedded, then combined with a
byte-weighted mean and L2 normalization. Original plots are retained in full.
This simple demo uses one pooled vector per movie, rather than ranking individual
plot chunks. The preprocessing version is part of the resume hash.

Configure credentials through environment variables or an ignored `.env` file.
Never commit API keys or database connection strings. The loader should cache
completed embedding work so repeated runs can resume without repeating all calls.

## Search behavior

1. Accept a query and optional movie filters.
2. Retrieve keyword candidates using BM25.
3. Embed the query through OpenRouter and retrieve vector candidates.
4. Fuse ranks with RRF, initially using `k = 60`.
5. Display the individual result lists and the combined ranking.

BM25 and vector scores use different scales. RRF combines ranks, rather than
adding their raw scores. Higher RRF scores rank first; lower BM25 operator scores
and cosine distances rank first. Apply the same filters to both retrieval paths.

## Webinar notes handoff

The intended story is why exact terms and semantic meaning complement each
other, and how PostgreSQL can perform both retrieval steps in one database.

Candidate queries to investigate:

- `Indiana Jones` — names and exact terms.
- `an archaeologist racing to recover an ancient artifact` — descriptive intent.
- `Indiana Jones supernatural artifact` — names combined with a theme.
- `someone reliving the same day` — a plot description without a movie title.

These are proposed examples, not validated results. Record actual rankings and
latencies after loading the data. Hybrid search is not guaranteed to win every
query. Distinguish database query time from embedding API time in demonstrations.

Webinar notes can be developed independently in `webinar-notes.md`. Keep claims
about completed features, search quality, corpus size, and performance aligned
with the implementation status below.

## Implementation status

- Django UI, movie loader, OpenRouter/FastEmbed integration, search SQL, and
  index setup commands are implemented under `src/`.
- Python dependencies are managed by `uv`; Python 3.14 is selected. The core
  dependencies have a generated lockfile and are installed locally.
- All 44 offline tests pass, along with Django system checks and migration
  consistency checks. SQL tests check construction and isolation, not live
  PostgreSQL execution.
- Live OpenRouter embedding generation has been started by the user. The faster
  concurrent loader passes offline tests, but its live throughput and end-to-end
  Tiger Cloud search have not yet been verified in this session.
- Local archive input is supported; FastEmbed's package and model download
  remain unverified in this session.
- HTML rendering and JavaScript syntax checks passed. Browser visual QA and a
  listening development server could not run under this session's restrictions.

## Checks

```sh
uv run python src/manage.py check
uv run python src/manage.py makemigrations --check --dry-run
uv run python src/manage.py test movies --settings=config.test_settings
```

Tests use local fixtures and mocked embedding responses. They do not call paid
APIs, download models, or modify Tiger Cloud. Live search verification additionally
requires the database connection, installed extensions, imported movies, and
generated embeddings.
