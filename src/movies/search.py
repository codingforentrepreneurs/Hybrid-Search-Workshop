"""Real PostgreSQL retrieval: BM25, StreamingDiskANN, and SQL rank fusion.

Candidate retrieval is bounded before assigning ranks. Materializing semantic
candidates also lets us correct DiskANN's relaxed ordering before rank fusion.
Scores are intentionally kept on their own scales: lower BM25/cosine values
rank first, while higher reciprocal rank fusion values rank first.
"""
import hashlib
import json
import math
from time import perf_counter

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import connection

from .models import Movie


class SearchUnavailable(Exception):
    """An actionable search setup issue safe to show in the interface."""


def empty_columns():
    return [
        {"key": "bm25", "label": "Keyword", "description": "BM25 · pg_textsearch", "results": [], "elapsed_ms": None},
        {"key": "semantic", "label": "Semantic", "description": "Cosine similarity · pgvector", "results": [], "elapsed_ms": None},
        {"key": "hybrid", "label": "Hybrid", "description": "Reciprocal Rank Fusion · k = 60", "results": [], "elapsed_ms": None},
    ]


def vector_column_name(config=None):
    """Name the physical vector column used by DiskANN for this model.

    DiskANN rescoring reads a heap attribute, so use a stored generated column
    rather than a cast expression index over the variable-dimension column.
    """
    from .embeddings import get_embedding_config

    config = config or get_embedding_config()
    identity = f"stored-v1:{config.model_id}:{config.dimensions}".encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()[:12]
    return f"embedding_{digest}"


def vector_index_name(config=None):
    return f"movies_{vector_column_name(config)}_diskann_idx"


def check_search_ready():
    if connection.vendor != "postgresql":
        raise SearchUnavailable("Connect your Tiger Cloud database with DATABASE_URL to enable search.")
    vector_index = vector_index_name()
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'vectorscale', 'pg_textsearch')"
        )
        installed = {row[0] for row in cursor.fetchall()}
        if installed != {"vector", "vectorscale", "pg_textsearch"}:
            raise SearchUnavailable("Run uv run python src/manage.py migrate to enable the PostgreSQL search extensions.")
        cursor.execute(
            """SELECT c.relname FROM pg_class c
               JOIN pg_index i ON i.indexrelid = c.oid
               JOIN pg_am am ON am.oid = c.relam
               WHERE i.indrelid = to_regclass(%s) AND i.indisvalid
                 AND (
                   (c.relname = %s AND am.amname = 'bm25')
                   OR (c.relname = %s AND am.amname = 'diskann'
                       AND i.indexprs IS NULL AND EXISTS (
                         SELECT 1 FROM pg_attribute a
                         WHERE a.attrelid = i.indrelid AND a.attnum = i.indkey[0]
                           AND a.attname = %s AND a.attgenerated = 's'
                           AND NOT a.attisdropped
                       ))
                 )""",
            [Movie._meta.db_table, settings.SEARCH_BM25_INDEX, vector_index, vector_column_name()],
        )
        ready_indexes = {row[0] for row in cursor.fetchall()}
        if ready_indexes != {settings.SEARCH_BM25_INDEX, vector_index}:
            raise SearchUnavailable(
                "Run uv run python src/manage.py setup_search_indexes after loading movies to enable both search indexes."
            )


def _filter_sql(*, genre, year):
    clauses = []
    values = []
    if genre:
        clauses.append("genres @> %s::jsonb")
        values.append(json.dumps([genre]))
    if year is not None:
        clauses.append("release_year = %s")
        values.append(year)
    return clauses, values


def _query_rows(sql, params):
    started = perf_counter()
    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        names = [field[0] for field in cursor.description]
        rows = [dict(zip(names, values)) for values in cursor.fetchall()]
    for row in rows:
        # psycopg's raw cursor returns Django JSONFields as serialized JSON.
        if isinstance(row.get("genres"), str):
            row["genres"] = json.loads(row["genres"])
        if "score" in row:
            row["score"] = float(row["score"])
        row.setdefault("bm25_rank", None)
        row.setdefault("semantic_rank", None)
    return rows, round((perf_counter() - started) * 1000, 2)


def retrieve_bm25(query, *, genre="", year=None, candidate_limit=None):
    limit = candidate_limit or settings.SEARCH_CANDIDATE_LIMIT
    clauses, params = _filter_sql(genre=genre, year=year)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    table = connection.ops.quote_name(Movie._meta.db_table)
    # The inner ORDER BY retains the indexable operator and LIMIT. Score ties
    # within the retrieved candidate set are broken by the source ID.
    sql = f"""
        WITH candidates AS MATERIALIZED (
            SELECT wikipedia_id, title, release_year, genres, plot, wikipedia_url,
                   search_text <@> to_bm25query(%s, %s) AS score
            FROM {table}{where}
            ORDER BY score
            LIMIT %s
        )
        SELECT *, ROW_NUMBER() OVER (ORDER BY score, wikipedia_id) AS rank
        FROM candidates WHERE score < 0
        ORDER BY score, wikipedia_id
    """
    return _query_rows(sql, [query, settings.SEARCH_BM25_INDEX, *params, limit])


def retrieve_semantic(embedding, *, genre="", year=None, candidate_limit=None):
    from .embeddings import get_embedding_config

    config = get_embedding_config()
    dimensions = int(config.dimensions)
    if not 1 <= dimensions <= 16000:
        raise SearchUnavailable("The configured embedding dimensions are outside pgvector's supported range.")
    limit = candidate_limit or settings.SEARCH_CANDIDATE_LIMIT
    try:
        vector = Movie._meta.get_field("embedding").to_python(embedding)
    except ValidationError:
        raise SearchUnavailable("The embedding service returned an unusable query vector. Try the query again.") from None
    if vector is None or len(vector) != dimensions or not any(value != 0 for value in vector):
        raise SearchUnavailable("The embedding service returned an unusable query vector. Try the query again.")
    clauses, params = _filter_sql(genre=genre, year=year)
    vector_column = connection.ops.quote_name(vector_column_name(config))
    # The generated column is NULL for every other model or dimension count.
    clauses.append(f"{vector_column} IS NOT NULL")
    where = " WHERE " + " AND ".join(clauses)
    table = connection.ops.quote_name(Movie._meta.db_table)
    statement = f"""
        WITH candidates AS MATERIALIZED (
            SELECT wikipedia_id, title, release_year, genres, plot, wikipedia_url,
                   {vector_column} <=> %s::vector({dimensions}) AS score
            FROM {table}{where}
            ORDER BY score
            LIMIT %s
        )
        SELECT *, ROW_NUMBER() OVER (ORDER BY score, wikipedia_id) AS rank
        FROM candidates
        ORDER BY score, wikipedia_id
    """
    return _query_rows(statement, [json.dumps(vector, allow_nan=False), *params, limit])


def fuse_rankings(bm25, semantic, *, result_limit=None, k=None):
    """Fuse only the bounded candidates, preserving absent ranks as null.

    Passing the already retrieved ranks into PostgreSQL avoids repeating either
    retrieval just to show three result lists and provides a separate RRF timer.
    """
    limit = result_limit or settings.SEARCH_RESULT_LIMIT
    smoothing = settings.SEARCH_RRF_K if k is None else k
    if not math.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("RRF k must be positive and finite.")
    if not bm25 and not semantic:
        return [], 0.0
    candidates = []
    for method, rows in (("bm25", bm25), ("semantic", semantic)):
        for row in rows:
            candidates.append({"wikipedia_id": row["wikipedia_id"], "rank": row["rank"], "method": method})
    table = connection.ops.quote_name(Movie._meta.db_table)
    sql = f"""
        WITH candidates AS (
            SELECT * FROM jsonb_to_recordset(%s::jsonb)
                AS c(wikipedia_id bigint, rank bigint, method text)
        ), scores AS (
            SELECT wikipedia_id,
                   SUM(1.0 / (%s + rank)) AS score,
                   MAX(CASE WHEN method = 'bm25' THEN rank END) AS bm25_rank,
                   MAX(CASE WHEN method = 'semantic' THEN rank END) AS semantic_rank
            FROM candidates
            GROUP BY wikipedia_id
        )
        SELECT m.wikipedia_id, m.title, m.release_year, m.genres, m.plot, m.wikipedia_url,
               s.score, s.bm25_rank, s.semantic_rank,
               ROW_NUMBER() OVER (ORDER BY s.score DESC, m.wikipedia_id) AS rank
        FROM scores s JOIN {table} m ON m.wikipedia_id = s.wikipedia_id
        ORDER BY s.score DESC, m.wikipedia_id
        LIMIT %s
    """
    return _query_rows(sql, [json.dumps(candidates), smoothing, limit])


def search_movies(query, *, genre="", year=None):
    """Return all three result lists and honest embedding/database timings."""
    from .embeddings import embed_texts

    check_search_ready()
    started = perf_counter()
    embedding_started = perf_counter()
    vectors = embed_texts([query], input_type="query")
    embedding_ms = round((perf_counter() - embedding_started) * 1000, 2)
    if len(vectors) != 1:
        raise SearchUnavailable("The embedding service returned an unexpected response. Try the query again.")
    bm25, bm25_ms = retrieve_bm25(query, genre=genre, year=year)
    semantic, semantic_ms = retrieve_semantic(vectors[0], genre=genre, year=year)
    hybrid, fusion_ms = fuse_rankings(bm25, semantic)
    bm25_ranks = {row["wikipedia_id"]: row["rank"] for row in bm25}
    semantic_ranks = {row["wikipedia_id"]: row["rank"] for row in semantic}
    for row in bm25 + semantic:
        row["bm25_rank"] = bm25_ranks.get(row["wikipedia_id"])
        row["semantic_rank"] = semantic_ranks.get(row["wikipedia_id"])
    columns = empty_columns()
    for column, results, elapsed in zip(
        columns, [bm25, semantic, hybrid], [bm25_ms, semantic_ms, fusion_ms]
    ):
        column["results"] = results[:settings.SEARCH_RESULT_LIMIT]
        column["elapsed_ms"] = elapsed
    return {
        "columns": columns,
        "timings": {
            "embedding_ms": embedding_ms,
            "total_ms": round((perf_counter() - started) * 1000, 2),
            "database_ms": round(bm25_ms + semantic_ms + fusion_ms, 2),
        },
    }
