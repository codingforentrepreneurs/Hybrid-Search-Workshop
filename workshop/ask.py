# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["fastembed==0.9.0", "psycopg[binary]>=3.2,<4"]
# ///
"""Compare keyword, semantic, and hybrid results for one query.

    uv run workshop/ask.py "someone reliving the same day"
"""
import json
import os
import sys
import time
from pathlib import Path

import psycopg
from fastembed import TextEmbedding

MODEL = "BAAI/bge-small-en-v1.5"

KEYWORD = """
    SELECT id, title, release_year
    FROM movies
    ORDER BY search_text <@> to_bm25query(%(q)s, 'movies_bm25_idx')
    LIMIT 10
"""

SEMANTIC = """
    SELECT id, title, release_year
    FROM movies
    ORDER BY embedding <=> %(v)s::vector
    LIMIT 10
"""

# Reciprocal Rank Fusion: score = sum of 1 / (k + rank) across both lists.
HYBRID = """
    WITH keyword AS (
        SELECT id, ROW_NUMBER() OVER (ORDER BY score) AS rank
        FROM (
            SELECT id, search_text <@> to_bm25query(%(q)s, 'movies_bm25_idx') AS score
            FROM movies ORDER BY score LIMIT 50
        ) k
    ), semantic AS (
        SELECT id, ROW_NUMBER() OVER (ORDER BY distance) AS rank
        FROM (
            SELECT id, embedding <=> %(v)s::vector AS distance
            FROM movies ORDER BY distance LIMIT 50
        ) s
    )
    SELECT m.id, m.title, m.release_year,
           COALESCE(1.0 / (60 + k.rank), 0) + COALESCE(1.0 / (60 + s.rank), 0) AS rrf_score
    FROM keyword k
    FULL OUTER JOIN semantic s USING (id)
    JOIN movies m ON m.id = COALESCE(k.id, s.id)
    ORDER BY rrf_score DESC
    LIMIT 10
"""


def run(cur, sql, params):
    started = time.perf_counter()
    cur.execute(sql, params)
    rows = cur.fetchall()
    return rows, (time.perf_counter() - started) * 1000


def main():
    url = os.environ.get("DATABASE_URL")
    if not url or len(sys.argv) < 2:
        sys.exit('Usage: DATABASE_URL=... uv run workshop/ask.py "your query"')
    query = " ".join(sys.argv[1:])

    model = TextEmbedding(MODEL, cache_dir=str(Path(__file__).parent / ".models"))
    started = time.perf_counter()
    vector = next(iter(model.query_embed([query]))).tolist()
    embed_ms = (time.perf_counter() - started) * 1000
    params = {"q": query, "v": json.dumps(vector)}

    with psycopg.connect(url) as conn, conn.cursor() as cur:
        columns = [(name, *run(cur, sql, params)) for name, sql in
                   (("KEYWORD (BM25)", KEYWORD), ("SEMANTIC (DiskANN)", SEMANTIC), ("HYBRID (RRF)", HYBRID))]

    print(f'\nQuery: "{query}"   (query embedding: {embed_ms:.0f} ms)\n')
    for name, rows, ms in columns:
        print(f"{name}  ·  {ms:.1f} ms")
        for rank, row in enumerate(rows, 1):
            print(f"  {rank:>2}. {row[1]} ({row[2] or '?'})")
        print()


if __name__ == "__main__":
    main()
