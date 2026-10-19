# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["fastembed==0.9.0", "psycopg[binary]>=3.2,<4"]
# ///
"""Embed the workshop sample locally and load it into the `movies` table.

    uv run workshop/load.py                  # embed + load
    uv run workshop/load.py --download-only  # before the webinar: fetch deps + model
"""
import json
import os
import sys
import time
from pathlib import Path

import psycopg
from fastembed import TextEmbedding

MODEL = "BAAI/bge-small-en-v1.5"  # 384 dimensions, runs on a laptop CPU
SAMPLE = Path(__file__).with_name("movies_sample.jsonl")


def search_text(movie):
    # Same shape the full app indexes: title, genres, then the plot.
    return "\n".join([movie["title"], ", ".join(movie["genres"]), "", movie["plot"]])


def main():
    if "--download-only" in sys.argv:
        TextEmbedding(MODEL, cache_dir=str(Path(__file__).parent / ".models"))
        print("Model downloaded. You're ready for the workshop.")
        return
    url = os.environ.get("DATABASE_URL") or (sys.argv[1] if len(sys.argv) > 1 else "")
    if not url:
        sys.exit("Set DATABASE_URL (or pass it as the first argument).")
    movies = [json.loads(line) for line in SAMPLE.read_text(encoding="utf-8").splitlines()]
    texts = [search_text(movie) for movie in movies]

    print(f"Embedding {len(movies):,} movies with {MODEL} ...")
    started = time.perf_counter()
    model = TextEmbedding(MODEL, cache_dir=str(Path(__file__).parent / ".models"))
    vectors = [vector.tolist() for vector in model.passage_embed(texts, batch_size=64)]
    print(f"Embedded in {time.perf_counter() - started:.1f}s")

    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO movies (id, title, release_year, genres, plot, search_text, embedding)
               VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
               ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding""",
            [
                (m["wikipedia_id"], m["title"], m["release_year"], m["genres"], m["plot"], text, json.dumps(vector))
                for m, text, vector in zip(movies, texts, vectors)
            ],
        )
        cur.execute("SELECT count(*), count(embedding) FROM movies")
        total, embedded = cur.fetchone()
    print(f"Loaded {total:,} movies ({embedded:,} with embeddings).")


if __name__ == "__main__":
    main()
