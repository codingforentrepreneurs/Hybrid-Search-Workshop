"""Rebuild workshop/movies_sample.jsonl from the CMU archive (run from the repo root).

    uv run python workshop/make_sample.py
"""
import json, random, sys
from pathlib import Path
sys.path.insert(0, "src")
from movies.corpus import iter_movies

# Films the demo queries rely on; everything else is a seeded random fill.
PINNED = {54166, 81503, 51888, 1252914, 142327, 25920477, 477099, 253828, 699208, 49800, 443011,
          578001, 89631, 1524536, 9467480, 10862322, 8905156, 52371, 68485, 74830, 30007, 23270459,
          42993, 23487440, 30327, 97646, 1186616, 52549, 53964, 50744, 205013, 53085, 239587,
          232711, 2750041, 805053, 168597, 2236324, 2864621, 462764}
movies = list(iter_movies(Path("data/MovieSummaries.tar.gz"), limit=0))
pinned = [m for m in movies if m["wikipedia_id"] in PINNED]
rest = [m for m in movies if m["wikipedia_id"] not in PINNED and m["release_year"] and len(m["plot"]) > 400]
SIZE = 200
random.seed(42)
sample = pinned + random.sample(rest, SIZE - len(pinned))
sample.sort(key=lambda m: m["wikipedia_id"])
with open("workshop/movies_sample.jsonl", "w", encoding="utf-8") as out:
    for m in sample:
        out.write(json.dumps({k: m[k] for k in ("wikipedia_id", "title", "release_year", "genres", "plot")}, ensure_ascii=False) + "\n")
print(len(pinned), len(sample))
