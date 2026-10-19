"""Download and join the original CMU Movie Summary Corpus, without extracting it."""

from __future__ import annotations

import csv
import io
import json
import math
import os
from pathlib import Path
import re
import tarfile
import tempfile
from collections import Counter
from collections.abc import Iterator

import httpx

DATASET_URL = "https://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"
METADATA_MEMBER = "MovieSummaries/movie.metadata.tsv"
PLOTS_MEMBER = "MovieSummaries/plot_summaries.txt"
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024


class CorpusError(Exception):
    """An archive could not be downloaded or parsed."""


def _check_archive(path: Path) -> None:
    try:
        with tarfile.open(path, "r:gz") as archive:
            for name in (METADATA_MEMBER, PLOTS_MEMBER):
                member = archive.getmember(name)
                if not member.isfile():
                    raise CorpusError(f"Archive member is not a regular file: {name}")
    except (OSError, tarfile.TarError, KeyError) as exc:
        raise CorpusError("Expected a CMU MovieSummaries.tar.gz archive with metadata and plots.") from exc


def download_archive(cache_dir: Path, *, url: str = DATASET_URL) -> Path:
    """Reuse a checked download; replace the cache only after a complete download."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / "MovieSummaries.tar.gz"
    if destination.exists():
        _check_archive(destination)
        return destination
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            with httpx.stream("GET", url, follow_redirects=True, timeout=60) as response:
                response.raise_for_status()
                downloaded = 0
                for chunk in response.iter_bytes():
                    downloaded += len(chunk)
                    if downloaded > MAX_ARCHIVE_BYTES:
                        raise CorpusError("Download exceeds the 128 MiB archive limit.")
                    output.write(chunk)
        _check_archive(temporary)
        os.replace(temporary, destination)
    except httpx.HTTPError as exc:
        raise CorpusError(
            f"Could not download the dataset. Download {DATASET_URL} in a browser, "
            "then pass --archive /path/to/MovieSummaries.tar.gz."
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def _labels(raw: str, stats: Counter) -> list[str]:
    if not raw.strip():
        return []
    try:
        values = json.loads(raw)
        if not isinstance(values, dict) or not all(isinstance(value, str) for value in values.values()):
            raise ValueError
        return sorted(set(value.strip() for value in values.values() if value.strip()))
    except (ValueError, TypeError):
        stats["malformed_metadata_fields"] += 1
        return []


def _member_text(archive: tarfile.TarFile, name: str) -> io.TextIOWrapper:
    try:
        member = archive.getmember(name)
        if not member.isfile():
            raise CorpusError(f"Archive member is not a regular file: {name}")
        handle = archive.extractfile(member)
        if handle is None:
            raise CorpusError(f"Archive member could not be read: {name}")
        return io.TextIOWrapper(handle, encoding="utf-8")
    except KeyError as exc:
        raise CorpusError(f"Archive is missing {name}.") from exc


def iter_movies(path: Path, *, limit: int = 10_000, stats: Counter | None = None) -> Iterator[dict]:
    """Yield joined movie dictionaries in source plot order (0 means no limit).

    Counts refer to scanned source rows; a limited run stops reading plots once
    enough valid movies have been produced. TSV quote handling is deliberately
    disabled because the metadata contains literal JSON and quoted movie titles.
    """
    if limit < 0:
        raise ValueError("limit must be zero or positive")
    stats = stats if stats is not None else Counter()
    metadata: dict[int, dict] = {}
    seen: set[int] = set()
    try:
        with tarfile.open(path, "r:gz") as archive:
            with _member_text(archive, METADATA_MEMBER) as source:
                for row in csv.reader(source, delimiter="\t", quoting=csv.QUOTE_NONE):
                    stats["metadata_rows"] += 1
                    if len(row) != 9:
                        stats["malformed_metadata_rows"] += 1
                        continue
                    try:
                        wikipedia_id = int(row[0])
                        if wikipedia_id <= 0 or not row[2].strip():
                            raise ValueError
                    except ValueError:
                        stats["malformed_metadata_rows"] += 1
                        continue
                    if wikipedia_id in metadata:
                        stats["duplicate_metadata_rows"] += 1
                        continue
                    release_date = row[3].strip()
                    match = re.match(r"^(\d{4})(?:-|$)", release_date)
                    release_year = int(match.group(1)) if match else None
                    runtime = None
                    if row[5].strip():
                        try:
                            runtime = float(row[5])
                            if not math.isfinite(runtime) or runtime <= 0:
                                raise ValueError
                        except ValueError:
                            stats["malformed_metadata_fields"] += 1
                            runtime = None
                    metadata[wikipedia_id] = {
                        "wikipedia_id": wikipedia_id,
                        "title": row[2].strip(),
                        "release_date": release_date,
                        "release_year": release_year,
                        "runtime_minutes": runtime,
                        "languages": _labels(row[6], stats),
                        "countries": _labels(row[7], stats),
                        "genres": _labels(row[8], stats),
                        "wikipedia_url": f"https://en.wikipedia.org/?curid={wikipedia_id}",
                    }
            with _member_text(archive, PLOTS_MEMBER) as source:
                for line in source:
                    stats["plot_rows"] += 1
                    parts = line.rstrip("\r\n").split("\t", 1)
                    try:
                        wikipedia_id = int(parts[0])
                        plot = parts[1].strip()
                        if wikipedia_id <= 0 or not plot:
                            raise ValueError
                    except (ValueError, IndexError):
                        stats["malformed_plot_rows"] += 1
                        continue
                    if wikipedia_id in seen:
                        stats["duplicate_plot_rows"] += 1
                        continue
                    if wikipedia_id not in metadata:
                        stats["plots_without_metadata"] += 1
                        continue
                    seen.add(wikipedia_id)
                    movie = dict(metadata[wikipedia_id], plot=plot)
                    movie["search_text"] = "\n".join(
                        [movie["title"], ", ".join(movie["genres"]), "", plot]
                    )
                    stats["movies"] += 1
                    yield movie
                    if limit and stats["movies"] >= limit:
                        stats["stopped_at_limit"] = 1
                        return
            stats["metadata_without_plot"] = len(metadata) - len(seen)
    except (OSError, tarfile.TarError, UnicodeError, csv.Error) as exc:
        raise CorpusError("Unable to read the CMU archive; check the download and UTF-8 source files.") from exc


def write_jsonl(movies: Iterator[dict], destination: Path) -> int:
    """Write an atomic, reusable UTF-8 preparation artifact."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    count = 0
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, mode="w", encoding="utf-8", suffix=".part", delete=False
        ) as output:
            temporary = Path(output.name)
            for movie in movies:
                output.write(json.dumps(movie, ensure_ascii=False) + "\n")
                count += 1
        os.replace(temporary, destination)
        return count
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
