"""Download, join, and import movie plots without generating paid embeddings."""

from collections import Counter
import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import connection, transaction

from movies.corpus import CorpusError, download_archive, iter_movies, write_jsonl


class Command(BaseCommand):
    help = "Load CMU movie plots, or prepare JSONL without a database or API calls."
    requires_system_checks = []

    def add_arguments(self, parser):
        parser.add_argument("--archive", type=Path, help="Use a local MovieSummaries.tar.gz instead of downloading.")
        parser.add_argument("--cache-dir", type=Path, default=Path("data"))
        parser.add_argument("--limit", type=int, default=10_000, help="Maximum valid movies (default 10000; 0 for all).")
        parser.add_argument("--batch-size", type=int, default=500)
        parser.add_argument("--prepare-only", action="store_true", help="Write JSONL; do not connect to a database.")
        parser.add_argument("--output", type=Path, default=Path("data/movies.jsonl"))

    def handle(self, *args, **options):
        if options["limit"] < 0 or options["batch_size"] <= 0:
            raise CommandError("--limit must be zero or positive and --batch-size must be positive.")
        if not options["prepare_only"] and connection.vendor != "postgresql":
            raise CommandError("Set DATABASE_URL to a PostgreSQL database, or use --prepare-only.")
        stats = Counter()
        try:
            archive = options["archive"] or download_archive(options["cache_dir"])
            movies = iter_movies(archive, limit=options["limit"], stats=stats)
            if options["prepare_only"]:
                count = write_jsonl(movies, options["output"])
                self.stdout.write(self.style.SUCCESS(f"Prepared {count:,} movies in {options['output']}. No API calls made."))
            else:
                batch = []
                for movie in movies:
                    batch.append(movie)
                    if len(batch) >= options["batch_size"]:
                        self._save_batch(batch, stats)
                        batch.clear()
                        self.stdout.write(f"Loaded {stats['saved']:,} movies...")
                if batch:
                    self._save_batch(batch, stats)
                self.stdout.write(self.style.SUCCESS(
                    f"Loaded {stats['saved']:,} movies ({stats['created']:,} new, {stats['updated']:,} existing). "
                    "Run embed_movies separately to generate embeddings."
                ))
        except (CorpusError, OSError) as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write("Source statistics: " + json.dumps(dict(stats), sort_keys=True))

    def _save_batch(self, rows, stats):
        from movies.models import Movie

        ids = [row["wikipedia_id"] for row in rows]
        existing = dict(Movie.objects.filter(pk__in=ids).values_list("pk", "search_text"))
        changed, unchanged = [], []
        for row in rows:
            movie = Movie(**row)
            previous = existing.get(movie.pk)
            if previous is None or previous != movie.search_text:
                movie.embedding = None
                movie.embedding_model = ""
                movie.embedding_hash = ""
                changed.append(movie)
            else:
                unchanged.append(movie)
        fields = [name for name in rows[0] if name != "wikipedia_id"]
        with transaction.atomic():
            if changed:
                Movie.objects.bulk_create(
                    changed, update_conflicts=True, unique_fields=["wikipedia_id"],
                    update_fields=fields + ["embedding", "embedding_model", "embedding_hash"],
                )
            if unchanged:
                Movie.objects.bulk_create(
                    unchanged, update_conflicts=True, unique_fields=["wikipedia_id"], update_fields=fields,
                )
        stats["saved"] += len(rows)
        stats["created"] += len(rows) - len(existing)
        stats["updated"] += len(existing)
