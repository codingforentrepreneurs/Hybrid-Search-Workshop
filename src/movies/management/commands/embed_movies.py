"""Generate resumable embeddings; completed unchanged rows are skipped."""

from time import perf_counter

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import connection, transaction
from django.db.models import BooleanField, Case, Value, When
from django.db.models.functions import Cast

from movies.embeddings import EmbeddingError, embed_texts, get_embedding_config, text_hash


class Command(BaseCommand):
    help = "Generate movie embeddings through the active provider; reruns resume completed work. OpenRouter uses credits."

    def add_arguments(self, parser):
        parser.add_argument("--batch-size", type=int, default=128, help="Movies committed per batch (default 128).")
        parser.add_argument("--workers", type=int, default=4, help="Concurrent OpenRouter requests, 1–32 (default 4).")
        parser.add_argument(
            "--request-batch-size", type=int, default=32,
            help="Text chunks per OpenRouter request, 1–128 (default 32).",
        )
        parser.add_argument("--limit", type=int, default=0, help="Maximum movies to embed (0 for all outstanding).")
        parser.add_argument("--force", action="store_true", help="Re-embed even unchanged movies.")
        parser.add_argument("--dry-run", action="store_true", help="Count pending work without calling an embedding provider.")

    def handle(self, *args, **options):
        from movies.models import Movie

        if options["batch_size"] <= 0 or options["limit"] < 0:
            raise CommandError("--batch-size must be positive and --limit zero or positive.")
        if not 1 <= options["workers"] <= 32:
            raise CommandError("--workers must be between 1 and 32.")
        if not 1 <= options["request_batch_size"] <= 128:
            raise CommandError("--request-batch-size must be between 1 and 128.")
        if connection.vendor != "postgresql":
            raise CommandError("Set DATABASE_URL to the PostgreSQL database containing loaded movies.")
        try:
            config = get_embedding_config()
        except EmbeddingError as exc:
            raise CommandError(str(exc)) from exc
        field_dimensions = Movie._meta.get_field("embedding").dimensions
        if field_dimensions is not None and config.dimensions != field_dimensions:
            raise CommandError("Configured embedding dimensions differ from the Movie vector column. Migrate the schema first.")
        if not options["dry_run"] and config.provider == "openrouter" and not settings.OPENROUTER_API_KEY.strip():
            raise CommandError("Set OPENROUTER_API_KEY or select EMBEDDING_PROVIDER=auto for local FastEmbed.")
        model = config.model_id
        if not options["dry_run"]:
            self.stdout.write(
                f"Provider: {config.provider}; movie batch: {options['batch_size']}; "
                f"OpenRouter workers: {options['workers']}; request batch: {options['request_batch_size']} chunks."
            )
            if config.provider == "fastembed":
                self.stdout.write("FastEmbed runs locally; OpenRouter request controls do not affect local inference.")
        rows = (
            Movie.objects.order_by("wikipedia_id")
            .annotate(has_embedding=Case(
                When(embedding__isnull=False, then=Value(True)), default=Value(False), output_field=BooleanField(),
            ))
            .values("wikipedia_id", "search_text", "embedding_model", "embedding_hash", "has_embedding")
        )
        pending, skipped, completed, stale = 0, 0, 0, 0
        batch = []
        started = perf_counter()
        for row in rows.iterator(chunk_size=500):
            digest = text_hash(row["search_text"], model=model)
            if (not options["force"] and row["has_embedding"] and row["embedding_model"] == model
                    and row["embedding_hash"] == digest):
                skipped += 1
                continue
            pending += 1
            if not options["dry_run"]:
                batch.append((row, digest))
                if len(batch) >= options["batch_size"]:
                    saved = self._embed_batch(
                        batch, model, workers=options["workers"], request_batch_size=options["request_batch_size"],
                    )
                    completed += saved
                    stale += len(batch) - saved
                    batch.clear()
                    self._report_progress(completed, skipped, started)
            if options["limit"] and pending >= options["limit"]:
                break
        if batch:
            saved = self._embed_batch(
                batch, model, workers=options["workers"], request_batch_size=options["request_batch_size"],
            )
            completed += saved
            stale += len(batch) - saved
            self._report_progress(completed, skipped, started)
        if options["dry_run"]:
            self.stdout.write(f"{pending:,} movies need embeddings; {skipped:,} already current. Model: {model}. No API calls made.")
        else:
            self.stdout.write(self.style.SUCCESS(
                f"Embedded {completed:,} movies; {skipped:,} already current; {stale:,} changed during generation (rerun). Model: {model}."
            ))

    def _report_progress(self, completed, skipped, started):
        rate = completed / max(perf_counter() - started, 0.001)
        embedding_seconds, database_seconds = self._batch_timings
        self.stdout.write(
            f"Embedded {completed:,} movies; {skipped:,} already current; {rate:.1f} movies/s; "
            f"batch embedding {embedding_seconds:.2f}s, database {database_seconds:.2f}s."
        )

    def _embed_batch(self, batch, model, *, workers=4, request_batch_size=32):
        from movies.models import Movie

        embedding_started = perf_counter()
        try:
            vectors = embed_texts(
                [row["search_text"] for row, _ in batch], model=model,
                concurrency=workers, request_batch_size=request_batch_size,
            )
        except EmbeddingError as exc:
            raise CommandError(f"{exc} Completed batches are saved; rerun this command to resume.") from exc
        except KeyboardInterrupt:
            raise CommandError("Interrupted. Completed batches are saved; rerun this command to resume.") from None
        embedding_seconds = perf_counter() - embedding_started
        if len(vectors) != len(batch):
            raise CommandError("The provider returned an unexpected vector count. Completed batches are saved; rerun to resume.")
        captured = {
            row["wikipedia_id"]: (row["search_text"], digest, vector)
            for (row, digest), vector in zip(batch, vectors, strict=True)
        }
        field = Movie._meta.get_field("embedding")
        database_started = perf_counter()
        try:
            with transaction.atomic():
                current = Movie.objects.select_for_update().filter(pk__in=captured).only("wikipedia_id", "search_text")
                updates = []
                for movie in current:
                    source_text, digest, vector = captured[movie.pk]
                    # Lock before checking text so a concurrent loader cannot
                    # attach the old vector to a newly imported plot.
                    if movie.search_text != source_text:
                        continue
                    # Each CASE branch needs a vector type: serialized vector
                    # values otherwise resolve as text in PostgreSQL bulk updates.
                    movie.embedding = Cast(Value(vector, output_field=field), output_field=field)
                    movie.embedding_model = model
                    movie.embedding_hash = digest
                    updates.append(movie)
                saved = Movie.objects.bulk_update(
                    updates, ["embedding", "embedding_model", "embedding_hash"], batch_size=128,
                )
        except KeyboardInterrupt:
            raise CommandError("Interrupted. Completed batches are saved; rerun this command to resume.") from None
        self._batch_timings = (embedding_seconds, perf_counter() - database_started)
        return saved
