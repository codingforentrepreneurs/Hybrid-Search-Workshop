"""Offline regression checks for fast, resumable embedding persistence."""

import io
from unittest.mock import patch

from django.core.management import call_command, CommandError
from django.db import connection
from django.db.models import Case, Value, When
from django.db.models.functions import Cast
from django.db.models.sql.subqueries import UpdateQuery
from django.test import SimpleTestCase, TestCase, override_settings
from django.test.utils import CaptureQueriesContext

from movies.embeddings import EmbeddingError, text_hash
from movies.management.commands.embed_movies import Command
from movies.models import Movie


COMMAND_MODULE = "movies.management.commands.embed_movies"
MODEL = "fastembed:local/test"


class EmbeddingCommandOptionsTests(SimpleTestCase):
    def test_invalid_options_fail_before_database_or_provider_access(self):
        invalid = [
            ({"batch_size": 0}, "--batch-size"),
            ({"limit": -1}, "--limit"),
            ({"workers": 0}, "--workers"),
            ({"workers": 33}, "--workers"),
            ({"request_batch_size": 0}, "--request-batch-size"),
            ({"request_batch_size": 129}, "--request-batch-size"),
        ]
        for options, message in invalid:
            with self.subTest(options=options), patch(f"{COMMAND_MODULE}.embed_texts") as embed:
                with self.assertRaisesMessage(CommandError, message):
                    call_command("embed_movies", stdout=io.StringIO(), **options)
                embed.assert_not_called()

    def test_vector_case_values_compile_with_explicit_postgres_vector_cast(self):
        from django.db.backends.postgresql.base import DatabaseWrapper

        database = DatabaseWrapper({"NAME": "never-connected", "OPTIONS": {}}, alias="compile_only")
        field = Movie._meta.get_field("embedding")
        query = UpdateQuery(Movie)
        query.add_update_values({"embedding": Case(
            When(pk=1, then=Cast(Value([1.0, 0.0], output_field=field), output_field=field)),
            output_field=field,
        )})
        statement, params = query.get_compiler(connection=database).as_sql()
        self.assertIn("::vector", statement)
        self.assertIn("[1.0,0.0]", params)
        self.assertIsNone(database.connection)


@override_settings(EMBEDDING_PROVIDER="fastembed", FASTEMBED_MODEL="local/test", FASTEMBED_DIMENSIONS=2)
class EmbeddingCommandPersistenceTests(TestCase):
    def make_movies(self, count):
        Movie.objects.bulk_create([
            Movie(wikipedia_id=index, title=f"Movie {index}", plot=f"Plot {index}",
                  search_text=f"Searchable plot {index}", wikipedia_url=f"https://en.wikipedia.org/?curid={index}")
            for index in range(1, count + 1)
        ])

    def batch(self):
        return [(row, text_hash(row["search_text"], model=MODEL)) for row in
                Movie.objects.values("wikipedia_id", "search_text")]

    def test_saves_batch_with_one_read_and_one_bulk_update(self):
        self.make_movies(16)
        batch = self.batch()
        with patch(f"{COMMAND_MODULE}.embed_texts", return_value=[[1, 0]] * 16), CaptureQueriesContext(connection) as queries:
            saved = Command()._embed_batch(batch, MODEL)
        self.assertEqual(saved, 16)
        statements = [query["sql"] for query in queries]
        self.assertEqual(sum(statement.startswith("SELECT") for statement in statements), 1)
        self.assertEqual(sum(statement.startswith("UPDATE") for statement in statements), 1)
        read = next(statement for statement in statements if statement.startswith("SELECT"))
        self.assertNotIn('"embedding"', read)
        self.assertEqual(Movie.objects.get(pk=8).embedding, [1, 0])
        self.assertEqual(Movie.objects.get(pk=8).embedding_hash, batch[7][1])

    def test_bulk_updates_are_bounded_to_128_movies(self):
        self.make_movies(130)
        batch = self.batch()
        with patch(f"{COMMAND_MODULE}.embed_texts", return_value=[[1, 0]] * 130), CaptureQueriesContext(connection) as queries:
            self.assertEqual(Command()._embed_batch(batch, MODEL), 130)
        self.assertEqual(sum(query["sql"].startswith("UPDATE") for query in queries), 2)

    def test_changed_or_deleted_movies_are_skipped_and_unchanged_movies_saved(self):
        self.make_movies(3)
        batch = self.batch()

        def provider(*args, **kwargs):
            Movie.objects.filter(pk=1).update(search_text="A replacement plot")
            Movie.objects.filter(pk=2).delete()
            return [[1, 0]] * 3

        with patch(f"{COMMAND_MODULE}.embed_texts", side_effect=provider):
            self.assertEqual(Command()._embed_batch(batch, MODEL), 1)
        self.assertIsNone(Movie.objects.get(pk=1).embedding)
        self.assertEqual(Movie.objects.get(pk=1).embedding_hash, "")
        self.assertFalse(Movie.objects.filter(pk=2).exists())
        self.assertEqual(Movie.objects.get(pk=3).embedding_hash, batch[2][1])

    def test_later_failure_preserves_committed_batches_and_retry_only_embeds_pending(self):
        self.make_movies(4)
        with patch(f"{COMMAND_MODULE}.connection") as command_connection, patch(
            f"{COMMAND_MODULE}.embed_texts", side_effect=[[[1, 0], [0, 1]], EmbeddingError("provider unavailable")],
        ):
            command_connection.vendor = "postgresql"
            with self.assertRaisesMessage(CommandError, "Completed batches are saved"):
                call_command("embed_movies", batch_size=2, stdout=io.StringIO())
        self.assertEqual(Movie.objects.filter(embedding__isnull=False).count(), 2)
        output = io.StringIO()
        with patch(f"{COMMAND_MODULE}.connection") as command_connection, patch(
            f"{COMMAND_MODULE}.embed_texts", return_value=[[0, 1], [1, 0]],
        ) as embed:
            command_connection.vendor = "postgresql"
            call_command("embed_movies", batch_size=2, workers=6, request_batch_size=48, stdout=output)
        embed.assert_called_once_with(
            ["Searchable plot 3", "Searchable plot 4"], model=MODEL, concurrency=6, request_batch_size=48,
        )
        self.assertEqual(Movie.objects.filter(embedding__isnull=False).count(), 4)
        self.assertIn("2 already current", output.getvalue())
        self.assertIn("movies/s", output.getvalue())
        self.assertIn("batch embedding", output.getvalue())
        self.assertIn("database", output.getvalue())

    def test_defaults_are_reported_and_passed_to_provider(self):
        self.make_movies(1)
        output = io.StringIO()
        with patch(f"{COMMAND_MODULE}.connection") as command_connection, patch(
            f"{COMMAND_MODULE}.embed_texts", return_value=[[1, 0]],
        ) as embed:
            command_connection.vendor = "postgresql"
            call_command("embed_movies", stdout=output)
        embed.assert_called_once_with(["Searchable plot 1"], model=MODEL, concurrency=4, request_batch_size=32)
        self.assertIn("movie batch: 128", output.getvalue())
        self.assertIn("workers: 4", output.getvalue())
        self.assertIn("request batch: 32", output.getvalue())

    def test_wrong_provider_vector_count_does_not_save_partial_batch(self):
        self.make_movies(2)
        batch = self.batch()
        with patch(f"{COMMAND_MODULE}.embed_texts", return_value=[[1, 0]]):
            with self.assertRaisesMessage(CommandError, "unexpected vector count"):
                Command()._embed_batch(batch, MODEL)
        self.assertFalse(Movie.objects.filter(embedding__isnull=False).exists())
