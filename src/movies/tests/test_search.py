"""SQL-construction regressions, not live PostgreSQL integration tests."""

import io
import json
from unittest.mock import patch

from django.conf import settings
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase, TestCase, override_settings

from movies import embeddings, search
from movies.management.commands.embed_movies import Command as EmbedCommand
from movies.models import Movie


@override_settings(EMBEDDING_PROVIDER="fastembed", FASTEMBED_MODEL="local/test", FASTEMBED_DIMENSIONS=2)
class SearchSQLTests(SimpleTestCase):
    def test_retrieval_parameters_and_model_specific_columns_keep_inputs_separate(self):
        query = "a film'); DROP TABLE movies_movie;--"
        genre = "Comedy' OR 1=1 --"
        with patch("movies.search._query_rows", return_value=([], 0)) as execute:
            search.retrieve_bm25(query, genre=genre, year=1999, candidate_limit=17)
            statement, params = execute.call_args.args
            self.assertNotIn(query, statement)
            self.assertNotIn(genre, statement)
            self.assertEqual(params, [query, settings.SEARCH_BM25_INDEX, json.dumps([genre]), 1999, 17])

            columns = set()
            for config in [embeddings.EmbeddingConfig("fastembed", "same/model", 2),
                           embeddings.EmbeddingConfig("openrouter", "same/model", 2),
                           embeddings.EmbeddingConfig("fastembed", "same/model", 3)]:
                with self.subTest(config=config), patch("movies.embeddings.get_embedding_config", return_value=config):
                    vector = [1.0] * config.dimensions
                    search.retrieve_semantic(vector, genre=genre, year=1999, candidate_limit=17)
                    statement, params = execute.call_args.args
                    column = search.vector_column_name(config)
                    columns.add(column)
                    self.assertIn(f'"{column}" <=> %s::vector({config.dimensions})', statement)
                    self.assertIn(f'"{column}" IS NOT NULL', statement)
                    self.assertNotIn("embedding::", statement)
                    self.assertNotIn(genre, statement)
                    self.assertEqual(params, [json.dumps(vector), json.dumps([genre]), 1999, 17])
            self.assertEqual(len(columns), 3, "Provider and dimension changes must target separate physical columns")

    def test_index_setup_guards_model_and_dimensions_and_indexes_physical_column(self):
        model = "movie'; DROP TABLE movies_movie;--"
        index_name = 'movie"; DROP TABLE movies_movie;--'
        with override_settings(FASTEMBED_MODEL=model, SEARCH_BM25_INDEX=index_name), patch(
            "movies.management.commands.setup_search_indexes.connection"
        ) as connection:
            connection.vendor = "postgresql"
            cursor = connection.cursor.return_value.__enter__.return_value
            call_command("setup_search_indexes", stdout=io.StringIO())
            statements = [call.args[0] for call in cursor.execute.call_args_list]
            statements = [statement if isinstance(statement, str) else statement.as_string() for statement in statements]
            column = search.vector_column_name()
        generated = next(statement for statement in statements if statement.startswith("ALTER TABLE"))
        diskann = next(statement for statement in statements if "USING diskann" in statement)
        bm25 = next(statement for statement in statements if "USING bm25" in statement)
        self.assertIn("GENERATED ALWAYS AS (CASE", generated)
        self.assertIn("embedding_model = 'fastembed:movie''; DROP TABLE movies_movie;--'", generated)
        self.assertIn("vector_dims(embedding) = 2", generated)
        self.assertIn("THEN embedding::vector(2) ELSE NULL END) STORED", generated)
        self.assertIn(f'USING diskann ("{column}" vector_cosine_ops)', diskann)
        self.assertIn(f'WHERE "{column}" IS NOT NULL', diskann)
        self.assertNotIn("::vector", diskann)
        self.assertIn('"movie""; DROP TABLE movies_movie;--"', bm25)

    def test_unusable_query_vectors_fail_before_database_execution(self):
        with patch("movies.search._query_rows") as execute:
            for vector in (None, [], [0, 0], [1], [1, 2, 3], [float("nan"), 1],
                           [float("inf"), 1], ["invalid", 1]):
                with self.subTest(vector=vector), self.assertRaises(search.SearchUnavailable):
                    search.retrieve_semantic(vector)
            with override_settings(FASTEMBED_DIMENSIONS=16001), self.assertRaises(search.SearchUnavailable):
                search.retrieve_semantic([1, 1])
        execute.assert_not_called()

    def test_rrf_rejects_invalid_smoothing_before_query_and_preserves_absent_ranks(self):
        with patch("movies.search._query_rows", return_value=([], 0)) as execute:
            for k in (0, -1, float("nan"), float("inf")):
                with self.subTest(k=k), self.assertRaises(ValueError):
                    search.fuse_rankings([], [], k=k)
            execute.assert_not_called()
            self.assertEqual(search.fuse_rankings([], []), ([], 0))
            execute.assert_not_called()
            search.fuse_rankings([{"wikipedia_id": 1, "rank": 1}], [{"wikipedia_id": 2, "rank": 3}], k=60)
            statement, params = execute.call_args.args
        self.assertEqual(json.loads(params[0]), [
            {"wikipedia_id": 1, "rank": 1, "method": "bm25"},
            {"wikipedia_id": 2, "rank": 3, "method": "semantic"},
        ])
        self.assertEqual(params[1], 60)
        self.assertIn("SUM(1.0 / (%s + rank))", statement)
        self.assertIn("ORDER BY s.score DESC, m.wikipedia_id", statement)
        self.assertNotIn("ELSE 0", statement)

    def test_readiness_requires_a_valid_physical_diskann_index(self):
        with patch("movies.search.connection") as connection:
            connection.vendor = "postgresql"
            cursor = connection.cursor.return_value.__enter__.return_value
            extensions = [(name,) for name in ("vector", "vectorscale", "pg_textsearch")]
            cursor.fetchall.side_effect = [extensions, [(settings.SEARCH_BM25_INDEX,)]]
            with self.assertRaisesMessage(search.SearchUnavailable, "setup_search_indexes"):
                search.check_search_ready()
            statement, params = cursor.execute.call_args.args
        self.assertIn("i.indisvalid", statement)
        self.assertIn("i.indexprs IS NULL", statement)
        self.assertIn("a.attgenerated = 's'", statement)
        self.assertEqual(params[-2:], [search.vector_index_name(), search.vector_column_name()])

    def test_test_settings_isolate_the_database_from_cloud_credentials(self):
        from config import test_settings

        self.assertEqual(test_settings.DATABASES["default"]["ENGINE"], "django.db.backends.sqlite3")
        database_name = test_settings.DATABASES["default"]["NAME"]
        self.assertTrue(database_name == ":memory:" or (
            database_name.startswith("file:memorydb_") and "mode=memory" in database_name
        ))


@override_settings(EMBEDDING_PROVIDER="fastembed", FASTEMBED_MODEL="local/test", FASTEMBED_DIMENSIONS=2)
class EmbeddingResumeTests(TestCase):
    def setUp(self):
        for movie_id in (1, 2):
            Movie.objects.create(wikipedia_id=movie_id, title=f"Movie {movie_id}", plot=f"Plot {movie_id}",
                                 search_text=f"Searchable plot {movie_id}", wikipedia_url=f"https://en.wikipedia.org/?curid={movie_id}")

    def test_failed_later_batch_preserves_completed_work_and_resumes_only_pending(self):
        with patch("movies.management.commands.embed_movies.connection") as connection, patch(
            "movies.management.commands.embed_movies.embed_texts",
            side_effect=[[[1, 0]], embeddings.EmbeddingError("provider unavailable")],
        ):
            connection.vendor = "postgresql"
            with self.assertRaisesMessage(CommandError, "Completed batches are saved"):
                call_command("embed_movies", batch_size=1, stdout=io.StringIO())
        self.assertEqual(Movie.objects.get(pk=1).embedding, [1, 0])
        self.assertIsNone(Movie.objects.get(pk=2).embedding)
        with patch("movies.management.commands.embed_movies.connection") as connection, patch(
            "movies.management.commands.embed_movies.embed_texts", return_value=[[0, 1]]
        ) as embed:
            connection.vendor = "postgresql"
            call_command("embed_movies", batch_size=1, stdout=io.StringIO())
        embed.assert_called_once_with(
            ["Searchable plot 2"], model="fastembed:local/test", concurrency=4, request_batch_size=32,
        )
        self.assertEqual(Movie.objects.get(pk=1).embedding, [1, 0])
        self.assertEqual(Movie.objects.get(pk=2).embedding, [0, 1])

    def test_concurrent_text_change_does_not_receive_a_stale_embedding(self):
        def change_movie_during_request(*args, **kwargs):
            Movie.objects.filter(pk=1).update(search_text="Changed during embedding")
            return [[1, 0]]

        row = {"wikipedia_id": 1, "search_text": "Searchable plot 1"}
        with patch("movies.management.commands.embed_movies.embed_texts", side_effect=change_movie_during_request):
            saved = EmbedCommand()._embed_batch([(row, "original-digest")], "fastembed:local/test")
        self.assertEqual(saved, 0)
        movie = Movie.objects.get(pk=1)
        self.assertEqual(movie.search_text, "Changed during embedding")
        self.assertIsNone(movie.embedding)
        self.assertEqual(movie.embedding_hash, "")
