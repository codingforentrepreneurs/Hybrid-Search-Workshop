"""Offline contracts for data ingestion, embedding providers, and the UI."""

from collections import Counter
from contextlib import contextmanager
import io
import json
import math
from pathlib import Path
import tarfile
import tempfile
from unittest.mock import MagicMock, patch

import httpx
from django.core.exceptions import ValidationError
from django.core.management import call_command
from django.db import DatabaseError
from django.test import SimpleTestCase, TestCase, override_settings

from movies import corpus, embeddings
from movies.fields import VectorField
from movies.management.commands.load_movies import Command as LoadCommand
from movies.models import Movie
from movies.search import empty_columns


def metadata_row(movie_id, title, *, runtime="90", languages=None, genres=None):
    return "\t".join([
        str(movie_id), "/m/example", title, "1999-03-01", "",
        runtime, json.dumps(languages or {}), "{}", json.dumps(genres or {}),
    ])


def make_archive(path, *, metadata="", plots="", omit=()):
    with tarfile.open(path, "w:gz") as archive:
        for name, value in ((corpus.METADATA_MEMBER, metadata), (corpus.PLOTS_MEMBER, plots)):
            if name in omit:
                continue
            data = value.encode("utf-8")
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return path


class CorpusTests(SimpleTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_literal_quotes_unicode_json_and_embedded_tabs_survive(self):
        archive = make_archive(
            self.root / "movies.tar.gz",
            metadata=metadata_row(1, '"Amélie" 日本', languages={"x": "Français"},
                                  genres={"b": "Romance", "a": "Comedy", "c": "Comedy"}),
            plots='1\tA café\tand a "quoted" adventure. 🌍\n',
        )
        movie, = corpus.iter_movies(archive)
        self.assertEqual(movie["title"], '"Amélie" 日本')
        self.assertEqual(movie["plot"], 'A café\tand a "quoted" adventure. 🌍')
        self.assertEqual(movie["languages"], ["Français"])
        self.assertEqual(movie["genres"], ["Comedy", "Romance"])
        self.assertIn(movie["plot"], movie["search_text"])
        self.assertEqual(movie["wikipedia_url"], "https://en.wikipedia.org/?curid=1")

    def test_malformed_duplicates_and_unmatched_rows_are_counted(self):
        metadata = "\n".join([
            metadata_row(1, "First", runtime="nan"),
            metadata_row(1, "Duplicate"), metadata_row(2, "No plot"),
            "broken\trow", metadata_row(-1, "Invalid ID"),
        ])
        archive = make_archive(self.root / "movies.tar.gz", metadata=metadata,
                               plots="1\tOriginal\n1\tDuplicate\n3\tMissing metadata\nbroken\n2\t \n")
        stats = Counter()
        movies = list(corpus.iter_movies(archive, limit=0, stats=stats))
        self.assertEqual([(m["title"], m["plot"]) for m in movies], [("First", "Original")])
        self.assertIsNone(movies[0]["runtime_minutes"])
        for key, count in {"duplicate_metadata_rows": 1, "duplicate_plot_rows": 1,
                           "malformed_metadata_rows": 2, "malformed_plot_rows": 2,
                           "malformed_metadata_fields": 1, "plots_without_metadata": 1,
                           "metadata_without_plot": 1}.items():
            with self.subTest(key=key):
                self.assertEqual(stats[key], count)

    def test_limit_applies_to_valid_joined_movies_in_plot_order(self):
        archive = make_archive(self.root / "movies.tar.gz",
                               metadata="\n".join(metadata_row(i, str(i)) for i in (1, 2)),
                               plots="3\tMissing\n2\tSecond\n1\tFirst\n")
        self.assertEqual([m["wikipedia_id"] for m in corpus.iter_movies(archive, limit=1)], [2])
        self.assertEqual([m["wikipedia_id"] for m in corpus.iter_movies(archive, limit=0)], [2, 1])
        with self.assertRaises(ValueError):
            list(corpus.iter_movies(archive, limit=-1))

    def test_missing_member_and_corrupt_archive_raise_actionable_errors(self):
        missing = make_archive(self.root / "missing.tar.gz", omit=(corpus.PLOTS_MEMBER,))
        corrupt = self.root / "corrupt.tar.gz"
        corrupt.write_bytes(b"not an archive")
        for archive in (missing, corrupt):
            with self.subTest(path=archive.name), self.assertRaises(corpus.CorpusError):
                list(corpus.iter_movies(archive))

    def test_valid_download_cache_skips_network(self):
        archive = make_archive(self.root / "MovieSummaries.tar.gz")
        with patch("movies.corpus.httpx.stream") as stream:
            self.assertEqual(corpus.download_archive(self.root), archive)
        stream.assert_not_called()

    def test_failed_download_removes_partial_files(self):
        with patch("movies.corpus.httpx.stream", side_effect=httpx.ConnectError("offline")):
            with self.assertRaisesMessage(corpus.CorpusError, "--archive"):
                corpus.download_archive(self.root)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_prepare_only_writes_unicode_jsonl_without_database_or_embeddings(self):
        archive = make_archive(self.root / "movies.tar.gz", metadata=metadata_row(1, "Amélie"),
                               plots="1\tUne aventure 🌍\n")
        output = self.root / "prepared.jsonl"
        with patch("movies.embeddings.embed_texts") as embed:
            call_command("load_movies", archive=archive, prepare_only=True, output=output,
                         stdout=io.StringIO())
        self.assertEqual(json.loads(output.read_text())["title"], "Amélie")
        embed.assert_not_called()
        output.write_text("previous good artifact")

        def broken_rows():
            yield {"title": "partial"}
            raise corpus.CorpusError("bad later row")

        with self.assertRaises(corpus.CorpusError):
            corpus.write_jsonl(broken_rows(), output)
        self.assertEqual(output.read_text(), "previous good artifact")
        self.assertEqual(list(self.root.glob("*.part")), [])


@override_settings(EMBEDDING_PROVIDER="auto", OPENROUTER_API_KEY="", FASTEMBED_MODEL="local/test",
                   FASTEMBED_DIMENSIONS=2, OPENROUTER_EMBEDDING_MODEL="remote/test",
                   OPENROUTER_EMBEDDING_DIMENSIONS=2)
class EmbeddingTests(SimpleTestCase):
    def test_auto_uses_local_for_blank_key_and_remote_for_present_key(self):
        for key, expected in [("", "fastembed"), (" \n\t", "fastembed"), (" secret ", "openrouter")]:
            with self.subTest(key=key), override_settings(OPENROUTER_API_KEY=key):
                self.assertEqual(embeddings.get_embedding_config().provider, expected)
        with override_settings(EMBEDDING_PROVIDER="unknown"):
            with self.assertRaises(embeddings.EmbeddingError):
                embeddings.get_embedding_config()

    def test_hash_separates_provider_dimensions_model_and_text(self):
        baseline = embeddings.text_hash("a movie")
        changes = [
            {"OPENROUTER_API_KEY": "key"}, {"FASTEMBED_DIMENSIONS": 3},
            {"FASTEMBED_MODEL": "local/other"},
        ]
        for change in changes:
            with self.subTest(change=change), override_settings(**change):
                self.assertNotEqual(embeddings.text_hash("a movie"), baseline)
        self.assertNotEqual(embeddings.text_hash("a different movie"), baseline)
        self.assertEqual(embeddings.text_hash("a movie"), baseline)
        with self.assertRaises(embeddings.EmbeddingError):
            embeddings.text_hash("a movie", model="openrouter:remote/test")

    def test_unicode_chunking_is_lossless_and_bounded(self):
        text = "雪🌍 café\n" * 120
        chunks = embeddings.chunk_text(text, max_bytes=19)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(0 < len(chunk.encode("utf-8")) <= 19 for chunk in chunks))
        for invalid in ("", " \n", None):
            with self.subTest(invalid=invalid), self.assertRaises(embeddings.EmbeddingError):
                embeddings.chunk_text(invalid)

    def test_local_model_uses_all_chunks_weighted_normalized_and_query_method(self):
        def vector(values):
            result = MagicMock()
            result.tolist.return_value = values
            return result

        local = MagicMock()
        local.passage_embed.return_value = [vector([1, 0]), vector([0, 1])]
        local.query_embed.return_value = [vector([0, 2])]
        # Local chunks are 400 bytes: both the opening and the ending contribute.
        with patch("movies.embeddings._local_model", return_value=local), patch("movies.embeddings.httpx.Client") as remote:
            result = embeddings.embed_texts(["a" * 400 + "b" * 200])
            query = embeddings.embed_texts(["find a movie"], input_type="query")
        local.passage_embed.assert_called_once_with(["a" * 400, "b" * 200])
        local.query_embed.assert_called_once_with(["find a movie"])
        self.assertAlmostEqual(result[0][0], 2 / math.sqrt(5))
        self.assertAlmostEqual(result[0][1], 1 / math.sqrt(5))
        self.assertEqual(query, [[0, 1]])
        remote.assert_not_called()

    def test_openrouter_sends_auth_correct_path_and_reorders_response(self):
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={"data": [
                {"index": 1, "embedding": [0, 4]}, {"index": 0, "embedding": [3, 0]},
            ]})

        real_client = httpx.Client
        with override_settings(OPENROUTER_API_KEY=" key ", OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"), patch(
            "movies.embeddings.httpx.Client", side_effect=lambda **kwargs: real_client(
                **kwargs, transport=httpx.MockTransport(respond)
            )
        ):
            result = embeddings.embed_texts(["first", "second"])
        self.assertEqual(result, [[1, 0], [0, 1]])
        self.assertEqual(str(requests[0].url), "https://openrouter.ai/api/v1/embeddings")
        self.assertEqual(requests[0].headers["Authorization"], "Bearer key")
        self.assertEqual(json.loads(requests[0].content)["input"], ["first", "second"])

    def test_invalid_provider_vectors_are_rejected(self):
        payloads = [
            {"data": [{"index": 0, "embedding": [1]}]},
            {"data": [{"index": 0, "embedding": [float("nan"), 1]}]},
            {"data": [{"index": 0, "embedding": [float("inf"), 1]}]},
            {"data": [{"index": 0, "embedding": [True, 1]}]},
            {"data": [{"index": 0, "embedding": ["1", 1]}]},
            {"data": [{"index": 1, "embedding": [1, 1]}]},
            {"data": []}, {},
        ]
        for payload in payloads:
            with self.subTest(payload=payload), self.assertRaises(embeddings.EmbeddingError):
                embeddings._vectors(payload, count=1, dimensions=2)
        with self.assertRaises(embeddings.EmbeddingError):
            embeddings._vectors({"data": [{"index": 0, "embedding": [1, 1]}] * 2}, count=2, dimensions=2)

    def test_rate_limit_and_transport_failures_retry_with_bounded_waits(self):
        client = MagicMock()
        client.post.side_effect = [
            httpx.ConnectError("network"), httpx.Response(429, headers={"Retry-After": "999"}),
            httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 0]}]}),
        ]
        with patch("movies.embeddings.time.sleep") as sleep:
            self.assertEqual(embeddings._request(client, ["plot"], "model", 2), [[1, 0]])
        self.assertEqual(client.post.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 8])
        client.post.reset_mock(side_effect=True)
        client.post.return_value = httpx.Response(503)
        with patch("movies.embeddings.time.sleep"), self.assertRaises(embeddings.EmbeddingError):
            embeddings._request(client, ["plot"], "model", 2)
        self.assertEqual(client.post.call_count, 4)

    def test_auth_failure_does_not_retry_or_fall_back_to_local_model(self):
        client = MagicMock()
        client.post.return_value = httpx.Response(401, text="provider secret details")
        with patch("movies.embeddings.time.sleep") as sleep, self.assertRaisesMessage(
            embeddings.EmbeddingError, "check OPENROUTER_API_KEY"
        ):
            embeddings._request(client, ["plot"], "model", 2)
        self.assertEqual(client.post.call_count, 1)
        sleep.assert_not_called()
        with override_settings(EMBEDDING_PROVIDER="openrouter", OPENROUTER_API_KEY=""), patch(
            "movies.embeddings._local_model"
        ) as local, self.assertRaises(embeddings.EmbeddingError):
            embeddings.embed_texts(["plot"])
        local.assert_not_called()

    def test_zero_vectors_cannot_be_normalized(self):
        with patch("movies.embeddings._local_vectors", return_value=[[0, 0]]):
            with self.assertRaisesMessage(embeddings.EmbeddingError, "cannot be normalized"):
                embeddings.embed_texts(["plot"])


class ImportTests(TestCase):
    def setUp(self):
        self.row = {
            "wikipedia_id": 10, "title": "Original", "plot": "A story", "search_text": "Original\nA story",
            "release_year": 2000, "release_date": "2000", "genres": ["Comedy"], "languages": [],
            "countries": [], "runtime_minutes": 100, "wikipedia_url": "https://en.wikipedia.org/?curid=10",
        }
        LoadCommand()._save_batch([self.row], Counter())
        Movie.objects.filter(pk=10).update(embedding=[1, 0], embedding_model="fastembed:test", embedding_hash="hash")

    def test_reimport_updates_metadata_without_losing_embedding_or_duplicating_movie(self):
        updated = dict(self.row, runtime_minutes=110)
        stats = Counter()
        LoadCommand()._save_batch([updated], stats)
        movie = Movie.objects.get(pk=10)
        self.assertEqual(Movie.objects.count(), 1)
        self.assertEqual(movie.runtime_minutes, 110)
        self.assertEqual(movie.embedding, [1, 0])
        self.assertEqual(movie.embedding_model, "fastembed:test")
        self.assertEqual(movie.embedding_hash, "hash")
        self.assertEqual(stats["updated"], 1)

    def test_changed_search_text_invalidates_embedding(self):
        LoadCommand()._save_batch([dict(self.row, plot="Changed", search_text="Original\nChanged")], Counter())
        movie = Movie.objects.get(pk=10)
        self.assertIsNone(movie.embedding)
        self.assertEqual(movie.embedding_model, "")
        self.assertEqual(movie.embedding_hash, "")
        self.assertEqual(movie.plot, "Changed")


class ViewTests(SimpleTestCase):
    @contextmanager
    def configured_database(self):
        with patch("movies.views.connection") as connection, patch("movies.views.Movie.objects") as manager, patch(
            "movies.views.check_search_ready"
        ), patch("movies.views.search_movies") as search:
            connection.vendor = "postgresql"
            manager.annotate.return_value.aggregate.return_value = {"total": 10, "embedded": 10}
            manager.exclude.return_value.order_by.return_value.values_list.return_value.distinct.return_value = [2000]
            connection.cursor.return_value.__enter__.return_value.fetchall.return_value = [("Comedy",)]
            search.return_value = {"columns": empty_columns(), "timings": {}}
            yield search, manager

    def test_setup_page_is_usable_without_database_and_get_only(self):
        with patch("movies.views.connection") as connection:
            connection.vendor = "sqlite"
            response = self.client.get("/")
            self.assertContains(response, "Set DATABASE_URL")
            self.assertEqual(len(response.context["columns"]), 3)
            self.assertFalse(response.context["database_ready"])
            searched = self.client.get("/", {"q": "archaeology"})
            self.assertContains(searched, "Search requires your Tiger Cloud database")
            self.assertEqual(self.client.post("/").status_code, 405)

    def test_invalid_query_genre_and_year_never_call_embedding_search(self):
        cases = [({"q": "x" * 1001}, "1,000"), ({"q": "film", "genre": "x" * 201}, "genre"),
                 ({"q": "film", "year": "nonsense"}, "1800"), ({"q": "film", "year": "2101"}, "1800")]
        with self.configured_database() as (search, _):
            for params, message in cases:
                with self.subTest(params=params):
                    response = self.client.get("/", params)
                    self.assertIn(message, response.context["error"])
            search.assert_not_called()

    def test_database_and_embedding_errors_are_safe_and_actionable(self):
        with self.configured_database() as (_, manager):
            manager.annotate.side_effect = DatabaseError("password=secret database internals")
            response = self.client.get("/", {"q": "film"})
            self.assertContains(response, "The movie database is unavailable")
            self.assertNotContains(response, "password=secret")
        with self.configured_database() as (search, _):
            search.side_effect = embeddings.EmbeddingError("OpenRouter rejected the credentials; check OPENROUTER_API_KEY.")
            response = self.client.get("/", {"q": "film"})
            self.assertContains(response, "check OPENROUTER_API_KEY")

    def test_results_escape_query_title_plot_and_filters(self):
        malicious = '<script>alert("x")</script>'
        columns = empty_columns()
        columns[0]["results"] = [{
            "wikipedia_id": 1, "title": malicious, "plot": malicious, "genres": [malicious],
            "rank": 1, "release_year": 2000, "score": -2, "wikipedia_url": "https://en.wikipedia.org/?curid=1",
        }]
        with self.configured_database() as (search, _):
            search.return_value = {"columns": columns, "timings": {}}
            response = self.client.get("/", {"q": malicious, "genre": malicious, "year": "2000"})
            search.assert_called_once_with(malicious, genre=malicious, year=2000)
        self.assertContains(response, "&lt;script&gt;")
        self.assertNotContains(response, malicious)


class VectorFieldTests(SimpleTestCase):
    def test_roundtrip_supports_provider_dimensions_and_json(self):
        field = VectorField(dimensions=2)
        self.assertEqual(field.to_python("[1,0.5]"), [1.0, 0.5])
        self.assertEqual(field.from_db_value(field.get_prep_value((1, 0.5)), None, None), [1.0, 0.5])
        self.assertIsNone(field.to_python(None))
        self.assertEqual(len(VectorField().to_python([0.5] * 384)), 384)

    def test_nonfinite_wrong_shape_and_non_numeric_vectors_are_rejected(self):
        field = VectorField(dimensions=2)
        for value in ([], [1], [1, 2, 3], [float("nan"), 1], [1, float("inf")],
                      {"x": 1}, "not json", [None, 1], [[1], [2]]):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                field.get_prep_value(value)
