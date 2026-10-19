"""Concurrent HTTP responses must retain their original chunk/movie mapping."""

import math
import threading
from unittest.mock import patch

from django.test import SimpleTestCase, override_settings

from movies import embeddings


@override_settings(
    EMBEDDING_PROVIDER="openrouter", OPENROUTER_API_KEY="test-key",
    OPENROUTER_EMBEDDING_MODEL="test/model", OPENROUTER_EMBEDDING_DIMENSIONS=2,
)
class ConcurrentEmbeddingTests(SimpleTestCase):
    def test_overlapping_requests_finish_out_of_order_without_mixing_movies(self):
        both_running = threading.Barrier(2, timeout=5)
        second_finished = threading.Event()
        completed = []

        def request(client, inputs, *, model, dimensions):
            text = inputs[0]
            if text[0] in {"a", "b"}:
                both_running.wait()
            if text[0] == "a":
                self.assertTrue(second_finished.wait(timeout=5))
                completed.append("a")
                return [[1, 0]]
            if text[0] == "b":
                completed.append("b")
                second_finished.set()
                return [[0, 1]]
            return [[1, 1]]

        texts = ["a" * 6000 + "b" * 3000, "c" * 100]
        original_hash = embeddings.text_hash(texts[0])
        with patch("movies.embeddings._request", side_effect=request):
            vectors = embeddings.embed_texts(texts, concurrency=2, request_batch_size=1)
        self.assertEqual(completed, ["b", "a"])
        self.assertAlmostEqual(vectors[0][0], 2 / math.sqrt(5))
        self.assertAlmostEqual(vectors[0][1], 1 / math.sqrt(5))
        self.assertAlmostEqual(vectors[1][0], 1 / math.sqrt(2))
        self.assertAlmostEqual(vectors[1][1], 1 / math.sqrt(2))
        self.assertEqual(original_hash, embeddings.text_hash(texts[0]))

    def test_request_batch_size_bounds_http_payloads(self):
        batches = []

        def request(client, inputs, *, model, dimensions):
            batches.append(len(inputs))
            return [[3, 4] for _ in inputs]

        with patch("movies.embeddings._request", side_effect=request):
            vectors = embeddings.embed_texts([f"Plot {i}" for i in range(70)], request_batch_size=32)
        self.assertEqual(batches, [32, 32, 6])
        self.assertEqual(vectors, [[0.6, 0.8]] * 70)

    def test_worker_failure_does_not_return_partial_or_misaligned_vectors(self):
        with patch("movies.embeddings._request", side_effect=embeddings.EmbeddingError("provider unavailable")):
            with self.assertRaisesMessage(embeddings.EmbeddingError, "provider unavailable"):
                embeddings.embed_texts(["one", "two"], concurrency=2, request_batch_size=1)

    def test_invalid_scheduling_options_never_open_a_client(self):
        with patch("movies.embeddings.httpx.Client") as client:
            for options in ({"concurrency": 0}, {"concurrency": 33},
                            {"request_batch_size": 0}, {"request_batch_size": 129}):
                with self.subTest(options=options), self.assertRaises(embeddings.EmbeddingError):
                    embeddings.embed_texts(["plot"], **options)
            client.assert_not_called()
