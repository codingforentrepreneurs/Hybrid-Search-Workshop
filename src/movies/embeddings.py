"""OpenRouter or local FastEmbed embeddings with complete long-text coverage."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import lru_cache, partial
from pathlib import Path

import httpx
from django.conf import settings

CHUNK_BYTES = 6_000
LOCAL_CHUNK_BYTES = 400
REQUEST_BATCH_SIZE = 16
PREPROCESSING_VERSION = "utf8-byte-weighted-mean-l2-v2"


class EmbeddingError(Exception):
    """A safe-to-display embedding configuration or provider failure."""


def _setting(name: str, default=None):
    if settings.configured:
        return getattr(settings, name, default)
    return os.environ.get(name, default)


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model: str
    dimensions: int

    @property
    def model_id(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def chunk_bytes(self) -> int:
        return LOCAL_CHUNK_BYTES if self.provider == "fastembed" else CHUNK_BYTES


def get_embedding_config() -> EmbeddingConfig:
    provider = str(_setting("EMBEDDING_PROVIDER", "auto")).strip().lower()
    if provider == "auto":
        provider = "openrouter" if str(_setting("OPENROUTER_API_KEY", "")).strip() else "fastembed"
    if provider == "openrouter":
        model = str(_setting("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"))
        dimensions = int(_setting("OPENROUTER_EMBEDDING_DIMENSIONS", 1536))
    elif provider == "fastembed":
        model = str(_setting("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5"))
        dimensions = int(_setting("FASTEMBED_DIMENSIONS", 384))
    else:
        raise EmbeddingError("EMBEDDING_PROVIDER must be auto, openrouter, or fastembed.")
    if dimensions <= 0 or not model.strip():
        raise EmbeddingError("Configure a nonempty embedding model and positive dimensions.")
    return EmbeddingConfig(provider, model, dimensions)


embedding_config = get_embedding_config


def _configuration(model: str | None = None) -> EmbeddingConfig:
    config = get_embedding_config()
    if model is not None:
        if ":" in model:
            provider, model = model.split(":", 1)
            if provider != config.provider:
                raise EmbeddingError("Requested embedding provider differs from the active provider; re-embed for the active model.")
        config = replace(config, model=model)
    return config


def embedding_model() -> str:
    """Provider-qualified identity for storage and resumable work."""
    return get_embedding_config().model_id


def text_hash(text: str, *, model: str | None = None) -> str:
    """Include the model, dimensions, and preprocessing so changes invalidate work."""
    config = _configuration(model)
    identity = json.dumps(
        [PREPROCESSING_VERSION, config.model_id, config.dimensions, config.chunk_bytes, text],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def chunk_text(text: str, *, max_bytes: int = CHUNK_BYTES) -> list[str]:
    """Keep every Unicode character in bounded UTF-8 chunks.

    The OpenRouter default uses 6,000 bytes for an 8,192-token model; the local
    BGE model uses 400 bytes for its 512-token limit, leaving space for query
    instructions and special tokens. Chunk vectors are byte-weighted mean pooled
    and normalized, giving every part of a long plot a contribution.
    """
    if not isinstance(text, str) or not text.strip():
        raise EmbeddingError("Embedding input must contain nonempty text.")
    if max_bytes < 4:
        raise ValueError("A UTF-8 chunk must allow at least four bytes.")
    encoded = text.encode("utf-8")
    chunks = []
    offset = 0
    while offset < len(encoded):
        chunk = encoded[offset : offset + max_bytes].decode("utf-8", errors="ignore")
        chunks.append(chunk)
        offset += len(chunk.encode("utf-8"))
    return chunks


def _vectors(payload, *, count: int, dimensions: int) -> list[list[float]]:
    try:
        data = payload["data"]
        if not isinstance(data, list) or len(data) != count:
            raise ValueError
        ordered: list[list[float] | None] = [None] * count
        for item in data:
            index = item["index"]
            values = item["embedding"]
            if (
                type(index) is not int or not 0 <= index < count or ordered[index] is not None
                or not isinstance(values, list) or len(values) != dimensions
                or any(type(value) not in (float, int) or not math.isfinite(value) for value in values)
            ):
                raise ValueError
            ordered[index] = [float(value) for value in values]
        if any(value is None for value in ordered):
            raise ValueError
        return ordered
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise EmbeddingError(
            f"OpenRouter returned invalid embeddings; expected {count} vectors of {dimensions} finite numbers."
        ) from exc


def _request(client: httpx.Client, inputs: list[str], model: str, dimensions: int) -> list[list[float]]:
    response = None
    for attempt in range(4):
        try:
            response = client.post(
                "/embeddings",
                json={"model": model, "input": inputs, "encoding_format": "float", "dimensions": dimensions},
            )
        except httpx.TransportError as exc:
            if attempt == 3:
                raise EmbeddingError("Could not reach OpenRouter after four attempts; try again later.") from exc
            time.sleep(min(2 ** attempt, 8))
            continue
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == 3:
                raise EmbeddingError("OpenRouter is rate limited or temporarily unavailable; rerun to resume.")
            try:
                delay = float(response.headers.get("Retry-After", 2 ** attempt))
                if not math.isfinite(delay):
                    raise ValueError
            except (TypeError, ValueError):
                delay = 2 ** attempt
            time.sleep(min(max(delay, 0), 8))
            continue
        if response.status_code in (401, 403):
            raise EmbeddingError("OpenRouter rejected the credentials; check OPENROUTER_API_KEY.")
        if response.status_code == 402:
            raise EmbeddingError("OpenRouter requires available account credits to generate embeddings.")
        if not response.is_success:
            raise EmbeddingError(
                f"OpenRouter rejected the embedding request (HTTP {response.status_code}); check the model and dimensions."
            )
        try:
            return _vectors(response.json(), count=len(inputs), dimensions=dimensions)
        except (ValueError, TypeError) as exc:
            raise EmbeddingError("OpenRouter returned an unreadable embedding response.") from exc
    raise EmbeddingError("OpenRouter did not return embeddings.")


@lru_cache(maxsize=2)
def _local_model(model: str, cache_dir: str):
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise EmbeddingError("Install local embeddings with `uv add fastembed==0.9.0`, then retry.") from exc
    try:
        return TextEmbedding(model_name=model, cache_dir=cache_dir)
    except Exception as exc:
        raise EmbeddingError(
            "FastEmbed could not load its model. The first run needs internet access to download "
            "the model into data/models; later runs can reuse the cache."
        ) from exc


def _local_vectors(inputs: list[str], config: EmbeddingConfig, input_type: str) -> list[list[float]]:
    cache_dir = str(_setting("FASTEMBED_CACHE_DIR", Path("data/models")))
    local_model = _local_model(config.model, cache_dir)
    try:
        embed = local_model.query_embed if input_type == "query" else local_model.passage_embed
        data = [
            {"index": index, "embedding": vector.tolist()}
            for index, vector in enumerate(embed(inputs))
        ]
        return _vectors({"data": data}, count=len(inputs), dimensions=config.dimensions)
    except EmbeddingError as exc:
        raise EmbeddingError("FastEmbed returned vectors that do not match the configured dimensions.") from exc
    except Exception as exc:
        raise EmbeddingError("FastEmbed could not embed this text; verify the local model configuration.") from exc


def embed_texts(
    texts: list[str], *, model: str | None = None, input_type: str = "document",
    concurrency: int = 1, request_batch_size: int = REQUEST_BATCH_SIZE,
) -> list[list[float]]:
    """Return normalized vectors in input order, with bounded OpenRouter concurrency.

    Request scheduling does not change chunking, pooling, or the resume hash.
    FastEmbed keeps its own local batching; concurrency applies only to HTTP.
    """
    if not 1 <= concurrency <= 32 or not 1 <= request_batch_size <= 128:
        raise EmbeddingError("Embedding concurrency must be 1–32 and request batch size 1–128.")
    if not texts:
        return []
    if input_type not in {"document", "query"}:
        raise EmbeddingError("Embedding input_type must be document or query.")
    config = _configuration(model)
    dimensions = config.dimensions
    groups = [chunk_text(text, max_bytes=config.chunk_bytes) for text in texts]
    inputs = [chunk for group in groups for chunk in group]
    vectors = []
    if config.provider == "fastembed":
        vectors = _local_vectors(inputs, config, input_type)
    else:
        key = str(_setting("OPENROUTER_API_KEY", "")).strip()
        if not key:
            raise EmbeddingError("Set OPENROUTER_API_KEY, or set EMBEDDING_PROVIDER=auto to use FastEmbed locally.")
        base_url = str(_setting("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/")
        with httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {key}"},
            timeout=float(_setting("OPENROUTER_TIMEOUT", 60)),
        ) as client:
            batches = (inputs[start : start + request_batch_size]
                       for start in range(0, len(inputs), request_batch_size))
            request = partial(_request, client, model=config.model, dimensions=dimensions)
            if concurrency == 1 or len(inputs) <= request_batch_size:
                for batch in batches:
                    vectors.extend(request(batch))
            else:
                # HTTPX clients share their connection pool safely across threads.
                # map returns in input order even when requests finish out of order;
                # buffersize bounds outstanding work if a request fails.
                with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="embeddings") as pool:
                    for batch_vectors in pool.map(request, batches, buffersize=concurrency):
                        vectors.extend(batch_vectors)
    result = []
    offset = 0
    for chunks in groups:
        pooled = [0.0] * dimensions
        for chunk in chunks:
            weight = len(chunk.encode("utf-8"))
            for index, value in enumerate(vectors[offset]):
                pooled[index] += value * weight
            offset += 1
        norm = math.sqrt(math.fsum(value * value for value in pooled))
        if not math.isfinite(norm) or norm == 0:
            raise EmbeddingError("The embedding provider returned vectors that cannot be normalized.")
        result.append([value / norm for value in pooled])
    return result
