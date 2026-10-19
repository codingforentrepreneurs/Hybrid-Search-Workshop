"""Offline test configuration, independent of the configured Tiger Cloud database."""

from .settings import *  # noqa: F403

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}
OPENROUTER_API_KEY = ""
EMBEDDING_PROVIDER = "fastembed"
FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
FASTEMBED_DIMENSIONS = 384
