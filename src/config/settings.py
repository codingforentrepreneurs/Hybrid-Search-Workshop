"""Local demo settings. Credentials are read from the environment or .env."""
import os
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlsplit

from django.core.exceptions import ImproperlyConfigured
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")

DEBUG = os.environ.get("DJANGO_DEBUG", "true").lower() in {"1", "true", "yes"}
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "local-movie-search-demo-only")
if not DEBUG and SECRET_KEY in {"local-movie-search-demo-only", "local-development-only-change-before-deployment"}:
    raise ImproperlyConfigured("Set DJANGO_SECRET_KEY when DJANGO_DEBUG is false.")
ALLOWED_HOSTS = [
    value.strip()
    for value in os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1],testserver").split(",")
    if value.strip()
]

INSTALLED_APPS = ["django.contrib.contenttypes", "django.contrib.staticfiles", "movies"]
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]
ROOT_URLCONF = "config.urls"
TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "DIRS": [BASE_DIR / "templates"],
    "APP_DIRS": True,
    "OPTIONS": {"context_processors": ["django.template.context_processors.request"]},
}]
WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

database_url = os.environ.get("DATABASE_URL", "").strip()
if database_url:
    try:
        parsed = urlsplit(database_url)
        if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
            raise ValueError
        database_name = unquote(parsed.path.lstrip("/"))
        if not database_name:
            raise ValueError
        options = dict(parse_qsl(parsed.query))
        options.setdefault("sslmode", "require")
        options.setdefault("connect_timeout", "10")
        DATABASES = {"default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": database_name,
            "USER": unquote(parsed.username or ""),
            "PASSWORD": unquote(parsed.password or ""),
            "HOST": parsed.hostname,
            "PORT": parsed.port or 5432,
            "OPTIONS": options,
            "CONN_MAX_AGE": 60,
            "CONN_HEALTH_CHECKS": True,
        }}
    except ValueError:
        raise ImproperlyConfigured("DATABASE_URL must be a PostgreSQL connection URL.") from None
else:
    # This database supports setup screens and offline loader checks only.
    # All actual retrieval requires Tiger Cloud's PostgreSQL extensions.
    DATABASES = {"default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": PROJECT_ROOT / "db.sqlite3",
    }}

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = False
USE_TZ = True
STATIC_URL = "static/"
STATIC_ROOT = PROJECT_ROOT / "staticfiles"
STATICFILES_DIRS = [BASE_DIR / "static"] if (BASE_DIR / "static").exists() else []
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_EMBEDDING_MODEL = os.environ.get("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
OPENROUTER_EMBEDDING_DIMENSIONS = int(os.environ.get("OPENROUTER_EMBEDDING_DIMENSIONS", "1536"))
OPENROUTER_TIMEOUT = float(os.environ.get("OPENROUTER_TIMEOUT", "60"))
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "auto")
FASTEMBED_MODEL = os.environ.get("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
FASTEMBED_DIMENSIONS = int(os.environ.get("FASTEMBED_DIMENSIONS", "384"))
FASTEMBED_CACHE_DIR = os.environ.get("FASTEMBED_CACHE_DIR", str(PROJECT_ROOT / "data" / "models"))

SEARCH_RESULT_LIMIT = 10
SEARCH_CANDIDATE_LIMIT = 100
SEARCH_RRF_K = 60
SEARCH_BM25_INDEX = "movies_search_text_bm25_idx"
