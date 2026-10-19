from django.core.exceptions import ValidationError
from django.db import DatabaseError, connection
from django.db.models import Count, Func, IntegerField, Q
from django.shortcuts import render
from django.views.decorators.http import require_GET

from .embeddings import EmbeddingError
from .models import Movie
from .search import SearchUnavailable, check_search_ready, empty_columns, search_movies


@require_GET
def index(request):
    query = request.GET.get("q", "").strip()
    genre = request.GET.get("genre", "").strip()
    year_text = request.GET.get("year", "").strip()
    context = {
        "query": query,
        "selected_genre": genre,
        "selected_year": year_text,
        "genres": [],
        "years": [],
        "movie_count": 0,
        "embedded_count": 0,
        "columns": empty_columns(),
        "error": "",
        "notices": [],
        "searched": bool(query),
        "timings": {},
        "database_ready": False,
    }
    if len(query) > 1000:
        context["error"] = "Please keep the search query to 1,000 characters or fewer."
    elif len(genre) > 200:
        context["error"] = "Please choose a genre from the filter."
    year = None
    if year_text:
        try:
            year = int(year_text)
            if not 1800 <= year <= 2100:
                raise ValueError
        except ValueError:
            context["error"] = "Please enter a release year between 1800 and 2100."

    if connection.vendor != "postgresql":
        context["notices"].append(
            "Set DATABASE_URL to your Tiger Cloud connection, run migrations, and load movies to enable search."
        )
        if query and not context["error"]:
            context["error"] = "Search requires your Tiger Cloud database. Finish setup to compare real results."
        return render(request, "movies/index.html", context)

    try:
        from .embeddings import get_embedding_config

        embedding_config = get_embedding_config()
        counts = Movie.objects.annotate(
            embedding_dimensions=Func("embedding", function="vector_dims", output_field=IntegerField())
        ).aggregate(
            total=Count("wikipedia_id"),
            embedded=Count("wikipedia_id", filter=Q(
                embedding__isnull=False, embedding_model=embedding_config.model_id,
                embedding_dimensions=embedding_config.dimensions,
            )),
        )
        context["movie_count"] = counts["total"]
        context["embedded_count"] = counts["embedded"]
        with connection.cursor() as cursor:
            table = connection.ops.quote_name(Movie._meta.db_table)
            cursor.execute(f"SELECT DISTINCT jsonb_array_elements_text(genres) AS genre FROM {table} ORDER BY genre")
            context["genres"] = [row[0] for row in cursor.fetchall()]
        context["years"] = list(
            Movie.objects.exclude(release_year=None)
            .order_by("-release_year").values_list("release_year", flat=True).distinct()
        )
        check_search_ready()
        context["database_ready"] = True
        if not counts["total"]:
            context["notices"].append("The database is ready. Load the CMU movie corpus to start searching.")
        elif counts["embedded"] < counts["total"]:
            context["notices"].append(
                f"{counts['embedded']:,} of {counts['total']:,} movies have embeddings for the selected model. "
                "Semantic search covers only those movies; finish embedding for a complete comparison."
            )
        if query and not context["error"]:
            if not counts["embedded"]:
                context["error"] = "Load movies and generate their embeddings before comparing the three search methods."
            else:
                context.update(search_movies(query, genre=genre, year=year))
    except SearchUnavailable as exc:
        context["notices"].append(str(exc))
        if query and not context["error"]:
            context["error"] = str(exc)
    except DatabaseError:
        context["database_ready"] = False
        context["error"] = (
            "The movie database is unavailable or needs setup. Check DATABASE_URL, "
            "run migrations, and create the search indexes."
        )
    except ValidationError:
        context["error"] = "The embedding response did not match the configured vector dimensions."
    except EmbeddingError as exc:
        context["error"] = str(exc)
    return render(request, "movies/index.html", context)
