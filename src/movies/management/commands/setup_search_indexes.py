from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import DatabaseError, connection
from psycopg import sql

from movies.embeddings import EmbeddingError, get_embedding_config
from movies.models import Movie
from movies.search import vector_column_name, vector_index_name


class Command(BaseCommand):
    help = "Create the pg_textsearch BM25 and pgvectorscale StreamingDiskANN indexes after loading movies."

    def handle(self, *args, **options):
        if connection.vendor != "postgresql":
            raise CommandError("Set DATABASE_URL to your Tiger Cloud PostgreSQL service first.")
        table = sql.Identifier(Movie._meta.db_table)
        try:
            config = get_embedding_config()
            dimensions = int(config.dimensions)
            if not 1 <= dimensions <= 16000:
                raise CommandError("Embedding dimensions must be between 1 and 16000.")
            with connection.cursor() as cursor:
                self.stdout.write("Preparing the model-specific vector column…")
                # DiskANN rescoring accesses a physical heap attribute. A stored
                # generated column provides one while the base embedding field
                # continues to accept either provider's dimension count.
                # CASE guards the cast when re-embedding rows for another model.
                vector_column = sql.Identifier(vector_column_name(config))
                cursor.execute(sql.SQL(
                    f"ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS {{column}} vector({dimensions}) "
                    f"GENERATED ALWAYS AS (CASE "
                    f"WHEN embedding_model = {{model_id}} AND vector_dims(embedding) = {dimensions} "
                    f"THEN embedding::vector({dimensions}) ELSE NULL END) STORED"
                ).format(table=table, column=vector_column, model_id=sql.Literal(config.model_id)))
                self.stdout.write("Creating the BM25 index…")
                cursor.execute(sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {index} ON {table} "
                    "USING bm25 (search_text) WITH (text_config = 'english')"
                ).format(index=sql.Identifier(settings.SEARCH_BM25_INDEX), table=table))
                self.stdout.write("Creating the StreamingDiskANN index…")
                cursor.execute(sql.SQL(
                    f"CREATE INDEX IF NOT EXISTS {{index}} "
                    f"ON {{table}} USING diskann ({{column}} vector_cosine_ops) "
                    f"WHERE {{column}} IS NOT NULL"
                ).format(index=sql.Identifier(vector_index_name(config)), table=table, column=vector_column))
                cursor.execute(sql.SQL("ANALYZE {table}").format(table=table))
        except EmbeddingError as exc:
            raise CommandError(str(exc)) from None
        except DatabaseError:
            raise CommandError(
                "Could not create the search indexes. Check the database connection, "
                "run migrations, and confirm pg_textsearch and vectorscale are available."
            ) from None
        self.stdout.write(self.style.SUCCESS("BM25 and StreamingDiskANN indexes are ready."))
