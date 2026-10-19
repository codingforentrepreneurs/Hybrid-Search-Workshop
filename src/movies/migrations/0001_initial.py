from django.db import migrations, models

import movies.fields


def enable_extensions(apps, schema_editor):
    if schema_editor.connection.vendor == "postgresql":
        with schema_editor.connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_textsearch")


class Migration(migrations.Migration):
    initial = True
    dependencies = []
    operations = [
        migrations.RunPython(enable_extensions, migrations.RunPython.noop),
        migrations.CreateModel(
            name="Movie",
            fields=[
                ("wikipedia_id", models.BigIntegerField(primary_key=True, serialize=False)),
                ("title", models.TextField()),
                ("plot", models.TextField()),
                ("release_year", models.IntegerField(blank=True, db_index=True, null=True)),
                ("release_date", models.TextField(blank=True)),
                ("genres", models.JSONField(default=list)),
                ("languages", models.JSONField(default=list)),
                ("countries", models.JSONField(default=list)),
                ("runtime_minutes", models.FloatField(blank=True, null=True)),
                ("wikipedia_url", models.URLField(max_length=500)),
                ("search_text", models.TextField()),
                ("embedding", movies.fields.VectorField(blank=True, null=True)),
                ("embedding_model", models.TextField(blank=True)),
                ("embedding_hash", models.TextField(blank=True)),
            ],
            options={"ordering": ["wikipedia_id"]},
        ),
    ]
