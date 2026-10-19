from django.db import models

from .fields import VectorField


class Movie(models.Model):
    wikipedia_id = models.BigIntegerField(primary_key=True)
    title = models.TextField()
    plot = models.TextField()
    release_year = models.IntegerField(null=True, blank=True, db_index=True)
    release_date = models.TextField(blank=True)
    genres = models.JSONField(default=list)
    languages = models.JSONField(default=list)
    countries = models.JSONField(default=list)
    runtime_minutes = models.FloatField(null=True, blank=True)
    wikipedia_url = models.URLField(max_length=500)
    search_text = models.TextField()
    embedding = VectorField(null=True, blank=True)
    embedding_model = models.TextField(blank=True)
    embedding_hash = models.TextField(blank=True)

    class Meta:
        ordering = ["wikipedia_id"]

    def __str__(self):
        return self.title
