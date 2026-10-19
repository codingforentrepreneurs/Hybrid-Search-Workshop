import json
import math

from django.core.exceptions import ValidationError
from django.db import models


class VectorField(models.Field):
    """A real pgvector column on PostgreSQL, serializable text in offline SQLite.

    SQLite exists only for the setup page and parser/loader tests. Retrieval
    explicitly refuses it; this fallback does not emulate vector search.
    """

    description = "A PostgreSQL pgvector embedding"

    def __init__(self, *args, dimensions=None, **kwargs):
        self.dimensions = dimensions
        super().__init__(*args, **kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        kwargs["dimensions"] = self.dimensions
        return name, path, args, kwargs

    def db_type(self, connection):
        if connection.vendor == "sqlite":
            return "text"
        return "vector" if self.dimensions is None else f"vector({int(self.dimensions)})"

    def to_python(self, value):
        if value is None:
            return None
        try:
            if isinstance(value, str):
                value = json.loads(value)
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError
            if self.dimensions is not None and len(value) != self.dimensions:
                raise ValueError
            vector = [float(item) for item in value]
            if not all(math.isfinite(item) for item in vector):
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            size = f"{self.dimensions} " if self.dimensions is not None else ""
            raise ValidationError(f"An embedding must contain {size}finite numbers.") from None
        return vector

    def from_db_value(self, value, expression, connection):
        return self.to_python(value)

    def get_prep_value(self, value):
        vector = self.to_python(value)
        return None if vector is None else json.dumps(vector, separators=(",", ":"), allow_nan=False)

    def value_to_string(self, obj):
        return self.get_prep_value(self.value_from_object(obj))
