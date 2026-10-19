"""Convenience entrypoint: uv run scripts/load_movies.py --prepare-only."""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

from django.core.management import execute_from_command_line

if __name__ == "__main__":
    execute_from_command_line([sys.argv[0], "load_movies", *sys.argv[1:]])
