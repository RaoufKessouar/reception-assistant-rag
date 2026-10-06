"""Normalisation commune de toutes les sources avant le chunking/indexation."""

from .pipeline import load, normalize_document, run

__all__ = ["load", "normalize_document", "run"]
