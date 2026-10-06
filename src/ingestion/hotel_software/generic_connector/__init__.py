from __future__ import annotations

import json

from ....config import path
from ....schema import CanonicalDocument
from ...base import HotelSoftwareKnowledgeSource
from . import docs as _docs
from . import qa as _qa


class GenericHotelSoftwareSource(HotelSoftwareKnowledgeSource):
    name = "generic_connector"

    def load_docs(self) -> list[CanonicalDocument]:
        return _docs.load()

    def load_qa(self) -> list[CanonicalDocument]:
        return _qa.load()

    def load_videos(self) -> list[CanonicalDocument]:
        """Lit la sortie du pipeline video (etape 5)."""
        f = path("data/processed/software_videos.jsonl")
        if not f.exists():
            return []
        return [
            CanonicalDocument(**json.loads(line))
            for line in f.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


__all__ = ["GenericHotelSoftwareSource"]
