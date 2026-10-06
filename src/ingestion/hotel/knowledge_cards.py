"""Chargement des fiches de procedures hotel (fichiers YAML)."""

from __future__ import annotations

import yaml

from ...config import path, settings
from ...schema import CanonicalDocument, KnowledgeCard


def load_cards(directory: str = "knowledge/hotel") -> list[KnowledgeCard]:
    cards = []
    for f in sorted(path(directory).glob("*.yaml")):
        if f.name.startswith("_"):  # _TEMPLATE.yaml
            continue
        data = yaml.safe_load(f.read_text(encoding="utf-8"))
        cards.append(KnowledgeCard(**data))
    return cards


def load_documents(directory: str = "knowledge/hotel") -> list[CanonicalDocument]:
    root = path(directory)
    hotel_id = settings().get("hotel", {}).get("id", "hotel_reference")
    documents = []
    for file in sorted(root.glob("*.yaml")):
        if file.name.startswith("_"):
            continue
        card = KnowledgeCard(**yaml.safe_load(file.read_text(encoding="utf-8")))
        documents.append(
            card.to_document().model_copy(
                update={
                    "hotel_id": hotel_id,
                    "source_path": file.relative_to(path("")).as_posix(),
                }
            )
        )
    return documents


def report(directory: str = "knowledge/hotel") -> dict:
    """Combien de fiches, et combien sont reellement utilisables."""
    cards = load_cards(directory)
    counts: dict[str, int] = {}
    for c in cards:
        counts[c.status] = counts.get(c.status, 0) + 1
    return {"total": len(cards), "par_statut": counts}
