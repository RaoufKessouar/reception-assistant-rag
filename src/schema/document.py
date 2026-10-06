"""
Le Document Canonique.

C'est LE fichier central du projet. Toutes les sources - documentation,
Q/R, video, procedures hotel - finissent transformees en objets de cette
classe. Le moteur de recherche ne connait que ca.

Regle : un seul modele avec des champs optionnels, jamais un modele par
source. C'est ce qui rend le moteur independant du logiciel de gestion hôtelière.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Domain = Literal["software", "hotel"]
SourceType = Literal["official_doc", "qa", "video", "internal_sop"]
Authority = Literal["official", "verified_internal", "community"]
Status = Literal["draft", "review_required", "verified", "deprecated"]
Language = Literal["fr", "en"]
ValidationMethod = Literal["human_individual", "batch_sampled", "automatic"]


class CanonicalDocument(BaseModel):
    """Un chunk pret a etre indexe."""

    # ─── Contenu ───
    id: str
    content: str = Field(min_length=1)
    title: str | None = None
    breadcrumb: str | None = None
    # breadcrumb = "Reservations > Modifier > Changer les dates"
    # On le colle en tete du contenu avant d'embedder : gain de recherche
    # important pour un cout nul (le chunk devient auto-suffisant).

    # ─── Routage (sert au filtrage et au query router) ───
    domain: Domain
    topic: str
    subtopic: str | None = None

    # ─── Provenance ───
    source_type: SourceType
    source_id: str
    authority: Authority
    source_category: str | None = None
    url: str | None = None
    source_path: str | None = None
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)

    # ─── Cycle de vie ───
    status: Status = "draft"
    language: Language = "fr"
    version: str | None = None
    last_reviewed: str | None = None
    validation_method: ValidationMethod | None = None
    validation_batch_id: str | None = None
    validation_model: str | None = None
    content_hash: str | None = None

    # ─── Chunking ───
    # Un document peut rester entier (1/1) ou produire plusieurs chunks.
    parent_document_id: str | None = None
    chunk_index: int | None = Field(default=None, ge=1)
    chunk_count: int | None = Field(default=None, ge=1)

    # ─── Specifique video ───
    timestamp_start: float | None = None
    timestamp_end: float | None = None
    screenshot: str | None = None
    screenshots: list[str] = Field(default_factory=list)

    # ─── Identifiant de la plateforme de gestion hôtelière ───
    software: str | None = None

    # ─── Specifique Hôtel ───
    hotel_id: str | None = None

    @field_validator("content")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def _valid_page_range(self) -> "CanonicalDocument":
        if self.page_end is not None and self.page_start is None:
            raise ValueError("page_start est requis lorsque page_end est renseigne")
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError("page_end doit etre superieur ou egal a page_start")
        chunk_fields = (
            self.parent_document_id,
            self.chunk_index,
            self.chunk_count,
        )
        if any(value is not None for value in chunk_fields) and any(
            value is None for value in chunk_fields
        ):
            raise ValueError(
                "parent_document_id, chunk_index et chunk_count doivent etre renseignes ensemble"
            )
        if (
            self.chunk_index is not None
            and self.chunk_count is not None
            and self.chunk_index > self.chunk_count
        ):
            raise ValueError("chunk_index ne peut pas depasser chunk_count")
        return self

    # ─── Utilitaires ───

    def text_to_embed(self) -> str:
        """Texte reellement envoye au modele d'embedding."""
        parts = []
        if self.breadcrumb:
            parts.append(self.breadcrumb)
        elif self.title:
            parts.append(self.title)
        parts.append(self.content)
        return "\n\n".join(parts)

    def citation(self) -> str:
        """Ligne de source affichee sous la reponse."""
        if self.source_type == "video" and self.timestamp_start is not None:
            mm, ss = divmod(int(self.timestamp_start), 60)
            return f"Tutoriel {self.software or ''} - {self.title or self.source_id} ({mm:02d}:{ss:02d})"
        if self.source_type == "internal_sop":
            return f"Procedure interne - {self.title or self.source_id}"
        label = f"Documentation {self.software or ''} - {self.title or self.source_id}"
        if self.page_start is not None:
            page = (
                f"p. {self.page_start}"
                if self.page_end in (None, self.page_start)
                else f"p. {self.page_start}-{self.page_end}"
            )
            label = f"{label} ({page})"
        return label.strip()

    @staticmethod
    def make_id(source_id: str, chunk_key: str) -> str:
        """Identifiant stable : re-indexer ne cree pas de doublons."""
        raw = f"{source_id}::{chunk_key}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def qdrant_id(self) -> str:
        """UUID stable accepte par Qdrant, quel que soit le format de ``id``."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"reception-rag:{self.id}"))
