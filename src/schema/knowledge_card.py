"""
Knowledge Card : une procedure de l'hotel, capturee aupres du personnel.

Ces fiches sont ecrites a la main (ou dictees a un LLM interviewer) puis
relues. Seules celles en status=verified entrent dans le RAG.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .document import CanonicalDocument


class KnowledgeCard(BaseModel):
    id: str
    title: str
    domain: Literal["hotel"] = "hotel"
    category: str  # ex: checkout
    subcategory: str | None = None  # ex: late_checkout

    situation: str  # quand cette fiche s'applique
    procedure: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    escalation: list[str] = Field(default_factory=list)

    status: Literal["draft", "review_required", "verified", "deprecated"] = "draft"
    language: Literal["fr", "en"] = "fr"
    last_reviewed: str | None = None
    review_questions: list[str] = Field(default_factory=list)

    @field_validator("last_reviewed", mode="before")
    @classmethod
    def _normalise_review_date(cls, value: object) -> object:
        """PyYAML convertit automatiquement une date ISO non quotee."""

        return value.isoformat() if isinstance(value, date) else value

    @model_validator(mode="after")
    def _verified_has_no_open_question(self) -> "KnowledgeCard":
        if self.status in {"review_required", "verified"} and not self.procedure:
            raise ValueError(
                "Une Knowledge Card review_required ou verified doit avoir une procedure"
            )
        if self.status == "verified" and self.review_questions:
            raise ValueError(
                "Une Knowledge Card verified ne peut pas garder de question ouverte"
            )
        return self

    def to_document(self) -> CanonicalDocument:
        """Une fiche = un document. On ne la decoupe pas : une procedure
        doit rester entiere pour rester comprehensible."""
        blocks = [f"Situation : {self.situation}", "Procedure :"]
        blocks += [f"{i}. {s}" for i, s in enumerate(self.procedure, 1)]
        if self.exceptions:
            blocks.append("Exceptions :")
            blocks += [f"- {e}" for e in self.exceptions]
        if self.escalation:
            blocks.append("Escalade :")
            blocks += [f"- {e}" for e in self.escalation]

        return CanonicalDocument(
            id=CanonicalDocument.make_id(self.id, "card"),
            content="\n".join(blocks),
            title=self.title,
            breadcrumb=f"{self.category} > {self.subcategory or self.title} > {self.title}",
            domain="hotel",
            topic=self.category,
            subtopic=self.subcategory,
            source_type="internal_sop",
            source_id=self.id,
            authority="verified_internal",
            status=self.status,
            language=self.language,
            last_reviewed=self.last_reviewed,
        )
