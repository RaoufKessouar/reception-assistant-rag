"""
Modeles du pipeline video.

Une etape par action, jamais un decoupage tous les N caracteres.
C'est le différenciant multimodal du projet.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .document import CanonicalDocument, Status


ValidationMethod = Literal["human_individual", "batch_sampled", "automatic"]


class TranscriptWord(BaseModel):
    """Mot aligne par WhisperX. Certains mots ne recoivent pas de timestamp."""

    text: str
    start: float | None = None
    end: float | None = None
    score: float | None = None


class TranscriptSegment(BaseModel):
    """Sortie de l'etape 1 (WhisperX)."""

    start: float
    end: float
    text: str
    words: list[TranscriptWord] = Field(default_factory=list)

    @model_validator(mode="after")
    def _valid_interval(self):
        if self.start < 0 or self.end < self.start:
            raise ValueError("Intervalle de transcription invalide")
        return self


class ScreenAnalysis(BaseModel):
    """
    Sortie de l'etape 3 (le VLM regarde une capture d'ecran).

    Ce schema est envoye tel quel a vLLM en guided decoding : le modele
    ne PEUT PAS produire autre chose. Les champs Optional valent None
    quand il n'est pas sur - on lui interdit d'inventer un nom de bouton.
    """

    model_config = ConfigDict(extra="forbid")

    screen_name: str | None = Field(None, description="Nom de l'ecran affiche")
    ui_element: str | None = Field(None, description="Libelle exact de l'element vise")
    location: str | None = Field(None, description="Position a l'ecran")
    action_type: (
        Literal["click", "type", "select", "scroll", "hover", "none"] | None
    ) = None
    expected_result: str | None = Field(
        None, description="Ce qui se passe apres l'action"
    )
    visual_description: str | None = Field(
        None, description="Description factuelle de l'ecran"
    )
    confident: bool = Field(
        False, description="True uniquement si le libelle est lisible sans ambiguite"
    )
    uncertainty_reason: str | None = None


class StepDraft(BaseModel):
    """Sortie structuree du LLM d'assemblage avant revue humaine."""

    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(min_length=1)
    action_type: (
        Literal["click", "type", "select", "scroll", "hover", "none"] | None
    ) = None
    action_target: str | None = None
    action_location: str | None = None
    before_state: str | None = None
    after_state: str | None = None
    visual_description: str | None = None
    confident: bool = False
    quality_flags: list[str] = Field(default_factory=list)


class ProceduralStep(BaseModel):
    """Sortie de l'etape 4 : une etape de procedure, prete a indexer."""

    procedure: str  # ex: modifier_les_dates
    step_number: int
    total_steps: int | None = None
    source_segment_index: int | None = None

    timestamp_start: float
    timestamp_end: float

    spoken_instruction: str  # ce qui est dit
    instruction: str  # instruction reecrite, complete et autonome

    screen_before: str | None = None
    screen_after: str | None = None
    action_type: (
        Literal["click", "type", "select", "scroll", "hover", "none"] | None
    ) = None
    action_target: str | None = None
    action_location: str | None = None
    visual_description: str | None = None
    screenshot_before: str | None = None
    screenshot_after: str | None = None
    supporting_screenshots: list[str] = Field(default_factory=list)
    confident: bool = False
    quality_flags: list[str] = Field(default_factory=list)

    video_id: str
    video_title: str | None = None
    video_path: str | None = None
    software: str = "generic_connector"

    status: Status = "review_required"
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    review_notes: str | None = None
    validation_method: ValidationMethod | None = None
    validation_batch_id: str | None = None
    validation_model: str | None = None

    @model_validator(mode="after")
    def _review_and_timestamps_are_valid(self):
        if self.timestamp_start < 0 or self.timestamp_end < self.timestamp_start:
            raise ValueError("Intervalle video invalide")
        if self.status == "verified" and (not self.reviewed_by or not self.reviewed_at):
            raise ValueError(
                "Une etape video verified doit avoir un reviewer et une date"
            )
        return self

    def to_document(self) -> CanonicalDocument:
        blocks = [self.instruction]
        if self.action_target:
            action = f"Action : {self.action_type or 'action'} sur {self.action_target}"
            if self.action_location:
                action += f" ({self.action_location})"
            blocks.append(action)
        if self.screen_before:
            blocks.append(f"Ecran de depart : {self.screen_before}")
        if self.screen_after:
            blocks.append(f"Resultat : {self.screen_after}")
        if self.visual_description:
            blocks.append(f"Preuve visuelle : {self.visual_description}")

        screenshots = list(
            dict.fromkeys(
                value
                for value in [
                    self.screenshot_before,
                    *self.supporting_screenshots,
                    self.screenshot_after,
                ]
                if value
            )
        )
        screenshot = screenshots[0] if screenshots else None

        return CanonicalDocument(
            id=CanonicalDocument.make_id(self.video_id, f"step{self.step_number}"),
            content="\n".join(blocks),
            title=self.video_title or self.procedure,
            breadcrumb=f"{self.procedure} > etape {self.step_number}",
            domain="software",
            topic=self.procedure,
            source_type="video",
            source_id=self.video_id,
            authority="official",
            status=self.status,
            last_reviewed=(
                self.reviewed_at.isoformat() if self.reviewed_at is not None else None
            ),
            validation_method=self.validation_method,
            validation_batch_id=self.validation_batch_id,
            validation_model=self.validation_model,
            timestamp_start=self.timestamp_start,
            timestamp_end=self.timestamp_end,
            screenshot=screenshot,
            screenshots=screenshots,
            source_path=self.video_path,
            software=self.software,
        )
