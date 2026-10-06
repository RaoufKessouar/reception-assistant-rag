"""Workflow de revue humaine des etapes video."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

from .....schema import ProceduralStep
from .common import atomic_write_json
from .s4_assemble import load, output_file

EDITABLE_FIELDS = {
    "instruction",
    "action_type",
    "action_target",
    "action_location",
    "screen_before",
    "screen_after",
    "visual_description",
    "screenshot_before",
    "screenshot_after",
    "supporting_screenshots",
}


def update_status(
    video_id: str,
    status: str,
    reviewer: str | None = None,
    step_numbers: set[int] | None = None,
    notes: str | None = None,
    allow_quality_flags: bool = False,
) -> list[ProceduralStep]:
    if status not in {"review_required", "verified", "deprecated"}:
        raise ValueError(f"Statut de revue invalide : {status}")
    if status == "verified" and not reviewer:
        raise ValueError("--reviewer est obligatoire pour approuver une etape")

    steps = load(video_id)
    selected = step_numbers or {step.step_number for step in steps}
    unknown = selected - {step.step_number for step in steps}
    if unknown:
        raise ValueError(f"Etapes inconnues : {sorted(unknown)}")

    now = datetime.now(timezone.utc)
    updated = []
    for step in steps:
        if step.step_number not in selected:
            updated.append(step)
            continue
        if status == "verified" and step.quality_flags and not allow_quality_flags:
            raise ValueError(
                f"Etape {step.step_number} avec alertes {step.quality_flags}. "
                "Corrige steps.json ou utilise --allow-quality-flags apres verification visuelle."
            )
        values = step.model_dump()
        values.update(
            status=status,
            reviewed_by=reviewer if status == "verified" else None,
            reviewed_at=now if status == "verified" else None,
            review_notes=notes,
            validation_method="human_individual" if status == "verified" else None,
            validation_batch_id=None,
            validation_model=None,
        )
        updated.append(ProceduralStep(**values))

    atomic_write_json(
        output_file(video_id),
        {
            "schema_version": 2,
            "video_id": video_id,
            "steps": [step.model_dump(mode="json") for step in updated],
        },
    )
    return updated


def summary(video_id: str) -> dict:
    steps = load(video_id)
    counts: dict[str, int] = {}
    flagged = 0
    for step in steps:
        counts[step.status] = counts.get(step.status, 0) + 1
        flagged += bool(step.quality_flags)
    return {
        "video_id": video_id,
        "steps": len(steps),
        "by_status": counts,
        "with_quality_flags": flagged,
    }


def apply_decisions(file: Path) -> list[dict]:
    """Applique des corrections humaines rejouables stockées en JSONL."""

    decisions = [
        json.loads(line)
        for line in file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped: dict[str, list[dict]] = {}
    for decision in decisions:
        grouped.setdefault(str(decision["video_id"]), []).append(decision)

    applied = []
    for video_id, video_decisions in grouped.items():
        steps = load(video_id)
        by_number = {step.step_number: step for step in steps}
        for decision in video_decisions:
            number = int(decision["step_number"])
            if number not in by_number:
                raise ValueError(f"Etape {number} absente de {video_id}")
            changes = dict(decision.get("changes", {}))
            unknown = set(changes) - EDITABLE_FIELDS
            if unknown:
                raise ValueError(f"Champs de correction interdits : {sorted(unknown)}")
            values = by_number[number].model_dump()
            values.update(changes)
            status = decision.get("status", "verified")
            reviewer = decision.get("reviewed_by")
            if status == "verified" and not reviewer:
                raise ValueError(f"Reviewer absent pour {video_id}, étape {number}")
            values.update(
                status=status,
                reviewed_by=reviewer if status == "verified" else None,
                reviewed_at=(
                    decision.get("reviewed_at") or datetime.now(timezone.utc)
                    if status == "verified"
                    else None
                ),
                review_notes=decision.get("review_notes"),
                validation_method=(
                    decision.get("validation_method", "human_individual")
                    if status == "verified"
                    else None
                ),
                validation_batch_id=(
                    decision.get("validation_batch_id")
                    if status == "verified"
                    else None
                ),
                validation_model=(
                    decision.get("validation_model")
                    if status == "verified"
                    else None
                ),
            )
            by_number[number] = ProceduralStep(**values)
            applied.append({"video_id": video_id, "step_number": number, "status": status})

        ordered = [by_number[step.step_number] for step in steps]
        atomic_write_json(
            output_file(video_id),
            {
                "schema_version": 2,
                "video_id": video_id,
                "steps": [step.model_dump(mode="json") for step in ordered],
            },
        )
    return applied
