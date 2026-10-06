"""Export portable des seules étapes vidéo au statut ``verified``."""

from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .....config import path
from .....schema import CanonicalDocument, ProceduralStep
from .common import (
    artifact_path,
    atomic_write_json,
    atomic_write_jsonl,
    project_relative,
)


def _archive_name(file: Path, project_root: Path) -> str:
    """Nom portable, y compris quand ``data`` pointe vers un autre disque."""

    resolved = file.resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        data_root = path("data").resolve()
        try:
            relative = resolved.relative_to(data_root)
        except ValueError as error:
            raise ValueError(f"Fichier d'archive hors projet : {resolved}") from error
        return (Path("data") / relative).as_posix()


def _all_steps() -> list[ProceduralStep]:
    steps = []
    for file in sorted(path("data/interim/videos").glob("*/steps.json")):
        payload = json.loads(file.read_text(encoding="utf-8"))
        rows = payload.get("steps", []) if isinstance(payload, dict) else payload
        steps.extend(ProceduralStep(**row) for row in rows)
    return steps


def _copy_screenshot(value: str | None, video_id: str) -> str | None:
    if not value:
        return None
    source = artifact_path(value)
    if not source.exists():
        raise FileNotFoundError(f"Screenshot valide introuvable : {source}")
    destination = path(f"data/processed/video_screenshots/{video_id}/{source.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return project_relative(destination)


def export_verified() -> tuple[Path, list[CanonicalDocument], dict]:
    """Reconstruit integralement l'export pour supprimer les etapes obsoletes."""
    all_steps = _all_steps()
    verified = [step for step in all_steps if step.status == "verified"]
    by_source: dict[str, list[ProceduralStep]] = {}
    for step in verified:
        by_source.setdefault(step.video_id, []).append(step)

    ordered_steps: list[ProceduralStep] = []
    for video_id in sorted(by_source):
        source_steps = sorted(
            by_source[video_id],
            key=lambda item: (item.timestamp_start, item.step_number),
        )
        total = len(source_steps)
        ordered_steps.extend(
            step.model_copy(update={"step_number": number, "total_steps": total})
            for number, step in enumerate(source_steps, 1)
        )

    documents = []
    for step in ordered_steps:
        document = step.to_document()
        screenshot = _copy_screenshot(document.screenshot, step.video_id)
        screenshots = [
            copied
            for value in document.screenshots
            if (copied := _copy_screenshot(value, step.video_id))
        ]
        documents.append(
            document.model_copy(
                update={"screenshot": screenshot, "screenshots": screenshots}
            )
        )

    documents.sort(
        key=lambda item: (item.source_id, item.timestamp_start or 0.0, item.id)
    )
    output = path("data/processed/software_videos.jsonl")
    atomic_write_jsonl(
        output, (document.model_dump(mode="json") for document in documents)
    )

    by_video: dict[str, int] = {}
    for document in documents:
        by_video[document.source_id] = by_video.get(document.source_id, 0) + 1
    report = {
        "schema_version": 1,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "documents": len(documents),
        "videos": by_video,
        "source_statuses": {
            status: sum(step.status == status for step in all_steps)
            for status in ("verified", "review_required", "deprecated")
        },
    }
    atomic_write_json(path("data/processed/video_export_report.json"), report)
    return output, documents, report


def package_verified(
    output: Path | None = None,
    prepared: tuple[Path, list[CanonicalDocument], dict] | None = None,
) -> Path:
    """Crée l'archive, en réutilisant l'export déjà préparé quand il est fourni."""

    jsonl, documents, report = prepared or export_verified()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    archive = output or path(f"data/exports/reception-rag-video-results-{timestamp}.zip")
    archive.parent.mkdir(parents=True, exist_ok=True)

    video_ids = set(report["videos"])
    files: set[Path] = {jsonl, path("data/processed/video_export_report.json")}
    manifest = path("data/interim/video_manifest.json")
    if manifest.exists():
        files.add(manifest)
    for document in documents:
        if document.screenshot:
            files.add(artifact_path(document.screenshot))
        files.update(artifact_path(value) for value in document.screenshots)
    for video_id in video_ids:
        for name in (
            "transcript.json",
            "keyframes.json",
            "descriptions.json",
            "steps.json",
        ):
            candidate = path(f"data/interim/videos/{video_id}/{name}")
            if candidate.exists():
                files.add(candidate)
        files.update(path(f"data/interim/videos/{video_id}/keyframes").glob("*.png"))

    project_root = path("").resolve()
    temporary = archive.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for file in sorted(files, key=lambda item: item.as_posix()):
            bundle.write(file, arcname=_archive_name(file, project_root))
    temporary.replace(archive)
    return archive
