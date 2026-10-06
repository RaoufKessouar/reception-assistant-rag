"""Construit un paquet léger pour la revue humaine ciblée des vidéos."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .....config import path
from .....schema import ProceduralStep
from .common import atomic_write_json, read_json


def _asset_path(value: str, project_root: Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else project_root / candidate


def _load_step(videos_root: Path, video_id: str, step_number: int) -> ProceduralStep:
    payload = read_json(videos_root / video_id / "steps.json")
    rows = payload.get("steps", []) if isinstance(payload, dict) else payload
    for row in rows:
        step = ProceduralStep(**row)
        if step.step_number == step_number:
            return step
    raise ValueError(f"Etape {step_number} absente de {video_id}")


def build(
    *,
    audit_file: Path | None = None,
    output_dir: Path | None = None,
    videos_root: Path | None = None,
    project_root: Path | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    project_root = (project_root or path("")).resolve()
    audit_file = audit_file or path("data/interim/video_quality_audit.json")
    output_dir = output_dir or path("data/exports/video_review_sample")
    videos_root = videos_root or path("data/interim/videos")
    report = read_json(audit_file)
    recommendations = report.get("recommended_review_steps", [])
    if not recommendations:
        raise ValueError("L'audit ne contient aucune étape de revue recommandée")

    output_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    markdown = [
        "# Échantillon de revue vidéo",
        "",
        "Chaque entrée contient la phrase prononcée, l'instruction reconstruite "
        "et les preuves visuelles à vérifier.",
        "",
    ]
    for index, recommendation in enumerate(recommendations, 1):
        video_id = str(recommendation["video_id"])
        step_number = int(recommendation["step_number"])
        step = _load_step(videos_root, video_id, step_number)
        item_dir = output_dir / f"{index:02d}-{video_id}-step-{step_number:03d}"
        item_dir.mkdir(parents=True, exist_ok=True)

        sources = [
            ("avant", step.screenshot_before),
            *(
                (f"support-{number}", value)
                for number, value in enumerate(step.supporting_screenshots, 1)
            ),
            ("apres", step.screenshot_after),
        ]
        screenshots = []
        for asset_index, (role, value) in enumerate(sources, 1):
            if not value:
                continue
            source = _asset_path(value, project_root)
            if not source.is_file():
                raise FileNotFoundError(f"Capture absente : {source}")
            destination = item_dir / f"{asset_index:02d}-{role}{source.suffix.lower()}"
            shutil.copy2(source, destination)
            screenshots.append(
                {
                    "role": role,
                    "file": destination.relative_to(output_dir).as_posix(),
                    "source": value,
                }
            )

        item = {
            "order": index,
            "video_id": video_id,
            "video_title": step.video_title,
            "procedure": step.procedure,
            "step_number": step.step_number,
            "total_steps": step.total_steps,
            "timestamp_start": step.timestamp_start,
            "timestamp_end": step.timestamp_end,
            "spoken_instruction": step.spoken_instruction,
            "instruction": step.instruction,
            "action_type": step.action_type,
            "action_target": step.action_target,
            "action_location": step.action_location,
            "screen_before": step.screen_before,
            "screen_after": step.screen_after,
            "visual_description": step.visual_description,
            "quality_flags": step.quality_flags,
            "quality_flag_categories": recommendation.get(
                "quality_flag_categories", []
            ),
            "screenshots": screenshots,
        }
        items.append(item)

        markdown.extend(
            [
                f"## {index}. {step.video_title} — étape {step.step_number}",
                "",
                f"- Temps : {step.timestamp_start:.1f}s à {step.timestamp_end:.1f}s",
                f"- Phrase prononcée : {step.spoken_instruction}",
                f"- Instruction reconstruite : {step.instruction}",
                f"- Action : {step.action_type or '-'} / {step.action_target or '-'}",
                f"- Emplacement : {step.action_location or '-'}",
                "",
            ]
        )
        for screenshot in screenshots:
            markdown.extend(
                [
                    f"### Capture {screenshot['role']}",
                    "",
                    f"![{screenshot['role']}]({screenshot['file']})",
                    "",
                ]
            )

    atomic_write_json(output_dir / "review_items.json", {"items": items})
    (output_dir / "README.md").write_text("\n".join(markdown), encoding="utf-8")
    return output_dir, items
