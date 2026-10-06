"""Orchestration reprenable et sure du pipeline video en quatre etapes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from filelock import FileLock

from .....config import path, settings
from .....schema import CanonicalDocument
from . import manifest, s1_transcribe, s2_keyframes, s3_describe, s4_assemble
from .common import (
    atomic_write_json,
    discover_videos,
    project_relative,
    slugify,
    video_id_for,
    work_dir,
)


@dataclass
class BatchRun:
    completed: list[str] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    documents: list[CanonicalDocument] = field(default_factory=list)


def _validate_stages(from_stage: int, stop_after: int) -> None:
    if from_stage not in {1, 2, 3, 4} or stop_after not in {1, 2, 3, 4}:
        raise ValueError("Les etapes doivent etre comprises entre 1 et 4")
    if from_stage > stop_after:
        raise ValueError("--stage ne peut pas etre superieur a --stop-after")
    if settings()["hw"].get("isolate_video_stages", False) and from_stage != stop_after:
        raise ValueError(
            "Ce serveur impose une etape GPU par invocation. "
            "Lance successivement --stage 1, 2, 3 puis 4 avec le meme --stop-after."
        )


def process(
    video: Path,
    procedure: str | None = None,
    from_stage: int = 1,
    stop_after: int = 4,
    force: bool = False,
) -> list[CanonicalDocument]:
    _validate_stages(from_stage, stop_after)
    video = video.resolve()
    video_id = video_id_for(video)
    procedure = procedure or slugify(video.stem)
    work_dir(video_id).mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(work_dir(video_id) / ".pipeline.lock"), timeout=0)

    with lock:
        segments = (
            s1_transcribe.run(video, video_id, force=force)
            if from_stage <= 1
            else s1_transcribe.load(video_id)
        )
        if stop_after == 1:
            return []

        frames = (
            s2_keyframes.run(video, video_id, segments, force=force)
            if from_stage <= 2
            else s2_keyframes.load(video_id)
        )
        if stop_after == 2:
            return []

        descriptions = (
            s3_describe.run(video_id, frames, force=force)
            if from_stage <= 3
            else s3_describe.load(video_id)
        )
        if stop_after == 3:
            return []

        s3_describe.unload()
        steps = s4_assemble.run(
            video_id=video_id,
            descriptions=descriptions,
            segments=segments,
            procedure=procedure,
            video_title=video.stem,
            video_path=project_relative(video),
            force=force,
        )
        return [step.to_document() for step in steps]


def process_all(
    from_stage: int = 1,
    stop_after: int = 4,
    force: bool = False,
) -> BatchRun:
    _validate_stages(from_stage, stop_after)
    videos = discover_videos()
    manifest_file = path("data/interim/video_manifest.json")
    if not manifest_file.exists():
        manifest.write(inspect_media=False)
    batch = BatchRun()

    try:
        for index, video in enumerate(videos, 1):
            video_id = video_id_for(video)
            print(
                f"[{index}/{len(videos)}] {video_id} | {project_relative(video)}",
                flush=True,
            )
            try:
                batch.documents.extend(
                    process(
                        video,
                        from_stage=from_stage,
                        stop_after=stop_after,
                        force=force,
                    )
                )
                batch.completed.append(video_id)
            except Exception as exc:  # noqa: BLE001 - le batch doit continuer sur la video suivante
                failure = {
                    "video_id": video_id,
                    "path": project_relative(video),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                batch.failures.append(failure)
                print(
                    f"   ECHEC {failure['error_type']} : {failure['error']}", flush=True
                )
    finally:
        if from_stage == 1:
            s1_transcribe.unload()
        if from_stage == 3:
            s3_describe.unload()
        if from_stage == 4:
            s4_assemble.unload()

    atomic_write_json(
        path("data/interim/video_batch_report.json"),
        {
            "stage": from_stage,
            "completed": batch.completed,
            "failures": batch.failures,
        },
    )
    return batch
