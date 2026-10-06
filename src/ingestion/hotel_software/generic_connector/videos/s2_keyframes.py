"""Etape 2 : selection de keyframes autour des actions et transitions UI."""

from __future__ import annotations

import re
import subprocess
import unicodedata
from pathlib import Path

from .....config import settings
from .....schema import TranscriptSegment
from .common import (
    atomic_write_json,
    project_relative,
    read_json,
    require_executable,
    work_dir,
)
from .manifest import probe


def output_file(video_id: str) -> Path:
    return work_dir(video_id) / "keyframes.json"


def load(video_id: str) -> list[dict]:
    file = output_file(video_id)
    if not file.exists():
        raise FileNotFoundError(f"Keyframes absentes pour {video_id} : {file}")
    payload = read_json(file)
    return payload.get("frames", []) if isinstance(payload, dict) else payload


def _normalized(text: str) -> str:
    value = (
        unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    )
    return re.sub(r"\s+", " ", value).lower()


def candidate_indices(segments: list[TranscriptSegment]) -> list[int]:
    cfg = settings()["video"]
    if cfg.get("capture_all_segments", False):
        return list(range(len(segments)))
    cues = tuple(_normalized(cue) for cue in cfg.get("action_cues", []))
    selected = [
        i
        for i, segment in enumerate(segments)
        if any(cue in _normalized(segment.text) for cue in cues)
    ]
    return selected or list(range(len(segments)))


def _grab(video: Path, timestamp: float, destination: Path, force: bool) -> float:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return timestamp
    destination.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = require_executable("ffmpeg")
    last_error = ""
    for rewind in (0.0, 0.25, 0.75, 1.5):
        candidate = max(0.0, timestamp - rewind)
        destination.unlink(missing_ok=True)
        result = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{candidate:.3f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-compression_level",
                "2",
                str(destination),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        last_error = result.stderr.strip()
        if (
            result.returncode == 0
            and destination.exists()
            and destination.stat().st_size > 0
        ):
            return candidate

    destination.unlink(missing_ok=True)
    detail = f" ({last_error})" if last_error else ""
    raise RuntimeError(
        f"Capture vide produite autour de {timestamp:.3f}s pour {video}{detail}"
    )


def _frame_record(
    *,
    video: Path,
    video_id: str,
    segment_index: int,
    moment: str,
    timestamp: float,
    text: str,
    force: bool,
    suffix: str = "",
) -> dict:
    frame_id = f"seg{segment_index:04d}-{moment}{suffix}"
    destination = work_dir(video_id) / "keyframes" / f"{frame_id}.png"
    captured_at = _grab(video, timestamp, destination, force=force)
    return {
        "frame_id": frame_id,
        "segment": segment_index,
        "moment": moment,
        "t": round(captured_at, 3),
        "file": project_relative(destination),
        "text": text,
    }


def from_transcript(
    video: Path,
    video_id: str,
    segments: list[TranscriptSegment],
    duration: float,
    force: bool,
) -> list[dict]:
    cfg = settings()["video"]
    frames = []
    for index in candidate_indices(segments):
        segment = segments[index]
        before_t = max(0.0, segment.start - float(cfg["frame_offset_before"]))
        after_t = min(
            max(0.0, duration - 0.05), segment.end + float(cfg["frame_offset_after"])
        )
        frames.append(
            _frame_record(
                video=video,
                video_id=video_id,
                segment_index=index,
                moment="before",
                timestamp=before_t,
                text=segment.text,
                force=force,
            )
        )
        if after_t - before_t >= float(cfg["min_frame_gap"]):
            frames.append(
                _frame_record(
                    video=video,
                    video_id=video_id,
                    segment_index=index,
                    moment="after",
                    timestamp=after_t,
                    text=segment.text,
                    force=force,
                )
            )
    return frames


def from_scene_detection(
    video: Path,
    video_id: str,
    segments: list[TranscriptSegment],
    fps: float,
    force: bool,
) -> list[dict]:
    from scenedetect import AdaptiveDetector, detect

    if not segments:
        return []
    cfg = settings()["video"]
    selected = candidate_indices(segments)
    scenes = detect(
        str(video),
        AdaptiveDetector(
            adaptive_threshold=float(cfg["scene_threshold"]),
            min_scene_len=max(1, round(float(cfg["min_scene_len"]) * (fps or 25.0))),
        ),
    )
    frames = []
    centers = {
        index: (segments[index].start + segments[index].end) / 2 for index in selected
    }
    for scene_index, (start, _end) in enumerate(scenes):
        timestamp = start.get_seconds() + 0.15
        segment_index = min(selected, key=lambda i: abs(centers[i] - timestamp))
        frames.append(
            _frame_record(
                video=video,
                video_id=video_id,
                segment_index=segment_index,
                moment="scene",
                timestamp=timestamp,
                text=segments[segment_index].text,
                force=force,
                suffix=f"-{scene_index:04d}",
            )
        )
    return frames


def from_semantic_cues(
    video: Path,
    video_id: str,
    segments: list[TranscriptSegment],
    duration: float,
    force: bool,
) -> list[dict]:
    """Capture les éléments montrés successivement dans une même phrase."""

    cfg = settings()["video"]
    cues = tuple(_normalized(cue) for cue in cfg.get("semantic_frame_cues", []))
    minimum = int(cfg.get("semantic_frame_min_matches", 3))
    maximum = int(cfg.get("semantic_frame_max_per_segment", 6))
    delay = float(cfg.get("semantic_frame_delay", 0.15))
    frames = []
    for segment_index in candidate_indices(segments):
        segment = segments[segment_index]
        matches: list[tuple[int, str, float]] = []
        seen: set[str] = set()
        for word_index, word in enumerate(segment.words):
            if word.start is None or word.end is None:
                continue
            normalized_word = _normalized(word.text)
            cue = next(
                (value for value in cues if value in normalized_word and value not in seen),
                None,
            )
            if cue is None:
                continue
            seen.add(cue)
            timestamp = min(
                max(0.0, duration - 0.05),
                max(segment.start, (word.start + word.end) / 2 + delay),
            )
            matches.append((word_index, cue, timestamp))
        if len(matches) < minimum:
            continue
        for word_index, cue, timestamp in matches[:maximum]:
            frames.append(
                _frame_record(
                    video=video,
                    video_id=video_id,
                    segment_index=segment_index,
                    moment="detail",
                    timestamp=timestamp,
                    text=segment.text,
                    force=force,
                    suffix=f"-{cue}-{word_index:03d}",
                )
            )
    return frames


def _deduplicate(frames: list[dict]) -> list[dict]:
    gap = float(settings()["video"]["min_frame_gap"])
    kept = []
    for frame in sorted(frames, key=lambda item: (item["t"], item["frame_id"])):
        duplicate = next(
            (
                other
                for other in kept
                if other["segment"] == frame["segment"]
                and abs(other["t"] - frame["t"]) < gap
            ),
            None,
        )
        if duplicate is None:
            kept.append(frame)
    return kept


def _remove_unreferenced_frames(video_id: str, frames: list[dict]) -> None:
    directory = work_dir(video_id) / "keyframes"
    if not directory.exists():
        return
    expected = {Path(frame["file"]).name for frame in frames}
    for candidate in directory.glob("*.png"):
        if candidate.name not in expected:
            candidate.unlink()


def run(
    video: Path,
    video_id: str,
    segments: list[TranscriptSegment],
    use_scenes: bool = True,
    force: bool = False,
) -> list[dict]:
    file = output_file(video_id)
    if file.exists() and not force:
        return load(video_id)

    media = probe(video)
    frames = from_transcript(video, video_id, segments, media["duration"], force=force)
    frames.extend(
        from_semantic_cues(
            video,
            video_id,
            segments,
            media["duration"],
            force=force,
        )
    )
    if use_scenes:
        frames.extend(
            from_scene_detection(video, video_id, segments, media["fps"], force=force)
        )
    frames = _deduplicate(frames)

    atomic_write_json(
        file,
        {
            "schema_version": 2,
            "video_id": video_id,
            "media": media,
            "frames": frames,
        },
    )
    _remove_unreferenced_frames(video_id, frames)
    return frames
