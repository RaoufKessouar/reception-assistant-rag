"""Inventaire reproductible des videos brutes."""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path

from .common import (
    atomic_write_json,
    discover_videos,
    project_relative,
    require_executable,
    video_id_for,
    video_root,
)


def probe(video: Path) -> dict:
    ffprobe = require_executable("ffprobe")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=index,codec_type,codec_name,width,height,avg_frame_rate,sample_rate,channels",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    video_stream = next(
        (s for s in payload.get("streams", []) if s.get("codec_type") == "video"), None
    )
    audio_stream = next(
        (s for s in payload.get("streams", []) if s.get("codec_type") == "audio"), None
    )
    if not video_stream:
        raise ValueError(f"Aucun flux video dans {video}")

    raw_rate = video_stream.get("avg_frame_rate") or "0/1"
    try:
        fps = float(Fraction(raw_rate))
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    return {
        "duration": float(payload.get("format", {}).get("duration") or 0.0),
        "fps": fps,
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "video_codec": video_stream.get("codec_name"),
        "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
        "audio_sample_rate": int(audio_stream.get("sample_rate") or 0)
        if audio_stream
        else None,
        "audio_channels": int(audio_stream.get("channels") or 0)
        if audio_stream
        else None,
    }


def build(directory: Path | None = None, inspect_media: bool = True) -> list[dict]:
    root = (directory or video_root()).resolve()
    entries = []
    for video in discover_videos(root):
        item = {
            "video_id": video_id_for(video, root),
            "path": project_relative(video),
            "relative_to_video_root": video.relative_to(root).as_posix(),
            "size_bytes": video.stat().st_size,
        }
        if inspect_media:
            item.update(probe(video))
        entries.append(item)
    return entries


def write(directory: Path | None = None, inspect_media: bool = True) -> Path:
    from .....config import path

    output = path("data/interim/video_manifest.json")
    atomic_write_json(output, {"videos": build(directory, inspect_media=inspect_media)})
    return output
