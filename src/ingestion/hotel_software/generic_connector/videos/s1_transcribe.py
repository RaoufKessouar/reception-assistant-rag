"""Etape 1 : transcription WhisperX avec alignement au mot."""

from __future__ import annotations

import gc
import subprocess
from functools import lru_cache
from pathlib import Path

from .....config import settings
from .....schema import TranscriptSegment, TranscriptWord
from .common import atomic_write_json, read_json, require_executable, work_dir


def output_file(video_id: str) -> Path:
    return work_dir(video_id) / "transcript.json"


def load(video_id: str) -> list[TranscriptSegment]:
    file = output_file(video_id)
    if not file.exists():
        raise FileNotFoundError(f"Transcription absente pour {video_id} : {file}")
    payload = read_json(file)
    rows = payload.get("segments", []) if isinstance(payload, dict) else payload
    return [TranscriptSegment(**row) for row in rows]


def extract_audio(video: Path, video_id: str, force: bool = False) -> Path:
    output = work_dir(video_id) / "audio.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        return output

    ffmpeg = require_executable("ffmpeg")
    subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-y",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    return output


def _asr_options(config: dict) -> dict:
    """Retourne uniquement les aides lexicales non vides de Whisper."""
    return {
        key: value
        for key in ("initial_prompt", "hotwords")
        if (value := config.get("asr", {}).get(key))
    }


@lru_cache(maxsize=1)
def _load_models():
    import whisperx

    cfg = settings()
    full_device = cfg["hw"]["asr_device"]
    device, _, raw_index = full_device.partition(":")
    device_index = int(raw_index) if raw_index else 0
    if device == "cuda":
        import torch

        torch.cuda.set_device(device_index)

    model = whisperx.load_model(
        cfg["asr"]["model"],
        device,
        device_index=device_index,
        compute_type=cfg["asr"]["compute_type"],
        language=cfg["asr"]["language"],
        asr_options=_asr_options(cfg) or None,
    )
    align_model, align_metadata = whisperx.load_align_model(
        language_code=cfg["asr"]["language"],
        device=full_device,
    )
    return model, align_model, align_metadata


def unload() -> None:
    """Libere explicitement la VRAM avant de passer au VLM."""
    _load_models.cache_clear()
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        return


def run(video: Path, video_id: str, force: bool = False) -> list[TranscriptSegment]:
    file = output_file(video_id)
    if file.exists() and not force:
        return load(video_id)

    import whisperx

    cfg = settings()
    full_device = cfg["hw"]["asr_device"]
    audio_path = extract_audio(video, video_id, force=force)
    audio = whisperx.load_audio(str(audio_path))
    model, align_model, align_metadata = _load_models()

    raw = model.transcribe(audio, batch_size=cfg["asr"]["batch_size"])
    aligned = whisperx.align(
        raw["segments"],
        align_model,
        align_metadata,
        audio,
        full_device,
        return_char_alignments=False,
    )

    segments = []
    for segment in aligned["segments"]:
        text = segment.get("text", "").strip()
        if not text:
            continue
        words = [
            TranscriptWord(
                text=word.get("word", "").strip(),
                start=word.get("start"),
                end=word.get("end"),
                score=word.get("score"),
            )
            for word in segment.get("words", [])
            if word.get("word", "").strip()
        ]
        segments.append(
            TranscriptSegment(
                start=float(segment["start"]),
                end=float(segment["end"]),
                text=text,
                words=words,
            )
        )

    atomic_write_json(
        file,
        {
            "schema_version": 2,
            "video_id": video_id,
            "language": cfg["asr"]["language"],
            "segments": [segment.model_dump(mode="json") for segment in segments],
        },
    )
    return segments
