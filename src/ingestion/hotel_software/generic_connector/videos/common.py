"""Utilitaires communs du pipeline video.

Tous les chemins enregistres dans les artefacts sont relatifs a la racine du
projet. Les resultats produits sur le serveur restent ainsi utilisables apres
leur copie sur une autre machine.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .....config import path, hotel_software_config

VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov"}


def video_root() -> Path:
    return path(hotel_software_config()["active"]["paths"]["videos"]).resolve()


def discover_videos(directory: Path | None = None) -> list[Path]:
    """Decouvre recursivement les videos, y compris dans les categories."""
    root = (directory or video_root()).resolve()
    if not root.exists():
        return []
    return sorted(
        (
            p.resolve()
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
        ),
        key=lambda p: p.as_posix().casefold(),
    )


def project_relative(file: Path) -> str:
    """Chemin POSIX portable, relatif au projet quand c'est possible."""
    resolved = file.resolve()
    project_root = path("").resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        # Sur le serveur, ``data`` peut être un lien vers un disque rapide
        # monté ailleurs. On conserve malgré tout un chemin portable ``data/...``.
        data_root = path("data").resolve()
        try:
            return (Path("data") / resolved.relative_to(data_root)).as_posix()
        except ValueError:
            return resolved.as_posix()


def slugify(value: str) -> str:
    normalized = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    )
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", normalized).strip("-").lower()
    return slug[:64] or "video"


def video_id_for(video: Path, directory: Path | None = None) -> str:
    """ID lisible et stable, sans collision entre deux sous-dossiers."""
    root = (directory or video_root()).resolve()
    resolved = video.resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError:
        relative = resolved.as_posix()
    digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:10]
    return f"{slugify(video.stem)}-{digest}"


def work_dir(video_id: str) -> Path:
    return path(f"data/interim/videos/{video_id}")


def artifact_path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else path(candidate.as_posix())


def resolve_video(selector: str, directory: Path | None = None) -> Path:
    """Resout un chemin relatif, un nom, un stem ou un video_id sans ambiguite."""
    root = (directory or video_root()).resolve()
    direct = (root / selector).resolve()
    if direct.is_file() and direct.suffix.lower() in VIDEO_SUFFIXES:
        return direct

    wanted = selector.casefold()
    matches = []
    for video in discover_videos(root):
        relative = video.relative_to(root).as_posix()
        keys = {
            relative.casefold(),
            video.name.casefold(),
            video.stem.casefold(),
            video_id_for(video, root),
        }
        if wanted in keys:
            matches.append(video)

    if not matches:
        raise FileNotFoundError(f"Video introuvable : {selector}")
    if len(matches) > 1:
        choices = "\n".join(f"- {p.relative_to(root).as_posix()}" for p in matches)
        raise ValueError(f"Selection video ambigue : {selector}\n{choices}")
    return matches[0]


def require_executable(name: str) -> str:
    executable = shutil.which(name)
    if not executable:
        raise RuntimeError(
            f"Executable requis introuvable : {name}. "
            "Installe ffmpeg/ffprobe dans l'environnement Conda du serveur."
        )
    return executable


def read_json(file: Path) -> Any:
    return json.loads(file.read_text(encoding="utf-8"))


def atomic_write_json(file: Path, payload: Any) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_name(f".{file.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, file)


def atomic_write_jsonl(file: Path, rows: Iterable[dict]) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_name(f".{file.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, file)
