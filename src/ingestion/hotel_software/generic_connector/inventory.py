"""Inventaire reproductible du corpus logiciel de gestion hôtelière."""

from __future__ import annotations

import json

from ....config import path, hotel_software_config
from . import docs, qa

VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov"}


def _raw_video_files() -> int:
    root = path(hotel_software_config()["active"]["paths"]["videos"])
    if not root.exists():
        return 0
    return sum(
        file.is_file() and file.suffix.lower() in VIDEO_SUFFIXES
        for file in root.rglob("*")
    )


def _processed_video_documents() -> tuple[int, int]:
    output = path("data/processed/software_videos.jsonl")
    if not output.exists():
        return 0, 0
    count = 0
    verified = 0
    for line_number, raw_line in enumerate(
        output.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not raw_line.strip():
            continue
        try:
            item = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{output}:{line_number}: JSON invalide ({exc.msg})"
            ) from exc
        count += 1
        verified += item.get("status") == "verified"
    return count, verified


def report() -> dict[str, dict[str, int]]:
    parsed_docs = docs.load()
    parsed_qa = qa.load()
    faq_manifest = qa.manifest_report()
    video_documents, verified_video_documents = _processed_video_documents()
    return {
        "documentation": {
            "raw_files": len(docs.discover_files()),
            "documents": len(parsed_docs),
            "verified": sum(doc.status == "verified" for doc in parsed_docs),
        },
        "faq": {
            "raw_files": len(qa.discover_files()),
            "documents": len(parsed_qa),
            "verified": sum(doc.status == "verified" for doc in parsed_qa),
            "expected": faq_manifest["expected"],
            "missing": len(faq_manifest["errors"]),
        },
        "videos": {
            "raw_files": _raw_video_files(),
            "documents": video_documents,
            "verified": verified_video_documents,
        },
    }
