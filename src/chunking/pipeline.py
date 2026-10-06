"""Chunking structure-aware du corpus normalisé.

La structure métier est prioritaire sur une taille arbitraire :

* documentation : section, sous-section, puis paragraphes si nécessaire ;
* FAQ : question et réponse restent ensemble ;
* vidéo : une action validée reste un chunk ;
* procédure hôtel : la fiche complète reste ensemble tant qu'elle est raisonnable.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import unicodedata
from typing import Any, Iterable

from ..config import path, settings
from ..normalization import pipeline as normalization
from ..schema import CanonicalDocument

OUTPUT = "data/processed/canonical_chunks.jsonl"
REPORT = "data/processed/chunking_report.json"


def _content_hash(content: str) -> str:
    normalized = unicodedata.normalize("NFKC", content).casefold().strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _split_sentences(text: str, max_chars: int) -> list[str]:
    """Dernier recours pour un paragraphe dépassant seul la limite."""

    sentences = re.split(r"(?<=[.!?;:])\s+", text.strip())
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            parts.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts or [text.strip()]


def _structural_parts(content: str, max_chars: int) -> list[str]:
    """Regroupe des paragraphes entiers avant tout découpage par taille."""

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", content) if part.strip()]
    atomic: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            atomic.append(paragraph)
        else:
            atomic.extend(_split_sentences(paragraph, max_chars))

    parts: list[str] = []
    current: list[str] = []
    current_size = 0
    for paragraph in atomic:
        separator = 2 if current else 0
        if current and current_size + separator + len(paragraph) > max_chars:
            parts.append("\n\n".join(current))
            current = [paragraph]
            current_size = len(paragraph)
        else:
            current.append(paragraph)
            current_size += separator + len(paragraph)
    if current:
        parts.append("\n\n".join(current))
    return parts or [content]


def _materialize(document: CanonicalDocument, parts: list[str]) -> list[CanonicalDocument]:
    total = len(parts)
    chunks: list[CanonicalDocument] = []
    for index, content in enumerate(parts, 1):
        split = total > 1
        chunk_id = (
            CanonicalDocument.make_id(document.source_id, f"{document.id}:chunk:{index}")
            if split
            else document.id
        )
        breadcrumb = document.breadcrumb
        if split and breadcrumb:
            breadcrumb = f"{breadcrumb} > partie {index}/{total}"
        chunks.append(
            document.model_copy(
                update={
                    "id": chunk_id,
                    "content": content,
                    "breadcrumb": breadcrumb,
                    "content_hash": _content_hash(content),
                    "parent_document_id": document.id,
                    "chunk_index": index,
                    "chunk_count": total,
                }
            )
        )
    return chunks


def chunk_document(
    document: CanonicalDocument,
    *,
    documentation_max_chars: int | None = None,
    hotel_procedure_max_chars: int | None = None,
) -> list[CanonicalDocument]:
    """Applique la règle correspondant au type de source."""

    cfg = settings()["chunking"]
    documentation_max_chars = (
        documentation_max_chars or cfg["documentation_max_chars"]
    )
    hotel_procedure_max_chars = (
        hotel_procedure_max_chars or cfg["hotel_procedure_max_chars"]
    )

    if document.source_type == "official_doc":
        parts = _structural_parts(document.content, documentation_max_chars)
    elif document.source_type == "internal_sop":
        parts = _structural_parts(document.content, hotel_procedure_max_chars)
    else:
        # Q/A pair-aware et vidéo action-aware : ces documents sont déjà les
        # unités métier atomiques produites par l'ingestion.
        parts = [document.content]
    return _materialize(document, parts)


def _percentile(values: list[int], ratio: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * ratio)]


def build_report(
    documents: list[CanonicalDocument], chunks: list[CanonicalDocument]
) -> dict[str, Any]:
    cfg = settings()["chunking"]
    by_parent = Counter(chunk.parent_document_id for chunk in chunks)
    sizes = [len(chunk.content) for chunk in chunks]
    qa_errors = [
        chunk.id
        for chunk in chunks
        if chunk.source_type == "qa"
        and not (
            chunk.chunk_count == 1
            and "Question :" in chunk.content
            and "Reponse :" in chunk.content
        )
    ]
    video_errors = [
        chunk.id
        for chunk in chunks
        if chunk.source_type == "video"
        and not (
            chunk.chunk_count == 1
            and chunk.timestamp_start is not None
            and chunk.timestamp_end is not None
            and chunk.screenshot
        )
    ]
    missing_parent = [chunk.id for chunk in chunks if not chunk.parent_document_id]
    duplicate_ids = [
        chunk_id for chunk_id, count in Counter(chunk.id for chunk in chunks).items() if count > 1
    ]
    source_counts = Counter(document.source_type for document in documents)
    chunk_counts = Counter(chunk.source_type for chunk in chunks)
    lost_sources = [
        source_type
        for source_type, count in source_counts.items()
        if chunk_counts[source_type] < count
    ]

    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_documents": len(documents),
        "output_chunks": len(chunks),
        "split_documents": sum(count > 1 for count in by_parent.values()),
        "by_source_type": dict(sorted(chunk_counts.items())),
        "limits_chars": dict(cfg),
        "size_chars": {
            "min": min(sizes, default=0),
            "median": int(statistics.median(sizes)) if sizes else 0,
            "p90": _percentile(sizes, 0.90),
            "p95": _percentile(sizes, 0.95),
            "max": max(sizes, default=0),
        },
        "warnings": {
            "long_qa": [
                chunk.id
                for chunk in chunks
                if chunk.source_type == "qa"
                and len(chunk.content) > cfg["qa_warning_chars"]
            ],
            "long_video": [
                chunk.id
                for chunk in chunks
                if chunk.source_type == "video"
                and len(chunk.content) > cfg["video_warning_chars"]
            ],
        },
        "quality": {
            "qa_pair_errors": qa_errors,
            "video_action_errors": video_errors,
            "missing_parent": missing_parent,
            "duplicate_ids": duplicate_ids,
            "lost_source_types": lost_sources,
        },
    }
    report["ready_for_vectorization"] = not any(report["quality"].values())
    return report


def _atomic_write_json(file: Path, payload: Any) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_name(f".{file.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, file)


def _atomic_write_jsonl(file: Path, documents: Iterable[CanonicalDocument]) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_name(f".{file.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for document in documents:
            stream.write(json.dumps(document.model_dump(mode="json"), ensure_ascii=False) + "\n")
    os.replace(temporary, file)


def run(
    *,
    input_file: Path | None = None,
    output_file: Path | None = None,
    report_file: Path | None = None,
) -> tuple[Path, list[CanonicalDocument], dict[str, Any]]:
    documents = normalization.load(input_file)
    chunks = [chunk for document in documents for chunk in chunk_document(document)]
    report = build_report(documents, chunks)
    output_file = output_file or path(OUTPUT)
    report_file = report_file or path(REPORT)
    _atomic_write_jsonl(output_file, chunks)
    _atomic_write_json(report_file, report)
    return output_file, chunks, report


def load(file: Path | None = None) -> list[CanonicalDocument]:
    file = file or path(OUTPUT)
    if not file.is_file():
        raise FileNotFoundError(
            f"Chunks absents : {file}. Lance d'abord corpus-normalize puis corpus-chunk."
        )
    return [
        CanonicalDocument(**json.loads(line))
        for line in file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
