"""Ingestion structuree de la documentation officielle logiciel de gestion hôtelière.

Le pipeline conserve les sections et leur hiérarchie. Les fichiers
inconnus restent donc en ``review_required`` tant qu'un manifeste de source ne
les declare pas explicitement ``verified``.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import yaml
from bs4 import BeautifulSoup

from ....config import path, hotel_software_config
from ....schema import CanonicalDocument

MAX_CHARS = 2000
SUPPORTED_EXTENSIONS = {".htm", ".html", ".markdown", ".md", ".pdf", ".txt"}
MANIFEST_NAME = "_sources.yaml"


def _clean_text(value: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", value).strip()


def _split_blocks(blocks: Iterable[str], max_chars: int = MAX_CHARS) -> list[str]:
    """Regroupe des blocs sans couper un paragraphe ni perdre de contenu."""

    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for raw_block in blocks:
        block = _clean_text(raw_block)
        if not block:
            continue
        added = len(block) + (1 if current else 0)
        if current and current_size + added > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_size = 0
        current.append(block)
        current_size += len(block) + (1 if len(current) > 1 else 0)
    if current:
        chunks.append("\n".join(current))
    return chunks


def _iso_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _source_metadata(root: Path) -> dict[str, dict[str, Any]]:
    manifest = root / MANIFEST_NAME
    if not manifest.exists():
        return {}
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    sources = payload.get("sources", payload)
    if not isinstance(sources, dict):
        raise ValueError(f"{manifest}: 'sources' doit etre un objet YAML")
    result: dict[str, dict[str, Any]] = {}
    for relative_path, metadata in sources.items():
        if not isinstance(relative_path, str) or not isinstance(metadata, dict):
            raise ValueError(f"{manifest}: entree de source invalide")
        result[Path(relative_path).as_posix()] = metadata
    return result


def discover_files(directory: str | Path | None = None) -> list[Path]:
    configured = directory or hotel_software_config()["active"]["paths"]["docs"]
    root = configured if isinstance(configured, Path) else path(configured)
    if not root.exists():
        return []
    return sorted(
        file
        for file in root.rglob("*")
        if file.is_file()
        and file.name != MANIFEST_NAME
        and file.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _metadata(
    file: Path,
    source_id: str | None,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    values = dict(metadata or {})
    values.setdefault("source_id", source_id or file.stem)
    values.setdefault("authority", "official")
    values.setdefault("status", "review_required")
    values.setdefault("language", hotel_software_config()["active"].get("language", "fr"))
    return values


def _make_documents(
    *,
    file: Path,
    title: str,
    breadcrumb: str,
    blocks: Iterable[str],
    source_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
) -> list[CanonicalDocument]:
    values = _metadata(file, source_id, metadata)
    chunks = _split_blocks(blocks)
    documents: list[CanonicalDocument] = []
    for part, chunk in enumerate(chunks, 1):
        location = breadcrumb
        if page_start is not None:
            page_label = (
                f"page {page_start}"
                if page_end in (None, page_start)
                else f"pages {page_start}-{page_end}"
            )
            location = f"{breadcrumb} > {page_label}"
        documents.append(
            CanonicalDocument(
                id=CanonicalDocument.make_id(
                    str(values["source_id"]), f"{location}::part{part}"
                ),
                content=chunk,
                title=title,
                breadcrumb=location,
                domain="software",
                topic=_topic_from([breadcrumb]),
                subtopic=values.get("subtopic"),
                source_type="official_doc",
                source_id=str(values["source_id"]),
                authority=values["authority"],
                status=values["status"],
                source_path=file.as_posix(),
                url=values.get("url"),
                version=_iso_value(values.get("version")),
                last_reviewed=_iso_value(values.get("last_reviewed")),
                language=values["language"],
                software=hotel_software_config()["active_software"],
                page_start=page_start,
                page_end=page_end,
            )
        )
    return documents


def parse_html_file(
    file: Path,
    source_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[CanonicalDocument]:
    """Decoupe un HTML par sections h1/h2/h3 et conserve le fil d'Ariane."""

    soup = BeautifulSoup(file.read_text(encoding="utf-8"), "lxml")
    for unwanted in soup(["script", "style", "noscript", "template"]):
        unwanted.decompose()

    values = _metadata(file, source_id, metadata)
    page_title = str(
        values.get("title")
        or (soup.title.get_text(" ", strip=True) if soup.title else file.stem)
    )
    trail: list[str] = [page_title]
    buffer: list[str] = []
    documents: list[CanonicalDocument] = []

    def flush() -> None:
        if not buffer:
            return
        breadcrumb = " > ".join(trail)
        documents.extend(
            _make_documents(
                file=file,
                title=trail[-1],
                breadcrumb=breadcrumb,
                blocks=buffer,
                source_id=str(values["source_id"]),
                metadata=values,
            )
        )
        buffer.clear()

    for element in soup.find_all(["h1", "h2", "h3", "p", "li", "td"]):
        if element.name in ("h1", "h2", "h3"):
            flush()
            level = int(element.name[1])
            heading = _clean_text(element.get_text(" ", strip=True))
            if heading:
                trail[:] = trail[:level] + [heading]
            continue
        if element.find_parent(["p", "li", "td"]):
            continue
        text = _clean_text(element.get_text(" ", strip=True))
        if text:
            buffer.append(f"- {text}" if element.name == "li" else text)
    flush()
    return documents


def _parse_heading_text(
    file: Path,
    *,
    heading_pattern: re.Pattern[str],
    source_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[CanonicalDocument]:
    values = _metadata(file, source_id, metadata)
    document_title = str(values.get("title") or file.stem)
    current_title = document_title
    buffer: list[str] = []
    documents: list[CanonicalDocument] = []

    def flush() -> None:
        if not buffer:
            return
        breadcrumb = (
            document_title
            if current_title == document_title
            else f"{document_title} > {current_title}"
        )
        documents.extend(
            _make_documents(
                file=file,
                title=current_title,
                breadcrumb=breadcrumb,
                blocks=buffer,
                source_id=str(values["source_id"]),
                metadata=values,
            )
        )
        buffer.clear()

    for raw_line in file.read_text(encoding="utf-8-sig").splitlines():
        line = _clean_text(raw_line)
        match = heading_pattern.match(line)
        if match:
            flush()
            current_title = _clean_text(match.group("title"))
        elif line:
            buffer.append(line)
    flush()
    return documents


def parse_markdown_file(
    file: Path,
    source_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[CanonicalDocument]:
    return _parse_heading_text(
        file,
        heading_pattern=re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*#*$"),
        source_id=source_id,
        metadata=metadata,
    )


def parse_text_file(
    file: Path,
    source_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[CanonicalDocument]:
    values = _metadata(file, source_id, metadata)
    title = str(values.get("title") or file.stem)
    return _make_documents(
        file=file,
        title=title,
        breadcrumb=title,
        blocks=file.read_text(encoding="utf-8-sig").splitlines(),
        source_id=str(values["source_id"]),
        metadata=values,
    )


def _pdf_headings(page: Any) -> set[str]:
    fragments: list[tuple[float, float, float, str]] = []

    def visitor(text: str, _cm: Any, tm: Any, _font: Any, size: float) -> None:
        cleaned = _clean_text(text)
        if cleaned and size >= 14:
            fragments.append((round(float(tm[5]), 1), float(tm[4]), size, cleaned))

    page.extract_text(visitor_text=visitor)
    lines: dict[float, list[tuple[float, float, str]]] = {}
    for y, x, size, text in fragments:
        if y <= 80:
            continue
        lines.setdefault(y, []).append((x, size, text))

    headings: set[str] = set()
    for fragments_on_line in lines.values():
        ordered = sorted(fragments_on_line)
        text = _clean_text(" ".join(value for _, _, value in ordered))
        max_size = max(size for _, size, _ in ordered)
        if text and not (len(text) == 1 and text.isalpha()) and max_size < 26:
            headings.add(text.casefold())
    return headings


def parse_pdf_file(
    file: Path,
    source_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[CanonicalDocument]:
    """Extrait les sections PDF signalees par une typographie de titre."""

    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - message d'installation
        raise RuntimeError("L'ingestion PDF requiert le paquet 'pypdf'.") from exc

    values = _metadata(file, source_id, metadata)
    document_title = str(values.get("title") or file.stem)
    skip_pages = int(values.get("skip_pages", 0))
    reader = PdfReader(file)
    documents: list[CanonicalDocument] = []
    current_title = document_title
    current_page_start: int | None = None
    current_page_end: int | None = None
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        breadcrumb = (
            document_title
            if current_title == document_title
            else f"{document_title} > {current_title}"
        )
        documents.extend(
            _make_documents(
                file=file,
                title=current_title,
                breadcrumb=breadcrumb,
                blocks=buffer,
                source_id=str(values["source_id"]),
                metadata=values,
                page_start=current_page_start,
                page_end=current_page_end,
            )
        )
        buffer.clear()

    footer_marker = hotel_software_config()["active"].get("docs_footer_domain")
    for page_number, page_object in enumerate(reader.pages, 1):
        if page_number <= skip_pages:
            continue
        headings = _pdf_headings(page_object)
        raw_text = page_object.extract_text() or ""
        for raw_line in raw_text.splitlines():
            line = _clean_text(raw_line)
            if not line or (footer_marker and footer_marker in line.casefold()):
                continue
            if len(line) == 1 and line.isalpha():
                continue
            if line.casefold() in headings:
                flush()
                current_title = line
                current_page_start = page_number
                current_page_end = page_number
                continue
            if current_page_start is None:
                current_page_start = page_number
            current_page_end = page_number
            buffer.append(line)
    flush()
    return documents


def parse_file(
    file: Path,
    *,
    source_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[CanonicalDocument]:
    suffix = file.suffix.lower()
    if suffix in {".htm", ".html"}:
        return parse_html_file(file, source_id, metadata)
    if suffix in {".md", ".markdown"}:
        return parse_markdown_file(file, source_id, metadata)
    if suffix == ".txt":
        return parse_text_file(file, source_id, metadata)
    if suffix == ".pdf":
        return parse_pdf_file(file, source_id, metadata)
    raise ValueError(f"Format documentaire non supporte: {file}")


def _topic_from(trail: list[str]) -> str:
    """Rattache une section à la taxonomie du logiciel hôtelier."""

    joined = (
        unicodedata.normalize("NFKD", " ".join(trail))
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    joined = re.sub(r"[^a-z0-9]+", " ", joined)
    table = (
        (("facture", "facturation", "avoir", "tva"), "billing"),
        (("paiement", "reglement", "acompte", "arrhes", "vcc"), "payments"),
        (("check in", "check out", "no show", "arrivee"), "check_in_check_out"),
        (("reservation", "planning"), "reservations"),
        (("rapport", "statistique", "reporting"), "reports"),
        (("chambre", "menage"), "rooms"),
        (("client", "cardex"), "guests"),
        (("utilisateur", "droit"), "users"),
        (("parametrage", "configuration"), "settings"),
    )
    for keywords, topic in table:
        if any(keyword in joined for keyword in keywords):
            return topic
    return "general"


def load(directory: str | Path | None = None) -> list[CanonicalDocument]:
    configured = directory or hotel_software_config()["active"]["paths"]["docs"]
    root = configured if isinstance(configured, Path) else path(configured)
    manifest = _source_metadata(root) if root.exists() else {}
    documents: list[CanonicalDocument] = []
    for file in discover_files(root):
        relative = file.relative_to(root).as_posix()
        metadata = manifest.get(relative, {})
        source_id = str(metadata.get("source_id") or Path(relative).with_suffix(""))
        documents.extend(parse_file(file, source_id=source_id, metadata=metadata))
    return documents
