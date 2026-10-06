"""Ingestion pair-aware des questions/reponses logiciel de gestion hôtelière.

Les exports officiels peuvent etre fournis en JSONL ou sous forme de PDF
accompagnes des manifestes produits lors de leur collecte. Une FAQ devient un
seul document canonique : la question n'est jamais separee de sa reponse.

Les PDF logiciel de gestion hôtelière ont une couche texte que ``pypdf`` decode mal pour les
caracteres accentues. Poppler ``pdftotext`` est donc utilise volontairement :
il restitue le texte UTF-8 visible sans modifier les PDF originaux.
"""

from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

from ....config import path, hotel_software_config
from ....schema import CanonicalDocument
from .docs import _topic_from

PDF_HEADER = "faq logiciel de gestion hôtelière"
PAGE_FOOTER = re.compile(r"^page\s+\d+\s*/\s*\d+$", re.IGNORECASE)
QUESTION_ID = re.compile(r"^(?P<id>\d+)-")


def _required_text(item: dict[str, Any], key: str, location: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: champ '{key}' absent ou vide")
    return value.strip()


def parse_jsonl_file(file: Path) -> list[CanonicalDocument]:
    software = hotel_software_config()["active_software"]
    documents: list[CanonicalDocument] = []
    for line_number, raw_line in enumerate(
        file.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not raw_line.strip():
            continue
        location = f"{file}:{line_number}"
        try:
            item = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{location}: JSON invalide ({exc.msg})") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{location}: chaque ligne doit contenir un objet JSON")

        question = _required_text(item, "question", location)
        answer = _required_text(item, "answer", location)
        source_id = item.get("source_id") or CanonicalDocument.make_id(
            file.stem, question
        )
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"{location}: 'source_id' doit etre une chaine non vide")

        documents.append(
            CanonicalDocument(
                id=CanonicalDocument.make_id(source_id, "qa"),
                content=f"Question : {question}\n\nReponse : {answer}",
                title=question,
                domain="software",
                topic=item.get("topic", "general"),
                subtopic=item.get("subtopic"),
                source_type="qa",
                source_id=source_id,
                authority=item.get("authority", "community"),
                status=item.get("status", "review_required"),
                url=item.get("url"),
                source_path=item.get("source_path", file.as_posix()),
                version=item.get("version"),
                last_reviewed=item.get("last_reviewed"),
                language=item.get("language", "fr"),
                screenshot=item.get("screenshot"),
                screenshots=item.get("screenshots", []),
                software=software,
            )
        )
    return documents


def _read_manifest(file: Path) -> dict[str, Any]:
    try:
        payload = json.loads(file.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{file}: JSON invalide ({exc.msg})") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{file}: le manifeste doit contenir un objet JSON")
    questions = payload.get("questions")
    if not isinstance(questions, list):
        raise ValueError(f"{file}: champ 'questions' absent ou invalide")
    return payload


@lru_cache(maxsize=None)
def _directory_manifest(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifests = sorted(directory.glob("manifest-categorie-*.json"))
    if not manifests:
        return {}, {}
    if len(manifests) > 1:
        raise ValueError(f"{directory}: plusieurs manifestes de categorie")

    payload = _read_manifest(manifests[0])
    by_filename: dict[str, Any] = {}
    for index, question in enumerate(payload["questions"], 1):
        if not isinstance(question, dict):
            raise ValueError(f"{manifests[0]}: question {index} invalide")
        filename = question.get("filename")
        if isinstance(filename, str) and filename:
            if filename in by_filename:
                raise ValueError(f"{manifests[0]}: fichier duplique '{filename}'")
            by_filename[filename] = question
    return payload, by_filename


def _extract_pdf_pages(file: Path) -> list[str]:
    executable = shutil.which("pdftotext")
    if not executable:
        raise RuntimeError(
            "L'ingestion des FAQ PDF requiert Poppler 'pdftotext'."
        )
    result = subprocess.run(
        [executable, "-enc", "UTF-8", "-layout", str(file), "-"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        details = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"{file}: pdftotext a echoue ({details or result.returncode})")
    text = result.stdout.decode("utf-8", errors="strict").replace("\r\n", "\n")
    pages = text.split("\f")
    while pages and not pages[-1].strip():
        pages.pop()
    if not pages:
        raise ValueError(f"{file}: PDF sans page textuelle")
    return pages


def _clean_page(raw_page: str) -> str:
    lines: list[str] = []
    previous_blank = True
    for raw_line in html.unescape(raw_page).splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if line.casefold() == PDF_HEADER or PAGE_FOOTER.match(line):
            continue
        if not line:
            if not previous_blank:
                lines.append("")
            previous_blank = True
            continue
        lines.append(line)
        previous_blank = False
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _comparison_key(value: str) -> str:
    value = html.unescape(value)
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii").casefold()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def _without_repeated_question(page: str, question: str) -> str:
    """Retire uniquement le titre repete au debut de la premiere page."""

    lines = page.splitlines()
    while lines and not lines[0]:
        lines.pop(0)
    if not lines:
        return ""

    end = 0
    while end < len(lines) and lines[end]:
        end += 1
    candidate = " ".join(lines[:end])
    candidate_key = _comparison_key(candidate)
    question_key = _comparison_key(question)
    similarity = SequenceMatcher(None, candidate_key, question_key).ratio()
    if candidate_key and question_key and (
        candidate_key in question_key
        or question_key in candidate_key
        or similarity >= 0.72
    ):
        lines = lines[end:]
        while lines and not lines[0]:
            lines.pop(0)
    return "\n".join(lines).strip()


def _portal_url(query: str) -> str:
    """URL du portail FAQ du logiciel de gestion hôtelière.
    La valeur réelle vient de la configuration privée ; le domaine générique
    n'est qu'un espace réservé pour le dépôt public.
    """
    base = (
        hotel_software_config()["active"].get("portal_url")
        or "https://software-portal.example/faq_client/index.php"
    )
    return f"{base}?{query}"


def _pdf_identity(file: Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
    manifest, by_filename = _directory_manifest(file.parent)
    item = by_filename.get(file.name, {})
    match = QUESTION_ID.match(file.name)
    question_id = str(item.get("id") or (match.group("id") if match else file.stem))
    return question_id, manifest, item


def parse_pdf_file(file: Path) -> CanonicalDocument:
    question_id, manifest, item = _pdf_identity(file)
    title = str(item.get("title") or file.stem).strip()
    category_id = str(manifest.get("category_id") or file.parent.name)
    pages = [_clean_page(page) for page in _extract_pdf_pages(file)]
    pages[0] = _without_repeated_question(pages[0], title)
    sections = [
        f"[Page {page_number}]\n{page_text}"
        for page_number, page_text in enumerate(pages, 1)
        if page_text
    ]
    if not sections:
        raise ValueError(f"{file}: reponse FAQ vide apres nettoyage")

    answer = "\n\n".join(sections)
    source_id = f"software-faq-{question_id}"
    subcategory_id = item.get("subcategory_id")
    query = f"question={question_id}&topic={category_id}"
    if subcategory_id not in (None, ""):
        query += f"&topic_quest={subcategory_id}"
    generated_at = manifest.get("generated_at")
    last_reviewed = str(generated_at)[:10] if generated_at else None
    downloaded = item.get("status") == "downloaded"

    return CanonicalDocument(
        id=CanonicalDocument.make_id(source_id, "qa"),
        content=f"Question : {title}\n\nReponse :\n{answer}",
        title=title,
        breadcrumb=f"FAQ logiciel de gestion hôtelière > categorie {category_id} > {title}",
        domain="software",
        topic=_topic_from([title, answer]),
        subtopic=f"faq_category_{category_id}",
        source_type="qa",
        source_id=source_id,
        authority="official",
        status="verified" if downloaded else "review_required",
        url=_portal_url(query),
        source_path=file.as_posix(),
        page_start=1,
        page_end=len(pages),
        last_reviewed=last_reviewed,
        language="fr",
        software=hotel_software_config()["active_software"],
    )


def discover_files(directory: str | Path | None = None) -> list[Path]:
    configured = directory or hotel_software_config()["active"]["paths"]["qa"]
    root = configured if isinstance(configured, Path) else path(configured)
    if not root.exists():
        return []
    return sorted(
        file
        for file in root.rglob("*")
        if file.is_file() and file.suffix.lower() in {".jsonl", ".pdf"}
    )


def manifest_report(directory: str | Path | None = None) -> dict[str, Any]:
    configured = directory or hotel_software_config()["active"]["paths"]["qa"]
    root = configured if isinstance(configured, Path) else path(configured)
    if not root.exists():
        return {
            "manifests": 0,
            "expected": 0,
            "downloaded": 0,
            "recovered": 0,
            "errors": [],
        }

    recovered_ids: set[str] = set()
    for jsonl_file in root.rglob("*.jsonl"):
        for line_number, raw_line in enumerate(
            jsonl_file.read_text(encoding="utf-8-sig").splitlines(), 1
        ):
            if not raw_line.strip():
                continue
            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{jsonl_file}:{line_number}: JSON invalide ({exc.msg})"
                ) from exc
            if (
                isinstance(item, dict)
                and item.get("status") == "verified"
                and isinstance(item.get("source_id"), str)
            ):
                recovered_ids.add(item["source_id"])

    expected: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    manifests = sorted(root.rglob("manifest-categorie-*.json"))
    for manifest_file in manifests:
        manifest = _read_manifest(manifest_file)
        category_id = str(manifest.get("category_id") or "unknown")
        for item in manifest["questions"]:
            if not isinstance(item, dict):
                continue
            question_id = str(item.get("id") or "")
            key = f"{category_id}:{question_id}"
            expected[key] = item
            source_id = f"software-faq-{question_id}"
            if item.get("status") != "downloaded" and source_id not in recovered_ids:
                errors.append(
                    {
                        "category_id": category_id,
                        "id": question_id,
                        "title": item.get("title"),
                        "error": item.get("error"),
                    }
                )
    return {
        "manifests": len(manifests),
        "expected": len(expected),
        "downloaded": sum(
            item.get("status") == "downloaded"
            or f"software-faq-{item.get('id')}" in recovered_ids
            for item in expected.values()
        ),
        "recovered": sum(
            item.get("status") != "downloaded"
            and f"software-faq-{item.get('id')}" in recovered_ids
            for item in expected.values()
        ),
        "errors": errors,
    }


def load(directory: str | Path | None = None) -> list[CanonicalDocument]:
    documents: list[CanonicalDocument] = []
    for file in discover_files(directory):
        if file.suffix.lower() == ".jsonl":
            documents.extend(parse_jsonl_file(file))
        else:
            documents.append(parse_pdf_file(file))
    return documents
