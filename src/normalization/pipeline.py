"""Normalisation du corpus universel.

Les fichiers sources ne sont jamais modifiés ici. Le pipeline charge les
quatre corpus, harmonise leur texte et leurs métadonnées, déduplique les
contenus strictement identiques et produit un JSONL canonique auditable.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable

from ..config import path, hotel_software_config, settings, taxonomy_contains
from ..ingestion.base import get_active_source
from ..ingestion.hotel import knowledge_cards
from ..schema import CanonicalDocument

OUTPUT = "data/processed/canonical_documents.jsonl"
REPORT = "data/processed/normalization_report.json"

AUTHORITY_PRIORITY = {"community": 1, "verified_internal": 2, "official": 3}
STATUS_PRIORITY = {"draft": 0, "review_required": 1, "deprecated": 2, "verified": 3}


def _ascii_key(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii").casefold()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def clean_text(value: str) -> str:
    """Nettoie le bruit typographique sans aplatir les listes ni paragraphes."""

    value = unicodedata.normalize("NFKC", value).replace("\ufeff", "")
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines: list[str] = []
    previous_blank = True
    for raw_line in value.split("\n"):
        line = re.sub(r"[ \t\f\v]+", " ", raw_line).strip()
        if re.fullmatch(r"Powered by TCPDF(?: \(www\.tcpdf\.org\))?", line, re.IGNORECASE):
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


def _contains(signal: str, *phrases: str) -> bool:
    return any(_ascii_key(phrase) in signal for phrase in phrases)


def _software_taxonomy_from_signal(signal: str) -> tuple[str, str | None] | None:
    """Classe un signal déjà normalisé, du plus spécifique au plus général."""

    rules: list[tuple[str, str | None, tuple[str, ...]]] = [
        ("billing", "modify_invoice", ("avoir", "annuler une facture", "modifier une facture", "annuler du chiffre d affaires")),
        ("billing", "split_invoice", ("scinder une facture", "factures individuelles", "split invoice")),
        ("billing", "company_invoice", ("facture entreprise", "facture societe", "company invoice")),
        ("check_in_check_out", "no_show", ("no show",)),
        ("check_in_check_out", "late_arrival", ("arrivee tardive",)),
        ("check_in_check_out", "check_in", ("check in", "enregistrer une arrivee")),
        ("check_in_check_out", "check_out", ("check out", "depart anticipe", "enregistrer un depart")),
        ("reservations", "cancel_reservation", ("annuler une reservation", "annulation tardive", "conditions d annulation")),
        ("reservations", "change_dates", ("changer les dates", "modifier les dates")),
        ("reservations", "room_assignment", ("attribuer une chambre", "de attribuer", "deplacer une reservation")),
        ("reservations", "search_reservation", ("rechercher une reservation", "retrouver un dossier")),
        ("reservations", "create_reservation", ("creer un dossier", "creer une reservation", "dossier groupe")),
        ("reservations", "modify_reservation", ("modifier une reservation", "modifier les informations d un dossier")),
        ("reservations", "notes", ("note dossier", "remarque dossier")),
        ("users", None, ("utilisateur", "profil utilisateur", "droits des profils")),
        ("reports", None, ("rapport", "reporting", "statistique", "cloture journaliere", "rapprochement bancaire", "retrouver le chiffre d affaires", "ca genere")),
        ("guests", None, ("fichier client", "cardex", "clientele")),
        ("rooms", None, ("planning menage", "menage", "ouvrir et fermer des chambres", "chambre communicante")),
        ("planning", None, ("planning journalier", "planning des connectivites", "evenement planning")),
        ("settings", None, ("parametrage", "configuration", "plan tarifaire", "grille de tarifs", "code promotion", "booking engine", "stock", "extra", "option", "modele d email", "acces cb", "agence")),
        ("payments", "vcc", ("vcc", "carte virtuelle")),
        ("payments", "ota_payment", ("paiement ota", "plateforme reservation", "plateforme reservation")),
        ("payments", "refund", ("remboursement", "rembourser")),
        ("payments", "deposit", ("acompte", "arrhes", "caution", "garantie")),
        ("payments", "payment", ("reglement", "paiement", "carte bancaire", "caisse", "tpe")),
        ("billing", "invoice", ("facture", "facturation", "tva")),
        ("reservations", None, ("reservation", "dossier")),
    ]
    for topic, subtopic, phrases in rules:
        if _contains(signal, *phrases):
            return topic, subtopic
    return None


def _software_taxonomy(document: CanonicalDocument) -> tuple[str, str | None]:
    # Le titre et le breadcrumb décrivent l'objet du document. Le corps peut
    # citer d'autres fonctions et ne doit donc servir qu'en second recours.
    identity_signal = _ascii_key(
        " ".join(
            filter(
                None,
                [
                    document.title,
                    document.breadcrumb,
                    document.source_id,
                    document.source_category,
                    document.topic,
                    document.subtopic,
                ],
            )
        )
    )
    if match := _software_taxonomy_from_signal(identity_signal):
        return match

    if taxonomy_contains("software", document.topic):
        subtopic = (
            document.subtopic
            if document.subtopic
            and taxonomy_contains("software", document.topic, document.subtopic)
            else None
        )
        # Pour la documentation et les FAQ, le topic fourni par l'ingestion
        # vient déjà du titre ou de la structure officielle de la source.
        if document.source_type != "video" or document.topic != "general":
            return document.topic, subtopic

    if match := _software_taxonomy_from_signal(_ascii_key(document.content)):
        return match

    if taxonomy_contains("software", document.topic):
        return document.topic, None
    return "general", None


HOTEL_SUBTOPICS: dict[str, tuple[str, str]] = {
    "standard_checkin": ("guest_journey", "check_in"),
    "standard_checkout": ("guest_journey", "check_out"),
    "early_checkin": ("guest_journey", "early_check_in"),
    "late_checkout": ("guest_journey", "late_check_out"),
    "luggage_storage": ("guest_journey", "luggage"),
    "visitors": ("guest_journey", "visitors"),
    "late_arrival": ("night_reception", "late_arrivals"),
    "deposit_guarantee": ("payments", "deposit"),
    "refund": ("payments", "refund"),
    "cash": ("payments", "cash"),
    "card_rejected": ("payments", "card_rejected"),
    "ota_vcc": ("payments", "ota"),
    "invoice_request": ("payments", "invoice"),
    "payment_dispute": ("payments", "payment_dispute"),
    "cancellation": ("reservations", "cancellation"),
    "modification": ("reservations", "modification"),
    "no_show": ("reservations", "no_show"),
    "walk_in": ("reservations", "walk_in"),
    "overbooking": ("reservations", "overbooking"),
    "room_change": ("reservations", "room_change"),
}


def _hotel_taxonomy(document: CanonicalDocument) -> tuple[str, str | None]:
    if document.subtopic in HOTEL_SUBTOPICS:
        return HOTEL_SUBTOPICS[document.subtopic]
    if taxonomy_contains("hotel", document.topic):
        subtopic = (
            document.subtopic
            if document.subtopic
            and taxonomy_contains("hotel", document.topic, document.subtopic)
            else None
        )
        return document.topic, subtopic
    raise ValueError(
        f"Taxonomie hôtel inconnue pour {document.source_id}: "
        f"{document.topic}/{document.subtopic}"
    )


def _portable_path(value: str | None) -> str | None:
    if not value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        return candidate.as_posix()
    resolved = candidate.resolve()
    roots = ((path("").resolve(), Path()), (path("data").resolve(), Path("data")))
    for root, prefix in roots:
        try:
            return (prefix / resolved.relative_to(root)).as_posix()
        except ValueError:
            continue
    return resolved.as_posix()


def _canonical_breadcrumb(
    document: CanonicalDocument,
    topic: str,
    subtopic: str | None,
) -> str:
    root = (
        hotel_software_config()["active"].get("display_name", document.software or "logiciel de gestion hôtelière")
        if document.domain == "software"
        else document.hotel_id or settings()["hotel"]["id"]
    )
    source_trail = (
        re.split(r"\s*>\s*", document.breadcrumb)
        if document.breadcrumb
        else []
    )
    values = [root, topic, subtopic, *source_trail, document.title]
    parts: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = clean_text(str(value)) if value else ""
        key = _ascii_key(cleaned)
        if cleaned and key not in seen:
            parts.append(cleaned)
            seen.add(key)
    return " > ".join(parts)


def _source_category(document: CanonicalDocument) -> str | None:
    if document.source_category:
        return document.source_category
    if document.domain == "software" and document.source_type == "video":
        return document.topic
    if document.source_type == "qa" and document.subtopic:
        return document.subtopic
    if document.domain == "hotel":
        return "/".join(value for value in (document.topic, document.subtopic) if value)
    return "/".join(value for value in (document.topic, document.subtopic) if value) or None


def normalize_document(document: CanonicalDocument) -> CanonicalDocument:
    """Retourne une copie canonique sans modifier le document source."""

    content = clean_text(document.content)
    if not content:
        raise ValueError(f"Document vide après nettoyage : {document.id}")
    if "\ufffd" in content:
        raise ValueError(f"Caractère de remplacement Unicode dans {document.id}")

    source_category = _source_category(document)
    if document.domain == "software":
        topic, subtopic = _software_taxonomy(document)
    else:
        topic, subtopic = _hotel_taxonomy(document)
    breadcrumb = _canonical_breadcrumb(document, topic, subtopic)
    content_hash = hashlib.sha256(_ascii_key(content).encode("utf-8")).hexdigest()
    screenshots = [
        portable
        for value in document.screenshots
        if (portable := _portable_path(value)) is not None
    ]
    screenshot = _portable_path(document.screenshot)
    if screenshot and screenshot not in screenshots:
        screenshots.insert(0, screenshot)

    return document.model_copy(
        update={
            "content": content,
            "title": clean_text(document.title or document.source_id),
            "breadcrumb": breadcrumb,
            "topic": topic,
            "subtopic": subtopic,
            "source_category": source_category,
            "source_path": _portable_path(document.source_path),
            "screenshot": screenshot,
            "screenshots": list(dict.fromkeys(screenshots)),
            "software": document.software or (hotel_software_config()["active_software"] if document.domain == "software" else None),
            "hotel_id": document.hotel_id or (settings()["hotel"]["id"] if document.domain == "hotel" else None),
            "content_hash": content_hash,
        }
    )


def _document_priority(document: CanonicalDocument) -> tuple[int, int, str, str]:
    return (
        AUTHORITY_PRIORITY[document.authority],
        STATUS_PRIORITY[document.status],
        document.source_type,
        document.id,
    )


def deduplicate(
    documents: Iterable[CanonicalDocument],
) -> tuple[list[CanonicalDocument], list[dict[str, str]]]:
    """Retire les contenus strictement identiques et conserve leur trace."""

    by_id: dict[str, CanonicalDocument] = {}
    for document in documents:
        previous = by_id.get(document.id)
        if previous and previous.content_hash != document.content_hash:
            raise ValueError(f"Identifiant dupliqué avec contenus différents : {document.id}")
        if previous is None or _document_priority(document) > _document_priority(previous):
            by_id[document.id] = document

    groups: dict[tuple[str, str], list[CanonicalDocument]] = defaultdict(list)
    for document in by_id.values():
        groups[(document.domain, document.content_hash or "")].append(document)

    kept: list[CanonicalDocument] = []
    duplicates: list[dict[str, str]] = []
    for group in groups.values():
        ordered = sorted(group, key=_document_priority, reverse=True)
        winner = ordered[0]
        kept.append(winner)
        duplicates.extend(
            {
                "dropped_id": duplicate.id,
                "kept_id": winner.id,
                "content_hash": winner.content_hash or "",
            }
            for duplicate in ordered[1:]
        )
    return sorted(kept, key=lambda item: item.id), duplicates


def _authority_errors(documents: Iterable[CanonicalDocument]) -> list[dict[str, str]]:
    errors = []
    for document in documents:
        valid = True
        if document.source_type in {"official_doc", "video"}:
            valid = document.authority == "official"
        elif document.source_type == "internal_sop":
            valid = document.authority == "verified_internal"
        elif document.source_type == "qa":
            valid = document.authority in {"official", "verified_internal", "community"}
        if not valid:
            errors.append(
                {
                    "id": document.id,
                    "source_type": document.source_type,
                    "authority": document.authority,
                }
            )
    return errors


def _missing_metadata(document: CanonicalDocument) -> list[str]:
    required = [
        "id",
        "content",
        "title",
        "breadcrumb",
        "domain",
        "topic",
        "source_type",
        "source_id",
        "authority",
        "status",
        "language",
        "source_path",
        "content_hash",
    ]
    if document.domain == "software":
        required.append("software")
    else:
        required.extend(["hotel_id", "last_reviewed"])
    if document.source_type == "video":
        required.extend(["timestamp_start", "timestamp_end", "screenshot"])
    return [field for field in required if getattr(document, field, None) in (None, "")]


def build_report(
    *,
    inputs: list[CanonicalDocument],
    outputs: list[CanonicalDocument],
    duplicates: list[dict[str, str]],
) -> dict[str, Any]:
    authority_errors = _authority_errors(outputs)
    taxonomy_errors = [
        {
            "id": document.id,
            "domain": document.domain,
            "topic": document.topic,
            "subtopic": document.subtopic,
        }
        for document in outputs
        if not taxonomy_contains(document.domain, document.topic)
        or (
            document.subtopic is not None
            and not taxonomy_contains(document.domain, document.topic, document.subtopic)
        )
    ]
    missing = [
        {"id": document.id, "fields": fields}
        for document in outputs
        if (fields := _missing_metadata(document))
    ]
    cleaned = sum(
        original.content != clean_text(original.content)
        for original in inputs
    )
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_documents": len(inputs),
        "output_documents": len(outputs),
        "duplicates_removed": len(duplicates),
        "text_documents_changed": cleaned,
        "by_domain": dict(sorted(Counter(doc.domain for doc in outputs).items())),
        "by_source_type": dict(
            sorted(Counter(doc.source_type for doc in outputs).items())
        ),
        "by_status": dict(sorted(Counter(doc.status for doc in outputs).items())),
        "by_authority": dict(
            sorted(Counter(doc.authority for doc in outputs).items())
        ),
        "by_topic": {
            domain: dict(sorted(Counter(doc.topic for doc in outputs if doc.domain == domain).items()))
            for domain in ("software", "hotel")
        },
        "quality": {
            "empty_content": sum(not doc.content for doc in outputs),
            "missing_metadata": missing,
            "taxonomy_errors": taxonomy_errors,
            "authority_errors": authority_errors,
            "duplicates": duplicates,
        },
    }
    report["ready_for_chunking"] = not any(
        [
            report["quality"]["empty_content"],
            missing,
            taxonomy_errors,
            authority_errors,
        ]
    )
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
    output_file: Path | None = None,
    report_file: Path | None = None,
) -> tuple[Path, list[CanonicalDocument], dict[str, Any]]:
    """Normalise les quatre corpus et écrit l’artefact canonique unique."""

    inputs = get_active_source().load_all() + knowledge_cards.load_documents()
    normalized = [normalize_document(document) for document in inputs]
    outputs, duplicates = deduplicate(normalized)
    report = build_report(inputs=inputs, outputs=outputs, duplicates=duplicates)
    output_file = output_file or path(OUTPUT)
    report_file = report_file or path(REPORT)
    _atomic_write_jsonl(output_file, outputs)
    _atomic_write_json(report_file, report)
    return output_file, outputs, report


def load(file: Path | None = None) -> list[CanonicalDocument]:
    """Charge l'artefact normalisé produit par :func:`run`."""

    file = file or path(OUTPUT)
    if not file.is_file():
        raise FileNotFoundError(
            f"Corpus normalisé absent : {file}. Lance d'abord corpus-normalize."
        )
    return [
        CanonicalDocument(**json.loads(line))
        for line in file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
