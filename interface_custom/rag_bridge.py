"""Adaptateur d'affichage vers l'API exacte du moteur RAG.

Ce module ne contient aucune logique de retrieval ou de génération. Il appelle
``src.generation.answer.ask`` et transforme uniquement son résultat en objets
simples que Streamlit peut conserver dans la session.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


class EngineError(RuntimeError):
    """Erreur présentable sans exposer les détails internes du serveur."""


@dataclass(frozen=True)
class Source:
    citation_index: int
    title: str
    detail: str = ""
    score: float | None = None
    snippet: str = ""
    images: tuple[str, ...] = ()
    timestamp_start: float | None = None


@dataclass
class Answer:
    text: str = ""
    sources: list[Source] = field(default_factory=list)
    abstained: bool = False
    route: str = "both"
    used_history: bool = False
    citations_valid: bool = False
    verification_decision: str = "not_run"


_engine = None


def engine():
    """Import paresseux pour ne charger les modèles qu'à la première question."""

    global _engine
    if _engine is None:
        try:
            from src.generation import answer as gen
        except Exception as exc:  # pragma: no cover - dépend du serveur
            raise EngineError("Le moteur RAG est momentanément indisponible.") from exc
        _engine = gen
    return _engine


def _format_history(messages: list[dict[str, Any]], max_turns: int) -> list[dict[str, str]]:
    """Conserve les derniers échanges antérieurs dans le format natif du RAG."""

    history = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
        if message.get("role") in {"user", "assistant"}
        and str(message.get("content", "")).strip()
    ]
    return history[-(max_turns * 2) :]


def _timestamp(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    minutes, remaining = divmod(int(seconds), 60)
    return f"{minutes:02d}:{remaining:02d}"


def _source_from_hit(index: int, hit: Any) -> Source:
    doc = hit.document
    details = [doc.domain.upper(), doc.source_type]
    if doc.page_start is not None:
        page = (
            f"p. {doc.page_start}"
            if doc.page_end in (None, doc.page_start)
            else f"p. {doc.page_start}-{doc.page_end}"
        )
        details.append(page)
    timestamp = _timestamp(doc.timestamp_start)
    if timestamp:
        details.append(timestamp)
    if doc.breadcrumb:
        details.append(doc.breadcrumb)

    image_values = []
    if doc.screenshot:
        image_values.append(doc.screenshot)
    image_values.extend(doc.screenshots or [])
    images = tuple(dict.fromkeys(value for value in image_values if value))

    return Source(
        citation_index=index,
        title=doc.title or doc.source_id,
        detail=" · ".join(details),
        score=float(hit.score),
        snippet=" ".join(doc.content.split())[:280],
        images=images,
        timestamp_start=doc.timestamp_start,
    )


def _normalise(result: Any) -> Answer:
    """Adapte précisément ``src.generation.answer.Answer`` pour l'interface."""

    text = str(result.text)
    cited_indices = {int(value) for value in re.findall(r"\[(\d+)\]", text)}
    sources = [
        _source_from_hit(index, hit)
        for index, hit in enumerate(result.hits, 1)
        if index in cited_indices
    ]
    return Answer(
        text=text,
        sources=sources,
        abstained=bool(result.abstained),
        route=str(result.route),
        used_history=bool(result.used_history),
        citations_valid=bool(result.citations_valid),
        verification_decision=str(result.verification_decision),
    )


def ask(
    question: str,
    top_k: int,
    messages: list[dict[str, Any]],
    history_max_turns: int,
) -> Answer:
    """Appelle le moteur existant sans modifier son architecture."""

    gen = engine()
    history = _format_history(messages, history_max_turns)
    try:
        result = gen.ask(
            question,
            top_k=top_k,
            hybrid=False,
            history=history,
        )
    except Exception as exc:  # ne jamais exposer chemin, prompt ou backend
        raise EngineError("Le moteur RAG est momentanément indisponible.") from exc
    return _normalise(result)
