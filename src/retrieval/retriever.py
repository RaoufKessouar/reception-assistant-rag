"""
Recherche.

Etape 1 du projet : on utilise le dense seul, et on REGARDE les resultats
a l'oeil. Le hybride (dense + sparse) est deja cable ici mais ne s'active
qu'apres la mesure de la baseline (section 54 : mesurer avant d'ajouter).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from qdrant_client import models

from ..config import settings
from ..schema import CanonicalDocument
from . import embeddings
from .vector_store import DENSE, SPARSE, client


@dataclass
class Hit:
    document: CanonicalDocument
    score: float

    def __str__(self) -> str:
        return f"[{self.score:.3f}] {self.document.citation()}"


def search(
    query: str,
    top_k: int | None = None,
    domain: str | None = None,
    topic: str | None = None,
    hybrid: bool = False,
    min_score: float | None = None,
    max_chunks_per_source: int | None = None,
    exclude_subtopics: tuple[str, ...] = (),
) -> list[Hit]:
    cfg = settings()
    top_k = top_k or cfg["qdrant"]["top_k"]
    name = cfg["qdrant"]["collection"]
    candidate_multiplier = max(
        1, int(cfg["qdrant"].get("candidate_multiplier", 1))
    )
    candidate_limit = top_k * candidate_multiplier

    # Filtre : on ne sert jamais autre chose que du verifie.
    must = [
        models.FieldCondition(key="status", match=models.MatchValue(value="verified"))
    ]
    if domain:
        must.append(
            models.FieldCondition(key="domain", match=models.MatchValue(value=domain))
        )
    if topic:
        must.append(
            models.FieldCondition(key="topic", match=models.MatchValue(value=topic))
        )
    must_not = [
        models.FieldCondition(
            key="subtopic",
            match=models.MatchValue(value=subtopic),
        )
        for subtopic in exclude_subtopics
    ]
    qfilter = models.Filter(must=must, must_not=must_not)

    v = embeddings.encode_one(query, is_query=True)
    effective_hybrid = hybrid and bool(v.get("sparse"))

    if not effective_hybrid:
        res = (
            client()
            .query_points(
                collection_name=name,
                query=v["dense"],
                using=DENSE,
                limit=candidate_limit,
                query_filter=qfilter,
            )
            .points
        )
    else:
        # Fusion RRF de la recherche par le sens et de la recherche par mots exacts.
        prefetch = [
            models.Prefetch(query=v["dense"], using=DENSE, limit=candidate_limit),
            models.Prefetch(
                query=models.SparseVector(
                    indices=list(v["sparse"].keys()),
                    values=list(v["sparse"].values()),
                ),
                using=SPARSE,
                limit=candidate_limit,
            ),
        ]
        res = (
            client()
            .query_points(
                collection_name=name,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=candidate_limit,
                query_filter=qfilter,
            )
            .points
        )

    # Un score cosine dense et un score RRF ne sont pas sur la même échelle.
    # Le seuil calibré ici ne s'applique donc qu'à la recherche dense.
    configured_threshold = (
        None if effective_hybrid else cfg["qdrant"].get("min_score")
    )
    threshold = min_score if min_score is not None else configured_threshold
    hits = [Hit(document=CanonicalDocument(**p.payload), score=p.score) for p in res]
    eligible = [hit for hit in hits if threshold is None or hit.score >= threshold]
    if not effective_hybrid:
        eligible = _rerank_title_matches(eligible, query=query)
    return _select_diverse(
        eligible,
        top_k=top_k,
        max_per_source=max(
            1,
            int(
                max_chunks_per_source
                if max_chunks_per_source is not None
                else cfg["qdrant"].get("max_chunks_per_source", top_k)
            ),
        ),
    )


def search_balanced(
    query: str,
    *,
    domains: tuple[str, ...] = ("hotel", "software"),
    top_k: int | None = None,
    hybrid: bool = False,
    max_chunks_per_source: int | None = None,
    exclude_subtopics: tuple[str, ...] = (),
) -> list[Hit]:
    """Garantit une preuve de chaque domaine pour une question mixte.

    Un enrichissement lexical très précis pour la partie logiciel de gestion hôtelière peut sinon remplir
    tout le top-k et évincer la politique hôtelière pourtant nécessaire.
    """

    limit = top_k or int(settings()["qdrant"]["top_k"])
    if limit < len(domains):
        return search(
            query,
            top_k=limit,
            hybrid=hybrid,
            max_chunks_per_source=max_chunks_per_source,
            exclude_subtopics=exclude_subtopics,
        )

    pools = [
        search(
            query,
            top_k=limit,
            domain=domain,
            hybrid=hybrid,
            max_chunks_per_source=max_chunks_per_source,
            exclude_subtopics=exclude_subtopics,
        )
        for domain in domains
    ]
    selected: list[Hit] = []
    selected_ids: set[str] = set()
    for pool in pools:
        if pool and pool[0].document.id not in selected_ids:
            selected.append(pool[0])
            selected_ids.add(pool[0].document.id)
    remaining = sorted(
        (
            hit
            for pool in pools
            for hit in pool
            if hit.document.id not in selected_ids
        ),
        key=lambda hit: hit.score,
        reverse=True,
    )
    for hit in remaining:
        if hit.document.id in selected_ids:
            continue
        selected.append(hit)
        selected_ids.add(hit.document.id)
        if len(selected) == limit:
            break
    return selected[:limit]


_TITLE_STOPWORDS = {
    "a",
    "au",
    "aux",
    "avec",
    "comment",
    "dans",
    "de",
    "des",
    "du",
    "et",
    "faire",
    "la",
    "le",
    "les",
    "pour",
    "un",
    "une",
}


def _lexical_tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    without_accents = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    return {
        token
        for token in re.findall(r"[a-z0-9]+", without_accents)
        if len(token) > 1 and token not in _TITLE_STOPWORDS
    }


def _title_match_bonus(query: str, title: str | None) -> float:
    """Petit bonus lexical; il ne remplace jamais le score sémantique."""

    title_tokens = _lexical_tokens(title or "")
    if not title_tokens:
        return 0.0
    overlap = title_tokens & _lexical_tokens(query)
    return 0.06 * (len(overlap) / len(title_tokens))


def _rerank_title_matches(hits: list[Hit], *, query: str) -> list[Hit]:
    reranked = [
        Hit(
            document=hit.document,
            score=min(
                1.0,
                hit.score + _title_match_bonus(query, hit.document.title),
            ),
        )
        for hit in hits
    ]
    return sorted(reranked, key=lambda hit: hit.score, reverse=True)


def _select_diverse(
    hits: list[Hit], *, top_k: int, max_per_source: int
) -> list[Hit]:
    """Conserve les meilleurs chunks sans laisser une source saturer le top-k."""

    selected: list[Hit] = []
    source_counts: dict[str, int] = {}
    for hit in hits:
        source_id = hit.document.source_id
        if source_counts.get(source_id, 0) >= max_per_source:
            continue
        selected.append(hit)
        source_counts[source_id] = source_counts.get(source_id, 0) + 1
        if len(selected) == top_k:
            break
    return selected
