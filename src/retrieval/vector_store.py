"""Qdrant : creation de la collection et indexation."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache

from qdrant_client import QdrantClient, models

from ..config import path, settings
from ..schema import CanonicalDocument
from . import embeddings

DENSE = "dense"
SPARSE = "sparse"


@lru_cache
def client() -> QdrantClient:
    cfg = settings()["qdrant"]
    if cfg.get("mode") == "local":
        return QdrantClient(path=str(path(cfg["path"])))
    return QdrantClient(url=cfg["url"])


def close() -> None:
    """Ferme proprement le client mis en cache avant l'arrêt de Python."""

    if not client.cache_info().currsize:
        return
    client().close()
    client.cache_clear()


def create_collection(reset: bool = False) -> None:
    cfg = settings()
    name = cfg["qdrant"]["collection"]
    c = client()

    if reset and c.collection_exists(name):
        c.delete_collection(name)
    if c.collection_exists(name):
        return

    create_kwargs = dict(
        collection_name=name,
        vectors_config={
            DENSE: models.VectorParams(
                size=cfg["embeddings"]["dense_dim"],
                distance=models.Distance.COSINE,
            )
        },
    )
    if cfg["embeddings"].get("use_sparse", False):
        create_kwargs["sparse_vectors_config"] = {
            SPARSE: models.SparseVectorParams()
        }
    c.create_collection(**create_kwargs)

    # Index sur les champs qu'on filtrera (metadata filtering, section 33).
    for field in (
        "domain",
        "topic",
        "subtopic",
        "source_type",
        "source_id",
        "status",
        "software",
        "hotel_id",
        "authority",
    ):
        c.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def index(docs: list[CanonicalDocument], batch_size: int = 32) -> int:
    """Indexe les verified et retire explicitement les versions non servables."""
    cfg = settings()
    c = client()
    name = cfg["qdrant"]["collection"]

    rejected_ids = [doc.qdrant_id() for doc in docs if doc.status != "verified"]
    if rejected_ids:
        c.delete(
            collection_name=name,
            points_selector=models.PointIdsList(points=rejected_ids),
            wait=True,
        )

    docs = [doc for doc in docs if doc.status == "verified"]
    if not docs:
        return 0

    for i in range(0, len(docs), batch_size):
        chunk = docs[i : i + batch_size]
        vecs = embeddings.encode([d.text_to_embed() for d in chunk])

        points = []
        for doc, v in zip(chunk, vecs):
            vector = {DENSE: v["dense"]}
            if v.get("sparse"):
                vector[SPARSE] = models.SparseVector(
                    indices=list(v["sparse"].keys()),
                    values=list(v["sparse"].values()),
                )
            points.append(
                models.PointStruct(
                    id=doc.qdrant_id(),
                    vector=vector,
                    payload=doc.model_dump(),
                )
            )
        c.upsert(collection_name=name, points=points, wait=True)

    return len(docs)


def sync(docs: list[CanonicalDocument], batch_size: int = 32) -> dict[str, int]:
    """Synchronise un corpus complet et supprime les points devenus orphelins."""
    indexed = index(docs, batch_size=batch_size)
    desired = {doc.qdrant_id() for doc in docs if doc.status == "verified"}
    cfg = settings()
    c = client()
    name = cfg["qdrant"]["collection"]

    existing: set[str | int] = set()
    offset = None
    while True:
        points, offset = c.scroll(
            collection_name=name,
            limit=256,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        existing.update(point.id for point in points)
        if offset is None:
            break

    stale = [point_id for point_id in existing if point_id not in desired]
    if stale:
        c.delete(
            collection_name=name,
            points_selector=models.PointIdsList(points=stale),
            wait=True,
        )
    return {"indexed": indexed, "deleted": len(stale)}


def audit() -> dict:
    """Vérifie le nombre de points et leurs métadonnées."""

    cfg = settings()
    c = client()
    name = cfg["qdrant"]["collection"]
    required = (
        "id",
        "content",
        "domain",
        "topic",
        "source_type",
        "source_id",
        "authority",
        "status",
        "language",
        "content_hash",
        "parent_document_id",
        "chunk_index",
        "chunk_count",
    )
    points = []
    offset = None
    while True:
        page, offset = c.scroll(
            collection_name=name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(page)
        if offset is None:
            break

    missing = []
    for point in points:
        payload = point.payload or {}
        fields = [field for field in required if payload.get(field) in (None, "")]
        if fields:
            missing.append({"point_id": str(point.id), "fields": fields})
    return {
        "points": len(points),
        "by_domain": dict(sorted(Counter((point.payload or {}).get("domain") for point in points).items())),
        "by_source_type": dict(
            sorted(Counter((point.payload or {}).get("source_type") for point in points).items())
        ),
        "missing_metadata": missing,
        "verified_points": sum(
            (point.payload or {}).get("status") == "verified" for point in points
        ),
        "ready": bool(points) and not missing,
    }
