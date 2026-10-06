"""
BGE-M3 : un seul modele, deux representations.

  dense  -> le sens        ("changer de chambre" ~ "deplacer un client")
  sparse -> les mots exacts ("VCC", "no-show", "OTA", noms de menus)

C'est la raison du choix de ce modele : la recherche par le sens seule
rate le vocabulaire metier. Ici les deux signaux sortent de la meme passe.
"""

from __future__ import annotations

from functools import lru_cache

from ..config import settings


@lru_cache
def _model():
    from FlagEmbedding import BGEM3FlagModel

    cfg = settings()
    device = cfg["hw"].get("embedding_device")
    return BGEM3FlagModel(
        cfg["embeddings"]["model"],
        # FP16 convient aux serveurs GPU, mais pas au repli CPU local.
        use_fp16=str(device).lower().startswith("cuda"),
        devices=device,
        cache_dir=str(cfg.get("hf_home")) if cfg.get("hf_home") else None,
    )


@lru_cache
def _api_client():
    from openai import OpenAI

    cfg = settings()["embeddings"]
    api_key = str(cfg.get("api_key") or "").strip()
    if not api_key:
        raise ValueError("RECEPTION_RAG_EMBEDDING_API_KEY est requis en mode API")
    return OpenAI(
        base_url=str(cfg.get("base_url") or "").strip() or None,
        api_key=api_key,
        timeout=float(cfg.get("timeout_seconds", 120)),
    )


def _encode_local(texts: list[str], is_query: bool) -> list[dict]:
    cfg = settings()["embeddings"]
    out = _model().encode(
        texts,
        batch_size=1 if is_query else cfg["batch_size"],
        max_length=512 if is_query else cfg["max_length"],
        return_dense=True,
        return_sparse=cfg["use_sparse"],
        return_colbert_vecs=False,
    )

    results = []
    for i in range(len(texts)):
        item = {"dense": out["dense_vecs"][i].tolist()}
        if cfg["use_sparse"]:
            weights = out["lexical_weights"][i]
            item["sparse"] = {int(k): float(v) for k, v in weights.items() if v > 0}
        results.append(item)
    return results


def _encode_api(texts: list[str]) -> list[dict]:
    cfg = settings()["embeddings"]
    request: dict[str, object] = {
        "model": cfg["model"],
        "input": texts,
    }
    if cfg.get("api_dimensions"):
        request["dimensions"] = int(cfg["api_dimensions"])
    response = _api_client().embeddings.create(**request)
    ordered = sorted(response.data, key=lambda item: item.index)
    vectors = [list(item.embedding) for item in ordered]
    expected = int(cfg["dense_dim"])
    if len(vectors) != len(texts):
        raise ValueError("L'API d'embeddings a retourne un nombre de vecteurs invalide")
    if any(len(vector) != expected for vector in vectors):
        raise ValueError(
            "La dimension retournee par l'API d'embeddings ne correspond pas "
            f"a embeddings.dense_dim={expected}"
        )
    return [{"dense": vector} for vector in vectors]


def encode(texts: list[str], is_query: bool = False) -> list[dict]:
    """Retourne les vecteurs denses et, si disponible, les poids sparse."""
    backend = str(settings()["embeddings"].get("backend", "local")).casefold()
    if backend == "local":
        return _encode_local(texts, is_query)
    if backend in {"api", "openai_compatible"}:
        return _encode_api(texts)
    raise ValueError(f"Backend d'embeddings inconnu: {backend}")


def encode_one(text: str, is_query: bool = False) -> dict:
    return encode([text], is_query=is_query)[0]
