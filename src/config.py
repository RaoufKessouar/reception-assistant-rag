"""Chargement de la configuration. Un seul endroit qui lit les YAML."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


@lru_cache
def settings() -> dict:
    load_dotenv(ROOT / ".env", override=False)
    with open(ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Le YAML garde les valeurs reproductibles du projet. Les variables
    # d'environnement permettent de choisir un profil ou un modele au lancement
    # sans modifier le depot entre deux machines/experiences.
    profile = os.getenv("RECEPTION_RAG_SERVER_PROFILE", cfg["server"])
    if profile not in cfg["servers"]:
        choices = ", ".join(sorted(cfg["servers"]))
        raise ValueError(
            f"Profil serveur inconnu: {profile!r}. Profils disponibles: {choices}"
        )
    cfg["server"] = profile
    cfg["hw"] = dict(cfg["servers"][profile])

    hardware_overrides = {
        "RECEPTION_RAG_LLM_MODEL": "llm_model",
        "RECEPTION_RAG_LLM_BACKEND": "llm_backend",
        "RECEPTION_RAG_LLM_DEVICE": "llm_device",
        "RECEPTION_RAG_VLM_MODEL": "vlm_model",
        "RECEPTION_RAG_VLM_BACKEND": "vlm_backend",
        "RECEPTION_RAG_VLM_DEVICE": "vlm_device",
        "RECEPTION_RAG_ASR_DEVICE": "asr_device",
        "RECEPTION_RAG_EMBEDDING_DEVICE": "embedding_device",
    }
    for variable, key in hardware_overrides.items():
        value = os.getenv(variable)
        if value:
            cfg["hw"][key] = value

    if value := os.getenv("RECEPTION_RAG_LLM_BASE_URL"):
        cfg["llm"]["base_url"] = value
    if value := os.getenv("RECEPTION_RAG_LLM_API_KEY"):
        cfg["llm"]["api_key"] = value
    if value := os.getenv("RECEPTION_RAG_LLM_TIMEOUT_SECONDS"):
        cfg["llm"]["timeout_seconds"] = float(value)

    embedding_overrides = {
        "RECEPTION_RAG_EMBEDDING_BACKEND": "backend",
        "RECEPTION_RAG_EMBEDDING_MODEL": "model",
        "RECEPTION_RAG_EMBEDDING_BASE_URL": "base_url",
        "RECEPTION_RAG_EMBEDDING_API_KEY": "api_key",
    }
    for variable, key in embedding_overrides.items():
        value = os.getenv(variable)
        if value:
            cfg["embeddings"][key] = value
    if value := os.getenv("RECEPTION_RAG_EMBEDDING_DENSE_DIM"):
        cfg["embeddings"]["dense_dim"] = int(value)
    if value := os.getenv("RECEPTION_RAG_EMBEDDING_USE_SPARSE"):
        cfg["embeddings"]["use_sparse"] = value.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    # Le VLM peut utiliser une API distante pendant les evaluations sans
    # modifier la configuration du LLM qui sert les reponses du RAG.
    cfg["vlm_api"] = {
        "base_url": os.getenv(
            "RECEPTION_RAG_VLM_BASE_URL", cfg["llm"]["base_url"]
        ),
        "api_key": os.getenv(
            "RECEPTION_RAG_VLM_API_KEY", cfg["llm"]["api_key"]
        ),
        "mode": os.getenv("RECEPTION_RAG_VLM_API_MODE", "local_vllm"),
    }
    if value := os.getenv("RECEPTION_RAG_QDRANT_PATH"):
        cfg["qdrant"]["mode"] = "local"
        cfg["qdrant"]["path"] = value
    if value := os.getenv("RECEPTION_RAG_LLM_ENABLE_THINKING"):
        cfg["llm"]["enable_thinking"] = value.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    return cfg


@lru_cache
def hotel_software_config() -> dict:
    with open(ROOT / "config" / "hotel_software.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["active"] = cfg["software"][cfg["active_software"]]
    return cfg


@lru_cache
def taxonomy_config() -> dict:
    """Charge et valide les taxonomies du logiciel et de l’établissement."""

    with open(ROOT / "config" / "taxonomy.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for domain in ("software", "hotel"):
        topics = cfg.get(domain)
        if not isinstance(topics, dict) or not topics:
            raise ValueError(f"Taxonomie absente ou vide pour le domaine {domain}")
        for topic, subtopics in topics.items():
            if not isinstance(topic, str) or not topic.strip():
                raise ValueError(f"Topic invalide dans la taxonomie {domain}")
            if not isinstance(subtopics, list) or any(
                not isinstance(value, str) or not value.strip() for value in subtopics
            ):
                raise ValueError(
                    f"Sous-topics invalides pour {domain}.{topic}"
                )
            if len(subtopics) != len(set(subtopics)):
                raise ValueError(f"Sous-topics dupliques pour {domain}.{topic}")
    return cfg


def taxonomy_contains(domain: str, topic: str, subtopic: str | None = None) -> bool:
    """Indique si une categorie appartient a la taxonomie de reference."""

    topics = taxonomy_config().get(domain, {})
    if topic not in topics:
        return False
    return subtopic is None or subtopic in topics[topic]


def path(relative: str) -> Path:
    """Chemin absolu depuis la racine du projet."""
    return ROOT / relative
