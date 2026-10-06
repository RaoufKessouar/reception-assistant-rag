"""La configuration fonctionnelle fait partie du contrat du projet."""

import pytest

from src.config import settings, taxonomy_config, taxonomy_contains


def test_taxonomies_software_et_hotel_sont_chargees():
    taxonomy = taxonomy_config()

    assert taxonomy["version"] == 1
    assert "reservations" in taxonomy["software"]
    assert "guest_journey" in taxonomy["hotel"]


def test_categories_cles_du_projet_sont_presentes():
    assert taxonomy_contains("software", "payments", "refund")
    assert taxonomy_contains("software", "check_in_check_out", "no_show")
    assert taxonomy_contains("hotel", "guest_journey", "late_check_out")
    assert taxonomy_contains("hotel", "incidents", "key_card")


def test_categorie_inconnue_est_refusee():
    assert not taxonomy_contains("hotel", "restaurant", "menu")
    assert not taxonomy_contains("unknown", "payments")


def test_server_profile_can_be_selected_without_editing_yaml(monkeypatch):
    monkeypatch.setenv("RECEPTION_RAG_SERVER_PROFILE", "server-a")
    settings.cache_clear()
    try:
        cfg = settings()
        assert cfg["server"] == "server-a"
        assert cfg["hw"]["vlm_device"] == "cuda:0"
    finally:
        settings.cache_clear()


def test_model_and_device_can_be_overridden_for_an_experiment(monkeypatch):
    monkeypatch.setenv("RECEPTION_RAG_LLM_MODEL", "Qwen/Qwen3-14B")
    monkeypatch.setenv("RECEPTION_RAG_LLM_DEVICE", "cuda:0")
    monkeypatch.setenv("RECEPTION_RAG_LLM_ENABLE_THINKING", "false")
    settings.cache_clear()
    try:
        cfg = settings()
        assert cfg["hw"]["llm_model"] == "Qwen/Qwen3-14B"
        assert cfg["hw"]["llm_device"] == "cuda:0"
        assert cfg["llm"]["enable_thinking"] is False
    finally:
        settings.cache_clear()


def test_vlm_api_configuration_is_separate_from_llm(monkeypatch):
    monkeypatch.setenv("RECEPTION_RAG_VLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("RECEPTION_RAG_VLM_API_KEY", "secret-for-test")
    monkeypatch.setenv("RECEPTION_RAG_VLM_API_MODE", "remote_json")
    settings.cache_clear()
    try:
        cfg = settings()
        assert cfg["vlm_api"] == {
            "base_url": "https://example.test/v1",
            "api_key": "secret-for-test",
            "mode": "remote_json",
        }
        assert cfg["llm"]["api_key"] == "EMPTY"
    finally:
        settings.cache_clear()


def test_portable_api_backends_can_be_configured_without_a_gpu(monkeypatch):
    monkeypatch.setenv("RECEPTION_RAG_SERVER_PROFILE", "local-api")
    monkeypatch.setenv("RECEPTION_RAG_LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("RECEPTION_RAG_LLM_API_KEY", "llm-secret")
    monkeypatch.setenv("RECEPTION_RAG_LLM_MODEL", "remote-chat-model")
    monkeypatch.setenv("RECEPTION_RAG_EMBEDDING_BACKEND", "openai_compatible")
    monkeypatch.setenv("RECEPTION_RAG_EMBEDDING_BASE_URL", "https://embed.example/v1")
    monkeypatch.setenv("RECEPTION_RAG_EMBEDDING_API_KEY", "embedding-secret")
    monkeypatch.setenv("RECEPTION_RAG_EMBEDDING_MODEL", "remote-embedding-model")
    monkeypatch.setenv("RECEPTION_RAG_EMBEDDING_DENSE_DIM", "768")
    monkeypatch.setenv("RECEPTION_RAG_EMBEDDING_USE_SPARSE", "false")
    settings.cache_clear()
    try:
        cfg = settings()
        assert cfg["server"] == "local-api"
        assert cfg["hw"]["llm_backend"] == "api"
        assert cfg["hw"]["llm_model"] == "remote-chat-model"
        assert cfg["llm"]["base_url"] == "https://llm.example/v1"
        assert cfg["embeddings"]["backend"] == "openai_compatible"
        assert cfg["embeddings"]["base_url"] == "https://embed.example/v1"
        assert cfg["embeddings"]["api_key"] == "embedding-secret"
        assert cfg["embeddings"]["model"] == "remote-embedding-model"
        assert cfg["embeddings"]["dense_dim"] == 768
        assert cfg["embeddings"]["use_sparse"] is False
    finally:
        settings.cache_clear()


def test_unknown_server_profile_is_rejected(monkeypatch):
    monkeypatch.setenv("RECEPTION_RAG_SERVER_PROFILE", "unknown")
    settings.cache_clear()
    try:
        with pytest.raises(ValueError, match="Profil serveur inconnu"):
            settings()
    finally:
        settings.cache_clear()
