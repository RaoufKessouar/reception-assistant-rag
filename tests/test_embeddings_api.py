from types import SimpleNamespace

import pytest

from src.retrieval import embeddings


class _EmbeddingsEndpoint:
    def __init__(self, vectors):
        self.vectors = vectors
        self.request = None

    def create(self, **request):
        self.request = request
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=vector)
                for index, vector in enumerate(self.vectors)
            ]
        )


def _configure(monkeypatch, endpoint, *, dense_dim=3):
    monkeypatch.setattr(
        embeddings,
        "settings",
        lambda: {
            "embeddings": {
                "backend": "openai_compatible",
                "model": "remote-embedding-model",
                "dense_dim": dense_dim,
            }
        },
    )
    monkeypatch.setattr(
        embeddings,
        "_api_client",
        lambda: SimpleNamespace(embeddings=endpoint),
    )


def test_api_embeddings_return_dense_vectors_without_local_model(monkeypatch):
    endpoint = _EmbeddingsEndpoint([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    _configure(monkeypatch, endpoint)

    result = embeddings.encode(["bonjour", "reservation"])

    assert result == [
        {"dense": [0.1, 0.2, 0.3]},
        {"dense": [0.4, 0.5, 0.6]},
    ]
    assert endpoint.request == {
        "model": "remote-embedding-model",
        "input": ["bonjour", "reservation"],
    }


def test_api_embeddings_reject_an_incompatible_dimension(monkeypatch):
    endpoint = _EmbeddingsEndpoint([[0.1, 0.2]])
    _configure(monkeypatch, endpoint, dense_dim=3)

    with pytest.raises(ValueError, match="dimension retournee"):
        embeddings.encode(["bonjour"])
