from qdrant_client import QdrantClient

from src.retrieval import vector_store
from src.schema import CanonicalDocument


def _document(identifier: str, status: str = "verified") -> CanonicalDocument:
    return CanonicalDocument(
        id=identifier,
        content="contenu",
        domain="hotel",
        topic="checkout",
        source_type="internal_sop",
        source_id=identifier,
        authority="verified_internal",
        status=status,
    )


def _memory_store(monkeypatch):
    memory = QdrantClient(":memory:")
    monkeypatch.setattr(vector_store, "client", lambda: memory)
    monkeypatch.setattr(
        vector_store.embeddings,
        "encode",
        lambda texts: [{"dense": [0.0] * 1024, "sparse": {1: 1.0}} for _ in texts],
    )
    vector_store.create_collection()
    return memory


def test_non_hex_document_id_can_be_indexed(monkeypatch):
    memory = _memory_store(monkeypatch)
    document = _document("hotel-checkout-001")
    assert vector_store.index([document]) == 1
    assert memory.retrieve("reception-rag", [document.qdrant_id()])


def test_deprecated_document_is_removed(monkeypatch):
    memory = _memory_store(monkeypatch)
    verified = _document("hotel-checkout-001")
    vector_store.index([verified])
    deprecated = verified.model_copy(update={"status": "deprecated"})
    assert vector_store.index([deprecated]) == 0
    assert memory.retrieve("reception-rag", [verified.qdrant_id()]) == []


def test_sync_removes_orphan_points(monkeypatch):
    memory = _memory_store(monkeypatch)
    old = _document("old")
    new = _document("new")
    vector_store.index([old])
    result = vector_store.sync([new])
    assert result == {"indexed": 1, "deleted": 1}
    assert memory.retrieve("reception-rag", [old.qdrant_id()]) == []


def test_audit_verifies_chunk_metadata(monkeypatch):
    _memory_store(monkeypatch)
    document = _document("hotel-checkout-001").model_copy(
        update={
            "content_hash": "hash",
            "parent_document_id": "hotel-checkout-001",
            "chunk_index": 1,
            "chunk_count": 1,
        }
    )
    vector_store.index([document])

    result = vector_store.audit()

    assert result["points"] == 1
    assert result["missing_metadata"] == []
    assert result["ready"] is True
