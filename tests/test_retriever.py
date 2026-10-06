from types import SimpleNamespace

from src.retrieval import retriever
from src.retrieval.retriever import Hit, _rerank_title_matches, _select_diverse
from src.schema import CanonicalDocument


def _hit(
    document_id: str,
    source_id: str,
    score: float,
    title: str | None = None,
    domain: str = "software",
) -> Hit:
    document = CanonicalDocument(
        id=document_id,
        content="contenu",
        title=title,
        domain=domain,
        topic="general",
        source_type="qa",
        source_id=source_id,
        authority="official",
        status="verified",
    )
    return Hit(document=document, score=score)


def test_select_diverse_limits_chunks_from_the_same_source():
    hits = [
        _hit("a1", "source-a", 0.9),
        _hit("a2", "source-a", 0.8),
        _hit("b1", "source-b", 0.7),
        _hit("c1", "source-c", 0.6),
    ]

    selected = _select_diverse(hits, top_k=3, max_per_source=1)

    assert [hit.document.source_id for hit in selected] == [
        "source-a",
        "source-b",
        "source-c",
    ]


def test_select_diverse_preserves_score_order_and_top_k():
    hits = [
        _hit("a1", "source-a", 0.9),
        _hit("b1", "source-b", 0.8),
        _hit("c1", "source-c", 0.7),
    ]

    selected = _select_diverse(hits, top_k=2, max_per_source=1)

    assert [hit.score for hit in selected] == [0.9, 0.8]


def test_dense_reranking_slightly_promotes_an_exact_title_match():
    hits = [
        _hit("generic", "generic", 0.74, "Formation réception"),
        _hit("specific", "specific", 0.70, "Modifier une réservation"),
    ]

    reranked = _rerank_title_matches(hits, query="Comment modifier une réservation ?")

    assert [hit.document.id for hit in reranked] == ["specific", "generic"]
    assert reranked[0].score > reranked[1].score


def test_balanced_search_keeps_one_result_from_each_domain(monkeypatch):
    hotel = _hit("hotel", "hotel-policy", 0.60, domain="hotel")
    software = _hit("software", "software-procedure", 0.90)
    extra = _hit("software-extra", "software-extra", 0.80)

    def fake_search(query, *, domain=None, **kwargs):
        return [hotel] if domain == "hotel" else [software, extra]

    monkeypatch.setattr(retriever, "search", fake_search)

    hits = retriever.search_balanced("question mixte", top_k=3)

    assert {hit.document.domain for hit in hits} == {"hotel", "software"}
    assert len(hits) == 3


def test_hybrid_request_falls_back_to_dense_for_api_embeddings(monkeypatch):
    calls = []

    class _Client:
        @staticmethod
        def query_points(**request):
            calls.append(request)
            return SimpleNamespace(points=[])

    monkeypatch.setattr(retriever, "client", lambda: _Client())
    monkeypatch.setattr(
        retriever.embeddings,
        "encode_one",
        lambda query, is_query=False: {"dense": [0.1, 0.2, 0.3]},
    )
    monkeypatch.setattr(
        retriever,
        "settings",
        lambda: {
            "qdrant": {
                "top_k": 5,
                "collection": "reception-rag",
                "candidate_multiplier": 1,
                "max_chunks_per_source": 1,
                "min_score": 0.0,
            }
        },
    )

    assert retriever.search("question", hybrid=True) == []
    assert "query" in calls[0]
    assert "prefetch" not in calls[0]
