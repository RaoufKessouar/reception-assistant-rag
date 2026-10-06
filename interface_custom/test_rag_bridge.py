from src.generation.answer import Answer as EngineAnswer
from src.retrieval.retriever import Hit
from src.schema import CanonicalDocument

from interface_custom.rag_bridge import _format_history, _normalise


def _hit(index: int, *, video: bool = False) -> Hit:
    document = CanonicalDocument(
        id=f"doc-{index}",
        content=f"Contenu vérifié {index}",
        title=f"Source {index}",
        breadcrumb="Réservations > Modifier",
        domain="software",
        topic="reservations",
        source_type="video" if video else "official_doc",
        source_id=f"source-{index}",
        authority="official",
        status="verified",
        timestamp_start=65.0 if video else None,
        screenshot="data/processed/capture.png" if video else None,
    )
    return Hit(document=document, score=0.8 + index / 100)


def test_normalise_preserves_real_citation_indices_and_media():
    result = EngineAnswer(
        question="Comment faire ?",
        text="Utiliser la première preuve [2], puis la seconde [5].",
        hits=[_hit(index, video=index == 5) for index in range(1, 6)],
        route="software",
        used_history=True,
        verification_decision="supported",
    )

    displayed = _normalise(result)

    assert [source.citation_index for source in displayed.sources] == [2, 5]
    assert displayed.sources[1].images == ("data/processed/capture.png",)
    assert displayed.sources[1].timestamp_start == 65.0
    assert displayed.route == "software"
    assert displayed.used_history is True


def test_history_excludes_non_messages_and_limits_complete_window():
    messages = [
        {"role": "system", "content": "ne pas transmettre"},
        {"role": "user", "content": "ancienne question"},
        {"role": "assistant", "content": "ancienne réponse"},
        {"role": "user", "content": "question récente"},
        {"role": "assistant", "content": "réponse récente"},
    ]

    history = _format_history(messages, max_turns=1)

    assert history == [
        {"role": "user", "content": "question récente"},
        {"role": "assistant", "content": "réponse récente"},
    ]
