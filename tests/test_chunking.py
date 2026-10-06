import json

from src.chunking.pipeline import build_report, chunk_document, load, run
from src.schema import CanonicalDocument


def _document(**changes):
    values = {
        "id": "doc-1",
        "content": "Premier paragraphe.\n\nDeuxième paragraphe.",
        "title": "Document de test",
        "breadcrumb": "logiciel de gestion hôtelière > reservations > Document de test",
        "domain": "software",
        "topic": "reservations",
        "source_type": "official_doc",
        "source_id": "source-1",
        "source_path": "data/raw/source.pdf",
        "authority": "official",
        "status": "verified",
        "software": "generic_connector",
        "content_hash": "original",
    }
    values.update(changes)
    return CanonicalDocument(**values)


def test_documentation_is_split_on_paragraph_boundaries():
    chunks = chunk_document(_document(), documentation_max_chars=25)

    assert [chunk.content for chunk in chunks] == [
        "Premier paragraphe.",
        "Deuxième paragraphe.",
    ]
    assert [chunk.chunk_index for chunk in chunks] == [1, 2]
    assert all(chunk.chunk_count == 2 for chunk in chunks)
    assert all(chunk.parent_document_id == "doc-1" for chunk in chunks)
    assert chunks[0].id != "doc-1"


def test_qa_pair_is_never_split_even_when_long():
    content = "Question : Pourquoi ?\n\nReponse :\n" + ("Réponse utile. " * 100)
    chunks = chunk_document(
        _document(source_type="qa", content=content),
        documentation_max_chars=50,
    )

    assert len(chunks) == 1
    assert chunks[0].content == content.strip()
    assert chunks[0].id == "doc-1"
    assert chunks[0].chunk_index == chunks[0].chunk_count == 1


def test_video_action_keeps_timestamps_and_screenshot():
    chunks = chunk_document(
        _document(
            source_type="video",
            content="Cliquez sur Enregistrer.",
            timestamp_start=12.0,
            timestamp_end=15.0,
            screenshot="data/processed/frame.jpg",
            screenshots=["data/processed/frame.jpg"],
        )
    )

    assert len(chunks) == 1
    assert chunks[0].timestamp_start == 12.0
    assert chunks[0].screenshot == "data/processed/frame.jpg"


def test_hotel_procedure_stays_complete_when_reasonable():
    chunks = chunk_document(
        _document(
            domain="hotel",
            topic="services",
            subtopic="wifi",
            source_type="internal_sop",
            authority="verified_internal",
            software=None,
            hotel_id="hotel_reference",
            last_reviewed="2026-08-25",
        ),
        hotel_procedure_max_chars=1000,
    )

    assert len(chunks) == 1
    assert chunks[0].parent_document_id == "doc-1"


def test_report_detects_valid_atomic_qa_and_video_chunks():
    qa = chunk_document(
        _document(
            id="qa",
            source_id="qa",
            source_type="qa",
            content="Question : Q ?\n\nReponse : R.",
        )
    )[0]
    video = chunk_document(
        _document(
            id="video",
            source_id="video",
            source_type="video",
            timestamp_start=1.0,
            timestamp_end=2.0,
            screenshot="data/frame.jpg",
        )
    )[0]

    report = build_report([qa, video], [qa, video])

    assert report["quality"]["qa_pair_errors"] == []
    assert report["quality"]["video_action_errors"] == []
    assert report["ready_for_vectorization"] is True


def test_run_and_load_round_trip(tmp_path):
    normalized = tmp_path / "normalized.jsonl"
    normalized.write_text(
        json.dumps(_document().model_dump(mode="json"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "chunks.jsonl"
    report = tmp_path / "report.json"

    _, chunks, result = run(
        input_file=normalized,
        output_file=output,
        report_file=report,
    )

    assert result["ready_for_vectorization"] is True
    assert load(output) == chunks
