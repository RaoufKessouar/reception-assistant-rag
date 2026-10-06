import json

from src.evaluation import retrieval_eval
from src.retrieval.retriever import Hit
from src.schema import CanonicalDocument


def test_recall_counts_all_expected_sources_and_unanswerable(monkeypatch):
    dataset = [
        {
            "question": "mixte",
            "expected_sources": ["hotel-a", "software-b"],
            "category": "mixed",
        },
        {"question": "inconnue", "expected_sources": [], "category": "unanswerable"},
    ]
    document = CanonicalDocument(
        id="a1",
        content="x",
        domain="hotel",
        topic="x",
        source_type="internal_sop",
        source_id="hotel-a",
        authority="verified_internal",
        status="verified",
    )
    monkeypatch.setattr(retrieval_eval, "load_dataset", lambda file: dataset)
    monkeypatch.setattr(
        retrieval_eval.retriever, "search", lambda *args, **kwargs: [Hit(document, 0.9)]
    )

    result = retrieval_eval.evaluate()
    assert result["metrics"]["mixed"]["hit_rate@5"] == 1.0
    assert result["metrics"]["mixed"]["recall@5"] == 0.5
    assert result["metrics"]["UNANSWERABLE"]["abstention_accuracy"] == 0.0
    assert result["failure_count"] == 1
    assert len(result["queries"]) == 2
    assert result["queries"][0]["retrieved"][0]["score"] == 0.9


def test_save_run_writes_complete_json(monkeypatch, tmp_path):
    result = {
        "generated_at": "2026-08-28T12:00:00+00:00",
        "configuration": {"retriever": "dense", "k": 5},
        "metrics": {"GLOBAL": {"recall@5": 0.8}},
        "failures": [],
        "queries": [],
    }
    monkeypatch.setattr(retrieval_eval, "path", lambda value: tmp_path / value)

    output = retrieval_eval.save_run(result)

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8"))["metrics"] == result["metrics"]


def test_dataset_audit_checks_balance_sources_and_unanswerable():
    software = CanonicalDocument(
        id="software",
        content="x",
        domain="software",
        topic="general",
        source_type="qa",
        source_id="software-source",
        authority="official",
        status="verified",
    )
    hotel = software.model_copy(
        update={
            "id": "hotel",
            "domain": "hotel",
            "source_type": "internal_sop",
            "source_id": "hotel-source",
            "authority": "verified_internal",
        }
    )
    dataset = []
    for index in range(25):
        dataset.append(
            {
                "question": f"logiciel de gestion hôtelière {index}",
                "expected_sources": ["software-source"] if index else [],
                "category": "software",
                "question_type": "direct" if index else "inconnue",
                "answerable": bool(index),
            }
        )
        dataset.append(
            {
                "question": f"Hotel {index}",
                "expected_sources": ["hotel-source"] if index else [],
                "category": "hotel",
                "question_type": "paraphrase" if index else "inconnue",
                "answerable": bool(index),
            }
        )

    report = retrieval_eval.audit_dataset(dataset, [software, hotel])

    assert report["questions"] == 50
    assert report["balanced_25_25"] is True
    assert report["unanswerable"] == 2
    assert report["errors"] == []
    assert report["ready_for_evaluation"] is True
