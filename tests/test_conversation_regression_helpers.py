import json

from scripts import evaluate_conversation_regression as regression
from scripts.evaluate_conversation_regression import (
    _expected_source_found,
    _load_cases,
    _missing_source_groups,
    _normalized_text,
    _source_identity,
)
from src.generation.answer import Answer
from src.retrieval.retriever import Hit
from src.schema import CanonicalDocument


def test_neutral_source_rename_keeps_same_identity():
    assert _source_identity("vendor-lexique-hotelier-2021-11") == (
        "lexique-hotelier-2021-11"
    )
    assert _source_identity("generic_connector-lexique-hotelier-2021-11") == (
        "lexique-hotelier-2021-11"
    )
    assert _source_identity("software-faq-84") == "faq-84"
    assert _source_identity("software-faq-84") == "faq-84"


def test_expected_source_accepts_neutral_alias_only_for_same_document():
    assert _expected_source_found(
        {"generic_connector-lexique-hotelier-2021-11"},
        {"vendor-lexique-hotelier-2021-11"},
    )
    assert not _expected_source_found({"procedure-a"}, {"procedure-b"})


def test_required_source_groups_require_one_source_from_each_group():
    groups = [["hotel-policy"], ["software-video", "software-faq"]]

    assert _missing_source_groups(groups, {"hotel-policy", "software-faq"}) == []
    assert _missing_source_groups(groups, {"hotel-policy"}) == [
        ["software-video", "software-faq"]
    ]


def test_dataset_loader_rejects_duplicate_ids(tmp_path):
    dataset = tmp_path / "duplicate.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"id": "same", "question": question})
            for question in ("A", "B")
        ),
        encoding="utf-8",
    )

    try:
        _load_cases(dataset)
    except ValueError as exc:
        assert "IDs dupliques" in str(exc)
    else:
        raise AssertionError("Un dataset avec des IDs dupliques doit etre refuse")


def test_generation_rejects_a_forbidden_cited_source(monkeypatch):
    document = CanonicalDocument(
        id="doc-1",
        content="Envoyer une facture par email.",
        title="Envoi de facture",
        domain="software",
        topic="billing",
        source_type="video",
        source_id="invoice-email",
        authority="official",
        status="verified",
    )
    hit = Hit(document=document, score=0.9)
    monkeypatch.setattr(
        regression.generation,
        "ask",
        lambda *args, **kwargs: Answer(
            question="Lien de paiement ?",
            text="Utilisez cette procédure [1].",
            hits=[hit],
            verification_decision="supported",
        ),
    )
    case = {
        "id": "forbidden-source",
        "question": "Lien de paiement ?",
        "answerable": True,
        "expected_domains": ["software"],
        "expected_sources": [],
        "forbidden_sources": ["invoice-email"],
    }

    result = regression._generation_result(case)

    assert result["automatic_pass"] is False
    assert result["failures"] == ["sources interdites citées : invoice-email"]


def test_forbidden_source_rejects_the_neutralized_alias(monkeypatch):
    document = CanonicalDocument(
        id="doc-faq-100",
        content="Parametrer une option.",
        title="Ajouter une option",
        domain="software",
        topic="configuration",
        source_type="qa",
        source_id="software-faq-100",
        authority="official",
        status="verified",
    )
    monkeypatch.setattr(
        regression.generation,
        "ask",
        lambda *args, **kwargs: Answer(
            question="Ajouter un extra au dossier ?",
            text="Utilisez cette procédure [1].",
            hits=[Hit(document=document, score=0.9)],
            verification_decision="supported",
        ),
    )
    case = {
        "id": "forbidden-neutral-alias",
        "question": "Ajouter un extra au dossier ?",
        "answerable": True,
        "expected_domains": ["software"],
        "expected_sources": [],
        "forbidden_sources": ["software-faq-100"],
    }

    result = regression._generation_result(case)

    assert result["automatic_pass"] is False
    assert result["failures"] == ["sources interdites citées : software-faq-100"]


def test_forbidden_phrase_matching_ignores_quotes_accents_and_punctuation():
    answer = 'Accédez à la section "Réservations" dans le logiciel.'

    assert _normalized_text("Accedez a la section Reservations") in _normalized_text(answer)


def test_generation_rejects_a_missing_required_phrase(monkeypatch):
    document = CanonicalDocument(
        id="doc-2",
        content="Informer ensuite le responsable.",
        title="Escalade",
        domain="hotel",
        topic="incident",
        source_type="internal_sop",
        source_id="escalation",
        authority="verified_internal",
        status="verified",
    )
    monkeypatch.setattr(
        regression.generation,
        "ask",
        lambda *args, **kwargs: Answer(
            question="Incident ?",
            text="Appelez la police [1].",
            hits=[Hit(document=document, score=0.9)],
            verification_decision="supported",
        ),
    )
    case = {
        "id": "required-phrase",
        "question": "Incident ?",
        "answerable": True,
        "expected_domains": ["hotel"],
        "required_phrases": ["responsable"],
    }

    result = regression._generation_result(case)

    assert result["automatic_pass"] is False
    assert result["failures"] == ["formulation attendue absente : responsable"]
