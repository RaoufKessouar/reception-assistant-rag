"""Tests minimaux : le schema est le contrat du projet, il doit tenir."""

from datetime import date

import pytest
from pydantic import ValidationError

from src.schema import CanonicalDocument, KnowledgeCard, ProceduralStep


def test_document_minimal():
    d = CanonicalDocument(
        id="a1",
        content="Le check-out est a 11h.",
        domain="hotel",
        topic="checkout",
        source_type="internal_sop",
        source_id="hotel-001",
        authority="verified_internal",
    )
    assert d.status == "draft"  # rien n'est verifie par defaut
    assert d.language == "fr"


def test_document_rejects_invalid_page_range():
    with pytest.raises(ValueError, match="page_end"):
        CanonicalDocument(
            id="doc-pages",
            content="Contenu",
            domain="software",
            topic="reservations",
            source_type="official_doc",
            source_id="guide",
            authority="official",
            page_start=4,
            page_end=2,
        )


def test_document_rejette_domaine_inconnu():
    with pytest.raises(ValidationError):
        CanonicalDocument(
            id="a2",
            content="x",
            domain="restaurant",
            topic="t",
            source_type="qa",
            source_id="s",
            authority="official",
        )


def test_breadcrumb_prefixe_le_texte_embedde():
    d = CanonicalDocument(
        id="a3",
        content="Ouvrez la fiche.",
        breadcrumb="Reservations > Modifier",
        domain="software",
        topic="reservations",
        source_type="official_doc",
        source_id="doc1",
        authority="official",
    )
    assert d.text_to_embed().startswith("Reservations > Modifier")


def test_card_vers_document():
    card = KnowledgeCard(
        id="hotel-checkout-001",
        title="Late check-out",
        category="checkout",
        situation="Le client veut partir plus tard.",
        procedure=["Verifier le planning", "Accorder jusqu'a 14h"],
        status="verified",
        last_reviewed=date(2026, 8, 25),
    )
    doc = card.to_document()
    assert doc.domain == "hotel"
    assert doc.authority == "verified_internal"
    assert "Verifier le planning" in doc.content
    assert doc.last_reviewed == "2026-08-25"


def test_card_verified_refuse_une_question_ouverte():
    with pytest.raises(ValidationError):
        KnowledgeCard(
            id="hotel-test-001",
            title="Test",
            category="test",
            situation="Situation de test.",
            procedure=["Action de test."],
            status="verified",
            review_questions=["Question non resolue"],
        )


def test_card_draft_peut_rester_incomplete():
    card = KnowledgeCard(
        id="hotel-test-002",
        title="Procedure a documenter",
        category="test",
        situation="Situation encore en entretien.",
        procedure=[],
        status="draft",
        review_questions=["Quelle procedure appliquer ?"],
    )
    assert card.procedure == []


def test_card_review_required_exige_une_procedure():
    with pytest.raises(ValidationError):
        KnowledgeCard(
            id="hotel-test-003",
            title="Procedure vide",
            category="test",
            situation="Situation de test.",
            procedure=[],
            status="review_required",
        )


def test_step_video_garde_timestamp_et_screenshot():
    step = ProceduralStep(
        procedure="modifier_les_dates",
        step_number=2,
        timestamp_start=45.1,
        timestamp_end=49.6,
        spoken_instruction="Puis vous cliquez ici.",
        instruction="Cliquez sur le bouton Modifier, en haut a droite.",
        screenshot_before="etape2.png",
        video_id="tuto_modifier_dates",
    )
    doc = step.to_document()
    assert doc.source_type == "video"
    assert doc.timestamp_start == 45.1
    assert doc.screenshot == "etape2.png"
    assert "00:45" in doc.citation()
    assert doc.status == "review_required"
