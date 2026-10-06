from src.normalization.pipeline import (
    build_report,
    clean_text,
    deduplicate,
    normalize_document,
)
from src.schema import CanonicalDocument


def _document(**changes):
    values = {
        "id": "doc-1",
        "content": "Cliquez sur Modifier pour modifier une réservation.",
        "title": "Modifier une réservation",
        "domain": "software",
        "topic": "modifier-une-reservation",
        "source_type": "video",
        "source_id": "video-1",
        "source_path": "data/raw/software_videos/video.mp4",
        "authority": "official",
        "status": "verified",
        "timestamp_start": 1,
        "timestamp_end": 2,
        "screenshot": "data/processed/frame.png",
        "screenshots": ["data/processed/frame.png"],
        "software": "generic_connector",
    }
    values.update(changes)
    return CanonicalDocument(**values)


def test_clean_text_preserves_structure_and_removes_noise():
    assert clean_text("  Titre  \r\n\r\n\r\n -  Étape\t1  \x00") == "Titre\n\n- Étape 1"


def test_clean_text_removes_pdf_generator_footer_only():
    assert (
        clean_text("Réponse utile\n\nPowered by TCPDF (www.tcpdf.org)")
        == "Réponse utile"
    )


def test_video_procedure_becomes_taxonomic_metadata():
    document = normalize_document(_document())

    assert document.topic == "reservations"
    assert document.subtopic == "modify_reservation"
    assert document.source_category == "modifier-une-reservation"
    assert document.content_hash
    assert document.breadcrumb.startswith("logiciel de gestion hôtelière > reservations")
    assert document.breadcrumb.count("Modifier une réservation") == 1


def test_title_has_priority_over_incidental_terms_in_body():
    user_faq = normalize_document(
        _document(
            title="Comment ajouter un nouvel utilisateur ?",
            content="Ouvrez le profil. Cet utilisateur pourra ensuite consulter une facture.",
            topic="billing",
            subtopic="faq_category_23",
            source_type="qa",
        )
    )
    email_video = normalize_document(
        _document(
            title="Paramétrage des modèles d'e-mails",
            content="Le modèle peut contenir une facture et un règlement.",
            topic="parametrage-des-modeles-d-e-mails",
        )
    )

    assert user_faq.topic == "users"
    assert user_faq.subtopic is None
    assert email_video.topic == "settings"
    assert email_video.subtopic is None


def test_generic_video_title_uses_step_content_as_fallback():
    document = normalize_document(
        _document(
            title="Formation à l'utilisation réception",
            content="Enregistrer un règlement par carte bancaire dans le dossier.",
            topic="formation-a-l-utilisation-reception",
        )
    )

    assert document.topic == "payments"
    assert document.subtopic == "payment"


def test_hotel_legacy_category_is_mapped_to_reference_taxonomy():
    document = normalize_document(
        _document(
            id="hotel-1",
            content="Effectuez le check-in.",
            title="Check-in standard",
            domain="hotel",
            topic="checkin",
            subtopic="standard_checkin",
            source_type="internal_sop",
            source_id="hotel-checkin-001",
            source_path="knowledge/hotel/check-in-standard.yaml",
            authority="verified_internal",
            last_reviewed="2026-08-25",
            timestamp_start=None,
            timestamp_end=None,
            screenshot=None,
            screenshots=[],
            software=None,
            hotel_id="hotel_reference",
        )
    )

    assert document.topic == "guest_journey"
    assert document.subtopic == "check_in"
    assert document.source_category == "checkin/standard_checkin"
    assert document.breadcrumb.count("Check-in standard") == 1


def test_deduplication_keeps_the_stronger_authority():
    community = normalize_document(
        _document(
            id="community",
            source_id="community",
            source_type="qa",
            authority="community",
        )
    )
    official = normalize_document(
        _document(id="official", source_id="official", source_type="official_doc")
    )

    kept, duplicates = deduplicate([community, official])

    assert [document.id for document in kept] == ["official"]
    assert duplicates == [
        {
            "dropped_id": "community",
            "kept_id": "official",
            "content_hash": official.content_hash,
        }
    ]


def test_report_is_ready_when_metadata_taxonomy_and_authority_are_valid():
    original = _document()
    normalized = normalize_document(original)
    report = build_report(inputs=[original], outputs=[normalized], duplicates=[])

    assert report["ready_for_chunking"] is True
    assert report["quality"]["missing_metadata"] == []
    assert report["quality"]["taxonomy_errors"] == []
    assert report["quality"]["authority_errors"] == []
