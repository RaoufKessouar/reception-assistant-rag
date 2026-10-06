import json

import pytest

from src.ingestion.hotel_software.generic_connector import qa
from src.ingestion.hotel_software.generic_connector.qa import load, parse_jsonl_file


def test_missing_qa_directory_is_optional(tmp_path):
    assert load(tmp_path / "absent") == []


def test_qa_keeps_question_and_answer_together(tmp_path):
    file = tmp_path / "faq.jsonl"
    file.write_text(
        json.dumps(
            {
                "question": "Comment ouvrir un dossier ?",
                "answer": "Utiliser le planning.",
                "topic": "reservations",
                "status": "verified",
                "authority": "official",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    documents = parse_jsonl_file(file)

    assert len(documents) == 1
    assert "Question : Comment ouvrir un dossier ?" in documents[0].content
    assert "Reponse : Utiliser le planning." in documents[0].content
    assert documents[0].status == "verified"


def test_qa_error_contains_file_and_line(tmp_path):
    file = tmp_path / "faq.jsonl"
    file.write_text('{"question": "Question sans reponse"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"faq\.jsonl:1: champ 'answer'"):
        parse_jsonl_file(file)


def test_pdf_faq_uses_manifest_and_keeps_all_pages_together(tmp_path, monkeypatch):
    category = tmp_path / "category-132"
    category.mkdir()
    pdf = category / "181-comment-passer-un-dossier-en-debiteur.pdf"
    pdf.write_bytes(b"%PDF-test")
    manifest = {
        "category_id": "132",
        "generated_at": "2026-08-25T12:00:00Z",
        "questions": [
            {
                "id": "181",
                "subcategory_id": "136",
                "title": "Comment passer un dossier en débiteur ?",
                "filename": pdf.name,
                "status": "downloaded",
                "size": 100,
            }
        ],
    }
    (category / "manifest-categorie-132.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(
        qa,
        "_extract_pdf_pages",
        lambda _file: [
            "FAQ logiciel de gestion hôtelière\nComment passer un dossier en débiteur ?\n\nÉtape une\nPage 1/2",
            "FAQ logiciel de gestion hôtelière\nÉtape deux\nPage 2/2",
        ],
    )

    documents = load(tmp_path)

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "Comment passer un dossier en débiteur ?"
    assert "Question : Comment passer un dossier en débiteur ?" in document.content
    assert "Étape une" in document.content
    assert "Étape deux" in document.content
    assert document.content.count("Question :") == 1
    assert "FAQ logiciel de gestion hôtelière" not in document.content
    assert "Page 1/2" not in document.content
    assert document.page_start == 1
    assert document.page_end == 2
    assert document.source_id == "software-faq-181"
    assert document.authority == "official"
    assert document.status == "verified"
    assert document.subtopic == "faq_category_132"
    assert "question=181&topic=132&topic_quest=136" in document.url


def test_manifest_report_keeps_failed_exports_visible(tmp_path):
    category = tmp_path / "category-20"
    category.mkdir()
    (category / "manifest-categorie-20.json").write_text(
        json.dumps(
            {
                "category_id": "20",
                "questions": [
                    {
                        "id": "26",
                        "title": "Comment supprimer un utilisateur ?",
                        "status": "error",
                        "error": "réponse non PDF",
                    },
                    {
                        "id": "37",
                        "title": "Comment modifier les droits ?",
                        "filename": "37-droits.pdf",
                        "status": "downloaded",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = qa.manifest_report(tmp_path)

    assert report["manifests"] == 1
    assert report["expected"] == 2
    assert report["downloaded"] == 1
    assert report["errors"] == [
        {
            "category_id": "20",
            "id": "26",
            "title": "Comment supprimer un utilisateur ?",
            "error": "réponse non PDF",
        }
    ]


def test_verified_jsonl_recovery_clears_manifest_error(tmp_path):
    category = tmp_path / "category-20"
    category.mkdir()
    (category / "manifest-categorie-20.json").write_text(
        json.dumps(
            {
                "category_id": "20",
                "questions": [
                    {
                        "id": "26",
                        "title": "Comment supprimer un utilisateur ?",
                        "status": "error",
                        "error": "réponse non PDF",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "recovered.jsonl").write_text(
        json.dumps(
            {
                "question": "Comment supprimer un utilisateur ?",
                "answer": "Choisir l'utilisateur puis confirmer.",
                "source_id": "software-faq-26",
                "status": "verified",
                "authority": "official",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = qa.manifest_report(tmp_path)

    assert report["expected"] == 1
    assert report["downloaded"] == 1
    assert report["recovered"] == 1
    assert report["errors"] == []
