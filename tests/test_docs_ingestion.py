import sys
from types import SimpleNamespace

from src.ingestion.hotel_software.generic_connector.docs import (
    load,
    parse_html_file,
    parse_markdown_file,
    parse_pdf_file,
)


def test_long_html_paragraph_is_not_cut_and_keeps_all_data(tmp_path):
    original = "A" * 2105
    file = tmp_path / "long.html"
    file.write_text(
        f"<html><head><title>T</title></head><body><h1>Réservation</h1><p>{original}</p></body></html>",
        encoding="utf-8",
    )

    documents = parse_html_file(file)

    assert len(documents) == 1
    assert "".join(document.content for document in documents) == original
    assert all(document.topic == "reservations" for document in documents)
    assert all(document.status == "review_required" for document in documents)


def test_html_manifest_preserves_provenance_and_verified_status(tmp_path):
    file = tmp_path / "guide.html"
    file.write_text(
        "<html><head><title>Ignore</title></head><body>"
        "<h1>Facturation</h1><p>Modifier une facture deja creee.</p>"
        "<script>contenu parasite</script></body></html>",
        encoding="utf-8",
    )
    (tmp_path / "_sources.yaml").write_text(
        """sources:
  guide.html:
    source_id: guide-officiel
    title: Guide officiel
    url: https://example.test/guide
    version: 2026-08
    status: verified
    authority: official
""",
        encoding="utf-8",
    )

    documents = load(tmp_path)

    assert len(documents) == 1
    assert documents[0].breadcrumb == "Guide officiel > Facturation"
    assert documents[0].topic == "billing"
    assert documents[0].status == "verified"
    assert documents[0].url == "https://example.test/guide"
    assert "parasite" not in documents[0].content


def test_markdown_is_split_on_headings(tmp_path):
    file = tmp_path / "guide.md"
    file.write_text(
        "# Reservations\nCreer un dossier de reservation.\n"
        "## Paiement\nEnregistrer un reglement.\n",
        encoding="utf-8",
    )

    documents = parse_markdown_file(file)

    assert [document.title for document in documents] == ["Reservations", "Paiement"]
    assert [document.topic for document in documents] == ["reservations", "payments"]


def test_pdf_keeps_heading_page_and_citation(tmp_path, monkeypatch):
    class FakePage:
        def extract_text(self, visitor_text=None):
            if visitor_text is not None:
                visitor_text("Reservation", None, [0, 0, 0, 0, 70, 600], None, 16)
            return "A\nReservation\nCreer puis enregistrer le dossier.\n"

    class FakeReader:
        def __init__(self, _file):
            self.pages = [FakePage()]

    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=FakeReader))
    file = tmp_path / "guide.pdf"
    file.write_bytes(b"fake")

    documents = parse_pdf_file(
        file,
        source_id="guide-pdf",
        metadata={"title": "Guide PDF", "status": "verified"},
    )

    assert len(documents) == 1
    assert documents[0].title == "Reservation"
    assert documents[0].page_start == 1
    assert documents[0].page_end == 1
    assert documents[0].citation().endswith("(p. 1)")
