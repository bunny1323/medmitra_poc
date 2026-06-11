from __future__ import annotations

import fitz  # PyMuPDF
import pytest

from app.services import ingestion_service


# ---------------------------------------------------------------------
# Temporary test environment
# ---------------------------------------------------------------------

@pytest.fixture
def setup_temp_books_dir(tmp_path):
    """
    Create isolated books and registry folders for each test.

    This prevents tests from modifying the real:
        data/books/
        data/registry/documents.json
    """

    original_books_dir = ingestion_service.BOOKS_DIR
    original_registry_dir = ingestion_service.REGISTRY_DIR
    original_registry_file = ingestion_service.REGISTRY_FILE

    temp_books_dir = tmp_path / "books"
    temp_registry_dir = tmp_path / "registry"
    temp_registry_file = temp_registry_dir / "documents.json"

    temp_books_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_registry_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ingestion_service.BOOKS_DIR = temp_books_dir
    ingestion_service.REGISTRY_DIR = temp_registry_dir
    ingestion_service.REGISTRY_FILE = temp_registry_file

    # Create a tiny valid PDF for unit testing
    dummy_pdf_path = temp_books_dir / "test_guideline.pdf"

    document = fitz.open()
    page = document.new_page()

    page.insert_text(
        (50, 50),
        (
            "This is a test medical guideline. "
            "Severe chest pain is critical."
        ),
    )

    document.save(dummy_pdf_path)
    document.close()

    yield (
        temp_books_dir,
        temp_registry_file,
        dummy_pdf_path,
    )

    # Restore real project paths after the test
    ingestion_service.BOOKS_DIR = original_books_dir
    ingestion_service.REGISTRY_DIR = original_registry_dir
    ingestion_service.REGISTRY_FILE = original_registry_file


# ---------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------

def test_registry_load_and_save(
    setup_temp_books_dir,
):
    _, _, _ = setup_temp_books_dir

    registry = ingestion_service.load_registry()

    assert isinstance(registry, list)
    assert len(registry) == 0

    test_entry = {
        "source_id": "test-id",
        "display_name": "Test PDF",
        "original_filename": "test.pdf",
        "document_hash": "hash123",
        "ingestion_status": "ingested",
        "page_count": 1,
        "chunk_count": 1,
        "created_at": "now",
        "updated_at": "now",
        "error_message": None,
    }

    registry.append(test_entry)

    ingestion_service.save_registry(
        registry
    )

    reloaded_registry = (
        ingestion_service.load_registry()
    )

    assert len(reloaded_registry) == 1

    assert (
        reloaded_registry[0]["display_name"]
        == "Test PDF"
    )


# ---------------------------------------------------------------------
# SHA-256 duplicate-detection tests
# ---------------------------------------------------------------------

def test_sha256_hash(
    setup_temp_books_dir,
):
    _, _, dummy_pdf_path = setup_temp_books_dir

    first_hash = ingestion_service.calculate_sha256(
        dummy_pdf_path
    )

    second_hash = ingestion_service.calculate_sha256(
        dummy_pdf_path
    )

    assert len(first_hash) == 64
    assert first_hash == second_hash


# ---------------------------------------------------------------------
# Display-name formatting tests
# ---------------------------------------------------------------------

def test_display_name_generation():
    assert (
        ingestion_service.make_display_name(
            "icmr_stw_volume_1.pdf"
        )
        == "ICMR STW Volume 1"
    )

    assert (
        ingestion_service.make_display_name(
            "who_imci_chart_booklet.pdf"
        )
        == "WHO IMCI Chart Booklet"
    )

    assert (
        ingestion_service.make_display_name(
            "mild-headache-and-flu.pdf"
        )
        == "Mild Headache And Flu"
    )


# ---------------------------------------------------------------------
# Flexible ingestion and duplicate-skip tests
# ---------------------------------------------------------------------

def test_duplicate_skip(
    setup_temp_books_dir,
    monkeypatch,
):
    _, _, _ = setup_temp_books_dir

    def fake_ingest_pdf_file(
        file_path,
        source_id,
        display_name,
    ):
        """
        Simulate a successful PDF ingestion without:
            loading embedding models
            contacting Qdrant
            uploading vectors
        """
        return 5

    monkeypatch.setattr(
        ingestion_service,
        "ingest_pdf_file",
        fake_ingest_pdf_file,
    )

    monkeypatch.setattr(
        ingestion_service,
        "enable_hnsw_indexing",
        lambda: None,
    )

    # First run:
    # the PDF is new and should be registered
    first_result = ingestion_service.append_books()

    assert first_result["ingested_count"] == 1
    assert first_result["replaced_count"] == 0
    assert first_result["skipped_count"] == 0

    assert (
        first_result["results"][0]["status"]
        == "ingested"
    )

    assert (
        first_result["results"][0]["chunk_count"]
        == 5
    )

    # Second run:
    # the PDF hash is unchanged and should be skipped
    second_result = ingestion_service.append_books()

    assert second_result["ingested_count"] == 0
    assert second_result["replaced_count"] == 0
    assert second_result["skipped_count"] == 1

    assert (
        second_result["results"][0]["status"]
        == "skipped"
    )

    assert (
        second_result["results"][0]["reason"]
        == "Duplicate SHA-256 hash"
    )