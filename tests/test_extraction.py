import pytest

from app.services.extraction import (
    EmptyDocumentError,
    UnreadableDocumentError,
    UnsupportedFileTypeError,
    extract_text,
)
from tests.fakes import make_pdf


def test_extracts_txt() -> None:
    assert extract_text("notes.txt", b"  Hello world.  ") == "Hello world."


def test_txt_falls_back_to_latin1() -> None:
    assert extract_text("old.txt", "caf\xe9".encode("latin-1")) == "caf\xe9"


def test_extension_check_is_case_insensitive() -> None:
    assert extract_text("NOTES.TXT", b"ok") == "ok"


def test_extracts_pdf_text() -> None:
    assert "Hello PDF world" in extract_text("doc.pdf", make_pdf("Hello PDF world"))


def test_rejects_unsupported_type() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        extract_text("image.png", b"\x89PNG")


def test_rejects_empty_text() -> None:
    with pytest.raises(EmptyDocumentError):
        extract_text("blank.txt", b"   \n  ")


def test_rejects_corrupt_pdf() -> None:
    with pytest.raises((UnreadableDocumentError, EmptyDocumentError)):
        extract_text("broken.pdf", b"this is not a pdf")
