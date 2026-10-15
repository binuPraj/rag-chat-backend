import io
from pathlib import PurePath

from pypdf import PdfReader
from pypdf.errors import PyPdfError


class DocumentError(ValueError):
    """Base class for problems with an uploaded document."""


class UnsupportedFileTypeError(DocumentError):
    """The file extension is not one we can read."""


class UnreadableDocumentError(DocumentError):
    """The file is corrupt or could not be parsed."""


class EmptyDocumentError(DocumentError):
    """The file was readable but contains no text (e.g. a scanned PDF)."""


SUPPORTED_EXTENSIONS = (".pdf", ".txt")


def extract_text(filename: str, data: bytes) -> str:
    suffix = PurePath(filename).suffix.lower()
    if suffix == ".txt":
        text = _decode_text(data)
    elif suffix == ".pdf":
        text = _extract_pdf(data)
    else:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix or filename}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    text = text.strip()
    if not text:
        raise EmptyDocumentError("No extractable text found in the document")
    return text


def _decode_text(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _extract_pdf(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except PyPdfError as exc:
        raise UnreadableDocumentError(f"Could not read PDF: {exc}") from exc
    return "\n\n".join(pages)
