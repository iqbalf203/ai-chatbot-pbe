import os
import re
from pathlib import Path

from fastapi import UploadFile
from pypdf import PdfReader
from docx import Document
from openpyxl import load_workbook

from app.core.config import ALLOWED_EXTENSIONS, CHUNK_OVERLAP, CHUNK_SIZE, MAX_FILE_SIZE_BYTES


def validate_file(file: UploadFile, max_size_bytes: int = MAX_FILE_SIZE_BYTES) -> None:
    if not file.filename:
        raise ValueError("A file name is required.")

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            "Unsupported file type. Supported types: PDF, DOCX, and TXT."
        )

    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size <= 0:
        raise ValueError("Uploaded file is empty.")

    if file_size > max_size_bytes:
        raise ValueError(
            f"File exceeds the maximum allowed size of {max_size_bytes} bytes."
        )


def clean_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t\f\v]+", " ", cleaned)
    cleaned = re.sub(r"\s*\n\s*", "\n", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def extract_text_from_pdf(file: UploadFile) -> str:
    try:
        reader = PdfReader(file.file)
        pages = []

        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                pages.append(text)

        document_text = "\n".join(pages)
        if not document_text.strip():
            raise ValueError("The PDF file is empty or no readable text could be extracted.")

        return clean_text(document_text)
    except Exception as exc:  # pragma: no cover - surfaced to caller
        raise ValueError(f"Failed to extract text from PDF: {exc}") from exc


def extract_text_from_docx(file: UploadFile) -> str:
    try:
        file.file.seek(0)
        document = Document(file.file)
        paragraphs = [paragraph.text for paragraph in document.paragraphs]
        text = "\n".join(paragraphs)

        if not text.strip():
            raise ValueError("The DOCX file is empty or no readable text could be extracted.")

        return clean_text(text)
    except Exception as exc:  # pragma: no cover - surfaced to caller
        raise ValueError(f"Failed to extract text from DOCX: {exc}") from exc


def extract_text_from_txt(file: UploadFile) -> str:
    try:
        file.file.seek(0)
        raw_text = file.file.read()
        decoded = raw_text.decode("utf-8-sig", errors="strict")
        if not decoded.strip():
            raise ValueError("The TXT file is empty.")
        return clean_text(decoded)
    except UnicodeDecodeError:
        file.file.seek(0)
        raw_text = file.file.read()
        decoded = raw_text.decode("utf-8-sig", errors="replace")
        if not decoded.strip():
            raise ValueError("The TXT file is empty.")
        return clean_text(decoded)
    except Exception as exc:  # pragma: no cover - surfaced to caller
        raise ValueError(f"Failed to extract text from TXT: {exc}") from exc


def extract_text_from_excel(file: UploadFile) -> str:
    try:
        file.file.seek(0)
        workbook = load_workbook(file.file, read_only=True, data_only=True)
        sheet_text: list[str] = []

        for worksheet in workbook.worksheets:
            for row in worksheet.iter_rows(values_only=True):
                values = [str(value).strip() for value in row if value is not None and str(value).strip()]
                if values:
                    sheet_text.append(" | ".join(values))

        workbook.close()

        combined_text = "\n".join(sheet_text)
        if not combined_text.strip():
            raise ValueError("The Excel file is empty or no readable text could be extracted.")

        return clean_text(combined_text)
    except Exception as exc:  # pragma: no cover - surfaced to caller
        raise ValueError(f"Failed to extract text from Excel: {exc}") from exc


def extract_text(file: UploadFile) -> str:
    extension = Path(file.filename or "").suffix.lower()

    if extension == ".pdf":
        return extract_text_from_pdf(file)
    if extension == ".docx":
        return extract_text_from_docx(file)
    if extension == ".txt":
        return extract_text_from_txt(file)
    if extension == ".xlsx":
        return extract_text_from_excel(file)

    raise ValueError("Unsupported file type. Supported types: PDF, DOCX, TXT, and XLSX.")


def split_text_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    if not text or not text.strip():
        raise ValueError("Document text is empty after cleaning.")

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("Chunk overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than the chunk size.")

    normalized = clean_text(text)
    chunks: list[str] = []
    start = 0

    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        chunk = normalized[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(normalized):
            break

        start = max(start + 1, end - chunk_overlap)

    if not chunks:
        raise ValueError("No chunks were created from the document text.")

    return chunks
