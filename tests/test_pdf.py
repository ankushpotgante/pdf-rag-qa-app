from __future__ import annotations

from pathlib import Path

import fitz
import pytest
from langchain_core.documents import Document

import main

def _create_pdf(path: Path, pages: list[str]) -> None:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()

def test_read_pdfs_raises_when_no_pdf_exists(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No PDF files found"):
        main.read_pdfs_from_data_folder(tmp_path)


def test_read_pdfs_reads_text_pages(tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"
    _create_pdf(pdf_path, ["First page", "Second page"])

    docs = main.read_pdfs_from_data_folder(tmp_path)

    assert len(docs) == 2
    assert docs[0].page_content == "First page"
    assert docs[0].metadata["page"] == 1
    assert docs[1].page_content == "Second page"
    assert docs[1].metadata["page"] == 2


def test_chunk_documents_creates_expected_chunks():
    docs = [Document(page_content="ABCDEFGHIJ", metadata={"source": "x.pdf"})]
    chunks = main.chunk_documents(docs, chunk_size=4, chunk_overlap=1)

    assert [c.page_content for c in chunks] == ["ABCD", "DEFG", "GHIJ"]


def test_chunk_documents_rejects_invalid_overlap():
    with pytest.raises(ValueError, match="chunk_overlap must be smaller than chunk_size"):
        main.chunk_documents([], chunk_size=100, chunk_overlap=100)
