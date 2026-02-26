import fitz  # PyMuPDF
from pathlib import Path
from langchain_core.documents import Document


def read_pdfs_from_data_folder(data_dir: Path) -> list[Document]:
    docs: list[Document] = []

    # Sorting makes document ingestion deterministic over multiple runs
    pdf_files = sorted(data_dir.glob("*.pdf"), key=lambda p: p.name.lower())

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {data_dir.resolve()}")

    for pdf_file in pdf_files:
        with fitz.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text().strip()
                if not text:
                    continue
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(pdf_file),
                            "page": page_num,
                        },
                    )
                )
    return docs


def chunk_documents(
    docs: list[Document], chunk_size: int = 1200, chunk_overlap: int = 200
) -> list[Document]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[Document] = []
    step = chunk_size - chunk_overlap

    for doc in docs:
        text = doc.page_content
        for start in range(0, len(text), step):
            end = start + chunk_size
            chunk_text = text[start:end].strip()
            if not chunk_text:
                continue
            chunk_metadata = {**doc.metadata, "chunk_start": start, "chunk_end": end}
            chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))
            if end >= len(text):
                break
    return chunks


def format_context(docs: list[Document]) -> str:
    blocks: list[str] = []
    for i, doc in enumerate(docs, start=1):
        blocks.append(f"[{i}] {doc.page_content}")
    return "\n\n".join(blocks)