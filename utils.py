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
