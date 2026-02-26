from utils import read_pdfs_from_data_folder, chunk_documents
from config import DATA_DIR, INDEX_DIR, embeddings, llm, CHUNK_OVERLAP, CHUNK_SIZE, NO_ANSWER_FALLBACK


def main():
    docs = read_pdfs_from_data_folder(data_dir=DATA_DIR)
    chunks = chunk_documents(docs=docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(len(chunks))


if __name__ == "__main__":
    main()
