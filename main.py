from utils import read_pdfs_from_data_folder
from config import DATA_DIR, INDEX_DIR, embeddings


def main():
    docs = read_pdfs_from_data_folder(data_dir=DATA_DIR)
    print(len(docs))


if __name__ == "__main__":
    main()
