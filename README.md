# PDF RAG QA App

A PDF Retrieval-Augmented Generation (LangChain + PyMuPDF + FAISS + OpenAI)  application that:

- Reads PDF files from `data/` using PyMuPDF
- Chunks document text
- Stores embeddings in a local FAISS vector index
- Retrieves relevant chunks for a user question
- Asks OpenAI to answer using only retrieved context
- Prints question, context, and answer to the console


## Project Structure

- `main.py`: Core app logic and CLI entrypoint
- `config.py`: Environment variable loading
- `utils.py`: Utility functions logic
- `.env`: API key configuration (gitignored)
- `.env.example`: Sample file for API key configuration
- `data/`: Input PDFs
- `faiss_index/`: Generated vector index (gitignored)


## Requirements

- Python `>=3.12`
- OpenAI API key with access to:
    - `text-embedding-3-small` (embeddings)
    - `gpt-3.5-turbo` (answer generation)


## Setup

1. Activate virtual environment (if available):
    ```powershell
    .venv\Scripts\activate
    ```

2. Install dependencies

    ```powershell
    uv sync
    ```
    or
    ```powershell
    pip install -r requirements.txt
    ```

3. Configure environment (create or update `.env`):

    ```env
    TEAMIFIED_OPENAI_API_KEY=<your_openai_api_key>
    ```

4. Put one or more `.pdf` files in `data/` (required)

## Run the App

```powershell
python main.py
```

You will be prompted to enter a question:

```text
Ask a question:
```

Output sections:

- `--- Question ---`
- `--- Retrieved Context ---` (retrieved context blocks)
- `--- Answer ---`


## Notes

- This project uses the OpenAI API as its LLM provider due to its ease of integration, strong documentation, and proven reliability in production environments. The selected models,`text-embedding-3-small` for embeddings and `gpt-3.5-turbo` for response generation, offer a practical balance of performance, scalability, and cost-effectiveness.
