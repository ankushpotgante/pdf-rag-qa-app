import os
import dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# load variablees from .env file
dotenv.load_dotenv()


# define constants
DATA_DIR = Path("data")
INDEX_DIR = Path("faiss_index")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.0
NO_ANSWER_FALLBACK = "I don't know from the provided context."

# read API key from environment variables
OPENAI_API_KEY = os.getenv("TEAMIFIED_OPENAI_API_KEY")

# initialize LLM
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=CHAT_MODEL, temperature=TEMPERATURE)

# initialize embeddings model
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)