"""All settings live here, so nothing is hardcoded in multiple files."""
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

INDEX_NAME = "mini-rag"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384  # output size of the embedding model above
LLM_MODEL = "llama-3.3-70b-versatile"

CHUNK_SIZE = 1000     # characters per chunk
CHUNK_OVERLAP = 200   # overlap so ideas aren't cut in half between chunks

TOP_K = 4      # final chunks handed to the LLM
FETCH_K = 10   # wider pool MMR picks the top_k from
