"""Step 2: the model that turns text into vectors.

Runs locally on your machine - free, no API calls, no per-chunk cost.
"""
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBED_MODEL

embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
