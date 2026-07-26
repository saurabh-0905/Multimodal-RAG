# Multimodal-RAG

Chat with a PDF using local embeddings + Pinecone (MMR retrieval) + Groq.

## Project structure
```
config.py          all settings (models, chunk size, k values)
loader.py           step 1: load PDF, split into chunks
embeddings.py        step 2: local embedding model
vector_store.py     step 3-4: Pinecone connect, store, MMR retrieval
llm.py              step 5: Groq call + grounding prompt
ingest.py           entry point - run once per PDF
chat.py             entry point - ask questions
```

## Get free API keys
- Groq: https://console.groq.com/keys
- Pinecone: https://app.pinecone.io (free serverless tier works fine)
