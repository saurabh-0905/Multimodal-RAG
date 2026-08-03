# Multimodal-RAG

A Retrieval-Augmented Generation system that lets you chat with your PDFs — grounded, source-based answers instead of LLM guesswork.

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="LangChain" src="https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white">
  <img alt="Pinecone" src="https://img.shields.io/badge/Pinecone-Vector%20DB-0A0F1C?logo=pinecone&logoColor=white">
  <img alt="Groq" src="https://img.shields.io/badge/Groq-LLM%20Inference-F55036">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

> **A note on the name:** this repo is named for where the project is headed, not just where it is today. Right now it's a text-only RAG pipeline — genuinely multimodal support (images, tables, audio) is an active roadmap item, detailed below. I'd rather be upfront about that than let the name overpromise.

---

## What it does

1. **Ingests** a PDF and splits it into overlapping chunks
2. **Embeds** each chunk locally (free, no API cost) and stores the vectors in Pinecone
3. **Retrieves** the most relevant chunks for a question using **MMR** (Maximal Marginal Relevance) — so results are relevant *and* non-redundant, not five near-duplicate chunks
4. **Generates** a grounded answer with Groq's LLM, instructed to answer only from retrieved context

## Why MMR instead of plain similarity search

Plain top-k similarity search often returns several chunks that all say nearly the same thing — especially in repetitive documents like policies or reports. MMR fixes this by picking chunks that are *both* relevant to the query *and* different from what's already been selected:

```
score = λ · relevance(chunk, query) − (1 − λ) · similarity(chunk, already_selected)
```

The result: a wider, more useful context window for the LLM instead of five variations on one point.

## Architecture

```
                 ┌──────────────┐
                 │   PDF file   │
                 └──────┬───────┘
                        │  loader.py
                        ▼
              ┌───────────────────┐
              │  Chunked text      │
              └─────────┬──────────┘
                        │  embeddings.py (local model)
                        ▼
              ┌───────────────────┐
              │  Pinecone index    │◄──── vector_store.py
              └─────────┬──────────┘
                        │  MMR retrieval
                        ▼
              ┌───────────────────┐
              │  Relevant chunks   │
              └─────────┬──────────┘
                        │  llm.py (Groq)
                        ▼
              ┌───────────────────┐
              │  Grounded answer   │
              └───────────────────┘
```

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Chunking | `langchain-text-splitters` | Recursive splitting with overlap, keeps context intact |
| Embeddings | `sentence-transformers` (local) | Free, runs offline, no per-chunk API cost |
| Vector store | Pinecone (serverless) | Managed, persistent, cloud-accessible — not tied to one machine |
| Retrieval | MMR (`langchain-core`) | Relevance *and* diversity, not just nearest-neighbor |
| Generation | Groq (`llama-3.3-70b-versatile`) | Fast inference, generous free tier |

## Project structure

```
config.py          all settings — models, chunk size, k values
loader.py           load a PDF, split it into chunks
embeddings.py        the local embedding model
vector_store.py     Pinecone connect / store / MMR retrieval
llm.py              Groq call + the grounding prompt
ingest.py           entry point — run once per PDF
chat.py             entry point — ask questions
```

Each file has one job — see [`vector_store.py`](./vector_store.py) if you want to see the retrieval logic specifically, or [`llm.py`](./llm.py) for the prompt that keeps answers grounded.

## Setup

```bash
git clone https://github.com/saurabh-0905/Multimodal-RAG.git
cd Multimodal-RAG

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
cp .env.example .env         # add your GROQ_API_KEY and PINECONE_API_KEY
```

**Get free API keys:**
- Groq → [console.groq.com/keys](https://console.groq.com/keys)
- Pinecone → [app.pinecone.io](https://app.pinecone.io) (free serverless tier is enough)

## Usage

```bash
# Ingest one or more PDFs (run once per file — adds to the same index)
python ingest.py yourfile.pdf
python ingest.py another.pdf

# Chat with everything you've ingested
python chat.py
```

```
Ask questions about your PDF (type 'exit' to quit)

You: What does the policy say about claim deadlines?
Bot: ...
```

## Roadmap: toward genuine multimodal RAG

The current pipeline is text-only. Making it truly multimodal means changing three layers, not just adding a feature flag:

- [ ] **Extraction** — pull images and table structure out of PDFs (e.g. `PyMuPDF`), not just raw text
- [ ] **Joint embeddings** — a CLIP-style shared vector space for text *and* images, so a text query can retrieve relevant images directly
- [ ] **Vision-capable generation** — pass retrieved images to a vision LLM (GPT-4o / Claude / Gemini) instead of a text-only model reasoning over a caption
- [ ] **Audio ingestion** — transcript-based embedding for voice/audio sources

Until these land, "multimodal" describes the destination, not the current state — tracked here rather than left unsaid.

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute.
