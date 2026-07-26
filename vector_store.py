"""Step 3 & 4: store chunk vectors in Pinecone, then retrieve them with MMR."""
import uuid
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from config import PINECONE_API_KEY, INDEX_NAME, EMBED_DIM, TOP_K, FETCH_K
from embeddings import embedding_model


def get_index():
    """Connect to the Pinecone index, creating it the first time only."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx["name"] for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(INDEX_NAME)


def add_chunks(chunks) -> None:
    """Embed each chunk and upload it to Pinecone, with its text as metadata."""
    index = get_index()
    vectors = [
        {
            "id": str(uuid.uuid4()),
            "values": embedding_model.embed_query(chunk.page_content),
            "metadata": {"text": chunk.page_content},
        }
        for chunk in chunks
    ]
    index.upsert(vectors=vectors)


def retrieve(question: str, k: int = TOP_K, fetch_k: int = FETCH_K) -> list[str]:
    """Fetch a wide pool of matches from Pinecone, then use MMR to keep
    the k that are relevant and non-redundant.
    """
    index = get_index()
    query_embedding = embedding_model.embed_query(question)

    results = index.query(
        vector=query_embedding, top_k=fetch_k, include_values=True, include_metadata=True
    )
    matches = results["matches"]
    if not matches:
        return []

    candidate_embeddings = [match["values"] for match in matches]
    picked = maximal_marginal_relevance(np.array(query_embedding), candidate_embeddings, k=k)
    return [matches[i]["metadata"]["text"] for i in picked]
