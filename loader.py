"""Step 1: turn a PDF into small, overlapping text chunks."""
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_and_split(pdf_path: str):
    # Read the PDF - keeps page numbers as metadata automatically
    pages = PyPDFLoader(pdf_path).load()

    # Split into overlapping chunks so each piece is small enough for the
    # LLM and the embedding model to handle well
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(pages)
