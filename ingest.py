"""Run this once per PDF to add it to the knowledge base.

Usage:
    python ingest.py yourfile.pdf
"""
import sys
from loader import load_and_split
from vector_store import add_chunks


def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_pdf>")
        return

    pdf_path = sys.argv[1]
    chunks = load_and_split(pdf_path)
    add_chunks(chunks)
    print(f"Ingested {len(chunks)} chunks from {pdf_path}")


if __name__ == "__main__":
    main()
