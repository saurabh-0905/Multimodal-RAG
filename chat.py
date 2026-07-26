"""Run this to ask questions about whatever you've ingested.

Usage:
    python chat.py
"""
from vector_store import retrieve
from llm import generate_answer


def main():
    print("Ask questions about your PDF (type 'exit' to quit)")

    while (question := input("\nYou: ")).lower() != "exit":
        chunks = retrieve(question)         # relevant, diverse chunks (MMR)
        context = "\n\n".join(chunks)       # merge into one block
        answer = generate_answer(context, question)
        print("Bot:", answer)


if __name__ == "__main__":
    main()
