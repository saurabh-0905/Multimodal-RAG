"""Step 5: ask the LLM to answer using only the retrieved context."""
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from config import LLM_MODEL

llm = ChatGroq(model=LLM_MODEL)

# Keeping the LLM grounded stops it from making up answers not in the PDF
PROMPT = ChatPromptTemplate.from_template(
    "Answer only using the context below. If the answer isn't in the "
    "context, say you don't know.\n\nContext:\n{context}\n\nQuestion: {question}"
)


def generate_answer(context: str, question: str) -> str:
    response = llm.invoke(PROMPT.format(context=context, question=question))
    return response.content
