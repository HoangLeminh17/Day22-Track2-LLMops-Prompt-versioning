"""
Step 1 - LangSmith-instrumented RAG Pipeline
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable

from config import build_vectorstore, create_llm, format_docs, get_retriever, load_project_env
from qa_pairs import SAMPLE_QUESTIONS

load_project_env()

llm = create_llm()

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer using only the provided context. "
            "If the context does not contain the answer, say you do not have enough information.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def build_rag_chain(vectorstore):
    retriever = get_retriever(vectorstore)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    return chain.invoke(question)


def main():
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)

    vectorstore = build_vectorstore()
    chain, _retriever = build_rag_chain(vectorstore)

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        answer = ask(chain, question)
        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question[:60]}")
        print(f"       A: {answer[:100]}\n")

    print(f"Sent {len(SAMPLE_QUESTIONS)} traces to LangSmith project.")
    print("Open https://smith.langchain.com to view traces.")


if __name__ == "__main__":
    main()
