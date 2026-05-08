import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
ENV_PATH = PROJECT_ROOT / ".env"
KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base.txt"

DEFAULT_CHAT_MODEL = "gpt-5.4-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_RETRIEVER_K = 3


def load_project_env() -> None:
    load_dotenv(ENV_PATH)
    os.environ["LANGCHAIN_TRACING_V2"] = get_env("LANGSMITH_TRACING", "true")
    os.environ["LANGCHAIN_API_KEY"] = get_env("LANGSMITH_API_KEY", required=True)
    os.environ["LANGCHAIN_PROJECT"] = get_env("LANGSMITH_PROJECT", "day22-rag")
    os.environ["LANGCHAIN_ENDPOINT"] = get_env(
        "LANGSMITH_ENDPOINT",
        DEFAULT_LANGSMITH_ENDPOINT,
    )


def get_env(name: str, default: str | None = None, *, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and (value is None or not str(value).strip()):
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or "day22-rag"


def get_prompt_names() -> tuple[str, str]:
    project_slug = slugify(get_env("LANGSMITH_PROJECT", "day22-rag"))
    return (f"{project_slug}-rag-prompt-v1", f"{project_slug}-rag-prompt-v2")


def get_prompt_variants():
    system_v1 = (
        "You are a helpful AI assistant. "
        "Answer the user's question using only the provided context. "
        "Keep the answer concise in 2 to 4 sentences. "
        "If the context is insufficient, say: 'I don't have enough information.'\n\n"
        "Context:\n{context}"
    )
    system_v2 = (
        "You are an expert AI tutor. Use only the provided context.\n"
        "Write a structured answer in 3 to 5 sentences.\n"
        "Start with the direct answer, then add one supporting detail.\n"
        "If the context is insufficient, say that explicitly.\n\n"
        "Context:\n{context}"
    )
    prompt_v1 = ChatPromptTemplate.from_messages(
        [("system", system_v1), ("human", "{question}")]
    )
    prompt_v2 = ChatPromptTemplate.from_messages(
        [("system", system_v2), ("human", "{question}")]
    )
    return {"v1": prompt_v1, "v2": prompt_v2}


def create_llm(*, temperature: float = 0.0, model: str | None = None) -> ChatOpenAI:
    load_project_env()
    return ChatOpenAI(
        model=model or get_env("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL),
        api_key=get_env("OPENAI_API_KEY", required=True),
        base_url=get_env("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
        temperature=temperature,
    )


def create_embeddings(model: str | None = None) -> OpenAIEmbeddings:
    load_project_env()
    return OpenAIEmbeddings(
        model=model or get_env("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        api_key=get_env("OPENAI_API_KEY", required=True),
        base_url=get_env("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
    )


def create_langsmith_client() -> Client:
    load_project_env()
    return Client(
        api_key=get_env("LANGSMITH_API_KEY", required=True),
        api_url=get_env("LANGSMITH_ENDPOINT", DEFAULT_LANGSMITH_ENDPOINT),
    )


def read_knowledge_base() -> str:
    if not KNOWLEDGE_BASE_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {KNOWLEDGE_BASE_PATH}. "
            "Expected data/knowledge_base.txt."
        )
    return KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8")


def build_vectorstore() -> FAISS:
    text = read_knowledge_base()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(text)
    print(f"Loaded dataset: {KNOWLEDGE_BASE_PATH}")
    print(f"Split into {len(chunks)} chunks")
    return FAISS.from_texts(chunks, create_embeddings())


def get_retriever(vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": DEFAULT_RETRIEVER_K})


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)
