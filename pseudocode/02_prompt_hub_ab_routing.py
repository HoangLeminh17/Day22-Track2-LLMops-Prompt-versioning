"""
Step 2 - Prompt Hub and A/B routing
"""

import hashlib
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

from config import (
    build_vectorstore,
    create_langsmith_client,
    create_llm,
    format_docs,
    get_prompt_names,
    get_prompt_variants,
    get_retriever,
    load_project_env,
)
from qa_pairs import SAMPLE_QUESTIONS

load_project_env()

PROMPT_V1_NAME, PROMPT_V2_NAME = get_prompt_names()
PROMPTS = get_prompt_variants()
PROMPT_V1 = PROMPTS["v1"]
PROMPT_V2 = PROMPTS["v2"]


def push_prompts_to_hub(client):
    for prompt_name, prompt_obj, description in [
        (PROMPT_V1_NAME, PROMPT_V1, "V1 concise answers"),
        (PROMPT_V2_NAME, PROMPT_V2, "V2 structured answers"),
    ]:
        try:
            url = client.push_prompt(prompt_name, object=prompt_obj, description=description)
            print(f"Pushed {prompt_name} -> {url}")
        except Exception as exc:
            print(f"Prompt push skipped for {prompt_name}: {exc}")


def pull_prompts_from_hub(client):
    prompts = {}
    fallbacks = {
        PROMPT_V1_NAME: PROMPT_V1,
        PROMPT_V2_NAME: PROMPT_V2,
    }
    for prompt_name in [PROMPT_V1_NAME, PROMPT_V2_NAME]:
        try:
            prompts[prompt_name] = client.pull_prompt(prompt_name)
            print(f"Pulled '{prompt_name}' from Prompt Hub")
        except Exception as exc:
            prompts[prompt_name] = fallbacks[prompt_name]
            print(f"Using local fallback for '{prompt_name}': {exc}")
    return prompts


def get_prompt_version(request_id: str) -> str:
    hash_int = int(hashlib.md5(request_id.encode("utf-8")).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    docs = retriever.invoke(question)
    context = format_docs(docs)
    answer = (prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )
    return {"question": question, "answer": answer, "version": version}


def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)

    client = create_langsmith_client()
    push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client)

    vectorstore = build_vectorstore()
    retriever = get_retriever(vectorstore)
    llm = create_llm()

    counts = {"v1": 0, "v2": 0}
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt = prompts[version_key]
        result = ask_ab(retriever, llm, prompt, question, version_tag)
        counts[version_tag] += 1
        print(f"[{i + 1:02d}] [prompt-{result['version']}] {question[:55]}...")

    print("\nRouting summary")
    print(f"  prompt-v1: {counts['v1']}")
    print(f"  prompt-v2: {counts['v2']}")
    print(f"  Prompt names: {PROMPT_V1_NAME}, {PROMPT_V2_NAME}")


if __name__ == "__main__":
    main()
