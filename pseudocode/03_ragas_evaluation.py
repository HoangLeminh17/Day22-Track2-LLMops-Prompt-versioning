"""
Step 3 - RAGAS Evaluation
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from config import (
    build_vectorstore,
    create_embeddings,
    create_llm,
    format_docs,
    get_prompt_variants,
    get_retriever,
)
from qa_pairs import QA_PAIRS

PROMPTS = get_prompt_variants()

REPORT_PATH = PROJECT_ROOT / "data" / "ragas_report.json"


@traceable(name="ragas-rag-query", tags=["ragas", "step3"])
def run_rag(retriever, llm, prompt, question: str) -> dict:
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    answer = (prompt | llm | StrOutputParser()).invoke(
        {"context": format_docs(docs), "question": question}
    )
    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list[dict]:
    retriever = get_retriever(vectorstore)
    llm = create_llm()
    prompt = PROMPTS[prompt_version]

    results = []
    print(f"\nRunning 50 questions with prompt {prompt_version} ...")
    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])
        results.append(
            {
                "question": qa["question"],
                "reference": qa["reference"],
                "answer": out["answer"],
                "contexts": out["contexts"],
            }
        )
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    return results


def build_ragas_dataset(rag_results: list[dict]) -> EvaluationDataset:
    samples = [
        SingleTurnSample(
            user_input=item["question"],
            response=item["answer"],
            retrieved_contexts=item["contexts"],
            reference=item["reference"],
        )
        for item in rag_results
    ]
    return EvaluationDataset(samples=samples)


def _extract_metric_values(result, key: str):
    try:
        values = result[key]
    except Exception:
        values = getattr(result, key)
    return [value for value in values if value is not None]


def run_ragas_eval(rag_results: list[dict], version: str) -> dict:
    print(f"\nRunning RAGAS evaluation for prompt {version} ...")
    dataset = build_ragas_dataset(rag_results)
    llm_eval = create_llm()
    emb_eval = create_embeddings()

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
    )

    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw_values = _extract_metric_values(result, key)
        scores[key] = float(np.mean(raw_values)) if raw_values else 0.0

    for metric_name, score in scores.items():
        star = " *" if metric_name == "faithfulness" and score >= 0.8 else ""
        print(f"  {metric_name:30s}: {score:.4f}{star}")
    return scores


def main():
    print("=" * 60)
    print("  Step 3: RAGAS Evaluation")
    print("=" * 60)

    vectorstore = build_vectorstore()
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    print("\nComparison")
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1 = v1_scores[metric]
        s2 = v2_scores[metric]
        winner = "V1" if s1 > s2 else "V2"
        print(f"  {metric:30s}: V1={s1:.4f}  V2={s2:.4f}  <- {winner}")

    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\nTarget met: faithfulness = {best_faith:.4f}")
    else:
        print(f"\nBelow target: faithfulness = {best_faith:.4f}")

    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {REPORT_PATH}")


if __name__ == "__main__":
    main()
