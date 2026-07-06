"""
End-to-end validation script.

Run this to prove the pipeline works without any UI:
    python test_pipeline.py

It ingests sample_data/sample_notes.txt, asks a batch of test
questions, and writes a full log (retrieved chunks, scores, timings,
and the generated answer for each question) to logs/validation_log.txt
plus a metrics summary to logs/system_metrics.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.pipeline import RAGPipeline

SAMPLE_DOC = str(Path(__file__).parent / "sample_data" / "sample_notes.txt")
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

TEST_QUESTIONS = [
    "What are the three broad categories of machine learning?",
    "What does the sigmoid function output in logistic regression?",
    "Why are CNNs effective for image data?",
    "What problem does the Transformer's self-attention mechanism solve compared to recurrence?",
    "What is Retrieval-Augmented Generation and why does it reduce hallucination?",
    "What is the capital of France?",  # deliberately NOT in the document -> tests grounding
]


def main():
    print("=" * 70)
    print("RAG PIPELINE - END TO END VALIDATION")
    print("=" * 70)

    pipeline = RAGPipeline()

    print("\n[1/2] Ingesting documents...")
    metrics = pipeline.ingest([SAMPLE_DOC])
    print(json.dumps(metrics, indent=2))

    print("\n[2/2] Running test questions...\n")
    log_lines = []
    log_lines.append("RAG PIPELINE VALIDATION LOG")
    log_lines.append("=" * 70)
    log_lines.append(f"System metrics: {json.dumps(metrics, indent=2)}")
    log_lines.append("=" * 70)

    for i, question in enumerate(TEST_QUESTIONS, 1):
        result = pipeline.ask(question)
        print(f"Q{i}: {question}")
        print(f"A{i}: {result['answer']}")
        print(f"     (retrieval: {result['retrieval_time_sec']}s | "
              f"generation: {result['generation_time_sec']}s)\n")

        log_lines.append(f"\nQuestion {i}: {question}")
        log_lines.append(f"Answer: {result['answer']}")
        log_lines.append(f"Retrieval time: {result['retrieval_time_sec']}s | "
                          f"Generation time: {result['generation_time_sec']}s")
        log_lines.append("Top retrieved chunks:")
        for rank, chunk in enumerate(result["retrieved_chunks"], 1):
            log_lines.append(
                f"  [{rank}] source={chunk['source']} chunk_index={chunk['chunk_index']} "
                f"final_score={chunk['final_score']:.4f} "
                f"(vector={chunk['vector_score']:.4f}, keyword={chunk['keyword_score']:.4f})"
            )
            log_lines.append(f"      text preview: {chunk['text'][:150]}...")
        log_lines.append("-" * 70)

    log_path = LOG_DIR / "validation_log.txt"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    metrics_path = LOG_DIR / "system_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("=" * 70)
    print(f"Validation log written to: {log_path}")
    print(f"System metrics written to: {metrics_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
