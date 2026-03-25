"""
evaluate.py
────────────
Docu_RAG RAG Evaluation using RAGAS.

Metrics:
  - Faithfulness        : Is the answer grounded in the retrieved context?
  - Answer Relevancy    : Does the answer address the question?
  - Context Precision   : Are the retrieved chunks actually relevant?
  - Context Recall      : Were all relevant chunks retrieved? (needs ground truth)

Usage:
  # Without ground truth (3 metrics):
  python evaluate.py --user user_123

  # With ground truth YAML (all 4 metrics):
  python evaluate.py --user user_123 --testset tests/eval_questions.yaml

  # Save report to CSV:
  python evaluate.py --user user_123 --output reports/eval_report.csv
"""
from __future__ import annotations

import argparse
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.config import get_settings
from src.rag_engine import MultiTenantRAGEngine

logger = logging.getLogger(__name__)
settings = get_settings()


# ──────────────────────────────────────────────────────────────────────────── #
#  Data structures                                                              #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass
class EvalQuestion:
    """A single evaluation question, optionally with ground truth."""
    question: str
    ground_truth: Optional[str] = None  # Required for context_recall


@dataclass
class EvalResult:
    """Raw result for one question before RAGAS scoring."""
    question: str
    answer: str
    contexts: list[str]
    ground_truth: Optional[str] = None
    source_docs: list[dict] = field(default_factory=list)  # for debugging


# ──────────────────────────────────────────────────────────────────────────── #
#  Evaluator LLM factory                                                        #
# ──────────────────────────────────────────────────────────────────────────── #

def _build_evaluator_llm():
    """
    Returns the LLM used as the RAGAS judge.
    Use the strongest available model — the evaluator quality
    directly affects score reliability.
    """
    # Prefer OpenAI for evaluation (more reliable as a judge)
    if settings.openai_api_key:
        logger.info("Using GPT-4o as RAGAS evaluator LLM")
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=settings.openai_api_key,
        )
    # Fallback to Groq — use the largest available model
    logger.info("Using Groq llama3-70b as RAGAS evaluator LLM")
    return ChatGroq(
        model_name="llama3-70b-8192",  # Stronger judge than gemma2-9b
        temperature=0,
        groq_api_key=settings.groq_api_key,
    )


def _build_evaluator_embeddings():
    """Returns embeddings for RAGAS answer relevancy scoring."""
    if settings.openai_api_key:
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.openai_api_key,
        )
    # RAGAS works without custom embeddings (uses its own defaults)
    return None


# ──────────────────────────────────────────────────────────────────────────── #
#  Core evaluator                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

class DocuMindEvaluator:
    """
    Runs RAGAS evaluation against a DocuMind RAG engine instance.

    Key improvements over the original:
      - Uses a dedicated session per question (no cross-contamination
        from conversational memory between eval questions)
      - Separates retrieval from generation so RAGAS gets raw context
      - Selects metrics dynamically based on whether ground truth is provided
      - Catches per-question failures gracefully (one bad answer doesn't
        abort the entire eval run)
      - Returns a structured EvalReport with per-metric summaries
    """

    def __init__(self, engine: Optional[MultiTenantRAGEngine] = None):
        self.engine = engine or MultiTenantRAGEngine()
        self.evaluator_llm = _build_evaluator_llm()
        self.evaluator_embeddings = _build_evaluator_embeddings()

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        user_id: str,
        questions: list[EvalQuestion],
        show_progress: bool = True,
    ) -> "EvalReport":
        """
        Run evaluation for a list of questions.

        Args:
            user_id:       Which user's document store to query.
            questions:     List of EvalQuestion objects.
            show_progress: Print progress to stdout.

        Returns:
            EvalReport with scores DataFrame and summary statistics.
        """
        if not questions:
            raise ValueError("No questions provided for evaluation.")

        has_ground_truth = any(q.ground_truth for q in questions)
        raw_results = self._collect_results(user_id, questions, show_progress)

        if not raw_results:
            raise RuntimeError("All questions failed during collection. Check your setup.")

        dataset = self._build_dataset(raw_results, has_ground_truth)
        metrics = self._select_metrics(has_ground_truth)

        if show_progress:
            print(f"\n⚖️  Running RAGAS scoring ({len(metrics)} metrics)…")

        # Build kwargs for evaluate()
        eval_kwargs: dict = {
            "dataset": dataset,
            "metrics": metrics,
            "raise_exceptions": False,
        }
        # Only pass llm/embeddings if they're set — avoids RAGAS version issues
        if self.evaluator_llm:
            eval_kwargs["llm"] = self.evaluator_llm
        if self.evaluator_embeddings:
            eval_kwargs["embeddings"] = self.evaluator_embeddings

        scores = evaluate(**eval_kwargs)
        df = scores.to_pandas()

        return EvalReport(
            scores_df=df,
            raw_results=raw_results,
            metrics_used=[m.name for m in metrics],
            user_id=user_id,
        )

    # ------------------------------------------------------------------ #
    #  Private                                                             #
    # ------------------------------------------------------------------ #

    def _collect_results(
        self,
        user_id: str,
        questions: list[EvalQuestion],
        show_progress: bool,
    ) -> list[EvalResult]:
        """
        Runs each question through the RAG pipeline individually.
        Each question gets its own session to prevent memory bleed.
        """
        results = []
        total = len(questions)

        for i, eq in enumerate(questions, 1):
            if show_progress:
                print(f"  [{i}/{total}] {eq.question[:70]}…")

            # Fresh session per question — critical for eval integrity
            # (memory from Q1 must not influence Q2's answer)
            session_id = f"eval_{uuid.uuid4().hex}"

            try:
                # Retrieve context directly (needed for RAGAS metrics)
                docs = self.engine.retriever_factory.retrieve(
                    user_id, eq.question
                )
                contexts = [doc.page_content for doc in docs]

                if not contexts:
                    logger.warning(
                        f"No context retrieved for: '{eq.question}'. "
                        "Skipping — this question will not appear in scores."
                    )
                    continue

                # Generate answer using the same pipeline
                response = self.engine.ask(
                    user_id=user_id,
                    session_id=session_id,
                    question=eq.question,
                )

                results.append(EvalResult(
                    question=eq.question,
                    answer=response["answer"],
                    contexts=contexts,
                    ground_truth=eq.ground_truth,
                    source_docs=response.get("sources", []),
                ))

            except Exception as e:
                logger.error(
                    f"Failed on question '{eq.question[:50]}…': {e}",
                    exc_info=True,
                )
                # Continue with remaining questions rather than aborting
                continue

        logger.info(
            f"Collected {len(results)}/{total} results successfully."
        )
        return results

    @staticmethod
    def _build_dataset(
        results: list[EvalResult], has_ground_truth: bool
    ) -> Dataset:
        """Converts raw results into a RAGAS-compatible HuggingFace Dataset."""
        rows = []
        for r in results:
            row: dict = {
                "question": r.question,
                "answer": r.answer,
                "contexts": r.contexts,
            }
            if has_ground_truth and r.ground_truth:
                row["ground_truth"] = r.ground_truth
            rows.append(row)
        return Dataset.from_list(rows)

    @staticmethod
    def _select_metrics(has_ground_truth: bool) -> list:
        """
        Selects RAGAS metrics based on available data.
        context_recall requires ground_truth — skip it if not provided.
        """
        base_metrics = [faithfulness, answer_relevancy, context_precision]
        if has_ground_truth:
            base_metrics.append(context_recall)
            logger.info("Ground truth provided — including context_recall metric.")
        else:
            logger.info(
                "No ground truth — skipping context_recall. "
                "Add 'ground_truth' to your questions for full evaluation."
            )
        return base_metrics


# ──────────────────────────────────────────────────────────────────────────── #
#  Report                                                                       #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass
class EvalReport:
    """Structured evaluation report."""
    scores_df: pd.DataFrame
    raw_results: list[EvalResult]
    metrics_used: list[str]
    user_id: str

    def summary(self) -> dict[str, float]:
        """Returns mean score per metric."""
        numeric_cols = self.scores_df.select_dtypes(include="number").columns
        return self.scores_df[numeric_cols].mean().round(4).to_dict()

    def print_summary(self) -> None:
        """Pretty-prints the evaluation summary to stdout."""
        summary = self.summary()
        print("\n" + "═" * 50)
        print("  📊 DocuMind RAG Evaluation Report")
        print(f"  User: {self.user_id}  |  Questions: {len(self.raw_results)}")
        print("═" * 50)
        for metric, score in summary.items():
            bar = "█" * int(score * 20)
            grade = _grade(score)
            print(f"  {metric:<25} {score:.3f}  {bar:<20} {grade}")
        print("═" * 50)
        print()

    def save(self, path: str) -> None:
        """Saves the full scores DataFrame to a CSV file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.scores_df.to_csv(out, index=False)
        print(f"  ✓ Report saved to: {out}")

    def failing_questions(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        Returns rows where any metric score falls below the threshold.
        Useful for identifying specific weak points.
        """
        numeric_cols = self.scores_df.select_dtypes(include="number").columns
        mask = (self.scores_df[numeric_cols] < threshold).any(axis=1)
        return self.scores_df[mask]


def _grade(score: float) -> str:
    if score >= 0.85: return "✅ Excellent"
    if score >= 0.70: return "🟡 Good"
    if score >= 0.50: return "🟠 Needs work"
    return "🔴 Poor"


# ──────────────────────────────────────────────────────────────────────────── #
#  Testset loader                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

def load_testset(path: str) -> list[EvalQuestion]:
    """
    Loads evaluation questions from a YAML file.

    Expected format:
      - question: "What is the refund policy?"
        ground_truth: "Refunds are processed within 5-7 business days."
      - question: "How do I reset my API key?"
        # ground_truth is optional
    """
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [
        EvalQuestion(
            question=item["question"],
            ground_truth=item.get("ground_truth"),
        )
        for item in data
    ]


# ──────────────────────────────────────────────────────────────────────────── #
#  CLI entry point                                                              #
# ──────────────────────────────────────────────────────────────────────────── #

DEFAULT_QUESTIONS = [
    EvalQuestion(
        question="What is DocuMind and what problem does it solve?",
        ground_truth=(
            "DocuMind is a RAG-powered document intelligence assistant "
            "that answers questions grounded in uploaded documents."
        ),
    ),
    EvalQuestion(
        question="What are the four stages of the retrieval pipeline?",
        ground_truth=(
            "The four stages are: query rewriting, hybrid search "
            "(semantic + BM25), FlashRank re-ranking, and LLM generation."
        ),
    ),
    EvalQuestion(
        question="What API endpoints does DocuMind expose?",
        ground_truth=(
            "DocuMind exposes POST /ingest, POST /query, GET /sources, "
            "DELETE /sources/{name}, and POST /session/clear."
        ),
    ),
    EvalQuestion(
        question="What is the average query latency?",
        ground_truth="Average query latency is 1.2 seconds using Groq inference.",
    ),
    EvalQuestion(
        question="What are the current limitations of DocuMind?",
        ground_truth=(
            "Limitations include no image support in PDFs, no streaming "
            "responses, and in-memory BM25 that doesn't scale beyond 10,000 docs."
        ),
    ),
]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="DocuMind RAG Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--user", "-u",
        default="eval_user",
        help="User ID whose document store to evaluate (default: eval_user)",
    )
    parser.add_argument(
        "--testset", "-t",
        default=None,
        help="Path to YAML testset file (uses built-in questions if omitted)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save CSV report (e.g. reports/eval.csv)",
    )
    parser.add_argument(
        "--failing-threshold",
        type=float,
        default=0.5,
        help="Score threshold below which a question is flagged as failing (default: 0.5)",
    )
    args = parser.parse_args()

    # Load questions
    if args.testset:
        print(f"📋 Loading testset from '{args.testset}'…")
        questions = load_testset(args.testset)
    else:
        print("📋 Using built-in default questions…")
        questions = DEFAULT_QUESTIONS

    print(f"   {len(questions)} question(s) loaded.\n")

    # Run evaluation
    evaluator = DocuMindEvaluator()
    report = evaluator.evaluate(
        user_id=args.user,
        questions=questions,
        show_progress=True,
    )

    # Print results
    report.print_summary()

    # Show full DataFrame
    print("📄 Full scores:\n")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(report.scores_df.to_string(index=False))

    # Flag weak questions
    failing = report.failing_questions(threshold=args.failing_threshold)
    if not failing.empty:
        print(f"\n⚠️  {len(failing)} question(s) scored below {args.failing_threshold}:")
        print(failing[["question"] + [c for c in failing.columns if c != "question"]].to_string(index=False))

    # Save report
    if args.output:
        report.save(args.output)
    else:
        # Always auto-save with timestamp
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report.save(f"reports/eval_{ts}.csv")


if __name__ == "__main__":
    main()