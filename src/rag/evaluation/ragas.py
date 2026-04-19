"""RAGAS evaluation harness. Designed to run on a scheduled basis, not on every PR.

RAGAS metrics (faithfulness, answer_relevancy) use an LLM as a judge internally.
This has two implications that must be accounted for in CI design:
  1. Each eval run costs money (~$0.01-0.05 per question depending on model)
  2. Scores vary +-0.03-0.05 between runs due to LLM non-determinism

Threshold values should be derived from empirical baselines, not set upfront.
Run eval >=3 times on a stable pipeline, average the scores, then set thresholds
at (average - 0.05) to absorb judge variance. Until baselines are established,
use warn_only=True to surface regressions without blocking CI.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from rag.exceptions import EvaluationError, ThresholdViolationError

logger = structlog.get_logger(__name__)


@dataclass
class EvalResult:
    """Result of a single RAGAS evaluation run."""

    scores: dict[str, float]
    failures: list[str] = field(default_factory=list)
    passed: bool = True
    report_path: Path | None = None


_DEFAULT_THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.75,
    "context_recall": 0.70,
}


class RAGASEvaluator:
    """Run RAGAS evaluation against a golden dataset and check metric thresholds.

    pipeline_fn: callable that accepts a question string and returns
                 {"answer": str, "contexts": list[str]}
    thresholds:  dict of metric -> minimum acceptable score
    warn_only:   if True, threshold violations are logged as warnings instead of
                 raising ThresholdViolationError
    """

    def __init__(
        self,
        pipeline_fn: Callable[[str], dict[str, Any]],
        thresholds: dict[str, float] | None = None,
        warn_only: bool = True,
    ) -> None:
        self._pipeline_fn = pipeline_fn
        self._thresholds = thresholds or _DEFAULT_THRESHOLDS
        self._warn_only = warn_only

    def run(self, golden_path: Path) -> EvalResult:
        """Execute the evaluation pipeline and return an EvalResult."""
        try:
            qa_pairs = self._load_golden(golden_path)
        except Exception as exc:
            raise EvaluationError(f"Failed to load golden dataset: {exc}") from exc

        logger.info("ragas_eval_start", num_questions=len(qa_pairs), golden=str(golden_path))

        rows = self._run_pipeline(qa_pairs)
        dataset = self._to_dataset(rows)
        scores = self._evaluate(dataset)
        failures = self._check_thresholds(scores)

        result = EvalResult(
            scores=scores,
            failures=failures,
            passed=len(failures) == 0,
        )
        logger.info("ragas_eval_complete", scores=scores, passed=result.passed)
        return result

    def write_report(self, result: EvalResult, output_path: Path) -> None:
        """Write evaluation results to a JSON report file."""
        try:
            import ragas  # noqa: PLC0415

            ragas_version = getattr(ragas, "__version__", "unknown")
        except ImportError:
            ragas_version = "unknown"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "scores": result.scores,
            "failures": result.failures,
            "passed": result.passed,
            "warn_only": self._warn_only,
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "ragas_version": ragas_version,
        }
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("ragas_report_written", path=str(output_path))

    @staticmethod
    def _load_golden(path: Path) -> list[dict[str, Any]]:
        """Read JSONL golden dataset. Each line: {question, ground_truth, expected_sources}."""
        pairs: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
        return pairs

    def _run_pipeline(self, qa_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Call pipeline_fn for each question and collect results."""
        rows: list[dict[str, Any]] = []
        for qa in qa_pairs:
            question = qa["question"]
            try:
                result = self._pipeline_fn(question)
                rows.append(
                    {
                        "question": question,
                        "answer": result.get("answer", ""),
                        "contexts": result.get("contexts", []),
                        "ground_truth": qa.get("ground_truth", ""),
                    }
                )
            except Exception as exc:
                logger.warning("pipeline_fn_failed", question=question[:60], error=str(exc))
                rows.append(
                    {
                        "question": question,
                        "answer": "",
                        "contexts": [],
                        "ground_truth": qa.get("ground_truth", ""),
                    }
                )
        return rows

    @staticmethod
    def _to_dataset(rows: list[dict[str, Any]]) -> Any:
        """Convert rows to a HuggingFace Dataset for RAGAS."""
        try:
            from datasets import Dataset  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("Install eval dependencies: pip install 'stratum[eval]'") from exc

        return Dataset.from_list(rows)

    @staticmethod
    def _evaluate(dataset: Any) -> dict[str, float]:
        """Run RAGAS evaluation and return per-metric scores."""
        try:
            from ragas import evaluate  # noqa: PLC0415
            from ragas.metrics import (  # noqa: PLC0415
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
        except ImportError as exc:
            raise ImportError("Install eval dependencies: pip install 'stratum[eval]'") from exc

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        return {k: float(v) for k, v in result.items()}

    def _check_thresholds(self, scores: dict[str, float]) -> list[str]:
        """Return a list of human-readable failure strings for metrics below threshold."""
        failures: list[str] = []
        for metric, threshold in self._thresholds.items():
            actual = scores.get(metric)
            if actual is None:
                continue
            if actual < threshold:
                err = ThresholdViolationError(metric=metric, actual=actual, required=threshold)
                if self._warn_only:
                    logger.warning("ragas_threshold_warning", detail=str(err))
                failures.append(str(err))
        return failures
