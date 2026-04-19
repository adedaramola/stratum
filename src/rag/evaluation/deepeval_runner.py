"""DeepEval evaluation harness for automated RAG quality gates.

Why DeepEval over RAGAS:
  - Pytest-native (assertions, not a separate runner)
  - Self-explaining metrics (failure includes diagnostic reasoning)
  - Pluggable judge — defaults to local Ollama (zero API cost), opt-in OpenAI
  - Active maintenance, stable API across minor versions

Why a local judge by default:
  LLM-as-judge metrics (faithfulness, answer_relevancy) call an LLM per question.
  At 200 golden questions x 4 metrics x every weekly run, hosted-model costs add
  up fast for a portfolio project. A local Ollama model (llama3.1:8b) gives ~85%
  agreement with GPT-4o-mini judging on standard RAG benchmarks at $0/run.
  Set STRATUM_EVAL_JUDGE_BACKEND=openai when you need higher-fidelity scoring.

Threshold strategy:
  Initial thresholds are starting points — not validated baselines. Run eval
  >=3 times on a stable pipeline, average the scores, then set thresholds at
  (average - 0.05) to absorb LLM judge variance. Until baselines exist, leave
  STRATUM_EVAL_WARN_ONLY=true so threshold misses log warnings without failing CI.
"""

from __future__ import annotations

import asyncio
import datetime
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from rag.config import Settings
from rag.exceptions import EvaluationError, ThresholdViolationError

logger = structlog.get_logger(__name__)

_DEFAULT_THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "contextual_precision": 0.75,
    "contextual_recall": 0.70,
}


@dataclass
class EvalResult:
    """Result of a single DeepEval evaluation run."""

    scores: dict[str, float]
    failures: list[str] = field(default_factory=list)
    passed: bool = True
    report_path: Path | None = None


# ---------------------------------------------------------------------------
# Ollama judge adapter
# ---------------------------------------------------------------------------


class OllamaJudge:
    """DeepEvalBaseLLM adapter for a locally running Ollama model.

    Satisfies the DeepEval LLM interface so any Ollama model can act as a
    judge — zero API cost, works offline after the model is pulled.
    """

    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url
        self._client: Any = None

    def load_model(self) -> Any:
        """Return (and cache) an ollama.Client instance."""
        if self._client is None:
            try:
                import ollama  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError("Install eval dependencies: pip install 'stratum[eval]'") from exc
            self._client = ollama.Client(host=self.base_url)
        return self._client

    def generate(self, prompt: str) -> str:
        """Synchronous generation — called by DeepEval metrics."""
        client = self.load_model()
        response = client.generate(model=self.model, prompt=prompt)
        return str(response.response)

    async def a_generate(self, prompt: str) -> str:
        """Async wrapper — DeepEval calls this in async contexts."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt)

    def get_model_name(self) -> str:
        """Return the model identifier string."""
        return self.model


# ---------------------------------------------------------------------------
# Judge factory
# ---------------------------------------------------------------------------


def build_judge(settings: Settings) -> OllamaJudge | Any:
    """Return the configured judge model instance.

    Returns OllamaJudge (zero cost) by default; GPTModel when
    STRATUM_EVAL_JUDGE_BACKEND=openai for higher-fidelity scoring.
    """
    if settings.eval_judge_backend == "ollama":
        return OllamaJudge(
            model=settings.eval_judge_model,
            base_url=settings.eval_ollama_base_url,
        )
    if settings.eval_judge_backend == "openai":
        try:
            from deepeval.models import GPTModel  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("Install eval dependencies: pip install 'stratum[eval]'") from exc
        if settings.openai_api_key is None:
            raise ValueError("STRATUM_OPENAI_API_KEY is required when eval_judge_backend=openai")
        return GPTModel(
            model=settings.eval_judge_openai_model,
            api_key=settings.openai_api_key.get_secret_value(),
        )
    raise ValueError(f"Unknown eval judge backend: {settings.eval_judge_backend}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class DeepEvalRunner:
    """Run DeepEval evaluation against a golden dataset and check metric thresholds.

    pipeline_fn: callable that accepts a question and returns
                 {"actual_output": str, "retrieval_context": list[str]}
    thresholds:  dict of metric name -> minimum acceptable score
    judge:       LLM judge instance (OllamaJudge or GPTModel)
    warn_only:   if True, threshold violations log warnings instead of raising
    """

    def __init__(
        self,
        pipeline_fn: Callable[[str], dict[str, Any]],
        thresholds: dict[str, float] | None = None,
        judge: Any = None,
        warn_only: bool = True,
    ) -> None:
        self._pipeline_fn = pipeline_fn
        self._thresholds = thresholds or _DEFAULT_THRESHOLDS
        self._judge = judge
        self._warn_only = warn_only

    def run(self, golden_path: Path) -> EvalResult:
        """Execute evaluation and return an EvalResult with per-metric mean scores."""
        try:
            qa_pairs = _load_golden(golden_path)
        except Exception as exc:
            raise EvaluationError(f"Failed to load golden dataset: {exc}") from exc

        logger.info(
            "deepeval_eval_start",
            num_questions=len(qa_pairs),
            golden=str(golden_path),
            judge=self._judge.get_model_name() if self._judge else "none",
        )

        test_cases = self._build_test_cases(qa_pairs)
        scores = self._evaluate(test_cases)
        failures = self._check_thresholds(scores)

        result = EvalResult(
            scores=scores,
            failures=failures,
            passed=len(failures) == 0,
        )
        logger.info("deepeval_eval_complete", scores=scores, passed=result.passed)
        return result

    def write_report(self, result: EvalResult, output_path: Path) -> None:
        """Write evaluation results to a JSON report file."""
        try:
            import deepeval  # noqa: PLC0415

            deepeval_version = getattr(deepeval, "__version__", "unknown")
        except ImportError:
            deepeval_version = "unknown"

        judge_backend = "none"
        judge_model = "none"
        if self._judge is not None:
            judge_model = self._judge.get_model_name()
            judge_backend = "ollama" if isinstance(self._judge, OllamaJudge) else "openai"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "scores": result.scores,
            "failures": result.failures,
            "passed": result.passed,
            "warn_only": self._warn_only,
            "judge_backend": judge_backend,
            "judge_model": judge_model,
            "deepeval_version": deepeval_version,
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("deepeval_report_written", path=str(output_path))

    def _build_test_cases(self, qa_pairs: list[dict[str, Any]]) -> list[Any]:
        """Call pipeline_fn for each Q&A pair and build LLMTestCase objects."""
        try:
            from deepeval.test_case import LLMTestCase  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("Install eval dependencies: pip install 'stratum[eval]'") from exc

        test_cases: list[Any] = []
        for qa in qa_pairs:
            question = qa["question"]
            try:
                output = self._pipeline_fn(question)
                actual_output = output.get("actual_output", "")
                retrieval_context = output.get("retrieval_context", [])
            except Exception as exc:
                logger.warning("pipeline_fn_failed", question=question[:60], error=str(exc))
                actual_output = ""
                retrieval_context = []

            test_cases.append(
                LLMTestCase(
                    input=question,
                    actual_output=actual_output,
                    retrieval_context=retrieval_context,
                    expected_output=qa.get("ground_truth", ""),
                )
            )
        return test_cases

    def _evaluate(self, test_cases: list[Any]) -> dict[str, float]:
        """Run all four metrics and return per-metric mean scores."""
        try:
            from deepeval.metrics import (  # noqa: PLC0415
                AnswerRelevancyMetric,
                ContextualPrecisionMetric,
                ContextualRecallMetric,
                FaithfulnessMetric,
            )
        except ImportError as exc:
            raise ImportError("Install eval dependencies: pip install 'stratum[eval]'") from exc

        metrics = [
            FaithfulnessMetric(
                threshold=self._thresholds.get("faithfulness", 0.85),
                model=self._judge,
                include_reason=True,
            ),
            AnswerRelevancyMetric(
                threshold=self._thresholds.get("answer_relevancy", 0.80),
                model=self._judge,
                include_reason=True,
            ),
            ContextualPrecisionMetric(
                threshold=self._thresholds.get("contextual_precision", 0.75),
                model=self._judge,
                include_reason=True,
            ),
            ContextualRecallMetric(
                threshold=self._thresholds.get("contextual_recall", 0.70),
                model=self._judge,
                include_reason=True,
            ),
        ]

        # Measure each metric on each test case and collect scores
        metric_scores: dict[str, list[float]] = {
            "faithfulness": [],
            "answer_relevancy": [],
            "contextual_precision": [],
            "contextual_recall": [],
        }
        metric_map = {
            "FaithfulnessMetric": "faithfulness",
            "AnswerRelevancyMetric": "answer_relevancy",
            "ContextualPrecisionMetric": "contextual_precision",
            "ContextualRecallMetric": "contextual_recall",
        }

        for test_case in test_cases:
            for metric in metrics:
                try:
                    metric.measure(test_case)
                    key = metric_map.get(type(metric).__name__, type(metric).__name__)
                    if metric.score is not None:
                        metric_scores[key].append(float(metric.score))
                except Exception as exc:
                    logger.warning(
                        "metric_measure_failed",
                        metric=type(metric).__name__,
                        error=str(exc),
                    )

        return {k: (sum(v) / len(v)) if v else 0.0 for k, v in metric_scores.items()}

    def _check_thresholds(self, scores: dict[str, float]) -> list[str]:
        """Return human-readable failure strings for metrics below threshold."""
        failures: list[str] = []
        for metric, threshold in self._thresholds.items():
            actual = scores.get(metric)
            if actual is None:
                continue
            if actual < threshold:
                err = ThresholdViolationError(metric=metric, actual=actual, required=threshold)
                if self._warn_only:
                    logger.warning("deepeval_threshold_warning", detail=str(err))
                failures.append(str(err))
        return failures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_golden(path: Path) -> list[dict[str, Any]]:
    """Read JSONL golden dataset. Each line: {question, ground_truth, expected_sources}."""
    pairs: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs
