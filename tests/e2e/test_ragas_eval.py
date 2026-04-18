"""RAGAS evaluation gate. Runs on weekly schedule and manual trigger — not every PR.

See docs/evaluation.md for guidance on curating golden datasets and setting
empirical baselines before setting STRATUM_EVAL_WARN_ONLY=false.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from rag.config import get_settings
from rag.evaluation.ragas import RAGASEvaluator
from rag.pipeline import RAGPipeline, build_pipeline


@pytest.fixture(scope="session")
def rag_pipeline() -> RAGPipeline:
    """Build a real RAGPipeline for e2e evaluation.

    Skipped if STRATUM_ANTHROPIC_API_KEY is not set in the environment.
    """
    if not os.environ.get("STRATUM_ANTHROPIC_API_KEY"):
        pytest.skip("STRATUM_ANTHROPIC_API_KEY not set — skipping e2e eval")
    return build_pipeline(get_settings())


@pytest.mark.e2e
def test_ragas_thresholds(rag_pipeline: RAGPipeline) -> None:
    """RAGAS gate: fail (or warn) if any metric falls below its threshold."""
    settings = get_settings()

    thresholds_path = Path("config/eval_thresholds.yaml")
    if not thresholds_path.exists():
        pytest.skip(f"Eval thresholds file not found: {thresholds_path}")

    thresholds: dict[str, float] = yaml.safe_load(thresholds_path.read_text())

    evaluator = RAGASEvaluator(
        pipeline_fn=rag_pipeline.pipeline_fn,
        thresholds=thresholds,
        warn_only=settings.eval_warn_only,
    )

    golden_path = settings.eval_golden_path
    if not golden_path.exists():
        pytest.skip(f"Golden dataset not found: {golden_path}. See docs/evaluation.md.")

    result = evaluator.run(golden_path=golden_path)

    Path("reports").mkdir(exist_ok=True)
    evaluator.write_report(result, settings.eval_report_path)

    if not result.passed:
        if settings.eval_warn_only:
            import structlog  # noqa: PLC0415

            structlog.get_logger().warning(
                "ragas_threshold_warnings",
                failures=result.failures,
                scores=result.scores,
            )
        else:
            failure_lines = "\n".join(f"  · {f}" for f in result.failures)
            pytest.fail(
                f"RAGAS gate failed:\n{failure_lines}\n\nAll scores: {result.scores}"
            )
