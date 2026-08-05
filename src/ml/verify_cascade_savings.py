"""Derive the stage-two inference savings from the held-out test artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path("data/models/two_stage_evaluation_15min.json"),
    )
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation.read_text())
    metrics = evaluation["stage1_metrics"]
    total = sum(
        metrics[name]
        for name in ("true_negatives", "false_positives", "false_negatives", "true_positives")
    )
    stage2_calls = metrics["false_positives"] + metrics["true_positives"]
    result = {
        "held_out_events": total,
        "full_model_calls_without_gate": total,
        "full_model_calls_with_gate": stage2_calls,
        "full_model_calls_avoided": total - stage2_calls,
        "full_model_call_reduction_pct": round(100 * (1 - stage2_calls / total), 4),
        "definition": "Reduction in expensive stage-two XGBoost invocations, not total CPU time.",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
