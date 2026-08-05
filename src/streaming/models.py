"""Two-stage XGBoost inference with an auditable stage-two skip rate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import xgboost as xgb


class TwoStageXGBoostCascade:
    """Run the expensive intensity model only when activity passes stage one."""

    def __init__(
        self,
        model_dir: str | Path = "data/models",
        horizon: str = "15min",
        stage1_threshold: Optional[float] = None,
        stage2_threshold: Optional[float] = None,
    ) -> None:
        model_path = Path(model_dir)
        stage1_metadata = json.loads(
            (model_path / f"stage1_metadata_{horizon}.json").read_text()
        )
        stage2_metadata = json.loads(
            (model_path / f"stage2_metadata_{horizon}.json").read_text()
        )
        thresholds = json.loads(
            (model_path / f"tuned_thresholds_{horizon}.json").read_text()
        )
        self.feature_names = stage1_metadata["feature_names"]
        if self.feature_names != stage2_metadata["feature_names"]:
            raise ValueError("stage-one and stage-two models use different feature schemas")

        self.stage1_threshold = float(
            thresholds["stage1_threshold"] if stage1_threshold is None else stage1_threshold
        )
        self.stage2_threshold = float(
            thresholds["stage2_threshold"] if stage2_threshold is None else stage2_threshold
        )
        # Booster avoids a scikit-learn runtime dependency in the serving image.
        self.stage1_model = xgb.Booster()
        self.stage1_model.load_model(model_path / f"stage1_model_{horizon}.ubj")
        self.stage2_model = xgb.Booster()
        self.stage2_model.load_model(model_path / f"stage2_model_{horizon}.ubj")
        self.total_predictions = 0
        self.stage2_invocations = 0

    @property
    def stage2_skip_rate(self) -> float:
        if not self.total_predictions:
            return 0.0
        return 1.0 - (self.stage2_invocations / self.total_predictions)

    def _row(self, features: Mapping[str, float]) -> np.ndarray:
        missing = [name for name in self.feature_names if name not in features]
        if missing:
            raise ValueError(f"missing model features: {', '.join(missing)}")
        return np.asarray(
            [[features[name] for name in self.feature_names]], dtype=np.float32
        )

    def predict(self, features: Mapping[str, float]) -> Dict[str, Any]:
        row = self._row(features)
        self.total_predictions += 1
        matrix = xgb.DMatrix(row, feature_names=self.feature_names)
        stage1_probability = float(self.stage1_model.predict(matrix)[0])
        stage1_prediction = int(stage1_probability >= self.stage1_threshold)

        if stage1_prediction:
            self.stage2_invocations += 1
            stage2_probability = float(self.stage2_model.predict(matrix)[0])
            stage2_prediction = int(stage2_probability >= self.stage2_threshold)
            stage2_executed = True
        else:
            stage2_probability = 0.0
            stage2_prediction = 0
            stage2_executed = False

        return {
            "stage1": {
                "prediction": stage1_prediction,
                "probability": stage1_probability,
                "threshold": self.stage1_threshold,
            },
            "stage2": {
                "prediction": stage2_prediction,
                "probability": stage2_probability,
                "threshold": self.stage2_threshold,
                "executed": stage2_executed,
            },
            "combined": {
                "prediction": int(stage1_prediction and stage2_prediction),
                "probability": stage2_probability if stage1_prediction else 0.0,
            },
        }
