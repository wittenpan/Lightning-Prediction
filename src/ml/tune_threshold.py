"""
Threshold Tuning for Two-Stage System

Finds optimal classification thresholds for both stages by maximizing F1 score
or other metrics on validation set.

Usage:
    python -m src.ml.tune_threshold --horizon 15min --metric f1
    python -m src.ml.tune_threshold --horizon 15min --metric precision --min-recall 0.5
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import json
import logging
import argparse
from sklearn.metrics import (
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ThresholdTuner:
    """Tune classification thresholds for two-stage system."""

    def __init__(self, horizon='15min', model_dir=None, data_dir=None):
        self.horizon = horizon
        self.model_dir = model_dir or Path('data/models')
        self.data_dir = data_dir or Path('data/processed')

        self.stage1_model = None
        self.stage2_model = None

    def load_models(self):
        """Load trained models."""
        logger.info("Loading models...")

        # Stage 1
        stage1_path = self.model_dir / f'stage1_model_{self.horizon}.ubj'
        self.stage1_model = xgb.XGBClassifier()
        self.stage1_model.load_model(str(stage1_path))
        logger.info(f"  Loaded Stage 1: {stage1_path.name}")

        # Stage 2
        stage2_path = self.model_dir / f'stage2_model_{self.horizon}.ubj'
        self.stage2_model = xgb.XGBClassifier()
        self.stage2_model.load_model(str(stage2_path))
        logger.info(f"  Loaded Stage 2: {stage2_path.name}")

    def load_validation_data(self):
        """Load validation data."""
        logger.info("\nLoading validation data...")

        # Stage 1 val data
        X1_val = pd.read_parquet(self.data_dir / 'stage1_features_val.parquet')
        y1_val = pd.read_parquet(self.data_dir / 'stage1_labels_val.parquet')['stage1_target_15min']
        X1_val_clean = X1_val.drop(columns=['h3_cell', 'timestamp'], errors='ignore')

        # Stage 2 val data
        X2_val = pd.read_parquet(self.data_dir / 'stage2_features_val.parquet')
        y2_val = pd.read_parquet(self.data_dir / 'stage2_labels_val.parquet')['stage2_target_15min']
        X2_val_clean = X2_val.drop(columns=['h3_cell', 'timestamp'], errors='ignore')

        # Full val data for ground truth
        X_full = pd.read_parquet(self.data_dir / 'features_val.parquet')
        y_full = pd.read_parquet(self.data_dir / 'labels_val.parquet')['target_binary']

        logger.info(f"  Stage 1: {len(X1_val_clean):,} samples")
        logger.info(f"  Stage 2: {len(X2_val_clean):,} samples")
        logger.info(f"  Full: {len(X_full):,} samples")

        return X1_val_clean, y1_val, X2_val_clean, y2_val, X_full, y_full

    def tune_stage1_threshold(self, X_val, y_val, metric='f1', min_recall=0.0):
        """
        Tune Stage 1 threshold.

        For Stage 1 (activity detection), we want HIGH RECALL (catch all activity)
        while maintaining reasonable precision.
        """
        logger.info("\n" + "="*80)
        logger.info("TUNING STAGE 1 THRESHOLD (Activity Detection)")
        logger.info("="*80)
        logger.info(f"Optimization metric: {metric}")
        logger.info(f"Minimum recall constraint: {min_recall:.2f}")

        # Get probabilities
        y_proba = self.stage1_model.predict_proba(X_val)[:, 1]

        # Try different thresholds
        thresholds = np.linspace(0.01, 0.99, 99)
        results = []

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)

            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)

            # Skip if below minimum recall
            if rec < min_recall:
                continue

            results.append({
                'threshold': thresh,
                'precision': prec,
                'recall': rec,
                'f1': f1
            })

        results_df = pd.DataFrame(results)

        if len(results_df) == 0:
            logger.error(f"No thresholds met min_recall={min_recall}")
            return 0.5

        # Find best threshold
        if metric == 'f1':
            best_idx = results_df['f1'].idxmax()
        elif metric == 'precision':
            best_idx = results_df['precision'].idxmax()
        elif metric == 'recall':
            best_idx = results_df['recall'].idxmax()
        else:
            raise ValueError(f"Unknown metric: {metric}")

        best = results_df.loc[best_idx]

        logger.info(f"\nBest threshold: {best['threshold']:.3f}")
        logger.info(f"  Precision: {best['precision']:.4f}")
        logger.info(f"  Recall:    {best['recall']:.4f}")
        logger.info(f"  F1:        {best['f1']:.4f}")

        logger.info("\nTop 5 thresholds:")
        top5 = results_df.nlargest(5, metric)
        for _, row in top5.iterrows():
            logger.info(f"  {row['threshold']:.3f}: P={row['precision']:.3f}, R={row['recall']:.3f}, F1={row['f1']:.3f}")

        return best['threshold'], results_df

    def tune_stage2_threshold(self, X_val, y_val, metric='f1', min_recall=0.0):
        """
        Tune Stage 2 threshold.

        For Stage 2 (intensity prediction), we want to balance precision and recall.
        """
        logger.info("\n" + "="*80)
        logger.info("TUNING STAGE 2 THRESHOLD (Intensity Prediction)")
        logger.info("="*80)
        logger.info(f"Optimization metric: {metric}")
        logger.info(f"Minimum recall constraint: {min_recall:.2f}")

        # Get probabilities
        y_proba = self.stage2_model.predict_proba(X_val)[:, 1]

        # Try different thresholds
        thresholds = np.linspace(0.01, 0.99, 99)
        results = []

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)

            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)

            # Skip if below minimum recall
            if rec < min_recall:
                continue

            results.append({
                'threshold': thresh,
                'precision': prec,
                'recall': rec,
                'f1': f1
            })

        results_df = pd.DataFrame(results)

        if len(results_df) == 0:
            logger.error(f"No thresholds met min_recall={min_recall}")
            return 0.5

        # Find best threshold
        if metric == 'f1':
            best_idx = results_df['f1'].idxmax()
        elif metric == 'precision':
            best_idx = results_df['precision'].idxmax()
        elif metric == 'recall':
            best_idx = results_df['recall'].idxmax()
        else:
            raise ValueError(f"Unknown metric: {metric}")

        best = results_df.loc[best_idx]

        logger.info(f"\nBest threshold: {best['threshold']:.3f}")
        logger.info(f"  Precision: {best['precision']:.4f}")
        logger.info(f"  Recall:    {best['recall']:.4f}")
        logger.info(f"  F1:        {best['f1']:.4f}")

        logger.info("\nTop 5 thresholds:")
        top5 = results_df.nlargest(5, metric)
        for _, row in top5.iterrows():
            logger.info(f"  {row['threshold']:.3f}: P={row['precision']:.3f}, R={row['recall']:.3f}, F1={row['f1']:.3f}")

        return best['threshold'], results_df

    def save_results(self, stage1_thresh, stage2_thresh):
        """Save tuned thresholds."""
        output_file = self.model_dir / f'tuned_thresholds_{self.horizon}.json'

        results = {
            'horizon': self.horizon,
            'stage1_threshold': float(stage1_thresh),
            'stage2_threshold': float(stage2_thresh),
            'tuning_date': pd.Timestamp.now().isoformat()
        }

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nSaved thresholds to: {output_file}")

    def run_tuning(self, metric='f1', stage1_min_recall=0.7, stage2_min_recall=0.5):
        """Run complete threshold tuning pipeline."""
        logger.info("\n" + "="*80)
        logger.info("THRESHOLD TUNING FOR TWO-STAGE SYSTEM")
        logger.info("="*80)
        logger.info(f"Optimization metric: {metric}")
        logger.info(f"Stage 1 min recall: {stage1_min_recall}")
        logger.info(f"Stage 2 min recall: {stage2_min_recall}")
        logger.info("="*80 + "\n")

        # Load models
        self.load_models()

        # Load data
        X1_val, y1_val, X2_val, y2_val, X_full, y_full = self.load_validation_data()

        # Tune Stage 1 (prioritize recall to catch activity)
        stage1_thresh, stage1_results = self.tune_stage1_threshold(
            X1_val, y1_val, metric=metric, min_recall=stage1_min_recall
        )

        # Tune Stage 2 (balance precision/recall)
        stage2_thresh, stage2_results = self.tune_stage2_threshold(
            X2_val, y2_val, metric=metric, min_recall=stage2_min_recall
        )

        # Save results
        self.save_results(stage1_thresh, stage2_thresh)

        # Summary
        logger.info("\n" + "="*80)
        logger.info("TUNING COMPLETE")
        logger.info("="*80)
        logger.info(f"\nOptimal Thresholds:")
        logger.info(f"  Stage 1 (Activity):  {stage1_thresh:.3f}")
        logger.info(f"  Stage 2 (Intensity): {stage2_thresh:.3f}")
        logger.info(f"\nRe-run evaluation with these thresholds:")
        logger.info(f"  python -m src.ml.eval_two_stage --horizon {self.horizon} \\")
        logger.info(f"    --stage1-threshold {stage1_thresh:.3f} \\")
        logger.info(f"    --stage2-threshold {stage2_thresh:.3f}")
        logger.info("="*80 + "\n")

        return stage1_thresh, stage2_thresh, stage1_results, stage2_results


def main():
    parser = argparse.ArgumentParser(
        description='Tune classification thresholds for two-stage system'
    )
    parser.add_argument(
        '--horizon',
        type=str,
        default='15min',
        choices=['15min', '1h'],
        help='Prediction horizon'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='f1',
        choices=['f1', 'precision', 'recall'],
        help='Metric to optimize (default: f1)'
    )
    parser.add_argument(
        '--stage1-min-recall',
        type=float,
        default=0.7,
        help='Minimum recall for Stage 1 (default: 0.7)'
    )
    parser.add_argument(
        '--stage2-min-recall',
        type=float,
        default=0.5,
        help='Minimum recall for Stage 2 (default: 0.5)'
    )

    args = parser.parse_args()

    tuner = ThresholdTuner(horizon=args.horizon)
    tuner.run_tuning(
        metric=args.metric,
        stage1_min_recall=args.stage1_min_recall,
        stage2_min_recall=args.stage2_min_recall
    )


if __name__ == "__main__":
    main()
