"""
Two-Stage System Evaluation

Evaluates the complete two-stage lightning prediction system:
- Stage 1: Activity Detection (≥1 strike)
- Stage 2: Intensity Prediction (≥5 strikes)

The two-stage system works by:
1. Stage 1 filters out inactive cells (reduces data by ~95%)
2. Stage 2 predicts high intensity in remaining active cells
3. Combined predictions have better precision and recall than single-stage

Usage:
    python -m src.ml.eval_two_stage --horizon 15min
    python -m src.ml.eval_two_stage --horizon 1h --threshold 0.3
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import json
import logging
import argparse
from datetime import datetime
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TwoStageEvaluator:
    """
    Evaluates the two-stage lightning prediction system.

    The system predicts high-intensity lightning (≥5 strikes) by:
    1. Stage 1: Filter out cells unlikely to have ANY activity (≥1 strike)
    2. Stage 2: Among active cells, predict high intensity (≥5 strikes)

    Combined prediction = Stage1(cell) AND Stage2(cell)
    """

    def __init__(self, horizon: str = '15min',
                 model_dir: Path = None,
                 data_dir: Path = None):
        """
        Initialize two-stage evaluator.

        Args:
            horizon: '15min' or '1h'
            model_dir: Directory with trained models
            data_dir: Directory with processed data
        """
        self.horizon = horizon
        self.model_dir = model_dir or Path('data/models')
        self.data_dir = data_dir or Path('data/processed')

        self.stage1_model = None
        self.stage2_model = None
        self.stage1_metadata = None
        self.stage2_metadata = None

        logger.info(f"Two-Stage Evaluator ({horizon})")
        logger.info(f"Model directory: {self.model_dir}")
        logger.info(f"Data directory: {self.data_dir}")

    def load_models(self):
        """Load both trained models."""
        logger.info("\n" + "="*80)
        logger.info("LOADING MODELS")
        logger.info("="*80)

        # Load Stage 1 model
        stage1_model_file = self.model_dir / f'stage1_model_{self.horizon}.ubj'
        stage1_metadata_file = self.model_dir / f'stage1_metadata_{self.horizon}.json'

        if not stage1_model_file.exists():
            raise FileNotFoundError(f"Stage 1 model not found: {stage1_model_file}")

        logger.info(f"\nLoading Stage 1 model...")
        self.stage1_model = xgb.XGBClassifier()
        self.stage1_model.load_model(stage1_model_file)

        with open(stage1_metadata_file, 'r') as f:
            self.stage1_metadata = json.load(f)

        logger.info(f"  Model: {stage1_model_file.name}")
        logger.info(f"  Training date: {self.stage1_metadata.get('training_date', 'N/A')}")
        logger.info(f"  Test ROC-AUC: {self.stage1_metadata.get('test_metrics', {}).get('roc_auc', 'N/A'):.4f}")

        # Load Stage 2 model
        stage2_model_file = self.model_dir / f'stage2_model_{self.horizon}.ubj'
        stage2_metadata_file = self.model_dir / f'stage2_metadata_{self.horizon}.json'

        if not stage2_model_file.exists():
            raise FileNotFoundError(f"Stage 2 model not found: {stage2_model_file}")

        logger.info(f"\nLoading Stage 2 model...")
        self.stage2_model = xgb.XGBClassifier()
        self.stage2_model.load_model(stage2_model_file)

        with open(stage2_metadata_file, 'r') as f:
            self.stage2_metadata = json.load(f)

        logger.info(f"  Model: {stage2_model_file.name}")
        logger.info(f"  Training date: {self.stage2_metadata.get('training_date', 'N/A')}")
        logger.info(f"  Test ROC-AUC: {self.stage2_metadata.get('test_metrics', {}).get('roc_auc', 'N/A'):.4f}")

        logger.info("="*80 + "\n")

    def load_test_data(self):
        """
        Load test data for both stages.

        Returns:
            Tuple of (stage1_X_test, stage1_y_test, stage2_X_test, stage2_y_test,
                     full_X_test, full_y_true)
        """
        logger.info("\n" + "="*80)
        logger.info("LOADING TEST DATA")
        logger.info("="*80)

        # Stage 1 data (all cells)
        logger.info("\nLoading Stage 1 test data (all cells)...")
        stage1_X = pd.read_parquet(self.data_dir / 'stage1_features_test.parquet')
        stage1_y_all = pd.read_parquet(self.data_dir / 'stage1_labels_test.parquet')
        stage1_y = stage1_y_all['stage1_target_15min' if self.horizon == '15min' else 'stage1_target_1h']

        # Keep h3_cell and timestamp for alignment before dropping
        stage1_identifiers = stage1_X[['h3_cell', 'timestamp']].copy()

        # Drop identifier columns for prediction
        id_cols = ['h3_cell', 'timestamp']
        stage1_X_clean = stage1_X.drop(columns=[col for col in id_cols if col in stage1_X.columns])

        logger.info(f"  Samples: {len(stage1_X):,}")
        logger.info(f"  Positive (≥1 strike): {stage1_y.sum():,} ({100*stage1_y.mean():.2f}%)")

        # Stage 2 data (active cells only)
        logger.info("\nLoading Stage 2 test data (active cells only)...")
        stage2_X = pd.read_parquet(self.data_dir / 'stage2_features_test.parquet')
        stage2_y_all = pd.read_parquet(self.data_dir / 'stage2_labels_test.parquet')
        stage2_y = stage2_y_all['stage2_target_15min' if self.horizon == '15min' else 'stage2_target_1h']

        # Keep identifiers
        stage2_identifiers = stage2_X[['h3_cell', 'timestamp']].copy()

        # Drop identifier columns
        stage2_X_clean = stage2_X.drop(columns=[col for col in id_cols if col in stage2_X.columns])

        logger.info(f"  Samples: {len(stage2_X):,}")
        logger.info(f"  Positive (≥5 strikes): {stage2_y.sum():,} ({100*stage2_y.mean():.2f}%)")

        # Load FULL test data to get true labels for ≥5 strikes
        logger.info("\nLoading full test data (for ground truth)...")
        full_X = pd.read_parquet(self.data_dir / 'features_test.parquet')
        full_y_all = pd.read_parquet(self.data_dir / 'labels_test.parquet')
        full_y_true = full_y_all['target_binary']  # Ground truth: ≥5 strikes

        # Keep identifiers for alignment
        full_identifiers = full_X[['h3_cell', 'timestamp']].copy()

        logger.info(f"  Total samples: {len(full_X):,}")
        logger.info(f"  True high intensity (≥5 strikes): {full_y_true.sum():,} ({100*full_y_true.mean():.2f}%)")

        logger.info("="*80 + "\n")

        return (stage1_X_clean, stage1_y, stage1_identifiers,
                stage2_X_clean, stage2_y, stage2_identifiers,
                full_identifiers, full_y_true)

    def evaluate_individual_stages(self, stage1_X, stage1_y, stage2_X, stage2_y,
                                   stage1_threshold=0.5, stage2_threshold=0.5):
        """Evaluate each stage independently."""
        logger.info("\n" + "="*80)
        logger.info("INDIVIDUAL STAGE PERFORMANCE")
        logger.info("="*80)

        # Stage 1 evaluation
        logger.info("\n" + "-"*80)
        logger.info("STAGE 1: Activity Detection (≥1 strike)")
        logger.info(f"Threshold: {stage1_threshold}")
        logger.info("-"*80)

        stage1_proba = self.stage1_model.predict_proba(stage1_X)[:, 1]
        stage1_pred = (stage1_proba >= stage1_threshold).astype(int)

        stage1_metrics = self._calculate_metrics(stage1_y, stage1_pred, stage1_proba, "Stage 1")

        # Stage 2 evaluation
        logger.info("\n" + "-"*80)
        logger.info("STAGE 2: Intensity Prediction (≥5 strikes, active cells)")
        logger.info(f"Threshold: {stage2_threshold}")
        logger.info("-"*80)

        stage2_proba = self.stage2_model.predict_proba(stage2_X)[:, 1]
        stage2_pred = (stage2_proba >= stage2_threshold).astype(int)

        stage2_metrics = self._calculate_metrics(stage2_y, stage2_pred, stage2_proba, "Stage 2")

        logger.info("="*80 + "\n")

        return stage1_metrics, stage2_metrics, stage1_proba, stage2_proba

    def evaluate_combined_system(self,
                                stage1_X, stage1_identifiers,
                                stage2_X, stage2_identifiers,
                                full_identifiers, full_y_true,
                                stage1_threshold=0.5, stage2_threshold=0.5):
        """
        Evaluate the combined two-stage system.

        Combined prediction:
        - Predict 0 (no high intensity) if Stage 1 predicts 0 (no activity)
        - Predict Stage 2 output if Stage 1 predicts 1 (activity detected)
        """
        logger.info("\n" + "="*80)
        logger.info("COMBINED TWO-STAGE SYSTEM")
        logger.info("="*80)
        logger.info(f"Stage 1 threshold: {stage1_threshold}")
        logger.info(f"Stage 2 threshold: {stage2_threshold}")
        logger.info("-"*80)

        # Stage 1: Predict on all cells
        logger.info("\nStep 1: Running Stage 1 on all cells...")
        stage1_proba = self.stage1_model.predict_proba(stage1_X)[:, 1]
        stage1_pred = (stage1_proba >= stage1_threshold).astype(int)

        stage1_positive_count = stage1_pred.sum()
        stage1_positive_pct = 100 * stage1_positive_count / len(stage1_pred)
        logger.info(f"  Stage 1 predicts activity: {stage1_positive_count:,} / {len(stage1_pred):,} ({stage1_positive_pct:.1f}%)")

        # Create DataFrame with Stage 1 predictions
        stage1_results = stage1_identifiers.copy()
        stage1_results['stage1_proba'] = stage1_proba
        stage1_results['stage1_pred'] = stage1_pred

        # Stage 2: Predict on active cells only
        logger.info("\nStep 2: Running Stage 2 on active cells...")
        stage2_proba = self.stage2_model.predict_proba(stage2_X)[:, 1]
        stage2_pred = (stage2_proba >= stage2_threshold).astype(int)

        # Create DataFrame with Stage 2 predictions
        stage2_results = stage2_identifiers.copy()
        stage2_results['stage2_proba'] = stage2_proba
        stage2_results['stage2_pred'] = stage2_pred

        # Merge predictions with ground truth
        logger.info("\nStep 3: Combining predictions...")
        full_results = full_identifiers.copy()
        full_results['y_true'] = full_y_true.values

        # Merge Stage 1 predictions
        full_results = full_results.merge(
            stage1_results,
            on=['h3_cell', 'timestamp'],
            how='left'
        )

        # Merge Stage 2 predictions
        full_results = full_results.merge(
            stage2_results,
            on=['h3_cell', 'timestamp'],
            how='left'
        )

        # Fill NaN values (cells not in Stage 2 data)
        full_results['stage2_proba'] = full_results['stage2_proba'].fillna(0)
        full_results['stage2_pred'] = full_results['stage2_pred'].fillna(0)

        # Combined prediction logic:
        # - If Stage 1 says "no activity" (0), then final prediction is 0
        # - If Stage 1 says "activity" (1), then use Stage 2 prediction
        full_results['combined_pred'] = (
            (full_results['stage1_pred'] == 1) &
            (full_results['stage2_pred'] == 1)
        ).astype(int)

        # Combined probability (multiplicative - both stages must agree)
        full_results['combined_proba'] = (
            full_results['stage1_proba'] * full_results['stage2_proba']
        )

        # Calculate combined metrics
        y_true = full_results['y_true'].values
        y_pred = full_results['combined_pred'].values
        y_proba = full_results['combined_proba'].values

        combined_metrics = self._calculate_metrics(y_true, y_pred, y_proba, "Combined System")

        # Show breakdown
        logger.info("\n" + "-"*80)
        logger.info("PREDICTION BREAKDOWN")
        logger.info("-"*80)

        stage1_only_positive = (
            (full_results['stage1_pred'] == 1) &
            (full_results['stage2_pred'] == 0)
        ).sum()
        both_positive = (
            (full_results['stage1_pred'] == 1) &
            (full_results['stage2_pred'] == 1)
        ).sum()
        both_negative = (
            (full_results['stage1_pred'] == 0) &
            (full_results['stage2_pred'] == 0)
        ).sum()

        logger.info(f"Both stages predict negative: {both_negative:,} ({100*both_negative/len(full_results):.1f}%)")
        logger.info(f"Stage 1 positive, Stage 2 negative: {stage1_only_positive:,} ({100*stage1_only_positive/len(full_results):.1f}%)")
        logger.info(f"Both stages predict positive: {both_positive:,} ({100*both_positive/len(full_results):.1f}%)")

        logger.info("="*80 + "\n")

        return combined_metrics, full_results

    def _calculate_metrics(self, y_true, y_pred, y_proba, name):
        """Calculate classification metrics."""
        metrics = {
            'accuracy': (y_pred == y_true).mean(),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'avg_precision': average_precision_score(y_true, y_proba),
        }

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score']
        })

        logger.info(f"\n{name} Metrics:")
        logger.info(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
        logger.info(f"  Avg Precision: {metrics['avg_precision']:.4f}")
        logger.info(f"  Precision:  {metrics['precision']:.4f}")
        logger.info(f"  Recall:     {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:   {metrics['f1_score']:.4f}")
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN: {tn:,}  |  FP: {fp:,}")
        logger.info(f"  FN: {fn:,}  |  TP: {tp:,}")

        return metrics

    def compare_with_baseline(self, full_identifiers, full_y_true):
        """
        Compare two-stage system with single-stage baseline.

        Loads the original single-stage model if available.
        """
        baseline_model_file = self.model_dir / 'lightning_model.ubj'

        if not baseline_model_file.exists():
            logger.info("\n⚠️  Single-stage baseline model not found, skipping comparison")
            return None

        logger.info("\n" + "="*80)
        logger.info("BASELINE COMPARISON: Single-Stage vs Two-Stage")
        logger.info("="*80)

        # Load baseline model
        logger.info("\nLoading single-stage baseline model...")
        baseline_model = xgb.XGBClassifier()
        baseline_model.load_model(baseline_model_file)

        # Load full features for baseline
        full_X = pd.read_parquet(self.data_dir / 'features_test.parquet')
        id_cols = ['h3_cell', 'timestamp']
        full_X_clean = full_X.drop(columns=[col for col in id_cols if col in full_X.columns])

        # Baseline predictions
        logger.info("Running baseline predictions...")
        baseline_proba = baseline_model.predict_proba(full_X_clean)[:, 1]
        baseline_pred = (baseline_proba >= 0.5).astype(int)

        baseline_metrics = self._calculate_metrics(
            full_y_true.values, baseline_pred, baseline_proba, "Single-Stage Baseline"
        )

        logger.info("="*80 + "\n")

        return baseline_metrics

    def save_results(self, stage1_metrics, stage2_metrics, combined_metrics,
                    baseline_metrics=None, output_dir=None):
        """Save evaluation results to JSON."""
        output_dir = output_dir or self.model_dir

        results = {
            'evaluation_date': datetime.now().isoformat(),
            'horizon': self.horizon,
            'stage1_metrics': stage1_metrics,
            'stage2_metrics': stage2_metrics,
            'combined_metrics': combined_metrics,
        }

        if baseline_metrics:
            results['baseline_metrics'] = baseline_metrics

        output_file = output_dir / f'two_stage_evaluation_{self.horizon}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to: {output_file}")

    def run_evaluation(self, stage1_threshold=0.5, stage2_threshold=0.5):
        """Run complete two-stage evaluation pipeline."""
        start_time = datetime.now()

        logger.info("\n" + "="*80)
        logger.info(f"TWO-STAGE SYSTEM EVALUATION ({self.horizon})")
        logger.info("="*80)
        logger.info(f"Started: {start_time}")
        logger.info("="*80 + "\n")

        # Load models
        self.load_models()

        # Load data
        (stage1_X, stage1_y, stage1_identifiers,
         stage2_X, stage2_y, stage2_identifiers,
         full_identifiers, full_y_true) = self.load_test_data()

        # Evaluate individual stages
        stage1_metrics, stage2_metrics, stage1_proba, stage2_proba = self.evaluate_individual_stages(
            stage1_X, stage1_y, stage2_X, stage2_y,
            stage1_threshold, stage2_threshold
        )

        # Evaluate combined system
        combined_metrics, full_results = self.evaluate_combined_system(
            stage1_X, stage1_identifiers,
            stage2_X, stage2_identifiers,
            full_identifiers, full_y_true,
            stage1_threshold, stage2_threshold
        )

        # Compare with baseline
        baseline_metrics = self.compare_with_baseline(full_identifiers, full_y_true)

        # Summary
        logger.info("\n" + "="*80)
        logger.info("FINAL SUMMARY")
        logger.info("="*80)

        logger.info(f"\nStage 1 (Activity Detection):")
        logger.info(f"  Precision: {stage1_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {stage1_metrics['recall']:.4f}")
        logger.info(f"  F1:        {stage1_metrics['f1_score']:.4f}")

        logger.info(f"\nStage 2 (Intensity Prediction):")
        logger.info(f"  Precision: {stage2_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {stage2_metrics['recall']:.4f}")
        logger.info(f"  F1:        {stage2_metrics['f1_score']:.4f}")

        logger.info(f"\nCombined Two-Stage System:")
        logger.info(f"  Precision: {combined_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {combined_metrics['recall']:.4f}")
        logger.info(f"  F1:        {combined_metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC:   {combined_metrics['roc_auc']:.4f}")

        if baseline_metrics:
            logger.info(f"\nSingle-Stage Baseline:")
            logger.info(f"  Precision: {baseline_metrics['precision']:.4f}")
            logger.info(f"  Recall:    {baseline_metrics['recall']:.4f}")
            logger.info(f"  F1:        {baseline_metrics['f1_score']:.4f}")

            # Calculate improvement
            prec_improvement = combined_metrics['precision'] - baseline_metrics['precision']
            recall_improvement = combined_metrics['recall'] - baseline_metrics['recall']
            f1_improvement = combined_metrics['f1_score'] - baseline_metrics['f1_score']

            logger.info(f"\nImprovement over baseline:")
            logger.info(f"  Precision: {prec_improvement:+.4f}")
            logger.info(f"  Recall:    {recall_improvement:+.4f}")
            logger.info(f"  F1:        {f1_improvement:+.4f}")

        # Save results
        self.save_results(stage1_metrics, stage2_metrics, combined_metrics, baseline_metrics)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info(f"\nDuration: {duration:.1f}s ({duration/60:.1f} min)")
        logger.info("="*80 + "\n")

        return {
            'stage1_metrics': stage1_metrics,
            'stage2_metrics': stage2_metrics,
            'combined_metrics': combined_metrics,
            'baseline_metrics': baseline_metrics,
            'full_results': full_results
        }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate two-stage lightning prediction system'
    )
    parser.add_argument(
        '--horizon',
        type=str,
        default='15min',
        choices=['15min', '1h'],
        help='Prediction horizon (default: 15min)'
    )
    parser.add_argument(
        '--stage1-threshold',
        type=float,
        default=0.5,
        help='Stage 1 classification threshold (default: 0.5)'
    )
    parser.add_argument(
        '--stage2-threshold',
        type=float,
        default=0.5,
        help='Stage 2 classification threshold (default: 0.5)'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        help='Directory with trained models'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Directory with processed data'
    )

    args = parser.parse_args()

    evaluator = TwoStageEvaluator(
        horizon=args.horizon,
        model_dir=Path(args.model_dir) if args.model_dir else None,
        data_dir=Path(args.data_dir) if args.data_dir else None
    )

    results = evaluator.run_evaluation(
        stage1_threshold=args.stage1_threshold,
        stage2_threshold=args.stage2_threshold
    )

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nTwo-stage system achieves:")
    print(f"  Precision: {results['combined_metrics']['precision']:.1%}")
    print(f"  Recall:    {results['combined_metrics']['recall']:.1%}")
    print(f"  F1 Score:  {results['combined_metrics']['f1_score']:.4f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
