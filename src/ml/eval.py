"""
Simple Model Evaluation Script

Evaluates a single trained model on test data.

Usage:
    # Evaluate Stage 1 model
    python -m src.ml.eval --model data/models/stage1_model_15min.ubj --stage 1

    # Evaluate Stage 2 model
    python -m src.ml.eval --model data/models/stage2_model_15min.ubj --stage 2

    # Evaluate with custom threshold
    python -m src.ml.eval --model data/models/stage1_model_15min.ubj --stage 1 --threshold 0.3
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Simple evaluator for a single model."""

    def __init__(self, model_path: Path, stage: int, horizon: str = '15min',
                 data_dir: Path = None):
        """
        Initialize evaluator.

        Args:
            model_path: Path to model file (.ubj)
            stage: 1 (activity) or 2 (intensity)
            horizon: '15min' or '1h'
            data_dir: Directory with processed data
        """
        self.model_path = Path(model_path)
        self.stage = stage
        self.horizon = horizon
        self.data_dir = data_dir or Path('data/processed')

        self.model = None
        self.metadata = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

    def load_model(self):
        """Load the trained model."""
        logger.info(f"Loading model from: {self.model_path}")

        self.model = xgb.XGBClassifier()
        self.model.load_model(str(self.model_path))

        # Try to load metadata
        metadata_path = self.model_path.parent / self.model_path.name.replace('.ubj', '_metadata.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded metadata from: {metadata_path}")
        else:
            logger.warning(f"Metadata file not found: {metadata_path}")
            self.metadata = {}

    def load_test_data(self):
        """Load appropriate test data based on stage."""
        logger.info(f"\nLoading test data for Stage {self.stage}...")

        if self.stage == 1:
            # Stage 1: All cells
            X_test = pd.read_parquet(self.data_dir / 'stage1_features_test.parquet')
            y_test_all = pd.read_parquet(self.data_dir / 'stage1_labels_test.parquet')
            target_col = 'stage1_target_15min' if self.horizon == '15min' else 'stage1_target_1h'
        elif self.stage == 2:
            # Stage 2: Active cells only
            X_test = pd.read_parquet(self.data_dir / 'stage2_features_test.parquet')
            y_test_all = pd.read_parquet(self.data_dir / 'stage2_labels_test.parquet')
            target_col = 'stage2_target_15min' if self.horizon == '15min' else 'stage2_target_1h'
        else:
            raise ValueError(f"Invalid stage: {self.stage}. Must be 1 or 2.")

        y_test = y_test_all[target_col]

        # Remove identifier columns
        id_cols = ['h3_cell', 'timestamp']
        X_test_clean = X_test.drop(columns=[col for col in id_cols if col in X_test.columns])

        logger.info(f"  Test samples: {len(X_test_clean):,}")
        logger.info(f"  Features: {X_test_clean.shape[1]}")
        logger.info(f"  Positive rate: {100*y_test.mean():.2f}%")

        return X_test_clean, y_test

    def evaluate(self, X_test, y_test, threshold=0.5):
        """Evaluate model on test data."""
        logger.info(f"\n" + "="*80)
        logger.info(f"EVALUATING STAGE {self.stage} MODEL")
        logger.info("="*80)
        logger.info(f"Threshold: {threshold}")
        logger.info("-"*80)

        # Predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Calculate metrics
        metrics = {
            'accuracy': (y_pred == y_test).mean(),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'avg_precision': average_precision_score(y_test, y_pred_proba),
        }

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'threshold': threshold
        })

        # Print results
        logger.info(f"\nTest Set Metrics:")
        logger.info(f"  ROC-AUC:       {metrics['roc_auc']:.4f}")
        logger.info(f"  Avg Precision: {metrics['avg_precision']:.4f}")
        logger.info(f"  Precision:     {metrics['precision']:.4f}")
        logger.info(f"  Recall:        {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:      {metrics['f1_score']:.4f}")
        logger.info(f"  Accuracy:      {metrics['accuracy']:.4f}")

        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN: {tn:,}  |  FP: {fp:,}")
        logger.info(f"  FN: {fn:,}  |  TP: {tp:,}")

        # Calculate rates
        if tp + fn > 0:
            true_positive_rate = tp / (tp + fn)
        else:
            true_positive_rate = 0

        if tn + fp > 0:
            true_negative_rate = tn / (tn + fp)
        else:
            true_negative_rate = 0

        if tp + fp > 0:
            positive_predictive_value = tp / (tp + fp)
        else:
            positive_predictive_value = 0

        logger.info(f"\nAdditional Metrics:")
        logger.info(f"  True Positive Rate (Sensitivity): {true_positive_rate:.4f}")
        logger.info(f"  True Negative Rate (Specificity): {true_negative_rate:.4f}")
        logger.info(f"  Positive Predictive Value:        {positive_predictive_value:.4f}")

        logger.info("="*80 + "\n")

        return metrics

    def analyze_feature_importance(self, top_n=20):
        """Analyze and display feature importance."""
        logger.info("\n" + "="*80)
        logger.info(f"TOP {top_n} MOST IMPORTANT FEATURES")
        logger.info("="*80)

        # Get feature importance
        importance = self.model.feature_importances_
        feature_names = self.model.feature_names_in_

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        # Print top features
        logger.info("\n")
        for i, row in importance_df.head(top_n).iterrows():
            logger.info(f"  {row['feature']:40s} {row['importance']:.4f}")

        logger.info("="*80 + "\n")

        return importance_df

    def save_results(self, metrics, output_dir=None):
        """Save evaluation results."""
        if output_dir is None:
            output_dir = self.model_path.parent

        output_file = output_dir / f"eval_stage{self.stage}_{self.horizon}.json"

        results = {
            'evaluation_date': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'stage': self.stage,
            'horizon': self.horizon,
            'metrics': metrics,
            'metadata': self.metadata
        }

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to: {output_file}")

    def run_evaluation(self, threshold=0.5):
        """Run complete evaluation pipeline."""
        logger.info("\n" + "="*80)
        logger.info(f"MODEL EVALUATION - STAGE {self.stage}")
        logger.info("="*80)
        logger.info(f"Model: {self.model_path.name}")
        logger.info(f"Horizon: {self.horizon}")
        logger.info("="*80 + "\n")

        # Load model
        self.load_model()

        # Load data
        X_test, y_test = self.load_test_data()

        # Evaluate
        metrics = self.evaluate(X_test, y_test, threshold)

        # Feature importance
        importance_df = self.analyze_feature_importance()

        # Save results
        self.save_results(metrics)

        logger.info("EVALUATION COMPLETE!\n")

        return metrics, importance_df


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained lightning prediction model'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model file (.ubj)'
    )
    parser.add_argument(
        '--stage',
        type=int,
        required=True,
        choices=[1, 2],
        help='Model stage: 1 (activity) or 2 (intensity)'
    )
    parser.add_argument(
        '--horizon',
        type=str,
        default='15min',
        choices=['15min', '1h'],
        help='Prediction horizon (default: 15min)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold (default: 0.5)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Directory with processed data'
    )

    args = parser.parse_args()

    evaluator = ModelEvaluator(
        model_path=Path(args.model),
        stage=args.stage,
        horizon=args.horizon,
        data_dir=Path(args.data_dir) if args.data_dir else None
    )

    metrics, importance_df = evaluator.run_evaluation(threshold=args.threshold)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Model:     Stage {args.stage} ({'Activity Detection' if args.stage == 1 else 'Intensity Prediction'})")
    print(f"Horizon:   {args.horizon}")
    print(f"Threshold: {args.threshold}")
    print(f"\nPerformance:")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()