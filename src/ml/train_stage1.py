"""
Stage 1 Model Training: Activity Detection

Predicts if a grid cell will have ANY lightning strikes (≥1) in the next time window.
This is the first stage of our two-stage prediction system.

Usage:
    python -m src.ml.train_stage1 --horizon 15  # 15-30 min prediction
    python -m src.ml.train_stage1 --horizon 60  # 1 hour prediction
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
    average_precision_score
)
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Stage1Trainer:
    """
    Stage 1: Activity Detection Model
    
    Predicts: Will this cell have ≥1 strike in next window?
    
    This is an easier problem than predicting high intensity,
    with better class balance (~10-20:1 instead of 873:1).
    """
    
    def __init__(self, horizon: str = '15min', output_dir: Path = None):
        """
        Initialize Stage 1 trainer.
        
        Args:
            horizon: '15min' or '1h'
            output_dir: Directory for saving models
        """
        self.horizon = horizon
        self.output_dir = output_dir or Path('data/models')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.feature_names = None
        self.metadata = {}
        
        # Target column based on horizon
        self.target_col = 'stage1_target_15min' if horizon == '15min' else 'stage1_target_1h'
        
    def load_data(self, data_dir: Path = None) -> tuple:
        """Load training data for Stage 1 (all cells)."""
        data_dir = data_dir or Path('data/processed')

        logger.info("="*80)
        logger.info(f"STAGE 1: ACTIVITY DETECTION ({self.horizon})")
        logger.info("="*80)
        logger.info(f"Target: ≥1 strike in next {self.horizon}")
        logger.info(f"Data directory: {data_dir}\n")

        # Load Stage 1 datasets (all cells, activity detection)
        logger.info("Loading training data...")
        X_train = pd.read_parquet(data_dir / 'stage1_features_train.parquet')
        y_train_all = pd.read_parquet(data_dir / 'stage1_labels_train.parquet')
        y_train = y_train_all[self.target_col]

        logger.info("Loading validation data...")
        X_val = pd.read_parquet(data_dir / 'stage1_features_val.parquet')
        y_val_all = pd.read_parquet(data_dir / 'stage1_labels_val.parquet')
        y_val = y_val_all[self.target_col]

        logger.info("Loading test data...")
        X_test = pd.read_parquet(data_dir / 'stage1_features_test.parquet')
        y_test_all = pd.read_parquet(data_dir / 'stage1_labels_test.parquet')
        y_test = y_test_all[self.target_col]
        
        # Remove identifier columns
        id_cols = ['h3_cell', 'timestamp']
        for col in id_cols:
            if col in X_train.columns:
                X_train = X_train.drop(columns=[col])
                X_val = X_val.drop(columns=[col])
                X_test = X_test.drop(columns=[col])
        
        self.feature_names = list(X_train.columns)
        
        logger.info(f"\nDataset sizes:")
        logger.info(f"  Train: {len(X_train):,} samples")
        logger.info(f"  Val:   {len(X_val):,} samples")
        logger.info(f"  Test:  {len(X_test):,} samples")
        logger.info(f"  Features: {len(self.feature_names)}")
        
        # Class distribution
        logger.info(f"\nClass distribution:")
        for name, y in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
            pos = y.sum()
            neg = len(y) - pos
            ratio = neg / pos if pos > 0 else 0
            logger.info(f"  {name}: {pos:,} positive ({100*y.mean():.2f}%), ratio {ratio:.1f}:1")
        
        logger.info("="*80 + "\n")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train Stage 1 model."""
        logger.info("="*80)
        logger.info("TRAINING STAGE 1 MODEL")
        logger.info("="*80)
        
        # Calculate class weights
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        auto_weight = neg_count / pos_count

        # Use actual class weight (don't cap - Stage 1 needs high recall!)
        scale_pos_weight = min(auto_weight, 60.0)  # Increase cap to 60
        
        logger.info(f"\nClass imbalance:")
        logger.info(f"  Negative: {neg_count:,}")
        logger.info(f"  Positive: {pos_count:,}")
        logger.info(f"  Auto ratio: {auto_weight:.1f}:1")
        logger.info(f"  Using scale_pos_weight: {scale_pos_weight:.1f}")
        
        params = {
            # Tree structure (optimized for HIGH RECALL - Stage 1 must catch everything!)
            'max_depth': 7,  # Deeper trees for better recall
            'min_child_weight': 1,  # Lower = more aggressive = better recall
            'max_delta_step': 1,  # Helps with extreme imbalance

            # Learning (careful but thorough)
            'learning_rate': 0.03,  # Lower for stability
            'n_estimators': 300,  # More trees

            # Light regularization (don't over-constrain)
            'gamma': 0.05,  # Lower = less conservative = better recall
            'reg_alpha': 0.01,  # Minimal L1
            'reg_lambda': 0.5,  # Light L2

            # Sampling
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,

            # Class imbalance (CRITICAL for Stage 1!)
            'scale_pos_weight': scale_pos_weight,

            # Objective
            'objective': 'binary:logistic',
            'eval_metric': ['aucpr', 'auc'],  # Prioritize PR-AUC

            # Performance
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 25  # Extra patience
        }
        
        logger.info("\nModel parameters:")
        for key, value in params.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("\nTraining...")
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=20)
        
        logger.info(f"\nTraining complete!")
        logger.info(f"  Best iteration: {self.model.best_iteration}")
        logger.info(f"  Best validation AUC: {self.model.best_score:.4f}")
        
        self.metadata['model_params'] = params
        self.metadata['best_iteration'] = int(self.model.best_iteration)
        self.metadata['best_score'] = float(self.model.best_score)
        
        logger.info("="*80 + "\n")
        
        return self.model
    
    def evaluate(self, X, y, dataset_name="Test"):
        """Evaluate Stage 1 model."""
        logger.info(f"Evaluating on {dataset_name} set...")
        
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = {
            'accuracy': (y_pred == y).mean(),
            'roc_auc': roc_auc_score(y, y_pred_proba),
            'avg_precision': average_precision_score(y, y_pred_proba),
        }
        
        report = classification_report(y, y_pred, output_dict=True)
        cm = confusion_matrix(y, y_pred)
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
        
        logger.info(f"\n{dataset_name} Set Metrics:")
        logger.info(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
        logger.info(f"  Precision:  {metrics['precision']:.4f}")
        logger.info(f"  Recall:     {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:   {metrics['f1_score']:.4f}")
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN: {tn:,}  |  FP: {fp:,}")
        logger.info(f"  FN: {fn:,}  |  TP: {tp:,}")
        
        return metrics
    
    def save_model(self):
        """Save Stage 1 model."""
        logger.info("="*80)
        logger.info("SAVING STAGE 1 MODEL")
        logger.info("="*80)
        
        model_file = self.output_dir / f'stage1_model_{self.horizon}.ubj'
        self.model.save_model(model_file)
        logger.info(f"  Saved model to: {model_file}")
        
        self.metadata['stage'] = 1
        self.metadata['horizon'] = self.horizon
        self.metadata['target'] = 'activity_detection'
        self.metadata['threshold'] = 1  # ≥1 strike
        self.metadata['training_date'] = datetime.now().isoformat()
        self.metadata['feature_count'] = len(self.feature_names)
        self.metadata['feature_names'] = self.feature_names
        
        metadata_file = self.output_dir / f'stage1_metadata_{self.horizon}.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        logger.info(f"  Saved metadata to: {metadata_file}")
        
        logger.info("="*80 + "\n")
    
    def run_pipeline(self, data_dir=None):
        """Run complete Stage 1 training pipeline."""
        start_time = datetime.now()
        
        logger.info("\n" + "="*80)
        logger.info(f"STAGE 1 TRAINING: ACTIVITY DETECTION ({self.horizon})")
        logger.info("="*80)
        logger.info(f"Started: {start_time}")
        logger.info("="*80 + "\n")
        
        # Load data
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_data(data_dir)
        
        # Train
        self.train(X_train, y_train, X_val, y_val)
        
        # Evaluate
        logger.info("\n" + "="*80)
        logger.info("EVALUATION")
        logger.info("="*80)
        
        val_metrics = self.evaluate(X_val, y_val, "Validation")
        test_metrics = self.evaluate(X_test, y_test, "Test")
        
        self.metadata['validation_metrics'] = val_metrics
        self.metadata['test_metrics'] = test_metrics
        
        logger.info("="*80 + "\n")
        
        # Save
        self.save_model()
        
        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*80)
        logger.info("STAGE 1 TRAINING COMPLETE!")
        logger.info("="*80)
        logger.info(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")
        logger.info(f"\nTest Performance:")
        logger.info(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {test_metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {test_metrics['f1_score']:.4f}")
        logger.info(f"\nModel saved to: {self.output_dir}")
        logger.info("="*80 + "\n")
        
        return self.model, test_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Train Stage 1 model (Activity Detection)'
    )
    parser.add_argument(
        '--horizon',
        type=str,
        default='15min',
        choices=['15min', '1h'],
        help='Prediction horizon (default: 15min)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Directory with processed data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for model'
    )
    
    args = parser.parse_args()
    
    trainer = Stage1Trainer(
        horizon=args.horizon,
        output_dir=Path(args.output_dir) if args.output_dir else None
    )
    
    model, metrics = trainer.run_pipeline(
        data_dir=Path(args.data_dir) if args.data_dir else None
    )
    
    print("\n" + "="*80)
    print("NEXT: Train Stage 2 model (Intensity Prediction)")
    print("="*80)
    print(f"\nRun: python -m src.ml.train_stage2 --horizon {args.horizon}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
