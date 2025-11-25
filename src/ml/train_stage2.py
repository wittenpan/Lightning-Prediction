"""
Stage 2 Model Training: Intensity Prediction

For ACTIVE cells (strike_count_30min > 0), predicts if they will have
HIGH intensity activity (≥5 strikes) in the next time window.

This is the second stage of our two-stage prediction system.

Usage:
    python -m src.ml.train_stage2 --horizon 15min
    python -m src.ml.train_stage2 --horizon 1h
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
    average_precision_score
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Stage2Trainer:
    """
    Stage 2: Intensity Prediction Model
    
    Predicts: Will ACTIVE cells have ≥5 strikes in next window?
    
    Only trains on cells with recent activity (strike_count_30min > 0).
    This reduces imbalance from 873:1 to ~20-50:1.
    """
    
    def __init__(self, horizon: str = '15min', output_dir: Path = None):
        """
        Initialize Stage 2 trainer.
        
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
        self.target_col = 'stage2_target_15min' if horizon == '15min' else 'stage2_target_1h'
        
    def filter_active_cells(self, X, y, split_name):
        """Filter to only active cells (strike_count_30min > 0)."""
        if 'strike_count_30min' not in X.columns:
            raise ValueError("strike_count_30min column not found!")
        
        active_mask = X['strike_count_30min'] > 0
        X_active = X[active_mask].copy()
        y_active = y[active_mask].copy()
        
        logger.info(f"{split_name}: {len(X):,} → {len(X_active):,} ({100*active_mask.mean():.1f}% active)")
        
        return X_active, y_active
    
    def load_data(self, data_dir: Path = None) -> tuple:
        """Load training data for Stage 2 (active cells only)."""
        data_dir = data_dir or Path('data/processed')

        logger.info("="*80)
        logger.info(f"STAGE 2: INTENSITY PREDICTION ({self.horizon})")
        logger.info("="*80)
        logger.info(f"Target: ≥5 strikes in next {self.horizon}")
        logger.info(f"Data: Pre-filtered to active cells (strike_count_30min > 0)")
        logger.info(f"Data directory: {data_dir}\n")

        # Load Stage 2 datasets (already filtered to active cells)
        logger.info("Loading training data (pre-filtered)...")
        X_train = pd.read_parquet(data_dir / 'stage2_features_train.parquet')
        y_train_all = pd.read_parquet(data_dir / 'stage2_labels_train.parquet')
        y_train = y_train_all[self.target_col]

        logger.info("Loading validation data (pre-filtered)...")
        X_val = pd.read_parquet(data_dir / 'stage2_features_val.parquet')
        y_val_all = pd.read_parquet(data_dir / 'stage2_labels_val.parquet')
        y_val = y_val_all[self.target_col]

        logger.info("Loading test data (pre-filtered)...")
        X_test = pd.read_parquet(data_dir / 'stage2_features_test.parquet')
        y_test_all = pd.read_parquet(data_dir / 'stage2_labels_test.parquet')
        y_test = y_test_all[self.target_col]

        # Remove identifier columns
        id_cols = ['h3_cell', 'timestamp']
        for col in id_cols:
            if col in X_train.columns:
                X_train = X_train.drop(columns=[col])
                X_val = X_val.drop(columns=[col])
                X_test = X_test.drop(columns=[col])

        self.feature_names = list(X_train.columns)

        logger.info(f"  Features: {len(self.feature_names)}")
        
        # Class distribution (in active cells)
        logger.info(f"\nClass distribution (active cells only):")
        for name, y in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
            pos = y.sum()
            neg = len(y) - pos
            ratio = neg / pos if pos > 0 else 0
            logger.info(f"  {name}: {pos:,} high ({100*y.mean():.2f}%), {neg:,} low, ratio {ratio:.1f}:1")
        
        logger.info("="*80 + "\n")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train(self, X_train, y_train, X_val, y_val, use_smote=False):
        """Train Stage 2 model with optional SMOTE resampling."""
        logger.info("="*80)
        logger.info("TRAINING STAGE 2 MODEL")
        logger.info("="*80)

        # Calculate original class imbalance
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        auto_weight = neg_count / pos_count

        logger.info(f"\nOriginal class imbalance:")
        logger.info(f"  Low intensity: {neg_count:,}")
        logger.info(f"  High intensity (≥{self.metadata.get('threshold', 3)}): {pos_count:,}")
        logger.info(f"  Imbalance ratio: {auto_weight:.1f}:1")

        # Apply SMOTE if enabled
        if use_smote and auto_weight > 15:  # Only use SMOTE if imbalance > 15:1
            try:
                from imblearn.over_sampling import SMOTE

                logger.info(f"\n🔄 Applying SMOTE to balance training data...")
                logger.info(f"  Target ratio: 10:1 (conservative to save memory)")

                smote = SMOTE(
                    sampling_strategy=0.1,  # Target 10:1 ratio (not 1:1 - saves memory!)
                    random_state=42,
                    k_neighbors=min(5, pos_count - 1)
                )

                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

                new_pos = (y_train_balanced == 1).sum()
                new_neg = (y_train_balanced == 0).sum()
                new_ratio = new_neg / new_pos

                logger.info(f"\n✅ SMOTE complete:")
                logger.info(f"  Before: {len(X_train):,} samples ({pos_count:,} positive)")
                logger.info(f"  After:  {len(X_train_balanced):,} samples ({new_pos:,} positive)")
                logger.info(f"  New ratio: {new_ratio:.1f}:1")
                logger.info(f"  Memory increase: {len(X_train_balanced)/len(X_train):.1f}x")

                # Use balanced data
                X_train = X_train_balanced
                y_train = y_train_balanced

                # Reduce scale_pos_weight since SMOTE already balanced
                scale_pos_weight = min(new_ratio, 10.0)

            except ImportError:
                logger.warning("⚠️  imbalanced-learn not installed. Install with: pip install imbalanced-learn")
                logger.warning("  Falling back to class weights only")
                scale_pos_weight = min(auto_weight, 80.0)
        else:
            logger.info(f"\n⏭️  Skipping SMOTE (imbalance {auto_weight:.1f}:1 is acceptable or SMOTE disabled)")
            scale_pos_weight = min(auto_weight, 80.0)

        logger.info(f"\nFinal scale_pos_weight: {scale_pos_weight:.1f}")
        
        params = {
            # Tree structure (balanced for precision/recall)
            'max_depth': 6,  # Keep moderate depth
            'min_child_weight': 5,  # Higher = more conservative = better precision

            # Learning
            'learning_rate': 0.03,  # Lower = more careful = better precision
            'n_estimators': 300,  # More trees to compensate for lower LR

            # Regularization (prevent overfitting)
            'gamma': 0.2,  # Higher = more conservative splits
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 2.0,  # L2 regularization (doubled for more regularization)

            # Sampling
            'subsample': 0.7,  # More aggressive sampling
            'colsample_bytree': 0.7,
            'colsample_bylevel': 0.7,

            # Class imbalance
            'scale_pos_weight': scale_pos_weight,

            # Objective
            'objective': 'binary:logistic',
            'eval_metric': ['aucpr', 'auc'],  # Prioritize PR-AUC for imbalanced data

            # Performance
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 20  # More patience for lower LR
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
        """Evaluate Stage 2 model."""
        logger.info(f"Evaluating on {dataset_name} set (active cells only)...")
        
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
        logger.info(f"  TN (low): {tn:,}  |  FP (predicted high, was low): {fp:,}")
        logger.info(f"  FN (predicted low, was high): {fn:,}  |  TP (high): {tp:,}")
        
        return metrics
    
    def save_model(self):
        """Save Stage 2 model."""
        logger.info("="*80)
        logger.info("SAVING STAGE 2 MODEL")
        logger.info("="*80)
        
        model_file = self.output_dir / f'stage2_model_{self.horizon}.ubj'
        self.model.save_model(model_file)
        logger.info(f"  Saved model to: {model_file}")
        
        self.metadata['stage'] = 2
        self.metadata['horizon'] = self.horizon
        self.metadata['target'] = 'intensity_prediction'
        self.metadata['threshold'] = 3  # ≥3 strikes (changed from 5)
        self.metadata['filter'] = 'strike_count_30min > 0'
        self.metadata['training_date'] = datetime.now().isoformat()
        self.metadata['feature_count'] = len(self.feature_names)
        self.metadata['feature_names'] = self.feature_names
        
        metadata_file = self.output_dir / f'stage2_metadata_{self.horizon}.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        logger.info(f"  Saved metadata to: {metadata_file}")
        
        logger.info("="*80 + "\n")
    
    def run_pipeline(self, data_dir=None):
        """Run complete Stage 2 training pipeline."""
        start_time = datetime.now()
        
        logger.info("\n" + "="*80)
        logger.info(f"STAGE 2 TRAINING: INTENSITY PREDICTION ({self.horizon})")
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
        logger.info("STAGE 2 TRAINING COMPLETE!")
        logger.info("="*80)
        logger.info(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")
        logger.info(f"\nTest Performance (active cells only):")
        logger.info(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {test_metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {test_metrics['f1_score']:.4f}")
        logger.info(f"\nModel saved to: {self.output_dir}")
        logger.info("="*80 + "\n")
        
        return self.model, test_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Train Stage 2 model (Intensity Prediction)'
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
    
    trainer = Stage2Trainer(
        horizon=args.horizon,
        output_dir=Path(args.output_dir) if args.output_dir else None
    )
    
    model, metrics = trainer.run_pipeline(
        data_dir=Path(args.data_dir) if args.data_dir else None
    )
    
    print("\n" + "="*80)
    print("BOTH STAGES TRAINED!")
    print("="*80)
    print("\nNext: Evaluate combined two-stage system")
    print(f"Run: python -m src.ml.evaluate_twostage --horizon {args.horizon}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
