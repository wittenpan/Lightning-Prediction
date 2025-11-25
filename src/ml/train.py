"""
Lightning Strike Prediction - Model Training

Trains an XGBoost classifier on processed lightning strike features to predict
whether a grid cell will have >5 strikes in the next 15 minutes.

Usage:
    python -m src.ml.train
    
    # With custom paths
    python -m src.ml.train --data-dir data/processed --output-dir data/models
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LightningPredictor:
    """
    Lightning strike prediction model trainer.
    
    Handles:
    - Loading processed data
    - Training XGBoost with class imbalance handling
    - Evaluation on validation and test sets
    - Feature importance analysis
    - Model saving
    """
    
    def __init__(self, data_dir: Path = None, output_dir: Path = None):
        """
        Initialize trainer.
        
        Args:
            data_dir: Directory with processed data (default: data/processed)
            output_dir: Directory for saving models (default: data/models)
        """
        self.data_dir = data_dir or Path('data/processed')
        self.output_dir = output_dir or Path('data/models')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.feature_names = None
        self.metadata = {}
        
    def load_data(self) -> tuple:
        """
        Load processed training, validation, and test data.
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        logger.info("="*80)
        logger.info("LOADING PROCESSED DATA")
        logger.info("="*80)
        logger.info(f"Data directory: {self.data_dir}")
        
        # Load training data
        logger.info("\nLoading training data...")
        X_train = pd.read_parquet(self.data_dir / 'features_train.parquet')
        y_train = pd.read_parquet(self.data_dir / 'labels_train.parquet')
        logger.info(f"  Train: {len(X_train):,} samples")
        
        # Load validation data
        logger.info("Loading validation data...")
        X_val = pd.read_parquet(self.data_dir / 'features_val.parquet')
        y_val = pd.read_parquet(self.data_dir / 'labels_val.parquet')
        logger.info(f"  Val:   {len(X_val):,} samples")
        
        # Load test data
        logger.info("Loading test data...")
        X_test = pd.read_parquet(self.data_dir / 'features_test.parquet')
        y_test = pd.read_parquet(self.data_dir / 'labels_test.parquet')
        logger.info(f"  Test:  {len(X_test):,} samples")
        
        # Remove identifier columns (h3_cell, timestamp)
        id_cols = ['h3_cell', 'timestamp']
        for col in id_cols:
            if col in X_train.columns:
                X_train = X_train.drop(columns=[col])
                X_val = X_val.drop(columns=[col])
                X_test = X_test.drop(columns=[col])
        
        self.feature_names = list(X_train.columns)
        
        # Use binary target
        y_train = y_train['target_binary']
        y_val = y_val['target_binary']
        y_test = y_test['target_binary']
        
        logger.info(f"\nFeatures: {len(self.feature_names)}")
        logger.info(f"Feature names: {self.feature_names[:5]}... (showing first 5)")
        
        # Class distribution
        logger.info("\nClass distribution:")
        for name, y in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
            pos_pct = 100 * y.sum() / len(y)
            logger.info(f"  {name}: {y.sum():,} positive ({pos_pct:.2f}%)")
        
        logger.info("="*80 + "\n")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
             X_val: pd.DataFrame, y_val: pd.Series) -> xgb.XGBClassifier:
        """
        Train XGBoost model with class imbalance handling.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Trained XGBoost model
        """
        logger.info("="*80)
        logger.info("TRAINING MODEL")
        logger.info("="*80)
        
        # Calculate class weights for imbalance
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        # instead of using ratio, lets use 30
        #scale_pos_weight = neg_count / pos_count
        scale_pos_weight = 30
        logger.info(f"\nClass imbalance:")
        logger.info(f"  Negative samples: {neg_count:,}")
        logger.info(f"  Positive samples: {pos_count:,}")
        logger.info(f"  Ratio (neg/pos): {scale_pos_weight:.2f}")
        logger.info(f"  Using scale_pos_weight={scale_pos_weight:.2f} to handle imbalance")
        
        # XGBoost parameters
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'scale_pos_weight': scale_pos_weight,  # Handle class imbalance
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',  # Faster for large datasets
            'early_stopping_rounds': 10  # Enable early stopping
        }
        
        logger.info("\nModel parameters:")
        for key, value in params.items():
            logger.info(f"  {key}: {value}")
        
        # Create model
        self.model = xgb.XGBClassifier(**params)
        
        # Train with early stopping
        logger.info("\nTraining model...")
        logger.info("(Early stopping: will stop if no improvement for 10 rounds)")
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=20  # Print every 20 rounds
        )
        
        logger.info(f"\nTraining complete!")
        
        # Get best iteration and score
        best_iteration = self.model.best_iteration
        best_score = self.model.best_score
        logger.info(f"  Best iteration: {best_iteration}")
        logger.info(f"  Best validation AUC: {best_score:.4f}")
        
        self.metadata['best_iteration'] = int(best_iteration)
        self.metadata['best_score'] = float(best_score)
        
        self.metadata['model_params'] = params
        
        logger.info("="*80 + "\n")
        
        return self.model
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, 
                dataset_name: str = "Test") -> dict:
        """
        Evaluate model on a dataset.
        
        Args:
            X: Features
            y: True labels
            dataset_name: Name for logging (e.g., "Test", "Validation")
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating on {dataset_name} set...")
        
        # Predictions
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': (y_pred == y).mean(),
            'roc_auc': roc_auc_score(y, y_pred_proba),
            'avg_precision': average_precision_score(y, y_pred_proba),
        }
        
        # Classification report
        report = classification_report(y, y_pred, output_dict=True)
        
        # Confusion matrix
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
        
        # Log results
        logger.info(f"\n{dataset_name} Set Metrics:")
        logger.info(f"  ROC-AUC:        {metrics['roc_auc']:.4f}")
        logger.info(f"  Avg Precision:  {metrics['avg_precision']:.4f}")
        logger.info(f"  Accuracy:       {metrics['accuracy']:.4f}")
        logger.info(f"  Precision:      {metrics['precision']:.4f}")
        logger.info(f"  Recall:         {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:       {metrics['f1_score']:.4f}")
        
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN: {tn:,}  |  FP: {fp:,}")
        logger.info(f"  FN: {fn:,}  |  TP: {tp:,}")
        
        return metrics
    
    def analyze_features(self, top_n: int = 20):
        """
        Analyze and visualize feature importance.
        
        Args:
            top_n: Number of top features to display
        """
        logger.info("\n" + "="*80)
        logger.info("FEATURE IMPORTANCE ANALYSIS")
        logger.info("="*80)
        
        # Get feature importance
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\nTop {top_n} Most Important Features:")
        for idx, row in feature_importance.head(top_n).iterrows():
            logger.info(f"  {row['feature']:30s}: {row['importance']:.4f}")
        
        # Save feature importance
        importance_file = self.output_dir / 'feature_importance.csv'
        feature_importance.to_csv(importance_file, index=False)
        logger.info(f"\nSaved feature importance to: {importance_file}")
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        
        plot_file = self.output_dir / 'feature_importance.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to: {plot_file}")
        plt.close()
        
        self.metadata['top_features'] = feature_importance.head(10).to_dict('records')
        
        logger.info("="*80 + "\n")
    
    def save_model(self):
        """Save trained model and metadata."""
        logger.info("="*80)
        logger.info("SAVING MODEL")
        logger.info("="*80)
        
        # Save XGBoost model (using .ubj format for XGBoost 2.0+)
        model_file = self.output_dir / 'lightning_model.ubj'
        self.model.save_model(model_file)
        logger.info(f"  Saved model to: {model_file}")
        
        # Save metadata
        self.metadata['training_date'] = datetime.now().isoformat()
        self.metadata['feature_count'] = len(self.feature_names)
        self.metadata['feature_names'] = self.feature_names
        
        metadata_file = self.output_dir / 'model_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        logger.info(f"  Saved metadata to: {metadata_file}")
        
        logger.info("="*80 + "\n")
    
    def run_training_pipeline(self):
        """Execute complete training pipeline."""
        start_time = datetime.now()
        
        logger.info("\n" + "="*80)
        logger.info("LIGHTNING STRIKE PREDICTION - MODEL TRAINING")
        logger.info("="*80)
        logger.info(f"Started: {start_time}")
        logger.info("="*80 + "\n")
        
        # 1. Load data
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_data()
        
        # 2. Train model
        self.train(X_train, y_train, X_val, y_val)
        
        # 3. Evaluate
        logger.info("\n" + "="*80)
        logger.info("MODEL EVALUATION")
        logger.info("="*80)
        
        val_metrics = self.evaluate(X_val, y_val, "Validation")
        test_metrics = self.evaluate(X_test, y_test, "Test")
        
        self.metadata['validation_metrics'] = val_metrics
        self.metadata['test_metrics'] = test_metrics
        
        logger.info("="*80 + "\n")
        
        # 4. Feature importance
        self.analyze_features()
        
        # 5. Save model
        self.save_model()
        
        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*80)
        logger.info(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"\nTest Set Performance:")
        logger.info(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall: {test_metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {test_metrics['f1_score']:.4f}")
        logger.info(f"\nModel saved to: {self.output_dir}")
        logger.info("="*80 + "\n")
        
        return self.model, test_metrics


def main():
    """Command-line interface for model training."""
    parser = argparse.ArgumentParser(
        description='Train lightning strike prediction model'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Directory with processed data (default: data/processed)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for model (default: data/models)'
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = LightningPredictor(
        data_dir=Path(args.data_dir) if args.data_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None
    )
    
    # Run training
    model, metrics = trainer.run_training_pipeline()
    
    print("\n" + "="*80)
    print("READY FOR INFERENCE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review feature importance in data/models/feature_importance.png")
    print("2. Check model_metadata.json for detailed metrics")
    print("3. Load model for inference:")
    print("   model = xgb.XGBClassifier()")
    print("   model.load_model('data/models/lightning_model.ubj')")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()