"""
Data Preparation Script for Lightning Strike Prediction

This script processes historical lightning strike data and prepares it for ML training:
1. Load all parquet files from data/raw/
2. Combine into single DataFrame
3. Run feature extraction (temporal, spatial, pattern)
4. Create target labels
5. Split into train/validation/test sets (TEMPORAL split)
6. Save processed data to data/processed/

Usage:
    python src/processing/data_preparation.py
    
    # Or with custom config
    python src/processing/data_preparation.py --config configs/custom_config.yml
    
Output Files:
    data/processed/features_train.parquet
    data/processed/features_val.parquet
    data/processed/features_test.parquet
    data/processed/labels_train.parquet
    data/processed/labels_val.parquet
    data/processed/labels_test.parquet
    data/processed/metadata.json
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import argparse
from datetime import datetime
from typing import Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from config_loader import GridConfig
from grid_manager import GridManager
from feature_engineering import FeatureExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreparation:
    """
    Orchestrates the complete data preparation pipeline.
    
    This class manages the end-to-end process of transforming raw
    lightning strike data into ML-ready features and labels.
    
    Process Flow:
    1. Load raw parquet files
    2. Validate data quality
    3. Extract features (temporal, spatial, pattern)
    4. Create target labels
    5. Split data temporally (train/val/test)
    6. Save processed datasets
    7. Generate metadata
    """
    
    def __init__(self, config: GridConfig = None):
        """
        Initialize data preparation pipeline.
        
        Args:
            config: GridConfig instance (creates default if None)
        """
        self.config = config or GridConfig()
        self.grid = GridManager(resolution=self.config.h3_resolution)
        self.extractor = FeatureExtractor(self.config, self.grid)
        
        logger.info("Data Preparation Pipeline Initialized")
        logger.info(f"H3 Resolution: {self.config.h3_resolution}")
        logger.info(f"Lookback Windows: {self.config.lookback_windows}")
        logger.info(f"Prediction Window: {self.config.default_prediction_window} minutes")
        logger.info(f"Strike Threshold: {self.config.strike_threshold}")
        
    def load_raw_data(self, data_dir: Path = None) -> pd.DataFrame:
        """
        Load all parquet files from raw data directory.
        
        Args:
            data_dir: Path to directory with parquet files
                     (defaults to config.raw_data_path)
        
        Returns:
            Combined DataFrame with all strikes
            
        Process:
        - Find all .parquet files in directory
        - Load each file
        - Combine into single DataFrame
        - Sort by timestamp
        - Validate required columns
        """
        if data_dir is None:
            data_dir = Path(self.config.raw_data_path)
        
        logger.info(f"Loading raw data from: {data_dir}")
        
        # Find all parquet files
        parquet_files = sorted(data_dir.glob('lightning_batch_*.parquet'))
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")
        
        logger.info(f"Found {len(parquet_files)} parquet files:")
        for f in parquet_files:
            logger.info(f"  - {f.name}")
        
        # Load files in parallel (3-5x faster for multiple files)
        logger.info(f"\nLoading {len(parquet_files)} files in parallel...")
        
        def load_single_file(file_path):
            """Helper function to load a single parquet file."""
            df = pd.read_parquet(file_path)
            return file_path.name, df, len(df)
        
        dfs = []
        if len(parquet_files) > 1:
            # Parallel loading for multiple files (using threads - perfect for I/O)
            with ThreadPoolExecutor(max_workers=min(4, len(parquet_files))) as executor:
                futures = {executor.submit(load_single_file, f): f for f in parquet_files}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Loading files", unit="file"):
                    file_name, df, strike_count = future.result()
                    logger.info(f"  Loaded {file_name}: {strike_count:,} strikes")
                    dfs.append(df)
        else:
            # Single file - no need for parallelization
            file_name, df, strike_count = load_single_file(parquet_files[0])
            logger.info(f"  Loaded {file_name}: {strike_count:,} strikes")
            dfs.append(df)
        
        # Combine all DataFrames
        logger.info("\nCombining DataFrames...")
        df_combined = pd.concat(dfs, ignore_index=True)
        
        logger.info(f"\nCombined data:")
        logger.info(f"  Total strikes: {len(df_combined):,}")
        logger.info(f"  Columns: {list(df_combined.columns)}")
        
        # Validate required columns
        required_cols = ['time', 'lat', 'lon']
        missing_cols = [col for col in required_cols if col not in df_combined.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert timestamp to datetime
        if df_combined['time'].dtype == 'int64':
            df_combined['time'] = pd.to_datetime(df_combined['time'], unit='ns')
        
        # Sort by time (CRITICAL for time series!)
        df_combined = df_combined.sort_values('time').reset_index(drop=True)
        
        logger.info(f"  Time range: {df_combined['time'].min()} to {df_combined['time'].max()}")
        logger.info(f"  Duration: {(df_combined['time'].max() - df_combined['time'].min()).total_seconds() / 3600:.1f} hours")
        
        return df_combined
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Check data quality and generate statistics.
        
        Args:
            df: Raw strikes DataFrame
            
        Returns:
            Dictionary with quality metrics
        """
        logger.info("\n" + "="*80)
        logger.info("DATA QUALITY VALIDATION")
        logger.info("="*80)
        
        metrics = {}
        
        # Basic stats
        metrics['total_strikes'] = len(df)
        metrics['unique_locations'] = df[['lat', 'lon']].drop_duplicates().shape[0]
        
        # Missing values
        missing = df.isnull().sum()
        if missing.any():
            logger.warning("Missing values detected:")
            for col, count in missing[missing > 0].items():
                logger.warning(f"  {col}: {count} ({100*count/len(df):.2f}%)")
                metrics[f'missing_{col}'] = int(count)
        else:
            logger.info("✓ No missing values")
        
        # Duplicate timestamps
        duplicates = df['time'].duplicated().sum()
        if duplicates > 0:
            logger.info(f"  Duplicate timestamps: {duplicates} ({100*duplicates/len(df):.2f}%)")
            logger.info("  (This is normal - multiple strikes can occur simultaneously)")
            metrics['duplicate_timestamps'] = int(duplicates)
        
        # Geographic bounds
        lat_min, lat_max = df['lat'].min(), df['lat'].max()
        lon_min, lon_max = df['lon'].min(), df['lon'].max()
        logger.info(f"\nGeographic bounds:")
        logger.info(f"  Latitude: {lat_min:.2f} to {lat_max:.2f}")
        logger.info(f"  Longitude: {lon_min:.2f} to {lon_max:.2f}")
        metrics['lat_range'] = [float(lat_min), float(lat_max)]
        metrics['lon_range'] = [float(lon_min), float(lon_max)]
        
        # Temporal coverage
        time_gaps = df['time'].diff()
        max_gap = time_gaps.max()
        logger.info(f"\nTemporal coverage:")
        logger.info(f"  Maximum gap between strikes: {max_gap}")
        metrics['max_time_gap_seconds'] = float(max_gap.total_seconds())
        
        # Strikes per minute
        strikes_per_min = len(df) / ((df['time'].max() - df['time'].min()).total_seconds() / 60)
        logger.info(f"  Average strike rate: {strikes_per_min:.2f} strikes/minute")
        metrics['avg_strike_rate_per_min'] = float(strikes_per_min)
        
        logger.info("="*80 + "\n")
        
        return metrics
    
    def extract_features_and_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run complete feature extraction pipeline.
        
        This is the main processing step that transforms raw strikes
        into ML-ready features.
        
        Args:
            df: Raw strikes DataFrame (with h3_cell column added)
            
        Returns:
            DataFrame with features and target labels
            
        Process:
        1. Extract temporal features (rolling windows)
        2. Extract spatial features (neighbors)
        3. Extract pattern features (time of day, etc.)
        4. Create target labels (future strike counts)
        5. Merge all features
        """
        logger.info("\n" + "="*80)
        logger.info("FEATURE EXTRACTION")
        logger.info("="*80)
        
        # Extract features (with memory optimization: skip cells with <2 total strikes)
        logger.info("\nExtracting features...")
        MIN_CELL_STRIKES = 2  # Skip cells with very few strikes (saves memory)
        logger.info(f"  Pre-filtering cells with <{MIN_CELL_STRIKES} total strikes (memory optimization)")
        features = self.extractor.extract_features(df, min_cell_strikes=MIN_CELL_STRIKES)
        
        logger.info(f"\nFeatures extracted:")
        logger.info(f"  Shape: {features.shape}")
        logger.info(f"  Unique cells: {features['h3_cell'].nunique():,}")
        logger.info(f"  Time points: {features['timestamp'].nunique():,}")
        
        # Log feature columns (before adding targets)
        feature_cols = [col for col in features.columns 
                       if col not in ['h3_cell', 'timestamp']]
        logger.info(f"  Feature count: {len(feature_cols)}")
        
        # Create targets
        logger.info("\nCreating target labels for TWO-STAGE system...")
        logger.info("  (This may take several minutes for large datasets)")
        features_with_targets = self.extractor.create_targets(features, df)
        
        # Add Stage 1 targets (activity detection: ≥1 strike)
        logger.info("\n" + "-"*80)
        logger.info("STAGE 1 TARGETS: Activity Detection (≥1 strike)")
        logger.info("-"*80)
        
        features_with_targets['stage1_target_15min'] = (features_with_targets['target_count'] >= 1).astype(int)
        
        stage1_pos = features_with_targets['stage1_target_15min'].sum()
        stage1_total = len(features_with_targets)
        stage1_pct = 100 * stage1_pos / stage1_total
        stage1_ratio = (stage1_total - stage1_pos) / stage1_pos if stage1_pos > 0 else 0
        
        logger.info(f"15-30 min horizon:")
        logger.info(f"  Positive (≥1 strike): {stage1_pos:,} ({stage1_pct:.2f}%)")
        logger.info(f"  Negative (0 strikes): {stage1_total - stage1_pos:,}")
        logger.info(f"  Imbalance ratio: {stage1_ratio:.1f}:1")
        
        # Add 1-hour targets (OPTIONAL - skip to save memory/time)
        # Set CREATE_1H_TARGETS = False to skip 1h target creation
        CREATE_1H_TARGETS = False  # Set to True if you need 1h predictions
        
        if CREATE_1H_TARGETS:
            logger.info("\nCreating 1-hour prediction targets...")
            logger.info("  ⚠️  This doubles memory usage and processing time!")
            features_1h = self.extractor.create_targets(features, df, horizon_min=60)
            features_with_targets['target_count_1h'] = features_1h['target_count']
            features_with_targets['target_binary_1h'] = (features_1h['target_count'] >= 5).astype(int)
            features_with_targets['stage1_target_1h'] = (features_1h['target_count'] >= 1).astype(int)
            
            stage1_1h_pos = features_with_targets['stage1_target_1h'].sum()
            stage1_1h_pct = 100 * stage1_1h_pos / stage1_total
            stage1_1h_ratio = (stage1_total - stage1_1h_pos) / stage1_1h_pos if stage1_1h_pos > 0 else 0
            
            logger.info(f"1 hour horizon:")
            logger.info(f"  Positive (≥1 strike): {stage1_1h_pos:,} ({stage1_1h_pct:.2f}%)")
            logger.info(f"  Imbalance ratio: {stage1_1h_ratio:.1f}:1")
        else:
            logger.info("\n⏭️  Skipping 1-hour targets (set CREATE_1H_TARGETS = True to enable)")
            # Create dummy columns so downstream code doesn't break
            features_with_targets['target_count_1h'] = 0
            features_with_targets['target_binary_1h'] = 0
            features_with_targets['stage1_target_1h'] = 0
        
        # Add Stage 2 targets (intensity: ≥5 strikes, for active cells only)
        logger.info("\n" + "-"*80)
        logger.info("STAGE 2 TARGETS: Intensity Prediction (≥5 strikes, active cells only)")
        logger.info("-"*80)
        
        features_with_targets['stage2_target_15min'] = features_with_targets['target_binary']
        features_with_targets['stage2_target_1h'] = features_with_targets['target_binary_1h']
        
        # Calculate stats for active cells only
        active_mask = features_with_targets['strike_count_30min'] > 0
        active_count = active_mask.sum()
        active_pct = 100 * active_count / stage1_total
        
        logger.info(f"Active cells filter (strike_count_30min > 0):")
        logger.info(f"  Total cells: {stage1_total:,}")
        logger.info(f"  Active cells: {active_count:,} ({active_pct:.2f}%)")
        
        if active_count > 0:
            stage2_pos = features_with_targets.loc[active_mask, 'stage2_target_15min'].sum()
            stage2_pct = 100 * stage2_pos / active_count
            stage2_ratio = (active_count - stage2_pos) / stage2_pos if stage2_pos > 0 else 0
            
            logger.info(f"\n15-30 min horizon (active cells only):")
            logger.info(f"  Positive (≥5 strikes): {stage2_pos:,} ({stage2_pct:.2f}%)")
            logger.info(f"  Negative (1-4 strikes): {active_count - stage2_pos:,}")
            logger.info(f"  Imbalance ratio: {stage2_ratio:.1f}:1")
            
            if CREATE_1H_TARGETS:
                stage2_1h_pos = features_with_targets.loc[active_mask, 'stage2_target_1h'].sum()
                stage2_1h_pct = 100 * stage2_1h_pos / active_count
                stage2_1h_ratio = (active_count - stage2_1h_pos) / stage2_1h_pos if stage2_1h_pos > 0 else 0
                
                logger.info(f"\n1 hour horizon (active cells only):")
                logger.info(f"  Positive (≥5 strikes): {stage2_1h_pos:,} ({stage2_1h_pct:.2f}%)")
                logger.info(f"  Imbalance ratio: {stage2_1h_ratio:.1f}:1")
        
        # Log all feature columns
        logger.info("\n" + "="*80)
        logger.info("ALL FEATURES:")
        logger.info("="*80)
        all_feature_cols = [col for col in features_with_targets.columns 
                           if col not in ['h3_cell', 'timestamp', 'target_count', 'target_binary',
                                         'target_count_1h', 'target_binary_1h',
                                         'stage1_target_15min', 'stage1_target_1h',
                                         'stage2_target_15min', 'stage2_target_1h']]
        for col in sorted(all_feature_cols):
            logger.info(f"  - {col}")
        
        logger.info(f"\nTarget columns created:")
        logger.info(f"  - target_count, target_binary (original 15-30min, ≥5 strikes)")
        if CREATE_1H_TARGETS:
            logger.info(f"  - target_count_1h, target_binary_1h (1 hour, ≥5 strikes)")
        logger.info(f"  - stage1_target_15min (15-30min, ≥1 strike)")
        if CREATE_1H_TARGETS:
            logger.info(f"  - stage1_target_1h (1 hour, ≥1 strike)")
        logger.info(f"  - stage2_target_15min (15-30min, ≥5 strikes)")
        if CREATE_1H_TARGETS:
            logger.info(f"  - stage2_target_1h (1 hour, ≥5 strikes)")
        else:
            logger.info(f"\n  Note: 1h targets skipped (set CREATE_1H_TARGETS=True to enable)")
        
        logger.info("="*80 + "\n")
        
        return features_with_targets
    
    def create_temporal_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally into train/validation/test sets.
        
        CRITICAL: We use TEMPORAL split, not random split!
        - Train on early data
        - Validate on middle data  
        - Test on late data
        
        This simulates real prediction scenario and prevents data leakage.
        
        Args:
            df: DataFrame with features and targets
            
        Returns:
            Tuple of (train_df, val_df, test_df)
            
        Split Strategy:
        - Sort by timestamp
        - Train: first 70% of time
        - Validation: next 15% of time
        - Test: final 15% of time
        """
        logger.info("\n" + "="*80)
        logger.info("TEMPORAL DATA SPLIT (OPTIMIZED)")
        logger.info("="*80)
        
        # Sort by timestamp (should already be sorted, but ensure)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate split points based on time, not sample count
        min_time = df['timestamp'].min()
        max_time = df['timestamp'].max()
        total_duration = (max_time - min_time).total_seconds()
        
        # Calculate time split points
        train_ratio = self.config.train_ratio
        val_ratio = self.config.validation_ratio
        
        train_end_time = min_time + pd.Timedelta(seconds=total_duration * train_ratio)
        val_end_time = train_end_time + pd.Timedelta(seconds=total_duration * val_ratio)
        
        # OPTIMIZATION: Use vectorized masks (faster, less memory than filtering 3 times)
        timestamps = df['timestamp']
        train_mask = timestamps <= train_end_time
        val_mask = (timestamps > train_end_time) & (timestamps <= val_end_time)
        test_mask = timestamps > val_end_time
        
        # Split data (no copy needed - we won't modify these)
        train_df = df[train_mask]
        val_df = df[val_mask]
        test_df = df[test_mask]
        
        logger.info(f"\nSplit method: {self.config.split_method}")
        logger.info(f"Train ratio: {train_ratio}")
        logger.info(f"Validation ratio: {val_ratio}")
        logger.info(f"Test ratio: {1 - train_ratio - val_ratio}")
        
        logger.info(f"\nTime ranges:")
        logger.info(f"  Train: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
        logger.info(f"  Val:   {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
        logger.info(f"  Test:  {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
        
        logger.info(f"\nSample counts:")
        logger.info(f"  Train: {len(train_df):,} samples ({100*len(train_df)/len(df):.1f}%)")
        logger.info(f"  Val:   {len(val_df):,} samples ({100*len(val_df)/len(df):.1f}%)")
        logger.info(f"  Test:  {len(test_df):,} samples ({100*len(test_df)/len(df):.1f}%)")
        
        # Check class balance in each set
        for name, data in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            pos_pct = 100 * data['target_binary'].sum() / len(data)
            logger.info(f"  {name} positive rate: {pos_pct:.2f}%")
        
        logger.info("="*80 + "\n")
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, 
                           train_df: pd.DataFrame,
                           val_df: pd.DataFrame, 
                           test_df: pd.DataFrame,
                           metadata: Dict,
                           output_dir: Path = None):
        """
        Save processed datasets to disk (OPTIMIZED for two-stage system).
        
        Saves THREE sets of data:
        1. Original features + original targets (for single-stage or analysis)
        2. Stage 1 features + Stage 1 labels (activity detection)
        3. Stage 2 features + Stage 2 labels (intensity prediction, active cells only)
        
        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            metadata: Dictionary with processing metadata
            output_dir: Output directory (defaults to config.processed_data_path)
        """
        if output_dir is None:
            output_dir = Path(self.config.processed_data_path)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("\n" + "="*80)
        logger.info("SAVING PROCESSED DATA (TWO-STAGE SYSTEM)")
        logger.info("="*80)
        logger.info(f"\nOutput directory: {output_dir}")
        
        # Define shared feature columns (exclude all target columns)
        target_related_cols = ['target_count', 'target_binary', 'target_count_1h', 'target_binary_1h',
                              'stage1_target_15min', 'stage1_target_1h',
                              'stage2_target_15min', 'stage2_target_1h']
        feature_cols = [col for col in train_df.columns if col not in target_related_cols]
        
        logger.info(f"\n📦 Saving THREE dataset variants:")
        logger.info(f"  1. Original (legacy compatibility)")
        logger.info(f"  2. Stage 1: Activity Detection (all cells)")
        logger.info(f"  3. Stage 2: Intensity Prediction (active cells only)")
        
        # === SAVE ORIGINAL FORMAT (for backward compatibility) ===
        logger.info(f"\n1️⃣  Saving ORIGINAL format (15-30min, ≥5 strikes)...")
        target_cols_original = ['target_count', 'target_binary']
        
        for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            features_file = output_dir / f'features_{name}.parquet'
            labels_file = output_dir / f'labels_{name}.parquet'
            
            df[feature_cols].to_parquet(features_file, index=False, compression='snappy')
            df[target_cols_original].to_parquet(labels_file, index=False, compression='snappy')
            
            logger.info(f"  ✓ {name}: {len(df):,} samples")
        
        # === SAVE STAGE 1 DATA (Activity Detection: ≥1 strike) ===
        logger.info(f"\n2️⃣  Saving STAGE 1 data (Activity Detection: ≥1 strike)...")
        stage1_label_col = 'stage1_target_15min'  # Binary: ≥1 strike or not
        
        for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            features_file = output_dir / f'stage1_features_{name}.parquet'
            labels_file = output_dir / f'stage1_labels_{name}.parquet'
            
            df[feature_cols].to_parquet(features_file, index=False, compression='snappy')
            df[[stage1_label_col]].to_parquet(labels_file, index=False, compression='snappy')
            
            pos_count = df[stage1_label_col].sum()
            pos_pct = 100 * pos_count / len(df)
            logger.info(f"  ✓ {name}: {len(df):,} samples ({pos_count:,} positive, {pos_pct:.2f}%)")
        
        # === SAVE STAGE 2 DATA (Intensity Prediction: ≥5 strikes, ACTIVE CELLS ONLY) ===
        logger.info(f"\n3️⃣  Saving STAGE 2 data (Intensity: ≥5 strikes, active cells only)...")
        stage2_label_col = 'stage2_target_15min'  # Binary: ≥5 strikes or not
        
        for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            # FILTER: Only active cells (strike_count_30min > 0)
            active_mask = df['strike_count_30min'] > 0
            df_active = df[active_mask]
            
            if len(df_active) == 0:
                logger.warning(f"  ⚠️  {name}: No active cells found! Skipping...")
                continue
            
            features_file = output_dir / f'stage2_features_{name}.parquet'
            labels_file = output_dir / f'stage2_labels_{name}.parquet'
            
            df_active[feature_cols].to_parquet(features_file, index=False, compression='snappy')
            df_active[[stage2_label_col]].to_parquet(labels_file, index=False, compression='snappy')
            
            pos_count = df_active[stage2_label_col].sum()
            pos_pct = 100 * pos_count / len(df_active)
            reduction_pct = 100 * (1 - len(df_active) / len(df))
            logger.info(f"  ✓ {name}: {len(df_active):,} samples ({pos_count:,} positive, {pos_pct:.2f}%) "
                       f"[{reduction_pct:.1f}% reduction from filtering]")
        
        # Save metadata
        logger.info(f"\n💾 Saving metadata...")
        metadata_file = output_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"  ✓ Saved {metadata_file.name}")
        
        # Log file sizes
        logger.info(f"\n📊 File sizes:")
        total_size_mb = 0
        for file in sorted(output_dir.glob('*.parquet')):
            size_mb = file.stat().st_size / 1024 / 1024
            total_size_mb += size_mb
            logger.info(f"  {file.name}: {size_mb:.2f} MB")
        logger.info(f"\n  Total: {total_size_mb:.2f} MB")
        
        logger.info("="*80 + "\n")
    
    def run_pipeline(self, data_dir: Path = None, output_dir: Path = None) -> Dict:
        """
        Execute the complete data preparation pipeline.
        
        This is the main entry point that orchestrates all steps.
        
        Args:
            data_dir: Input directory with raw parquet files
            output_dir: Output directory for processed data
            
        Returns:
            Dictionary with pipeline metadata
            
        Pipeline Steps:
        1. Load raw data
        2. Validate data quality
        3. Extract features
        4. Create targets
        5. Split data temporally
        6. Save processed datasets
        """
        start_time = datetime.now()
        
        logger.info("\n" + "="*80)
        logger.info("LIGHTNING STRIKE DATA PREPARATION PIPELINE")
        logger.info("="*80)
        logger.info(f"Started: {start_time}")
        logger.info("="*80 + "\n")
        
        # Step 1: Load raw data
        df_raw = self.load_raw_data(data_dir)
        
        # Step 2: Validate data quality
        quality_metrics = self.validate_data_quality(df_raw)
        
        # Step 3: Extract features and targets
        df_features = self.extract_features_and_targets(df_raw)
        
        # Step 4: Create temporal split
        train_df, val_df, test_df = self.create_temporal_split(df_features)
        
        # Step 5: Prepare metadata
        metadata = {
            'pipeline_version': '1.0',
            'processing_date': start_time.isoformat(),
            'config': {
                'h3_resolution': self.config.h3_resolution,
                'lookback_windows': self.config.lookback_windows,
                'prediction_window': self.config.default_prediction_window,
                'strike_threshold': self.config.strike_threshold,
                'neighbor_rings': self.config.neighbor_rings,
            },
            'raw_data': {
                'total_strikes': len(df_raw),
                'time_range': [str(df_raw['time'].min()), str(df_raw['time'].max())],
            },
            'processed_data': {
                'total_samples': len(df_features),
                'unique_cells': int(df_features['h3_cell'].nunique()),
                'feature_count': len([col for col in df_features.columns 
                                     if col not in ['h3_cell', 'timestamp', 'target_count', 'target_binary']]),
                'train_samples': len(train_df),
                'val_samples': len(val_df),
                'test_samples': len(test_df),
            },
            'quality_metrics': quality_metrics,
        }
        
        # Step 6: Save processed data
        self.save_processed_data(train_df, val_df, test_df, metadata, output_dir)
        
        # Pipeline complete
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"Completed: {end_time}")
        logger.info("="*80 + "\n")
        
        metadata['processing_duration_seconds'] = duration
        
        return metadata


def main():
    """Command-line interface for data preparation."""
    parser = argparse.ArgumentParser(
        description='Prepare lightning strike data for ML training'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to grid_config.yml (optional)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Directory with raw parquet files (optional)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for processed data (optional)'
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = GridConfig(args.config)
    else:
        config = GridConfig()
    
    # Initialize pipeline
    pipeline = DataPreparation(config)
    
    # Run pipeline
    metadata = pipeline.run_pipeline(
        data_dir=Path(args.data_dir) if args.data_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None
    )
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Processed {metadata['raw_data']['total_strikes']:,} lightning strikes")
    print(f"✓ Generated {metadata['processed_data']['total_samples']:,} training samples")
    print(f"✓ Created {metadata['processed_data']['feature_count']} features")
    print(f"✓ Split into train/val/test sets")
    print(f"✓ Saved to: {config.processed_data_path}")
    print("="*80)
    print("\nReady for model training!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
