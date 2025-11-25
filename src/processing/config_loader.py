"""
Configuration loader for lightning prediction system.
Loads and validates grid, model, and system configurations.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List


class GridConfig:
    """
    Loads and provides access to grid configuration parameters.
    Usage:
        config = GridConfig()
        resolution = config.h3_resolution
        windows = config.lookback_windows
    """
    
    def __init__(self, config_path: str = None):
        """
        Load grid configuration from YAML file.
        
        Args:
            config_path: Path to grid_config.yml (default: configs/grid_config.yml)
        """
        if config_path is None:
            # Default to project root / configs / grid_config.yml
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "configs" / "grid_config.yml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self._validate()
    
    def _validate(self):
        """Basic validation of required config sections."""
        required_sections = ['h3', 'temporal', 'spatial', 'target', 'features']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
    
    # H3 Grid Properties
    @property
    def h3_resolution(self) -> int:
        """H3 hexagon resolution level (0-15)."""
        return self.config['h3']['resolution']
    
    @property
    def avg_area_km2(self) -> float:
        """Average cell area in km²."""
        return self.config['h3']['avg_area_km2']
    
    @property
    def avg_edge_length_km(self) -> float:
        """Average cell edge length in km."""
        return self.config['h3']['avg_edge_length_km']
    
    @property
    def max_neighbor_distance_km(self) -> float:
        """Maximum distance to consider for neighbors."""
        return self.config['h3']['max_neighbor_distance_km']
    
    # Temporal Properties
    @property
    def lookback_windows(self) -> List[int]:
        """Historical lookback windows in minutes."""
        return self.config['temporal']['lookback_windows']
    
    @property
    def prediction_horizon_min(self) -> int:
        """Minimum prediction horizon in minutes."""
        return self.config['temporal']['prediction_horizon_min']
    
    @property
    def prediction_horizon_max(self) -> int:
        """Maximum prediction horizon in minutes."""
        return self.config['temporal']['prediction_horizon_max']
    
    @property
    def default_prediction_window(self) -> int:
        """Default prediction window in minutes."""
        return self.config['temporal']['default_prediction_window']
    
    @property
    def aggregation_interval(self) -> int:
        """Temporal aggregation interval in minutes."""
        return self.config['temporal']['aggregation_interval']
    
    @property
    def max_data_age(self) -> int:
        """Maximum age of data to keep in memory (minutes)."""
        return self.config['temporal']['max_data_age']
    
    # Spatial Properties
    @property
    def neighbor_rings(self) -> List[int]:
        """Neighbor ring levels for spatial features."""
        return self.config['spatial']['neighbor_rings']
    
    @property
    def use_grid_distance(self) -> bool:
        """Whether to calculate grid distance between cells."""
        return self.config['spatial'].get('use_grid_distance', True)
    
    @property
    def use_geographic_distance(self) -> bool:
        """Whether to calculate geographic distance."""
        return self.config['spatial'].get('use_geographic_distance', True)
    
    @property
    def clustering_enabled(self) -> bool:
        """Whether clustering features are enabled."""
        return self.config['spatial']['clustering'].get('enabled', True)
    
    @property
    def min_strikes_for_cluster(self) -> int:
        """Minimum strikes to form a cluster."""
        return self.config['spatial']['clustering'].get('min_strikes_for_cluster', 3)
    
    @property
    def max_cluster_distance_km(self) -> float:
        """Maximum distance for cluster membership."""
        return self.config['spatial']['clustering'].get('max_cluster_distance_km', 15)
    
    # Target Properties
    @property
    def strike_threshold(self) -> int:
        """Primary strike threshold for binary classification."""
        return self.config['target']['strike_threshold']
    
    @property
    def alternative_thresholds(self) -> List[int]:
        """Alternative thresholds to experiment with."""
        return self.config['target'].get('alternative_thresholds', [])
    
    @property
    def min_lookback_strikes(self) -> int:
        """Minimum strikes in lookback to make a prediction."""
        return self.config['target'].get('min_lookback_strikes', 1)
    
    # Feature Properties
    @property
    def temporal_features(self) -> List[str]:
        """List of temporal features to calculate."""
        return self.config['features']['temporal_features']
    
    @property
    def spatial_features(self) -> List[str]:
        """List of spatial features to calculate."""
        return self.config['features']['spatial_features']
    
    @property
    def pattern_features(self) -> List[str]:
        """List of temporal pattern features."""
        return self.config['features']['pattern_features']
    
    # Data Quality
    @property
    def min_training_strikes(self) -> int:
        """Minimum strikes in training window for valid sample."""
        return self.config['quality'].get('min_training_strikes', 1)
    
    @property
    def min_observations_per_cell(self) -> int:
        """Minimum observations per cell."""
        return self.config['quality'].get('min_observations_per_cell', 5)
    
    @property
    def fill_missing_with_zero(self) -> bool:
        """Whether to fill missing data with zero."""
        return self.config['quality'].get('fill_missing_with_zero', True)
    
    @property
    def outlier_threshold_std(self) -> float:
        """Outlier threshold in standard deviations."""
        return self.config['quality'].get('outlier_threshold_std', 5.0)
    
    # Paths
    @property
    def raw_data_path(self) -> Path:
        """Path to raw data directory."""
        return Path(self.config['paths']['raw_data'])
    
    @property
    def processed_data_path(self) -> Path:
        """Path to processed data directory."""
        return Path(self.config['paths']['processed_data'])
    
    @property
    def grid_cache_path(self) -> Path:
        """Path to grid cache directory."""
        return Path(self.config['paths']['grid_cache'])
    
    # Data Split
    @property
    def split_method(self) -> str:
        """Data split method ('temporal' or 'random')."""
        return self.config['data_split']['method']
    
    @property
    def train_ratio(self) -> float:
        """Training data ratio."""
        return self.config['data_split']['train_ratio']
    
    @property
    def validation_ratio(self) -> float:
        """Validation data ratio from training set."""
        return self.config['data_split']['validation_ratio']
    
    # Performance
    @property
    def n_jobs(self) -> int:
        """Number of parallel jobs (-1 for all cores)."""
        return self.config['performance'].get('n_jobs', -1)
    
    @property
    def batch_size(self) -> int:
        """Batch size for processing."""
        return self.config['performance'].get('batch_size', 10000)
    
    @property
    def cache_features(self) -> bool:
        """Whether to cache computed features."""
        return self.config['performance'].get('cache_features', True)
    
    # Logging
    @property
    def log_level(self) -> str:
        """Logging level."""
        return self.config['logging'].get('level', 'INFO')
    
    def get_all_feature_names(self) -> List[str]:
        """Get a complete list of all feature names to be calculated."""
        features = []
        features.extend(self.temporal_features)
        features.extend(self.spatial_features)
        features.extend(self.pattern_features)
        return features
    
    def print_summary(self):
        """Print a summary of the configuration."""
        print("=" * 80)
        print("GRID CONFIGURATION SUMMARY")
        print("=" * 80)
        print(f"\nH3 Grid:")
        print(f"  Resolution: {self.h3_resolution}")
        print(f"  Avg Area: {self.avg_area_km2} km²")
        print(f"  Avg Edge Length: {self.avg_edge_length_km} km")
        
        print(f"\nTemporal Windows:")
        print(f"  Lookback: {self.lookback_windows} minutes")
        print(f"  Prediction: {self.default_prediction_window} minutes")
        
        print(f"\nSpatial:")
        print(f"  Neighbor Rings: {self.neighbor_rings}")
        print(f"  Clustering: {'Enabled' if self.clustering_enabled else 'Disabled'}")
        
        print(f"\nTarget:")
        print(f"  Strike Threshold: {self.strike_threshold}")
        print(f"  Alternative Thresholds: {self.alternative_thresholds}")
        
        print(f"\nFeatures ({len(self.get_all_feature_names())} total):")
        print(f"  Temporal: {len(self.temporal_features)}")
        print(f"  Spatial: {len(self.spatial_features)}")
        print(f"  Pattern: {len(self.pattern_features)}")
        
        print(f"\nData Split:")
        print(f"  Method: {self.split_method}")
        print(f"  Train Ratio: {self.train_ratio}")
        print(f"  Validation Ratio: {self.validation_ratio}")
        
        print("=" * 80)


if __name__ == "__main__":
    # Test the config loader
    config = GridConfig()
    config.print_summary()
    
    print("\nAll feature names:")
    for feature in config.get_all_feature_names():
        print(f"  - {feature}")
