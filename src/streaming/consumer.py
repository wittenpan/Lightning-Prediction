"""
Kafka Consumer: Real-time Lightning Prediction Pipeline

Consumes strikes from Kafka → Computes features → Makes predictions → Publishes results.

Pipeline:
1. Consume strike from Kafka
2. Map to H3 cell + neighbors
3. Update Redis cache (strike history)
4. Compute features (with Redis caching)
5. Run two-stage prediction
6. Publish prediction to output topic
7. Update metrics
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path

import h3
import numpy as np
import xgboost as xgb
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

from redis_cache import LightningCache

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LightningPredictionConsumer:
    """
    Real-time lightning prediction consumer.

    Architecture:
    - Kafka Consumer: Reads strikes from 'lightning-strikes' topic
    - Redis Cache: Stores strike history + features + predictions
    - XGBoost Models: Two-stage cascade (Stage 1: activity, Stage 2: intensity)
    - Kafka Producer: Publishes predictions to 'lightning-predictions' topic
    """

    def __init__(
        self,
        kafka_bootstrap_servers: str = 'localhost:9092',
        input_topic: str = 'lightning-strikes',
        output_topic: str = 'lightning-predictions',
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        model_dir: str = 'data/models',
        h3_resolution: int = 7,
        stage1_threshold: float = 0.55,
        stage2_threshold: float = 0.78,
        feature_windows: Optional[List[int]] = None
    ):
        """
        Initialize prediction consumer.

        Args:
            kafka_bootstrap_servers: Kafka brokers
            input_topic: Input topic (strikes)
            output_topic: Output topic (predictions)
            redis_host: Redis host
            redis_port: Redis port
            model_dir: Directory with trained models
            h3_resolution: H3 grid resolution
            stage1_threshold: Stage 1 classification threshold
            stage2_threshold: Stage 2 classification threshold
            feature_windows: Time windows for features (minutes)
        """
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.h3_resolution = h3_resolution
        self.stage1_threshold = stage1_threshold
        self.stage2_threshold = stage2_threshold

        # Feature time windows (minutes → microseconds)
        if feature_windows is None:
            feature_windows = [5, 15, 30, 60, 90]
        self.feature_windows = {
            f'{w}min': w * 60 * 1_000_000 for w in feature_windows
        }

        # Initialize Redis cache
        self.cache = LightningCache(host=redis_host, port=redis_port)

        # Load models
        model_path = Path(model_dir)
        self.model_stage1 = xgb.XGBClassifier()
        self.model_stage1.load_model(model_path / 'stage1_model_15min.ubj')

        self.model_stage2 = xgb.XGBClassifier()
        self.model_stage2.load_model(model_path / 'stage2_model_15min.ubj')

        logger.info(f"✓ Loaded models from {model_dir}")

        # Initialize Kafka consumer
        self.consumer = KafkaConsumer(
            input_topic,
            bootstrap_servers=kafka_bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',  # Start from latest (real-time)
            enable_auto_commit=True,
            group_id='lightning-prediction-group',
            max_poll_records=500  # Process in batches
        )

        # Initialize Kafka producer (for predictions)
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            compression_type='gzip'
        )

        # Metrics
        self.strikes_processed = 0
        self.predictions_made = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0
        self.start_time = datetime.now()
        self.last_metric_log = datetime.now()

        logger.info(f"Consumer initialized: {input_topic} → {output_topic}")

    def lat_lon_to_h3(self, lat: float, lon: float) -> str:
        """Convert lat/lon to H3 cell."""
        return h3.latlng_to_cell(lat, lon, self.h3_resolution)

    def get_neighbors(self, h3_cell: str) -> Tuple[List[str], List[str]]:
        """
        Get neighbor cells (with caching).

        Returns:
            (ring1_neighbors, ring2_neighbors)
        """
        # Check cache first
        ring1_cached = self.cache.get_neighbors(h3_cell, ring=1)
        ring2_cached = self.cache.get_neighbors(h3_cell, ring=2)

        if ring1_cached and ring2_cached:
            self.cache_hits += 1
            return list(ring1_cached), list(ring2_cached)

        # Compute if not cached
        self.cache_misses += 1
        ring1 = list(h3.grid_disk(h3_cell, 1) - {h3_cell})
        ring2 = list(h3.grid_disk(h3_cell, 2) - set(ring1) - {h3_cell})

        # Cache for next time
        self.cache.cache_neighbors(h3_cell, set(ring1), set(ring2))

        return ring1, ring2

    def compute_features(
        self,
        h3_cell: str,
        current_time_us: int
    ) -> Optional[Dict[str, float]]:
        """
        Compute features for cell at given time.

        Uses Redis cache for strike history.

        Features (31 total):
        - Temporal: strike counts/rates/densities for 5 time windows
        - Spatial: neighbor activity (ring1, ring2)
        - Temporal patterns: acceleration, time since last strike
        - Time features: hour, day of week, is weekend

        Args:
            h3_cell: H3 cell ID
            current_time_us: Current timestamp (microseconds)

        Returns:
            Feature dict or None if error
        """
        try:
            features = {}

            # === TEMPORAL FEATURES ===
            for window_name, window_us in self.feature_windows.items():
                start_time = current_time_us - window_us
                count = self.cache.get_strike_count(h3_cell, start_time, current_time_us)

                # Strike count
                features[f'strike_count_{window_name}'] = count

                # Strike rate (strikes per minute)
                window_minutes = window_us / (60 * 1_000_000)
                features[f'strike_rate_{window_name}'] = count / window_minutes if window_minutes > 0 else 0.0

                # Strike density (normalized by cell area ~5.16 km²)
                cell_area_km2 = 5.16
                features[f'strike_density_{window_name}'] = count / cell_area_km2

            # Strike acceleration (change in rate)
            rate_5min = features.get('strike_rate_5min', 0)
            rate_15min = features.get('strike_rate_15min', 0)
            features['strike_acceleration'] = rate_5min - rate_15min

            # Time since last strike
            recent_strikes = self.cache.get_strikes(
                h3_cell,
                current_time_us - self.feature_windows['5min'],
                current_time_us
            )
            if recent_strikes:
                last_strike = max(recent_strikes)
                time_since_us = current_time_us - last_strike
                features['time_since_last_strike'] = time_since_us / 1_000_000  # Convert to seconds
            else:
                features['time_since_last_strike'] = 999999.0  # Large value = no recent strikes

            # === SPATIAL FEATURES (NEIGHBORS) ===
            ring1, ring2 = self.get_neighbors(h3_cell)

            for ring_num, neighbors in [(1, ring1), (2, ring2)]:
                # Get strike counts for all neighbors (batch query)
                neighbor_counts = []
                for neighbor in neighbors:
                    count = self.cache.get_strike_count(
                        neighbor,
                        current_time_us - self.feature_windows['15min'],
                        current_time_us
                    )
                    neighbor_counts.append(count)

                # Aggregate neighbor features
                features[f'neighbor_count_ring{ring_num}'] = sum(neighbor_counts)
                features[f'neighbor_density_ring{ring_num}'] = np.mean(neighbor_counts) if neighbor_counts else 0.0
                features[f'max_neighbor_ring{ring_num}'] = max(neighbor_counts) if neighbor_counts else 0
                features[f'spatial_gradient_ring{ring_num}'] = np.std(neighbor_counts) if len(neighbor_counts) > 1 else 0.0

                # Cluster size (number of active neighbors)
                features[f'cluster_size_ring{ring_num}'] = sum(1 for c in neighbor_counts if c > 0)

            # === TIME FEATURES ===
            dt = datetime.utcfromtimestamp(current_time_us / 1_000_000)
            features['hour_of_day'] = dt.hour
            features['time_of_day_cat'] = 0 if 6 <= dt.hour < 18 else 1  # 0=day, 1=night
            features['day_of_week'] = dt.weekday()
            features['is_weekend'] = 1 if dt.weekday() >= 5 else 0

            return features

        except Exception as e:
            logger.error(f"Feature computation error for {h3_cell}: {e}")
            self.errors += 1
            return None

    def predict(
        self,
        features: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Two-stage prediction.

        Stage 1: Activity detection (≥1 strike in next 15min)
        Stage 2: Intensity prediction (≥3 strikes in next 15min) [only if Stage 1 predicts activity]

        Args:
            features: Feature dict

        Returns:
            Prediction result with both stages
        """
        # Convert features to numpy array (ensure correct order)
        feature_names = sorted(features.keys())
        X = np.array([[features[f] for f in feature_names]])

        # Stage 1: Activity Detection
        stage1_proba = self.model_stage1.predict_proba(X)[0, 1]
        stage1_pred = 1 if stage1_proba >= self.stage1_threshold else 0

        # Stage 2: Intensity Prediction (only if Stage 1 predicts activity)
        if stage1_pred == 1:
            stage2_proba = self.model_stage2.predict_proba(X)[0, 1]
            stage2_pred = 1 if stage2_proba >= self.stage2_threshold else 0
        else:
            stage2_proba = 0.0
            stage2_pred = 0

        # Combined prediction
        combined_pred = 1 if (stage1_pred == 1 and stage2_pred == 1) else 0

        return {
            'stage1': {
                'prediction': stage1_pred,
                'probability': float(stage1_proba)
            },
            'stage2': {
                'prediction': stage2_pred,
                'probability': float(stage2_proba)
            },
            'combined': {
                'prediction': combined_pred,
                'probability': float(stage2_proba) if combined_pred == 1 else 0.0
            }
        }

    def process_strike(self, strike: Dict[str, Any]):
        """
        Process single strike: update cache → compute features → predict → publish.

        Args:
            strike: Strike data from Kafka
        """
        try:
            # Extract strike data
            lat = strike['latitude']
            lon = strike['longitude']
            timestamp_us = strike['timestamp']

            # Map to H3 cell
            h3_cell = self.lat_lon_to_h3(lat, lon)

            # Update Redis cache
            self.cache.add_strike(h3_cell, timestamp_us)

            # Compute features
            features = self.compute_features(h3_cell, timestamp_us)
            if not features:
                return  # Skip if feature computation failed

            # Make prediction
            prediction_result = self.predict(features)

            # Cache prediction
            current_time_s = int(timestamp_us / 1_000_000)
            self.cache.cache_prediction(
                h3_cell, stage=1,
                prediction=prediction_result['stage1']['prediction'],
                probability=prediction_result['stage1']['probability'],
                timestamp_s=current_time_s
            )
            self.cache.cache_prediction(
                h3_cell, stage=2,
                prediction=prediction_result['stage2']['prediction'],
                probability=prediction_result['stage2']['probability'],
                timestamp_s=current_time_s
            )

            # Publish prediction
            output_message = {
                'h3_cell': h3_cell,
                'latitude': lat,
                'longitude': lon,
                'timestamp': timestamp_us,
                'prediction': prediction_result,
                'predicted_at': datetime.utcnow().isoformat()
            }

            self.producer.send(
                self.output_topic,
                key=h3_cell.encode('utf-8'),
                value=output_message
            )

            self.strikes_processed += 1
            self.predictions_made += 1

        except Exception as e:
            logger.error(f"Strike processing error: {e}")
            self.errors += 1

    def run(self):
        """Main consumer loop."""
        logger.info("Starting Lightning Prediction Consumer...")
        logger.info(f"  Input: {self.input_topic}")
        logger.info(f"  Output: {self.output_topic}")
        logger.info(f"  Stage 1 threshold: {self.stage1_threshold}")
        logger.info(f"  Stage 2 threshold: {self.stage2_threshold}")

        try:
            for message in self.consumer:
                strike = message.value

                # Process strike
                self.process_strike(strike)

                # Log metrics every 100 strikes
                if self.strikes_processed % 100 == 0:
                    self.log_metrics()

        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
        finally:
            self.consumer.close()
            self.producer.close()
            logger.info("Consumer stopped.")

    def log_metrics(self):
        """Log performance metrics."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.strikes_processed / elapsed if elapsed > 0 else 0

        cache_total = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / cache_total if cache_total > 0 else 0

        logger.info("="*70)
        logger.info("CONSUMER METRICS")
        logger.info("="*70)
        logger.info(f"  Strikes processed: {self.strikes_processed:,}")
        logger.info(f"  Predictions made: {self.predictions_made:,}")
        logger.info(f"  Rate: {rate:.1f} strikes/sec")
        logger.info(f"  Cache hit rate: {cache_hit_rate*100:.1f}%")
        logger.info(f"  Errors: {self.errors}")
        logger.info(f"  Uptime: {elapsed:.0f}s")
        logger.info("="*70)


def main():
    """Run consumer."""
    consumer = LightningPredictionConsumer(
        kafka_bootstrap_servers='localhost:9092',
        input_topic='lightning-strikes',
        output_topic='lightning-predictions',
        redis_host='localhost',
        redis_port=6379,
        model_dir='data/models',
        stage1_threshold=0.55,
        stage2_threshold=0.78
    )

    consumer.run()


if __name__ == "__main__":
    main()
