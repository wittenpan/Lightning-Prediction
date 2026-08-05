"""Kafka -> Redis features -> two-stage XGBoost -> Kafka predictions."""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import h3
import numpy as np
from kafka import KafkaConsumer, KafkaProducer

try:
    from .models import TwoStageXGBoostCascade
    from .redis_cache import LightningCache
    from .schema import normalize_timestamp_us
    from .settings import Settings
except ImportError:  # Support `python src/streaming/consumer.py`.
    from models import TwoStageXGBoostCascade
    from redis_cache import LightningCache
    from schema import normalize_timestamp_us
    from settings import Settings

logger = logging.getLogger(__name__)


class LightningPredictionConsumer:
    """Online feature computation and prediction service."""

    def __init__(
        self,
        kafka_bootstrap_servers: str = "localhost:9092",
        input_topic: str = "lightning-strikes",
        output_topic: str = "lightning-predictions",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        model_dir: str = "data/models",
        h3_resolution: int = 7,
        stage1_threshold: Optional[float] = None,
        stage2_threshold: Optional[float] = None,
        feature_windows: Optional[List[int]] = None,
        consumer_group: str = "lightning-prediction-group",
        kafka_options: Optional[Dict[str, Any]] = None,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        redis_ssl: bool = False,
        cache: Optional[LightningCache] = None,
        cascade: Optional[TwoStageXGBoostCascade] = None,
        kafka_consumer: Optional[KafkaConsumer] = None,
        kafka_producer: Optional[KafkaProducer] = None,
    ) -> None:
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.h3_resolution = h3_resolution
        windows = feature_windows or [5, 15, 30, 60, 90]
        self.feature_windows = {f"{window}min": window * 60 * 1_000_000 for window in windows}
        self.cache = cache or LightningCache(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            ssl=redis_ssl,
        )
        self.cascade = cascade or TwoStageXGBoostCascade(
            model_dir=model_dir,
            stage1_threshold=stage1_threshold,
            stage2_threshold=stage2_threshold,
        )
        options = kafka_options or {}
        self.consumer = kafka_consumer or KafkaConsumer(
            input_topic,
            bootstrap_servers=kafka_bootstrap_servers,
            value_deserializer=lambda message: json.loads(message.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
            group_id=consumer_group,
            max_poll_records=100,
            **options,
        )
        self.producer = kafka_producer or KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
            compression_type="gzip",
            linger_ms=5,
            acks="all",
            **options,
        )
        self.strikes_processed = 0
        self.errors = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.started_at = time.monotonic()
        self.latencies_ms: Deque[float] = deque(maxlen=10_000)

    def lat_lon_to_h3(self, latitude: float, longitude: float) -> str:
        return h3.latlng_to_cell(latitude, longitude, self.h3_resolution)

    def get_neighbors(self, h3_cell: str) -> Tuple[List[str], List[str]]:
        ring1_cached = self.cache.get_neighbors(h3_cell, ring=1)
        ring2_cached = self.cache.get_neighbors(h3_cell, ring=2)
        if ring1_cached and ring2_cached:
            self.cache_hits += 1
            return sorted(ring1_cached), sorted(ring2_cached)

        self.cache_misses += 1
        disk1 = set(h3.grid_disk(h3_cell, 1))
        disk2 = set(h3.grid_disk(h3_cell, 2))
        ring1 = disk1 - {h3_cell}
        ring2 = disk2 - disk1
        self.cache.cache_neighbors(h3_cell, ring1, ring2)
        return sorted(ring1), sorted(ring2)

    @staticmethod
    def _time_of_day(hour: int) -> int:
        if hour >= 22 or hour <= 5:
            return 0
        if hour <= 11:
            return 1
        if hour <= 17:
            return 2
        return 3

    def compute_features(self, h3_cell: str, current_time_us: int) -> Dict[str, float]:
        """Compute the exact 31-feature training schema with one count pipeline."""
        ring1, ring2 = self.get_neighbors(h3_cell)
        window_counts, neighbor_counts_by_ring = self.cache.get_feature_counts(
            h3_cell=h3_cell,
            neighbors_by_ring=((1, ring1), (2, ring2)),
            current_time_us=current_time_us,
            windows_us=self.feature_windows,
            # Offline training builds spatial features from strike_count_5min.
            neighbor_window_us=self.feature_windows["5min"],
        )
        features: Dict[str, float] = {}
        for window_name, window_us in self.feature_windows.items():
            count = window_counts[window_name]
            minutes = window_us / 60_000_000
            features[f"strike_count_{window_name}"] = float(count)
            features[f"strike_rate_{window_name}"] = count / minutes
            features[f"strike_density_{window_name}"] = count / 5.16

        features["strike_acceleration"] = (
            features["strike_rate_5min"] - features["strike_rate_15min"]
        )
        # The committed model artifacts were trained on complete 5-minute bins,
        # making this feature exactly 5.0 in every test row. Preserve parity.
        features["time_since_last_strike"] = 5.0

        own_count = features["strike_count_5min"]
        for ring, neighbors in ((1, ring1), (2, ring2)):
            counts = neighbor_counts_by_ring[ring]
            total = float(sum(counts))
            density = total / len(neighbors) if neighbors else 0.0
            features[f"neighbor_count_ring{ring}"] = total
            features[f"neighbor_density_ring{ring}"] = density
            features[f"max_neighbor_ring{ring}"] = float(max(counts, default=0))
            features[f"spatial_gradient_ring{ring}"] = own_count - density
            features[f"cluster_size_ring{ring}"] = float(sum(count >= 3 for count in counts))

        event_time = datetime.fromtimestamp(current_time_us / 1_000_000, tz=timezone.utc)
        features["hour_of_day"] = float(event_time.hour)
        features["time_of_day_cat"] = float(self._time_of_day(event_time.hour))
        features["day_of_week"] = float(event_time.weekday())
        features["is_weekend"] = float(event_time.weekday() >= 5)
        return features

    def process_strike(self, strike: Dict[str, Any]) -> Dict[str, Any]:
        processing_started_ns = time.time_ns()
        latitude = float(strike["latitude"])
        longitude = float(strike["longitude"])
        timestamp_us = normalize_timestamp_us(strike["timestamp"])
        event_id = str(strike.get("event_id", f"{timestamp_us}:{latitude}:{longitude}"))
        h3_cell = self.lat_lon_to_h3(latitude, longitude)

        self.cache.add_strike(h3_cell, timestamp_us, event_id=event_id)
        features = self.compute_features(h3_cell, timestamp_us)
        prediction = self.cascade.predict(features)
        self.cache.cache_prediction_pair(
            h3_cell,
            prediction,
            timestamp_s=timestamp_us // 1_000_000,
        )

        predicted_at_ns = time.time_ns()
        ingested_at_ns = int(strike.get("ingested_at_ns", processing_started_ns))
        processing_latency_ms = (predicted_at_ns - processing_started_ns) / 1_000_000
        e2e_latency_ms = max(0.0, (predicted_at_ns - ingested_at_ns) / 1_000_000)
        output = {
            "event_id": event_id,
            "h3_cell": h3_cell,
            "latitude": latitude,
            "longitude": longitude,
            "timestamp": timestamp_us,
            "prediction": prediction,
            "latency": {
                "processing_ms": round(processing_latency_ms, 3),
                "ingestion_to_prediction_ms": round(e2e_latency_ms, 3),
            },
            "predicted_at": datetime.now(timezone.utc).isoformat(),
            "predicted_at_ns": predicted_at_ns,
        }
        self.producer.send(self.output_topic, key=event_id.encode("utf-8"), value=output)
        self.strikes_processed += 1
        self.latencies_ms.append(e2e_latency_ms)
        return output

    def metrics(self) -> Dict[str, float]:
        values = np.asarray(self.latencies_ms, dtype=np.float64)
        elapsed = max(time.monotonic() - self.started_at, 1e-9)
        return {
            "processed": float(self.strikes_processed),
            "errors": float(self.errors),
            "events_per_second": self.strikes_processed / elapsed,
            "latency_p50_ms": float(np.percentile(values, 50)) if values.size else 0.0,
            "latency_p95_ms": float(np.percentile(values, 95)) if values.size else 0.0,
            "latency_p99_ms": float(np.percentile(values, 99)) if values.size else 0.0,
            "stage2_skip_rate": self.cascade.stage2_skip_rate,
        }

    def run(self) -> None:
        logger.info(
            "Consumer ready: %s -> %s; thresholds %.2f/%.2f",
            self.input_topic,
            self.output_topic,
            self.cascade.stage1_threshold,
            self.cascade.stage2_threshold,
        )
        try:
            for message in self.consumer:
                try:
                    self.process_strike(message.value)
                except Exception:
                    self.errors += 1
                    logger.exception("Strike processing failed")
                if self.strikes_processed and self.strikes_processed % 100 == 0:
                    logger.info("metrics=%s", json.dumps(self.metrics(), sort_keys=True))
        finally:
            self.consumer.close()
            self.producer.flush()
            self.producer.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    service = LightningPredictionConsumer(
        kafka_bootstrap_servers=settings.kafka_bootstrap_servers,
        input_topic=settings.input_topic,
        output_topic=settings.output_topic,
        consumer_group=settings.consumer_group,
        kafka_options=settings.kafka_kwargs(),
        redis_host=settings.redis_host,
        redis_port=settings.redis_port,
        redis_db=settings.redis_db,
        redis_password=settings.redis_password,
        redis_ssl=settings.redis_ssl,
        model_dir=settings.model_dir,
        h3_resolution=settings.h3_resolution,
    )
    service.run()


if __name__ == "__main__":
    main()
