"""Reproducible Kafka-to-prediction latency benchmark."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from kafka import KafkaConsumer, KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

try:
    from .schema import make_event
    from .settings import Settings
except ImportError:
    from schema import make_event
    from settings import Settings


def percentile(values: List[float], percent: float) -> float:
    ordered = sorted(values)
    position = min(len(ordered) - 1, int((len(ordered) - 1) * percent))
    return ordered[position]


def run_benchmark(events: int, rate: float, timeout_s: float) -> Dict[str, Any]:
    settings = Settings.from_env()
    kafka_options = settings.kafka_kwargs()
    run_id = uuid.uuid4().hex[:12]
    admin = KafkaAdminClient(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        **kafka_options,
    )
    try:
        admin.create_topics(
            [
                NewTopic(settings.input_topic, num_partitions=1, replication_factor=1),
                NewTopic(settings.output_topic, num_partitions=1, replication_factor=1),
            ],
            validate_only=False,
        )
    except TopicAlreadyExistsError:
        pass
    finally:
        admin.close()
    output_consumer = KafkaConsumer(
        settings.output_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_deserializer=lambda message: json.loads(message.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=False,
        group_id=f"lightning-benchmark-{run_id}",
        consumer_timeout_ms=500,
        **kafka_options,
    )
    # Force partition assignment before traffic is published.
    output_consumer.poll(timeout_ms=1000)
    producer = KafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        compression_type="gzip",
        linger_ms=5,
        acks="all",
        **kafka_options,
    )

    sent: Dict[str, int] = {}
    processing_latencies: List[float] = []
    round_trip_latencies: List[float] = []
    stage2_calls = 0

    def collect(timeout_ms: int) -> None:
        nonlocal stage2_calls
        records = output_consumer.poll(timeout_ms=timeout_ms, max_records=events)
        received_at_ns = time.time_ns()
        for batch in records.values():
            for record in batch:
                prediction = record.value
                event_id = prediction.get("event_id")
                if event_id not in sent:
                    continue
                processing_latencies.append(
                    float(prediction["latency"]["ingestion_to_prediction_ms"])
                )
                round_trip_latencies.append((received_at_ns - sent.pop(event_id)) / 1_000_000)
                stage2_calls += int(prediction["prediction"]["stage2"]["executed"])

    interval = 1.0 / rate if rate else 0.0
    for index in range(events):
        event = make_event(
            {
                "lat": 28.5 + random.uniform(-0.25, 0.25),
                "lon": -81.4 + random.uniform(-0.25, 0.25),
                "time": time.time_ns(),
                "mds": f"{run_id}-{index}",
            },
            source=f"benchmark:{run_id}",
        )
        sent[event["event_id"]] = event["ingested_at_ns"]
        producer.send(settings.input_topic, key=event["event_id"].encode(), value=event)
        collect(0)
        if interval:
            time.sleep(interval)
    producer.flush()

    deadline = time.monotonic() + timeout_s
    while len(processing_latencies) < events and time.monotonic() < deadline:
        collect(500)

    producer.close()
    output_consumer.close()
    if not processing_latencies:
        raise RuntimeError("no predictions received; is the prediction service running?")

    result = {
        "run_id": run_id,
        "requested_events": events,
        "received_events": len(processing_latencies),
        "input_rate_per_second": rate,
        "ingestion_to_prediction_ms": {
            "p50": round(percentile(processing_latencies, 0.50), 3),
            "p95": round(percentile(processing_latencies, 0.95), 3),
            "p99": round(percentile(processing_latencies, 0.99), 3),
            "mean": round(statistics.fmean(processing_latencies), 3),
        },
        "kafka_round_trip_ms": {
            "p50": round(percentile(round_trip_latencies, 0.50), 3),
            "p95": round(percentile(round_trip_latencies, 0.95), 3),
            "p99": round(percentile(round_trip_latencies, 0.99), 3),
        },
        "stage2_invocations": stage2_calls,
        "stage2_skip_rate": round(1 - stage2_calls / len(processing_latencies), 6),
        "under_100ms_p95": percentile(processing_latencies, 0.95) < 100.0,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output_path = Path(os.getenv("BENCHMARK_OUTPUT", "benchmark-results.json"))
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the complete local stream")
    parser.add_argument("--events", type=int, default=1000)
    parser.add_argument("--rate", type=float, default=100.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    result = run_benchmark(args.events, args.rate, args.timeout)
    print(json.dumps(result, indent=2))
    if result["received_events"] != result["requested_events"] or not result["under_100ms_p95"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
