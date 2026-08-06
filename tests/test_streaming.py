import json
import time
from pathlib import Path

import fakeredis
import pytest

from src.streaming.consumer import LightningPredictionConsumer
from src.streaming.candidate_producer import CandidateCellProducer
from src.streaming.models import TwoStageXGBoostCascade
from src.streaming.producer import LightningStrikeProducer
from src.streaming.redis_cache import LightningCache
from src.streaming.schema import decode_blitzortung_message, make_event, normalize_timestamp_us
from src.streaming.settings import Settings
from src.dashboard.api import build_dashboard_state


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1_762_749_762, 1_762_749_762_000_000),
        (1_762_749_762_123, 1_762_749_762_123_000),
        (1_762_749_762_123_456, 1_762_749_762_123_456),
        (1_762_749_762_123_456_789, 1_762_749_762_123_456),
    ],
)
def test_timestamp_normalization(value, expected):
    assert normalize_timestamp_us(value) == expected


def test_wire_event_has_stable_microsecond_schema():
    event = make_event({"lat": 28.5, "lon": -81.4, "time": 1_762_749_762_123_456_789})
    assert event["timestamp"] == 1_762_749_762_123_456
    assert event["timestamp_unit"] == "microseconds"
    assert len(event["event_id"]) == 24
    assert event["ingested_at_ns"] > 0


def test_live_style_lzw_message_with_json_prefix_decodes():
    raw = '{"time":1762749762911321600,"lat":28.5,"lon":-81.4}'
    dictionary = {chr(i): i for i in range(256)}
    dictionary_size = 256
    current = ""
    codes = []
    for char in raw:
        candidate = current + char
        if candidate in dictionary:
            current = candidate
        else:
            codes.append(dictionary[current])
            dictionary[candidate] = dictionary_size
            dictionary_size += 1
            current = char
    if current:
        codes.append(dictionary[current])
    compressed = "".join(chr(code) for code in codes)
    assert compressed.startswith("{")
    assert decode_blitzortung_message(compressed) == json.loads(raw)


def test_producer_parses_plain_fixture_without_real_kafka():
    producer = LightningStrikeProducer.__new__(LightningStrikeProducer)
    producer.errors = 0
    strike = producer.parse_strike(json.dumps({"lat": 28.5, "lon": -81.4, "time": 1_762_749_762_123_456_789}))
    assert strike["latitude"] == 28.5
    assert strike["timestamp"] == 1_762_749_762_123_456
    assert producer.errors == 0


def test_redis_keeps_simultaneous_strikes():
    cache = LightningCache.__new__(LightningCache)
    cache.redis = fakeredis.FakeRedis(decode_responses=False)
    cache.STRIKE_TTL = 7200
    timestamp = 1_762_749_762_123_456
    cache.add_strike("cell", timestamp, "event-a")
    cache.add_strike("cell", timestamp, "event-b")
    assert cache.get_strike_count("cell", timestamp, timestamp) == 2
    assert cache.get_strikes("cell", timestamp, timestamp) == [timestamp, timestamp]


class FeatureCache:
    def get_neighbors(self, _cell, ring):
        return {f"ring{ring}-a", f"ring{ring}-b"}

    def get_feature_counts(self, **_kwargs):
        return (
            {"5min": 6, "15min": 9, "30min": 12, "60min": 15, "90min": 18},
            {1: [3, 1], 2: [0, 6]},
        )


def test_online_features_match_training_definitions():
    service = LightningPredictionConsumer.__new__(LightningPredictionConsumer)
    service.h3_resolution = 7
    service.feature_windows = {
        f"{window}min": window * 60 * 1_000_000 for window in (5, 15, 30, 60, 90)
    }
    service.cache = FeatureCache()
    service.cache_hits = 0
    service.cache_misses = 0
    features = service.compute_features("cell", 1_762_749_762_123_456)
    metadata = json.loads(Path("data/models/stage1_metadata_15min.json").read_text())
    assert set(features) == set(metadata["feature_names"])
    assert features["neighbor_count_ring1"] == 4
    assert features["neighbor_density_ring1"] == 2
    assert features["spatial_gradient_ring1"] == 4
    assert features["cluster_size_ring1"] == 1
    assert features["time_since_last_strike"] == 5.0
    assert features["time_of_day_cat"] in {0.0, 1.0, 2.0, 3.0}


def test_committed_cascade_loads_tuned_thresholds_and_gates():
    cascade = TwoStageXGBoostCascade()
    features = {name: 0.0 for name in cascade.feature_names}
    features["time_since_last_strike"] = 5.0
    result = cascade.predict(features)
    assert cascade.stage1_threshold == pytest.approx(0.71)
    assert cascade.stage2_threshold == pytest.approx(0.78)
    assert result["stage2"]["executed"] is False
    assert cascade.stage2_skip_rate == 1.0


class CapturingProducer:
    def __init__(self):
        self.messages = []

    def send(self, topic, key, value):
        self.messages.append((topic, key, value))

    def flush(self):
        return None

    def close(self):
        return None


def test_candidate_producer_scores_neighbors_without_adding_strikes():
    client = fakeredis.FakeRedis(decode_responses=False)
    seed = "8744a1d92ffffff"
    now_us = int(time.time() * 1_000_000)
    client.zadd(f"strikes:{seed}", {"strike-a": now_us})
    producer = CapturingProducer()
    service = CandidateCellProducer(
        Settings(),
        seed_limit=1,
        ring=12,
        redis_client=client,
        producer=producer,
    )

    count = service.publish_cycle()

    assert count == 72
    assert len(producer.messages) == 72
    assert all(message[2]["event_type"] == "candidate" for message in producer.messages)
    assert all(message[2]["h3_cell"] for message in producer.messages)
    assert client.zcard(f"strikes:{seed}") == 1


def test_dashboard_state_summarizes_h3_predictions():
    client = fakeredis.FakeRedis(decode_responses=False)
    cell = "8744a1d92ffffff"
    now_s = int(time.time())
    stage1 = {
        "prediction": 1,
        "probability": 0.82,
        "timestamp": now_s,
        "metadata": {"e2e_ms": 0, "ingestion_to_prediction_ms": 12.5},
    }
    stage2 = {"prediction": 0, "probability": 0.33, "timestamp": now_s}
    client.setex(f"prediction:{cell}:stage1", 300, json.dumps(stage1))
    client.setex(f"prediction:{cell}:stage2", 300, json.dumps(stage2))
    now_us = int(time.time() * 1_000_000)
    client.zadd(f"strikes:{cell}", {"strike-a": now_us - 30_000_000})

    snapshot = build_dashboard_state(client)

    assert snapshot["summary"]["active_cells"] == 1
    assert snapshot["summary"]["stage1_positive_cells"] == 1
    assert snapshot["summary"]["stage2_skip_rate"] == 0.0
    assert snapshot["summary"]["strikes_by_window"] == {"1m": 1, "5m": 1, "30m": 1}
    assert snapshot["summary"]["e2e_p95_ms"] == 12.5
    assert snapshot["cells"][0]["h3_cell"] == cell
    assert snapshot["cells"][0]["strike_counts"]["30m"] == 1


def test_dashboard_gate_metrics_include_candidates_outside_display_limit():
    client = fakeredis.FakeRedis(decode_responses=False)
    now_s = int(time.time())
    strike_cell = "8744a1d92ffffff"
    candidate_cell = "8744a1d93ffffff"
    for cell, prediction, event_type, score in (
        (strike_cell, 1, "strike", now_s),
        (candidate_cell, 0, "candidate", now_s - 1),
    ):
        stage1 = {
            "prediction": prediction,
            "probability": 0.8 if prediction else 0.08,
            "timestamp": score,
            "metadata": {"event_type": event_type},
        }
        client.setex(f"prediction:{cell}:stage1", 300, json.dumps(stage1))
        client.setex(f"prediction:{cell}:stage2", 300, json.dumps({"prediction": 0}))
        client.zadd("prediction:recent", {cell: score})

    snapshot = build_dashboard_state(client, limit=1)

    assert snapshot["summary"]["active_cells"] == 1
    assert snapshot["summary"]["candidate_cells"] == 1
    assert snapshot["summary"]["stage2_skip_rate"] == 1.0
    assert snapshot["cells"][0]["h3_cell"] == strike_cell


def test_dashboard_reserves_map_space_for_candidate_cells():
    client = fakeredis.FakeRedis(decode_responses=False)
    now_s = int(time.time())
    cells = [
        "8744a1d92ffffff",
        "8744a1d93ffffff",
        "8744a1d94ffffff",
        "8744a1d95ffffff",
        "8744a1d96ffffff",
    ]
    for index, cell in enumerate(cells):
        is_candidate = index == len(cells) - 1
        stage1 = {
            "prediction": 0 if is_candidate else 1,
            "probability": 0.08 if is_candidate else 0.8,
            "timestamp": now_s - index,
            "metadata": {"event_type": "candidate" if is_candidate else "strike"},
        }
        client.setex(f"prediction:{cell}:stage1", 300, json.dumps(stage1))
        client.setex(f"prediction:{cell}:stage2", 300, json.dumps({"prediction": 0}))
        client.zadd("prediction:recent", {cell: now_s - index})

    snapshot = build_dashboard_state(client, limit=4)

    assert len(snapshot["cells"]) == 4
    assert any(row["event_type"] == "candidate" for row in snapshot["cells"])
