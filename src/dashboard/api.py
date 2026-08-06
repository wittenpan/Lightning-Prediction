"""FastAPI surface for the private H3 dashboard."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import redis
import h3
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from src.streaming.settings import Settings

app = FastAPI(title="Lightning H3 Dashboard API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
    allow_private_network=True,
)


def redis_client() -> redis.Redis:
    settings = Settings.from_env()
    return redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password,
        ssl=settings.redis_ssl,
        decode_responses=False,
        socket_timeout=5,
        socket_connect_timeout=5,
    )


def _decode(value: bytes | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}


WINDOWS_US = {
    "1m": 60_000_000,
    "5m": 300_000_000,
    "30m": 1_800_000_000,
}


def _recent_cells(client: redis.Redis, limit: int, now_s: float) -> tuple[list[str], int]:
    """Read the bounded recency index, with a fallback for older deployments."""
    indexed = client.zrevrangebyscore(
        "prediction:recent",
        now_s,
        now_s - 300,
        start=0,
        num=limit,
    )
    if indexed:
        cells = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in indexed]
        total = int(client.zcount("prediction:recent", now_s - 300, now_s))
        return cells, total

    stage1_keys = list(client.scan_iter(match="prediction:*:stage1", count=500))[:limit]
    cells = [
        key.decode("utf-8").split(":")[1] if isinstance(key, bytes) else key.split(":")[1]
        for key in stage1_keys
    ]
    return cells, len(cells)


def build_dashboard_state(client: redis.Redis, limit: int = 500) -> dict[str, Any]:
    """Build a bounded dashboard snapshot with one Redis pipeline."""
    now_s = time.time()
    now_us = int(now_s * 1_000_000)
    cells, total_prediction_cells = _recent_cells(client, limit, now_s)
    pipeline = client.pipeline(transaction=False)
    for cell in cells:
        pipeline.get(f"prediction:{cell}:stage1")
        pipeline.get(f"prediction:{cell}:stage2")
        for window_us in WINDOWS_US.values():
            pipeline.zcount(f"strikes:{cell}", now_us - window_us, now_us)
    values = iter(pipeline.execute())

    rows: list[dict[str, Any]] = []
    for cell in cells:
        stage1 = _decode(next(values))
        stage2 = _decode(next(values))
        strike_counts = {name: int(next(values)) for name in WINDOWS_US}
        if not stage1:
            continue
        metadata = stage1.get("metadata", {})
        updated_s = float(stage1.get("timestamp", now_s))
        stage1_positive = int(stage1.get("prediction", 0))
        stage2_positive = int(stage2.get("prediction", 0))
        rows.append(
            {
                "h3_cell": cell,
                "resolution": h3.get_resolution(cell),
                "stage1_probability": float(stage1.get("probability", 0.0)),
                "stage1_prediction": stage1_positive,
                "stage2_probability": float(stage2.get("probability", 0.0)),
                "stage2_prediction": stage2_positive,
                "stage2_executed": bool(stage1_positive),
                "combined_prediction": int(stage1_positive and stage2_positive),
                "strike_counts": strike_counts,
                "strike_count_1m": strike_counts["1m"],
                "strike_count_5m": strike_counts["5m"],
                "strike_count_30m": strike_counts["30m"],
                "processing_ms": float(metadata.get("processing_ms", 0.0)),
                "e2e_ms": float(metadata.get("ingestion_to_prediction_ms", 0.0)),
                "event_type": metadata.get("event_type", "strike"),
                "latitude": metadata.get("latitude"),
                "longitude": metadata.get("longitude"),
                "event_id": metadata.get("event_id"),
                "updated_at": datetime.fromtimestamp(updated_s, tz=timezone.utc).isoformat(),
                "age_seconds": max(0.0, round(now_s - updated_s, 1)),
            }
        )

    rows.sort(key=lambda row: row["updated_at"], reverse=True)
    stage2_calls = sum(int(row["stage2_executed"]) for row in rows)
    candidate_rows = [row for row in rows if row["event_type"] == "candidate"]
    gate_rows = candidate_rows or rows
    gate_stage2_calls = sum(int(row["stage2_executed"]) for row in gate_rows)
    # Scheduled candidate sweeps measure gate efficiency; live-observation
    # latency remains the user-facing ingestion SLO.
    e2e_values = sorted(
        row["e2e_ms"]
        for row in rows
        if row["event_type"] != "candidate" and row["e2e_ms"] > 0
    )
    p95_index = max(0, min(len(e2e_values) - 1, int(len(e2e_values) * 0.95)))
    strikes_by_window = {
        name: sum(row["strike_counts"][name] for row in rows)
        for name in WINDOWS_US
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "active_cells": len(rows),
            "total_prediction_cells": total_prediction_cells,
            "stage1_positive_cells": sum(row["stage1_prediction"] for row in rows),
            "stage2_positive_cells": sum(row["stage2_prediction"] for row in rows),
            "candidate_cells": sum(row["event_type"] == "candidate" for row in rows),
            "observed_cells": sum(row["event_type"] != "candidate" for row in rows),
            "strikes_by_window": strikes_by_window,
            "strikes_1m": strikes_by_window["1m"],
            "strikes_5m": strikes_by_window["5m"],
            "strikes_30m": strikes_by_window["30m"],
            "stage2_skip_rate": 1.0 - gate_stage2_calls / len(gate_rows) if gate_rows else 0.0,
            "overall_stage2_skip_rate": 1.0 - stage2_calls / len(rows) if rows else 0.0,
            "e2e_p95_ms": e2e_values[p95_index] if e2e_values else 0.0,
        },
        "cells": rows,
    }


@app.get("/health")
def health() -> dict[str, str]:
    redis_client().ping()
    return {"status": "ok"}


@app.get("/api/state")
def state(limit: int = Query(default=500, ge=1, le=2_000)) -> dict[str, Any]:
    return build_dashboard_state(redis_client(), limit=limit)
