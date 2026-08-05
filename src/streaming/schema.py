"""Wire schema helpers for lightning events."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def normalize_timestamp_us(value: Any) -> int:
    """Normalize seconds, milliseconds, microseconds, or nanoseconds to microseconds."""
    timestamp = int(value)
    magnitude = abs(timestamp)
    if magnitude >= 100_000_000_000_000_000:  # nanoseconds
        return timestamp // 1_000
    if magnitude >= 100_000_000_000_000:  # microseconds
        return timestamp
    if magnitude >= 100_000_000_000:  # milliseconds
        return timestamp * 1_000
    return timestamp * 1_000_000  # seconds


def lzw_decompress(codes: list[int]) -> str:
    """Decode the LZW representation used by the Blitzortung websocket."""
    if not codes:
        raise ValueError("empty LZW message")
    dictionary = {i: chr(i) for i in range(256)}
    dictionary_size = 256
    previous = codes[0]
    output = [dictionary[previous]]

    for code in codes[1:]:
        if code in dictionary:
            entry = dictionary[code]
        elif code == dictionary_size:
            entry = dictionary[previous] + dictionary[previous][0]
        else:
            raise ValueError(f"invalid LZW code: {code}")
        output.append(entry)
        dictionary[dictionary_size] = dictionary[previous] + entry[0]
        dictionary_size += 1
        previous = code
    return "".join(output)


def decode_blitzortung_message(message: str) -> Optional[Dict[str, Any]]:
    """Decode either a compressed live message or plain JSON test fixture."""
    try:
        if message.lstrip().startswith("{"):
            try:
                return json.loads(message)
            except json.JSONDecodeError:
                # Blitzortung LZW payloads often retain a literal JSON prefix.
                pass
        return json.loads(lzw_decompress([ord(char) for char in message]))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def make_event(data: Dict[str, Any], source: str = "blitzortung") -> Dict[str, Any]:
    """Validate and convert a provider event into the Kafka wire schema."""
    for field in ("lat", "lon", "time"):
        if field not in data:
            raise ValueError(f"missing required field: {field}")

    ingested_at_ns = time.time_ns()
    timestamp_us = normalize_timestamp_us(data["time"])
    identity = f"{timestamp_us}:{data['lat']}:{data['lon']}:{data.get('mds', '')}"
    event_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return {
        "event_id": event_id,
        "latitude": float(data["lat"]),
        "longitude": float(data["lon"]),
        "timestamp": timestamp_us,
        "timestamp_unit": "microseconds",
        "altitude": int(data.get("alt", 0)),
        "polarity": int(data.get("pol", 0)),
        "ingested_at_ns": ingested_at_ns,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
    }
