"""Environment-driven settings shared by local Docker and AWS services."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _optional(name: str) -> Optional[str]:
    value = os.getenv(name)
    return value if value else None


@dataclass(frozen=True)
class Settings:
    kafka_bootstrap_servers: str = "localhost:9092"
    input_topic: str = "lightning-strikes"
    output_topic: str = "lightning-predictions"
    consumer_group: str = "lightning-prediction-group"
    kafka_security_protocol: str = "PLAINTEXT"
    kafka_sasl_mechanism: Optional[str] = None
    kafka_sasl_username: Optional[str] = None
    kafka_sasl_password: Optional[str] = None
    kafka_ssl_cafile: Optional[str] = None
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_ssl: bool = False
    model_dir: str = "data/models"
    h3_resolution: int = 7
    metrics_port: int = 8000

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            kafka_bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            input_topic=os.getenv("KAFKA_INPUT_TOPIC", "lightning-strikes"),
            output_topic=os.getenv("KAFKA_OUTPUT_TOPIC", "lightning-predictions"),
            consumer_group=os.getenv("KAFKA_CONSUMER_GROUP", "lightning-prediction-group"),
            kafka_security_protocol=os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"),
            kafka_sasl_mechanism=_optional("KAFKA_SASL_MECHANISM"),
            kafka_sasl_username=_optional("KAFKA_SASL_USERNAME"),
            kafka_sasl_password=_optional("KAFKA_SASL_PASSWORD"),
            kafka_ssl_cafile=_optional("KAFKA_SSL_CAFILE"),
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_db=int(os.getenv("REDIS_DB", "0")),
            redis_password=_optional("REDIS_PASSWORD"),
            redis_ssl=os.getenv("REDIS_SSL", "false").lower() in {"1", "true", "yes"},
            model_dir=os.getenv("MODEL_DIR", "data/models"),
            h3_resolution=int(os.getenv("H3_RESOLUTION", "7")),
            metrics_port=int(os.getenv("METRICS_PORT", "8000")),
        )

    def kafka_kwargs(self) -> Dict[str, Any]:
        """Return connection options accepted by kafka-python clients."""
        options: Dict[str, Any] = {"security_protocol": self.kafka_security_protocol}
        if self.kafka_sasl_mechanism:
            options["sasl_mechanism"] = self.kafka_sasl_mechanism
        if self.kafka_sasl_username:
            options["sasl_plain_username"] = self.kafka_sasl_username
        if self.kafka_sasl_password:
            options["sasl_plain_password"] = self.kafka_sasl_password
        if self.kafka_ssl_cafile:
            options["ssl_cafile"] = self.kafka_ssl_cafile
        return options
