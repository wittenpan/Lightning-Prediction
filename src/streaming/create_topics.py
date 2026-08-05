"""Create the input/output Kafka topics before long-running services start."""

from __future__ import annotations

import logging
import os
import time

from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

from .settings import Settings

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    partitions = int(os.getenv("KAFKA_TOPIC_PARTITIONS", "3"))
    replication_factor = int(os.getenv("KAFKA_REPLICATION_FACTOR", "1"))

    admin = None
    for attempt in range(1, 31):
        try:
            admin = KafkaAdminClient(
                bootstrap_servers=settings.kafka_bootstrap_servers,
                client_id="lightning-topic-init",
                **settings.kafka_kwargs(),
            )
            break
        except Exception as exc:
            if attempt == 30:
                raise
            logger.warning("Kafka not ready (%s); retrying in 10 seconds", exc)
            time.sleep(10)

    assert admin is not None
    existing = set(admin.list_topics())
    topics = [
        NewTopic(
            name=topic_name,
            num_partitions=partitions,
            replication_factor=replication_factor,
        )
        for topic_name in (settings.input_topic, settings.output_topic)
        if topic_name not in existing
    ]
    try:
        if topics:
            admin.create_topics(new_topics=topics, validate_only=False)
            logger.info("Created Kafka topics: %s", ", ".join(topic.name for topic in topics))
        else:
            logger.info("Kafka topics already exist")
    except TopicAlreadyExistsError:
        logger.info("Kafka topics already exist")
    finally:
        admin.close()


if __name__ == "__main__":
    main()
