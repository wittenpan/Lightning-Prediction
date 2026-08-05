FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN groupadd --system lightning && useradd --system --gid lightning lightning
COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

COPY src ./src
COPY data/models ./data/models
RUN chown -R lightning:lightning /app
USER lightning

CMD ["python", "-m", "src.streaming.consumer"]
