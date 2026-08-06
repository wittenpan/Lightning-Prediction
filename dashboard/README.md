# StormSignal local operations console

The dashboard is intentionally local-only. It renders live H3 predictions from
the loopback dashboard API and never exposes Redis, Kafka, or EC2 publicly.

## Run locally

From the repository root, start the local pipeline and API:

```bash
docker compose --profile dashboard up -d --build
```

Start the frontend in a second terminal:

```bash
cd dashboard
npm install
npm run dev
```

Open `http://127.0.0.1:3000`.

## View the AWS POC privately

Start the SSM port-forward command printed by the CloudFormation stack. It maps
the EC2 API to `127.0.0.1:8765`. Keep the tunnel running, start this frontend,
and select **AWS LIVE**. No inbound application port is opened in AWS.

The 1-, 5-, and 30-minute controls change the strike-density window used to
rank and emphasize cells. Cell color represents the stage-one probability:
blue is low, yellow is watch, orange is at least 50%, and red is at least 80%.

This is an engineering proof of concept, not a safety-grade warning product.
