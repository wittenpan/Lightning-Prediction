# StormSignal dashboard

A private H3 operations view for the real-time lightning pipeline. It renders
recent Redis prediction state as MapLibre polygons and summarizes event volume,
active cells, cascade skip rate, prediction latency, and model probabilities.

## Local development

From the repository root, start the streaming stack and loopback API:

```bash
docker compose --profile dashboard up -d --build
```

Then run the UI:

```bash
cd dashboard
npm install
npm run dev
```

Open `http://localhost:3000`. Preview mode uses deterministic sample data. Live
mode polls `http://127.0.0.1:8765/api/state` every two seconds.

## AWS connection

The EC2 security group has no inbound application ports. Start the SSM tunnel
printed by the CloudFormation `DashboardPortForwardCommand` output, visit the
private hosted dashboard, and select **Live AWS**. The tunnel exposes neither
Redis nor the dashboard API publicly. If a browser blocks hosted-to-loopback
access, run `npm run dev` and use `http://localhost:3000` instead.

This is a portfolio/engineering observability view, not a safety-grade weather
warning product.
