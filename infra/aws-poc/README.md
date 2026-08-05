# AWS proof of concept

The CloudFormation stack supports two modes:

- `self-hosted`: Kafka, Redis, prediction, live producer, and dashboard API run
  on one EC2 instance.
- `managed`: Amazon MSK and ElastiCache provide Kafka and Redis; EC2 runs the
  containerized application services.

Managed mode is intentionally short lived. It creates billable services, even
when EC2 reports free-tier eligibility. A monthly budget alert and an automatic
stack deletion timer are guardrails, not a hard spending limit.

## Deploy

```bash
./deploy_aws.sh \
  --mode managed \
  --profile lightning-poc \
  --email you@example.com \
  --budget-usd 10 \
  --ttl-hours 4 \
  --confirm-managed-costs
```

The helper validates the template, caller, region, and Git source. The remote
branch must resolve to the exact local commit so EC2 cannot silently run stale
code.

## Connect the private dashboard

Get the generated port-forwarding command:

```bash
./infra/aws-poc/aws_poc.sh status --profile lightning-poc
```

Run the `DashboardPortForwardCommand` output and append
`--profile lightning-poc`. It maps the EC2 loopback API to
`http://127.0.0.1:8765` through AWS Systems Manager. No inbound application port
is opened. Visit the private hosted dashboard and choose **Live AWS**. If the
browser blocks hosted-to-loopback access, run `cd dashboard && npm run dev` and
use `http://localhost:3000` instead.

## Inspect or remove

```bash
./infra/aws-poc/aws_poc.sh status --profile lightning-poc
./cleanup_aws.sh --profile lightning-poc --yes
```

The EventBridge Scheduler cleanup is set during deployment. Delete sooner after
testing to avoid unnecessary MSK and ElastiCache charges.
