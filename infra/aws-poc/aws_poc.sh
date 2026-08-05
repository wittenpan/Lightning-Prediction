#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
TEMPLATE_FILE="$SCRIPT_DIR/template.yaml"

ACTION=${1:-help}
if [[ $# -gt 0 ]]; then shift; fi

REGION=${AWS_REGION:-us-east-1}
PROFILE=${AWS_PROFILE:-}
STACK_NAME=lightning-poc
MODE=self-hosted
INSTANCE_TYPE=t3.small
REPOSITORY_URL=https://github.com/wittenpan/Lightning-Prediction.git
REPOSITORY_REF=main
BUDGET_USD=10
TTL_HOURS=4
BUDGET_EMAIL=""
CONFIRM_MANAGED=false
ALLOW_SOURCE_MISMATCH=false
CONFIRM_DELETE=false

usage() {
  cat <<'EOF'
Usage: infra/aws-poc/aws_poc.sh ACTION [options]

Actions:
  preflight   Check AWS identity, template, region, instance type, and source ref
  deploy      Create or update the CloudFormation stack
  status      Show stack status and outputs
  destroy     Delete the stack (requires --yes)

Options:
  --mode self-hosted|managed  self-hosted runs Kafka and Redis on EC2 (default)
  --email ADDRESS             required for budget alerts during deploy
  --region REGION             default: us-east-1
  --profile NAME              AWS CLI profile (or set AWS_PROFILE)
  --stack NAME                default: lightning-poc
  --instance-type TYPE        t3.micro or t3.small; default: t3.small
  --repo-url URL              Git repository cloned by EC2
  --repo-ref REF              branch or tag; default: main
  --budget-usd AMOUNT         monthly alert budget; default: 10
  --ttl-hours HOURS           automatic stack deletion after 2-24 hours; default: 4
  --confirm-managed-costs     required for managed MSK + ElastiCache mode
  --allow-source-mismatch     allow deploy when the remote ref differs from local HEAD
  --yes                       required for destroy

Nothing is created by preflight. AWS Budgets alerts are delayed notifications,
not a hard spending cap. The stack TTL is an additional cost guardrail.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE=${2:?missing value for --mode}; shift 2 ;;
    --email) BUDGET_EMAIL=${2:?missing value for --email}; shift 2 ;;
    --region) REGION=${2:?missing value for --region}; shift 2 ;;
    --profile) PROFILE=${2:?missing value for --profile}; shift 2 ;;
    --stack) STACK_NAME=${2:?missing value for --stack}; shift 2 ;;
    --instance-type) INSTANCE_TYPE=${2:?missing value for --instance-type}; shift 2 ;;
    --repo-url) REPOSITORY_URL=${2:?missing value for --repo-url}; shift 2 ;;
    --repo-ref) REPOSITORY_REF=${2:?missing value for --repo-ref}; shift 2 ;;
    --budget-usd) BUDGET_USD=${2:?missing value for --budget-usd}; shift 2 ;;
    --ttl-hours) TTL_HOURS=${2:?missing value for --ttl-hours}; shift 2 ;;
    --confirm-managed-costs) CONFIRM_MANAGED=true; shift ;;
    --allow-source-mismatch) ALLOW_SOURCE_MISMATCH=true; shift ;;
    --yes) CONFIRM_DELETE=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

if [[ -n "$PROFILE" ]]; then
  export AWS_PROFILE="$PROFILE"
fi

validate_inputs() {
  [[ "$MODE" == self-hosted || "$MODE" == managed ]] || die "--mode must be self-hosted or managed"
  [[ "$STACK_NAME" =~ ^[a-z][a-z0-9-]{2,30}$ ]] || die "--stack must match [a-z][a-z0-9-]{2,30}"
  [[ "$INSTANCE_TYPE" == t3.micro || "$INSTANCE_TYPE" == t3.small ]] || die "unsupported --instance-type"
  [[ "$TTL_HOURS" =~ ^[0-9]+$ ]] || die "--ttl-hours must be an integer"
  (( TTL_HOURS >= 2 && TTL_HOURS <= 24 )) || die "--ttl-hours must be from 2 through 24"
  [[ "$BUDGET_USD" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "--budget-usd must be numeric"
  [[ "$REPOSITORY_URL" == https://* ]] || die "--repo-url must use HTTPS"
  [[ "$REPOSITORY_REF" =~ ^[A-Za-z0-9._/-]+$ ]] || die "--repo-ref contains unsupported characters"
}

require_aws() {
  command -v aws >/dev/null 2>&1 || die "AWS CLI v2 is required (macOS: brew install awscli)"
}

check_source_ref() {
  command -v git >/dev/null 2>&1 || die "git is required to verify the deployment source"
  local remote_sha local_sha
  remote_sha=$(git ls-remote "$REPOSITORY_URL" "$REPOSITORY_REF" | awk 'NR == 1 {print $1}')
  [[ -n "$remote_sha" ]] || die "repository ref does not resolve: $REPOSITORY_URL $REPOSITORY_REF"
  local_sha=$(git -C "$REPO_DIR" rev-parse HEAD)
  printf 'Remote source: %s (%s)\n' "$remote_sha" "$REPOSITORY_REF"
  printf 'Local HEAD:    %s\n' "$local_sha"

  if [[ "$remote_sha" != "$local_sha" ]]; then
    if [[ "$ALLOW_SOURCE_MISMATCH" == true ]]; then
      printf 'warning: remote source differs from local HEAD; override accepted.\n' >&2
    else
      die "remote source differs from local HEAD; push the intended commit or pass --allow-source-mismatch"
    fi
  fi
  if [[ -n $(git -C "$REPO_DIR" status --porcelain) ]]; then
    if [[ "$ACTION" == deploy && "$ALLOW_SOURCE_MISMATCH" != true ]]; then
      die "local changes are uncommitted and will not be present on EC2; commit/push them first"
    fi
    printf 'warning: local changes are uncommitted and will not be present on EC2.\n' >&2
  fi
}

preflight() {
  validate_inputs
  require_aws
  printf 'Checking AWS caller identity in %s...\n' "$REGION"
  aws sts get-caller-identity --output table
  aws cloudformation validate-template \
    --template-body "file://$TEMPLATE_FILE" \
    --region "$REGION" >/dev/null

  local az_count free_tier_flag
  if [[ "$MODE" == managed ]]; then
    az_count=$(aws ec2 describe-availability-zones \
      --region "$REGION" \
      --filters Name=state,Values=available \
      --query 'length(AvailabilityZones)' \
      --output text)
    (( az_count >= 2 )) || die "$REGION needs at least two available Availability Zones for managed mode"
  fi
  free_tier_flag=$(aws ec2 describe-instance-types \
    --region "$REGION" \
    --instance-types "$INSTANCE_TYPE" \
    --query 'InstanceTypes[0].FreeTierEligible' \
    --output text)
  printf 'EC2 %s reports FreeTierEligible=%s for this API/region.\n' "$INSTANCE_TYPE" "$free_tier_flag"
  check_source_ref
  printf 'Preflight passed. No AWS resources were created.\n'
}

show_status() {
  validate_inputs
  require_aws
  aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query 'Stacks[0].{Status:StackStatus,Created:CreationTime,Outputs:Outputs}' \
    --output table
}

deploy() {
  [[ "$BUDGET_EMAIL" == *@*.* ]] || die "--email is required for budget alerts"
  if [[ "$MODE" == managed && "$CONFIRM_MANAGED" != true ]]; then
    die "managed mode creates billable MSK and ElastiCache resources; pass --confirm-managed-costs"
  fi
  preflight
  printf 'Deploying %s in %s mode; automatic deletion is scheduled after %s hours.\n' \
    "$STACK_NAME" "$MODE" "$TTL_HOURS"
  aws cloudformation deploy \
    --template-file "$TEMPLATE_FILE" \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --capabilities CAPABILITY_IAM \
    --no-fail-on-empty-changeset \
    --parameter-overrides \
      ProjectName="$STACK_NAME" \
      DeploymentMode="$MODE" \
      InstanceType="$INSTANCE_TYPE" \
      RepositoryUrl="$REPOSITORY_URL" \
      RepositoryRef="$REPOSITORY_REF" \
      MonthlyBudgetUsd="$BUDGET_USD" \
      BudgetEmail="$BUDGET_EMAIL" \
      AutoDeleteHours="$TTL_HOURS"
  show_status
}

destroy() {
  validate_inputs
  require_aws
  [[ "$CONFIRM_DELETE" == true ]] || die "destroy requires --yes"
  printf 'Deleting CloudFormation stack %s in %s...\n' "$STACK_NAME" "$REGION"
  aws cloudformation delete-stack --stack-name "$STACK_NAME" --region "$REGION"
  aws cloudformation wait stack-delete-complete --stack-name "$STACK_NAME" --region "$REGION"
  printf 'Stack deleted. Confirm Billing and Cost Management shows no unexpected resources.\n'
}

case "$ACTION" in
  preflight) preflight ;;
  deploy) deploy ;;
  status) show_status ;;
  destroy) destroy ;;
  help|-h|--help) usage ;;
  *) usage >&2; die "unknown action: $ACTION" ;;
esac
