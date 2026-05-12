#!/usr/bin/env bash
# Start the AutoGPT TradingAgent + Ruflo swarm.
# Usage: ./start_trading_swarm.sh [--port 8000] [--strategy rsi] [--dry-run]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=8000
STRATEGY="rsi"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)     PORT="$2";     shift 2 ;;
    --strategy) STRATEGY="$2"; shift 2 ;;
    --dry-run)  DRY_RUN=true;  shift   ;;
    *) echo "Unknown arg: $1"; exit 1  ;;
  esac
done

# ── Environment ─────────────────────────────────────────────────────────────
export DATABASE_STRING="${DATABASE_STRING:-sqlite:///trading_agent.db}"
export AGENT_WORKSPACE="${AGENT_WORKSPACE:-${REPO_ROOT}/workspace}"
export TRADING_BOT_PATH="${TRADING_BOT_PATH:-${REPO_ROOT}/trading-bot}"
export PYTHONPATH="${TRADING_BOT_PATH}:${PYTHONPATH:-}"
export SYMBOLS="${SYMBOLS:-BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT}"
export DEFAULT_STRATEGY="${STRATEGY}"
export CLAUDE_FLOW_WORKFLOW="${REPO_ROOT}/.claude-flow/workflows/trading_swarm.yaml"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║       AutoGPT + Ruflo Trading Swarm                 ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Agent Protocol : http://localhost:${PORT}/ap/v1/     ║"
echo "║  API Docs       : http://localhost:${PORT}/docs        ║"
echo "║  Symbols        : ${SYMBOLS}        ║"
echo "║  Default strat  : ${STRATEGY}                               ║"
echo "║  Workflow       : .claude-flow/workflows/trading_swarm.yaml ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
  echo "Dry-run: environment configured. Run without --dry-run to start the server."
  exit 0
fi

mkdir -p "${AGENT_WORKSPACE}"

cd "${REPO_ROOT}/autogpts/forge"

# Verify dependencies
if ! command -v poetry &>/dev/null; then
  echo "Error: poetry not found. Run: pip install poetry"
  exit 1
fi

echo "Starting TradingAgent on port ${PORT}..."
exec poetry run uvicorn forge.trading.app:app \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --reload \
  --log-level info
