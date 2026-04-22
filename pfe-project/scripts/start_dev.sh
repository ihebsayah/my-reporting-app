#!/usr/bin/env bash
# ── PFE Dev Stack Launcher ────────────────────────────────────────────────────
# Starts all three services in the background using the project .env.
#
# Usage:
#   chmod +x scripts/start_dev.sh
#   ./scripts/start_dev.sh          # start all
#   ./scripts/start_dev.sh stop     # stop all
#   ./scripts/start_dev.sh status   # check ports

set -euo pipefail

PYTHON="${PYTHON:-/Applications/Xcode.app/Contents/Developer/usr/bin/python3}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/artifacts/logs"
ENV_FILE="$PROJECT_ROOT/.env"

mkdir -p "$LOG_DIR" "$PROJECT_ROOT/data" \
         "$PROJECT_ROOT/artifacts/models/rf_confidence" \
         "$PROJECT_ROOT/artifacts/models/ner" \
         "$PROJECT_ROOT/artifacts/agent_logs"

# ── Load .env ─────────────────────────────────────────────────────────────────
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

# Ensure absolute DATABASE_URL for SQLite
export DATABASE_URL="sqlite:///$PROJECT_ROOT/data/reporting_app.db"

# ── Stop mode ─────────────────────────────────────────────────────────────────
if [[ "${1:-start}" == "stop" ]]; then
    echo "Stopping services..."
    pkill -f "uvicorn app.main"    2>/dev/null && echo "  Main API stopped"    || echo "  Main API was not running"
    pkill -f "uvicorn agents.main" 2>/dev/null && echo "  Agent svc stopped"   || echo "  Agent svc was not running"
    pkill -f "streamlit run"       2>/dev/null && echo "  Dashboard stopped"   || echo "  Dashboard was not running"
    exit 0
fi

# ── Status mode ───────────────────────────────────────────────────────────────
if [[ "${1:-start}" == "status" ]]; then
    for svc in \
        "Main API     http://localhost:8000/health" \
        "Agent svc    http://localhost:8001/health" \
        "Dashboard    http://localhost:8501/healthz"; do
        name=$(echo "$svc" | awk '{print $1, $2}')
        url=$(echo "$svc" | awk '{print $3}')
        code=$(curl -s -o /dev/null -w '%{http_code}' "$url" 2>/dev/null || echo "000")
        if [[ "$code" == "200" ]]; then
            echo "  ✅ $name — UP ($url)"
        else
            echo "  ❌ $name — DOWN ($url)"
        fi
    done
    exit 0
fi

# ── Start mode ────────────────────────────────────────────────────────────────
echo "🚀 Starting PFE dev stack from $PROJECT_ROOT"
echo ""

cd "$PROJECT_ROOT"

# 1. Main FastAPI (port 8000)
echo "  Starting Main API (port 8000)..."
nohup "$PYTHON" -m uvicorn app.main:app \
    --host 0.0.0.0 --port 8000 \
    --log-level warning \
    > "$LOG_DIR/main_api.log" 2>&1 &
echo "  PID $! → $LOG_DIR/main_api.log"

sleep 3

# 2. Agent microservice (port 8001)
echo "  Starting Agent Service (port 8001)..."
nohup "$PYTHON" -m uvicorn agents.main:app \
    --host 0.0.0.0 --port 8001 \
    --log-level warning \
    > "$LOG_DIR/agent_svc.log" 2>&1 &
echo "  PID $! → $LOG_DIR/agent_svc.log"

sleep 3

# 3. Streamlit dashboard (port 8501)
echo "  Starting Dashboard (port 8501)..."
nohup "$PYTHON" -m streamlit run app/dashboard/app.py \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false \
    > "$LOG_DIR/dashboard.log" 2>&1 &
echo "  PID $! → $LOG_DIR/dashboard.log"

sleep 5

echo ""
echo "Stack status:"
bash "$0" status
echo ""
echo "Logs: $LOG_DIR/"
echo "Dashboard: http://localhost:8501"
echo "Agent API docs: http://localhost:8001/docs"
echo "Main API docs:  http://localhost:8000/docs"
