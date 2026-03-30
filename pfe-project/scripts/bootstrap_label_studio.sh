#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_DIR="${PROJECT_ROOT}/deploy/label_studio"

cd "${COMPOSE_DIR}"
docker compose --env-file .env.example up -d

cd "${PROJECT_ROOT}"
python3 -m app.annotation.cli generate-assets --output-dir docs/annotation

echo "Label Studio services are starting on http://localhost:${LABEL_STUDIO_PORT:-8080}"
echo "Generated annotation assets in ${PROJECT_ROOT}/docs/annotation"
