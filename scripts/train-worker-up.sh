#!/usr/bin/env bash
# One-command (re)launch of the detached trainer. Idempotent: run it on a
# fresh box (after copying docker/.env) or to restart after a crash — the
# worker resumes from the candidate manifest, replaying any unpublished
# windows from R2, so relaunching is always safe.
#
# Usage:   ./scripts/train-worker-up.sh          # (re)start
#          ./scripts/train-worker-up.sh logs     # tail the worker
#          ./scripts/train-worker-up.sh down     # stop
set -euo pipefail
cd "$(dirname "$0")/../docker"

COMPOSE=(docker compose -f docker-compose.train-worker.yml)

case "${1:-up}" in
  up)
    test -f .env || {
      echo "docker/.env missing — copy .env.example.train-worker and fill it in" >&2
      exit 1
    }
    "${COMPOSE[@]}" pull --quiet 2>/dev/null || true  # local tags have no remote
    "${COMPOSE[@]}" up -d --force-recreate
    echo "--- train-worker started; first log lines: ---"
    sleep 3
    docker logs --tail 30 reliquary-train-worker
    ;;
  logs)
    docker logs -f --tail 100 reliquary-train-worker
    ;;
  down)
    "${COMPOSE[@]}" down
    ;;
  *)
    echo "usage: $0 [up|logs|down]" >&2
    exit 2
    ;;
esac
