#!/usr/bin/env bash
set -euo pipefail
PACKET_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRACKER_DIR="${JEWEL_TRACKER_DIR:-$PACKET_DIR/../jewel_tracker/jewel_tracker}"
PACKET_PYTHON="${PACKET_PYTHON:-$TRACKER_DIR/.venv-slic/bin/python}"
if [[ ! -x "$PACKET_PYTHON" ]]; then
    echo "SLIC environment not found. Set JEWEL_TRACKER_DIR or PACKET_PYTHON." >&2
    exit 1
fi
export JEWEL_TRACKER_DIR="$TRACKER_DIR"
cd "$PACKET_DIR"
exec "$PACKET_PYTHON" "$PACKET_DIR/packet_app.py" "$@"
