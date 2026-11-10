#!/usr/bin/env bash
set -euo pipefail
PACKET_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 -m venv "$PACKET_DIR/.venv-ocr"
"$PACKET_DIR/.venv-ocr/bin/python" -m pip install -r "$PACKET_DIR/requirements-ocr.txt"
echo "OCR environment installed. Run ./run_packet_sealing.sh --check-ocr to download and test OCR models."
