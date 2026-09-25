#!/usr/bin/env bash
set -euo pipefail
SUITE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${QWENFUSE_PYTHON:-python}" -u "$SUITE_DIR/run_suite.py" "$@"
