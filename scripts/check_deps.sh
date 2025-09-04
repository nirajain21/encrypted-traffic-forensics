#!/usr/bin/env bash
set -e
command -v tshark >/dev/null || { echo "tshark not found. Install Wireshark/tshark."; exit 1; }
python3 - <<'PY'
import importlib
for m in ["pandas","numpy","sklearn","matplotlib","joblib"]:
    importlib.import_module(m)
print("Python deps OK")
PY
echo "All good."
