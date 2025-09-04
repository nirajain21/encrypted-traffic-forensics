#!/usr/bin/env bash
set -e
python3 -m venv cfm-env
source cfm-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Activate with: source cfm-env/bin/activate"
