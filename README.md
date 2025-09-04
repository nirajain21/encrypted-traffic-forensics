# Encrypted Traffic Forensics (Flow Metadata + ML)

This repo contains code to extract flow-level features from PCAPs (via `tshark`),
train ML models (RandomForest / XGBoost), and run cross-dataset evaluations (ISCX VPN-nonVPN, USTC subset).

## Quickstart
```bash
python3 -m venv cfm-env
source cfm-env/bin/activate
pip install -r requirements.txt
make check
```

## Train + plots
```bash
make train
```

## Single-PCAP pipeline
```bash
make pipeline
```

## USTC subset evaluation
```bash
make ustc
```

## Setup from scratch
```bash
bash scripts/bootstrap_env.sh
source cfm-env/bin/activate
make check
```
