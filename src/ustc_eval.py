#!/usr/bin/env python3
import os
import subprocess
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Config ----------
USTC_ROOT = os.path.expanduser('~/datasets/ustc/USTC-TFC2016-master')
BENIGN_PCAPS = ['BitTorrent.pcap', 'FTP.pcap']          # small benign examples
MAL_PCAPS    = ['Miuref.pcap', 'Tinba.pcap']            # two malware examples
MODEL_PATH   = os.path.expanduser('~/models/rf_iscx_tshark.pkl')
OUT_DIR      = os.path.expanduser('~/flows/ustc_eval')
os.makedirs(OUT_DIR, exist_ok=True)

# Fields to extract per packet
PKT_FIELDS = [
    'ip.src', 'ip.dst',
    'tcp.srcport', 'udp.srcport',
    'tcp.dstport', 'udp.dstport',
    'ip.proto', 'frame.len', 'frame.time_relative'
]

def tshark_packets_to_csv(pcap_path: str, out_csv: str) -> pd.DataFrame:
    """
    Extract packets with tshark as a CSV of the fields in PKT_FIELDS.
    Uses correct -E key=value arguments (separator, header).
    """
    cmd = [
        'tshark', '-r', pcap_path,
        '-Y', 'ip and (tcp or udp)',
        '-T', 'fields',
        '-E', 'separator=,',
        '-E', 'header=y',
    ]
    for f in PKT_FIELDS:
        cmd += ['-e', f]

    try:
        raw = subprocess.check_output(cmd, text=True, errors='ignore')
    except subprocess.CalledProcessError as e:
        print(f"[!] tshark failed on {pcap_path}: {e}")
        return pd.DataFrame()

    # write and load robustly
    with open(out_csv, 'w') as f:
        f.write(raw)

    # Read with python engine and skip bad lines safely
    try:
        df = pd.read_csv(out_csv, engine='python', on_bad_lines='skip')
    except Exception as e:
        print(f"[!] pandas read_csv failed on {out_csv}: {e}")
        return pd.DataFrame()

    # If header missing/odd, force names
    if df.shape[1] != len(PKT_FIELDS):
        try:
            df = pd.read_csv(out_csv, engine='python', on_bad_lines='skip',
                             names=PKT_FIELDS, header=0)
        except Exception:
            try:
                df = pd.read_csv(out_csv, engine='python', on_bad_lines='skip',
                                 names=PKT_FIELDS, header=None)
            except Exception as e2:
                print(f"[!] pandas read_csv (force names) failed on {out_csv}: {e2}")
                return pd.DataFrame()

    return df

def packets_to_flows(df_packets: pd.DataFrame) -> pd.DataFrame:
    """
    Build very simple flow stats from packets: (5-tuple), pkts, bytes, duration, bps, pps, avg_len.
    """
    if df_packets.empty:
        return pd.DataFrame()

    # Ensure numeric types; fill missing ports w/ 0
    for c in ['tcp.srcport','udp.srcport','tcp.dstport','udp.dstport','ip.proto','frame.len','frame.time_relative']:
        if c in df_packets.columns:
            df_packets[c] = pd.to_numeric(df_packets[c], errors='coerce')

    # Build unified src/dst port columns: prefer TCP, fallback to UDP, else 0
    srcp = df_packets.get('tcp.srcport', pd.Series([np.nan]*len(df_packets))).fillna(
           df_packets.get('udp.srcport', pd.Series([np.nan]*len(df_packets)))).fillna(0).astype(int)
    dstp = df_packets.get('tcp.dstport', pd.Series([np.nan]*len(df_packets))).fillna(
           df_packets.get('udp.dstport', pd.Series([np.nan]*len(df_packets)))).fillna(0).astype(int)

    df = pd.DataFrame({
        'ip.src': df_packets.get('ip.src', ''),
        'ip.dst': df_packets.get('ip.dst', ''),
        'src_port': srcp,
        'dst_port': dstp,
        'proto': df_packets.get('ip.proto', 0).fillna(0).astype(int),
        'frame.len': df_packets.get('frame.len', 0).fillna(0),
        'frame.time_relative': df_packets.get('frame.time_relative', 0).fillna(0.0),
    })

    # group by 5-tuple
    grp = df.groupby(['ip.src','ip.dst','src_port','dst_port','proto'], dropna=False)

    def agg(g):
        pkts = len(g)
        bytes_ = g['frame.len'].sum()
        duration = float(g['frame.time_relative'].max() - g['frame.time_relative'].min()) if pkts > 1 else 0.0
        bps = (bytes_ / duration) if duration > 0 else float(bytes_)  # if only 1 pkt, bps=bytes
        pps = (pkts / duration) if duration > 0 else float(pkts)      # similarly
        avglen = bytes_ / pkts if pkts > 0 else 0.0
        return pd.Series({
            'pkts': pkts,
            'bytes': bytes_,
            'duration': duration,
            'bps': bps,
            'pps': pps,
            'avg_len': avglen
        })

    flows = grp.apply(agg).reset_index()
    return flows

def score_flows(model, flows_df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Score flows with the trained pipeline expecting columns:
    ['pkts','bytes','duration','bps','pps','avg_len','src_port','dst_port','proto']
    """
    if flows_df.empty:
        return pd.DataFrame()

    X = flows_df[['pkts','bytes','duration','bps','pps','avg_len','src_port','dst_port','proto']].copy()
    preds = model.predict(X)
    out = flows_df.copy()
    out['Prediction'] = preds
    out['TrueLabel'] = label
    return out

def plot_cm(cm: np.ndarray, labels: list, path: str):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('USTC Cross-Dataset Confusion Matrix')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    print(f"[*] Benign samples: {BENIGN_PCAPS}")
    print(f"[*] Malware samples: {MAL_PCAPS}")

    if not os.path.exists(MODEL_PATH):
        print(f"[!] Model not found: {MODEL_PATH}")
        return
    print(f"[*] Loading model: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    all_rows = []

    # BENIGN
    for name in BENIGN_PCAPS:
        pcap = os.path.join(USTC_ROOT, 'Benign', name)
        label = 'BENIGN'
        print(f"[+] Processing {pcap}  -> label={label}")
        pkt_csv = os.path.join(OUT_DIR, f"{os.path.splitext(name)[0]}_pkts.csv")
        dfp = tshark_packets_to_csv(pcap, pkt_csv)
        if dfp.empty:
            print(f"[!] No packets from {pcap}, skipping.")
            continue
        flow_csv = os.path.join(OUT_DIR, f"{os.path.splitext(name)[0]}_flows.csv")
        dff = packets_to_flows(dfp)
        if dff.empty:
            print(f"[!] No flows extracted from {pcap}, skipping.")
            continue
        dff.to_csv(flow_csv, index=False)
        scored = score_flows(model, dff, label)
        all_rows.append(scored)

    # MALICIOUS
    for name in MAL_PCAPS:
        pcap = os.path.join(USTC_ROOT, 'Malware', name)
        label = 'MALICIOUS'
        print(f"[+] Processing {pcap}  -> label={label}")
        pkt_csv = os.path.join(OUT_DIR, f"{os.path.splitext(name)[0]}_pkts.csv")
        dfp = tshark_packets_to_csv(pcap, pkt_csv)
        if dfp.empty:
            print(f"[!] No packets from {pcap}, skipping.")
            continue
        flow_csv = os.path.join(OUT_DIR, f"{os.path.splitext(name)[0]}_flows.csv")
        dff = packets_to_flows(dfp)
        if dff.empty:
            print(f"[!] No flows extracted from {pcap}, skipping.")
            continue
        dff.to_csv(flow_csv, index=False)
        scored = score_flows(model, dff, label)
        all_rows.append(scored)

    if not all_rows:
        print("[!] Nothing to score; all extracts empty.")
        return

    results = pd.concat(all_rows, ignore_index=True)
    results_path = os.path.join(OUT_DIR, 'ustc_all_predictions.csv')
    results.to_csv(results_path, index=False)

    # Build CM on BENIGN vs MALICIOUS (ignore NONVPN/VPN labels from training)
    y_true = results['TrueLabel'].values
    y_pred = np.where(results['Prediction'].isin(['MALICIOUS','VPN']), 'MALICIOUS', 'BENIGN')
    labels = ['BENIGN','MALICIOUS']
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)

    print("\n=== USTC cross-dataset results (trained on ISCX) ===")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix (rows=true: [BENIGN, MALICIOUS]):")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=labels))

    cm_path = os.path.join(OUT_DIR, 'ustc_confusion.png')
    plot_cm(cm, labels, cm_path)

    print("\n[✓] Artifacts:")
    print(f" - {results_path}")
    print(f" - {cm_path}")

if __name__ == "__main__":
    main()
