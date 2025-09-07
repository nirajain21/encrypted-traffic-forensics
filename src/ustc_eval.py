# Cross-dataset validation (USTC dataset tested with ISCX-trained model)
#!/usr/bin/env python3
import os, subprocess, pandas as pd, numpy as np, joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Config ----------
# Location of USTC dataset (Benign and Malware traffic samples)
USTC_ROOT = os.path.expanduser('~/datasets/ustc/USTC-TFC2016-master')

# Specific PCAPs chosen for quick testing (small, not the full dataset)
BENIGN_PCAPS = ['BitTorrent.pcap', 'FTP.pcap']   # benign examples
MAL_PCAPS    = ['Miuref.pcap', 'Tinba.pcap']     # malware examples

# Use model trained on ISCX VPN-nonVPN dataset
MODEL_PATH   = os.path.expanduser('~/models/rf_iscx_tshark.pkl')

# Where to save intermediate CSVs + plots
OUT_DIR      = os.path.expanduser('~/flows/ustc_eval')
os.makedirs(OUT_DIR, exist_ok=True)

# Fields we want from each packet (used to build flow features later)
PKT_FIELDS = [
    'ip.src','ip.dst','tcp.srcport','udp.srcport',
    'tcp.dstport','udp.dstport','ip.proto',
    'frame.len','frame.time_relative'
]

# ---------- Step 1: Extract packets with tshark ----------
def tshark_packets_to_csv(pcap_path: str, out_csv: str) -> pd.DataFrame:
    """
    Run tshark to extract fields per packet into a CSV.
    Returns DataFrame of packets.
    """
    cmd = ['tshark','-r',pcap_path,'-Y','ip and (tcp or udp)','-T','fields',
           '-E','separator=,','-E','header=y']
    for f in PKT_FIELDS: cmd += ['-e', f]

    try:
        raw = subprocess.check_output(cmd, text=True, errors='ignore')
    except subprocess.CalledProcessError as e:
        print(f"[!] tshark failed: {e}"); return pd.DataFrame()

    with open(out_csv,'w') as f: f.write(raw)

    # Robust CSV read
    try:
        df = pd.read_csv(out_csv, engine='python', on_bad_lines='skip')
    except Exception:
        return pd.DataFrame()
    return df

# ---------- Step 2: Aggregate packets into flows ----------
def packets_to_flows(df_packets: pd.DataFrame) -> pd.DataFrame:
    """
    Group packets into flows and compute:
    pkts, bytes, duration, bps, pps, avg_len
    """
    if df_packets.empty: return pd.DataFrame()

    # Convert columns to numeric where needed
    for c in ['tcp.srcport','udp.srcport','tcp.dstport','udp.dstport','ip.proto',
              'frame.len','frame.time_relative']:
        if c in df_packets.columns:
            df_packets[c] = pd.to_numeric(df_packets[c], errors='coerce')

    # Merge TCP/UDP ports into unified src/dst
    srcp = df_packets.get('tcp.srcport', pd.Series()).fillna(
           df_packets.get('udp.srcport', pd.Series())).fillna(0).astype(int)
    dstp = df_packets.get('tcp.dstport', pd.Series()).fillna(
           df_packets.get('udp.dstport', pd.Series())).fillna(0).astype(int)

    df = pd.DataFrame({
        'ip.src': df_packets.get('ip.src',''),
        'ip.dst': df_packets.get('ip.dst',''),
        'src_port': srcp,
        'dst_port': dstp,
        'proto': df_packets.get('ip.proto',0).fillna(0).astype(int),
        'frame.len': df_packets.get('frame.len',0).fillna(0),
        'frame.time_relative': df_packets.get('frame.time_relative',0).fillna(0.0),
    })

    # Group by 5-tuple (src IP, dst IP, src port, dst port, protocol)
    grp = df.groupby(['ip.src','ip.dst','src_port','dst_port','proto'])

    def agg(g):
        pkts = len(g)
        bytes_ = g['frame.len'].sum()
        duration = (g['frame.time_relative'].max() - g['frame.time_relative'].min()) if pkts>1 else 0.0
        bps = (bytes_/duration) if duration>0 else float(bytes_)
        pps = (pkts/duration) if duration>0 else float(pkts)
        avglen = bytes_/pkts if pkts>0 else 0.0
        return pd.Series({'pkts':pkts,'bytes':bytes_,'duration':duration,
                          'bps':bps,'pps':pps,'avg_len':avglen})
    return grp.apply(agg).reset_index()

# ---------- Step 3: Score flows ----------
def score_flows(model, flows_df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Run trained RF model on flows.
    Adds Prediction + TrueLabel.
    """
    if flows_df.empty: return pd.DataFrame()
    X = flows_df[['pkts','bytes','duration','bps','pps','avg_len','src_port','dst_port','proto']].copy()
    preds = model.predict(X)
    out = flows_df.copy()
    out['Prediction'] = preds
    out['TrueLabel'] = label
    return out

# ---------- Step 4: Plot confusion matrix ----------
def plot_cm(cm: np.ndarray, labels: list, path: str):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title('USTC Cross-Dataset Confusion Matrix')
    plt.tight_layout()
    plt.savefig(path); plt.close()

# ---------- Step 5: Main pipeline ----------
def main():
    print(f"[*] Benign samples: {BENIGN_PCAPS}")
    print(f"[*] Malware samples: {MAL_PCAPS}")

    # Load trained ISCX RF model
    if not os.path.exists(MODEL_PATH):
        print("[!] Model not found"); return
    model = joblib.load(MODEL_PATH)

    all_rows = []

    # Process benign PCAPs
    for name in BENIGN_PCAPS:
        pcap = os.path.join(USTC_ROOT,'Benign',name)
        dfp = tshark_packets_to_csv(pcap, f"{OUT_DIR}/{name}_pkts.csv")
        dff = packets_to_flows(dfp)
        if not dff.empty:
            dff.to_csv(f"{OUT_DIR}/{name}_flows.csv", index=False)
            scored = score_flows(model, dff, 'BENIGN')
            all_rows.append(scored)

    # Process malicious PCAPs
    for name in MAL_PCAPS:
        pcap = os.path.join(USTC_ROOT,'Malware',name)
        dfp = tshark_packets_to_csv(pcap, f"{OUT_DIR}/{name}_pkts.csv")
        dff = packets_to_flows(dfp)
        if not dff.empty:
            dff.to_csv(f"{OUT_DIR}/{name}_flows.csv", index=False)
            scored = score_flows(model, dff, 'MALICIOUS')
            all_rows.append(scored)

    if not all_rows: 
        print("[!] No flows extracted"); return

    results = pd.concat(all_rows, ignore_index=True)
    results.to_csv(f"{OUT_DIR}/ustc_all_predictions.csv", index=False)

    # Collapse VPN label → MALICIOUS, NONVPN → BENIGN for comparison
    y_true = results['TrueLabel'].values
    y_pred = np.where(results['Prediction'].isin(['MALICIOUS','VPN']), 'MALICIOUS', 'BENIGN')

    labels = ['BENIGN','MALICIOUS']
    cm = confusion_matrix(y_true,y_pred,labels=labels)
    acc = accuracy_score(y_true,y_pred)

    print("\n=== USTC cross-dataset results (trained on ISCX) ===")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix:"); print(cm)
    print("Classification report:\n", classification_report(y_true,y_pred,labels=labels))

    plot_cm(cm, labels, f"{OUT_DIR}/ustc_confusion.png")
    print(f"\n[✓] Results saved in {OUT_DIR}")

if __name__ == "__main__":
    main()
