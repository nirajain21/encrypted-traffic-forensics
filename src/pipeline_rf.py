# Apply trained RF model to a new single PCAP.

#!/usr/bin/env python3
import os
import subprocess
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ----------- Config section -----------
# Path to trained Random Forest model
MODEL_PATH   = os.path.expanduser('~/models/rf_iscx_tshark.pkl')

# Output folders (for CSVs + figures)
FIG_DIR      = os.path.expanduser('~/figures')
FLOW_DIR     = os.path.expanduser('~/flows')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(FLOW_DIR, exist_ok=True)

# Picking one NONVPN and one VPN PCAP from ISCX dataset
NONVPN_PCAP  = os.path.expanduser('~/datasets/iscx/aim_chat_3a_fixed.pcap')
VPN_PCAP     = os.path.expanduser('~/datasets/iscx/VPN/vpn_facebook_chat1b_classic.pcap')

# Features expected by the model (order must match training)
FEATURE_ORDER = ['pkts','bytes','duration','bps','pps','avg_len',
                 'src_port','dst_port','proto']


# ----------- Helper functions -----------

def extract_packets(pcap_path: str, out_csv: str) -> pd.DataFrame:
    """
    Extract per-packet fields from PCAP using tshark and save as CSV.
    Returns a pandas DataFrame of packets.
    """
    cmd = [
        "tshark", "-r", pcap_path,              # read PCAP file
        "-Y", "ip and (tcp or udp)",            # filter for IP packets using TCP/UDP
        "-T", "fields",                         # output only fields
        "-E", "header=y", "-E", "separator=,",  # CSV format
        "-E", "occurrence=f",
        "-e", "ip.src", "-e", "ip.dst",         # source/destination IP
        "-e", "tcp.srcport", "-e", "udp.srcport",  # source ports
        "-e", "tcp.dstport", "-e", "udp.dstport",  # destination ports
        "-e", "ip.proto",                       # protocol number
        "-e", "frame.len",                      # packet length
        "-e", "frame.time_relative",            # timestamp
    ]
    raw = subprocess.check_output(cmd, text=True, errors="ignore")
    with open(out_csv, "w") as f:
        f.write(raw)

    # Load into pandas and rename columns for consistency
    df = pd.read_csv(out_csv)
    cols = {
        'ip.src':'src_ip', 'ip.dst':'dst_ip', 'ip.proto':'proto',
        'frame.len':'len', 'frame.time_relative':'t',
        'tcp.srcport':'tcp_s', 'udp.srcport':'udp_s',
        'tcp.dstport':'tcp_d', 'udp.dstport':'udp_d'
    }
    df = df.rename(columns=cols)

    # Convert numeric columns safely
    for c in ['tcp_s','udp_s','tcp_d','udp_d','proto','len','t']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Coalesce ports: prefer TCP, fallback to UDP
    df['src_port'] = df['tcp_s'].fillna(0).astype('Int64')
    df.loc[df['src_port'] == 0, 'src_port'] = df['udp_s']
    df['dst_port'] = df['tcp_d'].fillna(0).astype('Int64')
    df.loc[df['dst_port'] == 0, 'dst_port'] = df['udp_d']

    # Keep only necessary columns
    keep = ['src_ip','dst_ip','src_port','dst_port','proto','len','t']
    df = df[keep].dropna()
    return df


def build_flow_features(df_packets: pd.DataFrame, out_csv: str) -> pd.DataFrame:
    """
    Aggregate per-packet rows into per-flow features (pkts, bytes, duration, rates).
    """
    if df_packets.empty:
        # Return empty DF with proper schema to avoid errors
        empty = pd.DataFrame(columns=['src_ip','dst_ip','src_port','dst_port','proto',
                                      'pkts','bytes','duration','bps','pps','avg_len'])
        empty.to_csv(out_csv, index=False)
        return empty

    # Ensure numeric columns
    for c in ['src_port','dst_port','proto','len','t']:
        df_packets[c] = pd.to_numeric(df_packets[c], errors='coerce')

    # Group by flow key (5-tuple: src_ip, dst_ip, ports, proto)
    grp_cols = ['src_ip','dst_ip','src_port','dst_port','proto']
    g = df_packets.groupby(grp_cols, dropna=True)

    # Aggregate flow-level features
    agg = g.agg(
        pkts=('len','count'),         # total packets
        bytes=('len','sum'),          # total bytes
        t_min=('t','min'),            # first packet time
        t_max=('t','max')             # last packet time
    ).reset_index()

    # Duration = last - first
    agg['duration'] = (agg['t_max'] - agg['t_min']).fillna(0.0)
    agg.drop(columns=['t_min','t_max'], inplace=True)

    # Avoid divide-by-zero issues
    agg.loc[agg['duration'] <= 0, 'duration'] = 1e-6

    # Calculate flow metrics
    agg['bps']     = (agg['bytes'] * 8.0) / agg['duration']   # bits per second
    agg['pps']     =  agg['pkts'] / agg['duration']           # packets per second
    agg['avg_len'] =  agg['bytes'] / agg['pkts']              # average packet length

    # Save and return
    agg.to_csv(out_csv, index=False)
    return agg


def save_prediction_plots(df_flows: pd.DataFrame, tag: str, out_prefix: str):
    """
    Plot bar + pie charts showing VPN vs NONVPN prediction distribution.
    """
    counts = df_flows['Prediction'].value_counts().reindex(['NONVPN','VPN'], fill_value=0)

    # Bar chart
    plt.figure()
    counts.plot(kind='bar')
    plt.title(f'{tag} Prediction Distribution')
    plt.ylabel('Flow count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    bar_path = os.path.join(FIG_DIR, f'{out_prefix}_predictions_bar.png')
    plt.savefig(bar_path)
    plt.close()
    print(f"[✓] Saved bar plot → {bar_path}")

    # Pie chart
    plt.figure()
    counts.plot(kind='pie', autopct='%1.1f%%')
    plt.title(f'{tag} Prediction Share')
    plt.ylabel('')
    plt.tight_layout()
    pie_path = os.path.join(FIG_DIR, f'{out_prefix}_predictions_pie.png')
    plt.savefig(pie_path)
    plt.close()
    print(f"[✓] Saved pie  plot → {pie_path}")


def predict_on_pcap(pcap_path: str, tag: str, out_prefix: str):
    """
    Full pipeline for one PCAP file:
    1. Extract packets
    2. Build flows
    3. Load trained RF model
    4. Predict VPN / NONVPN
    5. Save CSV + plots
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Train it first.")

    pkt_csv  = os.path.join(FLOW_DIR, f'{out_prefix}_packets.csv')
    flow_csv = os.path.join(FLOW_DIR, f'{out_prefix}_flows.csv')
    pred_csv = os.path.join(FLOW_DIR, f'{out_prefix}_predictions.csv')

    print(f"\n=== PIPELINE → {tag} ===")
    print(f"[+] Extracting packets from {pcap_path}")
    df_packets = extract_packets(pcap_path, pkt_csv)

    print(f"[+] Building flow features → {flow_csv}")
    df_flows = build_flow_features(df_packets, flow_csv)

    if df_flows.empty:
        print("[!] No flows extracted — skipping predictions for this file.")
        return

    # Load trained model
    print(f"[+] Loading model: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    # Ensure features match model’s training order
    for col in FEATURE_ORDER:
        if col not in df_flows.columns:
            df_flows[col] = 0
    X = df_flows[FEATURE_ORDER]

    # Run prediction
    pred = model.predict(X)
    df_flows['Prediction'] = pred
    df_flows.to_csv(pred_csv, index=False)
    print(f"[✓] Predictions saved to {pred_csv}")

    # Quick console preview
    print(df_flows[['pkts','bytes','duration','bps','pps','avg_len',
                    'src_port','dst_port','proto','Prediction']].head())

    # Save bar + pie plots
    save_prediction_plots(df_flows, tag, out_prefix)


# ----------- Main execution -----------
if __name__ == "__main__":
    # Run pipeline on NONVPN and VPN samples
    predict_on_pcap(NONVPN_PCAP, "NONVPN_SAMPLE", "nonvpn_sample")
    predict_on_pcap(VPN_PCAP,    "VPN_SAMPLE",    "vpn_sample")
    print("\n[Done] CSVs in ~/flows/ and plots in ~/figures/")
