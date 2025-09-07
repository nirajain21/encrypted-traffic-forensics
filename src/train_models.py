# Train both Random Forest and XGBoost on ISCX dataset.

# Core libraries
import os, time, json, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Scikit-learn utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# -------- Optional Libraries --------
# XGBoost and SHAP are optional — if not installed, script falls back safely
HAS_XGB = False
HAS_SHAP = False
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    pass
try:
    import shap
    HAS_SHAP = True
    shap.logger.setLevel("ERROR")
except Exception:
    pass

# -------- Paths --------
CSV   = os.path.expanduser('~/flows/iscx_tshark.csv')   # dataset
FIG   = Path(os.path.expanduser('~/figures')); FIG.mkdir(parents=True, exist_ok=True)
MODELS= Path(os.path.expanduser('~/models'));  MODELS.mkdir(parents=True, exist_ok=True)
REPORT= FIG / 'metrics_summary.txt'

# -------- Load & Clean Dataset --------
df = pd.read_csv(CSV)
df = df.replace([np.inf, -np.inf], np.nan).dropna()  # remove invalid rows

# Ensure required columns exist
assert {'label','pkts','bytes','duration','bps','pps','avg_len','src_port','dst_port','proto'}.issubset(df.columns)

# Split into features + labels
y = df['label']  # Target = VPN or NONVPN
X = df[['pkts','bytes','duration','bps','pps','avg_len','src_port','dst_port','proto']].copy()

# Numeric + categorical columns
num_cols = ['pkts','bytes','duration','bps','pps','avg_len','src_port','dst_port']
cat_cols = ['proto']  # categorical: protocol (6=TCP, 17=UDP)

# Preprocessing: numeric passthrough, categorical one-hot encoded
pre = ColumnTransformer([
    ('num', 'passthrough', num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# Train/test split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                      random_state=42, stratify=y)

# -------- Evaluation Helper --------
def evaluate_and_plot(model_name, pipe, use_binary_y=False):
    """Train, evaluate, plot, and save model + metrics."""

    out = {}

    # Convert labels to 0/1 if needed (for XGBoost)
    if use_binary_y:
        ytr_fit = (ytr == 'VPN').astype(int)
        yte_fit = (yte == 'VPN').astype(int)
    else:
        ytr_fit = ytr
        yte_fit = yte

    # Training time
    t0 = time.perf_counter()
    pipe.fit(Xtr, ytr_fit)
    t1 = time.perf_counter()
    out['train_seconds'] = round(t1 - t0, 4)

    # Inference time
    t2 = time.perf_counter()
    pred_fit = pipe.predict(Xte)
    t3 = time.perf_counter()
    out['infer_seconds'] = round(t3 - t2, 4)
    out['infer_ms_per_1000_flows'] = round(1000 * out['infer_seconds'] / max(len(Xte),1), 4)

    # Convert back to string labels if we used 0/1
    if use_binary_y:
        pred = np.where(pred_fit==1, 'VPN', 'NONVPN')
        y_true_labels = yte.values
    else:
        pred = pred_fit
        y_true_labels = yte.values

    # -------- Metrics --------
    acc = accuracy_score(y_true_labels, pred)
    cm  = confusion_matrix(y_true_labels, pred, labels=['NONVPN','VPN'])
    cr  = classification_report(y_true_labels, pred, digits=4)
    out.update(dict(accuracy=acc, confusion_matrix=cm.tolist(), classification_report=cr))

    # ROC-AUC curve
    try:
        y_true_bin = (y_true_labels == 'VPN').astype(int)
        if hasattr(pipe, "predict_proba"):
            if use_binary_y:
                probs = pipe.predict_proba(Xte)[:, 1]
            else:
                probs = pipe.predict_proba(Xte)[:, list(pipe.classes_).index('VPN')]
        else:
            probs = (pred == 'VPN').astype(int)

        auc = roc_auc_score(y_true_bin, probs)
        fpr, tpr, _ = roc_curve(y_true_bin, probs)
        out['roc_auc'] = float(auc)

        # Save ROC plot
        plt.figure()
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
        plt.plot([0,1],[0,1],'--')
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve – {model_name}")
        plt.legend()
        roc_path = FIG / f"roc_{model_name}.png"
        plt.tight_layout(); plt.savefig(roc_path); plt.close()
        out['roc_plot'] = str(roc_path)
    except Exception as e:
        out['roc_error'] = str(e)

    # -------- Feature Importance --------
    # Extract feature names (numeric + encoded categorical)
    ohe = pipe.named_steps['pre'].named_transformers_['cat']
    cat_names = list(ohe.get_feature_names_out(cat_cols))
    feat_names = num_cols + cat_names

    try:
        if HAS_SHAP:
            # SHAP analysis (explains each prediction feature contribution)
            Xtr_tx = pipe.named_steps['pre'].transform(Xtr)
            clf    = pipe.named_steps['clf']
            explainer = shap.TreeExplainer(clf)
            sample_idx = np.random.choice(Xtr_tx.shape[0], size=min(2000, Xtr_tx.shape[0]), replace=False)
            shap_vals = explainer.shap_values(Xtr_tx[sample_idx])

            if isinstance(shap_vals, list):
                shap_abs = np.mean(np.abs(np.vstack([v for v in shap_vals])), axis=0)
            else:
                shap_abs = np.mean(np.abs(shap_vals), axis=0)

            idx = np.argsort(shap_abs)[::-1][:15]
            names = [feat_names[i] for i in idx]; vals = shap_abs[idx]
            plt.figure(); plt.bar(range(len(vals)), vals)
            plt.xticks(range(len(vals)), names, rotation=45, ha='right')
            plt.title(f"SHAP importance – {model_name}")
            plt.tight_layout()
            imp_path = FIG / f"shap_importance_{model_name}.png"
            plt.savefig(imp_path); plt.close()
            out['importance_plot'] = str(imp_path); out['importance_type'] = 'SHAP'
        else:
            raise RuntimeError("SHAP not available")
    except Exception:
        # Fallback: permutation importance
        r = permutation_importance(pipe, Xte, y_true_labels, n_repeats=5, n_jobs=-1, random_state=42)
        idx = np.argsort(r.importances_mean)[::-1][:15]
        names = [feat_names[i] for i in idx]; vals = r.importances_mean[idx]
        plt.figure(); plt.bar(range(len(vals)), vals)
        plt.xticks(range(len(vals)), names, rotation=45, ha='right')
        plt.title(f"Permutation importance – {model_name}")
        plt.tight_layout()
        imp_path = FIG / f"perm_importance_{model_name}.png"
        plt.savefig(imp_path); plt.close()
        out['importance_plot'] = str(imp_path); out['importance_type'] = 'PermutationImportance'

    # -------- Save Model --------
    import joblib
    model_path = MODELS / f"{model_name}.pkl"
    joblib.dump(pipe, model_path)
    out['model_path'] = str(model_path)

    # -------- Print Summary --------
    print(f"\n=== {model_name} ===")
    print(f"Train time: {out['train_seconds']}s | Inference: {out['infer_seconds']}s "
          f"({out['infer_ms_per_1000_flows']} ms / 1000 flows)")
    print(f"Accuracy: {out['accuracy']:.6f}")
    if 'roc_auc' in out: print(f"ROC-AUC: {out['roc_auc']:.6f}")
    print("Confusion matrix (rows=true: [NONVPN, VPN]):")
    print(np.array(out['confusion_matrix']))
    print("Classification report:\n", out['classification_report'])
    print(f"Saved model → {out['model_path']}")
    print(f"Importance ({out['importance_type']}) → {out['importance_plot']}")
    if 'roc_plot' in out: print(f"ROC plot → {out['roc_plot']}")
    return out


# -------- Train Models --------
results = {}

# Random Forest (multi-class string labels: VPN/NONVPN)
rf = Pipeline([
    ('pre', pre),
    ('clf', RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1, class_weight='balanced'
    ))
])
results['RandomForest'] = evaluate_and_plot('rf_iscx_tshark', rf, use_binary_y=False)

# XGBoost (binary 0/1 targets)
if HAS_XGB:
    xgb = Pipeline([
        ('pre', pre),
        ('clf', XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=42, n_jobs=-1, tree_method='hist',
            objective='binary:logistic'
        ))
    ])
    results['XGBoost'] = evaluate_and_plot('xgb_iscx_tshark', xgb, use_binary_y=True)
else:
    print("\n[!] xgboost not available in this Python; skipped XGB model.\n")

# Save all metrics summary
with open(REPORT, 'w') as f:
    f.write(json.dumps(results, indent=2))
print(f"\n[✓] Wrote metrics summary → {REPORT}")
