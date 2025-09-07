#Focused training for Random Forest only.
import os, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import joblib

csv = os.path.expanduser('~/flows/iscx_tshark.csv')
df  = pd.read_csv(csv, on_bad_lines='skip').replace([np.inf,-np.inf], np.nan).dropna()

y = df['label']
X = df[['frame.len','frame.time_relative','tcp.srcport','tcp.dstport','ip.proto']]

num_cols = ['frame.len','frame.time_relative','tcp.srcport','tcp.dstport']
cat_cols = ['ip.proto']

pre = ColumnTransformer([
    ('num', 'passthrough', num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

clf  = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
pipe = Pipeline([('pre', pre), ('clf', clf)])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
pipe.fit(Xtr, ytr)
pred = pipe.predict(Xte)

print("Accuracy:", accuracy_score(yte, pred))
print("Confusion matrix:\n", confusion_matrix(yte, pred))
print("Classification report:\n", classification_report(yte, pred))

# Feature importance
oh  = pipe.named_steps['pre'].named_transformers_['cat']
cat_names = list(oh.get_feature_names_out(cat_cols))
feat_names = num_cols + cat_names
imps = pipe.named_steps['clf'].feature_importances_
idx  = np.argsort(imps)[::-1][:15]

os.makedirs(os.path.expanduser('~/figures'), exist_ok=True)
plt.figure()
plt.bar(range(len(idx)), imps[idx])
plt.xticks(range(len(idx)), [feat_names[i] for i in idx], rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.expanduser('~/figures/rf_feature_importance.png'))

os.makedirs(os.path.expanduser('~/models'), exist_ok=True)
joblib.dump(pipe, os.path.expanduser('~/models/rf_iscx_tshark.pkl'))
print("[✓] Saved model and plot")
