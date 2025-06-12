import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report, confusion_matrix
import os
import random
import torch

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# 設定路徑
feature_dir = "/scratch/s5944562/HuBERT/features_cmdc_layer9"
output_file = "/scratch/s5944562/HuBERT/output/cmdc_l9_results.txt"
os.makedirs("output", exist_ok=True)

# 載入資料
y_train = np.load(os.path.join(feature_dir, "y_train.npy"))
y_val = np.load(os.path.join(feature_dir, "y_val.npy"))
y_test = np.load(os.path.join(feature_dir, "y_test.npy"))

X_train = np.load(os.path.join(feature_dir, "X_train.npy")).reshape(y_train.shape[0], -1)
X_val   = np.load(os.path.join(feature_dir, "X_val.npy")).reshape(y_val.shape[0], -1)
X_test  = np.load(os.path.join(feature_dir, "X_test.npy")).reshape(y_test.shape[0], -1)

# 合併 train + val 作為完整訓練資料
X_all = np.concatenate([X_train, X_val], axis=0)
y_all = np.concatenate([y_train, y_val], axis=0)

# 訓練模型
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_all, y_all)

# 測試評估
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

f1 = f1_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# 輸出結果
with open(output_file, "w") as f:
    f.write("📊 CMD-C Logistic Regression Results\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"ROC-AUC: {auc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm))

print("✅ Evaluation complete. Results saved to", output_file)