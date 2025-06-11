import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import random
import torch

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# 🔹 載入資料路徑
feature_dir = "/scratch/s5944562/HuBERT/features_edaic_layer9"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 🔹 載入特徵與標籤
X_train = np.load(os.path.join(feature_dir, "X_train.npy"))
y_train = np.load(os.path.join(feature_dir, "y_train.npy"))
X_val = np.load(os.path.join(feature_dir, "X_val.npy"))
y_val = np.load(os.path.join(feature_dir, "y_val.npy"))
X_test = np.load(os.path.join(feature_dir, "X_test.npy"))
y_test = np.load(os.path.join(feature_dir, "y_test.npy"))

# 🔹 合併 train + val
X_all = np.concatenate([X_train, X_val], axis=0)
y_all = np.concatenate([y_train, y_val], axis=0)

# 🔹 建立並訓練模型
model = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000)
model.fit(X_all, y_all)

# 🔹 預測與評估
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

# 🔹 顯示結果
print("\n📊 EDAIC Logistic Regression Results")
print(f"Accuracy:  {acc:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {auc:.4f}")
print("\nConfusion Matrix:")
print(cm)

# 🔹 儲存文字結果
with open(os.path.join(output_dir, "edaic_l9_results.txt"), "w") as f:
    f.write("📊 EDAIC Logistic Regression Results\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"ROC-AUC: {auc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

# 🔹 繪圖
#plt.figure(figsize=(5,4))
#sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#plt.xlabel("Predicted")
#plt.ylabel("Actual")
#plt.title("Confusion Matrix")
#plt.savefig(os.path.join(output_dir, "edaic_confusion_matrix.png"))

print("\n✅ Results saved to output/edaic_results.txt")