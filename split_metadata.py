import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", type=str, required=True, help="Path to input metadata CSV")
parser.add_argument("--output_csv", type=str, default=None, help="Path to output CSV with split column")
parser.add_argument("--train_ratio", type=float, default=0.6)
parser.add_argument("--val_ratio", type=float, default=0.2)
args = parser.parse_args()

# 載入 metadata（需包含 file_path 和 label 欄位）
df = pd.read_csv(args.input_csv)

# 第一步：Train vs Val+Test
df_train, df_temp = train_test_split(
    df, stratify=df["label"], test_size=(1 - args.train_ratio), random_state=42
)

# 第二步：Val vs Test（在 temp 中切）
val_size = args.val_ratio / (1 - args.train_ratio)
df_val, df_test = train_test_split(
    df_temp, stratify=df_temp["label"], test_size=(1 - val_size), random_state=42
)

# 加上 split 標記
df_train["split"] = "train"
df_val["split"] = "val"
df_test["split"] = "test"

# 合併 & 打亂順序
df_split = pd.concat([df_train, df_val, df_test]).sample(frac=1, random_state=42)

# 輸出結果
output_path = args.output_csv or args.input_csv.replace(".csv", "_split.csv")
df_split.to_csv(output_path, index=False)
print(f"✅ Saved split metadata to: {output_path}")