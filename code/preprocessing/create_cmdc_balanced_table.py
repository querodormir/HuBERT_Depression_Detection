import pandas as pd

# 載入最新的 updated CSV
df = pd.read_csv("utterance_table_cmdc_updated.csv")

# 分開兩類
df_0 = df[df["label"] == 0]  # HC
df_1 = df[df["label"] == 1]  # MDD

# 根據較小類別（HC）決定平衡數量
min_count = min(len(df_0), len(df_1))

# 隨機下採樣兩類至 min_count
df_0_sampled = df_0.sample(n=min_count, random_state=42)
df_1_sampled = df_1.sample(n=min_count, random_state=42)

# 合併 & 打亂
df_balanced = pd.concat([df_0_sampled, df_1_sampled]).sample(frac=1, random_state=42)

# 儲存新 balanced 表
df_balanced.to_csv("utterance_table_cmdc_balanced.csv", index=False)

print(f"✅ Balanced set created: {min_count} samples per class, total {len(df_balanced)}")