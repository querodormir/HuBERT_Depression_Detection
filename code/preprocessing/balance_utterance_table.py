import pandas as pd

# 讀入 metadata
df = pd.read_csv("/scratch/s5944562/HuBERT/datasets/edaic/utterance_table.csv")

# 分類
df_0 = df[df["label"] == 0]
df_1 = df[df["label"] == 1]

# 隨機下採樣 0 類，使兩類數量相同（或略多）
df_0_sampled = df_0.sample(n=len(df_1), random_state=42)

# 合併新的 balanced dataframe
df_balanced = pd.concat([df_0_sampled, df_1]).sample(frac=1, random_state=42)  # 打亂順序

# 儲存新的 table
df_balanced.to_csv("utterance_table_balanced.csv", index=False)
