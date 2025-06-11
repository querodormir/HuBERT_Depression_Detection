import os
import csv
import torchaudio

# 設定路徑與輸出檔
root_dir = "/scratch/s5944562/HuBERT/datasets/cmdc"
output_csv = "utterance_table_cmdc_updated.csv"
data = []

# 篩選條件（秒）
MIN_SEC = 2
MAX_SEC = 60  # 原為 30，提升至 60 秒以保留更多語料

# 兩類別：HC=0, MDD=1
for group, label in [("HC", 0), ("MDD", 1)]:
    group_dir = os.path.join(root_dir, group)
    for speaker in sorted(os.listdir(group_dir)):
        speaker_path = os.path.join(group_dir, speaker)
        if not os.path.isdir(speaker_path):
            continue
        for file in os.listdir(speaker_path):
            if file.endswith(".wav"):
                wav_path = os.path.join(speaker_path, file)
                try:
                    duration = torchaudio.info(wav_path).num_frames / torchaudio.info(wav_path).sample_rate
                    if MIN_SEC <= duration <= MAX_SEC:
                        data.append({"file_path": wav_path, "label": label})
                except Exception as e:
                    print(f"❌ Failed to read {wav_path}: {e}")

# 輸出為 CSV
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["file_path", "label"])
    writer.writeheader()
    writer.writerows(data)

print(f"✅ Filtered and wrote {len(data)} entries to {output_csv}")
