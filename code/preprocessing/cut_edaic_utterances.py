import os
import pandas as pd
import csv
from pydub import AudioSegment
AudioSegment.converter = "/scratch/s5944562/HuBERT/datasets/edaic/ffmpeg"

# 資料夾路徑
base_dir = "/scratch/s5944562/HuBERT/datasets/edaic/edaic_audio"
output_dir = "/scratch/s5944562/HuBERT/datasets/edaic/edaic_chunks"
os.makedirs(output_dir, exist_ok=True)

# 讀取 label metadata
label_df = pd.read_csv("metadata_mapped.csv")
label_dict = dict(zip(label_df["Participant_ID"], label_df["PHQ_Binary"]))

# 排除缺漏的 participant
valid_ids = [i for i in range(300, 426) if i not in [342, 394, 398]]

# 儲存 metadata
metadata = []

for pid in valid_ids:
    try:
        transcript_path = f"{base_dir}/{pid}/{pid}_P/{pid}_Transcript.csv"
        audio_path = f"{base_dir}/{pid}/{pid}_P/{pid}_AUDIO.wav"

        if not os.path.exists(transcript_path) or not os.path.exists(audio_path):
            print(f"⚠️ Skipping {pid} - missing transcript or audio")
            continue

        df = pd.read_csv(transcript_path)
        audio = AudioSegment.from_wav(audio_path)
        # df = df[df["Speaker"] == "Participant"] 無 Speaker 欄位，跳過此步
        df = df.dropna(subset=["Start_Time", "End_Time"])  # 保險起見只保留有時間標註的行

        for i, row in df.iterrows():
            start_ms = int(float(row["Start_Time"]) * 1000)
            end_ms = int(float(row["End_Time"]) * 1000)
            if end_ms - start_ms < 200:
                continue

            utt = audio[start_ms:end_ms]
            utt_path = f"{output_dir}/{pid}_{i}.wav"
            utt.export(utt_path, format="wav")

            label = label_dict.get(pid)
            if label is not None:
                metadata.append({"file_path": utt_path, "label": label})

        print(f"✅ Processed {pid}")

    except Exception as e:
        print(f"❌ Error with {pid}: {e}")

# 寫出 utterance_table.csv
with open("utterance_table.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["file_path", "label"])
    writer.writeheader()
    writer.writerows(metadata)
print("✅ Finished writing utterance_table.csv")
print(f"🔍 Total utterances extracted: {len(metadata)}")
print(f"🔍 Total participants processed: {len(set([os.path.basename(row['file_path']).split('_')[0] for row in metadata]))}")