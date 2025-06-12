import os
import torchaudio
from tqdm import tqdm
import pandas as pd

# 參數設定
source_csv = "/scratch/s5944562/HuBERT/cmdc/utterance_table_cmdc_balanced.csv"
output_dir = "/scratch/s5944562/HuBERT/datasets/cmdc_segments"
window_length = 3.0  # seconds
stride = 1.5         # seconds (50% overlap)
target_sr = 16000

# 建立輸出資料夾
os.makedirs(output_dir, exist_ok=True)

# 讀入表格
df = pd.read_csv(source_csv)
segments = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    wav_path = row["file_path"]
    label = row["label"]

    try:
        waveform, sr = torchaudio.load(wav_path)
        if sr != target_sr:
            resample = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resample(waveform)
            sr = target_sr

        duration = waveform.shape[1] / sr
        if duration < window_length:
            continue  # 太短就跳過

        step = int(stride * sr)
        size = int(window_length * sr)
        total_samples = waveform.shape[1]
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        parent_name = os.path.basename(os.path.dirname(wav_path))

        for i, start in enumerate(range(0, total_samples - size + 1, step)):
            end = start + size
            segment = waveform[:, start:end]
            seg_name = f"{parent_name}_{base_name}_seg{i}.wav"
            seg_path = os.path.join(output_dir, seg_name)
            torchaudio.save(seg_path, segment, sr)
            segments.append({"file_path": seg_path, "label": label})

    except Exception as e:
        print(f"❌ Failed on {wav_path}: {e}")

# 儲存新 metadata
seg_df = pd.DataFrame(segments)
seg_df.to_csv("utterance_table_cmdc_segmented.csv", index=False)
print(f"✅ Done! Total segments: {len(seg_df)}")